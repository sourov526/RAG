import {
  ChromaClient,
  IncludeEnum,
  type ChromaClientArgs,
  type Collection,
  type CollectionMetadata,
  type EmbeddingFunction,
  type GetResult,
  type Metadata,
  type QueryResult,
  type Where,
  type WhereDocument,
} from "chromadb";
import { config as loadEnv } from "dotenv";
import { URL as NodeURL, pathToFileURL } from "node:url";
import OpenAI from "openai";

loadEnv();

const DEFAULT_CHROMA_URL = "http://localhost:8000";
const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_OPENAI_BATCH_SIZE = 96;
const DEMO_COLLECTION_NAME = "chromadb-demo-fruits";

const DEMO_RECORDS: VectorRecord[] = [
  {
    id: "apple",
    document:
      "Apples are crisp red fruits that balance sweetness with light tartness.",
    metadata: { color: "red", family: "pome" },
  },
  {
    id: "banana",
    document:
      "Bananas are bright yellow fruits packed with potassium for quick energy.",
    metadata: { color: "yellow", family: "berry" },
  },
  {
    id: "lime",
    document: "Limes are tart green citrus fruits loaded with vitamin C.",
    metadata: { color: "green", family: "citrus" },
  },
];

export type VectorMetadata = Metadata;

export interface ChromaVectorStoreOptions {
  apiUrl?: string;
  host?: string;
  port?: number;
  ssl?: boolean;
  tenant?: string;
  database?: string;
  defaultCollection?: string;
  embeddingFunction?: EmbeddingFunction;
  openAIApiKey?: string;
  openAIModel?: string;
  openAIBatchSize?: number;
  simpleEmbeddingDimensions?: number;
}

export interface CreateCollectionOptions {
  name: string;
  metadata?: CollectionMetadata;
}

export interface VectorRecord {
  id: string;
  document?: string;
  embedding?: number[];
  metadata?: VectorMetadata;
}

export interface UpsertEmbeddingsOptions {
  collectionName?: string;
  records: VectorRecord[];
}

export interface GetEmbeddingsOptions {
  collectionName?: string;
  ids?: string[];
  where?: Where;
  whereDocument?: WhereDocument;
  limit?: number;
  includeDocuments?: boolean;
  includeMetadata?: boolean;
}

export interface SimilaritySearchOptions {
  collectionName?: string;
  queryTexts?: string[];
  queryEmbeddings?: number[][];
  topK?: number;
  where?: Where;
  whereDocument?: WhereDocument;
  includeDocuments?: boolean;
  includeMetadata?: boolean;
  includeEmbeddings?: boolean;
}

type QueryResponse<TMeta extends VectorMetadata> = QueryResult<TMeta>;
type RetrievalResponse<TMeta extends VectorMetadata> = GetResult<TMeta>;

type OpenAIEmbeddingOptions = {
  apiKey: string;
  model?: string;
  batchSize?: number;
};

const createOpenAITextEmbeddingFunction = (
  options: OpenAIEmbeddingOptions
): EmbeddingFunction => {
  // Create shared OpenAI client + configuration for all future requests.
  const client = new OpenAI({ apiKey: options.apiKey });
  const model = options.model ?? DEFAULT_EMBEDDING_MODEL;
  const batchSize = options.batchSize ?? DEFAULT_OPENAI_BATCH_SIZE;

  return {
    name: `openai::${model}`,
    async generate(texts: string[]): Promise<number[][]> {
      // Short‑circuit when there is nothing to embed.
      if (texts.length === 0) {
        return [];
      }

      // Split the workload into batches so each API call stays within bounds.
      const batches = chunkArray(texts, batchSize);
      const embeddings: number[][] = [];

      for (const batch of batches) {
        // Request a vector for every string in the current batch.
        const response = await client.embeddings.create({
          model,
          input: batch,
        });

        // Push copies of the vectors so downstream consumers can mutate safely.
        response.data.forEach((row) => embeddings.push([...row.embedding]));
      }

      return embeddings;
    },
  };
};

const createSimpleEmbeddingFunction = (dimensions = 64): EmbeddingFunction => {
  // Keep the fallback vector size configurable for quick experimentation.
  const safeDimensions = dimensions;

  const toVector = (text: string): number[] => {
    // Initialize an empty vector that we will gradually populate.
    const vector = new Array(safeDimensions).fill(0);

    for (let index = 0; index < text.length; index += 1) {
      // Spread each character code across dimensions to capture order + content.
      const charCode = text.charCodeAt(index);
      const bucket = index % safeDimensions;
      vector[bucket] += (charCode % 32) / 100;
    }

    // Normalize so cosine similarity remains meaningful.
    return normalize(vector);
  };

  return {
    name: "simple::hash",
    async generate(texts: string[]): Promise<number[][]> {
      // Produce a deterministic hash-based vector for each text input.
      return texts.map((text) => toVector(text));
    },
  };
};

export interface ChromaVectorStoreAPI {
  createCollection(options: CreateCollectionOptions): Promise<Collection>;
  listCollections(limit?: number, offset?: number): Promise<Collection[]>;
  deleteCollection(name: string): Promise<void>;
  upsertEmbeddings(options: UpsertEmbeddingsOptions): Promise<void>;
  getEmbeddings<TMeta extends VectorMetadata = VectorMetadata>(
    options?: GetEmbeddingsOptions
  ): Promise<RetrievalResponse<TMeta>>;
  similaritySearch<TMeta extends VectorMetadata = VectorMetadata>(
    options: SimilaritySearchOptions
  ): Promise<QueryResponse<TMeta>>;
}

export const createChromaVectorStore = (
  options: ChromaVectorStoreOptions = {}
): ChromaVectorStoreAPI => {
  // Resolve connection details from options or environment variables.
  const clientArgs = buildClientArgs(options);
  // Instantiate the Chroma client so we can interact with the server.
  const client = new ChromaClient(clientArgs);
  // Remember the default collection so callers can omit the name later.
  const defaultCollection =
    options.defaultCollection ?? process.env.CHROMA_DEFAULT_COLLECTION;
  // Pick the embedding function that collections will use for vectors.
  const embedder =
    options.embeddingFunction ?? createEmbeddingFunction(options);
  const registerEmbedderWithServer = Boolean(options.embeddingFunction);

  const resolveCollection = async (
    name?: string,
    metadata?: CollectionMetadata
  ): Promise<Collection> => {
    // Resolve the final name (default or explicit) before touching the server.
    const collectionName = name ?? defaultCollection;
    if (!collectionName) {
      throw new Error(
        "A collection name is required. Pass one explicitly or set CHROMA_DEFAULT_COLLECTION."
      );
    }

    // Ensure the collection exists and is wired up with the correct embedder.
    const payload: Parameters<ChromaClient["getOrCreateCollection"]>[0] = {
      name: collectionName,
      embeddingFunction: registerEmbedderWithServer ? embedder : null,
    };
    if (metadata) {
      payload.metadata = metadata;
    }

    return client.getOrCreateCollection(payload);
  };

  const createCollection = async (options: CreateCollectionOptions) => {
    // Delegate to Chroma so the collection is created (or returned if it exists).
    return resolveCollection(options.name, options.metadata);
  };

  const listCollections = async (limit = 50, offset = 0) => {
    // Ask the server for the current collections so callers can inspect them.
    return client.listCollections({ limit, offset });
  };

  const deleteCollection = async (name: string) => {
    // Remove the entire collection (and its vectors) from the server.
    await client.deleteCollection({ name });
  };

  const upsertEmbeddings = async (options: UpsertEmbeddingsOptions) => {
    // Bail early when nothing needs to be written.
    if (!options.records.length) {
      return;
    }

    // Make sure every record contains data that can become a vector.
    options.records.forEach((record) => {
      if (!record.document && !record.embedding) {
        throw new Error(
          `Record "${record.id}" must provide a document or a pre-computed embedding.`
        );
      }
    });

    // Request a collection instance before we manipulate its contents.
    const collection = await resolveCollection(options.collectionName);

    // Identify the records that still need embeddings generated.
    const pending = options.records.filter((record) => !record.embedding);
    if (pending.length) {
      // Gather the source documents that can be converted into vectors.
      const documents = pending.map((record) => {
        if (!record.document) {
          throw new Error(
            `Record "${record.id}" must include a document when no embedding is supplied.`
          );
        }
        return record.document;
      });
      // Use the configured embedding function to vectorize the text.
      const generated = await embedder.generate(documents);
      // Reattach the generated vectors to their respective records.
      pending.forEach((record, index) => {
        record.embedding = generated[index]!;
      });
    }

    // Only pass documents through when every record includes one.
    const documents = options.records.every(
      (record) => typeof record.document === "string"
    )
      ? options.records.map((record) => record.document as string)
      : undefined;

    // Each record needs metadata even if it is just an empty object.
    const metadatas: VectorMetadata[] = options.records.map(
      (record) => (record.metadata ?? {}) as VectorMetadata
    );

    // Persist the vectors so they become available for future searches.
    const payload: Parameters<Collection["upsert"]>[0] = {
      ids: options.records.map((record) => record.id),
      embeddings: options.records.map((record) => record.embedding as number[]),
      metadatas,
    };
    if (documents) {
      payload.documents = documents;
    }

    await collection.upsert(payload);
  };

  const getEmbeddings = async <TMeta extends VectorMetadata = VectorMetadata>(
    options: GetEmbeddingsOptions = {}
  ): Promise<RetrievalResponse<TMeta>> => {
    // Access the target collection, falling back to the default when needed.
    const collection = await resolveCollection(options.collectionName);

    // Decide which payload fields should be returned to the caller.
    const include: IncludeEnum[] = [IncludeEnum.embeddings];
    if (options.includeDocuments ?? true) {
      include.push(IncludeEnum.documents);
    }
    if (options.includeMetadata ?? true) {
      include.push(IncludeEnum.metadatas);
    }

    const args: Parameters<Collection["get"]>[0] = { include };
    if (options.ids) {
      args.ids = options.ids;
    }
    if (options.where) {
      args.where = options.where;
    }
    if (options.whereDocument) {
      args.whereDocument = options.whereDocument;
    }
    if (options.limit !== undefined) {
      args.limit = options.limit;
    }

    // Fetch the requested vectors using the supplied filters.
    return collection.get<TMeta>(args);
  };

  const similaritySearch = async <
    TMeta extends VectorMetadata = VectorMetadata
  >(
    options: SimilaritySearchOptions
  ): Promise<QueryResponse<TMeta>> => {
    // Ensure callers pass either raw text or embeddings to query against.
    if (!options.queryEmbeddings?.length && !options.queryTexts?.length) {
      throw new Error("Provide queryTexts or queryEmbeddings to run a search.");
    }

    // Prepare the include list so results contain the desired context.
    const include: IncludeEnum[] = [IncludeEnum.distances];
    if (options.includeDocuments ?? true) {
      include.push(IncludeEnum.documents);
    }
    if (options.includeMetadata ?? true) {
      include.push(IncludeEnum.metadatas);
    }
    if (options.includeEmbeddings) {
      include.push(IncludeEnum.embeddings);
    }

    // Work with the provided collection (or the default if omitted).
    const collection = await resolveCollection(options.collectionName);

    const args: Parameters<Collection["query"]>[0] = {
      nResults: options.topK ?? 5,
      include,
    };

    if (options.queryEmbeddings?.length) {
      args.queryEmbeddings = options.queryEmbeddings;
    } else if (options.queryTexts?.length) {
      if (registerEmbedderWithServer) {
        args.queryTexts = options.queryTexts;
      } else {
        const embeddedQueries = await embedder.generate(options.queryTexts);
        args.queryEmbeddings = embeddedQueries;
      }
    }

    if (options.where) {
      args.where = options.where;
    }
    if (options.whereDocument) {
      args.whereDocument = options.whereDocument;
    }

    // Run the actual similarity query against ChromaDB.
    return collection.query<TMeta>(args);
  };

  return {
    createCollection,
    listCollections,
    deleteCollection,
    upsertEmbeddings,
    getEmbeddings,
    similaritySearch,
  };
};

const buildClientArgs = (
  options: ChromaVectorStoreOptions
): Partial<ChromaClientArgs> => {
  // Accept either an explicit API URL or discrete host/port settings.
  const clientArgs: Partial<ChromaClientArgs> = {};
  const explicitUrl = options.apiUrl ?? process.env.CHROMA_API_URL;

  if (explicitUrl) {
    // Parse the provided URL so we can populate host, port, and ssl flags.
    const parsed = new NodeURL(explicitUrl);
    clientArgs.host = parsed.hostname;
    if (parsed.port) {
      clientArgs.port = Number(parsed.port);
    }
    clientArgs.ssl = parsed.protocol === "https:";
    return clientArgs;
  }

  const host = options.host ?? process.env.CHROMA_HOST;
  const port = options.port ?? parseNumber(process.env.CHROMA_PORT);
  const sslFlag = options.ssl ?? parseBoolean(process.env.CHROMA_SSL);
  const tenant = options.tenant ?? process.env.CHROMA_TENANT;
  const database = options.database ?? process.env.CHROMA_DATABASE;

  if (host) {
    clientArgs.host = host;
  }
  if (port) {
    clientArgs.port = port;
  }
  if (sslFlag !== undefined) {
    clientArgs.ssl = sslFlag;
  }
  if (tenant) {
    clientArgs.tenant = tenant;
  }
  if (database) {
    clientArgs.database = database;
  }

  // Fall back to the local server when nothing else is configured.
  if (!Object.keys(clientArgs).length) {
    clientArgs.path = DEFAULT_CHROMA_URL;
  }

  return clientArgs;
};

const createEmbeddingFunction = (
  options: ChromaVectorStoreOptions
): EmbeddingFunction => {
  // Prefer OpenAI when a key is available so vectors stay high quality.
  const apiKey = options.openAIApiKey ?? process.env.OPENAI_API_KEY;
  if (apiKey) {
    const embedderOptions: OpenAIEmbeddingOptions = { apiKey };
    const model = options.openAIModel ?? process.env.OPENAI_EMBEDDING_MODEL;
    if (model) {
      embedderOptions.model = model;
    }
    if (options.openAIBatchSize !== undefined) {
      embedderOptions.batchSize = options.openAIBatchSize;
    }
    return createOpenAITextEmbeddingFunction(embedderOptions);
  }

  // Otherwise rely on the lightweight deterministic hash embedding.
  return createSimpleEmbeddingFunction(options.simpleEmbeddingDimensions);
};

const chunkArray = <T>(items: T[], size: number): T[][] => {
  // Convert the flat list into evenly sized chunks for batched processing.
  const chunks: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
};

const normalize = (vector: number[]): number[] => {
  // Calculate the vector magnitude so we can scale each component.
  const magnitude = Math.sqrt(
    vector.reduce((sum, value) => sum + value * value, 0)
  );
  if (magnitude === 0) {
    return vector;
  }
  // Divide each entry by the magnitude to keep the vector unit length.
  return vector.map((value) => value / magnitude);
};

const parseNumber = (value?: string): number | undefined => {
  // Skip parsing when the string is empty or undefined.
  if (!value) {
    return undefined;
  }
  const numeric = Number(value);
  return Number.isNaN(numeric) ? undefined : numeric;
};

const parseBoolean = (value?: string): boolean | undefined => {
  // Respect explicit truthy/falsey strings and ignore everything else.
  if (value === undefined) {
    return undefined;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "1") {
    return true;
  }
  if (normalized === "false" || normalized === "0") {
    return false;
  }
  return undefined;
};

export const chromaVectorStore = createChromaVectorStore();

const DEMO_QUESTION = "Which fruit has vitamin C?";

const runDemoCli = async (): Promise<void> => {
  const question = DEMO_QUESTION;

  console.log("Running ChromaDB demo with question:\n>", question, "\n");

  const store = createChromaVectorStore({
    defaultCollection: DEMO_COLLECTION_NAME,
  });

  await store.createCollection({
    name: DEMO_COLLECTION_NAME,
    metadata: { purpose: "demo" },
  });

  await store.upsertEmbeddings({
    collectionName: DEMO_COLLECTION_NAME,
    records: DEMO_RECORDS,
  });

  const results = await store.similaritySearch({
    collectionName: DEMO_COLLECTION_NAME,
    queryTexts: [question],
    topK: 2,
    includeEmbeddings: false,
  });

  const matches = results.ids?.[0] ?? [];
  const documents = results.documents?.[0] ?? [];
  const distances = results.distances?.[0] ?? [];

  if (!matches.length) {
    console.log("No matches found. Try asking a broader question.");
    return;
  }

  matches.forEach((matchId, index) => {
    console.log(`#${index + 1} • ${matchId}`);
    console.log(`Distance: ${distances[index]?.toFixed(4) ?? "n/a"}`);
    console.log(`Document: ${documents[index]}`);
    console.log("");
  });

  console.log("Answer:", documents[0]);
};

const maybeRunCliDemo = (): void => {
  const entryHref = process.argv[1]
    ? pathToFileURL(process.argv[1]).href
    : undefined;

  if (entryHref && import.meta.url === entryHref) {
    runDemoCli().catch((error) => {
      console.error("Demo failed:", error);
      process.exitCode = 1;
    });
  }
};

maybeRunCliDemo();
