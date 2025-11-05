import dotenv from "dotenv";
import { readFileSync, writeFileSync } from "fs";
import { OpenAI } from "openai";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

dotenv.config();

const openai = new OpenAI();

type Fruits = {
  id: string;
  name: string;
  description: string;
};

type FruitWithEmbedding = Fruits & { embedding: number[] };

// 👇 Recreate __dirname in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export function loadFruitJsonFile<T>(fileName: string): T {
  const filePath = join(__dirname, "..", "data", fileName);
  const rawData = readFileSync(filePath, "utf-8");
  return JSON.parse(rawData.toString());
}

function saveFruitEmbeddings(
  fruits: FruitWithEmbedding[],
  fileName: string
): void {
  const filePath = join(__dirname, "..", "data", fileName);
  const serialized = JSON.stringify(fruits, null, 2);
  writeFileSync(filePath, serialized, "utf-8");
  console.log(`Saved ${fruits.length} fruit embeddings to: ${filePath}`);
}

async function generateEmbeddings(
  fruitDescriptions: string[]
): Promise<number[][]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: fruitDescriptions,
  });

  console.log("Generated Embedding response:", response);

  return response.data.map((item) => item.embedding);
}

async function run() {
  const fruitData =
    loadFruitJsonFile<Array<{ Id: string; name: string; description: string }>>(
      "fruits.json"
    );
  const fruits: Fruits[] = fruitData.map((fruit) => ({
    id: fruit.Id,
    name: fruit.name,
    description: fruit.description,
  }));

  const fruitDescriptions = fruits.map((fruit) => fruit.description);
  console.log("Show the fruitDescriptions : ", fruitDescriptions);

  const embeddings = await generateEmbeddings(fruitDescriptions);

  const fruitWithEmbeddings = fruits.map((fruit, index) => ({
    ...fruit,
    embedding: (() => {
      const embedding = embeddings[index];
      if (!embedding) {
        throw new Error(`Missing embedding for fruit: ${fruit.name}`);
      }
      return embedding;
    })(),
  }));

  saveFruitEmbeddings(fruitWithEmbeddings, "fruits_with_embeddings.json");
}

run();
