# ChromaDB Demo Setup & Runbook

This package ships a demo script (`src/index.ts`) that seeds a tiny fruit knowledge base and answers a hard-coded question using ChromaDB similarity search. Follow the steps below to get it running.

## 1. Prerequisites

- **Node.js 18+** installed locally.
- **Docker** (or another way to host ChromaDB).  
  Start ChromaDB locally with:
  ```bash
  docker run -d --name chroma-local -p 8000:8000 chromadb/chroma
  ```
- **OpenAI account & API key (optional)** if you want high-quality embeddings. Sign up at https://platform.openai.com/signup and create a key under “API keys”.

## 2. Install Dependencies

```bash
cd chromadb
npm install
```

## 3. Configure Environment (optional when using defaults)

Create `chromadb/.env` to override defaults if needed:

```
CHROMA_API_URL=http://localhost:8000
OPENAI_API_KEY=sk-your-key    # only if you plan to use OpenAI embeddings
```

If no OpenAI key is provided, the script falls back to a built-in hash-based embedder.

## 4. Run the Demo Script

Node 20 requires registering the loader before running TypeScript files. Use this exact command inside the `chromadb` folder:

```bash
node --import 'data:text/javascript,import { register } from "node:module"; import { pathToFileURL } from "node:url"; register("ts-node/esm", pathToFileURL("./"));' src/index.ts
```

What happens:
- Seeds a demo collection (`chromadb-demo-fruits`)
- Upserts a few fruit descriptions
- Runs the hard-coded question “What fruit is rich in potassium?”
- Prints the ranked matches and the top answer to the terminal

## 5. Verifying the Chroma Server

If the script fails to connect, confirm the server is reachable:

```bash
curl http://localhost:8000/api/v1/collections
```

You should see JSON (even if empty). If not, restart the Docker container and rerun the command above.
