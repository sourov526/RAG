import dotenv from "dotenv";
import { readFileSync } from "fs";
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

// 👇 Recreate __dirname in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export function loadFruitJsonFile<T>(fileName: string): T {
  const filePath = join(__dirname, "..", "data", fileName);
  const rawData = readFileSync(filePath, "utf-8");
  return JSON.parse(rawData.toString());
}

async function generateEmbedding(fruitDescription: string) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: "The quick brown fox jumps over the lazy dog",
  });
  console.log("Generated Embedding response:", response);
}

async function run() {
  const fruits: Fruits[] = loadFruitJsonFile("fruits.json");
  const fruitDescriptions = fruits.map((fruit) => fruit.description);
  console.log("Show the fruitDescriptions : ", fruitDescriptions);
}

run();
