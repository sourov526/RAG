import {OpenAI} from "openai";
import dotenv from "dotenv";

dotenv.config();

const openai = new OpenAI();

async function run() {
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {role: "system", content: "You are a helpful assistant that translates English to Bangla."},
      {role: "user", content: "Translate the following English text to Bangla: 'Hello, how are you?'"},
    ],
  });
  if (response.choices[0]?.message) {
    console.log("Show the response", response.choices[0].message);
  } else {
    console.log("No response message received.");
  }
}

run();