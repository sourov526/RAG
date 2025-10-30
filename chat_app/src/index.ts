const { OpenAI } = require("openai");
const dotenv = require("dotenv");
dotenv.config();

const openai = new OpenAI();

type Context = {
  role: "system" | "user" | "assistant";
  content: string;
}[];
const context: Context = [
  { role: "system", content: "You are a helpful assistant." },
];

async function chatCompletion() {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: context,
  });

  const responseMessage =
    response.choices?.[0]?.message?.content ?? "No response";
  context.push({ role: "assistant", content: responseMessage });

  console.log("AI Response:", responseMessage);
}

async function run() {
  const input = require("prompt-sync")({ signit: true }); // takke input from user(terminal)

  while (true) {
    const userInput = input("Please your question : ");
    if (userInput.toLowerCase() === "exit") {
      console.log("Exiting chat...");
      break;
    }

    context.push({ role: "user", content: userInput });
    console.log("show context:", context);
    await chatCompletion();
  }
}

run();
