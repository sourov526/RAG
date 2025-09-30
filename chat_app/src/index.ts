const { OpenAI } = require("openai");
const dotenv = require("dotenv");
dotenv.config();

const openai = new OpenAI();


async function run() {
    const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Hello! How can you assist me today?" }
        ],
    });

    console.log(response.choices?.[0]?.message ?? "No message received.");
    
}

run();