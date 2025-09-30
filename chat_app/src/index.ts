const { OpenAI } = require("openai");
const dotenv = require("dotenv");
dotenv.config();

const openai = new OpenAI();


async function run() {
    const input = require("prompt-sync")({signit: true}); // takke input from user(terminal)

    while (true) {
        const userInput = input("You: ");
        if (userInput.toLowerCase() === "exit") {
            console.log("Exiting chat...");
            break;
        }

        const response = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: userInput }
            ],
        });

        console.log("AI Response:", response.choices?.[0]?.message?.content ?? "No response");
    }
    
}

run();