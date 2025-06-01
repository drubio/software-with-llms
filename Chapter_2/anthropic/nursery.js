import Anthropic from "@anthropic-ai/sdk";
import "dotenv/config";

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const response = await anthropic.messages.create({
  model: "claude-3-5-sonnet-20241022",
  max_tokens: 100,
  messages: [{ role: "user", content: "Twinkle, Twinkle, Little" }],
});

console.log(response.content[0].text);
