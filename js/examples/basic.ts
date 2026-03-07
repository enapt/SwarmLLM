import { SwarmLLMClient } from "../src";

async function main() {
  // Auto-discovers local swarm at localhost:8800
  const client = new SwarmLLMClient();

  // With custom config:
  // const client = new SwarmLLMClient({
  //   baseUrl: 'http://192.168.1.100:8800',
  //   apiKey: 'sk-...',
  // });

  // List models
  const models = await client.listModels();
  console.log("Available models:", models.map((m) => m.id));

  if (models.length === 0) {
    console.log("No models available. Download shards first.");
    return;
  }

  const model = models[0].id;

  // Non-streaming chat
  const response = await client.chat({
    model,
    messages: [{ role: "user", content: "Hello! What are you?" }],
    max_tokens: 256,
    temperature: 0.7,
  });
  console.log("\nResponse:", response.choices[0].message.content);
  console.log("Usage:", response.usage);

  // Streaming chat
  console.log("\nStreaming:");
  for await (const chunk of client.chatStream({
    model,
    messages: [{ role: "user", content: "Write a haiku about distributed computing." }],
    max_tokens: 128,
  })) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) process.stdout.write(content);
  }
  console.log("\n");

  // Node status
  const status = await client.status();
  console.log("Node:", status.node_id, "| Peers:", status.peers_connected);

  // Admin stats
  const stats = await client.admin.stats();
  console.log("Credits:", stats.credits_balance, "| Tier:", stats.credit_tier);
}

main().catch(console.error);
