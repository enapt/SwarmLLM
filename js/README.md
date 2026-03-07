# SwarmLLM JavaScript/TypeScript SDK

Zero-dependency client for the SwarmLLM decentralized inference network. Works in Node.js 18+ and modern browsers.

## Install

```bash
npm install swarmllm
```

## Quick Start

```typescript
import { SwarmLLMClient } from 'swarmllm';

const client = new SwarmLLMClient(); // auto-discovers localhost:8800

// Chat completion
const response = await client.chat({
  model: 'TinyLlama-1.1B',
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(response.choices[0].message.content);
```

## Streaming

```typescript
for await (const chunk of client.chatStream({
  model: 'TinyLlama-1.1B',
  messages: [{ role: 'user', content: 'Tell me a story' }],
})) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

## Configuration

```typescript
const client = new SwarmLLMClient({
  baseUrl: 'http://192.168.1.100:8800',  // default: http://localhost:8800
  apiKey: 'sk-...',                        // optional Bearer token
  timeout: 60_000,                         // default: 120000ms
});
```

## API

### `client.chat(params)` — Chat completion (non-streaming)
### `client.chatStream(params)` — Chat completion (streaming, async iterable)
### `client.listModels()` — List available models
### `client.status()` — Node status
### `client.health()` — Health check
### `client.admin.stats()` — Node statistics
### `client.admin.peers()` — Connected peers

## OpenAI Compatibility

Request/response formats match the OpenAI API, so you can swap `SwarmLLMClient` for the OpenAI SDK with minimal changes. The same `messages`, `model`, `temperature`, `max_tokens`, `tools`, and `stream` parameters are supported.

## Browser Usage

The SDK uses the Fetch API and works in any modern browser:

```html
<script type="module">
  import { SwarmLLMClient } from './dist/index.mjs';
  const client = new SwarmLLMClient({ baseUrl: 'http://localhost:8800' });
  const res = await client.chat({
    model: 'TinyLlama-1.1B',
    messages: [{ role: 'user', content: 'Hi!' }],
  });
  document.body.textContent = res.choices[0].message.content;
</script>
```

## License

MIT
