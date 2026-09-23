# SwarmLLM JavaScript/TypeScript SDK

Zero-dependency client for the SwarmLLM decentralized inference network. Works in Node.js 18+.

## Install

Not yet published to npm, so `npm install swarmllm` will not find it. Build it from this repository:

```bash
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM/js && npm install && npm run build && npm pack
# then, in your project:
npm install /path/to/SwarmLLM/js/swarmllm-0.1.0.tgz
```

The build is CommonJS only: use `require('swarmllm')`, or `import` from TypeScript compiled to CommonJS. A native ES-module `import` fails, because `dist/index.mjs` is not built yet.

## Quick Start

```typescript
import { SwarmLLMClient } from 'swarmllm';

const client = new SwarmLLMClient({
  apiKey: process.env.SWARMLLM_API_KEY, // required: Settings → Access Token in the app
}); // talks to http://localhost:8800 by default

// Chat completion
const response = await client.chat({
  model: 'llama-3.2-3b-instruct-q4-k-m', // any id from client.listModels()
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(response.choices[0].message.content);
```

## Streaming

```typescript
for await (const chunk of client.chatStream({
  model: 'llama-3.2-3b-instruct-q4-k-m',
  messages: [{ role: 'user', content: 'Tell me a story' }],
})) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

## Configuration

```typescript
const client = new SwarmLLMClient({
  baseUrl: 'http://192.168.1.100:8800',  // default: http://localhost:8800
  apiKey: 'your-key',                      // required: Settings → Access Token in the app
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

Not supported yet. The build is CommonJS only, and a SwarmLLM node accepts
browser requests only from its own address (`http://localhost:8800`), so a page
served from anywhere else is blocked by the browser. Call the node from Node.js
or from your own server instead.

## License

MIT
