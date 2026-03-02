# OpenAI-Compatible API

SwarmLLM provides a drop-in replacement for the OpenAI API. All endpoints require Bearer token authentication.

## POST /v1/chat/completions

Chat completions with streaming support.

```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Rust?"}
    ],
    "stream": true,
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Request Body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | yes | — | Model name |
| `messages` | array | yes | — | Chat messages (`role` + `content`) |
| `stream` | boolean | no | `false` | Enable SSE streaming |
| `max_tokens` | integer | no | `2048` | Max tokens to generate |
| `temperature` | float | no | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | float | no | `1.0` | Nucleus sampling threshold |
| `session_id` | string | no | — | Reuse KV-cache from a previous request |

### Response (non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "qwen2.5-coder-7b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Rust is a systems programming language..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  }
}
```

### Streaming (SSE)

When `stream: true`, responses arrive as Server-Sent Events:

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Rust"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":" is"},"index":0}]}

data: [DONE]
```

## GET /v1/models

List available models.

```bash
curl http://localhost:8800/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-coder-7b",
      "object": "model",
      "owned_by": "swarmllm"
    }
  ]
}
```

## GET /v1/status

Node status (SwarmLLM extension).

```bash
curl http://localhost:8800/v1/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Using with OpenAI Client Libraries

### Python (openai)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript (openai)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8800/v1",
  apiKey: "YOUR_API_KEY",
});

const stream = await client.chat.completions.create({
  model: "qwen2.5-coder-7b",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

### curl (streaming)

```bash
curl -N http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```
