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
| `model` | string | yes | — | Model name (or `"auto"` for first available) |
| `messages` | array | yes | — | Chat messages (`role` + `content`). Roles: `system`, `user`, `assistant`, `tool` |
| `stream` | boolean | no | `false` | Enable SSE streaming |
| `max_tokens` | integer | no | `2048` | Max tokens to generate (clamped to 1–32768) |
| `temperature` | float | no | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | float | no | `1.0` | Nucleus sampling threshold |
| `stop` | string or array | no | — | Stop sequence(s), 1–256 chars each, max 16 |
| `frequency_penalty` | float | no | `0.0` | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | no | `0.0` | Presence penalty (-2.0 to 2.0) |
| `tools` | array | no | — | Tool/function definitions for function calling |
| `tool_choice` | string or object | no | — | `"none"`, `"auto"`, `"required"`, or `{"type":"function","function":{"name":"..."}}`. `"none"` is honoured for a local model by not describing its tools at all — the only place it can be enforced. Any other value leaves them available: a local model cannot be compelled to call one |
| `logprobs` | boolean | no | `false` | Log probabilities per output token. **Cloud models only** — a request for these against a model running locally is refused with a 400 explaining why, rather than answered without them |
| `top_logprobs` | integer | no | — | Number of top log probabilities per token (0-20, requires `logprobs: true`). Cloud models only, as above |
| `session_id` | string | no | — | Reuse KV-cache from a previous request |
| `lora_adapter` | string | no | — | LoRA adapter ID for fine-tuned inference |

### Response (non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "qwen2.5-coder-7b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Rust is a systems programming language..."},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  }
}
```

### Response with logprobs

When `logprobs: true` and `top_logprobs: 3`:

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "Hello"},
    "finish_reason": "stop",
    "logprobs": {
      "content": [{
        "token": "Hello",
        "logprob": -0.234,
        "bytes": null,
        "top_logprobs": [
          {"token": "Hello", "logprob": -0.234, "bytes": null},
          {"token": "Hi", "logprob": -1.456, "bytes": null},
          {"token": "Hey", "logprob": -2.012, "bytes": null}
        ]
      }]
    }
  }]
}
```

### Response with tool_calls

When the model calls a tool, `finish_reason` is `"tool_calls"` and `content` is null:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"NYC\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
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

# Basic streaming
response = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Python — Function calling

```python
response = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)

if response.choices[0].finish_reason == "tool_calls":
    for tc in response.choices[0].message.tool_calls:
        print(f"Call {tc.function.name}({tc.function.arguments})")
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

## POST /v1/embeddings

Returns `503 Service Unavailable`. Text embeddings are not supported via the subprocess inference path. Use a dedicated embedding provider or the OpenAI embeddings API directly.

## GET /v1/providers

List configured cloud providers and their available models.

```bash
curl http://localhost:8800/v1/providers \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Returns an array of `{ name, models: [...] }` objects for each configured provider.
