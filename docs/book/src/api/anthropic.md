# Anthropic Messages API

SwarmLLM provides a full Anthropic Messages API at `POST /v1/messages`, enabling it to serve as a drop-in backend for Claude Code and other Anthropic-compatible clients.

## Claude Code Integration

Use SwarmLLM as your Claude Code backend to access all models (local, network, and cloud) through a single endpoint:

```bash
ANTHROPIC_BASE_URL=http://localhost:8800 ANTHROPIC_AUTH_TOKEN="$SWARMLLM_KEY" \
  claude --model qwen2.5-coder-7b-instruct-q4-k-m
```

### Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_BASE_URL` | Point to your SwarmLLM node (e.g., `http://localhost:8800`) |
| `ANTHROPIC_AUTH_TOKEN` | Your node's API key (from Settings or `/api/admin/api-key`) |
| `ANTHROPIC_MODEL` | Default model to use |

The node accepts its key as `Authorization: Bearer <key>` or as `x-api-key: <key>`, which is what the Anthropic SDKs send.

## POST /v1/messages

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model name (local GGUF, network model, or cloud model like `gpt-4o`) |
| `messages` | array | yes | Chat messages with `role` + `content` |
| `max_tokens` | integer | yes | Maximum tokens to generate; outside 1–32768 it is refused (400), never clamped |
| `system` | string or array | no | System prompt (supports `cache_control` blocks) |
| `stream` | boolean | no | Enable SSE streaming |
| `temperature` | float | no | Sampling temperature |
| `top_p` | float | no | Nucleus sampling |
| `top_k` | integer | no | Top-k sampling (used locally and forwarded to cloud providers) |
| `stop_sequences` | array | no | Stop sequences, 1–256 chars each, max 16 |
| `tools` | array | no | Tool definitions for function calling |
| `tool_choice` | object | no | Tool selection strategy |
| `metadata` | object | no | Forwarded to cloud providers; not used by local and swarm models |
| `thinking` | object | no | Forwarded to Claude / Anthropic only; a local or swarm model does not use it |

### Content Block Types

Messages can contain these content block types:

```json
// Text
{"type": "text", "text": "Hello, world!"}

// Image (base64)
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}

// Tool use (assistant response)
{"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "NYC"}}

// Tool result (user message)
{"type": "tool_result", "tool_use_id": "toolu_123", "content": "72F, sunny"}

// Thinking (extended thinking)
{"type": "thinking", "thinking": "Let me reason about this..."}

// Redacted thinking
{"type": "redacted_thinking", "data": "..."}
```

### Response

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "model": "qwen2.5-coder-7b",
  "content": [
    {"type": "text", "text": "Here's my response..."}
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 150
  }
}
```

## Model Routing

A model this node or the swarm holds is always used first. Only a name that is
neither goes to a cloud provider, chosen by its name:

| Model Pattern | Route | Details |
|---|---|---|
| A model on this node or in the swarm | Local or swarm inference | Tool calls the model makes come back as `tool_use` blocks; earlier `tool_use`, `tool_result` and `thinking` blocks in the conversation are given to the model as text |
| `claude-*` | Anthropic API | Full pass-through (all fields preserved including tools and thinking) |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `o1`, `o3`, `o4` | OpenAI | Anthropic→OpenAI format translation |
| `deepseek*` | DeepSeek | Anthropic→OpenAI format translation |
| `mistral*`, `magistral*`, `ministral*`, `codestral*`, `pixtral*` | Mistral | Anthropic→OpenAI format translation |
| `nvidia/*`, `nim/*`, a name containing `nemotron`; with NIM set up, any `org/model` name | NVIDIA NIM | Anthropic→OpenAI format translation |
| `llama-*`, `gemma*` (only when Groq is set up) | Groq | Anthropic→OpenAI format translation |
| `moonshot-*`, `kimi*`, `k2*` | Moonshot/Kimi | Anthropic→OpenAI format translation |
| `accounts/fireworks/*` | Fireworks AI | Anthropic→OpenAI format translation |
| `provider:model`, e.g. `cerebras:llama3.1-8b` | That provider | The only way to reach Cerebras, SambaNova, Together, DeepInfra and custom providers; it also overrides the name rules above |

All 12 cloud providers are supported. Configure API keys in the dashboard (the gear icon, Settings → **Cloud Providers**) or by placing a `.env` file in the data directory (`~/.local/share/swarmllm/.env`) with standard variable names (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).

## System Blocks with Cache Control

Anthropic-compatible prompt caching:

```json
{
  "system": [
    {"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}
  ]
}
```

## Streaming (SSE)

When `stream: true`, responses arrive as Server-Sent Events following the Anthropic streaming format:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"qwen2.5-coder-7b","content":[]}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: message_stop
data: {"type":"message_stop"}
```

### Errors during a stream

If the request fails after the stream has opened — the model is unavailable, the
node refuses by policy, the prompt is too long — the failure arrives as an
`error` event, in the same shape the Anthropic API uses:

```
event: error
data: {"type":"error","error":{"type":"invalid_request_error","message":"This conversation is too long for …"}}
```

The `error` event is **terminal**: the stream ends there, with no
`message_delta` or `message_stop` after it. The `error.type` is one Anthropic
defines (`invalid_request_error`, `not_found_error`, `authentication_error`,
`permission_error`, `rate_limit_error`, `request_too_large`, `overloaded_error`,
or `api_error`), so a client matching on it behaves the same as against the real
API.

A failure is never delivered as assistant text, and `stop_reason` never carries
a value the API does not define — so a reply your client receives as content is
always something the model actually produced.
