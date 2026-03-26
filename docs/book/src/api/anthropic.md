# Anthropic Messages API

SwarmLLM provides a full Anthropic Messages API at `POST /v1/messages`, enabling it to serve as a drop-in backend for Claude Code and other Anthropic-compatible clients.

## Claude Code Integration

Use SwarmLLM as your Claude Code backend to access all models (local, network, and cloud) through a single endpoint:

```bash
ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b
```

### Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_BASE_URL` | Point to your SwarmLLM node (e.g., `http://localhost:8800`) |
| `ANTHROPIC_AUTH_TOKEN` | Your node's API key (from Settings or `/api/admin/api-key`) |
| `ANTHROPIC_MODEL` | Default model to use |

## POST /v1/messages

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model name (local GGUF, network model, or cloud model like `gpt-4o`) |
| `messages` | array | yes | Chat messages with `role` + `content` |
| `max_tokens` | integer | yes | Maximum tokens to generate (clamped to 1–32768) |
| `system` | string or array | no | System prompt (supports `cache_control` blocks) |
| `stream` | boolean | no | Enable SSE streaming |
| `temperature` | float | no | Sampling temperature |
| `top_p` | float | no | Nucleus sampling |
| `stop_sequences` | array | no | Stop sequences, 1–256 chars each, max 16 |
| `tools` | array | no | Tool definitions for function calling |
| `tool_choice` | object | no | Tool selection strategy |
| `metadata` | object | no | Request metadata |
| `thinking` | object | no | Extended thinking configuration |

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

Requests are routed based on the model name:

| Model Pattern | Route | Details |
|---|---|---|
| Local GGUF model | Local inference | Tool calls and thinking blocks converted to text |
| `claude-*` | Anthropic API | Full pass-through (all fields preserved including tools and thinking) |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI | Anthropic→OpenAI format translation |
| `deepseek-*` | DeepSeek | Anthropic→OpenAI format translation |
| `mistral-*`, `codestral-*`, `pixtral-*` | Mistral | Anthropic→OpenAI format translation |
| `llama-*`, `groq-*` | Groq | Anthropic→OpenAI format translation |
| `nim-*` | NVIDIA NIM | Anthropic→OpenAI format translation |
| `cerebras-*` | Cerebras | Anthropic→OpenAI format translation |
| `samba-*` | SambaNova | Anthropic→OpenAI format translation |
| `fireworks-*`, `accounts/fireworks/*` | Fireworks AI | Anthropic→OpenAI format translation |
| `together-*` | Together AI | Anthropic→OpenAI format translation |
| `deepinfra-*` | DeepInfra | Anthropic→OpenAI format translation |
| `moonshot-*`, `kimi-*` | Moonshot/Kimi | Anthropic→OpenAI format translation |
| Network model | Distributed inference | Routed through swarm P2P network |

All 12 cloud providers are supported. Configure API keys via the dashboard Settings page or by placing a `.env` file in the data directory (`~/.local/share/swarmllm/.env`) with standard variable names (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).

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
