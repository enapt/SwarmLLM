# MCP Server

SwarmLLM includes a native [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server at `POST /mcp`. This enables AI agents like Claude Code, Cursor, VS Code Copilot, and other MCP-compatible tools to use your SwarmLLM node as a tool provider.

**Protocol version:** 2025-11-25 (JSON-RPC 2.0 over HTTP); 2025-06-18, 2025-03-26 and 2024-11-05 are also accepted.

## Endpoint

```
POST /mcp
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

All requests use JSON-RPC 2.0 format. All tools include [tool annotations](https://modelcontextprotocol.io/specification/2025-11-25) (`readOnlyHint`, `destructiveHint`, etc.).

## Available Tools

### `chat`

Send a message to any model available on the node (local, network, or cloud).
The result's `_meta["com.swarmllm/route"]` names the computers that served it
(as does `delegate`'s; `compare`, `research` and `batch_prompts` give each
model's result its own `route`). A cloud provider's reply carries none.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "chat",
    "arguments": {
      "model": "qwen2.5-coder-7b",
      "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain Rust's ownership model"}
      ],
      "temperature": 0.7,
      "max_tokens": 2048
    }
  },
  "id": 1
}
```

### `models`

List all available models (local + network + cloud).

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": { "name": "models", "arguments": {} },
  "id": 2
}
```

### `compare`

Send the same prompt to multiple models concurrently and get side-by-side results. Up to 10 models per comparison.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "compare",
    "arguments": {
      "prompt": "Write a function to check if a number is prime",
      "models": ["qwen2.5-coder-7b", "gpt-4o", "claude-sonnet-4-20250514"],
      "system": "Write clean, efficient code.",
      "max_tokens": 1024
    }
  },
  "id": 3
}
```

### `research`

Fan out a research question to multiple models in parallel. Designed for knowledge gathering — offload questions to cheap/fast models to get diverse perspectives without using expensive model tokens. If `models` is omitted, it picks up to `max_models` (default 5) models on this node or the swarm. It never picks a paid cloud model by itself — name cloud models in `models` to use them.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "research",
    "arguments": {
      "question": "What are the tradeoffs between ring-allreduce and star topology for tensor parallelism?",
      "models": ["deepseek-chat", "gpt-4o-mini", "qwen2.5-coder-7b"],
      "system": "Be concise and technical.",
      "max_tokens": 2048
    }
  },
  "id": 4
}
```

**Response:**

```json
{
  "question": "What are the tradeoffs...",
  "models_queried": 3,
  "successful_responses": 3,
  "total_tokens_used": 1847,
  "results": [
    {
      "model": "deepseek-chat",
      "response": "Ring-allreduce...",
      "input_tokens": 24,
      "output_tokens": 512,
      "latency_ms": 2100,
      "status": "ok"
    }
  ]
}
```

### `batch_prompts`

Execute multiple independent prompts in parallel, each targeting a specific model. Ideal for offloading parallel subtasks — e.g., ask one model to summarize, another to translate, another to review code, all at once.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "batch_prompts",
    "arguments": {
      "tasks": [
        {
          "id": "summary",
          "model": "gpt-4o-mini",
          "prompt": "Summarize this error log: ...",
          "max_tokens": 512
        },
        {
          "id": "fix",
          "model": "qwen2.5-coder-7b",
          "prompt": "Write a fix for this bug: ...",
          "max_tokens": 1024
        },
        {
          "id": "translate",
          "model": "deepseek-chat",
          "prompt": "Translate to Japanese: ...",
          "max_tokens": 256
        }
      ]
    }
  },
  "id": 5
}
```

**Response:**

```json
{
  "tasks_submitted": 3,
  "tasks_completed": 3,
  "results": [
    {
      "task_id": "summary",
      "model": "gpt-4o-mini",
      "content": "The error log shows...",
      "latency_ms": 890,
      "status": "ok"
    }
  ]
}
```

### `delegate`

Offload a task to the most appropriate model based on a tier preference. Tiers: `fast` picks the lowest-latency local model, `cheap` picks a small/free model, `smart` picks the most capable available model (may use cloud). Saves subscription tokens by routing routine work to local/cheap models.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "delegate",
    "arguments": {
      "prompt": "Summarize this function in one sentence: ...",
      "tier": "fast",
      "max_tokens": 256
    }
  },
  "id": 6
}
```

**Tiers** (`tier` is required):
- `fast` — lowest-latency local model
- `cheap` — smallest/free model available
- `smart` — most capable model (may use cloud provider)

### `node_info`

Get information about the SwarmLLM node: its version, the models it holds and serves, connected peers, request counts, and how many cloud models each provider offers, with a sample of ten. The `models` tool lists every model.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": { "name": "node_info", "arguments": {} },
  "id": 7
}
```

## Available Resources

### `swarmllm://status`

Returns node status information (version, model loaded, peer count).

### `swarmllm://models`

All models available for inference — local, network and cloud providers — with
capabilities and status.

### `swarmllm://peers`

Currently connected computers, with latency, trust, load and shard information.

```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": { "uri": "swarmllm://status" },
  "id": 8
}
```

## IDE Integration

### Claude Code

**MCP tools** — keep Claude as your model and let it hand research, batches and comparisons to the swarm's models:

```bash
claude mcp add --transport http swarmllm http://localhost:8800/mcp \
  --header "Authorization: Bearer YOUR_API_KEY"
```

**Model backend** — run Claude Code on the swarm's models instead: see [Anthropic Messages API § Claude Code Integration](anthropic.md#claude-code-integration).

### VS Code (Copilot Chat)

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "swarmllm": {
      "type": "http",
      "url": "http://localhost:8800/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

Copilot Chat will discover SwarmLLM's tools automatically. Use them by asking Copilot to research, compare models, or batch prompts.

### Cursor / Windsurf / Other MCP Clients

Any MCP-compatible client can connect via HTTP:

```
URL: http://localhost:8800/mcp
Transport: HTTP (Streamable HTTP)
Auth: Bearer token in Authorization header
```

### Continue.dev (OpenAI API)

If your IDE extension supports the OpenAI API format, point it directly at SwarmLLM:

```json
{
  "models": [{
    "title": "SwarmLLM Local",
    "provider": "openai",
    "model": "qwen2.5-coder-7b",
    "apiBase": "http://localhost:8800/v1",
    "apiKey": "YOUR_API_KEY"
  }]
}
```

## Model Compare Dashboard

The compare functionality is also available in the web dashboard via the **Compare** tab. Select 2-10 models, enter a prompt, and view results side-by-side with latency, token counts, and response content.

## Errors

A tool that ran and failed answers the way MCP (2025-11-25) asks: a normal
result with `isError: true` and the reason in its text, so the model driving the
tool reads it and can correct the call, choose another model or tell you. That
covers a model name that does not exist, a prompt longer than the context
window, a missing required field, no models loaded, and a request that timed
out. The text carries the same advice the dashboard shows, and
`_meta["com.swarmllm/error"]` carries the API's error object (`type`, `message`,
`hint`, `hint_key`) for a program that wants to branch on it:

```json
{"jsonrpc": "2.0", "id": 4, "result": {
  "content": [{"type": "text", "text": "Model not available: no-such-model\n\nThis model isn't available yet. …"}],
  "isError": true,
  "_meta": {"com.swarmllm/error": {"type": "not_found_error", "hint_key": "model_not_available", "…": "…"}}
}}
```

Only a call no tool could run is a JSON-RPC error: an unknown tool name
(`-32602`), an unreadable keep-on-this-computer instruction (`-32602`), and a
node whose inference service is not running (`-32603`). Fan-out tools (`compare`, `research`, `batch_prompts`) do not
fail the whole call when one model fails — each result carries its own `status`
and `error`, and the summary counts only the ones that actually answered.
