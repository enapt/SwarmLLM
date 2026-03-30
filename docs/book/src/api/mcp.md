# MCP Server

SwarmLLM includes a native [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server at `POST /mcp`. This enables AI agents like Claude Code, Cursor, VS Code Copilot, and other MCP-compatible tools to use your SwarmLLM node as a tool provider.

**Protocol version:** 2025-11-05 (JSON-RPC 2.0 over HTTP).

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

Fan out a research question to multiple models in parallel. Designed for knowledge gathering — offload questions to cheap/fast models to get diverse perspectives without using expensive model tokens. If `models` is omitted, auto-selects available models (local first, then cloud).

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

**Tiers:**
- `fast` — lowest-latency local model (default)
- `cheap` — smallest/free model available
- `smart` — most capable model (may use cloud provider)

### `node_info`

Get detailed information about the SwarmLLM node: loaded models, connected peers, credit balance, available cloud providers, and network status.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": { "name": "node_info", "arguments": {} },
  "id": 6
}
```

## Available Resources

### `swarmllm://status`

Returns node status information (version, model loaded, peer count).

```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": { "uri": "swarmllm://status" },
  "id": 7
}
```

## IDE Integration

### Claude Code

**Option A: MCP tools** — access SwarmLLM's tools (research, batch, compare) alongside your normal model:

```bash
claude mcp add --transport http swarmllm http://localhost:8800/mcp \
  --header "Authorization: Bearer YOUR_API_KEY"
```

**Option B: Model backend** — use SwarmLLM as your inference backend (routes all requests through the swarm):

```bash
ANTHROPIC_BASE_URL=http://localhost:8800 ANTHROPIC_AUTH_TOKEN=YOUR_API_KEY \
  claude --model qwen2.5-coder-7b
```

**Option C: Both** — use Claude for reasoning, SwarmLLM MCP for offloading research to cheap models:

```bash
# Add SwarmLLM as MCP server
claude mcp add --transport http swarmllm http://localhost:8800/mcp \
  --header "Authorization: Bearer YOUR_API_KEY"

# Then use Claude normally — it can call research/batch/compare tools via MCP
claude
```

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
