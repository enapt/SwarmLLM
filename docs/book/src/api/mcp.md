# MCP Server

SwarmLLM includes a native [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server at `POST /mcp`. This enables AI agents like Claude Code, Cursor, and other MCP-compatible tools to use your SwarmLLM node as a tool provider.

## Endpoint

```
POST /mcp
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

All requests use JSON-RPC 2.0 format.

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
      "prompt": "Explain Rust's ownership model",
      "model": "qwen2.5-coder-7b",
      "system": "You are a helpful coding assistant.",
      "temperature": 0.7,
      "max_tokens": 2048
    }
  },
  "id": 1
}
```

### `models`

List all available models (local + cloud).

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "models",
    "arguments": {}
  },
  "id": 2
}
```

### `compare`

Send the same prompt to multiple models concurrently and get side-by-side results. Supports up to 10 models per comparison.

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
      "temperature": 0.7,
      "max_tokens": 1024
    }
  },
  "id": 3
}
```

**Response:**

```json
{
  "prompt": "Write a function to check if a number is prime",
  "models_compared": 3,
  "results": [
    {
      "model": "qwen2.5-coder-7b",
      "content": "def is_prime(n): ...",
      "input_tokens": 18,
      "output_tokens": 95,
      "latency_ms": 1234,
      "status": "ok"
    },
    {
      "model": "gpt-4o",
      "content": "def is_prime(n): ...",
      "input_tokens": 18,
      "output_tokens": 112,
      "latency_ms": 856,
      "status": "ok"
    }
  ]
}
```

## Available Resources

### `swarmllm://status`

Returns node status information (node ID, version, peers, models, credits).

```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "swarmllm://status"
  },
  "id": 4
}
```

## Claude Code MCP Configuration

Add SwarmLLM as an MCP server in your Claude Code settings:

```json
{
  "mcpServers": {
    "swarmllm": {
      "command": "curl",
      "args": ["-s", "-X", "POST", "http://localhost:8800/mcp", "-H", "Content-Type: application/json", "-H", "Authorization: Bearer YOUR_API_KEY"]
    }
  }
}
```

Or use SwarmLLM directly as your model backend (recommended for full Claude Code compatibility):

```bash
ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b
```

## Model Compare Dashboard

The compare functionality is also available in the web dashboard via the **Compare** tab. Select 2-10 models, enter a prompt, and view results side-by-side with latency, token counts, and response content.
