# Claude Subscription Provider

Use your existing Claude Pro, Max, Team, or Enterprise subscription to access Claude models through SwarmLLM — no API key or per-token charges needed.

> **Feature-gated**: Build with `--features claude-subscription` to enable. This feature is isolated behind a compile-time flag for easy removal.

## How It Works

When enabled, SwarmLLM spawns the `claude` CLI as a subprocess for each Claude model request:

```
Client Request (OpenAI or Anthropic format)
  → SwarmLLM API (openai.rs / anthropic/mod.rs)
    → Provider resolution: model starts with "claude-"
      → Claude subscription enabled? → Spawn subprocess
      → Else: use Anthropic API key (existing behavior)
    → claude -p --output-format stream-json --model <model> "<prompt>"
    → Parse NDJSON → Translate to API format → Return response
```

Both the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints are supported, with streaming and non-streaming modes.

## Setup

### 1. Install the Claude CLI

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Log in with your subscription

```bash
claude login
```

This opens a browser window. Sign in with your Claude Pro/Max/Team/Enterprise account.

### 3. Build SwarmLLM with the feature

```bash
cargo build --no-default-features --features dev,claude-subscription
```

### 4. Enable via the dashboard

Open Settings → Cloud Providers → Claude Subscription, click "Check Status" to verify your CLI is detected, then enable the toggle.

Or via API:

```bash
curl -X PUT http://localhost:8800/api/admin/providers \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"claude_subscription_enabled": true}'
```

### 5. Send requests

```bash
# OpenAI format
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Anthropic format
curl http://localhost:8800/v1/messages \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Multi-Turn Conversations

Multi-turn conversations work by serializing the full message history into the prompt on each request. The format uses XML tags that Claude understands natively:

- System messages → `<system>...</system>`
- Assistant messages → `<previous_response>...</previous_response>`
- User messages → bare text

This is the same stateless approach used by OpenAI-compatible APIs — the client sends the full conversation every time, and the server doesn't maintain session state.

## Configuration

All configuration is in the `providers.claude_subscription` section, manageable via the admin API or dashboard:

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Route Claude requests through the CLI |
| `claude_binary` | `"claude"` | Path to the `claude` binary |
| `default_model` | *(from request)* | Override model for all requests |
| `max_concurrent` | `3` | Max concurrent subprocess invocations |
| `timeout_secs` | `300` | Timeout per request (seconds) |
| `working_dir` | *(system temp)* | Working directory for the subprocess |

### Working Directory

By default, the subprocess runs in the system temp directory to avoid loading project-specific CLAUDE.md files, hooks, and MCP servers. Set `working_dir` to a project path if you want Claude to have project context for its responses.

## Routing Priority

When a `claude-*` model is requested:

1. **Claude subscription** (if enabled and CLI detected) — subprocess path, uses subscription
2. **Anthropic API key** (if configured) — direct API proxy, pay-per-token
3. **Error** — no provider available

The subscription provider takes priority over the API key. Disable the subscription toggle to fall back to API key billing.

## Rate Limits

Subscription rate limits are per rolling 5-hour window (not per-second RPM like API keys). The concurrency limiter (default 3) prevents spawning too many concurrent processes. Community reports suggest ~3-5 parallel Opus sessions before degradation.

Rate limit info is returned in the NDJSON output and logged. The `GET /api/admin/claude-subscription/status` endpoint shows the current rate limit tier.

## Removal

If this feature needs to be removed:

```bash
git rm src/api/claude_sub.rs
# Remove "claude-subscription = []" from Cargo.toml
grep -rn 'claude.subscription\|claude_sub' src/ frontend/
# Remove the ~6 #[cfg] blocks found by grep
```

Single commit, clean removal. No deep dependencies on the rest of the codebase.
