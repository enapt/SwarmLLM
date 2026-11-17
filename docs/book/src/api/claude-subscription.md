# Claude Subscription Provider

Use your existing Claude Pro, Max, Team, or Enterprise subscription to access Claude models through SwarmLLM — no API key or per-token charges needed.

> **Feature-gated**: Build with `--features claude-subscription` to enable. This feature is isolated behind a compile-time flag for easy removal.

## How It Works

When enabled, SwarmLLM spawns the `claude` CLI as a subprocess for each Claude model request:

```
Client Request (OpenAI or Anthropic format)
  → SwarmLLM API (openai/mod.rs / anthropic/mod.rs)
    → Provider resolution: model starts with "claude-"
      → Claude subscription enabled? → Spawn subprocess
      → Else: use Anthropic API key (existing behavior)
    → claude --print --output-format stream-json|json --model <model> "<the whole conversation>"
      or, for the next turn of a conversation this node answered:
      claude --print … --resume <its session> --fork-session "<the new message>"
    → Parse NDJSON → Translate to API format → Return response
```

Both the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints are supported, with streaming and non-streaming modes.

**A conversation continues its session.** Each reply's `claude` session is kept for 10 minutes under the conversation it now holds. When the next request carries that conversation plus a new message, the node resumes the session, forked, and sends only the new message, so the earlier turns are read from Claude's prompt cache instead of being sent again: on Haiku a second turn wrote 102 tokens to the cache instead of 17,426 and read 31,722 instead of 14,328 (on the API a cache write is priced at about twelve times a cache read). A fork leaves the stored session as it was, so editing or regenerating the last message resumes the conversation before it. If the session cannot be resumed, the whole conversation is sent instead. The sessions are saved where the `claude` CLI saves its own, under `~/.claude/projects/`, and the CLI deletes them after `cleanupPeriodDays` (30 days by default).

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
cargo build --release --features claude-subscription
```

The official downloads do not include this feature, so it needs a build from
source.

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
    "model": "claude-sonnet-5-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Anthropic format
curl http://localhost:8800/v1/messages \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5-5",
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

As with any OpenAI-compatible API, the client sends the whole conversation every time.

### Long-lived sessions

For interactive use, where re-sending the whole conversation every turn would be wasteful, the same build also offers sessions backed by one persistent `claude` subprocess (`--input-format stream-json`). Like everything on this page, they exist only in a build with `--features claude-subscription`:

- `GET /api/claude-code/sessions` — list live sessions
- `POST /api/claude-code/session` — start one (spawns the subprocess)
- `GET /api/claude-code/session/{id}` — its state
- `DELETE /api/claude-code/session/{id}` — close it and reap the subprocess
- `POST /api/claude-code/session/{id}/message` — send a turn
- `POST /api/claude-code/session/{id}/permission` — answer a tool-permission prompt the CLI raised

These sessions never use Claude Code's experimental Code Mode, which runs only in an interactive terminal: with `CLAUDE_CODE_EXPERIMENTAL_CODE_MODE=1` in your own settings, `claude -p` still starts with its ordinary tools (checked on v2.1.326).

## Configuration

All configuration is in the `providers.claude_subscription` section, manageable via the admin API or dashboard:

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Route Claude requests through the CLI |
| `claude_binary` | `"claude"` | Path to the `claude` binary |
| `default_model` | *(from request)* | Override model for all requests |
| `max_concurrent` | `3` | Max concurrent subprocess invocations (at most 8, whatever is set) |
| `timeout_secs` | `300` | Timeout per request (seconds) |
| `working_dir` | *(system temp)* | Working directory for the subprocess |

### Working Directory

By default, the subprocess runs in the system temp directory, so it loads no project's CLAUDE.md, hooks or MCP servers. Set `working_dir` to a project path if you want Claude to have that project's context; an empty value or `none` means the default.

## Routing Priority

When a `claude-*` model is requested:

1. **Claude subscription** (if enabled and CLI detected) — subprocess path, uses subscription
2. **Anthropic API key** (if configured) — direct API proxy, pay-per-token
3. **Error** — no provider available

The subscription provider takes priority over the API key. Disable the subscription toggle to fall back to API key billing.

## Rate Limits

Subscription rate limits are per rolling 5-hour window (not per-second RPM like API keys). The concurrency limiter (default 3, at most 8) stops the node spawning too many processes at once. Community reports suggest ~3-5 parallel Opus sessions before degradation.

Rate limit info is returned in the NDJSON output and logged. The `GET /api/admin/claude-subscription/status` endpoint shows the current rate limit tier.

## Removal

The feature is compiled in only with `--features claude-subscription`. To remove it:

- delete `src/api/claude_sub.rs` and `src/api/claude_session/`;
- remove `claude-subscription = []` from `Cargo.toml`, and `claude-subscription` from the four aliases in `.cargo/config.toml`;
- remove the `cfg(feature = "claude-subscription")` items, about 30 of them across 11 files under `src/api/`, `src/config/`, `src/daemon/` and `src/health/` (`grep -rn 'claude-subscription' src`);
- remove the Claude Subscription card and the Claude Code sessions view from the frontend (`settings.js`, `data.js`, `claude-code.js`), along with their translation keys.
