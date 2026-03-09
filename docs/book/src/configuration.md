# Configuration

SwarmLLM works out of the box with sensible defaults. This section covers customization.

## Config Priority

Settings are read from four sources, in order of priority:

1. **Command-line flags** (highest) — e.g., `--port 9000`
2. **Environment variables** — e.g., `SWARMLLM_NODE_LISTEN_PORT=9000`
3. **Config file** — `config.toml` in your data directory
4. **Built-in defaults** (lowest)

Provider API keys have an additional source: a **`.env` file** in the data directory or current working directory. Standard env var names are used (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, etc.). The `.env` file does not override existing environment variables or keys already set via the dashboard.

## Config File Location

| OS | Path |
|---|---|
| **Linux** | `~/.local/share/swarmllm/config.toml` |
| **macOS** | `~/Library/Application Support/swarmllm/config.toml` |
| **Windows** | `%APPDATA%\swarmllm\config.toml` |

Specify a custom path: `--config /path/to/config.toml`

## Minimal Example

```toml
[node]
listen_port = 8800
contribution = "moderate"

[resources]
max_disk_mb = 50000

[identity]
region = "US"

[inference]
gpu_layers = 35

[auto_manage]
enabled = true
```

## Chapters

- [Config File Reference](./configuration/reference.md) — Every option explained
- [Shard-Only Mode](./configuration/shard-only.md) — Distributed inference with partial models
- [CLI Flags & Environment Variables](./configuration/cli-env.md) — Command-line and env var reference
