# CLI Flags & Environment Variables

## CLI Flags

| Flag | Short | Description |
|---|---|---|
| `--port <PORT>` | `-p` | Listen port |
| `--data-dir <PATH>` | `-d` | Data directory |
| `--config <PATH>` | `-c` | Config file path |
| `--model <PATH>` | `-m` | Path to a GGUF model file |
| `--gpu-layers <N>` | | Layers to put on the graphics card (`0` = processor only; leave it out for automatic placement) |
| `--bootstrap <ADDR>` | | Bootstrap peer address (repeatable) |
| `--shards <RANGE>` | | Advanced: shard range for split inference (e.g., `"0-4"`; `all` clears a saved range) |
| `--verbose` | `-v` | Increase log verbosity (`-v`, `-vv`, `-vvv`) |

## Subcommands

| Command | Description |
|---|---|
| `run` | Start SwarmLLM (the default when no subcommand is given). `--anchor` runs a relay-only computer with no models |
| `status` | Show what the running SwarmLLM is doing (`--json` for raw output) |
| `chat` | Chat in the terminal with streaming replies |
| `bench` | Measure speed against the running SwarmLLM |
| `peers` | List connected computers (`--json` for raw output) |
| `diagnostics` | Print a report for a bug report — safe to post publicly (`--full` includes network addresses; don't post that) |
| `get-model [smoke\|standard\|stress]` | Download a shared test model (no tier lists them; `--all` downloads every part) |
| `privacy <model-id>` | Fetch the first and last parts of a model so prompt privacy can switch on for it |
| `unload <model-id>` | Stop a model's worker and free its memory; the files stay |
| `remove-model <model-id>` | Delete a model from this computer and tell the network it's gone (`-y` skips the question) |
| `update` | Check for an update and install it (`--check-only` to only check) |
| `pool` | Link your own devices into a private group |
| `test-split` | Developer diagnostic: test split inference locally |
| `version` | Print version |

### `chat` Options

| Flag | Default | Description |
|---|---|---|
| `--model <ID>` | first available | Model to chat with |
| `--max-tokens <N>` | `1024` | Maximum tokens per reply |
| `--temperature <F>` | `0.7` | Sampling temperature |

### `bench` Options

| Flag | Default | Description |
|---|---|---|
| `--model-id <ID>` | first listed | Model to benchmark. Not `--model`, which names a GGUF file for `run` |
| `--prompt <TEXT>` | "Explain the theory of relativity in simple terms." | Benchmark prompt |
| `--max-tokens <N>` | `100` | Tokens to generate |
| `--iterations <N>` | `5` | Number of runs |
| `--concurrency <N>` | `1` | Requests sent at once |
| `--stream` | off | Stream replies and report time to first token |
| `--json` | off | Machine-readable output |

### `pool` Subcommands

Link your own devices so they serve each other privately. With the group's
**Private Mode** on, your requests stay on these devices (and, by default,
other SwarmLLM computers on your local network).

| Command | Description |
|---|---|
| `pool create --name "My Devices"` | Create a device group (this machine becomes the main device) |
| `pool invite-code` | Generate an invite code — a long `swarmpool://…` code, valid for 24 hours, single use |
| `pool join <CODE>` | Link this device using a code from your main machine |
| `pool status` | Show linked devices, their contribution level and online status |
| `pool leave` | Unlink this device from the group |

**Example flow:**
```bash
# Main device:
swarmllm pool create --name "My Devices"
swarmllm pool invite-code   # → swarmpool://… (a long code)

# On each other device:
swarmllm pool join 'swarmpool://…'
```

> **Note**: This links YOUR own devices. It's different from connecting to the SwarmLLM network (which uses `swarm://` peer addresses).

## Environment Variables

A small, fixed set of options — the ones a headless or Docker deployment needs
before a config file exists — can be set with a `SWARMLLM_` environment
variable. **This table is the complete list**; any other `SWARMLLM_*` variable
is ignored. Everything else is set in `config.toml` or from the dashboard.

| Config Path | Environment Variable | Notes |
|---|---|---|
| `node.listen_port` | `SWARMLLM_NODE_LISTEN_PORT` | |
| `node.data_dir` | `SWARMLLM_NODE_DATA_DIR` | |
| `logging.level` | `SWARMLLM_LOGGING_LEVEL` | `trace`, `debug`, `info`, `warn` or `error`; `-v`/`-vv` still take precedence |
| `inference.model_path` | `SWARMLLM_INFERENCE_MODEL_PATH` | |
| `inference.gpu_layers` | `SWARMLLM_INFERENCE_GPU_LAYERS` | `0` = processor only |
| `api.api_key` | `SWARMLLM_API_KEY` | a deterministic key for Docker; empty means unset |
| `network.bootstrap_peers` | `SWARMLLM_NETWORK_BOOTSTRAP_PEERS` | multiaddrs separated by commas, spaces or newlines |

Example:
```bash
SWARMLLM_NODE_LISTEN_PORT=9000 ./swarmllm run -v
```

## Provider API Keys via Environment

Cloud provider API keys use standard environment variable names:

| Provider | Environment Variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| SambaNova | `SAMBANOVA_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| DeepInfra | `DEEPINFRA_API_KEY` |
| Moonshot/Kimi | `MOONSHOT_API_KEY` |

These can also be placed in a `.env` file in your data directory:

```bash
# ~/.local/share/swarmllm/.env
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-...
NVIDIA_NIM_API_KEY=nvapi-...
```

The `.env` file is loaded at startup. It does not override existing environment variables or keys already configured via the dashboard/database. The dashboard settings UI shows "From .env" for keys loaded this way.
