# CLI Flags & Environment Variables

## CLI Flags

| Flag | Short | Description |
|---|---|---|
| `--port <PORT>` | `-p` | Listen port |
| `--data-dir <PATH>` | `-d` | Data directory |
| `--config <PATH>` | `-c` | Config file path |
| `--model <PATH>` | `-m` | Path to a GGUF model file |
| `--gpu-layers <N>` | | Layers to offload to GPU |
| `--bootstrap <ADDR>` | | Bootstrap peer address (repeatable) |
| `--shards <RANGE>` | | Shard range for split inference (e.g., `"0-4"`) |
| `--verbose` | `-v` | Increase log verbosity (`-v`, `-vv`, `-vvv`) |

## Subcommands

| Command | Description |
|---|---|
| `run` | Start the daemon (default if no subcommand) |
| `status` | Query running daemon status |
| `chat` | Interactive CLI chat with streaming |
| `bench` | Benchmark inference (tokens/sec, TTFT) |
| `peers` | List connected peers |
| `pool` | Device pool management (link your machines) |
| `test-split` | Test split inference locally (diagnostic) |
| `version` | Print version |

### `chat` Options

| Flag | Default | Description |
|---|---|---|
| `--model-name <NAME>` | auto-detect | Model to chat with |
| `--system <TEXT>` | none | System prompt |
| `--max-tokens <N>` | `2048` | Max tokens per response |
| `--temperature <F>` | `0.7` | Sampling temperature |

### `bench` Options

| Flag | Default | Description |
|---|---|---|
| `--model-name <NAME>` | auto-detect | Model to benchmark |
| `--prompt <TEXT>` | "Write a short essay..." | Benchmark prompt |
| `--max-tokens <N>` | `128` | Tokens to generate |
| `--iterations <N>` | `1` | Number of benchmark runs |

### `pool` Subcommands

Link your personal devices so credits are combined on one main machine.

| Command | Description |
|---|---|
| `pool create --name "My Devices"` | Create a device group (this machine becomes the main device) |
| `pool invite-code` | Generate an 8-character invite code to share |
| `pool join <CODE>` | Link this device using a code from your main machine |
| `pool status` | Show linked devices, credits, and online status |
| `pool leave` | Unlink this device from the group |

**Example flow:**
```bash
# Main device:
swarmllm pool create --name "My Devices"
swarmllm pool invite-code   # → A3F7K2M9

# On each other device:
swarmllm pool join A3F7K2M9
```

> **Note**: This links YOUR own devices. It's different from connecting to the SwarmLLM network (which uses `swarm://` peer addresses).

## Environment Variables

Every config option can be set via `SWARMLLM_` prefix:

| Config Path | Environment Variable |
|---|---|
| `node.listen_port` | `SWARMLLM_NODE_LISTEN_PORT` |
| `node.data_dir` | `SWARMLLM_NODE_DATA_DIR` |
| `logging.level` | `SWARMLLM_LOGGING_LEVEL` |
| `inference.model_path` | `SWARMLLM_INFERENCE_MODEL_PATH` |
| `inference.gpu_layers` | `SWARMLLM_INFERENCE_GPU_LAYERS` |

Example:
```bash
SWARMLLM_NODE_LISTEN_PORT=9000 SWARMLLM_LOGGING_LEVEL=debug ./swarmllm run
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
