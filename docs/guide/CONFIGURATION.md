# Configuration Guide

SwarmLLM works out of the box with sensible defaults — you don't need to change anything to get started. But if you want to customize how it runs, this guide shows you how.

## How Configuration Works

SwarmLLM reads settings from three places, in this order of priority:

1. **Command-line flags** (highest priority) — e.g., `--port 9000`
2. **Environment variables** — e.g., `SWARMLLM_NODE_LISTEN_PORT=9000`
3. **Config file** — `config.toml` in your data directory
4. **Built-in defaults** (lowest priority)

This means a command-line flag always wins over whatever is in your config file.

### Where is my config file?

| Operating System | Config File Location |
|---|---|
| **Linux** | `~/.local/share/swarmllm/config.toml` |
| **macOS** | `~/Library/Application Support/swarmllm/config.toml` |
| **Windows** | `%APPDATA%\swarmllm\config.toml` |

The config file is created automatically on first run. You can also copy the example from the SwarmLLM download folder (`config/default.toml`) and edit it.

> **Tip:** You can specify a custom config file location with `--config /path/to/my-config.toml`.

---

## Common Recipes

### Change the port

By default, SwarmLLM runs on port **8800** (for both the web dashboard and peer-to-peer networking).

**Command line:**
```bash
./swarmllm run --port 9000
```

**Environment variable:**
```bash
SWARMLLM_NODE_LISTEN_PORT=9000 ./swarmllm run
```

**Config file:**
```toml
[node]
listen_port = 9000
```

---

### Set your display name

Give yourself a nickname that other nodes on the network can see (instead of a random ID string).

**In the web dashboard:** Click the gear icon (top-right) > Settings > type a nickname > Save.

**Config file:** Nicknames are set through the dashboard or API, not the config file. But you can set your region:
```toml
[identity]
region = "US"
```

Your region shows up on the Network Map (the world heatmap). Use a 2-letter country code like `US`, `DE`, `JP`, `BR`, `GB`, etc.

---

### Limit VRAM usage

If you have a GPU and want to control how much of its memory (VRAM) SwarmLLM uses:

```toml
[resources]
max_gpu_vram_mb = 4096    # Limit to 4 GB of VRAM
```

Set this to `0` (the default) to let SwarmLLM auto-detect and use available VRAM.

---

### Control GPU offloading

To use your GPU for faster inference, set how many model layers to offload to the GPU:

```toml
[inference]
gpu_layers = 35    # Offload 35 layers to GPU (higher = faster, uses more VRAM)
```

**Command line:**
```bash
./swarmllm run --gpu-layers 35
```

Set to `0` (the default) for CPU-only operation. A typical 7B model has ~32 layers; setting `gpu_layers = 99` offloads everything.

---

### Limit disk usage

Control how much disk space SwarmLLM uses for storing model shards:

```toml
[resources]
max_disk_mb = 50000    # 50 GB (this is the default)
```

---

### Enable or disable auto-download

When auto-manage is enabled, SwarmLLM automatically downloads popular model shards that are rare on the network — helping everyone while keeping your node useful.

```toml
[auto_manage]
enabled = true           # Turn on (default) or off
max_storage_mb = 25000   # Limit auto-downloads to 25 GB (0 = use 50% of max_disk_mb)
interval_minutes = 5     # How often to check for new shards to download
max_concurrent_downloads = 3  # Download up to 3 shards at once
```

---

### Adjust shard size

Shards are the pieces that models get split into for sharing across the network. Smaller shards = more granular distribution, but more network overhead.

```toml
[model]
shard_size_mb = 512    # Default: 512 MB per shard. Range: 64–2048.
```

> **Note:** Changing this only affects newly created shards. Existing shards keep their original size.

---

### Set API authentication

SwarmLLM has an OpenAI-compatible API that you can use with any tool that supports the OpenAI format. The API is protected by a bearer token (a password for API access).

**Finding your API key:** Open the dashboard, click the gear icon (top-right) > Settings. Your API key is shown at the top. Click **Copy** to copy it.

**Using the API key:**
```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model-name", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Setting a custom API key** (instead of the auto-generated one):
```toml
[api]
api_key = "my-secret-key-here"
```

If you leave `api_key` empty (the default), SwarmLLM generates a random key on first run and stores it securely.

> **Note:** The web dashboard itself does not require the API key — only external API calls need it.

---

### Connect to peers

SwarmLLM discovers peers automatically using multiple methods — you usually don't need to configure anything:

- **Same network (LAN):** Peers on the same Wi-Fi/LAN are found automatically via mDNS.
- **Returning user:** Previously-seen peers are remembered and reconnected on startup.
- **Invite codes:** Share a simple code with a friend to connect directly (see the Dashboard).
- **Peer exchange:** Once connected to any peer, you automatically discover more through them.

If you need to manually add a bootstrap peer:

```toml
[network]
bootstrap_peers = [
    "/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW...",
    "/ip4/198.51.100.20/udp/8800/quic-v1/p2p/12D3KooW..."
]
```

**Command line (for a single session):**
```bash
./swarmllm run --bootstrap "/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW..."
```

### Disable LAN discovery

If you don't want SwarmLLM to discover peers on your local network:

```toml
[network]
enable_mdns = false
```

### Create a private network

To run a private network that doesn't mix with the public SwarmLLM network:

```toml
[network]
gossip_network_id = "my-private-network"
```

Nodes with different `gossip_network_id` values can't see each other's gossip messages.

---

### Change logging verbosity

```toml
[logging]
level = "info"     # Options: error, warn, info, debug, trace
format = "pretty"  # Options: pretty, json
```

**Command-line shortcut:**
```bash
./swarmllm run -v      # Debug logging (SwarmLLM only)
./swarmllm run -vv     # Debug logging + libp2p networking details
./swarmllm run -vvv    # Trace logging (everything — very verbose)
```

**Log to a file:**
```toml
[logging]
file = "/var/log/swarmllm.log"
```

---

### Disable browser auto-open

```toml
[ui]
open_browser_on_start = false
```

---

### Set resource contribution level

```toml
[node]
contribution = "moderate"   # "minimal", "moderate", or "maximum"
```

- **Minimal** — Low impact, best for shared or low-spec machines.
- **Moderate** — Balanced, good for dedicated machines.
- **Maximum** — Uses as many resources as allowed. Best for dedicated servers.

---

### Schedule reduced usage during certain hours

```toml
[resources.schedule]
enabled = true
reduced_hours_start = 22    # 10 PM
reduced_hours_end = 8       # 8 AM
reduced_contribution = "minimal"
prune_aggressiveness = "aggressive"  # More aggressively prune shards at night
```

This automatically switches to "minimal" contribution between 10 PM and 8 AM, and prunes over-replicated shards more aggressively during those hours.

> **Tip:** The resource schedule can also be managed at runtime via the Dashboard (Schedule card) or API (`GET/PUT /api/admin/schedule`) without editing the config file.

---

## Full Config Reference

Below is every configuration option, organized by section.

### `[node]` — Basic Node Settings

| Option | Type | Default | Description |
|---|---|---|---|
| `listen_port` | integer | `8800` | Port for the web dashboard and P2P networking. |
| `data_dir` | path | Platform-specific (see above) | Where SwarmLLM stores data (models, database, keys). |
| `contribution` | string | `"minimal"` | How much resources to contribute: `"minimal"`, `"moderate"`, or `"maximum"`. |

### `[resources]` — Resource Limits

| Option | Type | Default | Description |
|---|---|---|---|
| `max_gpu_vram_mb` | integer | `0` | Max GPU memory in MB. `0` = auto-detect. |
| `max_ram_mb` | integer | `0` | Max system RAM in MB. `0` = auto (50% of system RAM). |
| `max_disk_mb` | integer | `50000` | Max disk space in MB for model storage (50 GB default). |
| `max_bandwidth_mbps` | integer | `0` | Max upload bandwidth in Mbps. `0` = unlimited. |

### `[resources.schedule]` — Usage Schedule

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Enable scheduled resource reduction. |
| `reduced_hours_start` | integer | `22` | Hour (0–23) to start reduced mode. |
| `reduced_hours_end` | integer | `8` | Hour (0–23) to end reduced mode. |
| `reduced_contribution` | string | `"minimal"` | Contribution level during reduced hours. |
| `prune_aggressiveness` | string | `"normal"` | Shard pruning aggressiveness during reduced hours: `"normal"`, `"aggressive"`, or `"conservative"`. |

### `[network]` — Networking

| Option | Type | Default | Description |
|---|---|---|---|
| `bootstrap_peers` | list of strings | `[]` | Peer addresses to connect to on startup. |
| `enable_mdns` | boolean | `true` | Discover peers on the same local network automatically. |
| `gossip_network_id` | string | none | Custom network ID for private networks (default: `"swarmllm-mainnet-v1"`). |
| `peer_exchange` | boolean | `true` | Share peer lists with connected nodes (helps everyone find each other). |
| `enable_relay` | boolean | `true` | Act as a relay for peers behind firewalls (helps the network). |
| `enable_relay_client` | boolean | `true` | Use relays to connect when behind a firewall. |
| `max_peers` | integer | `200` | Maximum number of simultaneous peer connections. |
| `auto_relay` | boolean | `true` | Automatically use relay when NAT (firewall) is detected. |
| `relay_max_circuit_duration_secs` | integer | `3600` | Max duration of a single relay circuit in seconds. |
| `relay_max_circuits` | integer | `16` | Max number of relay circuits to serve at once. |

### `[inference]` — AI Model Inference

| Option | Type | Default | Description |
|---|---|---|---|
| `default_model` | string | `""` | Default model for inference. Empty = use first available. |
| `session_timeout_seconds` | integer | `600` | How long a chat session stays in memory (10 min default). |
| `max_concurrent_requests` | integer | `10` | Max requests processed at the same time. |
| `model_path` | path | none | Path to a specific GGUF model file to load directly. |
| `gpu_layers` | integer | `0` | Number of model layers to offload to GPU. `0` = CPU only. |
| `kv_cache_ttl_secs` | integer | `600` | How long KV-cache (conversation memory) is kept. |
| `max_batch_size` | integer | `1` | Max requests batched together. `1` = no batching. |
| `batch_timeout_ms` | integer | `50` | Milliseconds to wait for a full batch before processing. |
| `speculative_decoding` | boolean | `false` | Enable speculative decoding (advanced, needs draft model). |
| `speculative_gamma` | integer | `4` | Draft tokens per verification step (speculative decoding). |
| `draft_model_path` | path | none | Path to a small draft model for speculative decoding. |
| `max_split_model_memory_mb` | integer | none | Max GPU memory for cached split-inference models. |
| `tensor_compression` | boolean | `false` | Enable zstd compression for tensor wire payloads. |
| `tensor_compress_level` | integer | `3` | Zstd compression level (1-22). Higher = smaller but slower. |
| `tensor_compress_threshold` | integer | `4096` | Minimum payload size in bytes to trigger compression. |
| `prefix_cache_max_entries` | integer | `256` | Maximum entries in the cross-request prefix cache. |

### `[logging]` — Log Output

| Option | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"info"` | Log detail level: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`. |
| `format` | string | `"pretty"` | Log format: `"pretty"` (human-readable) or `"json"`. |
| `file` | path | none | Write logs to this file (in addition to the terminal). |

### `[ui]` — Web Interface

| Option | Type | Default | Description |
|---|---|---|---|
| `open_browser_on_start` | boolean | `true` | Open the web dashboard in your browser when SwarmLLM starts. |
| `theme` | string | `"dark"` | UI color theme: `"dark"` or `"light"`. |

### `[api]` — API Authentication

| Option | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | none | Bearer token for API access. Empty = auto-generated on first run. |

### `[model]` — Model Storage

| Option | Type | Default | Description |
|---|---|---|---|
| `shard_size_mb` | integer | `512` | Size of each shard in MB when splitting models. Range: 64–2048. |

### `[auto_manage]` — Automatic Shard Management

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `true` | Automatically download popular model shards. |
| `max_storage_mb` | integer | `0` | Max disk space for auto-downloads in MB. `0` = 50% of `max_disk_mb`. |
| `interval_minutes` | integer | `5` | How often to check for new shards to download. |
| `max_shards` | integer | `0` | Max shards to hold. `0` = unlimited (within disk budget). |
| `max_concurrent_downloads` | integer | `3` | Max simultaneous shard downloads. |
| `prune_enabled` | boolean | `true` | Automatically remove over-replicated shards to free disk/VRAM. |
| `min_replicas` | integer | `2` | Minimum network-wide replica count before pruning is allowed. |
| `prune_cooldown_secs` | integer | `300` | Minimum seconds between prune actions for the same model. |
| `max_holder_load_for_prune` | integer | `3` | Block pruning if remaining holders' average load exceeds this. |

### `[pool]` — Device Pool

| Option | Type | Default | Description |
|---|---|---|---|
| `max_pool_size` | integer | `10` | Max devices in a single device pool. |
| `invitation_ttl_hours` | integer | `24` | How long a pool invitation stays valid. |
| `rate_limit_per_hour` | integer | `3` | Max pool join/leave/invite operations per hour. |
| `gossip_interval_secs` | integer | `600` | How often to share pool state with peers (10 min). |
| `credit_earn_rate` | float | `1.0` | Multiplier for credit earning rates within this pool. |
| `credit_spend_rate` | float | `1.0` | Multiplier for credit spending rates within this pool. |

### `[updates]` — Auto-Update

| Option | Type | Default | Description |
|---|---|---|---|
| `auto_update` | string | `"stable"` | Auto-update policy: `"disabled"`, `"stable"`, or `"all"`. |
| `check_interval_hours` | integer | `6` | How often to check for new versions. |
| `auto_restart` | boolean | `true` | Automatically restart after updating. |
| `keep_versions` | integer | `3` | Number of old versions to keep on disk. |

### `[identity]` — Your Identity

| Option | Type | Default | Description |
|---|---|---|---|
| `region` | string | none | Your country code for the network map (e.g., `"US"`, `"DE"`, `"JP"`). Voluntary. |

---

## Environment Variables

Every config option can be overridden with an environment variable using the `SWARMLLM_` prefix, with sections and options separated by underscores:

| Config Path | Environment Variable |
|---|---|
| `node.listen_port` | `SWARMLLM_NODE_LISTEN_PORT` |
| `node.data_dir` | `SWARMLLM_NODE_DATA_DIR` |
| `logging.level` | `SWARMLLM_LOGGING_LEVEL` |
| `inference.model_path` | `SWARMLLM_INFERENCE_MODEL_PATH` |
| `inference.gpu_layers` | `SWARMLLM_INFERENCE_GPU_LAYERS` |

**Example:**
```bash
SWARMLLM_NODE_LISTEN_PORT=9000 SWARMLLM_LOGGING_LEVEL=debug ./swarmllm run
```

---

## CLI Flags

| Flag | Short | Description |
|---|---|---|
| `--port <PORT>` | `-p` | Listen port |
| `--data-dir <PATH>` | `-d` | Data directory |
| `--config <PATH>` | `-c` | Config file path |
| `--model <PATH>` | `-m` | Path to a GGUF model file |
| `--gpu-layers <N>` | | Layers to offload to GPU |
| `--bootstrap <ADDR>` | | Bootstrap peer address (can be used multiple times) |
| `--verbose` | `-v` | Increase log verbosity (use `-v`, `-vv`, or `-vvv`) |

---

## Example: Minimal Config File

Here's a simple config file for everyday use:

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
