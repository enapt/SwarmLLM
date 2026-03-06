# SwarmLLM

[![CI](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml)

Decentralized peer-to-peer LLM inference network. A single Rust binary that shards large language models across a network of contributing nodes, enabling access to 70B+ parameter models without expensive hardware.

**Like Ollama, but you don't need a beefy GPU — because the network IS your GPU.**

## How It Works

SwarmLLM distributes transformer model layers across a pool of peer-to-peer nodes. Each participant contributes a fraction of the required compute, and the network orchestrates inference pipelines that chain these nodes together. The result: anyone can run state-of-the-art open-weight models by pooling resources with others.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node A  │────▶│  Node B  │────▶│  Node C  │
│ Layers   │     │ Layers   │     │ Layers   │──▶ Response
│  0-15    │     │  16-47   │     │  48-79   │
└──────────┘     └──────────┘     └──────────┘
```

- **No central server** — fully peer-to-peer with no single point of failure
- **No accounts or subscriptions** — just a cryptographic identity
- **Zero configuration** — nodes find each other automatically via mDNS, peer cache, and peer exchange
- **BitTorrent-inspired incentives** — contribute compute, earn priority access
- **OpenAI-compatible API** — drop-in replacement for any tool that speaks OpenAI

## Quick Start

**Download a binary** from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases) for your platform, extract it, and run:

```bash
./swarmllm run
```

Your browser opens to `localhost:8800`. The setup wizard auto-detects your hardware. Pick a model, download it, start chatting.

See the full [Getting Started Guide](docs/guide/GETTING_STARTED.md) for platform-specific instructions.

**Or use Docker:**

```bash
docker run -p 8800:8800 -v swarmllm-data:/data swarmllm/swarmllm
```

**Or use the API directly:**

```bash
# Get your API key from the dashboard or:
curl http://localhost:8800/api/admin/api-key

# Use it for inference:
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama3-70b-q4km",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Connecting to the Network

SwarmLLM uses a 5-layer discovery stack — no manual configuration needed:

| Layer | How it works | When it kicks in |
|-------|-------------|------------------|
| **mDNS** | Automatically discovers peers on the same LAN/Wi-Fi | Instantly on startup |
| **Peer Cache** | Remembers peers from previous sessions (redb-backed, max 200) | On restart |
| **Invite Codes** | Share a `swarm://...` code with a friend to connect directly | First time joining |
| **Peer Exchange** | Connected peers share their known peer lists with you | On each new connection |
| **Kademlia DHT** | Distributed hash table for network-wide peer routing | Continuously |

**First launch on a LAN:** Two nodes on the same network find each other in seconds — zero config.

**First launch alone:** The dashboard shows your invite code. Share it with a friend. Once you're connected to one peer, PEX and DHT discover the rest of the network automatically.

**Returning user:** Cached peers reconnect in under a second. The invite code UI auto-hides once your node knows 20+ peers.

> For private networks, set `gossip_network_id = "my-private-net"` in config to isolate your nodes from the public network.

## Features

### Inference
- **Distributed Inference** — Model layers sharded across nodes with automatic pipeline assembly, ~130ms per-token pipeline latency over TCP. Candle-based direct tensor computation with E2E encryption
- **Architecture-Aware** — Automatic detection of model architecture (Llama, Llama 4, Qwen2, Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3, GLM-4) with correct RoPE, attention biases, EOS tokens, and context lengths from GGUF metadata
- **DeepSeek MoE+MLA** — Full support for DeepSeek-V2/V3 models: Multi-head Latent Attention (low-rank Q/KV compression), Mixture-of-Experts (router-based top-k expert selection with shared experts), per-layer dense/MoE detection
- **Multi-Provider Gateway** — Route requests to cloud providers (OpenAI, Anthropic, DeepSeek, Mistral, Groq) when the model isn't available locally. Native Anthropic Messages API at `/v1/messages`. Model prefix routing or explicit `provider:model` syntax
- **OpenAI-Compatible API** — `POST /v1/chat/completions` with streaming, tool calling, logprobs, embeddings. Drop-in for Open WebUI, SillyTavern, LangChain, etc.
- **Tensor Parallelism** — Automatic tensor-parallel splitting for LAN peers on the same subnet, complementing pipeline parallelism for WAN
- **Vision & Adapters** — VLM support (LLaVA, Qwen2-VL) and per-request LoRA adapter loading
- **Speculative Decoding** — Draft model + rejection sampling for 2-3x local inference throughput
- **Batched Inference** — True GPU batching: multiple concurrent requests stacked into batch tensors for parallel computation
- **Multi-turn KV-cache** — Session-aware cache reuse, cross-request prefix caching, chunked prefill, flash attention (CPU + GPU), paged attention (CUDA block pool)

### Networking & Security
- **Zero-Config Discovery** — 5-layer stack: mDNS, persistent peer cache, shareable invite codes, peer exchange (PEX), Kademlia DHT
- **P2P Networking** — libp2p with Kademlia DHT, GossipSub, TCP+Yamux (primary) and QUIC transport, NAT traversal (auto-relay), connection limits, gossip replay protection
- **End-to-End Encryption** — Three-tier: pairwise sessions (X25519 + ChaCha20-Poly1305 with forward secrecy), pipeline sealing, and authenticated sealed gossip
- **Hidden States API** — `/v1/internal/hidden-states` exposes per-layer activations for research (adapter insertion, activation inspection)
- **Sybil Resistance** — Ed25519-signed balance reports, peer reputation scoring with trust decay, subnet clustering detection, leaderboard spoofing protection
- **API Authentication** — Bearer token middleware with auto-generated keys, CORS lockdown, SSRF protection, Content-Security-Policy

### Economy & Identity
- **Credit System** — Earn credits by serving inference, hosting shards, and seeding data. Higher contribution = faster responses. Anti-gaming protection, transaction replay prevention, and credit escrow for large requests
- **Identity & Pools** — Cryptographic nicknames, leaderboard, multi-device credit pooling with dual-signature invitation protocol
- **Auto-Shard Management** — VRAM-aware automatic shard acquisition from HuggingFace (with resume, retry, and Range headers) and peers with popularity/rarity scoring. Smart pruning auto-removes over-replicated shards based on demand, resource pressure, and region diversity

### Operations
- **Built-in Web UI** — Admin dashboard with operation mode indicator (Swarm/Cloud/Hybrid/Standalone), chat interface, model browser, shard visualization, first-run setup wizard, collapsible panels, mobile-responsive layout
- **Fault Tolerant** — JoinSet-based task supervisor with restart-on-crash, hot-standby failover, shard replication, automatic rebalancing, atomic shard writes, download retry with backoff
- **Observability** — Prometheus `/metrics` endpoint, startup readiness probe `/health/ready`, structured startup logging, database integrity checks
- **Config Hot-Reload** — Change operational parameters without restarting via SIGHUP or API
- **Auto-Updater** — Checks GitHub releases for new versions, downloads and replaces binary with restart prompt
- **Python SDK** — `pip install swarmllm-client` for programmatic access

## Architecture

Single Rust binary, three simultaneous functions:

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| P2P Node | Peer discovery, shard hosting, distributed inference, credits | libp2p / TCP+QUIC |
| LLM API Server | OpenAI-compatible inference endpoint | `localhost:8800/v1/*` |
| Management UI | Dashboard, settings, model browser, chat | `localhost:8800/admin` |

Internally, the daemon runs 10 async Tokio tasks communicating via channels:

```
┌──────────────────────────────────────────────────────────────────────┐
│                           daemon.rs                                   │
│                                                                      │
│  NetworkManager ──── InferenceRouter ──── CreditLedger               │
│       │                    │                   │                      │
│  MessageDispatcher    ApiServer          HealthMonitor                │
│       │                    │                   │                      │
│  PoolManager        AutoShardManager    ShardRebalancer               │
│       │                    │                   │                      │
│  AcquisitionManager ───────┴───────────────────┘                     │
│                                                                      │
│              All connected via mpsc channels                         │
│              Shared state via Arc<SharedState> + DashMap              │
└──────────────────────────────────────────────────────────────────────┘
```

## Node Tiers

| Tier | Requirements | Role |
|------|-------------|------|
| Super Node | Full model in VRAM, high bandwidth | Serves full inference independently |
| Standard Node | Partial VRAM/RAM, moderate bandwidth | Holds layer shards, joins inference pipelines |
| Light Node | Minimal resources | Primarily consumer, contributes bandwidth |

## Credit System

| Action | Effect |
|--------|--------|
| Serve inference | +credits (per layer per token) |
| Host model shards | +credits (per GB per hour) |
| Seed shard data | +credits (per GB transferred) |
| Submit inference request | -credits (per layer per token) |

Credits determine your priority tier:

- **Platinum** (top 10%) — near-instant responses
- **Gold** (top 30%) — 1-3 second queue
- **Silver** (positive balance) — 5-15 second queue
- **Bronze** (zero/negative) — 30+ second queue, but never locked out

## Supported Models

SwarmLLM supports 10 transformer architectures via native candle inference with GGUF quantization:

| Architecture | Examples | Special Features |
|-------------|----------|-----------------|
| **Llama** | Llama 2/3, CodeLlama, TinyLlama | Interleaved RoPE, GQA |
| **Llama 4** | Llama 4 Scout (17B), Maverick (400B) | iRoPE (NoPE every 4th layer), MoE |
| **Qwen2** | Qwen2.5-Coder-7B/32B | QKV biases, 32k context |
| **DeepSeek-V2/V3** | DeepSeek-V2-Lite, DeepSeek-V3 (671B) | MLA attention + MoE FFN |
| **GLM-4** | GLM-4-9B, GLM-4.7 MoE | Partial RoPE, extreme GQA (16:1) |
| **Gemma/Gemma2** | Gemma 2B/7B, Gemma2 9B/27B | Contiguous RoPE |
| **Phi-3** | Phi-3-mini, Phi-3-medium | Su/YaRN RoPE, biases |
| **Mistral** | Mistral 7B, Mistral Nemo | GQA, interleaved RoPE |
| **Starcoder2** | Starcoder2 3B/7B/15B | Code-optimized, biases |
| **Mixtral** | Mixtral 8x7B, 8x22B | MoE (via llama.cpp) |

Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16

## Installation

### Pre-built Binaries (Recommended)

Download from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases) — available for Linux, macOS (Intel & Apple Silicon), and Windows. CUDA-accelerated Linux builds included.

### Docker

```bash
# Single node
docker run -p 8800:8800 -v swarmllm-data:/data swarmllm/swarmllm

# With NVIDIA GPU
docker run --gpus all -p 8800:8800 -v swarmllm-data:/data swarmllm/swarmllm:cuda
```

### Building from Source

```bash
# Requirements: Rust 1.80+, cmake (for llama.cpp, optional)
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM

# CPU-only build (no model loading)
cargo build --release

# With llama.cpp inference support
cargo build --release --features llama

# With GPU acceleration
cargo build --release --features cuda    # NVIDIA
cargo build --release --features metal   # Apple Silicon
cargo build --release --features rocm    # AMD
```

## CLI

```
swarmllm <COMMAND>

Commands:
  run         Start the SwarmLLM daemon (default if omitted)
  status      Show node status (queries running daemon)
  chat        Interactive terminal chat with a running node
  bench       Run inference benchmarks (tokens/sec, latency)
  peers       List connected peers with latency and trust
  test-split  Test split inference locally (single-node diagnostic)
  version     Print version information

Options:
  -c, --config <PATH>       Config file path
  -p, --port <PORT>         Listen port [default: 8800]
  -d, --data-dir <PATH>     Data directory [default: ~/.local/share/swarmllm]
  -m, --model <PATH>        Path to a GGUF model file to load
      --gpu-layers <N>      Number of layers to offload to GPU (0 = CPU only)
      --bootstrap <ADDR>    Bootstrap peer multiaddr
      --shards <RANGE>      Claim specific layer range for split inference (e.g. "0-15")
  -v, --verbose             Increase log verbosity (-v, -vv, -vvv)
```

## Configuration

Config lives at `~/.local/share/swarmllm/config.toml` (Linux), `~/Library/Application Support/swarmllm/config.toml` (macOS), or `%APPDATA%\swarmllm\config.toml` (Windows). All values can be overridden with environment variables using the `SWARMLLM_` prefix:

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_LOGGING_LEVEL=debug
```

### Key Config Sections

| Section | Key Settings |
|---------|-------------|
| `[node]` | `listen_port`, `contribution` (minimal/moderate/maximum), `data_dir` |
| `[resources]` | `max_gpu_vram_mb`, `max_ram_mb`, `max_disk_mb`, `max_bandwidth_mbps` |
| `[network]` | `bootstrap_peers`, `enable_mdns`, `enable_autonat`, `enable_dcutr`, `enable_encryption`, `gossip_network_id`, `enable_relay`, `max_peers` |
| `[inference]` | `model_path`, `gpu_layers`, `session_timeout_seconds`, `max_batch_size` |
| `[pool]` | `max_pool_size`, `invitation_ttl_hours`, `rate_limit_per_hour` |
| `[auto_manage]` | `enabled`, `max_storage_mb`, `interval_minutes`, `max_concurrent_downloads`, `prune_enabled`, `min_replicas` |
| `[providers]` | `openai_api_key`, `anthropic_api_key`, `deepseek_api_key`, `mistral_api_key`, `groq_api_key`, custom providers |
| `[logging]` | `level`, `format` (pretty/json) |
| `[ui]` | `open_browser_on_start`, `theme` |

See the [Configuration Guide](docs/guide/CONFIGURATION.md) for the full reference.

## Platform Support

| Platform | GPU Support | Status |
|----------|------------|--------|
| Linux x86_64 | CUDA + ROCm | Primary target |
| macOS aarch64 | Metal | Supported |
| Windows x86_64 | CUDA | Supported |
| macOS x86_64 | CPU only | Best-effort |
| Linux aarch64 | CPU only | Best-effort |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Rust (2021 edition) |
| Async Runtime | Tokio |
| Networking | libp2p 0.55 (TCP+Yamux + QUIC + mDNS + Kademlia + GossipSub) |
| Serialization | serde_json (API), binary with type-tag (tensors), zstd compression |
| HTTP Server | Axum 0.7 |
| Inference | candle (split/distributed, CUDA, flash/paged attention), llama.cpp (single-node) |
| Database | redb (embedded, ACID) |
| Cryptography | Ed25519 (identity), X25519 + ChaCha20-Poly1305 (E2E), BLAKE3 (integrity) |
| Monitoring | Prometheus + Grafana |
| SDK | Python (`swarmllm-client`) |

## How SwarmLLM Compares

| Feature | SwarmLLM | Petals | Exo | Bittensor |
|---------|----------|--------|-----|-----------|
| **Language** | Rust (single binary) | Python | Python | Python + Substrate |
| **Deployment** | Download & run | pip install | pip install | pip + blockchain setup |
| **Scale** | Internet-scale (NAT traversal, DHT, relay) | Internet (volunteer) | LAN only | Internet (blockchain) |
| **E2E Encryption** | X25519 + ChaCha20 + forward secrecy | None | None | Minimal |
| **Incentives** | Credit tiers (no token) | None | None | TAO token (real money) |
| **Parallelism** | Pipeline + tensor (LAN) | Pipeline | Tensor + pipeline | Subnet routing |
| **Model Architectures** | 10 (incl. DeepSeek MoE+MLA, GLM-4, Llama 4) | 4 | 6+ | Any |
| **Shard-Only Mode** | Yes (no full model needed) | No | No | N/A |
| **Multi-Provider Gateway** | Yes (OpenAI, Anthropic, etc.) | No | No | No |
| **VLM + LoRA** | Yes | LoRA only | No | Subnet-specific |
| **API Compatibility** | OpenAI + Anthropic | PyTorch | OpenAI basic | Subnet-defined |
| **Auto-Update** | Built-in version check + self-update | No | No | No |
| **Test Suite** | 525 tests | Limited | Limited | Varies |

See the full [Competitive Analysis](docs/COMPETITIVE_ANALYSIS.md) for detailed breakdowns.

## Documentation

- **[Getting Started](docs/guide/GETTING_STARTED.md)** — Download, install, and start chatting in minutes
- **[Configuration](docs/guide/CONFIGURATION.md)** — All config options, environment variables, CLI flags
- **[Troubleshooting](docs/guide/TROUBLESHOOTING.md)** — Common issues and solutions
- **[Architecture](docs/ARCHITECTURE.md)** — Deep dive into subsystems, protocols, and security model

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
