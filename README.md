# SwarmLLM

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
- **BitTorrent-inspired incentives** — contribute compute, earn priority access
- **OpenAI-compatible API** — drop-in replacement for any tool that speaks OpenAI

## Features

- **Distributed Inference** — Model layers sharded across nodes with automatic pipeline assembly using candle for direct tensor computation
- **Architecture-Aware** — Automatic detection of model architecture (Llama, Qwen2, Mistral, etc.) with correct RoPE, attention biases, EOS tokens, and context lengths from GGUF metadata
- **OpenAI-Compatible API** — `POST /v1/chat/completions` and `/v1/completions` with streaming support, works with Open WebUI, SillyTavern, LangChain, etc.
- **Credit System** — Earn credits by serving inference, hosting shards, and seeding data. Higher contribution = faster responses. Anti-gaming protection, transaction replay prevention, and credit escrow for large requests
- **P2P Networking** — libp2p with Kademlia DHT, GossipSub, QUIC transport, NAT traversal (auto-relay on NAT detection), connection limits, and gossip replay protection
- **End-to-End Encryption** — Three-tier encryption: pairwise sessions (X25519 + ChaCha20-Poly1305 with forward secrecy via ephemeral ECDH), pipeline sealing, and authenticated sealed gossip
- **Sybil Resistance** — Ed25519-signed balance reports with timestamp freshness, peer reputation scoring with trust decay, leaderboard spoofing protection
- **Identity & Pools** — Cryptographic nicknames, leaderboard, multi-device credit pooling with dual-signature invitation protocol and privacy-preserving blind signatures
- **Auto-Shard Management** — VRAM-aware automatic shard acquisition from HuggingFace (with resume, retry, and Range headers) and peers with popularity/rarity scoring
- **Speculative Decoding** — Draft model + rejection sampling for 2-3x local inference throughput
- **Batched Inference** — True GPU batching: multiple concurrent requests stacked into batch tensors for parallel computation
- **Multi-turn KV-cache** — Session-aware cache reuse skips redundant prefill across chat messages
- **Built-in Web UI** — Admin dashboard, chat interface, model browser, shard visualization, first-run setup wizard, mobile-responsive layout
- **Fault Tolerant** — JoinSet-based task supervisor with restart-on-crash, hot-standby failover, shard replication, automatic rebalancing, atomic shard writes, download retry with backoff
- **Observability** — Prometheus `/metrics` endpoint, startup readiness probe `/health/ready`, structured startup logging, database integrity checks
- **API Authentication** — Bearer token middleware with auto-generated keys, CORS lockdown, SSRF protection, Content-Security-Policy, config hot-reload via SIGHUP

## Quick Start

```bash
# Download and run
./swarmllm run

# Browser opens to localhost:8800
# Setup wizard auto-detects hardware
# Start chatting in minutes
```

Or use the API directly:

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

## Architecture

Single Rust binary, three simultaneous functions:

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| P2P Node | Peer discovery, shard hosting, distributed inference, credits | libp2p / QUIC |
| LLM API Server | OpenAI-compatible inference endpoint | `localhost:8800/v1/*` |
| Management UI | Dashboard, settings, model browser, chat | `localhost:8800/admin` |

Internally, the daemon runs 10 async Tokio tasks communicating via channels:

- **NetworkManager** — libp2p swarm lifecycle, peer discovery, message routing
- **InferenceRouter** — request queuing, pipeline assembly, execution
- **MessageDispatcher** — routes inbound network messages to subsystems
- **CreditLedger** — balance tracking, transaction signing, anti-gaming
- **HealthMonitor** — periodic checks, rebalancing triggers, stale channel cleanup
- **ShardRebalancer** — shard redistribution on node join/leave
- **AcquisitionManager** — BLAKE3-verified model download from peers with retry/backoff
- **ApiServer** — Axum HTTP server, WebSocket with ping/pong heartbeat
- **PoolManager** — device pool management, credit forwarding, invitation protocol
- **AutoShardManager** — VRAM-aware automatic shard acquisition from HuggingFace/peers

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

SwarmLLM targets decoder-only transformer architectures with GGUF quantization:

- Llama 2/3, CodeLlama
- Mistral, Mixtral (MoE)
- Qwen/Qwen2
- DeepSeek (including MoE variants)
- Phi-family

Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16

## Building from Source

```bash
# Requirements: Rust 1.75+, cmake (for llama.cpp, optional)
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
  test-split  Test split inference locally (single-node diagnostic)
  version     Print version information

Options:
  -c, --config <PATH>       Config file path
  -p, --port <PORT>         Listen port [default: 8800]
  -d, --data-dir <PATH>     Data directory [default: ~/.swarmllm]
  -m, --model <PATH>        Path to a GGUF model file to load
      --gpu-layers <N>      Number of layers to offload to GPU (0 = CPU only)
      --bootstrap <ADDR>    Bootstrap peer multiaddr
      --shards <RANGE>      Claim specific layer range for split inference (e.g. "0-15")
  -v, --verbose             Increase log verbosity (-v, -vv, -vvv)
```

## Configuration

Config lives at `~/.swarmllm/config.toml`. All values can be overridden with environment variables using the `SWARMLLM_` prefix (invalid values log warnings instead of silently falling back):

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_LOGGING_LEVEL=debug
```

### Default Config Sections

| Section | Key Settings |
|---------|-------------|
| `[node]` | `listen_port`, `contribution` (minimal/moderate/maximum), `data_dir` |
| `[resources]` | `max_gpu_vram_mb`, `max_ram_mb`, `max_disk_mb`, `max_bandwidth_mbps` |
| `[network]` | `bootstrap_peers`, `enable_relay`, `max_peers` |
| `[inference]` | `model_path`, `gpu_layers`, `session_timeout_seconds`, `max_concurrent_requests` |
| `[credit]` | Starting balance, earn/spend rates |
| `[pool]` | `max_pool_size`, `invitation_ttl_hours`, `rate_limit_per_hour` |
| `[auto_manage]` | `enabled`, `max_storage_mb`, `interval_minutes`, `max_shards`, `max_concurrent_downloads` |
| `[logging]` | `level`, `format` (pretty/json) |
| `[ui]` | `open_browser_on_start` |

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
| Networking | libp2p 0.54 (QUIC transport) |
| Serialization | Cap'n Proto (tensors), serde_json (API) |
| HTTP Server | Axum 0.7 |
| Inference | candle (split/distributed, CUDA), llama.cpp (single-node) |
| Database | sled (embedded, schema-versioned) |
| Cryptography | Ed25519 (identity), X25519 + ChaCha20-Poly1305 (E2E), BLAKE3 (integrity) |

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
