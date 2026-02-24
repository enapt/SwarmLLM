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

- **Distributed Inference** — Model layers sharded across nodes with automatic pipeline assembly
- **OpenAI-Compatible API** — `POST /v1/chat/completions` with streaming support, works with Open WebUI, SillyTavern, LangChain, etc.
- **Credit System** — Earn credits by serving inference, hosting shards, and seeding data. Higher contribution = faster responses
- **P2P Networking** — libp2p with Kademlia DHT, GossipSub, QUIC transport, NAT traversal
- **Built-in Web UI** — Admin dashboard, chat interface, and first-run setup wizard
- **Fault Tolerant** — Hot-standby failover, shard replication, automatic rebalancing
- **Self-Updating** — Decentralized governance, proposals, voting, and canary rollouts

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
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
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

Internally, the daemon runs as async Tokio tasks communicating via channels:

- **NetworkManager** — libp2p swarm lifecycle, peer discovery, message routing
- **InferenceRouter** — request queuing, pipeline assembly, execution
- **CreditLedger** — balance tracking, transaction signing, priority tiers
- **HealthMonitor** — periodic checks, rebalancing triggers
- **ApiServer** — Axum HTTP server, WebSocket for live updates

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
# Requirements: Rust 1.75+, cmake (for llama.cpp)
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM
cargo build --release
```

## CLI

```
swarmllm <COMMAND>

Commands:
  run         Start the SwarmLLM daemon (default)
  status      Show node status
  models      List available models
  credits     Show credit balance and tier
  config      Print current configuration
  identity    Manage node identity (export/import/show)
  version     Print version information

Options:
  -c, --config <PATH>     Config file path [default: ~/.swarmllm/config.toml]
  -p, --port <PORT>       Listen port [default: 8800]
  -d, --data-dir <PATH>   Data directory [default: ~/.swarmllm]
  -v, --verbose           Increase log verbosity (-v, -vv, -vvv)
  --headless              No browser, no setup wizard
```

## Configuration

Config lives at `~/.swarmllm/config.toml`. All values can be overridden with environment variables using the `SWARMLLM_` prefix:

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_LOGGING_LEVEL=debug
```

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
| Networking | libp2p (QUIC transport) |
| Serialization | Cap'n Proto (tensors), serde_json (API) |
| HTTP Server | Axum |
| Inference | llama.cpp bindings (GGUF) |
| Database | sled (embedded) |
| Cryptography | Ed25519, BLAKE3, Argon2id |

## License

TBD
