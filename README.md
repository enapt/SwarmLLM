# SwarmLLM

[![CI](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust 1.80+](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/enapt/swarmllm)
[![Release](https://img.shields.io/github/v/release/enapt/SwarmLLM?include_prereleases&label=release)](https://github.com/enapt/SwarmLLM/releases)

Decentralized peer-to-peer LLM inference network. A single Rust binary that shards large language models across a network of contributing nodes, enabling access to 70B+ parameter models without expensive hardware or paid API tokens.

**Join the swarm. Run AI together — for free.**

> **Status:** Alpha — actively developed. Distributed inference stable, tested across multi-node deployments on real networks. 649 tests, comprehensive security auditing. [Report issues](https://github.com/enapt/SwarmLLM/issues).

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [What Is This?](#what-is-this)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Connecting to the Network](#connecting-to-the-network)
- [Features](#features)
- [Architecture](#architecture)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [CLI](#cli)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [How SwarmLLM Compares](#how-swarmllm-compares)
- [Documentation](#documentation)
- [Development Transparency](#development-transparency)
- [Contributing](#contributing)
- [License](#license)

</details>

## What Is This?

SwarmLLM lets you run AI chatbots (like ChatGPT, but open-source) on your own computer — or share the work with others across the internet. Think of it like BitTorrent, but for AI: instead of downloading movies, you're sharing the computing power needed to run large language models.

**Why does this matter?**

Running a smart AI model (like Llama 3 70B) normally requires a $10,000+ GPU. With SwarmLLM, the model gets split into pieces — your computer handles some layers, your friend's handles others, and together you can run models none of you could run alone. No cloud subscription, no API fees, and **all traffic is encrypted end-to-end** — relay nodes never see your data.

**What can you do with it?**

- **Chat with AI** — Open `localhost:8800` in your browser, pick a model, and start chatting. Works just like ChatGPT.
- **Use it as an API** — Any tool that works with OpenAI (LangChain, Open WebUI, SillyTavern, etc.) works with SwarmLLM. Just point it at `localhost:8800`.
- **Use it with Claude Code** — SwarmLLM speaks the Anthropic API natively, so you can use it as a backend for Claude Code with any model.
- **Access cloud models too** — Configure API keys for OpenAI, Anthropic, DeepSeek, or 9 other providers, and access everything through one endpoint.
- **Run it on a LAN** — Two laptops on the same Wi-Fi find each other automatically. No configuration needed.
- **Access remotely via Tailscale** — Connect nodes across the internet with [Tailscale](https://tailscale.com), WireGuard, or any VPN. Chat with your home GPU from anywhere. [Setup guide →](docs/book/src/operations/tailscale-wan.md)

**Who is this for?**

- Developers who want local/private AI without cloud dependencies
- Teams who want to pool their GPUs for larger models
- Researchers who need custom model access with full control
- Anyone who wants to contribute spare compute to a public AI network
- Privacy-conscious users who don't want their prompts leaving their machine

**What makes it different from Petals, Exo, or Bittensor?**

| | SwarmLLM | Others |
|---|---|---|
| **Privacy** | E2E encrypted + optional encrypted pipeline (no remote sees plaintext) | Unencrypted — peers can read your prompts and outputs (Petals); no encryption (Exo) |
| **Install** | Single binary, zero dependencies | Python environments, pip, Docker, blockchain setup |
| **Cloud + Local** | 12 cloud providers as fallback through one API | Local only, no cloud integration |
| **Claude Code** | Full Anthropic Messages API — native Claude Code backend | No Anthropic API support (Exo added basic support recently) |
| **Security** | ~90-fix audit (auth, SSRF, replay, caps) | No documented security audits |

## How It Works

SwarmLLM distributes transformer model layers across a pool of peer-to-peer nodes. Each participant contributes a fraction of the required compute, and the network orchestrates inference pipelines that chain these nodes together. The result: anyone can run state-of-the-art open-weight models by pooling resources with others.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node A  │────▶│  Node B  │────▶│  Node C  │
│ Layers   │     │ Layers   │     │ Layers   │──▶ Response
│  0-15    │     │  16-47   │     │  48-79   │
└──────────┘     └──────────┘     └──────────┘
```

- **Private by default** — all P2P traffic is end-to-end encrypted (X25519 + ChaCha20-Poly1305). Relay nodes never see your data, unlike Petals or Exo where peers have no encryption layer
- **No central server** — fully peer-to-peer with no single point of failure
- **No accounts or subscriptions** — just a cryptographic identity (Ed25519 keypair)
- **Zero configuration** — nodes find each other automatically via mDNS, peer cache, and peer exchange
- **Single binary, zero dependencies** — one ~47MB Rust binary. No Python, no Docker, no runtime installs
- **BitTorrent-inspired incentives** — contribute compute, earn priority access
- **OpenAI + Anthropic + MCP compatible** — drop-in replacement for any tool that speaks OpenAI, Claude, or MCP

## Quick Start

**Download a binary** from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases) for your platform, extract it, and run:

```bash
./swarmllm run
```

Your browser opens to `localhost:8800`. The setup wizard auto-detects your hardware. Pick a model, download it, start chatting.

Available binaries:
| Platform | File | Notes |
|----------|------|-------|
| Linux x86_64 | `swarmllm-linux-x86_64.tar.gz` | CPU inference |
| Linux x86_64 + CUDA | `swarmllm-linux-x86_64-cuda.tar.gz` | NVIDIA GPU acceleration |
| **Windows x86_64** | **`SwarmLLM-Setup.exe`** | **Recommended — installer, auto-detects GPU** |
| Windows x86_64 (GPU) | `swarmllm-windows-x86_64-gpu.zip` | Raw binary: Vulkan + CUDA static |
| Windows x86_64 (CPU) | `swarmllm-windows-x86_64-cpu.zip` | Raw binary: CPU-only fallback |
| macOS Apple Silicon | `swarmllm-macos-aarch64.tar.gz` | CPU inference (Metal planned) |

**Windows users**: just download `SwarmLLM-Setup.exe` and run it. The installer automatically detects your GPU (NVIDIA, AMD, or Intel) and picks the right binary — no CUDA Toolkit or special drivers needed beyond your normal graphics drivers.

See the full [Getting Started Guide](docs/book/src/getting-started.md) for platform-specific instructions.

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

**Or use as a Claude Code backend:**

Point Claude Code directly at your SwarmLLM node — it speaks the full Anthropic Messages API (`/v1/messages`) with tools, thinking, and streaming:

```bash
ANTHROPIC_BASE_URL="http://localhost:8800" \
ANTHROPIC_AUTH_TOKEN="YOUR_API_KEY" \
claude --model "qwen2.5-coder-7b"
```

Claude Code now has access to every model in the swarm — local GGUF, distributed across peers, or any of 12 cloud providers. Use `claude --model gpt-4o` to route through OpenAI, or `claude --model claude-sonnet-4-6` to proxy to Anthropic — all through your SwarmLLM node.

**Or use with MCP (Claude Code / Cursor):**

Add SwarmLLM as an MCP server for tool-based access. The `compare` tool sends the same prompt to multiple models simultaneously:

```json
{
  "mcpServers": {
    "swarmllm": {
      "url": "http://localhost:8800/mcp"
    }
  }
}
```

MCP tools: `chat` (inference), `models` (list available), `compare` (multi-model side-by-side), `research` (fan out to multiple models), `batch_prompts` (batch processing), `delegate` (route to provider), `node_info` (node status).

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
- **Distributed Inference** — Model layers sharded across nodes with automatic pipeline assembly, verified on multi-node LAN deployments with 5 models, crash recovery, auto-reconnect. Candle-based direct tensor computation with E2E encryption
- **Architecture-Aware** — Automatic detection of model architecture (Llama, Llama 4, Qwen2, Qwen 3.5, Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3, GLM-4) with correct RoPE, attention biases, EOS tokens, embedding scaling, logit softcapping, and context lengths from GGUF metadata
- **DeepSeek MoE+MLA** — Full support for DeepSeek-V2/V3 models: Multi-head Latent Attention (low-rank Q/KV compression), Mixture-of-Experts (router-based top-k expert selection with shared experts), per-layer dense/MoE detection
- **Qwen 3.5 Hybrid SSM** — Gated Delta Networks (3 SSM + 1 attention per 4 layers), recurrent state + KV-cache coexistence
- **Cloud Providers** — Route to 12 cloud providers (OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi). API keys can be set via the dashboard, config.toml, environment variables (`OPENAI_API_KEY`, etc.), or a `.env` file in your data directory
- **OpenAI-Compatible API** — `POST /v1/chat/completions` with streaming, tool calling, logprobs, embeddings. Drop-in for Open WebUI, SillyTavern, LangChain, etc.
- **Anthropic Messages API** — `POST /v1/messages` with full Claude Code compatibility: tools, tool_choice, thinking blocks, cache_control, streaming SSE. Use SwarmLLM as a drop-in Claude Code backend (`ANTHROPIC_BASE_URL=http://localhost:8800`). Non-Claude models auto-translated from Anthropic→OpenAI format and routed to cloud providers
- **MCP Server** — Native Model Context Protocol server with 7 tools: `chat`, `models`, `compare` (multi-model side-by-side), `research` (fan out to multiple models), `batch_prompts`, `delegate` (route to specific provider), and `node_info`
- **Prompt Cache Control** — Client-directed KV caching with Anthropic-compatible `cache_control` fields (ephemeral/persistent)
- **Tensor Parallelism** — Automatic tensor-parallel splitting for LAN peers (auto-detected via RTT measurement ≤10ms), with ring-allreduce for 4+ ranks. Complements pipeline parallelism for WAN
- **Vision & Adapters** — VLM support (LLaVA-v1.5-7B verified, Qwen2-VL) with chat UI image upload (camera button, paste, drag-drop), distributed mmproj encoding, and per-request LoRA adapter loading
- **Speculative Decoding** — Draft model + rejection sampling for 2-3x local inference throughput (requires `llama` feature flag)
- **Batched Inference** — True GPU batching: multiple concurrent requests stacked into batch tensors for parallel computation, filling pipeline bubbles in distributed inference
- **Multi-turn KV-cache** — Session-aware cache reuse with pipeline affinity (multi-turn sessions reuse the same nodes for KV-cache locality), cross-request prefix caching, chunked prefill, flash attention (CPU + GPU), VRAM-aware LRU eviction
- **On-Demand Loading** — Models with shards on disk auto-load into VRAM on first inference request, with LRU eviction to make room

### Networking & Security
- **Zero-Config Discovery** — 5-layer stack: mDNS, persistent peer cache, shareable invite codes, peer exchange (PEX), Kademlia DHT
- **P2P Networking** — libp2p with Kademlia DHT, GossipSub (6 topics), TCP+Yamux (primary) and QUIC transport, NAT traversal (auto-relay + DCUtR hole punching), connection limits, gossip replay protection
- **End-to-End Encryption** — Three-tier encryption: pairwise sessions (X25519 + ChaCha20-Poly1305 with forward secrecy via key rotation), pipeline sealing (final segment encrypts output tokens for requester's X25519 key), and authenticated sealed gossip. All peer-to-peer traffic is encrypted in transit. Intermediate pipeline nodes process activation tensors but never see the plaintext output — see [Security Model](docs/book/src/architecture/security.md) for details. By comparison, Petals [explicitly warns](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety) that "peers can recover input data and model outputs" with no encryption layer, and Exo has no encryption at all
- **Encrypted Pipeline** — Optional "boomerang" topology where the requesting node holds both the first shard (embedding) and last shard (token sampling), so **no remote node ever sees plaintext** — only intermediate activation tensors. Per-model toggle via API/dashboard or global config. Auto-enables local embedding privacy. Adds ~1 RTT per token for the return hop. Requires 3+ shard models. See [Encrypted Pipeline](docs/book/src/architecture/security.md#encrypted-pipeline)
- **Security Hardened** — ~90-fix security audit across 5 rounds: authenticated P2P dispatch, signed DHT records, ephemeral key auth, path traversal fix, HF input validation, constant-time auth, CSP hardening, WebSocket Origin check, SSRF protection, resource caps, input limits, credit signature verification, XSS fixes
- **Local Embedding Privacy** — requesting node performs token→embedding locally so remote first-segment nodes never see raw tokens
- **Sybil Resistance** — Ed25519-signed balance reports, peer reputation scoring with trust decay, subnet clustering detection, leaderboard spoofing protection
- **API Authentication** — Bearer token middleware with auto-generated keys, CORS lockdown, SSRF protection, Content-Security-Policy, IP-based rate limiting

### Economy & Identity
- **Credit System** — Earn credits by serving inference, forwarding activations, hosting shards (hourly), seeding data, and relaying traffic. Priority tiers enforced per-request (Platinum/Gold/Silver/Bronze) with concurrent limits. Anti-gaming protection, failure penalties, transaction replay prevention, credit escrow with automatic refund on failure
- **Identity & Pools** — Cryptographic nicknames, leaderboard, multi-device credit pooling with dual-signature invitation protocol
- **Model Trust** — Demand-driven trust system (Discovered→Pinned→DemandVerified→NetworkPopular) gates auto-manage to prevent trash model propagation
- **Auto-Shard Management** — VRAM-aware automatic shard acquisition from HuggingFace (with resume, retry, and Range headers) and peers with popularity/rarity scoring. Smart pruning auto-removes over-replicated shards based on demand, resource pressure, and region diversity

### Operations
- **Built-in Web UI** — Swarm-first dashboard with chat interface (image upload for VLM), model browser, shard visualization, first-run setup wizard, network map, leaderboard, model compare page, mobile-responsive layout. 21 languages (i18n), light/dark/system theme
- **Fault Tolerant** — JoinSet-based task supervisor with restart-on-crash for all 11 subsystems, hot-standby failover, shard replication, automatic rebalancing, atomic shard writes, download retry with backoff
- **Observability** — Prometheus `/metrics` endpoint, readiness probe `/health/ready`, structured tracing with request ID correlation, database integrity checks
- **Config Hot-Reload** — Change operational parameters without restarting via SIGHUP or API (`/api/admin/config/reload`)
- **Auto-Updater** — Checks GitHub releases for new versions, downloads and replaces binary with restart prompt
- **Packaging** — Homebrew formula, AUR PKGBUILD, deb/rpm packages, systemd service file, Docker (CPU + CUDA + docker-compose cluster)
- **Multi-SDK Ecosystem** — Python (`pip install swarmllm-client`), JavaScript/TypeScript (zero-dep), LangChain (`ChatSwarmLLM`), LlamaIndex (`SwarmLLM`)

## Architecture

Single Rust binary (~47MB release), three simultaneous functions:

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| P2P Node | Peer discovery, shard hosting, distributed inference, credits | libp2p / TCP+QUIC |
| LLM API Server | OpenAI + Anthropic + MCP inference endpoints | `localhost:8800/v1/*` |
| Management UI | Dashboard, settings, model browser, chat | `localhost:8800/admin` |

Internally, the daemon runs 11 async Tokio tasks communicating via channels:

```
┌──────────────────────────────────────────────────────────────────────┐
│                           daemon/                                     │
│                                                                      │
│  NetworkManager ──── InferenceRouter ──── CreditLedger               │
│       │                    │                   │                      │
│  MessageDispatcher    ApiServer          HealthMonitor                │
│       │                    │                   │                      │
│  PoolManager        AutoShardManager    ShardRebalancer               │
│       │                    │                   │                      │
│  AcquisitionManager    UpdateChecker                                 │
│       └────────────────────┴───────────────────┘                     │
│                                                                      │
│              All connected via mpsc channels                         │
│              Shared state via Arc<SharedState> + DashMap              │
└──────────────────────────────────────────────────────────────────────┘
```

### Codebase

Cargo workspace with 3 crates, 107 Rust source files (~73K lines), plus vanilla frontend (~13K lines HTML/CSS/JS):

| Crate | Path | Purpose |
|-------|------|---------|
| `swarmllm` | `/` (root) | Main binary — daemon, networking, inference, API, all subsystems |
| `swarmllm-types` | `crates/swarmllm-types/` | Shared data types (75 types: NodeId, ModelManifest, SwarmMessage, etc.) |
| `swarmllm-frontend` | `crates/swarmllm-frontend/` | Frontend asset serving (embedded in release, disk-based in dev mode) |

Key source directories:
- `src/daemon/` — startup, shared state, message dispatch, shard loading
- `src/network/` — libp2p networking, peer discovery, transport, relay, peer cache
- `src/inference/` — router, pipeline, split inference, allreduce, KV-cache, speculative decoding, vision, chat templates
- `src/api/` — Axum HTTP server, OpenAI/Anthropic/MCP endpoints, admin API, WebSocket, middleware
- `src/model/` — manifests, shards, acquisition, auto-manage, HuggingFace integration, LoRA
- `src/credit/` — ledger, transactions, priority tiers, anti-gaming, trust, escrow
- `src/crypto/` — session encryption, pipeline sealing, gossip sealing, key rotation, provider key encryption
- `src/pool/` — device pool management, crypto, credit forwarding
- `frontend/` — vanilla HTML/CSS/JS dashboard (13 component JS files, 13 HTML templates, 21 language translations)

649 tests (581 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E), all passing, clippy clean.

## Node Tiers

| Tier | Requirements | Role |
|------|-------------|------|
| Super Node | Full model in VRAM, high bandwidth | Serves full inference independently |
| Standard Node | Partial VRAM/RAM, moderate bandwidth | Holds layer shards, joins inference pipelines |
| Light Node | Minimal resources | Primarily consumer, contributes bandwidth |

## Credit System

| Action | Effect | Status |
|--------|--------|--------|
| Serve inference | +credits (per layer per token) | Active |
| Forward activations | +credits (per layer processed) | Active |
| Host model shards | +credits (per GB per hour) | Active |
| Seed shard data | +credits (per GB transferred) | Active |
| Relay traffic | +credits (per connection hour) | Active |
| Submit inference request | -credits (per layer per token) | Active |
| Distributed failure | -credits (penalty) | Active |

Credits determine your priority tier:

- **Platinum** (top 10%) — near-instant responses
- **Gold** (top 30%) — 1-3 second queue
- **Silver** (positive balance) — 5-15 second queue
- **Bronze** (zero/negative) — 30+ second queue, but never locked out

## Supported Models

SwarmLLM supports 11 transformer architectures via native candle inference with GGUF quantization:

| Architecture | Examples | Special Features |
|-------------|----------|-----------------|
| **Llama** | Llama 2/3, CodeLlama, TinyLlama | Interleaved RoPE, GQA |
| **Llama 4** | Llama 4 Scout (17B), Maverick (400B) | iRoPE (NoPE every 4th layer), MoE |
| **Qwen2** | Qwen2.5-Coder-7B/32B | QKV biases, 32k context |
| **Qwen 3.5** | Qwen3.5-3B/14B/32B | Hybrid SSM+attention (Gated Delta Networks) |
| **DeepSeek-V2/V3** | DeepSeek-V2-Lite, DeepSeek-V3 (671B) | MLA attention + MoE FFN |
| **GLM-4** | GLM-4-9B, GLM-4.7 MoE | Partial RoPE, extreme GQA (16:1) |
| **Gemma/Gemma2** | Gemma 2B/7B, Gemma2 9B/27B | Gemma RmsNorm (+1), embedding scaling, logit softcapping |
| **Phi-3** | Phi-3-mini, Phi-3-medium | Su/YaRN RoPE, fused QKV/FFN |
| **Mistral** | Mistral 7B, Mistral Nemo | GQA, interleaved RoPE |
| **Starcoder2** | Starcoder2 3B/7B/15B | Code-optimized, biases |
| **Mixtral** | Mixtral 8x7B, 8x22B | MoE (via llama.cpp) |

Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16

## Benchmarks

Measured on a single node with `swarmllm bench`. Prompt: "Explain the theory of relativity in simple terms." 100 output tokens, average of 3 runs.

**Hardware:** AMD Ryzen 7 5800H (8C/16T), NVIDIA RTX 3070 Laptop (8GB VRAM), WSL2

| Model | Params | Quant | GPU (RTX 3070) | CPU Only | GPU Speedup |
|-------|--------|-------|----------------|----------|-------------|
| TinyLlama 1.1B | 1.1B | Q4_K_M | **27.2 tok/s** | 4.2 tok/s | 6.5x |
| Gemma-2 2B IT | 2.5B | Q4_K_M | **20.6 tok/s** | 3.5 tok/s | 5.9x |
| Phi-3.5 Mini | 3.8B | Q4_K_M | **46.4 tok/s** | 1.8 tok/s | 25.8x |
| Qwen2.5-Coder 7B | 7.6B | Q4_K_M | **29.0 tok/s** | 2.4 tok/s | 12.1x |

**Distributed (2-node LAN, TinyLlama):** ~29 tok/s with split inference across WSL2 laptop + Proxmox server.

> GPU inference uses candle with CUDA (`--features candle-cuda`). CPU inference uses candle with native BLAS. All models Q4_K_M quantized, loaded from GGUF shard files. Phi-3.5 benefits most from GPU due to its fused QKV/FFN architecture.

Run your own benchmarks:
```bash
swarmllm bench --max-tokens 100 --iterations 5 --concurrency 4 --json
```

## Installation

### Pre-built Binaries (Recommended)

Download from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases) — available for Linux (CPU and CUDA) and Windows. Extract and run `./swarmllm run`.

### Package Managers

```bash
# Homebrew (macOS/Linux)
brew tap enapt/swarmllm
brew install swarmllm

# AUR (Arch Linux)
yay -S swarmllm

# Debian/Ubuntu
sudo dpkg -i swarmllm_0.1.0_amd64.deb

# RPM (Fedora/RHEL)
sudo rpm -i swarmllm-0.1.0-1.x86_64.rpm
```

### Docker

```bash
# Pre-built images (recommended)
docker run -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest

# GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest-cuda

# Or use docker compose (see docker-compose.yml)
cp .env.example .env && docker compose up -d

# 3-node dev cluster
docker compose -f docker-compose.dev.yml up
```

### Cargo Install

```bash
# Requires Rust 1.80+
cargo install --git https://github.com/enapt/SwarmLLM.git
swarmllm run
```

### Building from Source

```bash
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM

# CPU-only build (candle inference engine)
cargo build --release

# With CUDA GPU acceleration (candle + flash attention)
cargo build --release --features candle-cuda

# With llama.cpp backend (optional, requires cmake + libclang)
cargo build --release --features llama

# Full CUDA build (candle + llama.cpp + flash attention)
cargo build --release --features cuda

# Windows GPU build: llama.cpp via Vulkan (all vendors) + candle CUDA static runtime
# Users need only standard GPU drivers — no CUDA Toolkit installation required
cargo build --release --features windows-gpu

# llama.cpp with Vulkan only (cross-platform local inference on NVIDIA/AMD/Intel)
cargo build --release --features llama-vulkan

# Apple Silicon (CPU — Metal via llama.cpp planned)
cargo build --release
```

## CLI

```
swarmllm <COMMAND>

Commands:
  run         Start the SwarmLLM daemon (default if omitted)
  status      Show node status (queries running daemon)
  chat        Interactive terminal chat with a running daemon
  bench       Run inference benchmarks (tokens/sec, latency)
  peers       List connected peers with latency and trust scores
  pool        Device pool management (combine credits across devices)
  test-split  Test split inference locally (single-node diagnostic)
  update      Check for and download updates
  version     Print version information

Options:
  -c, --config <PATH>       Config file path
  -p, --port <PORT>         Listen port [default: 8800]
  -d, --data-dir <PATH>     Data directory [default: ~/.local/share/swarmllm]
  -m, --model <PATH>        Path to a GGUF model file to load
      --gpu-layers <N>      Number of layers to offload to GPU (0 = CPU only)
      --bootstrap <ADDR>    Bootstrap peer multiaddr
  -v, --verbose             Increase log verbosity (-v, -vv, -vvv)
```

## Configuration

Config lives at `~/.local/share/swarmllm/config.toml` (Linux), `~/Library/Application Support/swarmllm/config.toml` (macOS), or `%APPDATA%\swarmllm\config.toml` (Windows). All values can be overridden with environment variables using the `SWARMLLM_` prefix:

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_LOGGING_LEVEL=debug
```

### .env File Support

Provider API keys can be loaded from a `.env` file placed in your data directory or the current working directory:

```bash
# ~/.local/share/swarmllm/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
NVIDIA_NIM_API_KEY=nvapi-...
```

The `.env` file is loaded at startup and does not override existing environment variables or keys already configured via the dashboard/database.

### Key Config Sections

| Section | Key Settings |
|---------|-------------|
| `[node]` | `listen_port`, `contribution` (minimal/moderate/maximum), `data_dir` |
| `[resources]` | `max_gpu_vram_mb`, `max_ram_mb`, `max_disk_mb`, `max_bandwidth_mbps` |
| `[resources.schedule]` | `enabled`, `reduced_hours_start/end`, `prune_aggressiveness` |
| `[network]` | `bootstrap_peers`, `enable_mdns`, `enable_encryption`, `gossip_network_id`, `enable_relay`, `max_peers`, `tensor_compression` |
| `[inference]` | `model_path`, `gpu_layers`, `session_timeout_seconds`, `max_batch_size`, `speculative_decoding`, `tp_max_latency_ms`, `local_embedding_privacy`, `encrypted_pipeline` |
| `[api]` | `api_key`, `rate_limit_rpm` |
| `[pool]` | `max_pool_size`, `invitation_ttl_hours`, `rate_limit_per_hour` |
| `[auto_manage]` | `enabled`, `max_storage_mb`, `interval_minutes`, `max_concurrent_downloads`, `prune_enabled`, `min_replicas` |
| `[providers]` | API keys for cloud providers (also via `OPENAI_API_KEY` env var / `.env` file), custom providers |
| `[logging]` | `level`, `format` (pretty/json), `file` |
| `[ui]` | `open_browser_on_start`, `theme` |
| `[updates]` | `auto_update` (disabled/stable/all), `check_interval_hours` |
| `[identity]` | `region` (country code for network map) |

See the [Configuration Reference](docs/book/src/configuration/reference.md) for the full list.

## API Endpoints

### Inference (Bearer auth required)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming + non-streaming) |
| POST | `/v1/messages` | Anthropic Messages API (full Claude Code compatibility) |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List available models |
| GET | `/v1/providers` | List configured cloud providers |
| GET | `/v1/status` | Node status |

### MCP Server
| Method | Path | Description |
|--------|------|-------------|
| POST | `/mcp` | JSON-RPC 2.0 MCP endpoint (7 tools: chat, models, compare, research, batch_prompts, delegate, node_info) |

### Admin (CORS-protected)
| Method | Path | Description |
|--------|------|-------------|
| GET/PUT | `/api/admin/config` | Configuration read/update |
| POST | `/api/admin/config/reload` | Hot-reload config |
| GET | `/api/admin/stats` | Node statistics + hardware info |
| GET | `/api/admin/models` | Model list with shard status |
| GET | `/api/admin/peers` | Connected peers with latency/trust |
| GET | `/api/admin/credits` | Credit balance and tier info |
| GET | `/api/admin/ws` | WebSocket for live updates |
| GET | `/api/admin/hf/search` | Search HuggingFace for GGUF models |
| POST | `/api/admin/hf/download-shards` | Download specific shards |
| GET/PUT | `/api/admin/models/:id/encrypted-pipeline` | Per-model encrypted pipeline toggle |
| POST | `/api/admin/shutdown` | Graceful shutdown (localhost only) |

Plus ~50 more admin routes for downloads, providers, adapters, identity, pools, scheduling, and more. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete list.

### Monitoring
| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus/OpenMetrics endpoint |
| GET | `/health` | Health check |
| GET | `/health/ready` | Readiness probe with subsystem status |

## Platform Support

| Platform | GPU Support | Status |
|----------|------------|--------|
| Linux x86_64 | CUDA (candle + llama.cpp) | Primary target, release binaries available |
| Windows x86_64 | **Vulkan** (NVIDIA/AMD/Intel local) + **CUDA static** (NVIDIA distributed) | Installer — no CUDA Toolkit needed |
| macOS aarch64 | CPU only (Metal via llama.cpp planned) | Binary available (beta) |
| macOS x86_64 | CPU only | Best-effort |
| Linux aarch64 | CPU only | Best-effort |

**Windows GPU detection** is automatic — the installer bundles GPU and CPU binaries and a launcher that picks the right one at startup:
- **NVIDIA GPU**: GPU-accelerated local inference (Vulkan) + GPU-accelerated distributed inference (CUDA, static runtime)
- **AMD / Intel GPU**: GPU-accelerated local inference (Vulkan), CPU distributed inference
- **No GPU**: CPU for everything (still participates fully in the network)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Rust (2021 edition), ~73K lines across 107 source files |
| Async Runtime | Tokio (multi-threaded) |
| Networking | libp2p 0.55 (TCP+Yamux + QUIC + mDNS + Kademlia + GossipSub) |
| Serialization | serde_json (API), binary with type-tag (tensors), zstd compression |
| HTTP Server | Axum 0.7 with WebSocket |
| Inference | candle (split/distributed, CUDA, flash attention), llama.cpp (single-node, optional) |
| Database | redb v3 (embedded, ACID) |
| Cryptography | Ed25519 (identity), X25519 + ChaCha20-Poly1305 (E2E), BLAKE3 (integrity) |
| Monitoring | Prometheus + Grafana (dashboards included) |
| Frontend | Vanilla HTML/CSS/JS, 21 languages, light/dark/system theme, ~13K lines |
| SDK | Python, JS/TS, LangChain, LlamaIndex |

## How SwarmLLM Compares

| Feature | SwarmLLM | Petals | Exo | Bittensor |
|---------|----------|--------|-----|-----------|
| **Language** | Rust (single ~47MB binary) | Python | Python | Python + Substrate |
| **Install** | Download & run | pip install | pip/source/macOS app | pip + blockchain setup |
| **Scale** | LAN + WAN + Tailscale/WireGuard (zero config) | Internet (volunteer) | LAN + Tailscale (manual) | Internet (blockchain) |
| **E2E Encryption** | **X25519 + ChaCha20 + forward secrecy** | **None** — peers can see your prompts | **None** | Minimal (blockchain-level) |
| **Privacy** | **Encrypted by default** — all traffic encrypted in transit. Optional **encrypted pipeline** ensures no remote node sees plaintext (boomerang topology) | Unencrypted — Petals' own wiki states peers can read your prompts and outputs | No encryption between nodes | Subnet-dependent |
| **Security Audit** | **~90-fix, 5-round hardening** (auth, SSRF, replay, caps) | None documented | None documented | PoA consensus (centralized) |
| **Incentives** | Credit tiers (no token, no blockchain) | Name on monitor page | None | TAO token (real money) |
| **Parallelism** | Pipeline + tensor (auto-detected LAN) | Pipeline | Tensor + pipeline | Subnet routing |
| **Model Architectures** | **11** (DeepSeek MoE+MLA, GLM-4, Llama 4, Qwen 3.5 SSM) | ~5 (Llama, Mixtral, Falcon, BLOOM) | ~5 (Llama, Mistral, Qwen, DeepSeek, LLaVA) | Any (subnet-defined) |
| **Shard-Only Mode** | **Yes** (no full model download needed) | No (loads full blocks) | No | N/A |
| **Cloud Fallback** | **12 providers** (OpenAI, Anthropic, DeepSeek, etc.) | No | No | No |
| **VLM + LoRA** | Both (LLaVA verified + per-request LoRA) | LoRA only | VLM experimental | Subnet-specific |
| **API Compatibility** | **OpenAI + Anthropic + MCP** (full Claude Code) | PyTorch/Transformers | OpenAI + Claude + Ollama | Subnet-defined |
| **Web UI** | Full dashboard, chat, model browser, setup wizard | Basic chatbot | Basic chat UI | No built-in UI |
| **SDKs** | Python + JS/TS + LangChain + LlamaIndex | Python native | — | Python |
| **i18n** | **21 languages**, light/dark theme | English | English | English |
| **Auto-Update** | Built-in self-update | No | No | No |
| **Maintained** | **Active** (2026) | Last release Sep 2023 | **Active** (2025) | **Active** (2025) |

**Why SwarmLLM?** If privacy matters to you, SwarmLLM is the only option with real E2E encryption — all peer-to-peer traffic is encrypted with forward secrecy, pipeline sealing ensures output tokens are encrypted for the requester, and the optional encrypted pipeline mode guarantees no remote node ever sees plaintext (prompt or response). It's also the only one that works as a drop-in backend for Claude Code, supports 12 cloud providers as fallback, and runs as a single binary with zero dependencies.

## Documentation

- **[Getting Started](docs/book/src/getting-started.md)** — Download, install, and start chatting in minutes
- **[Configuration Reference](docs/book/src/configuration/reference.md)** — All config options with defaults
- **[Configuration Guide](docs/book/src/configuration.md)** — Environment variables, CLI flags, `.env` files
- **[API Reference](docs/ARCHITECTURE.md#http-api-routes)** — Complete HTTP API route documentation
- **[Architecture](docs/ARCHITECTURE.md)** — Deep dive into subsystems, protocols, and security model
- **[Tailscale & WAN Access](docs/book/src/operations/tailscale-wan.md)** — Access your node remotely via Tailscale, WireGuard, or any VPN
- **[Troubleshooting](docs/book/src/troubleshooting.md)** — Common issues and solutions
- **[Diagnostics Guide](docs/DIAGNOSTICS.md)** — DIAG: log instrumentation for debugging
- **[Security Policy](SECURITY.md)** — Responsible disclosure

See the full [mdBook documentation](docs/book/) for detailed guides on networking, inference, credits, security, deployment, and monitoring.

## Development Transparency

SwarmLLM was developed collaboratively between a human developer and Claude (Anthropic's AI). The entire codebase — Rust backend, JavaScript frontend, P2P networking, distributed inference pipeline, credit system, security hardening, and documentation — was written by Claude Code across 20+ build phases. The human developer provided architecture direction, testing, and review, but zero lines of code were manually written.

This is an honest disclosure. The project has been through rigorous quality assurance — 649 passing tests, continuous security auditing, dozens of parallel multi-agent code sweeps, and multi-node distributed inference tested on real networks. Every commit passes `cargo fmt`, `cargo clippy -- -D warnings`, and the full test suite before push.

We believe AI-assisted development should be transparent. Judge the code on its technical merit — contributions, scrutiny, and feedback are all welcome.

## Contributing

Contributions are welcome! Whether it's bug reports, feature ideas, code, or documentation — all help is appreciated.

- **[Contributing Guide](CONTRIBUTING.md)** — Build from source, run tests, submit PRs
- **[Open Issues](https://github.com/enapt/SwarmLLM/issues)** — Browse or file bug reports and feature requests
- **Security vulnerabilities** — See [SECURITY.md](SECURITY.md) for responsible disclosure

```bash
# Quick dev setup
git clone https://github.com/enapt/SwarmLLM.git && cd SwarmLLM
cargo test                # 649 tests
cargo clippy -- -D warnings  # Zero warnings policy
cargo run -- run          # Start a node
```

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
