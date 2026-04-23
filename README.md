# SwarmLLM

[![CI](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust 1.80+](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/enapt/swarmllm)
[![Release](https://img.shields.io/github/v/release/enapt/SwarmLLM?include_prereleases&label=release)](https://github.com/enapt/SwarmLLM/releases)

Decentralized peer-to-peer LLM inference network. A single Rust binary that shards large language models across a network of contributing nodes, enabling access to 70B+ parameter models without expensive hardware or paid API tokens.

**Join the swarm. Run AI together — for free.**

> **Status:** Alpha — actively developed, moving into broader testing. Distributed inference stable and tested on multi-node deployments. Most recent headline: cross-node prefix-KV sharing delivers a **12.9× iter-1 TTFT speedup** on 7B prompts when a peer has the same prefix cached (Round 6 bench, 2026-04-20); Windows release binaries validated at Linux parity (Round 8, 2026-04-23). 781 tests, continuous security sweeps. [Report issues](https://github.com/enapt/SwarmLLM/issues).

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [What Is This?](#what-is-this)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Connecting to the Network](#connecting-to-the-network)
- [Private Mode](#private-mode)
- [Features](#features)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Installation](#installation)
- [CLI](#cli)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Platform Support](#platform-support)
- [How SwarmLLM Compares](#how-swarmllm-compares)
- [Documentation](#documentation)
- [Support](#support)
- [Development Transparency](#development-transparency)
- [Contributing](#contributing)
- [License](#license)

</details>

## What Is This?

SwarmLLM lets you run AI chatbots (like ChatGPT, but open-source) on your own computer — or share the work with others across the internet. Think of it like BitTorrent, but for AI: instead of downloading movies, you're sharing the computing power needed to run large language models.

Running a smart AI model (like Llama 3 70B) normally requires a $10,000+ GPU. With SwarmLLM, the model gets split into pieces — your computer handles some layers, your friend's handles others, and together you can run models none of you could run alone. No cloud subscription, no API fees, and all peer-to-peer traffic is encrypted end-to-end.

**What can you do with it?**

- **Chat with AI** — Open `localhost:8800` in your browser, pick a model, start chatting.
- **Use it as an API** — Any tool that speaks OpenAI (LangChain, Open WebUI, SillyTavern, etc.) works with SwarmLLM.
- **Use it with Claude Code** — SwarmLLM speaks the Anthropic Messages API natively. Full tool use, thinking, and streaming.
- **Route to cloud too** — Configure keys for 12 cloud providers and reach them through one endpoint.
- **LAN or WAN** — Two laptops on the same Wi-Fi find each other automatically. For remote access, [Tailscale](docs/book/src/operations/tailscale-wan.md) works out of the box.

**Who is this for?**

- Developers who want local/private AI without cloud dependencies
- Teams who want to pool their GPUs for larger models
- Researchers who need custom model access with full control
- Anyone who wants to contribute spare compute to a public AI network
- Privacy-conscious users who don't want their prompts leaving their machine

## How It Works

SwarmLLM distributes transformer model layers across a pool of peer-to-peer nodes. Each participant contributes a fraction of the required compute, and the network orchestrates inference pipelines that chain these nodes together.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node A  │────▶│  Node B  │────▶│  Node C  │
│ Layers   │     │ Layers   │     │ Layers   │──▶ Response
│  0-15    │     │  16-47   │     │  48-79   │
└──────────┘     └──────────┘     └──────────┘
```

**Key properties**

- **Encrypted by default** — X25519 + ChaCha20-Poly1305 with forward secrecy on all P2P traffic. Relay nodes can't read your prompts or outputs. Optional boomerang pipeline ensures no remote node ever sees plaintext.
- **No central server** — fully peer-to-peer, no single point of failure, no accounts.
- **Zero configuration** — nodes find each other via mDNS, peer cache, invite codes, peer exchange, and Kademlia DHT.
- **Single binary** — one Rust binary (~33–50 MB depending on platform and features). No Python, no Docker, no runtime installs.
- **BitTorrent-inspired incentives** — contribute compute, earn priority access.
- **OpenAI + Anthropic + MCP compatible** — drop-in replacement for any tool that speaks OpenAI, Claude, or MCP.
- **Shard-only nodes** — a node never needs the full model to participate; shards download individually via byte-range requests.

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

**Windows users**: download `SwarmLLM-Setup.exe` and run it. The installer detects your GPU (NVIDIA, AMD, or Intel) and picks the right binary — no CUDA Toolkit or special drivers beyond normal graphics drivers.

See the [Getting Started Guide](docs/book/src/getting-started.md) for platform-specific instructions.

**Use the API directly:**

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

**As a Claude Code backend** — full Anthropic Messages API with tools, thinking, and streaming:

```bash
ANTHROPIC_BASE_URL="http://localhost:8800" \
ANTHROPIC_AUTH_TOKEN="YOUR_API_KEY" \
claude --model "qwen2.5-coder-7b"
```

Claude Code gets access to every model in the swarm — local GGUF, distributed across peers, or any of 12 cloud providers. Use `claude --model gpt-4o` to route through OpenAI, or `claude --model claude-sonnet-4-6` to proxy to Anthropic.

**As an MCP server** — add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "swarmllm": {
      "url": "http://localhost:8800/mcp"
    }
  }
}
```

MCP tools: `chat`, `models`, `compare` (multi-model side-by-side), `research` (fan out to multiple models), `batch_prompts`, `delegate` (route to provider), `node_info`.

## Connecting to the Network

SwarmLLM uses a 5-layer discovery stack — no manual configuration needed:

| Layer | How it works | When it kicks in |
|-------|-------------|------------------|
| **mDNS** | Automatically discovers peers on the same LAN/Wi-Fi | Instantly on startup |
| **Peer Cache** | Remembers peers from previous sessions (redb-backed, max 200) | On restart |
| **Invite Codes** | Share a `swarm://...` code with a friend to connect directly | First time joining |
| **Peer Exchange** | Connected peers share their known peer lists with you | On each new connection |
| **Kademlia DHT** | Distributed hash table for network-wide peer routing | Continuously |

**First launch on a LAN:** two nodes on the same network find each other in seconds — zero config.

**First launch alone:** the dashboard shows your invite code. Share it with a friend. Once connected to one peer, PEX and DHT discover the rest of the network automatically.

**Returning user:** cached peers reconnect in under a second. The invite code UI auto-hides once your node knows 20+ peers.

> For private networks, set `gossip_network_id = "my-private-net"` in config to isolate your nodes from the public network.

## Private Mode

For maximum privacy, enable **Private Mode** to restrict all your outbound inference to your device pool. Your prompts never leave your machines. Toggle via the dashboard shield icon or the API; a confirmation dialog shows your pool's model coverage before activating.

```bash
curl -X PUT http://localhost:8800/api/pool/private-mode \
  -H "Authorization: Bearer $KEY" \
  -d '{"enabled": true}'
```

| Mode | Config | Behavior |
|------|--------|----------|
| **Pool Only** | `private_mode = true` | Inference restricted to pool members |
| **Pool + LAN** | `private_mode_allow_lan = true` (default) | Pool + mDNS-discovered LAN peers |
| **Offline** | `offline_mode = true` | Air-gapped: no internet, mDNS only |

Private mode is one-way: your data stays private, but your nodes still serve the swarm (processing inference for others, hosting shards, earning credits). **Shard Pinning** lets pool owners assign specific models to specific devices — auto-manage downloads pinned shards with highest priority and never prunes them. The **Coverage Dashboard** shows per-model availability with color-coded bars and estimated download sizes to fill gaps.

## Features

### Inference
- **Distributed pipelines** — layers sharded across nodes with automatic pipeline assembly, crash recovery, and auto-reconnect. Candle-based direct tensor computation with E2E encryption.
- **Speedup stack (default-on)** — remote-generate fast path (1.93× decode on single-segment pipelines), cross-request prefix cache (29.4× wall-clock on prompt re-submission), cross-node prefix-KV sharing (12.9× iter-1 TTFT on 7B CPU prompts when a peer has the same prefix cached), continuous batching (1.34–1.55× GPU throughput at batch 2–8), Sarathi chunked prefill + batched fusion (17–23× TTFT fairness at concurrency), Parallax scheduler (shortest-path DP over observed per-layer latencies). See [Performance & Inference Speedups](docs/book/src/operations/performance.md) for the full stack.
- **Flag-gated speedups** — distributed speculative decoding (`speculative_distributed`), SWIFT self-speculative (`swift_self_speculative`), DSD multi-segment speculation (`decentralized_spec_decoding`), Q8_0 activation compression (`activation_compression`, ~3.76× wire).
- **Tensor parallelism** — automatic TP splitting for LAN peers (RTT ≤10ms), ring-allreduce for 4+ ranks. Complements pipeline parallelism for WAN.
- **Vision & LoRA** — VLM support (LLaVA-v1.5-7B verified, Qwen2-VL) with distributed mmproj encoding and per-request LoRA adapter loading.
- **KV-cache reuse** — session-aware cache with pipeline affinity, cross-request prefix caching, chunked prefill, flash attention (CPU + GPU), VRAM-aware LRU eviction.
- **On-demand loading** — models with shards on disk auto-load into VRAM on first request, with LRU eviction to make room.

### APIs
- **OpenAI-compatible** — `POST /v1/chat/completions` with streaming, tool calling, logprobs, embeddings.
- **Anthropic Messages API** — `POST /v1/messages` with full Claude Code compatibility: tools, tool_choice, thinking blocks, `cache_control`, streaming SSE. Non-Claude models auto-translated from Anthropic→OpenAI format and routed to cloud providers.
- **MCP server** — native Model Context Protocol with 7 tools: `chat`, `models`, `compare`, `research`, `batch_prompts`, `delegate`, `node_info`.
- **Cloud fallback** — route to 12 cloud providers (OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi). Keys via dashboard, config, env vars, or `.env` file.
- **Prompt cache control** — client-directed KV caching with Anthropic-compatible `cache_control` fields.

### Networking & Security
- **libp2p transport** — Kademlia DHT, GossipSub (6 topics), TCP+Yamux (primary) and QUIC, NAT traversal (auto-relay + DCUtR hole punching), connection limits, gossip replay protection.
- **Three-tier encryption** — pairwise sessions (X25519 + ChaCha20-Poly1305 with forward secrecy via key rotation), pipeline sealing (final segment encrypts output tokens for the requester's key), authenticated sealed gossip. Intermediate pipeline nodes process activation tensors but never see plaintext output. See [Security Model](docs/book/src/architecture/security.md).
- **Encrypted pipeline (optional)** — "boomerang" topology where the requesting node holds both first (embedding) and last (sampling) shards, so no remote node ever sees plaintext. Adds ~1 RTT per token for the return hop. Requires 3+ shard models.
- **Local embedding privacy** — the requesting node performs token→embedding locally so remote first-segment nodes never see raw tokens.
- **Sybil resistance** — Ed25519-signed balance reports, peer reputation with trust decay, subnet clustering detection, leaderboard spoofing protection.
- **API auth** — Bearer token middleware with auto-generated keys, CORS lockdown, SSRF protection, CSP headers, IP-based rate limiting.

### Economy & Identity
- **Credits** — earn by serving inference, forwarding activations, hosting shards, seeding data, and relaying. Priority tiers (Platinum/Gold/Silver/Bronze) enforced per-request with concurrent limits. Anti-gaming, failure penalties, transaction replay prevention, credit escrow with automatic refund on failure.
- **Pools** — cryptographic nicknames, leaderboard, multi-device credit pooling with dual-signature invitation protocol.
- **Model trust** — demand-driven states (Discovered→Pinned→DemandVerified→NetworkPopular) gate auto-manage to prevent trash model propagation.
- **Auto-shard management** — VRAM-aware shard acquisition from HuggingFace (resume, retry, Range headers) and peers with popularity/rarity scoring. Smart pruning auto-removes over-replicated shards based on demand, resource pressure, and region diversity.

### Operations
- **Web UI** — swarm-first dashboard with chat (image upload for VLM), model browser, shard visualization, first-run setup wizard, network map, leaderboard, compare page, mobile-responsive. 21 languages, light/dark/system theme.
- **Fault tolerance** — JoinSet-based task supervisor with restart-on-crash for all 11 subsystems, hot-standby failover, shard replication, automatic rebalancing, atomic shard writes.
- **Observability** — Prometheus `/metrics`, readiness probe `/health/ready`, structured tracing with request-ID correlation, database integrity checks.
- **Config hot-reload** — change operational parameters without restarting via SIGHUP or `/api/admin/config/reload`.
- **Auto-updater** — checks GitHub releases, downloads and replaces binary with restart prompt.
- **Packaging** — Homebrew, AUR, deb/rpm, systemd, Docker (CPU + CUDA + docker-compose cluster).
- **SDKs** — Python (`pip install swarmllm-client`), JavaScript/TypeScript (zero-dep), LangChain (`ChatSwarmLLM`), LlamaIndex (`SwarmLLM`).

## Supported Models

12 transformer architectures via native candle inference with GGUF quantization:

| Architecture | Examples | Special Features |
|-------------|----------|-----------------|
| **Llama** | Llama 2/3, CodeLlama, TinyLlama | Interleaved RoPE, GQA |
| **Llama 4** | Llama 4 Scout (17B), Maverick (400B) | iRoPE (NoPE every 4th layer), MoE |
| **Qwen2** | Qwen2.5-Coder-7B/32B | QKV biases, 32k context |
| **Qwen 3.5** | Qwen3.5-3B/14B/32B (incl. MoE) | Hybrid SSM+attention (Gated Delta Networks) |
| **DeepSeek-V2/V3** | DeepSeek-V2-Lite, DeepSeek-V3 (671B) | MLA attention + MoE FFN |
| **GLM-4** | GLM-4-9B, GLM-4.7 MoE | Partial RoPE, extreme GQA (16:1) |
| **Gemma / Gemma2** | Gemma 2B/7B, Gemma2 9B/27B | Gemma RmsNorm (+1), embedding scaling, logit softcapping |
| **Phi-3** | Phi-3-mini, Phi-3-medium | Su/YaRN RoPE, fused QKV/FFN |
| **Mistral** | Mistral 7B, Mistral Nemo | GQA, interleaved RoPE |
| **Starcoder2** | Starcoder2 3B/7B/15B | Code-optimized, biases |
| **Mixtral** | Mixtral 8x7B, 8x22B | MoE (via llama.cpp backend) |

Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16. Context length, RoPE type, attention biases, EOS tokens, and embedding scaling are all detected from GGUF metadata.

## Benchmarks

Single-node, `swarmllm bench`. Prompt: *"Explain the theory of relativity in simple terms."* 100 output tokens, average of 3 runs. **Hardware:** AMD Ryzen 7 5800H (8C/16T), NVIDIA RTX 3070 Laptop (8GB VRAM), WSL2.

| Model | Params | Quant | GPU (RTX 3070) | CPU Only | GPU Speedup |
|-------|--------|-------|----------------|----------|-------------|
| TinyLlama 1.1B | 1.1B | Q4_K_M | **27.2 tok/s** | 4.2 tok/s | 6.5× |
| Gemma-2 2B IT | 2.5B | Q4_K_M | **20.6 tok/s** | 3.5 tok/s | 5.9× |
| Phi-3.5 Mini | 3.8B | Q4_K_M | **46.4 tok/s** | 1.8 tok/s | 25.8× |
| Qwen2.5-Coder 7B | 7.6B | Q4_K_M | **29.0 tok/s** | 2.4 tok/s | 12.1× |

**Cross-node prefix-KV sharing** (Round 6 bench, 2026-04-20): two daemons on loopback, Qwen2.5-Coder-7B Q4, 672-token prompt. When the second node fetches the first node's prefix-KV snapshot instead of re-prefilling locally, **iter-1 TTFT drops from 151.7 s → 11.8 s (12.9×)**. See [Performance chapter](docs/book/src/operations/performance.md#item-8--cross-node-prefix-kv-sharing) and [round6.md](docs/plans/benchmarks/round6.md).

Run your own:
```bash
swarmllm bench --max-tokens 100 --iterations 5 --concurrency 4 --json
```

## Architecture

Single Rust binary, three simultaneous functions:

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| P2P Node | Peer discovery, shard hosting, distributed inference, credits | libp2p / TCP+QUIC |
| LLM API Server | OpenAI + Anthropic + MCP inference endpoints | `localhost:8800/v1/*` |
| Management UI | Dashboard, settings, model browser, chat | `localhost:8800/admin` |

Internally the daemon runs 11 async Tokio tasks wired via mpsc channels, sharing `Arc<SharedState>` + DashMap:

```
┌──────────────────────────────────────────────────────────────────────┐
│                              daemon/                                 │
│                                                                      │
│  NetworkManager ──── InferenceRouter ──── CreditLedger               │
│       │                    │                   │                     │
│  MessageDispatcher    ApiServer          HealthMonitor               │
│       │                    │                   │                     │
│  PoolManager        AutoShardManager    ShardRebalancer              │
│       │                    │                   │                     │
│  AcquisitionManager    UpdateChecker                                 │
└──────────────────────────────────────────────────────────────────────┘
```

Cargo workspace with 3 crates (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`). See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full subsystem deep-dive.

### Node Tiers

| Tier | Requirements | Role |
|------|-------------|------|
| Super Node | Full model in VRAM, high bandwidth | Serves full inference independently |
| Standard Node | Partial VRAM/RAM, moderate bandwidth | Holds layer shards, joins inference pipelines |
| Light Node | Minimal resources | Primarily consumer, contributes bandwidth |

### Credit Priority

Credits determine your request priority. Everyone is served — Bronze just waits longer.

- **Platinum** (top 10%) — near-instant responses
- **Gold** (top 30%) — 1–3 second queue
- **Silver** (positive balance) — 5–15 second queue
- **Bronze** (zero/negative) — 30+ second queue, never locked out

## Installation

### Pre-built Binaries (recommended)

Download from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases) for Linux (CPU + CUDA), Windows, and macOS. Extract and run `./swarmllm run`.

### Package Managers

```bash
brew tap enapt/swarmllm && brew install swarmllm       # Homebrew (macOS/Linux)
yay -S swarmllm                                        # AUR (Arch Linux)
sudo dpkg -i swarmllm_0.1.0_amd64.deb                  # Debian/Ubuntu
sudo rpm -i swarmllm-0.1.0-1.x86_64.rpm                # Fedora/RHEL
```

### Docker

```bash
docker run -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest

# GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest-cuda

# docker-compose and 3-node dev cluster also provided
cp .env.example .env && docker compose up -d
```

### From Source

```bash
# Requires Rust 1.80+
git clone https://github.com/enapt/SwarmLLM.git && cd SwarmLLM

cargo build --release                             # CPU (candle)
cargo build --release --features candle-cuda      # NVIDIA GPU
cargo build --release --features windows-gpu      # Windows: Vulkan + CUDA static
cargo build --release --features llama-vulkan     # Cross-platform Vulkan (NVIDIA/AMD/Intel)
```

Full feature-flag matrix in [CONTRIBUTING.md](CONTRIBUTING.md).

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
```

Run `swarmllm --help` for the full flag list.

## Configuration

Config lives at `~/.local/share/swarmllm/config.toml` (Linux), `~/Library/Application Support/swarmllm/config.toml` (macOS), or `%APPDATA%\swarmllm\config.toml` (Windows). Every value can be overridden with a `SWARMLLM_`-prefixed environment variable:

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_LOGGING_LEVEL=debug
```

Provider API keys are also loaded from a `.env` file in the data directory:

```bash
# ~/.local/share/swarmllm/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

### Key Sections

| Section | Key Settings |
|---------|-------------|
| `[node]` | `listen_port`, `contribution`, `data_dir` |
| `[resources]` | `max_gpu_vram_mb`, `max_ram_mb`, `max_disk_mb`, `max_bandwidth_mbps` |
| `[network]` | `bootstrap_peers`, `enable_mdns`, `gossip_network_id`, `enable_relay`, `max_peers` |
| `[inference]` | `gpu_layers`, `session_timeout_seconds`, `max_batch_size`, `tp_max_latency_ms`, `encrypted_pipeline` |
| `[pool]` | `private_mode`, `private_mode_allow_lan`, `offline_mode`, `invitation_ttl_hours` |
| `[auto_manage]` | `enabled`, `max_storage_mb`, `prune_enabled`, `min_replicas` |
| `[providers]` | API keys for 12 cloud providers, custom providers |
| `[updates]` | `auto_update` (disabled/stable/all), `check_interval_hours` |

See the [Configuration Reference](docs/book/src/configuration/reference.md) for the full list.

## API Endpoints

### Inference (Bearer auth)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| POST | `/v1/messages` | Anthropic Messages API (full Claude Code compatibility) |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List available models |
| GET | `/v1/providers` | List configured cloud providers |
| POST | `/mcp` | MCP JSON-RPC endpoint (7 tools) |

### Admin (CORS-protected)

| Method | Path | Description |
|--------|------|-------------|
| GET/PUT | `/api/admin/config` | Read / update config |
| POST | `/api/admin/config/reload` | Hot-reload config |
| GET | `/api/admin/stats` | Node statistics + hardware info |
| GET | `/api/admin/models` | Model list with shard status |
| GET | `/api/admin/peers` | Connected peers with latency/trust |
| GET | `/api/admin/credits` | Credit balance and tier info |
| GET | `/api/admin/ws` | WebSocket for live updates |
| POST | `/api/admin/hf/download-shards` | Download specific shards from HuggingFace |

### Pools & Private Mode

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pool/state` | Pool membership, stats, private_mode status |
| POST | `/api/pool/create` \| `/api/pool/join` | Create or join a device pool |
| GET/PUT | `/api/pool/private-mode` | Toggle private mode |
| GET | `/api/pool/coverage` | Per-model pool coverage with shard gaps |
| GET/POST/DELETE | `/api/pool/pins` | Manage shard-to-device pins |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus / OpenMetrics |
| GET | `/health` | Health check |
| GET | `/health/ready` | Readiness probe with subsystem status |

Plus ~50 more admin routes for downloads, providers, adapters, identity, and scheduling. See [ARCHITECTURE.md](docs/ARCHITECTURE.md#http-api-routes) for the complete list.

## Platform Support

| Platform | GPU Support | Status |
|----------|------------|--------|
| Linux x86_64 | CUDA (candle + llama.cpp) | Primary target, release binaries, full CI test suite |
| Windows x86_64 (CPU) | — | Runtime-validated 2026-04-23 — single-node, multi-node loopback, split-shard 2-segment pipeline, graceful shutdown all green |
| Windows x86_64 (GPU) | **Vulkan** (NVIDIA/AMD/Intel local) + **CUDA dynamic-loading** (NVIDIA distributed) | Installer bundles CUDA redist DLLs — no CUDA Toolkit needed. Runtime-validated 2026-04-23 (RTX 3070, model loaded on `device=Cuda`) |
| macOS aarch64 | CPU only (Metal planned) | Binary available, compile-validated only |
| macOS x86_64 | CPU only | Best-effort |
| Linux aarch64 | CPU only | Best-effort |

**Windows GPU detection** is automatic — the installer bundles GPU and CPU binaries and a launcher that picks the right one at startup. NVIDIA gets GPU local + GPU distributed, AMD/Intel get GPU local + CPU distributed, no-GPU machines run everything on CPU.

## How SwarmLLM Compares

| Feature | SwarmLLM | Petals | Exo | Bittensor |
|---------|----------|--------|-----|-----------|
| **Language** | Rust (single binary) | Python | Python | Python + Substrate |
| **Install** | Download & run | pip install | pip/source/macOS app | pip + blockchain setup |
| **Scale** | LAN + WAN + Tailscale (zero config) | Internet (volunteer) | LAN + Tailscale (manual) | Internet (blockchain) |
| **E2E Encryption** | **X25519 + ChaCha20 + forward secrecy** | **None** — peers can see prompts | **None** | Minimal (blockchain-level) |
| **Privacy** | Encrypted by default + Private Mode + encrypted pipeline | Unencrypted ([per Petals wiki](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety)) | No encryption between nodes | Subnet-dependent |
| **Incentives** | Credit tiers (no token, no blockchain) | Name on monitor page | None | TAO token (real money) |
| **Parallelism** | Pipeline + tensor (auto-detected LAN) | Pipeline | Tensor + pipeline | Subnet routing |
| **Architectures** | **12** (DeepSeek MoE+MLA, GLM-4, Llama 4, Qwen 3.5 SSM) | ~5 (Llama, Mixtral, Falcon, BLOOM) | ~5 (Llama, Mistral, Qwen, DeepSeek, LLaVA) | Any (subnet-defined) |
| **Shard-only** | **Yes** (no full model download) | No (loads full blocks) | No | N/A |
| **Cloud Fallback** | **12 providers** | No | No | No |
| **VLM + LoRA** | Both (LLaVA verified + per-request LoRA) | LoRA only | VLM experimental | Subnet-specific |
| **API** | **OpenAI + Anthropic + MCP** (full Claude Code) | PyTorch/Transformers | OpenAI + Claude + Ollama | Subnet-defined |
| **Web UI** | Full dashboard + chat + setup wizard | Basic chatbot | Basic chat UI | No built-in UI |
| **SDKs** | Python + JS/TS + LangChain + LlamaIndex | Python native | — | Python |
| **i18n** | **21 languages** | English | English | English |
| **Maintained** | **Active** (2026) | Last release Sep 2023 | **Active** (2025) | **Active** (2025) |

## Documentation

- **[Getting Started](docs/book/src/getting-started.md)** — download, install, start chatting
- **[Configuration Reference](docs/book/src/configuration/reference.md)** — all config options with defaults
- **[Performance & Inference Speedups](docs/book/src/operations/performance.md)** — the default-on speedup stack and flag-gated options
- **[Benchmarking](docs/book/src/operations/benchmarking.md)** — `swarmllm bench`, cross-node KV-sharing recipes
- **[Architecture](docs/ARCHITECTURE.md)** — subsystems, protocols, security model
- **[Tailscale & WAN Access](docs/book/src/operations/tailscale-wan.md)** — remote access via Tailscale, WireGuard, or any VPN
- **[Troubleshooting](docs/book/src/troubleshooting.md)** — common issues and solutions
- **[Diagnostics Guide](docs/DIAGNOSTICS.md)** — DIAG: log instrumentation for debugging
- **[Changelog](CHANGELOG.md)** — release notes and unreleased work
- **[Security Policy](SECURITY.md)** — responsible disclosure

See the full [mdBook documentation](docs/book/) for detailed guides on networking, inference, credits, security, deployment, and monitoring.

## Support

- **Bug reports & feature requests** — [GitHub Issues](https://github.com/enapt/SwarmLLM/issues)
- **Security vulnerabilities** — see [SECURITY.md](SECURITY.md) (email `security@enapt.dev`, do not open a public issue)
- **Questions & discussion** — [GitHub Discussions](https://github.com/enapt/SwarmLLM/discussions)

## Development Transparency

SwarmLLM was developed collaboratively between a human developer and Claude (Anthropic's AI). The entire codebase — Rust backend, JavaScript frontend, P2P networking, distributed inference pipeline, credit system, security hardening, and documentation — was written by Claude Code. The human developer provided architecture direction, testing, and review, but zero lines of code were manually written.

This is an honest disclosure. The project has been through rigorous QA: 771 passing tests, continuous multi-agent code sweeps, security auditing, and multi-node distributed inference tested on real networks. Every commit passes `cargo fmt`, `cargo clippy -- -D warnings`, and the full test suite before push.

We believe AI-assisted development should be transparent. Judge the code on its technical merit — contributions, scrutiny, and feedback are all welcome.

## Contributing

Contributions are welcome — bug reports, feature ideas, code, and documentation all help.

- **[Contributing Guide](CONTRIBUTING.md)** — build from source, run tests, submit PRs
- **[Open Issues](https://github.com/enapt/SwarmLLM/issues)** — browse or file bug reports and feature requests
- **Security vulnerabilities** — see [SECURITY.md](SECURITY.md) for responsible disclosure

```bash
git clone https://github.com/enapt/SwarmLLM.git && cd SwarmLLM
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- run
```

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
