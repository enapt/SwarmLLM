# SwarmLLM

[![CI](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/enapt/SwarmLLM/actions/workflows/ci.yml)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust 1.80+](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://ghcr.io/enapt/swarmllm)
[![Release](https://img.shields.io/github/v/release/enapt/SwarmLLM?include_prereleases&label=release)](https://github.com/enapt/SwarmLLM/releases)
[![Discord](https://img.shields.io/badge/discord-join%20chat-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/R3tamKNaj)

A peer-to-peer LLM inference network in a single Rust binary. Pool hardware with other nodes to run 70B+ parameter models on machines that couldn't host them alone — no API tokens, no cloud fees, and encrypted traffic between every peer.

**Join the swarm. Run AI together — for free.**

> **Status — alpha**, actively developed. Distributed inference is stable across multi-node deployments. 1218 lib tests + 79 integration tests run on every PR; continuous security sweeps. [Report issues](https://github.com/enapt/SwarmLLM/issues).
>
> **Recent work (July 2026) — inference across NAT.** Two machines behind ordinary home routers can now run a model together: a sealed application-level relay carries the tensor traffic when no direct path exists, and direct connections are established opportunistically on top. Verified end-to-end by an external tester — a real three-segment pipeline split across two home machines on different continents. Local models also gained working **tool calling** on both API surfaces, streaming included.
>
> **Benchmarks:** prompt processing is up to **3× faster** and replying inside a long conversation up to **5.5× faster** as of v0.3.81 (measured 2026-08-07, see [Benchmarks](#benchmarks)). Cross-node prefix-KV sharing delivers a **12.9× iter-1 TTFT speedup** on 7B prompts when a peer has the same prefix cached (measured 2026-04-20). Windows release binaries reach Linux parity on single-node and split inference (validated 2026-04-23).

For long-form documentation see the [SwarmLLM book](https://enapt.github.io/SwarmLLM/).

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [Quick Start](#quick-start)
- [Use it as an API](#use-it-as-an-api)
- [What it does](#what-it-does)
- [Networking & Privacy](#networking--privacy)
- [Capabilities](#capabilities)
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
- [Contributing & Support](#contributing--support)
- [Development Transparency](#development-transparency)
- [License](#license)

</details>

## Quick Start

Download a binary from [GitHub Releases](https://github.com/enapt/SwarmLLM/releases), extract, and run:

```bash
./swarmllm run
```

Your browser opens to `localhost:8800`. The setup wizard auto-detects your hardware. Pick a model, download it, start chatting.

**It connects to the live network on its own.** On first run your node auto-joins the public swarm — nothing to configure, no ports to forward. A built-in bootstrap anchor gets you onto the network, UPnP opens your port when your router supports it, AutoNAT v2 tells you your reachability, and a relay fallback keeps you connected even behind CGNAT. Peers and shared models appear on the dashboard within seconds.

> 💬 **New here? [Join the Discord](https://discord.gg/R3tamKNaj).** It's the fastest way to find peers to pool with, share node addresses, and get help — the network grows one member at a time, so come say hi.

| Platform | File | Notes |
|----------|------|-------|
| **Windows x86_64** | **`SwarmLLM-Setup.exe`** | **Recommended** — installer auto-detects GPU (NVIDIA / AMD / Intel) |
| Linux x86_64 + CUDA | `swarmllm-linux-x86_64-cuda.tar.gz` | NVIDIA GPU acceleration — **RTX 30-series or newer** |
| Linux x86_64 | `swarmllm-linux-x86_64.tar.gz` | CPU inference |
| Windows x86_64 (GPU) | `swarmllm-windows-x86_64-gpu.zip` | Raw binary: Vulkan + CUDA static |
| Windows x86_64 (CPU) | `swarmllm-windows-x86_64-cpu.zip` | Raw binary: CPU-only fallback |
| macOS Apple Silicon | `swarmllm-macos-aarch64.tar.gz` | CPU inference (Metal planned) |

> **NVIDIA GPU acceleration needs an RTX 30-series or newer** (compute
> capability 8.0+ — Ampere, Ada, Blackwell; the RTX 20-series and GTX 16-series
> are below it). This is FlashAttention's own requirement, and it is what makes
> attention fast enough to be worth shipping. **Older cards are not left
> broken**: SwarmLLM detects them at startup, says so in plain language, and
> runs on the processor instead. On Windows, local inference goes through Vulkan
> and is unaffected on any GPU — it is the distributed path that needs CUDA.

See the [Getting Started Guide](https://enapt.github.io/SwarmLLM/getting-started.html) for platform-specific instructions, or [Installation](#installation) below for package managers, Docker, and source builds.

## Use it as an API

Your access key is written to `api_key` in SwarmLLM's data directory, and is
shown under Settings → Access Token in the dashboard.

```bash
# Linux; macOS uses ~/Library/Application Support/swarmllm/api_key
KEY=$(cat ~/.local/share/swarmllm/api_key)

curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "llama3-70b-q4km",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**As a Claude Code backend** — full Anthropic Messages API with tools, thinking, and streaming. Claude Code reaches every model in the swarm: local GGUF, distributed across peers, or any of 12 cloud providers (`claude --model gpt-5.4`, `claude --model claude-sonnet-5`, etc.).

```bash
ANTHROPIC_BASE_URL="http://localhost:8800" \
ANTHROPIC_AUTH_TOKEN="$KEY" \
claude --model "qwen2.5-coder-7b"
```

**As an MCP server** — add to `~/.claude/settings.json`:

```json
{ "mcpServers": { "swarmllm": { "url": "http://localhost:8800/mcp" } } }
```

Tools: `chat`, `models`, `compare` (multi-model side-by-side), `research` (fan-out), `batch_prompts`, `delegate`, `node_info`.

## What it does

SwarmLLM distributes transformer model layers across a pool of peer-to-peer nodes. Each node contributes a fraction of the compute, and the network orchestrates inference pipelines that chain nodes together — like BitTorrent, but the thing being shared is the work of running the model.

```text
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Node A  │────▶│  Node B  │────▶│  Node C  │
│ Layers   │     │ Layers   │     │ Layers   │──▶ Response
│  0–15    │     │  16–47   │     │  48–79   │
└──────────┘     └──────────┘     └──────────┘
```

Running a 70B-class model on your own normally requires a $10K+ GPU. With SwarmLLM your computer holds a few layers, your friend's holds others, and together you run something neither of you could run alone — no cloud subscription, no API fees.

**Who it's for.** Anyone who wants to chat with AI without paying subscription fees or sharing data with a cloud service. Also: developers who want local/private AI, teams who want to pool GPUs, researchers who need full-control model access, and anyone who wants to contribute spare compute to a public network.

**Key properties.** End-to-end encrypted by default (X25519 + ChaCha20-Poly1305 with forward secrecy); no central server; zero-config peer discovery (mDNS, peer cache, invite codes, PEX, Kademlia DHT); single Rust binary (~33–50 MB); BitTorrent-inspired credit incentives; OpenAI + Anthropic + MCP compatible; shard-only — a node never needs the full model file. See [Capabilities](#capabilities) for the full list.

## Networking & Privacy

A layered discovery stack means nodes find each other without manual configuration:

| Layer | How it works | When |
|-------|-------------|------|
| **Default Anchor** | Fresh installs bootstrap off a built-in public anchor — auto-join the live network, zero config | Instantly on first run |
| **mDNS** | Auto-discovers peers on the same LAN/Wi-Fi | Instantly on startup |
| **Peer Cache** | Remembers peers from previous sessions (redb-backed, max 200) | On restart |
| **Invite Codes** | Share a `swarm://...` network code, or a `swarmpool://...` code to link your own devices into a pool | First time joining |
| **Peer Exchange** | Connected peers share their known peer lists | On each new connection |
| **Kademlia DHT** | Network-wide peer routing | Continuously |

Two laptops on the same Wi-Fi find each other in seconds. A brand-new install auto-joins the public network via a built-in bootstrap anchor — nothing to configure. Returning users reconnect cached peers in under a second. For private networks, set `gossip_network_id` in config to isolate from the public network.

**Connecting across the internet** used to need manual NAT/port-forwarding — now it mostly just works:

- **UPnP (default on)** asks your router to open the P2P port and learns your public address automatically.
- **AutoNAT v2** probes your real reachability per-address and tells you (on the dashboard) whether you're publicly reachable or need a relay — no guesswork.
- **A relay that carries real traffic, not just introductions.** When two machines genuinely cannot reach each other — symmetric NAT, CGNAT, strict firewalls — inference runs *through* a mutually-reachable peer over an end-to-end sealed channel. The relay is a dumb pipe: it forwards bytes it cannot read. This is the guarantee; hole punching is an optimisation layered on top, never a requirement.
- **Circuit Relay v2 + DCUtR hole-punching** upgrade a relayed pair to a direct connection when the network allows it, moving traffic off the relay. Requires both machines on v0.3.21 or newer.
- **Any publicly reachable node contributes relay capacity automatically**, so capacity grows with the network rather than resting on a handful of dedicated machines. Opt out with `network.relay_forwarding_auto = false`.
- **Built-in bootstrap anchor** means there's a reachable node to find on day one, before the network is dense enough for DHT discovery alone.

If you want to run your own publicly-reachable **anchor node** to help bootstrap the network — or your invite code says *"only works on your local network"* — see the **[Networking guide](docs/NETWORKING.md)** (CGNAT check, port-forwarding, dynamic DNS, anchor setup) and the turnkey installer in **[`deploy/anchor/`](deploy/anchor/)**.

### Private Mode

Restricts your *outbound* inference to your device pool — your prompts never leave your machines. Toggle via the dashboard shield icon or the API; a confirmation dialog shows your pool's model coverage before activating.

| Mode | Config | Behaviour |
|------|--------|-----------|
| **Pool only** | `private_mode = true` | Inference restricted to pool members |
| **Pool + LAN** | `private_mode_allow_lan = true` *(default in private mode)* | Pool + mDNS-discovered LAN peers |
| **Offline** | `offline_mode = true` | Air-gapped: no internet, mDNS only |

Private mode is one-way: your data stays private, but your nodes still serve the swarm (processing inference, hosting shards, earning credits). **Shard pinning** lets pool owners assign specific models to specific devices; auto-manage downloads pinned shards with highest priority and never prunes them. The **Coverage Dashboard** shows per-model availability and estimated download sizes to fill gaps.

## Capabilities

### Inference

- **Distributed pipelines** — layers sharded across nodes; automatic pipeline assembly, crash recovery, auto-reconnect; Candle-based direct tensor computation; E2E encrypted hop-by-hop.
- **Default-on speedups** — remote-generate fast path, cross-request prefix cache, cross-node prefix-KV sharing, continuous batching, Sarathi chunked prefill, batched fusion, Parallax scheduler. Numbers + tuning knobs in [Performance & Inference Speedups](https://enapt.github.io/SwarmLLM/operations/performance.html).
- **Flag-gated speedups** — distributed speculative decoding, SWIFT self-speculative, DSD multi-segment speculation, Q8_0 activation compression (~3.76× wire).
- **Tensor parallelism** — automatic TP splitting for LAN peers (RTT ≤ 10 ms), ring-allreduce for 4+ ranks; complements pipeline parallelism for WAN.
- **Vision & LoRA** — VLM support (LLaVA-v1.5-7B verified, Qwen2-VL) with distributed mmproj encoding; per-request LoRA adapter loading.
- **KV-cache reuse** — session-aware cache with pipeline affinity, cross-request prefix caching, chunked prefill, flash attention (CPU + GPU), VRAM-aware LRU eviction.
- **On-demand loading** — models auto-load into VRAM on first request; LRU eviction makes room.

### APIs

- **OpenAI-compatible** — `POST /v1/chat/completions` with streaming, tool calling, stop sequences and JSON mode. Options a model on your own machine cannot honour — `logprobs`, `n` above 1, `logit_bias` — are refused with an explanation rather than accepted and quietly ignored.
- **Tool calling with local models** — not just cloud ones. A locally-run GGUF is told about your tools and its reply is parsed back into proper `tool_calls` / `tool_use` blocks, covering the formats different model families emit natively (Hermes/Qwen, Mistral, Llama 3.x) as well as the generic one. Works streaming and non-streaming on both API surfaces. Output cut off mid-call is reported as text rather than guessed at.
- **Anthropic Messages API** — `POST /v1/messages` with full Claude Code compatibility (tools, `tool_choice`, thinking blocks, `cache_control`, streaming SSE). Non-Claude models auto-translated and routed to cloud providers.
- **MCP server** — native Model Context Protocol with 7 tools.
- **Cloud fallback** — route to 12 providers (OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi). Keys via dashboard, config, env vars, or `.env`.
- **Prompt cache control** — Anthropic-compatible `cache_control` fields.

### Networking & security

- **libp2p transport** — Kademlia DHT, GossipSub, TCP+Yamux + QUIC (with DNS-resolving `/dns4` dialing), connection limits, gossip replay protection.
- **Zero-config internet reachability** — UPnP/IGD port-mapping (default on), AutoNAT v2 per-address reachability probing, Circuit Relay v2 + DCUtR hole-punching for CGNAT, and a built-in default bootstrap anchor so fresh installs auto-join the live network. Internet-capable `swarmpool://` invite blobs carry reachable addresses so two fresh nodes can find each other before the DHT is dense.
- **Three-tier encryption** — pairwise sessions with forward secrecy, pipeline sealing (final segment encrypts output for the requester's key), authenticated sealed gossip. Intermediate pipeline nodes process activation tensors but never see plaintext output. See [Security Model](https://enapt.github.io/SwarmLLM/architecture/security.html).
- **Encrypted pipeline (optional)** — boomerang topology where the requester holds first + last shards, so no remote node ever sees plaintext. Adds ~1 RTT per token.
- **Local embedding privacy** — token→embedding happens locally so first-segment nodes never see raw tokens.
- **Sybil resistance** — Ed25519-signed balance reports, peer reputation with trust decay, subnet clustering detection, leaderboard spoofing protection.
- **API auth** — Bearer token middleware with auto-generated keys, CORS lockdown, SSRF protection, CSP headers, IP-based rate limiting.

### Economy & operations

- **Credits** — earn by serving inference, forwarding activations, hosting shards, seeding data, relaying. Priority tiers (Platinum / Gold / Silver / Bronze) enforced per-request.
- **Pools** — cryptographic nicknames, leaderboard, multi-device credit pooling with dual-signature invitations.
- **Auto-shard management** — VRAM-aware acquisition from HuggingFace and peers with popularity/rarity scoring; smart pruning auto-removes over-replicated shards.
- **Web UI** — chat, model browser, shard visualization, first-run wizard, network map, leaderboard, compare page; mobile-responsive; 21 languages; light/dark/system theme.
- **Fault tolerance** — JoinSet-based supervisor with restart-on-crash for all 12 subsystems; hot-standby failover; shard replication; atomic shard writes.
- **Every answer says where it came from** — chat shows "1.25s · 33.8 tok/s · via 2 peers", and every API response carries the route in headers (`x-swarm-route`, `x-swarm-nodes`, plus standard [`Server-Timing`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Server-Timing) durations your browser's devtools renders natively). "Why was that slow" is the first question anyone asks, and it is now answerable without server access.
- **Observability** — Prometheus `/metrics` with time-to-first-token and time-per-output-token named to the [OpenTelemetry GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai), so collectors and community Grafana dashboards work without a translation layer. Readiness probe `/health/ready`, structured tracing with request-ID correlation, and one greppable summary line per request carrying the whole route and where the time went.
- **Self-service diagnostics** — `GET /api/admin/diagnostics` reports whether your machine is reachable from the internet, whether it has managed direct connections, recent requests with per-segment timings, per-peer serving performance (ping, ms/layer, latency, region — slowest first), what your node has served for others, and the most recent failures including *which machine served each one*. That last detail is what separates "my node has a problem" from "one peer has a problem", and it is the single most useful thing to include in a bug report.
- **Config hot-reload** — change parameters without restarting via SIGHUP or `/api/admin/config/reload`.
- **Auto-updater** — checks GitHub releases, downloads & replaces binary with restart prompt.
- **SDKs** — Python (`pip install swarmllm-client`), JS/TS (zero-dep), LangChain, LlamaIndex.

## Supported Models

12 transformer architectures via native candle inference with GGUF quantization:

| Architecture | Examples | Special features |
|--------------|----------|------------------|
| **Llama** | Llama 2/3, CodeLlama, TinyLlama | Interleaved RoPE, GQA |
| **Llama 4** | Llama 4 Scout (17B), Maverick (400B) | iRoPE (NoPE every 4th layer), MoE |
| **Qwen2** | Qwen2.5-Coder-7B/32B | QKV biases, 32k context |
| **Qwen 3.5** | Qwen3.5-3B/14B/32B (incl. MoE) | Hybrid SSM + attention (Gated Delta Networks) |
| **DeepSeek-V2/V3** | DeepSeek-V2-Lite, DeepSeek-V3 (671B) | MLA attention + MoE FFN |
| **GLM-4** | GLM-4-9B, GLM-4.7 MoE | Partial RoPE, extreme GQA (16:1) |
| **Gemma / Gemma2** | Gemma 2B/7B, Gemma2 9B/27B | Gemma RmsNorm (+1), embedding scaling, logit softcapping |
| **Phi-3** | Phi-3-mini, Phi-3-medium | Su/YaRN RoPE, fused QKV/FFN |
| **Mistral** | Mistral 7B, Mistral Nemo | GQA, interleaved RoPE |
| **Starcoder2** | Starcoder2 3B/7B/15B | Code-optimized, biases |
| **Mixtral** | Mixtral 8x7B, 8x22B | MoE (via llama.cpp backend) |

Quantization: Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16. Context length, RoPE type, attention biases, EOS tokens, and embedding scaling are all detected from GGUF metadata.

## Benchmarks

Single-node, `swarmllm bench`, 100 output tokens, average of 5 runs after a
warm-up. **Hardware:** AMD Ryzen 7 5800H (8C/16T), NVIDIA RTX 3070 Laptop (8 GB
VRAM), WSL2. **Measured 2026-08-07 on v0.3.81-alpha.**

| Model | Params | Quant | GPU (RTX 3070) | CPU only | GPU speedup |
|-------|--------|-------|----------------|----------|-------------|
| Llama-3.2 3B Instruct | 3.2B | Q4_K_M | **29.9 tok/s** | 6.0 tok/s | 5.0× |
| Phi-3.5 Mini | 3.8B | Q4_K_M | **34.5 tok/s** | 3.4 tok/s | 10.1× |

Prompt processing (how fast your prompt is read before the first word appears),
CPU only, Llama-3.2 3B Q4_K_M — this is where recent work landed:

| Prompt length | Before | Now |
|---|---|---|
| ~420 tokens | 12.3 tok/s | **23.0 tok/s** |
| ~1,540 tokens | 6.1 tok/s | **18.6 tok/s** |

Generating inside a long conversation improved separately: at ~1,150 tokens of
context a reply went from 1368 ms per word to **249 ms**.

**On an NVIDIA card, FlashAttention speeds up both phases, and by more the
longer your conversation gets.** Measured end to end on Llama-3.2 3B Q4_K_M
(RTX 3070 Laptop, best of three), switching the method on and off inside a
single binary so nothing else differs:

| Prompt length | Reading the prompt | Each word of the reply |
|---|---|---|
| ~900 tokens | 1576 → **2039 tok/s** (1.3×) | 27.5 → 27.7 tok/s (unchanged) |
| ~2,048 tokens | 1212 → **2034 tok/s** (1.7×) | 15.3 → **32.3 tok/s** (2.1×) |
| ~3,072 tokens | 947 → **1944 tok/s** (2.0×) | 10.9 → **25.6 tok/s** (2.4×) |

Replies are unchanged on the shortest conversation by design: below ~1,024
tokens of context the older method is faster for generation, so SwarmLLM keeps
using it there and switches over above that. Models that do not share key/value
heads across queries — Phi-3.5, for instance — stay on the older method for
generation at every length, because the fused kernel is slower for them.

Per *attention call* in isolation the fused kernel is 2.8–7.4× faster, which is
the figure earlier notes quoted. End to end it is the smaller numbers above:
attention is only part of a forward pass, and the quantized matrix multiplies
around it are unchanged. Both are true; the table is the one you feel.

> **CPU figures from before v0.3.79 are not comparable.** Release binaries were
> compiled without the vectorised quantized kernels, which cost roughly 3× on
> every processor since 2013. Older numbers in this file's history measured that
> defect, not the hardware.

> **If a model is slower than you expect on a GPU, check your contribution
> setting.** `contribution` (default `minimal`) caps the GPU budget at half your
> VRAM, so on an 8 GB card a 3.8B model at Q4 does not fit and silently runs on
> the CPU — 3.4 tok/s instead of 34.5 in the table above. Raise it to `moderate`
> or `maximum` in Settings if the machine is yours to use.

**Cross-node prefix-KV sharing** (measured 2026-04-20): two daemons on loopback, Qwen2.5-Coder-7B Q4, 672-token prompt. When the second node fetches the first's prefix-KV snapshot instead of re-prefilling locally, **iter-1 TTFT drops from 151.7 s → 11.8 s (12.9×)**. See [Performance chapter](https://enapt.github.io/SwarmLLM/operations/performance.html#cross-node-prefix-kv-sharing).

```bash
swarmllm bench --max-tokens 100 --iterations 5 --concurrency 4 --json
```

## Architecture

A single Rust binary running three simultaneous functions on the same port (8800):

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| P2P node | Peer discovery, shard hosting, distributed inference, credits | libp2p / TCP+QUIC |
| HTTP server | OpenAI + Anthropic + MCP + admin endpoints | `localhost:8800/v1/*` |
| Web dashboard | Setup wizard, chat, models, network map, settings | `localhost:8800/admin` |

Full subsystem deep-dive in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

<details>
<summary>Implementation details (for contributors)</summary>

Internally the daemon runs 12 async Tokio tasks wired via mpsc channels, sharing `Arc<SharedState>` + DashMap:

```text
NetworkManager ─── InferenceRouter ─── CreditLedger
       │                  │                  │
MessageDispatcher    ApiServer         HealthMonitor
       │                  │                  │
PoolManager        AutoShardManager   ShardRebalancer
       │                  │                  │
AcquisitionManager   UpdateChecker       HfWatcher
```

Cargo workspace with 3 crates (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`).

### Node tiers & credit priority

| Tier | Requirements | Role |
|------|-------------|------|
| Super node | Full model in VRAM, high bandwidth | Serves inference independently |
| Standard node | Partial VRAM/RAM, moderate bandwidth | Holds layer shards, joins pipelines |
| Light node | Minimal resources | Primarily consumer, contributes bandwidth |

Credits determine request priority. Everyone is served — Bronze just waits longer.

- **Platinum** (top 10%) — near-instant
- **Gold** (top 30%) — 1–3 second queue
- **Silver** (positive balance) — 5–15 second queue
- **Bronze** (zero/negative) — 30+ second queue, never locked out

</details>

## Installation

[Pre-built binaries](#quick-start) cover the most common cases. For other paths:

### Package managers

```bash
brew tap enapt/swarmllm && brew install swarmllm       # Homebrew (macOS / Linux)
yay -S swarmllm                                        # AUR (Arch Linux)
sudo dpkg -i swarmllm_0.3.14-alpha_amd64.deb            # Debian / Ubuntu
sudo rpm -i swarmllm_0.3.14-alpha.x86_64.rpm            # Fedora / RHEL
```

### Docker

```bash
docker run -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest

# GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8800:8800 -v swarmllm-data:/data ghcr.io/enapt/swarmllm:latest-cuda

# docker-compose (single + 3-node dev cluster provided)
cp .env.example .env && docker compose up -d
```

### From source

```bash
# Requires Rust 1.80+
git clone https://github.com/enapt/SwarmLLM.git && cd SwarmLLM

cargo build --release                             # CPU (candle)
cargo build --release --features candle-cuda      # NVIDIA GPU
cargo build --release --features windows-gpu      # Windows: Vulkan + CUDA static
cargo build --release --features llama-vulkan     # Cross-platform Vulkan (NVIDIA / AMD / Intel)
```

Full feature-flag matrix in [CONTRIBUTING.md](CONTRIBUTING.md).

## CLI

```
swarmllm <COMMAND>

Commands:
  run         Start the daemon (default if omitted)
  status      Show node status (queries running daemon)
  chat        Interactive terminal chat
  bench       Run inference benchmarks
  peers       List connected peers with latency and trust scores
  pool        Device pool management
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

| Section | Key settings |
|---------|--------------|
| `[node]` | `listen_port`, `contribution`, `data_dir` |
| `[resources]` | `max_gpu_vram_mb`, `max_ram_mb`, `max_disk_mb`, `max_bandwidth_mbps` |
| `[network]` | `bootstrap_peers`, `enable_mdns`, `gossip_network_id`, `enable_relay`, `max_peers` |
| `[inference]` | `gpu_layers`, `session_timeout_seconds`, `max_batch_size`, `tp_max_latency_ms`, `encrypted_pipeline` |
| `[pool]` | `private_mode`, `private_mode_allow_lan`, `offline_mode`, `invitation_ttl_hours` |
| `[auto_manage]` | `enabled`, `max_storage_mb`, `prune_enabled`, `min_replicas` |
| `[providers]` | API keys for 12 cloud providers, custom providers |
| `[updates]` | `auto_update` (`disabled` / `stable` / `all`), `check_interval_hours` |

Full list: [Configuration Reference](https://enapt.github.io/SwarmLLM/configuration/reference.html).

## API Endpoints

### Inference (Bearer auth)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| POST | `/v1/messages` | Anthropic Messages API (full Claude Code compatibility) |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List available models |
| GET | `/v1/providers` | List configured cloud providers |
| POST | `/mcp` | MCP JSON-RPC endpoint |

### Admin & operations

| Method | Path | Description |
|--------|------|-------------|
| GET / PUT | `/api/admin/config` | Read / update config |
| POST | `/api/admin/config/reload` | Hot-reload config |
| GET | `/api/admin/stats` | Node statistics + hardware info |
| GET | `/api/admin/models` | Model list with shard status |
| GET | `/api/admin/peers` | Connected peers with latency / trust |
| GET | `/api/admin/credits` | Credit balance and tier info |
| GET | `/api/admin/diagnostics` | Plain-text health report for a shell or a bug report |
| GET | `/api/admin/performance` | Routes, per-segment timings, per-peer performance, hourly trend (JSON) |
| GET | `/api/admin/ws` | WebSocket for live updates |
| GET | `/api/pool/state` | Pool membership, stats, private-mode status |
| GET / PUT | `/api/pool/private-mode` | Toggle private mode |
| GET | `/metrics` | Prometheus / OpenMetrics |
| GET | `/health/ready` | Readiness probe with subsystem status |

Plus ~60 more admin / pool / scheduling routes. Full reference in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#http-api-routes).

## Platform Support

| Platform | GPU | Status |
|----------|-----|--------|
| Linux x86_64 | CUDA (candle + llama.cpp) | Primary target — release binaries, full CI test suite |
| Windows x86_64 (CPU) | — | Runtime-validated 2026-04-23 (single-node, multi-node loopback, split-shard 2-segment pipeline, graceful shutdown) |
| Windows x86_64 (GPU) | **Vulkan** (NVIDIA / AMD / Intel local) + **CUDA dynamic-loading** (NVIDIA distributed) | Installer bundles CUDA redist DLLs — no CUDA Toolkit needed. Runtime-validated 2026-04-23 (RTX 3070, model loaded on `device=Cuda`) |
| macOS aarch64 | CPU only (Metal planned) | Binary available, compile-validated; CI runs `cargo test --lib` + clippy on `macos-15` |
| macOS x86_64 (Intel) | CPU only | Best-effort |
| Linux aarch64 | CPU only | Best-effort |

The Windows installer bundles GPU and CPU binaries plus a launcher that picks the right one at startup: NVIDIA gets GPU local + GPU distributed, AMD/Intel get GPU local + CPU distributed, no-GPU machines run everything on CPU.

## How SwarmLLM Compares

| Feature | SwarmLLM | Petals | Exo | Bittensor |
|---------|----------|--------|-----|-----------|
| **Language** | Rust (single binary) | Python | Python | Python + Substrate |
| **Install** | Download & run | `pip install` | pip / source / macOS app | pip + blockchain setup |
| **Scale** | LAN + WAN + Tailscale (zero config) | Internet (volunteer) | LAN + Tailscale (manual) | Internet (blockchain) |
| **E2E Encryption** | **X25519 + ChaCha20 + forward secrecy** | None — peers can see prompts | None | Minimal (blockchain-level) |
| **Privacy** | Encrypted by default + Private Mode + encrypted pipeline | Unencrypted ([per Petals wiki](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety)) | None between nodes | Subnet-dependent |
| **Incentives** | Credit tiers (no token, no blockchain) | Name on monitor page | None | TAO token (real money) |
| **Parallelism** | Pipeline + tensor (auto-detected LAN) | Pipeline | Tensor + pipeline | Subnet routing |
| **Architectures** | **12** (DeepSeek MoE+MLA, GLM-4, Llama 4, Qwen 3.5 SSM) | ~5 (Llama, Mixtral, Falcon, BLOOM) | ~5 (Llama, Mistral, Qwen, DeepSeek, LLaVA) | Any (subnet-defined) |
| **Shard-only** | **Yes** (no full model download) | No (loads full blocks) | No | N/A |
| **Cloud Fallback** | **12 providers** | No | No | No |
| **VLM + LoRA** | Both (LLaVA verified + per-request LoRA) | LoRA only | VLM experimental | Subnet-specific |
| **API** | **OpenAI + Anthropic + MCP** (full Claude Code) | PyTorch / Transformers | OpenAI + Claude + Ollama | Subnet-defined |
| **Web UI** | Full dashboard + chat + setup wizard | Basic chatbot | Basic chat UI | None built-in |
| **SDKs** | Python + JS/TS + LangChain + LlamaIndex | Python native | — | Python |
| **i18n** | **21 languages** | English | English | English |
| **Maintained** | **Active** (2026) | Last release Sep 2023 | **Active** (2025) | **Active** (2025) |

## Documentation

- **[Getting Started](https://enapt.github.io/SwarmLLM/getting-started.html)** — download, install, start chatting
- **[Configuration Reference](https://enapt.github.io/SwarmLLM/configuration/reference.html)** — all config options with defaults
- **[Performance & Inference Speedups](https://enapt.github.io/SwarmLLM/operations/performance.html)** — the default-on stack and flag-gated options
- **[Architecture](docs/ARCHITECTURE.md)** — subsystems, protocols, security model
- **[Tailscale & WAN](https://enapt.github.io/SwarmLLM/operations/tailscale-wan.html)** — remote access via Tailscale, WireGuard, or any VPN
- **[Troubleshooting](https://enapt.github.io/SwarmLLM/troubleshooting.html)** — common issues and solutions
- **[Diagnostics Guide](docs/DIAGNOSTICS.md)** — DIAG: log instrumentation for debugging
- **[Changelog](CHANGELOG.md)** — release notes and unreleased work
- **[Security Policy](SECURITY.md)** — responsible disclosure

Full mdBook site: [https://enapt.github.io/SwarmLLM/](https://enapt.github.io/SwarmLLM/).

## Contributing & Support

- **Community chat** — [Join the Discord](https://discord.gg/R3tamKNaj) — share node addresses, coordinate the network, get help
- **Bug reports & feature requests** — [GitHub Issues](https://github.com/enapt/SwarmLLM/issues)
- **Questions & discussion** — [GitHub Discussions](https://github.com/enapt/SwarmLLM/discussions)
- **Security vulnerabilities** — [SECURITY.md](SECURITY.md) (email `security@enapt.dev`, do not open a public issue)
- **Contributing guide** — [CONTRIBUTING.md](CONTRIBUTING.md) — build, test, submit PRs

```bash
git clone https://github.com/enapt/SwarmLLM.git && cd SwarmLLM
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- run
```

## Development Transparency

SwarmLLM was developed collaboratively between a human developer and Claude Code. The human provided architecture direction, testing, and review; Claude wrote the code. We disclose this openly so you can judge the project on its technical merits — 1169 lib tests + 75 integration tests run on every PR, every commit passes `cargo fmt` and `cargo clippy -- -D warnings`, and continuous multi-agent code sweeps and security audits track findings in `.claude/sweep-log.jsonl`. Contributions, scrutiny, and feedback all welcome.

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
