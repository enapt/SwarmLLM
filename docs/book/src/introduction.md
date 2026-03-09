# SwarmLLM

**Join the swarm. Run AI together — for free.**

SwarmLLM is a single Rust binary that turns your computer into a node in a distributed AI inference network. Multiple nodes combine their hardware to run large language models that no single machine could handle alone — for free, with no API tokens or cloud fees.

## Key Features

- **Single Binary** — No Python, no Docker required. Download and run.
- **Combine Resources** — Pool your GPU/CPU with others to run 70B+ models that no one could run alone.
- **OpenAI-Compatible API** — Drop-in replacement for `POST /v1/chat/completions`. Works with any tool that supports the OpenAI format.
- **Anthropic Messages API** — Full `POST /v1/messages` compatibility. Use SwarmLLM as a Claude Code backend to access all models through one endpoint.
- **MCP Server** — Native Model Context Protocol server with `chat`, `models`, and `compare` tools for Claude Code, Cursor, and MCP-compatible agents.
- **Shard-Only Operation** — Nodes only need small pieces (shards) of a model. A phone with 2GB can contribute to running a 70B model.
- **E2E Encryption** — All peer-to-peer communication is encrypted with X25519 + ChaCha20-Poly1305 with forward secrecy.
- **Credit Incentives** — Earn credits by serving inference, hosting shards, and relaying traffic. Higher credits = higher priority.
- **VRAM-Aware** — Automatic shard management based on available GPU memory with on-demand model loading and LRU eviction.
- **Model Trust System** — Automatic trust levels (Discovered, Pinned, DemandVerified, NetworkPopular) gate shard downloads and pruning decisions.
- **Zero-Config Networking** — LAN discovery via mDNS, peer exchange, persistent peer cache, invite codes.
- **Multi-SDK** — Python, JavaScript/TypeScript, LangChain, and LlamaIndex integrations.
- **Web Dashboard** — Built-in swarm-first UI with chat, model browser, network map, and model compare.
- **Cloud Fallback** — Optionally route to 12 cloud providers (incl. Moonshot/Kimi) when no swarm peers have the model you need.

## Performance

Single-node inference on an NVIDIA RTX 3070 Laptop (8GB VRAM):

| Model | GPU | CPU |
|-------|-----|-----|
| TinyLlama 1.1B Q4 | 27.2 tok/s | 4.2 tok/s |
| Gemma-2 2B Q4 | 20.6 tok/s | 3.5 tok/s |
| Phi-3.5 3.8B Q4 | 46.4 tok/s | 1.8 tok/s |
| Qwen2.5-Coder 7B Q4 | 29.0 tok/s | 2.4 tok/s |

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                  Your Computer                       │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │  │
│  │  Node    │  │  Server      │  │  (embedded)  │  │
│  │  (QUIC)  │  │  (Axum)      │  │              │  │
│  └──────────┘  └──────────────┘  └──────────────┘  │
│                                                     │
│         All running on a single port (8800)          │
└─────────────────────────────────────────────────────┘
```

Each SwarmLLM node:
1. Connects to the P2P network over QUIC/UDP
2. Downloads and hosts shard files for popular models
3. Participates in distributed inference pipelines
4. Serves an OpenAI-compatible HTTP API on the same port
5. Provides a web dashboard for monitoring and chat

## Quick Start

```bash
# Download and run (Linux)
tar xzf swarmllm-linux-x86_64.tar.gz
cd swarmllm-linux-x86_64
./swarmllm run

# Open http://localhost:8800 in your browser
```

See the [Getting Started](./getting-started.md) chapter for full instructions.

## Platform Support

| Platform | Priority | GPU Support |
|---|---|---|
| Linux x86_64 | P0 | CUDA + ROCm |
| macOS aarch64 (Apple Silicon) | P1 | Metal (via llama.cpp) |
| Windows x86_64 | P1 | CUDA |
| macOS x86_64 (Intel) | P2 | CPU only |
| Linux aarch64 | P3 | CPU only |
