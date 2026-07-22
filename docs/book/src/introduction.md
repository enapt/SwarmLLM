# SwarmLLM

> **Run AI together — for free.** A single Rust binary that turns your computer into a node in a peer-to-peer LLM inference network. Pool hardware with others to run models too large for any single machine, with no API tokens, no cloud fees, and end-to-end encryption between every peer.

This site is the long-form reference. For source code, releases, and issues, head to [`enapt/SwarmLLM`](https://github.com/enapt/SwarmLLM).

## What you can do with it

- **Chat with AI locally** — open `localhost:8800` after running the binary; the dashboard auto-detects your hardware and walks you through downloading a model.
- **Use it as a drop-in API** — OpenAI-compatible `/v1/chat/completions`, the Anthropic Messages API at `/v1/messages` (full Claude Code support), an MCP server with seven tools, plus 12 cloud providers reachable through one endpoint.
- **Pool hardware** — your phone with 2 GB of RAM can host a few shards of a 70B model and contribute alongside someone else's GPU. Shards download individually via byte-range requests; no node ever needs the full file.
- **Stay private** — every P2P hop uses X25519 + ChaCha20-Poly1305 with forward secrecy. The optional boomerang pipeline ensures no remote node ever sees plaintext.

## Single-node performance (RTX 3070 Laptop, 8 GB VRAM)

| Model                | GPU        | CPU       |
|----------------------|------------|-----------|
| TinyLlama 1.1B Q4    | 27.2 tok/s | 4.2 tok/s |
| Gemma-2 2B Q4        | 20.6 tok/s | 3.5 tok/s |
| Phi-3.5 3.8B Q4      | 46.4 tok/s | 1.8 tok/s |
| Qwen2.5-Coder 7B Q4  | 29.0 tok/s | 2.4 tok/s |

**Distributed-inference speedups (all default-on):** prefix-caching, batched prefill, the Parallax scheduler, and cross-node KV sharing. The cross-node prefix-KV benchmark (2026-04-20) measured a **12.9× iter-1 TTFT speedup** on a 672-token Qwen-7B prompt when a peer had the same prefix already cached (151.7 s → 11.8 s, CPU-CPU, localhost). Each knob is documented in [Performance & Inference Speedups](./operations/performance.md).

## How a node fits together

```text
┌──────────────────────────────────────────────────────────────┐
│                      Your computer (port 8800)                │
│                                                              │
│   P2P node          HTTP server          Web dashboard       │
│   TCP+QUIC          OpenAI · Anthropic   (embedded)          │
│   Noise+Yamux       MCP · Admin          21 languages        │
│                                                              │
│   ─────────────────────────────────────────────────────────  │
│   12 Tokio subsystems · DashMap shared state · redb storage  │
└──────────────────────────────────────────────────────────────┘
```

Each node simultaneously: connects over TCP and QUIC, serves four HTTP API surfaces (OpenAI · Anthropic · MCP · admin) on the same port, hosts shard files for popular models, participates in distributed inference pipelines, and ships an embedded web dashboard.

## Where to go next

<div class="next-steps">
<a href="./getting-started.html"><strong>Getting Started →</strong><span>Install the binary, download your first model, send your first prompt.</span></a>
<a href="./architecture/overview.html"><strong>Architecture →</strong><span>Subsystems, network protocols, encryption model, scheduler design.</span></a>
<a href="./api/openai.html"><strong>API Reference →</strong><span>OpenAI · Anthropic · MCP · Responses · Admin endpoints with examples.</span></a>
<a href="./operations/performance.html"><strong>Performance →</strong><span>The full speedup stack and how to tune it for your network.</span></a>
<a href="./operations/tailscale-wan.html"><strong>WAN setup →</strong><span>Run a swarm across the internet with Tailscale.</span></a>
<a href="./troubleshooting.html"><strong>Troubleshooting →</strong><span>Common pitfalls, diagnostics, and how to file useful bug reports.</span></a>
</div>

## Status

Alpha — actively developed and moving into broader testing. Distributed inference is stable across multi-node deployments. Windows release binaries reach Linux parity (Round 8, 2026-04-23). 1124 lib tests + 75 integration tests run on every PR; continuous security sweeps. [Report issues](https://github.com/enapt/SwarmLLM/issues).

## Platform support

| Platform                          | Status                                | GPU                  |
|-----------------------------------|---------------------------------------|----------------------|
| Linux x86_64                      | Available                             | CUDA                 |
| Windows x86_64                    | Available                             | CUDA                 |
| macOS aarch64 (Apple Silicon)     | Binary available; compile-validated   | CPU only (Metal planned) |
| macOS x86_64 (Intel)              | Best-effort                           | CPU only             |
| Linux aarch64                     | Best-effort                           | CPU only             |

> macOS aarch64 runs `cargo test --lib` + `cargo clippy` on `macos-15` in CI. Integration tests stay Linux-only for now.

All binaries live on the [Releases page](https://github.com/enapt/SwarmLLM/releases).
