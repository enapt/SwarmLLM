# System Overview

SwarmLLM is a single Rust binary that simultaneously functions as:

1. **A P2P network node** — connects to peers over TCP (Noise+Yamux) and QUIC/UDP using libp2p
2. **An HTTP API server** — serves OpenAI + Anthropic-compatible endpoints, MCP server, and cloud provider proxy via Axum
3. **A web dashboard** — embedded frontend (component-based vanilla HTML/CSS/JS, 12 HTML templates, no build step)

All three share a single port (default 8800) and a common `Arc<SharedState>`.

```
┌──────────────────────────────────────────────────────────┐
│                      swarmllm binary                      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │       │
│  │  Node    │  │  Server      │  │  (embedded)  │       │
│  │(TCP+QUIC)│  │  (Axum)      │  │              │       │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘       │
│       │               │                 │                │
│  ┌────┴───────────────┴─────────────────┴─────────────┐  │
│  │              Shared State (Arc)                     │  │
│  │  DashMap<NodeId, PeerInfo>      — peer registry     │  │
│  │  ModelRegistry                  — models + shards   │  │
│  │  state.events (EventBus)        — activity + dashboard│ │
│  │  state.credits (CreditPool)     — balance + pool     │  │
│  │  state.models (ModelMgmt)       — acquisition + trust │  │
│  │  state.metrics (MetricsProviders)— stats + providers │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **Config priority:** CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Data directory:** `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- **Port layout:** HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- **Shard-only:** Nodes never need a full GGUF. Shards are downloaded individually.
- **No blockchain:** Credit system uses dual-signed transactions, not a token or chain

## Technology Stack

| Component | Library |
|---|---|
| Async runtime | Tokio (multi-threaded) |
| P2P networking | libp2p 0.55 (Kademlia, GossipSub, QUIC) |
| HTTP server | Axum 0.7 |
| Tensor compute | candle-core/candle-transformers |
| GGUF inference | llama-cpp-2 (optional backend) |
| Cryptography | ed25519-dalek, x25519-dalek, chacha20poly1305 |
| Content hashing | BLAKE3 |
| Database | redb (pure-Rust, ACID; sled migration via `migrate-sled` feature) |
| Concurrent maps | DashMap 6 |
