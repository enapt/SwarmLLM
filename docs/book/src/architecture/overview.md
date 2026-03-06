# System Overview

SwarmLLM is a single Rust binary that simultaneously functions as:

1. **A P2P network node** — connects to peers over QUIC/UDP using libp2p
2. **An HTTP API server** — serves OpenAI-compatible endpoints via Axum
3. **A web dashboard** — embedded frontend (HTML/CSS/JS, no build step)

All three share a single port (default 8800) and a common `Arc<SharedState>`.

```
┌──────────────────────────────────────────────────────────┐
│                      swarmllm binary                      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │       │
│  │  Node    │  │  Server      │  │  (embedded)  │       │
│  │  (QUIC)  │  │  (Axum)      │  │              │       │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘       │
│       │               │                 │                │
│  ┌────┴───────────────┴─────────────────┴─────────────┐  │
│  │              Shared State (Arc)                     │  │
│  │  DashMap<NodeId, PeerInfo>      — peer registry     │  │
│  │  ModelRegistry                  — models + shards   │  │
│  │  DashMap<ShardId, Vec<NodeId>>  — shard locations   │  │
│  │  Arc<RwLock<CreditBalance>>     — credit balance    │  │
│  │  TrustManager                   — peer trust scores │  │
│  │  broadcast::Sender<()>          — models changed    │  │
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
