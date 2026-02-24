# SwarmLLM Build Phase Checklist

> Each phase is a deployable milestone. Build in this exact order.

---

## Phase 1: Local Inference Daemon

**Goal**: Single binary that loads a GGUF model and serves it via OpenAI-compatible API.

- [x] `src/main.rs` — CLI with clap (subcommands: `run`, `version`)
- [x] `src/config.rs` — Load config from TOML, environment, defaults
- [x] `src/error.rs` — SwarmError enum with thiserror
- [x] `src/types.rs` — Core types: ModelId, SamplingParams, ChatMessage, Role
- [x] `src/inference/mod.rs` — Module structure
- [x] `src/inference/executor.rs` — Wrap llama-cpp-2 for GGUF loading and token generation
- [x] `src/inference/sampling.rs` — Temperature, top-p, top-k sampling
- [x] `src/api/mod.rs` — Module structure
- [x] `src/api/server.rs` — Axum server on localhost:8800
- [x] `src/api/openai.rs` — `/v1/chat/completions` (streaming + non-streaming), `/v1/models`
- [x] `src/api/middleware.rs` — Request logging, CORS
- [x] `src/storage/mod.rs` — Module structure
- [x] `src/storage/db.rs` — sled wrapper (config storage)
- [x] `src/lib.rs` — Re-exports
- [x] `Cargo.toml` — Dependencies for Phase 1
- [x] `config/default.toml` — Default configuration

**Acceptance test**: `curl localhost:8800/v1/chat/completions -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":true}'` returns streamed tokens from a local GGUF model.

---

## Phase 2: P2P Networking Foundation

**Goal**: Nodes discover each other, exchange shard information, and can transfer shard data.

- [x] `src/identity/mod.rs` — Module structure
- [x] `src/identity/keypair.rs` — Ed25519 key generation, storage, export/import
- [x] `src/identity/keystore.rs` — Encrypted storage (AES-256-GCM + Argon2id)
- [x] `src/network/mod.rs` — Module structure
- [x] `src/network/transport.rs` — QUIC transport setup
- [x] `src/network/behaviour.rs` — Custom NetworkBehaviour (Kademlia + GossipSub + request_response)
- [x] `src/network/discovery.rs` — Bootstrap, peer discovery loop, PEX
- [x] `src/network/manager.rs` — Swarm lifecycle, message routing
- [x] `src/network/protocol.rs` — SwarmMessage serialization (serde_json initially)
- [x] `src/model/mod.rs` — Module structure
- [x] `src/model/manifest.rs` — Parse .swarm manifests
- [x] `src/model/shard.rs` — Shard loading, BLAKE3 verification
- [x] `src/model/distribution.rs` — Shard request/response protocol
- [x] `src/model/registry.rs` — Track known models and shard locations
- [x] `src/health/mod.rs` — Module structure
- [x] `src/health/monitor.rs` — Periodic health pings
- [x] `src/daemon.rs` — Top-level daemon orchestration (spawns all tasks)
- [x] `src/types.rs` — Add NodeId, NodeCapability, ShardInfo, ShardId, SwarmMessage types

**Acceptance test**: Start 3 nodes on LAN. Node A has a shard. Node B discovers A, downloads shard. Node C verifies shard exists on both A and B.

---

## Phase 3: Distributed Inference

**Goal**: Inference request flows through a pipeline of multiple nodes.

- [ ] `src/inference/router.rs` — Request queuing, pipeline assembly trigger
- [ ] `src/inference/scheduler.rs` — Pipeline assembly algorithm (greedy layer assignment)
- [ ] `src/inference/pipeline.rs` — Pipeline execution: forward activations between nodes
- [ ] `src/inference/kv_cache.rs` — Session-based KV-cache management
- [ ] Wire NetworkManager <-> InferenceRouter <-> ShardExecutor communication
- [ ] Implement LayerForward and LayerResult message handling in NetworkManager
- [ ] Implement hot-standby failover in pipeline.rs
- [ ] Add PipelineAssignment to SharedState for monitoring

**Acceptance test**: 3 nodes, each holding different layer ranges. Client sends request to Node A, pipeline assembled across all 3 nodes, returns generated text.

---

## Phase 4: Credit System

**Goal**: Incentive-aligned credit system with priority tiers.

- [ ] `src/credit/mod.rs` — Module structure
- [ ] `src/credit/ledger.rs` — Local balance tracking, credit operations
- [ ] `src/credit/transaction.rs` — Transaction creation, dual signing
- [ ] `src/credit/priority.rs` — Tier calculation, queue ordering
- [ ] `src/credit/anti_gaming.rs` — Spot-check verification, rate limiting
- [ ] Wire credit events into inference pipeline (earn on serve, spend on request)
- [ ] Add credit gossip for percentile estimation
- [ ] Priority queue in InferenceRouter respects tiers

**Acceptance test**: Node A serves 100 requests for Node B. A's balance increases. B's balance decreases. A achieves higher priority tier.

---

## Phase 5: Web UI and UX

**Goal**: Polished web interface with setup wizard, dashboard, and chat.

- [ ] `frontend/setup.html` + `frontend/js/setup.js` — First-run wizard
- [ ] `frontend/index.html` + `frontend/js/app.js` — Admin dashboard
- [ ] `frontend/chat.html` + `frontend/js/chat.js` — Chat interface
- [ ] `frontend/css/style.css` — Dark theme styling
- [ ] `build.rs` — Embed frontend assets with include_dir
- [ ] `src/ui/mod.rs` + `src/ui/assets.rs` — Serve embedded files
- [ ] `src/api/admin.rs` — All admin REST endpoints
- [ ] `src/api/websocket.rs` — Real-time dashboard updates via WebSocket
- [ ] Hardware auto-detection in config.rs (GPU probing, RAM, disk)
- [ ] `open_browser_on_start` logic in daemon.rs

**Acceptance test**: Fresh install. Run binary. Browser opens. Setup wizard detects hardware. User completes wizard. Dashboard shows real-time stats. Chat works.

---

## Phase 6: Hardening and Scale

**Goal**: Production-ready networking, performance, and cross-platform support.

- [ ] NAT traversal: AutoNAT, DCUtR, relay
- [ ] `src/network/relay.rs` — Circuit relay client/server
- [ ] Protocol migration from serde_json to Cap'n Proto for tensor data
- [ ] `proto/messages.capnp` — Cap'n Proto schema definitions
- [ ] `src/model/quantization.rs` — GGUF format handling, quantization utilities
- [ ] Shard rebalancing on node join/leave
- [ ] Model governance voting
- [ ] Cross-platform builds and testing (Linux, macOS, Windows)
- [ ] Speculative decoding groundwork
- [ ] MoE-optimized sharding

---

## Phase 7: Self-Governance

**Goal**: Fully decentralized development lifecycle — no central repository needed.

- [ ] `src/governance/mod.rs` — Module structure
- [ ] `src/governance/proposals.rs` — RFC creation, validation, lifecycle management
- [ ] `src/governance/voting.rs` — Weighted voting, quorum calculation, tallying
- [ ] `src/governance/issues.rs` — Bug reports, feature requests, prioritization
- [ ] `src/governance/releases.rs` — Release candidate management, threshold signing
- [ ] `src/governance/testing.rs` — Distributed test coordination, canary rollouts
- [ ] `src/governance/changelog.rs` — Auto-generated changelog from merged proposals
- [ ] Wire governance messages into NetworkManager (new GossipSub topics)
- [ ] Wire governance into ApiServer (new admin endpoints)
- [ ] Wire governance into admin dashboard (new tabs)
- [ ] `UpdateManager` integration into daemon.rs
- [ ] Genesis period bootstrap logic

**Acceptance test**: Full lifecycle from issue filing through proposal, vote, build, test, approve, and canary rollout — with no central coordination.
