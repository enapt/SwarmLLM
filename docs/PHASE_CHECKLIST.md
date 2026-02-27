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
- [x] `src/model/acquisition.rs` — Secure model acquisition from network (BLAKE3-verified)
- [x] `src/health/mod.rs` — Module structure
- [x] `src/health/monitor.rs` — Periodic health pings
- [x] `src/daemon.rs` — Top-level daemon orchestration (spawns all tasks)
- [x] `src/types.rs` — Add NodeId, NodeCapability, ShardInfo, ShardId, SwarmMessage types

**Acceptance test**: Start 3 nodes on LAN. Node A has a shard. Node B discovers A, downloads shard. Node C verifies shard exists on both A and B.

---

## Phase 3: Distributed Inference

**Goal**: Inference request flows through a pipeline of multiple nodes.

- [x] `src/inference/router.rs` — Request queuing, pipeline assembly trigger
- [x] `src/inference/scheduler.rs` — Pipeline assembly algorithm (greedy layer assignment)
- [x] `src/inference/pipeline.rs` — Pipeline execution: forward activations between nodes
- [x] `src/inference/kv_cache.rs` — Session-based KV-cache management
- [x] Wire NetworkManager <-> InferenceRouter <-> ShardExecutor communication
- [x] Implement LayerForward and LayerResult message handling in NetworkManager
- [x] Implement hot-standby failover in pipeline.rs
- [x] Add PipelineAssignment to SharedState for monitoring

**Acceptance test**: 3 nodes, each holding different layer ranges. Client sends request to Node A, pipeline assembled across all 3 nodes, returns generated text.

---

## Phase 4: Credit System

**Goal**: Incentive-aligned credit system with priority tiers.

- [x] `src/credit/mod.rs` — Module structure
- [x] `src/credit/ledger.rs` — Local balance tracking, credit operations
- [x] `src/credit/transaction.rs` — Transaction creation, dual signing
- [x] `src/credit/priority.rs` — Tier calculation, queue ordering
- [x] `src/credit/anti_gaming.rs` — Spot-check verification, rate limiting
- [x] Wire credit events into inference pipeline (earn on serve, spend on request)
- [x] Add credit gossip for percentile estimation
- [x] Priority queue in InferenceRouter respects tiers

**Acceptance test**: Node A serves 100 requests for Node B. A's balance increases. B's balance decreases. A achieves higher priority tier.

---

## Phase 5: Web UI and UX

**Goal**: Polished web interface with setup wizard, dashboard, and chat.

- [x] `frontend/index.html` + `frontend/js/app.js` — Single-page app (dashboard + chat + setup wizard)
- [x] `frontend/css/style.css` — Dark theme styling
- [x] `build.rs` — Embed frontend assets with include_dir
- [x] `src/ui/mod.rs` + `src/ui/assets.rs` — Serve embedded files
- [x] `src/api/admin.rs` — All admin REST endpoints
- [x] `src/api/websocket.rs` — Real-time dashboard updates via WebSocket
- [x] Hardware auto-detection in config.rs (GPU probing, RAM, disk)
- [x] `open_browser_on_start` logic in daemon.rs

**Acceptance test**: Fresh install. Run binary. Browser opens. Setup wizard detects hardware. User completes wizard. Dashboard shows real-time stats. Chat works.

---

## Phase 6: Hardening and Scale

**Goal**: Production-ready networking, performance, and cross-platform support.

- [x] NAT traversal: AutoNAT, DCUtR, relay
- [x] `src/network/relay.rs` — Circuit relay client/server
- [x] Protocol migration from serde_json to Cap'n Proto for tensor data
- [x] `proto/messages.capnp` — Cap'n Proto schema definitions
- [x] `src/model/quantization.rs` — GGUF format handling, quantization utilities
- [x] Shard rebalancing on node join/leave
- [x] Model governance voting
- [x] Cross-platform builds and testing (Linux, macOS, Windows)
- [x] Speculative decoding groundwork
- [x] MoE-optimized sharding


## Phase 7: Security, Identity & Device Pools

**Goal**: End-to-end encryption, user identity with nicknames/leaderboard, multi-device pooling.

- [x] `src/crypto/session.rs` — X25519 ECDH + ChaCha20-Poly1305 pairwise session encryption
- [x] `src/crypto/pipeline_seal.rs` — Per-request prompt sealing for multi-node pipelines
- [x] `src/crypto/gossip_seal.rs` — Epoch-based group key encryption for GossipSub
- [x] `src/crypto/key_rotation.rs` — Background key rotation (10min sessions, 1hr group keys)
- [x] `src/identity/nickname.rs` — Signed nicknames, leaderboard, anonymous-by-default
- [x] `src/api/identity.rs` — Nickname CRUD + leaderboard + peer identity endpoints
- [x] `src/pool/types.rs` — Pool data structures with dual-sig invitation protocol
- [x] `src/pool/crypto.rs` — Pool invitation/acceptance/removal/forward signing
- [x] `src/pool/manager.rs` — PoolManager subsystem (9th Tokio task)
- [x] `src/pool/forward.rs` — Credit forwarding from pool members to owner
- [x] `src/api/pool.rs` — 8 pool management endpoints
- [x] Security hardening: path traversal fix, SSRF blocklist, CORS lockdown, key permissions, input validation
- [x] Bug fixes: failover peer_id, pending_layer_results leak, OOM cap, shutdown DB flush, credit race
- [x] Nonce overflow fix, pool signature verification (credit forward, invitation, gossip state)
- [x] Rate limiter ordering fix, gossip key derivation from bootstrap addresses, pool name validation
- [x] Shutdown endpoint restricted to localhost, CORS restricted to specific methods/headers

**Acceptance test**: Two nodes. Set nicknames. Create device pool. Member earns credits that forward to owner. Tensor traffic is encrypted. Leaderboard shows pool ranking.

> **Note**: Self-governance was removed. Issues, proposals, releases, and project management are handled via the GitHub repository.

---

## Phase 8: Production Hardening

**Goal**: API authentication, dynamic EOS tokens, credit system integrity, performance optimizations.

- [x] `src/api/middleware.rs` — Bearer token authentication middleware with exempt paths
- [x] `src/config.rs` — `ApiConfig` with `api_key` field, auto-generation on first run
- [x] `GET /api/admin/api-key` — Retrieve current API key for dashboard display
- [x] EOS tokens loaded from GGUF metadata (`tokenizer.ggml.eos_token_id`) with architecture fallbacks
- [x] `eos_tokens: Vec<u32>` in `SplitModel` and `LoadedModelInfo` — no more hardcoded Qwen2 tokens
- [x] `acquisition_progress` DashMap cleanup: HealthMonitor removes Complete/Failed entries after 1hr
- [x] `apply_credit_direct()` helper in `credit/ledger.rs` — proper balance persistence for all credit ops
- [x] Credit mutations in `router.rs` and `pipeline.rs` route through ledger with DB persistence
- [x] `model_loaded: AtomicBool` in SharedState — lock-free readiness check, executor Mutex only for generation
- [x] Queue drain via `tokio::sync::Notify` — zero-latency dispatch replacing 50ms fixed polling
- [x] `AntiGaming` wired into CreditTransaction handler with periodic cleanup in HealthMonitor

**Acceptance test**: API returns 401 without Bearer token. Different GGUF models use correct EOS tokens. Credit operations persist across restarts. Queue responds instantly to new requests.

---

## Phase 9: Auto-Manage & VRAM Awareness

**Goal**: Nodes automatically acquire missing shards with VRAM-aware scoring, card-based model UI with shard visualization.

- [x] `src/model/auto_manage.rs` — AutoShardManager subsystem (10th Tokio task)
- [x] Config: `[auto_manage]` section — enabled, max_storage_mb, interval_minutes, max_shards
- [x] Scoring: `popularity × rarity_bonus × configured_bonus × vram_fitness`
- [x] Configured shard range priority: 100x bonus for shards in `--shards` range
- [x] Configured range focus: `candidates.retain()` filters to only configured-range shards when any are missing
- [x] VRAM awareness: `estimate_model_vram_mb()`, `global_pool_vram_mb()`, `local_vram_mb()` with nvidia-smi fallback
- [x] VRAM fitness multiplier (0.1–1.0x) in candidate scoring
- [x] Disk existence checks: shard registration verifies file exists (startup + generate_and_register)
- [x] `GET /api/admin/shard-storage` returns `pool_vram_mb`, `local_vram_mb`, `estimated_vram_mb` per model
- [x] Model UI: card-based layout with numbered shard grid (green=local, blue=peer, pulsing=downloading, dashed=missing)
- [x] Model cards: status badges, meta info, shard legend, download progress bars, VRAM estimates
- [x] HF model browser: shows VRAM fitness indicator (green/red) against pool capacity
- [x] Bug fix: shard_registry dedup (prevent unbounded growth on repeated ShardAnnounce)
- [x] Bug fix: pending_layer_results cleanup on failover error paths (memory leak)
- [x] Bug fix: /api/admin/api-key excluded from auth exemption (sensitive endpoint)
- [x] Bug fix: configured_range filter constrained by model_id (multi-model correctness)
- [x] Bug fix: pool/manager.rs signature byte conversion uses safe error handling (no unwrap)

**Acceptance test**: Node starts with incomplete shards. Auto-manager downloads only the missing configured-range shards. Shard grid shows local/peer/missing/downloading status. VRAM fitness influences model scoring.
