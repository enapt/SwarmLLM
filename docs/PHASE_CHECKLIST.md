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

---

## Phase 10: System-Wide Audit Hardening

**Goal**: Fix all 130 bugs/issues identified by 6-agent parallel audit. Strengthen security, correctness, reliability, and scalability.

**Date**: 2026-02-28 | **Findings**: 31 critical, 51 important, 48 minor

### Security / Credit / Crypto / Pools (21 fixes)
- [x] Authenticated MemberLeft — Ed25519-signed leave notices required
- [x] Pool credit forwarding functional — apply_credit_direct to owner + dual-sig cosigning
- [x] Transaction replay protection — UUID dedup before accepting
- [x] AntiGaming wired — atomic check_and_record_transaction in credit flow
- [x] Nonce reuse fix — session epoch mixed into HKDF key derivation
- [x] Gossip authenticity — Ed25519 origin signature on sealed gossip messages
- [x] Pool state gossip — acceptance_signature verified per member
- [x] Saturating arithmetic for balance/lifetime counters
- [x] Priority tier consistency — tier_name delegates to calculate_tier
- [x] Nickname timestamp freshness check (1hr window)
- [x] Credit forward amount validation (>0)
- [x] Replay detection — monotonic recv nonce tracking
- [x] Windows key file ACL restriction
- [x] Dual-sig completion for pool credit forwards

### Networking / P2P (22 fixes)
- [x] pending_shard_requests keyed by OutboundRequestId (not PeerId)
- [x] file.seek() errors propagated (not silently ignored)
- [x] Shard transfer timeout 300s (not 10s default)
- [x] O(1) PeerId→NodeId reverse lookup (DashMap)
- [x] Cleanup download state on disconnect
- [x] Peer registry pruning on disconnect
- [x] Async peer count updates (write().await)
- [x] GossipSub publish buffering at startup
- [x] Connection limits (500 total, 2 per peer)
- [x] DHT record expiry (1 hour)
- [x] Kademlia Client/Server mode based on NAT
- [x] Backpressure on outbound channel (try_send + warn)
- [x] Dead transport code removed, relay logging, wire format comment, gossip timestamp validation

### Model / Shard Management (18 fixes)
- [x] EOS tokens from GGUF metadata (not hardcoded [2])
- [x] Zero-hash bypass restricted (network shards require real hash)
- [x] Dead request_shard() deleted
- [x] 3-retry exponential backoff for shard downloads
- [x] Atomic shard writes (.tmp + rename)
- [x] reconstruct_gguf hash verification (not just size)
- [x] Auto-manage HF path verifies shards
- [x] acquisition_progress total_shards from manifest
- [x] HF 429/503 retry with Retry-After
- [x] Graceful progress task shutdown
- [x] Quarantine rename failure handling
- [x] Governance vote overflow protection
- [x] HashSet for shard_holders (O(1) dedup)
- [x] MoE VRAM estimate + context length factor

### Inference Pipeline (20 fixes)
- [x] KV-cache concurrent access documented + serialized
- [x] Random sampler seed (not fixed 42)
- [x] rand::random PRNG (not homebrew hash)
- [x] Context window from GGUF metadata (not hardcoded 4096)
- [x] Failover passes actual index_pos (not 0)
- [x] top_p from params (not hardcoded 0.95)
- [x] Earned credits persisted to DB
- [x] TOCTOU-safe split model loading (entry API)
- [x] Load-aware scheduling (active pipeline count)
- [x] Skip batch timeout for single requests
- [x] LRU mask cache eviction (max 16)
- [x] ShardReader cached file lengths (no extra seeks)
- [x] KvCacheManager wired to router
- [x] Non-ASCII token count estimation
- [x] Consistent TensorFormat in failover
- [x] Strict top_k, pass-through request IDs, shard-aware is_first/is_last

### Daemon / Storage / Health (30 fixes)
- [x] Graceful shutdown with DB flush
- [x] Rebalancer only announces held shards
- [x] Blocking I/O wrapped in spawn_blocking
- [x] SIGTERM handler for systemd/Docker
- [x] AutoShardManager handle tracked in select!
- [x] LayerForward semaphore (max 64 concurrent)
- [x] Stale channel cleanup
- [x] DB deserialization failure logging
- [x] Config validation (ranges, port 0)
- [x] Env var parse failure logging
- [x] Disk space for data_dir partition only
- [x] DB schema versioning
- [x] shard_range persistence
- [x] Per-model rebalance cooldown
- [x] GGUF architecture mapping
- [x] DB manifest hash verification
- [x] HTTP status codes: 402 (credits), 503 (peer), 5xx logging
- [x] API key stderr-only, reqwest for status, config log level wiring

### API / Frontend (18 fixes)
- [x] Auth middleware protects /v1/* and sensitive admin endpoints
- [x] Leaderboard wired to credit data (not hardcoded)
- [x] /v1/completions streaming support
- [x] SSRF: IPv6 link-local + IPv4-mapped blocked
- [x] EOS from GGUF in HF download path
- [x] Shared reqwest::Client (LazyLock)
- [x] Sampling parameter clamping for completions
- [x] HTTP 400 for admin validation errors (not 200)
- [x] WebSocket ping/pong heartbeat (30s)
- [x] Duplicate escapeHtml removed
- [x] Request body size limit (2MB)
- [x] Setup wizard server-side persistence
- [x] Model selection persistence (localStorage)
- [x] Credit sparkline deltas, REST poll pausing, CSP header

**Test results**: 226 tests passing (210 unit + 16 integration), 0 failures, clean clippy

---

## Phase 11: Roadmap Blitz

**Goal**: Implement all 32 remaining roadmap items from IDEAS_ROADMAP.md via parallel agent teams.

**Date**: 2026-02-28 | **Method**: Two waves of parallel agents (5 + 7 agents in isolated worktrees)

### Wave 1 — Quick Wins + Core Improvements (18 items, 5 agents)
- [x] A6: Prometheus `/metrics` endpoint (OpenMetrics text format, 5 metrics)
- [x] A7: Structured startup log with resolved config
- [x] A8: Database integrity check at startup (4 critical sled trees)
- [x] B1: Per-request KV-cache isolation (DashMap keyed by request_id)
- [x] B3: Load-aware scheduling via health ping piggybacking
- [x] B5: Manifest versioning with `schema_version` field
- [x] B6: Shard download concurrency cap via Semaphore (default 3)
- [x] C7: JoinSet-based task supervisor with restart-on-crash + exponential backoff
- [x] C8: Startup readiness probe `GET /health/ready` with subsystem status
- [x] D5: Auto-activate relay on NAT detection
- [x] D6: API key copy button in settings UI
- [x] D7: Token counter in chat input (~N tokens, color warnings)
- [x] D8: Filter model selector to ready models only
- [x] D9: WebSocket reconnect indicator banner
- [x] D11: Cancel button for in-progress HF downloads + backend endpoint
- [x] D12: Shard storage cleanup UI + `DELETE /api/admin/models/:id` endpoint
- [x] Backend: `POST /api/admin/downloads/:model_id/cancel` (CancellationToken-based)
- [x] Backend: `DELETE /api/admin/models/:model_id` (disk + DB + SharedState cleanup)

### Wave 2 — Major Features (14 items, 7 agents)
- [x] B2: Multi-turn KV-cache reuse (skip prefill for cached prefix, session_id tracking)
- [x] B4: HuggingFace download resilience (resume from partial, Range headers, Retry-After parsing)
- [x] B7: Config hot-reloading via SIGHUP + `POST /api/admin/config/reload`
- [x] B8: VRAM-aware split_models LRU cache eviction (SplitModelEntry, configurable budget)
- [x] B9: Persist shard_range to database (CLI override, restore on restart)
- [x] C1: Sybil resistance via Ed25519-signed balance reports (freshness check, weighted scoring)
- [x] C2: Reputation scoring (TrustManager, 5 event types, decay toward 0.5, sled persistence)
- [x] C3: Forward secrecy with ephemeral ECDH (per-session X25519, automatic zeroization)
- [x] C4: Credit escrow for large requests (create/release/refund lifecycle, 10min expiry)
- [x] C5: Speculative decoding (draft model + rejection sampling, 2-3x throughput)
- [x] C6: True batched split inference (BatchForwarder, stacked tensors, per-request attention)
- [x] D1: Privacy-preserving pool membership (blind signatures, BLAKE3 commitments)
- [x] D2: Leaderboard spoofing protection (min lifetime + verified tx count filters)
- [x] D10: Mobile-responsive layout (hamburger menu, media queries, 44px touch targets)

### New modules created:
- `src/api/metrics.rs` — Prometheus metrics
- `src/credit/trust.rs` — TrustManager with reputation scoring
- `src/credit/escrow.rs` — EscrowManager with credit escrow lifecycle

**Test results**: 340 tests passing (324 unit + 16 integration), 0 failures, clean clippy
