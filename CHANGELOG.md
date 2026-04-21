# Changelog

All notable changes to SwarmLLM are documented here.

## [0.1.0] - 2026-04-21

> First public release. 765 tests, all 20 build phases complete, distributed
> inference speedup arc (Items 1–16) landed. An internal snapshot was taken
> 2026-03-18 with 674 tests; the sole prior tag was `v0.1.0-alpha.1`. The
> work below is the full surface of this release.

### Distributed Inference Speedup Arc

A multi-session effort to speed up distributed inference, tracked in
`docs/plans/distributed_inference_speedup.md`. Items 1–16 numbered in
plan order; default-on items landed as they shipped, flag-gated items
are off until benchmarked on real workloads.

**Default-on stack (user-facing in [Performance chapter](docs/book/src/operations/performance.md)):**

- **Item 3 — Continuous batching** (2026-04-19): fused `forward_batch`
  over concurrent Generate requests. 1.34–1.55× GPU throughput at batch
  2–8. CPU falls through to sequential with no regression.
- **Item 4 — Remote-generate fast path**: single-segment distributed
  inference runs the full decode loop on the remote worker instead of
  per-token coordinator round-trips. **1.93× decode speedup**.
- **Item 5 — Cross-request prefix cache**: worker keeps an LRU of prefill
  KV snapshots keyed by prompt prefix. **29.4× wall-clock** on
  re-submission of the same 513-token prompt.
- **Item 7 Phase 1+2 — BatchGenerate + Sarathi chunked prefill**
  (2026-04-19): SlotTable admits concurrent requests, each Prefilling
  slot advances by `prefill_chunk_tokens` (default 128) per decode tick.
  **17–23× TTFT fairness** at concurrency 2/4/8 on RTX 3070 +
  TinyLlama Q4. See `docs/plans/benchmarks/round4.md`.
- **Item 7 Phase 4 — Batched prefill forward** (2026-04-19): fuses
  concurrent same-shape prefill chunks into one `forward_batch`.
  **1.57× aggregate tok/s @ c=4** with uniform 180/180/180 ms TTFT.
  See `docs/plans/benchmarks/round5.md`.
- **Item 8 — Cross-node prefix KV sharing** (2026-04-19/20): when node B
  receives a prompt whose prefix peer A already prefilled, B fetches
  A's KV snapshot over the wire instead of re-prefilling locally.
  Full pipeline: PrefixCacheAnnounce gossip → cross-node index →
  PrefixFetchProbe → trust-gated SendPrefixKvFetch → BLAKE3 verify →
  NaN/Inf scan → hydrate → suffix-prefill.
  **Measured 12.9× iter-1 TTFT speedup on Qwen-7B CPU-CPU localhost**
  (151.7 s → 11.8 s on a 672-token prompt, Round 6 bench 2026-04-20).
  TinyLlama on GPU is the fast-prefill corner case where the fetch path
  is ~100 ms slower than re-prefilling (28 MB snapshot vs 460 ms
  prefill).
- **Item 16 — Parallax scheduler** (2026-04-18/19): shortest-path DP
  over observed per-layer latencies (EMA over recent forwards), replacing
  the greedy latency-only sort. Phase B.2 cross-gossips top-32 observed
  latencies via `NodeCapability.observed_latencies`. Phase C.2 adds a
  soft acquire/prune bias in `AutoShardManager` driven by a per-shard
  stability counter (≥3 consistent ticks before it acts); hard
  constraints (pinning, trust, VRAM) always win.

**Flag-gated:**

- **Item 2 — Distributed speculative decoding** (`speculative_distributed`):
  draft-target speculation across nodes. 40–52% accept rate in a
  llama-cpp-draft / candle-target pairing.
- **Item 6 — SWIFT self-speculative** (`swift_self_speculative`):
  target model acts as its own draft by skipping a layer range. Shelved
  on CPU until flash-attn-with-mask lands.
- **Item 12 — DSD (decentralized speculative decoding)**
  (`decentralized_spec_decoding`, 2026-04-18): multi-segment pipeline
  with γ-token speculation + KV truncation primitives + ~410 LOC
  coordinator loop in `pipeline/dsd.rs`. End-to-end WAN benchmark
  pending.
- **Item 13 — Activation compression Q8_0** (`activation_compression`):
  intermediate pipeline hidden states quantized to Q8_0 on the wire.
  ~3.76× compression, RMS error <0.005. End-to-end multi-segment
  benchmark pending.
- **Item 1 — Persistent pipeline stream** (`persistent_pipeline_stream`):
  one long-lived libp2p bidirectional stream per pipeline session.
  Wire-verified; no measured latency win because the bottleneck was
  elsewhere (Items 4 + 7 solved it).

### Round 6 Bench Findings (2026-04-20)

The Item 8 two-daemon loopback bench caught three wire bugs before the
measured numbers above landed:

1. `SwarmMessage::PrefixCacheAnnounce` missing from the `TOPIC_MODELS`
   arm in `NetworkManager::handle_broadcast` — Phase 1 announces
   silently dropped at the gossip layer. Loopback self-index path
   masked it in single-node tests.
2. `WorkerMsg::PrefixSnapshotResponse` / `DaemonMsg::PrefixFetchResult`
   carried `payload: Option<Vec<u8>>` inside the JSON-framed IPC header.
   `serde_json` encodes `Vec<u8>` as a JSON array of integers (~5× size
   bloat), so a 28 MB snapshot became a ~102 MB header and blew past
   the 64 MiB `MAX_HEADER` cap.
3. Three chained cross-node-fetch timeouts (`PREFIX_FETCH_TIMEOUT_MS=500`
   in the worker, 400 ms daemon network timeout, 500 ms serving-worker
   IPC timeout) were sized for TinyLlama's 28 MB snapshot. A Qwen-7B
   snapshot is 73 MB and takes ~500–1000 ms to serialize+wire — every
   timeout fired and silently converted real hits into misses. Bumped
   to 3000 / 2500 / 2000 ms respectively, keeping the worker timeout as
   the outer bound.

### Code Sweep (105 issues found, 58 fixed)
- **Round 1**: 10 parallel review agents across all 109 .rs files — 68 issues (9 CRITICAL, 32 HIGH, 22 MEDIUM), 41 fixed
- **Round 2**: Second pass — 37 new issues (5 CRITICAL, 22 HIGH, 10 MEDIUM), 17 fixed
- Key fixes: max_seq_len 2048 cap, ShardReader cross-tensor bleed, TensorPayload auth, escrow double-charge, IPC framing overflow, API key leak, Gemma embedding scale in forward_batch, hardcoded sampling in distributed forward

### Credit System Overhaul
- Balanced rates: `rate × tokens` on both earn and spend (no layer multiplier)
- Minimum balance enforcement: `MIN_BALANCE_FOR_INFERENCE = -1000`
- Atomic credit accumulation via `pending_credit_earn` AtomicI64
- Anti-Sybil peer balance deduplication by NodeId
- Priority tiers require positive balance for Gold/Platinum

### Device Pool Invite Codes
- 8-char one-time codes (e.g., `A3F7K2M9`), 24h expiry, Ed25519 signed
- CLI: `swarmllm pool create/invite-code/join/status/leave`
- API: `/api/pool/generate-code`, `/api/pool/join`, `/api/pool/device-name`, `/api/pool/credit-split`

### Pool UX Overhaul
- Device nicknames, online/offline status, per-device stats, combined VRAM display
- QR code for invite codes, credit split configuration (0-50%)
- "My Devices" tab with full management UI

### Terminology Clarification
- "My Devices" vs "Swarm Peers" — clear separation in setup wizard, share popover, dashboard

## [0.1.0] - 2026-03-18

### Release & Scale Readiness (Phase 19)
- **Docker release packaging**: Production `docker-compose.yml` (CPU default, GPU via `--profile gpu`), `.env.example` with all configurable env vars, GitHub Actions CI/CD pushing CPU + CUDA images to GHCR on git tag
- **Docker dev cluster**: 3-node `docker-compose.dev.yml` with static subnet, TCP bootstrap, container-optimized config (`config/docker-cluster.toml`)
- **Setup wizard redesign**: 4 steps → 3 steps (About You → Connect → Ready), invite code paste field, auto-download ON by default, hardware-aware model recommendations based on VRAM, dynamic summary
- **mDNS simultaneous-dial race fix**: When two nodes discover each other via mDNS simultaneously, both connections could fail. Added `pending_redial` queue with hash-based jitter (2-5s) for automatic recovery
- **Upload bandwidth enforcement**: `max_bandwidth_mbps` config now enforced on shard serving with proportional delay (was stored but never applied)
- **Manifest publisher claim**: Copied shard directories now properly gossiped — publisher set to local node_id with manifest hash recomputed on startup shard scan
- **Invite code error messages**: Invalid invite codes now return descriptive 400 errors instead of generic 500
- **Scalability (S1)**: Shard announce delta compression — only broadcasts when shard set changes or periodic re-announce every 10 cycles
- **Scalability (S2)**: P2P shard transfer fallback in auto-manage — when no HuggingFace source known, downloads from peer holders instead of doing nothing
- **Scalability (S3)**: peer_registry capped at 200 entries — evicts highest-latency non-LAN non-pipeline peers when over limit
- **Scalability (S4)**: Gossip broadcast frequency scales with `log(peer_count)` — 30s at ≤10 peers, 120s at 1K, 240s at 10K. Health pings stay at 30s
- **Docker image**: 181MB CPU image (debian:bookworm-slim), multi-stage build, non-root user, health checks
- **Tested**: 5-node Phi-3.5 distribution on Proxmox server (trust promotion, auto-manage, target replicas)

### Local Embedding Privacy
- **Config**: `local_embedding_privacy: true` in `[inference]` — requesting node embeds tokens locally, sends hidden-state activations (not raw token IDs) to remote first-segment nodes
- **LocalEmbedder**: Loaded from shard_000.bin at startup, uses candle for token→embedding conversion
- **Pipeline integration**: Pre-embedded activations skip remote embedding, reducing token exposure to relay nodes

### Deep Code Sweep — 56 fixes across 4 passes (16 parallel review agents)
- **Pass 1 (15 fixes)**: gossip_seal future-epoch removal, manager.rs unwrap safety + shard download cap, AllReduce zstd zip-bomb cap, shard atomic truncate, KV-cache eviction map fix, pipeline pending_vision cleanup + VLM hidden_dim expansion, escrow persist-before-balance, ledger bucket_balance div_euclid, error body truncation, manifest streaming BLAKE3, huggingface u64 range + progress retry, acquisition duplicate guard, protocol pre_embedded defaults
- **Pass 2 (10 fixes)**: sampling order consistency (temperature→top-k→softmax), TP block_in_place wrapper, stale logprobs clear, pending_tensor_channels leak fix (Instant timestamps + periodic sweep), num_layers saturating_sub, protocol unwrap→expect, anthropic proxy error truncation, escrow cleanup balance persist, model_id u16 length guard, max_tokens=0 early return
- **Pass 3 (17 fixes)**: duplicate streaming finish event guard, pending_layer_results failover leak, KV-cache orphan eviction, multi-turn session overwrite prevention, TP tp_size minimum guard (≥2), peer_http_url LAN/Tailscale fix, DB backup-on-upgrade, inverted --shards range validation, tied_output_weight streaming read, pool cosign cryptographic separation, pool_registry cleanup, escrow cleanup count, gossip epoch bound (reject >2 epochs old), invite code decrypt-fail no fallthrough, auto_update download gate, multi-image VLM remote guard, dead code removal
- **Pass 4 (14 fixes)**: completion_tokens EOS corruption (used clean_tokens.len → generated_tokens.len), keystore 0o600 permissions on private key files, WebSocket connection counter RAII guard, credit earn crash-window (single persist after earn+forwarding), MCP error body API key scrubbing, all_shards_available cache cap (1000 entries), TP attention GQA modulo wrap for tp_size > n_kv_head, AllReduce duplicate rank warning, pool state db.remove() on leave (fixes null deserialization type mismatch), health monitor future-timestamp evasion (clamp to zero), supervisor dead code cleanup (MAX_RESTART_ATTEMPTS), GGUF total_size saturating_add (overflow on malicious headers), gossip epoch fallback tightened (3→2 epoch window)
- **Model/storage (6 fixes)**: huggingface total_size==0 guard, retry HTTP status check, atomic tmp+rename for mmproj/header, LoRA rank==0 guard, auto_manage path traversal sanitization

### Feature Wiring — 8 previously unwired features now fully integrated
- **Priority tier enforcement**: `calculate_tier()` with real network percentile from peer credit gossip; `max_concurrent_for_tier()` enforces per-tier concurrent request limits in `drain_queue()`
- **Apply penalty on failure**: Credit penalty (configurable `penalty_serve_failure`, default -50) applied on distributed inference failure; penalty uses `apply_credit_direct` for immediate balance update
- **AllReduce registry cleanup**: `cleanup_stale()` removes entries where the receiver was dropped (timed out), wired into HealthMonitor's periodic 30s tick
- **Pipeline error broadcast**: `broadcast_pipeline_error()` notifies all pipeline participants on distributed inference failure, enabling peers to update shard availability
- **Pipeline affinity (KV cache reuse)**: Multi-turn sessions reuse previous pipeline assignment when all nodes are still connected, avoiding cold KV-cache on every turn
- **Relay service credits**: Tracks relay circuit open/close times in SharedState (`active_relay_circuits` DashMap), accumulates seconds in `relay_seconds_served` atomic counter, drains periodically in CreditLedger to `earn_relay_service()`
- **DHT record verification**: `verify_dht_value()` Ed25519 signature check on all Kademlia `GetRecordOk` results in NetworkManager — unsigned/invalid records are logged and ignored
- **Logprobs in API response**: `sample_token_with_params_and_logprobs()` in tensor_util collects per-token log probabilities via `SamplingContext`, stored in `PipelineExecutor.collected_logprobs` (Mutex), mapped to OpenAI-compatible `ChoiceLogProbs` in the `/v1/chat/completions` response. Works for split model (candle) inference paths

### Security Audit (Phase 16) — ~90 fixes across 5 rounds
- **Round 1-3**: Mandatory gossip signing, transport-authenticated dispatch, RFC 6479 anti-replay, signed DHT records, ephemeral key auth, path traversal fix, HF input validation, constant-time auth, CSP hardening, rate limiter cleanup, queue caps, input limits, WebSocket Origin validation, credit signature verification, XSS fixes
- **Round 4**: StreamingToken auth guard, peer IP bypass scoped to inference paths only, `.env` loader blocks dangerous env vars (LD_PRELOAD/PATH/DYLD_*), TOCTOU guard via `loading_models` DashMap with RAII `LoadGuard`, metadata hostname blocklist (Azure/AWS/DO/Oracle/Alibaba), IPv6 multiaddr extraction
- **Round 5**: All dispatch handlers require `authenticated_sender` (LayerResult, InferenceRequest, PipelineAssignment, InferenceError, TpAllReduceResponse), plaintext fallback removed (seal failure → drop), PEX SSRF filter (private/loopback/link-local IPs), shard serve requires peer_registry membership, pending_tensor_channels capped at 256, pending_tp_partials capped at 512, image_data 20MB cap, PoolMessage identity binding (CreditForward.from_node_id/MemberLeft.node_id must match sender), tool params size limits, lora_adapter validation, peer error body truncation/scrubbing, invite code capped at 4K, MCP research restricted to local/network models

### Frontend Polish
- CSS: removed unused variables, fixed hardcoded colors → CSS vars, removed duplicate rules, `@media (prefers-reduced-motion)`, light theme semantic color overrides
- Accessibility: `role="alert"` + `aria-live` on WebSocket banner, `aria-expanded` on hamburger, `aria-live="polite"` on chat messages, `scope="col"` on table headers
- JS: replaced inline styles with CSS classes, wired aria-expanded toggle

### Bug Fixes (Post-Audit)
- **Critical**: Credit balance overflow (`i64 +=` → `saturating_add`) and missing persistence in `track_forward_participation` — credits now survive daemon restart
- **High**: Divide-by-zero panic from malformed GGUF with `head_count == 0` (4 sites, remotely triggerable via HF probe)
- **High**: Silent null body sent to cloud provider on serialization failure (`unwrap_or_default` → proper error propagation)
- **Medium**: `model_request_counts` DashMap unbounded growth — now gated on registered models only
- **Medium**: `peer_shard_downloads` orphaned entries on peer disconnect — cleanup in ConnectionClosed handler
- **Medium**: Rate limiter cleanup task ignoring shutdown signal — now uses `tokio::select` with `shutdown_rx`

### Infrastructure
- **Workspace migration**: 3-crate Cargo workspace (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`)
- **Ring AllReduce**: Bandwidth-optimal for ≥4 TP ranks, auto-selected by `choose_allreduce_strategy()`
- **Package distribution**: Homebrew formula, AUR PKGBUILD, deb/rpm packages, systemd service file
- **macOS CI**: Re-enabled on macos-15 runner
- **Docker**: Fixed Dockerfiles for workspace build
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### UX & Internationalization
- **i18n** — 20 languages (Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese)
- **Theme toggle** — Light / Dark / System theme with persistent preference
- **Basic/Advanced mode** — Toggle for simplified vs power-user UI
- **Plain-English UX pass** — Removed jargon, clearer labels and error messages for beginners
- **Compare UX** — Prompt textarea moved out of collapsed section, All/Local/Cloud filter buttons, chat source indicators, tok/s display fix for slow models (shows 0.5 instead of 0)
- **Provider UX** — `.env` file support for API keys, key source selector (auto/env/dashboard), error badges with click-to-settings
- **GPU OOM → CPU fallback** — Models that exceed GPU VRAM automatically retry on CPU (split fast-path preserved, not slow pipeline path)
- **Anthropic API model routing fix** — Requests now route to the correct model instead of always using the first loaded model

### Codebase Quality
- **Refactored**: `daemon.rs` (4015 lines → module directory), `admin.rs` (4225 lines → 4 modules), `split.rs` (10K lines → 6 modules)
- **Extracted**: `swarmllm-frontend` crate with dev mode for instant UI changes without full rebuild
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Model Trust & On-Demand Loading (Phase 14)
- **Model Trust System** — demand-driven trust prevents trash models from auto-propagating
  - `ModelTrustLevel` enum: Discovered → Pinned → DemandVerified → NetworkPopular
  - Auto-manage only downloads shards for `DemandVerified`+ or user-`Pinned` models
  - Models promoted to `DemandVerified` after 3 real inference requests
  - `NetworkPopular` promotion when 3+ unique holder nodes serve a model
  - 7-day inactivity decay (Pinned models immune), persisted to redb
  - Trust level exposed in admin API (`trust_level` field on all model objects)
- **On-Demand Shard Loading** — inference requests trigger auto-loading from disk
  - Router detects shards on disk but not loaded in VRAM, triggers `check_and_load_model()`
  - LRU eviction makes room automatically (protected: active pipeline models)
  - Loading coordination via `DashMap<ModelId, Notify>` prevents concurrent loads
  - No more need to pre-load all models at startup
- **Kimi 2.5 support** — `k2*` prefix routing to Moonshot provider
  - Static fallback models: kimi-k2-0527, moonshot-v1-8k/32k/128k
  - Existing kimi* and moonshot-* routing preserved
- **UI improvements**
  - Trust level badges on model cards: Popular (green), Verified (accent), Pinned (yellow), Unverified (gray)
  - HF browser: prominent "On Swarm — N nodes" badge vs "New to network"
  - Download button renamed to "Add to node" (clarifies seed shard semantics)
  - "Local only" indicator when no peers host the model
- **Storage**: `get_all_json()` method on Database for key-value iteration with subkeys
### Claude Code Integration (Phase 13)
- **Full Anthropic Messages API** (`POST /v1/messages`) — complete Claude Code compatibility
  - `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking) request fields
  - `tool_use`, `tool_result`, `thinking`, `redacted_thinking` content blocks
  - `cache_control` on system blocks (Anthropic prompt caching)
  - Full pass-through to Anthropic cloud (all fields preserved including tools and thinking)
  - Anthropic→OpenAI translation proxy for non-Claude cloud models (GPT-4o, DeepSeek, etc.)
  - Tool calls and thinking blocks converted to text for local GGUF inference
  - `ResponseContentBlock` refactored from struct to enum (Text, ToolUse, Thinking variants)
- **MCP `compare` tool** — send same prompt to multiple models concurrently (up to 10)
  - Returns side-by-side results with `content`, `latency_ms`, `input_tokens`, `output_tokens`, `status`
  - Supports local, network, and cloud models in same comparison
  - Routes through `/v1/messages` for consistent routing logic
- **Claude Code as client**: `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`
- **Model Compare dashboard page** — side-by-side multi-model comparison UI with streaming
- 6 new unit tests (tool_use, tool_result, thinking, tools request, response serialization, internal conversion)
- 665 tests passing (597 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Published Benchmark Data
- **GPU (RTX 3070 8GB):** TinyLlama 1.1B 27.2 tok/s, Gemma-2 2B 20.6 tok/s, Phi-3.5 3.8B 46.4 tok/s, Qwen2.5 7B 29.0 tok/s
- **CPU (Ryzen 7 5800H):** TinyLlama 4.2 tok/s, Gemma-2 3.5 tok/s, Phi-3.5 1.8 tok/s, Qwen2.5 2.4 tok/s
- GPU speedups: 6.5x to 25.8x depending on architecture
- Methodology: 100 output tokens, 3-run average, single model loaded, Q4_K_M quantization

## [0.1.0-alpha.1] — 2026-03-07

First public release. Single Rust binary (~31MB) for decentralized P2P LLM inference.

### Inference Engine
- **11 model architectures**: Llama, Llama 4, Qwen2, Qwen 3.5 (hybrid SSM+attention), Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3 (MoE+MLA), GLM-4
- **4 architectures verified** with real models: Llama (TinyLlama-1.1B), Qwen2 (Qwen2.5-Coder-7B), Phi-3 (Phi-3.5-mini), Gemma2 (Gemma-2-2B-IT)
- **Distributed inference** verified on 2-node real LAN (WSL2 laptop + Proxmox server) with 5 models, crash recovery, auto-reconnect
- **Tensor parallelism** via AllReduce (star topology) with RTT-based LAN peer detection
- **VLM support**: LLaVA-v1.5-7B verified end-to-end (CLIP vision encoder + correct fine-tuned text model from second-state/Llava-v1.5-7B-GGUF), distributed mmproj, chat UI image upload (camera button, paste, drag-drop)
- **LoRA adapters**: per-request loading, verified with Qwen2.5-Coder-7B + rank-16 adapter
- **Speculative decoding** with draft model + rejection sampling
- **Cross-request batching** (GPU batch tensors, configurable `max_batch_size`)
- **Multi-turn KV-cache** with session reuse, cross-request prefix caching, chunked prefill
- **Flash attention** (CPU + GPU) and **paged attention** (CUDA block pool)
- **Structured output**: ResponseFormat API with JSON grammar state machine + schema validation
- **Sampling**: temperature, top-k, top-p, frequency/presence penalty, stop sequences

### API & Compatibility
- **OpenAI-compatible API**: `POST /v1/chat/completions` with streaming (SSE), `tool_calls`, `tool_choice`, `logprobs`, `top_logprobs`, Tool role
- **Anthropic Messages API**: `POST /v1/messages` — full Claude Code compatibility (tools, tool_choice, thinking, cache_control, metadata)
- **MCP server** at `/mcp` — `chat`, `models`, and `compare` (multi-model comparison) tools for Claude Code, Cursor, and MCP-compatible agents
- **12 cloud provider fallback**: OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi
- **Hidden states API**: `/v1/internal/hidden-states` for research (activation inspection, adapter insertion)
- **Embeddings**: `POST /v1/embeddings`
- **~62 admin REST routes** for dashboard, config, model management, downloads, providers
- **WebSocket** live updates (2s stats + prune event notifications)
- **Prometheus metrics** at `/metrics` (6 gauges + histogram)

### SDKs & Integrations
- **Python SDK**: `pip install swarmllm-client` — sync + async clients, streaming
- **JavaScript/TypeScript SDK**: zero-dependency, streaming support
- **LangChain integration**: `ChatSwarmLLM` provider
- **LlamaIndex integration**: `SwarmLLM` provider
- **Benchmark CLI**: `swarmllm bench` — sequential latency + concurrent throughput, JSON output

### Networking
- **P2P**: libp2p 0.55 with TCP+Yamux (primary) and QUIC transport
- **5-layer discovery**: mDNS (LAN), persistent peer cache (redb), encrypted invite codes, peer exchange (PEX), Kademlia DHT
- **NAT traversal**: libp2p relay circuits + DCUtR hole punching
- **GossipSub**: 6 topics for shard announcements, credit gossip, health, governance
- **Unified protocol**: `/swarmllm/1.0.0` — JSON control messages + binary tensor payloads (type-tag byte)
- **Wire compression**: zstd for tensor payloads

### Security
- **E2E encryption**: X25519 key exchange + ChaCha20-Poly1305 symmetric encryption
- **Forward secrecy**: ephemeral re-keying with key rotation
- **Sealed gossip**: all gossip messages authenticated (no plaintext fallback)
- **Replay protection**: nonce tracking + rejection
- **Shard integrity**: BLAKE3 content hash verified on every load
- **API auth**: Bearer token with auto-generation, loopback-only key retrieval
- **Provider key security**: at-rest encryption (AES-GCM), zeroize on drop, log scrubbing
- **Content-Security-Policy** header, IP-based rate limiting, CORS lockdown, SSRF protection
- **KV-cache privacy mode**: configurable per-session data isolation

### Model Management
- **Shard-only operation**: nodes download individual shards (~512MB each), never need a full model
- **HuggingFace integration**: search, browse, byte-range shard downloads with resume/retry
- **VRAM-aware auto shard management**: rarity-scored acquisition, popularity-based scoring
- **Smart shard pruning**: auto-remove over-replicated shards based on demand, resource pressure, and region diversity
- **Per-shard lock/pin** and per-model prune toggle
- **BLAKE3 integrity verification** on every shard load

### Credit System
- **Credit ledger**: earn credits by serving inference, hosting shards, seeding data
- **4 priority tiers**: Platinum (top 10%), Gold (top 30%), Silver (positive), Bronze (zero/negative)
- **Dual-signed transactions**: Ed25519 signatures from both parties
- **Credit escrow** for large requests
- **Anti-gaming**: rate limits, spot-check verification, subnet clustering detection
- **Sybil resistance**: trust scoring with decay, reputation tracking

### Identity & Pools
- **Ed25519 cryptographic identity** per node
- **Nicknames** with leaderboard
- **Device pools**: multi-device credit pooling with dual-signature invitation protocol

### Frontend
- **Embedded web dashboard** (vanilla HTML/CSS/JS, no build step, < 200KB)
- **4-step setup wizard** for first-run experience
- **Chat interface**: multi-turn + streaming, switchable Linear/Messenger layout, image upload (camera button, paste, drag-drop) for VLM models
- **Model browser**: HuggingFace search, shard grid visualization, download progress
- **Network map**: peer visualization with region grouping
- **Mobile-responsive** layout with dark theme
- **Reasoning model support**: DeepSeek R1 think token rendering

### Operations
- **Single binary**: ~31MB, zero runtime dependencies
- **CLI**: `run`, `status`, `chat`, `bench`, `peers`, `test-split`, `version`
- **Config priority**: CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Config hot-reload** via SIGHUP or API
- **Graceful shutdown**: SIGTERM handler with subsystem drain
- **Auto-updater**: checks GitHub Releases, downloads + self-replaces with restart prompt
- **JoinSet task supervisor**: automatic restart-on-crash for all 10 subsystems
- **Database**: redb v3 (embedded, ACID, ~15% faster than v2)

### Platform Support
- Linux x86_64 (CPU + CUDA + ROCm)
- macOS aarch64 Apple Silicon (Metal)
- macOS x86_64 Intel (CPU)
- Windows x86_64 (CPU + CUDA)

### Model Loading
- **Auto-extract gguf_header.bin** from shard_000.bin when header is missing (daemon pre-pass)
- **Single-GGUF manifest**: Full GGUF files stored as shard_000.bin generate 1-shard manifests (not split into logical shards)
- **Single-shard mmap fallback**: Models with 1 shard and no tensor entries load via mmap instead of ShardReader
- **Probed flag fix**: Models with gguf_header.bin correctly show as probed in admin API

### Test Suite
- 659 tests: 591 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E
- All passing, clippy clean, rustfmt clean
- CI: GitHub Actions (fmt → clippy → test → build)

[0.1.0-alpha.1]: https://github.com/enapt/SwarmLLM/releases/tag/v0.1.0-alpha.1
