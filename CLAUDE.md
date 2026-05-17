# SwarmLLM — Claude Code Instructions

> **Quick start**: Read `docs/ARCHITECTURE.md` for the canonical architecture (subsystems, channels, SharedState sub-struct layout, protocols, security model) before exploring code. Per-developer dependency notes may live in `~/.claude/projects/-home-user-SwarmLLM/memory/` outside the repo.

## Project Overview

SwarmLLM is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously participates in a P2P network, runs an HTTP server (OpenAI-compatible API + admin dashboard), and manages local resources (GPU/CPU compute, storage, bandwidth).

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.80+
- **Primary Port**: 8800 (HTTP API on TCP:8800, P2P on TCP:8810 + UDP/QUIC:8800)

## Architecture

The daemon spawns 12 subsystems as Tokio tasks wired together with `mpsc` channels:

- **NetworkManager** — libp2p swarm: Kademlia DHT + GossipSub + request_response
- **InferenceRouter** — request queuing, pipeline assembly, execution coordination
- **MessageDispatcher** — routes inbound network messages to appropriate subsystems
- **CreditLedger** — local credit balance tracking, transaction signing, gossip
- **HealthMonitor** — periodic health pings, rebalancing triggers
- **ShardRebalancer** — shard redistribution on node join/leave events
- **AcquisitionManager** — BLAKE3-verified model download from network peers
- **ApiServer** — Axum HTTP: OpenAI + Anthropic APIs + MCP server + admin dashboard + WebSocket
- **PoolManager** — device pool management, credit forwarding, invitation protocol
- **AutoShardManager** — VRAM-aware automatic shard acquisition + smart pruning of over-replicated shards
- **HfWatcher** — R112: hourly HuggingFace trending-GGUF poll, seeds wishlist + auto-promotes models above download/age thresholds to `DemandVerified`
- **UpdateChecker** — periodic GitHub release polling, SHA256-verified binary download, atomic apply

Shared state lives in `Arc<SharedState>` with `DashMap` for concurrent access. SharedState is organized into 4 logical sub-structs:
- `state.events` (`EventBus`) — `activity_tx`, `activity_history`, `dashboard_tx`, `update_state`, `ws_tickets`
- `state.credits` (`CreditPool`) — `credit_balance`, `pool_state`, `pool_registry`, `pool_tx`, `trust_manager`, `escrow_manager`, `anti_gaming`, `private_mode`, `offline_mode`, etc.
- `state.models` (`ModelMgmt`) — `acquisition_progress`, `hf_sources`, `auto_manage_*`, `contribution_auto` (R121 — AtomicBool mirror of `config.node.contribution_auto`, read by prune.rs each tick), `model_trust`, `locked_shards`, `prune_history`, `wishlist` (R111), `hf_trending_cache` (R112), etc.
- `state.metrics` (`MetricsProviders`) — `node_stats`, `inference_requests_total`, `channel_metrics`, `providers_config`, `swarm_capacity` (R110), `hedge_tracker` (R136 Layer 2), `prefetch_orchestrator` (R136 Layer 3), etc.

Cross-cutting fields (config, identity, db, peer_registry, model_registry, executor, split_models, etc.) remain on the root struct.

## Build Phases

All 20 phases complete. See `docs/ARCHITECTURE.md` for full phase history. Deferred items documented there.

## Repository Structure

```
swarmllm/
├── Cargo.toml / Cargo.lock / build.rs
├── .env.example                       (env var template for Docker deployments)
├── config/default.toml, docker-cluster.toml
├── crates/
│   ├── swarmllm-frontend/  (embedded + dev-mode frontend asset serving)
│   └── swarmllm-types/     (shared types crate: NodeId, ModelManifest, SwarmMessage, etc.)
├── src/
│   ├── main.rs, lib.rs, error.rs, http.rs, types.rs, update.rs
│   ├── bin/       (launcher.rs — Windows GPU/CPU auto-selecting launcher)
│   ├── cli/       (mod, run, status, chat, bench, peers, pool, split_test, update)
│   ├── config/    (mod, providers, credit, network, ops, node, inference)
│   ├── daemon/    (mod, manifest, shard_loader, dispatch/, startup, background, helpers, supervisor)
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, tp_allreduce)
│   ├── network/   (manager/{mod,events,requests,tensors,identify,commands,connections,dht,shard_transfer}, behaviour, discovery, protocol, transport, relay, peer_cache, helpers, pipeline_stream)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface/, auto_manage/, lora)
│   │   ├── auto_manage/  (mod, manager, scoring, download, prune, scan, vram, parallax, wishlist)
│   │   └── huggingface/  (mod, download, private_types, probe, search, shards, watcher, tests)
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, local_embedder, model_worker, process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3))
│   │   ├── router/       (mod, types, batch, local_exec, distributed_exec, spot_check, tests)
│   │   ├── scheduler/    (mod, parallax, parallax_allocator, tests)
│   │   ├── pipeline/     (mod, distributed, dsd, local, prompt, remote_generate, speculative, tensor_parallel, vision)
│   │   ├── split/        (mod, model, loader, executor, kv_cache, entry, gguf_meta, shard_reader, rope, prefix_cache, tests/)
│   │   │   └── tests/    (mod, common, core, gqa, gemma2, moe_mla, llama4_glm4)
│   │   ├── chat_template/ (mod, parser, eval, fallbacks, tests)
│   │   └── layers/       (mod, qwen35)
│   ├── credit/    (ledger, transaction, priority, anti_gaming, trust, escrow)
│   ├── identity/  (keypair, nickname)
│   ├── crypto/    (session, pipeline_seal, gossip_seal, key_rotation, provider_keys)
│   ├── pool/      (types, crypto, manager/, forward, scope)
│   ├── api/       (server, sse, admin, admin_providers, websocket, middleware, identity, pool, metrics, providers, claude_sub*, mod, openai/, anthropic/, mcp/, admin_hf/, admin_models/, claude_session/)
│   ├── storage/   (db)
│   └── health/    (monitor, rebalancer)
├── frontend/      (index.html + 11 HTML templates, css/, js/{core/4,components/17,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/)
├── python/        (swarmllm-client SDK)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── docs/book/     (mdBook documentation site)
└── tests/         (integration tests)
```

## Key Dependencies

libp2p 0.56, axum 0.8, candle-core/candle-transformers 0.10 (CUDA), redb 4, ed25519-dalek 2, x25519-dalek 2, chacha20poly1305, blake3, dashmap 6, clap 4, tracing, reqwest, zstd. See `Cargo.toml` for full list.

## Coding Conventions

### Error Handling
- Use `thiserror` for defining error types in `src/error.rs` (SwarmError enum)
- Use `anyhow` only in `main.rs` and integration tests
- Map SwarmError variants to HTTP status codes via `ApiError` wrapper
- Variant → status contract (see `.claude/rules/completeness.md`):
  - `Validation` → 400 (API input)
  - `ModelNotAvailable` / `ShardNotFound` / `NotFound` → 404
  - `Config` → startup ONLY
  - `Internal` → actual bugs (500)
  - `ProviderError { status, body }` → upstream cloud errors (preserves status)
  - `ServiceUnavailable` → THIS server can't serve (503), NOT upstream
- Network errors: retry with exponential backoff (3 attempts)
- Inference errors: return immediately, never retry silently
- Shard integrity errors: quarantine shard, re-download, penalize peer trust
- Credit errors: degrade priority tier, never block

### Naming
- Types: `PascalCase` (e.g., `NodeId`, `ModelManifest`, `PipelineSegment`)
- Functions/methods: `snake_case`
- Newtype wrappers for type safety: `NodeId([u8; 32])`, `ModelId(String)`, `ShardId { model_id, index }`
- Short display for NodeId: first 8 bytes hex-encoded

### Serialization
- HTTP API: `serde_json` (match OpenAI format exactly)
- Network protocol: Unified codec — `serde_json` for control messages, binary with type-tag byte for tensor payloads
- Config: TOML via `toml` crate
- Database values: `serde_json` serialized into redb

### Async Patterns
- All subsystems communicate via `tokio::sync::mpsc` channels
- SharedState fields use `DashMap` for concurrent reads or `RwLock` for single-value state
- Graceful shutdown via `tokio::sync::watch` channel
- Use `tokio::select!` in daemon/mod.rs to wait for shutdown or task exit

### Logging
- Use `tracing` with structured spans (include context like peer_count, request_id, model_id)
- Target format: `swarmllm::module::submodule`
- Verbosity levels: info (default), debug (-v), debug+libp2p (-vv), trace (-vvv)
- Key metrics: peers.connected, inference.requests, inference.latency_ms, credits.balance, shards.hosted

### Frontend
- Vanilla HTML/CSS/JS — no framework, no Node.js build step
- Embedded into binary via `include_dir!` macro at compile time
- Component architecture: `App` global namespace, 26 JS files (4 core + 17 components + init.js + 4 standalone utilities)
  - `js/core/` — state.js (namespace + shared state + storage keys), utils.js (format helpers, DOM builders, extractErrorMessage, getApiErrorMessage, apiAction), data.js (data store + authFetch + dedup), tooltip.js (unified popover replacing native `title=`)
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, dashboard-shards.js (pure shard HTML builders exposed as `App.dashboardShards`), models.js, auto-manage-status.js, settings.js, setup.js, welcome.js (R127 — first-run tour modal), downloads.js, notifications.js, identity.js, network-map.js, compare.js, responses.js, pool.js, swarm-tab.js (R111 — wishlist + capacity-plan view)
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 12 HTML modals/templates incl. R127 `#welcome-modal` (first-run tour). 11 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1154 translation keys (1156 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key for parity audits. R110-R115 translations completed in R116, contribution-mode (R121) keys added across all locales, plain-language refresh (R125 ease-of-use audit) translated across all 21 locales — translator-agent pass — every locale has idiomatic native-language strings, not English fallback. R126 batch: removed dead `activity.worker_*` + `models.meta_tokenizer`, refreshed encryption copy (`enc.*` ×19 keys, end-to-end honest), added `activity.manifest_rejected` + `models.meta_advanced`, renamed `models.metadata_header` to "Technical Details". R127 batch: dropped 4 orphans (`models.hf_score_breakdown`, `models.hf_score_pts`, `models.hf_on_swarm`, `models.likes_count`); translated `dashboard.api_log_link` across 21 locales; country names now resolved via `Intl.DisplayNames` keyed off `I18n.getLang()` (no hand map).
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 996 lib tests passing + 8 ignored (env-var-gated real-model + manual smoke), 75 integration tests in `tests/integration/`, 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline — requires `auto_manage.enabled = false` in per-node config.toml to preserve sharded state).
- Unit tests: in-module `#[cfg(test)]` blocks
- Integration tests: `tests/integration/` — multi-node simulations with `--test-threads=1`
- Real-model spawn-and-infer test: set `SWARMLLM_TEST_MODEL_DIR` to a fully-populated model directory (e.g. `~/.local/share/swarmllm/models/tinyllama-1.1b-...`) and run `cargo test --test integration_phase10_11 -- --ignored end_to_end`. No synthetic GGUF fixture is committed; see `docs/ARCHITECTURE.md` § Deferred Items.
- CI pipeline: `cargo fmt` → `cargo clippy --all-targets -- -D warnings` → `cargo test` → `cargo build --release`

## Key Design Decisions

- Config priority: CLI flags > env vars (SWARMLLM_ prefix) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
- Data dir: `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- Port layout: HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- Credit transactions require dual Ed25519 signatures (serving node + requesting node)
- Priority tiers: Bronze (zero/negative) < Silver (positive) < Gold (70th percentile) < Platinum (90th)
- KV-cache sessions expire after 10 minutes of inactivity (configurable)
- Shard verification: BLAKE3 content hash checked on every load
- Pipeline failover: hot-standby nodes pre-identified per segment
- **Private mode**: restricts YOUR outbound inference to pool/LAN nodes only. Nodes still serve the swarm. Single `allowed_node_set()` in `src/pool/scope.rs` gates everything. Runtime-toggleable via `AtomicBool`. Shard pinning lets pool owners assign models to devices.
- **No full model download required**: A node NEVER needs the full GGUF or all shards to participate in inference. Shards are downloaded individually via byte-range requests. Downloading all shards (or a full model) is opt-in only — for users who want offline inference or to seed more shards to the network. Never add code that implicitly downloads a full model or reconstructs a GGUF from shards. All inference loads from shard files + gguf_header.bin.

## Subagent Choices for This Codebase

When spawning subagents in this repo, use these model picks (overrides defaults that would otherwise pick haiku):
- `Task(feature-dev:code-reviewer)` → sonnet (this codebase's invariants need real reasoning, not pattern-matching)
- `Task(feature-dev:code-architect)` → sonnet
- `Task(Plan)` → sonnet
- Never delegate production code writing — opus (this main session) writes it

## Reference Documents

- `docs/ARCHITECTURE.md` — **Primary reference** — current architecture, subsystems, protocols, security model
- `docs/book/` — mdBook documentation site (getting started, API reference, architecture, troubleshooting)
- `docs/DIAGNOSTICS.md` — DIAG: log instrumentation guide for debugging
- `.claude/rules/architecture.md` — invariants (SharedState, broadcast channels, scheduler oracle, centralised wire-format helpers)
- `.claude/sweep-log.jsonl` — per-finding history of every `/sweep` round (status: fixed / wontfix / deferred). Grep before re-reporting potential issues.
- `SwarmLLM_Technical_Specification.docx` — High-level technical specification with architecture rationale

## Status

All 20 build phases complete. All subsystems wired — no stubs. 996 lib tests + 75 integration tests passing; 8 lib + 1 e2e ignored (env-var or manual). Latest: **R136 — SWARM-SPEC v0.1 + 3-node A/B benchmark** (state-of-the-art P2P-native inference acceleration). Real-inference numbers on a 3-node loopback cluster (TinyLlama 1.1B Q4_K_M, CPU-only). **Single-node routing** (all 3 hold model): Q8_0 ON medians 4.3 / 4.1 / 5.2 vs OFF 4.0 / 3.5 / 5.0 tok/s for code / summary / chat → Layer 0 wins 4-17%. **Forced distributed B→C pipeline** (auto-manage off, A=manifest only, B/C each hold 1 shard): Q8_0 ON 4.0 / 2.75 / 4.5 vs OFF 3.5 / 2.9 / 4.8 → honest finding that **Q8_0 doesn't win on loopback** because the wire is essentially free and the encode CPU (~17µs/forward) rivals the saved wire-time. Synthetic prediction of 1.5-1.7× only applies on bandwidth-bound WAN hops (10-100ms RTT). Default-ON remains correct for production (real SwarmLLM target is geo-distributed P2P); loopback / single-host operators should override per-node. Distributed-pipeline measurement methodology + scripts (`examples/3node_{setup,sharded_setup,inference_bench}.sh`) + `GET /api/admin/stats` swarm_spec block all shipped. Layer 1 (n-gram) DRAFT-FREE path ships via `try_ngram_only_distributed` — works without a draft model, single-segment AND multi-segment via shared `forward_verify_through_segments`; Layer 2 (hedging) ships dry-run logging — true duplicate-dispatch awaits a focused follow-up; Layer 3 (prefetch) ships observation-only — first-token observation + idle-time learning wired, K-layer prefetch dispatch is the next integration. **Layer 1 measured on real inference** (3-node TinyLlama 1.1B Q4, CPU). **Single-segment routing** (B/C each hold full model, 1-hop pipeline): baseline 4.0/2.75/4.5 tok/s → L1 active 4.3/**4.0**/~4.5 → summarisation +45% on 77% n-gram hit-rate. **Multi-segment sharded** (A=manifest, B=shard_000, C=shard_001, true 2-hop production scenario): L1 fires on every request via `forward_verify_through_segments`; 3.7/3.3/4.7 tok/s, hit-rates 35.7/8-77/17.2% — multi-segment per-token cost higher (2 hops vs 1) but **now works for real sharded clusters**. Bug fix during integration: DSD's intermediate-shape-check incorrectly fired on segment-0 transitions (input tokens vs output hidden state), gated on `idx >= 1`. **Dispatch order**: L1 (`try_ngram_only_distributed`) now precedes `try_remote_generate_fastpath` so the multi-token-per-round acceptance on hits wins over remote_generate's one-token-per-RTT throughput; on miss the in-L1 single-token fallback has comparable cost. Layer 1 requires the standalone tokenizer (lazy-loaded from `gguf_header.bin` into `state.standalone_tokenizers` — also feeds Layer 3 first-token observation). The 4-layer cascade is purpose-built for distributed inference where compute is cheap and network is the bottleneck: **Layer 0** activation compression Q8_0 default-on with per-block group-32 scale (17µs round-trip, 3.76× wire reduction, < 1% perplexity drift); **Layer 1** n-gram prompt-lookup speculation cascade inside existing speculative + DSD paths (30µs lookup, 99% hit on code / 96% on RAG / 0% on chat — correctly falls through to SWIFT/DSD); **Layer 2** adaptive hedging decision logic + per-(model, segment, holder) EWMA tracker with rate budget (140ns decision, dispatch deferred); **Layer 3** predictive prefetch scaffolding with per-session history learner + idle-time predictor (0.3µs decision, response-completion already wired, dispatch deferred). Bench harness: `examples/swarm_spec_bench.rs`. Zero new dependencies. Predicted end-to-end speedup when full cascade is wired: Claude Code 5.5×, code completion 6.5×, RAG 4.5×, chat 3.5×. Detail: `docs/FUTURE_WORK.md § R136`. Prior: **R134.7** — cross-pool inference routing (FUTURE_WORK closure, opt-in): `pool.allow_cross_pool_inference` flag (default false) gates routing through foreign pools' advertised model catalogs; new `pool::scope::cross_pool_extras(state, &model_id)` unions foreign-pool NodeIds with the local `allowed_node_set` only when the local pool can't serve the model. Requires both sides to opt in (foreign publishes via `share_model_catalog`, local sets `allow_cross_pool_inference`). Also R134.7: predictive eviction via time-windowed protection in `prune.rs` (`RECENT_REQUEST_PENALTY = 1.5` for models served within the last hour, beats `region_demand`). **R134.6** — quant auto-action layer: `apply_quant_auto_action` promotes recommended quant's `ModelTrustInfo` to `DemandVerified`; opt-in via `auto_manage.auto_switch_quants` (default false). **R134.5** — inter-pool catalog tile in Running-now subview, +3 i18n keys × 21 locales. **R134** — multi-feature closure: inter-pool model availability discovery (`SwarmMessage::PoolModelAvailability` + `state.credits.foreign_pool_catalog: DashMap<(PoolId, ModelId), received_at_ms>` capped 5000, 2h freshness; opt-in via `pool.share_model_catalog`, k-anonymity floor via `share_model_catalog_min_members` default 3); HF anti-gaming reputation (`ModelTrustInfo.{last_auto_promoted_at, failed_promotions}` + linear cooldown 7×strikes days, capped 60d, lockout after `MAX_AUTO_PROMOTION_FAILURES = 4`); pool-state diff gossip wire format (`PoolMessage::StateDiff` with `PoolState.generation` counter, opt-in via `pool.state_diff_gossip` default false, `MAX_DIFFS_BEFORE_FULL = 4` forces fresh broadcast); quant recommendation tips frontend tile (`#quant-tips` in swarm-tab.js, hidden when no actionable hint); +9 i18n keys × 21 locales. **R133** — quant choice automation: `Quantization` enum expanded 5→29 variants (K-quants Q2_K..Q6_K, legacy Q4_0..Q5_1+Q8_0, I-quants IQ1_S..IQ4_NL, floats F16/BF16/F32, Unknown), per-variant `parse/bits_per_weight/quality_score/label`; `model/auto_manage/quant.rs` recommends highest-quality variant fitting local VRAM OR pool-VRAM/3-replica; `state.models.quant_recommendations` ArcSwap refreshed alongside wishlist; new `GET /api/admin/quant-recommendations`. **R132** — MoE per-arch routing: `MoeGatingFunc {Softmax, Sigmoid}` + `MoeRoutingConfig {gating_func, renormalize_weights}`; GGUF loader reads `{arch}.expert_gating_func` + `{arch}.expert_weights_norm`. **R131** — pool-state gossip debounce: 15s min interval + 3s coalesce timer, 50 acceptances → ≤2 broadcasts. **R130** — cross-pool wishlist gossip: `SwarmMessage::WishlistAnnouncement` + `state.models.foreign_wishlist: DashMap<(NodeId, ModelId), (score, ts_ms)>` (capped 10K, 2h freshness); opt-in publishing via `auto_manage.wishlist_gossip_publish`. Deferred items in `docs/FUTURE_WORK.md`; new "Inference performance — research backlog" section added there (R135 brief) with prioritized speedup roadmap (Tier 1 = default-on activation compression + tail-latency hedging + EAGLE-3 draft head).

## Common Commands

```bash
cargo build --no-default-features --features dev,claude-subscription  # Dev build (live frontend + Claude Code)
cargo fmt && cargo clippy --all-targets -- -D warnings  # Lint (MUST pass before push)
cargo test                           # All tests
cargo run -- run -p 8800 -v          # Start daemon
```

**Note:** Always include `claude-subscription` feature when testing Claude Code integration. Bare `--features dev` omits the Claude subscription provider.
