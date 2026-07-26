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
- `state.models` (`ModelMgmt`) — `acquisition_progress`, `hf_sources`, `auto_manage_*`, `contribution_auto` (R121 — AtomicBool mirror of `config.node.contribution_auto`, read by prune.rs each tick), `model_trust`, `locked_shards`, `prune_history`, `wishlist` (R111), `hf_trending_cache` (R112), `shard_download_backoff` (per-shard exponential download cooldown so one stuck download can't monopolize a slot), etc.
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
│   ├── cli/       (mod, run, status, chat, bench, peers, pool, split_test, update, get_model — R150 `swarmllm get-model` reference-model opt-in)
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
│   ├── api/       (server, sse, tool_parse (local-model tool-call parser), admin, admin_providers, websocket, middleware, identity, pool, metrics, providers, claude_sub*, mod, openai/, anthropic/, mcp/, admin_hf/, admin_models/, claude_session/)
│   ├── storage/   (db)
│   └── health/    (monitor, rebalancer)
├── frontend/      (index.html + 10 HTML templates, css/, js/{core/4,components/19,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/)
├── python/        (swarmllm-client SDK)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── deploy/anchor/ (R143 — hardened bootstrap/relay anchor kit: setup-anchor.sh, systemd unit, config.toml, runbook)
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
- Component architecture: `App` global namespace, 28 JS files (4 core + 19 components + init.js + 4 standalone utilities)
  - `js/core/` — state.js (namespace + shared state + storage keys), utils.js (format helpers, DOM builders, extractErrorMessage, getApiErrorMessage, apiAction), data.js (data store + authFetch + dedup), tooltip.js (unified popover replacing native `title=`)
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, dashboard-shards.js (pure shard HTML builders exposed as `App.dashboardShards`), models.js, auto-manage-status.js, settings.js, setup.js, welcome.js (R127 — first-run tour modal), downloads.js, notifications.js, identity.js, network-map.js, compare.js, responses.js, pool.js, swarm-tab.js (R111 — wishlist + capacity-plan view), reference-models.js (R148 — shared test-model picker, `App.referenceModels`)
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 3 modal overlays (setup, settings, R127 `#welcome-modal` first-run tour) + 10 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1220 translation keys (1222 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1221 lib tests passing + 8 ignored (env-var-gated real-model + manual smoke), 79 integration tests in `tests/integration/` + 2 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 2 repo-consistency, 26 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline — requires `auto_manage.enabled = false` in per-node config.toml to preserve sharded state; splits any shard count across B/C, not just 2). Both scripts take `SWARM_BENCH_MODEL`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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
- **Encryption — two layers, distinct concerns:**
  - **Layer 1 — `network.enable_encryption` (DEFAULT TRUE).** ChaCha20-Poly1305 sealing of activations between hops via per-session X25519 ECDH. Every inter-node tensor forward is encrypted on the wire. AAD covers cleartext header + spec/kv-truncate/chunk-meta trailers (`build_layer_forward_aad` is the single source of truth). On the receiver side, decryption is offloaded from the NetworkManager event loop via `tokio::spawn` (R139 Phase C). Failure is hard: there is NO plaintext fallback on `seal()` failure — the forward is dropped with `LayerResult::error`. Disabling this flag is only sensible for local-loopback debugging.
  - **Layer 2 — `inference.encrypted_pipeline` ("boomerang", DEFAULT FALSE, per-model override).** Forces the local node to handle BOTH the first segment (embedding) AND the last segment (sampling). Remote nodes only see intermediate encrypted hidden states — no remote node ever sees the plaintext prompt OR the sampled tokens. Requires the local node to hold shard 0 + final shard. Adds ~1 RTT/token. This is the strongest privacy mode; Layer 1 alone leaves entry/exit nodes able to read the cleartext at their boundary.
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

All 20 build phases complete. All subsystems wired — no stubs. **1221 lib + 79 integration + 2 repo-consistency + 26 swarmllm-types tests passing**; 8 lib + 2 e2e ignored (env-var or manual). Clippy clean default + features dev,claude-subscription + `--features llama`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.27-alpha (2026-07-26): provider routing + metadata fixes

Three findings from tester checklist sections 11/12. All three are the same
shape as the rest of this week — a rule applied on one path but not another.

- **`provider:model` prefix forwarded upstream.** The prefix selects the
  provider; the provider has never heard of it. `strip_provider_prefix` existed
  and ONLY `anthropic/convert.rs` called it, so the OpenAI-compatible proxy sent
  the body verbatim and DeepSeek rejected `deepseek:deepseek-v4-flash`. Stripped
  at the proxy boundary (`providers.rs::strip_prefix_in_body`, Cow — clones only
  when a prefix is present).
- **`PUT /api/admin/providers` silent no-op.** Fields are `<provider>_key`; no
  `deny_unknown_fields`, so a wrong name → all `None` → nothing written → still
  `status: "ok"`. Now returns Validation naming the expected shape.
- **Quantization mislabelled `Q4KM`.** Ids carry `q8-0` (sanitiser rewrites
  `_`→`-`) but `Quantization::parse` stripped only `_`, AND `trailing_tag` took
  only the last segment (`"0"`). Both fixed — `trailing_tag` now tries the last
  1–3 segments, longest first.

### Recent releases (one line each; full detail in CHANGELOG.md + `memory/round_log_*.md`)

**This week was dominated by one recurring defect — a shared invariant
implemented per-path — documented as a rule in `.claude/rules/architecture.md`.
Read that before touching any request/response path.**

- **v0.3.26** (07-26): control-token scrubbing at the THREE text sources
  (`executor.rs` / `process_pool.rs` / `pipeline/distributed.rs`) — closed a
  template leak reported across .22/.24/.25; strip BEFORE stop-truncation.
  Gotchas #167-168. Detail: `round_log_overnight_0726.md`.
- **v0.3.25** (07-25): template stop-markers on the non-streaming split path;
  `include_usage` on `split_stream_response`; Responses API tool cap (128);
  dashboard chat single silent retry on a cold model.
- **v0.3.24** (07-25): tool-call parser accepts prose around the JSON
  (`parse_embedded_json`, balanced-brace scan); universal `<|...|>` stop markers.
- **v0.3.23** (07-25): **the big networking fix** — we recorded an INBOUND
  connection's ephemeral source port as a dialable address and published it, so
  peers learned dead addresses for us. One bug, four symptoms (all dials refused
  → no circuit → DCUtR 0 attempts → absent from `connected_node_ids` → "no node
  available"). Gotcha #165. Poisoned entries persist in caches — nodes must
  RESTART, not just swap binaries.
- **v0.3.22** (07-25): local tool calling implemented (`src/api/tool_parse.rs`,
  4 formats, all four streaming paths); recent-failures diagnostics ring;
  serving-side lost-reply logging.
- **v0.3.21** (07-25): networking audit — `max_established_per_peer = 1` was
  structurally disabling DCUtR (gotcha #163); scheduler relay tier;
  `relay_forwarding_auto`; `max_circuits` 16→128. **Hole punching verified live
  07-25 16:15Z.** Detail: `round_log_networking_audit.md`.
- **v0.3.18-v0.3.20** (07-24→25): the networking release line — app-level
  inference relay + tensor relay across NAT, additive protocol/feature
  handshake, multi-relay + DHT relay discovery, auto-updater size cap
  500MB→2GB (GPU builds), deterministic LAN dialer. Detail:
  `round_log_networking_plan.md`.
- **v0.3.15-v0.3.17** (07-23→24): external-tester arc — WSL2 mirrored-mode
  detection, Docker bridge filtering, FULLBACKUP lockdown, Claude Code
  tool-description cap 4096→32768, OpenAI model-echo. Detail:
  `round_log_v0315_livetest.md`.


### Prior rounds (one line each; full detail in `memory/round_log_*.md`)

- **R150** (07-23): GPU coverage — candle `CUDA_COMPUTE_CAP` 80→75 (PTX floor: RTX 20-series → Blackwell), llama.cpp `CMAKE_CUDA_ARCHITECTURES` pin + CUDA 12.8 native sm_120. Gotchas #159-160.
- **R149** (07-23): AutoShardManager per-shard exponential download backoff (`shard_download_backoff` 30→300s); hourly negative-balance decay; portable Linux binary (ubuntu-22.04 / glibc 2.35).
- **R148** (07-22): stale shard-holder retraction (`complete_for_models`); escrow estimate→actual; VLM chat-template empty-prompt + CLIP `ffn` per-file (`clip_ffn_is_swapped`); updater CUDA-variant match. v0.3.6 → v0.3.11.
- **R147** (07-22): request cancellation end-to-end (`DaemonMsg::CancelRequest`, `ResponseGuard.disarm`); FUTURE_WORK closures. GPU-verified on RTX 3070.
- **R146** (07-22): TP-group-on-local-model hard-fail → `inference.tensor_parallel` default false + graceful degrade; worker VRAM leak → `worker_error_is_fatal`; `gpu_layers` plumbed (default 0→-1); `failure_is_penalty_worthy`.
- **R145** (07-21): cloud model refresh — Opus 4.8 / Sonnet 5 / +Fable 5, Kimi K3, Moonshot `.ai`, DeepSeek v4.
- **R143-R144** (07-20→21): internet reachability / NAT — UPnP default-on, external-address invite codes, AutoNAT v1→v2, `--anchor` mode + `deploy/anchor/` kit; dashboard peer taxonomy (`multiaddr_is_local`, Pool > LAN > Remote).
- **R140-R142** (05-22→07-19): `swarmpool://` invite codes v2; auto-manage cold-start UX; autonomous 8h sweep (3 silent frontend↔backend wire-format prod bugs).
- **R136-R139** (05): SWARM-SPEC v0.1 acceleration cascade (L0 Q8_0, L1 n-gram, L2 hedge, L3 prefetch); Tier 4K comm-compute overlap; FUTURE_WORK deferral batches. See `round_log_R126_R137.md`, `round_log_R139.md`.
- **Pre-R136**: 20 build phases (P2P/libp2p, split + distributed inference, credits, pools, OpenAI+Anthropic+MCP API, frontend, VLM, Claude Code integration). See `docs/ARCHITECTURE.md` § phase history + `round_log_R126_R135.md`.

## Public-Facing Repo (2026-07-22)

The repo is public and a **GitHub webhook relays activity to the project Discord** —
every commit and push is broadcast to real users, including non-technical ones
deciding whether to run this software. Commit subjects must stand alone in a feed
with no context; lead with user-visible impact before mechanism; never name a person
or paste private correspondence; get sign-off before force-pushes or history rewrites
(they surface in the feed and look like something broke). Full guidance in
`.claude/rules/workflow.md` § "Pushes are public-facing".

## Common Commands

```bash
cargo build --no-default-features --features dev,claude-subscription  # Dev build (live frontend + Claude Code)
cargo fmt && cargo clippy --all-targets -- -D warnings  # Lint (MUST pass before push)
cargo test                           # All tests
cargo run -- run -p 8800 -v          # Start daemon
```

**Note:** Always include `claude-subscription` feature when testing Claude Code integration. Bare `--features dev` omits the Claude subscription provider.
