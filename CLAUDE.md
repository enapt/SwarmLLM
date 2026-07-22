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
├── frontend/      (index.html + 11 HTML templates, css/, js/{core/4,components/18,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/)
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
- Component architecture: `App` global namespace, 27 JS files (4 core + 18 components + init.js + 4 standalone utilities)
  - `js/core/` — state.js (namespace + shared state + storage keys), utils.js (format helpers, DOM builders, extractErrorMessage, getApiErrorMessage, apiAction), data.js (data store + authFetch + dedup), tooltip.js (unified popover replacing native `title=`)
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, dashboard-shards.js (pure shard HTML builders exposed as `App.dashboardShards`), models.js, auto-manage-status.js, settings.js, setup.js, welcome.js (R127 — first-run tour modal), downloads.js, notifications.js, identity.js, network-map.js, compare.js, responses.js, pool.js, swarm-tab.js (R111 — wishlist + capacity-plan view)
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 12 HTML modals/templates incl. R127 `#welcome-modal` (first-run tour). 11 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1186 translation keys (1188 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key for parity audits. R110-R115 translations completed in R116, contribution-mode (R121) keys added across all locales, plain-language refresh (R125 ease-of-use audit) translated across all 21 locales — translator-agent pass — every locale has idiomatic native-language strings, not English fallback. R126 batch: removed dead `activity.worker_*` + `models.meta_tokenizer`, refreshed encryption copy (`enc.*` ×19 keys, end-to-end honest), added `activity.manifest_rejected` + `models.meta_advanced`, renamed `models.metadata_header` to "Technical Details". R127 batch: dropped 4 orphans (`models.hf_score_breakdown`, `models.hf_score_pts`, `models.hf_on_swarm`, `models.likes_count`); translated `dashboard.api_log_link` across 21 locales; country names now resolved via `Intl.DisplayNames` keyed off `I18n.getLang()` (no hand map). R146 batch: added `leaderboard.balance_unknown`, `activity.model_cpu_fallback`, `hw.vram_live_tip`, `hw.vram_live_only_tip`; dropped `hw.vram_active` + `hw.vram_reserved_tip` (the estimated-VRAM display they belonged to is gone).
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1118 lib tests passing + 8 ignored (env-var-gated real-model + manual smoke), 75 integration tests in `tests/integration/`, 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline — requires `auto_manage.enabled = false` in per-node config.toml to preserve sharded state).
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

All 20 build phases complete. All subsystems wired — no stubs. **1118 lib tests + 75 integration tests passing**; 8 lib + 1 e2e ignored (env-var or manual). Clippy clean default + features dev,claude-subscription + `--features llama`.

### Latest: R146 — External bug report (raw-pc / raw-proxamd5, v0.3.4-alpha) (2026-07-22)

Five bugs from the second external user, found while deliberately splitting a
9-shard 8B model across two home machines. Four were real defects; one was a
display lie that cost them debugging time. A sixth (fabricated peer credit
balances) fell out of investigating #4.

1. **Unneeded TP group → hard failure** (`inference/scheduler/mod.rs`). The
   single-local-segment fast path still called `detect_tp_groups`, so a
   fully-replicated small model formed a tensor-parallel group with a LAN peer
   the request never needed; when the peer went quiet the whole request died
   with `AllReduce timeout after 10s for layer 0`. Fixed three ways: fast path
   no longer forms groups, new **`inference.tensor_parallel` flag (default
   false** — per-layer AllReduce over Ethernet costs more than the compute it
   splits), and `forward_through_segments` now degrades to plain local compute
   on TP failure (truncating the request's KV to `index_pos` first, since
   `kv_model_key` carries no TP rank so TP and non-TP share a KV namespace).
2. **Worker VRAM leak on inference failure** (`inference/process_pool.rs`). Only
   explicit unloads ever killed a worker, so a CUDA OOM left the subprocess
   resident holding 4456 MB indefinitely and every retry had less VRAM than the
   last. `WorkerMsg::Error` gained a `fatal` flag stamped via the new
   `worker_ipc::worker_error_is_fatal`; the three daemon receive sites route
   through `classify_worker_error`, which evicts (→ `Drop` → kill) and returns
   `ServiceUnavailable`.
3. **`gpu_layers` was dead code for sharded models** (`config/inference.rs`,
   `daemon/shard_loader.rs`, `inference/model_worker.rs`). It was read only by
   the legacy llama.cpp executor; the split path called
   `Device::cuda_if_available(0)` and never saw it — hence an identical 4456 MB
   at gpu_layers 20, 8 and 0. Worse, the shipped default was `0` documented as
   "CPU only". Now `i32` with llama.cpp semantics (**default flipped 0 → -1** =
   auto, so honouring it doesn't drop CUDA nodes to CPU on upgrade), plumbed
   pool → `--gpu-layers` → worker → `ShardLoadParams.force_cpu`. A GPU OOM pins
   that model to CPU for the run. Partial offload is unsupported and now says
   so instead of ignoring the number (deferred: `docs/FUTURE_WORK.md`).
4. **Flat credit penalty on framework bugs** (`router/distributed_exec.rs`).
   Every failure charged `penalty_serve_failure`; debugging bugs 1-3 drove the
   reporter's own node to -470 with no peer ever misbehaving. New
   `failure_is_penalty_worthy` requires a remote segment AND a
   non-locally-attributable error.
4b. **Peer credits were fabricated** (`api/identity.rs`). With no gossiped
   balance the leaderboard rendered `trust_score * 5000.0` — at DEFAULT_TRUST
   0.5 that is exactly the mysterious "+2500" the reporter saw for a node
   showing itself at -90. Now `credits: null` + `balance_known: false`, shown
   as an em dash.
5. **Dashboard VRAM gauge showed estimates as live usage**
   (`frontend/js/components/dashboard.js`). Summed `estimated_vram_mb` over
   loaded models and displayed it in place of the live nvidia-smi figure,
   with a clarifying tooltip only when real usage *exceeded* the estimate. An
   idle machine read "5.3 GB / 5.7 GB — 93%" on a red gauge against a real
   ~1 GB. Live usage is now always primary; the estimate is tooltip-only.

i18n: 4 keys added / 2 removed across 21 locales (1186 → 1188 entries).
1099 → 1118 lib tests.

### Prior: R145 — Cloud model/provider refresh (Claude Opus 4.8 / Sonnet 5 / Fable 5, Kimi, Claude Code 2.1) (2026-07-21)

Periodic currency pass — the cloud-provider surface hadn't been touched since
R142 (~2 months). Research-driven (claude-api skill + web) refresh of every stale
model ID and one broken endpoint. **No routing-architecture changes** — the
providers that fetch `/models` dynamically + route by prefix (openai, groq,
nvidia_nim, cerebras, sambanova, fireworks, together, deepinfra) pick up new
models automatically; only hardcoded lists / aliases / one base URL were stale.

- **Claude lineup Opus 4.7→4.8, Sonnet 4.6→5, +Fable 5** (Haiku 4.5 unchanged).
  Single alias bump point `anthropic/convert.rs::resolve_model` (`opus`→
  `claude-opus-4-8`, `sonnet`→`claude-sonnet-5`, `haiku`→`claude-haiku-4-5`, NEW
  `fable`→`claude-fable-5`); three display lists in `admin_providers.rs` (picker
  + subscription, ctx windows corrected to 1M/1M/200K/1M); `claude_sub.rs` default
  fallback `claude-sonnet-4-6`→`claude-sonnet-5` (matches Claude Code 2.1's new
  default). `anthropic-version: 2023-06-01` header still current — unchanged.
- **Claude Code 2.1.215 compat**: verified every subprocess flag SwarmLLM spawns
  is still valid (no breaking removals). Added `"manual"` to
  `claude_session` `ALLOWED_PERMISSION_MODES` (2.1.200+ alias for `default`).
  `set_model` mid-turn control request deferred (enhancement, not a fix).
- **Moonshot/Kimi**: base URL `api.moonshot.cn`→**`api.moonshot.ai`** (`.cn` is
  China-only; `.ai` is the international platform — matters for a global audience)
  + static list `kimi-k2-0527`/`moonshot-v1-*` (all discontinued 2026-05-25)→
  `kimi-k3`/`kimi-k2.7-code`/`kimi-k2.6`/`kimi-k2.5`.
- **DeepSeek**: `deepseek-chat`/`deepseek-reasoner` legacy names discontinue
  2026-07-24 — probe + doc examples → `deepseek-v4-flash` (routing needs nothing,
  `deepseek` prefix matches the new IDs).
- **Mistral**: added `magistral`/`ministral` prefix routing (new families that
  don't start with `mistral`). **OpenAI**: no code change — GPT-5.x matches
  `gpt-`, o-series matches `o3-`/`o4-`; refreshed `mcp/tools.rs` `smart_prefixes`
  (`gpt-4`→`gpt-`, added `o4`/`kimi`/`deepseek`) + doc examples.
- Docs: README, `docs/book` admin + claude-subscription examples. Test fixtures
  refreshed to current IDs across `anthropic/mod.rs`, `providers.rs`,
  `claude_sub.rs`, `anthropic_bridge.rs`. 1099 lib tests unchanged (0 regressions),
  clippy clean default + dev,claude-subscription.

### R144 — Dashboard peer-clarity + reachability docs (2026-07-21)

Follow-on to R143, driven by the first external user (#16) testing 0.3.x live.
Their home node bootstrapped to the anchor and the dashboard called the **remote
anchor "LAN"** (`1 peer 1 lan`). Fixes (`round_log_R144.md`, commits
`938e8de4`+`245c8062`):

- **LAN misclassification** (`network/manager/identify.rs`): a peer was tagged
  LAN if **any** advertised `listen_addr` was private/loopback — but a public
  `0.0.0.0` node advertises `127.0.0.1`, so every remote peer counted as LAN. New
  `multiaddr_is_local` classifies only on the actual connection addr +
  observed-us addr, never on advertised listen_addrs (+2 tests).
- **Peer taxonomy Pool > LAN > Remote** in the WS stats builder
  (`pool_peers+lan_peers+remote_peers == connected`, keyed on
  `connected_node_ids`) + per-peer `is_pool_member` in `serialize_peer_to_json`.
  Frontend: header chips read "1 internet peer / N on your network / N pool
  devices"; every peer row gets a Pool(green)/LAN(purple)/Internet(blue) badge.
- **Version in header** (`#app-version` by the logo; version added to WS stats).
- **Honest empty-state**: `#models-empty` no longer says "Connecting…" when
  `peers>0` — new `models.empty_connected` copy.
- **Swarm-resources strip** (`#netstatus-resources`): computers online (incl.
  you), GPU machines, combined VRAM, shared storage, regions — from
  `swarm_capacity`.
- **i18n**: 14 new keys × 21 locales + translated adjacent English-fallback
  netstatus chip keys (1172 → 1186/locale).
- **Docs**: README promotes out-of-the-box auto-join (anchor + UPnP + AutoNAT v2
  + relay) + Discord; `docs/NETWORKING.md` gained an explicit AutoNAT-v2 note.

Unreleased on `main` → bundle into **v0.3.3-alpha** once v0.3.2 (DNS fix)
publishes. 1097 → 1099 lib tests (+2).

### R143 — Internet reachability & NAT traversal (2026-07-20)

Closes the critical "remote/internet nodes are not discoverable / invite code
carries no public IP" gap reported by the first real external user (issue #16 +
Discord). Root cause: the whole NAT-traversal stack (AutoNAT, DCUtR, relay) was
wired but **inert** because there was no public anchor node, AND
`refresh_listen_multiaddrs` read only `swarm.listeners()` (bound sockets =
private LAN on a NAT'd node), so invite codes silently shipped a LAN-only
address that worked on the LAN and died over the internet.

Self-contained code fixes (infra decision — a self-hosted anchor VPS/Proxmox —
tracked separately with the user):

1. **UPnP default-on** (`libp2p-upnp` added to features). New
   `upnp: Toggle<upnp::tokio::Behaviour>` in `SwarmBehaviour`; `enable_upnp`
   config (default true, auto-off on WSL2 like autonat/dcutr). On a cooperative
   home router this opens the P2P port + confirms the public address with the
   swarm automatically. UPnP `Event` handled in `events.rs`: `NewExternalAddr`
   → success toast + refresh; `NonRoutableGateway` → CGNAT-detected nat_status.
2. **External addresses unioned into invite codes.** `refresh_listen_multiaddrs`
   now unions `swarm.listeners()` ∪ `swarm.external_addresses()` (UPnP /
   AutoNAT / relay-circuit / manual). Extracted to unit-tested
   `build_reachable_multiaddr_list` + `ensure_p2p_suffix` helpers.
3. **`network.external_address` manual override** — a port-forwarded box / VPS /
   dyndns anchor declares its reachable address (`/dns4/...` or `/ip4/...`,
   no `/p2p`); added via `Swarm::add_external_address` at startup so it flows
   into identify, DHT, and every invite code.
4. **Killed the silent LAN-only invite.** New pure `pool::invite::any_internet_reachable`
   (public IP / DNS / relay-circuit; excludes LAN + CGN/Tailscale). When an
   invite has no internet-reachable address, generation still succeeds but emits
   a `pool`/`invite_lan_only` warning toast.
5. **Docs + config**: `docs/NETWORKING.md` (CGNAT check, port-forwarding,
   dynamic DNS, step-by-step anchor setup), `config/default.toml` anchor +
   `external_address` examples, README Discord link + networking pointer,
   book joining-network + ARCHITECTURE + architecture-rules updates.

i18n: 2 new keys (`activity.invite_lan_only`, `activity.upnp_mapped`) × 21
locales, idiomatic.

**Anchor mode + deploy kit** (same round): `--anchor` / `[node] anchor_mode` /
`SWARMLLM_NODE_ANCHOR_MODE` — a bootstrap/relay-only run mode. `Config::apply_anchor_mode`
forces every inference/model knob off; the daemon skips HfWatcher, AutoShardManager,
and model-autoload spawns (no models load — the RAM win); the API binds loopback
(`api/server.rs` reads `node.anchor_mode`). NetworkManager (relay/AutoNAT/DCUtR/UPnP/
DHT/gossip) is untouched. `GET /api/admin/network-code` now also returns `peer_id`
+ `listen_multiaddrs` (the exact `/dns4/…/p2p/<id>` bootstrap string). Turnkey
`deploy/anchor/` kit: hardened sandboxed systemd unit, self-contained installer
(non-root user, SHA256-verified binary, ufw, DuckDNS systemd timer, unattended-
upgrades), anchor `config.toml`, runbook README. NOTE: this is a *runtime* slim
mode (candle still compiled in); the compile-time candle-ectomy (`--features anchor`
→ tiny binary) is the deferred Phase 2.

**Dual-transport + AutoNAT v2 + security (same round):**
- `network.external_addresses` (list; single-string `external_address` still
  accepted via serde alias) so a DuckDNS name is advertised on TCP **and** QUIC.
- **AutoNAT v1 → v2 migration** (research-driven: v1 falsely reports NAT'd nodes
  as "Public" over QUIC, rust-libp2p #3900, → never reserve a relay → unreachable).
  `SwarmBehaviour.autonat_client` + `autonat_server` (both `v2::*::Behaviour::default()`,
  toggled by `enable_autonat`, off on WSL2). v2 client emits `ExternalAddrConfirmed`
  on a reachable result (flows into listen_multiaddrs) and `AddressNotReachable`
  → `NetworkManager::try_activate_relay` (extracted from the old NET-M3 block,
  now rate-limited + retryable). **Belt-and-suspenders relay fallback** on the
  liveness tick: reserve a relay if no internet-reachable address
  `RELAY_FALLBACK_DELAY_SECS=45` after startup, so CGNAT reachability doesn't
  depend on AutoNAT firing. **Relay/DCUtR CGNAT path is wired but needs live
  multi-NAT validation** (deferred — needs the anchor + a real CGNAT node).
- Security sweep (anchor + network/comms): verdict well-hardened (auth-gated
  tensor injection, size caps, conn/relay per-peer limits, signed gossip, no
  plaintext fallback). Applied: installer strict input validation (injection/
  typo guard), `relay_max_circuits`→64 anchor tunable. Research: DCUtR ~70%
  hole-punch success → relay is the essential fallback; QUIC/TCP punch equally.

Tests: +18 (8 invite reachability, 4 listen-addr union/suffix, 2 config default,
2 anchor-mode, 1 external_addresses string-or-list, 1 AutoNAT-v2 toggle; the
AutoNAT v2 relay path itself needs live multi-NAT validation, not unit tests).
1075 → 1097 lib tests. Clippy clean default + features dev,claude-subscription
+ features llama.

### Prior: R142 — Autonomous 8-hour sweep (2026-05-22→05-23)

14 sweep rounds, 60+ findings closed, 15 commits to `main`. Standout
finds were **3 silent production bugs from frontend↔backend JSON
wire-format drift** — bugs no test would catch because the broken
path emits no error:

1. **R141 chat empty-state catalog never rendered** — WS
   `stats_update` payload was never merged into `App.data.cache.stats`,
   so `buildSwarmCatalog()` reading `cache.stats.wishlist` always saw
   `undefined`. Three-row Serveable/Aspirational/Candidate catalog
   that's the entire point of R141 was permanently invisible.
2. **R140 maturity-fade button stuck prominent** —
   `pool.js:199` read `statsCache.peer_count` but `/api/admin/stats`
   serializes that field as `peers` (a different endpoint at
   `admin.rs:1070` exposes `peer_count`). `connectedPeers` was always
   0 → `swarmIsMature` always false → R140's prominent
   "Add Another Device" button never demoted to settings as the swarm
   grew.
3. **Auto-manage activity orb permanently zero** —
   `auto-manage-status.js` matched PascalCase `'Downloading'`/
   `'Queued'`/`'Verifying'`; backend `AcquisitionState` is
   `#[serde(rename_all = "snake_case")]` so values arrive as
   `"downloading"`, `"awaiting_manifest"`. `activeDownloads` always 0;
   users got no visual signal while auto-manage was pulling shards.

Plus **real concurrency bugs**:
- 3 TOCTOU on multi-step DashMap ops (`try_assemble_chunked_forward`
  could double-dispatch, `remove_shard_holder` could drop fresh
  holders, `evict_split_models_lru` cache drift). All fixed with
  atomic `remove_if` predicates.
- 2 clock-dependence bugs in hedging + prefetch
  `maybe_reset_window` — backward NTP correction froze the rate-
  budget windows indefinitely. Fixed with `start > now` recovery.
- Atomic ordering on hedge/prefetch rate counters — `Relaxed` loads
  paired with `Release` stores were broken on weak-memory archs.
- Batch `active_count.fetch_add` outside spawn closure (matched
  R103-class single-path leak).
- `register_multi_turn` orphaned `KvCacheSession` slots on re-
  register.
- HF download size cap bypassed when `Content-Length` absent (DoS).
- 4× wrong `SwarmError::Internal` for worker-died (should be
  `ServiceUnavailable` — operators misattributed 500s to bugs).
- 2 discarded errors that silently closed SSE streams.
- Anthropic `anthropic_split_stream` dropped `matched_stop_sequence`
  — Claude Code's stop_sequence detection broke on split-model
  inference.
- Scheduler oracle violation in `allocate_offline` + mmproj prune
  missing the same `connected_node_ids` filter.
- Hot-path perf: per-token `format!()` in `tracing::info!` allocated
  even when filtered; `vec![forward.clone()]` deep-copied the
  activation buffer on the persistent-stream non-chunked path.

Plus 11 helper extractions consolidating duplicated logic
(`compute_budget_max_bytes`, `local_shard_indices_in`,
`extract_provider_error` pub-elevated, `collect_tree_keys`,
`emit_first_streaming_token`/`emit_streaming_batch`,
`fail_pending_forward`, `resolve_peer_id_for_segments`,
`cap_utf8_to_bytes`, `hf_downloads_normalised`,
`purge_split_model_index_entries`, `_normaliseCode`).

19 new tests pinning invariants (`compute_budget_max_bytes`,
`cap_utf8_to_bytes`, `hf_downloads_normalised`, hedge eviction,
last-holder tombstone, Anthropic stop_sequence wire format).

Docs cleaned: 4 stale `.claude/rules/architecture.md` entries, 3
stale `docs/DIAGNOSTICS.md` DIAG strings, book introduction (11→12
subsystems, 887→1075 tests), `auto_update` default flipped to
disabled note, `contribution_auto` (R121) row added, CHANGELOG R141
test-count baseline, `learn.md` gotchas filename + CLAUDE.md line-
target relaxed.

Deferred to `docs/FUTURE_WORK.md § R142 deferred items`: VLM
`ffn_up/down` weight-loading inversion (needs LLaVA integration
test), LLaVA chat-template eval-failure fallback, Python SDK
R140 endpoints, test-binary `spawn_test_server` shared-helper
extraction, streaming + invite-code v2 config-reference rows,
`apply_update_with_version` Option cleanup, worker compute waste
on cancel (needs new IPC message).

1056 → 1075 lib tests (+19). Clippy clean default + features dev,
claude-subscription + features llama. Detail: commits
05233184..dfcfaa8d + `memory/round_log_R142.md`.

### Prior: R141 — Auto-manage cold-start UX (non-tech-user fixes)

Closes the long-standing "fresh node has nothing to chat with" gap by
removing every silent gate that blocked auto-manage from acting and
surfacing what the swarm already runs directly in the chat empty state.

**Backend**:

1. **Trusted-publisher allowlist** in `src/model/huggingface/watcher.rs`.
   `TRUSTED_HF_PUBLISHERS` covers official model authors (meta-llama,
   mistralai, Qwen, google, microsoft, deepseek-ai, etc.) + curator
   community (bartowski, TheBloke, unsloth, lmstudio-community,
   MaziyarPanahi, QuantFactory, second-state). Models from trusted
   publishers promote to `DemandVerified` at 10k downloads instead of
   100k — fixes the "Phi-4 / Qwen3 / fresh Mistral release just landed
   but auto-manage won't touch it for a month" UX. Helper
   `is_trusted_publisher` re-exported via `crate::model::huggingface`
   for use in the wishlist scorer.

2. **Wishlist `Candidate` status** in `src/model/auto_manage/wishlist.rs`.
   `compute_wishlist` now merges HfTrending entries the swarm hasn't
   adopted yet (cap 24) as `Candidate` rows with new fields
   `hf_repo_id` + `task_tags`. Frontend renders these with a
   "Set this up" CTA that opens the HF browse pre-filtered to the
   repo — user picks the quant variant (no auto-pick, preserving the
   existing trust boundary). Trusted publishers get a +10 score bonus
   and the `wishlist.why.trusted_publisher` tag.

3. **`auto_switch_quants` default → `true`** in `src/config/inference.rs`.
   A recommendation surface that requires the user to read it and
   click a button isn't a recommendation, it's a chore. Trust + prune
   cooldown already guard bandwidth; operators on metered links can
   flip back off.

4. **`P2P_PERMIT_STALL_SECS = 180`** (was 600) in `manager.rs`. A
   non-tech user staring at a stuck download for 10 minutes is
   product-broken; 3 minutes still covers an honest 32 MiB chunk over
   a slow link.

5. **Activity event on `hf_sources` cap** in `daemon/dispatch/mod.rs`.
   New `activity.hf_sources_cap_reached` event fires on the 1st and
   every 50th drop, with a warning toast pointing the user at the
   Settings cleanup path. Previously silent → user wouldn't notice
   they were losing models from peer gossip.

**Frontend**:

6. **Chat empty state shows swarm-available models** — `createEmptyState`
   in `frontend/js/core/utils.js` builds a `buildSwarmCatalog()` block
   when no model is selected. Three rows: Serveable (one-click select),
   Aspirational ("the swarm is gathering these"), Candidate (only
   shown when nothing is Serveable, routes to HF browse). Chip click
   handlers route through `App.models.selectDropdown` +
   `App.chat.newSession` so the user is in a fresh chat with the model
   loaded in one click. Re-rendered on every `stats_update` so the
   catalog comes alive within ~2s of daemon start. Style: new
   `.chat-empty-catalog*` rules in `frontend/css/style.css`.

7. **Wishlist `Candidate` CTA** in `frontend/js/components/swarm-tab.js`
   — `wishlist.cta_candidate` button routes to `App.swarmTab.openSearch`
   with the HF repo_id, dropping the user into the existing search
   subtab pre-filtered to the right repo.

**i18n**: 15 new keys translated across all 21 locales (1156 → 1172
entries per locale) — idiomatic, not English fallback. New keys cover
the chat catalog (titles, hints, chip meta, replica counts), the
wishlist Candidate status + CTA, the trusted-publisher tag, and the
hf_sources cap activity event.

**Tests**: +5 watcher tests (trusted-publisher thresholds, case-
insensitive match, repo_id parsing edge cases), +3 wishlist tests
(i18n key parity, Candidate serialisation roundtrip, hf_repo_id/
task_tags omitted from wire when empty). 1048 → 1053 lib tests. Clippy
clean default + features dev,claude-subscription + features llama.

### Prior: R140 — Pool invite codes v2 (bootstrap-before-decentralization)

The 8-character pool invite code (`A3F7K2M9`) worked only when both nodes
were already on the same libp2p swarm — useful in a mature decentralized
network, but useless for the case the invite code was originally designed
for: helping two fresh nodes find each other before decentralization is
achieved. R140 closes that gap.

**New `swarmpool://...` blob** (`src/pool/invite.rs`) wraps the existing
8-char code with the inviter's reachable listen multiaddrs. Encoded as
JSON → ChaCha20-Poly1305 (random key embedded, anti-IP-harvesting only) →
base64url. ~300-500 chars, fits in a copy-paste. Inner payload:
`{ version, pool_id, pool_name, multiaddrs[], code (8-char), expires_at_unix }`.

**`SharedState.listen_multiaddrs: arc_swap::ArcSwap<Vec<String>>`** —
live snapshot rebuilt by NetworkManager on `NewListenAddr` /
`ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed`. Each entry
is suffixed with `/p2p/<local_peer_id>` for identity verification. Filtered
via a new `addr_is_remotely_reachable` that drops loopback / unspecified /
link-local / AWS-IMDS but **keeps** Tailscale CGN (100.64.0.0/10) — the
existing `is_non_public_addr` filter is for anti-gaming / PEX-leak
prevention and explicitly rejects CGN, the exact range the WAN-bootstrap
use case needs.

**`handle_join_with_code` dual-mode**: v2 blob → dial each multiaddr via
`NetworkCommand::DialAddress` then broadcast the existing
`PoolMessage::JoinRequest`. Legacy 8-char → direct broadcast (preserves
on-swarm flow). **Wire protocol unchanged** — v2 is purely the rendezvous
wrapper around the existing pool-join handshake.

**Generation rejects empty addresses**: if `listen_multiaddrs` is empty
(daemon hasn't bound yet), `handle_generate_invite_code` returns
`ServiceUnavailable` instead of silently handing out a useless code.

**Frontend**: dropped the fake-QR pattern (only hashed 8 chars, was never
scannable — misleading), replaced with a monospace code box + Copy button
sized for the ~500-char v2 blob. Paste field upgraded from
`<input maxlength=8>` to `<textarea>` so the full blob fits without
scroll. Join handler in `pool.js` + `setup.js` sniffs prefix to route to
v2 (case-preserved) or legacy 8-char (uppercased).

**Maturity-fade UI**: while the local node sees <50 swarm peers, the
"Add Another Device" button sits prominently in the dashboard header
(bootstrap-before-decentralization mission — invite codes are how the
swarm grows in this phase). Once peer count ≥50, the button demotes
to a settings-area panel: the swarm is mature enough that Kademlia DHT
discovery is reliable, so explicit rendezvous via invite code is no
longer the load-bearing path. The threshold is against **connected swarm
peers (read from `stats.peer_count`)** — NOT pool member count, since
pools cap at `max_pool_size=10` and a pool-member threshold would never
fire for the typical 2-3 device user.

**i18n**: 5 strings refreshed across 21 locales (`pool.code_invalid`,
`pool.enter_code`, `pool.share_code`, `pool.how_to_join`, +
`pool.enter_code_hint` new), 1 dead key removed (`pool.scan_or_type` —
was for the fake QR). All translated, not English-fallback.

**Tests**: 18 new (10 codec — roundtrip, tamper, expiry, version, truncated,
oversized, whitespace, prefix sniff, malformed token, missing-prefix; 3
listen-addr filter; 5 PoolManager — empty-addrs error, decode roundtrip,
v2 dials-then-broadcasts, legacy skips-dial, garbage rejected).
1030 → 1048 lib tests. Clippy clean default + features dev,claude-subscription.

### Prior: R139 — Tier 4K communication-computation overlap

Four commits closing FUTURE_WORK Tier 4K with a research-driven scope pivot.
Phase B turned out to be already shipped via the existing async architecture
(`router::distributed_exec` per-request `tokio::spawn` + `pipeline_stream`
keyed-by-(peer, request_id) + `process_pool::batch_scheduler_loop`); R139
documented and skipped it.

1. **Phase C — encrypt/decrypt off the NetworkManager event loop** (commit
   11333f67). The CPU-bound ChaCha20-Poly1305 sealing in `handle_send_tensor`
   and the open in `handle_tensor_payload` (TENSOR_TAG_ENCRYPTED arm) are
   offloaded to `tokio::spawn` tasks. New `NetworkCommand::SendEncodedTensor`
   variant carries the encrypt-result back to the event loop for the
   `send_request` + bookkeeping step. Default config (`enable_encryption=true`,
   `persistent_pipeline_stream=false`) sees ~50–200µs/forward event-loop block
   savings; under concurrent decode traffic this is the difference between
   smooth event-loop responsiveness and observable jitter on libp2p ping /
   gossip / connection events.

2. **Phase A pivot** — original "worker streams row-tiled output during
   matmul" framing matched the SGLang PD-disaggregation anti-pattern
   (rolled back per-tile streaming, 2-5× slower at high concurrency due to
   per-chunk fixed costs). 2026-05-19 research pass found no production
   inference system streams forward-output tensors (Triton decoupled, vLLM v1,
   NVIDIA Dynamo/NIXL — all single-tensor responses). Pivoted to
   **daemon-side STREAM-chunked encrypt+send** on a single libp2p stream
   (age STREAM construction + TokenWeave K=2-4 sweet spot + Tink Streaming
   AEAD precedent). Single stream → QUIC preserves byte order → no
   receiver assembly state machine.

3. **A-rev.1 — wire format 0x05 + AAD binding** (commit 4b5fc10c).
   New `ChunkMeta { chunk_idx: u32, total_chunks: u32 }` field on
   `LayerForward`. Encoded as optional 0x05 trailer. Bound into AAD via
   `build_layer_forward_aad` so reorder, wrong-total, and cross-transfer
   substitution attempts fail Poly1305 before reaching dispatch. 11 new
   tests across plaintext + encrypted paths (roundtrip first/middle/last,
   trailer adjacency vs 0x04, decoder rejection of invalid metas, AAD
   bytes diverge on chunk_idx/total_chunks flip, backward compat).

4. **A-rev.2/3 — receiver assembly + helper + config** (commit 1d0a5d55).
   `ChunkAssemblyState` slot table on
   `SharedState.pending_activation_chunks: DashMap<Uuid, ChunkAssemblyState>`
   (root-level — cross-cuts RR + persistent stream paths, mirrors
   `pending_layer_results` precedent).
   `SharedState.try_assemble_chunked_forward` accumulates chunks under
   entry-lock with `total_chunks` consistency check, sender-peer binding,
   duplicate-chunk_idx rejection. `chunk_layer_forward` splits at byte
   offsets; passthrough when activation ≤ chunk_size. 4 config knobs:
   `streaming_chunked_send` (default false), `streaming_chunk_size_bytes`
   (default 256 KiB), `streaming_min_activation_bytes` (default 64 KiB
   floor), `streaming_chunk_assembly_ttl_secs` (default 30s). Receiver
   wiring in both `tensors.rs` decrypt-spawn and `pipeline_stream.rs`
   reader paths.

5. **A-rev.4 — sender wired to persistent-stream path** (commit e32c0a5d).
   `pipeline/distributed.rs::forward_through_segments` consults the
   eligibility check (streaming_chunked_send + persistent_pipeline_stream +
   activation > min_bytes) and ships K chunks on the same libp2p stream
   when on. Chunked-over-RR fallback path stays untouched — the existing
   1:1 ResponseChannel pattern would need explicit per-chunk Ack handling
   to support chunking; that's tracked in FUTURE_WORK § Tier 4K remaining.

Remaining for full Tier 4K close-out (small follow-ons): chunked-over-RR
support, worker-side row-tiled output streaming (literal "true" Tier 4K,
3-4 weeks, deferred pending slow-WAN bench data that justifies fighting
the SGLang result). The pending_activation_chunks TTL sweep landed in
`health/monitor.rs` alongside R142's hedge/prefetch eviction wiring.
The chunked-send microbench (`bench_chunked_send`) shipped as part of
`examples/swarm_spec_bench.rs`.

1015 → 1030 lib tests (+15) and +4 swarmllm-types. Clippy clean default +
features dev,claude-subscription. Detail: commits 11333f67..e32c0a5d.

### Prior: R138 — autonomous defer-batch (sweep-log triage + 4 real fixes)

Eight commits closing ~20 deferred sweep-log items. Real changes:

1. **Auto-manage rescan respects `auto_manage_paused`** (closes R104). `model/auto_manage/manager.rs` reads `auto_manage_enabled` atomic and passes `Option<&network_tx>` accordingly; rescan still runs locally (correctness — picking up manually-placed shards) but the network re-announce is gated on the pause toggle. Manual `POST /api/admin/rescan-shards` always announces.

2. **`active_count` fetch_add inside the spawn closure** (closes R103). `inference/router/mod.rs:718` moved into the spawned task so a `tokio::spawn` OOM panic can no longer leak the tier-cap counter.

3. **`CreditBalance` schema-upgrade safety** (closes R105 deferral about forward-compat). `#[serde(default)]` on `balance/lifetime_earned/lifetime_spent/last_updated`; `node_id` intentionally not defaulted (missing identity is data corruption, fail loudly). Type-level doc spells out the rule for future additions. 4 regression tests + drive-by `serde_json` dev-dep added to `swarmllm-types` so 15 previously-dead crate-local tests now run.

4. **`private_mode`/`offline_mode` moved out of `pool_state` tree** (closes R105). New `TREE_NODE_MODES` + one-shot legacy migration via `restore_node_mode()` helper. Each tree single-typed; no namespace-collision risk for `iter_json::<PoolState>`.

5. **`check_integrity` strict per-tree type validation** (closes R105). `validate_strict` routes each `CRITICAL_TREES` entry through the actual `swarmllm_types` type (`ModelManifest` / `CreditBalance` / `NicknameRecord` / `PoolState` / `CreditTransaction`) or `i64` (`pool_removal_replays`). Type mismatches that previously passed JSON-Value validation are now reported as corrupt. Dropped the unused "identity" entry. Updated existing tests + 2 new tests demonstrating the R105 concern in concrete form.

6. **`credit_percentile_cache` no longer held across `DashMap` iter** (closes R97). Three-phase pattern (peek under lock → iter outside → re-lock to write) replaces the lock-over-iter that could block the router task on long iters.

7. **`api.metrics_auth_required` config flag** (closes R101/R102 about `/metrics` credit-balance disclosure). Default `false` preserves Prometheus convention + dashboard loopback scrape; when `true`, `auth_middleware` short-circuits the loopback `/metrics` exemption.

8. **Credit forward per-window TOTAL value cap** (closes R102). `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW = 200_000` credits (2× per-tx max). `credit_forward_rl` now stores `(Instant, i64)` pairs; `check_credit_forward_rate` sums amounts in the window and rejects if projected total exceeds the cap, atomic with the existing count cap. 3 unit tests.

Plus ~15 verification-only sweep-log closures for items intervening rounds had already addressed (`x-swarm-forwarded` dead code, relay defaults safe per libp2p 0.20.x, umask race serialised by `spawn_lock`, scan zero-hash auto-compute, `peer_cache` `replace_tree` atomicity, `BACKGROUND_CANCEL_AGES` TTL sweep, MCP `sampling/createMessage` explicit arm, Anthropic→OpenAI tool block translation, SWIFT `emit_token` cap, `auto_manage` config hot-reload first-tick, etc.).

1005 → 1015 lib tests (+10) and 15 newly-runnable `swarmllm-types` tests. Clippy clean default + features dev,claude-subscription + features llama. Detail: `round_log_R138.md` and full sweep-log.

### Prior: R137 — extended FUTURE_WORK deferrals batch

1. **Hot-reloadable cross-pool flags** (closes R135 deferral). `state.credits.allow_cross_pool_inference` + `state.credits.share_model_catalog` AtomicBool mirrors of `config.pool.*`. PUT /api/admin/config writes both atomic + persists TOML; GET surfaces runtime atomic. Pattern follows R121's `contribution_auto`. `pool::scope::cross_pool_extras` + `health::broadcast_pool_model_availability` read from the atomic. New `cross_pool_extras_honors_runtime_flag_toggle` regression test.

2. **Latency sample ring time-coverage** (closes R105 deferral). `inference_latency_samples` migrated from `VecDeque<f64>` to `VecDeque<(Instant, f64)>` with `LATENCY_SAMPLE_MAX_AGE = 600s`. Drop-by-age both on insert AND on every read (per-call read filter needed at low rates). Fixes stale p99 on lightly-loaded nodes. +16KB worst case. Monotonic `_count`/`_sum` atomics unchanged.

3. **L1 forward_verify_through_segments test coverage** (partial closure of R136 deferral). 4 new pipeline tests: 3 pure-function (`pack_verify_tokens_to_le_bytes`, `build_spec_verify_forward`, `build_kv_truncate_forward` — wire-format drift now fails fast), 1 network-drop unit test (closed channel → assert error + `pending_layer_results` clean). Full multi-segment orchestration still needs worker subprocess infra.

4. **L1 hit/miss lifetime counters** in `MetricsProviders`. Bumped at both call sites of `ngram_lookup_drafts` (draft-free in `ngram_only_spec.rs` + draft+ngram in `speculative.rs`). Surfaced in `GET /api/admin/stats → swarm_spec.ngram = { hits, misses, total, hit_rate }`.

5. **`foreign_pool_catalog` eviction O(K×N) → O(N)** (closes R135 sweep deferral). Replaced K-iteration full-scan loop with `select_nth_unstable_by_key` partial-sort + batched removes; ~10× faster on the eviction path. New stress test exercises 1000+200 entries.

6. **12-provider list extraction** (closes R72 deferral). `ProvidersConfig::keyed_entries()` + `ProvidersUpdate::keyed_entries()` collapse the 4× duplicated name list across `admin_providers.rs`. Adding a 13th provider now requires editing 2 lists instead of 5+.

Also: verification-only sweep-log closures for 10 R69-R76 deferred items that intervening rounds have already addressed (is_valid_draft_pair dead code; ShardPin::matches extracted; sample_token_with_params delegates; prefix_cache last_hit is AtomicU64; check_multi_turn_reuse uses HashSet; pool/manager rate-limiter releases lock; protocol.rs read_wire_frame extracted; build_json_frame extracted; anti_gaming TTL is 1h not 24h; sleep-in-handle_acquire removed).

Doc sweep alongside: i18n key counts (1156→1154 entries), test counts (943→1005 across CLAUDE.md/README/book/plans), SharedState diagram in ARCHITECTURE.md updated with R130/R133/R134/R136 fields.

### Prior: R136 SWARM-SPEC v0.1 — 4-layer P2P-native inference acceleration cascade

All 4 layers shipped with **true dispatch** (not scaffolding) and real-inference validation on a 3-node local cluster:

- **Layer 0 Q8_0 wire compression** (`inference/quant.rs`, group-32 + f16 scale): default-on, 3.76× compression, ~17µs round-trip. Real bench: +4-17% single-segment routing; loopback distributed within noise (encode CPU rivals saved wire-time — real win is WAN-only, synthetic prediction 1.5-1.7× on bandwidth-bound hops).
- **Layer 1 n-gram cascade** (`inference/ngram_lookup.rs` + `pipeline/ngram_only_spec.rs`): draft-free via standalone tokenizer cache (lazy-loaded from `gguf_header.bin`); multi-segment via shared `pipeline::forward_verify_through_segments`. Real bench single-segment: summary **+45%** on 77% n-gram hit-rate (2.75 → 4.0 tok/s). Multi-segment sharded: 3.7/3.3/4.7 tok/s with 35/9-77/17% hit-rates.
- **Layer 2 adaptive hedging** (`inference/hedging.rs` + `pipeline/hedge_dispatch.rs::forward_verify_with_hedge`): full true-dispatch (race primary vs duplicate to alt holder, winner returns, loser dropped). Fast-path optimisations zero-overhead when disabled/multi-segment/no-alt-holder/insufficient-EWMA-samples/budget-exhausted. Default off. min_samples=20 (bumped from 5 after review found warm-up over-firing).
- **Layer 3 predictive prefetch** (`inference/prefetch.rs`): per-session first-token histogram + idle-time learner + throttling. Observability-complete dispatch: `record_dispatch` + ActivityEvent on decision. K-layer activation compute deferred (workload-dependent — small models on fast hardware have negligible prefill).

### Dispatch order in `pipeline/distributed.rs::execute_distributed`

1. `try_dsd_distributed` (multi-segment + draft model)
2. `try_speculative_distributed` (single-segment + draft model)
3. `try_ngram_only_distributed` (NEW R136 — no draft model needed)
4. `try_remote_generate_fastpath` (no per-token coordinator)
5. standard execute_distributed loop

L1 precedes remote_generate because multi-token-per-round acceptance on hits wins over remote_generate's one-token-per-RTT throughput.

### Bench infrastructure shipped (zero new dependencies)

- `examples/swarm_spec_bench.rs` — microbench per layer primitive
- `examples/3node_{setup,sharded_setup,inference_bench}.sh` — local cluster + bench scripts
- `GET /api/admin/stats → swarm_spec` block — hedge + prefetch metrics for operator visibility

### Prior rounds (detail in commit messages + `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_R126_R135.md`)

- **R134.7**: cross-pool inference routing (opt-in via `pool.allow_cross_pool_inference`) + predictive eviction (`RECENT_REQUEST_PENALTY` in prune.rs)
- **R134.6**: quant auto-action layer (opt-in `auto_manage.auto_switch_quants`)
- **R134**: inter-pool model availability discovery (`SwarmMessage::PoolModelAvailability`), HF anti-gaming reputation (`ModelTrustInfo.failed_promotions`), pool-state diff gossip (`PoolMessage::StateDiff` with `PoolState.generation`)
- **R133**: quant choice automation (`Quantization` enum 29 variants + `model/auto_manage/quant.rs` recommender)
- **R132**: MoE per-arch routing (`MoeGatingFunc`)
- **R131**: pool-state gossip debounce
- **R130**: cross-pool wishlist gossip

## Common Commands

```bash
cargo build --no-default-features --features dev,claude-subscription  # Dev build (live frontend + Claude Code)
cargo fmt && cargo clippy --all-targets -- -D warnings  # Lint (MUST pass before push)
cargo test                           # All tests
cargo run -- run -p 8800 -v          # Start daemon
```

**Note:** Always include `claude-subscription` feature when testing Claude Code integration. Bare `--features dev` omits the Claude subscription provider.
