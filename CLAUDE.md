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
- `state.metrics` (`MetricsProviders`) — `node_stats`, `inference_requests_total`, `channel_metrics`, `providers_config`, `swarm_capacity` (R110), `hedge_tracker` (R136 Layer 2), `prefetch_orchestrator` (R136 Layer 3), `peer_speed` + `peer_model_warm_at` (measured per-peer prefill/decode speed — sizes segment timeouts and ranks candidates), etc.

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
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, peer_speed, perf_history, relay, tp_allreduce)
│   ├── network/   (manager/{mod,events,requests,tensors,identify,commands,connections,dht,shard_transfer}, behaviour, discovery, protocol, transport, relay, peer_cache, helpers, pipeline_stream)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface/, auto_manage/, lora)
│   │   ├── auto_manage/  (mod, manager, scoring, download, prune, scan, vram, parallax, wishlist)
│   │   └── huggingface/  (mod, download, private_types, probe, search, shards, watcher, tests)
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, local_embedder, model_worker, process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3), trace (per-request route + timing record))
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
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, dashboard-shards.js (pure shard HTML builders exposed as `App.dashboardShards`), models.js, auto-manage-status.js, settings.js, setup.js, welcome.js (R127 — first-run tour modal), downloads.js, notifications.js, identity.js, network-map.js, compare.js, responses.js, pool.js, swarm-tab.js (R111 — wishlist + capacity-plan + performance view), reference-models.js (R148 — shared test-model picker, `App.referenceModels`)
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 3 modal overlays (setup, settings, R127 `#welcome-modal` first-run tour) + 10 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1287 translation keys (1289 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1596 lib tests passing + 11 ignored (env-var-gated real-model + manual smoke), 80 integration tests in `tests/integration/` + 2 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 2 repo-consistency, 1 in `tests/api_key_side_effects.rs` (deliberately an INTEGRATION test — see gotcha #230), 30 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root), 6 in the vendored request-response patch (`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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

All 20 build phases complete. All subsystems wired — no stubs. **1596 lib + 81 integration + 2 repo-consistency + 30 swarmllm-types tests passing**; 9 lib + 2 e2e ignored (env-var or manual). Clippy clean default + features dev,claude-subscription + `--features llama`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.60 → .65 (2026-08-01→03): the split works, machines in one house find each other, and memory stays bounded

**v0.3.65** — diagnostic only. The GPU memory-admission gate logged ONLY on
refusal, so a model that was admitted and then died with
`CUDA_ERROR_OUT_OF_MEMORY` left no record of what the gate expected it to cost.
Three causes were indistinguishable from outside: a low estimate, a budget still
held by a model mid-eviction, and a genuinely unbounded allocation. Now
`DIAG: admitting model to GPU` prints estimate/committed/headroom — compare
against the worker's `vram_after_load_mb`.

**v0.3.64** — a node advertised model pieces it no longer had. The shard
registry is built at startup and updated by events; nothing watched the disk, so
deleting a model folder (the only way, there is still no CLI remove command)
left the node offering peers work it could not do until restart. The health
monitor now reconciles against disk each announce cycle; `enable-privacy` stats
the files itself, because it asserts where a user's text goes and must be true
when said. Existence only — a mid-download shard occupies its final path.

**Open from tester reports (2026-08-03), read `docs/FUTURE_WORK.md`:**
(1) **A node with full local coverage never consults the network, even with no
headroom** — `scheduler/mod.rs` returns one local segment + `standbys: vec![]`
the moment it holds every layer. Holding the weights and having room to use them
are different questions and only the first is asked, so a node fails alone with
an idle LAN peer beside it. Deliberate, with good reasons in the comment; the
gap is what coverage stands in for. **"Shard storage-sharing, not
inference-sharing."** (2) Key rotation still breaks in-flight distributed
inference every 10 min (one key, no previous-key grace). (3) A peer whose return
path is dead stays `is_connected=true` and schedulable.

**Ruled out, do not re-investigate:** the .61/.63 memory fixes apply to BOTH
forward implementations — `forward_inner_impl` AND `forward_batch` call the same
`lw.forward_attn` / `Mlp::forward`, so no execution path misses them. A tester's
remaining OOM at 5143 tokens (5021 passes) is a margin being crossed, not a
missing bound.

**v0.3.63** — a long prompt exhausted graphics memory in the **feed-forward**,
the step right after the attention that .61 fixed. Same shape, still unbounded:
three buffers, each `[tokens x intermediate]`, live at once (~700 MB for a 5k
prompt on a small model). Blocked on the token axis; exact, tests assert max abs
diff **0.0**. **The .61 changelog claim that single-machine requests were never
at risk was WRONG** and a tester reproduced it — a machine holding a whole model
is served as a ONE-SEGMENT PIPELINE through `handle_forward`, which does not
chunk. **Chunked prefill bounds tokens on the LOCAL generate path only**; any new
per-token temporary needs its own bound. Same release: two LAN machines routed
through the anchor VPS at ~3 s and then went mutually invisible for hours — four
causes, see `round_log_lan_peering_0802.md` and gotcha **#234** (a safety fix
that silently disabled an unrelated fix through shared state). Also: a decrypt
failure now answers `LayerResult::error` instead of dropping silently and
burning the whole segment budget.

**Open, found 08-02, NOT fixed:** `KEY_ROTATION_INTERVAL` is 600 s and
`crypto/session.rs` keeps ONE key with no previous-key grace, so **every session
has a re-key window every 10 minutes and any forward crossing it is discarded** —
likely cause of the intermittent distributed failures across several releases.
Recommendation (previous-key grace) in `docs/FUTURE_WORK.md`. Also open: a peer
whose return path is dead stays `is_connected=true` and schedulable.

**v0.3.60/.61 — RESOLVED: distributed multi-segment inference.** `Tensor bytes
too short` was never a wire-format defect. `pending_layer_results` is keyed by
`request_id` alone, but a failed-over request has TWO forwards outstanding; the
abandoned one's late error resolved the **standby's** waiter (gotcha #229).
Waiters now record the node they expect and resolve through
`SharedState::resolve_pending_layer_result`. Plus peer-speed-sized segment
budgets, `ReachTier` (direct beats relayed by construction), and the f16
embedding table. **.61** blocked the attention query axis — the shipped GPU build
is the only one with quadratic attention memory, because `flash-attn` is
excluded from the `cuda` feature (gotcha #232).

**Release process:** `scratchpad/tag_when_green.sh <tag>` is the ONLY way to tag
(requires conclusion `success` AND every job green, BEFORE tagging). `Cargo.toml`
+ `Cargo.lock` in the SAME commit (CI is `--locked`). **Do not push between the
release commit and the tag** — it cancels the run the tagger waits on. After the
workflow publishes: `gh release edit <tag> --prerelease=false --latest` — BOTH
flags, clearing prerelease alone does not move `/releases/latest` (gotcha #225).

### Superseded — v0.3.49 → .59 (2026-07-29→08-01) — pointers only

Full detail in `memory/round_log_*.md` and the CHANGELOG. Read those before
re-deriving any of it.

- **SPM tokenizer CLOSED** (.49-.51): `spm_encode` applied stale merge-queue
  entries; **64.9% of inputs mis-tokenised** on Phi-3.5's real vocab, now 0
  mismatches vs reference `sentencepiece` over 4,128 inputs. Pinned by
  `spm_merge_tests`.
- **.51**: a hash cannot tell "wrong bytes" from "not all the bytes" —
  `verify_shard` checks `size_bytes` FIRST; trust docked only for
  right-size-wrong-hash (#203).
- **.50**: one abandoned request froze a model for everyone (cancellation wired
  only to an explicit header). RAII guard on handler drop, **non-streaming
  only**. Also un-stranded every pre-.44 node — `/releases/latest` 404s while
  every release is prerelease.
- **.56-.59**: memory admission (reclaim from idle-but-resident models BEFORE
  refusing; in-flight read from the worker pool's own channels). A model could
  be freed *while answering* — `active_pipelines` is coordinator-only and
  `serving_models` is peer-served-only, so a node answering its OWN client was
  in neither (#194). `cargo test` overwrote a running node's API key (#226).
  A shared CI cache shipped a release with no Windows GPU build for ten
  releases (#222/#223/#224).

### v0.3.39 – v0.3.46 (07-27→29) — pointers; detail in `round_log_overnight_0728.md`

- **.46**: local replies had `▁` for every space — `CachedDecoder` built
  `is_sentencepiece:false, has_tokenizer:false` under a comment that went stale,
  so decoding could only take a **GPT-2** byte fallback. Same defect produced
  the unexplained `<0x0A>` — one cause, two symptoms a day apart (#200).
  **Peer-served work is decoded on the SERVING side, so every cross-node check
  looked clean.**
- **.45**: my own .44 shard check ate good shards — an all-zero manifest hash is
  FAILURE when auditing a held shard, but means *nothing to compare* at an
  accept gate.
- **.44**: overlay trust was satisfiable by coincidence (`100.64.0.0/10` is
  shared CGNAT) → Tailscale `whois`, where **`Unavailable` must never read as
  yes** (#199). **Credits are self-attested and unenforced by design.**
- **.42**: a default that lives only in `#[serde(default)]` never reaches a
  config the daemon already wrote (#198). **Empty was NOT always accidental.**
- **.41**: **`is_loopback()` means "the last TCP hop began in this daemon's
  netns" and is wrong BOTH ways** (#195) — subnet routers SNAT.
  `api::dashboard_trust::classify` is the one decision point.
- **.40**: `prefill_chunk_tokens` bounded decode interruption in TOKENS not time
  (#191). **The first fix made GPUs WORSE**; the pacer self-disables if a shrink
  did not help — do not remove that check.
- **.39**: `current_exe()` returns `"...(deleted)"` once the binary is replaced,
  and replacing it IS updating (#188). Rule: **timeouts must bound what actually
  varies** (#189, #190).

### v0.3.15 – v0.3.38 (07-23→28) — pointers; full detail in the round logs

Read the named `memory/round_log_*.md` before re-deriving any of these.

- **Read gotcha #179 before touching connection selection.** A relay carrying an
  INBOUND connection is a bare `/p2p/<peer>` with no transport component, so it
  counted as direct and won every send. And **retraction alone is futile — the
  blacklist is REQUIRED**, since the DHT re-advertises a retracted holder.
- **#165**: we published an inbound connection's ephemeral source port as
  dialable; **poisoned caches need a node RESTART, not just a new binary.**
- **#163**: `max_established_per_peer = 1` structurally disabled DCUtR — a hole
  punch needs a 2nd concurrent connection. Hole punching verified live 07-25.
- **#169**: the control-token leak chased across four releases was a **prompt**
  bug, not an output-scrubber bug — `grep "chat template failed" node.log` had
  been firing on every request for releases.
  `inference::finalize_reply_text` owns the ordered scrub→truncate→trim.
- **"tok/s per node per shard" is NOT measurable in a pipeline** — use each
  segment's share of inter-token latency.
- **v0.3.38**: idle VRAM was never reclaimed. The demand-EMA gate exists because
  `record_request` is called ONLY from the outbound router path, so serving a
  peer never updates it.

### Prior rounds (pre-v0.3.15)

- **R143-R150** (07-20→23): NAT/internet reachability (UPnP default-on, AutoNAT
  v1→v2, `--anchor`), request cancellation, `gpu_layers` plumbing. Gotchas #150-160.
- **R136-R142**: SWARM-SPEC v0.1 cascade (L0 Q8_0, L1 n-gram, L2 hedge, L3
  prefetch); `swarmpool://` invites v2; cross-pool gossip/routing.
- **Pre-R136**: the 20 build phases. `docs/ARCHITECTURE.md` § phase history.


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
