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
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, perf_history, relay, tp_allreduce)
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

- 1519 lib tests passing + 9 ignored (env-var-gated real-model + manual smoke), 80 integration tests in `tests/integration/` + 2 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 2 repo-consistency, 30 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root), 6 in the vendored request-response patch (`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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

All 20 build phases complete. All subsystems wired — no stubs. **1519 lib + 80 integration + 2 repo-consistency + 30 swarmllm-types tests passing**; 9 lib + 2 e2e ignored (env-var or manual). Clippy clean default + features dev,claude-subscription + `--features llama`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.49 / .50 / .51-alpha (2026-07-29): attribution, and a tokenizer bug still open

**FIXED 2026-07-29 — the SPM tokenizer defect is closed.** `spm_encode` applied
stale entries from its merge priority queue: merging extends `left` to cover
`right`, so a queued bigram naming `left` still named a live, adjacent symbol
but one whose text had grown past the piece that was scored. Applying it built a
symbol for text never checked against the vocabulary, and the final lookup
missed and dumped the span through byte fallback. `Merge` now carries the
combined size it was scored at and the loop rejects any entry whose symbols no
longer match it — the same guard as llama.cpp's `llm_tokenizer_spm`.

Scope was **much worse than the "1 word in 4" estimate**: against Phi-3.5's real
vocabulary over a 4,128-line corpus, **64.9% of inputs were mis-tokenised**.
Verified against the real `sentencepiece` library with Phi-3.5's own
`tokenizer.model` — **0 mismatches on 4,128 inputs** after the fix. Live: "What
colour is a banana?" went from `The text "a␦␦␦ debido a que debido a que…` to a
correct answer. Pinned by `spm_merge_tests`; `examples/spm_probe.rs` diffs any
GGUF header against a reference. The BPE path rescans from current state each
iteration and was never exposed.

**v0.3.51 — stop blaming peers for OUR truncated downloads.** A failed shard
hash always docked the sender's trust, but a hash cannot tell "wrong bytes" from
"only some bytes arrived". Tell: one peer failed the same shard four times with
a DIFFERENT computed hash each time (corrupt storage gives the SAME wrong hash;
varying output means varying amounts arrived) while timing out constantly.
`verify_shard` now checks the manifest's `size_bytes` FIRST via the existing
`quarantine_shard_if_size_mismatch` — which was already called from the startup
and periodic scans and NOT from the accept gate for untrusted bytes. New
`SwarmError::ShardIncomplete`; trust is docked only for right-size-wrong-hash.

**v0.3.50 — one abandoned request froze a model for everyone.** Cancellation was
only ever wired to an explicit `x-swarmllm-cancel-token` header that one internal
caller sets, so a client that simply disconnected signalled nothing: the request
ran to completion holding the executor, and every later request queued behind it.
Requests now always carry a cancel flag, flipped by an RAII guard when the
handler future drops. **Non-streaming paths ONLY** — a streaming handler returns
as soon as the SSE body exists and generation continues after, so arming it there
would cut every stream.

**v0.3.50 also un-stranded every pre-.44 node.** Those query
`/releases/latest`, which 404s while every release is `prerelease: true`, so they
were told "you are running the latest version" forever. Publishing one release
as non-prerelease is the ONLY mechanism that reaches them. Keep doing this, or
they strand again.

**v0.3.49 shipped a changelog claim that was false** — it was cut to fix a stray
`▁` in shared answers, the fix addressed a real sibling decoder path, and the
symptom was unchanged because the cause was the tokenizer bug above. CHANGELOG
was corrected after release rather than left standing.

**Process — two self-inflicted failures worth not repeating.** (1) I tagged
`.50` on a commit whose CI had FAILED, because the wait script conflated
"completed" with "succeeded"; caught while still a draft with 0 assets, tag and
draft deleted. Use `scratchpad/tag_when_green.sh`: it requires the run
conclusion to be exactly `success` AND every individual job green, verified
BEFORE tagging. (2) The break itself was a `Cargo.toml` bump without the matching
`Cargo.lock` — CI builds `--locked`. Run `cargo check` and confirm `Cargo.lock`
is in the release commit. Also: rapid pushes cancel the in-flight CI run you are
waiting on; stop pushing during a release window.

**Left running twice**: a stray test node kept advertising shards to the swarm
after I reported cleanup done. `pkill` errored both times and I did not verify.
Always confirm with `pgrep -af "bin/swarmllm"` after killing.

### v0.3.39 – v0.3.46 (07-27→29) — one line each; detail in `round_log_overnight_0728.md` + CHANGELOG

- **v0.3.46**: local replies had `▁` in place of every space — `CachedDecoder`
  was built `is_sentencepiece:false, has_tokenizer:false` under a comment that
  went stale when `standalone_tokenizer()` was added, so decoding could only
  take a **GPT-2** byte-decoder fallback. Same defect produced the unexplained
  `<0x0A>` — one cause, two symptoms a day apart (#200). **Peer-served work is
  decoded on the SERVING side, so every cross-node check looked clean.** Also:
  over-long prompt returned 500 (a real `Validation` flattened crossing the
  worker IPC) → now 400; Windows GPU build fixed (Vulkan lib dir never added to
  `LIB`; `msvc-dev-cmd` replaces `LIB` afterwards; SDK version now pinned).
- **v0.3.45**: my own .44 shard check ate good shards — `verify_shard` treats an
  all-zero manifest hash as FAILURE, right when auditing a held shard, wrong as
  an accept-gate where a missing hash means *nothing to compare*. Every shard of
  a hash-less manifest was rejected, **deleted**, and its sender penalised.
  Also: `provider-model-status` got its own rate bucket (it sat with human-
  triggered mutations while being fired automatically by the dashboard).
- **v0.3.44**: external security audit — overlay trust was satisfiable by
  coincidence (`100.64.0.0/10` is shared CGNAT and `listen_multiaddrs` includes
  every BOUND interface), now answered by Tailscale's `whois` LocalAPI with a
  three-way verdict where **`Unavailable` must never read as yes** (#199); P2P
  shards were announced *before* verification; `vec![0u8; len]` committed a
  declared size before any payload arrived. Update lifecycle drains then `exec`s.
  **Credits are self-attested and unenforced by design** — researched, written
  up in FUTURE_WORK, NOT fixed.
- **v0.3.42**: nodes set up before 2026-07-21 could never rejoin — a default
  that lives only in `#[serde(default)]` never reaches a config the daemon
  already wrote, and the wizard writes every field on "Start SwarmLLM" (#198).
  **Empty was NOT always accidental** (the anchor and the bench cluster rely on
  it), so the naive fix would have broken both.
- **v0.3.41**: the dashboard works from another device. **`is_loopback()` means
  "the last TCP hop began in this daemon's netns" and is wrong BOTH ways**
  (#195) — a same-host reverse proxy passes it for a remote phone; a Tailscale
  subnet router never does, because they **SNAT by default**.
  `api::dashboard_trust::classify` is now the one decision point. Also: an async
  fn runs to its first `await` before returning, so a memo from the return value
  cannot break a cycle — that caused a **1266-request storm** (#197).
- **v0.3.40**: a slow machine no longer stalls everyone else —
  `prefill_chunk_tokens` bounded decode interruption in TOKENS not time (#191);
  CPU 470s→49s, GPU 14.8s→1.3s. **The first version made GPUs WORSE** (fixed
  per-call cost dominates, so shrinking raised apparent ms/token — a feedback
  loop pinned at the floor); the pacer self-disables if a shrink did not help —
  **do not remove that check**. `active_pipelines` is the COORDINATOR's map and
  never holds peer-served work (#194) — use `state.serving_models`.
- **v0.3.39**: `current_exe()` returns `"...(deleted)"` once the binary is
  replaced, and replacing it IS updating — an updated-not-restarted node failed
  EVERY inference while still advertising its shards (#188). Two "separate
  crashes" were ONE defect (`split_models` keyed `(model,start,end)`, two
  lookups, DashMap order picked, #187). New rule: **timeouts must bound what
  actually varies** (#189, #190).

### v0.3.15 – v0.3.38 (07-23→28) — pointers only; full detail in the round logs

Read the named `memory/round_log_*.md` before re-deriving any of these.

- **v0.3.38**: idle VRAM was never reclaimed — the demand-EMA gate's reprieve
  applied indefinitely. The gate exists because `record_request` is called ONLY
  from the outbound router path, so serving a peer never updates it.
- **v0.3.37**: `swarmllm chat --model X` panicked (clap compares INNER value
  types of same-named args, #183); a wrong-sized shard was registered as HELD
  because startup checks only `exists()` (#184). **Ragged batching measured on
  GPU: NOT worth building** — 4x work, 23% throughput.
- **v0.3.36**: the dashboard rate-limited ITSELF on load (#182) — found by
  loading the page in a real browser, which no test does.
- **v0.3.35**: six fixes, all traced not diffed — retry killed BOTH attempts
  (#180); `FIRST_TOKEN_TIMEOUT` ignored prefill (#181).
  **`parallax_partial_ranges` ships OFF** (~12.0s vs ~10.2s).
- **v0.3.33/34**: **read gotcha #179 before touching connection selection.** A
  relay carrying an INBOUND connection is a bare `/p2p/<peer>` with no transport
  component, so it counted as direct and won every send. And **retraction alone
  is futile — the blacklist is REQUIRED**, since the DHT re-advertises a
  retracted holder.
- **v0.3.30-32**: one `RequestTrace` feeds every surface; stale shard-holder
  claims self-correct; a fresh node gets a 1.5s DHT grace. **"tok/s per node per
  shard" is NOT measurable in a pipeline** — use each segment's share of
  inter-token latency.
- **v0.3.22-29**: the control-token leak chased across four releases was a
  **prompt** bug, not an output-scrubber bug (#169) — `grep "chat template
  failed" node.log` had been firing on every request for releases.
  `inference::finalize_reply_text` now owns the ordered scrub→truncate→trim.
- **v0.3.15-21**: the networking line — NAT relay, additive protocol/feature
  handshake, multi-relay + DHT discovery. **Hole punching verified live 07-25.**
  We published an inbound connection's ephemeral source port as dialable (#165);
  poisoned caches need a node RESTART, not just a new binary.

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
