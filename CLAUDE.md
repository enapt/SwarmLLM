# SwarmLLM — Claude Code Instructions

> **Quick start**: Read `docs/ARCHITECTURE.md` for the canonical architecture (subsystems, channels, SharedState sub-struct layout, protocols, security model) before exploring code. Per-developer dependency notes may live in `~/.claude/projects/-home-user-SwarmLLM/memory/` outside the repo.

## Project Overview

SwarmLLM is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously participates in a P2P network, runs an HTTP server (OpenAI-compatible API + admin dashboard), and manages local resources (GPU/CPU compute, storage, bandwidth).

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.89+ (set by `redb`; enforced by `msrv_claim_matches_the_dependency_tree` in `tests/repo_consistency.rs` — do not edit by hand, run that test)
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
- `state.models` (`ModelMgmt`) — `acquisition_progress`, `hf_sources`, `auto_manage_*`, `model_trust`, `locked_shards`, `prune_history`, `wishlist` (R111), `hf_trending_cache` (R112), `shard_download_backoff` (per-shard exponential download cooldown so one stuck download can't monopolize a slot), etc.
- `state.metrics` (`MetricsProviders`) — `node_stats`, `inference_requests_total`, `channel_metrics`, `providers_config`, `swarm_capacity` (R110), `hedge_tracker` (R136 Layer 2), `prefetch_orchestrator` (R136 Layer 3), `peer_speed` + `peer_model_warm_at` (measured per-peer prefill/decode speed — sizes segment timeouts and ranks candidates), etc.

Cross-cutting fields (identity, db, peer_registry, model_registry, executor, split_models, etc.) remain on the root struct, along with the two configs: `config` (the boot-time snapshot, for startup-only decisions) and `live_config` (the current one — **read it via `state.cfg()`** for anything the user can change while the node runs).

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
│   ├── cli/       (mod, run, status, chat, bench, peers, pool, split_test, update, get_model, remove_model — R150 `swarmllm get-model` reference-model opt-in)
│   ├── config/    (mod, providers, credit, network, ops, node, inference)
│   ├── daemon/    (mod, manifest, shard_loader, gpu_support (CUDA compute-capability floor + pre-Ampere CPU fallback), dispatch/, startup, background, helpers, supervisor)
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, peer_speed, perf_history, relay, tp_allreduce)
│   ├── network/   (manager/{mod,events,requests,tensors,identify,commands,connections,dht,shard_transfer}, behaviour, discovery, protocol, transport, relay, peer_cache, helpers, pipeline_stream)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface/, auto_manage/, lora)
│   │   ├── auto_manage/  (mod, manager, scoring, download, prune, scan, vram, parallax, wishlist)
│   │   └── huggingface/  (mod, download, private_types, probe, search, shards, watcher, tests)
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, attn_softmax (fused scale+softcap+mask+softmax CPU kernel), cpu_pools (per-phase rayon pools: prefill wide, decode narrow), local_embedder, mem_bandwidth (measured memory bandwidth — what a CPU node advertises as its speed, replacing a hardcoded 50 GB/s assumption), model_worker, token_embedding (quantized token_embd, rows dequantized on lookup — CPU), process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3), trace (per-request route + timing record), prof (SWARMLLM_PROFILE=1 per-stage forward-pass profiler))
│   │   ├── router/       (mod, types, batch, local_exec, distributed_exec, spot_check, tests)
│   │   ├── scheduler/    (mod, parallax, parallax_allocator, tests)
│   │   ├── pipeline/     (mod, distributed, dsd, local, prompt, remote_generate, speculative, tensor_parallel, vision)
│   │   ├── split/        (mod, model, loader, executor, kv_cache, entry, gguf_meta, shard_reader, rope, prefix_cache, tests/)
│   │   │   └── tests/    (mod, common, core, gqa, gemma2, moe_mla, llama4_glm4)
│   │   ├── chat_template/ (mod, parser, eval, fallbacks, tests, fixtures/llama3_official.jinja)
│   │   └── layers/       (mod, qwen35)
│   ├── credit/    (ledger, transaction, priority, anti_gaming, trust, escrow)
│   ├── identity/  (keypair, nickname)
│   ├── crypto/    (session, pipeline_seal, gossip_seal, key_rotation, provider_keys)
│   ├── pool/      (types, crypto, manager/, forward, scope)
│   ├── api/       (server, sse, tool_parse (local-model tool-call parser), admin, admin_providers, websocket, middleware, identity, pool, metrics, providers, claude_sub*, mod, openai/, anthropic/, mcp/, admin_hf/, admin_models/, claude_session/)
│   ├── storage/   (db)
│   └── health/    (monitor, rebalancer)
├── frontend/      (index.html + 10 HTML templates, css/, js/{core/4,components/19,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/, fonts/ (IBM Plex woff2, SIL OFL — see LICENSE-THIRD-PARTY.md))
├── python/        (swarmllm-client SDK)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── deploy/anchor/ (R143 — hardened bootstrap/relay anchor kit: setup-anchor.sh, systemd unit, config.toml, runbook)
├── packaging/     (swarmllm.service + deb/{postinst,prerm} maintainer scripts — prerm acts on $1: an upgrade must never `systemctl disable`, gotcha #313)
├── docs/          (ARCHITECTURE, CREDITS_DESIGN, FUTURE_WORK, DIAGNOSTICS, REFERENCE_MODELS)
├── docs/book/     (mdBook documentation site)
├── vendor/        (patched upstream crates, all workspace-`exclude`d; every patch marked `SwarmLLM patch:`)
│   ├── candle/                (k_quants::matmul tiled; cudarc dynamic-linking hardcode removed;
│   │                          QTensor::gather_rows — read rows out of a quantized tensor
│   │                          without dequantizing it whole, CPU slice + CUDA index_select
│   │                          over a byte view; the embedding table is the caller;
│   │                          CUDA dequantize_f16 falls back to the host for UNQUANTIZED
│   │                          F16/BF16/F32 GGUFs like its dequantize sibling already did —
│   │                          without it a GPU node loaded such a model then failed every
│   │                          request, gotcha #288)
│   ├── candle-flash-attn/     (cudart linked STATICALLY so the binary needs only the display driver;
│   │                          18 bf16 kernels + the FP16_SWITCH bf16 branch dropped — unreachable, 37→19)
│   ├── candle-paged-attention/ (kernels only — NOTHING references it; PagedAttention was never wired, #257)
│   ├── libp2p-request-response/ (9 tests, `--lib`)
│   └── float8/
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
- 3 modal overlays (setup, settings, R127 `#welcome-modal` first-run tour) + 11 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1315 translation keys (1317 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Frontend payload: **~1065 KB** (html 130 + css 247 + js 687), plus **one** locale at a time (~83 KB en; Thai is the largest at 150 KB) and **88 KB of bundled fonts** (`frontend/fonts/`, IBM Plex Latin subsets — NOT counted by the payload test, which sums `js|css|html` only) — the other 20 locales are never fetched. Measured byte-accurate and capped by `frontend_payload_stays_within_budget` in `tests/repo_consistency.rs`; the long-standing "< 200KB target" in this file was 5.6x out and nothing checked it. The cap is a regression budget, not a goal: it fails on a step change, not on ordinary growth.
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1912 lib tests passing + 11 ignored with `--features dev,claude-subscription` (1923 total); 1902 + 11 with default features; 1903 with `dev,candle-cuda` (the CUDA embedding-gather test only runs there) — the claude-subscription provider carries its own tests, so **always say which feature set a count came from**. 79 integration tests in `tests/integration/` (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 30 repo-consistency, 1 in `tests/api_key_side_effects.rs` (deliberately an INTEGRATION test — see gotcha #230), 30 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root; CI runs it explicitly since 2026-08-09), 9 in the vendored request-response patch (CI: path-triggered `.github/workflows/vendored.yml`; locally `cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). End-to-end forward-pass bench (no daemon): `SWARM_BENCH_MODEL=<model shard dir> RAYON_NUM_THREADS=4 cargo run --release --no-default-features --features dev --example prefill_bench` — loads a real model from its shard directory and drives `SplitModel::forward` directly, so prompt-processing and decode changes can be A/B'd without chunking policy, batching or the API in the way. Pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown. Attention-op bench: `examples/attn_bench.rs`. Quantized-matmul bench: `cargo run --release --no-default-features --features dev --example qmatmul_bench` — prices the kernel against batch size AND asserts the tiled path is bit-identical to the upstream ordering; it also sweeps rayon pool size. **Use min-of-N on an idle machine**: the same unchanged code path measured 0.42 ms and 0.97 ms across runs on the WSL2 test box. End-to-end smoke test (any binary, isolated node, never touches a running one): `examples/smoke_test.sh [binary] [port]` — starts, admin API, a setting applying without restart, reload response shape, non-empty inference, streaming, the Anthropic surface, and zero startup errors. Run it after a refactor: the test suite passing and the daemon still booting and serving are different questions. Isolated two-node cross-node test: `examples/two_node_test.sh [binary]` — one node holds the models, one holds none, both on a private gossip network so the scheduler cannot pick a public peer; asserts the reply is non-empty and that the server's `streamed_count` matches the client's `completion_tokens`. **Expect it to fail on a single multi-interface host** (the documented connection-churn case — it names that failure explicitly rather than implying a regression, and reproduces on released binaries); run it across two machines. Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons on ports 8890-8892 — deliberately NOT 8800, and it stops only its own nodes, because the old broad `killall swarmllm` took down whatever else was running) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. Leak soak: `examples/soak_test.sh [binary]` (`HOURS=`, `MODEL=` env) — sustained inference against an ISOLATED node, sampling worker RSS / KV occupancy / threads / fds / ok-fail; it PROVES it is exercising this node (shard-file preflight, private `gossip_network_id`, aborts if a `Pipeline segment` line names a peer — no-bootstrap + no-mDNS is NOT isolation on a machine with a live node, loopback discovery is unconditional, gotcha #311); analyse with `examples/soak_report.sh`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
- Unit tests: in-module `#[cfg(test)]` blocks
- Integration tests: `tests/integration/` — multi-node simulations with `--test-threads=1`
- Real-model spawn-and-infer test: set `SWARMLLM_TEST_MODEL_DIR` to a fully-populated model directory (e.g. `~/.local/share/swarmllm/models/tinyllama-1.1b-...`) and run `cargo test --test integration_phase10_11 -- --ignored end_to_end`. No synthetic GGUF fixture is committed; see `docs/ARCHITECTURE.md` § Deferred Items.
- CI pipeline: `cargo fmt` → `cargo clippy --all-targets -- -D warnings` → `cargo test` → `cargo build --release`

## Key Design Decisions

- Config priority: CLI flags > env vars (SWARMLLM_ prefix) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
- Data dir: `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- Port layout: HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- Credit transactions require dual Ed25519 signatures (serving node + requesting node)
- **Credits are DORMANT (2026-08-17) — they gate nothing.** `MIN_BALANCE_FOR_INFERENCE = 0` and `calculate_tier` returns a constant, so no balance affects who is served, how fast, or what the dashboard shows. The accounting still runs. Reason: credit has never moved between nodes as payment for work — each node mints its own figure (the one real transfer, pool credit *forwarding*, just concentrates self-minted numbers). Design + exit criteria in `docs/CREDITS_DESIGN.md`; `credits_stay_dormant` in `tests/repo_consistency.rs` fails the build if a balance starts gating again.
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
- `Task(root-cause)` → sonnet. Reach for it BEFORE attributing a failure or reverting. Its verdict is evidence, not opinion: it must have observed the symptom absent when the suspect is absent.

## Reference Documents

- `docs/ARCHITECTURE.md` — **Primary reference** — current architecture, subsystems, protocols, security model
- `docs/book/` — mdBook documentation site (getting started, API reference, architecture, troubleshooting)
- `docs/DIAGNOSTICS.md` — DIAG: log instrumentation guide for debugging
- `docs/CREDITS_DESIGN.md` — **read before touching credits.** Why the economy is
  switched off, what is actually true today, the bilateral-settlement design, and
  the exit criteria that must hold before any of it is switched back on
- `docs/FUTURE_WORK.md` — deferred items with enough context to pick up cold
- `.claude/rules/architecture.md` — invariants (SharedState, broadcast channels, scheduler oracle, centralised wire-format helpers)
- `.claude/rules/diagnosis.md` — **read before blaming any change for any symptom, and before implementing anything non-trivial.** Rule 0: look up how the failure mode is solved elsewhere first — WireGuard's per-keypair replay counter and vLLM's Head-Room Admission each changed an implementation the same day. Then: baseline before blaming, verify the mechanism fired, check the test fails without the fix.
- `.claude/agents/root-cause.md` — `Task(root-cause)` establishes CAUSED / NOT-CAUSED / UNDETERMINED for a suspected cause, and never proposes a fix. Use it before reverting or attributing, especially when the suspect is your own recent change.
- `.claude/sweep-log.jsonl` — per-finding history of every `/sweep` round (status: fixed / wontfix / deferred). Grep before re-reporting potential issues.
- `SwarmLLM_Technical_Specification.docx` — High-level technical specification with architecture rationale

## Status

All 20 build phases complete. All subsystems wired — no stubs. **1912 lib (dev,claude-subscription) / 1902 (default) / 1903 (dev,candle-cuda) + 79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`) + 30 repo-consistency + 1 api_key_side_effects + 30 swarmllm-types tests passing**; 11 lib + 1 e2e ignored (env-var or manual). Counts re-measured 2026-08-18 (post v0.3.103; lib counts confirmed on both dev,claude-subscription and default). Clippy clean on default, `dev,claude-subscription`, **`dev,candle-cuda --all-targets`** (the config that hides gated breakage, #264) and a `--features llama` check.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.101 / .102 / .103-alpha (2026-08-18): models need ~750 MB less, machines say what they are, + an h2 security patch

Three releases in one day, verified on the DOWNLOADED artifact. **.103 is a security patch only** — `h2` 0.4.13 → 0.4.16 for RUSTSEC-2026-0258 (unbounded empty DATA frames, published 08-17). It was caught by CI's `cargo audit` on the commit AFTER the .102 tag, so **.102 shipped vulnerable**; anyone on .102 or earlier should update.
Detail: `memory/round_log_0818_quantized_embedding.md`.

- **Models need up to 750 MB less memory, CPU and GPU.** `token_embd.weight` is
  read a row at a time out of the QUANTIZED table instead of being dequantized
  whole at load (`inference::split::token_embedding`, backed by the vendored
  `QTensor::gather_rows`). Measured on llama-3.2-3b: **754 MB off CPU peak RSS,
  736 MiB off an RTX 3070**, against 751 predicted. Weight-tied models save the
  FULL f16 copy — they held the tensor twice. Bit-identical, asserted by exact
  equality on both devices.
- **`GPU_ADMISSION_KV_CONTEXT`** — GPU admission may charge under the worst case
  ONLY where a runtime check catches the difference (GPU has one, CPU does not).
  Raising `max_seq_len_override` no longer costs a model its place on the card.
  `DEFAULT_MAX_SEQ_LEN` 4096 → 8192: an agentic system prompt is ~5000 tokens and
  did not fit AT ALL.
- **Peer delegation** (`scheduler::delegation_target`) — a node whose GPU cannot
  fit a model hands it to a nearby capable peer instead of falling back to its own
  CPU. With prompt privacy on it builds the BOOMERANG instead (first+last local,
  middle to the peer, encrypted) — privacy changes the shape, it does not veto the
  peer. Verified on two nodes both ways.
- **A machine's speed is measured, not assumed** (`inference::mem_bandwidth`) —
  every CPU node advertised a hardcoded 1.70 tok/s; now measured (29.9 GB/s on
  this laptop vs 50 assumed). `NodeCapability.cpu` names the processor, so a peer
  without a card is no longer the bare word "CPU". Interop tested against the
  released binary in both directions.
- **Honesty fixes** — the dashboard no longer contradicts the daemon about whether
  a model fits; `swarmllm bench` warms up before timing; the CPU tooltip no longer
  promises credits.

**Confirmed on the real swarm, not just the bench.** Four of five nodes now run
.103 and report measured speeds: anchor EPYC-Milan (2c) **~0.31 tok/s**, Proxmox
i5-10500T (6c) ~0.85, Oz i7-7700 (6c) ~0.92 — against the hardcoded **1.70** every
one of them used to claim, so the old constant overstated the smallest by 5.5x.
Nodes still on .100/.101 correctly show no processor details, which is the
mixed-version fallback working live. **The `.deb` upgrade now restarts the service
itself and leaves the unit enabled** — the long-standing "always `systemctl start`
afterwards" instruction is obsolete (see `memory/env_proxmox_test_node.md`).

**Gotchas from this round: #326-#334.** Most load-bearing: **#328** (a single GPU
A/B pair is worthless — within-arm spread 15-29%; find the measurement the machine
CAN make), **#330/#333** (a gossiped field with no consumer is unverified — free
VRAM was zero and CPU speed was a constant, swarm-wide, for as long as they existed),
**#326** (candle's `as_t_slice` takes a `Cow` by value and returns a slice into it —
`Cow::Owned` is a use-after-free), **#327** (`temperature: 0` + seed does NOT
reproduce, so generated text cannot compare two inference paths here), **#334**
(`cargo audit` runs in CI, NOT in Release — a tag can publish while the tree is
vulnerable, and the failure lands on the NEXT commit wearing its title).

### Earlier rounds — one line each; full detail in `memory/round_log_*.md` + CHANGELOG

Read the named round log before re-deriving any of these.

- **v0.3.100** (08-17): credits switched OFF (they were enforced: a -1000 floor refused peers, tier gave 8x concurrency); error hints translated across 21 locales; log severity follows blame (#315-#317); release build 54 min → 16m29s (#318/#319). `round_log_0817_honesty.md`.
- **v0.3.99** (08-16): long prompts stalled for MINUTES — `insert_from_kv` snapshotted at EVERY block boundary, O(prompt²) ≈ 15 GB copied per 2.9k-token request (repeats 115.5s→4.1s, #312); silent-peer 500s→503 (`PeerUnresponsive`/`SegmentFailoverExhausted`); Docker `latest` had NEVER been published (#314).
- **v0.3.98** (08-16): **1.41x CPU generation** — GQA decode stopped expanding the KV cache with `repeat_kv` (`grouped_gqa_decode_attention`); this REVERSED the CPU decode-kernel routing at every length. 4h soak, 3474/3474 ok.
- **v0.3.97** (08-15): **models you own could not be reached** — a node that loaded a model with `-m` announced it under its DISPLAY NAME, invisible as a holder, phantom `shard_count: 0` entries everywhere (three id derivations, no two agreeing → `slugify_model_name`, #310); removing `shard_range` from the config did nothing (#306); loopback-only endpoints told remote admins to fetch a key they had already sent (`LocalOnly`, #309); manifest-less models no longer claim `available`; embeddings 501 not 503.
- **v0.3.96** (08-12/15): **a failure could not report itself** — six defects, one shape: the error's type known in one place, a literal written in another (empty Anthropic streams, `[inference failed]` as assistant text, peers DOCKED for callers' mistakes). Fix = make classification reachable: `classify_error` (HTTP + both SSE encoders), `reclassify_flattened_error` (typeless boundaries), `AnthropicSseEvent::Error`. Gotchas #300-#305. `round_log_0812_error_reporting.md`.
- **v0.3.88-.94** (08-09→12): four rounds where the fix was always "make the one right answer reachable" — a new node could see the swarm's models and run NONE of them (holders erased each other's `publisher` until nobody broadcast → `manifests_to_gossip`, #296); a model asked a question in ANOTHER model's format (#294); **settings saved, said "ok", did nothing** (`state.config` is a boot snapshot → **`SharedState::cfg()`**, #281); serving the swarm paid and reported nothing (accounting on the LESS travelled path → `record_peer_serve`, #279/#280); an f16 KV mirror for 1.4x on long GQA chats — **GQA-gated because the MHA null control came out SLOWER**; a GPU could not run ANY unquantized model (#288-#290); **replies from distant peers arrived SCRAMBLED** (`token_id` hardcoded 0 at all five sites → `StreamReassembler`, #282); a "private network" shared PUBLIC gossip topics (#285). **GPU decode is LAUNCH-BOUND — do not size it from FLOPs. ⚠ #283: `pkill -x swarmllm` killed the user's live node — kill by PID.** `round_log_0811_retry_advice.md`, `round_log_0810_kv_mirror.md`, `round_log_0809_night.md`.
- **v0.3.81-.87** (08-06→09): the round where every defect was a claim nothing could contradict — batching NEVER engaged (0/156) → **2.4x** GPU; a timed rescan undid the shard split; CPU inference measured not guessed (**it was attention**, wrong in BOTH phases in opposite directions, #254-#256); CPU fused attention +19%; KV reservation held 940 MB for a 100-token chat (#261). **⚠ #266 measure the FORWARD, not the isolated call. ⚠ #267 this box cannot resolve a GPU change below ~25%.** `round_log_0808_night.md`, `round_log_0807_*.md`, `round_log_0806_batching.md`.
- **v0.3.78/.79** (08-06): **the whole prompt pipeline was wrong** — Llama-3 tokenised at ~2x, system prompt rendered TWICE, wrong date; all invisible, found by diffing against `tokenizers`/`jinja2` references built from the model's OWN vocab (#246-#253). **.79** shipped AVX2 — releases had candle's quantized kernels COMPILED OUT, **3.09x**. `round_log_0805_prompt_pipeline.md`.
- **v0.3.39-.77** (07-27→08-05): **v0.3.72 our API key was being sent to strangers** — nothing in that code changed, its PREMISE did (#238); SPM tokenizer mis-tokenised **64.9%** of inputs (**a hash cannot tell "wrong bytes" from "not all the bytes" — check `size_bytes` FIRST**, #203); local replies took a GPT-2 byte fallback under a stale comment (#200). `round_log_0805_security.md`, `round_log_0803*.md`, `round_log_overnight_0728.md`.
- **v0.3.15-.38** (07-23→28): **read #179 before touching connection selection** — a relay carrying an INBOUND connection is a bare `/p2p/<peer>`, counted as direct, and wins every send; **retraction alone is futile, the blacklist is REQUIRED**. `max_established_per_peer = 1` structurally disabled DCUtR (#163). `round_log_networking_audit.md`.
- **R136-R150** (07-20→23): NAT/internet reachability (UPnP default-on, AutoNAT v1→v2, `--anchor`), request cancellation, `gpu_layers` plumbing, per-shard download backoff (#150-#160); SWARM-SPEC v0.1 cascade; `swarmpool://` invites v2; cross-pool routing.
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
