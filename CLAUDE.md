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
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, attn_softmax (fused scale+softcap+mask+softmax CPU kernel), cpu_pools (per-phase rayon pools: prefill wide, decode narrow), local_embedder, model_worker, process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3), trace (per-request route + timing record), prof (SWARMLLM_PROFILE=1 per-stage forward-pass profiler))
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
├── frontend/      (index.html + 10 HTML templates, css/, js/{core/4,components/19,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/)
├── python/        (swarmllm-client SDK)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── deploy/anchor/ (R143 — hardened bootstrap/relay anchor kit: setup-anchor.sh, systemd unit, config.toml, runbook)
├── packaging/     (swarmllm.service + deb/{postinst,prerm} maintainer scripts — prerm acts on $1: an upgrade must never `systemctl disable`, gotcha #313)
├── docs/book/     (mdBook documentation site)
├── vendor/        (patched upstream crates, all workspace-`exclude`d; every patch marked `SwarmLLM patch:`)
│   ├── candle/                (k_quants::matmul tiled; cudarc dynamic-linking hardcode removed;
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
- i18n: 1285 translation keys (1287 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Frontend payload: **~1044 KB** (html 132 + css 230 + js 682), plus **one** locale at a time (~78 KB) — the other 20 are never fetched. Measured byte-accurate and capped by `frontend_payload_stays_within_budget` in `tests/repo_consistency.rs`; the long-standing "< 200KB target" in this file was 5.6x out and nothing checked it. The cap is a regression budget, not a goal: it fails on a step change, not on ordinary growth.
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1868 lib tests passing + 11 ignored with `--features dev,claude-subscription` (1879 total); 1858 + 11 with default features — the claude-subscription provider carries its own tests, so **always say which feature set a count came from**. 79 integration tests in `tests/integration/` (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 27 repo-consistency, 1 in `tests/api_key_side_effects.rs` (deliberately an INTEGRATION test — see gotcha #230), 30 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root; CI runs it explicitly since 2026-08-09), 9 in the vendored request-response patch (CI: path-triggered `.github/workflows/vendored.yml`; locally `cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). End-to-end forward-pass bench (no daemon): `SWARM_BENCH_MODEL=<model shard dir> RAYON_NUM_THREADS=4 cargo run --release --no-default-features --features dev --example prefill_bench` — loads a real model from its shard directory and drives `SplitModel::forward` directly, so prompt-processing and decode changes can be A/B'd without chunking policy, batching or the API in the way. Pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown. Attention-op bench: `examples/attn_bench.rs`. Quantized-matmul bench: `cargo run --release --no-default-features --features dev --example qmatmul_bench` — prices the kernel against batch size AND asserts the tiled path is bit-identical to the upstream ordering; it also sweeps rayon pool size. **Use min-of-N on an idle machine**: the same unchanged code path measured 0.42 ms and 0.97 ms across runs on the WSL2 test box. End-to-end smoke test (any binary, isolated node, never touches a running one): `examples/smoke_test.sh [binary] [port]` — starts, admin API, a setting applying without restart, reload response shape, non-empty inference, streaming, the Anthropic surface, and zero startup errors. Run it after a refactor: the test suite passing and the daemon still booting and serving are different questions. Isolated two-node cross-node test: `examples/two_node_test.sh [binary]` — one node holds the models, one holds none, both on a private gossip network so the scheduler cannot pick a public peer; asserts the reply is non-empty and that the server's `streamed_count` matches the client's `completion_tokens`. **Expect it to fail on a single multi-interface host** (the documented connection-churn case — it names that failure explicitly rather than implying a regression, and reproduces on released binaries); run it across two machines. Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons on ports 8890-8892 — deliberately NOT 8800, and it stops only its own nodes, because the old broad `killall swarmllm` took down whatever else was running) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. Leak soak: `examples/soak_test.sh [binary]` (`HOURS=`, `MODEL=` env) — sustained inference against an ISOLATED node, sampling worker RSS / KV occupancy / threads / fds / ok-fail; it PROVES it is exercising this node (shard-file preflight, private `gossip_network_id`, aborts if a `Pipeline segment` line names a peer — no-bootstrap + no-mDNS is NOT isolation on a machine with a live node, loopback discovery is unconditional, gotcha #311); analyse with `examples/soak_report.sh`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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
- `Task(root-cause)` → sonnet. Reach for it BEFORE attributing a failure or reverting. Its verdict is evidence, not opinion: it must have observed the symptom absent when the suspect is absent.

## Reference Documents

- `docs/ARCHITECTURE.md` — **Primary reference** — current architecture, subsystems, protocols, security model
- `docs/book/` — mdBook documentation site (getting started, API reference, architecture, troubleshooting)
- `docs/DIAGNOSTICS.md` — DIAG: log instrumentation guide for debugging
- `.claude/rules/architecture.md` — invariants (SharedState, broadcast channels, scheduler oracle, centralised wire-format helpers)
- `.claude/rules/diagnosis.md` — **read before blaming any change for any symptom, and before implementing anything non-trivial.** Rule 0: look up how the failure mode is solved elsewhere first — WireGuard's per-keypair replay counter and vLLM's Head-Room Admission each changed an implementation the same day. Then: baseline before blaming, verify the mechanism fired, check the test fails without the fix.
- `.claude/agents/root-cause.md` — `Task(root-cause)` establishes CAUSED / NOT-CAUSED / UNDETERMINED for a suspected cause, and never proposes a fix. Use it before reverting or attributing, especially when the suspect is your own recent change.
- `.claude/sweep-log.jsonl` — per-finding history of every `/sweep` round (status: fixed / wontfix / deferred). Grep before re-reporting potential issues.
- `SwarmLLM_Technical_Specification.docx` — High-level technical specification with architecture rationale

## Status

All 20 build phases complete. All subsystems wired — no stubs. **1868 lib (dev,claude-subscription) / 1858 (default) + 79 integration + 27 repo-consistency + 1 api_key_side_effects + 30 swarmllm-types tests passing**; 11 lib + 1 e2e ignored (env-var or manual). Counts re-measured suite-by-suite 2026-08-16 (post-.98 issue-reduction round). Clippy clean default + features dev,claude-subscription + `--features llama` check.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.98-alpha (2026-08-16): 1.41x faster CPU generation

**Shipped 2026-08-16.** All four workflows green; 25 assets, not a draft.
Verified on the DOWNLOADED CUDA artifact — sha256 match, smoke 8/8, and the
speedup itself re-measured on it: pure-decode 2.35 → 4.69 tok/s (2.0x) at
~2868 tokens context vs the v0.3.97 release binary, CPU-forced, null-control
pattern — before the live node was updated (rollback:
`~/.local/bin/swarmllm.0.3.97.bak`). That verification also surfaced a
PRE-EXISTING long-prompt stall (identical on v0.3.97 — see memory Open items).

GQA decode on CPU no longer expands the KV cache with
`repeat_kv` on every token — the query heads are regrouped as extra matmul rows
against the unexpanded cache (`grouped_gqa_decode_attention` inside
`standard_attention`; identical arithmetic, byte-equivalence pinned against the
expanded path, MHA pinned byte-identical). Decode 4.71 → 6.63 tok/s on
llama-3.2-3b; prefill unchanged; GPUs unaffected.

This **reversed the CPU decode routing**: GQA decode had been sent to the fused
kernel precisely because of the `repeat_kv` cost, and with it gone the same
benchmark reports the opposite at every length (3-9x) — all CPU decode now takes
standard. The control run reproduces the OLD verdict on reverted code, so the
flip is attributable. CUDA GQA-decode routing rested on the same premise and is
now a re-measure candidate (`docs/FUTURE_WORK.md`) — left as-is because GPUs
already route to a fused kernel and this box cannot resolve small GPU deltas
(#267).

**Validated by a 4-hour soak before release**: 3474/3474 requests ok, worker RSS
byte-flat for the final 2.5 h, KV bounded (`memory/soak_0816_cpu_speedup.md`).
The soak tooling itself was hardened first — its initial run silently exercised
a PEER instead of this node (metadata-only model + unconditional loopback
discovery, gotcha #311); `examples/soak_test.sh` now proves it is soaking THIS
node and aborts otherwise. Plus: `load_peer_cache` announces once at startup
instead of on every re-dial pass (141 false "restarted" log lines in 5 h).

**Unreleased on main since the cut**: `7b38322c` — **the .deb upgrade itself ran
`systemctl disable`** (prerm ignored its `$1` role), so every update silently
undid the admin's enablement and left the node down; upgrades now
stop-swap-restart and only a true removal disables. Found live updating the
Proxmox node to .98; one transition release still runs the old prerm (gotcha
#313). Also `e73bb167` (test-node cleanup no longer prefilters by process name;
a renamed release artifact survived it).

### Earlier rounds — one line each; full detail in `memory/round_log_*.md` + CHANGELOG

Read the named round log before re-deriving any of these.

- **v0.3.97** (08-15): **models you own could not be reached** — a node that loaded a model with `-m` announced it under its DISPLAY NAME, invisible as a holder, phantom `shard_count: 0` entries everywhere (three id derivations, no two agreeing → `slugify_model_name`, #310); removing `shard_range` from the config did nothing (#306); loopback-only endpoints told remote admins to fetch a key they had already sent (`LocalOnly`, #309); manifest-less models no longer claim `available`; embeddings 501 not 503.
- **v0.3.96** (08-12/15): **a failure could not report itself** — six defects, one shape: the error's type known in one place, a literal written in another (empty Anthropic streams, `[inference failed]` as assistant text, peers DOCKED for callers' mistakes). Fix = make classification reachable: `classify_error` (HTTP + both SSE encoders), `reclassify_flattened_error` (typeless boundaries), `AnthropicSseEvent::Error`. Gotchas #300-#305. `round_log_0812_error_reporting.md`.
- **v0.3.93/.94** (08-11/12): **a new node could see the swarm's models and run NONE of them** — holders each rewrote a manifest's `publisher` to themselves to earn broadcast rights, erasing each other until nobody broadcast (81 registrations, 50 publishers, one model); the right predicate existed in the startup path, missing from the 30s timer → `manifests_to_gossip` (#296). Plus #295/#297/#298. `round_log_0811_retry_advice.md`.
- **v0.3.92** (08-11): a model could be asked a question in ANOTHER model's format — `resolve_chat_template` returned the RESIDENT model's template on the non-split route (#294). `round_log_0810_kv_mirror.md`.
- **v0.3.91** (08-10/11): **1.4x on long GQA conversations** via an f16 KV mirror — GQA-gated because **the MHA null control came out 3-8% SLOWER**; empty+charged replies from a routing node (#293). **GPU decode is LAUNCH-BOUND — do not size it from FLOPs.**
- **v0.3.90** (08-10): **a GPU could not run ANY unquantized model** — F16/BF16/F32 loaded clean on CUDA then failed 100% of requests (#288-#290); verified with a null control. Also `max_model_len` on `/v1/models`, shard-delete guard.
- **v0.3.89** (08-09/10): **replies from distant peers arrived SCRAMBLED** — each token an independent rr send, `token_id` hardcoded 0 at all five sites → `StreamReassembler` (#282). A "private network" shared PUBLIC gossip topics (#285); a fixed 10s ACK deadline killed a 6s peer (#284). **#283: `pkill -x swarmllm` killed the user's live node.** `round_log_0809_night.md`.
- **v0.3.88** (08-09): **settings saved, said "ok", did nothing** — `state.config` is a boot snapshot, patched around FOUR times → **`SharedState::cfg()`** (#281). **Serving the swarm paid nothing and reported nothing** — accounting lived only on the LESS travelled path (#279/#280) → `record_peer_serve`.
- **v0.3.85/.86/.87** (08-09): every defect was a claim nothing could contradict — batching NEVER engaged (0/156) → **2.4x** GPU; **a rescan on a timer undid the shard split**; credits never persisted (#278); **51 tests ran in NO automation.** `round_log_0808_night.md`.
- **v0.3.82/.83** (08-07/08): CPU fused attention **+19%** prompt processing; CUDA decode routing corrected — the crossover came from timing the call in ISOLATION and was wrong at every length (**#266: measure the FORWARD**). KV reservation: a 100-token chat held 940 MB (#261). **⚠ #267: this box cannot resolve a GPU change below ~25%.** `round_log_0807_*.md`.
- **v0.3.81** (08-06/07): **CPU inference measured, not guessed** — every guess was <2.5% COMBINED; **it was attention**, wrong in BOTH phases in opposite directions (#254-#256). Prefill 4571→640 ms. `round_log_0806_batching.md`.
- **v0.3.78/.79** (08-06): **the whole prompt pipeline was wrong** — Llama-3 tokenised at ~2x, system prompt rendered TWICE, wrong date. All invisible; found by diffing against `tokenizers`/`jinja2` references built from the model's OWN vocab (#246-#253). **.79** shipped AVX2 — release binaries had candle's quantized kernels COMPILED OUT, **3.09x**. `round_log_0805_prompt_pipeline.md`.
- **v0.3.60-.77** (08-02→05): **v0.3.72 our API key was being sent to strangers** — `forward_to_peer` forwarded the caller's `Authorization` verbatim; **nothing in that code changed, its PREMISE did** (#238). Concurrent requests failed outright on CPU-only nodes — invisible on GPU, total on CPU (#241). `round_log_0805_security.md`, `round_log_0803*.md`.
- **v0.3.49-.59** (07-29→08-01): **SPM tokenizer CLOSED** — stale merge-queue entries mis-tokenised **64.9%** of inputs. **A hash cannot tell "wrong bytes" from "not all the bytes" — check `size_bytes` FIRST** (#203). `cargo test` overwrote a running node's API key (#226).
- **v0.3.39-.46** (07-27→29): local replies took a **GPT-2 byte fallback** under a stale comment (#200) — **peer-served work is decoded on the SERVING side, so cross-node checks looked clean**. `current_exe()` returns `"…(deleted)"` once the binary is replaced (#188). **Timeouts must bound what actually varies** (#189/#190). `round_log_overnight_0728.md`.
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
