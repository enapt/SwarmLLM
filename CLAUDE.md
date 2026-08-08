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
├── docs/book/     (mdBook documentation site)
├── vendor/        (patched upstream crates, all workspace-`exclude`d; every patch marked `SwarmLLM patch:`)
│   ├── candle/                (k_quants::matmul tiled; cudarc dynamic-linking hardcode removed)
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
- 3 modal overlays (setup, settings, R127 `#welcome-modal` first-run tour) + 10 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1275 translation keys (1277 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1748 lib tests passing + 11 ignored (env-var-gated real-model + manual smoke; 1759 total), 79 integration tests in `tests/integration/` (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 8 repo-consistency, 1 in `tests/api_key_side_effects.rs` (deliberately an INTEGRATION test — see gotcha #230), 30 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root), 9 in the vendored request-response patch (`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). End-to-end forward-pass bench (no daemon): `SWARM_BENCH_MODEL=<model shard dir> RAYON_NUM_THREADS=4 cargo run --release --no-default-features --features dev --example prefill_bench` — loads a real model from its shard directory and drives `SplitModel::forward` directly, so prompt-processing and decode changes can be A/B'd without chunking policy, batching or the API in the way. Pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown. Attention-op bench: `examples/attn_bench.rs`. Quantized-matmul bench: `cargo run --release --no-default-features --features dev --example qmatmul_bench` — prices the kernel against batch size AND asserts the tiled path is bit-identical to the upstream ordering; it also sweeps rayon pool size. **Use min-of-N on an idle machine**: the same unchanged code path measured 0.42 ms and 0.97 ms across runs on the WSL2 test box. Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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

All 20 build phases complete. All subsystems wired — no stubs. **1748 lib + 79 integration + 8 repo-consistency + 1 api_key_side_effects + 30 swarmllm-types tests passing**; 11 lib + 1 e2e ignored (env-var or manual). Counts re-measured suite-by-suite 2026-08-07. Clippy clean default + features dev,claude-subscription + `--features llama`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — UNRELEASED on main (2026-08-07): fused attention tail on CPU, +19% prompt processing

**Attention's tail was FOUR passes over an 11 MB tensor.** `masked_fill`,
`softmax` and the scale each materialised their own
`[batch, heads, q_len, kv_len]` score temporary and read the previous one back —
~90 MB of traffic per layer per chunk for ~3 MB of arithmetic. `src/inference/
attn_softmax.rs` now does scale + Gemma-2 soft-cap + mask + softmax in ONE pass
(a candle `CustomOp2`), and the mask is additive f32 everywhere instead of a `u8`
predicate plus a float copy the flash arm rebuilt per call. `masked_fill` and
`neg_inf` are gone from three weight structs and every attention signature.

**Measured, min of 3, llama-3.2-3b Q4_K_M, 896-token prompt, 4 threads**
(`examples/prefill_bench.rs`, new): prompt processing **22.04 → 26.14 tok/s
(1.19x)**, decode unchanged. `SWARMLLM_PROFILE=1` on the same run confirms the
mechanism instead of inferring it — **attention core 9066 → 3274 ms (2.77x)**
while every other stage moves under 1.6%. Attention was 22.4% of a prompt chunk
and is now 9.5%.

**Prompt processing is now 84.5% quantized matmul.** Do not spend another round
on CPU attention without re-profiling; two rounds have gone into stages that
turned out to be minorities.

**Two process findings.** An equivalence test where both sides call the same
helper is a tautology — `fused_matches_composed_reference` passes with the scale
INVERTED, which is why a second test compares against candle's own `/ f64`; all
three injected defects were confirmed to go red. And **`SWARMLLM_PROFILE=1` did
nothing on its own** — the dump was env-gated but the clock feeding it started
only under DEBUG logging, so the documented way to profile printed nothing.

**Second change, same day: a short conversation reserved as much memory as a
long one.** candle's `Cache::new(dim, n)` sets `grow_by` AND `max_seq_len` to
`n`, and `append` allocates the full buffer on the FIRST append — so passing a
model's context length (which every site did, because the parameter is *named*
`max_seq_len`) reserved the whole window from token one. **A 100-token chat held
940 MB at 3% utilisation.** `layers::new_kv_cache` + `KV_CACHE_GROWTH_TOKENS`
now grow on demand: **940 → 117 MB** (short chat) and **940 → 235 MB** (896-token
prompt, 22% → 88% utilisation), speed unchanged, no conversation shortened.

**RSS could not have found it**: across that 4-8x reservation change process RSS
moved ~5%, in both directions, because `Tensor::zeros` gets lazily-faulted pages
— it tracks USAGE while the defect was in RESERVATION. `KvCacheStore::occupancy()`
now reports it directly (`DIAG: KV-cache occupancy`). Gotchas **#261**, **#262**.

**Third change: contributing more of your machine made replies SLOWER.** Reading
a prompt and writing a reply want opposite thread counts — re-measured after the
attention fix, prompt processing scales to **1.83x** past decode's optimum while
decode gets **2.0x worse** at the same setting (it is bandwidth-bound at 69% of
roofline). One pool served both, so raising `contribution` sped prompts 1.5x and
slowed replies 13%. `inference::cpu_pools` caps decode only, at one choke point;
A/B in a single binary via `SWARMLLM_DECODE_THREADS=0`: **decode 1.42x at 8
threads, 1.53x at 14, prompt processing IDENTICAL**. No change at the default.
**Re-measuring first was the value** — the old FUTURE_WORK table said 1.18x and
did not know about the perverse effect.

Detail: `memory/round_log_0807_fused_attn.md`.

### Prior — UNRELEASED on main (2026-08-07): FlashAttention on CUDA, compute cap 75 → 80

**Per attention call: prompt processing 2.4-7.8x on an NVIDIA card,
long-context GQA generation 1.4-4.9x. END TO END (measured 2026-08-08, the
number users feel): prompt processing 1.3x at ~900 tokens rising to 2.0x at
~3072, decode unchanged below the 1024 crossover then 2.1-2.4x above it.** The
per-call figure was quoted in the CHANGELOG as if it were the user-visible one;
attention is only part of a forward pass and the quantized matmuls around it are
unchanged. Pre-Ampere cards (RTX 20-series / GTX 16-series) lose the candle GPU
path** and fall back to CPU with an explicit message — `cuda_if_available`
SUCCEEDS on those cards and only module load fails, so without the probe a node
starts clean, logs "GPU detected", and then fails every request. Detail in
`memory/round_log_0807_flash_attn.md`.

**Turning flash-attn on the obvious way would have made generation MUCH slower.**
candle-flash-attn ships **no split-KV kernels**, so one query row leaves the card
idle: **4x-25x slower on MHA decode**, widening with context. GQA reverses above
~1k because `standard_attention` rebuilds the `repeat_kv` expansion every token.
**Gotcha #255 again on a different device** — right kernel OPPOSITE for prefill vs
decode, turning on GQA. Shipped rule `cuda_decode_prefers_standard`: prefill
always flash; decode flash only when GQA AND `k_len >= 1024`.

| | prefill | decode |
|---|---|---|
| phi-3.5 (MHA) | 2.4-4.0x | unchanged — router keeps standard |
| llama-3.2 (GQA) | 4.2-7.8x | 1.4-4.9x past ~1k context |

**PagedAttention was NOT restored — it never ran** (#257): `None` at its own
introducing commit, no call site, deleted three weeks later as "never wired", and
the 46.4 tok/s figure it was credited for was measured *between* those dates.

**Two non-performance findings.** The CUDA flash path **silently dropped
`attn_logit_softcap`** (the plain entry point hardcodes it off) → **Gemma-2
quietly wrong on GPU only**, right shapes and plausible output (#258). And
flash-attn's `dylib=cudart` would have broken the **driver-only install model**
AND failed the CI link outright → `vendor/candle-flash-attn` with `cudart_static`
+ the 18 unreachable bf16 kernels dropped (37 → 19, halving the kernel build).

**Measurement discipline paid three times, all against my own claims:** the
numerics assertion caught the benchmark comparing **causal against full
attention** (relative 3.15 vs an F16 budget of 0.05); median-of-9 gave **3.08 ms
and 191 ms for the same shape** → min-of-20; and a build-time *estimate* stated as
fact (~2.5 h) measured **76 min / 66 min**. `SWARMLLM_FORCE_STANDARD_ATTN=1` A/Bs
the kernel inside ONE binary — two builds differ in more than the kernel.

**Before tagging**: end-to-end GPU tok/s NOT re-measured (README says so rather
than quoting numbers). CI 13/13 green; cache-warm green on all three cells, both
GPU builds verified in-log as compiling `19 of 19 kernels`.

### Earlier rounds — one line each; full detail in `memory/round_log_*.md` + CHANGELOG

Read the named round log before re-deriving any of these.

- **v0.3.81** (08-06/07): **CPU inference, measured instead of guessed** — batching was worth ~1.05x (candle's `k_quants::matmul` made the batch row the OUTER SEQUENTIAL loop → **tiled it**, bit-identical); that moved prefill only 1.2x, so a stage profiler (`SWARMLLM_PROFILE=1`) was built and **every guess about the rest was under 2.5% COMBINED — it was attention**, wrong in BOTH phases in opposite directions. Prefill 4571→640 ms, decode 1368→249 ms/token at 1150 KV. Gotchas #254-#256. Log: `round_log_0806_batching.md`.
- **v0.3.78/.79** (08-06): **the whole prompt pipeline was wrong** — Llama-3
  tokenised at **~2x** (`pre="llama-bpe"` absent from the match → whitespace-split
  fallback), the **system prompt rendered TWICE**, every Llama-3 model **told the
  date was 26 Jul 2024**. All invisible; found by diffing against `tokenizers` +
  `jinja2` references built from the model's OWN vocab/template — **encode→decode
  round-trips passed the whole time**. Also tool-call ids were model-invented, and
  a **network-status panel shipped in English to 20 languages** (key parity checks
  KEYS not VALUES). **.79** shipped AVX2: release binaries had candle's quantized
  kernels **compiled out** (`#[cfg(target_feature)]` is compile-time), **3.09x**;
  x86-64 assets now `-C target-cpu=x86-64-v3` with blocking `-baseline` assets for
  pre-2013 CPUs. Gotchas #246-#253. Log: `round_log_0805_prompt_pipeline.md`.
- **v0.3.60-.77** (08-02→05): **v0.3.72 our API key was being sent to strangers** —
  `forward_to_peer` forwarded the caller's `Authorization` header verbatim and that
  key also guards `/api/admin/*`; **nothing in that code changed, its PREMISE did**
  (#238). **.73/.74** concurrent requests failed OUTRIGHT on CPU-only nodes —
  `forward_batch` lacked the f32 widening and a single item returns early, so it was
  **invisible on GPU and total on CPU** (#241); `max_tokens` off by one (#240); emoji
  → `���` (#239); deleting a model mid-reply destroyed it (#243). **.77** peer latency
  was measured then discarded, wiped by every Identify. Logs:
  `round_log_0805_security.md`, `round_log_releases_0802_0803.md`,
  `round_log_lan_peering_0802.md`, `round_log_0803*.md`.
- **v0.3.49-.59** (07-29→08-01): **SPM tokenizer CLOSED** — stale merge-queue entries
  mis-tokenised **64.9% of inputs** on Phi-3.5's vocab, now 0 vs reference. **A hash
  cannot tell "wrong bytes" from "not all the bytes" — check `size_bytes` FIRST**
  (#203). A model could be freed *while answering* because `active_pipelines` is
  coordinator-only and `serving_models` peer-served-only (#194). `cargo test`
  overwrote a running node's API key (#226).
- **v0.3.39-.46** (07-27→29, `round_log_overnight_0728.md`): `CachedDecoder` built
  `is_sentencepiece:false` under a stale comment, so local replies took a **GPT-2
  byte fallback** (#200) — **peer-served work is decoded on the SERVING side, so
  cross-node checks looked clean**. Overlay trust was satisfiable by coincidence
  (`100.64.0.0/10` is shared CGNAT) (#199). A default living only in
  `#[serde(default)]` never reaches a config the daemon already wrote (#198).
  `is_loopback()` is wrong BOTH ways (#195). `prefill_chunk_tokens` bounded decode
  interruption in TOKENS not time (#191) — **the first fix made GPUs WORSE**; the
  pacer self-disables if a shrink did not help, do not remove that check.
  `current_exe()` returns `"...(deleted)"` once the binary is replaced, and
  replacing it IS updating (#188). **Timeouts must bound what actually varies**
  (#189, #190).
- **v0.3.15-.38** (07-23→28, `round_log_networking_audit.md`): **read gotcha #179
  before touching connection selection** — a relay carrying an INBOUND connection is
  a bare `/p2p/<peer>` and counted as direct, winning every send; and **retraction
  alone is futile, the blacklist is REQUIRED**. Publishing an inbound connection's
  ephemeral port poisoned caches, which need a node RESTART (#165).
  `max_established_per_peer = 1` structurally disabled DCUtR (#163). The
  control-token leak chased across four releases was a **prompt** bug (#169).
  **"tok/s per node per shard" is NOT measurable in a pipeline.**
- **R136-R150** (07-20→23): NAT/internet reachability (UPnP default-on, AutoNAT v1→v2,
  `--anchor`), request cancellation, `gpu_layers` plumbing, per-shard download backoff
  (#150-160); SWARM-SPEC v0.1 cascade; `swarmpool://` invites v2; cross-pool routing.
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
