# SwarmLLM — Claude Code Instructions

> **Quick start**: Read `docs/ARCHITECTURE.md` for the canonical architecture (subsystems, channels, SharedState sub-struct layout, protocols, security model) before exploring code. Per-developer dependency notes may live in `~/.claude/projects/-home-user-SwarmLLM/memory/` outside the repo.

## Project Overview

SwarmLLM is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously participates in a P2P network, runs an HTTP server (OpenAI-compatible API + admin dashboard), and manages local resources (GPU/CPU compute, storage, bandwidth).

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.90+ (set by `redb`; enforced by `msrv_claim_matches_the_dependency_tree` in `tests/repo_consistency.rs` — do not edit by hand, run that test)
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
- `state.models` (`ModelMgmt`) — `acquisition_progress`, `hf_sources`, `auto_manage_*`, `model_trust`, `locked_shards`, `removed_by_user` (user-deleted shard tombstones, 08-21), `prune_history`, `wishlist` (R111), `hf_trending_cache` (R112), `shard_download_backoff` (per-shard exponential download cooldown so one stuck download can't monopolize a slot), `shards_needing_repair` (shards found CORRUPT and awaiting a fresh verified copy — written only via `mark_shard_for_repair`), `shards_pending_verification` (held shards whose EXPECTED hash changed, so their bytes must be re-checked — how a node learns from the swarm that what it serves is wrong), etc.
- `state.metrics` (`MetricsProviders`) — `node_stats`, `inference_requests_total`, `channel_metrics`, `providers_config`, `swarm_capacity` (R110), `hedge_tracker` (R136 Layer 2), `prefetch_orchestrator` (R136 Layer 3), `peer_speed` + `peer_model_warm_at` (measured per-peer prefill/decode speed — sizes segment timeouts and ranks candidates), etc.

Cross-cutting fields (identity, db, peer_registry, model_registry, executor, split_models, `local_memory_refusals`, etc.) remain on the root struct, along with the two configs: `config` (the boot-time snapshot, for startup-only decisions) and `live_config` (the current one — **read it via `state.cfg()`** for anything the user can change while the node runs).

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
│   ├── cli/       (mod, run, status, chat, bench, peers, pool, split_test, update, get_model, remove_model, unload_model (`swarmllm unload` — retire a worker, keep the files), diagnostics (pasteable node report) — R150 `swarmllm get-model` reference-model opt-in)
│   ├── config/    (mod, providers, credit, network, ops, node, inference)
│   ├── daemon/    (mod, manifest, shard_loader, gpu_support (CUDA compute-capability floor + pre-Ampere CPU fallback), dispatch/, startup, background, helpers, supervisor)
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, peer_speed, perf_history, relay, removed_shards, repair, retained_replies (fast-path replies kept for ResendTokens, #438), tp_allreduce)
│   ├── network/   (manager/{mod,events,requests,tensors,identify,commands,connections,dht,shard_transfer}, behaviour, discovery, protocol, transport, relay, peer_cache, redact (address redaction for the pasteable diagnostics report), helpers, pipeline_stream)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface/, auto_manage/, lora)
│   │   ├── auto_manage/  (mod, manager, scoring, download, prune, scan, vram, parallax, wishlist)
│   │   └── huggingface/  (mod, download, private_types, probe, search, shards, watcher, tests)
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, attn_softmax (fused scale+softcap+mask+softmax CPU kernel), decode_attn (single-position CPU attention straight over the KV cache — +24% decode), fast_math (AVX2 expf + fused SiLU×up), cpu_pools (per-phase rayon pools: prefill wide, decode narrow), local_embedder, mem_bandwidth (measured memory bandwidth — what a CPU node advertises as its speed, replacing a hardcoded 50 GB/s assumption), model_worker, token_embedding (quantized token_embd, rows dequantized on lookup — CPU), process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3), trace (per-request route + timing record), prof (SWARMLLM_PROFILE=1 per-stage forward-pass profiler))
│   │   ├── router/       (mod, types, batch, local_exec, distributed_exec, spot_check, tests)
│   │   ├── scheduler/    (mod, parallax, parallax_allocator, tests)
│   │   ├── pipeline/     (mod, distributed, dsd, local, prompt, remote_generate, speculative, tensor_parallel, vision)
│   │   ├── split/        (mod, model, loader, executor, kv_cache, kv_budget, entry, gguf_meta, shard_reader, rope, prefix_cache, hybrid (which layers of a segment go on the card — .145, #431), token_embedding, tests/)
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
├── integrations/openclaw/  (OpenClaw provider plugin, TypeScript — `npm test`; built on OpenClaw's own self-hosted-provider SDK helper; see its README)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── deploy/anchor/ (R143 — hardened bootstrap/relay anchor kit: setup-anchor.sh, systemd unit, config.toml, runbook)
├── packaging/     (swarmllm.service + deb/{postinst,prerm} maintainer scripts — prerm acts on $1: an upgrade must never `systemctl disable`, gotcha #313)
├── docs/          (ARCHITECTURE, CREDITS_DESIGN, FUTURE_WORK, DIAGNOSTICS, REFERENCE_MODELS)
├── docs/book/     (mdBook documentation site)
├── vendor/        (patched upstream crates, all workspace-`exclude`d; every patch marked `SwarmLLM patch:`)
│   ├── candle/                (k_quants::matmul tiled + row-blocked + `vec_dot_rows` multi-row AVX2 Q4_K/Q6_K kernels, bit-identical, exactness-asserted by qmatmul_bench; cudarc dynamic-linking hardcode removed;
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
  - `LocalMemoryUnavailable` → 503, this node's own memory budget refused the load — the one local failure the router re-plans
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
  - `js/core/` — state.js (namespace + shared state + storage keys), utils.js (format helpers, DOM builders, extractErrorMessage, getApiErrorMessage, apiAction, `renderMarkdown`/`inlineMarkdown` — the ONE markdown renderer — plus `renderReplyInto`, the ONE way a model's reply is rendered (chat and compare both go through it; it keeps the markdown source on `_rawText` so Copy returns what the model wrote), and `initTopBannerOffset`, which keeps `--top-banner-height` in step with the DOM so a fixed top banner never covers the header), data.js (data store + authFetch + dedup), tooltip.js (unified popover replacing native `title=`)
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, dashboard-shards.js (pure shard HTML builders exposed as `App.dashboardShards`), models.js, auto-manage-status.js, settings.js, setup.js, welcome.js (R127 — first-run tour modal), downloads.js, notifications.js, identity.js, network-map.js, compare.js, responses.js, pool.js, swarm-tab.js (R111 — wishlist + capacity-plan + performance view), reference-models.js (R148 — shared test-model picker, `App.referenceModels`)
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 3 modal overlays (setup, settings, R127 `#welcome-modal` first-run tour) + 11 `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1335 translation keys (1337 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Frontend payload: **~1065 KB** (html 130 + css 247 + js 687), plus **one** locale at a time (~83 KB en; Thai is the largest at 150 KB) and **88 KB of bundled fonts** (`frontend/fonts/`, IBM Plex Latin subsets — NOT counted by the payload test, which sums `js|css|html` only) — the other 20 locales are never fetched. Measured byte-accurate and capped by `frontend_payload_stays_within_budget` in `tests/repo_consistency.rs`; the long-standing "< 200KB target" in this file was 5.6x out and nothing checked it. The cap is a regression budget, not a goal: it fails on a step change, not on ordinary growth.
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111), **hardware** (2026-09-06 — RAM/GPU/disk; read from a CACHE refreshed on a blocking thread at most every 6 s, because measuring it spawns `nvidia-smi` at ~90 ms and this payload is shared by every client)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- **Counts** (re-measured 2026-09-09): **2492 lib** + 12 ignored with `--features dev,claude-subscription` — the claude-subscription provider carries its own tests, so **always say which feature set a count came from**. 79 integration (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored e2e, 76 repo-consistency, 1 `api_key_side_effects`, 36 `swarmllm-types` (**not** covered by a bare `cargo test`; CI runs it explicitly), 11 in the vendored request-response patch (`--manifest-path vendor/libp2p-request-response/Cargo.toml --lib`). Clippy clean.
- **Benches and harnesses — see `docs/DIAGNOSTICS.md` § Benchmarks for the full list and the traps.** The ones reached for most: `examples/prefill_bench.rs` (drives `SplitModel::forward` directly, no daemon — `SWARM_BENCH_MODEL`, `SWARM_BENCH_PROMPT`, `SWARM_BENCH_DECODE`, `SWARM_BENCH_REPS`, `SWARM_BENCH_DEVICE=cuda`, and `SWARM_BENCH_SPEC_WIDTHS=1,2,4,8` which prices a K-token forward against a 1-token one at the same history depth — the number that decides whether speculation pays; pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown), `examples/qmatmul_bench.rs` (asserts the tiled kernel is bit-identical to upstream), `examples/smoke_test.sh [binary] [port]` (9 checks on an isolated node — run it on the DOWNLOADED release artifact; it now reports checks that COULD NOT RUN separately and never says "all checks passed" over them, and fails fast if the node it started dies — before 2026-08-25 it skipped the three inference checks silently and still claimed success, so "smoke 8/8" had been passing here without ever exercising inference), `examples/constrained_node_test.sh` (a SMALL node reproduced locally — every memory defect since #452 came from one, and none reproduced on a dev box; isolated node + a small `max_ram_mb`, checks the itemised over-budget refusal, then `SIGKILL`s the worker and requires the NEXT request to be served), `examples/soak_test.sh` (`HOURS=` must be a WHOLE number; data dir is per-`PORT`, so two soaks no longer kill each other), `examples/failover_kv_probe.rs` (does a stand-in taking over a segment mid-reply compute the same thing? Two real segments, `SWARM_KV_PROBE_MODEL`/`_SPLIT`/`_DECODE`; **every run carries a control** that holds the history and must reproduce the healthy machine exactly, so the probe can report a null result. Measured 2026-09-09: P(healthy token) 0.997 → 0.119 replacing just 4 of 28 layers), `examples/tokenizer_scaling.rs` (`SWARM_TOK_HEADER` at a model's `gguf_header.bin` — times `encode` against prompt length and prints `tokenizer_model`/`merges`/`scores`, which is what decides WHICH encode path a GGUF takes; a doubling that quadruples the time is the signature. `SWARM_TOK_TEXT` prints the ids for one string, which is how our output gets compared against HuggingFace `tokenizers`. Found #420 and #421).
- **Measurement discipline** (paid for repeatedly): min-of-N on an IDLE box — the same unchanged code measured 0.42 ms and 0.97 ms across runs here, and a benchmark taken while a build runs is worthless. **min-of-N is for benchmarks, not for live measurement** (#367). A/B inside ONE binary via an env switch (`SWARMLLM_DECODE_CALIBRATE=0`, `SWARMLLM_DECODE_ATTN=standard`, `SWARMLLM_FORCE_STANDARD_ATTN`, `SWARMLLM_FLASH_OFFSET_CAUSAL=0`, `SWARMLLM_GQA_DECODE_FLASH=1`, `SWARMLLM_GROUPED_GQA_DECODE_ONLY=1`), never across two builds. **Verify the mechanism fired**, not just that the outcome improved. Pinned reference models: `docs/REFERENCE_MODELS.md`.
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
  - **Layer 2 — `inference.encrypted_pipeline` ("boomerang", DEFAULT FALSE, per-model override).** Forces the local node to handle BOTH the first segment (embedding) AND the last segment (sampling). No remote node ever sees the plaintext prompt OR the sampled tokens. **It does see the intermediate hidden states in PLAINTEXT** — activations are sealed hop-to-hop by Layer 1, and `network/manager/tensors.rs` calls `session_manager.open(...)` and hands the plaintext to the worker, because a matmul cannot run on ciphertext. This is a STRUCTURAL guarantee (the ends stay here), not a cryptographic one against the computing node, and hidden states are partially invertible back to input text — published recovery is ~81% at the final layer, which is also why a "keep more layers local" dial is not the answer (`docs/FUTURE_WORK.md`). Real encrypted compute means FHE/MPC: BERT-Base at 128 tokens on 4x A100 is ~193 s and ~1.3 GB of inter-device traffic, so it is three orders of magnitude away from usable here. Requires the local node to hold shard 0 + final shard. Adds ~1 RTT/token. This is the strongest privacy mode; Layer 1 alone leaves entry/exit nodes able to read the cleartext at their boundary.
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

All 20 build phases complete. All subsystems wired — no stubs. **2492 lib (dev,claude-subscription) — re-measured 2026-09-09, full suite green (exit 0)** + 79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`) + 76 repo-consistency + 1 api_key_side_effects + 36 swarmllm-types tests passing; 12 lib + 1 e2e ignored (env-var or manual). Clippy clean on default, `--no-default-features --features dev,claude-subscription` (that combination is the documented one — plain `--features dev` leaves `embedded` on too and fails on dead code), a `--features llama` check, and `flash-attn --lib`. `cargo audit` reports only advisories already documented and accepted in `SECURITY.md` — at the .165 release, two (`hickory-proto` RUSTSEC-2026-0118/0119, both transitive via libp2p — 0.26.1 is a semver-MAJOR bump pinned by libp2p 0.56, so it is genuinely unreachable without upgrading libp2p; re-checked 2026-09-08, not merely re-accepted) plus the `paste` unmaintained warning.

**Released and deployed: v0.3.166-alpha (2026-09-09, tag on `d2975927`).**
Both nodes verified: local `225e6fe7f2b5cd74` (CUDA artifact — published sha256
matched AND the installed binary byte-identical to the download,
`ggml_cuda_init` present, 0 ERROR since startup, node id kept and
`identity.key` unchanged, inference confirmed; rollback
`~/.local/bin/swarmllm.0.3.165-alpha.bak`, backups pruned to newest 3) and
Proxmox `9684263580c6660f` (.deb `0.3.166-alpha-1` over `0.3.165-alpha-1`, hash
re-verified AFTER transfer, `active` + `enabled`, no `.dpkg-old`, journal errors
"-- No entries --"). Back in each other's peer lists on .166. Gate clean: CI
green on the TAGGED commit across all 13 jobs including the three feature
compile-checks (flash-attn, candle-cuda, windows-gpu-no-flash) and macOS,
**Cache warm green on the bump commit `626185ec`** — the changelog push changed
no dependency so it did not re-run, which is the documented behaviour and was
checked by sha rather than assumed — Docker green, 25 assets, not draft,
`latest`, **smoke 9/9 + shapes 7/7 on the DOWNLOADED artifact against a .165
baseline taken FIRST that scored the same**.
⚠ **CI on the bump commit reads "cancelled"** — the changelog push superseded it
through the concurrency group. Normal; what matters is green on the tagged commit.

**The round that stopped a silent wrong answer.** Seven commits on .165, and the two that matter share one
subject: what happens to a reply when the machine serving part of it goes away.

- **A failed request now hands back what it had already generated.** Report
  #028's second residual: a 4m43s reply, already decoding, was discarded whole
  when its tail peer dropped. The salvage is recorded by the pipeline and taken
  by the router only after the retry has ALSO failed — a complete answer from a
  second route still beats a truncated one — and the executor still returns the
  `Err`, so the peer penalty, the trust update and the logging are untouched.
  `finish_reason` is `"error"` (vLLM's own value); Anthropic has no
  interrupted-turn member and its catch-all is `end_turn`, so that surface gets
  an explicit arm to `max_tokens`.
- **A reply already under way is no longer moved to a machine that cannot
  continue it.** Found while scoping report #028's OTHER residual, and it
  inverted it. A stand-in holds none of the failed machine's KV cache, nothing
  rebuilds it, and `split::executor` reads `kv_offset` from the cache rather
  than from `index_pos` — so the replacement answered from the current token
  alone while the reply carried on looking normal. **Measured**
  (`examples/failover_kv_probe.rs`): replacing 4 of 28 layers takes
  P(the healthy machine's own token) from **0.997 to 0.119**; half the model,
  to **0.005**. There is no safe early window — one decode step in is WORSE
  (0.0000), because what is missing is the PROMPT. Failover is correct on the
  prompt pass and only there, so `failover_can_restore_state` refuses after it
  and the retry/salvage above cover the request.
- **The route-cost instrument could not see a hand-off**, which was most of the
  traffic and all of the traffic it was added for: wired at one of
  `assemble_pipeline_for`'s six returns, with the four hand-off returns ahead of
  it. Zero samples in ten hours of live traffic. **So no log before 2026-09-09
  carries hand-off data, and its absence is not evidence they are rare.**
- Docs: item #8 had been resolved on 09-06 while its triage row still asked for
  a measurement, and survived the 09-08 re-verification pass.

⚠ **Field data for `ASSUMED_FORWARD_PASSES` can only be collected once this is
DEPLOYED** — the live node must run the fixed instrument.

**Previously released and deployed: v0.3.165-alpha (2026-09-08, tag on `3b55bcd4`).**
Both nodes verified: local `225e6fe7f2b5cd74` (CUDA artifact — published sha256
matched AND the installed binary byte-identical to the download, `ggml_cuda_init`
present, 0 ERROR, node id kept, inference confirmed; rollback
`~/.local/bin/swarmllm.0.3.164-alpha.bak`, backups pruned to newest 3) and
Proxmox `9684263580c6660f` (.deb `0.3.165-alpha-1` over `0.3.164-alpha-1`, hash
re-verified AFTER transfer, `active` + `enabled`, no `.dpkg-old`, journal errors
"-- No entries --"). Back in each other's peer lists at 2 ms. Gate fully clean:
CI green on all 13 jobs INCLUDING the three feature compile-checks (flash-attn,
candle-cuda, windows-gpu-no-flash) and macOS, **Cache warm green on the bump
commit itself** (the version bump edits `Cargo.lock`, so it re-ran), Docker
green, 25 assets, not draft, `latest`, **smoke 9/9 + shapes 7/7 on the
DOWNLOADED artifact against a .164 baseline taken FIRST that scored the same**.

**The round a code review of the previous release started.** `/code-review` on
the .164 fixes returned 15 findings; 9 of the top ones were confirmed against
the code before anything was changed, and the headline was uncomfortable —
**#495, shipped the day before, could not observe the failures it was written
for**, for two independent reasons that neither the suite nor the field could
distinguish from a healthy swarm (`expected_attempts_multiplier` reads 1.0 both
for a reliable peer and for one nothing is recording).

- **A transport failure was recorded as a perfect delivery.** The ACK fast-fail
  sweep, a departed peer and a closed stream all end a forward by resolving the
  waiter with a `LayerResult::error` THIS NODE built, which arrives in the same
  arm as a peer's own refusal. With the ACK deadline well inside the segment
  budget, that was the ordinary way a dead link was seen. Fixed structurally:
  `#[serde(skip)] locally_constructed` — the wire format answers "did this come
  from the peer", so no call site can forget.
- **The sample cadence buried what was left**: once per segment per token on the
  chain path against once per reply on the fast path, into one EMA at α=0.3.
  Now the prompt pass plus every failure.
- **The prompt-trust bar was on two of the three paths that assign layer 0**,
  and *the bar failing the search is itself a route into the path with no bar* —
  so tightening it made the exposure MORE likely on that shape. Also found while
  fixing it, and not in the review: `find_standbys` took no `encrypted_pipeline`
  at all, so a remote node could stand by for the prompt or the sampled tokens.
- **Per-peer goodput** (`GoodputEstimator` + `VertexCost::transfer_ms`), the open
  half of issue #21, researched from BBR's bottleneck-bandwidth estimator first:
  a windowed MAX not an average, an app-limited sample may raise but never
  establish or lower, and the round trip comes out before dividing. **Not yet
  field-verified against netem** — the reporter has that lab and has been asked.
- Four smaller: `cheapest_peer` now names the fact that disqualifies it (#460's
  fix reproducing #460 one level down); a worker's memory charge follows the
  ranges it drops; a node told to stay off its card stops advertising the card's
  speed (35.6 tok/s broadcast against 4.95 measured); and `NoComparison` now
  compares instead of discarding a chain already priced cheaper.
- **The routing cost model is now instrumented rather than tuned**: every priced
  route logs `predicted_ms` and `assumed_forward_passes` beside `total_ms` and
  `tokens`. Nothing acts on either; `ASSUMED_FORWARD_PASSES` needs field data.

⚠ **Two claims corrected in place this round, both mine.** `NoComparison` is not
a cold-start window — `measured_gbps` measures on first call, so reaching it
means the bandwidth measurement FAILED (a memory-starved machine). And the
`r134` flake's `try_recv` hypothesis is disproved (the send is awaited inline);
not reproduced in 48 runs including 8 full-suite runs at load 15.95, and
deliberately not "fixed".

⚠ **Null controls earned their keep three times.** One found a real defect in the
goodput estimator (a small forward could ESTABLISH a wrongly-low figure two
window rotations on — worse than unknown). Twice a control did NOT fire, which
is a finding about the guard: one guard matched its own comment, another matched
a binding on the line above. **Plant the violation in the form it would really
appear** (gotchas #502, #503).

**Report #028, fixed across .165's tail and .166.** A 4m43s generation, already
streaming, was lost outright when its tail peer's connection dropped and the
retry answered `Could not decrypt forward`. **A disconnect destroyed the session
key**, and the two ends never drop together: `handle_connection_closed` keeps the
session when the peer is `in_active_pipeline`, but that reads `active_pipelines`
— the COORDINATOR's map, which holds nothing for work a node is SERVING
(gotcha #194). So the server cleared while the coordinator kept sealing. Keys are
now RETIRED rather than destroyed: openable, never sealable, with their own
replay window (`.claude/rules/architecture.md` § "A disconnect retires a session
key"). Of the two residuals, the discarded tokens are fixed in .166 (salvage)
and the missing standby turned out to be the smaller half — see .166's KV
finding above. Still open: a standby cannot be assembled from several nodes
covering a range between them, and that is only worth building once a stand-in
can inherit the conversation state (`docs/FUTURE_WORK.md` items 17 and 18).

**Release-gate warnings — procedure, not history. Read before every release.**
⚠ **Cache warm runs only when the dependency graph changes**, so for a release
whose last commits are docs it is green on the BUMP commit, not the tag — same
code, and the cache it writes lands on `main` where the tag build reads it.
Check the sha it ran on rather than assuming it re-ran. (A version bump edits
`Cargo.lock`, so it does re-run for that commit — confirmed again at .165.)
⚠ **`release_shapes.sh`'s `greedy check has a live control` is PROBABILISTIC.**
It sends one prompt twice at DEFAULT sampling and needs the two to DIFFER; on a
24-token reply they coincide often enough to report COULD NOT RUN — 2 of 5 runs
on one binary at the .165 release. It alternated convincingly against .164
(OK, CNR, CNR, OK) and that was pure chance: a manual replay of the exact
sequence passed, then three further runs passed. **Do not attribute it to a
release on four samples.** And `text()` returns `''` on an ERROR response, so
two FAILED requests compare equal and get that same misleading wording — read
the node log before believing it.
⚠ **`release_shapes.sh` is LOAD-SENSITIVE — take the baseline on an IDLE box.**
`long system prompt, cold` FAILED once beside a `cargo build` and passed on a
quiet re-run. One shapes failure is a re-run candidate, not a signal.
⚠ **`release_shapes.sh` on port 8810 reports "node did not start" — that is the
LIVE node's P2P port** (`8800 + 10`), not a release fault. Use 8819.
⚠ **`gh release download` needs `-R <owner>/<repo>` when run outside a git
directory**, and it fails QUIETLY: a checksum loop that compares two empty
strings reports MATCH. Guard every artifact check on the file existing and the
digest being 64 hex characters.
⚠ **The Proxmox LXC has no `curl`** — a remote health/peer check using it hangs
rather than failing. Check that node from the WSL side, or over `systemctl` and
`journalctl`.

### Earlier rounds — one line each. Detail in `memory/round_log_*.md`,
gotcha numbers index `memory/gotchas.md`. **Read the named round log before
re-deriving any of these.**

- **.164** (09-08): the STABILITY round — #447(iii) (the gate proposes, the search chooses), the prompt-trust bar, and #495 (the loss term had NO input on the chain path, found from OUTSIDE by a contributor's netem lab). ⚠ The .164 field A/B was CONFOUNDED by two different BUILDS (#496). `round_log_0908_stability_round.md`.

- **.162** (09-07): EIGHT fixes from the 16 GB Mac mini tester's eight reports (#017-#024). ⚠ THREE of the eight were WRONG about the CAUSE; #017/#018 are ONE knot and a naive cost comparison would have shipped a 503. `round_log_0907_macmini_eight_reports.md`.

- **.160/.161** (09-06): TEN fixes + a correction, mostly from questions parked on a reporter who was assumed never to answer; #484 a FALSE PRIVACY ASSURANCE, #481 a regression we shipped in .154. .161 was a same-day hotfix — the dashboard would not load AT ALL. `round_log_0906_delegation_shape.md`.
- **.159** (09-06): 16 fixes — the small-machine harness (`examples/constrained_node_test.sh`), #472 a content hash recomputed mid-corrections, the dashboard memory figure that never counted workers. `round_log_0906_dashboard_chat.md`.
- **.158** (09-05): 3 from ONE Qwen3 report — a backend asking for the whole TRAINING window as its context; a prompt left on a CLOSED turn; a `<think>` scratchpad returned as the answer. ⚠ The reporter's headline diagnosis was WRONG (grep against a source-less tarball).
- **.156/.157** (09-05): 23 defects. **#467 — #461 covered 3 of NINE worker-removal sites, missing the CUDA-OOM path both reports came from.** ⚠ Grep the OPERATION; a report's call-site list is symptoms, not scope. `round_log_0905_dead_worker_memory.md`.
- **.153/.154/.155** (09-04): #449 ALL inference broken on every Mac (a socket path over `sun_path`'s 104 bytes — a platform limit a single-platform suite cannot see); #451 an ~8000-token ceiling on EVERY distributed prompt; #454-#460 from one tester's two-node pool. `round_log_0904_*.md`.
- **.147-.152** (09-02/03): the processor route (#444); #438 resend ladder; #440 prefix-cache snapshots never charged (5 → 23 tok/s); three failover defects from ONE trace (#434/#435/#436). `round_log_0902_*.md`, `round_log_0903_processor_route.md`.
- **.142-.146** (09-01): hybrid GPU/CPU layer splitting (#431, 5.0 → 12.25 tok/s); **no Mac had EVER been able to update itself (#430)**; **every node was lying about how fast it is (#428/#429)**; **the "safe to share" button shared everyone's IP address (#426)**. `round_log_0901_diagnostics_privacy.md`.
- **.136-.141** (08-30/31): faults visible ONLY over the network or ONLY on a released binary — #416 multi-byte replies refused as lost, #414 replies arriving twice, #418 244 s vs 0.80 s, #420 naive BPE (141 s), #421 every newline sent as `<unk>`. ⚠ **The perf bug was the LEAD, not the bug**, and a tokenizer compared only against its own past cannot be shown correct. `round_log_0831_tokenizer_quadratic.md` (LAST section first).
- **.132-.135** (08-29/30): **the guards were the defect** — five tested by PLANTING the violation, four could not see what they guard (#413); #410 GGUF headers off an UNBUFFERED file (11.2 s → 0.21 s). ⚠ Split utime/stime BEFORE theorising. `round_log_0830_guard_audit.md`.
- **.15-.131** (07-23→08-28): the era that produced most of the rules. A corrupt shard PROVED to spread and only the ORIGIN settles it (#382), then .121 quarantined the GOOD copy (#384) — **a repair mechanism is a destruction mechanism**; CPU prefill +20-40% / decode +25-37%; 25.7x from a budget read off the BOOT SNAPSHOT (#281, third time, → `SharedState::cfg()`); credits switched OFF; the whole prompt pipeline wrong (#246-#253); AVX2 compiled OUT of releases (3.09x). ⚠ **#367 min-of-N is for benchmarks, NOT live measurement.** `round_log_0825_overnight_watch.md` and siblings.
- **R136-R150 + the 20 build phases**: NAT/reachability, SWARM-SPEC cascade, `swarmpool://` v2, cross-pool routing. `docs/ARCHITECTURE.md` § phase history.

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
