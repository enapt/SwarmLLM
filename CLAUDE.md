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
- i18n: 1322 translation keys (1324 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Frontend payload: **~1065 KB** (html 130 + css 247 + js 687), plus **one** locale at a time (~83 KB en; Thai is the largest at 150 KB) and **88 KB of bundled fonts** (`frontend/fonts/`, IBM Plex Latin subsets — NOT counted by the payload test, which sums `js|css|html` only) — the other 20 locales are never fetched. Measured byte-accurate and capped by `frontend_payload_stays_within_budget` in `tests/repo_consistency.rs`; the long-standing "< 200KB target" in this file was 5.6x out and nothing checked it. The cap is a regression budget, not a goal: it fails on a step change, not on ordinary growth.
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- **Counts** (re-measured 2026-09-03 evening): **2318 lib** + 12 ignored with `--features dev,claude-subscription` — the claude-subscription provider carries its own tests, so **always say which feature set a count came from**. 79 integration (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored e2e, 56 repo-consistency, 1 `api_key_side_effects`, 30 `swarmllm-types` (**not** covered by a bare `cargo test`; CI runs it explicitly), 11 in the vendored request-response patch (`--manifest-path vendor/libp2p-request-response/Cargo.toml --lib`). Clippy clean.
- **Benches and harnesses — see `docs/DIAGNOSTICS.md` § Benchmarks for the full list and the traps.** The ones reached for most: `examples/prefill_bench.rs` (drives `SplitModel::forward` directly, no daemon — `SWARM_BENCH_MODEL`, `SWARM_BENCH_PROMPT`, `SWARM_BENCH_DECODE`, `SWARM_BENCH_REPS`, `SWARM_BENCH_DEVICE=cuda`, and `SWARM_BENCH_SPEC_WIDTHS=1,2,4,8` which prices a K-token forward against a 1-token one at the same history depth — the number that decides whether speculation pays; pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown), `examples/qmatmul_bench.rs` (asserts the tiled kernel is bit-identical to upstream), `examples/smoke_test.sh [binary] [port]` (9 checks on an isolated node — run it on the DOWNLOADED release artifact; it now reports checks that COULD NOT RUN separately and never says "all checks passed" over them, and fails fast if the node it started dies — before 2026-08-25 it skipped the three inference checks silently and still claimed success, so "smoke 8/8" had been passing here without ever exercising inference), `examples/soak_test.sh` (`HOURS=` must be a WHOLE number; data dir is per-`PORT`, so two soaks no longer kill each other), `examples/tokenizer_scaling.rs` (`SWARM_TOK_HEADER` at a model's `gguf_header.bin` — times `encode` against prompt length and prints `tokenizer_model`/`merges`/`scores`, which is what decides WHICH encode path a GGUF takes; a doubling that quadruples the time is the signature. `SWARM_TOK_TEXT` prints the ids for one string, which is how our output gets compared against HuggingFace `tokenizers`. Found #420 and #421).
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

All 20 build phases complete. All subsystems wired — no stubs. **2318 lib (dev,claude-subscription) — re-measured 2026-09-04, full suite green (exit 0)** + 79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`) + 56 repo-consistency + 1 api_key_side_effects + 30 swarmllm-types tests passing; 12 lib + 1 e2e ignored (env-var or manual). Clippy clean on default, `--no-default-features --features dev,claude-subscription` (that combination is the documented one — plain `--features dev` leaves `embedded` on too and fails on dead code), a `--features llama` check, and `flash-attn --lib`. `cargo audit` reports only advisories already documented and accepted in `SECURITY.md` — at the .154 release, two (`hickory-proto` RUSTSEC-2026-0118/0119, both transitive via libp2p) plus the `paste` unmaintained warning.

**Next up (after the v0.3.154 release)**: (0) **field-verify the three .154
fixes** — #451 with an 11k-token distributed prompt, #452 with the Mac mini
owner (its arithmetic is MARGINAL on their numbers: ~36 layers allowed against
36 asked, so ask for their own `max_hostable_layers=` line), #453 with an
OpenClaw user; (1) **#447 (iii)** — make the priced search the ONE
decision-maker when the local node is on its processor (`delegation_target`
becomes a filter; the DP compares every shape). Halves (i)+(ii) shipped in .153;
then re-run the live #444 measurement, which is still unmeasured; (2) **#446
first half** — a multi-MB forward over ONE QUIC stream is killed by quinn's
1024-gap limit: rank TCP above QUIC for large requests, or chunked-over-RR
behind a feature bit (the new contributor's AWS node is the right place to
reproduce it); (3) prefix-keyed remote KV across turns; (4) f16 stored KV on
small cards; (5) accept/reject at the tail — also the precondition for issue
#21's economics; then the #438 structural rung, peer↔peer RTT gossip, ring
decode, prefill microbatching. Ordered list with pointers in `memory/MEMORY.md`
§ NEXT UP; entries in `docs/FUTURE_WORK.md`. **Parked**: the OpenClaw ClawHub /
npm publish, which needs the user's credentials.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

**Released and deployed: v0.3.154-alpha (2026-09-04, tag on `4b3a8080`).** Local
`225e6fe7` (CUDA asset, downloaded sha256 OK, size == published, ready in
seconds, 0 ERROR lines, `release ok` served, rollback
`~/.local/bin/swarmllm.0.3.153-alpha.bak`) and Proxmox `96842635` (.deb
`0.3.154-alpha-1` over `0.3.153-alpha-1`, `enabled` + `active`, no `.dpkg-old`,
journal errors "-- No entries --") both on it, node ids kept — and the USER
updated the ANCHOR (`bfbfeebc`) to .154 the same afternoon. **5 of the 6 nodes
were on .154 within ~20 minutes, most unattended**; only the Belgian RTX 4050
(`bf7b3263`) lagged on .153, which matters because #451's cap is still live
there. Gate: CI + Cache warm green (Cache warm ~22 min this time, not the
1h56m of .153), 25 assets, not draft, `latest`, sha256 CUDA + deb,
`ggml_cuda_init` = 1, **smoke 9/9 (none skipped) + shapes 7/7 on the DOWNLOADED
artifact** — and both harnesses were baselined green on .153 FIRST, so a failure
would have been a real regression rather than a harness fault.
Three fixes, all reported from real nodes this week: **#451** an ~8000-token
ceiling on every distributed prompt (a fixed element cap in `bytes_to_tensor`
that was exactly 8192 positions at hidden 4096 — silent since March, because
local inference never serializes an activation); **#452** the planner priced the
LOCAL node as memory-unbounded while the loader priced a whole model for a
partial segment, so a whole-model holder could serve no PART of a model too big
for it; **#453** a tool-carrying reply got no streaming at all (120 content
deltas → 1), which is what "OpenClaw times out" was.

Prior line: v0.3.153-alpha (2026-09-04, tag on `8f8edd61`) — the two macOS fixes (#449/#450), #448's storage budget, six future-work closures.

Older line: v0.3.152-alpha (2026-09-03, tag on `bbecda89`). Local
`225e6fe7` (CUDA asset, installed sha256 `cd9eefec…` == published, ready in 2 s,
0 ERROR lines, first inference OK, rollback `~/.local/bin/swarmllm.0.3.151-alpha.bak`)
and Proxmox `96842635` (.deb `0.3.152-alpha-1` over `0.3.151-alpha-1`, stayed
`enabled` + `active`, 0 errors) both on it, node ids kept. Gate: CI + Cache warm
green, 25 assets, `latest`, sha256 on CUDA + deb, `ggml_cuda_init` = 1, **smoke
all checks passed (none skipped) + shapes all passed on the DOWNLOADED artifact**.
Ten fixes in ONE release (user: batch — the Windows CUDA build is the long pole).
⚠ The first CI run of the bump went RED: a routing test measured the CI runner's
memory bandwidth as the local node's speed — `PipelineScheduler::
with_local_processor_speed` pins it now; the tag went on the fix commit.
⚠ A Monitor on `gh run list` never matched the Release run and timed out; the
release had been done for two hours — read `gh release view` directly.

### Earlier rounds — one line each; detail in `memory/round_log_*.md` + CHANGELOG

Read the named round log before re-deriving any of these. Gotcha numbers index
into `memory/gotchas.md`.

- **v0.3.154** (09-04): **#451** a fixed element cap in `bytes_to_tensor` (32M = exactly 8192 positions at hidden 4096) was an ~8000-token ceiling on EVERY distributed prompt, silent since March because local inference never serialises an activation — the payload now bounds the allocation; **#452** the planner priced the LOCAL node as memory-unbounded while the loader priced a WHOLE model for a partial segment, so a whole-model holder could serve no PART of a model too big for it; **#453** a tool-carrying reply got NO streaming (120 deltas → 1) — unparked by reading vLLM, which streams pre-marker text AND keeps the prose non-streaming. ⚠ Baseline the smoke/shapes harnesses on the PREVIOUS release first. ⚠ `jq` is not installed here — use `gh --jq`. `round_log_0904_tensor_cap.md`.
- **v0.3.153** (09-04): **#449** ALL inference broken on every Mac (worker socket path over `sun_path`'s 104 bytes) and **#450** "cannot self-update" was a bare bool so the CLI downloaded ~1 GB before failing; **#448** the storage budget had THREE accountants and wedged a node "out of budget" for ever (→ `storage_budget` / `held_shard_bytes` / `storage_budget_now`, a refusal that shows its arithmetic); six future-work closures incl. #447 halves (i)+(ii) — the capacity bound charges the prompt's KV, the segment path admits a whole prompt. ⚠ A `gh release download` was reset mid-transfer and failed its checksum: compare SIZE against `gh release view` before doubting a release.
- **v0.3.152** (09-03): the processor route — **#444** a processor-only node holding a whole model gets a pipeline (fast path returned before the priced search; the local candidate was priced by a card it would not use; no boomerang split points under auto-on privacy); **#440** third half (KV budget reconciled with the card at every admission); **#445** a cancel is READ at every long wait and SET by every surface; **#446** the ACK deadline includes the payload. **#447 found live** and still OPEN in part: `delegation_target` is a yes/no gate firing BEFORE the priced search. ⚠ Measure at steady state. ⚠ Test from the API, not the function. `round_log_0903_processor_route.md`.
- **v0.3.149 → .151** (09-02 evening → 09-03): #438 resend ladder (RetainedReplies + `ResendTokens`/`Dropped`, live drop harness 40/40 vs 5/40); the agent-prompt crawl was VRAM SPILL — #440 prefix-cache snapshots never charged (admit whole prompt, evict cached prompts, size the snapshot, guard evicts before refusing: 5 → 23 tok/s), not the O(1) rollback (#439); `ack_srtt_ms` into routing; testers' #441 cancel-during-prefill (per-layer probe), #442 CPU-only delegation (`serves_on_cpu`), #443 the fast path never reached the scheduler (`local_fast_path_for`). ⚠ A candle-only local CUDA build is NOT the release binary. ⚠ Both arms of a memory-pressure A/B must be in the SAME memory state. `round_log_0902_perf_commits.md`.
- **v0.3.148** (09-02): OpenClaw provider plugin BUILT + verified (`integrations/openclaw/`), `context_length` on `/v1/models`, **#437** idle unload trusted a timestamp nothing writes, Rust 1.90 + 26 deps, Dependabot hygiene; measured the agent-prompt reality (14,633 tokens + 8192 reservation; 5 GB KV at 14.6k). `round_log_0902_openclaw.md`.
- **v0.3.147** (09-02): three failover defects from ONE trace — **#434** a decode step of a remote segment 0 budgeted as a PREFILL (kind decides first), **#435** a standby's ERROR taken as the segment's output (`Tensor bytes too short` blamed on the wrong segment), **#436** a DEPARTED peer now fails its pinned forwards at once (`examples/departed_peer_test.sh`, 503 in 10.4 s). `round_log_0902_failover_paths.md`.
- **v0.3.146** (09-01): benchmarking .145 on the LIVE node — hybrid holds on a 2nd arch (7B 13/28 **6.8** vs 3.8 CPU; 8B 12/32 **5.2** vs 4.0); **#432** a worker's decode-thread calibration settled by one-layer card-only segments it served for a peer (2.9 tok/s, ONE thread for life → keyed by processor depth, depth 0 never timed; confirmed in the field on the deployed binary); **#433** a streamed no-coverage request answered `200`+empty `stop`, and a peer's honest "not hosted" failed a 5-candidate request (both field-confirmed fixed). ⚠ The README's own `SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB` never existed. `round_log_0901_diagnostics_privacy.md`, `perf_baseline_0901_hybrid.md`.
- **v0.3.145** (09-01): **a model 20% too big for the card no longer loses it (#431)** — `force_cpu_for` was `gpu_layers == 0`, all or nothing; three reports, the last with 5151 MB free while a 14B tripped a thermal limit. Split between layers (contiguous, card-first), sized with KV folded into the per-layer cost; **measured 5.0 → 7.07 (14/28) → 12.25 (20/28), 2.4x and monotone**; a 3000 MB budget chose 13/28 unprompted and settled at 2613 MB. ⚠ **The slow baseline reproduced the bug by accident — check `cpu_placement_reason` before quoting a GPU number (#422).** ⚠ **Applied by SHADOWING `device`/`cos`/`sin` at each loop head, not by editing ~128 sites — and the missed-path hazard FIRED (Qwen 3.5's own RoPE); an unused-shadow warning is all that surfaced it → `arch_supports_hybrid` is an ALLOWLIST.** The design was far smaller than the deferral note assumed: read the code before costing work from a months-old list.

- **v0.3.144** (09-01): **no Mac had EVER been able to update itself (#430)**, reported by a NON-TECH user whose every visible surface said otherwise. `host_has_avx2()` returns `false` on non-x86 — meaning *no such instruction set*, not *old processor* — routing every Apple node to a `-baseline` asset that is never published; no asset → `Ok(None)` → **which is how this module says "already up to date"**. Nine releases behind, on the swarm's largest shard holder. ⚠ **The existing guard ASSERTED THE BUG AS CORRECT and was green throughout.** ⚠ **The fix cannot arrive by updating — an Apple node needs ONE manual install.** A node that cannot update now says so.
- **v0.3.143** (09-01): **every node was lying about how fast it is (#428/#429).** Processor efficiency 0.15 against a measured ~0.82 of roofline (**5.2x low**), and the card's 0.30 was HIGHER than the processor's when it should be LOWER — at batch 1 a card reaches 0.37 of roofline, a processor 0.82. Now **0.75/0.35**, measured via `prefill_bench`, **validated on a SECOND machine** (Ryzen 0.818, i5 0.895). Propagation confirmed within the hour: Proxmox 0.88→4.41, Belgium 1.20→5.89, Belgium GPU 20.45→23.86. Also a CPU-only node reported its OWN speed as 0; the unmeasurable fallback would have gone 1.70→8.52 (**a nominal must move with the efficiency it feeds**). ⚠ **My HTTP-derived GPU number was wrong in DIRECTION and published before retraction — that path is not a decode measurement.** ⚠ **CI caught what two local machines did not, and my FIRST fix was insufficient: widening a margin is not removing a threshold.** `round_log_0901_diagnostics_privacy.md`.
- **v0.3.142** (09-01): **the "safe to share" button shared everyone's IP address (#426)** — the dashboard's one-click Copy diagnostics, hinted "No keys or invite codes are included", also copied this machine's addresses AND ten peer-cache multiaddrs, i.e. real users' home IPs; a JS comment claimed it was already redacted. ONE pass over the finished report (`?full=1` opts out), keeping kind + port + peer id + `/p2p-circuit`; **tag salted per report because IPv4 is 2^32**; anchor + loopback exempt. Added `swarmllm diagnostics` + the `-- this machine --` section. ⚠ **A query-string `bool` accepts ONLY `true`/`false` — my own documented `?full=1` 400'd.** `round_log_0901_diagnostics_privacy.md`.
- **v0.3.141** (08-31): **a model no single node can hold is SERVED** — `qwen2.5-14b` over a 3-4 segment chain, reproduced independently; the greedy fallback had ZERO references to `max_hostable_layers` (⚠ my first fix was too STRICT and was caught pre-ship — parallax routes unbounded rather than refusing, and the fallback now matches); **#425** shards scored on rarity alone scattered holdings, 4 WAN round trips/token instead of 2 → contiguity term, **Petals-informed: an INVARIANT combined WITH rarity, not traded against it** (arXiv 2209.01188); an array-wrapped tool call ignored (**confirmed fixed in the field 09-01**). `parallax: no valid source vertex` is NOT a bug. `round_log_0831_tokenizer_quadratic.md`.
- **v0.3.139/.140** (08-31): **models the network is asked to host now SPREAD.** #423 — trust was granted only to models in HF's *trending* feed (a DISCOVERY signal used as VERIFICATION), and the gate hid because a node already holding a shard is EXEMPT, so machines finish what they start and never start anything. Now asks the ORIGIN directly, same thresholds. **Confirmed by a tester: stuck at 1/16 reachable for 12 min, then 15/16 and `peers_hosting` 1→4 on updating.** Also: delegation bound 200→1000 ms (a 600 ms GPU peer serves 21-25 tok/s vs our 9-10 CPU); the wishlist could not tell a 0.6B from a 120B and sized MoE 10x small; #424 a retried download truncated away good ranges (unflushed length used as a `set_len` target — a RACE), which cost that tester GB of bandwidth. `round_log_0831_tokenizer_quadratic.md`.
- **v0.3.138** (08-31): two tokenizer faults. **#420** `bpe_encode_word` was naive BPE — fine for a word, and the SentencePiece branch has no pre-tokenizer so it got the WHOLE PROMPT as one "word": 90 KB took **141 s**, daemon CPU **200.25 s → 1.19 s (168x)**, output identical. **#421** every tab and newline went to the model as `<unk>` (chat templates are full of newlines) — found only by checking against HuggingFace `tokenizers`, since the internal tests compare the tokenizer to an older version of ITSELF. **17/17 samples now agree.** ⚠ **The perf bug was the LEAD, not the bug.** `round_log_0831_tokenizer_quadratic.md`.
- **v0.3.137** (08-31): three faults found by RUNNING the released .136. A peer-held model took 244/210/200/182 s where the LAN holder answers in 0.80 s — it was a candidate every time and lost on ADVERTISED speed, because `record_peer_segment_latency` had only TWO production callers and the speculative path was neither (#418; **proven live 16.87 → 1.36 s**). `/api/admin/stats` took 273 ms, **178 of it KERNEL** — `detect_hardware` enumerated every process TWICE for four numbers (#417, 273 → 6.1 ms). The first-run banner announced a file write the code deliberately does NOT do (#419). ⚠ **FOUR wrong theories killed by measurement before they reached a commit.** `round_log_0830_functional_bench.md`.
- **v0.3.136** (08-30): **three faults visible ONLY over the network** — clean on every locally-held model, so all three survived release, CI and smoke. Non-English replies REFUSED as lost (#416 — a multi-byte char is several GENERATED tokens but ONE SEND); streamed replies arrived TWICE (#414 — one of five coordinators never sent a terminal finish event); an over-long prompt blamed the network and docked the peer (#415). ⚠ **THREE wrong stories on #416; the second fix was WRITTEN and killed by a paired test — the fix was correct while its justification was wrong.** `round_log_0830_network_path.md`.
- **v0.3.132-.135** (08-29/30): **the guards were the defect** — five repo-consistency guards tested by PLANTING the violation, **four could not see what they guard** (#413); line scanning cannot see a chain rustfmt WRAPPED. The Models page took 11 s and **9.6 were the KERNEL** (#410 — GGUF headers off an UNBUFFERED `File` at 7 sites; **11.2 s → 0.21 s**). A model you had EVER served could not be deleted (#409); the forward doing the whole PREFILL was budgeted as a DECODE (#407); one NUL byte hid a doc from grep (#408); one dial per ADDRESS not per peer (#405, paired 13→3); a name is not a content identity (#406). ⚠ **Split utime/stime BEFORE theorising. A guard too weak to fire is too weak to be checked. TWO published causal claims were WRONG first.** `round_log_0830_guard_audit.md`, `round_log_0829_*.md`.
- **v0.3.120-.131** (08-25→28): a corrupt shard PROVED to spread — **only the ORIGIN settles it; peer agreement is not evidence in a network that copies from itself** (#382); **.121 quarantined the GOOD copy (#384) — a repair mechanism is a destruction mechanism**; ACK deadline from ping RTT → RFC 6298 (#386); a KV refusal was a RATCHET (#387); `chars/4` used as `index_pos` — **an estimate of a statistic used as a COORDINATE** (#400, cold-only); ANY libp2p node could become a "peer" (#396) and only `allow_block_list` stopped the dialling (#404); device placement — **the fix was to DELETE the second accountant** (#401/#402). ⚠ **A gate refusing NOTHING is the measurement. Local verification had been a strict SUBSET of CI's** → `examples/release_shapes.sh`. `round_log_0825_overnight_watch.md`, `round_log_0827_*.md`.
- **v0.3.101-.119** (08-18→24): CPU prefill +20-40% / decode +25-37%; ~750 MB less per model (quantized `token_embd` gather); 25.7x from a memory budget read off the BOOT SNAPSHOT (#281, third time); peer delegation — **privacy changes the SHAPE (boomerang), not the verdict**; stale DHT record outranked a retraction (#364). ⚠ **#367 min-of-N is for benchmarks, NOT live measurement. #334 `cargo audit` ran in CI NOT Release — .102 shipped vulnerable.** `round_log_0824_correctness.md`, `round_log_0822_perf_night.md`.
- **v0.3.15-.100** (07-23→08-17): the era that produced most of the rules. Credits switched OFF (they WERE enforced); **the whole prompt pipeline was wrong** — Llama-3 tokenised at ~2x, system prompt rendered TWICE (#246-#253); AVX2 COMPILED OUT of releases (3.09x); batching NEVER engaged (2.4x GPU); a failure could not report itself (#300-#305 → `classify_error`); three id derivations, none agreeing → `slugify_model_name` (#310); settings saved, said ok, did nothing (#281 → **`SharedState::cfg()`**); distant peers' replies SCRAMBLED (#282); our API key was sent to strangers — its PREMISE changed (#238). ⚠ **#283 kill by PID. #334 `cargo audit` ran in CI NOT Release — .102 shipped vulnerable. #266 measure the FORWARD, not the isolated call. #267 this box cannot resolve a GPU change below ~25%. #179 before touching connection selection. #163 retraction alone is futile.** `round_log_0817_honesty.md` and siblings.
- **R136-R150 and the 20 build phases**: NAT/reachability, SWARM-SPEC cascade, `swarmpool://` v2, cross-pool routing. `docs/ARCHITECTURE.md` § phase history.

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
