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
│   ├── cli/       (mod, run, status, chat, bench, peers, pool, split_test, update, get_model, remove_model — R150 `swarmllm get-model` reference-model opt-in)
│   ├── config/    (mod, providers, credit, network, ops, node, inference)
│   ├── daemon/    (mod, manifest, shard_loader, gpu_support (CUDA compute-capability floor + pre-Ampere CPU fallback), dispatch/, startup, background, helpers, supervisor)
│   │   └── state/        (mod, activity, capacity, capacity_plan, credits, events, hf, metrics, models, peer_speed, perf_history, relay, removed_shards, repair, tp_allreduce)
│   ├── network/   (manager/{mod,events,requests,tensors,identify,commands,connections,dht,shard_transfer}, behaviour, discovery, protocol, transport, relay, peer_cache, helpers, pipeline_stream)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface/, auto_manage/, lora)
│   │   ├── auto_manage/  (mod, manager, scoring, download, prune, scan, vram, parallax, wishlist)
│   │   └── huggingface/  (mod, download, private_types, probe, search, shards, watcher, tests)
│   ├── inference/ (executor, sampling, kv_cache, speculative, swift, dsd_controller, quant, tokenizer, tensor_util, shard_layout, model_arch, vision, allreduce, attn_kernel, attn_softmax (fused scale+softcap+mask+softmax CPU kernel), decode_attn (single-position CPU attention straight over the KV cache — +24% decode), fast_math (AVX2 expf + fused SiLU×up), cpu_pools (per-phase rayon pools: prefill wide, decode narrow), local_embedder, mem_bandwidth (measured memory bandwidth — what a CPU node advertises as its speed, replacing a hardcoded 50 GB/s assumption), model_worker, token_embedding (quantized token_embd, rows dequantized on lookup — CPU), process_pool, slot_table, worker_ipc, ngram_lookup (R136 L1), hedging (R136 L2), prefetch (R136 L3), trace (per-request route + timing record), prof (SWARMLLM_PROFILE=1 per-stage forward-pass profiler))
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
- i18n: 1318 translation keys (1320 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key; parity + these counts are asserted by `tests/repo_consistency.rs` (update the count in BOTH CLAUDE.md and `docs/ARCHITECTURE.md` when adding keys). Every locale carries idiomatic native strings, not English fallback — a new key MUST be translated across all 21 locales (see `.claude/rules/i18n.md`). Per-batch history in `memory/`.
- Frontend payload: **~1065 KB** (html 130 + css 247 + js 687), plus **one** locale at a time (~83 KB en; Thai is the largest at 150 KB) and **88 KB of bundled fonts** (`frontend/fonts/`, IBM Plex Latin subsets — NOT counted by the payload test, which sums `js|css|html` only) — the other 20 locales are never fetched. Measured byte-accurate and capped by `frontend_payload_stays_within_budget` in `tests/repo_consistency.rs`; the long-standing "< 200KB target" in this file was 5.6x out and nothing checked it. The cap is a regression budget, not a goal: it fails on a step change, not on ordinary growth.
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- **Counts** (re-measured 2026-08-24, after the capacity round): **2056 lib** + 11 ignored with `--features dev,claude-subscription` — the claude-subscription provider carries its own tests, so **always say which feature set a count came from**. 79 integration (31 api_test + 34 phase10_11 + 14 yamux_substream) + 1 ignored e2e, 33 repo-consistency, 1 `api_key_side_effects`, 30 `swarmllm-types` (**not** covered by a bare `cargo test`; CI runs it explicitly), 9 in the vendored request-response patch (`--manifest-path vendor/libp2p-request-response/Cargo.toml --lib`). Clippy clean.
- **Benches and harnesses — see `docs/DIAGNOSTICS.md` § Benchmarks for the full list and the traps.** The ones reached for most: `examples/prefill_bench.rs` (drives `SplitModel::forward` directly, no daemon — `SWARM_BENCH_MODEL`, `SWARM_BENCH_PROMPT`, `SWARM_BENCH_DECODE`, `SWARM_BENCH_REPS`, `SWARM_BENCH_DEVICE=cuda`, and `SWARM_BENCH_SPEC_WIDTHS=1,2,4,8` which prices a K-token forward against a 1-token one at the same history depth — the number that decides whether speculation pays; pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown), `examples/qmatmul_bench.rs` (asserts the tiled kernel is bit-identical to upstream), `examples/smoke_test.sh [binary] [port]` (8 checks on an isolated node — run it on the DOWNLOADED release artifact), `examples/soak_test.sh` (`HOURS=` must be a WHOLE number; data dir is per-`PORT`, so two soaks no longer kill each other).
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

All 20 build phases complete. All subsystems wired — no stubs. **2056 lib (dev,claude-subscription) — re-measured 2026-08-24 at v0.3.119-alpha** + 79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`) + 33 repo-consistency + 1 api_key_side_effects + 30 swarmllm-types tests passing; 11 lib + 1 e2e ignored (env-var or manual). Clippy clean on default, `--no-default-features --features dev,claude-subscription` (that combination is the documented one — plain `--features dev` leaves `embedded` on too and fails on dead code), a `--features llama` check, and `flash-attn --lib`. `cargo audit` clean against the six advisories documented in `SECURITY.md`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.120/.121/.122-alpha (2026-08-25): a corrupt shard PROVED to spread; then the fix turned on the good copy

**Read #384 before touching shard hashes.** .121 added a re-check of a HELD shard
when its expected hash changes — and within the hour, on the live node, it
quarantined our GOOD shard 7 and began an unbounded ~500 MB refetch loop.
The log gave the direction: `expected ab5bc674… (a peer's self-certified hash for
its CORRUPT copy) got 597dcfe8… (our bytes, verified against HuggingFace)`. A peer
gossiped the hash it had computed from its own bad bytes; `register_manifest` is
last-writer-wins and a real hash replaces a real hash (deliberately, for
re-publishes), so the wrong one displaced the right one and the new re-check
faithfully destroyed the good copy.
**The bug was not the re-check — it was that a hash carried no PROVENANCE**, so
"verified against the origin" and "asserted by a stranger" were the same kind of
fact. #382 had already established that only the origin settles a hash; the lesson
was written down and not encoded. .122 adds `ModelRegistry::origin_verified`:
origin-derived hashes outrank gossip, are applied BEFORE change-detection (so a
disproved claim provokes no re-check), and are persisted. Deliberately local and
never gossiped — provenance that travels the network is just another assertion.
**A repair mechanism is a destruction mechanism pointed at whatever it believes is
wrong; ask what happens when the REFERENCE is wrong.** And: ship it, then go and
look.

**.121 completes .120.** .120 stopped a corrupt shard being accepted and passed
on, and made one that IS found bad get replaced rather than merely quarantined.
But a node ALREADY holding a bad shard still could not find out: the only
re-check of an already-held shard is the one-shot startup sweep, which runs ~2 s
after boot against whatever the DB held, and the rescan skips shards already
registered. The corrected hash arrived by gossip AFTER the only check that would
have used it. **.121 re-checks a held shard whenever its expected hash changes**
(`shards_pending_verification`), so a node can be told by the swarm that what it
is serving is wrong. **The lesson is about the answer, not the code**: every
individual piece of "detect and repair" was present and the mechanism still could
not fire, because nothing re-asked the question after the reference changed.
Trace the trigger; do not reason forward from the parts.

Integrity round. Found on the live node by RUNNING it, and settled against an
independent reference rather than reasoning. Detail: gotchas #381/#382,
`docs/FUTURE_WORK.md`.

- **"No hash to check against" silently meant "verified", in three places.** A
  manifest is built from what its author holds on disk, so a partial holder
  publishes real hashes for its own shards and all-zero placeholders for the
  rest — and `register_manifest` was a blind `insert`, so a placeholder ERASED a
  hash we already had. The P2P accept path verifies only when a hash exists, so
  a lost hash meant the next copy was taken on trust, announced as held, and
  re-served: **a propagation channel, not just a local hole**.
- **Measured, not theorised**: shard 7 of meta-llama-3.1-8b failed verification
  three times with the IDENTICAL wrong digest from **two distinct peers**. That
  symmetry invites the wrong conclusion — I formed it — that our expectation was
  the outlier. **Fetching the shard's exact GGUF byte range from the origin repo
  and hashing it returned OUR expected hash**, so the peers were wrong and the
  corruption had spread. **Agreement among peers is not evidence in a network
  that copies from itself.**
- **The verifier reported work it had not done** — `all shards OK verified=21`,
  twice, over five shards it never hashed, because a zero-hash shard was counted
  as `verified`. `unchecked` is now its own count and the all-clear is withheld.
- **Fixes**: hash knowledge is MONOTONIC (`merge_known_shard_hashes`, unknown →
  known, never back; a real re-publish still wins) and PERSISTED (a hash learned
  by gossip used to die with the process). An uncheckable shard is now fetched
  from the model's ORIGIN rather than trusted — **a node is never reduced to
  trusting a peer**, which is what made the old carve-out look unavoidable. The
  sender is NOT penalised: "cannot tell" is neither fine nor the peer's fault.
  Self-limiting, not a retreat from P2P — the origin download supplies the hash,
  which then spreads.
- **Never throw away data you cannot replace.** The refetch runs OUTSIDE the
  `auto_manage.enabled` gate (same distinction as `try_idle_vram_unload`); that
  switch means "don't decide what to fetch for me", not "abandon a shard I asked
  for".
- **⚠ A shard is NOT always a contiguous GGUF range.** Shard 5 has two runs with
  a 122 MB gap; reconstructing it as one span gave a confident FALSE mismatch on
  a healthy file. Assert the reconstructed byte count equals `size_bytes` first.

### Earlier — v0.3.119-alpha (2026-08-24): 25.7x, from a memory budget, on an idle card

Small release, one large measured effect. Detail:
`memory/round_log_0824_correctness.md`.

- **`compute_vram_budget` read the BOOT SNAPSHOT.** Setting `max_gpu_vram_mb`
  answered `{"status":"ok"}`, wrote to `config.toml`, and left the daemon on its
  startup value. Measured end to end on the 3070: an 8B model needing 6033 MB
  refused against a 4095 MB budget with **7187 MB free** (card 88% empty) →
  **1.00 tok/s on the CPU against 25.7 on the card** (21.88/27.97/25.72).
  Gotcha #281 for the THIRD time in one session, in the setting that decides
  GPU-vs-CPU. Its sibling `ram_budget_now` got this fix in August (#362); this
  one was left behind. Pinned by
  `the_vram_budget_is_read_live_like_the_ram_budget_beside_it`.
- **AND the default was the real bug: the budget was a fraction of TOTAL, not of
  what is free.** Half the card whatever was on it, so the cap could not tell an
  idle card from a busy one. Now **RESERVES** a slice and admits against the rest:
  `usable = total - other_process_vram - clamp(10%/7%/5%, 512 MB, 2048 MB)`. **The reserve is CLAMPED because what it protects (desktop, browser, driver) does not scale with the card** — a flat 15% left the 8B fitting by 99 MB and held 3.7 GB back on a 24 GB card..
  **Strictly more protective where it matters** — with 4 GB in use elsewhere a
  default node admits LESS than the old 4096 cap; idle, it admits 6800 and the
  model runs. **The reserve is expressed against card SIZE, not free VRAM**,
  because `admit_to_gpu` weighs it against `committed + estimated` where
  `committed` is our own resident models — a free-based budget subtracts them
  twice. Unreadable usage = assume idle, never invent a restriction.
- **The download bar overshot** — progress is keyed by model and each "Download
  this part" click replaced the last one's entry, so the total stayed at one
  shard while bytes summed across all in-flight transfers. Observed at **156%**.
- **The distributed speculation gate now prices the RTT it saves**, not only the
  bytes it costs (513 KB is ~410 ms on 10 Mbps, ~41 ms on 100). A more distant
  peer now needs FEWER tokens/round — the boomerang case, pinned at ~1 RTT/token.
  Unit-tested only; that path ran 25 times in 25 days here.

**Two reframes worth reading before optimising the distributed path** (both in
`docs/FUTURE_WORK.md`, both measured):
- **Pipeline parallelism across the internet is a CAPACITY mechanism, not a
  speed one.** For one request it is strictly negative — stages run in series,
  every hop is a round trip, the slowest node sets the pace. An 8B request split
  10 layers local / 22 on a 0.86 tok/s peer runs at the peer's speed.
- **Prefill is 94% of a long request and the pipeline is idle (N-1)/N of it.**
  GPipe microbatching would fix that and the machinery is ~80% built — but the
  measured ceiling here is **1.45x**, not ~3x, because the 10/22 split is a
  PLACEMENT failure, not a routing one. Taking that measurement first is what
  stopped days going into the wrong thing.

**⚠ A local `--no-default-features --features dev,claude-subscription` build has
NO CUDA.** It logs `Split model using CPU (no CUDA available)` while admission
logs `admitting model to GPU` and the loader logs `force_cpu=false` — it looks
exactly like a placement bug. I overwrote the live node's CUDA release asset with
one and chased it for several steps. Check
`strings <binary> | grep -c ggml_cuda_init` before believing any placement result.

### Earlier rounds — one line each; full detail in `memory/round_log_*.md` + CHANGELOG

Read the named round log before re-deriving any of these.

- **v0.3.120/.121** (08-25): a corrupt shard PROVED to spread between peers — a placeholder hash ERASED a known one, so P2P copies were accepted unverified, announced and re-served. Hashes now monotonic + persisted; an uncheckable shard is fetched from the ORIGIN; a corrupt one is REPLACED, not just quarantined. Gotchas #381-#383.
- **v0.3.113-.115** (08-22/23): the .114 decode-width calibration was right on the Ryzen and wrong on the i5 — **min-of-N is for benchmarks, not live measurement** (#367); a stale DHT provider record outranked a holder's own retraction (#364); peer refusals arrived cut mid-word (#365). `round_log_0822_perf_night.md`.
- **v0.3.109-.112** (08-21/22): CPU prefill +20-40% / decode +25-37% (multi-row Q4_K/Q6_K kernels, decode attention kernel, AVX2 exp, mimalloc); direct peer chaining ON; relay-carried inbound no longer counted as "direct" (#356); receipt ACK cut a quiet peer's cost 300 s → ~26 s (#357).

- **v0.3.101-.103** (08-18): models need ~750 MB less (quantized `token_embd` row gather); machine speed MEASURED (`mem_bandwidth`); peer delegation — **privacy changes the SHAPE (boomerang), not the verdict**. **#334 `cargo audit` runs in CI NOT Release — .102 shipped vulnerable.** `round_log_0818_quantized_embedding.md`.
- **v0.3.96-.100** (08-12→17): credits switched OFF (they WERE enforced); **a failure could not report itself** — its type known in one place, a literal written in another (#300-#305 → `classify_error`); log severity follows blame (#315-#317); release build 54 → 16 min. `round_log_0817_honesty.md`.
- **v0.3.97-.99** (08-15/16): **models you own could not be reached** — three id derivations, no two agreeing → `slugify_model_name` (#310); long prompts stalled for MINUTES, O(prompt²) KV snapshotting (#312); **1.41x CPU generation** from dropping `repeat_kv` in GQA decode.
- **v0.3.88-.94** (08-09→12): four rounds of "make the one right answer reachable" — a new node could see the swarm's models and run NONE (#296); **settings saved, said "ok", did nothing** (`state.config` is a boot snapshot → **`SharedState::cfg()`**, #281); serving paid and reported nothing (#279/#280); distant-peer replies arrived SCRAMBLED (#282 → `StreamReassembler`); a "private network" shared PUBLIC topics (#285). **GPU decode is LAUNCH-BOUND. ⚠ #283: `pkill -x swarmllm` killed the live node — kill by PID.**
- **v0.3.78-.87** (08-05→09): **the whole prompt pipeline was wrong** — Llama-3 tokenised at ~2x and the system prompt rendered TWICE, invisible until diffed against `tokenizers`/`jinja2` (#246-#253). Releases had AVX2 kernels COMPILED OUT (**3.09x**); batching NEVER engaged (**2.4x** GPU). **⚠ #266 measure the FORWARD, not the isolated call. ⚠ #267 this box cannot resolve a GPU change below ~25%.**
- **v0.3.15-.77** (07-23→08-05): our API key was sent to strangers — nothing in that code changed, its PREMISE did (#238); SPM tokenizer mis-tokenised **64.9%** (**a hash cannot tell "wrong bytes" from "not all the bytes"**, #203); **read #179 before touching connection selection**; **retraction alone is futile** (#163).
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
