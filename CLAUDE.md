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
- i18n: 1170 translation keys (1172 entries per locale incl. `_lang` + `_dir`) across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes. All files sorted by key for parity audits. R110-R115 translations completed in R116, contribution-mode (R121) keys added across all locales, plain-language refresh (R125 ease-of-use audit) translated across all 21 locales — translator-agent pass — every locale has idiomatic native-language strings, not English fallback. R126 batch: removed dead `activity.worker_*` + `models.meta_tokenizer`, refreshed encryption copy (`enc.*` ×19 keys, end-to-end honest), added `activity.manifest_rejected` + `models.meta_advanced`, renamed `models.metadata_header` to "Technical Details". R127 batch: dropped 4 orphans (`models.hf_score_breakdown`, `models.hf_score_pts`, `models.hf_on_swarm`, `models.likes_count`); translated `dashboard.api_log_link` across 21 locales; country names now resolved via `Intl.DisplayNames` keyed off `I18n.getLang()` (no hand map).
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions, **swarm_capacity** (R110), **wishlist** (R111)), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 1053 lib tests passing + 8 ignored (env-var-gated real-model + manual smoke), 75 integration tests in `tests/integration/`, 1 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline — requires `auto_manage.enabled = false` in per-node config.toml to preserve sharded state).
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

All 20 build phases complete. All subsystems wired — no stubs. **1053 lib tests + 75 integration tests passing**; 8 lib + 1 e2e ignored (env-var or manual). Clippy clean default + `--features llama`.

### Latest: R141 — Auto-manage cold-start UX (non-tech-user fixes)

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
