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

- 1438 lib tests passing + 9 ignored (env-var-gated real-model + manual smoke), 80 integration tests in `tests/integration/` + 2 ignored end-to-end (`cargo test --test integration_phase10_11 -- --ignored`), 2 repo-consistency, 26 in `swarmllm-types` (`cargo test -p swarmllm-types` — NOT covered by a bare `cargo test` from the root), 6 in the vendored request-response patch (`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib` — the crate is workspace-`exclude`d, and its own integration tests need `libp2p-swarm-test` so use `--lib`), clippy clean. Microbench: `cargo run --release --no-default-features --features dev,claude-subscription --example swarm_spec_bench` (R136 — measures all 4 SWARM-SPEC layer primitives + synthetic cascade hit-rate). Local-cluster bench: `examples/3node_setup.sh` (boots 3 daemons) + `examples/3node_inference_bench.sh` (runs 3 workloads × 3 trials and prints tok/s + swarm_spec metrics). Sharded variant: `examples/3node_sharded_setup.sh` (forced distributed pipeline; writes its own per-node config disabling auto-manage and bootstrap so the split survives). **Its inference step is EXPECTED to fail on a single multi-interface host** — that is the zero-redundancy same-host case documented in `docs/FUTURE_WORK.md` § "Connection churn on multi-interface hosts", not a distributed-inference regression (confirmed on released v0.3.28, 2026-07-26). Validate the forward path on two real machines. Both scripts take `SWARM_BENCH_MODEL`. **Pinned reference models for cross-swarm comparison: `docs/REFERENCE_MODELS.md`** (smoke / standard / stress tiers + `examples/fetch_reference_model.sh` to opt in).
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

All 20 build phases complete. All subsystems wired — no stubs. **1438 lib + 80 integration + 2 repo-consistency + 26 swarmllm-types tests passing**; 9 lib + 2 e2e ignored (env-var or manual). Clippy clean default + features dev,claude-subscription + `--features llama`.

Per-round history lives in `~/.claude/projects/-home-user-SwarmLLM/memory/round_log_*.md` and the CHANGELOG; `docs/ARCHITECTURE.md` is the canonical architecture. This section keeps only the current release line plus one-line prior-round pointers.

### Latest — v0.3.43-alpha (2026-07-28): external security audit + self-updating

**Three security fixes from an external audit of .42, and the update lifecycle.**

**Overlay trust was satisfiable by coincidence (gotcha #199).** `100.64.0.0/10`
is shared RFC 6598 space, and `node_is_on_overlay` accepted any address of ours
in it — but `listen_multiaddrs` is `swarm.listeners()` (every BOUND interface,
since we bind `0.0.0.0`) ∪ external addrs, and `addr_is_remotely_reachable`
deliberately keeps CGNAT. So a host on a cellular carrier / an ISP numbering a
customer LAN / a coincidental VPN interface declared itself on a tailnet having
never joined, and would hand its API key to any browser in that block. **The
correct instinct was one function away**: `publicly_reachable` uses confirmed
external addrs and NEVER a bound listener. Fix: ask `tailscaled` —
`GET /localapi/v0/whois?addr=<ip>` over `/run/tailscale/tailscaled.sock`
(`src/api/tailscale.rs`), which answers for BOTH sides at once. Three-way verdict
`Member`/`NotAMember`/**`Unavailable`** — the third must never read as yes, since
an unreadable socket is what a sandboxed service looks like. `classify()` is now
async. Narrowed address test kept as fallback (Tailscale's own
`fd7a:115c:a1e0::/48` ULA or a `tailscale*` interface — never shared v4 space).
**Untested against a real tailnet** — no Tailscale on this machine.

**P2P shards were announced before verification.** The HF path verified; the
*untrusted* path did not. A corrupt/forged shard was recorded as held, announced
and re-served until the ~5min scan re-hashed it. Now verified before announcing,
with the quarantine+penalise policy CLAUDE.md already documented; new
`TrustEvent::ShardVerificationFail` (-0.2, weighted with signature violation).

**`vec![0u8; len]` committed the declared size before any payload arrived** — a
5-byte header claiming 256 MiB cost that much per stream, free to the sender.
Now grows in 1 MiB steps as bytes land.

**`layer_range` bounds were checked but untested** — extracted to
`layer_range_is_valid` with tests. Both transports funnel through the one
dispatcher; keep it that way.

**Update lifecycle**: apply now drains (`active_pipelines` AND `serving_models`,
per #194) then `exec`s into the new binary, so it works under systemd and a bare
terminal alike. Checking is decoupled from installing and ON by default —
`UpdateMode` is `Option` precisely because a serde default cannot reach a config
the daemon already wrote (#198 again). Managed installs (deb/rpm, hardened
anchor) can never write their own binary under `ProtectSystem=strict`, so they
are told to use their package manager rather than shown a button that fails.
Anchor timer daily→hourly; releases now carry the CHANGELOG section.

**Credits are self-attested and unenforced by design** — researched, NOT fixed,
written up in `docs/FUTURE_WORK.md` with four approaches (challenge-response on
random ranges is the cheapest; paying for *observed service* rather than claimed
storage may sidestep proof-of-storage entirely) plus the free-Sybil-reset problem
any enforcement would rest on.

### v0.3.42-alpha (2026-07-28): pre-21-July nodes could never rejoin

Found while updating the local test node for the .41 release — it came up with
an empty peer list and nothing explaining why. **A default that lives only in
`#[serde(default = "…")]` is never applied to a config the daemon already
wrote, and the daemon writes every field** (gotcha #198). The built-in anchor
landed 2026-07-21 (`aeedb35c`, v0.3.1); before that the default was `vec![]`.
`PUT /api/admin/config` serialises the WHOLE `Config`, and **the setup wizard
calls it on "Start SwarmLLM"** — so every user set up before that date has a
literal `bootstrap_peers = []` on disk, which outranks the anchor forever. No
bootstrap, no DHT route, no peers, no log line. Reads as "the update broke my
networking".

**Empty was NOT always accidental, which is what made the obvious fix wrong**:
`deploy/anchor/config.toml` uses it ("the anchor IS the bootstrap — don't dial
itself") and `examples/3node_sharded_setup.sh` uses it to keep the bench cluster
off the public swarm, where a stray peer silently collapses the split being
measured. IPFS is the precedent for keeping empty meaningful. So: empty → the
built-in anchors, with `network.disable_default_bootstrap` to genuinely mean
none, implied by `node.anchor_mode`, and set in both consumers.

**Measurement trap worth remembering**: the first isolation test "failed" — an
opt-out node found 5 peers — because four other nodes were running on the same
host and PEX handed it the anchor ~10ms after the first loopback connection.
`bootstrap_peers=0` in its own startup log was the tell that the config was
right. Re-run alone before believing an isolation result.

**Found, NOT fixed:** `/etc/swarmllm/default.toml` is never read — the systemd
unit runs `swarmllm run` with no `--config` and the daemon loads
`<data_dir>/config.toml`. The `.deb` ships a config in the conventional place
that the daemon ignores (which is why package installs were not stranded).

### v0.3.41-alpha (2026-07-28): the dashboard works from another device

From a tester who could open his node's dashboard on his phone over Tailscale
but whose setup wizard's "Start SwarmLLM" button did nothing. **Every symptom
was one cause, and the reported one was the least of it**: `/admin` is
auth-exempt at any origin, so the page rendered, but the key handout was gated
on loopback — so every admin call 401'd, including the hardware probe, which is
why the wizard also said "CPU only" on a machine with an RTX 3070.

**`addr.ip().is_loopback()` does not mean what it reads as** (gotcha #195). It
means "the last TCP hop began inside this daemon's netns", and it is wrong in
BOTH directions — both reproduced. A same-host reverse proxy (plain `tailscale
serve`) dials us over loopback and hands the key to a fully remote phone; a
container publish or a Tailscale **subnet router** never satisfies it, *not even
from the host's own localhost*, because subnet routers **SNAT by default**. That
last point killed the obvious fix: the tester has a subnet route, so his
container sees the Proxmox host's private address — **an allowlist of Tailscale's
`100.64.0.0/10` would not have fixed the bug that motivated it.**

`api::dashboard_trust::classify` is now the one decision point. Tailnet trust is
default-on but only when THIS node holds an overlay address (the IPv4 half is
shared CGNAT space ISPs also use); LAN is opt-in via a runtime atomic, because
the user flips it precisely when they cannot reach the node to restart it.
Untrusted origins are not a dead end — the page names **the address the daemon
saw** (invisible behind NAT, and the one you'd need to allow) and takes a pasted
key. **Threat model, stated plainly: on a trusted network, reachability of the
API port is admin access.** The nonce only stops a non-browser local process
that cannot read the served HTML — `/admin` is unauthenticated, so scraping a
nonce and spending it is two requests (verified). A code comment claiming more
than that was corrected.

**The same hardcoded-loopback assumption was independently present in the WS
Origin check**, so every remote dashboard silently lost live updates and fell
back to polling. The property wanted is same-origin: `Origin` vs the request's
own `Host`, never a fixed list.

Two more, both found by loading the page rather than reading the diff: **six
panels 401'd on EVERY dashboard load including loopback** (callers *sampled*
whether the key exchange had started instead of ensuring it; `_restFetch`
discards failures, so they just rendered nothing forever) — and the first fix
for it caused a **1266-request storm**, because `loadApiKey` fetched via
`authFetch` which now called `ensureApiKey`, re-entering while the memo was
still null (gotcha #197: an async fn runs to its first `await` before it
returns, so a memo from the return value cannot break a cycle; the bootstrap
request is unauthenticated by definition and must not use `authFetch`).

Also: **`#[derive(Default)]` ignores `#[serde(default = "...")]`** (gotcha #196),
so the new default-on flag shipped off to exactly the fresh installs it was for,
then wrote that `false` into the config it generated. Whole-tree audit found no
other instance. And the README's `curl … /api/admin/api-key` has returned an
error since the nonce landed in May.

**Not fixed, flagged:** there is no `Host`-header validation anywhere, so DNS
rebinding against loopback remains possible. Predates this work; hardening it
can break legitimate reverse proxies, so it wants its own change.

### v0.3.40-alpha (2026-07-28): a slow machine no longer stalls everyone else

Three changes, all measured on real hardware (RTX 3070 + a CPU-only Proxmox
container) rather than reasoned about. **Two of the three were found by running
the thing; one of those was found by the soak, not by review.**

**`prefill_chunk_tokens` bounded decode interruption in TOKENS, not time**
(gotcha #191). Chunked prefill exists so a long admission cannot stall active
decode slots, and the guarantee held exactly as written — the trap was the unit.
128 prompt tokens is ~130ms on a GPU and 45–59s on a modest CPU, so a
co-scheduled request advanced **one token per tick** for the whole of a long
prefill. `inference/prefill_pacer.rs` now sizes the quantum from measured wall
time (`inference.prefill_target_ms`, default 200ms); `prefill_chunk_tokens` is
demoted to a ceiling. Measured, same prompt and machine:

| | co-scheduled | long prompt |
|---|---|---|
| CPU | 470.5s → **48.9s** | 490.9s → 192.6s |
| GPU | 14.8s → **1.3s** | 89.3s → 63.7s |

**The GPU number is the interesting one — the first version made GPUs WORSE.**
Cost there is dominated by fixed per-call overhead, not per-token work: 128
tokens/tick ≈ 130ms but 8 tokens/tick ≈ 790ms. Dividing a near-constant tick
time by fewer tokens *raises* apparent ms/token, which shrinks the chunk further
— a feedback loop pointing the wrong way, pinned at the floor. The pacer now
checks whether a shrink actually made the tick cheaper and disables itself
permanently if not. **Do not remove that check**; the CPU numbers alone look
like a clean win.

**Progress + ETA for anything pre-first-token** — `WorkerMsg::Progress` →
forwarder → `RequestTrace`, surfaced on the dashboard, `/api/admin/performance`
`active`, SSE comment frames (OpenAI *and* Anthropic), and a DIAG line. Two
gotchas fell out: API ids are `swarm-<hex>` so `Uuid::parse_str` fails on EVERY
request — two call sites each falling back to a random uuid could never agree,
hence `crate::api::request_uuid` as the ONE deterministic derivation; and
`elapsed_ms` was computed at write time, so a reader between updates saw a
frozen number (six consecutive polls at 0.0s).

**A node that ONLY serves peers had its worker killed mid-answer** — found by
the soak. `active_pipelines` is the *coordinator's* map and never contains
peer-served work, so the idle-VRAM guard believed a pure-server node was idle,
and v0.3.39's hard-unload ceiling fired while it was answering. New
`state.serving_models` (`ServingState` + RAII `ServingGuard`) is the real signal
the "regional demand" proxy was approximating. **Anything asking "is this model
in use?" MUST consult it as well as `active_pipelines`.**

**Analysed, NOT fixed — the routing ratchet** (`docs/FUTURE_WORK.md`). Slower
nodes go dark and cannot recover: the load compensator is ~3 500 concurrent
requests too weak to span a GPU/CPU gap, and `observed_latency_ms_per_layer` is
only recorded when we route, with no decay — so an unrouted node is never
re-measured and an unmeasured one is priced at `UNKNOWN_COMPUTE_MS`. Replication
does not fix it. Cheapest remedy is EMA decay toward the prior; ε-greedy
exploration is the principled one. Measure first.

### v0.3.39-alpha (2026-07-28): updating a node left it unable to answer

Seven fixes; full detail in `round_log_overnight_0728.md`. **The theme is that
every one was found by running something, and three of them were only reachable
because the previous fix removed the ceiling hiding them.**

**Biggest user impact — `current_exe()` returns `"...(deleted)"` once the binary
is replaced, and replacing the binary IS updating** (rename / `dpkg` / `mv`; only
`cp` escapes, reusing the inode, which is why manual testing never showed it).
Three consumers: worker spawning (**every inference on the node fails**) and both
halves of the self-updater (would have written `swarmllm (deleted).tmp` and left
the real binary alone — a silent no-op update). Severity is in the combination:
the node kept advertising its shards and stayed in `connected_node_ids`, so the
swarm went on routing work to a node that could do none. Gotcha **#188**.

**Two "separate crashes" were one defect.** `split_models` is keyed
`(model, layer_start, layer_end)`, so a node can hold a whole-model entry AND a
tail entry it serves to peers. `has_complete_split_model` asked "is ANY entry
complete?" to pick the local fast path, then `get_split_model_meta` took the
**first entry matching the model id** for the layer range — two lookups, and
`DashMap` order picks. Non-first → no embedding table → prompt token ids hit
block 21's rms-norm; non-last → no output head → sampler gets hidden states. The
range lookup now requires `is_complete`, so decision and data cannot diverge.
Reproduced deterministically in a unit test, both insertion orders. Gotcha **#187**.

**Timeouts: bound what actually varies** — now a rule in
`.claude/rules/architecture.md`. A blanket 300s `TimeoutLayer` capped the
prompt-scaled 600s first-token budget shipped hours earlier, its comment
asserting two things that were both false (**#189**); `UPDATE_DOWNLOAD_TIMEOUT`
demanded a sustained ~3.1 MB/s for the ~933MB GPU build, so a slower line could
**never** update (**#190**). Fix is an inactivity timeout, not a bigger number.
**The risk in the `TimeoutLayer` fix is layer order** — generation routes must
merge BEFORE the auth layer or they answer without a key; existing auth tests
only covered `/v1/models`, which was not one of the moved routes. Pinned by
`generation_routes_still_require_a_key`.

**Retry needed both halves**: a peer's `ServiceUnavailable` is retryable (gated
on the trace showing a remote segment actually ran, since our own worker failing
reads identically) AND that peer is blacklisted for the request — without the
blacklist the retry re-picked the same broken peer. The v0.3.34 lesson recurring.

**A clock stepping backwards discarded every saved conversation.**
`SystemTime::duration_since` errs when its argument is in the FUTURE, and the
fallback was `Duration::MAX` — treating the freshest possible record as
infinitely old. Save-at-shutdown/restore-at-startup straddles exactly when NTP
corrects a clock, so a **1ms** step silently dropped every multi-turn session.
Found as a test that failed once under load and passed in isolation — worth
chasing, not re-running. Gotcha **#192**.

**Found reading the soak log, NOT fixed** (in `docs/FUTURE_WORK.md`):
`prefill_chunk_tokens` bounds decode interruption in **tokens, not time**. 128
tokens is milliseconds on a GPU and **45–59s** on a modest CPU node, so a
co-scheduled request decodes one token per tick — measured 8 tokens in 5.5min
beside a 3 968-token prefill, which reads as a hang. Newly reachable *because*
#189 removed the 300s ceiling that used to kill such prefills. Gotcha **#191**.

### v0.3.30 – v0.3.38 (07-26→28) — one line each; detail in the round logs

- **v0.3.38**: idle VRAM was never actually reclaimed — two workers resident on
  an 8GB card **2h16 past the last request** with `idle_unload_secs = 300`.
  `try_idle_vram_unload`'s second gate refuses to unload while regional demand
  >= `IDLE_DEMAND_EMA_THRESHOLD = 0.1` and the reported models sat *just* over
  it, so the reprieve applied indefinitely. **The gate exists because
  `record_request` is called ONLY from the outbound router path** — serving a
  peer never updates it, so without the gate a node evicts models it is actively
  serving. Reprieve now expires at 12x the window. **Better fix, not done**:
  track last-served-at and drop the demand proxy (needs a per-model timestamp).
  Gotcha from its own test: `idle_unload_secs as i64` wraps a huge window
  NEGATIVE, inverting the check — use `try_from`.

- **v0.3.37**: three shipped bugs from an external report — `swarmllm chat
  --model X` panicked (clap compares INNER value types of same-named args;
  renaming the id does NOT fix it, gotcha **#183**); a wrong-sized shard was
  registered as HELD because `daemon/startup.rs` checks only `exists()`, so
  nothing re-downloaded it (**#184**); requests outlived dead clients (TCP
  keepalive, ~90s). **Privacy**: on by default where the node holds both ends;
  `encrypted_pipeline_for` is the ONE answer; one-step enable via button +
  `swarmllm privacy <model>`. **"End-to-end encrypted" named two different
  guarantees** — wire vs compute boundary (**#185**). **Ragged batching measured
  on GPU: NOT worth building** — 4x work, 23% throughput.

- **v0.3.36**: the dashboard rate-limited ITSELF on load — `ws-ticket` shared a
  5/min bucket (keyed `(ip, BucketKind)`) with cloud-provider probes, so the
  WebSocket got refused and live updates died; probes fired once per provider
  card. Separate buckets + coalesced probes + skip-before-auth. Gotcha **#182** —
  found by loading the page in a real browser, which no test does. **The auth
  guard's first version was wrong**: it read `_apiKeyFull` synchronously, but the
  key arrives asynchronously via a bootstrap nonce, so it skipped legitimate
  calls on every NORMAL load. `authFetch` awaits `_apiKeyPromise`; any new
  credential gate must await it too, not sample it.

- **v0.3.35**: six fixes, all traced not diffed — retry killed BOTH attempts and
  evicted a healthy worker (`responses` keyed by `request_id`, which is not unique
  across *attempts*, gotcha #180); `FIRST_TOKEN_TIMEOUT` was flat 120s sized for
  *generation* and ignored *prefill*, so long prompts could not succeed on modest
  nodes at all (gotcha #181, now sized by the real tokenizer since a chars/token
  guess under-budgets CJK ~2.5x); two LAN peers could mutually forget each other
  permanently (re-dial fired only for mid-pipeline or never-identified peers);
  shard downloads stream instead of buffering a range in RAM; delete/prune guards
  protected only the FIRST shard of a segment (`shards_spanned_by_segment` is now
  the one answer); and the scheduler priced routes on wrong signals — unmeasured
  candidates cost **zero** so cold-node routing was decided by vertex iteration
  order, and the local node was never measured at all. **`parallax_partial_ranges`
  ships OFF**: a node holding every shard is otherwise the only representable
  route, but the split measured slower (~12.0s vs ~10.2s). Remaining gap and the
  two invalid-A/B traps are in `docs/FUTURE_WORK.md`.

- **v0.3.34**: connection selection is now *fewest un-answered, tie-break newest*
  (`pending_outbound_responses` was a signal the crate already tracked). And
  **retraction alone is futile — the blacklist is REQUIRED**: the DHT re-advertises
  a retracted holder, so a retry re-learns the claim and picks the same dead peer.
- **v0.3.33**: **read gotcha #179 before touching connection selection.** A relay
  carrying an INBOUND connection is a bare `/p2p/<peer>` with **no transport
  component at all**, so `connection_is_relayed` counted it as direct and — being
  newest — it won every send, silently dropping them. Affects any NAT'd user;
  31s→8.6s. Diagnostic worth reusing: `grep "connection established peer_id=<peer>"
  node.log | grep -oE "remote_addr=[^ ]+" | sort | uniq -c`.
- **v0.3.32**: first request after a restart failed outright — holder claims come
  from gossip and a full re-announce is ~40min, so a fresh node knew no holders
  while the DHT query was fire-and-forget. `assemble_awaiting_dht` grants a 1.5s
  grace, gated so ONLY that failure waits.
- **v0.3.31**: stale shard-holder claims self-correct. **Scoping is the part not
  to undo** — a segment spans several shards and `segment.shard_id` is only the
  first, so retraction must cover every shard overlapping the failed *span*
  (v0.3.35 generalised this into `shards_spanned_by_segment`).
- **v0.3.30**: one `RequestTrace` (`inference/trace.rs`) feeds every surface —
  DIAG line, `x-swarm-*` + `Server-Timing` headers, admin endpoints, Prometheus,
  hourly redb rollups. **The finding was that nearly all of it was already
  measured and thrown away.** TTFT is stamped by the token CHANNEL
  (`StreamingTokenTx`) because tokens leave from seven sites. Three decisions not
  to undo: Prometheus carries `(route, outcome)` ONLY; headers flush before the
  body so SSE cannot carry TTFT; **"tok/s per node per shard" is NOT measurable
  in a pipeline** — use each segment's share of inter-token latency. Also fixed:
  weight-tied models were unservable from a node lacking shard 0 (gotcha #178).

### Recent releases (one line each; full detail in CHANGELOG.md + `memory/round_log_*.md`)

**This week was dominated by one recurring defect — a shared invariant
implemented per-path — documented as a rule in `.claude/rules/architecture.md`.
Read that before touching any request/response path.**

- **v0.3.27** (07-26): `provider:model` prefix stripped at the OpenAI proxy
  boundary; `PUT /api/admin/providers` no longer a silent no-op on a wrong
  field name; `Quantization::parse` + `trailing_tag` handle hyphenated
  multi-part tags. All three needed follow-up in .28 — see above.
- **v0.3.29** (07-26): tool calls carried schema-shaped arguments that parsed
  and validated while being wrong (#174); Anthropic streaming emitted nested
  rather than sequential content blocks (#175); MCP ignored the client's protocol
  revision; a taken port printed `Failed to listen on QUIC: ` with nothing after
  the colon (#177). First-run UX in 21 locales. **`3node_sharded_setup.sh`
  inference is EXPECTED to fail on one multi-interface host** (#176) — confirmed
  identical on released .28. Detail: `round_log_v0329_livetest.md`.
- **v0.3.28** (07-26): **read gotcha #169 before touching reply text.** The
  control-token leak that .22/.24/.25/.26 chased through the output *scrubber*
  was a **prompt** bug — `eval_block` missed `{% set loop_messages = messages %}`
  so every Llama-3 model was prompted in ChatML and replied in ChatML. The `grep
  "chat template failed" node.log` WARN had fired on every request for releases.
  The fallback chain was also unreachable on 6 of 7 paths (`build_prompt`
  hardcoded `model_name: None`, #171). **Meta-fix**:
  `inference::finalize_reply_text` owns the ordered scrub→truncate→trim sequence
  for all three text sources; escalation ladder in
  `.claude/rules/architecture.md` (#173). Mistral's template still can't be
  *rendered* and degrades via fallback; `minijinja` proposed in FUTURE_WORK.
- **v0.3.22-v0.3.26** (07-25→26): control-token scrubbing at the THREE text
  sources, strip BEFORE stop-truncation (#167-168); template stop-markers on the
  non-streaming split path; `include_usage` on `split_stream_response`; Responses
  API tool cap; local tool calling (`src/api/tool_parse.rs`, 4 formats, all four
  streaming paths); recent-failures diagnostics ring. Detail:
  `round_log_overnight_0726.md`.
- **v0.3.23** (07-25): **the big networking fix** — we recorded an INBOUND
  connection's ephemeral source port as a dialable address and published it, so
  peers learned dead addresses for us. One bug, four symptoms. Gotcha #165.
  Poisoned entries persist in caches — nodes must RESTART, not just swap
  binaries.
- **v0.3.15-v0.3.21** (07-23→25): the networking release line — app-level
  inference + tensor relay across NAT, additive protocol/feature handshake,
  multi-relay + DHT discovery, auto-updater cap 500MB→2GB, deterministic LAN
  dialer, WSL2 mirrored-mode detection, Docker bridge filtering. Audit found
  `max_established_per_peer = 1` was structurally disabling DCUtR (#163);
  `max_circuits` 16→128. **Hole punching verified live 07-25.** Detail:
  `round_log_networking_audit.md`, `round_log_networking_plan.md`,
  `round_log_v0315_livetest.md`.

### Prior rounds (pre-v0.3.15)

- **R143-R150** (07-20→23): NAT/internet reachability (UPnP default-on, AutoNAT v1→v2, `--anchor` + `deploy/anchor/`), request cancellation, `worker_error_is_fatal`, `gpu_layers` plumbing, per-shard download backoff, GPU coverage (`CUDA_COMPUTE_CAP` 80→75). Gotchas #150-160.
- **R136-R142** (05→07-19): SWARM-SPEC v0.1 acceleration cascade (L0 Q8_0, L1 n-gram, L2 hedge, L3 prefetch); `swarmpool://` invite codes v2; cross-pool gossip/routing; autonomous 8h sweep.
- **Pre-R136**: the 20 build phases (P2P/libp2p, split + distributed inference, credits, pools, OpenAI+Anthropic+MCP API, frontend, VLM, Claude Code integration).

Full detail for any round: `memory/round_log_*.md` + `docs/ARCHITECTURE.md` § phase history.

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
