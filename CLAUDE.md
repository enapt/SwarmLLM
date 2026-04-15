# SwarmLLM — Claude Code Instructions

> **Quick start**: Read `memory/code-map.md` for the component dependency map, data flows, and SharedState sub-struct layout before exploring code.

## Project Overview

SwarmLLM is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously participates in a P2P network, runs an HTTP server (OpenAI-compatible API + admin dashboard), and manages local resources (GPU/CPU compute, storage, bandwidth).

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.80+
- **Primary Port**: 8800 (HTTP API on TCP:8800, P2P on TCP:8810 + UDP/QUIC:8800)

## Architecture

The daemon spawns 11 subsystems as Tokio tasks wired together with `mpsc` channels:

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
- **UpdateChecker** — periodic GitHub release polling, SHA256-verified binary download, atomic apply

Shared state lives in `Arc<SharedState>` with `DashMap` for concurrent access. SharedState is organized into 4 logical sub-structs:
- `state.events` (`EventBus`) — `activity_tx`, `activity_history`, `dashboard_tx`, `update_state`
- `state.credits` (`CreditPool`) — `credit_balance`, `pool_state`, `pool_registry`, `pool_tx`, `trust_manager`, `escrow_manager`, `anti_gaming`, `private_mode`, `offline_mode`, etc.
- `state.models` (`ModelMgmt`) — `acquisition_progress`, `hf_sources`, `auto_manage_*`, `model_trust`, `locked_shards`, `prune_history`, etc.
- `state.metrics` (`MetricsProviders`) — `node_stats`, `inference_requests_total`, `channel_metrics`, `providers_config`, etc.

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
│   ├── main.rs, lib.rs, config.rs, error.rs, types.rs, update.rs
│   ├── daemon/    (mod, state, manifest, shard_loader, dispatch)
│   ├── network/   (manager, behaviour, discovery, protocol, transport, relay, peer_cache, helpers)
│   ├── model/     (manifest, shard, distribution, registry, acquisition, huggingface, auto_manage/, lora)
│   │   └── auto_manage/  (mod, manager, scoring, download, prune, scan, vram)
│   ├── inference/ (router, pipeline, scheduler, executor, sampling, kv_cache, speculative, split/, layers, model_arch, tokenizer, tensor_util, shard_layout, vision, allreduce, chat_template, local_embedder, model_worker, process_pool, worker_ipc)
│   │   └── split/        (mod, model, loader, executor, kv_cache, entry, gguf_meta, shard_reader, rope, tests)
│   ├── credit/    (ledger, transaction, priority, anti_gaming, trust, escrow)
│   ├── identity/  (keypair, nickname)
│   ├── crypto/    (session, pipeline_seal, gossip_seal, key_rotation, provider_keys)
│   ├── pool/      (types, crypto, manager, forward, scope)
│   ├── api/       (server, openai, anthropic, mcp, sse, admin, admin_hf, admin_models, admin_providers, websocket, middleware, identity, pool, metrics, providers, claude_sub*)
│   ├── storage/   (db)
│   └── health/    (monitor, rebalancer)
├── frontend/      (index.html + 12 HTML templates, css/, js/{core/3,components/14,init.js,i18n.js,providers.js,neural-bg.js,topojson-client.min.js}, i18n/)
├── python/        (swarmllm-client SDK)
├── monitoring/    (Grafana + Prometheus + docker-compose)
├── docs/book/     (mdBook documentation site)
└── tests/         (integration tests)
```

## Key Dependencies

libp2p 0.55 (pin to 0.55.x), axum 0.7, candle-core/candle-transformers (CUDA), ed25519-dalek 2, x25519-dalek 2, chacha20poly1305, blake3, redb, dashmap 6, clap 4, tracing, reqwest, zstd. See `Cargo.toml` for full list.

## Coding Conventions

### Error Handling
- Use `thiserror` for defining error types in `src/error.rs` (SwarmError enum)
- Use `anyhow` only in `main.rs` and integration tests
- Map SwarmError variants to HTTP status codes via `ApiError` wrapper
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
- Component architecture: `App` global namespace, 23 JS files (4 core + 14 components + init.js + 4 standalone utilities)
  - `js/core/` — state.js (namespace + shared state + storage keys), utils.js (format helpers, DOM builders, extractErrorMessage, getApiErrorMessage), data.js (data store + authFetch + dedup), tooltip.js (unified popover replacing native `title=`)
  - `js/components/` — ui.js, chat.js, claude-code.js, dashboard.js, models.js, auto-manage-status.js, settings.js, setup.js, downloads.js, notifications.js, identity.js, network-map.js, compare.js, pool.js
  - `js/init.js` — event binding, initialization, public API export
  - `js/i18n.js`, `js/providers.js`, `js/neural-bg.js`, `js/topojson-client.min.js` — standalone utilities (loaded before App)
- 12 HTML `<template>` elements for repeating UI structures (session items, chat messages, toasts, model cards, etc.)
- All storage keys registered as named constants on `App` (e.g., `App.SESSIONS_KEY`, `App.MODEL_SORT_KEY`)
- Dark/light/system theme toggle, CSS custom properties for theming
- i18n: 1117 keys across 21 languages via `frontend/i18n/{lang}.json`, `I18n.t()` + `data-i18n` attributes (28 newest keys added 2026-04-15: only en.json currently translated — other 20 locales await translation)
- Total frontend size target: < 200KB
- Communication: WebSocket for real-time, REST for initial load, SSE for chat streaming
- WebSocket message types (only 5): `activity_event` (unified event bus — all subsystem events, toasts, prune history), `stats_update` (2s interval — stats, shard registry, acquisitions), `peer_list` (full peer snapshot on change), `models_changed` (shard download/load/prune signals dashboard refresh), `update_available` (new version detected)
- Broadcast channels (only 2): `activity_tx` (ActivityEvent — 256 capacity) for all events + `dashboard_tx` (DashboardSignal enum — 32 capacity) for PeersChanged/ModelsChanged/UpdateAvailable signals
- Frontend single entry point: all events flow through `_handleActivityEvent()` in notifications.js — handles routing (activity vs network panel), toast display (via `toast_level` field), prune history, per-model ticker, pool refresh
- Activity events are i18n-ready: frontend formats via `I18n.t('activity.<kind>', params)` with fallback to backend English message

## Testing

- 646 tests total, all passing, clippy clean
- Unit tests: in-module `#[cfg(test)]` blocks
- Integration tests: `tests/integration/` — multi-node simulations with `--test-threads=1`
- Test model: `tests/fixtures/tiny_model/` — 2-layer, 128 hidden dim, ~1MB, 2 shards
- CI pipeline: `cargo fmt` → `cargo clippy --all-targets -- -D warnings` → `cargo test` → `cargo build --release`

## Key Design Decisions

- Config priority: CLI flags > env vars (SWARMLLM_ prefix) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
- Data dir: `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- Port layout: HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- Credit transactions require dual Ed25519 signatures (serving node + requesting node)
- Priority tiers: Bronze (negative balance) < Silver (positive) < Gold (70th percentile) < Platinum (90th)
- KV-cache sessions expire after 10 minutes of inactivity (configurable)
- Shard verification: BLAKE3 content hash checked on every load
- Pipeline failover: hot-standby nodes pre-identified per segment
- **Private mode**: restricts YOUR outbound inference to pool/LAN nodes only. Nodes still serve the swarm. Single `allowed_node_set()` in `src/pool/scope.rs` gates everything. Runtime-toggleable via `AtomicBool`. Shard pinning lets pool owners assign models to devices.
- **No full model download required**: A node NEVER needs the full GGUF or all shards to participate in inference. Shards are downloaded individually via byte-range requests. Downloading all shards (or a full model) is opt-in only — for users who want offline inference or to seed more shards to the network. Never add code that implicitly downloads a full model or reconstructs a GGUF from shards. All inference loads from shard files + gguf_header.bin.

## Automated Workflow

### Slash Commands

| Command | Model | Runs As | Purpose |
|---|---|---|---|
| `/build-phase <N>` | opus | inline | Execute all tasks for build phase N — spawns architect (sonnet) + explorer (haiku) subagents |
| `/next-task` | opus | inline | Auto-detect next uncompleted task, implement it with subagent assistance |
| `/check` | haiku | forked | Run fmt, clippy, test, build pipeline — report-only, cheap |
| `/test-module <mod>` | haiku | forked | Run tests for a specific module — report-only, cheap |
| `/review [file]` | sonnet | forked | Review code against spec — spawns code-reviewer (haiku) for security analysis |
| `/simplify` | (built-in) | inline | Review changed code for reuse, quality, and efficiency — built into Claude Code |
| `/loop <interval> <cmd>` | — | inline | Run a prompt or slash command on a recurring interval (e.g., `/loop 5m /check`) |
| `/plan <desc>` | — | inline | Enter plan mode and start immediately with description (v2.1.72+) |
| `/branch` | — | inline | Fork conversation into a new branch (renamed from `/fork` in v2.1.77, `/fork` still works) |
| `/copy [N]` | — | inline | Copy latest (or Nth-latest) assistant response to clipboard (v2.1.77+) |
| `/powerup` | — | inline | Interactive lessons teaching Claude Code features with animated demos (v2.1.90+) |
| `/release-notes` | — | inline | Interactive version picker showing changelog (v2.1.92+) |
| `/schedule` | — | inline | Create/manage scheduled remote agents (cron triggers) |
| `/team-onboarding` | — | inline | Generate teammate ramp-up guide (v2.1.101+) |
| `/ultraplan` | — | inline | Auto-creates cloud environment for deep planning (v2.1.101+) |

> Claude Code v2.1.104. Opus 4.6 with 1M context. Agent teams enabled. Default effort: high for API-key/enterprise (changed v2.1.94 from medium); dial to medium/low for latency-sensitive or simple tasks.

### Agent Model Strategy

Use the cheapest model that can handle each task to minimize cost and latency.
Current model family: Claude 4.5/4.6 — Opus 4.6 (`claude-opus-4-6[1m]`), Sonnet 4.6 (`claude-sonnet-4-6`), Haiku 4.5 (`claude-haiku-4-5-20251001`). All support 1M context window. Default model overrides via `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` env vars (v2.1.84+).

| Task Type | Model | Rationale |
|---|---|---|
| **Code implementation** | opus | Complex reasoning, spec compliance, architecture decisions |
| **Architecture/design** | sonnet | Module design, dependency graphs, code review |
| **Exploration/search** | haiku | File scanning, codebase inventory, grep/glob |
| **Command execution** | haiku | Running cargo check/test/clippy, reporting results |
| **Security review** | sonnet | OWASP analysis, vulnerability detection |

When spawning subagents from skills or the main context:
- `Task(Explore)` with `model: haiku` — codebase searches
- `Task(feature-dev:code-architect)` with `model: sonnet` — module design
- `Task(feature-dev:code-reviewer)` with `model: haiku` — bug/security scans
- `Task(Plan)` with `model: sonnet` — implementation planning
- `Task(Bash)` with `model: haiku` — running and reporting command output
- Never delegate code writing to subagents — opus writes all production code

### Development Loop

The standard automated development cycle:

1. `/next-task` — finds and implements the next item in the build sequence (opus, uses haiku/sonnet subagents for research)
2. `/check` — validates compilation, linting, tests (haiku, forked — cheap)
3. `/review` — verifies implementation matches the spec (sonnet, forked)
4. Commit when a logical unit is complete

For larger sessions: `/build-phase 1` will implement an entire phase end-to-end with parallel subagent orchestration.

### Hooks (configured in `.claude/settings.json`)

- **PreToolUse(Edit)**: Blocks edits to spec documents (supports `if` conditional field, v2.1.85+)
- **PostToolUse(Edit|Write)**: `cargo check` on .rs files — errors fed back immediately
- **PreCompact**: Blocks compaction if uncommitted changes or build failures
- **Stop**: Session summary + integrity checks logged to `.claude/logs/`
- **TaskCreated** (available, v2.1.84+): Fires when tasks created via TaskCreate
- **PermissionDenied** (available, v2.1.89+): Fires after auto mode denials, can retry

### Prompting Opus 4.6 (per Anthropic best practices)

- **Adaptive thinking is the default** on 4.6 — use `thinking: {type: "adaptive"}` with `output_config.effort` (low/medium/high/max). `budget_tokens` is deprecated. Omit `thinking` entirely when you don't need it.
- **Prefilled assistant responses on the final turn are deprecated** in 4.6 — use structured outputs, clear instructions, or tool calls instead.
- **Dial back "anti-laziness" prompting.** Opus 4.6 overtriggers on aggressive "CRITICAL: you MUST..." language — use normal phrasing. Tools that undertriggered on older models now trigger appropriately.
- **Opus 4.6 tends to overengineer and overuse subagents** — explicitly scope work ("only change what's asked"), and guide subagent use ("delegate only for parallel or isolated work; use direct grep/read for simple exploration"). This aligns with the existing `simplify` + completeness rules.
- **Structure long-context prompts** with longform data near the top, instructions/queries at the bottom, wrap docs in `<document>` XML tags. Queries-at-end yield up to 30% better quality on multi-doc tasks.
- **Tell Claude what to do, not what not to do.** Match prompt style to desired output style.
- For multi-context-window agent runs, inform Claude its context will be compacted so it doesn't artificially wrap up early (our PreCompact hook already enforces state-saving — prompt accordingly).

### Teams & Permissions

Agent teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, `teammateMode: "tmux"`). Permission mode: `acceptEdits`. Pre-approved: cargo, git, file ops, all Task agent types. Denied: `cargo publish`.

## Reference Documents

- `docs/ARCHITECTURE.md` — **Primary reference** — current architecture, subsystems, protocols, security model
- `docs/book/` — mdBook documentation site (getting started, API reference, architecture, troubleshooting)
- `docs/DIAGNOSTICS.md` — DIAG: log instrumentation guide for debugging
- `SwarmLLM_Technical_Specification.docx` — High-level technical specification with architecture rationale

## Status

All 20 build phases complete. All subsystems wired — no stubs. 646 tests passing. Deferred items in `docs/ARCHITECTURE.md` § "Deferred Items".

## Common Commands

```bash
cargo build --no-default-features --features dev,claude-subscription  # Dev build (live frontend + Claude Code)
cargo fmt && cargo clippy --all-targets -- -D warnings  # Lint (MUST pass before push)
cargo test                           # All tests
cargo run -- run -p 8800 -v          # Start daemon
```

**Note:** Always include `claude-subscription` feature when testing Claude Code integration. Bare `--features dev` omits the Claude subscription provider.
