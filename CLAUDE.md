# SwarmLLM — Claude Code Instructions

> **Quick start**: `docs/ARCHITECTURE.md` is the canonical architecture reference —
> subsystems, channels, source tree, protocols, security model. Per-subsystem rules
> load on their own when you open the files they govern (`.claude/rules/arch-*.md`).

## Project Overview

SwarmLLM is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously participates in a P2P network, runs an HTTP server (OpenAI-compatible API + admin dashboard), and manages local resources (GPU/CPU compute, storage, bandwidth).

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.90+ (set by `redb`; enforced by `msrv_claim_matches_the_dependency_tree` in `tests/repo_consistency.rs` — do not edit by hand, run that test)
- **Primary Port**: 8800 (HTTP API on TCP:8800, P2P on TCP:8810 + UDP/QUIC:8800)

## Architecture

12 Tokio tasks wired with `mpsc` channels — NetworkManager, InferenceRouter,
MessageDispatcher, CreditLedger, HealthMonitor, ShardRebalancer,
AcquisitionManager, ApiServer, PoolManager, AutoShardManager, HfWatcher,
UpdateChecker. Shared state is `Arc<SharedState>` (`DashMap` / `RwLock`),
organized into 4 sub-structs — `state.events`, `state.credits`, `state.models`,
`state.metrics` — plus cross-cutting root fields and **two** configs: `config`
(boot snapshot, startup-only decisions) and `live_config`, **read via
`state.cfg()`** for anything the user can change while the node runs.

All 20 build phases complete; no stubs. Deferred items are in
`docs/ARCHITECTURE.md` § "Deferred Items", never as a `// TODO`.

**Per-subsystem rules load automatically when you open that subsystem's files**
(`.claude/rules/arch-*.md`); the index is in `.claude/rules/architecture.md`.
Source tree, subsystem detail and protocols: `docs/ARCHITECTURE.md`.

## Key Dependencies

libp2p 0.56, axum 0.8, candle 0.10 (CUDA, **vendored + patched** — see
`vendor/`), redb 4, ed25519/x25519-dalek 2, chacha20poly1305, blake3, dashmap 6,
tokio, clap 4, tracing, reqwest, zstd. Full list in `Cargo.toml`.

Two that carry contracts rather than just versions:

- **minijinja 2.24 + minijinja-contrib (pycompat)** renders chat templates — the
  engine HF's TGI and SGLang use. Its `trim_blocks` / `lstrip_blocks` /
  `keep_trailing_newline` / `pycompat` settings are part of the contract.
- **`serde_json` and `minijinja` are both built with `preserve_order`.** Drop
  either and every tool schema reaches the model alphabetised.

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
- Vanilla HTML/CSS/JS — no framework, no build step; embedded via `include_dir!`.
  `App` global namespace; 28 JS files (4 `core/` + 19 `components/` + `init.js` + 4
  standalone); one `index.html` with 11 `<template>`s and 3 modal overlays.
- Nav is **four** destinations — Chat · Models · Dashboard · Network (the map AND
  the leaderboard) — plus Compare and My Devices under "More". Rank by how OFTEN
  a destination is wanted, never by how expert you must be to want it.
- Storage keys are named constants on `App` (state.js), never raw literals. Fetch
  model/stats data via `App.data.*`, never a bare `authFetch`.
- **5** WS message types, all handled by `_handleActivityEvent()`; **2** broadcast
  channels. Do not add to either set.
- i18n: **1389 translation keys** (**1391 entries per locale** incl. `_lang` +
  `_dir`) × 21 languages, sorted by key. Parity and counts are asserted — **update
  BOTH CLAUDE.md and `docs/ARCHITECTURE.md`**. A new key MUST be translated into
  all 21; no English fallback (`.claude/rules/i18n.md`).
- Payload **~1172 KB** (html 141 + css 260 + js 771, 2026-09-14) + one locale
  (~90 KB en, Thai 167 KB) + 88 KB fonts (not counted). Capped by
  `frontend_payload_stays_within_budget` — a regression budget, not a goal.

## Testing

**Always say which feature set a count came from.** Current, with
`--features dev,claude-subscription`: **2705 lib** (+12 ignored),
79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`),
113 repo-consistency, 1 `api_key_side_effects`, 36 `swarmllm-types` (**not** run
by a bare `cargo test` — CI runs it explicitly),
and 11 in the vendored request-response patch.
Clippy clean. That last suite is run on its own:
`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib`.

⚠ **A count edited after the test run is an untested change.** Counts live in
`CLAUDE.md` ×2 and `README.md` ×2 and are cross-checked by a guard, so re-run
`cargo test --test repo_consistency` after editing one, before `git add`. This
has put main red twice.

- Unit tests in-module `#[cfg(test)]`; integration in `tests/`, `--test-threads=1`.
  Real-model run: set `SWARMLLM_TEST_MODEL_DIR`, then
  `cargo test --test integration_phase10_11 -- --ignored end_to_end`.
- CI is **14 jobs, all 14 required** by branch protection. `examples/check_ci_gate.sh`
  reports required-vs-produced drift — run it against a **COMPLETED** run only.
- **Benches, harnesses and their traps: `docs/DIAGNOSTICS.md` § Benchmarks.** The
  release gate's three (`smoke_test.sh`, `release_shapes.sh`,
  `family_conformance.sh`) all run on the DOWNLOADED artifact.
- **Measurement discipline**: min-of-N on an IDLE box, for benchmarks only — **not
  for live measurement** (#367). A/B inside ONE binary via an env switch, never
  across two builds. **Verify the mechanism fired**, not just that the outcome
  improved. Pinned models: `docs/REFERENCE_MODELS.md`.

## Key Design Decisions

- Config priority: CLI flags > env vars (SWARMLLM_ prefix) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
- Data dir: `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- Port layout: HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- Credit transactions require dual Ed25519 signatures (serving node + requesting node)
- **Credits are DORMANT (2026-08-17) — they gate nothing.** `MIN_BALANCE_FOR_INFERENCE = 0`
  and `calculate_tier` returns a constant; the accounting still runs but no balance
  affects who is served or how fast. `credits_stay_dormant` fails the build if one
  starts gating again. **Read `docs/CREDITS_DESIGN.md` before touching credits** —
  why it is off, what is actually true today, and the exit criteria.
- KV-cache sessions expire after 10 minutes of inactivity (configurable)
- Shard verification: BLAKE3 content hash checked on every load
- Pipeline failover: hot-standby nodes pre-identified per segment
- **Encryption — two layers, distinct concerns:**
  - **Layer 1 — `network.enable_encryption` (DEFAULT TRUE)**: ChaCha20-Poly1305
    sealing of every inter-node activation, per-session X25519 ECDH. AAD via
    `build_layer_forward_aad` (the single source of truth — every optional wire
    trailer must be bound there). **No plaintext fallback**: a `seal()` failure
    drops the forward. Turn it off only for local-loopback debugging.
  - **Layer 2 — `inference.encrypted_pipeline` ("boomerang", DEFAULT FALSE)**:
    this node keeps BOTH ends, so no peer sees the prompt or the sampled tokens.
    ⚠ **Peers DO see intermediate hidden states in plaintext** — a matmul cannot
    run on ciphertext, and those states are ~81% invertible back to text at the
    final layer. It is a STRUCTURAL guarantee, not a cryptographic one against
    the computing node. Costs ~1 RTT/token. Full reasoning, and why FHE/MPC is
    three orders of magnitude away: `docs/ARCHITECTURE.md` § Pipeline Privacy
    Model and `docs/FUTURE_WORK.md`.
- **Private mode** restricts YOUR outbound inference to pool/LAN nodes only; the node
  still serves the swarm. `pool::scope::allowed_node_set()` gates everything.
- **No full model download, ever implicitly.** A node NEVER needs the whole GGUF or
  every shard to serve. Shards come individually over byte-range requests, and
  inference loads from shard files + `gguf_header.bin`. Downloading everything is
  opt-in (offline use, seeding). **Never add code that implicitly downloads a full
  model or reconstructs a GGUF from shards.**

## Subagent Choices for This Codebase

Override the default that would otherwise pick haiku — this codebase's invariants
need real reasoning, not pattern-matching. `Task(feature-dev:code-reviewer)`,
`Task(feature-dev:code-architect)`, `Task(Plan)` and `Task(root-cause)` → **sonnet**.
**Never delegate production code writing** — the main session writes it.
`Task(root-cause)` (`.claude/agents/root-cause.md`) returns CAUSED / NOT-CAUSED /
UNDETERMINED and never a fix; reach for it BEFORE attributing a failure or
reverting, especially when the suspect is your own recent change.

## Reference Documents

- `docs/ARCHITECTURE.md` — **primary reference**: subsystems, source tree, protocols, security model
- `docs/invariants/` — the evidence behind each rule (7 topics). **Read the topic file before changing code a rule names.**
- `.claude/rules/diagnosis.md` — **read before blaming any change for any symptom, and before implementing anything non-trivial.** Rule 0 is research-first; then baseline before blaming, verify the mechanism fired, check the test fails without the fix.
- `docs/DIAGNOSTICS.md` — `DIAG:` instrumentation, benches and their traps
- `docs/FUTURE_WORK.md` — deferred items, with enough context to pick up cold
- `docs/CREDITS_DESIGN.md` — read before touching credits · `docs/book/` — mdBook site
- `.claude/sweep-log.jsonl` — every `/sweep` finding and its status. **Grep before re-reporting.**
- `SwarmLLM_Technical_Specification.docx` — **gitignored, absent from a clone.** Never link a contributor to it.

## Status

**v0.3.182-alpha released and deployed (2026-09-15).** Nothing functional is
unreleased. Release procedure: **`memory/release_gate.md`** — the ordered steps
and every caution earned at a past gate; do not re-derive it. Per-release
history: `memory/round_history.md`. Gotchas index: `memory/gotchas.md`
(next free index 613). Standing cautions: `memory/open_cautions.md` — **read at
session start.** (`memory/` is the auto-memory dir outside the repo:
`~/.claude/projects/-home-user-SwarmLLM/memory/`.)

`cargo audit` reports only advisories documented and accepted in `SECURITY.md`.

## Pushes are public-facing

The repo is public and a webhook relays every commit to the project Discord,
read by non-technical users deciding whether to run this software. Commit
subjects must stand alone with no context, lead with user-visible impact before
mechanism, and never name a person. Get sign-off before a force-push. Full
guidance: `.claude/rules/workflow.md` § "Pushes are public-facing".

## Common Commands

```bash
cargo build --no-default-features --features dev,claude-subscription  # Dev build (live frontend + Claude Code)
cargo fmt && cargo clippy --all-targets -- -D warnings  # Lint (MUST pass before push)
cargo test                           # All tests
cargo run -- run -p 8800 -v          # Start daemon
```

**Note:** Always include `claude-subscription` feature when testing Claude Code integration. Bare `--features dev` omits the Claude subscription provider.
