# SwarmLLM — Claude Code Instructions

> **Start here**: `docs/ARCHITECTURE.md` — subsystems, source tree, protocols,
> security model. Per-subsystem rules (`.claude/rules/arch-*.md`, indexed by
> `architecture.md`) load when you open a file they govern — **on the Read tool
> ONLY**. `cat` / `sed` / `grep` through Bash do not trigger them, so a session
> that reads through Bash has none of its subsystem rules in context. Hooks
> enforce this: `.claude/rules/workflow.md` § "What the hooks enforce".

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

No stubs anywhere. Deferred items belong in `docs/ARCHITECTURE.md` § "Deferred
Items", never as a `// TODO`.

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
- `thiserror` for `SwarmError` in `src/error.rs`; `anyhow` only in `main.rs` and
  integration tests.
- **Never choose an error type at a call site** — `classify_error` is the single
  answer. The full variant → status contract, and the two follow-ups a new
  variant must not skip, are in `.claude/rules/completeness.md` (always-on).
- Network errors: retry with exponential backoff (3 attempts). Inference errors:
  return immediately, never retry silently. Shard integrity: quarantine,
  re-download, penalize peer trust. Credits: degrade tier, never block.

### Naming
`PascalCase` types, `snake_case` fns. Newtype wrappers for safety —
`NodeId([u8; 32])`, `ModelId(String)`, `ShardId { model_id, index }`; NodeId
displays as its first 8 bytes hex.

### Serialization
`serde_json` for the HTTP API (match OpenAI exactly) and for redb values; TOML
for config; the network uses a unified codec — JSON control messages, binary
with a type-tag byte for tensor payloads.

### Async Patterns
- All subsystems communicate via `tokio::sync::mpsc` channels
- SharedState fields use `DashMap` for concurrent reads or `RwLock` for single-value state
- Graceful shutdown via `tokio::sync::watch` channel
- Use `tokio::select!` in daemon/mod.rs to wait for shutdown or task exit

### Logging
`tracing`, structured spans carrying request_id / model_id / peer_count, target
`swarmllm::module::submodule`. Verbosity: info, `-v` debug, `-vv` +libp2p,
`-vvv` trace.

### Frontend
- Vanilla HTML/CSS/JS — no framework, no build step; embedded via `include_dir!`.
  `App` global namespace. File inventory: `docs/ARCHITECTURE.md` source tree.
- **5** WS message types, **2** broadcast channels. Do not add to either set.
- Nav ranking, storage-key constants and `App.data.*` are in
  `.claude/rules/arch-frontend.md`, which loads when you open `frontend/`.
- i18n: **1390 translation keys** (**1392 entries per locale** incl. `_lang` +
  `_dir`) × 21 languages, sorted by key. Parity and counts are asserted — **update
  BOTH CLAUDE.md and `docs/ARCHITECTURE.md`**. A new key MUST be translated into
  all 21; no English fallback (`.claude/rules/i18n.md`).
- Payload **~1196 KB** (html 142 + css 265 + js 789, 2026-09-17) + one locale
  (~90 KB en, Thai 167 KB) + 88 KB fonts (not counted). Capped by
  `frontend_payload_stays_within_budget` — a regression budget, not a goal.

## Testing

**Always say which feature set a count came from.** Current, with
`--features dev,claude-subscription`: **2811 lib** (+12 ignored),
79 integration (31 `integration` + 34 `integration_phase10_11` + 14 `yamux_substream`),
141 repo-consistency, 1 `api_key_side_effects`, 40 `swarmllm-types`, and 11 in
the vendored request-response patch. Clippy clean. The last two are **not** run
by a bare `cargo test` — CI runs the types crate explicitly, and the vendored
one needs
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
- **Measurement**: min-of-N on an IDLE box, benchmarks only — **not live** (#367).
  A/B inside ONE binary via an env switch, never across two builds. **Verify the
  mechanism fired.** Pinned models: `docs/REFERENCE_MODELS.md`.

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
- KV-cache sessions expire after 10 min idle; shards are BLAKE3-verified on every
  load; pipeline failover uses per-segment hot standbys.
- **Encryption is two layers**: `network.enable_encryption` (DEFAULT TRUE, seals
  every inter-node activation, no plaintext fallback) and
  `inference.encrypted_pipeline` ("boomerang", DEFAULT FALSE, this node keeps
  both ends). ⚠ **Boomerang is a STRUCTURAL guarantee, not a cryptographic one**
  — peers still see intermediate hidden states in plaintext, ~81% invertible
  back to text. Never describe it as hiding data from the computing node.
  → `docs/ARCHITECTURE.md` § Pipeline Privacy Model.
- **Private mode** restricts YOUR outbound inference to pool/LAN nodes only; the node
  still serves the swarm. `pool::scope::allowed_node_set()` gates everything.
- **No full model download, ever implicitly.** A node NEVER needs the whole GGUF or
  every shard to serve. Shards come individually over byte-range requests, and
  inference loads from shard files + `gguf_header.bin`. Downloading everything is
  opt-in (offline use, seeding). **Never add code that implicitly downloads a full
  model or reconstructs a GGUF from shards.**

## Subagent Choices for This Codebase

Override the default that would pick haiku — these invariants need reasoning, not
pattern-matching: `code-reviewer`, `code-architect`, `Plan`, `root-cause` →
**sonnet**. **Never delegate production code writing.** `Task(root-cause)` returns
CAUSED / NOT-CAUSED / UNDETERMINED and never a fix — reach for it BEFORE blaming
a change, especially your own.

## Reference Documents

- `docs/ARCHITECTURE.md` — **primary reference**: subsystems, source tree, protocols, security model
- `docs/invariants/` — the evidence behind each rule (7 topics). **Read the topic file before changing code a rule names.**
- `.claude/rules/diagnosis.md` — **read before blaming any change for any symptom, and before implementing anything non-trivial.** Rule 0 is research-first; then baseline before blaming, verify the mechanism fired, check the test fails without the fix.
- `docs/FUTURE_WORK.md` — deferred items. ⚠ **An entry's own SCOPE is a hypothesis** (gotcha #654) and its line numbers are often wrong (#645) — trace a producer to its CONSUMER before planning from it.
- `docs/DIAGNOSTICS.md` (`DIAG:` instrumentation, bench traps) · `docs/CREDITS_DESIGN.md` (before touching credits) · `docs/book/` — mdBook site
- `.claude/sweep-log.jsonl` — every `/sweep` finding. **Grep before re-reporting.** `SwarmLLM_Technical_Specification.docx` is **gitignored, absent from a clone.**

## Status

**v0.3.190-alpha released and deployed to both nodes (2026-09-19).** Nothing
functional is unreleased; conformance on the downloaded artifact was identical
line for line to the .189 baseline. It closed `docs/FUTURE_WORK.md` #55, #89,
#86 and #17.
⚠ **#90's CAUSE IS STILL UNKNOWN** — a 45-minute dispatcher stall. The node now
names one and stops peers routing into it; nothing prevents or explains one.
⚠ **#17 (composite standbys) has never run on a live multi-node failover** —
unit tests and a guard only, on the token hot path. Rig recipe: #85.
Release procedure: **`memory/release_gate.md`** — the ordered steps and every
caution earned at a past gate; do not re-derive it. History:
`memory/round_history.md`. Gotchas: `memory/gotchas.md` (next free index lives in
`memory/MEMORY.md`, not here — it drifted when both claimed it). Standing
cautions: `memory/open_cautions.md` — **read at session start.** `memory/` is
`~/.claude/projects/-home-user-SwarmLLM/memory/`, outside the repo.

## Pushes are public-facing

The repo is public and a webhook relays every commit to the project Discord, read
by non-technical users deciding whether to run this software. Subjects must stand
alone, lead with user-visible impact before mechanism, and never name a person.
Sign-off before a force-push. → `.claude/rules/workflow.md`.

## Common Commands

```bash
# Always BOTH features: bare `dev` omits the Claude subscription provider.
cargo build --no-default-features --features dev,claude-subscription
cargo fmt && cargo clippy --all-targets -- -D warnings   # MUST pass before push
cargo test                            # counts above are from this + both features
cargo run -- run -p 8800 -v           # start daemon
```
