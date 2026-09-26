# SwarmLLM — Claude Code Instructions

> **Start here**: `docs/ARCHITECTURE.md` — subsystems, source tree, protocols,
> security model. Per-subsystem rules (`.claude/rules/arch-*.md`, indexed by
> `architecture.md`) load when you open a file they govern — **on the Read tool
> ONLY**. `cat` / `sed` / `grep` through Bash do not trigger them, so a session
> that reads through Bash has none of its subsystem rules in context. Hooks
> enforce this: `.claude/rules/workflow.md` § "What the hooks enforce".

## Project Overview

A single Rust binary: a peer-to-peer node in a decentralized LLM inference network. Each node joins the P2P swarm, runs an HTTP server (OpenAI-compatible API + dashboard) and manages local compute, storage and bandwidth.

- **Language**: Rust (2021 edition)
- **Async Runtime**: Tokio (multi-threaded)
- **Minimum Rust Version**: 1.90+ (set by `redb`; enforced by `msrv_claim_matches_the_dependency_tree` — do not hand-edit, run that test)
- **Primary Port**: 8800 (HTTP API on TCP:8800, P2P on TCP:8810 + UDP/QUIC:8800)

## Architecture

12 Tokio tasks wired with `mpsc` channels (named in `docs/ARCHITECTURE.md`).
Shared state is `Arc<SharedState>` in 4 sub-structs — `state.events`,
`state.credits`, `state.models`, `state.metrics` — plus root fields and **two**
configs: `config` (boot snapshot, startup-only) and `live_config`, **read via
`state.cfg()`** for anything the user can change while the node runs.

No stubs anywhere. Deferred items belong in `docs/ARCHITECTURE.md` § "Deferred
Items", never as a `// TODO`.

## Key Dependencies

libp2p 0.56, axum 0.8, candle 0.10 (CUDA, **vendored + patched**, see `vendor/`),
redb 4, dalek 2, chacha20poly1305, blake3, dashmap 6, tokio, clap 4, tracing.
Full list in `Cargo.toml`. Three carry contracts, not just versions:

- **minijinja 2.24 + minijinja-contrib (pycompat)** renders chat templates — the
  engine HF's TGI and SGLang use. Its `trim_blocks` / `lstrip_blocks` /
  `keep_trailing_newline` / `pycompat` settings are part of the contract.
- **`serde_json` and `minijinja` are both built with `preserve_order`.** Drop
  either and every tool schema reaches the model alphabetised.
- **`libp2p-gossipsub` is a DIRECT dep only to enable its `metrics` feature**
  (the facade's does not). If it drifts from libp2p's pin, cargo builds two
  copies and the traffic split reads as silently absent. Guarded.

## Coding Conventions

### Error Handling
- `thiserror` for `SwarmError` in `src/error.rs`; `anyhow` only in `main.rs` and
  integration tests.
- **Never choose an error type at a call site** — `classify_error` is the single
  answer. The full variant → status contract, and the two follow-ups a new
  variant must not skip, are in `.claude/rules/completeness.md` (always-on).
- Network: retry with backoff (3). Inference: return immediately, never retry
  silently. Shard integrity: quarantine, re-download, penalize trust.

### Naming
`PascalCase` types, `snake_case` fns. Newtype wrappers for safety —
`NodeId([u8; 32])`, `ModelId(String)`, `ShardId { model_id, index }`; NodeId
displays as its first 8 bytes hex.

### Serialization
`serde_json` for the HTTP API (match OpenAI exactly) and for redb values; TOML
for config; the network uses a unified codec — JSON control messages, binary
with a type-tag byte for tensor payloads.

### Async Patterns
Subsystems talk over `tokio::sync::mpsc`; SharedState uses `DashMap` for
concurrent reads and `RwLock` for single values; shutdown is a
`tokio::sync::watch`, awaited beside task exit in `daemon/mod.rs`'s `select!`.
**A DashMap guard never lives across `.await`** — its writer parks an OS thread
until readers leave (#90's shape); `clippy.toml` fails the build on one.

### Logging
`tracing`, structured spans carrying request_id / model_id / peer_count, target
`swarmllm::module::submodule`. Verbosity: info, `-v` debug, `-vv` +libp2p,
`-vvv` trace.

### Frontend
- Vanilla HTML/CSS/JS — no framework, no build step, embedded via `include_dir!`.
  Detail in `.claude/rules/arch-frontend.md`, which loads when you open `frontend/`.
- **5** WS message types, **2** broadcast channels. Do not add to either set.
- i18n: **1395 translation keys** (**1397 entries per locale** incl. `_lang` + `_dir`) × 21,
  sorted. Counts asserted — **update BOTH CLAUDE.md and `docs/ARCHITECTURE.md`**.
  A new key MUST be translated into all 21; **no English fallback.**
- Payload ~1196 KB, capped by `frontend_payload_stays_within_budget` — a
  regression budget, not a goal.

## Testing

**Always say which feature set a count came from.** With
`--features dev,claude-subscription`: **2993 lib** (+14 ignored),
79 integration (31 + 34 + 14 `yamux_substream`), **179 repo-consistency**,
1 `api_key_side_effects`, 56 `swarmllm-types`, and 11 in the vendored
request-response patch — plus 16 in the `swarmllm` BIN target (`cli::*`, counted
nowhere else). Clippy clean. The types crate and the vendored patch are **not**
run by a bare `cargo test`:
`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib`.

⚠ **A count edited after the test run is an untested change.** Counts live in
`CLAUDE.md` ×2 and `README.md` ×1 and are cross-checked by a guard, so re-run
`cargo test --test repo_consistency` after editing one, before `git add`. This
has put main red twice.

- Unit tests in-module `#[cfg(test)]`; integration in `tests/`, `--test-threads=1`.
  Real-model run: set `SWARMLLM_TEST_MODEL_DIR`, then
  `cargo test --test integration_phase10_11 -- --ignored end_to_end`.
- CI is **14 jobs, all 14 required** by branch protection. `examples/check_ci_gate.sh`
  reports required-vs-produced drift — run it against a **COMPLETED** run only.
- **Benches and their traps: `docs/DIAGNOSTICS.md` § Benchmarks.** The gate's three
  (`smoke_test.sh`, `release_shapes.sh`, `family_conformance.sh`) run on the
  DOWNLOADED artifact. Pinned models: `docs/REFERENCE_MODELS.md`.
- **Measurement**: min-of-N on an IDLE box, benchmarks only — **not live** (#367).
  A/B inside ONE binary via an env switch. **Verify the mechanism fired** — and
  when the mechanism IS a measurement, check it against a known answer.

## Key Design Decisions

- Config priority: CLI flags > env vars (SWARMLLM_ prefix) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
- Data dir: `~/.local/share/swarmllm/` (Linux), `~/Library/Application Support/swarmllm/` (macOS), `%APPDATA%\swarmllm\` (Windows)
- Port layout: HTTP API on TCP:port, P2P TCP on port+10 (Noise+Yamux), P2P QUIC on UDP:port
- Credit transactions require dual Ed25519 signatures (serving node + requesting node)
- **Credits are DORMANT — they gate nothing**, and `credits_stay_dormant` fails
  the build if one starts. **Read `docs/CREDITS_DESIGN.md` before touching them.**
- KV-cache sessions expire after 10 min idle; shards are BLAKE3-verified on every
  load; pipeline failover uses per-segment hot standbys.
- **Encryption is two layers**: `network.enable_encryption` (DEFAULT TRUE) and
  `inference.encrypted_pipeline` ("boomerang", DEFAULT FALSE). ⚠ **Boomerang is
  STRUCTURAL, not cryptographic** — peers still see hidden states in plaintext,
  ~81% invertible to text. **Never describe it as hiding data from the computing
  node.** → `docs/ARCHITECTURE.md` § Pipeline Privacy Model.
- **Private mode** restricts YOUR outbound inference to pool/LAN nodes only; the node
  still serves the swarm. `pool::scope::allowed_node_set()` gates everything.
- **No full model download, ever implicitly.** A node never needs the whole GGUF
  to serve — shards arrive individually and inference loads from them plus
  `gguf_header.bin`. **Never add code that implicitly downloads a full model or
  reconstructs a GGUF from shards.**

## Subagent Choices for This Codebase

These invariants need reasoning, not pattern-matching: `code-reviewer`,
`code-architect`, `Plan`, `root-cause` → **sonnet**, never haiku. **Never
delegate production code writing.** `root-cause` returns CAUSED / NOT-CAUSED /
UNDETERMINED and never a fix — use it BEFORE blaming a change, especially yours.

## Reference Documents

- `docs/ARCHITECTURE.md` — **primary reference**: subsystems, source tree, protocols, security model
- `docs/invariants/` — the evidence behind each rule (7 topics). **Read the topic file before changing code a rule names.**
- `.claude/rules/diagnosis.md` — **read before blaming any change for any symptom, and before implementing anything non-trivial.** Rule 0 is research-first; then baseline before blaming, verify the mechanism fired, check the test fails without the fix.
- `docs/FUTURE_WORK.md` — deferred items. ⚠ **An entry's own SCOPE is a hypothesis** (gotcha #654) and its line numbers are often wrong (#645) — trace a producer to its CONSUMER before planning from it.
- `docs/plans/regional_pipelines.md` — **why split inference is slow and the staged fix.** Read before touching routing, placement or the cost model.
- `docs/DIAGNOSTICS.md` (`DIAG:`, bench traps) · `docs/CREDITS_DESIGN.md` · `docs/book/`
- `.claude/sweep-log.jsonl` — every `/sweep` finding. **Grep before re-reporting.**

## Status

**v0.3.207-alpha is the live release (2026-09-26), signed and on both nodes** — contents in
`memory/round_history.md`. **`main` is AHEAD of it, unreleased** — agent workloads on processor-only
nodes (field report 2026-09-26): SSE keep-alive data chunk, a warm prompt priced warm
(`scheduler::cached_prefix`), prefix-cache ceiling derived from bytes, `models_run_on_card`, the
owner's prompts on every core (`process_pool::Requester`), the GQA attention cliff past a 5,461-long
cache; and from the 2026-09-26 benchmark pass, the card's free memory read after a synchronize (#121)
and a reply reserved by its `max_tokens` (#122). Qwen 3.5 is on local branch `qwen35-support` (#117); Gemma 4 scoped, not built (#115).
⚠ **A family in `supported_list` is a claim: check it against a REAL file's header** (#715). **Next** → `memory/next_up.md`.

⛔ **This PC had two unclean shutdowns on 2026-09-26 (#716 bluescreen, #718 hard hang), causes
unknown.** Heavy CPU/GPU benches run only with the user present and the live node stopped — a bench
beside the live node also reads about HALF (#119).

⚠ **Behaviour gate = `reply_ab.sh` + `split_rig.sh`** (incl. `failover`), not
conformance alone — `family_conformance.sh` pins `gpu_layers = 0` and never
splits, so it could not see #93. Recipe: `memory/release_gate.md`. ⛔ A rig
shares the machine with NOTHING — every node's loopback probe finds it (#708).
Judge a split reply against llama.cpp (`examples/score_against_reference.py`),
never by byte-equality.

⛔ **v0.3.199-alpha SHIPPED BROKEN and was WITHDRAWN** — cleared under
`--features candle-cuda` (no flash-attn, no llama backend) while the release is
`--features cuda`. **A BUILD gate is not a BEHAVIOUR gate** (#683); the gate
verifies the downloaded ARTIFACT before signing.

**Local GPU decode is bound by SUBMISSION COUNT** — layer count predicts cost.
Our kernels live in `kernels/*.cu` (PTX via `build.rs`), each bit-identical to
the candle ops it replaces. ⚠ **ONE CUDA stream per device**;
`SWARMLLM_CUDA_OWN_STREAM=1` stays opt-in and OFF. Plan:
`docs/plans/local_decode_submissions.md`.

⚠ **The privacy mode is STRUCTURAL** — in the UI "Start and finish on this
computer", never "end-to-end", "encrypted pipeline" or "private". **"At this
machine" is `RequestOrigin::is_this_machine`**, never `is_loopback()` (#689).

⚠ **A new wire trailer is NOT a no-op for an older peer** — gate it at the
SENDER. **Gossip says what CHANGED, to everyone; what ONE peer lacks, to that
peer** (#673). **A split is only fast when the machines are CLOSE** (0.35 tok/s
Thailand↔Italy vs 6.76 at 18 ms); nothing routes on coordinates yet →
`docs/plans/regional_pipelines.md`.

**Releases are SIGNED; CI leaves a DRAFT** — the signer can take the WRONG tag silently; the
checks are in `memory/release_gate.md`. ⚠ **#90's cause is unknown.** ⚠ **`gossip_network_id` is
NOT isolation** — only pool + `private_mode` + `private_mode_allow_lan = false` isolates (#352).

⛔ **Nothing may block compaction, so commit as you go** (#687).

`memory/` is `~/.claude/projects/-home-user-SwarmLLM/memory/` — `MEMORY.md`
indexes it. **Read `open_cautions.md` and `next_up.md` at session start**, and
`release_gate.md` before a release rather than re-deriving it.

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
