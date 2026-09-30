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
- i18n: **1396 translation keys** (**1398 entries per locale** incl. `_lang` + `_dir`) × 21,
  sorted. Counts asserted — **update BOTH CLAUDE.md and `docs/ARCHITECTURE.md`**.
  A new key MUST be translated into all 21; **no English fallback.**
- Payload ~1196 KB, capped by `frontend_payload_stays_within_budget` — a
  regression budget, not a goal.

## Testing

**Always say which feature set a count came from.** With
`--features dev,claude-subscription`: **3126 lib** (+15 ignored),
79 integration (31 + 34 + 14 `yamux_substream`), **187 repo-consistency**,
1 `api_key_side_effects`, 56 `swarmllm-types`, and 11 in the vendored
request-response patch — plus 17 in the `swarmllm` BIN target (`cli::*`, counted
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

**v0.3.214-alpha is the live release (published 2026-09-30 15:47 UTC), signed and on both nodes** — .213 plus the
#761 fix (a refused CUDA-graph capture left the f16 KV mirror claiming positions it never got, so split checks could
drift to a 2nd-choice word; `4fac99bf`). .213 brought card decode as CUDA graphs (a 7B at llama.cpp parity), card
prompts ~1.5×, a split's per-token overhead cut (~80% of local), split speculation with an in-engine small drafter ON
by default (`pipeline::engine_drafter`; γ = 0 where guessing does not pay). Gate record: `memory/release_gate.md`.
The drafter is a pool GUEST (#747); attempt-scoped state is never keyed by the request id, which retries reuse (#749).
Split checks STREAM by default from v0.3.215 (`pipeline::dsd_stream`, `SWARMLLM_SPEC_STREAM=0` = rounds): +20-57% over
rounds TH↔IT with a far PROCESSOR (FUTURE_WORK #149). ⚠ **A rig sharing ONE card reads it ~15% SLOWER** (#759).
**Why split decode is slow and what beats it → `docs/plans/faster_than_local.md`; the speculation plan and its
measurements → `docs/plans/split_speculation.md`.** ⛔ **No design may need a user to hold the whole model, not even a
low-bit copy** (user, 2026-09-28) → parallelize the REPLY, not the token: `docs/plans/wan_parallel.md`.
Qwen 3.5 is on local branch `qwen35-support` (#117); Gemma 4 scoped, not built (#115).
⚠ **A family in `supported_list` is a claim: check it against a REAL file's header** (#715). **Next** → `memory/next_up.md`.

⛔ **This PC had four unclean shutdowns 2026-09-26/30 (#716, #718, #754, #762), all under sustained load.**
The third was a concurrency probe on the live node — **simultaneous requests to a node ARE a stress
test.** Before any: research → deep dive → conceptualize → research again, then the user's go-ahead
for THAT run, live node stopped (`memory/feedback_research_before_stress_tests.md`; #146's
`card_pace` is the node's own guard). A bench beside the live node also reads about HALF (#119).
⚠ **On this WSL2 host, fresh card allocations slow ~1000× over two days of Windows uptime** (#146 deep dive):
a card figure that drifts across releases may be UPTIME, not code — note each run's uptime. The 3rd and 4th
(0x116 mid-gate) both came at 52-54 h of uptime: **read uptime before any card run; above ~24 h, ask for a restart.**

⚠ **Behaviour gate = `reply_ab.sh` + `split_rig.sh`** (incl. `failover`), not conformance alone —
`family_conformance.sh` pins `gpu_layers = 0` and never splits (#93). ⛔ **v0.3.199 shipped BROKEN**: a BUILD gate
is not a BEHAVIOUR gate (#683) — the gate verifies the downloaded ARTIFACT before signing. Recipe:
`memory/release_gate.md`. ⛔ A rig shares the machine with NOTHING (#708). Judge a split reply against llama.cpp
(`examples/score_against_reference.py`), never by byte-equality — and a gap that RECURS at the same sentence is a
bug, not a near-tie (#761: compare arms at an identical prefix, graphs on and off).

**Local GPU decode is bound by SUBMISSION COUNT** — layer count predicts cost. Our kernels are `kernels/*.cu`
(PTX via `build.rs`). ⚠ **ONE CUDA stream per device** — its OWN stream (`SWARMLLM_CUDA_OWN_STREAM=0` = legacy);
decode steps AND speculative checks go to the card as CUDA graphs two layers at a time (`SWARMLLM_CUDA_GRAPH=0` =
off). ⚠ Anything a capture changes on the HOST must be undone if the capture is refused (#761).
Plan: `docs/plans/local_decode_submissions.md`.

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
