# SwarmLLM — Claude Code Instructions

> **Start here**: `docs/ARCHITECTURE.md` — subsystems, source tree, protocols,
> security model. Per-subsystem rules (`.claude/rules/arch-*.md`, indexed by
> `architecture.md`) load when you open a file they govern — **on the Read tool
> ONLY**, never through `cat`/`sed`/`grep`. Hooks enforce it
> (`.claude/rules/workflow.md` § "What the hooks enforce").

## Project Overview

A single Rust binary: a peer-to-peer node in a decentralized LLM inference network. Each node joins the P2P swarm, runs an HTTP server (OpenAI-compatible API + dashboard) and manages local compute, storage and bandwidth.

- Rust 2021 on Tokio (multi-threaded); port 8800 (HTTP API on TCP:8800, P2P on TCP:8810 + UDP/QUIC:8800)
- **Minimum Rust Version**: 1.90+ (set by `redb`; enforced by `msrv_claim_matches_the_dependency_tree` — do not hand-edit, run that test)

## Architecture

12 Tokio tasks wired with `mpsc` channels (named in `docs/ARCHITECTURE.md`).
Shared state is `Arc<SharedState>` in 4 sub-structs — `state.events`,
`state.credits`, `state.models`, `state.metrics` — plus root fields and **two**
configs: `config` (boot snapshot, startup-only) and `live_config`, **read via
`state.cfg()`** for anything the user can change while the node runs.

No stubs. Deferred items go in `docs/ARCHITECTURE.md` § "Deferred Items", never a `// TODO`.

## Key Dependencies

libp2p 0.56, axum 0.8, candle 0.10 (CUDA, **vendored + patched**, see `vendor/`),
redb 4, dalek 2, chacha20poly1305, blake3, dashmap 6, tokio, clap 4, tracing.
Three carry contracts, not just versions:

- **minijinja 2.24 + minijinja-contrib (pycompat)** renders chat templates — the
  engine HF's TGI and SGLang use. Its `trim_blocks` / `lstrip_blocks` /
  `keep_trailing_newline` / `pycompat` settings are part of the contract.
- **`serde_json` and `minijinja` are both built with `preserve_order`.** Drop
  either and every tool schema reaches the model alphabetised.
- **`libp2p-gossipsub` is a DIRECT dep only to enable its `metrics` feature**
  (the facade's does not). If it drifts from libp2p's pin, cargo builds two
  copies and the traffic split reads as silently absent. Guarded.

## Coding Conventions

- **Errors**: `thiserror` for `SwarmError` in `src/error.rs`; `anyhow` only in
  `main.rs` and integration tests. **Never choose an error type at a call site**
  — `classify_error` is the single answer; the variant → status contract is in
  `.claude/rules/arch-errors.md`. Network: retry with backoff (3). Inference:
  return immediately, never retry silently. Shard integrity: quarantine,
  re-download, penalize trust.
- **Naming**: `PascalCase` types, `snake_case` fns. Newtypes —
  `NodeId([u8; 32])`, `ModelId(String)`, `ShardId { model_id, index }`; NodeId
  displays as its first 8 bytes hex.
- **Serialization**: `serde_json` for the HTTP API (match OpenAI exactly) and
  redb values; TOML for config; the network codec is JSON control messages plus
  binary with a type-tag byte for tensor payloads.
- **Async**: subsystems talk over `tokio::sync::mpsc`; SharedState uses `DashMap`
  for concurrent reads and `RwLock` for single values; shutdown is a
  `tokio::sync::watch`, awaited beside task exit in `daemon/mod.rs`'s `select!`.
  **A DashMap guard never lives across `.await`** — its writer parks an OS
  thread (#90's shape); `clippy.toml` fails the build on one.
- **Logging**: `tracing`, structured spans carrying request_id / model_id /
  peer_count, target `swarmllm::module::submodule`. `-v` debug, `-vv` +libp2p,
  `-vvv` trace.

### Frontend
- Vanilla HTML/CSS/JS, no framework or build step, embedded via `include_dir!` (rules: `arch-frontend.md`).
- **5** WS message types, **2** broadcast channels. Do not add to either set.
- i18n: **1400 translation keys** (**1402 entries per locale** incl. `_lang` + `_dir`) × 21,
  sorted. Counts asserted — **update BOTH CLAUDE.md and `docs/ARCHITECTURE.md`**.
  A new key — from the frontend OR minted in Rust — MUST be translated into all 21; never rely on
  the runtime English fallback.
- Payload ~1196 KB, capped by `frontend_payload_stays_within_budget` — a
  regression budget, not a goal.

## Building and Testing

**One feature set, through the aliases in `.cargo/config.toml`** — `cargo lint`,
`cargo dev-test`, `cargo dev-build` all mean `--no-default-features --features
dev,claude-subscription` (`cargo dev-run` too). Switching sets recompiles the crate, and a
default-feature test build silently replaces the dev binary with one that embeds
a stale frontend (gotchas #573/#578). No hook compiles after an edit: run
`cargo lint` once when a change is complete. The pre-push hook adds a
default-feature lint only when a push touches code only that set compiles.

```bash
cargo fmt && cargo lint               # MUST pass before push (CI lints default features too)
cargo dev-test                        # lib + integration; counts below come from this
cargo dev-run -- run -p 8800 -v       # start a daemon serving the frontend from disk
```

**Counts, measured at v0.3.224 (dev,claude-subscription)**: **3191 lib** (+16 ignored),
79 integration (31 + 34 + 14 `yamux_substream`), **195 repo-consistency**,
1 `api_key_side_effects`, 58 `swarmllm-types`, and 12 in the vendored
request-response patch — plus 17 in the `swarmllm` BIN target (`cli::*`, counted
nowhere else). They are refreshed at release (`memory/release_gate.md`), not
per change, so a higher count since then is expected. The types crate and the
vendored patch are **not** run by a bare `cargo test`:
`cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib`. Two VLM
targets need model files: `llava_e2e` (`#[ignore]`) and `vlm_mmproj_e2e` (skips without an
mmproj). Never edit a count without the run behind it (`completeness.md`).

- Unit tests in-module `#[cfg(test)]`; integration in `tests/`, `--test-threads=1`. Real-model run:
  `SWARMLLM_TEST_MODEL_DIR=… cargo dev-test --test integration_phase10_11 -- --ignored end_to_end`.
- CI is **14 jobs, all 14 required** by branch protection. `examples/check_ci_gate.sh`
  reports required-vs-produced drift — run it against a **COMPLETED** run only.
- **Benches and their traps: `docs/DIAGNOSTICS.md` § Benchmarks.** The gate's three
  (`smoke_test.sh`, `release_shapes.sh`, `family_conformance.sh`) run on the
  DOWNLOADED artifact. Pinned models: `docs/REFERENCE_MODELS.md`.
- **Measurement**: min-of-N on an IDLE box, never live (#367); A/B inside ONE binary via an env
  switch; **verify the mechanism fired**, and check a measuring mechanism against a known answer.

## Key Design Decisions

- Config priority: CLI flags > seven named `SWARMLLM_*` env vars (no generic `<SECTION>_<KEY>` rule, #722) > config.toml > defaults. Provider API keys also loaded from `.env` file in data dir (standard names: `OPENAI_API_KEY`, etc.)
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
  node**; in the UI it is "Start and finish on this computer", never
  "end-to-end", "encrypted pipeline" or "private". → `docs/ARCHITECTURE.md` §
  Pipeline Privacy Model.
- **Private mode** restricts YOUR outbound inference to pool/LAN nodes only (the node still serves
  the swarm); `pool::scope::allowed_node_set()` gates everything. **`gossip_network_id` is NOT
  isolation** — only pool + `private_mode` + `private_mode_allow_lan = false` isolates (#352).
- **No full model download, ever implicitly.** A node never needs the whole GGUF
  to serve — shards arrive individually and inference loads from them plus
  `gguf_header.bin`. **Never add code that implicitly downloads a full model or
  reconstructs a GGUF from shards.** ⛔ And **no design may need a user to hold
  the whole model, not even a low-bit copy** (user, 2026-09-28) →
  `docs/plans/wan_parallel.md`.
- ⛔ **Every node holds the SAME upload of a model** (`model::canonical`, #151):
  registry, disk manifest, header and parts all describe one upload (#776).
- **A family in `supported_list` is a claim**: check it against a REAL file's
  header (#715).
- **A split is only fast when the machines are CLOSE** (0.35 tok/s
  Thailand↔Italy vs 6.76 at 18 ms); nothing routes on coordinates yet →
  `docs/plans/regional_pipelines.md`. Why split decode is slow →
  `docs/plans/faster_than_local.md`, `docs/plans/split_speculation.md`.
- **Local GPU decode is bound by SUBMISSION COUNT**; ONE CUDA stream per device,
  decode and speculative checks submitted as CUDA graphs → `arch-inference.md`,
  `docs/plans/local_decode_submissions.md`.

## Releases, gates and this machine

- **Releases are SIGNED; CI leaves a DRAFT.** Read `memory/release_gate.md`
  before a release — the recipe and every caution a past gate earned. ⛔ A BUILD
  gate is not a BEHAVIOUR gate (#683): `reply_ab.sh` + `split_rig.sh` run on the
  DOWNLOADED artifact, a rig shares the machine with NOTHING (#708), and a reply
  is judged against llama.cpp (`examples/score_against_reference.py`), never by
  byte-equality.
- ⚠ **Windows code is tested ON Windows before it ships** (MinGW cross-build, run
  natively from WSL — `memory/env_windows_test_node.md`); a reproduction must FAIL
  on the broken build first. std's `Command` hands a Windows child every inheritable
  handle (#769): an outliving process starts via `spawn_without_inherited_handles`
  (`update_restart`); workers die with the daemon (job object, #153).
- ⛔ **This PC had five unclean shutdowns under sustained load (2026-09-26 → 10-01),
  causes undetermined.** Every gate, rig, bench or long run uses the safety kit
  (`~/swarmllm-gate-common/safety.sh`); never 4 simultaneous chats (simultaneous
  requests to a node ARE a stress test); ONE cargo build at a time (#684).
  ⛔ **Do NOT ask for a go-ahead or a restart, and never refuse on uptime**
  (user, 2026-10-02) — run it and log the uptime (#146).

## Subagents, workflows and usage (Max 5x plan)

- Reasoning agents (`feature-dev:code-reviewer`, `feature-dev:code-architect`, `Plan`, `root-cause`) →
  **sonnet, never haiku**; haiku only to run a command and report. **Never
  delegate production code writing.** `root-cause` returns CAUSED / NOT-CAUSED /
  UNDETERMINED, never a fix — use it BEFORE blaming a change, especially yours.
- Every subagent except `Explore`/`Plan` loads this file and the always-on rules
  (~50 KB) before it starts. Search with `Explore`; fork when the side task needs
  this conversation (a fork shares its prompt cache); look up a known file yourself.
- Parallel Opus agents can hit the session limit with no warning (#574) — spawn
  one and see it return first. Agent teams (~7x tokens) are off for this project
  (`.claude/settings.json` sets `0` over the user-level `1`); never give a one-shot
  agent a `name` — named agents run as teammates. Workflows only on explicit opt-in.
- `/clear` between tasks; `/compact` is itself a large request. `/usage` attributes plan
  usage; `/doctor prompt-audit` checks these instruction files for stale references.

## Reference Documents

- `docs/ARCHITECTURE.md` — **primary reference**: subsystems, source tree, protocols, security model
- `docs/invariants/` — the evidence behind each rule (7 topics). **Read the topic file before changing code a rule names.**
- `docs/FUTURE_WORK.md` — open work, ranked (history: `FUTURE_WORK_ARCHIVE.md`). **A fix moves its item to § "Closed" in the SAME commit.** ⚠ Its SCOPE is a hypothesis (#654), line numbers drift (#645).
- `docs/DIAGNOSTICS.md` (`DIAG:`, bench traps) · `docs/CREDITS_DESIGN.md` · `docs/book/` ·
  `.claude/sweep-log.jsonl` (every `/sweep` finding — **grep before re-reporting**)
- `memory/` is `~/.claude/projects/-home-user-SwarmLLM/memory/` (not in the repo).
  `MEMORY.md` indexes it and holds the live status; **read `open_cautions.md` and
  `next_up.md` at session start.**

## Compact instructions

Keep: uncommitted files and why they changed; measured numbers with their command;
decisions and reasons; running background task/agent ids; the next step. Drop file dumps.
