# Completeness Rules

## Never defer or suppress

Fix dead code, stale references, or broken patterns now — don't paper over with `#[allow(dead_code)]`, `// TODO`, or `// FIXME`. Delete genuinely unreachable code (verify with `grep -rn`). Deferred features must be listed in `docs/ARCHITECTURE.md` § "Deferred Items", not commented in source.

## After renaming or refactoring

`grep -rn` for the old name across ALL files — not just `src/`. Check `docs/`, `frontend/`, `tests/`, `python/`. Fix every stale reference. Update `///` doc comments if behavior changed. Check whether `CLAUDE.md` Architecture section still matches.

## Pre-push integrity grep checks

Run when you've touched SharedState, frontend JS, or done a multi-commit refactor:
```
grep -rn "\.events\.events\.\|\.models\.models\.\|\.credits\.credits\.\|\.metrics\.metrics\." src/    # double sub-struct
grep -rn "shared_state\.activity_tx\b" src/ | grep -v state.rs                                       # direct field bypass
grep -rn "console\.\(log\|error\|warn\)" frontend/js/                                                # console debug left behind
for f in frontend/js/**/*.js; do node -c "$f"; done                                                  # JS syntax
```

Run `/cleanup` after committing changes to: SharedState fields, API endpoints, JS file structure, broadcast channels, WebSocket message formats, error type → HTTP status mappings.

## A count edited after the test run is an untested change

Test counts live in FOUR places — `CLAUDE.md` twice, `README.md` twice — and
`the_readme_test_counts_agree_with_each_other_and_with_claude_md` fails the
build when they disagree. The i18n key count lives in `CLAUDE.md` and
`docs/ARCHITECTURE.md`, with its own guard.

The trap is not "run the tests". It is that the count can only be written down
AFTER the run that produced it, so the edit that breaks the guard is the one
edit no run has seen. This has now put main red twice — the second time
(2026-09-07) after a full green suite, a clean `-D warnings` clippy, and two
verified null controls, because the number went in afterwards.

**After editing any count, re-run `cargo test --test repo_consistency` before
`git add`.** It is the cheapest target in the suite. The same applies to any
figure a guard cross-checks between files: the frontend payload budget, the
i18n key totals, the MSRV.

## Error type discipline

**Never choose an error type at a call site.** `crate::error::classify_error`
returns `(StatusCode, client-safe message, error_type)` and is the single answer
for every surface — HTTP, both SSE encoders, the Responses API, MCP. A literal
next to `"type":` is the bug: it produced the same failure reported four
different ways at once (gotchas #300-#305), and
`a_streamed_error_names_the_same_failure_as_its_non_streaming_sibling` in
`tests/repo_consistency.rs` now fails the build on a new one.

A surface may **refine** — Responses names a provider failure `upstream_error`,
MCP maps 503 to `RESOURCE_UNAVAILABLE`, Anthropic translates into its own
vocabulary — when the refinement names something *more precisely* than the
canonical answer, and it must be commented at its definition. Naming the same
meaning with a different word is a divergence, not a refinement, and is a bug.

**A type that crossed a boundary is already lost.** `SwarmError` survives neither
the worker IPC hop nor the network hop; both deliver a `String` that gets
re-wrapped as `Inference` → 500. Call `reclassify_flattened_error` before falling
back. This is the ONLY sanctioned place to derive a class from a message, and it
matches `SwarmError`'s own `#[error(...)]` Display prefixes — which are part of
the type, not prose written for a human (#295's trap).

The variant → status contract:

- `SwarmError::Validation` → 400, API input errors
- `SwarmError::ModelNotAvailable` / `ShardNotFound` → 404
- `SwarmError::Config` → startup / config file only
- `SwarmError::ServiceUnavailable` → 503, *this server* can't serve (missing local binary, subprocess spawn/I/O failure, broken pipe, init timeout). Use for all subprocess lifecycle failures (R118-R119 cleanup).
- `SwarmError::ProviderError {status, body}` → upstream returned an error OR upstream response couldn't be parsed (matches R119 translate.rs fix: parsing malformed upstream chat-completions response uses ProviderError, NOT Internal and NOT Validation, even though the upstream data passes through user-triggered code paths). Preserves upstream HTTP status.
- `SwarmError::Internal` → actual bugs (500). Reach for it only when no external party can be blamed. Serializing our own well-typed struct failing is Internal; subprocess crashing is NOT.
- `SwarmError::PeerUnresponsive` → 503, a peer took the request and went silent (ACK sweep, first-token deadline, or the per-segment result deadline in `pipeline/local.rs::segment_timeout_error`). Penalty-ELIGIBLE — as a `PipelineError` the silent peer inherited that variant's penalty exemption (fixed 2026-08-16 for the first two paths, 2026-08-25 for the third, gotcha #389). **`is_transient_remote_failure` does NOT retry all of them**: it matches on message PROSE (`"never acknowledged"`, `"silent drop"`, `"remote-generate timed out"`), which is the #295 trap, so the segment-deadline wording gets no retry. Adding the type arm needs `blacklist_holder_for_request` applied on a timeout first, or a single-holder request just waits the deadline twice.
- `SwarmError::SegmentFailoverExhausted` → 503, mid-pipeline holder failure with no standby. Deliberately NOT `ModelIncompleteInSwarm` (variant-matched by `assembly_failed_for_lack_of_holders` → pointless DHT wait) and NOT `ServiceUnavailable` (triggers the peer-blacklist retry); penalty-exempt because it names no culprit.

Never use `Config` or `Internal` for request validation. When unsure between Internal vs ProviderError vs ServiceUnavailable, look at the surrounding code in the same function — it usually picks a clear pattern (translate.rs lines 549-559 use ProviderError, so the tool_call missing-field arms should too).

### A policy refusal is 503, and must never be told to retry

A request this node declines *by configuration* — private mode, prompt privacy —
is `ServiceUnavailable`-shaped (503), not `Internal` (500). 500 reports a
deliberate setting as a crash: it tells monitoring this node has a bug and the
user nothing they can act on. `PrivateModeUnavailable` and
`PromptPrivacyUnavailable` are the two worked examples — each is its own variant
with its own `error_type`, not a string stuffed into a general variant.

**Give a permanent failure its own variant rather than filing it under
`PipelineError(String)`.** That variant mixes transient and permanent causes, and
its hint is picked by substring-matching user-facing prose that gets rewritten —
so a new permanent failure silently inherits a *default* hint asserting that a
peer went offline and to try again. Three failures have now been given that
advice for a condition retrying can never fix (gotcha #295).

**A variant carrying two causes that need opposite advice is the same bug.**
`Unauthorized` meant both *no valid credential* and *valid credential, wrong
machine* (the loopback-only update and shutdown endpoints). One hint had to serve
both, so remote admins were told to go and fetch an API key they had already sent
successfully — the advice could not work, and following it looped. Split into
`LocalOnly` → 403 `permission_error` (gotcha #309). Ask of any shared variant:
*would the two situations get the same next step?* If not, they are two variants.

Two follow-ups a new `SwarmError` variant must not skip:

- **`failure_is_penalty_worthy`** (`router/distributed_exec.rs`) defaults to
  `_ => true`, i.e. blame the peer. A variant describing OUR OWN config or a
  local fault must join the local-only list or it docks credits from innocent
  peers.
- **`error_hint`** — if retrying cannot help, say so. Test on the ADVICE, not the
  wording: a test pinned to a phrase passes while the user is still looping.

## Verify before deleting sweep findings

Sweep agents report dead code / orphaned keys with confidence ≥80%, but their grep may miss call sites in adjacent directories (R120 caught Agent 4 missing 6 `enc.*` callers in `init.js`, `core/utils.js`, `chat.js`). Before deleting anything an agent flagged:
```bash
grep -rn "thing_name" frontend/js/ frontend/index.html frontend/css/   # full frontend
grep -rn "thing_name\b" src/ tests/ crates/                            # word-boundary; catches re-exports
```
Cheap to verify; expensive to mis-restore. If callers exist, log as wontfix in `.claude/sweep-log.jsonl` to prevent re-report.

## Re-exports and visibility downgrades

Before downgrading a `pub` symbol to `pub(super)` or private, check if it's re-exported via `pub use` in any `mod.rs`. Downgrading without removing the re-export is inconsistent; downgrading and removing the re-export can break consumers (notably test modules using `use super::*`). Always re-grep for the symbol after the change and run `cargo clippy --all-targets` — R120 hit a test-only breakage on `coalesce_byte_ranges` exactly this way.
