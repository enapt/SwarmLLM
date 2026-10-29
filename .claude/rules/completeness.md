# Completeness Rules

## Never defer or suppress

Fix dead code, stale references, or broken patterns now — don't paper over with `#[allow(dead_code)]`, `// TODO`, or `// FIXME`. Delete genuinely unreachable code (verify with `grep -rn`). Deferred features must be listed in `docs/ARCHITECTURE.md` § "Deferred Items", not commented in source.

## After renaming or refactoring

`grep -rn` for the old name across ALL files — not just `src/`. Check `docs/`, `frontend/`, `tests/`, `python/`. Fix every stale reference. Update `///` doc comments if behavior changed. Check whether `CLAUDE.md` Architecture section still matches.

Run `/cleanup` after committing changes to: SharedState fields, API endpoints, JS file structure, broadcast channels, WebSocket message formats, error type → HTTP status mappings.

## A count edited after the test run is an untested change

Test counts live in `CLAUDE.md` and `README.md`, cross-checked by
`the_readme_test_counts_agree_with_each_other_and_with_claude_md`; the i18n key
count lives in `CLAUDE.md` and `docs/ARCHITECTURE.md` with its own guard. A
count can only be written AFTER the run that produced it, so the edit that
breaks the guard is the one edit no run has seen — main went red twice that way.
`commit-gate.sh` now runs the guard on any commit touching those files.

**Test counts are refreshed at release, not per change** (2026-10-02: 26
count-only commits since August had each been broadcast to Discord). The same
care applies to every figure a guard cross-checks between files: the frontend
payload budget, the i18n totals, the MSRV.

## README speeds are refreshed in every release's bump commit (user, 2026-10-29)

The README's "How fast is it?" section shows what users can DOWNLOAD, so it
changes at release, never per change. In the bump commit, beside the counts:

1. **One computer**: `python3 -I examples/readme_speeds.py
   ~/swarmllm-gate/v<previous release>/out/speed.jsonl` rewrites that table from
   the last release check's own records (the downloaded release; each run's best
   repeat, the range across runs). Never type those figures.
2. **Across computers**: each row names the version it was measured on. Replace
   a row when this release's CHANGELOG entry or `faster_than_local.md` carries a
   newer measurement of that setup; add a row for a setup newly measured. A
   figure only main has waits for the release that ships it. A range spans every
   run of the setup, never one run's best (gotcha #990), and names the machines.
3. Set `reviewed for <new version>` in the section's opening marker.

Guard: `the_readme_speeds_are_reviewed_at_every_release` — the bump changes
`Cargo.toml`'s version, so the commit gate runs it there. A row's wording follows
CLAUDE.md's privacy wording ("Start and finish on this computer" is never private).

## Error types

**Never choose an error type at a call site** — `classify_error` is the single
answer. The variant → status contract and the rules for a new variant are in
`.claude/rules/arch-errors.md`, which loads with any file under `src/`.

## Re-exports and visibility downgrades

Before downgrading a `pub` symbol to `pub(super)` or private, check if it's re-exported via `pub use` in any `mod.rs`. Downgrading without removing the re-export is inconsistent; downgrading and removing the re-export can break consumers (notably test modules using `use super::*`). Always re-grep for the symbol after the change and run `cargo lint` (it covers `--all-targets`) — R120 hit a test-only breakage on `coalesce_byte_ranges` exactly this way.
