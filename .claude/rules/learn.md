# Continuous Learning Rules

Update knowledge artifacts the same commit as the change — not at session end.

## Where to update what

| Discovery | Update target |
|---|---|
| New architectural pattern or convention | `.claude/rules/architecture.md` (the rule) + `docs/invariants/<topic>.md` (the evidence) |
| New gotcha or pitfall | `memory/gotchas.md` (append at the next free index) |
| Changed file structure (new/renamed/deleted) | `CLAUDE.md` Repository Structure |
| Changed API endpoint | `docs/ARCHITECTURE.md` HTTP API |
| Changed SharedState fields/sub-structs | `CLAUDE.md` Architecture + `architecture.md` |
| Changed broadcast channels or WS message types | `CLAUDE.md` Frontend + `docs/book/src/architecture/daemon.md` |
| Changed frontend JS file structure | `CLAUDE.md` Frontend + `docs/ARCHITECTURE.md` Frontend |
| New debugging technique | `docs/DIAGNOSTICS.md` |
| Test count changed | `CLAUDE.md` Testing (after `cargo test` confirms) |
| New i18n keys | Propagate to all 21 language files |
| Repeated mistake | Create or update a `.claude/rules/` file |

## Code map

`memory/code-map.md` is the dependency map index. Update when adding/removing/renaming a subsystem, API handler, JS component, channel, or WS message type. Read it at session start instead of re-exploring.

## Where a rule lives

`.claude/rules/architecture.md` loads into **every** session, so it holds the rule
statement and nothing else. The evidence — what the rule replaced, what it was
measured at, what a change must keep — goes in `docs/invariants/<topic>.md`,
which is read on demand.

Adding a rule means both halves: a heading and a short statement in the rules
file with a `→ docs/invariants/<topic>.md` pointer, and the full reasoning in
that topic file. A statement with no evidence behind it is an assertion; evidence
with no statement in the loaded file will not be found in time to matter.

The file was 4,573 lines and 308 KB before this split — every session paid for
all of it, and one section had become a 1,657-line landfill for anything shaped
like "there is one place that does X". Keep statements short.

## Memory hygiene

- `MEMORY.md` under 200 lines (loaded limit). Move details to topic-specific files.
- `CLAUDE.md` target: **~200-250 lines** (per Claude Code best practice — it loads every session, so every line costs context). Keep round-log entries terse: the "Latest" section is a few paragraphs on the current release line, and older rounds are compressed to one-line CHANGELOG-style pointers to the full `memory/round_log_*.md`. When it drifts past ~300 lines, prune the round history again (2026-07-24: pruned 1008→259 by collapsing R136-R150 into one-liners).
- Delete entries verifiably outdated (check code first).
- Don't duplicate between `CLAUDE.md` and `rules/` — `CLAUDE.md` summarizes, `rules/` has detail.
