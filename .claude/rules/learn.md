# Continuous Learning Rules

Update knowledge artifacts the same commit as the change — not at session end.

## Where to update what

| Discovery | Update target |
|---|---|
| New architectural pattern or convention | `.claude/rules/architecture.md` |
| New gotcha or pitfall | `memory/MEMORY.md` Key Technical Gotchas |
| Changed file structure (new/renamed/deleted) | `CLAUDE.md` Repository Structure |
| Changed API endpoint | `docs/ARCHITECTURE.md` HTTP API |
| Changed SharedState fields/sub-structs | `CLAUDE.md` Architecture + `architecture.md` |
| Changed broadcast channels or WS message types | `CLAUDE.md` Frontend + `docs/book/daemon.md` |
| Changed frontend JS file structure | `CLAUDE.md` Frontend + `docs/ARCHITECTURE.md` Frontend |
| New debugging technique | `docs/DIAGNOSTICS.md` |
| Test count changed | `CLAUDE.md` Testing (after `cargo test` confirms) |
| New i18n keys | Propagate to all 21 language files |
| Repeated mistake | Create or update a `.claude/rules/` file |

## Code map

`memory/code-map.md` is the dependency map index. Update when adding/removing/renaming a subsystem, API handler, JS component, channel, or WS message type. Read it at session start instead of re-exploring.

## Memory hygiene

- `MEMORY.md` under 200 lines (loaded limit). Move details to topic-specific files.
- `CLAUDE.md` under 250 lines.
- Delete entries verifiably outdated (check code first).
- Don't duplicate between `CLAUDE.md` and `rules/` — `CLAUDE.md` summarizes, `rules/` has detail.
