# Continuous Learning Rules

Update knowledge artifacts the same commit as the change — not at session end.

## Where to update what

| Discovery | Update target |
|---|---|
| New architectural pattern or convention | `.claude/rules/arch-<topic>.md` (the rule; `architecture.md` only if it applies everywhere) + `docs/invariants/<topic>.md` (the evidence) |
| New gotcha or pitfall | `memory/gotchas.md` (append at the next free index) |
| Changed file structure (new/renamed/deleted) | `docs/ARCHITECTURE.md` § Annotated source tree (moved out of `CLAUDE.md` 2026-09-16) |
| Changed API endpoint | `docs/ARCHITECTURE.md` HTTP API |
| Changed SharedState fields/sub-structs | `.claude/rules/arch-state-and-config.md` (per-field rules); `CLAUDE.md` only if a sub-struct is added or removed |
| Changed broadcast channels or WS message types | `CLAUDE.md` Frontend + `docs/book/src/architecture/daemon.md` |
| Changed frontend JS file structure | `CLAUDE.md` Frontend + `docs/ARCHITECTURE.md` Frontend |
| New debugging technique | `docs/DIAGNOSTICS.md` |
| Test count changed | `CLAUDE.md` Testing (after `cargo test` confirms) |
| New i18n keys | Propagate to all 21 language files |
| Repeated mistake | Create or update a `.claude/rules/` file — path-scoped unless it applies everywhere |
| New rules file | Give it `paths:` frontmatter and verify every glob matches (see below) |

## Code map

`memory/code-map.md` is the dependency map index. Update when adding/removing/renaming a subsystem, API handler, JS component, channel, or WS message type. Read it at session start instead of re-exploring.

## Where a rule lives

Three tiers, and the difference is **when it loads**:

| Tier | File | Loads |
|---|---|---|
| Always-on rule | `.claude/rules/architecture.md` | every session — keep it to what applies EVERYWHERE |
| Path-scoped rule | `.claude/rules/arch-<topic>.md` | only when a file matching its `paths:` frontmatter is read |
| Evidence | `docs/invariants/<topic>.md` | on demand, when someone follows the pointer |

**Adding a rule means two halves**: a heading and a SHORT statement in the right
rules file with a `→ docs/invariants/<topic>.md` pointer, and the full reasoning
— what it replaced, what it was measured at, what a change must keep — in that
topic file. A statement with no evidence is an assertion; evidence with no
statement will not be found in time to matter.

**Default to a path-scoped file.** A rule earns a place in the always-on core
only if it applies across subsystems (the SharedState accessors, the event
system, additive protocol evolution, "one invariant, N paths", timeouts). If it
names a module, it belongs in that module's `arch-*.md`.

When you add a rules file, give it `paths:` frontmatter and **check every glob
matches something** — a typo'd glob is silently never loaded, which reads
exactly like a rule nobody follows:

```bash
python3 - <<'EOF'
import glob,re
for f in glob.glob('.claude/rules/arch-*.md'):
    for p in re.findall(r'  - "(.*)"', open(f).read().split('---')[1]):
        print(f"{len(glob.glob(p, recursive=True)):5d}  {p}  {f}")
EOF
```

**Why the tiers exist.** `architecture.md` was 4,573 lines and 308 KB before the
2026-09-09 evidence split, and 1,787 lines before the 2026-09-16 path split —
every session paid for all of it, on every task. It is now 162 lines. Anthropic's
documented target is under 200 lines per always-loaded file, because "longer
files consume more context and reduce adherence". **Keep statements short.**

If you are reasoning about a subsystem without opening its files, its rules file
has not loaded — read it directly.

## Memory hygiene

- `MEMORY.md` is an INDEX: **Claude Code loads only its first 200 lines / 25 KB and silently drops the rest.** Keep it near 100 lines, one line per entry, detail in topic files. It was at 192/200 on 2026-09-16 — one entry from losing content.
- `CLAUDE.md` target: **under 200 lines** — Anthropic's documented figure, because it loads every session and "bloated CLAUDE.md files cause Claude to ignore your actual instructions". Release history belongs in `memory/round_history.md`, the source tree in `docs/ARCHITECTURE.md`, subsystem rules in `arch-*.md`. For each line ask: *would removing this cause a mistake?* If not, cut it.
- **Emphasis is a budget.** `**bold**`, ⚠ and IMPORTANT work by contrast; when most lines carry one, none of them stands out. Reserve them for rules that have actually been broken.
- Delete entries verifiably outdated (check code first).
- Don't duplicate between `CLAUDE.md` and `rules/` — `CLAUDE.md` summarizes, `rules/` has detail.
