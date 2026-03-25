# Continuous Learning Rules

## Learn as you go — update knowledge artifacts in real time

When you discover something new about the codebase during work, update the relevant knowledge source IMMEDIATELY — don't wait until the end of the session.

### What to update and when

| Discovery | Update target | When |
|---|---|---|
| New architectural pattern or convention | `.claude/rules/architecture.md` | Immediately after implementing |
| New gotcha or pitfall | `memory/MEMORY.md` Key Technical Gotchas section | Immediately after encountering |
| Changed file structure (new/renamed/deleted files) | `CLAUDE.md` Repository Structure section | Same commit as the change |
| Changed API endpoint (added/removed/renamed) | `docs/ARCHITECTURE.md` HTTP API section | Same commit as the change |
| Changed SharedState fields or sub-structs | `CLAUDE.md` Architecture section + `.claude/rules/architecture.md` | Same commit as the change |
| Changed broadcast channels or WS message types | `CLAUDE.md` Frontend section + `docs/book/daemon.md` | Same commit as the change |
| Changed frontend JS file structure | `CLAUDE.md` Frontend section + `docs/ARCHITECTURE.md` Frontend section | Same commit as the change |
| New error pattern or debugging technique | `docs/DIAGNOSTICS.md` | After verifying the technique works |
| Test count changed | `CLAUDE.md` Testing section | After confirming with `cargo test` |
| New i18n keys added | Propagate to all 20 language files | Same commit as the JS change |
| Found a repeated mistake | `.claude/rules/` — create or update the relevant rule | After fixing the second occurrence |

### Code map maintenance

`memory/code-map.md` contains the component dependency map. Update it when:
- Adding/removing/renaming a subsystem, API handler, or JS component
- Changing data flow (e.g., new channel, new WS message type, new IPC path)
- Changing SharedState sub-struct membership
- Adding a new frontend component or changing component dependencies

Read `memory/code-map.md` at session start for quick orientation instead of exploring the codebase from scratch.

### Self-audit trigger

After every 3+ commits in a session, check:
1. Did I change any architecture? → Update CLAUDE.md + ARCHITECTURE.md
2. Did I add/remove files? → Update repo structure docs
3. Did I learn a new gotcha? → Add to MEMORY.md
4. Did I repeat a mistake? → Add a rule to prevent it

### Memory hygiene

- Keep MEMORY.md under 200 lines (currently loaded limit)
- Keep CLAUDE.md under 250 lines (truncation risk above 200)
- Move detailed session notes to topic-specific memory files (not MEMORY.md)
- Delete memory entries that are verifiably outdated (check code first)
- Never duplicate info between CLAUDE.md and .claude/rules/ — CLAUDE.md is the summary, rules/ has the detail
