# Self-Management Rules

Mandatory + automatic. (Commit/push and CI checks live in `workflow.md` — don't duplicate them here.)

## After every refactoring session, before declaring done

Run these checks even if not asked:
- `cargo test` — all tests pass
- No double sub-struct access: `grep -rn "\.events\.events\.\|\.models\.models\." src/`
- No direct field bypass: `grep -rn "shared_state\.activity_tx\b" src/ | grep -v state.rs`
- No console debug: `grep -rn "console\.\(log\|error\|warn\)" frontend/js/`
- Frontend syntax: `for f in frontend/js/**/*.js; do node -c "$f"; done`

## Stay focused

- Work tasks one at a time. Don't re-investigate already-fixed issues.
- Don't start new sweeps until current fixes are committed.
- If stuck >5 min, skip and note for later.
- NEVER add `#[allow(dead_code)]`, `// TODO`, or `// FIXME` — fix it, delete it, or add to CLAUDE.md deferred items.
- After removing/renaming code: `grep -rn` for the old name and fix every stale reference.
