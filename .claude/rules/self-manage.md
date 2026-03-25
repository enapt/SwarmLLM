# Self-Management Rules

These rules are MANDATORY and automatic. The user should never need to remind you.

## After EVERY code change (automated by hooks, but verify)

1. `cargo fmt` — format all Rust code
2. `cargo clippy --all-targets -- -D warnings` — zero warnings
3. If either fails, fix BEFORE proceeding to next task

## After EVERY task completion

1. Verify the change compiles and passes clippy
2. `git add -A && git commit -m "descriptive message" && git push origin main`
3. If you changed architecture (SharedState, channels, WS types, API endpoints): verify docs are updated
4. If you removed/renamed code: `grep -rn` for the old name across the entire codebase and fix all stale references
5. Move to the next task immediately — don't wait for user confirmation

## After EVERY refactoring session (multi-commit)

Run these checks AUTOMATICALLY before telling the user you're done:
- `cargo test` — all tests must pass
- No double sub-struct access: `grep -rn "\.events\.events\.\|\.models\.models\." src/`
- No direct field bypass: `grep -rn "shared_state\.activity_tx\b" src/ | grep -v state.rs`
- No console debug: `grep -rn "console\.\(log\|error\|warn\)" frontend/js/`
- No broken refs: `grep -rn "_lastModelsData\|wsHealthy" frontend/js/`
- Frontend syntax: `for f in frontend/js/**/*.js; do node -c "$f"; done`

## Before context compaction (enforced by PreCompact hook)

1. All changes must be committed — the hook blocks compaction if there are uncommitted changes
2. Build must pass — the hook blocks compaction if cargo check fails
3. Update memory files with significant findings from this session

## Stay focused

- Work through tasks one at a time
- Don't re-investigate already-fixed issues
- Don't start new sweeps until current fixes are committed
- If stuck on a fix for >5 minutes, skip it and note it for later
- NEVER add `#[allow(dead_code)]` — either fix the code or delete it
- NEVER add `// TODO` or `// FIXME` — either do it now or add it to CLAUDE.md deferred items
