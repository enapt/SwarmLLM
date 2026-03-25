---
name: cleanup
description: Verify all recent changes are complete — no stale refs, no broken tests, no deferred items lost, docs updated
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Task, Agent
model: opus
effort: high
---

# Post-Change Cleanup Verification

Run after any significant refactoring or multi-commit session to ensure nothing was left behind.

## Checklist (execute ALL items)

### 1. Compilation + Tests
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test 2>&1 | grep "test result:"
```
If any fail, fix them before proceeding.

### 2. Stale References Scan
Search for references to any recently removed/renamed items:
- Grep for old function names, old field names, old channel names
- Grep for old WS message types
- Check all `// NOTE:`, `// TODO:`, `// FIXME:` comments — are any stale?
- Check frontend JS for references to removed backend fields

### 3. Sub-Struct Consistency (SharedState)
```bash
# No double prefix
grep -rn "\.events\.events\.\|\.models\.models\.\|\.credits\.credits\.\|\.metrics\.metrics\." src/

# No direct bypass
grep -rn "shared_state\.activity_tx\b\|shared_state\.credit_balance\b" src/ | grep -v state.rs
```

### 4. Frontend Integrity
```bash
# Syntax check all JS
for f in frontend/js/core/*.js frontend/js/components/*.js frontend/js/init.js; do node -c "$f"; done

# No debug output
grep -rn "console\.\(log\|error\|warn\|debug\)" frontend/js/

# No broken refs
grep -rn "_lastModelsData\|wsHealthy\|showLoading\|hideLoading" frontend/js/
```

### 5. i18n Completeness
- All `I18n.t()` keys used in JS must exist in en.json
- All keys in en.json must exist in all 20 other language files

### 6. Doc Freshness
- CLAUDE.md: test count, file counts, architecture description match current code
- docs/ARCHITECTURE.md: endpoints, channels, WS types match code
- memory/MEMORY.md: current state section reflects latest session

### 7. Git Hygiene
- All changes committed and pushed
- No large uncommitted diffs
- Commit messages are descriptive

## Output
Report PASS/FAIL for each item with details on any failures found.
