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

## Error type discipline

- `SwarmError::Validation` → 400, API input errors
- `SwarmError::ModelNotAvailable` / `ShardNotFound` → 404
- `SwarmError::Config` → startup / config file only
- `SwarmError::Internal` → actual bugs (500)
- `SwarmError::ProviderError` → upstream cloud errors (preserves status)

Never use `Config` or `Internal` for request validation.
