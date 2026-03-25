# Completeness Rules

## Never defer or suppress — fix or delete

When you encounter dead code, stale references, or broken patterns:
- **Fix it now** — do not add `#[allow(dead_code)]`, `// TODO`, or `// FIXME` annotations
- **Delete it** if it's genuinely unreachable (verify with `grep -rn` across entire codebase)
- **If it's a deferred feature** (listed in CLAUDE.md "Deferred" section), leave it but note it's deferred

## After every refactoring change, verify:

1. `grep` for old names — renamed a function? grep for the old name across ALL files
2. `grep` for the old pattern — changed `state.field` to `state.sub.field`? grep for bare `state.field`
3. Check doc comments — did the behavior change? Update the `///` docs
4. Check CLAUDE.md — does the architecture section still match?
5. Check tests — do test assertions reference the old behavior?

## After every commit, before moving on:

Run `/cleanup` if any of these are true:
- You changed SharedState fields or sub-structs
- You added/removed/renamed API endpoints
- You added/removed JS files or changed the script load order
- You changed broadcast channel types or WebSocket message formats
- You modified error types or their HTTP status mappings

## Error types matter

- `SwarmError::Validation` — API input errors (400)
- `SwarmError::Config` — config file / startup errors only
- `SwarmError::Internal` — actual bugs / programming errors (500)
- Never use `Config` or `Internal` for request validation
