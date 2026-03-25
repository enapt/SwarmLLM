---
name: check
description: Run full quality checks on SwarmLLM (fmt, clippy, test, build)
user-invocable: true
allowed-tools: Bash, Read, Glob
model: haiku
context: fork
---

# Quality Check Pipeline

Run the full SwarmLLM quality check pipeline in order. Report results for each step.

Working directory: `/home/user/SwarmLLM`

## Steps

1. **Format check**: `cargo fmt --check`
   - If formatting issues found, run `cargo fmt` to fix them and report what changed
2. **Lint**: `cargo clippy -- -D warnings`
   - Report any warnings/errors with file:line references
   - Do NOT fix anything — just report
3. **Test**: `cargo test`
   - Report pass/fail counts
   - If failures, report test name and assertion message
4. **Build**: `cargo build`
   - Confirm clean build or report errors

## Output

Provide a concise summary table:

| Step | Status | Details |
|------|--------|---------|
| fmt  | PASS/FAIL | ... |
| clippy | PASS/FAIL | N warnings |
| test | PASS/FAIL | N passed, N failed |
| build | PASS/FAIL | ... |

If all pass, say "All clear." If any fail, list only the failures with actionable details.
