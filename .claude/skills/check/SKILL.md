---
name: check
description: Run SwarmLLM's full local quality checks (fmt, clippy, tests, types crate) on the one local feature set and report
disable-model-invocation: true
allowed-tools: Bash, Read
model: haiku
context: fork
background: false
---

# Quality Check Pipeline

Run in order, one at a time (never two cargo commands at once — gotcha #684), and report. Fix nothing.

1. `cargo fmt --check`
2. `cargo lint` — clippy, all targets, `-D warnings`, on `--no-default-features --features dev,claude-subscription`
3. `cargo dev-test 2>&1 | grep -E "^test result|FAILED|panicked"`
4. `cargo test --locked -p swarmllm-types --quiet 2>&1 | grep -E "^test result|error"` — a separate
   package the alias does not cover, so plain `cargo test` here is deliberate

| Step | Status | Details |
|------|--------|---------|
| fmt | PASS/FAIL | files |
| lint | PASS/FAIL | first errors as file:line |
| tests | PASS/FAIL | passed / failed / ignored per target |
| types | PASS/FAIL | |

"All clear." if everything passed; otherwise only the failures, with test name, assertion and file:line.
