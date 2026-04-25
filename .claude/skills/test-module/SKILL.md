---
name: test-module
description: Run tests for a specific SwarmLLM module and report results
argument-hint: "<module-path>"
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
model: haiku
context: fork
---

# Module Test Runner

Run tests for a specific SwarmLLM module. The user specified: `$ARGUMENTS`

Working directory: `.`

## Instructions

1. Determine the test target from the argument:
   - Module name like `network`, `inference`, `credit`, `api` → run `cargo test $ARGUMENTS`
   - File path → determine the module and run targeted tests
   - `integration` → run `cargo test --test integration -- --test-threads=1`
   - `all` → run `cargo test`

2. Run the tests

3. Report concisely:
   - Tests run / passed / failed / ignored
   - For failures: test name + assertion message + file:line
   - If no tests exist for the module, state that clearly
