---
name: test-module
description: Run the SwarmLLM tests for one module, test target or crate and report results
argument-hint: "<module | test-target | types | vendored | all>"
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
model: haiku
context: fork
background: false
---

# Module Test Runner

Argument: `$ARGUMENTS`. Every command uses the one local feature set via the `cargo dev-test` alias.

- A module name or path (`network`, `src/inference/split/…`) → `cargo dev-test --lib <module path as a filter>`
- `integration`, `integration_phase10_11` → `cargo dev-test --test <name> -- --test-threads=1`
- `yamux_substream`, `repo_consistency`, `api_key_side_effects` → `cargo dev-test --test <name>`
- `types` → `cargo test --locked -p swarmllm-types`
- `vendored` → `cargo test --manifest-path vendor/libp2p-request-response/Cargo.toml --lib`
- `all` → `cargo dev-test`

Report tests run / passed / failed / ignored; for each failure the test name, assertion and file:line. Say so
plainly if the filter matched no tests — "0 tests ran" is not a pass.
