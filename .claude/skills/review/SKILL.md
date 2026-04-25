---
name: review
description: Review recent changes against the SwarmLLM spec for correctness
argument-hint: "[module-or-file]"
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Task
model: sonnet
context: fork
---

# Code Review

Review SwarmLLM code for correctness against the architecture and conventions.

## Scope

If `$ARGUMENTS` is provided, review only that module/file. Otherwise, review all uncommitted changes.

## Instructions

1. Get the scope of changes:
   - If argument provided: read the specified files under ``
   - Otherwise: run `git diff` and `git diff --cached` in `.` to find changed files

2. For each changed file, read `docs/ARCHITECTURE.md` for the relevant subsystem

3. Check for:
   - **Type correctness**: Do data types match existing patterns? (field names, types, derives)
   - **API compliance**: Do HTTP endpoints match existing route patterns and response formats?
   - **Error handling**: Does it follow the error propagation rules from CLAUDE.md?
   - **Naming conventions**: PascalCase types, snake_case functions, proper newtypes
   - **Missing functionality**: Is anything expected but not implemented?
   - **Over-engineering**: Is anything implemented beyond what's needed?

4. For reviews touching 3+ files, spawn a `feature-dev:code-reviewer` subagent (model: haiku) for security and bug analysis in parallel.

5. Report findings as:
   - **BLOCKER**: Must fix before proceeding (architecture violations, security issues)
   - **WARNING**: Should fix (minor deviations, missing edge cases)
   - **NOTE**: Optional improvements

Keep output concise. No findings = just say "Clean."
