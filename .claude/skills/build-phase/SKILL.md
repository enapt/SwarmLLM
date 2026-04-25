---
name: build-phase
description: Execute a specific SwarmLLM build phase from the dev spec
argument-hint: "<phase-number>"
user-invocable: true
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Task
model: opus
---

# Build Phase Executor

You are executing build phase `$ARGUMENTS` for the SwarmLLM project.

## Setup

1. Read `docs/ARCHITECTURE.md` for current architecture
2. Read `CLAUDE.md` for conventions and build phase descriptions

## Execution Strategy

### Pre-implementation (parallelize with subagents)
Before writing code, spawn these in parallel:
- **Explore agent** (model: haiku): scan existing `src/` tree, `Cargo.toml` deps, and report what exists
- **Code-architect agent** (model: sonnet): design a dependency graph of the files to implement, identifying which can be built in parallel vs which depend on others

### Implementation
1. If `Cargo.toml` needs new dependencies, add them first
2. Implement each file in order. For groups of independent files (no cross-dependencies), you may write them in parallel
3. After every 2-3 files, run `cargo check` to catch errors early

### Subagent model assignment
- **Research/exploration** (reading files, scanning codebase): use haiku
- **Architecture decisions** (module design, dependency analysis): use sonnet
- **Code review** (spec compliance, security): use sonnet
- **Running commands** (cargo check, cargo test): use haiku
- **Implementation** (writing actual code): do this yourself (opus)

### Post-implementation
1. Run `cargo fmt` to fix formatting
2. Run `cargo clippy -- -D warnings` and fix any lints
3. Run `cargo test` and fix any failures
4. Report summary: files created, tests passing, any known issues

## Critical Rules

- Follow CLAUDE.md conventions exactly
- Do NOT modify spec documents (SwarmLLM_Technical_Specification.docx)
- If something is ambiguous, check docs/ARCHITECTURE.md or the Technical Specification docx
