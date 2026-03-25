---
name: next-task
description: Determine and start the next task in the SwarmLLM build sequence
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Task
model: opus
---

# Next Task Resolver

Determine what should be built next in the SwarmLLM project and begin working on it.

## Instructions

1. **Assess state** (use haiku-model subagents in parallel for speed):
   - Spawn Explore agent: scan `/home/user/SwarmLLM/src/` tree to inventory existing files
   - Read `/home/user/SwarmLLM/docs/plans/NEXT_STEPS.md` for the prioritized roadmap
   - Run `cargo check` to verify current compilation state

2. Identify the highest-priority incomplete item from NEXT_STEPS.md

3. Report what you're about to build:
   ```
   Task: [description]
   Files: [paths]
   Dependencies: [what must exist first]
   ```

4. Implement the task following CLAUDE.md conventions
   - Read `docs/ARCHITECTURE.md` for current architecture context
   - Use `feature-dev:code-architect` agent (model: sonnet) for complex module design if the task involves 3+ interconnected files
   - After writing each file, run `cargo check`

5. After implementation:
   - Run `cargo check` to verify compilation
   - Run `cargo clippy -- -D warnings`
   - Run `cargo test`
   - Report what was completed and what the NEXT task would be
