---
name: next-task
description: Pick the top item of SwarmLLM's live work queue whose preconditions are met, and start it
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

# Next Task

1. Read the queue: `~/.claude/projects/-home-user-SwarmLLM/memory/next_up.md` § "Ranked queue", and
   `docs/FUTURE_WORK.md` § "▶ PRIORITIES". Take the highest item whose preconditions are met; read its
   FUTURE_WORK entry BODY — its scope is a hypothesis (gotcha #654) and its line numbers drift (#645).
2. Research it before touching code (`.claude/rules/workflow.md` § "Research EVERY task"): how a system with
   more scars does it, the pinned crate's current API, and what `gotchas.md` / `docs/invariants/` already know.
3. Say in two lines what you will change and how you will know it worked (the mechanism, not just the outcome).
4. Implement it yourself (never delegate production code). Use `Explore` (`model: sonnet`) for wide searches,
   `digest` (Haiku) for any bulky log/output/diff, and a sonnet `feature-dev:code-architect` only for a design
   spanning 3+ interconnected files (CLAUDE.md § Subagents).
5. Finish with `cargo fmt && cargo lint` and the narrowest `cargo dev-test` that covers the change, then commit
   and push (`.claude/rules/workflow.md`), and update `next_up.md`.
