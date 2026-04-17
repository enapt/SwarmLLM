# Workflow Rules

## Context Window

Main session runs Opus 4.7 with 1M context window (Claude Code v2.1.104). Default effort: medium. Escalate to high effort for complex tasks (multi-file refactors, architecture changes, debugging cross-system issues). This means:
- You have massive context — use it. Don't compact prematurely.
- Subagents should use the cheapest model that works: haiku for search/commands, sonnet for review/design, opus for complex implementation.
- Compact at ~70% usage (not 50%) since we have 1M tokens.
- Note: thinking summaries disabled by default since v2.1.89 (set `showThinkingSummaries: true` to restore).
- Note: autocompact has thrash-loop detection since v2.1.89 — stops after 3x refill-to-limit cycles.

## Commit and Push After Each Task

After each task completes successfully (compiles, tests pass, or otherwise verified working):

1. `git add` the relevant changed files
2. `git commit` with a clear, descriptive message summarizing what was done
3. `git push origin main` immediately after committing
4. Update any relevant docs (CLAUDE.md, docs/, memory/) if the task changed architecture, added features, or fixed significant issues

This is MANDATORY — do not batch commits to the end of a session. Each logical unit of work gets its own commit+push cycle. This prevents work loss during long sessions and context compactions.

## MANDATORY: Run CI Checks Before Every Push

**Before every `git push`**, you MUST run:

1. `cargo fmt` — auto-format all Rust code (NOT `--check`, actually apply formatting)
2. `cargo clippy --all-targets -- -D warnings` — zero warnings allowed

If either fails, fix the issues BEFORE pushing. Never push code that hasn't passed both fmt and clippy locally. The sequence for every push is:
```
cargo fmt && cargo clippy --all-targets -- -D warnings && git add -A && git commit -m "..." && git push origin main
```

If you edited `.rs` files, ALWAYS run `cargo fmt` before committing.

## Memory Management Around Compaction

**Pre-compaction (~70% context usage):** Proactively update memory files with findings and state from the current session BEFORE compaction hits. The PreCompact hook will BLOCK compaction if there are uncommitted changes or build failures.

**Post-compaction:** ALWAYS read memory files FIRST before taking any action:
1. Read `memory/MEMORY.md`
2. Run `git log --oneline -10`
3. Run `git diff HEAD~3 --stat`
4. Do NOT re-do work that's already committed

## Model Selection by Task Complexity

- **Haiku**: File searches, running commands, grep/glob, simple review. Use low effort.
- **Sonnet**: Architecture design, module planning, code review, security review. Use medium effort.
- **Opus**: Complex multi-file implementations, spec compliance, debugging cross-system issues. Use high effort.

## Parallel Work

Use agent teams for large parallel workloads (enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`):
- 3-5 teammates max for most workflows
- Each teammate owns a different set of files — avoid conflicts
- Lead coordinates, teammates self-claim from shared task list
- Use `teammateMode: "tmux"` for split-pane visibility
