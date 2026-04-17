# Workflow Rules

## Context Window & Effort

Opus 4.7 with 1M context, global effort `xhigh` (Claude Code v2.1.112). xhigh is Anthropic's recommended default for coding/agentic work — don't downshift unless you have a reason.
- Use the 1M window — don't compact prematurely. Compact at ~70%, not 50%.
- Toggle effort mid-session with `/effort` for hot loops (drop to medium for repetitive lint loops; bump to max for hardest design/debug tasks).
- Subagents pick model by task: haiku for search/commands, sonnet for review/design/planning, opus for complex implementation only.
- Autocompact has thrash-loop detection (stops after 3 refill-to-limit cycles).

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

## Subagent Strategy (Opus 4.7)

4.7 spawns FEWER subagents by default than 4.6. To get parallelism, frame it positively and explicitly:
- "Spawn specialists for frontend, backend, database in parallel"
- "Fan out one Explore agent per directory under src/inference/"

For single-file edits, simple lookups, or sequential work — work directly. The subagent overhead doesn't pay off.

Model picks:
- **Haiku**: file search, command output, simple greps — low effort
- **Sonnet**: architecture, planning, code/security review — medium effort
- **Opus**: production code, multi-file refactors, cross-system debugging — high or xhigh

## Parallel Work via Agent Teams

Agent teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, `teammateMode: "tmux"`). 3-5 teammates max, each owns a different file set, lead coordinates.
