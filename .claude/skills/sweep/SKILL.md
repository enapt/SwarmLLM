---
name: sweep
description: Deploy parallel agents to scan the entire codebase for dead code, duplication, inconsistencies, and stale references
user-invocable: true
allowed-tools: Read, Write, Grep, Glob, Bash, Task, Agent
model: opus
effort: high
---

# Codebase Sweep

Deploy 4 parallel review agents to scan the entire SwarmLLM codebase for issues. Each agent focuses on a different category. All findings are collected, deduplicated, and presented as a prioritized action list.

## Pre-Sweep: Load Prior Findings

Before launching agents, check if `.claude/sweep-log.jsonl` exists. If it does:
1. Read its contents — each line is a JSON object: `{"file":"...","line":N,"kind":"...","summary":"...","status":"fixed|wontfix|deferred","date":"YYYY-MM-DD"}`
2. Extract all entries where `status` is `"fixed"` or `"wontfix"` — these are KNOWN issues
3. Pass the known-issues list to EACH agent with explicit instructions: "Do NOT re-report any of these known issues. Focus on finding NEW issues not in this list."

If the file doesn't exist, proceed normally (first sweep).

## File Rotation Strategy

To avoid always scanning files in the same order (which causes convergence):

1. Get the list of all `.rs` files: `find src/ -name '*.rs' | sort`
2. Get the list of all `.js` files: `find frontend/js/ -name '*.js' | sort`
3. Pick a rotation offset based on the current sweep count (line count of sweep-log.jsonl ÷ 10, modulo file count)
4. Tell Agent 1+3 to start from offset N in the Rust file list, wrapping around
5. Tell Agent 2+4 to start from offset N in the JS/frontend file list, wrapping around

This ensures each sweep round examines files in a different order.

## Agents to Deploy (in parallel, with isolation: "worktree")

IMPORTANT: Launch all agents with `isolation: "worktree"` so they get clean context without session history pollution.

### Agent 1: Dead Code + Stale References (model: sonnet, type: code-reviewer)
- pub functions with zero external callers
- Unused imports, dead constants, unreachable match arms
- Stale comments referencing removed code ("NOTE: X removed", "replaced by Y")
- References to old channel names, old struct fields, removed endpoints
- `#[allow(dead_code)]` that suppress legitimate warnings

### Agent 2: Duplication + Copy-Paste (model: sonnet, type: code-reviewer)
- Nearly identical code blocks in different files (>5 lines)
- Same data transformation done in multiple places
- Duplicate API response shapes for the same data
- Frontend: duplicate fetch calls, duplicate DOM manipulation patterns

### Agent 3: Consistency + Production Readiness (model: sonnet, type: code-reviewer)
- SwarmError type misuse (Config/Internal for validation)
- Unbounded collections without cleanup
- Missing input validation on API endpoints
- Hardcoded magic numbers that should be named constants
- Bare string tracing calls without structured fields

### Agent 4: Frontend + i18n + Docs (model: sonnet, type: code-reviewer)
- Hardcoded English strings bypassing I18n.t()
- Dead CSS rules, dead JS functions, broken references
- Stale doc comments that don't match current code
- CLAUDE.md, ARCHITECTURE.md, book/ out of sync with code

## After Agents Return

1. Deduplicate findings across agents
2. Compare against known issues from sweep-log.jsonl — drop any re-reports
3. Rate each NEW finding by priority (CRITICAL > HIGH > MEDIUM > LOW) and effort (small/medium/large)
4. **Triage into two buckets:**
   - **Auto-fix** (do immediately, no prompting): dead code removal, unused imports, stale comments, dead CSS/JS, missing i18n keys, hardcoded strings, duplicate code extraction, stale doc updates, magic number constants, simple consistency fixes. Anything where the correct fix is obvious and low-risk.
   - **Needs discussion** (present to user): architectural changes, behavior changes, ambiguous deletions (might be used via reflection/macros), security-sensitive fixes, anything touching the inference hot path, changes that affect the public API contract, or findings where you're <90% confident in the fix.
5. **Research before fixing** — Before implementing any non-trivial fix, WebSearch for:
   - Latest docs/best practices for the relevant library or pattern (e.g., libp2p API changes, axum middleware patterns, candle tensor ops)
   - Similar open-source projects solving the same problem — check how they handle it
   - GitHub issues/discussions if the fix involves a known library quirk
   - Even for fixes you're confident about, a quick search often reveals a better idiomatic approach
   - Skip research only for truly mechanical fixes (deleting dead code, removing unused imports, fixing typos)
6. Fix everything in the auto-fix bucket immediately — commit as you go
7. Present only the "needs discussion" items to the user, if any

## After Fixes Are Applied

For every finding that was addressed (fixed, deferred, or won't-fix), append a line to `.claude/sweep-log.jsonl`:
```json
{"file":"src/api/server.rs","line":42,"kind":"dead_code","summary":"unused handle_legacy() function","status":"fixed","date":"2026-04-04"}
```

This log ensures future sweeps skip known issues and focus on genuinely new problems.

## Rules
- Every finding must include: file, line, what's wrong, confidence (80%+ only)
- Do NOT report items that are intentionally deferred (check CLAUDE.md deferred list)
- Do NOT report test-only code as dead (check if it's used in #[cfg(test)] blocks)
- Do NOT re-report anything already in sweep-log.jsonl
- Each agent MUST scan its full assigned file range, not just "interesting" files
- If a sweep round finds 0 new issues, report that clearly — don't manufacture findings
