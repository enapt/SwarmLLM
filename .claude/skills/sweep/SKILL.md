---
name: sweep
description: Deploy parallel agents to scan the entire codebase for dead code, duplication, inconsistencies, and stale references
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Task, Agent
model: opus
effort: high
---

# Codebase Sweep

Deploy 4 parallel review agents to scan the entire SwarmLLM codebase for issues. Each agent focuses on a different category. All findings are collected, deduplicated, and presented as a prioritized action list.

## Agents to Deploy (in parallel)

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
2. Rate each by priority (CRITICAL > HIGH > MEDIUM > LOW) and effort (small/medium/large)
3. Present as a ranked action list
4. Ask user which items to fix, then execute

## Rules
- Every finding must include: file, line, what's wrong, confidence (80%+ only)
- Do NOT report items that are intentionally deferred (check CLAUDE.md deferred list)
- Do NOT report test-only code as dead (check if it's used in #[cfg(test)] blocks)
