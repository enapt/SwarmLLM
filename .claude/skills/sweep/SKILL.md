---
name: sweep
description: Deploy parallel agents to scan the SwarmLLM codebase for dead code, duplication, inconsistencies and stale references, then fix the obvious ones
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Agent
effort: high
---

# Codebase Sweep

Four parallel sonnet `feature-dev:code-reviewer` agents, one category each. They only READ, so no worktree
isolation (every subagent already starts with a fresh context; a worktree only isolates files).

## Prior findings — by grep, never pasted

`.claude/sweep-log.jsonl` holds every finding ever made (~440 KB). **Never paste it into a prompt** — that is
~110K tokens per agent. Tell each agent: *before reporting a finding, `grep -n '<file or symbol>'
.claude/sweep-log.jsonl`; drop it if a `fixed` or `wontfix` entry covers it.*

## Rotation

Offset = (line count of the sweep log ÷ 10) mod file count. Agents 1 and 3 start at that offset in
`find src/ -name '*.rs' | sort`, agents 2 and 4 in `find frontend/js/ -name '*.js' | sort`, wrapping around,
and each scans its whole range.

## Agents (launch all four in one message, `model: sonnet`)

1. **Dead code + stale references** — pub items with no external caller, unreachable arms, comments naming
   removed code, `#[allow(dead_code)]` hiding a real warning.
2. **Duplication** — near-identical blocks (>5 lines), one transformation done in several places, duplicate
   fetches or DOM patterns in the frontend.
3. **Consistency** — an error type chosen at a call site, unbounded collections, unvalidated API input, magic
   numbers, unstructured `tracing` calls.
4. **Frontend + i18n + docs** — English bypassing `I18n.t()`, dead CSS/JS, doc comments or `docs/` text that
   no longer match the code.

Each finding: file, line, what is wrong, confidence ≥ 80%. Not test-only code, not an item listed in
`docs/ARCHITECTURE.md` § "Deferred Items", and "0 new issues" is a valid result — never manufacture one.

## After the agents return

1. Deduplicate, then **verify every finding yourself before acting** — agents miss call sites in adjacent
   directories (`.claude/rules/completeness.md` § "Verify before deleting sweep findings").
2. Fix what is obvious and low-risk immediately, committing as you go. For the rest, research (diagnosis rule
   0) and decide — you manage this project; raise with the user only what needs their hands.
3. Append one line per addressed finding to `.claude/sweep-log.jsonl`:
   `{"file":"…","line":N,"kind":"…","summary":"…","status":"fixed|wontfix|deferred","date":"YYYY-MM-DD"}`
