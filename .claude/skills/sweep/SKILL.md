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
~110K tokens per agent. Tell each agent: *before reporting a finding, search `.claude/sweep-log.jsonl` for the
file or symbol with the Grep tool; drop it if a `fixed` or `wontfix` entry covers it.* (The reviewer agents
have Grep, Glob and Read, but no Bash.)

## Rotation

YOU compute each agent's file list and paste it into its prompt — the agents cannot run `find`. Offset =
(line count of the sweep log ÷ 10) mod file count; agents 1 and 3 take `find src/ -name '*.rs' | sort` from
that offset, agents 2 and 4 `find frontend/js/ -name '*.js' | sort`, wrapping around.

## Docs drift — run it yourself first (seconds, no agent)

`python3 examples/docs_drift.py --memory` checks mechanically what agents read past: every backticked name a
doc cites still exists, a `module::path::item` is defined under that path, each DIAGNOSTICS DIAG row's level
and fields match its `tracing` call, every `file.md § "Heading"` resolves. It is a report — read each line
(proposals and other projects' names are legitimately absent), fix the real ones, log the rest `wontfix`.
Agent 4 then spends its budget on prose and numbers, not names.

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

1. Deduplicate, then **verify every finding yourself before acting** — see "Verify before deleting" below.
2. Fix what is obvious and low-risk immediately, committing as you go. For the rest, research (diagnosis rule
   0) and decide — you manage this project; raise with the user only what needs their hands.
3. Append one line per addressed finding to `.claude/sweep-log.jsonl`:
   `{"file":"…","line":N,"kind":"…","summary":"…","status":"fixed|wontfix|deferred","date":"YYYY-MM-DD"}`

## Verify before deleting

Sweep agents report dead code / orphaned keys with confidence ≥80%, but their grep may miss call sites in adjacent directories (R120 caught Agent 4 missing 6 `enc.*` callers in `init.js`, `core/utils.js`, `chat.js`). Before deleting anything an agent flagged:
```bash
grep -rn "thing_name" frontend/js/ frontend/index.html frontend/css/   # full frontend
grep -rn "thing_name\b" src/ tests/ crates/                            # word-boundary; catches re-exports
```
Cheap to verify; expensive to mis-restore. If callers exist, log as wontfix in `.claude/sweep-log.jsonl` to prevent re-report.
