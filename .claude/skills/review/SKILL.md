---
name: review
description: Review uncommitted SwarmLLM changes (or a named module/file) against this repo's rules and invariants
argument-hint: "[module-or-file]"
allowed-tools: Read, Grep, Glob, Bash
model: sonnet
context: fork
background: false
---

# Code Review

Scope: `$ARGUMENTS` if given, otherwise everything in `git diff` and `git diff --cached`.

1. **Open each changed file with the Read tool** — that is what loads its `.claude/rules/arch-*.md` rules;
   `cat`/`grep` do not. Read the `docs/invariants/<topic>.md` section a rule points to before judging code it names.
2. Check, in this order:
   - **One invariant, N paths** (`.claude/rules/architecture.md`): does a shared helper exist that this change
     re-implements or skips? Enumerate the other paths the same property must hold on.
   - **Error typing**: no error type chosen at a call site; `classify_error` / `reclassify_flattened_error`
     (`.claude/rules/completeness.md`).
   - **Live config**: `state.cfg()`, never `state.config`, for anything changeable at runtime.
   - **Wire compatibility**: a new message, field or trailer is gated at the SENDER on a `features` bit.
   - **A DashMap guard across `.await`**, a fixed timeout on variable-size work, i18n for user-visible text.
   - **Tests**: does each new test fail without the fix (diagnosis rule 5)?
3. Report each finding as BLOCKER / WARNING / NOTE with `file:line`, the concrete failure, and the rule it
   breaks. Only findings you would bet on; "Clean." when there are none.
