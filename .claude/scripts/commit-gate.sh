#!/usr/bin/env bash
# PreToolUse hook (Bash): before `git commit`, run the one test that cross-checks
# figures between documents — but only when the commit actually touches a file
# those guards read.
#
# WHY: .claude/rules/completeness.md § "A count edited after the test run is an
# untested change". Test counts live in CLAUDE.md x2 and README.md x2, the i18n
# key totals in CLAUDE.md and docs/ARCHITECTURE.md, the MSRV in several places.
# The trap is not "run the tests" — it is that the number can only be written
# down AFTER the run that produced it, so the edit that breaks the guard is the
# one edit no run has seen. That has put main red twice, the second time after a
# fully green suite and a clean clippy.
#
# A stamp file was the obvious design and is the wrong one: it records that the
# test ran, not that it ran against THIS content, and it goes stale in exactly
# the case that matters. Running the guard here needs no state and cannot be
# fooled. It costs ~25 s and only on a commit that touches one of these files.
#
# Fails OPEN everywhere: unparseable payload, missing cargo, or a hook timeout
# all let the commit through. This gate exists to catch an honest mistake, not
# to be a lock.
set -uo pipefail

INPUT=$(cat)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

CMD=$(printf '%s' "$INPUT" | python3 -c \
  "import sys,json; print((json.load(sys.stdin).get('tool_input') or {}).get('command',''))" \
  2>/dev/null || echo "")

# Only `git commit`. Not `git commit --dry-run`, not a commit inside a string.
case "$CMD" in
    *"git commit"*) ;;
    *) exit 0 ;;
esac
case "$CMD" in
    *--dry-run*|*--amend*) exit 0 ;;   # amend re-runs nothing new; dry-run writes nothing
esac

cd "$PROJECT_DIR" 2>/dev/null || exit 0
command -v cargo >/dev/null 2>&1 || exit 0

# Files whose figures a repo_consistency guard reads out of ANOTHER file. A
# change to one of these can only be verified by running the guard.
CROSS_CHECKED='^(CLAUDE\.md|README\.md|docs/ARCHITECTURE\.md|Cargo\.toml|frontend/i18n/.*\.json|docs/book/src/getting-started/installation\.md)$'

CHANGED=$( { git diff --cached --name-only; git diff --name-only; } 2>/dev/null \
           | sort -u | grep -E "$CROSS_CHECKED" || true)
[ -z "$CHANGED" ] && exit 0

OUT=$(cargo test --test repo_consistency --no-default-features \
        --features dev,claude-subscription 2>&1) || {
    FAIL=$(printf '%s' "$OUT" | grep -E "^(test .* FAILED|failures:|thread)" | head -20)
    DETAIL=$(printf '%s' "$OUT" | grep -A12 "^failures:" | head -40)
    python3 - "$CHANGED" "$FAIL" "$DETAIL" <<'PY'
import json, sys
changed, fail, detail = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": (
        "`cargo test --test repo_consistency` FAILS, and this commit touches a "
        "file whose figures another document restates:\n\n"
        f"{changed}\n\n{fail}\n\n{detail}\n\n"
        "A count edited after the test run is an untested change — this is the "
        "guard that has put main red twice. Fix the figure (or the guard), "
        "re-run the test, then commit."
    ),
}}))
PY
    exit 0
}
exit 0
