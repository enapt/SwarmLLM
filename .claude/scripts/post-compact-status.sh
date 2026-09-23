#!/usr/bin/env bash
# SessionStart(compact) hook: after a compaction, tell the model what is still
# uncommitted. Plain stdout from a SessionStart hook is added to its context.
#
# This REPLACES a PreCompact hook that blocked compaction while anything was
# uncommitted, and that hook is what killed session e3b7669c (2026-09-23):
# it kept compaction from running while work sat uncommitted for hours, so
# the context filled; `/compact` was then refused ("33 unstaged ... 2
# untracked"), and "commit and compact" failed with "Prompt is too long",
# because there was no room left to run the commit the hook was asking for.
# Claude Code documents exactly that: a PreCompact block during the
# compaction that recovers from a context-limit error surfaces the error and
# fails the request (code.claude.com/docs/en/hooks, PreCompact).
#
# Its premise was wrong as well. Compaction never touches the disk, so no
# uncommitted file is lost by it; what is lost is the MODEL'S KNOWLEDGE that
# the work is pending and why. That is what this restores — without ever
# being able to block anything. Always exits 0.
set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_DIR" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

UNSTAGED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')

if [ "${UNSTAGED:-0}" -gt 0 ] || [ "${STAGED:-0}" -gt 0 ] || [ "${UNTRACKED:-0}" -gt 0 ]; then
    echo "UNCOMMITTED WORK SURVIVED COMPACTION: $UNSTAGED unstaged, $STAGED staged, $UNTRACKED untracked file(s)."
    echo "The files are intact on disk; the summary may not say what they are or why."
    echo "Before new work: read them (git diff), then verify and commit per .claude/rules/workflow.md."
    echo ""
    git diff --cached --stat 2>/dev/null | tail -25
    git diff --stat 2>/dev/null | tail -25
    if [ "${UNTRACKED:-0}" -gt 0 ]; then
        echo "Untracked:"
        git ls-files --others --exclude-standard 2>/dev/null | head -20 | sed 's/^/  /'
    fi
fi
exit 0
