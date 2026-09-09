#!/usr/bin/env bash
# Stop hook: verify no stale work left behind when Claude stops
# Must always exit 0 — stop hooks should never block

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_DIR" 2>/dev/null || exit 0

LOG_DIR="$PROJECT_DIR/.claude/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || true
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/session_${TIMESTAMP}.log"

exec > "$LOG_FILE" 2>&1 || exit 0

# Keep the newest 30. Nothing reads these — the Stop hook's stdout goes to the
# debug log, not to the model — so they are a local trail, not a guard. They
# had reached 2,908 files and 12 MB, 507 of them carrying warnings nobody ever
# opened. If a check here needs to REACH anyone it has to exit 2, which also
# blocks stopping; that trade has not been made deliberately, so this stays a
# log and says so.
ls -t "$LOG_DIR"/session_*.log 2>/dev/null | tail -n +31 | xargs -r rm -f

echo "=== SwarmLLM Session Summary ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Git status
echo "--- Uncommitted Changes ---"
UNSTAGED=$(git diff --name-only 2>/dev/null | wc -l | xargs)
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | xargs)
echo "Unstaged: $UNSTAGED files, Staged: $STAGED files"
if [ "$UNSTAGED" != "0" ] || [ "$STAGED" != "0" ]; then
    echo "WARNING: Uncommitted work detected!"
    git diff --stat 2>/dev/null || true
fi
echo ""

# Recent commits
echo "--- Recent Commits ---"
git log --oneline -5 2>/dev/null || echo "(no commits)"
echo ""

# Quick stale ref check
echo "--- Quick Integrity Check ---"
DOUBLE=$(grep -rn '\.events\.events\.\|\.models\.models\.\|\.credits\.credits\.\|\.metrics\.metrics\.' src/ 2>/dev/null | wc -l | xargs)
if [ "$DOUBLE" != "0" ]; then
    echo "WARNING: $DOUBLE double sub-struct access patterns found"
else
    echo "PASS: No double sub-struct access"
fi

CONSOLE=$(grep -rn 'console\.\(log\|error\|warn\|debug\)' frontend/js/ 2>/dev/null | wc -l | xargs)
if [ "$CONSOLE" != "0" ]; then
    echo "WARNING: $CONSOLE console debug statements in frontend"
else
    echo "PASS: No console debug output"
fi

# Always exit 0
exit 0
