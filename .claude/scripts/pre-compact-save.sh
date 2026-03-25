#!/usr/bin/env bash
# PreCompact hook: save work state before context compaction
# Ensures nothing is lost when the context window gets compressed
set -uo pipefail

PROJECT_DIR="/home/user/SwarmLLM"
cd "$PROJECT_DIR" 2>/dev/null || exit 0

# 1. Ensure all changes are committed
UNSTAGED=$(git diff --name-only 2>/dev/null | wc -l)
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l)
if [ "$UNSTAGED" -gt 0 ] || [ "$STAGED" -gt 0 ]; then
    echo "WARNING: $UNSTAGED unstaged + $STAGED staged changes before compaction. Commit first." >&2
    exit 2
fi

# 2. Verify build state (quick check, don't block on slow builds)
if ! cargo check 2>&1 | grep -q "Finished"; then
    echo "WARNING: cargo check may have errors. Verify before compaction." >&2
    # Don't block — just warn. The build might be in progress.
fi

exit 0
