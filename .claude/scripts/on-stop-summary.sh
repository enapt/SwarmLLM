#!/usr/bin/env bash
# On-stop hook: logs session summary for continuity
set -euo pipefail

PROJECT_DIR="/home/user/SwarmLLM"
LOG_DIR="$PROJECT_DIR/.claude/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/session_${TIMESTAMP}.log"

{
    echo "=== SwarmLLM Session Summary ==="
    echo "Timestamp: $(date -Iseconds)"
    echo ""

    if [ -d "$PROJECT_DIR/.git" ]; then
        echo "--- Git Status ---"
        cd "$PROJECT_DIR"
        git diff --stat 2>/dev/null || echo "(no git changes)"
        echo ""
        echo "--- Recent Commits ---"
        git log --oneline -5 2>/dev/null || echo "(no commits)"
        echo ""
    fi

    if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
        echo "--- Last Build Status ---"
        if [ -f /tmp/swarmllm-lint-last.log ]; then
            cat /tmp/swarmllm-lint-last.log
        else
            echo "(no build log)"
        fi
    fi
} > "$LOG_FILE" 2>/dev/null

exit 0
