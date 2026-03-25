#!/usr/bin/env bash
# Post-write hook: runs cargo check after Rust files are written
# Async - runs in background, doesn't block Claude
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('path',''))" 2>/dev/null || echo "")

PROJECT_DIR="/home/user/SwarmLLM"

# Only lint Rust files
case "$FILE_PATH" in
    *.rs)
        # Check if Cargo.toml exists (project initialized)
        if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
            cd "$PROJECT_DIR"
            # Quick syntax check - cargo check is faster than build
            cargo check --message-format=short 2>&1 | tail -20 > /tmp/swarmllm-lint-last.log || true
        fi
        ;;
esac

exit 0
