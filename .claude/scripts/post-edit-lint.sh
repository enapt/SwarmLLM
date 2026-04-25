#!/usr/bin/env bash
# Post-edit/write hook: runs cargo check after Rust files are modified
# Feeds errors back to Claude so it can fix them immediately
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path', d.get('path','')))" 2>/dev/null || echo "")

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Only lint Rust files
case "$FILE_PATH" in
    *.rs)
        if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
            cd "$PROJECT_DIR"
            OUTPUT=$(cargo check --message-format=short 2>&1 | tail -30)
            if echo "$OUTPUT" | grep -q "^error"; then
                # Feed errors back to Claude
                echo "$OUTPUT" >&2
                exit 2
            fi
        fi
        ;;
esac

exit 0
