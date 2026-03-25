#!/usr/bin/env bash
# Pre-edit hook: validates that edits target expected project files
# Receives JSON on stdin with: { "path": "...", "type": "edit" }
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('path',''))" 2>/dev/null || echo "")

if [ -z "$FILE_PATH" ]; then
    exit 0  # Allow - can't parse, let it through
fi

# Block edits to spec documents
case "$FILE_PATH" in
    */SWARMLLM_DEV_SPEC.md|*/SwarmLLM_Technical_Specification.docx)
        echo '{"decision":"deny","reason":"Cannot edit specification documents - they are read-only reference material"}'
        exit 1
        ;;
esac

# Allow everything else
exit 0
