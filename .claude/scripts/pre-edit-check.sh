#!/usr/bin/env bash
# PreToolUse hook: refuse edits to read-only reference material.
#
# Reads the path from `tool_input.file_path` (and `notebook_path`, which is
# what NotebookEdit uses), which is where Claude Code actually puts it. This
# script previously read a top-level `path` key, which no PreToolUse payload
# has ever contained, so FILE_PATH was always empty and the script fell through
# its "can't parse, let it through" arm on every call. It was inert for as long
# as it existed; verified by feeding it a real payload naming a protected file
# and watching it return allow.
#
# It is registered for Edit|Write|NotebookEdit, not Edit alone: Write is the
# tool that can replace a protected file WHOLE, which is the more destructive
# of the two, and a matcher covering only Edit left it open.
#
# Blocking is exit code 2 with the reason on stderr — per the hooks contract,
# exit 2 blocks the tool call whether or not JSON is printed.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | python3 -c \
  "import sys,json; t=json.load(sys.stdin).get('tool_input',{}); print(t.get('file_path') or t.get('notebook_path') or '')" \
  2>/dev/null || echo "")

# Unparseable input allows the edit: a hook that cannot read its payload must
# not become a hook that blocks everything.
[ -z "$FILE_PATH" ] && exit 0

case "$FILE_PATH" in
    */SWARMLLM_DEV_SPEC.md|*/SwarmLLM_Technical_Specification.docx)
        echo "Refusing to edit $(basename "$FILE_PATH"): specification documents are read-only reference material." >&2
        exit 2
        ;;
esac

exit 0
