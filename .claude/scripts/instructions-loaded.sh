#!/usr/bin/env bash
# InstructionsLoaded hook — records which instruction files actually reached the
# model, and why. Exists because .claude/rules/workflow.md documented a PreCompact
# hook that had silently not existed for five months: a documented safety net is a
# claim to verify, not a fact.
#
# Writes one JSONL line per load to .claude/logs/instructions-loaded.jsonl.
# What loaded this session, newest last:
#   jq -r '"\(.at)  \(.reason)  \(.lines)L  \(.file)"' .claude/logs/instructions-loaded.jsonl | tail -20
# Total always-on cost:
#   jq -s '[.[]|select(.reason=="session_start")]|{files:length,bytes:(map(.bytes)|add)}' .claude/logs/instructions-loaded.jsonl
set -uo pipefail
LOG_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/logs"
mkdir -p "$LOG_DIR"
# The hook payload arrives on stdin. Hand it to python through the environment,
# because stdin is already spoken for by the script heredoc — having both is what
# made the first version of this file a silent no-op.
HOOK_INPUT="$(cat)" LOG_PATH="$LOG_DIR/instructions-loaded.jsonl" python3 <<'PY'
import json, os, datetime
log = os.environ["LOG_PATH"]
try:
    d = json.loads(os.environ.get("HOOK_INPUT", ""))
except Exception:
    raise SystemExit(0)          # not our payload; never fail the hook
content = d.get("file_content") or ""
path = d.get("file_path", "")
rec = {
    "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "reason": d.get("load_reason", "?"),
    "file": os.path.basename(path) or path,
    "path": path,
    "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
    "bytes": len(content.encode("utf-8")),
}
with open(log, "a") as f:
    f.write(json.dumps(rec) + "\n")
try:                              # keep the log bounded
    lines = open(log).readlines()
    if len(lines) > 2000:
        open(log, "w").writelines(lines[-2000:])
except OSError:
    pass
PY
exit 0
