#!/usr/bin/env bash
# InstructionsLoaded hook — records which instruction files actually reached the
# model, and why. Exists because .claude/rules/workflow.md documented a PreCompact
# hook that had silently not existed for five months: a documented safety net is a
# claim to verify, not a fact.
#
# Writes one JSONL line per load to .claude/logs/instructions-loaded.jsonl.
# What loaded this session, newest last:
#   jq -r '"\(.at)  \(.reason)  \(.lines)L  \(.file)"' .claude/logs/instructions-loaded.jsonl | tail -20
# Always-on cost of the most recent session start:
#   .claude/scripts/instruction-cost.sh
#
# ⚠ `lines`/`bytes` came from a `file_content` key that no payload has ever
# carried, so both were 0 on every record ever written and the cost query in
# this header reported `bytes: 0`. The size is now measured by STATTING `path`,
# which the payload does carry and which the first version already logged
# correctly. Same family as the `path`-vs-`file_path` bug that made
# pre-edit-check.sh inert: a hook whose exit code is 0 and whose fields are
# empty looks exactly like a hook that is working (gotcha #614, #617).
#
# `session_id` is recorded so a gate can ask "did this rules file load in THIS
# session" without guessing from timestamps; research-gate.sh depends on it.
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

path = d.get("file_path", "")
# Measure the file on disk. `file_content` is not a key any payload has carried;
# trusting it is what made this log report every instruction file as 0 bytes.
lines = bytes_ = 0
try:
    raw = open(path, "rb").read()
    bytes_ = len(raw)
    lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
except OSError:
    pass

rec = {
    "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "session": d.get("session_id", ""),
    "reason": d.get("load_reason", "?"),
    "file": os.path.basename(path) or path,
    "path": path,
    "lines": lines,
    "bytes": bytes_,
    # Recorded so a payload schema change is VISIBLE in the log rather than
    # silently zeroing a field, which is how the bug above survived.
    "keys": sorted(d.keys()),
}
with open(log, "a") as f:
    f.write(json.dumps(rec) + "\n")
try:                              # keep the log bounded
    lines_ = open(log).readlines()
    if len(lines_) > 2000:
        open(log, "w").writelines(lines_[-2000:])
except OSError:
    pass
PY
exit 0
