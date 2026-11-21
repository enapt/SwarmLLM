#!/usr/bin/env bash
# PreToolUse hook (Edit|Write|NotebookEdit|Bash): refuse to rewrite IN PLACE a script that a running shell is
# executing.
#
# Bash reads a script by byte offset as it goes, so a script rewritten under a running shell resumes part-way
# through the new text: gotcha #551, #813 (2026-10-09, a rig died on a syntax error and its waiter sat 30 min) and
# #1153 (2026-11-21: the safety kit's harness case file was rewritten mid-case; the case died on `line 68: syntax
# error`, its watch loop ran on orphaned). Knowing it did not stop the third, so it is checked here.
#
# Only a rewrite of the SAME inode hurts: the Edit/Write tools, a `>`/`>>` redirect, `tee`, `cp` onto it, `truncate`,
# a python `open(..., 'w')`. A rename leaves the running shell on the old inode, so `sed -i`, `perl -i`, `mv`,
# `install` and `rm` are allowed — and writing a temp file and `mv`-ing it over is the way through a denial.
# Hook scripts under .claude/scripts/ are skipped: every hook of a tool call runs at once, so each would see the
# others "running" while an edit to one of them waits for none.
#
# Fails OPEN (unparseable payload, no /proc). Verify by planting, never by exit code (#614):
#     bash .claude/scripts/running-script-gate.sh --probe
set -uo pipefail

if [ "${1:-}" = "--probe" ]; then
  self=$(readlink -f "$0"); dir=$(mktemp -d); cd "$dir" || exit 1
  printf 'sleep 30\n' > run.sh; printf 'echo\n' > idle.sh
  bash run.sh & runner=$!; sleep 0.3
  fails=0
  probe() {   # want, tool, json tool_input
    local got
    got=$(printf '{"tool_name":"%s","tool_input":%s,"cwd":"%s"}' "$2" "$3" "$dir" | bash "$self" | grep -q '"deny"' && echo denied || echo allowed)
    [ "$got" = "$1" ] && echo "  ok      $got  $2 $3" || { echo "  WRONG   $got (want $1)  $2 $3"; fails=$((fails + 1)); }
  }
  probe denied  Edit '{"file_path":"'"$dir"'/run.sh"}'
  probe denied  Write '{"file_path":"run.sh"}'
  probe denied  Bash '{"command":"echo x > run.sh"}'
  probe denied  Bash '{"command":"echo x >> '"$dir"'/run.sh"}'
  probe denied  Bash '{"command":"printf x | tee run.sh"}'
  probe denied  Bash '{"command":"cp /etc/hostname run.sh"}'
  probe denied  Bash '{"command":"cd '"$dir"' && python3 - <<EOF\nopen(\"run.sh\", \"w\").write(\"x\")\nEOF"}'
  probe allowed Bash '{"command":"sed -i s/30/31/ run.sh"}'
  probe allowed Bash '{"command":"cp /etc/hostname run.sh.new && mv run.sh.new run.sh"}'
  probe allowed Bash '{"command":"cat run.sh > copy.sh"}'
  probe allowed Edit '{"file_path":"'"$dir"'/idle.sh"}'
  probe allowed Bash '{"command":"echo x > idle.sh"}'
  kill "$runner" 2>/dev/null; wait "$runner" 2>/dev/null; cd /; rm -rf "$dir"
  [ "$fails" = 0 ] && echo "probe: all 12 as expected" || echo "probe: $fails WRONG"
  exit "$fails"
fi

HOOK_INPUT="$(cat)" PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}" python3 -I <<'PY'
import json, os, re, sys

try:
    d = json.loads(os.environ.get("HOOK_INPUT", ""))
except Exception:
    sys.exit(0)

def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}))
    sys.exit(0)

# Scripts a shell is running now: argv = [shell, opts..., script, args...]; `-c` runs a string, not a file.
SHELLS = {"bash", "sh", "dash", "zsh", "ksh"}
HOOKS = os.path.realpath(os.path.join(os.environ.get("PROJECT_DIR", ""), ".claude/scripts")) + os.sep
running = {}
try:
    pids = [p for p in os.listdir("/proc") if p.isdigit()]
except Exception:
    sys.exit(0)
for pid in pids:
    try:
        argv = [a.decode(errors="replace") for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
    except Exception:
        continue
    if not argv or os.path.basename(argv[0]) not in SHELLS:
        continue
    rest, i = argv[1:], 0
    while i < len(rest) and rest[i][:1] in "-+":
        if not rest[i].startswith("--") and "c" in rest[i][1:]:
            rest = []
            break
        i += 2 if rest[i] in ("-o", "+o", "-O", "+O") else 1
    if i >= len(rest):
        continue
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        cwd = "/"
    path = os.path.realpath(os.path.join(cwd, rest[i]))
    if os.path.isfile(path) and not path.startswith(HOOKS):
        running.setdefault(path, (pid, " ".join(argv)[:160]))
if not running:
    sys.exit(0)

tool = d.get("tool_name", "")
ti = d.get("tool_input", {}) or {}
here = d.get("cwd") if os.path.isabs(d.get("cwd") or "") else os.getcwd()

def absp(p, base):
    return os.path.realpath(os.path.join(base, os.path.expanduser(p)))

targets = set()
if tool in ("Edit", "Write", "NotebookEdit"):
    p = ti.get("file_path") or ti.get("notebook_path")
    if p:
        targets.add(absp(p, here))
elif tool == "Bash":
    cmd = ti.get("command", "") or ""
    lead = re.match(r"""\s*cd\s+(?:'([^']*)'|"([^"]*)"|([^\s;&|]+))\s*(?:&&|;)""", cmd)
    if lead:
        here = absp(next(g for g in lead.groups() if g is not None), here)
    TOK = r"""['"]?([~A-Za-z0-9_./-]+)"""
    for pat in (r"(?<![0-9&<])>>?\s*" + TOK,                       # > file, >> file
                r"\btee\b\s+(?:-a\s+)?" + TOK,
                r"\bcp\b(?:\s+-[a-zA-Z]+)*\s+\S+\s+" + TOK,          # cp src DEST (same inode)
                r"\btruncate\b[^|;&]*\s" + TOK):
        for m in re.finditer(pat, cmd):
            targets.add(absp(m.group(1), here))
    # A python that rewrites a file in place: every path-like token is a candidate (only running scripts matter).
    if re.search(r"""open\s*\([^)]*['"][wa]\+?['"]|\.write_text\s*\(""", cmd):
        for m in re.finditer(r"[~A-Za-z0-9_./-]+", cmd):
            targets.add(absp(m.group(0), here))

for t in sorted(targets):
    if t in running:
        pid, argv = running[t]
        deny(
            f"{t} is being run by a shell right now (pid {pid}: {argv}).\n\n"
            f"Bash reads a script by byte offset as it goes, so rewriting it in place makes the running copy resume "
            f"part-way through the new text — gotchas #551, #813, #1153 (2026-11-21: a harness case died on a syntax "
            f"error and its watch loop ran on orphaned).\n\n"
            f"Fix: write the new version to a temp file and `mv` it over the old (a rename leaves the running shell "
            f"on the old inode; `sed -i` does that too), or wait for pid {pid} to end."
        )
sys.exit(0)
PY
exit 0
