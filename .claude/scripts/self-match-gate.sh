#!/usr/bin/env bash
# PreToolUse hook (Bash|Write|Edit): refuse a `pgrep -f` / `pkill -f` that can match the shell running it.
#
# `-f` matches the full command line, and the shell that runs the pgrep carries the pattern in its OWN command line
# (`bash -c '... pgrep -f run_x.sh ...'`, every Bash tool call, every `timeout ... bash -c` waiter). pgrep leaves out
# only itself, so a waiter `while pgrep -f X; do sleep 20; done` waits on itself forever — gotchas #274, #283, #385,
# and 2026-11-24, when a rig queued behind another sat 13 min after the first ended. Known three times, walked into a
# fourth, so it is checked here.
#
# Allowed: `-A` / `--ignore-ancestors` (procps-ng 4.0.3+; this box has 4.0.4 — it leaves out every ancestor, the
# running shells included), `-P <pid>` (children of one process only), `-x` (the whole line must match), a bracketed
# pattern (`'[r]un_x.sh'` — the classic `ps | grep '[f]oo'`: the regex never matches its own text), or no `-f` at all
# (`pgrep -x swarmllm` matches a process NAME). Scanned: a Bash command (a `git commit` message is left alone), and
# what Write/Edit puts in a `.sh` / `.bash` file. A comment line is skipped.
#
# Fails OPEN (unparseable payload). Verify by planting, never by exit code (#614):
#     bash .claude/scripts/self-match-gate.sh --probe
set -uo pipefail

if [ "${1:-}" = "--probe" ]; then
  self=$(readlink -f "$0"); fails=0; n=0
  probe() {   # want, tool, json tool_input
    local got; n=$((n + 1))
    got=$(printf '{"tool_name":"%s","tool_input":%s}' "$2" "$3" | bash "$self" | grep -q '"deny"' && echo denied || echo allowed)
    [ "$got" = "$1" ] && echo "  ok      $got  $2 $3" || { echo "  WRONG   $got (want $1)  $2 $3"; fails=$((fails + 1)); }
  }
  probe denied  Bash '{"command":"while pgrep -f run_x.sh > /dev/null; do sleep 20; done"}'
  probe denied  Bash '{"command":"until ! pgrep -f \"cargo build\"; do sleep 5; done"}'
  probe denied  Bash '{"command":"for p in $(pgrep -f model-worker); do kill $p; done"}'
  probe denied  Bash '{"command":"pkill -f stage_bin"}'
  probe denied  Bash '{"command":"pgrep -af '\''run_k|stage_bin'\'' | head"}'
  probe denied  Bash '{"command":"timeout 60 pkill -TERM -f -- \"$MATCH\""}'
  probe denied  Bash '{"command":"x=1; pgrep --full swarmllm"}'
  probe denied  Write '{"file_path":"/tmp/w.sh","content":"#!/bin/bash\nwhile pgrep -f run_x.sh; do sleep 1; done\n"}'
  probe denied  Edit '{"file_path":"/tmp/w.bash","old_string":"a","new_string":"if ! pgrep -f foo; then echo; fi"}'
  probe allowed Bash '{"command":"while pgrep -A -f run_x.sh > /dev/null; do sleep 20; done"}'
  probe allowed Bash '{"command":"pgrep -Af run_x.sh"}'
  probe allowed Bash '{"command":"pgrep --ignore-ancestors -f run_x.sh"}'
  probe allowed Bash '{"command":"pgrep -f '\''[r]un_x.sh'\''"}'
  probe allowed Bash '{"command":"pgrep -P $DAEMON -f model-worker"}'
  probe allowed Bash '{"command":"pgrep -x swarmllm"}'
  probe allowed Bash '{"command":"pgrep -f -x \"bash run.sh\""}'
  probe allowed Bash '{"command":"grep -n '\''pgrep -f'\'' gotchas.md"}'
  probe allowed Bash '{"command":"git commit -m \"a waiter; pgrep -f matched itself\""}'
  probe allowed Write '{"file_path":"/tmp/notes.md","content":"while pgrep -f x; do :; done"}'
  probe allowed Write '{"file_path":"/tmp/w.sh","content":"# never: while pgrep -f run_x.sh\npgrep -x swarmllm\n"}'
  probe allowed Bash '{"command":"ls -la"}'
  probe allowed Bash '{"command":"python3 - <<'\''EOF'\''\nnew = \"a waiter `until ! pgrep -f X` never ends\"\nEOF"}'
  probe allowed Bash '{"command":"echo \"refuses a `pgrep -f` waiter\""}'
  probe denied  Bash '{"command":"bash <<'\''EOF'\''\nwhile pgrep -f run_x.sh; do sleep 1; done\nEOF"}'
  probe denied  Bash '{"command":"cat > /tmp/w.sh <<'\''EOF'\''\nwhile pgrep -f run_x.sh; do sleep 1; done\nEOF\necho after"}'
  [ "$fails" = 0 ] && echo "probe: all $n as expected" || echo "probe: $fails of $n WRONG"
  exit "$fails"
fi

HOOK_INPUT="$(cat)"
case "$HOOK_INPUT" in *pgrep*|*pkill*) ;; *) exit 0 ;; esac
HOOK_INPUT="$HOOK_INPUT" python3 -I <<'PY'
import json, os, re, sys

try:
    d = json.loads(os.environ.get("HOOK_INPUT", ""))
except Exception:
    sys.exit(0)
tool = d.get("tool_name", "")
ti = d.get("tool_input", {}) or {}
if tool == "Bash":
    text = ti.get("command", "") or ""
    if re.match(r"\s*(?:cd\s+\S+\s*&&\s*)?git\s+commit\b", text):
        sys.exit(0)
elif tool in ("Write", "Edit"):
    if not re.search(r"\.(sh|bash)$", ti.get("file_path", "") or ""):
        sys.exit(0)
    text = ti.get("content") if tool == "Write" else ti.get("new_string")
    text = text or ""
else:
    sys.exit(0)

# pgrep/pkill where a command starts: line start, after ; & | ( { or $(, or after a keyword that runs one. Not
# after a backtick: in prose that is markdown quoting, and this repo's shell code writes $( ) (11-24's first refusal
# was a rule's text inside a python heredoc).
CALL = re.compile(r"(?:^|[;&|({]|\$\(|\b(?:do|then|else|if|while|until|exec|sudo|xargs|timeout\s+\S+)\b)"
                  r"\s*(?:!\s*)?\b(pgrep|pkill)\b([^\n;&|)`]*)", re.M)
SHORT = set("aAcdfFgGilnoOPrsStuUvwxe")
TAKES = set("PuUgGstFe")           # a short option that takes the next token as its value
# A heredoc's body is text for its command, so it is scanned only when a shell runs it or it becomes a .sh
# (`bash <<EOF`, `cat > x.sh <<EOF`); a python's or a markdown file's body is prose.
lines, keep, end = [], True, None
for line in text.splitlines():
    if end is not None:
        if line.strip() == end:
            end, keep = None, True
        elif keep:
            lines.append(line)
        continue
    lines.append(line)
    h = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
    if h:
        end = h.group(1)
        keep = bool(re.search(r"\b(?:bash|sh|ssh)\b", line[:h.start()]) or re.search(r">\s*\S+\.(?:sh|bash)\b", line))
bad = []
for line in lines:
    if line.lstrip().startswith("#"):
        continue
    for m in CALL.finditer(line):
        toks = [t.strip("'\"") for t in m.group(2).split()]
        full = anc = parent = exact = False
        pattern, i = None, 0
        while i < len(toks):
            t = toks[i]
            if t == "--":
                pattern = toks[i + 1] if i + 1 < len(toks) else None
                break
            if t.startswith("--"):
                full |= t == "--full"; anc |= t == "--ignore-ancestors"
                parent |= t.startswith("--parent"); exact |= t == "--exact"
                if t in ("--parent", "--signal", "--uid", "--euid", "--group", "--session", "--terminal", "--pidfile"):
                    i += 1
            elif t.startswith("-") and len(t) > 1 and set(t[1:]) <= SHORT:
                full |= "f" in t; anc |= "A" in t; parent |= "P" in t; exact |= "x" in t
                if t[-1] in TAKES:
                    i += 1
            elif t.startswith("-"):
                pass                    # a signal: -9, -TERM, -USR1
            elif t.startswith(">") or t.startswith("2>") or t.startswith("<"):
                break
            else:
                pattern = t
                break
            i += 1
        if full and not (anc or parent or exact or (pattern and "[" in pattern)):
            bad.append(f"{m.group(1)}{m.group(2)}".strip())
if bad:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            "`" + "`, `".join(bad[:3]) + "` can match the shell running it: `-f` reads full command lines, and the "
            "shell that runs a pgrep carries the pattern in its own (every Bash tool call, every `bash -c` waiter). "
            "A waiter on it never ends — gotchas #274/#283/#385, and a rig that sat 13 min behind its own wait on "
            "2026-11-24.\n\nFix: add `-A` (--ignore-ancestors: leaves out this shell and every one above it), or "
            "bracket one character of the pattern (`'[r]un_x.sh'`), or wait on a PID (`wait`, `tail --pid`, "
            "`while kill -0 $pid`), or match a process name with `pgrep -x`."),
    }}))
sys.exit(0)
PY
exit 0
