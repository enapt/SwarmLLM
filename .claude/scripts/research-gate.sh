#!/usr/bin/env bash
# PreToolUse hook (Edit|Write|NotebookEdit|Bash): enforce "research the task and
# check the existing code" at the only moment it is mechanically checkable — the
# instant before a file is MUTATED. Reads are never blocked, so every denial this
# hook issues is resolvable by reading something.
#
# WHY THIS EXISTS, measured 2026-09-16
# ------------------------------------
# .claude/rules/arch-*.md carry `paths:` frontmatter and load automatically when
# you open a file they govern. That mechanism fires on the **Read tool only**.
# `cat`, `sed -n`, `head` and `grep` through Bash do NOT trigger it — confirmed
# against .claude/logs/instructions-loaded.jsonl: a session that had read
# tests/repo_consistency.rs, CLAUDE.md and several src/ files entirely through
# Bash had logged zero `path_glob_match` events, and one Read of
# tests/api_key_side_effects.rs immediately logged arch-guards-and-tests.md.
#
# That matters because a session in bypass-permissions mode is told to prefer
# Bash for reading and editing. Under that instruction the per-subsystem rules
# never load and Claude Code's own read-before-edit guard never applies — the
# whole path-scoped rules architecture is silently inert. Two ways to do the
# thing, one watched (gotcha #590).
#
# THE THREE CHECKS
# ----------------
#   1. per SUBSYSTEM, once per session — mutating a file governed by an
#      arch-*.md that has not loaded is denied, naming the file to Read. The
#      Read loads the rules through the supported mechanism rather than this
#      hook reimplementing it.
#   2. per FILE — mutating an existing file this session has never looked at is
#      denied. This restores read-before-edit for the Bash path.
#   3. per TASK (`prompt_id`) — the first code mutation of a task is denied
#      unless the session consulted what this repo already knows (gotchas,
#      docs/invariants, FUTURE_WORK, sweep-log, closed_findings) or searched the
#      web. This is .claude/rules/workflow.md § "Research EVERY task" item 3,
#      which is the only one of its three items a machine can check.
#
# Fails OPEN on any parse error: a hook that cannot read its payload must not
# become a hook that blocks everything (the lesson pre-edit-check.sh was fixed
# for). Verify it by planting the violation, never by its exit code.
set -uo pipefail

HOOK_INPUT="$(cat)" PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}" python3 <<'PY'
import json, os, re, sys, glob

try:
    d = json.loads(os.environ.get("HOOK_INPUT", ""))
except Exception:
    sys.exit(0)

ROOT = os.environ["PROJECT_DIR"]
tool = d.get("tool_name", "")
ti = d.get("tool_input", {}) or {}
session = d.get("session_id", "")
prompt_id = d.get("prompt_id", "")
transcript = d.get("transcript_path", "")

def allow():
    sys.exit(0)

def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}))
    sys.exit(0)

# ---------------------------------------------------------------- target paths
def rel(p):
    if not p:
        return None
    p = os.path.abspath(os.path.join(ROOT, os.path.expanduser(p)))
    if not p.startswith(ROOT + os.sep):
        return None
    return os.path.relpath(p, ROOT)

targets = []
if tool in ("Edit", "Write", "NotebookEdit"):
    targets = [rel(ti.get("file_path") or ti.get("notebook_path"))]
elif tool == "Bash":
    cmd = ti.get("command", "") or ""
    # Only the mutation forms this repo's sessions actually use — but ALL of
    # them. `examples/research_gate_probe.py` plants one violation per form and
    # prints which are caught; run it after touching anything below, because a
    # pattern that silently stops matching reads exactly like a rule nobody
    # breaks. The first version of this list missed `sed -i -e`, every python
    # in-place edit (a routine editing method here — see `open_cautions.md` on
    # multi-edit python scripts), `perl -pi`, `git checkout --`, `git apply`,
    # `patch <` and `rm`, i.e. most of the Bash path it exists to close.
    pats = [
        r">>?\s*([A-Za-z0-9_./-]+)",          # > file, >> file  (incl. heredocs)
        r"\btee\b\s+(?:-a\s+)?([A-Za-z0-9_./-]+)",
        r"\b(?:mv|cp|install)\b\s+\S+\s+([A-Za-z0-9_./-]+)",
        r"\btruncate\b[^|;&]*\s([A-Za-z0-9_./-]+)",
        r"\brm\b\s+(?:-[a-zA-Z]+\s+)*([A-Za-z0-9_./-]+)",
        r"\bpatch\b[^|;&]*?\s([A-Za-z0-9_./-]+)",
        r"\bgit\s+(?:checkout|restore)\b[^|;&]*?--\s+([A-Za-z0-9_./-]+)",
    ]

    # `sed -i`, `perl -pi` and an in-place python edit all put the path
    # somewhere a single regex cannot reliably find: the expression may be
    # quoted or not, `-e` may separate it from the file, and python builds the
    # path inside a string. Once the command is KNOWN to edit in place, take
    # every path-like token in it and let the `watched` filter below decide —
    # a token that is not a governed repo file is dropped there anyway.
    ANY_PATH = r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)"
    edits_in_place = (
        re.search(r"\bsed\b[^|;&]*?(?:-i\b|--in-place)", cmd)
        or re.search(r"\bperl\b\s+-[a-zA-Z]*i[a-zA-Z]*\b", cmd)
        # A python that WRITES. Reading one must not be gated, so this looks for
        # the write itself rather than for the interpreter.
        or re.search(
            r"""open\s*\([^)]*['"][wa]\+?['"]|\.write_text\s*\(|\.writelines\s*\(|"""
            r"""\.write\s*\(|os\.replace\s*\(|os\.remove\s*\(|shutil\.(?:copy|move)""",
            cmd,
        )
    )
    if edits_in_place:
        pats.append(ANY_PATH)

    for pat in pats:
        for m in re.finditer(pat, cmd):
            targets.append(rel(m.group(1)))

    # `git apply` names its targets inside the PATCH, not on the command line.
    # Read them out of it; an unreadable patch falls through and allows, like
    # every other unknown here.
    m = re.search(r"\bgit\s+(?:apply|am)\b[^|;&]*?\s([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)", cmd)
    if m:
        try:
            with open(os.path.join(ROOT, m.group(1))) as fh:
                for line in fh:
                    if line.startswith("+++ b/"):
                        targets.append(rel(line[6:].strip()))
        except Exception:
            pass

# Paths this hook must never gate: its own logs, build output, scratch.
SKIP = (".claude/logs/", "target/", ".git/", "node_modules/")
targets = [t for t in targets if t and not t.startswith(SKIP)]
if not targets:
    allow()

GOVERNED = ("src/", "frontend/", "crates/", "tests/", "examples/", "vendor/")
DOCS = ("docs/", "CLAUDE.md", "README.md", ".claude/")
watched = [t for t in targets if t.startswith(GOVERNED) or t.startswith(DOCS)]
if not watched:
    allow()

# ------------------------------------------------- what has this session done?
# Two strengths of evidence, deliberately NOT merged:
#   read_tool — opened with the Read tool. Authoritative: the whole file is in
#               context, and this is what triggers the arch-*.md auto-load.
#   seen      — merely NAMED in a Bash/Grep command. `grep -n pat file` proves
#               you looked at ONE LINE, not that you read the file. Treating the
#               two as equal is what made the first version of this hook inert:
#               a grep of arch-network.md for a single needle satisfied "the
#               network rules have loaded", and check 1 never fired.
read_tool, seen, searched_web, knowledge = set(), set(), False, False
KNOWLEDGE = ("docs/invariants/", "docs/FUTURE_WORK.md", "gotchas.md",
             "sweep-log.jsonl", "closed_findings.md", "docs/DIAGNOSTICS.md",
             "open_cautions.md", "docs/ARCHITECTURE.md")
this_task_seen, this_task_web, this_task_knowledge = set(), False, False
cur_prompt = None
prompt_seen_in_transcript = False
try:
    with open(transcript) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid = r.get("promptId") or r.get("prompt_id")
            if pid:
                cur_prompt = pid
                if pid == prompt_id:
                    prompt_seen_in_transcript = True
            msg = r.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name, inp = b.get("name", ""), b.get("input", {}) or {}
                blob = json.dumps(inp)
                in_task = (cur_prompt == prompt_id)
                if name in ("WebSearch", "WebFetch"):
                    searched_web = True
                    if in_task:
                        this_task_web = True
                if name in ("Read", "NotebookRead"):
                    p = rel(inp.get("file_path") or inp.get("notebook_path"))
                    if p:
                        read_tool.add(p)
                        seen.add(p)
                        if in_task:
                            this_task_seen.add(p)
                if name in ("Grep", "Glob", "Bash"):
                    for p in re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+", blob):
                        rp = rel(p)
                        if rp:
                            seen.add(rp)
                            if in_task:
                                this_task_seen.add(rp)
                if any(k in blob for k in KNOWLEDGE):
                    knowledge = True
                    if in_task:
                        this_task_knowledge = True
except Exception:
    allow()          # no transcript to reason from -> do not block

# --------------------------------------------------- 1. governing rules loaded
def glob_to_re(p):
    out, i = "", 0
    while i < len(p):
        if p[i:i+2] == "**":
            out += ".*"
            i += 2
        elif p[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    return re.compile("^" + out + "$")

rules = {}
for f in glob.glob(os.path.join(ROOT, ".claude/rules/arch-*.md")):
    txt = open(f).read()
    if not txt.startswith("---"):
        continue
    for pat in re.findall(r'-\s*"(.*?)"', txt.split("---")[1]):
        rules.setdefault(os.path.relpath(f, ROOT), []).append(glob_to_re(pat))

loaded = set()
logp = os.path.join(ROOT, ".claude/logs/instructions-loaded.jsonl")
try:
    for line in open(logp):
        r = json.loads(line)
        if r.get("session") == session:
            loaded.add(r.get("file", ""))
except Exception:
    pass

for t in watched:
    for rf, regexes in rules.items():
        if any(rx.match(t) for rx in regexes):
            if os.path.basename(rf) in loaded or rf in read_tool:
                continue
            deny(
                f"{t} is governed by {rf}, which has NOT loaded in this session.\n\n"
                f"Those rules auto-load on the **Read tool only** — cat/sed/grep "
                f"through Bash do not trigger them, so working this file through "
                f"Bash means its subsystem rules were never in context.\n\n"
                f"Fix: Read({rf}) — or Read the file you are about to change, "
                f"which loads the same rules via the supported mechanism. Then retry."
            )

# ------------------------------------------------------ 2. look before you cut
for t in watched:
    if not os.path.exists(os.path.join(ROOT, t)):
        continue          # creating a new file: nothing to have read
    if t in seen:
        continue
    deny(
        f"{t} already exists and this session has not looked at it.\n\n"
        f"Read it (or cat/sed it) before overwriting. Claude Code enforces "
        f"read-before-edit for the Edit tool; a Bash redirect or `sed -i` "
        f"bypasses that, which is the hole this check closes."
    )

# ------------------------------------------ 3. research, once per task, on code
if any(t.startswith(GOVERNED) for t in watched):
    # If the current prompt_id never appears in the transcript, prompt scoping is
    # not observable (schema change, compaction, a resumed session). Fall back to
    # session scope rather than denying every edit: a gate that cannot read its
    # own input must not become a gate that blocks everything.
    scoped = this_task_knowledge or this_task_web if prompt_seen_in_transcript \
        else knowledge or searched_web
    if not scoped:
        deny(
            "First code change of this task, and nothing in it has consulted what "
            "this repo already knows.\n\n"
            ".claude/rules/workflow.md § \"Research EVERY task before touching "
            "code\": half the 'new' defects here were re-derivations. Check at "
            "least one before editing:\n"
            "  - memory/gotchas.md            (numbered traps, grep the symptom)\n"
            "  - memory/closed_findings.md    (already investigated and closed)\n"
            "  - docs/invariants/<topic>.md   (the evidence behind a rule)\n"
            "  - docs/FUTURE_WORK.md          (the entry BODY, not the title)\n"
            "  - .claude/sweep-log.jsonl      (grep before re-reporting)\n"
            "  - or WebSearch/WebFetch how a system with more scars solves it\n\n"
            "Then retry. Research that changed nothing is still worth one line "
            "saying so."
        )

allow()
PY
exit 0
