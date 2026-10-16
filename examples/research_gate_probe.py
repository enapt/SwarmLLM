#!/usr/bin/env python3
"""Plant one violation per Bash mutation form and report which the gate catches.

`.claude/scripts/research-gate.sh` exists to close the Bash editing path, where
`arch-*.md` rules never load and Claude Code's own read-before-edit check does
not apply. Its coverage is a list of regexes, and a regex that silently stops
matching reads EXACTLY like a rule nobody breaks — gotcha #413, and the reason
the hook's own header says to verify it by planting the violation rather than by
its exit code (#614: a hook is designed to exit 0).

This is `arch-guards-and-tests.md` § "A source-scanning guard is only as good as the
spellings it knows (2026-08-30)" — its rule to give every scan a self-test that plants
the violation — applied to a guard that lives in bash rather than in
`repo_consistency.rs`. It is not run by `cargo test` — it shells out to bash and
python and would add that dependency to CI for a file that changes rarely. Run
it by hand after touching the gate's patterns:

    python3 examples/research_gate_probe.py

Every form under MUST_CATCH has to print `denied`; every form under MUST_ALLOW
has to print `allowed`. A `NOT CAUGHT` is either a real gap to close or a form
that belongs in a documented-gap list with its reason.

The governed path is assembled at runtime rather than written as a literal, so
this file does not trip the live gate when the repo itself is being edited.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, ".claude", "scripts", "research-gate.sh")
TARGET = "/".join(["src", "network", "manager", "relay.rs"])

# Must all be denied: a session that has not read the file or its rules cannot
# mutate it through any of these.
MUST_CATCH = [
    ("redirect >",            "cat > {t} <<'EOF'"),
    ("append >>",             "echo x >> {t}"),
    ("tee",                   "echo x | tee {t}"),
    ("cp onto file",          "cp /tmp/x {t}"),
    ("mv onto file",          "mv /tmp/x {t}"),
    ("truncate",              "truncate -s 0 {t}"),
    ("rm",                    "rm {t}"),
    ("rm -f",                 "rm -f {t}"),
    ("awk redirect",          "awk '{{print}}' x > {t}"),
    ("sed -i, quoted",        "sed -i 's/a/b/' {t}"),
    ("sed -i, unquoted",      "sed -i s/a/b/ {t}"),
    ("sed -i -e",             "sed -i -e 's/a/b/' {t}"),
    ("sed --in-place",        "sed --in-place 's/a/b/' {t}"),
    ("perl -pi -e",           "perl -pi -e 's/a/b/' {t}"),
    ("python open(w)",        "python3 -c \"open('{t}','w').write('x')\""),
    ("python heredoc",        "python3 <<'EOF'\nopen('{t}','w').write('x')\nEOF"),
    ("python write_text",     "python3 -c \"from pathlib import Path; Path('{t}').write_text('x')\""),
    ("git checkout --",       "git checkout -- {t}"),
    ("git restore --",        "git restore -- {t}"),
    ("patch < diff",          "patch {t} < /tmp/p.patch"),
]

# Must NOT be denied — reading is never gated, and a gate that blocks reads is
# exactly the failure `pre-edit-check.sh` sat in for months.
MUST_ALLOW = [
    ("cat (read)",            "cat {t}"),
    ("grep (read)",           "grep -n foo {t}"),
    ("sed -n (read)",         "sed -n '1,20p' {t}"),
    ("python read-only",      "python3 -c \"print(open('{t}').read())\""),
    ("outside the repo",      "echo hi > /tmp/foo.txt"),
    ("build output",          "echo hi > target/debug/x.txt"),
]


def run(cmd, transcript, tool="Bash", tool_input=None):
    payload = json.dumps({
        "tool_name": tool,
        "tool_input": tool_input or {"command": cmd},
        "session_id": "PROBE-SESSION-NOT-IN-LOG",
        "prompt_id": "PROBE-PROMPT",
        "transcript_path": transcript,
    })
    r = subprocess.run(
        ["bash", GATE], input=payload, capture_output=True, text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": ROOT},
    )
    return bool(r.stdout.strip())


def transcript_with(records):
    tf = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    tf.write(json.dumps({"promptId": "PROBE-PROMPT",
                         "message": {"role": "user", "content": []}}) + "\n")
    for r in records:
        tf.write(json.dumps(r) + "\n")
    tf.close()
    return tf.name


# Check 2 (look before you cut) reads the transcript: a file this session WROTE
# counts as seen, but only when the write succeeded — a write the gate denied is
# in the transcript too, flagged `is_error`. An existing file no arch-*.md
# governs, so check 1 cannot be what answers.
SEEN_TARGET = "/".join(["docs", "TESTING.md"])


def write_then_result(is_error):
    use = {"type": "tool_use", "id": "probe-w1", "name": "Write",
           "input": {"file_path": os.path.join(ROOT, SEEN_TARGET), "content": "x"}}
    res = {"type": "tool_result", "tool_use_id": "probe-w1", "content": "probe"}
    if is_error:
        res["is_error"] = True
    return [{"promptId": "PROBE-PROMPT", "message": {"role": "assistant", "content": [use]}},
            {"promptId": "PROBE-PROMPT", "message": {"role": "user", "content": [res]}}]


SEEN_CASES = [
    # (description, transcript records, must be denied)
    ("unseen file, empty transcript", [], True),
    ("a successful Write opens it", write_then_result(False), False),
    ("a DENIED Write does not", write_then_result(True), True),
]


def main():
    tf = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    tf.write(json.dumps({"promptId": "PROBE-PROMPT",
                         "message": {"role": "user", "content": []}}) + "\n")
    tf.close()

    failures = []
    try:
        print("must be DENIED (mutating a file whose rules never loaded):")
        for desc, tmpl in MUST_CATCH:
            caught = run(tmpl.format(t=TARGET), tf.name)
            print(f"  {'denied' if caught else 'NOT CAUGHT':<12} {desc}")
            if not caught:
                failures.append(f"not caught: {desc}")

        print("\nmust be ALLOWED (reads, and paths the gate does not govern):")
        for desc, tmpl in MUST_ALLOW:
            caught = run(tmpl.format(t=TARGET), tf.name)
            print(f"  {'DENIED' if caught else 'allowed':<12} {desc}")
            if caught:
                failures.append(f"wrongly denied: {desc}")

        print("\nlook before you cut (Edit of an existing file, by what the transcript shows):")
        for desc, records, must_deny in SEEN_CASES:
            t = transcript_with(records)
            try:
                denied = run(None, t, tool="Edit",
                             tool_input={"file_path": os.path.join(ROOT, SEEN_TARGET),
                                         "old_string": "a", "new_string": "b"})
            finally:
                os.unlink(t)
            ok = denied == must_deny
            print(f"  {('denied' if denied else 'allowed') if ok else ('DENIED' if denied else 'ALLOWED'):<12} {desc}")
            if not ok:
                failures.append(f"look-before-you-cut: {desc}")

        # A probe that cannot fire proves nothing (gotcha #502/#525). If the
        # gate were inert every line above would read "allowed", which the
        # first section would catch — but say it explicitly rather than leaving
        # it to be inferred.
        print("\nnull control: the gate must be capable of denying at all")
        if not run(f"echo x > {TARGET}", tf.name):
            failures.append("null control: the gate denied NOTHING — it is inert")
            print("  INERT     the gate is not firing at all")
        else:
            print("  ok        the gate fires")
    finally:
        os.unlink(tf.name)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
