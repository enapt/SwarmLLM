#!/usr/bin/env bash
# PreToolUse hook (Bash): refuse a `git commit` that touches product code
# (src/, crates/, vendor/, frontend/) while this version's release is TAGGED but
# not yet PUBLISHED — the window in which a fix lands on main and misses the
# release the user is about to sign.
#
# WHY: the user asked twice for one release carrying all the work ("ship all
# fixes when releasing. i dont want to make 2 releases", 2026-10-30; "you need
# to included ALL fixes in releases", 2026-11-07). Both times fixes landed after
# the tag: v0.3.276 was tagged at 17:10Z for a 00:00Z signing and 7 fixes landed
# after it. The rule lived in notes and was not followed, so it is enforced here
# (`memory/feedback_batch_releases.md`, CLAUDE.md § Releases).
#
# A fix that MUST ship in the release being signed re-cuts it: tell the user, then
# commit with SWARMLLM_RECUT=1 in the command, abandon the unsigned draft and cut
# the release again from the new HEAD. Docs, tests outside those directories and
# memory are never refused.
#
# Fails OPEN when the release state cannot be read (no gh, offline), and says so.
# Test hooks: RWG_TEST_STATE=draft|published|missing and RWG_TEST_CHANGED=<files>
# replace the GitHub lookup and the working-tree diff.
set -uo pipefail

INPUT=$(cat)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CMD=$(printf '%s' "$INPUT" | python3 -c \
  "import sys,json; print((json.load(sys.stdin).get('tool_input') or {}).get('command',''))" \
  2>/dev/null || echo "")

case "$CMD" in *"git commit"*) ;; *) exit 0 ;; esac
case "$CMD" in *--dry-run*|*SWARMLLM_RECUT=1*) exit 0 ;; esac
cd "$PROJECT_DIR" 2>/dev/null || exit 0

if [ -n "${RWG_TEST_CHANGED:-}" ]; then
    CHANGED=$(printf '%s\n' $RWG_TEST_CHANGED)
else
    CHANGED=$( { git diff --cached --name-only; git diff --name-only; } 2>/dev/null | sort -u)
fi
PRODUCT=$(printf '%s\n' "$CHANGED" | grep -E '^(src|crates|vendor|frontend)/' | head -8 || true)
[ -z "$PRODUCT" ] && exit 0

VER=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml 2>/dev/null | head -1)
[ -z "$VER" ] && exit 0
TAG="v$VER"
# Not tagged yet: either released work is behind us or the bump is being
# prepared — a fix now is cut into the tag. Nothing to refuse.
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1 || exit 0

if [ -n "${RWG_TEST_STATE:-}" ]; then
    STATE=$RWG_TEST_STATE
else
    OUT=$(timeout 15 gh release view "$TAG" --repo enapt/SwarmLLM --json isDraft --jq .isDraft 2>&1)
    case "$OUT" in
        false) STATE=published ;;
        true) STATE=draft ;;
        *"not found"*) STATE=missing ;;   # tagged, the Release run has not made the draft yet
        *) STATE=unknown ;;
    esac
fi

case "$STATE" in
    published) exit 0 ;;
    unknown)
        python3 - "$TAG" <<'PY'
import json, sys
msg = (f"release-window gate: could not read whether {sys.argv[1]} is published (gh failed); "
       "the commit is allowed. If that release is still a draft, this product change misses it.")
print(json.dumps({"systemMessage": msg, "hookSpecificOutput": {
    "hookEventName": "PreToolUse", "additionalContext": msg}}))
PY
        exit 0 ;;
esac

python3 - "$TAG" "$STATE" "$PRODUCT" <<'PY'
import json, sys
tag, state, files = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": (
        f"{tag} is TAGGED and not published ({state}). Product code committed now misses the "
        "release the user is about to sign — the user's rule (2026-10-30, 2026-11-07): the release "
        "they sign carries EVERY fix on main.\n\nProduct files in this commit:\n" + files + "\n\n"
        "Until the publish: research, measure, write designs, FUTURE_WORK and docs — no product code. "
        "If this fix must ship in the release being signed, RE-CUT: tell the user, commit with "
        "SWARMLLM_RECUT=1 in the command, abandon the unsigned draft and cut the release again from "
        "the new HEAD (memory/feedback_batch_releases.md)."
    ),
}}))
PY
exit 0
