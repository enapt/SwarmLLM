#!/usr/bin/env bash
# PreCompact hook: do not compact over uncommitted work.
#
# Compaction is where work gets lost: what is not committed survives only as
# much of the transcript as the summary keeps. This project's workflow rule is
# to commit after every logical unit precisely so that compaction is cheap.
#
# Exit 2 blocks compaction and puts the stderr text in front of the model, so
# it can commit and let compaction proceed. It is self-limiting: committing
# clears the condition. It deliberately does NOT run `cargo check` — the
# version this replaces did, could not fail on it, and so only added ~30s to
# every compaction.
#
# This script did not exist between 2026-04-08 and 2026-09-09: a frontend
# commit renamed its two sibling hooks and dropped this one, while
# settings.json went on invoking it and the workflow rule went on promising
# what it did. Five months of a documented safety net that was not there.
set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_DIR" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

UNSTAGED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
# Untracked files are the MOST vulnerable thing here: a file written this
# session and never `git add`ed exists only on disk and in the transcript, and
# `git diff` cannot see it. `--exclude-standard` keeps .gitignore'd build
# output and data dirs out of the count.
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')

if [ "${UNSTAGED:-0}" -gt 0 ] || [ "${STAGED:-0}" -gt 0 ] || [ "${UNTRACKED:-0}" -gt 0 ]; then
    {
        echo "Compaction blocked: $UNSTAGED unstaged, $STAGED staged and $UNTRACKED untracked file(s) are uncommitted."
        echo "Commit them before compacting — compaction keeps a summary, not the diff."
        echo ""
        git diff --stat 2>/dev/null | tail -20
        git diff --cached --stat 2>/dev/null | tail -20
        if [ "${UNTRACKED:-0}" -gt 0 ]; then
            echo "Untracked:"
            git ls-files --others --exclude-standard 2>/dev/null | head -20
        fi
    } >&2
    exit 2
fi

exit 0
