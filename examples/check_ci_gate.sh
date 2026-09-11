#!/usr/bin/env bash
# Does branch protection still require the checks CI actually produces?
#
# Why this exists
# ---------------
# A required status check is matched to a job by NAME. Rename or remove a job
# and the protection rule keeps naming the old one — which never reports, so
# GitHub waits for it for ever and **every pull request is permanently
# unmergeable**. Two of this repository's required checks were in that state on
# 2026-09-10 (gotcha #530): they enforced nothing, and they blocked everything.
#
# Nothing in CI can catch this on its own. Reading branch protection needs admin
# rights, and the `GITHUB_TOKEN` a workflow gets does not have them, so this is
# a script you run rather than a job that runs itself. It is part of the release
# gate for that reason.
#
# The failure is silent in both directions, so both are reported:
#   * required but never produced  -> blocks every PR, for ever
#   * produced but not required   -> the job runs and its failure gates nothing
#
# Usage:  examples/check_ci_gate.sh [owner/repo] [branch]
# Exit:   0 = in agreement, 1 = drift found, 2 = could not check
set -uo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)}"
BRANCH="${2:-main}"

if [ -z "${REPO:-}" ]; then
  echo "FAIL: no repository given and 'gh repo view' could not name one." >&2
  exit 2
fi

echo "Repository: $REPO   branch: $BRANCH"
echo

req=$(mktemp) actual=$(mktemp)
trap 'rm -f "$req" "$actual"' EXIT

if ! gh api "repos/$REPO/branches/$BRANCH/protection/required_status_checks" \
       --jq '.contexts[]' 2>/dev/null | sort -u > "$req"; then
  echo "COULD NOT CHECK: reading branch protection needs admin rights on $REPO." >&2
  echo "Run this as a user who has them, or check the setting by hand." >&2
  exit 2
fi

# The most recent CI run on the branch is the authority on what job names exist.
# Its own names are what GitHub matches the protection contexts against.
run=$(gh run list --repo "$REPO" --workflow=CI --branch "$BRANCH" \
        --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)
if [ -z "${run:-}" ]; then
  echo "COULD NOT CHECK: no CI run found on $BRANCH to read job names from." >&2
  exit 2
fi
gh run view "$run" --repo "$REPO" --json jobs --jq '.jobs[].name' 2>/dev/null \
  | sort -u > "$actual"

if [ ! -s "$actual" ]; then
  echo "COULD NOT CHECK: CI run $run reported no jobs." >&2
  exit 2
fi

echo "Required contexts: $(wc -l < "$req")    CI jobs in run $run: $(wc -l < "$actual")"
echo

phantom=$(comm -23 "$req" "$actual")
unenforced=$(comm -13 "$req" "$actual")
rc=0

if [ -n "$phantom" ]; then
  echo "BLOCKING — required but no job produces them. Every PR waits for these for ever:"
  echo "$phantom" | sed 's/^/    /'
  echo
  rc=1
fi

if [ -n "$unenforced" ]; then
  echo "UNENFORCED — the job runs on every PR and its failure gates nothing:"
  echo "$unenforced" | sed 's/^/    /'
  echo
  rc=1
fi

if [ "$rc" -eq 0 ]; then
  echo "OK: protection and CI job names agree."
else
  echo "Fix by editing the branch's required status checks to match the job names above."
fi
exit "$rc"
