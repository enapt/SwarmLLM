#!/usr/bin/env bash
# Tag a release ONLY after CI on that exact commit is fully green.
#
# Recreated 2026-07-29 (gotcha #204). v0.3.50-alpha was tagged on a commit whose
# CI had FAILED, because the wait loop treated "run completed" as "run
# succeeded" — a `failure` conclusion satisfied it. Two things are checked here
# and both must hold BEFORE the tag is created:
#
#   1. the run's conclusion is exactly `success`, and
#   2. every individual job in that run concluded `success`
#
# A cancelled run (which is what a second push during the wait produces) is a
# hard stop, not something to wait out.
#
# Usage: scratchpad/tag_when_green.sh v0.3.52-alpha
set -euo pipefail

TAG="${1:?usage: tag_when_green.sh <tag>}"
SHA="$(git rev-parse HEAD)"
echo "Waiting for CI on ${SHA:0:8} before tagging $TAG"

# Only TRACKED changes matter: a tag names a commit, and untracked files are
# not in it. Refusing on untracked content would block on, among other things,
# this script's own first run before it has been committed.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "REFUSING: tracked files are modified — the tag would not match what was tested." >&2
  exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=normal | grep '^??' || true)" ]; then
  echo "NOTE: untracked files present (not part of the tag):"
  git status --porcelain | grep '^??' | sed 's/^/  /'
fi

for _ in $(seq 1 120); do
  read -r status conclusion id <<<"$(gh run list --commit "$SHA" --workflow CI --limit 1 \
    --json status,conclusion,databaseId --jq '.[0] | "\(.status) \(.conclusion) \(.databaseId)"' 2>/dev/null || echo "none none none")"

  case "$status" in
    completed)
      if [ "$conclusion" != "success" ]; then
        echo "REFUSING: CI run $id concluded '$conclusion', not 'success'." >&2
        exit 1
      fi
      # Belt and braces: a run can conclude success while a job was skipped in
      # a way that matters. Require every job green.
      bad="$(gh run view "$id" --json jobs --jq \
        '[.jobs[] | select(.conclusion != "success") | "\(.name)=\(.conclusion)"] | join(", ")')"
      if [ -n "$bad" ]; then
        echo "REFUSING: run $id is 'success' but these jobs are not: $bad" >&2
        exit 1
      fi
      echo "CI green (run $id, all jobs success). Tagging $TAG."
      git tag -a "$TAG" -m "$TAG"
      git push origin "$TAG"
      exit 0
      ;;
    none)
      echo "  no CI run for this commit yet..." ;;
    *)
      echo "  CI $status..." ;;
  esac
  sleep 30
done

echo "REFUSING: timed out waiting for CI. Nothing tagged." >&2
exit 1
