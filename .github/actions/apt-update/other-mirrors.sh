#!/usr/bin/env bash
# After a failed apt fetch on a GitHub runner: put Ubuntu mirrors on OTHER
# domains first in the runner's mirror list, for the rest of the job.
#
# The runner image sends every Ubuntu fetch, the security pocket included,
# through `mirror+file:/etc/apt/apt-mirrors.txt` (actions/runner-images'
# configure-apt-sources.sh: azure.archive.ubuntu.com priority 1,
# archive.ubuntu.com 2, security.ubuntu.com 3), which apt walks lowest
# `priority:` first (apt-transport-mirror(1)). On 2026-11-11 the runner's
# resolver failed for every *.ubuntu.com name for hours while NVIDIA's and
# Microsoft's hosts resolved, so every retry through that list failed the same
# way. The mirrors below carry every noble pocket (checked 2026-11-11), and apt
# checks every index against Ubuntu's archive key whichever mirror serves it:
# a mirror can be slow or stale, never trusted with what it sends.
#
# Called from the retry branch of apt-update, apt-install and gpu-build-env.
set -euo pipefail

list=/etc/apt/apt-mirrors.txt
[ -f "$list" ] || exit 0
# ports.ubuntu.com serves the other architectures, under another path.
[ "$(dpkg --print-architecture)" = amd64 ] || exit 0
# Once a job.
if grep -q 'mirrors.edge.kernel.org' "$list"; then
  exit 0
fi

new=$(mktemp)
{
  printf '%s\tpriority:%s\n' \
    http://mirrors.edge.kernel.org/ubuntu/ 1 \
    http://ftp.osuosl.org/pub/ubuntu/ 2 \
    http://mirror.math.princeton.edu/pub/ubuntu/ 3
  # The runner's own entries, tried after them.
  awk '{
    if (match($0, /priority:[0-9]+/)) {
      n = substr($0, RSTART + 9, RLENGTH - 9) + 10
      sub(/priority:[0-9]+/, "priority:" n)
    }
    print
  }' "$list"
} > "$new"
sudo install -m 0644 "$new" "$list"
rm -f "$new"
echo "apt: Ubuntu mirrors on other domains put first in the runner's mirror list for this job"
