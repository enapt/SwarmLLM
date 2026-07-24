#!/usr/bin/env bash
#
# Root-run SwarmLLM anchor updater — invoked by swarmllm-update.timer.
#
# The hardened anchor daemon runs as a NON-root, sandboxed service and cannot
# replace its own binary (ProtectSystem=strict + root-owned /usr/local/bin) —
# by design. This root-privileged updater fills that gap the way
# unattended-upgrades does for the OS: check for a newer release, verify its
# checksum, swap the binary, restart the service. Pre-release-aware. Refuses to
# install anything it can't verify, and never downgrades.
set -euo pipefail

REPO="enapt/SwarmLLM"
BIN="/usr/local/bin/swarmllm"
ASSET="swarmllm-linux-x86_64"
SERVICE="swarmllm-anchor.service"

log() { echo "[swarmllm-update] $*"; }

# curl with retry. GitHub's release-asset CDN can return a transient 504 while
# it warms up on a freshly-uploaded binary (the ~minutes right after a release
# is cut) — a single failure must NOT abort the update. --retry covers 5xx and
# timeouts (504 included); --retry-max-time bounds the total retry window.
dl() { curl -fsSL --retry 6 --retry-delay 15 --connect-timeout 30 --retry-max-time 1800 "$@"; }

[[ -x "$BIN" ]] || { log "no binary at $BIN — nothing to update"; exit 0; }
command -v jq >/dev/null || { log "jq not installed"; exit 1; }

CUR=$("$BIN" --version 2>/dev/null | awk '{print $NF}')   # e.g. 0.3.1-alpha
[[ -n "$CUR" ]] || { log "could not read current version"; exit 1; }

# Newest non-draft release (pre-release inclusive; GitHub returns newest-first).
META=$(dl "https://api.github.com/repos/${REPO}/releases" | jq '[.[] | select(.draft==false)][0]')
LATEST=$(echo "$META" | jq -r '.tag_name // empty' | sed 's/^v//')
[[ -n "$LATEST" ]] || { log "no release found"; exit 0; }

if [[ "$CUR" == "$LATEST" ]]; then
  log "already on latest ($CUR)"; exit 0
fi
# Forward-only: latest must sort >= current (version sort), else don't downgrade.
newest=$(printf '%s\n%s\n' "$CUR" "$LATEST" | sort -V | tail -1)
[[ "$newest" == "$LATEST" ]] || { log "current $CUR newer than release $LATEST — not downgrading"; exit 0; }

log "update available: $CUR -> $LATEST"
DL=$(echo "$META"  | jq -r ".assets[]? | select(.name==\"$ASSET\") | .browser_download_url")
SHA=$(echo "$META" | jq -r ".assets[]? | select(.name==\"${ASSET}.sha256\") | .browser_download_url")
[[ -n "$DL" && "$DL" != "null" ]] || { log "release $LATEST has no $ASSET asset"; exit 1; }
[[ -n "$SHA" && "$SHA" != "null" ]] || { log "no .sha256 sidecar — refusing to install unverified binary"; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
dl "$DL" -o "$tmp/sw"
dl "$SHA" -o "$tmp/sw.sha256"
( cd "$tmp" && echo "$(awk '{print $1}' sw.sha256)  sw" | sha256sum -c - ) \
  || { log "CHECKSUM FAILED — aborting, binary untouched"; exit 1; }
chmod +x "$tmp/sw"
# Sanity: the new binary must run and report exactly the version we expect.
GOT=$("$tmp/sw" --version 2>/dev/null | awk '{print $NF}')
[[ "$GOT" == "$LATEST" ]] || { log "new binary reports '$GOT', expected '$LATEST' — aborting"; exit 1; }

install -m 0755 "$tmp/sw" "$BIN"
systemctl restart "$SERVICE"
log "updated $CUR -> $LATEST and restarted $SERVICE"
