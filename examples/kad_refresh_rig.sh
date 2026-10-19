#!/bin/bash
# FUTURE_WORK #91 (2h): how often idle peers make a node answer their Kademlia
# routing-table refresh. Three empty nodes in a private network namespace
# (re-runs itself under `unshare -rn`, like manifest_digest_rig.sh), seeing each
# other on loopback and nothing else; B and C bootstrap from A. After WARMUP_S,
# A's `/api/admin/stats` is read every SAMPLE_S for WINDOW_S. Every rise in its
# `kad_inbound` `find_node` count is printed with its time, so the refresh
# cadence shows as bursts (in a swarm smaller than k, ~5 lookups per asker per
# refresh); then the window's totals: requests, the peers the answers listed,
# and the bytes A's Kademlia answers sent (`by_protocol`, the
# `/swarmllm/kad/1.0.0` streams the peers opened).
#
# Mechanism check: each node logs the interval it refreshes at (`Kademlia
# routing-table refresh interval`, from the build that sets it); the rig prints
# what each logged — an older binary logs nothing.
#
# A/B on one binary: SWARMLLM_KAD_REFRESH_SECS=300 in the caller's environment
# reaches every node (the old cadence); unset, 600.
#
# Compare arms by the NUMBER of bursts: a refresh's length (one lookup per
# bucket beyond the nearest neighbour's) depends on the nodes' random ids, so a
# burst's size differs from run to run.
#
# ✅ 2026-10-19, three arms at once: v0.3.244 — 5 bursts ~296 s apart, 27
# find_node in 1503 s; this build at SWARMLLM_KAD_REFRESH_SECS=300 — 5 bursts,
# 15; this build — 2 bursts 592 s apart, 6.
#
# Usage: examples/kad_refresh_rig.sh <binary> [out_dir] [window_s]
#   window_s default 1500; WARMUP_S (default 120), SAMPLE_S (default 20)
set -u
if [ -z "${KAD_RIG_IN_NETNS:-}" ]; then
  exec env KAD_RIG_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
BIN="$(readlink -f "$1")"; OUT="${2:-./kad_refresh_out}"; WINDOW_S="${3:-1500}"
WARMUP_S="${WARMUP_S:-120}"; SAMPLE_S="${SAMPLE_S:-20}"
mkdir -p "$OUT"
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }
BASE=$(mktemp -d "$HOME/.kad-rig.XXXXXX")
make_node() { # dir bootstrap
  mkdir -p "$1"
  cat > "$1/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-kad-rig"
enable_mdns = false
enable_upnp = false
relay_forwarding_auto = false

[auto_manage]
enabled = false
prune_enabled = false

[updates]
mode = "off"

[ui]
open_browser_on_start = false
CFG
}
start() { # dir port
  env SWARMLLM_NODE_DATA_DIR="$1" SWARMLLM_INFERENCE_GPU_LAYERS=0 "$BIN" run -p "$2" -v >> "$1/node.log" 2>&1 & echo $!
}
up() {
  for _ in $(seq 1 120); do
    [ -f "$1/api_key" ] && curl -s -m 3 "localhost:$2/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "node on $2 never came up — see $1/node.log"; return 1
}
addr_of() {
  local id
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$1/node.log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/tcp/$(( $2 + 10 ))" "$1/node.log" && break
    sleep 1
  done
  [ -n "$id" ] && echo "/ip4/127.0.0.1/tcp/$(( $2 + 10 ))/p2p/$id"
}
stop() { # pid — the nodes are not this shell's children, so `wait` cannot see them
  kill "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null
}
cleanup() {
  for p in ${PA:-} ${PB:-} ${PC:-}; do stop "$p"; done
  for n in A B C; do cp "$BASE/$n/node.log" "$OUT/$n.log" 2>/dev/null; done
  rm -rf "$BASE"
}
trap cleanup EXIT
read_a() { # file
  curl -s -m 10 -H "Authorization: Bearer $(cat "$BASE/A/api_key")" localhost:8960/api/admin/stats > "$1"
}
# find_node requests, closer peers listed, kad bytes sent on streams peers opened
fields() { # file
  python3 -I -c '
import json, sys
t = json.load(open(sys.argv[1]))["network_traffic"]
f = next((r for r in t.get("kad_inbound", []) if r["kind"] == "find_node"), {})
k = next((r for r in t.get("by_protocol", [])
          if r["protocol"] == "/swarmllm/kad/1.0.0" and r["opened_by"] == "peer"), {})
print(f.get("requests", 0), f.get("closer_peers", 0), k.get("sent_bytes", 0))' "$1"
}

make_node "$BASE/A" ""
PA=$(start "$BASE/A" 8960); up "$BASE/A" 8960 || exit 1
AADDR=$(addr_of "$BASE/A" 8960); [ -n "$AADDR" ] || { echo "A published no listen address"; exit 1; }
make_node "$BASE/B" "\"$AADDR\""; make_node "$BASE/C" "\"$AADDR\""
PB=$(start "$BASE/B" 8980); PC=$(start "$BASE/C" 9000)
up "$BASE/B" 8980 || exit 1; up "$BASE/C" 9000 || exit 1
echo "binary: $("$BIN" --version)   SWARMLLM_KAD_REFRESH_SECS=${SWARMLLM_KAD_REFRESH_SECS:-unset}"
for n in A B C; do
  echo "  $n: $(grep -a -m1 -oE 'kad_refresh_secs=[0-9]+' "$BASE/$n/node.log" || echo 'no refresh interval logged')"
done
sleep "$WARMUP_S"

read_a "$OUT/r0.json"; read -r R0 P0 S0 <<< "$(fields "$OUT/r0.json")"
T0=$(date +%s); LAST=$R0
echo "window ${WINDOW_S} s from $(date -u +%T)Z, A's find_node so far: $R0"
while [ $(( $(date +%s) - T0 )) -lt "$WINDOW_S" ]; do
  sleep "$SAMPLE_S"
  read_a "$OUT/now.json" || continue
  read -r R _ _ <<< "$(fields "$OUT/now.json")"
  [ "${R:-0}" -gt "$LAST" ] && echo "  +$(( $(date +%s) - T0 )) s: find_node +$(( R - LAST ))"
  LAST=${R:-$LAST}
done
read_a "$OUT/r1.json"; read -r R1 P1 S1 <<< "$(fields "$OUT/r1.json")"
SECS=$(( $(date +%s) - T0 ))
echo "RESULT: in $SECS s A answered $(( R1 - R0 )) find_node" \
  "($(python3 -I -c "print(round(($R1 - $R0) * 60 / $SECS, 2))") a minute)," \
  "listing $(( P1 - P0 )) peers, $(( S1 - S0 )) bytes sent" \
  "($(python3 -I -c "print(round(($S1 - $S0) / $SECS, 1))") B/s)"
exit 0
