#!/bin/bash
# FUTURE_WORK #91: a node announces its capability (`NodeCapabilityUpdate`) when
# something in it changes, or every `CAPABILITY_HEARTBEAT` (5 min) — not on every
# 30 s round as it did up to v0.3.233. Measured on the release node on
# 2026-10-10, the every-round announcement was ~60% of the gossip an idle node
# received first-hand (`docs/invariants/network.md` § "The capability, once
# Trickle had worked").
#
# Two empty nodes in a private network namespace (re-runs itself under
# `unshare -rn`, like ceiling_test.sh), seeing each other on loopback and
# nothing else — not the live node (#708), not the internet. Counts the
# capability updates B receives from A over WINDOW seconds after they connect.
# A holds no model, so nothing in its capability changes on its own.
#
# Expect: a gated build 1 in 420 s (the heartbeat) — 2 if A's own figures move;
# v0.3.233 or older ~14 (one per round). With VERBOSE=1 node A logs WHY each
# publish went (`DIAG: capability published why=first|heartbeat|<fields>`) and
# the reasons are printed at the end: a field name there on an idle node is a
# figure holding the gate open.
#
# A busy host is the hard case: free RAM moves with every other program. Run
# `CHURN=1` to allocate and free 1.5 GB every 15 s beside the nodes.
#
# ✅ 2026-10-10: v0.3.232 14; gate with free RAM deadbanded 2 (`ram_available_mb`
# twice); gate with free RAM carried over, CHURN=1, 1 (`first`, `heartbeat`).
#
# Usage: examples/capability_gate_rig.sh <binary> [out_dir] [window_s=420]
set -u
if [ -z "${CAPGATE_IN_NETNS:-}" ]; then
  exec env CAPGATE_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
BIN="$(readlink -f "$1")"; OUT="${2:-./capability_gate_out}"; WINDOW="${3:-420}"
mkdir -p "$OUT"
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }
BASE=$(mktemp -d "$HOME/.capgate-rig.XXXXXX")
V=""; [ "${VERBOSE:-0}" = 1 ] && V="-v"
make_node() { # dir bootstrap
  mkdir -p "$1"
  cat > "$1/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-capgate-rig"
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
start() { env SWARMLLM_NODE_DATA_DIR="$1" SWARMLLM_INFERENCE_GPU_LAYERS=0 "$BIN" run -p "$2" $V >> "$1/node.log" 2>&1 & echo $!; }
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
stats() { curl -s -m 10 -H "Authorization: Bearer $(cat "$1/api_key")" "localhost:$2/api/admin/stats"; }
# "B received NodeCapabilityUpdate (first copies)", "B peers"
reading() {
  python3 -c '
import json, sys
b = json.loads(sys.argv[1])
cap = sum(k["recv_msgs"] for k in b["network_traffic"]["gossip_recv_by_kind"] if k["kind"] == "NodeCapabilityUpdate")
print(cap, b["peers"])' "$(stats "$BASE/B" 8980)"
}
cleanup() {
  kill ${PA:-} ${PB:-} ${PC:-} 2>/dev/null; sleep 4
  for n in A B; do cp "$BASE/$n/node.log" "$OUT/$n.log" 2>/dev/null; done
  rm -rf "$BASE"
}
trap cleanup EXIT

make_node "$BASE/B" ""
PB=$(start "$BASE/B" 8980); up "$BASE/B" 8980 || exit 1
BADDR=$(addr_of "$BASE/B" 8980); [ -n "$BADDR" ] || { echo "B published no listen address"; exit 1; }
make_node "$BASE/A" "\"$BADDR\""
PA=$(start "$BASE/A" 8960); up "$BASE/A" 8960 || exit 1
for _ in $(seq 1 60); do [ "$(reading | awk '{print $2}')" -ge 1 ] 2>/dev/null && break; sleep 2; done
if [ "${CHURN:-0}" = 1 ]; then
  python3 -c '
import sys, time
end = time.time() + float(sys.argv[1])
while time.time() < end:
    b = bytearray(1536 * 1024 * 1024)
    for i in range(0, len(b), 4096):
        b[i] = 1
    time.sleep(15)
    del b
    time.sleep(15)' "$WINDOW" &
  PC=$!
fi
echo "version: $("$BIN" --version)"
r0=$(reading); echo "t=0     B got capability / B peers: $r0"
sleep "$WINDOW"
r1=$(reading); echo "t=${WINDOW}s B got capability / B peers: $r1"
set -- $r0 $r1
echo "RESULT: over ${WINDOW}s B received $(( $3 - $1 )) capability updates (B peers at the end: $4)"
[ -n "$V" ] && grep -a "DIAG: capability published" "$BASE/A/node.log" | sed -E 's/^([^ ]+) .*why=/\1 why=/'
exit 0
