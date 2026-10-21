#!/bin/bash
# Stage 3 end to end (docs/plans/regional_pipelines.md § Stage 3,
# `auto_manage/coverage/near_chain.rs`): two computers close to each other fetch
# complementary parts of a model that NO computer can run whole, although a FAR
# computer already holds every part of it — and a request is then served by a
# split over the two close ones.
#
# Five nodes in a private network namespace (`unshare -rn`, so a live node on
# this machine is invisible), reaching each other over QUIC (TCP to the P2P
# ports dropped, as split_rig.sh's RIG_TRANSPORT=quic does):
#   F, G  far: each holds every part but is capped like A and B, so neither can
#      run the model whole. Every packet to or from their UDP ports waits
#      FAR_MS/2 (netem on the namespace's loopback, filtered by port), so the
#      others measure them ~FAR_MS away. TWO far copies, because one request
#      raises the routine replica target to 2 (`geo_target_replicas`: demand
#      factor 1.5 on a floor of 1) — with one far copy the control's A fetched
#      part 1 by routine replication and the run could not tell the near chain
#      from it. Carrying (#231) is satisfied too: F and G carry every layer
#      between them. So only a near chain moves a part. (The first run let F
#      run the model whole: the router then handed F every request, rightly —
#      one round trip for the whole reply beats a hop per token on equal
#      processors. A near chain is for a model nobody can run whole.)
#   A, B  near each other (no delay), each holding part 0 only, each capped
#      (`max_ram_mb` at 3/4 of the footprint F names under a tiny cap) to run
#      ~14 of TinyLlama's 22 layers: neither can run it whole, together they
#      can. Holding part 0 lets each fill the model in (the trust gate's
#      exception: no internet reaches the namespace to vouch for it);
#   K  the client, in another region, with a 64 MB cap: its demand reaches A and
#      B as REGIONAL demand (K decays its counts every 600 s, then gossips) —
#      demand in A's own region would raise A's routine replica target and fetch
#      the part for the wrong reason (carry_test.sh's first run).
# Both requests ask for the same long reply: the router prices a far hop per
# token against a near machine's cold start, and a 16-token reply rightly kept
# the warm far route (the second run: B, holding both parts, led a split with F
# rather than load A cold for nine tokens). The third run (160 tokens) still
# went B + far: B held layers 0-12 loaded from the first request, and a router
# holds a peer to its room BESIDE what it has loaded — so the fetcher now gives
# back layers loaded outside its segment once idle past the conversation gap
# (`near_chain::loaded_outside_near_segment`).
# Expect: K's first request is split with F or G (only they hold layers 12-22:
# an AU region in its route); exactly ONE of A and B fetches part 1
# (rendezvous order picks which), logging "DIAG: fetching this computer's part
# of a chain of nearby machines"; K's second request is served over A and B
# alone (regions=TH,TH), faster than the first.
# Control: NEAR_CHAIN_AB=0 starts A and B with SWARMLLM_NEAR_CHAIN=0 (same
# binary, no near chain planned) — neither fetches, and the second request is
# split with a far machine like the first.
#
# usage: [NEAR_CHAIN_AB=0] examples/near_chain_rig.sh <binary-for-F-G-K> [<binary-for-A-and-B>]
set -u
NEW_BIN="${1:?binary for F and K}"
AB_BIN="${2:-$1}"
if [ -z "${NEAR_IN_NETNS:-}" ]; then
  exec env NEAR_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }

MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
WAIT_SECS="${WAIT_SECS:-1500}"
REPLY_TOKENS="${REPLY_TOKENS:-160}"
FAR_MS="${FAR_MS:-200}"
FP=8895; AP=8897; BP=8899; KP=8901; GP=8903
export SWARMLLM_CANONICAL_UPLOADS=0 SWARMLLM_INFERENCE_GPU_LAYERS=0
[ -f "$SRC/shard_000.bin" ] && [ -f "$SRC/shard_001.bin" ] || { echo "no parts of $MODEL in $SRC"; exit 2; }
BASE=$(mktemp -d); echo "base=$BASE model=$MODEL A/B binary=$AB_BIN near chain on A/B: ${NEAR_CHAIN_AB:-on}"
PIDS=""
kill_workers_under() { for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -q "model-worker.*$1" && kill -9 "$(basename "$p")" 2>/dev/null; done; }
cleanup() {
  for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; kill_workers_under "$BASE"
  declare -F safety_stop > /dev/null && safety_stop
}
trap cleanup EXIT
# The gate safety kit (thermal and driver stops, keep-awake, telemetry).
if [ -f ~/swarmllm-gate-common/safety.sh ] && source ~/swarmllm-gate-common/safety.sh \
  && safety_start near_chain_rig "$BASE"; then echo "rig: safety kit on"
else echo "rig: safety kit NOT running"; fi

# Every node reaches the others over QUIC: TCP to each P2P port (API + 10) is
# dropped on arrival, so the loopback probe's TCP dial never connects.
tc qdisc add dev lo handle ffff: ingress || { echo "cannot add an ingress qdisc"; exit 2; }
for p in $FP $AP $BP $KP $GP; do
  tc filter add dev lo parent ffff: protocol ip prio 1 u32 match ip protocol 6 0xff \
    match ip dport "$((p + 10))" 0xffff action drop || { echo "cannot drop TCP to $((p + 10))"; exit 2; }
done
# F and G are far: each one's QUIC socket is its one UDP port, so a filter on
# that port, either way, delays every packet it sends or receives — and nothing
# else. (A packet is classified once: F↔G is FAR_MS too.)
tc qdisc add dev lo root handle 1: prio bands 3 priomap 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
  || { echo "cannot add the prio qdisc"; exit 2; }
tc qdisc add dev lo parent 1:1 handle 10: netem delay "$((FAR_MS / 2))ms" || { echo "cannot add netem"; exit 2; }
for port in $FP $GP; do
  for dir in sport dport; do
    tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip protocol 17 0xff \
      match ip "$dir" "$port" 0xffff flowid 1:1 || { echo "cannot steer $port's $dir"; exit 2; }
  done
done
echo "rig: F and G are ${FAR_MS} ms away from everyone; A, B and K are next to each other"

config() { # dir bootstrap region max_ram_mb auto_manage
  cat > "$1/config.toml" <<CFG
[identity]
region = "$3"

[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-near-chain-test"
enable_mdns = false
enable_upnp = false

[auto_manage]
enabled = $5
prune_enabled = false
interval_seconds = 15
min_replicas = 1

[resources]
max_ram_mb = $4

[updates]
mode = "off"

[ui]
open_browser_on_start = false
CFG
}
up() { # dir port
  for _ in $(seq 1 90); do
    [ -f "$1/api_key" ] && curl -s -m 3 "localhost:$2/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "node on $2 never came up"; tail -20 "$1/log"; return 1
}
start() { # name bin port -> sets ADDR_<name> (QUIC)
  local D="$BASE/$1" near=""
  case "$1" in A|B) near="${NEAR_CHAIN_AB:-}" ;; esac
  SWARMLLM_NEAR_CHAIN="$near" SWARMLLM_NODE_DATA_DIR="$D" "$2" run -p "$3" -v >> "$D/log" 2>&1 &
  PIDS="$PIDS $!"
  up "$D" "$3" || exit 1
  local id=""
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$D/log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/udp/$3/quic-v1" "$D/log" && break
    sleep 1
  done
  [ -n "$id" ] || { echo "$1 never logged its peer id"; exit 1; }
  eval "ADDR_$1=/ip4/127.0.0.1/udp/$3/quic-v1/p2p/$id"
}
ask() { # label -> "<http> <wall>"
  local K t0 t1 code; K=$(cat "$BASE/K/api_key"); t0=$(date +%s.%N)
  code=$(curl -s -m 900 -o "$BASE/$1.json" -w '%{http_code}' -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    "localhost:$KP/v1/chat/completions" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Count from one to sixty in words, one number per line."}],"max_tokens":'"$REPLY_TOKENS"',"temperature":0}')
  t1=$(date +%s.%N); echo "$code $(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')"
}
reply_of() { # label -> tokens, and the start of the reply
  python3 -I -c "
import json, sys
d=json.load(open(sys.argv[1]))
c=(d.get('choices') or [{}])[0].get('message',{}).get('content')
n=(d.get('usage') or {}).get('completion_tokens')
print(n, repr(c if c is not None else d.get('error',{}).get('message',''))[:120])" "$BASE/$1.json" 2>/dev/null
}
# The regions of the route that served the newest request any node completed.
routes() { grep -ah "DIAG: request complete" "$BASE"/{F,G,A,B,K}/log 2>/dev/null | grep -oE "route=[a-z_]+ segments=[0-9]+|regions=[A-Z,]+" | paste -sd' ' | tail -c 400; }

# F's footprint, by its own admission: asked locally under a tiny cap.
mkdir -p "$BASE/F/models/$MODEL"; ln "$SRC"/* "$BASE/F/models/$MODEL/" 2>/dev/null || cp "$SRC"/* "$BASE/F/models/$MODEL/"
config "$BASE/F" "" AU 64 false
start F "$NEW_BIN" $FP
FK=$(cat "$BASE/F/api_key")
N=$(curl -s -m 120 -H "Authorization: Bearer $FK" -H 'Content-Type: application/json' "localhost:$FP/v1/chat/completions" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' | grep -oE "needs about [0-9]+ MB" | head -1 | grep -oE "[0-9]+")
[ -n "$N" ] || { echo "INVALID: F did not name its footprint"; exit 1; }
for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; PIDS=""; kill_workers_under "$BASE"; sleep 2
CAP=$((N * 3 / 4))
echo "footprint N = $N MB; F, G, A and B each capped at $CAP MB"

config "$BASE/F" "" AU "$CAP" false
start F "$NEW_BIN" $FP
mkdir -p "$BASE/G/models/$MODEL"; ln "$SRC"/* "$BASE/G/models/$MODEL/" 2>/dev/null || cp "$SRC"/* "$BASE/G/models/$MODEL/"
config "$BASE/G" "\"$ADDR_F\"" AU "$CAP" false
start G "$NEW_BIN" $GP
for n in A B; do
  mkdir -p "$BASE/$n/models/$MODEL"
  for f in manifest.json gguf_header.bin hf_source.json shard_000.bin; do
    cp "$SRC/$f" "$BASE/$n/models/$MODEL/" 2>/dev/null
  done
done
config "$BASE/A" "\"$ADDR_F\", \"$ADDR_G\"" TH "$CAP" true
start A "$AB_BIN" $AP
config "$BASE/B" "\"$ADDR_F\", \"$ADDR_G\", \"$ADDR_A\"" TH "$CAP" true
start B "$AB_BIN" $BP
mkdir -p "$BASE/K/models/$MODEL"
for f in manifest.json gguf_header.bin; do cp "$SRC/$f" "$BASE/K/models/$MODEL/"; done
config "$BASE/K" "\"$ADDR_F\", \"$ADDR_G\", \"$ADDR_A\", \"$ADDR_B\"" BE 64 false
start K "$NEW_BIN" $KP
echo "nodes up ($(date -u +%T)); letting capabilities and round trips settle"; sleep 60

read -r CODE1 WALL1 < <(ask first)
echo "first request: http=$CODE1 wall=${WALL1}s reply=$(reply_of first)"
echo "  route so far: $(routes)"

echo -n "waiting for A or B to fetch part 1 (up to ${WAIT_SECS}s)"
t0=$(date +%s); FETCHED=""
while [ $(($(date +%s) - t0)) -lt "$WAIT_SECS" ]; do
  for n in A B; do [ -f "$BASE/$n/models/$MODEL/shard_001.bin" ] && FETCHED="$FETCHED $n"; done
  [ -n "$FETCHED" ] && break
  echo -n "."; sleep 15
done
echo " fetched by:${FETCHED:- nobody} after $(($(date +%s) - t0))s"
# A second fetcher, if the plans disagreed, shows up here; and the fetcher,
# idle past the conversation gap, gives back the layers it loaded for the first
# request outside its segment, and says so in its capability.
sleep 60
BOTH=""; for n in A B; do [ -f "$BASE/$n/models/$MODEL/shard_001.bin" ] && BOTH="$BOTH $n"; done
for n in A B; do
  echo "--- $n: near-chain DIAG $(grep -a -c "fetching this computer's part of a chain of nearby machines" "$BASE/$n/log") time(s)"
  grep -aE "near chain planned|fetching this computer's part of a chain|carry it|requesting shard download|P2P shard download complete" "$BASE/$n/log" \
    | cut -c1-240 | head -5
  grep -a "Idle unload" "$BASE/$n/log" | grep -oE "idle_secs=[0-9-]+ threshold_secs=[0-9]+ outside_near_chain_segment=[a-z]+" | sed "s/^/  $n released: /"
done
read -r CODE2 WALL2 < <(ask second)
echo "second request: http=$CODE2 wall=${WALL2}s reply=$(reply_of second)"
LAST=$(grep -ah "DIAG: request complete" "$BASE"/{F,G,A,B,K}/log | sort | tail -1 | grep -oE "route=[a-z_]+ segments=[0-9]+|regions=[A-Z,]+" | paste -sd' ')
echo "  newest request's route: $LAST"
echo
echo "RESULT A/B=$AB_BIN footprint=${N}MB cap=${CAP}MB ; part 1 held by:${BOTH:- nobody} ; first: http=$CODE1 ${WALL1}s $(reply_of first | cut -d' ' -f1) tok ; second: http=$CODE2 ${WALL2}s $(reply_of second | cut -d' ' -f1) tok route: $LAST"
echo "(Stage 3: first split with a far machine (an AU region), exactly one of A/B holds part 1, its DIAG logged, the second regions=TH,TH and faster; fetch off: nobody fetches, second like the first)"
