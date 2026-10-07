#!/bin/bash
# FUTURE_WORK #231 end to end: a model its holders cannot run is fetched by a
# machine that can, and then served. A tester's report (2026-10-07, v0.3.229):
# the only holder of a whole Qwen 3.5 9B was a peer whose memory cap could never
# hold it, while three graphics cards held none of it; every part had a holder,
# so nothing asked a capable machine to fetch it.
#
# Three nodes in a private network namespace (`unshare -rn`, so a live node on
# this machine is invisible), each in its own region:
#   S  holds the whole model, its `max_ram_mb` at half the model's footprint
#      (its own admission's figure) — it can carry ~7 of TinyLlama's 22 layers;
#   C  holds part 0 only (layers 0-12), plenty of memory, auto-manage on (prune
#      off: this rig is about the FETCH; prune's half is unit-tested);
#      `min_replicas = 1`, so routine replication is ALREADY satisfied by S's one
#      copy — at the default (2) C fetched part 1 thirty seconds in, before any
#      request, for the wrong reason (the first run of this rig). The mechanism
#      check: C's log names carrying ("fetching parts of it to carry it");
#   K  the client, a different region, a 64 MB cap (it can carry nothing, so it
#      is never the carrier). Its demand reaches C as regional demand — demand
#      in C's own region would raise C's routine replica target and fetch the
#      part for the wrong reason.
# Expect: K's first request refused at once ("Not enough memory in the swarm");
# once K's demand reaches C (K decays its counts every 600 s, then gossips),
# C logs "cannot run it — fetching parts of it to carry it" and fetches part 1
# from S; K's next request is served. Run with a v0.3.229 C as the control: C
# never fetches, the second request fails too.
#
# Part 0 is the trust gate's own exception (a node holding part of a model may
# fill it in); with no internet the HuggingFace watcher cannot vouch for it.
#
# usage: examples/carry_test.sh <binary-for-S-and-K> <binary-for-C>
set -u
NEW_BIN="${1:?binary for S and K (must advertise model_memory_ceiling_mb)}"
C_BIN="${2:?binary for C}"
if [ -z "${CARRY_IN_NETNS:-}" ]; then
  exec env CARRY_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }

MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
WAIT_SECS="${WAIT_SECS:-1200}"
SP=8895; CP=8897; KP=8899
export SWARMLLM_CANONICAL_UPLOADS=0 SWARMLLM_INFERENCE_GPU_LAYERS=0
[ -f "$SRC/shard_000.bin" ] && [ -f "$SRC/shard_001.bin" ] || { echo "no parts of $MODEL in $SRC"; exit 2; }
BASE=$(mktemp -d); echo "base=$BASE model=$MODEL"
PIDS=""
kill_workers_under() { for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -q "model-worker.*$1" && kill -9 "$(basename "$p")" 2>/dev/null; done; }
cleanup() { for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; kill_workers_under "$BASE"; }
trap cleanup EXIT

config() { # dir bootstrap region max_ram_mb auto_manage
  cat > "$1/config.toml" <<CFG
[identity]
region = "$3"

[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-carry-test"
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
start() { # name bin port -> sets ADDR_<name>
  local D="$BASE/$1"
  SWARMLLM_NODE_DATA_DIR="$D" "$2" run -p "$3" -v >> "$D/log" 2>&1 &
  PIDS="$PIDS $!"
  up "$D" "$3" || exit 1
  local id=""
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$D/log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/tcp/$(($3 + 10))" "$D/log" && break
    sleep 1
  done
  [ -n "$id" ] || { echo "$1 never logged its peer id"; exit 1; }
  eval "ADDR_$1=/ip4/127.0.0.1/tcp/$(($3 + 10))/p2p/$id"
}
ask() { # label -> "<http> <wall>"
  local K t0 t1 code; K=$(cat "$BASE/K/api_key"); t0=$(date +%s.%N)
  code=$(curl -s -m 600 -o "$BASE/$1.json" -w '%{http_code}' -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    "localhost:$KP/v1/chat/completions" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0}')
  t1=$(date +%s.%N); echo "$code $(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')"
}
reply_of() {
  python3 -c "
import json
d=json.load(open('$BASE/$1.json'))
c=(d.get('choices') or [{}])[0].get('message',{}).get('content')
print(repr(c if c is not None else d.get('error',{}).get('message','')))" 2>/dev/null | cut -c1-200
}

# S's footprint, by its own admission: asked locally under a tiny cap.
mkdir -p "$BASE/S/models/$MODEL"; cp "$SRC"/* "$BASE/S/models/$MODEL/"
config "$BASE/S" "" AU 64 false
start S "$NEW_BIN" $SP
SK=$(cat "$BASE/S/api_key")
N=$(curl -s -m 120 -H "Authorization: Bearer $SK" -H 'Content-Type: application/json' "localhost:$SP/v1/chat/completions" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' | grep -oE "needs about [0-9]+ MB" | head -1 | grep -oE "[0-9]+")
[ -n "$N" ] || { echo "INVALID: S did not name its footprint"; exit 1; }
for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; PIDS=""; kill_workers_under "$BASE"; sleep 2
echo "footprint N = $N MB; S capped at $((N / 2)) MB"

config "$BASE/S" "" AU $((N / 2)) false
start S "$NEW_BIN" $SP
mkdir -p "$BASE/C/models/$MODEL"
for f in manifest.json gguf_header.bin hf_source.json shard_000.bin; do cp "$SRC/$f" "$BASE/C/models/$MODEL/" 2>/dev/null; done
config "$BASE/C" "\"$ADDR_S\"" TH 0 true
start C "$C_BIN" $CP
mkdir -p "$BASE/K/models/$MODEL"
for f in manifest.json gguf_header.bin; do cp "$SRC/$f" "$BASE/K/models/$MODEL/"; done
config "$BASE/K" "\"$ADDR_S\", \"$ADDR_C\"" BE 64 false
start K "$NEW_BIN" $KP
echo "nodes up ($(date -u +%T)); letting capabilities settle"; sleep 45

read -r CODE1 WALL1 < <(ask first)
echo "first request: http=$CODE1 wall=${WALL1}s reply=$(reply_of first)"

echo -n "waiting for C to fetch part 1 (up to ${WAIT_SECS}s)"
t0=$(date +%s); FETCHED=no
while [ $(($(date +%s) - t0)) -lt "$WAIT_SECS" ]; do
  if [ -f "$BASE/C/models/$MODEL/shard_001.bin" ]; then FETCHED=yes; break; fi
  echo -n "."; sleep 15
done
echo " $FETCHED after $(($(date +%s) - t0))s"
CARRIED=$(grep -a -c "fetching parts of it to carry it" "$BASE/C/log")
echo "--- C (carrying named $CARRIED time(s))"; grep -aE "fetching parts of it to carry it|requesting shard download|P2P shard download complete" "$BASE/C/log" | cut -c1-220 | head -4
sleep 20
read -r CODE2 WALL2 < <(ask second)
echo "second request: http=$CODE2 wall=${WALL2}s reply=$(reply_of second)"
echo
echo "RESULT C=$C_BIN footprint=${N}MB first: http=$CODE1 ${WALL1}s ; C fetched part 1: $FETCHED (carrying named: $CARRIED) ; second: http=$CODE2 (fix: first 503 at once, fetched yes with carrying named, second 200; v0.3.229 C: fetched no)"
