#!/bin/bash
# FUTURE_WORK #129 end to end: a requester hands a whole model to a peer that
# must first LOAD it, and the load takes longer than the old flat 120 s
# first-token budget (fault injection on the server:
# SWARMLLM_FAULT_LOAD_DELAY_SECS). On 2026-10-07 exactly this failed live: a
# peer that had restarted six minutes earlier gave no first token inside 132 s,
# the request failed with no other holder to re-plan onto, and the same
# question 47 s later was answered in 2.7 s.
#
# Three arms, each run with the client binary under test:
#   load   the server's load takes DELAY s; the reply must arrive.
#   leave  the server is killed LEAVE_AFTER s into its load; the client must
#          give up within seconds of that, not at the end of its budget.
#   price  what a client CHARGES the server for a cold load when choosing a
#          route (#129's other half): a client that has heard no rate from it
#          prices it at the prior; after the server has timed its own (slowed,
#          PRICE_DELAY s) load, unloaded the model and gossiped its rate, a
#          fresh client prices it at that rate. Read from each client's
#          `pipeline candidate … cold_load_ms=` line.
# Run it once with the build under test as the client and once with a release
# that predates the fix: the control must fail `load` at ~132 s and sit out
# `leave` for ~132 s, or this test cannot see the fix.
#
# Fully isolated: it re-runs itself inside a private network namespace
# (`unshare -rn`), so neither node can find a live node on this machine
# (loopback discovery is unconditional) or the internet. COPIES of one model,
# server only; auto-manage and the canonical heal off on both. Placement is
# pinned to the processor on both nodes.
#
# usage: examples/cold_load_test.sh <server-binary> <client-binary> [load|leave|price|both]
#        (`both` = load + leave; `price` needs a client with the cold-load price)
set -u
SERVER_BIN="${1:?server binary (must have SWARMLLM_FAULT_LOAD_DELAY_SECS)}"
CLIENT_BIN="${2:?client binary}"
ARMS="${3:-both}"
if [ -z "${COLD_LOAD_IN_NETNS:-}" ]; then
  exec env COLD_LOAD_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }

MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
DELAY="${DELAY:-200}"
LEAVE_AFTER="${LEAVE_AFTER:-20}"
SP=8895; CP=8896
export SWARMLLM_CANONICAL_UPLOADS=0 SWARMLLM_INFERENCE_GPU_LAYERS=0
ls "$SRC"/shard_*.bin >/dev/null 2>&1 || { echo "no parts of $MODEL in $SRC"; exit 2; }
SHARDS=$(ls "$SRC"/shard_*.bin | wc -l)
BASE=$(mktemp -d); echo "base=$BASE model=$MODEL parts=$SHARDS delay=${DELAY}s"
PIDS=""
kill_workers_under() { for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -q "model-worker.*$1" && kill -9 "$(basename "$p")" 2>/dev/null; done; }
cleanup() { for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; kill_workers_under "$BASE"; }
trap cleanup EXIT

config() { # dir bootstrap
  cat > "$1/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-cold-load-test"
enable_mdns = false
enable_upnp = false

[auto_manage]
enabled = false

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

start_server() { # arm [load delay, default DELAY]
  S="$BASE/$1-server"; mkdir -p "$S/models/$MODEL"
  cp "$SRC"/* "$S/models/$MODEL/"
  config "$S" ""
  SWARMLLM_FAULT_LOAD_DELAY_SECS=${2:-$DELAY} SWARMLLM_NODE_DATA_DIR="$S" "$SERVER_BIN" run -p $SP -v >> "$S/log" 2>&1 &
  SPID=$!; PIDS="$PIDS $SPID"
  up "$S" $SP || exit 1
  local id=""
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$S/log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/tcp/$((SP + 10))" "$S/log" && break
    sleep 1
  done
  [ -n "$id" ] || { echo "server never logged its peer id"; exit 1; }
  ADDR="/ip4/127.0.0.1/tcp/$((SP + 10))/p2p/$id"
  echo "server pid=$SPID addr=$ADDR"
}

start_client() { # arm
  C="$BASE/$1-client"; mkdir -p "$C"
  config "$C" "\"$ADDR\""
  SWARMLLM_NODE_DATA_DIR="$C" "$CLIENT_BIN" run -p $CP -v >> "$C/log" 2>&1 &
  CPID=$!; PIDS="$PIDS $CPID"
  up "$C" $CP || exit 1
  CK=$(cat "$C/api_key")
  echo -n "waiting for the client to learn the server holds $MODEL"
  for _ in $(seq 1 60); do
    if curl -s -m 5 -H "Authorization: Bearer $CK" "localhost:$CP/api/admin/models" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); ms=d if isinstance(d,list) else d.get('models',[])
m=[x for x in ms if x.get('id')=='$MODEL']
sh=m[0].get('shards',[]) if m else []
sys.exit(0 if len(sh)==$SHARDS and all((s.get('holders') or 0)>=1 for s in sh) else 1)" 2>/dev/null; then
      echo " — known"; return 0
    fi
    echo -n "."; sleep 5
  done
  echo; echo "the client never learned the model"; exit 1
}

ask() { # label -> prints "<http> <wall>"
  local t0 t1 code
  t0=$(date +%s.%N)
  code=$(curl -s -m 900 -o "$BASE/$1.json" -w '%{http_code}' -H "Authorization: Bearer $CK" -H 'Content-Type: application/json' \
    "localhost:$CP/v1/chat/completions" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0}')
  t1=$(date +%s.%N)
  echo "$code $(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')"
}

reply_of() { # label
  python3 -c "
import json
d=json.load(open('$BASE/$1.json'))
c=(d.get('choices') or [{}])[0].get('message',{}).get('content')
print(repr(c if c is not None else d.get('error',{}).get('message','')))" 2>/dev/null
}

arm_load() {
  echo "=== ARM load: the server's load takes ${DELAY}s ==="
  start_server load; start_client load
  read -r CODE WALL < <(ask load)
  echo "http=$CODE wall=${WALL}s reply=$(reply_of load)"
  echo "--- client"; grep -aE "fast path: request sent|timed out waiting for token|disconnected before its first token|execute_request completed|inference FAILED" "$C/log" | cut -c1-260 | tail -6
  echo "--- server"; grep -aE "FAULT INJECTION|model-worker: Model loaded|handling RemoteGenerateRequest" "$S/log" | cut -c1-200 | tail -4
  LOAD_RESULT="http=$CODE wall=${WALL}s"
  kill -9 "$CPID" "$SPID" 2>/dev/null; kill_workers_under "$BASE"; sleep 2
}

arm_leave() {
  echo "=== ARM leave: the server is killed ${LEAVE_AFTER}s into a ${DELAY}s load ==="
  start_server leave; start_client leave
  ( sleep "$LEAVE_AFTER"; kill -9 "$SPID" 2>/dev/null; kill_workers_under "$S"; echo "server killed at +${LEAVE_AFTER}s" ) &
  local killer=$!
  read -r CODE WALL < <(ask leave)
  # Only the killer: a bare `wait` would also wait on the client daemon.
  wait "$killer"
  echo "http=$CODE wall=${WALL}s reply=$(reply_of leave)"
  echo "--- client"; grep -aE "fast path: request sent|timed out waiting for token|disconnected before its first token|inference FAILED" "$C/log" | cut -c1-260 | tail -5
  LEAVE_RESULT="http=$CODE wall=${WALL}s"
  kill -9 "$CPID" 2>/dev/null; kill_workers_under "$BASE"; sleep 2
}

# The cold-load price a client's planner put on the server, from its own log.
cold_price_of_server() { # client log
  grep -a "DIAG: pipeline candidate node=${SNODE}" "$1" | tail -1 | grep -oE "cold_load_ms=[0-9.]+" | cut -d= -f2
}

arm_price() {
  local d=${PRICE_DELAY:-20}
  echo "=== ARM price: what a client charges the server for loading (its load slowed to ${d}s) ==="
  start_server price "$d"
  SK=$(cat "$S/api_key")
  SNODE=$(curl -s -m 5 -H "Authorization: Bearer $SK" "localhost:$SP/v1/status" | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])')
  GIB=$(python3 -c "import json; print(json.load(open('$S/models/$MODEL/manifest.json'))['total_size_bytes']/2**30)")
  start_client price-a
  read -r CODE WALL < <(ask price-a)
  local a; a=$(cold_price_of_server "$C/log")
  echo "client A (heard no rate): http=$CODE wall=${WALL}s, priced the server's cold load at ${a:-?} ms (prior 10000 ms/GiB x ${GIB} GiB)"
  local rate; rate=$(grep -a "DIAG: model load timed" "$S/log" | tail -1 | grep -oE "ms_per_gib=Some\([0-9]+\)" | grep -oE "[0-9]+")
  echo "server timed its own load: ${rate:-?} ms/GiB"
  curl -s -m 10 -X POST -H "Authorization: Bearer $SK" "localhost:$SP/api/admin/models/$MODEL/unload" >/dev/null
  kill -9 "$CPID" 2>/dev/null
  echo "server unloaded the model; waiting 40 s for its next capability broadcast"; sleep 40
  start_client price-b
  read -r CODE WALL < <(ask price-b)
  local b; b=$(cold_price_of_server "$C/log")
  echo "client B (heard the rate): http=$CODE wall=${WALL}s, priced the server's cold load at ${b:-?} ms (expect ${rate:-?} x ${GIB})"
  PRICE_RESULT=$(python3 -c "
a=float('${a:-nan}'); b=float('${b:-nan}'); r=float('${rate:-nan}'); g=$GIB
ok_a=abs(a-10000*g)<1; ok_b=abs(b-r*g)<0.01*r*g+1
print(f'prior {a:.0f}~{10000*g:.0f} {\"ok\" if ok_a else \"WRONG\"}; measured {b:.0f}~{r*g:.0f} {\"ok\" if ok_b else \"WRONG\"}')")
  echo "$PRICE_RESULT"
  kill -9 "$CPID" "$SPID" 2>/dev/null; kill_workers_under "$BASE"; sleep 2
}

LOAD_RESULT=skipped; LEAVE_RESULT=skipped; PRICE_RESULT=skipped
case "$ARMS" in
  load) arm_load ;;
  leave) arm_leave ;;
  price) arm_price ;;
  *) arm_load; arm_leave ;;
esac
echo
echo "RESULT client=$CLIENT_BIN load: $LOAD_RESULT ; leave: $LEAVE_RESULT ; price: $PRICE_RESULT (fix: load 200 at ~${DELAY}s; leave gives up within ~$((LEAVE_AFTER + 10))s)"
echo "logs kept under $BASE"
