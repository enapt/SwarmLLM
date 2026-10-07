#!/bin/bash
# A tester's report, 2026-10-07 (v0.3.229): a coordinator handed a whole model to
# a processor-only peer whose memory cap could never hold it (6688 MB needed,
# 5200 MB allowed), and the peer refused it 8.8 s later from another continent.
# Since v0.3.230 each node advertises the most it could EVER give a model
# (`NodeCapability::model_memory_ceiling_mb`) and the planner never plans past
# it, weighed with the peer's own admission arithmetic.
#
# Two nodes in a private network namespace; the server holds a model on its
# processor, its `resources.max_ram_mb` set around the model's real footprint —
# the server's own figure, read from its admission's refusal when asked locally
# under a tiny cap ("needs about N MB"):
#   below  cap = 80% of N (the report was 78%): the client must refuse ITSELF,
#          at once, "Not enough memory in the swarm", and the server must never
#          be asked;
#   above  cap = N + 64 MB: the client must be served — the ceiling excludes
#          nothing the server's admission would accept.
# Run again with a client that predates the fix: `below` is then refused by the
# SERVER after a round trip, or this test cannot see the fix. The server must
# have the fix (it is what advertises the ceiling).
#
# Fully isolated (`unshare -rn`, like cold_load_test.sh), so it can run beside a
# live node. COPIES of one model, server only; placement pinned to the processor.
#
# usage: examples/ceiling_test.sh <server-binary> <client-binary>
set -u
SERVER_BIN="${1:?server binary (must advertise model_memory_ceiling_mb)}"
CLIENT_BIN="${2:?client binary}"
if [ -z "${CEILING_IN_NETNS:-}" ]; then
  exec env CEILING_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }

MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
SP=8895; CP=8896
export SWARMLLM_CANONICAL_UPLOADS=0 SWARMLLM_INFERENCE_GPU_LAYERS=0
ls "$SRC"/shard_*.bin >/dev/null 2>&1 || { echo "no parts of $MODEL in $SRC"; exit 2; }
SHARDS=$(ls "$SRC"/shard_*.bin | wc -l)
BASE=$(mktemp -d); echo "base=$BASE model=$MODEL parts=$SHARDS"
PIDS=""
kill_workers_under() { for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null < "$p/cmdline" | grep -q "model-worker.*$1" && kill -9 "$(basename "$p")" 2>/dev/null; done; }
cleanup() { for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; kill_workers_under "$BASE"; }
trap cleanup EXIT

config() { # dir bootstrap max_ram_mb
  cat > "$1/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-ceiling-test"
enable_mdns = false
enable_upnp = false

[auto_manage]
enabled = false

[resources]
max_ram_mb = $3

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

start_server() { # arm cap_mb
  S="$BASE/$1-server"; mkdir -p "$S/models/$MODEL"
  cp "$SRC"/* "$S/models/$MODEL/"
  config "$S" "" "$2"
  SWARMLLM_NODE_DATA_DIR="$S" "$SERVER_BIN" run -p $SP -v >> "$S/log" 2>&1 &
  SPID=$!; PIDS="$PIDS $SPID"
  up "$S" $SP || exit 1
  SK=$(cat "$S/api_key")
  local id=""
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$S/log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/tcp/$((SP + 10))" "$S/log" && break
    sleep 1
  done
  [ -n "$id" ] || { echo "server never logged its peer id"; exit 1; }
  ADDR="/ip4/127.0.0.1/tcp/$((SP + 10))/p2p/$id"
}

start_client() { # arm
  # The model's HEADER and nothing else: the planner weighs a peer's ceiling
  # with the model's own geometry, which a coordinator on the internet fetches
  # from HuggingFace before planning (`ensure_model_geometry`) and keeps for
  # every request after. This namespace has no internet, so it is given the
  # header a connected node would have — no part of the model.
  C="$BASE/$1-client"; mkdir -p "$C/models/$MODEL"
  cp "$SRC/gguf_header.bin" "$C/models/$MODEL/"
  config "$C" "\"$ADDR\"" 0
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
      echo " — known"; sleep 10; return 0
    fi
    echo -n "."; sleep 5
  done
  echo; echo "the client never learned the model"; exit 1
}

ask() { # port key label -> prints "<http> <wall>"
  local t0 t1 code
  t0=$(date +%s.%N)
  code=$(curl -s -m 600 -o "$BASE/$3.json" -w '%{http_code}' -H "Authorization: Bearer $2" -H 'Content-Type: application/json' \
    "localhost:$1/v1/chat/completions" \
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

stop_all() { for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; PIDS=""; kill_workers_under "$BASE"; sleep 2; }

echo "=== the model's footprint, by the server's own admission (cap 64 MB, asked locally) ==="
start_server footprint 64
read -r CODE WALL < <(ask $SP "$SK" footprint)
MSG=$(reply_of footprint)
N=$(echo "$MSG" | grep -oE "needs about [0-9]+ MB" | head -1 | grep -oE "[0-9]+")
echo "http=$CODE: $MSG" | cut -c1-300
[ -n "$N" ] || { echo "INVALID: the server did not name its footprint"; exit 1; }
echo "footprint N = $N MB"
stop_all

arm() { # name cap_mb
  echo "=== ARM $1: server cap $2 MB against a footprint of $N MB ==="
  start_server "$1" "$2"; start_client "$1"
  read -r CODE WALL < <(ask $CP "$CK" "$1")
  local asked; asked=$(grep -a -c "handling RemoteGenerateRequest\|Not enough system memory for this model right now" "$S/log")
  echo "http=$CODE wall=${WALL}s reply=$(reply_of "$1" | cut -c1-200)"
  echo "server asked: $asked time(s)"
  echo "--- client"; grep -a -oE "pipeline candidate node=[0-9a-f]+ .*max_hostable_layers_at_ceiling=[A-Za-z0-9()]+" "$C/log" | sed -E 's/ ranges=.*(max_hostable_layers_at_ceiling)/ … \1/' | tail -1
  grep -aE "Not enough memory in the swarm|inference FAILED" "$C/log" | cut -c1-220 | tail -2
  eval "${1^^}_RESULT=\"http=$CODE wall=${WALL}s server_asked=$asked\""
  stop_all
}

arm below $(( N * 80 / 100 ))
arm above $(( N + 64 ))
echo
echo "RESULT client=$CLIENT_BIN footprint=${N}MB below: $BELOW_RESULT ; above: $ABOVE_RESULT (fix: below 503 at once with server_asked=0; above 200)"
