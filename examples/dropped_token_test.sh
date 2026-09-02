#!/bin/bash
# Gotcha #438 end-to-end: one content token of a fast-path reply is LOST on the
# serving side (fault injection: SWARMLLM_FAULT_DROP_STREAM_TOKEN=5) and the
# reply must still arrive whole — the requester notices the hole, asks the
# peer to resend, and the peer answers from what it retained. The control arm
# runs the same loss with resends disabled on the requester
# (SWARMLLM_RESEND_TOKENS=0) and must show the old truncation: the reply stops
# at the hole and the requester "gave up on tokens that never arrived" after
# the 15 s straggler wait. Fully isolated: private gossip id, no bootstrap, no
# mDNS, a COPY of one model on the server only (never a symlink to the real
# dir, auto-manage off on both nodes).
#
# usage: examples/dropped_token_test.sh [binary]
set -u
BIN="${1:-./target/debug/swarmllm}"
MODEL=llama-3.2-3b-instruct-q4-k-m
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
SP=8895; CP=8896
DROP_AT=5
BASE=$(mktemp -d); S="$BASE/server"; C="$BASE/client"
mkdir -p "$S/models/$MODEL" "$C"
echo "base=$BASE"
# The server holds the WHOLE model; the client holds nothing, so its only
# route is the remote-generate fast path (single remote segment).
for f in "$SRC"/*; do cp "$f" "$S/models/$MODEL/"; done
for d in "$S" "$C"; do cat > "$d/config.toml" <<CFG
[network]
bootstrap_peers = []
disable_default_bootstrap = true
gossip_network_id = "swarmllm-dropped-token-test"
enable_mdns = false

[auto_manage]
enabled = false
CFG
done
PIDS=""
FOREIGN=""
cleanup() { for p in $PIDS; do kill -9 $p 2>/dev/null; done; for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q "model-worker.*$BASE" && kill -9 $(basename $p) 2>/dev/null; done; }
trap cleanup EXIT

start_server() {
  SWARMLLM_FAULT_DROP_STREAM_TOKEN=$DROP_AT SWARMLLM_NODE_DATA_DIR="$S" "$BIN" run -p $SP -v >> "$S/log" 2>&1 & SPID=$!; PIDS="$PIDS $SPID"
  for _ in $(seq 1 90); do [ -f "$S/api_key" ] && curl -s -m 3 "http://localhost:$SP/health" >/dev/null 2>&1 && break; sleep 2; done
  SK=$(cat "$S/api_key" 2>/dev/null || true); [ -z "$SK" ] && { echo "server never came up"; tail -20 "$S/log"; exit 1; }
  ADDR=$(curl -s -m 8 -H "Authorization: Bearer $SK" "http://localhost:$SP/api/admin/diagnostics?full=true" | grep -oE "/ip4/[0-9.]+/tcp/[0-9]+/p2p/[A-Za-z0-9]+" | grep -v p2p-circuit | grep -v "10\.255\.255\.254" | head -1)
  [ -z "$ADDR" ] && { echo "no server addr"; exit 1; }
  SNODE=$(curl -s -m 8 -H "Authorization: Bearer $SK" "http://localhost:$SP/api/admin/diagnostics" | grep -oE "^node: +[0-9a-f]+" | grep -oE "[0-9a-f]{16,}" | head -1)
  echo "server pid=$SPID node=${SNODE:0:16} addr=$ADDR (drops content token $DROP_AT of every reply, once)"
}

start_client() {   # $1 = extra env assignments
  sed -i "s|bootstrap_peers = \[.*\]|bootstrap_peers = [\"$ADDR\"]|" "$C/config.toml"
  env $1 SWARMLLM_NODE_DATA_DIR="$C" "$BIN" run -p $CP -v >> "$C/log" 2>&1 & CPID=$!; PIDS="$PIDS $CPID"
  for _ in $(seq 1 90); do [ -f "$C/api_key" ] && curl -s -m 3 "http://localhost:$CP/health" >/dev/null 2>&1 && break; sleep 2; done
  CK=$(cat "$C/api_key" 2>/dev/null || true); [ -z "$CK" ] && { echo "client never came up"; tail -20 "$C/log"; exit 1; }
  echo -n "waiting for the client to learn the server holds $MODEL"
  ok=0
  for _ in $(seq 1 60); do
    if curl -s -m 5 -H "Authorization: Bearer $CK" "http://localhost:$CP/api/admin/models" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); ms=d if isinstance(d,list) else d.get('models',[])
m=[x for x in ms if x.get('id')=='$MODEL']
sh=m[0].get('shards',[]) if m else []
sys.exit(0 if len(sh)==4 and all((s.get('holders') or 0)>=1 for s in sh) else 1)" 2>/dev/null; then ok=1; echo " — known"; break; fi
    echo -n "."; sleep 5
  done
  [ "$ok" = 1 ] || { echo; echo "the client never learned the model"; exit 1; }
}

run_request() {   # $1 = label; prints delivered tokens + wall
  local label=$1 t0 t1
  t0=$(date +%s.%N)
  curl -s -N -m 600 -H "Authorization: Bearer $CK" -H 'Content-Type: application/json' "http://localhost:$CP/v1/chat/completions" \
    -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Count from 1 to 60, separated by commas, digits only."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    > "$BASE/$label.sse"
  t1=$(date +%s.%N)
  python3 - "$BASE/$label.sse" <<'PY'
import sys, json
n=0; text=''; finish=None; usage=None; err=None
for line in open(sys.argv[1]):
    line=line.strip()
    if not line.startswith('data:'): continue
    d=line[5:].strip()
    if d=='[DONE]': break
    j=json.loads(d)
    if 'error' in j: err=j['error']; continue
    if j.get('usage'): usage=j['usage']
    for c in j.get('choices',[]):
        if c.get('delta',{}).get('content'): n+=1; text+=c['delta']['content']
        if c.get('finish_reason'): finish=c['finish_reason']
print(f"delivered_chunks={n} finish={finish} usage={usage} error={err}")
print(f"text={text[:80]!r}")
PY
  echo "wall=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')s"
  # Loopback discovery is unconditional, so a live node on this machine that
  # holds the model can be chosen instead of the test server — and then this
  # test measures nothing. Say so rather than failing an arm mysteriously.
  local routed
  routed=$(grep -a "DIAG: request complete" "$C/log" | tail -1 | grep -oE "nodes=[0-9a-f,]+" | cut -d= -f2)
  # The route line carries the 8-character short form of each node id.
  if [ -n "$routed" ] && [ -n "${SNODE:-}" ] && [ "${routed:0:8}" != "${SNODE:0:8}" ]; then
    echo "ROUTED TO A FOREIGN NODE (${routed:0:8}, expected ${SNODE:0:8}) — a live node on this machine holds the model; stop it or rerun when it is idle"
    FOREIGN=1
  fi
}

start_server
echo "=== ARM 1: fix (requester asks for resends) ==="
: > "$C/log"
start_client ""
run_request fix
echo "--- client: hole handling"
grep -nE "asking the peer to resend|gave up on tokens|tokens the peer sent did not all arrive" "$C/log" | tail -5 || true
echo "--- server: fault + resend"
grep -nE "FAULT INJECTION|resending tokens the coordinator|ResendTokens refused" "$S/log" | tail -5 || true
FIX_OK=0
if grep -q "asking the peer to resend" "$C/log" && grep -q "resending tokens the coordinator" "$S/log" && ! grep -q "gave up on tokens" "$C/log"; then FIX_OK=1; fi
kill -9 $CPID 2>/dev/null; sleep 1
cp "$C/log" "$BASE/client_fix.log" 2>/dev/null

echo "=== ARM 2: control (SWARMLLM_RESEND_TOKENS=0 on the requester — the old behaviour) ==="
rm -rf "$C"; mkdir -p "$C"; cat > "$C/config.toml" <<CFG
[network]
bootstrap_peers = ["$ADDR"]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-dropped-token-test"
enable_mdns = false

[auto_manage]
enabled = false
CFG
start_client "SWARMLLM_RESEND_TOKENS=0"
run_request control
echo "--- client: hole handling"
grep -nE "asking the peer to resend|gave up on tokens|tokens the peer sent did not all arrive" "$C/log" | tail -5 || true
CTRL_TRUNCATED=0
if grep -q "gave up on tokens" "$C/log" && ! grep -q "asking the peer to resend" "$C/log"; then CTRL_TRUNCATED=1; fi

echo
echo "VERDICT: fix arm whole-reply-after-a-lost-token=$FIX_OK ; control arm truncated-as-before=$CTRL_TRUNCATED (both must be 1)${FOREIGN:+ — INVALID: a request was routed to a foreign node}"
cp "$C/log" "$BASE/client_control.log" 2>/dev/null; cp "$S/log" "$BASE/server.log" 2>/dev/null
echo "logs kept: $BASE/client_fix.log $BASE/client_control.log $BASE/server.log"
[ "$FIX_OK" = 1 ] && [ "$CTRL_TRUNCATED" = 1 ]
