#!/bin/bash
# #436 end-to-end: a segment's peer is killed mid-request; the coordinator must
# Passed 2026-09-02 against the v0.3.147 fix (clean 503 in 10.4 s); see
# docs/DIAGNOSTICS.md § Benchmarks and gotcha #436 for what it verifies.
# fail the forward within seconds of the failed re-dial (DIAG: peer departed),
# not wait out the segment deadline. Fully isolated: private gossip id, no
# bootstrap, no mDNS, partial COPIES of one model (never a symlink to the real
# dir, and auto-manage off on both nodes).
set -u
BIN="${1:-./target/debug/swarmllm}"
MODEL=llama-3.2-3b-instruct-q4-k-m
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
SP=8894; CP=8893
BASE=$(mktemp -d); S="$BASE/server"; C="$BASE/client"
mkdir -p "$S/models/$MODEL" "$C/models/$MODEL"
echo "base=$BASE"
# Client holds the ENDS (0,3) + tied output; server holds the MIDDLE (1,2).
# No node holds everything, so the request MUST become a pipeline with the
# middle segment on the server.
for f in gguf_header.bin manifest.json shard_000.bin shard_003.bin tied_output_weight.bin; do cp "$SRC/$f" "$C/models/$MODEL/"; done
for f in gguf_header.bin manifest.json shard_001.bin shard_002.bin; do cp "$SRC/$f" "$S/models/$MODEL/"; done
for d in "$S" "$C"; do cat > "$d/config.toml" <<CFG
[network]
bootstrap_peers = []
disable_default_bootstrap = true
gossip_network_id = "swarmllm-departed-test"
enable_mdns = false

[auto_manage]
enabled = false
CFG
done
cleanup() { kill -9 ${SPID:-} ${CPID:-} 2>/dev/null; }
trap cleanup EXIT

SWARMLLM_NODE_DATA_DIR="$S" "$BIN" run -p $SP -v > "$S/log" 2>&1 & SPID=$!
for _ in $(seq 1 90); do [ -f "$S/api_key" ] && curl -s -m 3 "http://localhost:$SP/health" >/dev/null 2>&1 && break; sleep 2; done
SK=$(cat "$S/api_key" 2>/dev/null || true); [ -z "$SK" ] && { echo "server never came up"; tail -20 "$S/log"; exit 1; }
ADDR=$(curl -s -m 8 -H "Authorization: Bearer $SK" "http://localhost:$SP/api/admin/diagnostics?full=true" | grep -oE "/ip4/[0-9.]+/tcp/[0-9]+/p2p/[A-Za-z0-9]+" | grep -v p2p-circuit | grep -v "10\.255\.255\.254" | head -1)
[ -z "$ADDR" ] && { echo "no server addr"; exit 1; }
SNODE=$(curl -s -m 8 -H "Authorization: Bearer $SK" "http://localhost:$SP/api/admin/diagnostics" | grep -oE "^node: +[0-9a-f]+" | grep -oE "[0-9a-f]{16,}" | head -1)
echo "server node=${SNODE:0:16} addr=$ADDR pid=$SPID"

sed -i "s|bootstrap_peers = \[\]|bootstrap_peers = [\"$ADDR\"]|" "$C/config.toml"
SWARMLLM_NODE_DATA_DIR="$C" "$BIN" run -p $CP -v > "$C/log" 2>&1 & CPID=$!
for _ in $(seq 1 90); do [ -f "$C/api_key" ] && curl -s -m 3 "http://localhost:$CP/health" >/dev/null 2>&1 && break; sleep 2; done
CK=$(cat "$C/api_key" 2>/dev/null || true); [ -z "$CK" ] && { echo "client never came up"; tail -20 "$C/log"; exit 1; }

echo -n "waiting for full swarm coverage of $MODEL"
ok=0
for _ in $(seq 1 60); do
  if curl -s -m 5 -H "Authorization: Bearer $CK" "http://localhost:$CP/api/admin/models" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); ms=d if isinstance(d,list) else d.get('models',[])
m=[x for x in ms if x.get('id')=='$MODEL']
sh=m[0].get('shards',[]) if m else []
sys.exit(0 if len(sh)==4 and all((s.get('holders') or 0)>=1 for s in sh) else 1)" 2>/dev/null; then ok=1; echo " — covered"; break; fi
  echo -n "."; sleep 5
done
[ "$ok" = 1 ] || { echo; echo "coverage never appeared"; exit 1; }

# Long prompt -> the middle segment's prefill takes tens of seconds on CPU.
python3 - > "$BASE/req.json" <<'PY'
import json
words = ("the quick brown fox jumps over the lazy dog and then considers the harvest moon while counting stones by the river " * 120).strip()
print(json.dumps({"model":"llama-3.2-3b-instruct-q4-k-m","messages":[{"role":"user","content":"Summarise this in one sentence: "+words}],"max_tokens":32,"stream":False}))
PY
T0=$(date +%s.%N)
curl -s -m 600 -H "Authorization: Bearer $CK" -H 'Content-Type: application/json' "http://localhost:$CP/v1/chat/completions" -d @"$BASE/req.json" > "$BASE/reply.json" & RPID=$!

# Kill the server the moment the client is waiting on it for a remote segment.
sent=0
for _ in $(seq 1 240); do
  if grep -q "waiting for remote segment result" "$C/log"; then sent=1; break; fi
  kill -0 $RPID 2>/dev/null || break
  sleep 0.5
done
[ "$sent" = 1 ] || { echo "request never produced a remote segment forward"; cat "$BASE/reply.json" 2>/dev/null | head -3; grep -E "route|pipeline|remote" "$C/log" | tail -10; exit 1; }
sleep 2   # let the forward genuinely be in flight
KT=$(date +%s.%N)
kill -9 $SPID
echo "server killed at t+$(echo "$KT $T0" | awk '{printf "%.1f", $1-$2}')s"

wait $RPID
T1=$(date +%s.%N)
AFTER_KILL=$(echo "$T1 $KT" | awk '{printf "%.1f", $1-$2}')
echo "request ended ${AFTER_KILL}s after the kill; reply:"; head -c 400 "$BASE/reply.json"; echo
echo "--- departed-peer DIAG on the client:"
grep -n "peer departed with forwards outstanding" "$C/log" || echo "  (DIAG LINE ABSENT)"
echo "--- surrounding failure lines:"
grep -nE "Dial failed|connection closed|Peer departed|failover|TIMED OUT|failed with no standby" "$C/log" | tail -12
echo "VERDICT: fail-after-kill=${AFTER_KILL}s (fix expects <30; pre-fix waits the prefill deadline, minutes)"
cp "$C/log" "$BASE/client.log.keep" 2>/dev/null
