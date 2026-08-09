#!/bin/bash
# Two nodes, isolated from the public swarm, on ports that cannot collide with
# a real one. One holds the models, the other holds nothing — so a request to
# the empty node MUST be served by its peer, exercising the cross-node path
# under controlled conditions.
#
# Why this exists: the unit tests cover logic and the public swarm covers
# reality, but neither lets you say "this request went to that node and came
# back correct". Without the isolation below, the scheduler picks whichever peer
# it likes and the test proves nothing about the code you just changed.
#
#   examples/two_node_test.sh                     # working-tree release build
#   examples/two_node_test.sh ./swarmllm-linux-x86_64
#
# SWARM_TWONODE_MODEL selects the model; it must be one this machine holds.
#
# EXPECT THIS TO FAIL ON A SINGLE MULTI-INTERFACE HOST. WSL2 and anything with a
# Docker bridge advertise several addresses, and libp2p will sometimes route the
# send to a stale half-open connection and drop it — the request then fails with
# "peer never acknowledged" and the server log shows it never arrived. That is
# documented in docs/FUTURE_WORK.md § "Connection churn on multi-interface
# hosts", reproduces identically on released binaries, and is detected and named
# below rather than reported as a fault in the build under test. It succeeds
# intermittently here and reliably across two real machines, which is where the
# cross-node path should be validated.
set -u

BIN="${1:-./target/release/swarmllm}"
MODEL="${SWARM_TWONODE_MODEL:-llama-3.2-3b-instruct-q4-k-m}"
MODELS_DIR="${SWARM_TWONODE_MODELS_DIR:-$HOME/.local/share/swarmllm/models}"
SERVER_PORT="${SWARM_TWONODE_SERVER_PORT:-8894}"
CLIENT_PORT="${SWARM_TWONODE_CLIENT_PORT:-8893}"

[ -x "$BIN" ] || { echo "not executable: $BIN"; exit 1; }
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR"; exit 1; }

BASE=$(mktemp -d)
S="$BASE/server"; C="$BASE/client"
mkdir -p "$S" "$C/models"
ln -s "$MODELS_DIR" "$S/models"

# Kill only what this script started. A broad `pkill swarmllm` takes down any
# other node on the machine, production included (gotcha #283).
cleanup() { kill ${SPID:-} ${CPID:-} 2>/dev/null; [ -n "${SWARM_TWONODE_KEEP:-}" ] || rm -rf "$BASE"; [ -n "${SWARM_TWONODE_KEEP:-}" ] && echo "  logs kept in $BASE"; }
trap cleanup EXIT

# A private gossip id is NOT isolation on its own: the DHT and mDNS still find
# whatever else is on the machine and the LAN, and the scheduler then routes to
# it. An earlier version of this script sent its request to a peer on the far
# side of the internet and "passed". So: mDNS off, no public bootstrap, and the
# client is pointed at the server explicitly once the server has an address.
cat > "$S/config.toml" <<CFG
[network]
bootstrap_peers = []
disable_default_bootstrap = true
gossip_network_id = "swarmllm-twonode-test"
enable_mdns = false

[auto_manage]
enabled = false
CFG

echo "two-node: $("$BIN" --version), server=$SERVER_PORT client=$CLIENT_PORT"
SWARMLLM_NODE_DATA_DIR="$S" "$BIN" run -p "$SERVER_PORT" -v > "$S/log" 2>&1 & SPID=$!

for _ in $(seq 1 90); do
  [ -f "$S/api_key" ] && curl -s -m 3 "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
SK=$(cat "$S/api_key" 2>/dev/null || true)
[ -z "$SK" ] && { echo "  server never came up"; tail -20 "$S/log"; exit 1; }

# Any direct (non-relayed) address the server publishes. Loopback is filtered
# out of that list on purpose, so take the LAN one — same machine either way.
SERVER_ADDRS=$(curl -s -m 8 -H "Authorization: Bearer $SK" \
  "http://localhost:$SERVER_PORT/api/admin/diagnostics" \
  | grep -oE "/ip4/[0-9.]+/tcp/[0-9]+/p2p/[A-Za-z0-9]+" | grep -v "p2p-circuit")
# 10.255.255.254 is WSL2's NAT gateway — reachable, and the documented source of
# the connection churn that loses sends between two nodes on one host. Prefer a
# real LAN address when the node has one.
SERVER_ADDR=$(echo "$SERVER_ADDRS" | grep -v "10\.255\.255\.254" | head -1)
[ -z "$SERVER_ADDR" ] && SERVER_ADDR=$(echo "$SERVER_ADDRS" | head -1)
if [ -z "$SERVER_ADDR" ]; then
  echo "  server never published a loopback address to dial"
  exit 1
fi
echo "  server at $SERVER_ADDR"

cat > "$C/config.toml" <<CFG
[network]
bootstrap_peers = ["$SERVER_ADDR"]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-twonode-test"
enable_mdns = false

[auto_manage]
enabled = false
CFG

SWARMLLM_NODE_DATA_DIR="$C" "$BIN" run -p "$CLIENT_PORT" -v > "$C/log" 2>&1 & CPID=$!
for _ in $(seq 1 90); do
  [ -f "$C/api_key" ] && curl -s -m 3 "http://localhost:$CLIENT_PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
CK=$(cat "$C/api_key" 2>/dev/null || true)
[ -z "$CK" ] && { echo "  nodes never came up"; tail -20 "$C/log"; exit 1; }

# Isolation cannot be asserted by peer count on a live LAN: other nodes dial IN,
# and no local setting prevents that. So verify the OUTCOME instead — that the
# request was answered by this script's server and not by something else the
# scheduler happened to like. An earlier version asserted neither and "passed"
# on a reply from a peer on the far side of the internet.
SERVER_NODE=$(curl -s -m 8 -H "Authorization: Bearer $SK" \
  "http://localhost:$SERVER_PORT/api/admin/diagnostics" \
  | grep -oE "^node: +[0-9a-f]+" | grep -oE "[0-9a-f]{16,}" | head -1)
[ -z "$SERVER_NODE" ] && { echo "  could not read the server's node id"; exit 1; }
echo "  server node: ${SERVER_NODE:0:16}"

echo -n "  waiting for the client to see $MODEL on its peer"
found=0
for _ in $(seq 1 60); do
  if curl -s -m 5 -H "Authorization: Bearer $CK" "http://localhost:$CLIENT_PORT/api/admin/models" 2>/dev/null \
     | python3 -c "
import sys,json
d=json.load(sys.stdin); ms=d if isinstance(d,list) else d.get('models',[])
m=[x for x in ms if x.get('id')=='$MODEL']
sh=m[0].get('shards',[]) if m else []
sys.exit(0 if sh and all((s.get('holders') or 0)>=1 for s in sh) else 1)
" 2>/dev/null; then found=1; echo " — visible"; break; fi
  echo -n "."; sleep 5
done
[ "$found" -eq 1 ] || { echo; echo "  peer never advertised the model"; exit 1; }

fails=0
echo
echo "  request from the node holding nothing:"
# Retried: on a host advertising several interfaces (WSL2's NAT gateway,
# link-local, a Docker bridge, the LAN address) libp2p can route the send to a
# stale half-open connection and drop it silently. That is a known same-host
# limitation, not a property of the code under test — see docs/FUTURE_WORK.md
# § "Connection churn on multi-interface hosts". It reproduces identically on
# released binaries, so treat it as environmental and try again.
for attempt in 1 2 3; do
  REPLY=$(curl -s -m 300 -H "Authorization: Bearer $CK" -H 'Content-Type: application/json' \
    "http://localhost:$CLIENT_PORT/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from six to ten.\"}],\"max_tokens\":32,\"stream\":false}")
  echo "$REPLY" | grep -q '"choices"' && break
  echo "    attempt $attempt: no reply, retrying"
  sleep 5
done
if echo "$REPLY" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if 'choices' in d and d['choices'][0]['message']['content'].strip():
    print('    reply:', repr(d['choices'][0]['message']['content'][:100]))
    print('    usage:', d.get('usage'))
else:
    print('    no reply:', str(d)[:220]); sys.exit(1)
"; then :; else
  fails=$((fails+1))
  if echo "$REPLY" | grep -q "never acknowledged" && ! grep -q "handling RemoteGenerateRequest" "$S/log"; then
    echo
    echo "    ^ the server never received the request at all. This is the known"
    echo "      same-host connection-churn case (docs/FUTURE_WORK.md § \"Connection"
    echo "      churn on multi-interface hosts\"), not a fault in the build under"
    echo "      test — it reproduces on released binaries too. Validate the"
    echo "      cross-node path on two real machines instead."
  fi
fi

echo "  served by:"
SERVED_BY=$(curl -s -m 8 -H "Authorization: Bearer $CK" "http://localhost:$CLIENT_PORT/api/admin/performance" \
  | python3 -c "
import sys,json
d=json.load(sys.stdin); r=(d.get('recent') or [{}])[0]
segs=[x.get('node_id','') for x in r.get('segments',[])]
print(segs[0] if segs else '')
" 2>/dev/null)
if [ -n "$SERVED_BY" ] && [ "$SERVED_BY" = "$SERVER_NODE" ]; then
  echo "    answered by this script's server (${SERVED_BY:0:16}) — the path under test"
elif [ -n "$SERVED_BY" ]; then
  echo "    answered by ${SERVED_BY:0:16}, NOT this script's server (${SERVER_NODE:0:16})"
  echo "    the result says nothing about the build under test"
  fails=$((fails+1))
else
  echo "    nothing served it"
  fails=$((fails+1))
fi

# The server numbers its content tokens and the done token carries the total;
# that total must equal what the client counted, or reassembly lost something.
echo "  token sequencing:"
SC=$(grep -o "streamed_count=[0-9]*" "$S/log" | tail -1 | cut -d= -f2)
CT=$(echo "$REPLY" | python3 -c "import sys,json;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',-1))" 2>/dev/null)
if [ -n "$SC" ] && [ "$SC" = "$CT" ]; then
  echo "    server streamed $SC, client received $CT — match"
else
  echo "    MISMATCH: server streamed '${SC:-?}', client counted '$CT'"
  fails=$((fails+1))
fi

echo
[ "$fails" -eq 0 ] && echo "two-node: all checks passed" || echo "two-node: $fails check(s) FAILED"
exit "$fails"
