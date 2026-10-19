#!/bin/bash
# What an MCP client — Claude Code itself — hands its model when this node's
# `/mcp` answers, for each binary given (FUTURE_WORK #196: a tool that ran and
# failed is an `isError` result; a call no tool could run is a JSON-RPC error).
#
#   examples/mcp_client_rig.sh <binary> [<binary> ...]
#
# Each binary runs as ONE empty node in a private network namespace
# (`unshare -rn`): it can reach no other node and none can reach it. Claude Code
# needs the internet, so it runs OUTSIDE, and reaches the node through two
# `examples/unix_tcp_bridge.py` bridges (inside: a Unix socket to the node's
# port; outside: 127.0.0.1:$PORT to that socket).
#
# Per binary it prints the raw reply to three `tools/call`s — `chat` with a
# model that does not exist (`CALL`), `chat` with no model (`MISSING`) and a
# tool that does not exist (`UNKNOWN`) — then asks Claude Code (`claude -p`,
# Haiku, only this server, only `mcp__swarm__chat` allowed) to make the first
# call once and say what it was told, and prints the tool result it handed its
# model (`is_error`, the text) and its answer. A run costs one short Haiku
# session per binary on the signed-in plan.
#
# ✅ 2026-10-19: v0.3.244 answered CALL and MISSING with JSON-RPC -32602;
# Claude Code still showed its model the message (`is_error=True`, prefix
# dropped) but none of the advice. The #196 build answered both `isError: true`
# with the hint in the text and `_meta["com.swarmllm/error"]`; UNKNOWN stayed
# -32602 on both. The model's answer then named the dashboard's Models tab.
set -u
[ $# -ge 1 ] || { echo "usage: $0 <binary> [<binary> ...]"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$(mktemp -d /tmp/mcp_client_rig.XXXXXX)}
PORT=${PORT:-18990}
command -v claude > /dev/null || { echo "claude (Claude Code) is not on PATH"; exit 2; }
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use — set PORT"; exit 2; fi
mkdir -p "$OUT"
[ -f ~/swarmllm-gate-common/safety.sh ] && source ~/swarmllm-gate-common/safety.sh && safety_start mcp_client_rig "$OUT"
echo "out: $OUT; uptime $(cut -d' ' -f1 /proc/uptime) s; start $(date -u +%FT%TZ)"

PROMPT='Use the chat tool from the swarm MCP server exactly once, with model "no-such-model" and messages [{"role":"user","content":"hi"}]. Do not retry and do not call any other tool. Then answer in one or two sentences: what did the tool tell you, and what would you suggest next?'
CALL='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"chat","arguments":{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}}}'
MISSING='{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"chat","arguments":{"messages":[{"role":"user","content":"hi"}]}}}'
UNKNOWN='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"no_such_tool","arguments":{}}}'

# The node inside the namespace: loopback up, the socket bridge, the node; it
# stops when the stop file appears.
INNER='
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }
python3 -I "$1/unix_tcp_bridge.py" unix2tcp "$4" 8990 > "$3/bridge_in.log" 2>&1 & br=$!
SWARMLLM_NODE_DATA_DIR="$3" "$2" run -p 8990 > "$3/node.log" 2>&1 & node=$!
while [ ! -f "$3/stop" ] && kill -0 "$node" 2>/dev/null; do sleep 1; done
kill "$node"; for _ in $(seq 1 30); do kill -0 "$node" 2>/dev/null || break; sleep 1; done
kill -9 "$node" "$br" 2>/dev/null; wait "$node" "$br" 2>/dev/null; echo "inner done"'

n=0
for BIN in "$@"; do
  n=$((n + 1)); D=$OUT/$n; SOCK=$(mktemp -u /tmp/mcp_rig.XXXXXX.sock)
  [ -x "$BIN" ] || { echo "== $n: $BIN is not executable"; continue; }
  mkdir -p "$D"
  cat > "$D/config.toml" <<CFG
[network]
bootstrap_peers = []
disable_default_bootstrap = true
gossip_network_id = "swarmllm-mcp-client-rig"
enable_mdns = false
enable_upnp = false
relay_forwarding_auto = false

[updates]
mode = "off"

[ui]
open_browser_on_start = false
CFG
  unshare -rn bash -c "$INNER" inner "$HERE" "$BIN" "$D" "$SOCK" > "$D/inner.log" 2>&1 &
  inner=$!
  for _ in $(seq 1 60); do [ -S "$SOCK" ] && break; sleep 1; done
  python3 -I "$HERE/unix_tcp_bridge.py" tcp2unix "$PORT" "$SOCK" > "$D/bridge_out.log" 2>&1 &
  br=$!
  up=0
  for _ in $(seq 1 120); do
    [ -f "$D/api_key" ] && curl -s -m 3 "localhost:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  echo "== $n: $BIN ($("$BIN" --version 2>/dev/null)) node up=$up"
  if [ "$up" = 1 ]; then
    KEY=$(cat "$D/api_key")
    for q in CALL MISSING UNKNOWN; do
      curl -s -m 200 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' "localhost:$PORT/mcp" -d "${!q}" > "$D/raw_$q.json" 2>&1
      echo "-- $q: $(head -c 600 "$D/raw_$q.json")"
    done
    printf '{"mcpServers":{"swarm":{"type":"http","url":"http://127.0.0.1:%s/mcp","headers":{"Authorization":"Bearer %s"}}}}\n' \
      "$PORT" "$KEY" > "$D/mcp.json"
    # From $OUT, so this repository's own settings and hooks do not load.
    (cd "$OUT" && env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT timeout 300 claude -p "$PROMPT" \
      --mcp-config "$D/mcp.json" --strict-mcp-config --allowedTools mcp__swarm__chat \
      --model haiku --output-format stream-json --verbose > "$D/claude.jsonl" 2>&1)
    python3 -I - "$D/claude.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    content = (e.get("message") or {}).get("content")
    if e.get("type") == "user" and isinstance(content, list):
        for b in content:
            if b.get("type") == "tool_result" and b.get("tool_use_id") and not str(b.get("content")).startswith("[{'type': 'tool_reference'"):
                print(f"-- handed to the model: is_error={b.get('is_error')} {json.dumps(b.get('content'))[:600]}")
    if e.get("type") == "result":
        print(f"-- its answer: {(e.get('result') or '')[:500]}")
PY
  fi
  touch "$D/stop"
  kill "$br" 2>/dev/null
  wait "$inner" "$br" 2>/dev/null
  rm -f "$SOCK"
done
declare -F safety_stop > /dev/null && safety_stop
echo "end $(date -u +%FT%TZ)"
