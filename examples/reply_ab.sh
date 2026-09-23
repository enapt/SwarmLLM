#!/bin/bash
# Greedy replies from one binary, one JSON line per (model, prompt).
#
# usage: examples/reply_ab.sh <binary> <port> <out.jsonl> [models...]
#   env PROMPTS="first|second"   prompt list, '|'-separated (default: two)
#   env EXTRA_TOML='[inference]\ngpu_layers = 0'   appended to the node config
#
# Run it for two binaries on the same box and diff the outputs. It found #93 at
# the v0.3.201 gate (a split reply decoded from an empty cache), which no other
# check could see: conformance pins `gpu_layers = 0` and never splits.
#
# ⚠ BYTE-IDENTICAL ACROSS RELEASES PROVES NO REGRESSION, NEVER CORRECTNESS.
# GLM-4 wrote broken code in every release up to 0.3.201 and this compared
# clean each time, because the previous release was broken identically (#96).
# For a family whose prompt or forward pass a release deliberately changes,
# judge the new reply against an INDEPENDENT implementation instead:
# llama-cpp-python on the same GGUF (`create_chat_completion`, temperature 0),
# and `examples/tokenizer_reference.py` for the prompt itself.
#
# Isolation as in the release shapes harness: auto-manage off (the node shares
# the real models dir), private gossip id. ⚠ A private gossip id is NOT
# isolation — the node still routes through the live node on this box, so stop
# the live node for replies that must run whole-local, and keep it up only when
# you WANT a split (memory/release_gate.md).
BIN="$1"; PORT="$2"; OUT="$3"; shift 3
MODELS="${*:-llama-3.2-3b-instruct-q4-k-m qwen3-1.7b-q8-0 phi-3.5-mini-instruct.q4-k-m microsoft-phi-4-mini-instruct-q4-k-m qwen2.5-coder-7b-instruct-q4-k-m gemma-2-2b-it-q4-k-m tinyllama-1.1b-chat-v1.0.q4-k-m thudm-glm-4-9b-0414-q4-k-m mistral-7b-instruct-v0.3-q4-k-m}"
MODELS_DIR="$HOME/.local/share/swarmllm/models"
[ -x "$BIN" ] || { echo "not executable: $BIN"; exit 2; }

D=$(mktemp -d)
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; sleep 2; rm -f "$D/models"; cp "$D/node.log" "${OUT%.jsonl}.node.log" 2>/dev/null; rm -rf "$D"; }
trap cleanup EXIT
ln -s "$MODELS_DIR" "$D/models"
cat > "$D/config.toml" <<'TOML'
[auto_manage]
enabled = false
prune_enabled = false

[network]
bootstrap_peers = []
disable_default_bootstrap = true
enable_mdns = false
gossip_network_id = "swarmllm-reply-ab"

[ui]
open_browser_on_start = false
TOML
[ -n "${EXTRA_TOML:-}" ] && printf '\n%b\n' "$EXTRA_TOML" >> "$D/config.toml"

echo "reply_ab: $("$BIN" --version) on port $PORT"
SWARMLLM_NODE_DATA_DIR="$D" "$BIN" run -p "$PORT" > "$D/node.log" 2>&1 &
PID=$!
for _ in $(seq 1 90); do
  [ -f "$D/api_key" ] && curl -s -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
K=$(cat "$D/api_key" 2>/dev/null) || { echo "node did not start"; exit 1; }
: > "$OUT"
for M in $MODELS; do
  IFS="|" read -r -a PLIST <<< "${PROMPTS:-What is the capital of France? Answer in one sentence.|Write a short Python function that returns the factorial of n.}"
  for P in "${PLIST[@]}"; do
    body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"max_tokens":96,"temperature":0,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$M" "$P")
    curl -s -m 900 -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
         -X POST "http://localhost:$PORT/v1/chat/completions" -d "$body" \
      | python3 -c 'import json,sys
m,p=sys.argv[1],sys.argv[2]
raw=sys.stdin.read()
try:
    d=json.loads(raw); t=d["choices"][0]["message"].get("content") or ""
    r=d["choices"][0]["message"].get("reasoning_content") or ""
    print(json.dumps({"model":m,"prompt":p[:24],"content":t,"reasoning":r,"tokens":d.get("usage",{}).get("completion_tokens")}))
except Exception as e:
    print(json.dumps({"model":m,"prompt":p[:24],"error":str(e),"raw":raw[:600]}))' "$M" "$P" >> "$OUT"
  done
  curl -s -m 60 -X POST -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/models/$M/unload" >/dev/null 2>&1
  echo "  $M done"
done
echo "reply_ab: finished, $(wc -l < "$OUT") lines"
