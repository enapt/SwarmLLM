#!/bin/bash
# Smoke-test a SwarmLLM binary end to end, on an isolated node.
#
# Tests pass and the daemon still has to start, serve and apply settings —
# those are different questions, and refactors break the second without
# touching the first. This boots a throwaway node on its own port and data
# directory, so it never touches a running one.
#
#   examples/smoke_test.sh                              # working-tree release build
#   examples/smoke_test.sh ~/.local/bin/swarmllm        # an installed binary
#   examples/smoke_test.sh ./swarmllm-linux-x86_64 8809 # a downloaded artifact
#
# Set SWARM_SMOKE_MODEL to a model this machine holds shards for; the
# inference checks are skipped if it is unset and no default is present.
#
# The checks describe the CURRENT tree's contract, so running this against an
# older binary can legitimately fail one — an already-released build predates
# whatever landed since. The check names say what is being asserted; read the
# failure before assuming a regression.
set -u

BIN="${1:-./target/release/swarmllm}"
PORT="${2:-8807}"
MODEL="${SWARM_SMOKE_MODEL:-llama-3.2-3b-instruct-q4-k-m}"
MODELS_DIR="${SWARM_SMOKE_MODELS_DIR:-$HOME/.local/share/swarmllm/models}"

[ -x "$BIN" ] || { echo "not executable: $BIN"; exit 1; }

D=$(mktemp -d)
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; rm -rf "$D"; }
trap cleanup EXIT

[ -d "$MODELS_DIR" ] && ln -s "$MODELS_DIR" "$D/models"

echo "smoke: $("$BIN" --version) on port $PORT"
SWARMLLM_NODE_DATA_DIR="$D" "$BIN" run -p "$PORT" > "$D/node.log" 2>&1 &
PID=$!

fails=0
check() { # name, condition-result
  if [ "$2" = "0" ]; then printf '  %-34s OK\n' "$1"; else printf '  %-34s FAIL\n' "$1"; fails=$((fails+1)); fi
}

for _ in $(seq 1 90); do
  [ -f "$D/api_key" ] && curl -s -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
K=$(cat "$D/api_key" 2>/dev/null || true)
if [ -z "$K" ]; then
  echo "  node never came up — last 20 log lines:"
  tail -20 "$D/node.log"
  exit 1
fi
check "starts and answers /health" 0

curl -s -m 8 -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/stats" >/dev/null 2>&1
check "admin API responds" $?

# A setting must reach the running node, not just the config file.
curl -s -m 8 -X PUT -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  "http://localhost:$PORT/api/admin/config" -d '{"max_disk_mb": 77777}' >/dev/null
sleep 1
AFTER=$(curl -s -m 8 -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/storage/breakdown" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("total_mb"))' 2>/dev/null)
[ "$AFTER" = "77777" ]; check "settings apply without restart" $?

curl -s -m 8 -X POST -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/config/reload" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);exit(0 if "applied" in d and "restart_required" in d else 1)' 2>/dev/null
check "reload separates applied/restart" $?

if curl -s -m 8 -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/models" \
     | grep -q "\"$MODEL\""; then
  R=$(curl -s -m 300 -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    "http://localhost:$PORT/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the single word: fig\"}],\"max_tokens\":10,\"stream\":false}")
  echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin)
exit(0 if 'choices' in d and d['choices'][0]['message']['content'].strip() else 1)
" 2>/dev/null
  check "inference returns non-empty text" $?

  N=$(curl -s -m 300 -N -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    "http://localhost:$PORT/v1/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the single word: plum\"}],\"max_tokens\":10,\"stream\":true}" \
    | grep -c "^data:")
  [ "${N:-0}" -gt 1 ]; check "streaming emits events" $?
else
  echo "  (inference checks skipped — $MODEL not present)"
fi

ERRS=$(grep -cE " ERROR " "$D/node.log" || true)
[ "${ERRS:-0}" -eq 0 ]; check "no errors logged at startup" $?
[ "${ERRS:-0}" -eq 0 ] || grep -E " ERROR " "$D/node.log" | head -5

echo
if [ "$fails" -eq 0 ]; then echo "smoke: all checks passed"; else echo "smoke: $fails check(s) FAILED"; fi
exit "$fails"
