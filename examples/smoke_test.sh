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

# That symlink points at the REAL models directory, so this throwaway node can
# delete the models of the node you actually run. Auto-manage prunes shards it
# considers over-replicated, and it is working as designed when it does — on
# 2026-08-10 a test node started exactly like this removed a live node's
# llama-3.2-3b shard ("Pruning over-replicated shard ... holders=4 target=2").
# Written before the daemon starts so there is no window in which the default
# (enabled) applies.
cat > "$D/config.toml" <<'TOML'
[auto_manage]
enabled = false
prune_enabled = false
TOML

echo "smoke: $("$BIN" --version) on port $PORT"
SWARMLLM_NODE_DATA_DIR="$D" "$BIN" run -p "$PORT" > "$D/node.log" 2>&1 &
PID=$!

fails=0
skipped=0
check() { # name, condition-result
  if [ "$2" = "0" ]; then printf '  %-34s OK\n' "$1"; else printf '  %-34s FAIL\n' "$1"; fails=$((fails+1)); fi
}
# A check that could not RUN is not a check that passed. Counted separately so
# the summary can never report "all checks passed" over work it never did — the
# same reporting bug that hid a corrupt shard for hours (gotcha #381).
skip() { printf '  %-34s SKIPPED (%s)\n' "$1" "$2"; skipped=$((skipped+1)); }

# The daemon we started must still be running. Without this, a node that dies
# mid-run is invisible: every later check just waits out its own curl timeout
# against a dead port — three inference checks at -m 300, plus the model wait,
# is twenty minutes of a script looking busy and testing nothing. Fail where the
# failure is, and print the log tail that says why.
alive() {
  if kill -0 "$PID" 2>/dev/null; then return 0; fi
  printf '  %-34s FAIL (the node exited)\n' "$1"
  fails=$((fails+1))
  echo "  --- last lines of node.log ---"
  tail -12 "$D/node.log" | sed 's/^/  /'
  return 1
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

# ...and a LATER change must not silently undo an earlier one. The Settings
# panel sends one section at a time, so a handler that rebuilds the config from
# the boot snapshot reverts every field the current request happens to omit —
# live and on disk. Changing a different setting and re-reading the first is the
# whole reproduction.
curl -s -m 8 -X PUT -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
  "http://localhost:$PORT/api/admin/config" -d '{"contribution": "moderate"}' >/dev/null
sleep 1
STILL=$(curl -s -m 8 -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/storage/breakdown" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("total_mb"))' 2>/dev/null)
[ "$STILL" = "77777" ]; check "one setting does not revert another" $?

curl -s -m 8 -X POST -H "Authorization: Bearer $K" "http://localhost:$PORT/api/admin/config/reload" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);exit(0 if "applied" in d and "restart_required" in d else 1)' 2>/dev/null
check "reload separates applied/restart" $?

# Wait for the model to be servable rather than probing once. A node with a
# large models directory is still scanning it when the checks above finish, so a
# single probe reported the model "not present" and quietly skipped every
# inference check — on the machine most likely to have a big models directory.
#
# Ask `/v1/models`, not `/api/admin/models`. The admin listing waits on the
# startup disk scan, which re-hashes every shard it finds: on a 15 GB models
# directory that ran past 60 s and skipped all three inference checks on both
# runs of the v0.3.129 verification, while `release_shapes.sh` — which asks
# `/v1/models` — found the model after 12 s and served it immediately. The
# client-facing listing is also the more meaningful oracle: it is what a caller
# would consult before sending the request these checks are about to send.
#
# The window is generous because the cost of being wrong is asymmetric. Waiting
# too long delays a release check; giving up too early reports a green run over
# inference that was never exercised, which is the fault this block already
# exists to prevent.
model_present=1
for _ in $(seq 1 120); do
  kill -0 "$PID" 2>/dev/null || break
  if curl -s -m 8 -H "Authorization: Bearer $K" "http://localhost:$PORT/v1/models" \
       | grep -q "$MODEL"; then
    model_present=0
    break
  fi
  sleep 2
done
alive "node still running" || { echo; echo "smoke: $fails check(s) FAILED"; exit "$fails"; }
if [ "$model_present" = "0" ]; then
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

  # The Anthropic surface is how Claude Code talks to a node, and it has its own
  # request translation, response assembly and streaming — a break here is
  # invisible to every check above.
  curl -s -m 300 -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    "http://localhost:$PORT/v1/messages" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the single word: pear\"}],\"max_tokens\":10}" \
    | python3 -c "
import sys,json
d=json.load(sys.stdin)
blocks=d.get('content') or []
text=''.join(b.get('text','') for b in blocks if isinstance(b,dict))
exit(0 if text.strip() else 1)
" 2>/dev/null
  check "Anthropic endpoint returns text" $?
else
  skip "chat completion returns text" "$MODEL not present"
  skip "streaming yields chunks" "$MODEL not present"
  skip "Anthropic endpoint returns text" "$MODEL not present"
fi

ERRS=$(grep -cE " ERROR " "$D/node.log" || true)
[ "${ERRS:-0}" -eq 0 ]; check "no errors logged at startup" $?
[ "${ERRS:-0}" -eq 0 ] || grep -E " ERROR " "$D/node.log" | head -5

echo
if [ "$fails" -ne 0 ]; then
  echo "smoke: $fails check(s) FAILED${skipped:+, $skipped skipped}"
elif [ "$skipped" -ne 0 ]; then
  # Deliberately NOT "all checks passed": some never ran, and the ones that get
  # skipped here are the inference checks — the only ones that prove the binary
  # can actually answer a request.
  echo "smoke: $skipped check(s) COULD NOT RUN; the rest passed"
else
  echo "smoke: all checks passed"
fi
exit "$fails"
