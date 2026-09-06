#!/bin/bash
# A SMALL node, reproduced locally.
#
# Every memory defect reported from the field since #452 — #454-#457, #461-#468
# — came from one 16 GB processor-only machine, and none of them reproduced on
# a developer box, which is why all of them shipped. Admission keys off
# `resources.max_ram_mb` rather than real RAM, so a small configured budget
# reproduces the whole class on any machine.
#
# Fully isolated: private gossip id, no bootstrap, no mDNS, a COPY of one model
# (never a symlink to the real dir — a test node's auto-manage would prune
# shared files), auto-manage off. Isolation is not cosmetic here: with peers
# reachable, prompt privacy turns a whole-model request into a two-layer
# boomerang, the budget never binds, and the test silently proves nothing.
#
# Checks, in order:
#   1. A model larger than the budget is refused, and the refusal SHOWS ITS
#      ARITHMETIC — weights, KV, overhead, the budget and the setting to change
#      (#448: a bare "no remaining budget" makes a competent reader invent a
#      mechanism).
#   2. With room, the same model loads and answers.
#   3. SIGKILL the worker — which is what an OS OOM-kill looks like, and the
#      exact case #461/#467 were written for — and the node serves the NEXT
#      request. Before that fix the dead worker kept its whole charge and the
#      node refused every model until restart.
#   4. Exactly one worker is charged afterwards: the release must not have run
#      twice, which would under-count real memory (the opposite failure).
set -u
BIN="${1:-./target/debug/swarmllm}"
MODEL=llama-3.2-3b-instruct-q4-k-m
SRC="$HOME/.local/share/swarmllm/models/$MODEL"
# A SECOND, smaller model, for the case the field report actually describes: a
# dead worker refusing a DIFFERENT model. Optional — that check is skipped if
# it is not on this machine.
MODEL2=qwen2.5-0.5b-instruct-fp16
SRC2="$HOME/.local/share/swarmllm/models/$MODEL2"
PORT=8866
# Sized from the model: too small to admit it, then comfortably large enough.
TIGHT_MB=2200
ROOMY_MB=4000

[ -x "$BIN" ] || { echo "FAIL: no binary at $BIN"; exit 1; }
[ -d "$SRC" ] || { echo "SKIP: $MODEL not present at $SRC"; exit 0; }

BASE=$(mktemp -d); D="$BASE/node"
mkdir -p "$D/models"
cp -r "$SRC" "$D/models/"
[ -d "$SRC2" ] && cp -r "$SRC2" "$D/models/"
echo "base=$BASE"

write_config() {  # $1 = budget in MB
  cat > "$D/config.toml" <<CFG
[resources]
max_ram_mb = $1

[auto_manage]
enabled = false

[inference]
gpu_layers = 0

[network]
bootstrap_peers = []
disable_default_bootstrap = true
gossip_network_id = "swarmllm-constrained-test"
enable_mdns = false
CFG
}

start_node() {
  rm -f "$D/node.log"
  SWARMLLM_NODE_DATA_DIR="$D" nohup setsid "$BIN" run -p $PORT -c "$D/config.toml" \
    > "$D/node.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -s -m 2 "http://localhost:$PORT/health/ready" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "FAIL: node did not become ready"; tail -20 "$D/node.log"; exit 1
}

stop_node() {
  for pid in $(pgrep -f swarmllm 2>/dev/null); do
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$D" && kill "$pid" 2>/dev/null
  done
  sleep 3
}
trap 'stop_node; rm -rf "$BASE"' EXIT

ask() {  # $1 = model id, defaults to the big one
  KEY=$(cat "$D/api_key")
  M="${1:-$MODEL}"
  timeout 500 curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -X POST "http://localhost:$PORT/v1/chat/completions" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8}"
}

fails=0
# A failing check must say what it actually saw, or it cannot be acted on.
check() {
  if [ "$2" = "1" ]; then echo "  ok   — $1"
  else
    echo "  FAIL — $1"
    [ -n "${3:-}" ] && echo "         got: $(echo "$3" | head -c 300)"
    fails=$((fails+1))
  fi
}

# ── 1. over budget: refused, with the arithmetic shown ────────────────────
echo "[1] a model larger than the budget (${TIGHT_MB} MB)"
write_config $TIGHT_MB
start_node
OUT=$(ask)
check "refused rather than served"        "$(echo "$OUT" | grep -qi 'error' && echo 1 || echo 0)"
check "names the weights"                 "$(echo "$OUT" | grep -qi 'MB of weights' && echo 1 || echo 0)"
check "names the KV cache"                "$(echo "$OUT" | grep -qi 'KV cache' && echo 1 || echo 0)"
check "names the budget and the setting"  "$(echo "$OUT" | grep -q 'max_ram_mb' && echo 1 || echo 0)"
echo "      $(echo "$OUT" | head -c 200)"
stop_node

# ── 2/3/4. room to load, then kill the worker ─────────────────────────────
echo "[2] the same model with ${ROOMY_MB} MB"
write_config $ROOMY_MB
start_node
OUT=$(ask)
check "answers"                           "$(echo "$OUT" | grep -q '"content"' && echo 1 || echo 0)"

echo "[3] SIGKILL the worker (what an OS OOM-kill looks like)"
WPID=$(for pid in $(pgrep -f 'model-worker' 2>/dev/null); do
         tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$D" && echo "$pid"
       done | head -1)
if [ -z "${WPID:-}" ]; then
  echo "  FAIL — no worker subprocess found to kill"; fails=$((fails+1))
else
  kill -9 "$WPID"; sleep 6
  OUT=$(ask)
  check "the node serves the NEXT request"  "$(echo "$OUT" | grep -q '"content"' && echo 1 || echo 0)" "$OUT"
  check "and did not refuse for memory"     "$(echo "$OUT" | grep -qi 'already in use' && echo 0 || echo 1)" "$OUT"
  check "the budget release is logged"      "$(grep -q 'memory budget released' "$D/node.log" && echo 1 || echo 0)"

  echo "[4] exactly one worker charged afterwards"
  KEY=$(cat "$D/api_key")
  N=$(curl -s -H "Authorization: Bearer $KEY" "http://localhost:$PORT/v1/status" \
      | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("workers",[])))' 2>/dev/null || echo "?")
  check "one live worker, not two or none (got $N)" "$([ "$N" = "1" ] && echo 1 || echo 0)" \
        "$(curl -s -H "Authorization: Bearer $KEY" "http://localhost:$PORT/v1/status" | head -c 300)"
fi

# ── 5. a dead worker must not refuse a DIFFERENT model ────────────────────
# This is the reported symptom — "a dead 14B refuses every model until
# restart" — and it is a different path from [3]. There, `get_or_spawn` looks
# the dead worker up under the SAME key and retires it; here nothing looks it
# up at all, and only the shared budget is consulted, so the release has to
# have come from the health tick. The budget is one figure for the whole node,
# which is why one corpse could refuse everything.
if [ -d "$SRC2" ]; then
  echo "[5] after a death, a DIFFERENT model still loads"
  stop_node; write_config $ROOMY_MB; start_node
  OUT=$(ask "$MODEL")
  check "the big model loads first" "$(echo "$OUT" | grep -q '"content"' && echo 1 || echo 0)" "$OUT"
  WPID2=$(for pid in $(pgrep -f 'model-worker' 2>/dev/null); do
            tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$D" && echo "$pid"
          done | head -1)
  if [ -n "${WPID2:-}" ]; then
    kill -9 "$WPID2"; sleep 8
    RELEASES_BEFORE=$(grep -c 'memory budget released' "$D/node.log")
    OUT=$(ask "$MODEL2")
    check "the OTHER model is served after that death" \
          "$(echo "$OUT" | grep -q '"content"' && echo 1 || echo 0)" "$OUT"
    # Assert the MECHANISM, because the outcome alone is ambiguous: on a
    # healthy node the same request also succeeds by RECLAIMING a live worker,
    # and at this budget the two models cannot both be resident anyway (each
    # charges ~3.1 GB against 4000). What distinguishes a leak is that a
    # corpse's charge cannot be reclaimed — `free_ram_for_admission` walks the
    # charge map and skips any entry whose worker is no longer in `workers` —
    # so if the dead worker's charge were still on the books nothing could
    # free it and this model would be refused. The release is the thing to
    # check for.
    check "the dead worker's charge was released, not merely reclaimed" \
          "$([ "$(grep -c 'memory budget released' "$D/node.log")" -gt "$RELEASES_BEFORE" ] && echo 1 || echo 0)" \
          "$(grep 'memory budget released' "$D/node.log" | tail -2)"
  else
    echo "  FAIL — no worker to kill for the cross-model case"; fails=$((fails+1))
  fi
else
  echo "[5] SKIPPED — $MODEL2 not on this machine"
fi

echo
if [ "$fails" = "0" ]; then echo "constrained-node checks: ALL PASSED"; else echo "constrained-node checks: $fails FAILED"; fi
exit $fails
