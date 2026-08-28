#!/usr/bin/env bash
# What does the GPU swap floor cost, in CONVERSATION?
#
# Two models that each fit the card but not together, alternating turns, each
# turn resending the whole conversation so a warm prefix cache is worth
# something. That last part is the point: a one-shot benchmark cannot see the
# cost eviction is supposed to have, since a swap kills the worker and the
# prefix cache is per-worker.
#
# Same binary for every arm — the floor is switched by
# SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS, never by rebuilding (gotcha #255).
#
# Measured 2026-08-28, RTX 3070, llama-3.2-3b + phi-3.5-mini, 6200 MB budget:
#
#     floor 60 s (old)  299.3 s   0 swaps, 2 processor placements
#     floor  0 s         81.9 s   7 swaps, 0
#     floor  5 s (new)   88.8 s   7 swaps, 0
#
# The reasoning the 60 s floor rested on — that re-prefilling after a swap would
# cost more than running on the processor — is what this refutes. Re-run it on
# other hardware before changing the default again; the shape that would remove
# the constant altogether is in docs/FUTURE_WORK.md.
#
#   BIN=./target/release/swarmllm bash examples/swap_patience.sh
#   ARMS=measured ... # just the current default
set -u

PORT="${PORT:-8903}"
BIN="${BIN:-./target/release/swarmllm}"
DIR="${DIR:-/tmp/swarmllm-swap-patience}"
A="${A:-llama-3.2-3b-instruct-q4-k-m}"
B="${B:-phi-3.5-mini-instruct.q4-k-m}"
TURNS="${TURNS:-4}"

# ~1200 tokens of stable preamble, so each turn has a real prefix to reuse.
PREAMBLE=$(python3 -c "
import json
para = ('The node keeps a per-worker cache of prompt prefixes so that a second '
        'turn in the same conversation does not have to be processed from the '
        'beginning. ')
print(json.dumps(para * 60))
")

req() { # model, conversation-json  -> seconds
  local model="$1" convo="$2" t0 t1
  t0=$(date +%s.%N)
  curl -s -m 600 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":$convo,\"max_tokens\":16,\"temperature\":0}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null
  t1=$(date +%s.%N)
  python3 -c "print(f'{$t1-$t0:.2f}')"
}

run_arm() { # floor_secs, label
  local floor="$1" label="$2"
  rm -rf "$DIR/data"; mkdir -p "$DIR/data/models"
  cp -al ~/.local/share/swarmllm/models/"$A" "$DIR/data/models/"
  cp -al ~/.local/share/swarmllm/models/"$B" "$DIR/data/models/"
  cp "$DIR/config.toml" "$DIR/data/config.toml"

  if [ -n "$floor" ]; then export SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS="$floor"; else unset SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS; fi
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 \
    setsid nohup "$BIN" run -p "$PORT" -d "$DIR/data" -c "$DIR/data/config.toml" -v \
    > "$DIR/$label.log" 2>&1 < /dev/null &
  sleep 22
  KEY=$(cat "$DIR/data/api_key")

  echo "--- arm: $label (floor=${floor:-measured}) ---"
  local convo_a convo_b
  convo_a="[{\"role\":\"system\",\"content\":$PREAMBLE}]"
  convo_b="[{\"role\":\"system\",\"content\":$PREAMBLE}]"
  local total=0
  for i in $(seq 1 "$TURNS"); do
    convo_a=$(python3 -c "
import json,sys
c=json.loads(sys.argv[1]); c.append({'role':'user','content':'Turn $i: name one benefit of caching, briefly.'})
print(json.dumps(c))" "$convo_a")
    ta=$(req "$A" "$convo_a")
    convo_a=$(python3 -c "
import json,sys
c=json.loads(sys.argv[1]); c.append({'role':'assistant','content':'Reusing work already done.'})
print(json.dumps(c))" "$convo_a")

    convo_b=$(python3 -c "
import json,sys
c=json.loads(sys.argv[1]); c.append({'role':'user','content':'Turn $i: name one benefit of caching, briefly.'})
print(json.dumps(c))" "$convo_b")
    tb=$(req "$B" "$convo_b")
    convo_b=$(python3 -c "
import json,sys
c=json.loads(sys.argv[1]); c.append({'role':'assistant','content':'Reusing work already done.'})
print(json.dumps(c))" "$convo_b")

    echo "  turn $i:  A=${ta}s  B=${tb}s"
    total=$(python3 -c "print(f'{$total+$ta+$tb:.2f}')")
  done
  echo "  TOTAL ${total}s"
  echo "  swaps: $(grep -ac 'Freeing graphics memory from an idle model' "$DIR/$label.log")"
  echo "  cpu placements: $(grep -ac 'Model will run on the CPU' "$DIR/$label.log")"

  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    e=$(readlink /proc/$p/exe 2>/dev/null) || true
    case "$e" in *SwarmLLM/target*) kill "$p" 2>/dev/null || true;; esac
  done
  sleep 8
}

mkdir -p "$DIR"
cat > "$DIR/config.toml" <<'EOF'
[network]
enable_mdns = false
bootstrap_peers = []

[auto_manage]
enabled = false
prune_enabled = false

[resources]
max_gpu_vram_mb = 6200
EOF

case "${ARMS:-all}" in
  measured) run_arm "" "measured" ;;
  *) run_arm 60 "floor60"; run_arm 0 "floor0"; run_arm "" "measured" ;;
esac
