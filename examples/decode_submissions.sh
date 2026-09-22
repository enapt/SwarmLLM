#!/bin/bash
# How many GPU submissions does one decoded token cost?
#
# WHY THIS EXISTS
# ---------------
# Decode on the GPU is bound by how many times the CPU thread talks to the
# driver, not by memory bandwidth — measured 2026-09-22 at 1,085 submissions
# and 17.6 of 23.0 ms/token with the card at 52% utilization, and `ms/layer`
# flat (0.48-0.58) across a 3.3x span of bytes/token.
# → `docs/invariants/inference.md` § "A decode token is bound by GPU submission
#   COUNT, not bandwidth".
#
# So a change claiming to make decode faster by doing less per layer is judged
# HERE, by the count, not by the clock: the count is deterministic while this
# box's tok/s spreads 10-18% run to run.
#
# WHAT IT CANNOT TELL YOU
# -----------------------
#   * Per-call TIMES are nsys-inflated (a launch reads ~12 us under nsys, and
#     WSL2 is itself ~2-3x native Linux). COUNTS are exact; quote those.
#   * Nothing about whether the GPU work itself is well shaped. A token that
#     submits 200 good kernels could still be slow.
#   * GPU-side kernel activity is usually unavailable under WSL2, so this is
#     the API side only. That is the side that was the bottleneck.
#   * Only the LAST sustained burst is counted. If the measured request is
#     served remotely, or the model reloads mid-run, the window is not what you
#     think — check the printed ms/token against the tok/s you expect.
#
# It restarts the node it profiles, so do not run it while a tester is using
# this one.
#
# Usage: decode_submissions.sh [MODEL] [TOKENS] [BINARY]
set -u

MODEL="${1:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
TOKENS="${2:-32}"
BIN="${3:-$HOME/.local/bin/swarmllm}"
PORT="${PORT:-8800}"
OUT="${OUT:-/tmp/swarm_decode_submissions}"
KEY_FILE="${KEY_FILE:-$HOME/.local/share/swarmllm/api_key}"

command -v nsys >/dev/null || { echo "nsys not found (CUDA toolkit)"; exit 2; }
[ -f "$KEY_FILE" ] || { echo "no api key at $KEY_FILE"; exit 2; }
KEY=$(cat "$KEY_FILE")
mkdir -p "$OUT"

stop_nodes() {
  # By /proc/PID/exe, never a cmdline pattern: `pgrep -f` matches shell
  # wrappers and has cost wasted restarts before.
  #
  # ⚠ The `(deleted)` arm is load-bearing, not defensive. `readlink` answers
  # `/path/to/swarmllm (deleted)` for a process whose binary has since been
  # REPLACED — which is the normal case here, because this script is run right
  # after a rebuild. Matching only `*swarmllm` silently skips exactly the
  # processes that need killing, and the next node then dies on
  # "Database already open. Cannot acquire lock."
  for p in $(pgrep -x swarmllm 2>/dev/null); do
    case "$(readlink "/proc/$p/exe" 2>/dev/null)" in
      *swarmllm | *"swarmllm (deleted)") kill "$p" ;;
    esac
  done
  for _ in $(seq 1 20); do pgrep -x swarmllm >/dev/null || break; sleep 1; done
}

echo "== stopping any running node =="
stop_nodes

rm -f "$OUT"/rep.*
echo "== profiling $BIN =="
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib \
nsys profile --output="$OUT/rep" --force-overwrite=true \
  --trace=cuda --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --session-new=swarmsubs \
  "$BIN" run -p "$PORT" > "$OUT/node.log" 2>&1 &

up=0
for i in $(seq 1 120); do
  if [ -n "$(curl -s -m 3 "http://localhost:$PORT/health/ready" 2>/dev/null)" ]; then
    up=$i; break
  fi
  sleep 1
done
[ "$up" -eq 0 ] && { echo "node never came up; see $OUT/node.log"; stop_nodes; exit 1; }
echo "node up after ${up}s"

gen() {
  curl -s -m 900 -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$1,\"temperature\":0}"
}

echo "== warm (loads the model; its calls are excluded below) =="
gen 4 "Hi" > /dev/null
sleep 2
echo "== measured: $TOKENS tokens =="
gen "$TOKENS" "Count from one to fifty in words." \
  | python3 -c "import json,sys; print('usage:', json.load(sys.stdin)['usage'])" 2>/dev/null

sleep 1
nsys stop --session=swarmsubs >/dev/null 2>&1
for _ in $(seq 1 180); do [ -f "$OUT/rep.nsys-rep" ] && break; sleep 1; done
stop_nodes

echo "== analysing =="
# Generates rep.sqlite as a side effect, which is what we read below.
nsys stats --report cuda_api_sum --format table "$OUT/rep.nsys-rep" >/dev/null 2>&1

python3 - "$OUT/rep.sqlite" "$TOKENS" <<'PYEOF'
import sqlite3
import sys
from collections import Counter

db, tokens = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
names = {i: n for i, n in c.execute("select id, value from StringIds")}
rows = list(c.execute(
    "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"))
if not rows:
    sys.exit("no CUDA API rows — did nsys capture the worker?")

# The measured generation is the last sustained burst. Everything before the
# last long gap is startup, weight upload and the warm-up request; counting
# those would overstate per-token work several-fold.
GAP_NS = 150_000_000
start = 0
for i in range(len(rows) - 1, 0, -1):
    if rows[i][0] - rows[i - 1][0] > GAP_NS:
        start = i
        break
win = rows[start:]
span_ms = (win[-1][1] - win[0][0]) / 1e6

cnt, tot = Counter(), Counter()
for s, e, nid in win:
    n = names.get(nid, f"?{nid}")
    cnt[n] += 1
    tot[n] += e - s

print(f"\ndecode window {span_ms:.0f} ms, {len(win)} API calls, "
      f"{tokens} tokens -> {span_ms/tokens:.2f} ms/token\n")
print(f"{'API':<30} {'calls':>7} {'per token':>10} {'med us':>8} {'ms/token':>9}")
print("-" * 68)
for n, k in cnt.most_common(12):
    print(f"{n:<30} {k:>7} {k/tokens:>10.1f} {tot[n]/k/1000:>8.2f} "
          f"{tot[n]/1e6/tokens:>9.2f}")


def pick(frag):
    return sum(k for n, k in cnt.items() if frag in n)


subs = pick("LaunchKernel") + pick("Memset")
print(f"\nSUBMISSIONS/TOKEN: {subs/tokens:.0f} "
      f"({pick('LaunchKernel')/tokens:.0f} launches + "
      f"{pick('Memset')/tokens:.0f} memsets)")
print(f"allocs/token {pick('MemAlloc')/tokens:.0f}, "
      f"event ops/token {pick('Event')/tokens:.0f}")
print(f"driver API time {sum(tot.values())/1e6/tokens:.2f} ms/token "
      f"of {span_ms/tokens:.2f} ms/token wall")
print("\nCompare the COUNT across arms. Times here are nsys-inflated.")
PYEOF
