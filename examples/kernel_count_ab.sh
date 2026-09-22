#!/bin/bash
# A/B two arms of one binary by KERNEL COUNT, and check the reply did not move.
#
# WHY THIS EXISTS
# ---------------
# Decode on the GPU is bound by how many times the CPU thread talks to the
# driver (`docs/invariants/inference.md` § "A decode token is bound by GPU
# submission COUNT, not bandwidth"), so a fusion is judged by the count it
# removes. But this box's tok/s spreads 10-18% run to run and cannot resolve a
# change below ~10% (`docs/DIAGNOSTICS.md` § "Traps that have cost real time"),
# which is larger than most single fusions are worth.
#
# So the count is the instrument and the clock is not. `decode_submissions.sh`
# gives the TOTAL submissions under nsys; this gives the per-kernel-name
# breakdown for two arms of the same binary, which is what says *which* kernel
# went away — and it needs no profiler, so it is cheap enough to run on every
# fusion.
#
# It also diffs the generated text between the arms. A fusion here is written
# to be bit-identical to the ops it replaces, so **anything other than an
# identical reply is a bug, not a rounding difference.**
#
# WHAT IT CANNOT TELL YOU
# -----------------------
#   * Nothing about time. `SWARMLLM_COUNT_KERNELS=1` takes a mutex per launch;
#     a tok/s taken with it on is meaningless. **Never quote one from here.**
#   * Nothing about cuBLAS's internal launches (~45/token), which do not pass
#     through candle. Take the total from `decode_submissions.sh`.
#   * Nothing about allocations or frees — only kernel launches. A fusion that
#     removes one op removes an alloc and a free too, and they are invisible
#     here; count them as one each per removed op.
#
# It restarts the node it measures, so do not run it while a tester is using
# this one.
#
# Usage: kernel_count_ab.sh VAR ON_VALUE OFF_VALUE [MODEL] [TOKENS] [BINARY]
#   e.g. kernel_count_ab.sh SWARMLLM_FUSE_SILU_MUL 1 0
set -u

VAR="${1:?usage: kernel_count_ab.sh VAR ON_VALUE OFF_VALUE [MODEL] [TOKENS] [BIN]}"
ON_VAL="${2:?on value}"
OFF_VAL="${3:?off value}"
MODEL="${4:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
TOKENS="${5:-24}"
BIN="${6:-$PWD/target/release/swarmllm}"
PORT="${PORT:-8800}"
OUT="${OUT:-/tmp/swarm_kernel_ab}"
KEY_FILE="${KEY_FILE:-$HOME/.local/share/swarmllm/api_key}"
PROMPT="${PROMPT:-Count from one to twenty in words.}"

[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 2; }
[ -f "$KEY_FILE" ] || { echo "no api key at $KEY_FILE"; exit 2; }
KEY=$(cat "$KEY_FILE")
mkdir -p "$OUT"

stop_nodes() {
  # By /proc/PID/exe, never a cmdline pattern: `pgrep -f` matches shell
  # wrappers (gotcha #666). The `(deleted)` arm is load-bearing — `readlink`
  # answers `... (deleted)` for a process whose binary has been REPLACED, which
  # is the normal case right after a rebuild, and matching only `*swarmllm`
  # silently skips exactly the process that needs killing (gotcha #678).
  for p in $(pgrep -x swarmllm 2>/dev/null); do
    case "$(readlink "/proc/$p/exe" 2>/dev/null)" in
      *swarmllm | *"swarmllm (deleted)") kill "$p" ;;
    esac
  done
  for _ in $(seq 1 20); do pgrep -x swarmllm >/dev/null || break; sleep 1; done
}

run_arm() {
  local label="$1" value="$2"
  local log="$OUT/$label.log" reply="$OUT/$label.reply"

  stop_nodes
  : > "$log"
  echo "== arm $label: $VAR=$value =="
  env "$VAR=$value" SWARMLLM_COUNT_KERNELS=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib \
    "$BIN" run -p "$PORT" >> "$log" 2>&1 &

  local up=0 i
  for i in $(seq 1 120); do
    if [ -n "$(curl -s -m 3 "http://localhost:$PORT/health/ready" 2>/dev/null)" ]; then
      up=$i; break
    fi
    sleep 1
  done
  [ "$up" -eq 0 ] && { echo "node never came up; see $log"; stop_nodes; return 1; }

  # The warm request loads the model. Its forwards are counted too, so the
  # counters are read from the LAST decode forward only — see the awk below.
  curl -s -m 900 -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"temperature\":0}" \
    > /dev/null
  sleep 2

  curl -s -m 900 -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$TOKENS,\"temperature\":0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
    > "$reply" 2>/dev/null

  stop_nodes

  # One decode forward's table: the LAST `KERNELS seq_len=1` block in the log.
  # seq_len=1 excludes prefill, whose mix is different and much larger.
  awk '
    /^KERNELS seq_len=1 /  { buf = $0 "\n"; inblk = 1; next }
    /^KERNELS /            { inblk = 0; next }
    inblk && /^  +[0-9]/   { buf = buf $0 "\n"; next }
    inblk                  { inblk = 0 }
    END                    { printf "%s", buf }
  ' "$log" > "$OUT/$label.kernels"

  if [ ! -s "$OUT/$label.kernels" ]; then
    echo "  no decode KERNELS block in $log"
    echo "  (is this a candle-cuda build, and did the request run LOCALLY?)"
    return 1
  fi
  cat "$OUT/$label.kernels"
}

run_arm on  "$ON_VAL"  || exit 1
echo
run_arm off "$OFF_VAL" || exit 1

echo
echo "== launches per layer: on vs off =="
python3 - "$OUT/on.kernels" "$OUT/off.kernels" <<'PYEOF'
import re, sys

def read(p):
    per, total = {}, None
    for line in open(p):
        m = re.match(r"^KERNELS .*— (\d+) launches", line)
        if m:
            total = int(m.group(1)); continue
        m = re.match(r"^\s+(\d+)\s+([\d.]+)/layer\s+(\S+)", line)
        if m:
            per[m.group(3)] = float(m.group(2))
    return total, per

(ton, on), (toff, off) = read(sys.argv[1]), read(sys.argv[2])
names = sorted(set(on) | set(off))
print(f"{'kernel':<34}{'on':>9}{'off':>9}{'delta':>9}")
for n in names:
    a, b = on.get(n, 0.0), off.get(n, 0.0)
    if abs(a - b) < 1e-9:
        continue
    print(f"{n:<34}{a:>9.2f}{b:>9.2f}{a-b:>+9.2f}")
print(f"{'TOTAL launches/token':<34}{ton:>9}{toff:>9}{ton-toff:>+9}")
PYEOF

echo
if diff -q "$OUT/on.reply" "$OUT/off.reply" > /dev/null 2>&1; then
  echo "== replies IDENTICAL =="
else
  echo "== ⚠ REPLIES DIFFER — the fusion is not bit-identical. This is a bug. =="
  diff "$OUT/on.reply" "$OUT/off.reply" | head -20
fi
