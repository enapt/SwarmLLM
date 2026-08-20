#!/usr/bin/env bash
# Does the swarm aggregate capacity?
#
# Two questions that are routinely conflated and have opposite answers:
#
#   1. Does adding nodes make ONE request faster?
#   2. Does adding nodes let the swarm serve MORE requests per second?
#
# Pipeline splitting answers (1) with "no" on anything but a fast local link —
# a split exchanges activations once per TOKEN, so a boundary costs a round trip
# per token while saving only compute. Request-level parallelism answers (2)
# with "yes" and needs no coordination at all. This script measures both rather
# than assuming either.
#
# Usage:
#   examples/swarm_scaling_bench.sh [-p PORT] [-m MODEL] [-t MAX_TOKENS] CONCURRENCY...
#
# Example:
#   examples/swarm_scaling_bench.sh -m llama-3.2-3b-instruct-q4-k-m 1 2 4 8
#
# Every request gets a UNIQUE prompt. Repeating one collapses to a prefix-cache
# hit (~1.5s) and measures the cache instead of the router — an invalid A/B that
# has already nearly produced a wrong conclusion here once.
set -uo pipefail

PORT=8800
MODEL=llama-3.2-3b-instruct-q4-k-m
MAX_TOKENS=60
while getopts "p:m:t:" opt; do
  case $opt in
    p) PORT=$OPTARG ;;
    m) MODEL=$OPTARG ;;
    t) MAX_TOKENS=$OPTARG ;;
    *) echo "usage: $0 [-p PORT] [-m MODEL] [-t MAX_TOKENS] CONCURRENCY..." >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
LEVELS=("$@")
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(1 2 4 8)

DATA_DIR="${SWARMLLM_NODE_DATA_DIR:-$HOME/.local/share/swarmllm}"
KEY=$(cat "$DATA_DIR/api_key" 2>/dev/null)
if [ -z "$KEY" ]; then echo "no api key at $DATA_DIR/api_key" >&2; exit 1; fi
BASE="http://localhost:$PORT"

if ! curl -s -m 5 -H "Authorization: Bearer $KEY" "$BASE/api/admin/stats" >/dev/null; then
  echo "daemon not answering on $BASE" >&2; exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Bodies are built BEFORE the clock starts, never inside the launch loop.
#
# They used to be built per request with a `python3` call, which costs tens of
# milliseconds each and runs sequentially — so eight "concurrent" requests
# actually arrived spread over about a third of a second. That is the same
# order as the window requests have to arrive within to be batched together,
# so batching engaged on some runs and not others and the results swung by
# 50% with nothing changed. A benchmark whose own launch jitter is the size of
# the effect cannot measure the effect.
prepare_bodies() {
  local n=$1
  python3 - "$WORK" "$MODEL" "$MAX_TOKENS" "$RUN_TAG" "$n" <<'PYEOF'
import json, sys, os
work, model, max_tokens, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
subjects = ["volcanoes","the printing press","tides","the Silk Road","antibiotics","sonar",
            "crop rotation","glaciers","the telegraph","yeast","monsoons","cartography",
            "the abacus","lighthouses","vaccination","steam power","coral reefs",
            "the compass","penicillin","irrigation","the loom","seismographs",
            "kites","radio waves","the sextant","fermentation","windmills",
            "the barometer","papermaking","the pendulum","sonnets","aqueducts"]
for i in range(1, n + 1):
    subject = subjects[i % len(subjects)]
    # Deliberately open-ended, so every request runs to `max_tokens` instead
    # of stopping wherever the model chose to. Aggregate tokens/sec divides by
    # wall clock, so a run whose completions happened to be longer scores
    # higher for no reason at all — which is how a 40% "win" appeared and then
    # evaporated when the harness stopped adding its own noise. The summary
    # reports whether every request hit the cap, so a run that did not is
    # visibly not comparable.
    # A counting task, because it mechanically cannot stop early. Anything
    # open-ended ends where the model chooses, and aggregate tokens/sec divides
    # by wall clock — so a run whose completions happened to be longer scores
    # higher for no reason at all. That is how a 40% "win" appeared and then
    # evaporated once the harness stopped adding its own noise. Every token
    # costs the same forward pass, so a dull sequence measures throughput just
    # as well as an interesting one, and the summary flags any run where a
    # request did not reach the cap.
    prompt = (f"Count upwards from {i * 1000}, one number per line, "
              f"and do not stop. (ref {tag}-{i})")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens}
    with open(os.path.join(work, f"body_{i}.json"), "w") as f:
        json.dump(body, f)
PYEOF
}

one_request() {
  local idx=$1 out=$2
  local body
  body=$(cat "$WORK/body_$idx.json")
  local s e r
  s=$(date +%s.%N)
  r=$(curl -s -m 600 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -X POST "$BASE/v1/chat/completions" -d "$body" 2>/dev/null)
  e=$(date +%s.%N)
  python3 - "$r" "$s" "$e" > "$out" <<'PY'
import json,sys
raw,s,e=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
try:
    d=json.loads(raw)
    ct=d.get("usage",{}).get("completion_tokens",0) or 0
    err=d.get("error",{}).get("message","") if isinstance(d.get("error"),dict) else ""
except Exception:
    ct,err=0,(raw[:120] or "no response")
print(json.dumps({"tokens":ct,"wall":e-s,"err":err}))
PY
}

echo "model=$MODEL max_tokens=$MAX_TOKENS levels=${LEVELS[*]}"
echo
printf '%-6s %-9s %-9s %-11s %-11s %-9s %s\n' \
  N ok fail "wall_s" "agg_tok/s" "per_req" "median_s"

for N in "${LEVELS[@]}"; do
  RUN_TAG="n${N}-$(date +%s)"
  export RUN_TAG
  rm -f "$WORK"/r_*.json "$WORK"/body_*.json
  prepare_bodies "$N"
  bs=$(date +%s.%N)
  for i in $(seq 1 "$N"); do
    one_request "$i" "$WORK/r_$i.json" &
  done
  wait
  be=$(date +%s.%N)

  python3 - "$WORK" "$N" "$bs" "$be" "$MAX_TOKENS" <<'PY'
import json,glob,sys,statistics
work,N,bs,be,MAXT=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
rows=[]
for f in glob.glob(work+"/r_*.json"):
    try: rows.append(json.load(open(f)))
    except Exception: pass
ok=[r for r in rows if r["tokens"]>0]
fail=[r for r in rows if r["tokens"]==0]
wall=be-bs
tot=sum(r["tokens"] for r in ok)
agg=tot/wall if wall>0 else 0
per=statistics.mean([r["tokens"]/r["wall"] for r in ok]) if ok else 0
med=statistics.median([r["wall"] for r in ok]) if ok else 0
capped = sum(1 for r in ok if r["tokens"] >= MAXT)
flag = "" if capped == len(ok) and ok else f"  <-- only {capped}/{len(ok)} hit the cap; not comparable"
print(f"{N:<6} {len(ok):<9} {len(fail):<9} {wall:<11.2f} {agg:<11.2f} {per:<9.2f} {med:.2f}{flag}")
for r in fail[:2]:
    if r["err"]: print(f"       ! {r['err'][:100]}")
PY
  sleep 3
done

echo
echo "Routes taken (from the daemon's own trace lines):"
LOG="$DATA_DIR/node.log"
if [ -r "$LOG" ]; then
  tail -400 "$LOG" | grep "DIAG: request complete" | tail -40 \
    | sed -E 's/.*route=([a-z]+).*segments=([0-9]+).*nodes=([^ ]*).*tok_per_sec=([0-9.]+).*/route=\1 segments=\2 nodes=\3 tok_s=\4/' \
    | sort | uniq -c | sort -rn
else
  echo "  (log not readable at $LOG)"
fi
