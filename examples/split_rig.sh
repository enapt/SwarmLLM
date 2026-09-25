#!/bin/bash
# Two nodes on this machine, each holding a CHOSEN subset of one model's shards,
# so the planner has to split a request between them — then either ask through
# the split, or break one side of it mid-reply and see what the client gets.
#
# WHY THIS EXISTS
# ---------------
# The release gate's behaviour checks need a SPLIT, and the family conformance
# run never makes one (`gpu_layers = 0`, whole model local). This rig found
# #93 — a split reply decoded from an empty cache, visible only when a peer's
# worker is retired between two forwards — and nothing else in the repo can
# reproduce it. Until 2026-09-24 it lived outside the repo and needed a
# hand-built directory; now it builds its own.
#
# MODES
#   split  two greedy questions through A; prints the route headers and replies.
#   kill   one long reply through A, then mid-reply: KILL=A kills A's worker,
#          KILL=B kills B's worker, KILL=B_UNLOAD unloads the model on B through
#          the admin API (lands between two forwards, as a user unloading from
#          the Dashboard would). The .201/.202 gate expectation for B_UNLOAD: B
#          refuses the next forward ("no longer holds the conversation"), one
#          retry, 200, no garbled text.
#   failover  FOUR nodes, for #17's composite stand-in (a failed segment taken
#          over by several nodes that cover its layers between them). A holds
#          shard 0 and coordinates, B holds every other shard, C holds all of
#          those but the last and D holds the last — so B's segment has no
#          single standby and C+D is the only cover. Three greedy runs of one
#          long prompt: B healthy; B's worker killed the moment it starts the
#          PROMPT pass (the only pass a composite is offered on); B stopped, so
#          the plan itself is A→C→D. PASS = the coordinator logged the
#          composite takeover, the router did NOT retry, and every request
#          answered. The first two are not optional: a failed takeover falls
#          through to the router's retry, which re-plans and produces a correct
#          reply without the takeover ever having run (gotcha #706). Whether the
#          reply is the MODEL's is judged against llama.cpp, not byte-equality
#          with the control: examples/score_against_reference.py.
#   repeat REPEAT=N (default 3) greedy runs of the long prompt through A's
#          split, saved to $OUT/repeat.jsonl with $OUT/prompt.txt, for scoring
#          against llama.cpp (FUTURE_WORK #106). The FIRST request takes the
#          n-gram-only path and the rest the standard loop (it self-disables
#          per process), so one arm yields both. Vary one thing per arm with
#          EXTRA_TOML, e.g. EXTRA_TOML=$'[inference]\nactivation_compression = false'.
#   context  the FAILOVER topology, but B serves a SHORTER conversation than the
#          long prompt (CEIL_B, default 512, as B's own max_seq_len_override) —
#          the field report of 2026-09-25, where an 8192-token peer refused an
#          8560-token prompt the coordinator served at 65536 and the request
#          was abandoned although a standby existed. Two asks: with C and D up,
#          PASS = A either skipped B (it ADVERTISES its ceiling) or failed over
#          from B's refusal, C+D took the segment over, no router retry, 200;
#          then with D stopped, nobody can serve the length, and PASS = a 400
#          that names the other machines' limit instead of telling the caller
#          to raise their own. Run BIN_B = an older release to see the refusal
#          path, and BIN_A = an older release for the baseline (it gives up).
#   failover_mid  FIVE nodes: #17's composite stand-in in the MIDDLE of a
#          pipeline, which the four-node `failover` cannot make (its B runs to
#          the last layer). A holds shard 0 and coordinates, B the middle
#          shards, C all of B's but its last, D B's last, E the model's last —
#          so the plan is A→B→E and C+D is the only cover for B. Same three
#          arms and PASS rule as `failover`; the control plan is A→C→D→E.
#          Needs a model with at least 4 shard files here.
#   whole  #111's WHOLE-MODEL half. A holds none of the model (so every plan is
#          one peer running all of it, `remote_generate`), B holds every part
#          but serves a SHORTER conversation than the long prompt (CEIL_B,
#          default 512) and has the graphics card (so it is priced first), C
#          holds every part at the default ceiling on the processor. Ask 1,
#          with C not yet started: nobody can serve the length, and PASS = a
#          400 naming B's limit ("serve at most"), not the peer's own advice
#          and not a 503. Ask 2, with C up: PASS = 200, and when A's plan tried
#          B first, A logged B's refusal and re-planned onto C. Run BIN_A = an
#          older release for the baseline (B's refusal comes back as the 400).
#   fetch  A holds every part, B only part 0; B is asked to download part
#          FETCH_SHARD (default 1) from A over P2P. Prints whether it landed and
#          every `network event loop stalled` line B logged meanwhile — the hash
#          of a downloaded part ran ON the event loop until FUTURE_WORK #108
#          (~229 ms per 512 MB part, over the loop's 100 ms tripwire).
#
# usage: split_rig.sh split|kill|failover|failover_mid|context|whole|repeat|fetch <binary> [<binary for B>]
#   MODEL      model id (default: tinyllama for split, llama-3.2-3b for kill
#              and failover)
#   SHARDS_A   shard indices A holds, comma-separated (default 0; kill: 0,LAST)
#   SHARDS_B   shard indices B holds (default: every shard A lacks; kill: all)
#   GPU_A/B    SWARMLLM_INFERENCE_GPU_LAYERS for each node ("" = auto)
#   EXTRA_TOML appended to EVERY node's config.toml (default: nothing)
#   MODELS_DIR where the shards come from (default: the live node's)
#   OUT        where logs and replies go (default: a temp dir, printed)
#
# The rig's shard files are HARD LINKS to MODELS_DIR's (no copy, no disk), so
# MODELS_DIR must be on the same filesystem as the rig ($HOME by default). The
# live node's own mDNS will find the rig, and a rig node that can see a third
# peer can route through it — so this refuses to run a request unless A sees
# exactly one peer. Stop the live node first. `gossip_network_id` is NOT
# isolation (#352).
set -u

MODE="${1:?usage: split_rig.sh split|kill|failover|context|repeat|fetch <binary> [<binary for B>]}"
BIN_A="${2:?binary}"
BIN_B="${3:-$BIN_A}"
case "$MODE" in split|kill|failover|failover_mid|context|whole|repeat|fetch) ;; *) echo "mode must be split, kill, failover, failover_mid, context, whole, repeat or fetch"; exit 2 ;; esac
if { [ "$MODE" = context ] || [ "$MODE" = whole ]; } && printf '%s' "${EXTRA_TOML:-}" | grep -q '^\[inference\]'; then
  echo "$MODE: EXTRA_TOML may not open [inference] — B's ceiling is written there"; exit 2
fi
[ -x "$BIN_A" ] && [ -x "$BIN_B" ] || { echo "binary not executable"; exit 2; }
if [ "$MODE" = split ]; then
  MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
else
  MODEL="${MODEL:-llama-3.2-3b-instruct-q4-k-m}"
fi
MODELS_DIR="${MODELS_DIR:-$HOME/.local/share/swarmllm/models}"
SRC="$MODELS_DIR/$MODEL"
[ -f "$SRC/manifest.json" ] || { echo "no manifest for $MODEL in $MODELS_DIR"; exit 2; }
SHARDS=$(ls "$SRC" | sed -n 's/^shard_\([0-9]*\)\.bin$/\1/p' | sed 's/^0*\([0-9]\)/\1/' | sort -n)
LAST=$(echo "$SHARDS" | tail -1)
N=$(echo "$SHARDS" | wc -l)
[ "$N" -ge 2 ] || { echo "$MODEL has $N shard file(s) here; a split needs at least 2"; exit 2; }
if [ "$MODE" = split ] || [ "$MODE" = repeat ]; then
  SHARDS_A="${SHARDS_A:-0}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vxF -f <(echo "$SHARDS_A" | tr ',' '\n') | paste -sd,)}"
  # Processor unless asked otherwise: the reference is scored on the processor.
  [ "$MODE" = repeat ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
elif [ "$MODE" = fetch ]; then
  SHARDS_A=$(echo "$SHARDS" | paste -sd,)
  SHARDS_B=0
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
  # Shard SERVING is throttled by the contribution level (~10 Mbit/s by
  # default here, 7 s an 8 MiB chunk), which is a rig measuring the hash's
  # cost, not the link's, waiting minutes for nothing.
  EXTRA_TOML="${EXTRA_TOML:-$'[resources]\nmax_bandwidth_mbps = 10000'}"
elif [ "$MODE" = failover ] || [ "$MODE" = context ]; then
  # B's range must need TWO nodes to cover it, so C stops one shard short.
  [ "$N" -ge 3 ] || { echo "failover needs a model with at least 3 shard files here; $MODEL has $N"; exit 2; }
  SHARDS_A=0
  SHARDS_B=$(echo "$SHARDS" | grep -vx 0 | paste -sd,)
  SHARDS_C=$(echo "$SHARDS" | grep -vx 0 | grep -vx "$LAST" | paste -sd,)
  SHARDS_D=$LAST
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = failover_mid ]; then
  # B's range must be neither the first nor the last, and need TWO nodes.
  [ "$N" -ge 4 ] || { echo "failover_mid needs a model with at least 4 shard files here; $MODEL has $N"; exit 2; }
  SHARDS_A=0
  MID=$(echo "$SHARDS" | grep -vx 0 | grep -vx "$LAST")
  MID_LAST=$(echo "$MID" | tail -1)
  SHARDS_B=$(echo "$MID" | paste -sd,)
  SHARDS_C=$(echo "$MID" | grep -vx "$MID_LAST" | paste -sd,)
  SHARDS_D=$MID_LAST
  SHARDS_E=$LAST
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = whole ]; then
  # A holds the header only, so it knows the model's declared context but can
  # run none of it; B and C each hold all of it.
  SHARDS_A=""
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
  SHARDS_C=$SHARDS_B
  GPU_A="${GPU_A:-0}"
else
  SHARDS_A="${SHARDS_A:-0,$LAST}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | paste -sd,)}"
fi
# ANY other SwarmLLM process on this machine joins the rig, whatever either
# side's config says: every node dials 127.0.0.1 on the ports within ±10 of its
# own and on the 8800/8900/9000/… bases at startup
# (`network::discovery::probe_loopback_peers`), and peer exchange then brings
# in everything THAT node knows — the public swarm included. mDNS off and a
# private gossip id stop none of it. A probe node on 8970 (D's range) put
# nine peers in front of A on 2026-09-25.
OTHERS=$(ps -eo pid,comm | awk '$2 ~ /swarmllm/ {print $1}' | paste -sd' ')
if [ -n "$OTHERS" ]; then
  echo "rig: other SwarmLLM processes are running (pids: $OTHERS). Stop them first —"
  echo "     any node on this machine finds the rig through its loopback probe."
  exit 2
fi
BASE=$(mktemp -d "$HOME/.split-rig.XXXXXX")
OUT="${OUT:-$BASE/out}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib

make_node() { # dir shards bootstrap
  local d="$1" m="$1/models/$MODEL"
  mkdir -p "$m"
  for f in gguf_header.bin hf_source.json manifest.json tied_output_weight.bin; do
    [ -f "$SRC/$f" ] && ln "$SRC/$f" "$m/$f"
  done
  for i in $(echo "$2" | tr ',' ' '); do
    ln "$SRC/$(printf 'shard_%03d.bin' "$i")" "$m/" || { echo "cannot link shard $i (another filesystem?)"; exit 2; }
  done
  cat > "$d/config.toml" <<CFG
[network]
bootstrap_peers = [$3]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-split-rig"
enable_mdns = false
enable_upnp = false
relay_forwarding_auto = false

[auto_manage]
enabled = false
prune_enabled = false

[updates]
mode = "off"

[ui]
open_browser_on_start = false
CFG
  [ -n "${EXTRA_TOML:-}" ] && printf '\n%s\n' "$EXTRA_TOML" >> "$d/config.toml"
  return 0
}
start() { # dir port bin gpu
  if [ -n "$4" ]; then
    SWARMLLM_INFERENCE_GPU_LAYERS="$4" SWARMLLM_NODE_DATA_DIR="$1" "$3" run -p "$2" > "$1/node.log" 2>&1 &
  else
    SWARMLLM_NODE_DATA_DIR="$1" "$3" run -p "$2" > "$1/node.log" 2>&1 &
  fi
  echo $!
}
up() { # dir port
  for _ in $(seq 1 90); do
    [ -f "$1/api_key" ] && curl -s -m 3 "localhost:$2/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "node on $2 never came up — see $1/node.log"; return 1
}
# Kill only what this script started (gotcha #283), and keep the logs.
cleanup() {
  kill ${PA:-} ${PB:-} ${PC:-} ${PD:-} ${PE:-} 2>/dev/null; sleep 3
  for n in A B C D E; do cp "$BASE/$n/node.log" "$OUT/$n.log" 2>/dev/null; done
  rm -rf "$BASE/A" "$BASE/B" "$BASE/C" "$BASE/D" "$BASE/E"
  [ "$OUT" = "$BASE/out" ] || rmdir "$BASE" 2>/dev/null
  echo "rig: logs and replies in $OUT"
}
trap cleanup EXIT

make_node "$BASE/A" "$SHARDS_A" ""
# The n-gram-only path takes every request a coordinator holding the header can
# tokenize, and it is a SEGMENT path (its own failover covers #111 there). The
# whole-model path under test is what a coordinator runs once that path is off
# for it — no header, or its payoff check has switched it off.
[ "$MODE" = whole ] && printf '\n[inference]\nngram_lookup_enabled = false\n' >> "$BASE/A/config.toml"
PA=$(start "$BASE/A" 8900 "$BIN_A" "${GPU_A:-}")
up "$BASE/A" 8900 || exit 1
KA=$(cat "$BASE/A/api_key")
# B dials A explicitly (mDNS is off). Any direct address A publishes will do;
# WSL2's NAT gateway 10.255.255.254 is the documented source of churn between
# two nodes on one host, so prefer another (two_node_test.sh has the detail).
ADDRS=$(curl -s -m 8 -H "Authorization: Bearer $KA" "localhost:8900/api/admin/diagnostics?full=1" \
  | grep -oE "/ip4/[0-9.]+/tcp/[0-9]+/p2p/[A-Za-z0-9]+" | grep -v "p2p-circuit")
ADDR=$(echo "$ADDRS" | grep -v "10\.255\.255\.254" | head -1)
[ -z "$ADDR" ] && ADDR=$(echo "$ADDRS" | head -1)
[ -n "$ADDR" ] || { echo "A published no address to dial"; exit 1; }
make_node "$BASE/B" "$SHARDS_B" "\"$ADDR\""
# B alone serves a conversation shorter than the long prompt.
{ [ "$MODE" = context ] || [ "$MODE" = whole ]; } && printf '\n[inference]\nmax_seq_len_override = %s\n' "${CEIL_B:-512}" >> "$BASE/B/config.toml"
PB=$(start "$BASE/B" 8920 "$BIN_B" "${GPU_B:-}")
up "$BASE/B" 8920 || exit 1
PEERS_EXPECTED=1
if [ "$MODE" = failover ] || [ "$MODE" = context ] || [ "$MODE" = failover_mid ]; then
  # Processor only unless asked otherwise (all four nodes): four daemons on
  # one card is #104's setup, and a KV refusal there would read as a failover
  # result.
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  make_node "$BASE/D" "$SHARDS_D" "\"$ADDR\""
  PD=$(start "$BASE/D" 8960 "$BIN_A" "${GPU_D:-0}")
  up "$BASE/C" 8940 || exit 1
  up "$BASE/D" 8960 || exit 1
  PEERS_EXPECTED=3
  echo "rig: C=[$SHARDS_C] D=[$SHARDS_D] gpu=${GPU_C:-0}/${GPU_D:-0}"
  if [ "$MODE" = failover_mid ]; then
    make_node "$BASE/E" "$SHARDS_E" "\"$ADDR\""
    PE=$(start "$BASE/E" 8980 "$BIN_A" "${GPU_E:-0}")
    up "$BASE/E" 8980 || exit 1
    PEERS_EXPECTED=4
    echo "rig: E=[$SHARDS_E] gpu=${GPU_E:-0}"
  fi
fi

echo "rig: $MODEL  A=[$SHARDS_A] $("$BIN_A" --version) gpu=${GPU_A:-auto}  B=[$SHARDS_B] $("$BIN_B" --version) gpu=${GPU_B:-auto}"
peers() {
  curl -s -m 5 -H "Authorization: Bearer $KA" localhost:8900/api/admin/peers | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); p=d if isinstance(d,list) else d.get("peers",[]); print(len(p))
except Exception: print(0)'
}
for _ in $(seq 1 60); do n=$(peers); [ "${n:-0}" -ge "$PEERS_EXPECTED" ] && break; sleep 2; done
sleep 5
n=$(peers)
if [ "${n:-0}" -ne "$PEERS_EXPECTED" ]; then
  echo "rig: A sees $n peers, not $PEERS_EXPECTED — something else on this machine or LAN is"
  echo "     reachable (or a rig node is not), and the planner may route through it."
  echo "     Stop every other SwarmLLM process on this machine and try again."
  exit 1
fi
echo "rig: A sees exactly its $PEERS_EXPECTED rig peer(s)"

ask() { # prompt max_tokens label  (writes $OUT/<label>.{hdr,body}, prints a JSON line)
  local body
  body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"max_tokens":int(sys.argv[3]),"temperature":0,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$1" "$2")
  curl -s -m 900 -D "$OUT/$3.hdr" -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
       -X POST localhost:8900/v1/chat/completions -d "$body" -o "$OUT/$3.body"
  python3 -c 'import json,sys
h=open(sys.argv[1]).read().lower()
status=h.splitlines()[0].strip() if h else "no response"
hdr=lambda k: [l.split(":",1)[1].strip() for l in h.splitlines() if l.startswith(k)]
raw=open(sys.argv[2]).read()
try:
  c=json.loads(raw)["choices"][0]["message"].get("content")
except Exception: c="ERR "+raw[:300]
print(json.dumps({"status":status,"route":hdr("x-swarm-route"),"segments":hdr("x-swarm-segments"),"content":c}))' "$OUT/$3.hdr" "$OUT/$3.body"
}

if [ "$MODE" = split ]; then
  ask "What is the capital of France? Answer in one sentence." 64 q1 | tee "$OUT/replies.jsonl"
  ask "Write a short Python function that returns the factorial of n." 64 q2 | tee -a "$OUT/replies.jsonl"
  exit 0
fi

# ~560 prompt tokens: long enough to catch a prompt pass mid-way (failover), and
# the prompt the #106 reference scores were taken on (repeat).
PROMPT="Here are some notes on household appliances. $(for i in $(seq 1 12); do printf 'A refrigerator moves heat from its inside to the room using a refrigerant that evaporates in the cold coils and condenses in the warm ones; the compressor drives the cycle and the thermostat decides when it runs. '; done)Using only these notes, explain step by step how a refrigerator keeps food cold."

if [ "$MODE" = fetch ]; then
  KB=$(cat "$BASE/B/api_key")
  FETCH_SHARD="${FETCH_SHARD:-1}"
  since=$(date -u +%Y-%m-%dT%H:%M:%S)
  echo "fetch: asking B for part $FETCH_SHARD: $(curl -s -m 30 -X POST -H "Authorization: Bearer $KB" \
    "localhost:8920/api/admin/models/$MODEL/shards/$FETCH_SHARD/download")"
  for _ in $(seq 1 600); do
    grep -qE "P2P shard download complete|did not arrive intact|failed hash verification" "$BASE/B/node.log" && break
    sleep 1
  done
  grep -E "P2P shard download complete|failed hash verification|did not arrive intact|removed while it was being checked" "$BASE/B/node.log" | cut -c1-200
  echo "fetch: event-loop stalls on B since the request:"
  awk -v t="$since" '$1 >= t' "$BASE/B/node.log" | grep "network event loop stalled" | cut -c1-220 || true
  echo "fetch: $(awk -v t="$since" '$1 >= t' "$BASE/B/node.log" | grep -c "network event loop stalled") stall line(s)"
  exit 0
fi

if [ "$MODE" = repeat ]; then
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  : > "$OUT/repeat.jsonl"
  for i in $(seq 1 "${REPEAT:-3}"); do
    ask "$PROMPT" 120 "repeat$i" | tee -a "$OUT/repeat.jsonl" | cut -c1-160
  done
  echo "repeat: n-gram path taken by $(grep -c 'try_ngram_only_distributed ELIGIBLE' "$BASE/A/node.log") of ${REPEAT:-3} requests (expected: the first)"
  echo "repeat: score with examples/score_against_reference.py <model.gguf> $OUT/repeat.jsonl $OUT/prompt.txt"
  exit 0
fi

if [ "$MODE" = whole ]; then
  # Ask 1 — B alone: it refuses the length, A bars it, and the re-plan finds
  # nobody. The caller must hear B's limit as a 400.
  ask "$PROMPT" 120 whole_nobody | tee "$OUT/whole.jsonl"
  refused1=$(grep -c 're-planning without it' "$BASE/A/node.log")
  grep -E 'too long for|tokens, longer than' "$BASE/B/node.log" | head -1 | cut -c1-260
  # Ask 2 — C up at the default ceiling: whatever A tries first, it answers.
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  for _ in $(seq 1 60); do [ "$(peers)" -ge 2 ] && break; sleep 2; done
  sleep 5
  echo "rig: A sees $(peers) peers"
  retried0=$(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log")
  ask "$PROMPT" 120 whole_takeover | tee -a "$OUT/whole.jsonl"
  refused2=$(( $(grep -c 're-planning without it' "$BASE/A/node.log") - refused1 ))
  retried=$(( $(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log") - retried0 ))
  echo "whole: ask 1 — A re-planned after B's refusal $refused1 time(s); ask 2 — $refused2 refusal(s), $retried router retr(ies)"
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  python3 - "$OUT/whole.jsonl" "$refused1" "$refused2" "$retried" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
refused1, refused2, retried = map(int, sys.argv[2:5])
first, second = rows[0], rows[1]
msg = first.get("content") or ""
ok1 = " 400 " in first["status"] + " " and "serve at most" in msg \
    and "Raise it in Settings" not in msg and refused1 >= 1
ok2 = second["status"].endswith("200 ok") and bool(second.get("content")) \
    and retried == refused2
print(f"whole: nobody serves it -> {first['status']}  {'PASS' if ok1 else 'FAIL'}: {msg[:220]}")
tried = "B first, re-planned onto C" if refused2 else (
    "C first (B's refusal not exercised on this ask)" if second["status"].endswith("200 ok")
    else "no re-plan logged")
print(f"whole: with C up -> {second['status']}  {'PASS' if ok2 else 'FAIL'} ({tried})")
print("whole: PASS" if ok1 and ok2 else "whole: FAIL")
sys.exit(0 if ok1 and ok2 else 1)
PY
  exit $?
fi

if [ "$MODE" = failover ] || [ "$MODE" = context ] || [ "$MODE" = failover_mid ]; then
  node_id() { # port -> the 16-hex-digit id a plan prints
    curl -s -m 5 -H "Authorization: Bearer $(cat "$1")" "localhost:$2/api/admin/stats" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])'
  }
  IB=$(node_id "$BASE/B/api_key" 8920); IC=$(node_id "$BASE/C/api_key" 8940); ID=$(node_id "$BASE/D/api_key" 8960)
  IE=none; [ "$MODE" = failover_mid ] && IE=$(node_id "$BASE/E/api_key" 8980)
  # plan_is <want>: A's route preview is A→B with a composite C+D behind B
  # ("healthy"), or A→C→D ("control").
  plan_is() {
    curl -s -m 10 -H "Authorization: Bearer $KA" "localhost:8900/api/admin/models/$MODEL/pipeline-plan" \
      | python3 -c 'import sys,json
want,b,c,d,e=sys.argv[1:6]
try: p=json.load(sys.stdin)
except Exception: sys.exit(1)
seg=[s["node_id"] for s in p.get("segments",[])]
sb={s["node_id"] for s in p.get("standbys",[])}
show=lambda k: " ".join("%s%s" % (s["node_id"][:8], s["layer_range"]) for s in p.get(k,[]))
print("  plan:", show("segments"), "| standbys:", show("standbys"), file=sys.stderr)
tail = [] if e == "none" else [e]
ok = (len(seg)==2+len(tail) and seg[1]==b and seg[2:]==tail and {c,d} <= sb and b not in sb) if want=="healthy" else (seg[1:]==[c,d]+tail)
sys.exit(0 if ok else 1)' "$1" "$IB" "$IC" "$ID" "$IE"
  }
  wait_plan() { # want
    for _ in $(seq 1 60); do plan_is "$1" 2>/dev/null && { plan_is "$1"; return 0; }; sleep 3; done
    echo "failover: the plan never became '$1':"; plan_is "$1"; return 1
  }
  wait_plan healthy || exit 1
  echo "failover: plan is A→B with C+D covering B's range between them"
fi

if [ "$MODE" = context ]; then
  retried0=$(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log")
  ask "$PROMPT" 120 context_takeover | tee "$OUT/context.jsonl"
  skipped=$(grep -c 'not sending the prompt to a peer that serves a shorter' "$BASE/A/node.log")
  refused=$(grep -c 'remote segment serves a shorter conversation than this' "$BASE/A/node.log")
  taken=$(grep -c 'segment taken over by several nodes' "$BASE/A/node.log")
  gave_up=$(grep -c 'Remote segment refused the request itself' "$BASE/A/node.log")
  retried=$(( $(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log") - retried0 ))
  echo "context: A skipped B $skipped time(s), failed over from B's refusal $refused, composite takeover $taken, gave up $gave_up, router retries $retried"
  grep -E 'This conversation is [0-9]+ tokens' "$BASE/B/node.log" | head -2 | cut -c1-260
  # Nobody left who serves the length: D stopped, so B's range has no cover.
  kill "$PD"; PD=""
  for _ in $(seq 1 30); do [ "$(peers)" -eq 2 ] && break; sleep 2; done
  ask "$PROMPT" 120 context_nobody | tee -a "$OUT/context.jsonl"
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  python3 - "$OUT/context.jsonl" "$skipped" "$refused" "$taken" "$retried" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
skipped, refused, taken, retried = map(int, sys.argv[2:6])
first, second = rows[0], rows[1]
ok1 = (skipped + refused) >= 1 and taken >= 1 and retried == 0 \
    and first["status"].endswith("200 ok") and bool(first.get("content"))
msg = second.get("content") or ""
ok2 = " 400 " in second["status"] + " " and "serve at most" in msg and "Raise it in Settings" not in msg
print(f"context: with C+D up -> {first['status']}  {'PASS' if ok1 else 'FAIL'}")
print(f"context: nobody serves it -> {second['status']}  {'PASS' if ok2 else 'FAIL'}: {msg[:220]}")
print("context: PASS" if ok1 and ok2 else "context: FAIL")
sys.exit(0 if ok1 and ok2 else 1)
PY
  exit $?
fi

if [ "$MODE" = failover ] || [ "$MODE" = failover_mid ]; then

  # failover_mid takes over the FIRST request, whose plan was just checked:
  # once the planner has measured B it may route the next one A→C→D→E and
  # never touch B (seen 2026-09-25 — a kill that hit nothing and a PASS-shaped
  # 200 with no takeover). The healthy reply is judged by the control instead.
  if [ "$MODE" = failover ]; then
    ask "$PROMPT" 120 healthy | tee "$OUT/failover.jsonl"
  else
    : > "$OUT/failover.jsonl"
  fi

  # Kill B's worker the moment it starts computing: B's own processes' CPU time
  # rising above what they had at the start is the prompt pass arriving. That
  # is the only pass a composite stand-in is offered on.
  cpu_of_children() { local t=0; for p in $(pgrep -P "$PB"); do
      t=$((t + $(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo 0))); done; echo $t; }
  taken0=$(grep -c 'segment taken over by several nodes' "$BASE/A/node.log")
  retried0=$(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log")
  base=$(cpu_of_children)
  ask "$PROMPT" 120 takeover >> "$OUT/failover.jsonl" &
  ASK=$!
  until [ $(( $(cpu_of_children) - base )) -ge 30 ]; do
    kill -0 $ASK 2>/dev/null || { echo "failover: the reply finished before B started computing"; break; }
    sleep 0.02
  done
  WORKERS=$(pgrep -P "$PB")
  echo "failover: killing B's worker ${WORKERS:-<none>} at the start of its prompt pass"
  [ -n "$WORKERS" ] && kill -9 $WORKERS
  wait $ASK
  taken=$(( $(grep -c 'segment taken over by several nodes' "$BASE/A/node.log") - taken0 ))
  # A takeover that fails is rescued by the router's retry on a fresh plan —
  # through B again once its worker respawns — and THAT reply is correct. It
  # passed for a takeover that never ran its second part (gotcha #706), so a
  # retry during this arm is a failure, whatever the reply says.
  retried=$(( $(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log") - retried0 ))
  grep -E 'segment taken over by several nodes|failing over to standby node' "$BASE/A/node.log" | tail -3

  # Control: B gone, so the plan itself is A→C→D — the shape the takeover
  # should have spliced in.
  kill "$PB"; PB=""
  for _ in $(seq 1 30); do [ "$(peers)" -eq $((PEERS_EXPECTED - 1)) ] && break; sleep 2; done
  wait_plan control || exit 1
  ask "$PROMPT" 120 control >> "$OUT/failover.jsonl"
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"

  python3 - "$OUT/failover.jsonl" "$taken" "$retried" "$MODE" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
taken, retried = int(sys.argv[2]), int(sys.argv[3])
if sys.argv[4] == "failover_mid":
    rows = [{"content": None, "status": "(no healthy arm) 200 ok"}] + rows
h, t, c = (r.get("content") for r in rows[:3])
print(f"failover: takeover logged {taken} time(s), router retries {retried}; statuses", [r["status"] for r in rows])
same = t is not None and t == c
print(f"failover: takeover reply {'==' if same else '!='} A→C→D control; healthy A→B reply {'==' if h == c else '!='} control")
# Byte-equality is information, not the verdict. Two greedy runs of ONE
# topology split at the same near-tie on 2026-09-25 (cold vs warm stand-ins,
# the prompt relayed vs chained), so '!=' says nothing by itself. What would
# show a broken takeover is the REFERENCE model ranking its tokens badly:
#   examples/score_against_reference.py <model.gguf> $OUT/failover.jsonl <prompt>
# (the takeover and the control should score alike).
ok = taken >= 1 and retried == 0 and all(r["status"].endswith("200 ok") and r.get("content") for r in rows if r.get("content") is not None or not r["status"].startswith("(no"))
print("failover: PASS" if ok else "failover: FAIL")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

# kill: a long reply, broken well into its decode steps.
ask "Explain in detail how a refrigerator works, step by step." 400 kill > "$OUT/kill.jsonl" &
ASK=$!
until [ "$(grep -c 'DIAG: segment result received' "$BASE/A/node.log" 2>/dev/null)" -ge 40 ]; do
  kill -0 $ASK 2>/dev/null || { echo "kill: the reply finished before the kill point"; break; }
  sleep 0.5
done
case "${KILL:-A}" in
  A) WORKERS=$(pgrep -P "$PA") ;;
  B)
    # Land BETWEEN two of B's forwards: kill the instant B reports one done.
    done0=$(grep -c 'LayerForward processed via worker subprocess' "$BASE/B/node.log")
    until [ "$(grep -c 'LayerForward processed via worker subprocess' "$BASE/B/node.log")" -gt "$done0" ]; do sleep 0.01; done
    WORKERS=$(pgrep -P "$PB") ;;
  B_UNLOAD)
    KB=$(cat "$BASE/B/api_key")
    echo "kill: unloading $MODEL on B: $(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST \
      -H "Authorization: Bearer $KB" "localhost:8920/api/admin/models/$MODEL/unload")"
    WORKERS="" ;;
  *) echo "KILL must be A, B or B_UNLOAD"; exit 2 ;;
esac
echo "kill: ${KILL:-A} after $(grep -c 'DIAG: segment result received' "$BASE/A/node.log") segment results${WORKERS:+ (killing $WORKERS)}"
[ -n "$WORKERS" ] && kill -9 $WORKERS
wait $ASK
cat "$OUT/kill.jsonl"
