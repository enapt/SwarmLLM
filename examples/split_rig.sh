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
#          REPEAT_GAP=S waits S seconds between runs (default 0) — past the
#          pool's 5 s reclaim floor, a worker idle between turns is reclaimable.
#          RIG_TEMPERATURE=0.7 asks every question at that temperature (default 0,
#          greedy) — for speculation that must hold at the temperatures clients send.
#   (any mode) DRAFTER=<model id> also gives A that model WHOLE, from the same
#          models directory — the small model speculation across computers guesses
#          with (`pipeline::engine_drafter`). Pair it with EXTRA_TOML switching
#          `speculative_decoding` and `decentralized_spec_decoding` on; A picks it
#          itself (or name it with `draft_model`). MODEL must share its vocabulary,
#          e.g. MODEL=qwen2.5-coder-7b-instruct-q4-k-m DRAFTER=qwen2.5-0.5b-instruct-fp16.
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
#   continue  FUTURE_WORK #236: a reply whose machine fails MID-STREAM is
#          continued on another, from what its reader received. A holds the
#          header only (every plan is one peer running all of it), B and C each
#          hold every part, all on the processor. One long STREAMED reply through
#          A; once ~KILL_AFTER (default 24) chunks have arrived, the worker of the
#          peer A handed it to is killed. PASS = the stream ends with a finish and
#          no error, chunks kept arriving after the kill, the reply does not begin
#          again (its opening appears once), A logged the continuation, and the
#          other peer served it. BIN_A = v0.3.230 is the control: the stream
#          ends with an error after the kill.
#   fetch  A holds every part, B only part 0; B is asked to download part
#          FETCH_SHARD (default 1) from A over P2P. Prints whether it landed and
#          every `network event loop stalled` line B logged meanwhile — the hash
#          of a downloaded part ran ON the event loop until FUTURE_WORK #108
#          (~229 ms per 512 MB part, over the loop's 100 ms tripwire).
#   cache  the field report of 2026-09-26: an agent's SECOND turn on a node
#          that runs the model on its processor. A and B each hold every part
#          (processor by default, so A's router is consulted and nothing
#          touches the card); two turns of an agent-shaped conversation — a
#          long system prompt, tool definitions, then the same plus a reply and
#          a new question — go to A. PASS = on turn 2 A's planner logged how
#          much of the prompt its worker holds, the plan was the single local
#          segment, and the worker's `prefix-cache HIT` matched at least that
#          much. Default model qwen2.5-0.5b (it renders tools natively, like
#          the reporter's Qwen-based xLAM). Run BIN_A = an older release for
#          the baseline: no planner line, and the route is decided cold.
#   splitcache  FUTURE_WORK #10: a SPLIT keeps its conversation's prompt
#          between turns. A holds shard 0, B every other shard, both on the
#          processor; two turns of an agent-shaped conversation (a long system
#          prompt, then the same plus a reply and a new question) go to A. PASS
#          = turn 1's prompt pass kept its prompt from position 0, turn 2's
#          RESUMED past it (A's `resumed_from` > 0) and B restored its part
#          (`HIT on a split's stored prompt`), both 200. MISS=1 restarts B
#          between the turns: PASS = turn 2 logged the miss and read the prompt
#          again from 0, 200. SWARMLLM_SPLIT_PROMPT_CACHE=0 is the control arm:
#          no prompt kept, both 200 — compare the two arms' turn-2 seconds and
#          replies ($OUT/splitcache.jsonl). Default model llama-3.2-3b. SPARE=1
#          adds C holding exactly B's parts: turn 2 resumes only if the plan
#          names the peer turn 1 used again, so the same PASS asks that too.
#   remote  A holds NONE of the model (the header only) and REMOTE_NODES (2,
#          default, or 3) other nodes hold it between them in contiguous parts
#          — B the first, then C (and D) — the shape a user who stores nothing
#          meets for every split model (FUTURE_WORK #143). REPEAT (default 3)
#          greedy runs of the long prompt through A, timed. DELAY_A=ms holds
#          every tensor A sends that long (`SWARMLLM_TEST_TENSOR_DELAY_MS` on A
#          only): A far from holders that are close to each other.
#          DELAY_B=ms does the same for B, the delegate: with REMOTE_NODES=3 its
#          checks either visit C and D one at a time (`SWARMLLM_CHAIN_VERIFY=0`)
#          or travel B→C→D→B as one trip (the default). The arm is
#          chosen by the caller's environment — `SWARMLLM_DELEGATE_SPLIT=0`
#          keeps the loop on A (the control); unset, A hands the request to B,
#          which leads it. PASS = every reply 200 with content, AND the
#          mechanism the arm claims: delegated arm — A logged the hand-off and B
#          logged leading it on every request, nothing fell back; control — no
#          hand-off at all. Score both arms' replies against llama.cpp
#          (score_against_reference.py, $OUT/remote.jsonl + prompt.txt).
#   mixed  FUTURE_WORK #156: a node whose header describes ANOTHER upload than
#          its parts. A holds shard 0; B holds every shard and C every shard but
#          0, and B's gguf_header.bin is replaced (never edited — the rig's
#          files are hard links to the live node's) by
#          examples/plant_mixed_header.py's copy: by default one WITHOUT the
#          chat template, so every tensor sits a few hundred bytes earlier and
#          every read lands in bytes B holds (PLANT=3808: a longer header, the
#          error shape of gotcha #776 instead). Three greedy asks through A:
#          control (B excluded), alone (C excluded — B is the only route), both.
#          PASS = alone is NOT a 200 (v0.3.221 answers it — with garbage, which
#          is the bug), B logged "Refusing to load", and both answers exactly
#          what control did. Canonical healing is off on every node here, so
#          nothing repairs B's header mid-run.
#   disputed  a node serving a part whose bytes are NOT what its manifest says
#          — the state a peer's gossip leaves behind (#382 adopts a
#          contradicting hash for a held part; the re-check keeps the bytes).
#          Same topology as `mixed`; B's LAST part is a corrupted COPY (never
#          the hard link) under the unchanged manifest, so B's startup sweep
#          finds the disagreement and keeps the bytes. v0.3.222 then announced
#          the part under the MANIFEST's build: A counted B as a holder of the
#          right bytes and routed the layers there — garbage. PASS = B recorded
#          the dispute, A counts B as another build, alone (C excluded) is NOT
#          a 200, and both answers exactly what control did.
#   spliced  a node whose own manifest VOUCHES for parts that are not the
#          upload's bytes — #775's splice (part i of one upload cut at another's
#          offsets), the shape left on peers from before v0.3.221. B holds every
#          part; its LAST part is zeroed (a copy, never the hard link) and B's
#          manifest is rewritten to that part's real hash, so every hash check
#          passes. Only B runs the canonical heal (HuggingFace needed): it finds
#          the part's bytes are not the upload's. B is asked DIRECTLY the
#          moment its heal says so — on v0.3.222 B answers from the wrong bytes
#          (garbage) while it stages a replacement; fixed, B has DELETED the part
#          and is fetching the upload's, so it cannot answer from it — and again
#          once B holds the upload's copy. PASS = B deleted the part, the ask
#          during the repair is not answered from the wrong bytes, B's part is
#          then byte-identical to the upload's, and the ask after answers.
#   disconnect  a client that hangs up on a NON-streamed split reply stops the
#          work. A holds shard 0, B the rest, both on the processor; one long
#          greedy ask through A whose client gives up after HANGUP_S (default 8)
#          seconds, then a short ask. PASS = A logged `client disconnected
#          before completion` for that request, at most 2 of its decode steps
#          came back from B more than 3 s after the hang-up, and the short ask
#          answered 200. Until 2026-10-10 the path a split request takes
#          (`dispatch_inference`) armed no guard and the reply ran on to its end
#          (FUTURE_WORK #220's rig): an older release is the control arm, and
#          should FAIL. Default model tinyllama.
#
# usage: split_rig.sh split|kill|failover|failover_mid|context|whole|continue|repeat|fetch|cache|splitcache|remote|mixed|disputed|spliced|disconnect <binary> [<binary for B>]
#   MODEL      model id (default: tinyllama for split, llama-3.2-3b for kill
#              and failover)
#   SHARDS_A   shard indices A holds, comma-separated (default 0; kill: 0,LAST)
#   SHARDS_B   shard indices B holds (default: every shard A lacks; kill: all)
#   GPU_A/B    SWARMLLM_INFERENCE_GPU_LAYERS for each node ("" = auto)
#   CPUS_A/B/… logical CPUs that node runs on (`taskset -c`, e.g. CPUS_A=0-7
#              CPUS_B=8-15): two nodes on disjoint halves of the machine stand in
#              for two computers when work across them overlaps (#171)
#   MEM_C/D/E  failover modes: a memory limit for that standby (e.g. 6G) — it runs
#              in a `systemd-run` scope with that MemoryMax (CPUQuota RIG_SCOPE_CPU,
#              default 400%) and reads it as its machine, so the room it advertises
#              follows its own use, not the machine's free RAM (#162 b)
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

MODE="${1:?usage: split_rig.sh split|kill|failover|context|repeat|fetch|cache|remote|mixed <binary> [<binary for B>]}"
BIN_A="${2:?binary}"
BIN_B="${3:-$BIN_A}"
case "$MODE" in split|kill|failover|failover_mid|context|whole|continue|repeat|fetch|cache|splitcache|remote|mixed|disputed|spliced|disconnect) ;; *) echo "mode must be split, kill, failover, failover_mid, context, whole, continue, repeat, fetch, cache, splitcache, remote, mixed, disputed, spliced or disconnect"; exit 2 ;; esac
if { [ "$MODE" = context ] || [ "$MODE" = whole ]; } && printf '%s' "${EXTRA_TOML:-}" | grep -q '^\[inference\]'; then
  echo "$MODE: EXTRA_TOML may not open [inference] — B's ceiling is written there"; exit 2
fi
[ -x "$BIN_A" ] && [ -x "$BIN_B" ] || { echo "binary not executable"; exit 2; }
if [ "$MODE" = split ] || [ "$MODE" = mixed ] || [ "$MODE" = disputed ] || [ "$MODE" = spliced ] || [ "$MODE" = disconnect ]; then
  MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
elif [ "$MODE" = cache ]; then
  MODEL="${MODEL:-qwen2.5-0.5b-instruct-fp16}"
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
if [ "$MODE" = split ] || [ "$MODE" = repeat ] || [ "$MODE" = disconnect ]; then
  SHARDS_A="${SHARDS_A:-0}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vxF -f <(echo "$SHARDS_A" | tr ',' '\n') | paste -sd,)}"
  # Processor unless asked otherwise: the reference is scored on the processor.
  [ "$MODE" = repeat ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
  [ "$MODE" = disconnect ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
elif [ "$MODE" = splitcache ]; then
  SHARDS_A="${SHARDS_A:-0}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vxF -f <(echo "$SHARDS_A" | tr ',' '\n') | paste -sd,)}"
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = cache ]; then
  SHARDS_A=$(echo "$SHARDS" | paste -sd,)
  SHARDS_B=$SHARDS_A
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
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
  # The takeover arm sends the healthy arm's prompt again, so with the split
  # prompt cache (FUTURE_WORK #10) it RESUMES from what B stored — and a failed
  # resumed pass is read again from position 0 through the same plan, by
  # design: B's respawned worker answers and no stand-in is ever asked (seen
  # 2026-10-09, mistaken at first for the kill's timing). This mode tests the
  # takeover, so the coordinator keeps no prompt; `splitcache MISS=1` tests the
  # resumed pass's own failure.
  [ "$MODE" = failover ] && export SWARMLLM_SPLIT_PROMPT_CACHE="${SWARMLLM_SPLIT_PROMPT_CACHE:-0}"
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
elif [ "$MODE" = remote ]; then
  RN="${REMOTE_NODES:-2}"
  { [ "$RN" = 2 ] || [ "$RN" = 3 ]; } || { echo "REMOTE_NODES must be 2 or 3"; exit 2; }
  [ "$N" -ge "$RN" ] || { echo "remote across $RN nodes needs at least $RN shard files here; $MODEL has $N"; exit 2; }
  # A holds the header only; the shards go to B, C (and D) in contiguous runs,
  # as even as the shard count allows.
  SHARDS_A=""
  part() { echo "$SHARDS" | awk -v n="$N" -v rn="$RN" -v want="$1" '{ if (int((NR-1)*rn/n) == want) print }' | paste -sd,; }
  SHARDS_B=$(part 0); SHARDS_C=$(part 1)
  [ "$RN" = 3 ] && SHARDS_D=$(part 2)
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = spliced ]; then
  SHARDS_A=0
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
  # A never heals; B does, and is started with this unset.
  export SWARMLLM_CANONICAL_UPLOADS=0
elif [ "$MODE" = whole ]; then
  # A holds the header only, so it knows the model's declared context but can
  # run none of it; B and C each hold all of it.
  SHARDS_A=""
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
  SHARDS_C=$SHARDS_B
  GPU_A="${GPU_A:-0}"
elif [ "$MODE" = continue ]; then
  # As `whole`, with both holders at the default ceiling and on the processor.
  SHARDS_A=""
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
  SHARDS_C=$SHARDS_B
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; GPU_C="${GPU_C:-0}"
elif [ "$MODE" = mixed ] || [ "$MODE" = disputed ]; then
  # B and C both hold A's missing range; only B's header is from "another
  # upload". B holds EVERY part: a header shifted earlier then reads, for its
  # first tensors, bytes B holds (the part before, or the header) — nothing
  # fails and the model computes on the wrong bytes, which is the silent shape
  # this exists to catch. Processor everywhere, so the replies compare exactly.
  SHARDS_A=0
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
  SHARDS_C=$(echo "$SHARDS" | grep -vx 0 | paste -sd,)
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
  # Healing would replace B's planted header (or part) from HuggingFace mid-run.
  export SWARMLLM_CANONICAL_UPLOADS=0
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
  # The shard-0 sidecars travel with the header, as on a real node that fetched
  # any shard from HuggingFace: a later-layer node needs rope_freqs.bin on a
  # Llama 3 model (#124) and tied_output_weight.bin on a weight-tied one.
  for f in gguf_header.bin hf_source.json manifest.json tied_output_weight.bin rope_freqs.bin; do
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
  local extra="${EXTRA_TOML:-}"
  # failover_mid needs chaining OFF: chained, a crashed middle segment is
  # re-run on the SAME node first (`chained run failed — re-running this
  # segment unchained`), B respawns its worker and serves it, and the composite
  # takeover this mode exists to see never happens — a FAIL with nothing wrong.
  # It was the caller's job (gate212.sh passed it); a run that forgot it
  # failed both arms on 2026-09-29. Merged into the caller's [inference], if any.
  if [ "$MODE" = failover_mid ] && ! printf '%s' "$extra" | grep -q 'pipeline_chaining'; then
    if printf '%s' "$extra" | grep -q '^\[inference\]'; then
      extra=$(printf '%s' "$extra" | sed 's/^\[inference\]$/[inference]\npipeline_chaining = false/')
    else
      extra=$(printf '%s\n[inference]\npipeline_chaining = false' "$extra")
    fi
  fi
  [ -n "$extra" ] && printf '\n%s\n' "$extra" >> "$d/config.toml"
  return 0
}
start() { # dir port bin gpu [memory limit]
  # A memory limit runs the node in a scope of its own, which the node reads as
  # its machine (`daemon::machine_memory`): its room then follows its own use,
  # not the free RAM of a machine three other rig nodes share (#162 b). The
  # scope sits outside a gate's safety scope, so it carries a CPU quota too.
  local scope=()
  [ -n "${5:-}" ] && scope=(systemd-run --user --scope --quiet -p "MemoryMax=$5" -p MemorySwapMax=0
    -p "CPUQuota=${RIG_SCOPE_CPU:-400%}" --)
  # CPUS_A / CPUS_B / …: the logical CPUs that node (and its workers, which
  # inherit it) may run on — two nodes on disjoint halves of this machine stand
  # in for two computers, where work that overlaps does not share cores.
  local cpus_var="CPUS_$(basename "$1")" pin=()
  [ -n "${!cpus_var:-}" ] && pin=(taskset -c "${!cpus_var}")
  if [ -n "$4" ]; then
    SWARMLLM_INFERENCE_GPU_LAYERS="$4" SWARMLLM_NODE_DATA_DIR="$1" "${scope[@]}" "${pin[@]}" "$3" run -p "$2" > "$1/node.log" 2>&1 &
  else
    SWARMLLM_NODE_DATA_DIR="$1" "${scope[@]}" "${pin[@]}" "$3" run -p "$2" > "$1/node.log" 2>&1 &
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
if [ -n "${DRAFTER:-}" ]; then
  DSRC="$MODELS_DIR/$DRAFTER"
  [ -f "$DSRC/manifest.json" ] || { echo "no manifest for drafter $DRAFTER in $MODELS_DIR"; exit 2; }
  mkdir -p "$BASE/A/models/$DRAFTER"
  for f in "$DSRC"/*; do ln "$f" "$BASE/A/models/$DRAFTER/" || { echo "cannot link $f"; exit 2; }; done
fi
# The n-gram-only path takes every request a coordinator holding the header can
# tokenize, and it is a SEGMENT path (its own failover covers #111 there). The
# whole-model path under test is what a coordinator runs once that path is off
# for it — no header, or its payoff check has switched it off.
{ [ "$MODE" = whole ] || [ "$MODE" = continue ]; } && printf '\n[inference]\nngram_lookup_enabled = false\n' >> "$BASE/A/config.toml"
# DELAY_A: A far from the others — every tensor it sends waits this long.
PA=$(SWARMLLM_TEST_TENSOR_DELAY_MS="${DELAY_A:-0}" start "$BASE/A" 8900 "$BIN_A" "${GPU_A:-}")
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
if [ "$MODE" = mixed ]; then
  # Unlink first: B's header is a hard link to the live node's file, and
  # writing through it would plant the bad header THERE.
  # PLANT=shorter (default): reads land EARLIER, inside bytes B holds — the
  # silent garbage of #156. PLANT=<bytes>: a longer header, reads land later and
  # the last tensor runs off the end — an error (#776's shape).
  rm -f "$BASE/B/models/$MODEL/gguf_header.bin"
  python3 "$(dirname "$0")/plant_mixed_header.py" "$SRC/gguf_header.bin" \
    "$BASE/B/models/$MODEL/gguf_header.bin" "${PLANT:-shorter}" || exit 2
  cmp -s "$SRC/gguf_header.bin" "$BASE/B/models/$MODEL/gguf_header.bin" \
    && { echo "mixed: the planted header is identical to the real one"; exit 2; }
fi
if [ "$MODE" = disputed ]; then
  # A COPY, never the hard link: writing through it would corrupt the live
  # node's part. Zeroed from 1 MiB to 1 MiB before the end, size unchanged.
  PART="$BASE/B/models/$MODEL/$(printf 'shard_%03d.bin' "$LAST")"
  rm -f "$PART"
  cp "$SRC/$(printf 'shard_%03d.bin' "$LAST")" "$PART" || exit 2
  SZ=$(stat -c %s "$PART")
  dd if=/dev/zero of="$PART" bs=1M seek=1 count=$(( SZ / 1048576 - 2 )) conv=notrunc status=none || exit 2
  [ "$(stat -c %s "$PART")" = "$SZ" ] || { echo "disputed: the corrupted part changed size"; exit 2; }
  cmp -s "$SRC/$(printf 'shard_%03d.bin' "$LAST")" "$PART" && { echo "disputed: the part is unchanged"; exit 2; }
fi
if [ "$MODE" = spliced ]; then
  # B's heal writes beside its parts (hf_source.json, the header, the manifest):
  # COPIES, never the hard links, or it would write into the live node's files.
  M="$BASE/B/models/$MODEL"
  for f in gguf_header.bin hf_source.json manifest.json; do
    [ -f "$M/$f" ] && { rm -f "$M/$f"; cp "$SRC/$f" "$M/$f"; }
  done
  PART="$M/$(printf 'shard_%03d.bin' "$LAST")"
  rm -f "$PART"
  cp "$SRC/$(printf 'shard_%03d.bin' "$LAST")" "$PART" || exit 2
  SZ=$(stat -c %s "$PART")
  dd if=/dev/zero of="$PART" bs=1M count=$(( SZ / 1048576 )) conv=notrunc status=none || exit 2
  # The manifest vouches for these bytes: their real hash, and a zero
  # manifest_hash (a locally loaded manifest may carry none).
  python3 - "$M/manifest.json" "$PART" "$LAST" <<'PY' || exit 2
import json, sys, blake3
path, part, last = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = json.load(open(path))
h = blake3.blake3(open(part, 'rb').read()).digest()
for s in m['shards']:
    if s['index'] == last:
        s['hash'] = list(h)
m['manifest_hash'] = [0] * 32
json.dump(m, open(path, 'w'))
print(f"spliced: B's part {last} zeroed, manifest hash {h.hex()[:16]}")
PY
fi
# B alone serves a conversation shorter than the long prompt.
{ [ "$MODE" = context ] || [ "$MODE" = whole ]; } && printf '\n[inference]\nmax_seq_len_override = %s\n' "${CEIL_B:-512}" >> "$BASE/B/config.toml"
# DELAY_B: B far from the nodes after it (remote mode: the delegate leading [B, C, D]).
[ "$MODE" = spliced ] && unset SWARMLLM_CANONICAL_UPLOADS
PB=$(SWARMLLM_TEST_TENSOR_DELAY_MS="${DELAY_B:-0}" start "$BASE/B" 8920 "$BIN_B" "${GPU_B:-}")
up "$BASE/B" 8920 || exit 1
PEERS_EXPECTED=1
if [ "$MODE" = continue ]; then
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  PEERS_EXPECTED=2
fi
if [ "$MODE" = splitcache ] && [ -n "${SPARE:-}" ]; then
  # A second peer that could run B's segment: what the plan chooses between.
  make_node "$BASE/C" "$SHARDS_B" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  PEERS_EXPECTED=2
  echo "rig: C=[$SHARDS_B] (a spare holding B's parts) gpu=${GPU_C:-0}"
fi
if [ "$MODE" = failover ] || [ "$MODE" = context ] || [ "$MODE" = failover_mid ]; then
  # Processor only unless asked otherwise (all four nodes): four daemons on
  # one card is #104's setup, and a KV refusal there would read as a failover
  # result.
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}" "${MEM_C:-}")
  make_node "$BASE/D" "$SHARDS_D" "\"$ADDR\""
  PD=$(start "$BASE/D" 8960 "$BIN_A" "${GPU_D:-0}" "${MEM_D:-}")
  up "$BASE/C" 8940 || exit 1
  up "$BASE/D" 8960 || exit 1
  PEERS_EXPECTED=3
  echo "rig: C=[$SHARDS_C] D=[$SHARDS_D] gpu=${GPU_C:-0}/${GPU_D:-0} memory=${MEM_C:-machine}/${MEM_D:-machine}"
  if [ "$MODE" = failover_mid ]; then
    make_node "$BASE/E" "$SHARDS_E" "\"$ADDR\""
    PE=$(start "$BASE/E" 8980 "$BIN_A" "${GPU_E:-0}" "${MEM_E:-}")
    up "$BASE/E" 8980 || exit 1
    PEERS_EXPECTED=4
    echo "rig: E=[$SHARDS_E] gpu=${GPU_E:-0}"
  fi
fi
if [ "$MODE" = mixed ] || [ "$MODE" = disputed ]; then
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  PEERS_EXPECTED=2
  if [ "$MODE" = mixed ]; then
    echo "rig: C=[$SHARDS_C] (the real header) gpu=${GPU_C:-0}; B's header is planted"
  else
    echo "rig: C=[$SHARDS_C] (real parts) gpu=${GPU_C:-0}; B's part $LAST is corrupted under its manifest"
  fi
fi
if [ "$MODE" = remote ]; then
  make_node "$BASE/C" "$SHARDS_C" "\"$ADDR\""
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  PEERS_EXPECTED=2
  if [ "$RN" = 3 ]; then
    make_node "$BASE/D" "$SHARDS_D" "\"$ADDR\""
    PD=$(start "$BASE/D" 8960 "$BIN_A" "${GPU_D:-0}")
    up "$BASE/D" 8960 || exit 1
    PEERS_EXPECTED=3
  fi
  echo "rig: remote across $RN nodes — C=[$SHARDS_C]${SHARDS_D:+ D=[$SHARDS_D]}, tensors delayed A ${DELAY_A:-0} / B ${DELAY_B:-0} ms, SWARMLLM_DELEGATE_SPLIT=${SWARMLLM_DELEGATE_SPLIT:-<unset: delegate>} SWARMLLM_CHAIN_VERIFY=${SWARMLLM_CHAIN_VERIFY:-<unset: chain>}"
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
  # RIG_ROUTE: a `swarm_route` object for this ask (e.g. nodes to exclude).
  body=$(python3 -c 'import json,os,sys
b={"model":sys.argv[1],"max_tokens":int(sys.argv[3]),"temperature":float(os.environ.get("RIG_TEMPERATURE","0")),"messages":[{"role":"user","content":sys.argv[2]}]}
if os.environ.get("RIG_ROUTE"): b["swarm_route"]=json.loads(os.environ["RIG_ROUTE"])
print(json.dumps(b))' "$MODEL" "$1" "$2")
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

if [ "$MODE" = disconnect ]; then
  long=$(python3 -c 'import json,sys
print(json.dumps({"model":sys.argv[1],"max_tokens":600,"temperature":0,"messages":[{"role":"user","content":
  "Write a long, detailed essay on the history of bridges, from rope bridges to suspension bridges, in many paragraphs."}]}))' "$MODEL")
  asked=$(date -u +%Y-%m-%dT%H:%M:%S.%6N)
  curl -s -m "${HANGUP_S:-8}" -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
       -X POST localhost:8900/v1/chat/completions -d "$long" -o /dev/null
  echo "disconnect: the client hung up after ${HANGUP_S:-8} s (curl exit $?)"
  gone=$(date -u +%Y-%m-%dT%H:%M:%S.%6N)
  sleep 20
  # Timed on /proc/uptime: WSL2's wall clock steps (back 1 s mid-request on
  # 2026-10-10), and a wall-clock difference then reads negative.
  read -r t1 _ < /proc/uptime
  ask "What is the capital of France? Answer in one sentence." 16 after > "$OUT/after.json"
  read -r t2 _ < /proc/uptime
  after_s=$(python3 -c "print(round($t2 - $t1, 1))")
  python3 - "$BASE/A/node.log" "$asked" "$gone" "$OUT/after.json" "$after_s" <<'PY'
import json, re, sys
from datetime import datetime, timedelta
log, asked, gone, after, after_s = sys.argv[1:6]
ts = lambda s: datetime.fromisoformat(s[:26])
asked, gone = ts(asked), ts(gone)
rid, guard, steps_before, steps_after = None, False, 0, 0
for line in open(log, errors="replace"):
    m = re.match(r"(\S+)Z\s", line)
    if not m:
        continue
    t = ts(m.group(1))
    if t < asked:
        continue
    if rid is None and "Queued inference request" in line:
        rid = re.search(r"request_id=(\S+)", line).group(1)
    if rid and rid in line and "client disconnected before completion" in line:
        guard = True
    if rid and rid in line and "dispatcher received LayerResult" in line:
        if t > gone + timedelta(seconds=3):
            steps_after += 1
        elif t <= gone:
            steps_before += 1
status = json.load(open(after))["status"]
print(f"disconnect: request {rid}: guard line {'LOGGED' if guard else 'absent'}; decode steps from B "
      f"{steps_before} before the hang-up, {steps_after} more than 3 s after it")
print(f"disconnect: the short ask after: {status} in {after_s} s")
ok = guard and steps_before > 0 and steps_after <= 2 and " 200" in status
if steps_before == 0:
    print("disconnect: no decode step came back before the hang-up — raise HANGUP_S; this run tests nothing")
print("disconnect: PASS" if ok else "disconnect: FAIL")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

if [ "$MODE" = mixed ]; then
  id8() { curl -s -m 5 -H "Authorization: Bearer $(cat "$1/api_key")" "localhost:$2/v1/status" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:8])'; }
  IB=$(id8 "$BASE/B" 8920); IC=$(id8 "$BASE/C" 8940)
  Q="Write a short Python function that returns the factorial of n."
  RIG_ROUTE="{\"exclude_nodes\":[\"$IB\"]}" ask "$Q" 48 control | tee "$OUT/mixed.jsonl"
  RIG_ROUTE="{\"exclude_nodes\":[\"$IC\"]}" ask "$Q" 48 alone | tee -a "$OUT/mixed.jsonl"
  ask "$Q" 48 both | tee -a "$OUT/mixed.jsonl"
  refused=$(grep -c "Refusing to load: this model's header and its tensor table" "$BASE/B/node.log")
  # The coordinator's half: it read B's refusal as "B cannot give those bytes"
  # and retracted B's claim, so no later plan comes back to it.
  retracted=$(grep -c "Retracted stale shard-holder claim after a missing-shard error.*holder=$IB" "$BASE/A/node.log")
  python3 - "$OUT/mixed.jsonl" "$refused" "$retracted" <<'PY'
import json, sys
control, alone, both = (json.loads(l) for l in open(sys.argv[1]))
refused, retracted = int(sys.argv[2]), int(sys.argv[3])
ok = lambda r: r["status"].endswith("200 ok")
print(f"mixed: control {control['status']}  route={control['route']}")
print(f"mixed: alone   {alone['status']}  route={alone['route']}  content={str(alone['content'])[:120]!r}")
print(f"mixed: both    {both['status']}  route={both['route']}")
print(f"mixed: B logged the refusal {refused} time(s); A retracted B {retracted} time(s)")
checks = {
    "control answered": ok(control),
    "alone was NOT answered (B is the only route, and its copy is mixed)": not ok(alone),
    "B refused to load its mixed copy": refused > 0,
    "A retracted B's claim": retracted > 0,
    # A's own copy is fine: the caller must not be told theirs is mixed.
    "the caller was not told its own copy is mixed": "Mixed model copy" not in str(alone["content"]),
    "both answered exactly what control did": ok(both) and both["content"] == control["content"],
}
for name, passed in checks.items():
    print(f"mixed:   {'ok  ' if passed else 'FAIL'} {name}")
print("mixed: PASS" if all(checks.values()) else "mixed: FAIL")
sys.exit(0 if all(checks.values()) else 1)
PY
  exit $?
fi

if [ "$MODE" = disputed ]; then
  id8() { curl -s -m 5 -H "Authorization: Bearer $(cat "$1/api_key")" "localhost:$2/v1/status" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:8])'; }
  IB=$(id8 "$BASE/B" 8920); IC=$(id8 "$BASE/C" 8940)
  other_build() { curl -s -m 10 -H "Authorization: Bearer $KA" localhost:8900/api/admin/models \
    | python3 -c "import sys,json; print(next((m.get('peers_other_build') or 0 for m in json.load(sys.stdin) if m['id']=='$MODEL'), 0))"; }
  # B's startup sweep hashes every part it holds; wait until it has kept the
  # corrupted one, then past two broadcast ticks so its announcement is out.
  for _ in $(seq 1 90); do grep -q "keeping bytes the swarm disagrees with" "$BASE/B/node.log" && break; sleep 2; done
  disputed=$(grep -c "keeping bytes the swarm disagrees with" "$BASE/B/node.log")
  sleep 70
  OB=$(other_build)
  echo "disputed: A counts $OB holder(s) of another build of $MODEL (B=$IB, C=$IC)"
  Q="Write a short Python function that returns the factorial of n."
  RIG_ROUTE="{\"exclude_nodes\":[\"$IB\"]}" ask "$Q" 48 control | tee "$OUT/disputed.jsonl"
  RIG_ROUTE="{\"exclude_nodes\":[\"$IC\"]}" ask "$Q" 48 alone | tee -a "$OUT/disputed.jsonl"
  ask "$Q" 48 both | tee -a "$OUT/disputed.jsonl"
  grep -a "keeping bytes the swarm disagrees with" "$BASE/B/node.log" | head -2 | cut -c1-260 | sed 's/^/disputed: B: /'
  python3 - "$OUT/disputed.jsonl" "$disputed" "${OB:-0}" <<'PY'
import json, sys
control, alone, both = (json.loads(l) for l in open(sys.argv[1]))
disputed, other = int(sys.argv[2]), int(sys.argv[3])
ok = lambda r: r["status"].endswith("200 ok")
print(f"disputed: control {control['status']}  route={control['route']}  content={str(control['content'])[:80]!r}")
print(f"disputed: alone   {alone['status']}  route={alone['route']}  content={str(alone['content'])[:120]!r}")
print(f"disputed: both    {both['status']}  route={both['route']}")
checks = {
    "control answered": ok(control),
    "B recorded the dispute (its bytes are not its manifest's)": disputed > 0,
    "A counts B as a holder of ANOTHER build": other > 0,
    "alone was NOT answered (B is the only route, and its bytes are wrong)": not ok(alone),
    "both answered exactly what control did": ok(both) and both["content"] == control["content"],
}
for name, passed in checks.items():
    print(f"disputed:   {'ok  ' if passed else 'FAIL'} {name}")
print("disputed: PASS" if all(checks.values()) else "disputed: FAIL")
sys.exit(0 if all(checks.values()) else 1)
PY
  exit $?
fi

if [ "$MODE" = spliced ]; then
  KB=$(cat "$BASE/B/api_key")
  askb() { # label — straight to B, which holds the whole model
    curl -s -m 600 -D "$OUT/$1.hdr" -H "Authorization: Bearer $KB" -H "Content-Type: application/json" \
      -X POST localhost:8920/v1/chat/completions -o "$OUT/$1.body" \
      -d "{\"model\":\"$MODEL\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one sentence.\"}]}"
    python3 -c 'import json,sys
h=open(sys.argv[1]).read().lower()
status=h.splitlines()[0].strip() if h else "no response"
raw=open(sys.argv[2]).read()
try: c=json.loads(raw)["choices"][0]["message"].get("content")
except Exception: c="ERR "+raw[:300]
print(json.dumps({"status":status,"content":c}))' "$OUT/$1.hdr" "$OUT/$1.body"
  }
  holding() { curl -s -m 10 -H "Authorization: Bearer $KB" localhost:8920/api/admin/models \
    | python3 -c "import sys,json; m=next((m for m in json.load(sys.stdin) if m['id']=='$MODEL'), {}); print(((m.get('shared_copy') or {}).get('this_computer') or {}).get('state','?'))"; }
  # Both arms ask at the same moment: right after B's heal finds its copy is
  # not the upload's (v0.3.222 logs the first line, the fix the second).
  VERDICT="same size as the canonical upload's but not its bytes|parts on this node are not the canonical upload's bytes"
  for _ in $(seq 1 240); do grep -qE "$VERDICT" "$BASE/B/node.log" && break; sleep 1; done
  found=$(grep -cE "$VERDICT" "$BASE/B/node.log")
  sleep 1
  echo "spliced: B's heal found the copy is not the upload's: $found; B holds: $(holding)"
  askb during | tee "$OUT/spliced.jsonl"
  echo "spliced: B holds after the ask: $(holding)"
  # Done when B holds the part again at its full size and its heal has judged
  # the copy (the remaining parts read `canonical` before the part is back, so
  # that alone is not the end).
  PARTF=$(printf 'shard_%03d.bin' "$LAST")
  WANT=$(stat -c %s "$SRC/$PARTF")
  T0=$(date +%s)
  for _ in $(seq 1 120); do
    [ "$(stat -c %s "$BASE/B/models/$MODEL/$PARTF" 2>/dev/null)" = "$WANT" ] && [ "$(holding)" = canonical ] && break
    sleep 5
  done
  echo "spliced: B holds: $(holding); part back after $(( $(date +%s) - T0 )) s"
  sleep 5
  askb after | tee -a "$OUT/spliced.jsonl"
  deleted=$(grep -c "deleted this node's parts that are not the canonical upload's" "$BASE/B/node.log")
  grep -aE "Fetching from the model's origin|P2P shard download complete|deleted this node's parts" "$BASE/B/node.log" | cut -c1-220 | sed 's/^/spliced: B: /'
  same=$(python3 -c 'import sys,blake3
h=lambda p: blake3.blake3(open(p,"rb").read()).hexdigest()
print(int(h(sys.argv[1])==h(sys.argv[2])))' "$SRC/$PARTF" "$BASE/B/models/$MODEL/$PARTF" 2>/dev/null || echo 0)
  python3 - "$OUT/spliced.jsonl" "$found" "$deleted" "$same" <<'PY'
import json, sys
during, after = (json.loads(l) for l in open(sys.argv[1]))
found, deleted, same = (int(x) for x in sys.argv[2:5])
ok = lambda r: r["status"].endswith("200 ok")
print(f"spliced: during {during['status']}  content={str(during['content'])[:120]!r}")
print(f"spliced: after  {after['status']}  content={str(after['content'])[:120]!r}")
print(f"spliced: heal verdicts {found}, deletions {deleted}, B's part is the upload's: {bool(same)}")
checks = {
    "B's heal found its copy is not the upload's": found > 0,
    "B deleted the part": deleted > 0,
    "the ask during the repair was NOT answered from the wrong bytes":
        not ok(during) or during["content"] == after["content"],
    "B's part is now byte-identical to the upload's": same == 1,
    "the ask after the repair answered": ok(after) and bool(after["content"]),
}
for name, passed in checks.items():
    print(f"spliced:   {'ok  ' if passed else 'FAIL'} {name}")
print("spliced: PASS" if all(checks.values()) else "spliced: FAIL")
sys.exit(0 if all(checks.values()) else 1)
PY
  exit $?
fi

# ~560 prompt tokens: long enough to catch a prompt pass mid-way (failover), and
# the prompt the #106 reference scores were taken on (repeat).
PROMPT="Here are some notes on household appliances. $(for i in $(seq 1 12); do printf 'A refrigerator moves heat from its inside to the room using a refrigerant that evaporates in the cold coils and condenses in the warm ones; the compressor drives the cycle and the thermostat decides when it runs. '; done)Using only these notes, explain step by step how a refrigerator keeps food cold."

if [ "$MODE" = cache ]; then
  # Two turns of an agent conversation: a long system prompt and tool schemas
  # sent unchanged every turn, then the history grows. Written to files and
  # posted as-is, so the tools reach the template exactly as a harness sends
  # them.
  python3 - "$MODEL" "$OUT" <<'PY'
import json, sys
model, out = sys.argv[1], sys.argv[2]
system = "You are a careful coding agent working in a user's repository. " + " ".join(
    f"Rule {i}: read the file before you change it, keep every change small, run the tests after "
    f"each change, and report exactly what you did and what you did not do." for i in range(1, 61))
tools = [{"type": "function", "function": {"name": n, "description": d,
          "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"},
                         "content": {"type": "string", "description": "text to write"}}, "required": ["path"]}}}
         for n, d in [("read_file", "Read a file and return its text."),
                      ("write_file", "Write text to a file, replacing it."),
                      ("list_dir", "List the entries of a directory."),
                      ("run_tests", "Run the project's test suite and return the summary.")]]
turn1 = [{"role": "system", "content": system}, {"role": "user", "content": "Say hello in one sentence."}]
turn2 = turn1 + [{"role": "assistant", "content": "Hello! I am ready to help with your repository."},
                 {"role": "user", "content": "Create the file notes.txt containing the word done."}]
for name, msgs in (("turn1", turn1), ("turn2", turn2)):
    json.dump({"model": model, "max_tokens": 24, "temperature": 0, "messages": msgs, "tools": tools},
              open(f"{out}/{name}.json", "w"))
PY
  for t in turn1 turn2; do
    # Turn 2's lines are the ones after this — turn 1 logs a local plan too.
    [ "$t" = turn2 ] && FROM=$(wc -l < "$BASE/A/node.log")
    curl -s -m 900 -D "$OUT/$t.hdr" -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
         -X POST localhost:8900/v1/chat/completions --data-binary "@$OUT/$t.json" -o "$OUT/$t.body"
    echo "cache: $t -> $(head -1 "$OUT/$t.hdr" | tr -d '\r')  route=$(grep -i '^x-swarm-route' "$OUT/$t.hdr" | cut -d' ' -f2- | tr -d '\r')"
  done
  python3 - "$BASE/A/node.log" "$OUT" "$FROM" <<'PY'
import json, re, sys
log = open(sys.argv[1], errors="replace").read().splitlines()[int(sys.argv[3]):]
out = sys.argv[2]
body = json.load(open(f"{out}/turn2.body"))
usage = body.get("usage", {})
def last(pat):
    hits = [l for l in log if re.search(pat, l)]
    return hits[-1] if hits else ""
credit = last(r"already holds the start of this prompt")
local = last(r"single-segment plan names this node")
# The worker's own line, from either admission path (`handle_generate` or the
# batched slot) — both go through `PrefixCache::lookup`.
hit = last(r"DIAG: prefix-cache HIT model_key")
num = lambda line, key: int(m.group(1)) if (m := re.search(key + r"=(\d+)", line)) else None
cached, matched = num(credit, "cached_locally"), num(hit, "matched_tokens")
print(f"cache: turn 2 prompt_tokens={usage.get('prompt_tokens')}  planner credit={cached}  worker HIT matched={matched}")
print(f"cache: planner line: {credit[:220] or '(none)'}")
print(f"cache: local plan:   {local[:160] or '(none)'}")
ok = bool(cached) and bool(local) and matched is not None and matched >= cached
print("cache: PASS" if ok else "cache: FAIL")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

if [ "$MODE" = splitcache ]; then
  # Two turns of an agent-shaped conversation through the split A -> B: a long
  # system prompt sent unchanged every turn, then the history grows.
  python3 - "$MODEL" "$OUT" <<'PY'
import json, sys
model, out = sys.argv[1], sys.argv[2]
system = "You are a careful coding agent working in a user's repository. " + " ".join(
    f"Rule {i}: read the file before you change it, keep every change small, run the tests after "
    f"each change, and report exactly what you did and what you did not do." for i in range(1, 61))
turn1 = [{"role": "system", "content": system}, {"role": "user", "content": "Say hello in one sentence."}]
turn2 = turn1 + [{"role": "assistant", "content": "Hello! I am ready to help with your repository."},
                 {"role": "user", "content": "Name two of the rules above, one sentence each."}]
for name, msgs in (("turn1", turn1), ("turn2", turn2)):
    json.dump({"model": model, "max_tokens": 32, "temperature": 0, "messages": msgs},
              open(f"{out}/{name}.json", "w"))
PY
  ask_turn() { # name
    local t
    t=$(curl -s -m 900 -o "$OUT/$1.body" -D "$OUT/$1.hdr" -w '%{time_total}' -H "Authorization: Bearer $KA" \
      -H "Content-Type: application/json" -X POST localhost:8900/v1/chat/completions --data-binary "@$OUT/$1.json")
    python3 - "$OUT" "$1" "$t" <<'PY' | tee -a "$OUT/splitcache.jsonl"
import json, sys
out, name, t = sys.argv[1:4]
status = (open(f"{out}/{name}.hdr").read().splitlines() or ["no response"])[0].strip()
try:
    content = json.load(open(f"{out}/{name}.body"))["choices"][0]["message"]["content"]
except Exception:
    content = "NOT A REPLY: " + open(f"{out}/{name}.body").read()[:200]
print(json.dumps({"turn": name, "status": status, "seconds": float(t or 0), "content": content}))
PY
  }
  ask_turn turn1
  if [ -n "${MISS:-}" ]; then
    # B forgets what it stored — restarted between the turns, as a peer's
    # daemon may be. Turn 2 must miss, read the prompt again from 0, answer.
    kill "$PB"
    for _ in $(seq 1 60); do kill -0 "$PB" 2>/dev/null || break; sleep 1; done
    PB=$(SWARMLLM_TEST_TENSOR_DELAY_MS="${DELAY_B:-0}" start "$BASE/B" 8920 "$BIN_B" "${GPU_B:-}")
    up "$BASE/B" 8920 || exit 1
    for _ in $(seq 1 60); do [ "$(peers)" -ge 1 ] && break; sleep 2; done
    sleep 10
    echo "splitcache: B restarted between the turns"
  fi
  FROM_A=$(wc -l < "$BASE/A/node.log"); FROM_B=$(wc -l < "$BASE/B/node.log")
  ask_turn turn2
  sleep 3
  python3 - "$BASE/A/node.log" "$BASE/B/node.log" "$OUT" "$FROM_A" "$FROM_B" "${MISS:-}" "${SWARMLLM_SPLIT_PROMPT_CACHE:-}" "$BASE/C/node.log" <<'PY'
import json, os, re, sys
a = open(sys.argv[1], errors="replace").read().splitlines()
b = open(sys.argv[2], errors="replace").read().splitlines()
out, from_a, from_b, miss, switch = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
# SPARE: C holds B's parts, and its hits count the same — C stored nothing
# before turn 1, so every hit in its log is turn 2's.
c = open(sys.argv[8], errors="replace").read().splitlines() if os.path.exists(sys.argv[8]) else []
off = switch in ("0", "off", "false")
def resumed(lines):
    return [int(m.group(1)) for l in lines if "a split prompt pass keeping its prompt" in l
            for m in [re.search(r"resumed_from=(\d+)", l)] if m]
turns = [json.loads(l) for l in open(f"{out}/splitcache.jsonl")]
both_200 = len(turns) == 2 and all(" 200" in t["status"] for t in turns)
t1, t2 = resumed(a[:from_a]), resumed(a[from_a:])
b_hits = sum("HIT on a split's stored prompt" in l for l in b[from_b:] + c)
b_stored = sum("stored a split's prompt" in l for l in b + c)
missed = any("a resumed prompt pass did not finish" in l and "Prompt cache miss" in l for l in a[from_a:])
print(f"splitcache: turn 1 resumed_from={t1}  turn 2 resumed_from={t2}  B hits on turn 2={b_hits}  "
      f"B stores={b_stored}  miss logged={missed}  replies 200={both_200}  "
      f"seconds {[t['seconds'] for t in turns]}")
if off:
    ok = both_200 and not t1 and not t2
    arm = "off (SWARMLLM_SPLIT_PROMPT_CACHE=0)"
elif miss:
    ok = both_200 and missed and 0 in t2
    arm = "miss (B restarted between the turns)"
else:
    ok = both_200 and t1 == [0] and bool(t2) and max(t2) > 0 and b_hits >= 1
    arm = "on"
print(f"splitcache [{arm}]: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

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
    [ "$i" -gt 1 ] && sleep "${REPEAT_GAP:-0}"
    ask "$PROMPT" 120 "repeat$i" | tee -a "$OUT/repeat.jsonl" | cut -c1-160
  done
  # Not "the first only" any more: since the tail walks the guesses (a few ids back,
  # not a vocabulary) the payoff gate keeps the loop on while rounds average past its
  # bar — llama-3.2-3b on this prompt reads 1.30 tokens/round, all 3 of 3 on .218 too.
  echo "repeat: n-gram path taken by $(grep -c 'try_ngram_only_distributed ELIGIBLE' "$BASE/A/node.log") of ${REPEAT:-3} requests (payoff_x100: $(grep -a -o 'payoff_x100=[0-9]*' "$BASE/A/node.log" | tail -1))"
  echo "repeat: score with examples/score_against_reference.py <model.gguf> $OUT/repeat.jsonl $OUT/prompt.txt"
  exit 0
fi

if [ "$MODE" = remote ]; then
  # Both planners must see the whole model before the first ask: A to plan it
  # among the others, and B — which only dialled A — to lead it, having met C
  # (and D) through A's peer exchange. A first ask racing that would read as
  # a fallback the delegation did not deserve.
  KB=$(cat "$BASE/B/api_key")
  covered() { # key port want_head(self|other)
    curl -s -m 10 -H "Authorization: Bearer $1" "localhost:$2/api/admin/models/$MODEL/pipeline-plan" \
      | python3 -c 'import sys,json
try: p=json.load(sys.stdin)
except Exception: sys.exit(1)
seg=p.get("segments",[])
print("  plan on :%s:" % sys.argv[1], " ".join("%s%s" % (s["node_id"][:8], s["layer_range"]) for s in seg), file=sys.stderr)
sys.exit(0 if len(seg) >= 2 else 1)' "$2"
  }
  for _ in $(seq 1 60); do covered "$KA" 8900 2>/dev/null && covered "$KB" 8920 2>/dev/null && break; sleep 3; done
  covered "$KA" 8900 && covered "$KB" 8920 || { echo "remote: A or B never planned the model across the others"; exit 1; }
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  : > "$OUT/remote.jsonl"
  : > "$OUT/remote_times.txt"
  for i in $(seq 1 "${REPEAT:-3}"); do
    t0=$(date +%s.%N)
    ask "$PROMPT" 120 "remote$i" | tee -a "$OUT/remote.jsonl" | cut -c1-160
    echo "remote$i $(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/remote_times.txt"
  done
  handed=$(grep -c 'delegated split: this node holds none of the plan' "$BASE/A/node.log")
  led=$(grep -c 'handling a delegated split' "$BASE/B/node.log")
  fell_back=$(grep -c 'coordinating the same plan from here instead' "$BASE/A/node.log")
  grep -E 'delegated split|handling a delegated split' "$BASE/A/node.log" "$BASE/B/node.log" | head -4 | cut -c1-220
  python3 - "$OUT" "${REPEAT:-3}" "$handed" "$led" "$fell_back" "${SWARMLLM_DELEGATE_SPLIT:-}" <<'PY'
import json, sys
out, n, handed, led, fell_back, switch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
rows = [json.loads(l) for l in open(f"{out}/remote.jsonl")]
times = dict(l.split() for l in open(f"{out}/remote_times.txt"))
ok = len(rows) == n
for i, r in enumerate(rows, 1):
    try:
        usage = json.load(open(f"{out}/remote{i}.body")).get("usage", {})
    except Exception:
        usage = {}
    secs = float(times.get(f"remote{i}", "nan"))
    toks = usage.get("completion_tokens") or 0
    good = r["status"].endswith("200 ok") and bool(r.get("content"))
    ok &= good
    print(f"remote: ask {i} {r['status']}  {toks} tokens in {secs:.1f} s = {toks / secs if secs else 0:.2f} tok/s end to end  route={r['route']}")
control = switch in ("0", "false", "FALSE")
if control:
    mech = handed == 0
    print(f"remote: control arm — hand-offs {handed} (expect 0)")
else:
    mech = handed == n and led == n and fell_back == 0
    print(f"remote: delegated arm — A handed {handed}/{n}, B led {led}/{n}, fell back {fell_back}")
print("remote: PASS" if ok and mech else "remote: FAIL")
sys.exit(0 if ok and mech else 1)
PY
  exit $?
fi

if [ "$MODE" = continue ]; then
  id16() { curl -s -m 5 -H "Authorization: Bearer $(cat "$1")" "localhost:$2/api/admin/stats" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])'; }
  IB=$(id16 "$BASE/B/api_key" 8920); IC=$(id16 "$BASE/C/api_key" 8940)
  Q="Explain in detail how a refrigerator works, step by step, covering the compressor, the condenser, the expansion valve and the evaporator."
  BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"max_tokens":300,"temperature":0,"stream":True,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$Q")
  : > "$OUT/continue.sse"
  curl -s -N -m 900 -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
    -X POST localhost:8900/v1/chat/completions -d "$BODY" > "$OUT/continue.sse" &
  CURL=$!
  chunks() { grep -c '"content"' "$OUT/continue.sse" 2>/dev/null || echo 0; }
  for _ in $(seq 1 600); do [ "$(chunks)" -ge "${KILL_AFTER:-24}" ] && break; kill -0 $CURL 2>/dev/null || break; sleep 0.5; done
  SERVER=$(grep -a 'remote-generate fast path: request sent' "$BASE/A/node.log" | tail -1 | grep -oE 'target=[0-9a-f]+' | cut -d= -f2)
  case "$SERVER" in "$IB"*) VICTIM=$PB; OTHER=$IC ;; "$IC"*) VICTIM=$PC; OTHER=$IB ;; *) VICTIM=""; OTHER="" ;; esac
  AT_KILL=$(chunks)
  WPID=$( [ -n "$VICTIM" ] && pgrep -P "$VICTIM" -f model-worker | head -1 )
  echo "continue: A handed it to ${SERVER:-?}; killing its worker (pid ${WPID:-none}) after $AT_KILL chunks"
  [ -n "$WPID" ] && kill -9 "$WPID"
  wait $CURL
  continued=$(grep -c "continuing it on a fresh route" "$BASE/A/node.log")
  served_by_other=$(grep -a 'remote-generate fast path: request sent' "$BASE/A/node.log" | grep -c "target=${OTHER:-none}")
  python3 - "$OUT/continue.sse" "$AT_KILL" "$continued" "$served_by_other" "${WPID:-}" <<'PY'
import json, sys
path, at_kill, continued, other, wpid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
text, chunks, finish, error = "", 0, None, None
for line in open(path, errors="replace"):
    line = line.strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        continue
    try:
        ev = json.loads(line[5:])
    except Exception:
        continue
    if "error" in ev:
        error = ev["error"]
        continue
    for ch in ev.get("choices", []):
        c = (ch.get("delta") or {}).get("content")
        if c:
            text += c
            chunks += 1
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
opening = text[:40]
once = bool(opening) and text.count(opening) == 1
after = chunks - at_kill
print(f"continue: {chunks} chunks ({after} after the kill), finish={finish}, error={'yes' if error else 'no'}, opening once={once}")
print(f"continue: A continued {continued} time(s); the other peer was handed the request {other} time(s)")
print("continue: reply: " + text[:400].replace("\n", " "))
ok = bool(wpid) and error is None and finish is not None and after > 0 and once and continued >= 1 and other >= 1
print("continue: PASS" if ok else "continue: FAIL" + ("" if wpid else " (no worker was killed)"))
sys.exit(0 if ok else 1)
PY
  exit $?
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
  had_worker=$(pgrep -P "$PB" | head -1)
  loaded0=$(grep -c 'Split model using' "$BASE/B/node.log")
  base=$(cpu_of_children)
  ask "$PROMPT" 120 takeover >> "$OUT/failover.jsonl" &
  ASK=$!
  # A worker spawned FOR this request (failover_mid has no healthy arm to warm
  # it) loads its model first, and the load burns CPU too: killed then, B's
  # pool just respawns it — the forward was still waiting on the spawn — and
  # the request is served with no takeover to observe. A rig FAIL with nothing
  # wrong, seen 2026-09-29 with A on the card (the plain arm failed alike).
  # So wait out the load, then time the prompt pass from there.
  if [ -z "$had_worker" ]; then
    until [ "$(grep -c 'Split model using' "$BASE/B/node.log")" -gt "$loaded0" ]; do
      kill -0 $ASK 2>/dev/null || break
      sleep 0.02
    done
    base=$(cpu_of_children)
  fi
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
