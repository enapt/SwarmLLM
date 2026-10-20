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
#   reasoning  one greedy question to a reasoning model (default qwen3-1.7b-q8-0)
#          through the split, streamed and not: the finished scratchpad must be
#          the same in message.reasoning_content and delta.reasoning_content,
#          and in neither content (#141). RIG_QUESTION=… asks another question.
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
#          REPEAT_MAX_TOKENS=N caps each reply (default 120) — under a tool that
#          slows the card ~100x (compute-sanitizer), 120 tokens outlast the
#          client's 900 s and the run sees one request boundary, not REPEAT.
#          RIG_TEMPERATURE=0.7 asks every question at that temperature (default 0,
#          greedy) — for speculation that must hold at the temperatures clients send.
#          SHARDS_C=… (set SHARDS_B too) adds C holding those parts: a three-machine
#          split, A → B → C, B handing C its output — with DRAFTER, the guess-checking
#          stream through a chain (#318). The summary counts A's streamed replies by
#          how many peers each crossed (`peers=`). C dials only A, as a node new to
#          the swarm would: B meets C when a chained run it cannot hand on makes it
#          dial (`layer_forward::dial_next_hop`), and the runs before that finish
#          unchained. C_MEETS_B=1 has C dial B as well, so the chain runs at once.
#   (any mode) DRAFTER=<model id> also gives A that model WHOLE, from the same
#          models directory — the small model speculation across computers guesses
#          with (`pipeline::engine_drafter`). Pair it with EXTRA_TOML switching
#          `speculative_decoding` and `decentralized_spec_decoding` on; A picks it
#          itself (or name it with `draft_model`). MODEL must share its vocabulary,
#          e.g. MODEL=qwen2.5-coder-7b-instruct-q4-k-m DRAFTER=qwen2.5-0.5b-instruct-fp16.
#   softunload  A holds the WHOLE model, larger than its card, with auto-manage at
#          its defaults (a 30 s cycle); B holds every part but the first and last,
#          on the processor. One question, two of A's prune passes, a second
#          question. Field report #035 (FUTURE_WORK #248): the soft-unload counted
#          a card full of A's OWN model as pressure and narrowed it, so the second
#          question went A → B → A instead of staying on A. PASS = no
#          "narrowing shard window" in A's log and q2's route is A alone.
#          Default model qwen2.5-14b-instruct-q4-k-m (48 layers; ~22 fit an 8 GB card).
#   layerprice  softunload's topology for FUTURE_WORK #250 (report #035): a
#          mixture-of-experts model whole on A, larger than its card, B holding
#          the middle on the processor. The planner priced every model's layer
#          as a dense 7B's, so a 30B-A3B layer (~3B of it active) cost ten times
#          what it does and the middle went to B. Reads the plan A would make
#          (the preview, nothing runs), then asks one question ONLY when that
#          plan is A alone — a plan through B gives A two workers on the card,
#          a shape this rig must not make on its own. Run it twice in one
#          binary: `SWARMLLM_LAYER_PRICE=reference` is the old price. PASS (the
#          default arm) = the plan is A alone, the reply is 200, and A logged
#          `layer_work=` below 1. Default model qwen3-30b-a3b-instruct-2507-q2-k.
#   keep   FUTURE_WORK #251 (report #035): "Keep on this computer". A holds
#          shard 0 and B the rest, both on the processor, so every answer needs
#          B. Three asks through A: the setting off (200 through B), on (A
#          refuses: 503 `kept_on_this_computer_error`, and its preview plan is
#          not routable), off again (200 through B). PASS = all three, and B
#          was handed nothing while the setting was on. Then the same choice per
#          REQUEST with the setting off (#251 (2b)): the header and the body's
#          `swarmllm` field (503, B handed nothing), a bad header value and a
#          misspelt field (400), the header saying `false` (200 through B), MCP's
#          `_meta` (refused, B handed nothing). Default model tinyllama.
#   route  FUTURE_WORK #251 (3): keep's topology (A shard 0, B the rest, both on
#          the processor). One question through A as chat, Anthropic and
#          Responses, each streamed and not, and MCP `chat` and `compare`.
#          PASS = every reply names two computers, one of them a peer: the
#          headers of the unstreamed chat reply, its body's `swarmllm.route`
#          (the same route), the streamed chat's FINISH chunk (and no other),
#          `message_delta`, `response.completed`, MCP's
#          `_meta["com.swarmllm/route"]` and compare's per-result `route` —
#          and none of it in the text the model reads. Default model tinyllama.
#          With SHARDS_A naming EVERY shard, A holds the model whole, it is
#          kept on A (so the local fast path answers — on the processor with a
#          peer up, the router would), and PASS = every reply names A alone
#          (one segment, no peer). v0.3.243's streamed forms said
#          `"segments": 0` there, which no split can see (#306).
#   owntiming  FUTURE_WORK #3 (f): split's topology (A shard 0, B the rest, both
#          on the processor). Each worker times its own decode steps and
#          publishes the figure beside the layers it holds
#          (`ResidentModelLayers::decode_us_per_layer`); a planner reads it as
#          `stated_ms_per_layer` on its candidate line. Only work run FOR
#          PEERS is timed. Three asks through A, TIMING_WAIT_S (default 75) for
#          two capability rounds, a fourth through A, one through B, the wait
#          again, one more through B. PASS = A's first plan had no figure for
#          B; its fourth had B's (timed on the segments B served A); B's first
#          plan had NONE for A (A had run only its own requests, never timed —
#          they would tell the swarm when the owner used a model); B's second
#          had A's (A served B's first segment). Run with an older BIN_B and
#          the second check fails: nothing publishes it.
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
#          other peer served it — and the finish chunk's `swarmllm.route` still
#          names BOTH peers in `all_nodes` (`all_peers` 2, `attempts` 2): the
#          killed one saw the request too (#302; v0.3.244 named only the peer it
#          finished on). BIN_A = v0.3.230 is the control: the stream ends with an
#          error after the kill.
#          NONSTREAM=1 asks the same question WITHOUT streaming and kills the
#          worker KILL_AFTER_S (default 8) seconds in. Prints the reply's
#          seconds, finish reason and length, which recovery A took (a re-plan
#          from the first token, or a continuation from what was generated) and
#          whether the opening appears once. PASS = 200, a finish other than
#          "interrupted", the opening once, and A CONTINUED the reply rather
#          than generating it again — which no build does yet (FUTURE_WORK
#          #167): v0.3.232 re-plans it from the first token (measured
#          2026-10-10: 200 in 47.9 s, the reply generated twice over).
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
#   handoffcache  FUTURE_WORK #3 (c): cache's two turns, but A holds the
#          header only and B every part (processor), so each turn is handed to
#          B whole. PASS = on turn 2 A's planner logged that B still holds the
#          opening it was handed (`handed_off`) and B's worker `prefix-cache HIT`
#          matched at least that much. An older BIN_A logs no credit (FAIL is
#          the baseline): the turn is priced as a full read.
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
#          WARM_SPARE=1 (with SPARE=1) first has the peer turn 1 did NOT use
#          lead a conversation of its own through the same split: both peers
#          warm, only one holding this conversation — PASS still needs turn 2
#          to resume, so the plan must pick the peer holding it.
#          SHARDS_C=… adds C holding those parts instead (set SHARDS_B too): a
#          three-machine split, A → B → C. B's and C's hits both count.
#   remote  A holds NONE of the model (the header only) and REMOTE_NODES (2,
#          default, or 3) other nodes hold it between them in contiguous parts
#          — B the first, then C (and D) — the shape a user who stores nothing
#          meets for every split model (FUTURE_WORK #143). REPEAT (default 3)
#          greedy runs of the long prompt through A, timed. DELAY_A=ms holds
#          every tensor A sends that long (`SWARMLLM_TEST_TENSOR_DELAY_MS` on A
#          only): A far from holders that are close to each other.
#          C_MEETS_B=1 has C dial B too: without it B may never meet C here
#          (identify drops a loopback address) and never plans the model.
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
#   latekv  FUTURE_WORK #263: a split whose LOCAL segment comes after a peer's,
#          where this node's conversation memory cannot hold a long prompt. A
#          holds the last two shards on the processor with a tight RAM budget
#          (RAM_A, MB, as A's `resources.max_ram_mb`), B every shard but the
#          last. q1 (short) loads A's worker; q2 is a ~5,500-token prompt. Set
#          RAM_A from a probe: run with RAM_A=6000, read A's `CPU worker KV
#          budget` line, then RAM_A = 6000 - headroom_mb + 10. An older release
#          is the control: B reads the prompt, A's segment refuses it, the
#          re-plan builds the same plan and B reads it again — 503 after two
#          prefills. PASS = A logged `KV admission up front` before B read
#          anything, A's late refusal (`refusing this prompt before prefill` on
#          a forward) never happened, and q2 answered 200 on a plan giving A
#          fewer layers. Default model llama-3.2-3b.
#   peerkv  FUTURE_WORK #252 (b): latekv's shape mirrored — the PEER holds the
#          last two shards on the processor with a tight budget (RAM_B, MB),
#          A every shard but the last. q1 (short) loads B's worker and prints
#          its `CPU worker KV budget`; q2 is the ~5,100-token prompt. Prints the
#          asks B answered before the pass and refused (#252 (b)), the forwards
#          it refused, A's re-plans and the routes A planned. An older release is
#          the control: B refuses a forward after A has read the prompt and is
#          barred. Default model llama-3.2-3b.
#   refusepiece FUTURE_WORK #303: peerkv's shape, the long prompt only. Run
#          with SWARMLLM_FAULT_REFUSE_DECRYPT=3: B treats its 3rd forward — a
#          PIECE of the prompt — as undecryptable. An older release is the
#          control: the refusal reaches no wait, A times the piece out twice and
#          re-plans (503 after 403.8 s on 2026-10-18). PASS = 200, `A took the
#          refusal: 1`, `A timed out: 0`, `A re-planned: 0`.
#   refusechain FUTURE_WORK #305: remote's shape with REMOTE_NODES=2 and the loop
#          kept on A (`SWARMLLM_DELEGATE_SPLIT=0`), so a decode step travels
#          A → B → C as one chained forward; C alone runs with
#          SWARMLLM_FAULT_REFUSE_DECRYPT=REFUSE_AT (default 6, past the prompt
#          pass) and refuses to B, the hop it came from. v0.3.244 is the
#          control: B holds no wait and drops the refusal, A waits out the
#          chain's deadline. PASS = `B passed it on: 1`, `A took the refusal: 1`,
#          `A timed out: 0`; the status and time are printed beside it.
#   killchain FUTURE_WORK #184: refusechain's topology without the fault — a long
#          reply whose decode steps travel A → B → C, and C's node is killed
#          (SIGKILL, workers too) 40 results in. PASS = the reply ends within
#          KILL_SLACK (default 60) s of the kill and `A timed out: 0` — C held
#          the last layers alone, so the end is a fast 503 or a salvaged partial,
#          never the chain's deadline.
#
# usage: split_rig.sh split|reasoning|kill|failover|failover_mid|context|whole|continue|repeat|fetch|cache|splitcache|remote|mixed|disputed|spliced|disconnect|softunload|layerprice|keep|route|latekv|peerkv|refusepiece|owntiming <binary> [<binary for B>]
#   MODEL      model id (default: tinyllama for split, llama-3.2-3b for kill
#              and failover)
#   SHARDS_A   shard indices A holds, comma-separated (default 0; kill: 0,LAST)
#   SHARDS_B   shard indices B holds (default: every shard A lacks; kill: all)
#   GPU_A/B    SWARMLLM_INFERENCE_GPU_LAYERS for each node ("" = auto)
#   ENV_A/B/…  extra VAR=value words for that node alone (e.g.
#              ENV_C="SWARMLLM_FAULT_REFUSE_DECRYPT=6"); the caller's environment
#              reaches every node
#   CPUS_A/B/… logical CPUs that node runs on (`taskset -c`, e.g. CPUS_A=0-7
#              CPUS_B=8-15): two nodes on disjoint halves of the machine stand in
#              for two computers when work across them overlaps (#171)
#   MEM_C/D/E  failover modes: a memory limit for that standby (e.g. 6G) — it runs
#              in a `systemd-run` scope with that MemoryMax (CPUQuota RIG_SCOPE_CPU,
#              default 400%) and reads it as its machine, so the room it advertises
#              follows its own use, not the machine's free RAM (#162 b)
#   EXTRA_TOML appended to EVERY node's config.toml (default: nothing)
#   RAM_B      peerkv: B's `resources.max_ram_mb`, the same way
#   RAM_A      latekv: A's `resources.max_ram_mb` (EXTRA_TOML may not open
#              [resources] beside it)
#   MODELS_DIR where the shards come from (default: the live node's)
#   OUT        where logs and replies go (default: a temp dir, printed)
#
# The rig's shard files are HARD LINKS to MODELS_DIR's (no copy, no disk), so
# MODELS_DIR must be on the same filesystem as the rig ($HOME by default). The
# live node's own mDNS will find the rig, and a rig node that can see a third
# peer can route through it — so this refuses to run a request unless A sees
# exactly one peer. Stop the live node first. `gossip_network_id` is NOT
# isolation (#352) — or set RIG_NETNS=1, below.
set -u

# RIG_NETNS=1: run in a private network namespace (`unshare -rn`, as
# ceiling_test.sh does), where no other node on this machine can reach the rig
# nor it them — so it can run beside the live node. Nothing in it reaches the
# internet either, so a mode that needs HuggingFace cannot run this way.
if [ "${RIG_NETNS:-}" = 1 ] && [ -z "${RIG_IN_NETNS:-}" ]; then
  exec env RIG_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
if [ -n "${RIG_IN_NETNS:-}" ]; then
  ip link set lo up || { echo "cannot bring up loopback"; exit 2; }
fi

# RIG_RTT_MS=n (needs RIG_NETNS=1): every packet between the rig's nodes waits
# n/2 ms each way (netem on the namespace's own loopback), so a pass pays a
# real round trip in the transport itself: handshakes, acknowledgements and
# libp2p's own frames included. SWARMLLM_TEST_TENSOR_DELAY_MS (DELAY_A/B)
# holds only our tensor messages. RIG_RATE=20mbit caps the link's rate as
# well. A worker talks to its node over a local socket, which netem never
# touches. Measured 2026-10-20: delay 50ms gave a 101-104 ms ping.
# RIG_LOSS=1% drops that share of packets as well (each way).
if [ -n "${RIG_RTT_MS:-}" ]; then
  [ -n "${RIG_IN_NETNS:-}" ] || { echo "RIG_RTT_MS needs RIG_NETNS=1"; exit 2; }
  tc qdisc add dev lo root netem delay "$((RIG_RTT_MS / 2))ms" ${RIG_RATE:+rate "$RIG_RATE"} \
    ${RIG_LOSS:+loss "$RIG_LOSS"} \
    || { echo "cannot add the delay to loopback (tc netem)"; exit 2; }
  echo "rig: loopback round trip ${RIG_RTT_MS} ms${RIG_RATE:+, rate $RIG_RATE}${RIG_LOSS:+, loss $RIG_LOSS}: $(ping -c 2 -q 127.0.0.1 | tail -1)"
fi
# RIG_TRANSPORT=quic (needs RIG_NETNS=1): the nodes reach each other over QUIC,
# as the live swarm's connections mostly do (2026-10-20: ~90% of the live
# node's), not TCP. Bootstrap addresses are QUIC, and every packet to a node's
# P2P TCP port (its API port + 10) is dropped on the namespace's loopback — the
# loopback probe (`discovery::probe_loopback_peers`) dials TCP — so a node keeps
# one connection per peer and it is the QUIC one. The HTTP API stays on TCP.
if [ "${RIG_TRANSPORT:-tcp}" = quic ]; then
  [ -n "${RIG_IN_NETNS:-}" ] || { echo "RIG_TRANSPORT=quic needs RIG_NETNS=1"; exit 2; }
  tc qdisc add dev lo handle ffff: ingress || { echo "cannot add an ingress qdisc to loopback"; exit 2; }
  for p in 8910 8930 8950 8970; do
    tc filter add dev lo parent ffff: protocol ip prio 1 u32 match ip protocol 6 0xff \
      match ip dport "$p" 0xffff action drop || { echo "cannot drop TCP to $p"; exit 2; }
  done
  echo "rig: nodes connect over QUIC (TCP to their P2P ports dropped)"
fi
# The address a node at API port $1 is dialled on, given its peer id $2.
rig_addr() {
  if [ "${RIG_TRANSPORT:-tcp}" = quic ]; then echo "/ip4/127.0.0.1/udp/$1/quic-v1/p2p/$2"
  else echo "/ip4/127.0.0.1/tcp/$(($1 + 10))/p2p/$2"; fi
}

MODE="${1:?usage: split_rig.sh split|kill|failover|context|repeat|fetch|cache|remote|mixed|softunload <binary> [<binary for B>]}"
BIN_A="${2:?binary}"
BIN_B="${3:-$BIN_A}"
case "$MODE" in split|reasoning|kill|failover|failover_mid|context|whole|continue|repeat|fetch|cache|handoffcache|splitcache|remote|mixed|disputed|spliced|disconnect|softunload|layerprice|keep|route|latekv|peerkv|refusepiece|refusechain|killchain|owntiming) ;; *) echo "mode must be split, reasoning, kill, failover, failover_mid, context, whole, continue, repeat, fetch, cache, handoffcache, splitcache, remote, mixed, disputed, spliced, disconnect, softunload, layerprice, keep, route, latekv, peerkv, refusepiece, refusechain, killchain or owntiming"; exit 2 ;; esac
# refusechain is remote's topology with one fault on C (#305).
REFUSECHAIN=""; KILLCHAIN=""
if [ "$MODE" = killchain ]; then
  MODE=remote; KILLCHAIN=1; REMOTE_NODES=2
  export SWARMLLM_DELEGATE_SPLIT=0
fi
if [ "$MODE" = refusechain ]; then
  MODE=remote; REFUSECHAIN=1; REMOTE_NODES=2
  export SWARMLLM_DELEGATE_SPLIT=0
  ENV_C="SWARMLLM_FAULT_REFUSE_DECRYPT=${REFUSE_AT:-6}${ENV_C:+ $ENV_C}"
fi
if [ -n "${RAM_A:-}${RAM_B:-}" ] && printf '%s' "${EXTRA_TOML:-}" | grep -q '^\[resources\]'; then
  echo "RAM_A/RAM_B are written to that node's [resources]; EXTRA_TOML may not open that table too"; exit 2
fi
if { [ "$MODE" = context ] || [ "$MODE" = whole ]; } && printf '%s' "${EXTRA_TOML:-}" | grep -q '^\[inference\]'; then
  echo "$MODE: EXTRA_TOML may not open [inference] — B's ceiling is written there"; exit 2
fi
[ -x "$BIN_A" ] && [ -x "$BIN_B" ] || { echo "binary not executable"; exit 2; }
if [ "$MODE" = reasoning ]; then
  MODEL="${MODEL:-qwen3-1.7b-q8-0}"
elif [ "$MODE" = split ] || [ "$MODE" = mixed ] || [ "$MODE" = disputed ] || [ "$MODE" = spliced ] || [ "$MODE" = disconnect ] || [ "$MODE" = keep ] || [ "$MODE" = route ] || [ "$MODE" = owntiming ]; then
  MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
elif [ "$MODE" = cache ] || [ "$MODE" = handoffcache ]; then
  MODEL="${MODEL:-qwen2.5-0.5b-instruct-fp16}"
elif [ "$MODE" = softunload ]; then
  MODEL="${MODEL:-qwen2.5-14b-instruct-q4-k-m}"
elif [ "$MODE" = layerprice ]; then
  MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507-q2-k}"
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
if [ "$MODE" = split ] || [ "$MODE" = repeat ] || [ "$MODE" = disconnect ] || [ "$MODE" = keep ] || [ "$MODE" = route ] || [ "$MODE" = owntiming ] || [ "$MODE" = reasoning ]; then
  SHARDS_A="${SHARDS_A:-0}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vxF -f <(echo "$SHARDS_A" | tr ',' '\n') | paste -sd,)}"
  # Processor unless asked otherwise: the reference is scored on the processor.
  [ "$MODE" = repeat ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
  [ "$MODE" = disconnect ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
  [ "$MODE" = keep ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
  [ "$MODE" = reasoning ] && { GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"; }
  if [ "$MODE" = route ]; then
    WHOLE_A=1
    for s in $SHARDS; do case ",$SHARDS_A," in *",$s,"*) ;; *) WHOLE_A=0 ;; esac; done
    GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
  fi
elif [ "$MODE" = peerkv ] || [ "$MODE" = refusepiece ]; then
  [ "$N" -ge 3 ] || { echo "$MODE needs a model with at least 3 shard files here; $MODEL has $N"; exit 2; }
  PREV=$(echo "$SHARDS" | tail -2 | head -1)
  SHARDS_B="${SHARDS_B:-$PREV,$LAST}"
  SHARDS_A="${SHARDS_A:-$(echo "$SHARDS" | grep -vx "$LAST" | paste -sd,)}"
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = latekv ]; then
  [ "$N" -ge 3 ] || { echo "latekv needs a model with at least 3 shard files here; $MODEL has $N"; exit 2; }
  PREV=$(echo "$SHARDS" | tail -2 | head -1)
  SHARDS_A="${SHARDS_A:-$PREV,$LAST}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vx "$LAST" | paste -sd,)}"
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = splitcache ]; then
  SHARDS_A="${SHARDS_A:-0}"
  SHARDS_B="${SHARDS_B:-$(echo "$SHARDS" | grep -vxF -f <(echo "$SHARDS_A" | tr ',' '\n') | paste -sd,)}"
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = softunload ] || [ "$MODE" = layerprice ]; then
  [ "$N" -ge 3 ] || { echo "$MODE needs a model with at least 3 shard files here; $MODEL has $N"; exit 2; }
  SHARDS_A=$(echo "$SHARDS" | paste -sd,)
  SHARDS_B=$(echo "$SHARDS" | grep -vx 0 | grep -vx "$LAST" | paste -sd,)
  GPU_B="${GPU_B:-0}"
elif [ "$MODE" = cache ]; then
  SHARDS_A=$(echo "$SHARDS" | paste -sd,)
  SHARDS_B=$SHARDS_A
  GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-0}"
elif [ "$MODE" = handoffcache ]; then
  # A holds the header only, so every turn is handed to B whole.
  SHARDS_A=""
  SHARDS_B=$(echo "$SHARDS" | paste -sd,)
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
if [ -n "$OTHERS" ] && [ -z "${RIG_IN_NETNS:-}" ]; then
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
  # softunload: A runs auto-manage as a default install does, on a short cycle
  # and without the HuggingFace watcher (the rig must not fetch anything).
  local am=$'enabled = false\nprune_enabled = false'
  [ "$MODE" = softunload ] && [ "$(basename "$d")" = A ] \
    && am=$'enabled = true\nprune_enabled = true\ninterval_seconds = 30\nhf_watcher_enabled = false'
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
$am

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
  [ "$(basename "$d")" = A ] && [ -n "${RAM_A:-}" ] \
    && printf '\n[resources]\nmax_ram_mb = %s\n' "$RAM_A" >> "$d/config.toml"
  [ "$(basename "$d")" = B ] && [ -n "${RAM_B:-}" ] \
    && printf '\n[resources]\nmax_ram_mb = %s\n' "$RAM_B" >> "$d/config.toml"
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
  # ENV_A / ENV_B / …: VAR=value words for this node alone (a fault on one hop).
  local env_var="ENV_$(basename "$1")" own=() words=()
  if [ -n "${!env_var:-}" ]; then read -r -a words <<< "${!env_var}"; own=(env "${words[@]}"); fi
  if [ -n "$4" ]; then
    SWARMLLM_INFERENCE_GPU_LAYERS="$4" SWARMLLM_NODE_DATA_DIR="$1" "${own[@]}" "${scope[@]}" "${pin[@]}" "$3" run -p "$2" > "$1/node.log" 2>&1 &
  else
    SWARMLLM_NODE_DATA_DIR="$1" "${own[@]}" "${scope[@]}" "${pin[@]}" "$3" run -p "$2" > "$1/node.log" 2>&1 &
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
# In its own namespace the rig has only loopback, which A does not publish: dial
# it there, as ceiling_test.sh does (P2P listens 10 above the API port).
if [ -z "$ADDR" ] && [ -n "${RIG_IN_NETNS:-}" ]; then
  ID_A=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$BASE/A/node.log" | cut -d= -f2)
  [ -n "$ID_A" ] && ADDR=$(rig_addr 8900 "$ID_A")
fi
if [ "${RIG_TRANSPORT:-tcp}" = quic ]; then
  ID_A=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$BASE/A/node.log" | cut -d= -f2)
  ADDR=$(rig_addr 8900 "$ID_A")
fi
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
if { [ "$MODE" = splitcache ] && [ -n "${SPARE:-}${SHARDS_C:-}" ]; } \
  || { [ "$MODE" = repeat ] && [ -n "${SHARDS_C:-}" ]; }; then
  # SPARE: a second peer that could run B's segment — what the plan chooses
  # between. SHARDS_C: a third machine with parts of its own, so the plan is
  # A → B → C and B hands C its output (a boundary between two peers, #171).
  C_BOOT="\"$ADDR\""
  if [ "$MODE" = repeat ] && [ -n "${SHARDS_C:-}" ] && [ -n "${C_MEETS_B:-}" ]; then
    ID_B=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$BASE/B/node.log" | cut -d= -f2)
    if [ "${RIG_TRANSPORT:-tcp}" = quic ]; then ADDR_B=$(rig_addr 8920 "$ID_B")
    else ADDR_B=$(echo "$ADDR" | sed -E "s#/tcp/[0-9]+/p2p/.*#/tcp/8930/p2p/$ID_B#"); fi
    [ -n "$ID_B" ] && C_BOOT="\"$ADDR\", \"$ADDR_B\""
  fi
  make_node "$BASE/C" "${SHARDS_C:-$SHARDS_B}" "$C_BOOT"
  PC=$(start "$BASE/C" 8940 "$BIN_A" "${GPU_C:-0}")
  up "$BASE/C" 8940 || exit 1
  PEERS_EXPECTED=2
  if [ -n "${SHARDS_C:-}" ]; then ROLE_C="a third segment"; else ROLE_C="a spare holding the parts B holds"; fi
  echo "rig: C=[${SHARDS_C:-$SHARDS_B}] ($ROLE_C) gpu=${GPU_C:-0}"
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
  C_BOOT="\"$ADDR\""
  if [ -n "$REFUSECHAIN$KILLCHAIN${C_MEETS_B:-}" ]; then
    # A chained step goes B → C, so B must already be connected to C: C dials
    # B as well as A (peer exchange alone can take minutes to introduce them,
    # and on this loopback rig never does: identify drops a 127.0.0.1 address,
    # gotcha #165, so B planned nothing across C in 3 minutes, 2026-10-20).
    # C_MEETS_B=1 asks for the same in the timed arms.
    ID_B=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$BASE/B/node.log" | cut -d= -f2)
    if [ "${RIG_TRANSPORT:-tcp}" = quic ]; then ADDR_B=$(rig_addr 8920 "$ID_B")
    else ADDR_B=$(echo "$ADDR" | sed -E "s#/tcp/[0-9]+/p2p/.*#/tcp/8930/p2p/$ID_B#"); fi
    [ -n "$ID_B" ] && C_BOOT="\"$ADDR\", \"$ADDR_B\""
  fi
  make_node "$BASE/C" "$SHARDS_C" "$C_BOOT"
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
  # RIG_BODY: a JSON object merged into the body; RIG_HEADER: one more header.
  body=$(python3 -c 'import json,os,sys
b={"model":sys.argv[1],"max_tokens":int(sys.argv[3]),"temperature":float(os.environ.get("RIG_TEMPERATURE","0")),"messages":[{"role":"user","content":sys.argv[2]}]}
if os.environ.get("RIG_ROUTE"): b["swarm_route"]=json.loads(os.environ["RIG_ROUTE"])
if os.environ.get("RIG_BODY"): b.update(json.loads(os.environ["RIG_BODY"]))
print(json.dumps(b))' "$MODEL" "$1" "$2")
  curl -s -m 900 -D "$OUT/$3.hdr" -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
       ${RIG_HEADER:+-H "$RIG_HEADER"} -X POST localhost:8900/v1/chat/completions -d "$body" -o "$OUT/$3.body"
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

if [ "$MODE" = keep ]; then
  keep() { curl -s -m 30 -X PUT -H "Authorization: Bearer $KA" -H 'Content-Type: application/json' \
             -d "{\"enabled\": $1}" "localhost:8900/api/admin/models/$MODEL/keep-on-this-computer"; }
  # B's results as A receives them — the count must MOVE on the open ask, or a
  # zero while kept would prove nothing.
  forwards() { grep -a -c 'dispatcher received LayerResult' "$BASE/A/node.log"; }
  Q="What is the capital of France? Answer in one sentence."
  FO=$(forwards)
  ask "$Q" 24 open | tee "$OUT/keep.jsonl" | cut -c1-200
  echo "keep: on  -> $(keep true | cut -c1-160)"
  F0=$(forwards)
  ask "$Q" 24 kept | tee -a "$OUT/keep.jsonl" | cut -c1-200
  F1=$(forwards)
  PREVIEW=$(curl -s -m 30 -H "Authorization: Bearer $KA" "localhost:8900/api/admin/models/$MODEL/pipeline-plan" \
    | python3 -c 'import sys,json; p=json.load(sys.stdin); print(p.get("routable"), (p.get("reason") or {}).get("type"))')
  echo "keep: off -> $(keep false | cut -c1-160)"
  ask "$Q" 24 reopened | tee -a "$OUT/keep.jsonl" | cut -c1-200
  python3 - "$OUT" "$F0" "$F1" "$PREVIEW" "$FO" <<'PY'
import json, sys
out, f0, f1, preview, fo = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
status = lambda l: (open(f"{out}/{l}.hdr").read().splitlines() or ["none"])[0].strip()
route = lambda l: [x.split(":", 1)[1].strip() for x in open(f"{out}/{l}.hdr").read().lower().splitlines() if x.startswith("x-swarm-route")]
try:
    kept_type = json.load(open(f"{out}/kept.body"))["error"]["type"]
except Exception:
    kept_type = "none"
print(f"keep: open {status('open')} {route('open')} | kept {status('kept')} type={kept_type} preview={preview} "
      f"B results during it={f1 - f0} (open: {f0 - fo}) | reopened {status('reopened')} {route('reopened')}")
ok = (" 200" in status("open") and f0 > fo and " 503" in status("kept") and kept_type == "kept_on_this_computer_error"
      and preview.startswith("False") and f1 == f0 and " 200" in status("reopened"))
print("keep:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  PER_MODEL=$?
  # The same choice per REQUEST, the setting off (#251 (2b)).
  H="X-SwarmLLM-Keep-On-This-Computer"
  G0=$(forwards)
  RIG_HEADER="$H: true" ask "$Q" 24 hdrkept | tee -a "$OUT/keep.jsonl" | cut -c1-200
  RIG_BODY='{"swarmllm": {"keep_on_this_computer": true}}' ask "$Q" 24 bodykept | tee -a "$OUT/keep.jsonl" | cut -c1-200
  RIG_HEADER="$H: maybe" ask "$Q" 24 badheader | tee -a "$OUT/keep.jsonl" | cut -c1-200
  RIG_BODY='{"swarmllm": {"keep_on_this_computr": true}}' ask "$Q" 24 typo | tee -a "$OUT/keep.jsonl" | cut -c1-200
  curl -s -m 120 -H "Authorization: Bearer $KA" -H 'Content-Type: application/json' -X POST localhost:8900/mcp \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"chat\",\"arguments\":{\"model\":\"$MODEL\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]},\"_meta\":{\"com.swarmllm/keep_on_this_computer\":true}}}" \
    -o "$OUT/mcpkept.body"
  echo "keep: mcp -> $(cut -c1-200 "$OUT/mcpkept.body")"
  G1=$(forwards)
  RIG_HEADER="$H: false" ask "$Q" 24 hdrfalse | tee -a "$OUT/keep.jsonl" | cut -c1-200
  G2=$(forwards)
  python3 - "$OUT" "$G0 $G1 $G2" <<'PY'
import json, sys
out = sys.argv[1]
g0, g1, g2 = map(int, sys.argv[2].split())
status = lambda l: (open(f"{out}/{l}.hdr").read().splitlines() or ["none"])[0].strip()
def err(l):
    try:
        e = json.load(open(f"{out}/{l}.body"))["error"]
        return e.get("type"), e.get("message", "")
    except Exception:
        return "none", ""
asked = lambda l: " 503" in status(l) and err(l)[0] == "kept_on_this_computer_error" and "asked to stay" in err(l)[1]
mcp = open(f"{out}/mcpkept.body").read()
checks = {
    "header 503": asked("hdrkept"),
    "body 503": asked("bodykept"),
    "bad header 400": " 400" in status("badheader"),
    "typo 400": " 400" in status("typo"),
    "mcp refused": "asked to stay" in mcp,
    "B handed nothing while asked": g1 == g0,
    "false 200 through B": " 200" in status("hdrfalse") and g2 > g1,
}
for k, v in checks.items():
    print(f"keep per request: {'ok  ' if v else 'FAIL'} {k}")
ok = all(checks.values())
print("keep per request:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  PER_REQUEST=$?
  [ "$PER_MODEL" = 0 ] && [ "$PER_REQUEST" = 0 ]
  exit $?
fi

if [ "$MODE" = route ]; then
  # One question through A in every form a reply can take. B holds every part
  # but shard 0, so each answer ran on two computers and must say so — unless
  # SHARDS_A names every shard: then A holds the model whole and must name
  # itself alone.
  echo "route: A holds $([ $WHOLE_A = 1 ] && echo 'the whole model — expecting ONE computer' || echo "shards $SHARDS_A — expecting TWO computers")"
  if [ $WHOLE_A = 1 ]; then
    # A model held whole takes the local fast path only where it runs on the graphics card or
    # nobody else may be asked (`local_fast_path_for`, gotcha #443); on the processor with B up
    # the router plans it, the fast path is never reached, and v0.3.243 PASSED this way while
    # its fast path named no computer (#306). Keeping the model on A leaves nobody to ask.
    curl -s -m 30 -X PUT -H "Authorization: Bearer $KA" -H 'Content-Type: application/json' \
         -d '{"enabled": true}' "localhost:8900/api/admin/models/$MODEL/keep-on-this-computer" > /dev/null
    echo "route: $MODEL kept on A, so the local fast path answers"
  fi
  post() { # path body label  (writes $OUT/<label>.{hdr,body})
    curl -s -m 300 -D "$OUT/$3.hdr" -H "Authorization: Bearer $KA" -H 'Content-Type: application/json' \
         -H 'anthropic-version: 2023-06-01' -X POST "localhost:8900$1" -d "$2" -o "$OUT/$3.body"
  }
  Q="What is the capital of France? Answer in one sentence."
  body() { python3 - "$MODEL" "$Q" "$1" <<'PY'
import json, sys
model, q, form = sys.argv[1:]
msgs = [{"role": "user", "content": q}]
b = {
    "chat": {"model": model, "max_tokens": 24, "temperature": 0, "messages": msgs},
    "resp": {"model": model, "max_output_tokens": 24, "temperature": 0, "input": q},
    "mcp_chat": {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "chat",
                 "arguments": {"model": model, "max_tokens": 24, "temperature": 0, "messages": msgs}}},
    "mcp_compare": {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "compare",
                    "arguments": {"models": [model], "prompt": q, "max_tokens": 24, "temperature": 0}}},
}[form.removesuffix("_stream")]
if form.endswith("_stream"):
    b["stream"] = True
print(json.dumps(b))
PY
  }
  post /v1/chat/completions "$(body chat)" chat
  post /v1/chat/completions "$(body chat_stream)" chat_stream
  post /v1/messages "$(body chat)" msgs
  post /v1/messages "$(body chat_stream)" msgs_stream
  post /v1/responses "$(body resp)" resp
  post /v1/responses "$(body resp_stream)" resp_stream
  post /mcp "$(body mcp_chat)" mcp_chat
  post /mcp "$(body mcp_compare)" mcp_compare
  python3 - "$OUT" "$WHOLE_A" <<'PY'
import json, sys
out, whole = sys.argv[1], sys.argv[2] == "1"
# A alone when it holds the model whole, else A and one peer.
want_seg, want_peers = (1, 0) if whole else (2, 1)
n = "one computer" if whole else "two computers"
def headers(l):
    lines = open(f"{out}/{l}.hdr").read().splitlines()
    return (lines[0].strip() if lines else "no response"), {
        k.strip().lower(): v.strip() for k, v in (x.split(":", 1) for x in lines[1:] if ":" in x)}
def events(l):  # the JSON payloads of an SSE body (or of an MCP reply sent as SSE)
    got = []
    for x in open(f"{out}/{l}.body").read().splitlines():
        if x.startswith("data:") and x[5:].strip() not in ("", "[DONE]"):
            try: got.append(json.loads(x[5:]))
            except ValueError: pass
    return got
def doc(l):
    raw = open(f"{out}/{l}.body").read()
    try: return json.loads(raw)
    except ValueError: return (events(l) or [{}])[-1]
def text_of(l):  # MCP: the tool result's text, and the result itself
    r = doc(l).get("result") or {}
    return (r.get("content") or [{}])[0].get("text", ""), r
status, h = headers("chat")
ref = (doc("chat").get("swarmllm") or {}).get("route")
two = lambda r: (isinstance(r, dict) and r.get("segments") == want_seg and r.get("peers") == want_peers
                 and len(r.get("nodes") or []) == want_seg)
same = lambda r: r == ref
chunks = events("chat_stream")
carrying = [c for c in chunks if "swarmllm" in c]
delta = [e for e in events("msgs_stream") if e.get("type") == "message_delta"]
done = [e for e in events("resp_stream") if e.get("type") == "response.completed"]
mcp_text, mcp_result = text_of("mcp_chat")
compare_text, _ = text_of("mcp_compare")
try: compared = json.loads(compare_text).get("results") or [{}]
except ValueError: compared = [{}]
routes = {
    "chat, not streamed": ref,
    "chat, streamed (finish chunk)": (carrying[-1].get("swarmllm") or {}).get("route") if carrying else None,
    "messages, not streamed": (doc("msgs").get("swarmllm") or {}).get("route"),
    "messages, streamed (message_delta)": (delta[-1].get("swarmllm") or {}).get("route") if delta else None,
    "responses, not streamed": (doc("resp").get("swarmllm") or {}).get("route"),
    "responses, streamed (response.completed)": ((done[-1].get("response") or {}).get("swarmllm") or {}).get("route") if done else None,
    "mcp chat (_meta)": (mcp_result.get("_meta") or {}).get("com.swarmllm/route"),
    "mcp compare (per result)": compared[0].get("route"),
}
print(f"route: chat {status} headers route={h.get('x-swarm-route')} segments={h.get('x-swarm-segments')} "
      f"peers={h.get('x-swarm-peers')} nodes={h.get('x-swarm-nodes')}")
checks = {
    f"the headers name {n}": " 200" in status and h.get("x-swarm-segments") == str(want_seg)
        and h.get("x-swarm-peers") == str(want_peers),
    "the body says what the headers say": two(ref) and ref.get("route") == h.get("x-swarm-route")
        and ",".join(ref.get("nodes", [])) == (h.get("x-swarm-nodes") or "").replace(" ", ""),
    "only the finish chunk carries it": len(carrying) == 1 and bool(carrying[0]["choices"])
        and carrying[0]["choices"][0].get("finish_reason") is not None,
    "the model's text holds no route": "com.swarmllm" not in mcp_text and '"segments"' not in mcp_text,
}
for form, r in routes.items():
    checks[f"{form}: {n}"] = two(r)
    print(f"route: {form:42s} {json.dumps(r)}")
for k, v in checks.items():
    print(f"route: {'ok  ' if v else 'FAIL'} {k}")
ok = all(checks.values())
print("route:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

if [ "$MODE" = layerprice ]; then
  # A plans from what it has HEARD: give B's holdings time to arrive, and read
  # the plan until it stops changing.
  plan() {
    curl -s -m 30 -H "Authorization: Bearer $KA" "localhost:8900/api/admin/models/$MODEL/pipeline-plan" \
      | python3 -c 'import sys,json
p=json.load(sys.stdin); me=p.get("local_node_id","")
seg=p.get("segments",[])
print(" ".join("%s%s" % ("A" if s["node_id"]==me else "B", tuple(s["layer_range"])) for s in seg) or "none: %s" % p.get("reason"))'
  }
  LAST_PLAN=""
  for _ in $(seq 1 20); do
    sleep 6; P=$(plan)
    [ -n "$P" ] && [ "$P" = "$LAST_PLAN" ] && break
    LAST_PLAN=$P
  done
  echo "layerprice: plan $P  (SWARMLLM_LAYER_PRICE=${SWARMLLM_LAYER_PRICE:-<unset: priced as its own model>})"
  case "$P" in
    *B*) echo "layerprice: the plan sends part of the model to B — not asked (it would put two workers on A's card)"; exit 1 ;;
    "A("*) ;;
    *) echo "layerprice: no plan"; exit 1 ;;
  esac
  T0=$(date +%s.%N)
  ask "Explain in three sentences why the sky is blue." 64 q1 | tee "$OUT/replies.jsonl" | cut -c1-200
  echo "layerprice: q1 took $(python3 -c "print(round($(date +%s.%N) - $T0, 1))") s"
  grep -a -o 'layer_work=[0-9.]*' "$BASE/A/node.log" | sort | uniq -c | head -3
  grep -a -E "Model will run|card/processor|layers on the GPU" "$BASE/A/node.log" | tail -2 | cut -c1-220
  exit 0
fi

if [ "$MODE" = softunload ]; then
  ask "What is the capital of France? Answer in one sentence." 32 q1 | tee "$OUT/replies.jsonl"
  grep -a -E "Model will run|on the card|card/processor|layers on the GPU" "$BASE/A/node.log" | tail -3 | cut -c1-220
  passes() { grep -a -c 'DIAG: evaluate_and_prune starting' "$BASE/A/node.log"; }
  P0=$(passes)
  for _ in $(seq 1 75); do [ "$(passes)" -ge $((P0 + 2)) ] && break; sleep 2; done
  sleep 15
  NARROWED=$(grep -a -c 'narrowing shard window' "$BASE/A/node.log")
  STRANDED=$(grep -a -c 'Retiring a worker with requests still in flight' "$BASE/A/node.log")
  echo "softunload: prune passes after q1: $(( $(passes) - P0 ))  narrowed: $NARROWED  stranded: $STRANDED"
  grep -a 'narrowing shard window' "$BASE/A/node.log" | head -2 | cut -c1-260
  T0=$(date +%s.%N)
  ask "Explain in three sentences why the sky is blue." 64 q2 | tee -a "$OUT/replies.jsonl"
  echo "softunload: q2 took $(python3 -c "print(round($(date +%s.%N) - $T0, 1))") s"
  exit 0
fi
if [ "$MODE" = refusepiece ]; then
  # B treats its Nth forward as undecryptable (SWARMLLM_FAULT_REFUSE_DECRYPT,
  # default 3: a PIECE of the long prompt, the first request B sees).
  PROMPT="Here are some notes on household appliances. $(for i in $(seq 1 120); do printf 'A refrigerator moves heat from its inside to the room using a refrigerant that evaporates in the cold coils and condenses in the warm ones; the compressor drives the cycle and the thermostat decides when it runs. '; done)Using only these notes, say in one sentence what the compressor does."
  T0=$(date +%s.%N)
  ask "$PROMPT" 8 q1 | tee "$OUT/replies.jsonl" | cut -c1-300
  echo "refusepiece: took $(python3 -c "print(round($(date +%s.%N) - $T0, 1))") s"
  echo "refusepiece: B refused (fault): $(grep -c 'FAULT: treating this forward as undecryptable' "$BASE/B/node.log")," \
    "A took the refusal: $(grep -c 'LayerResult delivered to pipeline.*Could not decrypt' "$BASE/A/node.log")," \
    "A timed out: $(grep -c 'segment TIMED OUT' "$BASE/A/node.log")," \
    "A re-planned: $(grep -c 'retrying with fresh pipeline' "$BASE/A/node.log")"
  exit 0
fi
if [ "$MODE" = peerkv ]; then
  ask "What is the capital of France? Answer in one sentence." 16 q1 | tee "$OUT/replies.jsonl"
  echo "peerkv: B's worker — $(grep -m1 'CPU worker KV budget' "$BASE/B/node.log" | grep -oE '(kv_budget_mb|typical_kv_mb|headroom_mb)=[0-9]+' | paste -sd' ')"
  PROMPT="Here are some notes on household appliances. $(for i in $(seq 1 120); do printf 'A refrigerator moves heat from its inside to the room using a refrigerant that evaporates in the cold coils and condenses in the warm ones; the compressor drives the cycle and the thermostat decides when it runs. '; done)Using only these notes, say in one sentence what the compressor does."
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  MARK_A=$(wc -l < "$BASE/A/node.log"); MARK_B=$(wc -l < "$BASE/B/node.log")
  T0=$(date +%s.%N)
  ask "$PROMPT" 16 q2 | tee -a "$OUT/replies.jsonl"
  echo "peerkv: q2 took $(python3 -c "print(round($(date +%s.%N) - $T0, 1))") s"
  tail -n +"$((MARK_A + 1))" "$BASE/A/node.log" > "$OUT/A.q2.log"
  tail -n +"$((MARK_B + 1))" "$BASE/B/node.log" > "$OUT/B.q2.log"
  # An up-front ask is only weighed (#252 (b)): the worker's refusal line is a
  # FORWARD's, and the ask's verdict is in the answer's DIAG line.
  echo "peerkv: B was asked up front: $(grep -c "answered a peer's up-front prompt admission" "$OUT/B.q2.log")," \
    "refused the ask: $(grep -c "answered a peer's up-front prompt admission.*Refused" "$OUT/B.q2.log")," \
    "refused a forward: $(grep -c 'refusing this prompt before prefill' "$OUT/B.q2.log")," \
    "A re-planned: $(grep -c 'retrying with fresh pipeline' "$OUT/A.q2.log")"
  grep -m3 -oE 'route=[a-z_]+ segments=[0-9]+ nodes=[0-9a-f,]+' "$OUT/A.q2.log" | sed 's/^/peerkv: /'
  grep -m2 -oE 'verdict=Refused.{0,200}' "$OUT/B.q2.log" | sed 's/^/peerkv: B asked: /'
  grep -m2 -oE 'refusing this prompt before prefill.{0,200}' "$OUT/B.q2.log" | sed 's/^/peerkv: B forward: /'
  exit 0
fi
if [ "$MODE" = latekv ]; then
  ask "What is the capital of France? Answer in one sentence." 16 q1 | tee "$OUT/replies.jsonl"
  echo "latekv: A's worker — $(grep -m1 'CPU worker KV budget' "$BASE/A/node.log" | grep -oE '(kv_budget_mb|typical_kv_mb|headroom_mb)=[0-9]+' | paste -sd' ')"
  PROMPT="Here are some notes on household appliances. $(for i in $(seq 1 120); do printf 'A refrigerator moves heat from its inside to the room using a refrigerant that evaporates in the cold coils and condenses in the warm ones; the compressor drives the cycle and the thermostat decides when it runs. '; done)Using only these notes, say in one sentence what the compressor does."
  printf '%s' "$PROMPT" > "$OUT/prompt.txt"
  MARK_A=$(wc -l < "$BASE/A/node.log"); MARK_B=$(wc -l < "$BASE/B/node.log")
  T0=$(date +%s.%N)
  ask "$PROMPT" 16 q2 | tee -a "$OUT/replies.jsonl"
  echo "latekv: q2 took $(python3 -c "print(round($(date +%s.%N) - $T0, 1))") s"
  tail -n +"$((MARK_A + 1))" "$BASE/A/node.log" > "$OUT/A.q2.log"
  tail -n +"$((MARK_B + 1))" "$BASE/B/node.log" > "$OUT/B.q2.log"
  # The worker logs `refusing this prompt before prefill` for BOTH kinds of
  # refusal; the coordinator's up-front line tells them apart.
  UPFRONT=$(grep -c 'KV admission up front — this node' "$OUT/A.q2.log")
  WORKER=$(grep -c 'refusing this prompt before prefill' "$OUT/A.q2.log")
  echo "latekv: A refused up front: $UPFRONT, A refused a forward: $((WORKER - UPFRONT))," \
    "A re-planned: $(grep -c 'retrying with fresh pipeline' "$OUT/A.q2.log")"
  grep -m3 -oE 'route=[a-z_]+ segments=[0-9]+ nodes=[0-9a-f,]+' "$OUT/A.q2.log" | sed 's/^/latekv: /'
  grep -oE 'prompt_memory_refused=true' "$OUT/A.q2.log" | sort | uniq -c | sed 's/^/latekv: candidate lines /'
  exit 0
fi
if [ "$MODE" = reasoning ]; then
  # #141: a reasoning model's finished scratchpad reaches the NON-streamed reply
  # as message.reasoning_content, as the stream sends it in
  # delta.reasoning_content; before, the non-streamed reply dropped it. Greedy,
  # so both should write the same tokens — after a warm-up, because a split's
  # FIRST request takes the n-gram path and its numerics can flip a near-tie
  # (the first run of this mode did, 40 tokens from the end). PASS = both carry
  # the same non-empty reasoning and the same content, and neither content
  # holds a scratchpad.
  Q="${RIG_QUESTION:-What is 17 + 25? Reply with just the number.}"
  ask "$Q" 16 warmup > /dev/null
  ask "$Q" 700 nonstream > "$OUT/reasoning.jsonl"
  body=$(python3 -c 'import json,sys
print(json.dumps({"model":sys.argv[1],"max_tokens":700,"temperature":0,"stream":True,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$Q")
  curl -s -N -m 900 -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
       -X POST localhost:8900/v1/chat/completions -d "$body" -o "$OUT/stream.sse"
  python3 - "$OUT/nonstream.body" "$OUT/stream.sse" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]
reasoning, content = m["message"].get("reasoning_content") or "", m["message"].get("content") or ""
s_reasoning, s_content, s_finish = "", "", None
for line in open(sys.argv[2]):
    line = line.strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        continue
    try:
        d = json.loads(line[5:])
    except ValueError:
        continue
    for ch in d.get("choices", []):
        delta = ch.get("delta") or {}
        s_reasoning += delta.get("reasoning_content") or ""
        s_content += delta.get("content") or ""
        s_finish = ch.get("finish_reason") or s_finish
print("non-streamed:", json.dumps({"finish": m["finish_reason"], "reasoning": reasoning[:200], "content": content[:200]}))
print("streamed:    ", json.dumps({"finish": s_finish, "reasoning": s_reasoning[:200], "content": s_content[:200]}))
leaked = any(t in c for c in (content, s_content) for t in ("<think>", "</think>"))
ok = bool(reasoning) and reasoning == s_reasoning and content == s_content and not leaked
print("reasoning: " + ("PASS" if ok else "FAIL") + f" (reasoning {len(reasoning)} vs {len(s_reasoning)} chars, content equal {content == s_content}, scratchpad in content {leaked})")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

if [ "$MODE" = split ]; then
  ask "What is the capital of France? Answer in one sentence." 64 q1 | tee "$OUT/replies.jsonl"
  ask "Write a short Python function that returns the factorial of n." 64 q2 | tee -a "$OUT/replies.jsonl"
  exit 0
fi

if [ "$MODE" = owntiming ]; then
  # The figure each node's planner was handed for the OTHER node, from that
  # request's `pipeline candidate` line: (stated_ms_per_layer, observed_ms_per_layer).
  seen() { # log node-hex-prefix line-count-before
    tail -n +"$(( $3 + 1 ))" "$1" | grep -a "DIAG: pipeline candidate node=$2" | tail -1 \
      | grep -oE "(stated_ms_per_layer|observed_ms_per_layer)=[A-Za-z0-9.()]+" | paste -sd' '
  }
  lines() { wc -l < "$1"; }
  IDA=$(curl -s -m 5 -H "Authorization: Bearer $KA" localhost:8900/api/admin/network-code \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])')
  KB=$(cat "$BASE/B/api_key")
  IDB=$(curl -s -m 5 -H "Authorization: Bearer $KB" localhost:8920/api/admin/network-code \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])')
  L0=$(lines "$BASE/A/node.log")
  ask "What is the capital of France? Answer in one sentence." 48 q1 | tee "$OUT/replies.jsonl" | cut -c1-160
  BEFORE=$(seen "$BASE/A/node.log" "$IDB" "$L0")
  ask "Write a short Python function that returns the factorial of n." 48 q2 | tee -a "$OUT/replies.jsonl" | cut -c1-160
  ask "Name three rivers in Europe." 48 q3 | tee -a "$OUT/replies.jsonl" | cut -c1-160
  echo "owntiming: waiting ${TIMING_WAIT_S:-75} s for two capability rounds"
  sleep "${TIMING_WAIT_S:-75}"
  L1=$(lines "$BASE/A/node.log")
  ask "What is the capital of Italy? Answer in one sentence." 16 q4 | tee -a "$OUT/replies.jsonl" | cut -c1-160
  A_OF_B=$(seen "$BASE/A/node.log" "$IDB" "$L1")
  # B holds every part but the first, so its plan needs A. A has run only its
  # OWN requests so far, which are never timed: B must be handed no figure.
  # The ask makes A serve B's first segment; after the rounds, B's next plan
  # is handed A's timing of THAT.
  ask_b() { # prompt max_tokens label
    local body
    body=$(python3 -c 'import json,sys
print(json.dumps({"model":sys.argv[1],"max_tokens":int(sys.argv[3]),"temperature":0,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$1" "$2")
    curl -s -m 600 -o "$OUT/$3.body" -w "{\"$3_status\":%{http_code}}\n" -H "Authorization: Bearer $KB" \
         -H "Content-Type: application/json" -X POST localhost:8920/v1/chat/completions -d "$body"
  }
  M0=$(lines "$BASE/B/node.log")
  # Long enough for A to serve well past the 4 steps a figure needs. B drives
  # it, and a coordinator that speculates with n-grams sends every step that
  # found no guess as a check of one position — the steps a first cut of this
  # never timed (2026-10-18): the line below says whether this run took them.
  ask_b "List the days of the week, one per line." 48 q5
  B_OF_A_OWN=$(seen "$BASE/B/node.log" "$IDA" "$M0")
  echo "owntiming: B drove q5 through n-gram checks: $(tail -n +"$(( M0 + 1 ))" "$BASE/B/node.log" | grep -ac 'ngram-only: starting')"
  sleep "${TIMING_WAIT_S:-75}"
  M1=$(lines "$BASE/B/node.log")
  ask_b "What is the capital of Portugal? Answer in one sentence." 16 q6
  B_OF_A=$(seen "$BASE/B/node.log" "$IDA" "$M1")
  echo "owntiming: A's plan, before any step:      B ${BEFORE:-<no candidate line>}"
  echo "owntiming: A's plan, after B served it:    B ${A_OF_B:-<no candidate line>}"
  echo "owntiming: B's plan, A ran only its own:   A ${B_OF_A_OWN:-<no candidate line>}"
  echo "owntiming: B's plan, after A served B:     A ${B_OF_A:-<no candidate line>}"
  python3 - "$BEFORE" "$A_OF_B" "$B_OF_A_OWN" "$B_OF_A" <<'PY'
import re, sys
before, a_of_b, b_of_a_own, b_of_a = sys.argv[1:5]
fig = lambda s: (re.search(r"stated_ms_per_layer=Some\(([0-9.]+)\)", s) or [None, None])[1]
ok = [
    ("A planned before any step with no figure for B", "stated_ms_per_layer=None" in before),
    ("A was handed B's own timing of the segments it served A", fig(a_of_b) is not None),
    ("B was handed no figure for A while A had run only its own requests",
     "stated_ms_per_layer=None" in b_of_a_own),
    ("B was handed A's own timing once A had served B", fig(b_of_a) is not None),
]
for what, good in ok:
    print(("PASS " if good else "FAIL ") + what)
print("owntiming:", "PASS" if all(g for _, g in ok) else "FAIL")
PY
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

if [ "$MODE" = cache ] || [ "$MODE" = handoffcache ]; then
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
    [ "$t" = turn2 ] && FROM=$(wc -l < "$BASE/A/node.log") && FROM_B=$(wc -l < "$BASE/B/node.log")
    curl -s -m 900 -D "$OUT/$t.hdr" -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
         -X POST localhost:8900/v1/chat/completions --data-binary "@$OUT/$t.json" -o "$OUT/$t.body"
    echo "$MODE: $t -> $(head -1 "$OUT/$t.hdr" | tr -d '\r')  route=$(grep -i '^x-swarm-route' "$OUT/$t.hdr" | cut -d' ' -f2- | tr -d '\r')"
  done
  if [ "$MODE" = handoffcache ]; then
    # FUTURE_WORK #3 (c): turn 2 is priced for what B was handed on turn 1.
    # B's worker logs its HIT to B's log, after FROM_B (taken before turn 2).
    python3 - "$BASE/A/node.log" "$BASE/B/node.log" "$OUT" "$FROM" "$FROM_B" <<'PY'
import json, re, sys
log_a = open(sys.argv[1], errors="replace").read().splitlines()[int(sys.argv[4]):]
log_b = open(sys.argv[2], errors="replace").read().splitlines()[int(sys.argv[5]):]
out = sys.argv[3]
def last(log, pat):
    hits = [l for l in log if re.search(pat, l)]
    return hits[-1] if hits else ""
credit = last(log_a, r"handed this prompt's opening to still holds it")
handed = [int(t) for t in re.findall(r", (\d+)\)", credit)]
hit = last(log_b, r"DIAG: prefix-cache HIT model_key")
m = re.search(r"matched_tokens=(\d+)", hit)
matched = int(m.group(1)) if m else None
print(f"handoffcache: planner credit={handed or None}  B's worker HIT matched={matched}")
print(f"handoffcache: planner line: {credit[:260] or '(none)'}")
ok = bool(handed) and matched is not None and matched >= max(handed)
print("handoffcache: PASS" if ok else "handoffcache: FAIL")
sys.exit(0 if ok else 1)
PY
    exit $?
  fi
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
  if [ -n "${WARM_SPARE:-}" ]; then
    # Both peers WARM, only one with this conversation (FUTURE_WORK #10's
    # residual): the peer that did not store turn 1 leads a conversation of its
    # own through the same split, so it loads the same layers and stores ITS
    # opening. A can then tell the two apart only by what it believes each holds.
    if grep -q "stored a split's prompt" "$BASE/B/node.log"; then KEPT=B OTHER=C OPORT=8940; else KEPT=C OTHER=B OPORT=8920; fi
    python3 - "$MODEL" "$OUT" <<'PY'
import json, sys
model, out = sys.argv[1], sys.argv[2]
system = "You are a patient tutor for a student learning chemistry. " + " ".join(
    f"Point {i}: explain one idea at a time, check the student understood it, and give a short example." for i in range(1, 21))
json.dump({"model": model, "max_tokens": 8, "temperature": 0,
           "messages": [{"role": "system", "content": system}, {"role": "user", "content": "What is an atom?"}]},
          open(f"{out}/warm.json", "w"))
PY
    curl -s -m 900 -o "$OUT/warm.body" -H "Authorization: Bearer $(cat "$BASE/$OTHER/api_key")" \
      -H "Content-Type: application/json" -X POST "localhost:$OPORT/v1/chat/completions" --data-binary "@$OUT/warm.json"
    # Its resident layers reach A with its next capability (published on change, ~30 s ticks).
    sleep 60
    echo "splitcache: turn 1 kept on $KEPT; $OTHER warmed with its own conversation (its stores: $(grep -c "stored a split's prompt" "$BASE/$OTHER/node.log"), reply: $(head -c 120 "$OUT/warm.body"))"
  fi
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
c_hits = sum("HIT on a split's stored prompt" in l for l in c)
b_stored = sum("stored a split's prompt" in l for l in b + c)
missed = any("a resumed prompt pass did not finish" in l and "Prompt cache miss" in l for l in a[from_a:])
print(f"splitcache: turn 1 resumed_from={t1}  turn 2 resumed_from={t2}  B hits on turn 2={b_hits}"
      f"{f' (of them on C: {c_hits})' if c else ''}  "
      f"B stores={b_stored}  miss logged={missed}  replies 200={both_200}  "
      f"seconds {[t['seconds'] for t in turns]}")
if off:
    ok = both_200 and not t1 and not t2
    arm = "off (SWARMLLM_SPLIT_PROMPT_CACHE=0)"
elif miss:
    ok = both_200 and missed and 0 in t2
    arm = "miss (B restarted between the turns)"
else:
    # Every pass before turn 2 started from 0: turn 1's, and with WARM_SPARE the
    # other conversation's too when its peer hands the lead to A (a plan
    # naming none of its own segments is led by its head).
    ok = both_200 and bool(t1) and not any(t1) and bool(t2) and max(t2) > 0 and b_hits >= 1
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
    ask "$PROMPT" "${REPEAT_MAX_TOKENS:-120}" "repeat$i" | tee -a "$OUT/repeat.jsonl" | cut -c1-160
  done
  # Not "the first only" any more: since the tail walks the guesses (a few ids back,
  # not a vocabulary) the payoff gate keeps the loop on while rounds average past its
  # bar — llama-3.2-3b on this prompt reads 1.30 tokens/round, all 3 of 3 on .218 too.
  echo "repeat: n-gram path taken by $(grep -c 'try_ngram_only_distributed ELIGIBLE' "$BASE/A/node.log") of ${REPEAT:-3} requests (payoff_x100: $(grep -a -o 'payoff_x100=[0-9]*' "$BASE/A/node.log" | tail -1))"
  echo "repeat: streamed replies by peers crossed: $(grep -a 'DSD: streamed checks complete' "$BASE/A/node.log" | grep -o 'peers=[0-9]*' | sort | uniq -c | paste -sd' ')"
  if [ -n "${SHARDS_C:-}" ]; then
    echo "repeat: chains that failed and finished unchained: checks $(grep -ac 'chained check failed — re-running it unchained' "$BASE/A/node.log"), streams $(grep -ac 'the chained stream failed — finishing the reply in rounds' "$BASE/A/node.log"); B dialled its next hop $(grep -ac "Dialing a chained run's next hop" "$BASE/B/node.log")x"
  fi
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
  # A drives refusechain and killchain itself: only A's plan matters there.
  b_plans() { [ -n "$REFUSECHAIN$KILLCHAIN" ] || covered "$KB" 8920; }
  for _ in $(seq 1 60); do covered "$KA" 8900 2>/dev/null && b_plans 2>/dev/null && break; sleep 3; done
  covered "$KA" 8900 && b_plans || { echo "remote: A or B never planned the model across the others"; exit 1; }
  if [ -n "$KILLCHAIN" ]; then
    ask "Explain in detail how a refrigerator works, step by step." 400 kill > "$OUT/kill.jsonl" &
    ASK=$!
    until [ "$(grep -c 'DIAG: segment result received' "$BASE/A/node.log" 2>/dev/null)" -ge 40 ]; do
      kill -0 $ASK 2>/dev/null || { echo "killchain: the reply finished before the kill point"; break; }
      sleep 0.5
    done
    CW=$(pgrep -P "$PC" | paste -sd' ')
    echo "killchain: killing C ($PC${CW:+, workers $CW}) after $(grep -c 'DIAG: segment result received' "$BASE/A/node.log") results"
    TK=$(date +%s.%N)
    kill -9 "$PC" $CW
    PC=""
    wait $ASK
    secs=$(python3 -c "print(round($(date +%s.%N) - $TK, 1))")
    status=$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["status"])' "$OUT/kill.jsonl")
    departed=$(grep -ac 'peer departed with forwards outstanding' "$BASE/A/node.log")
    hop=$(grep -ac 'passing the next hop' "$BASE/B/node.log")
    timed=$(grep -ac 'segment TIMED OUT' "$BASE/A/node.log")
    echo "killchain: ended $secs s after the kill, $status; A failed it on departure: $departed," \
      "B passed a failure on: $hop, A timed out: $timed"
    if python3 -c "import sys; sys.exit(0 if $secs <= ${KILL_SLACK:-60} else 1)" && [ "$timed" = 0 ]; then
      echo "killchain: PASS"; exit 0
    fi
    echo "killchain: FAIL"; exit 1
  fi
  if [ -n "$REFUSECHAIN" ]; then
    T0=$(date +%s.%N)
    ask "Explain in three sentences why the sky is blue." 48 q1 | tee "$OUT/replies.jsonl" | cut -c1-300
    secs=$(python3 -c "print(round($(date +%s.%N) - $T0, 1))")
    refused=$(grep -c 'FAULT: treating this forward as undecryptable' "$BASE/C/node.log")
    at_b=$(grep -a 'dispatcher received LayerResult' "$BASE/B/node.log" | grep -c 'Could not decrypt')
    passed=$(grep -ac 'passing the next hop' "$BASE/B/node.log")
    took=$(grep -ac 'LayerResult delivered to pipeline.*Could not decrypt' "$BASE/A/node.log")
    timed=$(grep -ac 'segment TIMED OUT' "$BASE/A/node.log")
    unchained=$(grep -ac 'chained run failed' "$BASE/A/node.log")
    status=$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["status"])' "$OUT/replies.jsonl")
    echo "refusechain: took $secs s, $status; C refused (fault): $refused, B received it: $at_b," \
      "B passed it on: $passed, A took the refusal: $took, A timed out: $timed, chained runs abandoned: $unchained"
    if [ "$refused" = 1 ] && [ "$passed" = 1 ] && [ "$took" = 1 ] && [ "$timed" = 0 ]; then
      echo "refusechain: PASS"; exit 0
    fi
    echo "refusechain: FAIL"; exit 1
  fi
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

if [ "$MODE" = continue ] && [ -n "${NONSTREAM:-}" ]; then
  id16() { curl -s -m 5 -H "Authorization: Bearer $(cat "$1")" "localhost:$2/api/admin/stats" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"][:16])'; }
  IB=$(id16 "$BASE/B/api_key" 8920); IC=$(id16 "$BASE/C/api_key" 8940)
  Q="Explain in detail how a refrigerator works, step by step, covering the compressor, the condenser, the expansion valve and the evaporator."
  BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"max_tokens":300,"temperature":0,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$Q")
  read -r t0 _ < /proc/uptime
  curl -s -m 900 -H "Authorization: Bearer $KA" -H "Content-Type: application/json" \
    -X POST localhost:8900/v1/chat/completions -d "$BODY" -o "$OUT/continue.json" -w '%{http_code}' > "$OUT/continue.code" &
  CURL=$!
  for _ in $(seq 1 120); do grep -aq 'remote-generate fast path: request sent' "$BASE/A/node.log" && break; sleep 0.5; done
  sleep "${KILL_AFTER_S:-8}"
  SERVER=$(grep -a 'remote-generate fast path: request sent' "$BASE/A/node.log" | tail -1 | grep -oE 'target=[0-9a-f]+' | cut -d= -f2)
  case "$SERVER" in "$IB"*) VICTIM=$PB ;; "$IC"*) VICTIM=$PC ;; *) VICTIM="" ;; esac
  WPID=$( [ -n "$VICTIM" ] && pgrep -P "$VICTIM" -f model-worker | head -1 )
  echo "continue: A handed it to ${SERVER:-?}; killing its worker (pid ${WPID:-none}) ${KILL_AFTER_S:-8} s in"
  [ -n "$WPID" ] && kill -9 "$WPID"
  wait $CURL
  read -r t1 _ < /proc/uptime
  restarted=$(grep -c "retrying with fresh pipeline" "$BASE/A/node.log")
  continued=$(grep -c "continuing it on a fresh route\|continuing the reply on a fresh route" "$BASE/A/node.log")
  salvaged=$(grep -c "returning the partial reply" "$BASE/A/node.log")
  python3 - "$OUT/continue.json" "$(cat "$OUT/continue.code")" "$t0" "$t1" "$restarted" "$continued" "$salvaged" "${WPID:-}" <<'PY'
import json, sys
path, code, t0, t1, restarted, continued, salvaged, wpid = sys.argv[1:9]
try:
    body = json.load(open(path))
    choice = body["choices"][0]
    text, finish = choice["message"]["content"] or "", choice.get("finish_reason")
    tokens = body.get("usage", {}).get("completion_tokens")
except Exception:
    text, finish, tokens = "", None, None
opening = text[:40]
once = bool(opening) and text.count(opening) == 1
print(f"continue: HTTP {code} in {float(t1) - float(t0):.1f} s, finish={finish}, completion_tokens={tokens}, "
      f"chars={len(text)}, opening once={once}")
print(f"continue: A re-planned from the first token {restarted} time(s), continued {continued} time(s), "
      f"handed back a partial {salvaged} time(s)")
print("continue: reply: " + text[:300].replace("\n", " "))
ok = bool(wpid) and code == "200" and finish not in (None, "interrupted") and once and int(continued) >= 1
print("continue: PASS" if ok else "continue: FAIL" + ("" if wpid else " (no worker was killed)"))
sys.exit(0 if ok else 1)
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
  python3 - "$OUT/continue.sse" "$AT_KILL" "$continued" "$served_by_other" "${WPID:-}" "$IB" "$IC" <<'PY'
import json, sys
path, at_kill, continued, other, wpid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
peers = {sys.argv[6][:8], sys.argv[7][:8]}
text, chunks, finish, error, route = "", 0, None, None, None
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
    route = (ev.get("swarmllm") or {}).get("route") or route
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
print(f"continue: route {json.dumps(route)}")
named_both = bool(route) and peers <= set(route.get("all_nodes") or [])
print(f"continue: the route names both peers: {named_both}")
ok = bool(wpid) and error is None and finish is not None and after > 0 and once and continued >= 1 and other >= 1 and named_both
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
