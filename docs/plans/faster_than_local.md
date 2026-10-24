# As fast as local, or faster: what distance allows, and the plan

**Written 2026-09-27 from a deep dive**: three research surveys (limits of split
decode, big models on one machine, where many machines beat one), the replayed
speculation measurements in `split_speculation.md`, and the v0.3.210 gate. The
question was: *a split across two GPUs decodes at 2.85 tok/s where each GPU
alone does 11-44 — how does the swarm become as fast as local, or faster?*

## 1. The limit, and why no scheduling trick removes it

Token t+1 enters the first layer only after token t has left the last. When
the layers are on two machines, every token crosses A→B and back:

    time per token  ≥  compute of every layer  +  one round trip

Nobody has published a theorem for this, because it is simply the dependency.
Petals measured it to the millisecond: Llama-2-70B on three T4s lost exactly
200 ms per token when a 50 ms one-way delay was added to each of its four hops
(arXiv 2312.08361, Table 2). With a 206 ms ICMP round trip Thailand↔Belgium,
**exact decoding of an unmodified model tops out near 4 tok/s**, and nothing
about the GPUs changes that.

**Where splitting a model IS faster than one device:** Groq (~40 µs per layer
across ~576 chips), Cerebras (~110 µs per token of link latency across four
wafers), TPU tensor parallelism (Pope et al., 2211.05102), and Exo over
Thunderbolt RDMA (<50 µs). All of them add up memory bandwidth across chips
over MICROSECOND links. Our RTX 3070 spends ~0.8 ms per layer (22.7 ms for 28),
so a layer split stays within 10% of local only below **~2 ms** round trip, and
tensor parallelism needs ~56 syncs per token each well under 400 µs. A 206 ms
link is two to three orders of magnitude past both. That is LAN territory, and
nothing else.

The only three ways to break the cycle for an off-the-shelf model:

1. **Do not cross the network per token** — keep decode on one machine.
2. **Shorten the round trip** — machines close to each other; our own overhead gone.
3. **Compute a token before the one before it is known** — verified prediction
   (speculative decoding, Lookahead). Output provably identical (Leviathan
   2211.17192, Chen 2302.01318); only the speed depends on how often it is right.

Everything else found needs a retrained model (StagFormer 2501.15665, Parallel
Loop Transformer, Ladder Residual, CLLM, parallel attention+MLP) or µs RDMA
plus large batches (Lamina, MegaScale-Infer, Step-3 AFD). Layer Parallelism
(2502.02790) is training-free but lossy (Llama-2-7B perplexity 6.26 → ~9.1) and
still needs NVLink-class links.

## 2. What that means, case by case

| The user's situation | Fastest honest answer | Expected |
|---|---|---|
| Model fits their GPU | Run it locally. The swarm speeds up what is NOT per-token: prompts already read elsewhere, independent requests fanned out | Decode = local; agent turns 5-20× faster to first token (§3.1) |
| Model is an MoE that fits GPU + RAM | Local, experts in system RAM, everything else on the card | Qwen3-30B-A3B class: **30-45 tok/s** on an 8 GB card + 32 GB RAM (measured by others, §3.2) |
| Dense model a little too big for the card | Local: a quantization that fits the card, or card + processor with the processor part fast | 14B on an 8 GB card: llama.cpp 6.8-7.0 tok/s; a default node 6.8-6.9 once decode ran inside its pool (§3.3) |
| CPU-only user, long prompt | Hand the prompt to a GPU peer | 30-50× faster to first token (§3.1) |
| Model fits nowhere, machines close (≤ 25 ms) | Split + verified prediction with a small drafter | ~local speed (`split_speculation.md` projection: 38-50 tok/s at 25 ms) |
| Model fits nowhere, machines far (~300 ms) | Split + verified prediction with a SHADOW of the far layers | 8-15 tok/s with a small drafter; 25-35 with a shadow (§3.5) |

## 3. The levers, ranked

### 3.1 Where many machines genuinely beat one (no per-token network cost)

- **Prefix affinity.** Agent traffic is ~100:1 input to output, and 85-97% of a
  turn's tokens repeat the last turn (Preble 2407.00023, Manus). Route a request
  to the node that already holds its prefix's cache instead of re-reading it:
  Prompt Cache 8× on GPU and 60× on CPU (2311.04934); Preble 1.5-14.5×. A request
  is kilobytes and its cache is hundreds of megabytes, so move the request. This
  node already prices ITS OWN warm prefix (`scheduler::cached_prefix`); the gap is
  peers' prefixes (gossiped digests), a TTL beyond the 10-minute idle expiry
  (agent sessions pause longer), and a RAM/disk tier.
- **CPU users hand long prompts to GPU peers.** A 10K-token prompt on a 7B:
  ~14 s on an RTX 3070 against 7-14 minutes on a laptop CPU. The delegation
  path exists (`delegation_target`); the cost model must compare the requester's
  measured prompt rate with round trip + the peer's rate.
- **Fan-out.** Independent requests (subagents, parallel tool branches, best-of-n)
  to distinct warm peers: ~2.8-3× over one 3070 batching four, bounded by how much
  of a task is parallel (research agents: a lot; coding agents: little).
- **Prompt reading pipelined across a split** (`split_speculation.md` Phase 3):
  large where the alternative is CPU offload; ≤ 1.7× for a model that fits.

Ruled out for our links: ring/sequence/tensor parallelism (needs ~460K-token
blocks at 100 Mbit/s to hide transfer), shipping KV caches between machines for
a GPU user (573 MB fp16 for 10K tokens of a 7B; wins only for CPU users).

### 3.2 Run MoE models the way consumer machines can

Decode speed ≈ memory bandwidth ÷ bytes read per token, and an MoE reads only its
active experts. llama.cpp's `-ot exps=CPU` / `--n-cpu-moe` keeps attention, the
router and shared weights on the card and the experts in system RAM:
Qwen3-Coder-30B-A3B at **32.5 tok/s** on an 8 GB RTX 3060 Ti, a 35B-A3B at
**44.4** on an 8 GB RTX 2070S; gpt-oss-20b at 56-64 on a 12 GB card. Our hybrid
placement was per LAYER (`split::hybrid::LayerPlacement`), so a large MoE went
processor-heavy for everything. **Per-tensor placement — experts on the
processor, the rest on the card — SHIPPED in v0.3.237** (FUTURE_WORK #138 (3)):
stage A 2026-10-13 (`LayerPlacement::experts_device_for`, decode 1.2-1.7x at equal
card bytes), stage B 2026-10-14 (`process_pool::card_split` chooses it for a MoE too
big for the card: ~10% faster decode, ~0.4-0.7 GB less of the card than whole layers).

**Measured on THIS laptop 2026-10-11 (FUTURE_WORK #138):** the figures above are
8 GB desktop cards. On the RTX 3070 Laptop + Ryzen 7 5800H, Qwen3-30B-A3B Q2_K at
~6.5 GB on the card, llama.cpp gets 23.4-23.7 tok/s from whole layers and 32.3 with
the experts of 20 layers in RAM — 1.37x, not the ~3x the desktop figures implied.
And at EQUAL placement we ran at ~40% of llama.cpp (10-13 vs 21-25 tok/s), mostly
the card's expert path (~1.6 ms of host overhead a layer). The processor's decode
step and then the card's experts (kept stacked, routed on the card, one launch per
projection) closed that the same day: a 40-45 ms step, ~23 tok/s, level with
llama.cpp's 21.5-25.4 at the same 26 layers. A PROMPT was then the gap — 681 tokens
spent ~15.5 s of ~22 s in the processor layers' experts — closed by #138 (1)-(1e)
(2026-10-13: a split reads its prompt on the card), before placement shipped.

### 3.3 Make our own card + processor path as fast as the memory allows

**Measured 2026-10-11 (FUTURE_WORK #137).** The 3.35 tok/s / ~17 GB/s premise was a 09-27
figure and no longer holds. On the Ryzen 7 5800H + RTX 3070 laptop, Qwen2.5-14B Q4_K_M with 30
of 48 layers on the card, cooled before each arm: llama.cpp 6.8-7.0 tok/s (`llama-bench -ngl 30
-t 4`); a default node 6.79-6.93 once decode runs INSIDE its thread pool — which it did not do
until 2026-10-11 (5.64-5.93 in the same A/B; § Closed in FUTURE_WORK; a 3B on a processor-only
node 10.97-13.04 → 14.53-15.29). What remains (#137, P3, measured in ONE session 2026-10-15 on
a 7B, processor only): 3% behind llama.cpp at 4 threads and 5-8% at 8, both gaining a little from 4
to 8 (the 8-thread loss was fixed 2026-10-14: idle rayon workers spin). Every product already reads
its weights at the laptop's memory bandwidth (38-41 GB/s); the gap is what runs between them.
llama.cpp's Q4_K repack, huge pages and the one-row split are REFUTED (gotchas #843/#844,
FUTURE_WORK #137). Two machine effects, shared with
llama.cpp: the card clocks down between split steps, and back-to-back runs heat the laptop
enough to fake a 30% gap. Rule of thumb from the limit above still holds: at ~50 GB/s one 206 ms
round trip reads ~10 GB of weights, so **local offload beats a far split whenever less than ~10
GB sits off the card** — which is every 14B.

### 3.4 Remove our own overhead from every round trip (exact, no prediction)

- **Protocol negotiation costs a full round trip per token on the default path.**
  libp2p opens a substream per request-response message and negotiates it with
  multistream-select `Version::V1`, which is "always at least one dedicated
  round-trip message exchange before application data" (multistream-select
  0.13.0 docs). `Version::V1Lazy` makes it 0-RTT when the dialer offers one
  protocol. The request-response crate is vendored, so the tensor protocol's
  outbound substreams can take V1Lazy alone, leaving gossip/kad/identify as they
  are. Measured per token today: 838 ms on request-response vs 351 ms on the
  persistent stream, at a 412 ms application-level round trip.
- **The persistent stream** (2.4×, measured) waits on #133 (a peer stalling on a
  large prompt frame; no receipt ACK). Chunking the prompt pass (Phase 3)
  removes the large frame.
- **~110 ms per token is still unexplained** on the stream path (351 ms against
  206 ms ICMP + ~35 ms compute). Candidates: relay vs direct, libp2p ping vs ICMP
  on the same connection, `tcp_slow_start_after_idle` resetting the window below
  one fp32 frame. Measure before fixing.

### 3.5 Verified prediction for the split that cannot be avoided

`split_speculation.md` has the measurements. In one line: time per token ≈
1/v + m·RTT, where v is the drafting side's local speed and m how often the far
side disagrees. A small same-family drafter (m ≈ 26%) gives 8-15 tok/s at 300 ms
and near-local at 25 ms. A **shadow** of the far layers (a 2-3-bit copy on the
near machine; its errors cost speed, never correctness) is estimated at
α ≈ 0.95-0.97 for a Q3 remote half (ML-SpecQD measured Qwen2.5-Coder-7B at 91% for
an MXFP4 whole-model copy against 62% for its 0.5B sibling) → 25-35 tok/s at
300 ms. **Shared randomness** (Gumbel-max with a position-keyed seed, "drafter-
invariant speculative decoding", Daliri et al. 2408.07978) keeps that agreement
at the temperatures clients actually send. Measurement: `~/swarmllm-ref/spec/`
`shadow.py` + `coupling.py`, results below when run.

## 4. The order of work

Status 2026-10-14: items 1, 2, 3, 4 and 6 have shipped (all released, the last in v0.3.237);
5 is half done and 7's chunked prompt pass shipped (v0.3.234) while the persistent stream stays
off by default. `docs/FUTURE_WORK.md` carries what is left: item 3's residual is #137, item 5 is
#10 (the split half shipped in v0.3.234), item 7 is #133 (with V1Lazy, request-response is as fast, so
there is no speed reason to default the stream on). The shadow of §3.5 is decided against as a
default (#144, § "Decided"): no design may need a user to hold a whole model, even at low bits.

1. ✅ **Ship v0.3.211**: the card/processor batched-forward fix found at the .210
   gate (every concurrent request to a model split between card and processor
   failed on .209 and earlier).
2. ✅ **V1Lazy for the tensor protocol** (§3.4) — small, exact, every split user.
   Measure on the real link, never loopback (gotcha #739).
3. ✅ **Our processor half** (§3.3) — measured 2026-10-11: decode ran outside its thread pool on
   every default node (fixed: a 14B split level with llama.cpp, a 3B processor-only 1.14-1.39x);
   left: ~5% behind llama.cpp at 4 threads and a ~40% loss at 8 on the processor alone (#137, P3).
4. ✅ **MoE experts on the processor** (§3.2) — #138 (3), v0.3.237.
5. **Prefix affinity across peers** (§3.1), then CPU users' prompt hand-off pricing — #10, PARTIAL.
6. ✅ **Split speculation Phase 1** (accept at the tail — shipped; the stream of checks followed in v0.3.216),
   then the shadow + coupling measurement decides whether Phase 4 is built.
7. **#133** (the chunked prompt pass shipped as #171 in v0.3.234; left: pieces chained between two
   peers for a pass that does not keep its prompt), then the persistent stream by default.

## 5. Where the flow stands (2026-10-23), and what is left

The flow (#134 (b), v0.3.248) runs the split a round trip and nothing else: **tok/s ≈ E / P**, E
the tokens its live rows get right per round trip, P the round trip. On TH↔AU (P ≈ 430 ms) E is
~6.5, 14-16.5 tok/s with the coder-1.5B Q8_0 drafter (`docs/invariants/scheduling.md` § "A tree
that flows"). Rows, levels and the live budget do not move E (E grows ~log in the rows; the sweep
found no knob beating the default). So the levers are E's quality and P.

**E — measured and built 2026-10-23** (`docs/invariants/inference.md`):
- A sampled reply's guesser read every row whole (608 KB each): drafting 41-48 ms a forward against
  greedy's 24-26 on the real link. Ranked on the card now (top-k radix select): 24-28 ms.
- A sampled tree rated its alternatives by the drafter's probability at T, ignoring the noise the far
  end draws with: second child over-rated by half, tail at half its worth, misses ~2× greedy's.
  Rated by `logit + T·g` now (fitted, calibrated at T 0.7 and 1.0): trees of 128 grow 6.90 tokens a
  round against 6.15 offline (greedy 7.19).
- Refuted offline: an online per-token bias from the far end's answers (no gain, worse when strong).

**E — what is left, by evidence:**
1. *A better drafter.* Agreement is what E is made of (rank-0 hit 0.755 for coder-1.5B vs the 7B;
   a tree's E ≈ the run a spine survives, ~1/(1−α) plus what branches rescue). The 1.5B over the
   0.5B gave +6% on the real link (16.46 vs 15.57) with misses 10-11 vs 17-19. For the targets users
   split (32B-72B) a 1.5B-7B drafter is allowed (`DRAFTER_MAX_FRACTION` 4) and fits beside the head.
2. *A drafter fed the target's own state — trained ONLINE from the far end's answers* (our design,
   not published as such): the requester computes the target's split-layer state for every row it
   sends, and the far end's draw for that row comes back — a labelled example per row, ~12K a
   400-token reply, for free. A small head (a tuned-lens translator, Belrose 2303.08112, or a
   steering vector into the drafter, 2511.09844 — +22-35% accepted tokens there) learned on that
   stream needs no download and no full copy. **Why the split layer and not the top**: a flow's
   drafter grows rows far past the walk, and any state from the far end (its top layer, an MTP
   head's drafts, `large_model_splits.md` build-order step 4) reaches the requester a round trip
   after the row it describes — by then the tree has grown beyond it, so it informs only the first
   step past the walk. The split layer's state exists for EVERY row the moment this node's half has
   run it, the deepest guesses included: the one piece of the target a flow can use where its tree
   actually grows. Unmeasured: how well the split layer predicts the
   next token at all (no published top-1 at ~50% depth). **The literature lowers the ceiling**
   (searched 2026-10-23): SD² (2511.09844) taps layers 3, L/2 and L−2 into a FULLY fine-tuned
   drafter and gains +8-10% block efficiency over a 0.6B drafter, +4% over a 1B (+38% only over a
   160M one); a stock model's mid-depth exit is near useless (LayerSkip 2404.16710: Llama3-8B at
   layer 16, HumanEval 0); online distillation (2310.07177) gained most for weak drafters. Against
   our 75.5%-agreeing 1.5B, expect a few points. First step, only if nothing better is open: dump
   the split-layer state and the target's token from our engine, fit an affine map + the target's
   norm and head by ridge regression, and measure held-out top-1 and the mix with the drafter's top
   k — **stop if the head alone is under ~40% or the mix under ~78%.** **Run 2026-10-24 and
   STOPPED** (§ 7.2): at an even split the trained head is right 32.8% among the drafter's 8 and
   the mix gains 0.2 points.
3. *Ruled out for now:* PARD-style parallel drafting (2504.18583) makes drafting cheaper, and the
   requester's drafting is not what binds E ("a cheaper requester grows more rows, not more reply");
   EAGLE-3 heads need the TOP layer's state, which is a round trip away, and none exist for Qwen2.5.

**P:** the round trip is geography. Near chains (Stage 3, v0.3.247) run the loop among close peers
when they hold the model; detour routing (#327: TH↔AU crosses Los Angeles, ~390 ms against ~185 ms
direct) needs a well-placed peer the swarm does not have yet.

## 6. Past the published methods: what this system alone can do (2026-10-23)

Published speculative decoding assumes one machine holds the target. Ours never does, and that
changes what is possible as well as what is not. Each item says what it rests on and what decides it.

**6.1 The law that bounds E, and what it rules out.** A best-first tree of N rows covers the paths
whose total surprise to the target is about ln N, so **E ≈ 1 + ln(N) / H**, H the target's mean
surprise (nats) at the guesser's ratings. It fits the offline trees: greedy H 0.858, N 128 → 6.65
(simulated 7.19); sampled before/after the shared-noise ratings H 0.965/0.883 → 6.03/6.50 (simulated
6.15/6.90). Doubling E takes N squared (16K rows a round trip — no uplink carries that) or H halved,
and only guesses written by something close to the target halve H. **But the measured flow's speed
follows E only weakly** (below): what holds a flow at ~15-18 tok/s on a 430 ms link is not settled,
so each design below says which quantity it moves and is judged on the link, never by E alone.

E/P itself is Little's law — the reply's rows in flight are its speed times the round trip — so it
says what a flow's speed IS, not what holds it there. Two caps could: **coverage** `tok/s ≤ D(N)/P`,
and **rows** `tok/s ≤ R / (rows grown a token)`, R the rows a second this node sends. Measured on
TH↔AU (`realrate.sh`, `realwide.sh`, 2026-10-23, coder-1.5B Q8_0):
- A reply grows 22-28 rows a token, and this node's busy time is ~1.3-1.6 ms a row (24 rows a call:
  38 ms busy; 37 rows: 47 ms). The best replies seen ran 15.6-18.2 tok/s, waiting 4.6-4.9 s of 22-26.
- **Speed does not follow D**: the 0.5B → 1.5B drafter raised D 41% offline (`treesim.py`, 5.08 →
  7.19 at 128 rows) and misses fell 42% on the link, but speed rose 6%; the shared-noise ratings cut
  misses a third for +4.5%. The 3B (+14% D over the 1.5B offline, `gumbeldump.py` c3-T07, top-1
  77.4% against 75.5%, at a costlier call) is not worth relaxing `DRAFTER_MAX_FRACTION` for.
- **A cheaper drafter pays, damped**: the same 0.5B as Q8_0 against fp16 (same guesses, 15 against 28
  ms a call) ran 15.57 against 13.98 tok/s (+11%) and waited 5 s MORE a reply (scheduling.md § "A
  tree that flows", the drafter table) — half the saving became waiting. The drafter is ~65% of this
  node's busy time; this node's work, not the link, is ~75% of a reply (busy 18-22 s, stalls 3-9 s).
- **Wider rows did not pay either**: 16 a level (37 rows a call, 21% cheaper a row busy) ran
  15.0-16.2 against 8's 15.6-18.2, waiting MORE (7.6-9.4 s against 4.6-4.9) and growing more rows a
  token. Neither cap alone explains both results; the flow's dynamics (rows killed at each advance,
  the live budget refilling, misses) are what a model of it must capture next.
- **This node's drafting does not bind the flow** (`realdelay.sh`, 2026-10-23, the same binary
  and guesses, each drafter call made 0 / 12 / 25 ms longer by `SWARMLLM_FAULT_DRAFT_DELAY_MS`,
  arms ABCCBA, greedy, 400 tokens × 2, coder-1.5B Q8_0, least answer ~400 ms):
  | Delay | tok/s | Drafting a call | Drafting a reply | Waiting | Forwards | Misses |
  |---|---|---|---|---|---|---|
  | 0 ms | 12.44-14.26 (13.22; unqueued 13.77) | 27-31 ms | 15.0-18.4 s | 4.4-8.2 s | 553-627 | 14-17 |
  | 12 ms | 13.02-13.16 | — | 20.1-21.1 s | 1.1-2.8 s | 435-490 | 12-14 |
  | 25 ms | 13.01-13.36 (13.19) | 48-50 ms | 21.9-23.2 s | 1.3-2.0 s | 454-470 | 11-16 |

  Nearly doubling the drafter's call cost 0-4%. The flow took it out of its waiting and sent fewer,
  fuller forwards. So taking the drafter off this node — the far end drafting with the prompt in
  hand, or a LAN machine as a drafting service (6.4) — would buy at most those few percent; not
  built. (One d12 arm ran no flow: its 8-token warm-up fitted the check cost as 12 ms fixed + 393 ms a
  guess and remembered "don't guess" — `dsd_controller`'s round-trip floor, 2026-10-23.) Today's d0
  ran below the 10-23 morning's 15.6-18.2 at the same settings: the far end queued in every arm
  (`queue_ms_most` 280-1,130 ms).
- ⚠ **Every real-link arm so far ran over TCP.** The harness starts a fresh node, which met the
  Melbourne peer through peer exchange. Peer exchange dialled the first address listed, and that
  was TCP in 6 of 6 `realdelay.sh` arms, while the live node uses QUIC to the same peer. So every
  flow figure on this link is Cubic plus slow start after idle, not BBR. Fixed 2026-10-23
  (`docs/invariants/network.md` § "One dial per PEER"). **Measured the same day, transport forced
  per arm, one binary: QUIC 16.68 tok/s against TCP 13.52** (8 arms each, interleaved; coder-1.5B
  Q8_0 drafter, 400 greedy tokens). So the 10-23 figures above (TCP) understate the flow by about
  a fifth; QUIC is the number a node gets once its peers (the anchor first) run 0.3.249.
- **The largest swing is the far end**: its queue (v0.3.248 reads every row back whole for its
  draws) took 3 of 10 and 2 of 6 arms to 10-13 tok/s. v0.3.249's far-end card draw shortens each
  row there; the episodes coming and going also point at that machine's own load or the path.

**6.2 The far end's lead cannot shorten a miss.** The far end learns the true token half a round
trip before this node, but every guess must pass this node's half before the far end can check it,
and the far end cannot compute that half. So no protocol where the far end only reorders, redraws
or re-ranks recovers a miss's round trip — it costs P whatever the far end does (checked case by
case: far-end drafting, far-end top-feature drafting à la EAGLE, transplanting a sibling row's
state). Two exits exist: a cycle that leaves this node out (near chains), or a far end that can
GUESS without this node's half — the target's exact tail fed an approximation of the head (6.6).

**6.3 Recycling the far end's draws on dead branches — REFUTED offline (2026-10-23).** A flow draws
EVERY row it is sent: ~12K draws for a 400-token reply, 97% of them on branches that died, each the
target's own next token in a context one alternative away from the true one, drawn with the very
noise the live branch draws with (keyed by POSITION). The hope: where the target's two distributions
agree at the draw's margin, the dead branch's draws ARE the live branch's, so at a miss the dead
spine hands over a target-written continuation for free. Prior art recycles the target's output
distributions (Token Recycling 2408.08696, Lookahead's n-gram pool 2402.02057) or its hidden states
at rejected positions as context for a TRAINED drafter (Carryover Drafting 2609.14717: +6.5-14.7%
accepted length; ReTrace 2608.29748; Lyanna 2602.21224); none had coupled draws. **Measured**
(`~/swarmllm-ref/spec/counterfactual.py` on `gumbeldump_c15-T07.npz`: 1,182 positions of sampled
coder-7B replies, T 0.7, the target continuing after the drafter's dead pick with the true noise):
one token's swap leaves the target's next token unchanged only **22.8%** of the time; at a miss of
the drafter's own spine (318 positions) the dead branch's draw is the true next token 15.4% of the
time, and with the drafter's re-rooted guess **either is right 72.3% against the drafter's 68.6%**,
the run collapsing past it (9.7% two deep). Once re-converged a dead run does follow (77%, then 90%
a token) — it is the re-convergence that is rare: one different word sends the target elsewhere. A
+3.7-point first child where the tree missed (~6% of tokens) is not worth a mechanism; not built.
The instrument was checked against the known answer: swapping a token for ITSELF gives back the true
continuation 197 of 197 times (`CF_CONTROL=1`).

**6.4 Is the requester's card worth more as drafter room than as a stage?** Holding a stage moves
neither the guesses nor the round trip (it only decides who covers those layers); a bigger drafter
guesses better. `DRAFTER_MAX_FRACTION` (sized for local speculation, where the drafter's time IS the
cost) refuses a 3B for a 7B. Where the requester's card cannot hold a bigger drafter, a peer on its
LAN could run it (3 ms away: drafting as a near service) — and would take the drafter's ~65% of
this node's busy time off its card, a third machine in the flow's pipeline (drafter → head → tail)
— but that time does not bind a flow (6.1's delay table: doubling it cost 0-4%); today's LAN peer has no card, and a processor drafter runs DSD chains (gotcha #941). **Measured for the 7B** (2026-10-23,
1,200 sampled positions each): top-1 agreement 0.646 / 0.755 / 0.774 for the 0.5B / 1.5B / 3B Q8_0;
best-first trees of 128, tokens a round — greedy 5.08 / 7.19 / 8.16, sampled (shared noise) 4.96 /
6.90 / 7.89. The 3B's +14% would reach the link as a few percent at most (6.1: speed does not follow
D) at a costlier call — not worth it for the 7B. For a large target the gap between a 1.5B and a 7B
drafter is wider; untested there.

**6.5 Place for the CYCLE, not the path.** Petals routes a chain by its path latency; a flow's P is
the cycle head → … → tail → whoever drafts, and the drafter must sit in it. When the requester is
far from every stage, the shortest cycle leaves it out (near chains generalise this: drafter and
head together, the requester a client). When stages can be placed, choose the assignment of layer
ranges to peers that minimises the cycle (a travelling-salesman order, not a shortest path).

**6.6 The far end as the guesser: the exact tail on an approximate head.** The only way past 6.2's
round trip short of a near chain: the far end approximates the head half's state for a token it
has just drawn (from the token's embedding and the EXACT states it already holds for the past),
runs its exact tail and draws — a guess of near-target quality made without this node — and this
node checks behind it. Prior art drafts with the target's FIRST layers (Kangaroo, LayerSkip, the
pipelined early exits of PPSD 2509.19368 and SPD 2605.30852) or a head on the TOP state (EAGLE);
none replaces the first half and keeps the exact second half (searched 2026-10-23), because none
had the second half on another machine. Against it, beyond early layers mattering most (layer-pruning
results) and a trained approximator per model: **it is dominated at both ends of the requester's
share.** Where this node holds FEW layers the approximation is easy — but then the far side holding
those few layers too is a near chain, exact and without this node's round trip; where it holds MANY
(and a five-way split leaves the tail a fifth) the approximation is poor. It can win only in the
middle, where this node's layers are needed for coverage and are few enough to mimic. **Parked**;
the first cheap test, if ever, is a ridge map fitted online on the states a flow already delivers,
on the 1.5B as a stand-in target.

**6.7 Every flow labels the drafter's mistakes — train it.** A flow returns the target's draw for
every row, the dead ones included: ~12K (context, target token) pairs a reply. A drafter LoRA
distilled on them (DistillSpec 2310.08461: +10-45% acceptance; Online SD 2310.07177) adapts to the
user's target AND domain on the user's own machine, no data leaving it. The crude form — a
per-token bias — was refuted (§ 5); a LoRA is not that. Needs a training loop on the card.

## 7. New levers, measured before any code (2026-10-24)

None of these was in the plans above. Each was measured offline on the coder-7B target and its
drafters before anything was built, and each says what the number decides. Data and scripts:
`~/swarmllm-318/{grammar,lens}/`; the probe dumps the state crossing a split
(`LOGITS_PROBE_HIDDEN`, `examples/logits_reference_probe.rs`).

**7.1 Decoding to the request's schema: a correctness gain, not a speed lever.** Today
`response_format` is an instruction in the prompt (`api/openai/types.rs`, `to_internal_messages`)
and a tool call is whatever the model writes, parsed afterwards. Constrained decoding
(llguidance 1.9.1: MIT, pure Rust; a mask costs ~50 µs a token at a 128K vocabulary by its
authors' figure; `rollback` lets one matcher walk a tree depth-first) makes the output valid by
construction. The hope for splits: a token the grammar FORCES (one legal token) needs no guess, so
a tree spends nothing on it.
- **Measured** (12 tool calls and a JSON reply rendered by Qwen 2.5's own template, the grammar
  walked over each reply's tokens; separators pinned to Qwen's `", "` / `": "` — llguidance's
  default is FLEXIBLE whitespace, which forces far less): **45% of the tokens inside the object
  are forced.** But the coder-1.5B drafter reads the tools in its prompt and already writes them:
  top-1 **0.994** on the forced tokens, **0.021 nats** a token — **3.3%** of its surprise over the
  object (361 tokens; free tokens 0.880 and 0.498). By E ≈ 1 + ln N / H (§ 6.1) that is worth
  ~3% at most; masking the free positions can add only what the drafter's 12% misses there hold.
  Consistent with the one published figure, TensorRT-LLM's +8% for masked drafts (2.65 → 2.86
  accepted on a chain of 3, an EAGLE-3 drafter, which reads no prompt); no published work
  combines a grammar with draft TREES (searched 2026-10-24).
- **What it is instead**: the only way to make `strict: true` true — OpenAI guarantees
  schema-valid output and we cannot — and to stop a small model's malformed call failing an
  agent: FUTURE_WORK #336. Two things the measurement found for that design: both coder models
  score Qwen's added `<tool_call>` token at ~e⁻²⁰ where a reply starts (the 7B begins with a plain
  `<` and spells the tag as text), so the trigger must match the TEXT as well as the token; and
  `"}}\n` is ONE token, so the grammar must own the closing text, not end at `}`.

**7.2 The requester's half of the target as a re-ranker — REFUTED (§ 5 item 2's first step,
run).** The requester computes the target's state at the split for every row it sends; if that
state could tell the drafter's candidates apart, a flow would gain E with no download and no
round trip. Measured on the coder-7B split at layer 14 of 28 (an even two-way split, the most a
requester holds), at 4,121 reply positions of 11 prompts, against the coder-1.5B Q8_0 drafter's
top 8 (drafter's first choice right 84.7%; the 8 hold the target's token 98.6%):
- the untrained lens (the target's own final norm and head on that state): **2.8%** top-1 over
  the vocabulary, **24.5%** among the drafter's 8 (chance 12.5%);
- a TRAINED translator (a rank-64 correction plus a bias, cross-entropy on the target's pick
  among the 8, four folds by prompt, scored on held-out prompts): **32.8%** among the 8;
- mixed with the drafter's log-prob at the weight best IN HINDSIGHT on the held-out positions:
  **0.849 against 0.847.** The stop rule (the head under 40%, the mix under the drafter + ~2.5)
  fired on every count.
- Instrument check: the same lens one layer from the top (split at 27) is right **83.8%** over
  the vocabulary, so the code is sound; the half-way state does not carry the next token in a
  form a linear head reads (LayerSkip's stock Llama3-8B at layer 16, § 5, agrees). A requester
  holding less than half holds less. Not built. The target state that does predict is the top
  layer's (EAGLE-3 / MTP heads), a round trip late in a flow (§ 5 item 2), and published heads
  cover Qwen 3 and Llama 3.x, not Qwen 2.5 (`split_speculation.md`, checked 2026-10-24).

**7.3 Accepting a near-tie: the simplest opt-in fast mode, measured.** Only a GREEDY reply
changes: a sampled one already accepts a near-tie in proportion to the target's own probability,
losslessly. Rule: keep the drafter's first choice where the target scores it within τ nats of
its own best. The same 4,121 positions, coder-1.5B:

| τ | chat: first choice kept | code | target log-prob given up (nats a token, chat / code) |
|---|---|---|---|
| 0 (lossless) | 0.755 | 0.919 | 0 |
| 0.25 | 0.783 | 0.933 | 0.003 / 0.002 |
| 0.5 | 0.803 | 0.943 | 0.010 / 0.005 |
| 1.0 | 0.845 | 0.961 | 0.042 / 0.019 |
| Medusa's typical acceptance (0.09, 0.3) | 0.919 | 0.982 | 0.158 / 0.050 |

By the link's measured slope (§ 5: first choice +4 points ≈ +15% tok/s), τ = 0.5 is worth about
+18% on chat and +9% on code for 0.005-0.010 nats a token; Medusa's rule ~+60% on chat at
sixteen times the loss. Two limits: these are per position on the target's own greedy path, so
what a run accepts AFTER a kept near-tie (the reply has left that path) is unmeasured; and nats
given up are not a quality score. It belongs to the planned opt-in fast mode
(`large_model_splits.md` item 6, Judge Decoding's rules: never the default, labelled as no longer
the model's own reply, scored against llama.cpp with `examples/score_against_reference.py`) as
its cheapest form: no judge to train. Next, only when that mode is built: replies under τ 0.5 and
1.0 scored against llama.cpp, then the link.

**7.4 Fewer bytes a row without losing anything: ~8% is all there is.** For a large model on a
home uplink the rows, not the round trip, set the tree's width: a 70B row is 8,192 values, ~8.7 KB
at Q8_0, so 64 guesses a round at ~6.5 rounds a second is ~29 Mbit/s a hop, past most home uploads,
and the flow's guard (`FlowGuard`) holds the tree down to fit. BloomBee (2604.21072) codes
activations losslessly; the question is what is left after OUR wire, which already sends Q8_0
through zstd level 1 (`network/protocol/mod.rs`, `compress_tensors`). Measured on the state crossing
the coder-7B's split at layer 14 (`LOGITS_PROBE_HIDDEN`), the wire's own Q8_0 reproduced
(`~/swarmllm-318/osd/rowbits.py`), on a chat reply (443 rows) and a long code prompt (1,913):
- **today: 7.73-7.74 bits a value** (raw Q8_0 is 8.5), the same at 8, 64 and 512 rows a message;
- the best ideal STATIC coder — a table per dimension for the quants, the scales coded too: **7.14
  on the 1,913 rows** (the 443-row figure, 6.86, is a plug-in entropy estimated from too few
  samples a dimension), **7.6% under today's wire**. A real coder does a little worse, so worth a
  few per cent of tok/s where bandwidth binds, and a new wire format. Not built.
- **Scaling each dimension by its typical size before the group quantization — REFUTED**
  (`rowlane.py`, calibrated on half the rows, scored on the other half): worse at every width, more
  error AND more bits (8 bits: 1.11% against Q8_0's 0.86% a row). A few dimensions run ~170× the
  median, and they carry the row's norm. Plain group quantization keeps its usual price, about
  twice the error a bit removed (7 bits, 1.73%), so fewer bytes a row means more loss: the opt-in
  mode's question, not the default's.

**7.5 A drafter that learns from the target as a session goes on (OSD 2310.07177, OmniDraft
2507.02659): its cheapest form REFUTED, the adapter forms still to run.** Every check hands the
requester the target's tokens, so the drafter could be trained on them for free, lossless by
construction (only the guesser changes). Measured offline (`~/swarmllm-318/osd/osd.py`, PyTorch on
the processor): the coder-1.5B drafter (bf16 weights, HF), replaying the 11 prompts as two
sessions in order (code: 6, chat: 5), each 32-token piece SCORED before it is learned from. The
frozen drafter reproduces the candle runs exactly (code_bst 0.965 both). **A learned bias per
vocabulary entry** (AdamW, lr 0.05, one step a piece, the previous conversation replayed): code
session 0.920 → 0.915 (−0.4 points; csv −3.1, code_debounce +1.9), chat −0.9 to −1.2 a
conversation. A word-level nudge cannot learn misses that depend on the context. Owed: a LoRA on
the top two layers (no backward past them, cheap to build in candle) and on every layer, the same
protocol; the stop rule is +2 points on a session.

**7.6 Copying from the user's EARLIER sessions, not only this one: worth it for agents, at a
session's start.** A copied run (`FlowRun::copy_ahead` → `ngram_lookup::copied_run`) looks up the
reply's last 4, 3 or 2 tokens in the prompt, then in the reply's last 500 tokens. SuffixDecoding
(2411.04975) adds a GLOBAL store, every earlier reply, and reports up to 5.3× on agent work.
Measured offline with today's lookup reproduced (`~/swarmllm-318/suffix/sim.py`: the most recent
match, run max(matched, 10) ≤ 32 tokens, a walk taking the run's matching prefix), the store tried
only where the prompt and the reply found nothing. Share of reply tokens a copy predicted:
- **400 SWE-agent sessions** (public, `nebius/SWE-agent-trajectories`, Llama-70B's replies, 1.17M
  reply tokens): **64.6% → 73.9%**; a session's first 3 replies **28.1% → 61.2%**.
- **30 coding-agent sessions of the `/v1/messages` kind** (this project's own Claude Code
  transcripts, only the counts read; 6.6M reply tokens, 95% of them tool-call inputs): **55.3% →
  57.9%** (text 17.8 → 20.9, tool calls 57.2 → 59.8); a session's first 10 replies **33.6% → 63.6%**.

So the store matters where the session's own history is short — an agent's opening calls, a new
chat on a familiar task — and adds a few points once the history has built up. Both workloads are
another model's text: this measures the repetition in agent work, not a drafter, and chat is
unmeasured (SuffixDecoding's WildChat gains are smaller). **A store is a side channel**: REST's
datastore leaks at over 25 tokens a second through the speed of the reply alone (2411.01076), so a
store holds only ONE requester's replies and is read only for that requester — never another API
key's, never for a peer's request (a head leading someone's split, a pool member's forward).
Filed as FUTURE_WORK #337.

## Sources

Petals 2312.08361 · Pope et al. 2211.05102 · StagFormer 2501.15665 · Layer
Parallelism 2502.02790 · Lookahead 2402.02057 · CLLM 2403.00835 · Lamina
2405.01814 · MegaScale-Infer 2504.02263 · BloomBee 2604.21072 · Prompt Cache
2311.04934 · Preble 2407.00023 · SkyWalker 2505.24095 · CacheGen 2310.07240 ·
LLMCompiler 2312.04511 · Skeleton-of-Thought 2307.15337 · SpecExec 2406.02532 ·
Sequoia 2402.12374 · ML-SpecQD 2503.13565 · QuantSpec 2502.10424 · SPEQ
2510.18525 · Daliri et al. 2408.07978 · multistream-select 0.13.0 `Version`
docs · llama.cpp `--n-cpu-moe` reports (discussions #15396, dev.to 8 GB
Qwen3-Coder-30B) · Leviathan 2211.17192 · Chen 2302.01318 · § 7: llguidance 1.9.1
(`docs/fast_forward.md`, `syntax.md`) · TensorRT-LLM tech blog 12 (guided + speculative
decoding) · vLLM `v1/structured_output` · Medusa 2401.10774 (typical acceptance) · Tuned Lens
2303.08112 · LayerSkip 2404.16710 · BloomBee 2604.21072 (lossless activation coding) · OSD
2310.07177 · OmniDraft 2507.02659 · SuffixDecoding 2411.04975 · speculation side channels
2411.01076.
