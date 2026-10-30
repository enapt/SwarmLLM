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
per-token bias — was refuted (§ 5), and so, within a session, was a LoRA on the top two layers
(§ 7.5: −0.7 / −1.3 points). Left untried: distilling across MANY sessions, DistillSpec's setting,
not one session's few thousand tokens.

## 7. New levers, measured before any code (2026-10-24)

None of these was in the plans above. Each was measured offline on the coder-7B target and its
drafters before anything was built, and each says what the number decides. Data and scripts:
`~/swarmllm-318/{grammar,lens}/`; the probe dumps the state crossing a split
(`LOGITS_PROBE_HIDDEN`, `examples/logits_reference_probe.rs`).

**7.1 Decoding to the request's schema: a correctness gain, not a speed lever.** Until #336
(2026-10-30) `response_format` was only an instruction in the prompt (`api/openai/types.rs`,
`to_internal_messages`); it is now held token by token where this node runs the model whole
(`inference::reply_grammar`) and still only asked for on a split, and a tool call is whatever the
model writes, parsed afterwards. Constrained decoding
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
2507.02659): REFUTED in both forms tried.** Every check hands the
requester the target's tokens, so the drafter could be trained on them for free, lossless by
construction (only the guesser changes). Measured offline (`~/swarmllm-318/osd/osd.py`, PyTorch on
the processor): the coder-1.5B drafter (bf16 weights, HF), replaying the 11 prompts as two
sessions in order (code: 6, chat: 5), each 32-token piece SCORED before it is learned from. The
frozen drafter reproduces the candle runs exactly (code_bst 0.965 both). **A learned bias per
vocabulary entry** (AdamW, lr 0.05, one step a piece, the previous conversation replayed): code
session 0.920 → 0.915 (−0.4 points; csv −3.1, code_debounce +1.9), chat −0.9 to −1.2 a
conversation. A word-level nudge cannot learn misses that depend on the context. **A LoRA on the
top two layers** (rank 16, 1.32M weights, lr 2e-4, one step a piece, replay 1; no backward past
them, the form cheap to build in candle): code session 0.920 → 0.913 (−0.7 points; code_debounce
+1.6, code_rust +1.5, code_sql −3.0, csv −3.4), chat 0.754 → 0.741 (−1.3; fridge −4.8). No session
reached the stop rule's +2 points, so the every-layer form was not run. At these settings a
session's tokens hurt the guesser about as often as they help it; the drafter gains from the
session by COPYING it (§ 7.6).

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

Several continuations as branches instead of the most recent one (`branches.py`, 200 of the
SWE-agent sessions, 10 tokens a branch): 63.9% at one, 67.1% at two (+34% copied rows), 68.4% at
four (+70%). The store costs rows too, since each copy it makes is a proposal the session found
nothing for: 3.29 → 4.17 copied rows a reply token (+27%, `vote.py`) for its 9.3 points, three
times what a second branch buys for fewer rows. Three more rules on the 400 sessions:
- **The LONGEST match, not the most recent** (LZ77's rule: of the last 64 hits, the one whose context
  agrees furthest back; `longest.py`): 64.6 → 65.7% local, 73.9 → 75.1% with the store, 7% fewer
  copies, still 62-63% wrong at the first token. Small, and it is what a hash-chain index of the store
  gives for nothing.
- **A floor on the match before copying** (`minmatch.py`; today 2 tokens): coverage 64.6 / 60.1 /
  56.5 / 50.1% at 2 / 3 / 4 / 6, copied rows a reply token 3.29 / 2.40 / 2.01 / 1.58. A 3-token floor
  sends 27% fewer copied rows for 4.5 points. That trade is for `FlowGuard` to make where rows are
  short, not a fixed default.
- **The store copies only where its earlier sessions AGREE** (`vote.py`). llama.cpp's lookup decoding
  keeps a cross-run n-gram cache on disk (`--lookup-cache-dynamic`) and drafts from it only under
  STRICT thresholds — the n-gram seen ≥ 4/3/2/2 times (n = 1..4), its leading next token ≥ 75/66/66/66%
  of them — against LAX ones in the current context (≥ 2/2/1/1, ≥ 66/50/50/50%;
  `common/ngram-cache.cpp`, read 2026-10-24). The same vote over the store's last 64 hits, local copies
  unchanged:

  | store gate | coverage | copied rows a reply token | the store's rows that copied | first 3 replies |
  |---|---|---|---|---|
  | none (today's rule) | 73.9% | 4.17 | 10% | 61.2% |
  | lax | 73.7% | 3.90 | 13% | 60.8% |
  | strict | 71.6% | 3.59 | 17% | 54.8% |
  | stricter (≥ 3/4/6, 75%) | 70.3% | 3.47 | 20% | 51.2% |

  The lax gate is nearly free: −0.2 points, the store's own rows 1.16 → 0.88 a reply token (−24%).
  Strict takes them to 0.52 (−41% more) for 2.1 points more, much of it at a session's start, where
  the store matters most. #337 takes the lax gate;
  past it the trade is `FlowGuard`'s, like the floor above.
- **A run ends where the earlier matches stop agreeing** (`cut.py`, local sources, today's lookup
  otherwise). SuffixDecoding scores a continuation by how often it followed the match and stops a draft
  where that frequency falls; today's run is the most recent hit's continuation, max(matched, 10) ≤ 32,
  whatever the other hits did. Walking the last 32 hits together and stopping once the leading next
  token holds less than a share P of those still following:

  | rule | coverage | copied rows a reply token | rows a covered token |
  |---|---|---|---|
  | today | 64.6% | 3.29 | 5.10 |
  | P = 1/3 | 64.8% | 3.63 | 5.60 |
  | P = 1/2 | 64.8% | 3.47 | 5.36 |
  | P = 2/3 | 64.7% | 2.62 | 4.04 |
  | P = 4/5 | 64.7% | 2.55 | 3.94 |

  At 2/3, **20% fewer copied rows for no coverage**, where the 3-token floor bought 27% for 4.5
  points. A low P lengthens runs: hits that agree carry a run past today's 10-token rule. One hit
  alone is cut by the length rule, as today. Rows matter only where the link's bandwidth binds; a wrong
  copy measured free on the rig (`docs/invariants/scheduling.md` § "Copied runs"). Filed as #338.

  **Capped as the flow caps a run** (2026-10-27, the agreed run held to max(matched, 10) ≤ 32 too,
  as `FlowRun::copy_ahead`'s `lead` holds it): today 64.6% / 3.29, P = 0.67 **64.6% / 2.36 (−28%)**,
  P = 0.8 64.6% / 2.33. Shipped at P = 0.67 in `ngram_lookup::copied_run` (#338, closed): the Rust
  function, replayed over the first 100 sessions, gives the script's own figures for both rules.
- **The store as built** (`combo.py`, 2026-10-24): each earlier reply its own (no match or run
  across a reply's end), bounded to the newest N tokens, the lax gate over the last K hits:

  | store | coverage | rows a reply token | from the store | first 3 replies |
  |---|---|---|---|---|
  | unbounded, K 64, lax (reproduces the table above) | 73.7% | 3.86 | 11.7 pt for 0.85 rows | 60.7% |
  | 256K, K 64, lax | 73.3% | 3.83 | 11.2 pt for 0.81 rows | 59.9% |
  | **256K, K 32, lax** | 73.3% | 3.84 | 11.3 pt for 0.82 rows | 59.9% |
  | 256K, K 32, lax + agreement cut 2/3 | 72.9% | 3.86 | 10.8 pt for 0.82 rows | 58.5% |
  | 128K, K 32, lax + cut | 72.6% | 3.82 | 10.5 pt for 0.77 rows | 58.1% |

  256K tokens (a fifth of the 1.18M reply tokens here) keeps 96% of the gain. The agreement cut
  saves no rows on the store: many earlier replies agreeing carry a run to 32, past the length
  rule. #337 is built at 256K, K 32, lax, today's run.
- **Unmeasured next source: a PUBLIC store.** REST (2311.08252) drafts from a datastore of public
  text alone: CodeLlama 7B on HumanEval took 2.12-2.36x with 2.7M Python files of The Stack
  (27 GB), and RASD's (2503.03434) table has 2.76 at 0.9 GB against 2.90 at 27 GB, so most of the
  gain is in a small store. CREST (2408.04678) keeps only each size's most frequent n-grams, each
  with its continuations precomputed as a tree, and matches REST at a tenth of the storage
  (acceptance length 2.37 against 2.17 on HumanEval, 1.9 against 1.77 on MT-Bench chat, as
  measured without any prompt lookup). A public store has none of the owner store's limits: no
  owner check, no peer rule, and it could be shared as parts like a model. What it would add here
  is the question. Its gains were measured with no prompt lookup and no drafter, and on chat this
  flow's copies are already nearly all wrong (1-6 walked of 292-369 rows). Measure before
  building: tokenize a public code corpus's most frequent 2-5-grams with Qwen's tokenizer, then
  add them to `combo.py` as a fourth source and count the coverage they add beyond the owner
  store on the SWE-agent sessions.

So the store matters where the session's own history is short — an agent's opening calls, a new
chat on a familiar task — and adds a few points once the history has built up. Both workloads are
another model's text: this measures the repetition in agent work, not a drafter, and chat is
unmeasured (SuffixDecoding's WildChat gains are smaller). **A store is a side channel**: REST's
datastore leaks at over 25 tokens a second through the speed of the reply alone (2411.01076), so a
store holds only the owner's replies and is read only for the owner's requests — never for a peer's
request (a head leading someone's split, a pool member's forward under the same key).
vLLM's suffix method builds ONE `SuffixDecodingCache` (Arctic Inference) for the whole server,
fed every request's output and keyed by request id. Its proposer shows no user or tenant key
(`vllm/v1/spec_decode/suffix_decoding.py`, read 2026-10-24; the cache's own matching is in Arctic
Inference, not read). Here the store is the owner's alone.
Filed as FUTURE_WORK #337.

## 8. Deep dive after the first 70B replies (2026-10-26)

The 70B answers at 1.29-1.47 tok/s through this PC (layers 0-21) and AU (`aa3d1e66`, layers
21-80 on an RTX 3060 and its processor). This section asks where that time goes and which
untried ideas could move it, measured before anything is built.

**8.1 A check's width on a processor — measured; llama.cpp is no faster.** `prefill_bench`'s
`SPEC WIDTH SWEEP` (Ryzen 7 5800H, 8 threads, min of pairs) against `llama-bench -p N` on the same
file (`~/swarmllm-bench/width1026/run2.out`; the ratio is the bench's, each width timed in pairs
with a 1-row forward):

| rows | our 8B Q4_K_M | × 1 row (paired) | our 70B, 8 layers | × 1 row (paired) | llama.cpp 8B (pp tok/s → ms) |
|---|---|---|---|---|---|
| 1 | 125.6 ms | 1.02 | 104.4 ms | 1.04 | 7.20 → 139 ms |
| 2 | 139.3 | 0.98 | 98.6 | 0.98 | 10.82 → 185 |
| 3 | 137.5 | 1.12 | 105.8 | 1.08 | 14.15 → 212 |
| 4 | 145.0 | 1.15 | 109.6 | 1.12 | 19.75 → 203 |
| 6 | 229.5 | 1.73 | 154.9 | 1.53 | 20.92 → 287 |
| 8 | 260.2 | 2.04 | 204.9 | 2.09 | 22.40 → 357 |
| 16 | 542.7 | 4.18 | 397.5 | 3.98 | 28.21 → 567 |
| 32 | 1096.6 | 8.24 | 802.1 | 7.66 | 28.39 → 1127 |
| 64 | 2244.4 | 16.73 | 1770.6 | 17.42 | 29.97 → 2135 |

- **Up to 4 rows is nearly free; past it each row costs ~0.26 of a step** — the knee where an
  AVX2 dot product (no VNNI on Zen 3) stops being bound by reading the weights and becomes bound
  by the arithmetic. Our kernels match or beat llama.cpp's at every width: the vendored tiled
  matmul already reads each weight column once for all rows (`k_quants.rs`, `vec_dot_rows_chosen`),
  so "each guess re-reads the weights" is refuted by the code and by the curve.
- **So a processor far end's chunk should fill the knee and stop there.** The stream's chunk is
  3 rows (2 drafts and a look-ahead, `dsd_stream::guesses_per_chunk`); a 4th row costs ~5%.
  A tree on a processor does not pay past ~4 rows. The old figure of "22 ms a position" came from
  the 7B TH↔IT processor peer, past the knee.

**8.2 Checks computed for nothing at the far end — built (the tail skip).** In reply 344f21c8 AU
answered a check every ~1.14 s and was never idle (three chunks out, a ~0.43 s round trip and
~0.5 s for this PC to draft and run its 21 layers). Of the 29 decode checks it computed, 10 stood
on a guess an earlier check had already refused: the restart reached AU ~1.0 s after the refusal
left it (0.57 s of that at this PC), so AU had started the next stale chunk the moment it
answered. The stream now skips them at the tail: a check walked there that refused a guess marks
the stream, and every later turn that is not a restart is answered as skipped
(`forward_streams` `StoodOnARefusal`, `docs/invariants/network.md` § "A stream of verifies").
The far end then idles until the restart arrives instead of finishing a stale check, so the
requester gains the far end's check less the restart's latency — ~0.14 s a miss here, ~3% of the
reply — and the far end's time is free for anyone else's work. It grows where the far end is slow
against the round trip. A wrong look-ahead (a quarter of the restarts) is not visible at the tail.

**8.3 A wide check through a card + processor far end: the card copy floors it, and trees still
lose here.** A forward of ≥ 32 rows copies each processor-held weight to the card for the call
(#138's op offload), so its cost should stop growing with the rows. Measured with `prefill_bench`'s
new `SWARM_BENCH_GPU_LAYERS` (the 70B's layers 0-21, 6 on the card and 15 on the processor, as a
worker given `--gpu-layers 6` places them; `~/swarmllm-bench/width1026/run4.out`):

| rows | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| ms a check | 216 | 258 | 488 | 929 | 1283 | 1281 | 1285 | 1301 |
| × 1 row (paired) | 1.00 | 1.22 | 2.15 | 3.65 | 6.03 | 5.62 | 6.00 | 6.01 |

From 32 rows on, the check costs the copy — 15 layers × ~0.52 GB at ~7 GB/s, ~1.1 s — whatever the
rows; below 32 the processor's curve (§ 8.1) reaches the same ~1.28 s at 24 rows, so the crossover
is ~24 rows, as `(copy − step) / slope + 1` predicted. (The arm with the copy switched off,
`SWARMLLM_PROMPT_CARD_LAYERS=0 SWARMLLM_PROMPT_CARD_PROJECTIONS=0`, read noisily — 1.5 s at 16 rows
against 0.93 with it on, where the two paths are the same — so the processor reference is § 8.1's
curve.) **What it decides**: a 64-row tree at 1.3 s against a 4-row chain at 0.26 s needs ~5× the
chain's ~2.8 tokens a check (~14) to break even on compute alone; this project's own trees kept
3.2-4.6 guesses a round (`docs/invariants/scheduling.md` § "A split checks a tree of guesses"). Not
built. The floor is the card's link: a desktop
copying pinned memory at ~25 GB/s would put it near 0.35 s, where a tree of tens of rows could beat
the chain — so the lever exists only where a far end measures its own copy rate and says so.

**8.4 Ideas checked and not built.**
- **Activation sparsity (TEAL, CATS)**: skips the weight columns a token's small activations do
  not need — lossy, and a check's rows need the UNION of their masks, which at 4 rows leaves little
  to skip.
- **Look-up-table kernels (T-MAC, bitnet.cpp)**: pay at 1-3 bits; our models are Q4_K.
- **Hybrid cores**: decode threads are already calibrated per machine (`cpu_pools::calibration_for`).
- **Tapering a tree along the chain** (a middle stage forwarding only the likeliest rows): exact,
  but the tokens a round keeps cannot exceed what the narrowest stage passed on.
- **Walking a tree level by level at the tail** (a cheap level, then the next): each level is a
  round of the tail's own work, worse than one batched check.
- **Filling the tail's idle time with alternative rows**: under the stream the tail is not idle.
- **Drafting the restart before the refusal** (SSD / Saguaro, 2603.03251: guess the far end's
  correction from the drafter's next-likeliest tokens and pre-draft each): saves the drafting part
  of a restart's 0.57 s, ~3% here.
- **A larger drafter for the 70B** (8B against the 3B): ~+9% acceptance by the 70B-family
  tables, and it does not fit beside this PC's layers on an 8 GB card.

**8.5 What is left, ranked by what it would give the 70B.**
1. **The far end's speed.** AU's ~1.14 s for 59 layers is ~22 GB/s over ~21 GB on its
   processor; the cause cannot be measured from here. Placement by speed (#356) is the lever the
   swarm has. **Researched 2026-10-30 — it is not half of what such a machine reads**: AU is an
   i7-12700 (8 P + 4 E cores); the best published llama.cpp processor decode on hybrid Intel is
   ~27 GB/s (a 12700H, 65B Q4_0, P-cores only — with every core 0.25 against 0.74 tok/s, ggml
   splitting work evenly so the P-cores wait on the E-cores; discussion #572, issue #842), no
   12700-class figure reaches its DDR peak, and AVX-VNNI does not help: ggml's Q4_K × Q8_K dot
   is `maddubs`/`madd` (`_mm256_dpbusd_avx_epi32` only in the Q4_1/Q5_1 helper). The swarm's
   fast processor for the 70B is the M4 Mac `99aafc41`, which holds it and states ~14 tok/s in
   7B units (~57 GB/s, ~2.6x AU's processor; llama.cpp on an M4's GPU reads 86-91 GB/s,
   discussion #4167) — at ~430 ms from here, so it pays only as a stage the stream keeps busy,
   and the planner prices that, not a rule.
2. **The chunk's width and the drafter's floor in it** — built, see 8.6.
3. **The restart's latency at the head** (0.57 s: drafting ~190 ms with the 3B on the processor at
   62 ms a guess, then 21 layers). A drafter on the card would cut the drafting to ~30 ms, but on an
   8 GB card it costs ~4 of the 70B's layers there.
4. **A public n-gram store** (§ 7.6, the user's question of 2026-10-26: why not look common phrases
   up instead of guessing them) — unmeasured. Checking a guess, not making it, is the cost here, and
   the drafter already reads the whole context at ~0.77 acceptance; a looked-up phrase saves only
   where it is right more often than the drafter, which the measurement in § 7.6 is to establish.

**8.6 The stream's chunk: guessed whole, and every chunk one width — built** (one of TWO widths since
2026-10-30, chosen by what each pays: #374, `docs/invariants/scheduling.md` § "A chunk is as wide as it
pays"). The chain stream
(what a split runs where trees do not: a drafter off the card, a far end partly on a processor —
the 70B) was swept on the real link (this PC's card holding the 7B's layers 0-13 and the
coder-1.5B Q8_0 drafter, AU's RTX 3060 the rest, trees and flow forced off,
`~/swarmllm-318/streamsweep.sh`, decode tok/s, arms alternating, one binary):

| arm | 400 tokens | 160 tokens |
|---|---|---|
| 3 guesses a chunk, floor 0.4 (as shipped) | 6.48 6.45 · 6.54 6.27 | 5.10 6.00 · 6.12 5.71 |
| 4 guesses, floor 0.4 | 6.57 6.57 | 5.66 5.66 |
| 5 guesses, floor 0.4 | 6.54 6.66 6.61 | 5.71 5.79 5.75 |
| 4 guesses, floor 0.67 | 4.80 4.77 · 4.81 4.73 | 4.39 4.26 · 4.53 4.39 |
| 3 guesses, NO floor | 7.52 7.61 · 7.48 7.33 | 6.80 · 6.42 (+2 stalled) |

- **The width barely matters; the drafter's confidence floor decides.** 4-5 guesses gave 1-3% over 3.
  Raising the floor to 0.67 cost 27%, and removing it gained 16%. The floor was put there for the
  ROUNDS (§ "An unsure guess ends the round" in `split_speculation.md`: there it saves the far computer
  ~25% of its positions at no speed cost). In the stream it does something else. A chunk the drafter
  cut short HOLDS the stream: nothing is built past it until its answer is back
  (`Frontier::held_by`, PipeInfer's "reactive speculation"). On a link whose round trip (~440 ms)
  outlasts the far end's check (~30-50 ms for 14 layers on a card), each hold idles the far end for a
  whole round trip. Building past the unsure guess loses only when that guess is wrong, and then only
  the work the restart throws away, which that idle far end had for free. The stream now drafts with no
  floor (`engine_drafter::stream_confidence_floor`; the rounds keep 0.4). `held` on the stream's line
  counts the holds left.
- **With no floor every chunk is the same width, and a narrow one late in a worker's life
  stalled.** Four of the 16 no-floor replies took 10-59 s longer. Each time it was ONE forward on this
  PC's card: the worker's first CUDA graph of a 2-row chunk (`decode graph … instantiated` +1,
  `forward_ms=56678`), the reply's last chunk cut to the room left. With the floor, cut chunks of 1
  and 2 rows came early and their graphs were built then. This is the WSL2 slowdown of fresh card
  memory with host uptime (gotcha #883: a pool growth 0.3-5.7 s; here 84 h up). The stream now sends
  every chunk at full width, the last one too, with the guesses past the reply's end checked and
  dropped (`chunk_guesses`, bounded by the context the chain serves). The far end's card is spared the
  narrow shape as well. A narrow shape can still first appear late on other paths: FUTURE_WORK #357.
- **Verified** (the same link, one binary, arms alternating, 2026-10-26 17:00Z, Windows up 84.6 h,
  `SWARMLLM_STREAM_DRAFT_CONFIDENCE=0.4,SWARMLLM_STREAM_FULL_CHUNKS=0` restoring the old stream):

  | arm | 400 tokens | 160 tokens | `held` (160 / 400) |
  |---|---|---|---|
  | new (no floor, full chunks) | 7.81 7.80 · 7.62 7.42 (mean 7.66) | 6.87 6.82 · 6.54 6.44 (6.67) | 0 / 0 |
  | old (floor 0.4, narrow) | 6.57 6.59 · 6.46 6.34 (6.49) | 6.11 5.82 · 6.16 5.77 (5.97) | 37 / 82 |
  | no floor, narrow | 7.86 7.79 · 7.67 7.85 (7.79) | 6.81 6.72 · 6.79 6.44 (6.69) | 0 / 0 |

  +18% at 400 tokens and +12% at 160. The replies are the same in every arm. The full-width tail
  fired as meant: each worker built ONE card graph in the new arm and 2-3 with narrow tails. No arm
  stalled in this run, so the stall is intermittent (4 of 16 replies in the sweep, 0 of 8 here); the
  full chunks remove its trigger at no cost in speed.
- **70B: level, one prompt 10% slower — against a far end WITHOUT the tail skip.** This node's
  first layers + AU the rest (pinned with `swarm_route`; unpinned, the planner chose the Mac + AU
  in half the arms and nothing here streamed), arms new / old / old / new, 96-token replies to
  three new prompts, decode tok/s:

  | prompt | new | old | `held` (old) |
  |---|---|---|---|
  | compost | 1.94 | 1.93 · 1.93 | 2 of 35 |
  | tea | 1.34 | 1.35 · 1.31 | 9 of 43 |
  | lighthouse | 0.91 | 1.00 · 1.03 | 24 of 49 |

  The second new arm is left out: its first ask was refused (#358) and the next one loaded the
  model. AU runs v0.3.256, which predates the tail skip (§ 8.2), and its check (~1.14 s)
  outlasts the round trip, so its queue is full: a chunk built on a wrong unsure guess is
  computed there in full, ahead of the restart. The hold had kept the restart from queuing behind
  it, which pays on the prompt with the most holds. The tail skip answers such a chunk the
  moment the refusal is walked; it reaches AU with this release, and the 70B is measured again
  then. If the floor still pays where the far end's check outlasts the round trip, it comes back
  for that case only (FUTURE_WORK #149).

  **Measured again with AU on v0.3.257** (2026-10-27, the same rig and prompts, pinned to this node
  + AU with every other peer excluded, arms new · old · old · new, Windows up 90.6-91.4 h, decode
  tok/s at 96 tokens; `~/swarmllm-70b/ab70after.out`):

  | prompt | new (no floor) | old (floor 0.4) | old's held chunks |
  |---|---|---|---|
  | lighthouse | 1.09 · 1.13 | 0.99 · 1.05 | 28 of 52, 27 of 52 |
  | compost | 1.90 | 1.75 · 2.01 | 7 of 35, 7 of 35 |
  | tea | 1.47 | 1.46 · 1.54 | 9 of 40, 9 of 40 |

  With the tail skip at the far end, the prompt with the most holds is now 9% FASTER without the
  floor (both new arms above both old ones), and the other two are inside the arms' own spread;
  `held` read 0 in every new arm. The floor does not come back. The last new arm lost compost and
  tea: AU disconnected at 02:05:05Z, and with Proxmox, the other holder of layers 28-79, excluded
  by the pin, both asks failed in under 5 s.

## 9. Measured on the real link, 2026-10-29

**A split across a continent ran a 14B twice as fast as this machine runs it alone, on code.**
Qwen2.5-14B Q4_K_M; this machine (RTX 3070 Laptop 8 GB, Thailand) against a swarm peer's RTX 3060
(Australia, ~420 ms round trip, v0.3.259); the 0.5B Q8_0 drafter on this machine's card where
named; 160 greedy tokens, a prose prompt then a code prompt per fresh node; every run ABBA inside
one binary (main at `4130e0a1`), the route pinned per ask and checked in the log
(`~/swarmllm-gate/deep/fr/real_link.sh`; round log `memory/archive/round_log_1029_real_link.md`).

| shape, guesser, this machine's share | code tok/s | prose tok/s |
|---|---|---|
| starts and finishes here, no guessing | 2.03-2.05 | 2.03-2.05 |
| starts and finishes here, far-run guesses (#366 (4)) | 3.74-4.01 | 2.93-3.61 |
| starts and finishes here, drafter (tree rounds), 25 of 48 | 12.38-16.34 | 5.37-7.10 |
| starts and finishes here, drafter (the flow, #369), 25 of 48 | 17.31-20.06 | 7.44-8.28 |
| default shape, drafter (the flow), 12 of 48 | 7.79-8.36 | 3.15-3.38 |
| **default shape, drafter (the flow), 24 of 48** | **29.20-29.94** | **9.17-10.02** |
| this machine alone, the whole 14B (card + processor, card guesses) | 13.6-15.4 | ~13 |

What each figure is made of, from the runs' own lines:
- **The flow at an even split** sent 23 rows a forward; on prose 305-315 forwards, 12-13 misses,
  no stale answers, 4.2-5.0 s waiting of 17.5-18.8 s; on code 139-144 forwards, 0-1 misses, 22.8-23.4
  tok/s inside the flow. The round trip is hidden: the far card answers within ~450 ms of each
  forward and the head keeps drafting meanwhile.
- **The tree rounds** (a chain that starts and finishes here — the flow refuses it,
  `flow_possible` needs the tail to walk at the far end) keep 2.4-2.8 tokens a round on prose and
  8.4-8.9 on code, and every round waits one whole round trip. Same link, same drafter, same split:
  the flow is 1.6x (prose) to 2.2x (code) as fast → FUTURE_WORK #369.
- **The flow for a chain that starts and finishes here** (#369, later the same day, one binary,
  arms alternating): this node's last layers draw every row, run on each answer in the order its
  forward was sent. Tree rounds 5.74 / 6.99 prose, 15.03 / 15.52 code; the flow 7.44 / 7.80 and
  17.31 / 18.88 (~1.2x); the default shape in the same run 10.18 / 8.53 and 22.33 / 21.90 (its code
  below the morning's 29.2-29.9 — the link's, not the build's: both shapes ran in one binary).
  A second run (one binary, arms alternating): the flow finishing here 7.52 / 8.28 and 18.49 /
  20.06, the default shape 8.05 / 7.74 and 19.41 / 27.88. Over both runs the default shape is
  ~1.1x ahead on prose and ~1.2x on code, inside its own code spread (19.4-29.9). Misses are alike:
  18-20 a prose reply here against 9-17, 2-4 on code against 0-2, and at most 5 of them a reply
  are the walk outrunning the tree in either shape (`misses_at_a_leaf`) — the rest are wrong
  guesses. The last layers here cost ~21 ms a forward, run serially with the drafter on one card.
- **The flow at 12 of 48** saturated the far card: 36 layers plus the 152k-vocabulary output head
  for every row of every forward, up to 2.2 s queued, 18-39 ms a row. The flow's speed is the
  slower side's; the split must follow the machines' speeds (#356).
- **The far-run guesses** (no drafter) are a chain of at most 4 at ~18 ms a guess, kept ~1 in 4 on
  prose and ~1 in 3 on code. A drafter's draft step is 3.4 ms and its trees are 20-46 rows. The
  guesses are for the models no drafter fits.

What it means for the aim. On a far link with a card at each end, the per-token round trip
is no longer the limit once a good drafter's tree flows. Code ran past local, because a code
reply is easy to guess. Prose is held by how often the drafter's guesses land (§ 7, § 8: the flow
is held by the drafter's first-choice rate), so the next prose gains are better guesses, not
shorter trips. The open question for users is whether they GET this shape: a head with the even
share, a drafter on its card, and, when the head holds both ends, the flow (since #369, within
~1.1-1.2x of the default shape's).

## 10. The head's own work binds a split that starts and finishes here (2026-10-30)

**Where a reply's time goes**, read off the 10-29 rig logs (`DSD: a tree that flowed is complete`;
Qwen2.5-14B, 0.5B Q8_0 drafter, TH↔AU):

| shape, reply | total | waiting | drafting | this node's first layers | its last layers | forwards, misses |
|---|---|---|---|---|---|---|
| starts and finishes here, prose | 22.6 s | 8.4 s | 4.3 s | 4.0 s | 6.0 s | 285, 20 |
| starts and finishes here, code | 9.2 s | 3.4 s | 1.7 s | 1.7 s | 2.4 s | —, 4 |
| default shape, prose | 17.0 s | 3.3 s | 5.2 s | 8.5 s | — | —, 9 |
| default shape, code | 7.1 s | 0.8 s | 2.2 s | 4.1 s | — | —, — |

Waiting is the misses: 20 × ~420 ms ≈ the 8.4 s. **The rest — 63% of the prose reply — is this
node's card**, serial on one stream: ~15 ms a drafter call, ~14 ms its first layers, ~21 ms its last
layers a forward (~26 ms at 56 rows). 285 forwards carried ~6,500 rows for 160 tokens: 2.4% of the
rows reached the reply, and every forward paid its fixed cost. So the lever is the COUNT of
forwards on this node, not their rows, and not the link.

**What follows, each with its test:**

1. **One forward per burst of answers** (FUTURE_WORK #369, built behind `SWARMLLM_FLOW_TAIL_JOIN`):
   the median tail forward has nothing waiting behind it, but answers land in bursts — 20-49% of
   tail forwards on prose and 31-59% on code had ≥ 1 behind them, up to 18 — so one forward per
   burst is 16-52% fewer tails, a first-order ceiling of 4-11% of a prose reply and 7-15% of a
   code reply (`~/swarmllm-gate/deep/fr/due_per_tail.py`). The joined payload is the bytes of the
   joined tensor (`tensor_util::join_positions`, f32 and Q8_0), the rows the worker reads as one
   only where each forward continues the last's cache and tree (`dsd_flow::reads_on_from`).
2. **An answer nobody will read** (counted in the same build, `dead_here`): after a miss, every
   forward still on its way carries rows under the missed branch — about a round trip of
   forwards, ~5 at this rate, so up to ~100 of 285 on prose. This node runs its last layers on
   them only to keep its cache in step with the peers' (must-keep (2) of #369). Their keys and
   values are read by no live row (`RowVisibility`: a row sees its ancestors), so a forward that
   only HOLDS their places would keep the cache in step for no compute. **Measured** (2026-10-30,
   rig `prose1030` arm 1, batched tail off): 121 of a prose reply's 257 tail forwards (2,358 rows)
   and 23 of a code reply's 105 carried only rows off the walk's way — and that counts only the
   tree's current epoch; the forwards out when it is rebased come back for rows the rebase cuts
   on top of those. **Built without a worker change** (`dsd_flow::dead_wait_on`, opt-in
   `SWARMLLM_FLOW_DEAD_WAIT=1`): such an answer WAITS (`reaches_the_walk`) — read in front of the
   next answer the walk can reach, in one joined forward (a forward is bound by its weights, ~3 ms
   for 23 more rows against ~19 alone), or never, where a rebase's forward cuts this node's cache
   at or below the first one waiting (`cuts_below`). Holding places instead would have needed a
   worker message, a zero write per layer and every cache layout's rules; waiting needs none.
   **Measured 2026-10-30 (rig `tailjoin1030`, item 1's numbers in FUTURE_WORK #369): item 1 ON
   (every joined arm beat the old arm beside it, prose +13-38%); this item OFF** — the tail a
   further ~35% cheaper and the replies no faster, every prose reply's forwards queued on their
   way (630-923 ms) and the guard narrowed the tree to 2 rows a level. Freeing this node let its
   sends leave in bursts — the likeliest reading, unproven; the guard narrows ROWS for a wait the
   rows do not cause (gotcha #193). Pacing the sends comes before it is tried again.
3. **Bigger sends** (6 levels or 16 rows a level a drafter call, 2026-10-30 real link, one
   binary): levels 6 read 7.11 / 15.27 tok/s (prose / code, two runs) and 16 rows 6.54 / 15.12
   against the default's ONE clean run 6.65 / 14.98 — the second default run lost its drafter to
   the processor (item 5) and ran the stream at 3.55 / 8.39. A few percent at most: the live
   budget, not the forward count, caps what a bigger call adds.
4. **The link is the route, not the software** — re-measured, the finding is #327's (TH↔AU
   crosses Los Angeles, ~390 ms against ~185 ms direct; § 5's "P"). A TCP handshake to the far
   peer's public address takes 359-387 ms; the swarm measures 373 (min), so our stack adds ~13 ms.
   A one-hop detour (RON, Andersen et al. 2001; colocation relays beat the direct path in 76% of
   pairs, a >100 ms gain in ~5%) still needs a peer near Singapore; the swarm has none today (the
   other peers are in Germany, Italy and this LAN). Every miss costs a round trip, so such a peer
   would cut the waiting by up to half. Coordinates cannot propose the detour: it is exactly a
   triangle-inequality violation, which they smooth away.
5. **A near-full card loses the drafter to the processor** (2026-10-30, rig `bigsend` arm 6): the
   planner left room for it (`room_for=25`), the spawn refused it by 23 MB (5,390 + 966 against
   6,333), and the reply ran the stream at half speed. The planner prices this node's layers as a
   middle segment (`process_pool::cost_curve_of`, the whole-model line, no end tables) while the
   spawn charges `[0,12)` its token table — the bound is about a layer too generous in both arms;
   in the good one 25 fitted by 38 MB. → FUTURE_WORK.
6. **Before any graph-capture work on 24-row forwards**, `SWARMLLM_PROFILE=1` (DIAGNOSTICS § 3d)
   settles whether a tail forward's ~21 ms is the card or the host: its weights are ~3 GB at
   ~448 GB/s, ~7 ms. Capturing ≥ 9 rows needs a fixed MMQ tile (`executor.rs`'s
   `MAX_POSITIONS`), and padding to a bucket was estimated at ~1 ms saved a forward. The excess is
   ~25 µs a launch at ~28 launches a layer — more than WSL2's ~10-12 µs a launch, so gaps or
   under-filled kernels are in it too; a timeline (nsys) separates gaps from kernel time before
   any of: rows padded to a few captured sizes (SGLang's and TensorRT-Edge-LLM's verify graphs pad
   the tree to a fixed width, masked), fewer launches a layer (residual + norm + q8_1 quantize in
   one), a megakernel (1.1-1.3x over graphed launches on L4/L40S/A10G, AutoMegaKernel 2606.09682;
   Hazy's Llama-1B 2.5x over vLLM is batch 1 on H100). Windows' hardware GPU scheduling, which
   NVIDIA names the first WSL2 setting, is already on here (dxdiag: `Enabled:True`, 2026-10-30).
7. **Ideas held for later, with their prior art.** The target's own top-k returned per row to
   grow the tree (Token Recycling, 2408.08696; here the tail is on THIS node when it starts and
   finishes here, so no wire change). An EAGLE-style head, possible in that shape because the
   target's top-layer features are computed HERE (EAGLE-3 reads a low, middle and high layer;
   the far range could end at the middle one) — llama.cpp's EAGLE-3 is chain-only (PR #18039).
   For a thin uplink, 4-bit activations with ~1/64 of channels kept wide (Dong et al.
   2411.07942: 97-99.5% accuracy) and a residual against a predictor both ends hold.
8. **Read off vLLM, TensorRT-LLM and NVIDIA Dynamo (survey, 2026-10-30, docs only).** Already here:
   suffix-style copies grafted onto the tree (`FlowOp::Copy`, SuffixDecoding 2411.04975), the
   fused QKV-bias transpose, best-first trees, shared-noise drawing. New, in order: (a) the
   verify's accept on the card (TensorRT-Edge-LLM `eagleAcceptKernels`: top-1 and the tree walk
   on the device) — worth it only if the timeline shows a host gap between forwards; (b) PARD
   (2504.18583, K drafts in one drafter forward) was set aside in § 5 because drafting does not
   bind E in the default shape — in a split that starts and finishes HERE the drafter shares this
   node's serial card time, so it returns if item 1's A/B shows that time binds; (c) Dynamo's
   router samples among routes by a softmax of their costs (`router_temperature`) so clients do
   not all herd onto the fastest worker — an idea for a swarm with many requesters, not this one
   yet. Not for us: NIXL (RDMA / UCX / GPUDirect, no WAN transport), Dynamo's planner (an
   autoscaler for replicas a swarm does not own).
9. **Past all of it: the flow's dead rows are free lessons** (2026-10-30, conceptual — each part
   is tested before it is built). A flow has the target judge ~40 rows for every token it emits
   (6,500 rows, 160 tokens) and ~97% of them die. Every system read here discards its verifier's
   output and verifies ~60 rows a round; this one produces thousands of verdicts per reply — the
   target's distribution after a plausible prefix — and, in a split that starts and finishes
   here, the drafter's state and the target's final state for the SAME rows land on this node.
   Three uses, in order:
   (a) **A dead row costs this node nothing** — item 2, measured first (`dead_here`).
   (b) **Then the breadth is paid by the far card's idle time**: 256 and 448 live rows did not
       beat 128 (§ "A tree that flows", 2026-10-23) while every dead row cost this node a
       forward of its last layers; with (a) the extra rows cost only the far card, idle about
       half of each cycle. Re-test the live budget once (a) is in, and widen where misses
       come, not everywhere.
   (c) **Teach the drafter during the reply from the target's verdicts** — first a
       per-conversation correction of the drafter's scores toward the tokens the target
       actually prefers (its top choices per row, a few bytes a row), then a ridge map from the
       drafter's final state into the target's final space, scored by the target's own head
       (here in that shape) — the stand-in's fitting machinery (`stand_in::ridge_inverse`)
       applied across MODELS rather than layers. Prior art found: online distillation (OSD,
       2310.07177) fine-tunes the drafter; EAGLE/Medusa/Hydra heads read the TARGET's state;
       model stitching (Bansal et al. 2021) maps between networks offline — none fits a
       cross-model map online from a verifier's by-product. Test first, offline: teacher-force
       recorded prose replies through both models (`~/llama.cpp-ref/dump_logits`), simulate the
       drafter's top-8 coverage of the target's choice with and without each correction learned
       position by position. Prose misses (~20 a reply) are what hold a prose split below local.
   (d) **Not the target's own middle state as a second guesser** (researched 2026-10-30, parked):
       this node holds the target's hidden state at its last layer for every row it sends, so a
       learned map from it through the real head could foresee the far end's draw a round trip
       early, or re-score the drafter's rows. No published top-1 agreement exists at that depth
       for 7-14B models (the tuned lens, 2303.08112, reports perplexity and KL only; one
       unverified 2026 preprint, 2607.15495, reads near zero mid-depth); exits at 38% depth
       accept 67-75% only after the whole model is trained for it (LayerSkip, 2404.16710, Llama2
       13B layer 15 of 40), and a light adapter at a shallow exit accepts 0.48-0.52 (EESD,
       2406.03853) — below the 0.5B drafter's 0.69 first choice on chat. No prior work re-scores
       a separate drafter's tree with the target's middle state. "Speculation at a Distance"
       (2606.25091) puts acceptance first over a long link, as § "A tree that flows" measured.
10. **A bigger drafter in the default shape — fewer misses, and the drafter's call is now the
    cost** (2026-10-30, rig `drafter1030`, one binary, this card [0,16) + AU [16,48) and the head,
    the far asked the SAME range by every arm — alternating ranges inside its worker's window
    grew its worker onto its processor or refused it, FUTURE_WORK #377). Decode tok/s, three
    runs each, arms interleaved, 160 tokens:

    | drafter | prose | code | prose misses | drafting ms a call | waiting (prose) |
    |---|---|---|---|---|---|
    | Qwen2.5-0.5B Q8_0 (general) | 6.58 7.96 7.65 | 18.97 18.92 19.82 | 12-18 | ~18 | 5.2-9.8 s |
    | Qwen2.5-1.5B Q8_0 | 9.23 9.80 9.32 | 21.64 24.40 26.93 | 10-11 | ~26 | 4.1-4.7 s |
    | Qwen2.5-3B Q4_K_M | 8.38 10.10 8.79 | 25.33 23.33 23.38 | 6-9 | ~35 | 2.3-3.6 s |

    The 3B halves the 0.5B's prose misses and its waiting, and loses it again in its own calls:
    a 3B prose reply spent 9.1-9.8 s drafting, 4.4-4.7 s in this node's 16 layers (~16 ms a
    forward of ~23 rows, `DIAG: worker forward answered`) and 2.3 s waiting — this node busy
    85%. The replies are greedy and part between runs only at near-ties, within one drafter as
    across them (prose after 178-605 characters; code: the six 1.5B and 3B replies identical, the
    0.5B's "computes" for "calculate") — a card's row moves with the check's width, and every run's
    tree differs. **So the next lever is the drafter's call**: a
    3-level call of the 3B at ~35 ms is ~12 ms a level against ~5 ms of weights at the card's
    bandwidth, the 0.5B's ~6 ms a level against ~1.3 — the excess is per layer, not per byte
    (0.25-0.32 ms a layer a level for all three). The worker now logs where a call goes
    (`DIAG: a guessing model's flowing tree ended`: its levels' forwards to their ranking's
    read-back, the host's queue, the requester's ops, rebases) — read it before choosing between
    a captured multi-level call, fewer launches a layer, or the hop to the worker. Halving the 3B's
    call alone would put that prose reply near 11 s, ~14 tok/s: past this machine alone (~13).

    **Read 2026-10-30 ~13:15-13:35Z — the call costs what a level costs on an IDLE card; nothing
    in the split adds to it.** The worker's log (0.5B prose: 686 levels, 5,818 rows, forwards
    3,631 ms of a 3,757 ms total; ops, rebases and the hop to the worker ~0.3 ms a call) against
    `prefill_bench` `SWARM_BENCH_GROW=64,8` on this card with nothing else running ("from
    nothing": 8 levels of 8 rows, graphs warm, min of 5): 0.5B 4.2 ms a level, 1.5B 7.9, 3B 11.0,
    graphs off 6.6 / 11.9 / 14.4 — in the flow 5.3 / 8.7 / 11.8-13.5 with graphs and 3B 18.1-18.5
    graphs off (rig `graphoff1030`, three arms each: code 17.81 with graphs against 15.66 without,
    prose 8.08 against 7.78). A tree's size does not move it (3B: 11.0 / 11.9 / 12.0 ms a level
    growing 64 / 256 / 512 rows). (The bench's "on a seed" line is ONE level after reading a seed,
    not eight — 1.6 ms "a level" was that misread.) What is left beyond the weights' read is
    ~0.14-0.2 ms a layer whatever the drafter: the host recording each level's ~20 operations a
    layer into the capture before it is updated and launched (`cuda_graph`'s "update" + "launch"
    are 0.3-0.5 ms a level of it). The lever is a level REPLAYED without being recorded — every
    input a level reads already sits in fixed device buffers (`layers::stage_tree_forward`); the
    lengths and positions that change are kernel arguments, which would have to be read from
    device memory instead — and the next level's rows chosen on the card from the ranking it
    already makes (`tree_rank`), so a 3-level call is one launch and one read back. Not small:
    every kernel a level runs takes its cache length from memory. Parked behind the release.

## 11. The 70B through this PC and AU, evening of 2026-10-30

The 70B (Llama-3.3 Q4_K_M) runs here as this node's [0,16) (10 layers on the card, 6 on the
processor, the 3B drafter on the processor) and AU's [16,80) (23 on its RTX 3060, 41 on its
i7-12700's processor), the chain stream between them, ~1.15-1.32 s from a chunk's send to its
answer. What was measured or ruled out:

1. **AU's checks ran at its narrow decode width — fixed** (#379, `cpu_pools::swarm_check_threads`).
   A check of 2-32 positions that a far end runs for someone else's split ran on the pool
   calibrated for ONE position (narrower than the cores its owner offered). On this machine,
   7B processor-only, 8 cores offered and decode pinned to 4: 4 positions a forward 191 → 128 ms,
   8 positions 353 → 218 ms; one position unchanged. AU's 41 processor layers are most of its
   check, so its answer gets shorter once it runs this release.
2. **The drafter: the 3B stays.** The 1B (Q8_0) against the 3B (Q4_K_M), both on the processor,
   one binary, arms d3 · d1 · d1 · d3 with three prose prompts at 96 tokens each
   (`~/swarmllm-70b/ab70drafter.sh`, rig data dir with exactly one drafter; AU pinned to
   [16,80)): 3B 1.79 2.08 1.93 · 1.78 2.01 1.97 tok/s, 1B 1.64 1.87 1.74 · 1.65 1.83 1.71. The
   1B guesses 1.6x faster (33-34 against 54-70 ms) but its acceptance falls 0.74-0.77 → 0.69-0.71
   and the stream restarts 25-30 times a reply against 19-24. Every restart waits a whole check
   at AU, so a cheaper, worse guess costs more than it saves.
3. **The processor's few-row kernels are not the lever.** They already beat llama.cpp at 2
   positions (gotcha #850), repacking bought nothing on this CPU (#843), and the vendored
   `k_quants` matmul is column-outer with 8-row Q4_K kernels.
4. **Copying processor-held weights to the card for a check pays only past ~31 rows here**
   (5 ms/GB a row at 8 threads against 154 ms/GB for the copy) — llama.cpp's 32. A chain's
   3-row chunk is far below it; nothing to change.
5. **A first token waits one round trip for the peers' memory answer** (~0.4 of 5.4 s warm) →
   FUTURE_WORK #380.
6. **The card's room is the next lever**: this node keeps the 70B's 564 MiB token table on its
   card, where llama.cpp keeps it on the processor — about one more of the 70B's 519 MiB layers,
   or part of a card drafter → FUTURE_WORK #347.
7. **A peer that cannot hold the whole model now stays in the plan** for the part it can (#378,
   verified live 14:21Z with the whole model forced onto AU: refused at 41,840 MB short,
   re-planned with AU held to fewer layers).
8. **Where the reply goes, and the lever it leaves** (answer arrival times in arm 4's log): an
   answer every ~0.69 s while AU is busy, 1.41-1.50 s after a refused guess — ~63% of a prose
   reply is restart cycles and AU idles ~0.73 s in each, waiting for this node's restart. The
   drafter's second choice is the 70B's token in 60% of its misses (teacher-forced replay of six
   70B replies, calibrated against the live alpha and restart count), so a restart prepared before
   the verdict and left at the far end would catch 28% (one spare a chunk) to 60% (one at every
   position) of them → FUTURE_WORK #381, ~+15% expected with two spares a chunk.

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
