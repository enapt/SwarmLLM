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
   k — **stop if the head alone is under ~40% or the mix under ~78%.**
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
6.15/6.90). Doubling E takes N squared (16K rows a round trip — no uplink carries that) or H halved.
**Only guesses written by something close to the target halve H.** Every design below is judged by
which of the two it moves.

E/P itself is Little's law — the reply's rows in flight are its speed times the round trip — so it
binds only where the tree cannot cover more. Two caps meet: **coverage** `tok/s ≤ D(N)/P`, and
**rows** `tok/s ≤ R·D(N)/N` — each token of the reply costs the N/D rows grown around it, R the rows
a second this node sends. On TH↔AU (R ≈ 24 rows / 54 ms ≈ 440, N 128, H 0.88, D 6.5) they are 15
and 23 tok/s: coverage binds, which is why a cheaper requester ran more forwards and no more reply,
and why 256 or 448 live rows did not pay (their D +0.8 is spent on rows the uplink and a .248 far
end, reading every row whole, must carry). The shared-noise ratings (H 0.965 → 0.883) predicted +8%
coverage and measured +4.5% unqueued with misses down a third (`docs/invariants/inference.md`). A
real-link reply's time is all this node's own work but 3-7 s of waiting in 26-30 s (`realrate.sh`
breakdown) — busy is not bound: the drafter grows whenever the budget lets it.

**6.2 The far end's lead cannot shorten a miss.** The far end learns the true token half a round
trip before this node, but every guess must pass this node's half before the far end can check it,
and the far end cannot compute that half. So no protocol where the far end only reorders, redraws
or re-ranks recovers a miss's round trip — it costs P whatever the far end does (checked case by
case: far-end drafting, far-end top-feature drafting à la EAGLE, transplanting a sibling row's
state). Two exits exist: a cycle that leaves this node out (near chains), or a far end that can
GUESS without this node's half — the target's exact tail fed an approximation of the head (6.6).

**6.3 The far end writes 30× more of the target than the reply keeps — recycle it.** A flow draws
EVERY row it is sent: ~12K draws for a 400-token reply, 97% of them on branches that died. Each is
the target's own next token in a context one alternative away from the true one, and because the
noise is keyed by POSITION, a dead branch draws with exactly the noise the live one will — so where
the target's two distributions agree at the draw's margin, the dead branch's draws ARE the live
branch's. At a miss, the dead spine's draws are a target-written continuation that arrives in the
same answers that reveal the miss. Prior art recycles the target's output distributions (Token
Recycling 2408.08696, Lookahead's n-gram pool 2402.02057) or its hidden states at rejected positions
as context for a TRAINED drafter (Carryover Drafting 2609.14717: +6.5-14.7% accepted length; ReTrace
2608.29748; Lyanna 2602.21224) — Carryover leaves trees as future work. None has coupled draws, which
make a dead branch's TOKENS the live branch's, untrained, and a flow gets them for free. **Decides it**: `~/swarmllm-ref/spec/counterfactual.py` — at a spine miss, how often the
dead branch's draw is the true next token where the drafter's re-rooted draw is not (rescue), and
how deep the dead run stays right.

**6.4 In a flow, the requester's card is worth more as drafter room than as a stage.** Speed is
set by the two caps of 6.1: holding a stage moves neither (it only decides who covers those layers),
while a bigger drafter lowers H — raising D and both caps — and costs R, which lowers the rows cap
alone. Coverage binds today (15 against 23 tok/s), so a drafter whose call costs ~1.5× (the 3B Q8_0
against the 1.5B's 27 ms for 3 levels: ~67 ms a forward, R ≈ 360) still pays where it lowers H: IF
the 3B's H were 0.7, N 128 would sit at ~18 (coverage) / ~22 (rows) — +22%. H is the unknown. The `DRAFTER_MAX_FRACTION` cap (sized for local
speculation, where the drafter's time IS the cost) is then the wrong bound on a far link. Where the
requester's card cannot hold the bigger drafter, a peer on its LAN can run it (3 ms away: drafting
as a near service). **Decides it**: the drafter-size curve (0.5B/1.5B/3B against the 7B,
`gumbeldump.py` + `treesim.py`: H and D at N 64/128), then the 3B's call on the card.

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

## Sources

Petals 2312.08361 · Pope et al. 2211.05102 · StagFormer 2501.15665 · Layer
Parallelism 2502.02790 · Lookahead 2402.02057 · CLLM 2403.00835 · Lamina
2405.01814 · MegaScale-Infer 2504.02263 · BloomBee 2604.21072 · Prompt Cache
2311.04934 · Preble 2407.00023 · SkyWalker 2505.24095 · CacheGen 2310.07240 ·
LLMCompiler 2312.04511 · Skeleton-of-Thought 2307.15337 · SpecExec 2406.02532 ·
Sequoia 2402.12374 · ML-SpecQD 2503.13565 · QuantSpec 2502.10424 · SPEQ
2510.18525 · Daliri et al. 2408.07978 · multistream-select 0.13.0 `Version`
docs · llama.cpp `--n-cpu-moe` reports (discussions #15396, dev.to 8 GB
Qwen3-Coder-30B) · Leviathan 2211.17192 · Chen 2302.01318.
