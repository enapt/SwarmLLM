# Many tokens per round trip: speculation across a split

**Written 2026-09-27 against v0.3.210-alpha, from a two-GPU split measured on
the live swarm** (RTX 3070 in Thailand, RTX 4050 in Belgium) **and replayed
acceptance data from two real model pairs.** Raw data and scripts:
`examples/spec_coverage.py`, `examples/spec_best_tree.py`, `examples/spec_projection.py` (they read and write under `~/swarmllm-ref/spec/`); the split
measurements are in `memory/perf_spread_0927_gpu.md`.

## The limit, stated as a number

A split decodes one token per trip around the machines. Token *n+1* cannot
start at the head until token *n* has been sampled at the tail and carried back.
Faster graphics cards make no difference to that trip.

| qwen2.5-coder-7b, 64-token replies | tok/s | ms/token |
|---|---|---|
| RTX 3070 alone | 43.7 | 22.9 |
| RTX 4050 alone (hand-off) | 11.45 | 87.3 |
| split 3070 L0-14 → 4050 L14-28, persistent stream | **2.85** | **351** |

Of the 351 ms, about 300 is the trip and about 50 is both cards' compute.
Tensor parallelism makes this worse, because it adds a network exchange at
every layer instead of one per token. **The only lever that breaks the limit is
producing more than one token per trip.** That is speculation. Everything below
measures how many tokens per trip real models allow, and what it would cost us
to collect them.

## Measured: how many tokens a trip can carry

Method (`spec_coverage.py`): the target model writes real replies (six prompts
covering prose, code, a code edit, Q&A and advice, 160 tokens each, llama.cpp
as the independent reference). Then the target and a small drafter of the same
tokenizer are both teacher-forced over the reply. At every position we record
where the drafter ranked the target's actual token, then replay the reply as a
sequence of network rounds under each scheme.

| | qwen2.5-coder-7b ← 0.5b drafter | llama-3.2-3b ← 1b drafter |
|---|---|---|
| positions | 891 | 909 |
| drafter's 1st guess is the target's token | 73.5% | 81.8% |
| … within its top 4 | 92.4% | 97.7% |
| n-gram (prompt lookup) hit | 9.5% | 6.5% |

Tokens per round trip, replayed on the real sequences:

| scheme | Qwen pair | Llama pair |
|---|---|---|
| plain decode (today) | 1.00 | 1.00 |
| n-gram lookup, chain ≤10 (the shipped default loop) | 1.10 | 1.07 |
| drafter chain of 4 (what `dsd.rs` does) | 2.81 | 3.42 |
| drafter chain of 8 | 3.35 | 4.46 |
| best-first draft tree, 16 nodes | 4.09 | 5.38 |
| … 32 nodes | 4.59 | 6.36 |
| … 64 nodes | 5.18 | 7.10 |
| … 128 nodes | 5.71 | 8.12 |

The trees (`spec_best_tree.py`) are shaped the way Sequoia shapes them. Their size
rule is the positional-acceptance model: a node's value is the product of
P(the true token is the drafter's k-th guess) along its path, and the tree is
the top-N nodes by value. Each shape was then **replayed** on the real
sequences, so correlation between neighbouring positions is measured rather
than assumed away. Model and replay agree within 4%.

**Retraction, in this document so nobody builds on it.** An earlier draft of
this research reported "12-20 tokens per round" for a tree extended
continuously at a fixed branching. That figure assumed every level had K
children precomputed along whichever path the tail took, which is a tree
exponential in depth. The head learns the tail's position one trip late, so the
depth it must cover is what the tail walks in one trip. Only the finite
best-first figures above are reachable.

## Why the shipped speculation does not deliver this today

`pipeline/dsd.rs` has the right shape (draft γ, verify all of them in one trip)
and cannot deliver it, for four reasons:

1. **It returns a full vocabulary per position.** A verify round carries back
   γ+1 f32 vectors: 5 × 152,064 × 4 B = **3.0 MB** for a Qwen round. At 20-50
   Mbit/s that transfer alone costs 0.5-1.2 s, which is more than the trips it
   saves. Projected, DSD as shipped runs at **1.8-3.4 tok/s** on this link,
   against 2.85 with no speculation at all. The n-gram loop has the same wire
   and the same problem (`FUTURE_WORK.md` § "The distributed n-gram miss round
   returns a whole vocabulary").
2. **Greedy only.** `speculative_common_eligible` refuses any temperature above
   0, and no client sends 0 by default. The n-gram loop already removed that
   gate for itself with the right rule, `speculative::sampled_accept_reject`:
   draw t ~ p and keep the draft iff t equals it. For a deterministic draft
   that IS the speculative-sampling rule, so it is exact at any temperature.
3. **The drafter is a GGUF path the user must set** (`draft_model_path`, llama
   feature). A default cannot depend on that, and a node must never fetch a
   whole model implicitly (CLAUDE.md). The drafter has to come through the
   shard system like any other model.
4. **Off by default**, behind `speculative_decoding` AND
   `decentralized_spec_decoding`.

## Projected speed, from measured parts

`spec_projection.py`. A round costs: the trip (300 ms TH↔BE, measured as 351 ms per
token minus compute), plus drafting (one drafter step per tree level, 4-10 ms),
plus the head's layers over the tree (12 ms + 0.25 ms/node), plus the hidden
states' transfer (3.8 KB/node, Q8_0, as measured on the prompt pass), plus the
tail's layers over the tree (1.3× the head's). Tokens per round come from the
replay table above. Qwen / Llama pair:

| link | plain | DSD as shipped | chain of 4, accepted at the tail | tree 16 | tree 32 | tree 64 |
|---|---|---|---|---|---|---|
| TH↔BE, 300 ms, 20 Mbit/s | 2.8 | 1.8 | 8.0 | 10.3 / 13.5 | 10.4 / 14.5 | 10.0 / 13.8 |
| TH↔BE, 300 ms, 50 Mbit/s | 2.8 | 3.4 | 8.1 | 10.7 / 14.1 | 11.2 / 15.5 | 11.4 / 15.6 |
| same continent, 60 ms, 50 Mbit/s | 9.0 | 4.7 | 25.9 | 28.6 / 37.7 | 27.0 / 37.4 | 24.0 / 32.8 |
| same country, 25 ms, 50 Mbit/s | 13.2 | 5.0 | 38.3 | 38.0 / 49.9 | 34.0 / 47.1 | 28.6 / 39.2 |

With a drafter at 10 ms per step instead of 4, the TH↔BE figures fall by about
10% (chain 7.5, tree 32 9.4 / 13.0).

What this says:

- **A drafter chain, verified at the tail, is ~2.8× on every link.** It is the
  bulk of the win and the cheapest to build.
- **A tree adds ~1.3-1.4× on a long link and little or nothing on a short one**,
  where the tree's transfer and compute stop being small next to the trip. Size
  the tree from the measured trip; never use a fixed size.
- **The factor does not shrink with distance.** `regional_pipelines.md` Stage 4
  said speculation is "only meaningful once Stage 2-3 have brought RTT down"; the
  measurement says the opposite. The saving per round IS a trip, so a long trip
  makes speculation cheaper to pay for, not dearer. Locality and speculation
  multiply; neither waits for the other.

## What was considered and rejected

- **Layer-skip self-speculation (SWIFT, CLaSp)** was Stage 4's first choice in
  `regional_pipelines.md`. On a split it cannot draft without crossing the
  network: the skipped model still needs the tail's last layers and output head
  for every drafted token, so every draft token pays the trip it was meant to
  save. The one local variant, boomerang's own first and last layers with the
  remote middle skipped, would draft from ~7 of 32 layers. That is far below a
  same-family drafter's 73-82% first-guess rate.
- **Early exit / logit lens at the split point** (the head's output through the
  final norm and output head). Without a trained adapter, half depth is a weak
  predictor. It also needs the output head at the head, which is 545 M
  parameters on Qwen.
- **n-gram lookup as the default** (the shipped loop): 1.07-1.10 tokens per
  round on general text. It still pays on copying workloads (code edits), and
  its payoff gate stays.
- **Batching many conversations** raises total throughput and does nothing for
  one person's reply.
- **Tensor parallelism** exchanges once per layer instead of once per token.

## The plan

### Phase 1 — drafter chain, accepted where the tail samples (~2.8×)

**Status 2026-09-27:** items 1 and 2 are built and measured (released in v0.3.211-alpha):
`split_rig.sh repeat` on llama-3.2-3b, every result from the tail 43-59 bytes
instead of 513 KB-2.5 MB, replies 119/121 at llama.cpp's first choice, and
byte-identical to a run whose v0.3.209 tail was sent the old request. Item 4
(γ from the trip) shipped in v0.3.211 too. **Item 3 built 2026-09-28, opt-in**
(`inference.draft_model = "<a model id this node holds>"`, beside the two DSD
switches): `pipeline::engine_drafter` drafts with a small same-family model run
by its own worker from its shards (`DaemonMsg::Draft`, `SplitModel::draft_after`)
— no `llama` build, no model file. The worker keeps the drafting cache between
rounds and each call reads only what the last check did not confirm. The
vocabulary check accepts a list padded with unused entries: Qwen2.5-Coder-7B
lists 152,064 tokens, Qwen2.5-0.5B 151,936, identical over the shared ids, the
7B's extra 128 `[PAD…]` of type unused. Unset, the drafter is chosen: the
largest held model that can draft (whole, no recurrent state, the target's
vocabulary) at most a quarter of the target by layers × width². Not yet:
acquiring one a node does not hold.

**Measured on the real link 2026-09-28** (same TH↔BE split and binary as the
table below, drafter qwen2.5-0.5b-instruct-fp16 on this RTX 3070, the far node
on v0.3.211; `~/swarmllm-bench-0928/run_drafter.sh`, `drafter.jsonl`):

| arm | tok/s (median, min-max) | guesses kept | γ settled |
|---|---|---|---|
| plain split, before | **2.71** (2.55-2.80) | — | — |
| engine drafter, greedy | **4.63** (3.21-5.55) | 0.65-0.73 | 3-7 |
| engine drafter, T=0.7, shared noise | **5.55** (4.78-6.07) | 0.64-0.75 | 3-5 |
| plain split, after | **2.29** (2.21-2.36) | — | — |

1.7-2.4× over the plain arms either side. The fitted check cost was 300-430 ms
fixed plus 0-32 ms per position (the far node's card), so γ stayed at 3-7
instead of the 11-16 the constant model chose. Mechanism in every request's
`DSD: request complete … drafter=qwen2.5-0.5b-instruct-fp16` line. Two warm-up
requests failed, neither from this path: the far node's worker ran out of card
memory on a prompt pass (`CUDA_ERROR_OUT_OF_MEMORY`, its own v0.3.211), and the
connection to it dropped once in the last plain arm.

**A drafter that fails finishes the reply without guessing, and a check that
does not come back fails the request** (both 2026-09-28). Checked on the same link
with `SWARMLLM_FAULT_DRAFT_FAIL=3` (the drafter fails its third call of every
reply): 3/3 replies complete at 64 tokens, each logging `DSD: the drafter failed —
finishing this reply without guessing ahead`, at 2.43 tok/s — plain-split speed
— against 5.45 for the same binary unfaulted. Before, a drafter failure ended the
reply where it happened. And a connection that dropped during the first check
(the far node's link dropped four times tonight) was answered as a finished
one-token reply with `finish_reason: stop`; the check's failure now propagates, as
the n-gram loop's always has, so `keeping_the_partial` reports it and the router
retries a request that has streamed nothing. The single-peer path (Item 2) had
the same swallow and is fixed the same way. A review of that fix found the same silent `stop` one
arm over, for a check reply that CANNOT BE READ (too few rows, non-finite logits,
a walk claiming tokens never guessed): in DSD, in both arms of the default n-gram
loop and in Item 2 — where non-finite logits went on to emit token id 0. All
four now fail the request the same way.

**The drafter reads the prompt while the target does** (2026-09-28): its first
call (`engine_drafter::read_ahead`, a guess-free `DaemonMsg::Draft`) is spawned
beside the target's prompt pass instead of after it. On a freshly started node its
worker's spawn began at 17:09:28 and the model was loaded at 17:09:38, against the
target's prompt pass ending at 17:09:36 — the first round waited ~1.6 s for it
instead of the whole load (9.4 s here, both workers loading from cold at once;
4.4 s alone).

**The drafter is a guest of this node's memory** (2026-09-28): it loads with
`process_pool::Tenancy::Guest` — only memory that is free, never another
model's — and reads ahead only once the target's own segments here are loaded.
As an ordinary load it could evict the target's segment, idle between chat turns
past the pool's 5 s reclaim floor, and push it onto the processor
(`docs/invariants/memory.md` § "A model loaded for another model's request is a
guest"). So on a card the target fills, the drafter runs on the processor: the
default decision must be measured there too (gate step 12f). **And a drafter that
has lost the reply's context says so**: `draft_after` refuses a cache shorter than
the one the call continues (expired, worker replaced), and the reply finishes
without guessing instead of guessing from a context never written.

**Shipped OFF by default in v0.3.212 — the decision and its evidence** (2026-09-28,
`~/swarmllm-gate-0212/{decide,verify}212.out`, then the artifact gate). The rule
was: flip only if, with it on, the failure rigs pass, replies score against
llama.cpp as well as plain, a coordinator with no drafter is unchanged, and the
drafter never takes the card from the target. Three held — greedy DSD 118/119 of
120 rank-1 against plain 116/118; no drafter → the n-gram path, identical replies;
12f, A on the card: the target stayed there and the drafter ran on the processor.
**Failover did not**: with the drafter resident, the plan for the failing request
had no standby, so B's range was retried rather than taken over (reply correct,
119/120). The same binary without speculation took it over. That is FUTURE_WORK
#140, and the flip waits on its discriminator (the drafter on A's card, so it
takes no shared RAM). The same rig also found #749 — attempt-scoped drafter state
keyed by the request id, which retries reuse — fixed before the tag.

**An unsure guess ends the round** (2026-09-28): the drafter stops after a guess
it gives less than 0.4 (`SplitModel::draft_after`, Hugging Face's
`ConfidenceCriteria` / `assistant_confidence_threshold`; `SWARMLLM_DRAFT_CONFIDENCE`
sets it, 0 turns it off). A/B on the same link (`run_drafter_conf.sh`,
`drafter_conf.jsonl`): at T=0.7 4.96 tok/s with it against 4.32 without, greedy
4.27 against 4.80 — **no speed difference within this link's noise** (a third
arm lost its far node twice and is not counted). What moved consistently is the
WORK: guesses proposed per 64-token reply fell from 85-110 to 64-79 with the kept
ones unchanged (29-39), i.e. ~25% fewer positions for the far computer to check —
compute a peer lends to the swarm, and the per-position cost `CheckCost` fits
when that peer checks on its processor.

1. **Accept at the tail.** A verify forward carries its draft tokens. The last
   segment walks them with the request's own sampler: sample position i with
   `sample_token_with_params_history`, `generated_ids` extended by the tokens
   already accepted this round; keep going while the sample equals the draft.
   It returns the accepted ids plus the one it sampled at the first mismatch
   (or the bonus), which is **4 bytes a token instead of 608 KB a position**.
   This is `speculative::sampled_accept_reject` moved to where the logits
   already are.
   - **Wire:** a new trailer flag on `LayerForward`, gated at the SENDER on a
     new `features` bit (CLAUDE.md: a new trailer is not a no-op for an older
     peer). A tail without the bit gets today's spec-logits verify.
   - **Boomerang** (the default split shape) already has the tail at the
     coordinator, so its logits never cross the network; there the fix is items
     2-3.
   - The same wire fixes the n-gram loop's miss round (FUTURE_WORK entry above)
     for free.
2. **Any temperature.** Drop `temperature == 0` from `speculative_common_eligible`
   once acceptance goes through the sampler. It is the rule the n-gram path
   already ships.
3. **A drafter from the shard system.** It must share the target's tokenizer
   (compare vocabularies, not embedding sizes: Qwen pads 151,936 to 152,064).
   Qwen2.5 → qwen2.5-0.5b-instruct, Llama 3.x → llama-3.2-1b-instruct, Gemma 2
   → gemma-2-2b-it. It runs as an ordinary model in its own worker through
   `ModelProcessPool`, drafting γ tokens as a greedy `generate` whose prompt
   is the conversation so far. The prefix cache makes each call incremental, so
   no new worker message is needed for a chain. Acquire it only for a family
   this node has actually coordinated a split for, and only within the storage
   budget.
4. **γ from the trip, not a constant — DONE 2026-09-27.** `dsd_controller::best_gamma`
   maximizes `E(α, γ) / (fixed + γ·draft)` with both costs measured each round
   and α estimated from whole rounds. `GammaController`'s multiplier could never
   move γ off 4 (4 × 1.1 rounds back to 4). Rig (CPU drafts at ~200 ms each,
   loopback verify ~170 ms): γ = 1, the right answer there; a 300 ms link with
   25 ms GPU drafts at α = 0.92 gives ≈ 12.

**Measure it:** `spread_bench.py` on the forced split
(`pretend_peer_holds`), A/B inside one binary via an env switch. Prove the
mechanism fired: accepted tokens per round in the log, and bytes per verify
result from `payload_len`, which should fall from ~3 MB to tens of bytes.

### Phase 2 — draft trees (a further ~1.3-1.4× on long links)

1. **Tree attention in the split forward**: a node list with parent indices;
   each node's position is its depth; a mask where each node attends to the
   committed cache plus its own ancestors. The CPU and CUDA attention kernels
   take causal masks today; the verify forward already forces standard
   attention (`ForceStandardAttnGuard`), which can take an explicit mask.
2. **Commit the accepted path in every segment's cache.** Today's
   `truncate_kv_to` handles a chain (keep a prefix). A tree needs a gather of
   the accepted nodes' positions. Every segment does it, from the ids the tail
   returns.
3. **Walk at the tail**: sample at the root; if the sample is a child, move to
   it and sample again; stop at the first sample with no child. Exact at any
   temperature for the same reason as the chain.
4. **Best-first tree from the drafter's own probabilities** (EAGLE-2's dynamic
   tree), sized from the measured trip and bandwidth by the round-cost model
   above (Sequoia's rule).
5. **Draft the next tree while a verify is in flight** (SpecEdge's proactive
   drafting), extending the most likely path. This hides drafting time rather
   than adding tokens per trip.

### Phase 3 — the prompt pass, pipelined in chunks

A split reads a prompt serially: the head's layers, then the transfer, then the
tail's layers. Sending the prompt in chunks lets the head read chunk k+1 while
chunk k crosses and the tail reads it. That comes close to the slowest stage
alone. On a ~2,000-token prompt with healthy cards that is roughly
6 s → 4 s. It also stops a prompt pass being one multi-megabyte frame, which is
the frame `FUTURE_WORK.md` #133 saw a healthy peer never finish reading.

### Phase 4 — a SHADOW of the far layers, and shared randomness

**The limit every phase above works inside:** time per token ≈ 1/v + m·RTT,
where v is how fast the near machine drafts and m how often the far machine
disagrees. A small drafter keeps m high (26-33%), so no amount of structure gets
past ~10-15 tok/s at 300 ms. The lever is m itself: draft with a near-copy of
the WHOLE model.

**The shadow.** The near machine keeps its real layers and a LOW-BIT copy of the
far machine's layers (plus the output head at full precision). Its drafts come
from nearly the real model; its errors cost speed, never correctness, because the
far machine still runs the real layers on every token.

**Shared randomness.** At temperature > 0 even a perfect copy "misses" whenever
the real model draws a different token (a deterministic draft is accepted with
probability p(draft)). Both sides sample by Gumbel-max with noise keyed by
(seed, position, token id); each side still draws an exact sample of its own
distribution, and a close copy draws the same token (Daliri et al., 2408.07978,
"drafter-invariant speculative decoding").

**Measured** (`~/swarmllm-ref/spec/coupling.py`, `coupling2.py`, `shadow.py`,
Qwen2.5-Coder-7B Q4_K_M as the target, 720 positions of its own replies, shadows
REQUANTIZED from the Q4 file — which is what a swarm that holds only Q4 shards
could make):

| predictor | greedy | T=1.0 deterministic draft | T=1.0 shared randomness | coupled top-2 |
|---|---|---|---|---|
| 0.5B drafter | 67.6% | 61.1% | 64.3% | — |
| whole model Q2_K | 81.8% | 70.6% | 82.7% | — |
| far half Q2_K | 84.0% / 86.8%* | 71.3% | 84.8% | 96.2% |
| whole model Q3_K_S | 90.7% | 73.5% | 89.9% | — |
| **far half Q3_K_S** | **93.8%** | — | **92.5%** | **98.7%** |

\* the two runs used different system prompts (`coupling.py` vs `coupling2.py`).
Shared randomness recovers ~19 points at T=1.0 for a close copy and ~3 for a
dissimilar drafter, as the theory says (it tracks distribution distance).
**Model size matters more than source precision:** Qwen3-1.7B's far half at Q3,
made from a Q8 file, agrees only 82.5% — small models are fragile under low-bit
copies, the large ones a split exists for are not.

**Projected, v = 44 tok/s (this RTX 3070's local 7B decode), far half Q3:**

| design | 300 ms (TH↔BE) | 30 ms |
|---|---|---|
| today, no speculation | 2.85 (measured) | ~13 |
| rounds (DSD's loop, γ=8, shadow as its drafter) | ~12 | ~33 |
| continuous stream, one draft line (m = 7.5% at T=1) | ~22 | ~40 |
| continuous stream, extra lines where the shadow is unsure (m ≈ 1.7%, ~1.3 lines) | ~29 | ~42 |

⚠ **Correction, same day:** an earlier projection counted the shadow's second
choice as avoiding a stall. In a stream it does not unless that choice was also
drafted ONWARD, which is what the extra lines are for — and they cost the head
a batch of lines per step, not one.

**What raises the ceiling beyond these:** v itself (the drafting engine is
submission-bound: `docs/plans/local_decode_submissions.md`), and RTT (a nearer
holder of the far half — `regional_pipelines.md`). As m·RTT shrinks the stream
runs at v, the near machine's speed on the SHADOW, which can exceed the real
model's local speed once decode is bound by bytes rather than submissions.

**Built and measured on the split rig 2026-09-27** (llama-3.2-3b, A=[shard 0]
coordinating, B=[1-3], DSD with the 3-bit far-half shadow as its drafter, γ=4):
greedy **3.8 tokens per round trip** (replies 119/121 at llama.cpp's first choice,
every tail answer ≤ 59 bytes); at T=0.7 with shared noise **4.11** against **3.71**
for fixed guesses (80.6% vs 70.0% of guesses kept). Round trips on loopback cost
nothing, so the rig measures tokens per trip, not speed; the WAN run needs the far
node on this build.

**Measured on the real link 2026-09-28, both ends on v0.3.211** (qwen2.5-coder-7b,
this RTX 3070 L0-14 in Thailand, bf7b3263's RTX 4050 laptop L14-28 in Belgium,
min RTT 226-233 ms, forced split, n-gram loop off, 64-token replies, one binary;
`~/swarmllm-bench-0928/run_wan.sh`, `wan.jsonl`):

| arm | tok/s (median, min-max) | notes |
|---|---|---|
| plain split, request-response | **2.86** (2.78-2.99), again **2.78** | V1Lazy on BOTH ends now |
| plain split, persistent stream | **2.81** (2.77-2.82) | no longer faster — see below |
| DSD, 3-bit far-half shadow, greedy | **3.35** (2.78-3.48) | warm-up request 7.31 at γ=4; α 0.82-0.91 |
| DSD, shadow, T=0.7, shared noise | **5.05** (4.91-7.34) | α 0.87-0.91 |
| DSD, shadow, T=0.7, fixed guesses | **6.06** (4.67-6.50) | α 0.78-0.80 |

- **The persistent stream's edge is gone.** #130 measured it at 351 vs 838
  ms/token on request-response, then V1Lazy took rr to ~580 with only OUR end on
  the build. With both ends negotiating V1Lazy, rr and the stream are equal: the
  far side's REPLY substream paid the remaining round trip (gotcha #743). #133 is
  now a robustness item only; the default path already runs at the stream's speed.
- **The far node checked on its PROCESSOR, and γ ran away.** Its check of 15
  positions cost 1.1-2.8 s a round in the greedy arm (a plain token's ~120 ms of
  compute fits the same picture). `best_gamma` modelled the check as a constant,
  saw a long round, and chose γ = 11-16 — making every round longer still: 3.35
  tok/s against a 7.31 tok/s warm-up request that had run at γ = 4. **Fixed:**
  `dsd_controller::CheckCost` fits the check as `fixed + per_position × positions`
  from the rounds, and `best_gamma_for_check` moves γ at most 2 a round so the
  positions spread enough to fit a slope (Dovetail, arXiv 2412.18934, keeps its
  processor-side candidate count small for the same reason). Unit-tested against
  the live shape (230 ms + 95 ms/position → γ settles 2-7, where the constant model
  climbed past 12). Not yet re-measured on the link.
- **The arms are not a clean A/B of shared noise.** The far node's check cost
  drifted between arms (0.42-0.70 s a round in the T=0.7 arms against 1.1-2.8 s in
  the greedy one — likely its worker moving between processor and card), so the
  fixed-guess arm's higher speed is the far node, not the coupling. What the arms
  DO show is the mechanism: guesses kept rose from 0.78-0.80 to 0.87-0.91 with
  shared noise, as on the rig.

**The shadow's memory — a premise to check before building 4e (2026-09-28).**
A shadow of the far layers costs the near machine ~76% of those layers at Q4
(Q3_K is 110 bytes per 256 weights against Q4_K's 144; Q2_K's mix ~66%) PLUS a
cache for every one of them — full precision in this engine, 393 KB per token
for a 14B's 48 layers. A machine with that much room could nearly hold the model
itself, which is exactly what a split is for when it cannot. Modelled on
consumer cards (card: 0.55 ms per layer, submission-bound; processor: layer bytes
at 17 GB/s; `docs/plans/faster_than_local.md` §3.3), a shadow does not fit for a
14B on 8 GB, an 8B on 6 GB, a 24B on 12 GB or a 32B on 16 GB, and every model
where it does fit already runs mostly on the card. **The measurements above
split a 7B that fits the 3070 alone** (`pretend_peer_holds`) — they show what a
shadow buys a round trip, not a machine that has the room for one in the splits
that happen. The same holds for the card+processor split of one machine, where
the shadow was built first (engine pieces, tested, parked on local branch
`shadow-drafter`: requantized far layers on the card sharing the real output
head, a round of guess-then-one-check-pass, the shadow's cache refreshed from the
real layers after every round). Before 4e: a shadow needs a cheaper cache (half
precision, or a window of recent positions) and a regime where the near machine
has spare memory for its OWN reasons — a split forced by shard availability
rather than by memory, where fetching the missing shards and running locally is
the competing answer. A small same-family drafter (Phase 1 item 3; 0.5 GB for a
0.5B) has no such problem and is the broader lever.

**Stages, cheapest first:**
- **4a. DSD with a shadow drafter** — config only once v0.3.211 reaches the far
  node (the tail must walk): `draft_model_path` → a shadow GGUF. Measures the
  round-based figure on the real link.
- **4b. Rounds in flight.** Keep drafting while a verify is out; an EPOCH number
  on every verify lets the tail drop work built on a prefix that just failed.
  Reaches the one-line stream figure without forking any cache.
- **4c. Shared randomness** in the worker's sampler and the drafter (counter-based
  noise per (seed, position, token id)), so 4a/4b hold at T > 0.
- **4d. Extra draft lines** — needs the tail's KV cache to fork per line.
- **4e. The shadow from the swarm** — derived "shadow shards" (the far layers
  requantized from the Q4 shard, deterministic, so any node of the same build can
  check one by recomputing it), never an implicit full download.

**The next batch, designed 2026-09-27 (after v0.3.211):**

*4e first — the drafter in OUR engine.* DSD drafts with llama.cpp from a whole
GGUF file, which production must not assemble from shards (CLAUDE.md) and which
duplicates the near layers in memory (the rig's 3-bit half-shadow of a 7B is
3.6 GB beside the worker's own copy of the same near layers). In our engine the
head worker already holds the REAL near layers; it adds the far layers at Q3
(shadow shards, requantized from the Q4 shards — dequantize, then candle's
k-quant quantizer) plus the final norm and output head, and one pass does both
jobs: the real near layers produce the exact hidden state the tail needs, the
shadow layers produce the guess. A guess and its verify input then cost ONE
near-half pass, not two, and the near cache is written once — truncated on a
rejection exactly as today. Worker op: "draft γ and hold" returning the guesses
and their near-half hidden states; coordinator: send those states to the tail
directly instead of re-running segment 0.

*4b — the continuous stream.* Rounds in flight need several outstanding verifies
per request, and `SharedState::pending_layer_results` is
`DashMap<Uuid, PendingLayerResult>` — one waiter per request, ~55 references in 9
files. Re-key it by (request, `ExpectedStep`) with a request-wide fallback for a
result that names no step (an older peer), so a late result of a discarded chunk
is refused by step, never by luck. Then chunks of a few guesses stream
continuously; a chunk carries the hash of the path it assumes (FlowSpec's
"continuous condition"), and both sides discard work built on a prefix the tail
has rejected — no cancel message needed. Projected on the 300 ms link with the
3-bit shadow: ~20 tok/s against ~14.5 for the best round-based γ; the gain comes
from overlapping the round trip, so it grows with distance.

Order: 4e before 4b — 4e is what makes shadow speculation usable without hand
configuration, and it removes the duplicate near-half pass 4b would otherwise
pay on every chunk. ⚠ **Revised 2026-09-28: see "The shadow's memory" above
before building 4e** — the near machine of a split that exists for memory
reasons has no room for a shadow, so 4b (which works with ANY drafter) and a
small drafter from the shard system now come first.

**KV refresh — measured: large for a small model, small for a 7B.** The tail
computes the far layers' exact K/V for every confirmed token anyway; sent back
(~28 KB/token for a 7B), the shadow attends over an EXACT history and
approximates only the token being drafted. Probe
(`split::tests::kv_refresh::kv_refresh_probe`, our own engine, CPU, the far
layers' cache replaced before every token by an independent copy of the
target's), far half at Q3_K_S, 400 positions over 4 prompts each:

| target | own cache | refreshed | misses |
|---|---|---|---|
| Qwen3-1.7B Q8 | 86.5% | 91.75% | −39% |
| Qwen2.5-Coder-7B Q4_K_M | 93.75% | 94.25% | −8% |

⚠ **Corrected the same day:** the 1.7B figure was first written up alone, as if it
carried to the model sizes a split is for. It does not: the small model's errors
accumulate through its history, the 7B's are mostly in the drafted token's own
path. The 7B's own-cache 93.75% also independently confirms llama.cpp's 93.8% for
the same pair. KV refresh is a minor lever at 7B; not a priority.

## The loop's round trip, not the user's (2026-09-28)

**Where "40-60 tok/s over a 500 ms split" came from, and what became of it.**
On 2026-09-27 the shadow measurements above were projected, in conversation, to
~36-39 tok/s TH↔BE (~46-51 at 60 tok/s drafting) for a continuous stream drafted
by the 3-bit far-half shadow with a second draft line where it was unsure. The
same day that was corrected to ~22-29 (the table in Phase 4: the second choice
avoids a stall only if it was drafted ONWARD). Then the shadow was parked for
memory (§ "The shadow's memory"), and what shipped in v0.3.212 is the first rung:
a 0.5B drafter, round-based. It measured 4.6-5.5 tok/s, and **that is its
ceiling, not a shortfall**: at α ≈ 0.7 a chain of guesses yields at most
1/(1−α) ≈ 3.3 tokens a round, ≈ 5.6 tok/s at a 0.6 s loop; the optimal guess
count grows only logarithmically with delay (UCB-SpecStop 2606.20591, the same
0.5B → 7B pair). No published lossless method reports more than ~10 tok/s, or
5×, with ≥ 100 ms inside the per-token loop (survey 2026-09-28, sources below).

**What 40-60 tok/s across a 500 ms loop needs.** Time per token ≈ 1/v + m·S.
At S ≈ 0.6 s and v ≈ 60, m must be ≲ 1.4% — 24-36 tokens a round, α ≈ 0.96-0.97.
That is Q6_K-to-Q8_0 agreement (llama.cpp, Llama-3-8B, same top token vs F16:
Q8_0 97.7%, Q6_K 96.0%, Q4_K_M 91.9%, Q2_K 71.1%; BF16 vs F16 of the SAME weights
99.74%), i.e. a near-complete copy of the model on the drafting machine — the
memory a split exists because nobody has. Low-rank error compensation (EoRA
2410.21271) adds 2-7% memory and reports no agreement near that; trained heads
(EAGLE, Kangaroo, mid-network drafters) top out at 0.65-0.88 acceptance.
**So across a loop that long, 40-60 tok/s is out of reach for an exact method
with the memory consumer machines have.** The honest ranges stand: ~5-8 with a
small drafter (reached), ~15-30 with a shadow where one fits.

**But the long link does not have to be IN the loop.** Only the round trip
between the machines that hold consecutive layers must be paid per token; the
requester's distance could be paid once, in time to the first token, with tokens
streamed back. **It is paid per token today** (checked 2026-09-28 — the open
unknown `regional_pipelines.md` Stage 5 carried): the tail answers the
coordinator, the coordinator starts the next token at segment 0
(`pipeline/distributed.rs:896`), `PipelineExecutor` is only ever built by the
requester's router (`router/distributed_exec.rs:927`), and delegation exists
only as a whole-model hand-off (`scheduler::delegation_target`, and only when this
node would run on its processor). A Thailand request split among European peers
20 ms apart therefore runs at ~0.5 s a token, where the same peers coordinated
from Europe run at the 25-60 ms rows of the projection table above (GPU peers):
9-13 tok/s plain, **26-38 with the drafter chain this release already ships**,
~38-50 with draft trees — the 40-60 range, reached by moving the loop rather
than by predicting past it. Projections, not measurements.
Parallax (2509.26182: last peer → first peer, "Petals puts the LM head on the
client") and Prime Intellect's ring do exactly this; Petals, Helix and we do
not. That is FUTURE_WORK #143, a delegated coordinator: the whole request to the
head holder, which plans among its near peers, runs the loop and the speculation
(`dsd.rs`, the n-gram loop and `engine_drafter` are `PipelineExecutor` methods
and move with it), and streams tokens back.

**What bounds it:** the fleet. On 2026-09-28 this node (Thailand) had five peers
at 210-290 ms and one LAN peer; four of the five are two operators' machines in
Belgium and Italy (one RTX 4050 laptop, the rest processors). A delegated loop
among them is fast only for the parts that run on a card, and placement still
converges on whole models per node (`regional_pipelines.md` Stage 0/3).

## Draft on your own card, check on everyone else's (2026-09-28)

**The one hard constraint, and what it leaves.** A token's first layer needs the
token before it, which exists only once the last layer has sampled it. So an
exact split decodes by "the near side guesses, the far side checks", and tokens
per trip are capped by how often the guess is right — no bandwidth moves that.
What the constraint does NOT forbid: the far machines checking a CONTINUOUS
stream of guesses, each stage working on different positions of the same reply
at once. Then the long hop is on the critical path only when the full model
disagrees, and time per token ≈ 1/v + m·S — the near machine's own speed plus
misses × the loop. Everything turns on m, the miss rate of a predictor the near
machine can actually hold.

**Measured** (`~/swarmllm-ref/spec/lopsided.py`, `lopsided_qwen7b.json`):
Qwen2.5-Coder-7B Q4_K_M as the target, llama.cpp as the reference, 709
positions of the target's own greedy replies to six prompts (prose, code, a code
edit, Q&A, advice, a story). Shadows REQUANTIZED from the Q4 file, so every
shadow figure is pessimistic. "≥ τ of best": the full model rates the guess at
least τ × its own top probability.

| predictor | GiB (target 4.36) | first guess = target's | top 2 | ≥ 0.5 of best | ≥ 0.3 of best |
|---|---|---|---|---|---|
| qwen2.5-coder-0.5b (today's drafter) | 0.63 | 72.6% | 85.3% | 77.7% | 81.8% |
| **whole model at Q3_K_S** | 3.25 | **94.2%** | **98.6%** | 98.4% | 99.7% |
| whole model at Q2_K | 2.81 | 85.6% | 95.9% | 92.5% | 95.6% |
| middle 7 of 28 layers at Q2_K | 4.01 | 94.1% | 99.0% | 99.3% | 99.9% |
| middle 14 of 28 at Q2_K | 3.68 | 92.0% | 98.0% | 97.6% | 99.0% |
| last 4 of 28 at Q2_K | 4.12 | 92.4% | 97.9% | 98.0% | 99.3% |
| last 7 of 28 at Q2_K | 3.98 | 88.9% | 97.0% | 95.8% | 98.0% |
| last 14 of 28 at Q2_K | 3.63 | 87.3% | 96.9% | 93.7% | 97.0% |
| middle 7 of 28 at Q3_K_S | 4.11 | 96.8% | 99.6% | 99.9% | 100% |
| middle 14 of 28 at Q3_K_S | 3.87 | 96.5% | 99.3% | 99.4% | 99.9% |
| last 7 of 28 at Q3_K_S | 4.08 | 95.5% | 99.3% | 99.7% | 100% |

What it says:

- **The drafter for a model is the same model at fewer bits.** A plain Q3_K_S
  file agrees 94.2% on the first guess against the 0.5B drafter's 72.6%, has the
  same tokenizer by construction, and exists on HuggingFace for every popular
  model — the shard system can fetch it like any other model. It is 25-36%
  smaller than the Q4 target, which is exactly the "a little too big for my card"
  gap (a 14B Q4 is 9 GB; its Q2_K is 5.8).
- **WHERE you approximate matters more than how much.** 2-bit on the middle 7
  layers (94.1%) beats 2-bit on only the last 4 (92.4%); the last layers are the
  quantization-sensitive ones. The far machines should hold the middle — the
  boomerang layout the prompt-privacy mode already plans.
- **Almost every disagreement is a near-tie.** Where a 3-bit copy differs, the
  full model rates the copy's token at least half as likely as its own pick
  98.4-99.9% of the time. The misses that cost a trip are the target's own
  coin-flips.
- **Exact has a floor of about 1%.** Run with the target drafting for itself
  (`relaxed_gen.py`, `relaxed_self-control.json`), llama.cpp's batched check
  disagreed with its own one-token-at-a-time decoding at 0.97 of every 100
  positions — batch shape moves the arithmetic enough to flip a near-tie. The two
  replies with no correction were byte-identical to the reference, the four with
  one diverged there. Byte-identity was never the right test for a split
  (CLAUDE.md), and this is the number that says how far "exact" can go.

**Projected** (not measured) with t = 1/v + m·S, v = this RTX 3070's measured
local speed on our engine, S = the TH↔BE loop plus far compute:

| 7B split TH↔BE (local 44 tok/s; measured today 4.6-5.5) | 0.5B drafter | whole Q3 copy | middle-7 Q3 copy |
|---|---|---|---|
| exact | 5.8 | 18.3 | 24.8 |
| exact, a second guess drafted where the copy is unsure (top 2) | 9.7 | 32.9 | 40.1 |
| relaxed, near-best accepted (≥ 0.3 of best) | 8.1 | 41.0 | 43.5 |

| 14B on an 8 GB card (today, card + processor: 3.35 tok/s) | whole Q3 copy | whole Q2 copy |
|---|---|---|
| checked by this node's own processor (~360 ms a miss) | 16-24 | 11-18 |
| checked by the swarm (~550 ms a miss) | 14-24 | 8-16 |

**The design this points to.** The near machine (the requester's card) holds a
low-bit copy of the whole model — or, in the boomerang layout, the exact first and
last layers and a low-bit middle — and generates continuously at its own speed.
The full-precision model checks the stream wherever it lives: this node's own
RAM for a model too big for its card (no network at all), or the swarm's peers
for a split. A miss rewinds the near machine to the corrected token and discards
what was in flight after it. For a GPU user this outranks moving the coordinator
(FUTURE_WORK #143): a 3070 drafting at 44 tok/s with a 500 ms loop behind it beats
handing the loop to processors 20 ms from each other. #143 still matters for users
with no card. And the checker can be MORE precise than anything the user could run
(Q8 split across peers) — faster than local and better than local at once, once
local decode is bound by bytes rather than submissions (a Q2 copy reads ~40% fewer
bytes per token than Q4; today's engine cannot cash that, `local_decode_submissions.md`).

**What is missing, in order:** (1) the continuous stream — the same Q3 drafter in
today's round-based DSD projects only ~13 tok/s TH↔BE, because on this engine a
7B-size drafter costs 23 ms per guess and ten guesses take 230 ms before each check
even leaves (measured: the 0.5B costs 17-24 ms a guess, the same as the 7B's 22.9 —
cost follows layer count); (2) the drafter chosen as "the same model at fewer
bits" (`engine_drafter`'s auto rule wants a quarter of the target's size and would
never pick it); (3) a second guess where the copy is unsure; (4) verification on
this node's own processor, the no-network case; (5) relaxed acceptance as an
opt-in, once its quality is measured (below). FUTURE_WORK #144.

## What the literature says (survey 2026-09-27)

The survey found no method that makes a WAN split fast without speculation. It
found several that confirm the pieces above, and each names a detail to keep:

- **DSD** (arXiv 2511.11733, what `dsd.rs` implements): its own speedup formula,
  S = [t₀+(N−1)t₁] / [t₀/ρ+(N−1)t₁/k], tends to k (tokens per round) as link
  latency t₁ grows. The paper's "3t₀ < t₁ < 10t₀" is the range it MEASURED,
  not a limit. That is the correction to `regional_pipelines.md` Stage 4 above.
- **SpecExec** (arXiv 2406.02532) is the verification rule: the target samples
  at each tree node and follows the branch if the sample is a child. It is
  exact at any temperature and needs no draft probabilities, so drafter,
  n-gram and suffix-tree candidates can share one tree. It is slightly worse
  than min(1, p/q) for SAMPLED drafts and identical for deterministic ones,
  which ours are. Under offloading, the regime most like a WAN, it reached
  12-21 tokens per round (7B draft → 70B, budgets 128-2048).
- **Sequoia** (arXiv 2402.12374) is the tree shape and the sizing rule. It
  maximises G(n,d) / (t(n) + d·c), where t(n) here must INCLUDE the tree's
  transfer. `spec_projection.py` is that rule with our measured parts. Its trees keep
  growing roughly logarithmically in size; chains cap at 1/(1−α).
- **SpecEdge** (arXiv 2505.17052) is the closest real-WAN evidence: draft on the
  edge, target on a server, tree 32, 3.4-4.6 tokens per verify at 14-65 ms RTT.
  **Splitting the target's layers across the same link was 2.73× slower than
  drafting across it.** Its proactive drafting, which extends the likely path
  while a verify is in flight, added 13% tokens per verification.
- **BloomBee** (arXiv 2604.21072, Petals lineage) is the bandwidth warning: tree
  speculation fell BELOW the non-speculative baseline at its lowest link
  bandwidth. It gives a break-even formula (Eq. 8), and prunes tree nodes at the
  stage boundary with a small trained head (60% fewer nodes sent, 96% of
  acceptance kept).
- **EAGLE-3** heads now load in llama.cpp as GGUF (chain drafting, 2.1-2.3×),
  and **multi-token-prediction** GGUFs exist for Qwen 3.5/3.6 and Gemma 4.
  These are drafters that need no separate small model. They come AFTER Phase
  1, for families with no small sibling. `qwen35-support` is the natural first
  user.
- **SuffixDecoding** (arXiv 2411.04975): suffix trees over the prompt and
  earlier replies. Up to 5.3× on agent workloads. It is a second candidate
  source for the same tree and the right successor to the n-gram lookup.
- **Pipelined speculation** (PipeInfer, FlowSpec, PipeDec, PEARL) keeps busy
  stages busy when verification is expensive (a LAN, many stages). With one
  trip dominating, several speculative rounds in flight are built from the same
  stale knowledge, so they do not raise tokens per trip. They hide drafting
  time, worth perhaps 10-25%, and that is Phase 2's last step, not a phase.
- **Exiting after the head's half without training does not work:** LayerSkip's
  own table has plain Llama2-7B exiting at layer 16 of 32 scoring 0% on
  HumanEval and GSM8K. Every early exit that works trains something.

**What bounds the whole approach:** at this link, matching the 4050's own 11
tok/s needs ~5 tokens per trip, and matching the 3070's 44 needs ~20. Speculation
brings a WAN split up to the slow end of one card's speed, not the fast end. For
a model that fits one machine, not splitting remains the larger win, and the
router already prefers it.

## Open unknowns

- Whether TCP/QUIC slow start after an idle gap throttles a 0.5 MB tree burst
  on a 300 ms path. That is the survey's inference, unverified here. The
  persistent stream keeps the connection warm; check it before sizing a tree
  from a cold measurement.
- The TH↔BE link's usable bandwidth. It is modelled at 20 and 50 Mbit/s and
  has not been measured. It decides the tree size.
- Our own drafter's step time for a 0.5B on the card. Local decode is
  submission-bound (`local_decode_submissions.md`), so 24 small layers may cost
  close to what 28 large ones do. Measured it moves the TH↔BE projection by
  about 10%, but on a short link it decides whether a tree pays.
- Acceptance at temperature 0.7 with a real sampler (top-p, repetition
  penalty). `spec_coverage.py` models plain temperature 0.7 only: the drafter's first
  guess covers 70.2% / 79.6% of the target's probability, against 73.5% / 81.8%
  greedy (Qwen / Llama pair).
- The 4050's figures on 2026-09-27 were processor-class (a pinned card, #125 on
  its side). Re-measure the tail's per-node cost on a healthy card.
