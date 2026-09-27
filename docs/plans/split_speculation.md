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

**Status 2026-09-27:** items 1 and 2 are built and measured (for v0.3.211):
`split_rig.sh repeat` on llama-3.2-3b, every result from the tail 43-59 bytes
instead of 513 KB-2.5 MB, replies 119/121 at llama.cpp's first choice, and
byte-identical to a run whose v0.3.209 tail was sent the old request. Items 3
and 4 (a drafter from the shard system, γ from the trip) remain.

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
4. **γ from the trip, not a constant.** `GammaController` already adapts to
   acceptance; add the trip time so a long link drafts deeper.

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

**KV refresh — measured, 39% fewer misses.** The tail computes the far layers'
exact K/V for every confirmed token anyway; sent back (~28 KB/token for a 7B),
the shadow attends over an EXACT history and approximates only the token being
drafted, so its errors stop accumulating across the context. Probe
(`split::tests::kv_refresh::kv_refresh_probe`, our own engine, CPU, the far
layers' cache replaced before every token by an independent copy of the
target's): Qwen3-1.7B Q8 target, far half at Q3_K_S, 400 positions over 4
prompts — **own cache 86.5% → refreshed 91.75%** top-1 agreement (misses 13.5%
→ 8.25%). Not yet run on the 7B (three 7B copies exceeded the build slice; the
probe now shares one shadow between both arms). ~9 Mbit/s at 40 tok/s — cheap
beside the hidden states already crossing.

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
