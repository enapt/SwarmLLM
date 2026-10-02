# Parallelize the reply, not the token

**Written 2026-09-28**, after measuring why a model split Thailand↔Belgium decodes
at 4.6-5.5 tok/s where one card does 44 (`split_speculation.md`), and after the
user set the constraint this document works inside:

> we dont want people to be running full models, especially since it probably
> wont be possible with larger models and our demographic. you need to find a
> way to parallelize it all

So **no machine holds the whole model — not even a low-bit copy.** Each holds
its layers, plus small things (a draft head, a 0.5B model, caches). The
"same model at fewer bits" drafter (FUTURE_WORK #144, 89-94% agreement) is
therefore at most an opt-in for users who can, never the plan.

## 1. What cannot be parallel, and what that leaves

A token's first layer needs the token before it, which exists only after the
last layer has sampled it. Every token therefore passes through every machine in
order: one **ring pass** per token, P = the sum of the hops plus each stage's
compute (~0.35-0.6 s TH↔BE, measured). No bandwidth, placement or engine change
removes that dependency; only a GUESS can cross it, and a guess is worth what its
acceptance says.

But a ring pass is almost free to widen. A stage visited by one token spends
~0.5 ms a layer on a card, and our decode is bound by GPU SUBMISSIONS, not bytes
(`local_decode_submissions.md`) — so it processes 1 row or 64 rows of a batch in
nearly the same time. Over a 500 ms pass every machine in a split is idle ~99% of
the time. **The unit to parallelize is the reply: make each pass carry many rows
that all belong to it.** Three sources, all compatible with "nobody holds the
model":

1. **More tokens per stream per pass** — a guess made where one stage already
   has what it needs (§2).
2. **More streams per reply** — several streams of the same reply riding the same
   pass (§3).
3. **The prompt pass**, which is parallel over positions already (§4).

And separately, **a shorter pass** (§5): P itself is set by where the layers are
and whether the requester sits inside the loop — today it always does
(FUTURE_WORK #143).

## 2. More tokens per stream: drafting that lives on ONE stage

| drafter | where it runs | tokens per ring pass | source |
|---|---|---|---|
| none | — | 1 | — |
| 0.5B same-family model, chain | head (coordinator) | 2.8 replayed; **3.5 measured** by whole-reply speculation (1.9 on a story, 10 on code) | `spec_coverage.py`; `jacobi.py` redraft mode (2026-09-28) |
| 0.5B, best-first tree 16-64 nodes | head | 4.1-5.2 replayed | `spec_best_tree.py` |
| **native multi-token-prediction (MTP) head** | **last stage** | **2.6-3.3** (llama.cpp: Qwen3.6-27B, Gemma-4) | llama.cpp PRs #22673, #25589; survey 2026-09-28 |
| EAGLE-3 head, trees | needs 3 spread layers' states | 5.9-6.7 (paper, T=0); 2.5-3.3× in llama.cpp | 2503.01840, llama.cpp #18039 |

**MTP is the drafter a split was waiting for.** Qwen3.5/3.6, Qwen3-Next, GLM-4.5+,
DeepSeek-V3+, MiMo ship a small head INSIDE the model that proposes the next
token(s) from the final hidden state, the token embedding and the output head —
all of which sit on the LAST stage already. That stage samples, drafts k more
locally, and returns [verdict + next drafts] in the message it sends anyway; the
next pass checks k+1. No second model, nothing on the requester, supported and
measured in llama.cpp. The fleet already carries `qwen3.5-9b-q4-k-m`; our Qwen 3.5
support is on local branch `qwen35-support` (#117). (A GGUF converted before
mid-May 2026 may lack the MTP tensors — check the file, gotcha #715.)

Ruled out as drafters here: **Jacobi / lookahead decoding** (drafter-free, exact) —
~1.2 tokens a pass on chat in a measured 80-100 ms split (arXiv 2602.16760), and
~100 extra rows a pass multiply every hop's payload; the whole-draft Jacobi variant
was measured here too (§7). **Early exit at the split point** without training
scores near zero (LayerSkip's own table).

## 3. More streams per reply

**Shared-cache workers (Hogwild!, arXiv 2504.06261, training-free).** Up to ~4
instances of the same model write ONE answer together, each attending to what the
others have written token by token: a common cache block (prompt + finished
paragraphs) plus one block per worker, keys stored at block-local RoPE positions,
each query rotated once per block. **Its own Appendix B says the layout splits by
layers: each device stores the cache blocks for its local layers** — which is
exactly what a stage of our split holds. Nothing crosses the network that does not
already: every worker's token rides the same pass, and the stages see each other's
tokens through their own caches. Measured by the authors on QwQ-32B: 20 / 36 / 69
tok/s aggregate for 1 / 2 / 4 workers at 50-58 ms a pass, accuracy per sequential
pass above the single worker on math/code reasoning. Works on reasoning models of
~8B and up (1.7-4B "get distracted"). → n tokens a pass for n ≤ 4, on the tasks
where replies are longest (thinking).

**Parallel sections (outline, then expand; SoT 2307.15337, Jupiter INFOCOM'25).**
Measured here on the live node (`~/swarmllm-ref/spec/parallel_reply.py`,
llama-3.1-8B, 16 prompts): where the answer is list-shaped (how-to, plans,
troubleshooting, code) the critical path fell 1.8-2.1×; on a story, an email or a
one-line fact it grew — and the 8B model split EVERYTHING into 5-6 parts when asked
to decline unsuitable questions, so the router must be a classifier, not a prompt
(SoT-R's finding too). Jupiter ran exactly this through a pipeline of Jetsons
(100 Mbit-1 Gbit): 3.6-3.9× over sequential with Medusa heads, switched off for
math and code. Applies to ~27-46% of real chat prompts. Quality: see §7.

**Agent sub-calls** are already separate streams (parallel tool calls, subagents);
they only need to ride the same passes.

## 4. The prompt pass is already parallel

Every prompt token is known, so the head reads chunk k+1 while chunk k crosses and
the tail reads it: time ≈ the slowest stage or the link, not their sum
(`split_speculation.md` Phase 3). For agent traffic (~100:1 input to output) this is
most of the work, and across a WAN it is bandwidth-bound, not latency-bound: at
~50 Mbit/s a 7B's Q8 activations (~3.5 KB a token) cross in ~0.56 ms a token, under
a 3070's ~1.25 ms to read it alone — a two-GPU split reads a long prompt faster than
either GPU. Below ~25 Mbit/s the link binds.

## 5. A shorter pass

- **Take the requester out of the loop** (FUTURE_WORK #143): today the coordinator
  starts every token and the tail answers it, so a Thailand request split among
  European peers 20 ms apart pays Thailand↔Europe per token.
- **Rings of close machines** (`regional_pipelines.md` Stage 3): placement that
  completes a model within a region, contiguous segments, no 4-layer slivers.

## 6. What it multiplies to (projection)

Effective tok/s for ONE reply ≈ tokens a pass per stream × streams × 1/P:

| ring | P | MTP alone | MTP + 3 streams |
|---|---|---|---|
| today's TH↔BE | ~0.5 s | ~6 | ~18 |
| same continent | ~0.15 s | ~20 | ~60 |
| same city / LAN-ish | ~0.06 s | ~50 | limited by the stages' compute |

Projections from measured parts, not measurements. The streams column applies only
where a reply decomposes (reasoning with workers, list-shaped answers, agents).

## 7. Measured and rejected, 2026-09-28

- **Weight streaming** ("a full-model stream to the requester"): every decode step
  needs all the weights (4.4 GB for a 7B) against a WAN of ~6 MB/s; even applying
  one weight stream to a whole draft, a reply needs several passes. Tens of GB a
  reply. Rejected on arithmetic.
- **Whole-reply repair (Jacobi seeded by the drafter)** — the full model reads the
  drafter's whole reply in one pass (the ring reads a known text in parallel) and
  repairs disagreements (`~/swarmllm-ref/spec/jacobi.py`, Qwen2.5-Coder-7B + its 0.5B,
  120-token replies, llama.cpp):
  - repair EVERY disagreement at once (Jacobi): **1.07 tokens a pass** — each repair
    invalidates the full model's predictions after it, which were made on the old
    prefix; seeding with a good draft does not change that (the literature's ~1.05);
  - repair the next 8 that way and re-draft the rest: 1.07 — the same, for the same
    reason;
  - settle up to the first disagreement and RE-DRAFT the rest with the small model:
    **3.50 a pass** (1.9 on a story, 10.0 on code), output identical to the full
    model's greedy reply where the control could say. That is ordinary speculation
    with the whole reply as the draft — the drafter's ceiling (§2), not a new regime.
- **Prompt-routed parallel sections at 8B**: the model does not decline unsuitable
  questions (§3). A same-size LLM judge (qwen2.5-coder-7B, both orders) returned a
  position-biased tie on every pair judged — no quality verdict; a proper check needs
  a stronger judge or task scores. (The judge also ran on a PEER when it did not fit
  beside the generator — gotcha #753.) Wall time was not measured cleanly: a CPU job
  beside it starved the one thread that feeds the card.

## 8. Engine prerequisites, in order

1. ~~Batch streams at different positions~~ — **already done** (2026-08-09): decode
   steps at different positions batch (`forward_batch`, RoPE per row); the live node
   logged 5,459 of 5,459 multi-request calls batched during this experiment. What a
   split still needs is for a STAGE to batch forwards from different streams that
   arrive as separate network messages into one visit — today each `LayerForward`
   is its own forward.
2. ✅ **The continuous stream** — built 2026-09-30, on by default since v0.3.216
   (`pipeline::dsd_stream`, chunks keyed by `stream_seq`; `SWARMLLM_SPEC_STREAM=0`
   switches it off; `split_speculation.md` § 4b).
3. **MTP at the last stage** — Qwen 3.5 first (#117 branch), then GLM-4.5+.
4. **Multi-block attention** (per-block query rotation) for shared-cache workers.
5. **The delegated coordinator** (#143) and regional placement (Stage 3).
