# Small slices, fast replies — the plan (2026-11-20)

**The aim, in the owner's words** (2026-11-20): *"we want to reduce the amount of processing any node should do,
and distribute the load over the network. nobody has datacentres, or wants their gpu locked up by an entire
model."* And (2026-11-13): *"like how torrents get much faster when theres more seeds."* A whole model on one
card is a reference in a measurement here, never a design; no design may need anyone to hold the whole model.

This plan combines a research run of 2026-11-20 (a speed model calibrated on our own figures, prior art to
2026-11, and ~50 earlier designs reconciled from `faster_than_local.md`, `split_speculation.md`,
`regional_pipelines.md`, `wan_parallel.md`, `direct_peer_chaining.md`, `large_model_splits.md`,
`pipelined_prompt_pass.md`, `split_prompt_cache.md`, the round logs and FUTURE_WORK) with the same day's
measurements. Each step ends on a measurement that can kill it.

## 1. Where it stands (measured 2026-11-20 unless dated)

Qwen3.8-27B (hybrid: recurrent and attention layers, its file's own guess head), one reply at a time, 160 tokens,
from the Thailand node (8 GB card), main `4f693a127`:

| shape | cards | reply speed | note |
|---|---|---|---|
| Australia holds it whole (REFERENCE) | 12 GB, part on its processor | prose 5.3, code 5.8 tok/s | the 16.5 GB model does not fit its card |
| here → Australia → here | 8 + 12 GB | prose 5.3, code 5.7-7.2 | the guess head beside the first layers; it needs the last layer's output, so the chain starts and finishes here |
| here → US laptop | 8 + 16 GB | prose 3.05, code 3.08 | ends on the peer: no guesser, one token a 272 ms round trip |
| here → Australia → US laptop | 8 + 12 + 16 GB | prose 2.78, code 2.23 | three slices, ends on the peer: no guesser, ~360-450 ms a token |
| **two conversations at once** through those three slices | 8 + 12 + 16 GB | **2.35 and 2.44 each** (4.8 together) | each as fast as one alone: the cards idle most of a round |

- **Qwen3-14B** split Thailand ↔ Australia with a small drafter's tree flowing from the far end: prose
  17.8-20.5, code 40-43 tok/s against 11.0-11.3 / 14.1-14.5 on one card (#411). The one case that beats local.
- **Llama 3.3 70B** over four computers (here, Australia, a processor-only node beside us, the US laptop):
  1.5-3.0 tok/s. Bound by the layers on processors: ~42 GB of model against ~26 GB of free card.
- **Qwen3.6-35B-A3B**: Australia whole 34-37 tok/s; forced splits 7.8-16.

What it says: a split that guesses matched a whole copy on a 12 GB card (5.3-7.2 against 5.3-5.8); the same split
without a guesser, ending on a peer, ran at half that, and each further slice without one cost a further share of
the loop (two cards 3.05-3.08, three 2.23-2.78). **For the hybrid families, WHERE the guesser sits is the lever.**
For dense models it is the flow, and how far it stretches past two computers. And **the cards' idle time is
already free capacity**: a second conversation through the same three slices ran as fast as the first alone —
the throughput half of the aim works today for plain chains; step 3 is what keeps it working once they guess.

## 2. The speed model

- **Flow** (forwards leave continuously; dense models with a drafter): `time a token = τ/e0 + m·P`. τ is the
  SLOWEST stage's forward (the head's including its drafter call), e0 the tokens a forward advances, m the misses
  a token, P the loop. **Slices do not add; only the slowest sets τ, and the loop only multiplies misses.** Fitted
  on the 14B pair at 130 ms: e0 1.03 prose / 1.93 code, m 0.09 / 0.0125.
- **Rounds** (one check in flight; every recurrent model today): `rate = E(γ,α) / (P + k·Σc + draft)` with
  `E = (1 − α^(γ+1)) / (1 − α)` (Leviathan et al., 2211.17192) and k the check's cost in plain steps (1.1-1.3).
- **Cut finer inside a region, not across oceans**: each extra hop costs m·h a token (prose: 1.8 ms at h = 20 ms,
  5.9 at 65). An all-card 70B in a short ring reads ~9-13 prose / 24-28 code tok/s (HYPOTHESIS, borrowed e0, m).
- **Known misfit**: 25-50% optimistic on long loops (308 ms: predicts 14.8 / 39.9, measured 11.7 / 27). Not
  separated: the live-row budget (`live_most`), the uplink (~120 KB a forward), misses rising with depth.
- **Ceiling**: once every slice is on a card and the ring is short, one reply stops at ~e0/τ (τ ≥ a forward's
  ~17.5 ms fixed cost). Past that, more nodes mean more conversations, not more tok/s.

## 3. Tried and dropped — do not redo

Measured and dropped: a drafter taught online from the target's verdicts (bias 0.920 → 0.915); re-ranking the
drafter by the target's middle state (0.849 vs 0.847); recycling dead branches' draws (+3.7 points); bigger
sends (levels, rows, live budget 256/448: lost to the default three sittings); overlapping the draft with the
head's layers on one card (#462: 40 → 35-37 ms, not 28); stream spares (#381); a low-bit SHADOW copy of far
layers (needs ~76% of them, and breaks the no-whole-model rule); early exit / layer skip (no training-free
acceptance near a drafter's); Jacobi and lookahead (1.07 tokens a pass); tensor or sequence parallelism over a
WAN (56 exchanges a token); hedged dispatch (#94, removed in v0.3.204 — the hedge ran on a machine that never saw
the conversation); thin slivers on an extra peer (one never returned in 580 s).

## 4. The plan, in order

### Step 1 — zero-code checks first (simulators, no node time)

Each decides whether a later step is built.

- **1a** `examples/tree_schedule_sim.py flow()` with K stages of their own costs and a ring P, at the measured
  prices. Kill step 4 if three balanced stages at P = 290 ms read under 8 tok/s on prose.
- **1b** The 11-19 field logs' kept counts (the block drafter on Qwen3.6-35B-A3B) replayed through
  `examples/stream_width_sim.py` as chunks in flight. Kill step 5 if code reads under 1.3×.
- **1c** In the flow sim, the head's cost as its layers alone plus half a loop on each rebase (the drafter on
  another card of the ring). Build it only at ≥ 1.2× at 130 ms.
- **1d** The long-loop misfit, in the same sim: vary `live_most` and the uplink separately at 308 ms and see which
  closes the gap to 11.7 / 27.

### Step 2 — BUILD: the guesser runs where the model's last layers are (Qwen 3.5 / 3.6 / 3.8 first)

- **Why first**: the most popular family (Qwen3.8-27B, Qwen3.6-35B-A3B) takes no separate drafter and no flow;
  today its guess head and block drafter attach only beside the FIRST segment (`pipeline::split_guess_head_here`,
  `block_draft::a_guesser_may_answer`), so a chain must start and finish on the asking computer to guess at all.
  Every chain that ends on a peer — the shape a spread swarm produces — runs one token a round trip (here → US
  laptop: 3.05-3.08 tok/s). It also frees the asker's small card from the tail layer and a second head.
- **Mechanism**: the tail samples, drafts the next γ with the file's head (it holds the last layer's output, the
  norm and the head), and returns [token, guesses, kept count] — ids only — with its reply. Each segment's rewind
  already travels with the next forward (`pending_truncate` → `truncate_kv_to`), the accounting vLLM had to add
  as a broadcast.
- **Expected**: ~1.4-1.8× on chains that end on a peer (here → US laptop: 3.1 → ~4.3-5.5). INFERRED from the two
  measured shapes; the loop is unchanged.
- **Protocol**: a new feature bit; a `LayerResult` trailer of guess ids, gated at the SENDER; an older tail
  returns none and the chain runs as today.
- **Recurrent risk**: counts known only at the last stage (vLLM's bug class) — here the rewind is explicit per
  forward, ≤ 32 rows, `RECURRENT_REWIND` on every segment.
- **Experiment first**: Qwen3.5-9B on the two-node rig at 130 ms, A [0,17) ↔ B [17,32) + head, B guessing,
  against A-B-A with the head at A. Kill if kept-a-round at the tail is more than 10% under the head's. Then the
  field: the 27B here → US laptop and here → Australia, both ending on the peer.
- **Prior art**: vLLM PR #44698 (MTP on the last pipeline rank, sampled + draft tokens + accepted counts broadcast
  to the other ranks, recurrent state rolled back by that count; closed unmerged 2026-10-14, nothing to port);
  SGLang's "PP tail-draft"; llama.cpp #27781 (an MTP head sharing a cache cost 8 points of prose acceptance —
  check acceptance against a known answer after moving it).

### Step 3 — the throughput half: one stage carries several conversations' forwards as one

- **Mechanism** (the 11-13 "shared buses", FUTURE_WORK #148): a card stage merges waiting forwards of DIFFERENT
  requests into one batch, each with its own cache base, positions and rows; never holds a forward to wait.
- **Expected**: ~3× the conversations a card stage at full speed (17.5 + 3.6 ms a user: six inside a 41 ms pace,
  against two served one after another). INFERRED; outside: two users 1.79× on a two-stage pipeline (2608.19147),
  Petals keeps 128.7 of 155.1 tok/s aggregate at 100 ms (2312.08361).
- **Gap**: `process_pool::forward_is_schedulable` and `model_worker::batch_eligible` refuse every guessing
  forward (draft tokens, tree parents, `stream_seq`, `truncate_kv_to`, checks). Recurrent layers need per-request
  states in one batch — later (llama.cpp #31656: DFlash 2 on a hybrid aborts at `--parallel 8`).
- **Experiment first**: `prefill_bench`, 2 × 23 tree rows from two caches in one forward against two forwards on
  the idle card. Prediction 25-27 ms against 42. Build only at ≥ 1.6×; dense models first.

### Step 4 — the flow through three or more card slices (dense models)

- Generalise the one beat-local case: every slice on a card, sized so each stage's forward matches the head's,
  the ring ordered for the shortest loop, the far end drawing, forwards chained node to node.
- **Gaps**: at most 8 forwards out through a chain (`forward_streams::MAX_STREAM_WINDOW`, binds past ~330 ms);
  the planner fills the head's card without counting its drafting (#411, est. +12%); `near_chain` stops at 3 card
  segments and 30 ms; nothing balances stages by measured forward time.
- **Real run** needs a third card holding a dense model with a drafter — none today (the US laptop holds no
  14B). Until then, step 1a decides it. Prediction for the 14B in thirds: prose 12-15 (halves: 17.8-20.5); code
  within 10% of the halves, and if code drops more than 20% the window or the uplink binds, not the misses.

### Step 5 — a block's chunks in flight on a recurrent model (code), only if 1b passes

- Send the block drafter's guesses as 2-4 chunks back to back so the far slice computes while the next chunk is
  on the wire (the chain stream's shape). ~1.5-2× on code, nothing on prose.
- **The hard part**: a recurrent layer keeps ONE step's rows to replay; chunks need the kept rows to span every
  forward since the last commit, or a state per forward on the device (vLLM keeps a state column per draft
  position; llama.cpp's host-side checkpoints cost 201 ms a token). Exactness asserted per row against the reply
  read one token at a time (llama.cpp #31472: DFlash on a hybrid drifted from greedy).

### Step 6 — slice size and ring order as the planner's job, and the owner's share of the card

- A node's slice = min(what its owner lets it lend, the layers that fit the ring's pace, weights + caches for
  the conversations it serves). Order the ring by the round-trip matrix (exact by brute force to K = 8); cross an
  ocean at most once out and once back. Prior art: Petals (2312.08361: the bottleneck segment first, move only
  for ≥ 20%), Parallax (2509.26182), BloomBee (2604.21072), 2512.21884 (session memory binds before compute).
- **The owner's setting** "lend at most X of my card" does not exist; it is the user-facing half of "nobody wants
  their GPU locked up" (non-technical wording, all 21 languages).
- `examples/placement_sim.py --scenario metro` with step 2's rate formula as the objective, before any planner
  change.

### Not planned

The drafter moved off the head (only if 1c passes); trees on recurrent prose continue as #437's residual
(capture the tree check); fewer misses is mostly spent (above); hedged or replicated stages (#94's lesson
first); parallel streams of one reply (no recurrent form, no completed measurement).

## 5. Stale statements this plan supersedes

`regional_pipelines.md`'s header (stage 3 unbuilt, stage 2 unconsumed); `wan_parallel.md` § 2 (the guess head
at the last stage — it shipped at the requester; step 2 reopens it) and § 4 / § 8 (the pipelined prompt pass
is built, #171); `split_speculation.md`'s shadow stages (dropped); `faster_than_local.md` § 3.4 (the persistent
stream's 2.4× went once both ends negotiated lazily) and § 10 item 9c (what § 7.5 measured dead).
