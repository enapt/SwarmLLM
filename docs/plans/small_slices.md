# Small slices, fast replies — the plan (2026-11-20)

**The aim, in the owner's words** (2026-11-20): *"we want to reduce the amount of processing any node should do,
and distribute the load over the network. nobody has datacentres, or wants their gpu locked up by an entire
model."* And (2026-11-13): *"like how torrents get much faster when theres more seeds."* A whole model on one
card is a reference in a measurement here, never a design; no design may need anyone to hold the whole model.
**The bar** (owner, 2026-11-20, hearing 3-7 tok/s for the 27B): *"we really need minimum 20-30t/s"* — a step is
judged by whether it moves a reply toward that, never by its ratio over a slow split.

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
| here → Australia → here | 8 + 12 GB | prose 5.3, code 5.7-7.2 | the guess head beside the first layers, fed the last layer's rows by whichever computer runs it |
| here → US laptop | 8 + 16 GB | prose 3.05, code 3.08 | the head found no room on this node's full card (corrected below): one token a 272 ms round trip |
| **here → US laptop, the head in system memory (#485, built the same day)** | 8 + 16 GB | **prose 5.0-5.9, code 5.95-6.2** | ~38% of its guesses kept, 12 ms a guess on the processor, ~2.6 tokens a round |
| here → Australia → US laptop | 8 + 12 + 16 GB | prose 2.78, code 2.23 | three slices; the head had no room here either (as above), ~360-450 ms a token |
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

**Results, 2026-11-20** (`python3 -I examples/tree_schedule_sim.py ring 12` and the same `flow_ring` at the live-row
budget; prose profile, the 14B pair's measured prices; `flow_ring` with one far stage reproduces `flow` exactly,
20.73 = 20.73):

| shape (prose, tok/s) | 128 live rows, window 8 (today) | 256 live rows, window 12 |
|---|---|---|
| 2 slices, 130 ms (measured 17.8-20.5) | 19.3 | 14.1 |
| 2 slices, 290 ms | 12.4 | 13.5 |
| **3 slices, 290 ms ring (TH → AU → US)** | **11.2** | 12.5 |
| 3 slices, 60 ms ring (one region) | 20.0 | 14.2 |
| 4 slices, 60 ms ring | 19.1 | 14.1 |
| 6 slices, 90 ms ring | 14.3 | 14.0 |

- **1a: step 4 survives** — three slices at 290 ms read 11.2 (kill line 8), about the 14B whole on one card
  (11.0-11.3). The loop's length costs, the slice count barely: one far computer at 290 ms reads 12.4, three
  slices 11.2; in one region three or four slices read 19-20 — each node a third or a quarter, ~1.75x one card.
- **1d: the long-loop misfit is the live-row budget.** At 128 rows (`SWARMLLM_FLOW_LIVE`, today's) the sim reads
  11.8-11.9 at 308 ms against 11.7-11.8 measured; four times the wire cost a row moves it only 11.9 → 11.4. The
  budget does not scale with the loop: 128 is right at 130 ms (more rows HURT there — the field's "bigger sends
  lost", three sittings), 256 with a window of 12 reads ~10% more at 290-308 ms. → a small lever: size the live
  rows and the chain window (`forward_streams::MAX_STREAM_WINDOW`, 8) by the measured loop; the window is a
  per-hop limit both ends must agree on, so a feature bit, gated at the sender. Field check on the US laptop
  (272 ms) once it holds a dense model with a drafter.
- **1c: not built** — the drafter off the head reads 1.01-1.10x (prose and code, 130 and 290 ms): with the
  measured prices the far slice already sets the pace (~30 ms a forward against the head's ~26).
- Caveat: the sim's walk advances at most one level a forward (the real flow three), so its CODE figures are
  capped (27.8 at 130 ms against 40-43 measured); prose is the calibrated one.
- **1b: re-framed, not a replay.** Chunks WITHIN one block cannot beat the loop: every chunk is drafted from the
  same anchor in one block forward, and the next block needs the verdict, so splitting the check into chunks only
  overlaps a few rows' wire and compute. The 1.5-2x the research gave assumed the reply moves at the far's pace,
  which needs the NEXT block drafted from the predicted end of this one — without the target's features for the
  guessed positions (a block drafter reads them for committed positions only). That acceptance is unmeasured
  anywhere. The check is offline, on Qwen3.5-9B whole here with its DFlash drafter (#428 stage 1's harness):
  draft block k+1 from block k's predicted end with the features up to the last commit, and score it against the
  greedy reply. Step 5 waits on it.

### Step 2 — DONE 2026-11-20: a chain that ends on a peer guesses (FUTURE_WORK #485, closed)

- **Corrected premise**: the file's guess head beside the FIRST segment never needed the chain to finish here —
  the last layer's rows come back from whichever computer runs it (`BlockRowsAsk`, `features::SENDS_BLOCK_ROWS`).
  It guessed nothing because it never loaded: a planner fills the first segment's card, and the 27B's 227 MB block
  and 994 MB output head found no room (a 127 MB budget on this node's 8 GB card).
- **Built**: the head goes to system memory where the card has no room (`mtp::split_head_home`), its output head
  read where its block runs, its hot rows on the processor. here → US laptop: 3.05-3.15 → prose 5.0-5.9, code
  5.95-6.2 tok/s (four arms of 4 replies, `~/swarmllm-gate/split27_head`). **Still far under the bar.**
- The guesser at the TAIL (vLLM PR #44698's shape) stays a memory design, not a speed one: it would spare the asker
  the head (1.2 GB of its memory), with the same one check a round trip.

### Step 2b — NEXT: several rounds in flight for the Qwen 3.5 family (FUTURE_WORK #486)

- **Why**: the family checks ONE round at a time (`ngram_only_spec`), so a reply is ~2.6 tokens a 437 ms round
  at 272 ms. The speed model (§ 2) with those 2.6 tokens a round (m = 0.39 misses a token) and this node's stage
  at ~70 ms a round (its layers plus the head's ~4 × 12 ms guesses, so τ/e0 ≈ 27 ms): rounds in flight reach
  ~7.6 tok/s at 272 ms (27 + 0.39 × 272 ms a token) and ~20 in a 60 ms ring (27 + 23 ms) — INFERRED. So the bar
  needs BOTH in-flight rounds and short rings (step 6), and a faster head (on a card) or better acceptance lifts
  both.
- **Mechanism**: the head guesses the next round from the predicted end of the one in flight (`HeadChain`
  already guesses while a check is abroad); each segment keeps a recurrent state per round in flight (#437's
  per-row parent states, ≤ `SsmState::MOST_ROWS_KEPT`) and drops those past a refusal.
- **Experiment first**: `tree_schedule_sim.py ring` with the measured per-guess acceptance (0.38) and a one-check
  head, against K rounds in flight, at 130 / 272 / 60 ms. Kill if a 60 ms ring reads under 15 tok/s (the bar is
  reached only in short rings; at 272 ms the estimate is ~1.3×).
- **Prior art**: PipeInfer (2407.11798, continuous asynchronous speculation), PEARL (post-verify drafting),
  SpecPipe (2504.04104, tokens a forward through a pipeline); the flow (#134 (b)) is ours for families with a
  separate drafter.

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
