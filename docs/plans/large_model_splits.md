# Large models across the swarm: the speed plan

**Written 2026-10-20 against v0.3.245-alpha**, after the user set the direction:

> the projects aim is to run LARGE models across the network - the splits cant be
> slow - we need to develop a solution to this, not just rely on existing research

A model that fits on one machine gains nothing from the swarm. The swarm exists to
run models that no member can hold, and a split must therefore be fast enough to
use. This document sets the target, states the one formula that bounds it, lists
what is new on top of the three earlier plans, and gives the build order with the
measurement that ends each step. The measured groundwork lives in those plans and
is not repeated here:

- `split_speculation.md`: how many tokens one pass can carry (drafter replays,
  best-first trees), and the round cost model.
- `wan_parallel.md`: parallelise the reply, not the token (MTP at the last stage,
  several streams per reply, the prompt pass).
- `regional_pipelines.md`: placement by distance (#192, Stage 3).

The constraint is unchanged: **no machine holds the whole model, not even a low-bit
copy** (user, 2026-09-28). Each machine holds its layers plus small things (a draft
head, a small model, caches).

## The target

A 70B-class dense model, or a large MoE (GLM-4.5-Air 106B-A12B, Qwen3-235B-A22B,
gpt-oss-120b), split across machines none of which can hold it. One user, judged
in tokens a second. Nothing is lost in quality unless the user opts in:

| ring | target |
|---|---|
| across continents | ≥ 20 tok/s |
| one region, machines with cards | ≥ 50 tok/s |
| processors only, dense 70B | ≥ 4 tok/s (MoE models higher) |

Today a 7B split Thailand↔Belgium runs at 2.85 tok/s plain and 5-13 tok/s with the
continuous stream. A 4-segment intercontinental chain measured 0.35 tok/s.

## The formula, and the mistake it rules out

**tok/s = E_N / P.** E_N is the number of tokens guessed right per ring pass, from a
guess set of N rows; P is the time of one pass.

- **P** = the hops + each stage's compute + per-pass overhead. Measured on the 7B
  Thailand↔Belgium: 351 ms a token, of which ~206 ms is the ICMP round trip, ~50 ms
  is both cards' compute, and **~95 ms is neither**. #148 found only 3.6-5.3 ms per
  segment after its forward.

  **Measured 2026-10-20: our path adds one round trip per token and nothing more.** The
  rig ran the v0.3.245 release binary, both nodes on the processor, Qwen2.5-Coder-7B Q4_K_M,
  A holding shards 0-3 and B 4-7, 96-token greedy replies to the same prompt. The figures
  are the warm requests (2 and 3 of 3). Every packet was delayed in the transport
  (`split_rig.sh` `RIG_RTT_MS`, ping 208.5-213.4 ms):

  | arm | 96 tokens | ms a token |
  |---|---|---|
  | whole on one node | 12,526-12,835 ms | ~130 |
  | split, 0 ms | 13,773-14,996 ms | 143-156 (+13-25: the split's own cost) |
  | split, 206 ms | 35,414-35,589 ms | 369-371 (+214 over 0 ms: one round trip) |

  So the 95 ms is the real link or the far machine, not a cost that grows with distance in
  our transport. The live node's peers right after the deploy: two European peers
  connected directly answer an exchange in 231-236 ms, one reached through a relay circuit
  in 330 ms (min 220). A relayed hop pays the relay's detour on every token, so a chain's
  hops must be direct. The arms "with guessing ahead" did not guess: the rig held no
  drafter, so they ran plain (no `DSD: streamed checks complete` line) and are not counted.
- **E_N grows about logarithmically with N.** These are measured best-first replays
  (`split_speculation.md` § "Measured"):

  | drafter → target | N = 16 | 32 | 64 | 128 |
  |---|---|---|---|---|
  | Qwen 0.5B → 7B | 4.09 | 4.59 | 5.18 | 5.71 |
  | Llama 1B → 3B | 5.38 | 6.36 | 7.10 | 8.12 |

- **A stream that "never waits" does not escape it.** The head learns where the
  tail stopped one pass late, so what one pass can cover is a finite tree (the
  retraction at `split_speculation.md` § "Measured"). A tree of follow-on trees is
  still a tree. The same law in rate form is time per token ≈ 1/v + m·P, but m, the
  miss rate no branch covers, is ≈ 1/E_N, not a free parameter.

  **Correction, recorded here so nobody builds on it:** on 2026-10-20 a first draft
  of this plan, in conversation only, treated m as free (7%). It projected ~40 tok/s
  across continents and "speed approaching the drafter's". Bounded by E_N, the
  lossless figure is 20-31 tok/s (table below).

## What is new here

0. **The stream must reach every machine of a chain.** Today the continuous stream runs only
   where ONE peer holds every layer past this node's (`dsd_stream::shape_of`: "two peers'
   segments would be two streams to keep in step"). A large model is split across three or more
   machines, so every large model guesses round by round today, at the rounds' cost. The serving
   side is already per segment (`forward_streams` is keyed by request, layer range and attempt);
   what is missing is the coordinator sending each chunk down the chain with the next-hop and
   reply-to trailers a plain chained forward carries (`layer_forward.rs` 0x06/0x07), the tail
   answering it, and the order kept at every hop.
   **Design (2026-10-20, from the code as it stands):**
   - A led plan [B, C] already streams: B coordinates and C is its one peer ("peer-last").
     Only a coordinator with two or more peers falls back to rounds. Those rounds already
     go machine to machine (`forward_verify_through_segments` → `plan_chain`,
     `CHAINED_VERIFY`, `SWARMLLM_CHAIN_VERIFY=0` turns it off). So the hops exist; what is
     missing is the ordering.
   - **Middle hop.** `layer_forward.rs` builds the onward forward with `stream_seq: None`,
     so the next hop runs it unordered (`forward_streams::turn` is skipped). Carry the
     chunk's `stream_seq` onward.
   - **Failure record.** `hop_forwards` keeps ONE record per request (#305), so a
     pipelined stream's second chunk replaces the first's. Key it per (request, seq).
   - **Gaps.** A middle hop that steps over a chunk (superseded by a restart) leaves a
     hole at the next hop. A restart did NOT close it: `turn()` made the restart wait
     until every earlier turn had arrived and taken its turn (`next >= turn`), and the
     stepped-over one never comes, so it waited `STREAM_TURN_WAIT` (60 s).
     **Built (2026-10-20): a restart goes the moment nothing runs.** The stream now
     records whether a turn is RUNNING, under the lock that moves `next`; when nothing
     runs, `next` jumps straight to the restart, and a superseded turn that turns up
     later is answered unrun. A turn is claimed under that lock, so a jump can never put
     two forwards of one request at the worker (gotcha #180), and a second copy of a
     running turn no longer runs beside it. Safe because every turn a restart builds on
     ran on every hop: the coordinator restarts only after the check before it was
     answered, and that check went through every hop to the tail; the worker refuses a
     forward whose position is not its cache's length after the cut. A first draft
     here had the middle hop send a "stepped over" notice per skipped turn; the jump
     needs no message at all. A hole a hop leaves by REFUSING a chunk (an admission
     cap, a failure) ends the stream at the coordinator, which re-plans under a new
     attempt number; the next hop's queued turns of the dead attempt wait out their
     60 s holding one stream slot, never a newer attempt's.
   - **Coordinator.** `dsd_stream`'s far side becomes the chain: the head peer, then each
     hop (`plan_chain`). The wait is `WaiterKey::streamed(request, seq)` with
     `awaiting` = the tail, `chain_members` = every hop and `ExpectedStep` = every hop's
     range, as the rounds' chained wait already has.
   - **Wire gate.** One new `features` bit, `CHAINED_STREAM` (27), advertised by a node
     that carries a chunk's number onward, keeps per-chunk onward records and lets a
     restart pass superseded turns. The coordinator streams through a chain only when
     EVERY hop advertises it plus `STREAMED_VERIFY | STREAM_AS_ONE_WORK | CHAINED_VERIFY`,
     and each machine appears once in the run; otherwise it keeps the rounds. No
     existing variant changes shape, so `PROTOCOL_VERSION` stays.
   - **Found while building it: a user who holds NONE of the model guesses nothing.**
     That is the usual way to run a large model, and the request is handed to the
     machine holding its first layers, which leads it (`delegation_eligible`,
     `try_delegated_split` runs first). The small model that guesses lives on the
     user's machine, which the hand-off leaves behind, so the head leads with no
     drafter: one token a pass, or prompt lookup at best. The fix is a priced choice
     between two loops: led by the head with no drafter, at the head's ring
     (tok/s ≈ 1 / P_head), or driven from the user's machine with its drafter through
     the whole chain, today excluded because "a plan that is the peers' alone is a
     hand-off" (`dsd_stream::shape_of`), at the longer ring (tok/s ≈ E_N / P_user).
     Driving from here wins wherever compute dominates the pass (every processor ring)
     or E_N exceeds P_user / P_head; leading wins across an ocean with no drafter. Next
     after the chain stream's rig.
   - **Built instead (2026-10-20): guesses from the requester — both, not a choice.** Measured
     first (`~/swarmllm-318/drive318.sh`, a 7B over two processor nodes, the requester holding
     none of it and the 0.5B whole, every link 206 ms): led without guesses 1.27-1.31 tok/s,
     driven with the drafter 1.10 — the user↔head trip in every pass cost more than the
     guesses won. So the head keeps leading and NEVER waits, and the user's machine guesses for
     it: it follows the reply as it streams in (it receives it anyway), drafts ahead of it, and
     sends each window (`SwarmMessage::RequesterGuesses { request_id, after, tokens }`). Each
     pass the head checks the newest window's guesses past its reply where the window agrees
     with what it produced since `after`, else prompt lookup, else one token. A late window
     costs rows, never a token (a guess is kept only where the target's own sample equals it);
     a window stays good across passes while the reply agrees with it, so the user drafts up to
     four checks ahead. Pieces: the reply's ids beside its text (`StreamingToken::reply_ids`,
     recorded by the head's loop before it emits each token); the user's seed for shared noise
     in the hand-off, used by every sample the head draws for the reply where both read the
     prompt as the same length; windows taken only from the peer that handed the reply over,
     ≤ 64 ids, each checked against the vocabulary; bit 28 `REQUESTER_GUESSES` at both ends.
     No published system does this; its parts are AMUSD's rollback, PEARL's post-verify
     drafting and PipeInfer's stale-run handling. What it gives up against a drafter beside
     the head ("Speculation at a Distance", arXiv 2606.25091): the window is a round trip
     user↔head late, so the reply must agree with the window's first tokens before its later
     ones are any use. A/B `SWARMLLM_REQUESTER_GUESSES=0`; rig `~/swarmllm-318/guess318.sh`.
   - **On a real link (2026-10-20, one run, `~/swarmllm-318/rl318.sh`).** This machine
     (Thailand, layers 0-13 of the 7B on its processor) led for a requester on the same machine;
     a peer's RTX 3060 in Australia, ~420 ms away, held the rest. Led with guesses: 2.5 tok/s
     over 160 tokens. Of 109 passes, 39 checked guesses (49 of 86 kept), 5 prompt lookup, 65
     one token. Driven from the requester: 2.27-2.38 (α 0.66, γ 2). The 65 one-token passes had
     no fresh window: the requester's drafter shared the head's busy processor and took
     200-450 ms after a miss. **The next step is a bounded wait for an expected window.** Waiting
     w turns a one-token pass into a guessed one when w < P·(E − 1), ~700 ms here — about 1.3x on
     this link. Not testable through `swarm_route`: any override switches delegation off
     (`request_may_be_led_by_its_head`), so the run above was led only because the requester did
     not yet trust the far peer with the prompt.
     **Correction (2026-10-21), recorded here so nobody builds on the rule above:** `w < P·(E − 1)`
     prices the one-token pass as if it threw the window away. It does not: the window lands during
     that pass and still counts at probability α (its first guess equals the token the pass chose).
     Comparing the two as rates, `E / (w + P_g)` against `(1 + α·E) / (P_1 + α·P_g)`, waiting pays
     only while `w < E·(P_1 + α·P_g)/(1 + α·E) − P_g` — ~220 ms on this link's figures (`P_1` 550,
     `P_g` 600, α 0.6, γ 2), not ~700, so the 1.3x was too high. **Built 2026-10-21** with the
     derived bound (`ngram_only_spec::worth_waiting_ms`), and the route instruction now goes with
     the hand-off (`features::CARRIED_ROUTE`, bit 30), so a led split can be measured through
     chosen peers.
   - **Rig:** `split_rig.sh repeat` with `SHARDS_C` (A holds the first parts and the
     drafter, gotcha #915), `RIG_NETNS=1 RIG_RTT_MS=206`; `C_MEETS_B=1` has C dial B.
   - **Built and measured (2026-10-20).** The stream through the chain: 1.24 tok/s
     against 1.07 for chained rounds, 1.04 star rounds, 1.03 without guessing (7B,
     96 greedy tokens, replies identical; C ran 180 numbered checks). The gain is small
     because the rig's three machines share one processor: a stream overlaps the hops'
     compute, and here there is one compute. **The rig found an older defect first:**
     wherever the middle machine had not met the last — the first minutes of any new
     swarm — every chained check failed and every reply ended after one token, the
     rounds' too (gotcha #916). Fixed in the same change: the check is re-run unchained,
     a failed chained stream finishes in rounds, and the middle machine refuses before
     working and dials its next hop, looking it up in the DHT when it has no address
     (on the loopback rig the DHT has none, so the run stays unchained: the dial after
     a lookup waits for a real link).
1. **N is bounded by throughput, not by the round.** The round cost model in
   `split_speculation.md` added a tree's compute and transfer to every round, which
   is why tree 64 lost to tree 32 on short links. In the continuous stream (§ 4b,
   shipped in v0.3.216) the next window's rows travel while the previous window is
   checked. N per pass is then bounded by link rate × P ÷ bytes per row, and by each
   stage's rows a second; neither adds latency. A card stage reads 1 or 64 rows in
   nearly the same time (`wan_parallel.md` § 1).
   - A 70B row on the wire is Q8_0, the default since `activation_compression`:
     8,192 × 34/32 ≈ 8.7 KB.
   - At P = 0.26 s that allows ~190 rows a pass at 50 Mbit/s and ~75 at 20 Mbit/s.
   - Windows stay small (8-16 rows), so a window's own serialisation (~11 ms for 8
     rows at 50 Mbit/s) does not lengthen P.
2. **Every machine hears the verdict directly.** Today a stale chunk is skipped only
   if it has not started (#149 (b)). Instead, when the tail walks it sends the
   accepted path to every segment, one direct message each. A segment drops the rows
   outside the surviving subtree before computing them, so in-flight N is spent on
   live branches only. This is PipeInfer's early cancellation (arXiv 2407.11798),
   generalised from a chain to a tree.
3. **The drafter sees the big model's own state.** In order of preference:
   - **MTP** at the last stage for models that ship it (`wan_parallel.md` § 2, #145).
     The tail samples, drafts and returns the verdict with the next drafts.
   - **EAGLE-3 heads** where one is published. The head needs three tapped layers'
     rows. Each tapping stage sends its row straight to the coordinator (8.7 KB a
     token per tap), and the tail adds its final row to the verdict. The coordinator
     already holds `token_embd` (`split/loader/mod.rs:146`).
   - **A same-family small drafter plus prompt lookup** for every other model.
4. **An MoE tree is priced by the experts it touches.** A tree's rows route to
   different experts, so the cost is the expert union, not the row count. On OLMoE a
   127-row tree touched 54 of 64 experts per layer (MoE-Spec, arXiv 2602.16052; a
   cost-aware selector, arXiv 2607.12696). Each stage measures cost(rows) on its own
   hardware, expert union included, and publishes it beside the decode time a peer
   already publishes (#3 (f)).
5. **Cut P before anything multiplies it.**
   - First measure where the 95 ms goes: timestamps per hop on ONE pass. Then remove
     the largest term.
   - Then build regional chains (#192, Stage 3), the largest single factor
     (~0.26 s → ~0.07-0.09 s).
6. **An opt-in fast mode that checks less strictly.** Judge Decoding (arXiv
   2501.19309) keeps a draft the target judges acceptable rather than identical:
   19.7 tokens accepted a pass against 6.3, at 405B/8B. Rules for this mode:
   - never the default;
   - labelled in the UI, which must say the reply is no longer the model's own;
   - scored against llama.cpp with `examples/score_against_reference.py`.

## Projections

Lossless unless marked. P across continents assumes the 95 ms is removed.

| ring | P | N a pass | E_N | tok/s |
|---|---|---|---|---|
| across continents, today's drafter class | 0.26 s | 64-128 | 5.2-8.1 | **20-31** |
| across continents, MTP or EAGLE-3 tree | 0.26 s | 64-128 | ~9 | ~35 |
| across continents, opt-in fast mode | 0.26 s | — | ~15-19 | ~60-70 |
| one region, cards | 0.07-0.09 s | ~80 | 5.4-7.5 | **50-80** (compute-bound) |
| processors only, dense 70B | ~1.0 s (two stages × ~0.45 s) | ~16 | 4.1-5.4 | **4-5** |

These are projections from measured parts, not measurements.

- **E_N for a 70B target** with its own family's drafter has not been measured;
  the rows above borrow the 7B and 3B replays.
- **A processor stage** turns compute-bound after a few rows. 8 rows roughly double
  a 70B stage's time (an estimate, unmeasured), so N stays small there.
- **MoE models make processor rings viable**: gpt-oss-120b has 5.1B active
  parameters and GLM-4.5-Air 12B. Their ceiling is set by item 4, not by the bytes.

## Draft trees (#134): the design, 2026-10-20

A chain's pass keeps at most the run of guesses before its first miss; a tree
keeps the longest run along whichever branch the model takes. Over a long link
the pass costs a round trip whatever it carries, so rows are nearly free there
until the bytes or a processor stage say otherwise.

- **What a pass carries.** Rows are `[kept tokens re-read, the token the last
  check chose, tree nodes]`. `draft_parents[i]` names row i's parent among the
  forward's rows, or none for the first row (its parent is the cache's last
  position). A chain is `parents[i] = i - 1`. It is one new `LayerForward`
  trailer (`0x10`), bound in the AAD and sent only to a peer advertising
  `features::DRAFT_TREES` (bit 29). Every hop carries it onward.
- **Every segment.** Row i sits at `index_pos + depth(i)` for RoPE and sees the
  whole cache plus its own ancestors in the forward. Standard attention is forced,
  because the card's flash kernel takes causality as a flag, and CUDA graphs are
  skipped. Only dense attention layers take a tree: a sliding window, Qwen 3.5's
  recurrent layers and DeepSeek's MLA refuse, and the request keeps the chain.
- **The tail's walk.** Sample at the first row and descend to the child whose
  token is the sample. Repeat until no child matches, then answer the kept tokens
  plus that last sample. This is exact at any temperature. A node's children
  carry distinct tokens, so "keep the child equal to the target's own draw" is
  `sampled_accept_reject`'s n-gram rule applied along one path (SpecExec §3).
  The shared noise is keyed by absolute position, `index_pos + depth + 1`.
- **The cache after a check: re-read, never gather.** The drafter's most likely
  chain is laid out first (rows 1..D), so the kept path is usually a prefix of
  the rows. The next forward cuts every cache to that prefix (`truncate_kv_to`,
  as today) and re-reads only the kept tokens past it, as a chain at the head of
  its rows. That costs a few rows a pass. It saves a second trailer and a gather
  in every segment: FlowSpec broadcasts a retain set instead, which costs more
  wire.
- **The drafter builds the tree in its own worker, in one call.** The search is
  best-first over cumulative log-probability (SpecExec's parallel search,
  EAGLE-2's global rerank). Each step expands the M best unexpanded nodes in one
  tree forward of the drafter, with the same mask and positions, and queues their
  top-k children. It stops at N nodes, then cuts the drafter's cache back to the
  confirmed context.
- **Size, from the measured pass.** `P(N) = check_fixed + N·row + ⌈N/M⌉·step`.
  Pick the N that maximises `(E_N + 1) / P(N)`, with E_N read from this request's
  own record of kept rows: Sequoia's rule over our measured terms. A processor
  stage's large `row` cost keeps its trees small with no special case.
- **Fallback.** The request keeps today's chain loop when any of these happens:
  a peer lacks bit 29, the model's layers refuse a tree, or a tree forward is
  refused.
- **Measured: trees need the segments AND the drafter on cards, so they are the
  default only there (2026-10-21) and the stream elsewhere.** With the head's
  part on a processor (2026-10-20, 385 ms emulated) a row cost the check 10-19
  ms; trees grown beside their check ran 4.75-5.06 tok/s against the stream's
  5.03-5.52. On the real Thailand↔Australia link (~405 ms) with both halves on
  cards and the drafter on the head's PROCESSOR, a row cost 2-3 ms and a tree
  kept 3.2-4.6 guesses a round; trees ran 5.33-5.74 against 5.01-5.05 over 160
  tokens, but 4.82/5.05/6.40 against 5.97/6.04/6.18 over 400 and ~10% slower
  over 28 — each missed seed waits ~300 ms for eight processor forwards, and
  the stream gains with length where trees do not (`docs/invariants/scheduling.md`
  § "A split checks a tree of guesses"). With the drafter on a card those eight
  forwards are ~40 ms, and on the same link (2026-10-22, arms alternating)
  trees ran 8.41 tok/s against the stream's 5.95 at 160 tokens and 9.36
  against 6.48 at 400 — the default there stands. A plan filling the head's
  card now leaves the drafter room on it (#275 (4), 2026-10-22).
  `SWARMLLM_DRAFT_TREES=1` / `=0` forces either way. Next: a stream of trees,
  fewer seed misses.
- **A stream of trees, weighed 2026-10-21.** FlowSpec (arXiv 2507.02620) and
  SpecPipe (2504.04104) keep tree pieces flowing through a pipeline: rows go
  out best-first as the drafter grows them, each stage prunes against the
  last verdict, and a verdict whose path the rows sent do not hold discards
  everything in flight — our seed miss. Both are LAN designs (transfer under a
  stage's compute); neither measured a long link. What a stream would change
  here: a hit saves a whole round trip (the follow-on rows are already at the
  tail), where a seed only saves the growth wait. What it would not: the law
  above — rows per pass still buy E_N logarithmically. Its cost: every
  segment's cache would hold unverified rows from earlier forwards, which
  breaks "the cache after a check is a prefix of the forward's rows" and
  needs a pruned tree cache in every worker. A self-contained variant avoids
  that: every check rooted at the confirmed reply, carrying `truncate_kv_to`
  at the oldest root still in flight, so each one is a restart and
  `forward_streams` skips the ones a later check overtook. Priced below; not
  started. **Re-priced 2026-10-22**: that variant is the one priced below; the
  shape that pays — rows the far end keeps reading, no restart per check — is §
  "A tree that flows".
- **The schedule, priced (2026-10-21, `examples/tree_schedule_sim.py`).** One
  round is priced by `spec_best_tree.py`; this prices the schedule around it,
  with Sequoia's positional acceptance and the costs measured on TH↔AU. It
  reproduces that link within ~10%. tok/s over 400 tokens, mean of 20 replies:

  | link, drafter | tree rounds (today) | regrow sized by what it buys | stream of trees | chain stream |
  |---|---|---|---|---|
  | 425 ms, card | 7.18 | 7.06 | 8.00 | 7.32 |
  | 425 ms, processor | 5.76 | 5.99 | 6.54 | 6.61 |
  | 20 ms, card | 21.66 | 40.98 | 33.86 | 47.61 |
  | 20 ms, processor | 11.41 | 29.34 | 13.97 | 21.13 |

  Three things follow. **On a long link the schedule is not the lever**: every
  shape lands within ~15% of the others, because confirmation advances about
  one tree depth per round trip whatever is in flight. Speed there comes from
  what a round keeps (a better drafter, build-order step 4) and from not crossing the
  long link (#192). **On a short link today's regrow halves trees**: a missed
  seed waits for eight drafter forwards before a check that itself costs
  ~30 ms, so trees run at 0.45× the chain stream. Sizing the regrow by what it
  buys (one, two, four or eight steps, whichever gives the most tokens per ms
  of the round) recovers most of it. **So trees are worth their default only
  on a long link** until a rig says otherwise: the near chains placement builds
  (#192) are exactly the short links where they lose. The model is optimistic
  for the chain stream (it leaves out each chunk's fixed cost and a restart's
  thrown-away work), so the crossover is measured, not taken from it: rig at
  20 and 300 ms. The stream of trees waits behind both; it pays only where
  the drafter is on a card and the link is long. **Measured 2026-10-21**
  (drafter on the processor, trees forced, 160 tokens): the sized regrow
  11.1-11.9 tok/s against eight forwards' 7.3-7.5 at 20 ms, 5.4-5.5 against
  5.1-5.3 at 300 ms; the chain stream 15.8-16.2 and 6.0 — built as the
  default with the 300 ms gate (`docs/invariants/scheduling.md`).

## A tree that flows (2026-10-22): the far end reads while the requester grows

**Where a tree round's time goes on TH↔AU** (`~/swarmllm-318/cardfill275q8.sh`'s default arm,
7B 0-16 here and the Q8 0.5B drafter on this card, 16-28 on the RTX 3060 ~411 ms away; 192
checks timed at their receipts): a round is 608-654 ms. ~485 ms of it is in flight — the round
trip 411, the uplink ~16 (0.087 ms/KB), the far check ~19 plus ~0.22 ms/KB of what it sends
back — and 130-168 ms (mean) is HERE, waiting for the drafter. Per reply the seed missed the
outcome on 32-51% of rounds, was sent as it was on 12-23%, and was grown ~8 forwards before the
check on the rest. `prefill_bench` on this card: a drafter forward costs nearly the same at 32
rows as at 8 (17.4 vs 14.9 ms, tree; 12.3 vs 8.5 chain), and a tree forward pays 6 ms (drafter)
to 10 ms (7B head at 52 rows: 34.0 vs 24.4) over a chain's — 48 host-to-card copies a step for
the rows' positions. Two levers built from it: the positions gathered once per forward
(`layers::TreeRows`), and growth beside the check at 32 rows a forward up to 416
(`dsd_tree::beside_growth`; A/B `SWARMLLM_TREE_ROPE=per-call`, `SWARMLLM_TREE_BESIDE=8,128`).

**The earlier "stream of trees" price was the wrong stream.** § "A stream of trees, weighed"
priced checks rooted at the CONFIRMED reply, each a restart — so confirmation still advanced one
tree per round trip, and it came out +11%. The shape FlowSpec runs on a LAN is different: rows
go out as they are grown and the far end keeps reading them, so the reply advances as deep as
the tree reaches along the model's path and stalls only where it holds no row — one round trip
per such miss, where a round pays one per check. `examples/tree_schedule_sim.py`'s `flow`
(2026-10-22, the tail reading what has arrived in one forward, the drafter and head sharing one
card), 400 tokens over 425 ms, card drafter:

| step costs | tree rounds | wide beside (built) | flow, 16 rows a step |
|---|---|---|---|
| today's | 7.06 | 7.46 | 9.87 |
| a tree step captured (guessed: call 2, step 11, head fixed 5 ms) | 7.69 | 8.06 | 11.31 |

+40-47% over today's rounds where the model put today's rounds 15-20% under the field (7.2
against 8.4-9.4 measured). It loses with the drafter on the processor (4.9-6.9 against 5.8-6.0)
and over 20 ms (25.7 against the chain stream's 47.6): a flow is for a long link and a card
drafter, the place trees already are the default. Its ceiling is the miss rate: a token no row
holds costs a round trip, so ~10 tokens between such misses at 425 ms is ~24 tok/s.

**Research (sonnet digest, 2026-10-22).** FlowSpec 2507.02620 (Jetson LAN, per-stage compute
~80 ms, latency never varied): score-ordered segments of ≤ 16 rows, the verdict back to the
DRAFTER, a retained-index broadcast of a few hundred bytes to every stage, a miss starts a new
round; 1.37-1.73× over non-streamed pipeline speculation. SpecPipe 2504.04104 (10 GbE): a tree
level per pipeline step, pruning propagated to every device. PipeInfer 2407.11798: runs in
flight, KV sequence per run, cancellation back-propagated. "Speculation at a Distance"
2606.25091 models client-side pipelining that waits for verdicts — gains gone past ~50 ms; it
does not model rows the far end reads without waiting. **Nothing published measures this shape
over hundreds of ms**; the design below is ours.

**The design — every piece but the loop already exists.**
- **A segment continues the tree it holds.** `DraftTree::rows_for(kv_offset, seq_len)`
  already reads a forward as the LAST rows of a tree whose earlier rows are in the cache — the
  drafter grows that way within one call. A flow forward appends its rows to the tree the
  segment keeps for the request (parents anywhere in it); a row reads the context and its own
  ancestors, so a dead branch stays in the cache and is read by nothing. No cut per verdict:
  the cache is cut and the confirmed path re-read only when the tree reaches its row limit.
- **The far end answers with its pick for each row** — the token the model draws there, with
  the noise keyed by position as today — never a walk. The REQUESTER walks: down from the
  confirmed row, to the child carrying each pick, until a pick names no child. Exact at any
  temperature (the draw at a row does not depend on which forward carried it), and the worker
  holds no walk state.
- **The drafter keeps its tree across calls**: the cached rows and the unexpanded children
  (today rebuilt each call by re-reading the seed), growth re-rooted under each confirmed row.
- **The loop**: grow a step, run it through this node's segment, send it, never wait; on each
  answer, walk, emit, re-root. Several forwards in flight on one connection, as the chain
  stream (`dsd_stream`) already keeps.
- **A feature bit** for the forward that continues a tree and the per-row picks back; anything
  else keeps the rounds.

**Build steps, each ending on a measurement.** (1) A segment continues a tree across forwards
and answers picks: a unit test that a tree read in three forwards gives every row the logits it
gets read in one (the control: a chain over the same rows must differ), and that a dead branch
changes no live row. (2) The drafter's growth session: the same equality for its rows. (3) The
loop, rig at a 300 ms injected delay against tree rounds on the same replies; then the real
link. Exit: ≥ 1.3× tree rounds over 400 tokens, replies scored against llama.cpp.

**Built 2026-10-22 (steps 1-3; released in v0.3.248).** Steps 1 and 2's equalities hold
(`a_tree_read_in_three_forwards_reads_as_one`,
`a_tree_that_flows_proposes_what_a_fresh_read_of_each_path_ranks_likeliest`). Step 3 on the card
rig at 385 ms: 8.99-9.44 tok/s against tree rounds' 8.03-8.27 over 400 tokens, every reply token
rank 1 or a near-tie against llama.cpp — +11-14%, short of the exit. The first build grew one row
of depth a drafter call and the walk followed at 0.63-0.74 tokens a call whatever the rows; three
levels a call (EAGLE-2's shape) took it to 1.05-1.17. The requester is compute-bound and the
drafter's levels are ~72% of it: next, its tree steps captured as CUDA graphs, then the real link
(`docs/invariants/scheduling.md` § "A tree that flows", Measured).

**Since then (2026-10-22/23), all on the requester's card.** The drafter's tree level is one tree
kernel and one CUDA graph (idle card, a flow call 36.0 → 26.5 ms) and ranks its tree in two card
launches (→ 21 ms); both shipped in v0.3.248 with the flow and FlowGuard. After the tag: a tree's
context is read by flash attention in runs, and Q, K and V reach attention heads-first in one launch
(a flow call 600 tokens in 14.1 → 12.7 ms; a 7B near half checking 24 rows 4,000 tokens in 22.7 →
14.8 ms; `docs/invariants/inference.md`, both sections).

**On the real link (2026-10-23).** TH↔AU, 7B, the far half on an RTX 3060 at 428-437 ms, this
machine's half and the 0.5B drafter on its card. Over 400 tokens: the flow 14.3-14.6 tok/s, tree
rounds 11.6, the chain stream 6.8. A plain split on a link like this runs ~2.85 tok/s (§ "The
target"); .247's default here was 9.36. The cheaper requester on main (−5% drafting, −10% near half a forward) gave
the same tok/s: it ran more forwards, not more reply. No knob moved it either: live rows 128 / 256
/ 448, 8×3 against 4×6 levels (`docs/invariants/scheduling.md` § "A tree that flows", On the REAL
link). **The flow now pays the round trip and nothing else, so it advances E — the tokens its live
rows get right, ~6.5 here — per round trip.** That is this plan's formula with P at its floor. Every
further factor on a long link has to raise E: a better drafter on the card, copied runs where the
reply repeats its context (2× on an edit), a drafter fed by the model's own state (§ "Build order"
item 4), or a shorter P through near chains (item 6). **The drafter, measured the same day:**
Qwen2.5-Coder-1.5B Q8_0 on the card ran 16.46 tok/s (10-11 misses a 400-token reply), the 0.5B
Q8_0 15.57 (17-19), the 0.5B fp16 13.98, and the coder 0.5B 12.06 (39-41). The cheaper copy of one
drafter turns its saving into waiting; only fewer misses gain. That is E again: ~7.4 tokens a round
trip with the 1.5B against ~6.5. The 20 tok/s target here needs E ≈ 9 a 450 ms round trip, so the
drafters a node can already fetch fall short. Next on E: a drafter fed by the model's state, or a
larger drafter where the card has room.

## Build order (each step ends on a measurement)

1. **Per-pass breakdown.** Per hop and in both directions, record:
   - send queued, on the wire, received;
   - compute start and end;
   - Q8 encode and decode.

   Run it on the rig with `SWARMLLM_TEST_TENSOR_DELAY_MS` and on one real
   Thailand↔Belgium pass. *Exit:* the 95 ms attributed to named terms within 10 ms.
   Then remove the largest term.

   **Rig half done 2026-10-20** (§ "The formula"): distance costs exactly one round trip,
   and the split's own cost is 13-25 ms a token on the processor. What is left is the real
   link: compare each pass with the round trip of the connection it crossed (direct or
   relayed, QUIC or TCP), on a split to a European peer.
2. **Tree verification in every segment** (`split_speculation.md` Phase 2, items
   1-3): a tree mask, a gather of the accepted path in the KV cache, and a walk at
   the tail. It is gated on a `features` bit at the sender. *Exit:* greedy replies
   identical to the chain's, and E per pass on the rig within 10% of
   `spec_best_tree.py`'s replay.
3. **The stream through a chain of machines (item 0), then trees in it, plus the verdict sent to
   every segment.** *Exit for the chain:* a 3-machine split streams (the `DSD: streamed checks
   complete` line, not the rounds) and its tok/s on the rig at a 300 ms injected delay beats the
   rounds on the same replies.
   *Exit for the trees:* tok/s on the rig at a 300 ms injected delay, against the chain stream
   on the same replies (aim ≥ 1.5×).
4. **A drafter fed by the model's state.**
   - MTP at the last stage. Qwen 3.5 first needs its recurrent state wound back
     (#228); GLM-4.5 needs family support.
   - Then EAGLE-3 taps where a head is published.

   *Exit:* E per pass against the 0.5B drafter on the same replies.
5. **The MoE cost curve in the tree-size rule.** *Exit:* the chosen N within 10% of
   the measured best on an MoE rig.
6. **Regional chains** (#192). *Exit:* a split coordinated inside the region runs
   at that region's P. **Placement built 2026-10-21** (Stage 3 phase 1,
   `regional_pipelines.md`): nearby machines fetch complementary parts of a model none
   of them can run whole, and give back layers they hold loaded outside their segment so a
   router can plan the chain; `examples/near_chain_rig.sh` (2026-10-21): the next 160-token
   reply over the two near machines in 13.9 s against 36.8-41.4 s split with a far one.
7. **The opt-in fast mode.** *Exit:* a quality report scored against llama.cpp.

## Traps already known

- **The retraction above.** Never count tokens per pass from a tree that assumes the
  head knows the tail's path.
- **TCP/QUIC slow start after an idle gap** on a tree burst. This is an open unknown
  in `split_speculation.md`; the persistent stream keeps the connection warm.
- **A chunk a worker has already started cannot be cancelled** (#149 (b), gotcha
  #180). Item 2's early drop acts only on rows not yet started.
- **Qwen 3.5's recurrent state cannot be wound back yet** (#228). A tree needs one
  state per branch, so that model stays on a chain until it can.
- **"Start and finish on this computer"** (`encrypted_pipeline`) makes the requester
  both head and tail, so P includes the requester's distance. A regional chain helps
  only when the requester is in that region.
- **Measuring.** The rig shares the machine with nothing (#708). Concurrent requests
  are a stress test. Use min-of-N on an idle box, and A/B inside one binary with an
  env switch.

## Sources

**Surveyed again 2026-10-21** (two web surveys and a ledger of this repo's own attempts, after
the user asked for "much much better" than 6 tok/s and to build on what was ALREADY tried). What
it changed: nothing in the build order, one emphasis.
- **No published system speculates over a link of 400 ms or more.** The WAN-like ones (Petals,
  BloomBee 2604.21072) report no absolute tok/s at that distance; PipeDec/SpecPipe 2504.04104
  (54.7 ms a token, 70B, 8 stages on 10 GbE), FlowSpec 2507.02620 and PipeInfer are LAN or
  datacenter. TH↔AU's ~6 tok/s is already past the published WAN figures, and the far ring's
  ceiling is still E_N/P: at ~405 ms and the 0.5B's E_128 ≈ 5.7, ~13 tok/s even with P at the
  bare round trip.
- **In one region the same papers project 40-60 tok/s for a 70B on two cards at 5-25 ms**
  (pipelined speculation with chunks in flight; tree width 32-64 was SpecPipe's best). That
  is this plan's "one region" row, so **placement (item 6) is the order-of-magnitude lever**,
  and the far ring's levers (items 1, 3, 4) are each worth well under 2×.
- **Whole-model hand-off already gives near-local speed where one machine can run the model**:
  the router prices a far whole-model holder's network once, a split's per token
  (`parallax::vertex_cost_as`), so a split only serves when nobody can run the model whole —
  the case this plan is for.

PipeInfer 2407.11798 · SpecPipe 2504.04104 · Judge Decoding 2501.19309 · EAGLE-3
2503.01840 · MoE-Spec 2602.16052 · cost-aware MoE speculation 2607.12696 ·
decentralized speculative decoding 2511.11733 · Speculation at a Distance
2606.25091 (over a WAN the gain is mostly capacity; latency gains need a short
loop, which is why item 5 comes first) · DeepSeek-V3 MTP 2412.19437 (85-90%
second-token acceptance).
