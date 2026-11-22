# The standing ring — one reply faster than local (concept, 2026-11-21)

Status: CONCEPT. Nothing here has run on a wire. Every figure is labelled MEASURED (our records), DERIVED (arithmetic
or simulation from them) or ASSUMED. Simulations: `examples/standing_ring_sim.py` (the design's model) and
`examples/standing_ring_check.py` (the independent re-check and corrections; `python3 -I`; both read the drafter
rank dumps in `~/swarmllm-ref/spec/`). Re-run 2026-11-21: both reproduce every figure below.

**How it was made** (owner, 2026-11-21: *"we may be focusing on prior art and solutions too much, and we will
eventually hit the same wall they are"*; *"we need something groundbreaking … get faster than local speeds through
split inference"*; *"thousands of users, with gpus, which could all be used in unison"*): six agents in turn — three
inventors (idle swarm, the ceiling, thousands of cards), an adversary, an architect and a judge — each told to reason
from our own measurements first and prior art second. It sits beside `small_slices.md` (the ranked steps toward
20-30 tok/s) and `faster_than_local.md` § 9 (the flow); where they disagree, the measurement in § 7 below decides.

**Read §§ 7 and 9 first** (MEASURED the same night): the verdict delay is ~180 ms, as the law has it, but misses ×
that delay are under half of a prose token's time and none of a code token's, and the law's mechanism is wrong in
part: on one card a shorter loop bought 1.54× (100 → 0 ms, 21.6 → 33.2 tok/s) at the SAME miss count, through more
of each forward's rows landing on the reply. The head stays busy 98-100% at every loop, so past a short loop the
pacer is the lever — unmeasured, and **not measurable on today's computers (§ 10)**: it needs a card that holds no
slice. **§ 11** (real drafter ranks): the model now matches every arm's misses. On today's pair no row policy moves
prose by more than a few percent; the next prose lever to price is a stronger drafter.

## Summary

1. We will build a **standing ring**: the fewest graphics cards that can hold a model between them, chosen where
   members sit close together, with a small guessing model that never stops guessing ahead; the person asking only
   sends their words and receives text.
2. It gets past today's limit because our own measurements show a reply's time is mostly **wrong guesses multiplied by
   how long a guess waits for its check** (about 40 of every 53 ms today), not the cards' own work, and a short ring
   of cards shrinks that wait three to four times. **MEASURED since (§ 7): ~20-24 of a 49-54 ms prose token and
   none of a 21 ms code token — the head's own pace is the larger part.**
3. We expect **3-8 times the speed of the same model on a member's own computer** (14B: 34-72 tok/s, 27B: 17-44,
   70B: 15-33 where no member can run it at all) and **about 1-2 times a high-end 24 GB card** for the 27B and 70B.
4. The condition is that the ring's cards are **within one city or region of each other** (20-60 ms) and every slice
   sits on a card; across an ocean the gain shrinks to today's.
5. More cards do **not** make one reply faster in proportion: past the ring, sixteen times the cards add about a
   tenth, so the swarm's size buys more rings (more conversations, and tasks split into parallel replies), and the
   one thing that could double a reply again is a better guesser, which is unmeasured.

## 1. What actually bounds one reply

The plan's ceiling reads `time a token = τ/e0 + m·P`, "one reply stops at e0/τ". That model treats e0 (tokens a
forward advances, 1.03 prose) as a constant. It is not one.

**The law that fits:** `time a token = d + m(W, n) × L`

- `d`: ms the drafter needs for one level of its tree. Today 13.7 ms, because the drafter call (14.5 ms for 3
  levels, MEASURED) runs in series with the head's layers (26.5 ms): 41 ms a forward ÷ 3 levels.
- `L`: ms from drafting a level to hearing the target's verdict on it. Today ~222 ms on the 130 ms pair (DERIVED).
- `n = L/d`: levels in flight. `W`: rows a level. `m`: the chance a level loses the target's token; a miss voids
  everything in flight, which costs `L`.
- `m` is not fitted. It is replayed on 1,200 recorded positions (`~/swarmllm-ref/spec/gumbeldump_c15-T07.npz`,
  1.5B drafter against a 7B target, T 0.7): **0.177 prose, 0.045 code at 8 rows, 8 or more levels in flight**
  (DERIVED from MEASURED records).

Today's 53 ms prose token is therefore **13.7 ms of pace and 39.5 ms of voided levels** (DERIVED). A forward
advances 0.78 tokens out of 3 drafted levels (MEASURED 0.72-0.82) because a miss voids ~16 levels in flight, not
because a level is weak.

**Evidence for the law** (one value fitted: 165 ms of wire and waiting, on the default arm's prose only):

| check | law | measured | source |
|---|---|---|---|
| Five arms, 14B pair 123-145 ms, 9 predicted cells | mean error 8% (16-row arm −18%, −26%) | 2026-11-15 arms | `docs/invariants/scheduling.md` |
| Tokens a forward, default arm | 0.78 prose / 1.74 code | 0.72-0.82 / 1.65-1.82 | same |
| 308 ms run (US card), not fitted | 10.4 prose / 23.3 code | 11.7-11.8 / 26.9-27.2 (−11%, −14%) | `faster_than_local.md` § 9; the old model read 14.8 / 39.9 |
| Slope of ms-a-token against the loop should equal m (0.177) | 0.177 | 0.179 (pair 130 → 308 ms); 0.191 (one-card rig, 100 → 60 ms) | `standing_ring_check.py` § 1 |

The last row is the judge's addition and the strongest: three independent sources (offline rank records, two
real links, the delay rig) give the same m. The old model's m of 0.09 came from fitting e0 and m together. Code is
weaker: the measured slope is 0.074 against a replayed 0.045 (200 positions from one reply), so **code figures
below may be up to ~30% high**.

**Consequence.** By this law today's shape reads 33 tok/s prose at a ZERO loop, not 25. And nothing in `d` or `L`
is tied to one card's forward: `d` is the drafter's (4.5-4.7 ms a level, MEASURED: calls of 2 / 3 / 4 levels cost
10.0 / 14.5 / 19.0 ms), and `L` is the ring's traversal.

## 2. The design

| role | holds | does |
|---|---|---|
| Asker | nothing of the model | sends prompt token ids once, receives text |
| Stage 1..K | one slice each; K = the fewest stages the members' lends allow, never 1 | forwards back to back; each carries every level drafted since its last |
| Tail (stage K) | last slice, output head | draws the target's token per row (shared position-keyed noise), walks the tree, publishes verdicts as ids |
| Pacer | the drafter (≤ 1 GB) on its OWN card, a few ms from the tail | extends an 8-row beam one level every `d`; never waits |
| Ledger (processor-only nodes) | each stage's input rows for accepted tokens (5-10 KB a token) | serves a replacement stage's replay after churn |

- **Wire.** Asker → stage 1: ids, once (the asker's 130-290 ms becomes a first-token delay, never a per-token
  term). Pacer → stage 1: level ids (bytes). Stage s → s+1: hidden rows on K−1 links (MEASURED 120 KB a forward of
  23 rows). Tail → pacer and asker: verdict ids. The pacer sizes W and its own pace to the slowest uplink.
- **A miss.** The tail publishes the target's token with a new epoch; stages drop older epochs' rows; the pacer
  restarts. Cost: `L`, once. No rollback on dense or MoE layers.
- **Exact.** Only tokens the tail drew from the full model reach the asker; the reply equals a single-chain run row
  for row. Drafts decide speed only.
- **Recurrent hybrids (Qwen3.8-27B).** The same ring, but rounds as shipped: one check in flight, the file's own
  guess head. The ring's gain here is only the short loop and all layers on cards, which the plan's simulator
  already reads (43-47 tok/s at 20 ms). The flow form needs a per-row recurrent state and a tree drafter that does
  not exist for this family.
- **MoE.** Flow as dense, with one correction to the architect: a forward reads the union of the experts its rows
  choose, so rows are NOT nearly free (ASSUMED independent choices, 8 of 128: ~30 rows read most of the slice). The
  pacer must keep MoE levels narrow.
- **Each extra stage costs.** One more hop adds a leg and 1.5 forwards to `L`: 14B at a metro ring 45 → 39 tok/s
  on 8-12 GB cards, 72 → 58 → 49 on 24 GB cards (DERIVED). "Spread thin" is priced per stage, and the member's lend
  sets K. No stage ever holds the whole model.
- **Privacy wording.** An id-only asker hands its prompt to stage 1 in the clear, which is more than today's shape
  exposes. It is never "private". "Start and finish on this computer" keeps the asker on the ring and pays the
  asker's loop on every miss. Under private mode the ring is drawn only from `pool::scope::allowed_node_set()`.

### Why this passes "one reply stops at e0/τ", and where it still stops

- It passes because the pace becomes the drafter's level (4.7 ms), not the slowest stage's forward (41 ms), and
  because e0 rises as `L` falls. Neither needs new physics: a forward's rows cost 0.15 ms each against a 17.5 ms
  fixed cost (MEASURED), so a stage can carry every level drafted while it worked.
- It stops at `1 / (d + m × L)` with `L ≥ ~1.5 × (every layer once) + the legs`. At a zero loop: 14B ~57 tok/s on
  8-12 GB cards, ~84 on 24 GB cards; 70B ~34 on 24 GB cards (DERIVED). A verdict must pass every layer in order; K
  cards give K times the memory bandwidth to K forwards in flight, not to one row's traversal.
- **More cards add a logarithm, and less than the debate assumed.** The architect priced 16 lanes as one global
  beam of 128 rows (m 0.177 → 0.120, +33%). Lanes are separate chains and cannot trade rows; replayed as a
  partition, 16 lanes give m 0.156-0.160 and 4 lanes 0.169: **+11% and +4%** (DERIVED, `standing_ring_check.py` § 2). Lanes are
  not a speed design.
- **On equal cards a local user who also speculates wins.** The same law with no legs reads 115 / 93 / 51 tok/s
  for a 14B / 27B / 70B held whole on 24 GB-class cards (DERIVED; llama.cpp's chain drafter in rounds: ~86 / 67 /
  35). The ring's edge is over what members own (L1), over plain llama.cpp on a 24 GB card (L2 as defined), and
  for models no single machine in the swarm can hold. It is not an edge over the same trick run locally.

### What is not new, and what is a measured kill's kin

- A short ring with every slice on a card is the plan's own step; the new parts are the law (e0 is not a constant),
  the free-running pacer and the asker off the ring.
- "Moving the drafter off the head" simulated 1.01-1.10x. This design agrees for prose on today's pair (18.8 →
  ~19). The new reason it matters: once `L` is short, `d` is a third of the token (metro 14B: 13.7 ms of a 33 ms
  token as shipped), and on code it already is (42 → ~51 predicted).
- Lanes are "bigger sends" on other cards: the same logarithm, confirmed small. Worlds for the hybrid are kin to
  "rounds in flight" (+4-10%); they differ in branching at an unsure position, and a replay on drafter ranks gives
  ×1.27 tokens a miss at 4 worlds and ×1.74 at 16 (DERIVED, not the guess head's ranks).

## 3. Predictions

One reply, **prose**, tok/s · ×L1 · ×L2, by ring loop. L1 = whole on the best participating machine (MEASURED 11.2 /
5.3 / 35.5; none for the 70B). L2 = plain llama.cpp-class speed on 24 GB cards (ASSUMED ~65 / ~40 / ~100 / ~17.5).
All ring cells are DERIVED from the law, with ASSUMED swarm inputs:

| swarm | cards | lend | uplink | computers a reply |
|---|---|---|---|---|
| ~30 members | 8-12 GB class (448 GB/s) | 7 GB | 40 Mbit/s | K + 1 |
| ~300 | mixed (700 GB/s) | 12 GB | 80 | 4 × (K + 1) |
| ~3,000 | 24 GB class (1 TB/s) | 20 GB | 100 (the home range's top; the architect's 150 is corrected) | 16 × (K + 1) idle, K + 1 shared by 4 replies busy |

| tier | swarm | 20 ms | 60 ms | 130 ms | 290 ms |
|---|---|---|---|---|---|
| 14B | today, 2 computers (MEASURED) | — | — | 17.8-20.5 · 1.6-1.8× · 0.3× | 11.7 · 1.0× · 0.2× (308 ms) |
| 14B | ~30, K 2 | 45 · 4.0× · 0.7× | 34 · 3.0× · 0.5× | 23 · 2.1× · 0.4× | 14 · 1.2× · 0.2× |
| 14B | ~300 | 59 · 5.3× · 0.9× | 42 · 3.7× · 0.6× | 28 · 2.5× · 0.4× | 16 · 1.4× · 0.2× |
| 14B | ~3,000 idle | 72 · 6.4× · 1.1× | 49 · 4.4× · 0.8× | 32 · 2.8× · 0.5× | 17 · 1.6× · 0.3× |
| 14B | ~3,000 busy | 37 · 3.3× · 0.6× | 28 · 2.5× · 0.4× | 21 · 1.8× · 0.3× | 13 · 1.1× · 0.2× |
| 27B rounds | today (MEASURED) | — | — | 5.3 · 1.0× · 0.1× | 5.0-5.9 · ~1× · 0.1× (272 ms) |
| 27B rounds | ~30, K 3 | 27 · 5.0× · 0.7× | 17 · 3.1× · 0.4× | 10 · 1.9× · 0.2× | 5 · 1.0× · 0.1× |
| 27B rounds | ~300, K 2 | 39 · 7.4× · 1.0× | 24 · 4.6× · 0.6× | 15 · 2.8× · 0.4× | 8 · 1.5× · 0.2× |
| 27B rounds | ~3,000 idle, one world | 44 · 8.2× · 1.1× | 26 · 4.9× · 0.7× | 15 · 2.9× · 0.4× | 8 · 1.5× · 0.2× |
| 27B rounds | ~3,000 idle, 16 worlds (ASSUMED ×1.7) | 74 · 14× · 1.8× | 44 · 8.3× · 1.1× | 26 · 4.9× · 0.6× | 13 · 2.5× · 0.3× |
| 27B rounds | ~3,000 busy | 32 · 6.0× · 0.8× | 21 · 4.1× · 0.5× | 14 · 2.6× · 0.3× | 7 · 1.4× · 0.2× |
| 35B MoE | today, forced split (MEASURED) | — | — | 7.8-16 · 0.2-0.5× · ≤0.2× | — |
| 35B MoE | ~30, K 3 | 38 · 1.1× · 0.4× | 26 · 0.7× · 0.3× | 18 · 0.5× · 0.2× | 10 · 0.3× · 0.1× |
| 35B MoE | ~300, K 2 | 57 · 1.6× · 0.6× | 40 · 1.1× · 0.4× | 27 · 0.8× · 0.3× | 15 · 0.4× · 0.2× |
| 35B MoE | ~3,000 idle | 70 · 2.0× · 0.7× | 47 · 1.3× · 0.5× | 31 · 0.9× · 0.3× | 17 · 0.5× · 0.2× |
| 35B MoE | ~3,000 busy | 39 · 1.1× · 0.4× | 30 · 0.9× · 0.3× | 22 · 0.6× · 0.2× | 14 · 0.4× · 0.1× |
| 70B | today, 4 computers (MEASURED) | — | — | 1.5-3.0 · none · 0.1-0.2× | — |
| 70B | ~30, K 6 | 15 · none · 0.9× | 11 · 0.6× | 8 · 0.5× | 5 · 0.3× |
| 70B | ~300, K 4 | 23 · none · 1.3× | 18 · 1.0× | 12 · 0.7× | 7 · 0.4× |
| 70B | ~3,000 idle, K 3 | 33 · none · 1.9× | 25 · 1.4× | 17 · 1.0× | 10 · 0.6× |
| 70B | ~3,000 busy | 20 · none · 1.1× | 16 · 0.9× | 12 · 0.7× | 8 · 0.5× |

- **Corrections to the architect's table** (it read 86 / 104 / 41 for the 14B / MoE / 70B at 3,000 idle, 20 ms):
  lanes priced as a partition, uplinks capped at 100 Mbit/s, MoE rows priced by the experts they read (with the
  architect's 0.05 ms a row the MoE row reads 87 / 55 / 34 / 18).
- **Code**, idle, 20 / 60 ms (DERIVED, may be ~30% high): 14B 102 / 82 at ~30 and 159 / 120 at ~3,000 (7-11× L1,
  1.6-2.5× L2); 70B 40 / 31 and 77 / 61 (2.3-4.4× L2); MoE 95 / 74 and 150 / 115 (2.7-4.2× L1, 1.0-1.5× L2).
- **Today's pair with the pacer free on the far card** (130 ms): prose ~19 (unchanged), code ~51 against 42.
- **Sensitivities** (20 ms, DERIVED): rows at 10 KB instead of the measured 5.2 KB cost 3-7%; 10 Mbit/s uplinks cost
  the 14B 45 → 37; a guesser with half the misses gives 14B 72 → 115 and 70B 33 → 59 (ASSUMED effect).

**Where the split beats local.** Against L1, everywhere inside a region for the 14B, 27B and 70B, by 3-8×, from
the first ring of three or four cards. Against L2: the 70B by 1.3-1.9× and the 27B by about 1.0-1.1× on 24 GB-class
rings in a metro; code replies by 1.5-4×.

**Where it does not.** 14B prose against a 24 GB card (0.7-1.1×): one good card already reads it in 17 ms and the
ring adds legs. The MoE (0.4-0.7× L2; over its own L1 only inside a metro): its pass is short, so the wire is
nearly all of `L`. Anything at 290 ms. A busy swarm gives about half the idle figure. And any local user running
the same speculation on equal cards.

**The 20-30 tok/s floor.** 14B: held out to 130 ms. 27B: metro only on small cards (17 at 60 ms), region on 24 GB
cards. 70B: needs 24 GB-class stages and a ring of 60 ms or less; a ring of six 8 GB lends reads 15. MoE: held to a
region from ~300 members.

**The assumption it rests on most:** that a miss costs exactly one `L`, and that `L` falls to about 1.5 traversals
plus the legs when the ring is short. The law is confirmed between 60 and 308 ms on two-stage shapes; no
measurement exists below 60 ms or on a ring this document would build.

## 4. Idle, busy, fair

- **Idle.** A reply gets a ring and a pacer on the best cards near each other. Spare cards go first to more rings
  (other replies, or one user's task split into sub-replies whose context is the prompt: tool batches, one file
  each of a multi-file change, document map-reduce; exact, no stitch, linear in rings). Lanes and worlds come last.
- **Busy.** Stages merge conversations into shared forwards (MEASURED 3.6 ms a merged user; two conversations each
  ran as fast as one). Four replies a ring is the busy column.
- **Fair.** Deficit round robin on rows a stage forward, by conversation; the card owner's own request first. No
  credit gating.
- **From today's computers.** No two cards share a metro, so today the design can only be checked for mechanism
  (kill ladder steps 1-4). The 27B, 35B and 70B stay at their measured speeds until every slice is on a card. Each
  new member with a card in an existing member's region makes one more model's ring possible.

## 5. Kill ladder

| # | test | cost | kills the next step if |
|---|---|---|---|
| 1 | Read levels drafted, levels voided and the verdict delay from the 11-15 arms' flow logs (add the three counters to the flow's reply line if absent) | no node time, or one diagnostics-only change | voided levels are under half of drafted on prose (law: 74%), or the verdict delay is not within 180-260 ms (law: 222) |
| 2 | Re-dump drafter ranks against the 14B and 70B targets on the rig's reference copies; replay m(8, deep) | offline, one evening | prose m > 0.25 for either (every prose cell falls by a third) |
| 3 | `examples/tree_schedule_sim.py`: drafter as its own stage, levels a forward uncapped; it must first reproduce the five arms and the 308 ms run | simulation | the calibrated pair reads under 1.5× today's shape at a 20 ms ring |
| 4 | The delay rig (`flowline.sh`) swept to 40 / 20 / 0 ms, stock binary | one sitting, one card | ms-a-token does not keep falling ~0.18 ms per ms of loop (the plan's e0/τ floor is real) |
| 5 | Free-running pacer behind an env switch, A/B in one binary on the Thailand ↔ Australia 14B pair | first build, one evening | code stays under 46 tok/s (law: ~51 from 42); prose is expected flat and decides nothing |
| 6 | Block head, offline: fit on ~200k (final row, next 8 tokens) pairs from the 14B tail | offline, processor | top-1 at block position 4 under 0.70 (the only route past section 3's table closes) |
| 7 | The first two cards inside one metro: id-only entry, K 2, pacer on a third machine | first real ring | 14B prose under 30 tok/s at ≤ 30 ms (law: 40-45) |

## 6. First build step, and what not to build

**Build first (after steps 1-4 pass):** the pacer split, in `src/inference/pipeline/dsd_flow.rs` — the drafting
loop (`levels_a_call` becoming "levels since the last forward", `rows_a_level` sized from the measured uplink)
separated from the stage loop, with verdict epochs — and `engine_drafter.rs` so the drafter is a role a member
other than the head can host. Receiving side: `src/inference/split/executor.rs` and `kv_cache.rs` drop rows by
epoch. One additive wire message for level ids from a pacer that is not the head, gated at the sender on a new
`swarmllm_types::features` bit; every count a peer sends is checked against what was asked. Behind an env switch
for the one-binary A/B of step 5.

**Then:** id-only entry (`remote_generate.rs`, `split/entry.rs`, `token_embedding.rs`; `flow_possible` accepting a
ring the asker is not on), and ring selection in `scheduler/parallax.rs` and `parallax_allocator.rs` (fewest
stages, legs from measured pairwise delay, uplink and card class as inputs).

**Do not build yet:**
- Lanes and worlds (+4-11% measured by replay; revisit only as warm spares, and "stream spares" was already
  dropped once).
- The row ledger, until a real ring exists to lose a stage.
- The block head's training and distribution, until step 6's number exists.
- Per-row recurrent state for the hybrid, until a tree drafter for that family exists.
- Verdict taps, twins, canopy rounds, scout-and-stitch, a drafter guild (appendix).
- Task-level fan-out scheduling, until the request log shows what share of real requests has that shape.

## 7. Kill ladder step 1 — MEASURED 2026-11-21 (a derivation from older lines came first, and was wrong)

**Measured** (`~/swarmllm-gate/deep/fr/real_link.sh` TAG=readme305, 14:28-14:35Z; main `7633b9296` built for the
card; Qwen2.5-14B Q4_K_M split [0, 24) on this 8 GB card + [24, 48) on the Australian 12 GB card, least round trip
119 ms; the 0.5B drafter; `flowhalf` arm, two 160-token replies of each kind; 0 driver events). The flow's reply
line now carries three counters (`dsd_flow::Verdicts`):

| reply | tok/s | forwards | misses | `verdict_ms` | `miss_verdict_ms` | `voided_levels` | least answer ms | far least ms | far queue ms, most |
|---|---|---|---|---|---|---|---|---|---|
| prose 1 | 20.35 | 202 | 18 | 178 | 180 | 84 | 159 | 25 | 60 |
| prose 2 | 18.65 | 209 | 21 | 185 | 179 | 115 | 155 | 22 | 24 |
| code 1 | 46.33 | 91 | 0 | 180 | — | 0 | 160 | 25 | 67 |
| code 2 | 48.31 | 88 | 0 | 180 | — | 0 | 168 | 27 | 13 |

- **The verdict delay `L` is ~180 ms**: the least answer (155-168 ms — the round trip and a forward each end) plus
  ~20. Nothing waits at the far end to speak of. Test 2 (180-260 ms, law 222) sits on its lower edge (178-185).
- **A miss throws away ~5 levels of depth** (4.7, 5.5 a miss), not the law's ~16; misses come every 7.6-8.9 prose
  tokens (0.11-0.13 a token; the law's m 0.177) and not once in 320 code tokens.
- **Test 1 fails, and with it the law's central claim.** Of the depth the tree reached, 37-45% was thrown away
  (84 / 115 levels against 142 / 139 kept) — under half. In time, misses × `L` is ~20-24 ms of a 49-54 ms prose
  token and nothing of a ~21 ms code token: a reply's time is NOT mostly wrong guesses × their wait (the Summary's
  item 2 said ~40 of 53 ms).
- **The larger part is this node's own pace.** It was busy 98-100% (`waiting_ms` 0-35 a reply) at ~38-39 ms a
  forward — drafting ~13-14 + its 24 layers ~24-25 — while the far's least was 22-27 ms. A forward of ~24 rows
  deepens the tree ~1.1 levels in prose and ~1.8 in code (rows go best-first by path probability: wide where the
  drafter is unsure). So `d`, what a level of depth costs, is ~30-35 ms in prose (`L` ÷ levels a miss throws
  away, or a token's time less misses × `L`) and ~21 ms in code — not § 1's 13.7.
- **What that does to the ring** (DERIVED). Removing the wire (~120 of `L`'s 180 ms) leaves `L` ≈ 60 ms and
  ~37-42 ms a prose token: ~24-28 tok/s on today's pair (1.2-1.45× the 19.5 measured), code unchanged at ~47 —
  not § 1's 33, nor § 3's 3-8×. Past that, the head's pace binds both kinds: drafting is ~36% of its forward, so
  the pacer (the drafter off the head's card, § 2) is the lever, then rows that deepen the tree more a forward.
  Step 3's model (§ 8) read the pacer alone at +16% prose and +26-34% code on today's pair. So the next useful
  tests are step 4 (what a short loop really buys, on one card) and step 5's pacer A/B — not step 2.
  **§ 9 (step 4) found a loop costs ~3× what misses × `L` account for**, so the figures in this bullet are low: by
  its slope, today's pair at zero wire reads ~29-33 tok/s prose.
- **Derived first, and wrong** (kept so the error can be checked): from older lines without these counters,
  `forwards × 3 − emitted` read ~30 levels a miss and `L` ~400-460 ms (`examples/standing_ring_ladder1.py`), and
  commit `5c62eac9a` blamed ~250 ms of it on the far card serving our own forwards one at a time (#412). Both are
  withdrawn: a forward deepens the tree ~1.1 levels, not three, and the measured `L` is ~180 ms with no queue
  behind it. The knob sweep's "a forward adds three levels … ~6 tokens a 430 ms round trip"
  (`docs/invariants/scheduling.md`) rested on the same three-levels reading and now carries this correction.

## 8. Kill ladder step 3 — the schedule model does not reproduce the arms (2026-11-21)

`examples/tree_schedule_sim.py`'s `flow` with the five arms' MEASURED stage times (§ 7's table source; head drafting + its
layers, far 24-46 ms a forward, 130 ms, 0.15 ms a row; 40 seeds, 160 tokens), and a variant with an explicit
8-rows-a-level beam:

| arm | prose model (best-first / beam) | prose measured | code model | code measured |
|---|---|---|---|---|
| 4 rows | 21.6 / 20.7 | 15.06-15.33 | 43.5 / 41.4 | 40.48-42.43 |
| 8 × 3 (default) | 21.4 / 20.8 | 17.61-19.93 | 42.0 / 40.0 | 40.94-45.43 |
| 16 rows | 20.0 / 18.6 | 15.72-17.89 | 31.1 / 31.4 | 33.57-34.72 |
| 4 levels | 22.6 / 21.3 | 16.85-19.38 | 43.5 / 41.2 | 40.23-41.53 |
| 308 ms (US card) | 13.9 / 13.0 | 11.73-11.78 | 28.2 / 26.1 | 26.85-27.16 |

- **Calibration FAILS on prose** (11-42% high in four of five arms); code is within ~±9%.
- **The model's verdict delay is 224 ms** (head layers 26 + wire out 72 + far 30 + wire back 65 + the head
  finishing its forward 30); **measured, ~180 ms** (§ 7). Its far queue is zero in every arm (the far, 24-46 ms a
  forward, is faster than the head, 34-58 ms) — as measured.
- Its ranks are drawn iid from fixed shares (prose .745/.10/.04/.02), not the replay dumps; it misses every 6-9
  prose tokens, tonight's replies every 7.6-8.9. Against tonight's default arm (prose 18.65-20.35, code
  46.33-48.31) it reads prose 2-15% high and code 13-17% low; against the older five arms, prose 11-42% high.
- **So the step does not rule yet.** Its pacer figures (indicative: today's pair +16% prose, +26-34% code; a 20 ms
  ring 30-42 prose, 59-87 code) need a model whose tree deepens as the real one does (~1.1 levels a forward in
  prose, ~1.8 in code, § 7) and whose code is not low. The scripts stay out of the repo until a calibration holds.

- **Recalibrated the same night on the counters (§§ 7, 9)**: the tree is now grown as `SplitModel::grow_flow`
  grows it (`src/inference/split/draft.rs`: one best-first queue of children over the live tree, a call of three
  levels of eight, a row's children only one level after it is added, nothing past 128 live rows) — which gives the
  measured ~1.1 levels a forward. Fitted on the 14B pair's default arm (prose: top-1 share 0.587, absent 0.081;
  code: top-1 0.936), it reproduces that arm — 19.55 prose / 47.56 code tok/s, 0.75 / 1.83 tokens a forward,
  verdict delay 176 ms UNFITTED against 178-185 — but code misses 6 a reply against 0. Fitted on step 4's 100 ms
  point only, it does NOT reproduce the shorter loops: its tokens a forward stop rising at 1.54 from 40 ms down
  (measured 1.42-1.47 / 1.50-1.51 / 1.84-1.88), so 0 ms reads 27.7 against 33.0-33.6 (−17%); no rank profile on a
  grid fits all four loops within 14%. The real tree wastes more rows to a longer wait than the queue model does;
  candidates left: rebases at the 512-row cap (5 a reply at 100 ms, 3 at 0) and the copied rows (66 against 102
  a reply). Ruled out from the rig's lines: the guard (it never narrowed), and the 40/20 ms points (they are the
  rig's own delay — one line against the measured `verdict_ms`, § 9).
- **Indicative only** (the rule withholds a ruling): on the 14B pair today's shape at a 20 ms loop reads 30.9 prose
  / 61.0 code (§ 9's slope gave ~29-33 prose), and the pacer there 32.3 / 92.9 (35.3 / 95.8 with the far's queue
  merged) — ~1.05× prose, ~1.5× code over the same loop without it; most of a prose gain is the short loop.
  Scripts in `~/swarmllm-gate/deep/step3/` (moved out of the night's scratch folder 11-22: `ring_sim3.py`, `fitall.py`, `evalcal.py`, `grid.py`,
  `pacerA.py`), kept out of the repo until step 4's shape is reproduced.

## 9. Kill ladder step 4 — a shorter loop on one card, MEASURED (2026-11-21)

`~/swarmllm-gate/step4/flowline.sh` (a copy of `~/swarmllm-318/flowline.sh` writing to its own folder), 14:43-15:05Z,
Windows up 3.1 h, 0 driver events; main `7633b9296` built for the card. Qwen2.5-Coder-7B Q4_K_M split 4 + 4 parts
over two processes on this one 8 GB card (they share it, gotcha #759), the 0.5B drafter beside the first, every
packet between them delayed for the round trip named; the rig's one prose prompt, 160 tokens,
`SWARMLLM_FLOW_COPY_STORE=0`; the flow (`SWARMLLM_DRAFT_TREES=1`) against the chain stream (`=0`), ABBA, three asks
an arm (the first carries the load and is left out). Flow figures are the node's own reply line:

| loop | flow tok/s | `verdict_ms` | least answer | misses a reply | `voided_levels` | forwards | tokens a forward | stream tok/s (client) |
|---|---|---|---|---|---|---|---|---|
| 100 ms | 21.27-21.90 | 188-191 | 157-160 | 7-8 | 32-38 | 127-130 | 1.23-1.26 | 21.68-23.92 |
| 40 ms | 25.46-25.90 | 140 | 96-106 | 5-6 | 18-31 | 109-113 | 1.42-1.47 | 28.12-29.88 |
| 20 ms | 23.86-26.62 | 134-139 | 72-80 | 8-12 | 28-58 | 106-107 | 1.50-1.51 | 27.87-32.77 |
| 0 ms | 32.96-33.57 | 76-78 | 57-63 | 8 | 35-36 | 85-87 | 1.84-1.88 | 18.15-19.31 (a), 36.27-39.70 (b) |

- One 40 ms flow arm (b) is left out: its drafter loaded on the processor (`reason="not_enough_vram"` — a guest
  takes only memory that stands free, and the other process held it), drafting 123 ms a call, 12.7-12.9 tok/s.
- **The kill test passes**: ms a token kept falling, 46.3 → 38.9 → 39.4 → 30.1, ~0.16 ms per ms of loop over
  100 → 0 (the test asks ~0.18), with a flat stretch between 40 and 20 ms. **The flat stretch is the rig's, not
  the flow's**: its verdict delay barely fell there (140 → 136 ms). Against the MEASURED delay the four points lie
  on one line — ms a token = 19.1 + 0.144 × `verdict_ms`, each within 0.6 ms — and the guard never narrowed
  (8 × 128 at start, least and end; no cuts) at any loop.
- **But not by the law's mechanism.** The head's forward cost the same at every loop (~56 ms: drafting ~31 — this
  card serves both halves — and its layers ~25; `waiting_ms` 0); misses stayed ~8 a reply, and a miss threw away
  ~4.5 levels at 100 ms and at 0. Misses × `L` fell only from ~9 to ~4 ms a token while a token's time fell 16.
  What rose is how many of each forward's ~24 rows the reply kept: 1.24 tokens a forward at 100 ms, 1.86 at 0. A
  longer wait makes the tree spend its rows deeper, where fewer land on the reply, between misses too. So the slope
  is not `m`: on this rig `m` is ~0.05 a token against a slope of 0.16; on the 14B pair 0.11-0.13 (§ 7) against
  § 1's 0.18. § 1's "slope equals m" check matched by coincidence.
- **For the ring**: a short loop is worth more than § 7's arithmetic, which priced only misses × `L`. Applied to
  today's 14B pair (DERIVED, prose only — the rig asked no code): 119 ms of wire × 0.16 ≈ 19 ms off a 49-54 ms
  prose token → ~29-33 tok/s at zero wire. The head stays busy at every loop, so beyond that the pacer is the
  lever; step 5 (its A/B) is the first build, after the release.
- The chain stream beat the flow on this rig at 100, 40 and 20 ms (as on 11-15: two processes on one card favour
  the stream); at 0 ms its two arms disagreed (18-19 against 36-40 tok/s) and are not read.

## 10. Kill ladder step 5 — the pacer cannot be A/B'd on today's computers (2026-11-22, not built)

Step 5 asked for a free-running pacer behind an env switch, A/B in one binary on the Thailand ↔ Australia 14B pair.
A pacer is worth something only on a device that is not busy with a slice, and today's pair has none:

1. **This card, beside the head's layers** — the cheapest form: start the next drafter call while this forward's
   layers run (the daemon awaits them in series, `FlowRun::run_with`: `grow().await`, then `send().await`). Already
   MEASURED as a loss on 2026-10-22: the drafter and the head's layers are two processes, and two processes
   time-slice one card — drafting 31 → 46-49 s and the near half 10-11 → 19-21 s a 400-token reply (gotcha #759);
   `arch-scheduling.md` forbids it. Prior art agrees: AMUSD (arXiv 2410.17375), SwiftSpec (2506.11309), PipeSpec
   (2505.01572) and Speculative Speculative Decoding (2603.03251, ~30% over tuned speculation on average) all put the
   drafter on hardware of its own.
2. **This computer's processor**: the 0.5B drafts a call of three levels of eight in ~123 ms there (§ 9's left-out
   arm) against ~39 ms a forward on the card. That pacer would be three times slower than the head it frees.
3. **The far card**: a peer's computer runs only published releases. A pacer there needs an additive wire message
   for level ids and a feature bit, shipped first, and it would again share a card with a slice (#759) unless it ran
   inside the slice's own process.
4. **A third card a few ms from the tail**: none in today's swarm.

The single-card levers on the head's ~39 ms forward are spent too. The drafter's tree levels are already captured as
graphs (−26%, `docs/invariants/inference.md` § "A tree level on a card"). Fewer layers on the head (`flow23`/`flow20`,
2026-11-16) gained nothing (FUTURE_WORK #411: at a ~130 ms link the flow runs at its tokens a round trip, not its
slowest stage's forward). Capturing the head's own ~25-row forward (vLLM's and SGLang's padded verify graphs) would
hide nothing. Its launches are issued in ~6 ms (`forward_ms` of the 11-21 sitting's 24-row forwards over 26 layers:
the host's side, before any copy-out), while the card needs at least ~17 ms to read those layers' 4.7 GB. That
leaves `near_ms` ~24.5 a forward, against the far's 22-27 on its own card, so the card itself sets the pace.

**What it means.** The plan's own step-5 prediction for today's pair was prose flat and code ~51 tok/s, against the
46-48 measured since (§ 7). That is under +10%, on the reply kind already over the 20-30 bar. The pacer's case is a
ring with a spare card beside the tail (step 7's shape), not this pair. On this pair, prose moves only through fewer
misses or more of each forward's rows kept (§ 9). Step 2's real ranks against the 14B are what step 3 needs to price
those (§ 8: its ranks were drawn from fixed shares). **Next: step 2, then step 3 recalibrated on its ranks.** (Done: § 11.)

## 11. Kill ladder steps 2 and 3 — real ranks fix the misses; no row policy moves prose on today's pair (2026-11-22)

**Step 2, MEASURED offline** (the processor, llama.cpp through `llama_cpp` 0.3.16; the 14B and the 0.5B instruct
drafter rebuilt from this node's shards for the run and deleted after):

- Two dumps are in `~/swarmllm-ref/spec/`, both in `gumbeldump.py`'s layout:
  - `greedydump.py` holds the 14B's own greedy replies to the rig's prompts and `gumbeldump.py`'s six.
  - `replaydump.py` holds the rigs' saved replies, with both models teacher-forced over them. The target's own
    top word is the reply's word at 151-158 of 160 positions.
- Where the 14B's word sits among the drafter's guesses (first, second, third, fourth; absent from its top 8):

  | | positions | 1st | 2nd | 3rd | 4th | absent from top 8 |
  |---|---|---|---|---|---|---|
  | prose | 1,792 | 0.622 | 0.151 | 0.063 | 0.033 | 0.079 |
  | code | 400 | 0.845 | 0.095 | 0.028 | — | 0.005 |

- **m(8, deep)**, the chance a level loses the target's word from an 8-row beam:
  - prose 0.221, which **passes** the 0.25 kill line, though it is above the law's 0.177;
  - code 0.075-0.080.

  The 70B has no file here and is not measured.
- The fitted ranks were wrong exactly where they mattered:
  - The 14B's code first-guess share was fitted at 0.936; the real figure is 0.845.
  - The one-card rig's absent share was fitted at 0.001; the real figure is 0.064.
  - Step 4's rig asked another prompt (household appliance notes). There the drafter's first guess is right 71-73%
    of the time, against 57-61% on the 14B pair's story.

**Step 3, replayed on the rigs' OWN replies** (`~/swarmllm-gate/deep/step3/cal5.py`; ring_sim3's tree with
ReplayTruth; nothing fitted):

| arm | tok/s, sim vs measured | tokens a forward, sim vs measured | misses, sim vs measured | `L`, sim vs measured |
|---|---|---|---|---|
| 14B pair, prose | 21.29 vs 18.65-20.35 (+9%) | 0.82 vs 0.77-0.79 | 18.5 vs 18-21 | 176 vs 178-185 |
| 14B pair, code | 53.33 vs 46.33-48.31 (+13%) | 2.05 vs 1.76-1.82 | 1 vs 0 | 176 vs 180 |
| one card, 100 ms | 23.26 vs 21.27-21.90 (+8%) | 1.30 vs 1.23-1.26 | 8.0 vs 7-8 | 185 vs 188-191 |
| one card, 40 ms | 29.07 vs 25.46-25.90 (+13%) | 1.62 vs 1.42-1.47 | 6.5 vs 5-6 | 125 vs 140 |
| one card, 20 ms | 29.33 vs 23.86-26.62 (+16%) | 1.63 vs 1.50-1.51 | 5.8 vs 8-12 | 105 vs 134-139 |
| one card, 0 ms | 29.13 vs 32.96-33.57 (−12%) | 1.62 vs 1.84-1.88 | 7.0 vs 8 | 85 vs 76-78 |

- **Misses now match at every point**, so the made-up ranks were their whole error.
- **What is left is the tree's growth.** The sim's tokens a forward stop at ~1.62 from 40 ms down, where the rig's
  reach 1.86 at 0 ms, and it reads 4-16% high elsewhere.
- **Ruled out: the drafter's ratings off the reply.** One step off the path, the drafter's top word has probability
  0.636 after its first wrong guess and 0.625 after its second (796 positions each, `offpath.py`), against 0.648
  on the path; its top-8 mass is the same. So ring_sim3's use of the on-path ratings there holds on average.
- **Still candidates**: the rebases at the row cap, and the copied rows (§ 8).

**Row policies priced at the 14B pair's loop** (`pol5.py`, same replies; change against the default 8 rows × 3
levels, which reads prose 21.3 / code 53.3):

| policy | prose | code |
|---|---|---|
| 4 rows | −4% | −6% |
| 6 rows | −1% | −3% |
| 12 rows | −2% | −5% |
| 16 rows | −1% | −9% |
| 2 levels | −5% | −19% |
| 4 levels | +4% | +5% |
| 5 levels | +5% | +9% |
| 64 live rows | −1% | −12% |
| 256 live rows | 0 | 0 |

**Checked against the 11-15 arms' measured order (§ 8), the sim gets the direction and a third to a half of the size:**

- **4 rows**: measured about −18% on prose.
- **16 rows**: measured about −10% on prose and −23% on code.

The sim prices a row at 0.15 ms and a call's drafting at a fixed 14 ms whatever its levels. So its only winners,
+4-5% on prose from more levels a call, are exactly the policies it underprices. And the 11-15 4-levels arm measured
no gain: 16.85-19.38, against 17.61-19.93 for the default.

**Ruling.** On today's 14B pair no row policy moves prose by more than a few percent. Prose moves with the loop
(§ 9: ~0.144 ms a token per ms of verdict delay) and with the drafter's misses. Code is already over the bar. The
next prose lever this pair can test is **a stronger prose drafter**, priced the same way: its ranks on these replies
(`replaydump.py`) and its card time per level. The open question is whether fewer misses pay for a slower level.

**Priced the same day: the 1.5B instruct drafter against the 0.5B on the 14B pair's own replies**
(`gumbeldump_q14-rig-d15`, `drafter5.py`):

- **Ranks, prose**:

  | drafter | top word right | absent from top 8 | m(8, deep) |
  |---|---|---|---|
  | 0.5B | 0.600 | 0.103 | 0.241 |
  | 1.5B | 0.691 | 0.047 | 0.166 |

  On code the top word is right 0.881 → 0.906.
- **Misses a prose reply in the model**: 18.5 → 7.0.
- **Speed**: a 1.5B level costs 8.7 ms in the flow against the 0.5B's 5.3 (faster_than_local § 10, item 10's 11-01
  read). Priced at that ×1.6 a call, with 0-6 of this card's layers handed to the far for its 1.6 GB, the model
  reads **prose +4..+19% and code −6..+7%**. With only two replies the model swings with timing, and it underprices
  a slower call (above). So this is a hypothesis the real pair must settle.
- **History**: the 10-30 real-link arms were run before levels were graphs. The 1.5B then read prose 9.2-9.8 against
  the 0.5B's 6.6-8.0.
- **Next, on the real pair** (it needs AU): `real_link.sh` arms `flowhalf` (0.5B) against `flowq:DRAFTER=` the 1.5B,
  and the 3B, ABBA. The live node holds both.

## Appendix — every concept raised

| concept | verdict | reason |
|---|---|---|
| Standing ring (inventor-swarm A; architect) | ADOPTED, corrected | the only shape whose law fits five arms, the 308 ms run and two measured slopes; reach is 3-8× L1, 1-1.9× L2 on big models |
| Free-running pacer off the head | ADOPTED as a part | nil for prose on today's pair (the measured kill stands there); pays once `L` is short and on code |
| Id-only asker, ring where cards are dense | ADOPTED | turns the asker's distance into a first-token delay; must never be called private |
| Rounds on the ring for the 27B | ADOPTED | the plan's own all-card short ring; nothing new beyond placement |
| Lanes (swarm D), Fan (idle B), Canopy (ceiling C1) | DEAD as speed, kept as spares | one logarithm three times; as a partition 16× the cards buys +11% |
| Linear worlds for the hybrid | UNPROVEN | replay on drafter ranks gives ×1.27-1.74; the guess head's ranks are unmeasured; kin to "rounds in flight" |
| Braid, phase-shifted twins (idle A) | DEAD | rows are nearly free, so one card carrying M forwards' rows does a twin's job; twins remove only queueing (~13%) |
| Drafter guild (ceiling C2a) | DEAD | measured today: 1.5B + 3B covers 0.969 with 8 rows, the 3B alone 0.970 |
| Feature-fed block head at the tail (ceiling C2b) | OPEN, one number decides | the only lever on both `d` and `m`; the 27B's shipped head keeps 38% |
| Ladder, a mid model as a rung (swarm B) | DEAD | a 3B drafter is top-1 0.774 against 0.755 for a 1.5B; a second ring for two points |
| Verdict taps (ceiling C3, swarm B) | WOUNDED, not built | 70B only, +20-25%, close kin to the measured middle-state kill |
| Scout and stitch (idle C) | WOUNDED | needs under 3% divergence a token; the target's own 3-bit copy on an identical prefix diverges 11% |
| Fan-out of a reply into sections (swarm C) | WOUNDED, lossy | measured here at 1.2-2.0× on six parts; the outline is serial |
| Exact fan-out where the prompt is the context (adversary 3) | KEPT as scheduling | the one place cards add linearly; a task-level feature over the same ring |
| Thin slices for their own sake | PRICED | each extra stage costs 10-20% of a reply at a metro ring |
