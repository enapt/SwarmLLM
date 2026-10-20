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
   at that region's P.
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

PipeInfer 2407.11798 · SpecPipe 2504.04104 · Judge Decoding 2501.19309 · EAGLE-3
2503.01840 · MoE-Spec 2602.16052 · cost-aware MoE speculation 2607.12696 ·
decentralized speculative decoding 2511.11733 · Speculation at a Distance
2606.25091 (over a WAN the gain is mostly capacity; latency gains need a short
loop, which is why item 5 comes first) · DeepSeek-V3 MTP 2412.19437 (85-90%
second-token acceptance).
