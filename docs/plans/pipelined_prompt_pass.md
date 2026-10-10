# A split reads its prompt in pieces, every machine at once (FUTURE_WORK #171)

**Status: built 2026-10-10** for the splits this node leads (`pipeline::prompt_chunks`);
`split_speculation.md` Phase 3 and `wan_parallel.md` §4 name it; this is the design as built.
Measured below; what is left is in `docs/FUTURE_WORK.md` #171.

## The cost today

A split's prompt pass runs one machine at a time: the head reads every position of the prompt
through its layers, the hidden states cross, then the next segment reads them all, and so on.
Measured 2026-10-10 on Llama-3.2-3B split on processors (`split_rig.sh splitcache`, 2,274-token
prompt): 60.5 s, of which the local head (21 layers) ~41 s and the peer (7 layers) 19.6 s, one
after the other. Read in pieces, the peer reads piece k while the head reads piece k+1: the pass
approaches its slowest stage plus one piece of the others — here ~1.4×, a balanced two-machine
split ~2×, and on a long link the transfer of each piece overlaps the compute too.

#10 already spares a conversation's later turns most of their prompt (they resume from what each
segment kept); this is for the turns it does not cover — the first, and a one-off long prompt
(a document, a code review) — and for the new part of every turn.

## What the systems with more scars do

- **Sequence pipeline parallelism** (Medha/Mnemosyne, arXiv 2409.17264): consecutive chunks of
  ONE long prompt pipelined across stages, for time-to-first-token. Their measured point that
  carries over: chunking costs little even at small chunks (64 tokens; ~40 the efficient minimum
  with grouped-query attention), so the chunk size is chosen for overlap, not for efficiency.
- **llama.cpp pipeline parallelism** (#6017): a prompt batch split into micro-batches
  (`n_ubatch`) that run on GPU k+1 while GPU k reads the next; it keeps several INPUT COPIES in
  flight (`LLAMA_SCHED_MAX_COPIES`, default 4) — overlap costs memory for the pieces in flight.
- This repo's own stream of checks (`dsd_stream`, `forward_streams`): several forwards of one
  request in flight to a peer, run there in their number's order (PipeInfer's non-overtaking
  rule), each answer echoing its number. A piece of a prompt is one more such stream.

## The design

1. **A piece names its pass.** `LayerForward::prompt_span = Some(PromptSpan { start, end })`:
   this forward (`sequence_num` 0) is the part of a prompt pass covering positions
   `index_pos..index_pos + n`, of a pass that covers `start..end`. Wire trailer `0x0E`
   (`start u32 | end u32`), after `0x0D`, written by the one function the plaintext frame, the
   encrypted frame and the AAD all call; sent only to a peer advertising
   `features::PROMPT_CHUNKS` (bit 21).
2. **The worker.** The FIRST piece (`index_pos == start`) does what a prompt pass always did:
   clears the request's cache, restores #10's stored opening, and is admitted for the WHOLE
   span (`end - start`) — gotcha #447: a prompt admitted piece by piece fills a card until
   attention's transient allocation fails mid-pass. A LATER piece clears nothing and is admitted
   nothing; it must continue exactly where the cache ends, or it is refused (a worker that lost
   the conversation mid-pass). A piece's input is always token ids on the first segment. #10's
   store runs on the FINAL piece only. The last segment samples every piece; the coordinator
   keeps the final one's token.
3. **The coordinator** (`pipeline::prompt_chunks`), entered from `forward_through_segments`
   and #10's kept pass alike. It tokenizes the prompt (`standalone_tokenizer`), cuts the
   positions to compute into pieces, and runs one driver per segment, concurrently: a driver
   takes its inputs in order from the segment before, runs its own segment on each (this
   node's directly; a peer's as a numbered stream — its own attempt tag per segment, so the
   coordinator's waits never meet — with at most two pieces out, so the peer never idles a
   round trip between pieces), and hands each output to the next driver as soon as it exists.
   Unchained: every piece comes back here (a chain answers only from its tail).
4. **Where it applies.** A prompt pass of a plan with two or more segments, none of them a
   tensor-parallel group and no machine twice (a worker holding two segments of one request
   would cross their replies, gotcha #180 — the boomerang stays whole), every peer advertising
   the bit, THIS node on every boundary between segments (pieces come back here; between two
   peers a whole pass is chained straight across, and a far coordinator relaying every piece
   could cost more than it saves) — unless the pass keeps its prompt for #10, which is never
   chained, so its relay is already the whole pass's cost — no image or pre-embedded input, a tokenizer here, and at
   least two pieces' worth of positions to compute. `SWARMLLM_PROMPT_CHUNKS=0` turns it off (the A/B arm);
   `SWARMLLM_PROMPT_CHUNK_TOKENS` sets the piece size.
5. **A failure is the old pass, once.** Any piece failing (but the request being cancelled)
   stops new pieces, waits for the ones out to settle — so none of them reaches a worker beside
   the pass that follows — and runs the prompt pass as it always ran, from its first position:
   that pass clears each segment's cache and keeps its failover. The pieces cost the time they
   ran; the reply is the old path's.
6. **What it gives up, for now.** A piece's input is not kept for a stand-in's replay, so a
   request whose prompt was read in pieces is continued by the router if a segment fails
   mid-reply (#236) rather than replayed onto a standby — the same limit #10's resume accepted.
   No per-piece speed sample is recorded (a piece's wait includes its predecessor's queue);
   unpieced passes keep teaching the router.

## How to measure it

`split_rig.sh splitcache` (two turns; turn 1 is a whole prompt pass) with the two nodes on
disjoint halves of the machine (`CPUS_A` / `CPUS_B`, `taskset`), arms inside one binary:
`SWARMLLM_PROMPT_CHUNKS=0` vs on. Mechanism check: A's log names the pieces
(`a prompt pass in N pieces`), each segment's log the pieces it ran. Replies byte-identical to
the unpieced arm at temperature 0 is the expectation, not the judge — a piece boundary changes
the order of a reduction; a moved reply is scored against llama.cpp.

## Measured (2026-10-10, `~/swarmllm-171/ab.sh`)

Llama-3.2-3B Q4_K_M, A shards 0-1 (layers 0-12) and B shards 2-3 (12-28), processor only,
`CPUS_A=0-7 CPUS_B=8-15`, a release CPU build of this change, arms alternating in one binary:

| Arm | Turn 1 pass (2,274 tokens) | Turn 2 (resumed from 2,240) |
|---|---|---|
| whole (`SWARMLLM_PROMPT_CHUNKS=0`) | 71.3 s | resumed, hit on B |
| in pieces (5) | 58.2 s | resumed, hit on B |
| whole | 69.6 s | resumed, hit on B |
| in pieces (5) | 57.0 s | resumed, hit on B |

1.22× on one box; replies byte-identical in every arm, both turns. The two nodes are pinned to
different cores but share one memory bus, and a processor reads a model at the bus's speed: B's
pieces took 13.3, 11.6, 9.3, 7.6 and 4.5 s — falling where later pieces attend to more positions
— as A finished its part. On two machines B would read every piece at the last ones' speed, and
the pass would approach B's own time (~1.8× here). Not measured: two real machines, a card, a
real link.

Split three ways (`~/swarmllm-171/ab3.sh`: A shards 0-1, B shard 2, C shard 3 — layers 0-12,
12-21, 21-28 — on CPUs 0-5, 6-10 and 11-15), a kept pass relayed through A between B and C:

| Arm | Turn 1 pass | Turn 2 |
|---|---|---|
| whole | 76.8 s | resumed, hits on B and C |
| in pieces (5) | 46.5 s | resumed, hits on B and C |
| whole | 70.1 s | resumed, hits on B and C |
| in pieces (5) | 47.0 s | resumed, hits on B and C |

1.49-1.65×, replies byte-identical; the gain grows with the stages, as the ceiling does.

