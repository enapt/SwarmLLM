# A split keeps its conversation's prompt across turns (FUTURE_WORK #10, the split half)

**Status: built and measured 2026-10-09 (`pipeline::split_prompt_cache`; turn 2 of a 3B split ~9x faster, below).** The local half shipped in v0.3.208 — a whole model on
this node prices and reuses its own cached prompt (`scheduler::cached_prefix`, the worker's
`PrefixCache`). This is the other half: a model SPLIT across computers.

## The cost today

A stateless client — an agent, which re-sends the whole conversation every turn — makes every
turn re-read the whole prompt through every segment of the split. FUTURE_WORK #10's figures: on a
14B, ~79 MB of hidden states per hop and ~50 s of wire per turn at 25 Mbps, before any compute;
one field report spent 847 s re-reading. Each segment's cache for the request is released when
the request ends (`router::release_request_on_peers`, `release_caches_of_cancelled`), and the
next turn is a new request with a new id, so nothing carries over.

## What the systems with more scars do

- **vLLM automatic prefix caching**: the prompt is cut into fixed blocks, each block's key hashes
  its tokens AND its parent block's key (so a key names the whole prefix up to it), and a new
  request reuses every leading block whose key it finds. With pipeline parallelism the central
  scheduler's block table serves every stage.
- **vLLM-Ascend KVPP** ([docs](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/kvpp.html)):
  each pipeline stage keeps and allocates its own cache, never shared across stages — the shape a
  split here has to take, since each segment holds only its layers' K/V.
- **Prefix-aware routing in a P2P network** ([arXiv 2606.17059](https://arxiv.org/abs/2606.17059)):
  each node keeps its own cache index and peers keep stale estimates — "stale metadata only causes
  cache misses, not incorrect outputs". An optimistic protocol with a refusal is enough.
- **SGLang's trap** (sgl-project/sglang#26263, read 2026-09-13): a routing key built from the first
  message made unrelated conversations look alike. Key on the full prompt, block by block.

## The design

1. **The coordinator decides.** It tokenizes the prompt (`SharedState::standalone_tokenizer` —
   no tokenizer, no resume) and computes the block chain, vLLM's rule, with a KEYED hash: a
   segment holds keys it cannot invert, so the middle of a boomerang learns nothing new about the
   text, and two coordinators' entries never meet. *As built*, the secret is random per process,
   not derived from the identity key: the belief table lives in memory and dies with the process
   anyway, so a key that outlived a restart would only name entries nobody believes in, and
   nothing is derived from the identity key that does not need to be.
2. **What a prompt pass carries** (a new forward trailer, gated at the SENDER on a new
   `features` bit): the keys of every full block of this prompt, and `resume_at` — the longest
   block-aligned prefix, short of the whole prompt, that EVERY segment of this plan is believed
   to hold. The belief is the coordinator's own table: (segment node, layer range, model) → the
   keys that segment said it stored after earlier turns. No probe round trip: a hit costs
   nothing extra, a miss one prompt pass.
3. **What a segment does with it.** A prompt pass clears the request's cache first
   (`model_worker`, `sequence_num == 0`), so the hydrate goes AFTER that clear: copy positions
   `0..resume_at` from its store by key into the request's cache — or refuse with a typed reason
   (a cache miss: evicted, restarted, never stored). Then compute `resume_at..n` as usual. The
   head is sent the prompt's TOKEN IDS from `resume_at` (today it is sent the text and tokenizes
   it at position 0). At the end of the prompt pass the segment stores the prompt's full blocks
   under their keys (byte-capped LRU — the worker's `PrefixCache` budget and its release path)
   and its answer says how many blocks it stored; the coordinator records that.
4. **A miss is a retry, never a penalty.** The coordinator forgets that segment's entries and
   sends the prompt pass again from 0 — once, in the same attempt; it is not the peer's fault.
5. **Where it applies**: a plan of two or more segments whose every remote segment advertises
   the bit, a prompt of at least two blocks. Plain splits, the DSD rounds and the stream all
   run their prompt pass through `forward_through_segments`, so one place covers them.

## What has to be true before building

- The worker's `PrefixCache` is keyed by `SplitModel::kv_model_key` (layer range included), so a
  segment's entries are its own — but `lookup` takes TOKENS; a segment past the first gets hidden
  states, so it needs a lookup by key chain.
- Snapshotting a segment's cache copies its K/V (the local path's `insert_from_kv` does the same);
  the byte budget must count it, and `kv_budget::admit_prompt` must be able to release it. **So
  the entries go IN the worker's `PrefixCache` — keyed entries beside its token-keyed ones — and
  never in a second store**: gotcha #440 is a cache whose snapshots nobody charged, and a long
  prompt's live cache spilled to host memory at 3-5 tok/s with nothing refused or logged. In the
  same cache they inherit the charge (`KvOccupancy::external_bytes`), the snapshot's sizing
  before the copy (`plan_snapshot`) and the eviction before a refusal (`claim_room`'s evictor).
- A prompt pass sent in chunks (`chunk_meta`) and the failover paths: a stand-in segment holds no
  entries — it must be planned with `resume_at = 0`, which the belief table gives for free.

## How to measure it

A rig turn pair (`split_rig.sh`, a new mode beside `cache`): two turns of an agent conversation
through A → B; turn 2's prompt pass is timed and its positions counted (expect it to send only
the new turn's tokens), replies scored against llama.cpp, A/B inside one binary with an env
switch, and a forced eviction on B (the refusal → one retry from 0).

## Measured (2026-10-09, `split_rig.sh splitcache`, `~/swarmllm-10/run.sh`)

One release CPU build of main (0.3.232 + this change), Llama-3.2-3B Q4_K_M split A → B on the
processor, live node stopped, under the safety kit. Turn 1: a ~2,240-token system prompt (an
agent's rules) and a short question; turn 2: the same conversation one exchange longer (2,306
prompt tokens). Arms in order, the env switch the only difference:

| Arm | Turn 2 resumed from | B restored | Turn 1 | Turn 2 |
|---|---|---|---|---|
| on | 2,240 | 1 hit | 69.1 s | **7.1 s** |
| off (`SWARMLLM_SPLIT_PROMPT_CACHE=0`) | — | — | 69.1 s | 65.2 s |
| on (again) | 2,240 | 1 hit | 72.6 s | **8.2 s** |
| miss (B restarted between the turns) | 2,240, then 0 | refused | 67.7 s | 70.2 s |

Turn 2 is ~9x faster when every segment still holds the opening; a miss costs what the old path
cost plus one refused forward. Replies scored against llama.cpp on the same 2,306-token
conversation (`score_against_reference.py` now takes a request body as its prompt): off and miss
are byte-identical, 31/32 rank-1; on picks " of" at a 0.067-logit near-tie ("Here are two of the
rules" vs "Here are two rules") and scores 29/32 rank-1, worst rank 2, largest gap 0.132 — the
same as off. The restored opening was computed in turn 1's prompt chunks, not turn 2's, so its
rounding differs; a correct reply moved at a near-tie, never a broken one (ranks stay ≤ 2).

Not measured here: a real WAN link, where the saving is the wire time as well (~50 s a turn for a
14B at 25 Mbps), and a GPU segment.

## Routing to the segment that holds it (2026-10-11)

What the release above left: the planner never read the belief table, so turn 2 resumed only
when the plan named the same segments again. With two WARM peers able to run the same layers —
only one holding this conversation — the choice was speed alone, and the other could win: a peer
measured a moment ago on a short request reads cheaper than one measured on a long one.

**The rule** is the one a prompt pass resumes by. `cached_prefix::plan_prompt` asks
`pipeline::held_openings` for every (node, layer range) this node believes holds the opening of
THIS prompt, in the same whole blocks `resume_at` would use (`resume_at` is now the least of
them, so the two cannot drift), and each candidate is handed its own
(`NodeCandidate::split_openings`). Both routers read `NodeCandidate::prompt_tokens_held`: the
search takes it off the prompt a segment reads and moves (`vertex_cost`'s prefill and transfer),
and the greedy fallback — which plans every split whose holders overlap and priced no prompt at
all — now prices the UNREAD prompt at the search's rate (`parallax::prefill_ms_per_layer`). A
bare credit there would have kept a long prompt on a slow holder that a card reads whole in a
fraction of the time (test `greedy_does_not_keep_a_turn_on_a_holder_a_card_outpaces`). A hit is
a price, never a pin — SGLang's lesson, as `cached_prefix` already keeps it.

**Only for the exact range.** A worker keys its stored prompts by its layer range, so a segment
one layer wider holds none of them. **Per vertex**, while a split resumes only as far as EVERY
segment holds: a plan pairing a credited segment with one holding nothing is priced as if it
resumed. That errs only where the holder is not the faster machine, by at most its credit, and
the pass it then runs stores the prompt on every segment.

**Measured** (`split_rig.sh splitcache SPARE=1 WARM_SPARE=1`, Llama-3.2-3B on the processor, A
holding shard 0, B and C shards 1-3, debug builds of one tree with and without the change, live
node stopped, safety kit):

| Arm | Planned by | Turn 2 went to | Resumed from | Turn 1 | Turn 2 | `predicted_ms` (turn 2) |
|---|---|---|---|---|---|---|
| before ×2 | search | B (the other conversation) | 0 | 184 / 180 s | 196 / 195 s | 421,272 |
| after ×2 | search | C (this conversation) | 2,240 | 185 / 184 s | **55.8 / 53.6 s** | 113,587 |
| before, A shards 0-2 (overlap) | greedy | B | 0 | 132.0 s | 142.8 s | 825,870 |
| after, A shards 0-2 (overlap) | greedy | C | 2,240 | 134.0 s | **52.3 s** | 124,308 |

In every run B was asked a conversation of its own after turn 1, so both peers were warm; in
the default shape B's plan named A and C, none of them B, and it handed the lead to A
(`delegated split`), which is why A's log shows two prompt passes before turn 2.
