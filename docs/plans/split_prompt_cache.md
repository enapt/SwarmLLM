# A split keeps its conversation's prompt across turns (FUTURE_WORK #10, the split half)

**Status: designed 2026-10-09, not built.** The local half shipped in v0.3.208 — a whole model on
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
   no tokenizer, no resume) and computes the block chain, vLLM's rule, with a KEYED hash (a
   per-node secret derived from the identity key): a segment holds keys it cannot invert, so the
   middle of a boomerang learns nothing new about the text, and two coordinators' entries never
   meet.
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
  the byte budget must count it, and `kv_budget::admit_prompt` must be able to release it.
- A prompt pass sent in chunks (`chunk_meta`) and the failover paths: a stand-in segment holds no
  entries — it must be planned with `resume_at = 0`, which the belief table gives for free.

## How to measure it

A rig turn pair (`split_rig.sh`, a new mode beside `cache`): two turns of an agent conversation
through A → B; turn 2's prompt pass is timed and its positions counted (expect it to send only
the new turn's tokens), replies scored against llama.cpp, A/B inside one binary with an env
switch, and a forced eviction on B (the refusal → one retry from 0).
