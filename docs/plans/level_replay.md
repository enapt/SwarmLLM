# Replaying a captured drafter level instead of recording it again (FUTURE_WORK #228)

Status: **planned, nothing built** (2026-11-23). The staged plan below comes from reading the code at
`18088f87b`; every file:line in it is a pointer to check, not a fact to build on (gotcha #645).

## Why

On the 9B Thailand ↔ Australia split this computer is the slower side: its drafter's levels (~45% of its time)
and its own layers (~50%) are what a forward waits for, not the link (`docs/invariants/scheduling.md`, "Where the
layers sit"). A captured 8-row level of the Qwen3.5-0.8B drafter is bound by RECORDING, not by the card:

- **Census** (`SWARMLLM_COUNT_KERNELS=1`, before `18088f87b`): 1,027 kernels a level × ~8.6 µs of host work to
  record each = the 8.8 ms level, against 5.19 ms of card time.
- **After the kept-rows cut** (`18088f87b`, 126 launches fewer): ~900 kernels, ~7.7 ms (UNVERIFIED).

Each level is recorded again because its parameters move: new tensors for the token ids and row tables, a longer
attended KV, a new write offset. A level whose parameters held still could be relaunched as recorded. That
costs the card's 5.2 ms plus a little host work (estimate: ~5.5 ms a level, ~5.7 ms with the re-recordings
counted).

**Prior art.**
- **llama.cpp** relaunches an unchanged CUDA graph and records only when `ggml_cuda_graph_update_required` finds
  a node that changed. It pads the KV it attends to a multiple of 256 so the length holds still. It writes new
  rows with `set_rows` and an index tensor, never at a baked offset.
- **vLLM and PyTorch** capture into STATIC input buffers, copy each step's inputs into them before the launch,
  and pad batch sizes to captured buckets.

We take all three: one fixed input block, KV padded to a bucket, rows written through an index.

## What the plan found first

1. **The parameter diff was blind to copies** (fixed the same day, Stage 0's first item). `cuda_graph.rs`
   `launches_of` read only KERNEL and MEM_ALLOC nodes. Candle's contiguous copies are MEMCPY nodes (`memcpy_dtod`, vendored candle-core
   `cuda_backend/mod.rs`), so `SWARMLLM_GRAPH_PARAM_DIFF=1` could not see a copy whose source moved.
2. **Skipping a forward skips its host bookkeeping.** `qwen35.rs` `forward_flowing` updates `keep_appended`,
   `advance_to` and `note_flow_read`, and `SeqCache::append` moves `current_seq_len`. A replay must run the same
   host code; only the recording goes.
3. **No test proves a captured level is bit-identical.** `a_captured_flowing_level_reads_as_the_ordinary_one`
   checks a worst logit difference under 1e-3.

A replay error in the DRAFTER costs speed, never a wrong reply: the target checks every guess. A change to a
kernel the TARGET also runs (the tree attention) must leave the target's path as it is.

## What moves between levels, and the fix

| # | What moves | Where it is made (pointer) | Fix |
|---|---|---|---|
| 1 | token ids | `split/executor.rs`, a `to_device` a level | a fixed per-shape LEVEL BLOCK: one card buffer, one host → card copy before the launch |
| 2 | tree visibility words | `layers/mod.rs` `tree_visibility_on` | a view of the level block (shape `[rows, 16]` is already fixed) |
| 3 | RoPE rows | `layers/mod.rs`: an index upload + two `index_select` | positions in the block; gathered into fixed cos/sin buffers |
| 4 | row tables (slots, parents, walks) | `qwen35.rs` `rows_on`, new content a level | fixed-capacity regions of the block |
| 5 | `out`, `kept_rows` | `split/executor.rs`, a `Tensor::zeros` a level | owned per shape, like `boundaries` |
| 6 | the anchor pair | `advance_to` makes a fresh pair when the walk moves | a fixed per-layer copy, refreshed when the anchor changes |
| 7 | the store | `make_room` makes a fresh store | compact in place where the capacity allows; real growth re-records |
| 8 | the attended KV length | `kv_cache.rs` `current_data`; `decode_attn.rs` `s_len`, grid, partials | pad to a 256 bucket, masked by the visibility bits (chunk sizes switch at 256/512/1024, so masked chunks add exact zeros); the V loop must SKIP masked rows, or a stale non-finite value × 0 is NaN |
| 9 | the KV write offset | `kv_cache.rs` append; f16 `WriteHalf { offset }` | a `put_rows` kernel writing to rows the block names (llama.cpp's `set_rows`) |
| 10 | the store write offset | `qwen35.rs` `put` (`slice_set` at `len`) | the same `put_rows`; kernels handed the whole-capacity view |
| 11 | `ahead` | `qwen35.rs`: the conv and delta-rule `seq`, output sizes, `narrow` | TAIL kernel variants that read `seq` from the block and write only the last rows |
| 12 | the tree base | changes on a rebase | part of the replay key |

The vendored MMQ, cuBLAS, norm and SiLU kernels take only the row count, fixed per step shape, and need nothing.
The new kernels (`put_rows`, the two tail variants, the V-loop guard) are ours, in `kernels/`.

## Stages

Each lands alone, behind its own A/B switch, with its own proof.

- **Stage 0 — the oracle and the floor** (~90 min, a diagnostic only). Do it first: it says whether ~5.2 ms
  is reachable before anything else is built.
  - ~~`launches_of` reads MEMCPY and MEMSET nodes too.~~ DONE 2026-11-23: `copy_of` / `set_of` in
    `cuda_graph.rs`, compared as launches named `memcpy` / `memset`.
  - `note_moved` gives each level a verdict: EXACT means the same topology, and no parameter moved except
    inside the graph's own memory.
  - A test-only `DecodeGraph::relaunch(shape, n)` in `flowing_levels_pace_captured_and_op_by_op`. Relaunching
    the last level is idempotent, so its time is the replay's floor.
- **Stage A — fixed inputs** (items 1-5, host only). A/B `SWARMLLM_LEVEL_INPUTS=per-level`.
  - Proof: the captured-level test made bitwise, still one graph made; the in-place store test still passing.
- **Stage B — scalars on the card**, in three parts:
  - **B1** `put_rows` (items 9-10). A/B `SWARMLLM_LEVEL_PUT=offset`.
  - **B2** the tail kernels (item 11). A/B `SWARMLLM_FLOW_TAIL=0`. Card test: the tail is the full output
    narrowed, bit for bit.
  - **B3** KV padding (item 8). A/B `SWARMLLM_TREE_KV_PAD=0`. Card test: padded equals unpadded with the
    padding poisoned with NaN.
- **Stage C — stable state addresses** (items 6-7). A/B `SWARMLLM_FLOW_FIXED_ANCHOR=0`,
  `SWARMLLM_FLOW_STORE_COMPACT=0`.
  - Proof: the oracle reads EXACT across calls except at growth, a bucket change or a rebase.
- **Stage D — the replay.** A/B `SWARMLLM_LEVEL_REPLAY=0|1|check`.
  - Split `forward_flowing` into a PLAN and a COMMIT, and `SeqCache::append` into reserve and write, so
    recording and replay share one host path.
  - The key (`LevelKey`): the step shape, the KV bucket, the tree base, the address and capacity of every
    buffer the level reads, a counter of kernel-choice changes, and the fixed buffers' generation.
  - A hit uploads the block, launches the level's graphs and commits; a miss records as today.
  - `check` records every level and asserts the oracle says EXACT wherever the key said hit; production runs
    1 replay in 256 as a check.
  - Test `a_replayed_flowing_level_reads_as_the_recorded_one`: the logits bitwise, and the host state (read,
    parents, kept, slots, the store's length, `current_seq_len`) against every level recorded.

## Risks, and what catches each

- **Memory freed under a capture** (gotcha #1165): the fixed buffers are made outside any capture, owned by
  `DecodeGraph`, replaced only between levels; the key holds their addresses, so a graph pointing at freed
  memory never launches.
- **A synchronize inside a capture** (#1163): a replay has no capture.
- **Growth** of the store or the KV: a new address misses the key and records.
- **One CUDA stream a device**: the upload, the launches and the commit's frees stay on it; the level block never
  goes through `clone_htod_beside` (#836).
- **Silent wrongness**:
  - a moving value missing from the key → `check` mode and the canary;
  - stale padding → the V-loop guard and the NaN-poisoned test;
  - drifted host state → the replayed-state test;
  - a launch failing partway → the refused-capture rollback.

## Expected gain (UNVERIFIED until Stage 0's floor)

An 8-row level 8.8 ms → ~5.5 ms replayed (~5.7 ms counting re-recordings, ~1 level in 20); a 2-row level
6.0-6.6 → ~4.1 ms. On the 9B ↔ AU split, drafting is ~18 of the ~38 ms this computer spends on a forward, so a
third off it is ~6 ms a forward (~15%) while this computer stays the slower side. Judge it on the link, by
tok/s and by this computer's drafting time a forward.
