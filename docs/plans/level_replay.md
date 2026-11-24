# Replaying a captured drafter level instead of recording it again (FUTURE_WORK #228)

Status: **Stage 0 done** (2026-11-23: the oracle reads copies and sets; the floor is 5.0 ms at 8 rows and
3.9 ms at 2, a relaunch bit-identical); stages A-D not built. The staged plan below comes from reading the code at
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

- **SGLang's EAGLE draft runners** (`eagle_draft_cuda_graph_runner.py`, read 2026-11-23) fix the tree's shape
  (top-k × steps) per graph, run every draft step inside ONE capture, and before each replay rewrite static
  `seq_lens` / `positions` / `out_cache_loc` and the KV index table in one grouped copy, with the padding rows
  zeroed (neutral), never only masked. The attention reads the committed prefix through an index table, not a
  length baked into a kernel argument; the host bookkeeping (`seq_lens_cpu`, the plan) still runs every step.
- **vLLM's hybrid (Mamba / gated delta-net) graphs** (`mamba_attn.py`, `gdn_attn.py`, `causal_conv1d_update`)
  make the recurrent state's read and write SLOT a device tensor (`state_indices_tensor`, a null sentinel for a
  padded row) and read the convolution's history from `num_accepted_tokens - 1`: our items 9-11 are this.
  FlashAttention under graphs takes `cache_seqlens` as a device tensor and PINS its split count
  (`max_num_splits`), as our chunk choice must be pinned per bucket.
- **Neither replays a TREE-shaped recurrent state**: vLLM's delta-net speculation is a chain. The `ahead`
  reads, the anchor and the store compaction are our own design; the replay's `check` mode is their guard.

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
  - ~~A test-only `DecodeGraph::relaunch(shape, n)` in `flowing_levels_pace_captured_and_op_by_op`.~~ DONE
    2026-11-23 (`cuda_graph::relaunch_probe`, called at the capture's own call site in `decode_step_with_graph`
    while every buffer the level read is held; probe switch `SWARMLLM_PACE_RELAUNCH=<n>`). **The floor, measured
    (Windows up 3.6 h, live node stopped, `~/swarmllm-gate/deep/fr/s0_1123/`):** a relaunched 8-row level of the
    0.8B drafter is **5.03 / 5.05 ms** (host 0.08 ms of it; two runs, 39 levels each), a 2-row level **3.89 /
    3.91 ms**; the Qwen3.5-4B's 8-row level 14.28 / 14.48 ms against 22.1 / 22.3 op by op (llama.cpp's own
    llama-bench on that file: pp8 25 ms, pp16 18.5 ms — our kernels are not the gap, the host is).
    **(c) bit identity: 39 of 39 relaunched levels' outputs (all 23 of the 4B's) keep the recorded bits**, worst
    difference 0 — a relaunch is idempotent, so a replay that holds the inputs fixed reads what a recording does.
    **The 2B drafter** (Qwen3.5-2B Q8_0, which drafts the 9B split ~16-23% faster than the 0.8B on the link,
    `recurrent_flow.md` § "The 2B drafter, captured"; `~/swarmllm-gate/deep/fr/s0_2b_1123/`, live node stopped):
    an 8-row level **17.3-18.0 ms op by op, 13.9-14.9 captured, 9.4-10.0 relaunched** (host 0.08), a 2-row level
    12.1-13.2 op by op, 7.26 relaunched; 39 of 39 bit-identical at each width. llama.cpp on the same file reads 8
    rows in 9.4 ms (pp8 850 tok/s) and 2 in 8.1 (pp2 247) — a replayed 2B level would be llama.cpp's.
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

## Expected gain (the floor measured 2026-11-23: 5.0 ms at 8 rows, 3.9 ms at 2)

An 8-row level 8.8 ms → ~5.5 ms replayed (~5.7 ms counting re-recordings, ~1 level in 20); a 2-row level
6.0-6.6 → ~4.1 ms. On the 9B ↔ AU split, drafting is ~18 of the ~38 ms this computer spends on a forward, so a
third off it is ~6 ms a forward (~15%) while this computer stays the slower side. With the 2B drafting, a level 14 → ~9.7 ms, ~9-10 ms off a two-level call's ~48 ms forward (~20%). Judge it on the link, by
tok/s and by this computer's drafting time a forward.

**The near forward too** (measured 2026-11-23 night, the whole Qwen3.5-9B Q4_K_M, 16 rows a forward as a two-level
call sends, `flowing_levels_pace_captured_and_op_by_op` at `SWARMLLM_PACE_ROWS=16`, `~/swarmllm-gate/deep/fr/wide_1123/`):
op by op 37.5-37.9 ms, captured (since `cuda_graph::MAX_TREE_POSITIONS`) 35.0-35.9, relaunched unchanged 28.0-28.5
with 29 of 29 outputs bit-identical (the card already at 76 °C and 1,560 MHz of 1,755 in that arm, so a floor no
lower than this). The capture took a third of the host's share; a replay would take the rest — on the near half
(layers [0, 17), 17 of the 9B's 32) ~3.5-4 ms more a forward, beside the drafter's ~9-10.

## Measured against the floor (2026-11-23 night)

Stages A-D were built and run on the card outside main (`~/swarmllm-gate/deep/fr/stageD_1123/`, `stageDm_1123/`):
a level whose key matched the last of its shape was launched again with no forward, its host side written beside
it. Every replayed level's logits were the ordinary forward's bit for bit (0.8B, 2B, 9B; 8 rows 13 of 23 levels
replayed, 2 rows 18 of 23). **It bought only 2-4% a level** (0.8B ~9.3 → ~8.9 ms, 2B ~13.4 → ~13.15): the card
already ran each recorded group while the host recorded the next, so removing the recording (3.2-3.6 ms of host)
exposed the card's own 6.5 ms (0.8B) / 10.6 ms (2B) after the launch. The floor above was the graph alone; a level
is its STAGING, then the card.

Marks through a level (test-only, `relaunch_probe`) then split the ~2 ms of staging: **1.3-1.4 ms of it was
`FlowRows::advance_to`**, the walk's replay — five gathers and a slot table a recurrent layer, 18 layers, to read a
path that is one row of the store. **Fixed on main**: `FlowRows::path_inputs` reads a run of consecutive slots in
place (a view: candle counts a size-1 dimension as contiguous, and the card replay reads offsets), else gathers as
before; A/B `SWARMLLM_FLOW_PATH=gathered`. `advance_to` 1.33-1.43 → 0.51-0.66 ms a level; pace (8 rows, minimum of
the runs, `mainpath_1123/`) 0.8B captured 8.39 → 7.73 ms, op by op 13.1 → 10.9; 2B captured 13.03 → 12.14. Bit for
bit in both arms: the card against the processor, lines read alone, a fresh read of each path, captured against
ordinary on all three models.

What is left of a level, in order: the card's own time (the next lever, kernel by kernel), the rest of
`advance_to` (two fresh state pairs — no longer zeroed, below — and two replays a layer), `make_room`'s growth (one gather a kind since 2026-11-24, below) on the levels that grow the
store (3.6-3.9 ms against ~1.8), and the anchor copy (~0.24 ms). The replay waits until those are smaller than the
recording it removes.

### The card's own time: llama.cpp's few-row kernel, timed (2026-11-24)

A level's quantized products at 8 rows go to MMQ (`mmq_few_rows_line`: Q8_0 from 7 rows, the K-quants from 4) where
the weight is at least 2 MB, else to candle's vector kernel, an older copy of llama.cpp's MMVQ that costs ~1.5x at 8
rows what it costs at 7. llama.cpp itself keeps every type on MMVQ up to 8 rows on this card generation
(`ggml_cuda_should_use_mmvq`, `MMVQ_MAX_BATCH_SIZE`), so the question was whether its NEWER MMVQ would beat our MMQ.
Timed with its own op tester (`test-backend-ops perf`, llama.cpp master `4b1a27f`, RTX 3070 Laptop, the node stopped,
two passes, `~/swarmllm-gate/deep/fr/mmvq_1124/`), Q8_0 at 4,096 × 14,336 (62 MB): 1 row 174 µs (~357 GB/s), 2 rows
175, 3: 184-188, 4: 203-214, 5: 198-217, **8 rows 287-294 µs, 1.67x one row (~214 GB/s)**; Q4_K 98-104 µs at 1 row,
296-297 at 8. **Its newer kernel jumps at 8 rows too**, and there reads about what our MMQ already reads on a large
weight (a 145 MB head, 560 µs ≈ 259 GB/s, nsys 2026-10-22). So porting it buys no kernel time on large weights.
What it could still buy is LAUNCHES on the small ones: MMQ on a weight of under ~7 tiles is quantize + product +
stream-K fixup (the fixup 4.8 µs on a 0.5B key), MMVQ quantize + product — and a level makes 392 MMQ launches.

**Timed at the 0.8B drafter's own shapes (2026-11-24): no port.** llama.cpp's product fed the drafter's Q8_0 shapes
(`--test-file`, `~/swarmllm-gate/deep/fr/target_1124/shapes_08b.txt`, two passes, the node stopped) against our
kernels off an nsys trace of a captured 8-row level (`nsys_unwritten.sqlite`, each product's shape read off the
activation's quantize launched before it). At 8 rows, llama.cpp → ours (product + stream-K fixup + quantize): FFN up
3584 × 1024 25-33 → ~20 µs, FFN down 1024 × 3584 23-25 → ~21.5, the 2,048-wide output projections 18.5-19 → ~16, the
248,320-row head 1,686-1,744 → 806. Every trunk product runs on all 40 SMs (stream-K) with its fixup (2.3-3.1 µs);
the trunk's weights read at 190-230 GB/s, the head at ~335. The fixups (~139 a level, ~0.4 ms) are what is left of
this lever, and the kernel that would drop them is slower. A level's MMQ path is ~3.7 ms of its ~5.2 ms of kernels.

### The walk's replay targets are not zeroed (2026-11-24)

Both of `advance_to`'s replays write their target pair whole — `conv_window_replay_f32` every element of the window,
`delta_rule_replay_f32_d*` every element of the state, and the composition sets both whole (`written_into`) — so
the pair comes from `qwen35::unwritten_like` (`CudaDevice::alloc_fully_overwritten`), no memset: up to four a
recurrent layer every forward the walk moved. A/B `SWARMLLM_SSM_REPLAY_TARGET=zeroed`. The card test hands both
paths NaN-filled targets (red when one is planted). One binary, alternating arms, the node stopped
(`~/swarmllm-gate/deep/fr/target_1124/`): memsets in a 12-level 0.8B trace 5,111 → 3,527; an 8-row level captured
(min / median of six) 0.8B 7.92 / 8.04 → 7.61 / 7.71 ms, 2B 12.27 / 12.39 → 12.01 / 12.10; captured against
ordinary bit for bit on the 0.8B, 2B and 9B.

### A full store grows in one gather a kind (2026-11-24)

One level in three grows the store on the 0.8B (8-row levels; `store_capacity_for` makes room for half as many again
as the rows kept), and a growth was a zero-filled store a kind, the kept rows gathered apart, then copied in: 90
fills, 180 allocations and 180 launches over the 18 recurrent layers' five kinds, which an nsys trace split by level
put at ~2 ms of a growth level. `FlowRows::make_room` now makes each new store in ONE gather: the kept rows' slots,
padded to the capacity with the first of them — a row past the kept ones is read only after `put` writes it. A/B
`SWARMLLM_FLOW_GROWTH=copied`. One binary, alternating arms, the node stopped (`~/swarmllm-gate/deep/fr/growth_1124/`):
fills in a 12-level trace 3,527 → 2,447 and launches 26,325 → 25,245 (90 each a growth); the MEAN of a run's levels
(what a reply pays — a median hides the growth levels) 0.8B 8.39-8.85 → 7.77-8.17 ms, the slowest third 9.98-11.12
→ 8.24-9.10; 2B 12.71-12.97 → 12.28-12.63, slowest third 14.16-14.58 → 12.78-13.22; captured against ordinary bit for
bit on the 0.8B, 2B and 9B, and the processor's flowing tests red with the kept rows' slots rotated by one.

**And a DeltaNet layer quantizes its input once for both projections** (2026-11-24): `wqkv` and `wz` read the same
`x` and each quantized it for itself; both now go through `QMatMul::forward_shared`, in a step and in a flowing
forward (`forward_deltanet_into`, `forward_flowing`). Quantize launches per traced 0.8B level 145.4 with nothing
shared (`SWARMLLM_SHARE_PROJECTIONS=0`), 118.7 before, 103.4 after — one fewer a recurrent layer, every Qwen 3.5
step on a card too; ~2.8 µs each with its gap, under a pace run's noise, so judged by the count. Captured against
ordinary bit for bit on the 0.8B, 2B and 9B; every `qwen35` card test passes (`~/swarmllm-gate/deep/fr/shared_1124/`).

**Measured and dropped: a stream-k grid of every resident block** (2026-11-24). A captured 0.8B level is 5.3-5.4 ms
of kernels: the trunk's 138 products 2.27 ms (16.5 µs each, ~250 GB/s against the head's ~330), the head 0.81, the
delta rule 0.61 (18 × 33.9 µs), the stream-k fixups 0.35 (138 × 2.5 µs; in a graph there is no gap around them).
llama.cpp launches stream-k as one block per SM (its rule, unchanged since PR #22298: tile when tiling loses ≤ 10%,
else `nsm` blocks and a fixup), and two of the 8-row kernel's blocks fit an SM here (118 registers, ~40 KB shared),
so the grid was made SMs × resident blocks, as CUTLASS's stream-k swizzle sizes it. Per product: 16.45 → 16.52 µs
(no gain), the fixup 2.53 → 3.60 µs (twice the slots); mean level 0.8B 7.97 → 8.11 ms, 2B 12.23 → 12.27 (three runs an
arm, alternating; `~/swarmllm-gate/deep/fr/persm_1124/`). Reverted: a second resident block does not read the weight
faster. Still open on the same products: the fixup's 0.35 ms folded into the product (the last block to finish a
tile sums its parts in the fixup's order — the same bits, no block waits on another).
