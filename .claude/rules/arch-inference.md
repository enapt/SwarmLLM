---
paths:
  - "src/inference/split/**"
  - "src/inference/layers/**"
  - "src/inference/executor.rs"
  - "src/inference/attn_kernel.rs"
  - "src/inference/attn_softmax.rs"
  - "src/inference/attn_rows.rs"
  - "src/inference/decode_attn.rs"
  - "src/inference/fast_math.rs"
  - "src/inference/cpu_pools.rs"
  - "src/inference/tokenizer.rs"
  - "src/inference/sampling.rs"
  - "src/inference/kv_cache.rs"
  - "src/inference/speculative.rs"
  - "src/inference/quant.rs"
  - "src/inference/model_arch.rs"
  - "src/model/lora.rs"
  - "src/inference/tensor_util.rs"
  - "src/inference/mem_bandwidth.rs"
  - "src/inference/vision.rs"
  - "src/inference/allreduce.rs"
  - "vendor/candle/**"
  - "vendor/candle-flash-attn/**"
  - "src/inference/shard_layout.rs"
  - "src/inference/prof.rs"
  - "src/inference/local_embedder.rs"
  - "src/inference/swift.rs"
  - "src/inference/mod.rs"
  - "src/inference/cuda_graph.rs"
  - "src/inference/prefill_attn.rs"
  - "src/inference/residual_norm.rs"
  - "src/inference/coupled_noise.rs"
---

# Inference kernels, caches and the tokenizer

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## A context that will not fit is shrunk, not refused

`inference::executor::context_retry_ladder` answers "what context size will this card accept": halve from the capped figure to a floor, serve the first size accepted, log what was granted. Never refuse a context a smaller one would fit. Called only from `llama`-gated code, so every default build reports it dead (gotcha #264).

→ `docs/FUTURE_WORK_ARCHIVE.md` § "A context that does not fit is refused instead of shrunk"
→ `docs/invariants/inference.md` § "A context that will not fit"

## Cross-feature compile checks

`cargo check` with default features compiles NO `cfg`-gated path. An ungated doc comment never LINKS (`[`…`]`) a `cfg(feature)` item (`cargo doc` fails, gotcha #832): name it in plain backticks. ⚠ **Before removing anything an unused-warning points at, grep the file for `#[cfg(`** (gotcha #264). Changing a signature callable from gated code: `cargo check --features llama` and `cargo check --features flash-attn --all-targets`, then `gh run list --workflow="Cache warm"`. **A card build's `SplitModel` is `Send` but not `Sync`**: one kept in `SharedState` sits behind a `Mutex` with a const `Send + Sync` assertion beside its holder (`local_embedder.rs`); check with `cargo check --locked --features candle-cuda --all-targets` (`CARGO_TARGET_DIR=~/swarm-target-cuda CUDA_COMPUTE_CAP=80`) (gotcha #912).

### The CUTLASS kernels live OUTSIDE `target/` (2026-08-17)

`CANDLE_FLASH_ATTN_BUILD_DIR` (`.flash-attn-build`) keeps the 19 kernels out of rust-cache's reach; confirm `All kernels up-to-date, skipping compilation` (#318). The directory must exist; an empty env var is not an unset one.

→ `docs/invariants/inference.md` § "Cross-feature compile checks"

## A decode token is bound by GPU submission COUNT, not bandwidth (2026-09-22)

Judge a change by launch count (`SWARMLLM_COUNT_KERNELS=1`, never on in a benchmark), not the clock; `SWARMLLM_ZERO_QMATMUL_BUFFERS=1` = old behaviour.

- **`CudaDevice::alloc_fully_overwritten`** only for a buffer the next kernel fully assigns.
- **`QMatMul::forward_shared`** — Q/K/V, gate/up and a DeltaNet layer's `wqkv` + `wz` share ONE activation; BOTH `mul_mat_vec_via_q8_1` and `mul_mat_via_q8_1` must; LoRA after it.
- **`layers::value_for_matmul`** — never `.contiguous()` a KV cache view before a matmul (`matmul_reads_rhs_in_place`).
- ⚠⚠ **ONE CUDA STREAM PER DEVICE** (`SWARMLLM_CUDA_OWN_STREAM`, `=0` legacy); a second needs `SWARMLLM_CUDA_EVENT_TRACKING=1`. The ONE exception: `CudaDevice::clone_htod_beside`'s copy stream (#138 (1d)), touching only its copy RING; never hand it a buffer candle frees (gotcha #836); a split's prompt copies reach it only via `layers::sent_to_card` / `QMatMul::weight_on`; a `to_device` to ANOTHER card is ordered on both streams in the vendored `transfer_to_device` (gotcha #828). ⛔ Moving off legacy SHIPPED BROKEN in v0.3.199 — reproduce under `--features cuda` on a real generation. Guard: `the_vendored_attention_kernels_launch_on_the_devices_stream`.
- **Fused kernels** (`kernels/*.cu`) are bit-identical to the candle ops they replace (except `decode_attn.cu`, `delta_rule.cu`: judged against llama.cpp); a layer-end residual is `Residual::Pending` → `Residual::add_norm`.
- **`residual_norm::norm_block`** is a card norm's ONE block size: 1024 threads from 512 columns, else 32 — ⛔ only 32 or 1024. A/B `SWARMLLM_NORM_WIDE_FROM=1024`. → § "A narrow row is normed by 1024 threads on a card"

→ technique in `docs/DIAGNOSTICS.md` § "Where a decode token actually goes"

→ `docs/invariants/inference.md` § "A decode token is bound by GPU submission COUNT, not bandwidth"

## A card between a split's forwards keeps its clocks (2026-11-20, #487)

A split's segment idles ~250-450 ms a round; the card's governor read it as idle (210-420 MHz of 1755) and the same 21-layer forward took 76 ms where it takes 25. **`inference::clock_hold::ClockHold`** holds it: from every step of a reply shared with other computers a worker is handed (**`clock_hold::starts_a_hold`**, one exhaustive match: forwards, batched forwards and the guesses beside them — never `Generate`), for `HOLD_FOR` (1 s) past the step's END (`note_handled`, after the handler), a `PIECE` (2 ms, one block, `kernels/clock_hold.cu`) every `PERIOD` (4 ms) on the device's ONE stream, started only in the worker loop's idle branch (a message is waited for between pieces), never while paced for heat (`pace::work_ahead_may_run`), off for good on a failed launch. A piece ends by the card's timer or a cycle bound — never near the TDR watchdog. Never queue more than one piece: one starts only on an idle stream (`cuStreamQuery`). **On unless `SWARMLLM_CARD_CLOCK_HOLD=0`**, because a hold **pauses while another of our card workers has a reply in flight** (#489): a second process on the card (a drafter model, the owner's own reply) is time-sliced against the pieces — the 14B split with its drafter here ran ~38 ms a forward held vs ~27 off before the pause. The daemon alone sees that: **`ModelProcessPool::tell_cards_who_is_busy`** (woken by `card_work_changed`, rung ONLY where a reply slot opens, `register_unseen`, and closes, `ResponseGuard`'s drop) sends a change-only `DaemonMsg::CardBusyBeside` — work counts for `CARD_WORK_LINGERS_FOR` (1 s) past its last step, so a drafter between two guesses still counts — the reader task sets `clock_hold::set_busy_beside` — and **a drafter model's `Draft` never starts a hold** (its target holds the card for both). A change to the hold is measured on three shapes: the 27B (one process), the 14B flow with its drafter, and the owner's reply beside a held card — from a model PROVEN to run here (`route=local`).

→ `docs/invariants/inference.md` § "A card between a split's forwards keeps its clocks"

## Several rows on a card go to llama.cpp's int8 tensor-core MMQ (2026-10-21, #325)

Vendored `quantized/cuda.rs::dequantize_matmul` sends ≥ 9 rows — fewer where **`mmq_few_rows_line(dtype)`** says (read by `mmq_takes_few_rows` and the card tests), inside a graph capture too once the type is READY (`mmq_prepare_few_rows_once`: nothing on the capturing thread may raise the shared-memory limit, gotcha #841's class; A/B `SWARMLLM_QMATMUL_MMQ_CAPTURED=0`) — to llama.cpp's MMQ (`candle-kernels/src/ggml_mmq`) wherever `swarm_mmq_available()`. ⛔ All of it in `namespace swarm_ggml` (gotcha #927, guard `the_vendored_mmq_keeps_its_own_namespace`). ⛔ Nothing may reach llama.cpp's `CUDA_CHECK` / `GGML_ASSERT` (they abort): the launcher answers a status. ⛔ Every vendored kernel LIBRARY carries its floor's PTX (`build.rs` `ptx_beside`; guard `every_vendored_kernel_library_carries_ptx`; check with `CUDA_FORCE_PTX_JIT=1`). The Linux release adds 12.0 machine code to flash and MoE (`SWARMLLM_CUDA_MACHINE_CODE_ALSO`, #326), ⛔ never MMQ; CI: `examples/check_card_machine_code.sh`. A/B `SWARMLLM_QMATMUL_MMQ=dp4a`, `SWARMLLM_QMATMUL_MMQ_MIN_ROWS`, `SWARMLLM_QMATMUL_MMQ_MAX_ROWS`; `mmq_mma_products()` proves it fired. A refused launch turns MMQ off for the process (`MMQ_MMA_FAILED`); a product with no card memory falls back for ITSELF only (`MmqFailure::NoRoom`, `mmq_mma_no_room()`) — never latch on a transient condition.

**2-8 rows of a Q8_0 weight, and 4-24 of a Q4_K or Q5_K weight of ≥ 4,096 rows (`ROWS_TC_K_LEAST_WEIGHT_ROWS`) or a Q6_K one of ≥ 2,048 (`ROWS_TC_Q6K_LEAST_WEIGHT_ROWS`), with K % 256 == 0 go AHEAD of both to `mul_mat_via_rows_tc`** (2026-11-24; `rows_tc_takes(dtype, rows, n, k)` the one rule, pure — a capture asks it too). Q8_0: 16 weight rows a block, ≤ 3 warps splitting K added in warp order (`rows_tc_warps`, from the shape only). Q4_K / Q5_K / Q6_K: 64 rows a block, four warps at the SAME K sharing one staged activation (`quantize_q8_1_rows_k`, llama.cpp's q8_1 "DS4": (d, Σx) — one layout for all three, so they share it), widths 8/16/24 whose rows' bits do not depend on the width; Q4_K / Q5_K one device function (`swarm_k_rows_tc<NT, Q5>`), Q6_K its own (`swarm_q6k_rows_tc<NT>`: 210-byte blocks staged as the 16-byte-aligned 224-byte windows covering them — the last row's reaches ≤ 14 bytes into the buffer's padding, so **a quantized card buffer keeps its `MATRIX_ROW_PADDING`** — two stages, the next issued under this one's arithmetic). Q4_K / Q5_K run three stages at 8-16 rows and, at 17-24, two with the next issued early (three were one block a multiprocessor for Q5_K); no launch asks past 48 KB of shared memory, so none raises a limit (`mul_mat_k_quant_via_rows_tc` refuses a layout past it). No fixup — the same bits every launch. A `forward_shared` group keeps one activation per layout (`SHARED_Q8`). ⛔ Never launched from a PTX below sm_80 (`__trap()` there; `quantized_ptx_target`). A/B `SWARMLLM_QMATMUL_ROWS_TC=0` (all), `SWARMLLM_QMATMUL_ROWS_TC_K=0` (the k-quants), `=q4k` / `=q5k` / `=q6k` or a list (`q4k,q5k`: those alone); `rows_tc_products()` proves it fired.

→ `docs/invariants/inference.md` § "A card multiplies several rows with llama.cpp's int8 tensor-core MMQ", § "A few rows: the line per type and size, inside a capture too", § "2-8 rows of a Q8_0 weight: one tensor-core kernel", § "4-24 rows of a large Q4_K weight", § "Q5_K on the same kernel", § "Q6_K: aligned windows over 210-byte blocks", § "17-24 rows: two stages, the next issued early"

## A prompt pass on the card multiplies quantized weights on the tensor cores (2026-09-29)

Where MMQ does not run: ≥ 64 rows go to `mul_mat_via_f16_cublas`; decode never reaches it. `prefill_pacer::CARD_CHUNK_TOKENS` (512) via `prompt_chunk_ceiling`, the ONE answer. A/B `SWARMLLM_QMATMUL_CUBLAS=0`. Accumulates in f16 (#147) except weights llama.cpp marks `GGML_PREC_F32`: **`ModelArch::accumulates_outputs_in_f32`** (GLM-4's attention output + FFN down), marked ONCE after load by `layers::accumulate_output_projections_in_f32`; a new arch's marks are read off llama.cpp. `SWARMLLM_QMATMUL_CUBLAS_ACC=32` = all f32; `SWARMLLM_F32_MARKS=0` = no marks. Judge replies against llama.cpp, never byte-equality.

→ `docs/invariants/inference.md` § "A prompt pass on the card multiplies quantized weights on the tensor cores"

## Every prompt pass is cut at the chunk ceiling — a lone request's too (2026-11-15, #456)

A worker prompt pass with no chunk of its own (`model_worker::handle_generate`, gotcha #1104) goes through **`SplitModel::forward_prompt_keeping`**: cut at `split::PROMPT_CHUNK_TOKENS` by `forward_prompt_in_chunks_keeping`, the width `widest_prompt_forward` tells the KV budget. Never `forward` / `forward_keeping` over a whole prompt (gotcha #1044). A/B `SWARMLLM_PROMPT_PASS=whole`. Test `a_lone_requests_prompt_pass_is_cut_where_the_budget_says_a_forward_ends`.

→ `docs/invariants/inference.md` § "Every prompt pass is cut at the chunk ceiling"

## A decode step goes to the card as CUDA graphs — ON by default since 2026-09-30

`inference::cuda_graph` + `SplitModel::forward_decode_as_graph`; `SWARMLLM_CUDA_GRAPH=0` off. Recorded in groups (`cuda_graph::group_layers`, `Cutter::cut`).

- ⛔ A pageable host→device copy inside a capture is silent garbage (`htod_copies_so_far` discards it): a new `clone_htod` on the decode path turns capture off, or is made BEFORE it and handed in (`executor::Prepared`, #295).
- ⛔ **No memory a group ALLOCATES may be read by a LATER group** (A/B `SWARMLLM_PER_LAYER_INPUTS=in-graph`; `SWARMLLM_PROFILE`'s "of which: launch it").
- Capture only where `KvCacheStore::every_cache_holds`; catch the mirror up BEFORE recording (`LayerKv::catch_up_mirror`, #761). A recurrent layer holds only with its NEXT buffers (`SsmState::rewind_to`, #228 (4)). A MoE layer is captured where its experts are STACKED — `executor::ffn_capture_ineligibility` (#138 (2)).
- **A graph pays only when UPDATED**: a shape the driver keeps refusing rests (`DecodeGraph::declines`, `REBUILDS_BEFORE_RESTING`, `REST_STEPS`), **only where the timeline says the rest pays** (`DecodeGraph::begin_step` → a REQUIRED `StepBegun` for `capture` / `note_uncaptured`; `Churn::weighed`; #233; A/B `SWARMLLM_CUDA_GRAPH_REST=count`).
- **A step's ids are ONE type before the capture** (u32, `cast_ids_before_capture`; #233; A/B `SWARMLLM_CUDA_GRAPH_ID_CAST=inside`). A TOPOLOGY refusal names no node — `node_census` does.
- ⛔ **No memset (`alloc_zeros`, `Tensor::zeros`) on memory a captured step allocates** (#221). ⛔ **No driver call from the capturing thread** besides the captured ops (`cuda_pool::current_pool` answers `None` while capturing, gotcha #841).

→ `docs/invariants/inference.md` § "A decode step can go to the card as one CUDA graph", § "A graph pays only when it is updated" and § "A rest is weighed on the card's timeline, and a step's ids are one type", § "Qwen 3.5's decode steps are captured too", § "A mixture-of-experts layer stacked on the card is captured too"

## A head width reaches the card only after the flash kernel is checked at it (2026-10-16)

Run **`examples/flash_reference_check.rs`** (every head width a family uses, against f32 attention) before claiming a new width on a card (gotcha #868). `kernel_traits.h` refuses a block shorter than its warps (guard `every_vendored_flash_block_is_a_whole_number_of_warps_tall`). Both kernel build scripts `.watch()` every header.

→ `docs/invariants/inference.md` § "The flash kernel is checked at every head width it is handed"

## A decoded token's attention on a card is ONE kernel, and never keeps the mirror in step (2026-09-29)

**`decode_attn::gqa_decode_attention_cuda`** answers a one-position attention on a card (A/B `SWARMLLM_DECODE_ATTN=standard`). A one-position append leaves the f16 mirror behind (`LayerKv::append`); `flash_operands` declines a lagging mirror (`SWARMLLM_KV_MIRROR_EAGER=1` = old). ⚠ Not bit-identical to the matmul path. **Same kernels at every cache length** (`decode_attn::combine_every_length`; A/B `SWARMLLM_DECODE_ATTN_COMBINE=when-needed`). **Chunk by cache length and head layout, never the card** (#309): **`decode_attn::chunk_for`** (smallest of 16/32/64 keeping ≤ 256 blocks and ≤ 64 chunks), a kernel ARGUMENT. A/B `SWARMLLM_DECODE_ATTN_CHUNK=64`. **A decode step over the half cache runs on the TENSOR CORES** where **`decode_attn::mma_takes`** says (d 64/128, ≤ 16 query heads, merged, 4-byte rows, sm_80 PTX — never the length or the card): one warp a run of positions, the group as `mma` rows, the old `partial` layout and merge; judged against llama.cpp (query and P in half). A/B `SWARMLLM_DECODE_ATTN_MMA=0`. → § "A decode step over the half cache runs on the tensor cores"

→ `docs/invariants/inference.md` § "A decoded token's attention on a card is one kernel"

## A card keeps its KV cache as f16 — `layers::kv_storage` is the one answer (2026-10-09)

**`layers::kv_storage`** says how a layer keeps its cache (`KvStorage::{F32, F32Mirrored, F16}`; its `KvAttention` argument is REQUIRED — MLA is f32 everywhere) for the cache as built AND every estimate of its cost: loader budget, split planner, admission (all via `kv_budget::KvLayout`, per layer), a peer (`kv_storage_as_advertised` + `features::KV_HALF_ON_CARD`). **F16 alone** where both kernels read it in place — flash and the decode kernel (`decode_attn::card_kernel_covers`). Every forward writes through **`layers::append_to_cache`** (`LayerKv::conform_to` corrects a hydrated cache); f32 → f16 is ONE launch (`kv_write::write_into`). A/B `SWARMLLM_KV_F16=0`, `SWARMLLM_KV_WRITE=compose`.

→ `docs/invariants/inference.md` § "A card keeps its KV cache as f16"

## A speculative check is captured too, and a layout is uploaded once (2026-09-30)

A forward of up to `cuda_graph::MAX_POSITIONS` (8) positions is captured when no causal mask is read (`SplitModel::mask_is_read`). A first-of-its-shape, instantiating or refused forward is a ONE-OFF (`DecodeGraph::one_off_forwards`, `cuda_graph::one_off`) — a price never records one. **An idle worker records every shape ahead** (`model_worker::warm_check_widths` → `SplitModel::widths_to_record` → `cuda_graph::shapes_to_record`; stops at a message; #357, #428). A/B `SWARMLLM_CUDA_GRAPH_WARM=0`. **A reply makes no graph before that recording** (`DecodeGraph::may_make_graph`, asked at the one capture site; only `record_check_widths`'s forwards make one, `SplitModel::set_recording_widths`): where the recording will come (`widths_recorded_ahead`), a shape with no graph runs the ordinary way (#357 (d)). A/B `SWARMLLM_CUDA_GRAPH_MADE=in-replies`. Tests `a_reply_after_the_idle_recording_makes_no_graph`, `a_reply_before_the_idle_recording_makes_no_graph`, `a_first_segment_records_its_shapes_before_a_reply_needs_them`. `CudaDevice::layout_params` keeps a layout on the device and **never caches one while the stream is capturing**. Graph state and give-ups are per position count; the drafter reads catch-up via `draft_after`, the γ choice `RecentMedian`.

→ `docs/invariants/inference.md` § "A speculative check is captured too, and a layout is uploaded once"

## A tree of guesses is read through ONE wrapper, each row at its own depth (2026-10-20, #134; on a card 2026-10-22)

**`SplitModel::with_draft_tree`** is the one way a tree reaches a forward (the worker's check, each step of `draft_tree_after`). Rows rotate at `base + depth` (`layers::RowPositions`, SET for every job a pool runs, `None` included, by `cpu_pools::with_the_callers_scope`; rope tables made ONCE per forward and table — `layers::TreeRows` / `tree_rope_rows`; A/B `SWARMLLM_TREE_ROPE=per-call`) and read the context plus their ancestors (`DraftTree::rows_for`). **`draft_tree::RowVisibility` is the ONE statement of what a row sees** — the matmul path's mask is made from its bits.

- Card (`gqa_tree_attention_cuda`): over the half cache ANY rows TILED (**`decode_attn::tree_tiles`**, shared memory under 48 KB so a capture may hold it) and **its context goes to flash** (`decode_attn::past_runs`, `auto_past_runs` runs, from `PAST_FLASH_FROM` work; the run count never depends on context length; A/B `SWARMLLM_TREE_ATTN_PAST=tiled`); over an f32 cache ≤ 8 rows one block a row (`TREE_KERNEL_MOST_ROWS` = `cuda_graph::MAX_POSITIONS`, gotcha #934), more to standard attention. A/B `SWARMLLM_TREE_ATTN_TILED=0`.
- A drafter's level (≤ 8 rows) is CAPTURED (`cuda_graph::StepShape`, inputs made before: `layers::stage_tree_forward`). A/B `SWARMLLM_TREE_ATTN=standard`, `SWARMLLM_TREE_GRAPH=0`. **A FLOWING forward to `cuda_graph::MAX_TREE_POSITIONS` (16) rows too** — a Qwen 3.5 split's near forward of two levels — only where every attention layer reads it tiled over the half cache and no layer has experts (`SplitModel::wide_tree_may_be_captured`); A/B `SWARMLLM_TREE_GRAPH_WIDE=0`; test `a_captured_wide_flowing_level_reads_as_the_ordinary_one`. → § "A flowing forward of up to 16 rows is captured too"
- **Rows are ranked on the card in two launches** (`tree_rank::ranked_on_the_card`; a tie goes to the smaller id; a kernel error ranks with the composition). A/B `SWARMLLM_TREE_TOPK=composed`.
- **A SAMPLED row is never read back whole**: its coupled draw reads only the request's `top_k` (`sampling::coupled_draw_reads_only_top_k`, the one rule); the card ranks those (`tree_rank::top_k_on_the_card`, k ≤ 64), the host draws via `sampling::sample_coupled_from_top_k` (`draw_among_candidates`, shared with `sample_among_top_k`). Paths: `RowScores::followers`, `coupled_guess_on_the_card`, `tree_rank::draw_each_row_on_the_card`. A/B `SWARMLLM_TREE_COUPLED_DRAW=host`.
- **Children are rated by the noise the far end draws with** — log-softmax of `logit + T·g` (`followers_rated_by_the_shared_noise`, inside `followers_drawn_from_top_k`), never the drafter's probability. A/B `SWARMLLM_TREE_SAMPLED_RATING=probability`.
- A tree whose rows do not end where the cache does is refused with the caches put back; a layer that keeps keys by position (window, shared cache, MLA) refuses up front (`draft_tree_refusal`). Tests: `split/tests/draft_trees.rs`.
- **Qwen 3.5's recurrent layers read a WHOLE tree of ≤ `SsmState::MOST_ROWS_KEPT` rows** (#437): each row's convolution up its own line (`DeltaNetWeights::causal_conv_along`), its delta rule from its PARENT's state (`delta_rule_cpu` / `delta_rule`'s `tree`), parents read once off the visibility (`qwen35::tree_parents_now` ← `RowVisibility::parents`; a pure chain is a step); on a card ONE launch each (`ssm_conv_silu_tree_f32_k4`, bit for bit `causal_conv_along`; `delta_rule_tree_f32_d*`, a row re-walks its line only where its parent is not the row before), parents copied once a step (`qwen35::TreeLines`). The tree's end is no history: `SsmState::mark_branched` refuses every read on until the verdict's take-back into the in-place path (the replay). A tree check that does not flow is never captured. Test `a_tree_checks_rows_are_their_lines_read_alone`. **A tree that FLOWS** across forwards reads from a checkpoint at the coordinator's walk (`qwen35::FlowRows`, the walk REQUIRED in `with_draft_tree`; on a card `delta_rule_flow_f32_d*`, lines past 32); a forward failed part-way is put back where it began (`SsmState::flow_begun`). **Its rows' inputs sit in a STORE a kind, each written once in place** (`FlowRows::put` after `make_room`; a row the walk left stays until the store is full, then the kept rows move to a larger store in ONE gather a kind, A/B `SWARMLLM_FLOW_GROWTH=copied`), on a card the convolution and delta rule read each kind's store IN PLACE through the lines' slot table (`qwen35::SlotTable` from `rows_the_lines_read`, every slot checked below the store's rows on the host; `ssm_conv_silu_tree_stored_f32_k4`, `delta_rule_flow_stored_f32_d*`; any other path gathers them, `SlotTable::gather`; test `a_store_read_in_place_is_its_rows_gathered_bit_for_bit`), a forward's row tables uploaded ONCE while their content stands (`qwen35::rows_on`: a pageable copy drains the stream) and the walk's replay in two launches into a pair never zeroed (`qwen35::unwritten_like`: both replays write it whole; A/B `SWARMLLM_SSM_REPLAY_TARGET=zeroed`) — bookkeeping was 54-64% of a forward's launches (2026-11-22); A/B `SWARMLLM_FLOW_STORE=gathered|joined|per-layer`. **A flowing level is CAPTURED** (2026-11-23; A/B `SWARMLLM_FLOW_GRAPH=0`): what it would make outside a kernel — row tables, the walk's replay, a scratch pair, room in a full store — is made BEFORE the capture (`layers::stage_flowing_states` → `SsmState::stage_flowing_level`), its room claimed there too (`SplitModel::claim_room_for_forward`: the claim synchronizes, and a synchronize inside a capture breaks it silently, gotcha #1163), a table missed there refuses the capture (`rows_on`), and a captured level always reads its lines' rows through a table, so every level is one topology (`rows_the_lines_read`), and a staged begin keeps its pre-stage copy through the capture (`SsmState::flow_from`: replaced inside, it freed memory made outside the capture — refused, leaked, gotcha #1165); a walk on the level's own rows or the A/B arms run it the ordinary way. A full store's claim is its GROWTH (`FlowRows::rows_a_forward_may_allocate`). Test `a_captured_flowing_level_reads_as_the_ordinary_one`. On the processor compare a tree row with the tree read WHOLE, never its path as a chain (gotcha #1160). → `docs/plans/recurrent_flow.md`

→ `docs/invariants/scheduling.md` § "A split checks a tree of guesses"; `docs/invariants/inference.md` § "A tree level on a card: one tree kernel, captured as a graph", § "A tree's rows are read TILED over the half cache", § "A tree's context is read by flash attention, in runs", § "A tree level's rows are ranked on the card in two launches"

## Attention kernel choice and the query-length cliff (2026-08-23)

Every CPU decode fast path was once gated on `q_len == 1` exactly (#369). Four helpers own those decisions — never re-derive one from `q_len`:

- **`layers::flash_handles_offset_causal`** — flash takes a query block on a warm prefix (bottom-right causal; #368). A/B `SWARMLLM_FLASH_OFFSET_CAUSAL=0`.
- **`layers::cuda_decode_prefers_standard`** — `q_len == 1` for every head geometry.
- **`layers::grouping_applies` + `grouped_gqa_attention`** — K/V at stored width for any query length that fits one pass; the mask is TILED (row `r * q_len + t` sees row `t`).
- **`cpu_pools::DECODE_SHAPED_MAX_TOKENS`** — pool choice turns on bandwidth-bound matmuls, not `seq_len == 1`; a short block never feeds the decode-width calibration.

**A GQA call too big for one pass is blocked over query POSITIONS and grouped per block** (`grouped_blocking_applies`, `mask_rows`) — never `repeat_kv`. Test `a_blocked_gqa_call_is_grouped_not_expanded`.

→ `docs/invariants/inference.md` § "Attention kernel choice and the query-length cliff"

## A row's attention on the processor is ONE arithmetic, whatever forward carries it (2026-10-19, #179)

**`inference::attn_rows`** (`chunk_partials`, `merge_into` + `finish_row` in chunk order) computes a row for BOTH processor kernels (`decode_attn::gqa_decode_attention_cpu`, `prefill_attn::gqa_prefill_attention_cpu`, `q_len >= 2`): fixed 256-key chunks of ABSOLUTE position over the mask's visible keys (`visible_range`); **`key_origin` is a REQUIRED argument** (`keys_in_window`). A new processor attention path calls `attn_rows`; its arithmetic is DEFINED (eight-lane fused multiply-add dot, one add tree, `lane_sum`), a faster kernel joins `the_vector_kernels_are_the_portable_arithmetic_exactly`. Guard: `a_row_s_attention_does_not_depend_on_the_forward_it_is_in`. The matmul paths (**`layers::matmul_attention`** — a test of blocking, grouping or in-place V calls it, not `standard_attention`) and every card are NOT invariant (#313). A check of a few rows splits its blocks' keys into jobs where blocks (KV group × 128 rows) are fewer than the pool's threads (`prefill_attn::splits_keys`, #174; `a_block_s_keys_split_into_jobs_give_the_same_bits`). A/B: `SWARMLLM_PREFILL_ATTN=standard`, `SWARMLLM_DECODE_ATTN=standard`, `SWARMLLM_PREFILL_SPLIT_KEYS=0`.

→ `docs/invariants/inference.md` § "A row's attention on the processor does not depend on the forward it is in" (supersedes § "Several query positions on the processor: tiled over the keys")

## A test comparing bits between forwards of different WIDTHS makes every matmul on its path exact (2026-11-19)

On ARM the `gemm` crate multiplies one row and several rows with different kernels, adding in different orders (`H_M = 0` for NEON); on x86 one kernel serves both, so such a test passes locally and fails only on CI's macOS job. Make EVERY matmul it reaches order-free — each product exact, at most one rounding addition an output (`qwen35::tests::weights_of`) — not only the first; and a new such test is done only once CI's macOS job has run it.

→ `docs/invariants/inference.md` § "A test across forward widths is exact only where every matmul is"

## Local speculative decoding — `inference::model_worker::ngram_spec_eligible`

The single answer to "may this request be speculated?", consulted by BOTH the
slot-admission gate and the decode loop (two copies disagree silently).

**Whether a solo request is diverted to guess is `diverts_to_guess`** (gotcha #1104): always where nothing else is in flight on the worker (`IpcGenerate::others_in_flight`) and nothing overlapped there for `OVERLAP_REMEMBERED`; otherwise the measured payoff decides (`spec_payoff_justifies_diverting`, gotcha #373).

→ `docs/invariants/inference.md`

## A prompt's length in tokens is a POSITION, not a statistic

**`inference::pipeline::prompt::prompt_positions`** is the single answer to "how
many positions does this prompt occupy?", for all five sites in that module.
Never open-code it, and never estimate it.

→ `docs/invariants/inference.md`

## A model's turn-ender is found in its vocabulary, not taken from its declared EOS

**`GgufTokenizerMeta::end_of_generation_ids_from_vocab`** finds turn-ending tokens BY NAME and every EOS-resolving path merges it (`eos_tokens_with_arch_fallback`, `split::entry::SplitModelEntry::from_header`). A declared EOS is never assumed complete (Phi-3/3.5/4 close with `<|end|>`). `<|end|>` is conditional — not a stop for harmony (gpt-oss) or solar-open; `chat_template::extract_stop_strings` carries the same exclusion.

→ `docs/invariants/inference.md` § "A model's turn-ender is found in its vocabulary, not taken from its declared EOS"

## Partial RoPE has one implementation, and it answers with a tensor the KV cache can write

**`inference::layers::rope_over_heads`** is the single partial-RoPE implementation; its result is contiguous (pass-through half made contiguous BEFORE `cat`; `slice_set` refuses a non-contiguous source; `SeqCache::append` too). Never write a second copy. On a card ONE launch (`inference::partial_rope`), bit-identical to `partial_rope_composed`; A/B `SWARMLLM_PARTIAL_ROPE=compose`. **`layers::LayerWeights::heads_first`** is the ONE preparation of `attend`'s Q, K and V; on a card with no per-head norm before RoPE ONE launch (`inference::heads_first`); A/B `SWARMLLM_HEADS_FIRST=compose`.

→ `docs/invariants/inference.md` § "`inference::layers::rope_over_heads` — partial RoPE has one implementation, and it answers with a tensor the KV cache can write", § "Q, K and V reach attention heads-first in one launch"

## A GGUF tensor the loader ignores is a feature silently missing (2026-09-27)

`rope_freqs.weight` was ignored (#124); now applied as llama.cpp does. **Shard-0 tensors a later-layer node needs are ONE list** — `GgufTensorMeta::sidecar_tensors` — walked by `extract_sidecar_tensors`, `download_sidecar_tensors`, `resolve_sidecars`; add to it, never a new producer. Diff a new architecture's tensor names against what the loader reads.

→ `docs/invariants/inference.md` § "A model's RoPE frequency factors are applied — and reach a node without shard 0"

## A recurrent state is taken back into its last check by replaying the rows it kept, or not speculated (2026-10-06; 2026-10-25, #228 (5))

`SsmState` keeps the pair its last step READ and, for a step of 2..=`SsmState::MOST_ROWS_KEPT` (32) positions, its inputs (`SsmRows`). **A truncate INTO the last step replays its accepted rows** (`rewind_to` → `qwen35::replay_rows` via the ONE kernel choice a step makes, `qwen35::delta_rule_into`; card AND processor; `SsmState::holds_a_step_to`, asked by `every_cache_holds`). A wider step keeps none; past the last step's start or rows is refused (`can_rewind_to` → `KvCacheEntry::truncate_to`). The worker's n-gram truncates a rejected check as any model's (`model_worker::ngram_spec_round`, `LocalSpec::next`; held to `MOST_ROWS_KEPT`; captured on a card, `mask_is_read` counts its layers as dense; `a_checks_row_reads_nothing_after_it`). **A split's n-gram loop** does so on peers only where every peer segment advertises `features::RECURRENT_REWIND` (bit 33) — **`pipeline::checks_can_be_taken_back`**, asked before the loop and each round; a worker whose truncate is refused refuses the forward. **Every other path asks first and stays off**: **`pipeline::distributed::speculation_can_roll_back`** ahead of the drafter's coordinator paths; SWIFT and a drafter with recurrent state (**`SplitModel::carries_recurrent_state`**, `engine_drafter`). Tests: `a_check_rewound_into_its_middle_is_where_its_kept_rows_lead`, `a_guessed_reply_is_the_reply_read_one_token_at_a_time` (ids EXACT on the processor up to 8 rows, #426).

→ `docs/invariants/inference.md` § "A recurrent state is taken back into its last check by replaying the rows it kept"

## A file's own guess head drafts where the n-gram has nothing (2026-11-07, #425)

Qwen 3.5/3.6/3.8 files carry a next-token-prediction block (`nextn_predict_layers` 1, `blk.{N}` past the trunk). **`split::mtp`** loads it only for a WHOLE-model segment (`block_to_load`), placed as the last layer is (`LayerPlacement::for_the_next_token_block`), its bytes off that device's KV budget (refused past a quarter). **`model_worker::MtpGuesses`** takes the trunk's last-layer rows of COMMITTED positions only (`MTP_PROMPT_ROWS`), normed by the output norm (`mtp_hidden_from`). `LocalSpec::next` asks it where the n-gram found nothing, declined, or is kept less often; ≤ `MOST_GUESSES` (4), counted by `best_gamma_beside_a_step`; the trunk's check is the n-gram's (`ngram_spec_round`). Any failure stops it for the reply (WARN), never fails one. On a card the captured step keeps the head's row itself (`executor::graph_keeps_rows`).

- **Beside a split's first segment** (#425 (a)): where this node runs the first segment from layer 0 and the parts hold the block, `output_norm` and `output.weight` whole (`loader::shards::guess_head_parts_held`) — fed the last layer's rows by whichever computer runs it, this one or a peer (`BlockRowsAsk`): `mtp::HeadBeside::FirstSegment` (`trunk_norm`, `model_worker::SplitHead`); `pipeline::block_draft::SplitBlockDrafter` chooses it or the block drafter (`choose`, `dsd_controller::rated_beside_a_step_per_{round,guess}`). A/B `SWARMLLM_SPLIT_GUESS_HEAD=0`.
- **A first segment's full card sends it to system memory** (#485): **`mtp::split_head_home`** is the one answer (card where block + output head fit beside one conversation, else the machine's live room at a quarter, else nowhere; `LayerPlacement::for_the_block_in_system_memory`); the output head is read WHERE ITS BLOCK RUNS unless a block drafter needs it on the card (`SplitModel::guess_head_in_system_memory`), charged to the processor-side budget, scored on its own device (`guess_scoring_device`). Never read the head's small tensors off the TRUNK placement. A/B `SWARMLLM_SPLIT_GUESS_HEAD_IN_RAM=0`.
- **It guesses the next round while the check is abroad** (#425 (f), PEARL's post-verify): `model_worker::HeadChain` (`MtpGuesses::real_len`, `MtpGuesses::extend_ahead`) answers only where the check kept every guess and its bonus word is the chain's next (`serve_ahead`; `note_forward` ends it). A/B `SWARMLLM_SPLIT_HEAD_AHEAD=0`.
- **On the processor a guess reads fewer words** (#425 (b), FR-Spec): the head's first `mtp::DRAFT_HOT_ROWS` (32,768) rows plus the conversation's own words (`model_worker::DraftWords`), picked in ONE place, `SplitModel::mtp_guess`; from a reply's start also the vocabulary's ALPHABET (`mtp::alphabet_past`), built only by `DraftWords::for_prompt(model, …)` (#428 stage 4); every word is read where the prompt is a third past the hot rows, own words pass `DRAFT_MOST_OWN_WORDS`, or new words pass one in six; a card reads the whole head. A/B `SWARMLLM_MTP_VOCAB=full|<rows>`, `SWARMLLM_MTP=0`. Test: `the_models_own_guess_head_writes_the_reply_read_one_token_at_a_time`.

→ `docs/invariants/inference.md` § "A file's own guess head"

## A block drafter guesses a whole block in one forward, from the model's own layers (2026-11-08, #428)

A DFlash drafter (`block_drafter.gguf` beside a WHOLE model) is loaded by **`split::block_drafter`** (`SplitModel::attach_block_drafter`, on the target's device). Its weights AND whole cache (`BLOCK_MOST_POSITIONS`) — and on a split the borrowed head (`drafter_head_bytes`) — come off the conversation budget while ONE conversation of the model's whole context still fits (`fits_beside_one_conversation`), else refused; weighed off its header BEFORE it loads (`block_drafter::charged`). It has no token table or head: the block `[next token, mask × γ]` is embedded by the TARGET's table and read by its head (`SplitModel::embed_ids` / `logits_from`). **Its scales are the target's own** (`dflash.embedding_scale` and soft cap load only where they match: `SplitModel::embedding_scale`, `block_drafter::DrafterFigures::refusal_beside`); a weight-tied target's head IS its token table, lent by a first segment (`drafter_reads_the_table`, `head_part_for_the_drafter`). Context = the target's OUTPUTS of `target_layers` for COMMITTED positions only (fused by `fc` + `hidden_norm`, NOT normed by the input norm). **`model_worker::BlockGuesses`** feeds it (`BLOCK_PROMPT_ROWS`; a missed position stops it for the reply); `LocalSpec::next` asks it before the guess head; count **`dsd_controller::best_gamma_beside_a_step_per_round`** (a draft costs one forward whatever it guesses).

- **Processor**: guesses read the guess head's hot rows and the conversation's words (`DraftWords`, `SplitModel::mtp_guesses`); with no hot rows it keeps its own (`SplitModel::drafter_hot`, `IpcBlockDraft::words`); A/B `SWARMLLM_MTP_VOCAB=full`. **A whole model on the processor with its own guess head attaches NO block drafter** (`SplitModel::head_guesses_better_here`); A/B `SWARMLLM_BLOCK_DRAFTER=processor`.
- Context rows are made at the next draft (`BlockGuesses::raw`, `join_row_pieces`), never per round. Direction: the file's `dflash.attention.causal`, else z-lab's rule; `SWARMLLM_BLOCK_DRAFTER_CAUSAL=none` attends both ways.
- **Card**: guesses picked there (`tree_rank::greedy_picks`), charged to the card's budget (`kv_budget_bytes`), EVERY captured step keeps the target layers' rows (`executor::graph_keeps_rows`); `LocalSpec` sets the block aside only where it writes less than plain steps (`arch-scheduling.md` § "Speculation that does not pay").
- **Split** (stage 3): beside the coordinator's FIRST segment (range starts at layer 0, head part on disk, `loader::shards::drafter_head_info`), fed the rows the segments send back (`arch-network.md` § "A block drafter's rows come back") by the n-gram loop (`pipeline::block_draft::SplitBlockDrafter`, γ 0 included), joined by `model_worker::block_rows_for_the_drafter` (contiguous: `fc` refuses a strided join); a gap or refusal stops it for the reply. **Priced as one computer's**: own step, own line from checks only (`dsd_controller::Step::OverTheChecksRoute`), probed again after `FIRST_PROBE_AFTER` declines (`arch-scheduling.md` § "Speculation that does not pay").
- **Fetched by a PIN** (stage 6, `model::huggingface::block_drafter`): `general.name` folded, architecture and width → one repo, file, size, SHA-256 (`pinned_for`), never a name search; ≤ `MOST_DRAFTER_BYTES` (2 GiB), after `metered_hold` and the disk reserve; only when NEITHER end is held (`Ends::Gone`).
- **A split's card with no room puts it in SYSTEM MEMORY** (#451; A/B `SWARMLLM_BLOCK_DRAFTER_IN_RAM=0`): **`block_drafter::drafter_home`** — the card first, else the KNOWN processor-side budget, same one-conversation rule; a whole model only.
- **DFlash 2** (Qwen3.8-27B, stage 2, 2026-11-22): four `DynamicConv` a layer (a half's coefficients shared by its two sides, block rows only) and a `Selector` (`selector_walk`: top-k by predecessor, the first max; codebooks stay in SYSTEM memory, `Codebook`), each checked against llama.cpp's index arithmetic; a tree's trunk is the selector's chain (`with_picks_first`). A/B `SWARMLLM_DFLASH2_SELECTOR=0`. → § "DFlash 2"
- **Never where no load attaches it** (`block_drafter::attaches_here` — `auto_manage::vram::models_go_to_the_card`, `split::mtp::head_in_shape`, `mtp::block_to_load`): none fetched, one on disk given back (`Step::NotUsedHere`). Off: `SWARMLLM_BLOCK_DRAFTER=0`. Tests: `a_blocks_mask_is_the_references`, `a_block_drafters_guesses_write_the_reply_read_one_token_at_a_time`.

→ `docs/invariants/inference.md` § "A block drafter"

## Qwen 3.5's delta rule is ONE op per layer (2026-10-11)

`layers::qwen35::forward_deltanet` runs a layer's recurrence in one op: **`delta_rule_cpu`**, on a card **`delta_rule_cuda`** → `kernels/delta_rule.cu` (scope `card_delta_rule_covers`). Value head h reads key head h % nk. **A Qwen3-Next file groups them (h / r)**: rows reordered at load, still quantized (`GroupedValueHeads::kernel_order`), `ssm_out` reads its input put back (`ValueHeadOrder`) — never a second order in a kernel. A family awaiting its real-file check is refused unless `SWARMLLM_UNCHECKED_FAMILIES` names it (`ModelArch::is_supported`, #390). NOT bit-identical to the composition `delta_rule`, which stays as the A/B (`SWARMLLM_DELTA_RULE=steps`) and the reference of `one_pass_delta_rule_matches_token_by_token`. Profile stage `delta rule`.

- **The card's delta rule gives a warp FOUR state rows** (`delta_rule_*_c4`, `qwen35::delta_rule_entry`; A/B `SWARMLLM_DELTA_RULE_ROWS=1`): a layer's warps fit the card at once and share each token's q, k and gates; each row's arithmetic unchanged, so BIT FOR BIT one row a warp (test `four_state_rows_a_warp_are_one_rows_bits`). → `docs/invariants/inference.md` § "Four state rows a warp".
- **Card convolution + SiLU is ONE launch** (`DeltaNetWeights::conv_on_the_card`) **and a taken-back check's replay TWO** (`cuda::replay`) — BIT FOR BIT the composition (`__fmul_rn` / `__fadd_rn`, candle's `silu_fwd`). Tests `the_card_convolution_is_the_composed_one_bit_for_bit`, `a_replay_on_the_card_lands_where_the_composed_replay_does`; A/B `SWARMLLM_SSM_CONV=composed`, `SWARMLLM_SSM_REPLAY=composed`.
- **Gates and q/k's L2 norm are one launch each** (`DeltaNetWeights::gates`, `ssm_gates_project_f32_r*` / `ssm_gates_f32`; `l2_norm_on_the_card`; `Gates::stacked`). NOT bit-identical (a sum's order): each row reads only itself; judged against llama.cpp. **Q8_0 `ssm_alpha` / `ssm_beta` (the 9B's, the 0.8B/2B drafters') are read IN PLACE by the one launch up to 8 rows** (`ssm_gates_project_q8_0_r*`, `cuda::q8_0_gate_weights`; bit for bit the f32 launch over the dequantized weights) — never dequantized at load: no budget counts an f32 copy. Tests `the_card_gates_are_the_composed_ones`, `the_card_gates_after_quantized_projections_are_the_composed_ones_bit_for_bit`, `the_card_gates_read_q8_0_weights_as_their_f32_dequantization_bit_for_bit`, `the_card_l2_norm_is_the_composed_one`, `a_deltanet_layer_on_the_card_is_the_processors`; A/B `SWARMLLM_SSM_GATES=composed`, `SWARMLLM_SSM_GATES=matmuls` (Q8_0 through the matmuls), `SWARMLLM_SSM_NORM=composed`.

→ `docs/invariants/inference.md` § "Qwen 3.5's delta rule is ONE op per layer", § "Qwen3-Next: value heads grouped by key head"

## A sliding window is cut where it binds, and its cache keeps the window (2026-10-15/16, #278, #281)

**`split::SlidingWindow::read`** is the one reading of a model's window (llama.cpp's `LLAMA_SWA_TYPE_STANDARD`, per family — Gemma 2: even layers, 4096; Phi-3's is off). **`LayerWeights::keys_in_window`** applies it, once, for both attention entry points (K/V narrowed from the cache's LENGTH less **`LayerKv::held_from`**); several queries read **`ForwardMasks::for_layer`**; flash gets `window_size_left = W - 1`. A/B `SWARMLLM_SLIDING_WINDOW=0`.

**A windowed layer's cache drops what its window no longer reads** (#281): **`LayerKv::slide_beyond`** (told by every forward, `layers::append_to_cache`) keeps `positions_kept_for_window` (W − 1 + `WINDOW_REWIND_POSITIONS`); `current_seq_len` stays absolute. **A rewind past `LayerKv::serves_from` is REFUSED** (`can_truncate_to`). A snapshot carries `held_from` / `serves_from`; **a slid snapshot never leaves the node** (`Entry::manifest` empty, `serialize` refuses). An f16-mirrored cache (`F32Mirrored`) never slides. **Charged as its window** (#281 (2)): **`kv_budget::KvGrowth`** is the one price of n positions (`most_positions_a_window_holds`); head-room reads the buffers (`KvCacheStore::bytes_a_forward_adds`); `kv_cache::window_kept` / `caches_slide` say whether it slides. **Estimates read the loader's layout** (#281 (3)): `kv_budget::KvLayout` → `KvPrice` (`VramFootprintInputs::kv`, `KvPrice::reading`). **A coordinator's bound on a PEER too** (#281 (4)): `scheduler::prompt_kv_bytes_per_layer` → `KvLayout::growth_on_peer`, at the window ONLY where the peer advertises `features::KV_WINDOW_SLIDES` (bit 22, not in `ALL`; `kv_budget::windows_advertised`, chunk 1..=`COPIED_CHUNK_MAX_TOKENS`). Prefix cache: `prefix_cache::SnapshotCost` (#281 (5)). A/B `SWARMLLM_KV_WINDOW_SLIDE=0`.

**A family's attention scale is read, never assumed `1/sqrt(head_dim)`** (#279): **`split::query_scale`** (Gemma 2's 27B = 46 layers: `1/sqrt(n_embd/n_head)`; the 9B is NOT; Granite's `attention.scale`, its other multipliers refused by name, `gguf_meta::unbuilt_granite_part`) → `LayerWeights::query_scale`; a new family's `f_attention_scale` is read off llama.cpp's `load_arch_hparams`. A/B `SWARMLLM_QUERY_SCALE=0`.

**A layer's rotation is read per layer** (Gemma 3, #115 (1)): **`split::rope::LayerRope::read`** — `rope.scaling` on every position, linear or YaRN (`RopeScaling`, #280), and windowed layers' own base/scale; **`LayerPlacement::rope_for`** is the one place a layer gets its tables; windowed layers come from `SlidingWindow::from_file`, never the A/B-switched `read`. A/B `SWARMLLM_ROPE_SCALING=0`, `SWARMLLM_LAYER_ROPE=0`.

**A layer's attention SHAPE is read per layer** (Gemma 4, #115 (3)): **`split::attention_shapes`** (`head_count_kv` per layer; `key_length_swa` / `rope.dimension_count_swa` on windowed layers); a model-wide figure (`GgufTensorMeta`'s, a peer's bound) is the WIDEST layer, never layer 0's; estimates price each layer (`kv_budget::KvLayout`). Gemma 4: V = K where `attn_v` is absent (`value_is_key`), V RMS-normed with no weight (`LayerWeights::v_rms_norm`), `layer_output_scale` (`executor::scaled_output`), queries × `sqrt(head_dim)` (`query_scale`), windowed layers turn their own width without `rope_freqs` (`rope::WindowedRope`); experts with a gate/up scale refused by name (`unbuilt_gemma4_part`). Reference: llama.cpp MASTER (`~/llama.cpp-ref/dump_logits`).

**The E2B/E4B's shared layers attend another layer's cache** (#115 (3c)): **`split::kv_sources`** (last earlier kept layer of the same kind — HF's rule; gotcha #884), **`LayerWeights::kv_slot`** the one answer to which slot a layer attends; a shared layer keeps no cache (`split::kv_layers` charges none), SWIFT never skips a layer that lends one, and a split with source and sharer on different devices goes through `layers::history_beside_queries`. **Per-layer inputs are built once per forward from the TOKENS** (`split::per_layer::PerLayerInputs`, `gguf_meta::host_resident`) and added in ONE place, **`executor::scaled_output`**, which refuses a layer with an embedding but no input. Only the first segment sees tokens, so **`GgufTensorMeta::runs_only_whole`** keeps such a model to ONE computer — the scheduler (`assemble_pipeline_for`), loader, local embedder and tensor parallelism each refuse otherwise; a new path that splits or pre-embeds asks it.

→ `docs/invariants/inference.md` § "A sliding window is cut where it binds", § "A windowed layer's cache keeps its window", § "A windowed layer is charged as its window", § "A family's attention scale is read", § "Gemma 3: each layer turns at its own rate", § "A file's RoPE scaling is read for every family", § "Gemma 4: each layer's attention has a shape of its own", § "Gemma 4's E2B and E4B: shared caches and per-layer inputs"

## A model that cannot write replies is refused off its header (2026-10-15)

**`GgufTensorMeta::refusal`** (→ `ModelRefusal`, one wording via `into_error`): no loader for the family, or an embedding model (`<arch>.pooling_type` non-zero — Ollama's rule). Asked by the loader, the router (`SharedState::refused_model`), auto-manage, the user's download, `/v1/models` and the dashboard list (`cannot_chat`). Before a header exists, `model::huggingface::is_embedding_repo` reads HuggingFace's tags. A new path that runs or fetches a model asks it.

→ `docs/invariants/inference.md` § "A model that cannot write replies is refused off its header"

## A header and its tensor table describe ONE upload — compared before a tensor is read (2026-10-03)

`split::loader::shards::first_header_disagreement` runs in `load_from_shards_inner`, the one place every load from parts passes: every table entry the parts hold must sit at the header's offset with the header's size (`split::tensor_byte_size`), or the load is refused as `SwarmError::MixedModelCopy` (#156). Compare per entry, never by re-deriving the layout. Rig `split_rig.sh mixed`; null control `every_real_copy_agrees_with_its_own_header`. Parts whose BYTES are not the table's upload (a splice) are invisible to it: `auto_manage::canonical`'s byte check deletes and re-fetches them (gotcha #782; rig `split_rig.sh spliced`).

→ `docs/invariants/inference.md` § "A header and its tensor table describe one upload"

## A RoPE layout is read off llama.cpp, per architecture

**`ModelArch::use_rope_contiguous` is llama.cpp's `llama_model_rope_type`**
(NEOX = contiguous, NORM = interleaved), arch by arch (GLM-4, Llama-4 and
DeepSeek-2 are NORM, #96). **A new arch gets its row
in `model_arch_properties` from llama.cpp's list, never from this function.**
Byte-identical replies across releases prove no REGRESSION, never correctness.

→ `docs/invariants/inference.md`

## An adapter meets the model in the model's row order (2026-09-25)

**`lora::QkRowOrder`** is the single answer to the q/k row order an adapter meets — a REQUIRED argument of the adapter loader (Llama/Mistral are reordered by llama.cpp's `LlamaModel.permute`). **`lora::check_fits`** refuses what the executor would silently skip. Verify with `examples/peft_lora_to_gguf.py` + `score_against_reference.py --lora`, on an adapter whose B is NOT zero.

→ `docs/invariants/inference.md` § "An adapter meets the model in the model's row order"

## A special token is what the vocabulary says it is, and a prompt gets ONE BOS

**`tokenizer::matched_whole`** is the ONE rule for what a prompt matches whole, both encoders (#334 (3)): what `token_type` declares — UNKNOWN / CONTROL / USER_DEFINED (`declared_special`, llama.cpp's `cache_special_tokens`) — and a `<…>` name SHAPE only for a vocabulary declaring no types. Never a shape beside declared types (`<>` in code, Qwen's ordinary `<s>`, a byte entry `<0x0A>`), never llama.cpp's end-of-generation NAMES. **`tokenizer::split_at_special_tokens`** is the one splitter; Gemma 4's text between markers goes through `BpeTokenizer::encode_gemma4`. **`tokenizer::PieceDecoder::decode`** is the one piece → bytes answer: a `<…>` piece the vocabulary does not declare text is hidden (gotcha #866). **`gguf_meta::add_bos_by_llama_cpp_rules`** decides whether a prompt gets a BOS; **`SplitTokenizer::encode`** adds none to a text already opening with one; **`gguf_meta::special_token_spacing_for`** covers what the GGUF omits. Check with `examples/tokenizer_reference.py` + `tokenizer_agrees_with_llama_cpp`. ⚠ For SentencePiece whitespace llama.cpp is no authority: `examples/tokenizer_hf_reference.py`.

→ `docs/invariants/inference.md` § "A special token is what the vocabulary says it is — and a prompt gets ONE BOS"

## An image sits at its marker, and its rows are counted as positions (2026-10-19, #115 (2a))

Every surface writes **`vision::MEDIA_MARKER`** where an image was (`vision::join_content`); **`vision::encode_prompt_with_images`** gives each marker one `IMAGE_TOKEN_ID` and the prompt one BOS; **`vision::splice_image_rows`** is the ONE place rows meet text (executor + local embedder) — no marker, or rows the markers do not divide, is REFUSED, never prepended; **`vision::positions_with_images`** is the one position count (rows from `vision::wire_rows`, never a cached module); **`vision::decode_wire_embeddings`** the one payload reader. A failed encode is the request's error. The payload goes only to a first segment advertising `features::IMAGES_AT_MARKERS`. A projector is dispatched on `clip.projector_type` — a new one is read off llama.cpp's `tools/mtmd/models/<type>.cpp`, preprocessing included. Check against `llama-mtmd-cli` (LLaVA: exact arithmetic, § "An image reader is read off its file").

→ `docs/invariants/inference.md` § "An image sits at its marker"

## A picture whose rows attend both ways is read whole, by every segment (2026-10-19, #115 (2b))

**`vision::rows_attend_both_ways`** names the families (llama.cpp's `mtmd_decode_use_non_causal`: Gemma 3). **`PipelineExecutor::image_spans_for`** computes the spans once per request from `encode_prompt_with_images` + `wire_rows`; **`image_spans_from(index_pos)`** is the one answer to what a send carries (every forward reaching into a picture, every segment, stand-ins and chained hops too — each must advertise `features::IMAGES_AT_MARKERS`). The worker takes them only through **`model_worker::image_spans_to_apply`** and applies them with **`SplitModel::with_image_spans`**; the mask is **`causal_mask_with_spans`** / `window_mask(.., spans)`, a forward must hold each picture whole (`spans_in_forward`), reads the mask (`ForceStandardAttnGuard`, set INSIDE `forward_inner_body` — `in_phase_pool` may run it on another thread), and is never graphed, batched or tensor-parallel. Check against `llama-mtmd-cli` as shipped.

→ `docs/invariants/inference.md` § "A picture's rows attend both ways"

## An image reader is read off its file, and LLaVA's is judged by exact arithmetic (2026-10-19, #310)

**`vision::load_from_mmproj_gguf`** reads what llama.cpp's `clip.cpp` reads: **`EncoderActivation::from_keys`** (`clip.use_gelu` → tanh, `clip.use_silu`, neither → CLIP's quick GELU), `v.post_ln` only where the file has one (`final_ln: Option` — never a weight-1 norm), **`encoder_depth`** (every block the file holds; one `clip.vision.feature_layer` → that many; several refused). **`VisionEncoder::forward`** drops a class row — a LLaVA picture is 576 rows (`TOKENS_PER_IMAGE`). `Preprocess` letterboxes on its projector's `pad` (LLaVA: `mean_as_bytes`, bicubic; Gemma 3: black, bilinear). ⚠ **LLaVA's rows are judged against `examples/llava_reader_reference.py` (float64), not llama.cpp** (A/B `SWARMLLM_LLAVA_DEPTH=llama-cpp`). Probe: `examples/image_rows_probe.rs`.

→ `docs/invariants/inference.md` § "LLaVA's image reader is checked against exact arithmetic"

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path (the single-character early return and
the output walk both need it; both were **`.unwrap_or(0)`**, i.e. `<unk>`).

→ `docs/invariants/inference.md`

## A mixture-of-experts layer keeps its experts QUANTIZED (2026-09-25)

**`split::loader::load_moe_ffn` is the one way a MoE feed-forward is loaded** — experts stay QUANTIZED (`split_expert_stack`); a new MoE family calls it. Routing defaults are per family (`moe_renormalizes_by_default`, `moe_gating_by_default`, `moe_weights_scale`), read off its llama.cpp loader. Verify with `examples/logits_reference_probe.rs` + `compare_logits_reference.py`; one outlier is a near-tie, a shift everywhere is a bug. Off the card: one host routing copy (`topk_host`), ONE `index_add` per layer. **On a card the experts stay STACKED** (`StackedExperts::fit`, `MoeFfn::experts_stacked`): routed on the card (`topk_on_device`), ONE `moe_gemm_gguf` launch per projection (`layers::moe_card`; guard `the_vendored_moe_prompt_kernel_keeps_its_barrier`; operands go through `own_storage`). **`moe_card::kernel_type` is the one list of types the kernels read** — Q8_0, Q2_K-Q6_K, the vendored 32-block Q4_0-Q5_1 (#292), MXFP4 (gpt-oss, #390 step 7); an id a kernel does not launch is uninitialised output, guard `every_type_the_card_moe_kernels_are_handed_is_launched`. A/B `SWARMLLM_MOE_CARD=per-expert`, `SWARMLLM_MOE_CARD_PROMPT=rows`. ⚠ Run each arm TWICE — two replies from ONE arm is a race, not numerics (gotcha #827). **On the processor every chosen expert runs in one parallel region per projection** (`MoeFfn::experts_in_one_region` → vendored `quantized::matmul_many`; `MOE_PAIRS_PER_REGION`), bit-identical to `experts_one_at_a_time` (`one_region_experts_are_bit_identical_to_one_at_a_time`, `many_row_products_in_one_region_match_forward`); A/B `SWARMLLM_MOE_CPU=per-expert`. Profile stages `moe ffn on the card / on the processor`.

**A layer's feed-forward runs through `FfnVariant::forward(normed, attn_out, lora)`** — every executor site, never a `match` on the variant (#115 (3b)). Gemma 4's 26B-A4B runs a dense branch BESIDE its experts (`layers::DenseBeside`; router on its own norm of `attn_out`, `1/√n_embd` folded in); `MoeFfn::forward` and tensor parallelism refuse it. `ExpertScales` (`ffn_down_exps.scale`) multiply the picked weights on every path (`expert_weights_scale` folded in at load); a selection bias (`MoeFfn::selection_bias`, `exp_probs_b`: DeepSeek-V3, GLM-4.5) is a REQUIRED argument of both routers (`topk_host`, `topk_on_device`) — it CHOOSES, never weighs; a fused `ffn_gate_up_exps` is cut at load (`split_expert_stack_rows`); activation per layer (`MoeFfn::activation`); a gate/up scale is refused by name. **gpt-oss**: a ROUTER bias weighs as well as chooses — read only through `MoeFfn::router_scores`; expert biases (`MoeFfn::expert_biases`) are added after each product on every path (`layers::expert_bias_rows`, by the UNSORTED flattened picks; down product unweighted, bias, then routing weight — llama.cpp's `build_moe_ffn`); `attn_output.bias` only in `LayerWeights::output_projection`. **Attention sinks** (`LayerWeights::attn_sinks`) are an argument of `standard_attention` and every kernel under it (`attn_rows::fold_sink`, `scaled_masked_softmax`, `decode_attn_combine_f32`); flash and the tree kernel read none, so a sink layer's prompt and tree checks take the matmul path (`LayerVariant::flash_may_attend`). `swiglu_oai` is ONE card launch (`fast_math::swiglu_oai_mul`). ⚠ Judge a tiny MoE file against llama.cpp's processor AND card builds; the decisive file is soft attention with every expert used (`N_EXPERT_USED=8`).

**A card/processor split's processor-held experts read a prompt ON THE CARD** — see § "A card/processor split multiplies a prompt on the card".

→ `docs/invariants/inference.md` § "A mixture-of-experts layer keeps its experts quantized", § "Gemma 4's 26B-A4B: a dense feed-forward beside GELU experts", § "GLM-4.5: experts CHOSEN with a bias", § "gpt-oss: a router bias that weighs", § "gpt-oss: attention sinks"

## A card/processor split multiplies a prompt on the card (2026-10-13, #138 (1)/(1b)/(1c))

llama.cpp's op offload (`ggml_backend_sched` "1.off"): a forward of ≥ `PROMPT_ON_CARD_MIN_TOKENS` (32) copies each processor-held weight to the card for the call. **`layers::QMatMul::forward` / `forward_shared` are the ONE choke point for projections** (`QMatMul::prompt_card`; a fused weight copied once, its f32 mark with it); chosen experts via `QTensor::stack_some` (`MoeFfn::processor_experts_on_the_card`). A refusal runs the processor path (WARN once). The loader gives the card ONCE, to host-memory weights only (`layers::let_processor_layers_read_prompts_on`, `prompt_card_for` the one token rule). **`SplitModel::prompt_reader(index_pos)` is the chunk-ceiling question** (`prefill_pacer::prompt_chunk_ceiling`; each chunk sized as it starts): at least 512 a forward and, from the card's room NOW, up to 4096 (#138 (1e)). Measured, never assumed: each prompt forward records what the pool took (`begin_prompt_measure` / `end_prompt_measure`, `USED_MEM_HIGH`, read BEFORE the profile's report resets it) and predicts the next (`MeasuredPrompt::predicted_bytes`; without the cache the forward added, #292); ⚠ **the pool is grown ONCE before the forward** (`SplitModel::prime_pool_for_prompt` → `cuda_pool::prime`, via `kv_bytes_a_forward_adds`), left alone when free memory covers it (#292, gotcha #883; A/B `SWARMLLM_PROMPT_POOL_PRIME=0`); a card fallback or forced standard attention pins 512. Never `runs_entirely_on_card` (the drafter's and admission timing's). A/B `SWARMLLM_PROMPT_CHUNK_ROOM=0`, `SWARMLLM_PROMPT_CARD_PROJECTIONS=0`, `SWARMLLM_MOE_PROMPT_CARD=0`.

**A processor-held `LayerVariant::Dense` layer runs a ≥ 32-token chunk WHOLE on the card** (1c): `LayerOnTheCard::of` copies its norms, biases and RoPE rows; `card_for` is the gate (marked attention and feed-forward, no adapter). **The KV cache stays HOME on the processor** (`attend`'s `cache_home`) — never give a processor layer a card cache. **A row a forward keeps goes to the MODEL's device** (`Capture::kept_on`, the device REQUIRED): a 32+ chunk runs a processor layer on the card and a shorter one on the processor, so one prompt's kept rows came from both and their join failed (#478). **`to_device` on a view copies the WHOLE storage**: `force_contiguous()` first. A refusal cuts the cache back (`truncate(held)`) and reruns on the processor (WARN once); the batched loop keeps (1b). A/B `SWARMLLM_PROMPT_CARD_LAYERS=0`.

→ `docs/invariants/inference.md` § "A split's processor-held experts read a prompt on the card", § "A split's processor-held projections read a prompt on the card", § "A split's processor-held layers read a prompt whole on the card"

## A card/processor split is placed in EVERY per-layer loop, and offered only where the loader makes it (2026-09-25)

**`split::hybrid::LayerPlacement` says where each layer goes; every per-layer loop in `split/loader/` shadows `device`, `cos`, `sin` from it on its first line.** `hybrid::layers_on_device` is the KV budget's one count; `hybrid::arch_supports_hybrid` is an allowlist the pool reads too (`process_pool::split_for_card`). A range's `gpu_layers` / `cpu_expert_layers` come from **`split::placement_of`** — the daemon's placement for that range (`DaemonMsg::PlaceRange`), else the spawn's flags (#265). Guard: `every_layer_loop_in_the_loader_places_its_layer`. Check with `swarmllm test-split --gpu-layers N` + `~/swarmllm-ref/qwen3moe/score_ids.py`.

**A card-held MoE layer may keep its routed experts in RAM** (#138 (3), llama.cpp's `--n-cpu-moe`): `LayerPlacement::experts_device_for` is the one answer and **`load_moe_ffn` takes the PLACEMENT, never a device**; the card budget takes `routed_expert_bytes` off. Such a layer works on the processor every step — ask `SplitModel::cpu_layer_count` / `works_on_the_processor`, never `layer_devices` alone. `test-split --cpu-moe-layers N`. The pool chooses it for a MoE model too big for the card — **`process_pool::card_split`** is the one placement decision, counted by **`split::routed_expert_bytes`**, which the loader uses too; A/B `SWARMLLM_EXPERTS_IN_RAM=0`.

→ `docs/invariants/inference.md` § "A card/processor split is placed in every per-layer loop", § "A card-held layer can keep its experts in RAM"

## A whole model's processor run sits early in the middle, so its card can guess (2026-10-28)

**`hybrid::processor_run(len, on_card, place)` is the one answer to which layers a split keeps on the processor**, its place from **`hybrid::run_place`** (`RunPlace`): where the card could guess without the run — **`hybrid::card_may_guess`**, off the header: the whole model, every layer keeping its own cache (`kv_layers`; not Qwen 3.5's recurrent layers nor Gemma 4 E2B/E4B's shared ones) — the run starts at `round(0.1 · len)` (`PROCESSOR_RUN_STARTS_AT`) so the card holds both ends (`MiddleForGuessing`; a card under a fifth, `CARD_GUESS_MIN_SHARE`, takes the tail); **a range that does not start the model gives the processor its FIRST layers** (`Start`, llama.cpp's `i_gpu_start` order, #461 (4)) so a range widened at its start keeps the layers it held on the card; a first segment keeps the card-first tail (`Tail`). Every reader asks it: the loader (`LayerPlacement::split`), `plan_gpu_layers`, the pool's charges (`LayerBytes::run_place` → `ram_charged_share`, `card_share_of_split_mb`), the planner (`NodeCandidate::processor_run_first`; a peer by `features::PROCESSOR_RUN_FIRST`, bit 45, NOT in `ALL`). A/B `SWARMLLM_PROCESSOR_RUN_FIRST=0` (Start → tail). The card then guesses where the n-gram finds nothing (`model_worker::CardGuesses`: `SplitModel::card_guess_skips` → `SplitModel::card_guesses`, skipping the run, cache cut back before the check; `ngram_spec_round`, count from `best_gamma_beside_a_step` with its REQUIRED `draft_ms_each`).

- **A stand-in covers the skipped run** — `split::RunStandIn` (`stand_in.rs`), fitted from the hidden states every real step and check computes at the run's ends (`forward_keeping` + `Capture`; the prompt pass captures the stand-in's own `window()`), learned from COMMITTED positions only, dropped for the reply on any error, never a reason to fail one. A/B `SWARMLLM_CARD_GUESS_STAND_IN=none`.
- **It starts from earlier replies** (`split::StandInPrior`, via `SplitModel::stand_in_prior_for`): their last `PRIOR_ROWS` (2,048) positions on the HOST (`model_worker::absorb_the_reply` → `StandInPrior::take_in`), refitted while the worker has NOTHING waiting (`refit_stale_priors`) else on the next reply's first step — a dual kernel ridge inverted ON THE CARD (`stand_in::ridge_inverse`; `SkipBridge::Recall`). A prompt longer than the window joins the fit for its OWN reply only (`StandInPrior::fit_for_prompt`), **learned on the reply's first step, never before its first token** (`CardGuesses::take_the_prompt`). A non-finite position is never learned (`RunStandIn::not_finite`). A/B `SWARMLLM_CARD_GUESS_PRIOR=0`.
- **A check's miss is carried forward** (`RunStandIn::with_carry`: `STAND_IN_CARRY` (0.5) × the mean miss at the last `STAND_IN_CARRY_POSITIONS` (4) committed positions, measured BEFORE learning). A/B `SWARMLLM_CARD_GUESS_CARRY=0`.
- ⚠ **A forward reports only the processor layers it RUNS** (`SplitModel::cpu_layers_run`, gotcha #983); **a speculating loop steps plainly until the decode width settles** (`SplitModel::decode_width_settled`, gotcha #984); a guessing worker must keep speculating (`drift_toward_retrying`, gotcha #985). Checks use every core (`cpu_pools::check_pool`; `examples/skip_agreement_probe.rs`). A/B `SWARMLLM_PROCESSOR_RUN=tail`, `SWARMLLM_CARD_GUESSES=0`.

→ `docs/invariants/inference.md` § "A whole model's processor-held layers sit early in the middle, so the card can guess"

## A chain that starts and finishes here guesses through a stand-in for its far run (2026-10-29, #366 (4))

Where this node runs a chain's first segment `[0, a)` AND its last `[b, L)` with a peer between ("Start and finish on this computer"), the n-gram loop guesses where it finds nothing to copy — **`ngram_only_spec::FarRunGuesses::for_plan`** is the one answer (both ends local, a peer between, no tensor-parallel group, `speculation_can_roll_back`). The worker learns a `RunStandIn` for the PEERS' layers (`RunStandIn::for_far_run`) from what it holds — the first segment's output and the last segment's input at the same positions (`FarRuns::keep`, `kept_pairs`) — never the far layers' weights, nothing new on the wire. A guess runs on the request's OWN caches and cuts both back (`split::guesses_through_far_run`: first → `SkipBridge::apply` → last, gotcha #508), via `DaemonMsg::HeadGuess`, never spawning a worker. A request learning one is never batched (`ModelProcessPool::learn_far_run`); γ from `best_gamma_for_check` at the guesses' measured cost — zero is an answer. A/B `SWARMLLM_FAR_RUN_GUESSES=0`.

→ `docs/invariants/inference.md` § "A chain that starts and finishes here guesses through a stand-in for its far run"

## One prefix-cache snapshot is sized by bytes, and keeps the opening (2026-09-26)

**`PrefixCache::positions_ceiling`** caps a snapshot BEFORE the copy: the explicit token ceiling if set, never more than the byte budget over what a position weighs. A longer prompt keeps its opening, never nothing. **A slid windowed layer is charged only the positions it holds** (`prefix_cache::SnapshotCost`, #281 (5)). **A snapshot COPIES (`force_contiguous`)** (gotcha #874). **A kept request's snapshot never leaves the node** (`Entry::kept_here`; an entry covering a kept one inherits it; the keep is a REQUIRED `KeepAsked` argument of `insert_from_kv`, #251 (2b)).

**A prompt's end-of-pass snapshot is copied once the REPLY is done** (`model_worker::prompt_saved_when_done`): after `GenerateDone` in `handle_generate`, in `finalize_slot` on the batched path (`Slot::save_when_done`) — never between the prompt pass and the first token. Not where a cache slides (`SplitModel::any_cache_slides`) nor a recurrent model. **Taken however the reply ends while the worker survives** (`model_worker::prompt_outlives`, asked by all three paths) and sized with no reply reserve (`ReplyAhead::Done`, a REQUIRED argument of every sizing). A/B `SWARMLLM_PROMPT_SAVE=at-prompt-end`.

**Room is made BEFORE the copy** (#367, `PrefixCache::make_room`, both inserts; llama.cpp's `server_prompt_cache::alloc` order). A/B `SWARMLLM_SNAPSHOT_ROOM=after-copy`. **The room is planned INSIDE the insert, with what the save replaces** (#367 (c)): an already-covered prompt plans nothing; the planner (a REQUIRED `RoomPlan` closure → `model_worker::snapshot_room`) gets the replaced entries' bytes per memory (`LetGo`) and returns its eviction (`SnapshotRoom::evict_bytes`), which only `make_room` makes. Never evict in a planner. **A restore TAKES the snapshot** (`hydrate_request_from_snapshot` consumes the `Arc`, #367): a request holding it through its own save kept the replaced snapshot's memory from the copy (332-952 ms of fresh pages against 118-506). Never keep a clone across the reply.

→ `docs/invariants/inference.md` § "One prefix-cache snapshot is sized by bytes"

## A recurrent model's prompt is saved short of its end, and restored whole (2026-10-18, #228 (8))

**`PrefixCache::save_point`** is the one answer to where a prompt pass saves: the end, or `RECURRENT_SAVE_BEFORE_END` (16) short of it for a model with recurrent layers (llama.cpp's context checkpoints). Both generate paths stop there (`Slot::save_at` cuts the chunk; `handle_generate` forwards in two) and save through **`model_worker::save_prompt_opening`**. The snapshot carries a COPY of each state (`KvSnapshot::ssm`), taken only at the attention layers' position, copied again on restore. Its `serves_from` is its whole length: never narrowed, never announced, exported or serialized; a recurrent model never probes a peer.

→ `docs/invariants/inference.md` § "Qwen 3.5's prompt is saved short of its end, and restored whole"

## Single-source-of-truth helpers — Inference kernels, caches and the tokenizer

Each names the ONE place a decision is made (`.claude/rules/architecture.md` § "One invariant, N paths"). **Read the topic file before changing one.**

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`: several-row AVX2 kernels for Q4_K, Q6_K, Q2_K, Q3_K and the 32-value formats (`avx.rs::rows_of_a_32_value_format`), bit-identical to the per-row loop; `k_quants::vec_dot_rows_chosen` is the one call (A/B `SWARMLLM_QMATMUL_ROWS=per-row`). A 2-3 row tail is ONE pass (`avx.rs::tail_in_one_pass`; A/B `SWARMLLM_QMATMUL_TAIL=per-row`). Q5_0/Q4_1/Q5_1 one-row AVX2 (A/B `SWARMLLM_QDOT_32=scalar`). ⚠ **No scalar `f16::to_f32` in a loop carrying a vector accumulator** (gotcha #920): `avx.rs::scale_product` / `block_scale_products`. → § "A 2-3 row tail shares one pass too", § "The 32-value formats run on AVX2"
- **`inference::decode_attn::gqa_decode_attention_cpu`** (each K/V row read ONCE per group; → § "It read the cache once per QUERY head"), **`inference::prefill_attn::gqa_prefill_attention_cpu`** (`q_len >= 2`), **`inference::attn_rows::chunk_partials` + `merge_into` / `finish_row`** (→ § "A row's attention on the processor does not depend on the forward it is in"), **`inference::fast_math`** (`exp_inplace`, `silu_mul`).
- **Vendored `rayon-core` (`vendor/rayon-core/src/sleep/mod.rs::spin_budget`)** — idle workers spin 200 µs before sleeping (8-thread decode 172 → 123-127 ms); A/B `SWARMLLM_RAYON_SPIN_US=0`; guard `rayon_core_is_the_vendored_copy_that_spins_before_sleeping`. → § "Idle rayon workers spin 200 µs before they sleep"
- **`inference::cpu_pools::in_phase_pool`** — one choke point; decode runs INSIDE a pool even at the global pool's width (`pool_for`) (#137). A bench reproduces a node only at its `RAYON_NUM_THREADS`.
- **`inference::layers::new_kv_cache`** — the only way to construct a KV cache; **`inference::split::kv_cache::LayerKv`** — one layer's; **`SeqCache` / `KvPair` + `LayerKv::truncate`** — **never re-introduce a copy on the rollback path.**
- **`split::kv_cache::positions_without_attention`** — where a segment with NO attention layer (a Qwen 3.5 run of recurrent layers only) stands: its recurrent states' positions. Every "how far do the caches reach" reading falls back to it — the store's count, a forward's offset, a batched step's (gotcha #1161). → § "A segment of recurrent layers only stands where its states do"
- **`inference::attn_softmax::scaled_masked_softmax`**; **`layers::standard_attention` grouped GQA decode** — no `repeat_kv` for `q_len == 1`; **`layers::cuda_decode_prefers_standard`** — `SWARMLLM_GQA_DECODE_FLASH=1` = old rule.
- **`inference::sampling::sample_among_top_k`** — never re-add a vocabulary-wide pass after top-k. → § "Top-k shrinks the candidate set"
- **`inference::sampling::argmax` + `tree_rank::greedy_picks`** — the ONE greedy pick, host and card: the first of equal maxima; never candle's `argmax` where a reply is decided (guard `no_reply_is_picked_by_candles_argmax`). → § "A greedy pick is the first of equal maxima, on the card too"
- **`inference::mem_bandwidth::measured_gbps`**; **`split::token_embedding::rows_on_demand_eligible`**; **`GgufTensorMeta::tied_output_location`**.
- **`inference::cancel::unless_cancelled`** — `InferenceRequest::cancel` is the ONE cancellation signal; read around every WAIT. **`KvCacheStore::set_cancel_oracle`** — `forward_was_cancelled` is the one reader.
- **`inference::split::read_gguf_header`** — the single way to parse a GGUF header off a PATH; repeated GEOMETRY reads go through **`split::HeaderCache`** (one parse while length and modification time stand, a stamp within 2 s of now never kept; `vocabulary_strings` keeps counts, not arrays) (#360). A/B `SWARMLLM_HEADER_CACHE=0`. → § "A header read for its geometry is parsed once while its file stands"
- **`SplitModel::embed_tokens`** — the ONE token lookup (f16 rows widened to f32, then Gemma's scale; gotchas #241, #901); guard `every_token_lookup_goes_through_embed_tokens`.
- **`model_worker::wire_token_ids`** — the ONE reading of a first segment's token ids off a forward's bytes: an id past the table (`SplitModel::embedding_rows`) is refused on the HOST. → § "A forward's token ids are checked against the table"
- **`SplitModel::logits_from`** — the ONE place a forward makes logits (single, batched, captured): Gemma 2's soft cap and the file's `suppress_tokens` (`SUPPRESSED_LOGIT`, finite: a peer's check rows must be, #282). → § "The output head is one place"
- **`split::holds_weights_the_card_cannot_read`** — IQ4_NL / IQ4_XS have llama.cpp's bits on the processor only (#404; `READABLE_IQ_TYPES`), so `process_pool::cpu_reason` keeps such a model there (`CpuReason::WeightFormatNeedsTheProcessor`); a type leaves it only with card kernels checked as MXFP4's were. → § "IQ4_NL and IQ4_XS: llama.cpp's bits on the processor, kept off the card"
- **`inference::shard_layout::part_serves_layers`** — does a part serve a run `[start, end)`? A part past the model's last layer (a next-token-prediction block, beside the head) serves the run that ENDS the model; every overlap asks it, never `a < d && c < b` (#398). → § "A part past the model's last layer serves the run that ends it"
- **`inference::shard_layout::parts_that_finish` / `parts_at_the_ends`** — the parts a reply finishes on (the last layer's AND the head's, maybe two) and those plus part 0; **`head_part`** is the one place the last part's NUMBER is named; never `shard_count - 1` as "the end" (#408; guard `no_production_code_judges_a_models_end_by_part_number`). → § "The parts a model finishes on are read off its layer ranges"

→ `docs/invariants/inference.md` § "Single-source-of-truth helpers — Inference kernels, caches and the tokenizer"

