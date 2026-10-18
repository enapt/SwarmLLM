---
paths:
  - "src/inference/split/**"
  - "src/inference/layers/**"
  - "src/inference/executor.rs"
  - "src/inference/attn_kernel.rs"
  - "src/inference/attn_softmax.rs"
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

`inference::executor::context_retry_ladder` is the answer to "what context size will this card accept": halve from the capped figure to a floor, serve the first size accepted, log what was granted. Never refuse a context a smaller one would fit. Called only from `llama`-gated code, so every default build reports it dead (gotcha #264).

→ `docs/FUTURE_WORK_ARCHIVE.md` § "A context that does not fit is refused instead of shrunk"
→ `docs/invariants/inference.md` § "A context that will not fit"

## Cross-feature compile checks

`cargo check` with default features compiles NO `cfg`-gated path; only the cache-warm workflow sees the GPU paths. ⚠ **Before removing anything an unused-warning points at, grep the file for `#[cfg(`** (gotcha #264, `DType` in `layers/mod.rs`). Changing a signature callable from gated code: run `cargo check --features llama` and `cargo check --features flash-attn --all-targets` (CI's form) before pushing, then `gh run list --workflow="Cache warm"`.

### The CUTLASS kernels live OUTSIDE `target/` (2026-08-17)

`CANDLE_FLASH_ATTN_BUILD_DIR` (`.flash-attn-build`) keeps the 19 kernels out of rust-cache's reach. "Cache hit" is not "the slow thing was cached" — confirm `All kernels up-to-date, skipping compilation` (#318). The directory must exist, and an empty env var is not an unset one.

→ `docs/invariants/inference.md` § "Cross-feature compile checks"

## A decode token is bound by GPU submission COUNT, not bandwidth (2026-09-22)

GPU decode is bound by submission COUNT, not bandwidth. Judge a change by launch count (`SWARMLLM_COUNT_KERNELS=1`, never on in a benchmark), not the clock; `SWARMLLM_ZERO_QMATMUL_BUFFERS=1` = old behaviour.

- **`CudaDevice::alloc_fully_overwritten`** only for a buffer the next kernel fully assigns.
- **`QMatMul::forward_shared`** — Q/K/V and gate/up share ONE activation; BOTH `mul_mat_vec_via_q8_1` and `mul_mat_via_q8_1` must; LoRA after it.
- **`layers::value_for_matmul`** — never `.contiguous()` a KV cache view before a matmul (`matmul_reads_rhs_in_place`).
- ⚠⚠ **ONE CUDA STREAM PER DEVICE** (`SWARMLLM_CUDA_OWN_STREAM`, `=0` legacy); a second needs `SWARMLLM_CUDA_EVENT_TRACKING=1`. The ONE exception: the copy stream of `CudaDevice::clone_htod_beside` (#138 (1d)) — it touches only its copy RING: the device's stream waits for each copy, copies the region card-to-card into a buffer of its own pool, and the ring rewrites a region only after that read. Never hand the copy stream a buffer candle frees: a pool reuses another stream's free only after a SYNCHRONIZE (gotcha #836, 768 MB for 48 × 16 MB). A split's prompt copies reach it only through `layers::sent_to_card` / `QMatMul::weight_on`. A buffer `to_device`'d to ANOTHER card device is ordered on both streams in the vendored `transfer_to_device` (gotcha #828). ⛔ Moving off legacy SHIPPED BROKEN in v0.3.199 — reproduce under `--features cuda` on a real generation. Guard: `the_vendored_attention_kernels_launch_on_the_devices_stream`.
- **Fused kernels** (`kernels/*.cu`) are bit-identical to the candle ops they replace (except `decode_attn.cu` and `delta_rule.cu`, judged against llama.cpp); a layer-end residual is `Residual::Pending` → `Residual::add_norm`.

→ technique in `docs/DIAGNOSTICS.md` § "Where a decode token actually goes"

→ `docs/invariants/inference.md` § "A decode token is bound by GPU submission COUNT, not bandwidth"

## A prompt pass on the card multiplies quantized weights on the tensor cores (2026-09-29)

Vendored `quantized/cuda.rs::dequantize_matmul` sends ≥ 64 activation rows to `mul_mat_via_f16_cublas` (tensor cores) instead of MMQ; decode never reaches it. `prefill_pacer::CARD_CHUNK_TOKENS` (512) via `prompt_chunk_ceiling`, the ONE answer. Replies may move by a near-tie — judge against llama.cpp, never byte-equality. A/B: `SWARMLLM_QMATMUL_CUBLAS=0`. **It accumulates in f16 by default since 2026-10-09** (#147, judged at 4K tokens on eight families), except the weights llama.cpp marks `GGML_PREC_F32`: **`ModelArch::accumulates_outputs_in_f32`** (GLM-4's attention output + FFN down), marked ONCE after load by `layers::accumulate_output_projections_in_f32`. A new arch's marks are read off llama.cpp. `SWARMLLM_QMATMUL_CUBLAS_ACC=32` = all f32; `SWARMLLM_F32_MARKS=0` = no marks.

→ `docs/invariants/inference.md` § "A prompt pass on the card multiplies quantized weights on the tensor cores"

## A decode step goes to the card as CUDA graphs — ON by default since 2026-09-30

`inference::cuda_graph` + `SplitModel::forward_decode_as_graph`: ON by default, `SWARMLLM_CUDA_GRAPH=0` off (the legacy stream, which cannot be captured, goes with it). Recorded in groups (`cuda_graph::group_layers`, `Cutter::cut`). ⛔ A pageable host→device copy inside a capture is replayed from the host address — silent garbage: `htod_copies_so_far` discards such a capture; a new `clone_htod` on the decode path turns capture off. Capture only where `KvCacheStore::every_cache_holds`; catch the mirror up BEFORE recording (`LayerKv::catch_up_mirror`, #761). A recurrent layer holds only with its NEXT buffers: on a card `SsmState` keeps two pairs, a step writes the pair it did not read, and the swap is host bookkeeping a refused capture undoes (`SsmState::rewind_to`, one step; #228 (4)). A MoE layer is captured where its experts are STACKED on the card — `executor::ffn_capture_ineligibility` is the per-layer answer (#138 (2)); MLA and the per-expert card path stay out. **A graph pays only when it is UPDATED**: a shape the driver keeps refusing to update rests (`DecodeGraph::declines`, `REBUILDS_BEFORE_RESTING`, `REST_STEPS`) — a rebuild costs 10-100 ms against the ~3 ms a graph saves (2026-10-05) — **but only where the card's timeline says the rest pays**: a churning shape's steps are timed (`DecodeGraph::begin_step` → a REQUIRED `StepBegun` for `capture` / `note_uncaptured`) and `Churn::weighed` declines or ends a rest a graph still wins (a 0.5B drafter: ~6 ms rebuild vs ~9 ms saved a step, #233; A/B `SWARMLLM_CUDA_GRAPH_REST=count`). **A step's ids are ONE type before the capture** (u32, `cast_ids_before_capture`): a cast inside changes the graph's shape when a conversation is fed both (#233; A/B `SWARMLLM_CUDA_GRAPH_ID_CAST=inside`). A TOPOLOGY refusal names no node — the debug census (`node_census`) of the group before and after does. ⛔ **No memset (`alloc_zeros`, `Tensor::zeros`) on memory a captured step allocates**: the driver can never update that graph — flash-attn's zeroed `softmax_lse` rebuilt every speculative check on a card (#221). ⛔ **No driver call from the capturing thread** besides the captured ops — a pool query from the profile's report invalidated every capture (`cuda_pool::current_pool` answers `None` while capturing, gotcha #841).

→ `docs/invariants/inference.md` § "A decode step can go to the card as one CUDA graph", § "A graph pays only when it is updated" and § "A rest is weighed on the card's timeline, and a step's ids are one type", § "Qwen 3.5's decode steps are captured too", § "A mixture-of-experts layer stacked on the card is captured too"

## A head width reaches the card only after the flash kernel is checked at it (2026-10-16)

The vendored flash kernel's 512-wide block on sm86 attended WRONGLY at every shape (32 rows on 4 warps; Gemma 4's global layers, gotcha #868). **`examples/flash_reference_check.rs`** checks every head width a family uses against f32 attention — run it before claiming a new width on a card. `kernel_traits.h` refuses a block shorter than its warps (guard `every_vendored_flash_block_is_a_whole_number_of_warps_tall`). Both kernel build scripts `.watch()` every header: cudaforge rebuilt an object only when its `.cu` changed, so a header patch never compiled.

→ `docs/invariants/inference.md` § "The flash kernel is checked at every head width it is handed"

## A decoded token's attention on a card is ONE kernel, and never keeps the mirror in step (2026-09-29)

**`decode_attn::gqa_decode_attention_cuda`** answers a one-position attention on a card (A/B `SWARMLLM_DECODE_ATTN=standard`). A one-position append leaves the f16 mirror behind (`LayerKv::append`); `flash_operands` declines a lagging mirror (`SWARMLLM_KV_MIRROR_EAGER=1` = old). ⚠ Not bit-identical to the matmul path — judge replies against llama.cpp.

→ `docs/invariants/inference.md` § "A decoded token's attention on a card is one kernel"

## A card keeps its KV cache as f16 — `layers::kv_storage` is the one answer (2026-10-09)

**`layers::kv_storage`** says how a layer keeps its cache (`KvStorage::{F32, F32Mirrored, F16}`; its `KvAttention` argument is REQUIRED — MLA is f32 everywhere, and two estimates priced it as the half cache when the rule was asked without it) for the cache as it is built AND every estimate of its cost: the loader's budget, the split planner and admission (all through `kv_budget::KvLayout`, per layer), a peer (`kv_storage_as_advertised` + `features::KV_HALF_ON_CARD`). **F16 alone** on a card where both its kernels read it in place — flash (a transposed BHSD view, no copy) and the decode kernel (`decode_attn::card_kernel_covers`); elsewhere the old layout. Every forward writes through **`layers::append_to_cache`** (`LayerKv::conform_to` corrects a hydrated cache); f32 → f16 is ONE launch (`kv_write::write_into`). A half cache reaching the matmul path is widened per call. A/B `SWARMLLM_KV_F16=0`, `SWARMLLM_KV_WRITE=compose`.

→ `docs/invariants/inference.md` § "A card keeps its KV cache as f16"

## A speculative check is captured too, and a layout is uploaded once (2026-09-30)

A forward of up to `cuda_graph::MAX_POSITIONS` (8) positions is captured when no causal mask is read (`SplitModel::mask_is_read`). `CudaDevice::layout_params` keeps a layout on the device and **never caches one while the stream is capturing**. Graph state and give-ups are per position count; the drafter reads catch-up via `draft_after`, and the γ choice reads `RecentMedian`.

→ `docs/invariants/inference.md` § "A speculative check is captured too, and a layout is uploaded once"

## Attention kernel choice and the query-length cliff (2026-08-23)

Every CPU decode fast path was once gated on `q_len == 1` exactly, so a 2-token
forward cost 7.8x a 1-token one (#369). Four helpers own those decisions now —
never re-derive one from `q_len`:

- **`layers::flash_handles_offset_causal`** — flash takes a query block on a warm
  prefix (the vendored kernel is bottom-right causal; #368). A/B `SWARMLLM_FLASH_OFFSET_CAUSAL=0`.
- **`layers::cuda_decode_prefers_standard`** — `q_len == 1` for every head geometry.
- **`layers::grouping_applies` + `grouped_gqa_attention`** — read K/V at stored width
  for any query length that fits one pass; the mask is TILED (row `r * q_len + t` sees row `t`).
- **`cpu_pools::DECODE_SHAPED_MAX_TOKENS`** — the pool choice turns on bandwidth-bound
  matmuls, not `seq_len == 1`; a short block never feeds the decode-width calibration.

**A GQA call too big for one pass is blocked over query POSITIONS and grouped
per block** (`grouped_blocking_applies`, `mask_rows`) — it never expands K/V
with `repeat_kv`. Expanding doubled attention for every prompt chunk past a
5,461-long cache on a 24-head model (+20% on a 6.9K prompt once removed).
Test `a_blocked_gqa_call_is_grouped_not_expanded` counts expansions.

→ `docs/invariants/inference.md` § "Attention kernel choice and the query-length cliff"

## Several query positions on the processor never write the score matrix (2026-09-26)

**`inference::prefill_attn::gqa_prefill_attention_cpu`** answers CPU attention with `q_len >= 2` before any matmul path: key-tiled in fixed 1,024-key chunks, the `[rows, kv_len]` matrix never written. The matmul paths are **`layers::matmul_attention`** — a test of blocking, grouping or in-place V must call it, not `standard_attention`. Sizes stay constants. A/B: `SWARMLLM_PREFILL_ATTN=standard`.

→ `docs/invariants/inference.md` § "Several query positions on the processor: tiled over the keys, the score matrix never written"

## Local speculative decoding — `inference::model_worker::ngram_spec_eligible`

The single answer to "may this request be speculated?", consulted by BOTH the
slot-admission gate and the decode loop. Two copies would eventually disagree,
and the failure is silent: the gate diverts a request off the batched path and
the loop then declines to speculate it, so it loses batching and gains nothing.

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

**`inference::layers::rope_over_heads`** is the single partial-RoPE implementation; its result is contiguous (the pass-through half is made contiguous BEFORE `cat`), because `slice_set` refuses a non-contiguous source. `SeqCache::append` makes its source contiguous too. Never write a second copy.

→ `docs/invariants/inference.md` § "`inference::layers::rope_over_heads` — partial RoPE has one implementation, and it answers with a tensor the KV cache can write"

## A GGUF tensor the loader ignores is a feature silently missing (2026-09-27)

`rope_freqs.weight` was ignored (#124); now applied as llama.cpp does. **Shard-0 tensors a later-layer node needs are ONE list** — `GgufTensorMeta::sidecar_tensors` — walked by `extract_sidecar_tensors`, `download_sidecar_tensors`, `resolve_sidecars`; add to it, never a new producer. Diff a new architecture's tensor names against what the loader reads.

→ `docs/invariants/inference.md` § "A model's RoPE frequency factors are applied — and reach a node without shard 0"

## A model whose state cannot be wound back is never speculated (2026-10-06)

Qwen 3.5's DeltaNet layers keep a recurrent state per request, and a rejected draft would leave its tokens in it. **`pipeline::distributed::speculation_can_roll_back`** is asked once ahead of every coordinator speculative path; the worker's own n-gram (`ngram_spec_eligible`, both call sites) and SWIFT ask **`SplitModel::carries_recurrent_state`**; `KvCacheEntry::truncate_to` refuses past the one step a card's state keeps (`SsmState::can_rewind_to`), so a new path that truncates further fails loudly. A new speculative path asks one of the two.

→ `docs/invariants/inference.md` § "A model whose state cannot be wound back is never speculated"

## Qwen 3.5's delta rule is ONE op per layer (2026-10-11)

`layers::qwen35::forward_deltanet` runs a layer's whole recurrence in one op: **`delta_rule_cpu`** (a task per sequence and value head, the state updated in place) and, on a card, **`delta_rule_cuda`** → `kernels/delta_rule.cu` (llama.cpp's `gated_delta_net_cuda`; scope `card_delta_rule_covers`). Value head h reads key head h % nk; q and k are normalised as one tensor. NOT bit-identical to the composition `delta_rule`, which stays as the A/B (`SWARMLLM_DELTA_RULE=steps`) and the reference of `one_pass_delta_rule_matches_token_by_token` — judge replies against llama.cpp. Profile stage `delta rule`.

→ `docs/invariants/inference.md` § "Qwen 3.5's delta rule is ONE op per layer"

## A sliding window is cut where it binds, and its cache keeps the window (2026-10-15/16, #278, #281)

**`split::SlidingWindow::read`** is the one reading of a model's window (llama.cpp's `LLAMA_SWA_TYPE_STANDARD`, per family — Gemma 2: even layers, 4096; Phi-3's is off in llama.cpp, so off here). **`LayerWeights::keys_in_window`** is the one place it is applied, for both attention entry points: K/V (and the f16 mirror) are narrowed to the window's start, taken from the cache's LENGTH less **`LayerKv::held_from`**. Several queries read **`ForwardMasks::for_layer`** (causal + windowed, built once per forward, moved together); flash gets `window_size_left = W - 1`. A window that does not bind is the old path, bit for bit. A/B `SWARMLLM_SLIDING_WINDOW=0`.

**A windowed layer's cache drops what its window no longer reads** (#281, llama.cpp's iSWA cache): **`LayerKv::slide_beyond`** — told the window by every forward (`layers::append_to_cache`, like `conform_to`) — keeps `positions_kept_for_window` (W − 1 + `WINDOW_REWIND_POSITIONS`) and grows no wider than that plus one quantum. `current_seq_len` stays absolute; `held_from` is where the buffer starts. **A rewind past `LayerKv::serves_from` is REFUSED** (`can_truncate_to` in `truncate_to_with`) — a new path that rewinds further fails loudly, never answers from lost keys. A prefix-cache snapshot carries `held_from` / `serves_from` and is found only for a prefix its window answers; **a slid snapshot never leaves the node** (`Entry::manifest` empty, `serialize` refuses: the wire format means "from position 0"). A cache with an f16 mirror (`F32Mirrored`) never slides. **The worker charges a windowed layer as its window** (#281 (2)): **`kv_budget::KvGrowth`** is its one price of n positions (built per layer by the loader; a windowed cache costs at most `most_positions_a_window_holds` — its keep plus the widest forward, so a forward wider than a quantum widens it), and the head-room guard reads the buffers (`KvCacheStore::bytes_a_forward_adds`, which simulates `append`). "Does it slide" is `kv_cache::window_kept` for an estimate, the same `caches_slide` the cache asks. **This node's estimates read the loader's layout** (#281 (3)): `kv_budget::KvLayout` → `KvPrice` in `VramFootprintInputs::kv` (admission, the cost curve, the card/processor split), each reader asking `KvPrice::reading` with its worker's prompt reader. **A coordinator's bound on a PEER reads it too** (#281 (4)): `scheduler::prompt_kv_bytes_per_layer` → `KvLayout::growth_on_peer`, a windowed layer at its window ONLY where the peer advertises `features::KV_WINDOW_SLIDES` (bit 22, not in `ALL`; `kv_budget::windows_advertised`: windows attended, caches slide, prompt chunk 1..=`COPIED_CHUNK_MAX_TOKENS` — the width it is then charged at). The prefix cache's byte ceiling charges a slid layer the positions it holds (`prefix_cache::SnapshotCost`, #281 (5)). A/B `SWARMLLM_KV_WINDOW_SLIDE=0`.

**A family's attention scale is read, never assumed `1/sqrt(head_dim)`** (#279): **`split::query_scale`** (Gemma 2's 27B = 46 layers: `1/sqrt(n_embd/n_head)`; the 9B is NOT, though its ratio also differs) → `LayerWeights::query_scale`, multiplied into the queries on both entry points. A new family's `f_attention_scale` is read off its llama.cpp `load_arch_hparams`. A/B `SWARMLLM_QUERY_SCALE=0`.

**A layer's rotation is read per layer** (Gemma 3, #115 (1)): **`split::rope::LayerRope::read`** is the one reading — `rope.scaling` on every position, linear or YaRN (`RopeScaling`, any family; YaRN's ramp + lift, #280), and windowed layers' own base/scale (Gemma 3: base 10,000, unscaled). **`LayerPlacement::rope_for`** is the one place a layer gets its tables, so no per-arch loop can hand a windowed layer the global ones. Which layers are windowed comes from `SlidingWindow::from_file`, never the A/B-switched `read`. A/B `SWARMLLM_ROPE_SCALING=0`, `SWARMLLM_LAYER_ROPE=0`.

**A layer's attention SHAPE is read per layer** (Gemma 4, #115 (3)): **`split::attention_shapes`** is the one reading (`head_count_kv` per layer; `key_length_swa` / `rope.dimension_count_swa` on windowed layers); a model-wide figure (`GgufTensorMeta`'s, a peer's bound) is the WIDEST layer, never layer 0's; this node's estimates price each layer (`kv_budget::KvLayout`). Gemma 4: V = K where `attn_v` is absent (`value_is_key`), V RMS-normed with no weight (`LayerWeights::v_rms_norm`, one site), `layer_output_scale` (`executor::scaled_output`), queries × `sqrt(head_dim)` (`query_scale`), windowed layers turn their own width without `rope_freqs` (`rope::WindowedRope`). Experts with a gate/up scale are refused by name (`unbuilt_gemma4_part`). Reference: llama.cpp MASTER (`~/llama.cpp-ref/dump_logits`).

**The E2B/E4B's shared layers attend another layer's cache** (#115 (3c)): **`split::kv_sources`** is the one reading (the last earlier kept layer of the same kind — HF's rule; llama.cpp hard-codes it, gotcha #884), **`LayerWeights::kv_slot`** the one answer to which slot a layer's attention gets on every executor path; a shared layer computes no K/V and keeps no cache (`split::kv_layers` charges none), and SWIFT never skips a layer that lends one; a card/processor split can put a shared layer and its source on different devices, so the history goes through `layers::history_beside_queries` at both attention entry points. **Every layer's per-layer input is built once per forward from the TOKENS** (`split::per_layer::PerLayerInputs`, the table in host memory — `gguf_meta::host_resident`) and added in ONE place, **`executor::scaled_output`**, which refuses a layer that has an embedding but no input. Only the first segment sees the tokens, so **`GgufTensorMeta::runs_only_whole`** keeps such a model to ONE computer holding every layer — the scheduler (`assemble_pipeline_for`), the loader, the local embedder and tensor parallelism each refuse otherwise; a new path that splits or pre-embeds asks it.

→ `docs/invariants/inference.md` § "A sliding window is cut where it binds", § "A windowed layer's cache keeps its window", § "A windowed layer is charged as its window", § "A family's attention scale is read", § "Gemma 3: each layer turns at its own rate", § "A file's RoPE scaling is read for every family", § "Gemma 4: each layer's attention has a shape of its own", § "Gemma 4's E2B and E4B: shared caches and per-layer inputs"

## A model that cannot write replies is refused off its header (2026-10-15)

**`GgufTensorMeta::refusal`** (→ `ModelRefusal`, one wording via `into_error`) is the one answer: no loader for the family, or an embedding model (`<arch>.pooling_type` non-zero — Ollama's rule). Asked by the loader, the router (`SharedState::refused_model`), auto-manage, the user's download, `/v1/models` and the dashboard list (`cannot_chat`). Before a header exists, `model::huggingface::is_embedding_repo` reads HuggingFace's tags (the pipeline tag first). A new path that runs or fetches a model asks it.

→ `docs/invariants/inference.md` § "A model that cannot write replies is refused off its header"

## A header and its tensor table describe ONE upload — compared before a tensor is read (2026-10-03)

`split::loader::shards::first_header_disagreement` runs in `load_from_shards_inner`, the one place every load from parts passes: every table entry the parts hold must sit at the header's offset with the header's size (`split::tensor_byte_size`, the size tables are cut by), or the load is refused as `SwarmError::MixedModelCopy`. A mismatch never fails on its own — each read lands shifted inside its entry and the model answers garbage (#156). Compare per entry, never by re-deriving the layout (older tables must load). Rig: `split_rig.sh mixed`; null control `every_real_copy_agrees_with_its_own_header`. Parts whose BYTES are not the table's upload (a splice — header and table both agree) are invisible to it: `auto_manage::canonical`'s byte check deletes and re-fetches them (gotcha #782; rig `split_rig.sh spliced`).

→ `docs/invariants/inference.md` § "A header and its tensor table describe one upload"

## A RoPE layout is read off llama.cpp, per architecture

**`ModelArch::use_rope_contiguous` is llama.cpp's `llama_model_rope_type`**
(NEOX = contiguous, NORM = interleaved), arch by arch. GLM-4, Llama-4 and
DeepSeek-2 are NORM and were contiguous here — GLM-4 wrote broken code on every
long reply while conformance passed, and four tests asserted the wrong answer
because they were written against the function (#96). **A new arch gets its row
in `model_arch_properties` from llama.cpp's list, never from this function.**
Byte-identical replies across releases prove no REGRESSION, never correctness.

→ `docs/invariants/inference.md`

## An adapter meets the model in the model's row order (2026-09-25)

**`lora::QkRowOrder`** is the single answer to the q/k row order an adapter meets — a REQUIRED argument of the adapter loader (Llama/Mistral are reordered by llama.cpp's `LlamaModel.permute`). **`lora::check_fits`** refuses what the executor would silently skip. Verify with `examples/peft_lora_to_gguf.py` + `score_against_reference.py --lora`, on an adapter whose B is NOT zero.

→ `docs/invariants/inference.md` § "An adapter meets the model in the model's row order"

## A special token is what the vocabulary says it is, and a prompt gets ONE BOS

**`tokenizer::declared_special`** (CONTROL / USER_DEFINED) decides what is matched whole, on both encoder paths, with llama.cpp's end-of-generation names (`LLAMA_CPP_END_OF_GENERATION_NAMES`, which it overrides to CONTROL); for Gemma 4 nothing is matched by name SHAPE (its 75 NORMAL `<…>` entries), and the text between goes through `BpeTokenizer::encode_gemma4`. **`tokenizer::PieceDecoder::decode`** is the one piece → bytes answer: a `<…>` piece the vocabulary does not declare text is hidden (gotcha #866). **`gguf_meta::add_bos_by_llama_cpp_rules`** decides whether a prompt gets a BOS; **`SplitTokenizer::encode`** adds none to a text already opening with one; **`gguf_meta::special_token_spacing_for`** covers what the GGUF omits. Check with `examples/tokenizer_reference.py` + `tokenizer_agrees_with_llama_cpp`. ⚠ For SentencePiece whitespace llama.cpp is no authority: `examples/tokenizer_hf_reference.py`.

→ `docs/invariants/inference.md` § "A special token is what the vocabulary says it is — and a prompt gets ONE BOS"

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

→ `docs/invariants/inference.md`

## A mixture-of-experts layer keeps its experts QUANTIZED (2026-09-25)

**`split::loader::load_moe_ffn` is the one way a MoE feed-forward is loaded** — experts stay QUANTIZED (`split_expert_stack`); a new MoE family calls it. Routing defaults are per family (`moe_renormalizes_by_default`): read the family's llama.cpp file. Verify with `examples/logits_reference_probe.rs` + `compare_logits_reference.py`; one isolated outlier is a near-tie, a shift everywhere is a bug. Off the card, routing is one host copy (`topk_host`) and ONE `index_add` for the layer. **On a card the experts stay STACKED** (`StackedExperts::fit`, `MoeFfn::experts_stacked`): routed on the card (`topk_on_device`), ONE `moe_gemm_gguf` launch per projection (`layers::moe_card`) — one-row kernel ≤ 8 tokens, tensor-core kernel for a prompt, which needs the vendored candle-kernels barrier (guard `the_vendored_moe_prompt_kernel_keeps_its_barrier`); the kernel reads STORAGE pointers, so every operand goes through `own_storage`. **`moe_card::kernel_type` is the one list of types the kernels read** — Q8_0, Q2_K-Q6_K and the vendored 32-block Q4_0-Q5_1 (#292: llama.cpp stores a row too narrow for a K-quant in one, every 26B-A4B's down experts); an id a kernel does not launch is uninitialised output, guard `every_type_the_card_moe_kernels_are_handed_is_launched`. A/B `SWARMLLM_MOE_CARD=per-expert`, `SWARMLLM_MOE_CARD_PROMPT=rows`. ⚠ Run each arm TWICE — two replies from ONE arm is a race, not numerics (gotcha #827). **On the processor, any token count runs every chosen expert in one parallel region per projection** (`MoeFfn::experts_in_one_region` → vendored `quantized::matmul_many`, llama.cpp's `mul_mat_id` shape: a task is 64 rows of one expert's weight × all of that expert's tokens; `MOE_PAIRS_PER_REGION` pairs a region) — bit-identical to `experts_one_at_a_time`, pinned by `one_region_experts_are_bit_identical_to_one_at_a_time` + `many_row_products_in_one_region_match_forward`; A/B `SWARMLLM_MOE_CPU=per-expert`. Time it with the `moe ffn on the card / on the processor` profile stages.

**A layer's feed-forward runs through `FfnVariant::forward(normed, attn_out, lora)`** — every executor site, never a `match` on the variant (#115 (3b)). Gemma 4's 26B-A4B runs a dense branch BESIDE its experts (`layers::DenseBeside`: each branch normed whole, the router on its own norm of `attn_out` with `1/√n_embd` folded into the weight); `MoeFfn::forward` refuses such a layer, and tensor parallelism does too. A per-expert output scale (`ExpertScales`, `ffn_down_exps.scale`) multiplies the picked weights on every path; a fused `ffn_gate_up_exps` is cut into gate and up stacks at load (`split_expert_stack_rows`); activation is per layer (`MoeFfn::activation`). A gate/up scale is refused by name. ⚠ Judge a tiny MoE file against llama.cpp's processor AND card builds — at decisive attention they disagree with each other as much as with us; the decisive file is soft attention with every expert used (`N_EXPERT_USED=8`).

**A card/processor split's processor-held experts read a prompt ON THE CARD** — see § "A card/processor split multiplies a prompt on the card".

→ `docs/invariants/inference.md` § "A mixture-of-experts layer keeps its experts quantized", § "Gemma 4's 26B-A4B: a dense feed-forward beside GELU experts"

## A card/processor split multiplies a prompt on the card (2026-10-13, #138 (1)/(1b)/(1c))

llama.cpp's op offload (`ggml_backend_sched` "1.off"): a forward of ≥ `PROMPT_ON_CARD_MIN_TOKENS` (32) copies each processor-held weight to the card for the call. **`layers::QMatMul::forward` / `forward_shared` are the ONE choke point for projections** (`QMatMul::prompt_card` → `on_the_card`; a fused weight copied once; its f32 mark travels with the copy); a MoE layer's chosen experts via `QTensor::stack_some` (`MoeFfn::processor_experts_on_the_card`). Any refusal runs the processor path (WARN once). The loader gives the card ONCE, to weights in host memory only (`layers::let_processor_layers_read_prompts_on`, `prompt_card_for` the one token rule). **`SplitModel::prompt_reader(index_pos)` is the chunk-ceiling question** for every prompt pass (`prefill_pacer::prompt_chunk_ceiling`; the forward's chunk loop sizes EACH chunk as it starts): such a split reads at least 512 a forward (at 128 every weight crosses four times as often) and, from the card's room NOW, up to 4096 (#138 (1e): every forward re-copies every processor weight). Measured, never assumed: each prompt forward records what the pool took (`begin_prompt_measure` / `end_prompt_measure`, `USED_MEM_HIGH`, read BEFORE the profile's report resets it) and the next is predicted from it (`MeasuredPrompt::predicted_bytes`, the causal mask's growth included; the take is measured WITHOUT the cache the forward added, which a reserved request's first chunk adds whole, #292); ⚠ **the pool is grown ONCE before the forward** by that prediction plus the cache's real bytes (`SplitModel::prime_pool_for_prompt` → `cuda_pool::prime`, the claim through `kv_bytes_a_forward_adds`, the one figure the head-room claim reads), and left alone when its free memory covers it — under WSL2 a growth after pageable copies stalls seconds with uptime, and a prompt with its experts in RAM read 24 s a chunk without it (#292, gotcha #883; A/B `SWARMLLM_PROMPT_POOL_PRIME=0`). A stage keeping the stacks for the forward measured no gain once primed — the pool reuses a layer's freed stacks; a card fallback or forced standard attention pins 512. Never `runs_entirely_on_card` (the drafter's and admission timing's). A/B `SWARMLLM_PROMPT_CHUNK_ROOM=0`. The link, not pinned memory, bounds the copy here (pageable 6.5 of 6.7 GB/s). A 14B split 20/28 read 888 tokens in 36.8 s and reads them in 6.5-7.0. A/B `SWARMLLM_PROMPT_CARD_PROJECTIONS=0`, `SWARMLLM_MOE_PROMPT_CARD=0`.

**A processor-held `LayerVariant::Dense` layer runs a ≥ 32-token chunk WHOLE on the card** (1c, llama.cpp's "expand gpu down/up" + pass 5): `LayerOnTheCard::of` copies its norms, biases and RoPE rows over, `card_for` is the gate (a marked attention and feed-forward, no adapter). **The KV cache stays HOME on the processor** (`attend`'s `cache_home`: new rows go down, history comes up) — never give a processor layer a card cache. **`to_device` on a view copies the WHOLE storage**: `force_contiguous()` before uploading a narrowed tensor (cache history, RoPE rows). A refusal cuts the cache back (`truncate(held)`) and reruns the layer on the processor (WARN once); the batched loop keeps (1b). A/B `SWARMLLM_PROMPT_CARD_LAYERS=0`.

→ `docs/invariants/inference.md` § "A split's processor-held experts read a prompt on the card", § "A split's processor-held projections read a prompt on the card", § "A split's processor-held layers read a prompt whole on the card"

## A card/processor split is placed in EVERY per-layer loop, and offered only where the loader makes it (2026-09-25)

**`split::hybrid::LayerPlacement` says where each layer goes; every per-layer loop in `split/loader/` shadows `device`, `cos`, `sin` from it on its first line.** `hybrid::layers_on_device` is the KV budget's one count; `hybrid::arch_supports_hybrid` is an allowlist the pool reads too (`process_pool::split_for_card`). A range's `gpu_layers` / `cpu_expert_layers` come from **`split::placement_of`** — the daemon's placement for that range (`DaemonMsg::PlaceRange`), else the spawn's flags (#265). Guard: `every_layer_loop_in_the_loader_places_its_layer`. Check with `swarmllm test-split --gpu-layers N` + `~/swarmllm-ref/qwen3moe/score_ids.py`.

**A card-held MoE layer may keep its routed experts in RAM** (#138 (3), llama.cpp's `--n-cpu-moe`): `LayerPlacement::experts_device_for` is the one answer and **`load_moe_ffn` takes the PLACEMENT, never a device**; the card budget takes `routed_expert_bytes` off. Such a layer works on the processor every step — ask `SplitModel::cpu_layer_count` / `works_on_the_processor`, never `layer_devices` alone (empty: the segment is unsplit, its caches all on the card). `test-split --cpu-moe-layers N`. The pool chooses it for a MoE model too big for the card — **`process_pool::card_split`** is the one placement decision (fewest experts in RAM when the rest fits, else whole layers), counted by **`split::routed_expert_bytes`**, which the loader uses too; A/B `SWARMLLM_EXPERTS_IN_RAM=0`.

→ `docs/invariants/inference.md` § "A card/processor split is placed in every per-layer loop", § "A card-held layer can keep its experts in RAM"

## One prefix-cache snapshot is sized by bytes, and keeps the opening (2026-09-26)

**`PrefixCache::positions_ceiling`** caps a snapshot BEFORE the copy: the
explicit token ceiling if set, never more than the byte budget over what a
position of this model weighs. A longer prompt keeps its opening — the system
prompt and tools every turn repeats — never nothing. The fixed 8,192 it
replaced refused most agent prompts outright. **A slid windowed layer is charged
only the positions it holds** (`prefix_cache::SnapshotCost`: `(budget +
slid_away) / per_position`, #281 (5)) — a slid snapshot cannot keep an opening
short of `serves_from`, so charging every position kept nothing. **A snapshot
COPIES (`force_contiguous`)**: `contiguous()` hands back the live buffer for a
one-KV-head view, which a window's in-place slide then rewrites (gotcha #874). **A kept request's snapshot
never leaves the node** (`Entry::kept_here`: an empty manifest, so never
announced or exported; an entry covering a kept one inherits it; the keep is a
REQUIRED `KeepAsked` argument of `insert_from_kv`, #251 (2b)).

→ `docs/invariants/inference.md` § "One prefix-cache snapshot is sized by bytes"

## A recurrent model's prompt is saved short of its end, and restored whole (2026-10-18, #228 (8))

**`PrefixCache::save_point`** is the one answer to where a prompt pass saves: the end, or `RECURRENT_SAVE_BEFORE_END` (16) short of it for a model with recurrent layers (llama.cpp's context checkpoints) — the next turn does not repeat a reply's opening. Both generate paths stop there (`Slot::save_at` cuts the chunk; `handle_generate` forwards in two) and save through **`model_worker::save_prompt_opening`**. The snapshot carries a COPY of each state (`KvSnapshot::ssm`), taken only at the attention layers' position, copied again on restore (a card's step writes into the pair it read). Its `serves_from` is its whole length: never narrowed, never announced, exported or serialized, and a recurrent model never probes a peer.

→ `docs/invariants/inference.md` § "Qwen 3.5's prompt is saved short of its end, and restored whole"

## Single-source-of-truth helpers — Inference kernels, caches and the tokenizer

Each names the ONE place a decision is made (`.claude/rules/architecture.md` § "One invariant, N paths"). **Read the topic file before changing one.**

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`. Several-row AVX2 kernels for Q4_K, Q6_K, Q2_K, Q3_K, bit-identical to the per-row loop; `k_quants::vec_dot_rows_chosen` is the one call (A/B `SWARMLLM_QMATMUL_ROWS=per-row`). A 2-3 row tail — a speculative check's whole call — goes through ONE pass (`avx.rs::tail_in_one_pass`; A/B `SWARMLLM_QMATMUL_TAIL=per-row`). → § "A 2-3 row tail shares one pass too"
- **`inference::decode_attn::gqa_decode_attention_cpu`** — each K/V row read ONCE per group. → § "It read the cache once per QUERY head"
- **`inference::prefill_attn::gqa_prefill_attention_cpu`** — `q_len >= 2`.
- **`inference::fast_math`** — `exp_inplace`, `silu_mul`.
- **Vendored `rayon-core` (`vendor/rayon-core/src/sleep/mod.rs::spin_budget`)** — every pool's idle workers spin 200 µs before sleeping, so a token's small regions pay no wake per worker (8-thread decode 172 → 123-127 ms); A/B `SWARMLLM_RAYON_SPIN_US=0`; guard `rayon_core_is_the_vendored_copy_that_spins_before_sleeping`. → § "Idle rayon workers spin 200 µs before they sleep"
- **`inference::cpu_pools::in_phase_pool`** — one choke point. Decode runs INSIDE a pool even at the global pool's width (`pool_for`); from the caller every parallel op is a cross-thread handoff (#137, 3B 111 → 82 ms/token). A bench reproduces a node only at the node's `RAYON_NUM_THREADS`.
- **`inference::layers::new_kv_cache`** — the only way to construct a KV cache.
- **`inference::split::kv_cache::LayerKv`** — one layer's KV cache.
- **`SeqCache` / `KvPair` + `LayerKv::truncate`** — **never re-introduce a copy on the rollback path.**
- **`inference::attn_softmax::scaled_masked_softmax`**
- **`layers::standard_attention` grouped GQA decode** — no `repeat_kv` for `q_len == 1`.
- **`layers::cuda_decode_prefers_standard`** — `SWARMLLM_GQA_DECODE_FLASH=1` = old rule.
- **`inference::sampling::sample_among_top_k`** — never re-add a vocabulary-wide pass after top-k. → § "Top-k shrinks the candidate set"
- **`inference::mem_bandwidth::measured_gbps`**
- **`inference::cancel::unless_cancelled`** — `InferenceRequest::cancel` is the ONE cancellation signal; read around every WAIT.
- **`KvCacheStore::set_cancel_oracle`** — `forward_was_cancelled` is the one reader.
- **`split::token_embedding::rows_on_demand_eligible`**
- **`inference::split::read_gguf_header`** — the single way to parse a GGUF header off a PATH.
- **`SplitModel::logits_from`** — the output head, the ONE place a forward makes logits (single, batched, captured): Gemma 2's soft cap and the file's `suppress_tokens` (`SUPPRESSED_LOGIT`, finite: a peer's check rows must be, #282). → § "The output head is one place"
- **`GgufTensorMeta::tied_output_location`**

→ `docs/invariants/inference.md` § "Single-source-of-truth helpers — Inference kernels, caches and the tokenizer"
