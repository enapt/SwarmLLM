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
- ⚠⚠ **ONE CUDA STREAM PER DEVICE** (`SWARMLLM_CUDA_OWN_STREAM`, `=0` legacy); a second needs `SWARMLLM_CUDA_EVENT_TRACKING=1`. ⛔ Moving off legacy SHIPPED BROKEN in v0.3.199 — reproduce under `--features cuda` on a real generation. Guard: `the_vendored_attention_kernels_launch_on_the_devices_stream`.
- **Fused kernels** (`kernels/*.cu`) are bit-identical to the candle ops they replace; a layer-end residual is `Residual::Pending` → `Residual::add_norm`.

→ technique in `docs/DIAGNOSTICS.md` § "Where a decode token actually goes"

→ `docs/invariants/inference.md` § "A decode token is bound by GPU submission COUNT, not bandwidth"

## A prompt pass on the card multiplies quantized weights on the tensor cores (2026-09-29)

Vendored `quantized/cuda.rs::dequantize_matmul` sends ≥ 64 activation rows to `mul_mat_via_f16_cublas` (tensor cores) instead of MMQ; decode never reaches it. `prefill_pacer::CARD_CHUNK_TOKENS` (512) via `prompt_chunk_ceiling`, the ONE answer. Replies may move by a near-tie — judge against llama.cpp, never byte-equality. A/B: `SWARMLLM_QMATMUL_CUBLAS=0`; `SWARMLLM_QMATMUL_CUBLAS_ACC=16` is OPT-IN.

→ `docs/invariants/inference.md` § "A prompt pass on the card multiplies quantized weights on the tensor cores"

## A decode step goes to the card as CUDA graphs — ON by default since 2026-09-30

`inference::cuda_graph` + `SplitModel::forward_decode_as_graph`: ON by default, `SWARMLLM_CUDA_GRAPH=0` off (the legacy stream, which cannot be captured, goes with it). Recorded in groups (`cuda_graph::group_layers`, `Cutter::cut`). ⛔ A pageable host→device copy inside a capture is replayed from the host address — silent garbage: `htod_copies_so_far` discards such a capture; a new `clone_htod` on the decode path turns capture off. Capture only where `KvCacheStore::every_cache_holds`; catch the mirror up BEFORE recording (`LayerKv::catch_up_mirror`, #761). **A graph pays only when it is UPDATED**: a shape the driver keeps refusing to update rests (`DecodeGraph::declines`, `REBUILDS_BEFORE_RESTING`, `REST_STEPS`) — a rebuild costs 10-100 ms against the ~3 ms a graph saves (2026-10-05). ⛔ **No memset (`alloc_zeros`, `Tensor::zeros`) on memory a captured step allocates**: the driver can never update that graph — flash-attn's zeroed `softmax_lse` rebuilt every speculative check on a card (#221).

→ `docs/invariants/inference.md` § "A decode step can go to the card as one CUDA graph" and § "A graph pays only when it is updated"

## A decoded token's attention on a card is ONE kernel, and never keeps the mirror in step (2026-09-29)

**`decode_attn::gqa_decode_attention_cuda`** answers a one-position attention on a card (A/B `SWARMLLM_DECODE_ATTN=standard`). A one-position append leaves the f16 mirror behind (`LayerKv::append`); `flash_operands` declines a lagging mirror (`SWARMLLM_KV_MIRROR_EAGER=1` = old). ⚠ Not bit-identical to the matmul path — judge replies against llama.cpp.

→ `docs/invariants/inference.md` § "A decoded token's attention on a card is one kernel"

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

Qwen 3.5's DeltaNet layers keep a recurrent state per request, and a rejected draft would leave its tokens in it. **`pipeline::distributed::speculation_can_roll_back`** is asked once ahead of every coordinator speculative path; the worker's own n-gram (`ngram_spec_eligible`, both call sites) and SWIFT ask **`SplitModel::carries_recurrent_state`**; `KvCacheEntry::truncate_to` refuses while such state is held, so a new path that truncates fails loudly. A new speculative path asks one of the two.

→ `docs/invariants/inference.md` § "A model whose state cannot be wound back is never speculated"

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

**`tokenizer::declared_special`** (CONTROL / USER_DEFINED) decides what is matched whole, on both encoder paths. **`gguf_meta::add_bos_by_llama_cpp_rules`** decides whether a prompt gets a BOS; **`SplitTokenizer::encode`** adds none to a text already opening with one; **`gguf_meta::special_token_spacing_for`** covers what the GGUF omits. Check with `examples/tokenizer_reference.py` + `tokenizer_agrees_with_llama_cpp`. ⚠ For SentencePiece whitespace llama.cpp is no authority: `examples/tokenizer_hf_reference.py`.

→ `docs/invariants/inference.md` § "A special token is what the vocabulary says it is — and a prompt gets ONE BOS"

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

→ `docs/invariants/inference.md`

## A mixture-of-experts layer keeps its experts QUANTIZED (2026-09-25)

**`split::loader::load_moe_ffn` is the one way a MoE feed-forward is loaded** — experts stay QUANTIZED (`split_expert_stack`); a new MoE family calls it. Routing defaults are per family (`moe_renormalizes_by_default`): read the family's llama.cpp file. Verify with `examples/logits_reference_probe.rs` + `compare_logits_reference.py`; one isolated outlier is a near-tie, a shift everywhere is a bug. Routing is one host copy (`topk_host`) and one `index_add` per expert.

→ `docs/invariants/inference.md` § "A mixture-of-experts layer keeps its experts quantized"

## A card/processor split is placed in EVERY per-layer loop, and offered only where the loader makes it (2026-09-25)

**`split::hybrid::LayerPlacement` says where each layer goes; every per-layer loop in `split/loader/` shadows `device`, `cos`, `sin` from it on its first line.** `hybrid::layers_on_device` is the KV budget's one count; `hybrid::arch_supports_hybrid` is an allowlist the pool reads too (`process_pool::split_for_card`). Guard: `every_layer_loop_in_the_loader_places_its_layer`. Check with `swarmllm test-split --gpu-layers N` + `~/swarmllm-ref/qwen3moe/score_ids.py`.

→ `docs/invariants/inference.md` § "A card/processor split is placed in every per-layer loop"

## One prefix-cache snapshot is sized by bytes, and keeps the opening (2026-09-26)

**`PrefixCache::positions_ceiling`** caps a snapshot BEFORE the copy: the
explicit token ceiling if set, never more than the byte budget over what a
position of this model weighs. A longer prompt keeps its opening — the system
prompt and tools every turn repeats — never nothing. The fixed 8,192 it
replaced refused most agent prompts outright.

→ `docs/invariants/inference.md` § "One prefix-cache snapshot is sized by bytes"

## Single-source-of-truth helpers — Inference kernels, caches and the tokenizer

Each names the ONE place a decision is made (`.claude/rules/architecture.md` § "One invariant, N paths"). **Read the topic file before changing one.**

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`.
- **`inference::decode_attn::gqa_decode_attention_cpu`** — each K/V row read ONCE per group. → § "It read the cache once per QUERY head"
- **`inference::prefill_attn::gqa_prefill_attention_cpu`** — `q_len >= 2`.
- **`inference::fast_math`** — `exp_inplace`, `silu_mul`.
- **`inference::cpu_pools::in_phase_pool`** — one choke point.
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
- **`GgufTensorMeta::tied_output_location`**

→ `docs/invariants/inference.md` § "Single-source-of-truth helpers — Inference kernels, caches and the tokenizer"
