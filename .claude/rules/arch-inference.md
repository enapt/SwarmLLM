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
---

# Inference kernels, caches and the tokenizer

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## A context that will not fit is shrunk, not refused

`inference::executor::context_retry_ladder` is the answer to "what context size
will this card actually accept": halve from the capped figure to a floor, serve
the first size accepted, and log what was granted. `effective_llama_context`
caps by a CONSTANT, and whether that constant fits is not constant — an 8B's
weights can leave less room than its 8192-token KV cache needs, which failed
every request on a 6 GB card.

llama.cpp will not say how much it needs and free memory read beforehand is
evidence rather than proof, so the size is asked for rather than predicted.
**Called only from `llama`-gated code, so every default build reports it dead**
(gotcha #264).

→ `docs/FUTURE_WORK.md` § "A context that does not fit is refused instead of shrunk"

## Cross-feature compile checks

`cargo check` with default features does NOT compile any `cfg`-gated
path. Nothing local sees them: `cargo fmt`, `cargo clippy --all-targets`,
the whole test suite and the pre-push hook are all default-features, and
so is the per-push CI run. **The only signal for the GPU paths is the
cache-warm workflow**, which does not run on every push.

Two gates matter here:

- **`llama`** — `pipeline/dsd.rs` and the spec/llama-gated code in
  `pipeline/speculative.rs`. Verify with `cargo check --features llama`,
  which is cheap. R91 caught a regression R90 had let through.
- **`flash-attn` / `cuda`** — the CUDA arm of
  `inference::layers::run_attention` and anything else under
  `#[cfg(feature = "flash-attn")]`. `cargo check --features flash-attn`
  works locally when `nvcc` is present (set `CUDA_COMPUTE_CAP=80` to match
  the release build) but compiles the kernels, so budget tens of minutes.

  **It must be `--all-targets`, and that is not a detail.** CI runs
  `cargo check --locked --features flash-attn --all-targets`; a plain
  `cargo build --features cuda` does NOT compile test code, so a gated
  `#[cfg(feature = "flash-attn")]` **test** is invisible to it. That is
  precisely how main went red on 2026-08-10: `run_attention` gained a
  parameter, every production caller was updated, and one caller inside a
  flash-attn-gated benchmark test was not — through a release `--features
  cuda` build, a default `--all-targets` clippy, 1819 passing tests and a
  green pre-push hook. **Changing the signature of anything callable from
  gated code means running the gated check with `--all-targets` before
  pushing.** Grep for the symbol first: `grep -rn "the_fn(" src/` shows the
  gated callers that no default build will compile.

  A debug-profile `cargo check --features flash-attn` rebuilds the kernels
  (tens of minutes) even though the release profile may already have them.
  Adding `--release` reuses them and is much faster, but then the
  **integration-test targets fail spuriously**: `Database::open_temp` is
  `#[cfg(any(test, debug_assertions))]`, and `--release` turns
  `debug_assertions` off, so `tests/integration/*` stop compiling with a
  wall of `no associated function named open_temp`. That is the profile,
  not a regression. Read which TARGET failed — `lib test` is the one that
  carries the gated unit tests and the one CI reports.

**The specific trap, which has now fired (gotcha #264): an import used only
inside a `cfg`-gated arm is reported UNUSED by every local build.** Acting on
that advice — which clippy gives confidently, and which is correct for the
configuration being compiled — deletes a symbol the GPU build needs, and
nothing local goes red. `DType` in `layers/mod.rs` is annotated
`#[cfg_attr(not(feature = "flash-attn"), allow(unused_imports))]` for exactly
this reason.

So: **before removing anything an unused-warning points at, grep the file for
`#[cfg(`.** If the file has gated arms, the warning is only telling you about
one configuration. And after pushing a change that touches gated code, check
the cache-warm run rather than assuming a green CI means the GPU builds work —
`gh run list --workflow="Cache warm"`.

### The CUTLASS kernels live OUTSIDE `target/` (2026-08-17)

`candle-flash-attn`'s 19 kernels are built into `.flash-attn-build` (via
`CANDLE_FLASH_ATTN_BUILD_DIR`, set by `.github/actions/gpu-build-env`) and cached
separately from the Rust build cache.

**Why, and the trap to remember**: `Swatinem/rust-cache` deletes everything in
`target/` belonging to a package whose manifest is inside the repo — which every
crate under `vendor/` is. So both GPU jobs restored a cache reporting
`full match: true` and then recompiled all 19 kernels anyway, ~39 min of the
Windows GPU build and ~27 of the Linux one, on every release, for months. Nothing
ever went red; the only symptom was 39 minutes of silence in the log between the
last `Compiling` line and the build script's output. **"Cache hit" is not "the
slow thing was cached" — read the compile lines, not the restore line**
(gotcha #318).

It works because `cudaforge`'s own `BuildCache` skips up-to-date kernels by
CONTENT HASH rather than mtime, so a directory restored from a tarball is
accepted. A warm run logs `All kernels up-to-date, skipping compilation`; that
line, plus `Cache restored from key: flash-attn-kernels-*`, is how you confirm
the mechanism fired rather than inferring it from a faster wall clock.

Two things a change here must preserve, both learned the hard way:

- **Create the directory.** Upstream panics `Directory doesn't exists` unless the
  override path already exists — i.e. on the very first run after introducing it.
- **An empty env var is not an unset one.** `std::env::var` returns `Ok("")` for a
  variable that is set but empty, and a matrix expression like
  `${{ matrix.x && '…' || '' }}` yields exactly that for every cell that does not
  want the override. The vendored build script filters empty explicitly.

The CI `flash-attn` compile-check cell points the build script at a
non-existent temp dir with `CANDLE_FLASH_ATTN_CHECK_ONLY=1`, which exercises this
patch in ~50 s on every push — nothing else in CI compiles that crate, because
compiling it is the cost being avoided.

## A decode token is bound by GPU submission COUNT, not bandwidth (2026-09-22)

Decode on the GPU spends most of a token in the CUDA driver API on ONE CPU
thread — measured 1,085 submissions and 17.6 of 23.0 ms/token, with the card at
52% and `ms/layer` flat (0.48-0.58) across a 3.3x span of bytes/token. **So
layer count predicts decode cost, not model size**, and the lever is fewer
submissions per layer, not faster arithmetic.

- **`CudaDevice::alloc_fully_overwritten`** is the only way to allocate a buffer
  the next kernel fills completely; it skips the `cuMemsetD8Async` that
  `alloc_zeros` submits. **Read the kernel first** — it must ASSIGN every
  element it owns. Load-time padded buffers must stay zeroed.
- **`QMatMul::forward_shared` is how several projections consume ONE
  activation** — Q/K/V, and the FFN's gate/up. It quantizes the activation once
  instead of per matmul, and for a `FusedSlice` model (Phi-3/3.5/4) runs the
  fused matmul once instead of once per slice: **+55% on phi-3.5-mini**.
  ⚠ Apply LoRA AFTER it returns; LoRA's matmuls do not share that activation.
  ⚠ **TWO quantized matmul paths must both share** — `mul_mat_vec_via_q8_1`
  (decode) and `mul_mat_via_q8_1` (MMQ: prefill, batch). The first fix did only
  the vec one and nothing went red; the per-kernel count is what showed prefill
  still at 7.05 quantizations per layer.
- ⚠ **TTFT cannot measure prefill work**: a repeated prompt is served from the
  PREFIX CACHE and reads ~0.02 s regardless. Use `PROF seq_len=N` with unique
  prompts, filtered to ONE chunk size.
- **`SWARMLLM_COUNT_KERNELS=1` prints launches by kernel name per forward**,
  which is how the per-layer mix is known at all (nsys has no GPU kernel table
  on WSL2). Never benchmark with it on — it takes a mutex per launch.
- **Judge such a change by the submission COUNT, not the clock** — the count is
  deterministic, this box spreads 10-18%. `SWARMLLM_ZERO_QMATMUL_BUFFERS=1`
  restores the old behaviour for a one-binary A/B.
- **"Per-layer dispatch" is the standing first suspect for a decode number that
  will not move** — the CPU backend reached the same conclusion independently.

→ `docs/invariants/inference.md` · technique in `docs/DIAGNOSTICS.md` § "Where a
decode token actually goes"

## Attention kernel choice and the query-length cliff (2026-08-23)

Four helpers now own decisions that used to be spread across call sites. All
four exist because a predicate that *reads* obviously correct was answering a
different question than the one that mattered.

→ `docs/invariants/inference.md`

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

**`GgufTokenizerMeta::end_of_generation_ids_from_vocab`** searches the
vocabulary BY NAME for the tokens that end a reply, and every path that resolves
EOS ids merges it in — `eos_tokens_with_arch_fallback` and
`split::entry::SplitModelEntry::from_header`. A declared EOS is trusted but
never assumed COMPLETE: the per-family id lists only ever ran when a GGUF
declared nothing, so a model that declares one token and ends its turns with
another got no help from them at all.

Phi-3/3.5/4 are that model. They declare `<|endoftext|>` and close every turn
with `<|end|>`, and nothing stopped the reply there — it ran to `max_tokens`
inventing further turns, visibly on a GPT-2-BPE vocabulary and invisibly on a
SentencePiece one.

**`<|end|>` is conditional, and the condition is the whole reason to read
upstream first.** For harmony (gpt-oss) and solar-open it separates messages
inside one reply, so stopping on it truncates every such reply at its first
message. Both the EOS search and `chat_template::extract_stop_strings` carry the
same exclusion, keyed on the same neighbours llama.cpp keys it on.

→ `docs/invariants/inference.md`

## Partial RoPE has one implementation, and it answers with a tensor the KV cache can write

**`inference::layers::rope_over_heads`** is the single implementation of "rotate
the leading `rope_dim` of each head, pass the rest through". Its result is
contiguous, and the pass-through half is made contiguous BEFORE the `cat`, not
the whole head after it.

`Tensor::cat` answers with a transposed VIEW rather than a fresh buffer when any
argument is non-contiguous and `dim != 0`. `slice_set` refuses a non-contiguous
source and is how the KV cache writes K, so two copies of this branch — one in
`LayerWeights`, one in `Qwen35AttnWeights`, both leaving the pass-through as a
`narrow` view — killed every request on every partial-RoPE model: Phi-4-mini,
GLM-4, Qwen 3.5. The discriminator is `rope_dim < head_dim`, not GQA.

`SeqCache::append` makes its source contiguous too, so a new producer of K or V
cannot bring the class back.

→ `docs/invariants/inference.md`

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

→ `docs/invariants/inference.md`

## Single-source-of-truth helpers — Inference kernels, caches and the tokenizer

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/inference.md`

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`.
- **`inference::decode_attn::gqa_decode_attention_cpu`** — single-position attention straight over the KV cache in its stored `[b, kvh, S, d]` layout, one rayon task per (batch, kv head).
- **`inference::fast_math`** — eight-lane AVX2 `expf` (`exp_inplace`, Cephes polynomial, ~2 ulp vs libm, pinned by `vectorised_exp_tracks_libm` over [-80, 80]) and the fused `silu_mul` CustomOp2. A new elementwise pass that calls `f32::exp` in a loop routes through here instead.
- **`inference::cpu_pools::in_phase_pool`** — binds a forward pass to the CPU thread pool that suits its phase, at ONE choke point: `SplitModel::forward_inner_impl` and `forward_batch`.
- **`inference::layers::new_kv_cache`** — the only way to construct a KV cache.
- **`inference::split::kv_cache::LayerKv`** — one layer's KV cache: the f32 BHSD cache every path reads, plus an optional f16 BSHD mirror for the CUDA flash kernel.
- **`inference::split::kv_cache::SeqCache` / `KvPair` + `LayerKv::truncate`** — the KV cache buffer is this project's own, not candle's, for ONE reason: candle's `Cache` keeps its length private, so the only way to keep the first `n` positions was snapshot + `reset()` + `append()` — two full copies per layer on every rejected speculative draft. **Never re-introduce a copy on the rollback path.**
- **`inference::attn_softmax::scaled_masked_softmax`** — the single expression of attention's tail: scale, optional Gemma-2 logit soft-cap, additive mask, softmax.
- **`inference::layers::standard_attention` grouped GQA decode** — (c4cc3b16, 2026-08-16) for `q_len == 1` with `n_kv_head < n_head`, standard attention no longer expands the KV cache with `repeat_kv`; it reshapes the query heads into matmul rows against the UNEXPANDED cache.
- **`inference::layers::cuda_decode_prefers_standard`** — on CUDA, `q_len == 1` takes standard for EVERY head geometry, prefill always flash. The GQA exclusion was retired on 2026-08-23 once `grouped_gqa_decode_attention` deleted the `repeat_kv` cost it existed to route around; `SWARMLLM_GQA_DECODE_FLASH=1` restores the old rule for an A/B inside one binary.
- **`inference::mem_bandwidth::measured_gbps`** — what this machine's memory actually delivers, measured once and cached.
- **`inference::cancel::unless_cancelled` — every wait that can run for minutes watches the request's cancel flag** — `InferenceRequest::cancel` is the ONE cancellation signal — set by `CancelOnDisconnect`, by both SSE surfaces on `sse_tx.closed()`, and by `/cancel`; read around every WAIT, never around a send.
- **A prompt pass asks between layers whether its request was cancelled** — `KvCacheStore::set_cancel_oracle` is probed once per layer by `forward_inner_impl`, which returns `CANCELLED_MID_FORWARD`; `forward_was_cancelled` is the one reader of that message.
- **`inference::split::token_embedding::rows_on_demand_eligible`** — the single answer to "is this model's `token_embd.weight` held quantized with its rows dequantized on lookup, or dequantized whole at load?".
- **`inference::split::read_gguf_header`** — the single way to parse a GGUF header off a PATH, and the buffering is the entire reason it exists.
- **`inference::split::GgufTensorMeta::tied_output_location`** — the single definition of "is this model weight-tied", i.e. does it reuse `token_embd.weight` as the LM head instead of shipping an `output.weight`. Both sidecar writers and the reader go through it.
