# Inference kernels, caches and the tokenizer

The evidence behind the rules in `.claude/rules/arch-inference.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## Attention kernel choice and the query-length cliff (2026-08-23)

Four helpers now own decisions that used to be spread across call sites. All
four exist because a predicate that *reads* obviously correct was answering a
different question than the one that mattered.

- **`inference::layers::flash_handles_offset_causal`** — may flash attention take
  a query block that lands on a warm prefix (`k_len > q_len && q_len > 1`)? Yes,
  and it is not a judgement call: the vendored kernel's
  `col_idx_limit_right = row_idx + 1 + max_seqlen_k - max_seqlen_q`
  (`vendor/candle-flash-attn/kernels/mask.h`) is bottom-right aligned causal,
  the same predicate `SplitExecutor::causal_mask` builds by hand. The dispatch
  used to divert that shape to `standard_attention` on the stated grounds that
  flash could not express the mask; since `prefill_chunk_tokens` is a CEILING
  that always applies, that meant EVERY prompt chunk after the first took the
  slower kernel on every GQA model (gotcha #368). A benchmark that prefills in
  one call cannot see it. `SWARMLLM_FLASH_OFFSET_CAUSAL=0` restores the old
  behaviour for A/B.

- **`inference::layers::cuda_decode_prefers_standard`** — now `q_len == 1` for
  EVERY head geometry, not just MHA. The GQA exclusion existed because
  `standard_attention` rebuilt the `repeat_kv` expansion every token;
  `grouped_gqa_decode_attention` deleted that work in August and the rule
  outlived its premise by a week. Re-measured: standard wins at every context
  length and the margin grows with it. **The isolated table overstates the
  penalty** — its flash arm runs without the f16 KV mirror that production
  always has — so end to end this is ~6% at 4120 KV and unresolvable at ~900.
  Both numbers are recorded at the benchmark; the forward is the one that
  describes a reply (#266). `SWARMLLM_GQA_DECODE_FLASH=1` restores the old rule.

- **`inference::layers::grouping_applies` + `grouped_gqa_attention`** — read the
  KV cache at its stored width instead of expanding it, for ANY query length
  whose score matrix fits in one pass (not just `q_len == 1`). Blocked calls keep
  the expanded path: the blocking loop slices the query axis, and under grouping
  that axis carries repeats and positions interleaved, so a block boundary would
  cut one query position's rows across two passes. The mask must be TILED to
  match — row `r * q_len + t` sees mask row `t` — and getting that transposed
  computes plausible garbage, which is why the test compares against the expanded
  path with a REAL causal mask rather than a permissive one.

- **`inference::cpu_pools::DECODE_SHAPED_MAX_TOKENS`** — the phase predicate is
  no longer `seq_len == 1`. What the pool choice turns on is whether the matmuls
  are bandwidth-bound, and a 4-token verify re-reads exactly the weights one
  token does. Measured with `examples/qmatmul_bench`, the narrow pool wins at
  every width to 32 and draws level at 128. **A short block must not feed the
  decode-width calibration** — the calibration compares candidate widths by the
  cost of one token, and a sample from a different query length is not that
  measurement (same class of error as #367).

Together these were one defect: every CPU decode fast path was gated on
`q_len == 1` exactly, so a 2-token forward lost the grouping, the single-position
kernel and the narrow pool at once and cost 7.8x a 1-token forward — for one
extra token (gotcha #369).

**Validated on a SECOND machine** (2026-08-23), because #367's lesson is that a
fix measured on one is not measured: an Intel i5-10500T, 6 cores, no GPU — the
same box whose first run found the .114 calibration defect. It holds there and by
more, and the cost of a wide forward is LOWER than on the 8-core Ryzen, which is
what fewer cores predicts (more bandwidth-bound, so extra rows cost less):

```text
  width   old (i5)   new (i5)   break-even accept
      2   373/362     148/149    1.10-1.16 of 2
      4   383/377     171/182    1.47-1.58 of 4
      8   462/453     242/254    2.10-2.20 of 8
```

Against 2.5 of 4 on the Ryzen. Decode was unchanged on the i5 (8.08/8.05 against
7.88/7.75 tok/s, arms overlapping) and prefill gained 1.5-2.9%, consistent in
direction across both pairings.

**The GPU changes above are still single-machine.** They were measured only on
the RTX 3070 here; no second card was available. Treat their magnitudes as
one machine's numbers until a second one confirms them.

## Local speculative decoding — `inference::model_worker::ngram_spec_eligible`

The single answer to "may this request be speculated?", consulted by BOTH the
slot-admission gate and the decode loop. Two copies would eventually disagree,
and the failure is silent: the gate diverts a request off the batched path and
the loop then declines to speculate it, so it loses batching and gains nothing.

Clauses: `!logprobs` (accepted tokens carry none back out), SWIFT off (it is
already speculating), and a non-zero draft width — which is how
`inference.ngram_lookup_enabled` arrives, so the switch cannot disagree with the
shape.

**Temperature is deliberately NOT one of them, and briefly was.** The argument
for excluding sampled requests — that comparing a draft against what the sampler
returned "is a verification only while the sampler is deterministic" — is wrong,
and it left the feature inert for essentially all traffic: the OpenAI surface
defaults to 0.7 and the Anthropic one to 1.0, so Claude Code and MCP tool use,
the workload `ngram_lookup` names as its reason for existing, never speculated.

With a deterministic draft `x` (`q = δ_x`) the speculative-sampling rule accepts
with probability `min(1, p(x)/q(x)) = p(x)` and otherwise draws from
`norm((p − q)₊)` — `p` with `x` removed, renormalised. "Draw `t ~ p`; keep the
draft iff `t == x`" has exactly those two branches, so sampling each position
with the real sampler and keeping a match IS that rule, at any temperature.
`accepting_only_on_a_match_preserves_the_sampled_distribution` pins it, with a
control that fails if the metric could not detect a bias.

Measured at temperature 0.7 on an RTX 3070: a copying reply speculated at 8.83
tokens per round — the same as greedy, because a copied token's distribution is
sharply peaked — and ran 1.79-1.90 s -> 0.58-0.69 s. An open-ended reply at the
same temperature accepted nothing and `SpecBackoff` suppressed 62 of 80 rounds,
which is the correct outcome rather than a failure.

**The DISTRIBUTED n-gram path carried the same gate for the same wrong reason**,
and it was fixed the same way — but note the mechanism differed. It took the
target's raw `argmax` (`greedy_accept_reject`), which ignores temperature, top-k,
top-p AND the repetition penalties, so there the gate was correct for the
implementation. `speculative::sampled_accept_reject` samples every position
through the real sampler instead; at temperature 0 with no penalties it
reproduces the argmax decision exactly (pinned by
`at_temperature_zero_it_agrees_with_the_argmax_it_replaces`), so nothing already
using it changes. It also closed a routing-dependent difference: penalties
always applied on the local worker and never across peers, so the same request
got a different answer depending on where it ran.

**The peer-supplied non-finite guard is not optional and must survive any change
of sampler.** These logits come from another node, and NaN comparisons are
non-deterministic in `argmax`, so a malicious segment could otherwise steer which
drafts are accepted. Both helpers reject the whole round, and a test asserts they
agree on that verdict so the two cannot drift.

**Draft-MODEL paths (`speculative.rs`'s main loop, `dsd.rs`) stay greedy-only on
purpose.** A draft model has a real distribution `q`, so doing this properly
needs `min(1, p/q)` and a residual built from both — a different algorithm, not
this one. The rule here works only because an n-gram draft is a point mass.

Three things a change here must keep:

- **It runs on the sequential loop only, when the worker is otherwise idle AND
  speculation has been paying.** `slot_admission_eligible` diverts a solo
  speculatable request there; one arriving while others decode joins the batch
  instead, because that loop owns the worker for its whole duration and
  diverting would stall everyone in flight.
  **The third condition was added after measuring, and matters as much as the
  other two.** The trade was originally justified with this project's ~3%
  batching figure (#348) — a PROCESSOR measurement. On a graphics card batching
  amortises kernel launches across requests, the same launch-bound property
  speculation exploits, so it is worth far more: 8 concurrent open-ended
  requests took 29.07 s diverted against 12.48 s batched, aggregate throughput
  77 -> 33 tok/s (gotcha #373). `spec_payoff_justifies_diverting` therefore gates
  it on the tokens-per-round speculation has actually been achieving; unknown
  lets one request find out, and the answer steers the rest. With it, both
  workloads beat either fixed policy — open-ended 8-way 11.03 s against 12.48 s
  batched, copy-heavy 8-way 3.37 s against 5.52 s.
- **It is not bit-identical** (gotcha #370). Do not describe it as such.
- **A miss is not free** (gotcha #371): the forward the draft provokes costs
  even when nothing is accepted, which is what `SpecBackoff` exists for. Measure
  the workload speculation CANNOT help, not just the one it can.

## A prompt's length in tokens is a POSITION, not a statistic

**`inference::pipeline::prompt::prompt_positions`** is the single answer to "how
many positions does this prompt occupy?", for all five sites in that module.
Never open-code it, and never estimate it.

`distributed.rs` sets `index_pos = ptc + vision_expand` after the prompt pass,
and ships `index_pos` to the worker as the **rotary position of the next token
and the offset it attends from**. It must equal the number of KV positions the
prefill actually wrote. It is not a report; it is a coordinate.

It was `(prompt.chars().count() / 4).max(1)`, commented *"approximate ... no
tokenizer in-process"* — true when written, and left in place after
`standalone_tokenizer` began lazily building one from `gguf_header.bin` three
lines below it. On a 24 KB tool-calling prompt the estimate came out **6053
against a true 5529**, so the first generated token was computed 524 positions
past the end of the cache. The logits are noise, and the model answers with
end-of-turn or with filler repeated to the token limit (gotcha #400).

Three properties a change here must keep:

- **It only ever bit the pipeline path**, which is the one every request takes
  while the model is not loaded — so the same request failed cold and succeeded
  warm. Anything that makes a placement or path decision differently for a cold
  request inherits this shape: *test it cold*.
- **The error scales with prompt length.** A short prompt misses by a position
  or two and still reads correctly. That is why four releases of `curl`
  reproduction attempts failed — a retry is warm and a minimal repro is short —
  and why the fix's test uses a prompt the old estimate demonstrably got wrong,
  with a control asserting exactly that.
- **The estimate survives only where there is genuinely no tokenizer** (no
  `gguf_header.bin`), and warns when it does. Refusing the request would be
  worse than a reply that may drift, but a node navigating by a guess must say
  so, because nothing else can tell it it is lost.

**The rule this encodes.** A value that is *reported* may be approximate; a
value that is *acted on* may not. `ptc` fed both `usage.prompt_tokens` and
`index_pos`, and the comment justifying the approximation was written for the
first while the second silently depended on it being exact. When one number
serves two purposes it inherits the stricter requirement — so a comment
explaining why an estimate is good enough is a place to go and check what else
reads it.

**And the diagnostic lesson**, which cost more than the fix: an earlier pass
cleared position bookkeeping by observing that `index_pos` "jumps correctly to
the prompt length and then increments by one". That compared the number against
ITSELF. The discriminator is `index_pos` against the worker's `kv_offset` on the
same forward — two values that must be equal, printed in adjacent log lines the
whole time. Ask what a number is supposed to EQUAL, not whether it looks
plausible.

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

**What that cost.** A SentencePiece vocabulary deliberately contains no bare tab
or newline; it carries `<0x09>` / `<0x0A>` and expects byte fallback, which
`spm_encode` has always done. The BPE path did not. So on any GGUF taking that
branch — `tokenizer_model == "llama"` that ALSO ships merges: TinyLlama,
Llama-2, Mistral, Vicuna — **every newline in every prompt was handed to the
model as `<unk>`**, chat-template newlines included, so essentially every
multi-line prompt was structurally corrupted (gotcha #421).

**Byte fallback is gated on `is_sentencepiece` and must stay that way.** A GPT-2
vocabulary maps every byte through `byte_encoder` into a character it does
contain and carries no `<0xNN>` tokens, so a miss there is a genuine vocabulary
problem and falling back would invent tokens. Pinned in both directions by
`a_character_with_no_vocabulary_entry_falls_back_to_its_byte_token` (with a
control: a character having neither a piece nor a byte token must still land on
`<unk>`) and `a_gpt2_vocabulary_does_not_get_byte_fallback`.

**The general rule, and why this one is worth writing down.** Nothing internal
could have found it: the tokenizer round-trips against itself perfectly, and the
equivalence tests written the same day compare it to an *earlier version of
itself* — both were equally wrong. It took HuggingFace `tokenizers` as an
outside reference, and nine minutes. **A component that is only ever checked
against its own past cannot be shown to be correct, only unchanged.**
`examples/tokenizer_scaling` with `SWARM_TOK_TEXT` is the harness; 17/17 samples
now agree on the fixed path and 6/6 on the untouched GPT-2 one.

**Do not add a third site that maps a piece to an id.** Call the helper.

## Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul

(2026-08-21
night) — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`. One weight
column against `rows` activation rows in one call; the AVX2 Q4_K and Q6_K kernels
(`dot_q4k_q8k_rows::<R>`, `dot_q6k_q8k_rows::<R>`) unpack the column once and
share it across R rows. **Overrides MUST stay bit-identical to the per-row loop**
(`vec_dot_rows_generic`): each row keeps its own accumulators and sees the
single-row kernel's operations in the same order; `examples/qmatmul_bench`
asserts exact equality against the upstream ordering for Q4_K and Q6_K at every
m it prices — run it after touching either kernel. **R is a register-pressure
knob, not a "bigger is better" one**: Q4_K at R=8 spilled the 16 ymm registers and
R=4 was 1.2x faster at every m. `matmul` also runs the column-outer loop per
`ROW_BLOCK = 128` rows so the quantized activations stay in L2 — a whole-prompt
forward had streamed ~3 MB from L3 per column, which is why a per-row cost measured
at m=128 never carried to large m. The `examples/prefill_bench` single-forward
number is only representative of the production 128-token chunks because of this.

## `inference::decode_attn::gqa_decode_attention_cpu`

(2026-08-21 night) — single-
position attention straight over the KV cache in its stored `[b, kvh, S, d]` layout,
one rayon task per (batch, kv head). Dispatched at the top of
`standard_attention` for `q_len == 1` on the CPU (MHA and GQA). The two batched
matmuls it replaces cost 1.3 ms/layer at ~920 KV for ~11 MFLOP — GEMM packing and
dispatch, a quarter of every decoded token. **Returns `Ok(None)` for anything
outside its scope** (non-CPU, non-f32, `q_len > 1`, K/V whose `(S, d)` plane is not
dense, a mask it cannot reduce to one row) and the caller carries on unchanged —
it is an accelerator, never a requirement; keep it that way. `SWARMLLM_DECODE_ATTN=
standard` disables it (same discipline as `SWARMLLM_FORCE_STANDARD_ATTN`); that is
how its +24% decode was attributed (A/B/A/B in one binary). Not bit-identical to
the matmul path (different summation order); `decode_kernel_matches_the_matmul_
path` bounds it (abs < 1e-5, rel < 1e-4 with a 0.05 floor — the first metric
flagged fp32 noise on a near-zero output as a failure). The DRAM floor for the
cache read at ~900 KV × 28 layers is ~7 ms/token on this box; the kernel sits at
~15 — the remainder is per-layer dispatch, not arithmetic.

## `inference::fast_math`

⚠ **`silu_mul` now fuses on CUDA as well as on the CPU (2026-09-22), and the two
halves are held to DIFFERENT bars.** The CPU arm uses the AVX2 polynomial `exp`
and is tolerance-tested (2e-6 rel). The CUDA arm is a kernel of ours
(`kernels/fused_decode.cu::silu_mul_f32`, PTX from `build.rs`, loaded through
`CudaDevice::get_or_load_custom_func`) and is held to **bit-identity** with the
`usilu_f32` + `bmul_f32` pair it replaces — same expression, same order, and
`build.rs` passes no `-use_fast_math` precisely so `expf` is the same function
on both sides. `cuda_silu_mul_is_bit_identical_to_the_composed_path` asserts it
per element.

**Why the harder bar on the GPU and not the CPU.** The CPU fusion was a
*throughput* change and its correctness question was "close enough for an
activation". The CUDA fusion is a *submission-count* change worth ~1 launch per
layer — below what this box's clock can resolve — so it is judged by
`examples/kernel_count_ab.sh`, which A/Bs the count inside one binary via
`SWARMLLM_FUSE_SILU_MUL=0`. That A/B only means anything if the two arms compute
the same thing exactly; a tolerance would make "did it get faster" and "is it
still right" the same question. **A future fused kernel here inherits that bar.**

## `inference::residual_norm` — residual add + RMS norm as one kernel (2026-09-23)

The second fused kernel, and the first that needed a TYPE rather than a helper.
One of its two sites straddles the layer boundary — the closing add of layer
*i* is fused into the attention norm of layer *i+1* (or the final norm) — so
fusing it means NOT doing the add where the layer ends, in eight hand-written
copies across `SplitModel`'s single-request and batched loops.
`Residual::Pending { delta, base }` carries the untaken sum and
`Residual::add_norm` is the one place it is resolved; `into_tensor()` takes it
where a plain tensor is needed (device transition, captured layer, non-final
segment output).

**Bit-identity, and why it holds.** `add_rmsnorm_f32` is candle-kernels
0.10.2's `rmsnorm` statement for statement — strided accumulation,
`__shfl_xor_sync` over masks 16..1, the shared-memory second stage above 32
threads, `rsqrtf(mean + eps)`, `(scale * x) * alpha` — launched with candle-nn
0.10.1's geometry (one block per row, 32 threads below 1024 columns, 1024 at or
above). The only difference is that `x = a + b` is computed in a register
instead of loaded from `badd_f32`'s output: one IEEE add either way, and FMA
contraction cannot fuse an add into the multiply that FOLLOWS it, so
`tmp += xi * xi` contracts identically. Both PTX builds use `-O3
-std=c++17` and no fast-math (cudaforge adds no flags of its own).

**Verified**, on the `--features cuda` build:
- `cuda_add_rms_norm_is_bit_identical_to_the_composed_path` — six shapes,
  both geometries, a prefill block and a batch, plus zero-copy (the sum is a
  view at offset N of the one 2N allocation, not a copy).
- **Its null control fired**: replacing the kernel's `tmp += xi * xi` with
  `tmp += __fmul_rn(xi, xi)` (no FMA) fails it on a single ulp — element 9216
  of `[1,128,3072]`, `-0.6935906` vs `-0.69359064`.
- Real generations on tinyllama and llama-3.2-3b are byte-identical to the
  released v0.3.200, fused and unfused.
- Kernel count, llama-3.2-3b decode: 763 → **707 launches/token** (−2.00/layer:
  `add_rmsnorm_f32` 2/layer replaces `badd_f32` 2/layer + `rmsnorm_f32` 2/layer;
  one `rmsnorm_f32` remains for layer 0, whose input is the embedding).

`SWARMLLM_FUSE_ADD_RMSNORM=0` is the off arm. Off CUDA nothing changes: the
pending sum resolves through the old ops in the old order, and
`a_pending_residual_resolves_to_the_composed_ops_on_the_cpu` pins it.

⚠ The profiler's `residual adds` stage is gone — the add is now timed inside
`residual add + rms norm` (`prof.rs`), because on CUDA there is no separate add
left to time.

(2026-08-21 night) — eight-lane AVX2 `expf`
(`exp_inplace`, Cephes polynomial, ~2 ulp vs libm, pinned by
`vectorised_exp_tracks_libm` over [-80, 80]) and the fused `silu_mul` CustomOp2.
Every `exp` on the CPU path had been a scalar libm call — ~540 M in the softmax
rows and ~205 M in SiLU for an 896-token llama-3.2-3b prompt. **Used by the fused
softmax (`attn_softmax::softmax_row`), the three SiLU×up call sites in
`layers/mod.rs`, and the decode kernel.** Not bit-identical; every consumer keeps
its tolerance test against the composed candle reference (softmax 1e-6 rel, silu
2e-6). A new elementwise pass that calls `f32::exp` in a loop is the thing to
route through here instead. Inputs below ~-87.3 underflow to 0 (libm: denormal),
above 88.37 saturate — right for softmax (shifted ≤ 0) and SiLU (limits).

## `inference::cpu_pools::in_phase_pool`

(2026-08-07) — binds a forward pass
to the CPU thread pool that suits its phase, at ONE choke point:
`SplitModel::forward_inner_impl` and `forward_batch`. Every entry point —
LoRA, speculative verify, pre-embedded segment, SWIFT skip-mask, batched
prefill — funnels through those, so a new one inherits it and cannot forget.
Do NOT call `install` at a call site instead.
**Reading a prompt and writing a reply want different thread counts**: decode
is bandwidth-bound (69% of roofline), so past the point that saturates memory
the extra threads only contend. **The cap is PHYSICAL CORES and must not
become a fraction of them.** A fraction was tried — `max(4, physical/2)`,
measured correctly on an 8-core Ryzen — and a second machine (6-core Intel
i5-10500T) showed decode climbing monotonically to all six, where that rule
would have cost 23%. Peak threads is bandwidth divided by per-core draw, which
core count cannot predict; physical-vs-SMT is the only part both machines and
the mechanism agree on. Prefill keeps the global pool untouched; decode is
capped only ever downward, and every contribution level is already at or below
physical, so the common path builds no pool and pays nothing.
`SWARMLLM_DECODE_THREADS` overrides, and `=0` restores the single-pool
behaviour for A/B measurement inside one binary — the same discipline as
`SWARMLLM_FORCE_STANDARD_ATTN`.
**The calibration is keyed by the forward's PROCESSOR DEPTH, and a forward
with no processor layers is never timed** (2026-09-01, gotcha #432).
`in_phase_pool` takes `cpu_layers` — `SplitModel::cpu_layer_count()`, zero for
a segment entirely on the card, the whole segment on a processor-only node,
the processor's share of a hybrid split — and `cpu_pools::Calibrations` keeps
one calibration per depth. Why: one worker serves every forward its model is
asked for, and while the 8B was loading as a 12/32 hybrid split the SAME
process served two one-layer card-only segments of it for a boomerang
request. Their 1-5 ms tokens settled the process-wide calibration on ONE
thread (`4:5ms 3:2ms 2:2ms 1:1ms`), and the full model then decoded its 20
processor layers single-threaded for the worker's life: 2.9 tok/s, below the
4.0 the model does on the processor alone. Re-run with the cold request kept
local it read `4:211ms 3:194ms 2:221ms 1:351ms`, chose 4, and did 5.2-5.6.
Any GPU holder that serves segments for peers can hit this. A new caller
passes the depth of THIS forward, never a property of the worker.

## `inference::layers::new_kv_cache`

(2026-08-07) — the only way to construct
a KV cache. **Never call `KvCache::new(2, max_seq_len)`**, which is what every
site did and which reads as obviously correct — the parameter is even called
`max_seq_len`. candle's `Cache::new(dim, n)` sets `grow_by` AND `max_seq_len`
to `n`, and `append` allocates the full buffer on the FIRST append, so passing
a model's context length reserved the whole context window from token one: a
100-token chat held 940 MB at 3% utilisation on llama-3.2-3b. The helper
passes `KV_CACHE_GROWTH_TOKENS` instead and lets `append` grow on demand; the
conversation's real ceiling is enforced separately by the
`total_seq > max_seq_len` guard in `forward_inner_impl`, so this value cannot
shorten a conversation. `kv_cache_reservation(positions)` is the sibling for a
cache that must hold N tokens immediately — prefix-cache hydration — and it
deliberately ignores the snapshot's recorded `max_seq_len`, because snapshots
cross the network and a peer on an older build recorded a whole-context value.
**Reason about KV memory from `KvCacheStore::occupancy()`, never from process
RSS**: the reservation is lazily-faulted zero pages, so a 4-8x change in
reserved bytes moved RSS ~5% and in both directions. Two conclusions drawn
from RSS about this cache were wrong before the counter existed.

## `inference::split::kv_cache::LayerKv`

(2026-08-10) — one layer's KV cache:
the f32 BHSD cache every path reads, plus an optional f16 BSHD mirror for the
CUDA flash kernel. **Never touch the inner `KvCache` directly.** `append` and
`reset` are INHERENT methods and so take priority over the `Deref`, which is
what stops an existing call site reaching the inner versions and leaving the
mirror behind; `KvCacheStore::truncate_to` (the speculative-decode path) got
correct behaviour for free from that, since it truncates via reset+append.
**Why a mirror rather than replacing the f32 cache**: rounding to f16 moves
from every-read to once-at-write, and since the f32 source is never itself
overwritten the flash kernel receives bitwise the same numbers — so the flash
path is numerically unchanged, not merely close, while `standard_attention`
keeps full precision. Published results on f16 KV divergence (arXiv 2604.15409)
are worst under long context and GQA, which is exactly our case, so the f32
copy stays.
**Three things a new caller must respect.** (1) The mirror is GQA-only —
`layers::model_wants_kv_mirror` gates it, because MHA decode reads the f32
cache and an unread mirror cost 3-8% per token plus 50% more KV memory
(measured on phi-3.5). (2) `set_mirror_wanted(true)` is deliberately INERT: a
mirror started against a cache that already holds positions can never catch up
and would be refused forever by the length guard while still costing memory.
(3) The mirror is real VRAM and `kv_budget::kv_bytes_per_token` must charge for
it — omitting it let a model be admitted and then OOM instead of returning the
503 that reroutes to a peer.
Worth 1.41x on GQA decode at ~2064 KV; the win is long-context only (~1.04x at
256), which is what an O(history) cost predicts.

## `inference::split::kv_cache::SeqCache` / `KvPair` + `LayerKv::truncate`

(2026-09-02, gotcha #439) — the KV cache buffer is this project's own, not
candle's, for ONE reason: candle's `Cache` keeps its length private, so the
only way to keep the first `n` positions was snapshot + `reset()` +
`append()` — two full copies of the retained prefix per layer (K and V, then
the f16 mirror rebuilt), and two prefix-sized TEMPORARIES on the device —
every time a speculative draft was rejected, i.e. once per generated token
on a prompt that drafts and misses. Found while chasing 33 → 2.8-4.7 tok/s
on a prompt of 32 tool schemas with the card at 7.9 of 8 GB — and measured
NOT to be that crawl's cause (one binary, both modes, same pressure: no
change; the cause is live KV spilling to host memory because the prefix
cache's snapshots are not charged, gotcha #439/#440). It is still waste
removed, and exact rollback accounting. `truncate` now moves
two length fields. Growth semantics are candle's, unchanged, because
`KvOccupancy` and the KV budget reason about that quantum. **Never
re-introduce a copy on the rollback path**, and never reach the inner
cache's `reset`/`append` from a call site — `LayerKv`'s inherent methods
keep the mirror in step. `SWARMLLM_KV_TRUNCATE=copy` restores the old path
for A/B inside one binary. Two tests pin it: bitwise equality with the copy
path after truncate + append, and buffer identity (no reallocation) with the
copy mode as the control that the check can see one.

## `inference::layers::rope_over_heads` — partial RoPE has one implementation, and it answers with a tensor the KV cache can write

(2026-09-11, gotcha #553, FUTURE_WORK #43.) A model whose rotary width is
narrower than its head dimension rotates the leading `rope_dim` of each head and
passes the rest through. The composition is the whole invariant:

```rust
let x_rot  = x.narrow(3, 0, rope_dim)?.contiguous()?;
let x_pass = x.narrow(3, rope_dim, head_dim - rope_dim)?;   // a VIEW
Tensor::cat(&[&rotated, &x_pass], 3)
```

**candle's `cat` answers with a transposed view rather than a fresh buffer when
any argument is non-contiguous and `dim != 0`** — it transposes every argument
to bring `dim` to the front, calls `cat0`, and transposes the result back. The
values are right; the layout is not. `slice_set` refuses a non-contiguous
source, and `slice_set` is how `SeqCache::append` writes K — so every request on
such a model died at its first layer with `attn: slice-set only supports
contiguous tensors`.

**What a change here must keep.** The result is contiguous, and the
pass-through half is made contiguous *before* the cat rather than the whole head
after it. That is not only the fix but the cheap form of it: `cat` then writes
both halves straight into one new buffer, where cat-then-`contiguous()` copies
the full head twice — 25 MB per layer on a 2048-token prefill of Phi-4-mini,
32 layers.

**Why one function.** Two copies of the branch existed —
`LayerWeights::apply_rotary_emb` and `Qwen35AttnWeights::apply_rotary_emb` — and
both were wrong. The DeepSeek MLA path next door builds its `cat` from two
explicitly `.contiguous()` halves and was always safe, with nothing recording
why. `SeqCache::append` now also makes its source contiguous, so a future
producer of K or V cannot reintroduce the class; that is free when the tensor is
already contiguous, which is every current caller.

**The discriminator is `rope_dim < head_dim`, and it is NOT GQA** — the first
cause recorded for this, from the two models' most visible difference:

| | head_count | head_count_kv | head_dim | rope.dimension_count | |
|---|---|---|---|---|---|
| Phi-3.5-mini | 32 | 32 | 96 | 96 | full RoPE — serves fine |
| Phi-4-mini | 24 | 8 | **128** | **96** | partial RoPE — died |

`head_dim` is `embedding_length / head_count` (3072/32 = 96 against 3072/24 =
128) while the rope width is 96 in both GGUFs, so GQA moves `head_dim` and
therefore correlates perfectly across these two models — and explains nothing.
A 32/8 model with `head_dim == rope_dim` never takes the branch, and a 32/32
model with partial rotary does. GLM-4 and Qwen 3.5 were broken the same way and
nobody saw it, because no such model has been run here.

⚠ **A test covering exactly this feature could not see it.**
`test_partial_rope_glm4_style` calls `apply_rotary_emb` and asserts the output
shape and that the pass-through half came back unchanged — both true of the
broken view. The only operation that refuses a view is the cache write, and the
test never reached one. `partial_rope_answers_with_a_k_the_cache_can_write` does:
it asserts contiguity directly, then drives a prefill, a decode, and the batched
path at mixed positions (which ropes *after* narrowing a row out, and so reaches
the same defect by another route).

## A RoPE layout is read off llama.cpp, per architecture (2026-09-24, FUTURE_WORK #96)

**`ModelArch::use_rope_contiguous` is llama.cpp's `llama_model_rope_type`, arch
by arch** — NEOX (contiguous halves) or NORM (interleaved pairs). GLM-4,
Llama-4 and DeepSeek-2 are NORM there and were contiguous here.

**What it cost.** GLM-4-9B-0414 answered short questions correctly and wrote
broken, repeating code for anything longer — `def factorial(n):` restarted
mid-function, indentation lost, lines cut off — on the released binary,
whole-model on the processor, no split, no GPU, speculation on or off.
**llama.cpp on the same file** (rebuilt, for this check only, from our shards:
the header plus every tensor written back to its manifest `gguf_offset`,
validated with the `gguf` package — a diagnostic done by hand, never something
the product does) **and the same 25 prompt tokens wrote a correct recursive
function.** "Short replies survive, long ones come apart" is
the signature of wrong POSITIONS, not wrong weights: the first few tokens barely
depend on rotation, later ones depend on it entirely.

**Removing the cause stops it.** The build carrying only the tokenizer fixes
(#97) — prompt already token-identical to llama.cpp's — was still garbled; the
same build plus the layout change answered llama.cpp's own function, near word
for word.

**Why nothing caught it, and what that means for a change here.**

- **Four tests asserted the wrong layout** (`test_glm4_arch_supported`,
  `test_llama4_arch_supported`, `test_deepseek_arch_supported`,
  `model_arch_properties`), each written against the function rather than the
  reference. A property test pinned to its own implementation is a change
  detector, not a check. `model_arch_properties` now states every supported
  arch's layout as llama.cpp gives it — **add a row there, from llama.cpp's
  list, for every new arch.**
- **Conformance cannot see coherence.** It asserts a reply ARRIVES, STOPS and
  leaks no marker; broken code passes all three.
- **The reply A/B compares a release with the previous one**, which was broken
  identically. Byte-identical across releases is evidence of no REGRESSION,
  never of correctness. The only check that saw this compared against a
  different implementation.
- ⚠ **DeepSeek-2's MLA bypassed the flag** — `MlaWeights::apply_rope` called the
  contiguous kernel directly. It is `rope_i` now. Llama-4 and DeepSeek-2 are
  matched to llama.cpp (and to HF, which rotates complex pairs for both) but
  **NOT run here**: no such model is on the fleet.

## A special token is what the vocabulary says it is — and a prompt gets ONE BOS (2026-09-24, FUTURE_WORK #97)

**`tokenizer::declared_special`** — CONTROL (3) or USER_DEFINED (4) in the
GGUF's `tokenizer.ggml.token_type` — decides which vocabulary entries are
matched whole in the text before the merge algorithm runs, on BOTH encoder
paths, beside the old name shapes (`<…>`, `<|…|>`) so no vocabulary that was
right changes. **`gguf_meta::add_bos_by_llama_cpp_rules`** decides whether a
prompt gets a BOS, and **`SplitTokenizer::encode`** gives none to a text that
already opens with one.

**What it cost.** Compared against llama.cpp on every local model's own chat
template (`examples/tokenizer_reference.py` → the ignored
`tokenizer_agrees_with_llama_cpp`):

| family | defect | llama.cpp agreement |
|---|---|---|
| GLM-4 | `[gMASK]` (every prompt's first token) spelled as 3 tokens; `<|endoftext|>` — one of its EOS tokens — prepended as BOS | 0/8 → **8/8** |
| Mistral v0.3 | `[INST]` / `[/INST]` (every turn) spelled as 3 tokens each; doubled BOS | special tokens now 7/7 |
| Gemma-2 | user-defined whitespace runs (all code indentation) split into single spaces; doubled BOS | 4/7 → **7/7** |
| Llama-3.x | doubled BOS (48 prompt tokens against 47) | 8/8 throughout |

Every GPT-2-style family (Qwen2.5/3, Phi-4-mini, xLAM, Llama-3) agrees in full.

**Where each came from.**

- The shapes were a guess at what "special" looks like. A vocabulary SAYS which
  of its entries are control tokens; llama.cpp reads that
  (`cache_special_tokens` / `tokenizer_st_partition`). User-defined entries are
  literal text (Gemma's are runs of spaces and newlines), which is why the raw
  vocabulary string is what gets matched.
- `add_bos_token` defaulted to TRUE on a comment saying the field "is consumed
  solely by the SPM path" — it was not; `SplitTokenizer::encode` prepends BOS
  for every variant, and an earlier fix had MOVED it there on purpose. The
  comment went stale when the consumer moved. llama.cpp: key when present; else
  SentencePiece → true, GPT-2 BPE → true only for the pre-tokenizers that ask
  (`llama-bpe`, `tekken`, …); glm4/chatglm-bpe → never.
- The doubled BOS: Llama-3, Gemma and Mistral templates render `{{ bos_token }}`,
  and the encoder then added its own. llama.cpp strips the template's copy when
  its tokenizer will add one (`common/chat.cpp`); llama-cpp-python tokenizes a
  rendered chat with no BOS of its own. Both leave exactly one.

**What the reference test asserts, and what it only reports.** It ASSERTS that
special tokens (by `token_type`) and BOS agree for every model, and that a
GPT-2-style vocabulary agrees in full. It only REPORTS SentencePiece whitespace,
where llama.cpp is no authority.

**Settled against Hugging Face, the model's own tokenizer (2026-09-24).**
`examples/tokenizer_hf_reference.py` re-references the same cases to each
model's `tokenizer.json`. llama.cpp agreed with HF in only 3/7 (Mistral), 5/8
(Phi-3.5) and 3/8 (TinyLlama) — it inserts a `▁` after every special token
(Mistral v0.3's `Metaspace(prepend_scheme: "first")` does not), segments
TinyLlama by score where HF uses its merges, and applies Phi-3's rule only by
NAME. Ours against HF:
- **TinyLlama 8/8, Mistral 6/7 — unchanged, ours was right.** The seventh is a
  text OPENING with spaces: HF skips the dummy prefix, SentencePiece itself —
  Mistral's own tokenizer — does not, and neither do we.
- **Phi-3.5 5/8 → 8/8 — `SpecialTokenSpacing::StripAfterSpecialAndPrefixEach`.**
  Its `tokenizer.json` marks every added token but `<unk>`/`<s>`/`<|endoftext|>`
  `rstrip`, and its normalizer prefixes `▁` per segment, so `<|user|>\nWrite` is
  `<|user|> ▁Write`: the newline the chat template writes after each marker is
  one the model's own tokenizer never produced. llama.cpp has the identical set
  (`llama-vocab.cpp`, the "phi-3"/"phi3" name branch) — and misses this fleet's
  file, named "Phi 3.5 Mini Instruct". Chosen here by ARCHITECTURE
  (`gguf_meta::special_token_spacing_for`), which a display name cannot defeat;
  a `phi3` GGUF with a GPT-2 vocabulary (Phi-4-mini) never reaches this encoder.
  Checked three ways: ours = HF = llama.cpp with its name rule forced on
  (`kv_overrides general.name`), 8/8 each.

**The harness is the part to keep.** No weights are read: each model gets a
SPARSE GGUF (header + a hole to the real size), because llama.cpp checks tensor
bounds even with `vocab_only`. A new model on the fleet, or any change to the
tokenizer, is one command away from being checked against an independent
implementation.

## `inference::attn_softmax::scaled_masked_softmax`

(2026-08-07) — the single
expression of attention's tail: scale, optional Gemma-2 logit soft-cap,
additive mask, softmax. Do NOT re-express those as separate candle ops in a
new attention path. Each one materialises a whole
`[batch, heads, q_len, kv_len]` score tensor — 11 MB at llama-3.2-3b prefill
shapes — so writing them out cost 34.6 ms where one fused pass costs 11.4,
and attention fell from 22.4% of a prompt chunk to 9.5% when they were folded
together. The fused CPU kernel declines anything it cannot index (non-CPU,
non-f32, strided, or a mask that is not a shared `[q_len, kv_len]` block) and
falls through to `composed`, which is the original expression and the
reference its tests compare against — so a new caller is always correct,
just possibly not fast.
**The mask is ADDITIVE f32 everywhere: `0.0` visible, `-inf` masked.** There
used to be two representations — a `u8` predicate for the standard path and a
float copy the flash arm rebuilt on every call — and a new attention backend
had to know which it was being handed. `SplitModel::causal_mask` is the only
producer. It also returns a CONTIGUOUS tensor deliberately: a `narrow()` view
costs 2.1x in `broadcast_add` and is refused by the fused kernel outright, so
any path that slices a mask (the query-blocking loop in `standard_attention`
does) must `.contiguous()` it before passing it on.
Changing the scale means changing `scale_from_head_dim`, which reproduces
candle's `tensor / f64` (an `affine(1/rhs)`, i.e. already a multiply) exactly.
`scale_matches_candle_division` pins that against candle itself rather than
against the helper — an equivalence test where both sides call the same
helper passes happily with the scale inverted.

## `inference::layers::standard_attention` grouped GQA decode

(c4cc3b16,
2026-08-16) — for `q_len == 1` with `n_kv_head < n_head`, standard attention
no longer expands the KV cache with `repeat_kv`; it reshapes the query heads
into extra matmul rows against the unexpanded cache
(`grouped_gqa_decode_attention`). Identical arithmetic — the reshape is valid
ONLY because `repeat_kv` numbers heads group-major (query head `h` belongs to
group `h / n_rep`); get that backwards and every head reads another group's
cache while still producing plausible logits, which is why
`grouping_query_heads_matches_expanding_kv_heads` compares against the
expanded path rather than asserting shapes. MHA was pinned byte-identical
(`mha_decode_matches_the_plain_path` — now within 1e-5, since the decode kernel
serves MHA decode too, 2026-08-22). This flipped the CPU decode routing: GQA decode
had been sent to the fused kernel precisely because of the `repeat_kv` cost,
and with it gone the same benchmark reports the opposite at every length
(3-9x) — so **all CPU decode now takes standard**, with the control run
reproducing the old verdict on the reverted code. 1.41x end-to-end CPU decode
on llama-3.2-3b; 4h-soak-validated (`soak_0816_cpu_speedup.md`).

## `inference::layers::cuda_decode_prefers_standard` (superseded note, 2026-08-08)

> **Superseded — read this first.** The GQA half of this note was overturned
> on 2026-08-23 and the code no longer implements it: `q_len == 1` now takes
> standard for EVERY head geometry. The live rule, with the re-measured
> table, is "Attention kernel choice and the query-length cliff (2026-08-23)"
> at the top of this file. What follows is the original write-up, kept
> because the MHA half and the forward-versus-per-call lesson still stand.

(2026-08-08) — on CUDA:
MHA decode takes standard, GQA decode takes flash **at every context length**;
prefill always flash. The GQA side rested on the same reason the CPU rule did —
`standard_attention` rebuilt the `repeat_kv` expansion every token — and that
premise changed with the grouped path above, so it is a re-measure candidate
(`docs/FUTURE_WORK.md`); it stands unchanged because GPUs already route GQA
decode to a fused kernel and this box cannot resolve a small GPU delta (#267).
The MHA side is not premise-dependent: flash has no split-KV kernel, one query
row cannot fill the card, up to 25x per call.
**There is no crossover, and re-introducing one needs a FORWARD measurement,
not a per-call one.** A 1024-token threshold shipped on 2026-08-07 from timing
the attention call in isolation; measured end to end the next day it was wrong
at every length (1.13x at kv~272, 1.42x at ~528, 1.61x at ~912 in flash's
favour). Isolated, `repeat_kv`'s allocation and bandwidth cost is amortised
against warm buffers and no competing traffic. **Third occurrence of
gotcha #255.** Controls that make the change attributable: at 2048 KV both arms were
identical (both already flash) and MHA identical to the decimal.

## `inference::layers::cuda_decode_prefers_standard` (superseded note, 2026-08-07)

(2026-08-07) — the
measured CUDA attention routing rule, extracted so it is testable without
a GPU. **The right kernel is opposite for prefill and decode, and it turns
on GQA** — the same lesson as the CPU crossover above it (gotcha #255) on
a different device. Flash unconditionally costs up to **25x per attention
call** on MHA decode, because candle-flash-attn ships no split-KV kernel
and one query row cannot fill the card; GQA reverses above ~1k context
because `standard_attention` rebuilds the `repeat_kv` expansion every
token. Changing the constant means re-running
`flash_vs_standard_attention_on_cuda` — the measured table lives in the
dispatch's comment and in `docs/FUTURE_WORK.md`, and the benchmark
asserts the dispatch never picks a kernel materially slower than
always-standard.

## `inference::mem_bandwidth::measured_gbps`

(2026-08-18) — what this machine's
memory actually delivers, measured once and cached. **The figure a processor-only
node advertises as its speed.** It was `estimate_tokens_per_sec_7b(50.0, false)` —
a hardcoded bandwidth for every machine — so every CPU node in the swarm quoted
the identical 1.70 tok/s whether it was an eight-channel server or a fanless
mini-PC. Nothing could tell two of them apart, which is why a delegation gate
comparing them would have been comparing a constant with itself. Measured 29.9
GB/s on the 5800H laptop this was written on, against the 50 assumed.
Buffer must exceed any last-level cache (256 MB) or it reports cache bandwidth;
min-of-3 because every error source is additive; reads at decode width, not
thread-per-core, so it ranks machines the way running a model does. Costs 254 ms
once, on the health-monitor task rather than the startup path.
**Adding a device class means giving it a real measurement, not a constant.**

## `inference::cancel::unless_cancelled` — every wait that can run for minutes watches the request's cancel flag

(2026-09-03, gotcha #445). The flag
(`InferenceRequest::cancel`) is the ONE cancellation signal: set by
`CancelOnDisconnect` (non-streaming), by both SSE surfaces on
`sse_tx.closed()` (they used to only drop `token_rx`, which the pipeline
notices at its next send — after the prompt pass), and by `/cancel`. Read
by `ModelProcessPool::forward_for_request` around the WAIT for the worker's
answer (never around the send: a half-written `Forward` frame corrupts the
worker's stream), by `PipelineExecutor::wait_for_result` for a remote
segment (the caller then sends `CancelInference` and does NOT fail over),
and by the per-token loop as before. Dropping the wait is the mechanism —
the armed `ResponseGuard` sends `CancelRequest`, the worker skips a queued
forward and stops a running one between layers. The router never retries a
request whose flag is set; the marker error is `REQUEST_ABANDONED`
(`ServiceUnavailable`, penalty-exempt, matched only by `is_request_
abandoned`). **A new wait longer than a token goes through this helper**, and
a new surface that learns the client left must set the flag — a tester's
worker ran 81 CPU-minutes on two one-layer segments after the client had
gone because the flag was read in one place and set in one other.

## A prompt pass asks between layers whether its request was cancelled

(2026-09-02, gotcha #441). `KvCacheStore::set_cancel_oracle` is installed by
the worker over its `cancelled` set; `forward_inner_impl` probes it once per
layer and returns `CANCELLED_MID_FORWARD`; `forward_was_cancelled` on the
worker is the ONE place that looks at that message (the sequential path
answers a normal `GenerateDone{finish_reason:"cancelled"}` and clears the
KV; the batch path lets the drain step collect the slot). Why: the cancel
check lived only in the decode loop, and the prompt is one forward before
the first token — minutes on a processor-only node with an agent-sized
prompt, which a tester found still pegging five cores eleven minutes after
"cancelling" was logged. A cancel check belongs at the granularity of the
WORK; a new long-running loop inside a forward inherits this probe only if
it runs per layer, so anything longer than a layer must probe on its own —
which is what `SplitModel::forward_prompt_in_chunks` does for the segment
path (2026-09-03, gotcha #445): a prompt longer than `prefill_chunk_tokens`
runs in chunks, `index_pos` advancing, and probes between them, because a
ONE-layer segment (the boomerang's local ends) has no between-layers at
all. Parity with the one-shot pass is pinned on the output and on the
cache left behind; a decode step is one position and takes the one-shot
path unchanged.

## `inference::split::token_embedding::rows_on_demand_eligible`

(2026-08-18) — the
single answer to "is this model's `token_embd.weight` held quantized with its rows
dequantized on lookup, or dequantized whole at load?". **Two places must agree**: the
loader, which allocates, and the footprint estimators, which decide whether the model
is admitted at all. A disagreement is invisible until a node either refuses a model
that would have fitted or is admitted and then runs out of memory — the same trap
`EMBEDDING_DTYPE` already carries a test for. `table_supports_row_gather` is the
device-independent half, for the estimators, which are built once and consulted for
both a CPU and a CUDA worker; the `SWARMLLM_DENSE_EMBEDDING` override lives in THAT
inner predicate so both callers inherit it, because putting it one level up left the
estimator pricing a gather the loader was not doing.
**The gather must stay on the device holding the table.** `QTensor::data()` is a
zero-copy borrow on CPU and a full device-to-host copy on CUDA, so the CPU
implementation reused on a GPU would move the whole table across PCIe every decode
step — llama.cpp measured that shape at 6.18 ms/token against 1.72 before
`k_get_rows_kq`. Both devices therefore go through the vendored
`QTensor::gather_rows`: CPU slices rows out of the borrow, CUDA runs `index_select`
over a `[vocab, row_bytes]` byte view (no new kernel — `is_u32_u8` is already in
`candle-kernels`, and the quantized buffer's padding is only ever trailing, so rows
are contiguous). Metal has none and keeps the dense table.
Measured 754 MB on CPU and 736 MB on an RTX 3070, both llama-3.2-3b against a 751 MB
prediction. Weight-tied models gain most because the loader used to load that tensor
TWICE — once dequantized for the lookup, once quantized for the LM head — and now
shares one `Arc<QTensor>` via `QMatMul::from_arc`.
**Verify a change here with DECODE RATE, not memory**: the failure mode above frees
exactly as much memory while being far slower. The check that rules it out is
PREFILL — gathering 512 rows costs no more than gathering 1 would if each row made a
host trip, so unchanged prefill is positive evidence the gather stayed on-device.
A new embedding path goes through `TokenEmbedding`, whose two variants both return
`EMBEDDING_DTYPE` so no call site can tell them apart.

## `inference::split::read_gguf_header`

(2026-08-29) — the single way to parse
a GGUF header off a PATH, and the buffering is the entire reason it exists.
`gguf_file::Content::read` walks the metadata with many tiny reads — for every
string a length, then its bytes — so handing it a bare `std::fs::File` turns
each one into a syscall. A 7.8 MB header carrying a 128k-token vocabulary and
280k merges is roughly 820k of them.
**Measured on the live node** (gotcha #410): `GET /api/admin/models` took a
stable 11.2 s, of which **9.6 s was KERNEL time** — it parses every local
model's header and seven call sites were passing an unbuffered handle.
Optimisation cannot touch syscall count, which is why the release binary was
no faster than a debug one on that path. Direct A/B on one header: 980 ms
unbuffered against 98 ms buffered.
**Two sites deliberately do NOT use it** — `local_embedder` and `vision` keep
their own `BufReader`, because the same handle goes on to read tensors and the
helper's handle dies with it. They still buffer; that is the invariant, not the
helper. A `Cursor` over a slice or an mmap already holds the bytes and needs
neither.
**`ModelProcessPool::footprint_inputs` also parsed the same file twice** — once
through `GgufTokenizerMeta::from_gguf_file`, which materialises the whole
vocabulary and merge list as owned `String`s, purely to reach `vocab.len()` as
a fallback, and once through its own reader for everything else. The count was
already in the parsed `Content`; counting the array borrows it.
`a_gguf_header_is_never_parsed_straight_off_an_unbuffered_file` in
`tests/repo_consistency.rs` fails the build on a new unbuffered site. It checks
PROXIMITY — a `File::open` within a few lines of a `Content::read` with no
`BufReader` or `Cursor` between them — not naming. A naming rule was the first
cut and it silently missed the `match File::open { Ok(mut f) => Content::read(
&mut f)` form, which is the shape one of the seven sites actually had.
**The general rule**: before theorising about why something is slow, split
user from system time (`utime`/`stime` in `/proc/<pid>/stat`). It is two
numbers and it partitions the hypothesis space in one step — kernel-dominated
means syscalls or waiting, and no amount of reading the code distinguishes
"parses a lot of metadata" from "makes 820k read calls". A **stable** duration
is a fixed amount of work, not contention; go and find the count.

## `inference::split::GgufTensorMeta::tied_output_location`

The single
definition of "is this model weight-tied", i.e. does it reuse
`token_embd.weight` as the LM head instead of shipping an `output.weight`.
Consumed by BOTH sidecar writers (`daemon::manifest::extract_tied_output_weight`,
`huggingface::probe::download_tied_output_weight`) AND the reader
(`inference::split::resolve_tied_output` → `ShardReader`). Producer and
consumer MUST agree on which tensor the sidecar holds; a new surface that
needs the predicate goes through this method rather than re-deriving
`contains_key("output.weight")`. The sidecar filename is
`inference::split::TIED_OUTPUT_FILENAME`, never a literal.
**Why this exists**: a node serving the LAST pipeline segment needs the output
head, but on a weight-tied model that tensor physically lives in shard 0 —
which that node frequently does not hold. The sidecar carries the raw bytes;
`ShardReader::new` maps them over the tensor's gguf byte range so
`ct.tensor(&mut reader, "token_embd.weight", …)` resolves unchanged. It maps
the sidecar ONLY when no local shard already covers that offset, since a
duplicate `gguf_offset` would make `find_shard`'s binary search ambiguous.
`tied_output` is a REQUIRED parameter on `ShardReader::new` with no
convenience wrapper — for three releases the sidecar had three writers and
zero readers, and every weight-tied model was unservable on any node lacking
shard 0 (gotcha #178).

## A model's turn-ender is found in its vocabulary, not taken from its declared EOS

**Rule:** `.claude/rules/arch-inference.md` § "A model's turn-ender is found in its
vocabulary, not taken from its declared EOS".

### What it replaced

`eos_tokens_with_arch_fallback` carried per-family id lists for `qwen*` and
`gemma*`, and they sat behind `if ids.is_empty()` — so they only ran for a GGUF
that declared NO EOS at all. A model that declares one token and ends its turns
with a different one was never considered. `split::entry` did not call the
function at all: it used the declared ids verbatim, or a hardcoded
`[2, 107, 32000]` when there were none.

### What it was measured at

Phi-3.5-mini-instruct-Q4_K_M, on the released v0.3.171 binary, 2026-09-11:

- `tokenizer.ggml.eos_token_id = 32000` (`<|endoftext|>`), no `eot_token_id`
  key, while `tokenizer.chat_template` closes every turn with `<|end|>` — token
  **32007**, `token_type = 3` (CONTROL).
- Resolved EOS set before: `[32000]`. After: `[32000, 32007]`.

**Confirmed on a second model, 2026-09-11**, which is what shows the name search
was right rather than lucky. Phi-4-mini-instruct-Q4_K_M has the identical defect —
`eos_token_id = 199999` (`<|endoftext|>`), turns closed with `<|end|>` — but
`<|end|>` is id **200020** there against 32007 in Phi-3.5, and its vocabulary is
`gpt2` BPE against Phi-3.5's `llama` SentencePiece. So the same fault sits at a
different id in a different vocabulary family under the same `general.architecture`
string, `phi3`. **A per-architecture id table — the obvious fix — needs an entry
per quantisation and breaks on the next one; a name is stable across both.**
`<|return|>` and `<|call|>` are absent from that vocabulary, so the harmony
exclusion correctly does not fire.
- End to end, a plain request with NO tools — "Say exactly: hello",
  `max_tokens: 120`, `temperature: 0` — returned `finish_reason: "length"` and
  120/120 completion tokens: *"Hello! How can I help you today? Hello! I'm Phi,
  an AI digital assistant. What can I do for you? 你好，我需要一个关于如何在
  Python中处理JSON数据的详细解释…"* — the model ended its turn, invented a second
  assistant turn, then a fabricated user turn in Chinese, and began answering it.

**The symptom differs by vocabulary family from this one cause**, which is why
two field reports read as unrelated bugs:

| Vocab | `decode_token_impl` does | Symptom |
|---|---|---|
| SentencePiece (Phi-3.5 GGUF) | any `<…>` token → empty | marker invisible, silent run-on |
| GPT-2 byte BPE (Phi-4-mini GGUF) | chars → their own bytes | literal `<|end|>` in content, plus run-on |

That is why the fix is the **token id**, not only a stop string: on the
SentencePiece path the text never contains `<|end|>`, so a stop string could
never match it.

### What a change must keep

- **The harmony exclusion.** `<|end|>` ends generation ONLY when the vocabulary
  does not also hold `<|return|>`/`<|call|>` (o200k_harmony, gpt-oss) or
  `<|calls|>`/`<|flush|>` (solar-open), where it separates messages inside a
  reply still being written. Adding `<|end|>` unconditionally is correct for Phi
  and silently truncates every harmony reply at its first message. llama.cpp
  carries the identical exclusion in `llama_vocab::impl::load`, and reading that
  before shipping is the only reason this is not in the codebase as a one-liner
  bug.
- **`</s>` and `<eos>` stay OUT**, though llama.cpp's candidate list has them.
  Every name in `END_OF_GENERATION_TOKENS` is a `<|…|>` form whose only role in
  any family is ending a turn; those two instead sit unused in the vocabularies
  of families that never emit them — Phi-3.5's SPM vocab holds `</s>` at id 2 and
  ends its turns with `<|end|>`. The severity ordering this file's sibling
  already documents decides it: **a wrong EOS truncates SILENTLY, an unknown one
  at worst runs to `max_tokens`, and those are not close.** A genuine Llama-2
  vocabulary still gets id 2, from the narrowly-scoped
  `ids.is_empty() && plausible(2)` branch.
- **Turn OPENERS are not end-of-generation tokens.** `<|user|>`,
  `<|assistant|>`, `<|system|>` mean the model has gone wrong and belong in
  `extract_stop_strings` and `CONTROL_TOKEN_NAMES`, not here.
- **Every path that resolves EOS ids merges the search in.** There were four,
  and one of them (`split::entry`) had never called the arch fallback either.

### The three places that act on a turn marker, and why each is separate

1. **The EOS token id** (`gguf_meta`) — stops generation. Works on both vocab
   families because it is checked on the id, before decoding.
2. **`chat_template::extract_stop_strings`** — can also REMOVE a leaked marker
   from the text, and is the only one that can. Template-gated, with the harmony
   exclusion applied textually.
3. **`inference::CONTROL_TOKEN_NAMES`** — scrubs a marker that leaked anyway.
   Safe unconditionally, harmony included: *removing a control marker from
   visible text and ending the reply at it are different decisions, and only the
   second one truncates.*

`extract_stop_strings(None)` returning ChatML's single marker was a fourth bug in
the same area: `build_prompt_inner`'s no-template branch asks
`fallback_by_model_name` first, so the prompt may be zephyr, llama3, gemma,
mistral or vicuna — and each of those was left with no stop for the marker it
will actually emit. It now returns the always-on list.


## A decode token is bound by GPU submission COUNT, not bandwidth (2026-09-22)

Measured on the live release node — v0.3.197-alpha, Ryzen 7 5800H, RTX 3070
Laptop 8 GB, WSL2 — with nsys and the worker's own forward-pass timer.

### What it replaced

The belief, written into `docs/plans/regional_pipelines.md` § "Why distance
costs so much", that "the 8B is 32 layers ≈ 28 ms/token on this card whether
one machine does it or four". The number is real; the attribution is not.
**Most of a decode token is CPU-side CUDA submission cost, and the card is idle
for about half of it.** That matters beyond bookkeeping: the plan's crossover
arithmetic (a 4-way GPU split beats local CPU below ~90 ms RTT) is computed
against a compute term that is mostly overhead, so the local arm of every such
comparison is a moving target until this is fixed.

### What it was measured at

Three independent readings, each naming its own mechanism.

**1. Per-layer cost is FLAT across a 3.3x span of bytes per token** — which a
bandwidth-bound decode cannot be. `SWARMLLM_PROFILE=1` brackets
`SplitModel::forward` alone (no IPC, no sampling, no HTTP); median of ~32
decode steps, each model loaded by itself on an otherwise clean card:

| model | quant | L | bytes/token | ms/token | ms/layer |
|---|---|---|---|---|---|
| tinyllama-1.1b | Q4_K_M | 22 | ~0.6 GB | 12.0 | 0.545 |
| qwen2.5-0.5b | **F16** | 24 | ~1.0 GB | 11.5 | 0.479 |
| gemma-2-2b-it | Q4_K_M | 26 | ~1.6 GB | 15.0 | 0.577 |
| qwen3-1.7b | Q8_0 | 28 | ~1.8 GB | 16.0 | 0.571 |
| llama-3.2-3b | Q4_K_M | 28 | ~2.0 GB | 16.0 | 0.571 |

qwen2.5-0.5b at F16 moves ~1.7x tinyllama's bytes per token and is **faster**.
Effective bandwidth runs 50 GB/s (tinyllama) to 125 GB/s (3b) against a card
that delivers ~384 — 13-33% of roofline, and **the smallest model is the least
efficient**, which is backwards for anything bandwidth-bound. `ms/layer` is
constant to within ±10% while `bytes/token` moves 3.3x, so layer count, not
size, predicts the cost.

**2. The bottleneck is ONE SATURATED CPU THREAD, not the GPU.**
`/proc/<worker>/stat` utime+stime across a 200-token generation, 3 reps, with
profiling off so nothing is added to the path:

```
worker burned 22.7 / 26.0 / 24.6 ms CPU per token
     against   21.0 / 24.1 / 22.6 ms WALL per token   = 1.08-1.09 cores busy
```

CPU time ≈ wall time means the thread is never waiting on the device. Were the
GPU the constraint, the thread would sit blocked and this would read far below
100%. The other side agrees: GPU utilization sampled every 200 ms through a
sustained decode was **median 52%, memory controller 33%**.

**3. What the thread is doing: 1,085 GPU submissions per token.** nsys
`--trace=cuda` with the worker captured via `--trace-fork-before-exec`,
counted over the steady-state decode window ONLY — model load and warm-up
excluded, because weight upload does thousands of calls that have nothing to do
with decode. tinyllama, 22 layers, per token:

| API | calls/token | median | ms/token |
|---|---|---|---|
| `cuLaunchKernel` | **761** | 12.0 us | 8.6 |
| `cuMemsetD8Async` | **323** | 11.8 us | 3.8 |
| `cuMemAllocAsync` | 705 | 1.4 us | 1.0 |
| `cuMemFreeAsync` | 704 | 1.3 us | 0.9 |
| `cuEventCreate` + `cuEventDestroy` | 2,818 | 0.3 us | 0.95 |

**17.55 of that token's 23.02 ms went inside the CUDA driver API**, 13.2 of it
in launch + memset alone — **34.6 kernel launches and 14.7 memsets per LAYER**,
for a model whose per-layer arithmetic is seven matmuls and a handful of
elementwise passes.

A launch costs ~10-12 us here because WSL2's virtualised driver marshals every
submission through to the Windows host driver; native Linux is ~3-5 us. **So
the CONSTANT is box-specific and the STRUCTURE is not** — at 5 us a launch the
same token still spends ~5 ms submitting, and every node that is not a
fully-native Linux GPU box pays nearer the number above. This has NOT been
A/B'd against a native-Linux GPU, because the fleet has exactly one GPU node
and it is this one; treat the multiplier as unvalidated and the ordering as
established.

⚠ **Counts are exact under nsys; the per-call TIMES are nsys-inflated.** Do not
quote 12.0 us as the unprofiled cost of a launch.

### The 323 memsets were free to delete, and that is what `alloc_fully_overwritten` is

14.7 memsets per layer is two per quantized matmul, and both come from
`alloc_zeros`, which is `alloc` **plus** a `cuMemsetD8Async`. Both buffers are
overwritten in full by the very next kernel:

- **`y_q8_1` / `input_quant`** — `quantize_q8_1`'s grid covers `kx_padded`, and
  the padding tail is written explicitly: `ix < kx ? x[iy*kx + ix] : 0.0f`.
  **That ternary only exists because the reference implementation does not
  pre-zero either** — upstream llama.cpp hands `quantize_row_q8_1_cuda` pool
  memory. A zeroed buffer would make the branch dead code.
- **`dst`, `mul_mat_vec_q`** — `dst[j*nrows_dst + row0 + threadIdx.x] = tmp[...]`,
  an assignment, and the grid covers every row (`nblocks == nrows` at b_size 1,
  `ceil_div(nrows, 2)` with `rows_per_cuda_block == 2` above it).
- **`dst`, MMQ** — assigns every in-range element, skipping only what is out of
  range (`col_dst >= ncols_dst` returns, `row_dst >= nrows_dst` continues), and
  the grid is `ceil_div` of both dimensions, so each element is covered once.

`CudaDevice::alloc_fully_overwritten` is now the single way to ask for such a
buffer. Upstream candle already does this for `dequantize_f32`'s output, so the
pattern is not novel — it was simply not applied on the hot path.

#### What it measured — A/B/A/B in one binary, 2026-09-22

Arms differ ONLY by `SWARMLLM_ZERO_QMATMUL_BUFFERS`, in one
`--features dev,claude-subscription,candle-cuda` release build, so nothing but
the zero-fill changes between them.

**The mechanism, from `examples/decode_submissions.sh`** — everything except
the memsets is byte-identical, which is what makes it an experiment rather than
an observation:

| per token | patched | zeroed |
|---|---|---|
| `cuLaunchKernel` | 625.3 | 625.3 |
| `cuMemsetD8Async` | **1.4** | **321.1** |
| `cuMemAllocAsync` | 656.9 | 656.9 |
| event ops | 2,625 | 2,625 |
| **submissions/token** | **672** | **992** |

**The outcome**, `examples/stream_bench.py`-style client-side decode window,
3 reps per arm, run A/B/A/B:

| model | A1 | B1 | A2 | B2 | A median | B median | delta |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b (22 L) | 72.4 | 53.0 | 72.5 | 58.3 | **72.5** | 55.7 | **+30%** |
| llama-3.2-3b (28 L) | 54.8 | 50.4 | 59.3 | 48.8 | **57.1** | 49.6 | **+15%** |

No overlap between arms on either model, and the repeat of A landed within
0.1 tok/s of the first on tinyllama. **Correctness: all four arms, and the
shipped v0.3.197 release binary, produced byte-identical replies** at
temperature 0 (`d923429be4`, `b6f3ae2560`) — which is the check that an
uninitialized buffer is in fact fully overwritten.

⚠ **The two models' deltas differ by more than the submission arithmetic
predicts** (-32% of submissions on both). Decode is submission-bound but not
*only* submission-bound; the 3b moves 3.3x the bytes per token, so bandwidth is
a larger share of its token and the same submissions removed buy proportionally
less. **Expect the win to shrink as models get bigger**, and do not quote the
tinyllama figure for a 7B.

⚠ **Not measured on native Linux.** Every figure here is WSL2, where a
submission costs ~2-3x native. The direction holds anywhere; the magnitude is
this box's.

### The 2,625 event ops per token were guarding a hazard that cannot occur

cudarc creates a read event AND a write event for every `CudaSlice` while
`is_event_tracking()` is on (the default), waits on both in `Drop`, and destroys
both — **four event API calls per allocation, ~700 allocations a token**. They
exist to synchronise a buffer used across MULTIPLE streams.

⚠ **RESTATED 2026-09-22 when the stream migration landed.** The justification
was "there is only ever one stream, process-wide", which was true while every
device took `default_stream()` — cudarc hands back `cu_stream: null_mut()` for
that, *the same legacy stream for all of them*. `BackendDevice::new` now takes
`context.new_stream()`, so each device has its OWN stream and that sentence is
false while the conclusion still holds. **The invariant is ONE STREAM PER
DEVICE, and buffers never crossing devices:**

- `BackendDevice::new` — the constructor every production path reaches, via
  `Device::new_cuda` / `cuda_if_available` — takes exactly one stream per
  device, whichever kind.
- Same-stream ordering needs no events: `cuMemFreeAsync` on the allocating
  stream is ordered after the work queued before it.
- Several devices ARE built (the daemon's capability probe, the shard loader).
  candle gives them different `DeviceId`s and refuses to mix tensors across
  devices, so a buffer cannot reach another device's stream. **This was a
  hypothetical about a constructor nobody called; it is now the load-bearing
  bullet.**
- cudarc's `is_managing_stream_synchronization()` is
  `is_in_multi_stream_mode() && is_event_tracking()`. `is_in_multi_stream_mode()`
  is now TRUE, so this is false *only* because tracking is off — before the
  migration it was false twice over.
  ⚠ **So `SWARMLLM_CUDA_EVENT_TRACKING=1` is no longer a pure revert**: it
  restores the events AND hands cudarc back stream-synchronisation management.
  It can only ADD synchronisation, so it remains a valid A/B; say which it is
  when quoting it.

**Why the migration happened at all**: CUDA refuses graph capture on the legacy
stream, and a graph is the one change that collapses a token's ~513 launches and
~1,300 alloc/free calls into a single submission. Measured, not assumed —
`examples/cuda_graph_probe.cu` arm A gets `cudaError 900`, and that arm is a null
control that would report the claim wrong if capture were permitted.
⛔ **The migration SHIPPED BROKEN in v0.3.199-alpha** — garbage from every
model in a `--features cuda` build — and is now opt-in via
`SWARMLLM_CUDA_OWN_STREAM=1`, default OFF. See gotcha #683.

### Why .199 broke: one kernel was not on the device's stream (found 2026-09-23)

The one-stream-per-device invariant above is only as good as the claim that
EVERY launch uses the device's stream. One did not. `vendor/candle-flash-attn/
kernels/flash_api.cu` ended `cudaStream_t stream = 0; // Use the default
stream.` The Rust side DID take `dev.cuda_stream()` — for its `device_ptr`
guards — which is what the Stage 4a review read. **"X uses the device's stream"
is verified at the LAUNCH, not at the first mention of the stream** (gotcha
#685).

cudarc's `new_stream()` creates `CU_STREAM_NON_BLOCKING`, and CUDA's runtime
docs are explicit: "The legacy default stream … synchronizes with all other
streams in the same CUcontext **except for non-blocking streams**." So on an own
stream, flash read Q/K/V before candle's kernels had written them and candle
read flash's output before it existed. Prefill is always flash on CUDA, so every
KV cache was built from garbage.

**Isolated on the published .200 binary with environment switches only** — no
rebuild, so nothing but the switch moved (tinyllama and llama-3.2-3b,
temperature 0):

| arm | replies |
|---|---|
| legacy stream, flash | correct |
| **own stream, flash** | **`给给给…` / `<\|reserved_special_token_247\|>…`** |
| own stream, flash, `CUDA_LAUNCH_BLOCKING=1` | identical to the first arm |
| own stream, `force_standard_attn` | identical to legacy + standard |

Serialising every launch cures it, so it is ORDERING, not arithmetic; removing
flash cures it, so it is the one kernel off the stream. Upstream candle hit the
identical race (PR #3596: "the attention kernels launch on a different stream
than the one that produced Q/K/V … a data race") and fixed it in 0.11.0 via
#3655; mistral.rs's own flash and paged-attention crates pass
`dev.cuda_stream().cu_stream()` into every launcher. We vendor 0.10.1.

**The fix is upstream's patch shape**: `run_mha(…, void *stream_ptr)` launches on
`reinterpret_cast<cudaStream_t>(stream_ptr)`, and both Rust call sites pass
`stream.cu_stream()`. "The types CUstream and cudaStream_t are identical and may
be used interchangeably" (CUDA runtime API, driver interop), and on the legacy
stream cudarc's handle is null — so the default build launches exactly where it
always did.

What keeps it fixed:

- `the_vendored_attention_kernels_launch_on_the_devices_stream`
  (`tests/repo_consistency.rs`, every push) reads the source with comments
  stripped, and `the_attention_stream_guard_catches_the_pre_fix_source` plants
  the old forms. It also covers `candle-paged-attention`, which hardcodes stream
  0 in three places and is exempt only while nothing depends on it.
- `flash_launches_on_the_devices_own_stream` (`layers/mod.rs`, needs a card and
  `--features flash-attn`) builds the device with `new_cuda_with_stream` and
  WIDENS the race: Q is produced as `q + 0` behind a 4096³ matmul, so on a
  wrong stream flash reads it before it exists. **Seen red**: with the old
  null-stream launch put back, round 0 passed and round 1 read `inf` — a race
  does not lose every time, hence three rounds. With the fix: 4.6e-4, all
  rounds.
- **Real generations on the fixed `--features cuda` build**: with
  `SWARMLLM_CUDA_OWN_STREAM=1`, tinyllama and llama-3.2-3b replies are
  byte-identical to the released v0.3.200 (which produced `给给给…` under the
  same switch), with and without `CUDA_LAUNCH_BLOCKING=1`.

⚠ **The switch stays OFF.** An own stream buys nothing until graph capture
exists; flipping the default is its own change and needs its own behaviour gate.

`Drop` has no synchronous fallback when the events are absent — it skips the two
`stream.wait()` calls and frees as before — so disabling is strictly less work.
`SWARMLLM_CUDA_EVENT_TRACKING=1` restores it.

**Mechanism**: 2,625 → 0 event ops per token, with launches (671), memsets (1)
and allocs (657) identical between arms — one variable.

**Outcome, re-measured on an idle box** (500-token generations, 8 reps, A/B/A/B,
one model resident). ⚠ The first attempt was taken with **Chrome's GPU process
at 109% of a core** — the same confound that made the 0901 baseline
non-comparable — and its arms overlapped; these replaced it:

| model | off (best-of-N) | on (best-of-N) | verdict |
|---|---|---|---|
| tinyllama-1.1b (22 L) | **107.7 / 109.1** | 93.6 / 93.6 | **+15%**, no overlap |
| llama-3.2-3b (28 L) | 71.1 / 78.8 | 72.9 / 74.7 | **no measurable effect** |

**It helps the small model and does nothing for the 3B** — the same shape as the
memset fix (+30% / +15%), and for the same reason: a fixed CPU saving is a
smaller share of a token that carries 3.3x the bytes. **Do not quote a single
figure for this change.** Replies were byte-identical across every arm.

⚠ **Read best-of-N here, not the median.** The benchmark node is joined to the
LIVE SWARM, and its daemon intermittently takes the core the decode thread needs
— one rep measured 23.7 tok/s among others at 80-110. That wrecks the median
(spreads 41-63% even on an idle box) while leaving the best sample clean, since
interference only ever subtracts. min-of-N on an idle box is the documented
method for exactly this reason (#367); the medians here overlap where the bests
separate cleanly, and the medians are the wrong statistic, not a contrary
result. An isolated node (private gossip id, no bootstrap, no mDNS — the shape
`examples/constrained_node_test.sh` uses) would remove the outliers at the
source and is the better instrument if this needs to get finer.

### Both changes together: +34%, and it corrects the per-change figures

The cleanest reading of the two, because it is the only one taken with the box
genuinely idle, 500-token generations, 8 reps, A/B/A/B, both switches flipped at
once inside ONE binary:

| model | both fixes (median / best) | neither | delta |
|---|---|---|---|
| tinyllama-1.1b | 92.7, 94.5 / 109.3, 120.5 | 69.1, 70.9 / 83.1, 86.9 | **+34%** |
| llama-3.2-3b | 71.7, 68.2 / 76.0, 73.0 | 53.0, 51.4 / 60.3, 52.9 | **+34%** |

**No overlap on either statistic, on either model** — the first measurement in
this whole investigation where the median and the best agree, which is itself
the sign the instrument was finally quiet.

⚠ **This CONTRADICTS the per-change numbers above and supersedes them.**
Compounding them (+30% then +15% on the 1.1B; +15% then nothing on the 3B)
predicts +50% and +15%; the measurement says +34% and +34%. They cannot all be
right, and the cumulative one is the better-conditioned: the per-change arms
were taken on a box with a browser holding a core, at 120 tokens a run, where
the median and best disagreed. **Quote +34% for the pair. Treat the per-change
split as indicative only**, and do not use it to argue that one change is worth
more than the other on a given model size — that split has not been measured
well enough to carry the claim.

The 3B gaining as much as the 1.1B is the part that did not survive: the earlier
reading had it at half, and the honest conclusion is that the earlier reading
was noise, not that big models benefit less. **The "smaller models gain more"
story is NOT established** by this data.

### Projections that share an activation must share the work — `QMatMul::forward_shared`

**How the kernel mix was established, and why guessing had to stop.** nsys gives
no GPU-side kernel table on WSL2, so `CudaDevice::get_or_load_func` — the path
every BUILT-IN candle kernel launch takes — counts launches by name under
`SWARMLLM_COUNT_KERNELS=1`, dumped per forward pass beside the stage profile.
⚠ **`get_or_load_custom_func` counts too, and must**: SwarmLLM's own PTX kernels
are the only thing on that path, so counting only candle's would make every
fusion read as removing one launch per layer more than it does — the instrument
wrong in exactly the direction that flatters the change it exists to judge.
⚠ **The flag is sufficient on its own now.** It was not: the block that prints
was gated on DEBUG-or-`SWARMLLM_PROFILE`, so setting it and seeing nothing was
indistinguishable from the thing not happening (gotcha #681).
One decode token, tinyllama, 22 layers, **601 launches, 27.3 per layer**:

| kernel | /layer | | kernel | /layer |
|---|---|---|---|---|
| `quantize_q8_1` | **7.05** | | `affine_f32` | 1.00 |
| `mul_mat_vec_q4_K/q6_K` | 7.04 | | `bmul_f32` | 1.00 |
| `rmsnorm_f32` | 2.05 | | `softmax_f32` | 1.00 |
| `badd_f32` | 2.00 | | `ucopy_f32` | 1.00 |
| `copy2d_f32` | 2.00 | | `usilu_f32` | 1.00 |
| `rope_i_f32` | 2.00 | | | |

**`quantize_q8_1` ran exactly once per matmul — 7 times for 4 distinct
activations.** Q/K/V are three matmuls against the same post-attention-norm
hidden state and the FFN's gate/up are two against the same post-FFN-norm one,
so 3 per layer rebuilt a buffer byte-for-byte identical to one just built. That
was the only outright *waste* in the table; the copies are KV-cache writes and
the rest is structure that fusion would address, not redundancy.

`QMatMul::forward_shared(xs, ws)` fixes both shapes it found:

- **`Standard` weights** (llama, qwen, gemma, mistral…) — candle quantizes the
  activation once for the group. Verified by count: `quantize_q8_1` **7.05 →
  4.05 per layer**, 601 → 535 launches, every other kernel count unchanged.
- **`FusedSlice` weights** (Phi-3/3.5/4, whose GGUF ships ONE fused QKV tensor)
  — each projection was running the **whole fused matmul** and narrowing its own
  slice out, so Q, K and V each computed all three. Now it runs once.

**Measured** (idle box, A/B/A/B where noted, `SWARMLLM_SHARE_PROJECTIONS=0` is
the off arm, replies byte-identical in every arm):

| model | shared | unshared | delta |
|---|---|---|---|
| **phi-3.5-mini** (fused) | **71.5, 70.4** med | 46.2, 44.7 | **+55%** |
| **phi-4-mini** (fused) | **49.4** med | 40.4 | **+22%** |
| llama-3.2-3b | 74.0, 72.1 best | 72.0, 68.0 | +4% |
| tinyllama-1.1b | 101.4, 99.5 med | 94.0, 93.0 | +7% |

⚠ **The two halves are worth wildly different amounts and must not be quoted
together.** The Phi win is real matmul work removed — two thirds of that
projection's arithmetic — and separates by a margin nothing on this box can
explain away. The shared-quantization win is ~11% of launches, i.e. ~5% of a
token, and sits at the edge of what is resolvable: on tinyllama the medians
separate and the bests overlap, on the 3B the reverse. **Do not present ~5% as
established for the non-fused models on this evidence.**

⚠ **`forward_shared` must stay optional.** It falls back to a plain loop for a
single weight, a mixed group, dequantized weights and non-CUDA, so it is always
correct to call — and LoRA is applied AFTER it returns, deliberately, because
LoRA's own matmuls do not share that activation.

#### Both quantized-matmul paths need it — the first version only fixed one

⚠ **There are TWO quantized matmul entry points and the sharing landed on one.**
`mul_mat_vec_via_q8_1` serves a single position (decode); `mul_mat_via_q8_1`
serves a batch through the MMQ kernel (prefill, and any `b*m` over `max_bm`).
After the first fix, decode showed 4.05 `quantize_q8_1` per layer and **prefill
still showed 7.05** — `.claude/rules/architecture.md` § "One invariant, N paths"
in miniature, on a helper written the same hour.

**Only the per-kernel count caught it.** Nothing failed, no test went red, and
the decode measurement looked like a complete success. The lesson is the rule's
own: *enumerate the paths before believing a shared fix is applied* — and when a
count exists, read it on every path, not the one you changed.

Both paths share now (prefill 736 → 670 launches, `quantize_q8_1` 7.05 → 4.05
per layer, one shared buffer layout — `k_padded × rows` — so the two are
interchangeable).

⚠ **No measurable prefill speed-up, and that is expected.** Min-of-N over 56
`seq_len=128` chunk forwards, back to back: **13 ms against 14 (min), 27 against
26 (median)** — inside the noise. Prefill does real arithmetic over 128 rows, so
launches are a far smaller share of its time than of a decode step's, and the
quantization it removes is small beside the matmuls. **Shipped for consistency
and verified by count, not for a wall-clock gain — do not claim one.**

⚠ **TTFT is the wrong instrument for prefill work.** It reads ~0.02 s for a
repeated prompt however slow prefill is, because the **prefix cache** serves it
and prefill never runs; and even with unique prompts it carries tokenization,
scheduling and HTTP, and disagreed with the forward timer on direction.
Measure `PROF seq_len=N`, make prompts unique, and filter to ONE chunk size —
a 600-token prompt arrives as 118- and 128-token chunks, which are not
comparable to each other.

### What a change must keep

- **Read the kernel before calling `alloc_fully_overwritten`.** The bound it
  needs is "assigns every element it owns". A kernel that leaves gaps, or one
  later changed to accumulate (`+=`) into `dst`, turns this into garbage in a
  reply rather than a visible failure — the worst shape of bug this file
  records. `out` in the `indexed_moe_forward_*` path is deliberately still
  zeroed for exactly this reason: nobody has read those kernels' coverage.
- **The load-time padded buffers must STAY zeroed.** `PaddedCudaSlice`'s
  padding is never written by any kernel; it exists so the matmul's padded row
  reads land on defined bytes. Those allocations are once per model load, not
  per token, so they cost nothing worth reclaiming.
- **A/B it inside one binary.** `SWARMLLM_ZERO_QMATMUL_BUFFERS=1` restores the
  zero-fill. That is how the effect was attributed, and it is the only way to
  do it without two builds differing in more than one variable.
- **Judge a submission-count change by the COUNT, not the clock.** The count is
  deterministic and the clock on this box spreads 10-18% run to run. Re-run the
  nsys window probe and compare calls/token; a change that does not move the
  count did not do what it claims.

### The value cache is read where it lies — `layers::value_for_matmul` (2026-09-24)

**What it replaced.** Both attention paths ended `att.matmul(&v.contiguous()?)`
— the grouped GQA path and the plain one MHA takes — inherited from upstream
candle's model code, written when candle's matmul took no strided operand at
all. The KV cache is a VIEW: `SeqCache` narrows a buffer reserved for the whole
reply (`kv_budget::positions_to_allocate`, since 2026-09-12) along the sequence
axis, so during decode `v` is strided on EVERY step and the "no-op" copied the
layer's whole V cache every token: one `ucopy_f32` launch, one host→device upload
of its dims and strides (`params_from_layout`), an alloc and a free — plus
memory traffic proportional to the context.

**How it was found.** The plan's ~30 `cuMemcpyHtoDAsync` per token had been read
as FIXED per token from a two-model comparison (30.3 at 22 layers, 31.8 at 28).
nsys cannot attribute a copy on WSL2 (no CPU sampling, stripped binary), so the
vendored candle now counts every host→device copy by `#[track_caller]` location
(`htod` rows of `SWARMLLM_COUNT_KERNELS=1`). First run: 22 of 24 at one candle
line, `copy_strided_src` — exactly one per layer. **The comparison had been
confounded; the attribution was not.** A count that "settles" a question by
differencing two models is weaker than one that names the line.

**What it was measured at** (one `candle-cuda` binary, `SWARMLLM_CONTIGUOUS_V`
0 vs 1, 24 greedy tokens, `examples/kernel_count_ab.sh`):

| model | layers | copies/token | launches/token | reply |
|---|---|---|---|---|
| tinyllama-1.1b (GQA 32/4) | 22 | 24 → 2 | 469 → 447 | identical |
| llama-3.2-3b (GQA 24/8) | 28 | 30 → 2 | 595 → 567 | identical |
| phi-3.5-mini (MHA 32/32) | 32 | 34 → 2 | 583 → 551 | identical |

Each removed launch also removes an alloc and a free. No tok/s claim: the count
is the instrument, and the context-proportional half of the saving only shows at
long context.

**Why it is safe.** A narrow along the sequence axis leaves each matrix's rows
contiguous and the batch axes collapsible (`stride[0] == stride[1] * dims[1]`),
which both matmuls this build runs accept — CUDA's `gemm_config` (`CUBLAS_OP_N`
arm + `stride_a`) and the CPU backend's `ab_skip`. Metal and Accelerate are not
enabled. `matmul_reads_rhs_in_place` mirrors those rules clause for clause and
declines anything else, which is then copied exactly as before — so a layout it
misjudges could only ever cost the old copy, never an error or a wrong value.
It is the same arithmetic over the same operands in the same order; only the
batch stride differs, which is why the replies are byte-identical rather than
close.

**What a change must keep.**
- `attention_over_a_reserved_cache_view_is_bit_identical_to_a_copy` (CPU,
  grouped and MHA, b = 1 and 2) proves the backend ACCEPTS the view as well as
  that the answer is unchanged; `a_reserved_value_cache_reaches_the_matmul_without_a_copy`
  goes red with `SWARMLLM_CONTIGUOUS_V=1` (null control run).
- A new backend feature (Metal, Accelerate, MKL) means re-checking its matmul's
  stride rules against `matmul_reads_rhs_in_place` before enabling it.
- The two copies left per token are the token id (`split/model.rs::token_tensor`)
  and `gather_rows`'s index layout; removing the first needs sampling on the
  device (Stage 5).

### Why this is the same defect the CPU path already found

§ `inference::decode_attn::gqa_decode_attention_cpu` above ends: "The DRAM
floor for the cache read at ~900 KV × 28 layers is ~7 ms/token on this box; the
kernel sits at ~15 — **the remainder is per-layer dispatch, not arithmetic.**"
That was the CPU backend, reached from a different direction. Both backends are
dispatch-bound per layer, and on both the arithmetic is a minority of the
token. Treat "per-layer dispatch" as this project's standing first suspect for
a decode number that will not move.

## Top-k shrinks the candidate set before anything else runs (2026-09-24)

`inference::sampling::sample_among_top_k` is the sampler whenever top-k is on —
which is every request that does not turn it off, since `api::DEFAULT_TOP_K` is
40 on all four surfaces. It selects the `k` largest logits in one pass, then
applies temperature, top-p and the draw to those `k` alone. llama.cpp's chain
has the same shape (`llama_sampler_top_k` shrinks `cur_p->size`; every later
sampler sees the shrunken array).

**What it replaced**: top-k copied the whole vocabulary into `(usize, f32)`
pairs to partition it; top-p then ran a softmax over all 152k entries and a
partial sort of 4096 of them, though only 40 were finite; the draw ran a third
vocabulary-wide `exp` and walk. Greedy computed a vocabulary-wide `exp` into a
buffer nothing read. And every call built a fresh `SamplingContext` with
vocabulary-sized buffers. Nothing had ever measured it, because
`SWARMLLM_PROFILE` brackets the forward pass alone.

**Measured** (min-of-7x40, x86-64-v3 and mimalloc as the release has them,
per token, a fresh context and the logits copy included as production does;
`sampling.rs` compiled standalone against both versions, interleaved runs):

| vocab | default (t 0.7, k 40, p 0.9) | greedy | + frequency penalty, 2000-token history |
|---|---|---|---|
| 32k | 329 → 26 µs | 122 → 43 µs | 349 → 46 µs |
| 128k | 1472 → 95 µs | 541 → 172 µs | 1566 → 157 µs |
| 152k | 1765 → 113 µs | 605 → 206 µs | 1879 → 206 µs |
| 262k | 3587 → 210 µs | 1088 → 384 µs | 3602 → 393 µs |

(Under glibc malloc the old default was ~1.8x worse again — 3205 µs at 152k —
because of the per-call vocabulary-sized context; the new one is unchanged.)
At ~0.5 ms per layer on the GPU, the old default was about a tenth of a
decoded token on 0.5-3B models with 150k+ vocabularies, and the n-gram verify
path paid it once per verified position. On the processor it is ~2% of a 7B
token.

**What a change must keep: the SAME token, not merely the same distribution.**
A masked entry's `exp(-inf - max)` is exactly `+0.0`, and adding `+0.0` to an
f32 sum changes nothing — so summing only the candidates IN INDEX ORDER gives
the vocabulary-wide sum bit for bit, and walking them in index order stops
where the full walk stopped. That is why the candidates are kept sorted by
index, not by value. `a_top_k_sample_is_the_same_token_the_full_vocabulary_path_picks`
compares the two paths token for token over 3,600 draws; it goes red if the
candidates are walked by value, or if temperature is not applied to them.
Two degenerate cases differ, both previously wrong: a uniform draw of exactly
`0.0` returned index 0 even when masked, and the rounding fallback returned
the last vocabulary entry; both now land on a candidate.

⚠ **Greedy is now `argmax` and nothing else**, ~200 µs at 152k. Its
`max_by` returns the LAST of equal maxima; do not swap it for a faster loop
with different tie-breaking without re-running the reply A/B, which compares
greedy replies across releases.

## An adapter meets the model in the model's row order (2026-09-25)

**What it replaced.** Restoring LoRA (FUTURE_WORK #110) first added each
adapter's deltas exactly as PEFT stored them. On TinyLlama the live check
passed every way — replies changed, repeats identical, alpha=0 matched
alpha=16 — and proved nothing: the test adapter was a fresh PEFT init whose
B matrices are all zero. alpha=0 matching alpha=16 is what gave it away.

**The detail research found.** llama.cpp's `convert_hf_to_gguf.py` reorders a
Llama/Mistral checkpoint's q and k rows for its interleaved RoPE
(`LlamaModel.permute`: rows viewed as `[heads, 2, rows/heads/2]`, middle axes
swapped — k over the KV head count). `convert_lora_to_gguf.py` runs the SAME
`modify_tensors` over an adapter, and its `LoraTorchTensor` routes the reshape
and swap onto `B` (the output rows). A GGUF therefore holds q/k in an order a
PEFT adapter does not, and adding the adapter's delta without the same
reordering puts each head's change on the wrong rotary dimensions. GLM-4,
Llama-4 and DeepSeek-2 rotate interleaved too, but their checkpoints are
already in that order (Llama-4 sets `undo_permute = False`) — so "NORM rope"
is NOT the test; "the converter permuted it" is.

**Measured** (`examples/peft_lora_to_gguf.py`, `score_against_reference.py
--lora`, Llama-3.2-3B, a random rank-8 q/k/v/o adapter, 60 greedy tokens):

| reply | vs llama.cpp + adapter (converter's layout) | vs + adapter NOT permuted |
|---|---|---|
| before the fix | 58/60 rank-1, gap 0.357 | 59/60, gap 0.010 |
| after | **60/60, gap 0.000** | 55/60, gap 0.414 |

Both base replies scored 59/60 against plain llama.cpp and were identical
across the two builds. Qwen2.5-Coder-7B (no permutation) with a real rank-16
adapter on all seven projections: base and LoRA replies byte-identical to
llama.cpp's.

**Half an adapter is the same bug.** The split executor hands an adapter to
`LayerVariant::Dense` layers and `FfnVariant::Dense` feed-forwards only;
DeepSeek-2's MLA, Qwen 3.5's layers and every MoE FFN ignore it. `check_fits`
refuses those (and fused `qkv_proj`/`gate_up_proj` adapters are refused at
registration, because `qkv_proj` CONTAINS `v_proj` and would be filed under
it).

**Three gates to the other executor.** The singleton executor (a whole GGUF on
the legacy path) applies no adapter, and three places hand it work. Two asked
`local_executor_serves(model_id)` — an id cannot carry the adapter, and a LoRA
plan is exactly the "one segment, on this node" shape one of them tests for;
the batch gate asked about the FIRST request of a batch grouped by model only.
The third read the bare `model_loaded` flag and would answer any model from the
resident one. Found by review, not by the tests: every test here went through
the split path. The predicate takes the REQUEST now, so a new request-scoped
feature cannot be invisible to it.

**What a change must keep.** The row order stays a required parameter, read
off the loaded model (`SplitModel::lora_qk_row_order`); a new architecture
gets its `QkRowOrder::for_arch` row from what llama.cpp's converter does to
its q/k, not from its RoPE type. Tests:
`llama_q_and_k_rows_are_reordered_as_llama_cpp_reorders_them` (hand-worked
permutation, k over the KV heads), `an_adapter_is_refused_where_it_cannot_all_be_applied`,
`an_adapter_on_fused_projections_is_refused` — each red with its check removed.


## A mixture-of-experts layer keeps its experts quantized

Found 2026-09-25 answering a field report asking for Qwen3-30B-A3B (`qwen3moe`).

**What it replaced.** DeepSeek-2, Llama-4 and Qwen 3.5-MoE each loaded their
routed experts with `ct.tensor(..).dequantize(..)` on the stacked
`ffn_{gate,up,down}_exps` tensors — the WHOLE `[n_experts, rows, cols]` stack in
f32. Experts are nearly all of a MoE model (Qwen3-30B-A3B: ~29B of ~30B
parameters), so the loaded size was ~4 bytes per parameter whatever the file's
quantization: 11 GB on disk, ~116 GB in memory. No real-sized MoE model could
load, and the capacity planner — which sizes by manifest bytes — under-priced
them by ~10x. The forward already ran one expert at a time on only its routed
tokens (`expert_ffn`), so nothing needed the stack.

**What replaced it.** `split_expert_stack` reads the stack on the processor,
cuts it at `rows × cols / block × type_size` bytes per expert (expert-major,
whole blocks because every row is), and builds each slice with candle's
`ggml_file::qtensor_from_ggml` on the target device. `ExpertFfn` holds three
`QMatMul`s; `expert_ffn` runs them with `QMatMul::forward_shared` for gate/up,
as the dense `Mlp` does. This is also how llama.cpp runs experts (`mul_mat_id`
over the quantized stack), so it moves the numerics TOWARD the reference.

**One loader.** `load_moe_ffn` replaced three near-identical per-family blocks
and is what the Qwen2 family (`qwen2moe`, `qwen3moe`) calls per layer from the
dense branch. Qwen2-MoE needed two things llama.cpp hardcodes in
`qwen2moe.cpp`: `norm_w = false` (`moe_renormalizes_by_default`) and a shared
expert scaled by `sigmoid(x · ffn_gate_inp_shexp)` (`MoeFfn::shared_gate_inp`).
`qwen2moe` had been in `supported_list` while the dense branch required
`ffn_down` of every layer — it could never load.

**Verified** against llama-cpp-python 0.3.16 with `logits_reference_probe.rs`
(prefill at every position + decode steps against the cache, one segment or
split): tiny qwen2moe (Q5_0/F16/Q5_1/Q8_0 mixed) 24/24 positions at cosine ≥
0.99994, whole and split; tiny qwen3moe 99 of 100 at median 0.99995, one
position at 0.99778 — the same token each run, a router near-tie (a systematic
error moves every position). Controls: dense Qwen3-1.7B agrees (worst 0.99987);
the same logits against shifted tokens DISAGREE (worst 0.877). Tests:
`an_expert_stack_is_cut_into_quantized_experts_holding_their_own_slice` (dtype
kept, each expert equals its slice exactly), `a_moe_layer_loads_with_every_expert_quantized`.

**Not verified:** a real-sized MoE model on this box; Qwen 3.5-MoE's shared
gate (now applied when the GGUF carries it, as Qwen3-Next does). FUTURE_WORK #114.
Llama-4 IS verified on tiny models (next section); DeepSeek-2 is refused (#116).

### Llama 4 routes like llama.cpp hardcodes it, and normalises Q/K after RoPE (2026-09-25)

llama.cpp's Llama 4 graph reads NO routing key: `build_moe_ffn(..., norm_w =
false, ..., LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)`, selects on the raw logits
(the same top-k, sigmoid being monotonic and there being no bias), and — the
part a generic MoE gets wrong — multiplies each chosen expert's INPUT by its
weight (`weight_before_ffn = arch == LLM_ARCH_LLAMA4`). We applied softmax +
renormalize to the OUTPUT, which on the top-1 Scout weighs every expert 1.0.
The loader now forces `MoeRoutingConfig { Sigmoid, no renorm, weight_before_ffn }`
for Llama 4 whatever the GGUF says.

The same check found the second difference: on every RoPE layer llama.cpp
applies `ggml_rms_norm` (no weight, `f_norm_rms_eps`) to Q and K AFTER RoPE —
`use_kq_norm`, true unless the model is the 128-expert Maverick. It lives in
`LayerWeights::apply_rotary_emb`, the one place every attention path ropes Q
and K, as `qk_rms_norm_after_rope`.

**Verified** with `examples/make_tiny_llama4_gguf.py` (4 layers, layer 3 NoPE,
4 experts + a shared one, SPM vocabulary) through `logits_reference_probe` +
`compare_logits_reference.py` against llama-cpp-python 0.3.16:

| fixture (unquantized) | before | after |
|---|---|---|
| top-1, whole | median cos 0.40 (Q8), top-1 2/24 | ≥ 0.999998, 24/24 |
| top-2, whole | — | ≥ 0.999996, 24/24 |
| top-2, split at layer 2 | — | ≥ 0.999996, 24/24 |
| all-dense (attention only) | 0.60 median | 0.999999 |

How it was found, which is the part worth keeping: the first "fix" changed
nothing measurable, and comparing the all-dense fixture at `expert_count` 4 and
128 (identical weights; only the Q/K-norm switch differs) showed llama.cpp's
logits move while ours were bit-identical — the new field had never been set,
because an edit script matched `is_nope` against a captured `is_nope,`.
**Toggle the reference's switch and check that YOUR output moves too.** And
judge on UNQUANTIZED weights: with Q8_0 both engines quantize activations and
agree only to ~0.9994, and a top-1 router near-tie at one position then
propagates through the KV cache and looks like a bug.

Not implemented, both past 8,191 tokens: `attn_temperature_tuning` (Q scaled on
NoPE layers by `log(floor((pos + 1) / 8192) + 1) * 0.1 + 1`) and chunked
attention in 8,192-token blocks on three of every four layers.

### Routing is one host copy per layer, and the scatter is one add per expert (2026-09-25)

`MoeFfn::forward` routed token by token: per token a `get` of the scores row, a
device→host copy, two host→device copies (indices, weights) and, on the default
policy, a five-op softmax on the device — then scattered per token AND chosen
expert with a `narrow` and an add, and finished with a `cat` over every token.
On a card each of those is a launch and the copies are syncs, so a 1000-token
prompt on Qwen3-30B-A3B (48 layers, top-8) paid on the order of ten thousand
small submissions and ~3000 syncs per layer.

Now: ONE `to_vec1` of the layer's `[tokens, experts]` scores, `topk_host` per row
on the host (the default policy's softmax over the k picks done the way
`candle_nn::ops::softmax` does it — max, `exp(x - max)`, in-order sum, divide),
and ONE `index_add` per chosen expert into a zeroed `[tokens, hidden]`
accumulator. The per-token accumulation order is unchanged (experts in index
order from zero, and `0 + x == x`), so it is the same sum.

**Verified bit-identical**, not just close: the unit test
`the_batched_moe_forward_is_bit_identical_to_routing_token_by_token` keeps the
old algorithm verbatim as its reference and compares `to_bits` for all four
routing policies, k ∈ {1,2,3}, a 37-token batch and a single decode token; and
`logits_reference_probe` on the tiny qwen3moe (whole and split) and qwen2moe
fixtures wrote BYTE-IDENTICAL logit files before and after (60.7 MB each, `cmp`).
Their agreement with llama.cpp is therefore exactly as recorded above.

**Not measured:** wall-clock on a card — no real MoE model here and no CUDA build
of this change at the time. The claim is the operation count, which is exact.
llama.cpp keeps routing ON the device (`argsort_top_k` + `mul_mat_id`); a device
top-k is the next step if one host copy per layer ever shows up in a profile.

## A card/processor split is placed in every per-layer loop (2026-09-25)

When a model does not fit the card whole, `ModelProcessPool::partial_gpu_layers`
chooses n, the worker is started with `--gpu-layers n`, and the loader puts the
segment's first n layers on the card and the rest on the processor. Three things
must agree about n, and on 2026-09-25 each disagreed once:

1. **The KV budget** charged the whole segment's weights and cache to the card
   (#104: GLM-4-9B at 36/40 was told it had 15 MB, 136 tokens, and refused every
   request). `hybrid::layers_on_device` is now the one count, read by the budget
   and by the placement.
2. **The loader's parallel whole-file path** (`parallel_data`, a thread per
   layer over the mmap) took the segment's device for every layer, so a split
   loaded whole on the card there while `placement.devices()` and the budget
   counted n. The other four per-layer loops shadow `device`/`cos`/`sin` from
   `placement.device_for`/`rope_for` on their first line (gotcha #714); the
   parallel one now does too. Verified on an RTX 3070 with
   `swarmllm test-split --gpu-layers N` on Llama-3.2-3B Q4_K_M: the card grew
   +779 / +1387 / +2123 MiB for 4 / 14 / 28 layers, and all three runs picked
   llama.cpp's first choice at every one of 40 tokens (`score_ids.py`, worst gap
   0.00000). Only a single-file model or a legacy `model.gguf`/`source_path`
   reaches this path, which is why no rig saw it; a reviewer reading the loader
   did.
3. **The pool** offered a split for architectures the loader refuses to split
   (`arch_supports_hybrid`: Qwen 3.5 builds its RoPE outside the loop,
   DeepSeek-2 is unverified MLA), and the loader then loaded them whole on a card
   they do not fit. `split_for_card` reads the same predicate through
   `VramFootprintInputs::splits_across_devices`; such a model goes to the
   processor. Test: `a_model_the_loader_cannot_split_is_not_offered_a_split`
   (with a splittable control).

**Guard:** `every_layer_loop_in_the_loader_places_its_layer` finds the per-layer
loops (both the `for layer_idx in layer_start..layer_end` and the
`(layer_start..layer_end).map(|layer_idx| …)` form), requires at least five, and
requires `placement.device_for(layer_idx)` in the opening 20 lines of each. Red
with the parallel fix undone, naming that loop.

## One prefix-cache snapshot is sized by bytes, and keeps the opening (2026-09-26)

`PrefixCache::positions_ceiling` is the most positions one snapshot keeps: the
explicit `prefix_cache_max_prompt_tokens` when set, never more than
`prefix_cache_max_mb / bytes_per_position` — the latter read off the live KV's
views (`bytes_per_position`, no copy). **A longer prompt keeps its OPENING** —
the system prompt and tools every later turn repeats — rather than nothing.

It replaced a fixed 8,192-token ceiling that REFUSED every longer prompt
outright. Agent harnesses send 4,100-14,400 tokens a turn (PicoClaw ~4.1K,
nanobot 7.3-9.2K, Hermes ~9.3K, ZeroClaw ~14.4K — field report 2026-09-26), so
by default most agent prompts were never cached. The constant protected no
memory: a snapshot is also held to the room beside the live cache
(`snapshot_positions_that_fit`, gotcha #440) and admission evicts cached prompts
before refusing a request. Default now 0 (no token ceiling). What a position
weighs varies ~3x between 3B models (73.7 KB qwen2.5-3b, 229 KB llama-3.2-3b),
which is why the ceiling is derived, not raised.

Sizing BEFORE the copy also retired "keep one entry over budget": that existed
because the whole prompt had already been copied, and discarding it wasted the
copy. Tests: `a_prompt_over_the_token_ceiling_keeps_its_opening`,
`with_no_token_ceiling_the_byte_budget_sizes_the_snapshot`,
`a_budget_smaller_than_one_prompt_keeps_the_opening_that_fits`.
