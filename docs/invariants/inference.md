# Inference kernels, caches and the tokenizer

The evidence behind the rules in `.claude/rules/architecture.md`: what each
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
