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
  the same predicate `SplitModel::causal_mask` builds by hand. The dispatch
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

### Q2_K and Q3_K have several-row kernels too, and every one is priced in one binary (2026-10-11)

A Q2_K mixture-of-experts file keeps each expert's gate and up in Q2_K and its down
in Q3_K, and since `matmul_many` (FUTURE_WORK #138) a prompt hands each expert
~8-40 rows a call — so those two types, which went through the per-row loop, carry
the processor share of such a prompt. `dot_q2k_q8k_rows::<4>` and
`dot_q3k_q8k_rows::<4>` (avx.rs) follow the Q6_K kernel's shape: per block the
column side — scales, mins, the 2-bit (and for Q3_K the high-bit) planes — is
unpacked once; each row keeps its own `acc` and `sumi`, applies its `dmin`/mins
term and then its 16-lane products in the single-row kernel's order. The integer
partial sums are regrouped only where i32 wrapping addition makes that exact.

**A/B inside one binary**: `SWARMLLM_QMATMUL_ROWS=per-row` sends every several-row
product through `vec_dot_rows_generic` (`k_quants::vec_dot_rows_chosen`, the one
call both `matmul` and `matmul_rows` make), so the kernels' gain — never written
down for Q4_K/Q6_K since August — is measured end to end. Tests:
`multi_row_kernels_are_bit_identical_to_the_per_row_ordering` (Q4_K, Q6_K, Q2_K,
Q3_K; red when a Q2_K row adds its mins term after its products, 263 of 384, and
when Q3_K reads the wrong high bit, 384 of 384); `examples/qmatmul_bench` asserts
exact equality for all four and prices each against the per-row loop
(`--rows-only` stops after them).

**Measured** (Ryzen 7 5800H, live node stopped, safety kit; `~/swarmllm-138/rows_chunk.sh`):

In process (`examples/qmatmul_bench --rows-only`): the several-row kernel's time per row
against the per-row loop on the same weights, by rows a call (m):

| weight | m = 4 | 8 | 32-43 | 128 | 896 |
|---|---|---|---|---|---|
| attention projection, Q4_K 3072 × 3072 | 1.16× | 1.38× | 1.34× | 1.34× | 1.38× |
| FFN up, Q4_K 3072 → 8192 | 1.25× | 1.29× | 1.37× | 1.39× | 1.42× |
| FFN down, Q6_K 8192 → 3072 | 1.17× | 1.20× | 1.35× | 1.39× | 1.36× |
| expert gate, Q2_K 2048 → 768 | 1.04× | 1.16× | 1.18× | 1.06× | — |
| expert down, Q3_K 768 → 2048 | 0.95× | 1.11× | 1.19× | 1.25× | — |

At m = 1 both are the single-row kernel (0.8-1.9× is timer noise on a 0.2-0.6 ms call).

End to end, ABBA in one binary (`test-split --prompt-chunk 128`, as a node reads a prompt
off the card; `SWARMLLM_QMATMUL_ROWS=per-row` turns off ALL FOUR kernels, so only the
experts' line is the Q2_K/Q3_K change alone):

| shape | default | per-row | stage that moved |
|---|---|---|---|
| Qwen3-30B-A3B Q2_K, 681 tokens, split 26/22 | 10.1-10.5 s | 11.4-11.5 s | experts on the processor 4.2-4.4 → 4.8-4.9 s |
| the same, every layer on the processor | 14.8-15.1 s | 18.0-18.1 s | experts 8.0-8.2 → 10.0-10.2 s |
| Llama-3.2-3B Q4_K_M, 699 tokens, processor | 10.5 s | 14.2-14.4 s | FFN 6.5 → 9.2 s (Q4_K/Q6_K, first written down) |

Replies identical in every arm.

### A 2-3 row tail shares one pass too — it is a speculative check's whole call (2026-10-14)

The several-row kernels take R = 4 rows a pass and used to hand the 1-3 rows left over to
the single-row kernel one row each. At a prompt's 128 rows that tail is noise; for a
speculative check it is the WHOLE call — 1-2 guesses are a 2-3 row matmul — so a check
shared nothing, and **a forward of 3 positions cost more than one of 4** (257 against
223 ms, below). On a processor tail each extra position cost ~45-60% of a whole forward,
and the guess-length controller (`dsd_controller::best_gamma_for_check`) rightly chose
zero guesses against it (Proxmox, xlam-2-3b: 59 / 86 / 115 ms for 1 / 2 / 3 positions).
llama.cpp pays the same on x86 (the table) — its Q4_K `vec_dot` takes one row there
(`ggml-cpu.c` type traits: `nrows = 2` only under `__ARM_FEATURE_MATMUL_INT8`;
`arch/x86/quants.c` asserts `nrc == 1`) — so this was parity, not a regression, and is now
a lead.

`avx.rs::tail_in_one_pass` sends a tail of 2 or 3 rows through ONE pass of
`dot_*_q8k_rows::<2|3>` for all four types; a lone row keeps the single-row kernel. The
`R`-row kernels are generic in `R` and every row keeps its own accumulators, so the
result is bit-identical (`multi_row_kernels_are_bit_identical_to_the_per_row_ordering`
covers m = 2, 3, 5, 7, 130; `examples/qmatmul_bench` asserts m = 2, 3, 6, 7 for all four).
**A/B inside one binary**: `SWARMLLM_QMATMUL_TAIL=per-row` = the old tail.

**Measured** (Ryzen 7 5800H, 4 threads, live node stopped, safety kit, 0 driver events;
Qwen2.5-Coder-7B Q4_K_M on the processor, `test-split --prompt-chunk N` over a ~250-token
prompt, the forward's median over every chunk past the third; llama.cpp `llama-bench -t 4
-p N -n 0 -d 320`; scratchpad `multipos.sh` then `ab_tail.sh`, ABBA):

| positions | tail in one pass | old tail | llama.cpp |
|---|---|---|---|
| 1 | 134-139 ms | (same kernel) | 131-168 ms |
| 2 | **151-156 ms** | 190-191 ms | 191-206 ms |
| 3 | **173-177 ms** | 248-257 ms | 260-274 ms |
| 4 | 223-227 ms | (no tail) | 319-329 ms |
| 8 | 413-415 ms | (no tail) | 591-659 ms |

An extra position now costs ~20 ms where it cost ~60; replies identical in every arm. **From
three rows the product is compute-bound**, so the 3 → 4 step (+48 ms) is arithmetic, not a
tail: the profile's FFN up + gate is 62.5 / 72.6 / 80.5 / 104.7 / 218.6 ms at 1 / 2 / 3 / 4 / 8
positions — memory-bound at one row, ~26 ms a row from three on (AVX2 `maddubs` + `madd`;
a VNNI `dpbusd` is the next lever, on processors that have it). Two to three positions are
where a check is nearly free. In
process (`qmatmul_bench --rows-only`) the gain over the per-row loop at m = 2 / 3 is 1.20 /
1.38× (attention Q4_K), 1.32 / 1.47× (FFN up Q4_K), 1.28 / 1.33× (FFN down Q6_K). The
small Q2_K/Q3_K expert shapes (768 × 2048) are dispatch-bound at m = 2-3 either way
(0.15 ms a call against 0.04 at m = 1) — a mixture-of-experts check goes through
`matmul_many` instead.

## Idle rayon workers spin 200 µs before they sleep — the vendored `rayon-core` (2026-10-14)

**The rule**: every rayon pool in the binary runs on `vendor/rayon-core` (1.13.0 + one patch,
`sleep/mod.rs::spin_budget`): an idle worker keeps looking for work for `SPIN_US_DEFAULT` = 200 µs
before it announces itself sleepy. `SWARMLLM_RAYON_SPIN_US=0` is upstream (32 `yield_now` rounds,
microseconds). Guard: `rayon_core_is_the_vendored_copy_that_spins_before_sleeping` (a `rayon` that
needs a newer `rayon-core` leaves a `[patch]` unused with only a warning) + its planted-violation
`the_rayon_core_guard_sees_a_registry_copy`.

**What it replaced.** A decoded token runs several parallel regions a layer — RoPE twice
(candle-nn's CPU `rope`/`rope_i` are `par_chunks` over heads), attention, the Q/K/V and output
projections, the FFN — with microseconds of serial work between them. Upstream's idle loop puts a
worker to sleep within that gap, so the next region pays a condvar wake per worker, and on WSL2's
Hyper-V a wake is expensive. FUTURE_WORK #137's stage profile (Qwen2.5-Coder-7B Q4_K_M, processor,
256 tokens) found the small stages growing from 4 threads to 8 — RoPE ×13, attention ×5.7, QKV ×2.2 —
while the big FFN products grew 9-19%. Refuted on the way: the rig's own CPU cap (an uncapped run
is no faster, 0 throttled periods) and SMT placement (one thread per core changes nothing).

**The mechanism, measured in isolation** (scratch `wakeprobe`, rayon 1.12 / rayon-core 1.13, live
node stopped): a region of 28 chunks × 128 floats (one layer's RoPE; 0.75 µs serially) after a
serial gap —

| pool width | gap 0 | gap 20 µs | gap 100 µs | gap 1000 µs |
|---|---|---|---|---|
| 4, upstream | 4.2 | 5.2 (p90 65) | 82 | 62 |
| 8, upstream | 5.5 | 120-146 (p90 415-434) | 229-232 | 267-272 |
| 16, upstream | 8.1 | 872-887 | 900-911 | 888-896 |
| 4, spin 200 µs | 4.1 | 4.4 | 5.3 | 63 |
| 8, spin 200 µs | 5.4 | 5.6 | 6.2 | 179 |

**End to end** (`test-split --gpu-layers 0 --max-tokens 256`, ABBA, each arm in its own uncapped
scope, scratch `t48s.sh`): ms a token, median of 251 forwards —

| | upstream | spin 200 µs |
|---|---|---|
| 4 threads | 126, 127 | **121, 122** |
| 8 threads | 172, 171 | **123, 127** |

RoPE at 8 threads 8.9 → 0.6 ms a token, attention 9.1 → 1.7, QKV 21.9 → 8.5. The replies are
identical in all eight runs. The processor time a run used went DOWN (cgroup `usage_usec`: 4 threads
155.6 / 145.6 → 146.3 / 146.3 s, 8 threads 297.4 / 292.6 → 264.1 / 270.2 s): a sleeping worker's wake
costs more than the spin it replaces, and a worker idle past 200 µs still sleeps, so an idle node
burns nothing. llama.cpp solves the same problem by keeping its workers spinning at a barrier between
the ops of one graph (`ggml_barrier`); rayon's maintainers describe the small-job overhead in
rayon-rs/rayon#614.

**What a change must keep**: the spin is a TIME budget measured from when the worker went idle, reset
on a full wake (`IdleState::wake_fully`), never on a partial one (a spurious wake goes straight back
to the sleepy protocol, as upstream). Re-measure with `SWARMLLM_RAYON_SPIN_US=0` as the A/B before
changing the default; 8 threads still do not beat 4 on this box (llama.cpp's 8 do: 110 against 119),
which is #137's residual.

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

### It read the cache once per QUERY head, not once per group (fixed 2026-09-26, #119)

The doc comment said a group's K and V were read once for all its query heads;
the loop ran `for r in 0..n_rep` OUTSIDE the scan over positions, so each query
head re-streamed the whole K plane and then the whole V plane — `n_rep` times the
traffic (3x on llama-3.2-3b, 4x on Llama-3.1-8B, 7x on Qwen2.5-7B, 8x on
TinyLlama). A per-head plane is ~1 MB at 2K positions against a 512 KB L2, so the
repeats came from L3 or DRAM. That is what #119 measured as decode slowing with
context ~2.5x faster than llama.cpp's while the matmuls stayed flat.

**Now**: one task per (batch, kv head, [`CHUNK`] = 256 positions). A task reads
each K row once and dots it with every query row of the group, then reads each V
row once and accumulates every query row's output; chunks are merged by
flash-decoding's reduction (`out = Σ e^(m_c−M) o_c / Σ e^(m_c−M) l_c`). The chunk
size is a CONSTANT so the summation order — and the result — does not depend on
the machine's thread count. A chunk fully masked for a head keeps max `-inf` and
is skipped by the merge (`-inf − -inf` is NaN); every position masked leaves
zeros where the reference would give NaN, which a decode position (it always
sees itself) cannot reach. With one chunk the merge weight is exactly 1.

**Measured** (A/B inside one binary via a temporary switch, `examples/prefill_bench`,
live node stopped, min of 2 interleaved, `SWARMLLM_DECODE_THREADS=4` — the width
calibration picks on the Ryzen):

| | old | new |
|---|---|---|
| llama-3.2-3b, ~544 cached | 62.2 ms/token | 61.5 |
| llama-3.2-3b, ~2,080 cached | 81.6 | **70.7** |
| Qwen2.5-7B, ~2,080 cached | 141.6 | **134.1** |

Growth from 544 to 2,080 cached: +19.1 → +8.2 ms (llama.cpp +7.3). ⚠ At 8 decode
threads every figure roughly doubled and swung ±10 ms between rounds, and the
Qwen A/B read level (209.3 / 208.2) — width noise swamps the kernel there; take
decode A/Bs at the width a node actually runs.

**Prior art**: llama.cpp's own CPU single-token flash-attention path had the same
shape — "every KV row is loaded and converted from f16 once per q head sharing
it" — and PrismML-Eng/llama.cpp PR #254 fixed it the same way (a tile of 16 KV
rows shared by the group's heads, online softmax per tile): +15-30% at 8-16K
context. The research changed nothing in the design; it confirmed the diagnosis.

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
capped only ever downward. `SWARMLLM_DECODE_THREADS` overrides, and `=0`
restores the single-pool behaviour for A/B measurement inside one binary — the
same discipline as `SWARMLLM_FORCE_STANDARD_ATTN`.
**Decode runs INSIDE a pool even at the global pool's own width** (2026-10-11,
FUTURE_WORK #137). This text used to say the common path "builds no pool and
pays nothing", and the common path was the slow one: every contribution level
is at or below physical, so at the default decode's width IS the global pool's,
`pool_for` returned `None`, and the forward ran as `f()` on the worker's tokio
thread. From a thread outside every pool rayon-core injects each parallel call
as a job and blocks the caller (`Registry::in_worker_cold`) — a few hundred
handoffs a token. Same binary and 4 threads, the only change the global pool's
width (4 = the node's case vs 10 = installed), mean ms per token after a 60 s
cool-down, two passes ABBA (`~/swarmllm-137/pool.sh`): llama-3.2-3b
processor-only 110.9/112.2 → 83.1/80.5 (1.36x); Qwen2.5-14B with 18/48 layers
on the processor 180.6/189.5 → 159.8/160.0 (1.16x); the 3B whole on the card
(null: no processor layers, no pool either way) 3.7/4.2 vs 4.2/4.0.
**No benchmark could see it**: `prefill_bench` and `test-split` run under the
default global pool (every logical core), wider than decode, so decode was
always installed there; only a worker started with `RAYON_NUM_THREADS` at the
contribution width ran outside — and the cooled node measured 5.3-5.4 tok/s on
that 14B where `test-split` measured 6.3-6.6. Through a real node, one binary,
ABBA: the split 5.64/5.93 → 6.79/6.93 tok/s (llama.cpp: 6.8-7.0), the 3B on a
processor-only node 10.97/13.04 → 14.53/15.29. `SWARMLLM_DECODE_INSTALL=0`
restores the old path. Guard: `decode_at_the_global_pools_width_still_runs_inside_a_pool`.
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

(2026-08-10; on a card superseded 2026-10-09 by the half cache — § "A card keeps
its KV cache as f16" — and kept for the processor and the geometries it does not
cover) — one layer's KV cache:
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
were read as worst under long context and GQA, so the f32 copy stayed — a
reading corrected on 2026-10-09 (the paper compares cache-on with cache-off, both
f16), when a card moved to the half cache (§ "A card keeps its KV cache as f16").
**Three things a new caller must respect.** (1) The mirror is GQA-only —
`layers::kv_storage` (was `model_wants_kv_mirror`) gives `F32Mirrored` only to a
GQA model, because MHA decode reads the f32
cache and an unread mirror cost 3-8% per token plus 50% more KV memory
(measured on phi-3.5). (2) A mirror is only ever turned OFF on a cache that
holds positions (`LayerKv::conform_to`, was `set_mirror_wanted`): a
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

**From the rules file (moved 2026-10-02):**

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
  `test_llama4_arch_supported`, `test_deepseek_arch_supported` (since deleted: DeepSeek-2 is
  now refused), `model_arch_properties`), each written against the function rather than the
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

**From the rules file (moved 2026-10-02):**

**`tokenizer::declared_special`** (CONTROL / USER_DEFINED in `token_type`)
decides what is matched whole, on both encoder paths — the `<…>` shape missed
GLM-4's `[gMASK]` and Mistral's `[INST]` (#97).
**`gguf_meta::add_bos_by_llama_cpp_rules`** decides whether a prompt gets a
BOS, and **`SplitTokenizer::encode`** gives none to a text already opening with
one. Check a tokenizer change with `examples/tokenizer_reference.py` +
`tokenizer_agrees_with_llama_cpp` — an independent implementation, not itself.
⚠ **For SentencePiece WHITESPACE llama.cpp is no authority** (it agreed with
HF in 3/7, 5/8, 3/8 on Mistral, Phi-3.5, TinyLlama) — re-reference with
`examples/tokenizer_hf_reference.py` against the model's own tokenizer.json.
**`gguf_meta::special_token_spacing_for`** decides by architecture what the
GGUF does not record: Phi-3 strips whitespace after its turn markers.

→ `docs/invariants/inference.md`

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
`api::submit_to_router`'s drop guard (every non-streaming caller — the guard
lives in the helper, not the callers), by both SSE surfaces on
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

**The non-streaming half recurred in the SETTING (2026-10-10).** The guard was
a drop guard armed by two callers of `submit_to_router` — the OpenAI handler's
fallback router branch and the Anthropic handler — and not by the OpenAI
branch for a model split across computers (`dispatch_inference`), which a
split request takes first. Found in FUTURE_WORK #220's sanitizer rig: the rig's
client gave up on a 7B split reply at 900 s, A's log never printed `client
disconnected before completion`, and the reply was still on the card 25 s
later when the next request's prompt pass began there. Hyper 1.11.1 does drop
the handler on an EOF mid-request (`proto/h1/conn.rs::mid_message_detect_eof`,
half-close off), so the signal was there; the path did not listen. The guard
now lives inside `submit_to_router` (`api::CancelOnDisconnect`) and makes the
flag when a request has none, so a new non-streaming surface cannot skip it —
`a_caller_that_stops_waiting_cancels_the_request` drops the helper's future
with a flagless request, as that path sent. vLLM does the same at one point
per route (`entrypoints::utils::with_cancellation`).

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
Consumed by BOTH sidecar writers (`daemon::manifest::extract_sidecar_tensors`,
`huggingface::probe::download_sidecar_tensors`) AND the reader
(`shard_reader::resolve_sidecars` → `ShardReader`) — since 2026-09-27 through
`GgufTensorMeta::sidecar_tensors`, the one list of shard-0 tensors a node
without shard 0 needs (the tied head and the RoPE factors, #124). Producer and
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
The sidecars are a REQUIRED parameter of `ShardReader::new` (`sidecars`,
once `tied_output`) with no convenience wrapper — for three releases the sidecar had three writers and
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

| Vocab | `PieceDecoder::decode` does | Symptom |
|---|---|---|
| SentencePiece (Phi-3.5 GGUF) | a `<…>` token not declared text → empty (`<|end|>` is CONTROL) | marker invisible, silent run-on |
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

**From the rules file (moved 2026-10-02):**

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
(Since v0.3.213 the own stream is ON by default again — `SWARMLLM_CUDA_OWN_STREAM=0` is
the legacy stream — after the gate described below.)

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
(Since v0.3.213 the default HAS been flipped, after that gate: own stream on, `SWARMLLM_CUDA_OWN_STREAM=0` is the legacy stream.)

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

**From the rules file (moved 2026-10-02):**

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
- **`layers::value_for_matmul` — the KV cache is read where it lies.** The
  cache is a strided VIEW of a reserved buffer on every decode step, so a
  `.contiguous()` on it is a full copy per layer per token (one per layer:
  34 → 2 host→device copies on phi-3.5-mini, replies byte-identical).
  Never `.contiguous()` a cache view before a matmul; `matmul_reads_rhs_in_place`
  is the backends' rule. **The `htod` rows** of `SWARMLLM_COUNT_KERNELS=1` name
  the source line of every host→device copy — nsys here cannot.
- ⚠ **TTFT cannot measure prefill work**: a repeated prompt is served from the
  PREFIX CACHE and reads ~0.02 s regardless. Use `PROF seq_len=N` with unique
  prompts, filtered to ONE chunk size.
- **`SWARMLLM_COUNT_KERNELS=1` prints launches by kernel name per forward**,
  which is how the per-layer mix is known at all (nsys has no GPU kernel table
  on WSL2). Never benchmark with it on — it takes a mutex per launch.
- ⚠⚠ **ONE CUDA STREAM PER DEVICE, and buffers never cross devices** — that is
  what the disabled per-allocation events rest on. **Giving one device a second
  stream means re-enabling them** (`SWARMLLM_CUDA_EVENT_TRACKING=1`) or its
  buffers cross streams unsynchronised — a silently wrong reply, not an error.
  ⛔ **Moving off the legacy stream SHIPPED BROKEN in v0.3.199-alpha**; it is the
  default again since 2026-09-30 (`SWARMLLM_CUDA_OWN_STREAM=0` = legacy), after the
  gate below. Every model emitted garbage in a
  `--features cuda` build while every test and the whole `candle-cuda` A/B
  stayed green — **the cheap gate has no flash-attn and no llama backend**
  (#677, #683). CUDA does refuse graph capture on the legacy stream (measured:
  `cuda_graph_probe.cu` arm A, `cudaError 900`), so this still has to be solved
  before graphs — but **reproduce under `--features cuda` on a real generation**,
  not a unit test.
  ✅ **Cause found and fixed 2026-09-23: vendored flash-attn launched on
  `cudaStream_t stream = 0`**, which a `CU_STREAM_NON_BLOCKING` device stream
  does not synchronise with. `run_mha` now takes the caller's stream
  (upstream's own 0.11.0 fix). ⚠ **Verify "X uses the device's stream" at the
  LAUNCH** — flash took `dev.cuda_stream()` for its pointer guards only
  (#685). Guarded in CI by
  `the_vendored_attention_kernels_launch_on_the_devices_stream`; graph capture
  now needs it, and it is on.
- **Fused kernels are OURS, in `kernels/*.cu`**, compiled to PTX by `build.rs`
  and loaded through candle's `get_or_load_custom_func` — not a fifth vendored
  crate, since `candle-kernels` is a registry dep. **Write each one to be
  bit-identical to the candle ops it replaces** (same expression, same order,
  no `-use_fast_math`), so an A/B moves the submission count and nothing else;
  `examples/kernel_count_ab.sh` checks both halves of that in one run.
  ⚠ **`get_or_load_custom_func` counts launches too** — it must, or a fusion
  reads as removing two launches where it removed one.
  Two ship: `silu_mul_f32` and `add_rmsnorm_f32` (2026-09-23, −2 launches per
  layer). **A residual add is never written at a layer's end** — the sum is
  carried as `inference::residual_norm::Residual::Pending` and resolved at the
  next norm by `Residual::add_norm`; take it with `into_tensor()` only where a
  plain tensor is required (device move, capture, a non-final segment's
  output). ⚠ cudarc 0.19 takes a `CudaViewMut` kernel argument only as
  `&mut` — `&view` does not compile, and only a `--features candle-cuda`
  build sees it.
- **Next moves are ordered in `docs/plans/local_decode_submissions.md`**: fusion
  first, then CUDA graphs (~1.2x batch-1 on an H100, likely more here), with
  stable buffers **no longer known to be a prerequisite** — graph memory nodes
  may give candle's per-op allocations fixed addresses for free.
  ⚠ **Do not re-derive the ordering from llama.cpp's**: their budget has no
  allocation line (ggml plans one compute buffer), and the "~10% per fusion"
  this file used to quote was their CEILING over all of them, not an estimate
  of each (gotcha #680).
- **Judge such a change by the submission COUNT, not the clock** — the count is
  deterministic, this box spreads 10-18%. `SWARMLLM_ZERO_QMATMUL_BUFFERS=1`
  restores the old behaviour for a one-binary A/B.
- **"Per-layer dispatch" is the standing first suspect for a decode number that
  will not move** — the CPU backend reached the same conclusion independently.

→ `docs/invariants/inference.md` · technique in `docs/DIAGNOSTICS.md` § "Where a
decode token actually goes"

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

**From the rules file (moved 2026-10-02):**

**`lora::QkRowOrder`** is the single answer to "what order are this model's
q/k rows in, against the checkpoint an adapter was trained on" — a REQUIRED
argument of the worker's adapter loader. Llama and Mistral are reordered by
llama.cpp's converter (`LlamaModel.permute`) and so must an adapter's `B` be;
applied in checkpoint order the adapter runs without error and answers
wrongly. **`lora::check_fits`** refuses what the executor would silently skip
(non-`Dense` layers, MoE feed-forwards) — half an adapter is #110 again.
Verify with `examples/peft_lora_to_gguf.py` + `score_against_reference.py
--lora`, on an adapter whose B is NOT zero.

→ `docs/invariants/inference.md` § "An adapter meets the model in the model's row order"


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

**Scored on a real-sized model 2026-10-11**: Qwen3-30B-A3B Q2_K with 26 layers on
the card against llama-cpp-python on the processor, median cosine 0.9994-0.9998
at 40 and 700 tokens (numbers in the stacked-experts section below). Qwen 3.5-MoE's shared
gate is checked on a small real file (#228 (1)). FUTURE_WORK #114.
Llama-4 IS verified on tiny models (next section); DeepSeek-2 is refused (#116).

### The processor's experts run in one region per projection, for any token count (2026-10-11)

**What it replaced.** `experts_one_at_a_time` runs each chosen expert over its
tokens: an `index_select`, two host→tensor copies, `forward_shared` (gate + up),
`silu_mul`, `down`, a `broadcast_mul` — each product its own rayon region. For one
decoded token on Qwen3-30B-A3B that is 8 experts × ~3 regions and ~80 tensor ops
per layer, around matrices of 768 × 2048. For a prompt it is worse in a different
way: a 681-token prompt spreads ~5,400 (token, expert) pairs over 128 experts,
~42 tokens each, and each expert's gate is a region of 768 rows — 12 tasks of 64
for 16 threads, 384 such regions a layer, every one a fork, a join and its tail.

**What llama.cpp does** (`ggml_compute_forward_mul_mat_id`, read at `4b1a27f`):
the activation is quantized once; ONE thread region per projection walks every
chosen expert's rows in chunks (`chunk_size = 64` for one activation row), each
chunk applied to all of that expert's tokens, with a per-expert atomic chunk
counter and NO barrier between experts.

**What replaced it.** For f32 on the processor against quantized experts,
`MoeFfn::experts_in_one_region` takes every token count — first (2026-10-11
morning) ≤ 8 tokens with one activation row per job, the same day any count with
several (FUTURE_WORK #138 (1)). The vendored `quantized::matmul_many` runs many
`MatMulJob`s — a weight, a `QuantizedRows` input and the rows of it this job reads
— in ONE region of 64-weight-row tasks, each task applying its rows to ALL the
job's activation rows (`vec_dot_rows` in blocks of 128 rows, `k_quants::matmul`'s
tiling), output column-major. Per region: the tokens' rows quantized once per
activation type (`QTensor::quantize_rows`) and each expert's copied out of them
(`QuantizedRows::gather` — a block copy, the same bits); every gate and up in one
`matmul_many`, `silu_mul_into` over the column-major outputs, each expert's
activations turned into rows and quantized once for down; every down in a second
`matmul_many`; the pairs summed per output column in expert order. A long prompt
runs `MOE_PAIRS_PER_REGION` (4,096) pairs a region — 512 tokens of a top-8 model —
which bounds the temporaries near 120 MB on Qwen3-30B-A3B; the host loops split
across threads from `MOE_PAIRS_TO_SPLIT_HOST_WORK` (64) pairs, so a decoded token
forks nothing it did not fork before.

**Why it is the same bits.** The same quantized activation (one `from_float` per
row, exactly `k_quants::matmul`'s) and the same `vec_dot` per output —
`vec_dot_rows` is bit-identical to the per-row loop by its contract, and
`matmul` itself uses it above one row; `silu_mul_into` is the kernel `silu_mul`
runs, and with the intermediate width a multiple of 8 every element goes through
its vector lanes either way, column-major or not; `index_add` on the processor
adds rows in listed order from a zeroed copy, which the per-column sum reproduces
(pairs in expert order, from 0.0). Tests:
`one_region_experts_are_bit_identical_to_one_at_a_time` — Q4_K/Q4_K/Q6_K,
Q4_0/Q4_K/Q8_0 (two activation types) and Q2_K/Q2_K/Q3_K; weights after or before
the expert (Llama 4); 1, 3, 8, 37 and 100 tokens, each in one region, 24 pairs a
region and one token a region. Red when the experts are summed in reverse (146 of
256 values) and when the sum reads the wrong column (576 of 768).
`many_row_products_in_one_region_match_forward` — `matmul_many` against `forward`
for 1 to 300 rows, a range and a gathered input, Q4_K/Q6_K (own several-row
kernels) and Q2_K/Q8_0; red when the row-block loop rereads its first block (192
of 12,480, the rows past 128).

**Measured** (Qwen3-30B-A3B-Instruct-2507 Q2_K, RTX 3070 Laptop + Ryzen 7 5800H,
`swarmllm test-split`, one binary, `SWARMLLM_MOE_CPU=per-expert` = old, ABBA,
live node stopped, safety kit; `~/swarmllm-138/cpu_prompt.sh`, `cpu_chunk.sh`):

| shape | old | new |
|---|---|---|
| ≤ 8 tokens (first step): `moe ffn on the processor` per decoded token, 26/22 | 22.1 / 28.3 / 24.9 ms | 19.2 / 16.5 / 17.8 ms |
| 681 tokens in one forward, 26/22: whole prompt (processor experts) | 19.9 / 19.5 s (14.6 / 14.3) | 9.9 / 10.3 s (4.5 / 4.7) |
| the same, every layer on the processor | 39.1 s (30.8) | 18.8 s (10.1) |
| 31-token prompt, 26/22: processor experts | 1.71 / 1.60 s | 0.32 / 0.36 s |
| 681 tokens, 128 a forward — what a node does unless the model is wholly on the card (`prefill_pacer::prompt_chunk_ceiling`), 26/22 | 46.6 / 46.2 s (37.9 / 37.7) | 13.3 / 12.9 s (5.6 / 5.4) |
| the same, every layer on the processor | 94.9 / 93.5 s (85.1 / 84.1) | 20.0 / 19.9 s (10.9 / 11.0) |

The old path was more than twice as slow in 128-token forwards as in one, and that is
the shape a user meets: each forward's experts get ~8 tokens, so its 384 regions a
layer do less work apiece, six times over. The new path reads each expert's weights
once per forward and loses ~30% to the smaller forwards.

Every reply identical across arms (bit-identical, as the tests say) and to the
build before this step.

### On a card the experts stay stacked: routed on the card, one launch per projection (2026-10-11)

**What it replaced.** On a card each chosen expert was its own three `QMatMul`s,
reached through `route`'s `to_vec1` of the router scores — a device→host copy
that waits for everything queued before it — then per expert an `index_select`,
host→device copies of its token list and weights, the products and an
`index_add`. On Qwen3-30B-A3B Q2_K that was ~1.6 ms of host work per layer for
~12 MB of experts a ~400 GB/s card reads in ~30 µs: the card's 26 MoE layers took
40-42 ms of a 77-109 ms step.

**What llama.cpp does** (`build_moe_ffn`, `ggml_cuda_mul_mat_id`): routing on the
device (`argsort_top_k`), the experts kept as the file stacks them, and one
`mul_mat_id` per projection over every chosen (token, expert) pair.

**What replaced it.** `StackedExperts::fit` keeps a layer's three
`[n_experts, rows, cols]` stacks whole on a card when `moe_gemm_gguf` reads
their types (Q8_0, Q2_K-Q6_K), each input width is a multiple of 8 and the
weight applies to the expert's output (not Llama 4); everything else keeps one
matrix per expert. `MoeFfn::experts_stacked`: the router product, then
`topk_on_device` — `arg_sort` and the policy's softmax/sigmoid on the card, the
host form's policies one for one (`routing_on_the_device_agrees_with_routing_on_the_host`)
— the pair ids grouped by expert with `sort_last_dim`, gate and up as one launch
each, `silu_mul`, down as one launch scaled by each pick's weight, and a sum over
the picks. No host copy anywhere, so the card never waits on the processor
inside a layer. candle's arg-sort holds a row in 48 KB of shared memory, so a
prompt runs in chunks of `4096 / k` tokens.

**Two kernels and a race.** `moe_gemm_gguf` has a one-row kernel (a warp per
output row of each pair, the input quantized to q8_1 once — llama.cpp's MMVQ
with ids) and a tensor-core kernel (`moe_wmma_gguf.cu`, f16 tiles per expert —
MMQ's role). The first build sent prompts (> 8 tokens) to the tensor-core
kernel, and its replies DIFFERED RUN TO RUN: three runs of one arm, three
replies, while the per-expert arms gave one. The kernel refills its shared
input tile at the top of each k step with no barrier after the previous step's
`mma_sync`, so a fast warp overwrites the tile a slower one is still reading.
Upstream had already added exactly this barrier to the non-GGUF copy
(`moe_wmma.cu`, "Fix shared memory mismatch on V100"); candle main still lacked
it in this one on 2026-10-11. With the one-row kernel for every token count,
three runs gave one reply — the per-expert path's, byte for byte — which located
the race; the vendored candle-kernels now carries the barrier (one line, guard
`the_vendored_moe_prompt_kernel_keeps_its_barrier` + a planted self-test), and a
prompt takes the tensor-core kernel again — three runs, one reply. That reply
leaves the per-expert one at a near-tie 32 characters in (an f16 prompt against
q8_1), every time. Why not leave prompts on the one-row kernel: it re-reads an
expert's weights for every token. The card's experts over a 681-token prompt,
synced: one-row 5.56 s, tensor-core 1.55 s, per expert 1.18 s.

**The kernel reads storage, not tensors.** `moe_gemm_gguf` passes each operand's
`CudaSlice` pointer and ignores the layout, so a view part-way into a buffer (a
later token chunk of the layer's input) would be read from the buffer's start.
Every operand goes through `layers::own_storage` (a copy only when the view does
not start at its storage), pinned by
`an_operand_handed_to_the_grouped_kernel_starts_at_its_own_storage` — which goes
red when the helper uses candle's `copy()`, which keeps the view's offset.

**A second race the checks found, not ours.** The tiny qwen3moe split into two
segments ON THE CARD in one process answered wrong decode rows in six of eight
runs (one NaN) — stacked AND per-expert alike, the whole model clean every time,
the processor split bit-identical to the whole. Each `Device::new_cuda` has its
own stream with event tracking off, and candle's `transfer_to_device` copied the
hidden state on the destination's stream with nothing ordering it after the
source's writes. Only the logits probe holds two card devices that hand each
other tensors; production segments get their input from the network or the
processor. The vendored `transfer_to_device` now joins both streams (gotcha #828).
**The baseline is what kept this from being blamed on the stacked experts**:
the per-expert control on the same split diverged too.

**Verified** against llama-cpp-python 0.3.16 (`logits_reference_probe`,
`LOGITS_PROBE_GPU_LAYERS=26` for the 30B): tiny qwen3moe Q8_0 whole on the card,
100 positions, median 0.99996, the one known near-tie (position 12) in every arm;
Qwen3-30B-A3B Q2_K, 26 layers on the card, the committed build — 40 tokens median
0.99949, worst 0.9753 at position 1, top-1 29/40 (per expert 0.99952, worst 0.9733
at the same position, 31/40 — synthetic ids, flat distributions); 700 tokens (two
prompt chunks) median 0.99980, worst 0.99818, top-1 19/20 (per expert 0.99978,
0.99811, 19/20). The two-segment card split, re-run with the transfer ordered:
20 of 20 runs agree with the whole model. Tests: the routing agreement above,
`only_a_card_with_types_the_grouped_kernel_reads_keeps_experts_stacked`.

**Measured** (Qwen3-30B-A3B-Instruct-2507 Q2_K, 26 layers on an RTX 3070 Laptop,
22 on a Ryzen 7 5800H, one binary, `SWARMLLM_MOE_CARD=per-expert` = old,
`~/swarmllm-138/{card_check,prompt_check,final_check}.sh`): decode step median
74-89 → 40-55 ms over five per-expert and twelve stacked runs (unsynced); `moe ffn on the
card` 39-52 → 5.4-7.9 ms unsynced, 47.5 → 13.2 ms synced. llama.cpp at the same
26 layers: 21.5-25.4 tok/s, i.e. 39-47 ms. A prompt is the processor's: 681
tokens took 21.5-24.1 s stacked and 23.2-24.8 s per expert, ~17 s of it in the
processor layers' experts.

### A split's processor-held experts read a prompt on the card (2026-10-13, #138 (1))

**What it replaced.** A card/processor split ran its processor layers' experts on
the processor at every token count. Qwen3-30B-A3B Q2_K split 26/22 spent 4.5 s of
an 867-token prompt's 11.0 s there, read 128 tokens a forward as a node reads it.

**What llama.cpp does.** Its scheduler hands an op whose weights sit in host
memory to the card when the batch is ≥ `GGML_OP_OFFLOAD_MIN_BATCH` (32;
`ggml_backend_cuda_device_offload_op`, `ggml-cuda.cu:5615-5634` at `4b1a27f`),
and for `MUL_MAT_ID` copies only the experts the ids name, in contiguous runs
(`ggml-backend.cpp:1693-1770`). It also keeps such weights in PINNED host memory
whenever a card is present (`llama-model.cpp:1090-1104`).

**What replaced it.** The loader marks once, after every layer is built, each
processor-held MoE layer the card kernel can read
(`layers::let_processor_layers_read_prompts_on` →
`MoeFfn::experts_could_stack_on_a_card`: every expert quantized on the
processor, one type and shape per projection, `StackedExperts::fit`'s rules — so
never Llama 4, which weights the experts' input). A forward of ≥
`PROMPT_ON_CARD_MIN_TOKENS` is routed on the processor as before, then
`MoeFfn::processor_experts_on_the_card` writes each token's k picks, copies the
chosen experts into a zeroed `[n_experts, rows, cols]` stack on the card
(`QTensor::stack_some` → vendored `cuda::load_quantized_entries`; an unchosen
slot is never named by an id) and runs `stacked_products` — the same products the
card's own layers run. Any error runs the processor path for that call (a WARN
once per worker): the card out of memory costs time, never the reply.

**Why 512 tokens a forward.** Every forward copies the chosen experts again, and
at top-8 a 128-token chunk already touches nearly all 128 — so 128 a forward is
four copies of what one 512 forward copies. `SplitModel::reads_prompts_on_card`
(through `prompt_reader` since (1e), which reads more than 512 where the card has
room) answers the chunk ceiling for every caller of `prompt_chunk_ceiling` (the slot
table's solo prompt, a segment's prompt pass); `runs_entirely_on_card` keeps its
own questions (the drafter's single-step reads, admission's card timing). The 2026-10-11 measurement of 512
against 128 alone (#138's entry) is why a split without the card path keeps 128.

**Measured** (867 tokens, 26/22, ABBA, one binary, `SWARMLLM_MOE_PROMPT_CARD=0`
= old, Windows up 3.1 h, `~/swarmllm-138/ab.sh`): 128 a forward 11.00 / 11.07 s
old against 12.69 / 12.90 s new; 512 a forward 9.23 / 9.39 s old against
7.80 / 7.92 s new. A node went from the first figure to the last: 1.4x. The new
stage costs ~57-62 ms a layer whatever the tokens (1.25 s for 22 layers at 355
tokens, 1.49 s at 512): ~34 ms of it the copy of ~226 MB, already at this card's
6.7 GB/s link (pageable, one copy per expert, moves 6.5 — pinned staging measured
slower, `~/swarmllm-138b/bench.sh`), the rest the card's expert kernel; #138 (1c).

**Verified** against llama-cpp-python 0.3.16, both arms on one probe binary
(`logits_reference_probe`, `LOGITS_PROBE_GPU_LAYERS=26`, the same 867 tokens read 512
a chunk, the last 64 positions + 4 decode steps kept, `~/swarmllm-138/probe.sh`): new
median cosine 0.99952, worst 0.9653, top-1 65/68; old 0.99955, 0.9683, 64/68 — the
worst at the SAME position in both, a Q2_K router near-tie, not the path. The arms'
logits differ on all 68 rows (max 1.99), so the switch did change the path. The
16-token replies: every arm of a path gives one reply, and the two paths part at
token 12, where the new path's pick is llama.cpp's third (0.105 logits behind) and
the old one's its first; at 128 a forward the old path parts from llama.cpp at token
15 instead (0.176). Tests: `a_stack_of_some_experts_puts_each_at_its_slot`,
`a_prompts_processor_experts_go_to_the_card_only_where_it_can_take_them` (the marking,
`reads_prompts_on_card`, and a refused card leaving the processor's answer bit for bit
— red when the fallback returns the error instead).

### A split's processor-held projections read a prompt on the card (2026-10-13, #138 (1b))

**What it replaced.** (1) gave the card only a MoE layer's experts. Every other
weight a processor layer holds — attention's q/k/v/output, a dense FFN's
gate/up/down, a MoE layer's shared expert, a DeltaNet layer's projections — was
multiplied on the processor at every token count. Qwen2.5-14B Q4_K_M split 20/28
read an 888-token prompt in 35.3-36.8 s, a dense model gaining nothing from (1).

**What llama.cpp does.** The same scheduler rule (1) cites is not MoE-specific:
`ggml_backend_sched` pass "1.off" (`ggml-backend.cpp:970-975` at `4b1a27f`) sends
ANY op whose weight sits in a host buffer to a backend whose `offload_op` accepts
it, and the CUDA one accepts every op of ≥ `op_offload_min_batch_size` (32)
rows — `get_op_batch_size` reads `ne[1]` for `MUL_MAT`. `llama-bench -nopo 1`
turns it off: the same 14B split pp512 399 ± 9 tok/s with it, 19.1 without
(`~/swarmllm-138b/dense.sh`). We read at 22 tok/s: llama.cpp without offload.

**What replaced it.** `QMatMul::forward` and `forward_shared` are the one choke
point every projection crosses, so the rule sits there, not in each layer
kind: a `QMatMul` the loader marked (`prompt_card`) that is handed ≥
`PROMPT_ON_CARD_MIN_TOKENS` rows on the processor (`prompt_card_for`, the one
token rule both kinds ask) copies its weight to the card for the call
(`weight_on`: the quantized bytes through `QStorage::from_data`, a fused weight
once for all its slices, the f32 mark (`accumulates_outputs_in_f32`) carried
to the copy), sends the activation, runs the card's own product and brings the
result back (`QMatMul::on_the_card`). Any refusal runs the processor path for
that call (WARN once): it costs time, never the reply. The loader marks once
(`let_processor_layers_read_prompts_on`, through `LayerVariant::projections_mut`
— every projection of every layer kind), and only weights held in HOST memory
(`weight_is_on_the_processor`): a card layer is never given a second copy. A
split whose every processor layer reads its heavy part on the card reads at least 512
tokens a forward (`prompt_reader` → `prompt_chunk_ceiling`; at 128 the
weights cross four times as often; more where the card has room, (1e)).

**Measured** (one binary `1afbfea4`, `SWARMLLM_PROMPT_CARD_PROJECTIONS=0` = old,
Windows up 4.3 h, live node stopped, safety kit, `~/swarmllm-138b/ab.sh`):

| | old | new |
|---|---|---|
| 14B Q4_K_M 20/28, 888 tokens, 512 a forward | 35.63 / 35.29 s | **7.03 / 6.49 s** |
| same, 128 a forward (a node before) | 36.80 s | 11.26 s |
| Qwen3-30B-A3B Q2_K 26/22, 867 tokens, 512 (experts on the card in both) | 8.43 / 8.45 s | **6.13 / 7.17 s** |

What a node did before is the 128 row's left: **36.8 → 6.5-7.0 s, 5.3-5.7x** for
the 14B; 11.0 s (before (1)) → 6.1-7.2 s for the 30B. Still ~2.8x behind
llama.cpp per 512-token forward (3.60 s against ~1.28 s): the profile's
remainder is the processor layers' attention (0.59 s), RoPE/norms/activation
(0.63 s) and 0.50 s unattributed — llama.cpp's scheduler ALSO moves the
weightless ops between two offloaded ones to the card (its "expand" passes), and
each of our projections ships its activation up and its product back (~150 MB
a layer at 512 tokens by arithmetic, about as much as the layer's weights). That
is #138 (1c).

**Verified** against llama-cpp-python 0.3.16, teacher-forced over each arm's
16-token reply (`score_many.py`, CPU): the 14B's new reply is llama.cpp's FIRST
pick at all 16 tokens; the old path at 512 parts at token 10 (llama.cpp's
second, 0.0003 behind) and at 128 at token 13 (second, 0.32). The 30B's new
reply picks llama.cpp's second at token 12 (0.069) where (1) alone picked its
third (0.105); token 8 is the same 0.76 near-tie in every arm. Each arm gave one
reply across its two runs. Test
`a_prompts_processor_projections_read_on_the_card_give_the_processors_answer`
(the CPU standing in for the card: a Q4_K weight, a dense one, a shared pair and
three fused slices at 1, 31, 32 and 100 tokens bit-identical to the plain path;
the marking; red when `on_the_card` hands every weight the first copy).

### A split's processor-held layers read a prompt whole on the card (2026-10-13, #138 (1c))

**What it replaced.** After (1b) every processor-held projection crossed to the
card alone: its activation up, its product back, and attention, RoPE, the norms
and the activation ran on the processor between them. llama.cpp's scheduler
does not stop at the weighted ops: pass 2's "expand gpu down/up"
(`ggml_backend_sched_split_graph`, `ggml-backend.cpp` 1129-1170 at `4b1a27f`)
puts every unassigned op between two offloaded ones on the card — norms, RoPE,
Q·K, softmax, ·V, the GLU — and only the K/V writes stay on the processor,
pinned by the cache's host buffer; pass 5 copies the cache views up
(1399-1420).

**What replaced it.** `split::executor::dense_layer_on_the_card` runs a
processor-held `LayerVariant::Dense` layer of a ≥ 32-token chunk on the card
(`LayerOnTheCard::card_for` the gate: the layer's attention and feed-forward
were marked by (1b)/(1), no adapter, `SWARMLLM_PROMPT_CARD_LAYERS` not `0`).
`LayerOnTheCard::of` copies the small tensors — norms, biases, Q/K norms, the
router — and the RoPE rows `0..index_pos + seq_len`; the projections copy their
own weights as (1b) does, now with the activation already there. **The cache
stays on the processor** (`LayerWeights::attend`'s `cache_home`): the chunk's
new K/V rows go down and are appended there, the history comes up. ⚠ candle's
`to_device` on a narrowed view uploads the WHOLE storage it views, so every
upload of a narrowed tensor (cache history, RoPE rows) goes through
`force_contiguous()` first. Any error cuts the cache back to what it held
(`truncate(held)`) and reruns the layer on the processor (WARN once). The
batched loop keeps (1b): its forwards are decode batches.

**Measured** (one binary, `SWARMLLM_PROMPT_CARD_LAYERS=0` = (1b), ABBA, Windows
up 6.8 h, live node stopped, safety kit, `~/swarmllm-138c/ab.sh`, 512 a forward,
the DIAG line in every new arm, no fallback):

| | (1b) | (1c) |
|---|---|---|
| Qwen2.5-14B Q4_K_M 20/28, 888 tokens | 8.18 / 8.24 s | **5.21 / 5.17 s** |
| Qwen3-30B-A3B Q2_K 26/22, 867 tokens | 6.74 / 7.12 s | **6.21 / 5.22 s** |

The box read (1b) slower than its own morning (6.5-7.0 s for the 14B); the
arms interleave in one binary, so the ratio holds: **1.6x** for the 14B, where
28 of 48 layers are on the processor, 1.2x for the 30B (22 of 48, its experts
already on the card). The 14B's first 512-token forward went 5.12 → 2.83 s.

**Where that forward goes now** (`SWARMLLM_PROFILE_SYNC=1`, `~/swarmllm-138c/sync.sh`,
two synced runs; read for the big stages only): the 28 processor layers' weights
copied to the card take 1.03-1.29 s of a 2.0-2.3 s first forward and 0.78-0.80 s
of the second (4.6 GB: ~4 GB/s cold, ~5.8 warm — the first forward also grows
the card's memory pool); the products of all 48 layers ~0.62-0.66 s; attention
64 ms. ⚠ An UNSYNCHRONISED profile charges the card's queued work to the next
weight copy — a copy from pageable memory waits for the stream first — so it
read this forward as "attention 34 ms, products 80%" and a one-token width probe
as 263 ms (its copy back to the processor waited out the card; removing it
measured level, 3.35 / 3.77 s against 3.65 / 3.37 s). llama.cpp waits the same
way (`ggml_backend_sched_compute_splits`: a host weight takes the synchronous
`ggml_backend_tensor_copy` after `ggml_backend_event_synchronize`, no copy
stream, 4b1a27f) from pinned memory: ~0.75 s of copies + ~0.5 s of products is
its 1.28 s. The copies now run beside the products — next section. ⚠ The cache history crosses up for every chunk: ~8 KB a position a
layer for the 14B at f32, ~2 GB a chunk at 8K tokens of 28 layers, about a
third of the weights it copies — llama.cpp's pass 5 pays the same, and the
processor's attention over that history costs more.

**Verified** against llama-cpp-python 0.3.16, teacher-forced (`score_many.py`):
each arm gave one reply across its two runs, and every pick where the arms
part is a near-tie already named above. The 14B's (1c) reply takes llama.cpp's
second at token 10, 0.0003 behind (the tie the morning's arms split the other
way); (1b)'s is its first at all 16. The 30B's (1c) reply takes llama.cpp's
third at token 12 (0.105, where (1) alone landed), (1b)'s its second (0.069);
token 8 is the same 0.76 tie in both. Test
`a_prompt_chunk_read_whole_on_the_card_gives_the_processors_answer` (the
processor standing in for the card: two 40-token chunks and a decoded token
through the executor bit-identical to the plain path; the layer run directly,
its output and its kept cache, because the executor's fallback would hide an
error; the gate at 31 tokens and with an adapter).

### A split's prompt copies run beside the card's work, through a ring (2026-10-13, #138 (1d))

**What it replaced.** After (1c) a processor-held layer's weights still crossed
to the card on the card's ONE stream, and a copy from pageable memory on a
stream first waits for every kernel queued there — so a 14B split's 512-token
forward copied (0.78-1.29 s) and multiplied (~0.65 s) in turn, never at once.
llama.cpp does the same (a synchronous `ggml_backend_tensor_copy` after
`ggml_backend_event_synchronize`, 4b1a27f). A CUDA micro-bench
(`~/swarmllm-138c/overlap.cu`) showed this card overlaps a pageable copy on a
second stream with a kernel perfectly (200 MB beside a 23 ms kernel: 32.1 ms,
the copy alone), and pinned memory is no faster (6.66 vs 6.54 GB/s).

**What replaced it.** `CudaDevice::clone_htod_beside` (vendored candle) copies
on the device's copy stream into a 256 MB copy ring (`CopyRing`, the device
pool's memory, made at the first copy, given back at the idle trim); the
device's stream waits for that copy, takes a buffer from its own pool, copies
the region into it card-to-card and records that it has read it; the copy
stream waits for that read before it writes the region again
(`CopyRing::place`). The buffer candle gets is the device stream's like any
other. `layers::sent_to_card` and `QMatMul::weight_on` are the only callers; a
copy over 128 MB, one made while a decode step is captured, and the legacy
stream go the old way. A/B `SWARMLLM_PROMPT_CARD_COPY_STREAM=0`.

**Why a ring and not a buffer the copy stream allocates** (gotcha #836). Two
designs failed first, both by the pool's RESERVED memory:

- From the device's pool, the copy freed on the device's stream after its
  product: the stream's own next allocations reuse its frees at once, in
  stream order, so the copy stream never found one — 7,776 MB reserved for a
  4,566 MB peak (the one-stream path reserved 4,640) and a 14B forward at
  3.45 s instead of 1.3 once the driver backed the rest with host memory.
- From a pool of the copy stream's own (`cuMemPoolCreate`), the copy stream
  waiting two layers back (PyTorch FSDP's `limit_all_gathers` rate limiter):
  48 copies of 16 MB took 768 MB, every one fresh.

`~/swarmllm-138c/reuse.cu` (48 × 16 MB, a ~4 ms kernel each, two runs) names
the rule: this driver reuses a free made on ANOTHER stream only once a
synchronize of that stream has observed it — 768 MB whether the allocating
stream waited for the event after the free (`REUSE_FOLLOW_EVENT_DEPENDENCIES`,
the documented route), the host waited for it, or the pool was allowed to
insert waits; 32 MB with a stream synchronize (no overlap), with the free made
on the allocating stream (PyTorch's `cudaMallocAsync` backend frees there), or
with the ring — the copies still under the kernels (252-299 ms against 227-252
overlapped and 421-456 serial). An NVIDIA forum thread (283061) reports the
same growth, unanswered. candle frees a buffer on the stream its tensor drops
on, so only the ring keeps the copy stream's memory its own.

**Measured** (one binary, `SWARMLLM_PROMPT_CARD_COPY_STREAM=0` = before, ABBA,
Windows up 10.8 h, live node stopped, safety kit, `~/swarmllm-138c/ab1e.sh`, no fallback, the DIAG line saying `beside=` as each arm
should):

| | before | the ring |
|---|---|---|
| Qwen2.5-14B Q4_K_M 20/28, 888 tokens, 512 a forward | 3.27 / 3.56 s | **3.15 / 3.07 s** |
| — its second (warm) forward | 1.50 / 1.55 s | 1.49 / 1.43 s |
| — the card pool's reserve (peak in use) | 4,640-4,768 MB (4,571) | 5,024 MB (4,827) |
| the same 14B, 128 a forward (7 forwards) | 8.71 / 8.71 s | **7.91 / 7.90 s** |
| Qwen3-30B-A3B Q2_K 26/22, 867 tokens, 512 a forward | 5.45 / 5.73 s | 5.69 / 5.53 s |
| — the card pool's reserve | 6,368-6,400 MB | 6,624-6,688 MB |

3-9% for the dense 14B (most at 128 a forward, where seven forwards each copy
every processor layer), level for the 30B (most of what it copies is its
experts, stacked by `QTensor::stack_some` for (1), which still go the waiting
way). The ring's 256 MB is the
only memory it adds. The warm forward's profile shows why it is not the
~0.8 s the copies alone would allow: copies 833 ms + unattributed 528 against
1,215 + 248 — the products now wait elsewhere. One reading measured and not
confirmed: this card has ONE async copy engine (`asyncEngineCount`), so the
ring's card-to-card move queues behind the next upload; `reuse.cu` variant 7
(the move as a kernel, off the copy engine) gave 296-298 ms against variant 6's
282-307 — no clear gain, so the memcpy stays.

**What it showed next.** Each forward re-copies every processor-held weight
(4.6 GB for the 14B), so FEWER forwards win more than overlap: the 14B's prompt
×3 (2,664 tokens) took 10.05 s at 512 a forward, 6.93 s at 1,024 and 5.81 s
whole, the pool's peak 4,989 / 5,097 / 5,461 MB — ~0.28-0.36 MB a token of
working set. That is FUTURE_WORK #138's next item, (1e) below.

**Verified.** `a_splits_prompt_copies_arrive_intact_and_hold_a_bounded_share_of_the_card`
(card only): 48 copies of 16 MB, three times round the ring, each multiplied
eight times so the copies run ahead — every one arrives as its own layer's,
and the card's pool grows 320 MB (the ring + 4 weights), 3 of 3; with the
ring's wait removed it fails 3 of 3 (layer 4 or 5 arrives as another's).
Replies: each pair gave the same 16 tokens at the same chunk size, all four
pairs (the ring changes when a weight arrives, never what it is).

### A split's prompt chunk is as large as the card has room for (2026-10-13, #138 (1e))

**What it replaced.** A card/processor split whose processor-held layers read a
prompt on the card read it 512 tokens a forward, whatever the card held — and
every forward re-copies every processor-held weight (4.6 GB for the 14B 20/28).
The (1d) rig read the 14B's prompt ×3 (2,664 tokens) in 10.05 s at 512 a
forward, 6.93 s at 1,024 and 5.81 s whole, the pool's peak 4,989 / 5,097 /
5,461 MB. llama.cpp fixes its chunk (`-ub`, default 512) and its users raise
it by hand — 380 → 2,091 tok/s at `-ub 8192` with experts in RAM on an RTX
3090, paid in card workspace; SGLang and vLLM fix theirs too and say to lower
it on an out-of-memory. Here a card over its memory under WSL2 does not fail —
it spills to host memory at link speed (gotcha #826) — so the size is read
from the card as it stands, not configured.

**What replaced it.** `SplitModel::prompt_reader(index_pos)` is the one answer
for every prompt pass (`prefill_pacer::prompt_chunk_ceiling`; the forward's own
chunk loop sizes EACH chunk as it starts, the batched table its solo prompt):
`CardCopyingWeights { room_tokens }`, clamped to 512-4096. The room is the
card's free memory beside what is in use after a synchronize
(`kv_budget::device_free_and_total_bytes`) less the margin every card budget
keeps; the forward it must hold is PREDICTED from the last one MEASURED
(`MeasuredPrompt::predicted_bytes`): every prompt forward of such a split
records what the pool took beyond what stayed (`begin_prompt_measure`
synchronizes and restarts `USED_MEM_HIGH`, `end_prompt_measure` reads it before
the profile's report resets it), and the next adds a slope per token — the
largest copying layer's activations, each projection's input and output rows at
f32 (`layers::prompt_bytes_per_token`, 0.36 MB a token for the 14B against
the 0.22-0.28 its pool grew by), and what the card-held caches keep — plus the
history a card-run processor layer pulls up per position and the causal mask's
growth (`[t, index_pos + t]` f32, the one term faster than the chunk). A
fixed part (copies in flight, the ring, dequantized weights) never needs a
model: it is inside the measurement. The first chunk of a worker's first prompt
has nothing measured and reads 512; a fallback of any prompt layer to the
processor, or forced standard attention (scores grow with the square), pins 512
for the worker's life. A/B `SWARMLLM_PROMPT_CHUNK_ROOM=0`.

**Measured** (one binary, ABBA, Windows up 12.3 h, binary `1a261725`, live node stopped, safety
kit, `~/swarmllm-138c/ab1h.sh`; `test-split` reads through
`forward_prompt_in_chunks` at the node's configured 128):

| Qwen2.5-14B Q4_K_M 20/28 | 512 a forward (before) | sized from the room | forwards | pool reserved / peak |
|---|---|---|---|---|
| 2,606 tokens (the #138 prompt ×3) | 10.13 / 8.77 s | **5.92 / 5.15 s** (1.7x) | 6 → 2 (512 + 2,094) | 5,216-5,248 / 4,989 → 5,824 / 5,364 MB |
| 4,324 tokens (×5) | 14.55 s | **9.48 s** (1.5x) | 9 → 2 (512 + 3,812) | 5,632 / 5,113 → 6,176 / 5,920 MB |

The card at most 6,905-6,941 MiB of 8,192 (nvidia-smi, the desktop's ~900
included) against 6,316-6,376 before: no spill. The sizing line fired on every
room arm — `room_mb=2608`, a 4,096-token forward predicted at 2,198 MB from the
512-token one's measured 650. The rig's binary divided the card's KV figure by
all 48 layers (33 KB a token kept, 4 of history, where 80 and 12 are right);
corrected before the commit, which lifts that prediction to ~2,350 MB — the
same choice on both prompts. The 30B 26/22 left no room: every chunk read 512
in both arms (no sizing line), and every arm of BOTH was refused part-way by
the card's conversation budget (371 MB at 2,585 tokens) — a budget that
charges the processor-held layers' caches, which live in host memory, against
the card (FUTURE_WORK #262), not this change.

**Replies** (scored against llama.cpp, `score_many.py`, 16 tokens): the
2,606-token prompt gave the same reply in all four arms, llama.cpp's first
choice at every pick. At 4,324 the two arms agree for 15 picks — both leave
llama.cpp's choice at pick 10 the same way (rank 2, 0.15 logits) — and differ
at the 16th, a near-tie in both (rank 3 at 0.31 sized, rank 2 at 0.11 before).

**Verified.** `a_split_copying_its_weights_reads_what_the_card_has_room_for`
(the clamp, 512-4096, and that a configured ceiling above it stands),
`a_prompt_forward_is_predicted_from_the_last_one_measured` (the slope, the
history and the mask terms against hand arithmetic), and the projections test
now asserts the loader's per-token figure (`4 × Σ(in + out)` over a layer's
projections) and that a split with nothing measured reads 512.
`chats_sharing_the_card_keep_the_pacers_choice` passes a reader that panics if
asked: while chats share the tick, the room is never read.

### A card-held layer can keep its experts in RAM (2026-10-13, #138 (3) stage A)

**What it replaced.** Placement was per LAYER: a mixture-of-experts model that
did not fit the card put its last layers wholly on the processor — attention,
cache, router and experts. A layer's experts are ~96% of its bytes on
Qwen3-30B-A3B Q2_K (208 of 216 MB), so the card's memory went mostly on
experts while the processor ran 22 layers' attention as well.

**What llama.cpp does.** `--n-cpu-moe N` keeps the routed experts of the
FIRST N layers in host memory (`llm_add_n_cpu_ffn_overrides` over
`LLM_FFN_EXPS_REGEX`, `common/arg.cpp` at `4b1a27f`) and everything else —
the shared expert included — on the card. Its scheduler copies the
activation and the ids down for each such layer's `MUL_MAT_ID`, and hands a
batch of ≥ 32 to the card with only the used experts copied (#138 (1)).

**What replaced it.** `hybrid::LayerPlacement::with_experts_on_the_processor(n)`
and `experts_device_for(layer)` say where a layer's experts go — the first n
CARD-held layers' to the processor; `split::loader::load_moe_ffn` takes the
PLACEMENT and the layer index rather than a device, so no per-family loop can
place experts differently. The loader's card budget takes those experts' bytes
off (`split::routed_expert_bytes`), and `card_beside_the_processor` gives the card
to such a layer for a prompt. The forward needed no new path:
`MoeFfn::forward_untimed` routes on the card, `route` brings the scores down,
the activation comes down once, the processor's one-region path runs the
experts, the answer goes back; ≥ 32 tokens go through
`processor_experts_on_the_card`. Every cache stays on the card (no
`layer_devices`, so `kv_homes` is the card alone). A segment holding such a
layer works on the processor: `SplitModel::cpu_layer_count` counts it (the
decode pool's key — without it each step's expert regions handed off from
outside a pool, #137), `runs_entirely_on_card` is false (the card's own step
timing), and `reads_prompts_on_card` asks it to read on the card. The profile
names a MoE stage by where its EXPERTS sit.

**Measured** (Qwen3-30B-A3B Q2_K, 26 whole layers vs every layer on the card
with 23 layers' experts in RAM — the same ~5.9 GB of layers either way; an
867-token prompt read 128 a forward, 64 reply tokens; ABBA, one binary,
Windows up 16.3 h, `~/swarmllm-138d/ab3.sh`): median decode step 49 / 68 ms
against **40 / 40 ms**; prompt 4.96 / 4.96 s against 5.07 / 4.90. A step at
the new placement: the processor's experts 21.6-24 ms, the card's 25 stacked
layers 4.6 ms. llama.cpp the same day, `llama-bench -t 4`: 27 whole layers
25.0 tok/s, `-ngl 99 -ncmoe 23` 34.5 (1.38x) — we run it at ~72%.

**Verified.** Replies teacher-forced through llama-cpp-python 0.3.16
(`~/swarmllm-138d/score.py`): both placements leave llama.cpp's first choice
at the same two tokens (our numerics either way); where they part, the new
one takes llama.cpp's first choice. Tests:
`the_first_card_layers_keep_their_experts_in_the_machines_memory` (the rule
by index), and on a card
`a_card_layer_with_its_experts_in_ram_answers_as_a_processor_layer_does` (the
router on the card, experts quantized one by one in RAM, the same picks, a
decode step within 1e-4 of the processor layer and a prompt of the processor
layer offered the card — red with the placement rule switched off) and
`a_card_segment_keeping_its_experts_in_ram_works_on_the_processor_for_them`.
The card's f16 prompt kernel is 0.149 off the processor's answer on the test's
synthetic layer (largest output 5.56), on the existing path alike — that path
was judged against llama.cpp on a real model (#138 (1)).

`test-split --cpu-moe-layers N` sets `split::CPU_EXPERT_LAYERS` directly; the
daemon's choice is the next section.

### The pool keeps experts in RAM before it splits by layer (2026-10-14, #138 (3) stage B)

**What it replaced.** For a model that did not fit the card the pool had one
answer, `split_for_card`: the first layers whole on the card, the rest whole
on the processor. Stage A made a better placement for a mixture-of-experts
model possible, and nothing asked for it.

**What llama.cpp does.** `--fit` (its fit-params logic) first asks whether the
model fits "with only dense weights in device memory", then overflows the
routed experts of the fewest layers to host memory ("48 layers (34
overflowing)") — and shortens the context to make room ("context size reduced
from 40960 to 4096"). We keep the first half and not the second: a split here
promises its full context (the #104 rule, `split_for_card`'s terms).

**What replaced it.** `process_pool::card_split` asks
`experts_in_ram_for_card` first: the fewest FIRST layers whose routed experts,
kept in RAM, leave weights − those experts + every layer's cache at the full
`effective_context` within the card's room less
`FORWARD_BUFFER_RESERVE_BYTES`. Only a model with no experts, or one that does
not fit even with every layer's experts in RAM, falls back to
`split_for_card`. The per-layer expert bytes come from the header in the same
read as the footprint (`segment_geometry`, `split::routed_expert_bytes` — the
one function the loader also uses). The result is a `CardSplit`:

- `Layers { on_card, total }` → `--gpu-layers on_card`, as before;
- `ExpertsInRam { in_ram, total }` → `--gpu-layers -1 --cpu-moe-layers in_ram`
  (`worker_placement_args`), the card charged the segment's estimate less those
  experts (`card_share_of_split_mb` → `experts_in_ram_mb`), the machine those
  experts alone (`ram_charged_share` → `RamShare::Experts`, through
  `admit_to_cpu_as`). No `record_cpu_kv_budget`: every cache is on the card.

The planner is told `(total − in_ram, total)` (`CardSplit::card_share`); the
worker summary, `/v1/status` and the models list carry `expert_layers_in_ram`,
and the dashboard says "the bulk of N layers in this computer's memory". A
refusal by the RAM budget names the experts (`describe_experts_refusal`).
A/B: `SWARMLLM_EXPERTS_IN_RAM=0` = the whole-layer split.

**Measured through the daemon** (Qwen3-30B-A3B Q2_K, 48 layers, 8 GB card;
candidate with `resources.max_ram_mb = 9000` in a copied config — see below;
128-token replies, 3 per arm after a warm-up, `SWARMLLM_PROFILE=1`, ABBA,
Windows up ~18-19 h, `~/swarmllm-138e/verify3.sh`, safety kit, 0 driver
events). The pool chose every layer on the card with the experts of the first
**25** in RAM (stage A's rig used 23: the daemon also charges every layer's
cache at 8192 positions and keeps the forward reserve) against **26** whole
layers. The card held 6262 MiB against 6678-6966; the machine was charged
5213 MB against 5530. Median decode step **50 / 54 ms** (p10 38 / 40) against
55 / 55 (p10 47 / 47); streamed replies 2-3 **20.2, 22.4 / 18.7, 19.5 tok/s**
against 17.2, 18.8 / 17.2, 18.5 (reply 1 is slow in every arm: the decode
pool's width is calibrated on a worker's first reply). The processor's experts
took 23.8-31.8 ms of a step — the gap to llama.cpp at this placement is still
there (stage A: ~72% of its speed).

**Found on the way, #266:** at the default contribution level a 16 GB GPU
node's RAM budget is 4997 MB, so since #264 (unreleased) it refused this model
under EITHER placement, which v0.3.236 runs.

**Verified.** Mechanism on the real path: the daemon's INFO line ("placing all
48 layers there and keeping the routed experts of the first 25"), the worker's
("keeping the routed experts of the first 25 card-held layers in RAM"), the
loader's (`experts_mb=5212`), `/v1/status` worker `gpu_layers_on_card 48,
layers_total 48, expert_layers_in_ram 25`. Tests:
`a_mixture_of_experts_model_keeps_every_layer_on_the_card_and_the_fewest_experts_in_ram`
(minimal to within one layer's experts; heavy-first layers need fewer; a dense
model and a card too small fall back),
`a_split_keeping_experts_in_ram_charges_each_device_for_its_share` (both
charges, the worker's arguments, the planner's share),
`a_resident_split_worker_reports_its_card_share_to_the_planner` and
`a_refusal_of_experts_kept_in_ram_says_what_they_are_and_what_limited_the_budget`.

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

**From the rules file (moved 2026-10-02):**

**`split::loader::load_moe_ffn` is the one way a MoE feed-forward is loaded** —
router dequantized, each routed expert sliced out of the GGUF's stack as its own
QUANTIZED matrix (`split_expert_stack`), shared expert and its sigmoid gate when
present. The three per-family copies it replaced dequantized every expert to f32:
an 11 GB Qwen3-30B-A3B needed ~116 GB, invisible to a capacity planner that sizes
by bytes on disk. A new MoE family calls it. **Routing defaults are per family**
(`moe_renormalizes_by_default` — llama.cpp hardcodes `norm_w` per graph): read the
family's llama.cpp file, never assume. Verify with `examples/logits_reference_probe.rs`
+ `compare_logits_reference.py` on a tiny model; ONE isolated outlier position is a
router near-tie, a shift at every position is a bug — and judge a top-1 model on
UNQUANTIZED weights (Q8 activation rounding flips near-ties). **Llama 4 is hardcoded in
llama.cpp** (sigmoid, no renorm, weight on the expert's INPUT, Q/K RMS-norm after RoPE);
`examples/make_tiny_llama4_gguf.py` builds the fixture. **Routing is one host copy of
the layer's scores and one `index_add` per expert** (`topk_host`) — never per token,
which was ~3 syncs and ~k+5 launches per token per layer on a card.

→ `docs/invariants/inference.md` § "A mixture-of-experts layer keeps its experts quantized"

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

**From the rules file (moved 2026-10-02):**

**`split::hybrid::LayerPlacement` says where each layer goes; every per-layer loop in
`split/loader/` shadows `device`, `cos` and `sin` from it on its first line** —
five loops, one of them the PARALLEL whole-file load, which was missed and put a
split model whole on the card. `hybrid::layers_on_device` is the one count the KV
budget charges, and `hybrid::arch_supports_hybrid` is an allowlist the POOL reads
too (`process_pool::split_for_card`): never offer a split the loader will not
make. Guard: `every_layer_loop_in_the_loader_places_its_layer`. Check on a card
with `swarmllm test-split --gpu-layers N` + `score_ids.py`.

→ `docs/invariants/inference.md` § "A card/processor split is placed in every per-layer loop"

**Qwen 3.5 (dense) joined the allowlist 2026-10-07 (FUTURE_WORK #228 (2)).** Its
first loop built `q35_cos`/`q35_sin` outside the loop, which is why it was left
off; the rewritten loop loads every tensor through the shadowed names, and a
DeltaNet layer creates its recurrent and convolution state on its INPUT's device
(`layers::qwen35::forward_deltanet`), which the executor's per-layer move puts on
the layer's device before any layer kind runs — so the state lives where its
layer does with no code of its own. Checked on the RTX 3070 against llama.cpp
master (`test-split --gpu-layers=N`, `~/llama.cpp-ref/score_ids_dump.py`, safety
kit, live node stopped, 0 driver events): the v0.3.229 binary asked for 12 layers
logs "not verified for this architecture" and loads the whole model on the card
(the control); this build logs "first 12 layers on the graphics card". 0.8B Q8_0
split 12/24: all three prompts PASS (prompt tokens identical, worst gap 0.012).
4B Q4_K_M split 16/32: France and Fibonacci PASS (gap 0); Mars FAILS the 0.05
margin at one position by **0.16411 — identically whole on the card, whole on the
processor and split 16**, and 0.13/0.22 split 8/24 — not the split's. Nor a Qwen
3.5 defect: on six prompts Llama-3.2-3B Q4_K_M misses the same margin by 0.306,
and the unquantized 0.8B matches llama.cpp exactly. ⚠ **`score_ids_dump.py`'s
0.05 margin flags 4-bit near-ties on ANY family** — judge a Q4 file against an
established family's distribution on the same prompts, and settle correctness on
an F32/Q8 file. The
split reproduces the unsplit reply. Sized the same day: admission, the split
planner (`process_pool::split_for_card`) and the loader's KV budget charged every
layer an attention layer's KV, where three in four are DeltaNet with a fixed state
— 805 MB too much for the 9B at the admission context. All three ask
`split::layers_keeping_kv` now, from the tensor table (`blk.N.ssm_alpha.weight`,
the loader's own test, `split::layer_is_recurrent`).

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
(`model_worker::snapshot_room`, gotcha #440) and admission evicts cached prompts
before refusing a request. Default now 0 (no token ceiling). What a position
weighs varies ~3x between 3B models (73.7 KB qwen2.5-3b, 229 KB llama-3.2-3b),
which is why the ceiling is derived, not raised.

Sizing BEFORE the copy also retired "keep one entry over budget": that existed
because the whole prompt had already been copied, and discarding it wasted the
copy. Tests: `a_prompt_over_the_token_ceiling_keeps_its_opening`,
`with_no_token_ceiling_the_byte_budget_sizes_the_snapshot`,
`a_budget_smaller_than_one_prompt_keeps_the_opening_that_fits`.

## A blocked GQA attention call is grouped per block, never expanded (2026-09-26)

`standard_attention` groups a GQA call's query heads against the unexpanded
cache whenever the scores fit `ATTN_SCORE_BUDGET_ELEMS` (16Mi) in one pass
(`grouping_applies`). Past the budget it BLOCKED the query axis and, because a
slice of the grouped axis would cut one position's repeats across two passes,
expanded K and V with `repeat_kv` first. For a 128-token prompt chunk on a
24-head model the budget is crossed at a 5,461-long cache, so from there EVERY
chunk copied its layer's whole K and V three times over.

The fix slices query POSITIONS first and groups each block
(`grouped_gqa_attention` on `q.narrow(2, …)` with `mask_rows`): no boundary
problem, and a grouped block holds exactly the score elements an ungrouped one
does, so the budget still bounds memory. Only for no mask or a 2D mask — the
shapes `grouped_gqa_attention` tiles itself; a 4D mask keeps the expanded path.
A/B: `SWARMLLM_GROUPED_GQA_BLOCKED=0` (this only), and the older
`SWARMLLM_GROUPED_GQA_DECODE_ONLY=1` still restores the expanded path.

**Measured** (Ryzen 7 5800H, llama-3.2-3b Q4_K_M, a node reading a 6,911-token
prompt on 8 threads, one binary): 208.12 → 173.80 s, 33.2 → 39.8 tok/s (+20%);
total attention 85.7 → 54.1 s. Per 1K cached positions the expanded path cost
303 ms at position 4096 and 309 at 5248, then 685 at 5376 — the crossing — and
686-729 to the end; grouped, 270-317 throughout. Agent prompts run 4,100-14,400
tokens (FUTURE_WORK #119), so most of an agent turn sits past the cliff.

Test `a_blocked_gqa_call_is_grouped_not_expanded` asserts equality with the
expanded unblocked pass (< 1e-5) AND that no expansion happened (a test-only
`KV_EXPANSIONS` counter) — the values alone match either way, so the count is
the part that fails without the fix.

**Superseded on the processor the same day** by `prefill_attn` (next section):
the grouped and blocked matmul paths now run only where that kernel declines,
and their tests call `matmul_attention` directly so they keep testing them.

## Several query positions on the processor: tiled over the keys, the score matrix never written (2026-09-26, #119)

**What the matmul path cost.** `attention_scores_block` writes the whole
`[rows, kv_len]` score matrix, the softmax reads and rewrites it, the second
matmul reads it again. For a 128-token chunk of llama-3.2-3b (8 groups × 384
rows) at 5,000 cached positions that matrix is 61 MB per layer, and filling
61 MB once took ~6 ms on the 5800H — about 250 MB of traffic per layer per
chunk for arithmetic that ran at ~20% of f32 peak. Op split (`examples/attn_bench.rs`,
`SWARM_ATTN_KV=5000`, 8 threads): scores matmul 25 ms at 78 GMAC/s (a 2D gemm
of the same shape: the same 75), softmax 22 ms, P·V 11 ms at 175 GMAC/s. At one
thread the scores matmul ran at 21 GMAC/s against P·V's 41: the matrix with the
huge OUTPUT was the slow one. Fresh-page faults were ruled out (a fresh 61 MB
buffer filled in 6.3 ms, a reused one in 5.8).

**Now**: `inference::prefill_attn::gqa_prefill_attention_cpu`, dispatched at the
top of `standard_attention` beside the decode kernel for `q_len >= 2` on the CPU.
FlashAttention's tiling as PyTorch's CPU kernel does it
(`aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`: a gemm per key tile, an
online softmax with a running max and sum per row, a gemm accumulating P·V) —
but split over fixed CHUNKS OF KEYS (1,024) per KV group, every query row of the
group in one task, merged by flash-decoding's reduction. PyTorch splits over
query blocks one head at a time; with grouped heads that would re-read a group's
K and V once per block. Tile 128 keys (the score tile beside the output fits L2),
row blocks of 512 (bounded buffers for a whole prompt in one forward). All three
sizes are constants, so the result does not depend on the thread count. q is
pre-scaled once; the mask is read where it lies (rank 2 or `[1,1,q,S]`, unit inner
stride); anything else declines to `matmul_attention`. `gemm` 0.19 is a direct
dependency for this — the same version vendored candle resolves — because the
per-tile products must reuse their buffers.

**Measured**, one binary, `SWARMLLM_PREFILL_ATTN=standard` as the control, live
node stopped, `examples/prefill_bench` reading the prompt in 128-token chunks as
a node does (`SWARM_BENCH_CHUNK=128`), 8 threads, arms interleaved, best of 2:

| | matmul path | tiled kernel |
|---|---|---|
| llama-3.2-3b, 2,048-token prompt | 53.9 tok/s | 56.3 (+4%) |
| llama-3.2-3b, 6,144-token prompt | 43.7 tok/s (140.5 s) | **51.1** (120.4 s, +17%) |
| Qwen2.5-7B, 2,048-token prompt (1 round) | 25.2 | 27.0 (+7%) |

Kernel alone against the unfused candle composition (`attn_bench`): 2.2x at 512
cached positions, 2.7x at 2,048, 3.7x at 5,000. The end-to-end gain is smaller
because production's softmax was already fused and the quantized matmuls around
attention are most of a forward. llama.cpp read a 6,911-token prompt at 49.9 tok/s
on these 8 threads (FUTURE_WORK #119, closed — see `docs/FUTURE_WORK_ARCHIVE.md`).

**Correctness.** `prefill_kernel_matches_the_matmul_path` (< 1e-5 abs) across one
tile, several tiles, several chunks, masks hiding whole tiles, soft-cap, MHA,
batch 2 and four row blocks; breaking the chunk merge's weight makes it fail
(0.24 abs at 2,100 keys). On a real model, `examples/logits_reference_probe.rs`
over 1,100 tokens (a 1,096-token prompt pass: seven row blocks, two chunks) scored
against llama-cpp-python 0.3.16: median cosine 0.99975 on both arms, top-1
agreement 969/1100 tiled vs 966/1100 matmul. Both arms show the SAME isolated
outliers (position 619 at cosine −0.07 / −0.04, 652 at 0.73 / 0.67) — pre-existing,
on the probe's synthetic token sequence, not from this change (FUTURE_WORK #124).

**What a change must keep.** The chunk, tile and row-block sizes stay constants.
Tests of the matmul paths (blocking exact, grouping never expands K/V, the value
cache read in place) call `matmul_attention`, never `standard_attention` — on the
processor the latter answers from this kernel first, and a test through it
passes without reaching the code it names.

**From the rules file (moved 2026-10-02):**

**`inference::prefill_attn::gqa_prefill_attention_cpu`** answers a CPU
attention call with `q_len >= 2` before any matmul path: tiled over fixed
1,024-key chunks per KV group (online softmax, flash-decoding merge), so a
group's K/V are read once and the `[rows, kv_len]` matrix never leaves the
cache — +17% on a 6K-token prompt (#119). The matmul paths are now
**`layers::matmul_attention`**, reached when a kernel declines and always on a
card; **a test of blocking, grouping or in-place V calls `matmul_attention`** —
through `standard_attention` it would pass without reaching them. Chunk, tile
and row-block sizes stay constants (result independent of thread count).
A/B: `SWARMLLM_PREFILL_ATTN=standard`.

→ `docs/invariants/inference.md` § "Several query positions on the processor"

## A model's RoPE frequency factors are applied — and reach a node without shard 0 (2026-09-27, #124)

**The defect.** Llama 3.1/3.2 GGUFs carry `rope_freqs.weight`: 64 per-pair
divisors (1.0 for fast pairs up to the scaling factor, 32 on Llama 3.2, for slow
ones), which is how llama.cpp's converter bakes the "llama3" RoPE scaling into
the file (llama.cpp PR #8676). llama.cpp divides each pair's angle by its factor
(`ggml_rope_ext`'s `freq_factors`). Our loader read only Phi-3's
`rope_factors_long/short`, so every Llama 3 model rotated its slow pairs up to
32x too fast — not what it was trained with — and the error grew with distance.

**How it was found.** `logits_reference_probe` over 1,100 tokens scored against
llama-cpp-python 0.3.16. On the probe's synthetic ids a few positions looked
catastrophic (cosine −0.07) on BOTH attention paths; on natural text
(`LOGITS_PROBE_IDS`, README prose tokenized by llama.cpp) those vanished but the
median drifted 0.9982 → 0.9963 from the first 100 positions to the last. It was
NOT llama.cpp's f16 KV cache: with `type_k = type_v = f32` the agreement was the
same (0.99786), while llama.cpp f16-vs-f32 agree at 0.9994.

**Measured fix** (llama-3.2-3b Q4_K_M, the same 1,100 natural tokens, vs llama.cpp):

| | plain RoPE | with `rope_freqs` |
|---|---|---|
| median cosine | 0.99782 | **0.99939** |
| worst position | 0.925 | **0.987** |
| top-1 agreement | 1059/1100 | **1075/1100** |
| positions 1000-1099 | 0.99629, 93/100 | **0.99943, 98/100** |

No drift with position remains; what is left is llama.cpp's own KV-precision noise.

**A node without shard 0.** The tensor is the first in the data region, in the
prefix shard. `GgufTensorMeta::sidecar_tensors` is now the ONE list of shard-0
tensors a later-layer node needs — the tied output head and these factors — and
all three consumers walk it: `extract_sidecar_tensors` (cut from a local
`shard_000.bin` at startup — located through shard 0's OWN tensor table, since a shard
file is packed data and the tensor's GGUF offset is not its offset there; found at
the gate when the GGUF offset read garbage), `download_sidecar_tensors` (a ranged GET per tensor
on every HuggingFace path, skipping files already present), and
`ShardReader::new(.., resolve_sidecars(..))`, which maps each file into the
virtual GGUF so the loader's read is unchanged. A node with neither shard 0 nor
`rope_freqs.bin` serves with plain RoPE and logs a warning — the behaviour of
every earlier release — rather than refusing layers a request may need.

**What a change must keep.** A new shard-0 tensor a later layer needs is added to
`sidecar_tensors`, never given its own producer. When checking an architecture,
diff the GGUF's tensor names against what the loader reads: a tensor we ignore
is a feature silently missing (gotcha #728). Tests: `unit_factors_are_plain_rope`,
`a_factor_divides_its_pairs_angle`,
`rope_factors_readable_without_shard_zero_only_through_their_sidecar`.

**From the rules file (moved 2026-10-02):**

Llama 3.1/3.2 ship `rope_freqs.weight` and nothing read it, so every Llama 3
rotated its slow RoPE pairs up to 32x too fast; agreement with llama.cpp drifted
with conversation length (#124). Now applied as llama.cpp does. **Shard-0 tensors
a later-layer node needs are ONE list** — `GgufTensorMeta::sidecar_tensors` (tied
head, RoPE factors) — walked by `extract_sidecar_tensors`,
`download_sidecar_tensors` and `resolve_sidecars`; add to the list, never a new
producer. When checking an architecture, diff the file's tensor names against
what the loader reads.

→ `docs/invariants/inference.md` § "A model's RoPE frequency factors are applied"

## A prompt pass on the card multiplies quantized weights on the tensor cores (2026-09-29)

**What it replaced.** Every quantized matmul with more than eight activation rows — a prompt
pass, on a card — ran the vendored MMQ kernels (`mul_mat_q4_K` and friends): llama.cpp's OLDER
dp4a kernels, int8 dot products on the ordinary cores, no tensor cores. And a prompt was read
128 tokens per forward (`inference.prefill_chunk_tokens`, a default sized for the processor,
gotcha #191). An 8B read a prompt at ~740 tok/s on an RTX 3070 Laptop — ~12 TFLOP/s effective
on a card whose fp16 tensor cores do several times that. Each 128-token chunk launched MMQ
grids of 64 blocks for a 4096-row projection, too few to fill the card.

**Research.** llama.cpp routes a dp4a MMQ to cuBLAS above `MMQ_DP4A_MAX_BATCH_SIZE` = 64 when
the card has fp16 tensor cores ("Max. batch size to use for dp4a MMQ kernels when FP16 tensor
cores are available", `ggml-cuda/mmq.cuh`; `ggml_cuda_should_use_mmq` returns
`!fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE`); its cuBLAS path
(`ggml_cuda_op_mul_mat_cublas`) dequantizes the weight to f16, casts the activation to f16 and
runs `cublasGemmEx`. Current llama.cpp keeps MMQ on Ampere only because its NEWER MMQ uses int8
tensor-core `mma` — which the vendored kernels do not. llama.cpp's default `n_ubatch` is 512.

**The rule.**
- Vendored `quantized/cuda.rs::dequantize_matmul`: ≥ `CUBLAS_MIN_ROWS` (64) rows on a card of
  compute capability ≥ 7.0 → `mul_mat_via_f16_cublas` (`dequantize_f16` kernel → f16 weight;
  candle's `cast_f32_f16` → f16 activation; `cublasGemmEx`, f16 accumulate since 2026-10-09 (f32 for a
  marked weight), f32 out — MMQ's
  layout). The only production caller of `mul_mat_via_q8_1`, so every quantized prompt pass
  passes the check. `SWARMLLM_QMATMUL_CUBLAS=0` = MMQ always; `SWARMLLM_QMATMUL_CUBLAS_MIN_ROWS`
  moves the threshold for a measurement; `SWARMLLM_QMATMUL_CUBLAS_ACC=32` accumulates every weight
  in f32 (below).
- `prefill_pacer::prompt_chunk_ceiling(configured, reader)` — the configured ceiling, raised to
  `CARD_CHUNK_TOKENS` (512) on a model that runs ENTIRELY on the card (`PromptReader::Card`).
  Read by the batched table for a prompt with nobody waiting (`PrefillPacer::chunk_size_for`;
  while chats share the tick the pacer's choice stands), and by
  `SplitModel::forward_prompt_in_chunks`, which a segment's prompt pass and the drafter take.
  A card/processor split that reads its prompts on the card takes 512 or more from the card's
  room (`PromptReader::CardCopyingWeights`, § "A split's prompt chunk is as large as the card
  has room for"); one that does not keeps the configured one.

**Measured 2026-09-28/29** (RTX 3070 Laptop 8 GB, WSL2, Windows uptime ~6 h, isolated node, live
node stopped, SINGLE requests; unique prompts, `max_tokens` 1, best of 3;
`~/swarmllm-pool-0928/{chunk_ab,f16_ab,multi_ab,multi_ab2}.sh`):
- **Chunk alone** (MMQ, Llama-3.1-8B): 128 → 512 took 742 → 856 tok/s (1,835 tokens) and 740 →
  852 (3,035); 1,024 added ~1%; replies byte-identical across chunk sizes.
- **Mechanism**: `SWARMLLM_COUNT_KERNELS=1` on a 512-token chunk shows 6 `dequantize_block_q4_K_f16`
  + 1 `dequantize_block_q6_K_f16` per layer (the layer's 7 projections) and no `mul_mat_q4_K`.
- **f16 + cuBLAS vs MMQ at 512** (tok/s, ~1,800 / ~3,000-token prompts, arms alternated):

  | model | MMQ | f16, f32 accumulate (default) | f16 accumulate (opt-in) |
  |---|---|---|---|
  | Llama-3.1-8B | 876 / 864, 839 / 831 | 1,129 / 1,158, 1,112 / 1,143 | 1,398 / 1,436, 1,357 / 1,391 |
  | Qwen2.5-Coder-7B | 904 / 899 | 1,178 / 1,213 | 1,454 / 1,494 |
  | Mistral-7B-v0.3 | 857 / 867 | 1,105 / 1,143 | 1,348 / 1,383 |
  | GLM-4-9B | 631 / — | 783 / — | 934 / — |
  | Phi-3.5-mini | 1,256 / 1,225 | 1,552 / 1,546 | 1,814 / 1,801 |
  | Phi-4-mini | 1,465 / 1,472 | 1,857 / 1,896 | 2,163 / 2,207 |
  | Llama-3.2-3B | 1,870 / 1,854 | 2,315 / 2,317 | 2,801 / 2,826 |
  | Gemma-2-2B | 2,294 / 2,338 | 2,985 / 3,010 | 3,420 / 3,485 |

  Together with the chunk: Llama-3.1-8B ~740 → ~1,150 tok/s (1.55×) by default, ~1,400 (1.9×)
  opt-in. Decode is untouched (~47 tok/s every arm — one row never reaches the path). GLM-4-9B's
  3,000-word prompt is refused in every arm by its LOAD-time budget (304 MB, ~2,600 tokens on
  this card beside the desktop) — not this change.
- **Replies against llama.cpp** (`score_against_reference.py`, llama-cpp-python 0.3.16, three
  ~200-350-token prompts per family, greedy 160 tokens; prompt-token counts equal to the node's):
  each arm reproduces itself exactly across processes; the arms differ from each other at
  near-ties (MMQ rounds activations to 8 bits, this path to f16). Share of reply tokens that
  llama.cpp ranks first, Qwen2.5-7B + Llama-3.2-3B + Gemma-2-2B: MMQ 98.3%, f16/f32-acc 97.5%,
  f16-acc 98.2% — every miss rank ≤ 3 with a logit gap ≤ 0.43. Phi-3.5 (SentencePiece, scored by
  near-ties only) is the same in all arms. Mistral-7B (SentencePiece), Phi-4-mini and GLM-4-9B
  (both with the node's default system turn, prompt counts equal): MMQ 464/480, 465/473, 417/464;
  f32-acc 464/480, 467/476, 413/460; f16-acc 460/475, 466/476, 417/454 — after the replies fork,
  each continuation meets its own near-ties, so a few tokens either way is noise; the worst
  ranks and largest gaps are the SAME positions in every arm (`~/swarmllm-pool-0928/score/`).
- ⚠ **Score a LongRoPE model (Phi-3.5, Phi-4-mini) at the node's window**: at the scorer's old
  fixed `n_ctx=4096` llama.cpp used Phi's SHORT rotary factors against a node on the LONG ones
  (it serves 8192), and Phi-4-mini read ~85% with 1.5-logit gaps on every path — a reference
  misconfiguration that looked like a model bug. `score_against_reference.py --n-ctx` now
  defaults to 8192 (gotcha #757). GLM-4-9B keeps ~20-logit gaps on every path at either size —
  FUTURE_WORK #147 item 6.
- ⚠ Llama-3.x replies change across midnight: its template writes today's date. Compare runs on
  the same day, or a family whose template has none (Qwen: byte-identical across runs).

**f16 accumulate is the default since 2026-10-09 (FUTURE_WORK #147), with llama.cpp's f32
marks.** It stayed opt-in until a card-side check at LONG context, since its failure (an overflow
or a lost low bit) would be a silently wrong reply that three short prompts could not rule out,
and the release gate's family check runs on the processor. The check
(`~/swarmllm-147/acc.sh`, `examples/logits_reference_probe.rs` with `LOGITS_PROBE_CHUNK=512
LOGITS_PROBE_KEEP_LAST=128`, `compare_logits_reference.py`'s per-arm reading): a 4,096-token
natural-text prompt read on the card in 512-token chunks, the last 128 prompt positions' logits
and 8 decode steps against llama.cpp on the processor, f32 and f16 accumulate in ONE binary.
Median cosine / top-1 of 128, f32 → f16: Qwen3-1.7B 0.999838/126 → 0.999832/124, Llama-3.2-3B
0.999648/126 → 0.999644/123, Gemma-2-2B 0.999585/124 → 0.999579/124, Qwen2.5-Coder-7B
0.999782/126 → 0.999773/127, Mistral-7B 0.999902/127 → 0.999901/127, Phi-3.5 0.999992/125 →
0.999991/124, Phi-4-mini 0.999987/123 → 0.999984/116, GLM-4-9B 0.999516/126 → 0.999496/125;
decode rows agree in every arm. Wherever the two arms' argmax differ, the f32 arm's own gap
between the two picks is ≤ 0.29 (Phi-4-mini's seven: 0.018-0.291) — near-ties, the precision
llama.cpp itself trades on every card. GLM-4's arms differed most from EACH OTHER (worst cosine
0.99956, against ≥ 0.99985 for every other family), which is llama.cpp's own reason: it marks
GLM4 / GLM4_MOE / JAIS2's `ffn_down` and attention `wo` `GGML_PREC_F32` ("numerical issues with
half-precision accumulators", `src/llama-graph.cpp` master 4b1a27f), as it does KQ and flash
attention (ours accumulate in f32 already and never take this path) and two MoE ops of models
we do not run. So `ModelArch::accumulates_outputs_in_f32` (GLM-4) has the loader mark those two
projections once every layer is built (`layers::accumulate_output_projections_in_f32`; the
storage flag `QCudaStorage::accumulate_in_f32` is read by `dequantize_matmul`). Verified on the
card (`~/swarmllm-147/glm.sh`, `SWARMLLM_F32_MARKS=0` the unmarked arm): `marked=80 missed=0`
(40 layers × 2); the marks change the result (marked vs unmarked median cosine 0.999976, one
argmax), but on this text bring it no closer to all-f32 (median 0.999985 vs 0.999982) — the rest
of the difference is the other projections'. llama.cpp's guard is against GLM-4 inputs whose
products leave f16's range, which this text does not reach; the marks are kept as its rule, at
the cost of two of seven projections per GLM-4 layer at the slower accumulate. Not covered:
Qwen 3.5 (its reference is llama.cpp master, not llama-cpp-python) and a MoE model whole on the
card (none fits 8 GB).

**What a change must keep.**
- The threshold is about the KERNEL, not the model: a future int8-`mma` MMQ would move it, as it
  did in llama.cpp.
- A card without fp16 tensor cores (< 7.0) keeps MMQ.
- Decode (≤ 8 rows) never takes it; batched decode stays on the vec kernel.
- Replies are judged against llama.cpp, never byte-equality with MMQ.
- **A new architecture's f32 marks are read off llama.cpp** (`ggml_prec_set_acc` in
  `src/llama-graph.cpp`) into `ModelArch::accumulates_outputs_in_f32`, never inferred — and
  checked with the 4K probe above in both arms, since a mark llama.cpp has and we lack is a
  silent overflow on that family only.

**From the rules file (moved 2026-10-02):**

**Vendored `quantized/cuda.rs::dequantize_matmul` sends ≥ 64 activation rows to
`mul_mat_via_f16_cublas`** — weight dequantized to f16, activation cast to f16,
one `cublasGemmEx` accumulating in f16 (f32 for a weight marked `accumulate_in_f32`) — instead of the MMQ kernel, which is
llama.cpp's OLD dp4a one with no tensor cores. llama.cpp's own rule for a dp4a
MMQ on a card with fp16 tensor cores (`MMQ_DP4A_MAX_BATCH_SIZE` = 64). Decode
never reaches it. **And a prompt alone on an all-card model reads in chunks of
≥ `prefill_pacer::CARD_CHUNK_TOKENS` (512)** — `prompt_chunk_ceiling` is the ONE
answer, for the batched table, a segment's prompt pass and the drafter.
Replies MAY move by a near-tie against MMQ (8-bit vs f16 activations): judge
them against llama.cpp, never byte-equality. A/B: `SWARMLLM_QMATMUL_CUBLAS=0`;
`SWARMLLM_QMATMUL_CUBLAS_ACC=32` accumulates every weight in f32 (the default
until 2026-10-09). The f16
path does not share its activation cast across q/k/v or gate/up (3 extra casts
per layer, ~1% of a chunk) — the third quantized path the sharing rule above
would name.

→ `docs/invariants/inference.md` § "A prompt pass on the card multiplies quantized weights on the tensor cores"

## A decode step can go to the card as one CUDA graph (2026-09-29; on by default since v0.3.213)

**What.** (Written when it was opt-in via `SWARMLLM_CUDA_OWN_STREAM=1 SWARMLLM_CUDA_GRAPH=1`; both are now on unless set to `0`.) It sends a one-position forward
(dense layers, the whole segment on one card, no LoRA) to the card as ONE graph:
`SplitModel::forward_decode_as_graph` captures `forward_inner_body` on the device's own stream,
`inference::cuda_graph` updates the model's one instantiated graph from the capture
(`cuGraphExecUpdate`) and launches it. Re-captured every token, as llama.cpp does — no length
classes, no stable-buffer arena: candle's per-op allocations become graph memory nodes.

**Why re-capture works here** (`examples/cuda_graph_cost_probe.cu`,
`examples/cuda_graph_capture_rules.cu`, RTX 3070 Laptop, WSL2): recording a launch costs ~2 µs
against ~10 µs to submit one, and the update accepts a graph whose allocation SIZES and launch
GRIDS changed since the last one — 199/199 updated, none re-instantiated — which is exactly what
a growing KV length does to the attention scores.

**The trap that shapes the code.** A PAGEABLE host→device copy inside a capture is not refused:
the capture records it and every LAUNCH reads the host address — the probe changed the host
buffer after capture and the card received the new value. Candle copies from temporaries it frees
at once, so a copy in a captured step is garbage, silently. Hence:
- Vendored candle counts every copy, always (`cuda_backend::device::HTOD_COPIES`,
  `candle_core::cuda::htod_copies_so_far`) — both copy functions are the only way in, which is
  why the per-line table was already complete. A capture during which the count moved is thrown
  away.
- The quantized embedding gather no longer uploads its layout: the kernel reads it only to ask
  whether the source is contiguous, and with zero dims `is_contiguous` says yes without reading.
  That was the one copy inside a decode step (the token id is made before the forward).
- A hidden state arriving from a peer is moved to the card BEFORE the capture.
- candle's `index_select` (every DENSE embedding lookup — an fp16 model such as the split drafter
  Qwen2.5-0.5B) uploads no layout for contiguous ids either (2026-09-30): the default-flip gate
  caught the drafter refused three times for "a host-to-device copy inside the capture" and giving
  up — the safety net working as designed. After the patch: replies identical to the old binary
  (3/3), captured with 0 refusals, 82-95 → 113-116 tok/s alone.

**What else a capture must not do, and how each is kept out** (`local_decode_submissions.md`
§ Stage 4b): free memory made before it (a KV buffer that grows, a mirror a hydrated snapshot
brought, dropped on the first decode step) — so only a step AFTER an ordinary one of the same
conversation (any position since 2026-09-30: a rollback or a jump past a several-position pass
has nothing of a first step left to do), whose every KV buffer already holds the new position
(`KvCacheStore::every_cache_holds`), is captured; keep memory made inside it — the output is
copied into a tensor made before the capture; synchronise — CUDA refuses that loudly.

**A refusal never fails the request.** `forward_inner_body` moved each layer's cache length on
the host even though nothing ran, so the caller truncates to `index_pos`
(`truncate_request_to`) and runs the step the ordinary way. That needed a change underneath: a
forward that failed part-way used to DROP the request's caches (taken out of the store, never put
back), leaving an entry with no layers; it now puts them back cut to their old length
(`a_step_that_fails_part_way_keeps_the_conversation`, red without it). Three refusals that are the
program's fault — a copy, a failed call, a broken capture — turn capture off for the model: one of
them (a free of outside memory) also leaks the buffer, so a systematic one would leak per token.

**Measured** (one binary, interleaved arms, isolated node, greedy, 3 prompts × 256 tokens, host
uptime ~21 h, `~/swarmllm-graph-0929/`):

| model | legacy stream | own stream | own stream + graph | replies vs legacy |
|---|---|---|---|---|
| TinyLlama 1.1B Q4_K_M (22 L) | 94-99 tok/s | — | **141-146** | 3/3 identical |
| Llama 3.2 3B Q4_K_M (28 L) | 62-70 | — | **75-77** | 6/6 identical |
| Llama 3.1 8B Q4_K_M (32 L) | 45.0-46.6 | 45.1-47.3 | 44.3-45.1 | 6/6 identical (own 6/6 too) |

- 773-780 launches per arm, all but the first updated in place, 0 refused. The own-stream arm is
  the first `--features cuda` generation on an own stream since .199 (#683/#685): identical.
- **No gain on the 8B, a slight loss**: its decode is card-bound, and issued op by op the card
  starts each kernel while the processor submits the next; a graph launches nothing until the
  whole step is recorded (~2-3 ms of an idle card). The gain is where submission, not the card,
  was the bottleneck — small models, and small segments of them.
- **A prompt pass 0.3-1.9 s slow is NOT the graph's doing** — corrected the same day. It showed in
  graph arms first (3B: a 53-token pass 310 ms against 17 ms, the thread blocked 455 ms inside the
  LAST layer — work already queued, the card busy elsewhere), but the Qwen 2.5 Coder 7B run then
  hit a LEGACY arm (0.98 s) and the llama.cpp arms (0.43-0.66 s against 0.14 s). A standalone probe
  found default-pool growth 2-3 ms with or without graph memory held. It is a host effect at ~22 h
  of Windows uptime (the #755 family), not a property of capture.
- **Why it stayed OFF (at 2026-09-29; the default was flipped in v0.3.213, see the flip's gate below)**: most models a split carries are 7-8B and card-bound, where it gains nothing
  and costs ~2%. Qwen 2.5 Coder 7B: legacy 46.7-48.7, graph 46.2-47.0 tok/s — and llama.cpp, in the
  same binary on the same card and file, **57.5-57.7**: the gap on a card-bound model is the KERNELS,
  ~20%, not submissions.

### A capture writes nothing below `index_pos` (2026-09-30, #761)

**What broke.** A decode step leaves the f16 mirror behind (the lazy mirror) and the
next multi-position append catches it up. With multi-position steps captured (a
speculative check, `mask_is_read` false), that catch-up was RECORDED — and it was the
very thing that got the capture refused: its slice of the lag (`narrow(have, lag)`) is
a layout seen for the first time, which candle uploads from the host (4 copies per
refusal, logged as `a host-to-device copy inside the capture`). Refused, nothing the
capture recorded ran, while each recorded layer's mirror length had already moved past
the catch-up; the rollback cut back to the step's own start (`index_pos`) and so left
the mirror CLAIMING the lagging positions it never received. Flash attention then read
whatever the buffer held there. Groups launched before the refused one (pipelined
capture) had run their catch-up for real, so only the refused group's two layers drifted.

**How it showed.** Split speculation's streamed checks (`dsd_stream`) send many
one-row checks, so they hit it often: at one sentence of a fixed prompt
("the rain battered ___") rounds and plain local decoding chose llama.cpp's first
choice ("the") in every run, while the stream chose its second ("against", 0.58-0.92
logits behind) in 6 of 11 — and in 0 of 3 with `SWARMLLM_CUDA_GRAPH=0`. Each node logged
3 refusals (then gave the shape up, so the stream also lost its graphs). Plausible text,
worst rank 2: only the scorer's larger gaps and a same-context A/B made it visible.
Reachable on the default split path too — the n-gram loop's miss is a one-row check and
its hit a multi-row one; DSD's γ = 0 rounds likewise.

**The fix.** `decode_step_with_graph` calls `KvCacheStore::catch_up_mirrors` before
recording any multi-position step (`LayerKv::catch_up_mirror`, uncaptured); the recorded
step then only appends what it adds, and a rollback to `index_pos` is exact. After:
3/3 "the", 0 refusals on both nodes, stream replies scored worst rank 2 with largest
gaps 0.156 / 0.074 / 0.113 (rounds' own: 0.156 / 0.120 / 0.003), and 24-26 tok/s against
17-25 (graphs no longer given up). Test: `a_lagging_mirror_is_caught_up_before_a_step_is_recorded`.

**What a change must keep:** anything a capture writes must be at or past the step's
`index_pos`, or be rolled back by something other than `truncate_request_to(index_pos)`.
A host-side counter that moves inside a capture is the trap — the device write it
stands for runs only at launch.

**From the rules file (moved 2026-10-02):**

**`inference::cuda_graph` + `SplitModel::forward_decode_as_graph`** re-capture a
one-position forward every token and update one instantiated graph in place
(llama.cpp's way) — on by default, `SWARMLLM_CUDA_GRAPH=0` off (so is the legacy stream,
which cannot be captured). The flip's gate, on a `--features cuda` build: five models, a 7B
split over two card nodes, `failover_mid`, 700-token replies, split speculation — identical or
scored against llama.cpp. ⚠ `split_rig.sh failover` with B ON THE CARD fails with or without
graphs: B's prompt pass ends before the kill lands, and a stand-in is offered only on it. **Recorded in groups of two layers, each launched as soon as it is recorded**
(`cuda_graph::group_layers`, `Cutter::cut` via the forward's per-layer hook), so the card runs
one group while the next is recorded — a whole-step graph left it idle for all 4.5 ms of recording
and gained nothing on a 7B. Replies byte-identical; TinyLlama 126-135 → 226-229 tok/s, 3B 80-90 →
105-106, Qwen2.5-Coder-7B 49-51 → 56-57 (llama.cpp 57.5), Llama 3.1 8B 50-51 → 55. Only the
residual stream crosses a group boundary, parked in `DecodeGraph::boundaries` (made outside every
capture). ⛔ **A pageable
host→device copy inside a capture is ACCEPTED and replayed from the host address
at LAUNCH** — silent garbage. So vendored candle counts every copy
(`htod_copies_so_far`), a capture that made one is thrown away, and a new
`clone_htod` on the decode path turns capture off for that model. Capture only a
step whose KV buffers already hold it (`KvCacheStore::every_cache_holds`); a
refusal rolls the cache back and runs the step the ordinary way. **A forward that
fails part-way now puts its caches back, cut to their old length** — it used to
drop them all.
**Nothing below `index_pos` is written inside a capture** — a rollback cuts back to
`index_pos` only. So a multi-position step catches the lazy f16 mirror up BEFORE it is
recorded (`LayerKv::catch_up_mirror`, from `decode_step_with_graph`): recorded, the
catch-up was what got the capture refused (its slice is a new layout, uploaded from the
host) and the refused capture left the mirror claiming positions it never received —
flash then read them, and a split's checks drifted to llama.cpp's 2nd choice (#761).

→ `docs/invariants/inference.md` § "A decode step can go to the card as one CUDA graph"

## A decoded token's attention on a card is one kernel (2026-09-29)

**What.** `decode_attn::gqa_decode_attention_cuda` + `kernels/decode_attn.cu`: one query position,
the f32 cache read where it lies, split into fixed 64-position chunks (Dao et al., "Flash-Decoding
for long-context inference", 2023; llama.cpp's `fattn-vec` + `flash_attn_combine_results`). One
block holds ONE KV head and EVERY query head of its group, so a K or V row is read once per group
(#119's lesson from the CPU kernel). Replaces `affine` + two cuBLAS matmuls + `softmax`. Pinned
against the matmul composition on the card by `cuda_decode_kernel_matches_the_matmul_path`
(10 shapes: n_rep 1-7, d 64/128/256, soft-cap, 8K positions, a narrowed cache; < 1e-5 abs).

**And the f16 mirror is lazy.** Decode never reads it (its reader is the warm-prefix flash path),
yet every token paid two casts and two copies per layer to keep it in step. A one-position append
now skips it; the next multi-position append converts the lag from the f32 cache first
(`decode_steps_leave_the_mirror_behind_and_the_next_chunk_catches_it_up`: equal to an eager
mirror, bit for bit). `SWARMLLM_GQA_DECODE_FLASH=1` — the one decode path that reads it — keeps it
eager; `SWARMLLM_KV_MIRROR_EAGER=1` restores the old behaviour for an A/B.

**Measured** (Qwen2.5-Coder-7B Q4_K_M, RTX 3070 Laptop, one binary per row, 3 prompts × 256):

| | tok/s | launches/layer |
|---|---|---|
| before | 46.7-48.7 | 27.2 |
| decode kernel + lazy mirror | **48.3-51.0** | 23.2 |
| llama.cpp, same binary/card/file | 57.5-57.7 | — |

The kernel ALONE did not move the 7B (47.0-48.4 either way, `SWARMLLM_DECODE_ATTN=standard`) —
the synchronised stage profile had overstated attention's share. Replies: each path repeats itself
byte for byte; new vs old diverge at near-ties (2/3 prompts). Against llama.cpp: 729/737 rank-1
(worst rank 3, gap 0.34) vs 728/739 (worst rank 2, gap 0.10) — the same agreement.

**Where the rest of the gap to llama.cpp is.** `examples/qmatvec_card_bench.rs` queues one token's
weight products (Qwen2.5-7B shapes, per-layer K/V so they cannot sit in L2) and synchronises once:
**16.9 ms** — llama.cpp's WHOLE token is 17.4. Big matrices run at 300-310 GB/s (~69% of the
card's 448), q/o at 211, k/v are launch-bound at ~22 µs each, the output head alone is 1.55 ms.
llama.cpp's current Q4_K `vec_dot` is the same algorithm (a branchless scale unpack is the only
change) and its batch-1 geometry is the same (4 warps, one row per block), so its advantage is
between kernels: fused gate+up+GLU and bias in the matvec, and CUDA graphs. Ours confirms it — a
graph step shows `recording_ms_per_launch` ≈ 4.5 ms against a 20.8 ms cycle, i.e. **~16 ms of card
time** against ~19.8 op by op; the recording is what eats the gain. Next: record layer groups and
launch each as it is recorded, so the card never waits for the recording.

**From the rules file (moved 2026-10-02):**

**`decode_attn::gqa_decode_attention_cuda`** (`kernels/decode_attn.cu`, flash-decoding: one
block per KV head and 64-position chunk, every query head of the group, a combine kernel past one
chunk) answers a one-position attention on a card, from `standard_attention` — the card's twin of
the CPU kernel, same `SWARMLLM_DECODE_ATTN=standard` A/B. **And a one-position append leaves the
f16 mirror behind** (`LayerKv::append`); the next multi-position append catches it up, and
`flash_operands` declines a lagging mirror quietly (`SWARMLLM_KV_MIRROR_EAGER=1` = old). Together
−4 launches per layer: Qwen2.5-Coder-7B 46.7-48.7 → 48.3-51.0 tok/s. ⚠ Not bit-identical to the
matmul path — judge replies against llama.cpp.

**Where a card-bound 7B's token goes** (`examples/qmatvec_card_bench.rs`): the weight products
ALONE take 16.9 ms — llama.cpp's whole token is 17.4, with the same Q4_K kernel (the current
`mmvq` differs only in a branchless scale unpack). The gap is BETWEEN kernels: a graph runs the
7B step in ~16 ms of card time but records for 4.5 ms first (`recording_ms_per_launch`).

→ `docs/invariants/inference.md` § "A decoded token's attention on a card is one kernel"

## A card keeps its KV cache as f16 (2026-10-09, FUTURE_WORK #194)

**What.** On a card, a layer's KV cache is f16 BHSD and nothing else — `KvStorage::F16`, the "half
cache" — where both of the card's attention kernels read it in place: flash (a prompt chunk, a
speculative check) through a `transpose(1, 2)` VIEW of the buffer, no copy, and the decode kernel
(`kernels/decode_attn.cu`, `decode_attn_f16kv`), which widens each element to f32 as it reads and
sums in f32 as before. **`layers::kv_storage`** is the one rule (`KvStorage::{F32, F32Mirrored,
F16}`): the processor keeps f32; a card keeps the half cache when the flash kernel is built in and
`decode_attn::card_kernel_covers` the head geometry (d a multiple of 32 up to 256, at most 16 query
heads per KV head); anywhere else the layout before it (f32, plus the f16 flash mirror for a GQA
model). MLA is always f32.

**What it replaced, and the reasoning that kept it.** Until 2026-10-09 the card kept f32 BHSD plus
the f16 BSHD mirror — 6 bytes an element, 3x llama.cpp's 2 — and an agent's 14k-token prompt took
~5 GB beside a 3B's weights. The f32 copy was kept on a reading of arXiv 2604.15409 ("The Illusion
of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference") as "f16 KV
diverges". **The paper compares cache-ON with cache-OFF decoding, both in f16** — a different
accumulation order, not a different storage width — and the reference every reply here is judged
against, llama.cpp, stores f16 by default (`--cache-type-k/-v f16`); vLLM and TensorRT-LLM store
the compute dtype. So the half cache moves us TOWARDS the reference.

**Every path writes through `layers::append_to_cache`** (all four forward sites: dense, batched,
MLA, Qwen 3.5's attention layers), which builds a fresh cache the way the layer keeps it
(`LayerKv::with_storage`) and conforms one that arrived from prefix-cache hydration
(`LayerKv::conform_to` — a snapshot carries `n_kv_head` but not `n_head`, nor the device's choice;
a wire snapshot is f32 and is converted once, O(history)). The f32 K/V a step produces is written
into the half cache in ONE launch (`split::kv_write::write_into` → `kernels/kv_append.cu`), the
launch count `slice_set` had for the f32 cache; candle's cast + copy would have been one more per
tensor per layer per token. A half cache reaching the matmul path (forced-standard attention, a
decode step with a mask) is widened per call — correct, O(history), and never on a default path.

**The rule takes the attention kind as a required argument** (`layers::KvAttention::{Standard,
Mla}`, from `ModelArch` via `KvAttention::of`): its first version took head geometry only, and the
review before it shipped found admission and the coordinator's peer bound pricing DeepSeek-2's
cache as the half cache (a V2-Lite shape — 16 heads, key width 192 — passes the decode kernel's
geometry) while the loader and `forward_mla` kept it f32. Admission's inputs now carry it
(`VramFootprintInputs::attention`), the scheduler reads it off the header's architecture.

**Every estimate asks the same rule**: the loader's KV budget, the card/processor split planner
(`process_pool`), admission (`vram::card_kv_storage` — the GPU estimator's KV term at the storage's
width), and the coordinator's bound on a PEER (`scheduler::KvDevice` →
`layers::kv_storage_as_advertised`, which reads `features::KV_HALF_ON_CARD` — bit 19, NOT in
`features::ALL`, ORed in by the health monitor when `kv_half_on_card_enabled()`; a peer without it
is priced as f32 + mirror, as before).

**Measured** (RTX 3070 Laptop 8 GB, Windows up ~39-40 h, `examples/prefill_bench.rs` and the real
daemon, A/B inside ONE binary via `SWARMLLM_KV_F16=0`, `~/swarmllm-kv194/`):

| | f32 + mirror | half cache |
|---|---|---|
| Qwen2.5-Coder-7B cache, 896 / 6000 positions | 176 / 1057 MB | 59 / 352 MB |
| its KV budget at a 32768 context (loader's own line) | 12,721 tokens | 38,164 tokens |
| bench, context 32768: longest prompt admitted | < 10,000 (refused) | 26,000 |
| decode ms/token at ~960 / ~6016 cached (7B) | 16.7-16.8 / 23.3-23.5 | 16.8-16.9 / 23.3 |
| prompt pass, stage profile synchronised, 6000 tokens: attention | 533 ms | 528 ms |
| greedy 400-token replies scored against llama.cpp (rank-1) | Llama-3.2-3B 399, Coder-7B 396, Phi-4-mini 392, GLM-4 388, Gemma-2 387, Phi-3.5 385 | 399, 396 (both byte-identical), 392, 388, 384, 383 |
| Phi-3.5 logits, same 202 tokens teacher-forced on the card (62 prompt + 140 decode): median cosine to llama.cpp / top-1 | 0.999821 / 191 of 201 | 0.999828 / 192 of 201 |

Card tests: `cuda_decode_kernel_matches_the_matmul_path` runs every shape over both cache dtypes
(< 1e-5 abs against the matmul composition over the same rounded values);
`the_cards_converting_write_matches_cast_then_slice_set` (bit for bit);
`flash_reads_the_half_cache_in_place_as_it_read_the_mirror` (bit for bit — same values, same
kernel, other strides). The processor tests of the bookkeeping go red with the conversion disabled.

**Traps met measuring it.** (1) A prompt-pass bench that does NOT reserve the cache grows it by a
quantum per chunk — a fresh card allocation each — and on this host at ~40 h uptime that growth,
not the forward, is what it measures: 244-300 tok/s on a first repetition in BOTH arms, ~1200 once
reserved. `prefill_bench` now reserves the prompt as a worker does (`SWARM_BENCH_RESERVE=0` to
grow). (2) Even reserved, single repetitions of either arm fell to ~400-800 tok/s at random — host
state, not the change (the synchronised stage profile is level). (3) A Phi-3.5 reply scored a 3.4
gap on a `##` token in the half arm only: teacher-forced on the same tokens, BOTH arms and
llama.cpp rank it second by 3.4-3.8 — a SentencePiece re-tokenization artifact of the scorer, and
the reason to compare LOGITS on one token sequence when a reply score moves. Give llama.cpp the
node's context (8192) for a LongRoPE model, or it runs the short factors (the scorer's own note).

**Qwen 3.5 split across card and processor (gate .233 step 12w, 2026-10-10).** The first gate to run
12w since the half cache shipped (.232's skipped its card steps) failed the 0.8B Q8_0 at 12 card
layers on two of three prompts, 0.064 and 0.090 over llama.cpp's first choice, where .230/.231 read
0.000 / 0.000 / 0.012. One binary pinned it: `SWARMLLM_KV_F16=0` gave back .231's numbers exactly,
`SWARMLLM_QMATMUL_CUBLAS_ACC=32` (#147) changed nothing, .232 failed and .231 passed alike
(`~/swarmllm-gate-0233/q35_ab.out`). Not a defect: each failure was ONE position whose two leading
tokens llama.cpp itself puts 0.06-0.09 apart, and over 8 prompts × 120 tokens the half cache is no
further from llama.cpp than the f32 one — 0.8B 5 of 497 positions over 0.05 (worst 0.119) vs f32 7
of 493 (worst 0.335); 4B Q4_K_M 5 of 523 (0.259) vs 3 of 521 (0.211). The reference also stores f16
(llama.cpp CPU, default KV type); the kernels that read it differ. The DeltaNet recurrent state is
f32 on both sides (llama.cpp hard-codes a hybrid's `recurrent_type_*` to F32) and never takes the
half cache. **Three prompts cannot tell a near-tie flip from a drift; judge a precision change on a
few hundred positions** (`~/swarmllm-gate-0233/q35_wide.py`).

**What a change must keep.** One rule for the cache and every estimate of it — a budget that
charges another width than the cache holds is #104/#447 again. The half cache only where the decode
kernel takes every one-position call, or each token widens the history. The flash call reads the
cache in place: a `.contiguous()` there is the O(history)-per-chunk copy the mirror existed to
remove. A/B: `SWARMLLM_KV_F16=0` (old layout; the node also stops advertising bit 19),
`SWARMLLM_KV_WRITE=compose`.

## A decode step's graph is recorded in groups, launched as it is recorded (2026-09-30)

The whole-step graph above cut ~4 ms of card-side gaps from a 7B token and then spent the same
recording it with the card idle (`recording_ms_per_launch` 4.5 of a 20.8 ms cycle). So the forward
is now recorded in groups of `cuda_graph::group_layers()` layers (default 2,
`SWARMLLM_CUDA_GRAPH_GROUP`, 0 = whole step): at every group boundary the forward's per-layer hook
(`LayerEndHook`, `SplitModel::forward_inner_body`) resolves the residual stream into one of two
buffers made outside every capture (`DecodeGraph::boundaries`, alternating), and `Cutter::cut` ends
the capture, updates that group's own exec and launches it, then begins the next. Only the card's
first group waits for recording. A refusal after some groups ran is safe for the same reason a
refusal was before: those groups wrote this position's KV, the host-side lengths are cut back, and
the ordinary rerun writes the same positions again.

Measured 2026-09-30 (one binary, interleaved arms, isolated node, greedy, 3 prompts × 256; tok/s):

| model | ordinary | 1 layer | **2 layers** | 4 layers | whole step |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 126-135 | 222-230 | **226-229** | 217-221 | — |
| Llama 3.2 3B | 80-90 | 105-106 | **106** | 103 | — |
| Qwen2.5-Coder-7B | 49-51 | 56.5-56.7 | **56.2-57.2** | 55.9-56.2 | 46.1-47.9 |
| Llama 3.1 8B | 50-51 | 54.7-55.1 | **54.8-54.9** | 54.5-54.8 | — |

Replies byte-identical to the ordinary path in all 12 graph arms (the boundary add is the same
IEEE add the fused norm does); 0 refusals. llama.cpp in the same binary on the same card and file:
57.5-57.7 on the Qwen 7B — parity.

**On a split too** (2026-09-30, `~/swarmllm-graph-0929/split_graph.sh`): Qwen2.5-Coder-7B over two
nodes on the one card (`split_rig.sh repeat`, REPEAT=10, two interleaved rounds, median tok/s of
the standard loop): legacy stream 27.7, own stream without graphs 27.3, **own stream + graphs
35.7 (+29%)**. The own stream alone costs a split nothing; each segment's forward is captured on
its own node (the pre-embedded path goes through the same entry). ⚠ A two-request run cannot see
this (the default-flip gate's first split step read "no change" from two samples).

## A speculative check is captured too, and a layout is uploaded once (2026-09-30)

**Measured on an emulated link** (`SWARMLLM_TEST_TENSOR_DELAY_MS`, one machine, Qwen2.5-Coder-7B
split 4 + 4 shards over two nodes on the one card, Qwen2.5-0.5B **Q8_0** drafter,
`~/swarmllm-link-0930/sweep.sh`, tok/s = completion tokens / decode time):

| round trip | plain split | speculation, 09-29 main | + drafter single steps + median cost | + captured checks + layout cache |
|---|---|---|---|---|
| 0 ms | 50 | (fp16 drafter on the processor: 45-50, γ 0) | 35-42 | **44-53** |
| 10 ms | 31 | — | 31.5-33 | **38.7-41.4** |
| 24 ms | 21 | 21-22 (γ 0) | 26.5-28 | **30.1-32.7** |
| 50 ms | 12 | 17-18 | 19 | **21.4-22.3** |

What each step found, from the debug timeline of one round:
- **The fp16 drafter is held as f32 on the card, ~3 GB for a 0.5B** — it fell to the processor at
  124 ms a guess and the controller rightly stepped aside. A Q8_0 build of the same model fits.
- **A draft call opened with a 2-token uncaptured forward** (the guess the last call never fed
  back + the check's token), 12-17 ms against ~3 ms for a captured step; now single steps on a
  card (`draft_after`, `SINGLE_STEP_READ_MAX`).
- **Capture required the very next position**, so every round's first step after a rollback ran
  uncaptured; now any step of a conversation that has taken one is captured
  (`DecodeGraph::follows_previous_step`).
- **The draft cost was an average**, and two 70 ms calls among 12 ms ones set γ to 0 for the rest
  of the request (a zero-guess round measures nothing) and for ten minutes after; now the median
  of the last 8 (`RecentMedian`).
- **A check of several positions was never captured**: it built a causal mask on the host that
  flash never reads on a card (`SplitModel::mask_is_read`), and candle uploaded a strided layout
  on every copy and broadcast op — 155 host copies in one 3-position check, 22 per two-layer
  group (`CudaDevice::layout_params` keeps each distinct one; the capture's copy counter found
  them). Graph state, boundary buffers and the give-up are per position count.

Correctness: replies of speculation with captured checks score 118/120 rank-1 against llama.cpp
(worst gap 0.035); single-node replies byte-identical to the previous build.

**The drafter's guesses stay on the card between steps** (`SplitModel::argmax_guesses_on_card`,
2026-09-30): an argmax guess (no shared noise) is chosen on the card and fed to the next step
there, every guess and its probability come back in one transfer, and the confidence floor cuts
after the first unsure one exactly as the host loop does (`guesses_chosen_where_the_logits_are_match_the_host_loop`).
The host loop read each 600 KB row back to choose, so recording (~3 ms on a 0.5B) and the card's
work (~2.7 ms) ran in series — steps were ~6 ms apart; now ~3 ms (recording-bound). ⚠ candle's
reductions uploaded their layout every call, and a PAGEABLE upload waits for the stream to drain
first (CUDA's documented pageable-copy behaviour) — the argmax between two steps would have
re-serialised the loop; they read `layout_params` now. A round at a 24 ms emulated round trip
moved only ~10%: its budget is drafting ~12 ms, A's check 11.5, B's check + walk 15.3, two hops
of RTT/2 + ~1.5 ms — the checks and the round trip, not the drafter, are now the long poles.

**From the rules file (moved 2026-10-02):**

A forward of up to `cuda_graph::MAX_POSITIONS` (8) positions is captured like a decode step when
no causal mask is read — on a card, dense layers, flash takes several positions and never reads
it (`SplitModel::mask_is_read`; `run_attention` refuses standard attention over several positions
without one). **Candle uploaded a strided layout on every call** (155 host copies in one 3-position
check on a 7B segment): `CudaDevice::layout_params` keeps each distinct layout on the device, and
**never caches one while the stream is capturing** (graph memory, gone when a capture is thrown
away). Graph state and give-ups are per position count — a check's defect must not turn off the
decode step's graphs. The drafter reads a short catch-up as single captured steps
(`draft_after`), a known conversation's step after a rollback is captured, and the draft cost the
γ choice reads is a median of recent calls (`RecentMedian`). Split speculation at an emulated
10 / 24 / 50 ms round trip: +30% / +48% / +80% over a plain split (was break-even / +30% / +57%).

→ `docs/invariants/inference.md` § "A speculative check is captured too"

## A graph pays only when it is updated — a shape that keeps being rebuilt rests (2026-10-05)

**What was seen.** The 60-second `decode graph` line counts launches that updated every group's
graph in place (`updated_in_place`) against launches where at least one group was built afresh
(`instantiated`). Where a shape keeps changing, nearly every launch was a rebuild, and the
recording time per launch — which the card waits for — went from ~1-6 ms to 12-108 ms:

| where | launched | updated | rebuilt | recording ms/launch |
|---|---|---|---|---|
| .225 gate baseline, 7B first segment on the card, split speculation (`repro767.sh`) | 135 | 30 | 105 | 65.4 |
| the same, other arms | 90 / 256 / 126 / 257 | 19 / 59 / 26 / 59 | 71 / 197 / 100 / 198 | 85-108 / 43 / 100 / 53 |
| its fp16 drafter beside it (one position) | 404 | 403 | 1 | 5.9 |
| live node 2026-10-01, Qwen2.5-14B layers 22-44 served for peers | 183 | 6 | 177 | 12.7 |
| live node 2026-10-01 morning, one position, steady | 942 | 888 | 54 | 1.2 |

A rebuild is `cuGraphInstantiate` (10-100 ms here); an update ~1 ms. On a card-bound 7B a graph
saves ~3 ms a step (§ "recorded in groups": 49-51 → 56-57 tok/s), so a shape rebuilt on most
launches costs far more than it ever saves — and every rebuild is driver work over WSL2's GPU
channel on a card that also drives the desktop.

**What it is NOT.** The suspicion that `cuGraphExecUpdate` accepts a graph whose captured
allocations GREW but keeps the sizes it was built with — which would make a kernel write past its
buffer — was tested and refuted (2026-10-05, `graph_alloc_growth.cu` in that session's scratch,
RTX 3070 Laptop, CUDA 13.0): a capture allocating A (growing each step) then B, updated in place 63
times, never corrupted B; each capture lays B out after A's NEW size; and growth past the graph
pool's 32 MB chunk grew its backing (`cudaGraphMemAttrUsedMemCurrent` 32 → 96 MB). CUDA's guide:
graph allocations have fixed addresses for the life of their NODE, assigned at node creation — a
re-capture makes new nodes, and the update adopts them.

**The rule.** llama.cpp's current one (`ggml_backend_cuda_graph_compute`, llama-cpp-sys-2 0.1.156)
never pays for a graph that will not be reused: a graph whose node properties changed runs
UNcaptured until two calls in a row agree. Ours keeps re-capturing every token (it must — the KV
length changes the arguments every step) and so learns from the driver instead: a number of
positions whose update the driver refused on `REBUILDS_BEFORE_RESTING` (3) of its last
`CHURN_WINDOW` (8) launches rests for `REST_STEPS` (256) of its own steps, run the ordinary way,
then is tried again; each further rest doubles, up to 16×, and `CLEAN_LAUNCHES_TO_FORGET` (64) clean
launches in a row start the rests over. A window, not a run in a row: the review of this change
pointed out that clean launches in between would keep breaking a run while a shape rebuilt on half
its launches still loses ~30 ms a step. A shape that churns costs three rebuilds per rest; a
rebuild now and then (one in 64) never rests; a shape that churned for a moment — 50 of the 54 above
came inside one minute (04:53-04:54 UTC) — gets its graphs back. `DecodeGraph::declines` is
the one question the caller asks (it covers the program-defect give-up too). The first refusal of
each kind is logged at info with the driver's result code and the node type it named
(`why_not_updated`), so the next churn says what changed. Test:
`a_shape_rebuilt_every_launch_rests_then_is_tried_again` (red with the rest disabled).

**Verified on a `--features cuda` build** (2026-10-05, `~/swarmllm-gpu-1005/verify.sh`, safety kit,
live node stopped, one CUDA process: `split_rig.sh repeat`, Qwen2.5-Coder-7B shards 0-3 on the card,
4-7 on the processor, REPEAT=3 — the first request takes the n-gram path, whose checks are
captured). The mechanism fired: `could not be updated in place — rebuilt (first of this kind)
positions=5 why=CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE … CU_GRAPH_EXEC_UPDATE_ERROR at a node of
type CU_GRAPH_NODE_TYPE_MEMSET`, then `keep being rebuilt … positions=5 rest_steps=256`; counters
`launched=217 updated_in_place=212 instantiated=5 rebuilt=3 rested=1 uncaptured=14` against v0.3.225's
`launched=227 updated_in_place=212 instantiated=15 uncaptured=4`. Replies byte-identical to v0.3.225,
3/3; no card fault, no driver event. Speed in THIS shape is unchanged within one sample (n-gram
request 47.0 s → 40.2 s, its prompt pass on the processor node varying by as much; plain requests
31.7 / 31.8 s → 31.7 / 31.5 s): here the churn was mild.

**The cause, found the same day (#221): a memset into memory the captured step allocates.** A
probe (`graph_memset.cu`) captured `cudaMallocAsync` + `cudaMemsetAsync` + a kernel + free, then
updated in place 39 times: refused 39/39 at the MEMSET node — at a CONSTANT size as much as a
growing one — while the same memset into a buffer allocated outside the graph updated 39/39. Each
re-capture gives its allocation a new address, and a memset node cannot be re-pointed to it (a
kernel node can). The memset in a check was candle-flash-attn's `softmax_lse = alloc_zeros(…)`,
which every several-position flash call makes; the one-position step (our own decode kernel) has
none, which is why decode updated and checks did not. With `num_splits = 1` the forward kernel only
writes `softmax_lse` and nothing reads it, so it is allocated uninitialized now (both paths;
upstream flash-attn uses `torch::empty`). After (`verify221b.sh`, same rig): 0 refusals, 225 of 227
launches updated in place, replies byte-identical, 3/3. **What a change must keep: no `alloc_zeros`
(or any memset) on memory a captured step allocates** — a buffer the next kernel fully writes is
`alloc` / `alloc_fully_overwritten`; one that truly needs zeros belongs before the capture. The rest
rule stays as the backstop, and its refusal line names the node type.

**On the shape it hurt most** (split speculation, 7B shards 0-3 + fp16 drafter on the card, B on
the processor, `~/swarmllm-gpu-1005/dsd221.sh`, 4 arms × 3 requests on `703b511d`): the 7B segment
updated 179 of 182 launches in place at 3.8-4.0 ms recording (10-04, same shape: 19-59 of 90-257
updated, 43-108 ms), the drafter 554 of 555; 12/12 replies, 0 driver events. Requests 46.9-51.4 s
(first) and 31.4-32.7 s, against 50-124 s and 32.6-64.5 s on 10-04 — which also ran with ~9 GB less
free RAM, so the time difference is not attributed to this alone.

## A rest is weighed on the card's timeline, and a step's ids are one type (2026-10-08, #233)

**What was seen.** Guess-and-check across a split on one card (the gate's step 12h: Qwen2.5-Coder-7B
shards 0-3 on node A's card, 4-7 on node B's, a 24 ms emulated link, the 0.5B Q8_0 drafter on A,
greedy) read slower since v0.3.226. Bisected on one boot under the safety kit, A/B/A/B
(`~/swarmllm-bisect233/h12_*.sh`, Windows up 17-18 h, 0 driver events), tok/s per request:

| arms | rounds | stream |
|---|---|---|
| .222 / .230 | 28.4-32.0, 31.1-34.9 / 25.5-26.0, 26.3-27.5 | 26.3-28.1, 25.6-27.4 / 20.2-21.8, 21.1-21.7 |
| .224 / .226 | 28.6-33.2, 30.3-34.2 / 25.3-27.1, 22.4-26.1 | 25.3-27.2, 26.6-28.0 / 19.7-21.7, 20.2-22.1 |
| .230, graphs off / on | 22.4-24.5 / 24.0-28.2 | 18.6-20.6 / 20.6-22.2 |

On the slow builds the drafter's FIRST request guessed at 5.4-6.9 ms a guess, as .222 did, and every
later one at 11-17 ms — graphs-off speed. The controller then priced a guess above what it saves,
chose γ = 0 and remembered it (§ "Speculation that does not pay steps aside"), so on the stream arm
requests 3 and 4 did not guess at all. Graph rebuild COST was the first suspect and was wrong:
graphs off read the same slow drafts with no rebuilds. ⚠ The log said "rebuilt" once per run because
that line is first-of-KIND (diagnosis rule 2); logging every refusal at debug counted 400-650.

**Why it rebuilt: one more kernel when the ids were i64.** A TOPOLOGY refusal names no node, so each
refusal now logs the group's node census before and after (`node_census`): group 0 went
`KERNEL 49, MEM_ALLOC 44` ↔ `KERNEL 50, MEM_ALLOC 45`, twice a round. The drafter feeds its
catch-up tokens from the host (`token_tensor`, i64) and its own guesses from the card's argmax
(u32), and `QuantizedRows::forward` casts i64 to u32 — inside the capture, one more kernel and
allocation. A conversation fed both changed its graph's shape at every switch (rebuilt on 0.75 of
launches). The .225 rig's drafter updated 403 of 404 because it sampled on the host, all i64.
Fixed at the choke point every captured step passes: `decode_step_with_graph` makes the ids u32
BEFORE the capture (`cuda_graph::cast_ids_before_capture`; A/B `SWARMLLM_CUDA_GRAPH_ID_CAST=inside`).
Not a global change of the id type — the wire format, the vision marker (−1) and the batch stacking
all carry i64.

**Why it then lost its graph: the rest was tuned on another model.** `REBUILDS_BEFORE_RESTING` (3 of
8) was justified by a 7B check whose rebuild cost 65-108 ms against ~3 ms saved; the drafter's
rebuild is ~6 ms against ~9 ms saved EVERY step, so its rests (256 steps, then 512…) were a pure
loss. A rest is now weighed: every step of a shape that has rebuilt within `SHARE_WINDOW` (64)
launches, or is resting, is timed on the card — two CUDA events a step from a ring of
`CLOCK_PAIRS` (8), read once the card has passed them, never waited for (a drafter queues its
guesses without reading back, so a single pair lost every rebuilt step's time). `Churn::weighed` =
share × rebuilt + (1 − share) × updated, against the ordinary way's median; a rest it says is a
loss is declined, and one begun before any ordinary step was timed ends after `SAMPLES_TO_JUDGE`
(5). The count still decides where a figure is missing, so a 7B check rests as before
(`a_shape_whose_graph_still_pays_is_not_rested`, red with the weighing off). Steps of a shape that
never rebuilds are not timed: an event is a driver call, the thing a graph saves. A/B
`SWARMLLM_CUDA_GRAPH_REST=count`. The old rule's own figures, logged on its rest: `a graph 4.4 ms a
step with 0.38 of launches rebuilt, the ordinary way 6.8`.

**Verified on a `--features cuda` build** (one binary, A/B/C/A/B/C, same rig, 0 driver events):

| arm | rounds | stream | the mechanism |
|---|---|---|---|
| old (`REST=count`, `ID_CAST=inside`) | 25.6-29.2, 24.7-26.4 | 21.5-21.8, 19.5-21.9 | 2 rests a run; γ = 0 from request 2 |
| weighing only (`ID_CAST=inside`) | 32.1-35.8, 30.2-35.2 | 25.8-27.7, 25.7-27.4 | rebuilt 400-650 group launches; rests ended/declined; all 4 requests guess |
| both (defaults) | 31.9-34.1, 23.9-31.1 | 26.8-28.4, 25.3-28.3 | group 0 no longer toggles; all 4 requests guess |

The weighing alone gives the speed back even while the drafter rebuilds on most launches; the id
fix removes those rebuilds. What remains with both is the decode-attention kernel's combine step
appearing once a cache passes 64 positions (`CUDA_CHUNK`): every group +2 kernels, at each
crossing and back when the next request starts below it (~10 launches a 4-request run), which the
weighing keeps from resting. The stream arm accepted the same guesses on every arm (77 of 140, 82
of 139, 81 of 139), which a changed drafter or target would not; the harness keeps no reply text.

## A header and its tensor table describe one upload — compared before a tensor is read (2026-10-03, #156)

**The defect.** A node loading from parts has two descriptions of one file: the
GGUF header (`gguf_header.bin` — every tensor's dtype, shape and offset) and the
manifest's tensor table (which bytes of which part file hold each tensor). Candle
reads the header and asks `ShardReader` for each tensor's bytes at the header's
offset; the reader maps that position into whichever table entry covers it.
Nothing compared the two. When they came from different uploads of one model —
3,808 bytes apart at the .220 gate (gotcha #776) — every read moved inside its own
entry, or into the one before it, and the reader returned shifted bytes rather
than an error. Splits through two peers answered `0000…`, `[PAD152063]` or
"Tensor contains non-finite values" with every part hash matching (2026-10-02).
Parts are hash-checked against the manifest at startup and rescan; the header was
checked against nothing.

**The check.** `split::loader::shards::first_header_disagreement` runs in
`load_from_shards_inner` — the one place every load from parts passes, a node's
own request and every peer's forward alike — on the header the loader has just
parsed: each table entry the parts hold must name a tensor the header has, at the
header's absolute offset, with the header's size (`split::tensor_byte_size`, the
same computation `GgufTensorMeta` and so every table are cut by). The first
disagreement refuses the load as `SwarmError::MixedModelCopy` (503, its own hint,
local-only for penalties). It compares entry by entry rather than re-deriving the
layout, so a table written by an older build of the layout code still loads.
llama.cpp draws the same line on a whole file: `gguf_init_from_reader` refuses a
header whose tensor offsets disagree with the data layout ("tensor '%s' has offset
%…, expected %…") instead of reading the wrong bytes.

**On a peer** the refusal crosses the wire as `PEER_FACING_MISSING_SHARDS` on BOTH
serving paths — `sanitize_peer_facing_error` on a layer forward (its "shard" rule),
`worker_failure_for_the_coordinator` on a whole-model hand-off — and the
coordinator's existing missing-shard path retracts that peer's claim for the range,
bars it for the request and fails over. No new message and no feature bit
(`a_peers_mixed_copy_refusal_reaches_the_coordinator_as_missing_shards`, twice).

**Measured** (`split_rig.sh mixed`, TinyLlama, CPU, 2026-10-03; A holds part 0, B
every part under the planted header, C parts 1+ under the real one; three greedy
asks: control without B, alone without C, both).

| planted header | v0.3.221 "alone" | this build "alone" |
|---|---|---|
| SHORTER (chat template dropped, data 448 bytes earlier) — the silent shape | **200 OK, `给给给给给…`** — one character repeated, the field's "output is a single repeated character" | 503 "No reachable node holds layers 12-21"; B logged the refusal once, A retracted B's claim, both == control |
| LONGER (+3,872 bytes, `PLANT=3808`) — #776's shape | 503 — the file's last tensor `output_norm` ran past the data ("failed to fill whole buffer") | the same 503, preceded by B's refusal |

So a longer header was already an error on either build; the shorter one is what
answered garbage, and is the rig's default. **The rig found a second path the first
cut missed**: a WHOLE-model hand-off (`remote_generate`) sends a worker's error back
verbatim, so B's refusal reached A as a `MixedModelCopy` of A's own — no
retraction, and a caller told "this computer's copy is mixed up" about a computer
that was fine. `daemon::dispatch::remote_generate::worker_failure_for_the_coordinator`
now translates that one refusal at the serving node (which also works for a
coordinator on an older release), and the rig asserts both: A logged the
retraction, and the caller's error does not say "Mixed model copy". Null control on
real copies: `every_real_copy_agrees_with_its_own_header` over the release node's 18
models (MoE, GLM-4, Phi's fused layout, tables written since 2026-09-11) — 18 pass.

**The check above cannot see the other half (2026-10-03, after v0.3.222).** A
header and a table that are BOTH the upload's agree with each other while the part
BYTES beside them were cut from another upload at this one's offsets — #775's
splice, whose leftovers were measured live the same day (one upload's part in three
byte versions on three peers, gotcha #782). Hash checks pass too: a node's manifest
carries the hash of its own bytes. On the `spliced` rig such a copy answered
`给给给…` on v0.3.222. Only `auto_manage::canonical`'s byte check against the upload
sees it, and it now DELETES such parts at once — mid-request if need be — and
re-fetches the upload's (`replace_parts`, `Doomed::WrongBytes`). A load-time refusal
(a marker beside the parts) was built first and removed the same day: with the
parts gone there is nothing to refuse.

**What a change must keep.** Every load from parts goes through
`load_from_shards_inner`; a new shard loader calls the check. Never compare by
re-deriving the layout. The Display of `MixedModelCopy` must keep the word "shard"
(the peer-facing translation depends on it, pinned by the tests above), and a new
path that returns a worker's error to another node translates this refusal too.

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

→ `docs/FUTURE_WORK_ARCHIVE.md` § "A context that does not fit is refused instead of shrunk"

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

## A model whose state cannot be wound back is never speculated

(2026-10-06, FUTURE_WORK #117.) Qwen 3.5's Gated DeltaNet layers (three of every four) carry a
recurrent state per request — a running summary, not a list of positions — so the tokens a rejected
draft wrote into it cannot be taken back. Every speculative path takes a rejection back by
truncating the attention caches; on such a model that would leave the two describing different
histories, a silently wrong reply.

- **Coordinator**: `pipeline::distributed::speculation_can_roll_back(architecture)` is asked ONCE in
  `execute_distributed`, after the delegated-split hand-off and ahead of DSD (stream included),
  single-segment speculative and n-gram-only — "one invariant, N paths", so it sits where all three
  pass. A delegated split's head asks the same on its own node.
- **Worker**: `ngram_spec_eligible` (slot admission AND the decode loop) and SWIFT ask
  `SplitModel::carries_recurrent_state`. ⚠ Slot admission can run before the model has loaded
  (observed 2026-10-06: the first request was admitted to the n-gram path with no model loaded); the
  decode loop's own check, made with the model in hand, is what declines it — keep both.
- **Backstop**: `KvCacheEntry::truncate_to` refuses unless every `ssm_states` entry can be taken
  back to the target — at most the one step a card keeps (`SsmState::can_rewind_to`, #228 (4)) —
  so a path added later that truncates further fails loudly instead of answering from the wrong
  history.
- **Prefix cache**: a snapshot is never taken of an entry holding SSM state, so a second turn reads
  its prompt in full. A prompt pass at position 0 clears the entry (state included), which is also
  how a failover replay rebuilds it on a stand-in.

Allowing speculation would need a snapshot of the recurrent state per draft (#228).

## Qwen 3.5's delta rule is ONE op per layer, on the processor and on a card (2026-10-11, FUTURE_WORK #228 (3))

**What it replaced.** `layers::qwen35::delta_rule` follows llama.cpp's
`build_delta_net_autoregressive` op for op: per token, per layer, a decay
multiply, three matmuls, a subtraction, two broadcasts and a stack — ~10 tensor
ops over a `[heads, 128, 128]` state, each its own pass over 2 MB (the 4B) and
its own fork and join. Three layers in four are DeltaNet layers, so a 681-token
prompt on the 4B made ~160,000 of them.

**What llama.cpp master runs instead** (`GGML_OP_GATED_DELTA_NET`, read at
`4b1a27f`): `fused_gdn_ar` / `fused_gdn_ch` are on by default, so EVERY prompt and
decode step goes through one op per layer. Its CPU body
(`ggml_compute_forward_gated_delta_net_one_chunk`) walks one (sequence, head) per
task, the state in one buffer, the tokens in a plain loop; its CUDA body puts a
warp on each state row and folds the decay into the update.

**What replaced it here.** `delta_rule_cpu`: one rayon task per (sequence, value
head), the state copied once into the output buffer and updated in place. A
token's update of row `S[j, :]` reads only row j, so each row is decayed, read
against k, corrected and read against q in one pass while it sits in L1 —
llama.cpp's CPU body makes four passes over the matrix for the same arithmetic.
Value head h reads key head h % nk (llama.cpp tiles the key heads; the tile copy
is gone) and the 1/√d scale multiplies the output, as the op applies it.
**On a card** the same day: `delta_rule_cuda` → `kernels/delta_rule.cu`, llama.cpp's
`gated_delta_net_cuda` (scalar gate): a warp per state row, each lane `D / 32` of it
in registers for the whole pass, the decay folded into the update as llama.cpp's
card op folds it — ONE launch per layer per forward where the composition made ~10
per token. q and k are L2-normalised as one tensor (`[b, seq, 2·nk, kd]`) for every
path, which the kernel reads as one input; candle's custom ops take three tensors,
so the state rides on the op. The kernel writes the new state into a second buffer
it is handed (`state_out` — the pair the step did not read, never the same buffer:
both are `__restrict__`) and the outputs into their own, so a state is never a view
of a chunk's outputs (until #228 (4) it was, copied out past 8 tokens).
Scope `card_delta_rule_covers`: equal key and value widths of 64 or 128 (every
Qwen 3.5 size has 128). Against llama.cpp master on the card (0.8B F32, 200 tokens, 8 decoded): worst cosine
0.999999, median 1.000000, top-1 200/200 — the same as the composition on the card.

**Same numbers, not the same bits.** The sums run in another order.
`one_pass_delta_rule_matches_token_by_token` bounds it (1e-5 against the
composition, sequence by sequence: the composition cannot take a batch of two;
more value than key heads, unequal widths, a carried state) — red when the decay
is applied after the key read and when value head h reads key head h / (nv/nk).
Against llama.cpp master on Qwen3.5-0.8B F32, 200 tokens with 8 decode steps
(`logits_reference_probe` + `~/llama.cpp-ref/dump_logits` +
`compare_f32_logits.py`): worst cosine 0.999999, top-1 200/200 — identical to the
composition's own agreement; the two differ by 2.7e-5 at most.
A/B: `SWARMLLM_DELTA_RULE=steps`. Profile stage: `delta rule`.

**Measured** (Qwen3.5-4B Q4_K_M, Ryzen 7 5800H, every layer on the processor,
685 tokens read 128 a forward on the processor and 512 on the card, 48 out; live node stopped, safety kit;
`~/swarmllm-138/rows_chunk.sh` § 3):

| | one op (default) | `steps` | llama.cpp master (`llama-bench pp681`) |
|---|---|---|---|
| processor, prompt | 19.8-21.1 s | 55.8-55.9 s | 14.9 s (45.6 tok/s, 8 threads) |
| — its `delta rule` stage | 0.70-0.76 s | 36.7 s | |
| processor, a decode step | 95-122 ms | 127 ms | |
| whole on the card, prompt (512 a forward) | 0.49-0.53 s | 2.56-2.70 s | 0.30 s (2,249 ± 503 tok/s) |
| — its `delta rule` stage | 13 ms | 2.2 s | |
| card, a decode step | 17 ms | 20-22 ms | |

The 48 reply tokens identical in all eight runs, processor and card alike. On the processor the rest of the
prompt is now the projections and the FFN, as for any dense model.

## Qwen 3.5's decode steps are captured too: a DeltaNet state is written into a second pair of buffers (2026-10-11, FUTURE_WORK #228 (4))

**What refused it.** `decode_graph_ineligibility` let only dense layers into a
capture, and `KvCacheStore::every_cache_holds` asked every layer for a KV cache —
three Qwen 3.5 layers in four keep none — so no step of the family was ever
captured. Behind the gate were two real blockers. The DeltaNet state was replaced
by a NEW tensor every step (the conv window by `cat` + `narrow`, the recurrent
state by the kernel's output), i.e. memory allocated inside the capture — the
graph's own, freed or reused on replay, leaving the store holding a dangling view.
And a refused capture re-runs its step: groups launched before the refusal HAVE
run (`DecodeGraph::capture`), which a KV append survives (it rewrites the same
position) and a state updated in place does not — it would take the step twice.

**What llama.cpp does.** Its recurrent memory keeps every layer's states in fixed
cache tensors and writes the new state back with `ggml_cpy` into a view of them —
fixed addresses, which is what lets its CUDA graphs cover these models.

**What replaced it.** `SsmState` keeps, on a card, a SECOND pair of buffers — the
state before the last step. A step reads `current`, writes the new state INTO the
second pair (the kernel through its `state_out` pointer, the conv window with one
`slice_set`) and the two swap (`SsmState::advance`). The state a step read
survives it, so taking the step back is a swap (`SsmState::rewind_to`, one step
only); the executor's failure path and `KvCacheEntry::truncate_to_with` both ask
`can_rewind_to` first and refuse — or drop the state, as before — when it cannot.
The swap is host bookkeeping a refused capture must undo (gotcha #761), and it is.
A recurrent layer holds a captured step once its second pair exists
(`every_cache_holds`); the pair is made by the first step on the card, which is
never captured. On the processor nothing is captured and no second pair is kept.
A step whose state does not describe its first position is refused, never
answered from zeros: a conversation the layer has lost (`forward_deltanet` takes
`index_pos`). Capture admits a Qwen 3.5 DeltaNet layer only on the card kernel
(`DeltaNetWeights::delta_rule_runs_on_the_card_kernel`) and only with a dense
feed-forward — MoE routing still copies scores to the host.

**Cost.** One more state pair per DeltaNet layer per conversation on the card:
2 MB a layer for the 4B and the 9B (48 MB a conversation), inside the
forward-buffer reserve the single pair already relied on (#228 (2)).

**Tests** (each red with its part undone): `a_deltanet_step_taken_back_and_run_again_lands_where_one_run_does`
(red when the rewind does not swap, and when a step writes into the state it read),
`a_recurrent_layer_holds_a_captured_step_only_with_its_next_buffers` (red when the
SSM arm ignores the second pair), `an_entry_with_recurrent_state_refuses_to_be_truncated`
(one step back only), `a_deltanet_step_out_of_step_with_its_state_is_refused`.

**Measured** (RTX 3070 Laptop, live node stopped, safety kit; `~/swarmllm-228/graph.sh`):

| whole on the card | capture (default) | `SWARMLLM_CUDA_GRAPH=0` |
|---|---|---|
| Qwen3.5-0.8B Q8_0 | **121-126 tok/s** | 50-54 tok/s |
| Qwen3.5-4B Q4_K_M | 35.5-37.8 tok/s | 37.9-39.3 tok/s |

1,200 tokens, ABBA; the rate is taken from the first flushed piece of the reply to the last
(`~/swarmllm-228/tok_timing.py` — load, prompt and the one-time rebuilds below left out; the
capture arm's window still starts a piece earlier, so it carries some of them). Wall clock
over whole runs could not resolve it: a load varied by seconds (`graph2.sh`, `graph3.sh`).
The ids are the same with capture on and off — 6 of 6 prompts × 96 tokens (`graph.sh` § 1)
and these 1,200. The capture fired with no refusal; in a 300-token reply on the 4B, 8 of its
16 groups were rebuilt once each (`TOPOLOGY_CHANGED`: one more kernel and one more
allocation in the later step), then updated in place. Against llama.cpp on the card the
prompt tokens are identical and the kernel probe is unchanged (top-1 200/200, worst cosine
0.999999). **The 4B is card-bound**, like Llama-3.1-8B in the dense check (2026-09-29): ~26
ms a step, ~96 GB/s of its 2.5 GB of weights against the card's 448 — where that goes is
FUTURE_WORK #228 (7).

## A mixture-of-experts layer stacked on the card is captured too (2026-10-14, FUTURE_WORK #138 (2))

**What refused it.** `decode_graph_ineligibility` let no MoE layer into a capture ("unchecked").
A layer whose experts are STACKED (`StackedExperts::fit`, `MoeFfn::experts_stacked`) needs nothing
the capture forbids: routing on the card (`topk_on_device`: arg-sort, gather, softmax), the ids
grouped by candle's sort, three `moe_gemm_gguf` launches on the device's own stream, whose one-row
kernel takes its q8_1 scratch with `cudaMallocAsync`/`cudaFreeAsync` — graph memory nodes, which
update like candle's own allocations — and no memset, no host copy. `ffn_capture_ineligibility`
is now the per-layer answer; the card's per-expert fallback (`route`: the scores to the host,
each expert's tokens gathered with ids sent back up) and MLA stay out.

**Checked on a real model's layers.** Nothing here that stacks fits the 8 GB card whole: the
small test MoE files' experts are 400 wide (Qwen 3.5-MoE 0.87B), not a multiple of 32, so their
down projection can only be F16 — which `fit` refuses — and Qwen3-30B-A3B is 11.3 GB at Q2_K.
So `~/swarmllm-138f/q3moe-24L.gguf` is that Q2_K file's first 24 layers (`truncate_gguf.py`:
`block_count` rewritten, the rest copied) — the same weights, routing and dtypes, 5.8 GB, its
text not meant to be good. `moegraph2.sh`, `test-split` greedy, 6,000 tokens, two interleaved
rounds, live node stopped, decode time = wall less the load read off the log (`steps2.py`):

| arm | ms a token | tok/s |
|---|---|---|
| `SWARMLLM_CUDA_GRAPH=0` | 14.75 / 15.53 | 67.8 / 64.4 |
| capture, 2 layers a group | 12.20 / 12.59 | 82.0 / 79.4 |
| capture, 1 layer a group | 12.43 / 12.69 | 80.5 / 78.8 |

The ids are the same in all six (`c1e1e775`). The capture fired: 4,874-5,101 launches a run, all
but one updated in place (one `TOPOLOGY_CHANGED` rebuild at the start), 0 refused, 10-11 steps
uncaptured (a conversation's first). Recording costs 4.2-5.0 ms a step on the host, overlapped
with the card's previous group.

⚠ **A reply hashed with the log in it is not a reply.** `test-split` logs to stdout, and the
capture's decision line lands INSIDE the streamed `Generated:` line — so two identical replies
hashed differently and "the captured MoE output differs run to run" was briefly believed (gotcha
#840). Compare the `Token IDs:` line, or strip timestamped lines first.

## A sliding window is cut where it binds (2026-10-15, FUTURE_WORK #278)

Gemma 2 alternates two kinds of attention layer: the even ones see only the last
4,096 positions, the odd ones the whole conversation. The loader read Gemma 2's
softcaps and extra norms but never its window, so every layer attended the whole
conversation. Under 4,096 tokens the two are the same computation, and every
reference check ever run on Gemma 2 was shorter than that (gotcha #863). Measured
on gemma-2-2b-it Q4_K_M, `examples/logits_reference_probe.rs` +
`compare_logits_reference.py` (`~/swarmllm-swa/probe.sh`: ARCHITECTURE.md tokenized,
read in 512-token chunks, the prompt's last 8 positions + 4 decode steps):

| prompt | window | worst cosine | median | top-1 |
|---|---|---|---|---|
| 4,600 | not read (before) | 0.98587 | 0.99563 | 10/12 — DISAGREES |
| 4,600 | read | 0.99939 | 0.99963 | 11/12 — agrees |
| 2,000 | not read (control) | 0.99914 | 0.99974 | agrees |
| 4,600, on the CARD | read | 0.99955 | 0.99973 | 12/12 — agrees |
| 4,600, on the CARD | `SWARMLLM_SLIDING_WINDOW=0` | 0.98503 | 0.99591 | 10/12 — DISAGREES |

The card rows are one CUDA build (`--features cuda`, flash prefill with `window_size_left` and
the decode kernel over the narrowed cache), both arms, run under the safety kit
(`~/swarmllm-swa/card.sh`, 0 driver events). The 2,000-token logits are byte-identical with and
without the change; the processor's 4,600-token `=0` arm is byte-identical to the build before it.

**The rule is llama.cpp's** `LLAMA_SWA_TYPE_STANDARD` (`llama_hparams::is_masked_swa`):
a key at `k` is hidden from a query at `p` when `p - k >= n_swa`. Which layers:
`load_swa_pattern(ml, 2)` → `set_swa_pattern(2)`, layer `il` windowed when
`il % 2 < 1`, unless the file's `{arch}.attention.sliding_window_pattern` says
otherwise (an integer period, or one flag per layer). Read at `4b1a27f`,
`src/models/gemma2.cpp`. Phi-3 declares a window llama.cpp switches off
(ggml-org/llama.cpp#13676), so it is not read here either; Llama 4's chunked
attention is a different rule (FUTURE_WORK #114 (b)).

**How it is applied:**

- `split::SlidingWindow::read` is the one reading; the loader sets each layer's
  `LayerWeights::sliding_window` from `for_layer(il)` (`il` in the WHOLE model,
  so a segment starting at an odd layer gets the right parity).
- `LayerWeights::keys_in_window` is the one place a window meets the cache, for
  both attention entry points (`attend`, `forward_attn_batched`). It narrows the
  K/V views (BHSD dim 2) and the f16 mirror (BSHD dim 1) to the window's start —
  a view, no copy. The start comes from the cache's LENGTH (its last `seq_len`
  positions are the queries), the same reading the forward's masks are built on
  (`kv_offset`), so the two cannot disagree. Since #281 the cache itself drops
  what the window no longer reads (`held_from` is where it starts) — next
  section.
- One query (decode) needs no mask: the narrowed view IS the window. Several
  read `ForwardMasks::for_layer` — the causal mask and the windowed one over the
  narrowed keys, built once per forward by `SplitModel::attention_masks`,
  moved between devices together (a card/processor split, a processor layer
  run on the card).
- The card's flash arm gets `window_size_left = W - 1`, right 0, over the
  narrowed keys (the vendored kernel aligns bottom-right). Every kernel honours a
  view's start offset: `decode_attn` on both devices, `prefill_attn` on the
  processor, candle-flash-attn.
- **A window that does not bind is the old path, bit for bit**
  (`window_start` answers `None` while the last query still sees position 0).
  Pinned by `a_window_that_keeps_nothing_out_changes_nothing`.

**The reference in the tests is independent of the implementation**: RoPE makes a
query's score depend only on its distance to the key, so on a one-layer model a
windowed layer's output at `p` equals the last position of a run over just the
`W` positions ending at `p`, started from 0 (`a_windowed_layer_reads_only_its_window_on_every_path`,
one pass, pieces of 4 and 3, one at a time; it fails by 0.045 with the window
off). `windowed_and_global_layers_each_read_their_own_mask` composes the two
kinds of layer, `a_batched_forward_reads_the_window_too` covers the batched path,
`the_window_mask_is_llama_cpps_standard_rule` checks the mask cell by cell.
Against llama.cpp at a window that binds within a few tokens:
`examples/make_tiny_gemma_gguf.py` (window 8, f32 weights) agrees at worst cosine 0.99928
over 40 positions, whole, in pieces and split at layer 3; 0.334 with the window off.

The cache keeping only the window came after (#281, next section).

## A windowed layer's cache keeps its window (2026-10-16, FUTURE_WORK #281)

Until #281 the window was cut from the KEYS a query read, not from memory: every
windowed layer kept every position. On Gemma 4's 12B (40 windowed layers at 8 KiB a
position, 8 global at 2 KiB, f16 K+V) a 32K-token conversation held 10.5 GiB of cache;
llama.cpp holds ~0.8.

**What llama.cpp and vLLM do.** llama.cpp gives windowed layers a second cache
(`llama_kv_cache_iswa`) of `GGML_PAD(min(size_base, n_swa * n_seq + n_ubatch), 256)` cells
— the window plus one batch; `--swa-full` keeps everything, and without it a prompt the
server cannot reuse is read again (its checkpoints, `--ctx-checkpoints`, exist to soften
that). vLLM's `SlidingWindowManager` frees blocks that fall out of the window and accepts a
prefix hit of length L only when positions [L − W + 1, L) are still cached. Both rules are
taken whole.

**How it is done here:**

- `SeqCache` keeps `start` — the position its buffer's first row holds — and `keep`. A full
  buffer holding more than `keep` moves its newest `keep` positions to the front
  (`kv_write::write_into`, one copy) instead of growing. `current_seq_len` stays the
  conversation's length, so every reader of "how far has this request got" is unchanged;
  `k()` / `v()` return the rows from `held_from` on.
- `keep = positions_kept_for_window(W)` = W − 1 + `WINDOW_REWIND_POSITIONS` (64): the keys the
  next query reads, plus room for every rewind this codebase makes — a refused capture, a
  rejected draft (≤ 8), a split's stored prompt cut at a whole 64-token block. The buffer
  grows no wider than `keep` + one quantum (512): the size whose slide always leaves a whole
  quantum free. Quantized growth without that cap settled one quantum wider (512 → 1,024 →
  1,536 for Gemma 3's 512 window, where 1,087 suffices).
- `LayerKv::slide_beyond` is told the window by EVERY forward (`layers::append_to_cache`,
  beside `conform_to`): a cache restored from a snapshot cannot know its layer's window. A
  cache with an f16 mirror (`F32Mirrored`) never slides — the mirror is kept in step by
  absolute position against the whole history.
- `keys_in_window` takes `held_from` and narrows at `start − held_from`; it `bail!`s if the
  window would start before the cache does — a loud error, never a read of the wrong rows.
- **A rewind is refused past `LayerKv::serves_from`** (`held_from + W − 1`):
  `KvCacheEntry::truncate_to_with` checks every layer first. Callers that can recover do
  (`reconcile_hydrated_prefix` → a full prompt pass).
- **The prefix cache**: a snapshot records `held_from` per layer and `serves_from`;
  `snapshot_at` copies [held_from, pos), `narrow_snapshot` refuses below `serves_from`,
  `lookup` / `lookup_chain` skip an entry whose `serves_from` passes the usable length, and
  an earlier turn shorter than the new snapshot's `serves_from` is KEPT beside it (it serves a
  branch the new one cannot — llama.cpp's checkpoints, for the same reason). Hydration
  starts the cache at `held_from` (`begin_at`). `bytes_per_position` — what the snapshot
  ceiling divides the byte budget by — divides a layer's views by the positions it HOLDS: by
  the conversation's length a slid layer read at a quarter of its weight
  (`a_slid_layer_weighs_the_same_per_position`, 40 bytes against 64 with that off).
  **A slid snapshot never leaves the node**:
  the wire format (SKVX v1) means "every layer from position 0", and a peer on an older build
  would read a partial layer as the opening — `Entry::manifest` is empty for it (not
  announced, not exported) and `serialize_snapshot_with_block_size` refuses it.
- `every_cache_holds` and `allocated_positions` read `SeqCache::reach` (`start` + buffer):
  a step inside it allocates nothing, so a CUDA graph may capture it; the step that slides
  is beyond it and runs uncaptured.

**Tests, each red with its part of the fix off** (2026-10-16, a mutation per part):
`a_windowed_cache_keeps_its_newest_positions_in_one_buffer` (the growth cap),
`a_slid_cache_takes_a_rewind_its_window_still_answers_and_refuses_one_past_it`,
`a_slid_snapshot_is_restored_where_its_window_starts_and_stays_on_this_node` (hydration),
`an_earlier_turn_a_slid_snapshot_cannot_serve_is_kept_and_found` (both the lookup skip and the
retention). `a_windowed_cache_that_slid_answers_as_one_that_kept_everything` runs a two-layer
Gemma 2 test model 640 positions past a 5-position window — prompt pieces, decode steps and a
batched step — with the slide on and off (its control asserts the off arm slid nothing).

**Against llama.cpp, on the real files** (`~/swarmllm-g3/slide281.sh`: Gemma 3 Q4_K_M on the
processor, the prompt read in 512-token chunks, the last 8 prompt positions + 4 decode steps,
`examples/compare_logits_reference.py`), the slide on and `SWARMLLM_KV_WINDOW_SLIDE=0`:

| model, prompt | windowed layers | slides (DIAG) | on vs off | vs llama.cpp (both arms) |
|---|---|---|---|---|
| 1B, 1,500 (window 512) | 22 of 26 | 44 / 0 | bit-identical | worst 0.999451, top-1 12/12 |
| 4B, 2,000 (window 1,024) | 29 of 34 | 58 / 0 | bit-identical | worst 0.998622 at 1,998, top-1 12/12 |

Both rows equal the runs from before #281, to the digit (the 4B's one position under 0.999 is
#115 (1)'s, unchanged). The slide count is the DIAG line's `windowed_slides`, one per K and
per V: 22 and 29 layers each slid once in the last prompt chunk.

**On the card** (`~/swarmllm-g3/card281.sh`, the CUDA build, under the safety kit, 0 driver
events): the half cache slides too. The 1B whole on the card with 64 decode steps slid 44 in its
last prompt chunk and 44 more on the decode step at position 1,473 — the one step CUDA graphs
leave uncaptured (`every_cache_holds` reads `reach`) — and its 72 logits are bit-identical with
the slide on and off. The 4B whole on the card (worst 0.994008) and split 17/34 card/processor
(0.994389), 58 slides each, equal the card run from before #281 to the digit. The 1B's 64-step
run sits at worst 0.997727 against llama.cpp on the processor with 5 top-1 flips in 72, every
one at a cosine ≥ 0.9996 with the top 5 agreeing — near-ties, in both arms alike.

The estimates came next (#281 (2), next section). A/B `SWARMLLM_KV_WINDOW_SLIDE=0`.

## A windowed layer is charged as its window — in the worker (2026-10-16, FUTURE_WORK #281 (2))

After (1) the cache held its window, but every estimate still charged each position on every
layer at the WIDEST layer's rate (`kv_bytes_per_token` × layers): Gemma 4's 12B at 32K tokens
charged 48 × 8 KiB × 32,768 = 12 GiB for ~1.0 held, so a card with 2 GiB of KV room refused a
prompt past ~5K tokens and reported that as its context.

**What vLLM and llama.cpp charge.** vLLM's `SlidingWindowSpec.max_memory_usage_bytes`: a
sliding-window layer needs `cdiv(sliding_window − 1 + max_num_batched_tokens, block_size) + 1`
blocks whatever the context — the window plus the widest batch. llama.cpp sizes its iSWA cache
the same way (`n_swa + n_ubatch`). The term that is easy to miss is the BATCH: a forward wider
than the growth quantum widens the buffer it lands in (`growth_step` lets a windowed buffer
reach `keep + seq_len`), so a bound built from the window alone is wrong for a one-shot prompt.

**How it is done here:**

- `kv_budget::KvGrowth` is the worker's ONE price of a conversation: `bytes_for(n)` =
  `every_position · n + windowed · min(n, cap)`, cap = `most_positions_a_window_holds(W,
  quantum, widest_forward)` = `positions_kept_for_window(W)` + max(quantum, widest forward) —
  what `SeqCache::append` can ever leave a windowed buffer at. `positions_within` is its
  inverse (pinned by `the_positions_a_budget_holds_invert_what_they_cost`). `plus` / `share`
  combine ranges (a merged window takes the widest).
- The loader builds it PER LAYER (`kv_growth_over`): `split::kv_layers` (recurrent layers keep
  none), each layer's own shape (`attention_shapes`; MLA's elements), its storage
  (`kv_storage`), and whether its cache slides — `kv_cache::window_kept(window, storage)`, the
  same predicate (`caches_slide`) `LayerKv::slide_beyond` asks, so the estimate and the cache
  cannot disagree about a mirrored cache (which never slides).
- The widest forward a model is handed is `SplitModel::widest_prompt_forward`: the worker's
  `prefill_chunk_tokens` (`split::PROMPT_CHUNK_TOKENS`, set once in `run_worker`) through
  `prompt_chunk_ceiling` — 512 on a card, up to `COPIED_CHUNK_MAX_TOKENS` on a split copying
  its weights, the context for 0 (one forward). `SplitModel::kv_budget` widens the growth to
  it. A forward this node does not size — a coordinator's piece of a prompt — is admitted at
  its own width (`ensure_room_for_prompt`'s `forward_width`).
- Readers: the loader's affordable tokens (its log lines' `kv_bytes_per_position` is the
  figure while no window is full), admission (`admit_prompt_over` → `admit_prompt_in`
  promises `bytes_for(positions)`), the snapshot plan (`plan_snapshot`), the context reported
  (`context_fits_in`, `longest_prompt_that_fits`), `host_ram_positions`,
  `positions_beside_live`.
- **The head-room guard reads the buffers, not the estimate**: `KvCacheStore::bytes_a_forward_adds`
  sums `LayerKv::bytes_added_by` — `SeqCache::positions_added_by` simulates `append` (slide,
  `growth_step`, a first allocation) at the dtype the buffer holds, and the f16 mirror as
  `append` treats it (nothing for a lazy decode step; caught up and grown on a chunk; built
  `initial` wide only where `wants_shadow`). The estimate answers only the first forward,
  before any buffer exists. Before, the claim was per-token × seq_len at the widest rate, a
  mirrored cache priced at 6 bytes an element whether or not a mirror existed.
  **A claim of 0 never meets the budget**: the guard's gate (`positions_to_allocate` over the
  least `reach`) opens at every windowed buffer's end, where the cache slides in place, and
  `claim_room` refuses even a zero claim once memory is over budget — a slide would have
  ended, and cleared, a conversation because another program took memory (found in review;
  `a_forward_that_only_slides_a_window_is_never_refused_for_room`, refused at position 576
  without the skip). The device query (`kv_budget_now`) is asked only for a real claim.
- **Hydration builds each layer as it keeps its cache** (`SplitModel::cache_layouts` →
  `CacheLayout { storage, window }`): a long snapshot taken before a slide — a peer's, or one
  kept with sliding off — is restored to a windowed layer as its newest `keep` positions,
  never rebuilt whole into memory the estimates say it does not hold.

**Tests, each red with its part off** (2026-10-16): with `growth_step`'s cap and
`bytes_for`'s `min` both off, six go red —
`a_forwards_claim_is_what_its_append_allocates` (a three-layer cache — windowed, global,
mirrored — through 40 forwards of mixed widths at three reservations: every claim equals the
bytes the append allocated, and the windowed buffer stays inside its cap and the global
layer's buffer), `a_windowed_cache_keeps_its_newest_positions_in_one_buffer`,
`a_windowed_layer_is_charged_no_further_than_its_window`,
`the_positions_a_budget_holds_invert_what_they_cost`,
`a_windowed_snapshot_is_planned_at_what_it_copies`,
`a_windowed_model_fits_the_conversation_its_windows_leave_room_for`. With the mirror's
pricing off the claim test fails at the first chunk that catches the mirror up (16,384 bytes
allocated, 0 claimed). `a_long_snapshot_is_restored_to_a_windowed_layer_as_its_window` has
its own control (no layouts: the layer is rebuilt whole). `positions_within` answered the cap
for a budget that held positions past it (`<=` where `<` belongs; 959,500 bytes → 1,599, not
1,600) — the inverse test now sweeps the cap in steps finer than a position.

**On the real files, on the card** (`~/swarmllm-g3/card281b.sh`, the CUDA build, under the safety kit,
0 driver events): accounting only, so the logits of the 1B (1,500 + 64 decode steps, 88 slides), the 4B
whole on the card and the 4B split 17/34 are bit-identical to (1)'s card runs (max |d| = 0.0). The
loader's figure moved as the arithmetic says: the 4B on the same card (7,387 MB free) covers **190,957**
tokens of conversation against **29,445** with `SWARMLLM_KV_WINDOW_SLIDE=0` — 29 windowed layers held at
1,599 positions (window 1,024: keep 1,087 + one 512 quantum), the 5 global layers growing:
(1,001,130 − 29 × 1,599) / 5 = 190,952 positions, where 1,001,130 = 34 × 29,445 is the budget in
one layer's positions — within 5 of the figure logged (rounding to whole positions).

## A family's attention scale is read, not assumed (2026-10-15, FUTURE_WORK #279)

Every attention here scaled its scores by `1/sqrt(head_dim)`. llama.cpp does that for most
families, but Gemma 2's 27B — the size it recognises by 46 layers — scales by
`1/sqrt(n_embd / n_head)`: 1/sqrt(144) against a head of 128 (`llama_model_gemma2::load_arch_hparams`
at `4b1a27f`, after gemma_pytorch's `config.py`; Gemma 3 has the same rule at 62 layers). The
9B is the trap for a reader who generalises: its `n_embd / n_head` (224) also differs from its
head (256), and llama.cpp scales it by the head.

`split::query_scale` is the one reading; the loader sets `LayerWeights::query_scale` to
`sqrt(head_dim / (n_embd / n_head))` and `scaled_queries` multiplies the queries by it — after
RoPE in `attend`, as llama.cpp's `ggml_scale(Qcur, f_attention_scale)`; before it in
`forward_attn_batched`, where RoPE may run per row (a rotation is linear). Multiplying the
queries rather than threading a scale through every kernel (standard, grouped, both flash
arms, the decode kernels) keeps one site: one elementwise launch per layer, against ~27 a
decode layer already makes (tinyllama's count, § "Projections that share an activation must
share the work"), on that shape only.

**Measured** with `examples/make_tiny_gemma_gguf.py` (46 layers, n_embd/n_head 16, head 32):

| file | `SWARMLLM_QUERY_SCALE=0` (old) | scale read |
|---|---|---|
| `QK_STD=0.25` | worst 0.727, top-1 19/40 | **0.99987**, 39/40 |
| default (0.5, decisive) | 0.657, 15/40 | 0.966, 34/40 |

The decisive file's residual is not logic: llama.cpp with `type_k = type_v = f32` moves it to
0.992 (gotcha #728's test — flip the reference's suspect setting and watch the gap), and the
softer file agrees to 0.99987. Rounding grows through 46 layers of sharp attention. Tests:
`only_gemma2_27b_scales_its_queries_by_the_model_width` (the 9B and 2B read none),
`the_query_scale_reaches_the_batched_path_too` (fails by 0.067 without the batched scale).

## Gemma 3: each layer turns at its own rate (2026-10-16, FUTURE_WORK #115 (1))

**What llama.cpp does** (`src/models/gemma3.cpp` and the shared hparams in `llama-model.cpp`
at master; llama-cpp-python 0.3.16's vendored loader sets the same values):

- a window only when the file names one above 0, on layer `il` when `il % 6 < 5`
  (`load_swa_pattern(ml, 6)`, `set_swa_pattern`);
- **a rotation per layer**, `llama_model::get_rope_freq_base/scale(cparams, il)` by
  `is_swa(il)`: a windowed layer turns at `rope_freq_base_train_swa` (the hparams default
  10,000; a file's `rope.freq_base_swa` wins) with `rope_freq_scale_train_swa` = 1, a global
  layer at the file's base with `1 / rope.scaling.factor`;
- linear scaling for EVERY family: `rope.scaling.type` defaults to `linear`, the factor is
  `rope.scaling.factor` or the older `rope.scale_linear`, and `freq_scale` multiplies the angle
  (`rope_yarn` with `ext_factor` 0: `theta = freq_scale * theta_extrap`);
- `1/sqrt(n_embd / n_head)` queries on the 27B only (62 layers, #279's rule), Q and K
  RMS-normalised per head before RoPE, norm weights already carrying Gemma's +1 (the
  converter's `norm_shift`), embeddings times `sqrt(n_embd)`, GELU feed-forward, post-norms,
  no attention soft-cap.

**The real headers** (gotcha #728's lesson, read before writing a line): gemma-3-1b-it is 26
layers, n_embd 1152, 4 heads / 1 KV head, key length 256, window 512, base 1,000,000, NO
scaling keys, context 32,768; gemma-3-4b-it is 34 layers, 2560, 8 / 4, 256, window 1,024,
base 1,000,000, `rope.scaling.type = linear`, `rope.scaling.factor = 8`, context 131,072.
Neither names `rope.freq_base_swa`. `<end_of_turn>` (106) ends a turn by name
(`END_OF_GENERATION_TOKENS`); the file's EOS is `<eos>` (1).

**What we do.** `split::rope::LayerRope::read` is the one reading of how each layer turns;
`LayerPlacement::rope_for` is the one place a layer's tables come from (the loops shadow
`cos`/`sin` from it), so no per-architecture loop can hand a windowed layer the global ones.
The loader builds a second table only where the windowed rotation differs — Gemma 2's copies
its global one, so it builds none. Which layers are windowed comes from
`SlidingWindow::from_file`, never the A/B-switched `read`, so `SWARMLLM_SLIDING_WINDOW=0`
changes the window alone. Before this, no family's `rope.scaling.factor` was read at all; the
survey of every reference GGUF here found none that carries one except Gemma 3 4B. A scale of 1
is plain RoPE bit for bit (`a_scale_multiplies_every_position`), so every model that ran
before runs the same tables. YaRN is read and NOT applied (#280): the factor without its ramp
would be wrong at every position.

**Measured** against llama.cpp 0.3.16 on the processor, our logits for the last 8 prompt
positions + 4 decode steps (`logits_reference_probe`, 512-token chunks), same binary per row:

| model, tokens | arm | worst cos | median | top-1 |
|---|---|---|---|---|
| 1B Q4_K_M, 300 | all read | **0.99977** | 0.99982 | 11/12 |
| 1B Q4_K_M, 300 | `SWARMLLM_LAYER_ROPE=0` | 0.97239 | 0.97833 | 4/12 |
| 1B Q4_K_M, 1,500 | all read | **0.99945** | 0.99985 | 12/12 |
| 1B Q4_K_M, 1,500 | `SWARMLLM_SLIDING_WINDOW=0` | 0.84652 | 0.94538 | 2/12 |
| 4B Q4_K_M, 300 | all read | **0.99872** | 0.99941 | 11/12 |
| 4B Q4_K_M, 300 | `SWARMLLM_ROPE_SCALING=0` | 0.99198 | 0.99733 | 11/12 |
| 4B Q4_K_M, 300 | `SWARMLLM_LAYER_ROPE=0` | 0.76893 | 0.88542 | 1/12 |
| 4B Q4_K_M, 2,000 | all read | **0.99862** | 0.99969 | 12/12 |
| 4B Q4_K_M, 2,000 | `SWARMLLM_SLIDING_WINDOW=0` | 0.74180 | 0.87378 | 2/12 |

The 4B's last 0.1% is not logic. With llama.cpp's KV cache in f32 (`REF_KV_F32=1`, the
control gotcha #864 names) it reads 0.99888 / 0.99844 — the f16 cache is not it. And a tiny
f32 file with every 4B mechanism (`make_tiny_gemma_gguf.py` `ARCH=gemma3`: 12 layers, window
8, factor 8, Q/K norms, n_embd/n_head ≠ head) agrees at **1.000000** on all 40 positions —
whole, in 5-token pieces and split at layer 7 — and at 0.98698 without scaling, 0.89172 with
the windowed layers on the global rotation, 0.60138 with no window. A logic error survives an
f32 file; the 4B's gap is its Q4_K/Q6_K weights (Gemma 2 2B Q4_K_M sits at 0.99939, the
qwen3moe file at worst 0.99818).

**On the card** (CUDA build, `~/swarmllm-g3/card.sh` under the safety kit, 0 driver events),
against the same processor reference: 1B at 1,500 worst 0.99916 (median 0.99978, 12/12); 4B at
2,000 worst 0.99401 (median 0.99962, 12/12) whole, and 0.99439 split card/processor at layer
17 — the windowed and global tables on both devices. The 4B's two low positions (1993, 1998)
are the ones that were worst on the processor too, and the most peaked (|ref|max 23 and 27
against 16-19). The control: llama.cpp master's OWN card run against its own processor run, on
those ids, reads **0.99690** at 1998 (top-1 11/12); ours on the card against llama.cpp on the
card reads 0.99856 there (median 0.99984, 12/12). That position is sensitive to precision in
llama.cpp itself; ours sits closer to llama.cpp's card than llama.cpp's card sits to its
processor. The tokenizer matched llama.cpp on 8/8 cases
(`tokenizer_reference.py` over the 1B's header) and the GGUF's chat template renders as
jinja2 does (`gemma3_actual_template_matches_jinja2`).

Tests: `gemma3_global_layers_scale_and_windowed_layers_turn_at_their_own_base`,
`gemma2_windowed_layers_rotate_as_the_rest_unless_the_file_names_a_base`,
`only_linear_scaling_is_applied`, `a_scale_multiplies_every_position`,
`windowed_layers_take_their_own_rope_tables` (red with `rope_for` ignoring the windowed
tables), `gemma3_windows_five_layers_in_six_only_when_the_file_names_a_window`,
`only_gemma3_27b_scales_its_queries_by_the_model_width`.

## A file's RoPE scaling is read for every family (2026-10-16, FUTURE_WORK #280)

llama.cpp reads `rope.scaling.type` (default `linear`), `rope.scaling.factor` and
`rope.scaling.original_context_length` in the hparams EVERY family shares (gotcha #865), so a
file's scaling applies whatever its family. `split::rope::RopeScaling` is the one reading:
`Linear(1 / factor)`, or `Yarn` — per `ggml_rope_yarn_corr_dims` and `rope_yarn`, the pairs
below floor(corr(32)) keep their angle, those from ceil(corr(1)) take `1 / factor`, a linear ramp
between, and cos/sin are lifted by `1 + 0.1 ln(factor)` (times `rope.scaling.attn_factor`). Master
gets the lift as `get_mscale(factor) / (1 + 0.1 ln factor)` in `llama_context` times
`rope_yarn`'s own `1 + 0.1 ln(1 / freq_scale)`; 0.3.16 as 1 times the same — one product. The
beta / `ext_factor` / `yarn_attn_factor` keys are read only by families this build does not
run (grok, mistral3), and DeepSeek-2's `yarn_log_multiplier`, where master and 0.3.16 differ,
stays with #116. `RopeScaling::per_pair` turns either into per-pair multipliers and a lift that
the existing factor-table builder takes; with no scaling it is the old builder, bit for bit.
Any other type is served unscaled with a warning at load.

**Measured** against llama.cpp 0.3.16, same binary, `SWARMLLM_ROPE_SCALING=0` = the old path:

| file | tokens | scaling read | `=0` |
|---|---|---|---|
| Qwen3-1.7B Q8_0, unchanged (no scaling — the baseline) | 300 / 1,500 | 0.99971 / 0.99946 | — |
| the same with Qwen's YaRN keys (factor 4, original 32,768) | 300 | **0.99943** (median 0.99985) | 0.97144 |
| the same | 1,500 | **0.99666** (median 0.99988) | 0.96603 |
| tiny f32 Gemma 3, YaRN factor 8, original 16 | 40 | **1.000000**, whole and split at 7 | 0.98727 |
| the same, original 4,096 (ramp across pairs 3-8 of 16) | 40 | **1.000000** | 0.99711 |

The 1,500 run's one low position (1488, the plain file's worst position too) is the
reference's rounding: with llama.cpp's KV cache in f32 (`REF_KV_F32=1`) it reads 0.99888, and
reading our prompt in one pass instead of 512-token chunks changes nothing. Tests:
`linear_and_yarn_scaling_are_read_and_nothing_else`, `yarn_ramps_the_slow_pairs_to_the_factor`
(Qwen3's 128K setting worked by hand: ramp from pair 23 to 40, lift 1 + 0.1 ln 4).

## Gemma 4: each layer's attention has a shape of its own (2026-10-16, FUTURE_WORK #115 (3))

**What llama.cpp master does** (`src/models/gemma4.cpp`, `llama_model_gemma4::load_arch_hparams`,
read at master 2026-10-16; no Python binding here runs it, so the reference is
`~/llama.cpp-ref/dump_logits` built against master):

- **a shape per layer**: `attention.head_count_kv` is a per-layer list and
  `attention.sliding_window_pattern` a per-layer bool list; a windowed layer reads
  `attention.key_length_swa` and `rope.dimension_count_swa` (`llama_hparams::n_head_kv(il)`,
  `n_embd_head_k(il)`, `n_rot(il)`). On the 12B (48 layers, n_embd 3,840, 16 heads) the
  windowed layers are 8 K/V heads × 256 over the last 1,024 positions, every sixth layer is
  global: 1 K/V head × 512;
- **a global layer carries no `attn_v`**: V is K's projection output
  (`use_alternative_attention`), taken before K's norm and RoPE;
- **Q and K are RMS-normed per head WITH weights, V per head with none**
  (`ggml_rms_norm(Vcur)`), and the attention scale is **1.0** (`f_attention_scale`, "no
  pre-attn scaling");
- **proportional RoPE**: windowed layers turn their 256 at base 10,000 and never read
  `rope_freqs`; global layers turn at base 1,000,000 with the file's `rope_freqs` [256] —
  the first quarter of each pair set turns and the rest is held by a factor of 1e30;
- `layer_output_scale` multiplies each layer's whole output, residual included (`out_scale`);
  GELU feed-forward, Gemma's embedding scale, a final soft-cap of 30;
- the tokenizer is its own pre-type (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`): spaces become `▁`
  with none in front, the text is cut into newline runs and the rest (`[^\n]+|[\n]+`), a
  newline run that is an entry is taken whole (PR #21343), BPE over raw UTF-8 with `<0xNN>`
  fallback, and BOS is forced whatever the file says (PR #21500).

**What we do.** `split::attention_shapes` is the one reading of every layer's shape, as
llama.cpp reads it for every family: a file naming one count and no `_swa` keys gets one
shape everywhere, which is what every release before read. Model-wide estimates (the loader's
KV budget, the planner, admission) charge the widest layer's shape, a safe upper bound: a
global layer's real cache (1 × 512, f32 mirrored) is at most the 8 × 256 f16 estimate per
position. `split::query_scale` multiplies Gemma 4's queries by `sqrt(head_dim)` of their own
layer, which the attention's `1/sqrt(head_dim)` cancels — one elementwise op, the site #279
built. `LayerWeights::v_rms_norm` is applied in ONE place (`normed_values`) for both
attention entry points; `LayerWeights::output_scale` where the layer's output is assembled
(`executor::scaled_output`); `split::rope::LayerRope` gives windowed layers their own width
and no file factors (`WindowedRope`). A Gemma 4 size whose layers are not built here is
refused by name, off its tensor table (`unbuilt_gemma4_part`): the 26B-A4B's experts and the
E2B/E4B's per-layer embeddings would otherwise load as the dense layout and write nonsense.
LoRA refuses the family (`lora::check_fits` judges one width per projection, and Gemma 4's
layers differ).

**The tokenizer.** Gemma 4 declares every marker it has (16 CONTROL, 7 USER_DEFINED such as
`<|tool_call>`, `<|"|>`), and 75 of its NORMAL entries are `<…>`-shaped (`<table>`,
`<unused5>`, even `<eos>`). llama.cpp matches only the declared ones whole in a prompt — plus
every name on its end-of-generation list, which it overrides to CONTROL — so for this
vocabulary the declared types are the whole rule (`BpeTokenizer::gemma4`). On 18 cases
from llama.cpp master's own tokenizer (`tok_cases` over a vocabulary-only header: the
template rendered with and without thinking, code, emoji, CJK, 40-newline runs, `\r\n`,
markup, tool calls), 18/18 agree; `<eos>` as plain text was the one miss before the
end-of-generation names were added. The GGUF's 18,922-character chat template renders as
jinja2 does (`the_gemma4_template_renders_exactly_as_jinja2_does`).

**Measured.** A tiny file with every 12B mechanism (`make_tiny_gemma_gguf.py ARCH=gemma4`: 12
layers, window 8, per-layer K/V heads 2 / 1, widths 32 / 64, proportional factors, V-is-K,
output scales) against llama.cpp master, 40 positions:

| arm | worst cos | median | top-1 |
|---|---|---|---|
| all read, whole / split at 5 and 8 / 5-token pieces | **0.999964** | 0.999990 | 40/40 |
| `SWARMLLM_QUERY_SCALE=0` | 0.229 | 0.529 | 2/40 |
| `SWARMLLM_LAYER_ROPE=0` | 0.287 | 0.668 | 9/40 |
| `SWARMLLM_SLIDING_WINDOW=0` | −0.045 | 0.303 | 8/40 |

The file's first cut, its Q/K norm weights at 1, read 0.944 worst: for a family that norms Q
and K and then scales by nothing, the WEIGHT spread (`QK_STD`) does not set how sharp the
attention is — the norm gain does, and 12 layers of near-one-hot attention amplify rounding
(gotcha #867). With the gain at `2 × QK_STD` it reads 0.99996.

**The real 12B** (unsloth's Q4_K_M, SHA-256 matching HuggingFace) on the processor, our logits
(`logits_reference_probe`, 512-token chunks) against llama.cpp master's (`dump_logits`). Its
gap sits below Gemma 3 4B's, so the known answer was measured first: **llama.cpp against
itself**, its repacked Q4_K kernels (its default on this processor) against its plain ones
(`DUMP_NO_REPACK=1`), on the same ids:

| prompt (positions compared) | pair | worst cos | median | top-1 |
|---|---|---|---|---|
| 28 tokens (all 28) | ours vs llama.cpp | 0.99402 | 0.99937 | 25/28 |
| | ours vs llama.cpp, plain kernels | 0.99542 | 0.99922 | 25/28 |
| | **llama.cpp vs itself** | **0.99380** | **0.99939** | **24/28** |
| 1,536 tokens (last 48 + 4 decode) | ours vs llama.cpp | 0.98950 | 0.99815 | 41/52 |
| | ours vs llama.cpp, plain kernels | 0.98463 | 0.99830 | 43/52 |
| | **llama.cpp vs itself** | **0.97629** | **0.99833** | **45/52** |

Each mechanism off, in the same binary, on the real file:

| arm | prompt | worst cos | median | top-1 |
|---|---|---|---|---|
| `SWARMLLM_SLIDING_WINDOW=0` | 1,536 | 0.78406 | 0.98146 | 28/52 |
| `SWARMLLM_LAYER_ROPE=0` (windowed layers on the global rotation) | 28 | 0.77089 | 0.99816 | 23/28 |
| `SWARMLLM_QUERY_SCALE=0` (queries scaled by 1/sqrt(head_dim)) | 28 | −0.72874 | 0.70521 | 8/28 |

We sit as close to either of llama.cpp's paths as they sit to each other, and the low
positions are the same ones (the 28-token run's positions 5, 8, 16 and 21; the long run's 1507). llama.cpp's
f32 KV cache (`DUMP_KV_F32=1`) changes nothing. This model is precision-sensitive at the level
of a kernel choice; the comparison cannot resolve finer than that.

**On a card** (the CUDA build, split 24/48 — the 7.1 GB file does not fit the 8 GB card whole —
against llama.cpp master's CUDA build at `n_gpu_layers = 24`, same commit as its processor build):

| prompt | pair | worst cos | median | top-1 |
|---|---|---|---|---|
| 28 tokens | ours on the card vs llama.cpp's processor | 0.98778 | 0.99925 | 26/28 |
| | ours on the card vs llama.cpp's card | 0.98814 | 0.99906 | 26/28 |
| | **llama.cpp's card vs its processor** | **0.99239** | **0.99893** | **25/28** |
| 1,536 tokens (last 48 + 4 decode) | ours on the card vs llama.cpp's processor | 0.98534 | 0.99809 | 40/52 |
| | **llama.cpp's card vs its processor** (last 48) | **0.97928** | **0.99787** | **41/48** |

Every pair's worst short-prompt position is 5; ours sits ~0.004 further off there than
llama.cpp's own card does, with the median and top-1 at or above it, and `SWARMLLM_KV_F16=0`
reads the identical 0.98778 (one pass: the cache's precision never enters). The first card run
gave all-NaN logits past ~1.5K tokens: the vendored flash kernel's 512-wide block was wrong at
every shape, see § "The flash kernel is checked at every head width it is handed".

Tests: `gemma4_reads_each_layers_attention_shape`, `gemma4_attention_is_not_scaled_on_any_layer`
(red with `SWARMLLM_QUERY_SCALE=0`),
`gemma4_windowed_layers_turn_their_own_width_without_the_files_factors`,
`gemma4_sizes_this_build_cannot_run_are_refused`, the three in `tokenizer::gemma4_tests`
(`declared_markers_are_matched_whole_and_ordinary_markup_is_merged` is red with the shape rules
left on).

## A model that cannot write replies is refused off its header (2026-10-15)

A tester's fresh node offered `Qwen/Qwen3-Embedding-0.6B-GGUF` in the HuggingFace
browser as "Chat", let it be downloaded, and put it in the chat picker. Asked a
question, it streamed "IP address" for 30 s, and the log recorded `outcome=error
tokens=0` (field report 2026-10-15). Its header is an ordinary `qwen3` — a family
that loads — with `qwen3.pooling_type = 3` (LAST, GGUF type UINT32) and no
`output.weight`: the tied embedding table stands in for a head that was never
trained to predict a next token.

**The rule is Ollama's** (`server/images.go`, `ggufCapabilities`): a GGUF that
names a pooling type is an embedding model, not a completion one. llama.cpp's
converter writes the key only for embedding models. Here NONE (0) reads as "not
said", so a generative file that spells it out still loads.

`GgufTensorMeta::refusal` is the one answer to "will this build run the model
to write replies" — a family with no loader (`ModelArch::is_supported`) or an
embedding model (`pooling_type != 0`) — and `ModelRefusal::into_error` the one
wording (400 `Validation`: no node and no retry can help, and
`every_holder_would_refuse` stops a coordinator failing over on a peer's
refusal). It is asked:

- by the **loader**, after `from_content` — every load of a model's layers, here
  or for a peer, so a node with no header to ask is still covered;
- by the **router** (`handle_submit`, through `SharedState::refused_model`)
  before a request is queued, when this node holds the header;
- by **auto-manage**: `gather_candidates`, and `trigger_download` on both
  branches (the HuggingFace probe's header, the header fetched before a peer
  transfer);
- by the user's **download** (`/api/admin/hf/download-shards`) before a byte moves;
- by `/v1/models` and the dashboard list (`cannot_chat`), which leave such a
  model out of what a client or the chat picker offers.

Before a header exists, HuggingFace's own metadata decides
(`model::huggingface::is_embedding_repo`): the pipeline tag when there is one —
`Qwen/Qwen3-Embedding-0.6B` is `feature-extraction` yet tagged `text-generation`,
which is how it passed the trending list's text filter, while a speculative
drafter is `text-generation` yet tagged `feature-extraction` — otherwise an
embedding or ranking task among the tags with no generation task beside it (the
GGUF re-upload has no pipeline tag). The trending list drops such a repo; a
search shows it tagged "Embedding", marked "Not a chat model", with no action.

`/v1/embeddings` still answers 501 (`FUTURE_WORK.md` #200's leftover): the
refusal is about chat, not a promise that embeddings are served. Tests:
`a_header_that_names_a_pooling_type_is_an_embedding_model`,
`an_embedding_model_is_refused_before_it_is_queued`,
`a_family_this_build_refuses_is_never_a_candidate`,
`an_embedding_repo_is_told_apart_from_a_chat_model`.

## The flash kernel is checked at every head width it is handed (2026-10-16, gotcha #868)

**What broke.** The vendored kernel's `run_mha_fwd_hdim512` chose, on a card with under 128 KB
of shared memory a block (every RTX 30/40-series, sm86/sm89), `Flash_fwd_kernel_traits<512, 32,
32, 4>`: a 32-query-row block on FOUR warps. `TiledMma` gives each warp 16 rows, so four warps
compute 64 — two of them rows past the block, read out of shared memory beyond sQ. Every other
configuration in the file is a whole number of warps tall. No family supported before had 512-wide
heads (MLA runs standard attention); Gemma 4's global layers do, and its 12B split 24/48 gave
all-NaN logits once a 48-row chunk landed on 1,484 cached positions. A 28-token prompt read
identically before and after the fix (worst 0.98778 both times) — its gap is not this.

**How it was found.** A bisect in one CUDA binary at the failing settings (a "clean" arm at KEEP_LAST
8 had been compared against a failure at 48 — no baseline, and it sent the first hypothesis to the
f32 marks): the processor layers' card paths off — still NaN; the window off — still NaN;
`SWARMLLM_FLASH_OFFSET_CAUSAL=0` (standard attention for a chunk on a warm cache) — clean. Then the
kernel alone, `examples/flash_reference_check.rs`: random f16 Q/K/V, flash on the card against f32
softmax attention on the processor over the same rounded values, causal bottom-right, windowed where
asked:

| head width | geometry | shapes | max \|diff\| before | after |
|---|---|---|---|---|
| 128 | 24/8 | 15 | 7e-5 - 1.0e-3 | same |
| 256 | 16/8 | 15 | 7e-5 - 1.1e-3 | same |
| 512 | 16/1 | 15 | **0.23 - 3.6 (all 15 wrong)** | 7e-5 - 1.2e-3 |

**The fix.** `<512, 32, 32, 2>`: the same 96 KB block on the two warps it needs — the useful work
the old block did. `64 x 16` on 4 warps fits the same shared memory but fails `dropout.h`'s
`static_assert` (kBlockN ≥ 32). The 128 KB arm (A100-class) was already `<512, 64, 32, 4>`. Cost,
min of 10 (16/1 heads, 512 queries on 4,096 keys): 10.9 ms against 2.0 for 256-wide 16/8 — ~2.7x
less work a millisecond, on one Gemma 4 layer in six. `kernel_traits.h` now has
`static_assert(kBlockM % (16 * kNWarps) == 0)`; `every_vendored_flash_block_is_a_whole_number_of_warps_tall`
reads the launch file on every push (CI compiles the kernels only in cache-warm and release).

**The fix did not compile, at first.** cudaforge rebuilds a kernel object when its `.cu` file's hash,
the nvcc arguments or a WATCHED path changes, and neither vendored build script watched a header:
after the header edit the build printed `All library kernels up-to-date, skipping compilation` and
linked the broken objects again. `vendor/candle-flash-attn/build.rs` and
`vendor/candle/candle-kernels/build.rs` now `.watch()` every header in their kernel directories
(read from the directory: the hand list had already missed `mask.h`, and candle-kernels' named three
of five). The first build after this recompiles every kernel once; on this laptop two kernel build
scripts at 4 nvcc threads each OOM'd the 10 GB build slice — `-j 1` with `CUDAFORGE_THREADS=3`
took 36 min.

## Single-source-of-truth helpers — Inference kernels, caches and the tokenizer

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/inference.md`

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`.
- **`inference::decode_attn::gqa_decode_attention_cpu`** — single-position attention straight over the KV cache in its stored `[b, kvh, S, d]` layout. **Each K/V row is read ONCE per group, never once per query head** (the per-head version was #119's long-context slowdown); tasks are (batch, kv head, fixed 256-position chunk) merged by flash-decoding's reduction — the chunk is a constant so the result never depends on the thread count. → `docs/invariants/inference.md` § "It read the cache once per QUERY head".
- **`inference::prefill_attn::gqa_prefill_attention_cpu`** — the same idea for `q_len >= 2`: key-tiled, the score matrix never written out; the matmul paths it replaces on the CPU live on as `layers::matmul_attention`.
- **`inference::fast_math`** — eight-lane AVX2 `expf` (`exp_inplace`, Cephes polynomial, ~2 ulp vs libm, pinned by `vectorised_exp_tracks_libm` over [-80, 80]) and the fused `silu_mul` CustomOp2. A new elementwise pass that calls `f32::exp` in a loop routes through here instead.
- **`inference::cpu_pools::in_phase_pool`** — binds a forward pass to the CPU thread pool that suits its phase, at ONE choke point: `SplitModel::forward_inner_impl` and `forward_batch`.
- **`inference::layers::new_kv_cache`** — the only way to construct a KV cache.
- **`inference::split::kv_cache::LayerKv`** — one layer's KV cache: the f32 BHSD cache every path reads, plus an optional f16 BSHD mirror for the CUDA flash kernel.
- **`inference::split::kv_cache::SeqCache` / `KvPair` + `LayerKv::truncate`** — the KV cache buffer is this project's own, not candle's, for ONE reason: candle's `Cache` keeps its length private, so the only way to keep the first `n` positions was snapshot + `reset()` + `append()` — two full copies per layer on every rejected speculative draft. **Never re-introduce a copy on the rollback path.**
- **`inference::attn_softmax::scaled_masked_softmax`** — the single expression of attention's tail: scale, optional Gemma-2 logit soft-cap, additive mask, softmax.
- **`inference::layers::standard_attention` grouped GQA decode** — (c4cc3b16, 2026-08-16) for `q_len == 1` with `n_kv_head < n_head`, standard attention no longer expands the KV cache with `repeat_kv`; it reshapes the query heads into matmul rows against the UNEXPANDED cache.
- **`inference::layers::cuda_decode_prefers_standard`** — on CUDA, `q_len == 1` takes standard for EVERY head geometry, prefill always flash. The GQA exclusion was retired on 2026-08-23 once `grouped_gqa_decode_attention` deleted the `repeat_kv` cost it existed to route around; `SWARMLLM_GQA_DECODE_FLASH=1` restores the old rule for an A/B inside one binary.
- **`inference::sampling::sample_among_top_k` — top-k shrinks the candidate set before temperature, top-p or the draw run**, and picks the SAME token as the full-vocabulary path for the same draw (candidates kept in INDEX order, so every sum is bit-identical). ~15x at a 152k vocabulary. Never re-add a vocabulary-wide pass after top-k. → `docs/invariants/inference.md` § "Top-k shrinks the candidate set"
- **`inference::mem_bandwidth::measured_gbps`** — what this machine's memory actually delivers, measured once and cached.
- **`inference::cancel::unless_cancelled` — every wait that can run for minutes watches the request's cancel flag** — `InferenceRequest::cancel` is the ONE cancellation signal — set by `api::submit_to_router`'s drop guard (every non-streaming caller), by both SSE surfaces on `sse_tx.closed()`, and by `/cancel`; read around every WAIT, never around a send.
- **A prompt pass asks between layers whether its request was cancelled** — `KvCacheStore::set_cancel_oracle` is probed once per layer by `forward_inner_impl`, which returns `CANCELLED_MID_FORWARD`; `forward_was_cancelled` is the one reader of that message.
- **`inference::split::token_embedding::rows_on_demand_eligible`** — the single answer to "is this model's `token_embd.weight` held quantized with its rows dequantized on lookup, or dequantized whole at load?".
- **`inference::split::read_gguf_header`** — the single way to parse a GGUF header off a PATH, and the buffering is the entire reason it exists.
- **`inference::split::GgufTensorMeta::tied_output_location`** — the single definition of "is this model weight-tied", i.e. does it reuse `token_embd.weight` as the LM head instead of shipping an `output.weight`. Both sidecar writers and the reader go through it.
