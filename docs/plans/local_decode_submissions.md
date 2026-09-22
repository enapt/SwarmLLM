# Making local decode fast: spend fewer submissions per token

**Written 2026-09-22 against v0.3.197-alpha, from measurements on the live
release node.** Evidence and per-model numbers:
`docs/invariants/inference.md` § "A decode token is bound by GPU submission
COUNT, not bandwidth". Method: `docs/DIAGNOSTICS.md` § "Where a decode token
actually goes". Diagnosis lesson: gotcha #676.

## The goal, stated as a number

A 3B Q4_K_M moves ~2.0 GB of weights per decoded token. This card delivers
~384 GB/s, so the arithmetic floor is **~5 ms/token, i.e. ~190 tok/s**. A real
implementation lands at 30-50% of roofline, so **~60-100 tok/s is the target**.

Measured today after the first fix: **57 tok/s** (17.5 ms/token). Before it,
49.6. So the 3B is roughly at the bottom of the plausible band and the 1.1B is
nowhere near it — it reads 72.5 tok/s where its own roofline is ~570.

**The gap is not arithmetic.** It is the CPU thread that submits the work.

## Why it is submissions and not bandwidth

Three readings, each independently sufficient:

1. **`ms/layer` is flat at 0.48-0.58 across a 3.3x span of bytes per token**,
   over five models and three quantizations. A 0.5B at **F16** moves ~1.7x a
   1.1B Q4_K_M's bytes and is *faster*. Layer count predicts the cost; size
   does not.
2. **The worker burns 22.7-26.0 ms of CPU per token against 21.0-24.1 ms of
   wall** — 1.08 cores, three reps. CPU time ≈ wall time means the thread never
   waits on the device. GPU utilization: **median 52%**.
3. **1,085 GPU submissions per token**, with **17.6 of 23.0 ms inside the CUDA
   driver API**.

A launch costs ~10-12 us on this box because WSL2 marshals every submission to
the Windows host driver; native Linux is ~3-5 us. ⚠ **The constant is
box-specific, the structure is not** — at 5 us the same token still spends
~5 ms submitting. **Not A/B'd against a native-Linux GPU box, because the fleet
has exactly one GPU node.** That is the single biggest unknown in this document.

## The budget, itemised

Per decoded token, tinyllama (22 layers), AFTER stage 1, from
`examples/decode_submissions.sh`. ⚠ Times are nsys-inflated — the counts are
exact, and it is the counts that stages below move:

| API | calls/token | per layer | ms/token | what it is |
|---|---|---|---|---|
| `cuLaunchKernel` | 625 | 28.4 | **15.08** | every kernel: 2 per quantized matmul, plus norms, rope, attention, residual, copies |
| `cuMemAllocAsync` | 657 | 29.9 | 1.81 | one per op output — candle allocates fresh every time |
| `cuMemFreeAsync` | 656 | 29.8 | 1.64 | the matching frees |
| `cuEventCreate` | 1,314 | 59.7 | 0.85 | **two events per allocation**, from cudarc's cross-stream tracking |
| `cuEventDestroy_v2` | 1,311 | 59.6 | 0.64 | |
| `cudaLaunchKernel_v7000` | 45 | 2.1 | 1.47 | runtime-API launches (cuBLAS internals) |
| `cuMemcpyHtoDAsync_v2` | **30** | 1.4 | 1.30 | **~one host→device copy per layer — unexplained** |
| `cuMemcpyDtoHAsync_v2` | 1 | — | 0.32 | logits to host for sampling; a blocking round trip |
| `cuMemsetD8Async` | 1.4 | — | 0.05 | was 321 before stage 1 |

## The plan

Ordered so each stage is independently shippable and measurable, and so no
stage depends on a later one. **Stage 3 is a prerequisite for stage 4** and that
ordering is the main non-obvious thing here.

### Stage 1 — Stop zero-filling buffers the next kernel overwrites ✅ SHIPPED

`CudaDevice::alloc_fully_overwritten`, commit on 2026-09-22. Removed 321 of 992
submissions per token (**-32%**) for **+30% tok/s on a 1.1B and +15% on a 3B**,
A/B/A/B in one binary, byte-identical replies in all four arms.
`SWARMLLM_ZERO_QMATMUL_BUFFERS=1` restores the old behaviour.

### Stage 2 — Turn off cudarc's per-allocation event tracking ✅ SHIPPED (count-verified)

**2,625 event ops per token → 0**, 2026-09-22. cudarc created a read event and a
write event for every `CudaSlice`, waited on both in `Drop` and destroyed both —
four event API calls per allocation, ~700 allocations a token.

The precondition turned out the opposite way to the first reading of it. Three
production sites construct a `CudaDevice`, which looks like several streams —
but `BackendDevice::new` takes `context.default_stream()`, and cudarc's
`default_stream()` is `cu_stream: null_mut()`, the legacy default stream. **All
of them share ONE stream**, `is_in_multi_stream_mode()` is false, and cudarc's
own `is_managing_stream_synchronization()` was therefore already false — it was
not consuming the events it was creating. Full argument in the invariants file.

**Measured on an idle box**, 500-token generations, 8 reps, A/B/A/B, best-of-N:
**tinyllama +15%** (107.7/109.1 against 93.6/93.6, no overlap) and
**llama-3.2-3b no measurable effect** (71.1/78.8 against 72.9/74.7, overlapping).
Same shape as stage 1 — a fixed CPU saving is a smaller share of a bigger
model's token. ⚠ Read best-of-N, not the median: the bench node is on the live
swarm and its daemon intermittently steals the decode thread's core.
`SWARMLLM_CUDA_EVENT_TRACKING=1` restores the old behaviour.

### Stage 2b — Projections that share an activation share the work ✅ SHIPPED

Counting launches by kernel name (`SWARMLLM_COUNT_KERNELS=1`, added for this)
showed `quantize_q8_1` running **7.05 times per layer for 7.04 matmuls** — once
per matmul, for only 4 distinct activations. `QMatMul::forward_shared` fixes
that (7.05 → 4.05 per layer, 601 → 535 launches) and, for the Phi family's
fused QKV tensor, stops each of Q/K/V recomputing the **whole** fused matmul.

| model | delta | |
|---|---|---|
| **phi-3.5-mini** | **+55%** | real matmul work removed; A/B/A/B, huge margin |
| **phi-4-mini** | **+22%** | one pair |
| llama-3.2-3b | +4% | at the edge of resolvable |
| tinyllama | +7% | medians separate, bests overlap |

⚠ **Two different effects — do not quote them together.** The Phi win is
arithmetic; the rest is ~11% of launches ≈ ~5% of a token, and not established.
`SWARMLLM_SHARE_PROJECTIONS=0` is the off arm.

⚠ **It took two goes: there are TWO quantized matmul paths** (vec for decode,
MMQ for prefill/batch) and the first fix did one. Prefill sat at 7.05
quantizations per layer while decode read 4.05, and only reading the count on
the path I had NOT changed found it (gotcha #679). Both share now — prefill
736 → 670 launches — but with **no measurable prefill speed-up** (min-of-N
13 ms vs 14 over 56 chunk forwards), because prefill does real arithmetic over
128 rows and submissions are a small share of its time.

⚠ **Prefill cannot be measured through TTFT**: a repeated prompt is served from
the prefix cache and reads ~0.02 s however slow prefill is. Use `PROF seq_len=N`
with unique prompts, filtered to one chunk size.

**What the kernel table says to do next**, now that it exists (per layer):
`rmsnorm_f32` 2.05 + `badd_f32` 2.00 are four launches for norms and residuals
that fusion could make two; `affine_f32` + `bmul_f32` + `softmax_f32` are three
launches for the attention tail that `scaled_masked_softmax` already describes
as one operation. Both need a new CUDA kernel, and `candle-kernels` is a
registry crate rather than a vendored one — so that means either vendoring it or
using candle's unused `get_or_load_custom_func` path with our own module.

### ▶ Ordering REVISED 2026-09-22 after reading how llama.cpp did this

The stages below were ordered by size of the line in the budget. Reading
llama.cpp's own decode work reorders them, and sizes two of them from someone
else's measurements instead of our guesses.

**Sources**: [NVIDIA on CUDA graphs in llama.cpp](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)
· [am17an, token-generation optimizations](https://am17an.bearblog.dev/new-post/)
(llama.cpp discussion #17621) · [issue #12152](https://github.com/ggml-org/llama.cpp/issues/12152)

**1. Fusion now comes FIRST, with reference numbers.** llama.cpp measured
**329 → 419 tok/s (~27%)** on an RTX 5090 / gpt-oss-20b from a set of decode
fusions, and each of the two that map onto our kernel table was worth ~10% on
its own:

| their fusion | ~gain | our kernels, per layer |
|---|---|---|
| RMS-norm fused with the preceding multiply/add | ~10% | `rmsnorm_f32` 2.05 + `badd_f32` 2.00 |
| GEMV fused with the gated activation | ~10% | `usilu_f32` 1.00 + `bmul_f32` 1.00 |
| TopK-MoE (softmax + expert select) | ~10% | MoE only — would also remove `topk_cpu`'s host round trip |

Their reasoning is ours: *"fusing kernels reduces memory traffic and kernel
launch time… token generation is memory-bound rather than compute-bound"*.

**2. Where our fused kernels go, without a fifth vendored crate.**
`candle-kernels` is a registry crate, but `CudaDevice::get_or_load_custom_func`
takes **PTX as a string** and has no callers — so a small `.cu` compiled to PTX
by our own `build.rs` under `candle-cuda` loads through it. That is the cheap
route in, and it makes each fusion independently shippable and A/B-able.

**3. Stage 3 before stage 4 is CONFIRMED, and was a guess before.** llama.cpp
patches only the KV-cache pointers in an already-instantiated graph each token
(`cudaGraphExecUpdate` for the rarer structural change) — **which works because
its ACTIVATION addresses are already stable, in a fixed compute buffer.** candle
allocates every output fresh, so every node's parameters would change each
token and patching them all buys nothing. Stable buffers really are the
prerequisite.
⚠ Trap to carry in: the `cudaKernelNodeParams` from
`cudaGraphKernelNodeGetParams` is **owned by the node** (#12152) — patch the
values it holds, never swap in your own pointers.

**4. Graphs are worth ~1.2x, batch-1 only — and likely MORE here.** That figure
is Llama 7B on an **H100**, where a launch costs ~3-5 us; this box measures
~10-12 us, so the same removal of submissions should buy proportionally more.

**5. Do NOT copy their concurrent streams, and know why.** llama.cpp also
parallelises Q/K/V across streams. NVIDIA describe the problem it solves as
*"GPU-side activities associated with each kernel launch"* and **gaps between
kernels** — the GPU idling between dependent launches. **Ours is CPU-side**: the
worker burns 22-26 ms of CPU per 21-24 ms of wall, 1.08 cores. Overlapping
streams does not reduce the CPU's submission work, so it fixes their bottleneck
and not ours. ⚠⚠ **And it would invalidate `disable_event_tracking`** (shipped
`939a86ca`), which rests on there being exactly one stream — see the hazard note
on that patch.

### Stage 3 — Reuse activation buffers instead of allocating 657 per token

**657 allocs + 656 frees, ~3.45 ms.** Every candle op allocates its output
fresh. `cuMemAllocAsync` is pooled, so this is cheaper than `cudaMalloc` would
be, but it is still ~1,300 submissions a token for buffers whose shapes repeat
identically on **every** token.

A decode step's activation shapes are fully determined by (model, batch, one
position). So an arena keyed on the shape sequence, allocated once per model and
reused, removes both counts almost entirely.

⚠ **This is also what unblocks stage 4**, and that is the reason to do it before
the more attractive-looking graph work. See below.

### Stage 4 — Capture the decode step as a CUDA graph

**625 launches, ~15 ms — the largest single line, and the reason the thread is
saturated.** A CUDA graph records a kernel sequence once and replays it with a
single submission, which is the canonical fix for a launch-bound decoder;
llama.cpp added exactly this (`GGML_CUDA_USE_GRAPHS`) for exactly this reason,
and it helps small models on decode most, which is the shape seen here.

cudarc 0.17.8 has the API: `CudaGraph::begin_capture` / `end_capture` /
`launch`.

⚠ **A graph replays against FIXED device pointers.** Today every op allocates a
new buffer per token, so a captured graph would replay against addresses that no
longer belong to it — the failure would be wrong numbers, not an error. **So
stage 3 is not an optimisation to be done first for tidiness; it is the
precondition.** llama.cpp's alternative is to keep the graph and *update* its
kernel parameters per token rather than recapture, which is worth reading before
choosing.

Also needs: one graph per distinct decode shape (KV length changes the attention
kernel's parameters, not usually its shape), and a fallback path for the first
token and for prefill.

**This is the highest-value stage and the highest-risk one.** Do not start it
before stages 2-3 have shown the instrumentation and the A/B discipline work.

### Stage 5 — Fuse kernels, which helps both backends

Independent of graphs, and the only stage that also helps the CPU backend (which
§ `gqa_decode_attention_cpu` records as dispatch-bound too):

- **Fuse the residual add and the RMS norm** into their neighbours. 28.4 launches
  per layer for ~7 matmuls means over half the launches are small elementwise
  passes.
- **Chase the ~30 `cuMemcpyHtoDAsync_v2` per token.** ⚠ **They are FIXED per
  token, not per layer** — measured 30.3/token on a 22-layer model against
  31.8 on a 28-layer one, i.e. +1.5 for +6 layers. The earlier reading of this
  line ("roughly one per layer") was wrong, and the two-model comparison is
  what settled it. So look in the per-token setup — embedding lookup, mask,
  LM head, the handoff to sampling — not in the blocks.
  Two are already identified and account for only two of them:
  `split/token_embedding.rs`'s `to_device` on the token ids, and
  `split/executor.rs`'s mask, both built on the host every forward. The
  remaining ~28 need nsys backtraces (`--sample=cpu`) to attribute; worth
  ~1.3 ms/token, so do it when something else already needs a profile run.
- **Sample on the device.** The one `cuMemcpyDtoHAsync_v2` per token copies
  vocab-sized logits back to be sampled on the host — a blocking round trip on
  a box where a round trip measured ~70 us. Greedy and top-k are both
  expressible on-device.

## Expected stacking, honestly

| stage | submissions removed | measured / estimated |
|---|---|---|
| **1 + 2 together ✅** | 321 memsets + 2,625 event ops | **measured: +34% on BOTH a 1.1B and a 3B** — the only reading taken on a genuinely idle box, and the only one where median and best agree |
| 1 memsets ✅ | 321 of 992 | +30% / +15% ⚠ contended box, superseded by the row above |
| 2 event tracking ✅ | **2,625 → 0 event ops** | +15% / nothing ⚠ same caveat |
| 3 buffer reuse | ~1,313 alloc/free | estimated ~3.5 ms/token |
| 4 CUDA graphs | most of 625 launches | large, unestimated |
| 5 fusion + D2H | tens of launches, 1 round trip | modest, and helps CPU too |

⚠ **These do not simply add, and measuring the pair proved it.** Compounding the
two per-change figures predicted +50% on the 1.1B and +15% on the 3B; measuring
both switches together on an idle box gave **+34% on each**. The per-change arms
were the contended ones, so the pair is what to trust. **Measure the combination
you intend to ship, not the sum of the parts** — and note the corollary: the
"small models gain more" story did not survive a quiet box.

Once submissions stop being the binding constraint, bandwidth becomes it, and
the 3B's ~5 ms floor is where this ends. Re-measure rather than projecting —
three estimates from sizes × intervals were each 10-100x wrong elsewhere in this
project (#673).

## What NOT to do

- **Do not chase a faster matmul first.** The matmuls are ~14 of 28.4 launches
  per layer and the arithmetic is a minority of the token. A 3-9x faster
  quantized matmul once moved prompt processing 1.15-1.24x, which is the note
  `inference/prof.rs` opens with.
- **Do not quote nsys per-call TIMES** as the real cost, or the tinyllama delta
  for a 7B.
- **Do not judge a submission-count change by the clock alone.** The count is
  deterministic; tok/s here spreads 10-18% run to run. Re-run
  `examples/decode_submissions.sh` and show the count moved.
- **Do not disable event tracking without answering the one-device question**
  in stage 2. Cross-stream corruption would show up as wrong replies.
- **Do not assume this reproduces on native Linux at the same magnitude.**

## Open unknowns

- **What a native-Linux GPU node measures.** Everything here is WSL2. The fleet
  has one GPU node. A tester with an NVIDIA card on native Linux running
  `examples/decode_bound_by.py` would settle how much of this is ours and how
  much is the platform — and it is one command that needs no build.
- **What the remaining ~28 host→device copies per token are.** Now known to be
  fixed per token rather than per layer, which is where to look; two of the ~30
  are identified. Stage 5.
- ~~**Whether the worker ever builds more than one `CudaDevice`.**~~ ANSWERED
  2026-09-22: it builds several, and they all share the **default** stream, so
  stream count is one. Stage 2 shipped on that.
- **How many launches a layer really costs.** Differencing the two models gives
  a marginal ~15.6 launches and ~23 allocations per layer — but they differ in
  vocabulary (32k vs 128k), KV-head count and tying as well as depth, so that is
  two points with several variables and **not** a number to plan from. A third
  and fourth model, or per-kernel-name counting inside
  `CudaDevice::get_or_load_func`, would make it real.
- **Whether the CPU backend's dispatch overhead has the same shape.** It is
  known to be dispatch-bound; nobody has counted anything there.
- **How this interacts with speculation.** A verify pass over γ drafted tokens
  is ONE forward pass, so on a submission-bound decoder speculation should pay
  close to the accepted-token count rather than being capped by arithmetic.
  n-gram lookup is already on by default (`num_pred_tokens = 10`) with a
  measured payoff gate, so the machinery exists; what its payoff looks like
  once submissions are cheaper has not been measured.
