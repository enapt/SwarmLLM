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
stage depends on a later one. ⚠ **This section originally said "Stage 3 is a
prerequisite for Stage 4"; that is WITHDRAWN** — see § Ordering item 3. Stage 4's
real preconditions are in item 3b, and the first of them is a blocker.

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
that fusion could make two; `affine_f32` + `softmax_f32` are two launches for the
attention tail that `scaled_masked_softmax` already describes as one operation.
⚠ **This list first named `bmul_f32` in the attention tail as well. It is not
there** — Stage 2c's A/B took `bmul_f32` from 1.00/layer to **zero**, so its only
caller was the gated activation. Both need a new CUDA kernel, and `candle-kernels` is a
registry crate rather than a vendored one — so that means either vendoring it or
using candle's `get_or_load_custom_func` path with our own module — which is
what Stage 2c below did, so the route is no longer hypothetical.

### Stage 2c — silu(gate) × up as ONE kernel ✅ SHIPPED (count-verified, no speed claim)

**The first kernel of our own, and the proof that the route in works.**
`kernels/fused_decode.cu::silu_mul_f32` replaces candle's `usilu_f32` +
`bmul_f32`; `build.rs` compiles it with `nvcc --ptx` and
`CudaDevice::get_or_load_custom_func` loads it. **No fifth vendored crate.**

Measured on tinyllama (22 layers), `examples/kernel_count_ab.sh
SWARMLLM_FUSE_SILU_MUL 1 0`:

| kernel | on | off |
|---|---|---|
| `silu_mul_f32` | **1.00/layer** | — |
| `usilu_f32` | — | 1.00/layer |
| `bmul_f32` | — | 1.00/layer |
| **total launches/token** | **513** | 535 |

−1.00 launch per layer (−22/token), plus the allocation and free that op no
longer makes: **−66 submissions per token**. **Replies identical.**

⚠ **No tok/s figure is quoted and none should be.** −66 of ~1,900 submissions is
~3%, and this box cannot resolve a decode change below ~10%. This ships on the
count and on bit-identity, the same basis as .198's prefill half. **Claiming a
speed-up here would be inventing one.**

⚠ `bmul_f32` went to **zero**, not to 1.00 — its only caller was this multiply.
The attention tail is `affine_f32` + `softmax_f32`, so the earlier note that it
also used `bmul_f32` was wrong.

**Two instrument bugs had to be fixed first, and both would have flattered this
result:**

1. **`get_or_load_custom_func` did not count launches** — only
   `get_or_load_func` did, and our PTX kernels are the only thing that takes the
   custom path. The table would have shown −2.00/layer for a change worth −1.00.
2. **`SWARMLLM_COUNT_KERNELS=1` printed nothing at all** at the default log
   level; the reporting block was gated on DEBUG-or-`SWARMLLM_PROFILE`
   (gotcha #681).

`SWARMLLM_FUSE_SILU_MUL=0` is the off arm.

### ▶ The add+RMS-norm fusion, designed but NOT built (2026-09-22)

The bigger of the two, and the design question that makes it bigger is worth
recording rather than re-deriving. Our pattern is

```
x      = attn + residual          badd_f32     ← needed later, as the residual
normed = rms_norm(x) * weight     rmsnorm_f32  ← fed to the FFN
```

**Two outputs are genuinely required**, and candle's `CustomOp` returns one
storage. The way through, verified rather than assumed:

- Allocate **one buffer of 2N** and have the kernel write the sum into the
  first half and the normed value into the second.
- Split it with `narrow(0, i, 1)?.reshape(orig)?`. **Both are zero-copy here** —
  `narrow` on dim 0 of a contiguous tensor keeps contiguous strides, and
  `Tensor::reshape` takes the `is_contiguous()` branch, which builds
  `Layout::contiguous_with_offset(shape, start_offset)` and copies nothing
  (`vendor/candle/candle-core/src/tensor.rs`). Checked in the source, because
  the whole saving would be given back by one hidden `copy_strided_src`.
- Cost: **−1 launch, −1 alloc, −1 free per site, ×2 sites = −6 submissions per
  layer**, against the silu×up fusion's −3.

⚠ **Do NOT copy llama.cpp's `rms_norm_f32` here — it fuses the OTHER order.**
Theirs is `RMS_NORM → MUL → ADD`, ending `dst[col] = scale * x[col] *
mul[mul_col] + add[add_col]`, with a `static_assert(!do_add || do_multiply)`.
Ours is `ADD → RMS_NORM → MUL`, which upstream has only as a separate
`add_rms_norm` path that is architecture-gated. **Same three ops, different
graph, and the kernel is not interchangeable.**

⚠ And their fused-GLU work goes further than ours does: `ggml_cuda_should_fuse_mul_mat`
fuses **`ffn_up` MUL_MAT + `ffn_gate` MUL_MAT + GLU**, writing the activated
result straight out of the matmul epilogue. `QMatMul::forward_shared` (shipped
in .198) is the *activation-sharing* half of that; the epilogue half — a fused
SwiGLU tail on `mul_mat_vec_via_q8_1` — would remove the gate/up intermediates
as well, and is the follow-up if the elementwise fusion pays.

### ▶ Ordering REVISED 2026-09-22 after reading how llama.cpp did this

The stages below were ordered by size of the line in the budget. Reading
llama.cpp's own decode work reorders them.

⚠ **Re-read the same day, and the second pass changed three of its claims** —
two of them in the direction that flattered the plan. The corrections are
inline below, marked; gotcha **#680** carries what to do differently. **The
re-ordering itself survived; the reasons for it did not.**

**Sources**: [NVIDIA on CUDA graphs in llama.cpp](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)
· [am17an, token-generation optimizations](https://am17an.bearblog.dev/new-post/)
(llama.cpp discussion #17621) · [issue #12152](https://github.com/ggml-org/llama.cpp/issues/12152)
· [CUDA Programming Guide § CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
(graph memory nodes, fixed addresses, update rules)
· [CUDA graph capture constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html)
(the legacy-stream prohibition, what may be allocated during capture)
· [huggingface/grout](https://github.com/huggingface/grout) — a Rust decoder that
captures decode as a graph "once per sequence length class"

**1. Fusion now comes FIRST.** llama.cpp measured **329 → 419 tok/s (~27%)** on
an RTX 5090 / gpt-oss-20b from a set of decode fusions, two of which map onto
our kernel table:

| their fusion | our kernels, per layer |
|---|---|
| RMS-norm fused with the preceding multiply/add | `rmsnorm_f32` 2.05 + `badd_f32` 2.00 |
| GEMV fused with the gated activation | `usilu_f32` 1.00 + `bmul_f32` 1.00 |
| TopK-MoE (softmax + expert select) | MoE only — would also remove `topk_cpu`'s host round trip |

⚠ **CORRECTED 2026-09-22 (same day): there is no "~10% each" in the source, and
this document asserted one.** What am17an writes is *"none of these PRs increase
the TG by more than 10%"* — a **CEILING on each**, published to explain why the
combined 27% is the number worth quoting. Reading it as a point estimate turned
someone else's upper bound into our forecast, which is how a plan acquires a
number nobody measured. **The honest statement is: each of these was worth
something under 10% to them, on their hardware, against their budget — and their
budget has no allocation line at all** (ggml plans one compute buffer; ours
allocates per op, § Stage 3). Size our fusions from OUR kernel table, and quote
theirs only as the ceiling it is.

Their reasoning is ours: *"fusing kernels reduces memory traffic and kernel
launch time… token generation is memory-bound rather than compute-bound"*.

**2. Where our fused kernels go, without a fifth vendored crate.**
`candle-kernels` is a registry crate, but `CudaDevice::get_or_load_custom_func`
takes **PTX as a string** and had no callers upstream — so a small `.cu` compiled to PTX
by our own `build.rs` under `candle-cuda` loads through it. That is the cheap
route in, and it makes each fusion independently shippable and A/B-able.

**3. ~~Stage 3 before stage 4 is CONFIRMED~~ — WITHDRAWN 2026-09-22, same day.**
The argument was: llama.cpp patches only the KV-cache pointers in an
already-instantiated graph each token (`cudaGraphExecUpdate` for the rarer
structural change), **which works because its ACTIVATION addresses are already
stable in a fixed compute buffer**; candle allocates every output fresh, so
every node's parameters would change each token and patching them all buys
nothing.

Every sentence of that is true, and the conclusion still does not follow,
because it assumes the only way to build the graph is *capture once, then patch
per token*. **There is a second way, and it is the one that suits candle.**
Stream capture turns `cuMemAllocAsync` / `cuMemFreeAsync` into **graph memory
nodes**, and the CUDA Programming Guide is explicit about what that buys:
*"Graph allocations have fixed addresses over the life of a graph including
repeated instantiations and launches."* The allocations candle makes during
capture become part of the graph and hand back **the same addresses on every
replay** — so there is nothing to patch, and a graph would remove most of the
alloc/free line (~1,300 submissions) as well as the launch line.

**So stage 3 is not a prerequisite.** ✅ **MEASURED 2026-09-22, not deduced** —
`examples/cuda_graph_probe.cu` on this box (RTX 3070 Laptop, sm_86, driver
13040, WSL2): a `cudaMallocAsync` issued *inside* a capture, used by two
kernels and freed inside the same capture, replays **correctly three times out
of three** against a persistent buffer zeroed before each replay. The pointer
handed out during capture (`0xa00000000`) is a graph-reserved address, not a
pool address.

⚠ **This establishes the PLATFORM, not the program.** A real decode step still
has to satisfy item 3b — in particular it must free inside the capture
everything it allocates there, and contain no host synchronisation. What the
probe removes is the *reason* stage 3 was called a prerequisite; the remaining
preconditions are about our code, not about CUDA.
✅ **And it answers a second open question for free: graph capture works under
WSL2 on this driver.** That was not safe to assume — capture was crashing on
WSL2 + Blackwell until WSL 2.7.0.
⚠ Trap to carry in either way: the `cudaKernelNodeParams` from
`cudaGraphKernelNodeGetParams` is **owned by the node** (#12152) — patch the
values it holds, never swap in your own pointers.

**3b. The capture preconditions nobody had written down.** Found by reading the
capture rules rather than the graph rules, and all four are checkable before any
code is written:

- ⛔ **Capture is IMPOSSIBLE on the stream candle uses today.** *"Stream capture
  can be used on any CUDA stream except `cudaStreamLegacy`"* — and
  `BackendDevice::new` takes `context.default_stream()`, which cudarc defines as
  `cu_stream: null_mut()`, i.e. exactly that stream. This is the same fact
  Stage 2 turned on; it cuts the other way here.
  ✅ **CONFIRMED on this driver, not just in the docs**: `cuda_graph_probe.cu`
  arm A is a null control that would report the plan wrong if capture were
  permitted. It is refused with `cudaError 900`.
  ⛔ **`per_thread_stream()` looks like the way out and is NOT.** It is
  capturable, and unlike `new_stream()` it does not flip cudarc's
  `is_in_multi_stream_mode()` — so it appears to keep Stage 2 valid for free.
  But per-thread means **one stream per OS thread**, and this forward is not
  pinned to one: `cpu_pools::in_phase_pool` runs it via `pool.install(f)` on a
  rayon worker, and *which* pool depends on the calibration state. Different
  tokens would land on different streams with event tracking off, which is the
  unsynchronised-buffer hazard — a wrong reply, not an error.
  ✅ **The way that survives the check is one explicitly created stream**
  (`new_stream()`) with **every** `CudaDevice` on it. Event tracking can stay
  off, because "one stream in use" is what that rests on, not "the default
  stream" — but note candle builds several devices and each takes
  `default_stream()` today (Stage 2's finding), so this is a change to
  `BackendDevice::new`, not a call site. ⚠ A half-migration, with some devices
  on the new stream and some on the legacy one, is worse than either end state.
- **Every `cuMemFreeAsync` inside the capture must free memory allocated inside
  the same capture.** A tensor that existed before the region and drops inside
  it aborts the capture.
- **No host synchronisation inside the region**, which puts the logits
  `cuMemcpyDtoHAsync` + sync at the boundary: capture the forward, sample
  outside — or do Stage 5's on-device sampling first and capture the lot.
- **The KV length changes every token**, so a graph is valid for one length
  unless the attention kernel stops taking it as a launch parameter. HuggingFace's
  own `grout` compiles *"once per sequence length class (prefill vs decode)"*,
  which is the shape to aim at; llama.cpp instead patches per token.

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

⚠ ~~**This is also what unblocks stage 4**~~ — **withdrawn**, see § Ordering
item 3. If capture turns these allocations into graph memory nodes, stage 4
removes this line too and stage 3 has no separate reason to exist. **Run that
experiment before spending a day here.**

### Stage 4 — Capture the decode step as a CUDA graph

**625 launches, ~15 ms — the largest single line, and the reason the thread is
saturated.** A CUDA graph records a kernel sequence once and replays it with a
single submission, which is the canonical fix for a launch-bound decoder;
llama.cpp added exactly this (`GGML_CUDA_USE_GRAPHS`) for exactly this reason,
and it helps small models on decode most, which is the shape seen here.

**cudarc 0.19.9** has the API: `CudaStream::begin_capture` / `end_capture` and
`CudaGraph::launch`.
⚠ **An earlier draft of this plan cited cudarc 0.17.8, which is the wrong
crate.** 0.17.8 IS in `Cargo.lock` — pulled by `ug-cuda`, behind candle's
optional `ug` feature, which this build does not enable. The version
`candle-core` actually builds against, and therefore the one whose `CudaStream`
our device holds, is **0.19.9**. The API exists in both, so the conclusion
survived; the check did not. `workflow.md` § research item 2 says to read the
registry source rather than recall it — this is what happens when you read the
lock file instead of the dependency.

⚠ **A graph replays against FIXED device pointers.** That is true and is the
whole design constraint — but see § Ordering item 3: **graph memory nodes give
candle's per-op allocations fixed addresses for free**, so this does not by
itself make stage 3 a precondition. llama.cpp's alternative is to keep the graph
and *update* its kernel parameters per token rather than recapture, which is
worth reading before choosing.

Also needs: a decision on how the changing KV length is handled (a graph per
length class, as HuggingFace's `grout` does, or per-token parameter patching, as
llama.cpp does), and a fallback path for the first token and for prefill. The
four capture preconditions are in § Ordering item 3b — **the stream one is a
blocker, not a detail.**

### ▶ Stage 4a — move every `CudaDevice` off the legacy stream ✅ SHIPPED

The precondition, done on its own so that the graph change has one variable.
⛔ **SHIPPED BROKEN and REVERTED to opt-in.** `BackendDevice::new` takes
`context.new_stream()` only under `SWARMLLM_CUDA_OWN_STREAM=1`; the default is
the legacy stream again. **One line, because everything follows the device's
stream**: `CudaBlas::new` and `CudaRng::new` are handed it (cublas via
`cublasSetStream_v2`), `candle-flash-attn` takes `dev.cuda_stream()`, every
launch uses `self.stream.launch_builder`, `synchronize()` syncs `self.stream` —
and `default_stream()` had exactly one use in the whole backend.

⚠ **It changes the shape of the event-tracking argument**, which is why the
comment on that patch and `docs/invariants/inference.md` were both rewritten:
all devices used to share the legacy stream, so there was one stream
process-wide; now each device has its own and the invariant is **one stream per
device, buffers never crossing devices** (candle refuses cross-device tensors).
⚠ And `SWARMLLM_CUDA_EVENT_TRACKING=1` stopped being a pure revert — with
multi-stream mode now true it also hands cudarc back stream-sync management.

### ▶ Stage 4b — what capturing OUR decode step still has to solve

The probe establishes the platform. These are the program's problems, found by
reading the forward rather than the CUDA docs, and each needs an answer before
any capture code is written:

1. **A tensor allocated BEFORE the region must not drop INSIDE it.**
   ✅ **MEASURED, and the outcome is worse than an abort.** `cudaFreeAsync` on
   memory allocated outside the capture returns **`invalid argument`
   (cudaError 1) while the capture SURVIVES** and `cudaStreamEndCapture`
   succeeds. So the memory is simply **not freed**: every captured token leaks
   whatever `layer_in` held on entry, and candle's `Drop` records the error
   rather than raising it at the cause. An abort would have been kinder.
   **Fix: hold a clone alive across the capture** so nothing pre-existing drops
   inside it.
2. **Everything allocated INSIDE must also be FREED inside — including the
   logits.** ⛔ **MEASURED, and this one is fatal, not cosmetic.** A graph with
   an allocation that is still live cannot be relaunched: replay 1 succeeds,
   **replay 2 fails with `invalid argument`**. Decoding token N+1 would simply
   stop. The docs say an unfreed graph allocation "persists"; they do not say
   the graph becomes unreplayable while it does.
   **So the captured region cannot hand a tensor out.** The logits have to be
   copied into a buffer allocated OUTSIDE the capture and the graph-allocated
   one freed inside — which also disposes of the aliasing worry this item used
   to carry.
   ⚠ Both results come from a scratch probe whose FIRST verdict for item 1 was
   wrong: it printed the failing return code and then computed "ALLOWED" from
   `end_capture` alone. Same shape as #614 — **read the output, not the one
   status you happened to branch on.**
3. **No host synchronisation inside the region.** The logits D2H happens after
   `SplitModel::forward` returns, so the natural boundary is the forward itself.
   Sampling on the device (Stage 5) would let the whole step be captured.
4. **KV length changes every token**, so the attention kernel's parameters move.
   Either a graph per length class (HuggingFace's `grout`) or per-token
   parameter patching (llama.cpp). ⚠ `cudaGraphKernelNodeGetParams` hands back
   params **owned by the node** (#12152) — patch the values, never swap in your
   own pointers.
5. **Capture must not run concurrently with another request** on the same
   device. The worker is one model per process, but the prefill/decode paths and
   the capability probe share a context.

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
| **2c silu×up fusion ✅** | 22 launches + 22 allocs + 22 frees | **count-verified −66/token; NO speed claim — below this box's resolution** |
| 3 buffer reuse | ~1,313 alloc/free | estimated ~3.5 ms/token — **NOT a prerequisite for stage 4** (measured, § Ordering item 3); worth doing only on its own merits |
| 4 CUDA graphs | most of 513 launches, **and the 1,313 alloc/free with them** | large, unestimated — and now the clear next big move |
| 5 fusion + D2H | tens of launches, 1 round trip | modest, and helps CPU too |

⚠ **These do not simply add, and measuring the pair proved it.** Compounding the
two per-change figures predicted +50% on the 1.1B and +15% on the 3B; measuring
both switches together on an idle box gave **+34% on each**. The per-change arms
were the contended ones, so the pair is what to trust. **Measure the combination
you intend to ship, not the sum of the parts** — and note the corollary: the
"small models gain more" story did not survive a quiet box.

⚠⚠ **Fusion and graphs are not additive in a deeper way: they bill the same
cost.** A fused kernel is worth a launch, an allocation and a free per layer —
but a CUDA graph replays the whole sequence with ONE submission, so under a
graph those savings are already taken and what is left of fusion is only the
memory traffic and the smaller node count. **Do not plan on fusion's win
surviving stage 4**, and do not let stage 4's size be argued from a launch count
that fusion has already reduced. The reason to do fusion first is not that it
compounds:

- it is **independently shippable today**, where stage 4 has four unmet
  preconditions and one of them is a blocker;
- it is **low risk** — an elementwise kernel with a bit-identical reference;
- it is **the only stage that also helps the CPU backend**, which is
  dispatch-bound too and will never get a CUDA graph;
- and it **proves the PTX route in**, which every later fused kernel needs.

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
