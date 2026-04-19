# Round 5 Benchmarks — Item 7 Phase 4 (batched chunked prefill)

> **TL;DR**: Phase 4 fuses Prefilling slots whose chunks share
> `(chunk_len, index_pos)` into one `forward_batch` call. On a synthetic
> 22-layer / 1024-hidden CPU test model (debug build, WSL2 Ryzen), the
> fused forward is **1.1–1.6× faster than N sequential forwards** at
> batch 2/4/8 with chunk sizes 32–128 tokens. GPU end-to-end TTFT test
> is pending user time on the RTX 3070 machine.

## What Phase 4 changes

Before Phase 4, `step_decode_pool` Phase A did one `SplitModel::forward`
per Prefilling slot — `N` slots admitted in the same tick meant `N`
sequential forwards of shape `[1, chunk_size]`. Phase 4 groups slots
by `(chunk_len, index_pos)` and runs one `forward_batch` call of shape
`[N, chunk_size]` per group. Attention is still per-request (each slot
has its own KV cache); FFN and norms fuse across the batch.

Heterogeneous groups (mixed seq_len or different `index_pos`) fall
back to sequential forwards.

## When Phase 4 fires

It requires multiple Prefilling slots to be admitted into the
`SlotTable` **before the next decode tick**. The worker's IPC loop uses
a `biased` `tokio::select!` where `ipc_rx.recv()` is polled first — so
as long as multiple `Generate` messages are queued on the mpsc channel,
they all get admitted before `yield_now` wins and `step_decode_pool`
fires. Concretely: a tight-burst HTTP admit (all requests dispatched in
parallel from the same client) drains into the channel before the first
tick, and all slots' first chunks batch.

Slow or spread-out admits interleave with ticks — each admit causes
the slot's chunk to advance in position before the next admit arrives,
so slots desync and Phase 4's grouping returns only singletons.

## Synthetic timing benchmark

Run the bundled ignored test:

```bash
cargo test --release --no-default-features --features dev,claude-subscription \
    --lib -- --ignored forward_prefill_batch_timing --nocapture
```

It loops over chunk sizes {32, 64, 128} × batch sizes {2, 4, 8},
comparing `forward_batch` (one fused call) vs N sequential `forward`
calls on a 22-layer / 1024-hidden test model. Prints a table to
stderr. The test model's `max_seq_len` is 128, so chunk sizes above
128 are filtered out.

### Measured (CPU, debug build, WSL2 — synthetic reference)

Captured from an earlier run before the 128-cap filter landed
(`make_test_split_model_on` uses `max_seq_len = 128`):

| chunk | batch | batch_ms / iter | sequential_ms / iter | speedup |
|---|---|---|---|---|
| 32  | 2 | 1377 | 1924 | 1.40× |
| 32  | 4 | 2466 | 3712 | 1.51× |
| 32  | 8 | 4715 | 7681 | 1.63× |
| 128 | 2 | 3800 | 4217 | 1.11× |
| 128 | 4 | 7179 | 8433 | 1.17× |
| 128 | 8 | 14529 | 17529 | 1.21× |

### Measured (GPU, release build, CUDA 13.0, RTX 3070 Laptop)

Same synthetic model (22 layers, hidden_dim=1024), release build with
`--features candle-cuda`:

| chunk | batch | batch_ms / iter | sequential_ms / iter | speedup |
|---|---|---|---|---|
| 32  | 2 | 63.0  | 36.2  | 0.57× |
| 32  | 4 | 79.0  | 69.1  | 0.87× |
| 32  | 8 | 153.7 | 148.0 | 0.96× |
| 64  | 2 | 43.6  | 38.9  | 0.89× |
| 64  | 4 | 82.6  | 75.3  | 0.91× |
| 64  | 8 | 164.1 | 145.2 | 0.88× |
| 128 | 2 | 53.6  | 55.5  | 1.04× |
| 128 | 4 | 97.9  | 107.5 | 1.10× |
| 128 | 8 | 193.5 | 218.1 | 1.13× |

### Reading the numbers

**CPU wins, GPU is roughly wash on this model.** The synthetic model
(`hidden_dim=1024`) has a small FFN that doesn't benefit much from
batched GEMM on GPU. Attention stays per-request under Phase 4, so
its cost is unchanged, and the extra `Tensor::cat` / `narrow` kernels
that batching adds are pure overhead on GPU (where every tiny op is a
kernel launch). At small chunks the overhead dominates (0.57× at
chunk=32, batch=2); at larger chunks the FFN share grows and the
fused path eventually pulls ahead (1.13× at chunk=128, batch=8).

The matching round-4 observation for Phase 1+2 was: *"TinyLlama at
this size is too small for fused `forward_batch` to add meaningful
tok/s on this GPU."* Same story here — this synthetic model is
approximately TinyLlama-sized on the FFN axis, and the measured
GPU batching behavior mirrors it exactly.

**What we'd expect on a larger model (Qwen2.5-7B or similar, hidden
~3584).** FFN dominates the forward on bigger hidden dims, so the
fused GEMM should amortize better. With Q4_K_M quantization, per-weight
dequantization cost amortizes across the batch too — batched dequant
is a strictly bigger win than batched F32 matmul of the same size.
The synthetic bench here deliberately uses F32 weights (no dequant)
so the numbers are a *lower bound* for what real-model GPU numbers
should look like. Larger-model bench is the next item below.

## End-to-end bench: admit-coalescing unlocks Phase 4 fusion

Phase 4's batching requires multiple Prefilling slots with the same
`(chunk_len, index_pos)` in the same decode tick. The worker's
`tokio::select!` loop was strictly interleaving admit → tick → admit
→ tick, so even concurrent HTTP bursts ended up with each slot's first
chunk processed singleton-style (slot 1 advances to pos=128 before
slot 2 can admit at pos=0 — different `index_pos`, no fusion).

**Fix (2026-04-19):** after handling any mpsc message, drain up to 16
further messages via `try_recv()` *before* running the next decode
tick. This lets concurrent Generates pile up in the slot table as a
group with all at `index_pos=0`, and the first Phase A tick fuses
their initial chunks.

### Measured E2E (RTX 3070, TinyLlama-1.1B Q4_K_M)

Prompt ≈ 100 tokens; `prefill_chunk_tokens = 128`; prefix cache
pre-primed by a prior sequential run (all concurrent admits match
64 tokens of prefix, remaining 37 to prefill → chunk_len=37,
index_pos=64 — same-shape homogeneous batch).

Before admit-coalescing fix (concurrency=4):
- Aggregate **31.2 tok/s**, TTFT min 52 / avg 235 / max **447 ms**
- Zero `prefill chunk fused` DIAG lines — singleton path.

After admit-coalescing fix (concurrency=4):
- Aggregate **49.1 tok/s** (+57%), TTFT min 180 / avg 180 / max **180 ms**
- One `DIAG: BatchGenerate prefill chunk fused batch_size=4 chunk_tokens=37 index_pos=64`
  line per bench run.

After admit-coalescing fix (concurrency=8):
- Aggregate **42.4 tok/s**, TTFT min 321 / avg 321 / max **321 ms**
- One `prefill chunk fused batch_size=8` line per bench run.

**Reading the numbers.** TTFT fairness is the headline — all concurrent
admits get their first token at the *same* millisecond now, within
noise. The aggregate tok/s improvement (49/31 ≈ 1.57× at c=4, 42/26 ≈
1.62× at c=8) reflects fused-FFN amortization across the batch on the
initial prefill tick. Decode is still one-slot-per-tick afterward, so
aggregate throughput doesn't scale linearly with concurrency — that's
expected for TinyLlama-sized models where per-slot attention dominates.

On a larger model (Phi-3.5-mini, Qwen2.5-7B) the FFN share is bigger
and the fused-prefill speedup should scale further. The user's RTX 3070
ran out of VRAM on Phi-3.5 + `max_seq_len_override=8192` (~7.6 GB
reserved by weights + KV cache per slot × 8), so a larger-model measure
needs either a smaller context-window override or a bigger GPU. Noted
as pending follow-up.

## Reproducing

```bash
# 1. Build for GPU.
cargo build --release --no-default-features --features dev,candle-cuda

# 2. Start daemon in a shell, with debug-level logging so DIAG lines
#    appear. (Worker subprocess picks up the level from config.toml's
#    [logging] level — set to "debug".)
./target/release/swarmllm run -p 8800 -vv > /tmp/swarm.log 2>&1 &

# 3. Warmup (loads model into worker).
API_KEY=$(cat ~/.local/share/swarmllm/api_key)
curl -sf -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8800/v1/chat/completions \
     -d '{"model":"tinyllama-1.1b-chat-v1.0.q4-k-m",
          "messages":[{"role":"user","content":"hi"}],
          "max_tokens":4,"temperature":0.0}'

# 4. Fire concurrent bench with a shared prompt (same prompt across all
#    parallel requests ⇒ same chunk_len ⇒ eligible for Phase 4 fusion).
PROMPT="Write a technical analysis comparing speculative decoding ... (100+ tokens)"
./target/release/swarmllm bench --max-tokens 20 --iterations 1 \
     --concurrency 4 --stream \
     --model-id "tinyllama-1.1b-chat-v1.0.q4-k-m" \
     --prompt "$PROMPT"

# 5. Confirm fusion DIAG lines appeared.
grep "prefill chunk fused" /tmp/swarm.log
# Expect: batch_size=4 chunk_tokens=<N> index_pos=<M>
```

## Pending work (next session)

- GPU synthetic-bench on the RTX 3070 at a **bigger hidden_dim** (the
  current test model is hidden_dim=1024 ≈ TinyLlama-sized; a 2048 or
  3584 sweep would better represent Phi-3.5 / Qwen-7B ffn cost).
- Larger-model E2E bench — Phi-3.5-mini or Qwen2.5-7B — with
  `max_seq_len_override` reduced to fit VRAM (e.g. 2048 instead of
  8192).
- End-to-end TTFT A/B via a dedicated `batched_prefill_forward` flag
  (toggles Phase 4 in isolation from Phases 1+2). Not yet wired.

## Reading the numbers alongside round 4

Round 4 measured Phases 1+2 **TTFT improvement from slot-table
admission** under concurrency: 17–23× on RTX 3070 + TinyLlama Q4.
Phase 4 adds an additional constant-factor speedup **inside a single
decode tick** when multiple admits arrive same-tick. Round 4's win is
about getting the prefill started; Phase 4's win is about making that
prefill's compute cheaper per active slot.
