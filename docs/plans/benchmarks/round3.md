# Round 3 Benchmarks — 2026-04-18 (CPU), 2026-04-19 (GPU)

Captures measured throughput for the Item 3 Phase 2b fused-batch forward and summarises
known numbers for items landed earlier (4, 5, 12, 13, 16). Tied to commit
`365d73a` (Phase C offline allocator) + CPU-fallback guard in `run_fused_batch_forward`.

## Environment

- CPU: AMD Ryzen (WSL2), 8 cores exposed to Linux
- GPU: NVIDIA RTX 3070 Laptop (8 GB VRAM) — **NOT exercised in this round** (all results below are CPU unless marked)
- Rust: release build, `--features dev,claude-subscription`
- Test harness: `cargo test --release --lib <name> -- --nocapture --ignored`

## Item 3 Phase 2b — `SplitModel::forward_batch` on CPU

Runs the fused-batch path against the sequential `forward()` path on a
synthetic 22-layer, 1024-hidden-dim model on CPU. 20 iterations per batch size
(fresh KV per iter so each path does equivalent work).

| Batch | Fused ms/iter | Sequential ms/iter | Speedup |
|---|---|---|---|
| 1 | 145.5 | 132.6 | **0.91×** (batch overhead) |
| 2 | 487.8 | 265.8 | **0.54×** |
| 4 | 650.7 | 536.6 | **0.82×** |
| 8 | 1036.8 | 1079.8 | **1.04×** |

**Finding.** Fused batching is a net loss on CPU at every tested batch size.

**Why.** `SplitModel::forward_batch` batches the QKV + FFN projections but
keeps attention per-slot (each slot has an independent KV cache history).
On CPU:
- Matmul is memory-bandwidth-bound, not kernel-launch-bound — the batched
  QKV+FFN doesn't amortize a fixed per-call cost the way it does on GPU.
- `Tensor::cat` / `Tensor::narrow` carry a real per-layer cost: for a
  22-layer model at batch=4 we pay that overhead 22× per forward.
- The per-slot attention loop dominates total time; batching the rest doesn't
  help enough.

**Implication.** `continuous_batching` is not profitable on CPU-only nodes.
The wire protocol and worker fused path are correct (see unit tests
`forward_batch_matches_sequential` and `forward_batch_single_item_matches_forward`),
but the default-off flag is appropriate.

**Next step (done 2026-04-19).** Re-run on GPU (CUDA) where the kernel-launch
cost amortization is the usual batching win.

## Item 3 Phase 2b — `SplitModel::forward_batch` on GPU (CUDA, RTX 3070)

Same synthetic model (22 layers, 1024 hidden_dim, 20 iters), CUDA:0 backend.

| Batch | Fused ms/iter | Sequential ms/iter | Speedup |
|---|---|---|---|
| 1 | 18.6 | 12.3 | **0.66×** (batch=1 pays only the cat/narrow overhead — expected) |
| 2 | 18.9 | 25.2 | **1.34×** |
| 4 | 32.1 | 46.8 | **1.46×** |
| 8 | 64.7 | 99.9 | **1.55×** |

**Finding.** GPU delivers the predicted batching win. Per-kernel launch cost on
the QKV + FFN matmuls amortizes across slots; per-slot attention still
serializes but doesn't eat the gains because the batched projections dominate
at these sizes. At batch=1 the tensor stacking overhead slightly dominates —
expected, and why the worker's CPU-fallback guard also skips batch=1 already
via `batch_eligible` (minimum 2 requests).

Cross-platform policy now:
- **CPU**: `run_fused_batch_forward` short-circuits with an error (caller falls
  through to sequential). See the CPU-fallback guard in `model_worker.rs`.
- **GPU**: fused path delivers 1.34–1.55× at batch 2–8.

The `continuous_batching` config flag is now **safe to enable on any device** —
worker picks the profitable path per request automatically.

## Reference: known measurements from earlier rounds

Captured in `../distributed_inference_speedup.md`, restated here for easy
comparison.

### Item 4 — Remote-generate fast path (default-on)

3-node loopback, TinyLlama Q4_K_M, 100-token greedy:

| Path | 100-tok wall | Decode rate |
|---|---|---|
| Per-token distributed | ~30 s | ~270 ms/tok |
| Item 4 fast path | ~15.9 s | ~125 ms/tok |

**1.93× decode, 1.75× wall-clock** on the typical single-peer single-segment
path. Default-on.

### Item 5 — Cross-request prefix cache (default-on)

TinyLlama Q4_K_M CPU, 513-token prompt, `max_tokens=5`:

| Request | Latency |
|---|---|
| Cold | 41.66 s |
| Warm (cache hit @ 512) | 1.42 s |

**29.4× wall-clock speedup** on a repeated prompt. Default-on.

### Item 13 — Q8_0 activation compression (flag)

Codec: `~3.76× wire compression`, RMS error `< 0.005` on synthetic activation
slices, `< 1e-4` between non-outlier blocks. End-to-end multi-segment
throughput vs raw f32 pending — single-segment fast path (Item 4) bypasses
hidden state transfer so Q8_0 only matters when `N >= 2` peers hold the
pipeline. Not measured in this round.

### Item 16 Phase A+B — Parallax routing (default-on)

Unit + integration tests confirm the DP picks the lower-cost chain over
greedy given synthetic peer signals. Empirical wins require a cluster with at
least one multi-segment route + at least one obviously-bad greedy choice the
DP would reject. Not captured in loopback (local fast path short-circuits
before the DP).

### Item 16 Phase C — Offline allocator

Recommendation-only in v1. Not wired into `ShardRebalancer` / `AutoShardManager`.
Output is advisory; not auto-applied. No benchmark needed yet.

## Reproduction

```bash
# Phase 2b CPU timing (takes ~90s)
cargo test --release --lib forward_batch_timing \
    --no-default-features --features dev,claude-subscription \
    -- --nocapture --ignored

# Phase 2b GPU timing (takes ~20s after build)
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    cargo test --release --lib forward_batch_timing \
    --no-default-features --features dev,candle-cuda \
    -- --nocapture --ignored

# Item 4 / Item 5 end-to-end require a running daemon + prompt injection;
# not automated in this session.
```

## Outstanding benchmarks

- **Item 3 Phase 2b**: ~~GPU benchmark~~ ✅ done 2026-04-19 (1.55× at batch=8). Next: wire a time-window batch scheduler into the router's batch dispatch path so concurrent arrivals auto-coalesce into `forward_batch`. Today callers must explicitly build a `Vec<LayerForward>` and call `ModelProcessPool::forward_batch()`.
- **Items 12 / 13 / 16 Phase A multi-segment** (real WAN topology): need a 2+-peer setup where no single peer covers all layers. Loopback doesn't exercise these paths because the local-full-coverage fast path takes over.
- **Parallax routing A/B** on a cluster with mixed peer latencies: compare 100-prompt aggregate tail latency with the flag on vs off.
