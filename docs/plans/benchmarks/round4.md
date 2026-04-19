# Round 4 Benchmarks — Item 7 BatchGenerate (recipe + run-yourself)

> Status: 2026-04-19. Phase 1 (worker SlotTable + slot-driven decode loop)
> + Phase 2 (Sarathi-style chunked prefill) both landed. This doc is the
> recipe for validating end-to-end concurrent-generate throughput against
> a live daemon. Numbers below are the **expected envelope** — fill in the
> measured column when you run.

## Why a separate recipe instead of a `cargo test`

Spawning a real `model-worker` subprocess in a `cargo test` context fails
because `std::env::current_exe()` resolves to the test binary, which has
no `model-worker` subcommand. The cleanest end-to-end harness is the
existing `swarmllm bench --concurrency N` CLI subcommand, which makes
real HTTP requests against `localhost:8800` and reports aggregate tok/s.

## Recipe

```bash
# 1. Build (any release flavour works; pick GPU if you have CUDA)
cargo build --release --no-default-features --features dev,claude-subscription
# or, if you have CUDA:
# cargo build --release --features candle-cuda

# 2. Verify TinyLlama is staged. If not, see memory/local_model_shards.md.
ls ~/.local/share/swarmllm/models/

# 3. Start the daemon. Defaults already enable continuous_batching = true,
#    so the worker spawns with --batch-generate true --prefill-chunk-tokens 128.
./target/release/swarmllm run -p 8800 -v >/tmp/swarm.log 2>&1 &

# 4. Wait for the daemon to be ready and load TinyLlama.
sleep 5
./target/release/swarmllm status
# Optionally trigger a load:
# curl -sH "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
#   -X POST http://localhost:8800/v1/chat/completions \
#   -d '{"model":"tinyllama","messages":[{"role":"user","content":"hi"}]}'

# 5. Sequential baseline (concurrency=1) — capture avg tok/s.
./target/release/swarmllm bench --max-tokens 100 --iterations 5 --concurrency 1

# 6. Concurrent run — capture aggregate tok/s.
./target/release/swarmllm bench --max-tokens 100 --iterations 1 --concurrency 2

# 7. Repeat at concurrency 4 and 8 to see how throughput scales with batch size.
./target/release/swarmllm bench --max-tokens 100 --iterations 1 --concurrency 4
./target/release/swarmllm bench --max-tokens 100 --iterations 1 --concurrency 8

# 8. Compare. Report avg sequential tok/s vs aggregate concurrent tok/s.
#    Expected: aggregate ≥ N * sequential * 0.85 (≥85% scaling efficiency)
#    on GPU. CPU falls through to sequential per-slot inside forward_batch
#    so concurrency=N gives ~1× throughput regardless (no regression, no win).
```

## Toggling Item 7 off for an A/B comparison

The two flags that gate the worker path:

```toml
# config.toml
[inference]
continuous_batching = false   # turns off both daemon-side coalescer
                              # AND worker-side BatchGenerate
prefill_chunk_tokens = 128    # Phase 2 chunk size
```

Restart the daemon to pick up changes (worker subprocesses get the values
at spawn time via CLI args).

For an A/B you want:
1. Run with `continuous_batching = true` (default) → record aggregate tok/s.
2. Restart with `continuous_batching = false` → record aggregate tok/s.
3. Speedup = (true) / (false).

## Expected envelope

### Phase 1 effect (slot-driven decode loop)

GPU (RTX 3070 8GB), TinyLlama-1.1B Q4, max_tokens=100, prompt ~50 tokens:

| Concurrency | Sequential tok/s/req | Aggregate tok/s | Speedup vs serial |
|---|---|---|---|
| 1 | ~50 | ~50 | 1.00× (baseline) |
| 2 | _measure_ | _measure_ | target ≥ 1.7× |
| 4 | _measure_ | _measure_ | target ≥ 3.0× |
| 8 | _measure_ | _measure_ | target ≥ 5.0× |

Single-GPU diminishing returns are expected past batch ~8 because the
per-slot attention (KV-cache lookup) doesn't batch — only the QKV / FFN
projections do (per Item 3 Phase 2b benchmarks at `round3.md`).

### Phase 2 effect (chunked prefill)

The win shows up under **mixed admission + decode**, not in steady-state
throughput. Procedure:

1. Start a long-running generate (max_tokens=200) — this will stay in the
   decode loop for ~4 seconds at TinyLlama GPU rates.
2. At the ~2-second mark, fire a second generate with a **long prompt**
   (e.g. 4 KB system prompt → ~1500 tokens).
3. Without Phase 2 (Phase 1 only), the first request stalls for the full
   prefill of the second (~1500 tokens at ~1 ms/token decode rate ≈ 1.5 s
   on GPU, several seconds on CPU). **Visible token-stream gap ≈ prefill
   duration.**
4. With Phase 2, the first request stalls for AT MOST one chunk's prefill
   per tick. With `prefill_chunk_tokens=128` on GPU at TinyLlama, that's
   roughly tens of milliseconds — barely perceptible in the stream.

The bench tool doesn't drive this scenario directly. Easiest manual test:

```bash
# Terminal 1: kick off the long generate, watch the stream rate.
curl -N -H "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
  -X POST http://localhost:8800/v1/chat/completions \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Write a 200-word story about a cat."}],"max_tokens":200,"stream":true}' \
  | grep -o '"content":"[^"]*"' | ts '%H:%M:%.S'

# Terminal 2 (~2 s later): fire a long-prompt admit.
LONG=$(yes "The quick brown fox jumps over the lazy dog. " | head -200 | tr -d '\n')
curl -sH "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
  -X POST http://localhost:8800/v1/chat/completions \
  -d "{\"model\":\"tinyllama\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":50}" >/dev/null
```

Look at Terminal 1's timestamp gap right after Terminal 2 fires. With
Phase 2 you should see a steady stream; without, a multi-second pause.

## What to record

When you run, capture into a follow-up commit on this file:

- Hardware (GPU model, VRAM, OS, Rust version, build features).
- TinyLlama avg sequential tok/s (concurrency=1).
- Aggregate tok/s + scaling efficiency at each concurrency tested.
- Stream-gap measurement for the Phase 2 mixed-load scenario (with vs
  without `continuous_batching`).
- DIAG log excerpts (`grep "DIAG: BatchGenerate"` in the daemon log).

## Diagnostic logging

Worker logs at `debug` level:

```
DIAG: BatchGenerate slot registered  (request_id, prompt_tokens, prefix_matched, remaining_to_prefill, slots_active)
DIAG: BatchGenerate prefill chunk ran  (request_id, chunk_tokens, index_pos, remaining_after)
DIAG: BatchGenerate prefill chunk forward failed — slot errored
DIAG: BatchGenerate prefill chunk tensor build failed — slot errored
DIAG: BatchGenerate first-token sample failed — slot errored
DIAG: BatchGenerate decode token_tensor failed — slot errored
DIAG: BatchGenerate decode sample failed — slot errored
```

Run with `-vv` (RUST_LOG=debug-equivalent) to see them. Use to confirm
both phases are firing and that error containment is engaging when a
slot misbehaves.

## Limits

- Single-process worker means concurrent slots share GPU memory pressure.
  Phase A prefill of a large prompt while Phase B decodes 8 slots can
  spike VRAM. If you OOM, drop `batch_generate_max_slots` (defaults to
  `max_concurrent_decode_batch = 8`).
- All slots in a single SlotTable must share `(layer_start, layer_end)`.
  In practice this means: same model + same layer assignment. Multiple
  models = separate worker subprocesses already, not a constraint.
- SWIFT-active requests (`temperature == 0.0` + `swift_self_speculative`)
  fall through to sequential `handle_generate`. So do `max_tokens=0`,
  vision/LoRA, and any request where the table is at capacity.
