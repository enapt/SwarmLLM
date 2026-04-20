# Benchmarking

SwarmLLM ships with a built-in bench command and a set of reproducible
recipes under `docs/plans/benchmarks/`. This chapter covers both.

## Quick: `swarmllm bench`

The bench subcommand runs a real `/v1/chat/completions` workload against
a daemon and reports latency + throughput.

```bash
./swarmllm bench \
    --max-tokens 100 \
    --iterations 5 \
    --concurrency 1 \
    --stream \
    --model-id tinyllama-1.1b-chat-v1.0.q4-k-m \
    --json
```

Key flags:

- `--max-tokens` — tokens to generate per request (default 100)
- `--iterations` — sequential iterations per concurrency level (default 5)
- `--concurrency` — concurrent requests for throughput tests (default 1)
- `--stream` — use streaming chat completions and report **TTFT**
  (time-to-first-token) per request. TTFT is the signal that captures
  Item 7 and Item 8 wins; non-streaming bench rolls prefill + decode into
  one total time and hides the difference.
- `--prompt` — custom prompt; default is a short prompt about
  relativity that won't stress prefix caching. Pass a longer prompt
  (≥500 tokens) to exercise prefix cache paths.
- `--model-id` — target a specific model when several are registered;
  otherwise uses the first one from `/v1/models`.
- `--json` — machine-readable output

The bench reads the API key from the daemon's data dir, so run it with
the same `SWARMLLM_NODE_DATA_DIR` or `-d` as the daemon.

## Single-node baselines

Reference numbers on an AMD Ryzen 7 5800H + RTX 3070 Laptop 8 GB VRAM
(WSL2, release build):

| Model | Params | Quant | GPU | CPU |
|---|---|---|---|---|
| TinyLlama 1.1B Chat | 1.1B | Q4_K_M | 27.2 tok/s | 4.2 tok/s |
| Gemma-2 2B IT | 2.5B | Q4_K_M | 20.6 tok/s | 3.5 tok/s |
| Phi-3.5 Mini | 3.8B | Q4_K_M | 46.4 tok/s | 1.8 tok/s |
| Qwen2.5-Coder 7B | 7.6B | Q4_K_M | 29.0 tok/s | 2.4 tok/s |

Single-node numbers are largely about your hardware. The interesting
benchmarks are distributed.

## Reproducing the speedup-arc benchmarks

Each entry in the speedup arc has a written recipe. Most require two
local daemons on loopback; a couple need three.

### Item 7: BatchGenerate TTFT fairness

[`docs/plans/benchmarks/round4.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/benchmarks/round4.md)

Measures TTFT at concurrency 2/4/8 with Phases 1+2 on vs off. The win is
fairness, not aggregate throughput: Sarathi chunked prefill prevents new
admits from waiting behind the full prior prefill.

### Item 7 Phase 4: Batched chunked prefill

[`docs/plans/benchmarks/round5.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/benchmarks/round5.md)

Measures aggregate tok/s and per-request TTFT spread with
`batched_prefill_forward` on vs off. The on-config fuses concurrent
same-shape prefill chunks so TTFT lands tightly clustered instead of
spreading.

### Item 8: Cross-node prefix KV sharing

[`docs/plans/benchmarks/round6.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/benchmarks/round6.md)

Two-daemon loopback TCP. Measures iter-1 TTFT with the cross-node fetch
path enabled vs gated off (via `cross_node_prefix_trust_min = 2.0`).
Same recipe runs against TinyLlama (fast-GPU corner case: fetch is
slightly slower than prefill) and Qwen-7B (**12.9× TTFT speedup** on
CPU-CPU because 7B CPU prefill is slow enough that the ~1 s fetch +
verify + hydrate buys back ~150 s of local prefill).

Sketch of the recipe:

```bash
# Node A on 8800
SWARMLLM_NODE_DATA_DIR=/tmp/swarm_a ./target/release/swarmllm run -p 8800 -v &

# Node B on 8900, bootstrapped off A
A_MADDR=$(grep -oE "peer_id=12D3KooW[A-Za-z0-9]+" /tmp/swarm_a.log | \
    head -1 | sed 's/peer_id=/\/ip4\/127.0.0.1\/tcp\/8810\/p2p\//')
SWARMLLM_NODE_DATA_DIR=/tmp/swarm_b ./target/release/swarmllm run \
    -p 8900 -v --bootstrap "$A_MADDR" &

# Copy shards into both data dirs (or download via /api/admin/hf/download-shards)
cp -r ~/.local/share/swarmllm/models/<model-id> /tmp/swarm_a/models/
cp -r ~/.local/share/swarmllm/models/<model-id> /tmp/swarm_b/models/

# Warm A with the long prompt (populates A's prefix cache, announces to B)
./swarmllm bench -p 8800 --stream --iterations 3 --max-tokens 100 \
    --prompt "$(cat long-prompt.txt)" --model-id <model-id>

# Measure B TTFT — iter 1 should fire the cross-node fetch
./swarmllm bench -p 8900 --stream --iterations 3 --max-tokens 100 \
    --prompt "$(cat long-prompt.txt)" --model-id <model-id> --json
```

Check B's log for `DIAG: cross-node prefix HIT — hydrated KV matched_tokens=... bytes=...`
to confirm the fetch path fired.

## Caveats

- **WSL2 localhost bandwidth** is much higher than any real network —
  localhost benches are the best case for compute-bound paths and the
  worst case for fetch paths. WAN numbers will be different.
- **TinyLlama is too small to show some speedups** — Item 8 in particular
  needs a larger model (Phi-3.5, Qwen-7B) to flip the sign between
  fetch-cost and prefill-cost. See the Round 6 doc for the cross-over
  math.
- **VRAM fit matters** — Qwen-7B Q4 weights fit in 8 GB but batched
  attention kernel scratch does not. CPU-mode works but the baseline
  numbers above change.
- **Pre-warm before measuring TTFT** — iter 1 of a model includes disk
  read + weight load + first CUDA context init; exclude this by
  pre-warming with a short unrelated prompt before the real measurement.

Standard pre-push gate is `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.
If you add a benchmark, add it under `docs/plans/benchmarks/roundN.md`
with the recipe + results + interpretation, and link it from here.
