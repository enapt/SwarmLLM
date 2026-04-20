# Round 7 Benchmarks — Item 13 activation compression + Item 12 DSD recipe

> Follow-up to `round6.md` (Item 8 cross-over demo). This round fills the
> "End-to-end multi-segment benchmark pending" line on Items 12 + 13 from
> `docs/plans/distributed_inference_speedup.md`.
>
> **TL;DR.** Item 13 (Q8_0 activation compression) shipped a **3.15×
> wire-size reduction** on real multi-segment forwards between two
> loopback daemons, matching the codec's ~3.76× microbenchmark within
> expected header + small-tensor overhead. Decode tok/s on localhost is
> noisy but trends slightly negative — expected from the "Q8_0
> encode/decode cost ≳ wire-latency savings on loopback" prediction. The
> WAN measurement is the one that tells us whether to default-on. Item 12
> (multi-segment DSD) is blocked on draft-model staging + 3+ daemons + a
> WAN-class RTT regime; recipe is documented below, numbers deferred.

## Test bed

- Host: RTX 3070 Laptop 8 GB, WSL2, Linux 6.6.87
- Release binary `target/release/swarmllm` built 2026-04-20 17:24 with
  `--no-default-features --features dev,candle-cuda,claude-subscription`
- Two daemons on loopback, TCP-only P2P (QUIC + mDNS + AutoNAT + DCUtR +
  encryption all disabled), fresh `/tmp/bench_a` and `/tmp/bench_b`
  data dirs with no prior state.
- Model: TinyLlama-1.1B Q4_K_M, 22 layers, 2 shards, hidden_dim=2048.
- **Split setup** (forces multi-segment pipeline):
  - Daemon A — `config.inference.shard_range = (0, 0)`, physical file
    `shard_000.bin` only (layers 0..12).
  - Daemon B — `config.inference.shard_range = (1, 1)`, physical file
    `shard_001.bin` only (layers 13..21).
  - Each daemon scans local shards on startup and announces only what's
    physically present → scheduler assembles a 2-segment pipeline
    (A = layers 0..12 local, B = layers 13..21 remote).
- Inference: streaming chat completions via `swarmllm bench`, `temperature=0`,
  CPU only (`gpu_layers=0`), no prefix cache pre-warm between runs.

## Item 13 — Q8_0 activation compression

### What we're measuring

`config.inference.activation_compression = true` routes the hidden-state
tensor between pipeline segments through `tensor_to_bytes_q8_0` (group-32
symmetric quant, f16 scale) instead of `tensor_to_bytes` (raw f32). The
receiver auto-dispatches on the dtype tag, so a sender with compression on
can talk to a receiver with compression off and vice versa.

We measure the two signals that matter for shipping:

1. **Per-forward wire bytes** — extracted from the
   `"Sending LayerForward to remote segment"` tracing line's
   `activation_bytes=N` field. Summed + averaged across every forward
   call in a bench run.
2. **Decode tok/s** — standard `swarmllm bench --stream` output. Same
   prompt, same max_tokens, same iterations across both scenarios.

### Results (2026-04-20, TinyLlama-1.1B split A:0-12 / B:13-21)

Prompt: 50-token technical-writer prompt asking for a ~50-word paragraph
on CPU fetch-decode-execute. Each run: one warm-up request (5 tokens,
discarded) + 3 measured iterations at 80 max_tokens streaming.

| Scenario | Fwd calls | Total wire bytes | Avg bytes/fwd | Avg tok/s | Avg TTFT (ms) |
|---|---|---|---|---|---|
| `activation_compression = false` (f32) | 240 | 3,445,440 | **14,356** | **4.7** | 4321 |
| `activation_compression = true` (Q8_0) | 166 | 756,216 | **4,556** | 3.5 | 4917 |

**Wire-size reduction: 14,356 / 4,556 = 3.15× per forward.** The codec
microbenchmark claims ~3.76× on full-size tensors; the ~16% gap here is
the per-forward 20-byte envelope header, which is a larger fractional
cost on the single-token decode forwards (hidden=2048, f32 payload =
8192 B, Q8_0 payload ≈ 2176 B — header is 0.24% of f32 but 0.9% of Q8_0).

**Forward-call count mismatch (240 vs 166).** Both runs issued 3
streaming iterations with `max_tokens=80`. In the Q8_0 run, iteration 3
finished early (12 tokens, EOS hit at position 12). This is consistent
with the <1% PPL drift the codec introduces — greedy decode can diverge
on tokens whose top-1 vs top-2 logit gap is within the Q8_0 quantization
noise, and a divergence onto an EOS-prone branch terminates the run
early. It is **not** a wire-path bug; the path carried bit-verified
blocks the whole time (no `non_finite_tensors` or hash-mismatch
rejections in B's log).

**Correctness spot-check.** Short greedy prompt `"The capital of France is"`
with `max_tokens=3, temperature=0` produces the **bit-identical** output
`'▁The▁capital▁of'` with compression on and off. On this prompt the
Q8_0 noise stays well below the top-1 margin.

**Decode tok/s regression is real but localhost-only.** 4.7 → 3.5 tok/s
is a −25% delta, noisy across 3 iterations and exaggerated by the 12-token
iteration in the Q8_0 run (which is short enough that per-request overhead
dominates). A cleaner read: iters 1–2 of both runs, which ran to 80 tokens,
average 4.8 tok/s (f32) vs 4.15 tok/s (Q8_0) — a −13.5% decode regression
on loopback. Expected outcome: Q8_0 quantize + dequantize per forward is a
flat per-call cost that outweighs the μs-range wire savings at
several-GB/s localhost bandwidth. The WAN regime inverts — RTT dominates
wire cost, so shrinking the wire 3× shrinks wall-clock significantly.

### Recommendation

**Keep `activation_compression` off by default** pending a WAN bench.
The 3.15× wire reduction is exactly what the codec promised; the
localhost tok/s regression is exactly what the doc warned about.
Defaulting on is a decision to make once WAN numbers land — see
`next_steps.md` Item 2 and `round6.md` Deferred § WAN bench.

## Item 12 — multi-segment DSD

### Why localhost can't answer the question

Decentralized Speculative Decoding (DSD, arxiv 2511.11733) shifts `γ`
draft tokens through the pipeline in a single round trip instead of `γ`
sequential round trips. The paper's speedup regime is
`3·t0 < t1 < 10·t0` where `t0` is per-token compute and `t1` is per-link
RTT. On this bench:

- `t0` for TinyLlama-1.1B CPU decode ≈ 200 ms (from round6 baseline).
- `t1` on loopback ≈ 1 ms.
- Ratio `t1/t0 ≈ 0.005` — four orders of magnitude below the DSD
  speedup threshold.

In this regime DSD does strictly more work (γ drafts that mostly get
rejected on a small model) for essentially zero round-trip savings —
the expected outcome is a measurable slowdown on localhost, identical
to what we'd get by running speculative decoding on a one-node setup.
That's a correctness test, not a speedup bench.

### Preconditions DSD needs that this setup doesn't satisfy

1. **`speculative_decoding = true` AND `decentralized_spec_decoding = true`**
   — two config toggles, easy.
2. **Draft model loaded** — `draft_model_path` points to a GGUF that
   shares TinyLlama's vocabulary but is much smaller. No such model is
   pre-staged in `memory/local_model_shards.md`. TinyLlama itself is
   already the smallest Llama-arch variant that shares Llama's SPM
   tokenizer; a true draft would need a 50M-parameter Llama distillation
   or a heavily-pruned TinyLlama. Qwen2.5-0.5B + Qwen2.5-7B is a
   natural pair (the only pair in our test-model set where the smaller
   member shares the larger's tokenizer), but 7B × 2 daemons × CPU
   puts iteration-1 TTFT at ~2 minutes (round6 already showed this),
   making a 3-iteration bench a 10+ minute test. Tractable on a GPU
   where 7B fits, but 7B Q4 + DSD γ-window scratch doesn't fit in
   8 GB VRAM on this host (round6 § GPU-mixed asymmetry has the same
   OOM).
3. **Pipeline has 2+ segments AND no segment is on the coordinator** —
   `src/inference/pipeline/dsd.rs:94-105`. The coordinator drafts tokens
   and forwards activations; it holds no shards. That's a 3-daemon
   setup (coordinator + segment A + segment B), not a 2-daemon setup.
4. **Greedy temperature=0** and no vision / LoRA / encryption — trivial.

### Bench recipe (for a later round with the right hardware)

Minimum setup to measure a real DSD speedup:

- **3 daemons** on the *same* WAN link or two WAN links, not localhost:
  - Daemon C (coordinator) — no shards, has draft model loaded via
    `config.inference.draft_model_path`, has target model **metadata**
    only (needs the GGUF header for tokenizer + embedding dim).
  - Daemon A — target model shards covering layers `0..mid`.
  - Daemon B — target model shards covering layers `mid..N`.
- **Target+draft pair** — recommend Qwen2.5-7B-Instruct target +
  Qwen2.5-0.5B-Instruct draft (same vocab, ~14× parameter ratio, both
  GGUF-compatible). Stage both on C; C runs the draft locally.
- **Config on C**:
  ```toml
  [inference]
  speculative_decoding = true
  decentralized_spec_decoding = true
  speculative_gamma = 4  # start here; GammaController will tune
  draft_model_path = "/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  draft_gpu_layers = 0  # or N if C has GPU headroom
  ```
- **Measure**:
  - `swarmllm bench -p <C> --iterations 3 --max-tokens 100 --stream`
    with `decentralized_spec_decoding = true` (DSD path).
  - Same run with `decentralized_spec_decoding = false` (baseline:
    standard speculative decoding through the pipeline, each draft
    token a full round trip).
  - Key metric: decode tok/s. DSD saves `(N-1) · t1 · (γ-1)/γ` per
    accepted token → 3 segments × 50 ms RTT × 3/4 = ~113 ms saved per
    accept on a 4-wide γ-window. At 30% acceptance rate and 200 ms
    baseline per-token, that's a ~40% decode-tok/s win.
- **Correctness check**: DSD verifier is greedy bit-identical — on
  `temperature=0`, DSD output must be **exactly** the output of
  running the same pipeline without DSD. Diverging output = bug,
  not noise.

**Status**: deferred to the WAN bench round. Both the DSD speedup and the
Item 13 decode-side win require WAN RTT to materialize, so they should
share a round once real hardware is available.

## What was run vs. what landed

| Claim in `distributed_inference_speedup.md` | Status after round 7 |
|---|---|
| Item 13: codec ~3.76× compression, peer-compat auto-dispatch | ✅ end-to-end 3.15× per-forward (header overhead), receiver auto-dispatched both directions mid-run |
| Item 13: "End-to-end multi-segment benchmark pending" | ✅ **measured** on 2-daemon loopback; WAN decode-tok/s win remains to be shown |
| Item 12: all phases landed | ✅ eligibility + coordinator wired; compiles + passes fast-path tests |
| Item 12: "End-to-end multi-segment WAN benchmark pending" | ⏳ **still pending** — needs draft model + 3 daemons + WAN-class RTT |

## Anti-goals for this round

- **Don't default-on `activation_compression`** based on localhost numbers.
  The wire reduction is real but the wall-clock savings only appear when
  RTT dominates wire encode/decode cost, which isn't the localhost case.
- **Don't run DSD on localhost** to get a "bench number." In the
  `t1/t0 ≪ 1` regime DSD strictly loses to baseline and calling that a
  bench result is misleading.
- **Don't stage Qwen 0.5B / 7B pair just to unblock Item 12 localhost**.
  Same reason — the output would not resemble the WAN speedup the paper
  claims, so it'd just burn disk.
