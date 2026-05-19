# Performance & Inference Speedups

SwarmLLM's distributed inference path ships with a stack of optimisations
that are **on by default** — you get them without touching a config. This
chapter names each one, explains what it does, and shows the measured win
so you can tell which levers matter for your workload.

A few are **flag-gated** because the win is workload-dependent or the path
is still being hardened; those are documented at the bottom so you can turn
them on intentionally.

The full design notes live in
[`docs/plans/archive/distributed_inference_speedup.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/archive/distributed_inference_speedup.md)
with benchmark recipes in
[`docs/plans/benchmarks/`](https://github.com/enapt/SwarmLLM/tree/main/docs/plans/benchmarks).

## The default-on stack

### Continuous batching

Concurrent `/v1/chat/completions` requests for the same model share one
forward pass per decode tick instead of running serially. GPU builds use a
fused `forward_batch` kernel; CPU workers fall through to sequential with
no regression.

- **Measured:** 1.34–1.55× GPU throughput at batch 2–8 on RTX 3070 +
  TinyLlama Q4
- **Config:** `inference.continuous_batching = true` (default)

### Remote-generate fast path

For single-segment distributed inference (the common case: one remote node
owns the whole model, requester does embedding + sampling), skip the
per-token coordinator round-trips and run the decode loop end-to-end on
the remote worker. Tokens stream back as they're sampled.

- **Measured:** 1.93× decode speedup
- **Config:** default-on — no flag, triggered automatically on
  single-segment pipelines

### Cross-request prefix cache

Each worker keeps an LRU cache of prefill KV snapshots keyed by the
prompt's token prefix. A re-submission with the same system prompt
(different user turn) skips prefill for the shared prefix and only
forwards the suffix.

- **Measured:** 29.4× wall-clock speedup on re-submission of the same
  513-token prompt (single node, TinyLlama)
- **Config:** `inference.prefix_cache_enabled = true` (default),
  `inference.prefix_cache_block_tokens = 64` (default — block granularity),
  `inference.prefix_cache_max_entries = 16` (default — per model)

### Batched prefill + chunked prefill

Sarathi-style chunked prefill: a long admission advances by
`prefill_chunk_tokens` (default 128) per decode tick, so new requests
don't wait behind a full prior prefill. Phase 4 adds
`batched_prefill_forward = true` (default), which fuses concurrent
same-shape prefill chunks into one `forward_batch` call.

- **Measured (Phases 1+2):** 17–23× TTFT fairness at concurrency 2/4/8 on
  RTX 3070 + TinyLlama Q4 vs serial prefill
- **Measured (Phase 4):** 1.57× aggregate tok/s at c=4 with uniform
  180/180/180 ms TTFT (vs pre-fix 52/235/447 ms spread)
- **Config:** `inference.continuous_batching = true`,
  `inference.prefill_chunk_tokens = 128`,
  `inference.batched_prefill_forward = true` (all default)

### Cross-node prefix-KV sharing

When node B receives a prompt whose prefix was already prefilled by peer A,
B fetches A's KV snapshot over the wire instead of re-prefilling locally.
The pipeline is:

```
A prefills → inserts prefix-cache block → gossips PrefixCacheAnnounce
B receives prompt → local cache miss → probe daemon → walk index
B sends SendPrefixKvFetch to A → A's worker exports snapshot
B verifies BLAKE3 + NaN/Inf → hydrates KV → prefill suffix only
```

- **Measured (TinyLlama, GPU-GPU):** fetched path is ~100 ms *slower* than
  local prefill — the 28 MB f32 snapshot takes ~260 ms to ship while the
  local prefill it replaces is only ~460 ms. TinyLlama is too small to
  demonstrate the win on localhost + fast GPU.
- **Measured (Qwen2.5-Coder-7B, CPU-CPU):** **12.9× TTFT speedup** on
  iter 1 — control full-prefill = 151.7 s, fetched path = 11.8 s. The
  73 MB f32 snapshot transfers in ~1 s while 640-token Qwen-7B CPU
  prefill runs ~150 s.
- **Config:** `inference.cross_node_prefix_trust_min = 0.5` (default —
  gates peers by trust score; set to `2.0` to disable the fetch path
  entirely).

The fetch path uses three chained timeouts (worker probe 3000 ms, daemon
network 2500 ms, serving IPC 2000 ms) sized for 7B-class f32 snapshots.
Missing the window degrades to a clean miss — no worse than not having
the feature. See the
[two-daemon loopback bench recipe](https://github.com/enapt/SwarmLLM/blob/main/docs/plans/benchmarks/round6.md)
for reproduction details.

### Parallax scheduler

Pipeline assignment uses shortest-path dynamic programming over observed
per-layer latencies (EMA over recent forwards) rather than a greedy
pick-the-closest-peer heuristic. Cross-gossip of top-32 observed
latencies via `NodeCapability.observed_latencies` lets every node keep
a current view of the network's compute profile. A soft acquire/prune
bias in `AutoShardManager` driven by a per-shard stability counter
(≥3 consistent ticks before it acts) drifts shards toward where they're
actually used without violating existing hard constraints.

- **Measured:** 10 routing + 7 allocator + 2 scheduler integration tests
  passing; real-world improvements depend on network heterogeneity. The
  biggest impact is in asymmetric setups where a cheap peer's low
  observed latency should beat a high-VRAM peer's big shard slot.
- **Config:** default-on. Multi-pipeline concurrency is deferred.

## Flag-gated features

Turn these on when you've measured that they match your workload.

### Distributed speculative decoding (`speculative_distributed`)

Draft model proposes γ tokens locally; target verifies all γ in one
remote forward pass.

- **Status:** End-to-end verified. 40–52% accept rate in a
  llama-cpp-draft / candle-target pairing (cross-backend numerical
  mismatch caps accept rate).
- **Config:** `inference.speculative_distributed = true`,
  `inference.draft_model_path = "path/to/draft.gguf"`,
  `inference.speculative_gamma = 4` (tokens per verify round)

### SWIFT self-speculative decoding (`swift_self_speculative`)

The target model acts as its own draft by skipping a contiguous range of
layers on the proposal pass. No external draft model needed.

- **Status:** Landed behind flag. Structurally slower than baseline on
  candle CPU until flash-attn-with-mask lands (attention kernel mismatch
  on multi-position verify). Shelved on CPU; may help on GPU.
- **Config:** `inference.swift_self_speculative = true`,
  `inference.swift_skip_ratio = 0.45` (fraction of layers to skip on the
  draft pass)

### DSD — decentralized speculative decoding (`decentralized_spec_decoding`)

Multi-segment distributed inference with speculative decoding woven in.
A γ-token decode on the last-segment worker plus KV truncation primitives
plus a coordinator loop in `pipeline/dsd.rs`.

- **Status:** All phases landed 2026-04-18 behind flag. End-to-end
  multi-segment WAN benchmark pending.
- **Config:** `inference.decentralized_spec_decoding = true`

### Activation compression Q8_0 (`activation_compression`)

Intermediate pipeline hidden-state activations are quantized from f16 to
Q8_0 before going over the wire. Receivers auto-dispatch on the dtype
tag.

- **Status:** Codec verified. ~3.76× wire compression, RMS error <0.005.
  End-to-end multi-segment benchmark pending.
- **Config:** `inference.activation_compression = true`

### Persistent pipeline stream (`persistent_pipeline_stream`)

Replace per-token request/response with one long-lived libp2p bidirectional
stream per pipeline session.

- **Status:** Landed behind flag. Wire-level verified; no measured latency
  win because the bottleneck was elsewhere (solved by remote-generate +
  batched prefill).
- **Config:** `inference.persistent_pipeline_stream = true`

### Daemon-side STREAM-chunked activation send (R139, `streaming_chunked_send`)

Split large activation tensors into K = ceil(size / chunk_size) chunks at
the wire boundary and ship them sequentially on a single libp2p stream.
Each chunk is sealed independently with ChaCha20-Poly1305; `chunk_idx` +
`total_chunks` are bound into the AAD so reorder / wrong-total /
cross-transfer-substitution fail Poly1305 before reaching dispatch.
Receiver reassembles via `SharedState.pending_activation_chunks` before
dispatching a single LayerForward to the worker.

- **Status:** Wire-format, AAD binding, receiver assembly, and sender
  wiring (persistent-stream path only) shipped in R139. Default
  off — needs WAN-bench evidence before the default flips. On
  LAN/loopback the activation send is already sub-millisecond and
  chunking adds per-chunk fixed cost with near-zero overlap win.
- **Requires:** `persistent_pipeline_stream = true` (RR fallback path
  not yet wired — needs per-chunk Acks).
- **Config:**
  - `inference.streaming_chunked_send = false` — master switch
  - `inference.streaming_chunk_size_bytes = 262144` — 256 KiB default
    (age STREAM construction + TokenWeave MLSys 2026 K=2-4 sweet spot)
  - `inference.streaming_min_activation_bytes = 65536` — 64 KiB floor;
    activations below this ship as one frame regardless of the flag
  - `inference.streaming_chunk_assembly_ttl_secs = 30` — receiver
    eviction TTL for stuck assemblies

### Encrypt/decrypt offload from event loop (R139 Phase C)

Unconditional benefit (no flag). ChaCha20-Poly1305 sealing in
`handle_send_tensor` and the open in `handle_tensor_payload`
(TENSOR_TAG_ENCRYPTED arm) are offloaded to `tokio::spawn` tasks so
the NetworkManager event loop stays responsive under concurrent
decode load. Saves ~50–200µs/forward of event-loop block time on the
default RR encrypted path; multiplied across concurrent traffic this
is the difference between smooth libp2p ping / gossip / connection
handling and observable jitter.

## Debugging slow inference

Default verbosity (`-v`) gives an `INFO`-level stream. Bump to `-vv` to
see per-request `DIAG:` logs, which include the per-feature speedup
signals:

```bash
./swarmllm run -vv 2>&1 | grep "DIAG:"
```

Key DIAG kinds:

- `DIAG: prefix-cache HIT` — local prefix cache hit
- `DIAG: cross-node prefix HIT` — cross-node prefix-KV fetch succeeded
- `DIAG: prefix-probe: fetch timed out` — cross-node fetch missed the
  window (see [Troubleshooting](../troubleshooting.md) for timeout
  sizing on 7B+ models)
- `DIAG: served PrefixKvFetch ... hit=true` — this node served a
  cross-node fetch
- `DIAG: BatchGenerate` — batched-prefill slot table activity
- `DIAG: chunk fused batch_size=N` — fused prefill chunks (Phase 4)
- `DIAG: Parallax` — Parallax scheduler decisions

For the full DIAG taxonomy and what each line means, see
[`docs/DIAGNOSTICS.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/DIAGNOSTICS.md).

## When should I turn a speedup off?

Almost never. The default-on features degrade cleanly under edge cases —
the prefix cache falls through to full prefill on a miss, cross-node
fetch falls through to local prefill on a timeout, batched prefill
falls back to sequential when concurrency is 1. If you suspect one is
the cause of a regression:

- **Prefix cache off:** `inference.prefix_cache_enabled = false`
- **Cross-node fetch off:** `inference.cross_node_prefix_trust_min = 2.0`
  (gates every peer out)
- **Continuous batching off:** `inference.continuous_batching = false`
  (also disables Phase 4 fusion)
- **Phase 4 fusion off, keep continuous batching:**
  `inference.batched_prefill_forward = false`

Please [open an issue](https://github.com/enapt/SwarmLLM/issues) if a
speedup is costing you — the benchmarks above are RTX 3070 + WSL2 + a
specific set of models, so real-world workloads will surface corners the
benches miss.
