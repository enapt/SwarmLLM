# Distributed Inference Speedup Plan — Landed Summary

> This plan was a multi-session effort to speed up distributed
> inference. Items 1–16 all landed. The per-item design notes and
> session logs that used to live here have been trimmed out — the
> canonical references now are:
>
> - **[CHANGELOG.md](../../CHANGELOG.md)** → `[Unreleased] - alpha`
>   → Distributed Inference Speedup Arc — user-facing summary with
>   measured wins
> - **[docs/book/src/operations/performance.md](../book/src/operations/performance.md)** —
>   per-item what-it-does + config knobs + DIAG log names
> - **[docs/book/src/operations/benchmarking.md](../book/src/operations/benchmarking.md)** —
>   bench recipes + caveats
> - **[docs/plans/benchmarks/round{3,4,5,6}.md](./benchmarks/)** —
>   per-round bench recipes + measured results
> - **[git log --oneline -- docs/plans/distributed_inference_speedup.md](https://github.com/enapt/SwarmLLM/commits/main/docs/plans/distributed_inference_speedup.md)** —
>   full per-item design notes in history

## Default-on stack

| Item | Status | Effect |
|---|---|---|
| **Item 4 — Remote-generate fast path** | ✅ LANDED & DEFAULT-ON | **1.93× decode speedup** on default config |
| **Item 5 — Cross-request prefix cache** | ✅ LANDED & DEFAULT-ON | **29.4× wall-clock** on cache-hit re-submission of the same 513-token prompt |
| **Item 3 — Continuous batching** (2026-04-19) | ✅ LANDED & DEFAULT-ON | 1.34–1.55× GPU throughput at batch 2–8; CPU falls through to sequential, no regression |
| **Item 7 Phases 1+2 — BatchGenerate + Sarathi chunked prefill** | ✅ LANDED & DEFAULT-ON (2026-04-19) | **18.2× TTFT @ c=2, 21.7× @ c=4, 23.5× @ c=8** vs off (RTX 3070 + TinyLlama Q4). See `benchmarks/round4.md` |
| **Item 7 Phase 4 — batched chunked prefill + admit-coalescing** | ✅ LANDED & DEFAULT-ON (2026-04-19) | **1.57× aggregate tok/s @ c=4** + uniform-ms TTFT fairness (180/180/180 ms vs pre-fix 52/235/447 ms). Isolation flag `batched_prefill_forward` (default `true`) toggles Phase 4 in isolation from Phases 1+2. See `benchmarks/round5.md` |
| **Item 8 — Cross-node prefix KV sharing** | ✅ ALL PHASES LANDED (2026-04-19/20) | **12.9× iter-1 TTFT speedup** on Qwen-7B CPU-CPU localhost (151.7 s → 11.8 s, 672-token prompt). TinyLlama on GPU is the fast-prefill corner case (~100 ms slower due to wire vs prefill cost). See `benchmarks/round6.md` |
| **Item 16 — Parallax scheduler (A+B+B.2+C+C.2)** | ✅ LANDED (2026-04-18/19) | All phases default-on except Phase D (multi-pipeline concurrency, deferred). Phase A shortest-path DP; Phase B per-layer latency EMA; Phase B.2 cross-node gossip via `NodeCapability.observed_latencies`; Phase C offline allocator `Z(k) = k²/s*(k)`; Phase C.2 soft acquire/prune bias with ≥3-tick stability gate |

## Flag-gated

| Item | Status |
|---|---|
| Item 1 — Persistent pipeline stream | ✅ Landed behind `persistent_pipeline_stream=false`. Wire-verified; no measured latency win because Items 4+7 solved the actual bottleneck |
| Item 2 — Distributed speculative decoding | ✅ Landed in 3 phases behind `speculative_distributed=false`. 40–52% accept rate in llama-cpp-draft / candle-target pairing |
| Item 6 — SWIFT self-speculative | 🟡 Landed behind `swift_self_speculative=false`. Structurally slower than baseline on candle CPU until flash-attn-with-mask lands. Shelved |
| Item 12 — DSD multi-segment spec decoding (2026-04-18) | ✅ All phases landed behind `decentralized_spec_decoding=false`. End-to-end multi-segment WAN benchmark pending |
| Item 13 — Activation compression (Q8_0) | ✅ Codec verified (~3.76× compression, RMS error <0.005, peer-compatible auto-dispatch). End-to-end 2-daemon bench in `round7.md` (2026-04-20): measured **3.15× wire reduction** per forward, −13.5% decode tok/s on loopback (expected — wall-clock win requires WAN RTT). Still behind `activation_compression=false` pending WAN numbers |

## Round 6 measurements (2026-04-20, RTX 3070 Laptop + WSL2 loopback)

### TinyLlama-1.1B Q4_K_M (GPU both sides), 672-token prompt

| Scenario | iter-1 TTFT | iter 2 | iter 3 |
|---|---|---|---|
| A cold (full prefill + model load) | 1548 ms | 249 ms | 246 ms |
| B with cross-node fetch enabled | **809 ms** | 256 ms | 231 ms |
| B control, `cross_node_prefix_trust_min=2.0` | **713 ms** | 253 ms | 253 ms |

TinyLlama is too small to show the fetch-path win on localhost: the
28 MB f32 snapshot takes ~260 ms to pull + hydrate while the local
prefill it saves is only ~460 ms, so the fetched path is ~100 ms
slower than re-prefilling — the "too-small" corner case.

### Qwen2.5-Coder-7B Q4_K_M (CPU both sides), 672-token prompt

Qwen-7B Q4 weights (4.7 GB) + CUDA scratch OOM on the 8 GB card;
both daemons run with `CUDA_VISIBLE_DEVICES=""` for apples-to-apples
comparison.

| Scenario | iter-1 TTFT | iter 2 | iter 3 |
|---|---|---|---|
| B with cross-node fetch enabled | **11.8 s** | 9.5 s | 9.2 s |
| B control, `cross_node_prefix_trust_min=2.0` | **151.7 s** | 9.6 s | 9.5 s |

This is the cross-over point: 640-token CPU prefill of Qwen-7B runs
~150 s, while the 73 MB f32 snapshot transfers in ~1 s over loopback.
Iter 2/3 are equal across scenarios because B's local prefix cache
populates after iter 1.

Full pipeline validated on both: announce → index → probe → trust-gate
→ fetch → BLAKE3 verify → NaN/Inf scan → hydrate → suffix prefill.

### Three wire bugs uncovered + fixed in-tree

1. `SwarmMessage::PrefixCacheAnnounce` missing from
   `NetworkManager::handle_broadcast`'s `TOPIC_MODELS` arm → Phase 1
   announces silently dropped at the gossip layer.
2. `WorkerMsg::PrefixSnapshotResponse` + `DaemonMsg::PrefixFetchResult`
   carried `payload: Option<Vec<u8>>` inside JSON-framed IPC headers.
   `serde_json` encodes `Vec<u8>` as an integer array (~5× bloat), so
   28 MB of binary became a ~102 MB header and blew past the 64 MiB
   `MAX_HEADER` cap, killing the worker. Fix: move bytes to the IPC
   binary-payload slot; keep a `present: bool` tag in the header.
3. Three chained cross-node-fetch timeouts (500 ms worker probe,
   400 ms daemon network, 500 ms serving IPC) were TinyLlama-sized
   (28 MB snapshot). Qwen-7B's 73 MB snapshot needs ~500–1000 ms to
   serialize+wire, tripping every timeout. Fix: bumped to 3000 / 2500
   / 2000 ms respectively, keeping the worker timeout as the outer
   bound.

## Deferred

- **GPU-mixed asymmetry** — A on GPU, B on CPU. Blocked on fitting
  Qwen-7B prefill scratch in 8 GB VRAM on this host. Either need a
  larger card or a smaller target model (Phi-3.5-mini).
- **WAN bench** — two daemons on different machines / regions. The
  Qwen-7B loopback cross-over is established; WAN sharpens the
  RTT-vs-prefill trade.
- **Zstd compression on `WIRE_TAG_PREFIX_KV`** — ✅ landed 2026-04-25 as
  `NetworkConfig::prefix_kv_compression: bool` (default off). Wire format
  reuses the existing flag byte: flag=0 (miss), flag=1 (raw), flag=2
  (zstd). Receivers always decompress regardless of the send-side flag.
  Falls back to raw when the compressed form isn't smaller. Awaiting WAN
  bench to decide default-on. See `next_steps.md` § 1.
- **Items 14 / 17 / 18** research candidates (Mirror Spec Decoding,
  disaggregated prefill/decode, per-token early-exit) — per-item
  assessment, fit-for-SwarmLLM analysis, and sequencing recommendation
  in [`items_14_17_18_research.md`](./items_14_17_18_research.md).
  Headline: 17 > 18 > 14 by signal-to-effort, all gated on WAN bench.
- **End-to-end benches** for Item 13 (activation compression) measured in
  `round7.md`; Item 12 (DSD) + Item 2 (distributed spec decoding with
  matched-backend draft) still deferred — both need 3+ daemons + WAN-class
  RTT to produce a meaningful speedup number (see `round7.md` § Item 12
  for the recipe and why localhost can't answer).
