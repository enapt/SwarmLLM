# Distributed Inference Speedup Plan

> **READ THIS FIRST (status as of 2026-04-19)**
>
> This doc captures a multi-session effort to speed up distributed inference.
> Four items were scoped originally; headline wins came from Items 4 and 5
> (not in the original plan), followed by Items 3 + 16 in later sessions.
>
> **Default-on speedup stack today (as of 2026-04-19):**
>
> - **Item 4** fast path — 1.93× decode for single-segment distributed inference
> - **Item 5** prefix cache — 29.4× wall-clock on cache hit
> - **Item 3** continuous batching — 1.34–1.55× GPU throughput for concurrent requests (CPU falls through to sequential, no regression)
> - **Item 16 A+B** Parallax routing — shortest-path pipeline chain via DP + observed per-peer latency EMA
> - **Item 7 Phases 1 + 2** worker-side SlotTable + Sarathi chunked prefill — measured **17–23× TTFT improvement** at concurrency 2/4/8 on RTX 3070 + TinyLlama Q4 with equivalent aggregate throughput (no regression). See `benchmarks/round4.md`.
> - **Item 7 Phase 4** batched chunked prefill + admit-coalescing — fused `forward_batch` over same-shape Prefilling slots; drain-before-tick unlocks fusion under HTTP-paced concurrent admits. Measured **1.57× aggregate tok/s @ c=4 and uniform-ms TTFT fairness** on RTX 3070 + TinyLlama Q4 vs pre-fix singleton path. See `benchmarks/round5.md`.
>
> Everything else is behind a flag (`speculative_distributed`, `persistent_pipeline_stream`, `decentralized_spec_decoding`, `activation_compression`, `swift_self_speculative`) or advisory (Phase C allocator).
>
> | Item | Status | Effect |
> |---|---|---|
> | **Item 4 — Remote-generate fast path** | ✅ LANDED & DEFAULT-ON | **1.93× decode speedup** on default config. Main user-visible win. |
> | **Item 5 — Cross-request prefix cache** | ✅ LANDED & DEFAULT-ON | **29.4× wall-clock** on cache-hit re-submission of the same 513-token prompt. |
> | Item 1 — Persistent pipeline stream | ✅ Landed behind `persistent_pipeline_stream=false` flag. Verified end-to-end but no measured latency win (bottleneck was elsewhere). |
> | Item 2 — Speculative decoding (distributed) | ✅ Landed in 3 phases behind `speculative_distributed=false` flag + requires loaded draft model. Working; 40–52% accept rate w/ backend mismatch (llama-cpp draft vs candle target). |
> | **Item 3 — Continuous batching** | ✅ LANDED & DEFAULT-ON 2026-04-19 (`continuous_batching=true`). Phase 1 (wire protocol), Phase 2a (response multiplexing), Phase 2b (fused `forward_batch` + CPU fallback + auto-coalescing scheduler in `ModelProcessPool`) all shipped. GPU: 1.34–1.55× throughput at batch 2–8. CPU: worker falls through to sequential, no regression. |
> | Item 6 — SWIFT self-speculative | 🟡 Landed behind `swift_self_speculative=false` flag. Structurally slower than baseline on candle until flash-attn-with-mask lands (kernel mismatch on multi-position verify). Shelved. |
> | **Item 13 — Activation compression (Q8_0)** | ✅ LANDED behind `activation_compression=false` flag. Codec verified (~3.76× compression, RMS error <0.005, peer-compatible auto-dispatch). End-to-end multi-segment benchmark pending. |
> | **Item 12 — DSD (decentralized speculative)** | ✅ ALL PHASES LANDED 2026-04-18 behind `decentralized_spec_decoding=false`. Worker γ-token decode + KV truncation primitives + γ controller + multi-segment spec-verify worker branch + ~410 LOC coordinator loop in `pipeline/dsd.rs`. End-to-end multi-segment WAN benchmark pending. |
> | **Item 16 — Parallax scheduler (Phases A+B+B.2+C+C.2)** | ✅ LANDED 2026-04-18/19. All phases default-on except Phase D (multi-pipeline concurrency, deferred). Phase A: shortest-path DP. Phase B: observed per-layer latency EMA. Phase B.2: cross-node gossip of top-32 observed latencies via `NodeCapability.observed_latencies`. Phase C: `parallax_allocator.rs` offline layer allocator with `Z(k) = k²/s*(k)` objective. Phase C.2 (2026-04-19): soft acquire/prune bias in `AutoShardManager` driven by a per-shard stability counter (≥3 ticks of consistent signal) — respects every existing hard constraint. Tests: 10 routing + 7 allocator + 2 scheduler integration + 1 EMA math + 5 merge + 8 stability. |
> | **Item 7 — BatchGenerate Phases 1 + 2** | ✅ LANDED 2026-04-19, **measured 2026-04-19** (RTX 3070 + TinyLlama-1.1B Q4, 3-iter avg): **18.2× TTFT @ c=2, 21.7× TTFT @ c=4, 23.5× TTFT @ c=8**, with equivalent aggregate throughput vs Phase 1+2 OFF. The win is TTFT fairness — Sarathi chunked prefill prevents new admits from waiting behind the full prior prefill+decode. Aggregate throughput is unchanged because TinyLlama is too small for fused `forward_batch` to add tok/s on this GPU. See `docs/plans/benchmarks/round4.md`. |
> | **Item 7 — Phase 4 batched chunked prefill + admit-coalescing** | ✅ LANDED & MEASURED 2026-04-19. `forward_batch` generalized for homogeneous prefill-chunk groups (same `seq_len > 1` + same `index_pos`). Admit-coalescing drain (extract `handle_daemon_msg`, `try_recv` up to 16 queued messages before each tick) unlocks fusion under HTTP-paced concurrent admits. Measured RTX 3070 + TinyLlama Q4: **49.1 tok/s aggregate @ c=4 (+57% vs pre-fix 31.2)**, TTFT uniform **180 / 180 / 180 ms** across 4 requests (vs pre-fix **52 / 235 / 447 ms** spread), `DIAG chunk fused batch_size=4` confirmed. `InferenceConfig::batched_prefill_forward` (default `true`) toggles fusion in isolation from Phases 1+2. Synthetic + E2E bench recipe in `benchmarks/round5.md`. 745 tests pass. |
>
> **Item 8 — Phases 1 + 2a LANDED 2026-04-19. Next: Phase 2b (worker
> admit hook + remote-serving IPC).**
>
> Phase 1 wired the bookkeeping: BLAKE3 chained block-hash computation
> in `PrefixCache`, new `WorkerMsg::PrefixManifestUpdate` IPC verb,
> daemon-side `prefix_manifest_tx` channel + `spawn_prefix_announce_forwarder`,
> new `SwarmMessage::PrefixCacheAnnounce` gossip variant, daemon
> dispatch handler that validates sender + replaces the per-`(peer,
> model)` index entry, `state.models.cross_node_prefix_index` +
> `peer_prefix_blocks` reverse index for O(1) cleanup on peer departure
> (wired into `ShardRebalancer::PeerLeft`). Single-node loopback
> verification: forwarder records our own NodeId under our blocks so a
> 1-node setup can confirm the wire path end-to-end via
> `cross_node_prefix_holders`. Tests: +6 prefix_cache (chain math,
> partial-block handling, zero-size guard, insert returns manifest,
> enumerate dedups, enumerate-zero is empty) +5 model index
> (replace/supersede/multi-peer/forget/empty-announce) = +11 net, **756
> total** on `dev,claude-subscription`.
>
> **NEXT SESSION — Item 8 Phase 2 (real multi-peer KV block transfer).**
>
> Item 7 Phase 4 + admit-coalescing + isolation flag all landed and
> measured on the RTX 3070 with TinyLlama Q4. **Item 8 — cross-node
> prefix cache sharing** is a multi-session arc that is SwarmLLM's
> biggest distinguishing P2P speedup story (Items 4–7 and 12 are mostly
> single-node wins or pipeline-parallelism refinements; Item 8 is the
> one that makes the swarm qualitatively more valuable than a single
> node).
>
> **Scope sketch (Item 8):**
> - Each worker announces BLAKE3 prompt-prefix hashes (64- or 128-token
>   blocks) it has in its local `PrefixCache`.
> - Announcements go over gossip with TTL, similar to shard announces.
> - Coordinator builds a `prefix_hash → {peer_id}` index per model.
> - On admit, compute the incoming prompt's block hashes; on cache miss
>   locally but hit on the index, fetch the KV block from the
>   announcing peer over a new IPC / stream message and seed the
>   per-request KV cache.
> - Content-addressed, so it naturally fits the existing shard
>   announcement infrastructure. Verify each block by BLAKE3 on receipt.
>
> **Phase 1 suggested:** wire hash announcement + index + loopback
> verification (coordinator fetches from its own peer loopback). Phase
> 2: real multi-peer KV transfer. Phase 3: authentication + trust
> integration so untrusted peers' KV blocks get re-verified. Phase 4:
> bench.
>
> **Other candidates (deferred for a later session):**
> - **Larger-model Phase 4 bench** (small): Phi-3.5-mini / Qwen-7B
>   numbers. OOMs on RTX 3070 8 GB at default `max_seq_len_override=8192`
>   × 8 slots; drop override to 2048 or shrink `batch_generate_max_slots`
>   to 2–4. Bigger hidden_dim should show more FFN-fusion win than
>   TinyLlama's 1.57×. Isolation flag `batched_prefill_forward` is now
>   available for clean A/B.
> - **Items 14 / 17 / 18** (large research items, see Round 3 list):
>   Mirror Speculative Decoding (Apple), disaggregated prefill/decode,
>   per-token early-exit.
>
> **Session state to recall on resume:**
> - Last session: Item 7 Phase 4 + admit-coalescing + isolation flag
>   landed + GPU E2E measurement (2026-04-19). Commits `7bb306c`,
>   `a29f273`, plus latest (flag). See `docs/plans/benchmarks/round5.md`.
> - Tests: 678 lib + 67 integration = **745 total** on `dev,claude-subscription`.
> - Default-on speedup stack: Items 4, 5, 3, 16 (A+B), 7 Phases 1+2, 7 Phase 4.
> - Config flags added: `batched_prefill_forward` (default `true`).
> - Pre-staged models: TinyLlama-1.1B, Phi-3.5-mini, Qwen2.5-Coder-7B
>   (see `memory/local_model_shards.md`). **Use Qwen2.5-7B or Phi-3.5
>   for GPU benchmarks, not TinyLlama** (too small — `memory/bench_large_models.md`).
>   Both OOM at default `max_seq_len_override=8192` × 8 slots; drop to
>   2048 or shrink `batch_generate_max_slots` to fit.
> - User env: RTX 3070 Laptop 8 GB, WSL2, CUDA works via `/usr/lib/wsl/lib`.
>   Default test build: `cargo build --no-default-features --features dev,claude-subscription`.
>   GPU work: `dev,candle-cuda` (release build takes ~7 min for LTO +
>   codegen-units=1 on full swarmllm crate). Worker subprocess picks up
>   log level from config.toml `[logging] level` — set to `"debug"` for
>   DIAG trace visibility.
>
> **Other work items (not prioritized for the next session but tracked):**
> - Item 2 with a matched-backend draft model (research flagged `Qwen2.5-0.5B`
>   as draft for `Qwen2.5-7B` target; 1.4× on llama.cpp benchmarks).
> - Item 13 Q8_0 end-to-end multi-segment benchmark (codec verified; needs
>   2+-peer pipeline to measure wire savings).
> - Item 12 DSD multi-segment WAN benchmark (needs real WAN topology; loopback
>   can't exercise it because `N≥2` pipelines route to a peer that holds more
>   layers via Item 4 fast path first).
> - Item 16 Phase A routing A/B on a cluster with mixed peer latencies.
> - Extending Item 4 to multi-segment (requires a "leader peer" model that
>   pulls hidden states from earlier segments — larger scope).
>
> See `memory/local_model_shards.md` for pre-staged benchmark assets.

> **Original baseline (2026-04-17, loopback, 3 nodes, TinyLlama-1.1B, 1 remote segment)**
> - Prefill (25 tokens): 3336 ms
> - Per-token decode: ~148 ms (logged as `segment_ms=147..150`)
> - Wall-clock 30-token response: 15.9 s
> - Original hypothesis: ~100 ms/token was libp2p `request_response` framing + ChaCha seal
> - **Actual finding**: after landing Item 1 (persistent stream), per-token `segment_ms` was unchanged — libp2p framing was NOT the bottleneck. The bottleneck was the per-token coordinator/remote IPC round trip itself, which Item 4 eliminated.

Items 1–3 are documented below in their original "plan" voice because a lot of infrastructure landed under them (codec trailers, IPC types, KV truncation primitives, etc.). Items 1–3 implementations are behind feature flags so the default path uses Item 4.

---

## Item 1 — Persistent Bidirectional Stream Per Pipeline Session

**Goal:** Replace per-token `request_response.send_request()` (fresh substream per token) with one long-lived libp2p stream per `(coordinator, remote_segment, request_id)`.

**Design:**
- New protocol `/swarmllm/pipeline/1.0.0` via `libp2p-stream = "0.3"` (compatible with our pinned libp2p 0.55). Coexists with the existing `/swarmllm/1.0.0` request_response — that path remains the fallback.
- Frame format on the stream: reuse the existing `[tag:1][len:4][payload]` codec from `src/network/protocol/mod.rs`. Payloads are exactly the output of `encode_layer_forward` / `encode_layer_result`. No new serialization.
- Coordinator opens the stream lazily on the first `LayerForward`. `SharedState` gets a `pipeline_streams: DashMap<Uuid, mpsc::Sender<Vec<u8>>>` for outbound frames.
- Correlation: the in-band `request_id: Uuid` already present in `LayerForward`/`LayerResult` replaces libp2p's `OutboundRequestId` correlation. Coordinator-side reader dispatches `LayerResult` frames to per-request `mpsc` receivers.
- Encryption: `SessionManager::seal/open` (src/crypto/session.rs:330/378) wraps the payload exactly as today. The `CachedSession.send_nonce` counter advances per frame on the same stream — monotonic, no reset mid-session.
- Backpressure: bounded `mpsc::channel(8)` per stream each direction. Natural backpressure via `send().await`.
- Lifecycle: stream closes when the HTTP request ends (`token_tx` closes), `finish_reason` is set, or a 10-minute idle timer fires (matches KV cache TTL).
- Failover: on stream `io::Error`, handle is removed from `pipeline_streams`; the existing `failover_segment` (src/inference/pipeline/distributed.rs:767) is called unchanged and uses `NetworkCommand::SendTensor` as the resilient fallback.

**Files to modify:**
- `Cargo.toml` — add `libp2p-stream = "0.3"`
- `src/network/behaviour.rs` — add `pipeline_stream: libp2p_stream::Behaviour` to `SwarmBehaviour`
- `crates/swarmllm-types/src/network.rs` — add `NetworkCommand::OpenPipelineStream { target_peer_bytes, request_id }`, `SendPipelineFrame { request_id, payload }`, `ClosePipelineStream { request_id }`
- `src/daemon/state/mod.rs` — add `pipeline_streams: DashMap<Uuid, mpsc::Sender<Vec<u8>>>` + per-request result receivers
- `src/network/manager/mod.rs` — `handle_open_pipeline_stream`, `handle_send_pipeline_frame`, incoming-stream acceptor, per-stream reader task
- `src/inference/pipeline/distributed.rs` — `forward_through_segments` (line 456): branch on `config.inference.persistent_pipeline_stream`; on true, use stream; otherwise unchanged `SendTensor` path
- `src/config.rs` — add `persistent_pipeline_stream: bool` (default `false` until validated)

**Files to add:**
- `src/network/pipeline_stream.rs` — custom `PipelineStreamBehaviour` wrapping `libp2p_stream::Control` + remote-side read loop that decodes frames → calls `handle_layer_forward` → writes `LayerResult` frames back

**Build order:**
1. Add `libp2p-stream = "0.3"` to `Cargo.toml`. Verify compilation against libp2p 0.55.
2. Extend `NetworkCommand` enum (3 new variants).
3. Add `pipeline_streams` map + result-receiver map to `SharedState`.
4. Create `src/network/pipeline_stream.rs` with frame I/O helpers (reusing existing codec).
5. Wire `pipeline_stream` behaviour into `SwarmBehaviour::build_behaviour()`.
6. Wire inbound stream handler in `NetworkManager`: spawn reader task that decodes + dispatches to `handle_layer_forward`, writes `LayerResult` back via the same stream.
7. Wire outbound path in `handle_open_pipeline_stream` / `handle_send_pipeline_frame`.
8. Modify `forward_through_segments` to prefer the stream path; keep `SendTensor` fallback on stream-open timeout (500 ms).
9. Wire stream close on session end + idle timer.

**Test plan:**
- Unit: `write_frame` / `read_frame` round-trip in `pipeline_stream.rs`.
- Integration: 2-node `tokio::test` — open stream, send 10 `LayerForward` frames, assert 10 `LayerResult` frames return with matching `request_id`.
- Latency regression: 3-node test, 30-token response, measure per-token `segment_ms` before and after; assert p50 drops by ≥50 ms.

**Success metric:** per-token `segment_ms` drops from ~148 ms → ≤50 ms on loopback.

---

## Item 2 — Speculative Decoding Across Distributed Pipeline

**Context:** `src/inference/speculative.rs` (370 lines) already has `SpeculativeDraftState`, `SpeculativeResult`, `accept_reject`. Already wired for the LOCAL path (`src/inference/pipeline/local.rs:24`); draft model loads at boot (`src/daemon/mod.rs:208-221`). Distributed path still uses single-token round-trips.

**Goal:** Coordinator runs draft model locally, proposes γ tokens per round-trip, remote verifies all γ in one forward pass, coordinator applies accept-reject and advances by k+1 tokens on average.

**Design:**
- `LayerForward` gains `draft_tokens: Vec<u32>` + `spec_logits_requested: bool` (serde default, backward-compatible). `LayerResult` gains `spec_logits: Vec<Vec<f32>>` + `top_k_logits: Option<u32>`.
- Binary codec: new `0x03` trailer markers in `encode_layer_forward` / `encode_layer_result`. Old decoders see an unknown trailer and ignore it only if the new sender side is gated — so we gate with `PipelineAssignment.supports_speculative: bool`, set by the router only when all segments' node versions support it.
- Remote verify: worker runs one forward over γ positions in KV-cache append mode (candle supports this — `LayerForward.activations` already carries i64 LE token IDs; extend to γ tokens). Returns γ+1 logit vectors.
- Bandwidth: full vocab × γ × 4 bytes = ~512 KB at γ=4, vocab=32k. Return only top-K=200 logits to cut this to ~4 KB.
- Accept-reject on coordinator side (cleaner — remote stays dumb). Uses existing `accept_reject()` from speculative.rs.
- KV truncation on rejection: coordinator sends next `LayerForward` with `index_pos` = position of the rejection point. `KvCacheStore` gets a `truncate_to(request_id, to_pos)` helper that resets each layer's `KvCache` above the threshold. Simpler alternative: overwrite path — candle's `KvCache` uses `index_pos` for the write offset, so next forward naturally overwrites positions ≥ rejection point.
- Config: `speculative_distributed: bool` (default `false`), `speculative_top_k_logits: u32` (default 200). `speculative_gamma` already exists.

**Coupling with Item 1:** Item 2 works on the `request_response` path before Item 1 is merged; the γ-batched `LayerForward` is just a larger payload. With Item 1's stream it flows naturally in the same frame format.

**Files to modify:**
- `crates/swarmllm-types/src/inference.rs` — extend `LayerForward`, `LayerResult`, `PipelineAssignment`
- `src/network/protocol/layer_forward.rs` — `0x03` trailer for `draft_tokens`
- `src/network/protocol/layer_result.rs` — `0x03` trailer for `spec_logits`
- `src/inference/pipeline/distributed.rs` — speculative branch in `execute_distributed`
- `src/inference/split/kv_cache.rs` — add `truncate_to` helper
- `src/inference/split/executor.rs` — multi-position forward mode when `draft_tokens.len() > 0`, collect γ+1 logit vectors
- `src/daemon/dispatch/layer_forward.rs` — populate `spec_logits` when `spec_logits_requested`
- `src/config.rs` — add config fields

**Files to add:**
- `src/inference/pipeline/speculative_distributed.rs` — `forward_speculative_distributed()` loop (γ-token batched round-trip), called from `execute_distributed` under the feature flag

**Build order:**
1. Extend wire types (serde-default for backward compat).
2. Update binary codec with `0x03` trailers.
3. Add `truncate_to` to `KvCacheEntry`.
4. Add multi-position forward to `split/executor.rs`.
5. Update `handle_layer_forward` to populate `spec_logits`.
6. Implement `forward_speculative_distributed()`.
7. Wire into `execute_distributed` behind capability check + config flag.
8. Add config fields.

**Test plan:**
- Correctness (greedy): temperature=0 output identical to non-speculative distributed output on the same prompt.
- Throughput: mean `tokens_per_roundtrip` ≥ 1.8 at γ=4.
- Rejection edge case: force-zero draft tokens, assert KV truncation + final output still coherent.
- Log `acceptance_rate` per request (already tracked by `SpeculativeDraftState::record_batch`).

**Success metric:** mean tokens per round-trip ≥ 1.8 at γ=4 on TinyLlama.

### Status (2026-04-17)

**Phase 1 landed** (commit `49bea39`): wire types, binary codec `0x03` trailers, `SplitModel::forward_verify_all_positions()`, worker multi-position verify branch, config flag, codec tests. Flag currently logs "coordinator loop pending" when enabled.

**Phase 2 landed** (this commit): KV truncation primitives.
- `KvCacheEntry::truncate_to(len)` — snapshot narrow + reset + re-append. O(len * hidden * layers) per call.
- `KvCacheStore::truncate_request_to(...)` — store-level wrapper.
- `LayerForward.truncate_kv_to: Option<u32>` wire field + `0x04` codec trailer (plain + encrypted).
- `IpcForward` pass-through; model worker applies truncation before forward.

**Phase 3 landed** (this commit): `src/inference/pipeline/speculative.rs`. `try_speculative_distributed` runs when conditions hold (single-segment remote, greedy temp=0, draft model loaded, no encryption/vision/LoRA). Inside one `draft_executor` lock, prefills the draft's llama-cpp context and runs the spec round loop: draft γ tokens → send `LayerForward { draft_tokens: [last_tok, q_1..q_γ], spec_logits_requested: true, truncate_kv_to }` → greedy argmax compare against returned spec_logits → emit `[q_1..q_k, bonus]` → sync draft KV (via `clear_kv_cache_seq` on partial reject) → set `pending_truncate = expected_kv_len` for next round. Eligibility falls through to non-speculative when not met. Exposes `ModelExecutor::raw_model()` / `raw_backend()` so the draft context can be created and held across async network round trips.

**Remaining work (future):**
- Wire speculative under encryption (reuse `encode_forward_for_wire` pattern from `pipeline_stream.rs`)
- Wire speculative for multi-segment pipelines (propagate spec_logits through intermediate segments)
- Non-greedy support via `speculative::accept_reject` (needs transmitting γ draft probabilities alongside tokens)
- Benchmark at various γ values to find sweet spot

---

## Item 3 — Continuous Batching on Remote Segment Holder

**Current state:** `ModelProcessPool` (src/inference/process_pool.rs:68) holds one `WorkerHandle` per `ModelId`, with the IPC socket behind a `Mutex`. Every `forward()` serializes through this mutex. No scheduler today.

**Goal:** Batch N concurrent decode steps from different requests into one forward on the holder.

**Design:**
- Decode-only batching first (seq_num > 0, single-token inputs). Prefill (seq==0) always runs solo.
- Replace `Mutex<socket>` in `WorkerHandle` with `requests_tx: mpsc::Sender<BatchRequest>` + a scheduler task per worker.
- Scheduler: waits for the first request (blocking), then uses `timeout(batch_collection_ms, rx.recv())` in a loop to collect up to `max_concurrent_decode_batch` additional requests within a time budget (default 5 ms, effective ~15 ms on WSL2 — acceptable).
- Batch IPC: new `WorkerIpcRequest::BatchForward { requests: Vec<LayerForward> }` + `WorkerIpcResponse::BatchResult { results: Vec<LayerResult> }`.
- Worker side, v1: sequential loop over requests inside the subprocess. This already removes mutex contention and halves the IPC overhead per request. v2 (optional later): `Tensor::stack` the hidden states for a single matmul.
- KV cache: each request keeps its own `KvCacheEntry` — no shared tensor. Decode shape `[1,1,hidden]` per request means no padding needed.
- Config: `continuous_batching: bool` (default `false`), `max_concurrent_decode_batch: usize` (default 8), `batch_collection_ms: u64` (default 5).

**Interaction with Items 1 & 2:** Item 1's persistent stream lets requests arrive asynchronously without per-token substream setup, making the batch scheduler's collection window effective. Item 2 stacks: each batch slot carries γ draft tokens, so effective throughput is `batch × γ` tokens per worker forward.

**Files to modify:**
- `src/inference/process_pool.rs` — replace `Mutex<socket>` with `mpsc::Sender<BatchRequest>`; spawn scheduler task per worker
- `src/inference/worker_ipc.rs` — add `BatchForward` / `BatchResult` variants
- `src/inference/model_worker.rs` — handle `BatchForward`: v1 sequential loop over requests
- `src/config.rs` — add config fields

**Files to add:**
- `src/inference/scheduler.rs` — `BatchScheduler` with time-budgeted collection loop

**Build order:**
1. Add config fields, defaults off.
2. Add `BatchForward` / `BatchResult` IPC variants.
3. Implement `forward_batch()` in `model_worker.rs` (sequential v1).
4. Implement `BatchScheduler`.
5. Modify `ModelProcessPool` to use scheduler when flag on.
6. Correctness tests.
7. (Phase 2, optional) Stacked-tensor batched matmul.

**Test plan:**
- Correctness: 2 concurrent requests produce identical tokens to 2 serial requests.
- Throughput: 2 concurrent requests, measure aggregate tok/s vs. serial baseline. Target ≥1.6×.
- Scheduler unit test: inject N `BatchRequest`s with artificial delay, assert grouping.

**Success metric:** 2 concurrent requests, aggregate tokens/sec ≥ 1.6× vs. serial baseline.

### Status (2026-04-18)

**Phase 1 landed**: wire protocol + config flags only.
- Config: `continuous_batching: bool`, `max_concurrent_decode_batch: u32` (default 8), `batch_collection_ms: u64` (default 5).
- IPC: `DaemonMsg::BatchForward { requests, activation_lens }` + `WorkerMsg::BatchResult { results, activation_lens }`.
- Worker: `handle_batch_forward` stub that dispatches sequentially through the existing `handle_forward` path (no mutex contention win on the daemon side, no compute-side batching yet).

**Phase 2a landed 2026-04-18**: response multiplexing for concurrent same-model requests.
- Replaced `WorkerHandle.socket: Mutex<(ReadHalf, WriteHalf)>` with `writer: Mutex<OwnedWriteHalf>` + a dedicated reader-actor task that owns the read half and routes each inbound `WorkerMsg` to a per-request `mpsc::Sender<(WorkerMsg, Vec<u8>)>` keyed by `request_id`.
- `forward()` / `generate()` register a response channel, briefly lock the write half to send one framed message, then drain their channel off-lock until a terminal message arrives. Unregistered via RAII `ResponseGuard` on drop.
- Concurrent requests for the same model no longer serialize on a full-request mutex — they interleave through the worker at the compute-side boundary instead.
- All 699 tests pass unchanged.

**Phase 2b landed 2026-04-18 + auto-coalescing scheduler landed DEFAULT-ON 2026-04-19**: compute-side fused batching available via `ModelProcessPool::forward_batch()`, and auto-wired into every `forward()` call when `continuous_batching=true` (default).
- `SplitModel::forward_batch` (already implemented under `#[cfg(test)]`) un-gated. Stacks per-request decode inputs into `[batch_size, 1, hidden]` for batched QKV + FFN matmul, per-slot attention on each slot's KV cache, then un-stacks outputs. Supports Dense/DeepSeek/Qwen3.5 attention + SSM variants.
- `handle_batch_forward` in `model_worker.rs` dispatches to `forward_batch` when `batch_eligible(&requests)` passes (same layer_range, no vision/LoRA/spec/TP/pre-embedded, all decode `seq_num > 0`). Emits a single `WorkerMsg::BatchResult` with N concatenated payloads. Falls back to sequential `handle_forward` loop on ineligibility or fused-path error.
- `ModelProcessPool::forward_batch(Vec<LayerForward>) -> Vec<LayerResult>` is the public entry point. Registers N per-request response channels, sends ONE `DaemonMsg::BatchForward` over the multiplexed path, and the reader actor fans out `BatchResult` back to the N callers (each sees a synthesized `LayerResult`).
- **Auto-coalescing scheduler (2026-04-19, default-on).** Single `batch_scheduler_loop` task spawned once per `ModelProcessPool` from `SharedState::new`. `forward()` checks `continuous_batching` + `forward_is_schedulable` (same eligibility as worker-side `batch_eligible`: decode-only, no vision/LoRA/spec/TP/pre-embedded/truncate) and enqueues `BatchSchedulerMsg::Forward { fwd, resp_tx }` through a bounded mpsc. Scheduler collects arrivals within `batch_collection_ms` (default 5 ms, effective ~15 ms on WSL2), groups by `model_id`, dispatches groups of ≥2 through `forward_batch` and singletons through `forward_direct`. Fan-out preserves caller order. Worker's CPU-fallback guard means this is safe on every device — GPU workers run fused, CPU workers run sequential, with zero caller-visible change.
- **Default.** `continuous_batching = true` as of 2026-04-19. Single-request workloads are unchanged (singletons bypass the batch path and go direct). Multi-request workloads pay the ~5 ms collection window in exchange for 1.34–1.55× GPU throughput at batch 2–8 (see `benchmarks/round3.md`).

---

## Cross-Item Sequencing

**Order:** 1 → 2 → 3.

- **Item 1** is independent; biggest single-user latency win. Ships first.
- **Item 2** builds cleanly on Item 1's stream but works on the `request_response` path too. All wire changes are serde-default backward-compatible.
- **Item 3** benefits from both — Item 1 makes concurrent requests arrive smoothly; Item 2 multiplies effective throughput per worker forward.

**Benchmark capture points:**
- After Item 1 → `docs/plans/benchmarks/item1.txt`: per-token `segment_ms` on 3-node loopback, TinyLlama, 30-token response.
- After Item 2 → `docs/plans/benchmarks/item2.txt`: mean `tokens_per_roundtrip` at γ=4.
- After Item 3 → `docs/plans/benchmarks/item3.txt`: 2-concurrent aggregate tok/s.

---

## Risk & Rollback Summary

- **Failover correctness:** stream break removes handle from `pipeline_streams`, existing `failover_segment` uses `NetworkCommand::SendTensor` (the unchanged `request_response` path) — so failover works even if the stream layer is broken.
- **Encryption/nonce safety:** `SessionManager.send_nonce` is a monotonic counter per `CachedSession`. Nonces never reset while the session lives; a stream rebreak triggers a fresh session (counter resets are correct). Pipeline-seal (ephemeral X25519 per result) is unchanged. Only one forward outstanding at a time per stream (coordinator awaits each result) — no nonce race, including with Item 2's batched payloads.
- **Backward compatibility:** all type additions are `#[serde(default)]`; old nodes ignore them. Binary codec additions use new `0x03` trailer markers gated by `PipelineAssignment.supports_speculative`. Stream protocol: old nodes don't negotiate `/swarmllm/pipeline/1.0.0`, coordinator falls back to `request_response`.
- **WSL2/Tokio timer quirks:** 5 ms batch window degrades to ~15 ms (still batches). 10-minute idle timer fires late, not on hot path.

**Rollback per item** — all single-config-line toggles:
- Item 1: `persistent_pipeline_stream = false` → unchanged `SendTensor` path.
- Item 2: `speculative_distributed = false` → unchanged single-token decode loop.
- Item 3: `continuous_batching = false` → unchanged `Mutex<socket>` path.

---

## Item 4 — Remote-generate fast path (LANDED 2026-04-18)

**Context for future readers**: Items 1–3 were research-driven infrastructure.
Item 4 shipped after external research (vLLM V1 architecture,
HuggingFace continuous-batching post, mistral.rs reference impl) identified
the actual bottleneck for single-user single-segment distributed inference:
**the per-token coordinator/remote round trip**, not libp2p framing or
compute batching.

### Design

When the distributed pipeline resolves to a single remote segment (one peer
holds the entire layer range), the coordinator sends ONE
`SwarmMessage::RemoteGenerateRequest { prompt, sampling, ... }` to the
holder. The holder runs the full decode loop inside its local worker
subprocess (same `handle_generate` path as local-API inference), and streams
every generated token back as a `SwarmMessage::StreamingToken` carrying
pre-decoded text. The coordinator registers a `streaming_token_txs[req_id]`
channel before sending and drains it until a `finish_reason` arrives.

### Eligibility

Fast path taken when ALL hold:
- single segment (`assignment.segments.len() == 1`)
- no TP groups
- segment is remote (not local)
- no vision / LoRA
- no pipeline sealing / local_embedding_privacy

Falls through to the standard per-token loop otherwise. Libp2p Noise already
encrypts the wire — no additional ChaCha session layer is added (matches the
security posture of `SwarmMessage::InferenceRequest` which also carries user
prompts in plaintext over Noise).

### Measured results

TinyLlama Q4_K_M, 3-node loopback, encryption default:

| Path | 100-token completion | Decode rate |
|---|---|---|
| Per-token (baseline) | ~30 s | ~270 ms/tok |
| Fast path (this patch) | ~15.9 s | ~125 ms/tok |

**1.93× decode speedup, 1.75× wall-clock** for typical single-user
single-segment workloads.

### Files

- `crates/swarmllm-types/src/inference.rs` — added `RemoteGenerateRequest`,
  `GenerateUsage`; extended `StreamingToken` with `text` + `usage` fields
  (backward-compatible serde defaults).
- `crates/swarmllm-types/src/network.rs` — new `SwarmMessage::RemoteGenerateRequest` variant.
- `src/daemon/dispatch/remote_generate.rs` — remote-side handler: invokes
  `ModelProcessPool::generate` with a token channel, forwards each token
  to the coordinator, emits a final done token with usage.
- `src/inference/pipeline/remote_generate.rs` — coordinator-side
  `try_remote_generate_fastpath`: eligibility + request send + streaming
  token collection.
- `src/inference/pipeline/distributed.rs` — dispatches to the fast path
  FIRST in `execute_distributed`.

### Remaining scope

- Multi-segment pipelines (pipeline sharded across 2+ peers). For MVP we
  only chase the single-segment case because it's the dominant deployment
  pattern. Multi-segment gains require propagating tokens back through a
  reverse-pipeline.
- Vision / LoRA — currently fall through to per-token. Adding these
  requires threading the extra inputs through `ModelProcessPool::generate`.
- Combining with Item 2 speculative: speculative's `draft_executor` still
  runs coordinator-local; the fast path eliminates the network round trip
  that speculative was partly trying to amortize. A future integration
  would make speculative drafting happen on the REMOTE with tokens streamed
  back, but the fast path already delivers most of the speculative speedup
  with far less complexity.

---

# Round 2 (2026-04-18) — Prefix caching, self-speculation, proper batching

> **Context:** Round 1 Items 1–3 landed behind flags; Item 4 landed default-on (1.93× decode). Research scan (2026-04-18) identified three higher-leverage wins that the original plan didn't cover. Item 3 Phase 2 as originally scoped is partly obsolete because the remote-generate fast path (Item 4) bypasses `handle_forward` — so the batching scope below targets `handle_generate` instead.

## Item 5 — Cross-request prefix caching (RadixAttention-style) ✅ LANDED 2026-04-18

### Validation

TinyLlama-1.1B Q4_K_M on CPU, 513-token prompt, `max_tokens=5`:

| Request | Latency | Notes |
|---|---|---|
| Cold (cache miss) | 41.66 s | Full prefill of 513 tokens + 6 decode |
| Warm (cache hit at 512) | 1.42 s | Prefill 1 suffix token + 6 decode |

**29.4× wall-clock speedup on a same-prompt re-submission.** The cache inserts
9 block-aligned snapshots on the first completion; the second request matches
at 512 (block-aligned, one token short of full match so one forward still runs
to produce sampling logits).

Log excerpt from the second request:

```
DIAG: prefix-cache HIT model_key="0-22-22" matched_tokens=512 prompt_tokens=513
DIAG: handle_generate prefix-cache HIT — prefilling suffix only
```



**Problem the current design doesn't solve.** Today `KvCacheManager` (src/inference/kv_cache.rs) does same-session multi-turn prefix matching — it requires the caller to provide `session_id`, and each session has a single cached prompt prefix. Multiple simultaneous requests that share a long system prompt (Claude Code's 10 KB agent scaffold, RAG templates, MCP tool descriptions) all re-run prefill from scratch. Every other production inference engine (SGLang, vLLM V1) has a shared radix tree of token-prefix → KV blocks, hit rate is routinely 50–99% in agentic workloads.

**Design.**
- Worker-side only (no cross-node sharing in v1 — that's Item 9 below).
- New `PrefixKvCache` type in `src/inference/split/kv_cache.rs` parallel to the existing per-request `KvCacheStore`. Data model:
  - Token-level radix tree: each node stores a `Vec<u32>` token sequence plus per-layer `(K, V)` tensors covering those tokens
  - Ref-counted (`Arc`) nodes; LRU eviction by last-hit timestamp
  - Bounded by a new `prefix_cache_max_tokens: u64` config field (default 262_144 — 256 K tokens worth of cached KV, ~1 GB for a 7B model at fp16)
- Hit flow: on `handle_generate`, after tokenizing the prompt, walk the radix tree to find the longest matching token prefix. Materialize a per-request `KvCacheEntry` by cloning the cached K/V tensors for matched layers (shallow clone — candle Tensors are reference-counted). Set `index_pos = matched_len`. Then run prefill only on the suffix.
- Miss flow: run full prefill, then call `prefix_cache.insert(token_prefix, kv_entry)` with a copy of the final KV state. Insertion is chunked into blocks (configurable, default 32 tokens) so the tree branches cleanly.
- Integration point: `handle_generate` in `src/inference/model_worker.rs` (and later `handle_batch_forward` too). The `KvCacheStore` per-request entries stay as they are — we're adding a read-through layer on top.
- Multi-turn session cache still works as-is, now backed by the shared radix tree instead of its own HashMap.

**Eligibility.**
- Only `handle_generate` path in v1 (covers the Item 4 fast path and local API). Extend to `handle_batch_forward` in Item 7.
- Skip if `privacy_mode` is on (don't persist prompts across requests).
- Skip if prompt has been templated with a time-varying component (detect this by checking whether prompt is stable — we just always cache and let the LRU evict low-hit entries).

**Files.**
- `src/inference/split/kv_cache.rs` — add `PrefixKvCache`, `RadixNode`, `insert`, `lookup` (returns `Option<MatchedPrefix { kv: KvCacheEntry, token_len: usize }>`), `evict_lru`.
- `src/inference/model_worker.rs` — `handle_generate`: lookup → populate KvCacheStore → prefill suffix only → on completion, insert final KV back.
- `src/config.rs` — add `prefix_cache_max_tokens: u64` (default 262_144), `prefix_cache_block_tokens: usize` (default 32), `prefix_cache_enabled: bool` (default **true** — this is a pure win).
- `src/daemon/state/mod.rs` — `ModelProcessPool` already owns the worker; prefix cache is per-worker-subprocess, stored in worker-local static.

**Success metric.** TTFT on a second request with matching 2048-token prefix: drops from prefill-time (~3 s on TinyLlama) to <100 ms on CPU. On repeated agent-scaffold requests, effective decode p50 should match single-user baseline regardless of prompt size.

**Build order.**
1. Add config fields + plumb through to worker.
2. Implement `PrefixKvCache` + radix tree in isolation with unit tests.
3. Wire into `handle_generate` behind config flag.
4. Benchmark: 10 sequential requests with same 2048-token system prompt, measure TTFT.

---

## Item 6 — SWIFT self-speculative decoding (layer-skip draft) ✅ LANDED 2026-04-18 (flag, no measured win on TinyLlama CPU)

### Status

**Landed behind `swift_self_speculative=false` flag.** Implementation:
- `src/inference/swift.rs` — `build_skip_mask()` produces a contiguous middle-band skip pattern with the outer 2 layers preserved on each side (per SWIFT paper). `SwiftCalibrator` tracks aggregate accept rate (currently observability only; v2 will rotate candidate patterns).
- `src/inference/split/executor.rs` — added `forward_with_skip_mask()`. Inside `forward_inner_impl`, layers in the skip mask are identity-passed (no attention, no MLP, no KV write).
- `src/inference/model_worker.rs` — `swift_decode_loop()` runs the γ-token draft → KV truncate → γ+1-token verify → greedy accept-reject → final KV truncate cycle. Falls through to baseline when SWIFT is off, temperature ≠ 0, model has < 8 layers, or `max_tokens < γ+1`.
- Plumbed through `process_pool.rs` → `model-worker` CLI args, identical pattern to prefix-cache config.

### Measured results (2026-04-18)

TinyLlama-1.1B Q4_K_M, single-node loopback, 100-token greedy completion:

| Setup | Baseline | SWIFT γ=4 skip=0.45 | SWIFT acceptance |
|---|---|---|---|
| CPU | 6.85 tok/s | 1.95 tok/s (3.5× slower) | 13.8% |
| GPU (CUDA, RTX 3070) | ~50 tok/s | 13.5 tok/s (3.7× slower) | 10.7% |

Sanity checks (skip_ratio=0, draft = full target → expected 100% acceptance):
- CPU: 92.0% (NOT 100%)

**SWIFT loses on TinyLlama on both CPU AND GPU. Root cause is an attention-kernel-dispatch mismatch in candle, not the SWIFT algorithm itself.**

**Why.** candle dispatches attention based on tensor shape:
- **CPU path** (`run_attention`): `seq_len=1` → `standard_attention` (matmul); `seq_len≥2` → `cpu_flash_attention`.
- **GPU path** (with `flash-attn`): single-position decode → `flash_attn`; multi-position with `k_len > q_len > 1` (KV cache pre-populated with prefix tokens) → **falls back to `standard_attention`** because flash-attn's boolean causal flag can't express the offset causal mask used in this case.

Net: baseline per-token decode and SWIFT's verify pass run on **different attention kernels**. Numerically close but not identical — the top-1 and top-2 logits flip in close-call cases. Even with `skip_ratio=0` (draft = full target = no actual skipping), draft-argmax and verify-argmax disagree on ~8% of positions on CPU. With real layer skipping, acceptance collapses to 10–14%, each round emits only ~1.5 tokens, and the per-token cost stays ~3.5× higher than baseline.

**Output also diverges from greedy baseline** by a few token choices (e.g. "studied and refined" vs "studied and tested" mid-paragraph) for the same reason — verify produces target's argmax under different attention numerics than per-token baseline.

### What unblocks SWIFT

1. **Force the same attention kernel everywhere** when SWIFT is active. Add a `force_standard_attn` flag that SWIFT sessions turn on, so baseline + draft + verify all use `standard_attention`. Costs a bit on baseline decode but makes SWIFT correct + speeds up verify amortization (verify cost grows sub-linearly with seq_len thanks to weight-load amortization, but only if both paths use the same kernel).
2. **Test on larger models.** Paper measures 0.45–0.50 acceptance on LLaMA-2-13B/70B (vs our 0.10–0.14 on 1.1B). Bigger model → more layer redundancy → much higher accept. Blocker: Phi-3.5-mini reports 131072 max context in its GGUF and candle pre-allocates the full KV buffer at first forward → OOMs an 8GB GPU. Would need a `max_seq_len_override` config to make any larger model fit.
3. **v2 calibration** of skip pattern (Bayesian / random search vs the fixed middle-band v1 ships).

### Round 2.1 (2026-04-18) — All three unblockers landed; SWIFT still loses

All three landed:
- `src/inference/attn_kernel.rs` + thread-local `ForceStandardAttnGuard` — baseline + draft + verify all run through `standard_attention` when SWIFT is active.
- `InferenceConfig::max_seq_len_override` + process-global `MAX_SEQ_LEN_OVERRIDE` — the loader clamps GGUF `context_length` so 128K-context models fit on small VRAM.
- v2 calibration in `swift.rs`: 5 candidate skip patterns (varying start position, fixed width = `skip_ratio × num_layers`), round-robin during the warmup window, then pin the highest-accept candidate.

Re-bench after the unblockers (RTX 3070 8GB, CUDA, 100-token greedy, force_standard_attn = true on baseline for fair comparison):

| Model | Skip | SWIFT decode | SWIFT accept | Baseline (force_standard) |
|---|---|---|---|---|
| TinyLlama-1.1B (22L) | 0.45 | 21.4 tok/s | 11.8% | ~50 tok/s |
| Phi-3.5-mini (32L) | 0.0 (sanity) | ~50 tok/s | 96.4% | ~50 tok/s |
| Phi-3.5-mini (32L) | 0.15 | 27 tok/s | 56.3% | ~50 tok/s |
| Phi-3.5-mini (32L) | 0.25 | 19 tok/s | 33.0% | ~50 tok/s |
| Phi-3.5-mini (32L) | 0.35 | 14 tok/s | 14.7% | ~50 tok/s |
| Phi-3.5-mini (32L) | 0.45 | 12 tok/s | 4.4% | ~50 tok/s |

The unblockers worked individually:
- skip=0 acceptance jumped from 92% (CPU pre-fix) → **96.4%** (GPU post-fix). The residual 3.6% is matmul-reduction-order noise between seq_len=1 and seq_len=γ+1 forwards — not a logic bug.
- max_seq_len_override let Phi-3.5 fit on 8 GB.
- v2 calibration runs end-to-end, picks `selected=Some(idx)` after the warmup, and the chosen pattern stays pinned for the rest of the request.

But SWIFT is still **structurally slower than baseline**:

```
Per-round cost   = γ·draft_forward_cost  +  verify_forward_cost
Per-round emit   = 1 + accepted_count
Per-token cost   = (γ·skip_factor + verify_factor) / (1 + accept)

With γ=4, skip_ratio=0.45 (skip_factor ≈ 0.55), verify_factor ≈ γ+1 = 5
  (verify_factor scales linearly because standard_attention is O(seq_len))
   → cost = (4·0.55 + 5) / (1 + accept) = 7.2 / (1 + accept)
   → To beat baseline (cost = 1) we need accept > 6.2, impossible at γ=4.
```

The verify pass costs ≈ γ+1 baseline forwards under `standard_attention` (linear in seq_len), and there's no setting where γ accept tokens amortize that cost. The SWIFT paper's published 1.3–1.6× speedup assumes verify uses **flash-attention**, where multi-position forward is roughly O(seq_len)-amortized but with a much smaller constant (weight loads dominate). Our verify can't use flash-attn because the offset causal mask isn't expressible via flash-attn's boolean causal flag.

### Conclusion: SWIFT shelved until candle gets flash-attn-with-mask

The three unblockers are correct and useful for any future speculative path, but SWIFT itself doesn't pay off on the candle backend without flash-attention support for offset causal masks (or some other multi-position kernel that runs sub-linearly in seq_len). Recommended next steps:

1. **Stop pushing SWIFT.** Leave it behind the flag. Current implementation is correct, just unprofitable.
2. **Track candle/flash-attn support for offset causal masks** as a prerequisite. Or implement our own GPU kernel.
3. **Move to higher-leverage items** from Round 3 (`Item 12 — DSD` for distributed-spec, `Item 13 — activation compression`, `Item 16 — Parallax scheduler`). DSD doesn't depend on the verify-cost problem because it amortizes across network round trips, not across batched decode positions.

### Why we kept it

The implementation is correct in structure (skip-mask draft + truncated KV + full verify + greedy accept-reject), and the value will appear on:
1. **Larger models** — acceptance rate scales with model size; SWIFT paper measures 0.45–0.50 acceptance on LLaMA-2-13B/70B vs ~0.14 we saw on 1.1B.
2. **GPU backends** — verify's batched matmul amortizes weight loads, breaking even sooner.
3. **High-accept patterns** — Bayesian calibration (v2) should beat the fixed middle-band pattern.

### Remaining work

- v2 calibration: rotate candidate skip patterns during the warmup window, pick best by acceptance.
- Force same attention kernel in verify and per-token paths (bit-equivalent output for tests).
- GPU benchmark (CUDA path uses flash-attn for both decode and verify, so no kernel-mismatch issue).
- Larger-model benchmark (Phi-3.5-mini, Qwen2.5-7B) to find the crossover point where SWIFT wins.

### Original design (kept for reference)

**Problem.** Item 2 (speculative decoding) is behind a flag and requires a pre-staged draft model. None of our pre-staged benchmark models have real draft pairs. SWIFT (arxiv 2410.06916, ICLR 2025) derives the draft from the target model itself by skipping intermediate layers — no extra weights, no training, no shard coordination. 1.3–1.6× decode speedup on every model, universally.

**Design.**
- Two-phase operation:
  1. **Calibration phase** (first N tokens, default 32): run both full-layer forward AND skip-candidate forwards for each position, measure per-candidate acceptance rate, select the best skip pattern (e.g., "skip layers 8–15 out of 32").
  2. **Acceleration phase** (remaining tokens): draft γ tokens using the skip-layer pass; verify all γ with one full forward; apply standard accept-reject.
- SWIFT calibration runs cheaply because the full forward has to happen anyway (we verify); we piggyback.
- Layer skipping is already supported structurally by our shard system — each layer is independently loaded. We just need a runtime "skip mask" in the forward loop.
- Config: `swift_self_speculative: bool` (default `false`; enable after validation), `swift_calibration_tokens: u32` (default 32), `swift_gamma: u8` (default 4), `swift_skip_range: (u8, u8)` (default start_pct=25%, end_pct=75% — skip candidates are contiguous ranges within the middle half of layers).

**Files.**
- `src/inference/split/executor.rs` — add `forward_with_skip_mask(skip_mask: &[bool])` that zeros the contribution of masked layers. For a transformer, this means: for each masked layer i, pass hidden state through UNCHANGED (identity), skipping the attn + FFN compute entirely. This IS what SWIFT does.
- `src/inference/swift.rs` (new) — `SwiftCalibrator` tracks per-skip-pattern acceptance; `pick_best_pattern` returns the winner after calibration.
- `src/inference/model_worker.rs` — `handle_generate`: integrate SWIFT as an alternative to the existing speculative path when `swift_self_speculative` is on and no external draft model is configured.

**Interaction with Item 5.** Orthogonal — Item 5 fixes prefill TTFT, Item 6 speeds decode. They stack multiplicatively.

**Success metric.** Acceptance rate ≥40% at γ=4 after calibration on TinyLlama. Decode rate improvement ≥1.3× vs baseline Item 4 fast path.

**Build order.**
1. Implement `forward_with_skip_mask` in executor; unit test that output matches full forward when no layers skipped.
2. Implement `SwiftCalibrator`.
3. Wire into `handle_generate` behind config flag.
4. Benchmark against Item 4 fast path baseline.

---

## Item 7 — `BatchGenerate` (replaces obsolete Item 3 Phase 2)

**Why the original Phase 2 was obsolete.** The landed Phase 1 batched `handle_forward`, which is only invoked on the per-token round-trip path. Item 4 bypasses that entirely — the dominant distributed flow is now `handle_generate`, not `handle_forward`. To serve concurrent users from one model worker, we need a `BatchGenerate` IPC verb that multiplexes many in-flight decode loops through a single worker process.

**Design.**
- New IPC verb `DaemonMsg::BatchGenerateStep { active_requests: Vec<DecodeStep> }` where `DecodeStep` contains `{ request_id, last_token_id, sampling_params }`. Response: `WorkerMsg::BatchStepResult { per_request: Vec<(request_id, next_token_id, finish_reason)> }`.
- Worker subprocess holds a `SlotTable` of up to `max_concurrent_decode_batch` slots. Each slot owns a `KvCacheEntry` and a `SamplingState`. Adding a request: prefill its prompt (optionally with prefix-cache from Item 5), allocate a slot, seed with sampled first token. Removing: on `finish_reason`, free slot.
- Per step: worker does a single `forward_batch` over all active slots' last tokens stacked into `[N, 1, hidden]`. Per-slot sampling locally (CPU cheap). Return `N` tokens in one response.
- Scheduler in `process_pool.rs`:
  - Replace `Mutex<socket>` with an actor task owning the socket.
  - Actor maintains per-worker `VecDeque<PendingRequest>` for new requests + `HashMap<RequestId, StreamingTokenTx>` for active requests.
  - Every loop iteration: try to admit new requests (up to `max_concurrent_decode_batch`), send `BatchGenerateStep` with all active slots, forward results to each request's `StreamingTokenTx`.
  - Sarathi-style chunked prefill (optional v2): when a new request needs prefill, chunk it into `prefill_chunk_tokens` and interleave chunks with ongoing decode steps instead of stalling decode.

**Eligibility.**
- Only for `handle_generate` / `pool.generate()` callers. Per-token `handle_forward` path stays single-request (Item 3 Phase 1 already in place, sufficient).
- Gated by `continuous_batching` config (existing flag, already plumbed through to InferenceRouter).

**Files.**
- `src/inference/worker_ipc.rs` — new `DaemonMsg::BatchGenerateStep`, `WorkerMsg::BatchStepResult` variants.
- `src/inference/model_worker.rs` — new `handle_batch_generate_step` with `SlotTable`; `SplitModel::forward_batch(inputs: &[Tensor], positions: &[usize])` — stacks along batch dim, runs one forward, unstacks.
- `src/inference/split/executor.rs` — `forward_batch` method.
- `src/inference/split/kv_cache.rs` — multi-request coexistence already works (per-request keys); just need to document that `forward_batch` reads/writes N separate entries per step.
- `src/inference/process_pool.rs` — replace `socket: Mutex<...>` on `WorkerHandle` with `requests_tx: mpsc::Sender<GenerateRequest>` + spawn actor task per worker. Actor owns the socket, drives the batched step loop.
- `src/config.rs` — already has `continuous_batching`, `max_concurrent_decode_batch`, `batch_collection_ms`. Add `prefill_chunk_tokens: u32` (default 512) for future Sarathi chunking.

**Success metric.** Two concurrent requests against the same model: aggregate tok/s ≥ 1.7× single-request baseline.

**Build order.**
1. Add `forward_batch` to `SplitModel` (stack → forward → unstack). Unit test against sequential forwards for identical outputs.
2. Add `BatchGenerateStep` / `BatchStepResult` wire types.
3. Implement worker-side `SlotTable` + `handle_batch_generate_step`.
4. Refactor `process_pool.rs` to actor model.
5. Integration test: 2 concurrent `pool.generate()` calls, assert correct outputs + aggregate throughput.

### Phase 1 (LANDED 2026-04-19): worker-side SlotTable + slot-driven decode loop

Scope cut vs. original design: **no new IPC verbs.** The `Generate` /
`Token` / `GenerateDone` / `Error` messages already multiplex per-request
through Phase 2a's reader actor in `process_pool.rs`, so the slot table
lives entirely inside the worker subprocess and the daemon side is
unchanged. Concurrent `pool.generate()` calls fan into the worker via
the existing multiplexed write path; the worker now interleaves their
decode steps instead of running them serially.

- **`src/inference/slot_table.rs`** (new, ~250 LOC). `Slot { request_id,
  req_id_str, model_key, layer_range, index_pos, last_token,
  last_token_logprob, generated_count, max_tokens, use_logprobs, eos,
  stop_sequences, accumulated_text, sampling, prompt_tokens,
  finish_reason }` + `SlotTable { slots, layer_range, capacity }`. The
  table pins a single `(layer_start, layer_end)` while non-empty —
  admits with a different range fall through to sequential
  `handle_generate`. 6 unit tests cover admission gating, capacity
  bound, layer-range pinning + release on drain, finish-reason
  one-shot semantics.
- **`src/inference/model_worker.rs::run_worker`** restructured into a
  `tokio::select!` loop over (a) `mpsc::Receiver<DaemonMsg>` fed by a
  spawned reader task — keeps `recv_framed` cancel-safe — and (b)
  `tokio::task::yield_now` whenever the slot table is non-empty. Every
  `Generate` first attempts admission via `try_admit_generate_slot`
  (prefill, prefix-cache lookup, sample first token, push slot). Slots
  ineligible for batching (SWIFT-active, max_tokens=0, layer_range
  mismatch, table full) fall through to the existing
  `handle_generate` path with the IpcGenerate restored.
- **`step_decode_pool`** runs once per tick when slots are active.
  Phase 1 emits `Token(last_token)` per slot (after EOS / stop-string
  gate); Phase 2 builds `BatchItem`s from the still-active subset and
  calls `SplitModel::forward_batch` (already CPU-falls-through, GPU
  fused); Phase 3 samples per-slot logits, advances `index_pos`, marks
  slots that hit `max_tokens` with `finish_reason=length`, and emits the
  off-by-one final Token inline. `finalize_slot` then sends
  `GenerateDone` and clears the per-request KV. Mirrors the per-token
  semantics of `handle_generate` byte-for-byte (including the off-by-one
  emit for `finish="length"` and the no-emit-on-EOS-first-sample case).
- **`process_pool.rs`** now passes `--batch-generate <bool>` and
  `--batch-generate-max-slots <u32>` to spawned workers. Driven by the
  same `continuous_batching` + `max_concurrent_decode_batch` atomics
  that gate the daemon-side `forward()` coalescer (Item 3 Phase 2b).
  Restart of an existing worker is needed to pick up flag changes —
  matches the prefix-cache / SWIFT settings, which behave the same way.
- **`src/main.rs`** ModelWorker subcommand grew `--batch-generate` and
  `--batch-generate-max-slots` flags, plumbed into `run_worker`.

Default-off for now: the daemon's `continuous_batching` flag controls
both the daemon-side scheduler (default-on as of 2026-04-19) and the
new worker-side slot loop. Leaving today's default-on continues to be
safe — workers that aren't passed `--batch-generate true` still
execute the legacy serial `handle_generate` path; workers that ARE
passed `true` admit eligible Generates into the slot table, fall
through cleanly when not eligible, and never block the runtime
between IPC reads and decode ticks (the select! arms guarantee one
yield between every tick).

735 lib + integration tests pass on `dev,claude-subscription`. End-to-
end concurrent-generate throughput benchmark + Sarathi-style chunked
prefill are Phase 2 work — gated until we can spin up two real
`pool.generate()` callers against a model worker (existing
`tests/integration` doesn't cover the pool because `current_exe()` in
test context isn't the swarmllm binary).

### Phase 2 (LANDED 2026-04-19): Sarathi-style chunked prefill

Slots now register in a `Prefilling { remaining_ids, next_chunk_index_pos }`
state at admit time and only sample their first decode token after the
final prefill chunk runs. Each tick:

1. **Phase A** — every `Prefilling` slot advances by up to
   `prefill_chunk_tokens` prompt tokens (default 128, configurable via
   `InferenceConfig::prefill_chunk_tokens`). When a slot's final chunk
   runs, it samples the first decode token, snapshots the completed
   prompt KV into the prefix cache, and transitions to
   `Decoding { last_token, last_token_logprob, generated_count, index_pos }`
   — joining the same tick's batched decode (Phase B).
2. **Phase B** — unchanged from Phase 1: per-slot EOS / stop-string
   gate, Token emit, batched `forward_batch` over all `Decoding` slots,
   per-slot sampling, off-by-one Token on `length` finish.

**Why it matters.** Phase 1 admit ran the entire prefill inside the
admit handler — a long-prompt admission (think 4 KB system prompt)
stalled every already-active decode slot for the full prefill duration
(seconds on TinyLlama CPU). Phase 2 bounds that interruption to
`prefill_chunk_tokens` of compute per tick, so a long admission costs
each in-flight decode at most ~one extra chunk of latency before the
next decode token streams.

**Slot state machine** (`src/inference/slot_table.rs`):

```rust
pub enum SlotState {
    Prefilling {
        remaining_ids: Vec<u32>,
        next_chunk_index_pos: usize,
    },
    Decoding {
        last_token: u32,
        last_token_logprob: Option<f32>,
        generated_count: usize,
        index_pos: usize,
    },
}
```

Helpers added: `take_prefill_chunk(chunk_size)`,
`promote_to_decoding(first_token, first_logprob)`, `is_prefilling()`,
`is_decoding()`. 6 new unit tests cover chunk math (drain/cap/zero-size/
prefix-cache-hit), `Decoding`-state rejection of `take_prefill_chunk`,
and `Prefilling → Decoding` transition. Total slot_table tests: 12.

**Files touched.**
- `src/inference/slot_table.rs` — state-machine refactor + `prompt_ids`
  field carried for prefix-cache snapshot at prefill completion.
- `src/inference/model_worker.rs` — `try_register_generate_slot`
  replaces the prefill-in-admit `try_admit_generate_slot`. New
  `step_decode_pool` two-phase tick. `finalize_slot` reads
  `generated_count` via `Slot::generated_count()` accessor (which
  returns 0 for `Prefilling` and the Decoding counter otherwise).
- `src/config.rs` — new `InferenceConfig::prefill_chunk_tokens` (u32,
  default 128). 0/1 degenerate to one-token-per-tick prefill.
- `src/inference/process_pool.rs` — `prefill_chunk_tokens` AtomicU32 +
  `set_prefill_chunk_tokens(u32)` setter; spawns workers with
  `--prefill-chunk-tokens`.
- `src/main.rs` — ModelWorker subcommand grew the new CLI flag.
- `src/daemon/state/mod.rs` — applies the config value to the pool at
  startup.

**Single-user perf.** Single Generate, no contention: tick 1 prefills
the whole prompt as one chunk (when `chunk_size >= prompt_len`) and
samples first token; tick 2 emits first token + decodes second; etc.
Same total compute as Phase 1, just shifted by one tick. For prompts
longer than `chunk_size`, single-user latency to first token grows
linearly with chunk count — operators can raise `prefill_chunk_tokens`
if running mostly single-user workloads.

**Multi-user perf.** Decode interruption bounded by chunk size. With
`prefill_chunk_tokens=128` on TinyLlama Q4 CPU (~1 ms/token in
prefill mode), an admit during active decode adds ~128 ms tick latency
to active streams instead of seconds — essentially the design goal.

741 lib + integration tests pass on `dev,claude-subscription` (+6 net
from 6 new SlotTable prefill state tests).

### Phase 2 follow-up (LANDED 2026-04-19): per-slot error containment + DIAG tracing + bench recipe

- **Per-slot error containment.** Phase 1+2 originally bubbled any
  forward / sample error out of `step_decode_pool`, which the outer
  loop then turned into a "BatchGenerate decode failed" Error for every
  in-flight slot. Now `Slot` carries an optional `error_message`,
  `Slot::finish_error(msg)` records it (first-write-wins like
  `finish_stop`/`finish_length`), and `step_decode_pool` catches per-slot
  errors at every hot point — Phase A `tensor_from_ids`, Phase A chunk
  forward, Phase A first-token sample, Phase B `token_tensor`, Phase B
  per-slot sample. `finalize_slot` routes errored slots to
  `WorkerMsg::Error`. One bad slot can no longer take down its
  neighbors.
- **DIAG tracing.** Worker logs at debug:
  `BatchGenerate slot registered`, `BatchGenerate prefill chunk ran`,
  plus a `slot errored` line on each containment branch. Discoverable
  via `grep "DIAG: BatchGenerate"`.
- **Benchmark recipe.** New `docs/plans/benchmarks/round4.md` documents
  the `swarmllm bench --concurrency N` workflow + a manual streaming
  test for the Phase 2 mixed-load scenario (long admit during active
  decode). Sidesteps the `current_exe()` integration-test gotcha and
  uses real HTTP to a live daemon.

742 lib + integration tests pass on `dev,claude-subscription` (+1 net
from the new `finish_error_records_message_and_blocks_other_finishers`
slot_table test).

### Phase 3 (LANDED 2026-04-19): bench measured + 3-iteration confirm

Ran the bench recipe end-to-end on the user's RTX 3070 setup, with a
3-iteration confirmation run because single-iteration TTFT is noisy.
Results in `docs/plans/benchmarks/round4.md`. Headline (3-run avg):

| Concurrency | Aggregate tok/s ON | Aggregate tok/s OFF | Avg TTFT ON | Avg TTFT OFF | TTFT speedup |
|---|---|---|---|---|---|
| 2 | 39.6 | 40.5 | 74 ms | 1347 ms | **18.2×** |
| 4 | 42.1 | 42.5 | 169 ms | 3673 ms | **21.7×** |
| 8 | 45.3 | 39.6 | 377 ms | 8859 ms | **23.5×** (single iter) |

Per-run variance is <10% on both columns — the 18–24× ratio is real, not
single-iteration luck. Aggregate throughput is equivalent (TinyLlama is
too small for the fused `forward_batch` matmul win at this batch size
on this GPU). The real-world impact is TTFT fairness under concurrency —
concurrent users get their first token in tens of ms instead of seconds.
Single-user is essentially unchanged (39.4 vs 40.1 tok/s, within noise).

The bench tool also gained `--stream` + `--model-id` flags this session.
Streaming mode parses SSE chunks and reports per-request TTFT; the
`--model-id` override avoids picking the wrong model when multiple are
registered (the auto-pick was hitting OOM on Qwen-7B).

### Phase 4 (LANDED 2026-04-19): batched chunked prefill

`SplitModel::forward_batch` now accepts any *homogeneous* batch —
either every item has `seq_len = 1` (the original decode path) or every
item has the **same** `seq_len > 1` and the **same** `index_pos` (the
new prefill-chunk path). When homogeneous with `seq_len > 1`, one
causal mask is built once from the first slot's KV length (identical
across slots by construction) and passed to each per-request
`forward_attn` / `forward_mla` call. FFN and norms already benefited
from the existing batched path. For `is_last` segments the output head
now slices `i((.., seq_len - 1, ..))` instead of the hardcoded
`i((.., 0, ..))`. Heterogeneous batches (mixed seq_lens or differing
`index_pos` at seq_len > 1) transparently fall back to sequential
forwards — no behavior change for callers.

`step_decode_pool`'s Phase A is a four-stage loop now:

1. **Collect** — every `Prefilling` slot's `take_prefill_chunk` plus a
   tensor build. Tensor build errors mark only that slot.
2. **Group** by `(chunk_len, index_pos)` into a `BTreeMap` for
   deterministic ordering.
3. **Forward** — singletons go through `model.forward`; groups of ≥2
   through `model.forward_batch`. A fused-forward failure errors every
   slot in that group (strict retry-sequential isn't worth the
   complexity — catastrophic forward failures would repeat
   per-request anyway).
4. **Finalize** per step — DIAG trace + first-token sample +
   prefix-cache insert + promote-to-decoding when `remaining_after ==
   0`.

Eight concurrent admits with the same chunk size now collapse into a
single `forward_batch` of shape `[8, chunk_size]`, replacing eight
sequential `[1, chunk_size]` forwards. The TTFT tightening only helps
when multiple admits land in the same tick with same-shape chunks —
single-user workloads stay on the singleton path with no overhead.

**Tests** (`src/inference/split/tests.rs`): 3 new, all green on CPU.

- `forward_batch_prefill_chunks_match_sequential` — batched vs.
  sequential forwards agree within `max_diff < 1e-4` at `seq_len=8,
  index_pos=0` across 2 requests.
- `forward_batch_mixed_seq_len_falls_back` — one decode + one prefill
  item stays correct and returns per-item shapes unchanged.
- `forward_batch_mixed_index_pos_falls_back` — two prefill items at
  differing `index_pos` stay correct via the sequential fallback.

Totals: 745 tests pass on `dev,claude-subscription` (+3 net, no
regressions).

**Not benchmarked end-to-end.** TTFT under concurrent same-prompt-size
admits should drop in rough proportion to group size (8 fused vs 8
sequential ≈ one `forward_batch` wall time vs eight `forward` wall
times). On TinyLlama + RTX 3070, single-user prefill is already fast
enough that the measurable win requires burst-admit of same-chunk-size
prompts; open item for round 5 benchmarks.

### Phase 4 synthetic bench (LANDED 2026-04-19)

`src/inference/split/tests.rs::forward_prefill_batch_timing` — ignored
timing test that compares `forward_batch` vs N sequential `forward`
calls across chunk sizes {32, 64, 128} × batch sizes {2, 4, 8} on a
22-layer / 1024-hidden test model. CPU debug: 1.11–1.63× speedup
depending on (chunk, batch). GPU release (RTX 3070): mostly wash at
this small hidden_dim, maxing at 1.13× at (128, 8). Numbers in
`docs/plans/benchmarks/round5.md`.

### Phase 4 E2E bench + admit-coalescing fix (LANDED 2026-04-19)

**The fix.** End-to-end measurement surfaced that Phase 4's batching
never fired under real HTTP-paced traffic. Root cause: the worker's
`tokio::select!` strictly interleaved admit → tick → admit → tick, so
even concurrent admits ended up at different `index_pos` by the time
they were in the slot table simultaneously. Fix: after handling any
mpsc message, drain up to 16 further messages via `try_recv()` *before*
running the next decode tick (see `src/inference/model_worker.rs`'s
`handle_daemon_msg` helper). The refactor extracted the message-handling
match into a free async function so it can be reused inside the drain
loop.

**Measured (RTX 3070, TinyLlama-1.1B Q4_K_M, same ~100-token prompt
across all concurrent requests):**

| Mode | Concurrency | Aggregate tok/s | TTFT min / avg / max (ms) | DIAG `chunk fused` |
|---|---|---|---|---|
| Before fix | 4 | 31.2 | 52 / 235 / 447 | 0 |
| After fix  | 4 | **49.1** (+57%) | **180 / 180 / 180** | 1 (batch_size=4) |
| After fix  | 8 | **42.4** | **321 / 321 / 321** | 1 (batch_size=8) |

TTFT uniformity (all concurrent requests served in the same ms after
the fix) is the headline. Aggregate tok/s uplift reflects fused-FFN
amortization on the prefill tick. Larger-model numbers (Phi-3.5-mini,
Qwen-7B) are pending — those OOM on the 8 GB RTX 3070 under the
current default `max_seq_len_override=8192` × 8-slot KV reservation.
See `docs/plans/benchmarks/round5.md` for the full reproducing recipe.

---

## Cross-item sequencing (Round 2)

**Order:** 5 → 6 → 7.

- **Item 5 (prefix cache)** is foundational and self-contained. Biggest single TTFT win. Doesn't touch async/scheduler surface.
- **Item 6 (SWIFT)** is a decode-loop mod inside `handle_generate`. Builds on top of Item 5 without conflict.
- **Item 7 (BatchGenerate)** is the largest refactor (actor model + SlotTable). Builds on Items 5 & 6 because both must work within a batched decode step.

## Deferred to future sessions

- **Item 8 — Cross-node prefix cache sharing** 🟡 PHASE 1 LANDED 2026-04-19. Announce BLAKE3 prompt-prefix hashes over gossip; peers that already prefilled a shared prefix serve KV blocks on demand. Content-addressed KV shards — fits our existing shard announcement infrastructure. Potentially novel for P2P.
  - **Phase 1 (this commit, 2026-04-19).** Hash announcement + cross-node index + loopback verification.
    - Chained BLAKE3 hashing in `src/inference/split/prefix_cache.rs`: `compute_block_hashes(tokens, block_size)` returns `Vec<PrefixBlockEntry { block_hash: [u8;32], token_count: u32 }>`. Hash chain: `h[0] = blake3(u32_le(tokens[0..B]))`, `h[i] = blake3(h[i-1] || u32_le(tokens[i*B..(i+1)*B]))`. Two prompts that share the first `k` blocks under the same `block_size` produce identical `block_hash[0..k]`.
    - `PrefixCache::insert_from_kv` returns the deduped post-insert manifest for the model so the caller can broadcast it. `enumerate_manifest(model_key)` exposes the same union for re-announce on demand. Trailing partial blocks are intentionally NOT hashed so cross-peer block boundaries align.
    - New IPC verb `WorkerMsg::PrefixManifestUpdate { model_id, blocks }` emitted by the worker after every prefix-cache insert (both `handle_generate` and the BatchGenerate Phase A path). Reader actor in `process_pool.rs` intercepts and routes through a daemon-installed `prefix_manifest_tx` channel using `try_send` so a slow daemon never backpressures the IPC reader.
    - New gossip variant `SwarmMessage::PrefixCacheAnnounce(PrefixCacheAnnounce)` with `node_id`, `model_id`, `blocks: Vec<PrefixBlockEntry>`, `timestamp`. Dispatch handler in `src/daemon/dispatch/mod.rs` validates the authenticated sender against `announce.node_id`, drops self-announces, caps blocks per announce at 1024 (memory DoS guard), then calls `state.models.replace_peer_prefix_blocks` to update the index.
    - State additions in `state.models`: `cross_node_prefix_index: DashMap<ModelId, DashMap<[u8;32], DashSet<NodeId>>>` and `peer_prefix_blocks: DashMap<NodeId, DashMap<ModelId, DashSet<[u8;32]>>>` (reverse index for O(1) cleanup). Helpers: `replace_peer_prefix_blocks` (diff-replace semantics, keeps other peers' entries), `forget_peer_prefix_blocks` (peer-departure cleanup), `cross_node_prefix_holders` (Phase 2 lookup hook).
    - `spawn_prefix_announce_forwarder` in `daemon/background.rs` drains the worker channel, broadcasts gossip, AND records our own blocks under our `node_id` in the index — that way a single-node loopback test can call `cross_node_prefix_holders(model, hash)` and observe the full wire path without needing a peer.
    - `RebalanceEvent::PeerLeft` now calls `forget_peer_prefix_blocks` so Phase 2's KV-fetch path will never dial a departed peer.
    - Tests: 6 prefix_cache (chain math, partial-block handling, zero-size guard, insert returns manifest, enumerate dedups, enumerate-zero is empty) + 5 cross-node index (replace, supersede, multi-peer, forget-only-their-entries, empty-announce) = 11 net new. **756 lib + integration on `dev,claude-subscription`** (up from 745).
  - **Phase 2a (this commit, 2026-04-19).** Wire format + daemon-side fetch infrastructure.
    - `KvSnapshot` binary wire format with magic `SKVX`, version byte, JSON header (`token_count`, `tokens`, `dim`, `max_seq_len`, per-layer `(k_shape, v_shape, k_count)`), + concatenated per-layer `[K_f32_le | V_f32_le]`. All tensors cast to f32 on wire for device-independence (CPU↔GPU peers). `serialize_snapshot` + `deserialize_snapshot` + `verify_token_hash_chain` helpers exposed from `src/inference/split/prefix_cache.rs`.
    - Codec extensions: new `SwarmRequest::PrefixKvFetch(PrefixKvFetchReq)` (JSON — small req) + `SwarmResponse::PrefixKvData(PrefixKvDataResp)` (new binary `WIRE_TAG_PREFIX_KV = 0x04` frame: `[tag][4B len_be][16B uuid][1B flag][snapshot bytes]`). Hits existing `MAX_MESSAGE_SIZE=256MB` limit so multi-MB KV payloads fit; zstd compression via the existing compressed-tensor path is a future optimization.
    - `NetworkCommand::SendPrefixKvFetch { peer, request_id, model_id, block_hash }` wired end-to-end. Daemon installs a oneshot on `SharedState::pending_prefix_kv_fetches` keyed by `request_id`; NetworkManager maps libp2p `OutboundRequestId → request_id` via `pending_prefix_kv_outbound` and resolves the caller's oneshot on `SwarmResponse::PrefixKvData` (or on `OutboundFailure` with `None`).
    - Inbound-request stub: the serving-side handler replies `SwarmResponse::PrefixKvData { payload: None }` for every `PrefixKvFetch` arrival. Phase 2b replaces this with a daemon→worker IPC round-trip that extracts the actual snapshot from the local `PrefixCache`.
    - `SharedState::best_cross_node_prefix_match` walks the chained-hash manifest longest-first, picks the lowest-latency non-self peer, returns `(peer, block_hash, token_count)`. Self-entries recorded by Phase 1's loopback forwarder are skipped (local hits are served by the in-process `PrefixCache` directly).
    - `SharedState::try_fetch_cross_node_prefix` one-call helper: lookup → dispatch → await (with timeout) → BLAKE3-verify the returned tokens match the requested block hash → prefix-check against our prompt (belt + braces) → deserialize the snapshot onto the caller's device. Returns `Ok(None)` for any failure path so the caller unconditionally falls through to normal prefill. RAII guard cleans up the `pending_prefix_kv_fetches` entry on cancellation.
    - Tests: +7 (hash-chain verify success + wrong-tokens + non-boundary rejection, snapshot roundtrip preserves tensors, bad magic + bad version + None-layer preservation). **763 lib + integration on `dev,claude-subscription`** (up from 756).
  - **Phase 2b (next session).** Admit-time integration. Add `WorkerMsg::PrefixFetchProbe { request_id, model_id, block_hashes }` + `DaemonMsg::PrefixFetchResult { request_id, payload }` IPC so `handle_generate` can ask the daemon to fetch + hydrate from a remote peer after a local prefix-cache miss. Add `DaemonMsg::ExportPrefixSnapshot` + `WorkerMsg::PrefixSnapshotResponse` for the serving side so the inbound-request handler can pull the cached snapshot out of the worker process. Wire into `try_register_generate_slot` as well as the singleton `handle_generate` path. Two-node loopback integration test.
  - **Phase 3 (later).** Trust-gated re-verification: untrusted peers' KV gets re-verified by re-running a single forward at the boundary; high-trust peers skip re-verification. Wire to `state.credits.trust_manager`.
  - **Phase 4.** Bench against Item 5 baseline + multi-node setup + plan/memory updates.
- **Item 9 — EAGLE-3 draft heads.** Per-target pretrained draft heads (SafeAILab HF) distributed as a new shard type. 3–6× ceiling, highest complexity.
- **Item 10 — FlowSpec / PPSD pipelined speculation.** Addresses the multi-segment pipeline case Item 4 didn't cover. Requires Item 1 stream to be default-on.
- **Item 11 — Lookahead decoding.** No-draft-model Jacobi iteration. Test on CPU to see if FLOP overhead beats the speedup.

---

# Round 3 (2026-04-18) — High-leverage P2P-specific research candidates

> **Context:** Research scan after Items 5+6 landed surfaced several techniques specifically designed for P2P / pipeline-parallel deployments, distinct from the single-GPU continuous-batching wins of Item 7. Listed in priority order with arxiv links and fit assessment for SwarmLLM (libp2p + candle + sharded peers).

### Item 12 — DSD: Decentralized Speculative Decoding 🟡 PHASE 1 LANDED 2026-04-18

**Source.** [arxiv 2511.11733](https://arxiv.org/abs/2511.11733) (Song et al., Gradient Network, Nov 2025) — primary DSD paper. [arxiv 2511.21669](https://arxiv.org/abs/2511.21669) — edge-cloud variant with the Adaptive Window Control (AWC) γ controller. Note: Parallax repo (`GradientHQ/parallax`) does NOT contain the DSD implementation in `main` as of audit (paper-only).

**Why it fits SwarmLLM.** The `T_DSD = γ·t0 + (N-1)·t1` vs `T_std = γ·(t0 + (N-1)·t1)` analysis says DSD's win grows with the `(N-1)·t1` term — i.e. with WAN RTT and pipeline depth. Paper 1's "most pronounced" regime is `3·t0 < t1 < 10·t0`, matching SwarmLLM's typical 50–150 ms WAN RTT vs 10–100 ms candle-CPU/GPU per-segment compute. Expected gains LARGER than the paper's 2.56× InfiniBand baseline, because their RTT was tens-of-µs and ours is tens-of-ms.

**Eligibility (when implemented).** All of: `multi-segment distributed pipeline (N ≥ 2)`, `draft model loaded on coordinator`, `greedy temp=0` for v1, no vision/LoRA/encrypted-pipeline. Single-segment workloads keep using the Item 4 fast path.

### Phase 1 (LANDED 2026-04-18, this commit)

Worker-side groundwork: `src/inference/model_worker.rs` first-segment decode branch now accepts γ token IDs (`γ × 8` bytes LE) instead of just one. Output tensor is `[1, γ]`; candle's transformer forward writes KV at positions `[index_pos..index_pos+γ]` automatically because every layer is shape-polymorphic in seq_len. Single-token decode (γ=1, the legacy default) continues to work identically — validated as a strict no-op refactor.

Config flag `InferenceConfig::decentralized_spec_decoding` (default `false`) added. Has no effect today; will gate the coordinator loop in Phase 4.

### Phase 2 (LANDED 2026-04-18): KV truncation primitives verified

5 unit tests in `src/inference/split/tests.rs` exercise `KvCacheStore::truncate_request_to` end-to-end:
- `kv_truncate_to_preserves_prefix_and_drops_suffix` — narrow/contiguous/reset/append round-trip preserves first N positions exactly
- `kv_truncate_to_target_geq_current_is_noop` — asking for more positions than exist doesn't corrupt
- `kv_truncate_unallocated_layer_is_skipped` — `None` layers are ignored, not panicked-on
- `kv_truncate_missing_request_is_noop` — silent no-op for unknown request_id
- `kv_truncate_all_layers_aligned` — all layers in a multi-layer entry end at the same target_len after one call

Worker-side already applies `LayerForward.truncate_kv_to` uniformly on every `handle_forward` regardless of segment position (verified at `model_worker.rs:473`). The coordinator-side responsibility — setting `truncate_kv_to` on every segment's LayerForward, not just the last — is Phase 4 work.

### Phase 3 (LANDED 2026-04-18): adaptive γ controller

`src/inference/dsd_controller.rs` — paper 2's "Dynamic window" baseline:

```
accept_ema  ← α · accept_ema + (1 − α) · accept_rate_this_round
γ_next      ← clamp(γ · (1 + β · (accept_ema − 0.5)), 2, 12)
```

Defaults `α=0.7`, `β=0.2`. 6 unit tests cover: initial-γ clamping, perfect-acceptance ratchet to max, zero-acceptance ratchet to min, EMA smoothing of alternating signal, high-α resistance to single bad rounds, zero-`proposed` safety. Per-request state (small `GammaController` struct). Will be wired into Phase 4's coordinator loop. Trained MLP variant is a future optimization.

### Phase 4 part 1 (LANDED 2026-04-18): worker speculative branch decoupling

Refactored `model_worker::handle_forward` to split the Item 2 single-segment
`speculative_verify` flag into two orthogonal concerns:
1. **Input** goes through the standard branches (Phase 1 multi-token first-segment decode for `is_first`, generic `bytes_to_tensor` for `!is_first`). The `is_first && is_last` precondition is gone.
2. **Output emission** is gated on `is_last && spec_logits_requested`. When set, the worker calls either `SplitModel::forward_verify_all_positions` (single-segment, raw-token input) or the new `forward_verify_all_positions_pre_embedded` (multi-segment, hidden-state input) and emits γ+1 logit vectors via `LayerResult.spec_logits`.

Item 2's `send_verify_batch` updated to encode all γ tokens in `activations` (γ × 8 bytes LE) instead of just the first token, matching the unified worker input path. All existing tests pass — the refactor is non-breaking for the single-segment Item 2 case.

Intermediate segments now correctly pass `[1, γ, hidden]` through their layer range without taking the verify branch, and the last segment of a multi-segment DSD pipeline can run verify on hidden-state input.

### Phase 4 part 2 (LANDED 2026-04-18): coordinator loop

`src/inference/pipeline/dsd.rs` (≈410 LOC) — feature-gated to `llama` for the local draft model. Key pieces:

- `eligible(exec)` — gates on `decentralized_spec_decoding && speculative_decoding`, multi-segment (`≥2`), no TP, greedy temp=0, draft loaded, no vision/LoRA/encryption, all segments remote.
- `try_dsd_distributed` (entry point dispatched FIRST in `execute_distributed`, before Item 2's single-segment spec):
  1. Resolve all peer IDs upfront (fall through cleanly if any segment's peer is unknown)
  2. Acquire→drop draft lock briefly to verify it's loaded (avoids overlapping borrow with the mutable-self prefill forward)
  3. Phase 1: standard `forward_through_segments` for prefill — reuses existing path, primes every segment's KV with the prompt, gets the first token
  4. Phase 2: re-acquire draft lock, `draft_prefill` (reusing Item 2's helper)
  5. Spec round loop: `controller.current_gamma()` → `draft_next_gamma` → multi-segment verify → greedy accept-reject → emit `[accepted..bonus]` → `controller.record_round(accepted, gamma)` → `draft_sync_after_round` → set `pending_truncate = Some(expected_kv_len)` for the next round
- `forward_verify_through_segments` (private to dsd.rs) — propagates `LayerForward { draft_tokens, spec_logits_requested=true, truncate_kv_to=pending }` through every segment in order: first segment gets γ+1 token IDs as bytes, intermediate segments get `[1, γ, hidden]` from the previous segment's `LayerResult.activations`, last segment returns γ+1 logit vectors via `spec_logits`. Each segment's worker applies `truncate_kv_to` independently before its forward.
- Visibility opens: `argmax`, `draft_prefill`, `draft_next_gamma`, `draft_sync_after_round` in `speculative.rs` are now `pub(super)` so dsd.rs can reuse them; `forward_through_segments` in distributed.rs is `pub(super)` for the prefill call.

Default off (`decentralized_spec_decoding = false`). Builds in both `dev` and `dev,llama` configs. All 698 tests still pass — no regressions to the existing standard distributed or Item 2 single-segment spec paths. Multi-segment loopback benchmark + correctness validation (greedy output equivalence) is the remaining empirical work; CPU-only loopback isn't a great signal because Item 4 fast path bypasses single-segment so the wins only show on real multi-segment WAN topologies.

### Future / out of v1 scope

- Paper 1 semantic key-token detection (entropy ratio + top-1 gap + NormMatch) for τ relaxation — only matters for non-greedy sampling
- Paper 2's trained MLP γ predictor — replace simple controller after measuring
- Activation compression (Item 13) on the γ-batched payload — already works because Q8_0 is shape-agnostic; no extra integration

### Stacks with

- **Item 4** stays as-is for single-segment fast path. DSD only kicks in when `N ≥ 2`.
- **Item 5** prefix cache: orthogonal — cache hit reduces prefill, DSD reduces decode.
- **Item 13** Q8_0: stacks. The γ-wide hidden state goes through the same compressed wire.
- **Item 2 single-node spec**: kept as the local-segment alternative when no remote pipeline exists.

### Complexity assessment (revised post-research)

Medium-large. Phase 1 is small (this commit). Phase 4 is the main coordinator loop refactor. Phase 2 needs a multi-node integration test. Phase 3 is trivial (heuristic).

### Item 13 — Activation compression for inter-node transfer ✅ LANDED 2026-04-18 (flag, codec verified)

**Source.** [arxiv 2411.09510](https://arxiv.org/html/2411.09510v3) (Hansen-Palmus et al., NVIDIA, v3 Jan 2026 — "Communication Compression for Tensor Parallel LLM Inference"); reference precedent llama.cpp Q8_0 wire format.

**What landed.** Q8_0 (group-32 symmetric quantization, llama.cpp-compatible block layout: `f16 scale + 32 i8 values = 34 B per group`) for intermediate-segment hidden state activations between pipeline peers. Compresses ~3.76× vs raw f32 with measured RMS error <0.005 on synthetic activation slices and <1e-4 between blocks even when one block contains 100× outliers (per-block scale isolates them — see `inference::quant::tests::outlier_block_isolated`).

**Files.**
- `src/inference/quant.rs` (new) — `quantize_q8_0` / `dequantize_q8_0` / `q8_0_byte_len`. 6 unit tests covering exact-block-size, partial-trailing-block, all-zeros, outlier isolation, malformed input, and 4096-element typical-hidden-state quality + compression ratio.
- `src/inference/tensor_util.rs` — added `tensor_to_bytes_q8_0` + dtype-tag dispatch in `bytes_to_tensor` (tag `0` = legacy raw f32, tag `1` = Q8_0). Receivers auto-dispatch — peers without the flag still decode quantized inputs correctly.
- `src/inference/model_worker.rs` — `handle_forward` takes a new `activation_compression: bool`; intermediate-segment output uses `tensor_to_bytes_q8_0` when the flag is on. Final-segment output (token IDs) is unaffected.
- `src/inference/process_pool.rs` — `set_activation_compression`, atomic flag, `--activation-compression <bool>` arg passed to spawned worker subprocesses.
- `src/main.rs` — new `model-worker --activation-compression` CLI flag.
- `src/config.rs` — `InferenceConfig::activation_compression: bool` (default `false`).
- `src/daemon/state/mod.rs` — applies the config flag to the pool at startup.

**Why dtype-tag inside the tensor envelope (rather than `LayerForward.format`).** The `tensor_to_bytes` envelope already carries a 4-byte dtype tag at a stable offset. Extending it keeps the wire compatible with peers that don't run quantization — they just see a different tag and dispatch in `bytes_to_tensor`. Adding `TensorFormat::Q8_0` at the LayerForward level would require coordinating decoder branches across `layer_forward.rs`, `encrypted.rs`, `pipeline_stream.rs`, and `manager/mod.rs`, with no functional benefit since the actual byte layout of `LayerForward.activations` is what changes.

**Eligibility.** Always safe to enable; receivers handle either tag. The fast-path single-segment case (Item 4) bypasses hidden state transfer entirely so this only helps when the pipeline spans 2+ peers (the actual pain point for compression). TP `AllReduce` payloads still use the legacy raw f32 path because their codec is independent (handled in `daemon/dispatch/layer_forward.rs` via `tensor_to_raw_f32`) — extending TP would be a separate increment.

**Reported (paper).** 3.5–4.5× activation-size reduction → 1.2–2× TTFT on slow links. Q8_0 specifically falls between the FP4 and FP5 results in the paper's Table 2 (PPL drift well under 1%); SmoothQuant W8A8 confirms <0.5% PPL on standard activations.

**Stacks with.** Every other item — orthogonal. Especially impactful for SwarmLLM's heterogeneous links (LAN 1Gbps to WAN 10–50 Mbps).

**Remaining work (future).**
- End-to-end multi-segment benchmark with the flag on (TinyLlama is single-segment in practice; need 2+ shard ranges across 2+ peers to measure WAN-style improvement).
- Link-aware auto-toggle: enable per-segment based on measured peer RTT/bandwidth from `peer_registry`. Cheapest implementation is a per-link bool on the segment plan rather than a global config.
- Extend to TP AllReduce payloads (separate codec path).
- Optional Q4_0 variant for very-slow links (~7× compression at modest quality cost).

### Item 14 — Mirror Speculative Decoding (Apple, 2025)

**What.** Bidirectional concurrency: draft speculates forward, target *simultaneously* speculates correction paths for the draft. Two parallel pipelines hide inter-peer RTT. Stacks naturally with SWIFT (Item 6) which provides the early-exit signal Mirror needs.

**Source.** [arxiv 2510.13161](https://arxiv.org/abs/2510.13161), [Apple ML blog](https://machinelearning.apple.com/research/mirror).

**Reported.** 2.8–5.8× wall-time on 14B–66B; 30% over EAGLE-3.

**Complexity.** Large.

### Item 15 — SpecPipe / PipeDec (pipeline-stage-granular speculation)

**What.** Draft tokens fed into the pipeline at *each stage* so every peer is verifying real speculative tokens at every step. PipeDec demonstrates 14-stage pipelines with LLaMA-3.2 1B as draft for 70B target — closer to SwarmLLM's typical hop count than most academic setups.

**Source.** [arxiv 2504.04104](https://arxiv.org/abs/2504.04104).

**Reported.** 4.19–7.79× over plain PP, 2.08–2.69× over tree SD on multi-stage pipelines.

**Stacks with.** Items 1 (persistent stream) + 2 (existing draft model). Partially overlaps SWIFT — pick the better fit per deployment.

**Complexity.** Medium.

### Item 16 — Parallax two-phase scheduler 🟡 PHASE A LANDED 2026-04-18

**What.** Joint optimization of layer placement across heterogeneous peers (latency + bandwidth) plus request-time path selection that *stitches layers from different replicas* into balanced end-to-end chains.

**Source.** [arxiv 2509.26182](https://arxiv.org/abs/2509.26182), [github GradientHQ/parallax](https://github.com/GradientHQ/parallax) (MIT-licensed).

**Reported.** Up to 3.6× throughput, 3.2× lower latency vs decentralized baselines.

**Fit.** Maps directly onto our `peer_registry` + trust scores + shard announcements. Drop-in scheduler upgrade.

**Complexity.** Medium.

#### Phase A — Request-time routing DP (LANDED & DEFAULT-ON 2026-04-18)

- **Adaptation.** Parallax's Phase 2 DP assumes peer-to-peer pipeline data flow (`edge_weight = rtt(peer_A → peer_B)`). SwarmLLM pipelines are **coordinator-relayed** — every segment routes through the local coordinator — so inter-segment edge cost collapses into per-vertex cost: `2 * rtt_local_to_peer + compute_ms + load_ms` (local node: just `compute_ms`). Still a shortest-path DP, just with per-vertex weighting instead of per-edge.
- **Algorithm.** `src/inference/scheduler/parallax.rs::route_shortest_path`:
  - Vertex = `(candidate_idx, range_idx)`; every `(node, layer_range)` from the existing `gather_candidates` output becomes one vertex.
  - DAG edges: `v → w` iff `ranges[v].end == ranges[w].start`.
  - Topological order: sort by `(range.start, range.end)`.
  - Source filter: `range.start == 0 && can_be_first` (and `node == local` when encrypted pipeline).
  - Sink filter: `range.end == num_layers && can_be_last` (and `node == local` when encrypted pipeline).
  - Forward DP: `best_cost[w] = min_v(best_cost[v] + cost[w])`; reconstruct the chain from `parent[]`.
- **Cost model.** `compute_ms = (1000 / est_tokens_per_sec) * (layers / 32)` when the peer has gossiped a capability estimate, else 0 (falls back to pure latency + load). `load_ms = active_request_count * 25 ms`. Constants tuned to make network and compute the dominant terms at typical values.
- **Integration.** `assemble_pipeline_for` calls `parallax::route_shortest_path` first when the flag is on, and **falls back to `greedy_assign` on any error** — so routing never regresses below the greedy baseline. The existing single-local-node fast path (line 178) and encrypted-pipeline local-first/last constraints are both preserved (enforced inside the DP's source/sink filters).
- **Config.** `InferenceConfig::parallax_routing: bool` (default `false`) in `src/config.rs`.
- **Tests.** 8 unit tests (`single_node_covers_all`, `picks_low_latency_chain`, `load_penalty_shifts_choice`, `encrypted_requires_local_first_and_last`, `no_first_capable_errors`, `no_sink_errors`, `disjoint_ranges_fail_cleanly`, `multi_hop_chain_minimizes_total_latency`) + 1 end-to-end scheduler test (`parallax_flag_picks_low_latency_peer_end_to_end`) confirming the flag routes through `PipelineScheduler` correctly. 630 lib tests pass.

#### Phase C.2 — auto-rebalance from `allocate_offline` (LANDED 2026-04-19)

Phase C's layer-allocation recommendation was advisory-only. Phase C.2
plumbs it into `AutoShardManager` as a soft score bias — no hard-coded
action, but the allocator's preferred shard placement nudges both
acquire and prune decisions once a stability window has passed.

- **Stability counter.** `ModelMgmt::parallax_stability: DashMap<ShardId, i32>`,
  clamped to `[-10, 10]`. Each auto-manage evaluation cycle calls
  `update_parallax_stability`: runs `PipelineScheduler::allocate_offline`
  for every known model, unions the `layer_range`s the allocator assigned
  to the local node across all recommended pipelines, and `+1`s any
  shard whose range overlaps one of those ranges — `-1` everything else.
- **Threshold.** `PARALLAX_STABILITY_THRESHOLD = 3`. The bias only
  activates after the allocator has consistently recommended (or
  consistently rejected) a shard for three ticks. A single noisy tick
  can't flip the bias; a long-stable recommendation is hard to dislodge.
- **Acquire bias.** `gather_candidates` multiplies the shard's score by
  `1.5` when stability is `≥ +3`. Same order of magnitude as
  `source_bonus` (regional peer presence) — noticeable without
  overriding rarity/popularity/configured-range signals.
- **Prune bias.** `evaluate_and_prune` adds `+0.5` to the prune score
  when stability is `≤ -3`. Additive, so it stacks with cold-shard /
  pressure bonuses but comes after every existing hard block (locked,
  pinned, encrypted pipeline, configured range, region-elimination,
  holder-busy, reacquire-possible).
- **Feasibility handling.** When `allocate_offline` returns `None` for
  a model (cluster can't cover it), that model's shards are skipped
  entirely — no spurious "not recommended" signal.
- **Flag.** `AutoManageConfig::parallax_auto_rebalance: bool`, default
  `true`. Disabling it makes both stability update and both bias queries
  no-ops in one line.

Files: `src/daemon/state/models.rs` (`parallax_stability` field),
`src/config.rs` (flag), `src/model/auto_manage/parallax.rs` (new —
stability update, bias queries, constants, overlap helper),
`src/model/auto_manage/manager.rs` (`evaluate` calls
`update_parallax_stability` first), `src/model/auto_manage/scoring.rs`
(acquire bias hook), `src/model/auto_manage/prune.rs` (prune bias hook).

8 unit tests in `parallax.rs` cover: overlap math (full/partial/edge/
disjoint/multi-candidate), single-node cluster increments all shards,
counter clamps at `+10` after 20 ticks, feature-flag-off is a no-op
everywhere, unknown shards are neutral, two-peer cluster where a fast
remote preempts local decrements every shard.

#### Phase B.2 — cross-node latency gossip (LANDED 2026-04-19)

Newly-joining nodes no longer need to route requests through a peer
before Parallax can price it. `NodeCapability` gained an
`observed_latencies: Vec<LatencyObservation>` field: each broadcast
carries the sender's top-32 observed per-layer-ms samples, ordered by
the sender's trust in the *observed peer*. Receivers merge each entry
through `SharedState::merge_peer_segment_latency(peer, sample, weight)`
where `weight` = the sender's own `trust_score` in `peer_registry`.

- **Trust-weighted EMA.** Effective α collapses to `0.3 · weight`, so
  trust=0 is a no-op and trust=1 matches a direct local sample.
- **Seed threshold.** A completely fresh entry can only be seeded by a
  sender with trust ≥ 0.3 (default trust is 0.5). Prevents a low-trust
  sender from painting us an out-of-band picture of a stranger.
- **Self + sender skip.** Merge loop skips entries where `obs.peer`
  equals our own `node_id` (we have direct data) or equals the sender
  (no self-promotion).
- **Serde-default.** `observed_latencies` has `#[serde(default, skip_serializing_if = "Vec::is_empty")]` — older peers keep interoperating.
- **Wire budget.** 32 entries × (32 B NodeId + 4 B f32) ≈ 1.2 KB per
  gossip, well under the 4 MB gossipsub cap.

Files: `crates/swarmllm-types/src/node.rs` (field + `LatencyObservation`),
`src/daemon/state/mod.rs` (`merge_peer_segment_latency`),
`src/health/monitor.rs` (outbound top-N snapshot),
`src/daemon/dispatch/mod.rs` (receive-side merge).

5 unit tests in `src/inference/scheduler/tests.rs` cover: zero-trust
no-op, below-seed-threshold skip-insert, weight-scaled EMA math
(weight 1.0 vs 0.5), matching-sample preserves direct observations,
non-finite/non-positive sample guards.

#### Phase B — observed-latency EMA (LANDED 2026-04-18)

Local coordinator view only (not gossiped cross-cluster in v1). After each
successful remote segment in `forward_through_segments`, the coordinator calls
`state.record_peer_segment_latency(node_id, segment_ms, layers)` to update an
EMA of `ms / layer` for that peer. `state.observed_latency_ms_per_layer(node)`
exposes the smoothed value. EMA parameters: α=0.3 (30% weight on latest
sample), stored in `metrics.peer_segment_latency_ms_per_layer: DashMap<NodeId, f32>`.

**Cost model change.** When a candidate has an observed per-layer latency, the
DP treats it as the *whole* segment cost (it already includes compute, network
round-trip, and peer-side queuing). That replaces both the `2 * latency_ms`
network term AND the `est_tokens_per_sec`-derived compute term. Without
observations the DP falls back to the two-part Phase A cost model.

**Why per-layer normalisation.** Different pipeline arrangements put different
layer widths on the same peer. Per-layer EMA lets a 4-layer segment's
observation on peer A inform the cost of a hypothetical 16-layer segment on
the same peer (multiply by width).

**Test coverage.** `observed_latency_overrides_static_estimate` proves the DP
prefers the lower observed candidate when static signals tie;
`peer_segment_latency_ema_math` verifies the EMA formula + width normalisation
+ zero-layer guard + unknown-peer fallback.

#### Phase C — offline layer allocator (LANDED 2026-04-18)

`src/inference/scheduler/parallax_allocator.rs` (≈320 LOC). Greedy multi-
pipeline packer (simpler than Parallax's DP; the DP's state space explodes on
heterogeneous peers and we observed most of the benefit is captured by
greedy + the `Z(k) = k² / s*(k)` objective).

- **Input.** `Vec<PeerCapacity>` (one per peer: layer_capacity, tokens_per_sec, latency_ms) + `num_layers` + `max_pipelines`.
- **Output.** `AllocationPlan { pipelines: Vec<PipelineAllocation>, throughput_score: f32 }`. Each `PipelineAllocation` is a contiguous full-coverage chain over `[0, num_layers)` plus an estimated end-to-end latency using the same cost model family as Phase A routing.
- **Algorithm.** For each candidate `k` in `[1..=max_pipelines]`: test feasibility (`total_capacity >= k * num_layers`), then greedy-pack in fastest-first peer order with a running per-peer remaining-capacity budget. Pick the plan with the highest `Z(k) = k² / avg_stages_per_pipeline`.
- **Integration.** `PipelineScheduler::allocate_offline(&self, &ModelId, max_pipelines) -> Option<AllocationPlan>` snapshots `peer_registry` + local node capacity (union of on-disk shard layer ranges), derives `PeerCapacity`, runs the allocator. Local capacity comes from actual shards on disk, not aspirational VRAM — keeps recommendations aligned with what this node is ready to serve today.
- **Tests.** 7 unit tests cover: single-node full coverage, balanced cluster prefers k = num_peers, heterogeneous greedy-by-throughput, infeasible when undersized, zero-capacity peer skipping, water-filling a big peer across multiple pipelines, `Z(k)` monotonic-in-k for balanced clusters.
- **Not wired.** `AutoShardManager` / `ShardRebalancer` don't auto-act on the recommendation yet — today it's operator-visible only. Wiring that into rebalance triggers is Phase C.2.

#### Remaining phases

- **Phase D — multi-pipeline concurrency.** Run multiple pipeline assignments in parallel for a single model when enough candidates exist; route requests across them via weighted load balancing. Relevant once Item 7 lands concurrent-user batching.
- **Phase C.2 — auto-rebalance from Phase C output.** LANDED 2026-04-19 — wired through `AutoShardManager`, not `ShardRebalancer`. See Phase C.2 section below.
- **Phase B follow-up — cross-node signal sharing.** Landed as Phase B.2 (2026-04-19) — see section above.

#### Stacks with

- **Item 7 BatchGenerate** — Parallax picks the best chain; Item 7 multiplexes requests through it. Complementary.
- **Item 12 DSD** — DSD amortizes network RTT across γ tokens; Parallax picks the chain with the lowest RTT to start with. Multiplicative.
- **Items 4, 5** — Fast path and prefix cache both bypass multi-segment routing, so Parallax Phase A helps only when pipelines span 2+ peers. Phase A flag is safe to enable universally because the single-segment local-coverage fast path still short-circuits ahead of the DP.

### Item 17 — Disaggregated prefill / decode (Mooncake-style for P2P)

**What.** Bandwidth-rich peer runs prefill (compute-bound), streams chunked KV cache to a latency-good peer for decode (memory-bound). Item 4's "remote-generate fast path" is a degenerate same-peer version; generalizing to two different peers unlocks cases where no single peer can do both.

**Source.** [Mooncake](https://kvcache-ai.github.io/Mooncake/), [DistServe arxiv 2401.09670](https://arxiv.org/abs/2401.09670), [LMCache tech report](https://lmcache.ai/tech_report.pdf).

**Reported.** DistServe: 7.4× request rate, 12.6× tighter SLO. Together.ai: 40% on long-context.

**Complexity.** Large. New chunked-KV transfer message type. Decode-resume semantics. Integrates with Items 5 (prefix cache) and 8 (cross-node prefix sharing).

### Item 18 — Per-token early-exit with adaptive depth (HELIOS / TIDE / DREX)

**What.** Some tokens exit at layer 12, others at layer 32 — natural extension of SWIFT (Item 6). In P2P, early-exiting tokens skip the trailing pipeline hops entirely, killing tail latency on multi-segment routes.

**Source.** [HELIOS arxiv 2504.10724](https://arxiv.org/abs/2504.10724), [DREX arxiv 2512.15705](https://arxiv.org/abs/2512.15705).

**Complexity.** Large. Per-token routers either trained per-model (TIDE) or speculative-exit signaled (SpecEE).

### Sequencing recommendation (Round 3)

**Order:** ~~13~~ → 12 → 16 → 14/17/18. (Item 13 landed 2026-04-18.)

- Item 12 (DSD) is the natural progression after Item 7 — turns the network round-trip from cost into compute.
- Item 16 (Parallax scheduler) is structural — improves every other item by routing requests to better-matched peer chains.
- Items 14, 17, 18 are larger swings; pick based on whether the dominant pain is latency hiding (14), prefill/decode imbalance (17), or tail latency on multi-hop routes (18).

### Skipped as not-a-fit

- Wide-EP MoE (needs NVLink-class interconnects)
- NIXL / direct RDMA KV transport (needs RoCE hardware)
- CXL shared-memory KV (rack-scale only)
- DiLoCo (training, not inference)
- Further weight quantization (we already ship Q4/Q8 GGUF; not distribution-specific)
