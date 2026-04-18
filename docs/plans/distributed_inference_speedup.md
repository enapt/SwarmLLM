# Distributed Inference Speedup Plan

> **READ THIS FIRST (status as of 2026-04-18)**
>
> This doc captures a multi-session effort to speed up distributed inference.
> Four items were scoped; the headline win came from an item that wasn't in
> the original plan (Item 4).
>
> | Item | Status | Effect |
> |---|---|---|
> | **Item 4 — Remote-generate fast path** | ✅ LANDED & DEFAULT-ON | **1.93× decode speedup** on default config. Main user-visible win. |
> | Item 1 — Persistent pipeline stream | ✅ Landed behind `persistent_pipeline_stream=false` flag. Verified end-to-end but no measured latency win (bottleneck was elsewhere). |
> | Item 2 — Speculative decoding (distributed) | ✅ Landed in 3 phases behind `speculative_distributed=false` flag + requires loaded draft model. Working; 40–52% accept rate w/ backend mismatch (llama-cpp draft vs candle target). |
> | Item 3 — Continuous batching | 🟡 Phase 1 (wire protocol) landed behind `continuous_batching=false` flag. Phase 2 (scheduler refactor + `SplitModel::forward_batch`) is the remaining work; blocker is replacing `Mutex<socket>` in `ModelProcessPool` with mpsc-fed scheduler. Not pursued in this session because Item 4 delivered the headline win. |
>
> **If starting a new session, the most useful things to pick up are:**
> 1. Item 3 Phase 2 (concurrent-user throughput — independent of Item 4)
> 2. Extending Item 4 to multi-segment pipelines
> 3. Item 2 with a matched-backend draft model (pre-trained candle-native draft, ~5% target size — research flagged `Qwen2.5-0.5B` as draft for `Qwen2.5-7B` target with 1.4× speedup on llama.cpp benchmarks)
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

**Phase 2 remaining**: the actual runtime benefit.
- Replace `WorkerHandle.socket: Mutex<...>` in `src/inference/process_pool.rs` with `requests_tx: mpsc::Sender<BatchRequest>` + spawn scheduler task per worker. Scheduler implements the 5 ms collection window (degrades to ~15 ms on WSL2 but still coalesces concurrent arrivals).
- Implement `SplitModel::forward_batch` that stacks per-request decode inputs into a single `[batch_size, 1, hidden]` tensor for a real matmul fusion (v1 sequential loop; v2 true tensor stacking).
- Switch `WorkerMsg::BatchResult` emission in `handle_batch_forward` (currently emits N `LayerResult` messages).

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

## Item 5 — Cross-request prefix caching (RadixAttention-style)

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

## Item 6 — SWIFT self-speculative decoding (layer-skip draft)

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

---

## Cross-item sequencing (Round 2)

**Order:** 5 → 6 → 7.

- **Item 5 (prefix cache)** is foundational and self-contained. Biggest single TTFT win. Doesn't touch async/scheduler surface.
- **Item 6 (SWIFT)** is a decode-loop mod inside `handle_generate`. Builds on top of Item 5 without conflict.
- **Item 7 (BatchGenerate)** is the largest refactor (actor model + SlotTable). Builds on Items 5 & 6 because both must work within a batched decode step.

## Deferred to future sessions

- **Item 8 — Cross-node prefix cache sharing.** Announce BLAKE3 prompt-prefix hashes over gossip; peers that already prefilled a shared prefix serve KV blocks on demand. Content-addressed KV shards — fits our existing shard announcement infrastructure. Potentially novel for P2P.
- **Item 9 — EAGLE-3 draft heads.** Per-target pretrained draft heads (SafeAILab HF) distributed as a new shard type. 3–6× ceiling, highest complexity.
- **Item 10 — FlowSpec / PPSD pipelined speculation.** Addresses the multi-segment pipeline case Item 4 didn't cover. Requires Item 1 stream to be default-on.
- **Item 11 — Lookahead decoding.** No-draft-model Jacobi iteration. Test on CPU to see if FLOP overhead beats the speedup.
