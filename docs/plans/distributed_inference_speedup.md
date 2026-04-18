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
> | **Item 5 — Cross-request prefix cache** | ✅ LANDED & DEFAULT-ON | **29.4× wall-clock** on cache-hit re-submission of the same 513-token prompt. |
> | Item 1 — Persistent pipeline stream | ✅ Landed behind `persistent_pipeline_stream=false` flag. Verified end-to-end but no measured latency win (bottleneck was elsewhere). |
> | Item 2 — Speculative decoding (distributed) | ✅ Landed in 3 phases behind `speculative_distributed=false` flag + requires loaded draft model. Working; 40–52% accept rate w/ backend mismatch (llama-cpp draft vs candle target). |
> | Item 3 — Continuous batching | 🟡 Phase 1 (wire protocol) landed behind `continuous_batching=false` flag. Phase 2 (scheduler refactor + `SplitModel::forward_batch`) is the remaining work; blocker is replacing `Mutex<socket>` in `ModelProcessPool` with mpsc-fed scheduler. Not pursued in this session because Item 4 delivered the headline win. |
> | Item 6 — SWIFT self-speculative | 🟡 Landed behind `swift_self_speculative=false` flag. Structurally slower than baseline on candle until flash-attn-with-mask lands (kernel mismatch on multi-position verify). Shelved. |
> | **Item 13 — Activation compression (Q8_0)** | ✅ LANDED behind `activation_compression=false` flag. Codec verified (~3.76× compression, RMS error <0.005, peer-compatible auto-dispatch). End-to-end multi-segment benchmark pending. |
> | Item 12 — DSD (decentralized speculative) | 🟡 Phases 1–3 + Phase 4 part 1 LANDED 2026-04-18: worker γ-token decode + truncation primitives verified + γ controller + worker speculative-verify branch decoupled (multi-segment pre-embedded verify variant added). Phase 4 part 2 (coordinator loop in `pipeline/dsd.rs`) remains — pure coordinator-side logic, ~300 LOC. |
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

### Phase 4 part 2 (NEXT): coordinator loop

Remaining: new `src/inference/pipeline/dsd.rs` with `try_dsd_distributed`:
1. Eligibility (multi-segment, draft model loaded, greedy temp=0, no TP/vision/LoRA/encryption)
2. `GammaController::new(speculative_gamma)` for adaptive γ
3. Reuse Item 2's `draft_prefill`, `draft_next_gamma`, `draft_sync_after_round` helpers from `pipeline/speculative.rs`
4. New `forward_through_segments_speculative` (a stripped-down sibling of `forward_through_segments`) that propagates `LayerForward { draft_tokens, spec_logits_requested: true, truncate_kv_to: pending }` through every segment — first segment receives γ × 8 byte token IDs, intermediates receive `[1, γ, hidden]`, last segment returns γ+1 logits via `spec_logits`
5. Greedy accept-reject loop, bonus token, `controller.record_round(accepted, gamma)`, update `pending_truncate = Some(index_pos + accepted + 1)` for the next round
6. Behind `decentralized_spec_decoding=false` config flag (default off until measured)

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

### Item 16 — Parallax two-phase scheduler

**What.** Joint optimization of layer placement across heterogeneous peers (latency + bandwidth) plus request-time path selection that *stitches layers from different replicas* into balanced end-to-end chains.

**Source.** [arxiv 2509.26182](https://arxiv.org/abs/2509.26182), [github GradientHQ/parallax](https://github.com/GradientHQ/parallax) (MIT-licensed).

**Reported.** Up to 3.6× throughput, 3.2× lower latency vs decentralized baselines.

**Fit.** Maps directly onto our `peer_registry` + trust scores + shard announcements. Drop-in scheduler upgrade.

**Complexity.** Medium.

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
