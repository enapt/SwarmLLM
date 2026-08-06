# Inference Pipeline

## Subprocess-Per-Model Isolation

Each loaded model runs in its own **`swarmllm model-worker` subprocess** (Ollama-style). When a model is unloaded, the subprocess is killed and the OS + CUDA driver immediately reclaim all GPU memory — no daemon restart required.

```
Main daemon                          model-worker subprocess (one per model)
───────────────────────────────      ───────────────────────────────────────
ModelProcessPool.generate()  ─────►  loads shards from disk on first request
ModelProcessPool.forward()   ─────►  runs forward passes / full decode loop
                             ◄─────  streams WorkerMsg::Token / LayerResult
unload_model()               ─────►  kill process → OS frees all VRAM
```

**IPC**: Unix domain socket with binary framing — `[4B json_len][json header][4B payload_len][raw tensor bytes]`. JSON carries message metadata; the payload carries raw activation bytes to avoid base64 overhead.

**Message types** (`src/inference/worker_ipc.rs`):

| Message | Direction | Purpose |
|---|---|---|
| `DaemonMsg::Forward` | daemon → worker | Single-step LayerForward (distributed inference) |
| `DaemonMsg::Generate` | daemon → worker | Full prompt→tokens decode loop (API inference) |
| `DaemonMsg::Unload` | daemon → worker | Drop a layer range (partial memory reclaim) |
| `DaemonMsg::Shutdown` | daemon → worker | Graceful worker exit |
| `WorkerMsg::Token` | worker → daemon | Streaming decoded token |
| `WorkerMsg::LayerResult` | worker → daemon | Activation result for pipeline forwarding |

**`SplitModelEntry`** is metadata-only — it caches `eos_tokens`, `vocab`, `chat_template`, `bos_token`, and `eos_token_str` from the GGUF header without loading model weights. The weights live exclusively in the worker subprocess.

**Worker granularity**: one process per `ModelId` (not per shard). A single worker handles all layer ranges for a model and owns its own `KvCacheStore`. Individual shard unload uses `DaemonMsg::Unload`; the process exits only when all shards are released.

## Split Inference Engine

The split inference engine (`src/inference/split/`) enables distributed inference using candle for direct tensor computation with quantized GGUF weights. Each node loads only its assigned transformer layers (in the worker subprocess), forwarding hidden-state activations between nodes. The module is split into: `model.rs` (SplitModel struct + accessors), `loader.rs` (GGUF/shard load), `executor.rs` (forward pass + tensor-parallel), `kv_cache.rs`, `entry.rs`, `gguf_meta.rs`, `shard_reader.rs`, `rope.rs`, `prefix_cache.rs`.

```
Client → API Server → InferenceRouter → Pipeline Assembly
                                              │
                      ┌───────────────────────┘
                      ▼
          ┌──────────────────────┐
          │   Pipeline Segment   │     Token IDs (prefill)
          │ Node A: Layers 0-15  │──── LayerForward ──►
          └──────────────────────┘                      │
                                        ┌───────────────┘
                                        ▼
                            ┌──────────────────────┐
                            │   Pipeline Segment   │
                            │ Node B: Layers 16-27 │── sample token ──►
                            └──────────────────────┘
```

## Pipeline Assembly

1. Fetch model manifest to determine layer ranges
2. **Pipeline affinity check**: if multi-turn session has a previous pipeline and all nodes are still connected, reuse it (KV cache locality)
3. Query model_registry.shard_holders for hosting nodes
4. **Liveness filter**: drop holders that aren't in `connected_node_ids` (the libp2p truth — DHT can re-inject providers for peers that just disconnected, and `peer_registry` is intentionally preserved across mid-pipeline disconnects for reconnect attempts)
5. Fetch node load/latency from peer_registry
6. **Parallax scheduler**: shortest-path dynamic programming over observed per-layer latencies (EMA over recent forwards), rather than a greedy latency-only sort. Cross-gossips top-32 observed latencies via `NodeCapability.observed_latencies` so every node has a current view of the network's compute profile
7. **Encrypted pipeline check**: if enabled for this model, force first and last segments to the local node (boomerang topology)
8. Assignment: widest contiguous layer range per node, merging on same-node
9. Identify standby nodes per segment (failover)
10. Send PipelineAssignment, wait for ACKs, begin forwarding

## Failure Handling

The router applies a **single retry** on transient remote failures
(silent rr drops, OutboundFailure, remote-generate timeouts). The
retry passes `preferred_pipeline = None` so the scheduler re-runs
and the dead/dropped peer is filtered out via the liveness oracle
above. Failure of the second attempt propagates to the user with a
"try again" hint.

Independently, streaming-tracked `SendDirectMessage` sends carry a
`delivery_request_id`; if the receiver doesn't ACK within
`RR_ACK_TIMEOUT_SECS` (10s), the daemon closes the caller's
streaming channel — converting a 120s `FIRST_TOKEN_TIMEOUT` hang
into a fast-fail in ~10–20s. This handles the rare case where
libp2p `request_response` accepts a `send_request` call but never
delivers it (no `OutboundFailure` event fires).

## Concurrent Request Throttling

Per-tier concurrency caps come from `max_concurrent_requests`
(default 10): Bronze=¼, Silver=½, Gold=1×, Platinum=2×. Requests
beyond the cap queue in the router. **The queue is event-driven**:
every `active_count.fetch_sub(1)` on completion is paired with
`queue_notify.notify_one()` so `drain_queue` wakes immediately.
Without that pairing, queued requests would sit indefinitely until
the next Submit arrived (a real bug found in stress testing — fix
in commit `da6f485`).

Pipeline affinity means that multi-turn conversations (with `session_id`) prefer to route through the same nodes, preserving KV-cache state and avoiding cold restarts on every turn.

The Parallax allocator also runs offline in `AutoShardManager` (Phase C.2)
with a soft acquire/prune bias driven by a per-shard stability counter
(≥3 consistent ticks of "this shard wants to move here" before it acts).
Hard constraints (pinning, trust gates, VRAM caps) always win.

## Architecture Detection

The SplitModel loader reads `general.architecture` from GGUF metadata and applies per-architecture handling:

| Architecture | RoPE | QKV Biases | Special Handling |
|---|---|---|---|
| **Llama** | Interleaved | No | Default EOS=2 |
| **Llama 4** | iRoPE (NoPE every 4th) | No | MoE FFN |
| **Qwen2** | Contiguous | Yes | EOS 151643+151645 |
| **Qwen 3.5** | Contiguous | No | Hybrid SSM+attention (Gated Delta Networks) |
| **Gemma/Gemma2** | Interleaved | No | Embedding scaling (sqrt(d)), Gemma RmsNorm (+1), EOS 107, attention + final logit softcapping, Gemma chat template fallback |
| **Phi-3** | Su/YaRN | Yes | Fused QKV/FFN tensors |
| **Mistral** | Interleaved | No | GQA |
| **DeepSeek-V2/V3** | Contiguous | No | MLA attention + MoE FFN |
| **GLM-4** | Contiguous | No | Partial RoPE, extreme GQA (16:1) |
| **Starcoder2** | Interleaved | Yes | Code-optimized |

## KV-Cache Management

- Per-request isolation via `DashMap<(ModelKey, RequestId), Cache>`
- Multi-turn reuse: `session_id` tracks conversations, prefix matching skips redundant prefill
- Configurable TTL (default 10 min)
- VRAM-aware LRU eviction for split model cache

## Prefix-Cache KV Sharing (Cross-Node)

Each worker stores a local prefix-cache keyed by BLAKE3 chained hashes
over fixed-size token blocks (`prefix_cache_block_tokens`, default 64).
Blocks are announced to peers via `SwarmMessage::PrefixCacheAnnounce`
on the `swarm/models` gossipsub topic and indexed in
`state.models.cross_node_prefix_index`.

When a local worker sees a prompt whose prefix it hasn't prefilled, it
emits `WorkerMsg::PrefixFetchProbe`; the daemon walks the index
(longest-match first), trust-gates candidate peers by
`cross_node_prefix_trust_min` (default 0.5), and issues a
`SendPrefixKvFetch` request-response to the best holder. The serving
daemon re-issues `DaemonMsg::ExportPrefixSnapshot` to its worker, which
narrows a stored `KvSnapshot` to the requested block boundary and returns
the serialized bytes in the IPC binary-payload slot. Back on the
requesting side, the bytes are BLAKE3-reverified against the requested
hash and NaN/Inf-scanned before hydrating a new `KvCacheEntry` for the
in-flight request, which then only has to prefill the suffix beyond the
cached block boundary.

Three chained timeouts (worker probe 3000 ms, daemon network 2500 ms,
serving IPC 2000 ms — sized for 7B-class f32 snapshots) guarantee that a
stuck peer degrades to a clean miss rather than blocking the request.
See the [Performance chapter](../operations/performance.md#cross-node-prefix-kv-sharing)
for measured TTFT numbers on TinyLlama (GPU, corner case where fetch is
slightly slower than prefill) vs Qwen2.5-7B (**12.9× iter-1 TTFT
speedup** on CPU-CPU localhost).

## Advanced Features

- **Speculative Decoding** — Draft model proposes K tokens, target verifies in one pass (flag-gated `speculative_distributed`)
- **SWIFT self-speculative** — Target model acts as its own draft by skipping a layer range (flag-gated `swift_self_speculative`)
- **DSD (Decentralized Speculative Decoding)** — Multi-segment pipeline with γ-token speculation woven in (flag-gated `decentralized_spec_decoding`)
- **Chunked Prefill** — Sarathi-style: each Prefilling slot advances by `prefill_chunk_tokens` (default 128) per decode tick so a long admission can't block decode
- **Continuous Batching** — default-on: concurrent `Generate` requests share one `forward_batch` per decode tick; GPU uses fused kernel, CPU falls through to sequential
- **Batched Prefill Forward** — default-on: fuses concurrent same-shape Prefilling chunks into one `forward_batch` call
- **Remote-generate Fast Path** — default-on: single-segment distributed inference runs the full decode loop on the remote worker instead of per-token coordinator round-trips (measured 1.93× decode speedup)
- **Cross-request Prefix Cache** — default-on: see "Prefix-Cache KV Sharing" above for the cross-node extension; the local cache alone is a 29.4× wall-clock win on prompt re-submission
- **Activation Compression (Q8_0)** — Intermediate pipeline activations wire-quantized ~3.76× (flag-gated `activation_compression`)
- **Flash Attention** — CPU and GPU fast paths (GQA-native, no `repeat_kv`)
- **PagedAttention** — Deferred; `paged-attn` feature flag reserved for future use (module removed, never wired to production)
- **Logprobs** — NOT returned by local inference. The sampler can compute them (`sample_token_with_logprobs`) and the response type serializes them, but every local execution path pins `token_logprobs: vec![]`, so nothing reaches the response. `/v1/chat/completions` therefore REFUSES `logprobs` for a locally-served model rather than answering 200 with the field absent, which is indistinguishable from a request that never asked. Cloud-routed models still return them. See `docs/ARCHITECTURE.md` § Deferred Items
- **Pipeline Error Broadcast** — On distributed inference failure, `broadcast_pipeline_error()` notifies all participants so peers can update shard availability and route around failures
- **Local Embedding Privacy** — When `local_embedding_privacy: true`, the requesting node performs token→embedding locally (~1ms) and sends pre-embedded hidden-state activations instead of raw token IDs to the first pipeline segment. Remote nodes never see the plaintext prompt. See [Security > Local Embedding Privacy](../architecture/security.md#local-embedding-privacy)
- **Encrypted Pipeline** — When enabled (per-model or global), forces a "boomerang" topology: the requesting node handles both the first segment (embedding) and last segment (token sampling). Remote nodes only process intermediate activations — no remote node ever sees plaintext input or output. See [Security > Encrypted Pipeline](../architecture/security.md#encrypted-pipeline)

## Vision Language Models (VLM)

### Distributed mmproj

The mmproj (vision encoder) is modeled as a sentinel shard (`index = u32::MAX`) decoupled from the text pipeline. Any node with mmproj can encode images — the router selects local → first-segment → any holder.

```
Image → JPEG compress → VisionEncodeRequest (remote) or encode locally
    → zstd+FP16 compressed embeddings
    → attached to first LayerForward (vision_embeddings field)
    → text pipeline processes as normal
```

Key types: `VisionEncodeRequest`, `VisionEncodeResponse`, `LayerForward.vision_embeddings`.

If no node has mmproj loaded, the API returns HTTP 503 (`VisionEncoderUnavailable`).

## Tensor Wire Format

```
[4B ndim][4B×ndim shape][4B dtype_tag][f32 data]
```

For a 7B model (hidden_dim=3584):
- Prefill (14 tokens): ~200 KB
- Decode (1 token): ~14 KB
