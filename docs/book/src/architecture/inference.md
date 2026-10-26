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

**IPC**: a local socket (a Unix domain socket on Linux and macOS, a named pipe on Windows) with binary framing — `[4B json_len][json header][4B payload_len][raw tensor bytes]`. JSON carries message metadata; the payload carries raw activation bytes to avoid base64 overhead.

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

The split inference engine (`src/inference/split/`) enables distributed inference using candle for direct tensor computation with quantized GGUF weights. Each node loads only its assigned transformer layers (in the worker subprocess), forwarding hidden-state activations between nodes. The module is split into: `model.rs` (SplitModel struct + accessors), `loader/` (GGUF/shard load), `executor.rs` (forward pass + tensor-parallel), `hybrid.rs` (how many of a segment's layers go on the card), `kv_cache.rs`, `kv_budget.rs` (card-aware KV and RoPE sizing), `kv_write.rs` (the card's f16 cache), `token_embedding.rs`, `draft.rs` (the in-engine drafter for speculation across computers), `entry.rs`, `gguf_meta.rs`, `shard_reader.rs`, `rope.rs`, `prefix_cache.rs`.

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
6. **Parallax scheduler**: shortest-path dynamic programming over observed per-layer latencies (EMA over recent forwards), rather than a greedy latency-only sort. Cross-gossips top-32 observed latencies via `NodeCapability.observed_latencies` so every node has a current view of the network's compute profile. Since 0.3.230 a candidate that must first load the model is also charged for the load: each node times its own loads and advertises `NodeCapability.model_load_ms_per_gib`, and the layers a candidate does not hold are priced at that rate (10 s/GB for a node that has not reported one) — this node included
7. **Start-and-finish check** (`encrypted_pipeline`): if enabled for this model, force first and last segments to the local node (boomerang topology)
8. Assignment: widest contiguous layer range per node, merging on same-node
9. Identify standby nodes per segment (failover)
10. Send PipelineAssignment, wait for ACKs, begin forwarding

**Each computer's part is weighed as what loading it takes** (since 0.3.255): a part
ending the model carries its output head and one starting it the token table, so a
70B's last layers weigh more than its middle ones. **Before the prompt is read, every
computer in the plan is asked whether its part fits**; one that says no also says how
many layers would (since 0.3.256, whether it counted layers that load or layers whose
conversation memory fits), and the next plan holds it to that.

**Teams for a model no single computer holds** are planned ahead by the computers
themselves, not per request. For a model somebody asked for that no nearby computer
(measured within about 30 ms) can run whole, every nearby computer works out the same
chain and fetches only its own part of it (since 0.3.247). Where the nearby computers
cannot carry the model between them, the same chain is planned over every computer in
reach (since 0.3.254) — slow, but it is how Llama 3.3 70B first ran, across four
computers in Thailand, Australia and Italy.

## Failure Handling

The router applies a **single retry** on transient remote failures
(silent rr drops, OutboundFailure, remote-generate timeouts). The
retry passes `preferred_pipeline = None` so the scheduler re-runs
and the dead/dropped peer is filtered out via the liveness oracle
above. Failure of the second attempt propagates to the user with a
"try again" hint.

One source of "silent rr drops" was found and removed in v0.3.219: the
request-response layer kept connections the swarm had already refused
(the per-peer limit of 3) and, whenever the real connections each had a
request in flight, sent the next one down a refused connection, where it
vanished until the 600 s request timeout. Bursts of streamed tokens lost
their tail this way. A refused connection is now forgotten and anything
sent to it reported failed (gotcha #774); the retry above still covers any
other loss.

Independently, streaming-tracked `SendDirectMessage` sends carry a
`delivery_request_id`; if the receiver doesn't ACK within
`RR_ACK_TIMEOUT_SECS` (10s), the daemon closes the caller's
streaming channel — converting a first-token-deadline hang (120 s
plus 0.5 s per prompt token, at most 600 s, plus 240 s per computer that may
first have to load the model) into a fast-fail in ~10–20s. This handles the rare case where
libp2p `request_response` accepts a `send_request` call but never
delivers it (no `OutboundFailure` event fires).

## Concurrent Request Throttling

Every request gets the same middle tier while credits are dormant,
and that tier runs at most HALF of `max_concurrent_requests` at once
(5 with the default 10). Requests beyond the cap queue in the router. **The queue is event-driven**:
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

The RoPE column is llama.cpp's per-architecture choice (`llama_model_rope_type`), which `ModelArch::use_rope_contiguous` copies arch by arch — it is not inferred.

| Architecture | RoPE | QKV Biases | Special Handling |
|---|---|---|---|
| **Llama** | Interleaved | No | Default EOS=2 |
| **Llama 4** | Interleaved; iRoPE (NoPE every 4th) | No | MoE FFN |
| **Qwen2 / Qwen3** | Contiguous | Yes (Qwen2) | EOS 151643+151645; per-head q/k norm (Qwen3) |
| **Qwen2-MoE / Qwen3-MoE** | Contiguous | Yes (Qwen2-MoE) | The same layout with routed experts per layer; Qwen2-MoE adds a gated shared expert and does not renormalise its top-k weights |
| **Qwen 3.5** | Contiguous (IMROPE over 64 of 256 dims; text positions only) | No | Dense and mixture-of-experts models: three of every four layers are Gated DeltaNet, which keeps a running state per conversation, and the fourth is gated attention. It guesses ahead since 0.3.253: that state cannot simply be cut back, so each check keeps what it read and a wrong guess is undone by replaying the words that were right from the state as it stood before the check; a split guesses only when every computer in it runs 0.3.253 or later. Since 0.3.243 a conversation's next turn starts from the state saved just before the end of the last prompt instead of reading the whole conversation again (Qwen3.5-0.8B on the processor: 16.7 s instead of 127.6 s for the second turn of a 2,416-token conversation). A card too small for the whole model takes as many layers as fit and the processor the rest (since 0.3.230), and only the attention layers are charged conversation memory. Checked against llama.cpp on Qwen3.5-0.8B and 4B, and the mixture-of-experts models (`qwen35moe`, since 0.3.235) on a small real file with the same layout as Qwen3.5-35B-A3B |
| **Gemma/Gemma2** | Contiguous | No | Embedding scaling (sqrt(d)), Gemma RmsNorm (+1), EOS 107, attention + final logit softcapping, Gemma chat template fallback; Gemma 2 sees only the last 4,096 positions on every other layer |
| **Gemma 3** | Contiguous; windowed layers at their own base, global layers with the file's linear scaling | No | Per-head q/k norm; five layers in six see only the last 512 (1B) or 1,024 (4B and up) positions; no softcapping. Checked against llama.cpp on the 1B and 4B. Reads pictures since 0.3.245, all of a picture's rows at once as it was trained to: scored through llama.cpp's own picture reader, 95-97% of its words are llama.cpp's first choice. A picture request split across computers needs every one of them on 0.3.245 or later |
| **Gemma 4** (text: 12B, 31B, 26B-A4B, E2B, E4B) | Contiguous; windowed layers turn 256 of their head at base 10,000, global layers 512 at 1,000,000 with the file's per-pair factors ("proportional") | No | Each layer has its own shape — on the 12B, five layers in six have 8 key/value heads 256 wide over the last 1,024 positions, every sixth has one head 512 wide over everything and no V projection (V is K). Q/K normed per head with weights, V without; no attention scale; a per-layer output scale; final softcap 30; its own tokenizer rules (spaces as `▁`, newline runs whole, BOS always). Checked against llama.cpp master on the 12B. The 26B-A4B runs a dense GELU feed-forward beside 128 GELU experts (8 per token) in every layer: each branch has its own norms, the router reads its own norm of the layer's input, and each expert's answer has its own scale; checked against llama.cpp master on the real Q2_K file. The E2B and E4B add two things: their last layers (20 of the E2B's 35) keep no cache of their own and attend the cache of the last earlier layer of their kind, and every layer adds an embedding of its own gated by an input built from the prompt's tokens. Because only the first layers see the tokens, these two run only on one computer holding every layer (a card and the processor of that computer may still share them). Their per-layer embedding table (1.8 GB on the E2B, 2.2 GB on the E4B) stays in the computer's memory, not on the card, so the E2B runs whole on a 4 GB card and the E4B on a 6 GB one (since 0.3.241); checked against llama.cpp master on the real E2B and E4B. The vision part is refused by name until it is built |
| **Phi-3** | Contiguous, Su/YaRN scaling | Yes | Fused QKV/FFN tensors |
| **Mistral** | Interleaved | No | GQA |
| **DeepSeek-V2/V3** | Interleaved | No | Not supported yet — recognised and refused (no real file loads; needs YaRN on its MLA layers and the current MLA tensor layout) |
| **GLM-4** | Interleaved | No | Partial RoPE, extreme GQA (16:1) |
| **Starcoder2** | Contiguous | Yes | Not supported yet — recognised and refused (a LayerNorm model with biases the loader does not read; no real file loads) |

**Mixture-of-experts layers keep every expert quantized**, one matrix per
expert cut from the GGUF's stacked tensor, the way llama.cpp runs them — so a
MoE model needs about its file's size in memory, and it is planned at its
full size (every expert is resident; nothing pages experts in and out).
On a graphics card too small for all of it, the experts of as many layers as
needed are kept in the computer's memory and everything else stays on the
card; the card runs experts stored as Q2_K-Q6_K, Q8_0, Q4_0, Q4_1, Q5_0 or
Q5_1 (the last four since 0.3.241, for Gemma 4's 26B-A4B).
Models quantized with the `IQ*` family (IQ1_S, IQ3_XXS, …) cannot be read by
this engine and are refused with a message naming the type; choose a K-quant.

## KV-Cache Management

- Per-request isolation via `DashMap<(ModelKey, RequestId), Cache>`
- Multi-turn reuse: `session_id` tracks conversations, prefix matching skips redundant prefill
- Configurable TTL (default 10 min)
- VRAM-aware LRU eviction for split model cache
- Sliding-window layers (Gemma 2, 3 and 4) keep only the positions their window still reads, as
  llama.cpp does — a Gemma 4 12B conversation at 32,000 tokens holds about 1 GB of cache instead of
  10.5 GB — and every memory estimate (this computer's admission and placement, and the planner's
  bound on a peer that advertises it) charges them that

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
- **DSD (Decentralized Speculative Decoding)** — Multi-segment pipeline with γ-token speculation woven in (`decentralized_spec_decoding`, on by default since v0.3.213)
- **Chunked Prefill** — Sarathi-style: each Prefilling slot advances by `prefill_chunk_tokens` (default 128) per decode tick so a long admission can't block decode
- **Continuous Batching** — default-on: concurrent `Generate` requests share one `forward_batch` per decode tick; GPU uses fused kernel, CPU falls through to sequential
- **Batched Prefill Forward** — default-on: fuses concurrent same-shape Prefilling chunks into one `forward_batch` call
- **Remote-generate Fast Path** — default-on: single-segment distributed inference runs the full decode loop on the remote worker instead of per-token coordinator round-trips (measured 1.93× decode speedup)
- **Delegated split** (v0.3.219) — default-on: a request whose plan is several segments, none of them this node's, is handed to the node holding the model's first layers, which leads it among its own peers and streams the reply back the same way. The requester no longer sits inside every token's round trip: measured on a rig with the requester 150 ms away, 1.4-1.7× faster with two or three holders. The delegate keeps the first segment itself (the requester chose it to read the prompt) and never hands the request on; it is not used in private mode or with "Start and finish on this computer" (`encrypted_pipeline`). `SWARMLLM_DELEGATE_SPLIT=0` switches it off
- **Chained checks** (v0.3.219) — a speculative check travels a run of remote segments like an ordinary decode step, one trip around them instead of one per segment (+31% on a guessing request across three nodes, the leader 150 ms from the others). `SWARMLLM_CHAIN_VERIFY=0` switches it off
- **Cross-request Prefix Cache** — default-on: see "Prefix-Cache KV Sharing" above for the cross-node extension; the local cache alone is a 29.4× wall-clock win on prompt re-submission
- **Activation Compression (Q8_0)** — Intermediate pipeline activations wire-quantized ~3.76× (`activation_compression`, on by default)
- **Flash Attention** — CPU and GPU fast paths (GQA-native, no `repeat_kv`)
- **PagedAttention** — Not implemented. A vendored kernel crate (`vendor/candle-paged-attention`) is in the workspace, but nothing references it and there is no feature flag for it (#257)
- **Logprobs** — NOT returned by local inference. The sampler can compute them (`sample_token_with_logprobs`) and the response type serializes them, but every local execution path pins `token_logprobs: vec![]`, so nothing reaches the response. `/v1/chat/completions` therefore REFUSES `logprobs` for a locally-served model rather than answering 200 with the field absent, which is indistinguishable from a request that never asked. Cloud-routed models still return them. See `docs/ARCHITECTURE.md` § Deferred Items
- **No failure broadcast** — A failed request is not announced to the swarm. It used to be, but receivers only logged it, and its text named the model, which told every peer what had failed here, in private mode too (#251). Receivers still accept the message from older nodes
- **Local Embedding Privacy** — When `local_embedding_privacy: true`, the requesting node performs token→embedding locally (~1ms) and sends pre-embedded hidden-state activations instead of raw token IDs to the first pipeline segment. Remote nodes never receive the prompt as text or token IDs, but the first segment is sent rows of the model's own embedding table, which it maps back to the tokens exactly — this keeps token IDs off the wire, it does not hide the prompt. See [Security > Local Embedding Privacy](../architecture/security.md#local-embedding-privacy)
- **Start and finish on this computer** (`encrypted_pipeline`, "boomerang") — When enabled (per-model or global), the requesting node runs both the first segment (embedding) and the last (token sampling), so no other computer is handed the prompt as text or picks the reply's words. This is structural, not cryptographic: the middle computers still compute on hidden states in plaintext, which published attacks turn back into much of the text. See [Security > Start and Finish on This Computer](../architecture/security.md#start-and-finish-on-this-computer-boomerang)

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
