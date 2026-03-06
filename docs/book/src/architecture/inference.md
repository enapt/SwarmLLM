# Inference Pipeline

## Split Inference Engine

The split inference engine (`src/inference/split.rs`) enables distributed inference using candle for direct tensor computation with quantized GGUF weights. Each node loads only its assigned transformer layers, forwarding hidden-state activations between nodes.

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
2. Query model_registry.shard_holders for hosting nodes
3. Fetch node load/latency from peer_registry
4. Sort candidates by (latency ASC, load ASC, trust DESC)
5. Greedy assignment: widest contiguous layer range per node
6. Merge contiguous segments on same node
7. Identify standby nodes per segment (failover)
8. Send PipelineAssignment, wait for ACKs, begin forwarding

## Architecture Detection

The SplitModel loader reads `general.architecture` from GGUF metadata:

| Feature | Llama | Qwen2 |
|---|---|---|
| RoPE variant | Interleaved (`rope_i`) | Contiguous (`rope`) |
| QKV biases | None | Present (broadcast_add) |
| Context length | 4096 default | 32768 from metadata |

## KV-Cache Management

- Per-request isolation via `DashMap<(ModelKey, RequestId), Cache>`
- Multi-turn reuse: `session_id` tracks conversations, prefix matching skips redundant prefill
- Configurable TTL (default 10 min)
- VRAM-aware LRU eviction for split model cache

## Advanced Features

- **Batched Inference** — `BatchForwarder` stacks concurrent decode requests into GPU batches
- **Speculative Decoding** — Draft model proposes K tokens, target verifies in one pass
- **Chunked Prefill** — Long prompts split into chunks to reduce peak memory
- **Flash Attention** — CPU and GPU fast paths (GQA-native, no `repeat_kv`)
- **PagedAttention** — Block-pool KV-cache allocation (CUDA-only, `paged-attn` feature)

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
