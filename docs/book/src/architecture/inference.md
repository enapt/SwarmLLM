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
2. **Pipeline affinity check**: if multi-turn session has a previous pipeline and all nodes are still connected, reuse it (KV cache locality)
3. Query model_registry.shard_holders for hosting nodes
4. Fetch node load/latency from peer_registry
5. Sort candidates by (latency ASC, load ASC, trust DESC)
6. Greedy assignment: widest contiguous layer range per node
7. Merge contiguous segments on same node
8. Identify standby nodes per segment (failover)
9. Send PipelineAssignment, wait for ACKs, begin forwarding

Pipeline affinity means that multi-turn conversations (with `session_id`) prefer to route through the same nodes, preserving KV-cache state and avoiding cold restarts on every turn.

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

## Advanced Features

- **Batched Inference** — `BatchForwarder` stacks concurrent decode requests into GPU batches
- **Speculative Decoding** — Draft model proposes K tokens, target verifies in one pass
- **Chunked Prefill** — Long prompts split into chunks to reduce peak memory
- **Flash Attention** — CPU and GPU fast paths (GQA-native, no `repeat_kv`)
- **PagedAttention** — Block-pool KV-cache allocation (CUDA-only, `paged-attn` feature)
- **Logprobs** — Per-token log probabilities via `sample_token_with_params_and_logprobs()`. When `logprobs: true` in the request, the sampling layer collects top-N token probabilities and returns them in the OpenAI-compatible response. Available on split model (candle) inference paths
- **Pipeline Error Broadcast** — On distributed inference failure, `broadcast_pipeline_error()` notifies all participants so peers can update shard availability and route around failures
- **Local Embedding Privacy** — When `local_embedding_privacy: true`, the requesting node performs token→embedding locally (~1ms) and sends pre-embedded hidden-state activations instead of raw token IDs to the first pipeline segment. Remote nodes never see the plaintext prompt. See [Security > Local Embedding Privacy](../architecture/security.md#local-embedding-privacy)

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
