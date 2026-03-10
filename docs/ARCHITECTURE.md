# SwarmLLM Architecture Reference

## Workspace Structure

Cargo workspace with three crates:

| Crate | Path | Purpose |
|-------|------|---------|
| `swarmllm` | `/` (root) | Main binary — daemon, networking, inference, API, all subsystems |
| `swarmllm-types` | `crates/swarmllm-types/` | Shared data types (69 types: NodeId, ModelManifest, SwarmMessage, etc.) |
| `swarmllm-frontend` | `crates/swarmllm-frontend/` | Frontend asset serving (embedded in release, disk-based in dev mode) |

Extension traits (`ModelManifestExt`, `NicknameRecordExt`, `BlindedPoolInvitationExt`) provide methods for types in `swarmllm-types` that depend on main crate functionality (filesystem, crypto, blake3).

## System Overview

Single Rust binary, three simultaneous functions:

```
┌──────────────────────────────────────────────────────────┐
│                      swarmllm binary                      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │       │
│  │  Node    │  │  Server      │  │  (embedded)  │       │
│  │(TCP+QUIC)│  │  (Axum)      │  │              │       │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘       │
│       │               │                 │                │
│  ┌────┴───────────────┴─────────────────┴─────────────┐  │
│  │              Shared State (Arc)                     │  │
│  │  DashMap<NodeId, PeerInfo>      — peer registry     │  │
│  │  ModelRegistry                  — models + shards   │  │
│  │  DashMap<Uuid, PipelineAssignment> — pipelines      │  │
│  │  Arc<RwLock<CreditBalance>>     — credit balance    │  │
│  │  RwLock<NodeStats>              — node statistics    │  │
│  │  SharedExecutor                 — llama.cpp model    │  │
│  │  DashMap<Blake3Hash, VoteTally> — model votes       │  │
│  │  DashMap<NodeId, NicknameRecord>— nickname registry │  │
│  │  DashMap<PoolId, PoolState>     — pool registry     │  │
│  │  DashMap<String, AcqProgress>   — download progress │  │
│  │  AtomicBool                     — model_loaded flag  │  │
│  │  AtomicBool                     — is_ready flag      │  │
│  │  AtomicU64                      — inference_requests  │  │
│  │  Notify                         — queue drain signal │  │
│  │  DashMap<ModelId, CancelFlag>   — download cancels   │  │
│  │  TrustManager                   — peer trust scores  │  │
│  │  watch::Sender<OperationalParams> — config reload    │  │
│  │  DashMap<ModelId, AtomicU64>    — request counts     │  │
│  │  RwLock<ResourceSchedule>       — resource schedule  │  │
│  │  broadcast::Sender<PruneEvent>  — prune events       │  │
│  │  broadcast::Sender<()>          — models changed      │  │
│  │  RwLock<VecDeque<PruneEvent>>   — prune history      │  │
│  │  DashMap<ShardId, bool>         — locked shards      │  │
│  │  DashMap<Uuid, oneshot::Sender>  — pending vision    │  │
│  │  DashMap<ModelId, VisionModule>  — vision modules    │  │
│  │  DashMap<ModelId, ModelTrustInfo> — model trust      │  │
│  │  DashMap<ModelId, Notify>       — loading models    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Daemon Task Architecture

```
                           ┌──────────────┐
                           │   daemon/    │
                           │  (bootstrap) │
                           └──────┬───────┘
                                  │ spawns tokio tasks
  ┌───────┬───────┬───────┬───────┼───────┬──────────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼       ▼       ▼       ▼          ▼          ▼          ▼          ▼
┌──────┐┌─────┐┌─────┐┌──────┐┌──────┐┌──────┐┌────────┐┌────────┐┌──────┐┌────────┐
│Netwrk││Infer││Crdit││Health││ API  ││Rebal-││Acquisi-││Message ││ Pool ││AutoShrd│
│Mangr ││Routr││Ledgr││Mon.  ││Servr ││ancer ││tion Mgr││Dispatc ││Mangr ││Manager │
└──┬───┘└──┬──┘└──┬──┘└──┬───┘└──┬───┘└──┬───┘└───┬────┘└───┬────┘└──┬───┘└───┬────┘
   │       │      │      │       │       │         │         │        │        │
   └───────┴──────┴──────┴───────┴───────┴─────────┴─────────┴────────┴────────┘
                              mpsc channels between tasks
```

### Channel Layout

| From | To | Channel | Message Types |
|---|---|---|---|
| NetworkManager | MessageDispatcher | `network_out_tx` | AuthenticatedMessage (transport-verified sender + SwarmMessage) |
| MessageDispatcher | InferenceRouter | `router_cmd_tx` | InferenceRequest, LayerForward, LayerResult, PipelineAssignment, InferenceError |
| InferenceRouter | NetworkManager | `network_tx` | SwarmMessage (outgoing P2P) |
| HealthMonitor | NetworkManager | `network_tx` | HealthPing |
| HealthMonitor | ShardRebalancer | `rebalance_tx` | RebalanceEvent |
| ApiServer | InferenceRouter | `router_cmd_tx` | RouterCommand (from HTTP) |
| ApiServer | AcquisitionManager | `acquisition_tx` | AcquisitionCommand (model download) |
| ApiServer | PoolManager | `pool_cmd_tx` | PoolCommand (pool CRUD) |
| AcquisitionManager | NetworkManager | `network_tx` | ShardAnnounce (shard requests) |
| CreditLedger | NetworkManager | `network_tx` | CreditGossip, CreditTransaction |
| PoolManager | NetworkManager | `network_tx` | PoolInvitation, PoolState gossip |
| AutoShardManager | AcquisitionManager | `acquisition_tx` | AcquisitionCommand (auto downloads) |

The **MessageDispatcher** is a dedicated task in `daemon/dispatch.rs` that routes inbound network messages to the appropriate subsystem. Inference messages go to InferenceRouter, CreditGossip updates peer balance distributions, ModelVote is processed by the model governance module, and pool messages go to PoolManager.

## Startup Sequence

```
1.  Parse CLI args (clap) — including optional --model and --gpu-layers
2.  Initialize tracing subscriber (verbosity: info → debug → debug+libp2p → trace)
3.  Load or create config (TOML + env + defaults + CLI overrides)
4.  Ensure data directory exists
5.  Load or generate Ed25519 identity
6.  Open redb database (auto-migrates from sled if `migrate-sled` feature enabled)
7.  Build Daemon { config, identity, db }
8.  Initialize ModelExecutor (load GGUF model if --model provided)
9.  Build Arc<SharedState> (includes ModelRegistry loaded from DB)
10. Scan local shards → register in model_registry (with disk existence verification)
11. Create mpsc channels (network, router, rebalance, acquisition, pool)
12. Spawn all tasks (10 tasks: NetworkManager, InferenceRouter, MessageDispatcher,
    HealthMonitor, ShardRebalancer, CreditLedger, AcquisitionManager, ApiServer,
    PoolManager, AutoShardManager)
13. Open browser if ui.open_browser_on_start is true (setup wizard or admin)
14. tokio::select! on Ctrl+C signal or any task exit
15. Signal graceful shutdown via watch channel, save peer cache, flush redb database
```

## Peer Discovery

SwarmLLM uses a 5-layer zero-config discovery stack. Each layer is independent — losing any layer doesn't break the others.

```
┌─────────────────────────────────────────────────────────────┐
│                     Discovery Stack                          │
│                                                             │
│  Layer 1: mDNS (LAN)                                        │
│    Toggle-wrapped libp2p mdns — discovers peers on same      │
│    network in seconds. Config: enable_mdns = true (default) │
│                                                             │
│  Layer 2: Persistent Peer Cache (redb)                       │
│    Saves up to 200 peer multiaddrs every 5 min + shutdown   │
│    Loads on startup → fastest reconnect path                │
│    File: src/network/peer_cache.rs                          │
│                                                             │
│  Layer 3: Encrypted Network Invite Codes                      │
│    Format: swarm://<base64url(key‖nonce‖encrypted_addr)>   │
│    Encryption: ChaCha20Poly1305 (IP not visible in code)    │
│    API: GET /api/admin/network-code                         │
│          POST /api/admin/join-network                       │
│                                                             │
│  Layer 4: Peer Exchange (PEX) + RTT Measurement              │
│    On each ConnectionEstablished, exchange up to 20 known   │
│    peer addresses. Uses request_response channel.           │
│    RTT measured on PEX request/response round-trip.         │
│    RTT < 5ms → auto-detect as LAN peer (enables TP).       │
│                                                             │
│  Layer 5: Kademlia DHT + Bootstrap                           │
│    Existing: --bootstrap flag, Kademlia re-bootstrap 60s    │
│                                                             │
│  Anti-Gaming: Subnet Clustering Detection                    │
│    Tracks /24 IPv4 prefixes. >5 nodes per /24 → 25%        │
│    spot-check rate (up from 5%). SubnetClustering trust     │
│    event penalty (-0.03).                                   │
│                                                             │
│  Gossip Network ID: "swarmllm-mainnet-v1" (fixed)           │
│    Configurable via gossip_network_id for private networks  │
└─────────────────────────────────────────────────────────────┘
```

### Discovery Startup Sequence

```
1. Listen on TCP (port+10, Noise+Yamux) and QUIC (port)
2. Subscribe to GossipSub topics
3. mDNS starts immediately (discovers LAN peers in seconds)
4. Dial cached peers from last session (fastest reconnect)
5. Dial user-configured --bootstrap peers and invite codes (if any)
6. Trigger Kademlia bootstrap
7. PEX fires on each ConnectionEstablished (exchanges peer lists)
8. Periodic: Kademlia re-bootstrap every 60s, peer cache save every 5min
9. On shutdown: save peer cache
```

## Networking Stack

```
libp2p Swarm
├── Kademlia (DHT)
│   ├── /swarm/node/{node_id}         → NodeCapability
│   ├── /swarm/shard/{model}/{index}  → Vec<NodeId> (batched per model)
│   └── /swarm/model/{model_id}       → ModelManifest
│   └── Records expire after 1 hour, re-published periodically
│
├── GossipSub (pub/sub, mesh_outbound_min=1)
│   ├── swarm/models/{model_id}       → ShardAnnounce, capacity
│   ├── swarm/governance              → ModelVote
│   ├── swarm/health                  → trust summaries
│   ├── swarm/identity                → NicknameRecord (signed, timestamp-checked)
│   └── swarm/pools                   → PoolState, PoolInvitation
│   └── Messages >5 min old are rejected (replay protection)
│   └── Failed publishes buffered and replayed on mesh formation
│
├── request_response (unified protocol, /swarmllm/1.0.0, 300s timeout)
│   ├── JSON control messages — SwarmMessage, ShardRequest/ShardResponse
│   └── Binary tensor payloads — LayerForward, LayerResult (type-tag byte: 0x00=JSON, 0x01=tensor, zstd compression optional)
│
├── TCP transport (Noise + Yamux, nodelay=true, port+10)
├── QUIC transport (port, fallback for NAT traversal)
├── mDNS (optional, LAN peer discovery — conditional dial, not added to Kademlia)
├── connection_limits (max 1/peer, 500 total)
├── Identify (protocol identification + peer_to_node reverse map)
├── AutoNAT (NAT detection → Kademlia Mode::Client/Server switch)
├── DCUtR (hole punching)
└── relay::client (circuit relay)
```

## Inference Pipeline

### Split Inference Engine

The split inference engine (`src/inference/split.rs`) enables true distributed inference
using candle for direct tensor computation with quantized GGUF weights. Each node loads
only the transformer layers it owns, forwarding hidden-state activations between nodes.

```
Client → API Server → InferenceRouter → Pipeline Assembly
                                              │
                      ┌───────────────────────┘
                      ▼
          ┌──────────────────────┐
          │   Pipeline Segment   │     Token IDs (prefill) or
          │ Node A: Layers 0-15  │     single token ID (decode)
          │ (embedding + layers) │──── LayerForward ───▶
          └──────────────────────┘                      │
                                        ┌───────────────┘
                                        ▼
                            ┌──────────────────────┐
                            │   Pipeline Segment   │   Hidden states
                            │ Node B: Layers 16-27 │   [1, seq, 3584]
                            │ (layers + norm + LM)  │── sample token ──▶
                            └──────────────────────┘                    │
                                                       ┌────────────────┘
                                                       ▼
                                                  LayerResult
                                                  (token IDs)
                                                       │
                                                       ▼
                                                    Client
```

### Architecture-Aware Model Loading

The SplitModel loader detects the model architecture from GGUF metadata
(`general.architecture`) and applies architecture-specific behavior:

| Feature | Llama | Llama 4 | Qwen2 | Qwen 3.5 | Gemma/Gemma2 | Phi-3 | Mistral | Starcoder2 | DeepSeek-V2/V3 | GLM-4 |
|---------|-------|---------|-------|-----------|--------------|-------|---------|------------|----------------|-------|
| RoPE variant | Interleaved (`rope_i`) | Contiguous (iRoPE) | Contiguous (`rope`) | Partial (25% head_dim) | Interleaved | Su/YaRN | Interleaved | Contiguous | Contiguous (MLA split) | Contiguous (partial) |
| QKV biases | None | None | Yes | Yes | None | Yes | None | Yes | None (MLA projections) | Yes |
| Attention | Standard MHA/GQA | Standard GQA | Standard MHA | Standard + output gate | Standard MHA | Standard MHA | Standard GQA | Standard MHA | MLA (low-rank Q/KV) | Extreme GQA (16:1) |
| FFN | Dense | Dense + MoE (mixed) | Dense | Dense | Dense | Dense | Dense | Dense | MoE (top-k) + shared | Dense |
| Context length | 4096 (default) | 131072 | 32768 | 131072 | 8192 | 4096 | 32768 | 16384 | 163840 | 131072 |
| Special | — | NoPE every 4th layer | — | Hybrid SSM+attention | Embedding scaling (sqrt(d)), Gemma RmsNorm (+1), attn + final logit softcap, EOS 107, Gemma chat template | Fused QKV/FFN | — | — | Per-layer dense/MLA | Partial RoPE (50%) |
| E2E verified | ✅ | — | ✅ | — | ✅ (Gemma2) | ✅ | — | — | — | — |

> **Phi-3 fused tensors**: Phi-3 GGUF models store `attn_qkv.weight` (Q+K+V concatenated) and `ffn_up.weight` (gate+up concatenated, no `ffn_gate.weight`). The loader dequantizes on CPU, splits by head dimensions, and re-quantizes to Q4_0 on the target device.

### DeepSeek-V2/V3 MoE + MLA Support

DeepSeek models use two specialized mechanisms that differ from standard transformers:

**Multi-head Latent Attention (MLA)** — compressed KV via low-rank projections:
- Q path: `x → q_a (compress) → RMSNorm → q_b (decompress) → split (q_nope, q_rope) → RoPE on q_rope`
- KV path: `x → kv_a (compress) → split (c_kv, k_rope) → RoPE on k_rope → RMSNorm(c_kv) → kv_b → split (k_nope, v)`
- Full K/V stored in KV cache (decompressed, not latent)
- Uses `standard_attention()` due to asymmetric key/value dimensions

**Mixture-of-Experts (MoE)** — router-selected sparse FFN:
- Router: `x.matmul(gate.T) → softmax → top-k selection (CPU argsort)`
- Expert loop: per-token top-k experts selected from stacked `[n_experts, dim, dim]` tensors via `index_select`
- Shared experts (always active) added to routed expert output
- Expert tensors dequantized at load time for `index_select` compatibility

**Per-layer type detection** — early DeepSeek layers (~1-3) use standard dense attention + dense FFN; subsequent layers use MLA + MoE. The `LayerVariant` enum handles both:
- `LayerVariant::Dense(LayerWeights)` — standard transformer layer (can use `FfnVariant::Dense` or `FfnVariant::MoE` for mixed models like Llama 4)
- `LayerVariant::DeepSeek { attention: MlaWeights, ffn: FfnVariant, ... }` — MLA + MoE/dense FFN

### Llama 4 Scout/Maverick Support

Llama 4 introduces two novel mechanisms within the standard dense `LayerVariant`:
- **iRoPE (interleaved RoPE)** — every 4th layer uses NoPE (no positional encoding), the rest use standard RoPE. This is handled by a per-layer flag in `LayerWeights`
- **Mixed Dense+MoE FFN** — `FfnVariant` enum (`Dense(Mlp)` | `MoE(MoeFfn)`) allows individual layers to use either dense or MoE FFN within the same model. Top-k expert routing reuses the same `MoeFfn` struct as DeepSeek

### Qwen 3.5 Hybrid SSM+Attention Support

Qwen 3.5 introduces a hybrid architecture combining SSM (Gated Delta Networks) with standard attention:

- **Layer pattern**: 3 SSM (DeltaNet) layers + 1 full attention layer per 4-layer group
- **GGUF arch strings**: `"qwen35"` (dense), `"qwen35moe"` (MoE variant)
- **SSM forward**: conv1d → delta_net_scan (recurrent) → gated_norm → output projection
- **Attention layers**: Standard attention with sigmoid output gate + partial RoPE (25% of head_dim)
- **State management**: `SsmState` (conv_state + recurrent_state) alongside KV-cache for attention layers
- **Per-layer detection**: SSM vs attention determined by presence of `ssm_alpha.weight` tensor in GGUF

### Tensor Parallelism (AllReduce)

When multiple LAN nodes hold the same shards, tensor parallelism splits computation within each layer across nodes:

```
Node A (rank 0, coordinator)          Node B (rank 1)
┌─────────────────────┐              ┌─────────────────────┐
│ Load full weights   │              │ Load full weights   │
│ Slice heads 0..N/2  │              │ Slice heads N/2..N  │
│ forward_attn_tp()   │              │ forward_attn_tp()   │
│ forward_tp() FFN    │              │ forward_tp() FFN    │
│ partial output      │              │ partial output      │
└────────┬────────────┘              └────────┬────────────┘
         │                                     │
         └──────────┐    ┌────────────────────┘
                    ▼    ▼
              AllReduce (star topology)
              Coordinator sums partials
              Broadcasts reduced tensor
                    │
              ┌─────┴─────┐
              ▼           ▼
         Node A        Node B
         (continue to next layer)
```

- **Topology**: Star AllReduce — rank 0 collects partials, element-wise sums, broadcasts result
- **LAN detection**: Auto-detected via PEX RTT measurement (< 5ms → `is_lan_peer = true`)
- **TP group formation**: Requires `is_lan_peer` OR measured `latency_ms ≤ 10`
- **Weight splitting**: Dynamic slicing at inference time (`forward_attn_tp` slices attention heads, `forward_tp` slices FFN intermediate dimension)
- **Wire format**: Partials zstd-compressed, sent via `SendAllReduceRequest` / `SendAllReduceResponse` NetworkCommand variants
- **Registry cleanup**: `AllReduceRegistry::cleanup_stale()` runs on each HealthMonitor tick (30s), removing entries where the receiver was dropped (timed out)
- **Files**: `src/inference/allreduce.rs` (coordinator + registry), `src/inference/scheduler.rs` (TP group detection)

### Vision Language Models (VLM)

SwarmLLM supports multimodal inference via `src/inference/vision.rs`:
- **LLaVA** — CLIP vision encoder + LLM backbone, image patches projected into token space
- **Qwen2-VL** — Native vision-language architecture with dynamic resolution
- Images are pre-processed, encoded into vision tokens, and inserted at the `<image>` token position in the prompt (matching llama.cpp's approach: prompt split at `<image>`, before/after tokenized separately, vision embeddings inserted at exact position)
- **mmproj GGUF loading** — `load_from_mmproj_gguf()` loads CLIP ViT weights directly from llama.cpp-compatible mmproj GGUF files (verified with LLaVA-v1.5-7B mmproj: 577 vision tokens × 4096 LLM dim)
- **Status**: Full E2E verified — LLaVA-v1.5-7B: base64 image → CLIP vision encoder (577 tokens × 4096 dim) → position-aware embedding insertion at `<image>` → 7B text model → correct output. CPU-only (~4min prefill, ~1.8s/token)

#### Distributed mmproj

The mmproj (vision encoder, ~600MB) is modeled as a **sentinel shard** (`index = u32::MAX`) within the existing shard infrastructure. Vision encoding is a **pre-processing step** decoupled from the text pipeline — no single node needs both the vision encoder and a text shard.

```
API Request (with image)
    │
    ▼
Router: does any node have mmproj?
    │
    ├── Local node has mmproj → encode locally
    ├── Remote node has mmproj → VisionEncodeRequest → get embeddings back
    └── Nobody has mmproj → HTTP 503 (VisionEncoderUnavailable)
    │
    ▼
Pre-computed embeddings (577 × 4096 = ~9.4MB, zstd+FP16 compressed)
    │
    ▼
Text Pipeline (unchanged): embeddings travel with LayerForward
```

**Key design decisions:**
- Sentinel shard index (`u32::MAX`) reuses all ShardId infrastructure (registry, announcements, auto-manage, pruning)
- `VisionEncodeRequest` / `VisionEncodeResponse` network messages for remote encoding (JPEG-compressed images on wire)
- `LayerForward.vision_embeddings: Option<Vec<u8>>` carries zstd-compressed FP16 embeddings on first forward
- Vision node selection: prefer local → first-segment node → any mmproj holder
- `precompute_vision_embeddings()` runs once before the token generation loop
- Auto-manage: 5x priority bonus for mmproj download, higher pruning floor (min 3 replicas), only prunes under extreme pressure (>0.95)

### BPE Tokenizer

A full GPT-2/Qwen2 BPE tokenizer is built from GGUF metadata at model load time:
- Vocabulary from `tokenizer.ggml.tokens`
- Merge rules from `tokenizer.ggml.merges`
- Pre-tokenization regex from `tokenizer.ggml.pre` (model-specific: qwen2, gpt2, default)
- GPT-2 byte encoding/decoding for proper UTF-8 handling

### Tensor Wire Format

Hidden states are serialized for network transmission:
```
[4B ndim][4B×ndim shape][4B dtype_tag][f32 data]
```

For a 7B model (hidden_dim=3584):
- Prefill (14 tokens): 1×14×3584×4 = ~200KB
- Decode (1 token): 1×1×3584×4 = ~14KB

**Tensor Compression** — optional zstd compression for wire tensors (configurable):
- `tensor_compression = true` — enable zstd compression on hidden-state payloads
- `tensor_compress_level = 3` — zstd compression level (1-22, default 3)
- `tensor_compress_threshold = 4096` — minimum payload bytes to trigger compression
- Reduces bandwidth for prefill payloads by 30-60% with minimal latency overhead

### Pipeline Assembly Algorithm

1. Fetch model manifest → determine layer ranges
2. Query model_registry.shard_holders for hosting nodes
3. Fetch node load/latency from peer_registry
4. Sort candidates by (latency ASC, load ASC, trust DESC)
5. Greedy assignment: widest contiguous layer range per node
6. Merge contiguous segments assigned to the same node
7. Identify standby nodes per segment
8. Send PipelineAssignment → all nodes ACK → begin forwarding

### KV-Cache Management

- Per-request KV-cache isolation via `DashMap<String, KvCacheEntry>` (key: `"model_key\0request_id"`)
- Each concurrent request gets its own cache — no corruption under concurrency
- Multi-turn reuse: session_id tracks conversations, prefix matching skips redundant prefill
- KV-cache is cleared when `sequence_num == 0` (start of new request)
- `index_pos` travels through the wire protocol so all nodes apply correct RoPE positioning
- Position tracking: `index_pos = prompt_token_count` after prefill, increments by 1 per decode step
- KvCacheManager tracks sessions and wired to inference router for cache reuse
- Causal masks cached with LRU eviction (max 16 entries) to prevent GPU memory leak
- Abandoned cache entries cleaned up after 10 minutes
- Sessions persisted across node restarts via redb

### Prefix Caching

Cross-request prefix caching (`src/inference/prefix_cache.rs`) shares KV entries for common system prompts:
- Trie-based prefix matching identifies shared token prefixes across requests
- Matching entries skip redundant prefill computation (50-80% reduction for shared system prompts)
- Configurable max entries via `prefix_cache_max_entries` (default 256)
- LRU eviction when cache is full

### Chunked Prefill

Long prompts are split into chunks for overlapped prefill and decode:
- Prevents head-of-line blocking from long-context requests
- Decode steps for other requests can interleave between prefill chunks
- Chunk size auto-tuned based on available VRAM

### Speculative Decoding

- Draft model (small/fast) proposes K candidate tokens per step (default 4)
- Target model verifies all K in one forward pass (amortized GPU cost)
- Rejection sampling ensures output distribution identical to non-speculative
- KV-cache resynchronization on rejection (trim + reseed)
- Config: `speculative_decoding`, `speculative_gamma`, `draft_model_path`
- Falls back to standard decoding if no draft model available

### Batched Inference (Pipeline Bubble Filling)

- `BatchForwarder` collects concurrent decode-step requests into GPU batches
- Position-independent ops (norms, MLP) run on stacked `[batch, seq, dim]` tensors
- Attention runs per-request (different KV-caches and positions)
- Output split back via `Tensor::narrow` per request
- Prefill and single-item batches use sequential path
- **Both local and remote requests** route through `BatchForwarder` — remote `LayerForward` requests (from `handle_layer_forward` in `daemon/dispatch.rs`) submit to the same batch queue as local pipeline segments, filling pipeline bubbles where a node would otherwise sit idle waiting for upstream/downstream nodes
- Timeout-based batch collection: when a request arrives and fewer than `max_batch_size` items are queued, the processor waits up to `batch_timeout_ms` for more requests before dispatching
- `SplitModelEntry` caches `eos_tokens: Vec<u32>` at construction, enabling lock-free sampling after batched forward passes (no model mutex needed for EOS detection)
- Config: `max_batch_size` (default 1 = no batching), `batch_timeout_ms` (default 50ms)

### VRAM-Aware Cache Eviction

- `SplitModelEntry` wraps models with `last_used` timestamp and `estimated_vram_mb`
- Configurable `max_split_model_memory_mb` budget (default unlimited)
- LRU eviction: least-recently-used models evicted when over budget
- Active models (with in-flight pipelines) are never evicted

### LoRA Adapter Support

LoRA (Low-Rank Adaptation) adapters are supported via `src/model/lora.rs`:
- Per-request adapter loading from safetensors files
- Low-rank weight updates applied at inference time without modifying base model weights
- Multiple adapters can be loaded simultaneously and selected per request
- Adapter files stored alongside model shards in the model directory
- **Verified** with Qwen2.5-Coder-7B + rank-16 LoRA adapter; output distribution changes confirmed

## Credit System

```
Earning (default rates, configurable per pool):
  +10 credits  per layer per token served
  +1  credit   per GB per hour hosting shards
  +5  credits  per GB seeding shard data
  +2  credits  per connection hour relay service

Spending (default rates, configurable per pool):
  -8  credits  per layer per token requested
  -50 credits  per distributed inference failure (automatic penalty)

Tiers (enforced per-request in InferenceRouter):
  Platinum  (≥90th percentile)  → immediate queue, 2× concurrent slots
  Gold      (≥70th percentile)  → 1-3s queue, base concurrent slots
  Silver    (positive balance)  → 5-15s queue, ½ concurrent slots
  Bronze    (zero/negative)     → 30s+ queue, ¼ concurrent slots
```

**Tier enforcement flow**: On each `handle_submit()`, the router computes the network percentile
from `peer_credit_balances` (populated via credit gossip), calls `calculate_tier()`, and sets the
request priority. In `drain_queue()`, `max_concurrent_for_tier()` limits how many concurrent
execution slots each tier can use. Higher tiers dequeue first via `tier_weight()` ordering.

**Relay credits**: NetworkManager tracks active relay circuits via `active_relay_circuits` DashMap.
On `CircuitReqAccepted`, records start time. On `CircuitClosed`, computes duration and adds to
`relay_seconds_served` atomic counter. CreditLedger drains this counter periodically and calls
`earn_relay_service()`.

**Failure penalties**: When distributed inference fails in `execute_request()`, the router applies
`penalty_serve_failure` credits (default -50) and broadcasts `InferenceError` to all pipeline
participants via `broadcast_pipeline_error()`.

Credit earn/spend rates are configurable per pool via the pool configuration API.

## Model Acquisition Security

Models enter the system through a verified pipeline — arbitrary files on disk are never
absorbed into the network.

```
Network Registry (GossipSub/DHT)
        │
        ▼
 ┌─────────────────┐     BLAKE3 hash
 │  Manifest Check │────────────────▶ Reject if tampered
 └────────┬────────┘
          │ verified manifest
          ▼
 ┌─────────────────┐     Rarest-first
 │ Shard Selection │────────────────▶ BitTorrent-style
 └────────┬────────┘
          │ request from peers (3 retries, exponential backoff)
          ▼
 ┌─────────────────┐     Atomic write to .tmp
 │  Download Loop  │────────────────▶ Rename to .bin on completion
 └────────┬────────┘
          │ complete shard
          ▼
 ┌─────────────────┐     BLAKE3 vs manifest (strict: no zero-hash bypass)
 │ Shard Verify    │────────────────▶ Quarantine + penalize on mismatch
 └────────┬────────┘
          │ all shards verified
          ▼
    Model Ready
```

**Key invariants**:
- Manifests MUST come from the network registry, not from disk
- Manifest integrity is verified (BLAKE3 self-hash) before trusting shard hashes
- DB-restored manifests are also hash-verified on startup
- Each downloaded shard is verified against the manifest hash (zero-hash bypass only for local HF downloads)
- Failed shards are renamed `.bin.quarantine` and the serving peer's trust is penalized
- Downloads are retried (3 attempts, 5s/30s/120s backoff) with alternate peer selection
- Atomic writes: shards written to `.tmp` then renamed, preventing corrupt partial files
- On startup, `load_all_local()` rejects model directories without a valid manifest
- On startup, every existing shard is re-verified against its manifest hash
- Stale `.tmp` files cleaned up on startup

**AcquisitionManager** (`src/model/acquisition.rs`) orchestrates this flow as a
long-running Tokio task, receiving commands via `mpsc` from the API server.

## Data Directory

```
~/.local/share/swarmllm/
├── config.toml          # User configuration
├── identity.key         # Ed25519 keypair (optionally encrypted)
├── db.redb              # redb database (migrated from sled db/ directory)
└── models/
    ├── llama3-70b-q4km/
    │   ├── manifest.json
    │   ├── tokenizer.json
    │   ├── shard_000.bin
    │   └── ...
    └── mistral-7b-q5km/
        └── ...
```

## redb Database Tables

Storage backend is **redb** (pure-Rust, ACID, single-file). The legacy **sled** backend is available behind the `migrate-sled` feature flag for one-time migration.

| Table | Key | Value |
|---|---|---|
| config | "config" | Config |
| config | "api_key" | String (32-byte hex Bearer token) |
| identity | "keypair" | Encrypted Ed25519 key |
| credits | "balance" | CreditBalance |
| credit_txns | {uuid} | CreditTransaction |
| peer_trust | {node_id_hex} | TrustScore |
| peer_cache | {multiaddr_string} | () (presence key) |
| shard_meta | {model_id}/{shard_index} | ShardInfo + path |
| model_meta | {model_id} | ModelManifest |
| sessions | {session_id} | KV-cache metadata (persisted across restarts) |
| nicknames | {node_id_hex} | NicknameRecord |
| identity_prefs | "nickname" | Local nickname preference |
| pool_state | "pool" | PoolState |
| pool_forwards | {uuid} | PoolCreditForward |
| trust_scores | {node_id_hex} | f64 trust score |
| escrow | {escrow_id} | EscrowEntry |
| hf_sources | {model_id} | HfSource metadata |
| locked_shards | {shard_id_json} | bool (presence = locked) |
| resource_schedule | "current" | ResourceSchedule JSON |

## Auto-Manage Shards

The **AutoShardManager** (`src/model/auto_manage.rs`) is a background subsystem that
automatically acquires missing shards based on a VRAM-aware scoring algorithm.

### Scoring Formula

```
score = model_popularity × rarity_bonus × configured_bonus × vram_fitness
```

| Factor | Value | Description |
|--------|-------|-------------|
| `model_popularity` | 1.0+ | Number of peers hosting any shard of the model |
| `rarity_bonus` | 1.0–10.0 | Fewer existing holders → higher priority |
| `configured_bonus` | 1.0 or 100.0 | 100x for shards within `--shards` range |
| `vram_fitness` | 0.1–1.0 | Model VRAM needs vs. global pool VRAM capacity |

### Key Behaviors

- **Configured range focus**: When any shards in the `--shards` range are missing,
  `candidates.retain()` filters to ONLY those shards (ignores others)
- **Disk verification**: Registration checks that shard files actually exist on disk
  (both at startup and in `generate_and_register_local_manifest`)
- **VRAM estimation**: `model_size × 1.15` (quantized weights + ~15% KV-cache overhead)
- **nvidia-smi fallback**: If `gpu_info` is None, falls back to `nvidia-smi` for local VRAM
- **Budget limits**: max_storage_mb, max_shards_per_cycle (2), skips in-progress acquisitions
- **mmproj support**: Vision encoder (mmproj.gguf) treated as download candidate with 5x priority bonus; full-file HF download (not byte-range); higher pruning floor (min 3 replicas), only pruned under extreme pressure (>0.95)
- **Config**: `[auto_manage]` section — `enabled`, `max_storage_mb`, `interval_minutes`, `max_shards`, `prune_enabled`, `min_replicas`, `prune_cooldown_secs`, `max_holder_load_for_prune`

### Smart Shard Pruning

When auto-manage is enabled and `prune_enabled = true`, the AutoShardManager also removes
over-replicated shards to free VRAM and disk on smaller nodes.

**Dynamic Target Replicas** — popularity-scaled based on per-model request counts (rolling 10-min window):
- 0 requests → base target (min_replicas, default 2)
- 1-10 requests → 1.5x base
- 11-50 requests → 2.0x base
- 51+ requests → 3.0x base

**Prune Scoring** (highest score pruned first):
```
+ redundancy_ratio (holder_count / target)
+ 1.0 if not loaded in VRAM (cold shard)
+ 0.5 × resource_pressure
- 0.5 if first/last shard (pipeline completeness)
- 0.3 if rarest shard for the model
- 0.2 if recently acquired (< 30 min)
```

**Safety Checks** — pruning is blocked if:
- Shard is locked/pinned by user
- Shard is in configured `--shards` range
- `holder_count <= adjusted_target_replicas`
- Would eliminate last holder in this node's region
- Average remaining holder load > `max_holder_load_for_prune`
- Model actively loaded and used in last 5 minutes
- No re-acquisition path available (no HF source or reachable peers)
- Cooldown not expired (5 min per model)

**Resource Pressure** — `max(disk_pressure, vram_pressure)`:
- VRAM pressure uses live `nvidia-smi` query (every 5 min tick) for actual GPU memory usage, with fallback to internal loaded-model tracking when nvidia-smi is unavailable
- < 0.5: relaxed (+1 to target, keep extras)
- 0.5–0.8: normal
- 0.8–0.95: eager (-1 from target)
- \> 0.95: urgent (-2, prune up to 2 shards/model/cycle)

**Resource Schedule** — configurable via API and UI, adds pressure bonus during reduced hours:
- "aggressive" → +0.3 pressure during reduced hours
- "normal" → +0.15 pressure
- "conservative" → no extra pressure

**Per-Model Control** — `PUT /api/admin/models/:id/auto-manage` with `prune_enabled: false` disables pruning per-model while keeping downloads active.

**Per-Shard Lock** — `PUT /api/admin/models/:id/shards/:index/lock` pins individual shards, preventing auto-pruning regardless of model-level settings.

**Notifications** — prune events pushed via WebSocket (`prune_event` type), toast notifications in UI, prune history accessible via `GET /api/admin/prune-history`

### Model Trust System

Prevents trash models from polluting the network when auto-manage is enabled.

**Trust Levels** (progressive, ordered):
```
Discovered → Pinned → DemandVerified → NetworkPopular
```

- **Discovered**: Seen via gossip, no local data. Auto-manage ignores these.
- **Pinned**: User explicitly downloaded/approved. Auto-manage propagates.
- **DemandVerified**: Model has received ≥3 real inference requests. Auto-manage propagates.
- **NetworkPopular**: ≥3 unique holder nodes across the network. Highest priority.

**Auto-manage gate**: `gather_candidates()` skips models below `DemandVerified` unless `pinned_by_user = true`. This means a node will never auto-download shards for a model nobody has actually used.

**Trust transitions**:
- Gossip-discovered models start as `Discovered` (auto-created on first manifest registration)
- User downloads via HF browser → `Pinned` (persisted immediately to redb)
- 3rd inference request → `DemandVerified` (persisted on promotion)
- 3+ unique holder nodes → `NetworkPopular` (checked periodically by AutoShardManager)
- 7 days without requests → decay (`NetworkPopular` → `DemandVerified`, `DemandVerified` → `Discovered`)
- Pinned models never decay

**Persistence**: `model_trust` tree in redb, keyed by model_id, values are JSON `ModelTrustInfo`.

**API**: Trust level exposed as `trust_level` field in `GET /api/admin/models` response.

**Scaling**: Trust decisions are local per-node (no consensus needed). Each node independently decides what to download based on its own observed demand. This scales to thousands of nodes without coordination overhead.

### On-Demand Shard Loading

Models are loaded into VRAM only when needed, not eagerly at startup.

**Trigger**: When `execute_request()` in the inference router encounters a model that has shards on disk but no entry in `split_models`:
1. Check if `shard_000.bin` or `model.gguf` exists in the model directory
2. If yes, call `check_and_load_model()` inline (runs in the spawned inference task, not the router loop)
3. VRAM budget is checked; LRU eviction frees space from least-recently-used models
4. Loading coordination: `DashMap<ModelId, Notify>` ensures only one task loads a model at a time; concurrent requests wait on the Notify

**VRAM Budget**: Configured via `resources.max_gpu_vram_mb` or auto-detected (80% of GPU VRAM). LRU eviction protects active pipeline models from eviction.

**Startup behavior**: Models are still auto-loaded at startup (in popularity order), but this is best-effort — if VRAM fills up, remaining models stay on disk and are loaded on first request.

## E2E Encryption

```
┌─────────────────────────────────────────────────────────────┐
│                    Three Encryption Tiers                     │
│                                                             │
│  Tier 1: Pairwise Sessions (unicast)                        │
│    Ed25519 → X25519 → ECDH → ChaCha20-Poly1305            │
│    Forward secrecy: ephemeral X25519 re-keying every 10min  │
│    Nonce reuse prevented by session clearing on disconnect   │
│    Replay protection: RFC 6479 sliding window (128-bit      │
│      bitmap) — allows reordered packets, rejects duplicates │
│    Pending ephemeral keys expire after 60s (memory safety)  │
│    Static DH fallback for initial session before first reke │
│                                                             │
│  Tier 2: Pipeline Sealing (inference prompts)               │
│    Per-request ephemeral key → sealed prompt/response       │
│    Wire tag: TENSOR_TAG_ENCRYPTED = 0x10                    │
│    Active: final segment seals output tokens for requester  │
│    Intermediate nodes see activations, not plaintext output │
│                                                             │
│  Tier 3: Sealed Gossip (broadcasts)                         │
│    Mandatory Ed25519 signing — unsigned messages rejected   │
│    Epoch-based group key + origin signature                 │
│    Transport-authenticated sender validation in dispatch    │
│    1hr rotation cycle                                       │
│                                                             │
│  Modules: src/crypto/{session, pipeline_seal, gossip_seal,  │
│           key_rotation}.rs                                   │
└─────────────────────────────────────────────────────────────┘
```

## Transport-Authenticated Dispatch

All inbound network messages are wrapped in `AuthenticatedMessage` with the transport-verified sender `NodeId` (from libp2p Noise protocol). The MessageDispatcher validates sender identity against message claims for all security-sensitive message types (ShardAnnounce, CreditTransaction, CreditGossip, NicknameGossip, HealthPing/Pong, EphemeralKeyExchange). Mismatched messages are logged and dropped.

## Signed DHT Records

Kademlia DHT records for capability and shard announcements are Ed25519-signed:
- Format: `[32B pubkey][64B signature][payload]`
- Functions: `sign_dht_value()` / `verify_dht_value()` in `src/network/discovery.rs`
- Records expire after 1 hour with automatic re-publication

## Identity & Nicknames

- Ed25519-signed nickname records with timestamp-wins conflict resolution
- Timestamp freshness check: rejects records older than 1 hour or >5min in future
- GossipSub topic `swarm/identity` for network-wide propagation
- Collision handling: `nickname#ab12` suffix from node ID prefix
- redb tables: `"nicknames"`, `"identity_prefs"`

## Device Pools

- Dual-signature invitation/acceptance protocol (owner signs invite, member signs acceptance)
- Pool state gossip verifies each member's acceptance signature
- Member removal requires Ed25519-signed leave notice (prevents forged ejection)
- Credit forwarding: member inference earnings → `PoolCreditForward` (dual-signed) → `apply_credit_direct` to owner's balance
- Pool leaderboard aggregates member contributions
- Invitation expiry checked at API layer with clear error messages
- Config: max_pool_size=10, invitation_ttl_hours=24, rate_limit_per_hour=3

## Reputation & Trust

- `TrustManager` in `src/credit/trust.rs` tracks per-peer trust scores (0.0–1.0, default 0.5)
- Trust-affecting events: InferenceSuccess (+0.01), SpotCheckFail (-0.1), InvalidGossip (-0.05), ValidTransaction (+0.02), SignatureViolation (-0.2)
- Decay toward 0.5 over time (1% per health ping cycle) — prevents permanent punishment
- Persisted in redb `trust_scores` table, hydrated on startup
- Trust factors into pipeline scheduling and credit tier weighting

## Credit Escrow

- `EscrowManager` in `src/credit/escrow.rs` holds credits for large requests (> threshold)
- Lifecycle: `create_escrow()` → `release_escrow()` (success) or `refund_escrow()` (failure)
- Entries expire after 10 minutes with automatic refund
- Persisted in redb `escrow` table

## Sybil Resistance

- Balance reports are Ed25519-signed with timestamp freshness check (5 min window)
- Only signed reports accepted; unsigned reports rejected outright
- Stale/replayed reports rejected
- Subnet clustering detection: >5 nodes per /24 → elevated spot-check rate (25% vs 5%)
- SubnetClustering trust penalty (-0.03 per cycle while clustered)

## Credit System Security

- Transaction replay protection: UUID deduplication checked against DB before accepting
- Balance arithmetic uses `saturating_add` (no overflow/underflow panics)
- Priority tier calculation consistent between scheduler and display
- AntiGaming wired into credit flow: atomic check+record prevents TOCTOU
- Peer balance gossip rejects implausible values (abs > 100M)

## API Authentication

- Bearer token middleware in `src/api/middleware.rs` (constant-time comparison)
- Auto-generated 32-byte hex API key on first run, persisted in redb
- **Protected paths**: `/v1/*` (inference), `/api/admin/config` (PUT), `/api/admin/shutdown`,
  `/api/admin/hf/*` (downloads), `/api/admin/api-key`, `/api/admin/provider-models`
- **Exempt paths**: `/`, `/health`, `/admin`, `/chat`, `/setup`, `/static/*`,
  read-only admin dashboard endpoints (GET `/api/admin/stats`, `/api/admin/models`, etc.)
- Request body size limit: 32MB (configurable via `DefaultBodyLimit`, raised from 2MB for VLM image payloads)
- Content-Security-Policy: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`
- WebSocket Origin validation (prevents cross-site WebSocket hijacking)
- Input validation: model name 256 chars, tools max 128, stop sequences max 16
- HuggingFace inputs validated (repo_id format, filename .gguf extension, no path traversal)
- HTTP timeout: 5 minutes (tower-http TimeoutLayer, Slowloris protection)
- Per-IP rate limiter with periodic 5-minute cleanup of stale entries
- Inference queue depth cap: 512 requests

## Shard-Only Mode

Nodes can operate with just shard files + manifest.json + gguf_header.bin (~6MB),
without needing the full multi-GB GGUF file:

```
~/.local/share/swarmllm/models/qwen2.5-coder-7b/
├── manifest.json        # Model metadata + shard layout
├── gguf_header.bin      # First ~6MB of GGUF (metadata + tensor index)
├── shard_000.bin        # 512MB shard
├── shard_001.bin
└── ...
```

`ShardReader` in `split.rs` constructs a virtual GGUF from header + shard files,
allowing candle to parse the full tensor index while only loading assigned layers.

## HTTP API Routes

### OpenAI-Compatible (Bearer auth required)
- `POST /v1/chat/completions` — Chat completions (streaming + non-streaming, tool_calls + logprobs support)
- `POST /v1/messages` — Anthropic Messages API (full Claude Code compatibility — tools, tool_choice, thinking, cache_control, metadata)
- `POST /v1/embeddings` — Text embeddings
- `GET  /v1/models` — List available models
- `GET  /v1/providers` — List configured cloud providers and their available models
- `GET  /v1/status` — SwarmLLM node status
- `POST /v1/internal/hidden-states` — Extract hidden states from model layers (gated by `api.expose_hidden_states` config)

### Anthropic Messages API (`/v1/messages`)
Full Anthropic Messages API compatibility for use as a Claude Code backend:
- **Request fields:** `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking), `cache_control` on system blocks
- **Content blocks:** `text`, `image`, `tool_use`, `tool_result`, `thinking`, `redacted_thinking`
- **Routing:** Claude models → Anthropic cloud (full pass-through); non-Claude models → Anthropic→OpenAI translation proxy; local GGUF models → tool calls/thinking converted to text for inference
- **Claude Code usage:** `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`

### MCP Server (Protocol v2025-11-05)
- `POST /mcp` — JSON-RPC 2.0 MCP endpoint for AI agent frameworks (Claude Code, VS Code Copilot, Cursor, etc.)
- Tools: `chat`, `models`, `compare`, `research`, `batch_prompts`, `node_info`
- Resources: `swarmllm://status` (node status)
- All tools include [tool annotations](https://modelcontextprotocol.io/specification/2025-11-25) (`readOnlyHint`, `destructiveHint`, etc.)
- **`compare`:** sends the same prompt to up to 10 models concurrently, returns side-by-side results
- **`research`:** fan-out a question to multiple models (auto-selects if models omitted), returns all responses with token usage
- **`batch_prompts`:** execute up to 20 independent {id, model, prompt} tasks in parallel
- **`node_info`:** detailed node status (loaded model, peers, credits, registry models, cloud providers)

### Cloud Fallback (Optional)
When a requested model isn't available locally or on the swarm, requests can optionally be routed to 12 cloud providers:
- **Providers:** OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks AI, Together AI, DeepInfra, Moonshot/Kimi
- Model prefix routing: `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek, `mistral-*` → Mistral, `moonshot-*`/`kimi*` → Moonshot
- Explicit syntax: `provider:model` (e.g., `openai:gpt-4o`, `groq:llama-3.1-70b`)
- Custom providers via `[providers.custom]` config section
- Provider health probes with per-model availability checking
- Admin API: `GET/PUT /api/admin/providers` — view/configure provider API keys

### Admin API (CORS-protected, no Bearer auth)
- `GET/PUT /api/admin/config` — Configuration read/update
- `GET     /api/admin/stats` — Node statistics + hardware info
- `GET     /api/admin/models` — Model list with shard status, VRAM estimates, acquisition state
- `POST    /api/admin/models/:id/add` — Trigger model acquisition
- `GET     /api/admin/models/:id/status` — Model acquisition progress
- `GET     /api/admin/peers` — Connected peers with latency/trust
- `GET     /api/admin/credits` — Credit balance and tier info
- `GET     /api/admin/shard-storage` — Per-model storage breakdown, disk/VRAM usage
- `GET     /api/admin/api-key` — Retrieve API key (Bearer auth required)
- `GET     /api/admin/ws` — WebSocket for live updates
- `GET     /api/admin/download-queue` — Download queue with priorities and progress

### HuggingFace Integration
- `GET  /api/admin/hf/search?q=...` — Search HuggingFace for GGUF models (grouped by repo with quant variants)
- `GET  /api/admin/hf/probe?repo_id=...&filename=...` — Probe remote GGUF (size, shard layout)
- `POST /api/admin/hf/download` — Download full GGUF model
- `POST /api/admin/hf/download-shards` — Download specific shard indices (supports `peer_fair_share` for smart distribution)
- `GET  /api/admin/hf/source/:model_id` — Lookup HuggingFace source info for a model

### Identity API
- `GET/PUT/DELETE /api/identity/nickname` — Manage local nickname
- `GET           /api/identity/leaderboard` — Network-wide credit leaderboard
- `GET           /api/identity/peers` — Peer identity directory

### Pool API
- `GET  /api/pool/state` — Current pool membership state
- `POST /api/pool/create` — Create a new device pool
- `POST /api/pool/invite` — Invite a node to the pool
- `POST /api/pool/accept` — Accept a pool invitation
- `POST /api/pool/remove` — Remove a member from the pool
- `POST /api/pool/leave` — Leave the current pool
- `GET  /api/pool/invitations` — List pending invitations
- `GET  /api/pool/leaderboard` — Pool member contribution rankings

### Discovery
- `GET    /api/admin/network-code` — Get shareable invite code, multiaddr, and network phase
- `POST   /api/admin/join-network` — Join network via invite code or multiaddr

### Utility
- `POST   /api/admin/shutdown` — Gracefully shut down the node (localhost only)
- `POST   /api/admin/config/reload` — Hot-reload operational config parameters
- `POST   /api/admin/downloads/:model_id/cancel` — Cancel in-progress HF download
- `DELETE /api/admin/models/:model_id` — Remove model (shards + manifest + state)
- `POST   /api/admin/models/:id/unload` — Unload model from VRAM (keep shards on disk)
- `DELETE /api/admin/models/:id/shards/:index` — Delete a single shard
- `GET/PUT /api/admin/models/:id/auto-manage` — Per-model auto-manage policy (incl. prune toggle)
- `PUT    /api/admin/models/:id/shards/:index/lock` — Lock/unlock a shard (prevent auto-pruning)
- `GET/PUT /api/admin/schedule` — Resource schedule management
- `GET    /api/admin/prune-history` — Recent auto-prune events
- `GET/POST /api/admin/adapters` — List/register LoRA adapters
- `DELETE /api/admin/adapters/:id` — Delete a LoRA adapter
- `GET/PUT /api/admin/providers` — View/configure cloud provider API keys
- `GET    /api/admin/provider-models` — List models available from cloud providers
- `GET    /api/admin/provider-health` — Probe cloud provider availability
- `POST   /api/admin/provider-model-status` — Check specific model availability on provider
- `GET    /api/admin/version` — Version info (binary version, git hash, build features)
- `POST   /api/admin/update/check` — Check for new SwarmLLM releases
- `POST   /api/admin/update/apply` — Download and apply update
- `GET    /api/admin/network-map` — Network topology heatmap data
- `GET    /api/admin/models/:id/metadata` — GGUF metadata (context length, quantization, layers)
- `GET    /metrics` — Prometheus/OpenMetrics endpoint (no auth)
- `GET    /health/ready` — Readiness probe with subsystem status (no auth)

### Static
- `/admin` — Dashboard SPA (single-page app — all routes serve index.html)
- `/chat` — Chat interface
- `/setup` — First-run wizard
- `/static/*path` — Embedded static assets (CSS, JS, i18n JSON)
- `/static/i18n/{lang}.json` — Translation files (20 languages)
- `/health` — Health check endpoint
- `/` → redirect to `/admin`

### Frontend Features
- **i18n**: 20 languages (en, es, fr, de, pt, it, nl, ru, zh, ja, ko, ar, tr, pl, sv, th, hi, vi, id, uk, cs). Auto-detects browser language. `I18n.t()` + `data-i18n` DOM attributes. "Continue in English" UX for non-English users who prefer English.
- **Theme**: Light / Dark / System toggle. `[data-theme="light"]` CSS overrides. Persisted in localStorage.
- **Basic/Advanced mode**: Hides technical details (shard grids, GGUF metadata, etc.) in basic mode. Persisted in localStorage.

## Node Tiers

| Tier | Requirements | Role |
|---|---|---|
| Super Node | Full model in VRAM/RAM, high bandwidth | Full inference, backbone |
| Standard Node | Partial VRAM/RAM, moderate bandwidth | Shard hosting, pipeline participation |
| Light Node | Minimal resources | Consumer, bandwidth contribution |

## Platform Targets

| Platform | Priority | GPU Support |
|---|---|---|
| Linux x86_64 | P0 | CUDA + ROCm |
| macOS aarch64 | P1 | Metal (via llama.cpp) |
| Windows x86_64 | P1 | CUDA |
| macOS x86_64 | P2 | CPU only |
| Linux aarch64 | P3 | CPU only |

## Networking Notes

### Virtualized Environments (WSL2, VMs)

Environments with multiple virtual network interfaces may experience connection races. Recommended settings for `config.toml`:

```toml
[network]
enable_autonat = false    # Prevents protocol noise on virtual adapters
enable_dcutr = false      # Hole punching unreliable through VM NAT
enable_mdns = false       # mDNS discovers multiple interfaces, causing connection races
enable_quic = false       # QUIC can win connection races but fail on large payloads
listen_address = "127.0.0.1"  # Bind to loopback only; avoids virtual NAT adapters
```

**Why:** With `max_established_per_peer=1`, simultaneous connection attempts via different interfaces cause mutual rejection. Disabling competing transports and binding to a single interface avoids this.

### Multi-Node Local Testing

Use `SWARMLLM_NODE_DATA_DIR` for per-node isolation:

```bash
# Node 1
SWARMLLM_NODE_DATA_DIR=/tmp/node1 ./swarmllm run -p 8800

# Node 2 (bootstrap via TCP, port+10)
SWARMLLM_NODE_DATA_DIR=/tmp/node2 ./swarmllm run -p 8801 \
  --bootstrap /ip4/127.0.0.1/tcp/8810
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup details.

## Benchmark Data

Single-node inference performance, measured with `swarmllm bench` (100 output tokens, 3-run average, Q4_K_M quantization).

**Hardware:** 8-core CPU, NVIDIA RTX 3070 (8GB VRAM)

| Model | Parameters | GPU (RTX 3070) | CPU (Ryzen 7 5800H) | GPU Speedup |
|-------|-----------|----------------|---------------------|-------------|
| TinyLlama 1.1B | 1.1B | 27.2 tok/s | 4.2 tok/s | 6.5x |
| Gemma-2 2B IT | 2.5B | 20.6 tok/s | 3.5 tok/s | 5.9x |
| Phi-3.5 Mini | 3.8B | 46.4 tok/s | 1.8 tok/s | 25.8x |
| Qwen2.5-Coder 7B | 7.6B | 29.0 tok/s | 2.4 tok/s | 12.1x |

**Notes:**
- GPU inference uses candle with CUDA (`--features candle-cuda`). CPU uses candle with native BLAS.
- Phi-3.5 benefits most from GPU due to its fused QKV/FFN architecture.
- With 8GB VRAM, only one 7B model can be loaded at a time. Multiple smaller models (1-3B) can coexist.
- On-demand model loading with LRU eviction loads models into VRAM only when requested.
