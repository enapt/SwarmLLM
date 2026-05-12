# SwarmLLM Architecture Reference

## Workspace Structure

Cargo workspace with three crates:

| Crate | Path | Purpose |
|-------|------|---------|
| `swarmllm` | `/` (root) | Main binary — daemon, networking, inference, API, all subsystems |
| `swarmllm-types` | `crates/swarmllm-types/` | Shared data types (78 types: NodeId, ModelManifest, SwarmMessage, etc.) |
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
│  │                                                     │  │
│  │  ┌─ EventBus (state.events) ──────────────────────┐ │  │
│  │  │  broadcast::Sender<ActivityEvent> (cap 256)    │ │  │
│  │  │  broadcast::Sender<DashboardSignal> (cap 32)   │ │  │
│  │  │  activity_history, update_state                │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ CreditPool (state.credits) ──────────────────┐ │  │
│  │  │  credit_balance, pool_state, pool_registry     │ │  │
│  │  │  trust_manager, escrow_manager, anti_gaming    │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ ModelMgmt (state.models) ────────────────────┐ │  │
│  │  │  acquisition_progress, hf_sources              │ │  │
│  │  │  auto_manage_*, model_trust, locked_shards     │ │  │
│  │  │  prune_history, download_cancel_flags          │ │  │
│  │  │  wishlist (R111), hf_trending_cache (R112)     │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ MetricsProviders (state.metrics) ────────────┐ │  │
│  │  │  node_stats, inference_requests_total          │ │  │
│  │  │  channel_metrics, inference_latency_samples    │ │  │
│  │  │  providers_config, provider_model_map          │ │  │
│  │  │  swarm_capacity (R110)                         │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                     │  │
│  │  Root: peer_registry, model_registry, executor,     │  │
│  │    identity, db, active_pipelines, config, ...      │  │
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
  ┌───────┬───────┬───────┬───────┬──┴────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼       ▼       ▼       ▼          ▼          ▼          ▼          ▼          ▼          ▼
┌──────┐┌─────┐┌─────┐┌──────┐┌──────┐┌──────┐┌────────┐┌────────┐┌──────┐┌────────┐┌────────┐┌────────┐
│Netwrk││Infer││Crdit││Health││ API  ││Rebal-││Acquisi-││Message ││ Pool ││AutoShrd││HfWatchr││Update │
│Mangr ││Routr││Ledgr││Mon.  ││Servr ││ancer ││tion Mgr││Dispatc ││Mangr ││Manager ││(R112)  ││Checker│
└──┬───┘└──┬──┘└──┬──┘└──┬───┘└──┬───┘└──┬───┘└───┬────┘└───┬────┘└──┬───┘└───┬────┘└───┬────┘└───┬────┘
   │       │      │      │       │       │         │         │        │        │         │         │
   └───────┴──────┴──────┴───────┴───────┴─────────┴─────────┴────────┴────────┴─────────┴─────────┘
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

The **MessageDispatcher** is a dedicated task in `daemon/dispatch/mod.rs` that routes inbound network messages to the appropriate subsystem. Inference messages go to InferenceRouter, CreditGossip updates peer balance distributions, and pool messages go to PoolManager.

## Startup Sequence

```
1.  Parse CLI args (clap) — including optional --model and --gpu-layers
2.  Initialize tracing subscriber (verbosity: info → debug → debug+libp2p → trace)
3.  Load or create config (TOML + env + defaults + CLI overrides)
4.  Ensure data directory exists
5.  Load or generate Ed25519 identity
6.  Open redb database
7.  Build Daemon { config, identity, db }
8.  Initialize ModelExecutor (load GGUF model if --model provided)
9.  Build Arc<SharedState> (includes ModelRegistry loaded from DB)
10. Scan local shards → register in model_registry (with disk existence verification).
    Claims manifest publisher as our node_id + recomputes BLAKE3 hash (allows gossiping copied shards).
11. Create mpsc channels (network, router, rebalance, acquisition, pool)
12. Spawn all tasks (12 tasks: NetworkManager, InferenceRouter, MessageDispatcher,
    HealthMonitor, ShardRebalancer, CreditLedger, AcquisitionManager, ApiServer,
    PoolManager, AutoShardManager, HfWatcher (R112), UpdateChecker)
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
10. mDNS race recovery: if simultaneous-dial kills both connections (max_per_peer=1),
    pending_redial queue schedules re-dial with hash-based jitter (2-5s)
```

### Peer Registry Scaling (S3)

peer_registry capped at 200 entries. On overflow, evicts highest-latency non-LAN non-pipeline peer.
Memory bounded at O(1) instead of O(N). LAN peers and pipeline-active peers are never evicted.

### DHT-Based Shard Holder Resolution (S5)

Two-tier shard holder discovery for 50K+ node scaling:

- **Tier 1 — Bounded in-memory cache**: `ModelRegistry.shard_holders` uses `HashMap<NodeId, Instant>` (not `HashSet<NodeId>`) with max 50 holders per shard. LRU eviction when at capacity. Local node never evicted. Populated by GossipSub `ShardAnnounce` + DHT query results. Sync `shard_holders()` API unchanged — scheduler hot path stays fast.

- **Tier 2 — Kademlia provider records**: Each node calls `kademlia.start_providing()` for its shards (key: `/swarm/provide/<model_id>/<shard_index>`). Provider records TTL 1 hour, republished every 20 minutes. `get_providers()` results are resolved from PeerId → NodeId (same Ed25519 key, bidirectional conversion in `transport.rs`) and merged into the bounded cache.

- **Pre-warm**: Router fires `dht_query_tx.try_send(model_id)` before pipeline assembly. NetworkManager issues `get_providers()` for all model shards, merging results into registry asynchronously. First request may miss cache; subsequent requests benefit.

- **Lifecycle wiring**: `NetworkCommand::StartProviding` on shard acquisition (startup scan, rescan, download complete). `NetworkCommand::StopProviding` on shard deletion (prune, admin API).

- **Disconnect eviction**: `handle_connection_closed` calls `model_registry.remove_peer_from_all_shards(node_id)` synchronously alongside `peer_registry.remove`. Prevents the scheduler from picking a just-disconnected peer (was a 90s window before the health-monitor stale-peer sweep ran). DHT can still re-inject the peer asynchronously, so the scheduler's `connected_node_ids` filter is the load-bearing guard.

Memory: O(shards × 50) bounded regardless of network size (was O(shards × nodes) unbounded).

## Networking Stack

```
libp2p Swarm
├── Kademlia (DHT)
│   ├── /swarm/node/{node_id}                  → NodeCapability
│   ├── /swarm/shards/{model_id}/{node_id_hex} → Vec<ShardIndex> (per-node; avoids last-writer-wins)
│   ├── /swarm/shard/{model}/{index}           → Vec<NodeId> (batched per model)
│   └── /swarm/model/{model_id}               → ModelManifest
│   └── Records expire after 1 hour, re-published periodically
│
├── GossipSub (pub/sub, mesh_n/mesh_n_low/mesh_n_high/mesh_outbound_min auto-scale with known_peers: 2/1/4/1 at <10 peers up to 8/6/16/4 at 10k+)
│   ├── swarm/models/{model_id}       → ShardAnnounce, capacity
│   ├── swarm/credits                 → CreditGossip
│   ├── swarm/health                  → trust summaries
│   ├── swarm/identity                → NicknameRecord (signed, timestamp-checked)
│   └── swarm/pools                   → PoolState, PoolInvitation
│   └── Messages >5 min old are rejected (replay protection)
│   └── Failed publishes buffered and replayed on mesh formation
│
├── request_response (unified protocol, /swarmllm/1.0.0, 600s timeout — slow CPU inference)
│   ├── JSON control messages — SwarmMessage, ShardRequest/ShardResponse
│   ├── Binary tensor payloads — LayerForward, LayerResult (type-tag byte: 0x00=JSON, 0x01=tensor, zstd compression optional)
│   ├── Binary shard data — ShardResponse payload (type-tag byte: 0x03=shard, 32MB chunks as raw bytes, bypasses 4MB JSON limit)
│   └── ACK-timeout fast-fail: streaming-tracked sends (`SendDirectMessage` with `delivery_request_id = Some(uuid)`) are mapped to a Uuid via `pending_rr_observability`. The 10s `RR_ACK_TIMEOUT_SECS` sweep closes `streaming_token_txs[uuid]` if no Response/OutboundFailure event fires (libp2p rr can silently drop sends under load); caller sees Err in ~10–20s instead of 120s
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

The split inference engine (`src/inference/split/`) enables true distributed inference
using candle for direct tensor computation with quantized GGUF weights. Each node loads
only the transformer layers it owns, forwarding hidden-state activations between nodes.

The module is split into focused subfiles: `model.rs` (SplitModel struct + accessors),
`loader.rs` (GGUF/shard load), `executor.rs` (forward pass + tensor-parallel),
`kv_cache.rs` (per-request KV-cache store), `entry.rs` (model entry + LRU eviction),
`gguf_meta.rs` (GGUF header parsing), `shard_reader.rs` (multi-shard virtual reader),
`rope.rs` (RoPE precomputation), `prefix_cache.rs` (cross-request prefix-KV reuse).

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
- **Per-step alpha/beta gating**: `ssm_alpha.weight` and `ssm_beta.weight` tensors are read from the GGUF and applied per timestep via the Gated DeltaNet formula: decay `g_t = exp(-softplus(α + dt))`, prediction error `error = β_v·v - g·S@(β_k·k)`, state update `S_t = g·S + error ⊗ (β_k·k)^T`.

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
- **Files**: `src/inference/allreduce.rs` (coordinator + registry), `src/inference/scheduler/mod.rs` (TP group detection)

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
3. Filter holders against `connected_node_ids` — drops peers whose libp2p
   connection is gone (DHT can re-inject stale providers; `peer_registry`
   is preserved across mid-pipeline disconnects for reconnect attempts so
   it's not the right liveness oracle)
4. Fetch node load/latency from peer_registry
5. Sort candidates by (latency ASC, load ASC, trust DESC)
6. Greedy assignment: widest contiguous layer range per node
7. Merge contiguous segments assigned to the same node
8. Identify standby nodes per segment
9. Send PipelineAssignment → all nodes ACK → begin forwarding

### Inference Correctness

**Stop sequence handling**: User-provided stop sequences (`stop` in OpenAI, `stop_sequences` in Anthropic) are enforced in all three inference execution paths:
1. `pipeline/distributed.rs` `execute_distributed` — accumulated text scanned after each token decode
2. `model_worker.rs` `handle_generate` — accumulated text checked after each token in the subprocess decode loop
3. `executor.rs` `generate_stream_llama` — accumulated text checked after each token in the llama.cpp loop

Empty stop sequences are rejected at the API validation layer (must be 1–256 chars).

**EOS token handling**: The distributed pipeline checks for EOS tokens in `result.token_ids` explicitly (not just via `result.finish_reason`), preventing runaway generation if the worker subprocess returns EOS as a token ID without setting the finish reason.

**Top-k sampling**: Uses `select_nth_unstable_by(k - 1, desc_cmp)` to partition the k largest logits into `[..k]`. The k-1 pivot ensures exactly k elements are retained (not k-1).

**RoPE position tracking**: Prompt token count estimated as `max(chars / 4, 1)` when no tokenizer is available on the coordinating node. The minimum of 1 prevents zero-position RoPE for short prompts.

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

### Chunked Prefill

Long prompts are split into chunks for overlapped prefill and decode:
- Prevents head-of-line blocking from long-context requests
- Decode steps for other requests can interleave between prefill chunks
- Chunk size auto-tuned based on available VRAM

### Prefix-Cache KV Sharing (Cross-Node)

Each worker stores a local prefix-cache keyed by BLAKE3 chained hashes over
fixed-size token blocks (`prefix_cache_block_tokens`, default 64). Blocks are
announced to peers via `SwarmMessage::PrefixCacheAnnounce` on the
`swarm/models` gossipsub topic, indexed by each recipient in
`state.models.cross_node_prefix_index`. When a local worker sees a prompt
whose prefix it hasn't prefilled, it emits `WorkerMsg::PrefixFetchProbe`; the
daemon walks the index (longest-match first), trust-gates candidate peers by
`cross_node_prefix_trust_min` (default 0.5), and issues a `SendPrefixKvFetch`
request-response to the best holder. The serving daemon re-issues
`DaemonMsg::ExportPrefixSnapshot` to its worker, which narrows a stored
`KvSnapshot` to the requested block boundary and returns the serialized bytes
in the IPC binary-payload slot. Back on the requesting side, the bytes are
BLAKE3-reverified against the requested hash and NaN/Inf-scanned before
hydrating a new `KvCacheEntry` for the in-flight request, which then only has
to prefill the suffix beyond the cached block boundary.

The fetch path uses three chained timeouts so a stuck peer or worker always
degrades to a clean miss rather than blocking the request: worker-side probe
(`PREFIX_FETCH_TIMEOUT_MS`, 3000 ms outer bound), daemon-side network
dispatch (2500 ms), and serving-side worker IPC
(`fetch_local_snapshot`, 2000 ms). Sized for 7B-class snapshots
(~70–150 MB f32) — a clean miss is no worse than not having the feature.

See `docs/plans/benchmarks/round6.md` for the two-daemon loopback bench
recipe and measured TTFT numbers: TinyLlama on GPU is too small to win on
localhost (28 MB fetch ≈ 460 ms prefill), but Qwen-7B on CPU crosses over
decisively at **12.9× iter-1 TTFT speedup** (151.7 s full local prefill
→ 11.8 s with fetch).

### Speculative Decoding

- Draft model (small/fast) proposes K candidate tokens per step (default 4)
- Target model verifies all K in one forward pass (amortized GPU cost)
- Rejection sampling ensures output distribution identical to non-speculative
- KV-cache resynchronization on rejection (trim + reseed)
- Config: `speculative_decoding`, `speculative_gamma`, `draft_model_path`
- Falls back to standard decoding if no draft model available

### Subprocess-Per-Model Isolation (Ollama-style)

Each loaded model runs in its own `swarmllm model-worker` subprocess. This guarantees that unloading a model **immediately** reclaims all GPU memory — the OS and CUDA driver free all allocations when the process exits, bypassing the CUDA allocator cache that prevents memory release within a single process.

```
Main daemon (control + P2P + API)          Worker subprocess per model
────────────────────────────────           ──────────────────────────
InferenceRouter                            model-worker --socket /tmp/...
  ↓                                          connects to daemon socket
ModelProcessPool.generate()  ──socket──▶   loads shards from disk
ModelProcessPool.forward()   ──socket──▶   runs forward passes / decode loop
                             ◀──socket──   streams tokens / LayerResult back
                                           exits on unload → VRAM freed
```

**Communication**: Unix domain socket with binary framing (`[4B json_len][json][4B payload_len][raw bytes]`). JSON carries message metadata; the payload carries raw activation tensor bytes to avoid base64 overhead.

**Message types** (`src/inference/worker_ipc.rs`):
- `DaemonMsg::Forward(IpcForward)` — single-step LayerForward for distributed inference
- `DaemonMsg::Generate(IpcGenerate)` — full prompt→tokens decode loop for API inference
- `DaemonMsg::Unload` — drop a layer range within the worker (partial memory reclaim)
- `DaemonMsg::Shutdown` — graceful exit
- `WorkerMsg::Token` — streaming token during Generate
- `WorkerMsg::LayerResult` — activation result for distributed pipeline forwarding

**`ModelProcessPool`** (`src/inference/process_pool.rs`):
- `DashMap<ModelId, Arc<WorkerHandle>>` — one worker per active model
- `get_or_spawn()` — lazily spawns a worker on the first request for a model
- `forward()` — routes a `LayerForward` to the subprocess, awaits `LayerResult`
- `generate()` — sends a full generate request, streams `WorkerMsg::Token` back
- `unload_model()` — kills the subprocess → OS/CUDA reclaims all GPU memory
- **Crash recovery**: if a worker subprocess crashes (OOM, CUDA fault, panic), the IO error from `send_daemon`/`recv_worker` evicts the dead `WorkerHandle` from the pool. The next inference request for that model automatically respawns a fresh worker via `get_or_spawn()`.
- **Socket cleanup**: a RAII guard (`SocketCleanup`) ensures the Unix socket file in `/tmp/` is removed if `spawn_worker` fails at any step after binding. On success, the guard is defused via `mem::forget` and `WorkerHandle::drop` handles cleanup.

**`SplitModelEntry`** is now **metadata-only** (no `Arc<Mutex<SplitModel>>` in the main process):
- Caches `eos_tokens`, `vocab`, `chat_template`, `bos_token`, `eos_token_str` from the GGUF header
- `estimated_vram_mb` from shard file sizes on disk
- The actual model weights live exclusively in the worker subprocess

**Granularity**: one process per `ModelId` (not per shard). A single worker handles all layer ranges for one model, owns its own `KvCacheStore`, and processes requests sequentially — matching the prior `Mutex<SplitModel>` serialization. Individual shard load/unload is handled within the worker via `DaemonMsg::Unload`; the process only exits when all shards are released or `Shutdown` is received.

**Dashboard responsiveness**: since inference never runs on the main Tokio runtime, API and WebSocket handlers always get a fast response even under heavy inference load.

### VRAM-Aware Cache Eviction

- `SplitModelEntry` tracks `last_used` timestamp and `estimated_vram_mb` (from shard file sizes)
- Configurable `max_split_model_memory_mb` budget (default unlimited)
- LRU eviction: `evict_split_models_lru` removes metadata entries and kills the corresponding worker subprocess — VRAM is guaranteed freed
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
  +10 credits  per token served (balanced with consume side)
  +1  credit   per GB per hour hosting shards
  +5  credits  per GB seeding shard data
  +2  credits  per connection hour relay service

Spending (default rates, configurable per pool):
  -10 credits  per token consumed (balanced with earn side)
  -50 credits  per distributed inference failure (automatic penalty)

Minimum balance enforcement:
  Nodes below -1000 credits have remote requests rejected.
  Local API requests (localhost) are always allowed.
  Earn credits by: hosting shards, serving inference, seeding data.

Tiers (enforced per-request in InferenceRouter):
  Platinum  (≥90th percentile, balance>0)  → 2× concurrent slots
  Gold      (≥70th percentile, balance>0)  → base concurrent slots
  Silver    (positive balance)             → ½ concurrent slots
  Bronze    (zero/negative)                → ¼ concurrent slots (min 1)
```

**Balanced rates**: Both earn and spend use `rate × tokens` (no layer multiplier). This prevents
credit inflation — a 22-layer model serving 100 tokens earns exactly as much as it costs to
consume. Previously the earn side multiplied by layers, causing 22× inflation per request.

**Tier enforcement flow**: On each `handle_submit()`, the router computes the network percentile
from `peer_credit_balances` (populated via credit gossip, **deduplicated by NodeId** to prevent
Sybil percentile stuffing), calls `calculate_tier()`, and sets the request priority. Balance must
be positive for Gold/Platinum tiers. In `drain_queue()`, `max_concurrent_for_tier()` limits
concurrent execution slots per tier.

**Queue draining**: `drain_queue` only fires on Submit/StreamSubmit commands or `queue_notify`.
Every path that calls `active_count.fetch_sub(1)` on completion MUST also call
`queue_notify.notify_one()` — otherwise queued requests beyond the per-tier cap sit indefinitely
until a new Submit arrives. Four call sites enforce this: `ActivePipelineGuard::drop` (panic
path), normal-completion in `dispatch_single`, `execute_distributed_batch` (spawn body + join-loop
panic arm), and `BatchCleanup` (`complete_one` + `Drop`) in `local_exec`.

**Transient-failure retry**: `dispatch_single` wraps `execute_request` with a single retry on
`is_transient_remote_failure` errors (silent rr drop, OutboundFailure, remote-generate timeout).
Retry passes `preferred_pipeline = None` so the scheduler re-runs and the dead/dropped peer is
filtered out via `connected_node_ids`. Bounded to one retry per request — failure of the second
attempt propagates to the user with a "try again" hint.

**Minimum balance enforcement**: Remote peers with balance below `MIN_BALANCE_FOR_INFERENCE`
(-1000) have their inference requests rejected with a descriptive error message telling them to
contribute. Local API requests (requester == NodeId([0;32])) bypass this check.

**Atomic credit accumulation**: Forward participation credits (earned during distributed inference
hot path) are accumulated in an `AtomicI64` (`pending_credit_earn`) to avoid lock contention.
The CreditLedger periodic persist (every 60s) flushes the accumulator to the balance + DB.
No credits are lost under high concurrency.

**Anti-Sybil balance gossip**: Peer balance reports are deduplicated by NodeId via a DashMap.
Each peer gets exactly one entry in the percentile calculation, preventing a single peer from
dominating the distribution by re-gossiping frequently.

**Relay credits**: NetworkManager tracks active relay circuits via `active_relay_circuits` DashMap.
On `CircuitReqAccepted`, records start time. On `CircuitClosed`, computes duration and adds to
`relay_seconds_served` atomic counter. CreditLedger drains this counter periodically.

**Failure penalties**: When distributed inference fails, the router applies `penalty_serve_failure`
credits (default -50) and broadcasts `InferenceError` to all pipeline participants.

Credit earn/spend rates are configurable per pool via the pool configuration API.

**Pool credit forwarding**: When a pool member earns credits, `earn_inference` attempts to forward them to the pool owner before crediting locally. If forwarding succeeds, the member retains nothing (return 0). If forwarding fails or the node is not in a pool, credits are applied locally. This prevents credit inflation when pool members accumulate credits that should belong to the owner.

**Pipeline completion credit earn**: Uses the configurable rate from `config.pool.credit_rates.inference_serve` (not the compile-time constant), with `saturating_mul` for overflow safety. Remote peers earn per-forward-step via `track_forward_participation`; local segments earn via `apply_credit_direct` at pipeline completion — these are separate code paths with no double-counting.

## Device Pools (Multi-Device Credit Linking)

Users with multiple machines can link them into a **device pool**. All credits earned by member
devices are forwarded to the owner (main) device, giving a combined credit balance.

```
Main Device (owner)                 Linked Device (member)
┌──────────────────┐               ┌──────────────────┐
│ Combined balance │◀──────────────│ Earns credits    │
│ Pool management  │  CreditForward│ Forwards to owner│
│ Invite codes     │  (dual-signed)│ Keeps split %    │
└──────────────────┘               └──────────────────┘
```

**Setup flow**:
1. Main device: `swarmllm pool create --name "My Devices"`
2. Main device: `swarmllm pool invite-code` → generates 8-char code (e.g., `A3F7K2M9`)
3. Linked device: `swarmllm pool join A3F7K2M9`
4. Code validated over gossip, invitation auto-created, member auto-accepted

**Invite code security**:
- 8-char uppercase alphanumeric (32^8 ≈ 1.1 trillion combos, no 0/O/1/I)
- One-time use, consumed immediately on claim
- 24h expiry (configurable `invitation_ttl_hours`)
- Max 5 active codes at once
- Code hash (BLAKE3) on the wire — plaintext code never transmitted
- Join requests signed with Ed25519

**Pool features**:
- Device nicknames (owner sets per device for easy identification)
- Online/offline status with last-seen timestamps
- Per-device stats (VRAM, shards hosted, forwards served, uptime)
- Combined VRAM display across all pool devices
- Credit split configuration (0-50% kept by member, rest to owner)
- Max 10 devices per pool (configurable), 10 operations/hour rate limit

**Credit forwarding**: When a pool member earns credits, `forward_credits_to_owner()` in
`pool/forward.rs` deducts the forward amount (respecting `member_credit_split_pct`) and sends a
`PoolCreditForward` message. The owner's `PoolManager` co-signs it and applies `apply_credit_direct()`
to the owner's balance. Both signatures (member + owner) are required — preventing forgery.

**Terminology**: "My Devices" / "Linked Devices" in the UI. Clearly distinguished from "Swarm Peers"
(other users' nodes on the P2P network). The dashboard, setup wizard, and share popover all use
distinct language to prevent confusion.

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
- **P2P shard wire format**: Shard chunks use `WIRE_TAG_SHARD` (0x03) binary framing — raw bytes sent directly without JSON serialization. This is essential: the 4MB JSON body limit would silently fail all P2P shard transfers (shards are typically 256MB–1GB)
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

Storage backend is **redb** (pure-Rust, ACID, single-file).

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

The **AutoShardManager** (`src/model/auto_manage/`) is a background subsystem that
automatically acquires missing shards based on a VRAM-aware scoring algorithm.
The module is split into: `manager.rs` (struct + run loop + housekeeping),
`scoring.rs` (candidate ranking), `download.rs` (download orchestration),
`prune.rs` (shard pruning logic), `scan.rs` (local shard scanning + model loading),
`vram.rs` (VRAM budget utilities).

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
- **Download priority**: HuggingFace CDN first (fast, doesn't burden peers). If no HF source available but peers hold the shard, falls back to P2P `ShardRequest` to a random holder. P2P is single-source per shard (future: multi-source parallel download)
- **Upload bandwidth cap**: `max_bandwidth_mbps` config enforced on shard serving via proportional delay after chunk reads. Tensor forwards exempt (latency-critical). Default 0 = unlimited
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
- Shard is actively being downloaded by this node (prevents download/prune race)
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

**Notifications** — prune events flow through the unified `activity_event` WebSocket message (kind: `shard_pruned`, with `toast_level: "info"` and structured prune data fields). Prune history accessible via `GET /api/admin/prune-history`.

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

**Trigger**: When `execute_request()` in the inference router encounters a model that has shards on disk but no worker subprocess running for it:
1. `ModelProcessPool.get_or_spawn()` spawns a `swarmllm model-worker` subprocess
2. Worker connects to the daemon's Unix socket and sends `WorkerMsg::Ready`
3. First `Forward` or `Generate` request causes the worker to load shards from disk
4. VRAM budget is tracked via `SplitModelEntry.estimated_vram_mb`; LRU eviction kills the oldest worker subprocess

**Loading coordination**: the process pool `Mutex<WorkerSocket>` serializes requests per model — if two requests arrive simultaneously for an unloaded model, the second waits for the first to complete spawning.

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
│    Final segment seals output tokens for requester's X25519 │
│    ⚠ Final-segment node sees tokens before sealing (must)   │
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

### Tier 1 AAD construction (single source of truth)

The `LayerForward` AAD bytes — `request_id(16) | sequence_num(4 LE) |
index_pos(4 LE) | fmt(1) | layer_start(4 LE) | layer_end(4 LE) |
model_id_len(2 LE) | model_id` — are produced by
`network::protocol::build_layer_forward_aad`. Both encrypt-side
(`network::manager::tensors::handle_send_tensor`) and decrypt-side
(`network::protocol::decode_layer_forward_encrypted`) call it. Any
field added to `LayerForward` that needs authentication MUST extend
this helper, not be re-appended on the encrypt side. Drift between
the two sides silently breaks every encrypted forward (AAD mismatch
fails AEAD verify; only a `seal/open mismatch` warn surfaces).

## Pipeline Privacy Model

What each node sees in a distributed pipeline (Requester → A → B → C):

```
┌──────────────────────────────────────────────────────────────────┐
│ Data exposure by pipeline position                               │
│                                                                  │
│                    Requester  Node A     Node B     Node C       │
│                    (author)   (first)    (middle)   (last)       │
│  ────────────────────────────────────────────────────────────    │
│  Plaintext prompt    ✓        *          ✗          ✗            │
│  Raw token IDs       ✓        *          ✗          ✗            │
│  Activations in      —        ✓          ✓          ✓            │
│  Activations out     —        ✓          ✓          —            │
│  Generated tokens    ✓(dec)   ✗          ✗          ✓(samples)  │
│  Final response      ✓(dec)   ✗          ✗          ✓(seals)    │
│                                                                  │
│  * Without local_embedding_privacy: Node A sees raw tokens       │
│    With local_embedding_privacy: Node A sees FP32 activations    │
│                                                                  │
│  ⚠ The final-segment node ALWAYS sees generated output.          │
│    This is inherent — sampling happens on the last node.         │
│    Pipeline sealing encrypts tokens on the wire, but the         │
│    node that samples them must see them.                          │
│                                                                  │
│  ⚠ Activation inversion: early-layer activations (especially    │
│    layer 0) can theoretically be reversed to recover tokens.     │
│    local_embedding_privacy eliminates the trivial case.          │
│    Deep-layer inversion is an open research problem.             │
└──────────────────────────────────────────────────────────────────┘
```

## Local Embedding Privacy

Optional privacy enhancement (`local_embedding_privacy: true` in `[inference]` config):

```
Without privacy:     Prompt text → [tokenize on first segment] → raw token IDs visible
With privacy:        Prompt text → [tokenize + embed locally] → FP32 activations sent
                     Remote nodes see activation tensors, not token IDs
```

- `LocalEmbedder` loads `token_embd.weight` from `shard_000.bin` at startup (~64MB for 7B Q4)
- Embedding lookup is a simple matmul (~1ms), negligible overhead
- Wire protocol: `LayerForward.pre_embedded: bool` (`#[serde(default)]` for backward compat)
- `SplitModel::forward_pre_embedded()` skips embedding lookup when `pre_embedded = true`
- Supports Gemma embedding scaling (`sqrt(hidden_dim)`)
- Trade-off: larger wire payloads (e.g., 512 tokens × 4096 dim × 4B = 8MB vs ~2KB text)
- Modules: `src/inference/local_embedder.rs`, `src/daemon/state/mod.rs` (`local_embedders` DashMap)

## Private Mode

Pool-only inference restriction that guarantees your prompts never leave your devices. Toggle via dashboard shield icon, pool section toggle, or `PUT /api/pool/private-mode`.

**Three scopes:**
- **Private (pool only)** — inference restricted to device pool members. Works over WAN or LAN.
- **Private + LAN** — pool members plus any mDNS-discovered LAN peer (`private_mode_allow_lan: true`, default).
- **Offline** — air-gapped operation. No bootstrap peers, no HF downloads, mDNS-only discovery.

**What Private Mode restricts (your outbound requests):**
- Inference pipeline assembly: scheduler filters candidates to allowed node set
- Auto-manage shard scoring: only counts holders/replicas within allowed set
- Auto-manage pruning: only considers pool-scoped replication
- Auto-manage downloads: only downloads from pool peers (HF fallback in online mode)
- VRAM pool calculation: only sums allowed peers' GPU VRAM

**What Private Mode does NOT restrict (you still contribute):**
- Serving inference requests from other swarm nodes
- Hosting and seeding shards to the network
- Earning credits for work done
- P2P gossip, DHT, health pings

**Implementation:** Single `allowed_node_set()` helper in `src/pool/scope.rs` returns `Option<HashSet<NodeId>>`. `None` = unrestricted (normal mode), `Some(set)` = only these nodes. All filtering flows from this one function. Runtime-toggleable via `AtomicBool` on `SharedState.credits.private_mode`.

**Shard Pinning:** Pool owners can pin specific models/shards to specific devices via `POST /api/pool/pin`. Pinned shards get 1000x scoring bonus on the target node and are never pruned. Enables manual shard distribution (e.g. GPU machine gets the big model).

**Coverage Dashboard:** `GET /api/pool/coverage` returns per-model coverage within the pool (total_shards, pool_shards, coverage_pct, missing indices, est_download_mb). Frontend shows color-coded bars and disk usage.

**Config:**
```toml
[pool]
private_mode = false           # Restrict inference to pool only
private_mode_allow_lan = true  # Include LAN peers when private
offline_mode = false           # Air-gapped: no internet, mDNS only
```

**API:**
- `GET /api/pool/private-mode` — state + coverage summary
- `PUT /api/pool/private-mode` — toggle `{ "enabled": true, "offline_mode": true }`
- `GET /api/pool/coverage` — per-model pool coverage
- `GET /api/pool/pins` — list shard pins
- `POST /api/pool/pin` — pin model to device
- `DELETE /api/pool/pin` — remove pin

**Error:** `SwarmError::PrivateModeUnavailable { model_id, missing_shards }` → HTTP 503 with specific missing shard list so users know exactly what's needed.

## Transport-Authenticated Dispatch

All inbound network messages are wrapped in `AuthenticatedMessage` with the transport-verified sender `NodeId` (from libp2p Noise protocol). The MessageDispatcher validates sender identity against message claims for all security-sensitive message types (ShardAnnounce, CreditTransaction, CreditGossip, NicknameGossip, HealthPing/Pong, EphemeralKeyExchange). Mismatched messages are logged and dropped.

## Signed DHT Records

Kademlia DHT records for capability and shard announcements are Ed25519-signed:
- Format: `[32B pubkey][64B signature][payload]`
- Functions: `verify_dht_value()` in `src/network/discovery.rs` (signing is inline in NetworkManager)
- Records expire after 1 hour with automatic re-publication

Kademlia provider records (S5) track shard holders at scale:
- Key: `/swarm/provide/<model_id>/<shard_index>` per shard
- Functions: `start_providing_shards()` / `stop_providing_shards()` / `query_shard_providers()`
- Provider TTL: 1 hour, republication: 20 minutes
- PeerId→NodeId via `peer_id_to_node_id()` in `transport.rs` (production); the reverse direction is test-only since libp2p derives PeerIds from keypairs directly

**DHT shard keys are per-node** to prevent last-writer-wins collisions: records are keyed
as `/swarm/shards/{model_id}/{node_id_hex}` (one record per node per model), not a single
shared key that any node can overwrite. Each node publishes only its own shard holdings.

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
- Config: max_pool_size=10, invitation_ttl_hours=24, rate_limit_per_hour=10

**Pool join security hardening**:
- Join request **signature verification is transport-authenticated**: the dispatch layer sets the requester `NodeId` from the verified Noise-authenticated sender, not from a self-reported field in the message body. Forgery of join origin is not possible.
- **Capacity check before invitation consumption**: pool size is validated before the invite code is marked as used, preventing invitee lockout when the pool is already full.
- **`auto_accept` bound to specific `code_hash`**: auto-acceptance only fires for the exact invitation that matches the code the joiner used, preventing cross-pool or stale auto-acceptance.
- **Removal freshness**: signed removal notices are rejected if their timestamp is more than 30 seconds in the future (previously `abs()` allowed ±5 min, enabling timestamp spoofing).
- **Invite code DoS prevention**: base64-encoded invite codes are capped at 512 characters before decode. Oversized payloads are rejected before any allocation.
- **`pending_credit_earn` atomics use `AcqRel` ordering** (was `Relaxed`) — ensures credit accumulator writes are visible across threads without data races.

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
- Signed-message freshness centralised in `credit::ledger::check_signed_freshness`
  (one-sided staleness, NEVER `.abs()`). Shared `pub(crate)` constants
  `CLOCK_SKEW_TOLERANCE_SECS = 30` and `BALANCE_REPORT_MAX_AGE_SECS = 300`
  apply to balance reports AND credit transactions, so a single tuning
  changes the replay window for every signed credit-typed message at once
- Regional gossip (`RegionShardSummary`, `ModelDemandGossip`) freshness
  goes through `daemon::dispatch::gossip_timestamp_fresh` — same one-sided
  invariant on `u64` millisecond timestamps. `saturating_sub` returns 0
  when `ts > now`, so the future-rejection branch is required separately
  from the staleness-rejection branch. Both that helper AND the
  GossipSub wire-level pre-filter in `network/manager/events.rs` route
  through the generic `daemon::dispatch::timestamp_fresh_one_sided`,
  so the one-sided invariant has a single implementation
- Pool removal freshness (`pool::manager::handle_inbound_removal`) routes
  through `credit::ledger::check_signed_freshness` so the same
  replay-window constants apply to every signed timestamp the daemon
  accepts

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
- **CORS**: `OPTIONS` preflight requests explicitly allowed (required for cross-origin browser clients)
- **Connectivity probe** (health check): narrowed to reject responses with content length > 20 chars, preventing false-positive pass-through from other services on the same port
- **Total prompt cap**: raised from 64KB to 4MB for Claude Code compatibility (tool call results and long context prompts can exceed 64KB)
- **Anthropic→OpenAI proxy**: now supports streaming — SSE events from the upstream OpenAI-compatible provider are translated to Anthropic SSE format and forwarded to the client in real time

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

`ShardReader` in `split/shard_reader.rs` constructs a virtual GGUF from header + shard files,
allowing candle to parse the full tensor index while only loading assigned layers.

## HTTP API Routes

### OpenAI-Compatible (Bearer auth required)
- `POST   /v1/chat/completions` — Chat completions (streaming + non-streaming, tool_calls + logprobs support)
- `POST   /v1/responses` — OpenAI Responses API (gpt-5 / o-series default)
- `GET    /v1/responses/{id}` — Retrieve a stored response (30-day TTL); pass `?stream=true&starting_after={seq}` to resume a background SSE stream
- `DELETE /v1/responses/{id}` — Delete a stored response
- `POST   /v1/responses/{id}/cancel` — Cancel a background response
- `GET    /v1/responses/{id}/input_items` — Paginated list of the original input items (synthetic ids `item_N`)
- `POST   /v1/messages` — Anthropic Messages API (full Claude Code compatibility — tools, tool_choice, thinking, cache_control, metadata)
- `POST   /v1/embeddings` — Text embeddings
- `GET    /v1/models` — List available models
- `GET    /v1/providers` — List configured cloud providers and their available models
- `GET    /v1/status` — SwarmLLM node status

### OpenAI Responses API (`/v1/responses`)
OpenAI-compatible Responses endpoint — the 2026 default API for o-series / gpt-5 / reasoning-era callers:
- **Request fields:** `input` (string or array of items), `instructions`, `previous_response_id`, `max_output_tokens`, `tools` (`function`), `tool_choice`, `reasoning`, `text.format`, `text.verbosity`, `service_tier`, `include[]`, `store`, `background`, `parallel_tool_calls`, `stream`, plus arbitrary forward-compat fields via `#[serde(flatten)] extras`.
- **Input items:** `message`, `function_call`, `function_call_output`, `reasoning` (cloud-proxy path), with content parts `input_text`, `input_image`, `input_file`, `input_audio`. Unknown item types round-trip via a `Raw(Value)` fallback.
- **Multimodal input (V2 of v2 plan):** `input_image{image_url}` (base64 data URIs pass through), `input_file{file_data}` (UTF-8 payloads inlined as text with a `[File: name]` header). `input_image{file_id}`, `input_file{file_id}`, `input_audio`, and non-UTF-8 file payloads are rejected with explicit errors pointing at the supported alternatives. 20 MiB cap per file.
- **Routing:** OpenAI-compatible cloud model (gpt-5, o-series, nvidia/*, etc.) → proxy verbatim to upstream `/responses`; Anthropic / claude-subscription provider → translate to Anthropic Messages, forward, translate back (V3 of v2 plan; `src/api/openai/responses/anthropic_bridge.rs`); otherwise local inference via Chat Completions translation.
- **Built-in tools** (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`, `custom`): rejected on local path (400); forwarded verbatim on cloud path (OpenAI hosts them).
- **Streaming:** SSE with monotonic `sequence_number`. V1 of v2 plan emits `response.created` + `response.in_progress` *before* the chat handler is awaited so the lifecycle events can never be blocked by preflight inside `chat_completions` (cold worker probe, queue wait, template build). Measured first-`data:` line arrival is ~2 ms on TinyLlama CPU at warmed-up steady state (see `docs/bench_results/README.md` for the full pre/post comparison). Events: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.function_call_arguments.delta/done`, terminal `response.completed` | `response.incomplete` | `response.failed` | `response.cancelled`.
- **Persistence:** redb tree `responses`, 30-day TTL, hourly background sweep. `store=false` opts out.
- **Chaining:** `previous_response_id` loads the stored record and flattens prior request.input + response.output into chat messages. Reasoning items round-trip in the stored record (so `encrypted_content` survives byte-for-byte for o-series chains) but are *not* re-injected as chat messages — local inference can't consume them and an empty assistant stub would confuse the prompt.
- **Background:** `background=true` spawns a tokio task, returns `status="queued"` immediately; `GET /v1/responses/{id}` polls; `POST /v1/responses/{id}/cancel` flips the cancel flag (cancel-wins: worker's final result is discarded if cancelled).
- **Background streaming (V8 of v2 plan):** `background=true && stream=true` returns **202 Accepted** + a `Location` header pointing at `/v1/responses/{id}?stream=true&starting_after=-1`. The server runs the inference internally via a spawned task that writes every SSE event into a per-response buffer (cap 2000 events, oldest-first eviction). State lives in `BACKGROUND_STATE: DashMap<id, Arc<BackgroundState>>` (cancel flag + buffer + completion flag + `tokio::sync::Notify`).
- **Resumable SSE (V5 of v2 plan):** `GET /v1/responses/{id}?stream=true&starting_after={seq}` replays buffered events whose `sequence_number > seq`, then live-tails new events until the response is marked completed. If the response already finished and there's no live `BackgroundState`, a synthetic minimal lifecycle (`response.created` + `response.in_progress` + terminal) is built from the stored record so reconnecting clients still close cleanly.
- **input_items pagination (V4 of v2 plan):** `GET /v1/responses/{id}/input_items?after={cursor}&limit={n}&order={asc|desc}`. Synthetic ids `item_N` map to the zero-based position in the original request. Returns the OpenAI list shape `{object: "list", data: [...], first_id, last_id, has_more}`. Default limit `INPUT_ITEMS_DEFAULT_PAGE_SIZE = 20`, max `INPUT_ITEMS_MAX_PAGE_SIZE = 100`, `MAX_INPUT_ITEMS_QUERY_LEN = 64` on each query string (`after`/`before`/`order`/`include`). `Text` input produces a single synthetic message item.
- **Ingress validation:** `validate_responses_ingress` caps `MAX_RESPONSES_INPUT_ITEMS = 1024` items, `MAX_RESPONSES_EXTRAS_COUNT = 32` per `extras` map (top-level AND per-`InputMessageItem`), `MAX_RESPONSES_EXTRA_VALUE_BYTES = 4 KiB` per extras value. Closes a DoS surface where thousands of message items each carrying their own `#[serde(flatten)]` extras could bypass the top-level cap.

### Anthropic Messages API (`/v1/messages`)
Full Anthropic Messages API compatibility for use as a Claude Code backend:
- **Request fields:** `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking), `cache_control` on system blocks
- **Content blocks:** `text`, `image`, `tool_use`, `tool_result`, `thinking`, `redacted_thinking`
- **Routing:** Claude models → Anthropic cloud (full pass-through); non-Claude models → Anthropic→OpenAI translation proxy (supports streaming — SSE events translated in real time); local GGUF models → tool calls/thinking converted to text for inference
- **Total prompt cap:** 4MB (raised from 64KB for Claude Code compatibility — tool results and long-context prompts can exceed the old limit)
- **Claude Code usage:** `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`

### MCP Server (Protocol v2025-11-05)
- `POST /mcp` — JSON-RPC 2.0 MCP endpoint for AI agent frameworks (Claude Code, VS Code Copilot, Cursor, etc.)
- Tools: `chat`, `models`, `compare`, `research`, `batch_prompts`, `delegate`, `node_info`
- Resources: `swarmllm://status` (node status)
- All tools include [tool annotations](https://modelcontextprotocol.io/specification/2025-11-25) (`readOnlyHint`, `destructiveHint`, etc.)
- **`compare`:** sends the same prompt to up to 10 models concurrently, returns side-by-side results
- **`research`:** fan-out a question to multiple models (auto-selects if models omitted), returns all responses with token usage
- **`batch_prompts`:** execute up to 20 independent {id, model, prompt} tasks in parallel
- **`delegate`:** offload a task to the best model for a given tier (fast/cheap/smart) — auto-selects model
- **`node_info`:** detailed node status (loaded model, peers, credits, registry models, cloud providers)

### Cloud Fallback (Optional)
When a requested model isn't available locally or on the swarm, requests can optionally be routed to 12 cloud providers:
- **Providers:** OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks AI, Together AI, DeepInfra, Moonshot/Kimi
- Model prefix routing: `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek, `mistral-*` → Mistral, `moonshot-*`/`kimi*` → Moonshot
- Explicit syntax: `provider:model` (e.g., `openai:gpt-4o`, `groq:llama-3.1-70b`)
- Custom providers via `[providers.custom]` config section
- Provider health probes with per-model availability checking
- Admin API: `GET/PUT /api/admin/providers` — view/configure provider API keys

### Claude Subscription Provider (Optional, feature-gated)
Routes Claude model requests through a locally-authenticated `claude` CLI subprocess, using the user's existing Pro/Max/Team/Enterprise subscription — no API key or per-token charges needed.

- **Feature flag:** `--features claude-subscription` (compile-time opt-in, isolated for easy removal)
- **How it works:** Spawns `claude -p --output-format stream-json` per request, parses NDJSON stdout, translates to OpenAI/Anthropic SSE or JSON responses
- **Multi-turn:** Full conversation serialized per request using XML tags (`<system>`, `<previous_response>`) — stateless, no server-side session state required
- **Routing priority:** Claude subscription (if enabled) > Anthropic API key > error. Configured via `providers.claude_subscription.enabled`
- **Concurrency:** Semaphore-limited (default 3 concurrent subprocesses) to respect subscription rate limits
- **Working directory:** Defaults to `/tmp` for clean context (no project hooks/CLAUDE.md). Configurable via `working_dir` for project-aware completions
- **Admin API:** `GET /api/admin/claude-subscription/status` — CLI detection, version, subscription type, rate limit tier
- **Dashboard:** Settings → Cloud Providers → Claude Subscription card with step-by-step setup guide, status detection, enable/disable toggle

### Admin API (CORS-protected, no Bearer auth)
- `GET/PUT /api/admin/config` — Configuration read/update
- `GET     /api/admin/stats` — Node statistics + hardware info
- `GET     /api/admin/swarm/capacity` — R110: collective capacity snapshot (online_nodes, total_vram_mb, serveable/aspirational/hosted_locally model lists, redundancy)
- `GET     /api/admin/swarm/capacity-plan` — R113: what-if scenarios + headline_target with concrete `contributors_needed` count
- `GET     /api/admin/storage/breakdown` — R110: stacked-bar data (total_mb, used_mb, auto_target_mb, free_mb)
- `GET     /api/admin/wishlist` — R111: ranked list of models the swarm wants (status, score, why_tags, swarm_replicas, target_replicas)
- `GET     /api/admin/hf/trending` — R112: cached HuggingFace trending-GGUF snapshot from HfWatcher
- `GET     /api/admin/responses` — List stored `/v1/responses` records for the dashboard (filter by `?status=…&limit=…`)
- `GET     /api/admin/models` — Model list with shard status, VRAM estimates, acquisition state
- `POST    /api/admin/models/{id}/add` — Trigger model acquisition
- `GET     /api/admin/models/{id}/status` — Model acquisition progress
- `GET     /api/admin/peers` — Connected peers with latency/trust
- `GET     /api/admin/credits` — Credit balance and tier info
- `GET     /api/admin/shard-storage` — Per-model storage breakdown, disk/VRAM usage
- `GET     /api/admin/api-key` — Retrieve API key (Bearer auth required)
- `POST    /api/admin/ws-ticket` — Issue a single-use 30s ticket (Bearer auth) — required pre-step for the WS upgrade
- `GET     /api/admin/ws` — WebSocket for live updates (consumes a ws-ticket)
- `GET     /api/admin/downloads` — Download queue with priorities and progress

### HuggingFace Integration
- `GET  /api/admin/hf/search?q=...` — Search HuggingFace for GGUF models (grouped by repo with quant variants). Note the parameter is `q`, NOT `query` (axum deserialises via `#[serde(rename = "q")]`).
- `GET  /api/admin/hf/probe?repo_id=...&filename=...` — Probe remote GGUF (size, shard layout)
- `POST /api/admin/hf/download` — Download full GGUF model. **⚠ Deprecated** for normal use — the frontend and all new code MUST use `/api/admin/hf/download-shards`. Full-GGUF download exists only for offline-inference / seeding workflows; never call it implicitly. See CLAUDE.md § "No implicit full model downloads".
- `POST /api/admin/hf/download-shards` — Download specific shard indices (supports `peer_fair_share` for smart distribution). **Preferred entry point.**
- `GET  /api/admin/hf/source/{model_id}` — Lookup HuggingFace source info for a model
- `GET  /api/admin/hf/search?q=...&tasks=chat,code,...` — R114: optional `tasks` filter narrows results to chat/code/vision/multilingual/reasoning task tags (server-side filter)

### Identity API
- `GET/PUT/DELETE /api/identity/nickname` — Manage local nickname
- `GET           /api/identity/leaderboard` — Network-wide credit leaderboard
- `GET           /api/identity/peers` — Peer identity directory

### Pool API
- `GET  /api/pool/state` — Current pool membership state
- `POST /api/pool/create` — Create a new device pool
- `POST /api/pool/invite` — Invite a node to the pool (by node_id)
- `POST /api/pool/accept` — Accept a pool invitation
- `POST /api/pool/remove` — Remove a member from the pool
- `POST /api/pool/leave` — Leave the current pool
- `GET  /api/pool/invitations` — List pending invitations
- `GET  /api/pool/leaderboard` — Pool member contribution rankings
- `POST /api/pool/generate-code` — Generate an 8-char invite code (owner only)
- `POST /api/pool/join` — Join a pool via invite code
- `POST /api/pool/device-name` — Set this device's display name
- `PUT  /api/pool/credit-split` — Set credit split percentage (owner only)
- `PUT  /api/pool/contribution` — Set per-member contribution level (owner only, `{"node_id": "...", "level": 75}` where level is 0–100)

### Discovery
- `GET    /api/admin/network-code` — Get shareable invite code, multiaddr, and network phase
- `POST   /api/admin/join-network` — Join network via invite code or multiaddr

### Utility
- `POST   /api/admin/shutdown` — Gracefully shut down the node (localhost only)
- `POST   /api/admin/config/reload` — Hot-reload operational config parameters
- `POST   /api/admin/downloads/{model_id}/cancel` — Cancel in-progress HF download
- `DELETE /api/admin/models/{model_id}` — Remove model (shards + manifest + state)
- `POST   /api/admin/models/{id}/unload` — Unload model from VRAM (keep shards on disk)
- `DELETE /api/admin/models/{id}/shards/{index}` — Delete a single shard
- `GET/PUT /api/admin/models/{id}/auto-manage` — Per-model auto-manage policy (incl. prune toggle)
- `PUT    /api/admin/models/{id}/shards/{index}/lock` — Lock/unlock a shard (prevent auto-pruning)
- `POST   /api/admin/models/{id}/shards/{index}/download` — Download a single shard from P2P network
- `POST   /api/admin/models/{id}/shards/{index}/load` — Load a shard into memory (expands shard window, restarts worker)
- `POST   /api/admin/models/{id}/shards/{index}/unload` — Unload a shard from memory (narrows shard window, restarts worker, frees RAM/VRAM)
- `GET/PUT /api/admin/schedule` — Resource schedule management
- `GET    /api/admin/prune-history` — Recent auto-prune events
- `GET/POST /api/admin/adapters` — List/register LoRA adapters
- `DELETE /api/admin/adapters/{id}` — Delete a LoRA adapter
- `GET/PUT /api/admin/providers` — View/configure cloud provider API keys
- `GET    /api/admin/provider-models` — List models available from cloud providers
- `GET    /api/admin/provider-health` — Probe cloud provider availability
- `POST   /api/admin/provider-model-status` — Check specific model availability on provider
- `GET    /api/admin/version` — Version info (binary version, git hash, build features)
- `POST   /api/admin/update/check` — Check for new SwarmLLM releases
- `POST   /api/admin/update/apply` — Download and apply update
- `GET    /api/admin/network-map` — Network topology heatmap data
- `GET    /api/admin/models/{id}/metadata` — GGUF metadata (context length, quantization, layers)
- `GET/PUT /api/admin/models/{id}/encrypted-pipeline` — Per-model encrypted pipeline policy
- `POST   /api/admin/rescan-shards` — Hot-reload shard files from disk without restart
- `GET/PUT /api/admin/pools/{id}/rates` — Get/set credit rate configuration for a pool
- `GET    /metrics` — Prometheus/OpenMetrics endpoint (no auth)
- `GET    /health/ready` — Readiness probe with subsystem status (no auth)

### Static
- `/admin` — Dashboard SPA (single-page app — all routes serve index.html)
- `/chat` — Chat interface
- `/setup` — First-run wizard
- `/static/*path` — Embedded static assets (CSS, JS, i18n JSON)
- `/static/i18n/{lang}.json` — Translation files (21 languages)
- `/health` — Liveness probe (`{"status": "ok"}`, no auth)
- `/` → redirect to `/admin`

### Frontend Architecture
- **No build step**: Vanilla HTML/CSS/JS — no framework, no bundler, no Node.js
- **Component architecture**: `App` global namespace with component sub-objects (`App.chat`, `App.dashboard`, etc.)
  - `frontend/js/core/state.js` — App namespace, shared mutable state, theme, storage keys
  - `frontend/js/core/utils.js` — format helpers (`formatBytes`, `formatDlProgress`, `escapeHtml`, etc.), DOM builders (`appendMessageToDOM`, `createEmptyState`), `extractErrorMessage`, `getApiErrorMessage`
  - `frontend/js/core/data.js` — data store with in-flight deduplication, `authFetch` wrapper
  - `frontend/js/core/tooltip.js` — unified popover replacing native `title=` attributes
  - `frontend/js/components/ui.js` — tab switching, banners, mode indicator, sidebar
  - `frontend/js/components/chat.js` — sessions, messages, SSE streaming, image upload, layout toggle
  - `frontend/js/components/claude-code.js` — Claude Code interactive sessions (subprocess, permission flow, SSE)
  - `frontend/js/components/dashboard.js` — stats, hardware, model cards, peers, shard grid live updates
  - `frontend/js/components/dashboard-shards.js` — pure-function shard HTML builders (progress bar, shard row, matrix, coverage ribbon); exposes `App.dashboardShards`, loaded before `dashboard.js`
  - `frontend/js/components/models.js` — model dropdown, HF search/download, auto-manage, metadata panel
  - `frontend/js/components/auto-manage-status.js` — auto-manage scan/VRAM-pressure status display
  - `frontend/js/components/settings.js` — settings panel (API keys, config, contribution)
  - `frontend/js/components/setup.js` — first-run setup wizard (3-step configuration)
  - `frontend/js/components/downloads.js` — download queue, prune history, resource schedule
  - `frontend/js/components/notifications.js` — unified event handler, toasts, WebSocket, REST polling, provider health
  - `frontend/js/components/identity.js` — network invite code, nickname, leaderboard
  - `frontend/js/components/network-map.js` — regional network map visualization
  - `frontend/js/components/compare.js` — multi-model comparison tool
  - `frontend/js/components/responses.js` — `/v1/responses` dashboard panel: retrieve-by-id, status-filtered list, cancel/delete/view per row, 5-second polling refresh while visible
  - `frontend/js/components/pool.js` — device pool management (create, join, members, contribution)
  - `frontend/js/init.js` — event binding, initialization, public API export (`window.SwarmLLM`)
- **HTML templates**: 11 `<template id="tmpl-*">` elements for repeating UI structures (session items, chat messages, toasts, compare cards, compare model chips, leaderboard rows, download queue items, peer rows, prune rows, storage model rows, pool member rows). Components clone templates via `template.content.cloneNode(true)` instead of innerHTML string building.
- Cross-component calls: `App.componentName.method()`. Shared state: `App.state.*`. Utilities: `App.utils.*`.

### Frontend Features
- **i18n**: 1122 translation keys (1124 entries per locale incl. `_lang` + `_dir`) across 21 languages (en, es, fr, de, pt, it, nl, ru, zh, ja, ko, ar, tr, pl, sv, th, hi, vi, id, uk, cs). Auto-detects browser language. `I18n.t()` + `data-i18n` DOM attributes. Interpolation via `{variable}` placeholders. Fallback chain: current language → English → raw key. "Continue in English" UX for non-English users who prefer English.
- **Theme**: Light / Dark / System toggle. `[data-theme="light"]` CSS overrides. Persisted in localStorage.
- **Neural network background**: Animated canvas particle network behind dashboard tiles (`frontend/js/neural-bg.js`). ~60 nodes with connecting edges, gentle drift, mouse repulsion/glow. State-reactive coloring: blue (idle) → cyan (active inference) → red-orange (unhealthy/disconnected). Peer count boosts vibrancy, active requests trigger node firing pulses. Pauses when tab hidden; reduced opacity in light theme.

## Activity Event System

A lightweight cross-subsystem event bus for real-time dashboard observability.

**Backend** (`ActivityEvent` defined in `src/daemon/state/activity.rs`, re-exported from `state/mod.rs`):
- `ActivityEvent` struct with fields: `category` (`&'static str`), `kind` (`&'static str`, e.g. `"shard_pruned"`), `message` (English), plus optional `model_id`, `model_name`, `node_id`, `detail_num`, `detail_str`, `toast_level`, `toast_duration_ms`, `shard_index`, `freed_bytes`, `holder_count_before`, `holder_count_after`, `remaining_local_shards`, `timestamp` (ISO 8601)
- `activity_tx: broadcast::Sender<ActivityEvent>` in `state.events` sub-struct (capacity 256, oldest events dropped on overflow)
- All 12 subsystems emit events via the `state.emit_activity(ActivityEvent::new(...))` builder — fire-and-forget (send errors ignored)
- Example event kinds (snake_case strings; see `ACTIVITY_ICONS` in `frontend/js/components/notifications.js` for the canonical list): `shard_download_complete`, `shard_pruned`, `inference_request`, `inference_completed`, `peer_connected`, `peer_disconnected`, `model_loaded`, `model_unloaded`, `worker_spawned`, `worker_unloaded`, `pool_device_joined`, `pool_created`, `config_updated`, `daemon_started`, and many more

**WebSocket delivery** (`src/api/websocket.rs`):
- ApiServer subscribes to `state.events.activity_tx` on WebSocket upgrade
- Events sent to client as `{"type":"activity_event","data":{...}}` JSON messages
- Dropped messages (slow client) are non-fatal — buffer overflow discards oldest events

**Frontend** (`js/components/dashboard.js`, `js/components/notifications.js`):
- Global activity log persisted to `sessionStorage` (survives tab refresh within the session)
- Category-based color coding by event kind (inference = blue, download = green, prune = orange, error = red, etc.)
- Per-model activity ticker: latest event shown inline on each model card; hover expands to last 5 events
- Global Activity panel: chronological log of all events with relative timestamps ("just now", "5m ago")
- Shard flash animation: model card shard cells glow white on `ShardDownloaded`/`ShardPruned` events

## Node Tiers

| Tier | Requirements | Role |
|---|---|---|
| Super Node | Full model in VRAM/RAM, high bandwidth | Full inference, backbone |
| Standard Node | Partial VRAM/RAM, moderate bandwidth | Shard hosting, pipeline participation |
| Light Node | Minimal resources | Consumer, bandwidth contribution |

## Platform Targets

| Platform | Priority | GPU Support |
|---|---|---|
| Linux x86_64 | P0 | CUDA (llama.cpp + candle) + ROCm (llama.cpp) |
| macOS aarch64 | P1 | Metal (via llama.cpp) |
| Windows x86_64 | P1 | Vulkan (llama.cpp, all vendors) + CUDA static (candle, NVIDIA) |
| macOS x86_64 | P2 | CPU only |
| Linux aarch64 | P3 | CPU only |

### Windows GPU Distribution Strategy

Windows uses a three-binary installer (`SwarmLLM-Setup.exe`) to support all GPU vendors without requiring users to install CUDA Toolkit or Vulkan SDK:

- **`swarmllm-gpu.exe`** — built with `--features windows-gpu`:
  - llama.cpp local inference via Vulkan (NVIDIA, AMD, Intel — `vulkan-1.dll` bundled with all GPU drivers)
  - candle distributed/split inference via CUDA with static runtime (`cudart_static` linked in — needs only GeForce drivers, not CUDA Toolkit)
- **`swarmllm-cpu.exe`** — CPU-only, works on any Windows PC
- **`swarmllm.exe`** (launcher) — detects `nvcuda.dll` in System32 at startup, transparently execs the appropriate binary

**AMD/Intel on Windows**: Local inference is GPU-accelerated via Vulkan. Split/distributed inference falls back to CPU — acceptable since serious multi-GPU distributed setups are predominantly NVIDIA.

### Future: wgpu/WebGPU Backend for Split Inference

The current candle-based split/distributed inference path supports CUDA (NVIDIA) only. A wgpu backend would enable cross-platform GPU acceleration (NVIDIA, AMD, Intel) for distributed inference using Vulkan, DX12, or Metal under the hood — eliminating the NVIDIA-only limitation for the distributed path.

Tracked upstream in [huggingface/candle](https://github.com/huggingface/candle). No stable wgpu backend exists as of 2026-03. When available, enable via a new `candle-wgpu` feature that activates `candle-core/wgpu`. The `windows-gpu` feature would then include it alongside llama-vulkan, giving full cross-vendor GPU support for both local and distributed inference.

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

## Deferred Items

The list is split into **open** (will be addressed) and **won't fix unless a concrete caller appears** (the work is understood but not justified by current demand). Per-finding history (status, resolution, deferral) is tracked in `.claude/sweep-log.jsonl`.

### Open

- **Binary signature on auto-update (audit_2026-04-29 C1)** — `src/update.rs` verifies the SHA256 sidecar fetched from the same GitHub release as the binary; a compromised maintainer account/CI token can publish a matching pair. Real fix: generate an offline signing keypair, embed the public key at compile time, publish a detached signature as a third release asset, and verify it before applying the rename. Deferred until a key-custody decision is made — see `memory/signing_options.md` for the three concrete options (raw Ed25519, minisign, or Sigstore/Cosign keyless), recommended approach (minisign), and step-by-step rollout plan. Until landed, defence-in-depth fixes keep the blast radius local: `update/check` + `update/apply` are loopback-only (`2e1c5b1`), `apply_update` re-checks `latest_version > running_version` at apply time (post-`cb2c688`), `info.downloaded` only flips true when the staging path is on the same filesystem as the binary, and auto-update is opt-in via `config.updates.auto_update` (default `Disabled`).

### Won't fix unless a concrete caller appears

- **Speculative decoding in subprocess** — IPC scaffolding for routing speculative decoding through worker subprocesses was removed; the path runs through the direct executor only. Speculative decoding is experimental and the worker-subprocess plumbing would need to track per-position logit returns, KV-cache state, and partial-accept truncation — substantial complexity for a feature whose target audience overlaps tightly with the user base that already has the legacy in-process executor working. Revisit if subprocess isolation becomes a hard requirement (e.g., per-model crash containment for a hosted deployment).

- **Local executor streaming serialization** — `executor.lock().await` in `api/openai/streaming.rs::stream_response` and `api/anthropic/mod.rs` holds the mutex for the entire streaming inference duration, serializing concurrent local streaming requests. Only affects the legacy single-GGUF executor path; the modern split-model path (`split_stream_response`) and distributed paths route through `ModelProcessPool` and are unaffected. The "fix" of routing legacy through `ModelProcessPool` is misleading: the pool only handles shard-based models, so closing this gap requires either teaching the pool to load full GGUFs (large rearchitecture) or retiring the legacy executor entirely. Documented limitation: legacy GGUF mode is single-stream-at-a-time; users who need concurrency should switch to shard mode.

- **V9: `POST /v1/responses/compact`** — Responses-API summary/compaction endpoint. No concrete caller has asked for it. Implement when one shows up.

- **Server-side `conversation` resource CRUD** — OpenAI's `conversation` parameter forwards through cloud proxy verbatim. A local conversation type with its own endpoints is a separate design that nobody is currently blocked on.

- **Built-in tools on the local path** (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`, `custom`) — rejected with 400 on local; forwarded verbatim on cloud. Each requires backing infrastructure (web crawler, code sandbox, image-gen model, etc.) that SwarmLLM intentionally does not run.

- **`custom` tools with Lark / regex grammars** — rejected on local, forwarded on cloud. Local grammar-constrained generation is a candle-side project; we can't ship until candle exposes the necessary sampler hooks.

- **Audio input on `/v1/responses`** — `input_audio` returns 400. Needs a Whisper-class transcription model SwarmLLM doesn't currently expose.

- **Binary file inputs** — `input_file{file_data}` accepts UTF-8 only; PDF / docx / image-bytes payloads are rejected with a clear hint pointing at `input_image` (for images) or server-side text extraction (for documents). A PDF parser is a deferred call-site question.

- **Synthetic tiny-model fixture (`tests/fixtures/tiny_model/`)** — empty placeholder. Originally specced as a 2-layer / 128-hidden / 2-shard llama-arch GGUF (~1 MB) committed to the repo so a multi-process spawn-and-infer integration test could run in CI without network. Stays unbuilt for two reasons: (1) generating a valid GGUF + matching `manifest.json` + `gguf_header.bin` + tokenizer requires a Python `gguf`-library generator script we don't maintain and that would version-drift against candle-transformers / our split loader; (2) random-weight outputs are gibberish, so the test would only catch GGUF-parse and worker-IPC plumbing bugs — both already covered by `tests/integration/end_to_end.rs` and `inference::split` unit tests. Pragmatic substitute: the env-var-gated `local_embedder_load_from_real_model` test (`SWARMLLM_TEST_MODEL_DIR`) and manual smoke tests against the TinyLlama-1.1B / Phi-3.5 / Qwen2.5-7B installs at `~/.local/share/swarmllm/models/`. Revisit if a CI worker-subprocess regression slips past the unit + in-process layers.

### Recently closed

- **ChaCha session encryption on speculative / DSD / remote-generate fast paths** (closed 2026-05-07) — the gate in `speculative_common_eligible` was stale conservatism; the encryption layer was already wired through `handle_send_tensor` and `encode_forward_for_wire` after the original fast-path commits, but the eligibility check was never re-evaluated. Removed the `enable_encryption` check; spec verify trailers (draft_tokens marker 0x03, truncate_kv_to marker 0x04) ride alongside the sealed activations in the encrypted envelope.
- **DSD multi-segment requires all-remote segments** (closed 2026-05-07) — `pipeline/dsd.rs::forward_verify_through_segments` now branches on `peer_id_for_segment[idx]`: `Some(peer_bytes)` keeps the existing `NetworkCommand::SendTensor` rendezvous; `None` dispatches to `model_process_pool.forward(layer_forward)`. The worker's `forward_verify_all_positions[_pre_embedded]` already gates `spec_logits` emission on `is_last`, so mixed-local pipelines produce hidden state for non-last local segments and `spec_logits` only on the final segment.
- **Spec-trailer fields now in AAD** (closed 2026-05-07) — `build_layer_forward_aad` was extended to append the spec trailer (marker 0x03 + flags + num_drafts + drafts) and kv-truncate trailer (marker 0x04 + target_len) bytes whenever those trailers are emitted on the wire. `decode_layer_forward_encrypted` now reconstructs AAD via the helper after parsing trailers instead of slicing the wire-bytes header. An active MITM cannot flip `spec_logits_requested` or modify `truncate_kv_to` without invalidating Poly1305. Wire-protocol bump for encrypted mode; mixed-version encrypted clusters fail decrypt during upgrade (acceptable for alpha — encrypted mode is opt-in).
- **API-key bootstrap nonce** (closed 2026-05-07) — `/api/admin/api-key` previously accepted `Sec-Fetch-Site: same-origin` as a browser-only signal, but curl/python can set that header. Replaced with per-page nonces: `/admin` (and `/chat`, `/setup`, plus catchalls) now routes through `serve_dashboard_with_nonce`, which substitutes a fresh 32-byte nonce into the served HTML's `<meta name="bootstrap-nonce">` tag and registers it in `AppState.bootstrap_nonces` with a 60s TTL. The dashboard JS sends the nonce as `X-Dashboard-Nonce` on its bootstrap fetch; the middleware validates and consumes it (one-time use). Cross-UID local attackers must now first scrape `/admin` to obtain a nonce — strictly raises the bar; bare curl/spoofed-header attacks return 401. Same-UID processes can still read the `api_key` file directly (mode 0o600), so this is a partial fix targeting the cross-UID threat model.

## Scalability (Phase 19)

Tested up to 5 nodes on real hardware (Proxmox). Estimated capacity: ~10K nodes.

| Mechanism | Description | Impact |
|---|---|---|
| Gossip frequency scaling (S4) | Broadcast interval = `log(peer_count)` × 30s | ~8× less gossip at 10K nodes |
| Shard announce delta compression (S1) | Only broadcasts when shard set changes + periodic re-announce | ~90% less shard announce traffic |
| P2P shard fallback (S2) | Auto-manage tries P2P when no HF source available | Distributes download load |
| Peer registry cap (S3) | Max 200 peers, evicts highest-latency non-LAN | Memory bounded O(1) not O(N) |
| Target replicas | `ceil(log2(pool_size))` × demand_factor | 10 replicas at 1K, 14 at 10K |
| Consistent hash ring | 10 virtual slots per node for shard assignment | Prevents thundering herd |
| Regional gossip summaries | O(regions × models) not O(nodes × shards) | Scales with regions not nodes |
| GossipSub mesh auto-scale | mesh_n/mesh_n_high scale with known_peers | Handles 1K+ peers |

### Bottlenecks beyond 10K nodes
- GossipSub message volume (linear with models × publishers)
- May need topic sharding (per-model-family topics instead of single `swarm/models`)
- mDNS doesn't scale beyond LAN — bootstrap peers or DHT-only discovery needed

## Smart VRAM Management (Phase 20)

Four features for intelligent VRAM and model management:

### Shard Windows (Smart VRAM Unload)
Workers load only a subset of on-disk shards into GPU memory. Shards stay on disk (still advertised to the network) but VRAM is freed by killing and restarting the worker with a narrower window.

- `ModelProcessPool.active_shard_windows`: per-model `Vec<u32>` of allowed shard indices
- `--shard-window 0,1,7` CLI arg passed to `model-worker` subprocess
- Auto-triggered by prune at VRAM pressure 0.7–0.95 (before hard-delete at 0.95+)
- `compute_optimal_shard_window()` always prefers shard 0 (embeddings) and last shard (output head) for boomerang inference
- API: `in_vram` field per shard in model detail; frontend shows V/D badges

### Bandwidth-Based Speed Estimation
- `gpu_memory_bandwidth_gbps()`: 30-GPU lookup table (RTX 20/30/40, A100, H100, Apple M-series, AMD)
- `estimate_tokens_per_sec_7b()`: `bandwidth / 4.4GB * efficiency` (0.30 GPU, 0.15 CPU)
- Gossiped via `NodeCapability.est_tokens_per_sec_7b`
- Used as scheduler tie-breaker (after latency, region, load, trust)

### MoE-Aware VRAM Accounting
- `estimate_model_vram_mb_arch()`: `active_fraction = 0.40 + 0.60 * (experts_per_token / num_experts)`
- Supports Mixtral, DeepSeek, Llama4, Qwen35Moe architectures
- Used in auto-manage scoring for accurate VRAM fitness

### Pre-Scored HF Model Browsing
- `composite_score = quality × fit × demand × size × 100` (0–150 range)
- `quality = log10(downloads + 10) / 7.0`; `fit` = boomerang:1.0, shard:0.6, none:0.1
- Server-side default sort by score; client-side sort dropdown (score/downloads/size)
- Score badge with color coding; `score_breakdown` in API response
