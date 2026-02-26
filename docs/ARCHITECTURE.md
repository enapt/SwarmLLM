# SwarmLLM Architecture Reference

## System Overview

Single Rust binary, three simultaneous functions:

```
┌──────────────────────────────────────────────────────────┐
│                      swarmllm binary                      │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │       │
│  │  Node    │  │  Server      │  │  (embedded)  │       │
│  │  (QUIC)  │  │  (Axum)      │  │              │       │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘       │
│       │               │                 │                │
│  ┌────┴───────────────┴─────────────────┴─────────────┐  │
│  │              Shared State (Arc)                     │  │
│  │  DashMap<NodeId, PeerInfo>      — peer registry     │  │
│  │  ModelRegistry                  — models + shards   │  │
│  │  DashMap<ShardId, Vec<NodeId>>  — shard locations   │  │
│  │  DashMap<Uuid, PipelineAssignment> — pipelines      │  │
│  │  Arc<RwLock<CreditBalance>>     — credit balance    │  │
│  │  RwLock<NodeStats>              — node statistics    │  │
│  │  SharedExecutor                 — llama.cpp model    │  │
│  │  DashMap<Blake3Hash, VoteTally> — model votes       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Daemon Task Architecture

```
                        ┌──────────────┐
                        │   daemon.rs  │
                        │  (bootstrap) │
                        └──────┬───────┘
                               │ spawns tokio tasks
       ┌───────┬───────┬───────┼───────┬───────────┬──────────┬──────────┐
       ▼       ▼       ▼       ▼       ▼           ▼          ▼          ▼
┌──────────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐
│ Network  │ │Infer│ │Credit│ │Health│ │ API  │ │Rebal-│ │Acquisi-│ │Message │
│ Manager  │ │Router│ │Ledger│ │Mon.  │ │Server│ │ancer │ │tion Mgr│ │Dispatch│
└────┬─────┘ └──┬──┘ └──┬──┘ └──┬───┘ └──┬───┘ └──┬───┘ └───┬────┘ └───┬────┘
     │          │       │       │         │        │          │          │
     └──────────┴───────┴───────┴─────────┴────────┴──────────┴──────────┘
                         mpsc channels between tasks
```

### Channel Layout

| From | To | Channel | Message Types |
|---|---|---|---|
| NetworkManager | MessageDispatcher | `network_out_tx` | All inbound SwarmMessage variants |
| MessageDispatcher | InferenceRouter | `router_cmd_tx` | InferenceRequest, LayerForward, LayerResult, PipelineAssignment, InferenceError |
| InferenceRouter | NetworkManager | `network_tx` | SwarmMessage (outgoing P2P) |
| HealthMonitor | NetworkManager | `network_tx` | HealthPing |
| HealthMonitor | ShardRebalancer | `rebalance_tx` | RebalanceEvent |
| ApiServer | InferenceRouter | `router_cmd_tx` | RouterCommand (from HTTP) |
| ApiServer | AcquisitionManager | `acquisition_tx` | AcquisitionCommand (model download) |
| AcquisitionManager | NetworkManager | `network_tx` | ShardAnnounce (shard requests) |
| CreditLedger | NetworkManager | `network_tx` | CreditGossip, CreditTransaction |

The **MessageDispatcher** is a dedicated task in `daemon.rs` that routes inbound network messages to the appropriate subsystem. Inference messages go to InferenceRouter, CreditGossip updates peer balance distributions, and ModelVote is processed by the model governance module.

## Startup Sequence

```
1.  Parse CLI args (clap) — including optional --model and --gpu-layers
2.  Initialize tracing subscriber (verbosity: info → debug → debug+libp2p → trace)
3.  Load or create config (TOML + env + defaults + CLI overrides)
4.  Ensure data directory exists
5.  Load or generate Ed25519 identity
6.  Open sled database
7.  Build Daemon { config, identity, db }
8.  Initialize ModelExecutor (load GGUF model if --model provided)
9.  Build Arc<SharedState> (includes ModelRegistry loaded from DB)
10. Scan local shards → register in model_registry + shard_registry
11. Create mpsc channels (network, router, rebalance, acquisition)
12. Spawn all tasks (8 tasks: NetworkManager, InferenceRouter, MessageDispatcher,
    HealthMonitor, ShardRebalancer, CreditLedger, AcquisitionManager, ApiServer)
13. Open browser if ui.open_browser_on_start is true (setup wizard or admin)
14. tokio::select! on Ctrl+C signal or any task exit
15. Signal graceful shutdown via watch channel
```

## Networking Stack

```
libp2p Swarm
├── Kademlia (DHT)
│   ├── /swarm/node/{node_id}         → NodeCapability
│   ├── /swarm/shard/{model}/{index}  → Vec<NodeId>
│   └── /swarm/model/{model_id}       → ModelManifest
│
├── GossipSub (pub/sub)
│   ├── swarm/models/{model_id}       → ShardAnnounce, capacity
│   ├── swarm/governance              → ModelVote
│   └── swarm/health                  → trust summaries
│
├── request_response
│   └── Direct inference messages (LayerForward, LayerResult, ShardRequest)
│
├── Identify (protocol identification)
├── AutoNAT (NAT detection)
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

| Feature | Llama | Qwen2 |
|---------|-------|-------|
| RoPE variant | Interleaved (`rope_i`) | Contiguous (`rope`) |
| QKV biases | None | `attn_q.bias`, `attn_k.bias`, `attn_v.bias` |
| Context length | 4096 (default) | 32768 (from metadata) |
| EOS tokens | 2 | 151643, 151645 |

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

### Pipeline Assembly Algorithm

1. Fetch model manifest → determine layer ranges
2. Query shard_registry for hosting nodes
3. Fetch node load/latency from peer_registry
4. Sort candidates by (latency ASC, load ASC, trust DESC)
5. Greedy assignment: widest contiguous layer range per node
6. Merge contiguous segments assigned to the same node
7. Identify standby nodes per segment
8. Send PipelineAssignment → all nodes ACK → begin forwarding

### KV-Cache Management

- Each SplitModel maintains per-layer KV-cache
- KV-cache is cleared when `sequence_num == 0` (start of new request)
- `index_pos` travels through the wire protocol so all nodes apply correct RoPE positioning
- Position tracking: `index_pos = prompt_token_count` after prefill, increments by 1 per decode step

## Credit System

```
Earning:
  +10 credits  per layer per token served
  +1  credit   per GB per hour hosting shards
  +5  credits  per GB seeding shard data
  +2  credits  per connection hour relay service

Spending:
  -8  credits  per layer per token requested
  -50 credits  per serve failure (timeout)

Tiers:
  Platinum  (≥90th percentile)  → immediate queue
  Gold      (≥70th percentile)  → 1-3s queue
  Silver    (positive balance)  → 5-15s queue
  Bronze    (zero/negative)     → 30s+ queue
```

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
          │ request from peers
          ▼
 ┌─────────────────┐     Per-chunk write
 │  Download Loop  │────────────────▶ Progressive to disk
 └────────┬────────┘
          │ complete shard
          ▼
 ┌─────────────────┐     BLAKE3 vs manifest
 │ Shard Verify    │────────────────▶ Quarantine + penalize on mismatch
 └────────┬────────┘
          │ all shards verified
          ▼
    Model Ready
```

**Key invariants**:
- Manifests MUST come from the network registry, not from disk
- Manifest integrity is verified (BLAKE3 self-hash) before trusting shard hashes
- Each downloaded shard is verified against the manifest hash
- Failed shards are renamed `.bin.quarantine` and the serving peer's trust is penalized
- On startup, `load_all_local()` rejects model directories without a valid manifest
- On startup, every existing shard is re-verified against its manifest hash

**AcquisitionManager** (`src/model/acquisition.rs`) orchestrates this flow as a
long-running Tokio task, receiving commands via `mpsc` from the API server.

## Data Directory

```
~/.swarmllm/
├── config.toml          # User configuration
├── identity.key         # Ed25519 keypair (optionally encrypted)
├── db/                  # sled database
└── models/
    ├── llama3-70b-q4km/
    │   ├── manifest.json
    │   ├── tokenizer.json
    │   ├── shard_000.bin
    │   └── ...
    └── mistral-7b-q5km/
        └── ...
```

## sled Database Trees

| Tree | Key | Value |
|---|---|---|
| config | "config" | Config |
| identity | "keypair" | Encrypted Ed25519 key |
| credits | "balance" | CreditBalance |
| credit_txns | {uuid} | CreditTransaction |
| peer_trust | {node_id_hex} | TrustScore |
| shard_meta | {model_id}/{shard_index} | ShardInfo + path |
| model_meta | {model_id} | ModelManifest |
| sessions | {session_id} | KV-cache metadata |

## HTTP API Routes

### OpenAI-Compatible
- `POST /v1/chat/completions` — Chat completions (streaming + non-streaming)
- `POST /v1/completions` — Text completions
- `GET  /v1/models` — List available models
- `GET  /v1/status` — SwarmLLM node status

### Admin API
- `GET/PUT /api/admin/config` — Configuration read/update
- `GET     /api/admin/stats` — Node statistics + hardware info
- `GET     /api/admin/models` — Model list with shard status
- `POST    /api/admin/models/:id/add` — Trigger model acquisition
- `GET     /api/admin/models/:id/status` — Model acquisition progress
- `GET     /api/admin/peers` — Connected peers with latency/trust
- `GET     /api/admin/credits` — Credit balance and tier info
- `GET     /api/admin/ws` — WebSocket for live updates

### HuggingFace Integration
- `GET  /api/admin/hf/search?q=...` — Search HuggingFace for GGUF models
- `POST /api/admin/hf/download` — Start downloading a GGUF model

### Utility
- `POST /api/admin/shutdown` — Gracefully shut down the node
### Static
- `/admin` — Dashboard SPA (single-page app — all routes serve index.html)
- `/chat` — Chat interface
- `/setup` — First-run wizard
- `/static/*path` — Embedded static assets (CSS, JS)
- `/health` — Health check endpoint
- `/` → redirect to `/admin`

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
