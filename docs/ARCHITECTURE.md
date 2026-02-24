# SwarmLLM Architecture Reference

## System Overview

Single Rust binary, three simultaneous functions:

```
┌─────────────────────────────────────────────────────┐
│                   swarmllm binary                    │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  P2P     │  │  HTTP API    │  │  Admin UI    │  │
│  │  Node    │  │  Server      │  │  (embedded)  │  │
│  │  (QUIC)  │  │  (Axum)      │  │              │  │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘  │
│       │               │                 │           │
│  ┌────┴───────────────┴─────────────────┴────────┐  │
│  │              Shared State (Arc)                │  │
│  │  DashMap<NodeId, PeerInfo>                     │  │
│  │  DashMap<ModelId, ModelManifest>               │  │
│  │  DashMap<ShardId, Vec<NodeId>>                 │  │
│  │  DashMap<Uuid, PipelineAssignment>             │  │
│  │  RwLock<CreditBalance>                         │  │
│  │  RwLock<NodeStats>                             │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Daemon Task Architecture

```
                        ┌──────────────┐
                        │   daemon.rs  │
                        │  (bootstrap) │
                        └──────┬───────┘
                               │ spawns tokio tasks
       ┌───────┬───────┬───────┼───────┬───────────┬──────────┐
       ▼       ▼       ▼       ▼       ▼           ▼          ▼
┌──────────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐
│ Network  │ │Infer│ │Credit│ │Health│ │ API  │ │Rebal-│ │Acquisi-│
│ Manager  │ │Router│ │Ledger│ │Mon.  │ │Server│ │ancer │ │tion Mgr│
└────┬─────┘ └──┬──┘ └──┬──┘ └──┬───┘ └──┬───┘ └──┬───┘ └───┬────┘
     │          │       │       │         │        │          │
     └──────────┴───────┴───────┴─────────┴────────┴──────────┘
                      mpsc channels between tasks
```

### Channel Layout

| From | To | Channel | Message Types |
|---|---|---|---|
| NetworkManager | InferenceRouter | `network_out_tx` | InferenceRequest, LayerForward, LayerResult |
| InferenceRouter | NetworkManager | `network_tx` | SwarmMessage (outgoing P2P) |
| InferenceRouter | CreditLedger | `credit_tx` | CreditTransaction |
| HealthMonitor | NetworkManager | `network_tx` | HealthPing |
| HealthMonitor | ShardRebalancer | `rebalance_tx` | RebalanceEvent |
| ApiServer | InferenceRouter | `router_cmd_tx` | RouterCommand (from HTTP) |
| ApiServer | AcquisitionManager | `acquisition_tx` | AcquisitionCommand (model download) |
| AcquisitionManager | NetworkManager | `network_tx` | ShardAnnounce (shard requests) |

## Startup Sequence

```
1. Parse CLI args (clap)
2. Initialize tracing subscriber
3. Load or create config (TOML + env + defaults)
4. Load or generate Ed25519 identity
5. Open sled database
6. Build Daemon { config, identity, db }
7. Create mpsc channels
8. Build Arc<SharedState>
9. Spawn all tasks
10. tokio::select! on shutdown signal or task exit
11. Graceful shutdown
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
│   ├── swarm/health                  → trust summaries
│   ├── swarm/gov/proposals           → Proposals, amendments
│   ├── swarm/gov/votes               → ProposalVote
│   ├── swarm/gov/issues              → Issues, comments
│   └── swarm/gov/releases            → ReleaseCandidates, TestReports
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

```
Client → API Server → InferenceRouter → Pipeline Assembly
                                              │
                      ┌───────────────────────┘
                      ▼
          ┌──────────────────────┐
          │   Pipeline Segment   │
          │ Node A: Layers 0-15  │──── LayerForward ───▶
          └──────────────────────┘                      │
                                        ┌───────────────┘
                                        ▼
                            ┌──────────────────────┐
                            │   Pipeline Segment   │
                            │ Node B: Layers 16-47 │── LayerForward ──▶
                            └──────────────────────┘                   │
                                                       ┌───────────────┘
                                                       ▼
                                           ┌──────────────────────┐
                                           │   Pipeline Segment   │
                                           │ Node C: Layers 48-79 │
                                           └──────────┬───────────┘
                                                      │
                                                LayerResult
                                                      │
                                                      ▼
                                                   Client
```

### Pipeline Assembly Algorithm

1. Fetch model manifest → determine layer ranges
2. Query shard_registry for hosting nodes
3. Fetch node load/latency from peer_registry
4. Sort candidates by (latency ASC, load ASC, trust DESC)
5. Greedy assignment: widest contiguous layer range per node
6. Identify standby nodes per segment
7. Send PipelineAssignment → all nodes ACK → begin forwarding

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
| issues | {issue_hash_hex} | Issue |
| proposals | {proposal_hash_hex} | Proposal |
| proposal_votes | {proposal_hash}/{voter_hex} | ProposalVote |
| releases | {version_string} | ReleaseCandidate |
| gov_params | "params" | GovernanceParams |

## HTTP API Routes

### OpenAI-Compatible
- `POST /v1/chat/completions` — Chat completions (streaming + non-streaming)
- `POST /v1/completions` — Text completions
- `GET  /v1/models` — List available models
- `GET  /v1/status` — SwarmLLM node status

### Admin API
- `GET/PUT /api/admin/config` — Configuration
- `GET     /api/admin/stats` — Node statistics
- `GET     /api/admin/models` — Model list
- `POST    /api/admin/models/:id/add` — Trigger model acquisition
- `GET     /api/admin/models/:id/status` — Model acquisition progress
- `GET     /api/admin/peers` — Connected peers
- `GET     /api/admin/credits` — Credit info
- `GET     /api/admin/ws` — WebSocket for live updates

### Governance API (Phase 7)
- `GET/POST /api/admin/issues` — Issue CRUD
- `GET/POST /api/admin/proposals` — Proposal CRUD
- `GET      /api/admin/releases` — Release info
- `GET      /api/admin/governance/*` — Governance state

### Static
- `/admin` — Dashboard SPA
- `/chat` — Chat interface
- `/setup` — First-run wizard
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
