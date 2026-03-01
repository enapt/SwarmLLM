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
| NetworkManager | MessageDispatcher | `network_out_tx` | All inbound SwarmMessage variants |
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

The **MessageDispatcher** is a dedicated task in `daemon.rs` that routes inbound network messages to the appropriate subsystem. Inference messages go to InferenceRouter, CreditGossip updates peer balance distributions, ModelVote is processed by the model governance module, and pool messages go to PoolManager.

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
10. Scan local shards → register in model_registry + shard_registry (with disk existence verification)
11. Create mpsc channels (network, router, rebalance, acquisition, pool)
12. Spawn all tasks (10 tasks: NetworkManager, InferenceRouter, MessageDispatcher,
    HealthMonitor, ShardRebalancer, CreditLedger, AcquisitionManager, ApiServer,
    PoolManager, AutoShardManager)
13. Open browser if ui.open_browser_on_start is true (setup wizard or admin)
14. tokio::select! on Ctrl+C signal or any task exit
15. Signal graceful shutdown via watch channel, save peer cache, flush sled database
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
│  Layer 2: Persistent Peer Cache (sled)                       │
│    Saves up to 200 peer multiaddrs every 5 min + shutdown   │
│    Loads on startup → fastest reconnect path                │
│    File: src/network/peer_cache.rs                          │
│                                                             │
│  Layer 3: Network Invite Codes                               │
│    Format: swarm://<base64url_encoded_multiaddr>            │
│    API: GET /api/admin/network-code                         │
│          POST /api/admin/join-network                       │
│    UI auto-hides once 20+ peers known                       │
│                                                             │
│  Layer 4: Peer Exchange (PEX)                                │
│    On each ConnectionEstablished, exchange up to 20 known   │
│    peer addresses. Uses request_response channel.           │
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
1. Listen on QUIC + TCP (unchanged)
2. Subscribe to GossipSub topics (unchanged)
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
│   └── Binary tensor payloads — LayerForward, LayerResult (type-tag byte: 0x00=JSON, 0x01=tensor)
│
├── mDNS (optional, LAN peer discovery — conditional dial, not added to Kademlia)
├── connection_limits (max 2/peer, 500 total)
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

- Per-request KV-cache isolation via `DashMap<(ModelKey, RequestId), Vec<(Tensor, Tensor)>>`
- Each concurrent request gets its own cache — no corruption under concurrency
- Multi-turn reuse: session_id tracks conversations, prefix matching skips redundant prefill
- KV-cache is cleared when `sequence_num == 0` (start of new request)
- `index_pos` travels through the wire protocol so all nodes apply correct RoPE positioning
- Position tracking: `index_pos = prompt_token_count` after prefill, increments by 1 per decode step
- KvCacheManager tracks sessions and wired to inference router for cache reuse
- Causal masks cached with LRU eviction (max 16 entries) to prevent GPU memory leak
- Abandoned cache entries cleaned up after 10 minutes

### Speculative Decoding

- Draft model (small/fast) proposes K candidate tokens per step (default 4)
- Target model verifies all K in one forward pass (amortized GPU cost)
- Rejection sampling ensures output distribution identical to non-speculative
- KV-cache resynchronization on rejection (trim + reseed)
- Config: `speculative_decoding`, `speculative_gamma`, `draft_model_path`
- Falls back to standard decoding if no draft model available

### Batched Inference

- `BatchForwarder` collects concurrent decode-step requests into GPU batches
- Position-independent ops (norms, MLP) run on stacked `[batch, seq, dim]` tensors
- Attention runs per-request (different KV-caches and positions)
- Output split back via `Tensor::narrow` per request
- Prefill and single-item batches use sequential path

### VRAM-Aware Cache Eviction

- `SplitModelEntry` wraps models with `last_used` timestamp and `estimated_vram_mb`
- Configurable `max_split_model_memory_mb` budget (default unlimited)
- LRU eviction: least-recently-used models evicted when over budget
- Active models (with in-flight pipelines) are never evicted

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
| config | "api_key" | String (32-byte hex Bearer token) |
| identity | "keypair" | Encrypted Ed25519 key |
| credits | "balance" | CreditBalance |
| credit_txns | {uuid} | CreditTransaction |
| peer_trust | {node_id_hex} | TrustScore |
| peer_cache | {multiaddr_string} | () (presence key) |
| shard_meta | {model_id}/{shard_index} | ShardInfo + path |
| model_meta | {model_id} | ModelManifest |
| sessions | {session_id} | KV-cache metadata |
| nicknames | {node_id_hex} | NicknameRecord |
| identity_prefs | "nickname" | Local nickname preference |
| pool_state | "pool" | PoolState |
| pool_forwards | {uuid} | PoolCreditForward |
| trust_scores | {node_id_hex} | f64 trust score |
| escrow | {escrow_id} | EscrowEntry |
| hf_sources | {model_id} | HfSource metadata |

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
- **Config**: `[auto_manage]` section — `enabled`, `max_storage_mb`, `interval_minutes`, `max_shards`

## E2E Encryption

```
┌─────────────────────────────────────────────────────────────┐
│                    Three Encryption Tiers                     │
│                                                             │
│  Tier 1: Pairwise Sessions (unicast)                        │
│    Ed25519 → X25519 → ECDH → ChaCha20-Poly1305            │
│    Forward secrecy: ephemeral X25519 re-keying every 10min  │
│    Session epoch mixed into key derivation (no nonce reuse) │
│    Replay protection: atomic fetch_max on recv nonce        │
│    (rejects nonce ≤ last_seen via lock-free TOCTOU-safe op) │
│    Static DH fallback for initial session before first reke │
│                                                             │
│  Tier 2: Pipeline Sealing (inference prompts)               │
│    Per-request ephemeral key → sealed prompt/response       │
│    Wire tag: TENSOR_TAG_ENCRYPTED = 0x10                    │
│                                                             │
│  Tier 3: Sealed Gossip (broadcasts)                         │
│    Epoch-based group key + Ed25519 origin signature         │
│    Verifies sender authenticity before processing           │
│    1hr rotation cycle                                       │
│                                                             │
│  Modules: src/crypto/{session, pipeline_seal, gossip_seal,  │
│           key_rotation}.rs                                   │
└─────────────────────────────────────────────────────────────┘
```

## Identity & Nicknames

- Ed25519-signed nickname records with timestamp-wins conflict resolution
- Timestamp freshness check: rejects records older than 1 hour or >5min in future
- GossipSub topic `swarm/identity` for network-wide propagation
- Collision handling: `nickname#ab12` suffix from node ID prefix
- Sled trees: `"nicknames"`, `"identity_prefs"`

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
- Persisted in sled `trust_scores` tree, hydrated on startup
- Trust factors into pipeline scheduling and credit tier weighting

## Credit Escrow

- `EscrowManager` in `src/credit/escrow.rs` holds credits for large requests (> threshold)
- Lifecycle: `create_escrow()` → `release_escrow()` (success) or `refund_escrow()` (failure)
- Entries expire after 10 minutes with automatic refund
- Persisted in sled `escrow` tree

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

- Bearer token middleware in `src/api/middleware.rs`
- Auto-generated 32-byte hex API key on first run, persisted in sled
- **Protected paths**: `/v1/*` (inference), `/api/admin/config` (PUT), `/api/admin/shutdown`,
  `/api/admin/hf/*` (downloads), `/api/admin/api-key`
- **Exempt paths**: `/`, `/health`, `/admin`, `/chat`, `/setup`, `/static/*`,
  read-only admin dashboard endpoints (GET `/api/admin/stats`, `/api/admin/models`, etc.)
- Request body size limit: 2MB (configurable via `DefaultBodyLimit`)
- Content-Security-Policy header enforced on all responses

## Shard-Only Mode

Nodes can operate with just shard files + manifest.json + gguf_header.bin (~6MB),
without needing the full multi-GB GGUF file:

```
~/.swarmllm/models/qwen2.5-coder-7b/
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
- `POST /v1/chat/completions` — Chat completions (streaming + non-streaming)
- `POST /v1/completions` — Text completions
- `GET  /v1/models` — List available models
- `GET  /v1/status` — SwarmLLM node status

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

### HuggingFace Integration
- `GET  /api/admin/hf/search?q=...` — Search HuggingFace for GGUF models
- `GET  /api/admin/hf/probe?repo_id=...&filename=...` — Probe remote GGUF (size, shard layout)
- `POST /api/admin/hf/download` — Download full GGUF model
- `POST /api/admin/hf/download-shards` — Download specific shard indices

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
- `GET    /metrics` — Prometheus/OpenMetrics endpoint (no auth)
- `GET    /health/ready` — Readiness probe with subsystem status (no auth)

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
