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
│  │  config      — boot-time snapshot (startup only)    │  │
│  │  live_config — current config, read via cfg()       │  │
│  │                                                     │  │
│  │  ┌─ EventBus (state.events) ──────────────────────┐ │  │
│  │  │  broadcast::Sender<ActivityEvent> (cap 256)    │ │  │
│  │  │  broadcast::Sender<DashboardSignal> (cap 32)   │ │  │
│  │  │  activity_history, update_state                │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ CreditPool (state.credits) ──────────────────┐ │  │
│  │  │  credit_balance, pool_state, pool_registry     │ │  │
│  │  │  trust_manager, escrow_manager, anti_gaming    │ │  │
│  │  │  foreign_pool_catalog (R134)                   │ │  │
│  │  │  allow_cross_pool_inference (R137)             │ │  │
│  │  │  share_model_catalog (R137)                    │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ ModelMgmt (state.models) ────────────────────┐ │  │
│  │  │  acquisition_progress, hf_sources              │ │  │
│  │  │  auto_manage_*, model_trust, locked_shards     │ │  │
│  │  │  removed_by_user (deleted-shard tombstones)    │ │  │
│  │  │  shards_needing_repair (corrupt → refetch)     │ │  │
│  │  │  prune_history, download_cancel_flags          │ │  │
│  │  │  wishlist (R111), hf_trending_cache (R112)     │ │  │
│  │  │  foreign_wishlist (R130)                       │ │  │
│  │  │  quant_recommendations (R133)                  │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │  ┌─ MetricsProviders (state.metrics) ────────────┐ │  │
│  │  │  node_stats, inference_requests_total          │ │  │
│  │  │  channel_metrics, inference_latency_samples    │ │  │
│  │  │  providers_config, provider_model_map          │ │  │
│  │  │  swarm_capacity (R110)                         │ │  │
│  │  │  hedge_tracker (R136 L2)                       │ │  │
│  │  │  prefetch_orchestrator (R136 L3)               │ │  │
│  │  │  ngram_hits / ngram_misses (R137 L1 telemetry) │ │  │
│  │  │  inference_latency_samples (R137: (Instant,f64))│ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                     │  │
│  │  Root: peer_registry, model_registry, executor,     │  │
│  │    identity, db, active_pipelines, config,          │  │
│  │    pending_layer_results, pending_stream_result_    │  │
│  │    routes, pending_prefix_kv_fetches,               │  │
│  │    pending_activation_chunks (R139 Tier 4K),        │  │
│  │    retained_replies (fast-path replies kept for      │  │
│  │      ResendTokens, gotcha #438, 2026-09-02),         │  │
│  │    standalone_tokenizers (R136 L1/L3 follow-on),    │  │
│  │    listen_multiaddrs (R140 pool invite v2),         │  │
│  │    publicly_reachable + hole_punch_successes /      │  │
│  │      hole_punch_failures (v0.3.21 NAT diagnostics), │  │
│  │    recent_failures (v0.3.22 diagnostics ring, 20)   │  │
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
│    Storing and dialling are separate questions (R148):      │
│      filter_storable — drops only always-junk (loopback,    │
│        circuits through our own id). Keeps private addrs    │
│        wherever we are, so a roaming laptop keeps its LAN   │
│        peers.                                               │
│      filter_dialable — adds context: a node with no         │
│        private address of its own cannot reach anyone       │
│        else's 192.168/10/172.16/CGNAT, so those are         │
│        dropped on read. Empty listen_multiaddrs means       │
│        "not bound yet", NOT "public" — unknown context      │
│        keeps everything.                                    │
│                                                             │
│  Layer 3: Encrypted Network + Pool Invite Codes              │
│    Network-only (single multiaddr):                         │
│      Format: swarm://<base64url(key‖nonce‖encrypted_addr)> │
│      API: GET /api/admin/network-code                       │
│            POST /api/admin/join-network                     │
│    Pool join (R140 — bundles discovery + join token):       │
│      Format: swarmpool://<base64url(key‖nonce‖encrypted_json)>│
│      Payload: { version, pool_id, pool_name, multiaddrs[],  │
│                 code, expires_at_unix }                     │
│      API: POST /api/pool/generate-code                      │
│            POST /api/pool/join (accepts both v2 + legacy 8) │
│      Module: src/pool/invite.rs                             │
│    Encryption: ChaCha20-Poly1305 (key embedded in code —    │
│      anti-IP-harvesting only; the code itself is the auth   │
│      token).                                                │
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
8. Periodic: discovery tick every 5min (Kademlia bootstrap + re-dial cached
   peers not currently connected), peer cache save every 5min. A bootstrap
   RETRY loop polls every 5s but only acts while zero peers are connected,
   on a backoff schedule.
9. On shutdown: save peer cache
10. mDNS race recovery: if simultaneous-dial kills both connections, the
    pending_redial queue schedules a re-dial with hash-based jitter (2-5s).
    `max_connections_per_peer` is 3, not 1 — DCUtR needs a relayed and a direct
    connection open at once to hole-punch (see `network/behaviour.rs`).
11. Dialling is ONE attempt per peer carrying all its known addresses, through
    `dial_checked` (foreign-peer gate, self-dial guard) with
    `DisconnectedAndNotDialing`. Never one dial per address — see
    `.claude/rules/architecture.md` § "One dial per PEER".
```

### Peer Registry Scaling (S3)

peer_registry capped at 200 entries. On overflow, evicts highest-latency non-LAN non-pipeline peer.
Memory bounded at O(1) instead of O(N). LAN peers and pipeline-active peers are never evicted.

### DHT-Based Shard Holder Resolution (S5)

Two-tier shard holder discovery for 50K+ node scaling:

- **Tier 1 — Bounded in-memory cache**: `ModelRegistry.shard_holders` uses `HashMap<NodeId, Instant>` (not `HashSet<NodeId>`) with max 50 holders per shard. LRU eviction when at capacity. Local node never evicted. Populated by GossipSub `ShardAnnounce` + DHT query results. Sync `shard_holders()` API unchanged — scheduler hot path stays fast. **The DHT merge can only ADD a holder** (`merge_dht_providers` loops `record_shard_holder`), and a kad provider record outlives the fact it asserts by up to 24h — so a holder's explicit retraction is recorded in `ModelRegistry.retracted_claims` and outranks the record for 26h; only the holder's own announcement clears it. Without that, a withdrawn claim was reinstated within seconds and every request was scheduled onto a node that no longer had the weights (gotcha #364, fixed v0.3.113).

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
│   ├── JSON control messages — SwarmMessage, ShardRequest/ShardResponse (type-tag 0x00=WIRE_TAG_JSON)
│   ├── Binary tensor payloads — LayerForward, LayerResult (type-tag 0x01=WIRE_TAG_TENSOR, or 0x02=WIRE_TAG_TENSOR_COMPRESSED for zstd, flag-gated; inner ChaCha20-Poly1305 encryption marked by TENSOR_TAG_ENCRYPTED=0x10)
│   ├── Binary shard data — ShardResponse payload (type-tag 0x03=WIRE_TAG_SHARD, 32MB chunks as raw bytes, bypasses 4MB JSON limit)
│   ├── Cross-node prefix-KV snapshots — (type-tag 0x04=WIRE_TAG_PREFIX_KV, Item 8 Phase 2 fetched path)
│   ├── Resend of a fast-path reply's tokens (2026-09-02, gotcha #438) — `SwarmMessage::ResendTokens{request_id, from, to}` from the requester when its reassembler sees a hole (after `hole_wait` = 4×RTT clamped 1-5 s, up to 4 asks), answered from the serving node's `retained_replies` (bounded 64 replies / 8192 tokens / 32 asks / 120 s) ONLY to the requester's peer; gated on `features::RESEND_TOKENS`. A `StreamingToken` the requester's dispatcher dropped is answered `SwarmResponse::Dropped` (to a peer advertising the bit) instead of `Ack`, and the sender re-sends it once
│   └── ACK-timeout fast-fail: streaming-tracked sends (`SendDirectMessage` with `delivery_request_id = Some(uuid)`) are mapped to a Uuid via `pending_rr_observability`. The 10s `RR_ACK_TIMEOUT_SECS` sweep closes `streaming_token_txs[uuid]` if no Response/OutboundFailure event fires (libp2p rr can silently drop sends under load); caller sees Err in ~10–20s instead of 120s
│
├── TCP transport (Noise + Yamux, nodelay=true, port+10)
├── QUIC transport (port, fallback for NAT traversal)
├── mDNS (optional, LAN peer discovery — conditional dial, not added to Kademlia)
├── connection_limits (max 1/peer, 500 total)
├── allow_block_list (blocked_peers — nodes Identify showed do not speak SwarmLLM;
│   refuses both directions at the swarm level. Declining to REGISTER them left
│   something inside libp2p re-dialling them a few times a minute each, and every
│   one of our own dial sites already refusing: gotcha #404, v0.3.131)
├── Identify (protocol identification + peer_to_node reverse map;
│   `peer_speaks_swarmllm` gates registration BEFORE the Kademlia insert)
├── AutoNAT v2 client+server (per-address reachability test → ExternalAddrConfirmed / relay activation; replaced v1 in R143 to fix false-"Public")
├── DCUtR (hole punching)
├── UPnP (IGD gateway port-mapping → auto-confirms public external address; default on, off on WSL2)
└── relay::client (circuit relay)
```

**Internet reachability (R143).** A node's advertised address set
(`state.listen_multiaddrs`, consumed by v2 invite codes) is the UNION of
`swarm.listeners()` (bound sockets — private LAN on a NAT'd node) and
`swarm.external_addresses()` (public addresses confirmed by UPnP, AutoNAT,
relay circuits, or the manual `network.external_address` override). This closes
the gap where a NAT'd node minted invite codes carrying only its LAN address.
UPnP (default on) auto-opens the gateway port for the common home-router case;
`network.external_addresses` lets a port-forwarded box / VPS / dyndns anchor
declare its reachable address(es) explicitly (list form covers TCP + QUIC).

**NAT detection + relay (R143).** AutoNAT **v2** (client + server, replacing v1)
tests each candidate address for real reachability; a confirmed address emits
`ExternalAddrConfirmed`, an `AddressNotReachable` result triggers relay
activation via `NetworkManager::try_activate_relay` (reserve a `/p2p-circuit` on
a `bootstrap_peers` relay). A belt-and-suspenders fallback in the run loop
reserves a relay if the node still has **no** internet-reachable address
`RELAY_FALLBACK_DELAY_SECS` after startup — so reachability for a CGNAT node
doesn't depend on AutoNAT producing a conclusive answer. The relay path
(reservation → circuit dial → DCUtR upgrade) is wired but still needs live
multi-NAT validation. See `docs/NETWORKING.md` for the operator guide.

**Application-level relay (post-R150, `docs/NETWORKING_PLAN.md`).** The libp2p
circuit relay above establishes a *connection* between two NAT'd peers, but a
`/p2p-circuit` cannot reliably round-trip a request_response substream under
load — so two NAT'd nodes can be `is_connected == true` yet unable to run
inference. A second, application-owned relay closes that gap the way Tailscale's
DERP does: a mutually-reachable third node forwards **already-sealed** payloads
without ever being able to read them.

- **Two message classes.** `SwarmMessage::RelayedEnvelope` carries a sealed
  *control* message (JSON, ≤256 KB); `SwarmRequest::RelayedTensor` carries a
  sealed *activation forward or its result* (binary, ≤32 MB, wire tag `0x06`).
  Both are opaque to the relay — it matches `relay_to` against its connected
  peers and re-sends; it holds no key.
- **Ephemeral-seal, not session-seal.** Session-sealed tensors are decrypted by
  the *transport sender's* key, so a relay-forwarded session tensor would fail to
  open at the target. Relayed payloads are instead sealed to the target's
  **static** X25519 key (derived from its Ed25519 identity via
  `ed25519_pubkey_to_x25519`) using a fresh ephemeral keypair per message
  (`crypto/relay_seal.rs`). AAD binds `origin ‖ relay_to ‖ request_id`, so a
  relay cannot redirect, replay across requests, or swap a payload between
  transfers without Poly1305 rejection.
- **Separate-request return path.** A relayed forward and its result are two
  independent relayed requests, never an RR response substream (which the relay
  cannot proxy). The coordinator's `pending_layer_results` oneshot resolves when
  the relayed-back `LayerResult` arrives; the relay-unwrap path stamps
  `sender_peer_bytes = origin` so the result routes home rather than being
  dropped as unattributed.
- **Feature-gated + prefer-direct.** `NodeCapability` advertises
  `protocol_version: u16` and a `features: u64` bitset
  (`features::{RELAY, TENSOR_RELAY, PIPELINE_CHAIN, PIPELINE_CHAIN_V2,
  FORWARD_ACK, RESEND_TOKENS}`); a node only attempts a relayed send when
  the *recipient* advertises the matching bit, so the protocol evolves additively
  with no flag-day. The relay is chosen only when there is no usable direct
  connection (`has_direct_connection` false — the circuit-only case); a real
  direct/QUIC path always wins. Learned relay routes live in
  `state.relay_routes` (`daemon/state/relay.rs`, 5-min TTL, swept on the
  HealthMonitor tick); selection + forwarding live in `network/manager/relay.rs`.

## Inference Pipeline

### Split Inference Engine

The split inference engine (`src/inference/split/`) enables true distributed inference
using candle for direct tensor computation with quantized GGUF weights. Each node loads
only the transformer layers it owns, forwarding hidden-state activations between nodes.

The module is split into focused subfiles: `model.rs` (SplitModel struct + accessors),
`loader.rs` (GGUF/shard load), `executor.rs` (forward pass + tensor-parallel),
`kv_cache.rs` (per-request KV-cache store; `LayerKv` holds each layer's f32 BHSD
cache plus an optional f16 BSHD mirror for the CUDA flash kernel — GQA only, worth
1.41x on long-context decode, see `.claude/rules/architecture.md`),
`entry.rs` (model entry + LRU eviction),
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

A full byte-level BPE tokenizer is built from GGUF metadata at model load time:
- Vocabulary from `tokenizer.ggml.tokens`
- Merge rules from `tokenizer.ggml.merges`
- Pre-tokenization patterns selected by `tokenizer.ggml.pre` — see below
- GPT-2 byte encoding/decoding for proper UTF-8 handling

**Pre-tokenization is where a byte-level BPE tokenizer is easiest to get
silently wrong**, so `pre_tokenizer_patterns` in `inference/tokenizer.rs`
mirrors llama.cpp's `regex_exprs` table (`src/llama-vocab.cpp`) and its
`tokenizer.ggml.pre` string → enum mapping, including every alias:

- **Llama-3** (`llama-bpe`, `llama3`, `llama-v3`, `falcon3`, `pixtral`, `dbrx`,
  `smaug-bpe`, `glm4`, …) — the pattern most GGUFs in circulation want.
- **Qwen2** (`qwen2`, `deepseek-r1-qwen`, `stablelm2`, …) — as Llama-3 but
  digits split one at a time (`\p{N}`, not `\p{N}{1,3}`).
- **Qwen3.5**, **GPT-4o/Llama-4**, **GPT-2**, and the sequential-list types
  (`default`, `falcon`, `starcoder`, `deepseek-coder`) whose patterns are
  applied **in order**, each pass re-splitting the previous pass's fragments.

Two properties are load-bearing and pinned by tests:

1. **An unrecognised name warns and falls back to the GPT-2 splitter**, never a
   whitespace split. `llama-bpe` was absent from the table for a long time and
   hit a whitespace fallback, which strands every leading space as its own token
   instead of attaching it to the following word. That is what a byte-level BPE
   model is trained on, so the effect was ~2x the tokens AND input in a shape
   the model had never seen — with no error anywhere (gotcha #247).
2. **Text a pattern does not match is kept, not dropped.** Several patterns
   cover only part of their input by design (the GPT-2 one does not match
   interior whitespace runs), so discarding the gaps deletes characters from the
   prompt outright.

Correctness is judged against a reference `tokenizers` BPE built from the SAME
vocab and merges — self-consistency (encode→decode round-trips) cannot detect
this class and passed throughout.

### Chat Template Evaluator

`inference/chat_template/` renders the Jinja template a GGUF carries in
`tokenizer.chat_template`, producing the exact text handed to the model.
`parser.rs` tokenizes, `eval.rs` evaluates, `fallbacks.rs` supplies a
family-appropriate format for a model whose template we cannot run.

It implements the subset real templates use, not Jinja. What it does NOT
implement must FAIL — `apply_chat_template` returns `None` and the caller falls
back by model name — rather than render approximately, because a prompt that is
nearly right is simply a wrong prompt with no error attached:

- **`{% set x = messages[1:] %}` binds a slice, and the offset is honoured.**
  Templates slice precisely to drop a message they have already placed by hand;
  ignoring the offset renders that message TWICE. Every Llama-3 system prompt
  was duplicated for exactly this reason (gotcha #248). A FILTER we do not
  implement (`| reverse`) is applied as identity, which is a harmless superset —
  the distinction between ignoring a refinement and ignoring a removal is the
  whole point.
- **`messages[0]['content']` indexes one message** and must not be mistaken for
  a binding to the list, or the expression is aliased instead of evaluated.
- **Comments obey trim markers.** `{#- … #}` drops its surrounding whitespace;
  skipping only the body leaves a blank line in the model's input.
- **`strftime_now` is provided.** Llama-3.x templates guard on it and fall back
  to a date hardcoded when the model shipped, so reporting it undefined told
  every Llama-3 model it was 26 July 2024.
- Output is capped (`MAX_TEMPLATE_OUTPUT`) and recursion bounded
  (`MAX_TEMPLATE_DEPTH`): the template is peer-supplied GGUF metadata.

The integration guard is
`the_official_llama3_template_renders_exactly_as_jinja2_does`, which renders the
real shipped template against the exact text jinja2 produces for it. Expected
strings are taken FROM jinja2 rather than derived from this evaluator.

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

**LayerForward optional trailers** — the wire envelope appends
optional trailer blocks after the activation bytes. Each trailer is a
single tag byte + fixed payload. Decoders scan in tag order and
ignore unknown tags (forward-compatible).

| Tag | Field | Layout | Purpose |
|-----|-------|--------|---------|
| 0x01 | layer_range + model_id | `start(4)+end(4)+len(2)+model_id` | Required — receiver loads correct segment weights |
| 0x02 | tp_meta | `rank(1)+size(1)+layer(4)+phase(1)+pre_embedded(1)` | Tensor-parallel AllReduce routing |
| 0x03 | speculative | `flags(1)+n_drafts(2)+drafts(n×4)` | Draft tokens + `spec_logits_requested` flag |
| 0x04 | kv_truncate | `target_len(4)` | Spec-decode KV-cache fixup after partial acceptance |
| 0x05 | chunk_meta (R139) | `chunk_idx(4)+total_chunks(4)` | Tier 4K daemon-side STREAM-chunked transport |

All trailers are bound into the encryption AAD via
`build_layer_forward_aad` (single source of truth in
`network/protocol/encrypted.rs`). An attacker who flips a trailer
byte on the wire fails Poly1305 on the receiver's decrypt.

**Tier 4K daemon-side chunked send (R139)** — gated by
`inference.streaming_chunked_send` (default `false`). When on AND
the activation exceeds `streaming_min_activation_bytes` (default
64 KiB), the coordinator splits the activation at byte-offset
boundaries into K = `ceil(size / streaming_chunk_size_bytes)` chunks
(default chunk size 256 KiB — matches age STREAM construction +
TokenWeave MLSys 2026 K=2-4 sweet spot). Each chunk carries the
same `request_id` and a distinct `(chunk_idx, total_chunks)` in the
0x05 trailer. All chunks ride the **same libp2p stream** (QUIC
preserves byte order within a stream → no receiver-side reorder
state machine). Receiver assembles in
`SharedState.pending_activation_chunks: DashMap<Uuid,
ChunkAssemblyState>` via `try_assemble_chunked_forward`, then
dispatches a single reassembled LayerForward to the worker. The
0x05 trailer is AAD-bound so reorder / wrong-total /
cross-transfer-substitution attempts fail Poly1305 before reaching
assembly.

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

### GPU Capability Floor

CUDA builds compile kernels for **compute capability 8.0** (Ampere: RTX 30-series
and newer). This is FlashAttention's own requirement — every `candle-flash-attn`
kernel source is `_sm80` and uses Ampere async-copy — and `flash-attn` is part of
both the `cuda` and `windows-gpu` features.

The floor is stated in four places that must agree, enforced by
`compute_cap_matches_release_workflow` in `tests/repo_consistency.rs`:
`daemon::gpu_support::MIN_COMPUTE_CAP` (runtime) and `CUDA_COMPUTE_CAP` in
`release.yml`, `cache-warm.yml`, `ci.yml` (build). A second test,
`flash_attn_and_the_compute_cap_floor_agree`, ties the floor to the feature in
both directions — the floor is only worth paying for *because* of flash-attn.

**Detection, not failure.** `Device::cuda_if_available` succeeds on a pre-Ampere
card; only module load fails, and it fails per-request with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. Without a probe such a node starts cleanly,
logs "GPU detected", advertises itself to the swarm as a GPU node, and then
fails everything. So:

- `daemon::gpu_support::local_gpu_is_supported()` probes the capability once at
  startup via `nvidia-smi --query-gpu=compute_cap` and caches it. An unreadable
  capability means *unknown*, never *unsupported* — a working card must never be
  demoted because a subprocess misbehaved.
- `ModelProcessPool::effective_gpu_layers` returns 0 for an unsupported card, so
  workers spawn on the CPU. This is the same choke point that handles the
  `gpu_layers` config and OOM CPU-pinning.
- `worker_ipc::permanent_gpu_failure` classifies an architecture-mismatch error
  as permanently GPU-fatal (distinct from OOM, which gets different user-facing
  copy), so a card that slips past the probe falls back on first failure rather
  than crash-looping. The two causes use **different ActivityEvent kinds** —
  `model_cpu_fallback` and `model_cpu_fallback_gpu_too_old` — because the
  frontend translates by kind, and one message must not be shown for the other.

`vendor/candle-flash-attn` carries two patches: static `cudart_static` (upstream
links it dynamically, which would put a hard `libcudart` dependency on a binary
that today needs only the display driver), and the 18 bf16 kernels removed
(`run_attention` casts to f16 before every call).

### Attention Kernel Selection

`inference::layers::run_attention` picks per device AND per shape. The rule is
measured:

| | prefill (`q_len > 1`) | decode (`q_len == 1`) |
|---|---|---|
| CPU | standard | standard (GQA takes the grouped no-copy path inside it) |
| CUDA | flash | flash if GQA, standard if MHA |

**CPU decode is standard for every shape** (since c4cc3b16, 2026-08-16).
`standard_attention` used to materialize the `repeat_kv` expansion every token —
free when `n_head == n_kv_head`, growing with context otherwise — and that cost
is precisely why GQA decode was routed to the fused kernel. It no longer pays
it: for `q_len == 1` it regroups the query heads as extra matmul rows against
the unexpanded cache (`grouped_gqa_decode_attention` — identical arithmetic,
zero copies, pinned byte-equivalent by test). Measured per attention call the
grouped path beats both the old expanded path (3-9x) and the fused kernel
(2-20x, kv 1024-8192); end to end it is **1.41x decode** on llama-3.2-3b
(4.71 → 6.63 tok/s), validated by a 4-hour soak.

**CUDA keeps flash for GQA decode.** Its measurement predates the grouped path
and rested on the same `repeat_kv` premise, so the routing is a re-measure
candidate (`docs/FUTURE_WORK.md`) — but GPUs already route GQA decode to a fused
kernel, and this box cannot resolve small GPU deltas (gotcha #267). The MHA side
is not in question: flash unconditionally would still cost up to **25x per
attention call** on MHA decode — candle-flash-attn has no split-KV kernel, so a
single query row cannot fill the card.

**There is no context-length crossover, and re-introducing one needs a
forward-pass measurement.** A `k_len >= 1024` threshold shipped on 2026-08-07,
taken from timing the attention call in isolation; measured end to end the next
day, flash won at every length (1.13x at kv~272, 1.42x at ~528, 1.61x at ~912).
Third occurrence of the same error — see gotcha #266. The rule lives in
`cuda_decode_prefers_standard`, pinned by unit tests needing no GPU. Full tables
in `docs/FUTURE_WORK.md`; `SWARMLLM_FORCE_STANDARD_ATTN=1` (`docs/DIAGNOSTICS.md`)
is the A/B switch.

### Inference Correctness

**Stop sequence handling**: User-provided stop sequences (`stop` in OpenAI, `stop_sequences` in Anthropic) are enforced in all three inference execution paths:
1. `pipeline/distributed.rs` `execute_distributed` — accumulated text scanned after each token decode
2. `model_worker.rs` `handle_generate` — accumulated text checked after each token in the subprocess decode loop
3. `executor.rs` `generate_stream_llama` — accumulated text checked after each token in the llama.cpp loop

Empty stop sequences are rejected at the API validation layer (must be 1–256 chars).

**EOS token handling**: The distributed pipeline checks for EOS tokens in `result.token_ids` explicitly (not just via `result.finish_reason`), preventing runaway generation if the worker subprocess returns EOS as a token ID without setting the finish reason.

**Top-k sampling**: Uses `select_nth_unstable_by(k - 1, desc_cmp)` to partition the k largest logits into `[..k]`. The k-1 pivot ensures exactly k elements are retained (not k-1).

**RoPE position tracking**: the prompt's length in tokens is a POSITION, not a
statistic — `index_pos` for the first generated token is set from it, so it must
equal the number of KV positions the prefill actually wrote.
`inference::pipeline::prompt::prompt_positions` is the single answer, and it
counts with the model's own tokenizer (`standalone_tokenizer`, lazily built from
`gguf_header.bin`).

`max(chars / 4, 1)` survives ONLY where a node genuinely has no tokenizer for the
model, and warns when it does. It used to be the answer in every case, including
when a tokenizer was available three lines below: on a 24 KB tool-calling prompt
it came out 6053 against a true 5529, so the first generated token was computed
524 positions past the end of the cache and the model answered with end-of-turn
or with filler repeated to `max_tokens` (gotcha #400, fixed v0.3.129). The
minimum of 1 remains, because position 0 is where a prompt starts rather than
where it ends.

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
- Sessions persisted across node restarts via redb, **stamped with the build
  that wrote them and discarded on a mismatch.** `cached_tokens` is a token
  COUNT used directly as the position to resume from, and nothing can check it
  against the saved prompt text without re-running the tokenizer that produced
  it — so a release that changes tokenization or prompt construction makes every
  persisted count wrong, and an auto-update restart inside the session TTL is
  exactly when they get read back. Re-reading a prompt costs a moment; resuming
  at the wrong position corrupts the answer.

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

Three paths, and they differ in where the draft comes from:

**Draft-model (opt-in).** A small model proposes K tokens per step (default 4);
the target verifies all K in one forward; KV is trimmed and reseeded on
rejection. Config: `speculative_decoding`, `speculative_gamma`,
`draft_model_path`. Greedy only by construction — a draft model has a real
distribution, so doing this properly needs `min(1, p/q)` and a residual built
from both sides. Falls back to standard decoding with no draft model.

**Draft-free n-gram, LOCAL (v0.3.116, on by default).** A whole model on one
machine drafts from an n-gram match against the prompt and its own output, and
verifies the batch in one forward — no second model, nothing downloaded. Lives
in `model_worker::ngram_spec_round`, i.e. in the worker's decode loop, which is
the choke point every local surface funnels through (streaming and not, OpenAI
and Anthropic). ~2x CPU / ~3x GPU on replies that copy from context.

**Draft-free n-gram, DISTRIBUTED.** `pipeline::ngram_only_spec` for pipelines
with a remote segment, verifying through `forward_verify_through_segments`.

Three invariants the two draft-free paths share, all learned the hard way:

- **Any temperature.** With a point-mass draft (`q = δ_x`) the rejection rule is
  "accept w.p. `p(x)`, else draw from `p` minus `x` renormalised", and "draw
  `t ~ p`, keep the draft iff `t == x`" has exactly those branches. Both paths
  were once gated on `temperature == 0`, which made them inert — the OpenAI
  default is 0.7 and Anthropic's 1.0.
- **NOT bit-identical** (gotcha #370). A verify forward reassociates, so a
  near-tie can land the other way. Each run is deterministic; that is how you
  tell reassociation from a race.
- **A miss is not free** (#371), and diverting a request out of the batched path
  is not free either (#373). `SpecBackoff` handles the first,
  `spec_payoff_justifies_diverting` the second.

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

### Graphics memory has one owner (v0.3.130)

**`ModelProcessPool` admits, charges and reclaims graphics memory. Nothing else
may take it from a loaded model.**

- Admission: `admit_to_gpu` weighs `estimate_worker_vram_mb` (weights **+ KV** at
  `ADMISSION_KV_CONTEXT`) against `vram_budget_mb`, charging `vram_reserved_mb`.
- On-demand reclaim: `free_vram_for_admission` → `plan_vram_reclaim` — plans the
  whole eviction first and abandons it if it still would not fit, never takes a
  busy worker, and never one used within the swap floor.
- Timed reclaim: `try_idle_vram_unload`, outside the `auto_manage.enabled` gate.
- A refusal places the model on the processor for *that spawn* and takes no
  standing pin; `cpu_pinned_models` means only that the card FAILED for a model
  (`classify_worker_error`).
- A processor-resident model returns to the card on its next request once there
  is room, including room the pool is willing to make
  (`should_return_to_gpu` + `reclaimable_vram_mb`).

**`SharedState.split_models` is a metadata cache, not a memory manager.**
`SplitModelEntry` caches `eos_tokens`, `vocab`, `chat_template`, `bos_token`,
`eos_token_str` and `estimated_vram_mb` read from the GGUF header; the weights
live in the worker. It is bounded by ENTRY COUNT
(`trim_split_model_cache`, `MAX_SPLIT_MODEL_ENTRIES`), LRU, protecting models
with an active pipeline — and cannot unload a worker. Trimming an entry that is
still wanted costs a header re-read, not a killed worker.

A separate **registration** budget decides whether to advertise another segment
as locally servable (`split_model_budget_with` + `split_models_committed_mb` +
`MemoryScope`: the graphics budget, or `inference.max_split_model_memory_mb` on
a node with no card). **It may refuse; it may never take.**

Until v0.3.130 this cache carried its own VRAM budget that evicted entries *and*
killed their workers — a second accountant with a smaller estimate, a weaker
in-flight oracle and no idle floor. See gotcha #402 and
`.claude/rules/architecture.md`.

**How long a model keeps the card** (`VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT`, 5 s
since v0.3.131) protects a model in active use, not against thrash: at the
previous 60 s, two models alternating in conversation took 299 s against 82 s
with no floor, because the model left on the processor was slower at every turn
than the reload it was spared. `examples/swap_patience.sh` is the harness;
`SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS` pins the value for A/B.

### LoRA Adapter Support

LoRA (Low-Rank Adaptation) adapters are supported via `src/model/lora.rs`:
- Per-request adapter loading from safetensors files
- Low-rank weight updates applied at inference time without modifying base model weights
- Multiple adapters can be loaded simultaneously and selected per request
- Adapter files stored alongside model shards in the model directory
- **Verified** with Qwen2.5-Coder-7B + rank-16 LoRA adapter; output distribution changes confirmed

## Credit System

> **DORMANT as of 2026-08-17 — credits gate nothing.** Everything below still
> runs and is still recorded, but `MIN_BALANCE_FOR_INFERENCE = 0` and
> `calculate_tier` returns a constant, so no balance affects who gets served,
> how fast, or what the dashboard shows; the leaderboard neither ranks by
> credits nor publishes them. The reason is that credit has never moved between
> nodes *as payment for work* — each node mints its own figure, so the books do
> not reconcile and acting on them meant rationing the product by a number
> nobody can stand behind. Read **`docs/CREDITS_DESIGN.md`** for the full
> account, the bilateral-settlement design that would fix it, and the exit
> criteria that must hold before any of this is switched back on.

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

**Serving credit earn**: recorded in ONE place, `SharedState::record_peer_serve`, reached only from the two inbound paths that do work for a peer — `dispatch::layer_forward` (one pipeline segment) and `dispatch::remote_generate` (the whole decode, the single-segment fast path). Both count the work and bill for it together; the token count is clamped to `MAX_CREDITABLE_TOKENS` because the requester chooses it on the wire. Credits accumulate in `pending_credit_earn` and are flushed by the ledger with the `inference_serve_earning` tag.

A node does **not** earn for work it does for itself. The router's own completion hook and its local segment inside a pipeline it coordinates are excluded — counting them credited a user for their own chat and told them they had served the swarm. Before v0.3.88 the accounting lived in `track_forward_participation`, which only the multi-segment path called, so a node serving through the fast path (how a machine holding a whole model answers a peer — the common case) recorded nothing and was paid nothing while the requester was still debited.

Note that `release_escrow` reconciles the **requester's** balance only; it records `to_node` but transfers nothing, and `credit::transaction::create_transaction` has no production callers. The accumulator above is currently the only way a serving node is paid.

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

**Setup flow** (R140 — bootstrap-before-decentralization):
1. Main device: `swarmllm pool create --name "My Devices"`
2. Main device: `swarmllm pool invite-code` → generates a `swarmpool://...` blob
   that bundles the 8-char join token with the device's reachable listen
   multiaddrs (LAN + Tailscale CGN + public — everything except loopback /
   link-local).
3. Linked device: `swarmllm pool join "swarmpool://..."`. The joiner decodes
   the blob, dials each multiaddr (Tailscale, LAN, public IP — whatever
   reaches the owner), then broadcasts the existing
   `PoolMessage::JoinRequest { code_hash, ... }` over GossipSub.
4. Owner's code_hash matches → invitation auto-created → member auto-accepted.

The legacy 8-char form (`A3F7K2M9`) is still accepted by `pool/join` for
nodes already on a shared swarm (LAN mDNS, DHT-bootstrapped). v2 codes are
strictly an additive wrapper; the JoinRequest wire protocol is unchanged.

**Invite code security**:
- 8-char uppercase alphanumeric inner token (32^8 ≈ 1.1 trillion combos,
  no 0/O/1/I) — single source of truth for join authorization.
- One-time use, consumed immediately on claim.
- 24h expiry (configurable `invitation_ttl_hours`); v2 blob carries the
  same expiry so decoders can fail fast before dialing.
- Max 5 active codes at once per owner.
- Code hash (BLAKE3) on the wire — plaintext code never transmitted.
- v2 blob: ChaCha20-Poly1305 encrypted with embedded key — protects
  against casual IP harvesting from a code pasted into chat, not a
  cryptographic boundary (the code IS the auth token).
- Join requests signed with Ed25519.
- API input cap: 4096 chars (typical v2 blob is ~300-500 chars).

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
| removed_shards | {shard_id_json} | bool (presence = the user deleted this shard; auto-manage will not re-acquire it until it is explicitly requested again) |
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
- **Budget limits**: ONE storage budget (`auto_manage::storage_budget` — `max_storage_mb` as written, else 25/50/75% of `max_disk_mb` by contribution level, capped at `max_disk_mb` and at held + 80% of free disk) shared by the download pass, prune pressure, the settings bar, the pool page and the diagnostics report (gotcha #448); max_shards_per_cycle (2); skips in-progress acquisitions
- **mmproj support**: Vision encoder (mmproj.gguf) treated as download candidate with 5x priority bonus; full-file HF download (not byte-range); higher pruning floor (min 3 replicas), only pruned under extreme pressure (>0.95)
- **Download priority**: HuggingFace CDN first (fast, doesn't burden peers). If no HF source available but peers hold the shard, falls back to P2P `ShardRequest` to a random holder. P2P is single-source per shard (future: multi-source parallel download)
- **Upload bandwidth cap**: `max_bandwidth_mbps` config enforced on shard serving via proportional delay after chunk reads. Tensor forwards exempt (latency-critical). Default 0 = unlimited
- **Config**: `[auto_manage]` section — `enabled`, `max_storage_mb`, `interval_minutes`, `max_shards`, `prune_enabled`, `min_replicas`, `prune_cooldown_secs`, `max_holder_load_for_prune`, `auto_switch_quants` (R134.6, **default true since R141**), `hf_watcher_enabled`, `wishlist_gossip_publish`
- **R141 — P2P stall timeout** = 180s (was 600s). A silently-dropped libp2p send now fails over to the HF fallback path in ~3 minutes instead of 10. The original ceiling was sized for worst-case slow peers + pessimistic retries; the new value still covers an honest 32 MiB chunk over a slow link (~150 KiB/s sustained). Constant: `P2P_PERMIT_STALL_SECS` in `model/auto_manage/manager.rs`.

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
+ redundancy_ratio (holder_count / effective_target)
+ 1.0 if not loaded in VRAM (cold shard)
+ 0.5 × resource_pressure
+ 1.0 if contribution_auto && holder_count ≥ 2 × target  (R121, severe saturation)
- 0.5 if first/last shard (pipeline completeness)
- 0.3 if rarest shard for the model
- 0.2 if recently acquired (< 30 min)
```

**R121 — contribution_auto scale-back.** When `config.node.contribution_auto`
is true (the default), a shard with `holder_count ≥ 1.5 × target` bypasses
the RELAXED-state +1 nudge in `pressure_adjusted_target` and is eligible
to prune even at zero local pressure. This lets an idle node shed slack
once the swarm has plenty of copies, instead of waiting for VRAM/disk
pressure to build. Manual mode (`contribution_auto = false`) keeps the
pre-R121 behaviour — pressure-driven only. The toggle is hot-reloadable: like every
user-settable value it is read from the live config via
`SharedState::cfg()`, which `PUT /api/admin/config` replaces. Pure helper:
`effective_prune_target(target, pressure, holder_count, contribution_auto,
min_replicas)` in `model/auto_manage/prune.rs`.

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

**Auto-manage gate**: `gather_candidates()` skips models below `DemandVerified` unless `pinned_by_user = true` or this node already hosts at least one shard. This means a node will never auto-download shards for a model nobody has actually used.

**Trust transitions**:
- Gossip-discovered models start as `Discovered` (auto-created on first manifest registration)
- User downloads via HF browser → `Pinned` (persisted immediately to redb)
- 3rd inference request → `DemandVerified` (persisted on promotion)
- 3+ unique holder nodes → `NetworkPopular` (checked periodically by AutoShardManager)
- HfWatcher trending feed promotes `Discovered` → `DemandVerified` when the matching HF repo crosses the per-publisher download floor + 24h age gate (anti-gaming). R141 tiered the floor:
  - **Trusted curators** (`TRUSTED_HF_PUBLISHERS` allowlist in `huggingface/watcher.rs` — meta-llama, mistralai, Qwen, google, microsoft, deepseek-ai, HuggingFaceH4, stabilityai, tiiuae, 01-ai, NousResearch, allenai, ibm-granite, CohereForAI, bartowski, TheBloke, unsloth, lmstudio-community, MaziyarPanahi, QuantFactory, second-state) promote at **10k downloads** (`MIN_DOWNLOADS_FOR_TRUST_TRUSTED`).
  - **Unknown publishers** keep the original **100k downloads** floor (`MIN_DOWNLOADS_FOR_TRUST`).
  - The 24h age gate is unchanged for both tiers — a fresh repo can still be a download-pump even from a trusted account if it's compromised.
  - Helper `is_trusted_publisher` is the canonical check, re-exported via `crate::model::huggingface`. Used by both the watcher's promotion path AND the wishlist scorer for the `wishlist.why.trusted_publisher` why-tag + score bonus.
- 7 days without requests → decay (`NetworkPopular` → `DemandVerified`, `DemandVerified` → `Discovered`)
- Auto-promoted models that decay back to `Discovered` with zero real swarm requests bump `failed_promotions` (anti-gaming cooldown for re-promotion; `FAILED_PROMOTION_COOLDOWN_BASE_DAYS = 7`, cap 4 strikes — beyond that only a user pin lifts the level)
- Pinned models never decay

**Persistence**: `model_trust` tree in redb, keyed by model_id, values are JSON `ModelTrustInfo`.

**API**: Trust level exposed as `trust_level` field in `GET /api/admin/models` response.

**Scaling**: Trust decisions are local per-node (no consensus needed). Each node independently decides what to download based on its own observed demand. This scales to thousands of nodes without coordination overhead.

### Wishlist (R111 + R141)

The **wishlist** (`src/model/auto_manage/wishlist.rs`) is the user-visible
face of auto-manage. Instead of the daemon downloading models in silence,
the dashboard renders a ranked list with status badges + human-readable
"why" tags so non-technical users understand *why* the swarm cares about
each entry.

**Generation cadence**: rebuilt on every auto-manage tick AND every WS
`stats_update` build. Cheap (single registry pass), so the dashboard
sees fresh data the moment it connects. Stored as
`ArcSwap<Wishlist>` on `state.models.wishlist`, refreshed via
`crate::model::auto_manage::refresh_wishlist(state)`.

**Status taxonomy** (`WishlistStatus` enum):

| Status | Meaning | CTA |
|--------|---------|-----|
| `hosting` | This node hosts ≥1 shard | "You're helping host this" (no action) |
| `serveable` | Network has every shard at least once — can route today | "Help host" (one-click contribute) |
| `aspirational` | Partial swarm coverage; gathering in progress | "Help unlock this" |
| `candidate` (**R141**) | HF trending model the swarm hasn't adopted yet | "Set this up" → routes to HF browse pre-filtered |
| `unreachable` | Larger than the whole swarm pool VRAM | Informational only |
| `blocked` | Trust gate / private mode / explicit user-ignore | "Awaiting verification" |

**R141 — Candidate entries**. `compute_wishlist` merges `HfTrending` entries
the swarm hasn't adopted (cap `MAX_CANDIDATE_ENTRIES = 24`) as `Candidate`
rows. Distinguishes from `Blocked` (trust-gated existing entries) and
`Aspirational` (real partial coverage). Candidate entries populate two
extra fields the frontend uses for routing:

- `hf_repo_id: Option<String>` — the HF repo identifier (`publisher/repo`).
  Frontend deep-links the user into the HF browse view pre-filtered to
  this repo so they pick the quant variant themselves (no auto-pick,
  preserving the user-controlled adoption flow).
- `task_tags: Vec<String>` — capability tokens (`chat`, `code`, `vision`,
  `multilingual`, `reasoning`) sourced from `HfWatcher::infer_task_tags`.
  Drives optional filter chips in the swarm-tab Search subtab.

Synthetic key format: `model_id = "hf-candidate:" + repo_id` so Candidate
rows dedup cleanly against the registry-walking loop without colliding
with real `ModelId`s. Both fields use `#[serde(skip_serializing_if = ...)]`
so the wire payload stays small for the 99% case (non-Candidate).

**Score blend** (0..100, advisory only — frontend renders a heat bar):
- Coverage component (0..40): fully serveable hits the cap; partial scales linearly.
- Popularity component (0..25): log-scaled unique holder count.
- Demand component (0..25): regional `region_demand` gossip.
- VRAM-fit component (0..10): pool VRAM / model VRAM ratio, clamped.
- HF trending boost (0..15): log10(downloads) when matching the cached
  HF trending feed.
- Foreign-wishlist boost (0..10, R130): cross-pool demand breadth × depth.
- First-host nudge (+10): no holders yet → strong "be the first" signal.
- R141 Candidate bonus (+10): models from `is_trusted_publisher` get a
  flat bonus so curator releases rank above unknown publishers with
  similar download counts.

**Why-tags** (each is an i18n key under `wishlist.why.*`):
`be_first_host`, `exceeds_swarm_capacity`, `fits_your_memory`,
`needs_review`, `no_regional_replica`, `other_nodes_want_this`,
`parts_missing|missing=N` (with params), `popular_on_hf`,
`popular_on_swarm`, `swarm_already_serves`, `you_already_host`,
`your_region_needs_this`, plus R141 additions `trusted_publisher` and
`candidate_one_click`.

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

## Error classification (all surfaces)

`crate::error::classify_error(&SwarmError) -> (StatusCode, message, error_type)`
is the single definition of what a failure IS to a caller. Every surface derives
from it, so the same failure cannot be named two things:

| surface | how it consumes the classification |
|---|---|
| HTTP envelope | `ApiError::into_response` — status + `error.type`/`code` |
| OpenAI SSE | `StreamEvent::Error { message, error_type }` |
| Anthropic SSE | `AnthropicSseEvent::Error`, translated by `anthropic_error_type` |
| Responses API | `responses::stream::classify_error_code` (both the streaming and background paths) |
| MCP | `mcp::types::tool_error_code` → JSON-RPC `-32602` / `-32000` / `-32603` |

Two **refinements** are deliberate and documented at their definition: Responses
names a provider failure `upstream_error`, and MCP maps 503 to
`RESOURCE_UNAVAILABLE`. A refinement names something more precisely than the
canonical answer; a *divergence* is the same meaning under a different word and
is a bug.

`crate::error::reclassify_flattened_error(&str) -> Option<SwarmError>` recovers a
class across a boundary that carries no types — `SwarmError` survives neither the
worker IPC hop nor the network hop, both of which deliver a `String` that would
otherwise be re-wrapped as `Inference` → HTTP 500. Used at both boundaries. It
matches on `SwarmError`'s own `#[error(...)]` Display prefixes, which are part of
the type; it must never be extended to match user-facing prose (gotcha #295).

## API Authentication

- Bearer token middleware in `src/api/middleware.rs` (constant-time comparison)
- Auto-generated 32-byte hex API key on first run, persisted in redb
- **Protected paths**: `/v1/*` (inference), `/api/admin/config` (PUT), `/api/admin/shutdown`,
  `/api/admin/hf/*` (downloads), `/api/admin/api-key`, `/api/admin/provider-models`
- **Exempt paths**: `/`, `/health`, `/admin`, `/chat`, `/setup`, `/static/*`,
  read-only admin dashboard endpoints (GET `/api/admin/stats`, `/api/admin/models`, etc.)
- **Loopback-only actions**: `POST /api/admin/update/check`, `/api/admin/update/apply`
  and `/api/admin/shutdown` additionally require the request to originate on the
  node's own machine. A valid API key is *not* sufficient — the first two write
  to disk and replace the running binary, so a remote key-holder must not be able
  to drive them. Refusal is `SwarmError::LocalOnly` → **403 `permission_error`**,
  never 401: the caller authenticated fine and is being refused on origin. Filing
  it under `Unauthorized` meant the hint told a remote admin to go and fetch an
  API key they had already used successfully, which could not work (gotcha #309).
  Note this is the one place `addr.ip().is_loopback()` is the right predicate —
  it is an origin restriction, not the "may we hand over the key?" question that
  `api::dashboard_trust::classify` answers (gotcha #195).
- Request body size limit: 32MB (configurable via `DefaultBodyLimit`, raised from 2MB for VLM image payloads)
- Content-Security-Policy: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`
- **Dashboard key bootstrap** (`src/api/dashboard_trust.rs`) — the dashboard has
  no Bearer token on page load and fetches one from `GET /api/admin/api-key`.
  That handout requires BOTH a trusted source network AND a valid single-use
  `X-Dashboard-Nonce`. `classify()` is the single decision point:
  `Loopback` always; `Overlay` for `100.64.0.0/10` / `fd7a:115c:a1e0::/48` when
  `api.dashboard_trust_overlay` (default true) AND this node itself holds such an
  address (the IPv4 range is shared CGNAT space, so the peer's address alone
  proves nothing); `LocalNetwork` for RFC1918/ULA only when the
  `state.dashboard_trust_lan` runtime atomic is set (default false, toggled live
  via `PUT /api/admin/config`). Untrusted origins are not blocked — the page
  prompts for the key and stores it per-origin.
  **Do NOT re-derive this with `addr.ip().is_loopback()`**: that predicate means
  "the last TCP hop began in this daemon's netns", which a same-host reverse
  proxy satisfies on a remote client's behalf and a container publish / Tailscale
  subnet router never satisfies even from the host's own localhost (subnet
  routers SNAT by default). Threat model: on a trusted network, reachability of
  the API port is equivalent to admin access — the nonce only stops a
  non-browser local process that cannot read the served HTML, since `/admin` is
  unauthenticated and a nonce can simply be scraped.
- WebSocket Origin validation (prevents cross-site WebSocket hijacking) —
  `websocket.rs::ws_origin_allowed` compares `Origin` against the request's own
  `Host`, plus the loopback forms for proxies that rewrite `Host`. It is
  deliberately NOT a fixed localhost allowlist: that refused the legitimate
  same-origin `Origin` of any dashboard served at a LAN or Tailscale address, so
  remote dashboards silently lost every live update and fell back to polling.
  The upgrade's real gate is the single-use ticket from the Bearer-authed
  `POST /api/admin/ws-ticket`.
- Input validation: model name 256 chars, tools max 128, stop sequences max 16
- HuggingFace inputs validated (repo_id format, filename .gguf extension, no path traversal)
- HTTP timeout: 5 minutes (tower-http TimeoutLayer, Slowloris protection).
  Routes that can run a model (`/v1/chat/completions`, `/v1/responses`,
  `/v1/messages`, `/mcp`) are merged OUTSIDE this layer — generation has no
  bounded duration, and prefill alone can exceed five minutes on a long prompt.
  They are bounded instead by the prompt-scaled first-token budget,
  client-disconnect cancellation and TCP keepalive. The merge sits before the
  auth layer, so they still require a key (pinned by
  `generation_routes_still_require_a_key`).
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
├── tied_output_weight.bin   # weight-tied models only — see below
└── ...
```

`ShardReader` in `split/shard_reader.rs` constructs a virtual GGUF from header + shard files,
allowing candle to parse the full tensor index while only loading assigned layers.

### Weight-tied models and the output head

A weight-tied model (the Llama-3.2 family, Gemma-2, and most small models) reuses
`token_embd.weight` as its LM head and ships no separate `output.weight`. That
tensor physically lives in **shard 0**, but the node serving the **last** pipeline
segment is the one that needs it — and in a real swarm that node frequently does
not hold shard 0.

`tied_output_weight.bin` carries the raw tensor bytes so the head can be loaded
without shard 0. It is produced by `extract_tied_output_weight` (local GGUF) and
`download_tied_output_weight` (HF byte-range), and consumed by
`resolve_tied_output` → `ShardReader::new`, which maps it over the tensor's gguf
byte range. Reads resolve through the ordinary tensor map, so
`ct.tensor(&mut reader, "token_embd.weight", …)` works unchanged. When the node
*does* hold shard 0 the sidecar is ignored and the shard is used.

`GgufTensorMeta::tied_output_location()` is the single definition of "weight-tied",
shared by both writers and the reader so they cannot disagree about which tensor
the sidecar holds.

## HTTP API Routes

### OpenAI-Compatible (Bearer auth required)
- `POST   /v1/chat/completions` — Chat completions (streaming + non-streaming, tool_calls). `logprobs` is refused for a model running locally — every local path pins `token_logprobs: vec![]`, so it is only ever returned by a cloud provider (see Deferred Items).
- `POST   /v1/responses` — OpenAI Responses API (gpt-5 / o-series default)
- `GET    /v1/responses/{id}` — Retrieve a stored response (30-day TTL); pass `?stream=true&starting_after={seq}` to resume a background SSE stream
- `DELETE /v1/responses/{id}` — Delete a stored response
- `POST   /v1/responses/{id}/cancel` — Cancel a background response
- `GET    /v1/responses/{id}/input_items` — Paginated list of the original input items (synthetic ids `item_N`)
- `POST   /v1/messages` — Anthropic Messages API (full Claude Code compatibility — tools, tool_choice, thinking, cache_control, metadata)
- `POST   /v1/embeddings` — Text embeddings
- `GET    /v1/models` — List available models. Each entry carries `max_model_len`
  (the effective context this node will serve for that model — prompt plus reply,
  after the shipped 8192 default and any `inference.max_seq_len_override`), which
  is the field vLLM added for the same purpose, and the identical figure as
  `context_length`, the name OpenClaw's self-hosted model discovery reads (with
  it absent OpenClaw assumes 128,000 — 2026-09-02). `ModelInfo::new` sets both
  from one value. Omitted when the model's declared context is unreadable, since
  a wrong figure is worse than an absent one.
- `GET    /v1/providers` — List configured cloud providers and their available models
- `GET    /v1/status` — SwarmLLM node status. `workers` (2026-09-03): every resident model-worker subprocess — `model`, `pid`, `device` (`graphics card`/`processor`), `cpu_reason`, `in_flight` (requests it is computing now, from the pool's own response map), `idle_secs`, `age_secs`, `dead`, `gpu_estimate_mb`. The answer to "what is my machine computing right now", and how a worker still busy for a client that has gone is found without `ps` (gotcha #445); retire one with `POST /api/admin/models/{id}/unload`. `swarmllm status` renders it.

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
- **Failure reporting (2026-08-12):** a failure is an `event: error` SSE frame —
  `{"type":"error","error":{"type":…,"message":…}}` — and never assistant text.
  The frame is TERMINAL: `build_anthropic_sse_response` ends the keepalive ticker
  on it as it does on `message_stop`, and no epilogue follows. `stop_reason`
  carries only values the API defines (`end_turn`, `max_tokens`, `stop_sequence`,
  `tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded`). The
  error `type` is translated from the canonical classification into Anthropic's
  own set by `anthropic_error_type`, defaulting to `api_error` rather than
  inventing a name. Before this the surface had no error frame at all: the
  router path reported every failure as an empty `end_turn` and the split path
  wrote `[inference failed: …]` into the message body (gotchas #300-#302).

### Tool calling on a LOCAL model

A cloud provider handles tools natively. A local GGUF only emits text, so tool
support is three pieces, shared by the OpenAI and Anthropic layers:

1. **`tool_parse::format_tool_prompt`** describes the tools in a system message.
   This is the ONLY way a local model learns they exist — which makes it the
   only place `tool_choice` can be enforced. `tool_choice_forbids_tools` (the
   OpenAI string `"none"` and the Anthropic `{"type":"none"}`) suppresses the
   injection entirely; every other value, `"required"` included, still describes
   them, because a local model cannot be compelled and refusing would be worse.
2. **`tool_parse::parse_tool_calls`** recovers calls from the model's text,
   trying the generic shape we prompt for, then Hermes/Qwen, Mistral and
   Llama-3 native formats. It does NOT repair truncated JSON: a generation cut
   off at `max_tokens` is reported as text rather than as a call carrying
   invented arguments.
3. **Call ids are assigned here, never taken from the model.** The id is how a
   client matches a result back to a request, so it must be unique across a
   conversation; models do not do that (llama-3.2-3b emits `call_1`, `call_2`,
   `call_3` for every tool-using reply it gives).

### MCP Server (Protocol v2025-11-05)
- `POST /mcp` — JSON-RPC 2.0 MCP endpoint for AI agent frameworks (Claude Code, VS Code Copilot, Cursor, etc.)
- Tools: `chat`, `models`, `compare`, `research`, `batch_prompts`, `delegate`, `node_info`
- `delegate` picks a model for you by tier. `fast` ranks **already loaded** above
  local above smallest — loading a cold model costs tens of seconds, which
  dominates every other difference (`fast_tier_rank`).
- Resources: `swarmllm://status` (node status)
- All tools include [tool annotations](https://modelcontextprotocol.io/specification/2025-11-25) (`readOnlyHint`, `destructiveHint`, etc.)
- **`compare`:** sends the same prompt to up to 10 models concurrently, returns side-by-side results
- **`research`:** fan-out a question to multiple models (auto-selects if models omitted), returns all responses with token usage
- **`batch_prompts`:** execute up to 20 independent {id, model, prompt} tasks in parallel
- **`delegate`:** offload a task to the best model for a given tier (fast/cheap/smart) — auto-selects model
- **`node_info`:** detailed node status (loaded model, peers, credits, registry models, cloud providers). The loaded-model object carries `servable_now`: `loaded_model_info` is cached when the node starts and says what was loaded, NOT whether a request can be routed now — those differ, and an operator reported the reported-loaded model failing every request while two others answered (2026-08-10). `servable_now` is whether any reachable node currently holds that model's first shard; `false` means requests will fail whatever "loaded" says.

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

**Long-lived sessions** (`src/api/claude_session.rs`) — a persistent `claude` subprocess driven with `--input-format stream-json`, for interactive use where re-sending the whole conversation per turn would be wasteful. All five routes exist ONLY under `--features claude-subscription`; without it the router serves none of them.

- `GET    /api/claude-code/sessions` — List live sessions
- `POST   /api/claude-code/session` — Start a session (spawns the subprocess)
- `GET    /api/claude-code/session/{id}` — Session state
- `DELETE /api/claude-code/session/{id}` — Close it and reap the subprocess
- `POST   /api/claude-code/session/{id}/message` — Send a turn
- `POST   /api/claude-code/session/{id}/permission` — Answer a tool-permission prompt the CLI raised

### Admin API (CORS-protected, no Bearer auth)
- `GET/PUT /api/admin/config` — Configuration read/update
- `GET     /api/admin/stats` — Node statistics + hardware info. The memory
  figures are `process_rss_mb` (this daemon PLUS every model worker — the
  weights and KV cache live in the worker subprocesses, so a daemon-only
  reading is blind to nearly all of it), broken out as `daemon_rss_mb` /
  `worker_rss_mb` / `worker_count`. The process refresh names an explicit pid
  list and must never become a whole-machine scan (gotcha #417).
- `GET     /api/admin/swarm/capacity` — R110: collective capacity snapshot (online_nodes, total_vram_mb, serveable/aspirational/hosted_locally model lists, redundancy)
- `GET     /api/admin/swarm/capacity-plan` — R113: what-if scenarios + headline_target with concrete `contributors_needed` count
- `GET     /api/admin/storage/breakdown` — R110: stacked-bar data (total_mb, used_mb, auto_target_mb, free_mb)
- `GET     /api/admin/wishlist` — R111: ranked list of models the swarm wants (status, score, why_tags, swarm_replicas, target_replicas)
- `GET     /api/admin/hf/trending` — R112: cached HuggingFace trending-GGUF snapshot from HfWatcher
- `GET     /api/admin/quant-recommendations` — R133: per-family quant-choice recommendations with rationale tags
- `GET     /api/admin/foreign-pool-catalog` — R134: discovery-only cache of models advertised by other pools (gated on `pool.share_model_catalog`)
- `GET     /api/admin/responses` — List stored `/v1/responses` records for the dashboard (filter by `?status=…&limit=…`)
- `GET     /api/admin/models` — Model list with shard status, VRAM estimates, acquisition state. `encrypted_pipeline` is the EFFECTIVE state (`flag && has_first && has_last`) so the UI's "privacy is on" indicator never claims privacy that is not happening; `encrypted_pipeline_blocked` reports the case that masking hides — the setting is on and this node cannot satisfy it, so every request for that model fails. Both are needed: without the second, the failing state appears on no surface a user looks at (gotcha #286).
- `POST    /api/admin/models/{id}/add` — Trigger model acquisition
- `GET     /api/admin/models/{id}/status` — Model acquisition progress
- `GET     /api/admin/peers` — Connected peers with latency/trust
- `GET     /api/admin/diagnostics` — Plain-text support dump, **address-redacted unless `?full=1`** (it is written to be pasted into a bug report; `swarmllm diagnostics` is the CLI wrapper). Sections: **this machine** (CPU, GPU,
  measured memory bandwidth, and the advertised 7B tok/s every peer's
  scheduler ranks this node on — the answer to "why does nobody route work
  to me?", which appeared on no pasteable surface before), reachable
  addresses, peers, **recent inference failures** (last 20: model, elapsed, and
  *which peer served each* — the field that separates "this node is broken" from
  "one peer is broken", plus a repeated-peer note), **NAT traversal**
  (`publicly reachable`, `donating relay capacity`, hole-punch success/failure
  counts with a reading of what zero means in context), peer cache (stored vs
  dialable), models, **recent completed requests** (last 50: route, per-phase
  timings, per-segment attribution), **per-peer serving performance** (RTT,
  ms/layer, EWMA latency, samples, region — slowest first), and **served for
  others** (segments/layers computed for peers, compute time, bytes out). This is
  the single most useful thing to include in a bug report.
- `GET     /api/admin/performance` — The JSON sibling of `diagnostics`, for the
  dashboard's Performance panel: `recent` (up to 50 traces), `peers`, `served`,
  and `hourly` (one bucket per hour for a week, persisted). Pulled on demand, not
  pushed on the WS stats tick — see § Request Tracing, cardinality rule.
- `GET     /api/admin/credits` — Credit balance and tier info
- `GET     /api/admin/shard-storage` — Per-model storage breakdown, disk/VRAM usage
- `GET     /api/admin/api-key` — Retrieve API key (Bearer auth required; the dashboard's key-less bootstrap needs a trusted source network per `api::dashboard_trust` AND a valid single-use `X-Dashboard-Nonce`)
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
- `POST /api/pool/generate-code` — Generate a v2 `swarmpool://` invite code (owner only). R140: bundles the 8-char join token with the device's reachable listen multiaddrs so joiners can bootstrap without a shared DHT.
- `POST /api/pool/join` — Join a pool via invite code. Accepts either a v2 `swarmpool://...` blob (dials embedded multiaddrs then broadcasts the join request) or the legacy 8-char form (broadcast-only — assumes joiner is already on the swarm).
- `POST /api/pool/device-name` — Set this device's display name
- `PUT  /api/pool/credit-split` — Set credit split percentage (owner only)
- `PUT  /api/pool/contribution` — Set per-member contribution level (owner only, `{"node_id": "...", "level": 75}` where level is 0–100)

### Discovery
- `GET    /api/admin/network-code` — Get shareable invite code, multiaddr, and network phase
- `POST   /api/admin/join-network` — Join network via invite code or multiaddr

### Utility
- `POST   /api/admin/shutdown` — Gracefully shut down the node (localhost only)
- `POST   /api/admin/config/reload` — Re-read config.toml into the live config; response splits `applied` from `restart_required`
- `POST   /api/admin/downloads/{model_id}/cancel` — Cancel in-progress HF download
- `DELETE /api/admin/models/{model_id}` — Remove model (shards + manifest + state)
- `POST   /api/admin/models/{id}/unload` — Unload model from VRAM (keep shards on disk)
- `DELETE /api/admin/models/{id}/shards/{index}` — Delete a single shard
- `GET/PUT /api/admin/models/{id}/auto-manage` — Per-model auto-manage policy (incl. prune toggle)
- `PUT    /api/admin/models/{id}/shards/{index}/lock` — Lock/unlock a shard (prevent auto-pruning)
- `POST   /api/admin/models/{id}/shards/{index}/download` — Download a single shard from P2P network
- `POST   /api/admin/models/{id}/shards/{index}/load` — Load a shard into memory (expands shard window, restarts worker)
- `POST   /api/admin/models/{id}/shards/{index}/unload` — Unload a shard from memory (narrows shard window, restarts worker, frees RAM/VRAM)
- `GET    /api/admin/models/{id}/pipeline-plan` — Pipeline assembly plan: ordered segments + holder candidates per shard window
- `POST   /api/admin/models/{id}/enable-privacy` — Fetch the first and last shards of a model so the encrypted "boomerang" pipeline can engage; privacy then turns itself on
- `POST   /api/admin/api-key/rotate` — Issue a new API key and invalidate the old one. Takes effect on the next daemon start (the running server holds the current key in immutable state). Until this existed the key could not be rotated at all: it lives in the database and `data/api_key` is only a published copy, so deleting that file republished the same value — leaving no remedy for a leaked key short of destroying the node's identity.
- `GET    /api/admin/credits/transactions` — Bounded log of recent balance movements (delta, kind, reason, resulting balance), oldest first. Added because only the running totals were kept, so a node reporting large spend/refund figures against zero requests could not be investigated by anyone. Note `lifetime_refunded` is partly synthetic — `backfill_historical_refunds` attributes unexplained gaps to refunds, so the books close by construction rather than as evidence the movements were understood.
- `GET    /api/admin/reference-models` — The pinned smoke/standard/stress models from `docs/REFERENCE_MODELS.md`, for cross-swarm comparison (opt-in via `swarmllm get-model`)
- `GET/PUT /api/admin/schedule` — Resource schedule management
- `GET    /api/admin/prune-history` — Recent auto-prune events
- `GET/POST /api/admin/adapters` — List/register LoRA adapters
- `DELETE /api/admin/adapters/{id}` — Delete a LoRA adapter
- `GET/PUT /api/admin/providers` — View/configure cloud provider API keys
- `GET    /api/admin/provider-models` — List models available from cloud providers
- `GET    /api/admin/provider-health` — Probe cloud provider availability
- `POST   /api/admin/provider-model-status` — Check specific model availability on provider
- `GET    /api/admin/version` — Version info (binary version, git hash, build features), plus `restart_required`: set when a NEWER version is installed on disk than the one running, i.e. the restart into it did not take effect. `null` when they agree or nothing has been installed. Exists because an in-place update `exec`s and therefore keeps the process id AND the kernel's start time, so `ps` cannot distinguish "updated" from "never restarted" (gotcha #277/#287) — an operator concluded twice from exactly that evidence that their node had missed eight releases, and nothing could contradict it. The node knows what it installed and what it runs; this is the comparison. A deliberate rollback (running newer than the record) is not flagged.
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
  - `frontend/js/components/swarm-tab.js` — Swarm tab: wishlist + Capacity Plan view (R111)
  - `frontend/js/init.js` — event binding, initialization, public API export (`window.SwarmLLM`)
- **HTML templates**: 10 `<template id="tmpl-*">` elements for repeating UI structures (session items, chat messages, toasts, compare cards, compare model chips, leaderboard rows, download queue items, prune rows, storage model rows, pool member rows). Components clone templates via `template.content.cloneNode(true)` instead of innerHTML string building. (The peer table is built by `dashboard.js renderPeers` via string concatenation, not a `<template>`.)
- Cross-component calls: `App.componentName.method()`. Shared state: `App.state.*`. Utilities: `App.utils.*`.

### Frontend Features
- **i18n**: 1331 translation keys (1333 entries per locale incl. `_lang` + `_dir`) across 21 languages (en, es, fr, de, pt, it, nl, ru, zh, ja, ko, ar, tr, pl, sv, th, hi, vi, id, uk, cs). Auto-detects browser language. `I18n.t()` + `data-i18n` DOM attributes. Interpolation via `{variable}` placeholders. Fallback chain: current language → English → raw key. "Continue in English" UX for non-English users who prefer English.
- **Theme**: Light / Dark / System toggle. `[data-theme="light"]` CSS overrides. Persisted in localStorage.
- **Neural network background**: Animated canvas particle network behind dashboard tiles (`frontend/js/neural-bg.js`). ~60 nodes with connecting edges, gentle drift, mouse repulsion/glow. State-reactive coloring: blue (idle) → cyan (active inference) → red-orange (unhealthy/disconnected). Peer count boosts vibrancy, active requests trigger node firing pulses. Pauses when tab hidden; reduced opacity in light theme.

## CPU Inference Performance

Three defects found on 2026-08-06, all invisible without measurement, all in how
work reaches the CPU rather than in what work is done.

### The quantized matmul is tiled over the batch dimension

`vendor/candle/candle-core/src/quantized/k_quants.rs::matmul` is patched. Upstream
makes the batch row the OUTER loop, so the full weight matrix is re-streamed for
every row and batching amortizes nothing — measured `ms/row` was flat from `m=1`
to `m=128`. The patch makes the weight column the outer loop for `m > 1`, so each
column is read once and applied to every row while it is still in cache, and
parallelizes the activation-quantize loop that then becomes the serial fraction.
`m == 1` keeps the original path, so decode is untouched by construction.

Measured on a 3072x3072 Q4_K shape: `3.00 -> 1.06 ms` at `m=4`,
`101.4 -> 11.4 ms` at `m=128`. Verified **bit-identical** to the original
ordering by `examples/qmatmul_bench.rs`, which reimplements the upstream loop and
asserts exact equality — each output element is one `vec_dot` over the same
operands either way, so there is no reduction reordering.

This is why `vendor/candle` exists as a patched copy; the pre-existing reason was
a `cudarc` linking hardcode (see `Cargo.toml` `[patch.crates-io]`).

### The right attention kernel is OPPOSITE for prefill and decode

`run_attention` (`src/inference/layers/mod.rs`) picks between a fused/flash CPU
kernel and `standard_attention`. Both choices were wrong, in opposite directions:

| phase | was | now | why |
|---|---|---|---|
| prefill (`seq_len > 1`) | fused | **standard** | fused parallelizes over KV tiles of 16 inside a per-query-row loop with a scratch allocation per tile; standard batches into two matmuls per head |
| GQA decode (`seq_len == 1`) | standard below a 2048 crossover | **fused always** | standard materializes the KV cache expanded to `n_head` every token every layer, so cost grows with the conversation |

Multi-head (non-GQA) decode keeps standard, where the expansion is a no-op.
SWIFT/spec sessions still force standard so draft and verify share numerics.

Prefill has many query rows and wants batched matmuls; decode has one row against
a long cache and wants the kernel that never materializes the expansion. **There
is no single "faster kernel" — ask per phase.**

### Attention's tail is one fused pass, and the mask is additive f32

`inference::attn_softmax::scaled_masked_softmax` does the scale, the optional
Gemma-2 logit soft-cap, the causal mask and the softmax in ONE pass over each
score row. Expressed as separate candle ops, each materialised its own
`[batch, heads, q_len, kv_len]` temporary — 11 MB at llama-3.2-3b prefill
shapes — so the tail moved ~90 MB per layer per chunk to do ~3 MB of
arithmetic: 34.6 ms against a matmul-plus-softmax floor of 11.4.

The mask is **additive f32** everywhere (`0.0` visible, `-inf` masked), produced
only by `SplitModel::causal_mask`. It used to be a `u8` predicate for the
standard path plus a float copy the CPU flash arm rebuilt per call. Masks handed
onward must be contiguous — a `narrow()` view costs 2.1x in `broadcast_add` and
the fused kernel declines it outright.

Measured: attention 22.4% of a prompt chunk -> 9.5%, prompt processing
1.19x end to end, decode unchanged. Prompt processing is now **84.5% quantized
matmul**.

### Prefill and decode run in different CPU thread pools

`inference::cpu_pools::in_phase_pool`, bound at `SplitModel::forward_inner_impl`
and `forward_batch` so every entry point inherits it. Prompt processing scales
to 1.83x past decode's optimum; decode is bandwidth-bound and gets 2.0x worse at
the same setting. One pool for both made `contribution` perverse — donating more
of the machine slowed replies down. Only decode is capped, only downward, and
not at all at the default contribution. The width is calibrated from a worker's
first real tokens, **per processor depth** — a forward that runs no layers on
the processor is never timed, and a hybrid split's forwards do not share a
verdict with the one-layer card-only segments the same worker may serve for a
peer (gotcha #432: that sharing settled a worker on one thread for its whole
life).

### Stage profiler

`SWARMLLM_PROFILE=1` (`src/inference/prof.rs`) prints a non-overlapping per-stage
breakdown of each forward pass, including what the stages do NOT account for. It
is what found the attention defects: attention was 2.3% of the arithmetic and 45%
of prompt-processing time. See `docs/DIAGNOSTICS.md`.

### Where the remaining headroom is

Decode is **bandwidth-bound at ~69% of the memory roofline** (72% of its time is
the quantized matmul moving the weights), so faster arithmetic will not help much.
Threads pull the phases apart — decode peaks at 4 and is 2.0x worse at 14, while
prompt processing keeps improving to 14. That split is now handled by
`inference::cpu_pools` rather than being a tradeoff the operator has to pick.

**Prompt processing is 84.5% quantized matmul** after the attention work, so the
next lever is fewer bytes per token or more tokens per weight read, NOT another
elementwise fusion. Full numbers, plus two measured dead ends (self-speculative
decoding is 3.3x slower; raising the global thread count hurts DECODE), in
`docs/FUTURE_WORK.md`. **Re-profile before optimising any stage** — three rounds
have now begun with a stage that turned out to be a minority of the total.

## Request Tracing & Performance Observability

Every inference request carries one `RequestTrace` (`src/inference/trace.rs`),
and that record is the **sole** input to every observability surface. Four
response paths each assembling their own timing struct is the recurring
"one invariant, N paths" defect in `.claude/rules/architecture.md`, and
observability is its worst home: the drift is invisible because nothing fails —
the numbers are just quietly wrong.

**Lifecycle.** Created at admission in `router/mod.rs::handle_submit` so
`queue_ms` measures real user-visible wait; `mark_dequeued` at dispatch;
`mark_assembled` in `distributed_exec::execute_request` with the route and
segment layout recorded together so they cannot disagree; `mark_first_token`
from the token channel; `mark_finished` + `publish_request_trace` at the single
completion arm.

**Route** (`trace::Route`) is classified from the pipeline assignment by
`classify_route`, never inferred from segment count — a one-segment *remote*
pipeline and a one-segment local one are different routes, and that distinction
is exactly what a user asking "why was that slow" needs. `Relayed` outranks
`Distributed` when any hop goes through an application-level relay.

**Time to first token** is stamped by the token *channel*, not by the emit
sites. Tokens leave from seven places (`local_exec`, `process_pool`, `dsd`,
`speculative`, `ngram_only_spec`, `pipeline/mod`), so `StreamingTokenTx` is a
newtype around `mpsc::Sender` that mirrors `send`/`try_send`/`clone`/
`is_closed`/`closed` — call sites read unchanged, and a **new** emit site
inherits the stamp with no author action. Only events carrying non-empty text
count, so a zero-token response reports no TTFT.

**Cost.** Phase boundaries only, plus one relaxed atomic load per token.
`.claude/rules` forbids hot-path overhead in `pipeline.rs`,
`split/executor.rs::forward` and `forward_through_segments`; a `Vec<SegmentTrace>`
allocated once per *request* is inside budget, per-token work is not.

**Per-segment attribution.** `state.active_traces` maps request id → in-flight
trace and has **exactly** the same lifetime as `active_pipelines` — inserted and
removed at the same sites — so it inherits that mechanism's already-correct
cleanup including the panic path via `ActivePipelineGuard::drop`. Deep pipeline
code calls `state.record_segment_timing(...)` unconditionally; it is a no-op when
no trace is registered.

### Surfaces

| Surface | Contents |
|---|---|
| `DIAG: request complete` log line | The whole route and timing set on one greppable line |
| Response headers | `x-swarm-route`, `x-swarm-segments`, `x-swarm-peers`, `x-swarm-nodes`, `x-swarm-regions` + W3C `Server-Timing`, attached by `api::attach_route_headers` |
| `GET /api/admin/diagnostics` | Plain text for a shell: recent requests, per-peer performance, served-for-others, failures. Addresses redacted unless `?full=1` |
| `GET /api/admin/performance` | Same as JSON, plus the hourly trend |
| `/metrics` | OTel-named TTFT/TPOT histograms + `requests_by_route{route,outcome}` + serving-side counters |
| Dashboard | Chat route line; Models → Performance panel |

**Headers flush before the body**, so on a streaming response only pre-body
facts (route, nodes, queue, schedule) can be sent. TTFT and decode are omitted
rather than reported as zero — a plausible-looking zero is worse than an absent
header — and a test asserts it.

### Cardinality rule

Prometheus carries `(route, outcome)` and nothing else: both are closed sets, so
that metric is 20 series regardless of swarm size. Per-peer, per-model and
per-shard dimensions are **unbounded** (50 peers × 10 models × 10 shards = 5 000
series from a single node, growing with the swarm) and live only in
`/api/admin/performance`, which is pulled on demand and retains nothing beyond
the bounded rings. Getting this wrong takes down the scrape long before anyone
benefits from the extra detail.

### Peer performance join

`SharedState::peer_performance_rows` joins the three places peer speed was
already known and none of which was readable from outside the scheduler: the
health-ping round trip (`PeerInfo.latency_ms`), the per-layer EMA the Parallax
router uses (`peer_segment_latency_ms_per_layer`), and `hedge_tracker`'s
per-(model, segment, holder) EWMA with variance and sample counts — collected
since R136 with zero consumers until this. Sorted slowest first; only peers that
have actually served something appear.

### "Tokens per second per node" is not directly measurable

In a pipeline the segments are **serialised**: every token traverses node A's
layers, then node B's. There is no independent per-node token rate —
`tokens / A_time` would show both nodes producing the full stream and the figures
would not compose. What is real and exported:

- **per-segment share of inter-token latency** (`SegmentTrace.elapsed_ms`) — these
  sum toward the total, so they identify the bottleneck hop
- **ms per layer per token** — comparable across peers serving differently-sized
  segments
- **derived node capacity** = `1000 / (ms_per_layer × layers_served)`, useful for
  scheduling and leaderboards but labelled derived, not measured

For a non-pipelined route the request's tok/s *is* that node's tok/s and is
reported plainly.

### Retention

Not a time-series database — `monitoring/` ships Prometheus + Grafana for that.
`PerfHistory` (`daemon/state/perf_history.rs`) keeps one bucket per hour capped
at a week (168), holding **sums** rather than averages so buckets can be merged
without averaging averages, persisted to redb only when the hour rolls over. Only
aggregates are persisted; per-request and per-peer rows stay in the in-memory
rings and are intentionally lost on restart.

## Activity Event System

A lightweight cross-subsystem event bus for real-time dashboard observability.

**Backend** (`ActivityEvent` defined in `src/daemon/state/activity.rs`, re-exported from `state/mod.rs`):
- `ActivityEvent` struct with fields: `category` (`&'static str`), `kind` (`&'static str`, e.g. `"shard_pruned"`), `message` (English), plus optional `model_id`, `model_name`, `node_id`, `detail_num`, `detail_str`, `toast_level`, `toast_duration_ms`, `shard_index`, `freed_bytes`, `holder_count_before`, `holder_count_after`, `remaining_local_shards`, `timestamp` (ISO 8601)
- `activity_tx: broadcast::Sender<ActivityEvent>` in `state.events` sub-struct (capacity 256, oldest events dropped on overflow)
- All 12 subsystems emit events via the `state.emit_activity(ActivityEvent::new(...))` builder — fire-and-forget (send errors ignored)
- Example event kinds (snake_case strings; see `ACTIVITY_ICONS` in `frontend/js/components/notifications.js` for the canonical list): `shard_download_complete`, `shard_pruned`, `inference_request`, `inference_completed`, `peer_connected`, `peer_disconnected`, `model_loaded`, `model_unloaded`, `pool_device_joined`, `pool_created`, `config_updated`, `daemon_started`, `hf_sources_cap_reached` (R141 — throttled 1st + every 50th, surfaces dropped HfSourceGossip due to `MAX_HF_SOURCES = 1024` cap), and many more

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

- **Split-KV (FlashDecoding) kernels for CUDA decode** — `candle-flash-attn` 0.10.1 ships none, so a single-token decode launches a grid of only `(1 × n_head × batch)` blocks and cannot fill the card. Measured on an RTX 3070: flash is **4x-25x slower than `standard_attention` for MHA decode** at every KV length, which is why `cuda_decode_prefers_standard` routes MHA decode away from it. (Short-context GQA decode was also routed away until 2026-08-08, when a forward-pass measurement showed flash winning at every length.) With split-KV, MHA decode could take the fused path too. Upstream flash-attention has the kernels (`flash_fwd_splitkv_*`); adding them to `vendor/candle-flash-attn` is the highest-value follow-on in this area. Full measurement table in `docs/FUTURE_WORK.md`.

- **Binary signature on auto-update (audit_2026-04-29 C1)** — `src/update.rs` verifies the SHA256 sidecar fetched from the same GitHub release as the binary; a compromised maintainer account/CI token can publish a matching pair. Real fix: generate an offline signing keypair, embed the public key at compile time, publish a detached signature as a third release asset, and verify it before applying the rename. Deferred until a key-custody decision is made — see `memory/signing_options.md` for the three concrete options (raw Ed25519, minisign, or Sigstore/Cosign keyless), recommended approach (minisign), and step-by-step rollout plan. Until landed, defence-in-depth fixes keep the blast radius local: `update/check` + `update/apply` are loopback-only (`2e1c5b1`), `apply_update` re-checks `latest_version > running_version` at apply time (post-`cb2c688`), `info.downloaded` only flips true when the staging path is on the same filesystem as the binary, and auto-update is opt-in via `config.updates.auto_update` (default `Disabled`).

### Won't fix unless a concrete caller appears

- **Per-token logprobs from local inference** — the machinery exists at both ends and is not joined up in the middle. `sampling::sample_token_with_logprobs` can compute them, `SamplingParams` carries `logprobs` / `top_logprobs` from both API layers, and `ChoiceLogProbs` / `TokenLogProb` serialize correctly (pinned by `logprobs_response_serializes`). But every local execution site pins `token_logprobs: vec![]` — see the note on `InferenceOutput::from_gen_result` — so nothing ever reaches the response. Completing it means returning per-token logits across the worker IPC boundary for the split path, and across the wire for the distributed path, on every token. Until then `/v1/chat/completions` REFUSES `logprobs` for a locally-served model rather than answering 200 with the field absent, which is indistinguishable from a request that never asked for it (`reject_unsupported_local_options`). Cloud-routed models are unaffected and still return logprobs.

- **`seed` is accepted and ignored for local models.** It rides in `extras` and is forwarded verbatim to a cloud provider, but nothing seeds the local sampler, so two requests with the same seed give different text (measured 2026-08-06). Unlike `n` and `logit_bias`, this is NOT refused: OpenAI documents `seed` as best-effort and explicitly does not guarantee determinism, so a caller cannot rely on it in the first place. Wiring it through would mean threading the seed into the sampler's RNG per request.


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
