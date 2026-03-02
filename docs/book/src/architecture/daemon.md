# Daemon & Subsystems

The daemon spawns 10 Tokio tasks wired together with `mpsc` channels:

```
                           ┌──────────────┐
                           │   daemon.rs  │
                           │  (bootstrap) │
                           └──────┬───────┘
                                  │ spawns tokio tasks
  ┌───────┬───────┬───────┬───────┼───────┬──────────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼       ▼       ▼       ▼          ▼          ▼          ▼          ▼
Network  Infer   Credit  Health   API    Rebal-   Acquisi-   Message    Pool     AutoShrd
Manager  Router  Ledger  Monitor  Server ancer    tion Mgr   Dispatch   Manager  Manager
```

## Subsystem Responsibilities

| Subsystem | File | Role |
|---|---|---|
| **NetworkManager** | `src/network/manager.rs` | libp2p swarm: Kademlia DHT + GossipSub + request/response |
| **InferenceRouter** | `src/inference/router.rs` | Request queuing, pipeline assembly, execution coordination |
| **MessageDispatcher** | `src/daemon.rs` | Routes inbound network messages to appropriate subsystems |
| **CreditLedger** | `src/credit/ledger.rs` | Credit balance tracking, transaction signing, gossip |
| **HealthMonitor** | `src/health/monitor.rs` | Periodic health pings, rebalancing triggers |
| **ShardRebalancer** | `src/health/rebalancer.rs` | Shard redistribution on node join/leave |
| **AcquisitionManager** | `src/model/acquisition.rs` | BLAKE3-verified model downloads from peers and HuggingFace |
| **ApiServer** | `src/api/server.rs` | Axum HTTP: OpenAI API + admin dashboard + WebSocket |
| **PoolManager** | `src/pool/manager.rs` | Device pool management, credit forwarding |
| **AutoShardManager** | `src/model/auto_manage.rs` | VRAM-aware shard acquisition + smart pruning |

## Channel Layout

| From | To | Message Types |
|---|---|---|
| NetworkManager | MessageDispatcher | All inbound SwarmMessage variants |
| MessageDispatcher | InferenceRouter | InferenceRequest, LayerForward, LayerResult |
| InferenceRouter | NetworkManager | Outgoing P2P messages |
| HealthMonitor | ShardRebalancer | RebalanceEvent |
| ApiServer | InferenceRouter | RouterCommand (from HTTP) |
| ApiServer | AcquisitionManager | AcquisitionCommand |
| AutoShardManager | AcquisitionManager | AcquisitionCommand |
| CreditLedger | NetworkManager | CreditGossip, CreditTransaction |

## Startup Sequence

1. Parse CLI args (clap)
2. Initialize tracing subscriber
3. Load/create config (TOML + env + defaults + CLI overrides)
4. Ensure data directory exists
5. Load/generate Ed25519 identity
6. Open redb database (auto-migrates from sled if `migrate-sled` feature enabled)
7. Build `Daemon { config, identity, db }`
8. Initialize ModelExecutor (load GGUF if `--model` provided)
9. Build `Arc<SharedState>` (includes ModelRegistry from DB)
10. Scan local shards, register in registries
11. Create mpsc channels
12. Spawn all 10 tasks
13. Open browser if configured
14. `tokio::select!` on Ctrl+C or task exit
15. Graceful shutdown: save peer cache, flush database

## Graceful Shutdown

Shutdown is triggered by `Ctrl+C` (SIGINT/SIGTERM) or any task exiting:
- A `watch` channel signals all subsystems
- Peer cache is saved to redb
- Database is flushed
- Open connections are drained
