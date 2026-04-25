# Daemon & Subsystems

The daemon spawns 11 Tokio tasks wired together with `mpsc` channels:

```
                           ┌──────────────┐
                           │  daemon/     │
                           │  (bootstrap) │
                           └──────┬───────┘
                                  │ spawns tokio tasks
  ┌───────┬───────┬───────┬───────┼───────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  ▼       ▼       ▼       ▼       ▼       ▼          ▼          ▼          ▼          ▼          ▼
Network  Infer   Credit  Health   API    Rebal-   Acquisi-   Message    Pool     AutoShrd   Update
Manager  Router  Ledger  Monitor  Server ancer    tion Mgr   Dispatch   Manager  Manager   Checker
```

## Subsystem Responsibilities

| Subsystem | File | Role |
|---|---|---|
| **NetworkManager** | `src/network/manager/` | libp2p swarm: Kademlia DHT + GossipSub + request/response |
| **InferenceRouter** | `src/inference/router/` | Request queuing, pipeline assembly, execution coordination |
| **MessageDispatcher** | `src/daemon/dispatch/mod.rs` | Routes inbound network messages to appropriate subsystems |
| **CreditLedger** | `src/credit/ledger.rs` | Credit balance tracking, transaction signing, gossip |
| **HealthMonitor** | `src/health/monitor.rs` | Periodic health pings, rebalancing triggers |
| **ShardRebalancer** | `src/health/rebalancer.rs` | Shard redistribution on node join/leave |
| **AcquisitionManager** | `src/model/acquisition.rs` | BLAKE3-verified model downloads from peers and HuggingFace |
| **ApiServer** | `src/api/server.rs` | Axum HTTP: OpenAI + Anthropic APIs + MCP server + admin dashboard + WebSocket |
| **PoolManager** | `src/pool/manager/` | Device pool management, credit forwarding |
| **AutoShardManager** | `src/model/auto_manage/` | VRAM-aware shard acquisition + smart pruning (manager, scoring, download, prune, scan, vram) |
| **UpdateChecker** | `src/update.rs` | Periodic GitHub release polling, SHA256-verified binary download, atomic apply |

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
| MessageDispatcher | (spawned task) | VisionEncodeRequest → handler → VisionEncodeResponse |

### Broadcast Channels

| Channel | Type | Subscribers | Purpose |
|---------|------|------------|---------|
| `activity_tx` | `broadcast::Sender<ActivityEvent>` (256) | WebSocket | Unified event bus — all subsystem events (shard ops, downloads, inference, pool, config changes). Events carry `toast_level` for frontend toast control. History replayed to new WS clients. |
| `dashboard_tx` | `broadcast::Sender<DashboardSignal>` (32) | WebSocket | Dashboard refresh signals — `PeersChanged` (peer connect/disconnect), `ModelsChanged` (shard download/load/prune), `UpdateAvailable(UpdateInfo)` (new version). |

> **Note**: Former separate channels (`prune_events_tx`, `models_changed_tx`, `lan_discovery_tx`, `system_notify_tx`, `peer_list_changed_tx`, `update_tx`) were consolidated into these two in the event system unification.

## Startup Sequence

1. Parse CLI args (clap)
2. Initialize tracing subscriber
3. Load/create config (TOML + env + defaults + CLI overrides)
4. Ensure data directory exists
5. Load/generate Ed25519 identity
6. Open redb database
7. Build `Daemon { config, identity, db }`
8. Initialize ModelExecutor (load GGUF if `--model` provided)
9. Build `Arc<SharedState>` (includes ModelRegistry from DB)
10. Scan local shards, register in registries
11. Create mpsc channels
12. Spawn all 11 tasks
13. Open browser if configured
14. `tokio::select!` on Ctrl+C or task exit
15. Graceful shutdown: save peer cache, flush database

## Graceful Shutdown

Shutdown is triggered by `Ctrl+C` (SIGINT/SIGTERM) or any task exiting:
- A `watch` channel signals all subsystems
- Peer cache is saved to redb
- Database is flushed
- Open connections are drained
