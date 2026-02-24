# SwarmLLM — Developer Build Specification

> **Purpose**: This document is the authoritative build reference for implementing SwarmLLM. It is intended to be read by an AI coding agent (Claude Code) and used as the source of truth for architecture decisions, file structure, dependencies, APIs, data structures, and implementation order. Every section is written to be actionable — not aspirational.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dependencies (Cargo.toml)](#3-dependencies)
4. [Core Data Types](#4-core-data-types)
5. [Daemon Architecture](#5-daemon-architecture)
6. [Networking Layer](#6-networking-layer)
7. [Model Management](#7-model-management)
8. [Inference Engine](#8-inference-engine)
9. [Credit System](#9-credit-system)
10. [Identity and Cryptography](#10-identity-and-cryptography)
11. [HTTP API Server](#11-http-api-server)
12. [Admin Web UI](#12-admin-web-ui)
13. [Configuration](#13-configuration)
14. [Database / Local State](#14-database--local-state)
15. [Error Handling](#15-error-handling)
16. [Testing Strategy](#16-testing-strategy)
17. [Build Phases](#17-build-phases)
18. [CLI Interface](#18-cli-interface)
19. [Logging and Observability](#19-logging-and-observability)
20. [Platform Support](#20-platform-support)
21. [Self-Governed Updates and Development](#21-self-governed-updates-and-development)

---

## 1. Project Overview

**SwarmLLM** is a single Rust binary that functions as a peer-to-peer node in a decentralized LLM inference network. Each node simultaneously:

- Participates in a P2P network (discover peers, host model shards, serve inference)
- Runs an HTTP server (OpenAI-compatible API + admin dashboard)
- Manages local resources (GPU/CPU compute, storage, bandwidth)

**Language**: Rust (2021 edition)
**Async Runtime**: Tokio (multi-threaded)
**Minimum Rust Version**: 1.75+

---

## 2. Repository Structure

```
swarmllm/
├── Cargo.toml
├── Cargo.lock
├── build.rs                    # Embed frontend assets at compile time
├── README.md
├── LICENSE
├── config/
│   └── default.toml            # Default configuration
├── src/
│   ├── main.rs                 # Entry point, CLI parsing, daemon bootstrap
│   ├── lib.rs                  # Re-exports for testing
│   ├── config.rs               # Configuration loading, validation, migration
│   ├── daemon.rs               # Top-level daemon orchestration (spawns all tasks)
│   ├── error.rs                # Unified error types (thiserror)
│   ├── types.rs                # Shared data types, newtypes, constants
│   │
│   ├── network/
│   │   ├── mod.rs
│   │   ├── manager.rs          # NetworkManager: libp2p swarm lifecycle
│   │   ├── behaviour.rs        # Custom NetworkBehaviour (Kademlia + GossipSub + RequestResponse)
│   │   ├── discovery.rs        # Peer discovery, bootstrap, PEX
│   │   ├── protocol.rs         # Protocol message definitions (Cap'n Proto schemas)
│   │   ├── transport.rs        # QUIC transport setup, NAT traversal config
│   │   └── relay.rs            # Circuit relay client/server
│   │
│   ├── model/
│   │   ├── mod.rs
│   │   ├── manifest.rs         # .swarm manifest parsing, validation
│   │   ├── shard.rs            # Shard loading, verification, storage management
│   │   ├── distribution.rs     # BitTorrent-style shard distribution
│   │   ├── registry.rs         # Local model registry (what's available, what's hosted)
│   │   └── quantization.rs     # GGUF format handling, quantization utilities
│   │
│   ├── inference/
│   │   ├── mod.rs
│   │   ├── router.rs           # InferenceRouter: request queuing, pipeline assembly
│   │   ├── pipeline.rs         # Pipeline execution: layer-by-layer forwarding
│   │   ├── scheduler.rs        # Node selection, latency optimization, load balancing
│   │   ├── executor.rs         # Local shard execution (GPU/CPU tensor compute)
│   │   ├── sampling.rs         # Token sampling strategies (temperature, top-p, top-k)
│   │   └── kv_cache.rs         # KV-cache management for multi-turn conversations
│   │
│   ├── credit/
│   │   ├── mod.rs
│   │   ├── ledger.rs           # CreditLedger: local balance tracking
│   │   ├── transaction.rs      # Credit transaction types, signing, verification
│   │   ├── priority.rs         # Priority tier calculation and queue ordering
│   │   └── anti_gaming.rs      # Spot-check verification, rate limiting
│   │
│   ├── identity/
│   │   ├── mod.rs
│   │   ├── keypair.rs          # Ed25519 key generation, storage, export/import
│   │   ├── trust.rs            # Per-peer trust scoring
│   │   └── keystore.rs         # Encrypted keystore (AES-256-GCM + Argon2id)
│   │
│   ├── api/
│   │   ├── mod.rs
│   │   ├── server.rs           # Axum HTTP server setup, routing
│   │   ├── openai.rs           # OpenAI-compatible endpoints (/v1/chat/completions, etc.)
│   │   ├── admin.rs            # Admin API endpoints (stats, settings, models)
│   │   ├── websocket.rs        # WebSocket handler for live dashboard updates
│   │   └── middleware.rs       # Request logging, CORS (localhost only), rate limiting
│   │
│   ├── ui/
│   │   ├── mod.rs
│   │   └── assets.rs           # Embedded frontend assets (include_dir! macro)
│   │
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── db.rs               # sled database wrapper
│   │   ├── shard_store.rs      # On-disk shard file management
│   │   └── migrations.rs       # Database schema migrations
│   │
│   └── health/
│       ├── mod.rs
│       └── monitor.rs          # HealthMonitor: periodic checks, rebalancing triggers
│
│   └── governance/
│       ├── mod.rs
│       ├── proposals.rs        # RFC creation, validation, lifecycle management
│       ├── voting.rs           # Weighted voting, quorum calculation, tallying
│       ├── releases.rs         # Release candidate management, threshold signing
│       ├── issues.rs           # Bug reports, feature requests, prioritization
│       ├── testing.rs          # Distributed test coordination, canary rollouts
│       └── changelog.rs        # Auto-generated changelog from merged proposals
│
├── proto/
│   └── messages.capnp           # Cap'n Proto schema definitions
│
├── frontend/
│   ├── index.html               # Admin dashboard SPA
│   ├── chat.html                # Built-in chat interface
│   ├── setup.html               # First-run setup wizard
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js               # Dashboard logic
│       ├── chat.js              # Chat interface logic
│       └── setup.js             # Setup wizard logic
│
└── tests/
    ├── integration/
    │   ├── network_test.rs      # Multi-node network simulation
    │   ├── inference_test.rs    # End-to-end inference pipeline
    │   ├── credit_test.rs       # Credit system invariants
    │   └── api_test.rs          # HTTP API contract tests
    └── fixtures/
        ├── tiny_model/          # Minimal test model (2-layer, random weights)
        └── test_config.toml
```

---

## 3. Dependencies

```toml
[package]
name = "swarmllm"
version = "0.1.0"
edition = "2021"
default-run = "swarmllm"

[[bin]]
name = "swarmllm"
path = "src/main.rs"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# Networking (P2P)
libp2p = { version = "0.54", features = [
    "tokio",
    "quic",
    "kad",
    "gossipsub",
    "request-response",
    "identify",
    "autonat",
    "dcutr",
    "relay",
    "noise",
    "dns",
    "tcp",
    "yamux",
    "macros",
    "serde",
] }

# HTTP server
axum = { version = "0.7", features = ["ws", "multipart"] }
axum-extra = { version = "0.9", features = ["typed-header"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "fs", "compression-gzip"] }
hyper = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
capnp = "0.19"
capnpc = "0.19"                # build dependency for Cap'n Proto
toml = "0.8"

# Cryptography
ed25519-dalek = { version = "2", features = ["serde", "rand_core"] }
blake3 = "1"
argon2 = "0.5"
aes-gcm = "0.10"
rand = "0.8"

# Database
sled = "0.34"

# GPU / Inference
# NOTE: Phase 1 uses llama-cpp bindings. Direct CUDA comes later.
llama-cpp-2 = "0.1"            # Rust bindings for llama.cpp
safetensors = "0.4"

# Utilities
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
anyhow = "1"
thiserror = "1"
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
include_dir = "0.7"            # Embed frontend at compile time
tokio-stream = "0.1"
futures = "0.3"
dashmap = "5"                  # Concurrent hashmap for shared state
bytes = "1"
hex = "0.4"
base64 = "0.22"
dirs = "5"                     # Platform-specific directories

[build-dependencies]
capnpc = "0.19"

[dev-dependencies]
tempfile = "3"
tokio-test = "0.4"
assert_cmd = "2"
predicates = "3"
wiremock = "0.6"

[profile.release]
opt-level = 3
lto = "thin"
strip = true
```

### Dependency Notes

- **libp2p version**: Pin to 0.54.x. The API changes significantly between minor versions. Check latest stable before starting.
- **llama-cpp-2**: This is the Rust binding for llama.cpp. It handles GGUF loading, quantized inference, GPU offloading. This is the inference backend for Phase 1-2. Direct CUDA/cuBLAS integration comes in Phase 3+.
- **capnp**: Used for network protocol messages only (tensor data, shard transfers). HTTP API uses serde_json.
- **sled**: Embedded key-value store. If it becomes a bottleneck later, replace with redb or sqlite.

---

## 4. Core Data Types

All core types live in `src/types.rs`. Use newtypes for type safety.

```rust
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::fmt;

// ──── Identity ────
/// Wrapper around Ed25519 public key. This IS the node's identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8])) // Short display
    }
}

// ──── Models ────
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String); // e.g., "llama3-70b-q4km"

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: ModelId,
    pub name: String,                    // Human-readable: "Llama 3 70B Q4_K_M"
    pub architecture: ModelArchitecture,
    pub num_layers: u32,
    pub num_params_billions: f32,
    pub quantization: Quantization,
    pub total_size_bytes: u64,
    pub shard_count: u32,
    pub shards: Vec<ShardInfo>,
    pub tokenizer_hash: Blake3Hash,
    pub manifest_hash: Blake3Hash,       // Hash of entire manifest for verification
    pub publisher: NodeId,
    pub publish_date: chrono::DateTime<chrono::Utc>,
    pub license: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Mixtral { num_experts: u32, experts_per_token: u32 },
    Qwen2,
    DeepSeek { num_experts: u32, experts_per_token: u32 },
    Phi,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Quantization {
    Q4KM,
    Q5KM,
    Q6K,
    Q8_0,
    FP16,
}

// ──── Shards ────
pub type Blake3Hash = [u8; 32];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub index: u32,                      // 0-indexed shard number
    pub layer_range: (u32, u32),         // Inclusive start, exclusive end
    pub size_bytes: u64,
    pub hash: Blake3Hash,                // Content hash for verification
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardId {
    pub model_id: ModelId,
    pub index: u32,
}

// ──── Node Capabilities ────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub gpu: Option<GpuInfo>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub disk_available_mb: u64,
    pub bandwidth_mbps: f32,             // Measured, not reported
    pub hosted_shards: Vec<ShardId>,
    pub max_contribution: ContributionLevel,
    pub uptime_seconds: u64,
    pub version: String,                 // Daemon version
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,                    // e.g., "NVIDIA GeForce RTX 3070"
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>, // CUDA compute capability
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,     // Light node
    Moderate,    // Standard node
    Maximum,     // Super node (if hardware supports)
}

// ──── Inference ────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: uuid::Uuid,
    pub model_id: ModelId,
    pub messages: Vec<ChatMessage>,      // For chat completions
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub requester: NodeId,
    pub priority: PriorityTier,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,        // Default 0.7
    pub top_p: f32,              // Default 0.9
    pub top_k: u32,              // Default 40
    pub max_tokens: u32,         // Default 2048
    pub stop: Vec<String>,       // Stop sequences
    pub frequency_penalty: f32,  // Default 0.0
    pub presence_penalty: f32,   // Default 0.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 2048,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

// ──── Credits ────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: NodeId,
    pub balance: i64,                    // Can go negative (Bronze tier)
    pub lifetime_earned: u64,
    pub lifetime_spent: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityTier {
    Bronze = 0,
    Silver = 1,
    Gold = 2,
    Platinum = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditTransaction {
    pub id: uuid::Uuid,
    pub from: NodeId,
    pub to: NodeId,
    pub amount: i64,
    pub reason: TransactionReason,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature_from: Vec<u8>,         // Ed25519 signature by `from`
    pub signature_to: Vec<u8>,           // Ed25519 signature by `to`
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransactionReason {
    InferenceServed { request_id: uuid::Uuid, tokens: u32 },
    ShardHosting { shard_id: ShardId, hours: f32 },
    ShardSeeding { shard_id: ShardId, bytes: u64 },
    RelayService { duration_seconds: u64 },
    InferenceConsumed { request_id: uuid::Uuid, tokens: u32 },
    Penalty { reason: String },
}

// ──── Pipeline ────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineAssignment {
    pub request_id: uuid::Uuid,
    pub segments: Vec<PipelineSegment>,
    pub standbys: Vec<PipelineSegment>,  // Backup nodes per segment
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSegment {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    pub layer_range: (u32, u32),
}

// ──── Network Messages ────
/// Top-level enum for all protocol messages sent over libp2p.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SwarmMessage {
    // Discovery
    ShardAnnounce(ShardAnnounce),
    NodeCapabilityUpdate(NodeCapability),

    // Inference pipeline
    InferenceRequest(InferenceRequest),
    PipelineAssignment(PipelineAssignment),
    LayerForward(LayerForward),
    LayerResult(LayerResult),
    InferenceError(InferenceError),

    // Credits
    CreditTransaction(CreditTransaction),

    // Health
    HealthPing { nonce: u64, timestamp: u64 },
    HealthPong { nonce: u64, timestamp: u64 },

    // Governance
    ModelVote(ModelVote),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardAnnounce {
    pub node_id: NodeId,
    pub shards: Vec<ShardId>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerForward {
    pub request_id: uuid::Uuid,
    pub sequence_num: u32,               // Token position in sequence
    pub activations: Vec<u8>,            // Serialized tensor data
    pub format: TensorFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorFormat {
    FP16,
    FP32,
    INT8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerResult {
    pub request_id: uuid::Uuid,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Error(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceError {
    pub request_id: uuid::Uuid,
    pub error: String,
    pub recoverable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVote {
    pub voter: NodeId,
    pub model_manifest_hash: Blake3Hash,
    pub vote: bool,                      // true = support, false = deprecate
    pub weight: u64,                     // Contribution-weighted
    pub signature: Vec<u8>,
}
```

---

## 5. Daemon Architecture

### Entry Point (`main.rs`)

```rust
// Pseudocode structure — implement fully
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Parse CLI args (clap)
    let cli = Cli::parse();

    // 2. Initialize tracing
    init_tracing(&cli);

    // 3. Load or create config
    let config = Config::load_or_create(&cli.config_path)?;

    // 4. Load or generate identity
    let identity = Identity::load_or_generate(&config.data_dir)?;

    // 5. Open database
    let db = Database::open(&config.data_dir)?;

    // 6. Build and run daemon
    let daemon = Daemon::new(config, identity, db).await?;
    daemon.run().await
}
```

### Daemon Orchestration (`daemon.rs`)

The daemon spawns all subsystems as Tokio tasks and wires them together with `tokio::sync::mpsc` channels.

```rust
pub struct Daemon {
    config: Config,
    identity: Identity,
    db: Database,
}

impl Daemon {
    pub async fn run(self) -> anyhow::Result<()> {
        // Create channels
        let (network_tx, network_rx) = mpsc::channel(1024);
        let (inference_tx, inference_rx) = mpsc::channel(256);
        let (credit_tx, credit_rx) = mpsc::channel(256);

        // Shared state (wrapped in Arc for cross-task access)
        let shared_state = Arc::new(SharedState::new(
            self.config.clone(),
            self.identity.clone(),
            self.db.clone(),
        ));

        // Spawn all tasks
        let network_handle = tokio::spawn(
            NetworkManager::new(shared_state.clone(), network_rx, inference_tx.clone())
                .run()
        );

        let inference_handle = tokio::spawn(
            InferenceRouter::new(shared_state.clone(), inference_rx, network_tx.clone(), credit_tx.clone())
                .run()
        );

        let credit_handle = tokio::spawn(
            CreditLedger::new(shared_state.clone(), credit_rx)
                .run()
        );

        let health_handle = tokio::spawn(
            HealthMonitor::new(shared_state.clone(), network_tx.clone())
                .run()
        );

        let api_handle = tokio::spawn(
            ApiServer::new(shared_state.clone(), inference_tx.clone())
                .run()
        );

        // Wait for shutdown signal
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received");
            }
            result = network_handle => {
                tracing::error!(?result, "NetworkManager exited");
            }
            // ... other handles
        }

        // Graceful shutdown
        shared_state.shutdown().await;
        Ok(())
    }
}
```

### SharedState

```rust
/// Thread-safe shared state accessible by all daemon tasks.
pub struct SharedState {
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: DashMap<ModelId, ModelManifest>,
    pub shard_registry: DashMap<ShardId, Vec<NodeId>>,  // Which nodes have which shards
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: RwLock<CreditBalance>,
    pub node_stats: RwLock<NodeStats>,
    pub shutdown: tokio::sync::watch::Sender<bool>,
}
```

---

## 6. Networking Layer

### NetworkManager (`network/manager.rs`)

The NetworkManager owns the libp2p Swarm and is the sole interface to the P2P network.

**Key implementation details:**

1. **Swarm setup**: Create a `libp2p::Swarm` with a custom `NetworkBehaviour` that combines:
   - `Kademlia` — DHT for peer/shard discovery
   - `GossipSub` — Pub/sub for network-wide announcements (model votes, capacity updates)
   - `request_response::Behaviour` — Direct request/response for inference pipeline messages
   - `Identify` — Protocol identification
   - `AutoNat` — NAT detection
   - `DCUtR` — Direct connection upgrade through relay
   - `relay::client::Behaviour` — Circuit relay client

2. **Bootstrap sequence**:
   ```
   a. Load bootstrap peer addresses from config
   b. Dial bootstrap peers
   c. Run Kademlia bootstrap query
   d. Subscribe to GossipSub topics: "swarm/models", "swarm/governance", "swarm/health"
   e. Announce own shards to DHT
   f. Begin periodic peer discovery (every 60s)
   ```

3. **Message routing**: NetworkManager receives `SwarmMessage` enums from other daemon tasks via channel, serializes them, and sends to appropriate peers. Incoming messages are deserialized and forwarded to the appropriate task's channel.

### GossipSub Topics

| Topic | Purpose | Message Types |
|---|---|---|
| `swarm/models/{model_id}` | Per-model coordination | ShardAnnounce, capacity updates |
| `swarm/governance` | Model voting, network governance | ModelVote |
| `swarm/health` | Aggregate network health | Anonymized trust summaries |

### DHT Keys

| Key Pattern | Value |
|---|---|
| `/swarm/node/{node_id}` | `NodeCapability` (serialized) |
| `/swarm/shard/{model_id}/{shard_index}` | List of `NodeId`s hosting this shard |
| `/swarm/model/{model_id}` | `ModelManifest` (serialized) |

---

## 7. Model Management

### Manifest Format (`.swarm`)

A `.swarm` file is a JSON manifest with accompanying shard data files:

```
llama3-70b-q4km/
├── manifest.json        # ModelManifest serialized as JSON
├── tokenizer.json       # Tokenizer config
├── tokenizer.model      # SentencePiece model (if applicable)
├── shard_000.bin        # Weight data for layers 0-3
├── shard_001.bin        # Weight data for layers 4-7
├── ...
└── shard_019.bin        # Weight data for layers 76-79
```

### ShardManager Responsibilities

1. **Storage**: Shards stored in `{data_dir}/models/{model_id}/shard_{index}.bin`
2. **Loading**: On startup, scan model directory and populate `shard_registry` in SharedState
3. **Verification**: On load, verify each shard's BLAKE3 hash matches manifest
4. **GPU Loading**: When assigned to a pipeline, load relevant shard weights into GPU VRAM (or RAM for CPU inference) via llama.cpp bindings
5. **Eviction**: If disk space is low, evict shards for least-popular models (respecting minimum replication)

### Distribution Protocol

Shard distribution uses a simple request/response pattern over libp2p:

```
1. New node joins, selects a model to support
2. Queries DHT for shard holders
3. Identifies rarest shards (fewest holders)
4. Sends ShardRequest to peers holding rare shards
5. Downloads shard data in chunks (1MB pieces)
6. Verifies BLAKE3 hash on completion
7. Announces new shard holdings to DHT
```

Shard data transfers use the `request_response` protocol with streaming. The requesting node can download from multiple peers simultaneously (like BitTorrent piece selection).

---

## 8. Inference Engine

### InferenceRouter (`inference/router.rs`)

The router is the brain of inference. It:

1. Receives inference requests from the API server
2. Checks credit balance / priority tier
3. Places request in priority queue
4. When resources are available, assembles a pipeline
5. Kicks off pipeline execution
6. Returns results to the API server (streaming or batch)

### Pipeline Assembly Algorithm

```
Input: InferenceRequest (includes model_id)
Output: PipelineAssignment

1. Fetch model manifest from registry
2. Determine required layer ranges (0..num_layers)
3. Query shard_registry for all nodes hosting shards of this model
4. For each node, fetch current load and latency from peer_registry
5. OPTIMIZATION: Greedy algorithm:
   a. Sort candidate nodes by (latency ASC, load ASC, trust DESC)
   b. Starting from layer 0, assign the best available node that covers
      the widest contiguous layer range (chunk coalescing)
   c. Continue until all layers are covered
6. If any layer range has no available node → request fails (queue for retry)
7. Identify standby nodes for each segment (next-best candidates)
8. Send PipelineAssignment to all participating nodes
9. Wait for all nodes to ACK (they pre-load shards into GPU)
10. Begin forwarding tokens
```

### Local Inference Execution (`inference/executor.rs`)

This wraps llama.cpp for actual tensor computation:

```rust
pub struct ShardExecutor {
    model_path: PathBuf,
    ctx: Option<LlamaContext>,       // llama.cpp context
    gpu_layers: u32,                 // Number of layers offloaded to GPU
}

impl ShardExecutor {
    /// Load shard weights into memory/GPU
    pub async fn load(&mut self, shard: &ShardInfo) -> Result<()>;

    /// Process activation tensors for a layer range
    /// Input: activation tensor from previous layer
    /// Output: activation tensor for next layer
    pub async fn forward(&self, input: &[u8], format: TensorFormat) -> Result<Vec<u8>>;

    /// Full local inference (for super nodes)
    pub async fn generate(
        &self,
        tokens: &[u32],
        params: &SamplingParams,
        callback: impl FnMut(u32) -> bool,  // Token callback, returns false to stop
    ) -> Result<Vec<u32>>;

    /// Unload from GPU/RAM
    pub async fn unload(&mut self) -> Result<()>;
}
```

### KV-Cache Strategy

For multi-turn conversations:
- KV-cache is stored on the nodes that computed each layer
- Subsequent messages in the same conversation include a `session_id`
- The router attempts to reassemble the same pipeline for the same session
- If the pipeline changes (node dropped), KV-cache is invalidated and context is reprocessed
- Sessions expire after 10 minutes of inactivity (configurable)

---

## 9. Credit System

### Earning Rates

These are initial values. They should be tunable via config and will need balancing.

| Action | Credits | Unit |
|---|---|---|
| Serve inference layer pass | 10 | per layer per token |
| Host shard (passive) | 1 | per GB per hour |
| Seed shard data | 5 | per GB transferred |
| Relay service | 2 | per connection hour |
| Submit inference request | -8 | per layer per token |
| Serve failure (timeout) | -50 | per incident |

### Priority Tier Calculation

```rust
fn calculate_tier(balance: i64, network_percentile: f32) -> PriorityTier {
    if network_percentile >= 0.90 { PriorityTier::Platinum }
    else if network_percentile >= 0.70 { PriorityTier::Gold }
    else if balance > 0 { PriorityTier::Silver }
    else { PriorityTier::Bronze }
}
```

The `network_percentile` is calculated locally based on gossip-propagated percentile boundaries. Approximately every 5 minutes, nodes gossip their balance (not exact — bucketed into ranges for privacy) and each node independently estimates the distribution.

### Transaction Signing

Every credit transaction requires dual signatures:

```
1. Serving node creates CreditTransaction with amount and reason
2. Serving node signs with its Ed25519 key
3. Sends to requesting node
4. Requesting node verifies, co-signs
5. Both nodes store the signed transaction locally
6. Periodically, nodes reconcile by exchanging transaction logs with shared peers
```

---

## 10. Identity and Cryptography

### Key Generation (first run)

```rust
pub fn generate_identity(data_dir: &Path) -> Result<Identity> {
    let mut rng = rand::thread_rng();
    let signing_key = ed25519_dalek::SigningKey::generate(&mut rng);
    let verifying_key = signing_key.verifying_key();

    let node_id = NodeId(verifying_key.to_bytes());

    // Store encrypted
    let keystore = Keystore::new(data_dir.join("identity.key"));
    keystore.save(&signing_key, None)?; // None = no passphrase (headless default)

    Ok(Identity { signing_key, node_id })
}
```

### Keystore Encryption

When a passphrase is set:

```
1. Derive key from passphrase using Argon2id (m=64MB, t=3, p=4)
2. Generate random 96-bit nonce
3. Encrypt Ed25519 secret key with AES-256-GCM
4. Store: [version(1B)][salt(16B)][nonce(12B)][ciphertext(48B)][tag(16B)]
```

---

## 11. HTTP API Server

### Axum Router Setup

```rust
pub fn build_router(state: Arc<SharedState>, inference_tx: mpsc::Sender<InferenceRequest>) -> Router {
    Router::new()
        // OpenAI-compatible API
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/models", get(openai::list_models))

        // SwarmLLM extensions
        .route("/v1/status", get(admin::status))

        // Admin API
        .route("/api/admin/stats", get(admin::stats))
        .route("/api/admin/config", get(admin::get_config).put(admin::update_config))
        .route("/api/admin/models", get(admin::list_models))
        .route("/api/admin/models/:id/add", post(admin::add_model_interest))
        .route("/api/admin/peers", get(admin::list_peers))
        .route("/api/admin/credits", get(admin::credit_info))
        .route("/api/admin/ws", get(websocket::handler))

        // Static files (embedded frontend)
        .route("/admin", get(ui::serve_dashboard))
        .route("/admin/*path", get(ui::serve_static))
        .route("/chat", get(ui::serve_chat))
        .route("/setup", get(ui::serve_setup))
        .route("/", get(|| async { Redirect::to("/admin") }))

        // Middleware
        .layer(CorsLayer::permissive())  // localhost only, so permissive is fine
        .with_state(AppState { shared: state, inference_tx })
}
```

### OpenAI-Compatible Chat Completion Endpoint

**Request format** (match OpenAI exactly):

```json
POST /v1/chat/completions
{
    "model": "llama3-70b-q4km",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": true
}
```

**Non-streaming response**:

```json
{
    "id": "swarm-abc123",
    "object": "chat.completion",
    "created": 1709000000,
    "model": "llama3-70b-q4km",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help?"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28
    }
}
```

**Streaming response** (SSE):

```
data: {"id":"swarm-abc123","object":"chat.completion.chunk","created":1709000000,"model":"llama3-70b-q4km","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"swarm-abc123","object":"chat.completion.chunk","created":1709000000,"model":"llama3-70b-q4km","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"swarm-abc123","object":"chat.completion.chunk","created":1709000000,"model":"llama3-70b-q4km","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

### Admin WebSocket

The `/api/admin/ws` endpoint pushes real-time updates to the dashboard:

```json
// Message types pushed to client:
{"type": "stats_update", "data": {"peers": 142, "credits": 5830, "active_requests": 3}}
{"type": "inference_progress", "data": {"request_id": "...", "tokens_generated": 45, "tokens_total": 200}}
{"type": "peer_joined", "data": {"node_id": "a1b2c3d4", "gpu": "RTX 3080"}}
{"type": "shard_status", "data": {"model": "llama3-70b-q4km", "shard": 12, "status": "downloaded"}}
```

Push interval: stats_update every 2 seconds, others as they occur.

---

## 12. Admin Web UI

### Implementation Approach

The frontend is vanilla HTML/CSS/JS (no framework). It is embedded into the binary at compile time using `include_dir!`. This keeps the build simple and avoids Node.js as a build dependency.

### Pages

**`/setup`** — First-run wizard (only shown if `~/.swarmllm/config.toml` doesn't exist):
- Step 1: Hardware detection results (auto-populated, user confirms)
- Step 2: Contribution slider (Minimal ↔ Maximum)
- Step 3: Model selection (checkboxes for available models)
- Step 4: Summary + "Start" button
- On completion: writes config.toml, restarts daemon tasks

**`/admin`** — Main dashboard:
- Header: Node ID (short), version, uptime, priority tier badge
- Panel: Network (peers count, total capacity, chart)
- Panel: Your Node (GPU/RAM/disk usage bars, hosted shards list)
- Panel: Credits (balance, earn/spend rates, tier, sparkline chart)
- Panel: Models (table of available models with health indicators)
- Panel: Active Requests (live list with progress bars)
- Panel: Settings (contribution sliders, scheduling, bandwidth limits)

**`/chat`** — Minimal chat interface:
- Model selector dropdown
- Scrollable message area
- Text input with send button
- Streaming response display
- Conversation history (localStorage)
- Settings gear icon (temperature, max_tokens)

### Frontend ↔ Backend Communication

- Dashboard data: WebSocket (`/api/admin/ws`) for real-time, REST (`/api/admin/*`) for initial load
- Chat: Standard fetch to `/v1/chat/completions` with `stream: true`, parsed via EventSource/ReadableStream
- Settings: PUT to `/api/admin/config`
- All requests go to `localhost:8800` — no CORS issues

### Styling

Use a dark theme by default (users running this are likely technical). Clean, minimal, monospace for data. Use CSS custom properties for theming. No external CSS frameworks. Total frontend size target: < 200KB.

---

## 13. Configuration

### Config File (`~/.swarmllm/config.toml`)

```toml
[node]
# Automatically detected, but can be overridden
data_dir = "~/.swarmllm"
listen_port = 8800                      # Both P2P and HTTP
external_address = ""                   # Auto-detected via AutoNAT
contribution = "moderate"               # "minimal", "moderate", "maximum"

[resources]
max_gpu_vram_mb = 0                     # 0 = auto-detect and use all available
max_ram_mb = 0                          # 0 = auto (50% of system RAM)
max_disk_mb = 50000                     # 50GB default for model storage
max_bandwidth_mbps = 0                  # 0 = unlimited

[resources.schedule]
# Reduce contribution during specified hours (local time)
enabled = false
reduced_hours_start = "19:00"
reduced_hours_end = "23:00"
reduced_contribution = "minimal"

[network]
bootstrap_peers = [
    "/ip4/BOOTSTRAP_IP_1/udp/8800/quic-v1/p2p/PEER_ID_1",
    "/ip4/BOOTSTRAP_IP_2/udp/8800/quic-v1/p2p/PEER_ID_2",
]
enable_relay = true                     # Act as relay for NAT'd peers
enable_relay_client = true              # Use relays if direct connection fails
max_peers = 200
peer_exchange = true

[inference]
default_model = ""                      # Empty = first available
session_timeout_seconds = 600           # KV-cache session expiry
max_concurrent_requests = 10
speculative_decoding = false            # Phase 5 feature

[identity]
encrypted = false                       # Set true to passphrase-protect identity

[ui]
open_browser_on_start = true            # Open dashboard in browser on first run
theme = "dark"                          # "dark" or "light"

[updates]
auto_update = "stable"                  # "disabled", "stable", "all"

[logging]
level = "info"                          # "trace", "debug", "info", "warn", "error"
format = "pretty"                       # "pretty", "json"
file = ""                               # Empty = stdout only
```

### Config Loading Priority

1. CLI flags (highest priority)
2. Environment variables: `SWARMLLM_NODE_LISTEN_PORT=9000` etc. (prefix: `SWARMLLM_`, underscores for nesting)
3. Config file (`~/.swarmllm/config.toml`)
4. Default values (lowest priority)

---

## 14. Database / Local State

### sled Keyspaces

Use sled trees (similar to tables) for logical separation:

| Tree Name | Key | Value | Purpose |
|---|---|---|---|
| `config` | `"config"` | Serialized Config | Persisted config state |
| `identity` | `"keypair"` | Encrypted Ed25519 key | Node identity |
| `credits` | `"balance"` | Serialized CreditBalance | Current credit state |
| `credit_txns` | `{uuid}` | Serialized CreditTransaction | Transaction history |
| `peer_trust` | `{node_id_hex}` | Serialized TrustScore | Per-peer trust data |
| `shard_meta` | `{model_id}/{shard_index}` | Serialized ShardInfo + path | Shard inventory |
| `model_meta` | `{model_id}` | Serialized ModelManifest | Known model manifests |
| `sessions` | `{session_id}` | KV-cache metadata | Active inference sessions |

### Data Directory Layout

```
~/.swarmllm/
├── config.toml
├── identity.key                 # Ed25519 keypair (optionally encrypted)
├── db/                          # sled database files
│   ├── conf
│   ├── db
│   └── snap.*
└── models/
    ├── llama3-70b-q4km/
    │   ├── manifest.json
    │   ├── tokenizer.json
    │   ├── shard_000.bin
    │   ├── shard_001.bin
    │   └── ...
    └── mistral-7b-q5km/
        ├── manifest.json
        └── ...
```

---

## 15. Error Handling

### Error Types (`error.rs`)

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SwarmError {
    // Network
    #[error("Network error: {0}")]
    Network(String),
    #[error("Peer not found: {0}")]
    PeerNotFound(NodeId),
    #[error("Connection failed to {peer}: {reason}")]
    ConnectionFailed { peer: NodeId, reason: String },

    // Inference
    #[error("Model not available: {0}")]
    ModelNotAvailable(ModelId),
    #[error("Insufficient network capacity for model {0}")]
    InsufficientCapacity(ModelId),
    #[error("Pipeline assembly failed: {0}")]
    PipelineError(String),
    #[error("Inference timeout after {0}s")]
    InferenceTimeout(u64),

    // Shards
    #[error("Shard verification failed: expected {expected}, got {actual}")]
    ShardIntegrity { expected: String, actual: String },
    #[error("Shard not found: {0:?}")]
    ShardNotFound(ShardId),

    // Credits
    #[error("Insufficient credits: balance={balance}, required={required}")]
    InsufficientCredits { balance: i64, required: i64 },
    #[error("Invalid transaction signature")]
    InvalidSignature,

    // Identity
    #[error("Keystore error: {0}")]
    Keystore(String),
    #[error("Wrong passphrase")]
    WrongPassphrase,

    // Storage
    #[error("Database error: {0}")]
    Database(#[from] sled::Error),
    #[error("Insufficient disk space: need {need_mb}MB, have {have_mb}MB")]
    InsufficientDisk { need_mb: u64, have_mb: u64 },

    // Config
    #[error("Configuration error: {0}")]
    Config(String),

    // Generic
    #[error("Internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

/// API-facing error that maps SwarmError to HTTP status codes
pub struct ApiError(SwarmError);

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self.0 {
            SwarmError::ModelNotAvailable(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            SwarmError::InsufficientCredits { .. } => (StatusCode::TOO_MANY_REQUESTS, self.0.to_string()),
            SwarmError::InferenceTimeout(_) => (StatusCode::GATEWAY_TIMEOUT, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".into()),
        };
        // Return OpenAI-compatible error format
        (status, Json(json!({
            "error": { "message": message, "type": "swarm_error", "code": status.as_u16() }
        }))).into_response()
    }
}
```

### Error Propagation Rules

1. **Network errors**: Log + retry with backoff (3 attempts, exponential). After max retries, propagate up.
2. **Inference errors**: Return to client immediately with appropriate HTTP status. Do NOT retry inference silently.
3. **Shard integrity errors**: Quarantine shard, re-download from another peer, penalize offending node's trust.
4. **Credit errors**: Never block on credit issues. Degrade to lower priority tier instead of failing.

---

## 16. Testing Strategy

### Unit Tests

Every module has its own unit tests. Focus on:
- `types.rs`: Serialization round-trips for all types
- `credit/ledger.rs`: Balance calculations, tier boundaries, overflow protection
- `credit/transaction.rs`: Signature creation and verification
- `identity/keypair.rs`: Key generation, export/import
- `identity/keystore.rs`: Encrypt/decrypt round-trip
- `inference/scheduler.rs`: Pipeline assembly with various node configurations
- `inference/sampling.rs`: Token sampling correctness
- `config.rs`: Config loading, merging, migration

### Integration Tests

Located in `tests/integration/`. These spin up multiple daemon instances in the same process:

- **`network_test.rs`**: 3 nodes discover each other, exchange shard announcements
- **`inference_test.rs`**: 3 nodes with a tiny test model (2-layer, random weights), end-to-end inference
- **`credit_test.rs`**: Serve inference → verify credit transfer → verify priority change
- **`api_test.rs`**: HTTP requests to OpenAI-compatible endpoint, verify response format

### Test Model

Create a minimal test model in `tests/fixtures/tiny_model/`:
- 2 transformer layers with random weights
- 128 hidden dim, 4 attention heads
- ~1MB total size
- Split into 2 shards (1 layer each)
- Used exclusively for integration testing, never for actual inference

### CI Pipeline

```yaml
# Conceptual — implement in GitHub Actions
steps:
  - cargo fmt --check
  - cargo clippy -- -D warnings
  - cargo test --workspace
  - cargo test --test integration -- --test-threads=1  # Integration tests are sequential
  - cargo build --release
```

---

## 17. Build Phases

### IMPORTANT: Build in this exact order. Each phase is a deployable milestone.

---

### Phase 1: Local Inference Daemon (Weeks 1-4)

**Goal**: Single binary that loads a GGUF model and serves it via OpenAI-compatible API.

**Build order**:
1. `main.rs` — CLI with clap (subcommands: `run`, `version`)
2. `config.rs` — Load config from TOML, environment, defaults
3. `error.rs` — Error types
4. `types.rs` — Core types (just ModelId, SamplingParams, ChatMessage, Role for now)
5. `inference/executor.rs` — Wrap llama-cpp-2 to load GGUF model and generate tokens
6. `inference/sampling.rs` — Temperature, top-p, top-k (may be handled by llama.cpp)
7. `api/server.rs` — Axum server on localhost:8800
8. `api/openai.rs` — `/v1/chat/completions` (streaming + non-streaming), `/v1/models`
9. `storage/db.rs` — sled wrapper (just config storage for now)

**Acceptance test**: `curl localhost:8800/v1/chat/completions -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":true}'` returns streamed tokens from a local GGUF model.

**What this does NOT include**: No networking, no P2P, no credits. It's basically a Rust Ollama at this point.

---

### Phase 2: P2P Networking Foundation (Weeks 5-10)

**Goal**: Nodes discover each other, exchange shard information, and can transfer shard data.

**Build order**:
1. `identity/keypair.rs` — Ed25519 key generation
2. `identity/keystore.rs` — Encrypted storage
3. `network/transport.rs` — QUIC transport setup
4. `network/behaviour.rs` — Custom NetworkBehaviour (Kademlia + GossipSub + request_response)
5. `network/discovery.rs` — Bootstrap, peer discovery loop
6. `network/manager.rs` — Swarm lifecycle, message routing
7. `network/protocol.rs` — SwarmMessage serialization (start with serde_json, move to Cap'n Proto later)
8. `model/manifest.rs` — Parse .swarm manifests
9. `model/shard.rs` — Shard loading, BLAKE3 verification
10. `model/distribution.rs` — Shard request/response protocol
11. `model/registry.rs` — Track known models and shard locations
12. `health/monitor.rs` — Periodic health pings

**Acceptance test**: Start 3 nodes on a LAN. Node A has a model shard. Node B discovers Node A, downloads the shard. Node C can verify the shard exists on both A and B.

---

### Phase 3: Distributed Inference (Weeks 11-18)

**Goal**: Inference request flows through a pipeline of multiple nodes.

**Build order**:
1. `inference/router.rs` — Request queuing, pipeline assembly trigger
2. `inference/scheduler.rs` — Pipeline assembly algorithm (greedy layer assignment)
3. `inference/pipeline.rs` — Pipeline execution: forward activations between nodes
4. `inference/kv_cache.rs` — Session-based KV-cache management
5. Wire `NetworkManager` ↔ `InferenceRouter` ↔ `ShardExecutor` communication
6. Implement `LayerForward` and `LayerResult` message handling in NetworkManager
7. Implement hot-standby failover in pipeline.rs
8. Add `PipelineAssignment` to SharedState for monitoring

**Acceptance test**: 3 nodes, each holding different layer ranges of Llama 3 70B. Client sends request to Node A, which assembles pipeline across all 3 nodes, returns generated text.

---

### Phase 4: Credit System (Weeks 19-22)

**Build order**:
1. `credit/ledger.rs` — Local balance tracking, credit operations
2. `credit/transaction.rs` — Transaction creation, dual signing
3. `credit/priority.rs` — Tier calculation, queue ordering
4. `credit/anti_gaming.rs` — Basic spot-check verification
5. Wire credit events into inference pipeline (earn on serve, spend on request)
6. Add credit gossip for percentile estimation
7. Priority queue in InferenceRouter respects tiers

**Acceptance test**: Node A serves 100 inference requests for Node B. Node A's balance increases proportionally. Node B's balance decreases. Node A achieves higher priority tier.

---

### Phase 5: Web UI and UX (Weeks 23-28)

**Build order**:
1. `frontend/setup.html` + `setup.js` — First-run wizard
2. `frontend/index.html` + `app.js` — Admin dashboard
3. `frontend/chat.html` + `chat.js` — Chat interface
4. `frontend/css/style.css` — Dark theme styling
5. `build.rs` — Embed frontend assets with include_dir
6. `ui/assets.rs` — Serve embedded files
7. `api/admin.rs` — All admin REST endpoints
8. `api/websocket.rs` — Real-time dashboard updates
9. Hardware auto-detection in config.rs (GPU probing, RAM, disk)
10. `open_browser_on_start` logic in daemon.rs

**Acceptance test**: Fresh install. Run binary. Browser opens. Setup wizard detects hardware. User completes wizard. Dashboard shows real-time stats. Chat works.

---

### Phase 6: Hardening and Scale (Weeks 29+)

- NAT traversal: AutoNAT, DCUtR, relay
- Protocol migration from serde_json to Cap'n Proto for tensor data
- Shard rebalancing on node join/leave
- Model governance voting
- Cross-platform builds and testing
- Speculative decoding
- MoE-optimized sharding

---

### Phase 7: Self-Governance (Weeks 35+)

**Goal**: Fully decentralized development, issue tracking, proposals, testing, and releases.

See [Section 21](#21-self-governed-updates-and-development) for complete specification.

**Build order**: governance/proposals.rs → governance/voting.rs → governance/issues.rs → governance/releases.rs → governance/testing.rs → governance/changelog.rs → NetworkManager integration → API endpoints → Dashboard tabs → UpdateManager

**Acceptance test**: Full lifecycle from issue filing through proposal, vote, build, test, approve, and canary rollout — with no central coordination.

---

## 18. CLI Interface

```
swarmllm 0.1.0
Decentralized peer-to-peer LLM inference network

USAGE:
    swarmllm <COMMAND>

COMMANDS:
    run         Start the SwarmLLM daemon (default)
    status      Show node status (queries running daemon)
    models      List available models
    credits     Show credit balance and tier
    config      Print current configuration
    identity    Manage node identity
      export    Export identity to file
      import    Import identity from file
      show      Show node ID
    version     Print version information
    help        Print help

OPTIONS:
    -c, --config <PATH>     Config file path [default: ~/.swarmllm/config.toml]
    -p, --port <PORT>       Listen port [default: 8800]
    -d, --data-dir <PATH>   Data directory [default: ~/.swarmllm]
    -v, --verbose           Increase log verbosity (repeat for more: -vv, -vvv)
    --no-browser            Don't open browser on start
    --headless              No browser, no setup wizard (use config file)
```

The `status`, `models`, and `credits` commands work by making HTTP requests to the running daemon's API. If the daemon isn't running, they print an error.

---

## 19. Logging and Observability

### Tracing Setup

```rust
fn init_tracing(cli: &Cli) {
    let filter = match cli.verbose {
        0 => "swarmllm=info",
        1 => "swarmllm=debug",
        2 => "swarmllm=debug,libp2p=info",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}
```

### Log Targets

Use structured tracing spans for context:

```rust
// In network manager
let span = tracing::info_span!("network", peer_count = peers.len());
let _guard = span.enter();
tracing::info!(peer_id = %peer, "New peer connected");

// In inference router
let span = tracing::info_span!("inference", request_id = %req.id, model = %req.model_id);
let _guard = span.enter();
tracing::debug!(pipeline_nodes = ?assignment.segments.len(), "Pipeline assembled");
```

### Key Metrics to Log

- `swarmllm.peers.connected` — gauge
- `swarmllm.inference.requests` — counter
- `swarmllm.inference.latency_ms` — histogram
- `swarmllm.inference.tokens_per_sec` — gauge
- `swarmllm.credits.balance` — gauge
- `swarmllm.shards.hosted` — gauge
- `swarmllm.network.bandwidth_in_bytes` — counter
- `swarmllm.network.bandwidth_out_bytes` — counter

These are logged as tracing events. A future phase could export to Prometheus via a `/metrics` endpoint.

---

## 20. Platform Support

### Build Targets

| Platform | Priority | GPU Support | Notes |
|---|---|---|---|
| Linux x86_64 | P0 | CUDA + ROCm | Primary development target |
| macOS aarch64 | P1 | Metal (via llama.cpp) | Apple Silicon |
| macOS x86_64 | P2 | CPU only | Intel Macs |
| Windows x86_64 | P1 | CUDA | Via MSVC toolchain |
| Linux aarch64 | P3 | CPU only | Raspberry Pi, ARM servers |

### GPU Detection

On startup, probe for GPU capabilities:

```rust
pub fn detect_gpu() -> Option<GpuInfo> {
    // 1. Try CUDA (nvidia-smi or CUDA runtime API)
    // 2. Try ROCm (rocm-smi)
    // 3. Try Metal (macOS IOKit)
    // 4. None = CPU-only mode
}
```

This is handled by llama.cpp's backend detection in Phase 1. Custom GPU probing comes in Phase 5.

### Data Directory

Use the `dirs` crate for platform-appropriate defaults:
- Linux: `~/.swarmllm/`
- macOS: `~/Library/Application Support/swarmllm/`
- Windows: `%APPDATA%\swarmllm\`

---

## 21. Self-Governed Updates and Development

### Overview

SwarmLLM has no central repository, no GitHub, no single maintainer. The entire development lifecycle — code changes, bug reports, feature requests, testing, releases, and rollouts — happens through the network itself. Every node is both a user and a potential contributor to the project's evolution.

The system is modeled on a combination of RFC-style governance (like Rust's own RFC process), BitTorrent's decentralized distribution, and blockchain's threshold signing — but without the blockchain.

### 21.1 Governance Architecture

#### GossipSub Topics for Governance

| Topic | Purpose | Message Types |
|---|---|---|
| `swarm/gov/proposals` | New proposals, amendments, status changes | `Proposal`, `ProposalAmendment`, `ProposalStatusChange` |
| `swarm/gov/votes` | Votes on active proposals | `ProposalVote` |
| `swarm/gov/issues` | Bug reports, feature requests | `Issue`, `IssueComment`, `IssueStatusChange` |
| `swarm/gov/releases` | Release candidates, test results, approvals | `ReleaseCandidate`, `TestReport`, `ReleaseApproval` |
| `swarm/gov/changelog` | Finalized changelogs for accepted releases | `ChangelogEntry` |

#### DHT Keys for Governance

| Key Pattern | Value |
|---|---|
| `/swarm/gov/proposal/{proposal_hash}` | Full `Proposal` content |
| `/swarm/gov/issue/{issue_hash}` | Full `Issue` content |
| `/swarm/gov/release/{version}` | `ReleaseCandidate` with binary hashes |
| `/swarm/gov/release/latest` | Latest stable version string + hash |
| `/swarm/gov/release/latest-pre` | Latest pre-release version string + hash |
| `/swarm/gov/council` | Current list of council member `NodeId`s |

### 21.2 Roles and Permissions

Permissions are earned through contribution, not granted by a central authority.

#### Role Tiers

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum GovernanceRole {
    /// Any node on the network. Can file issues, comment, vote on proposals.
    Member,

    /// Top 20% contributors by lifetime credits AND 30+ days uptime.
    /// Can create proposals (RFCs), submit release candidates.
    Contributor,

    /// Top 5% contributors by lifetime credits AND 90+ days uptime AND
    /// at least 3 accepted proposals. Can approve releases (threshold signer).
    Maintainer,

    /// Elected by Maintainer vote. 7-seat council with 1-year terms.
    /// Can emergency-veto proposals, break tie votes, set release signing threshold.
    Council,
}

impl GovernanceRole {
    pub fn from_node_stats(stats: &NodeStats, network_stats: &NetworkStats) -> Self {
        if stats.is_council_member {
            GovernanceRole::Council
        } else if stats.lifetime_earned_percentile >= 0.95
            && stats.uptime_days >= 90
            && stats.accepted_proposals >= 3
        {
            GovernanceRole::Maintainer
        } else if stats.lifetime_earned_percentile >= 0.80
            && stats.uptime_days >= 30
        {
            GovernanceRole::Contributor
        } else {
            GovernanceRole::Member
        }
    }
}
```

#### Role Capabilities

| Action | Member | Contributor | Maintainer | Council |
|---|---|---|---|---|
| File issues | ✓ | ✓ | ✓ | ✓ |
| Comment on issues/proposals | ✓ | ✓ | ✓ | ✓ |
| Vote on proposals | ✓ | ✓ | ✓ | ✓ |
| Create proposals (RFCs) | — | ✓ | ✓ | ✓ |
| Submit release candidates | — | ✓ | ✓ | ✓ |
| Approve releases (threshold sign) | — | — | ✓ | ✓ |
| Emergency veto | — | — | — | ✓ |
| Set governance parameters | — | — | — | ✓ (supermajority) |

### 21.3 Issue Tracking

Issues are the equivalent of GitHub Issues but stored and propagated through the network.

#### Data Types

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub hash: Blake3Hash,                    // Hash of (author + title + body + created_at)
    pub author: NodeId,
    pub title: String,                       // Max 200 chars
    pub body: String,                        // Max 10,000 chars, plaintext + basic markdown
    pub category: IssueCategory,
    pub severity: Option<IssueSeverity>,      // Only for bugs
    pub status: IssueStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub upvotes: u32,                        // Aggregated from network
    pub tags: Vec<String>,                   // Max 5 tags, max 30 chars each
    pub signature: Vec<u8>,                  // Ed25519 by author
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IssueCategory {
    Bug,
    FeatureRequest,
    Performance,
    Security,
    Documentation,
    ModelRequest,                             // Request to add a new model
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,   // Network-breaking, data loss, security vulnerability
    High,       // Major feature broken, significant performance regression
    Medium,     // Feature partially broken, workaround exists
    Low,        // Cosmetic, minor inconvenience
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IssueStatus {
    Open,
    Acknowledged,   // A Contributor+ has confirmed the issue
    InProgress,     // Linked to a Proposal
    Resolved,       // Fix released
    Closed,         // Won't fix / duplicate / invalid
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueComment {
    pub issue_hash: Blake3Hash,
    pub author: NodeId,
    pub body: String,                        // Max 5,000 chars
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueUpvote {
    pub issue_hash: Blake3Hash,
    pub voter: NodeId,
    pub weight: u64,                         // Contribution-weighted
    pub signature: Vec<u8>,
}
```

#### Issue Lifecycle

```
1. Any node creates an Issue → gossips to `swarm/gov/issues`
2. Other nodes receive, store locally, display in dashboard
3. Nodes can upvote (weighted by contribution) and comment
4. A Contributor+ can change status to Acknowledged
5. A Contributor+ can create a Proposal that references the issue
6. When the linked Proposal is accepted and released → status becomes Resolved
7. Issues with no activity for 90 days auto-close (nodes run this locally)
```

#### Issue Deduplication and Spam Prevention

- Issues are identified by their BLAKE3 content hash — identical issues are automatically deduplicated
- Nodes must have a positive credit balance to file issues (prevents zero-effort spam)
- Issues from nodes with < 24 hours uptime are deprioritized in the UI (anti-spam for new identities)
- Upvote weight is contribution-based, so legitimate users' priorities naturally rise

### 21.4 Proposal System (RFCs)

Proposals are the mechanism for all code changes, protocol modifications, and governance updates. Think of them as GitHub PRs + Rust RFCs combined.

#### Data Types

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub hash: Blake3Hash,                    // Hash of (author + title + body + created_at)
    pub author: NodeId,
    pub title: String,                       // Max 200 chars
    pub body: String,                        // Max 50,000 chars, markdown
    pub category: ProposalCategory,
    pub status: ProposalStatus,
    pub linked_issues: Vec<Blake3Hash>,      // Issues this proposal addresses
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub voting_deadline: chrono::DateTime<chrono::Utc>,  // created_at + voting_period
    pub signature: Vec<u8>,

    // If this proposal includes code changes:
    pub patch: Option<ProposalPatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalCategory {
    CodeChange,          // Bug fix, feature implementation, refactor
    ProtocolChange,      // Changes to network protocol (requires supermajority)
    GovernanceChange,    // Changes to governance rules (requires supermajority)
    ModelAddition,       // Add a new model to the network
    ModelDeprecation,    // Remove a model from the network
    ParameterTuning,     // Adjust credit rates, replication factors, thresholds
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalPatch {
    /// Unified diff format. The actual code change.
    /// For large changes, this is a hash pointing to the full diff
    /// stored as a shard in the network (same distribution mechanism as models).
    pub diff_hash: Blake3Hash,
    pub diff_size_bytes: u64,
    pub files_changed: Vec<String>,          // List of file paths modified
    pub insertions: u32,
    pub deletions: u32,

    /// If the diff is small enough (< 64KB), inline it directly.
    /// Otherwise, nodes download it via shard distribution.
    pub inline_diff: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProposalStatus {
    Draft,              // Author is still editing (not yet open for voting)
    Open,               // Open for discussion and voting
    Amended,            // Author has posted an amendment (resets voting period)
    Accepted,           // Passed vote threshold
    Rejected,           // Failed vote threshold or vetoed
    Implemented,        // Code merged into a release candidate
    Released,           // Included in a stable release
    Withdrawn,          // Author withdrew the proposal
}
```

#### Voting Mechanism

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalVote {
    pub proposal_hash: Blake3Hash,
    pub voter: NodeId,
    pub vote: VoteChoice,
    pub weight: u64,                     // Derived from lifetime_earned credits
    pub role: GovernanceRole,            // Voter's role at time of voting
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VoteChoice {
    Approve,
    Reject,
    Abstain,        // Counts toward quorum but not approval ratio
}
```

#### Voting Rules

| Proposal Category | Voting Period | Quorum | Approval Threshold |
|---|---|---|---|
| CodeChange | 7 days | 10% of network weight | >50% of votes cast |
| ProtocolChange | 14 days | 20% of network weight | >66% of votes cast |
| GovernanceChange | 21 days | 25% of network weight | >75% of votes cast |
| ModelAddition | 7 days | 10% of network weight | >30% of votes cast |
| ModelDeprecation | 14 days | 15% of network weight | >50% of votes cast |
| ParameterTuning | 7 days | 10% of network weight | >50% of votes cast |

**Weight calculation**: A node's vote weight equals its `lifetime_earned` credits. This means long-term, high-contributing nodes have the most influence. New nodes can still vote but carry less weight.

**Quorum**: Calculated as percentage of total network vote weight (sum of all active nodes' lifetime_earned). "Active" means the node has been online in the last 7 days.

#### Proposal Lifecycle

```
1. Contributor+ creates Proposal (status: Draft)
2. Author finalizes → status: Open, voting_deadline set
3. Voting period begins. Nodes vote via `swarm/gov/votes`
4. Author can post ProposalAmendment → status: Amended, deadline resets
   (max 3 amendments, then proposal must be withdrawn and resubmitted)
5. At deadline:
   a. If quorum met AND approval threshold met → status: Accepted
   b. Otherwise → status: Rejected
6. Council can emergency veto within 48h of acceptance (requires 5/7 council votes)
7. Accepted proposal enters implementation queue
8. When included in a ReleaseCandidate → status: Implemented
9. When release goes stable → status: Released
```

### 21.5 Release System

Releases are how accepted proposals become running code on the network. There is no central build server — releases are built by contributors and validated by the network.

#### Release Candidate

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseCandidate {
    pub version: SemVer,                     // e.g., "0.2.0-rc.1"
    pub builder: NodeId,                     // Who built this candidate
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub changelog: String,                   // Auto-generated from accepted proposals
    pub included_proposals: Vec<Blake3Hash>, // Proposals in this release
    pub binaries: Vec<ReleaseBinary>,
    pub source_hash: Blake3Hash,             // Hash of source tarball
    pub signature: Vec<u8>,                  // Builder's signature
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseBinary {
    pub platform: Platform,
    pub arch: Architecture,
    pub hash: Blake3Hash,                    // BLAKE3 hash of the binary
    pub size_bytes: u64,
    pub shard_manifest: Blake3Hash,          // For P2P distribution
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Platform { Linux, MacOS, Windows }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Architecture { X86_64, Aarch64 }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre: Option<String>,                 // "rc.1", "beta.2", etc.
}
```

#### Release Build Process

Since there's no CI server, releases are built by Contributor+ nodes and verified by the network:

```
1. Contributor builds release from accepted proposals:
   a. Applies all accepted proposal patches to current stable source
   b. Runs full test suite locally
   c. Builds binaries for all supported platforms (cross-compilation or
      multiple contributors build for their native platform)
   d. Creates ReleaseCandidate with hashes of all binaries + source

2. ReleaseCandidate is published to `swarm/gov/releases`

3. Verification phase (7 days for stable, 3 days for patches):
   a. Other nodes download the source tarball via P2P
   b. Build from source and verify their binary hash matches the candidate
      (reproducible builds — see section below)
   c. Run test suite
   d. Submit TestReport to `swarm/gov/releases`

4. Approval phase:
   a. Maintainers review test reports and verify binary hashes
   b. Each approving Maintainer submits a ReleaseApproval (threshold signature)
   c. Threshold: 3 of N Maintainers must approve (where N = total Maintainers)
   d. Council members count as Maintainers for this purpose

5. Once threshold is met:
   a. Release is marked as stable
   b. Binary shards are distributed via P2P (same mechanism as model distribution)
   c. `/swarm/gov/release/latest` DHT key is updated
   d. Nodes with auto_update = "stable" begin downloading
```

#### Reproducible Builds

Critical for trustless verification. The build must be deterministic:

```rust
// build.rs ensures deterministic compilation
fn main() {
    // Pin all dependency versions via Cargo.lock (committed to source)
    // Use SOURCE_DATE_EPOCH for timestamps
    // Strip debug symbols deterministically
    // Document exact rustc version in release manifest

    println!("cargo:rustc-env=SOURCE_DATE_EPOCH={}", /* release timestamp */);
    println!("cargo:rustc-env=SWARMLLM_VERSION={}", /* from Cargo.toml */);
}
```

Build reproducibility requirements:
- Cargo.lock is committed and included in source tarball
- Exact rustc version is specified in the ReleaseCandidate
- SOURCE_DATE_EPOCH eliminates timestamp-based non-determinism
- `strip = true` in release profile for consistent binary size
- Docker build container image hash is included for full reproducibility (optional)

If a verifying node builds from source and gets a different hash, they submit a `TestReport` with `status: HashMismatch`, which blocks the release until investigated.

#### Threshold Signing

Release approval uses a simple threshold scheme (not full multisig):

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseApproval {
    pub release_version: SemVer,
    pub approver: NodeId,
    pub role: GovernanceRole,                // Must be Maintainer or Council
    pub binary_hashes_verified: bool,        // Approver independently built and verified
    pub test_suite_passed: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,                  // Ed25519 signature over (version + binary hashes)
}
```

The daemon validates a release by:
1. Collecting all `ReleaseApproval` messages for that version
2. Verifying each signature against known Maintainer/Council NodeIds
3. Checking that at least `threshold` unique approvers signed
4. Verifying the binary hash matches the signed hash
5. Only then applying the update

The threshold defaults to 3 but is adjustable via GovernanceChange proposal.

### 21.6 Update Distribution and Application

#### Binary Distribution

Release binaries are chunked and distributed identically to model shards:

```
~/.swarmllm/updates/
├── 0.2.0/
│   ├── release.json          # ReleaseCandidate metadata
│   ├── approvals/            # Collected ReleaseApproval messages
│   │   ├── a1b2c3d4.json
│   │   ├── e5f6g7h8.json
│   │   └── i9j0k1l2.json
│   └── swarmllm              # The new binary
└── 0.1.0/
    └── swarmllm              # Previous version (for rollback)
```

#### Update Application Process

```rust
pub struct UpdateManager {
    current_version: SemVer,
    update_dir: PathBuf,
    config: UpdateConfig,
}

impl UpdateManager {
    /// Check for available updates (called periodically)
    pub async fn check(&self) -> Option<ReleaseCandidate> {
        // 1. Query DHT for /swarm/gov/release/latest
        // 2. Compare version with current_version
        // 3. If newer, return the ReleaseCandidate
    }

    /// Download and verify an update
    pub async fn download(&self, release: &ReleaseCandidate) -> Result<PathBuf> {
        // 1. Download binary shard via P2P distribution
        // 2. Verify BLAKE3 hash matches release manifest
        // 3. Verify threshold approvals exist and signatures are valid
        // 4. Store in update_dir/{version}/
        // 5. Return path to new binary
    }

    /// Apply update (replace current binary and restart)
    pub async fn apply(&self, new_binary: &Path) -> Result<()> {
        // 1. Copy current binary to update_dir/{current_version}/ (rollback backup)
        // 2. Replace current binary with new binary
        //    Linux/macOS: atomic rename
        //    Windows: rename-on-reboot or use a helper process
        // 3. Spawn new binary with same args
        // 4. New process performs health check on startup
        // 5. If health check fails within 30s, rollback automatically
        // 6. Old process exits
    }

    /// Rollback to previous version
    pub async fn rollback(&self) -> Result<()> {
        // 1. Find previous version in update_dir/
        // 2. Replace current binary with previous version
        // 3. Restart
        // 4. Broadcast rollback event to gossip (so network knows)
    }
}
```

#### Health Check After Update

On startup after an update, the daemon runs a self-test:

```rust
pub async fn post_update_health_check() -> Result<()> {
    // 1. Config loads successfully
    // 2. Identity loads successfully
    // 3. Database opens successfully
    // 4. Network listener binds successfully
    // 5. At least 1 peer connection established within 30s
    // 6. API server responds to GET /v1/models within 5s
    // If ANY check fails → trigger automatic rollback
}
```

#### Update Modes

Configured via `config.toml`:

```toml
[updates]
auto_update = "stable"      # "disabled" | "stable" | "all"
check_interval_hours = 6    # How often to check for updates
auto_restart = true         # Apply immediately or wait for manual restart
keep_versions = 3           # Number of old versions to keep for rollback
```

| Mode | Behavior |
|---|---|
| `disabled` | Never auto-update. Dashboard shows available updates with manual apply button. |
| `stable` | Auto-download and apply stable releases. Pre-releases shown but not auto-applied. |
| `all` | Auto-download and apply all releases including pre-releases (for contributors/testers). |

### 21.7 Distributed Testing

Before a release is approved, it needs testing across diverse hardware and configurations. The network itself serves as the test infrastructure.

#### Test Reports

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestReport {
    pub release_version: SemVer,
    pub tester: NodeId,
    pub platform: Platform,
    pub architecture: Architecture,
    pub gpu: Option<String>,                 // GPU model if present
    pub status: TestStatus,
    pub results: TestResults,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed { failures: Vec<String> },        // List of failed test names
    HashMismatch { expected: String, actual: String },
    BuildFailed { error: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestResults {
    pub tests_run: u32,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub tests_skipped: u32,
    pub build_time_seconds: u64,
    pub binary_hash: Blake3Hash,
    pub binary_size_bytes: u64,
}
```

#### Canary Rollout

Releases don't go to 100% of the network immediately:

```
Phase 1 (Day 0-3):   Nodes with auto_update = "all" (contributors/testers)
Phase 2 (Day 3-7):   5% of auto_update = "stable" nodes (random selection)
Phase 3 (Day 7-10):  25% of auto_update = "stable" nodes
Phase 4 (Day 10+):   100% of auto_update = "stable" nodes
```

Each phase only advances if:
- No TestReports with `status: Failed` or `HashMismatch` from the current phase
- No emergency veto from Council
- Network health metrics haven't degraded (inference success rate, average latency)

If problems are detected at any phase, rollout halts and the release is flagged for investigation. The network automatically serves the previous stable version to new updaters.

#### CLI Testing Command

```
swarmllm test                        # Run full test suite locally
swarmllm test --release 0.2.0-rc.1  # Build RC from source, run tests, submit TestReport
swarmllm test --report               # Submit TestReport for current running version
```

### 21.8 Patch Distribution for Code Changes

Since there's no git repository, code patches are distributed through the network:

#### Patch Storage

Accepted proposal patches are stored as content-addressed blobs in the network, using the same shard distribution mechanism as models:

```
1. Proposal author creates a unified diff
2. Diff is hashed (BLAKE3) and referenced in the Proposal
3. If diff < 64KB: inlined in the Proposal message
4. If diff >= 64KB: stored as a shard, distributed via P2P
5. Any node can download the diff and apply it locally to verify
```

#### Source Tarball

Each stable release includes a source tarball (content-hashed, P2P distributed):

```
swarmllm-0.2.0-source.tar.gz
├── src/                     # Complete source tree
├── Cargo.toml
├── Cargo.lock
├── build.rs
├── proto/
├── frontend/
├── tests/
├── config/
├── CHANGELOG.md             # Auto-generated
└── RELEASE_NOTES.md         # From the ReleaseCandidate
```

Nodes that want to verify or contribute can:
1. Download the source tarball for any version
2. Apply proposal patches to create the next version
3. Build and verify against the release candidate hashes

### 21.9 Dashboard Integration

The governance system is fully accessible through the admin dashboard:

#### Dashboard Panels

**Issues Tab** (`/admin#issues`):
- List of open issues, sorted by upvote weight
- Filter by category, severity, status
- "New Issue" button (for nodes with positive credit balance)
- Comment thread per issue
- Upvote button with live count

**Proposals Tab** (`/admin#proposals`):
- List of active proposals with voting status (progress bar toward quorum + approval)
- Filter by category, status
- "New Proposal" button (for Contributor+ nodes)
- Full proposal body with markdown rendering
- Vote buttons (Approve / Reject / Abstain)
- Linked issues display
- Diff viewer for code changes (syntax highlighted)

**Releases Tab** (`/admin#releases`):
- Current version display
- Available updates with changelog
- "Update Now" button (if auto_update is disabled)
- Release history with test report summaries
- Canary rollout progress indicator
- Rollback button (to previous stable)

**Governance Tab** (`/admin#governance`):
- Your role and eligibility status
- Council members list with term expiry
- Network governance parameters (voting thresholds, quorum requirements)
- Upcoming council elections

#### Admin API Endpoints

```
GET  /api/admin/issues                   # List issues (paginated, filterable)
POST /api/admin/issues                   # Create new issue
GET  /api/admin/issues/:hash             # Get issue details + comments
POST /api/admin/issues/:hash/comment     # Add comment
POST /api/admin/issues/:hash/upvote      # Upvote issue

GET  /api/admin/proposals                # List proposals (paginated, filterable)
POST /api/admin/proposals                # Create new proposal (Contributor+)
GET  /api/admin/proposals/:hash          # Get proposal details + votes
POST /api/admin/proposals/:hash/vote     # Cast vote
GET  /api/admin/proposals/:hash/diff     # Get code diff

GET  /api/admin/releases                 # List releases
GET  /api/admin/releases/latest          # Get latest stable release info
GET  /api/admin/releases/:version        # Get specific release details
POST /api/admin/releases/:version/test   # Submit test report
POST /api/admin/releases/:version/apply  # Trigger manual update

GET  /api/admin/governance/role          # Get your current role
GET  /api/admin/governance/council       # Get council members
GET  /api/admin/governance/params        # Get governance parameters
```

### 21.10 sled Database Trees for Governance

| Tree Name | Key | Value | Purpose |
|---|---|---|---|
| `issues` | `{issue_hash_hex}` | Serialized `Issue` | All known issues |
| `issue_comments` | `{issue_hash_hex}/{timestamp}` | Serialized `IssueComment` | Comments on issues |
| `issue_upvotes` | `{issue_hash_hex}/{voter_hex}` | Serialized `IssueUpvote` | Upvote records |
| `proposals` | `{proposal_hash_hex}` | Serialized `Proposal` | All known proposals |
| `proposal_votes` | `{proposal_hash_hex}/{voter_hex}` | Serialized `ProposalVote` | Vote records |
| `releases` | `{version_string}` | Serialized `ReleaseCandidate` | Release candidates |
| `release_approvals` | `{version_string}/{approver_hex}` | Serialized `ReleaseApproval` | Threshold approvals |
| `test_reports` | `{version_string}/{tester_hex}` | Serialized `TestReport` | Test results |
| `gov_params` | `"params"` | Serialized `GovernanceParams` | Current governance config |

### 21.11 Governance Parameters

These are the tunable values that govern the governance system itself. They can only be changed via a GovernanceChange proposal (75% supermajority).

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceParams {
    // Voting
    pub code_change_voting_days: u32,            // Default: 7
    pub protocol_change_voting_days: u32,        // Default: 14
    pub governance_change_voting_days: u32,       // Default: 21
    pub code_change_quorum_pct: f32,             // Default: 0.10
    pub protocol_change_quorum_pct: f32,         // Default: 0.20
    pub governance_change_quorum_pct: f32,        // Default: 0.25
    pub code_change_approval_pct: f32,           // Default: 0.50
    pub protocol_change_approval_pct: f32,       // Default: 0.66
    pub governance_change_approval_pct: f32,      // Default: 0.75

    // Roles
    pub contributor_percentile: f32,             // Default: 0.80
    pub contributor_min_uptime_days: u32,        // Default: 30
    pub maintainer_percentile: f32,              // Default: 0.95
    pub maintainer_min_uptime_days: u32,         // Default: 90
    pub maintainer_min_accepted_proposals: u32,  // Default: 3
    pub council_seats: u32,                      // Default: 7
    pub council_term_days: u32,                  // Default: 365

    // Releases
    pub release_approval_threshold: u32,         // Default: 3 maintainers
    pub release_verification_days_stable: u32,   // Default: 7
    pub release_verification_days_patch: u32,    // Default: 3
    pub canary_phase1_days: u32,                 // Default: 3
    pub canary_phase2_pct: f32,                  // Default: 0.05
    pub canary_phase3_pct: f32,                  // Default: 0.25
    pub max_proposal_amendments: u32,            // Default: 3

    // Issues
    pub min_credit_balance_for_issues: i64,      // Default: 0 (positive balance)
    pub issue_auto_close_days: u32,              // Default: 90
}
```

### 21.12 Build Phase for Governance

This is a **Phase 6+** feature. It should NOT be built until Phases 1-5 are stable. However, the data types and gossip topics should be defined early so the protocol can accommodate them.

**Build order when ready**:
1. `governance/proposals.rs` — Proposal CRUD, validation, lifecycle state machine
2. `governance/voting.rs` — Vote collection, quorum calculation, tallying, deadline enforcement
3. `governance/issues.rs` — Issue CRUD, upvoting, commenting, auto-close
4. `governance/releases.rs` — ReleaseCandidate management, threshold approval verification
5. `governance/testing.rs` — TestReport collection, canary phase advancement logic
6. `governance/changelog.rs` — Auto-generate changelog from accepted proposals
7. Wire governance messages into `NetworkManager` (new GossipSub topics)
8. Wire governance into `ApiServer` (new admin endpoints)
9. Wire governance into admin dashboard (new tabs)
10. `UpdateManager` integration into `daemon.rs`

**Acceptance test**: Node A (Contributor) creates a Proposal. Nodes B, C, D vote to approve. Node E (Maintainer) builds a ReleaseCandidate. Nodes F, G, H (Maintainers) verify and approve. Node I (regular node with auto_update=stable) receives and applies the update during canary rollout.

### 21.13 Bootstrap Problem

The governance system has a chicken-and-egg problem: early in the network's life, there aren't enough Maintainers or Council members to run governance. Solution:

**Genesis Period** (first 6 months or until 50+ nodes with 30+ days uptime):
- The project founders' NodeIds are hardcoded as the initial Council
- Governance thresholds are relaxed (quorum 5% instead of 10-25%)
- Release approval threshold is 1 instead of 3
- This is explicitly temporary and expires automatically

**Transition**:
- When the network reaches the maturity threshold, a special GovernanceChange proposal is auto-generated to activate full governance
- The hardcoded Council is replaced by elected Council through the standard voting mechanism
- This transition is one-way and irreversible

```rust
pub fn is_genesis_period(network_stats: &NetworkStats) -> bool {
    let genesis_expiry = chrono::Utc.ymd(2027, 6, 1).and_hms(0, 0, 0); // Hard expiry
    let sufficient_nodes = network_stats.nodes_with_30d_uptime >= 50;

    chrono::Utc::now() < genesis_expiry && !sufficient_nodes
}
```

---

### Ports

| Port | Protocol | Purpose |
|---|---|---|
| 8800 | UDP (QUIC) | P2P node communication |
| 8800 | TCP (HTTP) | API server + Admin UI |

Both share the same port number but different protocols. Axum handles TCP, libp2p handles UDP.

### Environment Variables

All config values can be overridden with `SWARMLLM_` prefix:

```bash
SWARMLLM_NODE_LISTEN_PORT=9000
SWARMLLM_RESOURCES_MAX_GPU_VRAM_MB=6000
SWARMLLM_NETWORK_ENABLE_RELAY=false
SWARMLLM_LOGGING_LEVEL=debug
```

### Key File Paths

| Path | Purpose |
|---|---|
| `~/.swarmllm/config.toml` | User configuration |
| `~/.swarmllm/identity.key` | Ed25519 keypair |
| `~/.swarmllm/db/` | sled database |
| `~/.swarmllm/models/` | Downloaded model shards |
