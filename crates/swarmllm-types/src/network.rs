//! Wire-level P2P protocol messages: SwarmMessage, commands, rebalancing.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::credits::{CreditGossip, CreditTransaction};
use crate::identity::NicknameGossip;
use crate::ids::{ModelId, NodeId, ShardId};
use crate::inference::{
    InferenceError, InferenceRequest, LayerForward, LayerResult, PipelineAssignment,
    RemoteGenerateRequest, StreamingToken, TpAllReduceRequest, TpAllReduceResponse, TpRingChunk,
    VisionEncodeRequest, VisionEncodeResponse,
};
use crate::model::ModelManifest;
use crate::node::{NodeCapability, PeerExchangeResponse};
use crate::pool::PoolMessage;

/// A SwarmMessage wrapped with transport-authenticated sender identity.
/// The `sender` field is set by NetworkManager from the Noise-authenticated PeerId
/// (for request_response) or from the Ed25519-verified gossip signature.
/// This allows the dispatch handler to verify that message-internal sender claims
/// (e.g., ShardAnnounce.node_id) match the actual authenticated sender.
#[derive(Clone, Debug)]
pub struct AuthenticatedMessage {
    /// Transport-authenticated sender NodeId. None only for locally-generated messages.
    pub sender: Option<NodeId>,
    pub message: SwarmMessage,
}

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

    // Model manifest distribution
    ModelManifest(ModelManifest),

    // Credits
    CreditTransaction(CreditTransaction),

    // Health
    HealthPing {
        nonce: u64,
        timestamp: u64,
        /// Sender's node ID (for updating peer load in registry).
        #[serde(default)]
        node_id: Option<NodeId>,
        /// Number of active inference requests on the sender.
        #[serde(default)]
        active_request_count: u32,
    },
    HealthPong {
        nonce: u64,
        timestamp: u64,
        /// Sender's node ID (for updating peer load in registry).
        #[serde(default)]
        node_id: Option<NodeId>,
        /// Number of active inference requests on the sender.
        #[serde(default)]
        active_request_count: u32,
    },

    // Credits — gossip
    CreditGossip(CreditGossip),

    // Identity
    NicknameGossip(NicknameGossip),

    // Device Pools
    PoolMessage(PoolMessage),

    // Streaming — incremental token from the final pipeline node back to the originator
    StreamingToken(StreamingToken),

    // Shard download progress — broadcast so other nodes see downloads in progress
    ShardDownloadProgress(ShardDownloadProgress),

    // HuggingFace source gossip — tells peers where to download model shards from HF
    HfSourceGossip(HfSourceGossip),

    // Forward secrecy — ephemeral ECDH key exchange
    EphemeralKeyExchange(EphemeralKeyExchange),

    // Peer Exchange (PEX) — exchange known peer addresses on connection
    PeerExchangeRequest,
    PeerExchangeResponse(PeerExchangeResponse),

    // Vision — distributed mmproj encoding
    VisionEncodeRequest(VisionEncodeRequest),
    VisionEncodeResponse(VisionEncodeResponse),

    // Remote-generate fast path for single-segment distributed inference
    RemoteGenerateRequest(RemoteGenerateRequest),

    // Tensor Parallelism — AllReduce coordination
    TpAllReduceRequest(TpAllReduceRequest),
    TpAllReduceResponse(TpAllReduceResponse),
    /// Ring AllReduce: chunk sent between adjacent TP ranks.
    TpRingChunk(TpRingChunk),

    // Geo-aware regional gossip — Phase 18
    RegionShardSummary(RegionShardSummary),
    ModelDemandGossip(ModelDemandGossip),

    // Cross-node prefix-cache sharing (Item 8 Phase 1)
    PrefixCacheAnnounce(PrefixCacheAnnounce),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardAnnounce {
    pub node_id: NodeId,
    pub shards: Vec<ShardId>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// State of a shard download.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadState {
    Queued,
    Downloading,
    Verifying,
    Complete,
    Failed,
}

impl fmt::Display for DownloadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Downloading => write!(f, "downloading"),
            Self::Verifying => write!(f, "verifying"),
            Self::Complete => write!(f, "complete"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Shard download progress broadcast — lets other nodes see downloads in real time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardDownloadProgress {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    /// 0-100 percentage
    pub progress_pct: u32,
    pub state: DownloadState,
}

/// HuggingFace source gossip — tells peers where to download a model's shards from HF CDN.
/// Without this, only the node that originally downloaded from HF knows the repo_id/filename.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfSourceGossip {
    pub model_id: ModelId,
    pub repo_id: String,
    pub filename: String,
    pub publisher: NodeId,
    /// Filename of the mmproj GGUF on HuggingFace (for VLM models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_filename: Option<String>,
}

/// Regional shard summary — compact per-region model availability gossip.
/// Published periodically by each node for its region on `swarm/regions`.
/// O(regions * models) entries (~1KB each), enabling geo-aware auto-manage
/// decisions without iterating all holders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionShardSummary {
    /// ISO 3166-1 alpha-2 region code (e.g. "US", "DE").
    pub region: String,
    pub model_id: ModelId,
    /// Per-shard holder count in this region: (shard_index, holder_count).
    pub shard_counts: Vec<(u32, u32)>,
    /// Total nodes in this region (as seen by the publisher).
    pub region_node_count: u32,
    pub publisher: NodeId,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// Model demand gossip — EMA-smoothed request rate per model per region.
/// Published alongside RegionShardSummary to inform replication targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDemandGossip {
    pub model_id: ModelId,
    /// ISO 3166-1 alpha-2 region code.
    pub region: String,
    /// Exponentially decayed request rate (requests per 10-minute window).
    pub decayed_rate: f64,
    /// Raw request count in the latest window (before decay).
    pub window_requests: u64,
    pub publisher: NodeId,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// Ephemeral ECDH key exchange message for forward secrecy.
///
/// When a node wants to establish a forward-secret session, it sends this message
/// containing an ephemeral X25519 public key. The recipient generates its own
/// ephemeral key, derives the session key from the DH, and replies with its
/// ephemeral public key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EphemeralKeyExchange {
    /// Unique session identifier.
    pub session_id: uuid::Uuid,
    /// The sender's node identity.
    pub node_id: NodeId,
    /// Ephemeral X25519 public key (generated fresh for this session).
    pub ephemeral_pubkey: [u8; 32],
    /// Whether this is an initiation (true) or a response (false).
    pub is_initiator: bool,
}

/// Commands sent from daemon tasks to the NetworkManager.
///
/// `Broadcast` wraps a `SwarmMessage` for GossipSub. `SendTensor` and
/// `SendTensorResult` route tensor data through the unified request_response
/// codec with binary type-tag framing (WIRE_TAG_TENSOR = 0x01).
#[derive(Clone, Debug)]
pub enum NetworkCommand {
    /// Broadcast a message via GossipSub to all subscribers.
    Broadcast(SwarmMessage),
    /// Send a tensor forward pass to a specific peer via binary type-tag codec.
    SendTensor {
        target_peer_bytes: Vec<u8>,
        forward: LayerForward,
    },
    /// Send a tensor result back to a specific peer as a new request.
    SendTensorResult {
        target_peer_bytes: Vec<u8>,
        result: LayerResult,
    },
    /// Send a streaming token to a specific peer (originator of the request).
    SendStreamingToken {
        target_peer_bytes: Vec<u8>,
        token: StreamingToken,
    },
    /// Send a shard transfer request to a specific peer.
    SendShardRequest {
        target_peer_bytes: Vec<u8>,
        request: ShardRequest,
    },
    /// Send an AllReduce partial to the TP coordinator.
    SendAllReduceRequest {
        target_peer_bytes: Vec<u8>,
        request: TpAllReduceRequest,
    },
    /// Send an AllReduce response to a specific TP rank.
    SendAllReduceResponse {
        target_peer_bytes: Vec<u8>,
        response: TpAllReduceResponse,
    },
    /// Send a ring AllReduce chunk to the right neighbor in the ring.
    SendRingChunk {
        target_peer_bytes: Vec<u8>,
        chunk: TpRingChunk,
    },
    /// Send a SwarmMessage directly to a specific peer via request_response.
    ///
    /// `delivery_request_id`, when set, opts into ACK-timeout tracking: if
    /// the receiver doesn't ACK within a few seconds, the daemon closes
    /// `streaming_token_txs[delivery_request_id]` so the caller fails fast
    /// instead of waiting the full FIRST_TOKEN_TIMEOUT (120s). Used by the
    /// remote-generate fast path to detect rare libp2p rr silent-drops.
    SendDirectMessage {
        target_peer_bytes: Vec<u8>,
        message: SwarmMessage,
        delivery_request_id: Option<uuid::Uuid>,
    },
    /// Dial a multiaddr to connect to a new peer.
    DialAddress(String),
    /// S5: Register as a Kademlia provider for the given shards.
    /// Called on shard acquisition (download complete, startup scan).
    StartProviding(Vec<ShardId>),
    /// S5: Stop providing the given shards via Kademlia.
    /// Called on shard deletion.
    StopProviding(Vec<ShardId>),
    /// Item 8 Phase 2: send a cross-node prefix KV fetch to `target_peer`.
    /// The daemon caller installs a `tokio::sync::oneshot::Sender<Option<Vec<u8>>>`
    /// in `SharedState` keyed by `request_id` BEFORE sending this command;
    /// NetworkManager resolves it with the payload bytes on
    /// `SwarmResponse::PrefixKvData` arrival (or `None` on failure/timeout).
    SendPrefixKvFetch {
        target_peer_bytes: Vec<u8>,
        request_id: uuid::Uuid,
        model_id: ModelId,
        block_hash: [u8; 32],
    },
    /// Item 8 Phase 2b: deliver an inbound-served PrefixKv response. The
    /// serving task extracts a snapshot from the local worker and posts
    /// this command; NetworkManager looks up the stored ResponseChannel
    /// by `ticket` and emits the reply. `payload=None` produces a miss
    /// reply; `Some(bytes)` produces a hit reply.
    DeliverPrefixKvResponse {
        ticket: uuid::Uuid,
        request_id: uuid::Uuid,
        payload: Option<Vec<u8>>,
    },
    /// Mirror of `DeliverPrefixKvResponse` for shard-transfer serving. The
    /// serving task does the disk read + bandwidth-throttle sleep off the
    /// swarm event loop and posts this command with the produced bytes.
    /// NetworkManager looks up the stored ResponseChannel by `ticket`,
    /// constructs `SwarmResponse::ShardData { data, total_size }`, emits
    /// the reply, and bumps the `shard_bytes_served` atomic for credit
    /// seeding.
    DeliverShardResponse {
        ticket: uuid::Uuid,
        data: Vec<u8>,
        total_size: u64,
    },
}

/// Events that trigger shard rebalancing.
#[derive(Clone, Debug)]
pub enum RebalanceEvent {
    PeerLeft(NodeId),
}

// ---- Shard transfer protocol ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardRequest {
    pub shard_id: ShardId,
    pub chunk_offset: u64,
    pub chunk_size: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardResponse {
    pub data: Vec<u8>,
    pub total_size: u64,
}

/// One block in a `PrefixCacheAnnounce`. The hash is a chained BLAKE3
/// rollup over the token IDs of the prompt prefix that produced this KV
/// snapshot, with `token_count` total tokens covered.
///
/// Hash chain: `block_hash[0] = blake3(u32_le(tokens[0..B]))`,
/// `block_hash[i] = blake3(block_hash[i-1] || u32_le(tokens[i*B..(i+1)*B]))`.
/// Two prompts that share the first `i` blocks (under the same block size
/// `B`) have identical `block_hash[0..i]`, so longest-prefix match by hash
/// works without ever transmitting tokens.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefixBlockEntry {
    /// Chained BLAKE3 hash uniquely identifying this token-prefix block.
    pub block_hash: [u8; 32],
    /// Number of prompt tokens covered by the prefix this block ends at
    /// (i.e. the strict prefix length the snapshot represents).
    pub token_count: u32,
}

/// Cross-node prefix cache announcement (Item 8 Phase 1).
///
/// Each worker broadcasts the BLAKE3 hashes of cached prompt-prefix blocks
/// it currently holds for a given model. Receivers build a per-model
/// `block_hash → {peer_id}` index that lets them, in Phase 2+, fetch a
/// pre-computed KV snapshot from the announcing peer instead of re-running
/// prefill.
///
/// Announcements supersede prior announcements from the same `(node_id,
/// model_id)` pair — the receiver replaces all prior entries for that pair
/// with the new `blocks` set, so the index reflects the announcer's
/// current cache state (including evictions).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixCacheAnnounce {
    pub node_id: NodeId,
    pub model_id: ModelId,
    pub blocks: Vec<PrefixBlockEntry>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Event emitted when auto-manage prunes (deletes) an over-replicated shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PruneEvent {
    pub model_id: ModelId,
    pub model_name: String,
    pub shard_index: u32,
    pub reason: String,
    pub freed_bytes: u64,
    pub remaining_local_shards: u32,
    pub holder_count_before: usize,
    pub holder_count_after: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
