use serde::{Deserialize, Serialize};
use std::fmt;

// ---- Identity ----
/// Wrapper around Ed25519 public key. This IS the node's identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8]))
    }
}

// ---- Models ----
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: ModelId,
    pub name: String,
    pub architecture: ModelArchitecture,
    pub num_layers: u32,
    pub num_params_billions: f32,
    pub quantization: Quantization,
    pub total_size_bytes: u64,
    pub shard_count: u32,
    pub shards: Vec<ShardInfo>,
    pub tokenizer_hash: Blake3Hash,
    pub manifest_hash: Blake3Hash,
    pub publisher: NodeId,
    pub publish_date: chrono::DateTime<chrono::Utc>,
    pub license: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Mixtral {
        num_experts: u32,
        experts_per_token: u32,
    },
    Qwen2,
    DeepSeek {
        num_experts: u32,
        experts_per_token: u32,
    },
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

// ---- Shards ----
pub type Blake3Hash = [u8; 32];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub index: u32,
    pub layer_range: (u32, u32),
    pub size_bytes: u64,
    pub hash: Blake3Hash,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShardId {
    pub model_id: ModelId,
    pub index: u32,
}

// ---- Node Capabilities ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub gpu: Option<GpuInfo>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub disk_available_mb: u64,
    pub bandwidth_mbps: f32,
    pub hosted_shards: Vec<ShardId>,
    pub max_contribution: ContributionLevel,
    pub uptime_seconds: u64,
    pub version: String,
    /// Voluntary ISO 3166-1 alpha-2 country code (e.g. "US", "DE").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,
    Moderate,
    Maximum,
}

// ---- Inference ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: uuid::Uuid,
    pub model_id: ModelId,
    pub messages: Vec<ChatMessage>,
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
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
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

// ---- Credits ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: NodeId,
    pub balance: i64,
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
    pub signature_from: Vec<u8>,
    pub signature_to: Vec<u8>,
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

// ---- Pipeline ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineAssignment {
    pub request_id: uuid::Uuid,
    pub segments: Vec<PipelineSegment>,
    pub standbys: Vec<PipelineSegment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSegment {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    pub layer_range: (u32, u32),
}

// ---- Network Messages ----
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
    },
    HealthPong {
        nonce: u64,
        timestamp: u64,
    },

    // Credits — gossip
    CreditGossip(CreditGossip),

    // Governance
    ModelVote(ModelVote),

    // Identity
    NicknameGossip(NicknameGossip),

    // Device Pools
    PoolMessage(PoolMessage),

    // Streaming — incremental token from the final pipeline node back to the originator
    StreamingToken(StreamingToken),

    // Encryption
    SealedInferenceRequest(crate::crypto::SealedPrompt),
    PeerKeyAdvertise {
        node_id: NodeId,
        x25519_public: [u8; 32],
        signature: Vec<u8>,
    },
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
    pub sequence_num: u32,
    /// The absolute position in the sequence for RoPE and KV-cache indexing.
    /// For the prefill pass (seq_num=0) this is 0; for subsequent tokens it equals
    /// the cumulative number of tokens already processed.
    #[serde(default)]
    pub index_pos: u32,
    pub activations: Vec<u8>,
    pub format: TensorFormat,
    /// Populated locally after receiving from the network — not serialized over the wire.
    /// Contains the libp2p PeerId bytes of the sender so we can route the result back.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
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
    pub finish_reason: Option<NetworkFinishReason>,
    /// Intermediate hidden-state activations (for non-final pipeline segments).
    /// Empty for the final segment (which returns token_ids instead).
    #[serde(default)]
    pub activations: Vec<u8>,
}

/// A single token streamed back from the final pipeline node to the originator.
/// Used for SSE streaming in distributed inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingToken {
    pub request_id: uuid::Uuid,
    pub token_id: u32,
    pub finish_reason: Option<NetworkFinishReason>,
}

/// Finish reason for network protocol messages (distinct from inference::executor::FinishReason).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NetworkFinishReason {
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

/// Bucketed credit balance gossip for network-wide percentile estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditGossip {
    pub node_id: NodeId,
    pub balance_bucket: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Nickname announcement gossiped across the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicknameGossip {
    pub record: crate::identity::nickname::NicknameRecord,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVote {
    pub voter: NodeId,
    pub model_manifest_hash: Blake3Hash,
    pub vote: bool,
    pub weight: u64,
    pub signature: Vec<u8>,
}

// ---- Pool Messages ----
/// Messages related to device pool management, sent over GossipSub.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolMessage {
    Invitation(crate::pool::types::PoolInvitation),
    Acceptance(crate::pool::types::PoolAcceptance),
    StateGossip(crate::pool::types::PoolState),
    CreditForward(crate::pool::types::PoolCreditForward),
    Removal(crate::pool::types::PoolRemoval),
    MemberLeft { pool_id: NodeId, node_id: NodeId },
}

// ---- Network Commands ----
/// Commands sent from daemon tasks to the NetworkManager.
///
/// `Broadcast` wraps a `SwarmMessage` for GossipSub. `SendTensor` and
/// `SendTensorResult` route tensor data through the Cap'n Proto
/// request_response protocol for zero-copy efficiency.
#[derive(Clone, Debug)]
pub enum NetworkCommand {
    /// Broadcast a message via GossipSub to all subscribers.
    Broadcast(SwarmMessage),
    /// Send a tensor forward pass to a specific peer via Cap'n Proto.
    SendTensor {
        target_peer_bytes: Vec<u8>,
        forward: LayerForward,
    },
    /// Send a tensor result back to a specific peer via Cap'n Proto.
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
}

// ---- Rebalancing ----
/// Events that trigger shard rebalancing.
#[derive(Clone, Debug)]
pub enum RebalanceEvent {
    PeerJoined(NodeId),
    PeerLeft(NodeId),
    DiskPressure { available_mb: u64 },
    ManualTrigger,
}

// ---- Peer State ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub addresses: Vec<String>,
    pub capability: Option<NodeCapability>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub latency_ms: Option<u32>,
    pub trust_score: f32,
    /// Raw libp2p PeerId bytes for directed request_response messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_id_bytes: Option<Vec<u8>>,
}

// ---- Node Stats ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeStats {
    pub peers_connected: u32,
    pub requests_served: u64,
    pub requests_made: u64,
    /// Layer forwards processed for other nodes in distributed inference.
    pub forwards_served: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub uptime_start: chrono::DateTime<chrono::Utc>,
    /// NAT status detected by AutoNAT ("Public", "Private", "Unknown").
    #[serde(default)]
    pub nat_status: Option<String>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            peers_connected: 0,
            requests_served: 0,
            requests_made: 0,
            forwards_served: 0,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            uptime_start: chrono::Utc::now(),
            nat_status: None,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_display_shows_short_hex() {
        let id = NodeId([0xab; 32]);
        assert_eq!(format!("{id}"), "abababababababab");
    }

    #[test]
    fn model_id_display() {
        let id = ModelId("llama3-70b-q4km".into());
        assert_eq!(format!("{id}"), "llama3-70b-q4km");
    }

    #[test]
    fn sampling_params_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.top_k, 40);
    }

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: Role::User,
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "hello");
    }

    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn swarm_message_serde_roundtrip() {
        let msg = SwarmMessage::HealthPing {
            nonce: 42,
            timestamp: 1000,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: SwarmMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            SwarmMessage::HealthPing { nonce, timestamp } => {
                assert_eq!(nonce, 42);
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn shard_id_equality() {
        let a = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let b = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        assert_eq!(a, b);
    }
}
