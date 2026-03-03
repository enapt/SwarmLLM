use serde::{Deserialize, Serialize};
use std::fmt;

/// Current manifest schema version.
const MANIFEST_SCHEMA_VERSION: u32 = 2;

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
    /// Schema version (current: 2). Older versions are rejected.
    #[serde(default)]
    pub schema_version: u32,
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

impl ModelManifest {
    /// Validate the schema version. Rejects legacy (< 2) and future versions.
    pub fn validate_version(&self) -> Result<(), String> {
        if self.schema_version < MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "Manifest schema_version {} is outdated (current: {})",
                self.schema_version, MANIFEST_SCHEMA_VERSION
            ));
        } else if self.schema_version > MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "Manifest schema_version {} is newer than supported version {}",
                self.schema_version, MANIFEST_SCHEMA_VERSION
            ));
        }
        Ok(())
    }

    /// Set schema_version to the current version before serialization.
    pub fn stamp_version(&mut self) {
        self.schema_version = MANIFEST_SCHEMA_VERSION;
    }
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
    /// LLaVA: CLIP ViT vision encoder + Llama/Mistral LLM backbone.
    LLaVA {
        vision_config: VisionConfig,
    },
    /// Qwen2-VL: ViT vision encoder + Qwen2 LLM backbone.
    Qwen2VL {
        vision_config: VisionConfig,
    },
    /// GLM-4: partial RoPE, extreme GQA (2 KV heads), QKV biases.
    Glm4,
    /// Llama 4 Scout/Maverick: iRoPE (NoPE every 4th layer) + MoE.
    Llama4 {
        num_experts: u32,
        experts_per_token: u32,
    },
}

/// Vision encoder configuration for multimodal models.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Image size the vision encoder expects (e.g. 336 for CLIP-ViT-L/14@336px).
    pub image_size: u32,
    /// Patch size for the ViT (e.g. 14).
    pub patch_size: u32,
    /// Hidden dimension of the vision encoder.
    pub vision_hidden_size: u32,
    /// Number of transformer layers in the vision encoder.
    pub vision_num_layers: u32,
    /// Number of attention heads in the vision encoder.
    pub vision_num_heads: u32,
    /// Dimension of the multimodal projection (maps vision → LLM hidden dim).
    pub projection_dim: u32,
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
    /// Tensors contained in this shard, sorted by GGUF offset.
    #[serde(default)]
    pub tensors: Vec<ShardTensorEntry>,
}

/// One tensor's location within a shard file and in the original GGUF.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardTensorEntry {
    pub name: String,
    /// Absolute byte offset of this tensor in the virtual GGUF file.
    pub gguf_offset: u64,
    /// Byte offset within this shard file where the tensor data starts.
    pub shard_offset: u64,
    /// Size in bytes.
    pub size: u64,
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

impl From<crate::config::ContributionMode> for ContributionLevel {
    fn from(mode: crate::config::ContributionMode) -> Self {
        match mode {
            crate::config::ContributionMode::Minimal => Self::Minimal,
            crate::config::ContributionMode::Moderate => Self::Moderate,
            crate::config::ContributionMode::Maximum => Self::Maximum,
        }
    }
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
    /// Optional session ID for multi-turn KV-cache reuse.
    /// When provided, the router attempts to reuse cached KV state from a
    /// previous turn that shares the same prompt prefix, skipping redundant
    /// prefill and setting `start_pos` to the cached token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional LoRA adapter ID for per-request fine-tuned inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Decoded image data for VLM inference (not serialized over the wire — populated
    /// by the API layer after parsing OpenAI-format image_url content parts).
    #[serde(skip)]
    pub images: Vec<ImageData>,
}

/// Decoded image ready for vision encoder processing.
#[derive(Clone, Debug)]
pub struct ImageData {
    /// Raw RGB pixel data (H*W*3 bytes, row-major).
    pub rgb_bytes: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
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
    /// Tensor-parallel groups: each group of LAN peers processes the same layers
    /// in parallel, splitting attention heads and MLP dimensions across nodes.
    /// When present, the pipeline executor uses layer-by-layer AllReduce instead
    /// of sequential pipeline forwarding for these layer ranges.
    #[serde(default)]
    pub tp_groups: Vec<TensorParallelGroup>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSegment {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    pub layer_range: (u32, u32),
}

// ---- Tensor Parallelism ----

/// A group of LAN-local nodes that execute the same layers via tensor parallelism.
/// Instead of pipeline (sequential layers), each node computes a fraction of each
/// layer's computation (subset of attention heads + MLP columns) and results are
/// summed via AllReduce.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorParallelGroup {
    /// Ordered list of nodes in this TP group. Index = tp_rank.
    pub nodes: Vec<NodeId>,
    /// Which layers this group covers.
    pub layer_range: (u32, u32),
    /// Shard IDs needed (all nodes must have these shards).
    pub shard_ids: Vec<ShardId>,
}

impl TensorParallelGroup {
    /// Number of nodes in this tensor-parallel group.
    pub fn tp_size(&self) -> usize {
        self.nodes.len()
    }

    /// Get the TP rank for a given node (its index in the group).
    pub fn rank_of(&self, node_id: &NodeId) -> Option<usize> {
        self.nodes.iter().position(|n| n == node_id)
    }
}

/// Tensor-parallel metadata attached to a LayerForward for TP execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorParallelMeta {
    /// This node's rank within the TP group (0-indexed).
    pub tp_rank: u8,
    /// Total number of nodes in the TP group.
    pub tp_size: u8,
    /// Process only this single layer (layer-by-layer TP execution).
    pub single_layer: u32,
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

    // Governance
    ModelVote(ModelVote),

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

    // Encryption
    SealedInferenceRequest(crate::crypto::SealedPrompt),
    PeerKeyAdvertise {
        node_id: NodeId,
        x25519_public: [u8; 32],
        signature: Vec<u8>,
    },

    // Forward secrecy — ephemeral ECDH key exchange
    EphemeralKeyExchange(EphemeralKeyExchange),

    // Peer Exchange (PEX) — exchange known peer addresses on connection
    PeerExchangeRequest,
    PeerExchangeResponse(PeerExchangeResponse),
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
    /// The model this forward belongs to. Every LayerForward must identify
    /// its model so the receiving node loads the correct weights.
    pub model_id: ModelId,
    /// The layer range this forward should be processed over.
    /// The receiving node uses this to look up the correct cached
    /// SplitModel segment (keyed by model_id + layer_start + layer_end).
    pub layer_range: (u32, u32),
    /// Tensor-parallel metadata. When present, the receiving node should process
    /// only the specified single layer using its TP rank/size for head and MLP slicing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_meta: Option<TensorParallelMeta>,
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
}

/// Peer Exchange response — a list of known peer multiaddrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerExchangeResponse {
    /// Up to 20 known peer multiaddrs.
    pub peers: Vec<String>,
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

/// Bucketed credit balance gossip for network-wide percentile estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditGossip {
    pub node_id: NodeId,
    pub balance_bucket: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Ed25519 signature over (node_id || balance_bucket || timestamp_secs).
    /// Empty for legacy unsigned gossip from older nodes.
    #[serde(default)]
    pub signature: Vec<u8>,
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
    /// SEC-M18: Privacy-preserving blinded invitation broadcast.
    BlindedInvitation(crate::pool::types::BlindedPoolInvitation),
    Acceptance(crate::pool::types::PoolAcceptance),
    StateGossip(crate::pool::types::PoolState),
    CreditForward(crate::pool::types::PoolCreditForward),
    Removal(crate::pool::types::PoolRemoval),
    MemberLeft {
        pool_id: NodeId,
        node_id: NodeId,
        signature: Vec<u8>,
    },
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
    /// Active inference request count reported by this peer's last health ping/pong.
    #[serde(default)]
    pub active_request_count: u32,
    /// When this peer was first discovered (Unix timestamp).
    /// Used for leaderboard eligibility: peers must be at least `min_lifetime_days` old.
    #[serde(default)]
    pub first_seen: u64,
    /// Number of verified dual-signed credit transactions from this peer.
    /// Used for leaderboard eligibility: peers need `min_verified_transactions`.
    #[serde(default)]
    pub verified_transaction_count: u32,
    /// Whether this peer was discovered via mDNS (on the same LAN).
    /// LAN peers have ~1ms latency and are automatically preferred by the scheduler.
    #[serde(default)]
    pub is_lan_peer: bool,
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

// ---- Pruning Events ----
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

// ---- Hidden States API ----

/// Request body for POST /v1/internal/hidden-states.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateRequest {
    /// Model identifier (must match a locally loaded model).
    pub model: String,
    /// Input prompt to run through the model.
    pub prompt: String,
    /// Layer indices at which to capture hidden-state activations.
    pub return_layers: Vec<usize>,
    /// Maximum tokens to process (default: prompt length only, no generation).
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// Response body for POST /v1/internal/hidden-states.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateResponse {
    /// Map from layer index → tensor data.
    pub hidden_states: std::collections::HashMap<usize, HiddenStateTensor>,
    /// Number of prompt tokens processed.
    pub tokens_processed: usize,
}

/// A single hidden-state tensor, base64-encoded for JSON transport.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateTensor {
    /// Tensor dimensions, e.g. [1, seq_len, hidden_dim].
    pub shape: Vec<usize>,
    /// Data type string, e.g. "f32".
    pub dtype: String,
    /// Raw tensor bytes encoded as base64.
    pub data_base64: String,
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
            images: vec![],
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
            node_id: Some(NodeId([1u8; 32])),
            active_request_count: 5,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: SwarmMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            SwarmMessage::HealthPing {
                nonce,
                timestamp,
                node_id,
                active_request_count,
            } => {
                assert_eq!(nonce, 42);
                assert_eq!(timestamp, 1000);
                assert_eq!(node_id, Some(NodeId([1u8; 32])));
                assert_eq!(active_request_count, 5);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn health_ping_backward_compat() {
        // Old messages without active_request_count/node_id should deserialize with defaults
        let json = r#"{"HealthPing":{"nonce":1,"timestamp":2}}"#;
        let parsed: SwarmMessage = serde_json::from_str(json).unwrap();
        match parsed {
            SwarmMessage::HealthPing {
                active_request_count,
                node_id,
                ..
            } => {
                assert_eq!(active_request_count, 0);
                assert_eq!(node_id, None);
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

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            id: ModelId("test".into()),
            name: "Test".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        }
    }

    #[test]
    fn manifest_schema_version_current_is_valid() {
        let m = test_manifest();
        assert!(m.validate_version().is_ok());
    }

    #[test]
    fn manifest_schema_version_legacy_rejected() {
        let mut m = test_manifest();
        m.schema_version = 0;
        assert!(m.validate_version().is_err());
    }

    #[test]
    fn manifest_schema_version_future_rejected() {
        let mut m = test_manifest();
        m.schema_version = MANIFEST_SCHEMA_VERSION + 1;
        assert!(m.validate_version().is_err());
    }

    #[test]
    fn manifest_missing_version_defaults_to_zero() {
        // Simulate a legacy JSON manifest without schema_version
        let json = r#"{
            "id": "test",
            "name": "Test",
            "architecture": "Llama",
            "num_layers": 2,
            "num_params_billions": 0.001,
            "quantization": "Q4KM",
            "total_size_bytes": 1024,
            "shard_count": 1,
            "shards": [],
            "tokenizer_hash": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "manifest_hash": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "publisher": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "publish_date": "2024-01-01T00:00:00Z",
            "license": "MIT"
        }"#;
        let m: ModelManifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.schema_version, 0);
    }

    #[test]
    fn manifest_stamp_version_sets_current() {
        let mut m = test_manifest();
        m.schema_version = 0;
        m.stamp_version();
        assert_eq!(m.schema_version, MANIFEST_SCHEMA_VERSION);
    }

    #[test]
    fn hidden_state_request_serde_roundtrip() {
        let req = HiddenStateRequest {
            model: "test-model".into(),
            prompt: "Hello world".into(),
            return_layers: vec![0, 5, 10],
            max_tokens: Some(32),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HiddenStateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "test-model");
        assert_eq!(parsed.return_layers, vec![0, 5, 10]);
        assert_eq!(parsed.max_tokens, Some(32));
    }

    #[test]
    fn hidden_state_request_max_tokens_optional() {
        let json = r#"{"model":"m","prompt":"p","return_layers":[1]}"#;
        let parsed: HiddenStateRequest = serde_json::from_str(json).unwrap();
        assert!(parsed.max_tokens.is_none());
    }

    #[test]
    fn hidden_state_response_serde_roundtrip() {
        let mut hs = std::collections::HashMap::new();
        hs.insert(
            0,
            HiddenStateTensor {
                shape: vec![1, 4, 128],
                dtype: "f32".into(),
                data_base64: "AAAA".into(),
            },
        );
        let resp = HiddenStateResponse {
            hidden_states: hs,
            tokens_processed: 4,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HiddenStateResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tokens_processed, 4);
        assert!(parsed.hidden_states.contains_key(&0));
        assert_eq!(parsed.hidden_states[&0].shape, vec![1, 4, 128]);
    }
}
