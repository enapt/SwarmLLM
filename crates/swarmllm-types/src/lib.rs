use serde::{Deserialize, Serialize};
use std::fmt;

// ---- Config types (moved from config.rs for cross-crate use) ----

/// Contribution mode from node config — maps to ContributionLevel.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContributionMode {
    #[default]
    Minimal,
    Moderate,
    Maximum,
}

// ---- Crypto types (moved from crypto/pipeline_seal.rs for cross-crate use) ----

/// A sealed (encrypted) inference prompt for E2E privacy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedPrompt {
    pub request_id: uuid::Uuid,
    /// ChaCha20-Poly1305 encrypted prompt bytes.
    pub encrypted_prompt: Vec<u8>,
    /// 12-byte nonce used for prompt encryption.
    pub nonce: [u8; 12],
    /// Ephemeral X25519 public key (32 bytes) of the sealer.
    pub ephemeral_pub: [u8; 32],
    /// The request_key encrypted for the first pipeline node's X25519 key.
    pub key_envelope: Vec<u8>,
}

// ---- Identity types (moved from identity/nickname.rs for cross-crate use) ----

/// A signed nickname record for a node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicknameRecord {
    pub node_id: NodeId,
    pub nickname: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Ed25519 signature over the signing payload.
    pub signature: Vec<u8>,
}

// ---- Pool types (moved from pool/types.rs for cross-crate use) ----

/// Pool identity is the owner's NodeId.
pub type PoolId = NodeId;

/// A single pool member record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolMembership {
    pub node_id: NodeId,
    pub credits_contributed: i64,
    pub joined_at: chrono::DateTime<chrono::Utc>,
    pub acceptance_signature: Vec<u8>,
    pub invitation_id: uuid::Uuid,
    /// User-chosen device nickname (e.g., "Gaming PC", "Laptop")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    /// Last time this device was seen on the network (updated via health pings)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen: Option<chrono::DateTime<chrono::Utc>>,
    /// Whether the device is currently online (derived from last_seen < 2 min)
    #[serde(default)]
    pub online: bool,
    /// Per-device stats reported via pool state gossip
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_stats: Option<PoolDeviceStats>,
    /// Contribution level set by the pool owner (0-100%).
    /// Controls how much of this device's resources are dedicated to the network.
    /// 100 = full contribution (default), 50 = half speed/bandwidth, 0 = paused.
    #[serde(default = "default_contribution_level")]
    pub contribution_level: u8,
}

fn default_contribution_level() -> u8 {
    100
}

/// Per-device performance stats within a pool.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PoolDeviceStats {
    /// Forwards served (inference segments processed)
    pub forwards_served: u64,
    /// Total inference requests served
    pub requests_served: u64,
    /// Number of model shards hosted
    pub shards_hosted: u32,
    /// GPU VRAM in MB (0 if CPU-only)
    pub vram_mb: u64,
    /// RAM in MB
    pub ram_mb: u64,
    /// Node uptime in seconds
    pub uptime_secs: u64,
    /// Model IDs currently loaded/hosted
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models_hosted: Vec<String>,
}

/// State of a device pool — owner + list of members.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolState {
    pub pool_id: PoolId,
    pub name: String,
    pub members: Vec<PoolMembership>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub total_lifetime_credits: i64,
    /// Credit split: percentage (0-100) of earnings kept by the member.
    /// The remainder is forwarded to the owner. Default: 0 (all to owner).
    #[serde(default)]
    pub member_credit_split_pct: u8,
    /// Shard pins: owner assigns specific models/shards to specific devices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_pins: Vec<ShardPin>,
}

/// A shard pinning assignment: a model (or specific shards) pinned to a target device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardPin {
    /// Model ID to pin.
    pub model_id: String,
    /// Specific shard indices to pin, or empty for all shards.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_indices: Vec<u32>,
    /// Target device NodeId.
    pub target_node_id: NodeId,
}

impl ShardPin {
    /// Check whether this pin applies to a given model/node/shard combination.
    /// Empty `shard_indices` means "all shards".
    pub fn matches(&self, model_id: &str, node_id: &NodeId, shard_index: u32) -> bool {
        self.model_id == model_id
            && self.target_node_id == *node_id
            && (self.shard_indices.is_empty() || self.shard_indices.contains(&shard_index))
    }

    /// Check whether this pin applies to a given model and shard (any node).
    pub fn matches_shard(&self, model_id: &str, shard_index: u32) -> bool {
        self.model_id == model_id
            && (self.shard_indices.is_empty() || self.shard_indices.contains(&shard_index))
    }
}

/// Invitation to join a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Privacy-preserving blinded invitation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindedPoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    /// H("pool_invitee_commit_v1" || invitee_node_id || invitation_id)
    pub invitee_commitment: [u8; 32],
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Acceptance of a pool invitation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolAcceptance {
    pub invitation_id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub invitee_signature: Vec<u8>,
    pub accepted_at: chrono::DateTime<chrono::Utc>,
}

/// Credit forwarding within a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolCreditForward {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    pub from_node_id: NodeId,
    pub to_node_id: NodeId,
    pub amount: i64,
    pub member_signature: Vec<u8>,
    pub owner_signature: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Removal of a member from a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolRemoval {
    pub pool_id: PoolId,
    pub removed_node_id: NodeId,
    pub owner_signature: Vec<u8>,
    pub removed_at: chrono::DateTime<chrono::Utc>,
    /// Unique ID to prevent replay attacks (new field, defaults to nil for old messages)
    #[serde(default = "uuid::Uuid::nil")]
    pub removal_id: uuid::Uuid,
}

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
        // SEC: Sanitize control characters to prevent log injection via network-supplied model IDs.
        // Newlines, carriage returns, and null bytes are replaced with their escape sequences.
        for ch in self.0.chars() {
            match ch {
                '\n' => write!(f, "\\n")?,
                '\r' => write!(f, "\\r")?,
                '\0' => write!(f, "\\0")?,
                c if c.is_control() => write!(f, "\\x{:02x}", c as u32)?,
                c => write!(f, "{c}")?,
            }
        }
        Ok(())
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
    /// Vision encoder (mmproj) metadata. Present only for VLM models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<MmprojInfo>,
}

/// Metadata for a VLM vision encoder (mmproj GGUF file).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MmprojInfo {
    pub size_bytes: u64,
    pub hash: Blake3Hash,
    /// HuggingFace filename for the mmproj GGUF (e.g. "llava-v1.5-7b-mmproj-model-f16.gguf").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_filename: Option<String>,
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
    /// Qwen 3.5 dense: hybrid attention + Gated Delta Network (SSM) layers.
    Qwen35,
    /// Qwen 3.5 MoE: hybrid attention + SSM with mixture-of-experts FFN.
    Qwen35Moe {
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

/// Sentinel shard index for mmproj (vision encoder) files.
/// Using u32::MAX avoids collisions with text model shard indices (0..N).
pub const MMPROJ_SHARD_INDEX: u32 = u32::MAX;

impl ShardId {
    /// Whether this shard ID refers to a mmproj (vision encoder) file.
    pub fn is_mmproj(&self) -> bool {
        self.index == MMPROJ_SHARD_INDEX
    }

    /// Human-friendly display label for a shard index (1-based, or "mmproj").
    pub fn display_index(index: u32) -> String {
        if index == MMPROJ_SHARD_INDEX {
            "mmproj".to_string()
        } else {
            format!("shard {}", index + 1)
        }
    }

    /// Safe 1-based index for display, returning "mmproj" for MMPROJ_SHARD_INDEX.
    /// Use this instead of `index + 1` to avoid u32 overflow on mmproj shards.
    pub fn display_index_short(index: u32) -> String {
        if index == MMPROJ_SHARD_INDEX {
            "mmproj".to_string()
        } else {
            format!("{}", index + 1)
        }
    }

    /// Create a ShardId for the mmproj of a given model.
    pub fn mmproj_for(model_id: ModelId) -> Self {
        Self {
            model_id,
            index: MMPROJ_SHARD_INDEX,
        }
    }
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
    /// Estimated tokens/s for a 7B Q4 model based on GPU memory bandwidth.
    /// Used by the scheduler as a speed tie-breaker.
    #[serde(default)]
    pub est_tokens_per_sec_7b: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
    /// Memory bandwidth in GB/s, looked up from GPU name.
    #[serde(default)]
    pub memory_bandwidth_gbps: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,
    Moderate,
    Maximum,
}

impl From<ContributionMode> for ContributionLevel {
    fn from(mode: ContributionMode) -> Self {
        match mode {
            ContributionMode::Minimal => Self::Minimal,
            ContributionMode::Moderate => Self::Moderate,
            ContributionMode::Maximum => Self::Maximum,
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

impl InferenceRequest {
    /// Create an inference request originating from the local API (not a network peer).
    pub fn local(
        model_id: ModelId,
        messages: Vec<ChatMessage>,
        sampling_params: SamplingParams,
        stream: bool,
        session_id: Option<String>,
        lora_adapter: Option<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            model_id,
            messages,
            sampling_params,
            stream,
            requester: NodeId([0u8; 32]),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
            session_id,
            lora_adapter,
        }
    }
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
    Tool,
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
    /// Whether to return log probabilities for sampled tokens.
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top log probabilities to return per token (0-20).
    #[serde(default)]
    pub top_logprobs: u32,
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
            logprobs: false,
            top_logprobs: 0,
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
    ShardSeeding { shard_id: ShardId, bytes: u64 },
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

/// Phase of a tensor-parallel layer computation.
///
/// Each transformer layer is split into two IPC calls so that FFN norm
/// is applied to the AllReduced post-attention tensor (not the partial).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum TpPhase {
    /// Full layer (non-TP path, used when tp_meta is None).
    #[default]
    Full,
    /// Phase 1: attn_norm → attention (head-sliced) → return partial.
    /// Pipeline AllReduces, adds residual, then sends FfnOnly.
    AttnOnly,
    /// Phase 2: ffn_norm → FFN (column-sliced) → return partial.
    /// Pipeline AllReduces, adds residual to get full layer output.
    FfnOnly,
    /// Embedding only: tokenize + embed → return hidden states (no layer processing).
    /// Used by the coordinator to get the residual for the first layer.
    EmbedOnly,
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
    /// Which phase of the layer to compute (default: Full for backwards compat).
    #[serde(default)]
    pub phase: TpPhase,
}

// ---- Network Messages ----

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

    // Tensor Parallelism — AllReduce coordination
    TpAllReduceRequest(TpAllReduceRequest),
    TpAllReduceResponse(TpAllReduceResponse),
    /// Ring AllReduce: chunk sent between adjacent TP ranks.
    TpRingChunk(TpRingChunk),

    // Geo-aware regional gossip — Phase 18
    RegionShardSummary(RegionShardSummary),
    ModelDemandGossip(ModelDemandGossip),
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
    /// Pre-computed vision embeddings (zstd-compressed FP16).
    /// Attached to the first LayerForward (seq_num==0) when the request contains images.
    /// Shape: (num_image_tokens, hidden_dim) — typically (577, 4096) for LLaVA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_embeddings: Option<Vec<u8>>,
    /// Populated locally after receiving from the network — not serialized over the wire.
    /// Contains the libp2p PeerId bytes of the sender so we can route the result back.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
    /// Pipeline sealing: Ed25519 public key of the requesting node (32 bytes).
    /// Present on distributed pipeline forwards so the final segment can seal
    /// the result (token IDs) for the requester's X25519 key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requester_node_id: Option<[u8; 32]>,
    /// Local embedding privacy: when true, `activations` contains pre-embedded
    /// hidden-state tensors (serialized via tensor_to_bytes) instead of raw token
    /// IDs or prompt text. The receiving node skips its embedding lookup.
    #[serde(default)]
    pub pre_embedded: bool,
    /// LoRA adapter ID to apply during inference. When set, the worker loads the
    /// adapter from the data_dir and applies its low-rank deltas per-layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorFormat {
    FP16,
    FP32,
    INT8,
}

/// Request to encode an image into vision embeddings on a remote node that holds mmproj.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionEncodeRequest {
    pub request_id: uuid::Uuid,
    pub model_id: ModelId,
    /// JPEG-compressed image bytes (resized to vision encoder input, typically 336x336).
    pub image_data: Vec<u8>,
    /// Populated locally after receiving from the network — not serialized over the wire.
    /// Contains the libp2p PeerId bytes of the sender so we can route the response back.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
}

/// Response carrying pre-computed vision embeddings from the mmproj holder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionEncodeResponse {
    pub request_id: uuid::Uuid,
    /// Zstd-compressed FP16 tensor: (num_image_tokens, hidden_dim).
    pub embeddings: Vec<u8>,
    pub num_tokens: u32,
    pub hidden_dim: u32,
}

/// AllReduce request: a node sends its partial tensor to be reduced.
///
/// Used in tensor-parallel inference where multiple nodes compute partial
/// results for the same layer and need to sum them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TpAllReduceRequest {
    /// Inference request this belongs to.
    pub request_id: uuid::Uuid,
    /// Which layer is being reduced.
    pub layer_idx: u32,
    /// This node's TP rank within the group.
    pub tp_rank: u32,
    /// Total number of TP ranks in the group.
    pub tp_size: u32,
    /// Zstd-compressed partial tensor data (FP32).
    pub partial_data: Vec<u8>,
    /// Shape of the partial tensor: [batch_size, seq_len, hidden_dim].
    pub shape: Vec<u32>,
    /// Reduction operation (currently only Sum).
    pub op: AllReduceOp,
    /// Populated locally after receiving from the network — not serialized over the wire.
    /// Contains the libp2p PeerId bytes of the sender so we can route the response back.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
}

/// AllReduce response: the reduced (summed) tensor from the coordinator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TpAllReduceResponse {
    pub request_id: uuid::Uuid,
    pub layer_idx: u32,
    /// Zstd-compressed reduced tensor (sum of all partials).
    pub reduced_data: Vec<u8>,
    pub shape: Vec<u32>,
}

/// Reduction operation for tensor parallelism.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AllReduceOp {
    /// Element-wise sum of partial tensors across TP ranks.
    Sum,
}

/// Ring AllReduce chunk message: sent between adjacent TP ranks during
/// scatter-reduce and allgather phases.
///
/// Each step of the ring algorithm sends one chunk of the tensor to the
/// right neighbor and receives one chunk from the left neighbor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TpRingChunk {
    /// Inference request this belongs to.
    pub request_id: uuid::Uuid,
    /// Which layer is being reduced.
    pub layer_idx: u32,
    /// Ring step index (0..2*(tp_size-1)).
    pub step: u32,
    /// Chunk index within the partitioned tensor.
    pub chunk_idx: u32,
    /// Phase: scatter-reduce (accumulate) or allgather (broadcast).
    pub is_allgather: bool,
    /// Zstd-compressed chunk data (FP32).
    pub chunk_data: Vec<u8>,
    /// Total number of chunks (= tp_size).
    pub num_chunks: u32,
    /// Populated locally after receiving from the network.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
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
    /// Pipeline sealing: sealed token IDs (SealedPrompt JSON).
    /// When present, `token_ids` is empty and the requester must unseal this
    /// with their X25519 secret to recover the real token IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_token_ids: Option<Vec<u8>>,
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
    /// Required — unsigned gossip is rejected.
    #[serde(default)]
    pub signature: Vec<u8>,
}

/// Nickname announcement gossiped across the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicknameGossip {
    pub record: NicknameRecord,
}

// ---- Model Trust ----

/// Trust level for a model in the auto-manage system.
///
/// Models progress through trust levels based on real usage. Auto-manage
/// only downloads shards for models that are `DemandVerified` or higher
/// (or explicitly `Pinned` by the user). This prevents trash models from
/// propagating across the network when auto-manage is enabled.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModelTrustLevel {
    /// Seen via gossip but never used or approved. Auto-manage ignores.
    Discovered = 0,
    /// User explicitly downloaded or approved this model for their node.
    Pinned = 1,
    /// Has received real inference requests (>= threshold). Auto-manage propagates.
    DemandVerified = 2,
    /// Multiple independent nodes (>= 3) actively serving it. High priority.
    NetworkPopular = 3,
}

impl std::fmt::Display for ModelTrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Discovered => write!(f, "discovered"),
            Self::Pinned => write!(f, "pinned"),
            Self::DemandVerified => write!(f, "demand_verified"),
            Self::NetworkPopular => write!(f, "network_popular"),
        }
    }
}

/// Per-model trust metadata for auto-manage gating and UI display.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelTrustInfo {
    pub trust_level: ModelTrustLevel,
    pub first_seen: chrono::DateTime<chrono::Utc>,
    pub total_requests: u64,
    /// Whether the user explicitly pinned (approved) this model.
    pub pinned_by_user: bool,
    pub last_request_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ModelTrustInfo {
    pub fn new_discovered() -> Self {
        Self {
            trust_level: ModelTrustLevel::Discovered,
            first_seen: chrono::Utc::now(),
            total_requests: 0,
            pinned_by_user: false,
            last_request_at: None,
        }
    }

    pub fn new_pinned() -> Self {
        Self {
            trust_level: ModelTrustLevel::Pinned,
            first_seen: chrono::Utc::now(),
            total_requests: 0,
            pinned_by_user: true,
            last_request_at: None,
        }
    }

    /// Record an inference request. Promotes to DemandVerified after threshold.
    pub fn record_request(&mut self) {
        self.total_requests += 1;
        self.last_request_at = Some(chrono::Utc::now());
        // Promote after 3 real requests (prevents single accidental request from promoting)
        if self.total_requests >= 3 && self.trust_level < ModelTrustLevel::DemandVerified {
            self.trust_level = ModelTrustLevel::DemandVerified;
        }
    }

    /// Check if this model should decay due to inactivity (7 days without requests).
    /// Pinned models never decay. NetworkPopular decays to DemandVerified.
    pub fn maybe_decay(&mut self) {
        if self.pinned_by_user {
            return;
        }
        let cutoff = chrono::Utc::now() - chrono::Duration::days(7);
        let inactive = self
            .last_request_at
            .map(|t| t < cutoff)
            .unwrap_or(self.first_seen < cutoff);
        if !inactive {
            return;
        }
        match self.trust_level {
            ModelTrustLevel::NetworkPopular => {
                self.trust_level = ModelTrustLevel::DemandVerified;
            }
            ModelTrustLevel::DemandVerified => {
                self.trust_level = ModelTrustLevel::Discovered;
            }
            _ => {}
        }
    }
}

// ---- Pool Messages ----
/// Messages related to device pool management, sent over GossipSub.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolMessage {
    /// SEC-M18: Privacy-preserving blinded invitation broadcast.
    BlindedInvitation(BlindedPoolInvitation),
    Acceptance(PoolAcceptance),
    StateGossip(PoolState),
    CreditForward(PoolCreditForward),
    Removal(PoolRemoval),
    MemberLeft {
        pool_id: NodeId,
        node_id: NodeId,
        /// Unix timestamp (seconds) when the leave notice was created.
        /// Receivers MUST reject notices more than ~5 minutes out of range,
        /// and dedup on the UUID below to prevent replay.
        #[serde(default)]
        left_at: i64,
        #[serde(default)]
        nonce: uuid::Uuid,
        signature: Vec<u8>,
    },
    /// Join request from a device that has an invite code.
    /// The code_hash is BLAKE3(code) — the code itself is never sent over the network.
    JoinRequest {
        code_hash: [u8; 32],
        requester: NodeId,
        /// Ed25519 signature over BLAKE3("pool_join_request_v1" || code_hash || requester)
        signature: Vec<u8>,
    },
    /// Periodic stats + nickname report from a pool member to the leader.
    DeviceStatsReport {
        pool_id: NodeId,
        node_id: NodeId,
        device_name: Option<String>,
        stats: PoolDeviceStats,
    },
}

// ---- Network Commands ----
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
    SendDirectMessage {
        target_peer_bytes: Vec<u8>,
        message: SwarmMessage,
    },
    /// Dial a multiaddr to connect to a new peer.
    DialAddress(String),
    /// S5: Register as a Kademlia provider for the given shards.
    /// Called on shard acquisition (download complete, startup scan).
    StartProviding(Vec<ShardId>),
    /// S5: Stop providing the given shards via Kademlia.
    /// Called on shard deletion.
    StopProviding(Vec<ShardId>),
}

// ---- Rebalancing ----
/// Events that trigger shard rebalancing.
#[derive(Clone, Debug)]
pub enum RebalanceEvent {
    PeerLeft(NodeId),
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
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn role_tool_deserializes() {
        let role: Role = serde_json::from_str("\"tool\"").unwrap();
        assert!(matches!(role, Role::Tool));
    }

    #[test]
    fn sampling_params_logprobs_defaults() {
        let params = SamplingParams::default();
        assert!(!params.logprobs);
        assert_eq!(params.top_logprobs, 0);
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
            mmproj: None,
        }
    }

    #[test]
    fn trust_level_ordering() {
        assert!(ModelTrustLevel::Discovered < ModelTrustLevel::Pinned);
        assert!(ModelTrustLevel::Pinned < ModelTrustLevel::DemandVerified);
        assert!(ModelTrustLevel::DemandVerified < ModelTrustLevel::NetworkPopular);
    }

    #[test]
    fn trust_info_record_request_promotes() {
        let mut info = ModelTrustInfo::new_discovered();
        assert_eq!(info.trust_level, ModelTrustLevel::Discovered);
        info.record_request();
        info.record_request();
        assert_eq!(info.trust_level, ModelTrustLevel::Discovered); // <3
        info.record_request();
        assert_eq!(info.trust_level, ModelTrustLevel::DemandVerified); // >=3
    }

    #[test]
    fn trust_info_pinned_never_decays() {
        let mut info = ModelTrustInfo::new_pinned();
        info.trust_level = ModelTrustLevel::DemandVerified;
        // Simulate old last_request
        info.last_request_at = Some(chrono::Utc::now() - chrono::Duration::days(30));
        info.maybe_decay();
        // Pinned models never decay
        assert_eq!(info.trust_level, ModelTrustLevel::DemandVerified);
    }

    #[test]
    fn trust_info_unpinned_decays_after_7_days() {
        let mut info = ModelTrustInfo::new_discovered();
        info.trust_level = ModelTrustLevel::DemandVerified;
        info.last_request_at = Some(chrono::Utc::now() - chrono::Duration::days(8));
        info.maybe_decay();
        assert_eq!(info.trust_level, ModelTrustLevel::Discovered);
    }

    #[test]
    fn trust_level_display() {
        assert_eq!(ModelTrustLevel::Discovered.to_string(), "discovered");
        assert_eq!(ModelTrustLevel::Pinned.to_string(), "pinned");
        assert_eq!(
            ModelTrustLevel::DemandVerified.to_string(),
            "demand_verified"
        );
        assert_eq!(
            ModelTrustLevel::NetworkPopular.to_string(),
            "network_popular"
        );
    }
}
