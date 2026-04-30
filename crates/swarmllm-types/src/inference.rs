//! Inference request, pipeline/TP types, and layer forward/result messages.

use serde::{Deserialize, Serialize};

use crate::credits::PriorityTier;
use crate::ids::{ModelId, NodeId, ShardId};

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
    /// Optional cancellation flag. The router/pipeline checks this between
    /// per-token forward calls; flipping it to `true` causes the loop to
    /// stop with `finish_reason = "stop"` on the next iteration. Set by the
    /// `/v1/responses/{id}/cancel` handler (and any other path that needs
    /// to interrupt an in-flight inference). Skipped over the wire — only
    /// the originating node observes it.
    #[serde(skip)]
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
            cancel: None,
        }
    }

    /// True iff the request has been cancelled (cancel flag flipped to true).
    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
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
    /// All segment holders advertise support for speculative verify-batch
    /// (multi-position forward with spec_logits in the result). Set by the
    /// scheduler after checking peer capability; false disables the speculative
    /// distributed path for this request.
    #[serde(default)]
    pub supports_speculative: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSegment {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    pub layer_range: (u32, u32),
}

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
    /// Decoded-so-far token IDs for OpenAI-style frequency_penalty /
    /// presence_penalty. Populated by the daemon coordinator on the
    /// final-segment forward when penalties are non-zero, so the worker
    /// applies penalties against the completion-so-far.
    /// Serialized only when non-empty to keep zero-penalty requests on
    /// the existing wire size.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generated_ids: Vec<u32>,
    /// LoRA adapter ID to apply during inference. When set, the worker loads the
    /// adapter from the data_dir and applies its low-rank deltas per-layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Speculative decoding: γ draft token IDs for the target model to verify.
    /// When non-empty, the worker runs a multi-position forward pass over these
    /// tokens and returns one logit vector per position (plus a bonus position)
    /// in `LayerResult.spec_logits`. `activations` carries the γ token IDs as
    /// i64 LE when `draft_tokens` is non-empty (redundant but keeps the single
    /// decode-path tensor build happy).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_tokens: Vec<u32>,
    /// Set to true by the coordinator when it wants `spec_logits` populated on
    /// the result. Ignored when `draft_tokens` is empty.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub spec_logits_requested: bool,
    /// Speculative decoding KV-cache fixup: if `Some(L)`, the worker truncates
    /// the per-request KV cache to exactly L sequence positions BEFORE running
    /// this forward. Used after partial acceptance to discard the trailing γ-k
    /// draft entries committed in the previous verify round.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_kv_to: Option<u32>,
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
    /// Speculative decoding: per-position logit vectors returned by the target
    /// model when `LayerForward.spec_logits_requested` was set. Length is
    /// always γ+1 — `greedy_accept_reject` indexes `spec_logits[drafts.len()]`
    /// (= γ) on the ALL-ACCEPTED branch as the bonus logit. A length-γ payload
    /// would OOB-panic that index. See gotcha #29 in `memory/MEMORY.md` for
    /// the R67 fix that nailed this contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spec_logits: Vec<Vec<f32>>,
}

/// A single token streamed back from the final pipeline node to the originator.
/// Used for SSE streaming in distributed inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingToken {
    pub request_id: uuid::Uuid,
    pub token_id: u32,
    pub finish_reason: Option<NetworkFinishReason>,
    /// Pre-decoded token text. Populated by the remote-generate fast path so
    /// the coordinator doesn't need to hold the tokenizer. Empty on the
    /// per-token pipeline path (backward-compatible via serde default).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    /// Token counts for the final "done" event. Populated only when
    /// `finish_reason` is `Some`. Ignored on in-flight tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerateUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GenerateUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Request a remote peer to run the full generation loop for a single-segment
/// pipeline. The remote tokenizes the prompt, prefills, runs the decode loop
/// in its local worker subprocess (no per-token IPC round trip to the
/// coordinator), and streams back each generated token as a `StreamingToken`.
///
/// Only eligible when:
/// - the pipeline is single-segment (one peer holds the entire layer range)
/// - no TP groups
/// - no vision / LoRA / pipeline sealing (those need coordinator involvement)
///
/// Reduces the per-token latency from ~140 ms (libp2p substream + IPC round
/// trip per token) to ~network-transit + compute (~20-30 ms on loopback).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteGenerateRequest {
    pub request_id: uuid::Uuid,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    /// Already-formatted prompt (chat template applied by coordinator).
    pub prompt: String,
    pub sampling: SamplingParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Populated locally after receiving from the network — never on the wire.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
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
