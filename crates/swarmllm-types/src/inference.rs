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
    /// Tier 4K — daemon-side STREAM-chunked activation send. Present when this
    /// frame carries one chunk of a multi-chunk activation transfer; absent
    /// when the activation fits in a single frame (the common case for decode
    /// where activations are ~16 KB). Bound into the AAD via
    /// `protocol::build_layer_forward_aad` so a receiver cannot accept chunks
    /// out of order, with a wrong total, or with a forged final-flag.
    ///
    /// Backward compat: frames without the trailer behave as
    /// `(chunk_idx=0, total_chunks=1)` — single-chunk implicit. Existing wire
    /// paths and older peers see exactly today's behaviour.
    ///
    /// `is_final` is derived: `chunk_idx + 1 == total_chunks`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_meta: Option<ChunkMeta>,
}

/// Tier 4K — chunked activation transport metadata. Carried as the optional
/// 0x05 trailer on `LayerForward`. AAD-bound so reorder/truncation attempts
/// fail authentication before reaching the dispatch path.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkMeta {
    /// 0-indexed chunk position within the logical activation transfer.
    pub chunk_idx: u32,
    /// Total number of chunks in this transfer. The receiver assembles all
    /// `total_chunks` frames (same `request_id`) into the original activation
    /// before dispatching to the worker.
    pub total_chunks: u32,
}

impl ChunkMeta {
    /// True for the last chunk of a transfer — terminates receiver assembly.
    pub fn is_final(&self) -> bool {
        self.chunk_idx + 1 == self.total_chunks
    }
}

/// Tier 4K receiver-side assembly state for a single chunked activation
/// transfer. Stored in `SharedState.pending_activation_chunks` keyed by
/// `request_id`. The receiver inserts each chunk at its `chunk_idx` slot;
/// when all slots are filled, the dispatch path concatenates and forwards
/// the reassembled activation to the worker as a single non-chunked
/// LayerForward.
///
/// `total_chunks` is set on the first chunk that arrives. Subsequent chunks
/// MUST agree on `total_chunks` (validated by AAD — flipping it on the wire
/// fails Poly1305 — and re-checked here as defence-in-depth). `last_update_at`
/// drives the TTL eviction sweep.
///
/// Fields are kept simple (no internal locks) — DashMap's per-entry lock is
/// sufficient because each chunk lookup/insert happens under `DashMap::entry`.
/// The dispatch path holds the entry while reassembling, then removes it
/// atomically.
#[derive(Debug)]
pub struct ChunkAssemblyState {
    /// Pre-allocated `Vec<Option<Vec<u8>>>` of length `total_chunks`.
    /// Index = `chunk_idx`. `None` until the chunk arrives.
    pub received: Vec<Option<Vec<u8>>>,
    /// Sender-asserted total, locked on first chunk; later chunks with a
    /// different `total_chunks` are rejected.
    pub total_chunks: u32,
    /// Cleartext template for the eventually-dispatched LayerForward
    /// (request_id, layer_range, model_id, sequence_num, ...). All chunks
    /// in a transfer carry identical cleartext metadata except for
    /// chunk_meta itself — captured once on first-chunk arrival.
    pub template: Box<LayerForward>,
    /// Sender peer ID (for routing the result back). Captured on first
    /// chunk; verified equal on subsequent chunks.
    pub sender_peer_bytes: Vec<u8>,
    /// Wall-clock instant of last chunk insertion. Used by the TTL sweep
    /// to evict stale incomplete assemblies.
    pub last_update_at: std::time::Instant,
    /// Cached count of slots filled — avoids an O(K) scan per insert.
    pub filled: u32,
}

impl ChunkAssemblyState {
    /// Allocate a new assembly with `total_chunks` empty slots.
    pub fn new(total_chunks: u32, template: LayerForward, sender_peer_bytes: Vec<u8>) -> Self {
        Self {
            received: (0..total_chunks).map(|_| None).collect(),
            total_chunks,
            template: Box::new(template),
            sender_peer_bytes,
            last_update_at: std::time::Instant::now(),
            filled: 0,
        }
    }

    /// True when every chunk slot has been filled.
    pub fn is_complete(&self) -> bool {
        self.filled == self.total_chunks
    }

    /// Concatenate received chunks in order. Caller must verify
    /// `is_complete()` first; this unwraps each `Option` directly.
    pub fn assemble(&self) -> Vec<u8> {
        let total: usize = self
            .received
            .iter()
            .map(|c| c.as_ref().map_or(0, |v| v.len()))
            .sum();
        let mut out = Vec::with_capacity(total);
        for bytes in self.received.iter().flatten() {
            out.extend_from_slice(bytes);
        }
        out
    }
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
    /// The user-provided stop sequence that triggered termination, if any.
    /// Populated only when `finish_reason == Stop` AND a sequence from
    /// `SamplingParams.stop` matched the accumulated text. Carried so the
    /// distributed-path coordinator can populate
    /// `InferenceOutput.matched_stop_sequence` and Anthropic clients see
    /// the actual matched sequence rather than `null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_stop_sequence: Option<String>,
    /// Per-token log probabilities streamed back through the distributed
    /// pipeline (one entry per token in `token_ids`). The local-worker
    /// path collects these via `IpcLayerResult.logprobs`; this field is
    /// the wire-side equivalent for cross-node inference. Empty when
    /// `logprobs=false` in the originating request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_logprobs: Vec<TokenLogProbEntry>,
}

/// A single token's log-probability info for distributed inference responses.
/// Mirrors the type used by the local worker path so the same structure flows
/// through both code paths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenLogProbEntry {
    /// The token text.
    pub token: String,
    /// Log probability of this token.
    pub logprob: f32,
    /// Top-N alternative tokens with their logprobs.
    pub top_logprobs: Vec<(String, f32)>,
}

impl LayerResult {
    /// Canonical empty/error response: no tokens, no activations, no spec
    /// logits, just a `finish_reason::Error(reason)`. Used by every site that
    /// has to fail a pending pipeline forward (encryption-session loss,
    /// pending-channel capacity exceeded, missing shards, stream-reader
    /// termination, dispatch failure). Centralising this constructor keeps
    /// the empty-error wire shape in one place — adding a new field to
    /// `LayerResult` only needs an update here, mirroring
    /// `ShardResponse::empty()` in `swarmllm-types`.
    pub fn error(request_id: uuid::Uuid, reason: impl Into<String>) -> Self {
        Self {
            request_id,
            token_ids: Vec::new(),
            finish_reason: Some(NetworkFinishReason::Error(reason.into())),
            activations: Vec::new(),
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        }
    }
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
    /// User-provided stop sequence that triggered termination, if any.
    /// Populated only on the terminal token (`finish_reason == Some(Stop)`),
    /// and only when the remote worker's stop-string detection produced a
    /// match. Mirrors the local-worker contract; safe to leave `None` on
    /// in-flight tokens or when EOS terminated generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_stop_sequence: Option<String>,
    /// Optional per-token log probability streamed alongside the token text
    /// for distributed remote-generate. Empty when `logprobs=false` in the
    /// request or when the remote worker doesn't compute logprobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprob: Option<TokenLogProbEntry>,
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

#[cfg(test)]
mod chunk_assembly_tests {
    use super::*;
    use uuid::Uuid;

    fn base_template() -> LayerForward {
        LayerForward {
            request_id: Uuid::from_u128(0xFEED_FACE_DEAD_BEEF_1234_5678_9ABC_DEF0),
            sequence_num: 1,
            index_pos: 0,
            activations: Vec::new(),
            format: TensorFormat::FP32,
            model_id: ModelId("test-model".into()),
            layer_range: (0, 8),
            tp_meta: None,
            vision_embeddings: None,
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
        }
    }

    #[test]
    fn chunk_meta_is_final_only_for_last_index() {
        assert!(!ChunkMeta {
            chunk_idx: 0,
            total_chunks: 4
        }
        .is_final());
        assert!(!ChunkMeta {
            chunk_idx: 1,
            total_chunks: 4
        }
        .is_final());
        assert!(!ChunkMeta {
            chunk_idx: 2,
            total_chunks: 4
        }
        .is_final());
        assert!(ChunkMeta {
            chunk_idx: 3,
            total_chunks: 4
        }
        .is_final());
    }

    #[test]
    fn assembly_state_reports_complete_after_all_slots_filled() {
        let mut state = ChunkAssemblyState::new(3, base_template(), vec![1, 2, 3]);
        assert!(!state.is_complete());
        state.received[0] = Some(vec![1, 2]);
        state.filled = 1;
        assert!(!state.is_complete());
        state.received[2] = Some(vec![5, 6]);
        state.filled = 2;
        assert!(!state.is_complete());
        state.received[1] = Some(vec![3, 4]);
        state.filled = 3;
        assert!(state.is_complete());
    }

    #[test]
    fn assembly_concatenates_chunks_in_index_order() {
        let mut state = ChunkAssemblyState::new(3, base_template(), vec![]);
        state.received[2] = Some(vec![7, 8, 9]);
        state.received[0] = Some(vec![1, 2, 3]);
        state.received[1] = Some(vec![4, 5, 6]);
        state.filled = 3;
        assert_eq!(state.assemble(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn assembly_with_uneven_chunks_preserves_total_length() {
        // 5000 bytes, K=3 → 2048, 2048, 904
        let mut state = ChunkAssemblyState::new(3, base_template(), vec![]);
        state.received[0] = Some(vec![0xAAu8; 2048]);
        state.received[1] = Some(vec![0xBBu8; 2048]);
        state.received[2] = Some(vec![0xCCu8; 904]);
        state.filled = 3;
        let asm = state.assemble();
        assert_eq!(asm.len(), 5000);
        assert_eq!(asm[0], 0xAA);
        assert_eq!(asm[2047], 0xAA);
        assert_eq!(asm[2048], 0xBB);
        assert_eq!(asm[4095], 0xBB);
        assert_eq!(asm[4096], 0xCC);
        assert_eq!(asm[4999], 0xCC);
    }
}
