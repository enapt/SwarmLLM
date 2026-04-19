//! IPC protocol between the SwarmLLM daemon and model worker subprocesses.
//!
//! Message framing:
//!   [4 bytes LE: json_len][json_len bytes: JSON header]
//!   [4 bytes LE: payload_len][payload_len bytes: raw bytes (0 if none)]

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

use crate::types::{
    ModelId, NetworkFinishReason, PrefixBlockEntry, SamplingParams, TensorFormat,
    TensorParallelMeta,
};

use crate::inference::router::TokenLogProbEntry;

// Max framing sizes
const MAX_HEADER: u32 = 64 * 1024 * 1024;
const MAX_PAYLOAD: u32 = 512 * 1024 * 1024;

/// Message from daemon → worker.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum DaemonMsg {
    /// Single-step forward pass (for distributed inference).
    /// Binary payload = activation bytes.
    Forward(IpcForward),
    /// Batched forward pass — multiple requests folded into one IPC call.
    /// Worker processes each request (v1: sequentially; v2: stacked tensor)
    /// and returns one result per request in order. Binary payload is the
    /// concatenation of each request's activation bytes, with `activation_lens`
    /// providing the slice boundaries.
    BatchForward {
        requests: Vec<IpcForward>,
        activation_lens: Vec<u32>,
    },
    /// Full generation loop (for local API inference).
    /// Worker tokenizes, runs prefill + decode, streams tokens back.
    Generate(IpcGenerate),
    /// Unload a specific layer range (free its GPU memory within the worker).
    Unload {
        layer_start: usize,
        layer_end: usize,
    },
    /// Graceful shutdown — worker exits cleanly.
    Shutdown,
}

/// Message from worker → daemon.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum WorkerMsg {
    /// Worker connected and ready.
    Ready,
    /// Single-step forward result (for Forward requests).
    /// Binary payload = activation bytes (if has_activations).
    LayerResult(IpcLayerResult),
    /// Batched forward results — one per request in `BatchForward.requests`.
    /// `activation_lens[i]` gives the byte length of request i's activations
    /// within the concatenated binary payload (0 if no activations).
    BatchResult {
        results: Vec<IpcLayerResult>,
        activation_lens: Vec<u32>,
    },
    /// A single decoded token (for streaming Generate).
    Token {
        request_id: Uuid,
        token_id: u32,
        text: String,
        is_eos: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
    },
    /// Generation complete (follows the last Token message).
    GenerateDone {
        request_id: Uuid,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: String,
    },
    /// Error for a specific request.
    Error { request_id: Uuid, message: String },
    /// Item 8 Phase 1: notify the daemon that the worker just inserted (or
    /// refreshed) prefix-cache entries for `model_id`. The daemon broadcasts
    /// this as a `SwarmMessage::PrefixCacheAnnounce` and updates its own
    /// cross-node index. Carries no `request_id` and is fanned out by the
    /// reader actor through a dedicated channel rather than per-request
    /// response routing.
    PrefixManifestUpdate {
        model_id: ModelId,
        blocks: Vec<PrefixBlockEntry>,
    },
    /// Worker is about to exit.
    Bye,
}

/// Forward-pass request header (activation bytes are the binary payload).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcForward {
    pub request_id: Uuid,
    pub sequence_num: u32,
    pub index_pos: u32,
    pub format: TensorFormat,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_meta: Option<TensorParallelMeta>,
    /// Vision embeddings (zstd FP16) included in JSON — only on first pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_embeddings: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requester_node_id: Option<[u8; 32]>,
    #[serde(default)]
    pub pre_embedded: bool,
    pub sampling: SamplingParams,
    /// LoRA adapter ID to apply during inference. The worker loads the adapter
    /// from the data_dir/adapters/ directory on first use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Speculative decoding: γ draft tokens. When non-empty, the worker runs
    /// a multi-position forward and returns per-position logits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_tokens: Vec<u32>,
    /// Coordinator wants per-position logit vectors populated on the result.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub spec_logits_requested: bool,
    /// KV cache truncation: if `Some(L)`, worker truncates per-request KV to
    /// L sequence positions BEFORE running the forward. Used by speculative
    /// partial-accept fixup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_kv_to: Option<u32>,
}

/// Forward-pass result header (activation bytes are the binary payload).
#[derive(Serialize, Deserialize, Debug)]
pub struct IpcLayerResult {
    pub request_id: Uuid,
    pub token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<NetworkFinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<TensorFormat>,
    #[serde(default)]
    pub sealed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_payload: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TokenLogProbEntry>>,
    /// True if binary payload contains output activation bytes.
    #[serde(default)]
    pub has_activations: bool,
    /// Speculative decoding: per-position logit vectors (one per γ draft
    /// positions). Empty on normal forwards.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spec_logits: Vec<Vec<f32>>,
}

/// Generate request (full decode loop in the worker).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcGenerate {
    pub request_id: Uuid,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    /// Already-formatted prompt (chat template applied by caller).
    pub prompt: String,
    pub sampling: SamplingParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Send a DaemonMsg to the socket. Payload = raw bytes (e.g., activations).
pub async fn send_daemon<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg: &DaemonMsg,
    payload: &[u8],
) -> std::io::Result<()> {
    send_framed(w, msg, payload).await
}

/// Send a WorkerMsg to the socket.
pub async fn send_worker<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg: &WorkerMsg,
    payload: &[u8],
) -> std::io::Result<()> {
    send_framed(w, msg, payload).await
}

/// Read a DaemonMsg from the socket.
pub async fn recv_daemon<R: AsyncReadExt + Unpin>(
    r: &mut R,
) -> std::io::Result<(DaemonMsg, Vec<u8>)> {
    recv_framed(r).await
}

/// Read a WorkerMsg from the socket.
pub async fn recv_worker<R: AsyncReadExt + Unpin>(
    r: &mut R,
) -> std::io::Result<(WorkerMsg, Vec<u8>)> {
    recv_framed(r).await
}

async fn send_framed<W: AsyncWriteExt + Unpin, T: Serialize>(
    w: &mut W,
    msg: &T,
    payload: &[u8],
) -> std::io::Result<()> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // SEC: Validate lengths fit in u32 before casting to prevent silent truncation
    let json_len = u32::try_from(json.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC header too large for u32",
        )
    })?;
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC payload too large for u32",
        )
    })?;
    w.write_all(&json_len.to_le_bytes()).await?;
    w.write_all(&json).await?;
    w.write_all(&payload_len.to_le_bytes()).await?;
    if !payload.is_empty() {
        w.write_all(payload).await?;
    }
    w.flush().await
}

async fn recv_framed<R: AsyncReadExt + Unpin, T: for<'de> Deserialize<'de>>(
    r: &mut R,
) -> std::io::Result<(T, Vec<u8>)> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4).await?;
    let json_len = u32::from_le_bytes(buf4);
    if json_len == 0 || json_len > MAX_HEADER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IPC header len {json_len} invalid (must be 1..{MAX_HEADER})"),
        ));
    }
    let mut json_buf = vec![0u8; json_len as usize];
    r.read_exact(&mut json_buf).await?;
    let msg: T = serde_json::from_slice(&json_buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    r.read_exact(&mut buf4).await?;
    let payload_len = u32::from_le_bytes(buf4);
    if payload_len > MAX_PAYLOAD {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IPC payload {payload_len} > max {MAX_PAYLOAD}"),
        ));
    }
    let mut payload = vec![0u8; payload_len as usize];
    if payload_len > 0 {
        r.read_exact(&mut payload).await?;
    }
    Ok((msg, payload))
}
