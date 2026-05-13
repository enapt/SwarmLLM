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

fn is_zero_u32_vision(v: &u32) -> bool {
    *v == 0
}

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
    /// Item 8 Phase 2b: daemon's reply to a `WorkerMsg::PrefixFetchProbe`.
    /// When `present == true`, the BLAKE3-verified `KvSnapshot` bytes are
    /// carried in the IPC binary-payload slot (not inside this JSON header).
    /// `present == false` means miss / timeout / verification failure.
    /// `matched_tokens` is the number of prompt tokens covered by the
    /// snapshot (equal to the chained-hash block boundary the daemon
    /// resolved against).
    PrefixFetchResult {
        request_id: Uuid,
        matched_tokens: u32,
        present: bool,
    },
    /// Item 8 Phase 2b: serving-side request. The daemon received an
    /// inbound `SwarmRequest::PrefixKvFetch` from a peer and needs the
    /// local worker to extract the matching snapshot bytes from its
    /// in-process `PrefixCache`. The worker replies with
    /// `WorkerMsg::PrefixSnapshotResponse`.
    ExportPrefixSnapshot {
        request_id: Uuid,
        model_id: ModelId,
        block_hash: [u8; 32],
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
        /// The user-provided stop sequence that matched, if any. Populated
        /// only when `finish_reason == "stop"` and the match came from
        /// `SamplingParams.stop` (not EOS). Carried back to the API layer
        /// so Anthropic clients see the actual matched sequence rather
        /// than `null`. Optional + default-skipping keeps wire-compat with
        /// older worker builds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        matched_stop_sequence: Option<String>,
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
    /// Item 8 Phase 2b: worker-initiated probe for a cross-node prefix KV
    /// fetch. Carries the chained-hash manifest of the current prompt's
    /// leading blocks; the daemon picks the longest-prefix peer match,
    /// fetches + verifies + returns bytes via `DaemonMsg::PrefixFetchResult`.
    /// `request_id` correlates the probe with the result.
    PrefixFetchProbe {
        request_id: Uuid,
        model_id: ModelId,
        blocks: Vec<PrefixBlockEntry>,
    },
    /// Item 8 Phase 2b: worker's reply to `DaemonMsg::ExportPrefixSnapshot`.
    /// When `present == true`, the serialized `KvSnapshot` bytes are carried
    /// in the IPC binary-payload slot (not inside this JSON header).
    /// Snapshots can be tens of MB; JSON-encoding a `Vec<u8>` bloats by ~5×
    /// and overflows the 64 MiB header cap. `present == false` means the
    /// worker couldn't produce a snapshot (eviction race / miss).
    PrefixSnapshotResponse { request_id: Uuid, present: bool },
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
    /// Byte length of the vision-embedding prefix at the head of the IPC
    /// binary payload (zero when absent). Vision embeddings used to live
    /// inside this JSON header as `Option<Vec<u8>>`, but serde_json encodes
    /// `Vec<u8>` as a JSON array of integers (~5× bloat). LLaVA-class
    /// mmproj output can exceed 1 MiB before zstd compression and tens of
    /// MiB after decompression in pathological cases, and the same latent
    /// bomb as spec_logits (gotcha #24) can push the JSON header past
    /// `MAX_HEADER` (64 MiB). Payload layout: `[vision_bytes][activation_bytes]`
    /// with the vision prefix length given here.
    #[serde(default, skip_serializing_if = "is_zero_u32_vision")]
    pub vision_embeddings_len: u32,
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
    /// OpenAI-style frequency_penalty / presence_penalty history: the
    /// completion-so-far token IDs. Populated by the daemon coordinator on
    /// the final-segment forward when `sampling.frequency_penalty != 0` or
    /// `sampling.presence_penalty != 0`; empty otherwise (default no-op,
    /// no wire bloat). Worker passes this to
    /// `apply_repetition_penalties` before sampling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generated_ids: Vec<u32>,
    /// Coordinator wants per-position logit vectors populated on the result.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub spec_logits_requested: bool,
    /// KV cache truncation: if `Some(L)`, worker truncates per-request KV to
    /// L sequence positions BEFORE running the forward. Used by speculative
    /// partial-accept fixup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_kv_to: Option<u32>,
}

/// Forward-pass result header.
///
/// The binary payload slot carries ONE of (mutually exclusive):
/// - activation bytes (when `has_activations`)
/// - raw f32 speculative logits, row-major `spec_logits_dims.0 × .1`
///   (when `has_spec_logits`)
/// - nothing (when both flags are false, e.g. last-segment token emit)
///
/// Speculative logits used to live inside this JSON header as
/// `Vec<Vec<f32>>`, but serde_json encodes each f32 as ~7 bytes of
/// decimal text — a 5–7× bloat over binary. On large-vocab models
/// (Qwen2.5 at 151 K, γ=4) that approaches `MAX_HEADER` (64 MiB) and
/// on γ=8 blows past it. Same class of bug as gotcha #24
/// (`Vec<u8>` in JSON header).
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
    /// User-provided stop sequence that triggered termination, if any.
    /// Populated by the worker when `finish_reason == Stop` and the match
    /// came from `SamplingParams.stop`. Plumbed up to `LayerResult` so the
    /// distributed-path coordinator can carry it back to API clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_stop_sequence: Option<String>,
    /// True if binary payload contains output activation bytes.
    #[serde(default)]
    pub has_activations: bool,
    /// True if binary payload contains raw f32 spec-verify logits.
    /// Mutually exclusive with `has_activations`.
    #[serde(default)]
    pub has_spec_logits: bool,
    /// `(n_positions, vocab_size)` — only meaningful when
    /// `has_spec_logits` is true. Used to reshape the f32 payload back
    /// into `Vec<Vec<f32>>` on the receiving side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_logits_dims: Option<(u32, u32)>,
}

/// Encode `Vec<Vec<f32>>` as a flat little-endian f32 byte buffer,
/// row-major. Companion to `decode_spec_logits` — sender side.
pub fn encode_spec_logits(logits: &[Vec<f32>]) -> (Vec<u8>, (u32, u32)) {
    let n_positions = logits.len() as u32;
    let vocab_size = logits.first().map(|r| r.len()).unwrap_or(0) as u32;
    let mut bytes = Vec::with_capacity((n_positions as usize) * (vocab_size as usize) * 4);
    for row in logits {
        for v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    (bytes, (n_positions, vocab_size))
}

/// Decode a flat little-endian f32 payload back into `Vec<Vec<f32>>`
/// given the `(n_positions, vocab_size)` dims recorded in the header.
/// Returns an error when the payload length disagrees with dims.
pub fn decode_spec_logits(bytes: &[u8], dims: (u32, u32)) -> Result<Vec<Vec<f32>>, String> {
    let n = dims.0 as usize;
    let v = dims.1 as usize;
    let expected = n
        .checked_mul(v)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| "spec_logits dims overflow usize".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "spec_logits payload len {} mismatch — expected {} (n={}, v={})",
            bytes.len(),
            expected,
            n,
            v
        ));
    }
    let mut out = Vec::with_capacity(n);
    for row_bytes in bytes.chunks_exact(v * 4) {
        let mut row = Vec::with_capacity(v);
        for chunk in row_bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.push(row);
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_logits_roundtrip_empty() {
        let (bytes, dims) = encode_spec_logits(&[]);
        assert_eq!(bytes, Vec::<u8>::new());
        assert_eq!(dims, (0, 0));
    }

    #[test]
    fn spec_logits_roundtrip_small() {
        let original: Vec<Vec<f32>> = vec![
            vec![0.1, -0.2, 2.5, f32::NEG_INFINITY],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-0.0, 0.0, f32::MIN_POSITIVE, f32::MAX],
        ];
        let (bytes, dims) = encode_spec_logits(&original);
        assert_eq!(dims, (3, 4));
        assert_eq!(bytes.len(), 3 * 4 * 4);
        let restored = decode_spec_logits(&bytes, dims).unwrap();
        assert_eq!(restored.len(), original.len());
        for (o, r) in original.iter().zip(restored.iter()) {
            assert_eq!(o.len(), r.len());
            for (a, b) in o.iter().zip(r.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "bit-exact roundtrip required");
            }
        }
    }

    #[test]
    fn spec_logits_decode_rejects_len_mismatch() {
        let err = decode_spec_logits(&[0u8; 3], (1, 2)).unwrap_err();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn vision_embeddings_len_elided_when_zero() {
        use crate::types::{ModelId, TensorFormat};
        use uuid::Uuid;
        let fwd = IpcForward {
            request_id: Uuid::nil(),
            sequence_num: 1,
            index_pos: 5,
            format: TensorFormat::FP32,
            model_id: ModelId("test".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings_len: 0,
            requester_node_id: None,
            pre_embedded: false,
            sampling: Default::default(),
            adapter_id: None,
            draft_tokens: vec![],
            generated_ids: vec![],
            spec_logits_requested: false,
            truncate_kv_to: None,
        };
        let json = serde_json::to_string(&fwd).unwrap();
        // When no vision payload, the field is elided to match pre-fix wire
        // shape and avoid bloating decode-hot-path forwards.
        assert!(
            !json.contains("vision_embeddings_len"),
            "expected vision_embeddings_len to be elided from JSON when zero, got: {json}"
        );
        assert!(!json.contains("vision_embeddings"));
    }

    #[test]
    fn vision_embeddings_len_present_when_nonzero() {
        use crate::types::{ModelId, TensorFormat};
        use uuid::Uuid;
        let fwd = IpcForward {
            request_id: Uuid::nil(),
            sequence_num: 0,
            index_pos: 0,
            format: TensorFormat::FP32,
            model_id: ModelId("test".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings_len: 12345,
            requester_node_id: None,
            pre_embedded: false,
            sampling: Default::default(),
            adapter_id: None,
            draft_tokens: vec![],
            generated_ids: vec![],
            spec_logits_requested: false,
            truncate_kv_to: None,
        };
        let json = serde_json::to_string(&fwd).unwrap();
        assert!(json.contains("\"vision_embeddings_len\":12345"));
        let back: IpcForward = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vision_embeddings_len, 12345);
    }

    #[test]
    fn spec_logits_realistic_vocab() {
        // γ=4, vocab=32000 — a TinyLlama-class spec verify payload.
        let mut original = Vec::with_capacity(5);
        for row_idx in 0..5 {
            let mut row = Vec::with_capacity(32000);
            for v in 0..32000 {
                row.push((row_idx as f32 * 0.01) + (v as f32 * 1e-6));
            }
            original.push(row);
        }
        let (bytes, dims) = encode_spec_logits(&original);
        assert_eq!(dims, (5, 32000));
        // Raw f32 binary = 5 * 32000 * 4 = 640_000 bytes — compare to the
        // old JSON encoding which would have been ~4-5 MB for the same
        // payload (each f32 ~7 bytes of decimal text).
        assert_eq!(bytes.len(), 640_000);
        let restored = decode_spec_logits(&bytes, dims).unwrap();
        assert_eq!(restored.len(), 5);
        assert_eq!(restored[0].len(), 32000);
        assert_eq!(restored[4][31999].to_bits(), original[4][31999].to_bits());
    }
}
