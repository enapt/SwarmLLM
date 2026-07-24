use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response;
use libp2p::StreamProtocol;

#[cfg(test)]
use crate::types::{LayerForward, LayerResult, ModelId, NetworkFinishReason, TensorFormat};
use crate::types::{ShardRequest, ShardResponse, SwarmMessage};

/// Protocol ID for SwarmLLM unified request/response (JSON control + binary tensor).
pub const PROTOCOL_ID: &str = "/swarmllm/1.0.0";

/// GossipSub topic for model coordination (shard announcements, capacity updates).
pub const TOPIC_MODELS: &str = "swarm/models";

/// GossipSub topic for credit balance gossip.
pub const TOPIC_CREDITS: &str = "swarm/credits";

/// GossipSub topic for network health.
pub const TOPIC_HEALTH: &str = "swarm/health";

/// GossipSub topic for identity/nickname announcements.
pub const TOPIC_IDENTITY: &str = "swarm/identity";

/// GossipSub topic for device pool management.
pub const TOPIC_POOLS: &str = "swarm/pools";

/// GossipSub topic for regional shard summaries and demand gossip.
pub const TOPIC_REGIONS: &str = "swarm/regions";

/// Maximum message size for request_response protocol (256 MB).
const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
/// Maximum JSON control message size (4 MB).
const MAX_JSON_MSG_SIZE: usize = 4 * 1024 * 1024;
/// Shard transfer chunk size (32 MB) — used by acquisition, download, and serve paths.
pub const SHARD_CHUNK_SIZE: u64 = 32 * 1024 * 1024;

/// Maximum activation payload size in layer results (128 MB).
pub(super) const MAX_ACTIVATION_SIZE: usize = 128 * 1024 * 1024;

/// Maximum token count in a single layer result (OOM guard).
pub(super) const MAX_RESULT_TOKENS: usize = 65536;

/// Maximum number of speculative draft tokens accepted in a `LayerForward`
/// 0x03 trailer. Defended in both the plaintext (`layer_forward.rs`) and
/// encrypted (`encrypted.rs`) decoders; mirrored on the receive side by
/// `layer_result.rs::MAX_SPEC_LOGITS_POSITIONS = 32`. Speculative γ is
/// bounded by `GammaController::DEFAULT_GAMMA_MAX = 12`; the cap leaves
/// generous headroom while keeping `Vec::with_capacity(num_drafts)`
/// bounded against malicious peers (R107).
pub(super) const MAX_DRAFT_TOKENS: usize = 32;

/// Codec for SwarmLLM request/response protocol using serde_json.
///
/// Compression knobs (all opt-in on the *send* side; the *receive* side
/// always handles both compressed and raw frames so a node with the flag
/// off can still decode payloads from a peer that has it on):
///
/// - `compress_tensors`: zstd-compress activation tensor payloads above
///   `compress_threshold` bytes (wire tag 0x02). Default on.
/// - `compress_prefix_kv`: zstd-compress cross-node prefix-KV snapshots
///   above `compress_threshold` bytes (wire tag 0x04, flag=2). Default
///   off — only worth flipping when the WAN bench shows wire size is the
///   binding constraint (localhost's RTT-vs-wire trade is roughly neutral).
///
/// Decompression of incoming compressed payloads always works regardless
/// of the send-side flag.
#[derive(Debug, Clone)]
pub struct SwarmCodec {
    /// Whether to compress outgoing tensor payloads.
    pub compress_tensors: bool,
    /// Whether to compress outgoing prefix-KV snapshot payloads
    /// (`SwarmResponse::PrefixKvData`). Off by default — see
    /// `docs/plans/archive/distributed_inference_speedup.md` § Deferred for the
    /// localhost-vs-WAN trade.
    pub compress_prefix_kv: bool,
    /// Zstd compression level (1-22). Shared between tensor and prefix-KV.
    pub compress_level: i32,
    /// Minimum payload size in bytes to trigger compression. Shared
    /// between tensor and prefix-KV (the call site only attempts compression
    /// when the relevant flag is on, so a single threshold is fine).
    pub compress_threshold: usize,
}

impl Default for SwarmCodec {
    fn default() -> Self {
        Self {
            compress_tensors: true,
            compress_prefix_kv: false,
            compress_level: 1,
            compress_threshold: 1024,
        }
    }
}

/// Request type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmRequest {
    Message(Box<SwarmMessage>),
    ShardTransfer(ShardRequest),
    /// Binary tensor data (LayerForward or LayerResult, already encoded).
    /// Sent as raw bytes to avoid JSON overhead on large activation tensors.
    TensorPayload(Vec<u8>),
    /// NETWORKING_PLAN tensor relay — a tensor forward/result routed through a
    /// relay because the sender can't reach the target directly. The relay
    /// forwards `sealed` blindly to `relay_to`; only `relay_to` can open it
    /// (ephemeral-sealed for its static key). Never JSON-encoded — see the
    /// `WIRE_TAG_RELAYED_TENSOR` codec path.
    RelayedTensor(RelayedTensor),
    /// Item 8 Phase 2: cross-node KV-block fetch request. JSON-encoded
    /// because the payload is small (model_id + 32-byte hash). The matching
    /// response is `SwarmResponse::PrefixKvData`, carrying the serialized
    /// `KvSnapshot` as a binary frame so the large reply avoids JSON
    /// inflation.
    PrefixKvFetch(PrefixKvFetchReq),
}

/// NETWORKING_PLAN tensor relay envelope (`SwarmRequest::RelayedTensor`). Carries
/// a cleartext routing header the relay forwards on, plus the target-sealed
/// tensor bytes the relay can't read. `sealed` is an ephemeral-sealed, already-
/// encoded `LayerForward` (`is_result=false`) or `LayerResult` (`is_result=true`)
/// — see `crypto::relay_seal::{seal,open}_relayed_tensor`.
#[derive(Debug, Clone)]
pub struct RelayedTensor {
    pub relay_to: crate::types::NodeId,
    pub origin: crate::types::NodeId,
    pub request_id: uuid::Uuid,
    /// false = LayerForward (coordinator→server), true = LayerResult (return).
    pub is_result: bool,
    pub ephemeral_pub: [u8; 32],
    pub sealed: Vec<u8>,
}

/// Encode a `RelayedTensor` frame BODY (no tag/length header — the codec adds
/// those). Layout: `[relay_to:32][origin:32][request_id:16][is_result:1][ephemeral_pub:32][sealed…]`.
pub fn encode_relayed_tensor(rt: &RelayedTensor) -> Vec<u8> {
    let mut buf = Vec::with_capacity(RELAYED_TENSOR_HEADER_LEN + rt.sealed.len());
    buf.extend_from_slice(&rt.relay_to.0);
    buf.extend_from_slice(&rt.origin.0);
    buf.extend_from_slice(rt.request_id.as_bytes());
    buf.push(rt.is_result as u8);
    buf.extend_from_slice(&rt.ephemeral_pub);
    buf.extend_from_slice(&rt.sealed);
    buf
}

/// Decode a `RelayedTensor` frame body. Errors if too short to hold the header.
pub fn decode_relayed_tensor(buf: &[u8]) -> io::Result<RelayedTensor> {
    if buf.len() < RELAYED_TENSOR_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "RelayedTensor frame too short",
        ));
    }
    let mut relay_to = [0u8; 32];
    relay_to.copy_from_slice(&buf[0..32]);
    let mut origin = [0u8; 32];
    origin.copy_from_slice(&buf[32..64]);
    let mut rid = [0u8; 16];
    rid.copy_from_slice(&buf[64..80]);
    let is_result = buf[80] != 0;
    let mut ephemeral_pub = [0u8; 32];
    ephemeral_pub.copy_from_slice(&buf[81..113]);
    let sealed = buf[RELAYED_TENSOR_HEADER_LEN..].to_vec();
    Ok(RelayedTensor {
        relay_to: crate::types::NodeId(relay_to),
        origin: crate::types::NodeId(origin),
        request_id: uuid::Uuid::from_bytes(rid),
        is_result,
        ephemeral_pub,
        sealed,
    })
}

/// Response type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmResponse {
    Message(Box<SwarmMessage>),
    ShardData(ShardResponse),
    Ack,
    /// Binary tensor response data (already encoded).
    TensorPayload(Vec<u8>),
    /// Item 8 Phase 2: cross-node KV-block fetch response. `None` means
    /// the serving peer doesn't hold the requested block (negative cache).
    /// `Some(bytes)` is a serialized `KvSnapshot` (see `KV_SNAPSHOT_MAGIC`).
    PrefixKvData(PrefixKvDataResp),
}

/// Phase 2: wire request for a cross-node prefix KV fetch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrefixKvFetchReq {
    pub request_id: uuid::Uuid,
    pub model_id: crate::types::ModelId,
    pub block_hash: [u8; 32],
}

/// Phase 2: wire response carrying a serialized `KvSnapshot`, or `None`
/// if the serving peer doesn't have the requested block anymore (eviction
/// race). `request_id` echoes the fetcher's ID for correlation on the
/// off-chance the caller is multiplexing multiple requests on one
/// substream (not today, but the field is cheap and future-proof).
#[derive(Debug, Clone)]
pub struct PrefixKvDataResp {
    pub request_id: uuid::Uuid,
    pub payload: Option<Vec<u8>>,
}

/// Wire format type tags for the unified codec.
/// First byte of each message distinguishes JSON from binary tensor payloads.
const WIRE_TAG_JSON: u8 = 0x00;
const WIRE_TAG_TENSOR: u8 = 0x01;
/// Zstd-compressed tensor payload. Peers that don't recognize this tag will
/// reject the message, but all nodes running this version or later support it.
const WIRE_TAG_TENSOR_COMPRESSED: u8 = 0x02;
/// Binary shard data frame: [tag][4B total_size_le][data...]
/// Avoids JSON serialization overhead for large shard chunks (32MB+).
const WIRE_TAG_SHARD: u8 = 0x03;
/// Item 8 Phase 2: cross-node prefix KV snapshot binary frame:
///   [tag][4B payload_len_be][16B request_id UUID][1B flag][data...]
/// Flag=0: no payload (miss).
/// Flag=1: `data` is a raw serialized `KvSnapshot`.
/// Flag=2: `data` is a zstd-compressed serialized `KvSnapshot`
///         (gated on `SwarmCodec::compress_prefix_kv`; the receive side
///         always decompresses regardless).
/// Avoids JSON inflation on multi-MB KV payloads.
const WIRE_TAG_PREFIX_KV: u8 = 0x04;

/// NETWORKING_PLAN tensor relay — a distributed-pipeline tensor forward/result
/// routed through a relay. Binary frame (tensors are large + already sealed):
/// `[relay_to:32][origin:32][request_id:16][is_result:1][ephemeral_pub:32][sealed…]`.
/// Feature-gated (`features::TENSOR_RELAY`) — only sent to peers that advertise
/// support, so an older node never receives an unknown tag.
const WIRE_TAG_RELAYED_TENSOR: u8 = 0x06;

/// Fixed header length of a `WIRE_TAG_RELAYED_TENSOR` frame body (everything
/// before `sealed`): 32 + 32 + 16 + 1 + 32.
const RELAYED_TENSOR_HEADER_LEN: usize = 32 + 32 + 16 + 1 + 32;

/// Request-codec wire tags whose bodies are large binary payloads and therefore
/// get the `MAX_MESSAGE_SIZE` frame limit in `read_wire_frame` (everything else
/// is capped at the small `MAX_JSON_MSG_SIZE`). `WIRE_TAG_RELAYED_TENSOR` MUST
/// stay here or every large relayed prefill forward would be rejected as
/// oversized on the receiver. RelayedTensor is request-only.
const REQUEST_LARGE_TAGS: &[u8] = &[
    WIRE_TAG_TENSOR,
    WIRE_TAG_TENSOR_COMPRESSED,
    WIRE_TAG_RELAYED_TENSOR,
];

/// Response-codec counterpart of `REQUEST_LARGE_TAGS`. Shard + prefix-KV frames
/// are response-only, so they belong here rather than in the request set.
const RESPONSE_LARGE_TAGS: &[u8] = &[
    WIRE_TAG_TENSOR,
    WIRE_TAG_TENSOR_COMPRESSED,
    WIRE_TAG_SHARD,
    WIRE_TAG_PREFIX_KV,
];

/// Read a wire frame header (tag byte + 4-byte BE length) and body from a stream.
/// `large_tags` lists tag values that get the larger `MAX_MESSAGE_SIZE` limit;
/// all other tags are capped at `MAX_JSON_MSG_SIZE`.
async fn read_wire_frame<T: AsyncRead + Unpin + Send>(
    io: &mut T,
    label: &str,
    large_tags: &[u8],
) -> io::Result<(u8, Vec<u8>)> {
    tracing::trace!("DIAG: codec {label} waiting for tag");
    let mut tag_buf = [0u8; 1];
    io.read_exact(&mut tag_buf).await?;

    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    tracing::trace!(tag = tag_buf[0], len, "DIAG: codec {label} header");

    let max_for_tag = if large_tags.contains(&tag_buf[0]) {
        MAX_MESSAGE_SIZE
    } else {
        MAX_JSON_MSG_SIZE
    };
    if len > max_for_tag {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} too large"),
        ));
    }

    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Ok((tag_buf[0], buf))
}

/// Decompress a zstd-compressed tensor payload with diagnostic tracing.
pub(super) fn decompress_tensor_payload(buf: Vec<u8>, label: &str) -> io::Result<Vec<u8>> {
    let decompressed = compression::decompress_tensor(&buf).map_err(|e| {
        tracing::error!(
            compressed_len = buf.len(),
            "DIAG: {label} tensor decompression failed: {e}"
        );
        io::Error::new(io::ErrorKind::InvalidData, e)
    })?;
    tracing::debug!(
        compressed_len = buf.len(),
        decompressed_len = decompressed.len(),
        ratio = format_args!(
            "{:.1}x",
            decompressed.len() as f64 / buf.len().max(1) as f64
        ),
        "DIAG: {label} tensor decompressed"
    );
    Ok(decompressed)
}

/// Build a JSON wire frame: [WIRE_TAG_JSON][4B BE length][JSON bytes].
fn build_json_frame<T: serde::Serialize>(msg: &T, label: &str) -> io::Result<Vec<u8>> {
    let data =
        serde_json::to_vec(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if data.len() > MAX_JSON_MSG_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "JSON {label} too large: {} bytes (max {})",
                data.len(),
                MAX_JSON_MSG_SIZE
            ),
        ));
    }
    let len = (data.len() as u32).to_be_bytes();
    let mut frame = Vec::with_capacity(1 + 4 + data.len());
    frame.push(WIRE_TAG_JSON);
    frame.extend_from_slice(&len);
    frame.extend_from_slice(&data);
    Ok(frame)
}

/// Build a relayed-tensor wire frame: [WIRE_TAG_RELAYED_TENSOR][4B BE length][body].
/// Bounded by `MAX_MESSAGE_SIZE` (the large-tensor limit) since the sealed
/// activation body can be multi-MB.
fn build_relayed_tensor_frame(rt: &RelayedTensor) -> io::Result<Vec<u8>> {
    let body = encode_relayed_tensor(rt);
    if body.len() > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "relayed tensor too large: {} bytes (max {MAX_MESSAGE_SIZE})",
                body.len()
            ),
        ));
    }
    let len = (body.len() as u32).to_be_bytes();
    let mut frame = Vec::with_capacity(1 + 4 + body.len());
    frame.push(WIRE_TAG_RELAYED_TENSOR);
    frame.extend_from_slice(&len);
    frame.extend_from_slice(&body);
    Ok(frame)
}

#[async_trait]
impl request_response::Codec for SwarmCodec {
    type Protocol = StreamProtocol;
    type Request = SwarmRequest;
    type Response = SwarmResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let (tag, buf) = read_wire_frame(io, "read_request", REQUEST_LARGE_TAGS).await?;

        match tag {
            WIRE_TAG_JSON => serde_json::from_slice(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            WIRE_TAG_TENSOR => Ok(SwarmRequest::TensorPayload(buf)),
            WIRE_TAG_TENSOR_COMPRESSED => Ok(SwarmRequest::TensorPayload(
                decompress_tensor_payload(buf, "request")?,
            )),
            WIRE_TAG_RELAYED_TENSOR => {
                Ok(SwarmRequest::RelayedTensor(decode_relayed_tensor(&buf)?))
            }
            unknown => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown wire tag: 0x{:02x}", unknown),
            )),
        }
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let (tag, buf) = read_wire_frame(io, "read_response", RESPONSE_LARGE_TAGS).await?;
        tracing::trace!(tag, len = buf.len(), "DIAG: codec read_response done");

        match tag {
            WIRE_TAG_JSON => serde_json::from_slice(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            WIRE_TAG_SHARD => {
                // Binary shard frame: first 8 bytes = total_size (little-endian u64), rest = data
                if buf.len() < 8 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Shard frame too short",
                    ));
                }
                let total_size = u64::from_le_bytes([
                    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
                ]);
                let data = buf[8..].to_vec();
                Ok(SwarmResponse::ShardData(crate::types::ShardResponse {
                    data,
                    total_size,
                }))
            }
            WIRE_TAG_TENSOR => Ok(SwarmResponse::TensorPayload(buf)),
            WIRE_TAG_TENSOR_COMPRESSED => Ok(SwarmResponse::TensorPayload(
                decompress_tensor_payload(buf, "response")?,
            )),
            WIRE_TAG_PREFIX_KV => {
                // Layout: [16B request_id][1B flag][optional data...]
                if buf.len() < 17 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "PrefixKv frame too short",
                    ));
                }
                let mut uuid_bytes = [0u8; 16];
                uuid_bytes.copy_from_slice(&buf[..16]);
                let request_id = uuid::Uuid::from_bytes(uuid_bytes);
                let flag = buf[16];
                let payload = match flag {
                    0 => None,
                    1 => Some(buf[17..].to_vec()),
                    2 => Some(compression::decompress_tensor(&buf[17..]).map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("PrefixKv zstd decompress: {e}"),
                        )
                    })?),
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Unknown PrefixKv flag: {other}"),
                        ))
                    }
                };
                Ok(SwarmResponse::PrefixKvData(PrefixKvDataResp {
                    request_id,
                    payload,
                }))
            }
            unknown => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown wire tag: 0x{:02x}", unknown),
            )),
        }
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // Build the complete frame in a single buffer before writing.
        // Quinn's QUIC stream has no BufWriter and flush() is a no-op,
        // so a single write_all() is more reliable than multiple small writes.
        let frame = match req {
            SwarmRequest::TensorPayload(payload) => compression::build_tensor_frame(
                &payload,
                self.compress_tensors,
                self.compress_level,
                self.compress_threshold,
            ),
            SwarmRequest::RelayedTensor(rt) => build_relayed_tensor_frame(&rt)?,
            other => build_json_frame(&other, "message")?,
        };
        let frame_len = frame.len();
        tracing::trace!(frame_len, "DIAG: codec write_request start");
        io.write_all(&frame).await?;
        tracing::trace!(frame_len, "DIAG: codec write_request done");
        // Do NOT call io.close() here — the request_response handler manages stream
        // lifecycle (close + read_response). Closing in the codec corrupts the QUIC
        // stream state and silently prevents message delivery.
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // Single write_all — see write_request comment.
        let frame = match resp {
            SwarmResponse::TensorPayload(payload) => compression::build_tensor_frame(
                &payload,
                self.compress_tensors,
                self.compress_level,
                self.compress_threshold,
            ),
            SwarmResponse::ShardData(ref shard) => {
                // Binary shard frame: [tag][4B payload_len_be][8B total_size_le][data...]
                let payload_len = 8 + shard.data.len();
                let len_bytes = (payload_len as u32).to_be_bytes();
                let mut frame = Vec::with_capacity(1 + 4 + payload_len);
                frame.push(WIRE_TAG_SHARD);
                frame.extend_from_slice(&len_bytes);
                frame.extend_from_slice(&shard.total_size.to_le_bytes());
                frame.extend_from_slice(&shard.data);
                frame
            }
            SwarmResponse::PrefixKvData(ref resp) => {
                // Binary prefix-KV frame: [tag][4B payload_len_be][16B uuid][1B flag][data...]
                // Compress when configured AND payload is above threshold AND
                // the compressed form is actually smaller; otherwise fall
                // through to the raw flag=1 frame.
                let (flag, body): (u8, std::borrow::Cow<'_, [u8]>) = match resp.payload.as_deref() {
                    None => (0, std::borrow::Cow::Borrowed(&[][..])),
                    Some(raw) => {
                        if self.compress_prefix_kv && raw.len() >= self.compress_threshold {
                            match compression::compress_tensor(raw, self.compress_level) {
                                Ok(c) if c.len() < raw.len() => {
                                    tracing::debug!(
                                        raw_len = raw.len(),
                                        compressed_len = c.len(),
                                        ratio = format_args!(
                                            "{:.2}x",
                                            raw.len() as f64 / c.len().max(1) as f64
                                        ),
                                        "DIAG: PrefixKv frame compressed"
                                    );
                                    (2, std::borrow::Cow::Owned(c))
                                }
                                _ => (1, std::borrow::Cow::Borrowed(raw)),
                            }
                        } else {
                            (1, std::borrow::Cow::Borrowed(raw))
                        }
                    }
                };
                let payload_len = 16 + 1 + body.len();
                let len_bytes = (payload_len as u32).to_be_bytes();
                let mut frame = Vec::with_capacity(1 + 4 + payload_len);
                frame.push(WIRE_TAG_PREFIX_KV);
                frame.extend_from_slice(&len_bytes);
                frame.extend_from_slice(resp.request_id.as_bytes());
                frame.push(flag);
                frame.extend_from_slice(&body);
                frame
            }
            other => build_json_frame(&other, "response")?,
        };
        let frame_len = frame.len();
        tracing::trace!(frame_len, "DIAG: codec write_response start");
        io.write_all(&frame).await?;
        tracing::trace!(frame_len, "DIAG: codec write_response done");
        // Do NOT call io.close() here — the request_response handler manages stream
        // lifecycle. Closing in the codec corrupts QUIC stream state.
        Ok(())
    }
}

/// Encode a SwarmMessage to JSON bytes.
pub fn encode_message(msg: &SwarmMessage) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(msg)
}

/// Decode a SwarmMessage from JSON bytes.
pub fn decode_message(data: &[u8]) -> Result<SwarmMessage, serde_json::Error> {
    serde_json::from_slice(data)
}

// ---- Tensor Encoding (binary wire format) ----

// Binary layout for LayerForward envelope:
//   [0..16]  request_id (UUID bytes)
//   [16..20] sequence_num (u32 LE)
//   [20..24] index_pos (u32 LE)
//   [24]     format tag: 0=FP16, 1=FP32, 2=INT8
//   [25..29] data_len (u32 LE)
//   [29..]   activation data

/// Tensor message type tag — first byte of every tensor protocol message.
pub const TENSOR_TAG_FORWARD: u8 = 0x01;
pub const TENSOR_TAG_RESULT: u8 = 0x02;
/// Encrypted tensor message tag (activations encrypted, header fields are cleartext AAD).
pub const TENSOR_TAG_ENCRYPTED: u8 = 0x10;

// Binary layout for LayerForward envelope (v2 with model_id + required layer_range):
//   [0]        tag = TENSOR_TAG_FORWARD (0x01)
//   [1..17]    request_id (UUID, 16 bytes)
//   [17..21]   sequence_num (u32 LE)
//   [21..25]   index_pos (u32 LE)
//   [25]       format tag: 0=FP16, 1=FP32, 2=INT8
//   [26..30]   data_len (u32 LE)
//   [30..30+N] activation data
//   -- required trailer --
//   [T]        marker = 0x01
//   [T+1..T+5] layer_start (u32 LE)
//   [T+5..T+9] layer_end (u32 LE)
//   [T+9..T+11] model_id_len (u16 LE)
//   [T+11..T+11+M] model_id (UTF-8 string)

mod encrypted;
mod layer_forward;
mod layer_result;

pub use encrypted::{
    build_layer_forward_aad, decode_layer_forward_encrypted, encode_layer_forward_encrypted,
};
pub use layer_forward::{decode_layer_forward, encode_layer_forward};
pub use layer_result::{decode_layer_result, encode_layer_result};

impl serde::Serialize for SwarmRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        #[serde(tag = "type")]
        enum Inner<'a> {
            Message { data: &'a SwarmMessage },
            ShardTransfer { data: &'a ShardRequest },
            PrefixKvFetch { data: &'a PrefixKvFetchReq },
        }
        match self {
            SwarmRequest::Message(m) => Inner::Message { data: m }.serialize(serializer),
            SwarmRequest::ShardTransfer(s) => {
                Inner::ShardTransfer { data: s }.serialize(serializer)
            }
            SwarmRequest::PrefixKvFetch(r) => {
                Inner::PrefixKvFetch { data: r }.serialize(serializer)
            }
            SwarmRequest::TensorPayload(_) => Err(serde::ser::Error::custom(
                "TensorPayload should not be JSON-serialized",
            )),
            SwarmRequest::RelayedTensor(_) => Err(serde::ser::Error::custom(
                "RelayedTensor should not be JSON-serialized",
            )),
        }
    }
}

impl<'de> serde::Deserialize<'de> for SwarmRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(tag = "type")]
        #[allow(clippy::large_enum_variant)]
        enum Inner {
            Message { data: SwarmMessage },
            ShardTransfer { data: ShardRequest },
            PrefixKvFetch { data: PrefixKvFetchReq },
        }
        match Inner::deserialize(deserializer)? {
            Inner::Message { data } => Ok(SwarmRequest::Message(Box::new(data))),
            Inner::ShardTransfer { data } => Ok(SwarmRequest::ShardTransfer(data)),
            Inner::PrefixKvFetch { data } => Ok(SwarmRequest::PrefixKvFetch(data)),
        }
    }
}

impl serde::Serialize for SwarmResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        #[serde(tag = "type")]
        enum Inner<'a> {
            Message { data: &'a SwarmMessage },
            ShardData { data: &'a ShardResponse },
            Ack,
        }
        match self {
            SwarmResponse::Message(m) => Inner::Message { data: m }.serialize(serializer),
            SwarmResponse::ShardData(s) => Inner::ShardData { data: s }.serialize(serializer),
            SwarmResponse::Ack => Inner::Ack.serialize(serializer),
            SwarmResponse::TensorPayload(_) | SwarmResponse::PrefixKvData(_) => Err(
                serde::ser::Error::custom("binary response variants should not be JSON-serialized"),
            ),
        }
    }
}

impl<'de> serde::Deserialize<'de> for SwarmResponse {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(tag = "type")]
        #[allow(clippy::large_enum_variant)]
        enum Inner {
            Message { data: SwarmMessage },
            ShardData { data: ShardResponse },
            Ack,
        }
        match Inner::deserialize(deserializer)? {
            Inner::Message { data } => Ok(SwarmResponse::Message(Box::new(data))),
            Inner::ShardData { data } => Ok(SwarmResponse::ShardData(data)),
            Inner::Ack => Ok(SwarmResponse::Ack),
        }
    }
}

// ---- Tensor Compression (zstd) ----

mod compression {
    use super::{WIRE_TAG_TENSOR, WIRE_TAG_TENSOR_COMPRESSED};

    /// Compress raw tensor bytes with zstd at the given level.
    pub fn compress_tensor(data: &[u8], level: i32) -> Result<Vec<u8>, String> {
        zstd::bulk::compress(data, level).map_err(|e| format!("zstd compress: {e}"))
    }

    /// Decompress zstd-compressed tensor bytes.
    pub fn decompress_tensor(data: &[u8]) -> Result<Vec<u8>, String> {
        // Cap decompressed size at 256 MB to prevent zip-bomb attacks.
        const MAX_DECOMPRESSED: usize = 256 * 1024 * 1024;
        zstd::bulk::decompress(data, MAX_DECOMPRESSED).map_err(|e| format!("zstd decompress: {e}"))
    }

    /// Build a wire frame for a tensor payload, optionally compressing it.
    /// Returns the complete frame bytes (tag + length + payload).
    pub fn build_tensor_frame(
        payload: &[u8],
        compress: bool,
        level: i32,
        threshold: usize,
    ) -> Vec<u8> {
        if compress && payload.len() >= threshold {
            if let Ok(compressed) = compress_tensor(payload, level) {
                // Only use compressed form if it's actually smaller.
                if compressed.len() < payload.len() {
                    let len = (compressed.len() as u32).to_be_bytes();
                    let mut frame = Vec::with_capacity(1 + 4 + compressed.len());
                    frame.push(WIRE_TAG_TENSOR_COMPRESSED);
                    frame.extend_from_slice(&len);
                    frame.extend_from_slice(&compressed);
                    return frame;
                }
            }
        }
        // Fallback: uncompressed tensor frame.
        let len = (payload.len() as u32).to_be_bytes();
        let mut frame = Vec::with_capacity(1 + 4 + payload.len());
        frame.push(WIRE_TAG_TENSOR);
        frame.extend_from_slice(&len);
        frame.extend_from_slice(payload);
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::compression::{compress_tensor, decompress_tensor};
    use super::*;

    #[test]
    fn relayed_tensor_encode_decode_roundtrip() {
        let rt = RelayedTensor {
            relay_to: crate::types::NodeId([7u8; 32]),
            origin: crate::types::NodeId([9u8; 32]),
            request_id: uuid::Uuid::from_bytes([3u8; 16]),
            is_result: true,
            ephemeral_pub: [5u8; 32],
            sealed: vec![0xAB; 5000],
        };
        let body = encode_relayed_tensor(&rt);
        assert_eq!(body.len(), RELAYED_TENSOR_HEADER_LEN + 5000);
        let got = decode_relayed_tensor(&body).unwrap();
        assert_eq!(got.relay_to.0, rt.relay_to.0);
        assert_eq!(got.origin.0, rt.origin.0);
        assert_eq!(got.request_id, rt.request_id);
        assert!(got.is_result);
        assert_eq!(got.ephemeral_pub, rt.ephemeral_pub);
        assert_eq!(got.sealed, rt.sealed);
    }

    #[test]
    fn relayed_tensor_decode_rejects_short_frame() {
        assert!(decode_relayed_tensor(&[0u8; 10]).is_err());
    }

    #[test]
    fn relayed_tensor_is_in_request_large_tag_set() {
        // A relayed tensor carries a full activation forward — it MUST get the
        // MAX_MESSAGE_SIZE frame limit, not the small JSON cap, or a large
        // prefill forward would be rejected as oversized on the receiver.
        // Guards against a future edit silently dropping the tag.
        assert!(REQUEST_LARGE_TAGS.contains(&WIRE_TAG_RELAYED_TENSOR));
        assert!(REQUEST_LARGE_TAGS.contains(&WIRE_TAG_TENSOR));
        assert!(REQUEST_LARGE_TAGS.contains(&WIRE_TAG_TENSOR_COMPRESSED));
        // RelayedTensor is only ever a SwarmRequest, never a response.
        assert!(!RESPONSE_LARGE_TAGS.contains(&WIRE_TAG_RELAYED_TENSOR));
        // Shard + prefix-KV are the response-only large frames.
        assert!(RESPONSE_LARGE_TAGS.contains(&WIRE_TAG_SHARD));
        assert!(RESPONSE_LARGE_TAGS.contains(&WIRE_TAG_PREFIX_KV));
    }

    #[test]
    fn build_relayed_tensor_frame_layout() {
        // The send-side framing counterpart of the codec read path:
        // [WIRE_TAG_RELAYED_TENSOR][4B BE length][body], and the body decodes
        // back to the same struct.
        let rt = RelayedTensor {
            relay_to: crate::types::NodeId([1u8; 32]),
            origin: crate::types::NodeId([2u8; 32]),
            request_id: uuid::Uuid::from_bytes([4u8; 16]),
            is_result: false,
            ephemeral_pub: [6u8; 32],
            sealed: vec![0xCD; 1234],
        };
        let frame = build_relayed_tensor_frame(&rt).unwrap();
        assert_eq!(frame[0], WIRE_TAG_RELAYED_TENSOR);
        let declared = u32::from_be_bytes(frame[1..5].try_into().unwrap()) as usize;
        assert_eq!(declared, frame.len() - 5);
        let decoded = decode_relayed_tensor(&frame[5..]).unwrap();
        assert_eq!(decoded.sealed, rt.sealed);
        assert_eq!(decoded.origin.0, rt.origin.0);
        assert!(!decoded.is_result);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let msg = SwarmMessage::HealthPing {
            nonce: 42,
            timestamp: 1000,
            node_id: None,
            active_request_count: 3,
        };
        let encoded = encode_message(&msg).unwrap();
        let decoded = decode_message(&encoded).unwrap();
        match decoded {
            SwarmMessage::HealthPing {
                nonce,
                timestamp,
                active_request_count,
                ..
            } => {
                assert_eq!(nonce, 42);
                assert_eq!(timestamp, 1000);
                assert_eq!(active_request_count, 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_serde_roundtrip() {
        let req = SwarmRequest::Message(Box::new(SwarmMessage::HealthPing {
            nonce: 1,
            timestamp: 2,
            node_id: None,
            active_request_count: 0,
        }));
        let json = serde_json::to_string(&req).unwrap();
        let parsed: SwarmRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            SwarmRequest::Message(msg) => match *msg {
                SwarmMessage::HealthPing { nonce, .. } => {
                    assert_eq!(nonce, 1);
                }
                _ => panic!("wrong variant"),
            },
            _ => panic!("wrong variant"),
        }
    }

    fn test_model_id() -> ModelId {
        ModelId("test-model".into())
    }

    #[test]
    fn layer_forward_encode_decode_roundtrip() {
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 42,
            index_pos: 0,
            activations: vec![1, 2, 3, 4, 5, 6, 7, 8],
            format: TensorFormat::FP16,
            model_id: test_model_id(),
            layer_range: (0, 4),
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
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();

        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 42);
        assert_eq!(decoded.activations, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(decoded.format, TensorFormat::FP16));
        assert_eq!(decoded.layer_range, (0, 4));
        assert_eq!(decoded.model_id, test_model_id());
    }

    #[test]
    fn layer_forward_all_formats() {
        for (fmt, tag) in [
            (TensorFormat::FP16, 0u8),
            (TensorFormat::FP32, 1u8),
            (TensorFormat::INT8, 2u8),
        ] {
            let forward = LayerForward {
                request_id: uuid::Uuid::nil(),
                sequence_num: 0,
                index_pos: 0,
                activations: vec![],
                format: fmt,
                model_id: test_model_id(),
                layer_range: (0, 2),
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
            };
            let encoded = encode_layer_forward(&forward).unwrap();
            assert_eq!(encoded[25], tag); // tag(1) + uuid(16) + seq(4) + index_pos(4) = 25
            let decoded = decode_layer_forward(&encoded).unwrap();
            assert!(matches!(
                (&forward.format, &decoded.format),
                (TensorFormat::FP16, TensorFormat::FP16)
                    | (TensorFormat::FP32, TensorFormat::FP32)
                    | (TensorFormat::INT8, TensorFormat::INT8)
            ));
        }
    }

    #[test]
    fn layer_forward_large_payload() {
        let data = vec![0xAB; 1024 * 1024]; // 1MB
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 100,
            index_pos: 0,
            activations: data.clone(),
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (0, 28),
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
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();
        assert_eq!(decoded.activations.len(), 1024 * 1024);
        assert_eq!(decoded.activations, data);
    }

    #[test]
    fn layer_forward_with_layer_range_roundtrip() {
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 7,
            index_pos: 128,
            activations: vec![0xAA; 64],
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (10, 14),
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
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();

        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 7);
        assert_eq!(decoded.index_pos, 128);
        assert_eq!(decoded.layer_range, (10, 14));
        assert_eq!(decoded.model_id, test_model_id());
    }

    #[test]
    fn layer_forward_missing_trailer_rejected() {
        // Messages without trailer should be rejected (no backward compat)
        let forward = LayerForward {
            request_id: uuid::Uuid::nil(),
            sequence_num: 0,
            index_pos: 0,
            activations: vec![1, 2, 3],
            format: TensorFormat::FP16,
            model_id: test_model_id(),
            layer_range: (0, 2),
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
        };
        let encoded = encode_layer_forward(&forward).unwrap();
        // Trim to remove the trailer — simulates an old encoder
        let trimmed = &encoded[..1 + 29 + 3]; // tag + header + 3 bytes data
        let result = decode_layer_forward(trimmed);
        assert!(result.is_err());
    }

    #[test]
    fn layer_result_encode_decode_roundtrip() {
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![100, 200, 300],
            finish_reason: Some(NetworkFinishReason::Stop),
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        };

        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();

        assert_eq!(decoded.request_id, result.request_id);
        assert_eq!(decoded.token_ids, vec![100, 200, 300]);
        assert!(matches!(
            decoded.finish_reason,
            Some(NetworkFinishReason::Stop)
        ));
    }

    #[test]
    fn layer_result_no_finish_reason() {
        let result = LayerResult {
            request_id: uuid::Uuid::nil(),
            token_ids: vec![42],
            finish_reason: None,
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        };

        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        assert!(decoded.finish_reason.is_none());
    }

    #[test]
    fn layer_result_error_with_message() {
        let result = LayerResult {
            request_id: uuid::Uuid::nil(),
            token_ids: vec![],
            finish_reason: Some(NetworkFinishReason::Error("OOM".to_string())),
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        };

        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        match decoded.finish_reason {
            Some(NetworkFinishReason::Error(msg)) => assert_eq!(msg, "OOM"),
            _ => panic!("Expected Error finish reason"),
        }
    }

    #[test]
    fn decode_layer_forward_too_short() {
        assert!(decode_layer_forward(&[0u8; 10]).is_err());
    }

    #[test]
    fn decode_layer_result_too_short() {
        assert!(decode_layer_result(&[0u8; 10]).is_err());
    }

    // ---- Tensor Compression Tests ----

    #[test]
    fn compress_decompress_roundtrip() {
        let data = vec![0xAB; 4096]; // 4KB of repetitive data
        let compressed = compress_tensor(&data, 1).unwrap();
        assert!(
            compressed.len() < data.len(),
            "repetitive data should compress"
        );
        let decompressed = decompress_tensor(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn build_tensor_frame_compresses_above_threshold() {
        let payload = vec![0x42; 2048]; // 2KB, above default 1024 threshold
        let frame = compression::build_tensor_frame(&payload, true, 1, 1024);
        // First byte should be the compressed tag
        assert_eq!(frame[0], WIRE_TAG_TENSOR_COMPRESSED);
        // Total frame should be smaller than uncompressed (1+4+2048)
        assert!(frame.len() < 1 + 4 + payload.len());
    }

    #[test]
    fn build_tensor_frame_skips_below_threshold() {
        let payload = vec![0x42; 512]; // 512 bytes, below 1024 threshold
        let frame = compression::build_tensor_frame(&payload, true, 1, 1024);
        // Should use uncompressed tag
        assert_eq!(frame[0], WIRE_TAG_TENSOR);
        assert_eq!(frame.len(), 1 + 4 + payload.len());
    }

    #[test]
    fn build_tensor_frame_skips_when_disabled() {
        let payload = vec![0x42; 4096];
        let frame = compression::build_tensor_frame(&payload, false, 1, 1024);
        assert_eq!(frame[0], WIRE_TAG_TENSOR);
        assert_eq!(frame.len(), 1 + 4 + payload.len());
    }

    #[test]
    fn build_tensor_frame_skips_when_compressed_is_larger() {
        // Random data typically doesn't compress well
        let mut payload = vec![0u8; 2048];
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(17).wrapping_add(37) % 256) as u8;
        }
        let frame = compression::build_tensor_frame(&payload, true, 1, 1024);
        // Should fall back to uncompressed if zstd can't shrink it
        // (or compressed — either is fine as long as roundtrip works)
        let tag = frame[0];
        assert!(tag == WIRE_TAG_TENSOR || tag == WIRE_TAG_TENSOR_COMPRESSED);
    }

    #[test]
    fn codec_tensor_frame_roundtrip() {
        // Simulate the full codec write → read cycle for a compressed tensor
        let payload = vec![0xAA; 8192]; // 8KB repetitive
        let frame = compression::build_tensor_frame(&payload, true, 1, 1024);
        assert_eq!(frame[0], WIRE_TAG_TENSOR_COMPRESSED);

        // Parse: tag(1) + len(4) + data
        let tag = frame[0];
        let len = u32::from_be_bytes(frame[1..5].try_into().unwrap()) as usize;
        let data = &frame[5..5 + len];

        let recovered = match tag {
            WIRE_TAG_TENSOR => data.to_vec(),
            WIRE_TAG_TENSOR_COMPRESSED => decompress_tensor(data).unwrap(),
            _ => panic!("unexpected tag"),
        };
        assert_eq!(recovered, payload);
    }

    #[test]
    fn compressed_layer_forward_roundtrip() {
        // Encode a LayerForward as tensor payload, compress, decompress, decode
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 99,
            index_pos: 512,
            activations: vec![0xBB; 4096],
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (2, 8),
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
        };
        let encoded = encode_layer_forward(&forward).unwrap();

        // Compress
        let compressed = compress_tensor(&encoded, 1).unwrap();
        assert!(compressed.len() < encoded.len());

        // Decompress
        let decompressed = decompress_tensor(&compressed).unwrap();
        assert_eq!(decompressed, encoded);

        // Decode back
        let decoded = decode_layer_forward(&decompressed).unwrap();
        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 99);
        assert_eq!(decoded.index_pos, 512);
        assert_eq!(decoded.activations.len(), 4096);
        assert_eq!(decoded.layer_range, (2, 8));
        assert_eq!(decoded.model_id, test_model_id());
    }

    #[test]
    fn layer_forward_speculative_trailer_roundtrip() {
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 7,
            index_pos: 100,
            activations: vec![0; 8],
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (0, 32),
            tp_meta: None,
            vision_embeddings: None,
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: vec![42, 137, 9000, 123456],
            spec_logits_requested: true,
            truncate_kv_to: None,
            chunk_meta: None,
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();
        assert_eq!(decoded.draft_tokens, vec![42, 137, 9000, 123456]);
        assert!(decoded.spec_logits_requested);
    }

    #[test]
    fn layer_forward_spec_logits_requested_round_trips_with_empty_drafts() {
        // The DSD verify path (`build_spec_verify_forward`) packs the
        // verify token IDs into `activations` and leaves `draft_tokens`
        // empty — the worker only needs `spec_logits_requested` as the
        // gate. The encoder MUST emit the 0x03 trailer in this case so
        // the flag survives the round-trip; the previous "skip trailer
        // when draft_tokens empty" gate silently broke DSD-with-remote.
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 0,
            index_pos: 0,
            activations: vec![0; 8],
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings: None,
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: vec![],
            spec_logits_requested: true,
            truncate_kv_to: None,
            chunk_meta: None,
        };
        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();
        assert!(decoded.draft_tokens.is_empty());
        assert!(
            decoded.spec_logits_requested,
            "spec_logits_requested must survive round-trip even when draft_tokens is empty"
        );
    }

    #[test]
    fn layer_forward_no_speculative_trailer_when_both_empty() {
        // When BOTH draft_tokens is empty AND spec_logits_requested is
        // false, no trailer should be emitted.
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 0,
            index_pos: 0,
            activations: vec![0; 8],
            format: TensorFormat::FP32,
            model_id: test_model_id(),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings: None,
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: vec![],
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
        };
        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();
        assert!(decoded.draft_tokens.is_empty());
        assert!(!decoded.spec_logits_requested);
    }

    #[test]
    fn layer_result_speculative_logits_roundtrip() {
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![],
            finish_reason: None,
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: vec![
                vec![1.0, 2.0, 3.0],
                vec![-1.5, 0.0, 2.5, 100.0],
                vec![0.0; 16],
            ],
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        };
        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        assert_eq!(decoded.spec_logits.len(), 3);
        assert_eq!(decoded.spec_logits[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(decoded.spec_logits[1], vec![-1.5, 0.0, 2.5, 100.0]);
        assert_eq!(decoded.spec_logits[2], vec![0.0; 16]);
    }

    #[test]
    fn layer_result_matched_stop_sequence_roundtrip() {
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![42],
            finish_reason: Some(NetworkFinishReason::Stop),
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: Some("\n\nHuman:".to_string()),
            token_logprobs: Vec::new(),
        };
        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        assert_eq!(decoded.matched_stop_sequence.as_deref(), Some("\n\nHuman:"));
    }

    #[test]
    fn layer_result_token_logprobs_roundtrip() {
        let entries = vec![
            swarmllm_types::TokenLogProbEntry {
                token: "hello".to_string(),
                logprob: -0.5,
                top_logprobs: vec![("hi".to_string(), -1.2)],
            },
            swarmllm_types::TokenLogProbEntry {
                token: " world".to_string(),
                logprob: -2.3,
                top_logprobs: Vec::new(),
            },
        ];
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![1, 2],
            finish_reason: None,
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: entries.clone(),
        };
        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        assert_eq!(decoded.token_logprobs.len(), 2);
        assert_eq!(decoded.token_logprobs[0].token, "hello");
        assert!((decoded.token_logprobs[0].logprob - (-0.5)).abs() < 1e-6);
        assert_eq!(
            decoded.token_logprobs[0].top_logprobs,
            vec![("hi".to_string(), -1.2)]
        );
        assert_eq!(decoded.token_logprobs[1].token, " world");
    }

    #[test]
    fn layer_result_old_decoder_skips_unknown_trailers() {
        // Encode a NEW-style result with both trailers; decode it. Verifies
        // the round-trip works even when both new optional trailers are
        // present alongside spec_logits.
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![7],
            finish_reason: Some(NetworkFinishReason::Stop),
            activations: vec![],
            sealed_token_ids: None,
            spec_logits: vec![vec![0.1, 0.2, 0.3]],
            matched_stop_sequence: Some("STOP".to_string()),
            token_logprobs: vec![swarmllm_types::TokenLogProbEntry {
                token: "x".to_string(),
                logprob: -1.0,
                top_logprobs: Vec::new(),
            }],
        };
        let encoded = encode_layer_result(&result).unwrap();
        let decoded = decode_layer_result(&encoded).unwrap();
        assert_eq!(decoded.spec_logits.len(), 1);
        assert_eq!(decoded.matched_stop_sequence.as_deref(), Some("STOP"));
        assert_eq!(decoded.token_logprobs.len(), 1);
    }

    /// Build a PrefixKv response frame the same way `write_response` does
    /// (test helper, mirrors the production write path) so we can exercise
    /// the codec from both ends without standing up a libp2p stream.
    fn build_prefix_kv_frame(
        request_id: uuid::Uuid,
        payload: Option<&[u8]>,
        compress: bool,
        level: i32,
        threshold: usize,
    ) -> Vec<u8> {
        let (flag, body): (u8, std::borrow::Cow<'_, [u8]>) = match payload {
            None => (0, std::borrow::Cow::Borrowed(&[][..])),
            Some(raw) => {
                if compress && raw.len() >= threshold {
                    match compression::compress_tensor(raw, level) {
                        Ok(c) if c.len() < raw.len() => (2, std::borrow::Cow::Owned(c)),
                        _ => (1, std::borrow::Cow::Borrowed(raw)),
                    }
                } else {
                    (1, std::borrow::Cow::Borrowed(raw))
                }
            }
        };
        let payload_len = 16 + 1 + body.len();
        let len_bytes = (payload_len as u32).to_be_bytes();
        let mut frame = Vec::with_capacity(1 + 4 + payload_len);
        frame.push(WIRE_TAG_PREFIX_KV);
        frame.extend_from_slice(&len_bytes);
        frame.extend_from_slice(request_id.as_bytes());
        frame.push(flag);
        frame.extend_from_slice(&body);
        frame
    }

    /// Decode the body of a PrefixKv frame (skipping tag + length header,
    /// matching the read_response branch on WIRE_TAG_PREFIX_KV).
    fn decode_prefix_kv_body(frame: &[u8]) -> (uuid::Uuid, u8, Option<Vec<u8>>) {
        assert_eq!(frame[0], WIRE_TAG_PREFIX_KV);
        let len = u32::from_be_bytes(frame[1..5].try_into().unwrap()) as usize;
        let body = &frame[5..5 + len];
        let mut uuid_bytes = [0u8; 16];
        uuid_bytes.copy_from_slice(&body[..16]);
        let request_id = uuid::Uuid::from_bytes(uuid_bytes);
        let flag = body[16];
        let payload = match flag {
            0 => None,
            1 => Some(body[17..].to_vec()),
            2 => Some(decompress_tensor(&body[17..]).unwrap()),
            other => panic!("unknown PrefixKv flag: {other}"),
        };
        (request_id, flag, payload)
    }

    #[test]
    fn prefix_kv_compresses_when_flag_on() {
        let req_id = uuid::Uuid::new_v4();
        // 8 KB of zero-ish data — the realistic shape (KV padding regions
        // beyond token_count are zero-initialized).
        let payload = vec![0u8; 8192];
        let frame = build_prefix_kv_frame(req_id, Some(&payload), true, 1, 1024);
        let (got_id, flag, recovered) = decode_prefix_kv_body(&frame);
        assert_eq!(got_id, req_id);
        assert_eq!(flag, 2, "should have used compressed flag");
        assert_eq!(recovered.unwrap(), payload);
        // Compressed body is smaller than raw + uuid + flag overhead
        assert!(frame.len() < 1 + 4 + 16 + 1 + payload.len());
    }

    #[test]
    fn prefix_kv_skips_compression_when_flag_off() {
        let req_id = uuid::Uuid::new_v4();
        let payload = vec![0u8; 8192];
        let frame = build_prefix_kv_frame(req_id, Some(&payload), false, 1, 1024);
        let (got_id, flag, recovered) = decode_prefix_kv_body(&frame);
        assert_eq!(got_id, req_id);
        assert_eq!(flag, 1, "should have used raw flag");
        assert_eq!(recovered.unwrap(), payload);
    }

    #[test]
    fn prefix_kv_skips_compression_below_threshold() {
        let req_id = uuid::Uuid::new_v4();
        let payload = vec![0u8; 512];
        let frame = build_prefix_kv_frame(req_id, Some(&payload), true, 1, 1024);
        let (_id, flag, recovered) = decode_prefix_kv_body(&frame);
        assert_eq!(flag, 1, "below threshold should stay raw");
        assert_eq!(recovered.unwrap(), payload);
    }

    #[test]
    fn prefix_kv_falls_back_when_compressed_is_larger() {
        // Random data doesn't compress well — codec should silently fall
        // back to flag=1 rather than emit a larger compressed frame.
        let req_id = uuid::Uuid::new_v4();
        let mut payload = vec![0u8; 2048];
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(31).wrapping_add(7) % 256) as u8;
        }
        let frame = build_prefix_kv_frame(req_id, Some(&payload), true, 1, 1024);
        let (_id, flag, recovered) = decode_prefix_kv_body(&frame);
        // Either path is acceptable as long as the round-trip is exact.
        assert!(flag == 1 || flag == 2);
        assert_eq!(recovered.unwrap(), payload);
    }

    #[test]
    fn prefix_kv_miss_has_no_payload_regardless_of_flag() {
        let req_id = uuid::Uuid::new_v4();
        let frame = build_prefix_kv_frame(req_id, None, true, 1, 1024);
        let (got_id, flag, recovered) = decode_prefix_kv_body(&frame);
        assert_eq!(got_id, req_id);
        assert_eq!(flag, 0, "miss frames always use flag=0");
        assert!(recovered.is_none());
    }
}
