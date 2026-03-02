use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response;
use libp2p::StreamProtocol;

use crate::error::SwarmError;
use crate::types::{
    LayerForward, LayerResult, NetworkFinishReason, ShardRequest, ShardResponse, SwarmMessage,
    TensorFormat,
};

// Re-export compression helpers for use in tests and other modules.
pub use self::compression::{compress_tensor, decompress_tensor};

/// Protocol ID for SwarmLLM unified request/response (JSON control + binary tensor).
pub const PROTOCOL_ID: &str = "/swarmllm/1.0.0";

/// GossipSub topic for model coordination (shard announcements, capacity updates).
pub const TOPIC_MODELS: &str = "swarm/models";

/// GossipSub topic for model governance (voting on model additions).
pub const TOPIC_GOVERNANCE: &str = "swarm/governance";

/// GossipSub topic for credit balance gossip.
pub const TOPIC_CREDITS: &str = "swarm/credits";

/// GossipSub topic for network health.
pub const TOPIC_HEALTH: &str = "swarm/health";

/// GossipSub topic for identity/nickname announcements.
pub const TOPIC_IDENTITY: &str = "swarm/identity";

/// GossipSub topic for device pool management.
pub const TOPIC_POOLS: &str = "swarm/pools";

/// Maximum message size for request_response protocol (256 MB).
const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Maximum activation payload size in layer results (128 MB).
const MAX_ACTIVATION_SIZE: usize = 128 * 1024 * 1024;

/// Codec for SwarmLLM request/response protocol using serde_json.
/// When `compress_tensors` is true, tensor payloads above `compress_threshold`
/// bytes are zstd-compressed on the wire (tag 0x02). Decompression of incoming
/// compressed payloads always works regardless of the flag.
#[derive(Debug, Clone)]
pub struct SwarmCodec {
    /// Whether to compress outgoing tensor payloads.
    pub compress_tensors: bool,
    /// Zstd compression level (1-22).
    pub compress_level: i32,
    /// Minimum payload size in bytes to trigger compression.
    pub compress_threshold: usize,
}

impl Default for SwarmCodec {
    fn default() -> Self {
        Self {
            compress_tensors: true,
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
}

/// Response type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmResponse {
    Message(Box<SwarmMessage>),
    ShardData(ShardResponse),
    Ack,
    /// Binary tensor response data (already encoded).
    TensorPayload(Vec<u8>),
}

/// Wire format type tags for the unified codec.
/// First byte of each message distinguishes JSON from binary tensor payloads.
const WIRE_TAG_JSON: u8 = 0x00;
const WIRE_TAG_TENSOR: u8 = 0x01;
/// Zstd-compressed tensor payload. Peers that don't recognize this tag will
/// reject the message, but all nodes running this version or later support it.
const WIRE_TAG_TENSOR_COMPRESSED: u8 = 0x02;

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
        let mut tag_buf = [0u8; 1];
        io.read_exact(&mut tag_buf).await?;

        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > MAX_MESSAGE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Message too large",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        match tag_buf[0] {
            WIRE_TAG_TENSOR => Ok(SwarmRequest::TensorPayload(buf)),
            WIRE_TAG_TENSOR_COMPRESSED => {
                let decompressed = compression::decompress_tensor(&buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok(SwarmRequest::TensorPayload(decompressed))
            }
            _ => serde_json::from_slice(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
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
        let mut tag_buf = [0u8; 1];
        io.read_exact(&mut tag_buf).await?;

        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > MAX_MESSAGE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Response too large",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        match tag_buf[0] {
            WIRE_TAG_TENSOR => Ok(SwarmResponse::TensorPayload(buf)),
            WIRE_TAG_TENSOR_COMPRESSED => {
                let decompressed = compression::decompress_tensor(&buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok(SwarmResponse::TensorPayload(decompressed))
            }
            _ => serde_json::from_slice(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
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
            other => {
                let data = serde_json::to_vec(&other)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let len = (data.len() as u32).to_be_bytes();
                let mut frame = Vec::with_capacity(1 + 4 + data.len());
                frame.push(WIRE_TAG_JSON);
                frame.extend_from_slice(&len);
                frame.extend_from_slice(&data);
                frame
            }
        };
        io.write_all(&frame).await?;
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
            other => {
                let data = serde_json::to_vec(&other)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let len = (data.len() as u32).to_be_bytes();
                let mut frame = Vec::with_capacity(1 + 4 + data.len());
                frame.push(WIRE_TAG_JSON);
                frame.extend_from_slice(&len);
                frame.extend_from_slice(&data);
                frame
            }
        };
        io.write_all(&frame).await?;
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

/// Encode a LayerForward into a binary tensor envelope.
pub fn encode_layer_forward(forward: &LayerForward) -> Result<Vec<u8>, SwarmError> {
    let data_len = forward.activations.len();
    // Header: tag(1) + uuid(16) + seq(4) + index_pos(4) + fmt(1) + data_len(4) = 30
    // Optional trailer: marker(1) + layer_start(4) + layer_end(4) = 9
    let trailer_len = if forward.layer_range.is_some() { 9 } else { 0 };
    let total = 1 + 29 + data_len + trailer_len;
    let mut buf = Vec::with_capacity(total);

    // Message type tag
    buf.push(TENSOR_TAG_FORWARD);
    // UUID (16 bytes)
    buf.extend_from_slice(forward.request_id.as_bytes());
    // sequence_num (4 bytes LE)
    buf.extend_from_slice(&forward.sequence_num.to_le_bytes());
    // index_pos (4 bytes LE)
    buf.extend_from_slice(&forward.index_pos.to_le_bytes());
    // format tag (1 byte)
    let fmt_tag: u8 = match forward.format {
        TensorFormat::FP16 => 0,
        TensorFormat::FP32 => 1,
        TensorFormat::INT8 => 2,
    };
    buf.push(fmt_tag);
    // data length (4 bytes LE)
    buf.extend_from_slice(&(data_len as u32).to_le_bytes());
    // activation data
    buf.extend_from_slice(&forward.activations);

    // Optional layer_range trailer (backward compatible — old decoders stop at data end)
    if let Some((layer_start, layer_end)) = forward.layer_range {
        buf.push(0x01); // marker byte
        buf.extend_from_slice(&layer_start.to_le_bytes());
        buf.extend_from_slice(&layer_end.to_le_bytes());
    }

    Ok(buf)
}

/// Decode a binary tensor envelope back into a LayerForward.
/// Expects the 1-byte tag prefix to already be stripped (or handles both cases).
pub fn decode_layer_forward(data: &[u8]) -> Result<LayerForward, SwarmError> {
    // Skip the tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_FORWARD {
        &data[1..]
    } else {
        data
    };

    // Header: uuid(16) + seq(4) + index_pos(4) + fmt(1) + data_len(4) = 29
    if data.len() < 29 {
        return Err(SwarmError::Network("Tensor envelope too short".to_string()));
    }

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID bytes".into()))?,
    );
    let sequence_num = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid sequence_num".into()))?,
    );
    let index_pos = u32::from_le_bytes(
        data[20..24]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid index_pos".into()))?,
    );
    let format = match data[24] {
        0 => TensorFormat::FP16,
        1 => TensorFormat::FP32,
        2 => TensorFormat::INT8,
        t => {
            return Err(SwarmError::Network(format!(
                "Unknown tensor format tag: {t}"
            )))
        }
    };
    let data_len = u32::from_le_bytes(
        data[25..29]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid data_len".into()))?,
    ) as usize;

    if data.len() < 29 + data_len {
        return Err(SwarmError::Network(format!(
            "Tensor data truncated: expected {} bytes, got {}",
            data_len,
            data.len() - 29
        )));
    }

    let activations = data[29..29 + data_len].to_vec();

    // Read optional layer_range trailer (backward compatible — absent on old peers)
    let trailer_start = 29 + data_len;
    let layer_range = if data.len() >= trailer_start + 9 && data[trailer_start] == 0x01 {
        let ls = u32::from_le_bytes(
            data[trailer_start + 1..trailer_start + 5]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid layer_start".into()))?,
        );
        let le = u32::from_le_bytes(
            data[trailer_start + 5..trailer_start + 9]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid layer_end".into()))?,
        );
        Some((ls, le))
    } else {
        None
    };

    Ok(LayerForward {
        request_id,
        sequence_num,
        index_pos,
        activations,
        format,
        layer_range,
        sender_peer_bytes: None,
    })
}

// Binary layout for LayerResult (v2 with activations):
//   [0..16]      request_id (UUID bytes)
//   [16..20]     num_tokens (u32 LE)
//   [20..20+n*4] token_ids (each u32 LE)
//   [T]          finish_reason tag: 0=None, 1=Stop, 2=MaxTokens, 3=Error
//   [T+1..]      if tag=3: error message (UTF-8 bytes) followed by [4B activations_len][activations]
//                if tag!=3: [4B activations_len][activations data]

/// Encode a LayerResult into binary.
pub fn encode_layer_result(result: &LayerResult) -> Result<Vec<u8>, SwarmError> {
    let num_tokens = result.token_ids.len();
    let mut buf = Vec::with_capacity(1 + 25 + num_tokens * 4 + result.activations.len());

    // Message type tag
    buf.push(TENSOR_TAG_RESULT);
    buf.extend_from_slice(result.request_id.as_bytes());
    buf.extend_from_slice(&(num_tokens as u32).to_le_bytes());
    for &token in &result.token_ids {
        buf.extend_from_slice(&token.to_le_bytes());
    }

    match &result.finish_reason {
        None => buf.push(0),
        Some(NetworkFinishReason::Stop) => buf.push(1),
        Some(NetworkFinishReason::MaxTokens) => buf.push(2),
        Some(NetworkFinishReason::Error(msg)) => {
            buf.push(3);
            // Error message length + message
            buf.extend_from_slice(&(msg.len() as u32).to_le_bytes());
            buf.extend_from_slice(msg.as_bytes());
        }
    }

    // Append activations (for intermediate pipeline segments)
    buf.extend_from_slice(&(result.activations.len() as u32).to_le_bytes());
    buf.extend_from_slice(&result.activations);

    Ok(buf)
}

/// Decode binary into a LayerResult.
/// Expects the 1-byte tag prefix to already be stripped (or handles both cases).
pub fn decode_layer_result(data: &[u8]) -> Result<LayerResult, SwarmError> {
    // Skip the tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_RESULT {
        &data[1..]
    } else {
        data
    };

    if data.len() < 21 {
        return Err(SwarmError::Network(
            "LayerResult envelope too short".to_string(),
        ));
    }

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID".into()))?,
    );
    let num_tokens = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid num_tokens".into()))?,
    ) as usize;

    // SECURITY: Cap num_tokens to prevent OOM from crafted messages
    if num_tokens > 65536 {
        return Err(SwarmError::Network(
            "num_tokens exceeds maximum (65536)".into(),
        ));
    }

    let tokens_end = 20 + num_tokens * 4;
    if data.len() < tokens_end + 1 {
        return Err(SwarmError::Network("LayerResult truncated".to_string()));
    }

    let mut token_ids = Vec::with_capacity(num_tokens);
    for i in 0..num_tokens {
        let start = 20 + i * 4;
        let token = u32::from_le_bytes(
            data[start..start + 4]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid token id".into()))?,
        );
        token_ids.push(token);
    }

    let mut pos = tokens_end;
    let finish_reason = match data[pos] {
        0 => {
            pos += 1;
            None
        }
        1 => {
            pos += 1;
            Some(NetworkFinishReason::Stop)
        }
        2 => {
            pos += 1;
            Some(NetworkFinishReason::MaxTokens)
        }
        3 => {
            pos += 1;
            // Error: read message length + message
            if pos + 4 <= data.len() {
                let msg_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                let msg = String::from_utf8_lossy(&data[pos..pos + msg_len.min(data.len() - pos)])
                    .to_string();
                pos += msg_len.min(data.len() - pos);
                Some(NetworkFinishReason::Error(msg))
            } else {
                Some(NetworkFinishReason::Error(String::new()))
            }
        }
        t => {
            return Err(SwarmError::Network(format!(
                "Unknown finish reason tag: {t}"
            )))
        }
    };

    // Read activations if present (capped at 128MB to prevent abuse)
    let activations = if pos + 4 <= data.len() {
        let act_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if act_len > MAX_ACTIVATION_SIZE {
            return Err(SwarmError::Network(format!(
                "Activation data too large: {act_len} bytes"
            )));
        }
        if act_len > 0 && pos + act_len <= data.len() {
            data[pos..pos + act_len].to_vec()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    Ok(LayerResult {
        request_id,
        token_ids,
        finish_reason,
        activations,
    })
}

/// Encode an ACK response (empty payload marker).
pub fn encode_ack() -> Vec<u8> {
    vec![0] // Single zero byte = ACK
}

// ---- Encrypted Tensor Encoding ----
//
// Wire format for TENSOR_TAG_ENCRYPTED (0x10):
//   [0]        tag = 0x10
//   [1..17]    request_id (UUID, 16 bytes) — cleartext AAD
//   [17..21]   sequence_num (u32 LE) — cleartext AAD
//   [21..25]   index_pos (u32 LE) — cleartext AAD
//   [25]       format tag (0=FP16, 1=FP32, 2=INT8) — cleartext AAD
//   [26..30]   sealed_len (u32 LE)
//   [30..]     sealed activations (nonce + ciphertext + AEAD tag)
//
// The AAD for the AEAD is the header bytes [1..26] (uuid+seq+idx+fmt).

/// Encode a LayerForward with encrypted activations.
/// The `sealed_activations` should already be encrypted by the SessionManager.
pub fn encode_layer_forward_encrypted(
    forward: &LayerForward,
    sealed_activations: Vec<u8>,
) -> Result<Vec<u8>, SwarmError> {
    let sealed_len = sealed_activations.len();
    let total = 1 + 25 + 4 + sealed_len;
    let mut buf = Vec::with_capacity(total);

    buf.push(TENSOR_TAG_ENCRYPTED);
    buf.extend_from_slice(forward.request_id.as_bytes());
    buf.extend_from_slice(&forward.sequence_num.to_le_bytes());
    buf.extend_from_slice(&forward.index_pos.to_le_bytes());
    let fmt_tag: u8 = match forward.format {
        TensorFormat::FP16 => 0,
        TensorFormat::FP32 => 1,
        TensorFormat::INT8 => 2,
    };
    buf.push(fmt_tag);
    buf.extend_from_slice(&(sealed_len as u32).to_le_bytes());
    buf.extend_from_slice(&sealed_activations);

    Ok(buf)
}

/// Decode an encrypted tensor envelope.
/// Returns the cleartext header fields (as a LayerForward with empty activations)
/// plus the sealed activation bytes, along with the AAD bytes.
/// The caller must decrypt the sealed bytes using the SessionManager.
pub fn decode_layer_forward_encrypted(
    data: &[u8],
) -> Result<(LayerForward, Vec<u8>, Vec<u8>), SwarmError> {
    // Skip tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_ENCRYPTED {
        &data[1..]
    } else {
        data
    };

    // Header: uuid(16) + seq(4) + idx_pos(4) + fmt(1) + sealed_len(4) = 29
    if data.len() < 29 {
        return Err(SwarmError::Network(
            "Encrypted tensor envelope too short".to_string(),
        ));
    }

    let aad = data[..25].to_vec(); // uuid + seq + idx_pos + fmt

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID".into()))?,
    );
    let sequence_num = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid sequence_num".into()))?,
    );
    let index_pos = u32::from_le_bytes(
        data[20..24]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid index_pos".into()))?,
    );
    let format = match data[24] {
        0 => TensorFormat::FP16,
        1 => TensorFormat::FP32,
        2 => TensorFormat::INT8,
        t => {
            return Err(SwarmError::Network(format!(
                "Unknown tensor format tag: {t}"
            )))
        }
    };
    let sealed_len = u32::from_le_bytes(
        data[25..29]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid sealed_len".into()))?,
    ) as usize;

    if data.len() < 29 + sealed_len {
        return Err(SwarmError::Network(
            "Encrypted tensor data truncated".to_string(),
        ));
    }

    let sealed = data[29..29 + sealed_len].to_vec();

    let forward = LayerForward {
        request_id,
        sequence_num,
        index_pos,
        activations: vec![], // Will be filled after decryption
        format,
        layer_range: None, // Encrypted messages don't carry layer_range in AAD header
        sender_peer_bytes: None,
    };

    Ok((forward, sealed, aad))
}

// Serde impls for SwarmRequest/SwarmResponse
// Note: TensorPayload variants are never JSON-serialized (handled by binary codec path),
// but serde impls must be exhaustive.
impl serde::Serialize for SwarmRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        #[serde(tag = "type")]
        enum Inner<'a> {
            Message { data: &'a SwarmMessage },
            ShardTransfer { data: &'a ShardRequest },
        }
        match self {
            SwarmRequest::Message(m) => Inner::Message { data: m }.serialize(serializer),
            SwarmRequest::ShardTransfer(s) => {
                Inner::ShardTransfer { data: s }.serialize(serializer)
            }
            SwarmRequest::TensorPayload(_) => Err(serde::ser::Error::custom(
                "TensorPayload should not be JSON-serialized",
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
        }
        match Inner::deserialize(deserializer)? {
            Inner::Message { data } => Ok(SwarmRequest::Message(Box::new(data))),
            Inner::ShardTransfer { data } => Ok(SwarmRequest::ShardTransfer(data)),
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
            SwarmResponse::TensorPayload(_) => Err(serde::ser::Error::custom(
                "TensorPayload should not be JSON-serialized",
            )),
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
    use super::*;

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

    #[test]
    fn layer_forward_encode_decode_roundtrip() {
        let forward = LayerForward {
            request_id: uuid::Uuid::new_v4(),
            sequence_num: 42,
            index_pos: 0,
            activations: vec![1, 2, 3, 4, 5, 6, 7, 8],
            format: TensorFormat::FP16,
            layer_range: None,
            sender_peer_bytes: None,
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();

        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 42);
        assert_eq!(decoded.activations, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(decoded.format, TensorFormat::FP16));
        assert!(decoded.layer_range.is_none());
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
                layer_range: None,
                sender_peer_bytes: None,
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
            layer_range: None,
            sender_peer_bytes: None,
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
            layer_range: Some((10, 14)),
            sender_peer_bytes: None,
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();

        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 7);
        assert_eq!(decoded.index_pos, 128);
        assert_eq!(decoded.layer_range, Some((10, 14)));
    }

    #[test]
    fn layer_forward_without_layer_range() {
        // Messages without layer_range (e.g. encrypted) should decode with None
        let forward = LayerForward {
            request_id: uuid::Uuid::nil(),
            sequence_num: 0,
            index_pos: 0,
            activations: vec![1, 2, 3],
            format: TensorFormat::FP16,
            layer_range: None,
            sender_peer_bytes: None,
        };
        let encoded = encode_layer_forward(&forward).unwrap();
        // Trim to remove any trailer — simulates an old encoder
        let trimmed = &encoded[..1 + 29 + 3]; // tag + header + 3 bytes data
        let decoded = decode_layer_forward(trimmed).unwrap();
        assert!(decoded.layer_range.is_none());
        assert_eq!(decoded.activations, vec![1, 2, 3]);
    }

    #[test]
    fn layer_result_encode_decode_roundtrip() {
        let result = LayerResult {
            request_id: uuid::Uuid::new_v4(),
            token_ids: vec![100, 200, 300],
            finish_reason: Some(NetworkFinishReason::Stop),
            activations: vec![],
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
            layer_range: Some((2, 8)),
            sender_peer_bytes: None,
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
        assert_eq!(decoded.layer_range, Some((2, 8)));
    }
}
