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

/// Protocol ID for SwarmLLM JSON request/response (control messages, shard transfers).
pub const PROTOCOL_ID: &str = "/swarmllm/1.0.0";

/// Protocol ID for SwarmLLM tensor request/response (Cap'n Proto, zero-copy).
pub const TENSOR_PROTOCOL_ID: &str = "/swarmllm/tensor/1.0.0";

/// GossipSub topic for model coordination (shard announcements, capacity updates).
pub const TOPIC_MODELS: &str = "swarm/models";

/// GossipSub topic for governance (model voting, legacy).
pub const TOPIC_GOVERNANCE: &str = "swarm/governance";

/// GossipSub topic for governance proposals.
pub const TOPIC_GOV_PROPOSALS: &str = "swarm/gov/proposals";

/// GossipSub topic for governance votes.
pub const TOPIC_GOV_VOTES: &str = "swarm/gov/votes";

/// GossipSub topic for governance issues.
pub const TOPIC_GOV_ISSUES: &str = "swarm/gov/issues";

/// GossipSub topic for governance releases.
pub const TOPIC_GOV_RELEASES: &str = "swarm/gov/releases";

/// GossipSub topic for governance changelog.
pub const TOPIC_GOV_CHANGELOG: &str = "swarm/gov/changelog";

/// GossipSub topic for credit balance gossip.
pub const TOPIC_CREDITS: &str = "swarm/credits";

/// GossipSub topic for network health.
pub const TOPIC_HEALTH: &str = "swarm/health";

/// Codec for SwarmLLM request/response protocol using serde_json.
#[derive(Debug, Clone, Default)]
pub struct SwarmCodec;

/// Request type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmRequest {
    Message(Box<SwarmMessage>),
    ShardTransfer(ShardRequest),
}

/// Response type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmResponse {
    Message(Box<SwarmMessage>),
    ShardData(ShardResponse),
    Ack,
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
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 256 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Message too large",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 256 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Response too large",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
        let data =
            serde_json::to_vec(&req).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = (data.len() as u32).to_be_bytes();
        io.write_all(&len).await?;
        io.write_all(&data).await?;
        io.close().await?;
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
        let data =
            serde_json::to_vec(&resp).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = (data.len() as u32).to_be_bytes();
        io.write_all(&len).await?;
        io.write_all(&data).await?;
        io.close().await?;
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

// ---- Cap'n Proto Tensor Protocol ----

/// Request type for the tensor request_response protocol.
///
/// Carries raw serialized tensor data for zero-copy efficiency on the
/// hot path (activation forwarding between pipeline nodes).
#[derive(Debug, Clone)]
pub struct TensorRequest {
    pub payload: Vec<u8>,
}

/// Response type for the tensor request_response protocol.
#[derive(Debug, Clone)]
pub struct TensorResponse {
    pub payload: Vec<u8>,
}

/// Codec for the tensor protocol.
///
/// Uses a length-prefixed binary format. The payload is a manually encoded
/// tensor envelope (compatible with the Cap'n Proto schema in proto/messages.capnp).
/// When the capnp compiler is available, the schema-generated code can replace
/// the manual encode/decode functions below.
#[derive(Debug, Clone, Default)]
pub struct TensorCodec;

#[async_trait]
impl request_response::Codec for TensorCodec {
    type Protocol = StreamProtocol;
    type Request = TensorRequest;
    type Response = TensorResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 256 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tensor payload too large (>256MB)",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(TensorRequest { payload: buf })
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 256 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tensor response too large (>256MB)",
            ));
        }

        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(TensorResponse { payload: buf })
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
        let len = (req.payload.len() as u32).to_be_bytes();
        io.write_all(&len).await?;
        io.write_all(&req.payload).await?;
        io.close().await?;
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
        let len = (resp.payload.len() as u32).to_be_bytes();
        io.write_all(&len).await?;
        io.write_all(&resp.payload).await?;
        io.close().await?;
        Ok(())
    }
}

// ---- Manual Tensor Encoding (Cap'n Proto compatible wire format) ----

// Binary layout for LayerForward envelope:
//   [0..16]  request_id (UUID bytes)
//   [16..20] sequence_num (u32 LE)
//   [20]     format tag: 0=FP16, 1=FP32, 2=INT8
//   [21..25] data_len (u32 LE)
//   [25..]   activation data

/// Tensor message type tag — first byte of every tensor protocol message.
pub const TENSOR_TAG_FORWARD: u8 = 0x01;
pub const TENSOR_TAG_RESULT: u8 = 0x02;

/// Encode a LayerForward into a binary tensor envelope.
pub fn encode_layer_forward(forward: &LayerForward) -> Result<Vec<u8>, SwarmError> {
    let data_len = forward.activations.len();
    // Header: tag(1) + uuid(16) + seq(4) + index_pos(4) + fmt(1) + data_len(4) = 30
    let total = 1 + 29 + data_len;
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

    Ok(LayerForward {
        request_id,
        sequence_num,
        index_pos,
        activations,
        format,
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

    // Read activations if present
    let activations = if pos + 4 <= data.len() {
        let act_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
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

// Serde impls for SwarmRequest/SwarmResponse
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let msg = SwarmMessage::HealthPing {
            nonce: 42,
            timestamp: 1000,
        };
        let encoded = encode_message(&msg).unwrap();
        let decoded = decode_message(&encoded).unwrap();
        match decoded {
            SwarmMessage::HealthPing { nonce, timestamp } => {
                assert_eq!(nonce, 42);
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_serde_roundtrip() {
        let req = SwarmRequest::Message(Box::new(SwarmMessage::HealthPing {
            nonce: 1,
            timestamp: 2,
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
            sender_peer_bytes: None,
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();

        assert_eq!(decoded.request_id, forward.request_id);
        assert_eq!(decoded.sequence_num, 42);
        assert_eq!(decoded.activations, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(decoded.format, TensorFormat::FP16));
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
            sender_peer_bytes: None,
        };

        let encoded = encode_layer_forward(&forward).unwrap();
        let decoded = decode_layer_forward(&encoded).unwrap();
        assert_eq!(decoded.activations.len(), 1024 * 1024);
        assert_eq!(decoded.activations, data);
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
}
