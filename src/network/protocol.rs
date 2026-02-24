use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response;
use libp2p::StreamProtocol;

use crate::types::{ShardRequest, ShardResponse, SwarmMessage};

/// Protocol ID for SwarmLLM request/response.
pub const PROTOCOL_ID: &str = "/swarmllm/1.0.0";

/// GossipSub topic for model coordination (shard announcements, capacity updates).
pub const TOPIC_MODELS: &str = "swarm/models";

/// GossipSub topic for governance (model voting).
pub const TOPIC_GOVERNANCE: &str = "swarm/governance";

/// GossipSub topic for network health.
pub const TOPIC_HEALTH: &str = "swarm/health";

/// Codec for SwarmLLM request/response protocol using serde_json.
#[derive(Debug, Clone, Default)]
pub struct SwarmCodec;

/// Request type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmRequest {
    Message(SwarmMessage),
    ShardTransfer(ShardRequest),
}

/// Response type for the request_response protocol.
#[derive(Debug, Clone)]
pub enum SwarmResponse {
    Message(SwarmMessage),
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

        if len > 64 * 1024 * 1024 {
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

        if len > 64 * 1024 * 1024 {
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
        enum Inner {
            Message { data: SwarmMessage },
            ShardTransfer { data: ShardRequest },
        }
        match Inner::deserialize(deserializer)? {
            Inner::Message { data } => Ok(SwarmRequest::Message(data)),
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
        enum Inner {
            Message { data: SwarmMessage },
            ShardData { data: ShardResponse },
            Ack,
        }
        match Inner::deserialize(deserializer)? {
            Inner::Message { data } => Ok(SwarmResponse::Message(data)),
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
        let req = SwarmRequest::Message(SwarmMessage::HealthPing {
            nonce: 1,
            timestamp: 2,
        });
        let json = serde_json::to_string(&req).unwrap();
        let parsed: SwarmRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            SwarmRequest::Message(SwarmMessage::HealthPing { nonce, .. }) => {
                assert_eq!(nonce, 1);
            }
            _ => panic!("wrong variant"),
        }
    }
}
