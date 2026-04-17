//! Shared types for SwarmLLM.
//!
//! Submodules group related types by concern:
//! - [`ids`] — `NodeId`, `ModelId`, `ShardId`, `Blake3Hash`
//! - [`identity`] — `SealedPrompt`, `NicknameRecord`, `NicknameGossip`
//! - [`pool`] — Device-pool membership, invitations, gossip
//! - [`model`] — `ModelManifest`, architecture/quantization, trust tracking
//! - [`node`] — `NodeCapability`, `GpuInfo`, `NodeStats`, `PeerInfo`
//! - [`credits`] — Credit ledger, priority tiers, transactions, gossip
//! - [`inference`] — Inference requests, pipeline/TP, layer forward/result
//! - [`network`] — `SwarmMessage`, `NetworkCommand`, wire-level structs

pub mod credits;
pub mod identity;
pub mod ids;
pub mod inference;
pub mod model;
pub mod network;
pub mod node;
pub mod pool;

pub use credits::{
    CreditBalance, CreditGossip, CreditTransaction, PriorityTier, TransactionReason,
};
pub use identity::{NicknameGossip, NicknameRecord, SealedPrompt};
pub use ids::{Blake3Hash, ModelId, NodeId, ShardId, MMPROJ_SHARD_INDEX};
pub use inference::{
    AllReduceOp, ChatMessage, ImageData, InferenceError, InferenceRequest, LayerForward,
    LayerResult, NetworkFinishReason, PipelineAssignment, PipelineSegment, Role, SamplingParams,
    StreamingToken, TensorFormat, TensorParallelGroup, TensorParallelMeta, TpAllReduceRequest,
    TpAllReduceResponse, TpPhase, TpRingChunk, VisionEncodeRequest, VisionEncodeResponse,
};
pub use model::{
    MmprojInfo, ModelArchitecture, ModelManifest, ModelTrustInfo, ModelTrustLevel, Quantization,
    ShardInfo, ShardTensorEntry, VisionConfig,
};
pub use network::{
    AuthenticatedMessage, DownloadState, EphemeralKeyExchange, HfSourceGossip, ModelDemandGossip,
    NetworkCommand, PruneEvent, RebalanceEvent, RegionShardSummary, ShardAnnounce,
    ShardDownloadProgress, ShardRequest, ShardResponse, SwarmMessage,
};
pub use node::{
    ContributionLevel, GpuInfo, NodeCapability, NodeStats, PeerExchangeResponse, PeerInfo,
};
pub use pool::{
    BlindedPoolInvitation, ContributionMode, PoolAcceptance, PoolCreditForward, PoolDeviceStats,
    PoolId, PoolInvitation, PoolMembership, PoolMessage, PoolRemoval, PoolState, ShardPin,
};

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

    #[allow(dead_code)]
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
