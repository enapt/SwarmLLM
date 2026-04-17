//! Model manifest, shard descriptors, architecture/quantization enums, and
//! auto-manage trust tracking.

use serde::{Deserialize, Serialize};

use crate::ids::{Blake3Hash, ModelId, NodeId};

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
