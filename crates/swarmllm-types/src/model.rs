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

/// Quantization level for a GGUF model. Captures the common k-quant,
/// i-quant, and float-precision variants used in the GGUF ecosystem.
///
/// Variants are ordered roughly by quality (lowest → highest), and each
/// carries an explicit `bits_per_weight()` so the auto-manage quant
/// recommender (R133) can compare candidates numerically. Renaming or
/// removing variants is a serde-breaking change — extend with new
/// variants instead. `Unknown` exists for filenames the parser doesn't
/// recognise, so the manifest still serialises cleanly.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Quantization {
    // K-quants (mainline llama.cpp)
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    // Legacy block-quants kept for completeness — rarely produced post-2024
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // I-quants — smaller, lossier than equivalent k-quant
    IQ1S,
    IQ1M,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ2M,
    IQ3XXS,
    IQ3XS,
    IQ3S,
    IQ3M,
    IQ4XS,
    IQ4NL,
    // Floats
    FP16,
    BF16,
    FP32,
    /// Filename / metadata didn't yield a recognised quant tag. Treated
    /// as the lowest-quality bucket by `quality_score`; downstream code
    /// should never use this to compute size estimates.
    Unknown,
}

impl Quantization {
    /// Parse a quant tag (case-insensitive) into a variant. Returns
    /// `Unknown` for unrecognised inputs. Accepts both underscore and
    /// no-underscore forms (`Q4_K_M` and `Q4KM`).
    pub fn parse(tag: &str) -> Self {
        let up = tag.to_uppercase();
        let stripped: String = up.chars().filter(|c| *c != '_').collect();
        match stripped.as_str() {
            "Q2K" => Self::Q2K,
            "Q3KS" => Self::Q3KS,
            "Q3KM" => Self::Q3KM,
            "Q3KL" => Self::Q3KL,
            "Q4KS" => Self::Q4KS,
            "Q4KM" => Self::Q4KM,
            "Q5KS" => Self::Q5KS,
            "Q5KM" => Self::Q5KM,
            "Q6K" => Self::Q6K,
            "Q40" => Self::Q4_0,
            "Q41" => Self::Q4_1,
            "Q50" => Self::Q5_0,
            "Q51" => Self::Q5_1,
            "Q80" => Self::Q8_0,
            "IQ1S" => Self::IQ1S,
            "IQ1M" => Self::IQ1M,
            "IQ2XXS" => Self::IQ2XXS,
            "IQ2XS" => Self::IQ2XS,
            "IQ2S" => Self::IQ2S,
            "IQ2M" => Self::IQ2M,
            "IQ3XXS" => Self::IQ3XXS,
            "IQ3XS" => Self::IQ3XS,
            "IQ3S" => Self::IQ3S,
            "IQ3M" => Self::IQ3M,
            "IQ4XS" => Self::IQ4XS,
            "IQ4NL" => Self::IQ4NL,
            "F16" | "FP16" => Self::FP16,
            "BF16" => Self::BF16,
            "F32" | "FP32" => Self::FP32,
            _ => Self::Unknown,
        }
    }

    /// Approximate bits per weight including k-quant block overhead.
    /// Source: llama.cpp `ggml-quants.c` block sizes + spec docs.
    /// Used to estimate model size from parameter count when only the
    /// quant level is known.
    pub fn bits_per_weight(self) -> f32 {
        match self {
            Self::Q2K => 2.625,
            Self::Q3KS => 3.4375,
            Self::Q3KM => 3.9,
            Self::Q3KL => 4.27,
            Self::Q4KS => 4.5,
            Self::Q4KM => 4.83,
            Self::Q5KS => 5.5,
            Self::Q5KM => 5.69,
            Self::Q6K => 6.5625,
            Self::Q4_0 => 4.5,
            Self::Q4_1 => 5.0,
            Self::Q5_0 => 5.5,
            Self::Q5_1 => 6.0,
            Self::Q8_0 => 8.5,
            Self::IQ1S => 1.5625,
            Self::IQ1M => 1.75,
            Self::IQ2XXS => 2.0625,
            Self::IQ2XS => 2.3125,
            Self::IQ2S => 2.5,
            Self::IQ2M => 2.7,
            Self::IQ3XXS => 3.0625,
            Self::IQ3XS => 3.3,
            Self::IQ3S => 3.4375,
            Self::IQ3M => 3.66,
            Self::IQ4XS => 4.25,
            Self::IQ4NL => 4.5,
            Self::FP16 | Self::BF16 => 16.0,
            Self::FP32 => 32.0,
            // Conservative for unknown — assume worst-case bits, so
            // size estimates over-state rather than under-state.
            Self::Unknown => 8.0,
        }
    }

    /// Coarse 0..100 quality score, calibrated against published
    /// perplexity-loss measurements from llama.cpp's quant docs.
    /// Higher = closer to FP16 reference output. Used by the quant
    /// recommender to compare candidates that all fit the swarm's
    /// VRAM budget.
    pub fn quality_score(self) -> u32 {
        match self {
            Self::FP32 => 100,
            Self::FP16 | Self::BF16 => 100,
            Self::Q8_0 => 99,
            Self::Q6K => 97,
            Self::Q5KM => 94,
            Self::Q5KS => 92,
            Self::Q5_1 => 91,
            Self::Q5_0 => 89,
            Self::Q4KM => 87,
            Self::Q4KS => 85,
            Self::Q4_1 => 83,
            Self::Q4_0 => 80,
            Self::IQ4NL => 86,
            Self::IQ4XS => 83,
            Self::Q3KL => 75,
            Self::Q3KM => 72,
            Self::Q3KS => 68,
            Self::IQ3M => 70,
            Self::IQ3S => 67,
            Self::IQ3XS => 64,
            Self::IQ3XXS => 60,
            Self::Q2K => 55,
            Self::IQ2M => 52,
            Self::IQ2S => 48,
            Self::IQ2XS => 44,
            Self::IQ2XXS => 40,
            Self::IQ1M => 28,
            Self::IQ1S => 20,
            Self::Unknown => 0,
        }
    }

    /// Canonical display label, matching the GGUF filename convention
    /// (uppercase with underscores). Use this for UI / logging.
    pub fn label(self) -> &'static str {
        match self {
            Self::Q2K => "Q2_K",
            Self::Q3KS => "Q3_K_S",
            Self::Q3KM => "Q3_K_M",
            Self::Q3KL => "Q3_K_L",
            Self::Q4KS => "Q4_K_S",
            Self::Q4KM => "Q4_K_M",
            Self::Q5KS => "Q5_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ2M => "IQ2_M",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3XS => "IQ3_XS",
            Self::IQ3S => "IQ3_S",
            Self::IQ3M => "IQ3_M",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ4NL => "IQ4_NL",
            Self::FP16 => "F16",
            Self::BF16 => "BF16",
            Self::FP32 => "F32",
            Self::Unknown => "unknown",
        }
    }
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
