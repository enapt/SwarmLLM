//! Model architecture detection from GGUF metadata.

// ── Model architecture detection ──

/// Known model architectures from GGUF `general.architecture` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    /// Llama family (Llama 1/2/3, CodeLlama, Yi, Mistral 7B)
    Llama,
    /// Qwen2 / Qwen2.5 / Qwen3
    Qwen2,
    /// Google Gemma 1
    Gemma,
    /// Google Gemma 2 — different RmsNorm (+1), Gelu activation, attention logit soft-capping
    Gemma2,
    /// Microsoft Phi-3/3.5 — SuRoPE scaling, partial rotary embedding
    Phi3,
    /// Mistral (when explicitly tagged, most Mistral GGUFs use "llama" arch)
    Mistral,
    /// StarCoder2
    Starcoder2,
    /// DeepSeek-V2/V3 — MoE + MLA
    DeepSeek2,
    /// GLM-4 — partial RoPE (half head dims), QKV biases, extreme GQA (2 KV heads)
    Glm4,
    /// Llama 4 Scout/Maverick — iRoPE (NoPE every 4th layer) + MoE
    Llama4,
    /// Qwen 3.5 dense — hybrid attention + Gated Delta Network (SSM) layers
    Qwen35,
    /// Qwen 3.5 MoE — hybrid attention + SSM layers with mixture-of-experts FFN
    Qwen35Moe,
    /// Architecture not recognized — falls back to Llama-like behavior
    Unknown(String),
}

impl ModelArch {
    /// Detect architecture from GGUF `general.architecture` metadata string.
    pub fn from_gguf_arch(arch: &str) -> Self {
        match arch {
            "llama" => ModelArch::Llama,
            "qwen2" | "qwen3" | "qwen2moe" => ModelArch::Qwen2,
            "gemma" => ModelArch::Gemma,
            "gemma2" => ModelArch::Gemma2,
            "phi3" => ModelArch::Phi3,
            "mistral" => ModelArch::Mistral,
            "starcoder2" => ModelArch::Starcoder2,
            "deepseek2" => ModelArch::DeepSeek2,
            "glm4" => ModelArch::Glm4,
            "llama4" => ModelArch::Llama4,
            "qwen35" => ModelArch::Qwen35,
            "qwen35moe" | "qwen3_5moe" => ModelArch::Qwen35Moe,
            other => ModelArch::Unknown(other.to_string()),
        }
    }

    /// Whether this architecture uses contiguous RoPE (NeoX-style halves) vs
    /// interleaved (original GPT-J/LLaMA pairs). Matches llama.cpp's
    /// `LLAMA_ROPE_TYPE_NEOX` (contiguous) vs `LLAMA_ROPE_TYPE_NORM`
    /// (interleaved), as `llama_model_rope_type` assigns them.
    ///
    /// **Read off llama.cpp, arch by arch — never inferred.** GLM-4, Llama-4 and
    /// DeepSeek-2 are NORM there and were contiguous here, with tests asserting
    /// the wrong answer because they were written against this function rather
    /// than against the reference. On GLM-4 that rotated every position of every
    /// layer the wrong way: short replies survived and anything longer came
    /// apart — broken, repeating code where llama.cpp, on the same file and the
    /// same 25 prompt tokens, wrote a correct function (FUTURE_WORK #96). HF's
    /// own implementations agree: GLM's `rotate_half` is the interleaved
    /// `0::2`/`1::2` form, Llama-4 and DeepSeek-V2 rotate complex pairs.
    /// ⚠ Llama-4 and DeepSeek-2 are matched to the reference and NOT run here —
    /// no such model is on the test fleet.
    pub fn use_rope_contiguous(&self) -> bool {
        // Interleaved (NORM): Llama, Mistral, GLM-4, Llama-4, DeepSeek-2
        // Contiguous (NEOX): everything else
        !matches!(
            self,
            ModelArch::Llama
                | ModelArch::Mistral
                | ModelArch::Glm4
                | ModelArch::Llama4
                | ModelArch::DeepSeek2
                | ModelArch::Unknown(_)
        )
    }

    /// Default activation function for this architecture's MLP.
    pub(crate) fn default_activation(&self) -> Activation {
        match self {
            ModelArch::Gemma | ModelArch::Gemma2 | ModelArch::Starcoder2 => Activation::Gelu,
            _ => Activation::SiLU,
        }
    }

    /// Whether this architecture uses the Gemma-style RmsNorm (adds 1 to weights).
    pub fn use_gemma_norm(&self) -> bool {
        matches!(self, ModelArch::Gemma | ModelArch::Gemma2)
    }

    /// Whether this architecture is supported for split inference.
    pub fn is_supported(&self) -> bool {
        !matches!(self, ModelArch::Unknown(_))
    }

    /// List of GGUF architecture strings supported by the split inference engine.
    pub fn supported_list() -> &'static [&'static str] {
        &[
            "llama",
            "qwen2",
            "qwen3",
            "qwen2moe",
            "gemma",
            "gemma2",
            "phi3",
            "mistral",
            "starcoder2",
            "deepseek2",
            "glm4",
            "llama4",
            "qwen35",
            "qwen35moe",
        ]
    }

    /// Whether this architecture uses hybrid attention + SSM (Gated Delta Network) layers.
    pub fn is_hybrid_ssm(&self) -> bool {
        matches!(self, ModelArch::Qwen35 | ModelArch::Qwen35Moe)
    }

    /// Map to the coarser `ModelArchitecture` variant used by `ModelManifest`.
    ///
    /// Most arches share the standard llama-compatible manifest layout, so they
    /// collapse to `Llama`. Only arches with distinct manifest handling (Qwen2,
    /// Qwen35, Qwen35Moe, Mistral, Phi) get their own variants. Unknown archs
    /// default to Llama — callers that want to log a warning should do so.
    pub fn to_manifest_architecture(&self) -> crate::types::ModelArchitecture {
        use crate::types::ModelArchitecture;
        match self {
            Self::Qwen2 => ModelArchitecture::Qwen2,
            Self::Qwen35 => ModelArchitecture::Qwen35,
            Self::Qwen35Moe => ModelArchitecture::Qwen35Moe {
                num_experts: 0,
                experts_per_token: 0,
            },
            Self::Mistral => ModelArchitecture::Mistral,
            Self::Phi3 => ModelArchitecture::Phi,
            Self::Llama
            | Self::Gemma
            | Self::Gemma2
            | Self::Starcoder2
            | Self::DeepSeek2
            | Self::Glm4
            | Self::Llama4
            | Self::Unknown(_) => ModelArchitecture::Llama,
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArch::Llama => write!(f, "llama"),
            ModelArch::Qwen2 => write!(f, "qwen2"),
            ModelArch::Gemma => write!(f, "gemma"),
            ModelArch::Gemma2 => write!(f, "gemma2"),
            ModelArch::Phi3 => write!(f, "phi3"),
            ModelArch::Mistral => write!(f, "mistral"),
            ModelArch::Starcoder2 => write!(f, "starcoder2"),
            ModelArch::DeepSeek2 => write!(f, "deepseek2"),
            ModelArch::Glm4 => write!(f, "glm4"),
            ModelArch::Llama4 => write!(f, "llama4"),
            ModelArch::Qwen35 => write!(f, "qwen35"),
            ModelArch::Qwen35Moe => write!(f, "qwen35moe"),
            ModelArch::Unknown(s) => write!(f, "{s}"),
        }
    }
}

/// Activation function used in the MLP/FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activation {
    /// SiLU / Swish — used by Llama, Qwen2, Mistral, Phi-3
    SiLU,
    /// Gelu — used by Gemma, Gemma 2
    Gelu,
}
