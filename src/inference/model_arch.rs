//! Model architecture detection from GGUF metadata.

// ── Model architecture detection ──

/// Known model architectures from GGUF `general.architecture` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    /// Llama family (Llama 1/2/3, CodeLlama, Yi, Mistral 7B)
    Llama,
    /// Qwen2 / Qwen2.5 / Qwen3, and their MoE variants (Qwen2-MoE, Qwen3-MoE)
    Qwen2,
    /// Google Gemma 1
    Gemma,
    /// Google Gemma 2 — different RmsNorm (+1), Gelu activation, attention logit soft-capping
    Gemma2,
    /// Google Gemma 3 (text) — Gemma 2's layout with per-head Q/K norms, no
    /// soft-capping, five sliding-window layers to every global one, and a
    /// second RoPE on the windowed layers (`split::rope::LayerRope`)
    Gemma3,
    /// Google Gemma 4 (text) — Gemma 3's layout with a head width and K/V
    /// head count per layer, global layers whose K doubles as V, an
    /// unweighted norm on V, no query scaling, "proportional" RoPE on the
    /// global layers and a scale on each layer's output. The
    /// mixture-of-experts 26B-A4B runs a dense feed-forward beside its GELU
    /// experts (`layers::DenseBeside`); the E2B/E4B add a per-layer
    /// embedding to every layer (`split::per_layer`) and their last layers
    /// attend an earlier layer's cache (`split::kv_sources`), so they run only
    /// whole on one computer (`GgufTensorMeta::runs_only_whole`).
    Gemma4,
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
            // The MoE members are the same layout with routed experts in place
            // of the feed-forward, which the loader finds per layer
            // (`split::loader::load_moe_ffn`).
            "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" => ModelArch::Qwen2,
            "gemma" => ModelArch::Gemma,
            "gemma2" => ModelArch::Gemma2,
            "gemma3" => ModelArch::Gemma3,
            "gemma4" => ModelArch::Gemma4,
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

    /// Whether a card keeps this architecture's attention output and
    /// feed-forward down projections accumulating in f32 in a prompt pass,
    /// where every other quantized product accumulates in f16 (FUTURE_WORK
    /// #147; `layers::accumulate_output_projections_in_f32`).
    ///
    /// **Read off llama.cpp, arch by arch** (`src/llama-graph.cpp`, master
    /// 4b1a27f, 2026-09-25): `ggml_prec_set_acc(cur, GGML_PREC_F32)` on
    /// `ffn_down` and on `wo` for `LLM_ARCH_GLM4`, `GLM4_MOE` and `JAIS2` —
    /// "seem to have numerical issues with half-precision accumulators". Of
    /// those, GLM-4 is the one that runs here. Its two arms already differed
    /// more than any other family's in the 4K check (worst cosine between them
    /// 0.99956 against ≥ 0.99985).
    pub(crate) fn accumulates_outputs_in_f32(&self) -> bool {
        matches!(self, ModelArch::Glm4)
    }

    /// Default activation function for this architecture's MLP.
    pub(crate) fn default_activation(&self) -> Activation {
        match self {
            ModelArch::Gemma
            | ModelArch::Gemma2
            | ModelArch::Gemma3
            | ModelArch::Gemma4
            | ModelArch::Starcoder2 => Activation::Gelu,
            _ => Activation::SiLU,
        }
    }

    /// Whether this architecture scales its token embeddings by
    /// `sqrt(hidden)` on the way in, as every Gemma does (llama.cpp's
    /// `ggml_scale(inpL, sqrtf(n_embd))`). The name is historical: Gemma's
    /// `+1` norm offset is already in the file's weights (the converter adds
    /// it), so no norm reads this.
    pub fn use_gemma_norm(&self) -> bool {
        matches!(
            self,
            ModelArch::Gemma | ModelArch::Gemma2 | ModelArch::Gemma3 | ModelArch::Gemma4
        )
    }

    /// Whether this architecture is supported for split inference — what the
    /// loader accepts, what a shard download and auto-manage will fetch.
    ///
    /// **DeepSeek-2 is recognised, not supported** (FUTURE_WORK #116). No file
    /// llama.cpp's converter writes can load here: every one names its KV
    /// down-projection `attn_kv_a_mqa` (the loader asks for `attn_kv_a`); a
    /// V2-Lite layer has `attn_q` and no `attn_q_a` (the loader then asks for
    /// `attn_k`/`attn_v`); files since mid-2025 split `attn_kv_b` into
    /// `attn_k_b`/`attn_v_b`; and every DeepSeek-V2/V3 file uses YaRN with a
    /// `yarn_log_multiplier`, which the MLA branch's own RoPE table does not
    /// apply (`split::rope::RopeScaling` reads YaRN for the other families,
    /// #280). Checked 2026-09-25 against the real header of
    /// DeepSeek-Coder-V2-Lite-Instruct. Claiming it let a user, or auto-manage,
    /// fetch gigabytes of a model that then could not load.
    ///
    /// **Qwen 3.5 was refused the same way** until its loader was rewritten
    /// against llama.cpp and a real file (dense #117, 2026-10-06; the MoE
    /// `qwen35moe` #228 (1), 2026-10-11 — an unquantized real file matched
    /// llama.cpp's logits at every position, whole and split, and the popular
    /// 35B-A3B upload's header names the tensors the loader reads).
    ///
    /// **And StarCoder2** (FUTURE_WORK #118): it is a LayerNorm model — its
    /// files carry `attention.layer_norm_epsilon` and a bias on every norm, on
    /// `attn_output`, `ffn_up` and `ffn_down` (starcoder2-3b header, read
    /// 2026-09-25). The shared metadata parser REQUIRES
    /// `attention.layer_norm_rms_epsilon`, so no file gets past it, and the
    /// loader reads none of those biases and normalises by RMS.
    pub fn is_supported(&self) -> bool {
        !matches!(
            self,
            ModelArch::Unknown(_) | ModelArch::DeepSeek2 | ModelArch::Starcoder2
        )
    }

    /// Every architecture this build knows by name, `Unknown` aside — what
    /// [`Self::manifest_architecture_is_refused`] reads a manifest's coarse
    /// variant against.
    const NAMED: [ModelArch; 14] = [
        ModelArch::Llama,
        ModelArch::Qwen2,
        ModelArch::Gemma,
        ModelArch::Gemma2,
        ModelArch::Gemma3,
        ModelArch::Gemma4,
        ModelArch::Phi3,
        ModelArch::Mistral,
        ModelArch::Starcoder2,
        ModelArch::DeepSeek2,
        ModelArch::Glm4,
        ModelArch::Llama4,
        ModelArch::Qwen35,
        ModelArch::Qwen35Moe,
    ];

    /// Does a manifest that says `arch` describe a model this build refuses?
    /// Only when every named architecture its (coarse) variant stands for is
    /// refused: `Qwen35` stands for Qwen 3.5 alone, while `Llama` also stands
    /// for StarCoder2 and DeepSeek-2 beside the families that run, so it is
    /// never refused on the manifest's word.
    ///
    /// For `/v1/models`, which used to list a Qwen 3.5 model two peers held
    /// parts of while every request for it was a 400 (FUTURE_WORK #117) —
    /// `SharedState::refused_model` knows only once the model's header
    /// is here, i.e. after someone had tried it. Follows [`Self::is_supported`],
    /// so admitting an architecture lists its models again with no edit here.
    pub fn manifest_architecture_is_refused(arch: &crate::types::ModelArchitecture) -> bool {
        Self::refused_by(arch, Self::is_supported)
    }

    /// [`Self::manifest_architecture_is_refused`] under a given answer to
    /// "does this family run" — so the test can plant a refusal now that every
    /// variant with a family of its own runs (Qwen 3.5's MoE since #228 (1)).
    fn refused_by(
        arch: &crate::types::ModelArchitecture,
        supported: impl Fn(&ModelArch) -> bool,
    ) -> bool {
        let variant = std::mem::discriminant(arch);
        let mut named = Self::NAMED
            .iter()
            .filter(|a| std::mem::discriminant(&a.to_manifest_architecture()) == variant)
            .peekable();
        named.peek().is_some() && named.all(|a| !supported(a))
    }

    /// List of GGUF architecture strings supported by the split inference engine.
    pub fn supported_list() -> &'static [&'static str] {
        &[
            "llama",
            "qwen2",
            "qwen3",
            "qwen2moe",
            "qwen3moe",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma4",
            "phi3",
            "mistral",
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

    /// Whether a split of this family is sent trees of guesses (#134): plain
    /// attention over a full cache on every layer. An allowlist, so a new
    /// family checks chains until someone has run a tree on it. Gemma 2-4 keep
    /// sliding windows, DeepSeek-2 attends with MLA, Qwen 3.5 keeps a recurrent
    /// state, Llama 4 skips RoPE on some layers and Starcoder2 declares a
    /// window. Each worker asks its own layers too
    /// (`SplitModel::draft_tree_refusal`), which is the answer that holds.
    pub fn reads_draft_trees(&self) -> bool {
        matches!(
            self,
            ModelArch::Llama
                | ModelArch::Qwen2
                | ModelArch::Phi3
                | ModelArch::Mistral
                | ModelArch::Glm4
        )
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
            | Self::Gemma3
            | Self::Gemma4
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
            ModelArch::Gemma3 => write!(f, "gemma3"),
            ModelArch::Gemma4 => write!(f, "gemma4"),
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
    /// Gelu — used by Gemma, Gemma 2, Gemma 3, Gemma 4
    Gelu,
}

#[cfg(test)]
mod tests {
    use super::ModelArch;
    use crate::types::ModelArchitecture;

    /// `NAMED` lists every variant: adding one breaks this match until it has
    /// a slot, and the slot must be its place in `NAMED`.
    #[test]
    fn every_named_architecture_is_listed() {
        fn slot(a: &ModelArch) -> Option<usize> {
            match a {
                ModelArch::Llama => Some(0),
                ModelArch::Qwen2 => Some(1),
                ModelArch::Gemma => Some(2),
                ModelArch::Gemma2 => Some(3),
                ModelArch::Gemma3 => Some(4),
                ModelArch::Gemma4 => Some(5),
                ModelArch::Phi3 => Some(6),
                ModelArch::Mistral => Some(7),
                ModelArch::Starcoder2 => Some(8),
                ModelArch::DeepSeek2 => Some(9),
                ModelArch::Glm4 => Some(10),
                ModelArch::Llama4 => Some(11),
                ModelArch::Qwen35 => Some(12),
                ModelArch::Qwen35Moe => Some(13),
                ModelArch::Unknown(_) => None,
            }
        }
        for (i, a) in ModelArch::NAMED.iter().enumerate() {
            assert_eq!(slot(a), Some(i), "{a} is out of place in NAMED");
        }
    }

    /// A manifest says which family a model is; one that only a refused
    /// family can be is not offered as a model (#117). Dense Qwen 3.5 runs
    /// since #117 and its mixture-of-experts sibling since #228 (1), so every
    /// variant runs today; the refusal itself is checked with one planted.
    #[test]
    fn a_manifest_only_a_refused_family_can_be_is_refused() {
        let moe = ModelArchitecture::Qwen35Moe {
            num_experts: 128,
            experts_per_token: 8,
        };
        for runs in [
            moe.clone(),
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Qwen35,
            ModelArchitecture::Mistral,
            ModelArchitecture::Phi,
        ] {
            assert!(
                !ModelArch::manifest_architecture_is_refused(&runs),
                "{runs:?} stands for a family that runs"
            );
        }
        // Planted: the MoE refused, everything else as this build answers.
        let refuse_moe = |a: &ModelArch| *a != ModelArch::Qwen35Moe && a.is_supported();
        assert!(ModelArch::refused_by(&moe, refuse_moe));
        assert!(!ModelArch::refused_by(
            &ModelArchitecture::Qwen35,
            refuse_moe
        ));
        // `Llama` also stands for families that run, so refusing one of the
        // families it covers (DeepSeek-2, StarCoder2 are) never hides it.
        assert!(!ModelArch::refused_by(
            &ModelArchitecture::Llama,
            ModelArch::is_supported
        ));
        // A variant no named family maps to says nothing either way.
        assert!(!ModelArch::manifest_architecture_is_refused(
            &ModelArchitecture::Glm4
        ));
    }
}
