// ── Split model: loads only a range of layers from a GGUF ──

use std::collections::HashMap;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::SwarmError;
use crate::model::lora::LoraAdapter;

#[cfg(test)]
use super::entry::BatchItem;
use super::kv_cache::KvCacheStore;
use super::rope::{load_longrope_factors, precompute_freqs_cis, precompute_freqs_cis_longrope};
use super::shard_reader::ShardReader;
use super::{
    Activation, DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights,
    Mlp, ModelArch, MoeFfn, QMatMul, Qwen35AttnWeights, SplitTokenizer, SsmState,
    DEFAULT_MAX_SEQ_LEN,
};

/// A partial transformer model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
/// Supports multiple architectures: Llama, Qwen2, Gemma 2, Phi-3, Mistral, Qwen 3.5.
pub struct SplitModel {
    /// Token embedding table (only loaded by the first segment).
    pub(super) tok_embeddings: Option<Embedding>,
    /// Transformer layers for this segment's range.
    pub(super) layers: Vec<LayerVariant>,
    /// Final RMSNorm (only loaded by the last segment).
    pub(super) norm: Option<RmsNorm>,
    /// LM head / output projection (only loaded by the last segment).
    pub(super) output: Option<QMatMul>,
    /// Causal attention mask: pre-allocated at a ceiling size, narrowed for smaller sequences.
    /// Tuple is (allocated_size, mask_tensor). `None` means no mask allocated yet.
    pub(super) masks: Option<(usize, Tensor)>,
    /// Layer range this model covers: [start, end) out of total_layers.
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
    /// Hidden dimension (embedding_length).
    pub hidden_dim: usize,
    /// Detected model architecture.
    pub arch: ModelArch,
    /// Device (CPU or CUDA).
    pub(super) device: Device,
    /// Vocabulary from GGUF (token ID → string), for decoding generated tokens.
    pub(super) vocabulary: Option<Vec<String>>,
    /// Tokenizer (BPE or sentencepiece/unigram) built from GGUF metadata.
    pub(super) tokenizer: Option<SplitTokenizer>,
    /// EOS token IDs loaded from GGUF metadata.
    pub(super) eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` (Jinja2 format).
    pub(super) chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    pub(super) bos_token: String,
    /// EOS token string from GGUF metadata.
    pub(super) eos_token: String,
    /// Maximum sequence length for KV cache pre-allocation.
    pub(super) max_seq_len: usize,
    /// Pre-computed KV cache store key: "{layer_start}-{layer_end}-{total_layers}".
    /// Avoids a `format!` allocation on every forward pass.
    pub(super) kv_model_key: String,
    /// Gemma 2 final logit soft-capping value (e.g. 30.0).
    pub(super) final_logit_softcap: Option<f32>,
}

impl SplitModel {
    /// Load a partial model from a GGUF file, only loading the specified layer range.
    ///
    /// - `layer_start..layer_end`: the transformer block range this node owns
    /// - `is_first`: if true, also loads the embedding table
    /// - `is_last`: if true, also loads the final norm and LM head
    pub fn load_from_gguf(
        gguf_path: &Path,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        let file = std::fs::File::open(gguf_path).map_err(SwarmError::Io)?;
        // SAFETY: Standard mmap usage — file is kept open for the duration of loading.
        // The mmap is dropped before the function returns; loaded tensors own their data.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| SwarmError::Internal(format!("Failed to mmap GGUF: {e}")))?;
        let mut file = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF: {e}")))?;

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if device.is_cuda() {
            tracing::info!(layers = %(layer_start..=layer_end).count(), layer_start, layer_end, "Split model using CUDA GPU");
        } else {
            tracing::info!(layers = %(layer_start..=layer_end).count(), layer_start, layer_end, "Split model using CPU (no CUDA available)");
        }

        Self::load_model_from_content(
            ct,
            &mut file,
            device,
            layer_start,
            layer_end,
            is_first,
            is_last,
            Some(mmap.as_ref()),
        )
    }

    /// Shared model loading body for both GGUF and shard paths.
    ///
    /// Parses GGUF metadata, loads tensors by architecture, extracts tokenizer/
    /// vocabulary/chat template, and constructs the SplitModel.
    ///
    /// `parallel_data`: when Some, enables parallel dense layer loading via
    /// per-thread Cursors (mmap path). When None, loads sequentially (ShardReader).
    #[allow(clippy::too_many_arguments)]
    fn load_model_from_content<R: std::io::Read + std::io::Seek>(
        ct: gguf_file::Content,
        mut file: &mut R,
        device: Device,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        parallel_data: Option<&[u8]>,
    ) -> Result<Self, SwarmError> {
        // Detect architecture prefix from GGUF metadata
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        let model_arch = ModelArch::from_gguf_arch(&arch_str);

        tracing::info!(arch = %model_arch, "Detected model architecture");

        if !model_arch.is_supported() {
            return Err(SwarmError::Validation(format!(
                "Unsupported model architecture '{}'. Supported architectures: {}",
                arch_str,
                ModelArch::supported_list().join(", ")
            )));
        }

        let arch = &arch_str; // keep for metadata key lookups
        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };

        let head_count = md_get("attention.head_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
        let head_count_kv = md_get("attention.head_count_kv")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        let block_count = md_get("block_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        let embedding_length = md_get("embedding_length")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))?
            as usize;
        // head_dim: prefer attention.key_length from GGUF (Qwen3 uses 128 vs embed/heads=64)
        let head_dim = ct
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or(head_dim as u32) as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
            .to_f32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as f64;
        let rope_freq_base = ct
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000f32);
        let context_length = md_get("context_length")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or(DEFAULT_MAX_SEQ_LEN as u32) as usize;

        // Gemma 2 attention logit soft-capping (from GGUF metadata)
        let attn_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        // Gemma 2 final logit soft-capping (from GGUF metadata)
        let final_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

        // Long RoPE (SuRoPE) for Phi-3.5 and similar extended-context models
        let longrope = load_longrope_factors(&ct, &mut file, arch, context_length);
        let (cos, sin) = if let Some((ref factors, attn_factor)) = longrope {
            tracing::info!(
                factors_len = factors.len(),
                attn_factor,
                "Using Long RoPE (SuRoPE)"
            );
            precompute_freqs_cis_longrope(
                rope_dim,
                rope_freq_base,
                context_length,
                factors,
                attn_factor,
                &device,
            )
            .map_err(|e| SwarmError::Internal(e.to_string()))?
        } else {
            precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?
        };
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        // Helper: create RmsNorm from GGUF weight tensor.
        // Note: GGUF norm weights for Gemma models already include the +1 offset
        // (added by convert_hf_to_gguf.py's modify_tensors), so we use them as-is.
        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            RmsNorm::from_qtensor(qtensor, eps).map_err(|e| SwarmError::Internal(e.to_string()))
        };

        // Load embedding table only for first segment
        let tok_embeddings = if is_first {
            let tok_embd = ct
                .tensor(&mut file, "token_embd.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load embeddings: {e}")))?;
            let tok_embd = tok_embd
                .dequantize(&device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?;
            Some(Embedding::new(tok_embd, embedding_length))
        } else {
            None
        };

        // Load output norm and LM head only for last segment
        let norm = if is_last {
            let norm_tensor = ct
                .tensor(&mut file, "output_norm.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load output_norm: {e}")))?;
            Some(make_norm(norm_tensor, rms_norm_eps)?)
        } else {
            None
        };

        let output = if is_last {
            let output_tensor = ct
                .tensor(&mut file, "output.weight", &device)
                .or_else(|_| ct.tensor(&mut file, "token_embd.weight", &device))
                .map_err(|e| SwarmError::Internal(format!("Failed to load output head: {e}")))?;
            Some(
                QMatMul::from_qtensor(output_tensor)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            None
        };

        // Load only the specified layer range (capped at actual block count)
        let layer_end = layer_end.min(block_count);
        let mut layers: Vec<LayerVariant> = Vec::with_capacity(layer_end - layer_start);

        if matches!(model_arch, ModelArch::DeepSeek2) {
            // ── DeepSeek-V2/V3: MLA + MoE loading ──
            let ds_meta = DeepSeekMeta {
                n_experts: md_get("expert_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                n_experts_used: md_get("expert_used_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(6) as usize,
                n_shared_experts: md_get("expert_shared_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                kv_lora_rank: md_get("attention.kv_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(512) as usize,
                q_lora_rank: md_get("attention.q_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                key_length: md_get("attention.key_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                value_length: md_get("attention.value_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                rope_dim,
            };

            // DeepSeek RoPE may use a different dimension for MLA layers
            let mla_rope_dim = ds_meta.rope_dim;
            let (mla_cos, mla_sin) =
                precompute_freqs_cis(mla_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;

            tracing::info!(
                n_experts = ds_meta.n_experts,
                n_experts_used = ds_meta.n_experts_used,
                n_shared_experts = ds_meta.n_shared_experts,
                kv_lora_rank = ds_meta.kv_lora_rank,
                q_lora_rank = ds_meta.q_lora_rank,
                key_length = ds_meta.key_length,
                value_length = ds_meta.value_length,
                "Loading DeepSeek-V2/V3 MoE+MLA model"
            );

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");

                // Per-layer detection: MLA vs dense attention
                let has_mla = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_q_a.weight"));
                // Per-layer detection: MoE vs dense FFN
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.attn_norm: {e}"))
                    })?;
                let ffn_norm_t = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.ffn_norm: {e}"))
                    })?;

                if has_mla {
                    // MLA attention weights
                    let q_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_a: {e}")))?;
                    let q_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_q_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_q_a_norm: {e}"))
                        })?;
                    let q_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_b: {e}")))?;
                    let kv_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_a: {e}")))?;
                    let kv_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_kv_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_kv_a_norm: {e}"))
                        })?;
                    let kv_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_b: {e}")))?;
                    let wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;

                    let attention = MlaWeights {
                        q_a: QMatMul::from_qtensor(q_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_a_norm: RmsNorm::from_qtensor(q_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_b: QMatMul::from_qtensor(q_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a: QMatMul::from_qtensor(kv_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a_norm: RmsNorm::from_qtensor(kv_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_b: QMatMul::from_qtensor(kv_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        output: QMatMul::from_qtensor(wo)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        n_head: head_count,
                        key_length: ds_meta.key_length,
                        value_length: ds_meta.value_length,
                        kv_lora_rank: ds_meta.kv_lora_rank,
                        rope_dim: mla_rope_dim,
                        cos: mla_cos.clone(),
                        sin: mla_sin.clone(),
                        neg_inf: neg_inf.clone(),
                    };

                    // FFN: MoE or dense
                    let ffn = if has_moe {
                        let gate_inp = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}"))
                            })?;
                        let gate_exps = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_gate_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                            })?;
                        let down_exps = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_down_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                            })?;
                        let up_exps = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}"))
                            })?;

                        // Dequantize stacked expert tensors for index_select routing
                        let gate_inp_t = gate_inp
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_inp dequant: {e}")))?;
                        let gate_exps_t = gate_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_exps dequant: {e}")))?;
                        let down_exps_t = down_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("down_exps dequant: {e}")))?;
                        let up_exps_t = up_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("up_exps dequant: {e}")))?;

                        // Shared experts (optional)
                        let shared_gate = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_gate_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared gate: {e}")))?;
                        let shared_down = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_down_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared down: {e}")))?;
                        let shared_up = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared up: {e}")))?;

                        FfnVariant::MoE(MoeFfn {
                            gate: gate_inp_t,
                            gate_exps: gate_exps_t,
                            down_exps: down_exps_t,
                            up_exps: up_exps_t,
                            shared_gate,
                            shared_down,
                            shared_up,
                            n_experts_used: ds_meta.n_experts_used,
                        })
                    } else {
                        // Dense FFN for early DeepSeek layers
                        let ffn_gate = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                        let ffn_down = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                        let ffn_up = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: Some(
                                QMatMul::from_qtensor(ffn_gate)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ),
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        })
                    };

                    layers.push(LayerVariant::DeepSeek {
                        attention,
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                    });
                } else {
                    // Dense attention layer (first few DeepSeek layers use standard attention)
                    let attention_wq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                    let attention_wk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                    let attention_wv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                    let attention_wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attention_bq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                    let attention_bk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                    let attention_bv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                    // Dense FFN for early DeepSeek layers
                    let ffn_gate = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;

                    // Dense DeepSeek layers use standard attention — wrap as LayerVariant::Dense
                    layers.push(LayerVariant::Dense(LayerWeights {
                        attention_wq: QMatMul::from_qtensor(attention_wq)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        attention_wk: QMatMul::from_qtensor(attention_wk)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        attention_wv: QMatMul::from_qtensor(attention_wv)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        attention_wo: QMatMul::from_qtensor(attention_wo)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        attention_bq,
                        attention_bk,
                        attention_bv,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        attn_q_norm: None,
                        attn_k_norm: None,
                        ffn: FfnVariant::Dense(Mlp {
                            ffn_gate: Some(
                                QMatMul::from_qtensor(ffn_gate)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ),
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        }),
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        post_attention_norm: None,
                        post_ffw_norm: None,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        neg_inf: neg_inf.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim,
                        skip_rope: false,
                    }));
                }
            }
        } else if matches!(model_arch, ModelArch::Llama4) {
            // ── Llama 4 Scout/Maverick: iRoPE + MoE loading ──
            // iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE (no positional encoding)
            // MoE: router selects top-k experts from stacked expert tensors

            // Read Llama 4 MoE metadata
            let n_experts = ct
                .metadata
                .get(&format!("{arch}.expert_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(0) as usize;
            let n_experts_used = ct
                .metadata
                .get(&format!("{arch}.expert_used_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(1) as usize;

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");
                let is_nope = layer_idx % 4 == 3; // NoPE every 4th layer

                let attention_wq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                let attention_wk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                let attention_wv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                let attention_wo = ct
                    .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let ffn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                // QKV biases (optional)
                let attention_bq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                let attention_bk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                let attention_bv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                // Check if this layer has MoE
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let ffn = if has_moe && n_experts > 0 {
                    let gate_inp = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
                    let gate_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                        })?;
                    let down_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                        })?;
                    let up_exps = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}")))?;

                    let gate_inp_t = gate_inp
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_inp dequant: {e}")))?;
                    let gate_exps_t = gate_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_exps dequant: {e}")))?;
                    let down_exps_t = down_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("down_exps dequant: {e}")))?;
                    let up_exps_t = up_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("up_exps dequant: {e}")))?;

                    // Shared experts (optional for Llama 4)
                    let shared_gate = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared gate: {e}")))?;
                    let shared_down = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared down: {e}")))?;
                    let shared_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared up: {e}")))?;

                    FfnVariant::MoE(MoeFfn {
                        gate: gate_inp_t,
                        gate_exps: gate_exps_t,
                        down_exps: down_exps_t,
                        up_exps: up_exps_t,
                        shared_gate,
                        shared_down,
                        shared_up,
                        n_experts_used,
                    })
                } else {
                    // Dense FFN — gate is optional (absent in Starcoder2's 2-layer MLP)
                    let ffn_gate_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .ok();
                    let ffn_down_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: ffn_gate_t
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_down: QMatMul::from_qtensor(ffn_down_t)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_up: QMatMul::from_qtensor(ffn_up_t)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        activation,
                    })
                };

                // Both MoE and dense FFN layers use standard attention via LayerVariant::Dense.
                // The FfnVariant enum handles MoE vs dense FFN dispatch in the forward pass.
                layers.push(LayerVariant::Dense(LayerWeights {
                    attention_wq: QMatMul::from_qtensor(attention_wq)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    attention_wk: QMatMul::from_qtensor(attention_wk)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    attention_wv: QMatMul::from_qtensor(attention_wv)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    attention_wo: QMatMul::from_qtensor(attention_wo)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    attention_bq,
                    attention_bk,
                    attention_bv,
                    attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    ffn,
                    ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                    post_attention_norm: None,
                    post_ffw_norm: None,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    use_rope_contiguous,
                    attn_logit_softcap,
                    rope_dim,
                    skip_rope: is_nope,
                }));
            }
        } else if model_arch.is_hybrid_ssm() {
            // ── Qwen 3.5 hybrid: attention + SSM (Gated Delta Network) loading ──
            // Read linear attention config from GGUF metadata
            let linear_conv_kernel_dim = ct
                .metadata
                .get(&format!("{arch}.ssm.conv_kernel"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(4) as usize;
            let linear_key_head_dim = ct
                .metadata
                .get(&format!("{arch}.ssm.inner_size"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(128);
            let linear_n_kv_head = ct
                .metadata
                .get(&format!("{arch}.attention.head_count_kv"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(head_count_kv as u32) as usize;
            let linear_n_v_head = linear_n_kv_head; // typically same as K heads for SSM
            let linear_value_head_dim = linear_key_head_dim;

            // Partial RoPE for Qwen 3.5: partial_rotary_factor * head_dim
            let partial_rotary_factor = ct
                .metadata
                .get(&format!("{arch}.rope.partial_rotary_factor"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(0.25);
            let qwen35_rope_dim = (head_dim as f32 * partial_rotary_factor) as usize;
            let (q35_cos, q35_sin) =
                precompute_freqs_cis(qwen35_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;

            // Determine layer types from GGUF metadata
            // Qwen 3.5 pattern: every 4th layer is full_attention, rest are linear_attention
            // We detect this per-layer by checking tensor presence
            let is_moe = matches!(model_arch, ModelArch::Qwen35Moe);

            tracing::info!(
                linear_conv_kernel_dim,
                linear_key_head_dim,
                linear_n_kv_head,
                qwen35_rope_dim,
                is_moe,
                "Loading Qwen 3.5 hybrid SSM+attention model"
            );

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");

                // Detect if this layer is SSM (linear_attention) or full attention
                // SSM layers have ssm_alpha.weight, attention layers have attn_q.weight
                let is_ssm_layer = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ssm_alpha.weight"));

                // Load norms (shared by both layer types)
                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let post_attn_norm = ct
                    .tensor(
                        &mut file,
                        &format!("{prefix}.attn_post_norm.weight"),
                        &device,
                    )
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_post_norm: {e}")))?;

                // Load FFN (dense or MoE)
                let ffn = if is_moe
                    && ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
                {
                    // MoE FFN
                    let gate_inp = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
                    let gate_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                        })?;
                    let down_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                        })?;
                    let up_exps = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}")))?;

                    let n_experts_used = ct
                        .metadata
                        .get(&format!("{arch}.expert_used_count"))
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(2) as usize;

                    // Shared experts (optional for MoE)
                    let shared_gate = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let shared_down = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let shared_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                        .ok()
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;

                    FfnVariant::MoE(MoeFfn {
                        gate: gate_inp
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        gate_exps: gate_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        down_exps: down_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        up_exps: up_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        shared_gate,
                        shared_down,
                        shared_up,
                        n_experts_used,
                    })
                } else {
                    // Dense FFN
                    let ffn_gate = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: Some(
                            QMatMul::from_qtensor(ffn_gate)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ),
                        ffn_down: QMatMul::from_qtensor(ffn_down)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_up: QMatMul::from_qtensor(ffn_up)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        activation,
                    })
                };

                if is_ssm_layer {
                    // SSM / Gated Delta Network layer
                    let wqkv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_qkv.weight"), &device)
                        .ok();
                    let (wq, wk, wv) = if wqkv.is_none() {
                        let q = ct
                            .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                            .ok();
                        let k = ct
                            .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                            .ok();
                        let v = ct
                            .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                            .ok();
                        (
                            q.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            k.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            v.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                        )
                    } else {
                        (None, None, None)
                    };
                    if wqkv.is_none() && (wq.is_none() || wk.is_none() || wv.is_none()) {
                        return Err(SwarmError::Internal(format!(
                            "{prefix}: missing attn_qkv and individual attn_q/k/v weights"
                        )));
                    }

                    let ssm_alpha = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_alpha.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_alpha: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_beta = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_beta.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_beta: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_dt = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_dt.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_dt: {e}")))?;
                    let ssm_conv1d = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_conv1d.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_conv1d: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_norm_t = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_norm: {e}")))?;
                    let ssm_out = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_out.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_out: {e}")))?;

                    layers.push(LayerVariant::Qwen35Ssm {
                        weights: DeltaNetWeights {
                            wqkv: wqkv
                                .map(|t| {
                                    QMatMul::from_qtensor(t).map_err(|e| {
                                        SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                    })
                                })
                                .transpose()?,
                            wq,
                            wk,
                            wv,
                            ssm_alpha,
                            ssm_beta,
                            ssm_dt: QMatMul::from_qtensor(ssm_dt)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ssm_conv1d,
                            ssm_norm: RmsNorm::from_qtensor(ssm_norm_t, rms_norm_eps)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ssm_out: QMatMul::from_qtensor(ssm_out)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            n_head: head_count,
                            n_kv_head: linear_n_kv_head,
                            n_v_head: linear_n_v_head,
                            key_head_dim: linear_key_head_dim,
                            value_head_dim: linear_value_head_dim,
                            conv_kernel_dim: linear_conv_kernel_dim,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                } else {
                    // Full attention layer (every 4th layer in Qwen 3.5)
                    let wqkv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_qkv.weight"), &device)
                        .ok();
                    let (wq, wk, wv) = if wqkv.is_none() {
                        let q = ct
                            .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                            .ok();
                        let k = ct
                            .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                            .ok();
                        let v = ct
                            .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                            .ok();
                        (
                            q.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            k.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            v.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                        )
                    } else {
                        (None, None, None)
                    };
                    if wqkv.is_none() && (wq.is_none() || wk.is_none() || wv.is_none()) {
                        return Err(SwarmError::Internal(format!(
                            "{prefix}: missing attn_qkv and individual attn_q/k/v weights"
                        )));
                    }
                    let wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attn_gate_t = ct
                        .tensor(&mut file, &format!("{prefix}.attn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_gate: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    // Q/K norms (optional)
                    let q_norm = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_norm.weight"), &device)
                        .ok()
                        .map(|t| {
                            RmsNorm::from_qtensor(t, rms_norm_eps).map_err(|e| {
                                SwarmError::Internal(format!("RmsNorm load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let k_norm = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k_norm.weight"), &device)
                        .ok()
                        .map(|t| {
                            RmsNorm::from_qtensor(t, rms_norm_eps).map_err(|e| {
                                SwarmError::Internal(format!("RmsNorm load failed: {e}"))
                            })
                        })
                        .transpose()?;

                    layers.push(LayerVariant::Qwen35Attn {
                        weights: Qwen35AttnWeights {
                            wqkv: wqkv
                                .map(|t| {
                                    QMatMul::from_qtensor(t).map_err(|e| {
                                        SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                    })
                                })
                                .transpose()?,
                            wq,
                            wk,
                            wv,
                            wo: QMatMul::from_qtensor(wo)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            attn_gate: attn_gate_t,
                            q_norm,
                            k_norm,
                            n_head: head_count,
                            n_kv_head: head_count_kv,
                            head_dim,
                            cos: q35_cos.clone(),
                            sin: q35_sin.clone(),
                            neg_inf: neg_inf.clone(),
                            rope_dim: qwen35_rope_dim,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                }
            }
        } else {
            // ── Standard dense architecture loading (Llama, Qwen2, Gemma, GLM-4, etc.) ──
            if let Some(mmap_ref) = parallel_data {
                // Parallel layer loading: each thread gets its own Cursor into mmap'd data.
                // ~N× speedup for N layers on NVMe/SSD.
                let ct_ref = &ct;
                let device_ref = &device;
                let layer_results: Vec<Result<LayerVariant, SwarmError>> =
                    std::thread::scope(|s| {
                        let handles: Vec<_> = (layer_start..layer_end)
                            .map(|layer_idx| {
                                let cos = cos.clone();
                                let sin = sin.clone();
                                let neg_inf = neg_inf.clone();
                                s.spawn(move || -> Result<LayerVariant, SwarmError> {
                                    let mut cursor = std::io::Cursor::new(mmap_ref);
                                    let prefix = format!("blk.{layer_idx}");

                                    // Try separate Q/K/V first; fall back to fused attn_qkv
                                    // (Phi-3 uses fused attn_qkv.weight instead of separate attn_q/k/v)
                                    let has_fused_qkv = ct_ref
                                        .tensor_infos
                                        .contains_key(&format!("{prefix}.attn_qkv.weight"));
                                    let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                                        // Fused QKV: keep original quantized tensor, split output in forward
                                        let fused_qt = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_qkv.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "{prefix}.attn_qkv: {e}"
                                                ))
                                            })?;
                                        let q_dim = head_count * head_dim;
                                        let k_dim = head_count_kv * head_dim;
                                        let fused = QMatMul::make_fused(fused_qt)
                                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                                        (
                                            QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                                            QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                                            QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
                                        )
                                    } else {
                                        let wq = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_q.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.attn_q: {e}"
                                                ))
                                            })?;
                                        let wk = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_k.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.attn_k: {e}"
                                                ))
                                            })?;
                                        let wv = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_v.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.attn_v: {e}"
                                                ))
                                            })?;
                                        (
                                            QMatMul::from_qtensor(wq)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                            QMatMul::from_qtensor(wk)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                            QMatMul::from_qtensor(wv)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                        )
                                    };
                                    let attention_wo = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_output.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.attn_output: {e}"
                                            ))
                                        })?;

                                    let attention_bq = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_q.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_q.bias dequant: {e}"
                                            ))
                                        })?;
                                    let attention_bk = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_k.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_k.bias dequant: {e}"
                                            ))
                                        })?;
                                    let attention_bv = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_v.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_v.bias dequant: {e}"
                                            ))
                                        })?;

                                    // FFN: try separate gate/up first; fall back to fused gate_up
                                    // (Phi-3 uses combined ffn_up = gate || up, no separate ffn_gate)
                                    let has_ffn_gate = ct_ref
                                        .tensor_infos
                                        .contains_key(&format!("{prefix}.ffn_gate.weight"));
                                    let ffn_down_qt = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.ffn_down.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.ffn_down: {e}"
                                            ))
                                        })?;
                                    let (ffn_gate_mm, ffn_down_mm, ffn_up_mm) = if has_ffn_gate {
                                        let gate = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.ffn_gate.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.ffn_gate: {e}"
                                                ))
                                            })?;
                                        let up = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.ffn_up.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.ffn_up: {e}"
                                                ))
                                            })?;
                                        (
                                            QMatMul::from_qtensor(gate)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                            QMatMul::from_qtensor(ffn_down_qt)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                            QMatMul::from_qtensor(up)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                        )
                                    } else {
                                        // Fused gate+up: use FusedSlice to avoid re-quantization
                                        let fused_qt = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.ffn_up.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.ffn_up: {e}"
                                                ))
                                            })?;
                                        let fused_shape = fused_qt.shape();
                                        let half = fused_shape.dims()[0] / 2;
                                        let fused = QMatMul::make_fused(fused_qt)
                                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                                        (
                                            QMatMul::from_fused_slice(fused.clone(), 0, half),
                                            QMatMul::from_qtensor(ffn_down_qt)
                                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                            QMatMul::from_fused_slice(fused, half, half),
                                        )
                                    };
                                    let attn_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_norm.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.attn_norm: {e}"
                                            ))
                                        })?;
                                    let ffn_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.ffn_norm.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.ffn_norm: {e}"
                                            ))
                                        })?;

                                    // Qwen3 QK normalization (optional)
                                    let attn_q_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_q_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!("attn_q_norm: {e}"))
                                        })?;
                                    let attn_k_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_k_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!("attn_k_norm: {e}"))
                                        })?;

                                    // Gemma 2 post-norms (optional)
                                    let post_attention_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.post_attention_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| make_norm(t, rms_norm_eps))
                                        .transpose()?;
                                    let post_ffw_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.post_ffw_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| make_norm(t, rms_norm_eps))
                                        .transpose()?;

                                    Ok(LayerVariant::Dense(LayerWeights {
                                        attention_wq: qkv_q,
                                        attention_wk: qkv_k,
                                        attention_wv: qkv_v,
                                        attention_wo: QMatMul::from_qtensor(attention_wo)
                                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                        attention_bq,
                                        attention_bk,
                                        attention_bv,
                                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                                        attn_q_norm,
                                        attn_k_norm,
                                        ffn: FfnVariant::Dense(Mlp {
                                            ffn_gate: Some(ffn_gate_mm),
                                            ffn_down: ffn_down_mm,
                                            ffn_up: ffn_up_mm,
                                            activation,
                                        }),
                                        ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                                        post_attention_norm,
                                        post_ffw_norm,
                                        n_head: head_count,
                                        n_kv_head: head_count_kv,
                                        head_dim,
                                        cos: cos.clone(),
                                        sin: sin.clone(),
                                        neg_inf: neg_inf.clone(),
                                        use_rope_contiguous,
                                        attn_logit_softcap,
                                        rope_dim,
                                        skip_rope: false,
                                    }))
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| {
                                h.join().unwrap_or_else(|_| {
                                    Err(SwarmError::Internal("layer load thread panicked".into()))
                                })
                            })
                            .collect()
                    });
                for result in layer_results {
                    layers.push(result?);
                }
            } else {
                // Sequential fallback for ShardReader (gaps between tensors prevent read_to_end).
                // Same per-layer logic as the parallel path, using ct.tensor(file, ...) directly.
                for layer_idx in layer_start..layer_end {
                    let prefix = format!("blk.{layer_idx}");
                    let has_fused_qkv = ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.attn_qkv.weight"));
                    let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                        let fused_qt = ct
                            .tensor(file, &format!("{prefix}.attn_qkv.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_qkv: {e}")))?;
                        let q_dim = head_count * head_dim;
                        let k_dim = head_count_kv * head_dim;
                        let fused = QMatMul::make_fused(fused_qt)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                        (
                            QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                            QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                            QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
                        )
                    } else {
                        let wq = ct
                            .tensor(file, &format!("{prefix}.attn_q.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                        let wk = ct
                            .tensor(file, &format!("{prefix}.attn_k.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                        let wv = ct
                            .tensor(file, &format!("{prefix}.attn_v.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                        (
                            QMatMul::from_qtensor(wq)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            QMatMul::from_qtensor(wk)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            QMatMul::from_qtensor(wv)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        )
                    };
                    let wo = ct
                        .tensor(file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attn_norm = ct
                        .tensor(file, &format!("{prefix}.attn_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                    let ffn_norm_t = ct
                        .tensor(file, &format!("{prefix}.ffn_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                    let bq = ct
                        .tensor(file, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bq: {e}")))?;
                    let bk = ct
                        .tensor(file, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bk: {e}")))?;
                    let bv = ct
                        .tensor(file, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bv: {e}")))?;

                    let post_attn_norm = ct
                        .tensor(
                            file,
                            &format!("{prefix}.post_attention_norm.weight"),
                            &device,
                        )
                        .ok();
                    let post_ffw_norm = ct
                        .tensor(file, &format!("{prefix}.post_ffw_norm.weight"), &device)
                        .ok();

                    // FFN: try separate gate/up first, fall back to fused gate_up (Phi-3)
                    // (MoE models hit their architecture-specific branches above, not here)
                    let has_ffn_gate = ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate.weight"));
                    let ffn_down_qt = ct
                        .tensor(file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn = if has_ffn_gate {
                        let gate = ct
                            .tensor(file, &format!("{prefix}.ffn_gate.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                        let up = ct
                            .tensor(file, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: Some(
                                QMatMul::from_qtensor(gate)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ),
                            ffn_down: QMatMul::from_qtensor(ffn_down_qt)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation,
                        })
                    } else {
                        // Fused gate+up (Phi-3): ffn_up = gate || up combined
                        let fused_qt = ct
                            .tensor(file, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        let fused_shape = fused_qt.shape();
                        let half = fused_shape.dims()[0] / 2;
                        let fused = QMatMul::make_fused(fused_qt)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: Some(QMatMul::from_fused_slice(fused.clone(), 0, half)),
                            ffn_down: QMatMul::from_qtensor(ffn_down_qt)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_fused_slice(fused, half, half),
                            activation,
                        })
                    };

                    let attn_q_norm = ct
                        .tensor(file, &format!("{prefix}.attn_q_norm.weight"), &device)
                        .ok();
                    let attn_k_norm = ct
                        .tensor(file, &format!("{prefix}.attn_k_norm.weight"), &device)
                        .ok();

                    layers.push(LayerVariant::Dense(LayerWeights {
                        attention_wq: qkv_q,
                        attention_wk: qkv_k,
                        attention_wv: qkv_v,
                        attention_wo: QMatMul::from_qtensor(wo)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        attention_bq: bq,
                        attention_bk: bk,
                        attention_bv: bv,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        attn_q_norm: attn_q_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        attn_k_norm: attn_k_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        ffn,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        post_attention_norm: post_attn_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        post_ffw_norm: post_ffw_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        neg_inf: neg_inf.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim,
                        skip_rope: false,
                    }));
                }
            }
        }

        // Extract tokenizer metadata via centralized GgufTokenizerMeta
        let tok_meta = super::gguf_meta::GgufTokenizerMeta::from_content(&ct);
        let vocabulary = if tok_meta.vocab.is_empty() {
            None
        } else {
            tracing::info!(vocab_size = tok_meta.vocab.len(), "Loaded GGUF vocabulary");
            Some(tok_meta.vocab.clone())
        };

        let tokenizer = tok_meta.build_tokenizer();
        if tokenizer.is_some() {
            tracing::info!(
                tokenizer_model = %tok_meta.tokenizer_model,
                pre_type = %tok_meta.pre_tokenizer,
                "Built tokenizer from GGUF metadata"
            );
        }

        // EOS tokens with architecture-specific fallbacks
        let eos_tokens = tok_meta.eos_tokens_with_arch_fallback(arch);
        tracing::info!(eos_tokens = ?eos_tokens, "Loaded EOS tokens from GGUF");

        // Chat template from GGUF metadata
        let chat_template = tok_meta.chat_template.filter(|s| !s.is_empty());

        // Resolve BOS/EOS token strings from their IDs + vocabulary
        let bos_id = ct
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());
        let eos_str_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());
        let vocab_ref = vocabulary.as_deref().unwrap_or(&[]);
        let bos_token = bos_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();
        let eos_token = eos_str_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();

        if let Some(ref tmpl) = chat_template {
            tracing::info!(
                bos = %bos_token,
                eos = %eos_token,
                template_len = tmpl.len(),
                template_preview = &tmpl[..tmpl.len().min(200)],
                "Loaded chat template from GGUF header"
            );
        }

        let has_biases = layers
            .first()
            .is_some_and(|l| matches!(l, LayerVariant::Dense(lw) if lw.attention_bq.is_some()));
        tracing::info!(
            arch = %model_arch,
            layers = format!("[{layer_start}..{layer_end})"),
            total = block_count,
            is_first,
            is_last,
            has_qkv_biases = has_biases,
            rope = if use_rope_contiguous { "contiguous" } else { "interleaved" },
            activation = ?activation,
            context_length,
            "Loaded split model segment"
        );

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            masks: None,
            layer_start,
            layer_end,
            total_layers: block_count,
            hidden_dim: embedding_length,
            arch: model_arch,
            device,
            vocabulary,
            tokenizer,
            eos_tokens,
            chat_template,
            bos_token,
            eos_token,
            max_seq_len: context_length,
            kv_model_key: format!("{layer_start}-{layer_end}-{block_count}"),
            final_logit_softcap,
        })
    }

    /// Load from shards, forcing CPU device (used as GPU OOM fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards_cpu(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            true,
        )
    }

    /// Load a partial model from local shard files + GGUF header.
    ///
    /// This is the shard-only alternative to `load_from_gguf`. Instead of needing
    /// the full GGUF file, it reads from:
    /// - `gguf_header.bin`: the raw GGUF header (metadata + tensor info table)
    /// - `shard_NNN.bin` files: layer-aligned shard files with packed tensor data
    ///
    /// The `ShardReader` uses the tensor entries to map virtual GGUF positions
    /// to shard-local offsets, so candle's GGUF parser works unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_from_shards_inner(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        force_cpu: bool,
    ) -> Result<Self, SwarmError> {
        let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
        if !header_path.exists() {
            return Err(SwarmError::Internal(format!(
                "GGUF header not found at {}. The originating node must generate this file.",
                header_path.display()
            )));
        }

        // Single shard with no tensor entries = full GGUF file as shard.
        // Load directly via mmap instead of ShardReader.
        let has_tensor_entries = tensor_entries.iter().any(|v| !v.is_empty());
        if shard_files.len() == 1 && !has_tensor_entries {
            let shard_path = &shard_files[0].1;
            tracing::info!(
                model_dir = %model_dir.display(),
                shard_path = %shard_path.display(),
                "Single-shard model with no tensor entries — loading as full GGUF via mmap"
            );
            // Respect force_cpu — don't delegate to load_from_gguf which always picks CUDA
            let file = std::fs::File::open(shard_path).map_err(SwarmError::Io)?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| SwarmError::Internal(format!("Failed to mmap GGUF: {e}")))?;
            let mut cursor = std::io::Cursor::new(mmap.as_ref());
            let ct = gguf_file::Content::read(&mut cursor)
                .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF: {e}")))?;
            let device = if force_cpu {
                Device::Cpu
            } else {
                Device::cuda_if_available(0).unwrap_or(Device::Cpu)
            };
            return Self::load_model_from_content(
                ct,
                &mut cursor,
                device,
                layer_start,
                layer_end,
                is_first,
                is_last,
                Some(mmap.as_ref()),
            );
        }

        // Read header to get tensor_data_offset
        let header_bytes = std::fs::read(&header_path).map_err(SwarmError::Io)?;
        let tensor_data_offset = {
            let mut cursor = std::io::Cursor::new(&header_bytes);
            let ct = gguf_file::Content::read(&mut cursor)
                .map_err(|e| SwarmError::Internal(format!("Failed to parse GGUF header: {e}")))?;
            ct.tensor_data_offset
        };

        tracing::info!(
            model_dir = %model_dir.display(),
            header_bytes = header_bytes.len(),
            tensor_data_offset,
            shards = shard_files.len(),
            layers = format!("[{layer_start}..{layer_end})"),
            "Loading split model from shard files"
        );

        let mut reader = ShardReader::new(
            &header_path,
            shard_files,
            tensor_entries,
            total_gguf_size,
            tensor_data_offset,
        )?;

        // Use the same GGUF parsing path as load_from_gguf, but reading from ShardReader
        let ct = gguf_file::Content::read(&mut reader).map_err(|e| {
            SwarmError::Internal(format!("Failed to read GGUF via ShardReader: {e}"))
        })?;

        // Verify tensor_data_offset matches between the two Content::read calls
        if ct.tensor_data_offset != tensor_data_offset {
            tracing::error!(
                expected = tensor_data_offset,
                actual = ct.tensor_data_offset,
                "DIAG: tensor_data_offset MISMATCH between header parse and ShardReader parse!"
            );
        }

        // Diagnostic: log first few tensor offsets from Content vs tensor_map
        for (name, info) in ct.tensor_infos.iter().take(5) {
            let seek_pos = ct.tensor_data_offset + info.offset;
            let size_in_bytes = info.ggml_dtype.type_size() * info.shape.elem_count()
                / info.ggml_dtype.block_size();
            let found = reader.find_shard(seek_pos);
            tracing::info!(
                tensor = %name,
                gguf_seek = seek_pos,
                size = size_in_bytes,
                shard_mapping = ?found,
                "DIAG: tensor mapping check"
            );
        }

        // DIAG: Read first 16 bytes of blk.0.attn_norm.weight via ShardReader
        // to verify data integrity
        if let Some(norm_info) = ct.tensor_infos.get("blk.0.attn_norm.weight") {
            use std::io::{Read as IoReadTrait, Seek as SeekTrait};
            let seek_pos = ct.tensor_data_offset + norm_info.offset;
            reader.seek(SeekFrom::Start(seek_pos)).ok();
            let mut probe = [0u8; 16];
            if reader.read_exact(&mut probe).is_ok() {
                tracing::info!(
                    seek_pos,
                    first_bytes = ?&probe,
                    "DIAG: blk.0.attn_norm.weight first 16 bytes via ShardReader"
                );
            }
        }

        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        if device.is_cuda() {
            tracing::info!(layer_start, layer_end, "Split model using CUDA GPU");
        } else if force_cpu {
            tracing::info!(
                layer_start,
                layer_end,
                "Split model using CPU (GPU OOM fallback)"
            );
        } else {
            tracing::info!(
                layer_start,
                layer_end,
                "Split model using CPU (no CUDA available)"
            );
        }

        Self::load_model_from_content(
            ct,
            &mut reader,
            device,
            layer_start,
            layer_end,
            is_first,
            is_last,
            None, // ShardReader can't be shared across threads for parallel loading
        )
    }

    /// Build a causal mask for the given sequence length.
    ///
    /// Pre-allocates a single mask at a ceiling size (min of max_seq_len and 4096),
    /// then uses `narrow()` to slice views for smaller sequences — zero-copy.
    /// Only re-allocates if `t` exceeds the current ceiling.
    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        // Fast path: existing mask is large enough — narrow-slice a view (no copy)
        if let Some((cached_size, ref mask)) = self.masks {
            if t <= cached_size {
                return mask.narrow(0, 0, t)?.narrow(1, 0, t);
            }
        }
        // Allocate at a reasonable ceiling to amortize future requests.
        // Cap at 4096 to avoid excessive memory (4096^2 = 16MB for u8).
        let alloc_size = t.max(self.max_seq_len.min(4096));
        let mask_data: Vec<_> = (0..alloc_size)
            .flat_map(|i| (0..alloc_size).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (alloc_size, alloc_size), &self.device)?;
        self.masks = Some((alloc_size, mask.clone()));
        if t < alloc_size {
            mask.narrow(0, 0, t)?.narrow(1, 0, t)
        } else {
            Ok(mask)
        }
    }

    /// Build a causal mask with KV offset for prefix-cached inference.
    ///
    /// When the KV cache has been pre-populated with prefix tokens,
    /// query tokens (suffix) attend to all prefix positions plus earlier suffix
    /// positions with proper causal ordering.
    ///
    /// `query_len`: number of new (suffix) tokens being processed.
    /// `kv_len`: total KV length (prefix_len + query_len).
    fn mask_with_offset(&self, query_len: usize, kv_len: usize) -> CandleResult<Tensor> {
        let offset = kv_len - query_len;
        let mask: Vec<_> = (0..query_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        Tensor::from_slice(&mask, (query_len, kv_len), &self.device)
    }

    /// Run the forward pass for this segment's layer range.
    ///
    /// - For the first segment: `input` is token IDs (i64 tensor, shape [1, seq_len]).
    ///   We apply the embedding lookup and return hidden states.
    /// - For intermediate segments: `input` is hidden state activations (f32, [1, seq, hidden_dim]).
    /// - For the last segment: returns logits (f32, [vocab_size]) for the last token position.
    /// - For intermediate segments: returns hidden states (f32, [1, seq, hidden_dim]).
    ///
    /// `kv_cache_store` and `request_id` provide per-request KV-cache isolation.
    /// The cache is stored externally in the `KvCacheStore`, keyed by request_id.
    pub fn forward(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        self.forward_with_lora(input, index_pos, kv_cache_store, request_id, None)
    }

    /// Forward pass with pre-embedded hidden states (local embedding privacy).
    ///
    /// The input tensor is already in hidden-state space (shape [1, seq, hidden_dim])
    /// — the embedding lookup is skipped even if this segment has `tok_embeddings`.
    /// Used when the requesting node performed embedding locally for privacy.
    pub fn forward_pre_embedded(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            true,
        )?;
        Ok(output)
    }

    /// Forward pass with optional LoRA adapter applied per-layer.
    ///
    /// When `lora_adapter` is `Some`, the adapter's low-rank deltas are applied
    /// to attention (Q/K/V/O) and MLP (gate/up/down) projections at each layer.
    pub fn forward_with_lora(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            None,
        )?;
        Ok(output)
    }

    /// Inner forward pass implementation. When `capture_layers` is Some, captures
    /// hidden states at the specified absolute layer indices.
    fn forward_inner(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            capture_layers,
            false,
        )
    }

    /// Core forward pass. When `skip_embedding` is true, the input is treated as
    /// pre-embedded hidden states even if this segment has `tok_embeddings`.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner_impl(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
        skip_embedding: bool,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        let forward_start = std::time::Instant::now();
        // Use component presence rather than layer indices for shard-aware is_first/is_last
        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Move input to model's device if needed (e.g. CPU → CUDA)
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer failed: {e}")))?;

        // Determine the hidden state to start from
        let mut layer_in = if is_first && !skip_embedding {
            // First segment: input is token IDs → apply embedding
            let mut emb = self
                .tok_embeddings
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                .forward(&input)
                .map_err(|e| SwarmError::Internal(format!("Embedding forward failed: {e}")))?;
            // Gemma models scale embeddings by sqrt(hidden_dim)
            if self.arch.use_gemma_norm() {
                let scale = (self.hidden_dim as f64).sqrt();
                emb = emb
                    .affine(scale, 0.0)
                    .map_err(|e| SwarmError::Internal(format!("Embedding scale failed: {e}")))?;
            }
            emb
        } else {
            // Non-first segment or pre-embedded: input is already hidden states
            input
        };

        // Get seq_len for mask
        let seq_len = layer_in
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        // Pre-flight check: reject sequences that exceed the model's context window
        // to avoid cryptic tensor dimension errors in attention.
        let total_seq = index_pos + seq_len;
        if total_seq > self.max_seq_len {
            return Err(SwarmError::Validation(format!(
                "Sequence length ({total_seq}) exceeds model context window ({}). \
                 Reduce your prompt or max_tokens.",
                self.max_seq_len
            )));
        }

        let num_layers = self.layers.len();
        // Build the cache key once — reused for both take and writeback (zero alloc on hot path).
        let cache_key = KvCacheStore::cache_key(&self.kv_model_key, request_id);

        // Get or create the per-request cache entry, extract the layer caches,
        // then drop the DashMap guard before running the (potentially slow) forward pass.
        // Use mem::take instead of clone to avoid copying the KV cache Vec.
        #[allow(unused_mut)]
        let (mut layer_kv_caches, mut layer_ssm_states): (
            Vec<Option<KvCache>>,
            Vec<Option<SsmState>>,
        ) = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            (
                std::mem::take(&mut entry.layers),
                std::mem::take(&mut entry.ssm_states),
            )
        };

        // Detect pre-populated prefix cache entries (KV already present from prefix
        // cache restoration). If so, build an offset causal mask that allows the
        // suffix query tokens to attend to all prefix KV positions.
        let kv_offset = layer_kv_caches
            .first()
            .and_then(|c| c.as_ref())
            .map(|c| c.current_seq_len())
            .unwrap_or(0);

        let mask = if seq_len == 1 {
            None
        } else if kv_offset > 0 {
            // Prefix cache: suffix query attends to (offset + seq_len) key positions
            Some(
                self.mask_with_offset(seq_len, kv_offset + seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            Some(
                self.mask(seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        };

        let max_seq_len = self.max_seq_len;
        let mut captured: HashMap<usize, Tensor> = HashMap::new();

        // Run through our layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let abs_layer = self.layer_start + layer_idx;
            let lora_param = lora_adapter.map(|a| (a, abs_layer));

            let layer_start_time = std::time::Instant::now();
            match layer {
                LayerVariant::Dense(lw) => {
                    let x = layer_in;
                    let residual = &x;
                    let x = lw
                        .attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
                    let mut attn = lw
                        .forward_attn(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                            lora_param,
                        )
                        .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                    // Gemma 2 post-attention norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn = post_norm
                            .forward(&attn)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (attn + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual = &x;
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, lora_param)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    // Gemma 2 post-FFN norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
                    layer_in = (x + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    let x = attention_norm
                        .forward(&layer_in)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;
                    let attn = attention
                        .forward_mla(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                        )
                        .map_err(|e| SwarmError::Internal(format!("mla: {e}")))?;
                    let x = (attn + &layer_in).map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let residual = &x;
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    layer_in =
                        (ffn_out + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Attn {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    let x = layer_in;
                    let residual = &x;
                    let x = attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_attn_norm: {e}")))?;
                    let attn = weights
                        .forward_attn(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                        )
                        .map_err(|e| SwarmError::Internal(format!("q35_attn: {e}")))?;
                    let x = (attn + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let residual = &x;
                    let normed = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_post_attn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("q35_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("q35_moe: {e}")))?,
                    };
                    layer_in =
                        (ffn_out + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Ssm {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    let x = layer_in;
                    let residual = &x;
                    let x = attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_ssm_norm: {e}")))?;
                    let ssm_out = weights
                        .forward_deltanet(&x, &mut layer_ssm_states[layer_idx])
                        .map_err(|e| SwarmError::Internal(format!("q35_deltanet: {e}")))?;
                    let x =
                        (ssm_out + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let residual = &x;
                    let normed = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_post_ssm_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("q35_ssm_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("q35_ssm_moe: {e}")))?,
                    };
                    layer_in =
                        (ffn_out + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
            }
            tracing::trace!(
                layer = abs_layer,
                layer_ms = layer_start_time.elapsed().as_millis() as u64,
                "DIAG: layer forward complete"
            );

            // Capture hidden state if requested (zero overhead when not capturing)
            if let Some(layers_to_capture) = capture_layers {
                if layers_to_capture.contains(&abs_layer) {
                    captured.insert(abs_layer, layer_in.clone());
                }
            }
        }

        // Write the updated KV-caches and SSM states back to the store.
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
            entry.ssm_states = layer_ssm_states;
            entry.last_accessed = std::time::Instant::now();
        }

        let result = if is_last {
            // Last segment: apply final norm, extract last token, project to logits
            let norm = self
                .norm
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
            let output = self
                .output
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;

            let x = norm
                .forward(&layer_in)
                .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
            let x = x
                .i((.., seq_len - 1, ..))
                .map_err(|e| SwarmError::Internal(format!("last_token_select: {e}")))?;
            let mut logits = output
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
            // Gemma 2 final logit soft-capping: tanh(logits / cap) * cap
            if let Some(cap) = self.final_logit_softcap {
                logits = logits
                    .affine(1.0 / cap as f64, 0.0)
                    .and_then(|t| t.tanh())
                    .and_then(|t| t.affine(cap as f64, 0.0))
                    .map_err(|e| SwarmError::Internal(format!("final_logit_softcap: {e}")))?;
            }
            Ok(logits)
        } else {
            // Intermediate segment: return hidden states for next segment
            Ok(layer_in)
        };

        let forward_ms = forward_start.elapsed().as_millis() as u64;
        tracing::debug!(
            request_id,
            index_pos,
            seq_len,
            num_layers,
            is_first,
            is_last,
            kv_offset,
            forward_ms,
            "DIAG: SplitModel forward pass complete"
        );

        result.map(|t| (t, captured))
    }

    /// Tensor-parallel forward pass for a single layer.
    ///
    /// Each TP node computes only its fraction of the computation:
    /// - Attention: processes `n_head / tp_size` heads (head-parallel)
    /// - MLP: processes `intermediate_dim / tp_size` columns (column-parallel gate/up,
    ///   row-parallel down)
    ///
    /// Returns a **partial** hidden state that must be summed (AllReduced) across all
    /// Forward pass for multimodal (vision + text) inference.
    ///
    /// If this is the first segment and `vision_embeddings` is provided, the input
    /// is expected to contain TWO segments of token IDs separated by a marker:
    /// `[before_image_tokens...][MARKER=-1][after_image_tokens...]`
    ///
    /// The marker value -1 (as i64) indicates where vision embeddings should be
    /// inserted. Each part is embedded separately, then concatenated:
    /// `[before_emb][vision_emb][after_emb]`
    ///
    /// This matches llama.cpp's LLaVA approach of splitting the prompt at `<image>`.
    ///
    /// `vision_embeddings` shape: (num_image_tokens, hidden_dim)
    pub fn forward_multimodal(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        vision_embeddings: Option<&Tensor>,
    ) -> Result<Tensor, SwarmError> {
        let is_first = self.tok_embeddings.is_some();

        match (is_first, vision_embeddings) {
            (true, Some(vision_emb)) => {
                let input_dev = input
                    .to_device(&self.device)
                    .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

                // Read the token IDs to find the -1 marker
                let token_ids: Vec<i64> = input_dev
                    .flatten_all()
                    .map_err(|e| SwarmError::Internal(format!("flatten: {e}")))?
                    .to_vec1()
                    .map_err(|e| SwarmError::Internal(format!("to_vec1: {e}")))?;

                let marker_pos = token_ids.iter().position(|&id| id == -1);

                let tok_emb_layer = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?;

                let merged = if let Some(pos) = marker_pos {
                    // Split at marker: embed before and after separately, insert vision between
                    let num_vision = vision_emb
                        .dim(0)
                        .map_err(|e| SwarmError::Internal(format!("vision dim: {e}")))?;
                    let hidden = vision_emb
                        .dim(1)
                        .map_err(|e| SwarmError::Internal(format!("vision dim1: {e}")))?;

                    // All parts must be (1, seq, hidden) for cat along dim 1
                    let vision_3d = vision_emb
                        .reshape(&[1, num_vision, hidden])
                        .map_err(|e| SwarmError::Internal(format!("vision reshape: {e}")))?;

                    let mut parts: Vec<Tensor> = Vec::new();

                    // Embed tokens before <image>
                    if pos > 0 {
                        let before_ids = Tensor::new(&token_ids[..pos], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("before tensor: {e}")))?
                            .reshape(&[1, pos])
                            .map_err(|e| SwarmError::Internal(format!("before reshape: {e}")))?;
                        let mut before_emb = tok_emb_layer
                            .forward(&before_ids)
                            .map_err(|e| SwarmError::Internal(format!("before embed: {e}")))?;
                        // Apply Gemma embedding scale (must match forward_inner_impl)
                        if self.arch.use_gemma_norm() {
                            let scale = (self.hidden_dim as f64).sqrt();
                            before_emb = before_emb
                                .affine(scale, 0.0)
                                .map_err(|e| SwarmError::Internal(format!("gemma scale: {e}")))?;
                        }
                        // Ensure 3D: (1, pos, hidden)
                        let before_3d = if before_emb.dims().len() == 2 {
                            before_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("before unsqueeze: {e}"))
                            })?
                        } else {
                            before_emb
                        };
                        parts.push(before_3d);
                    }

                    // Insert vision embeddings
                    parts.push(vision_3d);

                    // Embed tokens after <image>
                    let after_len = token_ids.len() - pos - 1;
                    if after_len > 0 {
                        let after_ids = Tensor::new(&token_ids[pos + 1..], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("after tensor: {e}")))?
                            .reshape(&[1, after_len])
                            .map_err(|e| SwarmError::Internal(format!("after reshape: {e}")))?;
                        let mut after_emb = tok_emb_layer
                            .forward(&after_ids)
                            .map_err(|e| SwarmError::Internal(format!("after embed: {e}")))?;
                        // Apply Gemma embedding scale (must match forward_inner_impl)
                        if self.arch.use_gemma_norm() {
                            let scale = (self.hidden_dim as f64).sqrt();
                            after_emb = after_emb
                                .affine(scale, 0.0)
                                .map_err(|e| SwarmError::Internal(format!("gemma scale: {e}")))?;
                        }
                        let after_3d = if after_emb.dims().len() == 2 {
                            after_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("after unsqueeze: {e}"))
                            })?
                        } else {
                            after_emb
                        };
                        parts.push(after_3d);
                    }

                    let refs: Vec<&Tensor> = parts.iter().collect();
                    tracing::info!(
                        request_id,
                        marker_pos = pos,
                        before_tokens = pos,
                        after_tokens = after_len,
                        vision_tokens = num_vision,
                        "VLM: inserting vision embeddings at <image> position"
                    );
                    Tensor::cat(&refs, 1)
                        .map_err(|e| SwarmError::Internal(format!("merge cat: {e}")))?
                } else {
                    // No marker found — fallback to prepending vision before text
                    let text_embeddings = tok_emb_layer
                        .forward(&input_dev)
                        .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;
                    crate::inference::vision::merge_vision_text_embeddings(
                        &text_embeddings,
                        vision_emb,
                        &[],
                    )?
                };

                tracing::info!(
                    request_id,
                    merged_seq_len = ?merged.dims(),
                    "VLM: merged embeddings ready for transformer"
                );

                // Run through layers with merged hidden states.
                // Temporarily remove tok_embeddings so forward() treats input as hidden states.
                let tok_emb = self.tok_embeddings.take();
                let result = self.forward(&merged, index_pos, kv_cache_store, request_id);
                self.tok_embeddings = tok_emb;
                result
            }
            _ => {
                // No vision embeddings or not the first segment — regular forward
                self.forward(input, index_pos, kv_cache_store, request_id)
            }
        }
    }

    /// Run a batched forward pass for multiple decode-step requests (seq_len=1 each).
    ///
    /// Stacks inputs along the batch dimension so that MLP/norm computations
    /// benefit from GPU parallelism.  Attention is still per-request because
    /// each request has its own `index_pos` and KV-cache.
    ///
    /// Returns one output tensor per request in the same order as `items`.
    /// Falls back to sequential `forward()` if any item has seq_len > 1.
    #[cfg(test)]
    pub fn forward_batch(
        &mut self,
        items: &[BatchItem<'_>],
        kv_cache_store: &KvCacheStore,
    ) -> Result<Vec<Tensor>, SwarmError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Fallback: if only 1 item or any item is a prefill (seq_len > 1), run sequentially
        if items.len() == 1 {
            let item = &items[0];
            let out = self.forward(item.input, item.index_pos, kv_cache_store, item.request_id)?;
            return Ok(vec![out]);
        }

        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Convert each input to its hidden state (apply embedding if first segment)
        let mut per_request: Vec<Tensor> = Vec::with_capacity(items.len());
        for item in items {
            let input = item
                .input
                .to_device(&self.device)
                .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;
            let hidden = if is_first {
                let mut emb = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                    .forward(&input)
                    .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;
                // Apply Gemma embedding scale (sqrt(hidden_dim)) — matches forward_inner_impl
                if self.arch.use_gemma_norm() {
                    let scale = (self.hidden_dim as f64).sqrt();
                    emb = emb
                        .affine(scale, 0.0)
                        .map_err(|e| SwarmError::Internal(format!("Gemma scale: {e}")))?;
                }
                emb
            } else {
                input
            };
            per_request.push(hidden);
        }

        // Check if all items have seq_len=1 (decode mode) — only then can we batch
        let all_decode = per_request.iter().all(|t| t.dim(1).unwrap_or(0) == 1);

        if !all_decode {
            // Mixed or prefill batch: fall back to sequential processing
            let mut results = Vec::with_capacity(items.len());
            for item in items {
                results.push(self.forward(
                    item.input,
                    item.index_pos,
                    kv_cache_store,
                    item.request_id,
                )?);
            }
            return Ok(results);
        }

        // Context window check: reject any item whose index_pos exceeds max_seq_len
        // (same guard as forward_inner_impl, prevents RoPE table out-of-bounds).
        for item in items {
            if item.index_pos + 1 > self.max_seq_len {
                return Err(SwarmError::Validation(format!(
                    "Batch item index_pos ({}) exceeds model context window ({})",
                    item.index_pos + 1,
                    self.max_seq_len
                )));
            }
        }

        let batch_size = items.len();
        let model_key = &self.kv_model_key;
        let num_layers = self.layers.len();

        // Extract all per-request KV-caches and SSM states up front (drop DashMap guards immediately).
        // Use mem::take instead of clone to avoid deep-copying all KV tensors.
        let mut all_kv_caches: Vec<Vec<Option<KvCache>>> = Vec::with_capacity(batch_size);
        let mut all_ssm_states: Vec<Vec<Option<SsmState>>> = Vec::with_capacity(batch_size);
        for item in items.iter() {
            let mut entry = kv_cache_store.get_or_create(model_key, item.request_id, num_layers);
            entry.last_accessed = std::time::Instant::now();
            all_kv_caches.push(std::mem::take(&mut entry.layers));
            all_ssm_states.push(std::mem::take(&mut entry.ssm_states));
        }

        let max_seq_len = self.max_seq_len;

        // Stack all hidden states into a single batch tensor: [batch, 1, hidden_dim]
        let batch_refs: Vec<&Tensor> = per_request.iter().collect();
        let mut batched = Tensor::cat(&batch_refs, 0)
            .map_err(|e| SwarmError::Internal(format!("Batch stack: {e}")))?;
        // Shape is now [batch_size, 1, hidden_dim]

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerVariant::Dense(lw) => {
                    let residual = batched.clone();
                    let normed = lw
                        .attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = lw
                            .forward_attn(
                                &x_i,
                                None,
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                                None,
                            )
                            .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let mut attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("attn restack: {e}")))?;
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn_batched = post_norm
                            .forward(&attn_batched)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (&attn_batched + &residual)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual2 = x.clone();
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
                    batched = (&x + &residual2).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    // DeepSeek batch: per-request attention (MLA), batched FFN
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = attention
                            .forward_mla(
                                &x_i,
                                None,
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                            )
                            .map_err(|e| SwarmError::Internal(format!("mla_batch: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("mla restack: {e}")))?;
                    let x = (&attn_batched + &batched)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual = x.clone();
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe_batch: {e}")))?,
                    };
                    batched =
                        (&ffn_out + &residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Attn {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    // Qwen 3.5 attention: per-request attention + batched FFN (same as Dense pattern)
                    let residual = batched.clone();
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("q35b_attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = weights
                            .forward_attn(
                                &x_i,
                                None,
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                            )
                            .map_err(|e| SwarmError::Internal(format!("q35b_attn: {e}")))?;
                        attn_outputs.push(attn_out);
                    }
                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("q35b_attn_restack: {e}")))?;
                    let x = (&attn_batched + &residual)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual2 = x.clone();
                    let normed2 = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35b_post_attn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed2, None)
                            .map_err(|e| SwarmError::Internal(format!("q35b_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed2)
                            .map_err(|e| SwarmError::Internal(format!("q35b_moe: {e}")))?,
                    };
                    batched =
                        (&ffn_out + &residual2).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Ssm {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    // Qwen 3.5 SSM: per-request DeltaNet (SSM state is per-request) + batched FFN
                    let residual = batched.clone();
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("q35b_ssm_norm: {e}")))?;

                    let mut ssm_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, req_ssm) in all_ssm_states.iter_mut().enumerate().take(batch_size)
                    {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let ssm_out = weights
                            .forward_deltanet(&x_i, &mut req_ssm[layer_idx])
                            .map_err(|e| SwarmError::Internal(format!("q35b_deltanet: {e}")))?;
                        ssm_outputs.push(ssm_out);
                    }
                    let ssm_refs: Vec<&Tensor> = ssm_outputs.iter().collect();
                    let ssm_batched = Tensor::cat(&ssm_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("q35b_ssm_restack: {e}")))?;
                    let x = (&ssm_batched + &residual)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual2 = x.clone();
                    let normed2 = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35b_post_ssm_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed2, None)
                            .map_err(|e| SwarmError::Internal(format!("q35b_ssm_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed2)
                            .map_err(|e| SwarmError::Internal(format!("q35b_ssm_moe: {e}")))?,
                    };
                    batched =
                        (&ffn_out + &residual2).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
            }
        }

        // Write updated KV-caches and SSM states back (take instead of clone to avoid copying)
        for (req_idx, item) in items.iter().enumerate() {
            let mut entry = kv_cache_store.get_or_create(model_key, item.request_id, num_layers);
            entry.layers = std::mem::take(&mut all_kv_caches[req_idx]);
            entry.ssm_states = std::mem::take(&mut all_ssm_states[req_idx]);
            entry.last_accessed = std::time::Instant::now();
        }

        // Split batch back into per-request outputs
        let mut results = Vec::with_capacity(batch_size);
        for req_idx in 0..batch_size {
            let per_req = batched
                .narrow(0, req_idx, 1)
                .map_err(|e| SwarmError::Internal(format!("split: {e}")))?;

            if is_last {
                let norm = self
                    .norm
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
                let output = self
                    .output
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;
                let x = norm
                    .forward(&per_req)
                    .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
                // seq_len=1, so i((.., 0, ..)) selects the single token
                let x = x
                    .i((.., 0, ..))
                    .map_err(|e| SwarmError::Internal(format!("last_token: {e}")))?;
                let mut logits = output
                    .forward(&x)
                    .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
                // Apply final logit softcap for Gemma 2 (must match forward_inner_impl)
                if let Some(cap) = self.final_logit_softcap {
                    logits = logits
                        .affine(1.0 / cap as f64, 0.0)
                        .and_then(|t| t.tanh())
                        .and_then(|t| t.affine(cap as f64, 0.0))
                        .map_err(|e| SwarmError::Internal(format!("final_logit_softcap: {e}")))?;
                }
                results.push(logits);
            } else {
                results.push(per_req);
            }
        }

        Ok(results)
    }

    /// Return a reference to the loaded vocabulary, if available.
    pub fn vocab(&self) -> Option<&Vec<String>> {
        self.vocabulary.as_ref()
    }

    /// Return a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&SplitTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the device this model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the number of KV heads from the first layer.
    pub fn n_kv_head(&self) -> usize {
        self.layers
            .first()
            .map(|l| match l {
                LayerVariant::Dense(w) => w.n_kv_head,
                LayerVariant::DeepSeek { .. } => 1, // MLA uses MQA
                LayerVariant::Qwen35Attn { weights, .. } => weights.n_kv_head,
                LayerVariant::Qwen35Ssm { .. } => 0,
            })
            .unwrap_or(1)
    }

    /// Pre-split all layer weights for tensor parallelism.
    ///
    /// Reduces VRAM per rank by splitting attention heads and FFN columns.
    /// Must be called once before any TP forward passes.
    pub fn pre_split_for_tp(&mut self, tp_rank: usize, tp_size: usize) -> Result<(), SwarmError> {
        if tp_size <= 1 {
            return Ok(());
        }
        let device = self.device.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let LayerVariant::Dense(ref mut lw) = layer {
                lw.pre_split_for_tp(tp_rank, tp_size, &device)
                    .map_err(|e| SwarmError::Internal(format!("TP split layer {i}: {e}")))?;
            }
        }
        tracing::info!(
            tp_rank,
            tp_size,
            layers = self.layers.len(),
            "Pre-split weights for tensor parallelism"
        );
        Ok(())
    }

    /// Forward a single layer in a specific TP phase (AttnOnly or FfnOnly).
    ///
    /// Unlike the full `forward()`, this processes exactly ONE layer and does NOT
    /// add the residual connection — the coordinator adds residuals after AllReduce.
    ///
    /// - **AttnOnly**: attention_norm → head-sliced attention → return partial
    /// - **FfnOnly**: ffn_norm → column-sliced FFN → return partial
    ///
    /// The model must have been pre-split via `pre_split_for_tp()`.
    pub fn forward_tp_phase(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        abs_layer: usize,
        phase: &crate::types::TpPhase,
    ) -> Result<Tensor, SwarmError> {
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

        let local_layer_idx = abs_layer.checked_sub(self.layer_start).ok_or_else(|| {
            SwarmError::Internal(format!(
                "TP layer {abs_layer} below model range [{}, {})",
                self.layer_start,
                self.layer_start + self.layers.len()
            ))
        })?;
        if local_layer_idx >= self.layers.len() {
            return Err(SwarmError::Internal(format!(
                "TP layer {abs_layer} above model range [{}, {})",
                self.layer_start,
                self.layer_start + self.layers.len()
            )));
        }

        let seq_len = input
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(
                self.mask(seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        };

        let num_layers = self.layers.len();
        let cache_key = KvCacheStore::cache_key(&self.kv_model_key, request_id);
        let mut layer_kv_caches = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            std::mem::take(&mut entry.layers)
        };

        let result = match &self.layers[local_layer_idx] {
            LayerVariant::Dense(lw) => match phase {
                crate::types::TpPhase::AttnOnly => {
                    let x = lw
                        .attention_norm
                        .forward(&input)
                        .map_err(|e| SwarmError::Internal(format!("tp attn_norm: {e}")))?;
                    lw.forward_attn(
                        &x,
                        mask.as_ref(),
                        index_pos,
                        &mut layer_kv_caches[local_layer_idx],
                        self.max_seq_len,
                        None,
                    )
                    .map_err(|e| SwarmError::Internal(format!("tp attn: {e}")))
                }
                crate::types::TpPhase::FfnOnly => {
                    let x = lw
                        .ffn_norm
                        .forward(&input)
                        .map_err(|e| SwarmError::Internal(format!("tp ffn_norm: {e}")))?;
                    match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("tp mlp: {e}"))),
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("tp moe: {e}"))),
                    }
                }
                crate::types::TpPhase::Full => Err(SwarmError::Internal(
                    "TpPhase::Full not valid for single-layer TP".into(),
                )),
            },
            _ => Err(SwarmError::Internal(
                "TP not supported for non-Dense layers".into(),
            )),
        }?;

        // Write back KV caches
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
        }

        Ok(result)
    }

    /// Return the KV cache model key (used for cache cleanup).
    pub fn kv_model_key(&self) -> &str {
        &self.kv_model_key
    }

    /// Return the EOS token IDs loaded from GGUF metadata.
    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    /// Return the chat template from GGUF metadata, if available.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Return the BOS token string from GGUF metadata.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// Return the EOS token string from GGUF metadata.
    pub fn eos_token_str(&self) -> &str {
        &self.eos_token
    }

    /// Whether this segment has the embedding table (first segment).
    pub fn is_first(&self) -> bool {
        self.tok_embeddings.is_some()
    }

    /// Whether this segment has the output projection (last segment).
    pub fn is_last(&self) -> bool {
        self.output.is_some()
    }

    /// Tokenize a prompt string and return the embedded hidden states.
    ///
    /// Used by tensor-parallel execution where embedding happens before
    /// layer-by-layer forwarding. Only works on the first segment (has embeddings).
    /// Tokenize a prompt and return (token_ids_tensor, num_tokens).
    ///
    /// Returns the token IDs as an I64 tensor with shape (1, seq_len),
    /// suitable for passing directly to `forward()` which handles embedding.
    pub fn tokenize(&self, prompt: &str) -> Result<(Tensor, usize), SwarmError> {
        let token_ids: Vec<i64> = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt)
        } else {
            prompt.bytes().map(|b| b as i64).collect()
        };
        let num_tokens = token_ids.len();
        // DIAG: dump token IDs for debugging tokenizer issues
        tracing::info!(
            num_tokens,
            tokens = ?&token_ids[..token_ids.len().min(30)],
            "DIAG: tokenize result"
        );
        let input = Tensor::new(&token_ids[..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;
        Ok((input, num_tokens))
    }

    /// Create a single-token tensor for autoregressive decoding.
    ///
    /// Returns an I64 tensor with shape (1, 1), suitable for `forward()`.
    pub fn token_tensor(&self, token_id: u32) -> Result<Tensor, SwarmError> {
        Tensor::new(&[token_id as i64][..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))
    }

    /// Estimate GPU memory usage in MB for this model segment.
    ///
    /// Uses a rough heuristic: for quantized models (Q4/Q5/Q6), each parameter
    /// uses ~0.5-0.75 bytes. We estimate based on `num_layers * hidden_dim^2 * 4`
    /// (4 weight matrices per layer: Q, K, V, O + MLP) with a quantization factor.
    pub fn estimate_vram_mb(&self) -> u64 {
        let num_layers = self.layers.len() as u64;
        let hidden = self.hidden_dim as u64;

        // Each layer has roughly:
        //   4 attention matrices (Q, K, V, O): ~4 * hidden^2 params
        //   3 MLP matrices (gate, up, down): gate + up = 2 * hidden * 4*hidden, down = 4*hidden * hidden
        //   So MLP ≈ 12 * hidden^2 params
        //   Total per layer ≈ 16 * hidden^2 params
        //   Quantized at ~0.5 bytes per param (Q4)
        let params_per_layer = 16 * hidden * hidden;
        let bytes_per_param_quantized: f64 = 0.5;
        let layer_bytes = (params_per_layer as f64 * bytes_per_param_quantized) as u64;
        let mut total_bytes = num_layers * layer_bytes;

        // Add embedding table if loaded (~vocab_size * hidden * 2 bytes for f16)
        if self.tok_embeddings.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        // Add output projection if loaded
        if self.output.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        total_bytes / (1024 * 1024) // Convert to MB
    }
}
