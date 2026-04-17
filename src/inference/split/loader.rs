//! Weight-loading paths for `SplitModel` — GGUF file loading, shard loading,
//! and the per-architecture tensor decode logic. Split out of `model.rs` so
//! the struct definition and the hot-path execution code stay readable.

// ── Split model: loads only a range of layers from a GGUF ──

use std::io::SeekFrom;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::SwarmError;

use super::model::SplitModel;
use super::rope::{load_longrope_factors, precompute_freqs_cis, precompute_freqs_cis_longrope};
use super::shard_reader::ShardReader;
use super::{
    Activation, DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights,
    Mlp, ModelArch, MoeFfn, QMatMul, Qwen35AttnWeights, DEFAULT_MAX_SEQ_LEN,
};

/// Bundle returned by [`load_qkv_weights`]: either fused `wqkv` is Some, or
/// all three of `wq/wk/wv` are Some.
type QkvWeights = (
    Option<QTensor>,
    Option<QMatMul>,
    Option<QMatMul>,
    Option<QMatMul>,
);

/// Load attn_qkv.weight (fused) or attn_q/k/v.weight (split) for a layer.
/// Used by Qwen35 hybrid architecture (SSM + full-attention layers).
/// Returns (wqkv, wq, wk, wv). Errors if neither fused nor individual weights are present.
fn load_qkv_weights<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    file: &mut R,
    device: &Device,
    prefix: &str,
) -> Result<QkvWeights, SwarmError> {
    fn load_qm<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        file: &mut R,
        device: &Device,
        name: &str,
    ) -> Result<Option<QMatMul>, SwarmError> {
        ct.tensor(file, name, device)
            .ok()
            .map(|t| {
                QMatMul::from_qtensor(t)
                    .map_err(|e| SwarmError::Internal(format!("QMatMul load failed: {e}")))
            })
            .transpose()
    }
    let wqkv = ct
        .tensor(file, &format!("{prefix}.attn_qkv.weight"), device)
        .ok();
    let (wq, wk, wv) = if wqkv.is_none() {
        (
            load_qm(ct, file, device, &format!("{prefix}.attn_q.weight"))?,
            load_qm(ct, file, device, &format!("{prefix}.attn_k.weight"))?,
            load_qm(ct, file, device, &format!("{prefix}.attn_v.weight"))?,
        )
    } else {
        (None, None, None)
    };
    if wqkv.is_none() && (wq.is_none() || wk.is_none() || wv.is_none()) {
        return Err(SwarmError::Internal(format!(
            "{prefix}: missing attn_qkv and individual attn_q/k/v weights"
        )));
    }
    Ok((wqkv, wq, wk, wv))
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
        let arch_str = super::gguf_arch_str(&ct);
        let model_arch = ModelArch::from_gguf_arch(&arch_str);

        tracing::info!(arch = %model_arch, "Detected model architecture");

        if !model_arch.is_supported() {
            return Err(SwarmError::Validation(format!(
                "Unsupported model architecture '{}'. Supported architectures: {}",
                arch_str,
                ModelArch::supported_list().join(", ")
            )));
        }

        // Extract the shared hyperparameters via the unified GgufTensorMeta
        // path so gguf_meta.rs remains the single source of truth for field
        // parsing. Architecture-specific extras (context_length, Gemma
        // softcaps) stay inline since they're not part of the base metadata.
        let meta = super::GgufTensorMeta::from_content(&ct)?;
        let head_count = meta.head_count;
        let head_count_kv = meta.head_count_kv;
        let block_count = meta.block_count;
        let embedding_length = meta.embedding_length;
        let head_dim = meta.head_dim;
        let rope_dim = meta.rope_dim;
        let rms_norm_eps = meta.rms_norm_eps;
        let rope_freq_base = meta.rope_freq_base;
        let arch = &arch_str; // keep a reference for the extras below

        // Arch-specific metadata lookups (context length, Gemma softcaps,
        // DeepSeek MoE/MLA params) that aren't part of the shared base.
        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };

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
                    let (wqkv, wq, wk, wv) = load_qkv_weights(&ct, &mut file, &device, &prefix)?;

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
                    let (wqkv, wq, wk, wv) = load_qkv_weights(&ct, &mut file, &device, &prefix)?;
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
}
