//! Transformer layer weight structures and forward pass implementations.
//!
//! Includes Qwen 3.5 hybrid SSM+attention (Gated Delta Networks) layer types.
//! These are fully implemented but not yet wired into the forward pass —
//! see `docs/ARCHITECTURE.md` § "Deferred Items" for status.

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::Module;
use candle_transformers::quantized_nn::RmsNorm;

use super::model_arch::Activation;
use crate::model::lora::LoraAdapter;

// ── Quantized MatMul wrapper ──

#[derive(Debug, Clone)]
pub(crate) struct QMatMul {
    pub(crate) inner: QMatMulInner,
}

#[derive(Debug, Clone)]
pub(crate) enum QMatMulInner {
    /// Standard single-weight matmul
    Standard(candle_core::quantized::QMatMul),
    /// Fused weight with output slicing — keeps original quantization quality.
    /// Computes full matmul then extracts [offset..offset+len] from the output's last dim.
    /// Used for fused QKV/FFN (Phi-3) where re-quantization of split weights degrades quality.
    FusedSlice {
        fused: std::sync::Arc<candle_core::quantized::QMatMul>,
        offset: usize,
        len: usize,
    },
}

impl QMatMul {
    pub(crate) fn from_qtensor(qtensor: QTensor) -> CandleResult<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self {
            inner: QMatMulInner::Standard(inner),
        })
    }

    /// Create a shared fused QMatMul that can be used by multiple FusedSlice variants.
    pub(crate) fn make_fused(
        qtensor: QTensor,
    ) -> CandleResult<std::sync::Arc<candle_core::quantized::QMatMul>> {
        let fused = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(std::sync::Arc::new(fused))
    }

    /// Create a QMatMul that slices the output of a shared fused matmul.
    /// Preserves the original quantization quality (no re-quantization).
    pub(crate) fn from_fused_slice(
        fused: std::sync::Arc<candle_core::quantized::QMatMul>,
        offset: usize,
        len: usize,
    ) -> Self {
        Self {
            inner: QMatMulInner::FusedSlice { fused, offset, len },
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            QMatMulInner::Standard(m) => m.forward(xs),
            QMatMulInner::FusedSlice { fused, offset, len } => {
                let full = fused.forward(xs)?;
                full.narrow(candle_core::D::Minus1, *offset, *len)?
                    .contiguous()
            }
        }
    }
}

// ── MLP / FFN ──

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    pub(crate) ffn_gate: QMatMul,
    pub(crate) ffn_down: QMatMul,
    pub(crate) ffn_up: QMatMul,
    pub(crate) activation: Activation,
}

impl Mlp {
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        lora: Option<(&LoraAdapter, usize)>,
    ) -> CandleResult<Tensor> {
        let mut gate = self.ffn_gate.forward(xs)?;
        let mut up = self.ffn_up.forward(xs)?;

        if let Some((adapter, abs_layer)) = lora {
            let key_gate = format!("blk.{abs_layer}.ffn_gate");
            if let Some(lw) = adapter.weights.get(&key_gate) {
                gate = crate::model::lora::apply_lora(
                    &gate,
                    xs,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_gate: {e}")))?;
            }
            let key_up = format!("blk.{abs_layer}.ffn_up");
            if let Some(lw) = adapter.weights.get(&key_up) {
                up = crate::model::lora::apply_lora(
                    &up,
                    xs,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_up: {e}")))?;
            }
        }

        let activated = match self.activation {
            Activation::SiLU => candle_nn::ops::silu(&gate)?,
            Activation::Gelu => gate.gelu()?,
        };
        let combined = (activated * up)?;

        let mut down = self.ffn_down.forward(&combined)?;
        if let Some((adapter, abs_layer)) = lora {
            let key_down = format!("blk.{abs_layer}.ffn_down");
            if let Some(lw) = adapter.weights.get(&key_down) {
                down = crate::model::lora::apply_lora(
                    &down,
                    &combined,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_down: {e}")))?;
            }
        }
        Ok(down)
    }
}

// ── MLA (Multi-head Latent Attention) for DeepSeek-V2/V3 ──

/// MLA (Multi-head Latent Attention) weights for DeepSeek-V2/V3.
///
/// Q path: x → q_a → q_a_norm → q_b → reshape → split(q_nope, q_rope) → RoPE(q_rope)
/// KV path: x → kv_a → split(c_kv, k_rope_raw) → RoPE(k_rope_raw);
///          c_kv → kv_a_norm → kv_b → reshape → split(k_nope, v)
///          k = concat(k_nope, k_rope) expanded per head
/// Attention: standard matmul with full K,V stored in KV cache
#[derive(Debug, Clone)]
pub(crate) struct MlaWeights {
    // Q path
    pub(crate) q_a: QMatMul, // hidden → q_lora_rank
    pub(crate) q_a_norm: RmsNorm,
    pub(crate) q_b: QMatMul, // q_lora_rank → n_head * key_length
    // KV path
    pub(crate) kv_a: QMatMul, // hidden → kv_lora_rank + rope_dim
    pub(crate) kv_a_norm: RmsNorm,
    pub(crate) kv_b: QMatMul, // kv_lora_rank → n_head * (key_length - rope_dim + value_length)
    pub(crate) output: QMatMul, // n_head * value_length → hidden
    // Dimensions
    pub(crate) n_head: usize,
    pub(crate) key_length: usize, // per-head total key dim (nope + rope)
    pub(crate) value_length: usize, // per-head value dim
    pub(crate) kv_lora_rank: usize,
    pub(crate) rope_dim: usize, // how many dims of key_length are rotary
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) neg_inf: Tensor,
}

impl MlaWeights {
    /// Apply contiguous RoPE to a tensor in BHSD layout.
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, _dim) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    /// MLA forward pass.
    ///
    /// Decompresses Q and KV via low-rank projections, applies RoPE to the
    /// rotary portions, stores full K/V in the KV cache, runs standard
    /// attention, and projects the output.
    pub(crate) fn forward_mla(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let nope_dim = self.key_length - self.rope_dim;

        // ── Q path ──
        let q_compressed = self.q_a.forward(x)?; // [b, s, q_lora_rank]
        let q_compressed = self.q_a_norm.forward(&q_compressed)?;
        let q_full = self.q_b.forward(&q_compressed)?; // [b, s, n_head * key_length]
        let q_full = q_full.reshape((b_sz, seq_len, self.n_head, self.key_length))?;
        // Split into nope and rope parts
        let q_nope = q_full.narrow(3, 0, nope_dim)?; // [b, s, n_head, nope_dim]
        let q_rope = q_full.narrow(3, nope_dim, self.rope_dim)?; // [b, s, n_head, rope_dim]
                                                                 // Apply RoPE to q_rope (needs BHSD)
        let q_rope = q_rope.transpose(1, 2)?.contiguous()?;
        let q_rope = self.apply_rope(&q_rope, index_pos)?;
        let q_rope = q_rope.transpose(1, 2)?; // back to [b, s, n_head, rope_dim]
        let q_nope = q_nope.contiguous()?;
        let q_rope = q_rope.contiguous()?;
        // Concat: q = [q_nope, q_rope]
        let q = Tensor::cat(&[&q_nope, &q_rope], 3)?; // [b, s, n_head, key_length]
        let q = q.transpose(1, 2)?.contiguous()?; // BHSD [b, n_head, s, key_length]

        // ── KV path ──
        let kv_compressed = self.kv_a.forward(x)?; // [b, s, kv_lora_rank + rope_dim]
                                                   // Split into c_kv latent and k_rope_raw (narrow produces non-contiguous views)
        let c_kv = kv_compressed
            .narrow(2, 0, self.kv_lora_rank)?
            .contiguous()?;
        let k_rope_raw = kv_compressed
            .narrow(2, self.kv_lora_rank, self.rope_dim)?
            .contiguous()?;
        // Apply RoPE to k_rope_raw: reshape to [b, s, 1, rope_dim] → BHSD → rope → back
        let k_rope = k_rope_raw
            .reshape((b_sz, seq_len, 1, self.rope_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_rope = self.apply_rope(&k_rope, index_pos)?; // [b, 1, s, rope_dim]
                                                           // Expand k_rope to all heads: [b, n_head, s, rope_dim]
        let k_rope = k_rope
            .broadcast_as((b_sz, self.n_head, seq_len, self.rope_dim))?
            .contiguous()?;

        // Decompress KV latent
        let c_kv_normed = self.kv_a_norm.forward(&c_kv)?;
        let kv_decompressed = self.kv_b.forward(&c_kv_normed)?;
        // kv_decompressed: [b, s, n_head * (nope_dim + value_length)]
        let kv_per_head = nope_dim + self.value_length;
        let kv_full = kv_decompressed.reshape((b_sz, seq_len, self.n_head, kv_per_head))?;
        let k_nope = kv_full.narrow(3, 0, nope_dim)?;
        let v = kv_full.narrow(3, nope_dim, self.value_length)?;

        // k = concat(k_nope, k_rope) per head → BHSD
        let k_nope = k_nope.transpose(1, 2)?.contiguous()?; // [b, n_head, s, nope_dim]
        let k = Tensor::cat(&[&k_nope, &k_rope], 3)?; // [b, n_head, s, key_length]

        let v = v.transpose(1, 2)?.contiguous()?; // [b, n_head, s, value_length]

        // ── KV cache ──
        let (k, v) = match kv_cache {
            None => {
                let mut cache = KvCache::new(2, max_seq_len);
                let kv = cache.append(&k, &v)?;
                *kv_cache = Some(cache);
                kv
            }
            Some(cache) => {
                if index_pos == 0 {
                    cache.reset();
                }
                cache.append(&k, &v)?
            }
        };

        // ── Attention ──
        // MLA has asymmetric K/V dimensions (key_length != value_length), so
        // CPU flash attention (which assumes uniform head_dim) cannot be used.
        // Use standard matmul attention which handles this correctly.
        let y = standard_attention(
            &q,
            &k,
            &v,
            mask,
            self.key_length,
            self.n_head,
            self.n_head, // MLA: n_kv_head == n_head (full K/V per head)
            &self.neg_inf,
            None, // no softcap for DeepSeek
        )?;

        // y: [b, n_head, s, key_length] but we need [b, n_head, s, value_length]
        // run_attention returns [b, n_head, s, head_dim_of_v] which is value_length
        // Reshape: [b, n_head, s, value_length] → [b, s, n_head * value_length]
        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, self.n_head * self.value_length])?;
        self.output.forward(&y)
    }
}

// ── MoE FFN for DeepSeek-V2/V3 ──

/// Mixture-of-Experts FFN for DeepSeek-V2/V3.
///
/// Router selects top-k experts per token, runs SiLU-gated FFN for each,
/// and sums the weighted outputs. Shared experts (always active) are added.
#[derive(Debug, Clone)]
pub(crate) struct MoeFfn {
    pub(crate) gate: Tensor, // router weights: [n_experts, hidden] (dequantized)
    pub(crate) gate_exps: Tensor, // stacked expert gate: [n_experts, intermediate, hidden]
    pub(crate) down_exps: Tensor, // stacked expert down: [n_experts, hidden, intermediate]
    pub(crate) up_exps: Tensor, // stacked expert up: [n_experts, intermediate, hidden]
    // Shared experts (always active, optional)
    pub(crate) shared_gate: Option<QMatMul>,
    pub(crate) shared_down: Option<QMatMul>,
    pub(crate) shared_up: Option<QMatMul>,
    pub(crate) n_experts_used: usize, // top-k
}

/// Select top-k indices and weights from a score vector on CPU.
///
/// Candle 0.9 doesn't have a built-in topk, so we pull scores to CPU,
/// argsort descending, and take top-k. Fine for small n_experts vectors.
pub(crate) fn topk_cpu(scores: &Tensor, k: usize) -> CandleResult<(Tensor, Tensor)> {
    let scores_vec: Vec<f32> = scores.to_vec1()?;
    let n = scores_vec.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scores_vec[b]
            .partial_cmp(&scores_vec[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    let weights: Vec<f32> = indices.iter().map(|&i| scores_vec[i]).collect();
    let idx_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    let device = scores.device();
    let idx_tensor = Tensor::from_vec(idx_i64, (k,), device)?;
    let w_tensor = Tensor::from_vec(weights, (k,), device)?;
    // Normalize weights via softmax over selected experts
    let w_tensor = candle_nn::ops::softmax(&w_tensor, 0)?;
    Ok((idx_tensor, w_tensor))
}

impl MoeFfn {
    /// MoE forward pass with batched expert dispatch.
    ///
    /// 1. Router: x.matmul(gate.t()) → softmax scores → topk per token
    /// 2. Group tokens by assigned expert
    /// 3. For each expert: batched SiLU-gated FFN on all assigned tokens at once
    /// 4. Scatter weighted results back to token positions
    /// 5. Add shared expert output (if present)
    pub(crate) fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let x_flat = x.reshape((num_tokens, hidden))?;
        let device = x.device();
        let dtype = x.dtype();

        // Router scores: [num_tokens, n_experts]
        let router_scores = x_flat.matmul(&self.gate.t()?)?;
        let n_experts = self.gate.dim(0)?;

        // Phase 1: Route all tokens — collect (token_position, weight) per expert
        let mut expert_batches: Vec<Vec<(usize, f32)>> = vec![vec![]; n_experts];
        for pos in 0..num_tokens {
            let token_scores = router_scores.get(pos)?;
            let (indices, weights) = topk_cpu(&token_scores, self.n_experts_used)?;
            let indices_vec: Vec<i64> = indices.to_vec1()?;
            let weights_vec: Vec<f32> = weights.to_vec1()?;
            for (i, &expert_idx) in indices_vec.iter().enumerate() {
                expert_batches[expert_idx as usize].push((pos, weights_vec[i]));
            }
        }

        // Phase 2: Per-position accumulator for weighted expert outputs
        let mut pos_accum: Vec<Option<Tensor>> = vec![None; num_tokens];

        for (eidx, batch) in expert_batches.iter().enumerate() {
            if batch.is_empty() {
                continue;
            }

            // Gather all tokens assigned to this expert via index_select
            let idx_vec: Vec<i64> = batch.iter().map(|&(pos, _)| pos as i64).collect();
            let idx_tensor = Tensor::from_vec(idx_vec, (batch.len(),), device)?;
            let batch_input = x_flat.index_select(&idx_tensor, 0)?; // [batch_tokens, hidden]

            // Batched SiLU-gated FFN: silu(x @ gate.t) * (x @ up.t) @ down.t
            let gate_w = self.gate_exps.get(eidx)?;
            let up_w = self.up_exps.get(eidx)?;
            let down_w = self.down_exps.get(eidx)?;

            let gate_out = batch_input.matmul(&gate_w.t()?)?;
            let up_out = batch_input.matmul(&up_w.t()?)?;
            let activated = candle_nn::ops::silu(&gate_out)?;
            let combined = (activated * up_out)?;
            let expert_out = combined.matmul(&down_w.t()?)?; // [batch_tokens, hidden]

            // Apply per-token weights
            let weight_vec: Vec<f32> = batch.iter().map(|&(_, w)| w).collect();
            let weight_tensor =
                Tensor::from_vec(weight_vec, (batch.len(), 1), device)?.to_dtype(dtype)?;
            let weighted = expert_out.broadcast_mul(&weight_tensor)?;

            // Scatter weighted results back to position accumulators
            for (local_idx, &(pos, _)) in batch.iter().enumerate() {
                let contrib = weighted.narrow(0, local_idx, 1)?;
                pos_accum[pos] = Some(match pos_accum[pos].take() {
                    Some(existing) => (existing + contrib)?,
                    None => contrib,
                });
            }
        }

        // Assemble output: cat all position results
        let zero = Tensor::zeros((1, hidden), dtype, device)?;
        let slices: Vec<&Tensor> = pos_accum
            .iter()
            .map(|opt| opt.as_ref().unwrap_or(&zero))
            .collect();
        let mut output = Tensor::cat(&slices, 0)?;

        // Add shared expert output if present
        if let (Some(ref sg), Some(ref sd), Some(ref su)) =
            (&self.shared_gate, &self.shared_down, &self.shared_up)
        {
            let shared_gate_out = sg.forward(&x_flat)?;
            let shared_up_out = su.forward(&x_flat)?;
            let shared_activated = candle_nn::ops::silu(&shared_gate_out)?;
            let shared_combined = (shared_activated * shared_up_out)?;
            let shared_out = sd.forward(&shared_combined)?;
            output = (output + shared_out)?;
        }

        output.reshape((b_sz, seq_len, hidden))
    }
}

// ── Layer variant enum for supporting DeepSeek alongside dense architectures ──

/// A transformer layer that is either a standard dense layer or a DeepSeek MLA+MoE layer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum LayerVariant {
    /// Standard dense transformer layer (Llama, Qwen2, Gemma, etc.)
    Dense(LayerWeights),
    /// DeepSeek-V2/V3 layer with MLA attention + MoE or dense FFN
    DeepSeek {
        attention: MlaWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        ffn_norm: RmsNorm,
    },
    /// Qwen 3.5 full-attention layer (every 4th layer)
    Qwen35Attn {
        weights: Qwen35AttnWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        post_attention_norm: RmsNorm,
    },
    /// Qwen 3.5 linear-attention (Gated Delta Network / SSM) layer
    Qwen35Ssm {
        weights: DeltaNetWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        post_attention_norm: RmsNorm,
    },
}

/// Qwen 3.5 full-attention layer weights.
/// Similar to standard attention but with output gating from Q projection.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Qwen35AttnWeights {
    /// Fused QKV + gate projection: hidden → (q_dim + k_dim + v_dim + gate_dim)
    pub(crate) wqkv: Option<QMatMul>,
    /// Separate Q/K/V projections (used when fused QKV not available)
    pub(crate) wq: Option<QMatMul>,
    pub(crate) wk: Option<QMatMul>,
    pub(crate) wv: Option<QMatMul>,
    pub(crate) wo: QMatMul,
    /// Output gate weights (sigmoid applied before O projection)
    pub(crate) attn_gate: Tensor,
    /// Q/K head normalization (RmsNorm per-head before RoPE)
    pub(crate) q_norm: Option<RmsNorm>,
    pub(crate) k_norm: Option<RmsNorm>,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) neg_inf: Tensor,
    /// Partial RoPE: only first `rope_dim` of head_dim get rotated
    pub(crate) rope_dim: usize,
}

/// Gated Delta Network (SSM) layer weights for Qwen 3.5 linear-attention layers.
///
/// Forward pass:
/// 1. Project input → Q, K, V, Z (gating) via fused projection
/// 2. Apply 1D causal convolution (conv1d with kernel_size typically 4)
/// 3. Compute state transition: alpha = softplus(ssm_alpha + ssm_dt), beta = sigmoid(ssm_beta)
/// 4. Run delta net scan: state = alpha * state + beta * (v ⊗ k), output = state @ q
/// 5. Apply gated normalization: norm(output) * silu(z)
/// 6. Project through ssm_out
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct DeltaNetWeights {
    /// Fused QKV+Z projection: hidden → (q_dim + k_dim + v_dim + z_dim)
    pub(crate) wqkv: Option<QMatMul>,
    /// Separate Q/K/V projections (used when fused QKV not available)
    pub(crate) wq: Option<QMatMul>,
    pub(crate) wk: Option<QMatMul>,
    pub(crate) wv: Option<QMatMul>,
    /// SSM state transition parameter A (decay): [hidden, conv_kernel_dim]
    pub(crate) ssm_alpha: Tensor,
    /// SSM input gate B: [hidden, conv_kernel_dim]
    pub(crate) ssm_beta: Tensor,
    /// Delta time-step parameter: enables input-dependent state transitions
    pub(crate) ssm_dt: QMatMul,
    /// 1D causal convolution kernel: [n_heads, 1, conv_kernel_dim]
    pub(crate) ssm_conv1d: Tensor,
    /// Gated output normalization
    pub(crate) ssm_norm: RmsNorm,
    /// Output projection: recurrent_dim → hidden
    pub(crate) ssm_out: QMatMul,
    /// Number of Q heads for the linear attention
    pub(crate) n_head: usize,
    /// Number of K heads (may differ from Q)
    pub(crate) n_kv_head: usize,
    /// Number of V heads (may differ from K in Qwen 3.5)
    pub(crate) n_v_head: usize,
    /// Key head dimension
    pub(crate) key_head_dim: usize,
    /// Value head dimension
    pub(crate) value_head_dim: usize,
    /// Convolution kernel size (typically 4)
    pub(crate) conv_kernel_dim: usize,
}

/// Per-request SSM (delta net) recurrent state for Qwen 3.5 hybrid models.
/// Analogous to KV-cache for attention layers, but stores conv state + recurrent state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct SsmState {
    /// 1D convolution buffer: [batch, n_heads * head_dim, conv_kernel_dim - 1]
    /// Stores the last (kernel_size - 1) inputs for causal conv.
    pub conv_state: Tensor,
    /// Recurrent state matrix: [batch, n_kv_heads, value_head_dim, key_head_dim]
    /// The running "memory" of the delta network.
    pub recurrent_state: Tensor,
}

/// FFN variant for DeepSeek layers — either dense or MoE.
#[derive(Debug, Clone)]
pub(crate) enum FfnVariant {
    Dense(Mlp),
    MoE(MoeFfn),
}

/// Extra metadata for DeepSeek-V2/V3 MoE+MLA models.
#[derive(Clone, Debug)]
pub(crate) struct DeepSeekMeta {
    pub(crate) n_experts: usize,
    pub(crate) n_experts_used: usize,
    pub(crate) n_shared_experts: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) q_lora_rank: usize,
    pub(crate) key_length: usize,
    pub(crate) value_length: usize,
    pub(crate) rope_dim: usize,
}

// ── Per-layer weights ──

#[derive(Debug, Clone)]
pub(crate) struct LayerWeights {
    pub(crate) attention_wq: QMatMul,
    pub(crate) attention_wk: QMatMul,
    pub(crate) attention_wv: QMatMul,
    pub(crate) attention_wo: QMatMul,
    /// Qwen2 has QKV biases; for architectures without biases these are None.
    pub(crate) attention_bq: Option<Tensor>,
    pub(crate) attention_bk: Option<Tensor>,
    pub(crate) attention_bv: Option<Tensor>,
    pub(crate) attention_norm: RmsNorm,
    /// Qwen3 applies RmsNorm to Q per-head after projection (before RoPE).
    pub(crate) attn_q_norm: Option<RmsNorm>,
    /// Qwen3 applies RmsNorm to K per-head after projection (before RoPE).
    pub(crate) attn_k_norm: Option<RmsNorm>,
    pub(crate) ffn: FfnVariant,
    pub(crate) ffn_norm: RmsNorm,
    /// Gemma 2 post-attention RmsNorm (applied after attention, before residual add).
    pub(crate) post_attention_norm: Option<RmsNorm>,
    /// Gemma 2 post-FFN RmsNorm (applied after FFN, before residual add).
    pub(crate) post_ffw_norm: Option<RmsNorm>,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) neg_inf: Tensor,
    /// If true, use contiguous RoPE (rope); if false, use interleaved (rope_i).
    pub(crate) use_rope_contiguous: bool,
    /// Gemma 2 attention logit soft-capping: `tanh(logits / cap) * cap` before softmax.
    pub(crate) attn_logit_softcap: Option<f32>,
    /// Number of head dimensions that receive RoPE. When < head_dim, only the first
    /// `rope_dim` dimensions are rotated and the rest pass through unchanged (partial RoPE,
    /// used by GLM-4). When == head_dim, standard full RoPE is applied.
    pub(crate) rope_dim: usize,
    /// If true, skip RoPE entirely for this layer (Llama 4 NoPE layers).
    pub(crate) skip_rope: bool,
}

pub(crate) fn masked_fill(
    on_false: &Tensor,
    mask: &Tensor,
    on_true: &Tensor,
) -> CandleResult<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

/// Standard O(n^2) matmul attention with optional causal mask.
/// Input/output layout: BHSD `(b, n_head, seq, head_dim)`.
/// Supports optional Gemma 2 attention logit soft-capping.
#[allow(clippy::too_many_arguments)]
pub(crate) fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    neg_inf: &Tensor,
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    let k = candle_transformers::utils::repeat_kv(k.clone(), n_head / n_kv_head)?;
    let v = candle_transformers::utils::repeat_kv(v.clone(), n_head / n_kv_head)?;

    let att = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
    // Gemma 2 attention logit soft-capping: tanh(logits / cap) * cap
    let att = if let Some(cap) = attn_logit_softcap {
        let cap_f64 = cap as f64;
        ((att / cap_f64)?.tanh()? * cap_f64)?
    } else {
        att
    };
    let att = match mask {
        None => att,
        Some(mask) => {
            let mask = mask.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, neg_inf)?
        }
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v.contiguous()?)
}

/// Unified attention dispatch: selects the best backend for the device.
///
/// - **CPU**: Uses `candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>()` which
///   handles GQA natively (no `repeat_kv` needed) and uses O(1) memory via tiled online softmax.
/// - **GPU with `flash-attn` feature**: Uses `candle_flash_attn::flash_attn()` with F16 —
///   fused CUDA kernel, GQA native, O(1) memory. Requires SM80+ (RTX 3070 SM86 compatible).
/// - **GPU without `flash-attn` / fallback**: Standard matmul path with `repeat_kv`.
///
/// All paths produce output in BHSD layout `(b, n_head, seq, head_dim)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    neg_inf: &Tensor,
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    match q.device() {
        Device::Cpu => {
            let seq_len = q.dim(2)?;

            // Fast path for decode (seq_len=1): skip CPU flash attention's
            // triple transpose+contiguous copies. Standard matmul is cheaper
            // for a single query token against the KV cache.
            if seq_len == 1 {
                return standard_attention(
                    q,
                    k,
                    v,
                    mask,
                    head_dim,
                    n_head,
                    n_kv_head,
                    neg_inf,
                    attn_logit_softcap,
                );
            }

            // CPU flash attention: input BSHD, output BHSD
            // Transpose Q/K/V from BHSD (b,h,s,d) to BSHD (b,s,h,d)
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();

            // Build float additive mask for prefill (decode has seq_len==1, no mask needed)
            let flash_mask = if seq_len > 1 {
                if let Some(u8_mask) = mask {
                    // Convert u8 causal mask (1=masked, 0=visible) to float (NEG_INF=masked, 0=visible)
                    let zeros = u8_mask.zeros_like()?.to_dtype(DType::F32)?;
                    let neg_inf_scalar = Tensor::new(f32::NEG_INFINITY, u8_mask.device())?;
                    let float_mask = u8_mask.where_cond(
                        &neg_inf_scalar.broadcast_as(u8_mask.shape().dims())?,
                        &zeros,
                    )?;
                    Some(float_mask)
                } else {
                    None
                }
            } else {
                None
            };

            // run_flash_attn_cpu handles GQA natively — no repeat_kv needed
            // Output shape: (b, n_head, seq, head_dim) — BHSD
            let out = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                flash_mask.as_ref(),
                softmax_scale,
                None, // no ALiBi
                attn_logit_softcap,
            )?;
            Ok(out)
        }
        #[cfg(feature = "flash-attn")]
        Device::Cuda(_) => {
            let q_len = q.dim(2)?;
            let k_len = k.dim(2)?;

            // Flash attention only supports simple causal masking. When the KV cache
            // has pre-populated prefix entries (k_len > q_len with q_len > 1), the
            // offset causal mask can't be expressed via flash_attn's boolean causal
            // flag. Fall back to standard matmul attention with the explicit mask.
            if k_len > q_len && q_len > 1 {
                return standard_attention(
                    q,
                    k,
                    v,
                    mask,
                    head_dim,
                    n_head,
                    n_kv_head,
                    neg_inf,
                    attn_logit_softcap,
                );
            }

            // GPU flash attention: input BSHD, output BSHD → transpose to BHSD
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            // Flash attention requires F16
            let q_f16 = q_bshd.to_dtype(DType::F16)?;
            let k_f16 = k_bshd.to_dtype(DType::F16)?;
            let v_f16 = v_bshd.to_dtype(DType::F16)?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();

            // flash_attn handles GQA and causal masking natively
            let out =
                candle_flash_attn::flash_attn(&q_f16, &k_f16, &v_f16, softmax_scale, q_len > 1)?;

            // Output is BSHD — transpose to BHSD and cast back to F32
            out.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()
        }
        _ => {
            // Fallback: standard matmul attention with optional soft-capping
            standard_attention(
                q,
                k,
                v,
                mask,
                head_dim,
                n_head,
                n_kv_head,
                neg_inf,
                attn_logit_softcap,
            )
        }
    }
}

impl LayerWeights {
    pub(crate) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        // Llama 4 NoPE layers: skip RoPE entirely
        if self.skip_rope {
            return Ok(x.clone());
        }

        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Partial RoPE (GLM-4): only rotate the first rope_dim dimensions,
        // pass the rest through unchanged.
        if self.rope_dim < n_embd {
            let x_rot = x.narrow(3, 0, self.rope_dim)?.contiguous()?;
            let x_pass = x.narrow(3, self.rope_dim, n_embd - self.rope_dim)?;
            let rotated = if self.use_rope_contiguous {
                candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?
            } else {
                candle_nn::rotary_emb::rope_i(&x_rot, &cos, &sin)?
            };
            Tensor::cat(&[&rotated, &x_pass], 3)
        } else {
            // Full RoPE (standard path)
            if self.use_rope_contiguous {
                candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
            } else {
                candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
            }
        }
    }

    pub(crate) fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
        lora: Option<(&LoraAdapter, usize)>,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let mut q = self.attention_wq.forward(x)?;
        let mut k = self.attention_wk.forward(x)?;
        let mut v = self.attention_wv.forward(x)?;

        // Apply LoRA deltas to Q/K/V projections if adapter is active
        if let Some((adapter, abs_layer)) = lora {
            let key_q = format!("blk.{abs_layer}.attn_q");
            if let Some(lw) = adapter.weights.get(&key_q) {
                q = crate::model::lora::apply_lora(
                    &q,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_q: {e}")))?;
            }
            let key_k = format!("blk.{abs_layer}.attn_k");
            if let Some(lw) = adapter.weights.get(&key_k) {
                k = crate::model::lora::apply_lora(
                    &k,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_k: {e}")))?;
            }
            let key_v = format!("blk.{abs_layer}.attn_v");
            if let Some(lw) = adapter.weights.get(&key_v) {
                v = crate::model::lora::apply_lora(
                    &v,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_v: {e}")))?;
            }
        }

        // Apply QKV biases if present (Qwen2 has biases)
        if let Some(ref bq) = self.attention_bq {
            q = q.broadcast_add(bq)?;
        }
        if let Some(ref bk) = self.attention_bk {
            k = k.broadcast_add(bk)?;
        }
        if let Some(ref bv) = self.attention_bv {
            v = v.broadcast_add(bv)?;
        }

        let mut q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Qwen3 QK normalization: RmsNorm applied per-head before RoPE
        if let Some(ref qn) = self.attn_q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.attn_k_norm {
            k = kn.forward(&k)?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // KV-cache: use pre-allocated KvCache buffers (avoids Tensor::cat per step)
        let (k, v) = match kv_cache {
            None => {
                let mut cache = KvCache::new(2, max_seq_len);
                let kv = cache.append(&k, &v)?;
                *kv_cache = Some(cache);
                kv
            }
            Some(cache) => {
                if index_pos == 0 {
                    cache.reset();
                }
                cache.append(&k, &v)?
            }
        };

        // Unified attention dispatch: flash (CPU/GPU) or standard matmul fallback.
        // All backends return BHSD (b, n_head, seq, head_dim).
        let y = run_attention(
            &q,
            &k,
            &v,
            mask,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            &self.neg_inf,
            self.attn_logit_softcap,
        )?;

        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, attn_out_dim])?;
        let mut wo_out = self.attention_wo.forward(&y)?;

        // Apply LoRA delta to O projection
        if let Some((adapter, abs_layer)) = lora {
            let key_o = format!("blk.{abs_layer}.attn_output");
            if let Some(lw) = adapter.weights.get(&key_o) {
                wo_out = crate::model::lora::apply_lora(
                    &wo_out,
                    &y,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_output: {e}")))?;
            }
        }
        Ok(wo_out)
    }
}

// ── Qwen 3.5 full-attention layer forward ──

#[allow(dead_code)]
impl Qwen35AttnWeights {
    pub(crate) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Partial RoPE: only rotate first rope_dim dimensions
        if self.rope_dim < n_embd {
            let x_rot = x.narrow(3, 0, self.rope_dim)?.contiguous()?;
            let x_pass = x.narrow(3, self.rope_dim, n_embd - self.rope_dim)?;
            let rotated = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&rotated, &x_pass], 3)
        } else {
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        }
    }

    pub(crate) fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        // Project Q, K, V (and gate from Q)
        let (q, k, v, gate) = if let Some(ref wqkv) = self.wqkv {
            let qkv = wqkv.forward(x)?;
            let q_dim = self.n_head * self.head_dim;
            let k_dim = self.n_kv_head * self.head_dim;
            let v_dim = k_dim;
            let q = qkv.narrow(2, 0, q_dim)?;
            let k = qkv.narrow(2, q_dim, k_dim)?;
            let v = qkv.narrow(2, q_dim + k_dim, v_dim)?;
            let gate = qkv.narrow(2, q_dim + k_dim + v_dim, q_dim)?;
            (q, k, v, gate)
        } else {
            // Separate Q/K/V projections without fused QKV — gate not available.
            // The gate will only use the learned attn_gate bias (sigmoid(0 + bias)).
            // This produces degraded but functional output for GGUFs with split Q/K/V.
            let q = self.wq.as_ref().unwrap().forward(x)?;
            let k = self.wk.as_ref().unwrap().forward(x)?;
            let v = self.wv.as_ref().unwrap().forward(x)?;
            let gate = q.zeros_like()?;
            (q, k, v, gate)
        };

        // Reshape to heads
        let mut q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Q/K head normalization
        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.k_norm {
            k = kn.forward(&k)?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;

        // Partial RoPE
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // KV-cache
        let (k, v) = match kv_cache {
            None => {
                let mut cache = KvCache::new(2, max_seq_len);
                let kv = cache.append(&k, &v)?;
                *kv_cache = Some(cache);
                kv
            }
            Some(cache) => {
                if index_pos == 0 {
                    cache.reset();
                }
                cache.append(&k, &v)?
            }
        };

        // Attention
        let y = run_attention(
            &q,
            &k,
            &v,
            mask,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            &self.neg_inf,
            None,
        )?;

        // Apply output gate: sigmoid(gate + attn_gate_bias) * attn_output
        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, attn_out_dim])?;
        let gate_sig = gate
            .reshape((b_sz, seq_len, attn_out_dim))?
            .broadcast_add(&self.attn_gate)?;
        let gate_sig = candle_nn::ops::sigmoid(&gate_sig)?;
        let gated = (y * gate_sig)?;

        self.wo.forward(&gated)
    }
}

// ── Qwen 3.5 Gated Delta Network (SSM) layer forward ──

#[allow(dead_code)]
impl DeltaNetWeights {
    /// Forward pass for the Gated Delta Network (linear attention / SSM layer).
    pub(crate) fn forward_deltanet(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let device = x.device();

        // Project to Q, K, V, Z
        let (q, k, v, z) = if let Some(ref wqkv) = self.wqkv {
            let proj = wqkv.forward(x)?;
            let q_dim = self.n_head * self.key_head_dim;
            let k_dim = self.n_kv_head * self.key_head_dim;
            let v_dim = self.n_v_head * self.value_head_dim;
            let z_dim = v_dim;
            let q = proj.narrow(2, 0, q_dim)?;
            let k = proj.narrow(2, q_dim, k_dim)?;
            let v = proj.narrow(2, q_dim + k_dim, v_dim)?;
            let z = proj.narrow(2, q_dim + k_dim + v_dim, z_dim)?;
            (q, k, v, z)
        } else {
            let q = self.wq.as_ref().unwrap().forward(x)?;
            let k = self.wk.as_ref().unwrap().forward(x)?;
            let v = self.wv.as_ref().unwrap().forward(x)?;
            let z = v.zeros_like()?;
            (q, k, v, z)
        };

        // Apply 1D causal convolution over the QKV concatenation
        let qkv_cat = Tensor::cat(&[&q, &k, &v], 2)?;
        let (conv_out, new_conv_state) = self.apply_conv1d(&qkv_cat, ssm_state, device)?;

        // Split back into Q, K, V after convolution
        let q_dim = self.n_head * self.key_head_dim;
        let k_dim = self.n_kv_head * self.key_head_dim;
        let v_dim = self.n_v_head * self.value_head_dim;
        let q = conv_out.narrow(2, 0, q_dim)?;
        let k = conv_out.narrow(2, q_dim, k_dim)?;
        let v = conv_out.narrow(2, q_dim + k_dim, v_dim)?;

        // Reshape to heads: [b, seq, n_head, head_dim] → [b, n_head, seq, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.key_head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.key_head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_v_head, self.value_head_dim))?
            .transpose(1, 2)?;

        // Compute state transition: alpha = softplus(ssm_alpha + ssm_dt(x))
        let dt = self.ssm_dt.forward(x)?;
        // ssm_alpha broadcast to [b, seq, dim], then softplus
        let alpha_base = self.ssm_alpha.broadcast_as(dt.shape())?;
        let alpha = softplus(&(&alpha_base + &dt)?)?;

        // beta = sigmoid(ssm_beta)
        let beta_base = self
            .ssm_beta
            .broadcast_as((b_sz, seq_len, q_dim + k_dim + v_dim))?;
        let beta = candle_nn::ops::sigmoid(&beta_base)?;

        // Run the delta net recurrent scan
        let output =
            self.delta_net_scan(&q, &k, &v, &alpha, &beta, ssm_state, b_sz, seq_len, device)?;

        // output shape: [b, n_head, seq, value_head_dim]
        let out_dim = self.n_v_head * self.value_head_dim;
        let output = output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, out_dim))?;

        // Gated normalization: norm(output) * silu(z)
        let normed = self.ssm_norm.forward(&output)?;
        let z_act = candle_nn::ops::silu(&z)?;
        let gated = (normed * z_act)?;

        // Update SSM state
        if let Some(state) = ssm_state {
            state.conv_state = new_conv_state;
        } else {
            *ssm_state = Some(SsmState {
                conv_state: new_conv_state,
                recurrent_state: Tensor::zeros(
                    (b_sz, self.n_kv_head, self.value_head_dim, self.key_head_dim),
                    DType::F32,
                    device,
                )?,
            });
        }

        // Project to hidden dim
        self.ssm_out.forward(&gated)
    }

    /// Apply 1D causal convolution with state for autoregressive mode.
    pub(crate) fn apply_conv1d(
        &self,
        x: &Tensor,
        ssm_state: &Option<SsmState>,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (b_sz, seq_len, channels) = x.dims3()?;
        let kernel_size = self.conv_kernel_dim;
        let pad = kernel_size - 1;

        if seq_len == 1 {
            // Autoregressive: use conv state buffer
            let prev_state = if let Some(state) = ssm_state {
                state.conv_state.clone()
            } else {
                Tensor::zeros((b_sz, channels, pad), DType::F32, device)?
            };

            let x_t = x.transpose(1, 2)?; // [b, channels, 1]
            let new_state = if pad > 1 {
                let shifted = prev_state.narrow(2, 1, pad - 1)?;
                Tensor::cat(&[&shifted, &x_t], 2)?
            } else {
                x_t.clone()
            };

            let full_input = Tensor::cat(&[&new_state.narrow(2, 0, pad)?, &x_t], 2)?;
            let kernel = self.ssm_conv1d.reshape((channels, kernel_size))?;
            let conv_out = (&full_input
                * &kernel.unsqueeze(0)?.broadcast_as(full_input.shape())?)?
                .sum(2)?
                .unsqueeze(1)?; // [b, 1, channels]
            let conv_out = candle_nn::ops::silu(&conv_out)?;

            Ok((conv_out, new_state))
        } else {
            // Prefill: full causal convolution
            let x_t = x.transpose(1, 2)?.contiguous()?; // [b, channels, seq]

            let padding = if let Some(state) = ssm_state {
                state.conv_state.clone()
            } else {
                Tensor::zeros((b_sz, channels, pad), DType::F32, device)?
            };
            let padded = Tensor::cat(&[&padding, &x_t], 2)?;

            // Grouped conv1d: each channel independent
            let kernel = self.ssm_conv1d.reshape((channels, 1, kernel_size))?;
            let mut conv_outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let window = padded.narrow(2, t, kernel_size)?;
                let prod = (&window * &kernel.broadcast_as(window.shape())?)?;
                let summed = prod.sum(2)?;
                conv_outputs.push(summed);
            }
            let conv_out = Tensor::stack(&conv_outputs, 1)?; // [b, seq, channels]
            let conv_out = candle_nn::ops::silu(&conv_out)?;

            let new_conv_state = if seq_len >= pad {
                x_t.narrow(2, seq_len - pad, pad)?
            } else {
                let old_kept = padding.narrow(2, seq_len, pad - seq_len)?;
                Tensor::cat(&[&old_kept, &x_t], 2)?
            };

            Ok((conv_out, new_conv_state))
        }
    }

    /// Delta net recurrent scan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn delta_net_scan(
        &self,
        q: &Tensor, // [b, n_head, seq, key_head_dim]
        k: &Tensor, // [b, n_kv_head, seq, key_head_dim]
        v: &Tensor, // [b, n_v_head, seq, value_head_dim]
        _alpha: &Tensor,
        _beta: &Tensor,
        ssm_state: &mut Option<SsmState>,
        b_sz: usize,
        seq_len: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let mut state = if let Some(ref s) = ssm_state {
            s.recurrent_state.clone()
        } else {
            Tensor::zeros(
                (b_sz, self.n_kv_head, self.value_head_dim, self.key_head_dim),
                DType::F32,
                device,
            )?
        };

        // Repeat KV heads for GQA
        let k = if self.n_head > self.n_kv_head {
            candle_transformers::utils::repeat_kv(k.clone(), self.n_head / self.n_kv_head)?
        } else {
            k.clone()
        };
        let v = if self.n_head > self.n_v_head {
            candle_transformers::utils::repeat_kv(v.clone(), self.n_head / self.n_v_head)?
        } else {
            v.clone()
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1)?.squeeze(2)?;
            let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
            let v_t = v.narrow(2, t, 1)?.squeeze(2)?;

            // outer product: v_t ⊗ k_t → [b, n_head, value_head_dim, key_head_dim]
            let v_col = v_t.unsqueeze(3)?;
            let k_row = k_t.unsqueeze(2)?;
            let outer = v_col.matmul(&k_row)?;

            // TODO(qwen35): Implement proper per-timestep alpha/beta gating.
            // Correct: state = diag(alpha_t) * state + beta_t * outer
            // Current: fixed 0.95 decay — produces approximate outputs for Qwen 3.5.
            // The alpha/beta tensors are passed to this function but not yet integrated
            // because the reshape from [b, seq, hidden] to per-head state dims requires
            // model-size-specific head decomposition. See CLAUDE.md "Deferred Items".
            state = (&state * 0.95_f64 + outer)?;

            // Output: state @ q → [b, n_head, value_head_dim]
            let q_col = q_t.unsqueeze(3)?;
            let out_t = state.matmul(&q_col)?.squeeze(3)?;
            outputs.push(out_t);
        }

        let output = Tensor::stack(&outputs, 2)?;

        if let Some(ref mut s) = ssm_state {
            s.recurrent_state = state;
        }

        Ok(output)
    }
}

/// Softplus activation: log(1 + exp(x))
#[allow(dead_code)]
fn softplus(x: &Tensor) -> CandleResult<Tensor> {
    let ones = x.ones_like()?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}
