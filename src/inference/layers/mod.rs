//! Transformer layer weight structures and forward pass implementations.
//!
//! Includes Qwen 3.5 hybrid SSM+attention (Gated Delta Networks) layer types.

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

    /// Narrow the weight matrix for tensor parallelism.
    ///
    /// - `dim`: 0 = row-parallel (split input dim), 1 = column-parallel (split output dim)
    /// - `offset`: starting index in the split dimension
    /// - `len`: number of elements in the split dimension
    ///
    /// Dequantizes the quantized weight to f32, narrows, and wraps in a new QMatMul.
    /// The dequantized slice uses more memory per element but the total is smaller
    /// (1/tp_size of the original).
    pub(crate) fn narrow_tp(
        &self,
        dim: usize,
        offset: usize,
        len: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        match &self.inner {
            QMatMulInner::Standard(m) => {
                // Dequantize the weight, narrow, store as f32 Tensor.
                // Weight layout: matmul computes x @ W^T, so W is [out_dim, in_dim].
                // column-parallel (dim=1): split out_dim → narrow dim 0
                // row-parallel (dim=0): split in_dim → narrow dim 1
                let weight_dim = if dim == 1 { 0 } else { 1 };
                let dequant = match m {
                    candle_core::quantized::QMatMul::QTensor(qt) => qt.dequantize(device)?,
                    candle_core::quantized::QMatMul::Tensor(t)
                    | candle_core::quantized::QMatMul::TensorF16(t) => t.clone(),
                };
                let sliced = dequant.narrow(weight_dim, offset, len)?.contiguous()?;
                Ok(Self {
                    inner: QMatMulInner::Standard(candle_core::quantized::QMatMul::Tensor(sliced)),
                })
            }
            QMatMulInner::FusedSlice { fused, .. } => {
                let weight_dim = if dim == 1 { 0 } else { 1 };
                let dequant = match fused.as_ref() {
                    candle_core::quantized::QMatMul::QTensor(qt) => qt.dequantize(device)?,
                    candle_core::quantized::QMatMul::Tensor(t)
                    | candle_core::quantized::QMatMul::TensorF16(t) => t.clone(),
                };
                let sliced = dequant.narrow(weight_dim, offset, len)?.contiguous()?;
                Ok(Self {
                    inner: QMatMulInner::Standard(candle_core::quantized::QMatMul::Tensor(sliced)),
                })
            }
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
    /// Gate projection (optional — absent in Starcoder2 which uses a 2-layer MLP).
    /// When present: output = act(gate(x)) * up(x) → down (GLU-style).
    /// When absent: output = act(up(x)) → down (simple MLP).
    pub(crate) ffn_gate: Option<QMatMul>,
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
        let mut up = self.ffn_up.forward(xs)?;

        if let Some((adapter, abs_layer)) = lora {
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

        // GLU-style (gate present): act(gate(x)) * up(x)
        // Simple MLP (no gate): act(up(x))
        let combined = if let Some(ref ffn_gate) = self.ffn_gate {
            let mut gate = ffn_gate.forward(xs)?;
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
            }
            let activated = match self.activation {
                Activation::SiLU => candle_nn::ops::silu(&gate)?,
                Activation::Gelu => gate.gelu()?,
            };
            (activated * up)?
        } else {
            // No gate — simple activation on up projection
            match self.activation {
                Activation::SiLU => candle_nn::ops::silu(&up)?,
                Activation::Gelu => up.gelu()?,
            }
        };

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

/// Gating function applied to raw router logits before top-k selection.
///
/// - `Softmax` — softmax over all experts (Mixtral, Qwen3-MoE, DeepSeek-V2
///   default, Llama 4). This is the historical default.
/// - `Sigmoid` — element-wise sigmoid (DeepSeek-V3 routed-experts gate).
///   Scores are independent per expert; no `Σ = 1` constraint pre-topk.
///
/// Maps from GGUF `{arch}.expert_gating_func` (uint, 1 = softmax, 2 =
/// sigmoid; matches llama.cpp's enum). Missing key → `Softmax`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum MoeGatingFunc {
    #[default]
    Softmax,
    Sigmoid,
}

/// R132: MoE router policy. Combines the gating function with whether the
/// top-k weights are renormalized to sum to 1. Default = Softmax + renorm
/// (Mixtral / Qwen3-MoE with `norm_topk_prob=true` / DeepSeek-V3 default).
///
/// Non-default combinations:
/// - `Softmax + renormalize=false`: DeepSeek-V2 strict spec — top-k weights
///   sum to less than 1.
/// - `Sigmoid + renormalize=true`: DeepSeek-V3 with weights normalization
///   on (the GGUF metadata key controls this independently of the gating
///   function).
/// - `Sigmoid + renormalize=false`: rare; sigmoid scores used directly.
///
/// Sourced from GGUF metadata: `{arch}.expert_gating_func` (uint) +
/// `{arch}.expert_weights_norm` (bool). Missing keys → softmax + renorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoeRoutingConfig {
    pub(crate) gating_func: MoeGatingFunc,
    pub(crate) renormalize_weights: bool,
}

impl Default for MoeRoutingConfig {
    fn default() -> Self {
        Self {
            gating_func: MoeGatingFunc::Softmax,
            renormalize_weights: true,
        }
    }
}

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
    /// R132: per-architecture routing policy (gating function + weight
    /// normalization). Defaults to softmax + renormalize — the historical
    /// behaviour that matches Mixtral / Qwen3-MoE / Llama 4 / DeepSeek-V3
    /// default. Differentiates DeepSeek-V2 strict (no renorm) and
    /// DeepSeek-V3 sigmoid gating.
    pub(crate) routing: MoeRoutingConfig,
}

/// Select top-k indices and weights from a RAW router-logit vector on CPU,
/// applying the gating function + optional renormalization specified by
/// `config`.
///
/// **Softmax + renormalize** (default; Mixtral / Qwen3-MoE
/// `norm_topk_prob=true` / DeepSeek-V3 default / Llama 4) takes the fast
/// path that exploits the algebraic identity
/// `softmax(raw[topk]) ≡ softmax(raw)[topk] / Σ_{j∈topk} softmax(raw)[j]`
/// — same numerical result, but `O(k)` work instead of
/// `O(n_experts)` softmax + `O(k)` renormalize. Picking is monotonic
/// in `raw` because softmax is monotonic, so sorting raw scores directly
/// yields the same top-k as sorting softmax probabilities.
///
/// **Softmax + no renormalize** (DeepSeek-V2 strict, Qwen3-MoE with
/// `norm_topk_prob=false`) computes the full softmax once, then takes
/// the top-k probabilities directly. Output weights sum to less than 1.
///
/// **Sigmoid + renormalize** (DeepSeek-V3 with weights_norm=true) applies
/// sigmoid element-wise (no `Σ = 1` constraint), picks top-k by score,
/// then renormalizes so weights sum to 1.
///
/// **Sigmoid + no renormalize** picks top-k sigmoid scores directly.
///
/// References: llama.cpp `build_moe_ffn`
/// (`llama_expert_gating_func_type` + `norm_topk_prob`),
/// `transformers` `modeling_deepseek_v3.py::topk_weights`, Mixtral
/// `modeling_mixtral.py::sparse_mixtral_block`.
pub(crate) fn topk_cpu(
    scores: &Tensor,
    k: usize,
    config: MoeRoutingConfig,
) -> CandleResult<(Tensor, Tensor)> {
    let device = scores.device();
    let scores_vec: Vec<f32> = scores.to_vec1()?;
    let n = scores_vec.len();
    let k = k.min(n);

    // Apply the gating function to ALL scores. Sigmoid we always need to
    // realize element-wise (top-k by sigmoid score = top-k by raw, since
    // sigmoid is monotonic — but the WEIGHTS need the sigmoid value).
    // Softmax has the fast-path identity that skips the all-experts
    // softmax when renormalizing.
    let gated_all: Option<Vec<f32>> = match (config.gating_func, config.renormalize_weights) {
        (MoeGatingFunc::Softmax, true) => None,
        (MoeGatingFunc::Softmax, false) => {
            // Need the full softmax so we can take the (un-renormalized)
            // probabilities at the top-k positions directly.
            let max = scores_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<f32> = scores_vec.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            if sum > 0.0 {
                for v in exps.iter_mut() {
                    *v /= sum;
                }
            }
            Some(exps)
        }
        (MoeGatingFunc::Sigmoid, _) => Some(
            scores_vec
                .iter()
                .map(|s| 1.0 / (1.0 + (-s).exp()))
                .collect(),
        ),
    };

    // Pick top-k by the (raw or sigmoid) score — sort is the same in
    // either case because both are monotonic in `raw`.
    let mut indices: Vec<usize> = (0..n).collect();
    let sort_key: &[f32] = gated_all.as_deref().unwrap_or(&scores_vec);
    indices.sort_by(|&a, &b| {
        sort_key[b]
            .partial_cmp(&sort_key[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);

    // Build the weights vector — semantics depend on gating + renorm.
    let weights: Vec<f32> = match (config.gating_func, config.renormalize_weights) {
        (MoeGatingFunc::Softmax, true) => {
            // Fast path: softmax(raw[topk]) ≡ renorm(softmax(raw)[topk])
            // Apply softmax-over-k at the end via candle.
            indices.iter().map(|&i| scores_vec[i]).collect()
        }
        (MoeGatingFunc::Softmax, false) => {
            // Take probabilities at top-k positions directly.
            let probs = gated_all
                .as_ref()
                .expect("gated_all set for softmax+no_norm");
            indices.iter().map(|&i| probs[i]).collect()
        }
        (MoeGatingFunc::Sigmoid, renorm) => {
            let probs = gated_all.as_ref().expect("gated_all set for sigmoid");
            let topk_probs: Vec<f32> = indices.iter().map(|&i| probs[i]).collect();
            if renorm {
                let sum: f32 = topk_probs.iter().sum();
                if sum > 0.0 {
                    topk_probs.iter().map(|w| w / sum).collect()
                } else {
                    topk_probs
                }
            } else {
                topk_probs
            }
        }
    };

    let idx_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    let idx_tensor = Tensor::from_vec(idx_i64, (k,), device)?;
    let w_tensor = Tensor::from_vec(weights, (k,), device)?;
    // Only the Softmax+renorm fast path still needs a candle softmax —
    // the others already produced final weights on CPU.
    let w_tensor = if matches!(
        (config.gating_func, config.renormalize_weights),
        (MoeGatingFunc::Softmax, true)
    ) {
        candle_nn::ops::softmax(&w_tensor, 0)?
    } else {
        w_tensor
    };
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
            let (indices, weights) = topk_cpu(&token_scores, self.n_experts_used, self.routing)?;
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

impl LayerWeights {
    /// Pre-split weights for tensor parallelism, reducing VRAM per rank.
    ///
    /// Follows the Megatron-LM TP strategy:
    /// - Column-parallel (Q/K/V/gate/up): split output dim across heads/columns
    /// - Row-parallel (output/down): split input dim to match column-parallel output
    ///
    /// Quantized weights are dequantized to f32 during splitting. The per-rank
    /// slice is smaller (1/tp_size), so total VRAM per rank decreases despite
    /// the f32 format.
    pub(crate) fn pre_split_for_tp(
        &mut self,
        tp_rank: usize,
        tp_size: usize,
        device: &Device,
    ) -> CandleResult<()> {
        if tp_size <= 1 {
            return Ok(());
        }

        let heads_per_rank = self.n_head / tp_size;
        let kv_heads_per_rank = self.n_kv_head.div_ceil(tp_size);
        let head_offset = tp_rank * heads_per_rank * self.head_dim;
        let head_len = heads_per_rank * self.head_dim;
        let kv_offset = tp_rank * kv_heads_per_rank * self.head_dim;
        let kv_len = kv_heads_per_rank * self.head_dim;

        // Column-parallel: Q, K, V (split output dim = dim 1)
        self.attention_wq = self
            .attention_wq
            .narrow_tp(1, head_offset, head_len, device)?;
        self.attention_wk = self.attention_wk.narrow_tp(1, kv_offset, kv_len, device)?;
        self.attention_wv = self.attention_wv.narrow_tp(1, kv_offset, kv_len, device)?;

        // Row-parallel: output projection (split input dim = dim 0)
        self.attention_wo = self
            .attention_wo
            .narrow_tp(0, head_offset, head_len, device)?;

        // Split biases if present
        if let Some(ref bq) = self.attention_bq {
            self.attention_bq = Some(bq.narrow(0, head_offset, head_len)?.contiguous()?);
        }
        if let Some(ref bk) = self.attention_bk {
            self.attention_bk = Some(bk.narrow(0, kv_offset, kv_len)?.contiguous()?);
        }
        if let Some(ref bv) = self.attention_bv {
            self.attention_bv = Some(bv.narrow(0, kv_offset, kv_len)?.contiguous()?);
        }

        // Split FFN weights
        if let FfnVariant::Dense(ref mut mlp) = self.ffn {
            // Get intermediate dim from ffn_down weight shape.
            // ffn_down weight is [hidden, intermediate] (row-parallel target).
            // Dequantize dim 1 to find intermediate size.
            let inter_dim = match &mlp.ffn_down.inner {
                QMatMulInner::Standard(m) => match m {
                    candle_core::quantized::QMatMul::QTensor(qt) => {
                        qt.shape().dims().get(1).copied().unwrap_or(0)
                    }
                    candle_core::quantized::QMatMul::Tensor(t)
                    | candle_core::quantized::QMatMul::TensorF16(t) => {
                        t.dims().get(1).copied().unwrap_or(0)
                    }
                },
                QMatMulInner::FusedSlice { fused, .. } => match fused.as_ref() {
                    candle_core::quantized::QMatMul::QTensor(qt) => {
                        qt.shape().dims().get(1).copied().unwrap_or(0)
                    }
                    candle_core::quantized::QMatMul::Tensor(t)
                    | candle_core::quantized::QMatMul::TensorF16(t) => {
                        t.dims().get(1).copied().unwrap_or(0)
                    }
                },
            };
            if inter_dim == 0 {
                return Err(candle_core::Error::Msg(
                    "Cannot determine FFN intermediate dim for TP split".into(),
                ));
            }
            let inter_per_rank = inter_dim / tp_size;
            let inter_offset = tp_rank * inter_per_rank;

            // Column-parallel: gate, up (split output dim)
            if let Some(ref gate) = mlp.ffn_gate {
                mlp.ffn_gate = Some(gate.narrow_tp(1, inter_offset, inter_per_rank, device)?);
            }
            mlp.ffn_up = mlp
                .ffn_up
                .narrow_tp(1, inter_offset, inter_per_rank, device)?;

            // Row-parallel: down (split input dim)
            mlp.ffn_down = mlp
                .ffn_down
                .narrow_tp(0, inter_offset, inter_per_rank, device)?;
        }

        // Update head counts to reflect the split
        self.n_head = heads_per_rank;
        self.n_kv_head = kv_heads_per_rank;

        Ok(())
    }
}

pub(crate) fn masked_fill(
    on_false: &Tensor,
    mask: &Tensor,
    on_true: &Tensor,
) -> CandleResult<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

/// Target size, in f32 elements, of ONE attention-score temporary.
///
/// The score matrix is `[batch, n_head, q_len, k_len]`, so it grows with the
/// PRODUCT of the query and key lengths — quadratically on a prefill, where
/// both are the prompt length. At 4600 prompt tokens and 8 heads that single
/// temporary is 646 MB, and the softcap/mask/softmax steps each produce
/// another of the same size. That is what exhausted a 6 GB card on a model
/// whose weights are 1.6 GB (reported 2026-08-01): the load succeeded and
/// reported 2883 MB resident, then the first long prompt died inside
/// attention.
///
/// 16 Mi elements = 64 MiB per temporary, so even several live at once stay
/// well inside the headroom a modest card has after the model is resident.
const ATTN_SCORE_BUDGET_ELEMS: usize = 16 * 1024 * 1024;

/// How many query positions to process at once so one score temporary stays
/// near [`ATTN_SCORE_BUDGET_ELEMS`].
///
/// Returns `q_len` (i.e. "do it in one pass") whenever the whole thing already
/// fits, so decode and short prefills take exactly the path they always did.
fn attention_query_block(q_len: usize, k_len: usize, heads: usize) -> usize {
    let per_query = heads.max(1).saturating_mul(k_len.max(1));
    if per_query == 0 {
        return q_len;
    }
    let budgeted = ATTN_SCORE_BUDGET_ELEMS / per_query;
    budgeted.clamp(1, q_len.max(1))
}

/// Standard O(n^2) matmul attention with optional causal mask.
/// Input/output layout: BHSD `(b, n_head, seq, head_dim)`.
/// Supports optional Gemma 2 attention logit soft-capping.
///
/// Long prompts are processed in blocks of query positions to bound peak
/// memory — see [`ATTN_SCORE_BUDGET_ELEMS`]. **This is exact, not an
/// approximation**: the softmax runs over the KEY axis, so output row `i`
/// depends only on query row `i` and the full key/value tensors. Splitting the
/// query axis therefore computes the identical arithmetic on each row, in the
/// same order, and merely materialises fewer of them at a time. No online
/// rescaling is involved, unlike a true flash-attention tiling over keys.
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
    // `v` is used once per block; make it contiguous here rather than inside
    // the loop so a blocked run does not repeat the copy per block.
    let v = v.contiguous()?;
    let kt = k.t()?;

    let q_len = q.dim(2)?;
    let k_len = k.dim(2)?;
    let block = attention_query_block(q_len, k_len, n_head);

    if block >= q_len {
        return attention_scores_block(q, &kt, &v, mask, head_dim, neg_inf, attn_logit_softcap);
    }

    let mut parts: Vec<Tensor> = Vec::with_capacity(q_len.div_ceil(block));
    let mut start = 0usize;
    while start < q_len {
        let len = block.min(q_len - start);
        // `q` reaches here via reshape/transpose and may not be contiguous;
        // narrowing it produces a strided view that candle's matmul rejects on
        // some backends. The block is [b, heads, len, head_dim] — a few MB —
        // so making it contiguous is cheap insurance against a device-specific
        // failure that would not show up on the CPU path used in tests.
        let q_blk = q.narrow(2, start, len)?.contiguous()?;
        // The mask is [q_len, k_len] (2D, broadcast over batch/head) or already
        // 4D. Either way the query axis is the second-from-last.
        let mask_blk = match mask {
            None => None,
            Some(m) => Some(match m.rank() {
                2 => m.narrow(0, start, len)?,
                r => m.narrow(r - 2, start, len)?,
            }),
        };
        parts.push(attention_scores_block(
            &q_blk,
            &kt,
            &v,
            mask_blk.as_ref(),
            head_dim,
            neg_inf,
            attn_logit_softcap,
        )?);
        start += len;
    }
    Tensor::cat(&parts, 2)
}

/// One block of [`standard_attention`] — the original body, over whatever
/// slice of query positions it is handed.
fn attention_scores_block(
    q: &Tensor,
    kt: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    neg_inf: &Tensor,
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    let att = (q.matmul(kt)? / (head_dim as f64).sqrt())?;
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
    att.matmul(v)
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
    let force_standard = crate::inference::attn_kernel::is_force_standard_attn();
    match q.device() {
        Device::Cpu => {
            let seq_len = q.dim(2)?;

            // Decode (seq_len=1) routing — measured crossover, see
            // `cpu_decode_bench::decode_seq1_fused_vs_standard`:
            //
            // * MHA (n_head == n_kv_head): standard_attention is 7-20×
            //   faster than fused flash at every KV length — repeat_kv is
            //   a no-op so fused's BHSD→BSHD transposes are pure overhead.
            // * GQA (n_head != n_kv_head): standard is faster up to ~1024
            //   KV, then repeat_kv's expansion-to-n_head cost dominates
            //   and fused wins. Crossover ~2048 on Qwen2.5-7B (28/4 GQA)
            //   and Llama-70B-style (32/8 GQA); at 4096 KV fused is 4-5×
            //   faster.
            //
            // SWIFT / spec sessions force standard regardless so prefill +
            // draft + verify share identical numerics (tiny softmax drift
            // breaks accept rate even at skip_ratio=0).
            const CPU_FUSED_DECODE_GQA_MIN_KV: usize = 2048;
            let is_gqa = n_head != n_kv_head;
            let kv_len = k.dim(2)?;
            let use_standard_for_decode =
                seq_len == 1 && (!is_gqa || kv_len < CPU_FUSED_DECODE_GQA_MIN_KV);
            if use_standard_for_decode || force_standard {
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
            // Also taken when SWIFT/spec sessions force standard attention so
            // baseline + draft + verify share identical numerics.
            if (k_len > q_len && q_len > 1) || force_standard {
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

mod qwen35;

#[cfg(test)]
mod cpu_decode_bench {
    //! Benchmark: fused CPU flash attention vs standard matmul on decode
    //! (`seq_len = 1`) across varied KV cache lengths. Answers whether the
    //! deliberate skip in `run_attention` at the top of this file is still
    //! the right call.
    //!
    //! Run with:
    //!   cargo test --release --no-default-features --features dev,claude-subscription \
    //!       --lib -- --ignored --nocapture inference::layers::cpu_decode_bench
    //!
    //! The test is `#[ignore]` so normal `cargo test` doesn't pay the time.

    use candle_core::{Device, Tensor};

    fn bench_one(
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        kv_len: usize,
        iters: usize,
    ) -> (f64, f64) {
        let device = Device::Cpu;
        let b = 1;
        let q_len = 1;

        // BHSD tensors (b, h, s, d) — matching run_attention's input layout.
        let q = Tensor::randn(0f32, 1.0, (b, n_head, q_len, head_dim), &device).expect("q alloc");
        let k =
            Tensor::randn(0f32, 1.0, (b, n_kv_head, kv_len, head_dim), &device).expect("k alloc");
        let v =
            Tensor::randn(0f32, 1.0, (b, n_kv_head, kv_len, head_dim), &device).expect("v alloc");
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).expect("neg_inf alloc");

        // --- warm up standard path ---
        for _ in 0..3 {
            let _ = super::standard_attention(
                &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
            )
            .expect("std warm");
        }

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = super::standard_attention(
                &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
            )
            .expect("std iter");
        }
        let std_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // --- warm up fused path ---
        // The fused path ALWAYS includes BHSD→BSHD transposes + contiguous
        // copies; we time that as part of the cost because the real code
        // pays it on every call.
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        for _ in 0..3 {
            let q_bshd = q.transpose(1, 2).unwrap().contiguous().unwrap();
            let k_bshd = k.transpose(1, 2).unwrap().contiguous().unwrap();
            let v_bshd = v.transpose(1, 2).unwrap().contiguous().unwrap();
            let _ = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                None,
                softmax_scale,
                None,
                None,
            )
            .expect("fused warm");
        }

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let q_bshd = q.transpose(1, 2).unwrap().contiguous().unwrap();
            let k_bshd = k.transpose(1, 2).unwrap().contiguous().unwrap();
            let v_bshd = v.transpose(1, 2).unwrap().contiguous().unwrap();
            let _ = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                None,
                softmax_scale,
                None,
                None,
            )
            .expect("fused iter");
        }
        let fused_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        (std_ms, fused_ms)
    }

    fn report(
        label: &str,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        kv_lens: &[usize],
        iters: usize,
    ) {
        println!("\n=== {label}  heads={n_head}  kv_heads={n_kv_head}  head_dim={head_dim} ===",);
        println!(
            "{:>6}  {:>10}  {:>10}  {:>8}",
            "kv_len", "std_ms", "fused_ms", "ratio"
        );
        for &kv in kv_lens {
            let (s, f) = bench_one(n_head, n_kv_head, head_dim, kv, iters);
            let ratio = f / s;
            let winner = if ratio < 0.95 {
                "fused WIN"
            } else if ratio > 1.05 {
                "std WIN"
            } else {
                "~tie"
            };
            println!(
                "{:>6}  {:>10.3}  {:>10.3}  {:>8.2}x  {}",
                kv, s, f, ratio, winner
            );
        }
    }

    #[test]
    #[ignore]
    fn decode_seq1_fused_vs_standard() {
        let kv_lens: &[usize] = &[128, 512, 1024, 2048, 4096, 8192];
        let iters = 50;

        // TinyLlama (MHA: n_head = n_kv = 32, head_dim = 64)
        report("TinyLlama-style MHA", 32, 32, 64, kv_lens, iters);
        // Llama-2-7B / -3-8B style (MHA: 32/32, head_dim=128)
        report("Llama-7B-style MHA", 32, 32, 128, kv_lens, iters);
        // Qwen2.5-7B (GQA: 28 heads / 4 kv_heads, head_dim=128)
        report("Qwen2.5-7B-style GQA", 28, 4, 128, kv_lens, iters);
        // Llama-3-70B / Mistral-7B style (GQA: 32/8, head_dim=128)
        report("Llama-70B-style GQA", 32, 8, 128, kv_lens, iters);
    }
}

#[cfg(test)]
mod blocked_attention_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Run `standard_attention` with the query-blocking threshold forced to a
    /// given block size, by choosing shapes that straddle it.
    fn run(
        q_len: usize,
        k_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        softcap: Option<f32>,
    ) -> Tensor {
        let dev = Device::Cpu;
        // Deterministic, non-trivial values — a constant tensor would hide any
        // row/column mix-up that blocking could introduce.
        let mk = |b: usize, h: usize, s: usize, d: usize, seed: f32| {
            let n = b * h * s * d;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.7 + seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(data, (b, h, s, d), &dev).unwrap()
        };
        let q = mk(1, n_head, q_len, head_dim, 0.0);
        let k = mk(1, n_kv_head, k_len, head_dim, 1.3);
        let v = mk(1, n_kv_head, k_len, head_dim, 2.9);
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &dev).unwrap();
        // Causal mask over the query block, offset so the last query attends to
        // everything — the shape a prefill actually uses.
        let offset = k_len - q_len;
        let m: Vec<u8> = (0..q_len)
            .flat_map(|i| (0..k_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        let mask = Tensor::from_vec(m, (q_len, k_len), &dev).unwrap();
        standard_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            head_dim,
            n_head,
            n_kv_head,
            &neg_inf,
            softcap,
        )
        .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            a.len(),
            b.len(),
            "shape mismatch between blocked and unblocked"
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The blocking claim is that it is EXACT, not approximate. Softmax runs
    /// over the key axis, so output row i depends only on query row i — the
    /// same arithmetic in the same order, just fewer rows materialised at once.
    ///
    /// Verified by forcing a block split through the real entry point: the
    /// budget is 16Mi elements, so heads x k_len chosen to make the block
    /// smaller than q_len.
    #[test]
    fn blocking_the_query_axis_is_bit_for_bit_identical() {
        // per_query = heads * k_len. Pick k_len large enough that the computed
        // block is < q_len, forcing the blocked path.
        let heads = 8;
        let k_len = 4096;
        let q_len = 4096;
        assert!(
            attention_query_block(q_len, k_len, heads) < q_len,
            "test must actually exercise the blocked path"
        );

        let blocked = run(q_len, k_len, heads, heads, 16, None);

        // Same computation with the block forced to cover everything: call the
        // single-block helper directly with identical inputs.
        let dev = Device::Cpu;
        let mk = |b: usize, h: usize, s: usize, d: usize, seed: f32| {
            let n = b * h * s * d;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.7 + seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(data, (b, h, s, d), &dev).unwrap()
        };
        let q = mk(1, heads, q_len, 16, 0.0);
        let k = mk(1, heads, k_len, 16, 1.3);
        let v = mk(1, heads, k_len, 16, 2.9).contiguous().unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &dev).unwrap();
        let offset = k_len - q_len;
        let m: Vec<u8> = (0..q_len)
            .flat_map(|i| (0..k_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        let mask = Tensor::from_vec(m, (q_len, k_len), &dev).unwrap();
        let unblocked =
            attention_scores_block(&q, &k.t().unwrap(), &v, Some(&mask), 16, &neg_inf, None)
                .unwrap();

        let d = max_abs_diff(&blocked, &unblocked);
        assert_eq!(d, 0.0, "query blocking must be exact, max abs diff = {d}");
    }

    /// Gemma 2's softcap adds three more full-size temporaries, so it is the
    /// model that benefits most — and the one whose numerics must not drift.
    #[test]
    fn blocking_is_exact_with_gemma_style_softcap() {
        let heads = 8;
        let k_len = 4096;
        let q_len = 4096;
        let blocked = run(q_len, k_len, heads, heads, 16, Some(50.0));

        let dev = Device::Cpu;
        let mk = |b: usize, h: usize, s: usize, d: usize, seed: f32| {
            let n = b * h * s * d;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.7 + seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(data, (b, h, s, d), &dev).unwrap()
        };
        let q = mk(1, heads, q_len, 16, 0.0);
        let k = mk(1, heads, k_len, 16, 1.3);
        let v = mk(1, heads, k_len, 16, 2.9).contiguous().unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &dev).unwrap();
        let offset = k_len - q_len;
        let m: Vec<u8> = (0..q_len)
            .flat_map(|i| (0..k_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        let mask = Tensor::from_vec(m, (q_len, k_len), &dev).unwrap();
        let unblocked = attention_scores_block(
            &q,
            &k.t().unwrap(),
            &v,
            Some(&mask),
            16,
            &neg_inf,
            Some(50.0),
        )
        .unwrap();

        assert_eq!(max_abs_diff(&blocked, &unblocked), 0.0);
    }

    /// GQA must keep working — `repeat_kv` happens once, before blocking.
    #[test]
    fn grouped_query_attention_still_matches() {
        let out = run(2048, 2048, 8, 4, 16, None);
        assert_eq!(out.dims(), &[1, 8, 2048, 16]);
    }

    /// Decode and short prefills must take the original single-pass path, so
    /// the common case pays nothing for this.
    #[test]
    fn short_sequences_are_not_blocked() {
        // Decode: one query position.
        assert_eq!(attention_query_block(1, 4096, 8), 1);
        // A 128-token chunk against a long cache still fits in one pass.
        assert_eq!(attention_query_block(128, 4600, 8), 128);
    }

    /// The whole point: peak score memory must stop growing with the square of
    /// the prompt.
    #[test]
    fn the_block_bounds_peak_score_memory() {
        let heads = 8;
        for q_len in [2048usize, 4600, 8192, 16384] {
            let blk = attention_query_block(q_len, q_len, heads);
            let elems = blk * q_len * heads;
            assert!(
                elems <= ATTN_SCORE_BUDGET_ELEMS,
                "q_len={q_len}: block {blk} gives {elems} elems, over budget"
            );
        }
        // Unblocked, 4600 tokens x 8 heads would have been 646 MB per temporary.
        let unbounded = 4600usize * 4600 * heads;
        assert!(unbounded > 10 * ATTN_SCORE_BUDGET_ELEMS);
    }
}
