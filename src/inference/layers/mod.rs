//! Transformer layer weight structures and forward pass implementations.
//!
//! Includes Qwen 3.5 hybrid SSM+attention (Gated Delta Networks) layer types.

use crate::inference::prof::Stage as P;
use candle_core::quantized::QTensor;
use candle_core::{Device, Result as CandleResult, Tensor};
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

/// Peak MLP intermediate elements to keep live at once, matching
/// [`ATTN_SCORE_BUDGET_ELEMS`]. 16 Mi elements = 64 MiB per temporary.
///
/// The feed-forward block holds `up` and `gate` — each
/// `[tokens, intermediate]` — plus their product, all live at the same moment.
/// At 5021 tokens and Gemma 2's 9216-wide intermediate that is ~185 MiB each,
/// so roughly 700 MiB peak on top of a resident model. Reported live
/// 2026-08-02 on a 6 GB card: the model loaded at 2829 MiB and the first long
/// prompt died in `mlp:`.
const MLP_INTERMEDIATE_BUDGET_ELEMS: usize = 16 * 1024 * 1024;

/// Token count below which blocking is skipped entirely.
///
/// Decode passes a single token and must not pay for the width probe below, and
/// a short prefill already fits — both take exactly the path they always did.
const MLP_MIN_TOKENS_TO_BLOCK: usize = 512;

impl Mlp {
    /// Feed-forward block.
    ///
    /// Long inputs are processed in blocks of tokens to bound peak memory — see
    /// [`MLP_INTERMEDIATE_BUDGET_ELEMS`]. **Each token receives exactly the same
    /// computation, not an approximation of it**: every operation here is
    /// pointwise across tokens (two projections, an activation, a product, a
    /// third projection), so row `i` of the output depends only on row `i` of
    /// the input. Splitting the token axis merely materialises fewer rows at
    /// once. No online rescaling is involved.
    ///
    /// On CPU that is bit-for-bit identical, asserted at 0.0 max abs diff in
    /// `blocked_mlp_tests`. **On GPU it is not guaranteed to be bit-identical**:
    /// cuBLAS selects a kernel from the problem shape, so changing the row count
    /// can change the order the inner dimension is accumulated in. Measured at
    /// 2.3e-4 relative on an RTX 3070 (`cuda_mlp_memory_probe`) — float
    /// reassociation, orders of magnitude below the error the model's own
    /// quantisation already carries. Do not "fix" this by forcing one kernel;
    /// the arithmetic per token is already correct.
    ///
    /// Chunked prefill does NOT make this redundant. It bounds tokens only on
    /// the local generate path; a node serving a pipeline segment goes through
    /// `handle_forward`, which has no chunking — and a single machine holding a
    /// whole model is served as a one-segment pipeline, which is why this was
    /// reachable without any second machine involved.
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        lora: Option<(&LoraAdapter, usize)>,
    ) -> CandleResult<Tensor> {
        self.forward_with_budget(xs, lora, MLP_INTERMEDIATE_BUDGET_ELEMS)
    }

    /// [`Mlp::forward`] with the memory budget injected, so a test can force the
    /// blocked path without allocating the hundreds of megabytes the real
    /// budget would require to exceed.
    fn forward_with_budget(
        &self,
        xs: &Tensor,
        lora: Option<(&LoraAdapter, usize)>,
        budget_elems: usize,
    ) -> CandleResult<Tensor> {
        // Token axis is the second-to-last dim for both [b, seq, hidden] and
        // [seq, hidden] inputs.
        let rank = xs.rank();
        if rank < 2 {
            return self.forward_block(xs, lora);
        }
        let token_dim = rank - 2;
        let tokens = xs.dim(token_dim)?;
        if tokens < MLP_MIN_TOKENS_TO_BLOCK {
            return self.forward_block(xs, lora);
        }

        // Measure the intermediate width instead of assuming an expansion
        // ratio — it varies from ~2.7x to ~8x of hidden across architectures,
        // and a guess would either under-bound (defeating the fix) or
        // over-block (needless launches). One single-token projection against a
        // prefill of hundreds is negligible, and it is never run for decode.
        let probe = xs.narrow(token_dim, 0, 1)?;
        let intermediate =
            crate::inference::prof::timed!(P::FfnProbe, self.ffn_up.forward(&probe))?
                .dims()
                .last()
                .copied()
                .unwrap_or(0);
        let block = budget_elems
            .checked_div(intermediate)
            .map_or(tokens, |b| b.clamp(1, tokens));
        if block >= tokens {
            return self.forward_block(xs, lora);
        }

        let mut parts: Vec<Tensor> = Vec::with_capacity(tokens.div_ceil(block));
        let mut start = 0usize;
        while start < tokens {
            let len = block.min(tokens - start);
            // Narrowing yields a strided view; the projections' matmul rejects
            // those on some backends, and a block is only a few MiB.
            let slice = xs.narrow(token_dim, start, len)?.contiguous()?;
            parts.push(self.forward_block(&slice, lora)?);
            start += len;
        }
        Tensor::cat(&parts, token_dim)
    }

    /// One block's worth of the feed-forward computation — the whole thing when
    /// the input is short enough not to need splitting.
    fn forward_block(
        &self,
        xs: &Tensor,
        lora: Option<(&LoraAdapter, usize)>,
    ) -> CandleResult<Tensor> {
        let mut up = crate::inference::prof::timed!(P::FfnUpGate, self.ffn_up.forward(xs))?;

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
            let mut gate = crate::inference::prof::timed!(P::FfnUpGate, ffn_gate.forward(xs))?;
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
            let __act_t = std::time::Instant::now();
            let activated = match self.activation {
                Activation::SiLU => candle_nn::ops::silu(&gate)?,
                Activation::Gelu => gate.gelu()?,
            };
            let __prod = (activated * up)?;
            crate::inference::prof::add(P::FfnAct, __act_t.elapsed().as_nanos() as u64);
            __prod
        } else {
            // No gate — simple activation on up projection
            match self.activation {
                Activation::SiLU => candle_nn::ops::silu(&up)?,
                Activation::Gelu => up.gelu()?,
            }
        };

        let mut down =
            crate::inference::prof::timed!(P::FfnDown, self.ffn_down.forward(&combined))?;
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
        let __kv_t = std::time::Instant::now();
        let (k, v) = match kv_cache {
            None => {
                let mut cache = new_kv_cache(max_seq_len);
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
        crate::inference::prof::add(P::KvCache, __kv_t.elapsed().as_nanos() as u64);

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
            None,        // no softcap for DeepSeek
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

/// One expert's gated feed-forward over the tokens routed to it, blocked on the
/// token axis exactly as [`Mlp::forward`] is.
///
/// Routing means each expert usually sees a fraction of the tokens, which is
/// why this was not the path that OOM'd first — but a fraction is not a bound.
/// A router that sends most tokens to one expert reproduces the dense shape:
/// `gate_out`, `up_out`, their activation and their product are all
/// `[tokens, intermediate]` and live at the same moment, which is what exhausted
/// a 6 GB card on the dense path.
///
/// Exact for the same reason: every step here treats each token independently,
/// so splitting rows changes only how many are materialised at once.
fn expert_ffn(
    batch_input: &Tensor,
    gate_w: &Tensor,
    up_w: &Tensor,
    down_w: &Tensor,
) -> CandleResult<Tensor> {
    let tokens = batch_input.dim(0)?;
    // `gate_w` is [intermediate, hidden]; the projection widens to its dim 0.
    let intermediate = gate_w.dim(0)?;
    let block = if tokens < MLP_MIN_TOKENS_TO_BLOCK || intermediate == 0 {
        tokens
    } else {
        (MLP_INTERMEDIATE_BUDGET_ELEMS / intermediate).clamp(1, tokens)
    };

    let run = |input: &Tensor| -> CandleResult<Tensor> {
        let gate_out = input.matmul(&gate_w.t()?)?;
        let up_out = input.matmul(&up_w.t()?)?;
        let activated = candle_nn::ops::silu(&gate_out)?;
        let combined = (activated * up_out)?;
        combined.matmul(&down_w.t()?)
    };

    if block >= tokens {
        return run(batch_input);
    }

    let mut parts: Vec<Tensor> = Vec::with_capacity(tokens.div_ceil(block));
    let mut start = 0usize;
    while start < tokens {
        let len = block.min(tokens - start);
        let slice = batch_input.narrow(0, start, len)?.contiguous()?;
        parts.push(run(&slice)?);
        start += len;
    }
    Tensor::cat(&parts, 0)
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

            let expert_out = expert_ffn(&batch_input, &gate_w, &up_w, &down_w)?; // [batch_tokens, hidden]

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

/// How many sequence positions a KV cache reserves at a time.
///
/// **This is a growth quantum, not a limit.** candle's `Cache::append`
/// allocates a buffer of `max_seq_len` positions on the FIRST append and, when
/// that fills, grows by `grow_by` — and `Cache::new(dim, n)` sets *both* to
/// `n`. Passing the model's whole context length therefore reserved the entire
/// context window from the very first token: measured on llama-3.2-3b Q4_K_M,
/// one 904-token request held **940 MB allocated for 207 MB of real tokens,
/// 22% utilisation**, and a twenty-token chat would have held the same 940 MB.
/// The conversation's real ceiling is enforced separately, by the
/// `total_seq > max_seq_len` guard in `forward_inner_impl`, so shrinking this
/// value does not shorten any conversation.
///
/// llama.cpp has the same defect for the same reason — it pre-allocates
/// `n_ctx` at startup — and its proposed fix is a paged KV cache with a block
/// table (ggml-org/llama.cpp#21961), which candle's `grow_by` lets us skip.
///
/// **Why 512 rather than something smaller.** Growth is a `Tensor::cat`, so it
/// copies. The copy is per layer and per K/V separately, not over the whole
/// cache, so the transient is ~2x ONE layer's single buffer (about 34 MB on
/// this model) rather than 2x the 940 MB total. Reaching 4096 positions costs
/// seven grows totalling ~3.3 GB of copying spread across the whole
/// conversation — around 110 ms against the minutes such a conversation spends
/// decoding. 512 keeps that negligible while cutting the reservation for a
/// typical chat by 8x.
pub(crate) const KV_CACHE_GROWTH_TOKENS: usize = 512;

/// Build the KV cache for one layer, reserving space incrementally.
///
/// **Every KV cache must be created here.** Calling `KvCache::new` with a
/// model's `max_seq_len` directly is the bug this exists to prevent, and it
/// looks completely reasonable at the call site — the parameter is even named
/// `max_seq_len`, so passing it reads as correct.
pub(crate) fn new_kv_cache(max_seq_len: usize) -> KvCache {
    KvCache::new(2, KV_CACHE_GROWTH_TOKENS.min(max_seq_len.max(1)))
}

/// Reservation for a cache that must hold `positions` tokens the moment it is
/// created — a prefix-cache snapshot being hydrated, rather than a
/// conversation growing a token at a time.
///
/// Rounds up to whole [`KV_CACHE_GROWTH_TOKENS`] quanta so the result is one
/// allocation followed by the same growth behaviour as any other cache.
pub(crate) fn kv_cache_reservation(positions: usize) -> usize {
    positions.max(1).div_ceil(KV_CACHE_GROWTH_TOKENS) * KV_CACHE_GROWTH_TOKENS
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
/// `mask` is ADDITIVE and f32 — `0.0` where a query may attend, `-inf` where
/// it may not — which is the representation `run_flash_attn_cpu` already
/// wanted and the one every other engine uses. It used to be a `u8` predicate
/// fed to `masked_fill`, which cost more per layer than both matmuls and the
/// softmax combined: `where_cond` against a stride-0 scalar fill took 15.9 ms
/// where adding the same information took 4.5 (measured with
/// `examples/attn_bench.rs`). See [`crate::inference::attn_softmax`], which
/// now folds the scale, the cap and the mask into the softmax's own pass.
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
        return attention_scores_block(q, &kt, &v, mask, head_dim, attn_logit_softcap);
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
        //
        // `.contiguous()` is not cosmetic. A narrowed view keeps the parent's
        // row pitch, and every consumer downstream is slower on one: the fused
        // kernel declines strided operands outright, and `broadcast_add`
        // measured 9.7 ms against 4.5 for the same data contiguous. The copy is
        // one block of mask — kilobytes — against a score tensor of megabytes.
        let mask_blk = match mask {
            None => None,
            Some(m) => Some(
                match m.rank() {
                    2 => m.narrow(0, start, len)?,
                    r => m.narrow(r - 2, start, len)?,
                }
                .contiguous()?,
            ),
        };
        parts.push(attention_scores_block(
            &q_blk,
            &kt,
            &v,
            mask_blk.as_ref(),
            head_dim,
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
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    // Scale, soft-cap, mask and softmax are one pass over the score tensor
    // rather than four. Each of them used to materialise its own
    // `[batch, heads, q_len, kv_len]` temporary — 11 MB at a llama-3.2-3b
    // prompt chunk — so the tail of attention cost more in memory traffic than
    // the two matmuls around it. See `crate::inference::attn_softmax`.
    let att = q.matmul(kt)?;
    let att = crate::inference::attn_softmax::scaled_masked_softmax(
        &att,
        mask,
        crate::inference::attn_softmax::scale_from_head_dim(head_dim),
        attn_logit_softcap,
    )?;
    att.matmul(v)
}

// Compiled only where it is reachable: the CUDA dispatch (which needs
// `flash-attn`) and the tests that pin the rule. Without a gate this is dead
// code in every CPU build, and `#[allow(dead_code)]` would just hide that the
// policy has no consumer if the dispatch ever stops calling it.
#[cfg(any(feature = "flash-attn", test))]
/// Minimum KV length at which flash-attention beats `standard_attention` for a
/// single-token GQA decode step on CUDA.
///
/// Measured, not chosen — see the table in [`run_attention`]'s CUDA branch and
/// the `flash_vs_standard_attention_on_cuda` benchmark at the bottom of this
/// file. At 512 flash still loses (0.66x); at 1024 it wins (3.5x).
pub(crate) const GQA_FLASH_DECODE_MIN_KV: usize = 1024;

// Compiled only where it is reachable: the CUDA dispatch (which needs
// `flash-attn`) and the tests that pin the rule. Without a gate this is dead
// code in every CPU build, and `#[allow(dead_code)]` would just hide that the
// policy has no consumer if the dispatch ever stops calling it.
#[cfg(any(feature = "flash-attn", test))]
/// Should a CUDA attention call take `standard_attention` rather than flash,
/// on grounds of shape alone?
///
/// Extracted from the dispatch so the measured routing rule is pinned by tests
/// that need no GPU. Getting this wrong is not a crash, it is a silent 4x-25x
/// slowdown in generation — the failure mode that made the CPU crossover worth
/// a gotcha entry (#255), reproduced on a different device.
///
/// Only answers the decode question; prefill (`q_len > 1`) always prefers
/// flash, and the caller separately handles the offset-causal-mask fallback
/// and the SWIFT/spec force-standard override.
pub(crate) fn cuda_decode_prefers_standard(
    q_len: usize,
    k_len: usize,
    n_head: usize,
    n_kv_head: usize,
) -> bool {
    let is_gqa = n_head != n_kv_head;
    q_len == 1 && !(is_gqa && k_len >= GQA_FLASH_DECODE_MIN_KV)
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
            // * GQA (n_head != n_kv_head): fused, at EVERY KV length. This
            //   replaces a ~2048 crossover that made decode collapse on long
            //   contexts. Profiled 2026-08-06 on llama-3.2-3b Q4_K_M (24/8 GQA),
            //   ms per generated token, standard -> fused:
            //       kv ~82     141.2 -> 129.2   (attention  16.8 ->  14.7)
            //       kv ~1150  1368.1 -> 249.2   (attention 1242  -> 136 )  5.5x
            //   At ~1150 KV attention was 91% of decode under standard: repeat_kv
            //   materializes the cache expanded to n_head EVERY token EVERY layer,
            //   so cost grows with context and swamps everything else. Generating
            //   after a long prompt was ~10x slower per token than after a short
            //   one, which is the normal case in a chat.
            //
            //   The old crossover was measured on Qwen2.5-7B (28/4) and a 32/8
            //   model, not re-measured here. Those have HIGHER expansion ratios
            //   than 24/8, and repeat_kv's cost scales with the ratio, so fused
            //   should win at least as early for them — 3:1 is the least
            //   favourable GQA case for fused and it already wins at kv=82.
            //
            // SWIFT / spec sessions force standard regardless so prefill +
            // draft + verify share identical numerics (tiny softmax drift
            // breaks accept rate even at skip_ratio=0).
            let is_gqa = n_head != n_kv_head;
            let use_standard_for_decode = seq_len == 1 && !is_gqa;
            // PREFILL also takes the standard path — measured 2026-08-06, and the
            // opposite of what the decode routing above might suggest.
            //
            // `run_flash_attn_cpu` parallelizes over KV TILES OF 16 inside a
            // per-query-row loop and heap-allocates a scratch vec per tile, so a
            // 128-token chunk against 384 KV issues ~2M allocations across 28
            // layers at a parallel granularity far too fine to pay for itself. It
            // also runs on its own rayon pool sized to every logical core, ignoring
            // `inference_cpu_threads`. Standard attention batches the same work
            // into two real matmuls per head.
            //
            // Stage profile, seq_len=128 against kv_len=384, llama-3.2-3b Q4_K_M:
            //   attention core   4571 ms (45.5% of the chunk) -> 640 ms (10.4%)  7.1x
            // End to end, prompt processing:
            //    412 tokens  15.3 -> 20.7 tok/s   (1.35x)
            //   1537 tokens   7.0 -> 14.5 tok/s   (2.07x)
            // The gain grows with context because the flash kernel's overhead
            // scales with kv_len. Measured up to ~1550 KV; beyond that the
            // crossover is unmeasured, but `standard_attention` blocks its score
            // matrix (ATTN_SCORE_BUDGET_ELEMS) so memory stays bounded either way.
            let use_standard_for_prefill = seq_len > 1;
            if use_standard_for_decode || use_standard_for_prefill || force_standard {
                return standard_attention(
                    q,
                    k,
                    v,
                    mask,
                    head_dim,
                    n_head,
                    n_kv_head,
                    attn_logit_softcap,
                );
            }

            // CPU flash attention: input BSHD, output BHSD
            // Transpose Q/K/V from BHSD (b,h,s,d) to BSHD (b,s,h,d)
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();

            // The mask is already the additive f32 form this kernel takes.
            // It used to be built here by converting a u8 predicate, which was
            // the second mask representation in the codebase and the reason the
            // standard path could not share one; there is one now.
            //
            // Reached only for GQA decode, where `seq_len == 1` and the caller
            // passes `None` — but pass it through rather than dropping it, so
            // a future caller that does supply one gets it applied instead of
            // silently ignored.
            let out = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                mask,
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
            //
            // DECODE ROUTING — measured 2026-08-07 on an RTX 3070 (sm_86) with
            // `flash_vs_standard_attention_on_cuda` at the bottom of this file.
            // This is the same lesson as the CPU crossover above (gotcha #255):
            // the right kernel is OPPOSITE for prefill and decode, and it turns
            // on GQA. Shipping flash for everything, which is what the plain
            // `q_len > 1` causal flag invites, makes generation far slower.
            //
            // ms per call, standard -> flash (min of 20, idle GPU):
            //
            //   phi-3.5  MHA 32/32 d96      llama-3.2  GQA 24/8 d128
            //   kv  512  0.121 ->  0.511    kv  512  0.217 -> 0.326   0.66x
            //   kv 1024  0.162 ->  3.012    kv 1024  2.501 -> 0.715   3.50x
            //   kv 2048  0.235 ->  4.802    kv 2048  4.145 -> 2.935   1.41x
            //   kv 4096  0.397 ->  9.622    kv 4096  5.895 -> 2.833   2.08x
            //   kv 8192  0.702 -> 12.461    kv 8192  9.416 -> 4.791   1.97x
            //
            // * MHA: flash is 4x-25x SLOWER at every KV length, and the gap
            //   widens with context. candle-flash-attn ships no split-KV
            //   kernels, so a single query row launches a grid of only
            //   (1 x n_head x batch) blocks and leaves most of the card idle,
            //   while standard decode is two efficient GEMVs with `repeat_kv`
            //   a no-op (n_head == n_kv_head).
            // * GQA: the reverse above ~1k of context, because `repeat_kv` is
            //   NOT a no-op there — standard materializes the cache expanded to
            //   n_head every token, and its cost climbs with KV (0.217 -> 9.416)
            //   while flash's stays roughly flat. Same mechanism as #255.
            //
            // 1024 is the measured crossover: at 512 flash still loses (0.66x),
            // at 1024 it wins. Prefill is unconditional — flash won every
            // prefill shape measured, 2.4x-7.8x.
            let decode_prefers_standard =
                cuda_decode_prefers_standard(q_len, k_len, n_head, n_kv_head);
            if (k_len > q_len && q_len > 1) || decode_prefers_standard || force_standard {
                return standard_attention(
                    q,
                    k,
                    v,
                    mask,
                    head_dim,
                    n_head,
                    n_kv_head,
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

            // flash_attn handles GQA and causal masking natively.
            //
            // Softcap must be threaded through explicitly: the plain
            // `flash_attn` entry point hardcodes `softcap: None`, so a model
            // that softcaps its attention logits — Gemma-2 caps at 50.0 —
            // would get a DIFFERENT, silently wrong distribution here than on
            // the CPU and standard paths, which both honour it. No error, no
            // warning, just worse answers on one device. The windowed entry
            // point takes it; `window_size_right = Some(0)` with
            // `window_size_left = None` is exactly how `flash_attn` itself
            // expresses `causal`, so the two arms differ only in the cap.
            let causal_right = if q_len > 1 { Some(0) } else { None };
            let out = match attn_logit_softcap {
                Some(cap) => candle_flash_attn::flash_attn_alibi_windowed_softcap(
                    &q_f16,
                    &k_f16,
                    &v_f16,
                    None, // no ALiBi — RoPE models only
                    softmax_scale,
                    None, // unlimited left context
                    causal_right,
                    cap,
                )?,
                None => {
                    candle_flash_attn::flash_attn(&q_f16, &k_f16, &v_f16, softmax_scale, q_len > 1)?
                }
            };

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
        let mut q = crate::inference::prof::timed!(P::QkvProj, self.attention_wq.forward(x))?;
        let mut k = crate::inference::prof::timed!(P::QkvProj, self.attention_wk.forward(x))?;
        let mut v = crate::inference::prof::timed!(P::QkvProj, self.attention_wv.forward(x))?;

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

        let __shape_t = std::time::Instant::now();
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
        crate::inference::prof::add(P::AttnShape, __shape_t.elapsed().as_nanos() as u64);

        // KV-cache: use pre-allocated KvCache buffers (avoids Tensor::cat per step)
        let (k, v) = match kv_cache {
            None => {
                let mut cache = new_kv_cache(max_seq_len);
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
        let y = crate::inference::prof::timed!(
            P::AttnCore,
            run_attention(
                &q,
                &k,
                &v,
                mask,
                self.n_head,
                self.n_kv_head,
                self.head_dim,
                self.attn_logit_softcap,
            )
        )?;

        let attn_out_dim = self.n_head * self.head_dim;
        let y = crate::inference::prof::timed!(
            P::AttnShape,
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, attn_out_dim])
        )?;
        let mut wo_out = crate::inference::prof::timed!(P::AttnOut, self.attention_wo.forward(&y))?;

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

        // --- warm up standard path ---
        for _ in 0..3 {
            let _ = super::standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None)
                .expect("std warm");
        }

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = super::standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None)
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
        // Causal mask over the query block, offset so the last query attends to
        // everything — the shape a prefill actually uses.
        let offset = k_len - q_len;
        let m: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..k_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
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
        let offset = k_len - q_len;
        let m: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..k_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(m, (q_len, k_len), &dev).unwrap();
        let unblocked =
            attention_scores_block(&q, &k.t().unwrap(), &v, Some(&mask), 16, None).unwrap();

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
        let offset = k_len - q_len;
        let m: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..k_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(m, (q_len, k_len), &dev).unwrap();
        let unblocked =
            attention_scores_block(&q, &k.t().unwrap(), &v, Some(&mask), 16, Some(50.0)).unwrap();

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

/// CUDA-only probe for the memory behaviour blocking exists to fix.
///
/// Ignored by default — needs a real GPU. Run with:
/// ```text
/// CUDA_COMPUTE_CAP=86 cargo test --release --features candle-cuda \
///   attention_survives -- --ignored --nocapture
/// ```
///
/// The geometry is a pipeline segment's prefill: `handle_forward` hands the
/// WHOLE prompt to one forward (unlike local generation, which chunked prefill
/// already bounds to 128 positions), so `q_len` is the full prompt length.
///
/// **Verified on an RTX 3070 Laptop (8 GB), 2026-08-01.** Raising
/// `ATTN_SCORE_BUDGET_ELEMS` so blocking never triggers makes this exact test
/// fail with `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` — the error a tester
/// reported from the field. With blocking it completes. Same binary, same
/// inputs, one constant changed: that is the causal evidence for this fix, and
/// re-running that A/B is the way to re-establish it if the code moves.
///
/// Note this cannot be reproduced through a normal local request: those take
/// `route=local`, where chunked prefill already caps `q_len` at 128. Measuring
/// peak VRAM on a local request showed no difference (7956 MB vs 7882 MB)
/// precisely because that path was never the vulnerable one.
#[cfg(test)]
mod cuda_attention_memory_probe {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[ignore]
    fn attention_survives_a_prefill_that_would_otherwise_exhaust_the_card() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device ({e}) — skipping");
                return;
            }
        };
        // phi-3.5-mini's shape, at a prompt length a coding agent really sends.
        let (heads, head_dim, n) = (32usize, 96usize, 8192usize);
        let unblocked_bytes = heads * n * n * 4;
        eprintln!(
            "q_len={n} heads={heads}: one UNBLOCKED score temporary would be {:.2} GB; \
             blocking targets {:.0} MB",
            unblocked_bytes as f64 / 1e9,
            (ATTN_SCORE_BUDGET_ELEMS * 4) as f64 / 1e6
        );
        eprintln!(
            "chosen query block = {}",
            attention_query_block(n, n, heads)
        );

        let mk = |s: usize, seed: f32| {
            let cnt = heads * s * head_dim;
            let data: Vec<f32> = (0..cnt)
                .map(|i| ((i as f32 * 0.001 + seed).sin()) * 0.2)
                .collect();
            Tensor::from_vec(data, (1, heads, s, head_dim), &dev).unwrap()
        };
        let q = mk(n, 0.0);
        let k = mk(n, 1.0);
        let v = mk(n, 2.0);

        let out = standard_attention(&q, &k, &v, None, head_dim, heads, heads, None)
            .expect("attention must complete without exhausting device memory");
        assert_eq!(out.dims(), &[1, heads, n, head_dim]);
        // Force the result to be realised before we claim success.
        let probe = out.narrow(2, 0, 1).unwrap().flatten_all().unwrap();
        let v0 = probe.to_vec1::<f32>().unwrap()[0];
        assert!(v0.is_finite(), "output must be finite, got {v0}");
        eprintln!("OK — completed, first output element {v0:.6}");
    }
}

#[cfg(test)]
mod blocked_expert_ffn_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn w(o: usize, i: usize, seed: f32) -> Tensor {
        let data: Vec<f32> = (0..o * i).map(|k| (k as f32 * seed).sin() * 0.05).collect();
        Tensor::from_vec(data, (o, i), &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
        let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// A lopsided router sends most tokens to one expert, which reproduces the
    /// dense shape that exhausted a 6 GB card. Blocking must be exact.
    #[test]
    fn blocking_one_expert_is_bit_for_bit_identical() {
        let (hidden, intermediate, tokens) = (48, 160, 700);
        let (g, u, d) = (
            w(intermediate, hidden, 0.017),
            w(intermediate, hidden, 0.023),
            w(hidden, intermediate, 0.031),
        );
        let input = Tensor::from_vec(
            (0..tokens * hidden)
                .map(|k| (k as f32 * 0.011).cos() * 0.4)
                .collect::<Vec<f32>>(),
            (tokens, hidden),
            &Device::Cpu,
        )
        .unwrap();

        // The real budget leaves this unblocked; compare against an explicit
        // single-shot run of the same arithmetic.
        let blocked = expert_ffn(&input, &g, &u, &d).unwrap();
        let whole = {
            let gate_out = input.matmul(&g.t().unwrap()).unwrap();
            let up_out = input.matmul(&u.t().unwrap()).unwrap();
            let activated = candle_nn::ops::silu(&gate_out).unwrap();
            let combined = (activated * up_out).unwrap();
            combined.matmul(&d.t().unwrap()).unwrap()
        };

        assert_eq!(blocked.dims(), whole.dims());
        assert_eq!(max_abs_diff(&blocked, &whole), 0.0);
    }

    /// An expert given only a handful of tokens takes the unblocked path, so
    /// routing that spreads tokens thinly pays nothing for this.
    #[test]
    fn a_lightly_loaded_expert_is_not_blocked() {
        let (hidden, intermediate, tokens) = (32, 128, 8);
        let (g, u, d) = (
            w(intermediate, hidden, 0.013),
            w(intermediate, hidden, 0.019),
            w(hidden, intermediate, 0.029),
        );
        let input = Tensor::from_vec(
            (0..tokens * hidden)
                .map(|k| k as f32 * 0.001)
                .collect::<Vec<f32>>(),
            (tokens, hidden),
            &Device::Cpu,
        )
        .unwrap();

        let out = expert_ffn(&input, &g, &u, &d).unwrap();
        assert_eq!(out.dims(), &[tokens, hidden]);
    }
}

#[cfg(test)]
mod blocked_mlp_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Build an `Mlp` from plain f32 tensors — `QMatMul::Tensor` skips
    /// quantisation so the comparison isolates the blocking, not rounding.
    fn mlp(hidden: usize, intermediate: usize, gated: bool, act: Activation) -> Mlp {
        let dev = Device::Cpu;
        let w = |o: usize, i: usize, seed: f32| {
            let n = o * i;
            let data: Vec<f32> = (0..n).map(|k| (k as f32 * seed).sin() * 0.05).collect();
            let t = Tensor::from_vec(data, (o, i), &dev).unwrap();
            QMatMul {
                inner: QMatMulInner::Standard(candle_core::quantized::QMatMul::Tensor(t)),
            }
        };
        Mlp {
            ffn_gate: gated.then(|| w(intermediate, hidden, 0.31)),
            ffn_up: w(intermediate, hidden, 0.17),
            ffn_down: w(hidden, intermediate, 0.23),
            activation: act,
        }
    }

    fn input(batch: usize, tokens: usize, hidden: usize) -> Tensor {
        let n = batch * tokens * hidden;
        let data: Vec<f32> = (0..n).map(|k| (k as f32 * 0.013).cos() * 0.5).collect();
        Tensor::from_vec(data, (batch, tokens, hidden), &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
        let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn blocking_the_token_axis_is_bit_for_bit_identical() {
        let (hidden, intermediate, tokens) = (64, 176, 600);
        let m = mlp(hidden, intermediate, true, Activation::SiLU);
        let xs = input(1, tokens, hidden);

        let whole = m.forward_with_budget(&xs, None, usize::MAX).unwrap();
        // 176-wide intermediate under a 8800-element budget => 50-token blocks.
        let budget = 8_800;
        assert!(
            budget / intermediate < tokens,
            "test must actually exercise the blocked path"
        );
        let blocked = m.forward_with_budget(&xs, None, budget).unwrap();

        assert_eq!(whole.dims(), blocked.dims());
        assert_eq!(
            max_abs_diff(&whole, &blocked),
            0.0,
            "token blocking must be exact — every op in the MLP is pointwise across tokens"
        );
    }

    #[test]
    fn blocking_is_exact_for_a_gateless_gelu_mlp() {
        // Starcoder2-style: no gate, so the activation runs on `up` directly.
        let (hidden, intermediate, tokens) = (48, 160, 550);
        let m = mlp(hidden, intermediate, false, Activation::Gelu);
        let xs = input(1, tokens, hidden);

        let whole = m.forward_with_budget(&xs, None, usize::MAX).unwrap();
        let blocked = m.forward_with_budget(&xs, None, 6_400).unwrap();

        assert_eq!(max_abs_diff(&whole, &blocked), 0.0);
    }

    #[test]
    fn blocking_is_exact_with_a_batch_dimension() {
        // The token axis is dim rank-2, so a batch must not be split or folded.
        let (hidden, intermediate, tokens) = (32, 128, 520);
        let m = mlp(hidden, intermediate, true, Activation::SiLU);
        let xs = input(3, tokens, hidden);

        let whole = m.forward_with_budget(&xs, None, usize::MAX).unwrap();
        let blocked = m.forward_with_budget(&xs, None, 5_120).unwrap();

        assert_eq!(whole.dims(), blocked.dims());
        assert_eq!(max_abs_diff(&whole, &blocked), 0.0);
    }

    #[test]
    fn a_decode_step_is_never_blocked_or_probed() {
        // Decode passes one token and is latency-critical: it must take the
        // unblocked path even with an absurdly small budget.
        let m = mlp(32, 128, true, Activation::SiLU);
        let xs = input(1, 1, 32);

        let out = m.forward_with_budget(&xs, None, 1).unwrap();
        let reference = m.forward_block(&xs, None).unwrap();

        assert_eq!(max_abs_diff(&out, &reference), 0.0);
    }

    #[test]
    fn a_short_prefill_takes_the_unblocked_path() {
        let tokens = MLP_MIN_TOKENS_TO_BLOCK - 1;
        let m = mlp(32, 128, true, Activation::SiLU);
        let xs = input(1, tokens, 32);

        let out = m.forward_with_budget(&xs, None, 1).unwrap();
        let reference = m.forward_block(&xs, None).unwrap();

        assert_eq!(max_abs_diff(&out, &reference), 0.0);
    }
}

/// GPU-only causal check for the feed-forward memory fix, mirroring
/// [`cuda_attention_memory_probe`].
///
/// Run with `cargo test --release --no-default-features
/// --features candle-cuda -- --ignored cuda_mlp`.
///
/// The A/B that makes it evidence rather than a smoke test: with blocking on it
/// completes; raising `MLP_INTERMEDIATE_BUDGET_ELEMS` so no split happens brings
/// back `CUDA_ERROR_OUT_OF_MEMORY`. One constant apart. Re-run both halves if
/// this code moves.
#[cfg(test)]
mod cuda_mlp_memory_probe {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[ignore]
    fn feed_forward_survives_a_prefill_that_would_otherwise_exhaust_the_card() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device ({e}) — skipping");
                return;
            }
        };
        // Gemma-2-2b's shape at the prompt length the tester actually sent
        // (~5000 tokens from a coding agent's system prompt).
        let (hidden, intermediate, tokens) = (2304usize, 9216usize, 5021usize);
        let one_temporary = tokens * intermediate * 4;
        eprintln!(
            "tokens={tokens} intermediate={intermediate}: ONE unblocked temporary is {:.0} MB, \
             and up/gate/product are live together; blocking targets {:.0} MB",
            one_temporary as f64 / 1e6,
            (MLP_INTERMEDIATE_BUDGET_ELEMS * 4) as f64 / 1e6
        );
        eprintln!(
            "chosen token block = {}",
            (MLP_INTERMEDIATE_BUDGET_ELEMS / intermediate).clamp(1, tokens)
        );

        let w = |o: usize, i: usize, seed: f32| {
            let data: Vec<f32> = (0..o * i).map(|k| (k as f32 * seed).sin() * 0.02).collect();
            let t = Tensor::from_vec(data, (o, i), &dev).unwrap();
            QMatMul {
                inner: QMatMulInner::Standard(candle_core::quantized::QMatMul::Tensor(t)),
            }
        };
        let mlp = Mlp {
            ffn_gate: Some(w(intermediate, hidden, 0.0007)),
            ffn_up: w(intermediate, hidden, 0.0011),
            ffn_down: w(hidden, intermediate, 0.0013),
            activation: Activation::Gelu,
        };

        let xs = Tensor::from_vec(
            (0..tokens * hidden)
                .map(|k| (k as f32 * 0.0003).cos() * 0.5)
                .collect::<Vec<f32>>(),
            (1, tokens, hidden),
            &dev,
        )
        .unwrap();

        // Blocked vs unblocked IN THE SAME PROCESS. On CPU these are bit-for-bit
        // identical (see `blocked_mlp_tests`). On GPU they need not be: cuBLAS
        // selects a kernel from the problem shape, and changing the row count
        // changes that choice, which changes the order the K dimension is
        // accumulated in. The arithmetic each row receives is the same; the
        // rounding of it is not guaranteed to be. Measure the gap rather than
        // assuming either way.
        let unblocked = mlp
            .forward_with_budget(&xs, None, usize::MAX)
            .expect("unblocked reference must run");
        let out = mlp
            .forward(&xs, None)
            .expect("feed-forward must complete without exhausting device memory");
        let diff = (&out - &unblocked)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let scale = unblocked
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!(
            "blocked vs unblocked on GPU: max abs diff {diff:.3e} against max |value| {scale:.3e}              (relative {:.3e})",
            diff / scale.max(f32::MIN_POSITIVE)
        );
        assert!(
            diff / scale.max(f32::MIN_POSITIVE) < 1e-3,
            "blocking must not change the result beyond float reassociation, got {diff:.3e}"
        );
        assert_eq!(out.dims(), &[1, tokens, hidden]);
        // Force realisation before claiming success.
        let probe = out.narrow(1, 0, 1).unwrap().flatten_all().unwrap();
        let v0 = probe.to_vec1::<f32>().unwrap()[0];
        assert!(v0.is_finite(), "output must be finite, got {v0}");
        eprintln!("OK — completed, first output element {v0:.6}");
    }
}

/// The CUDA attention routing rule, pinned. Runs anywhere — no GPU needed.
#[cfg(test)]
mod cuda_attention_routing {
    use super::*;

    // phi-3.5-mini: MHA, 32 query heads and 32 KV heads.
    const MHA: (usize, usize) = (32, 32);
    // llama-3.2-3b: GQA, 24 query heads over 8 KV heads.
    const GQA: (usize, usize) = (24, 8);

    #[test]
    fn mha_decode_always_takes_standard() {
        // Measured 4x-25x faster than flash at EVERY KV length, because
        // candle-flash-attn has no split-KV kernel and one query row cannot
        // fill the card. There is no crossover to look for here.
        for kv in [128, 512, 1024, 2048, 4096, 8192, 32768] {
            assert!(
                cuda_decode_prefers_standard(1, kv, MHA.0, MHA.1),
                "MHA decode at kv={kv} must take standard"
            );
        }
    }

    #[test]
    fn gqa_decode_switches_to_flash_at_the_measured_crossover() {
        // Below the crossover standard still wins (0.66x at kv=512)...
        for kv in [1, 128, 512, GQA_FLASH_DECODE_MIN_KV - 1] {
            assert!(
                cuda_decode_prefers_standard(1, kv, GQA.0, GQA.1),
                "GQA decode at kv={kv} is below the crossover and must take standard"
            );
        }
        // ...and above it flash wins, by more as context grows, because
        // standard's repeat_kv expansion is rebuilt every token.
        for kv in [GQA_FLASH_DECODE_MIN_KV, 2048, 4096, 8192, 32768] {
            assert!(
                !cuda_decode_prefers_standard(1, kv, GQA.0, GQA.1),
                "GQA decode at kv={kv} is at/above the crossover and must take flash"
            );
        }
    }

    #[test]
    fn prefill_is_never_routed_to_standard_by_this_rule() {
        // Flash won every prefill shape measured, 2.4x-7.8x, for both
        // attention layouts. The offset-causal-mask fallback is a SEPARATE
        // condition in the caller and is not this function's business.
        for (n_head, n_kv_head) in [MHA, GQA] {
            for q in [2, 128, 512, 1536, 4096] {
                assert!(
                    !cuda_decode_prefers_standard(q, q, n_head, n_kv_head),
                    "prefill q={q} must not be sent to standard by shape"
                );
            }
        }
    }

    #[test]
    fn the_crossover_is_a_gqa_only_concept() {
        // The whole reason GQA crosses over is repeat_kv materialisation,
        // which does not exist when n_head == n_kv_head. If a refactor ever
        // makes MHA follow the same branch as GQA, this fails.
        let kv = GQA_FLASH_DECODE_MIN_KV * 4;
        assert!(cuda_decode_prefers_standard(1, kv, 32, 32), "MHA");
        assert!(!cuda_decode_prefers_standard(1, kv, 32, 8), "GQA 4:1");
        assert!(!cuda_decode_prefers_standard(1, kv, 32, 1), "MQA 32:1");
    }
}

/// CUDA-only A/B of the two attention kernels, used to price flash-attention-2.
///
/// Ignored by default — needs a real GPU and the `flash-attn` feature. Run with:
/// ```text
/// CUDA_COMPUTE_CAP=86 cargo test --release \
///   --no-default-features --features dev,claude-subscription,flash-attn \
///   flash_vs_standard -- --ignored --nocapture
/// ```
///
/// **Why a microbenchmark and not an end-to-end run.** Two separately-built
/// binaries differ in more than the kernel, so an end-to-end delta between them
/// is not attributable to attention (diagnosis rule 4: prove the mechanism
/// fired). Here the same process runs both branches over identical tensors, and
/// the only difference is `ForceStandardAttnGuard` — which is exactly the switch
/// `SWARMLLM_FORCE_STANDARD_ATTN=1` exposes for whole-daemon measurement.
///
/// It also keeps the measurement off the user's desktop-driving GPU at any real
/// scale: these shapes allocate tens of MB, not gigabytes (gotcha #251).
///
/// `Device::synchronize` around each timed region is not optional. CUDA launches
/// are asynchronous, so timing without it measures how fast the queue accepts
/// work — which would show any kernel as instantaneous.
#[cfg(all(test, feature = "flash-attn"))]
mod flash_vs_standard {
    use super::*;
    use crate::inference::attn_kernel::ForceStandardAttnGuard;
    use candle_core::{Device, Tensor};

    /// Upper-triangular causal mask, `[q_len, k_len]`, u8, 1 = masked.
    ///
    /// **Not optional, and getting it wrong invalidates everything.** The flash
    /// branch passes `causal = q_len > 1` to the kernel, which applies the mask
    /// internally; `standard_attention` applies only the mask it is handed. Run
    /// the comparison with `mask: None` and the two sides compute *different
    /// functions* — flash does causal attention, standard does full attention —
    /// so the timings compare unequal work and the outputs disagree entirely.
    /// The first version of this benchmark did exactly that; the numerics
    /// assertion at the end is what caught it (relative diff 3.1, against an
    /// F16 rounding budget of 0.05).
    ///
    /// Aligned q/k only: position i attends to keys 0..=i.
    fn causal_mask(q_len: usize, k_len: usize, dev: &Device) -> Tensor {
        assert_eq!(q_len, k_len, "this mask assumes aligned q/k");
        let data: Vec<f32> = (0..q_len)
            .flat_map(|i| (0..k_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(data, (q_len, k_len), dev).unwrap()
    }

    /// Fastest of `iters` runs, after `warmup` untimed runs.
    ///
    /// **Min, not median or mean** — the project's standing rule for this
    /// machine (CLAUDE.md: "use min-of-N on an idle machine"). Every source of
    /// error here is additive: a scheduler preemption, the desktop compositor
    /// taking the GPU, a first-touch allocation. None of them can make a kernel
    /// run faster than it can run, so the minimum is the least contaminated
    /// estimate of the thing being compared.
    ///
    /// A median at 9 samples was NOT enough: the same llama prefill shape
    /// measured 3.08 ms and 191 ms in two consecutive runs of this benchmark,
    /// which would have been reported as a 62x difference in the kernel rather
    /// than as the sampling artefact it was.
    fn time_ms(warmup: usize, iters: usize, dev: &Device, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        dev.synchronize().unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = std::time::Instant::now();
            f();
            dev.synchronize().unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        best
    }

    #[test]
    #[ignore]
    fn flash_vs_standard_attention_on_cuda() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no CUDA device ({e}) — skipping");
                return;
            }
        };

        // (label, n_head, n_kv_head, head_dim) — the two models the README
        // benchmarks quote. Phi-3.5 is MHA, Llama-3.2 is 3:1 GQA, and the
        // repeat_kv cost that dominates the CPU side only exists for GQA, so
        // both shapes are needed before generalising.
        let models: &[(&str, usize, usize, usize)] = &[
            ("phi-3.5-mini  MHA 32/32 d96", 32, 32, 96),
            ("llama-3.2-3b  GQA 24/8  d128", 24, 8, 128),
        ];
        // (label, q_len, kv_len). Prefill sends the whole prompt in one
        // forward on a pipeline segment; decode is one query against a cache
        // that grows with the conversation.
        let shapes: &[(&str, usize, usize)] = &[
            ("prefill  q=512  kv=512", 512, 512),
            ("prefill  q=1536 kv=1536", 1536, 1536),
            ("decode   q=1    kv=512", 1, 512),
            ("decode   q=1    kv=1024", 1, 1024),
            ("decode   q=1    kv=2048", 1, 2048),
            ("decode   q=1    kv=3072", 1, 3072),
            ("decode   q=1    kv=4096", 1, 4096),
            ("decode   q=1    kv=8192", 1, 8192),
        ];

        // The right column is the SHIPPED dispatch, not "flash": since the
        // routing rule landed, `run_attention` sends MHA decode and short-KV
        // GQA decode to standard on purpose. A row at ~1.00x therefore means
        // the router chose standard there, which is the intended outcome, not
        // a null result. To re-derive the crossover itself, widen
        // `GQA_FLASH_DECODE_MIN_KV` temporarily so flash is taken everywhere.
        println!(
            "\n{:<30} {:<24} {:>10} {:>10} {:>9}",
            "model", "shape", "standard", "dispatch", "speedup"
        );
        println!("{}", "-".repeat(88));

        for (mlabel, n_head, n_kv_head, head_dim) in models {
            for (slabel, q_len, kv_len) in shapes {
                let q = Tensor::randn(0f32, 1.0, (1, *n_head, *q_len, *head_dim), &dev).unwrap();
                let k =
                    Tensor::randn(0f32, 1.0, (1, *n_kv_head, *kv_len, *head_dim), &dev).unwrap();
                let v =
                    Tensor::randn(0f32, 1.0, (1, *n_kv_head, *kv_len, *head_dim), &dev).unwrap();

                // The dispatch only reaches flash when q_len == kv_len or
                // q_len == 1 — an offset causal mask (a warm prefix cache)
                // cannot be expressed through flash_attn's boolean causal flag
                // and falls back by design. Measuring those shapes would
                // compare standard against itself and report a meaningless 1.0x.
                assert!(
                    q_len == kv_len || *q_len == 1,
                    "{slabel} would fall back to standard on both sides"
                );

                // Decode (q_len == 1) needs no mask and flash is told
                // `causal = false`, so both sides already agree there.
                let mask = (*q_len > 1).then(|| causal_mask(*q_len, *kv_len, &dev));
                let run = |force_standard: bool| {
                    let _g = ForceStandardAttnGuard::new(force_standard);
                    let out = run_attention(
                        &q,
                        &k,
                        &v,
                        mask.as_ref(),
                        *n_head,
                        *n_kv_head,
                        *head_dim,
                        None,
                    )
                    .unwrap();
                    // Force realisation — candle is lazy enough that dropping
                    // an unread result can skip work.
                    out.narrow(2, 0, 1).unwrap().sum_all().unwrap();
                };

                let std_ms = time_ms(5, 20, &dev, || run(true));
                let dispatch_ms = time_ms(5, 20, &dev, || run(false));

                println!(
                    "{mlabel:<30} {slabel:<24} {std_ms:>9.3}ms {dispatch_ms:>9.3}ms {:>8.2}x",
                    std_ms / dispatch_ms
                );

                // The routing rule, checked against the machine rather than
                // against the comment describing it. Whatever the dispatch
                // picks must not be materially worse than simply always using
                // standard — otherwise the kernel choice is costing time, which
                // is exactly the regression that shipping flash unconditionally
                // would have caused (25x on MHA decode).
                //
                // 1.35x of headroom: these are sub-millisecond calls at the
                // small shapes, where fixed launch overhead is a large fraction
                // and run-to-run spread is real even at min-of-20.
                assert!(
                    dispatch_ms < std_ms * 1.35,
                    "{mlabel} / {slabel}: the dispatch chose a kernel {:.2}x SLOWER than \
                     standard ({dispatch_ms:.3} ms vs {std_ms:.3} ms). Check \
                     `cuda_decode_prefers_standard` against the measured table above it.",
                    dispatch_ms / std_ms
                );
            }
        }

        // Numerics, not just speed: flash runs in F16 while standard runs in
        // F32, so the two must be checked to agree before any speed figure
        // means anything. A fast wrong answer is not an optimisation.
        let (n_head, n_kv_head, head_dim, q_len) = (24usize, 8usize, 128usize, 256usize);
        let q = Tensor::randn(0f32, 1.0, (1, n_head, q_len, head_dim), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, n_kv_head, q_len, head_dim), &dev).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, n_kv_head, q_len, head_dim), &dev).unwrap();
        // Causal on BOTH sides — see `causal_mask`. Flash applies it from the
        // `causal` flag, standard only from this tensor.
        let m = causal_mask(q_len, q_len, &dev);
        let mask = Some(&m);
        let std_out = {
            let _g = ForceStandardAttnGuard::new(true);
            run_attention(&q, &k, &v, mask, n_head, n_kv_head, head_dim, None).unwrap()
        };
        let flash_out = run_attention(&q, &k, &v, mask, n_head, n_kv_head, head_dim, None).unwrap();
        assert_eq!(std_out.dims(), flash_out.dims(), "layout must match");

        let diff = (&std_out - &flash_out)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let scale = std_out
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let rel = diff / scale.max(f32::MIN_POSITIVE);
        println!(
            "\nnumerics: max abs diff {diff:.3e} vs max |value| {scale:.3e} (relative {rel:.3e})"
        );
        // F16 carries ~3 decimal digits, and the flash path casts to it, so
        // this cannot be tightened to the 1e-6 an F32-vs-F32 check would use.
        assert!(
            rel < 5e-2,
            "flash and standard attention disagree by {rel:.3e} — beyond F16 rounding"
        );

        // Gemma-2 softcaps its attention logits at 50.0. `candle_flash_attn`'s
        // plain entry point silently drops the cap, so before this was routed
        // to the windowed+softcap entry point the GPU produced a different
        // distribution from the CPU for that model — no error, just worse
        // answers on one device. A cap small enough to bite (2.0) is used here
        // so that dropping it would visibly change the result: at Gemma's own
        // 50.0 most logits are already below the cap and the bug hides.
        let softcap = 2.0f32;
        let capped_std = {
            let _g = ForceStandardAttnGuard::new(true);
            run_attention(&q, &k, &v, mask, n_head, n_kv_head, head_dim, Some(softcap)).unwrap()
        };
        let capped_flash =
            run_attention(&q, &k, &v, mask, n_head, n_kv_head, head_dim, Some(softcap)).unwrap();
        let cap_diff = (&capped_std - &capped_flash)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let cap_scale = capped_std
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let cap_rel = cap_diff / cap_scale.max(f32::MIN_POSITIVE);
        println!("softcap={softcap}: relative diff {cap_rel:.3e}");
        assert!(
            cap_rel < 5e-2,
            "flash ignores attn_logit_softcap — relative diff {cap_rel:.3e}. This is the \
             Gemma-2 correctness bug: flash_attn() hardcodes softcap: None, so the cap must \
             go through flash_attn_alibi_windowed_softcap instead."
        );

        // And the guard against a vacuous check: the cap must actually change
        // the standard-path result at this magnitude, or the assertion above
        // would pass whether or not flash honoured it (diagnosis rule 5).
        //
        // Compared ELEMENT-WISE against the uncapped output, not by max
        // |value|. The first version of this guard used the latter and fired
        // with an effect of exactly 0: attention output is a convex
        // combination of V rows (softmax weights sum to 1), so its magnitude
        // is bounded by max |V| whether or not the logits were capped. The cap
        // changes which rows get weight, not how big the answer can be.
        let cap_effect = (&capped_std - &std_out)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            / cap_scale.max(f32::MIN_POSITIVE);
        assert!(
            cap_effect > 1e-2,
            "softcap={softcap} barely changed the standard result ({cap_effect:.3e}), so the \
             check above proves nothing — raise the logit magnitude or lower the cap"
        );
        println!("softcap changed the standard result by {cap_effect:.3e} (guard: check is live)");
    }
}
