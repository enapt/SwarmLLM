//! Qwen 3.5 layers, written op by op against llama.cpp's `src/models/qwen35.cpp`
//! and `delta-net-base.cpp` (master `4b1a27f`) — the plan and its reasoning are
//! in `docs/plans/qwen35_support.md` (FUTURE_WORK #117).

use crate::inference::split::kv_cache::LayerKv;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::Module;

use super::{run_attention, DeltaNetWeights, Qwen35AttnWeights, SsmState};

// ── Qwen 3.5 full-attention layer forward ──

impl Qwen35AttnWeights {
    pub(crate) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, _head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        // IMROPE with every position component equal to the token index —
        // what text is — rotates each pair by `pos * base^(-2i/n_rot)`, i.e.
        // plain NEOX-style partial RoPE over the first `rope_dim` dims.
        super::rope_over_heads(x, &cos, &sin, self.rope_dim, true)
    }

    pub(crate) fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<LayerKv>,
        max_seq_len: usize,
        kv_reserve: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let hd = self.head_dim;

        // `attn_q` answers [q | gate] per head, interleaved head by head
        // (llama.cpp views the even and odd head_dim halves).
        let q_gate = self
            .wq_gate
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_head, 2 * hd))?;
        let q = q_gate.narrow(3, 0, hd)?.contiguous()?;
        let gate =
            q_gate
                .narrow(3, hd, hd)?
                .contiguous()?
                .reshape((b_sz, seq_len, self.n_head * hd))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_kv_head, hd))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_kv_head, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head RMSNorm with weights, BEFORE RoPE.
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?.contiguous()?;
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match kv_cache {
            None => {
                let mut cache = super::new_kv_cache(
                    max_seq_len,
                    super::model_wants_kv_mirror(self.n_head, self.n_kv_head),
                    kv_reserve,
                );
                let kv = cache.append(&k, &v)?;
                *kv_cache = Some(cache);
                kv
            }
            Some(cache) => {
                if index_pos == 0 {
                    cache.reset();
                }
                cache.set_mirror_wanted(super::model_wants_kv_mirror(self.n_head, self.n_kv_head));
                cache.append(&k, &v)?
            }
        };

        let mirror = kv_cache.as_ref().and_then(|c| c.flash_operands());
        let y = run_attention(
            &q,
            &k,
            &v,
            mask,
            self.n_head,
            self.n_kv_head,
            hd,
            None,
            mirror.as_ref().map(|(k, v)| (k, v)),
        )?;
        let y = y
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_head * hd))?;
        // Output gated by sigmoid(gate), then `attn_output`.
        let gated = (y * candle_nn::ops::sigmoid(&gate)?)?;
        self.wo.forward(&gated)
    }
}

// ── Qwen 3.5 Gated DeltaNet ("linear attention") layer forward ──

impl DeltaNetWeights {
    fn key_dim(&self) -> usize {
        self.n_k_heads * self.k_head_dim
    }

    fn value_dim(&self) -> usize {
        self.n_v_heads * self.v_head_dim
    }

    /// `build_layer_attn_linear`: projections, causal conv, L2-normalised
    /// q/k, the gated delta rule, gated RMSNorm, `ssm_out`.
    pub(crate) fn forward_deltanet(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let device = x.device();
        let (nk, kd, nv, vd) = (
            self.n_k_heads,
            self.k_head_dim,
            self.n_v_heads,
            self.v_head_dim,
        );

        let mixed = self.wqkv.forward(x)?; // [b, seq, 2·nk·kd + nv·vd]
        let z = self.wz.forward(x)?; // [b, seq, nv·vd]
        let beta = candle_nn::ops::sigmoid(&self.w_beta.forward(x)?)?; // [b, seq, nv]
                                                                       // g = softplus(alpha + dt_bias) · a, with a = -exp(A_log): a log-decay ≤ 0.
        let alpha = self.w_alpha.forward(x)?.broadcast_add(&self.dt_bias)?;
        let g = softplus(&alpha)?.broadcast_mul(&self.a)?; // [b, seq, nv]

        let (conv, conv_state) = self.causal_conv(&mixed, ssm_state.as_ref(), device)?;
        let conv = candle_nn::ops::silu(&conv)?; // [b, seq, C]

        let q = conv
            .narrow(2, 0, self.key_dim())?
            .reshape((b_sz, seq_len, nk, kd))?;
        let k = conv
            .narrow(2, self.key_dim(), self.key_dim())?
            .reshape((b_sz, seq_len, nk, kd))?;
        let v = conv
            .narrow(2, 2 * self.key_dim(), self.value_dim())?
            .reshape((b_sz, seq_len, nv, vd))?;
        let q = l2_norm(&q, self.eps)?;
        let k = l2_norm(&k, self.eps)?;
        // More value heads than key heads: llama.cpp TILES the key heads
        // (`ggml_repeat_4d`), value head h reading key head h % nk.
        let (q, k) = if nv > nk {
            let r = nv / nk;
            let tile = |t: &Tensor| Tensor::cat(&vec![t; r], 2);
            (tile(&q)?, tile(&k)?)
        } else {
            (q, k)
        };
        let q = (q * (1.0 / (kd as f64).sqrt()))?;

        let state = match ssm_state.as_ref() {
            Some(s) => s.recurrent_state.clone(),
            None => Tensor::zeros((b_sz, nv, vd, kd), DType::F32, device)?,
        };
        let (out, state) = delta_rule(&q, &k, &v, &g, &beta, state)?; // out [b, seq, nv, vd]

        // Gated RMSNorm per value head: RMSNorm(o, ssm_norm) · silu(z).
        let z = z.reshape((b_sz, seq_len, nv, vd))?;
        let normed = self.ssm_norm.forward(&out)?;
        let gated = (normed * candle_nn::ops::silu(&z)?)?;
        let gated = gated.reshape((b_sz, seq_len, nv * vd))?;

        *ssm_state = Some(SsmState {
            conv_state,
            recurrent_state: state,
        });
        self.ssm_out.forward(&gated)
    }

    /// Causal depthwise convolution with the previous `kernel − 1` inputs as
    /// state (`ggml_ssm_conv`): output t reads inputs t−(K−1) … t, tap j
    /// weighing input t−(K−1)+j. Returns `[b, seq, C]` and the new state
    /// `[b, C, K−1]` — the last K−1 inputs, carried to the next call.
    pub(crate) fn causal_conv(
        &self,
        x: &Tensor,
        ssm_state: Option<&SsmState>,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (b_sz, seq_len, channels) = x.dims3()?;
        let k = self.conv_kernel;
        let pad = k - 1;
        let prev = match ssm_state {
            Some(s) => s.conv_state.clone(),
            None => Tensor::zeros((b_sz, channels, pad), x.dtype(), device)?,
        };
        let x_t = x.transpose(1, 2)?.contiguous()?; // [b, C, seq]
        let padded = Tensor::cat(&[&prev, &x_t], 2)?; // [b, C, seq + K − 1]
        let w = self.conv1d.reshape((1, channels, k))?;
        let mut acc: Option<Tensor> = None;
        for j in 0..k {
            let tap = padded
                .narrow(2, j, seq_len)?
                .broadcast_mul(&w.narrow(2, j, 1)?)?;
            acc = Some(match acc {
                None => tap,
                Some(a) => (a + tap)?,
            });
        }
        let out = acc
            .expect("kernel has at least one tap")
            .transpose(1, 2)?
            .contiguous()?;
        let new_state = padded.narrow(2, seq_len, pad)?.contiguous()?;
        Ok((out, new_state))
    }
}

/// The gated delta rule, token by token (`build_delta_net_autoregressive`),
/// per value head with state `S` `[v_dim, k_dim]`:
///
/// ```text
/// S = S · exp(g)          e = (v − S·k) · beta
/// S = S + e ⊗ k           o = S · q
/// ```
///
/// q/k/v `[b, seq, heads, dim]`, g/beta `[b, seq, heads]`, state
/// `[b, heads, v_dim, k_dim]`. Returns `[b, seq, heads, v_dim]` and the state.
fn delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    mut state: Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    let seq_len = q.dim(1)?;
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(3)?; // [b, h, kd, 1]
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?; // [b, h, kd]
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?; // [b, h, vd]
        let decay = g
            .narrow(1, t, 1)?
            .squeeze(1)?
            .exp()?
            .unsqueeze(2)?
            .unsqueeze(3)?; // [b, h, 1, 1]
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(2)?; // [b, h, 1]
        state = state.broadcast_mul(&decay)?;
        let sk = state.matmul(&k_t.unsqueeze(3)?)?.squeeze(3)?; // [b, h, vd]
        let e = (v_t - sk)?.broadcast_mul(&beta_t)?; // [b, h, vd]
        state = (state + e.unsqueeze(3)?.matmul(&k_t.unsqueeze(2)?)?)?;
        outs.push(state.matmul(&q_t)?.squeeze(3)?); // [b, h, vd]
    }
    Ok((Tensor::stack(&outs, 1)?, state))
}

/// `build_gdn_l2_norm`: `x / sqrt(Σx² + eps)` over the last dimension
/// (llama.cpp spells it `rms_norm(x, eps/n) / sqrt(n)`).
fn l2_norm(x: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let sum_sq = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_div(&(sum_sq + eps)?.sqrt()?)
}

/// Softplus, `log(1 + exp(x))`, in the form that cannot overflow:
/// `max(x, 0) + log(1 + exp(−|x|))` (ggml switches to `x` above 20).
fn softplus(x: &Tensor) -> CandleResult<Tensor> {
    x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(device: &Device) -> DeltaNetWeights {
        let (hidden, nk, kd, nv, vd, k) = (8usize, 2usize, 4usize, 4usize, 4usize, 4usize);
        let c = 2 * nk * kd + nv * vd;
        let mm = |o: usize, i: usize| {
            super::super::QMatMul::from_dense(Tensor::randn(0f32, 0.3, (o, i), device).unwrap())
        };
        DeltaNetWeights {
            wqkv: mm(c, hidden),
            wz: mm(nv * vd, hidden),
            w_beta: mm(nv, hidden),
            w_alpha: mm(nv, hidden),
            dt_bias: Tensor::randn(0f32, 0.3, (nv,), device).unwrap(),
            a: Tensor::randn(0f32, 0.3, (nv,), device)
                .unwrap()
                .abs()
                .unwrap()
                .neg()
                .unwrap(),
            conv1d: Tensor::randn(0f32, 0.3, (c, k), device).unwrap(),
            ssm_norm: crate::inference::residual_norm::RmsNorm::from_qtensor(
                candle_core::quantized::QTensor::quantize(
                    &Tensor::ones((vd,), DType::F32, device).unwrap(),
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap(),
                1e-6,
            )
            .unwrap(),
            ssm_out: mm(hidden, nv * vd),
            n_k_heads: nk,
            k_head_dim: kd,
            n_v_heads: nv,
            v_head_dim: vd,
            conv_kernel: k,
            eps: 1e-6,
        }
    }

    /// A prompt run in one pass and the same prompt run as a prefix pass plus
    /// single-token steps must give the same outputs: the decode path carries
    /// the convolution's last `kernel − 1` inputs and the delta rule's state.
    /// The first version fed the current token twice and dropped the oldest
    /// tap on every decode step, which no prefill-only check could see.
    #[test]
    fn a_deltanet_decoded_step_by_step_matches_one_pass() {
        let device = Device::Cpu;
        let w = weights(&device);
        let x = Tensor::randn(0f32, 1.0, (1, 7, 8), &device).unwrap();
        let mut whole_state = None;
        let whole = w.forward_deltanet(&x, &mut whole_state).unwrap();

        let mut state = None;
        let mut parts = vec![w
            .forward_deltanet(&x.narrow(1, 0, 3).unwrap(), &mut state)
            .unwrap()];
        for t in 3..7 {
            parts.push(
                w.forward_deltanet(&x.narrow(1, t, 1).unwrap(), &mut state)
                    .unwrap(),
            );
        }
        let stepped = Tensor::cat(&parts, 1).unwrap();
        let diff: f32 = (whole - stepped)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-5, "stepped decode drifts from one pass by {diff}");
    }
}
