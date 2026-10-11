//! Qwen 3.5 layers, written op by op against llama.cpp's `src/models/qwen35.cpp`
//! and `delta-net-base.cpp` (master `4b1a27f`) — the plan and its reasoning are
//! in `docs/plans/qwen35_support.md` (FUTURE_WORK #117).

use crate::inference::decode_attn::{axpy, dot};
use crate::inference::split::kv_cache::LayerKv;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::Module;
use rayon::prelude::*;

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

        let (k, v) = super::append_to_cache(
            kv_cache,
            &k,
            &v,
            index_pos,
            super::kv_storage(
                x.device().is_cuda(),
                super::KvAttention::Standard,
                self.n_head,
                self.n_kv_head,
                hd,
            ),
            max_seq_len,
            kv_reserve,
        )?;

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

        // q and k side by side, `[b, seq, 2·nk, kd]` (heads 0..nk are q), so
        // one L2 norm serves both and the card kernel reads them as one input.
        let qk = conv
            .narrow(2, 0, 2 * self.key_dim())?
            .reshape((b_sz, seq_len, 2 * nk, kd))?;
        let qk = l2_norm(&qk, self.eps)?;
        let v = conv
            .narrow(2, 2 * self.key_dim(), self.value_dim())?
            .reshape((b_sz, seq_len, nv, vd))?;
        let scale = 1.0 / (kd as f64).sqrt();

        let state = match ssm_state.as_ref() {
            Some(s) => s.recurrent_state.clone(),
            None => Tensor::zeros((b_sz, nv, vd, kd), DType::F32, device)?,
        };
        let one_pass = delta_rule_in_one_pass();
        let (out, state) =
            crate::inference::prof::timed!(crate::inference::prof::Stage::DeltaRule, {
                let on_card = if one_pass && device.is_cuda() && card_delta_rule_covers(kd, vd) {
                    delta_rule_cuda(&qk, nk, &v, &g, &beta, &state, scale as f32)?
                } else {
                    None
                };
                let (q, k) = (qk.narrow(2, 0, nk)?, qk.narrow(2, nk, nk)?);
                match on_card {
                    Some(done) => done,
                    None if one_pass && device.is_cpu() => {
                        delta_rule_cpu(&q, &k, &v, &g, &beta, &state, scale as f32)?
                    }
                    None => {
                        // More value heads than key heads: llama.cpp TILES the key
                        // heads (`ggml_repeat_4d`), value head h reading key head h % nk.
                        let (q, k) = if nv > nk {
                            let r = nv / nk;
                            let tile = |t: &Tensor| Tensor::cat(&vec![t; r], 2);
                            (tile(&q)?, tile(&k)?)
                        } else {
                            (q, k)
                        };
                        delta_rule(&(q * scale)?, &k, &v, &g, &beta, state)?
                    }
                }
            }); // out [b, seq, nv, vd]

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

/// `SWARMLLM_DELTA_RULE=steps` → the token-by-token composition
/// ([`delta_rule`]) on the processor and the card alike: the A/B switch for
/// [`delta_rule_cpu`] and [`delta_rule_cuda`]. Read once.
fn delta_rule_in_one_pass() -> bool {
    static ONE_PASS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONE_PASS
        .get_or_init(|| !matches!(std::env::var("SWARMLLM_DELTA_RULE").as_deref(), Ok("steps")))
}

/// The gated delta rule on the processor, every token of a head in ONE task
/// with its state in one buffer — llama.cpp's `GGML_OP_GATED_DELTA_NET`
/// (`ggml_compute_forward_gated_delta_net`, read at `4b1a27f`), which master
/// runs for every prompt and decode step, instead of [`delta_rule`]'s ~10
/// tensor ops per token per layer (FUTURE_WORK #228 (3)).
///
/// A token's update of state row `S[j, :]` (one value dimension) reads only
/// row j, so each row is decayed, read against k, corrected and read against q
/// in one pass while it sits in L1 — llama.cpp's CPU op makes four passes
/// over the whole matrix for the same arithmetic. Value head h reads key head
/// h % nk (llama.cpp tiles the key heads) and the 1/√d scale multiplies the
/// output, as the op applies it.
///
/// q/k `[b, seq, nk, kd]` (L2-normalised, unscaled), v `[b, seq, nv, vd]`,
/// g/beta `[b, seq, nv]`, state `[b, nv, vd, kd]`; all f32. Returns
/// `[b, seq, nv, vd]` and the state. Not bit-identical to [`delta_rule`] (its
/// sums run in another order): `one_pass_delta_rule_matches_token_by_token`
/// bounds it.
fn delta_rule_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
) -> CandleResult<(Tensor, Tensor)> {
    let (b, seq, nk, kd) = q.dims4()?;
    let (_, _, nv, vd) = v.dims4()?;
    let flat = |t: &Tensor| t.contiguous()?.flatten_all()?.to_vec1::<f32>();
    let (q, k, v, g, beta) = (flat(q)?, flat(k)?, flat(v)?, flat(g)?, flat(beta)?);
    let mut s = flat(state)?; // [b, nv, vd, kd]
    let mut out = vec![0f32; b * nv * seq * vd]; // [b, nv, seq, vd]: one head's tokens contiguous
    s.par_chunks_mut(vd * kd)
        .zip(out.par_chunks_mut(seq * vd))
        .enumerate()
        .for_each(|(bh, (s, o))| {
            let (bi, h) = (bh / nv, bh % nv);
            for t in 0..seq {
                let qk = ((bi * seq + t) * nk + h % nk) * kd;
                let (q_t, k_t) = (&q[qk..qk + kd], &k[qk..qk + kd]);
                let gb = (bi * seq + t) * nv + h;
                let (decay, beta_t) = (g[gb].exp(), beta[gb]);
                let v_t = &v[gb * vd..(gb + 1) * vd];
                let o_t = &mut o[t * vd..(t + 1) * vd];
                for (j, row) in s.chunks_exact_mut(kd).enumerate() {
                    row.iter_mut().for_each(|x| *x *= decay);
                    let delta = (v_t[j] - dot(row, k_t)) * beta_t;
                    axpy(delta, k_t, row);
                    o_t[j] = dot(row, q_t) * scale;
                }
            }
        });
    let out = Tensor::from_vec(out, (b, nv, seq, vd), &Device::Cpu)?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((out, Tensor::from_vec(s, (b, nv, vd, kd), &Device::Cpu)?))
}

/// Does `kernels/delta_rule.cu` take this layer? A lane holds `D / 32` of a
/// state row in registers, with equal key and value widths, D ∈ {64, 128}
/// (every Qwen 3.5 size has 128).
fn card_delta_rule_covers(kd: usize, vd: usize) -> bool {
    kd == vd && matches!(kd, 64 | 128)
}

/// Up to this many tokens the new state stays a view into the card kernel's
/// output buffer, which then lives until the next forward replaces the state.
/// Past it the buffer holds megabytes of outputs per layer (a 512-token chunk
/// of the 4B: 8 MB, ×24 layers), so the state is copied out — one launch over
/// the whole chunk, where a decode step pays none. 8 is the most positions a
/// captured speculative check runs (`cuda_graph::MAX_POSITIONS`).
#[cfg(feature = "candle-cuda")]
const STATE_KEPT_AS_A_VIEW_MAX_TOKENS: usize = 8;

/// PTX for the card kernel, compiled from `kernels/delta_rule.cu` by
/// `build.rs` and loaded through candle's `get_or_load_custom_func`.
#[cfg(feature = "candle-cuda")]
const DELTA_RULE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/delta_rule.ptx"));

/// [`delta_rule_cpu`]'s twin on a card: every token of the layer in ONE launch
/// of `kernels/delta_rule.cu` (llama.cpp's `gated_delta_net_cuda`, a warp per
/// state row), where the composition makes ~10 per token.
///
/// `qk` `[b, seq, 2·nk, kd]` (q then k, L2-normalised, q unscaled), v
/// `[b, seq, nv, vd]`, g/beta `[b, seq, nv]`, state `[b, nv, vd, kd]`, all f32
/// on a card, a width [`card_delta_rule_covers`]. `None` off a card build or
/// outside that scope — the caller keeps the composition. Not bit-identical to
/// it; replies are judged against llama.cpp.
fn delta_rule_cuda(
    qk: &Tensor,
    nk: usize,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
) -> CandleResult<Option<(Tensor, Tensor)>> {
    #[cfg(feature = "candle-cuda")]
    {
        if [qk, v, g, beta, state]
            .iter()
            .any(|t| t.dtype() != DType::F32)
        {
            return Ok(None);
        }
        let (b, seq, nv, vd) = v.dims4()?;
        let gb = Tensor::stack(&[g, beta], 3)?; // [b, seq, nv, 2]
        let op = cuda::DeltaRule {
            state: state.contiguous()?,
            nk,
            scale,
        };
        let packed = qk.contiguous()?.apply_op3_no_bwd(v, &gb, &op)?;
        let n_out = b * seq * nv * vd;
        let out = packed.narrow(0, 0, n_out)?.reshape((b, seq, nv, vd))?;
        let new_state = packed
            .narrow(0, n_out, b * nv * vd * vd)?
            .reshape((b, nv, vd, vd))?;
        let new_state = if seq > STATE_KEPT_AS_A_VIEW_MAX_TOKENS {
            new_state.copy()?
        } else {
            new_state
        };
        Ok(Some((out, new_state)))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (qk, nk, v, g, beta, state, scale);
        Ok(None)
    }
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

    pub(super) struct DeltaRule {
        /// `[b, nv, vd, kd]`, contiguous: the fourth operand, which a
        /// three-operand op carries itself.
        pub(super) state: Tensor,
        pub(super) nk: usize,
        pub(super) scale: f32,
    }

    impl CustomOp3 for DeltaRule {
        fn name(&self) -> &'static str {
            "gated-delta-rule"
        }

        /// Never reached: [`super::delta_rule_cuda`] runs only on a card, and
        /// the processor has its own kernel.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("gated-delta-rule's kernel is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
            s2: &candle_core::CudaStorage,
            l2: &Layout,
            s3: &candle_core::CudaStorage,
            l3: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let (b, seq, two_nk, d) = l1.shape().dims4()?;
            let (vb, vseq, nv, vd) = l2.shape().dims4()?;
            let vs = l2.stride();
            if two_nk != 2 * self.nk
                || (vb, vseq, vd) != (b, seq, d)
                || vs[3] != 1
                || l3.shape().dims() != [b, seq, nv, 2]
                || self.state.dims() != [b, nv, d, d]
            {
                candle_core::bail!("gated-delta-rule: unexpected shapes");
            }
            let (Some((qo, qe)), Some((go, ge))) =
                (l1.contiguous_offsets(), l3.contiguous_offsets())
            else {
                candle_core::bail!("gated-delta-rule: qk and gb must be contiguous");
            };
            let (st_storage, st_layout) = self.state.storage_and_layout();
            let candle_core::Storage::Cuda(st) = &*st_storage else {
                candle_core::bail!("gated-delta-rule: the state is not on the card");
            };
            let Some((so, se)) = st_layout.contiguous_offsets() else {
                candle_core::bail!("gated-delta-rule: the state must be contiguous");
            };

            let qk = s1.as_cuda_slice::<f32>()?.slice(qo..qe);
            let v = s2.as_cuda_slice::<f32>()?.slice(l2.start_offset()..);
            let gb = s3.as_cuda_slice::<f32>()?.slice(go..ge);
            let state = st.as_cuda_slice::<f32>()?.slice(so..se);
            let (n_out, n_state) = (b * seq * nv * d, b * nv * d * d);
            // Every element is assigned: each (sequence, token, head, row)
            // output by its warp's lane 0, each state element by its lane.
            let mut dst = dev.alloc_fully_overwritten::<f32>(n_out + n_state)?;
            let func = dev.get_or_load_custom_func(
                if d == 64 {
                    "delta_rule_f32_d64"
                } else {
                    "delta_rule_f32_d128"
                },
                "swarmllm_delta_rule",
                super::DELTA_RULE_PTX,
            )?;
            let cfg = LaunchConfig {
                grid_dim: (nv as u32, b as u32, (d / 4) as u32),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let (nk_i, nv_i, seq_i) = (self.nk as i32, nv as i32, seq as i32);
            let (sv_b, sv_t, sv_h) = (vs[0] as i64, vs[1] as i64, vs[2] as i64);
            let mut builder = func.builder();
            builder.arg(&qk);
            builder.arg(&v);
            builder.arg(&gb);
            builder.arg(&state);
            builder.arg(&mut dst);
            builder.arg(&nk_i);
            builder.arg(&nv_i);
            builder.arg(&seq_i);
            builder.arg(&sv_b);
            builder.arg(&sv_t);
            builder.arg(&sv_h);
            builder.arg(&self.scale);
            // SAFETY: ffi. Shapes and contiguity are checked above and the
            // width by `card_delta_rule_covers` (d ∈ {64, 128}, the kernel's
            // two entry points); every read is within the views by those
            // shapes and v's strides, and `dst` is sized for exactly what the
            // kernel assigns.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev),
                Shape::from_dims(&[n_out + n_state]),
            ))
        }
    }
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

    /// The one-pass kernel against the composition it replaces, on shapes
    /// that catch the index arithmetic: more value heads than key heads (the
    /// tiling), unequal key and value widths, two sequences, a carried state.
    #[test]
    fn one_pass_delta_rule_matches_token_by_token() {
        let device = Device::Cpu;
        let (nk, nv, kd, vd) = (2usize, 4usize, 8usize, 6usize);
        for (b, seq) in [(1usize, 1usize), (2, 9), (1, 40)] {
            let unit = |t: Tensor| l2_norm(&t, 1e-6).unwrap();
            let q = unit(Tensor::randn(0f32, 1.0, (b, seq, nk, kd), &device).unwrap());
            let k = unit(Tensor::randn(0f32, 1.0, (b, seq, nk, kd), &device).unwrap());
            let v = Tensor::randn(0f32, 1.0, (b, seq, nv, vd), &device).unwrap();
            let g = Tensor::randn(0f32, 0.5, (b, seq, nv), &device)
                .unwrap()
                .abs()
                .unwrap()
                .neg()
                .unwrap();
            let beta =
                candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0, (b, seq, nv), &device).unwrap())
                    .unwrap();
            let state = Tensor::randn(0f32, 0.5, (b, nv, vd, kd), &device).unwrap();
            let scale = 1.0 / (kd as f64).sqrt();

            let (out, s) = delta_rule_cpu(&q, &k, &v, &g, &beta, &state, scale as f32).unwrap();
            // The composition takes one sequence at a time (its per-token
            // slices of a batch of two are strided in a way candle's matmul
            // refuses), so the batch is checked sequence by sequence.
            let tile = |t: &Tensor| Tensor::cat(&vec![t; nv / nk], 2).unwrap();
            let one = |t: &Tensor, i: usize| t.narrow(0, i, 1).unwrap().contiguous().unwrap();
            let (outs, states): (Vec<_>, Vec<_>) = (0..b)
                .map(|i| {
                    delta_rule(
                        &(tile(&one(&q, i)) * scale).unwrap(),
                        &tile(&one(&k, i)),
                        &one(&v, i),
                        &one(&g, i),
                        &one(&beta, i),
                        one(&state, i),
                    )
                    .unwrap()
                })
                .unzip();
            let want_out = Tensor::cat(&outs, 0).unwrap();
            let want_s = Tensor::cat(&states, 0).unwrap();
            let worst = |a: &Tensor, b: &Tensor| -> f32 {
                (a - b)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar()
                    .unwrap()
            };
            assert_eq!(out.dims(), want_out.dims());
            assert!(worst(&out, &want_out) < 1e-5, "output, b={b} seq={seq}");
            assert!(worst(&s, &want_s) < 1e-5, "state, b={b} seq={seq}");
        }
    }
}
