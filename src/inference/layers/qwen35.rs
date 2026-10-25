//! Qwen 3.5 layers, written op by op against llama.cpp's `src/models/qwen35.cpp`
//! and `delta-net-base.cpp` (master `4b1a27f`) — the plan and its reasoning are
//! in `docs/plans/qwen35_support.md` (FUTURE_WORK #117).

use crate::inference::decode_attn::{axpy, dot};
use crate::inference::split::kv_cache::LayerKv;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::Module;
use rayon::prelude::*;

use super::{run_attention, DeltaNetWeights, Qwen35AttnWeights, SsmRows, SsmSlot, SsmState};

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
            None,
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

    /// Whether this layer's delta rule, on a card, is the card kernel: the
    /// one-pass rule is on (`SWARMLLM_DELTA_RULE`) and the kernel covers its
    /// widths. What a CUDA graph capture has been checked on — the
    /// composition it replaced has not (`SplitModel`'s capture rule).
    pub(crate) fn delta_rule_runs_on_the_card_kernel(&self) -> bool {
        delta_rule_in_one_pass() && card_delta_rule_covers(self.k_head_dim, self.v_head_dim)
    }

    /// `build_layer_attn_linear`: projections, causal conv, L2-normalised
    /// q/k, the gated delta rule, gated RMSNorm, `ssm_out` — the step from
    /// `index_pos` to `index_pos + seq_len`. The state it read is kept
    /// ([`SsmState`]); a card writes the new one into the pair the last step
    /// read, the processor into fresh tensors.
    pub(crate) fn forward_deltanet(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
        index_pos: usize,
    ) -> CandleResult<Tensor> {
        self.forward_deltanet_into(x, ssm_state, index_pos, !x.device().is_cpu())
    }

    /// [`Self::forward_deltanet`], the new state written into the pair the last
    /// step read (`into_spare`, a card's two fixed addresses) or into fresh
    /// tensors (the processor, whose delta rule makes one anyway).
    fn forward_deltanet_into(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
        index_pos: usize,
        into_spare: bool,
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

        // The state must describe the position this step starts from: a step
        // from 0 starts over, any other on a state out of step — or none — is
        // a conversation this layer has lost, refused rather than answered
        // from a summary of some other history.
        match ssm_state.as_ref().map(SsmState::positions) {
            Some(at) if at == index_pos => {}
            _ if index_pos == 0 => {
                let channels = mixed.dim(2)?;
                *ssm_state = Some(SsmState::new(
                    SsmSlot {
                        conv_state: Tensor::zeros(
                            (b_sz, channels, self.conv_kernel - 1),
                            mixed.dtype(),
                            device,
                        )?,
                        recurrent_state: Tensor::zeros((b_sz, nv, vd, kd), DType::F32, device)?,
                    },
                    0,
                ));
            }
            at => candle_core::bail!(
                "a DeltaNet layer's state describes {at:?} positions, the step starts at {index_pos}"
            ),
        }
        let held = ssm_state.as_mut().expect("set above");
        let read = held.current().clone();
        // The buffers this step writes into: on a card the previous step's,
        // so the state read survives the step and lives at two fixed
        // addresses (`SsmState`). Made here when missing — never inside a
        // capture, which asks `SsmState::holds_a_step_to` first.
        let target = if into_spare {
            Some(match held.take_spare() {
                Some(spare) if spare.recurrent_state.dims() == read.recurrent_state.dims() => spare,
                _ => SsmSlot {
                    conv_state: read.conv_state.zeros_like()?,
                    recurrent_state: read.recurrent_state.zeros_like()?,
                },
            })
        } else {
            None
        };

        let (conv, new_conv) = self.causal_conv(&mixed, &read.conv_state)?;
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

        let (out, new_state, written) = delta_rule_into(
            &qk,
            nk,
            &v,
            &g,
            &beta,
            &read.recurrent_state,
            target.as_ref().map(|t| &t.recurrent_state),
        )?; // out [b, seq, nv, vd]

        // Gated RMSNorm per value head: RMSNorm(o, ssm_norm) · silu(z).
        let z = z.reshape((b_sz, seq_len, nv, vd))?;
        let normed = self.ssm_norm.forward(&out)?;
        let gated = (normed * candle_nn::ops::silu(&z)?)?;
        let gated = gated.reshape((b_sz, seq_len, nv * vd))?;

        let in_place = target.is_some();
        let next = written_into(target, new_conv, new_state, written)?;
        // What a rewind into this step replays (`SsmState::rewind_to`).
        held.keep_rows(
            SsmRows {
                mixed,
                qk,
                v: v.contiguous()?,
                g,
                beta,
            },
            in_place,
        )?;
        held.advance(next, index_pos + seq_len);
        self.ssm_out.forward(&gated)
    }

    /// Causal depthwise convolution with the previous `kernel − 1` inputs as
    /// state (`ggml_ssm_conv`): output t reads inputs t−(K−1) … t, tap j
    /// weighing input t−(K−1)+j. `prev` is `[b, C, K−1]`; returns `[b, seq, C]`
    /// and the new state `[b, C, K−1]` — the last K−1 inputs, carried to the
    /// next call.
    pub(crate) fn causal_conv(&self, x: &Tensor, prev: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_b_sz, seq_len, channels) = x.dims3()?;
        let k = self.conv_kernel;
        let pad = k - 1;
        let x_t = x.transpose(1, 2)?.contiguous()?; // [b, C, seq]
        let padded = Tensor::cat(&[prev, &x_t], 2)?; // [b, C, seq + K − 1]
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

/// The new state as a step or a replay leaves it: on a card in `target`, the
/// buffers it keeps (the delta rule's kernel `written` its state there
/// already), on the processor the tensors made.
fn written_into(
    target: Option<SsmSlot>,
    conv_state: Tensor,
    recurrent_state: Tensor,
    written: bool,
) -> CandleResult<SsmSlot> {
    Ok(match target {
        Some(t) => {
            t.conv_state.slice_set(&conv_state, 0, 0)?;
            if !written {
                t.recurrent_state
                    .slice_set(&recurrent_state.contiguous()?, 0, 0)?;
            }
            t
        }
        None => SsmSlot {
            conv_state,
            recurrent_state,
        },
    })
}

/// The delta rule over a step's rows, from `state` — the one choice of kernel
/// a step and a replay share, so a replay's state is the step's: the card
/// kernel (written straight into `into`, the third answer `true`), the
/// processor's one pass, or the token-by-token composition
/// (`SWARMLLM_DELTA_RULE=steps`). q and k are `qk` `[b, seq, 2·nk, kd]`,
/// L2-normalised; the scale is `1/√kd`. Returns the outputs
/// `[b, seq, nv, vd]` and the new state.
#[allow(clippy::too_many_arguments)]
fn delta_rule_into(
    qk: &Tensor,
    nk: usize,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    into: Option<&Tensor>,
) -> CandleResult<(Tensor, Tensor, bool)> {
    let (_, _, nv, vd) = v.dims4()?;
    let kd = qk.dim(3)?;
    let device = qk.device();
    let scale = 1.0 / (kd as f64).sqrt();
    let one_pass = delta_rule_in_one_pass();
    crate::inference::prof::timed!(crate::inference::prof::Stage::DeltaRule, {
        let on_card = if one_pass && device.is_cuda() && card_delta_rule_covers(kd, vd) {
            let out_buffer = match into {
                Some(t) => t.clone(),
                None => state.zeros_like()?,
            };
            delta_rule_cuda(qk, nk, v, g, beta, state, &out_buffer, scale as f32)?
                .map(|out| (out, out_buffer, into.is_some()))
        } else {
            None
        };
        let (q, k) = (qk.narrow(2, 0, nk)?, qk.narrow(2, nk, nk)?);
        Ok(match on_card {
            Some(done) => done,
            None if one_pass && device.is_cpu() => {
                let (out, new) = delta_rule_cpu(&q, &k, v, g, beta, state, scale as f32)?;
                (out, new, false)
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
                let (out, new) = delta_rule(&(q * scale)?, &k, v, g, beta, state.clone())?;
                (out, new, false)
            }
        })
    })
}

/// The state after the first `rows` rows of a step, from the state the step
/// read (`read`) and the inputs it kept (`kept`) — `SsmState::rewind_to` into
/// a check. Only the convolution's window and the delta rule run: the window is
/// the last `kernel − 1` of `read`'s and those rows' inputs, as
/// [`DeltaNetWeights::causal_conv`] leaves it, and the delta rule takes each
/// row's update in the step's order with the step's kernel
/// ([`delta_rule_into`]), so the state is bit for bit the one the step passed
/// at that row (`a_check_rewound_into_its_middle_is_where_its_kept_rows_lead`).
/// On a card it is written into `into`'s buffers, which a captured step reads.
pub(super) fn replay_rows(
    read: &SsmSlot,
    kept: &SsmRows,
    rows: usize,
    into: Option<&SsmSlot>,
) -> CandleResult<SsmSlot> {
    let first = |t: &Tensor| t.narrow(1, 0, rows)?.contiguous();
    let pad = read.conv_state.dim(2)?;
    let inputs = first(&kept.mixed)?.transpose(1, 2)?.contiguous()?; // [b, C, rows]
    let conv_state = Tensor::cat(&[&read.conv_state, &inputs], 2)?
        .narrow(2, rows, pad)?
        .contiguous()?;
    let qk = first(&kept.qk)?;
    let nk = qk.dim(2)? / 2;
    let (_, recurrent_state, written) = delta_rule_into(
        &qk,
        nk,
        &first(&kept.v)?,
        &first(&kept.g)?,
        &first(&kept.beta)?,
        &read.recurrent_state,
        into.map(|t| &t.recurrent_state),
    )?;
    written_into(into.cloned(), conv_state, recurrent_state, written)
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
/// on a card, a width [`card_delta_rule_covers`]. The new state is written
/// INTO `state_out` (contiguous, the state's shape, never the state itself —
/// `SsmState`'s second pair); the outputs `[b, seq, nv, vd]` are returned.
/// `None` off a card build or outside that scope — the caller keeps the
/// composition. Not bit-identical to it; replies are judged against llama.cpp.
#[allow(clippy::too_many_arguments)]
fn delta_rule_cuda(
    qk: &Tensor,
    nk: usize,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    state_out: &Tensor,
    scale: f32,
) -> CandleResult<Option<Tensor>> {
    #[cfg(feature = "candle-cuda")]
    {
        if [qk, v, g, beta, state, state_out]
            .iter()
            .any(|t| t.dtype() != DType::F32)
            || !state_out.is_contiguous()
        {
            return Ok(None);
        }
        let (b, seq, nv, vd) = v.dims4()?;
        let gb = Tensor::stack(&[g, beta], 3)?; // [b, seq, nv, 2]
        let op = cuda::DeltaRule {
            state: state.contiguous()?,
            state_out: state_out.clone(),
            nk,
            scale,
        };
        let out = qk.contiguous()?.apply_op3_no_bwd(v, &gb, &op)?;
        Ok(Some(out.reshape((b, seq, nv, vd))?))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (qk, nk, v, g, beta, state, state_out, scale);
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
        /// Where the new state is written — the same shape, contiguous, other
        /// buffers than `state` (`SsmState`'s second pair). Written through
        /// the device pointer: candle has no four-operand in-place op.
        pub(super) state_out: Tensor,
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
                || self.state_out.dims() != [b, nv, d, d]
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
            let (out_storage, out_layout) = self.state_out.storage_and_layout();
            let candle_core::Storage::Cuda(st_out) = &*out_storage else {
                candle_core::bail!("gated-delta-rule: the new state's buffer is not on the card");
            };
            let Some((oo, oe)) = out_layout.contiguous_offsets() else {
                candle_core::bail!("gated-delta-rule: the new state's buffer must be contiguous");
            };

            let qk = s1.as_cuda_slice::<f32>()?.slice(qo..qe);
            let v = s2.as_cuda_slice::<f32>()?.slice(l2.start_offset()..);
            let gb = s3.as_cuda_slice::<f32>()?.slice(go..ge);
            let state = st.as_cuda_slice::<f32>()?.slice(so..se);
            let state_out = st_out.as_cuda_slice::<f32>()?.slice(oo..oe);
            let n_out = b * seq * nv * d;
            // Every element is assigned: each (sequence, token, head, row)
            // output by its warp's lane 0. Each element of `state_out` is
            // assigned by its lane too.
            let mut dst = dev.alloc_fully_overwritten::<f32>(n_out)?;
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
            builder.arg(&state_out);
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
            // shapes and v's strides, `dst` is sized for exactly the outputs
            // and `state_out` is the state's shape. It is written through a
            // read view, which only the ONE stream per device orders
            // (`SWARMLLM_CUDA_OWN_STREAM`; event tracking is off by default).
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev),
                Shape::from_dims(&[n_out]),
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
        weights_of(device, false)
    }

    /// `t` rounded to a multiple of `1 / per`: a sum of products of such
    /// values is exact in f32, so it is the same whatever order a kernel adds
    /// it in.
    fn exact(t: Tensor, per: f64) -> Tensor {
        t.affine(per, 0.0)
            .unwrap()
            .round()
            .unwrap()
            .affine(1.0 / per, 0.0)
            .unwrap()
    }

    /// The test layer; with `exact_projections`, its input projections hold
    /// multiples of 1/32, so fed multiples of 1/8 ([`exact`]) every row they
    /// make is the same however many rows are multiplied together. On an ARM
    /// processor it is not otherwise: the `gemm` crate multiplies one row by
    /// a matrix-vector kernel and several by its blocked one (its small-size
    /// dot-product kernel, which serves both on x86, is `H_M = 0` for NEON),
    /// and the two add in different orders — so a check's rows part from the
    /// same rows read one at a time in the last bit (CI's macOS job,
    /// 2026-10-25).
    fn weights_of(device: &Device, exact_projections: bool) -> DeltaNetWeights {
        let (hidden, nk, kd, nv, vd, k) = (8usize, 2usize, 4usize, 4usize, 4usize, 4usize);
        let c = 2 * nk * kd + nv * vd;
        let mm = |o: usize, i: usize| {
            let w = Tensor::randn(0f32, 0.3, (o, i), device).unwrap();
            super::super::QMatMul::from_dense(if exact_projections { exact(w, 32.0) } else { w })
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
        // Both ways a new state is written: into fresh tensors (the
        // processor) and into the previous step's buffers (a card, `SsmState`).
        for into_spare in [false, true] {
            let mut whole_state = None;
            let whole = w
                .forward_deltanet_into(&x, &mut whole_state, 0, into_spare)
                .unwrap();

            let mut state = None;
            let mut parts = vec![w
                .forward_deltanet_into(&x.narrow(1, 0, 3).unwrap(), &mut state, 0, into_spare)
                .unwrap()];
            for t in 3..7 {
                parts.push(
                    w.forward_deltanet_into(&x.narrow(1, t, 1).unwrap(), &mut state, t, into_spare)
                        .unwrap(),
                );
            }
            let stepped = Tensor::cat(&parts, 1).unwrap();
            let diff = max_abs_diff(&whole, &stepped);
            assert!(
                diff < 1e-5,
                "into_spare={into_spare}: stepped decode drifts from one pass by {diff}"
            );
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
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
    }

    /// A refused CUDA graph capture takes a decode step back and runs it again
    /// (`cuda_graph::DecodeGraph::capture`: groups launched before the refusal
    /// HAVE run). With the state's second pair kept, the step is taken back to
    /// the state it read and the re-run lands exactly where an uninterrupted
    /// run does — red if a step wrote into the state it read, or if rewinding
    /// left the state's position or buffers behind (gotcha #761). Taking back
    /// more than one step is refused, never faked.
    #[test]
    fn a_deltanet_step_taken_back_and_run_again_lands_where_one_run_does() {
        let device = Device::Cpu;
        let w = weights(&device);
        let x = Tensor::randn(0f32, 1.0, (1, 6, 8), &device).unwrap();
        let step = |state: &mut Option<SsmState>, t: usize| {
            w.forward_deltanet_into(&x.narrow(1, t, 1).unwrap(), state, t, true)
                .unwrap()
        };

        let mut plain = None;
        w.forward_deltanet_into(&x.narrow(1, 0, 3).unwrap(), &mut plain, 0, true)
            .unwrap();
        let plain_outs: Vec<Tensor> = (3..6).map(|t| step(&mut plain, t)).collect();

        let mut redone = None;
        w.forward_deltanet_into(&x.narrow(1, 0, 3).unwrap(), &mut redone, 0, true)
            .unwrap();
        // Step 3 runs, is taken back, and runs again; then 4 and 5 as usual.
        step(&mut redone, 3);
        let held = redone.as_mut().unwrap();
        assert!(held.can_rewind_to(3) && held.rewind_to(3).unwrap());
        assert_eq!(held.positions(), 3);
        assert!(held.holds_a_step_to(4), "the re-run must find its buffers");
        assert!(
            !held.can_rewind_to(4),
            "a taken-back step's pair describes nothing"
        );
        let redone_outs: Vec<Tensor> = (3..6).map(|t| step(&mut redone, t)).collect();
        for (t, (a, b)) in plain_outs.iter().zip(&redone_outs).enumerate() {
            assert_eq!(
                max_abs_diff(a, b),
                0.0,
                "step {} differs after a take-back",
                t + 3
            );
        }
        let (a, b) = (plain.unwrap(), redone.unwrap());
        assert_eq!(
            max_abs_diff(&a.current().recurrent_state, &b.current().recurrent_state),
            0.0
        );
        assert_eq!(
            max_abs_diff(&a.current().conv_state, &b.current().conv_state),
            0.0
        );

        // Two steps back is more than the pair holds.
        let mut b = Some(b);
        assert!(!b.as_mut().unwrap().rewind_to(4).unwrap());
        let mut two = None;
        w.forward_deltanet_into(&x.narrow(1, 0, 3).unwrap(), &mut two, 0, false)
            .unwrap();
        w.forward_deltanet_into(&x.narrow(1, 3, 1).unwrap(), &mut two, 3, false)
            .unwrap();
        w.forward_deltanet_into(&x.narrow(1, 4, 1).unwrap(), &mut two, 4, false)
            .unwrap();
        assert!(!two.as_mut().unwrap().rewind_to(3).unwrap());
        assert_eq!(two.as_ref().unwrap().positions(), 5);
    }

    /// The processor writes each step's state into fresh tensors and keeps the
    /// one it read, so a step of several positions — a check of guesses — is
    /// taken back to its start and read again shorter, landing exactly where
    /// reading only the shorter run does. Red if the
    /// processor dropped the state it read, or if the delta rule updated it in
    /// place.
    #[test]
    fn a_processor_step_of_guesses_is_taken_back_and_read_again_shorter() {
        let device = Device::Cpu;
        let w = weights(&device);
        let x = Tensor::randn(0f32, 1.0, (1, 8, 8), &device).unwrap();
        let run = |state: &mut Option<SsmState>, from: usize, len: usize| {
            w.forward_deltanet_into(&x.narrow(1, from, len).unwrap(), state, from, false)
                .unwrap()
        };

        let mut plain = None;
        run(&mut plain, 0, 3);
        run(&mut plain, 3, 2);
        let plain_next = run(&mut plain, 5, 1);

        let mut guessed = None;
        run(&mut guessed, 0, 3);
        // Positions 3..8 checked at once; only 3 and 4 kept.
        run(&mut guessed, 3, 5);
        let held = guessed.as_mut().unwrap();
        assert!(
            held.rewind_to(3).unwrap(),
            "the processor keeps the state it read"
        );
        assert_eq!(held.positions(), 3);
        run(&mut guessed, 3, 2);
        let guessed_next = run(&mut guessed, 5, 1);

        assert_eq!(max_abs_diff(&plain_next, &guessed_next), 0.0);
        let (a, b) = (plain.unwrap(), guessed.unwrap());
        assert_eq!(
            max_abs_diff(&a.current().recurrent_state, &b.current().recurrent_state),
            0.0
        );
        assert_eq!(
            max_abs_diff(&a.current().conv_state, &b.current().conv_state),
            0.0
        );
    }

    /// A check rewound INTO its middle — a rejected guess (#228 (5)) — replays
    /// only the rows it accepted from the state it read, and lands bit for bit
    /// where reading those rows alone does: the state, the convolution's
    /// window and the next step's output. Both ways a step writes its state
    /// (fresh tensors, the processor; the kept buffers, a card), and both ways
    /// a replay does (`replay_rows` into the current pair's buffers is the
    /// card's). A second rewind further back still replays from the state the
    /// check read; one to the check's start is the swap; past where the state
    /// is now, or wider than the rows kept, is refused. Red if a replay reads
    /// the wrong window (the conv state's last `kernel − 1`), skips a row's
    /// decay, or starts from the state after the check.
    #[test]
    fn a_check_rewound_into_its_middle_is_where_its_kept_rows_lead() {
        let device = Device::Cpu;
        // Exact projections: the one-at-a-time reference multiplies one row
        // where the check multiplied six, which an ARM processor's `gemm`
        // rounds differently (`weights_of`) — a property of the matrix
        // multiply, not of the replay this test checks.
        let w = weights_of(&device, true);
        let x = exact(Tensor::randn(0f32, 1.0, (1, 10, 8), &device).unwrap(), 8.0);
        let same = |a: &SsmState, b: &SsmState, what: &str| {
            assert_eq!(a.positions(), b.positions(), "{what}: positions");
            assert_eq!(
                max_abs_diff(&a.current().recurrent_state, &b.current().recurrent_state),
                0.0,
                "{what}: recurrent state"
            );
            assert_eq!(
                max_abs_diff(&a.current().conv_state, &b.current().conv_state),
                0.0,
                "{what}: convolution window"
            );
        };
        for into_spare in [false, true] {
            let run = |state: &mut Option<SsmState>, from: usize, len: usize| {
                w.forward_deltanet_into(&x.narrow(1, from, len).unwrap(), state, from, into_spare)
                    .unwrap()
            };
            // One at a time from 3, as a reply read without guessing does.
            let alone = |upto: usize| {
                let mut s = None;
                run(&mut s, 0, 3);
                for t in 3..upto {
                    run(&mut s, t, 1);
                }
                s
            };
            let mut plain = alone(5);
            let plain_next = run(&mut plain, 5, 1);

            let mut checked = None;
            run(&mut checked, 0, 3);
            // Positions 3..9 checked at once; 3 and 4 kept.
            run(&mut checked, 3, 6);
            let held = checked.as_mut().unwrap();
            assert!(held.can_rewind_to(5) && held.rewind_to(5).unwrap());
            same(
                held,
                alone(5).as_ref().unwrap(),
                &format!("into_spare={into_spare}, at 5"),
            );
            // Back further, within the check: still from the state it read.
            assert!(held.rewind_to(4).unwrap());
            same(
                held,
                alone(4).as_ref().unwrap(),
                &format!("into_spare={into_spare}, at 4"),
            );
            assert!(
                !held.can_rewind_to(5) && !held.rewind_to(5).unwrap(),
                "a rewind never goes forward"
            );
            assert!(held.rewind_to(4).unwrap(), "where it is already");
            // Read on from 4 as the reply would: 4 again, then 5.
            run(&mut checked, 4, 1);
            let next = run(&mut checked, 5, 1);
            assert_eq!(
                max_abs_diff(&plain_next, &next),
                0.0,
                "into_spare={into_spare}: the step after a replay"
            );

            // The card's replay: the state written into the current pair.
            let mut card = None;
            run(&mut card, 0, 3);
            run(&mut card, 3, 6);
            let held = card.as_mut().unwrap();
            let (read, kept) = (held.previous.clone().unwrap(), held.rows.clone().unwrap());
            let into = held.current().clone();
            replay_rows(&read, &kept, 2, Some(&into)).unwrap();
            let fresh = replay_rows(&read, &kept, 2, None).unwrap();
            // Written into the pair's own buffers, as a captured step reads them.
            assert_eq!(
                max_abs_diff(&into.recurrent_state, &fresh.recurrent_state),
                0.0
            );
            assert_eq!(max_abs_diff(&into.conv_state, &fresh.conv_state), 0.0);
            same(
                &SsmState::new(fresh, 5),
                alone(5).as_ref().unwrap(),
                "a replay into fresh tensors",
            );
            // To the check's start is the swap, as a refused capture's.
            assert!(held.rewind_to(3).unwrap());
            same(held, alone(3).as_ref().unwrap(), "at the check's start");
            assert!(!held.can_rewind_to(4), "the swap leaves no rows to replay");
        }

        // A step of one position keeps no rows; a prompt pass wider than any
        // check keeps none either.
        let mut wide = None;
        let long = Tensor::randn(0f32, 1.0, (1, SsmState::MOST_ROWS_KEPT + 1, 8), &device).unwrap();
        w.forward_deltanet_into(&long, &mut wide, 0, false).unwrap();
        let held = wide.as_ref().unwrap();
        assert!(held.rows_held == 0 && !held.can_rewind_to(5));
    }

    /// A state that does not describe the step's first position is a lost
    /// conversation: refused, never answered from zeros or someone's history.
    #[test]
    fn a_deltanet_step_out_of_step_with_its_state_is_refused() {
        let device = Device::Cpu;
        let w = weights(&device);
        let x = Tensor::randn(0f32, 1.0, (1, 4, 8), &device).unwrap();
        let mut state = None;
        assert!(w
            .forward_deltanet(&x.narrow(1, 0, 1).unwrap(), &mut state, 2)
            .is_err());
        w.forward_deltanet(&x.narrow(1, 0, 2).unwrap(), &mut state, 0)
            .unwrap();
        assert!(w
            .forward_deltanet(&x.narrow(1, 2, 1).unwrap(), &mut state, 3)
            .is_err());
        // A step from 0 starts the conversation over.
        w.forward_deltanet(&x.narrow(1, 0, 1).unwrap(), &mut state, 0)
            .unwrap();
        assert_eq!(state.as_ref().unwrap().positions(), 1);
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
