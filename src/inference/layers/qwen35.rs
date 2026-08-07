use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::Module;

use super::{run_attention, DeltaNetWeights, Qwen35AttnWeights, SsmState};

// ── Qwen 3.5 full-attention layer forward ──

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
                let mut cache = super::new_kv_cache(max_seq_len);
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

    /// Delta net recurrent scan with per-timestep alpha (decay) and beta (input gate).
    ///
    /// alpha: `[b, seq, n_head * key_head_dim]` — softplus(ssm_alpha + ssm_dt(x))
    /// beta:  `[b, seq, q_dim + k_dim + v_dim]` — sigmoid(ssm_beta)
    ///
    /// State update: `state_t = diag(alpha_t) * state_{t-1} + (beta_v * v) ⊗ (beta_k * k)`
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn delta_net_scan(
        &self,
        q: &Tensor, // [b, n_head, seq, key_head_dim]
        k: &Tensor, // [b, n_kv_head, seq, key_head_dim]
        v: &Tensor, // [b, n_v_head, seq, value_head_dim]
        alpha: &Tensor,
        beta: &Tensor,
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

        // Pre-split beta into Q/K/V portions along dim 2
        let q_dim = self.n_head * self.key_head_dim;
        let k_dim = self.n_kv_head * self.key_head_dim;
        let v_dim = self.n_v_head * self.value_head_dim;
        let beta_k_all = beta.narrow(2, q_dim, k_dim)?;
        let beta_v_all = beta.narrow(2, q_dim + k_dim, v_dim)?;

        // Reshape beta_k/v to per-head: [b, seq, n_heads, head_dim]
        let beta_k_heads = beta_k_all
            .reshape((b_sz, seq_len, self.n_kv_head, self.key_head_dim))?
            .transpose(1, 2)?;
        let beta_v_heads = beta_v_all
            .reshape((b_sz, seq_len, self.n_v_head, self.value_head_dim))?
            .transpose(1, 2)?;

        // Repeat KV beta heads for GQA to match expanded k/v
        let beta_k_heads = if self.n_head > self.n_kv_head {
            candle_transformers::utils::repeat_kv(beta_k_heads, self.n_head / self.n_kv_head)?
        } else {
            beta_k_heads
        };
        let beta_v_heads = if self.n_head > self.n_v_head {
            candle_transformers::utils::repeat_kv(beta_v_heads, self.n_head / self.n_v_head)?
        } else {
            beta_v_heads
        };

        // Reshape alpha: [b, seq, n_head * key_head_dim] → [b, n_head, seq, key_head_dim]
        let alpha_heads = alpha
            .reshape((b_sz, seq_len, self.n_head, self.key_head_dim))?
            .transpose(1, 2)?;

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1)?.squeeze(2)?;
            let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
            let v_t = v.narrow(2, t, 1)?.squeeze(2)?;

            // Alpha decay: g_t = exp(-alpha_t) ∈ (0, 1]
            let alpha_t = alpha_heads.narrow(2, t, 1)?.squeeze(2)?; // [b, n_head, key_head_dim]
            let decay = alpha_t.neg()?.exp()?;
            // Broadcast decay over value_head_dim: [b, n_head, 1, key_head_dim]
            let decay_expanded = decay.unsqueeze(2)?;

            // Decay the state: decayed = g_t * S_{t-1}
            let decayed_state = (&state * &decay_expanded)?;

            // Beta gates for K and V at this timestep
            let bk_t = beta_k_heads.narrow(2, t, 1)?.squeeze(2)?; // [b, n_head, key_head_dim]
            let bv_t = beta_v_heads.narrow(2, t, 1)?.squeeze(2)?; // [b, n_head, value_head_dim]

            // Gate K and V with beta: k_gated = β_k * k, v_gated = β_v * v
            let k_gated = (&k_t * &bk_t)?;
            let v_gated = (&v_t * &bv_t)?;

            // Prediction error (delta rule): error = v_gated - decayed_state @ k_gated
            let k_col = k_gated.unsqueeze(3)?; // [b, n_head, key_head_dim, 1]
            let prediction = decayed_state.matmul(&k_col)?.squeeze(3)?; // [b, n_head, value_head_dim]
            let error = (&v_gated - &prediction)?;

            // Error-correcting state update: S_t = decayed + error ⊗ k_gated^T
            let err_col = error.unsqueeze(3)?; // [b, n_head, value_head_dim, 1]
            let k_row = k_gated.unsqueeze(2)?; // [b, n_head, 1, key_head_dim]
            let update = err_col.matmul(&k_row)?; // [b, n_head, value_head_dim, key_head_dim]
            state = (&decayed_state + &update)?;

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
fn softplus(x: &Tensor) -> CandleResult<Tensor> {
    let ones = x.ones_like()?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}
