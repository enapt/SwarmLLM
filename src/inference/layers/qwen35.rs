//! Qwen 3.5 layers, written op by op against llama.cpp's `src/models/qwen35.cpp`
//! and `delta-net-base.cpp` (master `4b1a27f`) — the plan and its reasoning are
//! in `docs/plans/qwen35_support.md` (FUTURE_WORK #117).

use crate::inference::decode_attn::{axpy, dot};
use crate::inference::split::kv_cache::LayerKv;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::Module;
use rayon::prelude::*;

use super::{
    run_attention, DeltaNetWeights, QMatMul, Qwen35AttnWeights, SsmRows, SsmSlot, SsmState,
};

// ── Qwen 3.5 full-attention layer forward ──

impl Qwen35AttnWeights {
    pub(crate) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, _head_dim) = x.dims4()?;
        // A tree check's rows turn at their own depths (`RowPositions`, #437),
        // as a dense layer's do (`LayerWeights::rope_tables`).
        let (cos, sin) = match super::tree_rope_rows(seq_len, &self.cos, &self.sin)? {
            Some(rows) => rows,
            None => (
                self.cos.narrow(0, index_pos, seq_len)?,
                self.sin.narrow(0, index_pos, seq_len)?,
            ),
        };
        // IMROPE with every position component equal to the token index —
        // what text is — rotates each pair by `pos * base^(-2i/n_rot)`, i.e.
        // plain NEOX-style partial RoPE over the first `rope_dim` dims —
        // Qwen3-Next's own type (`llama_model_rope_type`).
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

// ── Value heads grouped by key head (Qwen3-Next) ──

/// The order a DeltaNet layer's FILE keeps its value heads in. Both families
/// give each key head `r = nv / nk` value heads, and llama.cpp spells out the
/// difference (`llama-model.cpp` `get_split_segments`, master `4b1a27f`):
/// Qwen 3.5 TILES them — value head `h` reads key head `h % nk`,
/// `[k0_v0, k1_v1, k0_v2, k1_v3]` — and Qwen3-Next GROUPS them — `h / r`,
/// `[k0_v0, k0_v1, k1_v2, k1_v3]` (`qwen3next.cpp` repeats each key head `r`
/// times in place). Every per-head weight follows: the value rows of
/// `attn_qkv` and of the convolution, `attn_gate`, beta, alpha, `ssm_dt`,
/// `ssm_a`, and `ssm_out`'s columns.
///
/// The kernels read the tiled order (`delta_rule.cu`, `delta_rule_cpu`), so
/// the loader puts a grouped file's rows in it ([`GroupedValueHeads`]) — whole
/// rows, still quantized. `ssm_out` alone keeps the file's order: its value
/// heads are COLUMNS, and a 256-wide K-quant block holds two 128-wide heads
/// that land apart, so its input is put back in the file's order instead (one
/// copy of `nv · vd` values a position).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueHeadOrder {
    /// The kernels' order — Qwen 3.5's files.
    Tiled,
    /// Grouped by key head — what a Qwen3-Next file's `ssm_out` reads.
    GroupedByKeyHead,
}

/// A Qwen3-Next DeltaNet layer's head geometry, for putting its file's rows in
/// the kernels' order ([`ValueHeadOrder`]). Kernel head `h` is the file's head
/// `(h % nk) · r + h / nk`: both read key head `h % nk`, and the `j`-th kernel
/// head of a key head is the `j`-th of its group.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GroupedValueHeads {
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) k_head_dim: usize,
    pub(crate) v_head_dim: usize,
}

/// A Qwen3-Next layer's input projections as its file holds them.
pub(crate) enum InputProjections {
    /// `attn_qkv` (q, k, the values) and `attn_gate` (z).
    Split { qkv: QTensor, gate: QTensor },
    /// The older files' `ssm_in`: per key head its q, its k, its group's
    /// values, then its group's gates (llama.cpp's "legacy (slower) path").
    Fused(QTensor),
}

/// A DeltaNet layer's per-head weights in the kernels' order — what
/// [`GroupedValueHeads::kernel_order`] answers, and what a Qwen 3.5 file
/// holds as it is.
pub(crate) struct KernelOrder {
    pub(crate) wqkv: QMatMul,
    pub(crate) wz: QMatMul,
    pub(crate) w_beta: QMatMul,
    pub(crate) w_alpha: QMatMul,
    pub(crate) conv1d: Tensor,
    pub(crate) dt_bias: Tensor,
    pub(crate) a: Tensor,
}

impl GroupedValueHeads {
    fn per_key_head(&self) -> usize {
        self.n_v_heads / self.n_k_heads
    }

    /// The file's value head for kernel head `h`.
    fn file_head(&self, h: usize) -> usize {
        (h % self.n_k_heads) * self.per_key_head() + h / self.n_k_heads
    }

    /// One entry per value head (`ssm_dt`, `ssm_a`), in kernel order.
    pub(crate) fn head_rows(&self) -> Vec<usize> {
        (0..self.n_v_heads).map(|h| self.file_head(h)).collect()
    }

    /// `v_head_dim` rows per value head, the file's heads starting at row
    /// `from`, in kernel order.
    fn value_rows(&self, from: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.n_v_heads).flat_map(move |h| {
            let start = from + self.file_head(h) * self.v_head_dim;
            start..start + self.v_head_dim
        })
    }

    /// `attn_qkv`'s rows — and the convolution's channels — in kernel order:
    /// q and k as they are, the values reordered.
    pub(crate) fn qkv_rows(&self) -> Vec<usize> {
        let keys = 2 * self.n_k_heads * self.k_head_dim;
        (0..keys).chain(self.value_rows(keys)).collect()
    }

    /// `attn_gate`'s rows in kernel order — and `ssm_out`'s columns.
    pub(crate) fn gate_rows(&self) -> Vec<usize> {
        self.value_rows(0).collect()
    }

    /// `ssm_ba`'s beta rows and its alpha rows, in kernel order: per key head
    /// the file holds its group's `r` betas, then their `r` alphas
    /// (`qwen3next.cpp` `build_layer_attn_linear`).
    pub(crate) fn beta_alpha_rows(&self) -> (Vec<usize>, Vec<usize>) {
        let r = self.per_key_head();
        let row = |h: usize, past: usize| (h % self.n_k_heads) * 2 * r + past + h / self.n_k_heads;
        (
            (0..self.n_v_heads).map(|h| row(h, 0)).collect(),
            (0..self.n_v_heads).map(|h| row(h, r)).collect(),
        )
    }

    /// The fused `ssm_in`'s rows for `attn_qkv` and for `attn_gate`, in kernel
    /// order ([`InputProjections::Fused`]).
    pub(crate) fn fused_rows(&self) -> (Vec<usize>, Vec<usize>) {
        let (nk, kd, vd, r) = (
            self.n_k_heads,
            self.k_head_dim,
            self.v_head_dim,
            self.per_key_head(),
        );
        let block = 2 * kd + 2 * r * vd;
        let q = (0..nk).flat_map(|kh| kh * block..kh * block + kd);
        let k = (0..nk).flat_map(|kh| kh * block + kd..kh * block + 2 * kd);
        let group = |h: usize, past: usize| {
            let start = (h % nk) * block + 2 * kd + past + (h / nk) * vd;
            start..start + vd
        };
        let v = (0..self.n_v_heads).flat_map(|h| group(h, 0));
        let z = (0..self.n_v_heads).flat_map(|h| group(h, r * vd));
        (q.chain(k).chain(v).collect(), z.collect())
    }

    /// The layer's per-head weights in kernel order, on `device`: the
    /// projections' rows gathered whole and still quantized, `ssm_ba` cut into
    /// beta and alpha, the convolution's channels and the per-head vectors
    /// reordered. `ssm_out` is not among them ([`ValueHeadOrder`]).
    pub(crate) fn kernel_order(
        &self,
        input: &InputProjections,
        ba: &QTensor,
        conv1d: &Tensor,
        dt_bias: &Tensor,
        a: &Tensor,
        device: &Device,
    ) -> CandleResult<KernelOrder> {
        if self.n_k_heads == 0 || !self.n_v_heads.is_multiple_of(self.n_k_heads) {
            candle_core::bail!(
                "{} value heads do not share {} key heads evenly",
                self.n_v_heads,
                self.n_k_heads
            )
        }
        // The head counts and widths are the header's numbers, and every row
        // list below is sized by them: they must describe the tensors the file
        // holds before one is built (a size another computer chose is checked
        // before it sizes anything).
        let rows_of = |t: &QTensor| t.shape().dims2().map(|(rows, _)| rows);
        let made = (|| {
            let keys = self
                .n_k_heads
                .checked_mul(self.k_head_dim)?
                .checked_mul(2)?;
            let values = self.n_v_heads.checked_mul(self.v_head_dim)?;
            Some((
                keys.checked_add(values)?,
                values,
                self.n_v_heads.checked_mul(2)?,
            ))
        })();
        let Some((qkv_rows, value_rows, ba_rows)) = made else {
            candle_core::bail!("the header's head geometry overflows: {self:?}")
        };
        let mut checks = vec![("ssm_ba", rows_of(ba)?, ba_rows)];
        match input {
            InputProjections::Split { qkv, gate } => {
                checks.push(("attn_qkv", rows_of(qkv)?, qkv_rows));
                checks.push(("attn_gate", rows_of(gate)?, value_rows));
            }
            // Per key head its q, its k, its group's values and gates.
            InputProjections::Fused(fused) => {
                checks.push((
                    "ssm_in",
                    rows_of(fused)?,
                    qkv_rows.saturating_add(value_rows),
                ));
            }
        }
        for (name, held, wanted) in checks {
            if held != wanted {
                candle_core::bail!(
                    "{name} holds {held} rows where the header's heads make {wanted}"
                )
            }
        }
        let matrix =
            |t: &QTensor, rows: &[usize]| QMatMul::from_qtensor(rows_in_order(t, rows, device)?);
        let (wqkv, wz) = match input {
            InputProjections::Split { qkv, gate } => (
                matrix(qkv, &self.qkv_rows())?,
                matrix(gate, &self.gate_rows())?,
            ),
            InputProjections::Fused(fused) => {
                let (qkv, z) = self.fused_rows();
                (matrix(fused, &qkv)?, matrix(fused, &z)?)
            }
        };
        let (beta, alpha) = self.beta_alpha_rows();
        let select = |t: &Tensor, rows: &[usize]| -> CandleResult<Tensor> {
            if t.dim(0)? != rows.len() {
                candle_core::bail!("{} rows where the heads make {}", t.dim(0)?, rows.len())
            }
            let ids: Vec<u32> = rows.iter().map(|&row| row as u32).collect();
            t.index_select(&Tensor::new(ids.as_slice(), t.device())?, 0)
        };
        Ok(KernelOrder {
            wqkv,
            wz,
            w_beta: matrix(ba, &beta)?,
            w_alpha: matrix(ba, &alpha)?,
            conv1d: select(conv1d, &self.qkv_rows())?,
            dt_bias: select(dt_bias, &self.head_rows())?,
            a: select(a, &self.head_rows())?,
        })
    }
}

/// Rows `rows` of a quantized matrix, in that order, as a new matrix on
/// `device` — whole rows, so whole quantization blocks: nothing is
/// dequantized. Every row asked for must be one of the matrix's.
fn rows_in_order(t: &QTensor, rows: &[usize], device: &Device) -> CandleResult<QTensor> {
    let (n_rows, cols) = t.shape().dims2()?;
    let dtype = t.dtype();
    if !cols.is_multiple_of(dtype.block_size()) {
        candle_core::bail!("a row of {cols} is not whole {dtype:?} blocks")
    }
    let row_bytes = cols / dtype.block_size() * dtype.type_size();
    let data = t.data()?;
    if data.len() != n_rows * row_bytes {
        candle_core::bail!("{} bytes are not {n_rows} rows of {row_bytes}", data.len())
    }
    let mut gathered = Vec::with_capacity(rows.len() * row_bytes);
    for &row in rows {
        if row >= n_rows {
            candle_core::bail!("row {row} of a {n_rows}-row matrix")
        }
        gathered.extend_from_slice(&data[row * row_bytes..(row + 1) * row_bytes]);
    }
    candle_core::quantized::ggml_file::qtensor_from_ggml(
        dtype,
        &gathered,
        vec![rows.len(), cols],
        device,
    )
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

    /// Whether a flowing level of this layer may be captured
    /// (`SplitModel::tree_steps_may_be_captured`): its convolution and delta
    /// rule are the card's kernels, reading tables staged before the capture
    /// (`FlowRows::stage_level`) — the composed convolution uploads its own.
    /// Never under `SWARMLLM_FLOW_GRAPH=0` ([`flowing_levels_captured`]).
    pub(crate) fn flowing_level_may_be_captured(&self) -> bool {
        #[cfg(feature = "candle-cuda")]
        {
            flowing_levels_captured()
                && self.delta_rule_runs_on_the_card_kernel()
                && conv_in_one_launch()
                && self.conv_kernel == 4
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            false
        }
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
        if let Some(walk) = super::flow_walk_now() {
            return self.forward_flowing(x, ssm_state, index_pos, walk as usize);
        }
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let device = x.device();
        let (nk, kd, nv, vd) = (
            self.n_k_heads,
            self.k_head_dim,
            self.n_v_heads,
            self.v_head_dim,
        );

        // One activation, quantized once for both (`QMatMul::forward_shared`).
        let [mixed, z]: [Tensor; 2] = QMatMul::forward_shared(x, &[&self.wqkv, &self.wz])?
            .try_into()
            .map_err(|_| candle_core::Error::Msg("two projections, two outputs".into()))?;
        // mixed [b, seq, 2·nk·kd + nv·vd], z [b, seq, nv·vd]
        let gates = self.gates(x)?;
        // A tree of guesses (#437): each row's parent among the step's rows —
        // its convolution reads up its own line and its delta rule starts
        // from its parent's state. `None` for a step's ordinary rows.
        let tree = tree_parents_now(seq_len)?
            .map(|parents| TreeLines::new(parents, x.device()))
            .transpose()?;

        // The state must describe the position this step starts from: a step
        // from 0 starts over, any other on a state out of step — or none — is
        // a conversation this layer has lost, refused rather than answered
        // from a summary of some other history. A branching tree's state is
        // its last row's: the verdict takes it back before a step reads on.
        match ssm_state.as_ref().map(|s| (s.positions(), s.is_branched())) {
            Some((at, false)) if at == index_pos => {}
            Some((at, true)) if at == index_pos && index_pos > 0 => candle_core::bail!(
                "a DeltaNet layer's state is a tree check's, not yet taken back to the path it kept"
            ),
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
                "a DeltaNet layer's state describes {:?} positions, the step starts at {index_pos}",
                at.map(|(at, _)| at)
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

        // q and k side by side, `[b, seq, 2·nk, kd]` (heads 0..nk are q), so
        // one L2 norm serves both and the card kernel reads them as one input.
        // On a card the convolution and its SiLU are one launch, written as
        // q | k and v apart and its window straight into the step's buffers
        // (`conv_on_the_card`); `new_conv` is then `None`.
        let fused = match target.as_ref() {
            Some(t) => {
                self.conv_on_the_card(&mixed, &read.conv_state, &t.conv_state, tree.as_ref(), None)?
            }
            None => None,
        };
        let (qk, v, new_conv) = match fused {
            Some((qk, v)) => (
                qk.reshape((b_sz, seq_len, 2 * nk, kd))?,
                v.reshape((b_sz, seq_len, nv, vd))?,
                None,
            ),
            None => {
                let (conv, new_conv) = match &tree {
                    None => self.causal_conv(&mixed, &read.conv_state)?,
                    Some(lines) => {
                        self.causal_conv_along(&mixed, &read.conv_state, &lines.parents)?
                    }
                };
                let conv = candle_nn::ops::silu(&conv)?; // [b, seq, C]
                (
                    conv.narrow(2, 0, 2 * self.key_dim())?
                        .reshape((b_sz, seq_len, 2 * nk, kd))?,
                    conv.narrow(2, 2 * self.key_dim(), self.value_dim())?
                        .reshape((b_sz, seq_len, nv, vd))?,
                    Some(new_conv),
                )
            }
        };
        let qk = match l2_norm_on_the_card(&qk, self.eps)? {
            Some(normed) => normed,
            None => l2_norm(&qk, self.eps)?,
        };

        let (out, new_state, written) = delta_rule_into(
            &qk,
            nk,
            &v,
            &gates,
            &read.recurrent_state,
            target.as_ref().map(|t| &t.recurrent_state),
            tree.as_ref(),
            None,
        )?; // out [b, seq, nv, vd]
        let gated = self.gated_output(&out, &z)?;

        let in_place = target.is_some();
        let next = written_into(target, new_conv, new_state, written)?;
        // What a rewind into this step replays (`SsmState::rewind_to`).
        held.keep_rows(
            SsmRows {
                mixed,
                qk,
                v: v.contiguous()?,
                g: gates.g,
                beta: gates.beta,
            },
            in_place,
        )?;
        held.advance(next, index_pos + seq_len);
        if tree.is_some() {
            held.mark_branched();
        }
        self.ssm_out.forward(&gated)
    }

    /// The delta rule's outputs `out` `[b, seq, nv, vd]` through the gated
    /// RMSNorm — RMSNorm(o, ssm_norm) · silu(z), per value head, `z` the
    /// step's `[b, seq, nv·vd]` — in the order `ssm_out` reads: what
    /// `ssm_out` is applied to.
    fn gated_output(&self, out: &Tensor, z: &Tensor) -> CandleResult<Tensor> {
        let (b_sz, seq_len, nv, vd) = out.dims4()?;
        let nk = self.n_k_heads;
        let z = z.reshape((b_sz, seq_len, nv, vd))?;
        let normed = self.ssm_norm.forward(out)?;
        let gated = (normed * candle_nn::ops::silu(&z)?)?;
        // `ssm_out` reads the file's head order (`ValueHeadOrder`): a grouped
        // file's kernel head `j·nk + kh` is its `kh·r + j`.
        let gated = match self.out_heads {
            ValueHeadOrder::Tiled => gated,
            ValueHeadOrder::GroupedByKeyHead => gated
                .reshape((b_sz, seq_len, nv / nk, nk, vd))?
                .transpose(2, 3)?
                .contiguous()?,
        };
        gated.reshape((b_sz, seq_len, nv * vd))
    }

    /// A forward of a tree that FLOWS (`pipeline::dsd_flow`,
    /// `docs/plans/recurrent_flow.md`): its rows are the tree's LAST, their
    /// parents rows of this forward or of earlier ones, `walk` the tree row the
    /// walk stood on when it was sent. Each row's state is rebuilt from the
    /// ANCHOR — the state after the walk's row, final — along its line: the
    /// rows of earlier forwards on the new rows' lines (their kept inputs, never
    /// projected again) and the new rows, read as ONE tree from the anchor by
    /// the step's own convolution and delta rule ([`Self::causal_conv_along`],
    /// [`delta_rule_into`]), so each row's arithmetic is its line's alone and
    /// its output the one a chain read of that line writes. Then the anchor
    /// moves to the walk's row ([`FlowRows::advance_to`]) and only rows
    /// following it are kept.
    ///
    /// A row whose line leaves the anchor's subtree is one the walk can no
    /// longer reach (its draw is never read): it is read from the anchor, for a
    /// finite output, and kept by nothing.
    ///
    /// Prior art: ReplaySSM (Dao AI Lab, 2026) caches each step's inputs and
    /// replays from one checkpoint; STree (arXiv 2505.14969) and TreeWY (arXiv
    /// 2608.20961) keep per-row activations and rebuild the accepted state along
    /// the accepted path; SGLang starts each tree node from its parent's state.
    /// All verify one tree and commit before the next is sent; here a dozen
    /// forwards are in flight, so the checkpoint follows the walk instead.
    fn forward_flowing(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
        index_pos: usize,
        walk: usize,
    ) -> CandleResult<Tensor> {
        let (_b_sz, seq_len, _hidden) = x.dims3()?;
        let Some(seen) = super::tree_scope_now().and_then(|scope| scope.visibility) else {
            candle_core::bail!("a flowing forward with no tree");
        };
        let (before, parents) = seen
            .parents_across_forwards()
            .map_err(candle_core::Error::msg)?;
        if parents.len() != seq_len || seen.base + before != index_pos {
            candle_core::bail!(
                "a flowing tree of {} rows from {} after {before} of them, in a DeltaNet step of {seq_len} at {index_pos}",
                parents.len(),
                seen.base
            );
        }
        let Some(held) = ssm_state.as_mut() else {
            candle_core::bail!("a flowing tree reached a DeltaNet layer with no state");
        };
        held.flow_from(seen.base, index_pos)?;
        let flow = held.flow_mut().expect("set by flow_from");
        flow.check_walk(walk, before, &parents)?;

        let [mixed, z]: [Tensor; 2] = QMatMul::forward_shared(x, &[&self.wqkv, &self.wz])?
            .try_into()
            .map_err(|_| candle_core::Error::Msg("two projections, two outputs".into()))?;
        let gates = self.gates(x)?;
        // The walk's row may be one an earlier forward read: the anchor goes
        // there first, so fewer rows are read again.
        flow.advance_to(walk)?;
        let lines = flow.lines(before, &parents);
        let device = x.device();
        let ahead = lines.earlier.len();
        // The rows the lines read — the earlier rows on them, then the new
        // ones — gathered from the store of kept rows once each kind's new rows
        // are PUT there (`FlowRows::put`): copied once, in place after the rows
        // written, the held rows never copied again; one gather a kind. Joining
        // the held rows to the new ones anew each forward was two copies a kind
        // a layer, 202 of a drafter level's 705 launches with the gathers 100
        // (2026-11-22, Qwen3.5-0.8B, `SWARMLLM_COUNT_KERNELS`). Copies and
        // gathers only, so the same bits every way. A/B `SWARMLLM_FLOW_STORE=
        // joined` (that), `=per-layer` (before it).
        let mode = flow_rows_mode();
        flow.make_room(seq_len)?;
        let earlier = match mode {
            FlowRowsMode::PerLayer => flow.inputs_of(&lines.earlier, device)?,
            FlowRowsMode::Stored | FlowRowsMode::Joined => None,
        };
        let read_rows = match mode {
            FlowRowsMode::PerLayer => None,
            FlowRowsMode::Stored | FlowRowsMode::Joined => {
                flow.rows_the_lines_read(&lines.earlier, seq_len, capturing(device), device)?
            }
        };
        // On a card the convolution and the delta rule read each kind's rows
        // where the store keeps them, through the table (`SlotTable`): the
        // gather a kind and the gates' stack were 7 of a layer's 12 copies, 126
        // launches of a drafter level (2026-11-23). The same values in the same
        // order: the same bits. A/B `SWARMLLM_FLOW_STORE=gathered`.
        let in_place = read_rows
            .as_ref()
            .filter(|_| device.is_cuda() && flow_reads_in_place());
        // Where the new rows go, as the card reads it: the table's tail, so
        // each kind is put in one launch however its slots fall (`put_rows_at`).
        let new_on_card = match &read_rows {
            Some(at) if device.is_cuda() => Some(at.table.narrow(0, ahead, seq_len)?),
            _ => None,
        };
        // A kind's new rows put in the store, and what the lines read of it:
        // read in place, every row written; else the lines' rows gathered.
        let read = |flow: &mut FlowRows, kind: RowKind, new: &Tensor| -> CandleResult<Tensor> {
            let all = flow.put(kind, new, new_on_card.as_ref())?;
            match (&earlier, &read_rows) {
                (Some(earlier), _) => Tensor::cat(&[kind.of(earlier), new], 1),
                (None, Some(_)) if in_place.is_some() => Ok(all),
                (None, Some(at)) => at.gather(&all),
                (None, None) => Ok(new.clone()),
            }
        };
        let (b_sz, nk, kd, nv, vd) = (
            mixed.dim(0)?,
            self.n_k_heads,
            self.k_head_dim,
            self.n_v_heads,
            self.v_head_dim,
        );
        let anchor = flow.anchor().clone();
        let tree = TreeLines::for_flow(lines.parents.clone(), device)?;
        let rows = ahead + seq_len;
        let all_mixed = read(flow, RowKind::Mixed, &mixed)?;
        // On a card the convolution and its SiLU are one launch along the
        // tree's lines (`ssm_conv_silu_tree_f32_k4`, bit for bit the
        // composition); the window it writes is not this forward's to keep.
        // Written, never read: the flow's scratch pair where it has one
        // (`FlowRows::stage_level`), else memory made for it.
        let scratch = flow.scratch.clone();
        let window = match (device.is_cuda(), &scratch) {
            (true, Some(scratch)) => Some(scratch.conv_state.clone()),
            (true, None) => Some(anchor.conv_state.zeros_like()?),
            (false, _) => None,
        };
        let fused = match &window {
            Some(window) => self.conv_on_the_card(
                &all_mixed,
                &anchor.conv_state,
                window,
                Some(&tree),
                in_place,
            )?,
            None => None,
        };
        let (qk, v) = match fused {
            Some((qk, v)) => (
                qk.reshape((b_sz, rows, 2 * nk, kd))?
                    .narrow(1, ahead, seq_len)?,
                v.reshape((b_sz, rows, nv, vd))?.narrow(1, ahead, seq_len)?,
            ),
            None => {
                // The composition uploads its columns: never inside a capture.
                #[cfg(feature = "candle-cuda")]
                if stream_is_capturing(device) {
                    candle_core::bail!("a captured flowing level reached the composed convolution");
                }
                // Read in place, the composition reads the lines' rows themselves.
                let all_mixed = match in_place {
                    Some(at) => at.gather(&all_mixed)?,
                    None => all_mixed.clone(),
                };
                let (conv, _) =
                    self.causal_conv_along(&all_mixed, &anchor.conv_state, &lines.parents)?;
                let conv = candle_nn::ops::silu(&conv.narrow(1, ahead, seq_len)?)?;
                (
                    conv.narrow(2, 0, 2 * self.key_dim())?
                        .reshape((b_sz, seq_len, 2 * nk, kd))?,
                    conv.narrow(2, 2 * self.key_dim(), self.value_dim())?
                        .reshape((b_sz, seq_len, nv, vd))?,
                )
            }
        };
        let qk = match l2_norm_on_the_card(&qk.contiguous()?, self.eps)? {
            Some(normed) => normed,
            None => l2_norm(&qk, self.eps)?,
        };
        let v = v.contiguous()?;
        let (out, _, _) = delta_rule_into(
            &read(flow, RowKind::Qk, &qk)?,
            nk,
            &read(flow, RowKind::V, &v)?,
            &Gates::apart(
                read(flow, RowKind::G, &gates.g)?,
                read(flow, RowKind::Beta, &gates.beta)?,
            ),
            &anchor.recurrent_state,
            scratch.as_ref().map(|s| &s.recurrent_state),
            Some(&tree),
            in_place,
        )?;
        let gated = self.gated_output(&out.narrow(1, ahead, seq_len)?, &z)?;
        flow.keep_appended(before, &parents, &lines.live);
        flow.advance_to(walk)?;
        held.note_flow_read();
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

    /// [`Self::causal_conv`] over a TREE's rows (#437): row i's taps read up
    /// its own line — itself, its parent, its parent's parent — and past the
    /// tree's first row the window before the step, last input first, where
    /// a chain's read the rows before it (Twig, arXiv 2608.01651: a node's
    /// convolution inputs are its root-to-node path, found by parent links).
    /// The taps are summed in the chain's order, so a row whose line IS the
    /// rows before it gets the chain's bits. The window returned is the
    /// chain's — a branching tree's state is taken back before it is read
    /// ([`SsmState::mark_branched`]).
    pub(crate) fn causal_conv_along(
        &self,
        x: &Tensor,
        prev: &Tensor,
        parents: &[Option<usize>],
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_b_sz, seq_len, channels) = x.dims3()?;
        if parents.len() != seq_len {
            candle_core::bail!("a tree of {} rows for a step of {seq_len}", parents.len());
        }
        let k = self.conv_kernel;
        let pad = k - 1;
        // `[tap, row]`: the column of `padded` (the window, then the step's
        // rows) tap j of row i reads — the input `pad − j` steps up i's line.
        let mut columns = vec![0u32; k * seq_len];
        for i in 0..seq_len {
            let (mut at, mut above) = (Some(i), 0usize);
            for back in 0..k {
                let column = match at {
                    Some(r) => pad + r,
                    None => {
                        above += 1;
                        pad - above
                    }
                };
                columns[(pad - back) * seq_len + i] = column as u32;
                at = at.and_then(|r| parents[r]);
            }
        }
        let columns = Tensor::from_vec(columns, (k, seq_len), x.device())?;
        let x_t = x.transpose(1, 2)?.contiguous()?; // [b, C, seq]
        let padded = Tensor::cat(&[prev, &x_t], 2)?; // [b, C, seq + K − 1]
        let w = self.conv1d.reshape((1, channels, k))?;
        let mut acc: Option<Tensor> = None;
        for j in 0..k {
            let tap = padded
                .index_select(&columns.get(j)?, 2)?
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

    /// [`Self::causal_conv`] and its SiLU in one launch on a card
    /// (`kernels/delta_rule.cu` `ssm_conv_silu_f32_k4`, bit for bit the
    /// composition): q | k `[b, seq, 2·key_dim]` and v `[b, seq, value_dim]`,
    /// each contiguous, with the window after the step written into
    /// `state_out` (the step's own buffers). `None`, nothing written, off a
    /// card build, outside the kernel's scope (one sequence, f32, a kernel of
    /// 4) or under `SWARMLLM_SSM_CONV=composed`, the A/B switch.
    ///
    /// Prior art: llama.cpp fuses the same pair on a card
    /// (`ggml_cuda_op_ssm_conv` with `apply_silu`, read at `4b1a27f`); the
    /// composition made 12 launches a layer at one position — ~290 a token of
    /// Qwen3.5-4B's 24 such layers (#428, card profile, 2026-11-08).
    ///
    /// `slots` (a flowing forward reading its store in place, [`SlotTable`]):
    /// `x` is the store, and row t's input its row `slots[t]`.
    fn conv_on_the_card(
        &self,
        x: &Tensor,
        prev: &Tensor,
        state_out: &Tensor,
        tree: Option<&TreeLines>,
        slots: Option<&SlotTable>,
    ) -> CandleResult<Option<(Tensor, Tensor)>> {
        #[cfg(feature = "candle-cuda")]
        {
            if !conv_in_one_launch() || self.conv_kernel != 4 || !x.device().is_cuda() {
                return Ok(None);
            }
            // A tree's rows read up their own lines (`ssm_conv_silu_tree_f32_k4`),
            // never the chain's kernel.
            let parents = match tree {
                None => None,
                Some(TreeLines {
                    on_card: Some(p), ..
                }) => Some(p.clone()),
                Some(_) => return Ok(None),
            };
            // Read in place, x is the flow's store and the rows the table's.
            let seq_len = match slots {
                Some(at) => at.rows()?,
                None => x.dim(1)?,
            };
            let split = 2 * self.key_dim();
            let op = cuda::ConvSilu {
                state_out: state_out.clone(),
                split,
                parents,
                slots: slots.cloned(),
            };
            if !op.takes(x, prev, &self.conv1d) {
                return Ok(None);
            }
            let out = x.apply_op3_no_bwd(prev, &self.conv1d, &op)?;
            let qk = out.narrow(0, 0, seq_len * split)?;
            let v = out.narrow(0, seq_len * split, seq_len * self.value_dim())?;
            Ok(Some((qk, v)))
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            let _ = (x, prev, state_out, tree, slots);
            Ok(None)
        }
    }

    /// A step's gates: the log-decay g = softplus(alpha + dt_bias) · a (a =
    /// −exp(A_log), so g ≤ 0) and the write strength beta = sigmoid(beta).
    /// On a card in one launch, written in the block layout the delta rule
    /// reads ([`Gates::stacked`]): the projections too where the file keeps
    /// `ssm_alpha` and `ssm_beta` in f32 — or in Q8_0, read in place (the
    /// 9B's and the 0.8B/2B drafters' uploads) — and the step is at most 8
    /// rows (NOT bit-identical to cuBLAS or to the quantized matmuls; a row's
    /// gates are the same at any width), else after them, bit for bit the
    /// composition. `SWARMLLM_SSM_GATES=composed` is the A/B switch,
    /// `SWARMLLM_SSM_GATES=matmuls` the Q8_0 one's.
    ///
    /// Prior art: llama.cpp keeps a small batch's f32 projection off cuBLAS
    /// (`ggml-cuda/mmvf.cu`, up to 8 columns). Composed, these were ~15
    /// launches a layer and 0.65 ms of a Qwen3.5-4B step's 11.4 on the card,
    /// and cuBLAS's `gemmSN` 0.47 ms more in a 2-row check (#428, card trace,
    /// 2026-11-08).
    fn gates(&self, x: &Tensor) -> CandleResult<Gates> {
        #[cfg(feature = "candle-cuda")]
        if x.device().is_cuda() && gates_in_one_launch() {
            if let Some(gates) = cuda::gates(self, x)? {
                return Ok(gates);
            }
        }
        Gates::composed(
            &self.w_beta.forward(x)?,
            &self.w_alpha.forward(x)?,
            &self.dt_bias,
            &self.a,
        )
    }
}

/// A DeltaNet step's gates, `[b, seq, nv]` each: the log-decay `g` and the
/// write strength `beta` ([`DeltaNetWeights::gates`]).
struct Gates {
    g: Tensor,
    beta: Tensor,
    /// `[2, b, seq, nv]`, g's block then beta's, when a card wrote them so —
    /// `g` and `beta` are then its two halves.
    #[cfg(feature = "candle-cuda")]
    stacked: Option<Tensor>,
}

impl Gates {
    /// From the two projections' outputs, as candle ops: `sigmoid(beta)` and
    /// `softplus(alpha + dt_bias) · a`.
    fn composed(
        beta_raw: &Tensor,
        alpha_raw: &Tensor,
        dt_bias: &Tensor,
        a: &Tensor,
    ) -> CandleResult<Self> {
        let beta = candle_nn::ops::sigmoid(beta_raw)?;
        let g = softplus(&alpha_raw.broadcast_add(dt_bias)?)?.broadcast_mul(a)?;
        Ok(Self::apart(g, beta))
    }

    /// Gates kept apart (a check's kept rows), stacked when a card asks.
    fn apart(g: Tensor, beta: Tensor) -> Self {
        Self {
            g,
            beta,
            #[cfg(feature = "candle-cuda")]
            stacked: None,
        }
    }

    /// `[2, b, seq, nv]`, g's block then beta's: what the card's delta rule
    /// reads — the buffer a card wrote, else the two stacked.
    #[cfg(feature = "candle-cuda")]
    fn stacked(&self) -> CandleResult<Tensor> {
        match &self.stacked {
            Some(stacked) => Ok(stacked.clone()),
            None => Tensor::stack(&[&self.g, &self.beta], 0),
        }
    }
}

/// [`l2_norm`] in one launch on a card (`l2_norm_rows_f32_d*`, a warp a row).
/// `None` off a card build, outside its scope (f32, contiguous, rows of 64 or
/// 128) or under `SWARMLLM_SSM_NORM=composed`, the A/B switch. NOT
/// bit-identical — the sum runs in another order; a row reads only itself.
///
/// Prior art: llama.cpp's card L2 norm is one launch a row (`ggml-cuda/norm.cu`
/// `l2_norm_f32`); the composition made five a layer (#428, card trace,
/// 2026-11-08).
fn l2_norm_on_the_card(x: &Tensor, eps: f64) -> CandleResult<Option<Tensor>> {
    #[cfg(feature = "candle-cuda")]
    {
        let width = x.dims().last().copied().unwrap_or(0);
        if !norm_in_one_launch()
            || !x.device().is_cuda()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
            || !matches!(width, 64 | 128)
        {
            return Ok(None);
        }
        Ok(Some(
            x.apply_op1_no_bwd(&cuda::L2NormRows { eps: eps as f32 })?,
        ))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (x, eps);
        Ok(None)
    }
}

/// `SWARMLLM_SSM_GATES=composed` → a DeltaNet step's gates as candle ops on a
/// card too: the A/B switch for [`DeltaNetWeights::gates`]. Read once.
#[cfg(feature = "candle-cuda")]
fn gates_in_one_launch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SSM_GATES").as_deref(),
            Ok("composed")
        )
    })
}

/// `SWARMLLM_SSM_GATES=matmuls` → a step's gates from Q8_0 `ssm_beta` and
/// `ssm_alpha` take the projections as matmuls (a quantize and an int8 product
/// each) before the gates' launch, as before #228 (1) (b): the A/B switch for
/// the one launch that reads the Q8_0 weights in place. `composed` turns it
/// off too. Read once.
#[cfg(feature = "candle-cuda")]
fn gates_project_q8_0() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SSM_GATES").as_deref(),
            Ok("matmuls" | "composed")
        )
    })
}

/// `SWARMLLM_SSM_NORM=composed` → q and k's L2 norm as candle ops on a card
/// too: the A/B switch for [`l2_norm_on_the_card`]. Read once.
#[cfg(feature = "candle-cuda")]
fn norm_in_one_launch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SSM_NORM").as_deref(),
            Ok("composed")
        )
    })
}

/// `SWARMLLM_SSM_CONV=composed` → a DeltaNet layer's convolution and SiLU as
/// candle ops on a card too: the A/B switch for
/// [`DeltaNetWeights::conv_on_the_card`]. Read once.
#[cfg(feature = "candle-cuda")]
fn conv_in_one_launch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SSM_CONV").as_deref(),
            Ok("composed")
        )
    })
}

/// The new state as a step or a replay leaves it: on a card in `target`, the
/// buffers it keeps (the delta rule's kernel `written` its state there
/// already, and a fused convolution its window — `conv_state` `None`), on the
/// processor the tensors made.
fn written_into(
    target: Option<SsmSlot>,
    conv_state: Option<Tensor>,
    recurrent_state: Tensor,
    written: bool,
) -> CandleResult<SsmSlot> {
    Ok(match target {
        Some(t) => {
            if let Some(conv_state) = conv_state {
                t.conv_state.slice_set(&conv_state, 0, 0)?;
            }
            if !written {
                t.recurrent_state
                    .slice_set(&recurrent_state.contiguous()?, 0, 0)?;
            }
            t
        }
        None => SsmSlot {
            conv_state: conv_state.ok_or_else(|| {
                candle_core::Error::Msg("a convolution window written into no buffer".into())
            })?,
            recurrent_state,
        },
    })
}

/// A buffer shaped as `like` for a replay to write WHOLE: on a card
/// `cuda::replay`'s two kernels assign every element of both of a target's
/// buffers, and the composition sets both whole ([`written_into`]) — so it
/// needs no memset (`CudaDevice::alloc_fully_overwritten`). Elsewhere zeros.
/// Test `a_replay_on_the_card_lands_where_the_composed_replay_does` hands
/// both paths NaN-filled targets. A/B `SWARMLLM_SSM_REPLAY_TARGET=zeroed`.
///
/// Prior art: llama.cpp's `gated_delta_net.cu` / `ssm-conv.cu` write outputs
/// ggml's allocator hands them unzeroed (its pool never clears), as PyTorch's
/// `torch.empty`; zeros are for a buffer read before it is written.
fn unwritten_like(like: &Tensor) -> CandleResult<Tensor> {
    #[cfg(feature = "candle-cuda")]
    if !replay_targets_zeroed() {
        if let Some(fresh) = cuda::unwritten_like(like)? {
            return Ok(fresh);
        }
    }
    like.zeros_like()
}

/// The delta rule over a step's rows, from `state` — the one choice of kernel
/// a step and a replay share, so a replay's state is the step's: the card
/// kernel (written straight into `into`, the third answer `true`), the
/// processor's one pass, or the token-by-token composition
/// (`SWARMLLM_DELTA_RULE=steps`). q and k are `qk` `[b, seq, 2·nk, kd]`,
/// L2-normalised; the scale is `1/√kd`. Returns the outputs
/// `[b, seq, nv, vd]` and the new state.
///
/// `slots` (a flowing forward reading its stores in place, [`SlotTable`]):
/// `qk`, `v` and the gates are the stores, row t's inputs their row
/// `slots[t]` — read so only by the card's flowing kernel, gathered for any
/// other way.
#[allow(clippy::too_many_arguments)]
fn delta_rule_into(
    qk: &Tensor,
    nk: usize,
    v: &Tensor,
    gates: &Gates,
    state: &Tensor,
    into: Option<&Tensor>,
    tree: Option<&TreeLines>,
    slots: Option<&SlotTable>,
) -> CandleResult<(Tensor, Tensor, bool)> {
    let (g, beta) = (&gates.g, &gates.beta);
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
            delta_rule_cuda(
                qk,
                nk,
                v,
                gates,
                state,
                &out_buffer,
                scale as f32,
                tree,
                slots,
            )?
            .map(|out| (out, out_buffer, into.is_some()))
        } else {
            None
        };
        // A store read in place reaches only the card's flowing kernel: any
        // other way reads the lines' rows themselves.
        let gathered = match (&on_card, slots) {
            (None, Some(at)) => Some((
                at.gather(qk)?,
                at.gather(v)?,
                at.gather(g)?,
                at.gather(beta)?,
            )),
            _ => None,
        };
        let (qk, v, g, beta) = match &gathered {
            Some((qk, v, g, beta)) => (qk, v, g, beta),
            None => (qk, v, g, beta),
        };
        let tree = tree.map(|lines| lines.parents.as_slice());
        let (q, k) = (qk.narrow(2, 0, nk)?, qk.narrow(2, nk, nk)?);
        Ok(match on_card {
            Some(done) => done,
            None if one_pass && device.is_cpu() => {
                let (out, new) = delta_rule_cpu(&q, &k, v, g, beta, state, scale as f32, tree)?;
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
                let (out, new) = delta_rule(&(q * scale)?, &k, v, g, beta, state.clone(), tree)?;
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
    // On a card, two launches a layer — the step's kernel needs the
    // one-pass rule, as here (`delta_rule_into`).
    #[cfg(feature = "candle-cuda")]
    if let Some(into) = into.filter(|_| replay_in_two_launches() && delta_rule_in_one_pass()) {
        if cuda::replay(read, kept, rows, into)? {
            return Ok(into.clone());
        }
    }
    replay_rows_composed(read, kept, rows, into)
}

/// [`replay_rows`] as candle ops: the processor's, and on a card the A/B arm
/// (`SWARMLLM_SSM_REPLAY=composed`) and whatever the two-launch replay does
/// not take.
fn replay_rows_composed(
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
        &Gates::apart(first(&kept.g)?, first(&kept.beta)?),
        &read.recurrent_state,
        into.map(|t| &t.recurrent_state),
        None,
        None,
    )?;
    written_into(into.cloned(), Some(conv_state), recurrent_state, written)
}

/// What a DeltaNet layer keeps of a tree that FLOWS through it
/// ([`DeltaNetWeights::forward_flowing`], `docs/plans/recurrent_flow.md`):
/// the tree's rows sit in the cache in the order SENT, a forward's rows have
/// parents in earlier forwards, and the requester's walk moves down the tree
/// while a dozen forwards are in flight.
///
/// - **The anchor**: the state after the row the walk stands on — final, every
///   row before it on its line is the reply's — and the state before it (after
///   its parent), which is the state at the start of a rebased tree whose
///   first row the walk's row becomes ([`Self::rebase_onto`]).
/// - **The inputs of every row following the anchor** (`kept`): what the
///   convolution and the delta rule read ([`SsmRows`]), never a state — 80 KB
///   a row a layer on Qwen3.8-27B against a state's 3 MiB. A row's state is
///   rebuilt from the anchor along its line when a later row needs it.
/// - **The states at the tree's starts**: what a cut back to a start puts
///   back — a rebase's, or a failed flow's, which cuts to the last start every
///   segment answered (`dsd_flow::FlowRun::settled_base`): the current one or,
///   with a rebase unanswered, the one before it.
#[derive(Debug, Clone)]
pub(crate) struct FlowRows {
    /// The cache position of tree row 0.
    base: usize,
    /// The states at the starts a cut may go back to, oldest first, at most
    /// two: `(position, state)`.
    starts: Vec<(usize, SsmSlot)>,
    /// Tree rows read so far: the cache holds `base + read` positions.
    read: usize,
    /// Every row read's parent, as a tree row.
    parents: Vec<Option<usize>>,
    /// The row the walk stands on; `None` before it reached one (the anchor
    /// is then the state at `base`).
    anchor_row: Option<usize>,
    anchor: SsmSlot,
    /// The state before `anchor_row` — after its parent.
    before_anchor: SsmSlot,
    /// The rows following the anchor's, ascending, and where each one's
    /// inputs are in the store.
    kept: Vec<usize>,
    slots: Vec<u32>,
    /// Slots below the store's rows written that held no kept row when this
    /// forward BEGAN (`SsmState::flow_from` → [`Self::note_begun`]),
    /// ascending: a forward's new rows go there first ([`Self::new_slots`]).
    /// Only those: a slot the walk leaves DURING a forward is still a row of
    /// its rollback copy (`SsmState::flow_begun`, which shares the store's
    /// buffers), and is written again from the next forward on. Empty in
    /// the A/B arms (`SWARMLLM_FLOW_SLOTS=compacted`, the rows joined anew).
    free: Vec<u32>,
    store: RowStore,
    /// On a card, where a forward's convolution writes its window and its
    /// delta rule the state after its last row — neither read: a tree's end is
    /// no state the reply holds. Made once, outside any capture
    /// ([`Self::stage_level`]); a forward made two memsets a layer without it,
    /// and a captured one may make none (#221). ONE pair every layer of the
    /// conversation shares (`layers::stage_flowing_states` lends it): written,
    /// never read, by kernels one after another on the device's one stream —
    /// llama.cpp likewise keeps one zeroed cell (`rs_z`) for every empty
    /// sequence of a batch to read, not one each.
    scratch: Option<SsmSlot>,
}

/// A flowing forward's rows read as one tree from the anchor
/// ([`FlowRows::lines`]).
pub(super) struct FlowLines {
    /// Kept rows on the new rows' lines, ascending: read again ahead of them.
    pub(super) earlier: Vec<usize>,
    /// Each row's parent among `earlier` then the new rows; `None` for a row
    /// following the anchor — and for a row whose line left it.
    pub(super) parents: Vec<Option<usize>>,
    /// Which new rows follow the anchor: the rows kept.
    pub(super) live: Vec<bool>,
}

impl FlowRows {
    /// A flow whose tree starts at `base`, from the state there.
    pub(super) fn new(base: usize, at_base: SsmSlot) -> Self {
        Self {
            base,
            starts: vec![(base, at_base.clone())],
            read: 0,
            parents: Vec::new(),
            anchor_row: None,
            anchor: at_base.clone(),
            before_anchor: at_base,
            kept: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
            store: RowStore::default(),
            scratch: None,
        }
    }

    /// A forward begins (`SsmState::flow_from`, as it takes the rollback
    /// copy): the slots no kept row holds now are the ones its rows may be
    /// written into ([`Self::free`]).
    pub(super) fn note_begun(&mut self) {
        self.free.clear();
        if !flow_slots_reused() {
            return;
        }
        let mut held = vec![false; self.store.len];
        for &slot in &self.slots {
            if let Some(h) = held.get_mut(slot as usize) {
                *h = true;
            }
        }
        self.free
            .extend((0..self.store.len as u32).filter(|&s| !held[s as usize]));
    }

    /// Rows a forward may put in the store without making room
    /// ([`Self::make_room`]): its free slots and the rows past those written.
    fn room(&self) -> usize {
        self.free.len() + self.store.capacity.saturating_sub(self.store.len)
    }

    /// The room the NEXT forward will find once it begins ([`Self::note_begun`]):
    /// every slot no kept row holds — what the claim, made before the forward
    /// begins (`SplitModel::claim_room_for_forward`), is weighed by.
    fn room_for_the_next_forward(&self) -> usize {
        if flow_slots_reused() {
            self.store.capacity.saturating_sub(self.slots.len())
        } else {
            self.store.capacity.saturating_sub(self.store.len)
        }
    }

    /// Where a forward's `rows` new rows go in the store: the free slots
    /// first, ascending, then past the rows written. The same answer for
    /// every kind's [`Self::put`], the tables ([`Self::rows_the_lines_read`])
    /// and [`Self::keep_appended`]: nothing between them changes the store.
    fn new_slots(&self, rows: usize) -> Vec<u32> {
        let taken = rows.min(self.free.len());
        let past = self.store.len as u32;
        self.free[..taken]
            .iter()
            .copied()
            .chain((0..(rows - taken) as u32).map(|i| past + i))
            .collect()
    }

    /// The store's rows written once a forward's `rows` are put: the free
    /// slots they take hold no new place.
    fn written_after(&self, rows: usize) -> usize {
        self.store.len + rows - rows.min(self.free.len())
    }

    /// The cache position of the tree's first row.
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    /// The cache's length: the tree's start and the rows read.
    pub(crate) fn positions(&self) -> usize {
        self.base + self.read
    }

    /// Rows of inputs a forward of `rows` new rows may newly allocate, a kind:
    /// none where the store has room for them, else the rows by which
    /// [`Self::make_room`] grows it, the store it replaces being counted as
    /// held already (`tensors`). That one forward holds both: the forward's
    /// rollback copy (`SsmState::flow_begun`) keeps the old store until the
    /// next forward begins, so its peak is above the claim by the old store —
    /// inside the budget's margin in every arm measured (2026-11-23), and
    /// FUTURE_WORK #228's residual. Claimed whole on top of the old, a full store's growth
    /// was refused on the drafter at 268 + 65 MB of a 308 MB budget and on the
    /// 9B's head at 308 + 184 of 491, and the reply went on at a third of its
    /// speed (2026-11-23, TH↔AU). Joined anew (the A/B arms), the rows written
    /// and the new ones. `None` for a flow not yet begun. Asked BEFORE the
    /// forward begins (both claim sites), so the room is what that forward
    /// will find (`room_for_the_next_forward`): its free slots counted.
    pub(crate) fn rows_a_forward_may_allocate(flow: Option<&Self>, rows: usize) -> usize {
        let (len, capacity, kept, room) = flow.map_or((0, 0, 0, 0), |f| {
            (
                f.store.len,
                f.store.capacity,
                f.kept.len(),
                f.room_for_the_next_forward(),
            )
        });
        match flow_rows_mode() {
            FlowRowsMode::Stored if rows <= room => 0,
            FlowRowsMode::Stored => store_capacity_for(kept + rows).saturating_sub(capacity),
            FlowRowsMode::Joined | FlowRowsMode::PerLayer => len + rows,
        }
    }

    /// The state at cache position `positions`, where a cut may put it: a
    /// start, or the start of the tree a rebase onto the walk's row would make
    /// (the walk's line a chain from the tree's first row).
    pub(crate) fn state_at(&self, positions: usize) -> Option<&SsmSlot> {
        if let Some((_, state)) = self.starts.iter().rev().find(|(at, _)| *at == positions) {
            return Some(state);
        }
        let row = self.anchor_row?;
        (positions == self.base + row && self.line_is_chain(row)).then_some(&self.before_anchor)
    }

    pub(super) fn anchor(&self) -> &SsmSlot {
        &self.anchor
    }

    /// The scratch pair, where a captured level made one ([`Self::stage_level`]).
    pub(super) fn scratch(&self) -> Option<&SsmSlot> {
        self.scratch.as_ref()
    }

    /// The oldest start kept, where no start shares `slot`'s buffers — what
    /// `SsmState::flow_from` points its `current` at once a rebase has let go
    /// of the start it shared ([`Self::rebase_onto`] keeps two).
    pub(super) fn oldest_start_unless_holding(&self, slot: &SsmSlot) -> Option<SsmSlot> {
        let held = slot.recurrent_state.id();
        if self
            .starts
            .iter()
            .any(|(_, start)| start.recurrent_state.id() == held)
        {
            return None;
        }
        self.starts.first().map(|(_, start)| start.clone())
    }

    /// Every tensor kept, for a cache's accounting.
    pub(crate) fn tensors(&self) -> Vec<&Tensor> {
        let mut out: Vec<&Tensor> = self
            .starts
            .iter()
            .flat_map(|(_, s)| [&s.conv_state, &s.recurrent_state])
            .collect();
        for slot in [&self.anchor, &self.before_anchor] {
            out.extend([&slot.conv_state, &slot.recurrent_state]);
        }
        out.extend(self.store.kinds.iter().flatten());
        if let Some(scratch) = &self.scratch {
            out.extend([&scratch.conv_state, &scratch.recurrent_state]);
        }
        out
    }

    /// The store, in words: what a DIAG line says
    /// of a flowing level run uncaptured.
    pub(super) fn why_no_level_of(&self, rows: usize) -> String {
        format!(
            "{rows} rows into a store of {} rows holding {} ({:?}, every kind made: {})",
            self.store.capacity,
            self.store.len,
            flow_rows_mode(),
            self.store.kinds.iter().all(Option::is_some),
        )
    }

    /// May a forward be readied to be CAPTURED ([`Self::stage_level`]) — the
    /// store made? A full one is given room there, before the capture: run
    /// the ordinary way instead, one level in ten of a reply's ran uncaptured
    /// (91 of a drafter's levels, Qwen3.5-0.8B on the TH↔AU link, 2026-11-23).
    /// `SsmState::holds_a_step_to`.
    pub(super) fn holds_a_level(&self) -> bool {
        flow_rows_mode() == FlowRowsMode::Stored && self.store.kinds.iter().all(Option::is_some)
    }

    /// Readies a forward of `seq_len` new rows (`parents`, the first at tree
    /// row `before`), once the anchor is at the walk's row ([`Self::advance_to`]),
    /// to be CAPTURED: everything it would make outside a kernel is made here,
    /// before the capture — the scratch pair, and the row tables its layers read
    /// ([`rows_on`]: the rows the lines read, the tree's parents, the walks),
    /// which inside the capture are hits — and room in a full store
    /// ([`Self::make_room`], claimed before: `claim_room_for_forward`). False
    /// where the forward would make more inside it: the rows joined anew (the
    /// A/B arms), or a walk standing on one of the forward's own rows — the
    /// anchor then moves AFTER the rows are kept, replaying them.
    pub(super) fn stage_level(
        &mut self,
        walk: usize,
        before: usize,
        parents: &[Option<usize>],
        seq_len: usize,
        device: &Device,
        shared_scratch: &mut Option<SsmSlot>,
    ) -> CandleResult<bool> {
        if !device.is_cuda() || walk >= before || !self.holds_a_level() {
            tracing::debug!(
                walk,
                before,
                why = %self.why_no_level_of(seq_len),
                "DIAG: a flowing level runs uncaptured"
            );
            return Ok(false);
        }
        self.make_room(seq_len)?;
        if self.scratch.is_none() {
            // The pair the other layers write, where it is this layer's shape:
            // one a layer was the whole state again — 35 pairs, 114 MB, on
            // the 27B's far card (US [18, 64), 2026-11-27), made outside any
            // claim.
            let like = |t: &Tensor, u: &Tensor| {
                t.shape() == u.shape()
                    && t.dtype() == u.dtype()
                    && t.device().same_device(u.device())
            };
            let fits = shared_scratch.as_ref().is_some_and(|s| {
                like(&s.conv_state, &self.anchor.conv_state)
                    && like(&s.recurrent_state, &self.anchor.recurrent_state)
            });
            if !fits {
                *shared_scratch = Some(SsmSlot {
                    conv_state: self.anchor.conv_state.zeros_like()?,
                    recurrent_state: self.anchor.recurrent_state.zeros_like()?,
                });
            }
            self.scratch = shared_scratch.clone();
        }
        let lines = self.lines(before, parents);
        self.rows_the_lines_read(&lines.earlier, seq_len, true, device)?;
        TreeLines::for_flow(lines.parents, device)?;
        Ok(true)
    }

    /// Is `row`'s line the tree's first rows, one after another?
    fn line_is_chain(&self, row: usize) -> bool {
        let mut at = row;
        loop {
            match self.parents.get(at) {
                Some(None) => return at == 0,
                Some(Some(p)) if *p + 1 == at => at = *p,
                _ => return false,
            }
        }
    }

    /// Does `row` follow `of` (or is it `of`)? Every row follows the start.
    fn follows(&self, row: usize, of: Option<usize>) -> bool {
        let Some(of) = of else { return true };
        let mut at = Some(row);
        while let Some(r) = at {
            if r == of {
                return true;
            }
            at = self.parents.get(r).copied().flatten();
        }
        false
    }

    /// The tree a rebase made goes on from `from` (`FlowTree::rebase`): the
    /// walk's row became its first row and the rows before it on its line —
    /// the tree's first rows, a chain — the reply. Rows are numbered from it,
    /// and the state before it is the new start's.
    pub(super) fn rebase_onto(&mut self, from: usize) -> CandleResult<()> {
        let shift = from - self.base;
        if self.anchor_row != Some(shift) || !self.line_is_chain(shift) {
            candle_core::bail!(
                "a flowing tree goes on from {from}, where the walk's row {:?} after {} is not its first",
                self.anchor_row,
                self.base
            );
        }
        let base = self.base;
        self.starts.retain(|(at, _)| *at == base);
        self.starts.push((from, self.before_anchor.clone()));
        self.parents = self.parents[shift..]
            .iter()
            .map(|p| p.and_then(|p| p.checked_sub(shift)))
            .collect();
        for row in &mut self.kept {
            *row -= shift;
        }
        self.anchor_row = Some(0);
        self.base = from;
        self.read -= shift;
        Ok(())
    }

    /// Refuses a walk this forward does not let the anchor reach: a row past
    /// the rows read with it, or one not following the anchor's — the walk
    /// only goes down the tree.
    pub(super) fn check_walk(
        &self,
        walk: usize,
        before: usize,
        parents: &[Option<usize>],
    ) -> CandleResult<()> {
        if before != self.read {
            candle_core::bail!(
                "a flowing forward after {before} rows of a tree this layer read {} of",
                self.read
            );
        }
        if walk >= before + parents.len() {
            candle_core::bail!(
                "a flowing tree's walk stands on row {walk} of {}",
                before + parents.len()
            );
        }
        let parent = |r: usize| match r.checked_sub(before) {
            Some(i) => parents[i],
            None => self.parents[r],
        };
        let mut at = Some(walk);
        while let Some(r) = at {
            if Some(r) == self.anchor_row {
                return Ok(());
            }
            if r < before && self.kept.binary_search(&r).is_err() {
                break;
            }
            at = parent(r);
        }
        if self.anchor_row.is_none() && at.is_none() {
            return Ok(());
        }
        candle_core::bail!(
            "a flowing tree's walk went to row {walk}, which does not follow row {:?}",
            self.anchor_row
        )
    }

    /// The inputs of kept rows `rows` (ascending), gathered in that order.
    pub(super) fn inputs_of(
        &self,
        rows: &[usize],
        device: &Device,
    ) -> CandleResult<Option<SsmRows>> {
        let [Some(mixed), Some(qk), Some(v), Some(g), Some(beta)] = &self.store.kinds else {
            return Ok(None);
        };
        if rows.is_empty() {
            return Ok(None);
        }
        let at = rows_on(self.slots_of(rows)?, device)?;
        let pick = |t: &Tensor| t.index_select(&at, 1);
        Ok(Some(SsmRows {
            mixed: pick(mixed)?,
            qk: pick(qk)?,
            v: pick(v)?,
            g: pick(g)?,
            beta: pick(beta)?,
        }))
    }

    /// The walk's path's inputs for [`Self::advance_to`]: where the path's
    /// rows sit one after another in the store (one row, the usual walk, is
    /// such a run), each kind's run read in place — a view, no table and no
    /// gather; otherwise gathered by [`Self::inputs_of`]. The same values
    /// either way. Five gathers and a table a recurrent layer were 1.3-1.4 ms
    /// of a 0.8B level's ~2 ms staging (pace probe marks, 2026-11-23). Only
    /// the walk's replay reads them, outside any capture: inside one, a
    /// level reads its rows through a table (`rows_the_lines_read`).
    fn path_inputs(&self, path: &[usize], device: &Device) -> CandleResult<Option<SsmRows>> {
        let [Some(mixed), Some(qk), Some(v), Some(g), Some(beta)] = &self.store.kinds else {
            return Ok(None);
        };
        let slots = self.slots_of(path)?;
        let Some(&first) = slots.first() else {
            return Ok(None);
        };
        let in_a_run = slots
            .iter()
            .enumerate()
            .all(|(i, &s)| s as usize == first as usize + i);
        if !in_a_run || path_inputs_gathered() {
            return self.inputs_of(path, device);
        }
        let run = |t: &Tensor| t.narrow(1, first as usize, slots.len());
        Ok(Some(SsmRows {
            mixed: run(mixed)?,
            qk: run(qk)?,
            v: run(v)?,
            g: run(g)?,
            beta: run(beta)?,
        }))
    }

    /// Where kept rows `rows` are in the store.
    fn slots_of(&self, rows: &[usize]) -> CandleResult<Vec<u32>> {
        rows.iter()
            .map(|r| {
                self.kept
                    .binary_search(r)
                    .map(|i| self.slots[i])
                    .map_err(|_| candle_core::Error::Msg(format!("flowing row {r} was not kept")))
            })
            .collect()
    }

    /// Which rows of the store a forward's lines read once its `new` rows
    /// are [`Self::put`] where [`Self::new_slots`] says: the `earlier` rows on
    /// them, then every new row — `None` where no earlier row is read, the new rows
    /// being the lines' own, unless `captured`. One gather a kind.
    ///
    /// `captured`: a level captured as a CUDA graph gathers even then, so
    /// every level runs the same kernels and its graph is updated in place.
    /// Without, a level reading no earlier row had five gathers a layer fewer
    /// and the driver refused every update (`CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`,
    /// 588 in one arm with the 2B drafter, 2026-11-23).
    pub(super) fn rows_the_lines_read(
        &self,
        earlier: &[usize],
        new: usize,
        captured: bool,
        device: &Device,
    ) -> CandleResult<Option<SlotTable>> {
        if earlier.is_empty() && !captured {
            return Ok(None);
        }
        let bound = self.written_after(new);
        let mut at = self.slots_of(earlier)?;
        at.extend(self.new_slots(new));
        // A kernel reads the store through these (`SlotTable`): none may name
        // a row past the ones the store holds once the new rows are put.
        if let Some(past) = at.iter().find(|&&slot| slot as usize >= bound) {
            candle_core::bail!("a flowing row's slot {past} is past the store's {bound} rows");
        }
        Ok(Some(SlotTable {
            table: rows_on(at, device)?,
            bound,
        }))
    }

    /// Makes room in the store for a forward's `rows` new rows, BEFORE its
    /// tables name slots ([`Self::rows_the_lines_read`]): a store whose free
    /// slots and unwritten rows ([`Self::room`]) cannot take them gathers the
    /// rows still kept to the front of a larger one
    /// ([`store_capacity_for`]), and their slots follow. Joined anew (the A/B
    /// arms) there is always room.
    pub(super) fn make_room(&mut self, rows: usize) -> CandleResult<()> {
        if flow_rows_mode() != FlowRowsMode::Stored || rows <= self.room() {
            return Ok(());
        }
        let capacity = store_capacity_for(self.kept.len() + rows);
        let Some(device) = self
            .store
            .kinds
            .iter()
            .flatten()
            .next()
            .map(|t| t.device().clone())
        else {
            self.store.capacity = capacity;
            return Ok(());
        };
        #[cfg(test)]
        FLOW_STORE_GROWTHS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // ONE gather a kind makes the whole new store: the kept rows' slots,
        // then the first of them again up to the capacity — rows past the
        // kept ones are only ever read after `put` writes them (a table names
        // no slot past the rows written, `rows_the_lines_read`), so what they
        // hold first is never read. It was a zero-filled store, the kept rows
        // gathered apart and copied in: 90 fills, 180 allocations and 180
        // launches a growth of the 0.8B drafter's 18 layers, ~2 ms (nsys,
        // 2026-11-24). A/B `SWARMLLM_FLOW_GROWTH=copied`.
        //
        // Prior art: llama.cpp has dropped its cache defragmentation and vLLM
        // never compacts — both read through tables, as this store does
        // between growths; a growth is the one copy left.
        let copied = flow_growth_copied();
        let at = match self.slots.first() {
            None => None,
            Some(&first) => {
                let mut at = self.slots.clone();
                if !copied {
                    at.resize(capacity, first);
                }
                Some(rows_on(at, &device)?)
            }
        };
        for kind in self.store.kinds.iter_mut().flatten() {
            *kind = match &at {
                Some(at) if !copied => kind.index_select(at, 1)?,
                _ => {
                    let mut dims = kind.dims().to_vec();
                    dims[1] = capacity;
                    let fresh = Tensor::zeros(dims, kind.dtype(), kind.device())?;
                    if let Some(at) = &at {
                        fresh.slice_set(&kind.index_select(at, 1)?, 1, 0)?;
                    }
                    fresh
                }
            };
        }
        self.slots = (0..self.kept.len() as u32).collect();
        self.store.len = self.kept.len();
        self.store.capacity = capacity;
        // The store is new: no slot past the kept rows has been written.
        self.free.clear();
        Ok(())
    }

    /// Writes a forward's new rows of `kind` where [`Self::new_slots`] says —
    /// slots the walk left first, then after the rows written
    /// ([`Self::make_room`] made room) — and answers every row written with
    /// them, `[b, written, …]` — copied once, in place, the rows held never
    /// copied again ([`FlowRowsMode::Stored`]); joined anew in the A/B arms.
    /// `on_card`: the slots as the forward's card reads them (the tail of
    /// its lines' table, [`Self::rows_the_lines_read`]), so a level writes
    /// its rows in ONE launch a kind wherever they land (`cuda::put_rows`).
    /// The rows count as written once every kind is put
    /// ([`Self::keep_appended`]): a forward that fails between leaves them to
    /// the next.
    pub(super) fn put(
        &mut self,
        kind: RowKind,
        new: &Tensor,
        on_card: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let new = new.contiguous()?;
        let rows = new.dim(1)?;
        let (len, capacity) = (self.store.len, self.store.capacity);
        let slots = self.new_slots(rows);
        let written = self.written_after(rows);
        let slot = &mut self.store.kinds[kind as usize];
        match flow_rows_mode() {
            FlowRowsMode::Stored => {
                if slot.is_none() {
                    let mut dims = new.dims().to_vec();
                    dims[1] = capacity.max(written);
                    *slot = Some(Tensor::zeros(dims, new.dtype(), new.device())?);
                }
                let held = slot.as_ref().expect("made above");
                put_rows_at(held, &new, &slots, len, on_card)?;
                held.narrow(1, 0, written)
            }
            FlowRowsMode::Joined | FlowRowsMode::PerLayer => {
                let all = match slot.as_ref() {
                    Some(held) if len > 0 => Tensor::cat(&[&held.narrow(1, 0, len)?, &new], 1)?,
                    _ => new,
                };
                *slot = Some(all.clone());
                Ok(all)
            }
        }
    }

    /// Keeps a forward's rows once every kind is [`Self::put`]: every row's
    /// parent, and the slots of the rows following the anchor (`live`).
    pub(super) fn keep_appended(
        &mut self,
        before: usize,
        parents: &[Option<usize>],
        live: &[bool],
    ) {
        debug_assert_eq!(before, self.read);
        self.parents.extend_from_slice(parents);
        self.read = before + parents.len();
        let new = self.new_slots(parents.len());
        let written = self.written_after(parents.len());
        for (i, _) in live.iter().enumerate().filter(|(_, live)| **live) {
            self.kept.push(before + i);
            self.slots.push(new[i]);
        }
        // The free slots the rows took; a row not kept (`live` false) leaves
        // its slot free from the next forward on (`note_begun`).
        let taken = parents.len().min(self.free.len());
        self.free.drain(..taken);
        self.store.len = written;
    }

    /// A forward's new rows (`parents` as tree rows, the first at tree row
    /// `before`) as one tree from the anchor: each live row's line back to it,
    /// through kept rows and rows of this forward.
    pub(super) fn lines(&self, before: usize, parents: &[Option<usize>]) -> FlowLines {
        let mut live = vec![false; parents.len()];
        let mut earlier = std::collections::BTreeSet::new();
        for (i, slot) in live.iter_mut().enumerate() {
            let mut on_line = Vec::new();
            let mut at = parents[i];
            let reached = loop {
                match at {
                    None => break self.anchor_row.is_none(),
                    Some(r) if Some(r) == self.anchor_row => break true,
                    Some(r) if r >= before => at = parents[r - before],
                    Some(r) if self.kept.binary_search(&r).is_ok() => {
                        on_line.push(r);
                        at = self.parents[r];
                    }
                    Some(_) => break false,
                }
            };
            if reached {
                *slot = true;
                earlier.extend(on_line);
            }
        }
        let earlier: Vec<usize> = earlier.into_iter().collect();
        let ahead = earlier.len();
        let index_of = |r: usize| -> Option<usize> {
            if Some(r) == self.anchor_row {
                return None;
            }
            match r.checked_sub(before) {
                Some(i) => Some(ahead + i),
                None => earlier.binary_search(&r).ok(),
            }
        };
        let mut ext: Vec<Option<usize>> = earlier
            .iter()
            .map(|&r| self.parents[r].and_then(index_of))
            .collect();
        ext.extend(
            parents
                .iter()
                .zip(&live)
                .map(|(p, &live)| if live { p.and_then(index_of) } else { None }),
        );
        FlowLines {
            earlier,
            parents: ext,
            live,
        }
    }

    /// Moves the anchor down the walk's line as far as rows are kept — to the
    /// walk's row once it has been read — replaying their inputs from the
    /// anchor with the step's own kernel ([`replay_rows`]), so the anchor is
    /// bit for bit the state a chain read of the reply reaches; then keeps
    /// only the rows following the new anchor.
    pub(super) fn advance_to(&mut self, walk: usize) -> CandleResult<()> {
        let mut path = Vec::new();
        let mut at = Some(walk);
        while let Some(r) = at {
            if Some(r) == self.anchor_row {
                break;
            }
            path.push(r);
            at = self.parents.get(r).copied().flatten();
        }
        path.reverse();
        let reach = path
            .iter()
            .take_while(|r| self.kept.binary_search(r).is_ok())
            .count();
        if reach == 0 {
            return Ok(());
        }
        let path = &path[..reach];
        let device = self.anchor.recurrent_state.device().clone();
        let rows = self
            .path_inputs(path, &device)?
            .expect("a path of kept rows has inputs");
        // On a card each replay is handed buffers of its own, so it takes the
        // two-launch replay (`cuda::replay` takes only a target): without one
        // every recurrent layer composed both replays, ~12 launches each, at
        // every forward the walk moved — bit for bit the same state either way
        // (`a_replay_on_the_card_lands_where_the_composed_replay_does`).
        // Fresh, never the pair being replaced: a forward's rollback copy
        // (`SsmState::flow_begun`) may still hold it. Never zeroed: either
        // replay writes the pair whole (`unwritten_like`). A/B
        // `SWARMLLM_SSM_REPLAY=composed`.
        let target = || -> CandleResult<Option<SsmSlot>> {
            if !device.is_cuda() {
                return Ok(None);
            }
            Ok(Some(SsmSlot {
                conv_state: unwritten_like(&self.anchor.conv_state)?,
                recurrent_state: unwritten_like(&self.anchor.recurrent_state)?,
            }))
        };
        let before = if reach > 1 {
            replay_rows(&self.anchor, &rows, reach - 1, target()?.as_ref())?
        } else {
            self.anchor.clone()
        };
        let last = |t: &Tensor| t.narrow(1, reach - 1, 1);
        let anchor = replay_rows(
            &before,
            &SsmRows {
                mixed: last(&rows.mixed)?,
                qk: last(&rows.qk)?,
                v: last(&rows.v)?,
                g: last(&rows.g)?,
                beta: last(&rows.beta)?,
            },
            1,
            target()?.as_ref(),
        )?;
        let row = path[reach - 1];
        self.before_anchor = before;
        self.anchor = anchor;
        self.anchor_row = Some(row);
        let still: Vec<usize> = self
            .kept
            .iter()
            .copied()
            .filter(|&r| r != row && self.follows(r, Some(row)))
            .collect();
        // Stored, the rows the walk left stay where they are; joined anew (the
        // A/B arms), the rows still kept are gathered as before the store.
        let slots = self.slots_of(&still)?;
        if flow_rows_mode() == FlowRowsMode::Stored {
            self.kept = still;
            self.slots = slots;
            return Ok(());
        }
        let kept = self.inputs_of(&still, &device)?;
        self.kept = still;
        self.slots = (0..self.kept.len() as u32).collect();
        self.store.len = self.kept.len();
        if let Some(kept) = kept {
            for kind in [
                RowKind::Mixed,
                RowKind::Qk,
                RowKind::V,
                RowKind::G,
                RowKind::Beta,
            ] {
                self.store.kinds[kind as usize] = Some(kind.of(&kept).clone());
            }
        }
        Ok(())
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
/// `[b, heads, v_dim, k_dim]`. Returns `[b, seq, heads, v_dim]` and the state
/// — a tree's (`tree`, each row's parent, #437) its last row's, each row
/// started from its parent's state.
fn delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    mut state: Tensor,
    tree: Option<&[Option<usize>]>,
) -> CandleResult<(Tensor, Tensor)> {
    let seq_len = q.dim(1)?;
    let mut outs = Vec::with_capacity(seq_len);
    let read = state.clone();
    let mut row_states: Vec<Tensor> = Vec::new();
    for t in 0..seq_len {
        if let Some(parents) = tree {
            state = match parents[t] {
                Some(p) => row_states[p].clone(),
                None => read.clone(),
            };
        }
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
        if tree.is_some() {
            row_states.push(state.clone());
        }
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
///
/// A TREE's rows (`tree`, each row's parent, #437) each start from their
/// parent's state — kept in a buffer each pool thread keeps, a state only
/// until the row's last child has read it ([`RowSlots`]) — with the same
/// arithmetic a row of a chain gets, so a row whose line is the rows before
/// it gets the chain's bits. The state returned is the last row's. Twig
/// (arXiv 2608.01651) solves the same tree in closed form over an ancestor
/// mask to keep no state per node; with only the states still to be read
/// kept, they fit.
#[allow(clippy::too_many_arguments)]
fn delta_rule_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    tree: Option<&[Option<usize>]>,
) -> CandleResult<(Tensor, Tensor)> {
    let (b, seq, nk, kd) = q.dims4()?;
    let (_, _, nv, vd) = v.dims4()?;
    let flat = |t: &Tensor| t.contiguous()?.flatten_all()?.to_vec1::<f32>();
    let (q, k, v, g, beta) = (flat(q)?, flat(k)?, flat(v)?, flat(g)?, flat(beta)?);
    let slots = tree.map(RowSlots::of);
    let mut s = flat(state)?; // [b, nv, vd, kd]
    let mut out = vec![0f32; b * nv * seq * vd]; // [b, nv, seq, vd]: one head's tokens contiguous
    s.par_chunks_mut(vd * kd)
        .zip(out.par_chunks_mut(seq * vd))
        .enumerate()
        .for_each(|(bh, (s, o))| {
            let (bi, h) = (bh / nv, bh % nv);
            let step = |s: &mut [f32], o: &mut [f32], t: usize| {
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
            };
            match tree {
                None => (0..seq).for_each(|t| step(s, o, t)),
                Some(parents) => TREE_ROW_STATES.with(|kept| {
                    let mut kept = kept.borrow_mut();
                    let size = vd * kd;
                    let slots = slots.as_ref().expect("a tree's slots");
                    kept.resize(slots.count * size, 0.0);
                    for (t, parent) in parents.iter().enumerate() {
                        let at = slots.of[t] * size;
                        match parent.map(|p| slots.of[p] * size) {
                            // A last child reads on in its parent's slot.
                            Some(from) if from == at => {}
                            Some(from) => kept.copy_within(from..from + size, at),
                            None => kept[at..at + size].copy_from_slice(s),
                        }
                        step(&mut kept[at..at + size], o, t);
                    }
                    if let Some(&last) = slots.of.last() {
                        s.copy_from_slice(&kept[last * size..(last + 1) * size]);
                    }
                    // A wide tree (a flow's, or another computer's choosing)
                    // gives its buffer back rather than keep it on the thread.
                    if kept.capacity() > SsmState::MOST_ROWS_KEPT * size {
                        *kept = Vec::new();
                    }
                }),
            }
        });
    let out = Tensor::from_vec(out, (b, nv, seq, vd), &Device::Cpu)?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((out, Tensor::from_vec(s, (b, nv, vd, kd), &Device::Cpu)?))
}

thread_local! {
    /// The states [`delta_rule_cpu`] keeps while it reads a tree
    /// ([`RowSlots`]), kept by the pool thread across calls: fresh pages
    /// every head of every layer would cost more than the rule. Up to
    /// `SsmState::MOST_ROWS_KEPT` states (64 KB a head at 128 × 128) stay;
    /// more is given back after the call.
    static TREE_ROW_STATES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Where each row of a tree keeps its state while [`delta_rule_cpu`] reads it
/// (rows in order, a parent before its children). A row's state is needed
/// only until its last child reads it: that child takes over its slot and
/// reads on in place, and a row with no child gives its slot up after its own
/// step (but the last row, whose state the step returns). So a tree keeps the
/// states still to be read — one for a chain, a few for a flow's lines —
/// however many rows it has; before, a state a row, so a flowing tree of 512
/// rows held 32 MB on every pool thread, for good (security review
/// 2026-11-22). Liveness, as a register allocator frees a value at its last
/// use.
struct RowSlots {
    /// Each row's slot.
    of: Vec<usize>,
    /// Slots used.
    count: usize,
}

impl RowSlots {
    fn of(parents: &[Option<usize>]) -> Self {
        let mut last_child = vec![None; parents.len()];
        for (t, parent) in parents.iter().enumerate() {
            if let Some(p) = *parent {
                last_child[p] = Some(t);
            }
        }
        let (mut of, mut free, mut count) = (vec![0; parents.len()], Vec::new(), 0);
        for (t, parent) in parents.iter().enumerate() {
            if let Some(p) = *parent {
                if last_child[p] == Some(t) {
                    free.push(of[p]);
                }
            }
            of[t] = free.pop().unwrap_or_else(|| {
                count += 1;
                count - 1
            });
            if last_child[t].is_none() && t + 1 < parents.len() {
                free.push(of[t]);
            }
        }
        Self { of, count }
    }
}

/// A tree check's rows as a recurrent layer reads them (#437): each row's
/// parent among the step's `seq_len` rows, `None` for one that follows the
/// state the step reads (`RowVisibility::parents`, off the tree the forward
/// set — `SplitModel::with_draft_tree`). `None` outside a tree, and for a
/// tree that is one chain — each row the previous row's child, a step's own
/// shape, read as a step is. A tree wider than the rows a step keeps
/// ([`SsmState::MOST_ROWS_KEPT`]) is refused: its verdict could not be taken
/// back into it.
fn tree_parents_now(seq_len: usize) -> CandleResult<Option<Vec<Option<usize>>>> {
    let Some(seen) = super::tree_scope_now().and_then(|scope| scope.visibility) else {
        return Ok(None);
    };
    let parents = seen.parents().map_err(candle_core::Error::msg)?;
    if parents.len() != seq_len {
        candle_core::bail!(
            "a tree of {} rows in a DeltaNet step of {seq_len}",
            parents.len()
        );
    }
    if parents
        .iter()
        .enumerate()
        .all(|(row, parent)| *parent == row.checked_sub(1))
    {
        return Ok(None);
    }
    if seq_len > SsmState::MOST_ROWS_KEPT {
        candle_core::bail!(
            "a tree of {seq_len} rows: a recurrent layer takes back at most {}",
            SsmState::MOST_ROWS_KEPT
        );
    }
    Ok(Some(parents))
}

/// A table of row numbers a tree forward hands its recurrent layers — the
/// tree's parents, a flow's walks, the kept rows a layer gathers — as a `u32`
/// tensor on `device`. Every recurrent layer of a forward reads the SAME
/// tables (one tree, one walk, one kept set), and an upload from pageable
/// memory waits for the card's stream to drain first
/// (`docs/invariants/inference.md` § "A speculative check is captured too, and
/// a layout is uploaded once"), so on a card a table is copied ONCE while its
/// content stands and handed out again by content: the 0.8B drafter's 18
/// recurrent layers made three or four copies each a level of its flow. A hit
/// is the very bytes the upload would make, read only. A flowing level that
/// is CAPTURED has every table it reads uploaded here before the capture
/// (`SsmState::stage_flowing_level`): inside one, a miss refuses the capture
/// rather than upload — a pageable copy there is replayed from the host
/// address (gotcha #758). Kept in a `static` — never dropped, so no card
/// memory is freed after the context at exit. A/B `SWARMLLM_FLOW_STORE=per-layer`.
///
/// Prior art: llama.cpp sets a graph's inputs once a micro-batch
/// (`llm_graph_input_*::set_input` — positions, masks, `s_copy`) and every
/// layer reads them; `CudaDevice::layout_params` keeps each distinct strided
/// layout on the card the same way.
fn rows_on(ids: Vec<u32>, device: &Device) -> CandleResult<Tensor> {
    #[cfg(feature = "candle-cuda")]
    if device.is_cuda() && !flow_rows_as_before() {
        let mut held = ROWS_ON_CARD.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = held
            .iter()
            .position(|(rows, on, _)| rows == &ids && on.same_device(device))
        {
            let hit = held.remove(at);
            let tensor = hit.2.clone();
            held.insert(0, hit);
            return Ok(tensor);
        }
        if stream_is_capturing(device) {
            candle_core::bail!(
                "a flowing level's row table was not staged before its capture ({} rows)",
                ids.len()
            );
        }
        let len = ids.len();
        let tensor = Tensor::from_vec(ids.clone(), len, device)?;
        held.insert(0, (ids, device.clone(), tensor.clone()));
        held.truncate(ROW_TABLES_HELD);
        return Ok(tensor);
    }
    let len = ids.len();
    Tensor::from_vec(ids, len, device)
}

/// Is `device`'s stream being captured into a CUDA graph right now? False
/// off a card, and in a build without CUDA.
pub(super) fn capturing(device: &Device) -> bool {
    #[cfg(feature = "candle-cuda")]
    {
        stream_is_capturing(device)
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = device;
        false
    }
}

/// Is `device`'s stream being captured into a CUDA graph right now?
#[cfg(feature = "candle-cuda")]
fn stream_is_capturing(device: &Device) -> bool {
    use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus;
    let Device::Cuda(dev) = device else {
        return false;
    };
    !matches!(
        dev.cuda_stream().capture_status(),
        Ok(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    )
}

/// `SWARMLLM_FLOW_GRAPH=0` → a flowing level is never captured
/// (`SplitModel::tree_steps_may_be_captured`, [`SsmState::stage_flowing_level`]):
/// the A/B switch. On by default since 2026-11-23: a captured drafter level
/// reads bit for bit as the ordinary one at 6.1-8.3 ms against 11.0-14.2 on
/// the card alone (`flowing_levels_pace_captured_and_op_by_op` at 2, 4 and
/// 8 rows); on the TH↔AU link wrote prose at 13.9-14.9 and code at 24.2-31.7 tok/s against 12.6-14.1 and 22.7-22.9 uncaptured, the drafter's call 25.7-28.4 against 37.3-40.2 ms a forward, no refusal in three arms each (`real_link_rf9b10_1123/`). It was off
/// while a staged level freed memory inside its capture (gotcha #1165,
/// `SsmState::flow_from`). Read once.
#[cfg(feature = "candle-cuda")]
fn flowing_levels_captured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FLOW_GRAPH").as_deref() != Ok("0"))
}

/// How many distinct tables [`rows_on`] keeps: a flowing forward hands its
/// layers up to five (parents, walks, the earlier rows, the walk's path, the
/// rows kept), a drafter's call three levels of them.
#[cfg(feature = "candle-cuda")]
const ROW_TABLES_HELD: usize = 16;

#[cfg(feature = "candle-cuda")]
#[allow(clippy::type_complexity)]
static ROWS_ON_CARD: std::sync::Mutex<Vec<(Vec<u32>, Device, Tensor)>> =
    std::sync::Mutex::new(Vec::new());

/// How a flowing layer keeps its rows' inputs (`SWARMLLM_FLOW_STORE`, read
/// once; named `SWARMLLM_FLOW_ROWS` until 2026-11-23, which also set the
/// rows a drafter level — `dsd_flow::rows_a_level` — and still does). The three give the same bits; they differ in copies and launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlowRowsMode {
    /// The default: a forward's new rows are copied once, in place, after the
    /// rows written ([`FlowRows::put`]); on a card the kernels read the lines'
    /// rows there ([`flow_reads_in_place`]), elsewhere — and under `gathered`
    /// — they are one gather a kind; a row the walk left stays where it is
    /// until the store is full.
    Stored,
    /// `joined` (2026-11-22, before the store): the held rows and the new ones
    /// joined into a new tensor a kind every forward, the rows the walk left
    /// gathered out as it moves.
    Joined,
    /// `per-layer` (before that): as `joined`, and the lines' rows gathered
    /// and joined to the new ones apart, every table uploaded again a layer
    /// ([`rows_on`]).
    PerLayer,
}

/// Whether the walk's path is always gathered (`SWARMLLM_FLOW_PATH=gathered`,
/// the A/B inside one binary) rather than read in place where its rows run
/// one after another ([`FlowRows::path_inputs`]). Read once.
fn path_inputs_gathered() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FLOW_PATH").as_deref() == Ok("gathered"))
}

/// Whether a full store grows as before 2026-11-24 — zero-filled, the kept
/// rows gathered apart and copied in (`SWARMLLM_FLOW_GROWTH=copied`, the A/B
/// inside one binary) — rather than in one gather a kind
/// ([`FlowRows::make_room`]). Read once.
fn flow_growth_copied() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FLOW_GROWTH").as_deref() == Ok("copied"))
}

/// Whether a flow's new rows go into the slots the walk left
/// ([`FlowRows::new_slots`]) — the default — rather than only after the rows
/// written, the kept rows gathered into a new store whenever it is full:
/// `SWARMLLM_FLOW_SLOTS=compacted`, as before 2026-11-25. Stored rows only
/// ([`FlowRowsMode::Stored`]). Read once.
///
/// Every third level of a 0.8B drafter's 8-row tree grew its 32-row store
/// (`store_capacity_for`'s floor): 90 gathers and their table, ~0.75 ms of
/// that level (memory `archive/fable_batch_1124.md` § 3), although the walk had
/// left most of the rows it held.
///
/// Prior art: vLLM's KV-cache manager hands a finished sequence's blocks to
/// the next allocation from a free queue and never compacts; llama.cpp's
/// `llama_kv_cache::find_slot` reuses any empty cell before it appends; and
/// epoch-based reclamation (Fraser 2004; crossbeam-epoch) frees memory only once
/// no reader of the epoch that held it remains — here the forward's rollback
/// copy, so a slot is reused only from the forward after the one that left it.
fn flow_slots_reused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        flow_rows_mode() == FlowRowsMode::Stored
            && std::env::var("SWARMLLM_FLOW_SLOTS").as_deref() != Ok("compacted")
    })
}

/// Stores grown this process, a layer each ([`FlowRows::make_room`]'s
/// gathers): what the pace probe prints, so an A/B of
/// `SWARMLLM_FLOW_SLOTS=compacted` shows the reuse fired.
#[cfg(test)]
static FLOW_STORE_GROWTHS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(all(test, feature = "candle-cuda"))]
pub(super) fn flow_store_growths() -> u64 {
    FLOW_STORE_GROWTHS.load(std::sync::atomic::Ordering::Relaxed)
}

fn flow_rows_mode() -> FlowRowsMode {
    static MODE: std::sync::OnceLock<FlowRowsMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SWARMLLM_FLOW_STORE").as_deref() {
        Ok("per-layer") => FlowRowsMode::PerLayer,
        Ok("joined") => FlowRowsMode::Joined,
        _ => FlowRowsMode::Stored,
    })
}

/// Whether a flowing forward's kernels read each kind's rows where the store
/// keeps them, through a [`SlotTable`] — the default — rather than gathered
/// out first: `SWARMLLM_FLOW_STORE=gathered` gathers them, as before
/// 2026-11-23. Read once.
///
/// Prior art: vLLM's PagedAttention (Kwon et al., SOSP 2023) reads a
/// sequence's cache through its block table rather than copying it
/// contiguous first; the table here names each row's slot in the store.
fn flow_reads_in_place() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        flow_rows_mode() == FlowRowsMode::Stored
            && !matches!(
                std::env::var("SWARMLLM_FLOW_STORE").as_deref(),
                Ok("gathered")
            )
    })
}

/// The rows a flowing forward's lines read, as slots of the flow's store
/// ([`FlowRows::rows_the_lines_read`]): `[rows]` u32 on the forward's device,
/// every slot below `bound` — the rows the store holds once the forward's own
/// are put, checked on the host. A kernel reading through it checks `bound`
/// against the store it is handed.
#[derive(Clone)]
pub(super) struct SlotTable {
    table: Tensor,
    bound: usize,
}

impl SlotTable {
    /// How many rows the lines read.
    #[cfg(feature = "candle-cuda")]
    fn rows(&self) -> CandleResult<usize> {
        self.table.dim(0)
    }

    /// The rows themselves, gathered out of `store` (`[b, ≥ bound, …]`).
    fn gather(&self, store: &Tensor) -> CandleResult<Tensor> {
        if store.dim(1)? < self.bound {
            candle_core::bail!(
                "a store of {} rows read through slots up to {}",
                store.dim(1)?,
                self.bound
            );
        }
        store.index_select(&self.table, 1)
    }
}

/// `SWARMLLM_FLOW_STORE=per-layer`: [`rows_on`] copies every table for every
/// layer, as before 2026-11-22.
#[cfg(feature = "candle-cuda")]
fn flow_rows_as_before() -> bool {
    flow_rows_mode() == FlowRowsMode::PerLayer
}

/// The rows a store is made for, holding `rows` now: half as many again, in
/// steps of 32 — room for a few forwards before the rows still kept are
/// gathered again. Twice them, the store of a 9B's 13 recurrent layers here
/// claimed 184 MB to grow, past a 491 MB budget holding 308, and the flow
/// ended for memory, its reply finished in tree rounds (2026-11-23, TH↔AU);
/// growth by half keeps a store's copies amortized (each row moved about
/// three times, against two), as Java's `ArrayList` and folly's `fbvector`
/// grow by half (CPython's lists by an eighth) to keep memory near what is
/// held, where Rust's `Vec` doubles.
fn store_capacity_for(rows: usize) -> usize {
    (rows + rows / 2).max(32).next_multiple_of(32)
}

/// Writes `new`'s rows `[b, rows, …]` to rows `slots` of `store`
/// ([`FlowRows::put`]): on a card in one launch through `on_card` (the slots
/// there, `cuda::put_rows`) — every level the same kernel, a captured one
/// included, wherever its rows land; else one copy a run of slots, the rows
/// past those written (`at`) in one.
fn put_rows_at(
    store: &Tensor,
    new: &Tensor,
    slots: &[u32],
    at: usize,
    on_card: Option<&Tensor>,
) -> CandleResult<()> {
    #[cfg(feature = "candle-cuda")]
    if let Some(on_card) = on_card {
        if store.device().is_cuda() && cuda::put_rows(store, new, on_card)? {
            return Ok(());
        }
    }
    #[cfg(not(feature = "candle-cuda"))]
    let _ = on_card;
    if slots.iter().enumerate().all(|(i, &s)| s as usize == at + i) {
        return store.slice_set(new, 1, at);
    }
    // Rows into slots the walk left, with no table on the card: an
    // uncaptured level reading no earlier row (`rows_the_lines_read`).
    #[cfg(feature = "candle-cuda")]
    if store.device().is_cuda()
        && cuda::put_rows(store, new, &rows_on(slots.to_vec(), store.device())?)?
    {
        return Ok(());
    }
    let mut i = 0;
    while i < slots.len() {
        let mut end = i + 1;
        while end < slots.len() && slots[end] == slots[end - 1] + 1 {
            end += 1;
        }
        store.slice_set(
            &new.narrow(1, i, end - i)?.contiguous()?,
            1,
            slots[i] as usize,
        )?;
        i = end;
    }
    Ok(())
}

/// Which of a row's inputs ([`SsmRows`]) a tensor of the store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowKind {
    Mixed,
    Qk,
    V,
    G,
    Beta,
}

impl RowKind {
    fn of(self, rows: &SsmRows) -> &Tensor {
        match self {
            Self::Mixed => &rows.mixed,
            Self::Qk => &rows.qk,
            Self::V => &rows.v,
            Self::G => &rows.g,
            Self::Beta => &rows.beta,
        }
    }
}

/// A flow's rows' inputs, a tensor a kind (`[b, capacity, …]`): a forward's
/// rows go into the slots the walk had left when it began, then at `len..`
/// ([`FlowRows::new_slots`], [`FlowRows::put`]). A kept row's place is its
/// slot (`FlowRows::slots`); a store with no room left for a forward's rows
/// has the rows still kept gathered to the front of a larger one
/// ([`FlowRows::make_room`]). Joined anew (the A/B arms) each tensor holds
/// exactly the rows written.
#[derive(Debug, Clone, Default)]
struct RowStore {
    kinds: [Option<Tensor>; 5],
    /// Rows each kind's tensor holds room for ([`FlowRowsMode::Stored`]).
    capacity: usize,
    /// Rows written.
    len: usize,
}

/// A tree check's rows as a recurrent layer reads them (#437): each row's
/// parent among the step's rows ([`tree_parents_now`]), and on a card the same
/// as `u32`s (`u32::MAX`: no parent), copied there once a step for both of its
/// kernels (`ssm_conv_silu_tree_f32_k4`, `delta_rule_tree_f32_d*`) — never
/// inside a capture: a recurrent model's tree is never captured
/// (`SplitModel::tree_steps_may_be_captured`).
struct TreeLines {
    parents: Vec<Option<usize>>,
    #[cfg(feature = "candle-cuda")]
    on_card: Option<Tensor>,
    /// A flowing forward's walks (`TreeLines::for_flow`): for each row whose
    /// parent is not the row before it, the rows of its line root first —
    /// `[seq + 1]` offsets, then the rows — on the card, for
    /// `delta_rule_flow_f32_d*`, whose lines may be deeper than the tree
    /// kernel's 32.
    #[cfg(feature = "candle-cuda")]
    walks_on_card: Option<Tensor>,
}

impl TreeLines {
    fn new(parents: Vec<Option<usize>>, device: &Device) -> CandleResult<Self> {
        #[cfg(feature = "candle-cuda")]
        {
            let on_card = if device.is_cuda() {
                let ids: Vec<u32> = parents
                    .iter()
                    .map(|p| p.map_or(u32::MAX, |p| p as u32))
                    .collect();
                Some(rows_on(ids, device)?)
            } else {
                None
            };
            Ok(Self {
                parents,
                on_card,
                walks_on_card: None,
            })
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            let _ = device;
            Ok(Self { parents })
        }
    }

    /// A flowing forward's rows ([`DeltaNetWeights::forward_flowing`]): as
    /// [`Self::new`], and on a card each row's walk named for the delta rule
    /// (`delta_rule_flow_f32_d*`), since a line read from the walk's anchor is
    /// as deep as the forwards in flight make it.
    fn for_flow(parents: Vec<Option<usize>>, device: &Device) -> CandleResult<Self> {
        #[cfg(feature = "candle-cuda")]
        {
            let mut lines = Self::new(parents, device)?;
            if device.is_cuda() {
                lines.walks_on_card = Some(rows_on(flow_walks(&lines.parents), device)?);
            }
            Ok(lines)
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            Self::new(parents, device)
        }
    }
}

/// [`TreeLines::for_flow`]'s walks: for each row whose parent is not the row
/// before it, the rows of its parent's line from the root down — what
/// `delta_rule_flow_f32_d*` replays from the state read before the row's own
/// update; `[rows + 1]` offsets into the rows that follow them.
#[cfg(feature = "candle-cuda")]
fn flow_walks(parents: &[Option<usize>]) -> Vec<u32> {
    let rows = parents.len();
    let mut offsets = Vec::with_capacity(rows + 1);
    let mut walked: Vec<u32> = Vec::new();
    for (t, parent) in parents.iter().enumerate() {
        offsets.push(walked.len() as u32);
        if *parent == t.checked_sub(1) {
            continue;
        }
        let from = walked.len();
        let mut at = *parent;
        while let Some(r) = at {
            walked.push(r as u32);
            at = parents[r];
        }
        walked[from..].reverse();
    }
    offsets.push(walked.len() as u32);
    offsets.extend(walked);
    offsets
}

/// Does `kernels/delta_rule.cu` take this layer? A lane holds `D / 32` of a
/// state row in registers, with equal key and value widths, D ∈ {64, 128}
/// (every Qwen 3.5 size has 128).
fn card_delta_rule_covers(kd: usize, vd: usize) -> bool {
    kd == vd && matches!(kd, 64 | 128)
}

/// State rows a warp owns in `kernels/delta_rule.cu`: four (its `_c4` entry
/// points), one under `SWARMLLM_DELTA_RULE_ROWS=1` — the A/B, and the kernel
/// before 2026-11-24. At one row a warp the 0.8B drafter's 2,048 warps a layer
/// were more than an RTX 3070 Laptop holds at once (1,920), so the last ran as
/// a second wave; four share each token's q, k and gates. Each row's
/// arithmetic is the same, so the bits are. Read once.
#[cfg(feature = "candle-cuda")]
fn delta_rule_rows_a_warp() -> usize {
    #[cfg(test)]
    if let Some(rows) = tests::DELTA_RULE_ROWS.with(std::cell::Cell::get) {
        return rows;
    }
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(
        || match std::env::var("SWARMLLM_DELTA_RULE_ROWS").as_deref() {
            Ok("1") => 1,
            _ => 4,
        },
    )
}

/// The entry point a launch of `base` (a `delta_rule_*_f32_d*` name) takes at
/// [`delta_rule_rows_a_warp`] rows a warp, and its grid's z for state width
/// `d`: four warps a block.
#[cfg(feature = "candle-cuda")]
fn delta_rule_entry(base: &str, d: usize) -> (std::borrow::Cow<'static, str>, u32) {
    match delta_rule_rows_a_warp() {
        1 => (std::borrow::Cow::Owned(base.to_owned()), (d / 4) as u32),
        rows => (
            std::borrow::Cow::Owned(format!("{base}_c{rows}")),
            (d / (4 * rows)) as u32,
        ),
    }
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
    gates: &Gates,
    state: &Tensor,
    state_out: &Tensor,
    scale: f32,
    tree: Option<&TreeLines>,
    slots: Option<&SlotTable>,
) -> CandleResult<Option<Tensor>> {
    #[cfg(feature = "candle-cuda")]
    {
        if [qk, v, &gates.g, &gates.beta, state, state_out]
            .iter()
            .any(|t| t.dtype() != DType::F32)
            || !state_out.is_contiguous()
            || (slots.is_some() && !(gates.g.is_contiguous() && gates.beta.is_contiguous()))
        {
            return Ok(None);
        }
        // A tree's rows each start from their parent's state
        // (`delta_rule_tree_f32_d*`), never the chain's kernel.
        let (parents, walks) = match tree {
            None => (None, None),
            Some(TreeLines {
                on_card: Some(p),
                walks_on_card,
                ..
            }) => (Some(p.clone()), walks_on_card.clone()),
            Some(_) => return Ok(None),
        };
        let (b, held, nv, vd) = v.dims4()?;
        // Read in place, the operands are the stores — g's the third, beta's
        // beside it, never stacked — and the rows the table's.
        let (seq, gates_operand, stored) = match slots {
            Some(at) => (
                at.rows()?,
                gates.g.clone(),
                Some((gates.beta.clone(), at.clone())),
            ),
            None => (held, gates.stacked()?, None), // [2, b, seq, nv]
        };
        let op = cuda::DeltaRule {
            state: state.contiguous()?,
            state_out: state_out.clone(),
            nk,
            scale,
            parents,
            walks,
            stored,
        };
        let out = qk.contiguous()?.apply_op3_no_bwd(v, &gates_operand, &op)?;
        Ok(Some(out.reshape((b, seq, nv, vd))?))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (qk, nk, v, gates, state, state_out, scale, tree, slots);
        Ok(None)
    }
}

/// `SWARMLLM_SSM_REPLAY=composed` → a rewind into a check replays its rows
/// with the composition ([`replay_rows_composed`]) on a card too: the A/B
/// switch for the two-launch replay (`cuda::replay`). Read once.
#[cfg(feature = "candle-cuda")]
fn replay_in_two_launches() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SSM_REPLAY").as_deref(),
            Ok("composed")
        )
    })
}

/// `SWARMLLM_SSM_REPLAY_TARGET=zeroed` → the walk's replay targets are
/// zero-filled as before [`unwritten_like`]: its A/B switch. Read once.
#[cfg(feature = "candle-cuda")]
fn replay_targets_zeroed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SWARMLLM_SSM_REPLAY_TARGET").as_deref(),
            Ok("zeroed")
        )
    })
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{
        CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor,
    };

    /// A DeltaNet step's convolution and SiLU as one launch
    /// (`ssm_conv_silu_f32_k4`): operands x `[1, seq, C]`, the window before
    /// the step `[1, C, 3]` and the filters `[C, 4]`; the output is q | k
    /// `[seq, split]` then v `[seq, C − split]`, flat. The window after the
    /// step is written INTO `state_out` `[1, C, 3]`, other buffers than the
    /// window read (`SsmState`'s second pair), as [`DeltaRule`] writes its
    /// state.
    pub(super) struct ConvSilu {
        pub(super) state_out: Tensor,
        pub(super) split: usize,
        /// A tree's parents (`super::TreeLines::on_card`, `[seq]` u32): each
        /// row's taps read up its own line (`ssm_conv_silu_tree_f32_k4`).
        pub(super) parents: Option<Tensor>,
        /// A flowing forward reading its store in place (`super::SlotTable`):
        /// `x` is the store, row t's input its row `slots[t]`
        /// (`ssm_conv_silu_tree_stored_f32_k4`; a tree's rows only).
        pub(super) slots: Option<super::SlotTable>,
    }

    /// A tree's parents as a kernel reads them: `[rows]` u32, contiguous, on
    /// the card — the guard that keeps the buffer alive while a launch reads it.
    struct ParentsView<'a> {
        storage: std::sync::RwLockReadGuard<'a, candle_core::Storage>,
        start: usize,
        end: usize,
    }

    impl<'a> ParentsView<'a> {
        fn of(parents: &'a Tensor, rows: usize, op: &str) -> Result<Self> {
            if parents.dtype() != DType::U32 || parents.dims() != [rows] {
                candle_core::bail!("{op}: a tree's parents must be [{rows}] u32");
            }
            let (storage, layout) = parents.storage_and_layout();
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{op}: a tree's parents must be contiguous");
            };
            Ok(Self {
                storage,
                start,
                end,
            })
        }

        fn slice(&self) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'_, u32>> {
            let candle_core::Storage::Cuda(on_card) = &*self.storage else {
                candle_core::bail!("a tree's parents are not on the card");
            };
            Ok(on_card.as_cuda_slice::<u32>()?.slice(self.start..self.end))
        }
    }

    /// A flowing forward's walks on the card (`super::flow_walks`): u32,
    /// contiguous, at least the `[rows + 1]` offsets. Their values are the
    /// host's, built from the same parents as the tree's (`TreeLines::for_flow`):
    /// every row a walk names is an ancestor of the row it walks for.
    struct WalksView<'a> {
        storage: std::sync::RwLockReadGuard<'a, candle_core::Storage>,
        start: usize,
        end: usize,
    }

    impl<'a> WalksView<'a> {
        fn of(walks: &'a Tensor, rows: usize) -> Result<Self> {
            if walks.dtype() != DType::U32 || walks.rank() != 1 || walks.dim(0)? < rows + 1 {
                candle_core::bail!(
                    "gated-delta-rule: a flow's walks must be u32, [{rows} + 1 + …]"
                );
            }
            let (storage, layout) = walks.storage_and_layout();
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("gated-delta-rule: a flow's walks must be contiguous");
            };
            Ok(Self {
                storage,
                start,
                end,
            })
        }

        fn slice(&self) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'_, u32>> {
            let candle_core::Storage::Cuda(on_card) = &*self.storage else {
                candle_core::bail!("a flow's walks are not on the card");
            };
            Ok(on_card.as_cuda_slice::<u32>()?.slice(self.start..self.end))
        }
    }

    impl ConvSilu {
        /// Whether the kernel takes these operands: one sequence, f32,
        /// contiguous, on a card, a kernel of 4.
        pub(super) fn takes(&self, x: &Tensor, prev: &Tensor, w: &Tensor) -> bool {
            let all = [x, prev, w, &self.state_out];
            if all.iter().any(|t| {
                t.dtype() != candle_core::DType::F32 || !t.is_contiguous() || !t.device().is_cuda()
            }) {
                return false;
            }
            let Ok((b, held, channels)) = x.dims3() else {
                return false;
            };
            // Read in place (a tree's rows only), x is the store, which must
            // hold the table's bound, and the rows are the table's.
            let rows = match &self.slots {
                Some(at) => match at.table.dims() {
                    [rows] => *rows,
                    _ => return false,
                },
                None => held,
            };
            let is_table = |t: &Tensor| {
                t.dtype() == candle_core::DType::U32
                    && t.dims() == [rows]
                    && t.is_contiguous()
                    && t.device().is_cuda()
            };
            b == 1
                && self.split < channels
                && prev.dims() == [1, channels, 3]
                && w.dims() == [channels, 4]
                && self.state_out.dims() == [1, channels, 3]
                && self.parents.as_ref().is_none_or(is_table)
                && self.slots.as_ref().is_none_or(|at| {
                    at.bound <= held && self.parents.is_some() && is_table(&at.table)
                })
        }
    }

    impl CustomOp3 for ConvSilu {
        fn name(&self) -> &'static str {
            "ssm-conv-silu"
        }

        /// Never reached: [`super::DeltaNetWeights::conv_on_the_card`] asks
        /// [`ConvSilu::takes`] first, which wants a card.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ssm-conv-silu's kernel is CUDA-only")
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
            let (b, held, channels) = l1.shape().dims3()?;
            // Read in place, x holds the store's rows and the rows are the
            // table's.
            let seq = match &self.slots {
                Some(at) => at.table.dim(0)?,
                None => held,
            };
            if b != 1
                || self.split >= channels
                || l2.shape().dims() != [1, channels, 3]
                || l3.shape().dims() != [channels, 4]
                || self.state_out.dims() != [1, channels, 3]
                || self
                    .slots
                    .as_ref()
                    .is_some_and(|at| at.bound > held || self.parents.is_none())
            {
                candle_core::bail!("ssm-conv-silu: unexpected shapes");
            }
            let (Some((xo, xe)), Some((po, pe)), Some((wo, we))) = (
                l1.contiguous_offsets(),
                l2.contiguous_offsets(),
                l3.contiguous_offsets(),
            ) else {
                candle_core::bail!("ssm-conv-silu: operands must be contiguous");
            };
            let (out_storage, out_layout) = self.state_out.storage_and_layout();
            let candle_core::Storage::Cuda(st_out) = &*out_storage else {
                candle_core::bail!("ssm-conv-silu: the window's buffer is not on the card");
            };
            let Some((oo, oe)) = out_layout.contiguous_offsets() else {
                candle_core::bail!("ssm-conv-silu: the window's buffer must be contiguous");
            };
            let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
            let prev = s2.as_cuda_slice::<f32>()?.slice(po..pe);
            let w = s3.as_cuda_slice::<f32>()?.slice(wo..we);
            let state_out = st_out.as_cuda_slice::<f32>()?.slice(oo..oe);
            let n_out = seq * channels;
            // Every element is assigned: each (position, channel) by its
            // channel's thread, into the q | k block or the v block.
            let mut out = dev.alloc_fully_overwritten::<f32>(n_out)?;
            let tree = match &self.parents {
                Some(p) => Some(ParentsView::of(p, seq, "ssm-conv-silu")?),
                None => None,
            };
            let parents = tree.as_ref().map(ParentsView::slice).transpose()?;
            let table = match &self.slots {
                Some(at) => Some(ParentsView::of(&at.table, seq, "ssm-conv-silu's slots")?),
                None => None,
            };
            let slots = table.as_ref().map(ParentsView::slice).transpose()?;
            let func = dev.get_or_load_custom_func(
                match (parents.is_some(), slots.is_some()) {
                    (true, true) => "ssm_conv_silu_tree_stored_f32_k4",
                    (true, false) => "ssm_conv_silu_tree_f32_k4",
                    (false, _) => "ssm_conv_silu_f32_k4",
                },
                "swarmllm_delta_rule",
                super::DELTA_RULE_PTX,
            )?;
            let cfg = LaunchConfig {
                grid_dim: (channels.div_ceil(128) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let (channels_i, split_i, seq_i) = (channels as i32, self.split as i32, seq as i32);
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&prev);
            builder.arg(&w);
            builder.arg(&mut out);
            builder.arg(&state_out);
            builder.arg(&channels_i);
            builder.arg(&split_i);
            builder.arg(&seq_i);
            if let Some(parents) = &parents {
                builder.arg(parents);
            }
            if let Some(slots) = &slots {
                builder.arg(slots);
            }
            // SAFETY: ffi. Shapes and contiguity are checked above: the kernel
            // reads x [seq, C] — read in place, [held, C] at the table's slots,
            // each below its bound (checked on the host by
            // `FlowRows::rows_the_lines_read`) and the bound within `held`
            // (checked above) — prev [C, 3] and w [C, 4], writes each of `out`'s
            // seq·C elements once and `state_out`'s C·3. `state_out` is the
            // step's spare pair, never `prev`'s buffer, written through a read
            // view ordered by the ONE stream per device, as `DeltaRule` writes.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&[n_out]),
            ))
        }
    }

    /// The most rows the gates' projections run in one launch
    /// (`ssm_gates_project_f32_r1` … `_r8`): a step or a check. A prompt
    /// pass keeps cuBLAS, as llama.cpp's `mmvf` stops at 8 columns.
    const GATES_PROJECTED_ROWS: usize = 8;

    const GATES_PROJECTED: [&str; GATES_PROJECTED_ROWS] = [
        "ssm_gates_project_f32_r1",
        "ssm_gates_project_f32_r2",
        "ssm_gates_project_f32_r3",
        "ssm_gates_project_f32_r4",
        "ssm_gates_project_f32_r5",
        "ssm_gates_project_f32_r6",
        "ssm_gates_project_f32_r7",
        "ssm_gates_project_f32_r8",
    ];

    const GATES_PROJECTED_Q8_0: [&str; GATES_PROJECTED_ROWS] = [
        "ssm_gates_project_q8_0_r1",
        "ssm_gates_project_q8_0_r2",
        "ssm_gates_project_q8_0_r3",
        "ssm_gates_project_q8_0_r4",
        "ssm_gates_project_q8_0_r5",
        "ssm_gates_project_q8_0_r6",
        "ssm_gates_project_q8_0_r7",
        "ssm_gates_project_q8_0_r8",
    ];

    /// `ssm_beta` and `ssm_alpha` as Q8_0 matrices `[nv, hidden]` on the card
    /// whose projections [`GatesProjectedQ8_0`] reads in the gates' one launch:
    /// a step of at most [`GATES_PROJECTED_ROWS`] rows of x `[1, seq, hidden]`
    /// on the card, `hidden` whole Q8_0 blocks, and not under
    /// `SWARMLLM_SSM_GATES=matmuls` (the A/B switch: the projections as
    /// matmuls). `None` otherwise.
    fn q8_0_gate_weights(
        w: &super::DeltaNetWeights,
        x: &Tensor,
    ) -> Result<Option<(std::sync::Arc<QTensor>, std::sync::Arc<QTensor>)>> {
        let (b, seq, hidden) = x.dims3()?;
        if !super::gates_project_q8_0()
            || b != 1
            || !(1..=GATES_PROJECTED_ROWS).contains(&seq)
            || !hidden.is_multiple_of(32)
            || x.dtype() != DType::F32
            || !x.is_contiguous()
            || !x.device().is_cuda()
        {
            return Ok(None);
        }
        let (Some(wb), Some(wa)) = (w.w_beta.quantized_weight(), w.w_alpha.quantized_weight())
        else {
            return Ok(None);
        };
        let fits = |q: &QTensor| {
            q.dtype() == GgmlDType::Q8_0
                && q.device().is_cuda()
                && q.shape().dims() == [w.n_v_heads, hidden]
        };
        Ok((fits(wb) && fits(wa)).then(|| (wb.clone(), wa.clone())))
    }

    /// [`super::DeltaNetWeights::gates`] on a card: the projections and the
    /// rest in one launch where both weights are f32 on the card, or Q8_0 on
    /// it ([`q8_0_gate_weights`]), and the step is one sequence of at most
    /// [`GATES_PROJECTED_ROWS`] rows, else the projections as matmuls and the
    /// rest in one (bit for bit the composition). `None` where `dt_bias` and
    /// `a` are not f32 `[nv]` on the card — the caller composes.
    pub(super) fn gates(w: &super::DeltaNetWeights, x: &Tensor) -> Result<Option<super::Gates>> {
        let nv = w.n_v_heads;
        let (b, seq, hidden) = x.dims3()?;
        let on_card =
            |t: &Tensor| t.dtype() == DType::F32 && t.is_contiguous() && t.device().is_cuda();
        if !(on_card(&w.dt_bias) && on_card(&w.a) && w.dt_bias.dims() == [nv] && w.a.dims() == [nv])
        {
            return Ok(None);
        }
        let vectors = GateVectors {
            dt_bias: w.dt_bias.clone(),
            a: w.a.clone(),
            nv,
        };
        let flat = match (w.w_beta.dense_f32_weight(), w.w_alpha.dense_f32_weight()) {
            (Some(wb), Some(wa))
                if b == 1
                    && (1..=GATES_PROJECTED_ROWS).contains(&seq)
                    && [x, wb, wa].into_iter().all(on_card)
                    && wb.dims() == [nv, hidden]
                    && wa.dims() == [nv, hidden] =>
            {
                x.apply_op3_no_bwd(wb, wa, &GatesProjected(vectors))?
            }
            _ => match q8_0_gate_weights(w, x)? {
                Some((w_beta, w_alpha)) => x.apply_op1_no_bwd(&GatesProjectedQ8_0 {
                    vectors,
                    w_beta,
                    w_alpha,
                })?,
                None => {
                    let (beta, alpha) = (w.w_beta.forward(x)?, w.w_alpha.forward(x)?);
                    if !(on_card(&beta)
                        && on_card(&alpha)
                        && beta.dims() == [b, seq, nv]
                        && alpha.dims() == [b, seq, nv])
                    {
                        return Ok(Some(super::Gates::composed(
                            &beta, &alpha, &w.dt_bias, &w.a,
                        )?));
                    }
                    beta.apply_op2_no_bwd(&alpha, &GatesAfterProjections(vectors))?
                }
            },
        };
        let stacked = flat.reshape((2, b, seq, nv))?;
        Ok(Some(super::Gates {
            g: stacked.get(0)?,
            beta: stacked.get(1)?,
            stacked: Some(stacked),
        }))
    }

    /// `ssm_dt.bias` and `ssm_a`, `[nv]` each, f32 and contiguous on the
    /// card: what the gates' kernels read besides their operands.
    pub(super) struct GateVectors {
        dt_bias: Tensor,
        a: Tensor,
        nv: usize,
    }

    /// A step's gates from x `[1, seq, hidden]` and the f32 weights `[nv,
    /// hidden]` of `ssm_beta` and `ssm_alpha`, in one launch: [g | beta],
    /// `[2, seq, nv]` flat.
    pub(super) struct GatesProjected(GateVectors);

    /// A step's gates from its projections' outputs `[b, seq, nv]` (beta,
    /// alpha), in one launch: [g | beta], `[2, b, seq, nv]` flat.
    pub(super) struct GatesAfterProjections(GateVectors);

    /// A step's gates from x `[1, seq, hidden]` and the Q8_0 weights `[nv,
    /// hidden]` of `ssm_beta` and `ssm_alpha`, read in place, in one launch:
    /// [g | beta], `[2, seq, nv]` flat — bit for bit [`GatesProjected`] over
    /// the dequantized weights (each weight is an f16 times an int8, exact in
    /// f32), where the matmuls made four launches more a layer (#228 (1) (b)).
    pub(super) struct GatesProjectedQ8_0 {
        vectors: GateVectors,
        w_beta: std::sync::Arc<QTensor>,
        w_alpha: std::sync::Arc<QTensor>,
    }

    impl CustomOp1 for GatesProjectedQ8_0 {
        fn name(&self) -> &'static str {
            "ssm-gates-projected-q8-0"
        }

        /// Never reached: [`gates`] runs only on a card.
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ssm-gates-projected-q8-0's kernel is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let nv = self.vectors.nv;
            let (b, seq, hidden) = l1.shape().dims3()?;
            let fits =
                |q: &QTensor| q.dtype() == GgmlDType::Q8_0 && q.shape().dims() == [nv, hidden];
            if b != 1
                || !(1..=GATES_PROJECTED_ROWS).contains(&seq)
                || !hidden.is_multiple_of(32)
                || !fits(&self.w_beta)
                || !fits(&self.w_alpha)
            {
                candle_core::bail!("ssm-gates-projected-q8-0: unexpected shapes");
            }
            let Some((xo, xe)) = l1.contiguous_offsets() else {
                candle_core::bail!("ssm-gates-projected-q8-0: x must be contiguous");
            };
            let (dt_storage, dt_layout) = self.vectors.dt_bias.storage_and_layout();
            let (a_storage, a_layout) = self.vectors.a.storage_and_layout();
            let (candle_core::Storage::Cuda(dt), candle_core::Storage::Cuda(av)) =
                (&*dt_storage, &*a_storage)
            else {
                candle_core::bail!("ssm-gates-projected-q8-0: dt_bias and a are not on the card");
            };
            let (Some((dto, dte)), Some((avo, ave))) = (
                dt_layout.contiguous_offsets(),
                a_layout.contiguous_offsets(),
            ) else {
                candle_core::bail!("ssm-gates-projected-q8-0: dt_bias and a must be contiguous");
            };
            let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
            // The weights' Q8_0 blocks from their first byte, read in place:
            // the model's own, never freed while it runs.
            let w_beta = self.w_beta.device_ptr()? as u64;
            let w_alpha = self.w_alpha.device_ptr()? as u64;
            let dt = dt.as_cuda_slice::<f32>()?.slice(dto..dte);
            let a = av.as_cuda_slice::<f32>()?.slice(avo..ave);
            let n_out = 2 * seq * nv;
            // Every element is assigned, as `GatesProjected`'s.
            let mut out = dev.alloc_fully_overwritten::<f32>(n_out)?;
            let func = dev.get_or_load_custom_func(
                GATES_PROJECTED_Q8_0[seq - 1],
                "swarmllm_delta_rule",
                super::DELTA_RULE_PTX,
            )?;
            let cfg = LaunchConfig {
                grid_dim: (nv as u32, 2, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let (hidden_i, nv_i) = (hidden as i32, nv as i32);
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&w_beta);
            builder.arg(&w_alpha);
            builder.arg(&dt);
            builder.arg(&a);
            builder.arg(&mut out);
            builder.arg(&hidden_i);
            builder.arg(&nv_i);
            // SAFETY: ffi. Shapes and contiguity are checked above: the kernel
            // reads x [seq, hidden], each weight's nv · hidden / 32 Q8_0 blocks
            // from its first byte (a QTensor's data starts there, padded after
            // its end), dt_bias and a [nv] (`gates` checked them), and writes
            // each of `out`'s 2·seq·nv elements once; `seq` names the entry.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&[n_out]),
            ))
        }
    }

    impl CustomOp3 for GatesProjected {
        fn name(&self) -> &'static str {
            "ssm-gates-projected"
        }

        /// Never reached: [`gates`] runs only on a card.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ssm-gates-projected's kernel is CUDA-only")
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
            let nv = self.0.nv;
            let (b, seq, hidden) = l1.shape().dims3()?;
            if b != 1
                || !(1..=GATES_PROJECTED_ROWS).contains(&seq)
                || l2.shape().dims() != [nv, hidden]
                || l3.shape().dims() != [nv, hidden]
            {
                candle_core::bail!("ssm-gates-projected: unexpected shapes");
            }
            let (Some((xo, xe)), Some((bo, be)), Some((ao, ae))) = (
                l1.contiguous_offsets(),
                l2.contiguous_offsets(),
                l3.contiguous_offsets(),
            ) else {
                candle_core::bail!("ssm-gates-projected: operands must be contiguous");
            };
            let (dt_storage, dt_layout) = self.0.dt_bias.storage_and_layout();
            let (a_storage, a_layout) = self.0.a.storage_and_layout();
            let (candle_core::Storage::Cuda(dt), candle_core::Storage::Cuda(av)) =
                (&*dt_storage, &*a_storage)
            else {
                candle_core::bail!("ssm-gates-projected: dt_bias and a are not on the card");
            };
            let (Some((dto, dte)), Some((avo, ave))) = (
                dt_layout.contiguous_offsets(),
                a_layout.contiguous_offsets(),
            ) else {
                candle_core::bail!("ssm-gates-projected: dt_bias and a must be contiguous");
            };
            let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
            let w_beta = s2.as_cuda_slice::<f32>()?.slice(bo..be);
            let w_alpha = s3.as_cuda_slice::<f32>()?.slice(ao..ae);
            let dt = dt.as_cuda_slice::<f32>()?.slice(dto..dte);
            let a = av.as_cuda_slice::<f32>()?.slice(avo..ave);
            let n_out = 2 * seq * nv;
            // Every element is assigned: each (row, head) g by its alpha
            // block's thread `row`, beta by its beta block's.
            let mut out = dev.alloc_fully_overwritten::<f32>(n_out)?;
            let func = dev.get_or_load_custom_func(
                GATES_PROJECTED[seq - 1],
                "swarmllm_delta_rule",
                super::DELTA_RULE_PTX,
            )?;
            let cfg = LaunchConfig {
                grid_dim: (nv as u32, 2, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let (hidden_i, nv_i) = (hidden as i32, nv as i32);
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&w_beta);
            builder.arg(&w_alpha);
            builder.arg(&dt);
            builder.arg(&a);
            builder.arg(&mut out);
            builder.arg(&hidden_i);
            builder.arg(&nv_i);
            // SAFETY: ffi. Shapes and contiguity are checked above: the kernel
            // reads x [seq, hidden], both weights [nv, hidden] and dt_bias and
            // a [nv] (`gates` checked them), and writes each of `out`'s
            // 2·seq·nv elements once; `seq` names the entry it launches.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&[n_out]),
            ))
        }
    }

    impl CustomOp2 for GatesAfterProjections {
        fn name(&self) -> &'static str {
            "ssm-gates"
        }

        /// Never reached: [`gates`] runs only on a card.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ssm-gates' kernel is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
            s2: &candle_core::CudaStorage,
            l2: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let nv = self.0.nv;
            let n = l1.shape().elem_count();
            if l2.shape() != l1.shape() || l1.shape().dims().last() != Some(&nv) {
                candle_core::bail!("ssm-gates: unexpected shapes");
            }
            let (Some((bo, be)), Some((ao, ae))) =
                (l1.contiguous_offsets(), l2.contiguous_offsets())
            else {
                candle_core::bail!("ssm-gates: operands must be contiguous");
            };
            let (dt_storage, dt_layout) = self.0.dt_bias.storage_and_layout();
            let (a_storage, a_layout) = self.0.a.storage_and_layout();
            let (candle_core::Storage::Cuda(dt), candle_core::Storage::Cuda(av)) =
                (&*dt_storage, &*a_storage)
            else {
                candle_core::bail!("ssm-gates: dt_bias and a are not on the card");
            };
            let (Some((dto, dte)), Some((avo, ave))) = (
                dt_layout.contiguous_offsets(),
                a_layout.contiguous_offsets(),
            ) else {
                candle_core::bail!("ssm-gates: dt_bias and a must be contiguous");
            };
            let beta = s1.as_cuda_slice::<f32>()?.slice(bo..be);
            let alpha = s2.as_cuda_slice::<f32>()?.slice(ao..ae);
            let dt = dt.as_cuda_slice::<f32>()?.slice(dto..dte);
            let a = av.as_cuda_slice::<f32>()?.slice(avo..ave);
            // Every element is assigned: g at i and beta at n + i by thread i.
            let mut out = dev.alloc_fully_overwritten::<f32>(2 * n)?;
            let func = dev.get_or_load_custom_func(
                "ssm_gates_f32",
                "swarmllm_delta_rule",
                super::DELTA_RULE_PTX,
            )?;
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (nv_i, n_i) = (nv as i32, n as i32);
            let mut builder = func.builder();
            builder.arg(&beta);
            builder.arg(&alpha);
            builder.arg(&dt);
            builder.arg(&a);
            builder.arg(&mut out);
            builder.arg(&nv_i);
            builder.arg(&n_i);
            // SAFETY: ffi. Both operands are `n` contiguous elements ending in
            // `nv` heads, dt_bias and a are [nv] (`gates` checked them); the
            // kernel reads head `i % nv` of each and writes `out`'s 2·n once.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&[2 * n]),
            ))
        }
    }

    /// q and k's L2 norm in one launch ([`super::l2_norm_on_the_card`]):
    /// rows of 64 or 128, contiguous; the output has the input's shape.
    pub(super) struct L2NormRows {
        pub(super) eps: f32,
    }

    impl CustomOp1 for L2NormRows {
        fn name(&self) -> &'static str {
            "l2-norm-rows"
        }

        /// Never reached: [`super::l2_norm_on_the_card`] wants a card.
        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("l2-norm-rows' kernel is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let width = l1.shape().dims().last().copied().unwrap_or(0);
            let name = match width {
                64 => "l2_norm_rows_f32_d64",
                128 => "l2_norm_rows_f32_d128",
                _ => candle_core::bail!("l2-norm-rows: rows of {width}"),
            };
            let Some((xo, xe)) = l1.contiguous_offsets() else {
                candle_core::bail!("l2-norm-rows: the input must be contiguous");
            };
            let n = l1.shape().elem_count();
            let rows = n / width;
            let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
            // Every element is assigned: each row by its warp.
            let mut out = dev.alloc_fully_overwritten::<f32>(n)?;
            let func =
                dev.get_or_load_custom_func(name, "swarmllm_delta_rule", super::DELTA_RULE_PTX)?;
            let cfg = LaunchConfig {
                grid_dim: (rows.div_ceil(4) as u32, 1, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let rows_i = rows as i32;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&mut out);
            builder.arg(&rows_i);
            builder.arg(&self.eps);
            // SAFETY: ffi. The input is `rows` contiguous rows of `width`, the
            // entry's width; a warp reads its row and writes it to `out`,
            // sized for exactly the input.
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                l1.shape().clone(),
            ))
        }
    }

    /// [`super::replay_rows`] on a card in two launches a layer: the
    /// convolution window (`conv_window_replay_f32`) and the delta rule over
    /// the kept rows (`delta_rule_replay_f32_d*`, the step's own update with
    /// no outputs), both written into `into`'s buffers straight from the rows
    /// kept, where the composition copied them first — 7 launches and 6
    /// allocations a layer: 168 a rewind of Qwen3.5-4B, 2.07 ms of which the
    /// card was busy 0.59 (#428, card, 2026-11-08). `false`, nothing written,
    /// outside its scope: one sequence, f32, contiguous buffers, a width
    /// [`super::card_delta_rule_covers`].
    ///
    /// Prior art: llama.cpp master writes a state per position of a check
    /// instead (`ggml_gated_delta_net(…, K)`, rollback picks a slot — 2 MB a
    /// slot a layer here); this keeps the replay's 32 KB a row and its
    /// bit-exactness, and drops only its launches.
    /// [`super::unwritten_like`] on a card: an f32 buffer with no memset;
    /// `None` for any other type or device.
    pub(super) fn unwritten_like(like: &Tensor) -> Result<Option<Tensor>> {
        let (candle_core::Device::Cuda(dev), DType::F32) = (like.device(), like.dtype()) else {
            return Ok(None);
        };
        let slice = dev.alloc_fully_overwritten::<f32>(like.elem_count())?;
        Ok(Some(Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                slice,
                dev.clone(),
            )),
            like.shape(),
            candle_core::op::BackpropOp::none(),
            false,
        )))
    }

    pub(super) fn replay(
        read: &super::SsmSlot,
        kept: &super::SsmRows,
        rows: usize,
        into: &super::SsmSlot,
    ) -> Result<bool> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let all = [
            &read.conv_state,
            &read.recurrent_state,
            &kept.mixed,
            &kept.qk,
            &kept.v,
            &kept.g,
            &kept.beta,
            &into.conv_state,
            &into.recurrent_state,
        ];
        if all.iter().any(|t| {
            t.dtype() != candle_core::DType::F32 || !t.is_contiguous() || !t.device().is_cuda()
        }) {
            return Ok(false);
        }
        let (b, cap, two_nk, kd) = kept.qk.dims4()?;
        let (_, channels, pad) = read.conv_state.dims3()?;
        let nv = kept.g.dim(2)?;
        let vd = kept.v.dim(3)?;
        if b != 1
            || two_nk % 2 != 0
            || !(1..=cap).contains(&rows)
            || !super::card_delta_rule_covers(kd, vd)
            || kept.v.dims() != [1, cap, nv, vd]
            || kept.mixed.dims() != [1, cap, channels]
            || kept.g.dims() != [1, cap, nv]
            || kept.beta.dims() != [1, cap, nv]
            || read.recurrent_state.dims() != [1, nv, vd, kd]
            || into.recurrent_state.dims() != [1, nv, vd, kd]
            || into.conv_state.dims() != [1, channels, pad]
        {
            return Ok(false);
        }
        let held: Vec<_> = all.iter().map(|t| t.storage_and_layout()).collect();
        let mut views = Vec::with_capacity(held.len());
        let mut device = None;
        for (storage, layout) in &held {
            let candle_core::Storage::Cuda(cs) = &**storage else {
                return Ok(false);
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                return Ok(false);
            };
            device.get_or_insert_with(|| cs.device.clone());
            views.push(cs.as_cuda_slice::<f32>()?.slice(start..end));
        }
        let (Some(dev), [read_conv, read_state, mixed, qk, v, g, beta, into_conv, into_state]) =
            (device, views.as_slice())
        else {
            return Ok(false);
        };

        let window = dev.get_or_load_custom_func(
            "conv_window_replay_f32",
            "swarmllm_delta_rule",
            super::DELTA_RULE_PTX,
        )?;
        let n = channels * pad;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (channels_i, pad_i, rows_i) = (channels as i32, pad as i32, rows as i32);
        let mut builder = window.builder();
        builder.arg(read_conv);
        builder.arg(mixed);
        builder.arg(into_conv);
        builder.arg(&channels_i);
        builder.arg(&pad_i);
        builder.arg(&rows_i);
        // SAFETY: ffi. Every buffer is contiguous f32 of the shape checked
        // above; the kernel reads `read` [C, pad] and the first `rows` ≤ cap
        // rows of `mixed` [cap, C], and writes each of `into`'s C·pad
        // elements once. `into` and `read` are a state's two pairs, never one
        // buffer; it is written through a read view, ordered by the ONE
        // stream per device, as `DeltaRule` writes its `state_out`.
        unsafe { builder.launch(cfg) }.w()?;

        let (name, grid_z) = super::delta_rule_entry(
            if kd == 64 {
                "delta_rule_replay_f32_d64"
            } else {
                "delta_rule_replay_f32_d128"
            },
            kd,
        );
        let rule =
            dev.get_or_load_custom_func(&name, "swarmllm_delta_rule", super::DELTA_RULE_PTX)?;
        let cfg = LaunchConfig {
            grid_dim: (nv as u32, 1, grid_z),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let (nk_i, nv_i) = ((two_nk / 2) as i32, nv as i32);
        let (sv_t, sv_h) = ((nv * vd) as i64, vd as i64);
        let mut builder = rule.builder();
        builder.arg(qk);
        builder.arg(v);
        builder.arg(g);
        builder.arg(beta);
        builder.arg(read_state);
        builder.arg(into_state);
        builder.arg(&nk_i);
        builder.arg(&nv_i);
        builder.arg(&rows_i);
        builder.arg(&sv_t);
        builder.arg(&sv_h);
        // SAFETY: ffi. As the step's launch (`DeltaRule::cuda_fwd`), for one
        // sequence of `rows` ≤ cap rows: qk [cap, 2·nk, kd], v [cap, nv, vd]
        // and g / beta [cap, nv] contiguous, the states [nv, vd, kd] with
        // kd == vd ∈ {64, 128}; it writes `into`'s state only, never `read`'s.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(true)
    }

    /// [`super::FlowRows::put`] on a card: row r of `new` `[1, rows, …]` to row
    /// `slots[r]` of `store` `[1, capacity, …]` (`flow_rows_put_f32`), one
    /// launch whatever the slots — so a captured level is one topology
    /// wherever its rows land. `false`, nothing written, outside its scope:
    /// one sequence, f32, contiguous, the same row shape, `slots` `[rows]` u32.
    pub(super) fn put_rows(store: &Tensor, new: &Tensor, slots: &Tensor) -> Result<bool> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let (store_dims, new_dims) = (store.dims(), new.dims());
        if store.dtype() != DType::F32
            || new.dtype() != DType::F32
            || slots.dtype() != DType::U32
            || !store.is_contiguous()
            || !new.is_contiguous()
            || !slots.is_contiguous()
            || store_dims.len() < 2
            || store_dims.len() != new_dims.len()
            || store_dims[0] != 1
            || new_dims[0] != 1
            || store_dims[2..] != new_dims[2..]
            || slots.dims() != [new_dims[1]]
        {
            return Ok(false);
        }
        let (capacity, rows) = (store_dims[1], new_dims[1]);
        let width: usize = store_dims[2..].iter().product();
        if rows == 0 || width == 0 {
            return Ok(true);
        }
        let held: Vec<_> = [store, new, slots]
            .iter()
            .map(|t| t.storage_and_layout())
            .collect();
        let mut cuda = Vec::with_capacity(3);
        for (storage, layout) in &held {
            let candle_core::Storage::Cuda(cs) = &**storage else {
                return Ok(false);
            };
            let Some(range) = layout.contiguous_offsets() else {
                return Ok(false);
            };
            cuda.push((cs, range));
        }
        let [(store_cs, store_at), (new_cs, new_at), (slots_cs, slots_at)] = cuda.as_slice() else {
            return Ok(false);
        };
        let dev = store_cs.device.clone();
        let into = store_cs
            .as_cuda_slice::<f32>()?
            .slice(store_at.0..store_at.1);
        let from = new_cs.as_cuda_slice::<f32>()?.slice(new_at.0..new_at.1);
        let at = slots_cs
            .as_cuda_slice::<u32>()?
            .slice(slots_at.0..slots_at.1);
        let func = dev.get_or_load_custom_func(
            "flow_rows_put_f32",
            "swarmllm_delta_rule",
            super::DELTA_RULE_PTX,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (width_i, capacity_i) = (width as i32, capacity as i32);
        let mut builder = func.builder();
        builder.arg(&from);
        builder.arg(&into);
        builder.arg(&at);
        builder.arg(&width_i);
        builder.arg(&capacity_i);
        // SAFETY: ffi. `new` is `rows` contiguous f32 rows of `width`, `store`
        // `capacity` of them, `slots` `rows` u32; block r copies row r to row
        // `slots[r]`, skipping one past `capacity`. The store is written
        // through a read view, ordered by the ONE stream per device, as
        // `DeltaRule` writes its `state_out`; `new` is another buffer (a
        // forward's fresh rows, never the store).
        unsafe { builder.launch(cfg) }.w()?;
        Ok(true)
    }

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
        /// A tree's parents (`super::TreeLines::on_card`, `[seq]` u32): each
        /// row starts from its parent's state (`delta_rule_tree_f32_d*`).
        pub(super) parents: Option<Tensor>,
        /// A flowing forward's walks (`super::TreeLines::walks_on_card`,
        /// `super::flow_walks`), beside `parents` (`delta_rule_flow_f32_d*`).
        pub(super) walks: Option<Tensor>,
        /// A flowing forward reading its stores in place (`super::SlotTable`):
        /// the third operand is g's store `[b, held, nv]` and this beta's, the
        /// rows the table's (`delta_rule_flow_stored_f32_d*`; a flow's only).
        pub(super) stored: Option<(Tensor, super::SlotTable)>,
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
            let (b, held, two_nk, d) = l1.shape().dims4()?;
            let (vb, vheld, nv, vd) = l2.shape().dims4()?;
            let vs = l2.stride();
            // Read in place, the operands are the stores, `held` rows each —
            // g's the third, beta's beside it — and the rows the table's.
            let (seq, gates_dims) = match &self.stored {
                Some((_, at)) => (at.table.dim(0)?, vec![b, held, nv]),
                None => (held, vec![2, b, held, nv]),
            };
            if two_nk != 2 * self.nk
                || (vb, vheld, vd) != (b, held, d)
                || vs[3] != 1
                || l3.shape().dims() != gates_dims.as_slice()
                || self.state.dims() != [b, nv, d, d]
                || self.state_out.dims() != [b, nv, d, d]
                || self.stored.as_ref().is_some_and(|(beta, at)| {
                    at.bound > held
                        || beta.dims() != gates_dims.as_slice()
                        || beta.dtype() != DType::F32
                        || self.walks.is_none()
                })
            {
                candle_core::bail!("gated-delta-rule: unexpected shapes");
            }
            let (Some((qo, qe)), Some((go, ge))) =
                (l1.contiguous_offsets(), l3.contiguous_offsets())
            else {
                candle_core::bail!("gated-delta-rule: qk and the gates must be contiguous");
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
            let gates = s3.as_cuda_slice::<f32>()?.slice(go..ge);
            let beta_held = self
                .stored
                .as_ref()
                .map(|(beta, _)| beta.storage_and_layout());
            let beta = match &beta_held {
                Some((storage, layout)) => {
                    let candle_core::Storage::Cuda(on_card) = &**storage else {
                        candle_core::bail!("gated-delta-rule: beta's store is not on the card");
                    };
                    let Some((bo, be)) = layout.contiguous_offsets() else {
                        candle_core::bail!("gated-delta-rule: beta's store must be contiguous");
                    };
                    Some(on_card.as_cuda_slice::<f32>()?.slice(bo..be))
                }
                None => None,
            };
            let state = st.as_cuda_slice::<f32>()?.slice(so..se);
            let state_out = st_out.as_cuda_slice::<f32>()?.slice(oo..oe);
            let n_out = b * seq * nv * d;
            // Every element is assigned: each (sequence, token, head, row)
            // output by its warp's lane 0. Each element of `state_out` is
            // assigned by its lane too.
            let mut dst = dev.alloc_fully_overwritten::<f32>(n_out)?;
            let tree = match &self.parents {
                Some(p) => Some(ParentsView::of(p, seq, "gated-delta-rule")?),
                None => None,
            };
            let parents = tree.as_ref().map(ParentsView::slice).transpose()?;
            // The walks name rows by index: every offset and row is checked
            // against the step's rows before the kernel reads one.
            let walked = match (&self.walks, &parents) {
                (Some(w), Some(_)) => Some(WalksView::of(w, seq)?),
                (Some(_), None) => candle_core::bail!("gated-delta-rule: walks with no tree"),
                (None, _) => None,
            };
            let walks = walked.as_ref().map(WalksView::slice).transpose()?;
            let table = match &self.stored {
                Some((_, at)) => Some(ParentsView::of(&at.table, seq, "gated-delta-rule's slots")?),
                None => None,
            };
            let slots = table.as_ref().map(ParentsView::slice).transpose()?;
            let (name, grid_z) = super::delta_rule_entry(
                match (d, parents.is_some(), walks.is_some(), slots.is_some()) {
                    (64, true, true, true) => "delta_rule_flow_stored_f32_d64",
                    (_, true, true, true) => "delta_rule_flow_stored_f32_d128",
                    (64, false, _, _) => "delta_rule_f32_d64",
                    (64, true, false, _) => "delta_rule_tree_f32_d64",
                    (64, true, true, false) => "delta_rule_flow_f32_d64",
                    (_, false, _, _) => "delta_rule_f32_d128",
                    (_, true, false, _) => "delta_rule_tree_f32_d128",
                    (_, true, true, false) => "delta_rule_flow_f32_d128",
                },
                d,
            );
            let func =
                dev.get_or_load_custom_func(&name, "swarmllm_delta_rule", super::DELTA_RULE_PTX)?;
            let cfg = LaunchConfig {
                grid_dim: (nv as u32, b as u32, grid_z),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let (nk_i, nv_i, seq_i, held_i) = (self.nk as i32, nv as i32, seq as i32, held as i32);
            let (sv_b, sv_t, sv_h) = (vs[0] as i64, vs[1] as i64, vs[2] as i64);
            let mut builder = func.builder();
            builder.arg(&qk);
            builder.arg(&v);
            builder.arg(&gates);
            if let Some(beta) = &beta {
                builder.arg(beta);
            }
            builder.arg(&state);
            builder.arg(&state_out);
            builder.arg(&mut dst);
            builder.arg(&nk_i);
            builder.arg(&nv_i);
            builder.arg(&seq_i);
            if slots.is_some() {
                builder.arg(&held_i);
            }
            builder.arg(&sv_b);
            builder.arg(&sv_t);
            builder.arg(&sv_h);
            builder.arg(&self.scale);
            if let Some(parents) = &parents {
                builder.arg(parents);
            }
            if let Some(walks) = &walks {
                builder.arg(walks);
            }
            if let Some(slots) = &slots {
                builder.arg(slots);
            }
            // SAFETY: ffi. Shapes and contiguity are checked above and the
            // width by `card_delta_rule_covers` (d ∈ {64, 128}, the kernel's
            // two entry points); every read is within the views by those
            // shapes and v's strides, `dst` is sized for exactly the outputs
            // and `state_out` is the state's shape. Read in place, each row is
            // read at its slot, every slot below the table's bound (checked on
            // the host, `FlowRows::rows_the_lines_read`) and the bound within
            // the stores' `held` rows (checked above). A flow's walks hold their
            // `seq + 1` offsets (`WalksView::of`), each naming rows below the
            // row it walks for, as `super::flow_walks` builds them. It is
            // written through a read view, which only the ONE stream per device
            // orders (`SWARMLLM_CUDA_OWN_STREAM`; event tracking is off by
            // default).
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
    /// 2026-10-25). The convolution's taps are multiples of 1/32 too, and
    /// `ssm_out` — which reads the delta rule's output, no multiple of
    /// anything — sums exactly two inner channels into each output at
    /// power-of-two weights: every product exact and one addition that
    /// rounds, the same in any order. With a dense `ssm_out` a tree's row and
    /// the same row read alone parted in the last bit on ARM (5.96e-8, CI's
    /// macOS job, 2026-11-19, `a_tree_checks_rows_are_their_lines_read_alone`).
    fn weights_of(device: &Device, exact_projections: bool) -> DeltaNetWeights {
        let (hidden, nk, kd, nv, vd, k) = (8usize, 2usize, 4usize, 4usize, 4usize, 4usize);
        let c = 2 * nk * kd + nv * vd;
        let mm = |o: usize, i: usize| {
            let w = Tensor::randn(0f32, 0.3, (o, i), device).unwrap();
            super::super::QMatMul::from_dense(if exact_projections { exact(w, 32.0) } else { w })
        };
        let ssm_out = if exact_projections {
            assert_eq!(nv * vd, 2 * hidden, "two inner channels an output");
            let mut pairs = vec![0f32; hidden * nv * vd];
            for o in 0..hidden {
                pairs[o * nv * vd + 2 * o] = 0.5;
                pairs[o * nv * vd + 2 * o + 1] = 0.25;
            }
            super::super::QMatMul::from_dense(
                Tensor::from_vec(pairs, (hidden, nv * vd), device).unwrap(),
            )
        } else {
            mm(hidden, nv * vd)
        };
        let conv1d = Tensor::randn(0f32, 0.3, (c, k), device).unwrap();
        let conv1d = if exact_projections {
            exact(conv1d, 32.0)
        } else {
            conv1d
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
            conv1d,
            ssm_norm: crate::inference::residual_norm::RmsNorm::from_qtensor(
                candle_core::quantized::QTensor::quantize(
                    &Tensor::ones((vd,), DType::F32, device).unwrap(),
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap(),
                1e-6,
            )
            .unwrap(),
            ssm_out,
            n_k_heads: nk,
            k_head_dim: kd,
            n_v_heads: nv,
            v_head_dim: vd,
            conv_kernel: k,
            eps: 1e-6,
            out_heads: ValueHeadOrder::Tiled,
        }
    }

    /// The loader reorders a Qwen3-Next file's rows still quantized (#390):
    /// whole rows are whole blocks, so the gathered matrix must be the chosen
    /// rows exactly — for a 32-value format and a 256-value K-quant alike. A
    /// row the matrix does not have is refused, never read past.
    #[test]
    fn rows_gathered_from_a_quantized_matrix_are_its_rows_exactly() {
        use candle_core::quantized::GgmlDType;
        let device = Device::Cpu;
        let rows = [4usize, 0, 5, 2];
        let ids = Tensor::new(&[4u32, 0, 5, 2], &device).unwrap();
        for (dtype, cols) in [(GgmlDType::Q8_0, 64), (GgmlDType::Q4K, 256)] {
            let dense = Tensor::randn(0f32, 1.0, (6, cols), &device).unwrap();
            let q = QTensor::quantize(&dense, dtype).unwrap();
            let got = rows_in_order(&q, &rows, &device).unwrap();
            assert_eq!(got.dtype(), dtype);
            let want = q
                .dequantize(&device)
                .unwrap()
                .index_select(&ids, 0)
                .unwrap();
            assert_eq!(
                got.dequantize(&device).unwrap().to_vec2::<f32>().unwrap(),
                want.to_vec2::<f32>().unwrap(),
                "{dtype:?}"
            );
            assert!(rows_in_order(&q, &[6], &device).is_err(), "{dtype:?}");
        }
    }

    /// A Qwen3-Next file keeps its value heads grouped by key head; the loader
    /// puts its rows in the kernels' tiled order and the layer puts `ssm_out`'s
    /// input back (#390). A tiled layer's weights, filed as Qwen3-Next files
    /// them — both input layouts — must answer as that layer does, over a
    /// prompt and the steps decoded after it.
    #[test]
    fn a_file_grouping_its_value_heads_answers_as_the_layer_it_files() {
        let device = Device::Cpu;
        let (hidden, nk, kd, nv, vd, k) = (8usize, 2usize, 4usize, 4usize, 4usize, 4usize);
        let c = 2 * nk * kd + nv * vd;
        let heads = GroupedValueHeads {
            n_k_heads: nk,
            n_v_heads: nv,
            k_head_dim: kd,
            v_head_dim: vd,
        };
        let randn =
            |rows: usize, cols: usize| Tensor::randn(0f32, 0.3, (rows, cols), &device).unwrap();
        let (qkv, z, out) = (
            randn(c, hidden),
            randn(nv * vd, hidden),
            randn(hidden, nv * vd),
        );
        let (beta, alpha, conv1d) = (randn(nv, hidden), randn(nv, hidden), randn(c, k));
        let dt_bias = Tensor::randn(0f32, 0.3, (nv,), &device).unwrap();
        let a = Tensor::randn(0f32, 0.3, (nv,), &device)
            .unwrap()
            .abs()
            .unwrap()
            .neg()
            .unwrap();

        // As a file holds them: its row `order[i]` is the kernel's row `i`.
        let filed = |t: &Tensor, order: &[usize]| {
            let mut kernel_row = vec![0u32; order.len()];
            for (i, &row) in order.iter().enumerate() {
                kernel_row[row] = i as u32;
            }
            t.contiguous()
                .unwrap()
                .index_select(&Tensor::new(kernel_row.as_slice(), &device).unwrap(), 0)
                .unwrap()
        };
        let quantized =
            |t: &Tensor| QTensor::quantize(t, candle_core::quantized::GgmlDType::F32).unwrap();
        let stacked = |a: &Tensor, b: &Tensor| Tensor::cat(&[a, b], 0).unwrap();
        let joined = |(a, b): (Vec<usize>, Vec<usize>)| a.into_iter().chain(b).collect::<Vec<_>>();
        let ba = quantized(&filed(
            &stacked(&beta, &alpha),
            &joined(heads.beta_alpha_rows()),
        ));
        let file_out = filed(&out.t().unwrap(), &heads.gate_rows())
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let file_conv = filed(&conv1d, &heads.qkv_rows());
        let file_dt = filed(&dt_bias, &heads.head_rows());
        let file_a = filed(&a, &heads.head_rows());

        let layer =
            |order: KernelOrder, ssm_out: &Tensor, out_heads: ValueHeadOrder| DeltaNetWeights {
                wqkv: order.wqkv,
                wz: order.wz,
                w_beta: order.w_beta,
                w_alpha: order.w_alpha,
                dt_bias: order.dt_bias,
                a: order.a,
                conv1d: order.conv1d,
                ssm_norm: crate::inference::residual_norm::RmsNorm::from_qtensor(
                    quantized(&Tensor::ones((vd,), DType::F32, &device).unwrap()),
                    1e-6,
                )
                .unwrap(),
                ssm_out: QMatMul::from_dense(ssm_out.clone()),
                n_k_heads: nk,
                k_head_dim: kd,
                n_v_heads: nv,
                v_head_dim: vd,
                conv_kernel: k,
                eps: 1e-6,
                out_heads,
            };
        let tiled = layer(
            KernelOrder {
                wqkv: QMatMul::from_dense(qkv.clone()),
                wz: QMatMul::from_dense(z.clone()),
                w_beta: QMatMul::from_dense(beta.clone()),
                w_alpha: QMatMul::from_dense(alpha.clone()),
                conv1d: conv1d.clone(),
                dt_bias: dt_bias.clone(),
                a: a.clone(),
            },
            &out,
            ValueHeadOrder::Tiled,
        );

        let x = Tensor::randn(0f32, 1.0, (1, 6, hidden), &device).unwrap();
        let run = |w: &DeltaNetWeights| {
            let mut state = None;
            let mut outs = vec![w
                .forward_deltanet_into(&x.narrow(1, 0, 4).unwrap(), &mut state, 0, false)
                .unwrap()];
            for t in 4..6 {
                outs.push(
                    w.forward_deltanet_into(&x.narrow(1, t, 1).unwrap(), &mut state, t, false)
                        .unwrap(),
                );
            }
            Tensor::cat(&outs, 1).unwrap()
        };
        let want = run(&tiled);
        for (layout, input) in [
            (
                "attn_qkv + attn_gate",
                InputProjections::Split {
                    qkv: quantized(&filed(&qkv, &heads.qkv_rows())),
                    gate: quantized(&filed(&z, &heads.gate_rows())),
                },
            ),
            (
                "ssm_in",
                InputProjections::Fused(quantized(&filed(
                    &stacked(&qkv, &z),
                    &joined(heads.fused_rows()),
                ))),
            ),
        ] {
            let order = heads
                .kernel_order(&input, &ba, &file_conv, &file_dt, &file_a, &device)
                .unwrap();
            let got = run(&layer(order, &file_out, ValueHeadOrder::GroupedByKeyHead));
            let off = got
                .sub(&want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(off < 1e-5, "{layout}: {off}");
        }
    }

    /// A header whose head counts do not describe the tensors is refused before
    /// a row list is sized by them — one claiming billions of heads would have
    /// aborted on the allocation (pre-release review of 0.3.268).
    #[test]
    fn a_head_geometry_the_tensors_do_not_hold_is_refused_before_it_sizes_anything() {
        let device = Device::Cpu;
        let q = |rows: usize, cols: usize| {
            QTensor::quantize(
                &Tensor::zeros((rows, cols), DType::F32, &device).unwrap(),
                candle_core::quantized::GgmlDType::F32,
            )
            .unwrap()
        };
        let input = InputProjections::Split {
            qkv: q(32, 8),
            gate: q(16, 8),
        };
        let ba = q(8, 8);
        let conv = Tensor::zeros((32, 4), DType::F32, &device).unwrap();
        let per_head = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let held = GroupedValueHeads {
            n_k_heads: 2,
            n_v_heads: 4,
            k_head_dim: 4,
            v_head_dim: 4,
        };
        assert!(held
            .kernel_order(&input, &ba, &conv, &per_head, &per_head, &device)
            .is_ok());
        for claimed in [
            GroupedValueHeads {
                n_v_heads: 8,
                ..held
            },
            GroupedValueHeads {
                k_head_dim: 8,
                ..held
            },
            GroupedValueHeads {
                n_k_heads: 1 << 40,
                n_v_heads: 1 << 41,
                k_head_dim: 1 << 40,
                v_head_dim: 1 << 40,
            },
        ] {
            let err = claimed
                .kernel_order(&input, &ba, &conv, &per_head, &per_head, &device)
                .err()
                .unwrap_or_else(|| panic!("{claimed:?} was taken"))
                .to_string();
            assert!(
                err.contains("where the header's heads make") || err.contains("overflows"),
                "{claimed:?}: {err}"
            );
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

    /// A DeltaNet step's convolution and SiLU in one launch on a card
    /// (`conv_on_the_card`) are the composition's bit for bit — q | k, v and
    /// the window after the step — at one position, a check's few and a
    /// prompt chunk's many, and the window lands in the buffer it is given.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_convolution_is_the_composed_one_bit_for_bit() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let w = weights_of(&device, false);
        let channels = w.conv1d.dim(0).unwrap();
        let split = 2 * w.key_dim();
        for seq in [1usize, 2, 3, 8, 37] {
            let x = Tensor::randn(0f32, 1.0, (1, seq, channels), &device).unwrap();
            let prev = Tensor::randn(0f32, 1.0, (1, channels, 3), &device).unwrap();
            let (conv, window) = w.causal_conv(&x, &prev).unwrap();
            let conv = candle_nn::ops::silu(&conv).unwrap();
            let state_out = Tensor::zeros((1, channels, 3), DType::F32, &device).unwrap();
            let (qk, v) = w
                .conv_on_the_card(&x, &prev, &state_out, None, None)
                .unwrap()
                .expect("the one-launch convolution took the step");
            let qk = qk.reshape((1, seq, split)).unwrap();
            let v = v.reshape((1, seq, channels - split)).unwrap();
            assert_eq!(
                max_abs_diff(&qk, &conv.narrow(2, 0, split).unwrap()),
                0.0,
                "{seq} positions: q | k"
            );
            assert_eq!(
                max_abs_diff(&v, &conv.narrow(2, split, channels - split).unwrap()),
                0.0,
                "{seq} positions: v"
            );
            assert_eq!(
                max_abs_diff(&state_out, &window),
                0.0,
                "{seq} positions: the window after the step"
            );
        }

        // A tree's rows (#437) read up their own lines in the one launch too,
        // bit for bit `causal_conv_along`; its window is the chain's. Red if
        // the kernel read the rows before each row (rows 3, 5, 6 and 7 land
        // elsewhere), or the chain's kernel ran.
        for parents in [
            vec![
                None,
                Some(0),
                Some(1),
                Some(0),
                Some(3),
                Some(1),
                None,
                Some(6),
            ],
            vec![None, Some(0)],
            vec![None, None, None],
        ] {
            let seq = parents.len();
            let x = Tensor::randn(0f32, 1.0, (1, seq, channels), &device).unwrap();
            let prev = Tensor::randn(0f32, 1.0, (1, channels, 3), &device).unwrap();
            let (conv, window) = w.causal_conv_along(&x, &prev, &parents).unwrap();
            let conv = candle_nn::ops::silu(&conv).unwrap();
            let state_out = Tensor::zeros((1, channels, 3), DType::F32, &device).unwrap();
            let lines = TreeLines::new(parents.clone(), &device).unwrap();
            let (qk, v) = w
                .conv_on_the_card(&x, &prev, &state_out, Some(&lines), None)
                .unwrap()
                .expect("the one-launch convolution took the tree");
            let qk = qk.reshape((1, seq, split)).unwrap();
            let v = v.reshape((1, seq, channels - split)).unwrap();
            let both = Tensor::cat(&[&qk, &v], 2).unwrap();
            assert_eq!(max_abs_diff(&both, &conv), 0.0, "tree {parents:?}");
            assert_eq!(
                max_abs_diff(&state_out, &window),
                0.0,
                "tree {parents:?}: window"
            );
            if parents
                .iter()
                .enumerate()
                .any(|(i, p)| *p != i.checked_sub(1))
            {
                let (chain, _) = w.causal_conv(&x, &prev).unwrap();
                let chain = candle_nn::ops::silu(&chain).unwrap();
                assert!(
                    max_abs_diff(&both, &chain) > 1e-3,
                    "tree {parents:?}: read as a chain"
                );
            }
        }
    }

    /// Weights with Qwen3.5-4B's gate sizes (hidden 2560, 32 value heads) on
    /// `device`: `ssm_beta` and `ssm_alpha` dense f32, or quantized to `as_type`.
    #[cfg(feature = "candle-cuda")]
    fn gate_weights(
        device: &Device,
        as_type: Option<candle_core::quantized::GgmlDType>,
    ) -> DeltaNetWeights {
        let (hidden, nv) = (2560usize, 32usize);
        let r = |dims: &[usize], s: f32| Tensor::randn(0f32, s, dims, &Device::Cpu).unwrap();
        let mm = |t: Tensor| match as_type {
            Some(dtype) => super::super::QMatMul::from_qtensor(
                QTensor::quantize_onto(&t, dtype, device).unwrap(),
            )
            .unwrap(),
            None => super::super::QMatMul::from_dense(t.to_device(device).unwrap()),
        };
        let mut w = weights_of(device, false);
        w.w_beta = mm(r(&[nv, hidden], 0.02));
        w.w_alpha = mm(r(&[nv, hidden], 0.02));
        w.dt_bias = r(&[nv], 0.3).to_device(device).unwrap();
        w.a = r(&[nv], 0.3)
            .abs()
            .unwrap()
            .neg()
            .unwrap()
            .to_device(device)
            .unwrap();
        w.n_v_heads = nv;
        w
    }

    /// A step's gates on a card (`cuda::gates`) with f32 weights — the
    /// projections and the rest in one launch — are the composition's to
    /// float rounding at every width up to 8, and each row's are its gates
    /// read alone bit for bit (a check's rows are its rows one at a time).
    /// Past 8 rows the matmuls run and the rest is bit for bit the
    /// composition. The comparison sees a gate in the other block.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_gates_are_the_composed_ones() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let w = gate_weights(&device, None);
        let x = Tensor::randn(0f32, 1.0, (1, 9, 2560), &device).unwrap();
        let composed = |x: &Tensor| {
            Gates::composed(
                &w.w_beta.forward(x).unwrap(),
                &w.w_alpha.forward(x).unwrap(),
                &w.dt_bias,
                &w.a,
            )
            .unwrap()
        };
        let card = |x: &Tensor| {
            cuda::gates(&w, x)
                .unwrap()
                .expect("the card took the gates")
        };
        let alone: Vec<Gates> = (0..8).map(|t| card(&x.narrow(1, t, 1).unwrap())).collect();
        for seq in 1..=9usize {
            let x = x.narrow(1, 0, seq).unwrap();
            let (want, got) = (composed(&x), card(&x));
            let (dg, db) = (
                max_abs_diff(&got.g, &want.g),
                max_abs_diff(&got.beta, &want.beta),
            );
            if seq > 8 {
                assert_eq!((dg, db), (0.0, 0.0), "{seq} rows: after the matmuls");
                continue;
            }
            assert!(dg < 1e-5 && db < 1e-5, "{seq} rows: g {dg}, beta {db} off");
            assert!(
                max_abs_diff(&got.g, &want.beta) > 0.1,
                "{seq} rows: g and beta must be told apart"
            );
            for (t, one) in alone.iter().enumerate().take(seq) {
                let row = |a: &Tensor| a.narrow(1, t, 1).unwrap();
                assert_eq!(
                    (
                        max_abs_diff(&row(&got.g), &one.g),
                        max_abs_diff(&row(&got.beta), &one.beta)
                    ),
                    (0.0, 0.0),
                    "{seq} rows: row {t} is its gates read alone"
                );
            }
        }
    }

    /// With weights a file quantized, the gates' matmuls run as they do and
    /// the rest in one launch (`ssm_gates_f32`) is the composition bit for
    /// bit, at a step's, a check's and a prompt chunk's widths — every width
    /// for a type the gates' own launch does not read (Q4_0), past 8 rows for
    /// Q8_0, which it reads up to 8 (the test below).
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_gates_after_quantized_projections_are_the_composed_ones_bit_for_bit() {
        use candle_core::quantized::GgmlDType;
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        for (dtype, widths) in [
            (GgmlDType::Q4_0, &[1usize, 2, 5, 8, 9, 37][..]),
            (GgmlDType::Q8_0, &[9, 37][..]),
        ] {
            let w = gate_weights(&device, Some(dtype));
            assert!(w.w_beta.dense_f32_weight().is_none());
            the_gates_after_the_matmuls_are_composed(&w, &device, widths);
        }
    }

    #[cfg(feature = "candle-cuda")]
    fn the_gates_after_the_matmuls_are_composed(
        w: &DeltaNetWeights,
        device: &Device,
        widths: &[usize],
    ) {
        for &seq in widths {
            let x = Tensor::randn(0f32, 1.0, (1, seq, 2560), device).unwrap();
            let (beta, alpha) = (
                w.w_beta.forward(&x).unwrap(),
                w.w_alpha.forward(&x).unwrap(),
            );
            let want = Gates::composed(&beta, &alpha, &w.dt_bias, &w.a).unwrap();
            let got = cuda::gates(w, &x)
                .unwrap()
                .expect("the card took the gates");
            assert!(got.stacked.is_some(), "{seq} rows: written by the card");
            assert_eq!(max_abs_diff(&got.g, &want.g), 0.0, "{seq} rows: g");
            assert_eq!(max_abs_diff(&got.beta, &want.beta), 0.0, "{seq} rows: beta");
        }
    }

    /// Q8_0 `ssm_beta` and `ssm_alpha` read in place by the gates' one launch
    /// (`ssm_gates_project_q8_0_r*`, #228 (1) (b)) are, bit for bit, the f32
    /// launch over the same weights dequantized — a weight is an f16 times an
    /// int8, exact in f32 — at every width up to 8, so each row's gates are
    /// its gates read alone as the f32 launch's are. Against the matmuls,
    /// which round x to Q8_1 first, they agree to that rounding. Red if the
    /// Q8_0 weights take the matmuls: those round x, so the first comparison
    /// differs.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_gates_read_q8_0_weights_as_their_f32_dequantization_bit_for_bit() {
        use candle_core::quantized::GgmlDType;
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let q = gate_weights(&device, Some(GgmlDType::Q8_0));
        let dense = |m: &super::super::QMatMul| {
            super::super::QMatMul::from_dense(
                m.quantized_weight()
                    .expect("quantized")
                    .dequantize(&device)
                    .unwrap(),
            )
        };
        let mut f = q.clone();
        f.w_beta = dense(&q.w_beta);
        f.w_alpha = dense(&q.w_alpha);
        let x = Tensor::randn(0f32, 1.0, (1, 8, 2560), &device).unwrap();
        let mut worst_against_the_matmuls = 0f32;
        for seq in 1..=8usize {
            let x = x.narrow(1, 0, seq).unwrap();
            let card = |w: &DeltaNetWeights| {
                cuda::gates(w, &x)
                    .unwrap()
                    .expect("the card took the gates")
            };
            let (got, want) = (card(&q), card(&f));
            assert!(got.stacked.is_some(), "{seq} rows: written by the card");
            assert_eq!(max_abs_diff(&got.g, &want.g), 0.0, "{seq} rows: g");
            assert_eq!(max_abs_diff(&got.beta, &want.beta), 0.0, "{seq} rows: beta");
            assert!(
                max_abs_diff(&got.g, &got.beta) > 0.1,
                "{seq} rows: g and beta must be told apart"
            );
            let matmuls = Gates::composed(
                &q.w_beta.forward(&x).unwrap(),
                &q.w_alpha.forward(&x).unwrap(),
                &q.dt_bias,
                &q.a,
            )
            .unwrap();
            // The matmuls round x to Q8_1 (32-value blocks at max|x| / 127: an
            // error of std ~0.006 an element), so a projection of std ~1 here
            // (weights 0.02 over 2,560) is off by std ~0.006, a gate by at most
            // a quarter of that through the sigmoid; 0.003-0.006 measured over
            // the eight widths (2026-11-23).
            let off = max_abs_diff(&got.g, &matmuls.g).max(max_abs_diff(&got.beta, &matmuls.beta));
            assert!(off < 1e-2, "{seq} rows: {off} from the matmuls' gates");
            worst_against_the_matmuls = worst_against_the_matmuls.max(off);
        }
        assert!(
            worst_against_the_matmuls > 0.0,
            "the Q8_0 weights took the matmuls: nothing rounded x"
        );
        eprintln!(
            "Q8_0 gates in one launch: {worst_against_the_matmuls} from the matmuls' at most"
        );
    }

    /// q and k's L2 norm in one launch on a card is the composition's to
    /// float rounding, for both head widths, and a row reads only itself.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_l2_norm_is_the_composed_one() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        for width in [64usize, 128] {
            let x = Tensor::randn(0f32, 1.0, (1, 8, 4, width), &device).unwrap();
            let got = l2_norm_on_the_card(&x, 1e-6)
                .unwrap()
                .expect("the card took the norm");
            let want = l2_norm(&x, 1e-6).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff < 1e-6, "width {width}: {diff} off");
            let first = l2_norm_on_the_card(&x.narrow(1, 0, 1).unwrap(), 1e-6)
                .unwrap()
                .unwrap();
            assert_eq!(
                max_abs_diff(&got.narrow(1, 0, 1).unwrap(), &first),
                0.0,
                "width {width}: a row reads only itself"
            );
        }
    }

    /// Weights at Qwen 3.5's head widths (kd = vd = 128, which the card's
    /// delta rule takes) and a small hidden size, the same values on the
    /// processor and on `device`: `(processor, device)`.
    #[cfg(feature = "candle-cuda")]
    fn head_width_weights(device: &Device) -> (DeltaNetWeights, DeltaNetWeights) {
        let (hidden, nk, kd, nv, vd, k) = (256usize, 2usize, 128usize, 4usize, 128usize, 4usize);
        let c = 2 * nk * kd + nv * vd;
        let r = |dims: &[usize], s: f32| Tensor::randn(0f32, s, dims, &Device::Cpu).unwrap();
        let (qkv, z, beta, alpha, out) = (
            r(&[c, hidden], 0.06),
            r(&[nv * vd, hidden], 0.06),
            r(&[nv, hidden], 0.06),
            r(&[nv, hidden], 0.06),
            r(&[hidden, nv * vd], 0.04),
        );
        let (dt_bias, conv1d) = (r(&[nv], 0.3), r(&[c, k], 0.3));
        let a = r(&[nv], 0.3).abs().unwrap().neg().unwrap();
        let on = |dev: &Device| {
            let mv = |t: &Tensor| t.to_device(dev).unwrap();
            let mm = |t: &Tensor| super::super::QMatMul::from_dense(mv(t));
            DeltaNetWeights {
                wqkv: mm(&qkv),
                wz: mm(&z),
                w_beta: mm(&beta),
                w_alpha: mm(&alpha),
                dt_bias: mv(&dt_bias),
                a: mv(&a),
                conv1d: mv(&conv1d),
                ssm_norm: crate::inference::residual_norm::RmsNorm::from_qtensor(
                    QTensor::quantize(
                        &Tensor::ones((vd,), DType::F32, dev).unwrap(),
                        candle_core::quantized::GgmlDType::F32,
                    )
                    .unwrap(),
                    1e-6,
                )
                .unwrap(),
                ssm_out: mm(&out),
                n_k_heads: nk,
                k_head_dim: kd,
                n_v_heads: nv,
                v_head_dim: vd,
                conv_kernel: k,
                eps: 1e-6,
                out_heads: ValueHeadOrder::Tiled,
            }
        };
        (on(&Device::Cpu), on(device))
    }

    /// A DeltaNet layer on a card — its convolution, gates and L2 norm one
    /// launch each, its delta rule reading the gates' two blocks — writes what
    /// the processor's writes, to float rounding: a prompt chunk, single
    /// steps and a 3-row check, each from the state the last one left. The
    /// gates swapped land far off, so the comparison sees a block misread.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_deltanet_layer_on_the_card_is_the_processors() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let (on_cpu, on_card) = head_width_weights(&device);
        assert!(on_card.delta_rule_runs_on_the_card_kernel());
        let x = Tensor::randn(0f32, 1.0, (1, 18, 256), &Device::Cpu).unwrap();
        let (mut cpu_state, mut card_state) = (None, None);
        let mut at = 0;
        for len in [12usize, 1, 1, 3, 1] {
            let chunk = x.narrow(1, at, len).unwrap();
            let want = on_cpu
                .forward_deltanet_into(&chunk, &mut cpu_state, at, false)
                .unwrap();
            let got = on_card
                .forward_deltanet_into(
                    &chunk.to_device(&device).unwrap(),
                    &mut card_state,
                    at,
                    true,
                )
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let scale = want.abs().unwrap().flatten_all().unwrap().max(0).unwrap();
            let scale: f32 = scale.to_scalar().unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(
                diff <= 1e-4 * scale.max(1.0),
                "{len} rows at {at}: {diff} off (outputs up to {scale})"
            );
            at += len;
        }

        // The null control: the gates swapped, through the card's delta rule.
        let qk = l2_norm(
            &Tensor::randn(0f32, 1.0, (1, 3, 4, 128), &device).unwrap(),
            1e-6,
        )
        .unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 3, 4, 128), &device).unwrap();
        let g = Tensor::rand(0f32, 1.0, (1, 3, 4), &device)
            .unwrap()
            .neg()
            .unwrap();
        let beta = Tensor::rand(0f32, 1.0, (1, 3, 4), &device).unwrap();
        let state = Tensor::randn(0f32, 0.3, (1, 4, 128, 128), &device).unwrap();
        let run = |gates: &Gates| {
            delta_rule_into(&qk, 2, &v, gates, &state, None, None, None)
                .unwrap()
                .0
        };
        let right = run(&Gates::apart(g.clone(), beta.clone()));
        let swapped = run(&Gates::apart(beta, g));
        assert!(
            max_abs_diff(&right, &swapped) > 1e-2,
            "the delta rule must read g and beta from their own blocks"
        );
    }

    /// A rewind on a card in two launches a layer (`cuda::replay`) lands
    /// where the composed replay does, bit for bit, for every count of rows
    /// a check keeps — window and state — and is the path that ran. A replay
    /// one row off lands elsewhere, so the comparison sees the error a wrong
    /// window or row count makes. Qwen 3.5's widths (kd = vd = 128).
    ///
    /// Both write their target WHOLE, which lets `FlowRows::advance_to` hand
    /// them unzeroed buffers (`unwritten_like`): every target starts as NaN,
    /// and an element either leaves unwritten fails here.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_replay_on_the_card_lands_where_the_composed_replay_does() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let (nk, d, nv, cap, pad) = (2usize, 128usize, 4usize, 8usize, 3usize);
        let channels = 2 * nk * d + nv * d;
        let randn = |dims: &[usize], s: f32| Tensor::randn(0f32, s, dims, &device).unwrap();
        let unit = |dims: &[usize]| Tensor::rand(0f32, 1f32, dims, &device).unwrap();
        let read = SsmSlot {
            conv_state: randn(&[1, channels, pad], 1.0),
            recurrent_state: randn(&[1, nv, d, d], 0.3),
        };
        let kept = SsmRows {
            mixed: randn(&[1, cap, channels], 1.0),
            qk: l2_norm(&randn(&[1, cap, 2 * nk, d], 1.0), 1e-6).unwrap(),
            v: randn(&[1, cap, nv, d], 1.0),
            g: unit(&[1, cap, nv]).neg().unwrap(),
            beta: unit(&[1, cap, nv]),
        };
        let blank = || SsmSlot {
            conv_state: Tensor::full(f32::NAN, (1, channels, pad), &device).unwrap(),
            recurrent_state: Tensor::full(f32::NAN, (1, nv, d, d), &device).unwrap(),
        };
        // On the host: a card reduction's max may pass a NaN over.
        let written_whole = |slot: &SsmSlot, what: &str| {
            for t in [&slot.conv_state, &slot.recurrent_state] {
                let unwritten = t
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .filter(|x| x.is_nan())
                    .count();
                assert_eq!(unwritten, 0, "{what}: elements left unwritten");
            }
        };
        let fused = |rows: usize| {
            let into = blank();
            assert!(
                cuda::replay(&read, &kept, rows, &into).unwrap(),
                "the two-launch replay took {rows} rows"
            );
            into
        };
        for rows in 1..cap {
            let composed = replay_rows_composed(&read, &kept, rows, Some(&blank())).unwrap();
            let into = fused(rows);
            written_whole(&composed, &format!("{rows} rows composed"));
            written_whole(&into, &format!("{rows} rows in two launches"));
            assert_eq!(
                max_abs_diff(&composed.conv_state, &into.conv_state),
                0.0,
                "{rows} rows: convolution window"
            );
            assert_eq!(
                max_abs_diff(&composed.recurrent_state, &into.recurrent_state),
                0.0,
                "{rows} rows: recurrent state"
            );
            let off = fused(rows + 1);
            assert!(
                max_abs_diff(&composed.conv_state, &off.conv_state) > 0.0
                    && max_abs_diff(&composed.recurrent_state, &off.recurrent_state) > 0.0,
                "{rows} rows: a replay one row further must land elsewhere"
            );
        }
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

    /// A tree check's rows (#437) are each what the row's own line gives read
    /// alone — the step from the state before the tree over the row's
    /// ancestors and itself, in order: every row, bit for bit, both ways a
    /// step writes its state. The state a branching tree leaves (its last
    /// row's) is refused until the verdict takes it back; taken back into the
    /// in-place path (the rows whose line is the rows before them) it is where
    /// that path read alone leads. Red if a row's convolution reads the rows
    /// before it instead of its line, if its delta rule starts from the
    /// previous row's state, or if the tree's end could be read on from.
    #[test]
    fn a_tree_checks_rows_are_their_lines_read_alone() {
        use crate::inference::draft_tree::DraftTree;
        let device = Device::Cpu;
        // Exact projections, as `a_check_rewound_into_its_middle_is_where_its_kept_rows_lead`.
        let w = weights_of(&device, true);
        let x = exact(Tensor::randn(0f32, 1.0, (1, 10, 8), &device).unwrap(), 8.0);
        let tree_x = x.narrow(1, 3, 7).unwrap();
        // Rows 0-2 one line (in place); 3 a second child of 0 and 4 its child;
        // 5 a second child of 1; 6 a second line from the context itself.
        let parents: Vec<Option<usize>> =
            vec![None, Some(0), Some(1), Some(0), Some(3), Some(1), None];
        let as_sent: Vec<Option<u16>> = parents.iter().map(|p| p.map(|p| p as u16)).collect();
        let tree = DraftTree::new(3, as_sent.clone()).unwrap();
        let (positions, seen) = tree.rows_for(3, 7).unwrap();
        assert_eq!(
            seen.parents().unwrap(),
            parents,
            "the parents read off the tree's bits"
        );
        // A recurrent layer reads a tree whole, never its last rows alone.
        let grown = DraftTree::new(3, as_sent).unwrap();
        assert!(grown.rows_for(6, 4).unwrap().1.parents().is_err());

        for into_spare in [false, true] {
            let run = |state: &mut Option<SsmState>, rows: &Tensor, from: usize| {
                w.forward_deltanet_into(rows, state, from, into_spare)
            };
            let prompt = || {
                let mut s = None;
                run(&mut s, &x.narrow(1, 0, 3).unwrap(), 0).unwrap();
                s
            };
            let mut held = prompt();
            let out = {
                let _tree = super::super::RowPositions::for_tree(positions.clone(), seen.clone());
                run(&mut held, &tree_x, 3).unwrap()
            };
            for row in 0..parents.len() {
                let mut line = vec![row];
                while let Some(p) = parents[*line.last().unwrap()] {
                    line.push(p);
                }
                line.reverse();
                let rows: Vec<Tensor> = line
                    .iter()
                    .map(|&r| tree_x.narrow(1, r, 1).unwrap())
                    .collect();
                let mut alone = prompt();
                let want = run(&mut alone, &Tensor::cat(&rows, 1).unwrap(), 3).unwrap();
                assert_eq!(
                    max_abs_diff(
                        &out.narrow(1, row, 1).unwrap(),
                        &want.narrow(1, line.len() - 1, 1).unwrap()
                    ),
                    0.0,
                    "into_spare={into_spare}: row {row}, its line {line:?}"
                );
            }

            let h = held.as_mut().unwrap();
            assert!(h.is_branched());
            assert!(
                !h.can_rewind_to(10) && !h.rewind_to(10).unwrap(),
                "the tree's end is no history"
            );
            assert!(
                run(&mut held, &x.narrow(1, 0, 1).unwrap(), 10).is_err(),
                "nothing reads on from a branching tree's state"
            );
            // The verdict kept rows 0-2, in place: the take-back replays them.
            let h = held.as_mut().unwrap();
            assert!(h.rewind_to(6).unwrap() && !h.is_branched());
            let mut alone = prompt();
            run(&mut alone, &tree_x.narrow(1, 0, 3).unwrap(), 3).unwrap();
            let (got, want) = (held.as_ref().unwrap(), alone.as_ref().unwrap());
            assert_eq!(got.positions(), want.positions());
            assert_eq!(
                max_abs_diff(
                    &got.current().recurrent_state,
                    &want.current().recurrent_state
                ),
                0.0,
                "into_spare={into_spare}: the state taken back into the kept path"
            );
            assert_eq!(
                max_abs_diff(&got.current().conv_state, &want.current().conv_state),
                0.0,
                "into_spare={into_spare}: the window taken back into the kept path"
            );
            // And the reply reads on from it as from the path read alone.
            let next = x.narrow(1, 9, 1).unwrap();
            assert_eq!(
                max_abs_diff(
                    &run(&mut held, &next, 6).unwrap(),
                    &run(&mut alone, &next, 6).unwrap()
                ),
                0.0
            );
        }
    }

    /// A tree: its start, the pool rows before it past the prompt, each row's
    /// parent and pool row.
    struct Epoch {
        base: usize,
        prefix: Vec<usize>,
        parents: Vec<Option<usize>>,
        inputs: Vec<usize>,
    }

    impl Epoch {
        /// Row `row`'s line as pool rows, from the prompt's end.
        fn line(&self, row: usize) -> Vec<usize> {
            let mut rows = vec![row];
            while let Some(p) = self.parents[*rows.last().unwrap()] {
                rows.push(p);
            }
            let mut line = self.prefix.clone();
            line.extend(rows.iter().rev().map(|&r| self.inputs[r]));
            line
        }
    }

    /// What [`flowing_script`] shows, each with the line — pool rows after the
    /// prompt — a chain read alone reaches it by.
    enum FlowSeen {
        /// A row the walk can still reach: its output.
        Row {
            what: String,
            line: Vec<usize>,
            out: Tensor,
        },
        /// An anchor, or a start a cut put back.
        State {
            what: String,
            line: Vec<usize>,
            state: SsmSlot,
        },
    }

    /// A tree that FLOWS through `w` (`docs/plans/recurrent_flow.md`), its rows
    /// read off `pool` (`[1, 64, hidden]` on `w`'s device; rows 0-2 the
    /// prompt): rows sent over several forwards, each forward's rows with
    /// parents in earlier ones, the walk moving down the tree between them —
    /// onto a row of the forward that carries it, too (a missed draw's row) —
    /// then a rebase that cuts the layer back to the tree's start and reads the
    /// walk's path again, a forward of the rebased tree and one whose last row's
    /// line runs 35 rows below the anchor. Every row the walk can still reach,
    /// each anchor and each start a cut puts back goes to `seen`; the
    /// refusals — a walk back up the tree, a step on from the tree's state, a
    /// cut that is no start — are asserted here.
    fn flowing_script(w: &DeltaNetWeights, pool: &Tensor, seen: &mut dyn FnMut(FlowSeen)) {
        use crate::inference::draft_tree::DraftTree;
        let row_of = |i: usize| pool.narrow(1, i, 1).unwrap();
        let mut held = None;
        w.forward_deltanet(&pool.narrow(1, 0, 3).unwrap(), &mut held, 0)
            .unwrap();
        // What a state holds, each tensor once — what the store's occupancy
        // counts it (`SsmState::tensors`).
        let held_bytes = |h: &SsmState| {
            let mut ids = std::collections::HashSet::new();
            h.tensors()
                .into_iter()
                .filter(|t| ids.insert(t.id()))
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum::<u64>()
        };
        // Sends tree rows `rows` of `tree` with the walk on `walk`; shows each
        // row in `live`.
        let send = |held: &mut Option<SsmState>,
                    tree: &Epoch,
                    rows: std::ops::Range<usize>,
                    walk: usize,
                    live: &[usize],
                    seen: &mut dyn FnMut(FlowSeen)|
         -> CandleResult<()> {
            let shape = DraftTree::new(
                tree.base,
                tree.parents[..rows.end]
                    .iter()
                    .map(|p| p.map(|p| p as u16))
                    .collect(),
            )
            .unwrap();
            let index_pos = tree.base + rows.start;
            let (positions, visible) = shape.rows_for(index_pos, rows.len()).unwrap();
            let x: Vec<Tensor> = rows.clone().map(|r| row_of(tree.inputs[r])).collect();
            let out = {
                let _flow = super::super::RowPositions::for_tree_walked(
                    positions,
                    visible,
                    Some(walk as u32),
                );
                w.forward_deltanet(&Tensor::cat(&x, 1).unwrap(), held, index_pos)?
            };
            for &row in live {
                seen(FlowSeen::Row {
                    what: format!("row {row} of the tree from {}", tree.base),
                    line: tree.line(row),
                    out: out.narrow(1, row - rows.start, 1).unwrap(),
                });
            }
            Ok(())
        };
        let anchor = |held: &Option<SsmState>, line: Vec<usize>, what: &str| FlowSeen::State {
            what: format!("the anchor {what}"),
            line,
            state: held.as_ref().unwrap().flow.as_ref().unwrap().anchor.clone(),
        };

        let first = Epoch {
            base: 3,
            prefix: Vec::new(),
            parents: vec![
                None,
                Some(0),
                Some(1),
                Some(0),
                Some(3),
                Some(2),
                Some(4),
                Some(5),
                Some(1),
                Some(7),
                Some(9),
                Some(5),
                Some(6),
                Some(5),
                Some(13),
                Some(13),
            ],
            inputs: (3..19).collect(),
        };
        // A flow's first forward claims no more than it adds: the `previous`
        // pair it drops is credited (#228 residual).
        let (claim, before) = {
            let h = held.as_ref().unwrap();
            (h.bytes_a_flowing_forward_may_add(5), held_bytes(h))
        };
        send(&mut held, &first, 0..5, 0, &[0, 1, 2, 3, 4], seen).unwrap();
        let grew = held_bytes(held.as_ref().unwrap()).saturating_sub(before);
        assert!(
            grew <= claim,
            "a first flowing forward added {grew} bytes past its claim of {claim}"
        );
        // The flow's start shares `current`'s tensors: the store's occupancy
        // (which sums `tensors()`) charges a shared tensor once, not once a
        // holder.
        {
            let h = held.as_ref().unwrap();
            let charged: u64 = h
                .tensors()
                .iter()
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum();
            assert_eq!(
                charged,
                held_bytes(h),
                "a tensor two holders share was charged twice"
            );
        }
        seen(anchor(&held, first.line(0), "after the first forward"));
        // Rows from both branches below the walk's row are read again ahead
        // of the new ones (1→2 and 3→4).
        send(&mut held, &first, 5..9, 0, &[5, 6, 7, 8], seen).unwrap();
        // The walk went 0 → 1 → 2: rows 3, 4, 6 and 8 no longer follow it,
        // and row 12 (under 6) is read from the anchor, never shown.
        send(&mut held, &first, 9..13, 2, &[9, 10, 11], seen).unwrap();
        {
            // A forward that fails part-way after this layer kept its rows (a
            // later layer refused; a card's rerun after a refusal) is put back
            // where it began, and read again lands where it did.
            let mut again = held.clone();
            let h = again.as_mut().unwrap();
            assert!(h.can_rewind_to(3 + 9) && h.rewind_to(3 + 9).unwrap());
            assert_eq!(h.positions(), 3 + 9);
            send(&mut again, &first, 9..13, 2, &[9, 10, 11], seen).unwrap();
            seen(anchor(&again, first.line(2), "after a forward put back"));
            // A layer that kept nothing of a failed forward is where it began.
            let h = again.as_mut().unwrap();
            assert!(h.can_rewind_to(3 + 13) && h.rewind_to(3 + 13).unwrap());
            assert_eq!(h.positions(), 3 + 13);
        }
        {
            // What a flowing forward claims for the rows a flow keeps is what
            // its store may newly allocate — nothing while it has room, the
            // larger store past it, a row's figure a row — and the two states
            // besides.
            let h = held.as_ref().unwrap();
            let flow = h.flow.as_ref().unwrap();
            let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            let store: u64 = flow.store.kinds.iter().flatten().map(bytes).sum();
            let states = 2 * (bytes(&h.current.conv_state) + bytes(&h.current.recurrent_state));
            assert!(!flow.kept.is_empty());
            if flow_rows_mode() == FlowRowsMode::Stored {
                let row = store / flow.store.capacity as u64;
                // The room the next forward finds: every slot no kept row
                // holds, the slots the walk left counted (`note_begun`).
                let room = if flow_slots_reused() {
                    flow.store.capacity - flow.kept.len()
                } else {
                    flow.store.capacity - flow.store.len
                };
                assert!(room > 0);
                // The states less the rollback copy's pair its begin lets go
                // (checked against what a forward adds below).
                let credit = states - h.bytes_a_flowing_forward_may_add(room);
                assert_eq!(
                    h.bytes_a_flowing_forward_may_add(room + 1),
                    (store_capacity_for(flow.kept.len() + room + 1)
                        .saturating_sub(flow.store.capacity)) as u64
                        * row
                        + states
                        - credit
                );
            } else {
                // The A/B arms join the held rows anew each forward: every
                // held row and the new ones, a row's figure a row.
                let held_rows = flow.store.len;
                assert!(held_rows > 0);
                assert!(flow
                    .store
                    .kinds
                    .iter()
                    .flatten()
                    .all(|t| t.dim(1).unwrap() == held_rows));
                let row = store / held_rows as u64;
                assert!(
                    h.bytes_a_flowing_forward_may_add(3) <= (held_rows + 3) as u64 * row + states
                );
            }
        }
        {
            // A forward claims no more than it adds, and credits what its
            // begin lets go (#228 residual): the last forward's rollback copy
            // holds the pair its walk replaced, dropped as this one begins.
            // Claimed whole, the pair was counted twice — on 11-27 every flow
            // of the 27B over a far card was refused at its first forward, 259
            // MB in use plus 224 claimed against 441.
            let mut next = held.clone();
            let h = next.as_ref().unwrap();
            let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            let states = 2 * (bytes(&h.current.conv_state) + bytes(&h.current.recurrent_state));
            let claim = h.bytes_a_flowing_forward_may_add(3);
            let before = held_bytes(h);
            assert!(
                claim < states,
                "the rollback copy's pair is credited: claimed {claim} with the states at {states}"
            );
            send(&mut next, &first, 13..16, 13, &[], &mut |_| {}).unwrap();
            let grew = held_bytes(next.as_ref().unwrap()).saturating_sub(before);
            assert!(
                grew <= claim,
                "a forward added {grew} bytes past its claim of {claim}"
            );
        }
        seen(anchor(&held, first.line(2), "after the walk reached row 2"));
        {
            let mut back = held.clone();
            let refused = send(&mut back, &first, 13..16, 1, &[], &mut |_| {});
            assert!(refused.is_err(), "a walk back up the tree is refused");
        }
        // Row 2's draw kept 5, whose draw no row carries: the walk is on the
        // missed draw's row, 13, sent in the forward that carries it.
        send(&mut held, &first, 13..16, 13, &[13, 14, 15], seen).unwrap();
        seen(anchor(
            &held,
            first.line(13),
            "on a row of the forward itself",
        ));

        // The rebase (`FlowTree::rebase`): the walk's path 0 1 2 5 becomes the
        // reply, its row 13 the first of a tree from 7 with 14 and 15.
        let path: Vec<usize> = [0, 1, 2, 5].iter().map(|&r| first.inputs[r]).collect();
        let rebase = Epoch {
            base: 3,
            prefix: Vec::new(),
            parents: vec![None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(4)],
            inputs: path
                .iter()
                .copied()
                .chain([13, 14, 15].iter().map(|&r| first.inputs[r]))
                .collect(),
        };
        {
            let h = held.as_mut().unwrap();
            assert!(
                !h.can_rewind_to(first.base + 5),
                "only a start is a cut a flow takes"
            );
            assert!(h.rewind_to(3).unwrap());
            seen(FlowSeen::State {
                what: "the first tree's start".into(),
                line: Vec::new(),
                state: h.current().clone(),
            });
        }
        send(&mut held, &rebase, 0..7, 4, &[0, 1, 2, 3, 4, 5, 6], seen).unwrap();
        seen(anchor(&held, rebase.line(4), "after the rebase's forward"));

        // The rebased tree, then a line deeper than the card's tree kernel
        // holds (32): rows 6-40 a chain under row 5, row 41 a second child of
        // row 38, its walk 35 rows below the anchor.
        let mut parents = vec![None, Some(0), Some(0), Some(1), Some(2), Some(3)];
        parents.extend((6..41).map(|r| Some(r - 1)));
        parents.push(Some(38));
        let mut inputs = vec![
            first.inputs[13],
            first.inputs[14],
            first.inputs[15],
            19,
            20,
            21,
        ];
        inputs.extend(22..58);
        let second = Epoch {
            base: 7,
            prefix: path.clone(),
            parents,
            inputs,
        };
        // The walk went 13 → 14: row 4 (under 15) is gone; row 5 reads on
        // from row 3 of the same forward.
        // A forward of the rebased tree at a position the layer does not hold
        // is refused, and leaves the flow on its old start.
        let mut wrong = held.clone();
        assert!(send(&mut wrong, &second, 4..6, 1, &[], seen).is_err());
        let flow = wrong.as_ref().unwrap().flow.as_ref().unwrap();
        assert_eq!((flow.base(), flow.positions()), (3, 10));
        send(&mut held, &second, 3..6, 1, &[3, 5], seen).unwrap();
        seen(anchor(
            &held,
            second.line(1),
            "after the rebased tree's forward",
        ));
        let deep: Vec<usize> = (6..42).collect();
        send(&mut held, &second, 6..42, 1, &deep, seen).unwrap();
        // Rebased a second time below, last: the cuts that follow read the
        // starts this copy shares.
        let rebased_once = held.clone();

        let next = row_of(60);
        let at = held.as_ref().unwrap().positions();
        assert!(
            w.forward_deltanet(&next, &mut held.clone(), at).is_err(),
            "nothing reads on from a flowing tree's state"
        );
        // A failed flow cuts to the last start every segment answered: the
        // rebased tree's, or the one before it.
        let mut earlier = held.clone();
        assert!(earlier.as_mut().unwrap().rewind_to(3).unwrap());
        seen(FlowSeen::State {
            what: "the first tree's start, after the rebase".into(),
            line: Vec::new(),
            state: earlier.as_ref().unwrap().current().clone(),
        });
        let h = held.as_mut().unwrap();
        assert!(!h.can_rewind_to(9));
        assert!(h.rewind_to(7).unwrap());
        assert_eq!(h.positions(), 7);
        seen(FlowSeen::State {
            what: "the rebased tree's start".into(),
            line: path.clone(),
            state: h.current().clone(),
        });
        let mut line = path;
        line.push(60);
        seen(FlowSeen::Row {
            what: "the reply read on from the rebased tree's start".into(),
            line,
            out: w.forward_deltanet(&next, &mut held, 7).unwrap(),
        });

        // A second rebase (the walk on the rebased tree's row 1, a chain from
        // its first row) lets go of the start the flow began from, which
        // `current` shared: it then points at the oldest start kept, or it
        // holds a whole state no cut reaches — one a recurrent layer, 114 MB
        // over the 27B's far card (2026-11-27).
        let mut held = rebased_once;
        let mut parents: Vec<Option<usize>> = second.parents[1..]
            .iter()
            .map(|p| p.and_then(|p| p.checked_sub(1)))
            .collect();
        parents.push(Some(0));
        let mut inputs = second.inputs[1..].to_vec();
        inputs.push(61);
        let third = Epoch {
            base: second.base + 1,
            prefix: Vec::new(),
            parents,
            inputs,
        };
        let rows = third.parents.len() - 1;
        send(&mut held, &third, rows..rows + 1, 0, &[], &mut |_| {}).unwrap();
        let h = held.as_ref().unwrap();
        let flow = h.flow.as_ref().unwrap();
        assert_eq!(flow.base(), third.base, "the second rebase was not taken");
        assert!(
            flow.tensors()
                .iter()
                .any(|t| t.id() == h.current().recurrent_state.id()),
            "past a second rebase `current` holds a state the flow does not"
        );
    }

    /// [`flowing_script`] on the processor (`docs/plans/recurrent_flow.md`
    /// stage 1): every row the walk can still reach writes what its line read
    /// alone as a chain writes, 0.0 apart — the projections exact, as
    /// `a_tree_checks_rows_are_their_lines_read_alone` — and each anchor and
    /// start is its line's state.
    #[test]
    fn a_flowing_trees_rows_are_their_lines_read_alone() {
        let device = Device::Cpu;
        let w = weights_of(&device, true);
        let pool = exact(Tensor::randn(0f32, 1.0, (1, 64, 8), &device).unwrap(), 8.0);
        // A line's pool rows read alone after the prompt: the last row's
        // output (none for an empty line) and the state after it.
        let alone = |line: &[usize]| {
            let mut s = None;
            w.forward_deltanet(&pool.narrow(1, 0, 3).unwrap(), &mut s, 0)
                .unwrap();
            if line.is_empty() {
                return (None, s.unwrap());
            }
            let rows: Vec<Tensor> = line
                .iter()
                .map(|&i| pool.narrow(1, i, 1).unwrap())
                .collect();
            let out = w
                .forward_deltanet(&Tensor::cat(&rows, 1).unwrap(), &mut s, 3)
                .unwrap();
            (Some(out.narrow(1, line.len() - 1, 1).unwrap()), s.unwrap())
        };
        let (mut rows, mut states) = (0, 0);
        flowing_script(&w, &pool, &mut |shown| match shown {
            FlowSeen::Row { what, line, out } => {
                let want = alone(&line).0.unwrap();
                assert_eq!(max_abs_diff(&out, &want), 0.0, "{what}, its line {line:?}");
                rows += 1;
            }
            FlowSeen::State { what, line, state } => {
                let want = alone(&line).1;
                assert_eq!(
                    max_abs_diff(&state.recurrent_state, &want.current().recurrent_state),
                    0.0,
                    "{what}: the state, its line {line:?}"
                );
                assert_eq!(
                    max_abs_diff(&state.conv_state, &want.current().conv_state),
                    0.0,
                    "{what}: the window, its line {line:?}"
                );
                states += 1;
            }
        });
        assert_eq!((rows, states), (64, 9), "every row and state shown");
    }

    /// A flow's new rows go into the slots the walk left
    /// ([`FlowRows::new_slots`]): twenty forwards of four rows, the walk two
    /// rows down each time, and the store never holds more than its first
    /// eight rows — compacted, it filled its 32 every eighth forward — while
    /// every row writes what its line read alone writes, 0.0 apart. And each
    /// forward put back where it began (a later layer refused) and read again
    /// writes what it wrote: the slots its own walk left are never written
    /// in that forward, so the copy it is put back to still holds the rows
    /// the walk replays.
    #[test]
    fn a_flows_new_rows_go_where_the_walk_left_old_ones() {
        use crate::inference::draft_tree::DraftTree;
        if !flow_slots_reused() {
            eprintln!(
                "SKIPPED: the store is compacted (SWARMLLM_FLOW_SLOTS / SWARMLLM_FLOW_STORE)"
            );
            return;
        }
        const FORWARDS: usize = 20;
        let device = Device::Cpu;
        let w = weights_of(&device, true);
        let pool = exact(Tensor::randn(0f32, 1.0, (1, 64, 8), &device).unwrap(), 8.0);
        // Forward k sends rows 4k..4k+4: 4k (under the last forward's 4k−3)
        // and 4k+1 the walk's line, 4k+2 and 4k+3 a branch under 4k it leaves.
        let mut parents: Vec<Option<usize>> = Vec::new();
        for k in 0..FORWARDS {
            let r = 4 * k;
            parents.extend([r.checked_sub(3), Some(r), Some(r), Some(r + 2)]);
        }
        let input = |row: usize| 3 + row % 61;
        let line = |row: usize| {
            let mut rows = vec![row];
            while let Some(p) = parents[*rows.last().unwrap()] {
                rows.push(p);
            }
            rows.iter().rev().map(|&r| input(r)).collect::<Vec<_>>()
        };
        let alone = |line: &[usize]| {
            let mut s = None;
            w.forward_deltanet(&pool.narrow(1, 0, 3).unwrap(), &mut s, 0)
                .unwrap();
            let rows: Vec<Tensor> = line
                .iter()
                .map(|&i| pool.narrow(1, i, 1).unwrap())
                .collect();
            let out = w
                .forward_deltanet(&Tensor::cat(&rows, 1).unwrap(), &mut s, 3)
                .unwrap();
            out.narrow(1, line.len() - 1, 1).unwrap()
        };
        let send = |held: &mut Option<SsmState>, k: usize| {
            let rows = 4 * k..4 * k + 4;
            let walk = if k == 0 { 0 } else { 4 * k - 3 };
            let shape = DraftTree::new(
                3,
                parents[..rows.end]
                    .iter()
                    .map(|p| p.map(|p| p as u16))
                    .collect(),
            )
            .unwrap();
            let index_pos = 3 + rows.start;
            let (positions, visible) = shape.rows_for(index_pos, rows.len()).unwrap();
            let x: Vec<Tensor> = rows.map(|r| pool.narrow(1, input(r), 1).unwrap()).collect();
            let _flow =
                super::super::RowPositions::for_tree_walked(positions, visible, Some(walk as u32));
            w.forward_deltanet(&Tensor::cat(&x, 1).unwrap(), held, index_pos)
                .unwrap()
        };
        let mut held = None;
        w.forward_deltanet(&pool.narrow(1, 0, 3).unwrap(), &mut held, 0)
            .unwrap();
        let mut reused = 0;
        for k in 0..FORWARDS {
            let flow = held.as_ref().unwrap().flow.as_ref();
            let (len_before, kept_before) = flow.map_or((0, 0), |f| (f.store.len, f.kept.len()));
            let out = send(&mut held, k);
            for i in 0..4 {
                let row = 4 * k + i;
                assert_eq!(
                    max_abs_diff(&out.narrow(1, i, 1).unwrap(), &alone(&line(row))),
                    0.0,
                    "forward {k}: row {row}, its line {:?}",
                    line(row)
                );
            }
            let flow = held.as_ref().unwrap().flow.as_ref().unwrap();
            if flow.store.len < len_before + 4 {
                reused += 1;
            }
            assert!(
                flow.store.len <= 8 && flow.store.capacity == 32,
                "forward {k}: the store holds {} of {} rows, {kept_before} kept as it began",
                flow.store.len,
                flow.store.capacity
            );
            if k == 0 {
                continue;
            }
            // Put back where it began and read again: what it wrote.
            let h = held.as_mut().unwrap();
            assert!(h.can_rewind_to(3 + 4 * k) && h.rewind_to(3 + 4 * k).unwrap());
            assert_eq!(
                max_abs_diff(&send(&mut held, k), &out),
                0.0,
                "forward {k}, put back and read again"
            );
        }
        assert!(
            reused >= FORWARDS - 2,
            "rows went where the walk left others: {reused}"
        );
    }

    /// [`flowing_script`] on a card (`docs/plans/recurrent_flow.md` stage 2)
    /// writes what it writes on the processor, every row, anchor and start
    /// within float rounding — the convolution along the tree's lines and the
    /// delta rule each one launch (`ssm_conv_silu_tree_f32_k4`,
    /// `delta_rule_flow_f32_d*`), the latter walking a line 35 rows below the
    /// anchor, deeper than the tree kernel's 32.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_flowing_tree_on_the_card_is_the_processors() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let (on_cpu, on_card) = head_width_weights(&device);
        assert!(on_card.delta_rule_runs_on_the_card_kernel());
        let pool = Tensor::randn(0f32, 1.0, (1, 64, 256), &Device::Cpu).unwrap();
        let mut want = Vec::new();
        flowing_script(&on_cpu, &pool, &mut |shown| want.push(shown));
        let mut got = Vec::new();
        flowing_script(&on_card, &pool.to_device(&device).unwrap(), &mut |shown| {
            got.push(shown)
        });
        assert_eq!(got.len(), want.len());
        let close = |got: &Tensor, want: &Tensor, what: &str| {
            let got = got.to_device(&Device::Cpu).unwrap();
            let scale: f32 = want
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            let diff = max_abs_diff(&got, want);
            assert!(
                diff <= 1e-4 * scale.max(1.0),
                "{what}: {diff} off (up to {scale})"
            );
        };
        for (got, want) in got.iter().zip(&want) {
            match (got, want) {
                (FlowSeen::Row { out: g, .. }, FlowSeen::Row { what, out: w, .. }) => {
                    close(g, w, what)
                }
                (FlowSeen::State { state: g, .. }, FlowSeen::State { what, state: w, .. }) => {
                    close(&g.recurrent_state, &w.recurrent_state, what);
                    close(&g.conv_state, &w.conv_state, what);
                }
                _ => panic!("the card and the processor showed different things"),
            }
        }
    }

    /// A tree check on a card (#437) — the convolution and the delta rule
    /// each ONE launch taking the tree's parents (`ssm_conv_silu_tree_f32_k4`,
    /// `delta_rule_tree_f32_d*`) — writes what the processor's tree read
    /// writes, to float rounding, every row; the verdict's take-back into the
    /// in-place rows and the step after it land where the processor's do.
    /// Both kernels are seen to run (a refusal would fall back to the
    /// composition, which matches too), and the delta rule read as a chain
    /// lands far off on the branching rows.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_tree_check_on_the_card_is_the_processors() {
        use crate::inference::draft_tree::DraftTree;
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let (on_cpu, on_card) = head_width_weights(&device);
        assert!(on_card.delta_rule_runs_on_the_card_kernel());
        let x = Tensor::randn(0f32, 1.0, (1, 10, 256), &Device::Cpu).unwrap();
        let parents: Vec<Option<u16>> =
            vec![None, Some(0), Some(1), Some(0), Some(3), Some(1), None];
        let (positions, seen) = DraftTree::new(3, parents).unwrap().rows_for(3, 7).unwrap();
        let close = |got: &Tensor, want: &Tensor, what: &str| {
            let got = got.to_device(&Device::Cpu).unwrap();
            let scale: f32 = want
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            let diff = max_abs_diff(&got, want);
            assert!(
                diff <= 1e-4 * scale.max(1.0),
                "{what}: {diff} off (up to {scale})"
            );
        };
        let (mut cpu_state, mut card_state) = (None, None);
        let prompt = x.narrow(1, 0, 3).unwrap();
        on_cpu
            .forward_deltanet_into(&prompt, &mut cpu_state, 0, false)
            .unwrap();
        on_card
            .forward_deltanet_into(
                &prompt.to_device(&device).unwrap(),
                &mut card_state,
                0,
                true,
            )
            .unwrap();
        let tree_x = x.narrow(1, 3, 7).unwrap();
        let (want, got) = {
            let _tree = super::super::RowPositions::for_tree(positions, seen.clone());
            (
                on_cpu
                    .forward_deltanet_into(&tree_x, &mut cpu_state, 3, false)
                    .unwrap(),
                on_card
                    .forward_deltanet_into(
                        &tree_x.to_device(&device).unwrap(),
                        &mut card_state,
                        3,
                        true,
                    )
                    .unwrap(),
            )
        };
        for row in 0..7 {
            close(
                &got.narrow(1, row, 1).unwrap(),
                &want.narrow(1, row, 1).unwrap(),
                &format!("tree row {row}"),
            );
        }
        for state in [&mut cpu_state, &mut card_state] {
            assert!(state.as_mut().unwrap().rewind_to(6).unwrap());
        }
        let (cpu, card) = (cpu_state.as_ref().unwrap(), card_state.as_ref().unwrap());
        close(
            &card.current().recurrent_state,
            &cpu.current().recurrent_state,
            "taken back",
        );
        close(
            &card.current().conv_state,
            &cpu.current().conv_state,
            "window taken back",
        );
        let next = x.narrow(1, 9, 1).unwrap();
        close(
            &on_card
                .forward_deltanet_into(&next.to_device(&device).unwrap(), &mut card_state, 6, true)
                .unwrap(),
            &on_cpu
                .forward_deltanet_into(&next, &mut cpu_state, 6, false)
                .unwrap(),
            "the step after",
        );

        // The mechanism: both kernels take the tree on the card.
        let lines = TreeLines::new(seen.parents().unwrap(), &device).unwrap();
        let mixed =
            Tensor::randn(0f32, 1.0, (1, 7, on_card.conv1d.dim(0).unwrap()), &device).unwrap();
        let window =
            Tensor::randn(0f32, 1.0, (1, on_card.conv1d.dim(0).unwrap(), 3), &device).unwrap();
        let window_out = window.zeros_like().unwrap();
        assert!(on_card
            .conv_on_the_card(&mixed, &window, &window_out, Some(&lines), None)
            .unwrap()
            .is_some());
        let qk = l2_norm(
            &Tensor::randn(0f32, 1.0, (1, 7, 4, 128), &device).unwrap(),
            1e-6,
        )
        .unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 7, 4, 128), &device).unwrap();
        let g = Tensor::rand(0f32, 1.0, (1, 7, 4), &device)
            .unwrap()
            .neg()
            .unwrap();
        let beta = Tensor::rand(0f32, 1.0, (1, 7, 4), &device).unwrap();
        let state = Tensor::randn(0f32, 0.3, (1, 4, 128, 128), &device).unwrap();
        let into = state.zeros_like().unwrap();
        let gates = Gates::apart(g, beta);
        let (as_tree, _, written) =
            delta_rule_into(&qk, 2, &v, &gates, &state, Some(&into), Some(&lines), None).unwrap();
        assert!(written, "the card's tree kernel writes the state itself");
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
        let (on_host, _) = delta_rule_cpu(
            &cpu(&qk.narrow(2, 0, 2).unwrap()),
            &cpu(&qk.narrow(2, 2, 2).unwrap()),
            &cpu(&v),
            &cpu(&gates.g),
            &cpu(&gates.beta),
            &cpu(&state),
            1.0 / (128f32).sqrt(),
            Some(&lines.parents),
        )
        .unwrap();
        close(
            &as_tree,
            &on_host,
            "the tree kernel against the processor's",
        );
        // Null control: read as a chain, a branching row lands far off.
        let (as_chain, _, _) =
            delta_rule_into(&qk, 2, &v, &gates, &state, Some(&into), None, None).unwrap();
        assert!(max_abs_diff(&cpu(&as_chain), &on_host) > 1e-2);
    }

    #[cfg(feature = "candle-cuda")]
    thread_local! {
        /// This thread's [`super::delta_rule_rows_a_warp`], where a test sets one.
        pub(super) static DELTA_RULE_ROWS: std::cell::Cell<Option<usize>> =
            const { std::cell::Cell::new(None) };
    }

    /// Four state rows a warp (`delta_rule_*_c4`) are one row a warp's bits —
    /// outputs and the state written — for a chain, a tree and a flow's walks,
    /// at both widths: each row's arithmetic is the one-row update's, in its
    /// order. The launch takes the entry point the setting names (checked on
    /// [`super::delta_rule_entry`] itself), so the two runs are two kernels.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn four_state_rows_a_warp_are_one_rows_bits() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let with_rows = |rows: usize, f: &dyn Fn() -> (Tensor, Tensor)| {
            DELTA_RULE_ROWS.with(|r| r.set(Some(rows)));
            let out = f();
            DELTA_RULE_ROWS.with(|r| r.set(None));
            out
        };
        DELTA_RULE_ROWS.with(|r| r.set(Some(4)));
        assert_eq!(
            super::delta_rule_entry("delta_rule_f32_d128", 128),
            ("delta_rule_f32_d128_c4".into(), 8)
        );
        DELTA_RULE_ROWS.with(|r| r.set(Some(1)));
        assert_eq!(
            super::delta_rule_entry("delta_rule_f32_d64", 64),
            ("delta_rule_f32_d64".into(), 16)
        );
        DELTA_RULE_ROWS.with(|r| r.set(None));

        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        let parents = vec![
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(5),
            Some(2),
        ];
        let (seq, nk, nv) = (parents.len(), 4, 8);
        for d in [128, 64] {
            let qk = l2_norm(
                &Tensor::randn(0f32, 1.0, (1, seq, 2 * nk, d), &device).unwrap(),
                1e-6,
            )
            .unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, seq, nv, d), &device).unwrap();
            let g = Tensor::rand(0f32, 1.0, (1, seq, nv), &device)
                .unwrap()
                .neg()
                .unwrap();
            let beta = Tensor::rand(0f32, 1.0, (1, seq, nv), &device).unwrap();
            let gates = Gates::apart(g, beta);
            let state = Tensor::randn(0f32, 0.3, (1, nv, d, d), &device).unwrap();
            let tree = TreeLines::new(parents.clone(), &device).unwrap();
            let flow = TreeLines::for_flow(parents.clone(), &device).unwrap();
            for (what, lines) in [
                ("chain", None),
                ("tree", Some(&tree)),
                ("flow", Some(&flow)),
            ] {
                let run = || {
                    let into = state.zeros_like().unwrap();
                    let (out, new_state, written) =
                        delta_rule_into(&qk, nk, &v, &gates, &state, Some(&into), lines, None)
                            .unwrap();
                    assert!(written, "{what}: the card's kernel wrote the state");
                    (out, new_state)
                };
                let (one, one_state) = with_rows(1, &run);
                let (four, four_state) = with_rows(4, &run);
                assert!(bits(&one) == bits(&four), "{what} d{d}: outputs differ");
                assert!(
                    bits(&one_state) == bits(&four_state),
                    "{what} d{d}: states differ"
                );
                // Null control: the comparison sees a one-row change.
                let moved = one
                    .broadcast_add(&Tensor::new(1e-3f32, &device).unwrap())
                    .unwrap();
                assert!(
                    bits(&moved) != bits(&four),
                    "{what} d{d}: the comparison is blind"
                );
            }
        }
    }

    /// A flowing forward's kernels read the store where its rows sit, through
    /// their slots ([`SlotTable`]): rows scattered through a store and read so
    /// are the gathered rows' bits — the convolution's outputs and window, the
    /// delta rule's outputs and state — through the stored kernels themselves
    /// (each call below answers `Some`, which no gathering fallback reaches),
    /// and a table naming other slots (the null control) is not.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_store_read_in_place_is_its_rows_gathered_bit_for_bit() {
        let _card_to_itself = crate::inference::cuda_graph::card_to_itself();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIPPED: no card");
            return;
        };
        let (_, on_card) = head_width_weights(&device);
        let parents = vec![None, Some(0), Some(1), Some(0), Some(3), Some(1), None];
        let lines = TreeLines::for_flow(parents, &device).unwrap();
        let slots: Vec<u32> = vec![9, 2, 5, 0, 7, 3, 10];
        let held = 11;
        let table = |slots: &[u32]| SlotTable {
            table: Tensor::from_vec(slots.to_vec(), slots.len(), &device).unwrap(),
            bound: held,
        };
        // Row i at slot `slots[i]` of a store of `held` rows, the rest noise.
        let scatter = |rows: &Tensor| {
            let mut dims = rows.dims().to_vec();
            dims[1] = held;
            let store = Tensor::randn(0f32, 1.0, dims, &device).unwrap();
            for (i, &slot) in slots.iter().enumerate() {
                store
                    .slice_set(&rows.narrow(1, i, 1).unwrap(), 1, slot as usize)
                    .unwrap();
            }
            store
        };
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let in_order: Vec<u32> = (0..7).collect();

        let channels = on_card.conv1d.dim(0).unwrap();
        let mixed = Tensor::randn(0f32, 1.0, (1, 7, channels), &device).unwrap();
        let window = Tensor::randn(0f32, 1.0, (1, channels, 3), &device).unwrap();
        let conv = |x: &Tensor, at: Option<&SlotTable>| {
            let out = window.zeros_like().unwrap();
            let (qk, v) = on_card
                .conv_on_the_card(x, &window, &out, Some(&lines), at)
                .unwrap()
                .expect("the card's convolution takes it");
            (bits(&qk), bits(&v), bits(&out))
        };
        let store = scatter(&mixed);
        let gathered = conv(&mixed, None);
        assert_eq!(
            conv(&store, Some(&table(&slots))),
            gathered,
            "the convolution read in place"
        );
        assert_ne!(
            conv(&store, Some(&table(&in_order))),
            gathered,
            "null control: other slots"
        );

        let qk = l2_norm(
            &Tensor::randn(0f32, 1.0, (1, 7, 4, 128), &device).unwrap(),
            1e-6,
        )
        .unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 7, 4, 128), &device).unwrap();
        let g = Tensor::rand(0f32, 1.0, (1, 7, 4), &device)
            .unwrap()
            .neg()
            .unwrap();
        let beta = Tensor::rand(0f32, 1.0, (1, 7, 4), &device).unwrap();
        let state = Tensor::randn(0f32, 0.3, (1, 4, 128, 128), &device).unwrap();
        let rule = |qk: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, at: Option<&SlotTable>| {
            let into = state.zeros_like().unwrap();
            let gates = Gates::apart(g.clone(), beta.clone());
            let out = delta_rule_cuda(
                qk,
                2,
                v,
                &gates,
                &state,
                &into,
                1.0 / (128f32).sqrt(),
                Some(&lines),
                at,
            )
            .unwrap()
            .expect("the card's delta rule takes it");
            (bits(&out), bits(&into))
        };
        let (qks, vs, gs, betas) = (scatter(&qk), scatter(&v), scatter(&g), scatter(&beta));
        let gathered = rule(&qk, &v, &g, &beta, None);
        assert_eq!(
            rule(&qks, &vs, &gs, &betas, Some(&table(&slots))),
            gathered,
            "the delta rule read in place"
        );
        assert_ne!(
            rule(&qks, &vs, &gs, &betas, Some(&table(&in_order))),
            gathered,
            "null control: other slots"
        );
    }

    /// The processor's tree pass keeps a state only while a row still has to
    /// read it ([`RowSlots`]): played out, every row starts from its parent's
    /// state (or the step's) and the last row's is there at the end — over
    /// chains, fans, deep lines and 400 trees of random parents — and a tree
    /// of 512 rows keeps a few states, not 512. (The arithmetic is the tree
    /// tests' — `a_tree_checks_rows_are_their_lines_read_alone` and the flow's
    /// — bit for bit their lines read alone.)
    #[test]
    fn a_tree_keeps_only_the_states_still_to_be_read() {
        // Plays the plan out: which row's state each slot holds.
        let play = |parents: &[Option<usize>]| -> usize {
            let slots = RowSlots::of(parents);
            let mut holds: Vec<Option<usize>> = vec![None; slots.count];
            for (t, parent) in parents.iter().enumerate() {
                if let Some(p) = *parent {
                    assert_eq!(
                        holds[slots.of[p]],
                        Some(p),
                        "row {t} reads row {p}'s state, overwritten: {parents:?}"
                    );
                }
                holds[slots.of[t]] = Some(t);
            }
            if let Some(last) = parents.len().checked_sub(1) {
                assert_eq!(holds[slots.of[last]], Some(last));
            }
            slots.count
        };
        let chain: Vec<Option<usize>> = (0..512).map(|t: usize| t.checked_sub(1)).collect();
        assert_eq!(play(&chain), 1, "a chain reads on in place");
        let fan: Vec<Option<usize>> = (0..512).map(|t| (t > 0).then_some(0)).collect();
        assert_eq!(play(&fan), 2);
        // Breadth first, every row two children: the frontier is what is kept.
        let wide: Vec<Option<usize>> = (0..511)
            .map(|t: usize| t.checked_sub(1).map(|p| p / 2))
            .collect();
        assert!(play(&wide) <= 257);
        // The flow's shape: lines deep in the tree, siblings under them.
        let mut deep: Vec<Option<usize>> = vec![None];
        for t in 1..512usize {
            deep.push(Some(if t.is_multiple_of(7) { t - 3 } else { t - 1 }));
        }
        assert!(play(&deep) <= 4);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..400 {
            let rows = 1 + (seed % 64) as usize;
            let parents: Vec<Option<usize>> = (0..rows)
                .map(|t| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (t > 0 && !seed.is_multiple_of(5)).then(|| (seed >> 33) as usize % t)
                })
                .collect();
            play(&parents);
        }
    }

    /// The token-by-token composition (`SWARMLLM_DELTA_RULE=steps`) reads a
    /// tree as the one-pass kernel does: each row from its parent's state.
    #[test]
    fn the_composed_delta_rule_reads_a_tree_as_the_one_pass_kernel_does() {
        let device = Device::Cpu;
        let (nk, nv, kd, vd, seq) = (2usize, 4usize, 8usize, 8usize, 6usize);
        let n = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, &device).unwrap();
        let unit = |t: Tensor| l2_norm(&t, 1e-6).unwrap();
        let (q, k, v) = (
            unit(n(&[1, seq, nk, kd])),
            unit(n(&[1, seq, nk, kd])),
            n(&[1, seq, nv, vd]),
        );
        let g = (n(&[1, seq, nv]).abs().unwrap() * -0.3).unwrap();
        let beta = candle_nn::ops::sigmoid(&n(&[1, seq, nv])).unwrap();
        let state = n(&[1, nv, vd, kd]);
        let scale = 1.0 / (kd as f64).sqrt();
        let parents = [None, Some(0), Some(0), Some(2), None, Some(4)];
        let (out, _) =
            delta_rule_cpu(&q, &k, &v, &g, &beta, &state, scale as f32, Some(&parents)).unwrap();
        let tile = |t: &Tensor| Tensor::cat(&vec![t; nv / nk], 2).unwrap();
        let (want, _) = delta_rule(
            &(tile(&q) * scale).unwrap(),
            &tile(&k),
            &v,
            &g,
            &beta,
            state.clone(),
            Some(&parents),
        )
        .unwrap();
        assert!(
            max_abs_diff(&out, &want) < 1e-4,
            "{}",
            max_abs_diff(&out, &want)
        );
        // Null control: read as a chain, the branching rows differ.
        let (chain, _) = delta_rule_cpu(&q, &k, &v, &g, &beta, &state, scale as f32, None).unwrap();
        assert!(max_abs_diff(&out, &chain) > 1e-3);
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

            let (out, s) =
                delta_rule_cpu(&q, &k, &v, &g, &beta, &state, scale as f32, None).unwrap();
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
                        None,
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
