//! A projection's output to the heads-first layout attention reads — bias
//! added and RoPE applied on the way — as ONE kernel on a card
//! (`kernels/fused_decode.cu`, `heads_first_f32`).
//!
//! `layers::LayerWeights::heads_first` is the one implementation; on a card it
//! asks [`on_the_card`] first and composes only when that declines. The
//! composition is `badd` (the bias), `ucopy` (`transpose(1, 2).contiguous()`)
//! and `rope` — three launches for Q, three for K and two for V, eight a layer
//! where this makes three (a V with no bias is only the copy, left to the
//! composition — see `fusable`), and a card's decode step and a drafter's level are
//! bound by their launch count (`docs/invariants/inference.md` § "A decode
//! token is bound by GPU submission COUNT, not bandwidth"). TensorRT-LLM's
//! `addFusedQKVBiasTranspose` is the same fusion; llama.cpp fuses the bias
//! into its vector kernel (decode only).
//!
//! Bit-identical to the composition, so the A/B switch
//! (`SWARMLLM_HEADS_FIRST=compose`) moves the launch count and nothing else.

use candle_core::{DType, Device, Result, Tensor};

/// `SWARMLLM_HEADS_FIRST=compose` puts the composition back on the card, so
/// the fusion can be A/B'd inside ONE binary. Read once: this sits on every
/// attention layer's Q, K and V.
fn fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_HEADS_FIRST").as_deref() != Ok("compose"))
}

/// The rotation an [`on_the_card`] call applies: `cos` / `sin` `[seq,
/// rope_dim / 2]` over the leading `rope_dim` channels of every head, the rest
/// copied. `contiguous_halves` is `ModelArch::use_rope_contiguous` (NEOX) —
/// false is interleaved pairs (NORM).
pub(crate) struct Rotation<'a> {
    pub(crate) cos: &'a Tensor,
    pub(crate) sin: &'a Tensor,
    pub(crate) rope_dim: usize,
    pub(crate) contiguous_halves: bool,
}

/// `lin` (`[b, seq, heads * head_dim]`, a projection's output) plus `bias`
/// (`[heads * head_dim]`), rotated by `rotation`, as `[b, heads, seq,
/// head_dim]` — in ONE launch, contiguous, when `lin` is on a card and every
/// precondition of the kernel holds; `None` otherwise, and the caller composes.
pub(crate) fn on_the_card(
    lin: &Tensor,
    bias: Option<&Tensor>,
    rotation: Option<Rotation<'_>>,
    heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if !fusable(lin, bias, rotation.as_ref(), heads, head_dim) {
        return Ok(None);
    }
    let op = HeadsFirst {
        heads,
        head_dim,
        rope_dim: rotation.as_ref().map_or(0, |r| r.rope_dim),
        interleaved: rotation.as_ref().is_some_and(|r| !r.contiguous_halves),
        bias: bias.cloned(),
    };
    match rotation {
        Some(r) => lin.apply_op3_no_bwd(r.cos, r.sin, &op),
        None => lin.apply_op1_no_bwd(&op),
    }
    .map(Some)
}

/// Every precondition the kernel relies on, checked rather than assumed: a
/// contiguous f32 projection of whole heads, a contiguous bias of one row of
/// heads, one cos/sin row per position, an even rotary width no wider than the
/// head, and indices that fit the kernel's 32-bit counts.
fn fusable(
    lin: &Tensor,
    bias: Option<&Tensor>,
    rotation: Option<&Rotation<'_>>,
    heads: usize,
    head_dim: usize,
) -> bool {
    if !matches!(lin.device(), Device::Cuda(_)) || !fused() {
        return false;
    }
    let Ok((b, seq, width)) = lin.dims3() else {
        return false;
    };
    let on_this_card = |t: &Tensor| {
        t.dtype() == DType::F32 && t.is_contiguous() && t.device().same_device(lin.device())
    };
    // With neither a bias nor a rotation there is nothing to fuse: the kernel
    // would be the copy alone, which the composition makes only where one is
    // needed — none for a decoded token, whose `transpose(1, 2)` of one
    // position is already contiguous. Fusing it ADDED a launch a layer to every
    // bias-free family's decode (TinyLlama: +3 heads-first −2 rope, 2026-10-23).
    if bias.is_none() && rotation.is_none() {
        return false;
    }
    let bias_fits = bias.is_none_or(|t| on_this_card(t) && t.dims() == [width]);
    let rotation_fits = rotation.is_none_or(|r| {
        let half = r.rope_dim / 2;
        r.rope_dim > 0
            && r.rope_dim.is_multiple_of(2)
            && r.rope_dim <= head_dim
            && on_this_card(r.cos)
            && on_this_card(r.sin)
            && r.cos.dims() == [seq, half]
            && r.sin.dims() == [seq, half]
    });
    seq > 0
        && width == heads * head_dim
        && on_this_card(lin)
        && bias_fits
        && rotation_fits
        && (b * seq)
            .checked_mul(width)
            .is_some_and(|n| n < u32::MAX as usize)
}

struct HeadsFirst {
    heads: usize,
    head_dim: usize,
    /// 0 = no rotation.
    rope_dim: usize,
    interleaved: bool,
    bias: Option<Tensor>,
}

/// A rotation's cos and sin tables as the launch receives them.
type Tables<'a> = (
    &'a candle_core::CudaStorage,
    &'a candle_core::Layout,
    &'a candle_core::CudaStorage,
    &'a candle_core::Layout,
);

impl HeadsFirst {
    /// One `heads_first_f32` launch. The output comes from
    /// `alloc_fully_overwritten`: the kernel assigns every element of every
    /// head (the rotated pairs and the copied rest).
    fn launch(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        tables: Option<Tables<'_>>,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s1.device.clone();
        let Some((xo, xe)) = l1.contiguous_offsets() else {
            candle_core::bail!("heads-first: the projection must be contiguous");
        };
        let &[b, seq, width] = l1.shape().dims() else {
            candle_core::bail!("heads-first: the projection must be [b, seq, heads * head_dim]");
        };
        let (heads, head_dim) = (self.heads, self.head_dim);
        let half = self.rope_dim / 2;
        if width != heads * head_dim || xe - xo != b * seq * width {
            candle_core::bail!(
                "heads-first: {:?} is not {heads} heads of {head_dim}",
                l1.shape().dims()
            );
        }
        let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
        // The bias's view, held for the launch.
        let bias_storage = self.bias.as_ref().map(|t| t.storage_and_layout());
        let bias = match &bias_storage {
            None => None,
            Some((storage, layout)) => {
                let candle_core::Storage::Cuda(c) = &**storage else {
                    candle_core::bail!("heads-first: the bias is off the card");
                };
                let Some((o, e)) = layout.contiguous_offsets() else {
                    candle_core::bail!("heads-first: the bias must be contiguous");
                };
                if e - o != width {
                    candle_core::bail!("heads-first: a bias of {} for {width} channels", e - o);
                }
                Some(c.as_cuda_slice::<f32>()?.slice(o..e))
            }
        };
        let tables = match tables {
            None => None,
            Some((s2, l2, s3, l3)) => {
                let (Some((co, ce)), Some((so, se))) =
                    (l2.contiguous_offsets(), l3.contiguous_offsets())
                else {
                    candle_core::bail!("heads-first: cos and sin must be contiguous");
                };
                if ce - co != seq * half || se - so != seq * half || self.rope_dim > head_dim {
                    candle_core::bail!(
                        "heads-first: cos {} / sin {} for {seq} positions of a {}-wide rotation",
                        ce - co,
                        se - so,
                        self.rope_dim
                    );
                }
                Some((
                    s2.as_cuda_slice::<f32>()?.slice(co..ce),
                    s3.as_cuda_slice::<f32>()?.slice(so..se),
                ))
            }
        };
        if tables.is_none() != (self.rope_dim == 0) {
            candle_core::bail!("heads-first: a rotation without its tables");
        }

        let numel = xe - xo;
        let out = dev.alloc_fully_overwritten::<f32>(numel)?;
        let func = dev.get_or_load_custom_func(
            "heads_first_f32",
            "swarmllm_fused_decode",
            crate::inference::fast_math::FUSED_DECODE_PTX,
        )?;
        let per_row = half + (head_dim - self.rope_dim);
        let cfg = LaunchConfig::for_num_elems((b * heads * seq * per_row) as u32);
        let (b_u, seq_u, heads_u, head_u, rope_u) = (
            b as u32,
            seq as u32,
            heads as u32,
            head_dim as u32,
            self.rope_dim as u32,
        );
        let interleaved = i32::from(self.interleaved);
        let null: u64 = 0;
        let mut builder = func.builder();
        builder.arg(&x);
        match &bias {
            Some(t) => builder.arg(t),
            None => builder.arg(&null),
        };
        match &tables {
            Some((c, s)) => {
                builder.arg(c);
                builder.arg(s);
            }
            None => {
                builder.arg(&null);
                builder.arg(&null);
            }
        }
        builder.arg(&out);
        builder.arg(&b_u);
        builder.arg(&seq_u);
        builder.arg(&heads_u);
        builder.arg(&head_u);
        builder.arg(&rope_u);
        builder.arg(&interleaved);
        // SAFETY: ffi. Contiguity and every length are checked above and in
        // `fusable` (the element count fits u32); the kernel reads `numel` of
        // `x`, `heads * head_dim` of the bias, `seq * rope_dim / 2` of `cos`
        // and `sin` (none when `rope_dim` is 0), and writes `numel` of `out`,
        // allocated at that length.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            candle_core::Shape::from_dims(&[b, heads, seq, head_dim]),
        ))
    }
}

impl candle_core::CustomOp1 for HeadsFirst {
    fn name(&self) -> &'static str {
        "heads-first"
    }

    /// Never reached: [`on_the_card`] hands only card tensors to this op, and
    /// every other device composes — which IS the processor implementation.
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("heads-first is card-only; the layer composes elsewhere")
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        self.launch(s1, l1, None)
    }
}

impl candle_core::CustomOp3 for HeadsFirst {
    fn name(&self) -> &'static str {
        "heads-first"
    }

    /// Never reached, as for the one-input op above.
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("heads-first is card-only; the layer composes elsewhere")
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CudaStorage,
        l3: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        self.launch(s1, l1, Some((s2, l2, s3, l3)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's one launch must agree with the composition EXACTLY, not to
    /// a tolerance — the bar `cuda_partial_rope_is_bit_identical_to_the_composed_path`
    /// sets: an A/B of `SWARMLLM_HEADS_FIRST` must move the launch count and
    /// nothing else. The composition is the layer's own, step for step
    /// (`badd`, the reshape, `transpose(1, 2).contiguous()`,
    /// `layers::rope_over_heads`): Q and K of a 7B and the 0.5B drafter
    /// decoding, a tree's rows and a prompt chunk; a partial rotation; both
    /// layouts; no bias; no rotation (V); two batches; cos/sin rows narrowed at
    /// an offset as the layer hands them.
    ///
    /// ⚠ Needs a card: run `cargo test --features candle-cuda --lib
    /// heads_first -- --nocapture` and look for no SKIPPED line.
    #[test]
    fn cuda_heads_first_is_bit_identical_to_the_composed_path() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED heads-first bit-identity: no CUDA device ({e})");
                return;
            }
        };
        // (batch, seq, heads, head_dim, rope_dim (0 = none), bias, at position)
        let shapes = [
            (1usize, 1usize, 28usize, 128usize, 128usize, true, 17usize),
            (1, 1, 4, 128, 128, true, 300),
            (1, 24, 28, 128, 128, true, 600),
            (1, 8, 14, 64, 64, true, 37),
            (1, 37, 2, 64, 64, false, 0),
            (1, 3, 16, 256, 64, true, 41),
            (2, 5, 8, 128, 96, true, 9),
            (1, 24, 4, 128, 0, true, 0),
            (1, 1, 2, 64, 0, false, 0),
        ];
        let mut fused_ran = 0;
        for interleaved in [false, true] {
            for &(b, s, h, d, rope_dim, with_bias, at) in &shapes {
                let lin = Tensor::randn(0f32, 1.0, (b, s, h * d), &dev).unwrap();
                let bias = with_bias.then(|| Tensor::randn(0f32, 1.0, h * d, &dev).unwrap());
                let table = 64 + at + s;
                let half = rope_dim.max(2) / 2;
                let cos_all = Tensor::randn(0f32, 1.0, (table, half), &dev).unwrap();
                let sin_all = Tensor::randn(0f32, 1.0, (table, half), &dev).unwrap();
                let cos = cos_all.narrow(0, at, s).unwrap();
                let sin = sin_all.narrow(0, at, s).unwrap();

                let biased = match &bias {
                    Some(bias) => lin.broadcast_add(bias).unwrap(),
                    None => lin.clone(),
                };
                let heads = biased
                    .reshape((b, s, h, d))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let want = if rope_dim == 0 {
                    heads
                } else {
                    crate::inference::layers::rope_over_heads(
                        &heads,
                        &cos,
                        &sin,
                        rope_dim,
                        !interleaved,
                    )
                    .unwrap()
                };
                let rotation = (rope_dim > 0).then_some(Rotation {
                    cos: &cos,
                    sin: &sin,
                    rope_dim,
                    contiguous_halves: !interleaved,
                });
                let got = on_the_card(&lin, bias.as_ref(), rotation, h, d).unwrap();
                if bias.is_none() && rope_dim == 0 {
                    // A copy alone is left to the composition (`fusable`).
                    assert!(got.is_none(), "a plain copy must not be fused");
                    continue;
                }
                let got = got.expect("every other shape here is one the kernel takes");
                fused_ran += 1;
                assert!(got.is_contiguous(), "the KV cache writes K with slice_set");
                assert_eq!(got.dims(), want.dims());
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let differ = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differ,
                    0,
                    "{b}x{s}x{h}x{d} rope {rope_dim} bias {with_bias} at {at}, \
                     interleaved {interleaved}: {differ} of {} elements differ",
                    got.len()
                );
            }
        }
        assert_eq!(fused_ran, 2 * (shapes.len() - 1));
    }
}
