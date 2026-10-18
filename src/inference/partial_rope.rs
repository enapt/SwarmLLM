//! Partial RoPE on a card as ONE kernel (`kernels/fused_decode.cu`,
//! `rope_partial_f32`).
//!
//! [`crate::inference::layers::rope_over_heads`] is the one implementation of
//! partial RoPE; on a card it asks [`on_the_card`] first and composes only when
//! that declines. The composition is five launches for each of Q and K (two
//! `contiguous` copies, the rotation, and a `cat` of two `copy2d`s): eight more
//! a layer than a full-width model makes, and a decode step on a card is bound
//! by how many launches it makes (`docs/invariants/inference.md` § "A decode
//! token is bound by GPU submission COUNT, not bandwidth"). llama.cpp copies the
//! unrotated channels inside its rope kernel (`ggml-cuda/rope.cu`); so does
//! this.
//!
//! Bit-identical to the composition, so the A/B switch
//! (`SWARMLLM_PARTIAL_ROPE=compose`) moves the launch count and nothing else.

use candle_core::{DType, Device, Result, Tensor};

/// `SWARMLLM_PARTIAL_ROPE=compose` puts the composition back on the card, so
/// the fusion can be A/B'd inside ONE binary. Read once: this sits on every
/// attention layer's Q and K.
fn fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_PARTIAL_ROPE").as_deref() != Ok("compose"))
}

/// `x` (`[b, heads, seq, head_dim]`) with its leading `rope_dim` of every head
/// rotated by `cos` / `sin` (`[seq, rope_dim / 2]`) and the rest copied — in
/// ONE launch, contiguous, when `x` is on a card and every precondition of the
/// kernel holds; `None` otherwise, and the caller composes.
/// `contiguous_halves` is `ModelArch::use_rope_contiguous` (NEOX) — false is
/// interleaved pairs (NORM).
pub(crate) fn on_the_card(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_dim: usize,
    contiguous_halves: bool,
) -> Result<Option<Tensor>> {
    if !fusable(x, cos, sin, rope_dim) {
        return Ok(None);
    }
    x.apply_op3_no_bwd(
        cos,
        sin,
        &PartialRope {
            rope_dim,
            interleaved: !contiguous_halves,
        },
    )
    .map(Some)
}

/// Every precondition the kernel relies on, checked rather than assumed: whole
/// contiguous f32 heads, one cos/sin row per position, an even rotary width
/// narrower than the head, and indices that fit the kernel's 32-bit counts.
fn fusable(x: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> bool {
    if !matches!(x.device(), Device::Cuda(_)) || !fused() {
        return false;
    }
    let Ok((b, heads, seq, head_dim)) = x.dims4() else {
        return false;
    };
    let half = rope_dim / 2;
    let rows = b * heads * seq;
    rope_dim > 0
        && rope_dim.is_multiple_of(2)
        && rope_dim < head_dim
        && seq > 0
        && x.dtype() == DType::F32
        && cos.dtype() == DType::F32
        && sin.dtype() == DType::F32
        && x.is_contiguous()
        && cos.is_contiguous()
        && sin.is_contiguous()
        && cos.dims() == [seq, half]
        && sin.dims() == [seq, half]
        && cos.device().same_device(x.device())
        && sin.device().same_device(x.device())
        && rows
            .checked_mul(head_dim)
            .is_some_and(|n| n < u32::MAX as usize)
}

struct PartialRope {
    rope_dim: usize,
    interleaved: bool,
}

impl candle_core::CustomOp3 for PartialRope {
    fn name(&self) -> &'static str {
        "partial-rope"
    }

    /// Never reached: [`on_the_card`] hands only card tensors to this op, and
    /// every other device composes — which IS the processor implementation. A
    /// second one here could only ever disagree with it.
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("partial-rope is card-only; rope_over_heads composes elsewhere")
    }

    /// One `rope_partial_f32` launch. The output comes from
    /// `alloc_fully_overwritten`: the kernel assigns every element of every
    /// head (the rotated pairs and the copied tail).
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage,
        l2: &candle_core::Layout,
        s3: &candle_core::CudaStorage,
        l3: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s1.device.clone();
        let (Some((xo, xe)), Some((co, ce)), Some((so, se))) = (
            l1.contiguous_offsets(),
            l2.contiguous_offsets(),
            l3.contiguous_offsets(),
        ) else {
            candle_core::bail!("partial-rope: inputs must be contiguous");
        };
        let &[b, heads, seq, head_dim] = l1.shape().dims() else {
            candle_core::bail!("partial-rope: x must be [b, heads, seq, head_dim]");
        };
        let half = self.rope_dim / 2;
        let rows = b * heads * seq;
        let numel = xe - xo;
        if numel != rows * head_dim || ce - co != seq * half || se - so != seq * half {
            candle_core::bail!(
                "partial-rope: shape mismatch x={numel} cos={} sin={} for {:?}, rope_dim {}",
                ce - co,
                se - so,
                l1.shape().dims(),
                self.rope_dim
            );
        }
        let x = s1.as_cuda_slice::<f32>()?.slice(xo..xe);
        let cos = s2.as_cuda_slice::<f32>()?.slice(co..ce);
        let sin = s3.as_cuda_slice::<f32>()?.slice(so..se);

        let out = dev.alloc_fully_overwritten::<f32>(numel)?;
        let func = dev.get_or_load_custom_func(
            "rope_partial_f32",
            "swarmllm_fused_decode",
            crate::inference::fast_math::FUSED_DECODE_PTX,
        )?;
        let per_row = half + (head_dim - self.rope_dim);
        let cfg = LaunchConfig::for_num_elems((rows * per_row) as u32);
        let (rows_u, seq_u, head_u, rope_u) = (
            rows as u32,
            seq as u32,
            head_dim as u32,
            self.rope_dim as u32,
        );
        let interleaved = i32::from(self.interleaved);
        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&cos);
        builder.arg(&sin);
        builder.arg(&out);
        builder.arg(&rows_u);
        builder.arg(&seq_u);
        builder.arg(&head_u);
        builder.arg(&rope_u);
        builder.arg(&interleaved);
        // SAFETY: ffi. Contiguity and every length are checked above and in
        // `fusable` (the element count fits u32); the kernel reads `numel` of
        // `x`, `seq * rope_dim / 2` of `cos` and `sin`, and writes `numel` of
        // `out`, allocated at that length.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            l1.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's one launch must agree with the composition EXACTLY, not to
    /// a tolerance — the same bar as `cuda_silu_mul_is_bit_identical_to_the_composed_path`:
    /// an A/B of `SWARMLLM_PARTIAL_ROPE` must move the launch count and
    /// nothing else. Both layouts, decode and prompt shapes, and cos/sin rows
    /// narrowed at an offset as `apply_rotary_emb` hands them.
    ///
    /// ⚠ Needs a card: run `cargo test --features candle-cuda --lib
    /// partial_rope -- --nocapture` and look for no SKIPPED line.
    #[test]
    fn cuda_partial_rope_is_bit_identical_to_the_composed_path() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED partial-rope bit-identity: no CUDA device ({e})");
                return;
            }
        };
        // (batch, heads, seq, head_dim, rope_dim, at position): Phi-4-mini's Q
        // and K decoding, a prompt chunk, GLM-4's heads, Qwen 3.5's.
        let shapes = [
            (1usize, 24usize, 1usize, 128usize, 96usize, 17usize),
            (1, 8, 1, 128, 96, 300),
            (1, 24, 37, 128, 96, 0),
            (2, 2, 5, 128, 64, 9),
            (1, 16, 3, 256, 64, 41),
        ];
        let mut fused_ran = 0;
        for interleaved in [false, true] {
            for &(b, h, s, d, rope_dim, at) in &shapes {
                let x = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev).unwrap();
                let table = 64 + at + s;
                let cos_all = Tensor::randn(0f32, 1.0, (table, rope_dim / 2), &dev).unwrap();
                let sin_all = Tensor::randn(0f32, 1.0, (table, rope_dim / 2), &dev).unwrap();
                let cos = cos_all.narrow(0, at, s).unwrap();
                let sin = sin_all.narrow(0, at, s).unwrap();

                let want = crate::inference::layers::partial_rope_composed(
                    &x,
                    &cos,
                    &sin,
                    rope_dim,
                    !interleaved,
                )
                .unwrap();
                let got = on_the_card(&x, &cos, &sin, rope_dim, !interleaved)
                    .unwrap()
                    .expect("every shape here is one the kernel takes");
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
                    "{b}x{h}x{s}x{d} rope {rope_dim} at {at}, interleaved {interleaved}: \
                     {differ} of {} elements differ",
                    got.len()
                );
            }
        }
        assert_eq!(fused_ran, 2 * shapes.len());
    }
}
