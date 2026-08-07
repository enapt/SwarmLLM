//! One fused pass for the tail of attention: scale, optional logit soft-cap,
//! additive mask, softmax.
//!
//! # Why this exists
//!
//! `attention_scores_block` used to express that tail as four separate candle
//! ops, each of which materialises a whole score tensor
//! `[batch, heads, q_len, kv_len]` and reads the previous one back. At the
//! shapes a llama-3.2-3b prompt chunk actually uses (24 heads, 128 queries,
//! 896 KV) that tensor is 11 MB, so the tail moved ~90 MB per layer per chunk
//! to do ~3 MB of arithmetic. Priced with `examples/attn_bench.rs` at the 4
//! threads the worker runs with:
//!
//! ```text
//!   scale + masked_fill + softmax              34.6 ms   (as shipped before)
//!   scale + broadcast_add + softmax            23.7 ms   (additive f32 mask)
//!   matmul + softmax only, no scale/mask       11.4 ms   (the floor)
//! ```
//!
//! The floor says the scale and the mask together cost more than the matmul
//! and the softmax combined, and essentially all of it is memory traffic and
//! allocation rather than arithmetic. Folding them into the softmax's own
//! per-row pass removes three of the four temporaries.
//!
//! This is the same idea flash-attention exists to exploit; it just does not
//! need flash's tiled online-softmax rewrite, because the whole score row is
//! already in hand. The CPU flash kernel candle ships is a *bad* implementation
//! of it on this hardware — see the routing comment in
//! [`crate::inference::layers::run_attention`].
//!
//! # Numerics
//!
//! Every step reproduces exactly what the composed candle ops did:
//!
//! * candle's `tensor / f64` is `affine(1.0 / rhs, 0.0)`, i.e. already a
//!   multiply by an f32-rounded reciprocal — not a division. [`scale_from_head_dim`]
//!   computes that same reciprocal the same way, so the scaled score is
//!   bit-identical.
//! * soft-cap is `tanh(x * (1/cap)) * cap` with both constants f32-rounded,
//!   matching `((att / cap)?.tanh()? * cap)?`.
//! * the mask is added, not substituted. Masked entries are `-inf` and a
//!   finite score plus `-inf` is `-inf`, which is what `masked_fill` wrote.
//! * the softmax is candle's algorithm verbatim: row max, `exp(x - max)`,
//!   sum, divide.
//!
//! The one place the result can differ from the composed path is the order of
//! the exponential sum: candle's `vec_sum` reduces f32 with AVX lanes and a
//! tree combine, which is not reachable from outside `candle-core`, so this
//! sums sequentially. That moves the softmax denominator by a few ULP.
//! `fused_matches_composed_reference` pins the difference at < 1e-6 relative,
//! which is four orders of magnitude below the noise of the Q4 weights feeding
//! it. Draft and verify passes both take this path, so speculative decoding
//! still sees identical numerics on both sides.
//!
//! # Scope
//!
//! CPU f32 only. Everything else — CUDA, Metal, non-f32, non-contiguous
//! operands, a mask whose shape is not `[q_len, kv_len]` — falls through to
//! [`composed`], which is the original expression. On CUDA the standard path is
//! reached only for decode and offset-mask prefill, where the score tensor is
//! small, and a CUDA kernel could not be benchmarked here anyway.

use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// The multiplier candle's `scores / (head_dim as f64).sqrt()` applies.
///
/// Kept in one place because the fused kernel and the fallback must agree
/// bit-for-bit; see the numerics note above.
pub(crate) fn scale_from_head_dim(head_dim: usize) -> f64 {
    1.0 / (head_dim as f64).sqrt()
}

/// Scale, soft-cap, mask and softmax one score row in place.
///
/// `dst` is written twice — first the masked logits, then the probabilities —
/// so the row is touched three times in cache rather than four times through
/// main memory.
#[inline]
fn softmax_row(
    src: &[f32],
    mask: Option<&[f32]>,
    dst: &mut [f32],
    scale: f32,
    softcap: Option<f32>,
) {
    let n = dst.len();
    // Phase 1: scaled (optionally capped) logits plus the mask, tracking the
    // row max as we go so the row is not re-read to find it.
    let mut max = f32::NEG_INFINITY;
    match (softcap, mask) {
        (None, None) => {
            for i in 0..n {
                let v = src[i] * scale;
                dst[i] = v;
                if v > max {
                    max = v;
                }
            }
        }
        (None, Some(m)) => {
            for i in 0..n {
                let v = src[i] * scale + m[i];
                dst[i] = v;
                if v > max {
                    max = v;
                }
            }
        }
        (Some(cap), None) => {
            let inv = 1.0f32 / cap;
            for i in 0..n {
                let v = (src[i] * scale * inv).tanh() * cap;
                dst[i] = v;
                if v > max {
                    max = v;
                }
            }
        }
        (Some(cap), Some(m)) => {
            let inv = 1.0f32 / cap;
            for i in 0..n {
                let v = (src[i] * scale * inv).tanh() * cap + m[i];
                dst[i] = v;
                if v > max {
                    max = v;
                }
            }
        }
    }
    // Phases 2 and 3: candle's softmax, verbatim. A fully-masked row leaves
    // `max` at -inf and yields NaN, exactly as `masked_fill` + `softmax_last_dim`
    // did; a causal mask never produces one, since position `i` always attends
    // to itself.
    let mut sum = 0.0f32;
    for d in dst.iter_mut() {
        let e = (*d - max).exp();
        *d = e;
        sum += e;
    }
    for d in dst.iter_mut() {
        *d /= sum;
    }
}

/// Fused scale → soft-cap → additive mask → softmax, as a candle custom op.
///
/// `s1` is the score tensor `[.., q_len, kv_len]`; `s2` is the additive mask
/// `[q_len, kv_len]` shared by every batch and head.
#[derive(Debug, Clone)]
struct ScaledMaskedSoftmax {
    scale: f32,
    softcap: Option<f32>,
}

impl CustomOp2 for ScaledMaskedSoftmax {
    fn name(&self) -> &'static str {
        "scaled-masked-softmax"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(scores), CpuStorage::F32(mask)) = (s1, s2) else {
            candle_core::bail!("scaled-masked-softmax: f32 only");
        };
        // `fuse_applies` has already established both layouts are contiguous
        // and the shapes line up; re-derive the offsets rather than trusting
        // that, so a future caller cannot turn a missed check into an
        // out-of-bounds read.
        let Some((so, se)) = l1.contiguous_offsets() else {
            candle_core::bail!("scaled-masked-softmax: scores must be contiguous");
        };
        let Some((mo, me)) = l2.contiguous_offsets() else {
            candle_core::bail!("scaled-masked-softmax: mask must be contiguous");
        };
        let scores = &scores[so..se];
        let mask = &mask[mo..me];

        let dims = l1.shape().dims();
        let kv_len = dims[dims.len() - 1];
        let q_len = dims[dims.len() - 2];
        if kv_len == 0 || q_len == 0 {
            candle_core::bail!("scaled-masked-softmax: empty score tensor");
        }
        if mask.len() != q_len * kv_len {
            candle_core::bail!(
                "scaled-masked-softmax: mask is {} elements, expected {}x{}",
                mask.len(),
                q_len,
                kv_len
            );
        }

        let mut dst = vec![0f32; scores.len()];
        // Rows run [batch, head, q] in order, so row `r` uses mask row
        // `r % q_len` — the mask is shared across batch and head.
        scores
            .par_chunks(kv_len)
            .zip(dst.par_chunks_mut(kv_len))
            .enumerate()
            .for_each(|(r, (src, out))| {
                let mrow = &mask[(r % q_len) * kv_len..][..kv_len];
                softmax_row(src, Some(mrow), out, self.scale, self.softcap);
            });

        Ok((CpuStorage::F32(dst), Shape::from_dims(l1.shape().dims())))
    }
}

/// Would the fused kernel handle this call?
///
/// Split out so the routing rule is testable without building tensors on every
/// backend, and so the reason for a fallback is stated once.
fn fuse_applies(scores: &Tensor, mask: &Tensor) -> bool {
    if !matches!(scores.device(), Device::Cpu) {
        return false;
    }
    if scores.dtype() != DType::F32 || mask.dtype() != DType::F32 {
        return false;
    }
    if !scores.is_contiguous() || !mask.is_contiguous() {
        return false;
    }
    let sd = scores.dims();
    let md = mask.dims();
    // The kernel indexes the mask as a flat [q_len, kv_len] block shared by
    // every batch and head. A 4D mask that genuinely varies per head is not
    // that, so it goes to the fallback.
    sd.len() >= 2 && md.len() == 2 && md[0] == sd[sd.len() - 2] && md[1] == sd[sd.len() - 1]
}

/// The original expression, op by op. The fallback for every shape and device
/// the fused kernel declines, and the reference the fused kernel is tested
/// against.
fn composed(
    scores: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    softcap: Option<f32>,
) -> Result<Tensor> {
    let att = (scores * scale)?;
    let att = match softcap {
        Some(cap) => {
            let cap = cap as f64;
            ((att / cap)?.tanh()? * cap)?
        }
        None => att,
    };
    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    candle_nn::ops::softmax_last_dim(&att)
}

/// Scale, optionally soft-cap, add `mask`, and softmax over the last dimension.
///
/// `scale` multiplies the scores — pass [`scale_from_head_dim`], not the head
/// dimension. `mask` is ADDITIVE: `0.0` where a query may attend, `-inf` where
/// it may not.
pub(crate) fn scaled_masked_softmax(
    scores: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    softcap: Option<f32>,
) -> Result<Tensor> {
    match mask {
        Some(m) if fuse_applies(scores, m) => scores.apply_op2_no_bwd(
            m,
            &ScaledMaskedSoftmax {
                scale: scale as f32,
                softcap,
            },
        ),
        _ => composed(scores, mask, scale, softcap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    /// Additive causal mask with a KV offset: 0 where visible, -inf where not.
    fn causal(q_len: usize, kv_len: usize, dev: &Device) -> Tensor {
        let offset = kv_len - q_len;
        let data: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&data, (q_len, kv_len), dev).unwrap()
    }

    fn max_rel_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if d == 0.0 {
                    0.0
                } else {
                    d / x.abs().max(y.abs()).max(f32::MIN_POSITIVE)
                }
            })
            .fold(0.0f32, f32::max)
    }

    /// The whole point: the fused kernel must agree with the ops it replaces.
    ///
    /// Only the softmax denominator can differ (candle sums with AVX lanes,
    /// this sums sequentially), so the bar is a few ULP rather than equality.
    #[test]
    fn fused_matches_composed_reference() {
        let dev = Device::Cpu;
        for (h, q_len, kv_len, softcap) in [
            (3usize, 5usize, 5usize, None),
            (3, 5, 11, None),          // KV offset — prefix-cached prefill
            (1, 1, 17, None),          // decode
            (4, 9, 9, Some(50.0f32)),  // Gemma-2 logit soft-cap
            (2, 7, 40, Some(30.0f32)), // soft-cap with an offset
        ] {
            let scores = Tensor::randn(0f32, 8., (1, h, q_len, kv_len), &dev).unwrap();
            let mask = causal(q_len, kv_len, &dev);
            let scale = scale_from_head_dim(64);

            // Without this the test is a tautology the day `fuse_applies`
            // stops matching one of these shapes: both sides would be
            // `composed` and it would pass having tested nothing.
            assert!(
                fuse_applies(&scores, &mask),
                "h={h} q={q_len} kv={kv_len} must reach the fused kernel"
            );
            let fused = scaled_masked_softmax(&scores, Some(&mask), scale, softcap).unwrap();
            let reference = composed(&scores, Some(&mask), scale, softcap).unwrap();

            assert_eq!(fused.dims(), reference.dims());
            let rel = max_rel_diff(&fused, &reference);
            assert!(
                rel < 1e-6,
                "h={h} q={q_len} kv={kv_len} softcap={softcap:?}: max relative diff {rel}"
            );
        }
    }

    /// A test that passes with the fix removed is not a regression test. The
    /// mask has to actually be doing something, or the comparison above would
    /// hold for a kernel that ignored it entirely.
    #[test]
    fn masked_positions_get_exactly_zero_probability() {
        let dev = Device::Cpu;
        let (q_len, kv_len) = (6usize, 10usize);
        let scores = Tensor::randn(0f32, 8., (1, 2, q_len, kv_len), &dev).unwrap();
        let mask = causal(q_len, kv_len, &dev);
        let out = scaled_masked_softmax(&scores, Some(&mask), scale_from_head_dim(64), None)
            .unwrap()
            .i(0)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let offset = kv_len - q_len;
        for (i, row) in out.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}");
            for (j, p) in row.iter().enumerate() {
                if j > offset + i {
                    assert_eq!(*p, 0.0, "row {i} col {j} is masked but has weight {p}");
                } else {
                    assert!(*p > 0.0, "row {i} col {j} is visible but has weight {p}");
                }
            }
        }
    }

    /// The scale must be the same multiply candle's `/ f64` applies, or every
    /// logit shifts slightly and the softmax with it.
    #[test]
    fn scale_matches_candle_division() {
        let dev = Device::Cpu;
        let t = Tensor::randn(0f32, 4., (64,), &dev).unwrap();
        for head_dim in [64usize, 96, 128, 256] {
            let ours = (&t * scale_from_head_dim(head_dim))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let candle = (&t / (head_dim as f64).sqrt())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(ours, candle, "head_dim {head_dim}");
        }
    }

    /// Shapes and devices the kernel cannot handle must fall back rather than
    /// produce a wrong answer or an error the caller has to know about.
    #[test]
    fn declines_shapes_it_cannot_index() {
        let dev = Device::Cpu;
        let scores = Tensor::randn(0f32, 1., (1, 2, 4, 6), &dev).unwrap();
        // Right rank, wrong dims.
        assert!(!fuse_applies(
            &scores,
            &Tensor::zeros((3, 6), DType::F32, &dev).unwrap()
        ));
        // A per-head 4D mask is not the shared [q, kv] block the kernel indexes.
        assert!(!fuse_applies(
            &scores,
            &Tensor::zeros((1, 2, 4, 6), DType::F32, &dev).unwrap()
        ));
        // Non-contiguous: a narrowed view of a larger mask.
        let big = Tensor::zeros((8, 12), DType::F32, &dev).unwrap();
        let strided = big.narrow(0, 0, 4).unwrap().narrow(1, 0, 6).unwrap();
        assert!(!strided.is_contiguous());
        assert!(!fuse_applies(&scores, &strided));
        // And the shape it does handle.
        assert!(fuse_applies(
            &scores,
            &Tensor::zeros((4, 6), DType::F32, &dev).unwrap()
        ));
    }

    /// No mask is still a valid call — decode passes `None` — and must not
    /// reach the fused kernel or change the answer.
    #[test]
    fn unmasked_call_matches_plain_softmax() {
        let dev = Device::Cpu;
        let scores = Tensor::randn(0f32, 3., (1, 2, 1, 33), &dev).unwrap();
        let scale = scale_from_head_dim(128);
        let ours = scaled_masked_softmax(&scores, None, scale, None).unwrap();
        let reference = candle_nn::ops::softmax_last_dim(&(&scores * scale).unwrap()).unwrap();
        assert_eq!(
            ours.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            reference.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
