//! Single-position (decode) attention on the CPU, straight over the KV cache.
//!
//! For one query position the two attention matmuls are tiny and awkward:
//! `[b, kvh, n_rep, d] × [b, kvh, d, S]` and `[b, kvh, n_rep, S] × [b, kvh, S, d]`
//! with `n_rep` of 1–4 rows. Measured on llama-3.2-3b at ~920 KV they cost
//! **1.3 ms per layer — 26% of a decode step — for ~11 MFLOP**: the generic
//! GEMM's packing, dispatch and the transposed K view dominate, not the
//! arithmetic. This kernel does the same computation as a handful of dot
//! products and axpys over the cache in the layout it is already stored in
//! (`[b, kvh, S, d]`, rows contiguous), one rayon task per (batch, kv head) so
//! a group's K and V are read once for all its query heads.
//!
//! Scope: `q_len == 1`, CPU, f32, K/V with contiguous `[S, d]` planes (what
//! `KvCache` hands out), optional additive mask row and Gemma-2 soft-cap. Anything
//! else returns `Ok(None)` and the caller keeps the matmul path — the kernel is
//! an accelerator, never a requirement.
//!
//! `SWARMLLM_DECODE_ATTN=standard` disables it for A/B inside one binary, the
//! same discipline as `SWARMLLM_FORCE_STANDARD_ATTN` and `SWARMLLM_DECODE_THREADS`.
//!
//! Numerics: the dot products are summed in a different order than the GEMM's
//! blocking, so results agree to ~1e-6 relative rather than bit-for-bit; the
//! softmax is the reference composition (scale → soft-cap → mask → max-shifted
//! exp → normalise), pinned against the matmul path by
//! `decode_kernel_matches_the_matmul_path`.

use candle_core::{CpuStorage, Device, Layout, Result, Storage, Tensor};
use rayon::prelude::*;

/// `SWARMLLM_DECODE_ATTN=standard` → never use the kernel (A/B switch).
pub fn decode_kernel_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_DECODE_ATTN").as_deref(),
            Ok("standard") | Ok("off") | Ok("0")
        )
    })
}

/// A borrowed f32 plane view: `base[offset + i*stride_i + j*stride_j + ...]`.
struct Planes<'a> {
    data: &'a [f32],
    offset: usize,
    /// strides for (b, h, s, d)
    strides: [usize; 4],
    dims: [usize; 4],
}

fn f32_planes<'a>(storage: &'a Storage, layout: &Layout) -> Option<Planes<'a>> {
    let data = match storage {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_slice(),
        _ => return None,
    };
    let dims = layout.dims();
    let strides = layout.stride();
    if dims.len() != 4 || strides.len() != 4 {
        return None;
    }
    // The last two axes (S, d) must be a dense plane: row stride == d, unit
    // inner stride. Head/batch strides may be anything (a `narrow` along S
    // leaves a gap between heads).
    if strides[3] != 1 || strides[2] != dims[3] {
        return None;
    }
    Some(Planes {
        data,
        offset: layout.start_offset(),
        strides: [strides[0], strides[1], strides[2], strides[3]],
        dims: [dims[0], dims[1], dims[2], dims[3]],
    })
}

/// Attention for a single query position. Returns `Ok(None)` when the inputs
/// are outside the kernel's scope (caller falls back to the matmul path).
///
/// `q`: `[b, n_head, 1, d]`; `k`, `v`: `[b, n_kv_head, S, d]`; `mask`: an
/// additive row broadcastable to `[S]` (`0.0` visible, `-inf` masked) or `None`;
/// `scale` multiplies the raw score; `softcap` is Gemma-2's tanh cap.
pub fn gqa_decode_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    if !decode_kernel_enabled() || !matches!(q.device(), Device::Cpu) {
        return Ok(None);
    }
    if q.dtype() != candle_core::DType::F32
        || k.dtype() != candle_core::DType::F32
        || v.dtype() != candle_core::DType::F32
    {
        return Ok(None);
    }
    let (b, n_head, q_len, d) = q.dims4()?;
    if q_len != 1 {
        return Ok(None);
    }
    let (kb, n_kv_head, s_len, kd) = k.dims4()?;
    if kb != b || kd != d || n_kv_head == 0 || n_head % n_kv_head != 0 || s_len == 0 {
        return Ok(None);
    }
    if v.dims4()? != (b, n_kv_head, s_len, d) {
        return Ok(None);
    }
    let n_rep = n_head / n_kv_head;

    // The mask, if any, as one additive row over S. Anything not reducible to
    // that (a genuinely per-row mask for q_len == 1 cannot exist) → fallback.
    let mask_row: Option<Vec<f32>> = match mask {
        None => None,
        Some(m) => {
            let flat = m.flatten_all()?;
            let n = flat.dim(0)?;
            if n == s_len {
                Some(flat.to_vec1::<f32>()?)
            } else if n % s_len == 0 {
                // e.g. [n_rep, S] broadcast copies — every row identical for one
                // query position; take the first.
                let all = flat.to_vec1::<f32>()?;
                let first = &all[..s_len];
                if all.chunks_exact(s_len).all(|r| r == first) {
                    Some(first.to_vec())
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }
    };

    let q = q.contiguous()?;
    let qv = q.flatten_all()?.to_vec1::<f32>()?; // b * n_head * d
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let Some(kp) = f32_planes(&k_storage, k_layout) else {
        return Ok(None);
    };
    let Some(vp) = f32_planes(&v_storage, v_layout) else {
        return Ok(None);
    };

    let mut out = vec![0f32; b * n_head * d];
    // One task per (batch, kv head): that group's K and V planes are read once
    // for all n_rep query heads.
    out.par_chunks_mut(n_rep * d)
        .enumerate()
        .for_each(|(g, out_g)| {
            let bi = g / n_kv_head;
            let h = g % n_kv_head;
            let k_base = kp.offset + bi * kp.strides[0] + h * kp.strides[1];
            let v_base = vp.offset + bi * vp.strides[0] + h * vp.strides[1];
            let k_plane = &kp.data[k_base..k_base + s_len * d];
            let v_plane = &vp.data[v_base..v_base + s_len * d];
            let mut scores = vec![0f32; s_len];
            for r in 0..n_rep {
                let qh = (bi * n_head + h * n_rep + r) * d;
                let qrow = &qv[qh..qh + d];
                // scores[s] = scale * (q · K[s])  → soft-cap → + mask
                let mut max = f32::NEG_INFINITY;
                for (s, sc) in scores.iter_mut().enumerate() {
                    let krow = &k_plane[s * d..(s + 1) * d];
                    let mut x = dot(qrow, krow) * scale;
                    if let Some(c) = softcap {
                        x = c * (x / c).tanh();
                    }
                    if let Some(m) = &mask_row {
                        x += m[s];
                    }
                    *sc = x;
                    if x > max {
                        max = x;
                    }
                }
                // softmax (max-shifted) and the weighted sum of V in one pass
                // over S: out = Σ exp(x_s - max) V[s]; then ÷ Σ exp.
                let o = &mut out_g[r * d..(r + 1) * d];
                o.iter_mut().for_each(|x| *x = 0.0);
                for x in scores.iter_mut() {
                    *x -= max;
                }
                crate::inference::fast_math::exp_inplace(&mut scores);
                let mut denom = 0f32;
                for (s, &p) in scores.iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    denom += p;
                    let vrow = &v_plane[s * d..(s + 1) * d];
                    axpy(p, vrow, o);
                }
                if denom > 0.0 {
                    let inv = 1.0 / denom;
                    o.iter_mut().for_each(|x| *x *= inv);
                }
            }
        });

    Ok(Some(Tensor::from_vec(out, (b, n_head, 1, d), &Device::Cpu)?))
}

/// 8-lane dot product (independent accumulators so LLVM vectorises it).
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

/// `y += a * x` over equal-length slices.
#[inline(always)]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel against the exact composition it replaces: grouped matmul →
    /// scaled softmax → matmul, on the shapes llama-3.2-3b decodes with.
    #[test]
    fn decode_kernel_matches_the_matmul_path() {
        let dev = Device::Cpu;
        for (n_head, n_kv_head, s_len, softcap, masked) in [
            (24usize, 8usize, 37usize, None, false),
            (24, 8, 920, None, true),
            (32, 32, 300, None, false),
            (8, 2, 64, Some(50.0f32), true),
        ] {
            let d = 128;
            let q = Tensor::randn(0f32, 1.0, (1, n_head, 1, d), &dev).unwrap();
            // K/V as a `narrow` of a larger buffer, the way KvCache hands them out.
            let kbuf = Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len + 17, d), &dev).unwrap();
            let vbuf = Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len + 17, d), &dev).unwrap();
            let k = kbuf.narrow(2, 0, s_len).unwrap();
            let v = vbuf.narrow(2, 0, s_len).unwrap();
            let mask = if masked {
                let mut m = vec![0f32; s_len];
                m[s_len - 1] = f32::NEG_INFINITY; // one masked position
                Some(Tensor::from_vec(m, (1, s_len), &dev).unwrap())
            } else {
                None
            };
            let scale = 1.0 / (d as f32).sqrt();

            // Reference: the grouped-matmul composition.
            let n_rep = n_head / n_kv_head;
            let qg = q.reshape((1, n_kv_head, n_rep, d)).unwrap();
            let att = qg.matmul(&k.t().unwrap()).unwrap();
            let att = (att * scale as f64).unwrap();
            let att = match softcap {
                Some(c) => ((att / c as f64).unwrap().tanh().unwrap() * c as f64).unwrap(),
                None => att,
            };
            let att = match &mask {
                Some(m) => att.broadcast_add(m).unwrap(),
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
            let want = att
                .matmul(&v.contiguous().unwrap())
                .unwrap()
                .reshape((1, n_head, 1, d))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let got = gqa_decode_attention_cpu(&q, &k, &v, mask.as_ref(), scale, softcap)
                .unwrap()
                .expect("kernel applies to these inputs")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
            // Two f32 summation orders agree to ~1e-7 absolute on outputs of
            // order 0.1-1; a wrong head or a dropped position is O(0.3). Bound
            // the absolute error and the relative error away from zero, so a
            // near-cancelling output does not fail on noise.
            let worst_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let worst_rel = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(0.05))
                .fold(0f32, f32::max);
            assert!(
                worst_abs < 1e-5 && worst_rel < 1e-4,
                "heads {n_head}/{n_kv_head} S={s_len} softcap={softcap:?} masked={masked}: worst abs {worst_abs} rel {worst_rel}"
            );
        }
    }

    #[test]
    fn the_kernel_declines_what_it_cannot_index() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 8, 2, 64), &dev).unwrap(); // q_len 2
        let k = Tensor::randn(0f32, 1.0, (1, 8, 10, 64), &dev).unwrap();
        assert!(gqa_decode_attention_cpu(&q, &k, &k, None, 0.1, None).unwrap().is_none());
        let q1 = Tensor::randn(0f32, 1.0, (1, 8, 1, 64), &dev).unwrap();
        // K whose (S, d) plane is not dense: a transposed view.
        let kt = k.transpose(2, 3).unwrap();
        assert!(gqa_decode_attention_cpu(&q1, &kt, &kt, None, 0.1, None).unwrap().is_none());
    }
}
