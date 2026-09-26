//! Single-position (decode) attention on the CPU, straight over the KV cache.
//!
//! For one query position the two attention matmuls are tiny and awkward:
//! `[b, kvh, n_rep, d] × [b, kvh, d, S]` and `[b, kvh, n_rep, S] × [b, kvh, S, d]`
//! with `n_rep` of 1–4 rows. Measured on llama-3.2-3b at ~920 KV they cost
//! **1.3 ms per layer — 26% of a decode step — for ~11 MFLOP**: the generic
//! GEMM's packing, dispatch and the transposed K view dominate, not the
//! arithmetic. This kernel does the same computation as a handful of dot
//! products and axpys over the cache in the layout it is already stored in
//! (`[b, kvh, S, d]`, rows contiguous).
//!
//! **Each K row and each V row is read ONCE for every query head of its
//! group**, and the positions are split into fixed [`CHUNK`]s merged by
//! flash-decoding's reduction. The first version said it read a group's K and V
//! once and read them once PER QUERY HEAD — 3x the cache traffic on
//! llama-3.2-3b, 7x on Qwen2.5-7B — which is why its decode slowed with context
//! ~2.5x faster than llama.cpp's (#119). Fixed 2026-09-26, A/B inside one
//! binary at 4 decode threads (the width calibration picks on the Ryzen),
//! min of 2 interleaved: llama-3.2-3b at ~2,080 cached 81.6 → 70.7 ms/token and
//! level at ~544 (62.2 / 61.5), so growth from 544 to 2,080 went +19.1 →
//! +8.2 ms against llama.cpp's +7.3; Qwen2.5-7B at ~2,080 141.6 → 134.1.
//! llama.cpp's own tiled single-token CPU kernel fixes the same thing for the
//! same reason (PrismML-Eng/llama.cpp PR #254: "every KV row is loaded … once
//! per q head sharing it").
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

/// Cached positions per task. FIXED — never derived from the pool width — so
/// the summation order, and with it the result, is the same whatever the
/// machine's thread count. 256 rows of K is 128 KB at `d = 128`, inside L2.
const CHUNK: usize = 256;

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

    // Pass 1, one task per (batch, kv head, chunk of positions): each K row and
    // each V row of the chunk is read ONCE, for every query head of the group.
    let n_chunks = s_len.div_ceil(CHUNK);
    let row = d + 2; // per query head: [chunk max, Σ exp, o[0..d]]
    let mut partial = vec![0f32; b * n_kv_head * n_chunks * n_rep * row];
    partial
        .par_chunks_mut(n_rep * row)
        .enumerate()
        .for_each(|(t, part)| {
            let g = t / n_chunks;
            let (bi, h) = (g / n_kv_head, g % n_kv_head);
            let s0 = (t % n_chunks) * CHUNK;
            let n = CHUNK.min(s_len - s0);
            let k_base = kp.offset + bi * kp.strides[0] + h * kp.strides[1] + s0 * d;
            let v_base = vp.offset + bi * vp.strides[0] + h * vp.strides[1] + s0 * d;
            let k_chunk = &kp.data[k_base..k_base + n * d];
            let v_chunk = &vp.data[v_base..v_base + n * d];
            let q0 = (bi * n_head + h * n_rep) * d;
            let q_group = &qv[q0..q0 + n_rep * d];

            // scores[r * n + s] = scale * (q_r · K[s]) → soft-cap → + mask
            let mut scores = vec![0f32; n_rep * n];
            for (s, krow) in k_chunk.chunks_exact(d).enumerate() {
                for (r, qrow) in q_group.chunks_exact(d).enumerate() {
                    let mut x = dot(qrow, krow) * scale;
                    if let Some(c) = softcap {
                        x = c * (x / c).tanh();
                    }
                    if let Some(m) = &mask_row {
                        x += m[s0 + s];
                    }
                    scores[r * n + s] = x;
                }
            }
            // Max-shifted exp per query head. A chunk every position of which is
            // masked for this head contributes nothing: its max stays -inf and
            // the merge below skips it (-inf - -inf would be NaN).
            for (r, sc) in scores.chunks_exact_mut(n).enumerate() {
                let mut max = f32::NEG_INFINITY;
                for &x in sc.iter() {
                    if x > max {
                        max = x;
                    }
                }
                part[r * row] = max;
                if max == f32::NEG_INFINITY {
                    sc.iter_mut().for_each(|x| *x = 0.0);
                    continue;
                }
                for x in sc.iter_mut() {
                    *x -= max;
                }
                crate::inference::fast_math::exp_inplace(sc);
            }
            // Σ exp · V, one read of each V row for every query head.
            for (s, vrow) in v_chunk.chunks_exact(d).enumerate() {
                for r in 0..n_rep {
                    let p = scores[r * n + s];
                    if p == 0.0 {
                        continue;
                    }
                    let acc = &mut part[r * row + 1..(r + 1) * row];
                    acc[0] += p;
                    axpy(p, vrow, &mut acc[1..]);
                }
            }
        });

    // Pass 2, one task per (batch, kv head): merge the chunks — flash-decoding's
    // reduction. out = Σ_c e^(m_c - M) o_c / Σ_c e^(m_c - M) l_c. With a single
    // chunk the weight is exactly 1 and this is the plain o / l.
    let mut out = vec![0f32; b * n_head * d];
    out.par_chunks_mut(n_rep * d)
        .enumerate()
        .for_each(|(g, out_g)| {
            let chunk_at = |c: usize, r: usize| ((g * n_chunks + c) * n_rep + r) * row;
            for (r, o) in out_g.chunks_exact_mut(d).enumerate() {
                let mut big_m = f32::NEG_INFINITY;
                for c in 0..n_chunks {
                    let m = partial[chunk_at(c, r)];
                    if m > big_m {
                        big_m = m;
                    }
                }
                if big_m == f32::NEG_INFINITY {
                    continue; // every position masked: leave zeros
                }
                let mut denom = 0f32;
                for c in 0..n_chunks {
                    let at = chunk_at(c, r);
                    let m = partial[at];
                    if m == f32::NEG_INFINITY {
                        continue;
                    }
                    let w = (m - big_m).exp();
                    denom += w * partial[at + 1];
                    axpy(w, &partial[at + 2..at + row], o);
                }
                if denom > 0.0 {
                    let inv = 1.0 / denom;
                    o.iter_mut().for_each(|x| *x *= inv);
                }
            }
        });

    Ok(Some(Tensor::from_vec(
        out,
        (b, n_head, 1, d),
        &Device::Cpu,
    )?))
}

/// 8-lane dot product (independent accumulators so LLVM vectorises it).
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
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
    /// How the mask of a case is built.
    #[derive(Clone, Copy, Debug)]
    enum Masked {
        No,
        /// The newest position hidden.
        Last,
        /// The first `CHUNK + 44` positions hidden: a whole chunk contributes
        /// nothing and the next one only partly — the -inf merge path.
        LeadingChunk,
    }

    #[test]
    fn decode_kernel_matches_the_matmul_path() {
        let dev = Device::Cpu;
        for (b, n_head, n_kv_head, s_len, softcap, masked) in [
            (1usize, 24usize, 8usize, 37usize, None, Masked::No),
            (1, 24, 8, 920, None, Masked::Last),
            (1, 32, 32, 300, None, Masked::No),
            (1, 8, 2, 64, Some(50.0f32), Masked::Last),
            // Several chunks, the last one partial, n_rep = 7 (Qwen2.5-7B).
            (1, 28, 4, 2055, None, Masked::No),
            (1, 24, 8, 1000, None, Masked::LeadingChunk),
            (1, 8, 2, 700, Some(30.0), Masked::LeadingChunk),
            // Exactly one chunk, and one chunk plus one position.
            (1, 24, 8, CHUNK, None, Masked::No),
            (1, 24, 8, CHUNK + 1, None, Masked::Last),
            (2, 24, 8, 600, None, Masked::Last),
        ] {
            let d = 128;
            let q = Tensor::randn(0f32, 1.0, (b, n_head, 1, d), &dev).unwrap();
            // K/V as a `narrow` of a larger buffer, the way KvCache hands them out.
            let kbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &dev).unwrap();
            let vbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &dev).unwrap();
            let k = kbuf.narrow(2, 0, s_len).unwrap();
            let v = vbuf.narrow(2, 0, s_len).unwrap();
            let mask = match masked {
                Masked::No => None,
                Masked::Last => {
                    let mut m = vec![0f32; s_len];
                    m[s_len - 1] = f32::NEG_INFINITY;
                    Some(Tensor::from_vec(m, (1, s_len), &dev).unwrap())
                }
                Masked::LeadingChunk => {
                    let mut m = vec![0f32; s_len];
                    m[..CHUNK + 44].fill(f32::NEG_INFINITY);
                    Some(Tensor::from_vec(m, (1, s_len), &dev).unwrap())
                }
            };
            let scale = 1.0 / (d as f32).sqrt();

            // Reference: the grouped-matmul composition.
            let n_rep = n_head / n_kv_head;
            let qg = q.reshape((b, n_kv_head, n_rep, d)).unwrap();
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
                .reshape((b, n_head, 1, d))
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
                "b={b} heads {n_head}/{n_kv_head} S={s_len} softcap={softcap:?} masked={masked:?}: worst abs {worst_abs} rel {worst_rel}"
            );
        }
    }

    #[test]
    fn the_kernel_declines_what_it_cannot_index() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 8, 2, 64), &dev).unwrap(); // q_len 2
        let k = Tensor::randn(0f32, 1.0, (1, 8, 10, 64), &dev).unwrap();
        assert!(gqa_decode_attention_cpu(&q, &k, &k, None, 0.1, None)
            .unwrap()
            .is_none());
        let q1 = Tensor::randn(0f32, 1.0, (1, 8, 1, 64), &dev).unwrap();
        // K whose (S, d) plane is not dense: a transposed view.
        let kt = k.transpose(2, 3).unwrap();
        assert!(gqa_decode_attention_cpu(&q1, &kt, &kt, None, 0.1, None)
            .unwrap()
            .is_none());
    }
}
