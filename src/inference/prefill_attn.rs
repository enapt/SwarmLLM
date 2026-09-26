//! Attention for a block of query positions on the CPU — a prompt chunk, a
//! speculative verify — tiled over the keys so the score matrix is never
//! written out (#119).
//!
//! The matmul path (`layers::attention_scores_block`) computes the whole
//! `[rows, kv_len]` score matrix, runs the softmax over it, then multiplies it
//! by V. At a long context that matrix IS the cost: for a 128-token chunk of
//! llama-3.2-3b at 5,000 cached positions it is 61 MB per layer, written by the
//! first matmul, read and rewritten by the softmax, and read again by the
//! second — about 250 MB of memory traffic per layer, where filling 61 MB once
//! takes ~6 ms on the Ryzen 5800H this was measured on (`examples/attn_bench.rs`,
//! `SWARM_ATTN_KV`). The arithmetic around it ran at ~20% of f32 peak.
//!
//! This is FlashAttention's tiling as PyTorch's CPU kernel does it
//! (`aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`: a `gemm` for each
//! key tile's scores, an online softmax with a running max and sum per row,
//! a `gemm` accumulating P·V), with one change for grouped-query models: the
//! work is split over fixed CHUNKS OF KEYS, each task taking every query row of
//! a KV group, and the chunks are merged by flash-decoding's reduction — so each
//! group's K and V are read once, where splitting over query blocks (PyTorch's
//! choice, one head at a time) would re-read them once per block. The chunk and
//! tile sizes are constants, never derived from the thread count, so the result
//! does not depend on how many cores a machine has.
//!
//! Scope: CPU, f32, `q_len >= 2`, K/V with dense `(S, d)` planes, an additive
//! mask of shape `[q_len, kv_len]` (rank 2, or rank 4 with leading 1s) or none,
//! optional Gemma-2 soft-cap. Anything else returns `Ok(None)` and the caller
//! keeps the matmul path — an accelerator, never a requirement.
//! `SWARMLLM_PREFILL_ATTN=standard` disables it for A/B inside one binary.
//!
//! Numerics: the online softmax and the chunk merge sum in a different order
//! from the two-pass softmax, so results agree to ~1e-6 rather than bit for bit;
//! pinned against the matmul path by `prefill_kernel_matches_the_matmul_path`.

use candle_core::{CpuStorage, Device, Layout, Result, Storage, Tensor};
use rayon::prelude::*;

use crate::inference::decode_attn::{f32_planes, Planes};

/// Keys per task. Fixed, so the merge order — and the result — never depends
/// on the thread count. 1,024 keys of K and V are 1 MB at `d = 128`.
const KV_CHUNK: usize = 1024;
/// Keys per tile inside a task: the score tile is `rows × KV_TILE`, 196 KB for
/// a 384-row group (a 128-token chunk of a 3-way GQA model) — inside L2 beside
/// the 196 KB output accumulator.
const KV_TILE: usize = 128;
/// Query rows per task, so a whole prompt handed over in ONE forward (a bench,
/// a short prompt) still keeps its per-task buffers bounded. A 128-token chunk
/// of a 3-way GQA model is 384 rows and takes one block.
const ROW_BLOCK: usize = 512;

/// `SWARMLLM_PREFILL_ATTN=standard` → never use the kernel (A/B switch).
pub fn prefill_kernel_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_PREFILL_ATTN").as_deref(),
            Ok("standard") | Ok("off") | Ok("0")
        )
    })
}

/// An additive mask read where it lies: element `(i, j)` at
/// `offset + i * row_stride + j`.
struct MaskRows<'a> {
    data: &'a [f32],
    offset: usize,
    row_stride: usize,
}

impl MaskRows<'_> {
    #[inline(always)]
    fn row(&self, i: usize, from: usize, len: usize) -> &[f32] {
        let at = self.offset + i * self.row_stride + from;
        &self.data[at..at + len]
    }
}

/// The mask as `[q_len, kv_len]` rows with unit inner stride, or `None` when it
/// has any other shape (the caller then keeps the matmul path).
fn mask_rows<'a>(
    storage: &'a Storage,
    layout: &Layout,
    q_len: usize,
    kv_len: usize,
) -> Option<MaskRows<'a>> {
    let data = match storage {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_slice(),
        _ => return None,
    };
    let dims = layout.dims();
    let strides = layout.stride();
    let (rows, cols, row_stride, col_stride) = match dims.len() {
        2 => (dims[0], dims[1], strides[0], strides[1]),
        4 if dims[0] == 1 && dims[1] == 1 => (dims[2], dims[3], strides[2], strides[3]),
        _ => return None,
    };
    if rows != q_len || cols != kv_len || (col_stride != 1 && cols > 1) {
        return None;
    }
    Some(MaskRows {
        data,
        offset: layout.start_offset(),
        row_stride,
    })
}

/// Attention for `q_len >= 2` query positions. Returns `Ok(None)` when the
/// inputs are outside the kernel's scope (caller falls back to the matmul path).
///
/// `q`: `[b, n_head, q_len, d]`; `k`, `v`: `[b, n_kv_head, S, d]`; `mask`: an
/// additive `[q_len, S]` (`0.0` visible, `-inf` masked) or `None`; `scale`
/// multiplies the raw score; `softcap` is Gemma-2's tanh cap.
pub fn gqa_prefill_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    if !prefill_kernel_enabled() || !matches!(q.device(), Device::Cpu) {
        return Ok(None);
    }
    if q.dtype() != candle_core::DType::F32
        || k.dtype() != candle_core::DType::F32
        || v.dtype() != candle_core::DType::F32
    {
        return Ok(None);
    }
    let (b, n_head, q_len, d) = q.dims4()?;
    if q_len < 2 {
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
    let rows = n_rep * q_len;

    let mask_sl = mask.map(|m| m.storage_and_layout());
    let mask_view = match &mask_sl {
        None => None,
        Some((storage, layout)) => match mask_rows(storage, layout, q_len, s_len) {
            Some(m) => Some(m),
            None => return Ok(None),
        },
    };

    // Contiguous `[b, n_head, q_len, d]` is already the grouped layout: a KV
    // group's `n_rep` heads are adjacent (`repeat_kv` numbers them group-major),
    // so group g's rows are one contiguous `[n_rep * q_len, d]` block with row
    // `rep * q_len + i`. The scale is folded into q once, rather than into every
    // score.
    let mut qv = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    qv.iter_mut().for_each(|x| *x *= scale);

    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let Some(kp) = f32_planes(&k_storage, k_layout) else {
        return Ok(None);
    };
    let Some(vp) = f32_planes(&v_storage, v_layout) else {
        return Ok(None);
    };

    let n_row_blocks = rows.div_ceil(ROW_BLOCK);
    let n_chunks = s_len.div_ceil(KV_CHUNK);
    let groups = b * n_kv_head;
    // Every task's partial result: per row, [running max, running sum], then
    // the unnormalised output rows. Rows beyond a short last row block are
    // left at their initial values and ignored by the merge.
    let row_stride_in_part = 2 + d;
    let per_task = ROW_BLOCK.min(rows) * row_stride_in_part;
    let tasks = groups * n_row_blocks * n_chunks;
    let mut partial = vec![0f32; tasks * per_task];

    partial
        .par_chunks_mut(per_task)
        .enumerate()
        .for_each(|(t, part)| {
            let chunk = t % n_chunks;
            let row_block = (t / n_chunks) % n_row_blocks;
            let ga = t / (n_chunks * n_row_blocks);
            let (bi, g) = (ga / n_kv_head, ga % n_kv_head);
            let r0 = row_block * ROW_BLOCK;
            let nr = ROW_BLOCK.min(rows - r0);
            let q_block = &qv[(bi * n_head + g * n_rep) * q_len * d + r0 * d..][..nr * d];
            let k_base = kp.offset + bi * kp.strides[0] + g * kp.strides[1];
            let v_base = vp.offset + bi * vp.strides[0] + g * vp.strides[1];
            attend_chunk(
                part,
                nr,
                row_stride_in_part,
                q_block,
                r0,
                q_len,
                d,
                &kp,
                k_base,
                &vp,
                v_base,
                chunk * KV_CHUNK,
                (chunk * KV_CHUNK + KV_CHUNK).min(s_len),
                mask_view.as_ref(),
                softcap,
            );
        });

    // Merge the chunks of each (batch, group, row) — flash-decoding's
    // reduction: out = Σ_c e^(m_c − M) o_c / Σ_c e^(m_c − M) l_c. With one chunk
    // the weight is exactly 1.
    let mut out = vec![0f32; b * n_head * q_len * d];
    out.par_chunks_mut(rows * d)
        .enumerate()
        .for_each(|(ga, out_g)| {
            for (r, o) in out_g.chunks_exact_mut(d).enumerate() {
                let (row_block, rr) = (r / ROW_BLOCK, r % ROW_BLOCK);
                let at = |c: usize| {
                    ((ga * n_row_blocks + row_block) * n_chunks + c) * per_task
                        + rr * row_stride_in_part
                };
                let mut big_m = f32::NEG_INFINITY;
                for c in 0..n_chunks {
                    let m = partial[at(c)];
                    if m > big_m {
                        big_m = m;
                    }
                }
                if big_m == f32::NEG_INFINITY {
                    continue; // every key masked for this row: zeros
                }
                let mut denom = 0f32;
                for c in 0..n_chunks {
                    let a = at(c);
                    let m = partial[a];
                    if m == f32::NEG_INFINITY {
                        continue;
                    }
                    let w = (m - big_m).exp();
                    denom += w * partial[a + 1];
                    for (x, p) in o.iter_mut().zip(&partial[a + 2..a + 2 + d]) {
                        *x += w * p;
                    }
                }
                if denom > 0.0 {
                    let inv = 1.0 / denom;
                    o.iter_mut().for_each(|x| *x *= inv);
                }
            }
        });

    Ok(Some(Tensor::from_vec(
        out,
        (b, n_head, q_len, d),
        &Device::Cpu,
    )?))
}

/// One task: `nr` query rows of one KV group against keys `c0..c1`, tile by
/// tile, leaving per row `[max, sum, o[0..d]]` in `part`.
#[allow(clippy::too_many_arguments)]
fn attend_chunk(
    part: &mut [f32],
    nr: usize,
    row_stride_in_part: usize,
    q_block: &[f32],
    r0: usize,
    q_len: usize,
    d: usize,
    kp: &Planes<'_>,
    k_base: usize,
    vp: &Planes<'_>,
    v_base: usize,
    c0: usize,
    c1: usize,
    mask: Option<&MaskRows<'_>>,
    softcap: Option<f32>,
) {
    // Accumulate into a dense [nr, d] matrix for the gemm, and keep max/sum
    // beside it; copied into `part`'s interleaved layout at the end.
    let mut o = vec![0f32; nr * d];
    let mut run_max = vec![f32::NEG_INFINITY; nr];
    let mut run_sum = vec![0f32; nr];
    let mut scores = vec![0f32; nr * KV_TILE];

    let mut t0 = c0;
    while t0 < c1 {
        let tl = KV_TILE.min(c1 - t0);
        let s = &mut scores[..nr * tl];
        // s[nr × tl] = q_block[nr × d] · K[t0 .. t0+tl]ᵀ. K's rows are the
        // columns of Kᵀ: element (kk, j) of Kᵀ is K[t0 + j][kk].
        unsafe {
            gemm::gemm(
                nr,
                tl,
                d,
                s.as_mut_ptr(),
                1,
                tl as isize,
                false,
                q_block.as_ptr(),
                1,
                d as isize,
                kp.data.as_ptr().add(k_base + t0 * d),
                d as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        }
        // Online softmax: soft-cap → mask → running max; rescale what was
        // accumulated under the old max; exp and sum under the new one.
        for r in 0..nr {
            let srow = &mut s[r * tl..(r + 1) * tl];
            let i = (r0 + r) % q_len;
            if let Some(c) = softcap {
                srow.iter_mut().for_each(|x| *x = c * (*x / c).tanh());
            }
            if let Some(m) = mask {
                for (x, mk) in srow.iter_mut().zip(m.row(i, t0, tl)) {
                    *x += mk;
                }
            }
            let mut tile_max = f32::NEG_INFINITY;
            for &x in srow.iter() {
                if x > tile_max {
                    tile_max = x;
                }
            }
            let old = run_max[r];
            let new = if tile_max > old { tile_max } else { old };
            if new == f32::NEG_INFINITY {
                // Every key so far masked for this row: it contributes nothing.
                srow.iter_mut().for_each(|x| *x = 0.0);
                continue;
            }
            for x in srow.iter_mut() {
                *x -= new;
            }
            crate::inference::fast_math::exp_inplace(srow);
            let tile_sum: f32 = srow.iter().sum();
            if new != old {
                let alpha = if old == f32::NEG_INFINITY {
                    0.0
                } else {
                    (old - new).exp()
                };
                run_sum[r] *= alpha;
                o[r * d..(r + 1) * d].iter_mut().for_each(|x| *x *= alpha);
            }
            run_sum[r] += tile_sum;
            run_max[r] = new;
        }
        // o[nr × d] += s[nr × tl] · V[t0 .. t0+tl].
        unsafe {
            gemm::gemm(
                nr,
                d,
                tl,
                o.as_mut_ptr(),
                1,
                d as isize,
                true,
                s.as_ptr(),
                1,
                tl as isize,
                vp.data.as_ptr().add(v_base + t0 * d),
                1,
                d as isize,
                1.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        }
        t0 += tl;
    }

    for r in 0..nr {
        let row = &mut part[r * row_stride_in_part..(r + 1) * row_stride_in_part];
        row[0] = run_max[r];
        row[1] = run_sum[r];
        row[2..].copy_from_slice(&o[r * d..(r + 1) * d]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel against the exact composition it replaces — grouped matmul,
    /// scaled masked softmax, matmul — across the shapes it must get right:
    /// one tile, several tiles with a partial last one, several chunks, a mask
    /// that hides whole tiles for some rows, soft-cap, MHA, batch 2, and more
    /// rows than one row block.
    #[test]
    fn prefill_kernel_matches_the_matmul_path() {
        let dev = Device::Cpu;
        for (b, n_head, n_kv_head, q_len, s_len, softcap, causal) in [
            (1usize, 24usize, 8usize, 16usize, 16usize, None, true),
            (1, 24, 8, 128, 300, None, true),
            (1, 24, 8, 128, 2100, None, true),
            (1, 8, 2, 40, 1500, Some(30.0f32), true),
            (1, 16, 16, 20, 700, None, false),
            (2, 12, 4, 33, 1030, None, true),
            (1, 8, 1, 200, 450, None, true), // 1,600 rows: four row blocks
        ] {
            let d = 64;
            let q = Tensor::randn(0f32, 1.0, (b, n_head, q_len, d), &dev).unwrap();
            // K/V as a `narrow` of a larger buffer, the way KvCache hands them out.
            let kbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 9, d), &dev).unwrap();
            let vbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 9, d), &dev).unwrap();
            let k = kbuf.narrow(2, 0, s_len).unwrap();
            let v = vbuf.narrow(2, 0, s_len).unwrap();
            // The prompt's own positions are the last q_len of the cache.
            let mask = causal.then(|| {
                let past = s_len - q_len;
                let m: Vec<f32> = (0..q_len)
                    .flat_map(|i| {
                        (0..s_len).map(move |j| {
                            if j <= past + i {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                // As a narrow view of a wider mask, the way the mask cache hands it out.
                let wide = Tensor::from_vec(m, (q_len, s_len), &dev).unwrap();
                let pad = Tensor::zeros((q_len, 5), candle_core::DType::F32, &dev).unwrap();
                Tensor::cat(&[&wide, &pad], 1)
                    .unwrap()
                    .narrow(1, 0, s_len)
                    .unwrap()
            });
            let scale = 1.0 / (d as f32).sqrt();

            let n_rep = n_head / n_kv_head;
            let qg = q.reshape((b, n_kv_head, n_rep * q_len, d)).unwrap();
            let att = (qg.matmul(&k.t().unwrap()).unwrap() * scale as f64).unwrap();
            let att = match softcap {
                Some(c) => ((att / c as f64).unwrap().tanh().unwrap() * c as f64).unwrap(),
                None => att,
            };
            let att = match &mask {
                Some(m) => {
                    let mg = m
                        .broadcast_as((n_rep, q_len, s_len))
                        .unwrap()
                        .reshape((n_rep * q_len, s_len))
                        .unwrap();
                    att.broadcast_add(&mg).unwrap()
                }
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
            let want = att
                .matmul(&v.contiguous().unwrap())
                .unwrap()
                .reshape((b, n_head, q_len, d))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let got = gqa_prefill_attention_cpu(&q, &k, &v, mask.as_ref(), scale, softcap)
                .unwrap()
                .expect("kernel applies to these inputs")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
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
                "b={b} heads {n_head}/{n_kv_head} q={q_len} S={s_len} softcap={softcap:?} \
                 causal={causal}: worst abs {worst_abs} rel {worst_rel}"
            );
        }
    }

    #[test]
    fn the_kernel_declines_what_it_cannot_read() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 8, 1, 64), &dev).unwrap(); // q_len 1: decode's
        let k = Tensor::randn(0f32, 1.0, (1, 8, 10, 64), &dev).unwrap();
        assert!(gqa_prefill_attention_cpu(&q, &k, &k, None, 0.1, None)
            .unwrap()
            .is_none());
        let q4 = Tensor::randn(0f32, 1.0, (1, 8, 4, 64), &dev).unwrap();
        // A per-head mask: not a shape the kernel reads.
        let m = Tensor::zeros((1, 8, 4, 10), candle_core::DType::F32, &dev).unwrap();
        assert!(gqa_prefill_attention_cpu(&q4, &k, &k, Some(&m), 0.1, None)
            .unwrap()
            .is_none());
        // K whose (S, d) plane is not dense.
        let kt = k.transpose(2, 3).unwrap();
        assert!(gqa_prefill_attention_cpu(&q4, &kt, &kt, None, 0.1, None)
            .unwrap()
            .is_none());
    }
}
