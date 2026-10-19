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
//! This is FlashAttention's tiling (the score matrix never leaves the cache)
//! with ONE arithmetic per row, shared with the decode kernel
//! (`inference::attn_rows`, #179): a row reads only the keys its mask shows,
//! in fixed 256-key chunks of ABSOLUTE position, and folds the chunks in order
//! by the online merge. A task takes up to `ROWS_PER_TASK` query rows of one KV
//! group over every chunk they see, so a group's K and V are read once per
//! task. A prompt-cache hit runs the last prompt token as a decode step; with
//! this, a row's result does not depend on which forward carried it — the
//! prompt's length, the piece it was in, or a decode step alone. The sizes are
//! constants, never derived from the thread count.
//!
//! Scope: CPU, f32, `q_len >= 2`, K/V with dense `(S, d)` planes, an additive
//! mask of shape `[q_len, kv_len]` (rank 2, or rank 4 with leading 1s) or none,
//! optional Gemma-2 soft-cap. Anything else returns `Ok(None)` and the caller
//! keeps the matmul path — an accelerator, never a requirement.
//! `SWARMLLM_PREFILL_ATTN=standard` disables it for A/B inside one binary.
//!
//! Numerics: the matmul path's two-pass softmax sums in another order, so the
//! two agree to ~1e-6 rather than bit for bit
//! (`prefill_kernel_matches_the_matmul_path`); this kernel and the decode
//! kernel agree bit for bit (`a_row_s_attention_does_not_depend_on_the_forward_it_is_in`).

use candle_core::{CpuStorage, Device, Layout, Result, Storage, Tensor};
use rayon::prelude::*;

use crate::inference::attn_rows;
use crate::inference::decode_attn::f32_planes;

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

/// Rows per task: a task takes a block of one KV group's query rows through
/// every chunk of keys. 128 rows' scores for one chunk are 128 KB, beside the
/// chunk's K and V (256 KB) in L2.
const ROWS_PER_TASK: usize = 128;

/// Attention for `q_len >= 2` query positions. Returns `Ok(None)` when the
/// inputs are outside the kernel's scope (caller falls back to the matmul path).
///
/// `q`: `[b, n_head, q_len, d]`; `k`, `v`: `[b, n_kv_head, S, d]`; `mask`: an
/// additive `[q_len, S]` (`0.0` visible, `-inf` masked) or `None`; `scale`
/// multiplies the raw score; `softcap` is Gemma-2's tanh cap; `key_origin` is
/// the absolute position of `k`'s first key — 0, except for a windowed layer,
/// whose keys start at its window ([`attn_rows`]).
pub fn gqa_prefill_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f32>,
    key_origin: usize,
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
    let mask_of = |i: usize| mask_view.as_ref().map(|m| m.row(i, 0, s_len));
    // The keys each query position reads, shared by its `n_rep` heads.
    let seen: Vec<std::ops::Range<usize>> = (0..q_len)
        .map(|i| attn_rows::visible_range(mask_of(i), s_len))
        .collect();

    // Contiguous `[b, n_head, q_len, d]` is already the grouped layout: a KV
    // group's `n_rep` heads are adjacent (`repeat_kv` numbers them group-major),
    // so group g's rows are one contiguous `[n_rep * q_len, d]` block with row
    // `rep * q_len + i`. Unscaled: the scale is applied to each score, as the
    // decode kernel applies it.
    let qv = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let Some(kp) = f32_planes(&k_storage, k_layout) else {
        return Ok(None);
    };
    let Some(vp) = f32_planes(&v_storage, v_layout) else {
        return Ok(None);
    };

    let w = attn_rows::partial_len(d);
    let mut out = vec![0f32; b * n_head * q_len * d];
    out.par_chunks_mut(rows * d)
        .enumerate()
        .flat_map(|(ga, out_g)| {
            out_g
                .par_chunks_mut(ROWS_PER_TASK * d)
                .enumerate()
                .map(move |(blk, o)| (ga, blk, o))
        })
        .for_each(|(ga, blk, o)| {
            let (bi, g) = (ga / n_kv_head, ga % n_kv_head);
            let r0 = blk * ROWS_PER_TASK;
            let nr = o.len() / d;
            let q_block = &qv[((bi * n_head + g * n_rep) * q_len + r0) * d..][..nr * d];
            let k_base = kp.offset + bi * kp.strides[0] + g * kp.strides[1];
            let v_base = vp.offset + bi * vp.strides[0] + g * vp.strides[1];
            let k_plane = &kp.data[k_base..k_base + s_len * d];
            let v_plane = &vp.data[v_base..v_base + s_len * d];
            let row_seen: Vec<_> = (0..nr).map(|r| seen[(r0 + r) % q_len].clone()).collect();
            // The chunks any of these rows reads; the others would only give
            // partials the merge skips.
            let from = row_seen
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.start)
                .min();
            let to = row_seen.iter().map(|s| s.end).max().unwrap_or(0);
            let chunks = match from {
                Some(from) => attn_rows::chunks_spanned(key_origin + from, to - from),
                None => 0..0,
            };
            // Each chunk is folded into the rows' merges as soon as it is
            // computed, in chunk order — a decode step folds the same way.
            let mut scores = vec![0f32; nr * attn_rows::CHUNK];
            let mut part = vec![0f32; nr * w];
            let mut acc = vec![0f32; nr * w];
            for a in acc.chunks_exact_mut(w) {
                attn_rows::empty_partial(a);
            }
            for c in chunks {
                attn_rows::chunk_partials(
                    q_block,
                    &row_seen,
                    k_plane,
                    v_plane,
                    d,
                    key_origin,
                    c,
                    &|r| mask_of((r0 + r) % q_len),
                    scale,
                    softcap,
                    &mut scores,
                    &mut part,
                );
                for (a, p) in acc.chunks_exact_mut(w).zip(part.chunks_exact(w)) {
                    attn_rows::merge_into(a, p);
                }
            }
            for (o_row, a) in o.chunks_exact_mut(d).zip(acc.chunks_exact(w)) {
                attn_rows::finish_row(a, o_row);
            }
        });

    Ok(Some(Tensor::from_vec(
        out,
        (b, n_head, q_len, d),
        &Device::Cpu,
    )?))
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

            let got = gqa_prefill_attention_cpu(&q, &k, &v, mask.as_ref(), scale, softcap, 0)
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

    /// #179: a row's attention is the same bits whichever forward carries it —
    /// a whole prompt, a later piece of it, a forward that ends at the row (a
    /// shorter prompt), or the row alone as a decode step (a prompt-cache hit's
    /// last token). Several chunks, GQA and MHA, soft-cap, and windowed layers
    /// with their keys cut where `LayerWeights::keys_in_window` cuts them — at
    /// the first query's window, a start that moves with the forward and is
    /// rarely a chunk boundary.
    #[test]
    fn a_row_s_attention_does_not_depend_on_the_forward_it_is_in() {
        let dev = Device::Cpu;
        for (n_head, n_kv_head, s_len, softcap, window) in [
            (24usize, 8usize, 56usize, None, None),
            (24, 8, 700, None, None),
            (16, 16, 300, Some(30.0f32), None),
            (28, 4, 530, None, Some(200usize)),
            (8, 2, 900, None, Some(300)),
        ] {
            let d = 128;
            let k = Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len, d), &dev).unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len, d), &dev).unwrap();
            let q_all = Tensor::randn(0f32, 1.0, (1, n_head, s_len, d), &dev).unwrap();
            let scale = 1.0 / (d as f32).sqrt();
            // Positions `from..to` as ONE forward, its keys cut and handed over
            // as a layer hands them: from the first query's window
            // (`layers::window_start`), with that position as their origin. A
            // lone position reads no mask, as a decode step reads none.
            let forward = |from: usize, to: usize| -> Vec<Vec<u32>> {
                let q_len = to - from;
                let cut = window
                    .and_then(|w| crate::inference::layers::window_start(w, from, q_len))
                    .unwrap_or(0);
                let q = q_all.narrow(2, from, q_len).unwrap();
                let kk = k.narrow(2, cut, to - cut).unwrap();
                let vv = v.narrow(2, cut, to - cut).unwrap();
                let out = if q_len == 1 {
                    crate::inference::decode_attn::gqa_decode_attention_cpu(
                        &q, &kk, &vv, None, scale, softcap, cut,
                    )
                } else {
                    let m: Vec<f32> = (from..to)
                        .flat_map(|p| {
                            (cut..to).map(move |j| {
                                if j > p || window.is_some_and(|w| p - j >= w) {
                                    f32::NEG_INFINITY
                                } else {
                                    0.0
                                }
                            })
                        })
                        .collect();
                    let m = Tensor::from_vec(m, (q_len, to - cut), &dev).unwrap();
                    gqa_prefill_attention_cpu(&q, &kk, &vv, Some(&m), scale, softcap, cut)
                }
                .unwrap()
                .expect("a kernel takes these inputs");
                // [position][head × d] as bits, so rows compare by position.
                (0..q_len)
                    .map(|i| {
                        out.narrow(2, i, 1)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect()
                    })
                    .collect()
            };
            let case = format!("{n_head}/{n_kv_head} S={s_len} window={window:?}");
            let whole = forward(0, s_len);
            let split_at = s_len / 2 + 3;
            for (i, row) in forward(split_at, s_len).iter().enumerate() {
                let p = split_at + i;
                assert_eq!(row, &whole[p], "{case}: position {p} in a later piece");
            }
            for (p, row) in forward(0, s_len - 1).iter().enumerate() {
                assert_eq!(row, &whole[p], "{case}: position {p} in a shorter prompt");
            }
            for p in [1, s_len / 3, s_len - 1] {
                assert_eq!(forward(p, p + 1)[0], whole[p], "{case}: position {p} alone");
            }
        }
    }

    #[test]
    fn the_kernel_declines_what_it_cannot_read() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 8, 1, 64), &dev).unwrap(); // q_len 1: decode's
        let k = Tensor::randn(0f32, 1.0, (1, 8, 10, 64), &dev).unwrap();
        assert!(gqa_prefill_attention_cpu(&q, &k, &k, None, 0.1, None, 0)
            .unwrap()
            .is_none());
        let q4 = Tensor::randn(0f32, 1.0, (1, 8, 4, 64), &dev).unwrap();
        // A per-head mask: not a shape the kernel reads.
        let m = Tensor::zeros((1, 8, 4, 10), candle_core::DType::F32, &dev).unwrap();
        assert!(
            gqa_prefill_attention_cpu(&q4, &k, &k, Some(&m), 0.1, None, 0)
                .unwrap()
                .is_none()
        );
        // K whose (S, d) plane is not dense.
        let kt = k.transpose(2, 3).unwrap();
        assert!(gqa_prefill_attention_cpu(&q4, &kt, &kt, None, 0.1, None, 0)
            .unwrap()
            .is_none());
    }
}
