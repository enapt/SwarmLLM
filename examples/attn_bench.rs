//! Prices the f32 tensor ops behind attention, which the stage profiler shows
//! at ~30% of prompt processing and ~8x less efficient per MAC than the
//! quantized matmul beside it.
//!
//! ```bash
//! cargo run --release --no-default-features --features dev --example attn_bench
//! ```
//!
//! `SWARM_ATTN_KV=2048,5000` instead times a grouped prompt chunk at long context:
//! each op, the whole matmul path against `prefill_attn`'s key-tiled kernel, and
//! what filling a score-sized buffer costs (#119).
use candle_core::{DType, Device, Tensor};

fn bench<F: Fn() -> candle_core::Result<Tensor>>(label: &str, macs: f64, f: F) {
    f().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..3 {
            f().unwrap();
        }
        best = best.min(t.elapsed().as_secs_f64() / 3.0);
    }
    println!(
        "  {label:<44} {:>8.1} ms   {:>7.1} GMAC/s",
        best * 1000.0,
        macs / 1e9 / best
    );
}

/// A prompt chunk as `grouped_gqa_attention` runs it at LONG context (#119):
/// llama-3.2-3b's 8 KV groups of 3 query heads x 128 positions = 384 rows per
/// group, against K/V that are a `narrow` of a larger reserved buffer, the way
/// the KV cache hands them out. `SWARM_ATTN_KV=2048,5000` picks the cache
/// lengths (both stay under the 16 Mi score budget, so one pass, no blocking).
fn grouped_long_context(kv_lens: &str) {
    let (groups, rows, d) = (8usize, 3 * 128usize, 128usize);
    let dev = Device::Cpu;
    println!(
        "grouped prompt chunk: {groups} groups x {rows} rows, d={d}, threads={}\n",
        rayon::current_num_threads()
    );
    for kv in kv_lens
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
    {
        let q = Tensor::randn(0f32, 1., (1, groups, rows, d), &dev).unwrap();
        let kbuf = Tensor::randn(0f32, 1., (1, groups, kv + 512, d), &dev).unwrap();
        let vbuf = Tensor::randn(0f32, 1., (1, groups, kv + 512, d), &dev).unwrap();
        let k = kbuf.narrow(2, 0, kv).unwrap();
        let v = vbuf.narrow(2, 0, kv).unwrap();
        let macs = (groups * rows * kv * d) as f64;
        println!("kv = {kv}  ({:.2} GMAC per matmul)", macs / 1e9);
        let kt = k.t().unwrap();
        bench("q @ k^T   (cache view, transposed)", macs, || q.matmul(&kt));
        let kt_c = k.t().unwrap().contiguous().unwrap();
        bench("q @ k^T   (contiguous k^T)", macs, || q.matmul(&kt_c));
        let qt = q.t().unwrap().contiguous().unwrap();
        let kc = k.contiguous().unwrap();
        bench("k @ q^T   (scores transposed: M=kv, N=rows)", macs, || {
            kc.matmul(&qt)
        });
        let scores = q.matmul(&kt).unwrap();
        bench("softmax_last_dim(scores)", 0.0, || {
            candle_nn::ops::softmax_last_dim(&scores)
        });
        let p = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        bench("p @ v     (cache view)", macs, || p.matmul(&v));
        let vc = v.contiguous().unwrap();
        bench("p @ v     (contiguous v)", macs, || p.matmul(&vc));
        let a = Tensor::randn(0f32, 1., (groups * rows, d), &dev).unwrap();
        let b = Tensor::randn(0f32, 1., (d, kv), &dev).unwrap();
        bench("reference 2D gemm (same MACs as q @ k^T)", macs, || {
            a.matmul(&b)
        });
        // The whole attention for this chunk, both ways: the matmul composition
        // (scores → softmax → · V, the score matrix written out) against the
        // key-tiled kernel that never writes it (#119). Causal mask as served:
        // the chunk's own positions are the cache's last 128.
        let q_len = rows / 3;
        let past = kv - q_len;
        let causal: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv).map(move |j| {
                    if j <= past + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(causal, (q_len, kv), &dev).unwrap();
        let mask_g = mask
            .broadcast_as((3, q_len, kv))
            .unwrap()
            .reshape((rows, kv))
            .unwrap()
            .contiguous()
            .unwrap();
        let scale = 1.0 / (d as f64).sqrt();
        bench(
            "WHOLE: matmul path (scores written out)",
            2.0 * macs,
            || {
                let att = (q.matmul(&kt)? * scale)?.broadcast_add(&mask_g)?;
                candle_nn::ops::softmax_last_dim(&att)?.matmul(&v)
            },
        );
        let q_heads = q.reshape((1, groups * 3, q_len, d)).unwrap();
        bench("WHOLE: key-tiled kernel (prefill_attn)", 2.0 * macs, || {
            Ok(
                swarmllm::inference::prefill_attn::gqa_prefill_attention_cpu(
                    &q_heads,
                    &k,
                    &v,
                    Some(&mask),
                    scale as f32,
                    None,
                )?
                .expect("in scope"),
            )
        });
        // What a fresh score-sized buffer costs before any arithmetic: every
        // op above allocates its output anew, and first touch of freshly mapped
        // pages faults once per page.
        let elems = groups * rows * kv;
        let fill = |buf: &mut Vec<f32>| {
            use rayon::prelude::*;
            buf.par_chunks_mut(64 * 1024).for_each(|c| c.fill(1.0));
        };
        let t = std::time::Instant::now();
        let mut fresh = vec![0f32; elems];
        fill(&mut fresh);
        let fresh_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut warm_best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            fill(&mut fresh);
            warm_best = warm_best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut fresh_best = fresh_ms;
        for _ in 0..4 {
            drop(std::mem::take(&mut fresh));
            let t = std::time::Instant::now();
            fresh = vec![0f32; elems];
            fill(&mut fresh);
            fresh_best = fresh_best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "  fill a FRESH {:.0} MB buffer                    {fresh_best:>8.1} ms\n  fill the SAME buffer again                      {warm_best:>8.1} ms",
            (elems * 4) as f64 / 1e6
        );
        println!();
    }
}

fn main() {
    if let Ok(kv) = std::env::var("SWARM_ATTN_KV") {
        grouped_long_context(&kv);
        return;
    }
    // llama-3.2-3b prefill chunk: 24 heads, 128 queries, 896 KV, head_dim 128.
    let (h, q_len, kv, d) = (24usize, 128usize, 896usize, 128usize);
    let dev = Device::Cpu;
    let q = Tensor::randn(0f32, 1., (1, h, q_len, d), &dev).unwrap();
    let k = Tensor::randn(0f32, 1., (1, h, kv, d), &dev).unwrap();
    let v = Tensor::randn(0f32, 1., (1, h, kv, d), &dev).unwrap();
    let macs_qk = (h * q_len * kv * d) as f64;

    println!("attention f32 ops, h={h} q={q_len} kv={kv} d={d}\n");
    let kt_view = k.t().unwrap();
    bench("q @ k^T   (transposed VIEW, as shipped)", macs_qk, || {
        q.matmul(&kt_view)
    });
    let kt_contig = k.t().unwrap().contiguous().unwrap();
    bench("q @ k^T   (contiguous copy)", macs_qk, || {
        q.matmul(&kt_contig)
    });

    let scores = q.matmul(&kt_view).unwrap();
    bench("softmax(scores)", 0.0, || {
        candle_nn::ops::softmax_last_dim(&scores)
    });
    let vc = v.contiguous().unwrap();
    bench("scores @ v", macs_qk, || scores.matmul(&vc));

    // The rest of the attention body: GQA expansion, transpose, mask, softcap.
    let kv_heads = 8usize;
    let k_gqa = Tensor::randn(0f32, 1., (1, kv_heads, kv, d), &dev).unwrap();
    let rep = h / kv_heads;
    bench("repeat_kv  (expand 8 kv heads -> 24)", 0.0, || {
        candle_transformers::utils::repeat_kv(k_gqa.clone(), rep)
    });
    let k_rep = candle_transformers::utils::repeat_kv(k_gqa.clone(), rep).unwrap();
    bench("repeat_kv then .contiguous()", 0.0, || {
        candle_transformers::utils::repeat_kv(k_gqa.clone(), rep)?.contiguous()
    });
    bench("k.t()  (transpose view)", 0.0, || k_rep.t());
    let mask = Tensor::randn(0f32, 1., (q_len, kv), &dev).unwrap();
    bench("broadcast_add mask over scores", 0.0, || {
        scores.broadcast_add(&mask)
    });

    // The two ops attention_scores_block adds around the matmuls.
    bench("scores / sqrt(head_dim)", 0.0, || &scores / 11.3137f64);
    let u8mask = Tensor::zeros((q_len, kv), DType::U8, &dev).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &dev).unwrap();
    bench("masked_fill (broadcast u8 mask)", 0.0, || {
        let m = u8mask.broadcast_as(scores.shape())?;
        let on_true = neg_inf.broadcast_as(scores.shape())?;
        m.where_cond(&on_true, &scores)
    });

    // Is where_cond slow, or is the stride-0 broadcast slow? Materialize the
    // mask to a contiguous 4D tensor first and re-measure.
    let mask4 = u8mask
        .broadcast_as(scores.shape())
        .unwrap()
        .contiguous()
        .unwrap();
    bench("masked_fill (contiguous 4D mask)", 0.0, || {
        let on_true = neg_inf.broadcast_as(scores.shape())?;
        mask4.where_cond(&on_true, &scores)
    });
    let neg4 = neg_inf
        .broadcast_as(scores.shape())
        .unwrap()
        .contiguous()
        .unwrap();
    bench("masked_fill (contiguous mask AND fill)", 0.0, || {
        mask4.where_cond(&neg4, &scores)
    });
    let fmask = Tensor::zeros((q_len, kv), DType::F32, &dev).unwrap();
    bench("additive float mask (broadcast_add)", 0.0, || {
        scores.broadcast_add(&fmask)
    });

    // Does the mask have to be CONTIGUOUS to get that price? The shipped cache
    // holds one big [N, N] mask and hands out `narrow()` views of it, which are
    // strided (row pitch N, only q_len/kv columns live). If the strided view
    // costs materially more, the cache must return contiguous slices instead —
    // and if it does not, the ceiling-and-narrow scheme can stay.
    let big = Tensor::zeros((q_len * 2, kv * 2), DType::F32, &dev).unwrap();
    let fmask_strided = big.narrow(0, 0, q_len).unwrap().narrow(1, 0, kv).unwrap();
    bench("additive float mask (STRIDED narrow view)", 0.0, || {
        scores.broadcast_add(&fmask_strided)
    });

    // Is the broadcast itself the cost, or the add? Expand the mask to the full
    // score shape once and add it contiguously.
    let fmask4 = fmask
        .broadcast_as(scores.shape())
        .unwrap()
        .contiguous()
        .unwrap();
    bench(
        "additive float mask (contiguous 4D, plain add)",
        0.0,
        || scores.add(&fmask4),
    );
    bench("expand mask [q,kv] -> contiguous [1,h,q,kv]", 0.0, || {
        fmask.broadcast_as(scores.shape())?.contiguous()
    });

    // The whole masked-softmax body, as the caller experiences it: fresh
    // allocations every call, not a warm preallocated `scores`. The gap between
    // this and the sum of the individual ops above IS the allocation traffic.
    bench("BLOCK: scale + masked_fill + softmax", 0.0, || {
        let att = (q.matmul(&kt_view)? / 11.3137f64)?;
        let m = u8mask.broadcast_as(att.shape())?;
        let att = m.where_cond(&neg_inf.broadcast_as(att.shape())?, &att)?;
        candle_nn::ops::softmax_last_dim(&att)
    });
    bench("BLOCK: scale + broadcast_add + softmax", 0.0, || {
        let att = (q.matmul(&kt_view)? / 11.3137f64)?;
        let att = att.broadcast_add(&fmask)?;
        candle_nn::ops::softmax_last_dim(&att)
    });
    // Same arithmetic with the scale folded into `q` before the matmul: `q` is
    // head_dim wide where the scores are kv_len wide, so it is 7x fewer
    // elements to touch and removes one whole score-sized temporary.
    let q_scaled = (&q / 11.3137f64).unwrap();
    bench("BLOCK: prescaled q + broadcast_add + softmax", 0.0, || {
        let att = q_scaled.matmul(&kt_view)?;
        let att = att.broadcast_add(&fmask)?;
        candle_nn::ops::softmax_last_dim(&att)
    });
    bench("BLOCK floor: matmul + softmax, no scale/mask", 0.0, || {
        let att = q.matmul(&kt_view)?;
        candle_nn::ops::softmax_last_dim(&att)
    });

    // Reference: a plain 2D f32 gemm of similar total work, to see whether the
    // batched-3D path is the problem or f32 gemm is simply this fast here.
    let a = Tensor::randn(0f32, 1., (q_len * h, d), &dev).unwrap();
    let b = Tensor::randn(0f32, 1., (d, kv), &dev).unwrap();
    bench(
        "reference 2D gemm (same MACs, one big matmul)",
        (q_len * h * d * kv) as f64,
        || a.matmul(&b),
    );
    let _ = DType::F32;
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, IndexOp, Tensor};

    /// The batch-parallel gemm path must agree with the folded single-gemm path
    /// it replaces. Uses a broadcast shape that folds to `b == 1` (and so takes
    /// the untouched code path) as the reference for the same arithmetic.
    #[test]
    fn batched_matmul_matches_per_head_reference() {
        let dev = Device::Cpu;
        let (h, m, k, n) = (8usize, 7usize, 33usize, 11usize);
        let a = Tensor::randn(0f32, 1., (1, h, m, k), &dev).unwrap();
        let b = Tensor::randn(0f32, 1., (1, h, k, n), &dev).unwrap();
        let batched = a
            .matmul(&b)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Reference: each head on its own, which is b == 1 per call.
        let mut reference = Vec::with_capacity(h * m * n);
        for head in 0..h {
            let ah = a.i((0, head)).unwrap();
            let bh = b.i((0, head)).unwrap();
            reference.extend(
                ah.matmul(&bh)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
        }
        assert_eq!(batched.len(), reference.len());
        for (x, y) in batched.iter().zip(&reference) {
            assert!((x - y).abs() < 1e-4, "batched {x} vs per-head {y}");
        }
    }
}
