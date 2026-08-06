//! Prices the f32 tensor ops behind attention, which the stage profiler shows
//! at ~30% of prompt processing and ~8x less efficient per MAC than the
//! quantized matmul beside it.
//!
//! ```bash
//! cargo run --release --no-default-features --features dev --example attn_bench
//! ```
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

fn main() {
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
