//! Prices candle's quantized CPU matmul as a function of the batch dimension.
//!
//! Decode calls it with `m = 1`, concurrent batching with `m = slots`, and
//! prefill with `m = chunk tokens`. If the kernel amortized the weight read
//! across rows, ms/row would fall as `m` grows. Run:
//!
//! ```bash
//! cargo run --release --no-default-features --features dev --example qmatmul_bench
//! ```
use candle_core::quantized::k_quants::{matmul, BlockQ4K, BlockQ6K, GgmlType};

/// Reproduces the ORIGINAL upstream algorithm (batch row outer, full weight
/// sweep per row) so the tiled kernel can be checked against it. Every dst
/// element is one `vec_dot` over identical operands either way, so equality
/// should be exact, not approximate.
fn reference<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) {
    let k_in_blocks = k / T::BLCK_SIZE;
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    for row_idx in 0..m {
        T::VecDotType::from_float(
            &lhs[row_idx * k..(row_idx + 1) * k],
            &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks],
        );
    }
    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        for col_idx in 0..n {
            let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
            dst[row_idx * n + col_idx] = T::vec_dot(k, rhs_col, lhs_row);
        }
    }
}

fn bench<T: GgmlType>(label: &str, k: usize, n: usize, ms: &[usize]) {
    let k_in_blocks = k / T::BLCK_SIZE;

    // Real quantized weights — zeroed blocks could take degenerate paths.
    let mut rhs = vec![T::zeros(); n * k_in_blocks];
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut noise = |scale: f32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale
    };
    let src: Vec<f32> = (0..n * k).map(|_| noise(1.0)).collect();
    T::from_float(&src, &mut rhs);

    println!("\n  {label}  (k={k}, n={n})");
    println!(
        "      {:>4}  {:>10}  {:>12}  {:>10}",
        "m", "total ms", "ms per row", "vs m=1"
    );
    let mut base = 0f64;
    for &m in ms {
        let lhs: Vec<f32> = (0..m * k).map(|_| noise(1.0)).collect();
        let mut dst = vec![0f32; m * n];
        matmul((m, k, n), &lhs, &rhs, &mut dst).unwrap(); // warm
        let mut want = vec![0f32; m * n];
        reference((m, k, n), &lhs, &rhs, &mut want);
        let bad = dst.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            bad,
            0,
            "tiled matmul differs from reference at m={m}: {bad}/{} elements",
            m * n
        );
        // Min-of-trials: this box is a WSL2 laptop and the same unchanged code
        // path measured 0.42ms and 0.97ms across runs. The minimum is the least
        // contaminated estimate; a mean just averages in the interference.
        let reps = if m <= 4 { 20 } else { 5 };
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                matmul((m, k, n), &lhs, &rhs, &mut dst).unwrap();
            }
            best = best.min(t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
        }
        let total = best;
        let per_row = total / m as f64;
        if m == 1 {
            base = per_row;
        }
        println!(
            "      {m:>4}  {total:>10.2}  {per_row:>12.3}  {:>9.2}x",
            base / per_row
        );
    }
}

fn pool_sweep() {
    // Decode (m=1) and prefill (m=128) under different rayon pool sizes, to see
    // whether the end-to-end thread curve comes from the matmul itself.
    let (k, n) = (3072usize, 8192usize);
    let k_in_blocks = k / BlockQ4K::BLCK_SIZE;
    let mut rhs = vec![BlockQ4K::zeros(); n * k_in_blocks];
    let src: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    BlockQ4K::from_float(&src, &mut rhs);
    println!("\n  rayon pool size sweep (k={k}, n={n})");
    // Widths 2..16 are the speculative-verify shapes. They are here because
    // `in_phase_pool` has to decide which of them are decode-shaped (narrow
    // pool) rather than prompt-shaped (global pool), and that boundary is a
    // property of the machine's memory system, not something to guess.
    let widths = [1usize, 2, 4, 8, 16, 32, 128];
    print!("      {:>7}", "threads");
    for m in widths {
        print!("  {:>9}", format!("m={m}"));
    }
    println!();
    for threads in [2usize, 4, 6, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut row = Vec::new();
        for m in widths {
            let lhs: Vec<f32> = (0..m * k)
                .map(|i| ((i % 89) as f32 - 44.0) / 44.0)
                .collect();
            let mut dst = vec![0f32; m * n];
            pool.install(|| matmul((m, k, n), &lhs, &rhs, &mut dst).unwrap());
            let reps = if m <= 16 { 30 } else { 5 };
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                pool.install(|| {
                    for _ in 0..reps {
                        matmul((m, k, n), &lhs, &rhs, &mut dst).unwrap();
                    }
                });
                best = best.min(t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
            }
            row.push(best);
        }
        print!("      {threads:>7}");
        for v in &row {
            print!("  {v:>9.3}");
        }
        println!();
    }
}

/// Is the plateau in ms/row the memory read, or the DEQUANTIZATION?
///
/// The tiled kernel keeps a weight column in L1 and applies it to every row, so
/// the memory read amortises across the batch. But `vec_dot` dequantizes that
/// column INSIDE the call — once per row — so that half of the work does not.
/// If that is why ms/row stops falling around m=8, then dequantizing a column
/// once into f32 and doing m plain dots must be materially cheaper.
///
/// Prices the two halves without touching the kernel, so the answer is
/// available before anyone commits to rewriting it.
fn dequant_split() {
    let k = 3072usize;
    let kb = k / BlockQ4K::BLCK_SIZE;
    let mut col = vec![BlockQ4K::zeros(); kb];
    let src: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    BlockQ4K::from_float(&src, &mut col);
    let lhs_f: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
    let mut lhs_q = vec![<BlockQ4K as GgmlType>::VecDotType::zeros(); kb];
    <BlockQ4K as GgmlType>::VecDotType::from_float(&lhs_f, &mut lhs_q);

    let mut acc = 0f32;
    let t = std::time::Instant::now();
    for _ in 0..20_000 {
        acc += BlockQ4K::vec_dot(k, &col, &lhs_q);
    }
    let quant_ns = t.elapsed().as_nanos() as f64 / 20_000.0;

    let mut col_f = vec![0f32; k];
    BlockQ4K::to_float(&col, &mut col_f);
    let mut acc2 = 0f32;
    let t = std::time::Instant::now();
    for _ in 0..20_000 {
        acc2 += col_f.iter().zip(&lhs_f).map(|(a, b)| a * b).sum::<f32>();
    }
    let f32_ns = t.elapsed().as_nanos() as f64 / 20_000.0;

    let t = std::time::Instant::now();
    for _ in 0..20_000 {
        BlockQ4K::to_float(&col, &mut col_f);
    }
    let deq_ns = t.elapsed().as_nanos() as f64 / 20_000.0;

    println!("\n  one weight column, k={k}   (checksums {acc:.0} / {acc2:.0})");
    println!("    vec_dot: dequantize + multiply-add   {quant_ns:8.0} ns");
    println!("    dequantize alone                     {deq_ns:8.0} ns");
    println!("    f32 dot alone (dequant already paid) {f32_ns:8.0} ns");
    println!("    m     per column now   dequant-once   speedup");
    for m in [2usize, 4, 8, 16, 128] {
        let now = quant_ns * m as f64;
        let then = deq_ns + f32_ns * m as f64;
        println!("    {m:<4}{now:14.0}{then:15.0}   {:.2}x", now / then);
    }
}

fn main() {
    println!("candle quantized matmul (Q4_K)");
    // llama-3.2-3b shapes: attention projection and FFN up.
    bench::<BlockQ4K>("attn proj Q4_K", 3072, 3072, &[1, 2, 3, 4, 8, 43, 128, 896]);
    bench::<BlockQ4K>("ffn up Q4_K", 3072, 8192, &[1, 2, 3, 4, 8, 43, 128, 896]);
    // Q4_K_M keeps ffn_down (and some attn_v) in Q6_K; its kernel has its own
    // multi-row path and must be checked and priced separately.
    bench::<BlockQ6K>("ffn down Q6_K", 8192, 3072, &[1, 4, 8, 43, 128, 896]);
    pool_sweep();
    dequant_split();
}
