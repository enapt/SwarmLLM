//! Prices candle's quantized CPU matmul as a function of the batch dimension.
//!
//! Decode calls it with `m = 1`, concurrent batching with `m = slots`, and
//! prefill with `m = chunk tokens`. If the kernel amortized the weight read
//! across rows, ms/row would fall as `m` grows. Run:
//!
//! ```bash
//! cargo run --release --no-default-features --features dev --example qmatmul_bench
//! ```
use candle_core::quantized::k_quants::{matmul, BlockQ4K, GgmlType};

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

fn bench(label: &str, k: usize, n: usize, ms: &[usize]) {
    let k_in_blocks = k / BlockQ4K::BLCK_SIZE;

    // Real quantized weights — zeroed blocks could take degenerate paths.
    let mut rhs = vec![BlockQ4K::zeros(); n * k_in_blocks];
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut noise = |scale: f32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale
    };
    let src: Vec<f32> = (0..n * k).map(|_| noise(1.0)).collect();
    BlockQ4K::from_float(&src, &mut rhs);

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
    println!(
        "      {:>6}  {:>12}  {:>12}",
        "threads", "m=1 ms", "m=128 ms"
    );
    for threads in [2usize, 4, 6, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut row = Vec::new();
        for m in [1usize, 128] {
            let lhs: Vec<f32> = (0..m * k)
                .map(|i| ((i % 89) as f32 - 44.0) / 44.0)
                .collect();
            let mut dst = vec![0f32; m * n];
            pool.install(|| matmul((m, k, n), &lhs, &rhs, &mut dst).unwrap());
            let reps = if m == 1 { 30 } else { 5 };
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
        println!("      {threads:>6}  {:>12.3}  {:>12.2}", row[0], row[1]);
    }
}

fn main() {
    println!("candle quantized matmul (Q4_K)");
    // llama-3.2-3b shapes: attention projection and FFN up.
    bench("attn proj", 3072, 3072, &[1, 2, 3, 4, 8, 43, 128]);
    bench("ffn up", 3072, 8192, &[1, 2, 3, 4, 8, 43, 128]);
    pool_sweep();
}
