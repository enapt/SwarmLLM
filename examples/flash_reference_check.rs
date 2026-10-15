//! The vendored flash-attention kernel against an f32 reference, at every head
//! width a supported family uses and at the query shapes production sends.
//!
//! CI has no graphics card, so nothing else runs the kernel on numbers. Each
//! row is random Q/K/V of one shape, rounded to f16 (what the kernel reads),
//! attended by flash on the card and by plain f32 softmax attention on the
//! processor over the SAME rounded values — causal aligned bottom-right (a
//! chunk landing on a warm prefix), with the sliding window where the row has
//! one. A healthy row differs by f16 rounding of the output (~1e-3); a NaN or
//! a difference near the values' own size is the kernel computing something
//! else.
//!
//! ```bash
//! CARGO_TARGET_DIR=~/swarm-target-cuda cargo run --release --features cuda \
//!   --example flash_reference_check -j 2
//! ```

use candle_core::{DType, Device, Tensor, D};

/// (label, n_head, n_kv_head, head_dim). Gemma 4's global layers are the only
/// 512-wide heads; its windowed layers and Gemma 2/3 are 256.
const GEOMETRIES: &[(&str, usize, usize, usize)] = &[
    ("llama GQA 24/8 d128", 24, 8, 128),
    ("gemma windowed 16/8 d256", 16, 8, 256),
    ("gemma4 global 16/1 d512", 16, 1, 512),
];

/// (q_len, kv_len, window_left). `kv_len > q_len` is a chunk on a warm prefix;
/// a window is applied the way `keys_in_window` hands it over.
const SHAPES: &[(usize, usize, Option<usize>)] = &[
    (100, 100, None),
    (64, 64, None),
    (8, 1536, None),
    (12, 1536, None),
    (16, 1536, None),
    (32, 1536, None),
    (33, 1536, None),
    (48, 1532, None),
    (48, 100, None),
    (52, 1536, None),
    (60, 1532, None),
    (64, 1536, None),
    (460, 1484, None),
    (504, 1528, None),
    (48, 1071, Some(1023)),
];

fn reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    window_left: Option<usize>,
) -> candle_core::Result<Tensor> {
    // BSHD in, BSHD out, all f32 on the processor.
    let (_, q_len, _, _) = q.dims4()?;
    let kv_len = k.dim(1)?;
    let group = n_head / n_kv_head;
    let q = q.transpose(1, 2)?.contiguous()?; // b h q d
    let k = k.transpose(1, 2)?.contiguous()?; // b kvh kv d
    let v = v.transpose(1, 2)?.contiguous()?;
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let last = i + kv_len - q_len;
        for j in 0..kv_len {
            let too_late = j > last;
            let too_early = window_left.is_some_and(|w| j + w < last);
            if too_late || too_early {
                mask[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (q_len, kv_len), &Device::Cpu)?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut heads = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let qh = q.narrow(1, h, 1)?.squeeze(1)?.squeeze(0)?; // q d
        let kh = k.narrow(1, h / group, 1)?.squeeze(1)?.squeeze(0)?; // kv d
        let vh = v.narrow(1, h / group, 1)?.squeeze(1)?.squeeze(0)?;
        let scores = (qh.matmul(&kh.t()?)? * scale)?.broadcast_add(&mask)?;
        let p = candle_nn::ops::softmax(&scores, D::Minus1)?;
        heads.push(p.matmul(&vh)?); // q d
    }
    Tensor::stack(&heads, 1)?.unsqueeze(0) // 1 q h d
}

fn main() -> candle_core::Result<()> {
    let card = Device::new_cuda(0)?;
    println!(
        "{:<26} {:>5} {:>5} {:>6}  {:>6} {:>10}",
        "geometry", "q", "kv", "window", "NaN", "max |diff|"
    );
    let mut bad = 0;
    for &(label, n_head, n_kv_head, head_dim) in GEOMETRIES {
        for &(q_len, kv_len, window_left) in SHAPES {
            let rounded = |heads: usize, len: usize| -> candle_core::Result<Tensor> {
                Tensor::randn(0f32, 1.0, (1, len, heads, head_dim), &Device::Cpu)?
                    .to_dtype(DType::F16)?
                    .to_dtype(DType::F32)
            };
            let q = rounded(n_head, q_len)?;
            let k = rounded(n_kv_head, kv_len)?;
            let v = rounded(n_kv_head, kv_len)?;
            let want = reference(&q, &k, &v, n_head, n_kv_head, head_dim, window_left)?;
            let on_card = |t: &Tensor| t.to_dtype(DType::F16)?.to_device(&card);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let right = (q_len > 1).then_some(0);
            let got = candle_flash_attn::flash_attn_windowed(
                &on_card(&q)?,
                &on_card(&k)?,
                &on_card(&v)?,
                scale,
                window_left,
                right,
            )?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
            let got_v = got.flatten_all()?.to_vec1::<f32>()?;
            let nan = got_v.iter().filter(|x| !x.is_finite()).count();
            let diff = (got - &want)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .filter(|x| x.is_finite())
                .fold(0f32, f32::max);
            let verdict = if nan > 0 || diff > 0.05 {
                bad += 1;
                "  WRONG"
            } else {
                ""
            };
            println!(
                "{label:<26} {q_len:>5} {kv_len:>5} {:>6}  {nan:>6} {diff:>10.5}{verdict}",
                window_left.map_or("-".to_string(), |w| w.to_string()),
            );
        }
    }
    println!("{bad} shape(s) wrong");

    // What a prompt chunk costs per head width, min of 10 with the card
    // synchronised inside the timed region — a block shape is chosen on this
    // as well as on being right.
    println!("\n{:<26} {:>5} {:>5}  {:>8}", "geometry", "q", "kv", "ms");
    for &(label, n_head, n_kv_head, head_dim) in GEOMETRIES {
        for (q_len, kv_len) in [(512usize, 512usize), (512, 4096)] {
            let half = |heads: usize, len: usize| {
                Tensor::randn(0f32, 1.0, (1, len, heads, head_dim), &card)?.to_dtype(DType::F16)
            };
            let (q, k, v) = (
                half(n_head, q_len)?,
                half(n_kv_head, kv_len)?,
                half(n_kv_head, kv_len)?,
            );
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut best = f64::MAX;
            for _ in 0..10 {
                card.synchronize()?;
                let t = std::time::Instant::now();
                let _ = candle_flash_attn::flash_attn(&q, &k, &v, scale, true)?;
                card.synchronize()?;
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("{label:<26} {q_len:>5} {kv_len:>5}  {best:>8.3}");
        }
    }
    Ok(())
}
