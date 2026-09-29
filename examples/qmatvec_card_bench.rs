//! How fast are the decode matrix-vector products on THIS card, per weight
//! byte — and do they alone explain a card-bound 7B decoding ~20% slower than
//! llama.cpp on the same card and the same file?
//!
//! Background (`docs/invariants/inference.md` § "A decode step can go to the
//! card as one CUDA graph"): Qwen2.5-Coder-7B Q4_K_M decodes at 47-48 tok/s
//! here and 57.5 through llama.cpp in the same binary. Neither CUDA graphs nor a
//! one-launch decode attention kernel moved it, so the time is in the weight
//! products or nowhere obvious.
//!
//! What this measures, on Qwen2.5-7B's shapes (28 layers):
//! - per shape, 200 products queued and one synchronize: ms per product and
//!   GB/s of weights read (the card's peak is ~448 GB/s on a 3070 Laptop);
//! - a whole token's products back to back (q k v o gate up down x 28 + the
//!   output head), one synchronize — the matmul share of a decoded token.
//!   K and V get their own matrices per layer: at 1-1.5 MB they fit the card's
//!   L2, and one reused copy would flatter them. The big matrices exceed L2
//!   whatever is done, so one copy each stands for all 28 layers.
//!
//! ```bash
//! CUDA_COMPUTE_CAP=86 cargo run --release --features candle-cuda --example qmatvec_card_bench
//! ```
//! Needs ~0.5 GB of card memory; run it with the node stopped.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn qmm(
    rows: usize,
    cols: usize,
    dtype: GgmlDType,
    dev: &Device,
) -> anyhow::Result<(QMatMul, usize)> {
    let w = Tensor::randn(0f32, 0.02, (rows, cols), &Device::Cpu)?;
    let q = QTensor::quantize_onto(&w, dtype, dev)?;
    let bytes = rows * cols / dtype.block_size() * dtype.type_size();
    Ok((QMatMul::from_qtensor(q)?, bytes))
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let (hidden, ffn, kv, vocab, layers) = (3584usize, 18944usize, 512usize, 152064usize, 28usize);
    let x_h = Tensor::randn(0f32, 1.0, (1, hidden), &dev)?;
    let x_f = Tensor::randn(0f32, 1.0, (1, ffn), &dev)?;

    let shapes = [
        ("q    3584x3584 Q4_K", hidden, hidden, GgmlDType::Q4K),
        ("k     512x3584 Q4_K", kv, hidden, GgmlDType::Q4K),
        ("v     512x3584 Q6_K", kv, hidden, GgmlDType::Q6K),
        ("o    3584x3584 Q4_K", hidden, hidden, GgmlDType::Q4K),
        ("gate 18944x3584 Q4_K", ffn, hidden, GgmlDType::Q4K),
        ("up   18944x3584 Q4_K", ffn, hidden, GgmlDType::Q4K),
        ("down 3584x18944 Q6_K", hidden, ffn, GgmlDType::Q6K),
        ("head 152064x3584 Q6_K", vocab, hidden, GgmlDType::Q6K),
    ];
    println!("per product, 200 queued, one synchronize (min of 5):");
    let mut mats = Vec::new();
    for (name, rows, cols, dt) in shapes {
        let (m, bytes) = qmm(rows, cols, dt, &dev)?;
        let x = if cols == ffn { &x_f } else { &x_h };
        for _ in 0..5 {
            m.forward(x)?;
        }
        dev.synchronize()?;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..200 {
                m.forward(x)?;
            }
            dev.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64() / 200.0);
        }
        println!(
            "  {name:24} {:8.1} us   {:6.1} GB/s   ({:.1} MB)",
            best * 1e6,
            bytes as f64 / best / 1e9,
            bytes as f64 / 1e6
        );
        mats.push((m, bytes, cols));
    }

    // A whole token: every layer's products in order, the head once.
    let ks: Vec<_> = (0..layers)
        .map(|_| qmm(kv, hidden, GgmlDType::Q4K, &dev).map(|m| m.0))
        .collect::<anyhow::Result<_>>()?;
    let vs: Vec<_> = (0..layers)
        .map(|_| qmm(kv, hidden, GgmlDType::Q6K, &dev).map(|m| m.0))
        .collect::<anyhow::Result<_>>()?;
    let (q, o, gate, up, down, head) = (
        &mats[0].0, &mats[3].0, &mats[4].0, &mats[5].0, &mats[6].0, &mats[7].0,
    );
    let token = || -> anyhow::Result<()> {
        for l in 0..layers {
            q.forward(&x_h)?;
            ks[l].forward(&x_h)?;
            vs[l].forward(&x_h)?;
            o.forward(&x_h)?;
            gate.forward(&x_h)?;
            up.forward(&x_h)?;
            down.forward(&x_f)?;
        }
        head.forward(&x_h)?;
        Ok(())
    };
    token()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let t = std::time::Instant::now();
        token()?;
        dev.synchronize()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let per_layer: usize = [0usize, 1, 2, 3, 4, 5, 6].iter().map(|&i| mats[i].1).sum();
    let bytes = per_layer * layers + mats[7].1;
    println!(
        "whole token's products: {:.2} ms  ({:.2} GB weights, {:.1} GB/s) — the 7B decodes a token in ~21 ms here, llama.cpp in ~17.4",
        best * 1e3,
        bytes as f64 / 1e9,
        bytes as f64 / best / 1e9
    );
    Ok(())
}
