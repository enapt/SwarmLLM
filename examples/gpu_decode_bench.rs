//! Where a CUDA decode step's attention time actually goes.
//!
//! The synchronised stage profile puts attention at ~40% of a generated token
//! even after flash-attention is routed correctly. Flash itself should not cost
//! that: at 528 KV positions the f16 cache it reads is ~60 MB across 28 layers,
//! which is ~1.7 ms of traffic at the ~70 GB/s the whole token achieves.
//!
//! The suspicion this prices: `run_attention`'s CUDA arm reshapes and converts
//! the WHOLE cache on every token, because the cache is stored f32 in BHSD and
//! flash wants f16 in BSHD:
//!
//! ```ignore
//! let k_bshd = k.transpose(1, 2)?.contiguous()?;   // full copy
//! let k_f16  = k_bshd.to_dtype(DType::F16)?;       // full pass again
//! ```
//!
//! That is O(history) work to add ONE position, so it grows with the
//! conversation — which is what a decode step should not do.
//!
//! ```bash
//! CUDA_COMPUTE_CAP=80 cargo run --release --features flash-attn --example gpu_decode_bench
//! ```
//!
//! Min of N with an explicit `synchronize()` inside the timed region: CUDA
//! queues work rather than running it, so a timer without one measures
//! submission. That mistake produced a 3977 tok/s reading earlier in this
//! project's history.

use candle_core::{DType, Device, Tensor};

fn bench<F: Fn() -> candle_core::Result<()>>(label: &str, bytes: f64, dev: &Device, f: F) {
    for _ in 0..3 {
        f().unwrap();
    }
    let _ = dev.synchronize();
    let mut best = f64::INFINITY;
    for _ in 0..12 {
        let t = std::time::Instant::now();
        f().unwrap();
        let _ = dev.synchronize();
        best = best.min(t.elapsed().as_secs_f64());
    }
    // Per-layer cost, and what it becomes across a 28-layer model.
    println!(
        "  {label:<46} {:>7.3} ms/layer   {:>6.2} ms/token   {:>6.1} GB/s",
        best * 1000.0,
        best * 1000.0 * 28.0,
        bytes / 1e9 / best
    );
}

fn main() -> anyhow::Result<()> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => anyhow::bail!("no CUDA device — build with --features flash-attn and run on a GPU"),
    };

    // llama-3.2-3b decode: one new query row against the cache.
    let (n_head, n_kv, head_dim) = (24usize, 8usize, 128usize);

    // **Measure ONE size per process when the number matters** — pass it as an
    // argument. Measuring several in one run inflates the LATER ones, badly.
    //
    // Measured 2026-08-10 on an idle RTX 3070, "WHOLE arm" at kv=912:
    //
    //     run alone                        13.04, 14.66 ms/token
    //     run after 272 and 528            77.33 ms/token      <- 5.9x inflated
    //     run first, before 528 and 272    16.24 ms/token
    //
    // Identical code, same card, same iteration count. Only the composite rows
    // drift — they allocate ~6 large temporaries per iteration, and the VRAM
    // state left behind by a previous size changes what that costs. The
    // single-op rows are stable throughout.
    //
    // The isolated figure REPRODUCES the number in `docs/FUTURE_WORK.md`
    // (13.04 at kv=912) to the decimal, so that table was measured correctly
    // and stands; it is the convenience of looping over sizes that is unsound.
    // An earlier version of this comment claimed the table was the unreliable
    // part — that was wrong, and measuring in isolation is what showed it.
    //
    //     cargo run --release --features flash-attn --example gpu_decode_bench -- 912
    //
    // **kv=2064 is suspect at ANY ordering** (~90-96 ms/token whether run first
    // or last). That cannot be what the forward pass does: end-to-end decode at
    // 3084 KV is ~40 ms/token in TOTAL across all 28 layers. In the real forward
    // `k` is a view into the cache's pre-allocated buffer with the model
    // resident, not a standalone allocation, so treat 2064 here as a property of
    // this bench's allocation pattern and not of decode. Prefer an end-to-end
    // measurement at that length — gotcha #266.
    let sizes: Vec<usize> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            vec![272usize, 528, 912, 2064]
        } else {
            args.iter().filter_map(|a| a.parse().ok()).collect()
        }
    };
    for kv_len in sizes {
        let f32b = (kv_len * n_kv * head_dim * 4) as f64;
        println!(
            "\nkv_len={kv_len}  (K or V alone: {:.1} MB f32, {:.1} MB f16 — x28 layers)",
            f32b / 1e6,
            f32b / 2e6
        );

        // Match how the real cache is shaped: `KvCache::k()` narrows a
        // preallocated `max_seq_len` buffer, so the live view is NOT contiguous
        // — the head stride spans the whole reservation, not just the used
        // part. A fresh exact-size tensor takes a different copy path, which is
        // how the first version of this benchmark produced an attention cost
        // larger than the whole measured token.
        let cap = kv_len.next_multiple_of(512).max(512);
        let q = Tensor::randn(0f32, 1., (1, n_head, 1, head_dim), &dev)?;
        let k = Tensor::randn(0f32, 1., (1, n_kv, cap, head_dim), &dev)?.narrow(2, 0, kv_len)?;
        let v = Tensor::randn(0f32, 1., (1, n_kv, cap, head_dim), &dev)?.narrow(2, 0, kv_len)?;

        // 1. The layout change alone.
        bench("transpose+contiguous on K and V", 4.0 * f32b, &dev, || {
            let _a = k.transpose(1, 2)?.contiguous()?;
            let _b = v.transpose(1, 2)?.contiguous()?;
            Ok(())
        });

        // 2. The dtype conversion alone, from an already-contiguous source.
        let k_bshd = k.transpose(1, 2)?.contiguous()?;
        let v_bshd = v.transpose(1, 2)?.contiguous()?;
        bench("to_dtype(F16) on K and V", 3.0 * f32b, &dev, || {
            let _a = k_bshd.to_dtype(DType::F16)?;
            let _b = v_bshd.to_dtype(DType::F16)?;
            Ok(())
        });

        // 2b. Both in ONE pass: read the strided f32 view, write contiguous
        //     f16. If `to_dtype` materialises a contiguous result from a
        //     strided source, the separate `.contiguous()` is pure waste.
        let fused = k.transpose(1, 2)?.to_dtype(DType::F16)?;
        println!(
            "       (fused transpose+to_dtype contiguous? {} — flash needs true)",
            fused.is_contiguous()
        );
        bench(
            "FUSED transpose->to_dtype(F16) on K and V",
            3.0 * f32b,
            &dev,
            || {
                let _a = k.transpose(1, 2)?.to_dtype(DType::F16)?;
                let _b = v.transpose(1, 2)?.to_dtype(DType::F16)?;
                Ok(())
            },
        );

        // 3. Flash itself, given operands already in the form it wants — the
        //    irreducible part of the step.
        let q_bshd = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)?;
        let k_f16 = k_bshd.to_dtype(DType::F16)?;
        let v_f16 = v_bshd.to_dtype(DType::F16)?;
        bench("flash_attn alone (operands ready)", f32b, &dev, || {
            let _ = candle_flash_attn::flash_attn(&q_bshd, &k_f16, &v_f16, 0.088388, false)?;
            Ok(())
        });

        // 4. The whole arm exactly as `run_attention` runs it today.
        bench("WHOLE arm, as shipped", 0.0, &dev, || {
            let qb = q.transpose(1, 2)?.contiguous()?;
            let kb = k.transpose(1, 2)?.contiguous()?;
            let vb = v.transpose(1, 2)?.contiguous()?;
            let out = candle_flash_attn::flash_attn(
                &qb.to_dtype(DType::F16)?,
                &kb.to_dtype(DType::F16)?,
                &vb.to_dtype(DType::F16)?,
                0.088388,
                false,
            )?;
            let _ = out.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            Ok(())
        });

        // 5. What it would cost if the cache were already f16 in BSHD, i.e.
        //    the ceiling a storage change could reach.
        bench("IF cache were f16+BSHD already", 0.0, &dev, || {
            let qb = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)?;
            let out = candle_flash_attn::flash_attn(&qb, &k_f16, &v_f16, 0.088388, false)?;
            let _ = out.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
            Ok(())
        });
    }
    Ok(())
}
