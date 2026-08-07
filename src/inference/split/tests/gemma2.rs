//! Gemma-2 specific behavior: GeLU activation, attention logit softcap,
//! and (under #[ignore]) real-GGUF vs shard-load cross-checks.

use super::super::super::layers::standard_attention;
use super::super::model::SplitModel;
use super::super::*;
use super::common::*;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};

#[test]
fn gemma2_gelu_activation_forward() {
    // Gemma 2 uses Gelu activation in MLP
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
    let mut model = make_gqa_test_model(
        1,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::Gelu,
        None,
        ModelArch::Gemma2,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "gemma2-gelu").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Gemma2 Gelu produced NaN/Inf"
    );
}

#[test]
fn gemma2_attn_logit_softcap() {
    // Test attention logit soft-capping (Gemma 2 feature)
    // Use stddev=1.0 so logits are large enough for softcap to visibly affect output
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 4, 2, 6, 32);

    let q = Tensor::randn(0f32, 1.0, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();

    // Without soft-capping
    let out_no_cap =
        standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None).unwrap();

    // With soft-capping (cap=50.0 like Gemma 2)
    let out_capped =
        standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, Some(50.0)).unwrap();

    // Both should produce valid output
    assert_eq!(out_no_cap.shape(), out_capped.shape());
    let flat: Vec<f32> = out_capped.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Soft-capped attention NaN/Inf"
    );

    // Outputs should differ (soft-capping changes the attention weights)
    let diff = (&out_no_cap - &out_capped).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff > 0.0, "Soft-capping should change the output");
}

#[test]
fn gemma2_full_forward_with_softcap() {
    // Gemma 2 end-to-end: Gelu + softcap + GQA
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::Gelu,
        Some(50.0),
        ModelArch::Gemma2,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Prefill
    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "gemma2-full").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 8, &kv_store, "gemma2-full").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

// ── Real-GGUF cross-checks (require Gemma-2 model on disk; #[ignore] by default) ──

#[test]
#[ignore] // Run with: cargo test gemma2_real_gguf -- --ignored --nocapture
fn gemma2_real_gguf_vs_shards() {
    use candle_core::Tensor;
    let gguf_path = std::path::Path::new(
        "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
    );
    if !gguf_path.exists() {
        eprintln!("Skipping: GGUF not found at {}", gguf_path.display());
        return;
    }

    // Load from full GGUF
    let mut model = SplitModel::load_from_gguf(gguf_path, 0, 26, true, true, false)
        .expect("Failed to load from GGUF");

    // Use the tokenizer to get the same tokens our API uses
    let prompt_tokens: Vec<u32> = vec![
        2, 2, 106, 1645, 108, 1841, 603, 573, 6037, 576, 6081, 235336, 107, 108, 106, 2516, 108,
    ];
    let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0) // [1, 17]
        .unwrap()
        .to_dtype(candle_core::DType::I64)
        .unwrap();

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let logits = model
        .forward(&input, 0, &kv_store, "gemma2-gguf-test")
        .expect("Forward pass failed");

    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let (argmax_idx, argmax_val) = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    eprintln!("GGUF logits: argmax={} score={:.4}", argmax_idx, argmax_val);
    eprintln!(
        "  min={:.4} max={:.4} dim={}",
        flat.iter().cloned().fold(f32::INFINITY, f32::min),
        flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        flat.len()
    );

    // Expected: token 651 ("The") should be near the top
    let the_score = flat[651];
    eprintln!("  token 651 ('The') score={:.4}", the_score);
    eprintln!("  token 235274 ('1') score={:.4}", flat[235274]);

    // Save logits for external comparison
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("/tmp/gemma2_our_logits.bin", &bytes).ok();
    eprintln!(
        "  Saved {} logits to /tmp/gemma2_our_logits.bin",
        flat.len()
    );
}

/// Test Gemma-2 with single token (no mask) to eliminate mask issues.
#[test]
#[ignore]
fn gemma2_single_token() {
    use candle_core::{Device, Tensor};
    let gguf_path = std::path::Path::new(
        "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
    );
    if !gguf_path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    // Single BOS token — no mask needed, no flash attention
    let prompt_tokens: Vec<u32> = vec![2];
    let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(candle_core::DType::I64)
        .unwrap();

    let mut model =
        SplitModel::load_from_gguf(gguf_path, 0, 26, true, true, false).expect("Failed to load");

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let logits = model
        .forward(&input, 0, &kv_store, "gemma2-single-tok")
        .expect("Forward failed");

    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let (argmax_idx, argmax_val) = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("Single token (BOS): argmax={argmax_idx} score={argmax_val:.4}");
    eprintln!("  token 2 score: {:.4}", flat[2]);
    eprintln!("  token 108 score: {:.4}", flat[108]);

    // Save for comparison
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("/tmp/gemma2_single_token_logits.bin", &bytes).ok();
    eprintln!("  Saved logits to /tmp/gemma2_single_token_logits.bin");
}

/// Test embedding dequantization matches Python reference.
#[test]
#[ignore]
fn gemma2_embedding_verification() {
    use candle_core::{quantized::gguf_file, Device, Tensor};

    let gguf_path = "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
    let path = std::path::Path::new(gguf_path);
    if !path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut cursor).unwrap();
    let device = Device::Cpu;

    // Load and dequantize embedding
    let embd_qt = ct
        .tensor(&mut cursor, "token_embd.weight", &device)
        .unwrap();
    let embd = embd_qt.dequantize(&device).unwrap();
    eprintln!("Embedding shape: {:?}", embd.shape());

    // Get row 2 (BOS token)
    let row2 = embd.i(2).unwrap();
    let row2_vals: Vec<f32> = row2.to_vec1().unwrap();
    eprintln!("Row 2 (BOS) first 8: {:?}", &row2_vals[..8]);
    eprintln!(
        "Row 2 (BOS) last 8: {:?}",
        &row2_vals[row2_vals.len() - 8..]
    );

    // Compare with Python reference
    let py_ref = std::fs::read("/tmp/gemma2_embed_row2.npy").ok();
    if let Some(ref npy_bytes) = py_ref {
        // Parse npy format: skip header, read f32 values
        // Simple npy parser: header starts with \x93NUMPY, has length info
        let header_len = 10 + npy_bytes[8] as usize + ((npy_bytes[9] as usize) << 8);
        let data_bytes = &npy_bytes[header_len..];
        let py_vals: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("Python ref first 8: {:?}", &py_vals[..8]);

        let mut max_diff = 0f32;
        let mut mismatches = 0;
        for (i, (a, b)) in row2_vals.iter().zip(py_vals.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-4 && i < 20 {
                eprintln!("  MISMATCH [{i}] rust={a:.6} python={b:.6} diff={diff:.6}");
                mismatches += 1;
            }
        }
        eprintln!("Max embedding diff: {max_diff:.6}, mismatches (>1e-4): {mismatches}");
    } else {
        eprintln!("No Python reference found at /tmp/gemma2_embed_row2.npy");
    }

    // Now test: embedding lookup → scale by sqrt(2304) → final norm → output projection
    // This is the 0-layer forward pass
    let emb = Embedding::new(embd.clone(), 2304);

    // Token 2 (BOS) lookup
    let ids = Tensor::new(&[2u32], &device).unwrap();
    let looked_up = emb.forward(&ids).unwrap(); // (1, 2304)
    let scaled = looked_up.affine((2304f64).sqrt(), 0.0).unwrap(); // scale by sqrt(hidden_dim)

    // Apply final norm (with +1 offset)
    let norm_qt = ct
        .tensor(&mut cursor, "output_norm.weight", &device)
        .unwrap();
    let norm_w = norm_qt.dequantize(&device).unwrap();
    let norm_w_plus1 = (norm_w + 1.0).unwrap(); // Gemma +1
    let normed = candle_nn::ops::rms_norm(&scaled, &norm_w_plus1, 1e-6).unwrap();

    // Output projection using dequantized embedding
    let logits = normed.matmul(&embd.t().unwrap()).unwrap();
    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let argmax = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("0-layer logits: argmax={} score={:.4}", argmax.0, argmax.1);
    eprintln!("  token 2 score: {:.4}", flat[2]);
    eprintln!("  token 108 score: {:.4}", flat[108]);
}

/// Test QMatMul vs dequantized matmul for output projection.
/// Diagnoses whether the sorted-correlation issue is in the output projection.
#[test]
#[ignore]
fn gemma2_output_projection_qmatmul_vs_deq() {
    use candle_core::{quantized::gguf_file, Device, Tensor};

    let gguf_path = "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
    let path = std::path::Path::new(gguf_path);
    if !path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut cursor).unwrap();
    let device = Device::Cpu;

    // Load token_embd.weight as both QTensor and dequantized
    let embd_qt = ct
        .tensor(&mut cursor, "token_embd.weight", &device)
        .unwrap();
    eprintln!("token_embd.weight QTensor shape: {:?}", embd_qt.shape());

    let embd_deq = embd_qt.dequantize(&device).unwrap();
    eprintln!("Dequantized embedding shape: {:?}", embd_deq.shape());

    // Create QMatMul from the QTensor
    let qmm = QMatMul::from_qtensor(
        ct.tensor(&mut cursor, "token_embd.weight", &device)
            .unwrap(),
    )
    .unwrap();

    // Create a random hidden state (simulating post-norm output)
    let hidden = Tensor::randn(0f32, 1.0, (1, 2304), &device).unwrap();

    // Method 1: QMatMul (our current approach)
    let logits_qmm = qmm.forward(&hidden).unwrap();
    let flat_qmm: Vec<f32> = logits_qmm.flatten_all().unwrap().to_vec1().unwrap();

    // Method 2: Dequantized matmul (reference approach)
    // embd_deq shape is (256000, 2304), we need hidden @ embd_deq.T
    let logits_deq = hidden.matmul(&embd_deq.t().unwrap()).unwrap();
    let flat_deq: Vec<f32> = logits_deq.flatten_all().unwrap().to_vec1().unwrap();

    eprintln!(
        "QMatMul logits: argmax={}",
        flat_qmm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    eprintln!(
        "Deq logits: argmax={}",
        flat_deq
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );

    // Check if they agree
    let mut max_diff = 0f32;
    let mut sum_diff_sq = 0f64;
    for (i, (a, b)) in flat_qmm.iter().zip(flat_deq.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff_sq += (diff as f64) * (diff as f64);
        if i < 5 {
            eprintln!("  [{i}] qmm={a:.6} deq={b:.6} diff={diff:.6}");
        }
    }
    let rmse = (sum_diff_sq / flat_qmm.len() as f64).sqrt();
    eprintln!("Max diff: {max_diff:.6}, RMSE: {rmse:.6}");

    // Check correlation
    let mean_q: f64 = flat_qmm.iter().map(|v| *v as f64).sum::<f64>() / flat_qmm.len() as f64;
    let mean_d: f64 = flat_deq.iter().map(|v| *v as f64).sum::<f64>() / flat_deq.len() as f64;
    let mut cov = 0f64;
    let mut var_q = 0f64;
    let mut var_d = 0f64;
    for (q, d) in flat_qmm.iter().zip(flat_deq.iter()) {
        let dq = *q as f64 - mean_q;
        let dd = *d as f64 - mean_d;
        cov += dq * dd;
        var_q += dq * dq;
        var_d += dd * dd;
    }
    let corr = cov / (var_q.sqrt() * var_d.sqrt());
    eprintln!("Pearson correlation (QMatMul vs Deq): {corr:.6}");

    // They should be highly correlated (>0.99) — just quantization error
    assert!(
        corr > 0.99,
        "QMatMul and dequantized matmul should agree: corr={corr}"
    );
}
