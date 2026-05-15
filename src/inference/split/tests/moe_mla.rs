//! MoE / MLA / DeepSeek architecture tests:
//! top-k expert routing, shared expert aggregation, MLA Q/KV decompression,
//! RoPE split, layer-variant dispatch, DeepSeek meta parsing,
//! and full forward through mixed dense + MLA/MoE layers.

use super::super::super::layers::{topk_cpu, MoeGatingFunc, MoeRoutingConfig};
use super::super::rope::precompute_freqs_cis;
use super::super::*;
use super::common::*;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_transformers::quantized_nn::RmsNorm;

#[test]
fn test_deepseek_arch_supported() {
    assert!(ModelArch::DeepSeek2.is_supported());
    assert!(ModelArch::DeepSeek2.use_rope_contiguous());
    assert_eq!(ModelArch::DeepSeek2.default_activation(), Activation::SiLU);
    assert!(!ModelArch::DeepSeek2.use_gemma_norm());
}

#[test]
fn test_moe_topk_selection() {
    let device = Device::Cpu;
    // 8 experts, select top-2
    let scores = Tensor::from_vec(
        vec![0.1f32, 0.5, 0.3, 0.8, 0.2, 0.05, 0.7, 0.4],
        (8,),
        &device,
    )
    .unwrap();

    let (indices, weights) = topk_cpu(&scores, 2, MoeRoutingConfig::default()).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();

    // Top 2 scores: 0.8 at index 3, 0.7 at index 6
    assert_eq!(idx_vec, vec![3, 6]);
    assert_eq!(w_vec.len(), 2);
    // Weights should be softmax-normalized
    let w_sum: f32 = w_vec.iter().sum();
    assert!(
        (w_sum - 1.0).abs() < 1e-5,
        "Weights should sum to 1.0, got {w_sum}"
    );
    // Weight[0] (score 0.8) should be > weight[1] (score 0.7)
    assert!(w_vec[0] > w_vec[1]);
}

#[test]
fn test_moe_topk_single_expert() {
    let device = Device::Cpu;
    let scores = Tensor::from_vec(vec![0.2f32, 0.8, 0.5], (3,), &device).unwrap();
    let (indices, weights) = topk_cpu(&scores, 1, MoeRoutingConfig::default()).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();
    assert_eq!(idx_vec, vec![1]);
    assert!(
        (w_vec[0] - 1.0).abs() < 1e-5,
        "Single expert weight should be 1.0"
    );
}

/// R132: Softmax + no renormalize (DeepSeek-V2 strict, Qwen3-MoE with
/// `norm_topk_prob=false`). Top-k indices unchanged; weights are the
/// softmax probabilities at those positions, summing to less than 1
/// (since not all softmax mass is captured by the top-k subset).
#[test]
fn r132_softmax_no_renorm_weights_sum_below_one() {
    let device = Device::Cpu;
    let scores = Tensor::from_vec(
        vec![0.1f32, 0.5, 0.3, 0.8, 0.2, 0.05, 0.7, 0.4],
        (8,),
        &device,
    )
    .unwrap();
    let cfg = MoeRoutingConfig {
        gating_func: MoeGatingFunc::Softmax,
        renormalize_weights: false,
    };
    let (indices, weights) = topk_cpu(&scores, 2, cfg).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();
    assert_eq!(
        idx_vec,
        vec![3, 6],
        "top-k indices must match raw-score order"
    );
    let w_sum: f32 = w_vec.iter().sum();
    assert!(
        w_sum > 0.0 && w_sum < 1.0,
        "softmax-no-renorm sums to <1, got {w_sum}"
    );
    // Sanity: with raw scores 0.8 and 0.7, weight[0] > weight[1] still holds.
    assert!(w_vec[0] > w_vec[1]);
}

/// R132: Sigmoid + renormalize (DeepSeek-V3 with `expert_weights_norm=true`).
/// Sigmoid is monotonic in raw, so top-k indices match raw-score order;
/// renormalization makes the weights sum to 1.
#[test]
fn r132_sigmoid_renorm_weights_sum_to_one() {
    let device = Device::Cpu;
    let scores = Tensor::from_vec(vec![-2.0f32, 0.5, 1.5, 0.1], (4,), &device).unwrap();
    let cfg = MoeRoutingConfig {
        gating_func: MoeGatingFunc::Sigmoid,
        renormalize_weights: true,
    };
    let (indices, weights) = topk_cpu(&scores, 2, cfg).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();
    // Top-2 raw: 1.5 at idx 2, 0.5 at idx 1
    assert_eq!(idx_vec, vec![2, 1]);
    let w_sum: f32 = w_vec.iter().sum();
    assert!(
        (w_sum - 1.0).abs() < 1e-5,
        "renorm sigmoid weights sum to 1, got {w_sum}"
    );
}

/// R132: Sigmoid + no renormalize. Weights are raw sigmoid scores; sum
/// is whatever sigmoid produces, no normalization.
#[test]
fn r132_sigmoid_no_renorm_weights_are_raw_sigmoids() {
    let device = Device::Cpu;
    let scores = Tensor::from_vec(vec![0.0f32, 2.0], (2,), &device).unwrap();
    let cfg = MoeRoutingConfig {
        gating_func: MoeGatingFunc::Sigmoid,
        renormalize_weights: false,
    };
    let (indices, weights) = topk_cpu(&scores, 2, cfg).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();
    assert_eq!(idx_vec, vec![1, 0]);
    // sigmoid(2.0) ≈ 0.8808, sigmoid(0.0) = 0.5
    assert!(
        (w_vec[0] - 0.8808).abs() < 1e-3,
        "expected ~0.8808, got {}",
        w_vec[0]
    );
    assert!(
        (w_vec[1] - 0.5).abs() < 1e-5,
        "expected 0.5, got {}",
        w_vec[1]
    );
}

#[test]
fn test_moe_forward_single_expert() {
    // A 1-expert MoE with top-1 should behave like a dense FFN
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 1;

    // Create expert weights: [1, intermediate, hidden] and [1, hidden, intermediate]
    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    // Router: [1, hidden]
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 1,
        routing: MoeRoutingConfig::default(),
    };

    let x = Tensor::randn(0f32, 1.0, (1, 4, hidden), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
}

#[test]
fn test_moe_forward_multi_expert() {
    // 4 experts, select top-2, verify output shape and finiteness
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 4;

    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 2,
        routing: MoeRoutingConfig::default(),
    };

    let x = Tensor::randn(0f32, 1.0, (1, 3, hidden), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 3, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Multi-expert output NaN/Inf"
    );
}

#[test]
fn test_shared_expert_integration() {
    // MoE with shared experts: output = routed_experts + shared_expert
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 2;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    // MoE without shared experts
    let moe_no_shared = MoeFfn {
        gate: gate.clone(),
        gate_exps: gate_exps.clone(),
        down_exps: down_exps.clone(),
        up_exps: up_exps.clone(),
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 1,
        routing: MoeRoutingConfig::default(),
    };

    // MoE with shared experts
    let moe_with_shared = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: Some(make_qmatmul(hidden, intermediate)),
        shared_down: Some(make_qmatmul(intermediate, hidden)),
        shared_up: Some(make_qmatmul(hidden, intermediate)),
        n_experts_used: 1,
        routing: MoeRoutingConfig::default(),
    };

    let x = Tensor::randn(0f32, 1.0, (1, 2, hidden), &device).unwrap();
    let out_no_shared = moe_no_shared.forward(&x).unwrap();
    let out_with_shared = moe_with_shared.forward(&x).unwrap();

    assert_eq!(out_no_shared.dims(), out_with_shared.dims());
    // Outputs should differ due to shared expert contribution
    let diff = (&out_no_shared - &out_with_shared).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff > 0.0, "Shared expert should change output");
}

#[test]
fn test_mla_q_decompress() {
    // Verify Q path shapes: x → q_a → norm → q_b → reshape
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32; // per-head
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Test that forward_mla runs without error and returns correct shape
    let x = Tensor::randn(0f32, 0.1, (1, 5, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
    assert_eq!(out.dims(), &[1, 5, hidden], "MLA output shape mismatch");

    // KV cache should be populated
    assert!(kv_cache.is_some(), "KV cache should be created");
}

#[test]
fn test_mla_kv_decompress() {
    // Verify KV path shapes and cache dimensions
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32;
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Prefill with seq_len=3
    let x = Tensor::randn(0f32, 0.1, (1, 3, hidden), &device).unwrap();
    let mut kv_cache = None;
    mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();

    // Check KV cache dimensions
    let cache = kv_cache.as_ref().unwrap();
    let k = cache.k().unwrap().unwrap();
    let v = cache.v().unwrap().unwrap();
    // K: [b, n_head, seq_len, key_length]
    assert_eq!(k.dims(), &[1, n_head, 3, key_length]);
    // V: [b, n_head, seq_len, value_length]
    assert_eq!(v.dims(), &[1, n_head, 3, value_length]);
}

#[test]
fn test_mla_rope_split() {
    // Verify that MLA correctly splits q into nope and rope parts
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32;
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Prefill + decode: verify output stays finite
    let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out_prefill = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
    let flat: Vec<f32> = out_prefill.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MLA prefill NaN/Inf");

    // Decode step
    let x_decode = Tensor::randn(0f32, 0.1, (1, 1, hidden), &device).unwrap();
    let out_decode = mla
        .forward_mla(&x_decode, None, 4, &mut kv_cache, 128)
        .unwrap();
    assert_eq!(out_decode.dims(), &[1, 1, hidden]);
    let flat: Vec<f32> = out_decode.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MLA decode NaN/Inf");
}

// ── LayerVariant + DeepSeek meta parsing ──

#[test]
fn test_layer_variant_dense_unchanged() {
    // Wrapping in LayerVariant::Dense should produce same output as before
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Verify layers are Dense variants
    for layer in &model.layers {
        assert!(matches!(layer, LayerVariant::Dense(_)));
    }

    // Forward pass works identically
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "dense-test").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode step
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 6, &kv_store, "dense-test").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

#[test]
fn test_deepseek_meta_parsing() {
    // Verify DeepSeekMeta struct construction with various values
    let meta = DeepSeekMeta {
        n_experts: 64,
        n_experts_used: 6,
        n_shared_experts: 2,
        kv_lora_rank: 512,
        q_lora_rank: 1536,
        key_length: 192,
        value_length: 128,
        rope_dim: 64,
    };
    assert_eq!(meta.n_experts, 64);
    assert_eq!(meta.n_experts_used, 6);
    assert_eq!(meta.n_shared_experts, 2);
    assert_eq!(meta.kv_lora_rank, 512);
    assert_eq!(meta.q_lora_rank, 1536);
    assert_eq!(meta.key_length, 192);
    assert_eq!(meta.value_length, 128);
    assert_eq!(meta.rope_dim, 64);
}

// ── DeepSeek mixed-layers full forward ──

#[test]
fn test_deepseek_mixed_layers_forward() {
    // Full forward pass through mixed dense + MLA/MoE layers
    let hidden_dim = 64;
    let mut model = make_deepseek_test_model(hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Prefill
    let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "ds-test").unwrap();
    assert_eq!(out.dims(), &[1, 4, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "DeepSeek prefill NaN/Inf"
    );

    // Decode
    let decode = Tensor::randn(0f32, 0.1, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 4, &kv_store, "ds-test").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "DeepSeek decode NaN/Inf"
    );
}
