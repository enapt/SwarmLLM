//! MoE / MLA / DeepSeek architecture tests:
//! top-k expert routing, shared expert aggregation, MLA Q/KV decompression,
//! RoPE split, layer-variant dispatch, DeepSeek meta parsing,
//! and full forward through mixed dense + MLA/MoE layers.

use super::super::super::layers::{topk_cpu, MoeGatingFunc, MoeRoutingConfig};
use super::super::rope::precompute_freqs_cis;
use super::super::*;
use super::common::*;
use crate::inference::residual_norm::RmsNorm;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};

/// DeepSeek-2 and StarCoder2 are RECOGNISED — so a refusal can name them — and
/// not offered: no file in circulation loads here (FUTURE_WORK #116, #118). The
/// list the refusal quotes must agree with the predicate the loader and the
/// download paths ask.
#[test]
fn a_family_no_real_file_loads_is_recognised_but_not_offered() {
    assert_eq!(ModelArch::from_gguf_arch("deepseek2"), ModelArch::DeepSeek2);
    assert!(!ModelArch::DeepSeek2.is_supported());
    for arch in ModelArch::supported_list() {
        assert!(ModelArch::from_gguf_arch(arch).is_supported(), "{arch}");
    }
    assert!(!ModelArch::supported_list().contains(&"deepseek2"));
    // StarCoder2 (#118) the same way.
    assert!(!ModelArch::from_gguf_arch("starcoder2").is_supported());
    assert!(!ModelArch::supported_list().contains(&"starcoder2"));
    // Qwen 3.5's MoE was refused the same way until checked (#228 (1)); the
    // list must carry it now that the predicate admits it.
    assert!(ModelArch::supported_list().contains(&"qwen35moe"));
    // llama.cpp: LLAMA_ROPE_TYPE_NORM (interleaved) — see `use_rope_contiguous`.
    assert!(!ModelArch::DeepSeek2.use_rope_contiguous());
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
        weight_before_ffn: false,
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
        weight_before_ffn: false,
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
        weight_before_ffn: false,
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
        experts: crate::inference::layers::ExpertFfn::from_stacked(
            &(gate_exps),
            &(up_exps),
            &(down_exps),
        )
        .unwrap(),
        stacked: None,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        shared_gate_inp: None,
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
        experts: crate::inference::layers::ExpertFfn::from_stacked(
            &(gate_exps),
            &(up_exps),
            &(down_exps),
        )
        .unwrap(),
        stacked: None,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        shared_gate_inp: None,
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
        experts: crate::inference::layers::ExpertFfn::from_stacked(
            &(gate_exps.clone()),
            &(up_exps.clone()),
            &(down_exps.clone()),
        )
        .unwrap(),
        stacked: None,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        shared_gate_inp: None,
        n_experts_used: 1,
        routing: MoeRoutingConfig::default(),
    };

    // MoE with shared experts
    let moe_with_shared = MoeFfn {
        gate,
        experts: crate::inference::layers::ExpertFfn::from_stacked(
            &(gate_exps),
            &(up_exps),
            &(down_exps),
        )
        .unwrap(),
        stacked: None,
        shared_gate: Some(make_qmatmul(hidden, intermediate)),
        shared_down: Some(make_qmatmul(intermediate, hidden)),
        shared_up: Some(make_qmatmul(hidden, intermediate)),
        shared_gate_inp: None,
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
    };

    // Test that forward_mla runs without error and returns correct shape
    let x = Tensor::randn(0f32, 0.1, (1, 5, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out = mla.forward_mla(&x, None, 0, &mut kv_cache, 128, 0).unwrap();
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
    };

    // Prefill with seq_len=3
    let x = Tensor::randn(0f32, 0.1, (1, 3, hidden), &device).unwrap();
    let mut kv_cache = None;
    mla.forward_mla(&x, None, 0, &mut kv_cache, 128, 0).unwrap();

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
    };

    // Prefill + decode: verify output stays finite
    let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out_prefill = mla.forward_mla(&x, None, 0, &mut kv_cache, 128, 0).unwrap();
    let flat: Vec<f32> = out_prefill.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MLA prefill NaN/Inf");

    // Decode step
    let x_decode = Tensor::randn(0f32, 0.1, (1, 1, hidden), &device).unwrap();
    let out_decode = mla
        .forward_mla(&x_decode, None, 4, &mut kv_cache, 128, 0)
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

/// The batched MoE forward (one host copy of the router scores per layer, one
/// `index_add` per expert) computes the SAME bits as the per-token algorithm it
/// replaced, on the processor, for every routing policy — FUTURE_WORK #114d.
///
/// The reference below is the old algorithm kept verbatim: per token, the
/// scores row copied to the host, top-k picked, the default policy's weights
/// softmaxed by the candle op on the device, and every weighted row added into
/// its token one `narrow` at a time, then a `cat` over all tokens.
#[test]
fn the_batched_moe_forward_is_bit_identical_to_routing_token_by_token() {
    use crate::inference::layers::{topk_host, ExpertFfn};

    fn per_token_reference(moe: &MoeFfn, x: &Tensor) -> Tensor {
        let (b, s, hidden) = x.dims3().unwrap();
        let n = b * s;
        let x_flat = x.reshape((n, hidden)).unwrap();
        let device = x.device();
        let scores = x_flat.matmul(&moe.gate.t().unwrap()).unwrap();
        let n_experts = moe.gate.dim(0).unwrap();
        let mut batches: Vec<Vec<(usize, f32)>> = vec![vec![]; n_experts];
        for pos in 0..n {
            let row: Vec<f32> = scores.get(pos).unwrap().to_vec1().unwrap();
            let (idx, w) = topk_host(&row, moe.n_experts_used, moe.routing);
            let w: Vec<f32> = if moe.routing == MoeRoutingConfig::default() {
                let raw: Vec<f32> = idx.iter().map(|&i| row[i]).collect();
                let k = raw.len();
                candle_nn::ops::softmax(&Tensor::from_vec(raw, (k,), device).unwrap(), 0)
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            } else {
                w
            };
            for (e, w) in idx.into_iter().zip(w) {
                batches[e].push((pos, w));
            }
        }
        let mut acc: Vec<Option<Tensor>> = vec![None; n];
        for (e, batch) in batches.iter().enumerate() {
            if batch.is_empty() {
                continue;
            }
            let ids: Vec<i64> = batch.iter().map(|&(p, _)| p as i64).collect();
            let ids = Tensor::from_vec(ids, (batch.len(),), device).unwrap();
            let input = x_flat.index_select(&ids, 0).unwrap();
            let gu = QMatMul::forward_shared(&input, &[&moe.experts[e].gate, &moe.experts[e].up])
                .unwrap();
            let out = moe.experts[e]
                .down
                .forward(&crate::inference::fast_math::silu_mul(&gu[0], &gu[1]).unwrap())
                .unwrap();
            let w: Vec<f32> = batch.iter().map(|&(_, w)| w).collect();
            let w = Tensor::from_vec(w, (batch.len(), 1), device).unwrap();
            let weighted = out.broadcast_mul(&w).unwrap();
            for (i, &(pos, _)) in batch.iter().enumerate() {
                let c = weighted.narrow(0, i, 1).unwrap();
                acc[pos] = Some(match acc[pos].take() {
                    Some(e) => (e + c).unwrap(),
                    None => c,
                });
            }
        }
        let zero = Tensor::zeros((1, hidden), DType::F32, device).unwrap();
        let rows: Vec<&Tensor> = acc.iter().map(|o| o.as_ref().unwrap_or(&zero)).collect();
        let mut output = Tensor::cat(&rows, 0).unwrap();
        // The shared expert, as the old path added it (unchanged by the batching).
        if let (Some(sg), Some(sd), Some(su)) = (&moe.shared_gate, &moe.shared_down, &moe.shared_up)
        {
            let combined = crate::inference::fast_math::silu_mul(
                &sg.forward(&x_flat).unwrap(),
                &su.forward(&x_flat).unwrap(),
            )
            .unwrap();
            let mut shared = sd.forward(&combined).unwrap();
            if let Some(w) = &moe.shared_gate_inp {
                let logit = x_flat.matmul(&w.reshape((hidden, 1)).unwrap()).unwrap();
                shared = shared
                    .broadcast_mul(&candle_nn::ops::sigmoid(&logit).unwrap())
                    .unwrap();
            }
            output = (output + shared).unwrap();
        }
        output.reshape((b, s, hidden)).unwrap()
    }

    let device = Device::Cpu;
    let (hidden, intermediate, n_experts) = (32, 48, 8);
    let policies = [
        MoeRoutingConfig::default(),
        MoeRoutingConfig {
            gating_func: MoeGatingFunc::Softmax,
            renormalize_weights: false,
            weight_before_ffn: false,
        },
        MoeRoutingConfig {
            gating_func: MoeGatingFunc::Sigmoid,
            renormalize_weights: true,
            weight_before_ffn: false,
        },
        MoeRoutingConfig {
            gating_func: MoeGatingFunc::Sigmoid,
            renormalize_weights: false,
            weight_before_ffn: false,
        },
    ];
    let dense = |o: usize, i: usize| {
        QMatMul::from_dense(Tensor::randn(0f32, 0.05, (o, i), &device).unwrap())
    };
    // No shared expert (DeepSeek-free families), an ungated one (Llama 4,
    // DeepSeek) and a sigmoid-gated one (Qwen2-MoE) — the batched scatter must
    // be the same sum beside each.
    for shared in 0..3 {
        for routing in policies {
            for k in [1usize, 2, 3] {
                let stack = |o, i| Tensor::randn(0f32, 0.05, (n_experts, o, i), &device).unwrap();
                let moe = MoeFfn {
                    gate: Tensor::randn(0f32, 0.3, (n_experts, hidden), &device).unwrap(),
                    experts: ExpertFfn::from_stacked(
                        &stack(intermediate, hidden),
                        &stack(intermediate, hidden),
                        &stack(hidden, intermediate),
                    )
                    .unwrap(),
                    stacked: None,
                    shared_gate: (shared > 0).then(|| dense(intermediate, hidden)),
                    shared_down: (shared > 0).then(|| dense(hidden, intermediate)),
                    shared_up: (shared > 0).then(|| dense(intermediate, hidden)),
                    shared_gate_inp: (shared == 2)
                        .then(|| Tensor::randn(0f32, 0.3, (hidden,), &device).unwrap()),
                    n_experts_used: k,
                    routing,
                };
                // A prompt-sized batch, a single decode token, and two sequences.
                for (batch, tokens) in [(1usize, 37usize), (1, 1), (2, 5)] {
                    let x = Tensor::randn(0f32, 1.0, (batch, tokens, hidden), &device).unwrap();
                    let got: Vec<f32> = moe
                        .forward(&x)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    let want: Vec<f32> = per_token_reference(&moe, &x)
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    let differing = got
                        .iter()
                        .zip(&want)
                        .filter(|(a, b)| a.to_bits() != b.to_bits() && !(**a == 0.0 && **b == 0.0))
                        .count();
                    assert_eq!(
                        differing,
                        0,
                        "{routing:?} k={k} batch={batch} tokens={tokens} shared={shared}: \
                     {differing} of {} values differ",
                        got.len()
                    );
                }
            }
        }
    }
}

/// A few tokens' routed experts on the processor, run as (token, expert) pairs
/// in one parallel region per projection (`experts_in_one_region`, #138),
/// compute the SAME bits as the per-expert path — for quantized experts of
/// one activation type and of mixed ones, weighting before or after the
/// expert, a decoded token to a prompt, in one region or cut into several
/// (down to one token each). And the processor takes it for every count.
#[test]
fn one_region_experts_are_bit_identical_to_one_at_a_time() {
    use crate::inference::layers::ExpertFfn;
    use candle_core::quantized::GgmlDType as G;

    let device = Device::Cpu;
    let (hidden, intermediate, n_experts) = (256usize, 512usize, 16usize);
    let quantized = |o: usize, i: usize, dtype: G| {
        let w = Tensor::randn(0f32, 0.05, (o, i), &device).unwrap();
        QMatMul::from_qtensor(QTensor::quantize(&w, dtype).unwrap()).unwrap()
    };
    // One activation type throughout; gate and up quantized for different
    // ones (Q8_0 vs Q8_K), so a token's row is quantized twice; and the mix a
    // Q2_K file carries.
    let mixes = [
        (G::Q4K, G::Q4K, G::Q6K),
        (G::Q4_0, G::Q4K, G::Q8_0),
        (G::Q2K, G::Q2K, G::Q3K),
    ];
    let policies = [
        MoeRoutingConfig::default(),
        MoeRoutingConfig {
            gating_func: MoeGatingFunc::Sigmoid,
            renormalize_weights: false,
            weight_before_ffn: true,
        },
    ];
    for (gate_t, up_t, down_t) in mixes {
        for routing in policies {
            for k in [2usize, 8] {
                let moe = MoeFfn {
                    gate: Tensor::randn(0f32, 0.3, (n_experts, hidden), &device).unwrap(),
                    experts: (0..n_experts)
                        .map(|_| ExpertFfn {
                            gate: quantized(intermediate, hidden, gate_t),
                            up: quantized(intermediate, hidden, up_t),
                            down: quantized(hidden, intermediate, down_t),
                        })
                        .collect(),
                    stacked: None,
                    shared_gate: None,
                    shared_down: None,
                    shared_up: None,
                    shared_gate_inp: None,
                    n_experts_used: k,
                    routing,
                };
                for tokens in [1usize, 3, 8, 37, 100] {
                    let x = Tensor::randn(0f32, 1.0, (1, tokens, hidden), &device).unwrap();
                    let x_flat = x.reshape((tokens, hidden)).unwrap();
                    let batches = moe.route(&x_flat).unwrap();
                    assert!(
                        moe.experts_fit_one_region(&x, &batches),
                        "{tokens} tokens of quantized experts on the processor must take one region"
                    );
                    let flat =
                        |t: Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
                    let want = flat(
                        moe.experts_one_at_a_time(&x_flat, &batches, tokens, hidden)
                            .unwrap(),
                    );
                    let got = flat(
                        moe.experts_in_one_region(&x_flat, &batches, tokens, hidden)
                            .unwrap(),
                    );
                    // The same tokens a few pairs per region, and one token each.
                    let cut = |pairs: usize| {
                        flat(
                            moe.experts_in_regions_of(&x_flat, &batches, tokens, hidden, pairs)
                                .unwrap(),
                        )
                    };
                    for (how, got) in [
                        ("one region", &got),
                        ("24 pairs", &cut(24)),
                        ("1 token", &cut(1)),
                    ] {
                        let differing = got
                            .iter()
                            .zip(&want)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        assert_eq!(
                            differing,
                            0,
                            "{gate_t:?}/{up_t:?}/{down_t:?} {routing:?} k={k} tokens={tokens} \
                             {how}: {differing} of {} values differ",
                            got.len()
                        );
                    }
                    assert_eq!(
                        flat(moe.forward(&x).unwrap()),
                        got,
                        "forward took another path"
                    );
                }
            }
        }
    }
}

/// The vendored `quantized::matmul_many` (#138) answers what `forward` answers
/// for the same rows, bit for bit — one row or many, across the 128-row block
/// `vec_dot_rows` is called on, a range of a quantized input and rows copied
/// out of one (`gather`), for types with their own several-row kernel (Q4_K,
/// Q6_K) and without (Q2_K, Q8_0), and a weight whose rows end mid-task.
#[test]
fn many_row_products_in_one_region_match_forward() {
    use candle_core::quantized::{matmul_many, GgmlDType as G, MatMulJob, QMatMul};
    use candle_core::Module;
    let device = Device::Cpu;
    let (n, k, total) = (96usize, 256usize, 300usize);
    let xs = Tensor::randn(0f32, 1.0, (total, k), &device).unwrap();
    let rows_of = |r: &[usize]| -> Vec<f32> {
        let ids: Vec<u32> = r.iter().map(|&i| i as u32).collect();
        let ids = Tensor::from_vec(ids, r.len(), &device).unwrap();
        let picked = xs.index_select(&ids, 0).unwrap();
        picked.flatten_all().unwrap().to_vec1().unwrap()
    };
    for dtype in [G::Q4K, G::Q6K, G::Q2K, G::Q8_0] {
        let w = Tensor::randn(0f32, 0.05, (n, k), &device).unwrap();
        let w = std::sync::Arc::new(QTensor::quantize(&w, dtype).unwrap());
        let qmm = QMatMul::from_arc(w.clone()).unwrap();
        let all = w
            .quantize_rows(&xs.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .unwrap();
        let gathered_rows: Vec<usize> = (0..total).rev().step_by(2).collect();
        let gathered = all.gather(&gathered_rows).unwrap();
        // (input, its rows the job reads, which rows of `xs` those are)
        let cases: Vec<(&_, std::ops::Range<usize>, Vec<usize>)> = vec![
            (&all, 0..1, vec![0]),
            (&all, 41..42, vec![41]),
            (&all, 3..10, (3..10).collect()),
            (&all, 7..137, (7..137).collect()),
            (&all, 0..total, (0..total).collect()),
            (&gathered, 5..140, gathered_rows[5..140].to_vec()),
        ];
        let mut outs: Vec<Vec<f32>> = cases
            .iter()
            .map(|(_, r, _)| vec![0f32; n * r.len()])
            .collect();
        let jobs = cases
            .iter()
            .zip(outs.iter_mut())
            .map(|((lhs, rows, _), dst)| MatMulJob {
                weight: &w,
                lhs,
                lhs_rows: rows.clone(),
                dst,
            })
            .collect();
        matmul_many(jobs).unwrap();
        for ((_, rows, picked), got) in cases.iter().zip(&outs) {
            let m = picked.len();
            let input = Tensor::from_vec(rows_of(picked), (m, k), &device).unwrap();
            let want: Vec<f32> = qmm
                .forward(&input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let differing = (0..m)
                .flat_map(|r| (0..n).map(move |j| (r, j)))
                .filter(|&(r, j)| got[j * m + r].to_bits() != want[r * n + j].to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "{dtype:?} rows {rows:?}: {differing} of {} differ",
                m * n
            );
        }
    }
}

/// Routing on the device (`topk_on_device`, the stacked experts' path on a
/// card) picks the experts `topk_host` picks and gives them its weights, for
/// every policy — checked here on the processor, where both run.
#[test]
fn routing_on_the_device_agrees_with_routing_on_the_host() {
    use crate::inference::layers::{topk_host, topk_on_device};
    let device = Device::Cpu;
    let (tokens, n_experts) = (5usize, 64usize);
    let scores = Tensor::randn(0f32, 2.0, (tokens, n_experts), &device).unwrap();
    let rows: Vec<Vec<f32>> = scores.to_vec2().unwrap();
    for gating_func in [MoeGatingFunc::Softmax, MoeGatingFunc::Sigmoid] {
        for renormalize_weights in [true, false] {
            let routing = MoeRoutingConfig {
                gating_func,
                renormalize_weights,
                weight_before_ffn: false,
            };
            for k in [1usize, 8] {
                let (ids, weights) = topk_on_device(&scores, k, routing).unwrap();
                let ids: Vec<Vec<u32>> = ids.to_vec2().unwrap();
                let weights: Vec<Vec<f32>> = weights.to_vec2().unwrap();
                for (t, row) in rows.iter().enumerate() {
                    let (want_ids, want_w) = topk_host(row, k, routing);
                    let got: Vec<usize> = ids[t].iter().map(|&i| i as usize).collect();
                    assert_eq!(got, want_ids, "{routing:?} k={k} token {t}");
                    for (g, w) in weights[t].iter().zip(&want_w) {
                        assert!(
                            (g - w).abs() <= 1e-6,
                            "{routing:?} k={k} token {t}: {g} vs {w}"
                        );
                    }
                }
            }
        }
    }
}

/// Which layers keep their experts stacked for the grouped card kernel.
#[test]
fn only_a_card_with_types_the_grouped_kernel_reads_keeps_experts_stacked() {
    use crate::inference::layers::StackedExperts;
    use candle_core::quantized::GgmlDType as G;
    let default = MoeRoutingConfig::default();
    let llama4 = MoeRoutingConfig {
        gating_func: MoeGatingFunc::Sigmoid,
        renormalize_weights: false,
        weight_before_ffn: true,
    };
    let fits = |card, dtypes, widths, routing| StackedExperts::fit(card, dtypes, widths, routing);
    assert!(fits(true, [G::Q4K, G::Q4K, G::Q6K], [2048, 768], default));
    assert!(fits(true, [G::Q2K, G::Q2K, G::Q3K], [2048, 768], default));
    assert!(fits(
        true,
        [G::Q8_0, G::Q8_0, G::Q8_0],
        [2048, 768],
        default
    ));
    // The processor keeps one matrix per expert (`experts_in_one_region`).
    assert!(!fits(false, [G::Q4K, G::Q4K, G::Q6K], [2048, 768], default));
    // A type the kernel does not read, anywhere in the three.
    assert!(!fits(true, [G::Q4_0, G::Q4K, G::Q6K], [2048, 768], default));
    assert!(!fits(true, [G::Q4K, G::Q4K, G::F16], [2048, 768], default));
    // Llama 4 weighs the expert's input; the kernel scales outputs.
    assert!(!fits(true, [G::Q4K, G::Q4K, G::Q6K], [2048, 768], llama4));
    // The kernel asserts every input width is a multiple of 8.
    assert!(!fits(
        true,
        [G::Q8_0, G::Q8_0, G::Q8_0],
        [2048, 772],
        default
    ));
}

/// The grouped kernel reads each operand from the start of its storage, so a
/// later token chunk of a layer's input — a view part-way into the buffer —
/// must reach it as its own buffer holding the same values.
#[test]
fn an_operand_handed_to_the_grouped_kernel_starts_at_its_own_storage() {
    use crate::inference::layers::own_storage;
    let x = Tensor::arange(0f32, 24.0, &Device::Cpu)
        .unwrap()
        .reshape((6, 4))
        .unwrap();
    // A second chunk: contiguous, but four rows into the buffer.
    let chunk = x.narrow(0, 2, 3).unwrap();
    assert!(chunk.is_contiguous() && chunk.layout().start_offset() == 8);
    let owned = own_storage(chunk.clone()).unwrap();
    assert_eq!(owned.layout().start_offset(), 0);
    assert!(owned.is_contiguous());
    assert_eq!(
        owned.to_vec2::<f32>().unwrap(),
        chunk.to_vec2::<f32>().unwrap()
    );
    // A transposed view is laid out in order too.
    let t = own_storage(x.t().unwrap()).unwrap();
    assert!(t.is_contiguous() && t.layout().start_offset() == 0);
    assert_eq!(
        t.to_vec2::<f32>().unwrap(),
        x.t().unwrap().to_vec2::<f32>().unwrap()
    );
}

/// Llama 4 weights each chosen expert's INPUT by `sigmoid(router logit)`
/// (llama.cpp `build_moe_ffn`'s `weight_before_ffn`), not its output.
///
/// With top-1, `x` and any positive multiple of it choose the same expert, and
/// softmax + renormalize over one pick weighs it exactly 1.0 — so the Llama 4
/// layer on `x` must equal the plain layer on `sigmoid(l)·x`, and must NOT
/// equal `sigmoid(l)` times the plain layer on `x` (the expert is nonlinear).
/// Checked end to end against llama.cpp on tiny Llama 4 GGUFs (FUTURE_WORK #114).
///
/// Deterministic, and `x` is scaled so the chosen expert's logit is 0.5
/// (`sigmoid` ≈ 0.62): with unseeded random weights a large top logit made
/// `sigmoid` ≈ 1, where the two weightings nearly coincide and the gap check
/// failed about one run in seven.
#[test]
fn a_llama4_expert_is_weighted_on_its_input() {
    let device = Device::Cpu;
    let (hidden, intermediate, n_experts) = (16usize, 24usize, 4usize);
    // A small deterministic generator: values in [-scale, scale].
    let det = |seed: u64, len: usize, scale: f32| -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let mut checked = 0;
    for seed in 1..=8u64 {
        let t = |salt: u64, dims: &[usize], scale: f32| {
            let len = dims.iter().product();
            Tensor::from_vec(det(seed * 31 + salt, len, scale), dims, &device).unwrap()
        };
        let (g, u, d) = (
            t(1, &[n_experts, intermediate, hidden], 0.5),
            t(2, &[n_experts, intermediate, hidden], 0.5),
            t(3, &[n_experts, hidden, intermediate], 0.5),
        );
        let gate = t(4, &[n_experts, hidden], 0.8);
        let moe = |routing| MoeFfn {
            gate: gate.clone(),
            experts: ExpertFfn::from_stacked(&g, &u, &d).unwrap(),
            stacked: None,
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            shared_gate_inp: None,
            n_experts_used: 1,
            routing,
        };
        let llama4 = moe(MoeRoutingConfig {
            gating_func: MoeGatingFunc::Sigmoid,
            renormalize_weights: false,
            weight_before_ffn: true,
        });
        let plain = moe(MoeRoutingConfig::default());

        let v = t(5, &[1, 1, hidden], 1.0);
        let top_of = |x: &Tensor| -> f32 {
            let logits: Vec<f32> = x
                .reshape((1, hidden))
                .unwrap()
                .matmul(&gate.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };
        let top = top_of(&v);
        if top <= 0.05 {
            continue; // scaling by a negative would change the chosen expert
        }
        let x = (&v * (0.5 / top) as f64).unwrap();
        let w = 1.0 / (1.0 + (-top_of(&x)).exp());

        let got = llama4.forward(&x).unwrap();
        let on_input = plain.forward(&(&x * w as f64).unwrap()).unwrap();
        assert_tensors_close(&got, &on_input, 1e-5, "Llama 4 = expert(sigmoid(l) * x)");
        let on_output = (plain.forward(&x).unwrap() * w as f64).unwrap();
        let gap: f32 = (&got - &on_output)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            gap > 1e-3,
            "seed {seed}: weighting the input must differ from weighting the output: {gap}"
        );
        checked += 1;
    }
    assert!(
        checked >= 3,
        "only {checked} seeds had a positive top logit"
    );
}
