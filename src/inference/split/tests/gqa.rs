//! GQA (Grouped-Query Attention) verification tests:
//! Llama 3 / MQA / Qwen 2 ratios, flash-vs-standard equivalence,
//! KV-cache shapes, multi-step decode, and the Qwen-2-with-bias path.

use super::super::super::layers::{run_attention, standard_attention};
use super::super::model::SplitModel;
use super::super::rope::precompute_freqs_cis;
use super::super::*;
use super::common::*;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_transformers::quantized_nn::RmsNorm;

#[test]
fn gqa_standard_attention_llama3_ratio() {
    // Llama 3 8B: GQA ratio=4 (scaled: n_head=8, n_kv_head=2)
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 12, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();

    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    let out =
        standard_attention(&q, &k, &v, Some(&mask), head_dim, n_head, n_kv_head, None).unwrap();
    assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Output contains NaN/Inf"
    );
}

#[test]
fn gqa_standard_attention_mqa_ratio() {
    // Multi-Query Attention: n_kv_head=1 (extreme GQA)
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 1, 6, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();

    let out = standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None).unwrap();
    assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
}

#[test]
fn gqa_flash_vs_standard_llama3_prefill() {
    // CPU flash vs standard with GQA ratio=4, causal mask
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 10, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();

    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    let out_std =
        standard_attention(&q, &k, &v, Some(&mask), head_dim, n_head, n_kv_head, None).unwrap();
    let out_flash =
        run_attention(&q, &k, &v, Some(&mask), n_head, n_kv_head, head_dim, None).unwrap();
    assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA ratio=4 flash vs standard");
}

#[test]
fn gqa_flash_vs_standard_llama3_decode() {
    // Decode step (seq_len=1 Q, longer KV) with GQA ratio=4
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, head_dim, kv_len) = (1, 8, 2, 32, 20);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, 1, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();

    let out_std = standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None).unwrap();
    let out_flash = run_attention(&q, &k, &v, None, n_head, n_kv_head, head_dim, None).unwrap();
    assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA decode flash vs standard");
}

#[test]
fn gqa_forward_llama3_style() {
    // End-to-end forward with Llama 3-style GQA
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

    // Prefill
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "llama3").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 6, &kv_store, "llama3").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

#[test]
fn gqa_forward_mistral_style() {
    // Mistral 7B: same GQA as Llama 3
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Mistral,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "mistral").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn gqa_forward_phi3_mha_style() {
    // Phi-3-mini: MHA (n_head == n_kv_head)
    let (hidden_dim, n_head, n_kv_head) = (192, 6, 6);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Phi3,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "phi3").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
}

// ── Qwen 2 GQA + bias path ──

#[test]
fn qwen2_forward_with_biases() {
    // Qwen2: GQA + contiguous RoPE + QKV biases
    let device = Device::Cpu;
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let head_dim = hidden_dim / n_head;
    let kv_dim = n_kv_head * head_dim;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
    let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
    let make_rms_norm = |w: &Tensor| {
        let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let layer = LayerWeights {
        attention_wq: make_qmatmul(hidden_dim, hidden_dim),
        attention_wk: make_qmatmul(hidden_dim, kv_dim),
        attention_wv: make_qmatmul(hidden_dim, kv_dim),
        attention_wo: make_qmatmul(hidden_dim, hidden_dim),
        attention_bq: Some(Tensor::randn(0f32, 0.01, (hidden_dim,), &device).unwrap()),
        attention_bk: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
        attention_bv: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(hidden_dim, hidden_dim * 4)),
            ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
            ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head,
        head_dim,
        cos,
        sin,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: false,
    };

    let mut model = SplitModel {
        tok_embeddings: None,
        layers: vec![LayerVariant::Dense(layer)],
        norm: None,
        output: None,
        masks: None,
        kv_budget_bytes: None,
        kv_bytes_per_token: 0,
        layer_start: 0,
        layer_end: 1,
        total_layers: 3,
        hidden_dim,
        arch: ModelArch::Qwen2,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-1-3"),
        final_logit_softcap: None,
        batch_calls: 0,
        batch_fellback: 0,
        batch_stats_reported_at: None,
    };

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "qwen2").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn gqa_kv_cache_dimensions() {
    // KV-cache stores with n_kv_head (not n_head)
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let head_dim = hidden_dim / n_head;
    let mut model = make_gqa_test_model(
        1,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let seq_len = 5;
    let input = Tensor::randn(0f32, 1.0, (1, seq_len, hidden_dim), &Device::Cpu).unwrap();
    model.forward(&input, 0, &kv_store, "cache-test").unwrap();

    let model_key = format!(
        "{}-{}-{}",
        model.layer_start, model.layer_end, model.total_layers
    );
    let entry = kv_store.get_or_create(&model_key, "cache-test", 1);
    let k = entry.layers[0].as_ref().unwrap().k().unwrap().unwrap();
    assert_eq!(
        k.dims(),
        &[1, n_kv_head, seq_len, head_dim],
        "KV cache should have n_kv_head={n_kv_head}, not n_head={n_head}"
    );
}

#[test]
fn gqa_multiple_decode_steps() {
    // Multiple decode steps with GQA
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

    let input = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    model.forward(&input, 0, &kv_store, "multi-decode").unwrap();

    for step in 0..10 {
        let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model
            .forward(&decode, 4 + step, &kv_store, "multi-decode")
            .unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "Step {step} NaN/Inf");
    }
}
