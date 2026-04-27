//! Shared test fixtures used across the split-inference test modules.
//!
//! Helpers are `pub(super)` so any sibling under `tests::` can pull them in
//! via `use super::common::*;` without exposing them outside the test tree.

use super::super::model::SplitModel;
use super::super::rope::precompute_freqs_cis;
use super::super::*;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_transformers::quantized_nn::RmsNorm;

/// Append a single position to a KvCache. Helper for the truncate tests.
pub(super) fn append_pos(cache: &mut KvCache, key: f32, val: f32) {
    let k = Tensor::from_vec(vec![key, key], &[1, 1, 1, 2], &Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![val, val], &[1, 1, 1, 2], &Device::Cpu).unwrap();
    cache.append(&k, &v).unwrap();
}

pub(super) fn make_dummy_entry(vram_mb: u64) -> SplitModelEntry {
    // Construct a minimal metadata-only SplitModelEntry for eviction tests.
    SplitModelEntry {
        last_used: std::sync::atomic::AtomicU64::new(0),
        estimated_vram_mb: vram_mb,
        is_complete: false,
        eos_tokens: vec![],
        eos_token_str: String::new(),
        bos_token: String::new(),
        cached_chat_template: None,
        vocab: None,
        layer_start: 0,
        layer_end: 0,
    }
}

/// Create a minimal SplitModel on the given device. Used by benchmarks that
/// want to test GPU paths.
pub(super) fn make_test_split_model_on(
    num_layers: usize,
    hidden_dim: usize,
    device: candle_core::Device,
) -> SplitModel {
    make_test_split_model_impl(num_layers, hidden_dim, device)
}

/// Create a minimal SplitModel with real layers for testing forward/forward_batch.
pub(super) fn make_test_split_model(num_layers: usize, hidden_dim: usize) -> SplitModel {
    make_test_split_model_impl(num_layers, hidden_dim, candle_core::Device::Cpu)
}

fn make_test_split_model_impl(
    num_layers: usize,
    hidden_dim: usize,
    device: candle_core::Device,
) -> SplitModel {
    // Build a minimal model with random weights for testing.
    // Identity-like weight matrices on the caller-chosen device.
    let head_dim = 64;
    let n_head = hidden_dim / head_dim;
    let n_kv_head = n_head; // no GQA in test model

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        // Create a random weight tensor and quantize it
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let rope_dim = head_dim;
    let freq_base = 10000.0f32;
    let theta: Vec<f32> = (0..rope_dim / 2)
        .map(|i| 1.0 / freq_base.powf(i as f32 * 2.0 / rope_dim as f32))
        .collect();
    let idx: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let theta_t = Tensor::from_vec(theta.clone(), (1, rope_dim / 2), &device).unwrap();
    let idx_t = Tensor::from_vec(idx.clone(), (max_seq_len, 1), &device).unwrap();
    let freqs = idx_t.matmul(&theta_t).unwrap();
    let cos = freqs.cos().unwrap();
    let sin = freqs.sin().unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mut layers = Vec::new();
    for _ in 0..num_layers {
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
        };
        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, hidden_dim),
            attention_wv: make_qmatmul(hidden_dim, hidden_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
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
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim,
            skip_rope: false,
        }));
    }

    SplitModel {
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: num_layers,
        total_layers: num_layers + 2, // Not last segment
        hidden_dim,
        arch: ModelArch::Llama,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
        final_logit_softcap: None,
    }
}

/// Helper: create a SplitModel with explicit GQA configuration.
#[allow(clippy::too_many_arguments)]
pub(super) fn make_gqa_test_model(
    num_layers: usize,
    hidden_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    use_rope_contiguous: bool,
    activation: Activation,
    attn_logit_softcap: Option<f32>,
    arch: ModelArch,
) -> SplitModel {
    let device = candle_core::Device::Cpu;
    let head_dim = hidden_dim / n_head;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let rope_dim = head_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let kv_dim = n_kv_head * head_dim;
    let mut layers = Vec::new();
    for _ in 0..num_layers {
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
        };
        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, kv_dim),
            attention_wv: make_qmatmul(hidden_dim, kv_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(&norm_w),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: Some(make_qmatmul(hidden_dim, hidden_dim * 4)),
                ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                activation,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            post_attention_norm: None,
            post_ffw_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
            use_rope_contiguous,
            attn_logit_softcap,
            rope_dim,
            skip_rope: false,
        }));
    }

    SplitModel {
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: num_layers,
        total_layers: num_layers + 2,
        hidden_dim,
        arch,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
        final_logit_softcap: None,
    }
}

/// Helper: assert two tensors are close within tolerance.
pub(super) fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, msg: &str) {
    assert_eq!(a.shape(), b.shape(), "{msg}: shape mismatch");
    let diff = (a - b).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff < tol, "{msg}: max_diff={max_diff} >= tol={tol}");
}

/// Build a test model with DeepSeek-style mixed layers (1 dense + 1 MLA/MoE)
pub(super) fn make_deepseek_test_model(hidden_dim: usize) -> SplitModel {
    let device = Device::Cpu;
    let n_head = 4;
    let key_length = hidden_dim / n_head; // per-head key dim
    let value_length = hidden_dim / n_head;
    let kv_lora_rank = 16;
    let q_lora_rank = 16;
    let rope_dim = 8;
    let intermediate = hidden_dim * 2;
    let n_experts = 4;
    let n_experts_used = 2;

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
    let max_seq_len = 128;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    // Layer 0: Dense (like first few DeepSeek layers)
    let head_dim = hidden_dim / n_head;
    let (dense_cos, dense_sin) =
        precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
    let dense_layer = LayerVariant::Dense(LayerWeights {
        attention_wq: make_qmatmul(hidden_dim, hidden_dim),
        attention_wk: make_qmatmul(hidden_dim, hidden_dim),
        attention_wv: make_qmatmul(hidden_dim, hidden_dim),
        attention_wo: make_qmatmul(hidden_dim, hidden_dim),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(hidden_dim),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(hidden_dim, intermediate)),
            ffn_down: make_qmatmul(intermediate, hidden_dim),
            ffn_up: make_qmatmul(hidden_dim, intermediate),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(hidden_dim),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos: dense_cos,
        sin: dense_sin,
        neg_inf: neg_inf.clone(),
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: false,
    });

    // Layer 1: DeepSeek MLA + MoE
    let mla = MlaWeights {
        q_a: make_qmatmul(hidden_dim, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden_dim, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden_dim),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos: cos.clone(),
        sin: sin.clone(),
        neg_inf: neg_inf.clone(),
    };

    let moe = MoeFfn {
        gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
        gate_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
            .unwrap(),
        down_exps: Tensor::randn(0f32, 0.02, (n_experts, hidden_dim, intermediate), &device)
            .unwrap(),
        up_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device).unwrap(),
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used,
    };

    let deepseek_layer = LayerVariant::DeepSeek {
        attention: mla,
        ffn: FfnVariant::MoE(moe),
        attention_norm: make_rms_norm(hidden_dim),
        ffn_norm: make_rms_norm(hidden_dim),
    };

    SplitModel {
        tok_embeddings: None,
        layers: vec![dense_layer, deepseek_layer],
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: 2,
        total_layers: 4,
        hidden_dim,
        arch: ModelArch::DeepSeek2,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-2-4"),
        final_logit_softcap: None,
    }
}
