//! Llama 4 + GLM4 architectural specifics:
//! partial RoPE (GLM4-style), NoPE skip-rope, iRoPE pattern,
//! FfnVariant::MoE forward, Llama-4 MoE layer, and arch-supported sanity.

use super::super::rope::precompute_freqs_cis;
use super::super::*;
use super::common::{assert_tensors_close, make_qmatmul, make_rms_norm, make_rms_norm_dim};
use candle_core::{DType, Device, Tensor};

#[test]
fn test_glm4_arch_supported() {
    assert!(ModelArch::Glm4.is_supported());
    // llama.cpp: LLAMA_ROPE_TYPE_NORM (interleaved) — see `use_rope_contiguous`.
    assert!(!ModelArch::Glm4.use_rope_contiguous());
    assert_eq!(ModelArch::Glm4.default_activation(), Activation::SiLU);
    assert!(!ModelArch::Glm4.use_gemma_norm());
    assert_eq!(ModelArch::from_gguf_arch("glm4"), ModelArch::Glm4);
}

#[test]
fn test_llama4_arch_supported() {
    assert!(ModelArch::Llama4.is_supported());
    // llama.cpp: LLAMA_ROPE_TYPE_NORM (interleaved) — see `use_rope_contiguous`.
    assert!(!ModelArch::Llama4.use_rope_contiguous());
    assert_eq!(ModelArch::Llama4.default_activation(), Activation::SiLU);
    assert!(!ModelArch::Llama4.use_gemma_norm());
    assert_eq!(ModelArch::from_gguf_arch("llama4"), ModelArch::Llama4);
}

/// Llama 4 normalises Q and K per head by their RMS, with no weight, AFTER
/// RoPE, on the layers that rope (llama.cpp `use_kq_norm` → `ggml_rms_norm`
/// after `ggml_rope_ext`); a NoPE layer is left alone. Without it every
/// position past the first disagreed with llama.cpp (median cosine 0.60 on a
/// dense tiny Llama 4; 0.999999 with it).
#[test]
fn llama4_normalises_q_and_k_after_rope_where_rope_runs() {
    let device = Device::Cpu;
    let (head_dim, n_head, seq_len) = (16, 2, 5);
    let (cos, sin) = precompute_freqs_cis(head_dim, 500000.0, 32, &device).unwrap();
    let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
    let dim = n_head * head_dim;
    let plain = LayerWeights {
        attention_wq: make_qmatmul(dim, dim, &device),
        attention_wk: make_qmatmul(dim, dim, &device),
        attention_wv: make_qmatmul(dim, dim, &device),
        attention_wo: make_qmatmul(dim, dim, &device),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(dim, dim * 4, &device)),
            ffn_down: make_qmatmul(dim * 4, dim, &device),
            ffn_up: make_qmatmul(dim, dim * 4, &device),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos,
        sin,
        use_rope_contiguous: false,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: false,
        qk_rms_norm_after_rope: None,
    };
    let eps = 1e-5;
    let normed = LayerWeights {
        qk_rms_norm_after_rope: Some(eps),
        ..plain.clone()
    };
    let x = Tensor::randn(0f32, 3.0, (1, n_head, seq_len, head_dim), &device).unwrap();

    // RoPE first, then x / sqrt(mean(x²) + eps) over each head.
    let roped = plain.apply_rotary_emb(&x, 2).unwrap();
    let ms = roped
        .sqr()
        .unwrap()
        .mean_keepdim(candle_core::D::Minus1)
        .unwrap();
    let want = roped
        .broadcast_div(&(ms + eps).unwrap().sqrt().unwrap())
        .unwrap();
    let got = normed.apply_rotary_emb(&x, 2).unwrap();
    assert_tensors_close(&got, &want, 1e-6, "Q/K normalised after RoPE");
    let rms: Vec<f32> = got
        .sqr()
        .unwrap()
        .mean_keepdim(candle_core::D::Minus1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(rms.iter().all(|r| (r - 1.0).abs() < 1e-3), "{rms:?}");

    // A NoPE layer neither ropes nor normalises.
    let nope = LayerWeights {
        skip_rope: true,
        ..normed
    };
    assert_tensors_close(
        &nope.apply_rotary_emb(&x, 2).unwrap(),
        &x,
        1e-7,
        "NoPE untouched",
    );
}

#[test]
fn test_partial_rope_glm4_style() {
    // GLM-4 uses partial RoPE: only first half of head_dim gets rotated
    let device = Device::Cpu;
    let head_dim = 16;
    let rope_dim = 8; // half of head_dim
    let n_head = 2;
    let seq_len = 4;
    let max_seq_len = 32;

    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();

    let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();

    let lw = LayerWeights {
        attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(
                n_head * head_dim,
                n_head * head_dim * 4,
                &device,
            )),
            ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim, &device),
            ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4, &device),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos,
        sin,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim,
        skip_rope: false,
        qk_rms_norm_after_rope: None,
    };

    // Test that apply_rotary_emb handles partial RoPE
    let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
    let result = lw.apply_rotary_emb(&x, 0).unwrap();
    assert_eq!(result.dims(), &[1, n_head, seq_len, head_dim]);

    // Verify first rope_dim dims are different (rotated) and last dims unchanged
    let x_pass = x.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
    let r_pass = result.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
    let diff: f32 = (&x_pass - &r_pass)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        diff < 1e-5,
        "Non-rotated dims should be unchanged, diff={diff}"
    );
}

#[test]
fn test_nope_skip_rope() {
    // Llama 4 iRoPE: every 4th layer skips RoPE entirely
    let device = Device::Cpu;
    let head_dim = 16;
    let n_head = 2;
    let seq_len = 4;

    let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, 32, &device).unwrap();

    let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();

    let lw = LayerWeights {
        attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim, &device),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(
                n_head * head_dim,
                n_head * head_dim * 4,
                &device,
            )),
            ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim, &device),
            ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4, &device),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos,
        sin,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: true, // NoPE layer
        qk_rms_norm_after_rope: None,
    };

    let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
    let result = lw.apply_rotary_emb(&x, 0).unwrap();

    // skip_rope=true means output == input (no rotation applied)
    let diff: f32 = (&x - &result)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        diff < 1e-6,
        "NoPE layer should not modify input, diff={diff}"
    );
}

#[test]
fn test_ffn_variant_moe_forward() {
    // Test that FfnVariant::MoE dispatches through MoeFfn correctly
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 4;
    let n_experts_used = 2;

    // Build small MoE FFN
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();
    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        experts: crate::inference::layers::ExpertFfn::from_stacked(
            &(gate_exps),
            &(up_exps),
            &(down_exps),
        )
        .unwrap(),
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        shared_gate_inp: None,
        n_experts_used,
        routing: MoeRoutingConfig::default(),
    };

    let ffn = FfnVariant::MoE(moe);
    let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();

    let out = match &ffn {
        FfnVariant::Dense(mlp) => mlp.forward(&x, None).unwrap(),
        FfnVariant::MoE(moe) => moe.forward(&x).unwrap(),
    };
    assert_eq!(out.dims(), &[1, 4, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
}

#[test]
fn test_irope_nope_pattern() {
    // Verify the iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE
    let nope_layers: Vec<bool> = (0..12).map(|i| i % 4 == 3).collect();
    assert_eq!(
        nope_layers,
        vec![false, false, false, true, false, false, false, true, false, false, false, true]
    );
}

#[test]
fn test_llama4_moe_layer_forward() {
    // End-to-end test: model with MoE FFN layers via FfnVariant
    let device = Device::Cpu;
    let hidden_dim = 32;
    let n_head = 4;
    let head_dim = hidden_dim / n_head;
    let n_kv_head = 2;
    let n_experts = 4;
    let n_experts_used = 2;
    let intermediate = 64;
    let max_seq_len = 32;
    let rope_dim = head_dim;

    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();

    let kv_dim = n_kv_head * head_dim;

    // Build a mix of dense + MoE layers (like Llama 4)
    let mut layers = Vec::new();
    for layer_idx in 0..4 {
        let is_nope = layer_idx % 4 == 3;

        let ffn = if layer_idx % 2 == 1 {
            // MoE layers on odd indices
            FfnVariant::MoE(MoeFfn {
                gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
                experts: crate::inference::layers::ExpertFfn::from_stacked(
                    &(Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
                        .unwrap()),
                    &(Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
                        .unwrap()),
                    &(Tensor::randn(0f32, 0.02, (n_experts, hidden_dim, intermediate), &device)
                        .unwrap()),
                )
                .unwrap(),
                shared_gate: None,
                shared_down: None,
                shared_up: None,
                shared_gate_inp: None,
                n_experts_used,
                routing: MoeRoutingConfig::default(),
            })
        } else {
            // Dense FFN on even indices
            FfnVariant::Dense(Mlp {
                ffn_gate: Some(make_qmatmul(hidden_dim, intermediate, &device)),
                ffn_down: make_qmatmul(intermediate, hidden_dim, &device),
                ffn_up: make_qmatmul(hidden_dim, intermediate, &device),
                activation: Activation::SiLU,
            })
        };

        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim, &device),
            attention_wk: make_qmatmul(hidden_dim, kv_dim, &device),
            attention_wv: make_qmatmul(hidden_dim, kv_dim, &device),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim, &device),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm_dim(hidden_dim, &device),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn,
            ffn_norm: make_rms_norm_dim(hidden_dim, &device),
            post_attention_norm: None,
            post_ffw_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim,
            skip_rope: is_nope,
            qk_rms_norm_after_rope: None,
        }));
    }

    let mut model = SplitModel {
        // Single-device test model: empty means "not split".
        layer_devices: Vec::new(),
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        kv_budget_bytes: None,
        kv_bytes_per_token: 0,
        layer_start: 0,
        layer_end: 4,
        total_layers: 8,
        hidden_dim,
        arch: ModelArch::Llama4,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-4-8"),
        final_logit_softcap: None,
        batch_calls: 0,
        batch_fellback: 0,
        batch_stats_reported_at: None,
    };

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let output = model.forward(&input, 0, &kv_store, "llama4-test").unwrap();
    assert_eq!(output.dims(), &[1, 4, hidden_dim]);
    let flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Llama 4 MoE output NaN/Inf"
    );
}
