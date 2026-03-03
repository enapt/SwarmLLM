//! Integration tests for speculative decoding and batched inference features (Phase 11).
//!
//! Tests speculative decoding correctness (accept/reject algorithm preserves
//! target distribution) and batch inference tracking.

use swarmllm::inference::speculative::{
    accept_reject, is_valid_draft_pair, softmax, SpeculativeDraftState,
};
use swarmllm::types::{ModelArchitecture, ModelId, ModelManifest, NodeId, Quantization};

fn make_manifest(id: &str, params_b: f32) -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        id: ModelId(id.to_string()),
        name: id.to_string(),
        architecture: ModelArchitecture::Llama,
        num_layers: 32,
        num_params_billions: params_b,
        quantization: Quantization::Q4KM,
        total_size_bytes: (params_b * 1e9) as u64,
        shard_count: 1,
        shards: vec![],
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "MIT".into(),
    }
}

/// Test speculative decoding correctness: when draft and target have identical
/// distributions, all tokens should be accepted (acceptance probability = 1.0).
/// This verifies the core invariant that speculative decoding never degrades
/// output quality when the draft model matches the target.
#[test]
fn test_speculative_decoding_identical_distributions_accept_all() {
    let draft_tokens = vec![0, 1, 2, 3];
    let uniform = vec![0.25, 0.25, 0.25, 0.25];
    let draft_probs = vec![uniform.clone(); 4];
    let target_probs = vec![uniform.clone(); 5]; // gamma + 1 entries

    let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();

    // All tokens accepted since distributions are identical
    assert_eq!(result.accepted_tokens, vec![0, 1, 2, 3]);
    assert!(result.bonus_token.is_some());
}

/// Test that when the draft model proposes a zero-probability token,
/// it is immediately rejected (guard against impossible draft samples).
#[test]
fn test_speculative_decoding_zero_prob_rejection() {
    let draft_tokens = vec![2]; // Draft proposed token 2
    let draft_probs = vec![vec![0.5, 0.5, 0.0, 0.0]]; // But assigned it 0 probability
    let target_probs = vec![
        vec![0.25, 0.25, 0.25, 0.25],
        vec![0.25, 0.25, 0.25, 0.25], // bonus
    ];

    let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();

    // Token should be rejected since draft prob is 0
    assert!(result.accepted_tokens.is_empty());
    assert!(result.bonus_token.is_some());
}

/// Test statistical behavior: when the target strongly disagrees with the draft,
/// most proposals should be rejected. Run many trials to verify the rejection
/// rate matches the theoretical expectation.
#[test]
fn test_speculative_decoding_high_rejection_rate() {
    let draft_tokens = vec![3]; // Draft likes token 3
    let draft_probs = vec![vec![0.1, 0.1, 0.1, 0.7]]; // p_draft[3] = 0.7
    let target_probs = vec![
        vec![0.9, 0.05, 0.04, 0.01],  // p_target[3] = 0.01 — target disagrees
        vec![0.25, 0.25, 0.25, 0.25], // bonus
    ];

    let mut rejection_count = 0;
    for _ in 0..200 {
        let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();
        if result.accepted_tokens.is_empty() {
            rejection_count += 1;
        }
    }

    // p_target[3]/p_draft[3] = 0.01/0.7 ~ 0.014 acceptance rate
    // So ~97% should be rejected
    assert!(
        rejection_count > 170,
        "Expected high rejection rate, got {rejection_count}/200 rejections"
    );
}

/// Test that the speculative result always includes a bonus token,
/// even when all draft tokens are accepted.
#[test]
fn test_speculative_decoding_always_has_bonus_token() {
    let draft_tokens = vec![1];
    let draft_probs = vec![vec![0.0, 1.0]]; // Deterministic draft
    let target_probs = vec![
        vec![0.0, 1.0], // Target agrees
        vec![0.5, 0.5], // Bonus position
    ];

    let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();
    assert_eq!(result.accepted_tokens, vec![1]);
    assert!(
        result.bonus_token.is_some(),
        "Should always produce a bonus token"
    );
}

/// Test draft-target pair validation: the draft model must be at most
/// 1/10th the parameter count of the target.
#[test]
fn test_valid_draft_pair_size_constraint() {
    let small = make_manifest("tinyllama", 1.1);
    let medium = make_manifest("llama-13b", 13.0);
    let large = make_manifest("llama-70b", 70.0);

    // 1.1B draft with 70B target: 1.1 * 10 = 11 <= 70 ✓
    assert!(is_valid_draft_pair(&small, &large));

    // 13B draft with 70B target: 13 * 10 = 130 > 70 ✗
    assert!(!is_valid_draft_pair(&medium, &large));

    // 1.1B draft with 13B target: 1.1 * 10 = 11 <= 13 ✓
    assert!(is_valid_draft_pair(&small, &medium));
}

/// Test that the speculative draft state correctly tracks acceptance rates
/// across multiple batches (for monitoring and tuning gamma).
#[test]
fn test_speculative_acceptance_rate_tracking() {
    let mut state = SpeculativeDraftState::new(
        ModelId("draft".into()),
        ModelId("target".into()),
        4, // gamma = 4
    );

    // Initially no proposals — rate is 0
    assert_eq!(state.acceptance_rate(), 0.0);

    // First batch: proposed 4 tokens, 3 accepted
    state.record_batch(4, 3);
    assert!((state.acceptance_rate() - 0.75).abs() < f32::EPSILON);

    // Second batch: proposed 4 tokens, 1 accepted
    state.record_batch(4, 1);
    // Cumulative: 4/8 = 0.5
    assert!((state.acceptance_rate() - 0.5).abs() < f32::EPSILON);

    // Third batch: proposed 4 tokens, 4 accepted (100%)
    state.record_batch(4, 4);
    // Cumulative: 8/12 = 0.667
    assert!((state.acceptance_rate() - 0.6667).abs() < 0.01);
}

/// Test softmax numerical stability with extreme logit values.
#[test]
fn test_softmax_numerical_stability() {
    // Very large logits — should not overflow
    let large_logits = vec![1000.0, 1001.0, 999.0];
    let probs = softmax(&large_logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
    assert!(probs[1] > probs[0]); // 1001 > 1000
    assert!(probs[0] > probs[2]); // 1000 > 999

    // Very negative logits — should not underflow to all zeros
    let neg_logits = vec![-1000.0, -999.0, -1001.0];
    let probs = softmax(&neg_logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");

    // Empty logits
    assert!(softmax(&[]).is_empty());
}
