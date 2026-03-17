use crate::types::{ModelId, ModelManifest};

/// State for an in-flight speculative decoding session.
///
/// Speculative decoding uses a small "draft" model to propose multiple tokens
/// at once, then verifies them in parallel with the larger "target" model.
/// This can significantly improve tokens/sec when the draft model has a high
/// acceptance rate.
#[derive(Debug, Clone)]
pub struct SpeculativeDraftState {
    pub session_id: uuid::Uuid,
    pub draft_model_id: ModelId,
    pub verify_model_id: ModelId,
    pub draft_tokens: Vec<u32>,
    pub accepted_count: u32,
    pub total_proposed: u32,
    /// Number of draft tokens to propose per verification step.
    pub gamma: u32,
}

impl SpeculativeDraftState {
    pub fn new(draft_model_id: ModelId, verify_model_id: ModelId, gamma: u32) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4(),
            draft_model_id,
            verify_model_id,
            draft_tokens: Vec::new(),
            accepted_count: 0,
            total_proposed: 0,
            gamma,
        }
    }

    /// Calculate the acceptance rate for monitoring.
    ///
    /// Returns a value in `[0.0, 1.0]` representing the fraction of
    /// draft tokens accepted by the verifier.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.accepted_count as f32 / self.total_proposed as f32
    }

    /// Record a batch of draft tokens and how many were accepted.
    pub fn record_batch(&mut self, proposed: u32, accepted: u32) {
        self.total_proposed += proposed;
        self.accepted_count += accepted;
        tracing::debug!(
            drafted_count = proposed,
            accepted_count = accepted,
            acceptance_rate = format!("{:.2}", self.acceptance_rate()),
            "DIAG: speculative batch"
        );
    }
}

/// Check if a model pair is eligible for speculative decoding.
///
/// The draft model must be at most 1/10th the parameter count of the target
/// to ensure the draft model is fast enough to provide a speedup.
pub fn is_valid_draft_pair(draft: &ModelManifest, target: &ModelManifest) -> bool {
    draft.num_params_billions * 10.0 <= target.num_params_billions
}

/// Result of the speculative accept/reject step.
///
/// After the draft model proposes `gamma` tokens and the target model verifies
/// them in a single forward pass, this struct holds the accepted tokens plus
/// an optional bonus token sampled from the target model at the rejection point.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeResult {
    /// Tokens accepted from the draft (may be 0..gamma).
    pub accepted_tokens: Vec<u32>,
    /// A bonus token sampled from the target model's distribution.
    /// This is always produced: either from the adjusted distribution at the
    /// rejection point, or from the target's distribution after the last
    /// accepted draft token.
    pub bonus_token: Option<u32>,
}

/// Convert raw logits to a probability distribution via softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Uniform fallback to avoid division by zero
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Sample a token from a probability distribution.
///
/// Returns the index of the sampled token.
fn sample_from_probs(probs: &[f32]) -> u32 {
    let r: f32 = rand::random::<f32>();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Core speculative decoding accept/reject algorithm.
///
/// Given draft token proposals with their probability distributions and the
/// target model's probability distributions for the same positions, determine
/// which draft tokens to accept using rejection sampling.
///
/// The algorithm guarantees that the output distribution is identical to sampling
/// from the target model alone — the draft model only provides a speedup, never
/// changes the output distribution.
///
/// # Arguments
/// * `draft_tokens` - Token IDs proposed by the draft model
/// * `draft_probs` - Per-position probability distributions from the draft model.
///   `draft_probs[i]` is the distribution the draft model used to sample `draft_tokens[i]`.
/// * `target_probs` - Per-position probability distributions from the target model.
///   `target_probs[i]` is the target model's distribution at position i (after seeing
///   the prompt + accepted tokens up to position i).
///
/// # Returns
/// A `SpeculativeResult` with accepted tokens and an optional bonus token.
pub fn accept_reject(
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
) -> Result<SpeculativeResult, crate::error::SwarmError> {
    if draft_tokens.len() != draft_probs.len() {
        return Err(crate::error::SwarmError::Internal(format!(
            "draft_tokens.len()={} != draft_probs.len()={}",
            draft_tokens.len(),
            draft_probs.len()
        )));
    }
    if target_probs.len() != draft_probs.len() + 1 {
        return Err(crate::error::SwarmError::Internal(format!(
            "target_probs.len()={} != draft_probs.len()+1={}",
            target_probs.len(),
            draft_probs.len() + 1
        )));
    }

    let mut accepted_tokens = Vec::new();

    for (i, &token) in draft_tokens.iter().enumerate() {
        let token_idx = token as usize;

        // Get probabilities for this token from both models
        let p_target = target_probs[i].get(token_idx).copied().unwrap_or(0.0);
        let p_draft = draft_probs[i].get(token_idx).copied().unwrap_or(0.0);

        // Acceptance criterion: accept with probability min(1, p_target / p_draft)
        if p_draft <= 0.0 {
            // Draft assigned zero probability — reject immediately.
            // Sample from the target model's distribution at this position.
            let bonus = sample_from_probs(&target_probs[i]);
            return Ok(SpeculativeResult {
                accepted_tokens,
                bonus_token: Some(bonus),
            });
        }

        let accept_prob = (p_target / p_draft).min(1.0);
        let r: f32 = rand::random::<f32>();

        if r < accept_prob {
            // Accept this draft token
            accepted_tokens.push(token);
        } else {
            // Reject: sample from the adjusted distribution
            // adjusted[t] = max(0, p_target[t] - p_draft[t]) / Z
            let adjusted = compute_adjusted_distribution(&target_probs[i], &draft_probs[i]);
            let bonus = sample_from_probs(&adjusted);
            return Ok(SpeculativeResult {
                accepted_tokens,
                bonus_token: Some(bonus),
            });
        }
    }

    // All draft tokens were accepted! Sample a bonus token from the target's
    // distribution at position gamma (the last entry in target_probs).
    let Some(last_probs) = target_probs.last() else {
        return Ok(SpeculativeResult {
            accepted_tokens,
            bonus_token: None,
        });
    };
    let bonus = sample_from_probs(last_probs);
    Ok(SpeculativeResult {
        accepted_tokens,
        bonus_token: Some(bonus),
    })
}

/// Compute the adjusted distribution for rejection sampling.
///
/// `adjusted[t] = max(0, p_target[t] - p_draft[t])`, then renormalized.
/// This ensures the combined accept/reject process samples from the exact
/// target distribution.
fn compute_adjusted_distribution(target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    let len = target_probs.len().max(draft_probs.len());
    let mut adjusted = Vec::with_capacity(len);

    for i in 0..len {
        let pt = target_probs.get(i).copied().unwrap_or(0.0);
        let pd = draft_probs.get(i).copied().unwrap_or(0.0);
        adjusted.push((pt - pd).max(0.0));
    }

    // Renormalize
    let sum: f32 = adjusted.iter().sum();
    if sum > 0.0 {
        for v in &mut adjusted {
            *v /= sum;
        }
    } else {
        // Fallback to target distribution (can happen when draft >= target everywhere)
        adjusted.clear();
        adjusted.extend_from_slice(target_probs);
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            for v in &mut adjusted {
                *v /= sum;
            }
        }
    }

    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_manifest(id: &str, params_b: f32) -> ModelManifest {
        ModelManifest {
            schema_version: 2,
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
            mmproj: None,
        }
    }

    #[test]
    fn valid_draft_pair() {
        let draft = make_manifest("small", 3.0);
        let target = make_manifest("large", 70.0);
        assert!(is_valid_draft_pair(&draft, &target));
    }

    #[test]
    fn invalid_draft_pair_too_large() {
        let draft = make_manifest("medium", 13.0);
        let target = make_manifest("large", 70.0);
        // 13 * 10 = 130 > 70
        assert!(!is_valid_draft_pair(&draft, &target));
    }

    #[test]
    fn acceptance_rate_empty() {
        let state =
            SpeculativeDraftState::new(ModelId("draft".into()), ModelId("target".into()), 4);
        assert_eq!(state.acceptance_rate(), 0.0);
    }

    #[test]
    fn acceptance_rate_tracking() {
        let mut state =
            SpeculativeDraftState::new(ModelId("draft".into()), ModelId("target".into()), 4);
        state.record_batch(4, 3); // 75% acceptance
        assert!((state.acceptance_rate() - 0.75).abs() < f32::EPSILON);

        state.record_batch(4, 2); // 50% acceptance this batch
                                  // Cumulative: 5/8 = 62.5%
        assert!((state.acceptance_rate() - 0.625).abs() < f32::EPSILON);
    }

    #[test]
    fn softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn softmax_uniform() {
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn accept_reject_all_identical_distributions() {
        // When draft and target have the same distribution, all tokens should be accepted
        // (acceptance probability = p_target/p_draft = 1.0 for all tokens).
        let draft_tokens = vec![0, 1, 2];
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let draft_probs = vec![uniform.clone(), uniform.clone(), uniform.clone()];
        let target_probs = vec![
            uniform.clone(),
            uniform.clone(),
            uniform.clone(),
            uniform.clone(), // bonus position
        ];

        let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();

        // All tokens should be accepted since distributions are identical
        assert_eq!(result.accepted_tokens, vec![0, 1, 2]);
        assert!(result.bonus_token.is_some());
    }

    #[test]
    fn accept_reject_draft_zero_prob_rejected() {
        // If draft assigned 0 probability to a token it sampled (impossible in
        // practice but tests the guard), it should be rejected immediately.
        let draft_tokens = vec![2];
        // Draft says token 2 has 0 probability (contradicts sampling it)
        let draft_probs = vec![vec![0.5, 0.5, 0.0, 0.0]];
        let target_probs = vec![vec![0.25, 0.25, 0.25, 0.25], vec![0.25, 0.25, 0.25, 0.25]];

        let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();
        // Token should be rejected since draft prob is 0
        assert!(result.accepted_tokens.is_empty());
        assert!(result.bonus_token.is_some());
    }

    #[test]
    fn accept_reject_target_prefers_different_token() {
        // Target strongly prefers token 0 but draft proposed token 3.
        // Acceptance probability = p_target[3]/p_draft[3] should be very low.
        let draft_tokens = vec![3];
        let draft_probs = vec![vec![0.1, 0.1, 0.1, 0.7]]; // draft likes 3
        let target_probs = vec![
            vec![0.9, 0.05, 0.04, 0.01],  // target likes 0
            vec![0.25, 0.25, 0.25, 0.25], // bonus
        ];

        // Run many times to check statistical behavior
        let mut rejection_count = 0;
        for _ in 0..100 {
            let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();
            if result.accepted_tokens.is_empty() {
                rejection_count += 1;
            }
        }
        // p_target[3]/p_draft[3] = 0.01/0.7 ≈ 0.014, so ~98.6% rejection rate
        assert!(
            rejection_count > 80,
            "Expected high rejection rate, got {rejection_count}/100"
        );
    }

    #[test]
    fn adjusted_distribution_sums_to_one() {
        let target = vec![0.5, 0.3, 0.2];
        let draft = vec![0.1, 0.6, 0.3];
        let adjusted = compute_adjusted_distribution(&target, &draft);
        let sum: f32 = adjusted.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "adjusted distribution should sum to 1.0, got {sum}"
        );
        // target[0] - draft[0] = 0.4 > 0 → kept
        assert!(adjusted[0] > 0.0);
        // target[1] - draft[1] = -0.3 < 0 → clamped to 0
        assert!(adjusted[1] == 0.0);
    }

    #[test]
    fn speculative_result_always_has_bonus() {
        // Even when all tokens accepted, we should get a bonus token
        let draft_tokens = vec![1];
        let draft_probs = vec![vec![0.0, 1.0]];
        let target_probs = vec![
            vec![0.0, 1.0],
            vec![0.5, 0.5], // bonus
        ];
        let result = accept_reject(&draft_tokens, &draft_probs, &target_probs).unwrap();
        assert_eq!(result.accepted_tokens, vec![1]);
        assert!(result.bonus_token.is_some());
    }
}
