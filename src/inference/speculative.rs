use crate::types::{ModelId, ModelManifest};

// TODO: This module contains data structures and validation only.
// Speculative decoding is not yet wired into the inference pipeline.
// Integration requires: draft model selection in scheduler, parallel
// draft/verify forward passes, and acceptance sampling in the token loop.

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
    }
}

/// Check if a model pair is eligible for speculative decoding.
///
/// The draft model must be at most 1/10th the parameter count of the target
/// to ensure the draft model is fast enough to provide a speedup.
pub fn is_valid_draft_pair(draft: &ModelManifest, target: &ModelManifest) -> bool {
    draft.num_params_billions * 10.0 <= target.num_params_billions
}

/// A registered draft-target model pair available for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativePair {
    pub draft_model_id: ModelId,
    pub target_model_id: ModelId,
    pub gamma: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

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
}
