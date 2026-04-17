//! Probabilistic output-validation for distributed inference. Only invoked
//! after a successful remote pipeline execution; on failure, offending peers
//! have their trust reduced.

use std::sync::Arc;

use crate::daemon::SharedState;
use crate::types::{InferenceRequest, NodeId, PipelineAssignment};

use super::types::InferenceOutput;

/// Probabilistic spot-check of distributed inference results.
///
/// After a successful distributed inference, randomly selects remote peers
/// (based on AntiGaming spot-check rate) and validates the output is plausible.
/// On failure, reduces trust for the offending peer.
pub(super) async fn spot_check_distributed_result(
    shared_state: &Arc<SharedState>,
    request: &InferenceRequest,
    assignment: &PipelineAssignment,
    local_node_id: &NodeId,
    output: &InferenceOutput,
) {
    // Only spot-check if there are remote peers in the pipeline
    let remote_peers: Vec<NodeId> = assignment
        .segments
        .iter()
        .filter(|s| s.node_id != *local_node_id)
        .map(|s| s.node_id.clone())
        .collect();
    if remote_peers.is_empty() {
        return;
    }

    // Ask anti-gaming whether this request should be spot-checked
    let should_check = {
        let ag = shared_state.credits.anti_gaming.lock().await;
        let rate = ag.effective_spot_check_rate(&remote_peers[0]);
        rand::random::<f64>() < rate
    };

    if !should_check {
        return;
    }

    tracing::info!(
        request_id = %request.id,
        remote_peers = remote_peers.len(),
        "Spot-check: verifying distributed inference result"
    );

    // Validation 1: Check output is non-empty and reasonable
    let text = &output.content;
    if text.is_empty() && output.completion_tokens > 0 {
        tracing::warn!(
            request_id = %request.id,
            "Spot-check FAIL: empty text with non-zero completion_tokens"
        );
        for peer in &remote_peers {
            report_spot_check_failure(shared_state, peer).await;
        }
        return;
    }

    // Validation 2: Check for garbage output (all same char, all whitespace for long outputs)
    if output.completion_tokens > 10 {
        let chars: Vec<char> = text.chars().collect();
        if !chars.is_empty() {
            let first = chars[0];
            if chars.iter().all(|&c| c == first) {
                tracing::warn!(
                    request_id = %request.id,
                    repeated_char = ?first,
                    "Spot-check FAIL: output is all repeated characters"
                );
                for peer in &remote_peers {
                    report_spot_check_failure(shared_state, peer).await;
                }
                return;
            }
        }
    }

    // Validation 3: Token count consistency
    if output.completion_tokens == 0 && !text.is_empty() {
        tracing::warn!(
            request_id = %request.id,
            text_len = text.len(),
            "Spot-check FAIL: non-empty text but zero completion_tokens"
        );
        for peer in &remote_peers {
            report_spot_check_failure(shared_state, peer).await;
        }
        return;
    }

    tracing::debug!(
        request_id = %request.id,
        "Spot-check PASS: output appears valid"
    );
}

/// Report a spot-check failure for a peer: reduce trust and log the penalty.
async fn report_spot_check_failure(shared_state: &Arc<SharedState>, peer: &NodeId) {
    // Update trust score
    shared_state.credits.trust_manager.update_trust(
        &shared_state.peer_registry,
        peer,
        crate::credit::trust::TrustEvent::SpotCheckFail,
    );

    // Report to anti-gaming
    let penalty = {
        let mut ag = shared_state.credits.anti_gaming.lock().await;
        ag.report_spot_check_failure(peer)
    };

    tracing::warn!(
        peer = %peer,
        penalty = ?penalty,
        "Spot-check failure reported — trust reduced"
    );
}
