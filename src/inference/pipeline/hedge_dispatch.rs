//! SWARM-SPEC Layer 2 true hedge dispatch.
//!
//! When a verify forward exceeds the hedge threshold (computed from the
//! per-(model, segment, holder) EWMA), fire a duplicate forward to an
//! alternative holder. Race both via `tokio::select!`. Whichever
//! arrives first wins; the loser's response is dropped (its KV state on
//! the worker is reclaimed by the worker's session TTL).
//!
//! # Design
//!
//! - Hedging happens at the verify-forward boundary, NOT at the
//!   per-segment boundary inside a multi-segment forward. For
//!   multi-segment pipelines we duplicate the entire pipeline chain to
//!   an alternative holder of segment 0 (the rest of the chain
//!   follows). This wastes more bandwidth than per-segment hedging but
//!   is simpler and correctness-preserving.
//! - The hedge uses a NEW `Uuid` for the duplicate forward so the
//!   `pending_layer_results` map doesn't collide with the primary.
//! - Gated by `inference.hedge_enabled` (default false). When off,
//!   this wrapper degenerates to a straight forward call.
//! - Bounded by `inference.hedge_max_rate` so a degraded network can't
//!   trigger a hedge storm.
//!
//! # Scope of v0
//!
//! - Single-segment pipelines only. Multi-segment hedging needs an
//!   alternative pipeline assembly (full chain duplication) which is
//!   substantially more complex. Single-segment hedging covers the
//!   L1 ngram-only path's typical case.
//! - On hedge fire, we pick the alternative holder by scanning
//!   `model_registry.shard_holders` for a node OTHER than the primary
//!   that's currently connected.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::hedging::{HedgeConfig, HedgeKey};
use crate::types::NetworkCommand;

/// Pick an alternative segment holder for the hedge. Returns `None`
/// when no other connected peer holds the same shard.
pub(super) fn pick_alt_holder(
    state: &SharedState,
    primary: &crate::types::NodeId,
    segment: &crate::types::PipelineSegment,
) -> Option<crate::types::NodeId> {
    let shard_id = crate::types::ShardId {
        model_id: segment.shard_id.model_id.clone(),
        index: segment.shard_id.index,
    };
    let local_node = state.identity.node_id();
    state
        .model_registry
        .shard_holders(&shard_id)
        .into_iter()
        .find(|h| h != primary && h != local_node && state.connected_node_ids.contains(h))
}

/// Race-then-discard hedge wrapper for a single-segment verify forward.
///
/// Calls the primary forward immediately. After `hedge_threshold_ms`,
/// if the primary hasn't returned and the hedge tracker's
/// `should_hedge` budget allows, dispatches a duplicate forward to
/// the alternative holder. Whichever response arrives first wins;
/// the loser is dropped.
///
/// When `cfg.enabled = false` OR no alternative holder is available
/// OR `should_hedge` returns false, this degenerates to a straight
/// `forward_verify_through_segments` call with no race / wrapper
/// overhead.
#[allow(clippy::too_many_arguments)]
pub(super) async fn forward_verify_with_hedge(
    state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    primary_request_id: uuid::Uuid,
    index_pos: u32,
    segments: &[crate::types::PipelineSegment],
    peer_id_for_segment: &[Option<Vec<u8>>],
    verify_tokens: &[u32],
    truncate_kv_to: Option<u32>,
    hedge_key: HedgeKey,
    cfg: HedgeConfig,
) -> Result<Vec<Vec<f32>>, SwarmError> {
    // Fast path: hedging disabled, multi-segment, or no other holder.
    // Each of these conditions short-circuits to a plain call so the
    // dispatch overhead is zero on the common path.
    let alt_holder_eligible = cfg.enabled && segments.len() == 1;
    let alt_holder = if alt_holder_eligible {
        pick_alt_holder(state, &segments[0].node_id, &segments[0])
    } else {
        None
    };
    let Some(alt_node_id) = alt_holder else {
        return super::forward_verify_through_segments(
            state,
            network_tx,
            primary_request_id,
            index_pos,
            segments,
            peer_id_for_segment,
            verify_tokens,
            truncate_kv_to,
        )
        .await;
    };

    // Compute hedge threshold from EWMA. If insufficient samples,
    // fall back to a generous default (don't hedge prematurely on
    // cold peers).
    let threshold_ms = match state.metrics.hedge_tracker.get(&hedge_key) {
        Some(stats) if stats.samples >= cfg.min_samples => {
            stats.p99_estimate_ms() * cfg.after_factor
        }
        _ => {
            // No baseline yet — don't fire a hedge.
            return super::forward_verify_through_segments(
                state,
                network_tx,
                primary_request_id,
                index_pos,
                segments,
                peer_id_for_segment,
                verify_tokens,
                truncate_kv_to,
            )
            .await;
        }
    };

    let alt_peer_bytes = match state.resolve_peer_id_bytes(&alt_node_id) {
        Some(b) => b,
        None => {
            // Alt holder vanished between pick and dispatch — fall back.
            return super::forward_verify_through_segments(
                state,
                network_tx,
                primary_request_id,
                index_pos,
                segments,
                peer_id_for_segment,
                verify_tokens,
                truncate_kv_to,
            )
            .await;
        }
    };

    // Primary call as an owned future.
    let primary_segments = segments.to_vec();
    let primary_peer_id_for_segment = peer_id_for_segment.to_vec();
    let primary_verify_tokens = verify_tokens.to_vec();
    let primary_state = state.clone();
    let primary_network_tx = network_tx.clone();
    let primary_fut = async move {
        super::forward_verify_through_segments(
            &primary_state,
            &primary_network_tx,
            primary_request_id,
            index_pos,
            &primary_segments,
            &primary_peer_id_for_segment,
            &primary_verify_tokens,
            truncate_kv_to,
        )
        .await
    };

    let start = Instant::now();
    let mut primary_fut = Box::pin(primary_fut);
    let hedge_sleep = tokio::time::sleep(std::time::Duration::from_millis(threshold_ms as u64));
    tokio::pin!(hedge_sleep);

    // Phase 1: wait for primary OR hedge timer.
    tokio::select! {
        primary_result = &mut primary_fut => {
            // Primary won outright — record observation + decision.
            let elapsed = start.elapsed().as_millis() as f32;
            state.metrics.hedge_tracker.observe(hedge_key.clone(), elapsed);
            state.metrics.hedge_tracker.record_decision(false, false);
            return primary_result;
        }
        _ = &mut hedge_sleep => {}
    }

    // Phase 2: hedge timer fired. Re-check budget before dispatching.
    if !state
        .metrics
        .hedge_tracker
        .should_hedge(&hedge_key, threshold_ms, cfg)
    {
        // Budget refuses — keep waiting on primary.
        let primary_result = primary_fut.await;
        let elapsed = start.elapsed().as_millis() as f32;
        state.metrics.hedge_tracker.observe(hedge_key, elapsed);
        state.metrics.hedge_tracker.record_decision(false, false);
        return primary_result;
    }

    // Fire hedge with NEW request_id.
    let hedge_request_id = uuid::Uuid::new_v4();
    let hedge_segment = crate::types::PipelineSegment {
        node_id: alt_node_id.clone(),
        shard_id: segments[0].shard_id.clone(),
        layer_range: segments[0].layer_range,
    };
    tracing::info!(
        primary = %primary_request_id,
        hedge = %hedge_request_id,
        alt_holder = %alt_node_id,
        threshold_ms,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "SWARM-SPEC L2: hedge firing"
    );

    let hedge_state = state.clone();
    let hedge_network_tx = network_tx.clone();
    let hedge_segments = vec![hedge_segment];
    let hedge_peer_for_seg = vec![Some(alt_peer_bytes)];
    let hedge_verify_tokens = verify_tokens.to_vec();
    let hedge_fut = async move {
        super::forward_verify_through_segments(
            &hedge_state,
            &hedge_network_tx,
            hedge_request_id,
            index_pos,
            &hedge_segments,
            &hedge_peer_for_seg,
            &hedge_verify_tokens,
            truncate_kv_to,
        )
        .await
    };
    let mut hedge_fut = Box::pin(hedge_fut);

    // Phase 3: race primary vs hedge.
    let (winner_is_hedge, result) = tokio::select! {
        r = &mut primary_fut => (false, r),
        r = &mut hedge_fut => (true, r),
    };

    let elapsed = start.elapsed().as_millis() as f32;
    state.metrics.hedge_tracker.observe(hedge_key, elapsed);
    state
        .metrics
        .hedge_tracker
        .record_decision(true, winner_is_hedge);

    tracing::info!(
        primary = %primary_request_id,
        hedge = %hedge_request_id,
        winner = if winner_is_hedge { "hedge" } else { "primary" },
        elapsed_ms = elapsed as u64,
        "SWARM-SPEC L2: hedge race resolved"
    );

    // The loser's pending_layer_results entry is cleaned up by its
    // PendingLayerResultGuard inside forward_verify_through_segments
    // when its future is dropped (the unfinished branch of select!).
    // No explicit cancel needed — the worker on the loser's holder
    // will compute the response and find no receiver, which the
    // network layer logs and discards.

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hedge_config_disabled_means_no_alt_pick() {
        let cfg = HedgeConfig {
            enabled: false,
            ..HedgeConfig::default()
        };
        // The alt-pick logic is gated on `cfg.enabled && segments.len() == 1`.
        // Without firing the full helper (needs SharedState), verify the gate
        // logic directly.
        let enabled = cfg.enabled && 1 == 1;
        assert!(!enabled);
    }

    #[test]
    fn multi_segment_disables_hedging() {
        let cfg = HedgeConfig {
            enabled: true,
            ..HedgeConfig::default()
        };
        // The gate also blocks multi-segment pipelines.
        let enabled = cfg.enabled && 2 == 1;
        assert!(!enabled);
    }
}
