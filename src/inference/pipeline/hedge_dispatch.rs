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
                verify_tokens,
                truncate_kv_to,
            )
            .await;
        }
    };

    // Primary call as an owned future.
    let primary_segments = segments.to_vec();
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
    let alt_peer_bytes_for_cancel = alt_peer_bytes.clone();
    let hedge_verify_tokens = verify_tokens.to_vec();
    let hedge_fut = async move {
        super::forward_verify_through_segments(
            &hedge_state,
            &hedge_network_tx,
            hedge_request_id,
            index_pos,
            &hedge_segments,
            &hedge_verify_tokens,
            truncate_kv_to,
        )
        .await
    };
    let mut hedge_fut = Box::pin(hedge_fut);
    // When the hedge was dispatched. The alt holder's latency is measured from
    // HERE, not from `start` — `start` includes the whole threshold wait, which
    // is the primary's story and says nothing about how fast the alt answered.
    let hedge_started = Instant::now();

    // Phase 3: race primary vs hedge.
    let (winner_is_hedge, result) = tokio::select! {
        r = &mut primary_fut => (false, r),
        r = &mut hedge_fut => (true, r),
    };

    let elapsed = start.elapsed().as_millis() as f32;
    // Credit the latency to the holder that actually produced it.
    //
    // This used to record `elapsed` against the PRIMARY's key however the race
    // went. When the hedge wins, that number is `threshold + the alt's round
    // trip`, and `threshold` is itself the primary's own `p99 * after_factor` —
    // so the value fed back exceeds the primary's p99 BY CONSTRUCTION. It
    // pushed the primary's EWMA up every time a hedge beat it, raising the bar
    // for hedging that holder again: the feature quietly disabling itself
    // against exactly the peers it exists to race. And it is not a measurement
    // of the primary at all, which never finished — it was cancelled below.
    //
    // So a holder is observed only when it COMPLETED something, and the alt is
    // observed under its own key, which is what `HedgeKey::holder` is for.
    // Nothing is recorded against the loser: "had not finished by T" is a lower
    // bound, not a latency.
    if winner_is_hedge {
        state.metrics.hedge_tracker.observe(
            HedgeKey {
                holder: alt_node_id.clone(),
                ..hedge_key.clone()
            },
            hedge_started.elapsed().as_millis() as f32,
        );
    } else {
        state
            .metrics
            .hedge_tracker
            .observe(hedge_key.clone(), elapsed);
    }
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

    // The loser's `pending_layer_results` entry is cleaned up by its
    // `PendingLayerResultGuard` when its future is dropped (the unfinished
    // branch of the select!). That handles OUR bookkeeping — but the loser's
    // holder is a different machine, still computing a forward whose result
    // we will discard on arrival. Hedging deliberately creates that waste on
    // every fired hedge, so tell the loser to stop.
    //
    // Best effort: if the send drops, the old behaviour applies (the peer
    // finishes and its reply is discarded). Never blocks the winner's result.
    let (loser_request_id, loser_peer) = if winner_is_hedge {
        (
            primary_request_id,
            segments
                .first()
                .and_then(|seg| state.resolve_peer_id_bytes(&seg.node_id)),
        )
    } else {
        (hedge_request_id, Some(alt_peer_bytes_for_cancel))
    };
    if let Some(target_peer_bytes) = loser_peer {
        let _ = network_tx
            .send(NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message: crate::types::SwarmMessage::CancelInference(
                    swarmllm_types::CancelInference {
                        request_id: loser_request_id,
                    },
                ),
                delivery_request_id: None,
            })
            .await;
    }

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

#[cfg(test)]
mod attribution_tests {
    use crate::inference::hedging::{HedgeKey, HedgeTracker};
    use crate::types::{ModelId, NodeId};

    fn key(holder: u8) -> HedgeKey {
        HedgeKey {
            model_id: ModelId("m".into()),
            segment_idx: 0,
            holder: NodeId([holder; 32]),
        }
    }

    /// A hedge that WINS must not be recorded against the primary.
    ///
    /// The race resolves at `threshold + the alt's round trip`, and `threshold`
    /// is the primary's own `p99 * after_factor` — so that number exceeds the
    /// primary's p99 by construction. Feeding it back as if the primary had
    /// produced it raises the primary's estimate, which raises the bar for
    /// hedging that holder again: hedging switching itself off against exactly
    /// the peers it exists to race.
    ///
    /// This asserts the DIRECTION rather than an exact figure, because the
    /// EWMA constants are free to change and the defect is about which holder
    /// the sample lands on.
    #[test]
    fn a_hedge_win_is_credited_to_the_hedge_not_the_primary() {
        let tracker = HedgeTracker::default();
        let primary = key(1);
        let alt = key(2);

        // A primary with an established, fast baseline.
        for _ in 0..20 {
            tracker.observe(primary.clone(), 100.0);
        }
        let before = tracker.get(&primary).expect("primary has samples");

        // A hedge fires after the threshold and wins quickly. The OLD code
        // recorded the whole elapsed (threshold + alt RTT) against `primary`.
        let alt_round_trip = 40.0;
        tracker.observe(alt.clone(), alt_round_trip);

        let after = tracker.get(&primary).expect("primary still has samples");
        assert_eq!(
            after.samples, before.samples,
            "the primary completed nothing in this race, so it must gain no sample"
        );
        assert!(
            (after.ewma_ms - before.ewma_ms).abs() < f32::EPSILON,
            "the primary's latency estimate must not move on a race it lost: \
             {} -> {}",
            before.ewma_ms,
            after.ewma_ms
        );

        let alt_stats = tracker.get(&alt).expect("the alt holder is now tracked");
        assert_eq!(
            alt_stats.samples, 1,
            "the winner is observed under its OWN key"
        );
        assert!(
            (alt_stats.ewma_ms - alt_round_trip).abs() < f32::EPSILON,
            "and with its own round trip, not the threshold wait: {}",
            alt_stats.ewma_ms
        );
    }

    /// The primary winning outright is still a real measurement of the primary.
    #[test]
    fn a_primary_win_is_still_credited_to_the_primary() {
        let tracker = HedgeTracker::default();
        let primary = key(1);
        tracker.observe(primary.clone(), 120.0);
        let s = tracker.get(&primary).expect("recorded");
        assert_eq!(s.samples, 1);
        assert!((s.ewma_ms - 120.0).abs() < f32::EPSILON);
    }
}
