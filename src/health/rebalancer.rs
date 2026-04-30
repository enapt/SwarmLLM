use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::model::acquisition::AcquisitionCommand;
use crate::types::{NetworkCommand, NodeId, RebalanceEvent, ShardId, SwarmMessage};

/// Minimum replication factor for each shard.
const MIN_REPLICATION: usize = 2;

/// Minimum time between rebalance operations for the same shard (5 minutes).
/// Prevents thundering herd when multiple peers leave simultaneously.
const REBALANCE_COOLDOWN_SECS: u64 = 300;

/// Manages shard rebalancing in response to network topology changes.
///
/// When the HealthMonitor detects a peer leaving or joining, it signals
/// the ShardRebalancer via the `RebalanceEvent` channel. The rebalancer
/// checks for under-replicated shards and triggers re-downloads.
pub struct ShardRebalancer {
    shared_state: Arc<SharedState>,
    rebalance_rx: mpsc::Receiver<RebalanceEvent>,
    network_tx: mpsc::Sender<NetworkCommand>,
    acquisition_tx: mpsc::Sender<AcquisitionCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Per-model cooldown tracking (DAE-I10).
    last_rebalance_per_model: std::collections::HashMap<crate::types::ModelId, Instant>,
    /// Queued PeerLeft events to batch-process after cooldown.
    pending_peer_left: Vec<NodeId>,
}

impl ShardRebalancer {
    pub fn new(
        shared_state: Arc<SharedState>,
        rebalance_rx: mpsc::Receiver<RebalanceEvent>,
        network_tx: mpsc::Sender<NetworkCommand>,
        acquisition_tx: mpsc::Sender<AcquisitionCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            shared_state,
            rebalance_rx,
            network_tx,
            acquisition_tx,
            shutdown_rx,
            last_rebalance_per_model: std::collections::HashMap::new(),
            pending_peer_left: Vec::new(),
        }
    }

    /// Run the rebalancer event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        tracing::info!(target: "swarmllm::health::rebalancer", "ShardRebalancer running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!(target: "swarmllm::health::rebalancer", "ShardRebalancer shutting down");
                        break;
                    }
                }
                event = self.rebalance_rx.recv() => {
                    match event {
                        Some(event) => self.handle_event(event).await,
                        None => {
                            tracing::info!(subsystem = "rebalancer", "Rebalance channel closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_event(&mut self, event: RebalanceEvent) {
        tracing::debug!(event = ?std::mem::discriminant(&event), "DIAG: rebalancer event received");
        match event {
            RebalanceEvent::PeerLeft(departed_peer) => {
                tracing::info!(
                    peer = %departed_peer,
                    "Peer departed, queuing for batch rebalance"
                );
                self.shared_state.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "network",
                        "rebalance_peer_left",
                        format!(
                            "Rebalancing: peer {} departed",
                            &format!("{}", departed_peer)[..8]
                        ),
                    )
                    .with_node(format!("{}", departed_peer))
                    .with_detail_str("peer_left".to_string()),
                );
                // Item 8 Phase 1: drop the departed peer's prefix-cache index
                // entries so Phase 2's KV-fetch path never tries to dial them.
                let dropped = self
                    .shared_state
                    .models
                    .forget_peer_prefix_blocks(&departed_peer);
                if dropped > 0 {
                    tracing::debug!(
                        peer = %departed_peer,
                        dropped,
                        "DIAG: cleared prefix-cache index entries for departed peer"
                    );
                }
                self.pending_peer_left.push(departed_peer);
                self.process_pending_departures().await;
            }
        }
    }

    /// Batch-process queued PeerLeft events, respecting per-model cooldown.
    async fn process_pending_departures(&mut self) {
        let departed: Vec<NodeId> = self.pending_peer_left.drain(..).collect();
        if departed.is_empty() {
            return;
        }

        // Prune stale cooldown entries (models not seen in 2x cooldown window)
        self.last_rebalance_per_model
            .retain(|_, instant| instant.elapsed().as_secs() < REBALANCE_COOLDOWN_SECS * 2);

        let local_node_id = self.shared_state.identity.node_id().clone();
        let now = Instant::now();

        for departed_peer in &departed {
            let underreplicated = self.find_underreplicated_shards(departed_peer);
            for (shard_id, holders) in &underreplicated {
                if holders.contains(&local_node_id) {
                    // We hold this shard — re-announce it so peers know
                    // it's still available. Re-announces are cheap
                    // (single GossipSub message, no DB or network fetch),
                    // so they don't need the per-model cooldown — under
                    // a thundering-herd departure, suppressing them
                    // leaves locally-held shards undiscoverable for up
                    // to REBALANCE_COOLDOWN_SECS while peers retry HF.
                    let announce = crate::types::ShardAnnounce {
                        node_id: local_node_id.clone(),
                        shards: vec![shard_id.clone()],
                        timestamp: chrono::Utc::now(),
                    };
                    let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
                    if let Err(e) = self.network_tx.send(msg).await {
                        tracing::warn!(error = %e, "Failed to broadcast shard rebalance announce");
                    }
                } else {
                    // We don't hold this shard — request acquisition.
                    // Acquisition is expensive (HF download or P2P
                    // fetch); the per-model cooldown gates ONLY this
                    // path so a multi-peer departure doesn't trigger
                    // duplicate downloads.
                    if let Some(last) = self.last_rebalance_per_model.get(&shard_id.model_id) {
                        if last.elapsed().as_secs() < REBALANCE_COOLDOWN_SECS {
                            tracing::debug!(
                                model = %shard_id.model_id,
                                "Rebalance acquisition cooldown active, skipping"
                            );
                            continue;
                        }
                    }
                    if self
                        .acquisition_tx
                        .try_send(AcquisitionCommand::Acquire {
                            model_id: shard_id.model_id.clone(),
                        })
                        .is_err()
                    {
                        self.shared_state
                            .metrics
                            .channel_metrics
                            .acquisition
                            .record_dropped();
                    } else {
                        self.shared_state
                            .metrics
                            .channel_metrics
                            .acquisition
                            .record_sent();
                    }
                    tracing::info!(
                        model = %shard_id.model_id,
                        shard = shard_id.index,
                        "Requesting acquisition of under-replicated shard"
                    );
                    self.last_rebalance_per_model
                        .insert(shard_id.model_id.clone(), now);
                }
            }
        }
    }

    /// Find all shards that are now under-replicated because `departed_peer` left.
    /// Uses the reverse index to only check shards the departed peer held,
    /// making this O(shards_held_by_peer) instead of O(all_shards).
    fn find_underreplicated_shards(&self, _departed_peer: &NodeId) -> Vec<(ShardId, Vec<NodeId>)> {
        let mut result = Vec::new();

        // The departed peer has already been removed from shard_holders by
        // remove_peer_from_all_shards() in health monitor, so we can't use the
        // reverse index (it was cleared). Instead, scope the scan to models the
        // departed peer's node was likely holding — but since we don't know which
        // models, scan all shards and check for under-replication.
        // This is O(all_shards) but only fires on peer departure (rare event).
        for entry in self.shared_state.model_registry.all_shard_entries() {
            let (shard_id, holders) = entry;
            if holders.len() < MIN_REPLICATION {
                result.push((shard_id, holders));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_replication_is_2() {
        assert_eq!(MIN_REPLICATION, 2);
    }

    #[test]
    fn cooldown_is_5_minutes() {
        assert_eq!(REBALANCE_COOLDOWN_SECS, 300);
    }
}
