use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{NetworkCommand, NodeId, RebalanceEvent, ShardId, SwarmMessage};

/// Minimum replication factor for each shard.
const MIN_REPLICATION: usize = 2;

/// Manages shard rebalancing in response to network topology changes.
///
/// When the HealthMonitor detects a peer leaving, it signals the
/// ShardRebalancer via the `RebalanceEvent` channel. The rebalancer
/// re-announces the under-replicated parts this node holds, and wakes the
/// download pass for the ones it does not — it never fetches anything itself.
pub struct ShardRebalancer {
    shared_state: Arc<SharedState>,
    rebalance_rx: mpsc::Receiver<RebalanceEvent>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Queued PeerLeft events to batch-process.
    pending_peer_left: Vec<NodeId>,
}

impl ShardRebalancer {
    pub fn new(
        shared_state: Arc<SharedState>,
        rebalance_rx: mpsc::Receiver<RebalanceEvent>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            shared_state,
            rebalance_rx,
            network_tx,
            shutdown_rx,
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
                            "Rebalancing: computer {} departed",
                            crate::identity::nickname::short_display_name(
                                &departed_peer,
                                &self.shared_state.nickname_registry
                            )
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

    /// Batch-process queued PeerLeft events.
    async fn process_pending_departures(&mut self) {
        let departed: Vec<NodeId> = std::mem::take(&mut self.pending_peer_left);
        if departed.is_empty() {
            return;
        }

        let local_node_id = self.shared_state.identity.node_id().clone();
        let mut not_held_here = 0usize;

        for departed_peer in &departed {
            let underreplicated = self.find_underreplicated_shards(departed_peer);
            for (shard_id, holders) in &underreplicated {
                if holders.contains(&local_node_id) {
                    // We hold this shard — re-announce it so peers know
                    // it's still available. Re-announces are cheap (a
                    // single GossipSub message, no DB or network fetch),
                    // and under a thundering-herd departure suppressing
                    // them would leave locally-held shards undiscoverable
                    // while peers retry HF. One shard — incremental.
                    let announce = crate::model::manifest::shard_announce(
                        &self.shared_state.model_registry,
                        local_node_id.clone(),
                        vec![shard_id.clone()],
                        Vec::new(),
                    );
                    let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
                    if let Err(e) = self.network_tx.send(msg).await {
                        tracing::warn!(error = %e, "Failed to broadcast shard rebalance announce");
                    }
                } else {
                    not_held_here += 1;
                }
            }
        }

        // A part down to one holder that this node does not hold is the
        // download pass's to fetch, or not: it already ranks a part by how few
        // hold it, and it alone asks everything a download nobody asked for
        // must — auto-download on, the storage budget, whether prune would keep
        // it (`would_shed_copy`: a model that cannot chat, one with its safety
        // training removed that nobody chose), the swarm's upload, a part the
        // owner removed, a metered connection. This branch used to send
        // `AcquisitionCommand::Acquire` for the WHOLE model and asked none of
        // them (pre-release review, 2026-10-15).
        if not_held_here > 0 {
            tracing::info!(
                parts = not_held_here,
                "Parts this computer does not hold are down to one copy — asking the download pass"
            );
            self.shared_state.models.auto_manage_notify.notify_one();
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
        for (shard_id, _unfiltered) in self.shared_state.model_registry.all_shard_entries() {
            // Count through `shard_holders`, not the raw map.
            //
            // `all_shard_entries` returns the holder map unfiltered;
            // `shard_holders` drops any holder positively claiming a DIFFERENT
            // GGUF build, because its bytes would fail our hash. Counting the
            // raw list makes a shard with one real copy and one build-conflict
            // holder read as replicated twice, so the one thing this function
            // exists to catch — a shard down to a single usable copy after a
            // peer left — is exactly what it misses.
            //
            // The filtered list is what goes into the result too: the caller
            // uses it to pick somewhere to fetch from, and a conflicting build
            // is not somewhere to fetch from.
            let holders = self.shared_state.model_registry.shard_holders(&shard_id);
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
}
