use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
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
    shutdown_rx: watch::Receiver<bool>,
    last_rebalance: Option<Instant>,
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
            last_rebalance: None,
        }
    }

    /// Run the rebalancer event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        tracing::info!("ShardRebalancer running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("ShardRebalancer shutting down");
                        break;
                    }
                }
                event = self.rebalance_rx.recv() => {
                    match event {
                        Some(event) => self.handle_event(event).await,
                        None => {
                            tracing::info!("Rebalance channel closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_event(&mut self, event: RebalanceEvent) {
        // Check cooldown
        if let Some(last) = self.last_rebalance {
            if last.elapsed().as_secs() < REBALANCE_COOLDOWN_SECS {
                tracing::debug!("Rebalance cooldown active, skipping");
                return;
            }
        }

        match event {
            RebalanceEvent::PeerLeft(departed_peer) => {
                tracing::info!(
                    peer = %departed_peer,
                    "Peer departed, checking shard replication"
                );
                self.handle_peer_left(&departed_peer).await;
            }
            RebalanceEvent::PeerJoined(new_peer) => {
                tracing::info!(
                    peer = %new_peer,
                    "New peer joined, re-announcing shards"
                );
                self.handle_peer_joined().await;
            }
            RebalanceEvent::DiskPressure { available_mb } => {
                tracing::warn!(
                    available_mb,
                    "Disk pressure detected, may need to evict shards"
                );
            }
            RebalanceEvent::ManualTrigger => {
                tracing::info!("Manual rebalance triggered");
                self.check_all_shards().await;
            }
        }

        self.last_rebalance = Some(Instant::now());
    }

    async fn handle_peer_left(&self, departed_peer: &NodeId) {
        let underreplicated = self.find_underreplicated_shards(departed_peer);

        if underreplicated.is_empty() {
            tracing::debug!("No under-replicated shards found");
            return;
        }

        tracing::info!(
            count = underreplicated.len(),
            "Found under-replicated shards after peer departure"
        );

        let local_node_id = self.shared_state.identity.node_id().clone();

        for (shard_id, holders) in &underreplicated {
            // Skip if we already hold this shard
            if holders.contains(&local_node_id) {
                continue;
            }

            // Announce willingness to host by re-broadcasting shard info
            let announce = crate::types::ShardAnnounce {
                node_id: local_node_id.clone(),
                shards: vec![shard_id.clone()],
                timestamp: chrono::Utc::now(),
            };

            let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::warn!(
                    error = %e,
                    model = %shard_id.model_id,
                    index = shard_id.index,
                    "Failed to broadcast shard rebalance offer"
                );
            }
        }
    }

    async fn handle_peer_joined(&self) {
        // Re-announce our own shard holdings to help the new peer
        // discover what's available on the network.
        let local_node_id = self.shared_state.identity.node_id().clone();
        let mut our_shards = Vec::new();

        for (shard_id, holders) in self.shared_state.model_registry.all_shard_entries() {
            if holders.contains(&local_node_id) {
                our_shards.push(shard_id);
            }
        }

        if !our_shards.is_empty() {
            let announce = crate::types::ShardAnnounce {
                node_id: local_node_id,
                shards: our_shards,
                timestamp: chrono::Utc::now(),
            };

            let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
            let _ = self.network_tx.send(msg).await;
        }
    }

    async fn check_all_shards(&self) {
        let departed = NodeId([0u8; 32]); // dummy — check all shards regardless
        let underreplicated = self.find_underreplicated_shards(&departed);

        if !underreplicated.is_empty() {
            tracing::info!(
                count = underreplicated.len(),
                "Found under-replicated shards during full check"
            );
        }
    }

    /// Find all shards that are now under-replicated because `departed_peer` left.
    fn find_underreplicated_shards(&self, departed_peer: &NodeId) -> Vec<(ShardId, Vec<NodeId>)> {
        let mut result = Vec::new();

        for (shard_id, holders) in self.shared_state.model_registry.all_shard_entries() {
            // Remove the departed peer from the holder list
            let remaining: Vec<NodeId> =
                holders.into_iter().filter(|h| h != departed_peer).collect();

            if remaining.len() < MIN_REPLICATION {
                result.push((shard_id, remaining));
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
