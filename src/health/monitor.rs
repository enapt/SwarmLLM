use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{NetworkCommand, RebalanceEvent, SwarmMessage};

/// Periodic health monitoring for the node and its peers.
///
/// Sends health pings to known peers and tracks their response latencies.
/// Peers that fail to respond after multiple missed intervals are considered offline.
/// When peers are detected as stale, a `RebalanceEvent::PeerLeft` is emitted.
pub struct HealthMonitor {
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    rebalance_tx: mpsc::Sender<RebalanceEvent>,
    shutdown_rx: watch::Receiver<bool>,
}

/// How often to send health pings.
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// Number of missed pings before a peer is considered dead.
const MAX_MISSED_PINGS: u32 = 3;

impl HealthMonitor {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        rebalance_tx: mpsc::Sender<RebalanceEvent>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            shared_state,
            network_tx,
            rebalance_tx,
            shutdown_rx,
        }
    }

    /// Run the health monitoring loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        let mut interval = tokio::time::interval(PING_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut nonce: u64 = 0;

        tracing::info!("HealthMonitor running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("HealthMonitor shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    nonce = nonce.wrapping_add(1);
                    self.send_health_ping(nonce).await;
                    self.broadcast_capabilities().await;
                    self.broadcast_manifests().await;
                    self.check_peer_health().await;
                }
            }
        }

        Ok(())
    }

    async fn send_health_ping(&self, nonce: u64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let msg = SwarmMessage::HealthPing { nonce, timestamp };

        if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
            tracing::warn!(error = %e, "Failed to send health ping");
        }
    }

    async fn broadcast_capabilities(&self) {
        let node_id = self.shared_state.identity.node_id().clone();

        // Gather hosted shards from model_registry (which respects --shards range).
        let mut hosted_shards = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in self.shared_state.model_registry.all_shard_entries() {
            let (shard_id, holders) = entry;
            if holders.contains(&node_id) && seen.insert(shard_id.clone()) {
                hosted_shards.push(shard_id);
            }
        }

        // If no shards from registry but we have a loaded model (and no shard_range),
        // represent the full model as shard index 0 for backward compatibility.
        if hosted_shards.is_empty() && self.shared_state.config.inference.shard_range.is_none() {
            if let Some(info) = self.shared_state.loaded_model_info.read().await.as_ref() {
                hosted_shards.push(crate::types::ShardId {
                    model_id: crate::types::ModelId(info.name.clone()),
                    index: 0,
                });
            }
        }

        let gpu_info = self
            .shared_state
            .gpu_info
            .as_ref()
            .map(|g| crate::types::GpuInfo {
                name: g.name.clone(),
                vram_total_mb: g.vram_total_mb,
                vram_available_mb: g.vram_free_mb,
                compute_capability: None,
            });

        // Use real uptime so message content changes each broadcast (avoids GossipSub dedup)
        let uptime_seconds = {
            let stats = self.shared_state.node_stats.read().await;
            (chrono::Utc::now() - stats.uptime_start)
                .num_seconds()
                .max(0) as u64
        };

        let cap = crate::types::NodeCapability {
            node_id: node_id.clone(),
            gpu: gpu_info,
            ram_total_mb: 0,
            ram_available_mb: 0,
            disk_available_mb: 0,
            bandwidth_mbps: 0.0,
            hosted_shards: hosted_shards.clone(),
            max_contribution: crate::types::ContributionLevel::Moderate,
            uptime_seconds,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let msg = NetworkCommand::Broadcast(SwarmMessage::NodeCapabilityUpdate(cap));
        let _ = self.network_tx.send(msg).await;

        // Also broadcast shard announcements for our hosted models
        if !hosted_shards.is_empty() {
            let announce = crate::types::ShardAnnounce {
                node_id,
                shards: hosted_shards,
                timestamp: chrono::Utc::now(),
            };
            let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
            let _ = self.network_tx.send(msg).await;
        }
    }

    /// Broadcast model manifests so peers can discover and acquire models.
    async fn broadcast_manifests(&self) {
        for manifest in self.shared_state.model_registry.models() {
            // Only broadcast manifests we published (or locally generated)
            let our_id = self.shared_state.identity.node_id();
            if manifest.publisher != *our_id {
                continue;
            }
            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelManifest(manifest));
            let _ = self.network_tx.send(msg).await;
        }
    }

    async fn check_peer_health(&self) {
        let now = chrono::Utc::now();
        let timeout =
            chrono::Duration::seconds((PING_INTERVAL.as_secs() * MAX_MISSED_PINGS as u64) as i64);

        let mut stale_peers = Vec::new();

        for entry in self.shared_state.peer_registry.iter() {
            let peer = entry.value();
            if now.signed_duration_since(peer.last_seen) > timeout {
                stale_peers.push(entry.key().clone());
            }
        }

        for peer_id in stale_peers {
            self.shared_state.peer_registry.remove(&peer_id);
            tracing::info!(peer = %peer_id, "Removed stale peer");
            // Signal the rebalancer that a peer has left
            let _ = self
                .rebalance_tx
                .try_send(RebalanceEvent::PeerLeft(peer_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_interval_is_30s() {
        assert_eq!(PING_INTERVAL, Duration::from_secs(30));
    }

    #[test]
    fn max_missed_pings_is_3() {
        assert_eq!(MAX_MISSED_PINGS, 3);
    }
}
