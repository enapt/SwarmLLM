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
                    self.cleanup_acquisition_progress();
                    self.cleanup_stale_channels();
                    self.cleanup_model_vote_tallies();
                    // Cleanup expired anti-gaming rate limit entries
                    self.shared_state.anti_gaming.lock().await.cleanup();
                    // Decay trust scores toward default (0.5) on each health ping cycle
                    self.shared_state.trust_manager.decay_all(&self.shared_state.peer_registry);
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

        let active_request_count = self.shared_state.active_pipelines.len() as u32;
        let node_id = Some(self.shared_state.identity.node_id().clone());
        let msg = SwarmMessage::HealthPing {
            nonce,
            timestamp,
            node_id,
            active_request_count,
        };

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

        // Populate real system metrics
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let ram_total_mb = sys.total_memory() / (1024 * 1024);
        let ram_available_mb = sys.available_memory() / (1024 * 1024);

        // Get disk space for the data_dir partition, not all disks combined.
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let data_dir = &self.shared_state.config.node.data_dir;
        let disk_available_mb: u64 = disks
            .list()
            .iter()
            .filter(|d| data_dir.starts_with(d.mount_point()))
            .max_by_key(|d| d.mount_point().as_os_str().len())
            .map(|d| d.available_space() / (1024 * 1024))
            .unwrap_or_else(|| {
                // Fallback: sum of all disks if data_dir mount not found
                disks
                    .list()
                    .iter()
                    .map(|d| d.available_space() / (1024 * 1024))
                    .sum()
            });

        let cap = crate::types::NodeCapability {
            node_id: node_id.clone(),
            gpu: gpu_info,
            ram_total_mb,
            ram_available_mb,
            disk_available_mb,
            bandwidth_mbps: 0.0,
            hosted_shards: hosted_shards.clone(),
            max_contribution: self.shared_state.config.node.contribution.clone().into(),
            uptime_seconds,
            version: env!("CARGO_PKG_VERSION").to_string(),
            region: self.shared_state.config.identity.region.clone(),
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

    /// Broadcast model manifests and HF sources so peers can discover and acquire models.
    async fn broadcast_manifests(&self) {
        let our_id = self.shared_state.identity.node_id().clone();
        for manifest in self.shared_state.model_registry.models() {
            // Only broadcast manifests we published (or locally generated)
            if manifest.publisher != our_id {
                continue;
            }
            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelManifest(manifest.clone()));
            let _ = self.network_tx.send(msg).await;

            // Also broadcast HfSourceGossip so late-joining peers discover the HF source
            if let Some(hf_source) = self.shared_state.hf_sources.get(&manifest.id) {
                let gossip = crate::types::HfSourceGossip {
                    model_id: manifest.id.clone(),
                    repo_id: hf_source.repo_id.clone(),
                    filename: hf_source.filename.clone(),
                    publisher: our_id.clone(),
                };
                let msg = NetworkCommand::Broadcast(SwarmMessage::HfSourceGossip(gossip));
                let _ = self.network_tx.send(msg).await;
            }
        }
    }

    async fn check_peer_health(&self) {
        let now = chrono::Utc::now();
        let timeout =
            chrono::Duration::seconds((PING_INTERVAL.as_secs() * MAX_MISSED_PINGS as u64) as i64);

        // Collect node IDs participating in active inference pipelines —
        // these must not be removed even if they appear stale (long forward passes).
        let mut active_nodes = std::collections::HashSet::new();
        for entry in self.shared_state.active_pipelines.iter() {
            for seg in &entry.value().segments {
                active_nodes.insert(seg.node_id.clone());
            }
        }

        let mut stale_peers = Vec::new();

        for entry in self.shared_state.peer_registry.iter() {
            let peer = entry.value();
            if now.signed_duration_since(peer.last_seen) > timeout {
                if active_nodes.contains(entry.key()) {
                    tracing::debug!(
                        peer = %entry.key(),
                        "Peer appears stale but is active in inference pipeline, skipping removal"
                    );
                    continue;
                }
                stale_peers.push(entry.key().clone());
            }
        }

        for peer_id in stale_peers {
            self.shared_state.peer_registry.remove(&peer_id);
            // Clean up stale peer from shard_registry to prevent phantom holder entries
            for mut entry in self.shared_state.shard_registry.iter_mut() {
                entry.value_mut().retain(|nid| nid != &peer_id);
            }
            tracing::info!(peer = %peer_id, "Removed stale peer (and shard registry entries)");
            // Signal the rebalancer that a peer has left
            if self
                .rebalance_tx
                .try_send(RebalanceEvent::PeerLeft(peer_id))
                .is_err()
            {
                self.shared_state.channel_metrics.rebalance.record_dropped();
            } else {
                self.shared_state.channel_metrics.rebalance.record_sent();
            }
        }
    }

    /// Remove stale pending_layer_results (closed oneshot channels) and
    /// streaming_token_txs (closed mpsc channels) to prevent memory leaks.
    fn cleanup_stale_channels(&self) {
        // pending_layer_results: remove entries where the receiver has been dropped
        let stale_layer: Vec<_> = self
            .shared_state
            .pending_layer_results
            .iter()
            .filter(|entry| entry.value().is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_layer.is_empty() {
            tracing::debug!(
                count = stale_layer.len(),
                "Cleaning up stale pending_layer_results"
            );
            for key in stale_layer {
                self.shared_state.pending_layer_results.remove(&key);
            }
        }

        // streaming_token_txs: remove entries where the receiver has been dropped
        let stale_stream: Vec<_> = self
            .shared_state
            .streaming_token_txs
            .iter()
            .filter(|entry| entry.value().is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_stream.is_empty() {
            tracing::debug!(
                count = stale_stream.len(),
                "Cleaning up stale streaming_token_txs"
            );
            for key in stale_stream {
                self.shared_state.streaming_token_txs.remove(&key);
            }
        }
    }

    /// Periodic cleanup for model_vote_tallies — remove closed entries older than 24 hours (DAE-M11).
    fn cleanup_model_vote_tallies(&self) {
        let to_remove: Vec<_> = self
            .shared_state
            .model_vote_tallies
            .iter()
            .filter(|entry| {
                let tally = entry.value();
                let age = chrono::Utc::now() - tally.opened_at;
                tally.closed && age > chrono::Duration::hours(24)
            })
            .map(|entry| *entry.key())
            .collect();
        if !to_remove.is_empty() {
            tracing::debug!(
                count = to_remove.len(),
                "Cleaning up old model vote tallies"
            );
            for key in to_remove {
                self.shared_state.model_vote_tallies.remove(&key);
            }
        }
    }

    /// Remove completed/failed acquisition entries older than 1 hour.
    fn cleanup_acquisition_progress(&self) {
        use crate::model::acquisition::AcquisitionState;

        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        let to_remove: Vec<_> = self
            .shared_state
            .acquisition_progress
            .iter()
            .filter(|entry| {
                let status = entry.value();
                match &status.state {
                    AcquisitionState::Complete | AcquisitionState::Failed { .. } => {
                        status.started_at.map_or(true, |s| s < cutoff)
                    }
                    _ => false,
                }
            })
            .map(|entry| entry.key().clone())
            .collect();

        if !to_remove.is_empty() {
            tracing::debug!(
                count = to_remove.len(),
                "Cleaning up stale acquisition progress entries"
            );
            for key in to_remove {
                self.shared_state.acquisition_progress.remove(&key);
            }
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
