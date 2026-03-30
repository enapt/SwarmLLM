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
    /// Track last broadcast shard set for delta compression.
    /// Full re-announce only when set changes or every FULL_REANNOUNCE ticks.
    last_announced_shards: std::collections::HashSet<crate::types::ShardId>,
    /// Counter for periodic full re-announce (ensures late-joining peers get data).
    shard_announce_counter: u64,
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
            last_announced_shards: std::collections::HashSet::new(),
            shard_announce_counter: 0,
        }
    }

    /// Compute how many base ticks (30s each) between heavy gossip broadcasts.
    /// Scales with log(peer_count) to reduce bandwidth at large network sizes.
    ///   ≤10 peers:  every tick   (30s)
    ///   ~100 peers: every 2 ticks (60s)
    ///   ~1K peers:  every 4 ticks (120s)
    ///   ~10K peers: every 8 ticks (240s)
    fn gossip_broadcast_interval(&self) -> u64 {
        let peer_count = self.shared_state.peer_registry.len();
        if peer_count <= 10 {
            1
        } else {
            // floor(log2(peer_count / 5)), clamped to [1, 10]
            let ratio = peer_count / 5;
            // bit_length - 1 = floor(log2(n)) for n >= 1
            let log2 = (usize::BITS - 1 - ratio.leading_zeros()) as u64;
            log2.clamp(1, 10)
        }
    }

    /// Run the health monitoring loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        let mut interval = tokio::time::interval(PING_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut nonce: u64 = 0;

        tracing::info!(target: "swarmllm::health::monitor", "HealthMonitor running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!(target: "swarmllm::health::monitor", "HealthMonitor shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    nonce = nonce.wrapping_add(1);

                    // Health pings and peer liveness: always run every 30s (critical)
                    self.send_health_ping(nonce).await;
                    self.check_peer_health().await;

                    // Heavy gossip broadcasts: scale frequency with network size.
                    // At ≤10 peers, every 30s (same as before).
                    // At 1K+ peers, every ~120s to reduce bandwidth.
                    let broadcast_every = self.gossip_broadcast_interval();
                    if nonce % broadcast_every == 0 {
                        self.broadcast_capabilities().await;
                        self.broadcast_manifests().await;
                        self.broadcast_region_summary().await;
                    }

                    // Cleanup tasks: run every tick (cheap, local-only)
                    self.cleanup_acquisition_progress();
                    self.cleanup_stale_channels();
                    self.cleanup_stale_peer_id_map();
                    // Cleanup expired anti-gaming rate limit entries
                    self.shared_state.credits.anti_gaming.lock().await.cleanup();
                    // Decay trust scores toward default (0.5) on each health ping cycle
                    self.shared_state.credits.trust_manager.decay_all(&self.shared_state.peer_registry);
                    // Clean up stale AllReduce entries (receiver dropped/timed out)
                    self.shared_state.allreduce_registry.cleanup_stale();
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

    async fn broadcast_capabilities(&mut self) {
        let node_id = self.shared_state.identity.node_id().clone();

        // Gather hosted shards using the reverse-index for O(1) lookup.
        let mut hosted_shards = self.shared_state.model_registry.shards_for_node(&node_id);

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

        let gpu_info = self.shared_state.gpu_info.as_ref().map(|g| {
            let bandwidth = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
            crate::types::GpuInfo {
                name: g.name.clone(),
                vram_total_mb: g.vram_total_mb,
                vram_available_mb: g.vram_free_mb,
                compute_capability: None,
                memory_bandwidth_gbps: bandwidth,
            }
        });

        // Use real uptime so message content changes each broadcast (avoids GossipSub dedup)
        let uptime_seconds = {
            let stats = self.shared_state.metrics.node_stats.read().await;
            (chrono::Utc::now() - stats.uptime_start)
                .num_seconds()
                .max(0) as u64
        };

        // Populate real system metrics.
        // sysinfo does blocking filesystem reads (/proc/*) — use block_in_place.
        let data_dir = self.shared_state.config.node.data_dir.clone();
        let (ram_total_mb, ram_available_mb, disk_available_mb) =
            tokio::task::block_in_place(|| {
                let mut sys = sysinfo::System::new();
                sys.refresh_memory();
                let ram_total = sys.total_memory() / (1024 * 1024);
                let ram_avail = sys.available_memory() / (1024 * 1024);

                let disks = sysinfo::Disks::new_with_refreshed_list();
                let disk_avail: u64 = disks
                    .list()
                    .iter()
                    .filter(|d| data_dir.starts_with(d.mount_point()))
                    .max_by_key(|d| d.mount_point().as_os_str().len())
                    .map(|d| d.available_space() / (1024 * 1024))
                    .unwrap_or_else(|| {
                        disks
                            .list()
                            .iter()
                            .map(|d| d.available_space() / (1024 * 1024))
                            .sum()
                    });
                (ram_total, ram_avail, disk_avail)
            });

        let est_tokens_per_sec_7b = gpu_info
            .as_ref()
            .map(|g| {
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(
                    g.memory_bandwidth_gbps,
                    true,
                )
            })
            .unwrap_or_else(|| {
                // CPU-only: assume ~50 GB/s DDR4/5 bandwidth
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(50.0, false)
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
            est_tokens_per_sec_7b,
        };

        let msg = NetworkCommand::Broadcast(SwarmMessage::NodeCapabilityUpdate(cap));
        if let Err(e) = self.network_tx.send(msg).await {
            tracing::debug!(error = %e, "DIAG: failed to broadcast capability update");
        }

        // Delta-compressed shard announcements: only broadcast when shard set
        // changes, or every 10 broadcast cycles as a full re-announce (ensures
        // late-joining peers get the full picture). At 10K peers with scaled
        // gossip interval (~240s), full re-announce happens every ~40 min.
        if !hosted_shards.is_empty() {
            let current_set: std::collections::HashSet<_> = hosted_shards.iter().cloned().collect();
            let shards_changed = current_set != self.last_announced_shards;
            self.shard_announce_counter += 1;
            // Full re-announce every 10 broadcast cycles so late-joining peers
            // eventually discover our shards even if nothing changed.
            let periodic_reannounce = self.shard_announce_counter % 10 == 0;

            if shards_changed || periodic_reannounce {
                let shard_count = hosted_shards.len();
                let announce = crate::types::ShardAnnounce {
                    node_id,
                    shards: hosted_shards,
                    timestamp: chrono::Utc::now(),
                };
                let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
                if let Err(e) = self.network_tx.send(msg).await {
                    tracing::debug!(error = %e, shard_count, "DIAG: failed to broadcast shard announce");
                }
                self.last_announced_shards = current_set;
            }
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
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast manifest");
            }

            // Also broadcast HfSourceGossip so late-joining peers discover the HF source
            if let Some(hf_source) = self.shared_state.models.hf_sources.get(&manifest.id) {
                let gossip = crate::types::HfSourceGossip {
                    model_id: manifest.id.clone(),
                    repo_id: hf_source.repo_id.clone(),
                    filename: hf_source.filename.clone(),
                    publisher: our_id.clone(),
                    mmproj_filename: hf_source.mmproj_filename.clone(),
                };
                let msg = NetworkCommand::Broadcast(SwarmMessage::HfSourceGossip(gossip));
                if let Err(e) = self.network_tx.send(msg).await {
                    tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast HF source");
                }
            }
        }
    }

    /// Broadcast compact per-region shard summaries and demand gossip.
    /// Published on every 30s tick to `swarm/regions` topic.
    async fn broadcast_region_summary(&self) {
        // Determine our region — skip if unknown
        let our_region = {
            let detected = self.shared_state.detected_region.read().await;
            match detected.as_ref() {
                Some(r) => r.to_uppercase(),
                None => {
                    if let Some(ref r) = self.shared_state.config.identity.region {
                        r.to_uppercase()
                    } else {
                        return; // No region — nothing to broadcast
                    }
                }
            }
        };

        let our_id = self.shared_state.identity.node_id().clone();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Count same-region peers (including self)
        let mut region_node_count: u32 = 1; // self
        for peer in self.shared_state.peer_registry.iter() {
            if let Some(ref cap) = peer.value().capability {
                if let Some(ref r) = cap.region {
                    if r.to_uppercase() == our_region {
                        region_node_count = region_node_count.saturating_add(1);
                    }
                }
            }
        }

        // Build a set of same-region node IDs once to avoid O(holders) peer_registry
        // lookups per shard (was O(models × shards × holders), now O(models × shards)).
        let same_region_nodes: std::collections::HashSet<crate::types::NodeId> = {
            let mut set = std::collections::HashSet::new();
            set.insert(our_id.clone());
            for entry in self.shared_state.peer_registry.iter() {
                if let Some(ref cap) = entry.capability {
                    if let Some(ref r) = cap.region {
                        if r.to_uppercase() == our_region {
                            set.insert(entry.key().clone());
                        }
                    }
                }
            }
            set
        };

        // For each model, count same-region shard holders
        for manifest in self.shared_state.model_registry.models() {
            let mut shard_counts: Vec<(u32, u32)> = Vec::new();
            for shard_info in &manifest.shards {
                let sid = crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: shard_info.index,
                };
                let holders = self.shared_state.model_registry.shard_holders(&sid);
                let regional_count = holders
                    .iter()
                    .filter(|h| same_region_nodes.contains(h))
                    .count() as u32;
                shard_counts.push((shard_info.index, regional_count));
            }

            if shard_counts.is_empty() {
                continue;
            }

            let summary = crate::types::RegionShardSummary {
                region: our_region.clone(),
                model_id: manifest.id.clone(),
                shard_counts,
                region_node_count,
                publisher: our_id.clone(),
                timestamp_ms: now_ms,
            };

            // Also update our own shared state
            let key = (our_region.clone(), manifest.id.clone());
            self.shared_state
                .region_shard_summaries
                .insert(key, summary.clone());

            let msg = NetworkCommand::Broadcast(SwarmMessage::RegionShardSummary(summary));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "Failed to broadcast region summary");
            }
        }

        // Broadcast demand gossip for models with recent requests
        for entry in self.shared_state.region_demand.iter() {
            let (model_id, region) = entry.key();
            let rate = *entry.value();
            if rate < 0.01 {
                continue; // Don't gossip negligible demand
            }
            let demand = crate::types::ModelDemandGossip {
                model_id: model_id.clone(),
                region: region.clone(),
                decayed_rate: rate,
                window_requests: 0, // Raw count already decayed into rate
                publisher: our_id.clone(),
                timestamp_ms: now_ms,
            };
            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelDemandGossip(demand));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, "Failed to broadcast demand gossip");
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
            let age = now
                .signed_duration_since(peer.last_seen)
                .max(chrono::Duration::zero());
            if age > timeout {
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

        if !stale_peers.is_empty() {
            tracing::warn!(
                stale_count = stale_peers.len(),
                total_peers = self.shared_state.peer_registry.len(),
                active_pipelines = self.shared_state.active_pipelines.len(),
                "DIAG: removing stale peers"
            );
        }
        for peer_id in stale_peers {
            self.shared_state.peer_registry.remove(&peer_id);
            // Clean up stale peer from model_registry shard holders
            self.shared_state
                .model_registry
                .remove_peer_from_all_shards(&peer_id);
            tracing::info!(peer = %peer_id, "Removed stale peer (and shard registry entries)");
            // Signal the rebalancer that a peer has left
            if self
                .rebalance_tx
                .try_send(RebalanceEvent::PeerLeft(peer_id))
                .is_err()
            {
                self.shared_state
                    .metrics
                    .channel_metrics
                    .rebalance
                    .record_dropped();
            } else {
                self.shared_state
                    .metrics
                    .channel_metrics
                    .rebalance
                    .record_sent();
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
            tracing::info!(
                count = stale_layer.len(),
                total_pending = self.shared_state.pending_layer_results.len(),
                request_ids = ?stale_layer.iter().take(5).map(|u| u.to_string()).collect::<Vec<_>>(),
                "DIAG: cleaning up stale pending_layer_results"
            );
            for key in stale_layer {
                self.shared_state.pending_layer_results.remove(&key);
            }
        }

        // pending_tp_partials: remove entries older than 60 seconds (stale AllReduce collectors)
        let stale_tp: Vec<_> = self
            .shared_state
            .pending_tp_partials
            .iter()
            .filter(|entry| entry.value().created_at.elapsed().as_secs() > 60)
            .map(|entry| *entry.key())
            .collect();
        if !stale_tp.is_empty() {
            tracing::info!(
                count = stale_tp.len(),
                "DIAG: cleaning up stale pending_tp_partials"
            );
            for key in stale_tp {
                self.shared_state.pending_tp_partials.remove(&key);
            }
        }

        // pending_vision_results: remove entries where the oneshot receiver has been dropped
        let stale_vision: Vec<_> = self
            .shared_state
            .pending_vision_results
            .iter()
            .filter(|entry| entry.value().1.is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_vision.is_empty() {
            tracing::info!(
                count = stale_vision.len(),
                "DIAG: cleaning up stale pending_vision_results"
            );
            for key in stale_vision {
                self.shared_state.pending_vision_results.remove(&key);
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
            tracing::info!(
                count = stale_stream.len(),
                total_streaming = self.shared_state.streaming_token_txs.len(),
                "DIAG: cleaning up stale streaming_token_txs"
            );
            for key in stale_stream {
                self.shared_state.streaming_token_txs.remove(&key);
            }
        }

        // active_relay_circuits: remove entries older than 1 hour (abnormally terminated)
        const RELAY_CIRCUIT_TTL_SECS: u64 = 3600;
        let stale_relay: Vec<_> = self
            .shared_state
            .active_relay_circuits
            .iter()
            .filter(|entry| entry.value().elapsed().as_secs() > RELAY_CIRCUIT_TTL_SECS)
            .map(|entry| *entry.key())
            .collect();
        if !stale_relay.is_empty() {
            tracing::debug!(
                count = stale_relay.len(),
                "DIAG: cleaning up stale active_relay_circuits"
            );
            for key in stale_relay {
                self.shared_state.active_relay_circuits.remove(&key);
            }
        }
    }

    /// Clean stale peer_id_map entries for peers no longer in peer_registry.
    /// Only runs when the map exceeds 1000 entries to avoid removing entries
    /// that are intentionally kept across disconnects for short periods.
    fn cleanup_stale_peer_id_map(&self) {
        const SOFT_CAP: usize = 8_000;
        const EVICT_TO: usize = 6_000;

        if self.shared_state.peer_id_map.len() <= SOFT_CAP {
            return;
        }
        // First pass: evict entries not in peer_registry (stale)
        let stale_peers: Vec<_> = self
            .shared_state
            .peer_id_map
            .iter()
            .filter(|entry| !self.shared_state.peer_registry.contains_key(entry.key()))
            .map(|entry| entry.key().clone())
            .collect();
        let removed = stale_peers.len();
        for nid in stale_peers {
            self.shared_state.peer_id_map.remove(&nid);
        }
        // Second pass: if still over target, evict oldest (arbitrary order from DashMap)
        let mut removed2 = 0;
        if self.shared_state.peer_id_map.len() > EVICT_TO {
            let excess = self.shared_state.peer_id_map.len() - EVICT_TO;
            let to_evict: Vec<_> = self
                .shared_state
                .peer_id_map
                .iter()
                .filter(|entry| !self.shared_state.peer_registry.contains_key(entry.key()))
                .take(excess)
                .map(|entry| entry.key().clone())
                .collect();
            removed2 = to_evict.len();
            for nid in &to_evict {
                self.shared_state.peer_id_map.remove(nid);
            }
        }
        let total_removed = removed + removed2;
        if total_removed > 0 {
            tracing::debug!(
                removed = total_removed,
                remaining = self.shared_state.peer_id_map.len(),
                "Cleaned stale peer_id_map entries"
            );
        }
    }

    /// Remove completed/failed acquisition entries older than 1 hour.
    fn cleanup_acquisition_progress(&self) {
        use crate::model::acquisition::AcquisitionState;

        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        let to_remove: Vec<_> = self
            .shared_state
            .models
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
                self.shared_state.models.acquisition_progress.remove(&key);
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
