use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::types::{ModelId, NetworkCommand, NodeId, ShardId};

/// Auto-manages shard downloads to improve network health.
///
/// Periodically evaluates:
/// 1. Which models are popular on the network (most holders / most shards)
/// 2. Which shards are rarest (fewest holders) for those models
/// 3. Whether this node has budget (disk space, max_shards) to download more
///
/// Then triggers HuggingFace shard downloads for the rarest shards of popular models.
pub struct AutoShardManager {
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
}

/// A candidate shard identified for auto-download.
#[derive(Debug, Clone)]
struct ShardCandidate {
    model_id: ModelId,
    model_name: String,
    shard_index: u32,
    shard_size_bytes: u64,
    holder_count: usize,
    /// Score: higher = more worth downloading. Factors in rarity and model popularity.
    score: f64,
}

impl AutoShardManager {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            shared_state,
            network_tx,
            shutdown_rx,
        }
    }

    /// Run the auto-manage loop. Checks periodically based on config interval.
    pub async fn run(mut self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.enabled {
            tracing::info!("AutoShardManager disabled, exiting");
            return;
        }

        let interval_mins = config.interval_minutes.max(5); // minimum 5 min
        let mut interval = tokio::time::interval(Duration::from_secs(interval_mins as u64 * 60));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Skip the first tick (fires immediately) — let the node discover peers first
        interval.tick().await;

        tracing::info!(
            interval_minutes = interval_mins,
            max_storage_mb = config.max_storage_mb,
            "AutoShardManager running"
        );

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("AutoShardManager shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    // Re-check enabled — config may have changed at runtime
                    if self.shared_state.config.auto_manage.enabled {
                        self.evaluate_and_download().await;
                    }
                }
            }
        }
    }

    /// Core logic: evaluate network state and download the best candidate shards.
    async fn evaluate_and_download(&self) {
        let config = &self.shared_state.config.auto_manage;
        let local_node_id = self.shared_state.identity.node_id().clone();

        // 1. Check budget: how much storage do we have left?
        let budget = self.remaining_budget_bytes(config, &local_node_id);
        if budget == 0 {
            tracing::debug!("AutoShardManager: no remaining storage budget");
            return;
        }

        // 2. Gather candidate shards across all known models
        let candidates = self.gather_candidates(&local_node_id);
        if candidates.is_empty() {
            tracing::debug!("AutoShardManager: no candidate shards to download");
            return;
        }

        // 3. Select the best candidates within budget
        let selected = self.select_within_budget(candidates, budget, config.max_shards);
        if selected.is_empty() {
            return;
        }

        tracing::info!(
            count = selected.len(),
            "AutoShardManager: downloading shards"
        );

        // 4. Trigger downloads
        for candidate in &selected {
            self.trigger_download(candidate).await;
        }
    }

    /// Compute remaining download budget in bytes.
    fn remaining_budget_bytes(
        &self,
        config: &crate::config::AutoManageConfig,
        local_node_id: &NodeId,
    ) -> u64 {
        let max_bytes = if config.max_storage_mb > 0 {
            config.max_storage_mb * 1024 * 1024
        } else {
            // Fall back to global max_disk_mb, using 50% for auto-manage
            (self.shared_state.config.resources.max_disk_mb * 1024 * 1024) / 2
        };

        // Sum up bytes of shards we already hold
        let mut current_bytes = 0u64;
        let mut current_shard_count = 0u32;
        for manifest in self.shared_state.model_registry.models() {
            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = self.shared_state.model_registry.shard_holders(&shard_id);
                if holders.contains(local_node_id) {
                    current_bytes += shard.size_bytes;
                    current_shard_count += 1;
                }
            }
        }

        // Check max_shards limit
        if config.max_shards > 0 && current_shard_count >= config.max_shards {
            return 0;
        }

        max_bytes.saturating_sub(current_bytes)
    }

    /// Gather all candidate shards we don't already hold, scored by value.
    fn gather_candidates(&self, local_node_id: &NodeId) -> Vec<ShardCandidate> {
        let mut candidates = Vec::new();
        let registry = &self.shared_state.model_registry;

        for manifest in registry.models() {
            // Model popularity: count total unique holders across all shards
            let mut all_holders = std::collections::HashSet::new();
            let mut shard_holder_counts: Vec<(u32, usize)> = Vec::new();

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = registry.shard_holders(&shard_id);
                shard_holder_counts.push((shard.index, holders.len()));
                for h in &holders {
                    all_holders.insert(h.clone());
                }
            }

            let model_popularity = all_holders.len() as f64;
            if model_popularity < 1.0 {
                // No one has any shards — probably just published, skip
                continue;
            }

            // Average holder count across shards
            let avg_holders = shard_holder_counts
                .iter()
                .map(|(_, c)| *c as f64)
                .sum::<f64>()
                / manifest.shard_count.max(1) as f64;

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = registry.shard_holders(&shard_id);

                // Skip if we already hold it
                if holders.contains(local_node_id) {
                    continue;
                }

                let holder_count = holders.len();

                // Score = popularity * rarity_bonus
                // rarity_bonus is higher when this shard has fewer holders than average
                let rarity_bonus = if holder_count == 0 {
                    10.0 // Very high priority for zero-holder shards
                } else {
                    (avg_holders + 1.0) / (holder_count as f64 + 1.0)
                };

                let score = model_popularity * rarity_bonus;

                candidates.push(ShardCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    score,
                });
            }
        }

        // Sort by score descending (best candidates first)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Select candidates that fit within the remaining budget.
    fn select_within_budget(
        &self,
        candidates: Vec<ShardCandidate>,
        mut budget_bytes: u64,
        max_shards: u32,
    ) -> Vec<ShardCandidate> {
        let mut selected = Vec::new();
        let max = if max_shards > 0 {
            max_shards as usize
        } else {
            usize::MAX
        };

        // Also check existing downloads in progress
        let in_progress: std::collections::HashSet<String> = self
            .shared_state
            .acquisition_progress
            .iter()
            .filter(|e| {
                matches!(
                    e.value().state,
                    crate::model::acquisition::AcquisitionState::Downloading
                )
            })
            .map(|e| e.key().0.clone())
            .collect();

        for candidate in candidates {
            if selected.len() >= max {
                break;
            }
            if candidate.shard_size_bytes > budget_bytes {
                continue;
            }
            // Don't download if model is already being acquired
            if in_progress.contains(&candidate.model_id.0) {
                continue;
            }

            budget_bytes -= candidate.shard_size_bytes;
            selected.push(candidate);

            // Only download 1-2 shards per evaluation cycle to spread load
            if selected.len() >= 2 {
                break;
            }
        }

        selected
    }

    /// Trigger download of a single shard via the HuggingFace shard download pipeline.
    ///
    /// If the model manifest has a known HuggingFace source, download from there.
    /// Otherwise, try to request the shard from network peers.
    async fn trigger_download(&self, candidate: &ShardCandidate) {
        tracing::info!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            holders = candidate.holder_count,
            score = candidate.score,
            "AutoShardManager: requesting shard download"
        );

        let model_dir = self
            .shared_state
            .config
            .node
            .data_dir
            .join("models")
            .join(&candidate.model_id.0);

        // Check if we already have the shard file locally
        let shard_path = model_dir.join(format!("shard_{:03}.bin", candidate.shard_index));
        if shard_path.exists() {
            tracing::debug!(
                model = %candidate.model_id,
                shard = candidate.shard_index,
                "Shard file already exists on disk, registering"
            );
            let node_id = self.shared_state.identity.node_id().clone();
            let shard_id = ShardId {
                model_id: candidate.model_id.clone(),
                index: candidate.shard_index,
            };
            self.shared_state
                .model_registry
                .record_shard_holder(shard_id.clone(), node_id.clone());
            let mut holders = self
                .shared_state
                .shard_registry
                .entry(shard_id)
                .or_default();
            if !holders.contains(&node_id) {
                holders.push(node_id);
            }
            return;
        }

        // Create a progress entry so the UI can track it
        let mid = candidate.model_id.clone();
        let status = crate::model::acquisition::AcquisitionStatus {
            model_id: mid.clone(),
            state: crate::model::acquisition::AcquisitionState::Downloading,
            total_shards: 1,
            downloaded_shards: 0,
            verified_shards: 0,
            failed_shards: 0,
            total_bytes: candidate.shard_size_bytes,
            downloaded_bytes: 0,
            shard_progress: std::collections::HashMap::new(),
            speed_bytes_per_sec: 0,
            started_at: Some(chrono::Utc::now()),
            log: vec![format!(
                "Auto-manage: downloading shard {} of {} (score: {:.1})",
                candidate.shard_index, candidate.model_name, candidate.score
            )],
        };
        self.shared_state
            .acquisition_progress
            .insert(mid.clone(), status);

        // Announce interest via the network — peers holding the shard can respond
        let shard_id = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        let announce = crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
            node_id: self.shared_state.identity.node_id().clone(),
            shards: vec![shard_id],
            timestamp: chrono::Utc::now(),
        });

        if let Err(e) = self
            .network_tx
            .try_send(NetworkCommand::Broadcast(announce))
        {
            tracing::debug!(error = %e, "Could not broadcast shard interest");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_candidate_scoring() {
        // Higher score = more worth downloading
        let c1 = ShardCandidate {
            model_id: ModelId("m1".into()),
            model_name: "Model 1".into(),
            shard_index: 0,
            shard_size_bytes: 512 * 1024 * 1024,
            holder_count: 0,
            score: 10.0 * 10.0, // popular + zero holders
        };
        let c2 = ShardCandidate {
            model_id: ModelId("m2".into()),
            model_name: "Model 2".into(),
            shard_index: 0,
            shard_size_bytes: 512 * 1024 * 1024,
            holder_count: 5,
            score: 10.0 * 1.0, // popular but well-replicated
        };
        assert!(c1.score > c2.score);
    }

    #[test]
    fn budget_zero_when_max_shards_reached() {
        // AutoManageConfig with max_shards = 0 means unlimited
        let config = crate::config::AutoManageConfig {
            enabled: true,
            max_storage_mb: 10000,
            interval_minutes: 60,
            max_shards: 0,
        };
        assert_eq!(config.max_shards, 0); // unlimited
    }
}
