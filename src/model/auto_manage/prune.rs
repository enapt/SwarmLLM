use std::time::Duration;

use chrono::Timelike;

use crate::types::{ModelId, NodeId, ShardId};

use super::manager::{AutoShardManager, PruneCandidate};
use super::vram::query_gpu_vram_used;

/// Protection window — a shard acquired within the last `SHARD_RECENTLY_ACQUIRED_SECS`
/// gets a scoring penalty so it's not pruned immediately after download.
const SHARD_RECENTLY_ACQUIRED_SECS: u64 = 1800;

/// Resource pressure thresholds for pruning decisions.
/// Relaxed: add a spare replica. Normal: no change. Eager: shed 1. Urgent: shed 2.
const PRESSURE_RELAXED: f64 = 0.5;
const PRESSURE_NORMAL: f64 = 0.8;
const PRESSURE_URGENT: f64 = 0.95;
/// VRAM soft-unload trigger — try narrowing shard windows before deleting files.
const PRESSURE_SOFT_UNLOAD: f64 = 0.7;

impl AutoShardManager {
    /// Evaluate and prune over-replicated shards. Called after downloads in each cycle.
    pub(super) async fn evaluate_and_prune(&self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.prune_enabled {
            return;
        }

        let local_node_id = self.shared_state.identity.node_id().clone();
        let registry = &self.shared_state.model_registry;
        let shard_store = self.shared_state.shard_store();

        // Pre-fetch VRAM usage off the Tokio thread (nvidia-smi is blocking I/O)
        let live_vram_used = tokio::task::spawn_blocking(query_gpu_vram_used)
            .await
            .ok()
            .flatten();

        // Compute resource pressure
        let resource_pressure = self.compute_resource_pressure(live_vram_used);
        let pressure_urgent = resource_pressure > PRESSURE_URGENT;
        tracing::info!(
            resource_pressure = %format_args!("{:.2}", resource_pressure),
            pressure_urgent,
            "DIAG: evaluate_and_prune starting"
        );

        // ── Phase 0: VRAM soft-unload ──
        // When VRAM pressure is moderate (SOFT_UNLOAD..URGENT), try narrowing the shard
        // window for loaded models instead of deleting files. This frees VRAM while
        // keeping shards on disk for the network.
        if resource_pressure > PRESSURE_SOFT_UNLOAD && !pressure_urgent {
            self.try_vram_soft_unload(resource_pressure).await;
        }

        // Check if we're in reduced hours
        let schedule_pressure = self.schedule_pressure_bonus().await;

        let pool_size = crate::pool::scope::effective_pool_size(&self.shared_state);
        // Private mode: allowed node set for holder filtering (None = unrestricted)
        let allowed_set = crate::pool::scope::allowed_node_set(&self.shared_state);

        // Track how many shards pruned per model in this cycle
        let mut pruned_per_model: std::collections::HashMap<ModelId, u32> =
            std::collections::HashMap::new();

        // Collect prune candidates across all models
        let mut prune_candidates: Vec<PruneCandidate> = Vec::new();

        for manifest in registry.models() {
            // Check per-model prune policy
            if let Some(policy) = self
                .shared_state
                .models
                .model_auto_manage_policies
                .get(&manifest.id)
            {
                if !policy.prune_enabled {
                    continue;
                }
            }

            // Check cooldown
            if let Some(last_prune) = self.last_prune_time(&manifest.id) {
                let elapsed = chrono::Utc::now()
                    .signed_duration_since(last_prune)
                    .num_seconds()
                    .max(0) as u64;
                if elapsed < config.prune_cooldown_secs {
                    continue;
                }
            }

            // Compute target replicas for this model (unified with download path)
            let target = self.geo_target_replicas(&manifest.id, config.min_replicas, pool_size);

            // Adjust target for resource pressure
            let adjusted_target =
                self.pressure_adjusted_target(target, resource_pressure, config.min_replicas);

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };

                // Only consider shards we hold locally
                let holders = registry.shard_holders(&shard_id);
                if !holders.contains(&local_node_id) {
                    continue;
                }
                // Private mode: filter holders to allowed set for replica counting
                let holders = crate::pool::scope::filter_allowed_holders(holders, &allowed_set);

                // Skip locked/pinned shards
                if self
                    .shared_state
                    .models
                    .locked_shards
                    .contains_key(&shard_id)
                {
                    continue;
                }

                // Skip shards pinned to this node via pool shard pinning
                {
                    let local_id = self.shared_state.identity.node_id();
                    let shard_pins = super::manager::read_shard_pins(&self.shared_state);
                    if shard_pins
                        .iter()
                        .any(|p| p.matches(&manifest.id.0, local_id, shard.index))
                    {
                        continue;
                    }
                }

                // Skip shards that are actively being downloaded by this node
                let is_downloading = self
                    .shared_state
                    .models
                    .is_shard_in_progress(&manifest.id, shard.index);
                if is_downloading {
                    continue;
                }

                // Skip shards for models with encrypted pipeline enabled.
                // E2E encryption requires local first/last segments -- pruning
                // any shard of an encrypted model would break the guarantee.
                if self
                    .shared_state
                    .encrypted_pipeline_models
                    .get(&manifest.id)
                    .map(|v| *v)
                    .unwrap_or(false)
                {
                    continue;
                }

                // Skip shards for models the user explicitly pinned/trusted
                if self
                    .shared_state
                    .models
                    .model_trust
                    .get(&manifest.id)
                    .map(|t| t.pinned_by_user)
                    .unwrap_or(false)
                {
                    continue;
                }

                // Skip if in configured --shards range
                if let Some((start, end)) = self.shared_state.config.inference.shard_range {
                    if shard.index >= start && shard.index <= end {
                        continue;
                    }
                }

                let holder_count = holders.len();

                // Skip if at or below target
                if holder_count <= adjusted_target as usize {
                    continue;
                }

                // Region-aware: block if we'd eliminate last holder in our region
                if self.would_eliminate_region(&shard_id, &local_node_id, &holders) {
                    continue;
                }

                // Load-aware: block if remaining holders are busy
                if self.remaining_holders_busy(
                    &holders,
                    &local_node_id,
                    config.max_holder_load_for_prune,
                ) {
                    continue;
                }

                // Skip if model actively loaded and used recently (unless pressure-urgent)
                if !pressure_urgent && self.shard_recently_used(&manifest.id) {
                    continue;
                }

                // Re-acquisition check: can we get it back?
                if !self.can_reacquire(&manifest.id, &shard_id, &holders, &local_node_id) {
                    continue;
                }

                // Compute prune score (higher = more prunable)
                let redundancy_ratio = holder_count as f64 / adjusted_target.max(1) as f64;
                let mut score = redundancy_ratio;

                // Cold shard bonus (not loaded in VRAM)
                let is_loaded = self
                    .shared_state
                    .split_models
                    .iter()
                    .any(|entry| entry.key().0 == manifest.id);
                if !is_loaded {
                    score += 1.0;
                }

                // Resource pressure bonus
                score += 0.5 * (resource_pressure + schedule_pressure);

                // Pipeline completeness penalty
                if shard.index == 0 || shard.index == manifest.shard_count.saturating_sub(1) {
                    score -= 0.5;
                }

                // Rarest shard penalty
                let min_holders = manifest
                    .shards
                    .iter()
                    .map(|s| {
                        let sid = ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        };
                        let h = registry.shard_holders(&sid);
                        crate::pool::scope::count_allowed_holders(&h, &allowed_set)
                    })
                    .min()
                    .unwrap_or(0);
                if holder_count == min_holders {
                    score -= 0.3;
                }

                // Recently acquired penalty (< 30 min)
                // Use file modified time as proxy
                let shard_path = shard_store.shard_path(&manifest.id, shard.index);
                let sp = shard_path.clone();
                let recently_acquired = tokio::task::spawn_blocking(move || {
                    std::fs::metadata(&sp)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map(|age| age < Duration::from_secs(SHARD_RECENTLY_ACQUIRED_SECS))
                        .unwrap_or(false)
                })
                .await
                .unwrap_or(false);
                if recently_acquired {
                    score -= 0.2;
                }

                // Phase C.2 Parallax bonus: the allocator consistently
                // wants this shard off this node across the stability
                // window. Additive, so it stacks with pressure/cold-shard
                // bonuses but still obeys the region / load / reacquire
                // guards above (which already filtered out hard blocks).
                if self.parallax_should_boost_prune(&shard_id) {
                    score += super::parallax::PARALLAX_PRUNE_BONUS;
                }

                // Regional demand penalty: protect shards for models with active
                // demand in our region. Higher demand -> harder to prune.
                {
                    let our_region = self.our_region().unwrap_or_default();
                    if !our_region.is_empty() {
                        let demand_key = (manifest.id.clone(), our_region);
                        let ema_rate = self
                            .shared_state
                            .region_demand
                            .get(&demand_key)
                            .map(|v| *v)
                            .unwrap_or(0.0);
                        if ema_rate > 10.0 {
                            score -= 1.0; // High demand -- strongly resist pruning
                        } else if ema_rate > 1.0 {
                            score -= 0.5; // Moderate demand
                        } else if ema_rate > 0.1 {
                            score -= 0.2; // Low but non-zero demand
                        }
                    }
                }

                prune_candidates.push(PruneCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    target_replicas: adjusted_target,
                    score,
                });
            }

            // -- T9: mmproj pruning with higher min_replicas floor --
            // mmproj needs wider availability for VLM requests, so use a higher floor.
            if manifest.mmproj.is_some() {
                let mmproj_shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: crate::types::MMPROJ_SHARD_INDEX,
                };
                let mmproj_holders = registry.shard_holders(&mmproj_shard_id);
                // Private mode: filter mmproj holders to allowed set
                let mmproj_holders =
                    crate::pool::scope::filter_allowed_holders(mmproj_holders, &allowed_set);
                if mmproj_holders.contains(&local_node_id) {
                    let mmproj_path = crate::model::shard::model_dir(
                        &self.shared_state.config.node.data_dir,
                        &manifest.id.0,
                    )
                    .join(crate::model::shard::MMPROJ_FILENAME);
                    if mmproj_path.exists() {
                        // Higher floor: at least 3 replicas (or pool_size, whichever is smaller)
                        let mmproj_min = (config.min_replicas + 1).min(pool_size as u32).max(3);
                        let mmproj_holder_count = mmproj_holders.len();
                        if mmproj_holder_count > mmproj_min as usize && pressure_urgent {
                            let mmproj_size =
                                manifest.mmproj.as_ref().map(|m| m.size_bytes).unwrap_or(0);
                            prune_candidates.push(PruneCandidate {
                                model_id: manifest.id.clone(),
                                model_name: manifest.name.clone(),
                                shard_index: crate::types::MMPROJ_SHARD_INDEX,
                                shard_size_bytes: mmproj_size,
                                holder_count: mmproj_holder_count,
                                target_replicas: mmproj_min,
                                score: 0.1, // Very low score -- only prune under extreme pressure
                            });
                        }
                    }
                }
            }
        }

        // Sort by score descending (most prunable first)
        prune_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Execute pruning with per-model limits
        let max_per_model = if pressure_urgent { 2u32 } else { 1u32 };

        for candidate in &prune_candidates {
            let count = pruned_per_model
                .get(&candidate.model_id)
                .copied()
                .unwrap_or(0);
            if count >= max_per_model {
                continue;
            }

            // Re-check holder count before deletion to avoid pruning based on stale data.
            // Between candidate collection and execution, new holders may have appeared.
            let shard_id_check = ShardId {
                model_id: candidate.model_id.clone(),
                index: candidate.shard_index,
            };
            let current_holders = {
                let h = registry.shard_holders(&shard_id_check);
                crate::pool::scope::count_allowed_holders(&h, &allowed_set)
            };
            // Re-compute pressure-adjusted target with current pressure as well.
            // As we prune candidates earlier in this cycle, local disk usage
            // drops and pressure-adjusted_target may have grown (i.e., we no
            // longer need to shed). Re-using the snapshot taken at scan time
            // could cause over-pruning. We re-use the cached VRAM read
            // (live_vram_used) — VRAM is not affected by file deletes, only
            // by reload/unload, which doesn't happen mid-cycle.
            let fresh_pressure = self.compute_resource_pressure(live_vram_used);
            let target_now = self.pressure_adjusted_target(
                self.geo_target_replicas(&candidate.model_id, config.min_replicas, pool_size),
                fresh_pressure,
                config.min_replicas,
            );
            let effective_target = target_now.max(candidate.target_replicas);
            // mmproj is added to candidates only when scan-time pressure was
            // urgent. Earlier prune executions in this same loop may have
            // dropped fresh_pressure below PRESSURE_URGENT — in that case
            // the strict mmproj guard no longer holds, so skip rather than
            // prune a 5x-bonus shard under merely soft pressure.
            if candidate.shard_index == crate::types::MMPROJ_SHARD_INDEX
                && fresh_pressure <= PRESSURE_URGENT
            {
                tracing::debug!(
                    model = %candidate.model_id,
                    fresh_pressure = %format_args!("{:.2}", fresh_pressure),
                    "Skipping mmproj prune — fresh pressure no longer urgent (earlier prunes freed disk)"
                );
                continue;
            }
            if current_holders <= effective_target as usize {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    current_holders,
                    target_at_scan = candidate.target_replicas,
                    target_now,
                    fresh_pressure = %format_args!("{:.2}", fresh_pressure),
                    "Skipping prune — holder count or fresh pressure says we no longer need to shed"
                );
                continue;
            }

            // Actually delete the shard file (or mmproj.gguf for sentinel)
            let shard_path = if candidate.shard_index == crate::types::MMPROJ_SHARD_INDEX {
                crate::model::shard::model_dir(
                    &self.shared_state.config.node.data_dir,
                    &candidate.model_id.0,
                )
                .join(crate::model::shard::MMPROJ_FILENAME)
            } else {
                shard_store.shard_path(&candidate.model_id, candidate.shard_index)
            };
            if shard_path.exists() {
                let sp = shard_path.clone();
                let result = tokio::task::spawn_blocking(move || std::fs::remove_file(&sp)).await;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!(
                            model = %candidate.model_id,
                            shard = candidate.shard_index,
                            error = %e,
                            "Failed to delete shard file during pruning"
                        );
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "spawn_blocking join error during prune");
                        continue;
                    }
                }
            }

            // Unregister from shard registry
            let shard_id = ShardId {
                model_id: candidate.model_id.clone(),
                index: candidate.shard_index,
            };
            registry.remove_shard_holder(&shard_id, &local_node_id);
            // S5: Stop providing this shard via DHT
            let _ = self
                .network_tx
                .try_send(crate::types::NetworkCommand::StopProviding(vec![
                    shard_id.clone()
                ]));

            // Count remaining local shards for this model
            let remaining_local = registry
                .models()
                .iter()
                .find(|m| m.id == candidate.model_id)
                .map(|m| {
                    m.shards
                        .iter()
                        .filter(|s| {
                            let sid = ShardId {
                                model_id: m.id.clone(),
                                index: s.index,
                            };
                            registry.shard_holders(&sid).contains(&local_node_id)
                        })
                        .count() as u32
                })
                .unwrap_or(0);

            let event = crate::types::PruneEvent {
                model_id: candidate.model_id.clone(),
                model_name: candidate.model_name.clone(),
                shard_index: candidate.shard_index,
                reason: format!(
                    "Over-replicated ({} holders, target {})",
                    candidate.holder_count, candidate.target_replicas
                ),
                freed_bytes: candidate.shard_size_bytes,
                remaining_local_shards: remaining_local,
                holder_count_before: candidate.holder_count,
                holder_count_after: candidate.holder_count.saturating_sub(1),
                timestamp: chrono::Utc::now(),
            };

            tracing::info!(
                model = %candidate.model_id,
                shard = candidate.shard_index,
                holders = candidate.holder_count,
                target = candidate.target_replicas,
                freed_mb = candidate.shard_size_bytes / (1024 * 1024),
                "Pruning over-replicated shard"
            );

            // If no local shards remain, kill the worker subprocess to free VRAM
            if remaining_local == 0
                && self
                    .shared_state
                    .model_process_pool
                    .is_loaded(&candidate.model_id)
            {
                tracing::info!(
                    model = %candidate.model_id,
                    "No local shards remain after pruning — unloading model worker to free VRAM"
                );
                self.shared_state.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "auto_manage",
                        "model_unloaded",
                        format!(
                            "Unloaded {} worker (no local shards remain after pruning)",
                            candidate.model_name
                        ),
                    )
                    .with_model(&candidate.model_id.0)
                    .with_model_name(&candidate.model_name),
                );
                self.shared_state
                    .model_process_pool
                    .unload_model(&candidate.model_id)
                    .await;
            }

            // Emit unified activity event (replaces separate prune_events_tx)
            self.shared_state
                .signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "auto_manage",
                    "shard_pruned",
                    format!(
                        "Pruned {} of {} — {} holders remain",
                        ShardId::display_index(event.shard_index),
                        event.model_name,
                        event.holder_count_after
                    ),
                )
                .with_model(&event.model_id.0)
                .with_model_name(&event.model_name)
                .with_detail_num(event.freed_bytes as i64)
                .with_detail_str(&event.reason)
                .with_toast("info", 6000)
                .with_shard_index(event.shard_index)
                .with_freed_bytes(event.freed_bytes)
                .with_holders(event.holder_count_before, event.holder_count_after)
                .with_remaining_local(remaining_local)
                .with_timestamp(event.timestamp.to_rfc3339()),
            );

            // Add to history
            {
                let mut history = self.shared_state.models.prune_history.write().await;
                if history.len() >= 100 {
                    history.pop_front();
                }
                history.push_back(event);
            }

            *pruned_per_model
                .entry(candidate.model_id.clone())
                .or_insert(0) += 1;
        }
    }

    /// Adjust target based on resource pressure.
    pub(super) fn pressure_adjusted_target(
        &self,
        target: u32,
        pressure: f64,
        min_replicas: u32,
    ) -> u32 {
        pressure_adjusted_target(target, pressure, min_replicas)
    }

    /// Compute resource pressure (0.0-1.0) based on VRAM and disk usage.
    fn compute_resource_pressure(&self, live_vram_used: Option<u64>) -> f64 {
        let config = &self.shared_state.config;
        let local_node_id = self.shared_state.identity.node_id().clone();

        // Disk pressure
        let budget_mb = if config.auto_manage.max_storage_mb > 0 {
            config.auto_manage.max_storage_mb
        } else {
            config.resources.max_disk_mb / 2
        };
        let (local_bytes, _) = self.local_shard_bytes(&local_node_id);
        let disk_pressure = if budget_mb > 0 {
            local_bytes as f64 / (budget_mb as f64 * 1024.0 * 1024.0)
        } else {
            0.0
        };

        // VRAM pressure -- prefer live nvidia-smi data over internal model tracking
        let vram_pressure = if let Some(ref gpu) = self.shared_state.gpu_info {
            if gpu.vram_total_mb > 0 {
                let used_mb = live_vram_used.unwrap_or_else(|| {
                    // Fallback: sum estimated VRAM of loaded models
                    self.shared_state
                        .split_models
                        .iter()
                        .map(|e| e.value().estimated_vram_mb)
                        .sum()
                });
                used_mb as f64 / gpu.vram_total_mb as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        disk_pressure.max(vram_pressure).min(1.0)
    }

    /// Compute schedule-based pressure bonus during reduced hours.
    async fn schedule_pressure_bonus(&self) -> f64 {
        let schedule = self.shared_state.models.resource_schedule.read().await;
        if !schedule.enabled {
            return 0.0;
        }

        let now_hour = chrono::Utc::now().hour();
        let in_reduced = if schedule.reduced_hours_start <= schedule.reduced_hours_end {
            now_hour >= schedule.reduced_hours_start && now_hour < schedule.reduced_hours_end
        } else {
            // Wraps midnight (e.g., 22-8)
            now_hour >= schedule.reduced_hours_start || now_hour < schedule.reduced_hours_end
        };

        if !in_reduced {
            return 0.0;
        }

        match schedule.prune_aggressiveness.as_str() {
            "aggressive" => 0.3,
            "normal" => 0.15,
            _ => 0.0, // "conservative"
        }
    }

    /// Check if removing us as holder would eliminate the last holder in our region.
    fn would_eliminate_region(
        &self,
        _shard_id: &ShardId,
        local_node_id: &NodeId,
        holders: &[NodeId],
    ) -> bool {
        let our_region = self.our_region().unwrap_or_default();

        if our_region.is_empty() {
            // No region data -- fallback: ensure at least 2 holders with low latency
            let low_latency_remaining = holders
                .iter()
                .filter(|h| {
                    *h != local_node_id
                        && self
                            .shared_state
                            .peer_registry
                            .get(*h)
                            .map(|p| p.latency_ms.unwrap_or(9999) < 200)
                            .unwrap_or(false)
                })
                .count();
            return low_latency_remaining < 2;
        }

        // Count remaining holders in our region (excluding us)
        let same_region_remaining = holders
            .iter()
            .filter(|h| {
                if *h == local_node_id {
                    return false;
                }
                if let Some(peer) = self.shared_state.peer_registry.get(*h) {
                    peer.capability
                        .as_ref()
                        .and_then(|c| c.region.as_deref())
                        .map(|r| r.eq_ignore_ascii_case(&our_region))
                        .unwrap_or(false)
                } else {
                    false
                }
            })
            .count();

        same_region_remaining == 0
    }

    /// Check if remaining holders (excluding us) are too busy.
    fn remaining_holders_busy(
        &self,
        holders: &[NodeId],
        local_node_id: &NodeId,
        max_load: u32,
    ) -> bool {
        let remaining: Vec<u32> = holders
            .iter()
            .filter_map(|h| {
                if h == local_node_id {
                    return None;
                }
                self.shared_state
                    .peer_registry
                    .get(h)
                    .map(|p| p.active_request_count)
            })
            .collect();

        if remaining.is_empty() {
            return true; // No peers to offload to
        }

        let avg_load = remaining.iter().sum::<u32>() as f64 / remaining.len() as f64;
        avg_load > max_load as f64
    }

    /// Check if this model's shards were used recently (last 5 min).
    fn shard_recently_used(&self, model_id: &ModelId) -> bool {
        let now_secs = crate::types::unix_now_secs();
        if let Some(ranges) = self.shared_state.split_model_index.get(model_id) {
            for &(s, e) in ranges.iter() {
                let key = (model_id.clone(), s, e);
                if let Some(entry) = self.shared_state.split_models.get(&key) {
                    let last = entry.value().last_used_secs();
                    if now_secs.saturating_sub(last) < 300 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if we can re-acquire this shard if needed later.
    fn can_reacquire(
        &self,
        model_id: &ModelId,
        _shard_id: &ShardId,
        holders: &[NodeId],
        local_node_id: &NodeId,
    ) -> bool {
        // Check HF source
        if self.shared_state.models.hf_sources.contains_key(model_id) {
            return true;
        }

        // Check if healthy peers hold it (excluding us)
        let peer_holders = holders.iter().any(|h| {
            h != local_node_id
                && self
                    .shared_state
                    .peer_registry
                    .get(h)
                    .map(|p| p.latency_ms.unwrap_or(9999) < 5000)
                    .unwrap_or(false)
        });
        peer_holders
    }

    /// Try to free VRAM by narrowing shard windows on loaded models.
    /// Keeps shards on disk (advertised to the network) but reduces what's in GPU memory.
    async fn try_vram_soft_unload(&self, pressure: f64) {
        let vram_budget_mb = super::vram::compute_vram_budget(&self.shared_state);
        let Some(budget) = vram_budget_mb else {
            return; // No GPU or no budget configured
        };

        // Scale budget by pressure: at SOFT_UNLOAD use 90% of budget, at URGENT use 60%
        let scale = 1.0 - (pressure - PRESSURE_RELAXED).clamp(0.0, 0.5);
        let effective_budget = (budget as f64 * scale) as u64;

        let registry = &self.shared_state.model_registry;
        let pool = &self.shared_state.model_process_pool;

        for model_id in pool.loaded_model_ids() {
            // Skip if already has a window (don't narrow twice per cycle)
            if pool.get_shard_window(&model_id).is_some() {
                continue;
            }

            let Some(manifest) = registry.get_manifest(&model_id) else {
                continue;
            };

            if manifest.shard_count <= 2 {
                continue; // Can't narrow below 2 shards
            }

            let model_vram = super::vram::estimate_model_vram_mb_arch(
                manifest.total_size_bytes,
                &manifest.architecture,
            );
            let shard_vram_each = model_vram / manifest.shard_count as u64;

            if shard_vram_each == 0 {
                continue;
            }

            let window = super::vram::compute_optimal_shard_window(
                manifest.shard_count,
                shard_vram_each,
                effective_budget,
            );

            if let Some(ref w) = window {
                if w.len() < manifest.shard_count as usize {
                    tracing::info!(
                        model = %model_id,
                        total_shards = manifest.shard_count,
                        window_shards = w.len(),
                        pressure = %format_args!("{:.2}", pressure),
                        "VRAM soft-unload: narrowing shard window (shards stay on disk)"
                    );
                    let mname = self
                        .shared_state
                        .model_registry
                        .get_manifest(&model_id)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| model_id.0.clone());
                    self.shared_state.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "auto_manage",
                            "vram_soft_unload",
                            format!(
                                "VRAM pressure: narrowed {} to {} of {} shards",
                                mname,
                                w.len(),
                                manifest.shard_count
                            ),
                        )
                        .with_model(&model_id.0)
                        .with_model_name(&mname)
                        .with_detail_num(w.len() as i64)
                        .with_toast("warning", 5000),
                    );
                    pool.restart_with_window(&model_id, w.clone()).await;
                    // Only do one model per cycle to avoid thundering restart
                    break;
                }
            }
        }
    }

    /// Get last prune time for a model from prune history.
    fn last_prune_time(&self, model_id: &ModelId) -> Option<chrono::DateTime<chrono::Utc>> {
        // Check prune history (we need a sync read, so try_read)
        if let Ok(history) = self.shared_state.models.prune_history.try_read() {
            history.iter().rev().find_map(|e| {
                if e.model_id == *model_id {
                    Some(e.timestamp)
                } else {
                    None
                }
            })
        } else {
            None
        }
    }
}

/// Adjust shard target replicas based on resource pressure (pure function).
/// Relaxed (<RELAXED): keep extras, Normal (<NORMAL): unchanged,
/// Eager (<URGENT): reduce by 1, Urgent (≥URGENT): reduce by 2.
pub(crate) fn pressure_adjusted_target(target: u32, pressure: f64, min_replicas: u32) -> u32 {
    if pressure < PRESSURE_RELAXED {
        target.saturating_add(1)
    } else if pressure < PRESSURE_NORMAL {
        target
    } else if pressure < PRESSURE_URGENT {
        target.saturating_sub(1).max(min_replicas)
    } else {
        target.saturating_sub(2).max(min_replicas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_pressure_adds_one_extra_replica() {
        // < 0.5 pressure: keep more replicas around (network is not
        // hot, so over-replicate slightly for redundancy).
        assert_eq!(pressure_adjusted_target(3, 0.0, 1), 4);
        assert_eq!(pressure_adjusted_target(3, 0.49, 1), 4);
    }

    #[test]
    fn normal_pressure_keeps_target() {
        // 0.5..0.8: replica target stays as-is.
        assert_eq!(pressure_adjusted_target(3, 0.5, 1), 3);
        assert_eq!(pressure_adjusted_target(3, 0.79, 1), 3);
    }

    #[test]
    fn eager_pressure_reduces_by_one() {
        // 0.8..0.95: shed one replica to free disk/VRAM.
        assert_eq!(pressure_adjusted_target(3, 0.80, 1), 2);
        assert_eq!(pressure_adjusted_target(3, 0.94, 1), 2);
    }

    #[test]
    fn urgent_pressure_reduces_by_two() {
        // ≥0.95: shed two replicas — the swarm is at capacity.
        assert_eq!(pressure_adjusted_target(5, 0.95, 1), 3);
        assert_eq!(pressure_adjusted_target(5, 1.0, 1), 3);
    }

    #[test]
    fn pressure_adjustment_floors_at_min_replicas() {
        // Even under urgent pressure, must not drop below min_replicas.
        assert_eq!(pressure_adjusted_target(3, 1.0, 3), 3);
        assert_eq!(pressure_adjusted_target(2, 0.95, 1), 1);
        // mmproj uses min=3 — at urgent pressure with only 2-3 replicas
        // should clamp at 3 not 1.
        assert_eq!(pressure_adjusted_target(3, 1.0, 3), 3);
    }

    #[test]
    fn pressure_adjustment_handles_saturating_sub() {
        // target=1, urgent pressure → saturating_sub(2) = 0 → max(0, min=1) = 1.
        assert_eq!(pressure_adjusted_target(1, 1.0, 1), 1);
        // target=0, any pressure → saturating_sub never goes negative.
        assert_eq!(pressure_adjusted_target(0, 1.0, 0), 0);
    }
}
