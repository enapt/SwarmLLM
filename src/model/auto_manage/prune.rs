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

/// Saturation factor for `contribution_auto`. When `holder_count >=
/// SATURATION_FACTOR_AUTO * target_replicas`, the shard is over-replicated
/// enough that auto-mode bypasses the RELAXED-state +1 nudge and uses the
/// raw target as the prune floor — letting an idle node shed slack at
/// swarm scale without waiting for local pressure to build.
const SATURATION_FACTOR_AUTO: f64 = 1.5;

/// R134.7: predictive-eviction time-window. Models with a real swarm
/// request within this window are protected from pruning regardless of
/// replication ratio. Default 60 min — captures the "might be needed in
/// the next 30 min" intuition from FUTURE_WORK without requiring a
/// dedicated forecasting subsystem. Effectively a 1.5h rolling window
/// once you average across user-perceived "I just used this".
const RECENT_REQUEST_PROTECT_SECS: i64 = 3600;

/// Region-demand EMA (requests/10min, decayed) below which a model counts as
/// "the network isn't asking for this either" for idle VRAM unload. Matches the
/// `ema_rate < 0.1` "no demand" boundary in `geo_target_replicas`.
const IDLE_DEMAND_EMA_THRESHOLD: f64 = 0.1;

/// Multiple of `idle_unload_secs` after which VRAM is reclaimed even for a model
/// the region still wants.
///
/// The demand check is a proxy for "someone may ask us shortly", and it is a
/// weak one — regional demand says nothing about whether requests are reaching
/// THIS node. Without a ceiling it means "keep forever", which is not what any
/// operator reads `idle_unload_secs` to mean. Twelve times the configured idle
/// window (one hour at the 5-minute default) is long enough that a genuinely
/// useful model is never evicted mid-use, and short enough that an unused one
/// cannot hold a card all day.
const IDLE_HARD_UNLOAD_MULTIPLIER: i64 = 12;

/// Idle seconds after which the region-demand reprieve no longer applies.
fn idle_hard_unload_secs(idle_unload_secs: u64) -> i64 {
    // `as i64` would wrap a very large configured window to a negative number,
    // which inverts the check and unloads immediately instead of never.
    i64::try_from(idle_unload_secs)
        .unwrap_or(i64::MAX)
        .saturating_mul(IDLE_HARD_UNLOAD_MULTIPLIER)
}

/// How long a loaded model can be considered idle.
///
/// `since_local` is our own outbound request history, `since_served` is work
/// done for peers, `residency_secs` is how long the worker has existed.
///
/// **A model with no request history is not idle for ever.** It cannot have
/// been idle longer than it has been loaded, so residency is the bound when
/// nothing else is known. That is a fact rather than a worst-case guess, and
/// getting it wrong evicted a model seven seconds after it loaded, killing the
/// request that had just loaded it: `last_request_at` is written only by the
/// distributed executor, so a locally-served model never records one and fell
/// through to "assume maximally idle" (reported 2026-07-31).
///
/// Returns `None` only when nothing at all is known — no history and no
/// residency — which callers must treat as "do not judge", not as idle.
fn effective_idle_secs(
    since_local: Option<i64>,
    since_served: Option<i64>,
    residency_secs: Option<u64>,
) -> Option<i64> {
    let observed = match (since_local, since_served) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    observed.or_else(|| residency_secs.map(|s| s.min(i64::MAX as u64) as i64))
}

/// R134.7: penalty applied when a model has a recent swarm request.
/// Combined with the existing `region_demand` penalty this means a
/// model that's actively being used by THIS node is much harder to
/// prune than one that's used elsewhere in the region.
const RECENT_REQUEST_PENALTY: f64 = 1.5;

/// Severe-saturation factor. Holder counts at or above this multiple of
/// the target get a flat +1 score bonus so they always outrank just-
/// barely-saturated shards in selection.
const SATURATION_FACTOR_SEVERE: f64 = 2.0;

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

        // SEC: cache pool shard pins ONCE per prune cycle, AND use the
        // blocking read variant. The previous per-shard re-read used
        // `pool_state.try_read()` — on lock contention (e.g. during
        // pool-membership gossip processing) it returned an empty Vec,
        // silently unblocking the prune of pinned shards (data-loss). The
        // blocking read awaits the lock; treating "contended" as "no pins"
        // is unsafe here. Caching before the loop also gives a consistent
        // snapshot — pin state can't flip mid-cycle.
        let shard_pins_cached = super::manager::read_shard_pins_blocking(&self.shared_state).await;

        // R110: snapshot the set of shards currently in flight on at least
        // one active pipeline. Pruning a shard mid-inference would cause
        // the next forward to fail (a `ShardNotFound` mid-token-loop). We
        // already register every assigned segment in `active_pipelines`
        // when the router dispatches; cross-referencing here is cheap
        // (single pass over a typically <10-entry DashMap). Keys are
        // `ShardId` so per-shard membership is O(1) below.
        let mut active_pipeline_shards: std::collections::HashSet<ShardId> =
            std::collections::HashSet::new();
        for entry in self.shared_state.active_pipelines.iter() {
            for seg in &entry.value().segments {
                if seg.node_id == local_node_id {
                    // Every shard the segment reads, not just its first —
                    // otherwise prune can evict a shard mid-inference.
                    for shard_id in self
                        .shared_state
                        .model_registry
                        .shards_spanned_by_segment(seg)
                    {
                        active_pipeline_shards.insert(shard_id);
                    }
                }
            }
        }
        if !active_pipeline_shards.is_empty() {
            tracing::debug!(
                count = active_pipeline_shards.len(),
                "DIAG: prune guarding active-pipeline shards"
            );
        }

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

            // Compute target replicas for this model (unified with download
            // path). The per-shard prune target is computed inside the shard
            // loop via `effective_prune_target`, which handles both the
            // pressure-adjusted nudge and the contribution-auto saturation
            // override — at swarm scale an idle node sheds slack without
            // waiting for local pressure to build.
            let target = self.geo_target_replicas(&manifest.id, config.min_replicas, pool_size);
            // Read from the AtomicBool, not `config.node.contribution_auto`
            // — the latter is startup-frozen because `state.config` is an
            // Arc that's never swapped. PUT /api/admin/config updates the
            // atomic so the toggle takes effect on the next prune tick.
            let contribution_auto = self
                .shared_state
                .models
                .contribution_auto
                .load(std::sync::atomic::Ordering::Relaxed);

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
                // SEC: count only LIVE holders (gotcha #86 / scheduler-liveness
                // oracle pattern). `shard_holders` returns the gossip-cached
                // list including peers that have been offline for hours
                // (registry entries persist until LRU eviction or explicit
                // remove). The previous logic counted offline peers toward
                // `holder_count`, so a shard whose 3 cached holders were all
                // disconnected passed the `holder_count <= adjusted_target`
                // guard and got pruned — losing the only live copy. Filter
                // against `connected_node_ids` (always include self).
                let holders: Vec<crate::types::NodeId> = holders
                    .into_iter()
                    .filter(|h| {
                        *h == local_node_id || self.shared_state.connected_node_ids.contains(h)
                    })
                    .collect();

                // Skip locked/pinned shards
                if self
                    .shared_state
                    .models
                    .locked_shards
                    .contains_key(&shard_id)
                {
                    continue;
                }

                // Skip shards pinned to this node via pool shard pinning.
                // Uses the cached snapshot from above — see SEC note there.
                {
                    let local_id = self.shared_state.identity.node_id();
                    if shard_pins_cached
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

                // R110: skip shards currently in use on an active pipeline
                // assigned to us. Pruning mid-inference produces a
                // ShardNotFound error halfway through a user's response.
                // The active_pipelines registry is cleaned up by the
                // dispatch path on completion / failure / cancellation,
                // so a "stuck" entry can't pin a shard forever — the
                // router's pipeline TTL sweeper evicts orphans within
                // its own window.
                if active_pipeline_shards.contains(&shard_id) {
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

                // Compute the effective target for THIS shard. In auto mode,
                // a shard over-replicated by ≥SATURATION_FACTOR_AUTO×target
                // bypasses the RELAXED nudge and uses the raw target — so a
                // node with zero local pressure still sheds shards once the
                // swarm has plenty. Severe saturation (≥SATURATION_FACTOR_
                // SEVERE×target) gets a score bonus below to break ties
                // against not-quite-as-saturated shards.
                let effective_target = effective_prune_target(
                    target,
                    resource_pressure,
                    holder_count,
                    contribution_auto,
                    config.min_replicas,
                );

                // Skip if at or below effective target
                if holder_count <= effective_target as usize {
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

                // Compute prune score (higher = more prunable). The
                // redundancy denominator stays as `effective_target`, but
                // the numerator uses the uncapped DHT-reported global count
                // when available — the local cached `holder_count` is bounded
                // by MAX_HOLDERS_PER_SHARD (=50), so at swarm scale the score
                // would saturate well before reaching truly over-replicated
                // shards. Fall back to the filtered live count when no DHT
                // response has landed yet (cold start, etc.).
                let global_count = registry
                    .global_holder_count(&shard_id)
                    .map(|c| c as usize)
                    .map(|c| c.max(holder_count))
                    .unwrap_or(holder_count);
                let redundancy_ratio = global_count as f64 / effective_target.max(1) as f64;
                let mut score = redundancy_ratio;

                // Severe-saturation bonus: shed shards held by ≥2×target
                // first when auto mode picks between multiple eligible
                // shards. The redundancy_ratio already grows with holder
                // count, but this adds a flat tier-break so a 2×target
                // shard always outranks a 1.6×target shard at the
                // selection step. Uses the same uncapped global count so
                // the bonus fires for truly severe over-replication at
                // scale, not just whenever the 50-cap is pegged.
                if contribution_auto
                    && (global_count as f64) >= (target as f64) * SATURATION_FACTOR_SEVERE
                {
                    score += 1.0;
                }

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

                // R134.7: predictive-eviction time-window. Protect shards
                // whose model had a real swarm request within the last
                // `RECENT_REQUEST_PROTECT_SECS`. The user's "I might need
                // this in the next 30 min" intuition translates directly
                // to "I used it recently"; rather than build a separate
                // forecasting subsystem we lean on the existing
                // `model_trust.last_request_at` signal that's already
                // updated per request.
                if let Some(trust) = self.shared_state.models.model_trust.get(&manifest.id) {
                    if let Some(last_req) = trust.last_request_at {
                        let age = (chrono::Utc::now() - last_req).num_seconds();
                        if (0..RECENT_REQUEST_PROTECT_SECS).contains(&age) {
                            score -= RECENT_REQUEST_PENALTY;
                        }
                    }
                }

                prune_candidates.push(PruneCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    target_replicas: effective_target,
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
                // SEC: also filter against connected_node_ids — mirrors the
                // regular shard path above. Without this, gossip-cached offline
                // peers inflate `mmproj_holder_count`, candidates get scored
                // against a wrong replica picture, and the activity log
                // reports counts that don't match real availability. The
                // execution-time re-check at lines 528-541 catches the
                // pre-delete race, but skipping work + correct logs are
                // worth the consistency.
                let mmproj_holders: Vec<crate::types::NodeId> = mmproj_holders
                    .into_iter()
                    .filter(|h| {
                        *h == local_node_id || self.shared_state.connected_node_ids.contains(h)
                    })
                    .collect();
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
                let h = crate::pool::scope::filter_allowed_holders(h, &allowed_set);
                // SEC: same liveness filter as candidate-collection loop.
                // Counting offline holders here would let the re-check pass
                // even if every other holder went down between selection
                // and execution.
                h.into_iter()
                    .filter(|peer| {
                        *peer == local_node_id
                            || self.shared_state.connected_node_ids.contains(peer)
                    })
                    .count()
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
            // Re-check is_shard_in_progress BEFORE deletion. The scan-time
            // skip at line 142 catches the static case, but a P2P download
            // for this shard could have started between scan and execute —
            // deleting now would race the in-progress write and could leave
            // a half-loaded file under the canonical path. The candidate
            // collection already drops actively-downloading shards; this
            // is the second-pass guard for the scan→execute window.
            if self
                .shared_state
                .models
                .is_shard_in_progress(&candidate.model_id, candidate.shard_index)
            {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Skipping prune — shard download started after scan"
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
                // Use the freshly-recomputed holder count (`current_holders`)
                // not `candidate.holder_count` (scan-time snapshot) — concurrent
                // prunes elsewhere on the network can shift the count between
                // scan and execute, and this event drives the dashboard's
                // "remaining replicas" display + persisted prune history.
                holder_count_before: current_holders,
                holder_count_after: current_holders.saturating_sub(1),
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
    /// VRAM efficiency (2026-07) — free the GPU memory of any loaded model that
    /// has had no local requests for `idle_unload_secs` AND shows negligible
    /// network demand. Shards stay on disk (the worker re-spawns on the next
    /// request) and our holder status is unchanged, so this only reclaims VRAM
    /// pinned for a model nobody is currently using. Runs every cycle,
    /// independent of memory pressure. `0` disables it.
    pub(super) async fn try_idle_vram_unload(&self, idle_unload_secs: u64) {
        if idle_unload_secs == 0 {
            return;
        }
        let pool = &self.shared_state.model_process_pool;
        let loaded = pool.loaded_model_ids();
        if loaded.is_empty() {
            return;
        }

        // Never unload a model with an in-flight request on this node —
        // whether this node ORIGINATED it (`active_pipelines`) or is SERVING it
        // for a peer (`serving_models`).
        //
        // Only the first was checked until 2026-07-28, when a soak caught this
        // killing a worker with two peer requests mid-generate. `active_pipelines`
        // is the coordinator's map: a node that only ever answers other people's
        // requests never appears in it, looks idle forever, and eventually trips
        // the hard-unload ceiling while busy.
        let local_node_id = self.shared_state.identity.node_id().clone();
        let mut active_models: std::collections::HashSet<ModelId> =
            std::collections::HashSet::new();
        for entry in self.shared_state.active_pipelines.iter() {
            for seg in &entry.value().segments {
                if seg.node_id == local_node_id {
                    active_models.insert(seg.shard_id.model_id.clone());
                }
            }
        }
        for entry in self.shared_state.serving_models.iter() {
            if entry.value().in_flight > 0 {
                active_models.insert(entry.key().clone());
            }
        }
        // And the authoritative source: the worker pool's own in-flight
        // requests. The two maps above are caller-side bookkeeping and each
        // covers one path — `active_pipelines` is inserted only by the
        // distributed executor, `serving_models` only by peer-served work — so
        // a node answering its OWN client locally was in neither, and its model
        // could be unloaded mid-generation. The pool sees every path.
        for model_id in pool.models_with_inflight_requests() {
            active_models.insert(model_id);
        }

        let our_region = self.our_region().unwrap_or_default();
        let now = chrono::Utc::now();
        let local_id = self.shared_state.identity.node_id().clone();
        let residency: std::collections::HashMap<ModelId, u64> =
            pool.model_residency_secs().into_iter().collect();
        // Pool shard pins snapshot (same source the pressure-prune uses).
        let shard_pins = super::manager::read_shard_pins_blocking(&self.shared_state).await;

        for model_id in loaded {
            if active_models.contains(&model_id) {
                continue;
            }
            // Never idle-unload a DELIBERATELY-held model: user-pinned,
            // pool-pinned, locked, or encrypted-pipeline. Those express an
            // explicit choice by someone.
            //
            // **Reference models are NOT exempt here, deliberately.** They used
            // to be, on the reasoning that the shared cross-swarm benchmark set
            // is "held ON PURPOSE so a consistent model stays warm". That
            // conflates two different things: staying a HOLDER of the shards
            // (on disk, cheap, and what actually keeps the set available to the
            // swarm) versus keeping the model resident in VRAM (expensive, and
            // costing exactly one reload to undo).
            //
            // `is_reference_model` is not consulted anywhere in the disk-prune
            // path, so the shards were never at risk from this loop — the
            // exemption only ever pinned memory. And because the project
            // actively encourages fetching that set (`swarmllm get-model`), the
            // effect was that following the project's own advice permanently
            // cost a user their GPU: once anything touched those models they
            // were resident for the life of the process, with no time bound at
            // all.
            //
            // Observed on the development machine 2026-08-04: two of three
            // resident models were reference models, the GPU sat at 7990 of
            // 8192 MiB, and nothing had been released for two days. A benchmark
            // can afford a cold load; a user's desktop cannot afford a
            // permanently full GPU.
            // Deliberate holds are a REPRIEVE, not a permanent exemption.
            //
            // Every one of these is a statement about keeping the model's
            // SHARDS — pinned by the user, pinned by a pool owner, locked, or
            // flagged for the encrypted pipeline. None of them says "and keep it
            // in memory for ever", but that is what an unconditional `continue`
            // meant: a single pin anywhere permanently removed that model's VRAM
            // from the machine, with no time bound.
            //
            // Memory must always come back eventually and stay inside the
            // configured limits, whatever the model is for. So these delay the
            // unload exactly as regional demand does below, and the same hard
            // ceiling (`idle_hard_unload_secs`, 12x the idle window) overrides
            // them. A pinned model stays warm while the machine is quiet and is
            // still released before it can hold a card all day; the shards are
            // untouched either way, so the pin keeps doing its actual job.
            let deliberately_held = self
                .shared_state
                .models
                .model_trust
                .get(&model_id)
                .map(|t| t.pinned_by_user)
                .unwrap_or(false)
                || self
                    .shared_state
                    .encrypted_pipeline_models
                    .get(&model_id)
                    .map(|v| *v)
                    .unwrap_or(false)
                || self
                    .shared_state
                    .models
                    .locked_shards
                    .iter()
                    .any(|e| e.key().model_id == model_id)
                || shard_pins
                    .iter()
                    .any(|p| p.model_id == model_id.0 && p.target_node_id == local_id);
            // Idle: no local request within the window (never-requested counts
            // as idle — it was loaded but has served nothing).
            let last_req = self
                .shared_state
                .models
                .model_trust
                .get(&model_id)
                .and_then(|t| t.last_request_at);
            // Most recent activity of ANY kind — a request this node made, or
            // one it served for a peer. Serving was invisible here until
            // 2026-07-28, which is why "regional demand" below had to stand in
            // for it: a pure-server node otherwise looks permanently idle.
            let since_local = last_req.map(|t| (now - t).num_seconds());
            let since_served = self
                .shared_state
                .serving_models
                .get(&model_id)
                .map(|s| s.last_served_at.elapsed().as_secs().min(i64::MAX as u64) as i64);
            let since_any =
                effective_idle_secs(since_local, since_served, residency.get(&model_id).copied());
            let idle = since_any.map_or(true, |s| s >= idle_unload_secs as i64);
            if !idle {
                continue;
            }
            // Past a much longer idle period, keep the VRAM rather than the
            // model. The demand check below keeps a wanted model warm while this
            // node is momentarily quiet, which is right — but it measures what
            // the REGION wants in the abstract, not whether anyone is asking US:
            // `last_request_at` is only set by our own outbound requests, never
            // by serving a peer. So a model nobody ever asks us for stays
            // resident indefinitely whenever regional demand sits above the
            // threshold, and on a small card that starves the owner's own work.
            // Reported externally 2026-07-27: two models resident 2h16 past their
            // last request, both with regional demand barely over the line
            // (0.107 and 0.126 against a 0.1 threshold), on a node that then hit
            // GPU-OOM.
            let hard_idle =
                since_any.map_or(true, |s| s >= idle_hard_unload_secs(idle_unload_secs));
            // Low network demand: the region isn't asking for it either. Keep a
            // wanted model warm even when THIS node is momentarily quiet.
            let ema = self
                .shared_state
                .region_demand
                .get(&(model_id.clone(), our_region.clone()))
                .map(|v| *v)
                .unwrap_or(0.0);
            if (ema >= IDLE_DEMAND_EMA_THRESHOLD || deliberately_held) && !hard_idle {
                continue;
            }

            // Free the VRAM; shards remain on disk for a fast re-spawn.
            pool.unload_and_clear_window(&model_id).await;
            let mname = self
                .shared_state
                .model_registry
                .get_manifest(&model_id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| model_id.0.clone());
            // Report the idle time actually observed, not the configured
            // threshold, and do not claim "GPU"/"VRAM" on a node that has
            // neither — a CPU-only node logging "Idle VRAM unload ...
            // idle_unload_secs=300" for a model loaded seconds earlier sent a
            // reporter looking for a GPU fault that did not exist.
            let device = if self.shared_state.gpu_info.is_some() {
                "graphics memory"
            } else {
                "system memory"
            };
            tracing::info!(
                model = %model_id,
                idle_secs = since_any.unwrap_or(-1),
                threshold_secs = idle_unload_secs,
                device,
                "Idle unload — freed model (shards kept on disk)"
            );
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "auto_manage",
                    "idle_vram_unload",
                    format!("Freed idle model {mname} from {device} — no recent requests"),
                )
                .with_model(&model_id.0),
            );
            self.shared_state
                .signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);
        }
    }

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

/// Effective prune target for a given shard, applying both pressure adjustment
/// and the contribution-auto saturation override.
///
/// When `contribution_auto` is true and the shard is over-replicated by
/// ≥SATURATION_FACTOR_AUTO×target, the function returns `target.max(min_replicas)`
/// — bypassing the RELAXED-state +1 nudge so an idle node sheds slack at
/// swarm scale. Otherwise it falls through to `pressure_adjusted_target`.
pub(crate) fn effective_prune_target(
    target: u32,
    pressure: f64,
    holder_count: usize,
    contribution_auto: bool,
    min_replicas: u32,
) -> u32 {
    let saturated =
        contribution_auto && (holder_count as f64) >= (target as f64) * SATURATION_FACTOR_AUTO;
    if saturated {
        target.max(min_replicas)
    } else {
        pressure_adjusted_target(target, pressure, min_replicas)
    }
}

#[cfg(test)]
mod tests {

    use super::effective_idle_secs;

    /// The reported failure: a model loaded 7s ago, with no request history
    /// because it was served locally, was treated as maximally idle and
    /// unloaded mid-generation. Residency bounds it — 7 seconds, not for ever.
    #[test]
    fn a_freshly_loaded_model_is_not_idle() {
        let idle = effective_idle_secs(None, None, Some(7)).expect("residency bounds it");
        assert_eq!(idle, 7);
        assert!(idle < 300, "must not trip a 300s idle threshold");
    }

    /// Before the fix this case yielded "unknown", and the call site read
    /// unknown as idle. Pinning that it now yields a number.
    #[test]
    fn no_request_history_still_produces_an_answer() {
        assert!(effective_idle_secs(None, None, Some(0)).is_some());
    }

    /// A genuinely old model is still collectable — the fix must not pin
    /// everything in memory for ever.
    #[test]
    fn a_long_resident_unused_model_is_still_idle() {
        let idle = effective_idle_secs(None, None, Some(4000)).unwrap();
        assert!(idle >= 300, "an actually-idle model must still unload");
    }

    /// Real history always wins over residency: a model loaded hours ago but
    /// requested a second ago is busy, not idle.
    #[test]
    fn recent_use_beats_long_residency() {
        assert_eq!(effective_idle_secs(Some(1), None, Some(9999)), Some(1));
        assert_eq!(effective_idle_secs(None, Some(2), Some(9999)), Some(2));
        // Most recent activity of either kind wins.
        assert_eq!(effective_idle_secs(Some(90), Some(3), Some(9999)), Some(3));
    }

    /// Nothing known at all must stay "do not judge" rather than becoming 0.
    #[test]
    fn total_absence_of_information_is_unknown() {
        assert_eq!(effective_idle_secs(None, None, None), None);
    }
    use crate::daemon::state::{ServingGuard, ServingState};

    fn serving_test_state() -> std::sync::Arc<crate::daemon::SharedState> {
        let config = crate::config::Config::default();
        let identity = crate::identity::Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);
        state
    }

    /// A node that only ever answers OTHER people's requests must not read as
    /// idle. Before 2026-07-28 it did: `active_pipelines` is the coordinator's
    /// map and never contains peer-served work, so the hard-unload ceiling
    /// eventually killed a worker mid-answer. Caught by soak, not by review.
    #[test]
    fn serving_a_peer_counts_as_the_model_being_in_use() {
        let state = serving_test_state();
        let mid = crate::types::ModelId("m".into());

        // Nothing in flight, nothing served: genuinely idle.
        assert!(state.serving_models.get(&mid).is_none());

        {
            let _g = ServingGuard::new(&state, mid.clone());
            let e = state.serving_models.get(&mid).expect("guard registers");
            assert_eq!(e.in_flight, 1, "a peer request must mark the model busy");
        }

        // Guard dropped: no longer in flight, but recently served.
        let e = state
            .serving_models
            .get(&mid)
            .expect("entry survives the guard");
        assert_eq!(e.in_flight, 0, "in-flight must fall back to zero on drop");
        assert!(
            e.last_served_at.elapsed().as_secs() < 5,
            "last_served_at must be fresh so the idle timer sees the activity"
        );
    }

    /// Concurrent peer requests must not let the first one's completion mark
    /// the model idle while the second is still running.
    #[test]
    fn overlapping_peer_requests_keep_the_model_busy_until_the_last_finishes() {
        let state = serving_test_state();
        let mid = crate::types::ModelId("m".into());
        let a = ServingGuard::new(&state, mid.clone());
        let b = ServingGuard::new(&state, mid.clone());
        assert_eq!(state.serving_models.get(&mid).unwrap().in_flight, 2);
        drop(a);
        assert_eq!(
            state.serving_models.get(&mid).unwrap().in_flight,
            1,
            "one finishing must not clear the other"
        );
        drop(b);
        assert_eq!(state.serving_models.get(&mid).unwrap().in_flight, 0);
    }

    #[test]
    fn serving_state_in_flight_never_underflows() {
        let state = serving_test_state();
        let mid = crate::types::ModelId("m".into());
        state.serving_models.insert(
            mid.clone(),
            ServingState {
                in_flight: 0,
                last_served_at: std::time::Instant::now(),
            },
        );
        drop(ServingGuard::new(&state, mid.clone()));
        assert_eq!(state.serving_models.get(&mid).unwrap().in_flight, 0);
    }

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

    #[test]
    fn effective_target_matches_pressure_path_when_not_saturated() {
        // holder_count below SATURATION_FACTOR_AUTO*target → fall through to
        // the existing pressure-adjusted target. Auto mode shouldn't change
        // behaviour for shards that are not over-replicated.
        // target=3, holder=4: 4 < 1.5*3=4.5 → not saturated.
        assert_eq!(effective_prune_target(3, 0.0, 4, true, 1), 4); // RELAXED +1
        assert_eq!(effective_prune_target(3, 0.5, 4, true, 1), 3); // NORMAL
    }

    #[test]
    fn effective_target_drops_relaxed_nudge_when_saturated_auto() {
        // target=3, holder=5: 5 >= 1.5*3=4.5 → saturated. Auto-mode returns
        // raw target (3), not the RELAXED +1 (4). Without saturation override
        // the node would refuse to prune at zero local pressure.
        assert_eq!(effective_prune_target(3, 0.0, 5, true, 1), 3);
        // Same holder count, manual mode: pressure-adjusted path applies.
        assert_eq!(effective_prune_target(3, 0.0, 5, false, 1), 4);
    }

    #[test]
    fn effective_target_extreme_saturation() {
        // target=3, holder=30 (10× over): saturated. Returns target.
        assert_eq!(effective_prune_target(3, 0.0, 30, true, 1), 3);
        // Floors at min_replicas — never drop below it even when saturated.
        assert_eq!(effective_prune_target(3, 0.0, 30, true, 5), 5);
    }

    #[test]
    fn effective_target_floor_at_boundary() {
        // holder_count exactly at 1.5*target counts as saturated.
        // target=4, holder=6: 6 >= 1.5*4=6 → saturated.
        assert_eq!(effective_prune_target(4, 0.0, 6, true, 1), 4);
        // target=4, holder=5: 5 < 6 → not saturated, RELAXED nudge applies.
        assert_eq!(effective_prune_target(4, 0.0, 5, true, 1), 5);
    }

    /// Demand-driven DISK contraction at the default config (FUTURE_WORK
    /// "disk replica contraction"). An idle model (`target` collapses to
    /// `min_replicas` when demand is ~0, since `geo_target_replicas` uses
    /// demand_factor=1.0) that is over-replicated across the swarm sheds its
    /// slack down to `min_replicas` with zero local pressure — and at the
    /// DEFAULT `min_replicas = 2` that floor IS the `IDLE_REPLICA_FLOOR (≥2,
    /// never a single point)` the deferred design specified. So the substance
    /// of demand-driven disk contraction is already realized for the default
    /// deployment; the only unbuilt piece is contracting BELOW a higher
    /// operator-set `min_replicas`, which is intentionally left alone (it would
    /// override the operator's explicit redundancy floor).
    #[test]
    fn idle_over_replicated_contracts_to_floor_never_below() {
        const DEFAULT_MIN_REPLICAS: u32 = 2; // = config::inference::default_min_replicas()
                                             // Idle model: geo_target_replicas collapses target to min_replicas.
        let idle_target = DEFAULT_MIN_REPLICAS;
        // 3, 8, 50 swarm-wide holders with zero local pressure, auto mode:
        // each sheds down to the floor of 2 — never to a single point.
        for holders in [3usize, 8, 50] {
            let eff = effective_prune_target(idle_target, 0.0, holders, true, DEFAULT_MIN_REPLICAS);
            assert_eq!(
                eff, DEFAULT_MIN_REPLICAS,
                "idle over-replicated ({holders} holders) contracts to the 2-holder floor"
            );
        }
        // A higher operator-set floor is respected — an idle target collapses
        // to min_replicas (=4 here), and contraction never dips below the
        // redundancy floor the operator explicitly chose.
        assert_eq!(effective_prune_target(4, 0.0, 50, true, 4), 4);
    }

    /// R134.7: predictive-eviction constants stay consistent.
    /// `RECENT_REQUEST_PENALTY` must out-weigh the largest regional
    /// demand bonus (1.0) so a single recent request beats even high
    /// regional demand at protecting against eviction.
    #[test]
    fn predictive_eviction_constants_consistent() {
        const _: () = assert!(RECENT_REQUEST_PROTECT_SECS > 0);
        // The penalty must be strictly larger than the strongest
        // region-demand penalty (1.0) so a recent local request
        // dominates region-level signal for the local node.
        const _: () = assert!(RECENT_REQUEST_PENALTY > 1.0);
        // 60 minutes is the conservative ceiling that captures the
        // 30-minute "I might use this soon" intuition; if this grows
        // we should also re-examine the prune cooldown interactions.
        const _: () = assert!(RECENT_REQUEST_PROTECT_SECS <= 2 * 3600);
    }

    /// R134.7: simulates the scoring branch — calling the same logic
    /// the prune loop runs in isolation. Verifies that a fresh
    /// `last_request_at` shaves >=1.0 from the prune score relative
    /// to a stale one, regardless of the rest of the surroundings.
    #[test]
    fn recent_request_subtracts_penalty() {
        let fresh = chrono::Utc::now() - chrono::Duration::seconds(30);
        let stale = chrono::Utc::now() - chrono::Duration::seconds(86_400);
        let fresh_age = (chrono::Utc::now() - fresh).num_seconds();
        let stale_age = (chrono::Utc::now() - stale).num_seconds();
        assert!((0..RECENT_REQUEST_PROTECT_SECS).contains(&fresh_age));
        assert!(stale_age >= RECENT_REQUEST_PROTECT_SECS);

        let mut score_with_fresh = 10.0;
        let mut score_with_stale = 10.0;
        if (0..RECENT_REQUEST_PROTECT_SECS).contains(&fresh_age) {
            score_with_fresh -= RECENT_REQUEST_PENALTY;
        }
        if (0..RECENT_REQUEST_PROTECT_SECS).contains(&stale_age) {
            score_with_stale -= RECENT_REQUEST_PENALTY;
        }
        assert!(score_with_fresh < score_with_stale - 1.0);
    }
}

#[cfg(test)]
mod idle_hard_unload_tests {
    use super::{idle_hard_unload_secs, IDLE_DEMAND_EMA_THRESHOLD, IDLE_HARD_UNLOAD_MULTIPLIER};

    /// The gate as the loop now evaluates it: demand OR a deliberate hold buys
    /// a reprieve, and the hard ceiling overrides both.
    fn keeps_reprieve(ema: f64, deliberately_held: bool, idle_secs: i64, window: u64) -> bool {
        let hard_idle = idle_secs >= idle_hard_unload_secs(window);
        (ema >= IDLE_DEMAND_EMA_THRESHOLD || deliberately_held) && !hard_idle
    }

    /// **Memory always comes back.** A pinned, locked, pool-pinned or
    /// encrypted-pipeline model used to `continue` unconditionally, so a single
    /// pin anywhere removed that model's VRAM from the machine permanently.
    /// Those flags are about keeping SHARDS, never about residency, and the
    /// shards are untouched by this loop either way.
    #[test]
    fn a_deliberately_held_model_is_still_released_at_the_ceiling() {
        let window = 300;
        let past_ceiling = idle_hard_unload_secs(window) + 1;
        assert!(
            !keeps_reprieve(0.0, true, past_ceiling, window),
            "a held model must still be released once past the hard ceiling — \
             memory has to come back whatever the model is for"
        );
        assert!(
            !keeps_reprieve(9.9, true, past_ceiling, window),
            "not even demand AND a pin may hold memory past the ceiling"
        );
    }

    /// The reprieve must still do its job while the machine is merely quiet,
    /// or pinning a model would stop meaning anything at all.
    #[test]
    fn a_deliberate_hold_still_keeps_a_briefly_idle_model_warm() {
        let window = 300;
        let briefly_idle = 600; // past the idle window, far under the ceiling
        assert!(
            keeps_reprieve(0.0, true, briefly_idle, window),
            "a pinned model should stay warm while the node is quiet"
        );
        assert!(
            !keeps_reprieve(0.0, false, briefly_idle, window),
            "an unpinned, unwanted model has nothing holding it and goes"
        );
    }

    /// The reported case: two models resident 2h16 past their last request,
    /// regional demand barely over the line, on a node that then hit GPU-OOM.
    /// At the 5-minute default the reprieve must have expired well before then.
    #[test]
    fn the_reported_two_hour_case_would_now_unload() {
        let ceiling = idle_hard_unload_secs(300);
        assert_eq!(ceiling, 3600, "12x the 5-minute default is one hour");
        let observed_idle_secs = 2 * 3600 + 16 * 60;
        assert!(
            observed_idle_secs >= ceiling,
            "2h16 idle must be past the ceiling, got {ceiling}s"
        );
    }

    /// The demand reprieve must still work for its intended case — a model this
    /// node is quiet on but the region wants, shortly after last use.
    #[test]
    fn a_briefly_idle_wanted_model_keeps_its_reprieve() {
        let ceiling = idle_hard_unload_secs(300);
        let idle_secs = 600; // 10 minutes: past idle_unload_secs, far under the ceiling
        assert!(idle_secs > 300, "past the plain idle window");
        assert!(
            idle_secs < ceiling,
            "but still inside the reprieve, so demand can protect it"
        );
        // And the demand figure from the report would indeed protect it there —
        // read from a variable so this stays a real comparison, not a constant
        // the compiler folds away.
        let reported_regional_demand = 0.107_f64;
        assert!(reported_regional_demand >= IDLE_DEMAND_EMA_THRESHOLD);
    }

    #[test]
    fn the_ceiling_scales_with_the_configured_window() {
        assert_eq!(idle_hard_unload_secs(60), 60 * IDLE_HARD_UNLOAD_MULTIPLIER);
        assert_eq!(
            idle_hard_unload_secs(1800),
            1800 * IDLE_HARD_UNLOAD_MULTIPLIER
        );
    }

    /// `idle_unload_secs = 0` disables the feature before any of this is
    /// reached, but the arithmetic must not misbehave regardless.
    #[test]
    fn absurd_windows_do_not_overflow() {
        assert_eq!(idle_hard_unload_secs(0), 0);
        assert_eq!(idle_hard_unload_secs(u64::MAX), i64::MAX);
    }
}
