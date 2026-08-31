use crate::types::{ModelId, NodeId, ShardId};

use super::manager::{hash_ring_position, AutoShardManager, ShardCandidate};
use super::vram::estimate_model_vram_mb_arch;

/// How much more a shard is worth when it EXTENDS a run this node already
/// holds, and when it CLOSES A HOLE in one.
///
/// Sized against the signals it sits beside: `rarity_bonus` spans 1-10x and
/// must stay in charge of WHERE a node starts on a model, so these are
/// deliberately smaller than its full range — enough to steer which way a
/// holding grows, not enough to drag a node onto a model the swarm already
/// covers well. `source_bonus` and `PARALLAX_ACQUIRE_BONUS` are both 1.5x, so
/// extending sits alongside them and closing a hole is worth twice that.
///
/// Gap-filling scores highest because it removes TWO pipeline hops rather than
/// one: a hole in a holder's range forces the chain out to another machine and
/// back again, and each crossing is a network round trip per token.
const CONTIGUITY_EXTEND_BONUS: f64 = 1.5;
const CONTIGUITY_GAP_FILL_BONUS: f64 = 3.0;

impl AutoShardManager {
    /// Compute remaining download budget in bytes.
    pub(super) fn remaining_budget_bytes(
        &self,
        config: &crate::config::AutoManageConfig,
        local_node_id: &NodeId,
    ) -> u64 {
        // Sum up bytes of shards we already hold (O(local_shards) via reverse index)
        let (current_bytes, current_shard_count) = self.local_shard_bytes(local_node_id);

        // Check max_shards limit
        if config.max_shards > 0 && current_shard_count >= config.max_shards {
            return 0;
        }

        let effective_max = super::compute_budget_max_bytes(
            config.max_storage_mb,
            self.shared_state.cfg().resources.max_disk_mb,
            &self.shared_state.contribution(),
            super::free_disk_bytes_for(&self.shared_state.config.node.data_dir),
        );
        effective_max.saturating_sub(current_bytes)
    }

    /// Gather all candidate shards we don't already hold, scored by value.
    ///
    /// Scoring factors:
    /// - **configured_bonus** (100x): shards in our `--shards` range missing from disk
    /// - **rarity_bonus** (1-10x): fewer holders -> higher priority
    /// - **popularity**: more unique holders across model -> higher value
    /// - **vram_fitness** (0.1-1.0x): models that fit in global VRAM pool score higher
    /// - **spread_bonus** (0.05-1.0x): deprioritizes models we already have many shards of
    pub(super) fn gather_candidates(
        &self,
        local_node_id: &NodeId,
        pool_vram_mb: u64,
    ) -> Vec<ShardCandidate> {
        let mut candidates = Vec::new();
        let registry = &self.shared_state.model_registry;
        let shard_store = self.shared_state.shard_store();
        let configured_range = self.shared_state.config.inference.shard_range;
        let default_cap = self
            .shared_state
            .models
            .auto_manage_default_model_cap
            .load(std::sync::atomic::Ordering::Relaxed);
        let min_replicas = self.shared_state.cfg().auto_manage.min_replicas as usize;
        // pool_size: in private mode, only count allowed nodes; otherwise all peers + self
        let pool_size = crate::pool::scope::effective_pool_size(&self.shared_state);
        // Allowed node set for private mode holder filtering (None = unrestricted)
        let allowed_set = crate::pool::scope::allowed_node_set(&self.shared_state);
        // Shard pins from pool state (for scoring bonus)
        let shard_pins = super::manager::read_shard_pins(&self.shared_state);

        // Build consistent hash ring ONCE for the entire evaluation cycle.
        // Each node gets VIRTUAL_SLOTS positions. Ring is sorted for binary search.
        //
        // SEC: build the ring from `connected_node_ids` rather than
        // `peer_registry`. Per gotcha #86 (scheduler liveness oracle),
        // `peer_registry` is intentionally preserved across mid-pipeline
        // disconnects for reconnect attempts; counting those entries here
        // gives stale shard ownership, possibly causing this node to skip
        // a shard it should claim while the "other holder" is offline.
        const VIRTUAL_SLOTS: u32 = 10;
        let hash_ring: Vec<(u32, NodeId)> = {
            let mut ring = Vec::with_capacity(pool_size * VIRTUAL_SLOTS as usize);
            for vn in 0..VIRTUAL_SLOTS {
                let pos = hash_ring_position(&local_node_id.0, vn);
                ring.push((pos, local_node_id.clone()));
            }
            for peer in self.shared_state.connected_node_ids.iter() {
                if peer.key() == local_node_id {
                    continue; // local already added above
                }
                for vn in 0..VIRTUAL_SLOTS {
                    let pos = hash_ring_position(&peer.key().0, vn);
                    ring.push((pos, peer.key().clone()));
                }
            }
            ring.sort_by_key(|(pos, _)| *pos);
            ring
        };

        for manifest in registry.models() {
            // Never acquire a backup-copy model (`<model>.FULLBACKUP`). Even if a
            // stale manifest reached the registry (e.g. an old redb record from
            // before the register_manifest guard existed), auto-manage must not
            // re-download it from a peer OR its hf_source — the exact live re-fetch
            // reported on v0.3.15. Defense-in-depth beyond load_from_db's purge.
            if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
                continue;
            }
            // -- Policy gate: skip models excluded from auto-manage --
            if let Some(policy) = self
                .shared_state
                .models
                .model_auto_manage_policies
                .get(&manifest.id)
            {
                if !policy.enabled {
                    tracing::debug!(
                        model = %manifest.id,
                        "Skipping model — auto-manage disabled by policy"
                    );
                    continue;
                }
            }

            // -- Trust gate: skip models not yet verified for auto-propagation --
            // Exception: if this node already hosts at least one shard, always
            // allow gap-filling regardless of trust level. Only new-model adoption
            // (zero local shards) requires explicit trust / user pinning.
            {
                let trust = self.shared_state.models.model_trust.get(&manifest.id);
                let trust_level = trust
                    .as_ref()
                    .map(|t| &t.trust_level)
                    .unwrap_or(&crate::types::ModelTrustLevel::Discovered);
                let is_pinned = trust.as_ref().map(|t| t.pinned_by_user).unwrap_or(false);
                let already_hosting = manifest.shards.iter().any(|s| {
                    let sid = ShardId {
                        model_id: manifest.id.clone(),
                        index: s.index,
                    };
                    self.shared_state
                        .model_registry
                        .shard_holders(&sid)
                        .contains(local_node_id)
                });
                if *trust_level < crate::types::ModelTrustLevel::DemandVerified
                    && !is_pinned
                    && !already_hosting
                {
                    tracing::debug!(
                        model = %manifest.id,
                        trust = %trust_level,
                        "Skipping model — insufficient trust for auto-manage"
                    );
                    continue;
                }
            }

            // -- Per-model cap: count local shards, skip if at cap --
            let local_shard_count = manifest
                .shards
                .iter()
                .filter(|s| {
                    let sid = ShardId {
                        model_id: manifest.id.clone(),
                        index: s.index,
                    };
                    registry.shard_holders(&sid).contains(local_node_id)
                })
                .count() as u32;

            // Which shard indices we already hold, for the contiguity bonus
            // below. Computed once per model rather than per shard.
            let local_indices: std::collections::HashSet<u32> = manifest
                .shards
                .iter()
                .filter(|s| {
                    let sid = ShardId {
                        model_id: manifest.id.clone(),
                        index: s.index,
                    };
                    registry.shard_holders(&sid).contains(local_node_id)
                })
                .map(|s| s.index)
                .collect();

            let effective_cap = self
                .shared_state
                .models
                .model_auto_manage_policies
                .get(&manifest.id)
                .and_then(|p| {
                    if p.max_shards > 0 {
                        Some(p.max_shards)
                    } else {
                        None
                    }
                })
                .or(if default_cap > 0 {
                    Some(default_cap)
                } else {
                    None
                });

            if let Some(cap) = effective_cap {
                if local_shard_count >= cap {
                    tracing::debug!(
                        model = %manifest.id,
                        local = local_shard_count,
                        cap = cap,
                        "Skipping model — at per-model shard cap"
                    );
                    continue;
                }
            }

            // -- Spread bonus: deprioritize models we already have many shards of --
            let local_fraction = if manifest.shard_count > 0 {
                local_shard_count as f64 / manifest.shard_count as f64
            } else {
                0.0
            };
            let spread_bonus = (1.0 - local_fraction).max(0.05);

            // Model popularity: count total unique holders across all shards
            let mut all_holders = std::collections::HashSet::new();
            let mut shard_holder_counts: Vec<(u32, usize)> = Vec::new();

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = registry.shard_holders(&shard_id);
                // Private mode: only count holders within allowed set
                let holders = crate::pool::scope::filter_allowed_holders(holders, &allowed_set);
                shard_holder_counts.push((shard.index, holders.len()));
                for h in &holders {
                    all_holders.insert(h.clone());
                }
            }

            // Popularity = number of unique nodes holding any shard of this model.
            // A value of 0 means no one has shards yet (manifest just arrived via gossip).
            // We still want to acquire shards for it if we have an HF source -- this is
            // the "complete the set" flow where one node downloads and others follow.
            let model_popularity = (all_holders.len() as f64).max(1.0);

            // VRAM fitness: does the global pool have enough VRAM to actually run this model?
            // Don't block downloads, but deprioritize models the pool can't run yet.
            // Use MoE-aware estimation so Mixtral/DeepSeek models get accurate VRAM estimates.
            let model_vram_needed =
                estimate_model_vram_mb_arch(manifest.total_size_bytes, &manifest.architecture);
            let vram_fitness = if pool_vram_mb == 0 {
                0.5 // No GPU info available, neutral score
            } else if model_vram_needed <= pool_vram_mb {
                1.0 // Model fits in pool -- full priority
            } else {
                // Model too large for current pool: scale down but don't zero out
                // ratio < 1.0 -> the bigger the gap, the lower the score
                let ratio = pool_vram_mb as f64 / model_vram_needed as f64;
                ratio.max(0.1) // Floor at 0.1x so it's still possible, just deprioritized
            };

            if vram_fitness < 1.0 {
                tracing::debug!(
                    model = %manifest.id,
                    model_vram_mb = model_vram_needed,
                    pool_vram_mb = pool_vram_mb,
                    vram_fitness = vram_fitness,
                    "Model exceeds current pool VRAM — deprioritizing"
                );
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
                // Private mode: only count holders within allowed set
                let holders = crate::pool::scope::filter_allowed_holders(holders, &allowed_set);

                // Skip if we already hold it (both in registry AND on disk)
                if holders.contains(local_node_id)
                    && shard_store.shard_path(&manifest.id, shard.index).exists()
                {
                    continue;
                }

                // Count peers actively downloading this shard -- treat them as
                // near-holders so we don't duplicate their work.
                let peer_dl_count = self
                    .shared_state
                    .models
                    .peer_shard_downloads
                    .get(&shard_id)
                    .map(|v| match allowed_set {
                        Some(ref allowed) => v.iter().filter(|(n, _)| allowed.contains(n)).count(),
                        None => v.len(),
                    })
                    .unwrap_or(0);
                let holder_count = holders.len() + peer_dl_count;

                // Check if this shard is in our configured --shards range but missing
                let in_configured_range = match configured_range {
                    Some((start, end)) => shard.index >= start && shard.index <= end,
                    None => false,
                };

                // Compute target replicas using the unified geo-aware formula.
                // Same calculation used by both download and prune paths.
                let target_replicas =
                    self.geo_target_replicas(&manifest.id, min_replicas as u32, pool_size) as usize;

                // Skip shards that already meet the replica target.
                // Using >= target_replicas (not > 0) ensures min_replicas drives
                // replication: each shard is spread across target_replicas nodes.
                if holder_count >= target_replicas && !in_configured_range {
                    tracing::debug!(
                        model = %manifest.id,
                        shard = shard.index,
                        holders = holder_count,
                        target = target_replicas,
                        "Skipping shard — replica target met"
                    );
                    continue;
                }

                // Consistent hash ring deduplication: use the pre-built ring to
                // determine if this node is responsible for downloading this shard.
                // On join/leave, only ~1/pool_size of assignments change.
                // BYPASS: if we already host other shards of this model, always allow
                // gap-filling so partial models get completed for local inference.
                // SEC: use connected_node_ids (gotcha #86) — peer_registry
                // includes recently-disconnected peers preserved for reconnect.
                let peers = self.shared_state.connected_node_ids.len();
                let already_hosting_model = local_shard_count > 0;
                if holder_count < target_replicas
                    && peers > 0
                    && !in_configured_range
                    && !already_hosting_model
                {
                    let replicas_needed =
                        (target_replicas as u32).saturating_sub(holder_count as u32);

                    let ring = &hash_ring;
                    let i_am_assigned = (0..replicas_needed).any(|replica_idx| {
                        let target_pos = crate::types::hash_parts_to_u32(&[
                            manifest.id.0.as_bytes(),
                            &shard.index.to_le_bytes(),
                            &replica_idx.to_le_bytes(),
                        ]);
                        // Binary search for the nearest node clockwise on the ring
                        let idx = match ring.binary_search_by_key(&target_pos, |(p, _)| *p) {
                            Ok(i) => i,
                            Err(i) => i % ring.len().max(1),
                        };
                        ring.get(idx)
                            .map(|(_, n)| n == local_node_id)
                            .unwrap_or(false)
                    });
                    if !i_am_assigned {
                        tracing::debug!(
                            model = %manifest.id,
                            shard = shard.index,
                            replicas_needed,
                            "Skipping shard — not assigned on consistent hash ring"
                        );
                        continue;
                    }
                }

                // Skip shards already being downloaded on THIS node (explicit or
                // auto-manage). Prevents racing with an in-flight download.
                if self
                    .shared_state
                    .models
                    .is_shard_in_progress(&manifest.id, shard.index)
                {
                    tracing::debug!(
                        model = %manifest.id,
                        shard = shard.index,
                        "Skipping shard — already downloading on this node"
                    );
                    continue;
                }

                // Skip shards in download backoff. A recently-failed/stalled
                // download is cooling down (exponential backoff); re-selecting
                // it immediately is what let one stuck shard monopolize the
                // download semaphore and starve fresh candidates a user was
                // waiting on (external report, 2026-07-23).
                if self.shared_state.models.shard_in_backoff(&shard_id) {
                    tracing::debug!(
                        model = %manifest.id,
                        shard = shard.index,
                        "Skipping shard — in download backoff after a recent failure"
                    );
                    continue;
                }

                // Regional rarity: prefer shards missing from our region.
                // Look up regional holder count from gossip summaries, falling back
                // to counting peers in the registry.
                let our_region = self.our_region();
                let regional_holders = if let Some(ref region) = our_region {
                    let key = (region.clone(), manifest.id.clone());
                    if let Some(summary) = self.shared_state.region_shard_summaries.get(&key) {
                        summary
                            .shard_counts
                            .iter()
                            .find(|(idx, _)| *idx == shard.index)
                            .map(|(_, c)| *c as usize)
                            .unwrap_or(0)
                    } else {
                        // Fallback: count from peer_registry
                        self.count_regional_holders(&holders, local_node_id, region)
                    }
                } else {
                    holder_count // No region -- use global count
                };

                // Per-region minimum: popular models need at least 2 copies per region.
                // Derive from target_replicas: if target >= 2x min_replicas, model is popular.
                let per_region_min: usize = if target_replicas >= (min_replicas * 2) {
                    2
                } else {
                    1
                };

                let regional_rarity = if regional_holders == 0 {
                    20.0 // No regional coverage -- very high priority
                } else if regional_holders < per_region_min {
                    10.0 / (regional_holders as f64 + 1.0)
                } else {
                    // Standard global rarity
                    (avg_holders + 1.0) / (holder_count as f64 + 1.0)
                };

                // Source bonus: prefer same-region peers or HF CDN.
                let source_bonus = if our_region.is_some() {
                    let has_regional_peer = holders.iter().any(|h| {
                        if let Some(peer) = self.shared_state.peer_registry.get(h) {
                            if let Some(ref cap) = peer.capability {
                                if let Some(ref r) = cap.region {
                                    return r.to_uppercase() == our_region.as_deref().unwrap_or("");
                                }
                            }
                        }
                        false
                    });
                    if has_regional_peer {
                        1.5
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                // -- Contiguity: prefer a shard that EXTENDS what we already hold --
                //
                // **A pipeline pays a network round trip every time it changes
                // machine, so where a node's shards sit matters as much as how
                // many it has.** Scoring each shard only on rarity scatters a
                // node's holdings across the model, and a scattered holder
                // makes the pipeline leave and come back. Measured on the live
                // swarm 2026-08-31 for a 48-layer model: one peer held layers
                // 0-8 AND 12-47 but not the four between, so the chain went
                // peer → other → same peer → third — four hops and four WAN
                // round trips per token, against a possible two. The tester
                // measured ~1.5 tok/s, most of it round trip rather than
                // compute.
                //
                // Petals treats this as an invariant rather than a preference:
                // "the interval of blocks allocated to each server is always
                // contiguous, since splitting it would harm the inference
                // latency", and a joining server picks a contiguous interval
                // positioned where the swarm is weakest — rarity and
                // contiguity together, not traded off (arXiv 2209.01188).
                //
                // We keep rarity in charge of WHERE to start and let this
                // decide WHICH WAY TO GROW: neutral when we hold nothing of a
                // model, so a first acquisition is unbiased; a bonus for
                // extending a run; the largest bonus for closing a hole,
                // because that removes two hops rather than one.
                let contiguity_bonus = if local_indices.is_empty() {
                    1.0
                } else {
                    // `checked_sub`, not `wrapping_sub`: shard 0 has no shard
                    // below it, and wrapping would let a (nonexistent) u32::MAX
                    // holding read as its neighbour.
                    let below = shard
                        .index
                        .checked_sub(1)
                        .is_some_and(|i| local_indices.contains(&i));
                    let above = local_indices.contains(&(shard.index + 1));
                    match (below, above) {
                        (true, true) => CONTIGUITY_GAP_FILL_BONUS,
                        (true, false) | (false, true) => CONTIGUITY_EXTEND_BONUS,
                        (false, false) => 1.0,
                    }
                };

                let configured_bonus = if in_configured_range { 100.0 } else { 1.0 };

                // Shard pinning bonus: if this shard is pinned to us, massive bonus.
                // If pinned to another node, score 0 (skip it).
                let pins_for_model: Vec<_> = shard_pins
                    .iter()
                    .filter(|p| p.model_id == manifest.id.0)
                    .collect();
                let pinned_to_us = pins_for_model
                    .iter()
                    .any(|p| p.matches(&manifest.id.0, local_node_id, shard.index));
                let pinned_to_other = pins_for_model.iter().any(|p| {
                    p.target_node_id != *local_node_id
                        && p.matches_shard(&manifest.id.0, shard.index)
                });
                let pin_bonus = if pins_for_model.is_empty() {
                    1.0 // No pins for this model
                } else if pinned_to_us {
                    1000.0 // Massive bonus — download this shard
                } else if pinned_to_other {
                    0.0 // Pinned elsewhere — don't download
                } else {
                    1.0 // Not pinned
                };

                if pin_bonus == 0.0 {
                    continue; // Skip shards pinned to other nodes
                }

                // The operator deleted this shard from this node by hand. That
                // is an instruction, not a gap to fill: re-downloading it erased
                // a deliberate two-machine split with no message (external
                // report, 2026-08-21). Only an equally explicit instruction
                // outranks it — a configured `shard_range` that claims it, or
                // a pool pin naming this device. Forgotten when the user asks
                // for the shard again (`clear_shard_removed_by_user`).
                if !in_configured_range
                    && !pinned_to_us
                    && self.shared_state.shard_removed_by_user(&shard_id)
                {
                    tracing::debug!(
                        model = %manifest.id,
                        shard = shard.index,
                        "Skipping shard — removed by the user; not re-acquiring on our own"
                    );
                    continue;
                }

                // Node-specific jitter (0.0-0.1) so nodes with identical views
                // of the network don't all pick the same shard to download.
                // BLAKE3(node_id || model_id || shard_index) -> deterministic per-node tiebreaker.
                let jitter = {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(&local_node_id.0);
                    hasher.update(manifest.id.0.as_bytes());
                    hasher.update(&shard.index.to_le_bytes());
                    let hash = hasher.finalize();
                    hash.as_bytes()[0] as f64 / 2550.0 // 0.0-0.1 range
                };

                // Phase C.2 Parallax bonus: the allocator has consistently
                // placed some part of this shard's layer range on us across
                // the stability window. Small multiplicative boost so the
                // allocator's view of balanced pipeline coverage nudges
                // scoring without overriding the existing rarity/popularity/
                // configured-range signals.
                let parallax_bonus = if self.parallax_should_boost_acquire(&shard_id) {
                    super::parallax::PARALLAX_ACQUIRE_BONUS
                } else {
                    1.0
                };

                let score = model_popularity
                    * regional_rarity
                    * configured_bonus
                    * contiguity_bonus
                    * pin_bonus
                    * vram_fitness
                    * spread_bonus
                    * source_bonus
                    * parallax_bonus
                    + jitter;

                candidates.push(ShardCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    score,
                });
            }

            // -- T7: mmproj as download candidate for VLM models --
            if let Some(ref mmproj_info) = manifest.mmproj {
                let mmproj_shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: crate::types::MMPROJ_SHARD_INDEX,
                };
                let mmproj_holders = registry.shard_holders(&mmproj_shard_id);
                let mmproj_path = crate::model::shard::model_dir(
                    &self.shared_state.config.node.data_dir,
                    &manifest.id.0,
                )
                .join(crate::model::shard::MMPROJ_FILENAME);

                if !mmproj_holders.contains(local_node_id) || !mmproj_path.exists() {
                    let holder_count = mmproj_holders.len();
                    // mmproj gets a high priority bonus -- every VLM node benefits from having it
                    let rarity_bonus = if holder_count == 0 {
                        10.0
                    } else {
                        3.0 / (holder_count as f64 + 1.0)
                    };
                    let mmproj_score = model_popularity * rarity_bonus * vram_fitness * 5.0; // 5x bonus for mmproj

                    candidates.push(ShardCandidate {
                        model_id: manifest.id.clone(),
                        model_name: manifest.name.clone(),
                        shard_index: crate::types::MMPROJ_SHARD_INDEX,
                        shard_size_bytes: mmproj_info.size_bytes,
                        holder_count,
                        score: mmproj_score,
                    });
                }
            }
        }

        // Sort by score descending (best candidates first)
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // If any configured-range shards are missing for the primary model,
        // focus exclusively on those first. Don't download extra shards until
        // our assigned range is complete.
        if let Some((start, end)) = configured_range {
            // Find the model that matches our configured shard range
            // (the one we were started with via --model + --shards)
            let configured_model: Option<ModelId> = registry
                .models()
                .iter()
                .find(|m| m.shards.iter().any(|s| s.index >= start && s.index <= end))
                .map(|m| m.id.clone());

            if let Some(ref mid) = configured_model {
                let has_configured_missing = candidates
                    .iter()
                    .any(|c| c.model_id == *mid && c.shard_index >= start && c.shard_index <= end);
                if has_configured_missing {
                    candidates.retain(|c| {
                        c.model_id == *mid && c.shard_index >= start && c.shard_index <= end
                    });
                }
            }
        }

        candidates
    }

    /// Select candidates that fit within the remaining budget.
    pub(super) fn select_within_budget(
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

        // Per-cycle cap scaled by ContributionMode:
        // Minimal: 1 shard/cycle regardless of network size
        // Moderate: 1-2 based on peer count (original behavior)
        // Maximum: 2-4 for aggressive seeding
        let peers = self.shared_state.peer_registry.len();
        let per_cycle_cap = match self.shared_state.contribution() {
            swarmllm_types::ContributionMode::Minimal => 1,
            swarmllm_types::ContributionMode::Moderate => {
                if peers < 5 {
                    1
                } else {
                    2
                }
            }
            swarmllm_types::ContributionMode::Maximum => {
                if peers < 5 {
                    2
                } else {
                    4
                }
            }
        };

        // Track which specific shards are currently downloading so we don't
        // start a duplicate. We check per-shard progress, NOT per-model -- otherwise
        // downloading shard 0 would block acquisition of shard 1.
        let downloading_shards: std::collections::HashSet<(String, u32)> = {
            let mut set = std::collections::HashSet::new();
            for entry in self.shared_state.models.acquisition_progress.iter() {
                if !matches!(
                    entry.value().state,
                    crate::model::acquisition::AcquisitionState::Downloading
                ) {
                    continue;
                }
                let mid = entry.key().0.clone();
                for (&idx, sp) in &entry.value().shard_progress {
                    if matches!(sp.state, crate::model::acquisition::ShardState::Downloading) {
                        set.insert((mid.clone(), idx));
                    }
                }
            }
            set
        };

        // Round-robin interleaving: group candidates by model, take one per model
        // in rotation instead of pure score-descending. This ensures shards from
        // different models are downloaded in the same cycle when possible.
        let mut by_model: std::collections::HashMap<String, Vec<ShardCandidate>> =
            std::collections::HashMap::new();
        let mut model_order: Vec<String> = Vec::new();
        for candidate in candidates {
            // Skip this specific shard if it's already being downloaded
            if downloading_shards.contains(&(candidate.model_id.0.clone(), candidate.shard_index)) {
                continue;
            }
            if !by_model.contains_key(&candidate.model_id.0) {
                model_order.push(candidate.model_id.0.clone());
            }
            by_model
                .entry(candidate.model_id.0.clone())
                .or_default()
                .push(candidate);
        }

        // Round-robin: take one candidate from each model in order
        let mut model_indices: Vec<usize> = vec![0; model_order.len()];
        'outer: loop {
            let mut any_taken = false;
            for (mi, model_key) in model_order.iter().enumerate() {
                if selected.len() >= max || selected.len() >= per_cycle_cap {
                    break 'outer;
                }
                let candidates_for_model = &by_model[model_key];
                while model_indices[mi] < candidates_for_model.len() {
                    let candidate = &candidates_for_model[model_indices[mi]];
                    model_indices[mi] += 1;
                    // SEC: skip candidates with zero `shard_size_bytes`. The
                    // field comes from `ShardInfo.size_bytes` populated from
                    // peer-gossiped manifests OR HF API responses where the
                    // file size may be missing (`unwrap_or(0)`). A zero-size
                    // candidate ALWAYS passes `<= budget_bytes` regardless
                    // of remaining budget, then subtracts 0 — silently
                    // letting unbounded zero-size shards through the budget
                    // gate. Treat unknown size as a refusal signal instead.
                    if candidate.shard_size_bytes == 0 {
                        tracing::debug!(
                            model = %candidate.model_id,
                            shard = candidate.shard_index,
                            "Auto-manage: skipping candidate with zero shard_size_bytes (unknown size)"
                        );
                        continue;
                    }
                    if candidate.shard_size_bytes <= budget_bytes {
                        budget_bytes -= candidate.shard_size_bytes;
                        selected.push(candidate.clone());
                        any_taken = true;
                        break; // move to next model
                    }
                }
            }
            if !any_taken {
                break;
            }
        }

        selected
    }

    /// Unified target replicas: log-scaled floor x demand factor.
    /// Used by BOTH download (gather_candidates) and prune (evaluate_and_prune) paths.
    ///
    /// | Pool size | Floor | Popular (3x) | Idle (1x) |
    /// |-----------|-------|-------------|-----------|
    /// | 10        | 3     | 9           | 3         |
    /// | 100       | 7     | 21          | 7         |
    /// | 1,000     | 10    | 30          | 10        |
    /// | 10,000    | 13    | 39          | 13        |
    pub(super) fn geo_target_replicas(
        &self,
        model_id: &ModelId,
        min_replicas: u32,
        pool_size: usize,
    ) -> u32 {
        let global_floor = if pool_size <= 1 {
            min_replicas as usize
        } else {
            let log2_pool = (pool_size as f64).log2().ceil() as usize;
            let upper = (pool_size / 3).max(min_replicas as usize);
            log2_pool.clamp(min_replicas as usize, upper).max(1)
        };

        // Use EMA demand from region_demand (smoothed) if available,
        // falling back to raw request counter.
        let our_region = self.our_region().unwrap_or_default();
        let demand_key = (model_id.clone(), our_region);
        let ema_rate = self
            .shared_state
            .region_demand
            .get(&demand_key)
            .map(|v| *v)
            .unwrap_or(0.0);

        // Convert EMA rate to demand factor. The EMA rate is in requests/10min
        // (decayed), so thresholds are lower than raw counts.
        let demand_factor = if ema_rate < 0.1 {
            // Also check raw counter as fallback (covers first window before decay kicks in)
            let raw = self
                .shared_state
                .models
                .model_request_counts
                .get(model_id)
                .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
            match raw {
                0 => 1.0,
                1..=5 => 1.5,
                6..=20 => 2.0,
                21..=100 => 2.5,
                _ => 3.0,
            }
        } else if ema_rate < 1.0 {
            1.5
        } else if ema_rate < 5.0 {
            2.0
        } else if ema_rate < 20.0 {
            2.5
        } else {
            3.0
        };

        let global_target = (global_floor as f64 * demand_factor).ceil() as u32;
        global_target.clamp(min_replicas, (pool_size as u32).max(min_replicas))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        make_test_manager, make_test_manager_with_config, register_manifest_with_shards,
    };
    use crate::config::Config;
    use crate::types::{NodeId, ShardId};

    /// Local holds shard 0 of a two-shard model, a connected peer holds shard 1:
    /// the gap-filling rule makes shard 1 a candidate. That is exactly the
    /// path that re-downloaded what a user had deleted to split a model across
    /// two machines (external report, 2026-08-21).
    fn split_model(
        state: &std::sync::Arc<crate::daemon::SharedState>,
    ) -> (NodeId, ShardId, ShardId) {
        // No per-model cap: the test is about the removal, not the cap.
        state
            .models
            .auto_manage_default_model_cap
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let mid = register_manifest_with_shards(state, "split-model", 16, &[(0, 8), (8, 16)]);
        let local = state.identity.node_id().clone();
        let peer = NodeId([7u8; 32]);
        let s0 = ShardId {
            model_id: mid.clone(),
            index: 0,
        };
        let s1 = ShardId {
            model_id: mid,
            index: 1,
        };
        state
            .model_registry
            .record_shard_holder(s0.clone(), local.clone());
        state
            .model_registry
            .record_shard_holder(s1.clone(), peer.clone());
        // Gotcha #86: connected_node_ids is the liveness oracle.
        state.connected_node_ids.insert(peer);
        (local, s0, s1)
    }

    #[test]
    fn a_shard_the_user_removed_is_not_re_acquired_until_asked_for_again() {
        let (state, manager) = make_test_manager();
        let (local, _s0, s1) = split_model(&state);

        let wants_s1 = |m: &super::AutoShardManager| {
            m.gather_candidates(&local, 0)
                .iter()
                .any(|c| c.model_id == s1.model_id && c.shard_index == 1)
        };
        assert!(
            wants_s1(&manager),
            "control: gap-filling offers the missing shard before any removal"
        );

        state.mark_shard_removed_by_user(&s1);
        assert!(
            !wants_s1(&manager),
            "a removal the user performed is an instruction, not a gap to fill"
        );

        // Asking for it again (HF shard download / single-shard download /
        // a pin to this device all call this) restores the old behaviour.
        assert!(state.clear_shard_removed_by_user(&s1));
        assert!(wants_s1(&manager));
    }

    #[test]
    fn a_configured_shard_range_outranks_a_removal() {
        let mut config = Config::default();
        config.inference.shard_range = Some((0, 1));
        let (state, manager) = make_test_manager_with_config(config);
        let (local, _s0, s1) = split_model(&state);
        state.mark_shard_removed_by_user(&s1);
        assert!(
            manager
                .gather_candidates(&local, 0)
                .iter()
                .any(|c| c.model_id == s1.model_id && c.shard_index == 1),
            "the config explicitly claims this shard, which is itself an instruction"
        );
    }
}

/// Where a node's shards sit matters as much as how many it has: a pipeline
/// pays a network round trip every time it changes machine.
#[cfg(test)]
mod contiguity {
    use std::collections::HashSet;

    /// Reproduces the arithmetic the scoring loop applies, so the ordering
    /// property can be asserted without standing up a whole registry.
    fn bonus(local: &HashSet<u32>, index: u32) -> f64 {
        if local.is_empty() {
            return 1.0;
        }
        let below = index.checked_sub(1).is_some_and(|i| local.contains(&i));
        let above = local.contains(&(index + 1));
        match (below, above) {
            (true, true) => super::CONTIGUITY_GAP_FILL_BONUS,
            (true, false) | (false, true) => super::CONTIGUITY_EXTEND_BONUS,
            (false, false) => 1.0,
        }
    }

    /// The case measured on the live swarm: a peer held layers 0-8 and 12-47
    /// but not the four between, so the pipeline went out to another machine
    /// and back — two extra WAN round trips per token. Closing that hole is
    /// worth more than extending an end, because it removes two hops not one.
    #[test]
    fn closing_a_hole_beats_extending_a_run() {
        let held: HashSet<u32> = [0u32, 1, 2, 4, 5].into_iter().collect();
        // 3 sits between 2 and 4 — a hole.
        assert!(bonus(&held, 3) > bonus(&held, 6));
        // 6 extends the run upward, which still beats an unrelated shard.
        assert!(bonus(&held, 6) > bonus(&held, 9));
    }

    /// **Rarity stays in charge of where a node starts.** Holding nothing of a
    /// model must leave the score untouched, or a node would be nudged toward
    /// models it has already begun regardless of what the swarm is missing —
    /// which is the opposite of what `rarity_bonus` is for.
    #[test]
    fn a_model_we_hold_nothing_of_is_unbiased() {
        let none: HashSet<u32> = HashSet::new();
        for i in [0u32, 1, 7, 15] {
            assert_eq!(bonus(&none, i), 1.0);
        }
    }

    /// Shard 0 has no shard below it; the wrapping subtraction must not make
    /// `u32::MAX` look like a neighbour we hold.
    #[test]
    fn shard_zero_does_not_wrap_into_a_false_neighbour() {
        let held: HashSet<u32> = [u32::MAX].into_iter().collect();
        assert_eq!(bonus(&held, 0), 1.0);
    }

    /// It steers, it does not dominate: a contiguous shard must not outrank the
    /// 1-10x rarity signal at its top end, or a node would fill its own holes
    /// in preference to a shard nobody in the swarm holds at all.
    #[test]
    fn contiguity_cannot_outweigh_a_shard_nobody_holds() {
        let held: HashSet<u32> = [0u32, 2].into_iter().collect();
        let contiguous_common = bonus(&held, 1) * 1.0; // gap fill, plentiful
        let scattered_rare = 1.0 * 10.0; // no contiguity, rarest possible
        assert!(scattered_rare > contiguous_common);
    }
}
