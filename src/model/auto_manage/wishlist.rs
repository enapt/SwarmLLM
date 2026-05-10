//! Wishlist — the actionable, ranked list of models the swarm wants.
//!
//! Sits alongside `gather_candidates` (which is shard-level and internal to
//! the auto-manage executor) and rolls everything up to **model level** for
//! the dashboard. Each entry has a status (Hosting / Serveable /
//! Aspirational / Unreachable / Blocked), a 0..100 score, and a list of
//! human-readable "why" tags so non-technical users see exactly *why* the
//! system is interested in a model.
//!
//! Continuously rebuilt — no waiting on an arbitrary tick. Producer:
//! `refresh_wishlist` is called from the auto-manage manager loop AND the
//! WS stats build. The latter ensures the dashboard always has a fresh
//! list even if auto-manage is paused (e.g., user toggled it off but is
//! still browsing what the swarm runs).
//!
//! R111. The HfWatcher in R112 plugs into the same wishlist by enriching
//! entries with HF trending download counts.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::daemon::SharedState;
use crate::types::{ModelId, NodeId, ShardId};

/// Actionable status of a single wishlist entry. Maps directly onto the
/// CTA the dashboard renders, no further interpretation needed.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WishlistStatus {
    /// We host at least one shard of this model.
    Hosting,
    /// Network has every shard at least once; this node could route
    /// inference to it today.
    Serveable,
    /// Some shards exist on the network but coverage is incomplete.
    /// Surfaced so the user sees "the swarm is working on this".
    Aspirational,
    /// Even with everyone helping, no individual node has the VRAM/disk
    /// to host. Effectively unreachable for this swarm size; we still
    /// show it so the user understands the upper bound.
    Unreachable,
    /// Trust gate / private mode / explicit user-ignore — auto-manage
    /// won't act on this without explicit consent.
    #[default]
    Blocked,
}

impl WishlistStatus {
    /// Single-token i18n key used by the frontend to localise the badge.
    /// Keeps copy in en.json, not in this enum.
    pub fn i18n_key(self) -> &'static str {
        match self {
            Self::Hosting => "wishlist.status.hosting",
            Self::Serveable => "wishlist.status.serveable",
            Self::Aspirational => "wishlist.status.aspirational",
            Self::Unreachable => "wishlist.status.unreachable",
            Self::Blocked => "wishlist.status.blocked",
        }
    }
}

/// Single wishlist entry — one per known model. Aggregates per-shard
/// scoring and capacity coverage into a model-level summary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WishlistEntry {
    pub model_id: String,
    /// User-facing display name. Falls back to `model_id` when the
    /// manifest doesn't carry a friendlier label.
    pub display_name: String,
    pub status: WishlistStatus,
    /// 0..100 — purely advisory ranking signal. Aggregated as the maximum
    /// shard-level score we'd compute from `gather_candidates`, normalised.
    /// Frontend renders as a soft heat indicator, not a precise number.
    pub score: u32,
    /// Human-readable reasons we surfaced this entry. Each tag is an i18n
    /// key under `wishlist.why.*` so the frontend localises them. Examples:
    /// `wishlist.why.your_region_needs_this`, `wishlist.why.popular_on_swarm`,
    /// `wishlist.why.fits_your_memory`, `wishlist.why.would_unlock_with_n_more`.
    pub why_tags: Vec<String>,
    /// Distinct nodes in the swarm currently hosting at least one shard.
    pub swarm_replicas: u32,
    /// Recommended replica target — derived from pool size + demand.
    /// 0 means "no minimum" (unranked discovery entry).
    pub target_replicas: u32,
    /// Total model size in MB across all shards (manifest-derived).
    pub size_mb: u64,
    /// Estimated VRAM (MB) needed to load the full model on a single
    /// node. Used in the "fits your memory" framing.
    pub vram_required_mb: u64,
    /// Coverage so the user can see "8 of 12 model parts are on the
    /// network already" without us exposing the word "shard".
    pub shards_covered: u32,
    pub total_shards: u32,
    /// Whether THIS node hosts at least one shard. Drives the "your
    /// contribution" pill on the entry card.
    pub hosted_by_us: bool,
}

/// The full wishlist — ranked, capped at MAX_ENTRIES.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Wishlist {
    pub entries: Vec<WishlistEntry>,
    /// Last time the wishlist was rebuilt (Unix seconds). Frontend uses
    /// this to render "updated 3 seconds ago" so users see liveliness.
    pub computed_at: i64,
}

/// Cap to keep the wishlist tractable for the frontend. Wishlist is a
/// "what to look at next" prompt, not a full catalogue — that's the
/// HF-search view in R114.
pub const MAX_WISHLIST_ENTRIES: usize = 100;

/// Recompute the wishlist and publish it via ArcSwap. Single pass over
/// the model registry; cheap. Caller invokes from the auto-manage tick
/// AND from the WS stats build.
pub fn refresh_wishlist(state: &SharedState) {
    let snapshot = compute_wishlist(state);
    state.models.wishlist.store(Arc::new(snapshot));
}

/// Build a fresh wishlist snapshot. Pure function over registry state.
pub fn compute_wishlist(state: &SharedState) -> Wishlist {
    let local_node_id = state.identity.node_id().clone();
    let local_region: Option<String> = state.config.identity.region.clone();
    let online_node_count = (state.connected_node_ids.len() as u32) + 1;
    let local_vram_mb = state
        .gpu_info
        .as_ref()
        .map(|g| g.vram_total_mb)
        .unwrap_or(0);
    let pool_vram_mb = crate::model::auto_manage::vram::global_pool_vram_mb(state);

    // Gather a quick map of which models are pinned by the user — those
    // entries always pass the trust gate and never get marked Blocked.
    let mut pinned_user: HashMap<ModelId, bool> = HashMap::new();
    for entry in state.models.model_trust.iter() {
        pinned_user.insert(entry.key().clone(), entry.value().pinned_by_user);
    }

    // R112: HF trending join. The HfWatcher caches the latest trending
    // GGUF download counts; we boost wishlist score when our local model
    // matches one of those repos, and add a `wishlist.why.popular_on_hf`
    // tag so users see *why* a model jumped up the list.
    let trending_snapshot = state.models.hf_trending_cache.load_full();
    let mut trending_by_repo: HashMap<String, u64> = HashMap::new();
    for e in &trending_snapshot.entries {
        trending_by_repo.insert(e.repo_id.clone(), e.downloads);
    }
    let mut trending_for_model: HashMap<ModelId, u64> = HashMap::new();
    for src in state.models.hf_sources.iter() {
        if let Some(downloads) = trending_by_repo.get(src.value().repo_id.as_str()) {
            trending_for_model.insert(src.key().clone(), *downloads);
        }
    }

    let mut entries: Vec<WishlistEntry> = Vec::new();

    for manifest in state.model_registry.models() {
        if manifest.shards.is_empty() {
            continue;
        }
        let mid = manifest.id.clone();
        let total_shards = manifest.shards.len() as u32;
        let mut shards_covered: u32 = 0;
        let mut holders: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut hosted_by_us = false;
        let mut local_region_has_a_holder = false;

        for shard in &manifest.shards {
            let sid = ShardId {
                model_id: mid.clone(),
                index: shard.index,
            };
            let live: Vec<NodeId> = state
                .model_registry
                .shard_holders(&sid)
                .into_iter()
                .filter(|h| h == &local_node_id || state.connected_node_ids.contains(h))
                .collect();
            if live.is_empty() {
                continue;
            }
            shards_covered += 1;
            for h in &live {
                if h == &local_node_id {
                    hosted_by_us = true;
                }
                holders.insert(h.clone());
                // Region match — used to drive the "your region needs
                // this" tag below. Cheap because we only check until we
                // find one.
                if let Some(ref my_region) = local_region {
                    if !local_region_has_a_holder {
                        if let Some(peer) = state.peer_registry.get(h) {
                            if let Some(cap) = peer.value().capability.as_ref() {
                                if let Some(ref r) = cap.region {
                                    if r.eq_ignore_ascii_case(my_region) {
                                        local_region_has_a_holder = true;
                                    }
                                }
                            }
                        } else if h == &local_node_id {
                            local_region_has_a_holder = true;
                        }
                    }
                }
            }
        }

        // Status classification — drives both colour + CTA on the card.
        let status = if shards_covered == total_shards {
            if hosted_by_us {
                WishlistStatus::Hosting
            } else {
                WishlistStatus::Serveable
            }
        } else if shards_covered > 0 {
            WishlistStatus::Aspirational
        } else if pinned_user.get(&mid).copied().unwrap_or(false) {
            // User-pinned but zero coverage — unblock as Aspirational
            // so we surface the user's interest even before any peer
            // joins with a shard.
            WishlistStatus::Aspirational
        } else {
            // Trust-gated; keep visible but with the "blocked" CTA.
            WishlistStatus::Blocked
        };

        // VRAM estimate — MoE-aware via existing helper.
        let vram_required_mb = crate::model::auto_manage::vram::estimate_model_vram_mb_arch(
            manifest.total_size_bytes,
            &manifest.architecture,
        );
        let size_mb = manifest.total_size_bytes / (1024 * 1024);

        // Score 0..100 — heuristic blend documented inline. Frontend
        // shows a soft heat bar; precise numbers aren't surfaced.
        let mut score: f64 = 0.0;
        let mut why_tags: Vec<String> = Vec::new();

        // Coverage component (0..40): fully serveable hits the cap;
        // partial coverage scales linearly. Hosting penalised slightly
        // so the wishlist favours unhosted-by-us entries the user can
        // help with.
        let coverage_pct = (shards_covered as f64) / (total_shards.max(1) as f64);
        score += 40.0 * coverage_pct;
        if hosted_by_us {
            score -= 5.0; // already helping; lower urgency
        }

        // Popularity component (0..25): unique holders, log-scaled so
        // a 100-replica model doesn't crowd out the long tail. Offset
        // by 1 so a 1-holder model still scores positive.
        let popularity = ((holders.len() + 1) as f64).log10();
        score += 12.5 * popularity.min(2.0); // log10(100) = 2.0 caps it
                                             // Surface the popularity boost as a why-tag once the swarm has
                                             // enough independent hosts that "popular here" is meaningful.
        if holders.len() >= 5 {
            why_tags.push("wishlist.why.popular_on_swarm".to_string());
        }

        // Demand component (0..25): regional demand from the gossip
        // index. We re-use the `region_demand` map already maintained
        // for auto-manage scoring.
        let mut regional_demand: f64 = 0.0;
        if let Some(ref my_region) = local_region {
            if let Some(d) = state
                .region_demand
                .get(&(mid.clone(), my_region.to_uppercase()))
            {
                regional_demand = *d.value();
            }
        }
        if regional_demand > 0.0 {
            score += 25.0 * (regional_demand / (regional_demand + 5.0));
            why_tags.push("wishlist.why.your_region_needs_this".to_string());
        }

        // VRAM-fit component (0..10): full points if model fits in the
        // network pool with margin; scaled down otherwise.
        let fit_factor = if pool_vram_mb == 0 {
            0.0
        } else if vram_required_mb == 0 {
            1.0
        } else {
            (pool_vram_mb as f64 / vram_required_mb as f64).clamp(0.0, 1.5) / 1.5
        };
        score += 10.0 * fit_factor;
        if local_vram_mb > 0 && local_vram_mb >= vram_required_mb {
            why_tags.push("wishlist.why.fits_your_memory".to_string());
        } else if vram_required_mb > pool_vram_mb {
            // Overshoots even the whole pool — flag as Unreachable.
            why_tags.push("wishlist.why.exceeds_swarm_capacity".to_string());
        }

        // Discovery / uniqueness component (0..10): if no holders, the
        // first contributor unlocks the model — give a strong nudge.
        if holders.is_empty() {
            score += 10.0;
            why_tags.push("wishlist.why.be_first_host".to_string());
        }

        // R112: HF trending boost (0..15) — if the wider HuggingFace
        // community is downloading this model, surface it on the swarm
        // wishlist too. Log-scaled so a 1M-download model doesn't
        // crowd out the long tail of niche-but-useful models.
        if let Some(&hf_downloads) = trending_for_model.get(&mid) {
            // log10(downloads).clamp(0, 7) maps 1 dl → 0, 10M → 7.
            let log = ((hf_downloads.max(1)) as f64).log10();
            let normalised = (log / 7.0).clamp(0.0, 1.0);
            score += 15.0 * normalised;
            why_tags.push("wishlist.why.popular_on_hf".to_string());
        }

        // Hosting / serveability tags — informational only.
        match status {
            WishlistStatus::Hosting => {
                why_tags.push("wishlist.why.you_already_host".to_string());
            }
            WishlistStatus::Serveable => {
                why_tags.push("wishlist.why.swarm_already_serves".to_string());
            }
            WishlistStatus::Aspirational => {
                let missing = total_shards.saturating_sub(shards_covered);
                if missing > 0 {
                    why_tags.push(format!("wishlist.why.parts_missing|missing={missing}"));
                }
            }
            WishlistStatus::Unreachable => {
                why_tags.push("wishlist.why.exceeds_swarm_capacity".to_string());
            }
            WishlistStatus::Blocked => {
                why_tags.push("wishlist.why.needs_review".to_string());
            }
        }

        // Region-coverage tag — if we have a region but no holder of
        // ours is in it, the swarm's redundancy is fragile from our
        // perspective.
        if local_region.is_some() && !local_region_has_a_holder && !holders.is_empty() {
            why_tags.push("wishlist.why.no_regional_replica".to_string());
        }

        // Override with Unreachable if score would be 0 and we know the
        // model exceeds capacity by 2× — no point ranking it.
        let final_status = if vram_required_mb > pool_vram_mb.saturating_mul(2)
            && !hosted_by_us
            && holders.is_empty()
        {
            WishlistStatus::Unreachable
        } else {
            status
        };

        // Recommended target replicas — light heuristic mirroring the
        // existing `geo_target_replicas` logic. Detailed scoring lives
        // in scoring.rs; here we only need a user-facing "needs N
        // hosts" number.
        let target_replicas = recommended_replica_target(online_node_count, regional_demand);

        let score_clamped = score.clamp(0.0, 100.0).round() as u32;
        entries.push(WishlistEntry {
            model_id: mid.0.clone(),
            display_name: manifest.name.clone(),
            status: final_status,
            score: score_clamped,
            why_tags,
            swarm_replicas: holders.len() as u32,
            target_replicas,
            size_mb,
            vram_required_mb,
            shards_covered,
            total_shards,
            hosted_by_us,
        });
    }

    // Sort: highest score first, then biggest model (proxy for
    // capability) so the headline candidates surface together.
    entries.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.size_mb.cmp(&a.size_mb))
            .then_with(|| a.display_name.cmp(&b.display_name))
    });
    entries.truncate(MAX_WISHLIST_ENTRIES);

    Wishlist {
        entries,
        computed_at: chrono::Utc::now().timestamp(),
    }
}

/// Recommended replica target — log-scaled with pool size, optionally
/// boosted by demand. Mirrors the existing `geo_target_replicas` shape
/// but inlined here so the wishlist doesn't reach into scoring.rs.
fn recommended_replica_target(online_nodes: u32, regional_demand: f64) -> u32 {
    let base = if online_nodes <= 1 {
        1
    } else {
        ((online_nodes as f64).log2().ceil() as u32).max(1)
    };
    let demand_factor = if regional_demand > 50.0 {
        2.0
    } else if regional_demand > 10.0 {
        1.5
    } else {
        1.0
    };
    let raw = ((base as f64) * demand_factor).ceil() as u32;
    raw.min(online_nodes.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replica_target_small_swarm() {
        // A 1-node swarm can have at most 1 holder.
        assert_eq!(recommended_replica_target(1, 0.0), 1);
    }

    #[test]
    fn replica_target_grows_with_pool() {
        // Should be log2(pool) and >=1.
        let t = recommended_replica_target(16, 0.0);
        assert!((1..=16).contains(&t));
    }

    #[test]
    fn replica_target_boosts_with_demand() {
        let no_demand = recommended_replica_target(64, 0.0);
        let high_demand = recommended_replica_target(64, 100.0);
        assert!(high_demand >= no_demand);
    }

    #[test]
    fn empty_wishlist_serialises() {
        let w = Wishlist::default();
        let json = serde_json::to_value(&w).unwrap();
        assert!(json["entries"].is_array());
        assert_eq!(json["entries"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn status_i18n_keys_match_expectations() {
        assert_eq!(
            WishlistStatus::Hosting.i18n_key(),
            "wishlist.status.hosting"
        );
        assert_eq!(
            WishlistStatus::Aspirational.i18n_key(),
            "wishlist.status.aspirational"
        );
    }
}
