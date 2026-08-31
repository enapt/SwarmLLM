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
    /// HF trending model that no peer is hosting yet. One-click adopt
    /// path — opens the HF browse view filtered to this repo so the
    /// user picks the quant variant. Distinguishes from `Blocked`
    /// (gated) and `Aspirational` (partial coverage in progress).
    Candidate,
    /// Even with everyone helping, no individual node has the VRAM/disk
    /// to host. Effectively unreachable for this swarm size; we still
    /// show it so the user understands the upper bound.
    Unreachable,
    /// Trust gate / private mode / explicit user-ignore — auto-manage
    /// won't act on this without explicit consent.
    #[default]
    Blocked,
}

#[cfg(test)]
impl WishlistStatus {
    /// Single-token i18n key used by the frontend to localise the badge.
    /// Test-only contract guard — the frontend derives the key from the
    /// serialized enum value (`STATUS_LABELS` in `swarm-tab.js`), so the
    /// production path never calls this. Kept under `cfg(test)` so renames
    /// of the serde tag break the test loudly.
    pub fn i18n_key(self) -> &'static str {
        match self {
            Self::Hosting => "wishlist.status.hosting",
            Self::Serveable => "wishlist.status.serveable",
            Self::Aspirational => "wishlist.status.aspirational",
            Self::Candidate => "wishlist.status.candidate",
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
    /// HF repo_id for `Candidate` entries (None for everything else).
    /// Frontend uses this to deep-link the user into the HF browse
    /// view pre-filtered to the repo so they can pick the quant
    /// variant. Other statuses leave this empty because the
    /// authoritative model identity is `model_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo_id: Option<String>,
    /// Free-form capability tokens for `Candidate` entries (chat /
    /// code / vision / multilingual / reasoning). Sourced from
    /// HfWatcher's `task_tags`. Empty for non-Candidate entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_tags: Vec<String>,
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

    // R130: foreign-wishlist boost. Aggregate per-model interest from
    // inbound `WishlistAnnouncement` gossip — drop expired entries on the
    // fly so a publisher that disappeared stops biasing our scoring after
    // `FOREIGN_WISHLIST_MAX_AGE_MS`. The aggregate carries (publisher
    // count, max score) so the scorer can ramp the boost smoothly with
    // both breadth (how many other nodes care) and depth (how strongly
    // the top voter cares).
    let foreign_wishlist_summary: HashMap<ModelId, (u32, u32)> = {
        let now_ms = crate::types::unix_now_ms();
        let mut per_model: HashMap<ModelId, (u32, u32)> = HashMap::new();
        let mut stale: Vec<(NodeId, ModelId)> = Vec::new();
        for entry in state.models.foreign_wishlist.iter() {
            let (publisher, model_id) = entry.key();
            let (score, ts_ms) = *entry.value();
            if now_ms.saturating_sub(ts_ms) > crate::daemon::state::FOREIGN_WISHLIST_MAX_AGE_MS {
                stale.push((publisher.clone(), model_id.clone()));
                continue;
            }
            let summary = per_model.entry(model_id.clone()).or_insert((0, 0));
            summary.0 = summary.0.saturating_add(1);
            if score > summary.1 {
                summary.1 = score;
            }
        }
        for k in stale {
            state.models.foreign_wishlist.remove(&k);
        }
        per_model
    };

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
            score += 15.0 * hf_downloads_normalised(hf_downloads);
            why_tags.push("wishlist.why.popular_on_hf".to_string());
        }

        // R130: foreign-wishlist boost (0..10) — other nodes in the
        // swarm have signalled they want this model. Blend breadth
        // (publisher count) with depth (max score). Capped low so it
        // can nudge ordering but never override local signals.
        if let Some(&(publisher_count, max_score)) = foreign_wishlist_summary.get(&mid) {
            // Breadth saturates at ~10 publishers (log10(10+1) ≈ 1.04),
            // multiplied by depth which is the publisher's own 0..100
            // score normalised to 0..1.
            let breadth = ((publisher_count + 1) as f64).log10().min(1.0);
            let depth = (max_score as f64) / 100.0;
            score += 10.0 * breadth * depth;
            why_tags.push("wishlist.why.other_nodes_want_this".to_string());
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
            // Unreachable here is the registry-walking loop, where status
            // is one of the four computed branches above — Candidate is
            // only built later from the trending feed and never lands in
            // this match arm. Tag inert so the exhaustiveness check stays
            // honest.
            WishlistStatus::Candidate => {}
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
            hf_repo_id: None,
            task_tags: Vec::new(),
        });
    }

    // Candidate entries: HF trending repos the swarm hasn't picked up
    // yet. Lets the user see "popular models the swarm could adopt"
    // without having to navigate to the Search subtab. One click on a
    // Candidate card opens the HF browse view pre-filtered to that
    // repo so the user picks the quant variant (we never auto-pick).
    //
    // Dedup against the registry: any repo already represented by a
    // local hf_sources entry is excluded — it's already in the loop
    // above with a real status.
    {
        let mut have_repos: std::collections::HashSet<String> = std::collections::HashSet::new();
        for src in state.models.hf_sources.iter() {
            have_repos.insert(src.value().repo_id.clone());
        }
        // Cap how many Candidate entries we add so they don't drown
        // out real (locally-known) entries. The frontend renders a
        // distinct section anyway, but the global MAX_WISHLIST_ENTRIES
        // truncate still applies after sorting.
        const MAX_CANDIDATE_ENTRIES: usize = 24;
        let mut candidate_count = 0usize;
        for entry in &trending_snapshot.entries {
            if candidate_count >= MAX_CANDIDATE_ENTRIES {
                break;
            }
            if have_repos.contains(&entry.repo_id) {
                continue;
            }
            // Score: HF downloads on a log scale (matches the boost
            // helper above). Trusted publishers get a flat +10 bonus
            // so curator-released models rank above randoms with
            // similar download counts.
            let mut cand_score = 60.0 * hf_downloads_normalised(entry.downloads);
            let mut why_tags: Vec<String> = vec!["wishlist.why.popular_on_hf".to_string()];
            if crate::model::huggingface::is_trusted_publisher(&entry.repo_id) {
                cand_score += 10.0;
                why_tags.push("wishlist.why.trusted_publisher".to_string());
            }
            why_tags.push("wishlist.why.candidate_one_click".to_string());

            // How big is it? A candidate has no manifest, so the only size
            // signal is the repo name — the same estimate the capacity plan
            // already reasons with. Without it every candidate was sized 0,
            // which reads as "fits perfectly" everywhere downstream: a 0.6B
            // and a 120B scored the same on fit and neither could ever be
            // flagged as beyond the swarm. `None` stays 0, which keeps the
            // previous behaviour for a name we cannot parse rather than
            // inventing a number.
            let cand_size_mb =
                crate::daemon::state::capacity_plan::estimate_q4_size_mb_from_repo_id_impl(
                    &entry.repo_id,
                )
                .unwrap_or(0);
            // Same 1.25x weights-to-VRAM rule the capacity plan uses.
            let cand_vram_mb = (cand_size_mb as f64 * 1.25) as u64;
            let mut cand_status = WishlistStatus::Candidate;
            if cand_vram_mb > 0 {
                if cand_vram_mb > pool_vram_mb {
                    why_tags.push("wishlist.why.exceeds_swarm_capacity".to_string());
                } else if local_vram_mb > 0 && local_vram_mb >= cand_vram_mb {
                    why_tags.push("wishlist.why.fits_your_memory".to_string());
                }
                // Same rule the registry path uses, so a candidate and an
                // adopted model are judged alike.
                if cand_vram_mb > pool_vram_mb.saturating_mul(2) {
                    cand_status = WishlistStatus::Unreachable;
                }
            }

            let display_name = entry
                .repo_id
                .split('/')
                .next_back()
                .unwrap_or(&entry.repo_id)
                .to_string();

            entries.push(WishlistEntry {
                // Synthetic key so the frontend can dedup across renders
                // without colliding with real model_ids.
                model_id: format!("hf-candidate:{}", entry.repo_id),
                display_name,
                status: cand_status,
                score: cand_score.clamp(0.0, 100.0).round() as u32,
                why_tags,
                swarm_replicas: 0,
                target_replicas: recommended_replica_target(online_node_count, 0.0),
                size_mb: cand_size_mb,
                vram_required_mb: cand_vram_mb,
                shards_covered: 0,
                total_shards: 0,
                hosted_by_us: false,
                hf_repo_id: Some(entry.repo_id.clone()),
                task_tags: entry.task_tags.clone(),
            });
            candidate_count += 1;
        }
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

/// Normalise an HF download count to the 0..1 wishlist boost range.
/// log10-scaled: 1 download → 0, 10M → 1. Shared between the
/// `Hosting`/`Serveable`/`Aspirational` HF-trending boost (×15) and the
/// `Candidate`-row score (×60) so the two stay in lockstep when the
/// curve is tweaked.
fn hf_downloads_normalised(downloads: u64) -> f64 {
    let log = (downloads.max(1) as f64).log10();
    (log / 7.0).clamp(0.0, 1.0)
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
    fn hf_downloads_normalised_zero_at_one_download() {
        // log10(1) = 0 → curve floor is 0.
        assert!(hf_downloads_normalised(1) < 0.01);
    }

    #[test]
    fn hf_downloads_normalised_clamps_at_ten_million() {
        // log10(10_000_000) = 7, divided by 7 = 1.0 → curve ceiling.
        assert!((hf_downloads_normalised(10_000_000) - 1.0).abs() < 0.001);
        // Even a 1B-download model stays clamped at 1.0.
        assert!((hf_downloads_normalised(1_000_000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn hf_downloads_normalised_monotonic() {
        let a = hf_downloads_normalised(1_000);
        let b = hf_downloads_normalised(10_000);
        let c = hf_downloads_normalised(100_000);
        let d = hf_downloads_normalised(1_000_000);
        assert!(a < b && b < c && c < d);
    }

    #[test]
    fn hf_downloads_normalised_midpoint_around_log_4_div_7() {
        // 10k downloads = log10 = 4, normalised = 4/7 ≈ 0.5714. Drift
        // here would silently re-rank every wishlist Candidate row.
        let got = hf_downloads_normalised(10_000);
        assert!((got - 4.0 / 7.0).abs() < 0.001, "got {got}");
    }

    #[test]
    fn hf_downloads_normalised_handles_zero() {
        // Saturating max(1) — log10(1) = 0 — clamp guards both ends.
        assert!(hf_downloads_normalised(0) < 0.01);
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

    /// The wishlist's HuggingFace candidates carried `size_mb: 0` — no
    /// manifest exists for a model nobody hosts yet — and everything
    /// downstream reads 0 as "fits perfectly": `fit_factor` becomes a
    /// constant 1.0, `exceeds_swarm_capacity` can never fire and
    /// `Unreachable` is itself unreachable. So the surface whose entire job
    /// is "what should this swarm add next" scored a 0.6B and a 120B alike.
    ///
    /// The estimate was in the next module the whole time, feeding the
    /// capacity plan. These are the repo ids actually on this swarm's
    /// wishlist on 2026-08-31.
    #[test]
    fn a_candidate_is_sized_from_its_repo_name() {
        let est = crate::daemon::state::capacity_plan::estimate_q4_size_mb_from_repo_id_impl;

        let big = est("Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF").expect("30B is parseable");
        let small = est("Qwen/Qwen3-0.6B").expect("0.6B is parseable");

        // The capacity plan already values this one at 17,697 MB and names it
        // the headline target; the wishlist must not disagree with its sibling.
        assert!(
            (15_000..=19_000).contains(&big),
            "30B should size to ~16.9 GB, got {big} MB"
        );
        assert!(
            small < 1_000,
            "0.6B should size well under 1 GB, got {small} MB"
        );
        assert!(
            big > small * 20,
            "the whole point is telling these apart: {big} vs {small}"
        );
    }

    /// A name with no parameter token must stay 0 rather than be guessed at,
    /// which is what keeps the previous behaviour for anything unparseable.
    #[test]
    fn an_unparseable_candidate_name_is_not_guessed_at() {
        let est = crate::daemon::state::capacity_plan::estimate_q4_size_mb_from_repo_id_impl;
        assert_eq!(est("some-org/mystery-model"), None);
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
        assert_eq!(
            WishlistStatus::Candidate.i18n_key(),
            "wishlist.status.candidate"
        );
    }

    /// A Candidate entry serialises with `hf_repo_id` populated and a
    /// synthetic `hf-candidate:` model_id. Frontend uses both to
    /// route the click handler.
    #[test]
    fn candidate_entry_round_trip() {
        let e = WishlistEntry {
            model_id: "hf-candidate:bartowski/Mistral-7B".into(),
            display_name: "Mistral-7B".into(),
            status: WishlistStatus::Candidate,
            score: 70,
            why_tags: vec!["wishlist.why.popular_on_hf".into()],
            swarm_replicas: 0,
            target_replicas: 2,
            size_mb: 0,
            vram_required_mb: 0,
            shards_covered: 0,
            total_shards: 0,
            hosted_by_us: false,
            hf_repo_id: Some("bartowski/Mistral-7B".into()),
            task_tags: vec!["chat".into()],
        };
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["status"], "candidate");
        assert_eq!(json["hf_repo_id"], "bartowski/Mistral-7B");
        assert_eq!(json["task_tags"][0], "chat");
        let round: WishlistEntry = serde_json::from_value(json).unwrap();
        assert_eq!(round.status, WishlistStatus::Candidate);
    }

    /// hf_repo_id and task_tags are omitted from the wire for non-Candidate
    /// entries (skip_serializing_if) — keeps the payload small for the
    /// common case.
    #[test]
    fn non_candidate_entry_omits_hf_fields() {
        let e = WishlistEntry::default();
        let json = serde_json::to_value(&e).unwrap();
        assert!(
            json.get("hf_repo_id").is_none(),
            "hf_repo_id should not serialise when None: {json}"
        );
        assert!(
            json.get("task_tags").is_none(),
            "task_tags should not serialise when empty: {json}"
        );
    }
}
