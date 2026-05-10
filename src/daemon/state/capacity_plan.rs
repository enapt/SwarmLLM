//! Capacity-plan / what-if calculator.
//!
//! "If N more nodes joined with X GB of memory, what would the swarm
//! unlock?" Answers this for the dashboard's Capacity Plan view (R113).
//! Drives the headline "be the catalyst" message that turns the value
//! prop ("contribute and run huge models") into a concrete next step.
//!
//! Pure function over the current capacity snapshot + a hypothetical
//! delta. Cheap; safe to call on every render. R113.

use serde::{Deserialize, Serialize};

use super::capacity::{compute_swarm_capacity, SwarmCapacity};
use super::SharedState;

/// One scenario in the capacity-plan view: hypothetically `added_nodes`
/// nodes join the swarm, each contributing `vram_gb_per_node` of memory.
/// We report the headline upgrade this would unlock.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapacityScenario {
    pub label: String,
    pub added_nodes: u32,
    pub vram_gb_per_node: u32,
    /// Total swarm VRAM in MB if the scenario realised.
    pub projected_total_vram_mb: u64,
    /// Models the swarm could newly serve under this scenario, beyond
    /// what's already serveable today. Top 3 by size.
    pub newly_unlocked: Vec<ProjectedModel>,
    /// True if the scenario unlocks something new — frontend surfaces
    /// this as the "if N more contributors joined" CTA.
    pub unlocks_anything: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectedModel {
    pub model_id: String,
    pub display_name: String,
    pub size_mb: u64,
    /// Why we believe this scenario unlocks the model. Currently only
    /// the simple "memory budget exceeded today, fits with delta" case;
    /// future iterations could fold in regional coverage etc.
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CapacityPlan {
    /// Current snapshot — duplicated here so the frontend doesn't have
    /// to make two requests to render the comparison.
    pub current: SwarmCapacity,
    /// Three scenarios, from "small contribution" to "big swarm event".
    pub scenarios: Vec<CapacityScenario>,
    /// Best aspirational model that's just out of reach today — the
    /// "headline target" the user could unlock with the smallest
    /// contribution. None when the swarm already runs everything in
    /// the registry that fits.
    pub headline_target: Option<HeadlineTarget>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadlineTarget {
    pub model_id: String,
    pub display_name: String,
    pub size_mb: u64,
    /// Memory deficit (MB) — how much more swarm-wide VRAM is needed.
    pub vram_shortfall_mb: u64,
    /// Suggested contribution — number of average-modern-GPU nodes
    /// (8 GB) that would close the gap.
    pub contributors_needed: u32,
}

/// Build a fresh capacity-plan snapshot. Pure function; safe to call on
/// every render. Three baked scenarios — small / medium / large — chosen
/// so a user with a typical home setup can see what their contribution
/// alone would do.
pub fn compute_capacity_plan(state: &SharedState) -> CapacityPlan {
    let current = compute_swarm_capacity(state);

    // Models the manifest registry knows about but that exceed swarm
    // capacity today — these are the headline targets. We compute the
    // largest model we *could* serve given memory only (no peer
    // distribution detail) and call out the deficit.
    let mut largest_by_vram: Option<(crate::types::ModelId, String, u64, u64)> = None;
    for manifest in state.model_registry.models() {
        let vram_required = crate::model::auto_manage::vram::estimate_model_vram_mb_arch(
            manifest.total_size_bytes,
            &manifest.architecture,
        );
        // Skip models we already serve.
        let already_serveable = current
            .serveable_models
            .iter()
            .any(|m| m.model_id == manifest.id.0);
        if already_serveable {
            continue;
        }
        if vram_required > current.total_vram_mb {
            let shortfall = vram_required.saturating_sub(current.total_vram_mb);
            let size_mb = manifest.total_size_bytes / (1024 * 1024);
            // Pick the smallest-shortfall, biggest-size combo so the
            // CTA feels achievable without underselling capability.
            match &largest_by_vram {
                None => {
                    largest_by_vram = Some((
                        manifest.id.clone(),
                        manifest.name.clone(),
                        size_mb,
                        shortfall,
                    ));
                }
                Some((_, _, _, current_shortfall)) => {
                    if shortfall < *current_shortfall {
                        largest_by_vram = Some((
                            manifest.id.clone(),
                            manifest.name.clone(),
                            size_mb,
                            shortfall,
                        ));
                    }
                }
            }
        }
    }
    let headline_target = largest_by_vram.map(|(id, name, size_mb, shortfall)| {
        // 8 GB is "modern home GPU" — used as the canonical contributor
        // size in the CTA. div_ceil rounded up so we never under-quote
        // the contributor count.
        let contributors_needed = ((shortfall as f64 / (8 * 1024) as f64).ceil() as u32).max(1);
        HeadlineTarget {
            model_id: id.0,
            display_name: name,
            size_mb,
            vram_shortfall_mb: shortfall,
            contributors_needed,
        }
    });

    let scenarios = vec![
        scenario(state, &current, "small", 3, 8),
        scenario(state, &current, "medium", 10, 8),
        scenario(state, &current, "large", 25, 16),
    ];

    CapacityPlan {
        current,
        scenarios,
        headline_target,
    }
}

fn scenario(
    state: &SharedState,
    current: &SwarmCapacity,
    label: &str,
    added_nodes: u32,
    vram_gb_per_node: u32,
) -> CapacityScenario {
    let added_vram_mb = (added_nodes as u64) * (vram_gb_per_node as u64) * 1024;
    let projected_total_vram_mb = current.total_vram_mb.saturating_add(added_vram_mb);

    // Newly-unlocked: any registry model whose VRAM requirement fits in
    // projected_total_vram_mb but didn't fit today.
    let mut unlocked: Vec<ProjectedModel> = Vec::new();
    for manifest in state.model_registry.models() {
        let vram_required = crate::model::auto_manage::vram::estimate_model_vram_mb_arch(
            manifest.total_size_bytes,
            &manifest.architecture,
        );
        if vram_required <= current.total_vram_mb {
            continue; // already fits
        }
        if vram_required > projected_total_vram_mb {
            continue; // still doesn't fit
        }
        let already_serveable = current
            .serveable_models
            .iter()
            .any(|m| m.model_id == manifest.id.0);
        if already_serveable {
            continue;
        }
        unlocked.push(ProjectedModel {
            model_id: manifest.id.0.clone(),
            display_name: manifest.name.clone(),
            size_mb: manifest.total_size_bytes / (1024 * 1024),
            reason: "memory_unlock".to_string(),
        });
    }
    unlocked.sort_by_key(|m| std::cmp::Reverse(m.size_mb));
    let unlocks_anything = !unlocked.is_empty();
    unlocked.truncate(3);

    CapacityScenario {
        label: label.to_string(),
        added_nodes,
        vram_gb_per_node,
        projected_total_vram_mb,
        newly_unlocked: unlocked,
        unlocks_anything,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_plan_serialises() {
        let plan = CapacityPlan::default();
        let json = serde_json::to_value(&plan).unwrap();
        assert!(json["scenarios"].is_array());
        assert!(json["headline_target"].is_null());
    }
}
