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

    // Scale tier sizes with the current swarm so the "what if more people
    // joined?" CTA stays meaningful as the swarm grows. A 2-node test cluster
    // and a 10,000-node production swarm need very different boost tiers —
    // hardcoded (3 / 10 / 25) makes the latter look trivial. The floors
    // (3 / 10 / 25) keep the small swarm case identical to the original.
    let nodes_now = current.online_nodes.max(1);
    let small_nodes = ((nodes_now as f32 * 0.5).ceil() as u32).max(3);
    let medium_nodes = ((nodes_now as f32 * 3.0).ceil() as u32).max(10);
    let large_nodes = ((nodes_now as f32 * 10.0).ceil() as u32).max(25);

    let scenarios = vec![
        scenario(state, &current, "small", small_nodes, 8),
        scenario(state, &current, "medium", medium_nodes, 8),
        scenario(state, &current, "large", large_nodes, 16),
    ];

    CapacityPlan {
        current,
        scenarios,
        headline_target,
    }
}

/// Estimate Q4_K_M GGUF size in MB from a HF repo_id by parsing the
/// parameter count (e.g. "Qwen3-70B-Instruct" → ~38 GB).
///
/// **The single answer to "roughly how big is this repo" for a model we do
/// not have on disk**, and there are two callers that need it: the
/// capacity-plan scenario builder, and the wishlist's HuggingFace-candidate
/// branch. The wishlist used to hardcode `size_mb: 0` for every candidate,
/// which made `fit_factor` a constant 1.0 — so a 0.6B and a 120B scored
/// identically on fit, `wishlist.why.exceeds_swarm_capacity` could never
/// fire, and `Unreachable` was unreachable. The estimator was sitting in
/// the next module the whole time.
pub(crate) fn estimate_q4_size_mb_from_repo_id_impl(repo_id: &str) -> Option<u64> {
    estimate_q4_size_mb_from_repo_id(repo_id)
}

/// Estimate Q4_K_M GGUF size in MB from a HF repo_id by parsing the
/// parameter count. See [`estimate_q4_size_mb_from_repo_id_impl`].
///
/// Heuristic: Q4_K_M ≈ 0.55 GB per billion parameters (empirical, holds
/// reasonably from 0.5B through 405B). MoE expert counts (`Nx7B`) are
/// expanded to total params. Returns None when the name has no clear
/// parameter token — those entries are dropped rather than guessed at.
fn estimate_q4_size_mb_from_repo_id(repo_id: &str) -> Option<u64> {
    let lower = repo_id.to_lowercase();
    // Match either `<num>x<num>B` (MoE) or just `<num>B`.
    // We scan tail-to-head so trailing "-70B" wins over an "8x" earlier.
    let chars: Vec<char> = lower.chars().collect();
    let mut best_b: Option<f64> = None;
    let mut i = 0;
    while i < chars.len() {
        // Find a digit run.
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let num: f64 = lower[start..i].parse().ok()?;
            // 'x' for MoE expert count, then another digit run, then 'b'.
            if i < chars.len() && chars[i] == 'x' {
                let after_x = i + 1;
                let mut j = after_x;
                while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                    j += 1;
                }
                if j > after_x && j < chars.len() && chars[j] == 'b' {
                    let per_expert: f64 = lower[after_x..j].parse().ok()?;
                    let total = num * per_expert;
                    if (0.1..=2000.0).contains(&total) {
                        best_b = Some(total);
                    }
                    i = j + 1;
                    continue;
                }
            }
            // Plain `<num>B` (case-insensitive already lowered).
            if i < chars.len() && chars[i] == 'b' {
                // Guard against accidental matches inside other tokens
                // (e.g. "embedding" starts with 'em' so digit+'b' is OK).
                let next_ok = i + 1 >= chars.len()
                    || !chars[i + 1].is_ascii_alphabetic()
                    || chars[i + 1] == '-'
                    || chars[i + 1] == '.'
                    || chars[i + 1] == '_';
                // `-A3B` is the ACTIVE parameter count of a sparse MoE, not
                // its size on disk: `Qwen3-Coder-30B-A3B` is 30B of weights
                // that activate 3B per token, and all 30B have to be stored
                // and distributed. Because that token comes LAST, the
                // tail-to-head rule below picked it and under-estimated the
                // model **~10x** — 1,689 MB against ~16,900. It reads as an
                // edge case and is not: it is the modern naming convention
                // for exactly the large models this estimate exists to
                // surface (Qwen3-30B-A3B, Qwen3-235B-A22B). The older
                // `8x7B` form was handled above and still is.
                let is_moe_active_token = start > 0 && chars[start - 1] == 'a';
                if next_ok && !is_moe_active_token && (0.1..=2000.0).contains(&num) {
                    best_b = Some(num);
                }
            }
        }
        i += 1;
    }
    best_b.map(|b| (b * 0.55 * 1024.0) as u64)
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

    let mut unlocked: Vec<ProjectedModel> = Vec::new();
    let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Registry models — concrete unlocks. Same logic as before: skip
    // already-serveable, skip still-doesn't-fit.
    for manifest in state.model_registry.models() {
        let vram_required = crate::model::auto_manage::vram::estimate_model_vram_mb_arch(
            manifest.total_size_bytes,
            &manifest.architecture,
        );
        if vram_required <= current.total_vram_mb {
            continue;
        }
        if vram_required > projected_total_vram_mb {
            continue;
        }
        let already_serveable = current
            .serveable_models
            .iter()
            .any(|m| m.model_id == manifest.id.0);
        if already_serveable {
            continue;
        }
        seen_keys.insert(manifest.id.0.to_lowercase());
        unlocked.push(ProjectedModel {
            model_id: manifest.id.0.clone(),
            display_name: manifest.name.clone(),
            size_mb: manifest.total_size_bytes / (1024 * 1024),
            reason: "memory_unlock".to_string(),
        });
    }

    // Trending HF repos — aspirational unlocks. Without these the three
    // scenarios collapse to the same handful of locally-registered models
    // and the user sees identical lists at every tier (the bug). Pulling
    // trending in lets a 24 GB boost surface 8B models, an 80 GB boost
    // surface 30B class models, and a 400 GB boost surface 70B+ models.
    let trending_snap = state.models.hf_trending_cache.load_full();
    for entry in trending_snap.entries.iter() {
        let est_size_mb = match estimate_q4_size_mb_from_repo_id(&entry.repo_id) {
            Some(s) => s,
            None => continue,
        };
        // VRAM requirement ≈ size × 1.25 (rule-of-thumb in vram.rs)
        let est_vram_mb = (est_size_mb as f64 * 1.25) as u64;
        if est_vram_mb <= current.total_vram_mb {
            continue;
        }
        if est_vram_mb > projected_total_vram_mb {
            continue;
        }
        let key = entry.repo_id.to_lowercase();
        if !seen_keys.insert(key) {
            continue;
        }
        unlocked.push(ProjectedModel {
            model_id: entry.repo_id.clone(),
            display_name: pretty_repo_name(&entry.repo_id),
            size_mb: est_size_mb,
            reason: "trending_aspirational".to_string(),
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

/// Drop the org prefix from a HF repo_id, strip GGUF/quant suffixes, and
/// titlecase the remaining tokens. Examples:
///   meta-llama/Llama-3.1-70B-Instruct      → "Llama 3.1 70B Instruct"
///   openai/gpt-oss-20b                     → "GPT OSS 20B"
///   bartowski/Qwen2.5-Coder-7B-Instruct-GGUF → "Qwen2.5 Coder 7B Instruct"
fn pretty_repo_name(repo_id: &str) -> String {
    let tail = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let cleaned = tail
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF")
        .replace(['_', '-'], " ");
    cleaned
        .split_whitespace()
        .filter(|tok| {
            let l = tok.to_lowercase();
            l != "gguf" && !l.starts_with("q4_") && !l.starts_with("q5_") && l != "q4" && l != "q8"
        })
        .map(|tok| {
            // Already mixed-case (e.g. "Qwen2.5") → keep as-is.
            // Has uppercase already → keep (preserves acronyms like GGUF/OSS).
            // All-lower → titlecase first letter.
            // Numeric/Bs (e.g. "70B") → uppercase.
            if tok.chars().any(|c| c.is_ascii_uppercase()) {
                tok.to_string()
            } else if tok
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == 'b' || c == 'B' || c == 'x')
            {
                tok.to_uppercase()
            } else {
                let mut cs = tok.chars();
                match cs.next() {
                    Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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

#[cfg(test)]
mod estimator_tests {
    use super::estimate_q4_size_mb_from_repo_id as est;

    /// `-A3B` is the ACTIVE parameter count of a sparse MoE, not the weights
    /// that have to be stored. It comes last in the name, so the tail-to-head
    /// rule picked it and under-estimated by ~10x — 1,689 MB for a model of
    /// ~16,900. That is the modern naming convention for precisely the large
    /// models this estimate exists to surface, so it was wrong exactly where
    /// it mattered, and wrong in the optimistic direction: a 235B looked like
    /// 12 GB and would have been reported as nearly within reach.
    #[test]
    fn a_sparse_moe_is_sized_by_its_total_parameters_not_its_active_ones() {
        let coder = est("Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF").expect("parseable");
        assert!(
            (15_000..=19_000).contains(&coder),
            "30B-A3B must size on 30B, got {coder} MB"
        );
        let big = est("Qwen/Qwen3-235B-A22B").expect("parseable");
        assert!(big > 100_000, "235B-A22B must size on 235B, got {big} MB");
    }

    /// The forms that already worked must keep working — this was a targeted
    /// fix, not a rewrite, and the `8x7B` MoE form is handled by a separate
    /// branch that the change must not disturb.
    #[test]
    fn the_naming_forms_that_already_worked_are_unchanged() {
        assert_eq!(est("Qwen/Qwen3-32B"), Some(18022));
        assert_eq!(est("openai/gpt-oss-120b"), Some(67584));
        assert_eq!(est("meta-llama/Llama-3.1-70B"), Some(39424));
        // 8 experts x 7B = 56B total.
        assert_eq!(est("mistralai/Mixtral-8x7B-Instruct-v0.1"), Some(31539));
        // A version number before the parameter token must not win.
        assert_eq!(est("Qwen/Qwen2.5-0.5B-Instruct"), Some(281));
        assert_eq!(est("some-org/mystery-model"), None);
    }
}
