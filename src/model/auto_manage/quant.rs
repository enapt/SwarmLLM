//! R133: quant-choice recommender.
//!
//! Groups models in the local registry by inferred base name (model name
//! with the quant tag stripped), then for each family computes a
//! recommendation: "given the swarm's aggregate VRAM and a target replica
//! count, which quant level should we host?"
//!
//! Pure read-only — the recommender does NOT change which model the
//! auto-manage system actually downloads. It surfaces a hint via REST so
//! the user (or a future auto-action layer) can decide. Mirrors the
//! shape of the wishlist's "we couldn't host this at Q5 but we could at
//! Q4" framing for non-technical users.
//!
//! See `docs/FUTURE_WORK.md` § "Quantisation choice automation" for
//! design rationale.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::daemon::SharedState;
use crate::types::{ModelManifest, Quantization};

/// Per-family quant recommendation surface. Cached on
/// `state.models.quant_recommendations` as `ArcSwap` and refreshed on the
/// same cadence as `compute_wishlist`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuantRecommendations {
    pub families: Vec<QuantFamilyRecommendation>,
    /// Unix seconds since the last refresh. Frontend uses this for the
    /// "computed N seconds ago" footer.
    pub computed_at: i64,
}

/// One quant family — all known variants of the same base model + the
/// recommendation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantFamilyRecommendation {
    /// Base name with the quant tag stripped. Used as the family key.
    pub base_name: String,
    /// Display name for the family — typically `base_name` cleaned up
    /// for human reading. Falls back to base_name when no cleanup is
    /// possible.
    pub display_name: String,
    /// All quant variants we currently know about for this family, in
    /// the order they appear in the registry.
    pub known_variants: Vec<QuantVariantInfo>,
    /// Index into `known_variants` of the quant we currently host /
    /// recommend. `None` if no variant fits the swarm VRAM budget.
    pub recommended_index: Option<usize>,
    /// Human-readable rationale for the recommendation. Frontend localises
    /// this via the `quant.rec.*` i18n namespace (caller supplies the key,
    /// the value's `params` slot carries interpolation variables).
    pub rationale_tag: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantVariantInfo {
    pub model_id: String,
    pub display_name: String,
    pub quantization: Quantization,
    /// Quality score 0..100 (see `Quantization::quality_score`).
    pub quality_score: u32,
    pub size_mb: u64,
    /// Estimated VRAM (MB) needed to load on a single node, MoE-aware.
    pub vram_required_mb: u64,
    /// Whether at least one node currently hosts every shard.
    pub serveable: bool,
}

/// Refresh + publish a fresh `QuantRecommendations` snapshot.
pub fn refresh_quant_recommendations(state: &SharedState) {
    let snapshot = compute_quant_recommendations(state);
    state
        .models
        .quant_recommendations
        .store(std::sync::Arc::new(snapshot));
}

/// R134.6: opt-in auto-action for the recommendation surface. When
/// `auto_manage.auto_switch_quants` is on, walks each family in the
/// current snapshot and promotes the recommended variant's trust level
/// to `DemandVerified` for any family where the user currently hosts
/// at least one shard of a *different* variant in the family. The
/// normal auto-manage scoring/download path then opportunistically
/// acquires the recommended variant. The old variant is NOT pruned
/// proactively — let standard prune handle dedup once VRAM pressure
/// hits, so there's no in-flight inference disruption window.
///
/// Returns the number of trust promotions performed (used by the
/// activity log + tests).
pub fn apply_quant_auto_action(state: &SharedState) -> usize {
    if !state.cfg().auto_manage.auto_switch_quants {
        return 0;
    }
    let snapshot = state.models.quant_recommendations.load_full();
    if snapshot.families.is_empty() {
        return 0;
    }
    let local_node_id = state.identity.node_id().clone();
    let mut promotions = 0usize;
    let mut pending_activity: Vec<(String, String, String)> = Vec::new();
    for fam in &snapshot.families {
        let Some(rec_idx) = fam.recommended_index else {
            continue;
        };
        let Some(recommended) = fam.known_variants.get(rec_idx) else {
            continue;
        };
        let rec_model_id = crate::types::ModelId(recommended.model_id.clone());
        // Skip when we already host shards of the recommended variant.
        if hosts_any_shard(state, &rec_model_id, &local_node_id) {
            continue;
        }
        // Switch candidate only when we host a SIBLING variant — otherwise
        // we'd promote a model we have no interest in. The recommender's
        // family grouping already requires sibling membership via base name.
        let hosts_sibling = fam.known_variants.iter().any(|v| {
            v.model_id != recommended.model_id
                && hosts_any_shard(
                    state,
                    &crate::types::ModelId(v.model_id.clone()),
                    &local_node_id,
                )
        });
        if !hosts_sibling {
            continue;
        }
        // Promote — only if currently below DemandVerified. User pins
        // and existing higher-trust entries are left alone. The
        // `upgraded` flag is captured here and the activity emit happens
        // OUTSIDE the entry guard so a stray contention can't deadlock.
        let upgraded = {
            let mut upgraded_inner = false;
            state
                .models
                .model_trust
                .entry(rec_model_id.clone())
                .and_modify(|t| {
                    if t.trust_level < crate::types::ModelTrustLevel::DemandVerified
                        && !t.pinned_by_user
                    {
                        t.trust_level = crate::types::ModelTrustLevel::DemandVerified;
                        upgraded_inner = true;
                    }
                })
                .or_insert_with(|| {
                    upgraded_inner = true;
                    let mut info = crate::types::ModelTrustInfo::new_discovered();
                    info.trust_level = crate::types::ModelTrustLevel::DemandVerified;
                    info
                });
            upgraded_inner
        };
        if upgraded {
            promotions += 1;
            tracing::info!(
                family = %fam.base_name,
                model = %rec_model_id,
                "auto_switch_quants: promoted recommended variant for opportunistic upgrade"
            );
            // Stash the activity payload — emit AFTER the iteration completes
            // so the broadcast send can't interact with the model_trust
            // DashMap iteration order through any reentrant subscriber.
            pending_activity.push((
                fam.display_name.clone(),
                recommended.display_name.clone(),
                rec_model_id.0.clone(),
            ));
        }
    }
    for (family, variant, model_id) in pending_activity {
        state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "auto_manage",
                "quant_auto_switch",
                format!("Auto-switching {family} to a better quality ({variant})"),
            )
            .with_model(&model_id)
            .with_model_name(&variant),
        );
    }
    promotions
}

/// Whether some node in the swarm holds every shard of `manifest` — i.e. the
/// model can actually be served right now.
///
/// This backs `QuantVariantInfo::serveable`, which was constructed as `false`
/// with a "filled below" comment describing code that was never written, so the
/// field read `false` for every variant on every response, including models the
/// asking node was hosting itself.
fn every_shard_has_a_holder(state: &SharedState, manifest: &ModelManifest) -> bool {
    !manifest.shards.is_empty()
        && manifest.shards.iter().all(|s| {
            !state
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: s.index,
                })
                .is_empty()
        })
}

fn hosts_any_shard(
    state: &SharedState,
    model_id: &crate::types::ModelId,
    local_node_id: &crate::types::NodeId,
) -> bool {
    let Some(manifest) = state.model_registry.get_manifest(model_id) else {
        return false;
    };
    manifest.shards.iter().any(|s| {
        let sid = crate::types::ShardId {
            model_id: model_id.clone(),
            index: s.index,
        };
        state
            .model_registry
            .shard_holders(&sid)
            .contains(local_node_id)
    })
}

/// Compute the recommendation snapshot — pure function over registry +
/// swarm capacity. Returns an empty result when the registry has no
/// models.
pub fn compute_quant_recommendations(state: &SharedState) -> QuantRecommendations {
    let pool_vram_mb = crate::model::auto_manage::vram::global_pool_vram_mb(state);
    // The device that would actually run the model, not the card alone — a
    // processor-only node recommended quantisations sized for the swarm's
    // replica share while ignoring its own RAM entirely (gotcha #483).
    let local_vram_mb = crate::model::auto_manage::vram::node_model_budget_mb(state).unwrap_or(0);

    // Group manifests by inferred base name. We do this on-the-fly
    // rather than storing a `base_name` on the manifest itself — keeps
    // the wire-format stable, and the cost is O(n) per refresh which
    // is dwarfed by the existing wishlist pass.
    let mut by_family: HashMap<String, Vec<ModelManifest>> = HashMap::new();
    for manifest in state.model_registry.models() {
        let base = inferred_base_name(&manifest.name);
        by_family.entry(base).or_default().push(manifest);
    }

    let mut families: Vec<QuantFamilyRecommendation> = by_family
        .into_iter()
        .filter(|(_, variants)| !variants.is_empty())
        .map(|(base_name, variants)| {
            build_family(state, &base_name, variants, pool_vram_mb, local_vram_mb)
        })
        .collect();

    // Sort by display name for stable frontend rendering.
    families.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    QuantRecommendations {
        families,
        computed_at: chrono::Utc::now().timestamp(),
    }
}

/// Strip a trailing quant tag from a model name to derive a family
/// identifier. The tag matches the patterns recognised by
/// `Quantization::parse` so name-and-filename quant detection stays in
/// sync. Returns the original name when no tag is detected.
pub fn inferred_base_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    // Strip a `.gguf` suffix first so the splitter doesn't trip on it.
    let core = lower.trim_end_matches(".gguf");
    // Try each delimiter set used by the HF filename convention.
    for delim in &['-', '.', '_'][..] {
        if let Some(pos) = core.rfind(*delim) {
            let suffix = &core[pos + 1..];
            if Quantization::parse(suffix) != Quantization::Unknown {
                return core[..pos].to_string();
            }
        }
    }
    // Multi-token quant tags like `q4_k_m` — try the last 3 segments
    // joined together (covers `q4`, `k`, `m`).
    let segs: Vec<&str> = core.split(['-', '.', '_'][..].as_ref()).collect();
    let n = segs.len();
    if n >= 3 {
        let tail = format!("{}{}{}", segs[n - 3], segs[n - 2], segs[n - 1]);
        if Quantization::parse(&tail) != Quantization::Unknown {
            let cutoff = core
                .rfind(segs[n - 3])
                .map(|i| i.saturating_sub(1))
                .unwrap_or(core.len());
            return core[..cutoff].to_string();
        }
    }
    core.to_string()
}

fn build_family(
    state: &SharedState,
    base_name: &str,
    variants: Vec<ModelManifest>,
    pool_vram_mb: u64,
    local_vram_mb: u64,
) -> QuantFamilyRecommendation {
    // Derive a display name from the first variant. Strip a common
    // model name prefix junk to make it presentable; fall back to the
    // base_name otherwise.
    let display_name = variants
        .first()
        .map(|m| base_name_to_display(base_name, &m.name))
        .unwrap_or_else(|| base_name.to_string());

    let mut known_variants: Vec<QuantVariantInfo> = variants
        .iter()
        .map(|manifest| {
            let q = parse_quant_from_manifest(manifest);
            let size_mb = manifest.total_size_bytes / (1024 * 1024);
            let vram_required_mb = crate::model::auto_manage::vram::estimate_model_vram_mb_arch(
                manifest.total_size_bytes,
                &manifest.architecture,
            );
            QuantVariantInfo {
                model_id: manifest.id.0.clone(),
                display_name: manifest.name.clone(),
                quantization: q,
                quality_score: q.quality_score(),
                size_mb,
                vram_required_mb,
                serveable: every_shard_has_a_holder(state, manifest),
            }
        })
        .collect();

    // Sort variants by quality DESCENDING so the recommended-index
    // lookup below picks the highest-quality fit first.
    known_variants.sort_by_key(|v| std::cmp::Reverse(v.quality_score));

    // Pick the highest-quality variant whose vram_required fits the
    // larger of (local VRAM, pool VRAM divided by a target replica
    // count). The replica target is fixed at 3 — the swarm wants at
    // least 3 holders for reliability. If even the lowest quant
    // doesn't fit, recommended_index is None.
    const TARGET_REPLICAS: u64 = 3;
    let replica_share_mb = if pool_vram_mb >= TARGET_REPLICAS {
        pool_vram_mb / TARGET_REPLICAS
    } else {
        pool_vram_mb
    };
    let budget_mb = local_vram_mb.max(replica_share_mb);

    let (recommended_index, rationale_tag) = if known_variants.is_empty() {
        (None, "quant.rec.no_variants".to_string())
    } else if budget_mb == 0 {
        // No GPU + no swarm VRAM — recommend the lowest-quality variant
        // since at least it fits on CPU + RAM.
        let lowest = known_variants
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.size_mb)
            .map(|(i, _)| i)
            .unwrap_or(0);
        (Some(lowest), "quant.rec.cpu_only".to_string())
    } else {
        let fit = known_variants
            .iter()
            .enumerate()
            .find(|(_, v)| v.vram_required_mb <= budget_mb);
        match fit {
            Some((idx, _)) => {
                // If this is the highest-quality variant the swarm has, use
                // the "happy path" tag; otherwise note that a better one
                // exists but doesn't fit.
                if idx == 0 {
                    (Some(idx), "quant.rec.best_fit".to_string())
                } else {
                    let too_big = &known_variants[0];
                    (
                        Some(idx),
                        format!(
                            "quant.rec.would_upgrade|next={}&need_mb={}",
                            too_big.quantization.label(),
                            too_big.vram_required_mb
                        ),
                    )
                }
            }
            None => {
                // Nothing fits — recommend the smallest anyway.
                let smallest = known_variants
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| v.vram_required_mb)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (
                    Some(smallest),
                    format!(
                        "quant.rec.too_big|need_mb={}&have_mb={}",
                        known_variants[smallest].vram_required_mb, budget_mb
                    ),
                )
            }
        }
    };

    QuantFamilyRecommendation {
        base_name: base_name.to_string(),
        display_name,
        known_variants,
        recommended_index,
        rationale_tag,
    }
}

fn parse_quant_from_manifest(manifest: &ModelManifest) -> Quantization {
    // Prefer the manifest's own quantization field. Today it's almost
    // always `Q4KM` (placeholder); when it's not, trust it. Otherwise
    // fall back to parsing the model identity.
    match manifest.quantization {
        Quantization::Unknown => parse_quant_from_identity(manifest),
        // Q4KM is also used as a defaulted placeholder in older code
        // paths; if the model identity parses to something different, prefer
        // that. Belt-and-braces — costs ~30 ns.
        Quantization::Q4KM => match parse_quant_from_identity(manifest) {
            Quantization::Unknown => Quantization::Q4KM,
            parsed => parsed,
        },
        q => q,
    }
}

/// Parse a quantization tag out of a manifest's id, then its name.
///
/// The id is tried FIRST because that is where the tag actually lives:
/// `llama-3.2-1b-instruct-q8-0`. `name` is a human display name — "Llama 3.2 1B
/// Instruct" — and carries no tag at all, so parsing it alone meant every model
/// fell through to the `Q4KM` placeholder no matter how good the parser was.
/// v0.3.27 fixed the parser and left this reading the wrong field, so a Q8_0
/// model was still reported as Q4KM (live-confirmed 2026-07-26).
///
/// `name` is still tried as a fallback: some acquisition paths set it from the
/// source filename (`model-q4_k_m.gguf`), which does carry the tag.
fn parse_quant_from_identity(manifest: &ModelManifest) -> Quantization {
    match Quantization::parse(&trailing_tag(&manifest.id.0)) {
        Quantization::Unknown => Quantization::parse(&trailing_tag(&manifest.name)),
        q => q,
    }
}

/// Best-effort quantization tag from the end of a model name.
///
/// Quant tags are multi-part (`q8-0`, `q4-k-m`), so taking only the text after
/// the LAST separator yields `0` or `m` and parses as Unknown. Tries the last
/// three separator-delimited segments, longest first, and returns the first
/// that parses to a real quantization — falling back to the final segment so
/// existing single-part behaviour is unchanged.
fn trailing_tag(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let core = lower.trim_end_matches(".gguf");
    let parts: Vec<&str> = core
        .split(['-', '.', '_'])
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return core.to_string();
    }
    for take in (1..=3.min(parts.len())).rev() {
        let candidate = parts[parts.len() - take..].join("-");
        if Quantization::parse(&candidate) != Quantization::Unknown {
            return candidate;
        }
    }
    parts[parts.len() - 1].to_string()
}

fn base_name_to_display(base: &str, fallback: &str) -> String {
    // Title-case the base name on simple delimiters. If the original
    // manifest name doesn't look raw (mixed case, spaces), prefer it
    // sans-quant-tag.
    let from_manifest = strip_trailing_quant(fallback);
    if from_manifest != fallback.to_ascii_lowercase() && !from_manifest.is_empty() {
        return from_manifest;
    }
    let mut out = String::with_capacity(base.len());
    let mut capitalize_next = true;
    for ch in base.chars() {
        if ch == '-' || ch == '_' || ch == '.' {
            out.push(' ');
            capitalize_next = true;
        } else if capitalize_next {
            out.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn strip_trailing_quant(name: &str) -> String {
    // Drop a trailing `-Q4_K_M.gguf` / `.q4_k_m.gguf` etc. while keeping
    // mixed-case content intact.
    let core = name.trim_end_matches(".gguf");
    for delim in &['-', '.', '_'][..] {
        if let Some(pos) = core.rfind(*delim) {
            let suffix = &core[pos + 1..];
            if Quantization::parse(suffix) != Quantization::Unknown {
                return core[..pos].to_string();
            }
        }
    }
    core.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model IDs carry hyphenated quant tags (`q8-0`, `q4-k-m`) because id
    /// sanitisation rewrites `_` to `-`. Taking only the text after the last
    /// separator gave `0` / `m`, which parsed as Unknown and fell back to the
    /// `Q4KM` placeholder — so a Q8_0 model was reported as Q4KM
    /// (external report 2026-07-26).
    #[test]
    fn hyphenated_quant_tags_in_model_ids_parse_correctly() {
        for (name, want) in [
            ("llama-3.2-1b-instruct-q8-0", Quantization::Q8_0),
            ("qwen2.5-coder-7b-instruct-q4-k-m", Quantization::Q4KM),
            ("tinyllama-1.1b-chat-v1.0.q4-k-m", Quantization::Q4KM),
            ("meta-llama-3.1-8b-instruct-q5-k-s", Quantization::Q5KS),
            ("some-model-q6-k", Quantization::Q6K),
            ("qwen2.5-0.5b-instruct-fp16", Quantization::FP16),
        ] {
            let tag = trailing_tag(name);
            assert_eq!(
                Quantization::parse(&tag),
                want,
                "name {name:?} produced tag {tag:?}"
            );
        }
    }

    /// Underscore filenames must keep working alongside the hyphen form.
    #[test]
    fn underscore_quant_tags_still_parse() {
        for (name, want) in [
            ("llama-3.2-1b-instruct-q8_0.gguf", Quantization::Q8_0),
            ("model-q4_k_m.gguf", Quantization::Q4KM),
        ] {
            assert_eq!(Quantization::parse(&trailing_tag(name)), want, "{name}");
        }
    }

    /// A name with no recognisable quant must not be coerced into one.
    #[test]
    fn names_without_a_quant_tag_stay_unknown() {
        for name in ["llama-3.2-1b-instruct", "some-model", "plainname"] {
            assert_eq!(
                Quantization::parse(&trailing_tag(name)),
                Quantization::Unknown,
                "{name} should not resolve to a quantization"
            );
        }
    }

    #[test]
    fn parse_canonical_quants() {
        assert_eq!(Quantization::parse("Q4_K_M"), Quantization::Q4KM);
        assert_eq!(Quantization::parse("q4_k_m"), Quantization::Q4KM);
        assert_eq!(Quantization::parse("Q8_0"), Quantization::Q8_0);
        assert_eq!(Quantization::parse("IQ3_M"), Quantization::IQ3M);
        assert_eq!(Quantization::parse("F16"), Quantization::FP16);
        assert_eq!(Quantization::parse("BF16"), Quantization::BF16);
        assert_eq!(Quantization::parse("FP32"), Quantization::FP32);
    }

    #[test]
    fn parse_unknown_falls_back() {
        assert_eq!(Quantization::parse(""), Quantization::Unknown);
        assert_eq!(Quantization::parse("ZX99"), Quantization::Unknown);
    }

    #[test]
    fn quality_orders_match_expected_ranking() {
        // Sanity ordering: f16 > q8 > q6 > q5 > q4 > q3 > q2 > iq1 > unknown
        let q = |x: Quantization| x.quality_score();
        assert!(q(Quantization::FP16) > q(Quantization::Q8_0));
        assert!(q(Quantization::Q8_0) > q(Quantization::Q6K));
        assert!(q(Quantization::Q6K) > q(Quantization::Q5KM));
        assert!(q(Quantization::Q5KM) > q(Quantization::Q4KM));
        assert!(q(Quantization::Q4KM) > q(Quantization::Q3KM));
        assert!(q(Quantization::Q3KM) > q(Quantization::Q2K));
        assert!(q(Quantization::Q2K) > q(Quantization::IQ1S));
        assert!(q(Quantization::IQ1S) > q(Quantization::Unknown));
    }

    #[test]
    fn bits_per_weight_ordered_with_quality() {
        // Higher-bit quants are usually higher quality. Float types
        // bracket the top; unknown is the conservative 8.0.
        assert!(Quantization::FP16.bits_per_weight() > Quantization::Q8_0.bits_per_weight());
        assert!(Quantization::Q8_0.bits_per_weight() > Quantization::Q4KM.bits_per_weight());
        assert!(Quantization::Q4KM.bits_per_weight() > Quantization::IQ2S.bits_per_weight());
    }

    #[test]
    fn base_name_strips_canonical_quant_tag() {
        assert_eq!(inferred_base_name("llama-2-7b-Q4_K_M.gguf"), "llama-2-7b");
        assert_eq!(inferred_base_name("tinyllama.q8_0.gguf"), "tinyllama");
        assert_eq!(
            inferred_base_name("phi-3.5-mini-instruct.Q4_K_M"),
            "phi-3.5-mini-instruct"
        );
    }

    #[test]
    fn base_name_passes_through_when_no_tag() {
        assert_eq!(inferred_base_name("plain-model"), "plain-model");
        assert_eq!(inferred_base_name(""), "");
    }

    /// R134.6: auto-action is a no-op when the config flag is off, even
    /// if there's a clear upgrade recommendation.
    #[test]
    fn apply_quant_auto_action_skips_when_flag_off() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use std::sync::Arc;
        use tokio::sync::Mutex;
        let mut config = Config::default();
        // R141 made `true` the user-facing default; this test exercises the
        // explicit-opt-out behaviour, so disable the flag manually.
        config.auto_manage.auto_switch_quants = false;
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);
        let count = apply_quant_auto_action(&state);
        assert_eq!(count, 0);
    }

    /// End-to-end: register two manifests for the same base model at
    /// different quants, verify `compute_quant_recommendations` groups
    /// them and surfaces the highest-quality fit given local VRAM.
    #[test]
    fn recommendation_groups_variants_and_picks_best_fit() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use crate::types::{ModelArchitecture, ModelId, ModelManifest, NodeId, ShardInfo};
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);

        let mk = |id: &str, name: &str, q: Quantization, bytes: u64| ModelManifest {
            id: ModelId(id.into()),
            name: name.into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 32,
            num_params_billions: 7.0,
            quantization: q,
            total_size_bytes: bytes,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 32),
                size_bytes: bytes,
                hash: [0u8; 32],
                tensors: vec![],
            }],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        };

        // Two variants of "llama-2-7b": Q4_K_M (~4 GB) and Q8_0 (~7 GB)
        state.model_registry.register_manifest(mk(
            "llama-2-7b-q4km",
            "llama-2-7b-Q4_K_M",
            Quantization::Q4KM,
            4_000_000_000,
        ));
        state.model_registry.register_manifest(mk(
            "llama-2-7b-q8",
            "llama-2-7b-Q8_0",
            Quantization::Q8_0,
            7_000_000_000,
        ));

        let recs = compute_quant_recommendations(&state);
        assert_eq!(
            recs.families.len(),
            1,
            "two variants should fuse into one family"
        );
        let fam = &recs.families[0];
        assert_eq!(fam.base_name, "llama-2-7b");
        assert_eq!(fam.known_variants.len(), 2);
        // Sorted by quality DESC → Q8_0 first, then Q4_K_M.
        assert_eq!(fam.known_variants[0].quantization, Quantization::Q8_0);
        assert_eq!(fam.known_variants[1].quantization, Quantization::Q4KM);
        // Without GPU info the budget falls through to CPU mode and
        // recommends the smallest. Either way recommended_index must
        // be Some.
        assert!(fam.recommended_index.is_some());
    }

    /// The parser fix in v0.3.27 was correct but read the wrong field: quant
    /// tags live in the model ID (`llama-3.2-1b-instruct-q8-0`), while `name`
    /// is a human display name ("Llama 3.2 1B Instruct") that carries no tag.
    /// Every model therefore still reported the `Q4KM` placeholder
    /// (live-confirmed on a running node 2026-07-26).
    #[test]
    fn quant_is_read_from_the_id_not_the_display_name() {
        use crate::types::{ModelArchitecture, ModelId, ModelManifest, NodeId, ShardInfo};

        let mk = |id: &str, name: &str| ModelManifest {
            id: ModelId(id.into()),
            name: name.into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 16,
            num_params_billions: 1.0,
            // The placeholder every real manifest on disk carries.
            quantization: Quantization::Q4KM,
            total_size_bytes: 1_000_000,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 16),
                size_bytes: 1_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            }],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        };

        for (id, name, want) in [
            (
                "llama-3.2-1b-instruct-q8-0",
                "Llama 3.2 1B Instruct",
                Quantization::Q8_0,
            ),
            (
                "qwen2.5-0.5b-instruct-fp16",
                "qwen2.5-0.5b-instruct",
                Quantization::FP16,
            ),
            (
                "llama-3.2-3b-instruct-q4-k-m",
                "Llama 3.2 3B Instruct",
                Quantization::Q4KM,
            ),
        ] {
            assert_eq!(
                parse_quant_from_manifest(&mk(id, name)),
                want,
                "id {id:?} / name {name:?}"
            );
        }
    }

    /// A name that carries the tag is still honoured when the id does not —
    /// some acquisition paths set `name` from the source filename.
    #[test]
    fn quant_falls_back_to_name_when_id_has_no_tag() {
        use crate::types::{ModelArchitecture, ModelId, ModelManifest, NodeId, ShardInfo};

        let m = ModelManifest {
            id: ModelId("some-opaque-id".into()),
            name: "model-q5_k_s.gguf".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 16,
            num_params_billions: 1.0,
            quantization: Quantization::Unknown,
            total_size_bytes: 1_000_000,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 16),
                size_bytes: 1_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            }],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        };
        assert_eq!(parse_quant_from_manifest(&m), Quantization::Q5KS);
    }
}
