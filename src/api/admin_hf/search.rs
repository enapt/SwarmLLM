use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::{count_unique_shard_holders, gguf_filename_to_model_id};

/// Heuristic shard sizing constants for HF search scoring.
const EST_SHARD_SIZE_BYTES: u64 = 800 * 1024 * 1024;
const EST_SHARD_COUNT_MIN: u64 = 2;
const EST_SHARD_COUNT_MAX: u64 = 16;
const BOOMERANG_SIZE_NUMERATOR: u64 = 12;
const BOOMERANG_SIZE_DENOMINATOR: u64 = 5;
const MODEL_SIZE_SCORE_MAX_GB: f64 = 8.0;

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    #[serde(rename = "q")]
    pub query: Option<String>,
    /// R114: optional task filter — comma-separated tokens
    /// (chat / code / vision / multilingual / reasoning). Results that
    /// don't match any of the requested tokens are dropped. Empty/missing
    /// = no filter (returns everything).
    #[serde(default)]
    pub tasks: Option<String>,
}

pub async fn hf_search(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.query.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }
    // chars().count() so non-ASCII queries don't get rejected by byte
    // counting when the user-facing limit is character-count (R93/R94
    // pool-name pattern).
    if query.chars().count() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Search query too long (max 256 chars)".into(),
        )));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| {
            // Upstream HuggingFace failure → ProviderError, NOT
            // ServiceUnavailable (which is for local-server outages).
            // Matches R93's probe.rs fix; same error-contract rule.
            ApiError(crate::error::SwarmError::ProviderError {
                status: 502,
                body: crate::api::scrub_truncate_error(&e),
            })
        })?;

    // R114: task-filter parsing. Tokens are case-insensitive; unknown
    // tokens are silently ignored so a future filter chip the backend
    // doesn't recognise yet doesn't break the response.
    if let Some(ref t) = params.tasks {
        if t.len() > 512 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "tasks filter too long (max 512 bytes)".into(),
            )));
        }
    }
    let task_filter: Option<std::collections::HashSet<String>> = params.tasks.as_ref().map(|s| {
        s.split(',')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    });

    // Map repo_id → task tags from the HfWatcher's trending cache. For
    // non-trending repos we infer tags from the repo name (best-effort
    // — tags help filtering, not correctness, so a miss just falls
    // back to "uncategorised").
    let trending_snapshot = state.shared_state.models.hf_trending_cache.load_full();
    let mut tags_by_repo: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for e in &trending_snapshot.entries {
        tags_by_repo.insert(e.repo_id.clone(), e.task_tags.clone());
    }

    // How large a model this node could host, on whichever device would run it.
    // Asking the card alone answered zero on every processor-only machine, so
    // nothing ever "fitted" and the frontend's fit filter hid every result of
    // every search (gotcha #483). `None` is unknowable, not zero — see below,
    // where it deliberately does not exclude.
    let node_budget_bytes =
        crate::model::auto_manage::vram::node_model_budget_bytes(&state.shared_state);

    // Group results by repo_id with quant variants (preserve HF API order = by downloads)
    let mut repo_order: Vec<String> = Vec::new();
    let mut repo_map: std::collections::HashMap<
        String,
        Vec<crate::model::huggingface::HfModelResult>,
    > = std::collections::HashMap::new();
    for r in results {
        if !repo_map.contains_key(&r.repo_id) {
            repo_order.push(r.repo_id.clone());
        }
        repo_map.entry(r.repo_id.clone()).or_default().push(r);
    }

    let mut values: Vec<serde_json::Value> = repo_order
        .into_iter()
        .filter_map(|repo_id| {
            let files = repo_map.remove(&repo_id)?;
            Some((repo_id, files))
        })
        .filter_map(|(repo_id, files)| {
            // R114: task tags + filter. Tags from the HfWatcher trending
            // cache (authoritative); fall back to a tiny name-based heuristic
            // for non-trending repos so the chips still do something useful.
            // An embedding model is tagged as what it is, so a "Chat" filter
            // drops it and the row offers nothing (`not_chat` below); every
            // other tag would have called it chat (field report 2026-10-15).
            let embedding = files.iter().any(|f| f.embedding);
            let task_tags: Vec<String> = if embedding {
                vec!["embedding".to_string()]
            } else {
                tags_by_repo
                    .get(&repo_id)
                    .cloned()
                    .unwrap_or_else(|| infer_task_tags_from_repo_name(&repo_id))
            };
            if let Some(ref filter) = task_filter {
                if !filter.is_empty()
                    && !task_tags.iter().any(|t| filter.contains(&t.to_lowercase()))
                {
                    return None;
                }
            }
            let downloads = files.first().map(|f| f.downloads).unwrap_or(0);
            let likes = files.first().map(|f| f.likes).unwrap_or(0);

            let variants: Vec<serde_json::Value> = files
                .iter()
                .map(|f| {
                    let quant = crate::model::huggingface::extract_quant_tag(&f.filename)
                        .unwrap_or_else(|| "unknown".into());
                    serde_json::json!({
                        "filename": f.filename,
                        "size_bytes": f.size_bytes,
                        "quant": quant,
                        // Why no node could host this file, by its name
                        // (`huggingface::cannot_host`); null when nothing says so.
                        "cannot_host": crate::model::huggingface::cannot_host(&f.filename)
                            .map(|why| why.as_str()),
                    })
                })
                .collect();

            let (recommended, runs_here) = offered_here(&files);

            let recommended_variant = recommended
                .and_then(|f| crate::model::huggingface::extract_quant_tag(&f.filename))
                .unwrap_or_else(|| "unknown".into());

            // VRAM fit levels: full model, boomerang (first+last shard), single shard
            let rec_size = recommended
                .map(|f| f.size_bytes)
                .unwrap_or(files.iter().map(|f| f.size_bytes).min().unwrap_or(u64::MAX));
            let est_shards =
                (rec_size / EST_SHARD_SIZE_BYTES).clamp(EST_SHARD_COUNT_MIN, EST_SHARD_COUNT_MAX);
            let est_shard_size = rec_size / est_shards;
            // Boomerang: first + last shard (~2.4x one shard due to embedding/output weights)
            let est_boomerang_size =
                est_shard_size * BOOMERANG_SIZE_NUMERATOR / BOOMERANG_SIZE_DENOMINATOR;

            // Unknown must not exclude: a node whose memory could not be read
            // is not a node with no room, and reporting "nothing fits" there
            // empties the browser for a user who cannot tell why.
            // Nothing fits where nothing loads: a family this build has no
            // loader for is "fits nowhere here" (`offered_here`).
            let fits =
                |size: u64| runs_here && node_budget_bytes.is_none_or(|budget| size < budget);
            let fits_full = fits(rec_size);
            let fits_boomerang = fits(est_boomerang_size);
            let fits_shard = fits(est_shard_size);
            // True if any participation mode fits
            let fits_vram = fits_full || fits_boomerang || fits_shard;

            // Network replication: count unique peers holding shards of any variant of this repo
            let variant_ids: Vec<crate::types::ModelId> = files
                .iter()
                .map(|f| crate::types::ModelId(gguf_filename_to_model_id(&f.filename)))
                .collect();
            let network_replicas =
                count_unique_shard_holders(&state.shared_state.model_registry, &variant_ids);
            // The variant the swarm can RUN right now — every part held by a
            // computer that can serve it — preferring the recommended one.
            // `network_replicas` counts copies of any part, which is not that:
            // one computer holding a model's first part is a copy, and nothing
            // anyone can chat with. The browser offers "Chat" on this, where it
            // offered Download to a node that cannot hold even one part.
            let registry = &state.shared_state.model_registry;
            let swarm_model_id = recommended
                .map(|f| crate::types::ModelId(gguf_filename_to_model_id(&f.filename)))
                .into_iter()
                .chain(variant_ids.iter().cloned())
                .find(|id| {
                    registry
                        .get_manifest(id)
                        .is_some_and(|m| registry.every_shard_has_a_holder(&m))
                });

            // Composite score: surfaces small, popular, scarce, VRAM-fitting models
            let quality = (downloads as f64 + 10.0).log10() / 7.0; // 0-1 popularity proxy
            let fit = if fits_boomerang {
                1.0
            } else if fits_shard {
                0.6
            } else {
                0.1
            };
            let demand = if network_replicas == 0 {
                1.5
            } else if network_replicas < 3 {
                1.2
            } else if network_replicas < 10 {
                1.0
            } else {
                0.7
            };
            let shard_gb = rec_size as f64 / (1024.0 * 1024.0 * 1024.0);
            let size_factor = (1.0 - shard_gb / MODEL_SIZE_SCORE_MAX_GB).clamp(0.1, 1.0);
            let composite_score = (quality * fit * demand * size_factor * 100.0) as u32;

            // R114: status-driven CTA. Drives a single button per result
            // instead of forcing the user to interpret a 0..100 composite
            // score. The mapping mirrors the wishlist's status taxonomy
            // (Hosting / Serveable / Aspirational / Unreachable / Blocked)
            // so non-technical users see consistent language across views.
            //
            // When the local node can't fit *any* shard, the model is still
            // reachable through remote inference — but only if the swarm holds
            // EVERY part of some variant. Holders of some parts
            // (`network_replicas > 0`) were read as that, and are not.
            let status = if embedding {
                // Downloading it is refused (`GgufTensorMeta::refusal`), and
                // nothing on the swarm can chat with it.
                "not_chat"
            } else if !runs_here {
                // HuggingFace names a family this build has no loader for —
                // every file of it is refused at the header, whatever its
                // quantization (Qwen3.8-Flash-Next's `qwen4exp`, field report
                // 2026-11-03). Where the swarm already runs it (a computer
                // with a newer loader), the row offers Chat through it.
                if swarm_model_id.is_some() {
                    "swarm_serveable"
                } else {
                    "family_unsupported"
                }
            } else if recommended.is_none() {
                // Every file is one no node could host (an IQ quantization, a
                // split upload, a file in a folder): the row offers nothing
                // to press, and each variant says why.
                "cannot_host"
            } else if !fits_full && !fits_boomerang && !fits_shard {
                if swarm_model_id.is_some() {
                    "swarm_serveable"
                } else {
                    "unreachable"
                }
            } else if network_replicas == 0 && (fits_boomerang || fits_full) {
                "be_first_host"
            } else if network_replicas > 0 && network_replicas < 3 {
                "needs_more_hosts"
            } else if network_replicas >= 3 {
                "well_replicated"
            } else {
                "downloadable"
            };

            Some(serde_json::json!({
                "repo_id": repo_id,
                "downloads": downloads,
                "likes": likes,
                "variants": variants,
                "recommended_variant": recommended_variant,
                // The file itself: two files can carry one quantization (one
                // in a folder), so the browser selects by name.
                "recommended_filename": recommended.map(|f| f.filename.as_str()),
                // The family HuggingFace read off the header; null if none.
                "architecture": files.iter().find_map(|f| f.architecture.as_deref()),
                "fits_vram": fits_vram,
                // Whether the WHOLE recommended file fits this node's memory
                // budget — the only fit that makes "Runs locally" true, and
                // the one that decides whether Download fetches every part.
                // The budget itself is sent too (null when unknown, which
                // counts as fitting, as above), so the browser can judge a
                // variant the user picks instead of the recommended one.
                "fits_full": fits_full,
                "memory_budget_bytes": node_budget_bytes,
                "fits_boomerang": fits_boomerang,
                "fits_shard": fits_shard,
                "est_shard_size": est_shard_size,
                "est_boomerang_size": est_boomerang_size,
                "network_replicas": network_replicas,
                "swarm_model_id": swarm_model_id.map(|m| m.0),
                "composite_score": composite_score,
                "score_breakdown": {
                    "quality": (quality * 100.0) as u32,
                    "fit": (fit * 100.0) as u32,
                    "demand": (demand * 100.0) as u32,
                    "size": (size_factor * 100.0) as u32,
                },
                "task_tags": task_tags,
                "swarm_cta_status": status,
            }))
        })
        .collect();

    // Sort by composite score descending (best-fit models first)
    values.sort_by(|a, b| {
        let sa = a
            .get("composite_score")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let sb = b
            .get("composite_score")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        sb.cmp(&sa)
    });

    Ok(Json(values))
}

/// R114: tiny name-based tag inference for non-trending repos. The
/// HfWatcher trending cache is the authoritative source; this fallback
/// just makes the chips do something useful for repos HfWatcher hasn't
/// seen. Best-effort; users can always remove a filter to see everything.
fn infer_task_tags_from_repo_name(repo_id: &str) -> Vec<String> {
    let lower = repo_id.to_lowercase();
    let mut tags: Vec<String> = Vec::new();
    if lower.contains("code")
        || lower.contains("coder")
        || lower.contains("starcoder")
        || lower.contains("granite-code")
    {
        tags.push("code".to_string());
    }
    if lower.contains("vision")
        || lower.contains("vlm")
        || lower.contains("llava")
        || lower.contains("multimodal")
    {
        tags.push("vision".to_string());
    }
    if lower.contains("math")
        || lower.contains("reasoning")
        || lower.contains("o1")
        || lower.contains("deepseek-r1")
    {
        tags.push("reasoning".to_string());
    }
    if lower.contains("multilingual")
        || lower.contains("aya")
        || lower.contains("nllb")
        || lower.contains("madlad")
    {
        tags.push("multilingual".to_string());
    }
    // Default fallback so the chips always do something. Most GGUFs
    // are chat / instruct fine-tunes anyway.
    if tags.is_empty() || lower.contains("chat") || lower.contains("instruct") {
        tags.push("chat".to_string());
    }
    tags
}

/// The file of a repo the search recommends: Q4_K_M, else the smallest Q4,
/// else the first — of the files this node could host only
/// (`huggingface::cannot_host`). The fallback took the first file whatever it
/// was, and a tester was recommended an IQ2_XS in a repository folder that no
/// node could take (field report 2026-11-03). `None`: no file can be hosted.
fn recommended_file(
    files: &[crate::model::huggingface::HfModelResult],
) -> Option<&crate::model::huggingface::HfModelResult> {
    use crate::model::huggingface::{cannot_host, extract_quant_tag};
    let hostable = || files.iter().filter(|f| cannot_host(&f.filename).is_none());
    let tag = |f: &crate::model::huggingface::HfModelResult| extract_quant_tag(&f.filename);
    // An IQ4_NL / IQ4_XS file is hostable but runs only on the processor
    // until a card kernel reads it (#404), so any other hostable file first.
    let card_reads = |f: &&crate::model::huggingface::HfModelResult| {
        !tag(f).is_some_and(|q| q.starts_with("IQ"))
    };
    hostable()
        .find(|f| tag(f).as_deref() == Some("Q4_K_M"))
        .or_else(|| {
            hostable()
                .filter(|f| tag(f).is_some_and(|q| q.starts_with("Q4")))
                .min_by_key(|f| f.size_bytes)
        })
        .or_else(|| hostable().find(card_reads))
        .or_else(|| hostable().next())
}

/// What a search row offers THIS node: the file to fetch, and whether
/// anything of the repo can run here. A family this build has no loader for
/// offers neither — the search flagged one `family_unsupported` and still said
/// `fits_vram: true` and recommended a file, sometimes a projector (field
/// report 2026-11-04); a row the swarm runs said "Runs locally" in the browser.
fn offered_here(
    files: &[crate::model::huggingface::HfModelResult],
) -> (Option<&crate::model::huggingface::HfModelResult>, bool) {
    if family_unsupported(files) {
        (None, false)
    } else {
        (recommended_file(files), true)
    }
}

/// HuggingFace names a model family this build has no loader for — the
/// loader's own answer (`ModelArch::is_supported`), read before any header is
/// fetched. A repo HuggingFace named no family for is not judged.
fn family_unsupported(files: &[crate::model::huggingface::HfModelResult]) -> bool {
    files
        .iter()
        .find_map(|f| f.architecture.as_deref())
        .is_some_and(|arch| {
            !crate::inference::split::ModelArch::from_gguf_arch(arch).is_supported()
        })
}

#[cfg(test)]
mod tests {
    use super::{family_unsupported, offered_here, recommended_file};
    use crate::model::huggingface::HfModelResult;

    fn files(names: &[(&str, u64)]) -> Vec<HfModelResult> {
        names
            .iter()
            .map(|&(filename, size_bytes)| HfModelResult {
                repo_id: "someone/model-GGUF".into(),
                filename: filename.into(),
                size_bytes,
                downloads: 0,
                likes: 0,
                embedding: false,
                architecture: None,
            })
            .collect()
    }

    fn of_family(arch: Option<&str>) -> Vec<HfModelResult> {
        let mut repo = files(&[("m-Q4_K_M.gguf", 1)]);
        repo[0].architecture = arch.map(String::from);
        repo
    }

    /// **A family this build cannot load is said up front** (field report
    /// 2026-11-03: Qwen3.8-Flash-Next is `qwen4exp`, so even its K-quant
    /// would be refused at the header after the click). A family the loader
    /// runs, or one HuggingFace did not name, is not.
    #[test]
    fn a_family_this_build_cannot_load_is_known_before_the_download() {
        assert!(family_unsupported(&of_family(Some("qwen4exp"))));
        assert!(family_unsupported(&of_family(Some("no-such-family"))));
        assert!(!family_unsupported(&of_family(Some("qwen2"))));
        assert!(!family_unsupported(&of_family(Some("llama"))));
        assert!(!family_unsupported(&of_family(None)));
    }

    /// **A family this build cannot load is offered nothing here** (field
    /// report 2026-11-04: flagged `family_unsupported`, yet `fits_vram: true`
    /// and a recommended file). No file, and no fit; a family it runs, or one
    /// HuggingFace did not name, keeps both.
    #[test]
    fn a_family_this_build_cannot_load_offers_no_file_and_fits_nowhere_here() {
        let unknown = of_family(Some("qwen4exp"));
        let (file, runs_here) = offered_here(&unknown);
        assert!(file.is_none(), "no file is recommended");
        assert!(!runs_here, "and nothing of it fits here");
        for arch in [Some("qwen2"), None] {
            let repo = of_family(arch);
            let (file, runs_here) = offered_here(&repo);
            assert_eq!(file.map(|f| f.filename.as_str()), Some("m-Q4_K_M.gguf"));
            assert!(runs_here, "{arch:?}");
        }
    }

    fn pick(names: &[(&str, u64)]) -> Option<String> {
        recommended_file(&files(names)).map(|f| f.filename.clone())
    }

    /// **The search recommends only a file this node could host** (field
    /// report 2026-11-03: an IQ2_XS in a repository folder). Q4_K_M first,
    /// then the smallest Q4, then any hostable file a card reads, then IQ4_NL /
    /// IQ4_XS (hostable since #404, processor only); any other IQ
    /// quantization, one file of a split upload or a file in a folder never —
    /// and a repo of nothing else has no recommendation.
    #[test]
    fn the_search_recommends_only_a_file_this_node_could_host() {
        assert_eq!(
            pick(&[("m-IQ2_XS.gguf", 1), ("m-Q4_K_M.gguf", 9)]).as_deref(),
            Some("m-Q4_K_M.gguf")
        );
        assert_eq!(
            pick(&[
                ("m-IQ4_XS.gguf", 1),
                ("m-Q4_K_S.gguf", 5),
                ("m-Q4_0.gguf", 4)
            ])
            .as_deref(),
            Some("m-Q4_0.gguf"),
            "the smallest Q4 — never IQ4_XS"
        );
        assert_eq!(
            pick(&[("m-IQ3_M.gguf", 1), ("m-Q3_K_M.gguf", 3)]).as_deref(),
            Some("m-Q3_K_M.gguf"),
            "the first HOSTABLE file, not the first file"
        );
        assert_eq!(
            pick(&[("m-IQ4_NL.gguf", 1), ("m-Q3_K_M.gguf", 3)]).as_deref(),
            Some("m-Q3_K_M.gguf"),
            "a file a card reads before an IQ4"
        );
        assert_eq!(
            pick(&[("m-IQ2_XS.gguf", 1), ("m-IQ4_XS.gguf", 3)]).as_deref(),
            Some("m-IQ4_XS.gguf"),
            "an IQ4 where nothing else is hostable"
        );
        assert_eq!(
            pick(&[
                ("IQ2_XS/m-IQ2_XS.gguf", 1),
                ("Q4_K_M/m-Q4_K_M.gguf", 2),
                ("m-Q8_0-00001-of-00002.gguf", 3),
                ("m-IQ3_XXS.gguf", 4),
            ]),
            None
        );
    }
}
