use crate::api::server::AppState;
use crate::types::ModelId;

use super::peer_forward::peer_http_url;

/// Find a peer that hosts shards for this model and return its HTTP base URL.
/// This is a fallback for when not all layers are covered network-wide — the
/// peer may be able to handle the request directly or assemble its own pipeline.
pub(super) fn find_peer_with_model(state: &AppState, model: &str) -> Option<String> {
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            let has_model = cap.hosted_shards.iter().any(|s| s.model_id.0 == model);
            if has_model {
                if let Some(url) = peer_http_url(peer) {
                    return Some(url);
                }
            }
        }
    }
    None
}

/// Short-lived per-model TTL for the `all_shards_available` cache. Kept tiny
/// to stay consistent with shard announcement propagation delay.
const SHARD_AVAIL_CACHE_TTL_MS: u64 = 100;

/// Check if all layers for a model are covered across the network (for distributed inference).
/// This does NOT require any single node to have all shards — it only requires that every
/// shard has at least one holder somewhere in the network so the pipeline scheduler can
/// assemble a complete pipeline across multiple nodes.
pub fn all_shards_available(state: &AppState, model_name: &str) -> bool {
    // Short-lived per-model cache to avoid repeated ModelId/ShardId allocations
    use std::sync::LazyLock;
    static CACHE: LazyLock<dashmap::DashMap<String, (std::time::Instant, bool)>> =
        LazyLock::new(dashmap::DashMap::new);
    let ttl = std::time::Duration::from_millis(SHARD_AVAIL_CACHE_TTL_MS);
    if let Some(entry) = CACHE.get(model_name) {
        if entry.0.elapsed() < ttl {
            return entry.1;
        }
    }

    let result = all_shards_available_inner(state, model_name);
    // Evict stale entries instead of clearing entire cache (prevents thundering herd)
    const SHARD_AVAIL_CACHE_MAX: usize = 1_000;
    if CACHE.len() > SHARD_AVAIL_CACHE_MAX {
        CACHE.retain(|_, (ts, _)| ts.elapsed() < ttl);
    }
    CACHE.insert(model_name.to_string(), (std::time::Instant::now(), result));
    result
}

fn all_shards_available_inner(state: &AppState, model_name: &str) -> bool {
    let model_id = ModelId(model_name.to_string());

    let manifest = match state.shared_state.model_registry.get_manifest(&model_id) {
        Some(m) => m,
        None => {
            tracing::debug!(model = %model_name, "all_shards_available: no manifest");
            return false;
        }
    };

    // Need a valid layer count for the scheduler to work
    if manifest.num_layers == 0 {
        tracing::debug!(model = %model_name, "all_shards_available: num_layers=0");
        return false;
    }

    let total = manifest.shards.len();
    let mut covered = 0;
    for shard_info in &manifest.shards {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        let holders = state.shared_state.model_registry.shard_holders(&shard_id);
        if holders.is_empty() {
            tracing::debug!(
                model = %model_name,
                shard = shard_info.index,
                "all_shards_available: no node in network holds this shard"
            );
            return false;
        }
        covered += 1;
    }

    tracing::debug!(
        model = %model_name,
        shards = total,
        covered,
        num_layers = manifest.num_layers,
        "all_shards_available: all layers covered across network"
    );
    true
}

/// Resolve a model name for inference: handles "auto" alias and display-name → registry-ID mapping.
///
/// 1. If a local model is loaded and the request matches (by "auto", registry ID, or display name),
///    returns the resolved registry ID.
/// 2. For "auto" with no loaded model, falls back to the first model in the registry.
/// 3. Otherwise returns the original model name unchanged.
pub async fn resolve_model_for_inference(state: &AppState, model: &str) -> String {
    let info = state.shared_state.loaded_model_info.read().await;
    if let Some(i) = info.as_ref() {
        let resolved = resolve_loaded_model_registry_id(state, &i.name);
        if model == "auto" || model == resolved || model == i.name {
            return resolved;
        }
    }
    if model == "auto" {
        if let Some(m) = state
            .shared_state
            .model_registry
            .models()
            .into_iter()
            .next()
        {
            return m.id.0.clone();
        }
    }
    model.to_string()
}

/// Resolve a loaded model's display name to its registry ID.
///
/// Looks up the model in the registry by slug, then by display name.
/// Returns the registry ID if found, otherwise returns the slugified name.
pub fn resolve_loaded_model_registry_id(state: &AppState, model_display_name: &str) -> String {
    let slug = crate::types::slugify_model_name(model_display_name);
    state
        .shared_state
        .model_registry
        .get_manifest(&crate::types::ModelId(slug.clone()))
        .map(|m| m.id.0.clone())
        .or_else(|| {
            state
                .shared_state
                .model_registry
                .models()
                .into_iter()
                .find(|m| m.name == model_display_name)
                .map(|m| m.id.0.clone())
        })
        .unwrap_or(slug)
}

/// Check if a requested model matches the currently loaded model.
///
/// Returns true if the request matches by slug, display name, or registry ID.
pub fn model_matches_loaded(state: &AppState, loaded_name: &str, requested: &str) -> bool {
    let slug = crate::types::slugify_model_name(loaded_name);
    requested == slug
        || requested == loaded_name
        || state
            .shared_state
            .model_registry
            .get_manifest(&crate::types::ModelId(requested.to_string()))
            .is_some()
}

/// Resolve the model name to look up for inference (returns display name if available).
///
/// Checks loaded_model_info cache first, verifying it matches the request.
/// Falls back to manifest registry. Used by openai and anthropic handlers.
pub async fn resolve_model_name(state: &AppState, requested_model: &str) -> Option<String> {
    let info = state.shared_state.loaded_model_info.read().await;
    let cached_name = info.as_ref().map(|i| i.name.clone());
    let matches_request = info.as_ref().is_some_and(|i| {
        let slug = crate::types::slugify_model_name(&i.name);
        slug == requested_model || i.name == requested_model
    });
    drop(info);
    if matches_request {
        cached_name
    } else {
        state
            .shared_state
            .model_registry
            .get_manifest(&crate::types::ModelId(requested_model.to_string()))
            .map(|m| m.name.clone())
    }
}

/// Metadata extracted from a split model entry for prompt building.
pub struct SplitModelMeta {
    pub chat_template: Option<String>,
    pub bos_token: String,
    pub eos_token_str: String,
    pub layer_range: (u32, u32),
}

/// Look up the metadata of a locally-held split model that covers the WHOLE
/// model. Used by the OpenAI and Anthropic local-complete fast paths.
///
/// **`is_complete` is load-bearing, not a nicety.** The fast path hands
/// `layer_range` straight to the worker and then feeds it the raw PROMPT, so
/// the entry must own both the embedding table and the output head. A node can
/// hold SEVERAL entries for one model — `split_models` is keyed by
/// `(model, layer_start, layer_end)` — for instance a whole-model entry plus a
/// tail-only `[21,28)` one it serves as a pipeline segment for peers.
///
/// This used to take the first entry matching the model id, while the caller
/// decided whether to use the fast path via `has_complete_split_model`, which
/// asks whether ANY entry is complete. Two questions, two answers, and
/// `split_models` is a `DashMap` whose iteration order is arbitrary — so the
/// fast path could be entered on the strength of the whole-model entry and then
/// run against the tail-only one. The worker dutifully loaded layers 21..28,
/// found no embedding table, and pushed the token ids into the first block:
/// `attn_norm: shape mismatch in rms-norm [1, 128] [3072]` — `[batch, seq_len]`
/// token ids where hidden states were expected. It reproduced only after a node
/// had picked up a second role for a model, which is why it looked like a
/// tail-segment bug rather than a selection bug (external report, 2026-07-27).
pub fn get_split_model_meta(
    shared_state: &crate::daemon::SharedState,
    model_id: &crate::types::ModelId,
) -> Option<SplitModelMeta> {
    shared_state
        .split_models
        .iter()
        .find(|e| e.key().0 == *model_id && e.value().is_complete)
        .map(|entry| {
            let e = entry.value();
            SplitModelMeta {
                chat_template: e.cached_chat_template.clone(),
                bos_token: e.bos_token.clone(),
                eos_token_str: e.eos_token_str.clone(),
                layer_range: (e.layer_start as u32, e.layer_end as u32),
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::split::SplitModelEntry;

    fn make_shared_state() -> std::sync::Arc<crate::daemon::SharedState> {
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

    fn entry(layer_start: usize, layer_end: usize, is_complete: bool) -> SplitModelEntry {
        SplitModelEntry {
            last_used: std::sync::atomic::AtomicU64::new(0),
            estimated_vram_mb: 0,
            is_complete,
            eos_tokens: vec![],
            eos_token_str: String::new(),
            bos_token: String::new(),
            cached_chat_template: None,
            vocab: None,
            layer_start,
            layer_end,
        }
    }

    /// A node serving a tail segment for peers ALSO holds the whole model. The
    /// local fast path must run against the whole-model entry — picking the
    /// tail-only one feeds prompt token ids to a block that expects hidden
    /// states. Both insertion orders, because `split_models` is a `DashMap` and
    /// the original bug was hidden by its arbitrary iteration order.
    #[test]
    fn fast_path_meta_never_returns_a_partial_entry() {
        for tail_first in [true, false] {
            let state = make_shared_state();
            let mid = ModelId("llama-3.2-3b-instruct-q4-k-m".into());
            let complete = (mid.clone(), 0, 28);
            let tail = (mid.clone(), 21, 28);
            if tail_first {
                state
                    .split_models
                    .insert(tail.clone(), entry(21, 28, false));
                state
                    .split_models
                    .insert(complete.clone(), entry(0, 28, true));
            } else {
                state
                    .split_models
                    .insert(complete.clone(), entry(0, 28, true));
                state
                    .split_models
                    .insert(tail.clone(), entry(21, 28, false));
            }

            let meta = get_split_model_meta(&state, &mid).expect("whole-model entry is present");
            assert_eq!(
                meta.layer_range,
                (0, 28),
                "fast path picked a partial entry (tail inserted first: {tail_first})"
            );
        }
    }

    /// Holding ONLY a tail segment must not qualify for the local fast path at
    /// all — the caller has to route the request through the pipeline instead.
    #[test]
    fn a_tail_only_node_has_no_fast_path_meta() {
        let state = make_shared_state();
        let mid = ModelId("llama-3.2-3b-instruct-q4-k-m".into());
        state
            .split_models
            .insert((mid.clone(), 21, 28), entry(21, 28, false));

        assert!(get_split_model_meta(&state, &mid).is_none());
        assert!(!state.has_complete_split_model(&mid));
    }
}
