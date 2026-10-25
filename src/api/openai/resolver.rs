use crate::api::server::AppState;
use crate::types::ModelId;

use super::peer_forward::peer_http_url;

/// A pool peer to forward a request to, and what waiting on it must allow for.
pub(super) struct PeerTarget {
    /// Its HTTP base URL.
    pub(super) url: String,
    /// Whether it may first have to load the model — asked here, where the
    /// peer and the model are both known, so the forward cannot skip it.
    pub(super) load: crate::inference::pipeline::LoadAllowance,
}

/// A place in the cold-start wait, or — for a request that finds every place
/// taken — the answer the wait gives when nothing arrives
/// (`SharedState::no_route_error`): a 404 naming the models this node knows,
/// `NoModelLoaded` only for a node that knows none. It was `NoModelLoaded` for
/// every node, which told one holding models to go and download one (#324).
/// Not a "busy" class: only a model with no manifest here reaches the wait (a
/// known one goes to the router), and a retry after its manifest lands skips
/// the wait altogether.
pub(super) fn cold_start_slot<'a>(
    state: &crate::daemon::SharedState,
    model: &ModelId,
    slots: &'a tokio::sync::Semaphore,
) -> Result<tokio::sync::SemaphorePermit<'a>, crate::error::SwarmError> {
    slots.try_acquire().map_err(|_| state.no_route_error(model))
}

/// Find a peer that hosts shards for this model and return how to reach it.
/// This is a fallback for when not all layers are covered network-wide — the
/// peer may be able to handle the request directly or assemble its own pipeline.
///
/// `Err` when a peer would be chosen and a privacy setting forbids handing it
/// the request whole (`SharedState::privacy_refuses_a_whole_hand_off`): the
/// peer renders the prompt from its text, and this node — which does not know
/// the model — has no other way to serve it, so the caller hears the setting's
/// refusal, not a "no model loaded" after the cold-start wait (#316's review).
pub(super) fn find_peer_with_model(
    shared: &crate::daemon::SharedState,
    model: &str,
    asked: crate::types::KeepAsked,
) -> Result<Option<PeerTarget>, crate::error::SwarmError> {
    let model_id = ModelId(model.to_string());
    // The request's scope (#251) — a model kept on this computer, or a request
    // whose caller asked to keep it here, is never forwarded, and private mode
    // bounds the rest. Asked beside the pool check below, never instead of it:
    // no restriction is not trust.
    let scope = crate::pool::scope::request_scope(shared, &model_id, asked);
    for entry in shared.peer_registry.iter() {
        if scope.as_ref().is_some_and(|s| !s.contains(entry.key())) {
            continue;
        }
        // SECURITY: forwarding sends this node's `Authorization` header
        // verbatim, and that header carries `shared_state.api_key` — the SAME
        // key that guards `/api/admin/*`. Over plain HTTP, to an address taken
        // from gossip.
        //
        // Without this gate the selection loop accepted ANY peer whose gossiped
        // `NodeCapability.hosted_shards` mentioned the model, so a node had only
        // to claim a popular model to be handed the admin key of everyone who
        // asked for it — no shards, no work, no interaction required. Whoever
        // held it could then read and rewrite config, delete models, mint pool
        // invites and spend credits on any of those nodes whose API they could
        // reach. Confirmed reachable on the public swarm 2026-08-04: a local
        // request produced an outbound POST to `http://<public-ip>:8800`.
        //
        // A peer is only eligible if the operator admitted it to their pool,
        // which is a deliberate act (issuing or accepting a `swarmpool://`
        // invite). Everything else falls through to the libp2p path below —
        // which is where this should go anyway: `pipeline::remote_generate`
        // covers exactly this case ("one remote peer holds the entire layer
        // range") over an authenticated, encrypted transport with no credential
        // crossing the boundary, and is documented as 5-7x faster than the
        // per-token alternative. This HTTP hop is a fallback that was quietly
        // costing more than it bought.
        if !crate::pool::scope::is_pool_member(shared, entry.key()) {
            continue;
        }
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            let has_model = cap.hosted_shards.iter().any(|s| s.model_id.0 == model);
            if has_model {
                if let Some(url) = peer_http_url(peer) {
                    // No route override applies yet (the router records one),
                    // and the automatic default needs this node to hold the
                    // model, which it does not here: the model form is exact.
                    let start_here = shared.encrypted_pipeline_for(&model_id);
                    if let Some(refusal) =
                        shared.privacy_refuses_a_whole_hand_off(&model_id, start_here)
                    {
                        return Err(refusal);
                    }
                    let load = crate::inference::pipeline::LoadAllowance::for_peer(
                        shared,
                        entry.key(),
                        &model_id,
                    );
                    return Ok(Some(PeerTarget { url, load }));
                }
            }
        }
    }
    Ok(None)
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
    // `ModelRegistry::model_is_servable` is the predicate; this wrapper adds the
    // short cache above and the diagnosis line below. Asking the registry rather
    // than re-deriving it here is what stopped `/v1/models` and the
    // "Available models" list in a 404 from disagreeing about the same fact.
    let servable = state
        .shared_state
        .model_registry
        .model_is_servable(&model_id);
    if !servable {
        tracing::debug!(
            model = %model_name,
            "all_shards_available: no manifest, no layer count, or some shard has no holder"
        );
    }
    servable
}

/// Resolve a model name for inference: handles "auto" alias and display-name → registry-ID mapping.
///
/// 1. `"auto"` → [`resolve_auto`].
/// 2. The loaded model's display name or registry ID → its registry ID.
/// 3. Otherwise the original model name, unchanged.
pub async fn resolve_model_for_inference(state: &AppState, model: &str) -> String {
    if model == "auto" {
        return resolve_auto(state)
            .await
            .unwrap_or_else(|| model.to_string());
    }
    let info = state.shared_state.loaded_model_info.read().await;
    if let Some(i) = info.as_ref() {
        let resolved = resolve_loaded_model_registry_id(state, &i.name);
        if model == resolved || model == i.name {
            return resolved;
        }
    }
    model.to_string()
}

/// What `auto` names: the model a caller with no preference gets.
///
/// In order — and past the first two, every step asks whether a model can be
/// SERVED, never only whether it is listed:
///
/// 1. the owner's `inference.default_model`, when it names a model this node
///    knows (documented as "Default model. Empty = first available", and read by
///    nothing until 2026-09-26);
/// 2. a whole model file this node was started with (`-m`), an owner's choice too;
/// 3. the model `auto` answered with last time, while it can still be served —
///    so a conversation held on `auto` stays on one model, and on its prefix cache;
/// 4. a model this node holds whole;
/// 5. a model the swarm can serve (every shard has a holder);
/// 6. anything listed.
///
/// Steps 3-6 skip a model that cannot write replies (`refused_model`) and one
/// with its safety training removed that the owner did not choose
/// (`unchosen_uncensored_model`): nobody named either.
///
/// Ties go by model id, so the answer never depends on a map's iteration order.
///
/// **What it replaced** (#120): `auto` took `loaded_model_info`, which the shard
/// scan overwrites with EVERY model it registers — partial holdings included —
/// and before anything had loaded, the registry's first entry in map order. Four
/// seconds after a restart it named a 14B this node held 14 of 48 layers of, and
/// the request was handed to two peers in turn, both of which refused it for
/// memory. The same overwrite could switch `auto` to another model between two
/// turns of one conversation whenever a scan ran.
pub async fn resolve_auto(state: &AppState) -> Option<String> {
    auto_model_for(&state.shared_state).await
}

/// [`resolve_auto`] on the node's state alone, so it can be tested without a server.
pub(crate) async fn auto_model_for(ss: &crate::daemon::SharedState) -> Option<String> {
    let default = ss.cfg().inference.default_model.clone();
    if !default.is_empty()
        && ss
            .model_registry
            .get_manifest(&crate::types::ModelId(default.clone()))
            .is_some()
    {
        return Some(default);
    }
    if ss.model_loaded.load(std::sync::atomic::Ordering::Relaxed) {
        let info = ss.loaded_model_info.read().await;
        if let Some(i) = info.as_ref() {
            return Some(registry_id_for_display_name(ss, &i.name));
        }
    }

    let whole_here = |id: &crate::types::ModelId| ss.has_complete_split_model(id);
    let swarm_serves = |id: &crate::types::ModelId| ss.model_registry.model_is_servable(id);
    // `auto` is a model nobody named, so never one that cannot write replies
    // (an embedding model) nor one with its safety training removed that the
    // owner did not choose (field report 2026-10-15).
    let may_answer = |id: &crate::types::ModelId| {
        ss.refused_model(id).is_none() && !ss.unchosen_uncensored_model(id)
    };
    let previous = ss.models.auto_model.lock().clone();
    if let Some(prev) = previous {
        if (whole_here(&prev) || swarm_serves(&prev)) && may_answer(&prev) {
            return Some(prev.0);
        }
    }
    let mut ids: Vec<crate::types::ModelId> = ss
        .model_registry
        .models()
        .into_iter()
        .map(|m| m.id)
        .filter(|id| may_answer(id))
        .collect();
    ids.sort_by(|a, b| a.0.cmp(&b.0));
    let choice = ids
        .iter()
        .find(|id| whole_here(id))
        .or_else(|| ids.iter().find(|id| swarm_serves(id)))
        .or_else(|| ids.first())?
        .clone();
    *ss.models.auto_model.lock() = Some(choice.clone());
    Some(choice.0)
}

/// Resolve a loaded model's display name to its registry ID.
///
/// Looks up the model in the registry by slug, then by display name.
/// Returns the registry ID if found, otherwise returns the slugified name.
pub fn resolve_loaded_model_registry_id(state: &AppState, model_display_name: &str) -> String {
    registry_id_for_display_name(&state.shared_state, model_display_name)
}

fn registry_id_for_display_name(
    ss: &crate::daemon::SharedState,
    model_display_name: &str,
) -> String {
    let slug = crate::types::slugify_model_name(model_display_name);
    ss.model_registry
        .get_manifest(&crate::types::ModelId(slug.clone()))
        .map(|m| m.id.0.clone())
        .or_else(|| {
            ss.model_registry
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

    /// A complete split entry whose shard files are gone must NOT win the local
    /// fast path.
    ///
    /// Observed live 2026-08-04: a node holding zero shard files for a model
    /// still claimed it here, so the request never reached the router. A worker
    /// was spawned, the loader failed with "No shard files found for model X in
    /// <dir>", and the caller got a **404 — while two connected peers held
    /// every shard and the dashboard reported the model `ready`**. The periodic
    /// disk reconcile had already fired and stopped announcing those shards to
    /// the swarm; it simply did not clear this cache.
    ///
    /// `make_shared_state` uses a temp data dir with no shard files written, so
    /// a complete-looking entry here is exactly the stale case.
    #[test]
    fn a_complete_entry_with_no_shard_files_does_not_win_the_fast_path() {
        let state = make_shared_state();
        let mid = ModelId("llama-3.2-3b-instruct-q4-k-m".into());
        // Spans the whole model and is marked complete — the shape that took
        // the fast path and produced the 404.
        state
            .split_models
            .insert((mid.clone(), 0, 28), entry(0, 28, true));

        assert!(
            !state.has_complete_split_model(&mid),
            "an entry with no shard files on disk must not claim the fast path"
        );
        // ...and the stale entry is dropped rather than left to mislead the
        // next caller.
        assert!(
            !state.split_models.iter().any(|e| e.key().0 == mid),
            "the stale entry should have been evicted, not merely ignored"
        );
    }

    /// A node whose data directory is its OWN, not the one every
    /// `Config::default()` in this test process shares
    /// (`config::node::default_data_dir`): "does this node hold the model
    /// whole" looks for shard files there.
    fn node_with(
        configure: impl FnOnce(&mut crate::config::Config),
    ) -> std::sync::Arc<crate::daemon::SharedState> {
        let temp = tempfile::tempdir().unwrap().keep();
        let db = crate::storage::db::Database::open(&temp).unwrap();
        let mut config = crate::config::Config::default();
        config.node.data_dir = temp;
        configure(&mut config);
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = crate::daemon::SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    /// How much of a model a test node holds.
    #[derive(Clone, Copy)]
    enum Held {
        /// Every shard here, registered whole — what the scan makes of it.
        WholeHere,
        /// Shard 0 here, the rest on a peer: servable by the swarm, not here.
        PartHere,
        /// Every shard on a peer.
        OnPeer,
        /// Shard 0 on a peer, the rest on nobody.
        Unservable,
    }

    fn add_model(state: &crate::daemon::SharedState, id: &str, held: Held) -> ModelId {
        let mid = ModelId(id.into());
        let shards = 2u32;
        state
            .model_registry
            .register_manifest(crate::types::ModelManifest {
                id: mid.clone(),
                name: id.into(),
                architecture: crate::types::ModelArchitecture::Llama,
                num_layers: 4,
                num_params_billions: 1.0,
                quantization: crate::types::Quantization::Q4KM,
                total_size_bytes: 2,
                shard_count: shards,
                shards: (0..shards)
                    .map(|i| swarmllm_types::ShardInfo {
                        index: i,
                        layer_range: (i * 2, i * 2 + 2),
                        size_bytes: 1,
                        hash: [0u8; 32],
                        tensors: Vec::new(),
                    })
                    .collect(),
                tokenizer_hash: [0u8; 32],
                manifest_hash: [0u8; 32],
                publisher: crate::types::NodeId([0u8; 32]),
                publish_date: chrono::Utc::now(),
                license: "MIT".into(),
                mmproj: None,
                context_length: None,
            });
        let me = state.identity.node_id().clone();
        let peer = crate::types::NodeId([9u8; 32]);
        for i in 0..shards {
            let sid = crate::types::ShardId {
                model_id: mid.clone(),
                index: i,
            };
            let holder = match (held, i) {
                (Held::WholeHere, _) | (Held::PartHere, 0) => Some(me.clone()),
                (Held::PartHere, _) | (Held::OnPeer, _) | (Held::Unservable, 0) => {
                    Some(peer.clone())
                }
                (Held::Unservable, _) => None,
            };
            if let Some(h) = holder {
                if h == me {
                    let path = state.shard_store().shard_path(&mid, i);
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    std::fs::write(&path, b"x").unwrap();
                }
                state.model_registry.record_shard_holder(sid, h);
            }
        }
        match held {
            Held::WholeHere => {
                state
                    .split_models
                    .insert((mid.clone(), 0, 4), entry(0, 4, true));
            }
            Held::PartHere => {
                state
                    .split_models
                    .insert((mid.clone(), 0, 2), entry(0, 2, false));
            }
            Held::OnPeer | Held::Unservable => {}
        }
        mid
    }

    /// #120: the shard scan writes `loaded_model_info` for every model it
    /// registers, a partial holding included, so the model it touched LAST was
    /// what `auto` meant — seen live as a 14B held 14 of 48 layers, handed to two
    /// peers that could not load it. A model this node can run whole wins.
    #[tokio::test]
    async fn auto_prefers_a_model_held_whole_over_the_last_one_the_scan_touched() {
        let state = node_with(|_| {});
        add_model(&state, "zzz-whole", Held::WholeHere);
        let partial = add_model(&state, "aaa-partial", Held::PartHere);
        *state.loaded_model_info.write().await = Some(crate::daemon::LoadedModelInfo {
            name: partial.0.clone(),
            size_bytes: 2,
            eos_tokens: vec![],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        });
        assert_eq!(auto_model_for(&state).await.as_deref(), Some("zzz-whole"));
    }

    /// A conversation held on `auto` stays on one model: a model that becomes
    /// available later — sorting first — does not take the next turn.
    #[tokio::test]
    async fn auto_keeps_answering_with_the_model_it_chose() {
        let state = node_with(|_| {});
        add_model(&state, "bbb", Held::WholeHere);
        add_model(&state, "ccc", Held::WholeHere);
        assert_eq!(auto_model_for(&state).await.as_deref(), Some("bbb"));
        add_model(&state, "aaa", Held::WholeHere);
        assert_eq!(
            auto_model_for(&state).await.as_deref(),
            Some("bbb"),
            "the next turn must not switch model"
        );
    }

    /// The owner's `inference.default_model` decides, when it names a known model.
    #[tokio::test]
    async fn auto_is_the_owners_default_model_when_one_is_set() {
        let state = node_with(|c| c.inference.default_model = "ccc".into());
        add_model(&state, "bbb", Held::WholeHere);
        add_model(&state, "ccc", Held::OnPeer);
        assert_eq!(auto_model_for(&state).await.as_deref(), Some("ccc"));
    }

    /// With nothing held whole here, a model the swarm can serve beats one it
    /// cannot, whatever the order.
    #[tokio::test]
    async fn auto_falls_back_to_a_model_the_swarm_can_serve() {
        let state = node_with(|_| {});
        add_model(&state, "aaa-unservable", Held::Unservable);
        add_model(&state, "bbb-on-peer", Held::OnPeer);
        assert_eq!(auto_model_for(&state).await.as_deref(), Some("bbb-on-peer"));
    }

    /// `auto` is a model nobody named, so never one with its safety training
    /// removed that the owner did not choose — a fresh node held one whole and
    /// had it pre-selected (field report 2026-10-15) — and the owner choosing
    /// it makes it a model like any other.
    #[tokio::test]
    async fn auto_never_names_an_uncensored_model_the_owner_did_not_choose() {
        let state = node_with(|_| {});
        let marked = add_model(&state, "aaa-coder-abliterated-q4", Held::WholeHere);
        add_model(&state, "bbb-coder-q4", Held::OnPeer);
        assert_eq!(
            auto_model_for(&state).await.as_deref(),
            Some("bbb-coder-q4")
        );

        *state.models.auto_model.lock() = None;
        state
            .models
            .model_trust
            .insert(marked, crate::types::ModelTrustInfo::new_pinned());
        assert_eq!(
            auto_model_for(&state).await.as_deref(),
            Some("aaa-coder-abliterated-q4"),
            "the owner pinned it: held whole, it is the answer like any other"
        );
    }

    /// A node whose pool holds one member, `0xC4`, hosting `model` and
    /// reachable at a public address — the peer the HTTP forward would pick.
    async fn pool_member_hosting(
        config: crate::config::Config,
        model: &str,
    ) -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::pool::types::{PoolMembership, PoolState};
        use crate::types::{NodeCapability, NodeId, PeerInfo, ShardId};
        let temp = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = crate::daemon::SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        let peer = NodeId([0xC4; 32]);
        *state.credits.pool_state.write().await = Some(PoolState {
            pool_id: NodeId([0xC0; 32]),
            name: "home".into(),
            created_at: chrono::Utc::now(),
            owner_signature: vec![],
            members: vec![PoolMembership {
                node_id: peer.clone(),
                credits_contributed: 0,
                joined_at: chrono::Utc::now(),
                acceptance_signature: vec![],
                invitation_id: uuid::Uuid::new_v4(),
                invitation_expires_at: chrono::Utc::now(),
                device_name: None,
                last_seen: None,
                online: false,
                device_stats: None,
                contribution_level: 100,
            }],
            shard_pins: Default::default(),
            total_lifetime_credits: 0,
            member_credit_split_pct: 0,
            generation: 0,
        });
        state.peer_registry.insert(
            peer.clone(),
            PeerInfo {
                node_id: peer.clone(),
                addresses: vec!["/ip4/203.0.113.7/udp/8800/quic-v1".into()],
                capability: Some(NodeCapability {
                    coord: None,
                    node_id: peer.clone(),
                    gpu: None,
                    cpu: None,
                    ram_total_mb: 0,
                    ram_available_mb: 0,
                    ram_model_budget_mb: None,
                    disk_available_mb: 0,
                    bandwidth_mbps: 0.0,
                    hosted_shards: vec![ShardId {
                        model_id: ModelId(model.into()),
                        index: 0,
                    }],
                    max_contribution: crate::types::ContributionLevel::Moderate,
                    uptime_seconds: 0,
                    version: String::new(),
                    region: None,
                    est_tokens_per_sec_7b: 0.0,
                    os: None,
                    observed_latencies: vec![],
                    relay_capable: false,
                    protocol_version: 0,
                    features: 0,
                    relay_reservations: vec![],
                    anchor_mode: false,
                    can_serve_inference: true,
                    resident_layers: Vec::new(),
                    context_ceiling_tokens: None,
                    model_load_ms_per_gib: None,
                    model_memory_ceiling_mb: None,
                    machine_room_beside_card_mb: None,
                    processor_tokens_per_sec_7b: None,
                }),
                last_seen: chrono::Utc::now(),
                latency_ms: Some(10),
                trust_score: 0.9,
                peer_id_bytes: None,
                ack_srtt_ms: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
                goodput_bytes_per_sec: None,
                goodput_samples: 0,
            },
        );
        state
    }

    /// #316's review: the HTTP forward to a pool member hands the peer the
    /// request WHOLE, and the peer renders the prompt from its text — so it
    /// asks the privacy question the libp2p hand-off asks, which it never
    /// had. With both settings off the member is chosen; under either it is
    /// refused in that setting's own words, never forwarded.
    #[tokio::test]
    async fn the_http_forward_to_a_pool_member_asks_both_privacy_settings() {
        use crate::error::SwarmError;
        let model = "m";
        let pick = |embedding: bool, start_here: bool| async move {
            let mut config = crate::config::Config::default();
            config.inference.local_embedding_privacy = embedding;
            config.inference.encrypted_pipeline = start_here;
            let state = pool_member_hosting(config, model).await;
            find_peer_with_model(&state, model, crate::types::KeepAsked::Unasked)
        };

        let open = pick(false, false).await;
        assert!(
            matches!(open, Ok(Some(_))),
            "control: the member is chosen with both settings off"
        );
        assert!(matches!(
            pick(true, false).await,
            Err(SwarmError::EmbeddingPrivacyUnavailable { .. })
        ));
        assert!(matches!(
            pick(false, true).await,
            Err(SwarmError::PromptPrivacyUnavailable { .. })
        ));
    }

    /// A node learns the swarm's manifests 12-31 s after it starts, and one
    /// that knew a model of its own answered 404 "not found" for every other
    /// until then — even one a connected peer said it held (a rig's requester,
    /// 2026-10-20). A model a CONNECTED peer advertises is early, not a typo —
    /// while that peer's catch-up can still be on its way; one only a departed
    /// peer advertised, or one a peer connected minutes ago lists and never
    /// sent, is still refused.
    #[tokio::test]
    async fn a_model_a_connected_peer_holds_is_not_called_unknown_before_its_manifest() {
        let state = pool_member_hosting(crate::config::Config::default(), "theirs").await;
        add_model(&state, "ours-q4", Held::WholeHere);
        let asked = ModelId("theirs".into());
        let peer = crate::types::NodeId([0xC4; 32]);
        assert!(
            matches!(
                state.reject_if_unknown_model(&asked),
                Some(crate::error::SwarmError::ModelNotAvailable(_))
            ),
            "control: advertised by a peer that is not connected, it is refused"
        );
        state.models.manifest_sync.note_new_connection(&peer);
        state.connected_node_ids.insert(peer);
        assert!(
            state.reject_if_unknown_model(&asked).is_none(),
            "a connected peer holds it: wait for its manifest"
        );
        assert!(
            state
                .reject_if_unknown_model(&ModelId("typo".into()))
                .is_some(),
            "a name no one advertises is still refused at once"
        );
        let later =
            std::time::Instant::now() + crate::daemon::state::manifest_sync::CATCH_UP_WITHIN;
        assert!(
            state.reject_if_unknown_model_at(&asked, later).is_some(),
            "past its catch-up, a model it lists and never sent is not waited for"
        );
    }

    /// A request that finds every cold-start place taken is answered as the
    /// wait would answer it: a node holding models is never told to download
    /// one (#324).
    #[tokio::test]
    async fn a_full_cold_start_wait_answers_as_the_wait_would() {
        let state = pool_member_hosting(crate::config::Config::default(), "theirs").await;
        let asked = ModelId("theirs".into());
        let full = tokio::sync::Semaphore::new(0);
        assert!(
            matches!(
                super::cold_start_slot(&state, &asked, &full),
                Err(crate::error::SwarmError::NoModelLoaded)
            ),
            "a node that knows no model is told to get one"
        );
        add_model(&state, "ours-q4", Held::WholeHere);
        let peer = crate::types::NodeId([0xC4; 32]);
        state.models.manifest_sync.note_new_connection(&peer);
        state.connected_node_ids.insert(peer);
        assert!(
            matches!(
                super::cold_start_slot(&state, &asked, &full),
                Err(crate::error::SwarmError::ModelNotAvailable(_))
            ),
            "a node holding models is told the name is unavailable, not to download one"
        );
        let free = tokio::sync::Semaphore::new(1);
        assert!(super::cold_start_slot(&state, &asked, &free).is_ok());
    }

    /// Once every way to serve a request has been tried, a node that knows
    /// models answers a name it holds no details for with 404 and the list of
    /// what can answer — one a connected peer advertises too, whose wait is
    /// over. "No model is loaded yet. Go to the dashboard…" is only for a node
    /// that knows none (pre-release review, 2026-10-20: both surfaces said it
    /// to a node holding models while a peer's catch-up was awaited).
    #[tokio::test]
    async fn the_last_answer_tells_a_node_holding_models_the_name_is_unavailable() {
        let state = pool_member_hosting(crate::config::Config::default(), "theirs").await;
        let asked = ModelId("theirs".into());
        assert!(
            matches!(
                state.no_route_error(&asked),
                crate::error::SwarmError::NoModelLoaded
            ),
            "a node that knows no model is told to get one"
        );
        add_model(&state, "ours-q4", Held::WholeHere);
        let peer = crate::types::NodeId([0xC4; 32]);
        state.models.manifest_sync.note_new_connection(&peer);
        state.connected_node_ids.insert(peer);
        assert!(
            state.reject_if_unknown_model(&asked).is_none(),
            "control: a connected peer's catch-up is waited for"
        );
        assert!(
            matches!(
                state.no_route_error(&asked),
                crate::error::SwarmError::ModelNotAvailable(_)
            ),
            "after the wait, a node holding models is told the name is unavailable"
        );
    }
}
