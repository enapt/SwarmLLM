//! Which upload of each model this node uses — the state half of
//! `model::canonical`. The pure rule lives there; the driver that checks
//! uploads against HuggingFace and replaces this node's wrong parts lives in
//! `model::auto_manage::canonical`. These are the only writers of
//! `origin_claims`, `canonical_builds` and `hf_sources`.

use super::{HfSource, SharedState};
use crate::model::canonical::{self, CanonicalBuild, Holding};
use crate::types::ModelId;

/// How long an upload HuggingFace would not serve is left alone before it is
/// asked about again. Long for "not there / private", short for anything
/// that may be a passing network fault.
pub const ORIGIN_REFUSED_PERMANENT_SECS: u64 = 24 * 60 * 60;
pub const ORIGIN_REFUSED_TRANSIENT_SECS: u64 = 30 * 60;

/// [`SharedState::judge_peer_manifest`]'s answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerManifest {
    /// Register it (`register_manifest` still adjudicates against origin
    /// knowledge).
    Adopt,
    /// It describes another upload than the one the swarm uses.
    AnotherUpload,
    /// It describes another build than the manifest a download running here
    /// is fetching against.
    ReplacesTheOneBeingDownloaded,
}

fn origin_key(source: &HfSource) -> String {
    format!("{}/{}", source.repo_id.to_lowercase(), source.filename)
}

impl SharedState {
    /// Remember that `source` is an upload of `model_id` — from a peer's
    /// gossip, a dashboard download, a search. Returns whether it was new.
    ///
    /// A model with no `hf_sources` entry yet gets this one provisionally, as
    /// it always did, so discovery keeps working before anything is verified;
    /// nothing may ACQUIRE from a provisional source
    /// (`canonical_allows_acquisition`), and the driver replaces it with the
    /// canonical upload once one is verified.
    pub fn note_origin_claim(&self, model_id: &ModelId, source: HfSource) -> bool {
        if !canonical::origin_names_model(&source, model_id) {
            return false;
        }
        let changed = {
            let mut claims = self
                .models
                .origin_claims
                .entry(model_id.clone())
                .or_default();
            canonical::note_claim(&mut claims, model_id, source.clone())
        };
        if !self.models.hf_sources.contains_key(model_id) {
            self.write_hf_source(model_id, source, false);
        }
        changed
    }

    /// Make `build` the upload this node uses for `model_id`, and point
    /// `hf_sources` at it. The ONLY writer of `canonical_builds`.
    pub fn adopt_canonical_build(&self, model_id: &ModelId, build: CanonicalBuild) {
        let mut source = build.source.clone();
        if source.mmproj_filename.is_none() {
            source.mmproj_filename = self
                .models
                .hf_sources
                .get(model_id)
                .filter(|s| canonical::same_origin(s, &build.source))
                .and_then(|s| s.mmproj_filename.clone());
        }
        if let Err(e) = self
            .db
            .put_json(canonical::CANONICAL_BUILDS_TREE, &model_id.0, &build)
        {
            tracing::warn!(model = %model_id, error = %e, "Could not persist the canonical upload — it is re-checked after a restart");
        }
        self.models
            .canonical_builds
            .insert(model_id.clone(), build.clone());
        self.write_hf_source(model_id, source, true);
        self.note_origin_claim(model_id, build.source);
    }

    /// Record what this node's copy of `model_id` is against the canonical
    /// upload (`None`: not judged — the choice moved, or the model was
    /// deleted). Returns whether it changed.
    ///
    /// The ONE writer of `canonical_holding`, and with it of whether the copy
    /// is offered to the swarm: a copy waiting to have its parts replaced is
    /// withheld (`ModelRegistry::set_model_withheld`) — not announced, not
    /// gossiped, not served — until they are. It used to keep serving "until the
    /// new parts are in", and a coordinator that took its hashes (the first a
    /// node holding none of the model hears) routed layers to bytes that were
    /// not the upload; with a header from the upload beside them, that is the
    /// garbage #156 answered. The owner's own requests still use it.
    pub fn note_canonical_holding(&self, model_id: &ModelId, holding: Option<Holding>) -> bool {
        // And of what this node tells peers it CHECKED: only a copy the heal
        // has compared with the upload on HuggingFace this run, never one
        // settled by agreeing with peers — an attestation that came from other
        // attestations would let one checked holder count as several.
        self.model_registry
            .set_model_origin_checked(model_id, holding == Some(Holding::Canonical));
        let withhold = holding.as_ref().is_some_and(Holding::is_another_upload);
        if self.model_registry.set_model_withheld(model_id, withhold) {
            if withhold {
                tracing::info!(model = %model_id, "DIAG: this node's copy is not the swarm's upload — no longer offering it to peers");
            } else {
                tracing::info!(model = %model_id, "DIAG: offering this node's copy of the model to peers again");
            }
        }
        let changed = match holding {
            Some(h) => self
                .models
                .canonical_holding
                .insert(model_id.clone(), h.clone())
                .is_none_or(|was| was != h),
            None => self.models.canonical_holding.remove(model_id).is_some(),
        };
        if changed {
            self.signal_dashboard(super::DashboardSignal::ModelsChanged);
        }
        changed
    }

    /// The upload the swarm uses for `model_id`, when this node has verified one.
    pub fn canonical_build(&self, model_id: &ModelId) -> Option<CanonicalBuild> {
        self.models
            .canonical_builds
            .get(model_id)
            .map(|b| b.value().clone())
    }

    /// Is `manifest` a description of some OTHER upload than the one this
    /// node knows the swarm uses? `false` when no canonical upload is known.
    pub fn manifest_is_another_upload(&self, manifest: &crate::types::ModelManifest) -> bool {
        canonical::canonical_uploads_enabled()
            && self
                .models
                .canonical_builds
                .get(&manifest.id)
                .is_some_and(|b| !b.describes(manifest))
    }

    /// What this node does with a manifest a PEER sent — the one decision for
    /// the dispatcher, the only door a peer's manifest comes in by.
    pub fn judge_peer_manifest(&self, manifest: &crate::types::ModelManifest) -> PeerManifest {
        if self.manifest_is_another_upload(manifest) {
            return PeerManifest::AnotherUpload;
        }
        // Before the canonical upload is known there is nothing to compare a
        // peer's manifest with, and adoption is last-writer-wins — except
        // while this node is downloading the model: its parts are being
        // fetched against the manifest registered now, and replacing that
        // with another build's leaves the parts described by one upload and
        // the registry by another (FUTURE_WORK #158, the root of gotcha #776).
        // Origin knowledge settles the same question inside `register_manifest`
        // once a part has landed; this closes the window before it does.
        let ours_is_another_build =
            self.model_registry
                .get_manifest(&manifest.id)
                .is_some_and(|ours| {
                    crate::model::registry::ModelRegistry::describes_a_different_build(
                        &ours, manifest,
                    )
                });
        if ours_is_another_build && self.models.model_download_under_way(&manifest.id) {
            return PeerManifest::ReplacesTheOneBeingDownloaded;
        }
        PeerManifest::Adopt
    }

    /// May this node fetch parts of `model_id` right now?
    ///
    /// The gate that stops a node acquiring parts of an upload the swarm does
    /// not use. Yes when the model's canonical upload is known AND the manifest
    /// this node fetches against describes it AND this node is not waiting to
    /// delete parts of another upload (`Holding::Replacing`);
    /// no while an upload still waits to be checked — fetching then would take
    /// whichever upload this node happened to hear of first, which is how the
    /// swarm split in the first place. A model with no HuggingFace origin at
    /// all, or a node in offline mode (which never reaches HuggingFace), keeps
    /// the old behaviour.
    pub fn canonical_allows_acquisition(&self, model_id: &ModelId) -> bool {
        if !canonical::canonical_uploads_enabled()
            || self
                .credits
                .offline_mode
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            return true;
        }
        if let Some(build) = self.canonical_build(model_id) {
            let replacing = matches!(
                self.models
                    .canonical_holding
                    .get(model_id)
                    .map(|h| h.value().clone()),
                Some(Holding::Replacing { .. } | Holding::OwnFile)
            );
            return !replacing
                && self
                    .model_registry
                    .get_manifest(model_id)
                    .is_some_and(|m| build.describes(&m));
        }
        !self.origin_pending_validation(model_id)
    }

    /// Is there an upload of `model_id` this node has heard of and not yet
    /// been refused by HuggingFace for? Then the canonical choice is still to
    /// be made.
    pub fn origin_pending_validation(&self, model_id: &ModelId) -> bool {
        let claims = self
            .models
            .origin_claims
            .get(model_id)
            .map(|c| c.value().clone())
            .unwrap_or_default();
        let provisional = self.models.hf_sources.get(model_id).map(|s| s.clone());
        claims.iter().chain(provisional.iter()).any(|c| {
            canonical::origin_names_model(c, model_id) && !self.origin_refused(model_id, c)
        })
    }

    /// Every upload of `model_id` this node knows of, best-first: the claims
    /// plus whatever `hf_sources` holds.
    pub fn origin_candidates(&self, model_id: &ModelId) -> Vec<HfSource> {
        let mut all = self
            .models
            .origin_claims
            .get(model_id)
            .map(|c| c.value().clone())
            .unwrap_or_default();
        if let Some(current) = self.models.hf_sources.get(model_id).map(|s| s.clone()) {
            canonical::note_claim(&mut all, model_id, current);
        }
        all
    }

    pub fn origin_refused(&self, model_id: &ModelId, source: &HfSource) -> bool {
        self.models
            .origin_refusals
            .get(&(model_id.clone(), origin_key(source)))
            .is_some_and(|until| std::time::Instant::now() < *until)
    }

    pub fn note_origin_refused(&self, model_id: &ModelId, source: &HfSource, for_secs: u64) {
        self.models.origin_refusals.insert(
            (model_id.clone(), origin_key(source)),
            std::time::Instant::now() + std::time::Duration::from_secs(for_secs),
        );
    }

    /// Forget every origin-derived hash of `model_id`, in memory and on disk.
    /// See `ModelRegistry::forget_origin_verified_for_model`.
    pub fn forget_origin_verified_for_model(&self, model_id: &ModelId) {
        self.forget_origin_verified_on_disk(
            model_id,
            self.model_registry
                .forget_origin_verified_for_model(model_id),
        );
    }

    /// The same for some parts only. See
    /// `ModelRegistry::forget_origin_verified_for_parts`.
    pub fn forget_origin_verified_for_parts(&self, model_id: &ModelId, parts: &[u32]) {
        self.forget_origin_verified_on_disk(
            model_id,
            self.model_registry
                .forget_origin_verified_for_parts(model_id, parts),
        );
    }

    fn forget_origin_verified_on_disk(&self, model_id: &ModelId, gone: Vec<crate::types::ShardId>) {
        for shard in gone {
            if let Ok(key) = serde_json::to_string(&shard) {
                if let Err(e) = self
                    .db
                    .remove(crate::model::registry::ORIGIN_VERIFIED_TREE, &key)
                {
                    tracing::warn!(model = %model_id, shard = shard.index, error = %e, "Could not forget an origin hash on disk");
                }
            }
        }
    }

    /// Fetch `model_id`'s GGUF header from its HuggingFace source into the
    /// model's directory — and only from a source that IS the upload this
    /// node's manifest describes. The ONE way a header is fetched for a model
    /// a node already knows (startup, routing, prompt rendering, chat
    /// templates). Each site used to fetch from whatever `hf_sources` held, so
    /// a node whose parts came from peers could put another upload's header
    /// beside them — the file that says where every tensor in them is.
    pub async fn fetch_model_header(
        &self,
        model_id: &ModelId,
    ) -> Result<(std::path::PathBuf, HfSource), String> {
        let source = self
            .models
            .hf_sources
            .get(model_id)
            .map(|s| s.value().clone())
            .ok_or_else(|| "no HuggingFace source is known for this model".to_string())?;
        let info = crate::model::huggingface::probe_gguf_file(
            &source.repo_id,
            &source.filename,
            canonical::canonical_shard_size_bytes(),
        )
        .await?;
        if let Some(manifest) = self.model_registry.get_manifest(model_id) {
            if manifest.total_size_bytes != info.total_size {
                return Err(format!(
                    "{}/{} is another upload of this model than the one this node fetches \
                     parts of ({} bytes, not {}) — its header would not describe them",
                    source.repo_id, source.filename, info.total_size, manifest.total_size_bytes
                ));
            }
        }
        let path = crate::model::huggingface::download_gguf_header(
            &source.repo_id,
            &source.filename,
            &self.model_dir(&model_id.0),
            info.header_size,
        )
        .await?;
        Ok((path, source))
    }

    /// `hf_sources` in memory, then in the DB and — once canonical —
    /// `hf_source.json` beside the model's parts, which startup reads back.
    ///
    /// The disk writes run OFF the calling task when there is a runtime: a
    /// gossip claim arrives on the message dispatcher, the one consumer of
    /// `network_out`, and redb serialises writers — a write waiting there
    /// stalls every message behind it (gotcha #74, FUTURE_WORK #90). Every
    /// reader consults the in-memory map, which is updated first.
    fn write_hf_source(&self, model_id: &ModelId, source: HfSource, canonical: bool) {
        self.models
            .hf_sources
            .insert(model_id.clone(), source.clone());
        let db = self.db.clone();
        let key = model_id.0.clone();
        let file = Some(self.model_dir(&model_id.0))
            .filter(|dir| canonical && dir.is_dir())
            .map(|dir| dir.join(crate::model::shard::HF_SOURCE_FILENAME));
        let persist = move || {
            if let Err(e) = db.put_json("hf_sources", &key, &source) {
                tracing::warn!(model = %key, error = %e, "Could not persist a model's HuggingFace source");
            }
            if let Some(path) = file {
                if let Ok(json) = serde_json::to_string_pretty(&source) {
                    if let Err(e) = std::fs::write(&path, json) {
                        tracing::warn!(path = %path.display(), error = %e, "Could not write hf_source.json");
                    }
                }
            }
        };
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                rt.spawn_blocking(persist);
            }
            Err(_) => persist(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose data directory is its own — `write_hf_source` writes
    /// beside a model's parts, and must never reach the developer's real node.
    fn test_state() -> std::sync::Arc<SharedState> {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.keep();
        let mut config = crate::config::Config::default();
        config.node.data_dir = dir.clone();
        let db = crate::storage::db::Database::open(&dir).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    fn src(repo: &str, file: &str) -> HfSource {
        HfSource {
            repo_id: repo.into(),
            filename: file.into(),
            mmproj_filename: None,
        }
    }

    fn build_for(source: HfSource, total: u64) -> CanonicalBuild {
        CanonicalBuild {
            source,
            total_size: total,
            header_size: 10,
            shard_sizes: vec![total - 10],
            shard_layers: vec![(0, 4)],
            shard_first_tensor: vec![(10, total - 10)],
            header_hash: [3; 32],
            resolved_at_ms: 0,
        }
    }

    /// Until an upload is verified a node must not fetch parts — it would
    /// take whichever upload it heard of first. Once one is verified, it may,
    /// but only against a manifest of THAT upload.
    #[test]
    fn nothing_is_fetched_until_the_canonical_upload_is_known_and_registered() {
        let state = test_state();
        let id = ModelId("x-q4-k-m".into());
        let a = src("stranger/X-GGUF", "X-Q4_K_M.gguf");
        assert!(
            state.canonical_allows_acquisition(&id),
            "no origin known at all: nothing to choose between"
        );
        state.note_origin_claim(&id, a.clone());
        assert!(
            !state.canonical_allows_acquisition(&id),
            "an unchecked upload waits"
        );

        state.adopt_canonical_build(&id, build_for(a.clone(), 1_000));
        assert!(
            !state.canonical_allows_acquisition(&id),
            "no manifest of the canonical upload registered yet"
        );
        assert!(crate::model::canonical::same_origin(
            &state.models.hf_sources.get(&id).unwrap(),
            &a
        ));
    }

    /// A peer's manifest of another upload is refused only once this node
    /// knows which upload the swarm uses — before that there is nothing to
    /// compare it with.
    #[test]
    fn a_manifest_of_another_upload_is_recognised_once_the_choice_is_known() {
        let state = test_state();
        let id = ModelId("x-q4-k-m".into());
        let shaped = |total: u64| {
            crate::model::manifest::build_manifest_from_gguf(
                crate::model::manifest::ManifestFromGguf {
                    id: id.clone(),
                    name: "x".into(),
                    architecture: crate::types::ModelArchitecture::Llama,
                    num_layers: 4,
                    total_size_bytes: total,
                    shard_count: 1,
                    shards: vec![crate::types::ShardInfo {
                        index: 0,
                        layer_range: (0, 4),
                        size_bytes: total - 10,
                        hash: [0; 32],
                        tensors: Vec::new(),
                    }],
                    publisher: crate::types::NodeId([0; 32]),
                    context_length: None,
                },
            )
        };
        let theirs = shaped(1_234);
        assert!(!state.manifest_is_another_upload(&theirs));
        state.adopt_canonical_build(
            &id,
            build_for(src("bartowski/X-GGUF", "X-Q4_K_M.gguf"), 1_000),
        );
        assert!(state.manifest_is_another_upload(&theirs));
        assert!(!state.manifest_is_another_upload(&shaped(1_000)));
    }

    /// FUTURE_WORK #158, the root of gotcha #776: before the canonical upload
    /// is known, a peer's manifest of another build replaced the one a running
    /// download was fetching against. While a download is under way — a part
    /// being written, or the gap before or between parts — it may not; once
    /// nothing is downloading, adoption is as before. A same-build manifest
    /// (hash updates) is never held back.
    #[test]
    fn a_peers_manifest_of_another_build_never_replaces_the_one_being_downloaded() {
        let state = test_state();
        let id = ModelId("x-q8-0".into());
        let shaped = |total: u64| {
            crate::model::manifest::build_manifest_from_gguf(
                crate::model::manifest::ManifestFromGguf {
                    id: id.clone(),
                    name: "x".into(),
                    architecture: crate::types::ModelArchitecture::Llama,
                    num_layers: 4,
                    total_size_bytes: total,
                    shard_count: 1,
                    shards: vec![crate::types::ShardInfo {
                        index: 0,
                        layer_range: (0, 4),
                        size_bytes: total - 10,
                        hash: [0; 32],
                        tensors: Vec::new(),
                    }],
                    publisher: crate::types::NodeId([0; 32]),
                    context_length: None,
                },
            )
        };
        // Ours: the upload a dashboard download is fetching. Theirs: another
        // upload, 3,808 bytes apart — the .220 gate's pair.
        state
            .model_registry
            .register_manifest(shaped(1_321_079_200));
        let theirs = shaped(1_321_083_008);
        assert_eq!(
            state.judge_peer_manifest(&theirs),
            PeerManifest::Adopt,
            "nothing downloading: last writer wins, as before"
        );

        let claim = state
            .models
            .claim_shard_download(&crate::types::ShardId {
                model_id: id.clone(),
                index: 0,
            })
            .unwrap();
        assert_eq!(
            state.judge_peer_manifest(&theirs),
            PeerManifest::ReplacesTheOneBeingDownloaded
        );
        assert_eq!(
            state.judge_peer_manifest(&shaped(1_321_079_200)),
            PeerManifest::Adopt,
            "the same build is not another build"
        );
        drop(claim);

        // The gap before the first part (or between parts): no claim, but a
        // download that began and has not finished.
        state.models.acquisition_progress.insert(
            id.clone(),
            crate::model::acquisition::AcquisitionStatus::new_downloading(
                id.clone(),
                1,
                1_321_079_200,
                "huggingface",
                "dashboard",
                "test",
            ),
        );
        assert_eq!(
            state.judge_peer_manifest(&theirs),
            PeerManifest::ReplacesTheOneBeingDownloaded
        );

        // And once the swarm's upload is known, another upload is that.
        state.adopt_canonical_build(
            &id,
            build_for(src("bartowski/X-GGUF", "X-Q8_0.gguf"), 1_321_079_200),
        );
        assert_eq!(
            state.judge_peer_manifest(&theirs),
            PeerManifest::AnotherUpload
        );
    }

    /// A refused upload is not "pending": a node whose HuggingFace access
    /// fails for every candidate falls back to fetching as before rather than
    /// never fetching at all.
    #[test]
    fn a_node_huggingface_refuses_is_not_held_waiting_for_ever() {
        let state = test_state();
        let id = ModelId("x-q4-k-m".into());
        let a = src("stranger/X-GGUF", "X-Q4_K_M.gguf");
        state.note_origin_claim(&id, a.clone());
        state.note_origin_refused(&id, &a, ORIGIN_REFUSED_TRANSIENT_SECS);
        assert!(state.canonical_allows_acquisition(&id));
    }

    /// The first claim becomes the provisional source; a later, better claim
    /// does NOT overwrite it — only a verified canonical upload does.
    #[test]
    fn only_a_verified_upload_replaces_the_source_a_node_fetches_from() {
        let state = test_state();
        let id = ModelId("x-q4-k-m".into());
        let first = src("stranger/X-GGUF", "X-Q4_K_M.gguf");
        let better = src("bartowski/X-GGUF", "X-Q4_K_M.gguf");
        state.note_origin_claim(&id, first.clone());
        state.note_origin_claim(&id, better.clone());
        assert!(crate::model::canonical::same_origin(
            &state.models.hf_sources.get(&id).unwrap(),
            &first
        ));
        assert!(crate::model::canonical::same_origin(
            &state.origin_candidates(&id)[0],
            &better
        ));
        state.adopt_canonical_build(&id, build_for(better.clone(), 1_000));
        assert!(crate::model::canonical::same_origin(
            &state.models.hf_sources.get(&id).unwrap(),
            &better
        ));
    }
}
