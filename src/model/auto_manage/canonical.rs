//! Keeping this node on the swarm's upload of every model it holds or may
//! fetch — and healing it onto that upload when it holds another.
//!
//! `model::canonical` is the rule (which upload of a model the swarm uses);
//! this is the part with side effects, run as one background task:
//!
//! 1. **Resolve.** For each model this node holds or may acquire, check the
//!    best upload anyone has claimed against HuggingFace — anonymously, so
//!    every node gets the same answer — and adopt it
//!    (`SharedState::adopt_canonical_build`). From then on `hf_sources` names
//!    it and every fetch is checked against it.
//! 2. **Check what is held.** A copy of the same SHAPE is checked byte for
//!    byte at the start of every part it holds (64 KB per part, read from
//!    HuggingFace) — cheap, and decisive between uploads, whose tensor bytes
//!    differ everywhere. The header beside the parts is checked by hash:
//!    a node that took its parts from peers took its header from whatever
//!    source it heard of first, which could be another upload's.
//! 3. **Heal.** A node holding parts of another upload fetches the canonical
//!    upload's parts covering the same layers into a staging directory
//!    (`<data_dir>/canonical/<model>`), from HuggingFace by byte range — never
//!    a whole GGUF — keeps serving its old parts meanwhile, and swaps only
//!    when every new part is on disk and the model is idle. The old parts are
//!    deleted in the swap. Nobody has to delete or re-download anything.
//!
//! One switch fetches at a time, so a node with several models to move does
//! not saturate its connection or its disk — and a switch that CANNOT go ahead
//! (no room on disk, HuggingFace refusing) holds nobody up ([`SwitchQueue`]).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::{HfSource, SharedState};
use crate::model::canonical::{self, CanonicalBuild, Holding};
use crate::model::manifest::ModelManifestExt;
use crate::types::{ModelId, NetworkCommand, ShardId};

const FIRST_PASS_AFTER: Duration = Duration::from_secs(30);
const PASS_EVERY: Duration = Duration::from_secs(120);
/// Uploads checked against HuggingFace per pass — each is one 16 MB probe.
const RESOLVES_PER_PASS: usize = 4;
/// How much of each held part is compared against the upload.
const SPOT_CHECK_BYTES: u64 = 64 * 1024;
/// Records which upload a staging directory holds parts of, so parts staged
/// for one upload are never swapped in for another.
const STAGED_UPLOAD_FILENAME: &str = "upload.json";

/// Where parts of the canonical upload wait until they replace a node's old
/// ones. Outside `models/`, so nothing scans it as a model.
pub fn staging_dir(state: &SharedState, model_id: &ModelId) -> PathBuf {
    state
        .config
        .node
        .data_dir
        .join("canonical")
        .join(crate::model::shard::sanitize_path_component(&model_id.0))
}

/// Spawn the task. Runs whether or not auto-manage is on: moving parts this
/// node ALREADY holds onto the swarm's upload is repair, not a new decision
/// about what to hold — the same line `complete_pending_shard_fetches` draws.
pub async fn run(
    state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut next = tokio::time::Instant::now() + FIRST_PASS_AFTER;
    let mut queue = SwitchQueue::default();
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    return;
                }
            }
            _ = tokio::time::sleep_until(next) => {
                pass(&state, &network_tx, &mut queue).await;
                next = tokio::time::Instant::now() + PASS_EVERY;
            }
        }
    }
}

/// What one turn at switching a model came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwitchTurn {
    /// Bytes came from HuggingFace in this turn: the pass's one switch.
    Fetched,
    /// Nothing was fetched — every part was already staged (the swap ran, or
    /// waits for the model to be idle), or there is no room on disk to stage
    /// them. Re-checked next pass, which costs nothing.
    NothingFetched,
    /// HuggingFace could not give the switch what it needs (the header, a
    /// part, the side files). Not asked again until its backoff runs out.
    HuggingFaceFailed,
}

/// Which model may switch now. One switch FETCHES per pass; a switch that
/// cannot go ahead never holds the turn, and one HuggingFace failed waits out
/// a doubling backoff before it is tried again.
///
/// Before this, the turn was taken before the switch was attempted, whatever
/// came of it — so the first model (in name order) that could not switch
/// blocked every model after it, on every pass, for ever, and was itself
/// retried against HuggingFace every two minutes. Measured on the live swarm
/// 2026-10-03, 13 h after every peer reached v0.3.221: `9594e1ff` had switched
/// all eight of its models, while `4a3ac72e` still held another upload of
/// four — `llama-xlam-2-8b` first in name order, the other three behind it —
/// and `bf7b3263` of `llama-xlam-2-8b` alone. Kubernetes met this as an
/// unschedulable pod blocking the head of its scheduling queue
/// (kubernetes#71486) and answered the same way: a failed attempt goes to a
/// backoff queue with an exponentially growing wait, and the queue moves on.
#[derive(Default)]
struct SwitchQueue {
    fetched_this_pass: bool,
    /// Per model: when it may next ask HuggingFace, and the wait that set it.
    backoff: HashMap<ModelId, (tokio::time::Instant, Duration)>,
}

/// The first wait after HuggingFace failed a switch, and the longest.
const SWITCH_BACKOFF_FIRST: Duration = Duration::from_secs(10 * 60);
const SWITCH_BACKOFF_MAX: Duration = Duration::from_secs(6 * 60 * 60);

/// Whether a model's switch runs now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Go,
    /// Another switch fetched this pass.
    Queued,
    /// HuggingFace failed this switch recently.
    BackedOff,
}

impl SwitchQueue {
    fn new_pass(&mut self) {
        self.fetched_this_pass = false;
    }

    fn turn_for(&self, model: &ModelId, now: tokio::time::Instant) -> Turn {
        if self.fetched_this_pass {
            Turn::Queued
        } else if self
            .backoff
            .get(model)
            .is_some_and(|(until, _)| now < *until)
        {
            Turn::BackedOff
        } else {
            Turn::Go
        }
    }

    fn record(&mut self, model: &ModelId, turn: SwitchTurn, now: tokio::time::Instant) {
        match turn {
            SwitchTurn::Fetched => {
                self.fetched_this_pass = true;
                self.backoff.remove(model);
            }
            SwitchTurn::NothingFetched => {}
            SwitchTurn::HuggingFaceFailed => {
                let wait = self
                    .backoff
                    .get(model)
                    .map_or(SWITCH_BACKOFF_FIRST, |(_, last)| {
                        (*last * 2).min(SWITCH_BACKOFF_MAX)
                    });
                self.backoff.insert(model.clone(), (now + wait, wait));
            }
        }
    }

    /// The model is on the canonical upload now, or no longer this node's to
    /// switch: whatever held it back is history.
    fn forget(&mut self, model: &ModelId) {
        self.backoff.remove(model);
    }
}

async fn pass(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    queue: &mut SwitchQueue,
) {
    if !canonical::canonical_uploads_enabled() {
        return;
    }
    if state
        .credits
        .offline_mode
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        return;
    }
    let models = models_to_settle(state);
    let mut asked = 0usize;
    for model in &models {
        if asked >= RESOLVES_PER_PASS {
            break;
        }
        if resolve(state, model).await {
            asked += 1;
        }
    }
    queue.new_pass();
    for model in &models {
        settle(state, net_tx, model, queue).await;
    }
}

/// Models this node holds a part of, plus models it may acquire — pinned by
/// its owner or trusted for auto-manage. Ordered, so passes are repeatable.
fn models_to_settle(state: &SharedState) -> Vec<ModelId> {
    let me = state.identity.node_id().clone();
    let mut models: Vec<ModelId> = state
        .model_registry
        .shards_for_node(&me)
        .into_iter()
        .filter(|s| s.index != crate::types::MMPROJ_SHARD_INDEX)
        .map(|s| s.model_id)
        .collect();
    let known: Vec<ModelId> = state
        .models
        .hf_sources
        .iter()
        .map(|e| e.key().clone())
        .collect();
    for model in known {
        let wanted = state.models.model_trust.get(&model).is_some_and(|t| {
            t.pinned_by_user || t.trust_level >= crate::types::ModelTrustLevel::DemandVerified
        });
        if wanted {
            models.push(model);
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models.dedup();
    models
}

/// Adopt the best claimed upload of `model` if it outranks what this node
/// uses now. Returns whether HuggingFace was asked.
async fn resolve(state: &Arc<SharedState>, model: &ModelId) -> bool {
    let current = state.canonical_build(model);
    let Some(best) = state
        .origin_candidates(model)
        .into_iter()
        .find(|c| !state.origin_refused(model, c))
    else {
        return false;
    };
    if let Some(cur) = &current {
        if canonical::same_origin(&best, &cur.source)
            || !canonical::outranks(model, &best, &cur.source)
        {
            return false;
        }
    }
    match verify_upload(state, model, &best).await {
        Ok(build) => {
            tracing::info!(
                model = %model,
                repo = %best.repo_id,
                file = %best.filename,
                replaces = ?current.as_ref().map(|c| c.source.repo_id.clone()),
                "DIAG: canonical upload — every node uses this file for this model"
            );
            state.adopt_canonical_build(model, build);
            // Judge the copy held here afresh against the new upload.
            state.models.canonical_holding.remove(model);
        }
        Err((why, permanent)) => {
            tracing::info!(
                model = %model,
                repo = %best.repo_id,
                file = %best.filename,
                permanent,
                reason = %why,
                "An upload of this model could not be checked on HuggingFace — trying the next one"
            );
            state.note_origin_refused(
                model,
                &best,
                if permanent {
                    crate::daemon::state::ORIGIN_REFUSED_PERMANENT_SECS
                } else {
                    crate::daemon::state::ORIGIN_REFUSED_TRANSIENT_SECS
                },
            );
        }
    }
    true
}

/// Check an upload on HuggingFace as any node would see it, keep its header
/// in the staging directory, and describe it. `Err((why, permanent))`.
async fn verify_upload(
    state: &SharedState,
    model: &ModelId,
    source: &HfSource,
) -> Result<CanonicalBuild, (String, bool)> {
    if !canonical::origin_names_model(source, model) {
        return Err(("the file does not name this model".into(), true));
    }
    let (info, header) = crate::model::huggingface::probe_public_upload(
        &source.repo_id,
        &source.filename,
        canonical::canonical_shard_size_bytes(),
    )
    .await
    .map_err(|e| {
        let permanent = crate::model::huggingface::probe_failure_is_user_fixable(&e);
        (e, permanent)
    })?;
    let arch = &info.tensor_meta.architecture;
    if !crate::inference::split::ModelArch::from_gguf_arch(arch).is_supported() {
        return Err((format!("unsupported architecture {arch}"), true));
    }
    let staging = staging_dir(state, model);
    prepare_staging(&staging, source).map_err(|e| (e, false))?;
    let header_path = staging.join(crate::model::shard::HEADER_FILENAME);
    match header {
        Some(bytes) => write_atomic(&header_path, &bytes).map_err(|e| (e, false))?,
        None => {
            crate::model::huggingface::download_gguf_header(
                &source.repo_id,
                &source.filename,
                &staging,
                info.header_size,
            )
            .await
            .map_err(|e| (e, false))?;
        }
    }
    let header_bytes = std::fs::read(&header_path).map_err(|e| (e.to_string(), false))?;
    Ok(CanonicalBuild::from_probe(
        source.clone(),
        &info,
        &header_bytes,
        chrono::Utc::now().timestamp_millis().max(0) as u64,
    ))
}

/// Make `staging` hold only files of `source`: a directory left by another
/// upload is emptied first, so its parts can never be swapped in.
fn prepare_staging(staging: &Path, source: &HfSource) -> Result<(), String> {
    let marker = staging.join(STAGED_UPLOAD_FILENAME);
    let same = std::fs::read(&marker)
        .ok()
        .and_then(|b| serde_json::from_slice::<HfSource>(&b).ok())
        .is_some_and(|s| canonical::same_origin(&s, source));
    if !same && staging.exists() {
        std::fs::remove_dir_all(staging).map_err(|e| format!("could not clear staging: {e}"))?;
    }
    std::fs::create_dir_all(staging).map_err(|e| format!("could not create staging: {e}"))?;
    let json = serde_json::to_vec(source).map_err(|e| e.to_string())?;
    write_atomic(&marker, &json)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
}

fn hash_matches(path: &Path, expected: &[u8; 32]) -> bool {
    std::fs::read(path).is_ok_and(|b| blake3::hash(&b) == blake3::Hash::from_bytes(*expected))
}

fn set_holding(state: &SharedState, model: &ModelId, holding: Holding) {
    let changed = state
        .models
        .canonical_holding
        .insert(model.clone(), holding.clone())
        .is_none_or(|was| was != holding);
    if changed {
        state.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);
    }
}

fn holding(state: &SharedState, model: &ModelId) -> Option<Holding> {
    state
        .models
        .canonical_holding
        .get(model)
        .map(|h| h.value().clone())
}

/// Bring what this node holds of `model` onto its canonical upload.
async fn settle(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    queue: &mut SwitchQueue,
) {
    let Some(build) = state.canonical_build(model) else {
        return;
    };
    let model_dir = state.model_dir(&model.0);
    // A model this node serves from a whole GGUF its owner gave it (`-m`):
    // that file is the owner's, and is never replaced on their behalf.
    if model_dir.join("source_path").exists() {
        set_holding(state, model, Holding::Stuck { reason: "own_file" });
        return;
    }
    // A download of this model is under way — judge the copy once it has
    // finished. Replacing the manifest or header under a running download
    // changes what its parts are described by mid-flight: seen on the rig
    // (2026-10-02), where a pass landed between a dashboard download's
    // manifest and its first part. Claims cover a part being written; a
    // recent `Downloading` entry covers the gaps between parts, and only a
    // recent one, so an entry a failed path left behind cannot hold this
    // model off for ever.
    if state.models.model_download_under_way(model) {
        return;
    }
    let me = state.identity.node_id().clone();
    let held: Vec<u32> = state
        .model_registry
        .local_shard_indices(model, &me)
        .into_iter()
        .filter(|&i| i != crate::types::MMPROJ_SHARD_INDEX)
        .collect();
    let manifest = state.model_registry.get_manifest(model);

    if held.is_empty() {
        set_holding(state, model, Holding::Nothing);
        queue.forget(model);
        if manifest.as_ref().is_some_and(|m| build.describes(m)) {
            // Nothing held and nothing to register: the header `verify_upload`
            // staged has no further use. `register_for_fetching` clears it when
            // it runs; on this branch it never did, and four staged headers
            // (~7 MB each) sat in `<data_dir>/canonical/` on the live node.
            let _ = std::fs::remove_dir_all(staging_dir(state, model));
        } else {
            register_for_fetching(state, model, &build).await;
        }
        return;
    }

    if manifest.as_ref().is_some_and(|m| build.describes(m)) {
        // Checked once per run: what is on disk changes only through a switch,
        // which sets the state itself.
        if holding(state, model) == Some(Holding::Canonical) {
            return;
        }
        match parts_are_from(&build, &model_dir, &held).await {
            Ok(true) => {}
            Ok(false) => {
                tracing::warn!(model = %model, "This node's parts are the same size as the canonical upload's but not its bytes — switching");
                switch_when_possible(
                    state,
                    net_tx,
                    model,
                    &build,
                    &held,
                    manifest.as_ref(),
                    queue,
                )
                .await;
                return;
            }
            Err(e) => {
                tracing::debug!(model = %model, error = %e, "Could not compare parts with HuggingFace; asking again next pass");
                return;
            }
        }
        // The parts are right; the file that says where every tensor in them
        // is must be too.
        ensure_header(state, model, &build).await;
        if !hash_matches(
            &model_dir.join(crate::model::shard::HEADER_FILENAME),
            &build.header_hash,
        ) {
            return; // asked again next pass
        }
        // And the manifest on disk, which the worker loads its tensor table
        // from. The registry's copy can describe the canonical upload while the
        // file beside the parts describes the one this node first downloaded:
        // caught by the .220 gate (step 12k), where a dashboard download wrote
        // its own manifest after the registry had adopted a peer's, and every
        // tensor offset was off by the two headers' difference — "position … is
        // in a missing region" on every request. (No earlier gotcha recorded a
        // disk/registry manifest disagreement; #394 is the same words, another
        // cause.)
        if !ensure_manifest(state, model, &build).await {
            return; // asked again next pass
        }
        tracing::info!(model = %model, parts = held.len(), "DIAG: this node's parts are the canonical upload's");
        set_holding(state, model, Holding::Canonical);
        queue.forget(model);
        let _ = std::fs::remove_dir_all(staging_dir(state, model));
        return;
    }

    switch_when_possible(
        state,
        net_tx,
        model,
        &build,
        &held,
        manifest.as_ref(),
        queue,
    )
    .await;
}

async fn switch_when_possible(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    build: &CanonicalBuild,
    held: &[u32],
    manifest: Option<&crate::types::ModelManifest>,
    queue: &mut SwitchQueue,
) {
    if matches!(
        holding(state, model),
        Some(Holding::Stuck {
            reason: "cancelled"
        })
    ) {
        return;
    }
    match queue.turn_for(model, tokio::time::Instant::now()) {
        Turn::Go => {}
        Turn::Queued => {
            // One switch fetches at a time; say that this one is queued —
            // unless it is stuck, which says more.
            if !matches!(
                holding(state, model),
                Some(Holding::Switching { .. } | Holding::Stuck { .. })
            ) {
                set_holding(
                    state,
                    model,
                    Holding::Switching {
                        fetched: 0,
                        needed: 0,
                    },
                );
            }
            return;
        }
        // Its `Stuck` state stays on show until it is tried again.
        Turn::BackedOff => return,
    }
    let turn = switch_to(state, net_tx, model, build, held, manifest).await;
    if turn == SwitchTurn::HuggingFaceFailed {
        tracing::info!(
            model = %model,
            "Switching this model to the canonical upload could not go ahead — trying the \
             other models now, and this one again later"
        );
    }
    queue.record(model, turn, tokio::time::Instant::now());
}

/// Is every held part's first tensor byte-identical to the upload's?
async fn parts_are_from(
    build: &CanonicalBuild,
    model_dir: &Path,
    held: &[u32],
) -> Result<bool, String> {
    for &index in held {
        let Some(&(offset, size)) = build.shard_first_tensor.get(index as usize) else {
            return Ok(false);
        };
        let len = size.min(SPOT_CHECK_BYTES);
        let ours = {
            let path = model_dir.join(crate::model::shard::shard_filename(index));
            tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
                use std::io::Read;
                let mut f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                let mut buf = vec![0u8; len as usize];
                f.read_exact(&mut buf).map_err(|e| e.to_string())?;
                Ok(buf)
            })
            .await
            .map_err(|e| e.to_string())??
        };
        let theirs = crate::model::huggingface::read_public_range(
            &build.source.repo_id,
            &build.source.filename,
            offset,
            len,
        )
        .await?;
        if ours != theirs {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The canonical upload's header in staging, fetched again if it is missing
/// or not this upload's. Returns its path.
async fn staged_header(
    state: &SharedState,
    model: &ModelId,
    build: &CanonicalBuild,
) -> Result<PathBuf, String> {
    let staging = staging_dir(state, model);
    prepare_staging(&staging, &build.source)?;
    let header = staging.join(crate::model::shard::HEADER_FILENAME);
    if !hash_matches(&header, &build.header_hash) {
        crate::model::huggingface::download_gguf_header(
            &build.source.repo_id,
            &build.source.filename,
            &staging,
            build.header_size,
        )
        .await?;
        if !hash_matches(&header, &build.header_hash) {
            return Err("HuggingFace served a different header than it did when checked".into());
        }
    }
    Ok(header)
}

/// A node holding none of `model` fetches parts against a manifest of the
/// canonical upload, and nothing else. Its header goes beside them, replacing
/// one another upload may have left, along with that upload's side files.
async fn register_for_fetching(state: &SharedState, model: &ModelId, build: &CanonicalBuild) {
    let header = match staged_header(state, model, build).await {
        Ok(h) => h,
        Err(e) => {
            tracing::debug!(model = %model, error = %e, "Could not fetch the canonical upload's header yet");
            return;
        }
    };
    let model_dir = state.model_dir(&model.0);
    let installed = (|| -> Result<CanonicalManifest, String> {
        std::fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;
        for side in side_files() {
            let _ = std::fs::remove_file(model_dir.join(side));
        }
        let dest = model_dir.join(crate::model::shard::HEADER_FILENAME);
        std::fs::copy(&header, &dest).map_err(|e| e.to_string())?;
        canonical_manifest(state, model, build, &dest)
    })();
    let manifest = match installed {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(model = %model, error = %e, "Could not register the canonical upload of this model");
            return;
        }
    };
    state.forget_origin_verified_for_model(model);
    state.model_registry.remove_manifest(model);
    state.model_registry.register_manifest(manifest.0.clone());
    persist(state, &manifest.0, &model_dir);
    state.gguf_meta.remove(model);
    state.standalone_tokenizers.remove(model);
    state.models.hf_probe_cache.remove(model);
    // The header now sits beside where the parts will go; nothing is staged.
    let _ = std::fs::remove_dir_all(staging_dir(state, model));
    tracing::info!(
        model = %model,
        repo = %build.source.repo_id,
        "DIAG: registered the canonical upload's manifest — parts fetched from now on are this upload's"
    );
    state.models.auto_manage_notify.notify_one();
}

/// The files beside a model's parts that are cut from ONE upload.
fn side_files() -> [&'static str; 2] {
    [
        crate::inference::split::TIED_OUTPUT_FILENAME,
        crate::inference::split::ROPE_FREQS_FILENAME,
    ]
}

struct CanonicalManifest(crate::types::ModelManifest);

fn canonical_manifest(
    state: &SharedState,
    model: &ModelId,
    build: &CanonicalBuild,
    header: &Path,
) -> Result<CanonicalManifest, String> {
    let layouts = build
        .layouts_from_header(header)
        .ok_or("the header does not reproduce the upload's parts")?;
    let (manifest, _) = crate::model::manifest::manifest_from_header(
        header,
        model,
        &build.source.filename,
        build.total_size,
        &layouts,
        state.identity.node_id().clone(),
    )?;
    if !build.describes(&manifest) {
        return Err("the manifest built from the header does not describe the upload".into());
    }
    Ok(CanonicalManifest(manifest))
}

fn persist(state: &SharedState, manifest: &crate::types::ModelManifest, model_dir: &Path) {
    if let Err(e) = state.model_registry.persist_manifest(&state.db, manifest) {
        tracing::warn!(model = %manifest.id, error = %e, "Could not persist the canonical manifest");
    }
    if let Err(e) = manifest.save_to_dir(model_dir) {
        tracing::warn!(model = %manifest.id, error = %e, "Could not save the canonical manifest beside the parts");
    }
}

/// Make the manifest in the registry AND on disk describe the canonical
/// upload, rebuilding it from the canonical header (hashing the parts held)
/// when either does not. Returns whether both now do.
async fn ensure_manifest(
    state: &Arc<SharedState>,
    model: &ModelId,
    build: &CanonicalBuild,
) -> bool {
    let model_dir = state.model_dir(&model.0);
    let on_disk = crate::types::ModelManifest::load_from_dir(&model_dir).ok();
    let registered = state.model_registry.get_manifest(model);
    if on_disk.as_ref().is_some_and(|m| build.describes(m))
        && registered.as_ref().is_some_and(|m| build.describes(m))
    {
        return true;
    }
    if state.model_is_in_use(model) {
        return false;
    }
    let header = model_dir.join(crate::model::shard::HEADER_FILENAME);
    let rebuilt = {
        let st = state.clone();
        let m = model.clone();
        let b = build.clone();
        tokio::task::spawn_blocking(move || canonical_manifest(&st, &m, &b, &header)).await
    };
    let manifest = match rebuilt {
        Ok(Ok(m)) => m.0,
        Ok(Err(e)) => {
            tracing::warn!(model = %model, error = %e, "Could not rebuild this model's manifest from the canonical header");
            return false;
        }
        Err(e) => {
            tracing::warn!(model = %model, error = %e, "Rebuilding this model's manifest failed");
            return false;
        }
    };
    state.evict_and_unload(model).await;
    state.forget_origin_verified_for_model(model);
    state.model_registry.remove_manifest(model);
    state.model_registry.register_manifest(manifest.clone());
    persist(state, &manifest, &model_dir);
    state.gguf_meta.remove(model);
    state.standalone_tokenizers.remove(model);
    state.models.hf_probe_cache.remove(model);
    tracing::warn!(
        model = %model,
        disk_bytes = on_disk.as_ref().map(|m| m.total_size_bytes),
        repo = %build.source.repo_id,
        "Rewrote this model's manifest — the one beside its parts described another upload"
    );
    crate::model::auto_manage::spawn_check_and_load(state.clone(), model.clone());
    true
}

/// Parts that ARE the canonical upload's can still sit beside another
/// upload's header and side files: a node that took its parts from peers took
/// those from whichever source it had heard of. Replace them, and reload.
async fn ensure_header(state: &Arc<SharedState>, model: &ModelId, build: &CanonicalBuild) {
    let model_dir = state.model_dir(&model.0);
    let ours = model_dir.join(crate::model::shard::HEADER_FILENAME);
    if hash_matches(&ours, &build.header_hash) {
        return;
    }
    if state.model_is_in_use(model) {
        return;
    }
    let header = match staged_header(state, model, build).await {
        Ok(h) => h,
        Err(e) => {
            tracing::debug!(model = %model, error = %e, "Could not fetch the canonical header yet");
            return;
        }
    };
    let staging = staging_dir(state, model);
    let meta = match crate::inference::split::GgufTensorMeta::from_gguf_file(&header) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(model = %model, error = %e, "The canonical header does not parse");
            return;
        }
    };
    for side in side_files() {
        let _ = std::fs::remove_file(staging.join(side));
    }
    if let Err(e) = crate::model::huggingface::download_sidecar_tensors(
        &build.source.repo_id,
        &build.source.filename,
        &staging,
        &meta,
    )
    .await
    {
        tracing::debug!(model = %model, error = %e, "Could not fetch the canonical side files yet");
        return;
    }
    state.evict_and_unload(model).await;
    let moved = (|| -> Result<(), String> {
        for side in side_files() {
            let _ = std::fs::remove_file(model_dir.join(side));
            let staged = staging.join(side);
            if staged.exists() {
                std::fs::rename(&staged, model_dir.join(side)).map_err(|e| e.to_string())?;
            }
        }
        std::fs::copy(&header, &ours).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if let Err(e) = moved {
        tracing::warn!(model = %model, error = %e, "Could not replace this model's header");
        return;
    }
    state.gguf_meta.remove(model);
    state.standalone_tokenizers.remove(model);
    tracing::warn!(
        model = %model,
        repo = %build.source.repo_id,
        "Replaced a header from another upload of this model with the canonical one — its parts \
         were right, the file describing them was not"
    );
    crate::model::auto_manage::spawn_check_and_load(state.clone(), model.clone());
}

/// Fetch the canonical upload's parts covering the layers this node holds,
/// then swap them in for the old ones. Says what the turn came to, so a switch
/// that cannot go ahead never holds up the others ([`SwitchQueue`]).
async fn switch_to(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    build: &CanonicalBuild,
    held: &[u32],
    manifest: Option<&crate::types::ModelManifest>,
) -> SwitchTurn {
    let header = match staged_header(state, model, build).await {
        Ok(h) => h,
        Err(e) => {
            tracing::info!(model = %model, error = %e, "Could not fetch the canonical upload's header to switch this model");
            set_holding(state, model, Holding::Stuck { reason: "download" });
            return SwitchTurn::HuggingFaceFailed;
        }
    };
    let Some(layouts) = build.layouts_from_header(&header) else {
        set_holding(state, model, Holding::Stuck { reason: "download" });
        return SwitchTurn::HuggingFaceFailed;
    };
    let ranges: Vec<(u32, u32)> = manifest
        .map(|m| {
            m.shards
                .iter()
                .filter(|s| held.contains(&s.index) && s.layer_range.1 > s.layer_range.0)
                .map(|s| s.layer_range)
                .collect()
        })
        .unwrap_or_default();
    let mut targets = build.indices_covering(&ranges);
    if targets.is_empty() {
        targets = (0..build.shard_count()).collect();
    }
    let staging = staging_dir(state, model);
    let missing: u64 = targets
        .iter()
        .filter(|&&i| !part_is_staged(&staging, build, i))
        .map(|&i| build.shard_sizes[i as usize])
        .sum();
    if let Err(e) = crate::model::check_disk_space(&staging, missing) {
        if holding(state, model) != Some(Holding::Stuck { reason: "disk" }) {
            state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "model_copy_stuck",
                    format!(
                        "{} needs {} MB free to switch to the swarm's shared copy: {e}",
                        state.model_registry.display_name(model),
                        missing / (1024 * 1024)
                    ),
                )
                .with_model(model.0.clone())
                .with_toast("warning", 8000),
            );
        }
        set_holding(state, model, Holding::Stuck { reason: "disk" });
        // A smaller switch queued behind this one may still fit.
        return SwitchTurn::NothingFetched;
    }
    if !matches!(holding(state, model), Some(Holding::Switching { needed, .. }) if needed > 0) {
        tracing::info!(
            model = %model,
            from_bytes = manifest.map(|m| m.total_size_bytes),
            to_repo = %build.source.repo_id,
            parts = targets.len(),
            "DIAG: switching this node's copy to the canonical upload"
        );
        state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "download",
                "model_copy_switching",
                format!(
                    "This computer holds a different upload of {} than the rest of the swarm — \
                     fetching the shared copy ({}) so they can work together. The old parts keep \
                     working until it is in.",
                    state.model_registry.display_name(model),
                    build.source.repo_id
                ),
            )
            .with_model(model.0.clone())
            .with_toast("info", 8000),
        );
    }
    let cancel = state.models.live_cancel_flag(model);
    let needed = targets.len() as u32;
    let mut fetched_any = false;
    for (done, &index) in targets.iter().enumerate() {
        set_holding(
            state,
            model,
            Holding::Switching {
                fetched: done as u32,
                needed,
            },
        );
        if part_is_staged(&staging, build, index) {
            continue;
        }
        fetched_any = true;
        if let Err(e) = crate::model::huggingface::download_shard(
            &build.source.repo_id,
            &build.source.filename,
            &staging,
            &layouts[index as usize],
            None,
            Some(cancel.as_ref()),
        )
        .await
        {
            let cancelled = cancel.load(std::sync::atomic::Ordering::Relaxed);
            tracing::warn!(model = %model, part = index, cancelled, error = %e, "Fetching a canonical part failed");
            set_holding(
                state,
                model,
                Holding::Stuck {
                    reason: if cancelled { "cancelled" } else { "download" },
                },
            );
            // A cancelled switch is the owner's decision and is not retried at
            // all (`switch_when_possible`); either way it holds nobody up.
            return SwitchTurn::HuggingFaceFailed;
        }
    }
    let meta = match crate::inference::split::GgufTensorMeta::from_gguf_file(&header) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(model = %model, error = %e, "The canonical header does not parse");
            set_holding(state, model, Holding::Stuck { reason: "download" });
            return SwitchTurn::HuggingFaceFailed;
        }
    };
    if let Err(e) = crate::model::huggingface::download_sidecar_tensors(
        &build.source.repo_id,
        &build.source.filename,
        &staging,
        &meta,
    )
    .await
    {
        tracing::warn!(model = %model, error = %e, "Fetching the canonical side files failed");
        set_holding(state, model, Holding::Stuck { reason: "download" });
        return SwitchTurn::HuggingFaceFailed;
    }
    set_holding(
        state,
        model,
        Holding::Switching {
            fetched: needed,
            needed,
        },
    );
    // Swap only while nothing is using the model and nothing else is writing
    // its parts; the staged parts wait for the next pass otherwise.
    if !(state.model_is_in_use(model) || state.models.model_has_live_shard_download(model)) {
        swap_in(state, net_tx, model, build, &targets, held).await;
    }
    if fetched_any {
        SwitchTurn::Fetched
    } else {
        SwitchTurn::NothingFetched
    }
}

fn part_is_staged(staging: &Path, build: &CanonicalBuild, index: u32) -> bool {
    std::fs::metadata(staging.join(crate::model::shard::shard_filename(index)))
        .is_ok_and(|m| Some(&m.len()) == build.shard_sizes.get(index as usize))
}

/// Replace this node's parts of `model` with the staged canonical ones, and
/// tell the swarm.
async fn swap_in(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    build: &CanonicalBuild,
    targets: &[u32],
    old_held: &[u32],
) {
    state.evict_and_unload(model).await;
    let model_dir = state.model_dir(&model.0);
    let staging = staging_dir(state, model);
    let swap_dir = model_dir.clone();
    let swap_targets = targets.to_vec();
    let swapped =
        tokio::task::spawn_blocking(move || swap_files(&swap_dir, &staging, &swap_targets))
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);
    if let Err(e) = swapped {
        tracing::error!(model = %model, error = %e, "Swapping in the canonical parts failed");
        set_holding(state, model, Holding::Stuck { reason: "download" });
        return;
    }
    let header = model_dir.join(crate::model::shard::HEADER_FILENAME);
    let manifest = {
        let st = state.clone();
        let m = model.clone();
        let b = build.clone();
        let h = header.clone();
        match tokio::task::spawn_blocking(move || canonical_manifest(&st, &m, &b, &h)).await {
            Ok(Ok(m)) => m.0,
            Ok(Err(e)) => {
                tracing::error!(model = %model, error = %e, "The swapped-in parts do not describe the canonical upload");
                set_holding(state, model, Holding::Stuck { reason: "download" });
                return;
            }
            Err(e) => {
                tracing::error!(model = %model, error = %e, "Building the canonical manifest failed");
                set_holding(state, model, Holding::Stuck { reason: "download" });
                return;
            }
        }
    };
    let me = state.identity.node_id().clone();
    state.forget_origin_verified_for_model(model);
    state
        .model_registry
        .retain_node_shards_for_model(model, &me, &HashSet::new());
    state.model_registry.remove_manifest(model);
    state.model_registry.register_manifest(manifest.clone());
    persist(state, &manifest, &model_dir);
    let new_shards: Vec<ShardId> = targets
        .iter()
        .map(|&index| ShardId {
            model_id: model.clone(),
            index,
        })
        .collect();
    for shard in &new_shards {
        if let Some(info) = manifest.shards.iter().find(|s| s.index == shard.index) {
            // These bytes came from the upload itself.
            state.record_origin_verified_hash(shard.clone(), info.hash);
        }
        state
            .model_registry
            .record_shard_holder(shard.clone(), me.clone());
    }
    state.gguf_meta.remove(model);
    state.standalone_tokenizers.remove(model);
    state.models.hf_probe_cache.remove(model);
    let dropped: Vec<ShardId> = old_held
        .iter()
        .filter(|i| !targets.contains(i))
        .map(|&index| ShardId {
            model_id: model.clone(),
            index,
        })
        .collect();
    if !dropped.is_empty() {
        let _ = net_tx.try_send(NetworkCommand::StopProviding(dropped));
    }
    let announce = crate::model::manifest::shard_announce(
        &state.model_registry,
        me,
        new_shards.clone(),
        vec![model.clone()],
    );
    let _ = net_tx.try_send(NetworkCommand::Broadcast(
        crate::types::SwarmMessage::ShardAnnounce(announce),
    ));
    let _ = net_tx.try_send(NetworkCommand::StartProviding(new_shards));
    let _ = net_tx.try_send(NetworkCommand::Broadcast(
        crate::types::SwarmMessage::ModelManifest(manifest),
    ));
    let _ = std::fs::remove_dir_all(staging_dir(state, model));
    set_holding(state, model, Holding::Canonical);
    tracing::info!(
        model = %model,
        repo = %build.source.repo_id,
        parts = targets.len(),
        "DIAG: switched this node's copy to the canonical upload"
    );
    state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "download",
            "model_copy_switched",
            format!(
                "{} now matches the rest of the swarm ({}) — its old parts were replaced",
                state.model_registry.display_name(model),
                build.source.repo_id
            ),
        )
        .with_model(model.0.clone())
        .with_toast("success", 6000),
    );
    crate::model::auto_manage::spawn_check_and_load(state.clone(), model.clone());
}

/// The file half of the swap: delete every part, header, manifest and side
/// file of the upload being left, then move the staged ones in.
fn swap_files(model_dir: &Path, staging: &Path, targets: &[u32]) -> Result<(), String> {
    std::fs::create_dir_all(model_dir).map_err(|e| e.to_string())?;
    let leaving: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("shard_")
                || name == crate::model::shard::HEADER_FILENAME
                || name == crate::model::shard::MANIFEST_FILENAME
                || side_files().contains(&name)
        })
        .collect();
    for path in leaving {
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?;
        }
    }
    let mut moving: Vec<String> = targets
        .iter()
        .map(|&i| crate::model::shard::shard_filename(i))
        .collect();
    moving.push(crate::model::shard::HEADER_FILENAME.to_string());
    for side in side_files() {
        if staging.join(side).exists() {
            moving.push(side.to_string());
        }
    }
    for name in moving {
        std::fs::rename(staging.join(&name), model_dir.join(&name))
            .map_err(|e| format!("move {name}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The swap leaves nothing of the upload being left — not a part beyond
    /// the new set, not its header, manifest or side files — and leaves the
    /// owner's other files alone.
    #[test]
    fn the_swap_replaces_every_file_of_the_old_upload_and_nothing_else() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("models/m");
        let staging = root.path().join("canonical/m");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        for name in [
            "shard_000.bin",
            "shard_001.bin",
            "shard_002.bin",
            "shard_002.bin.tmp",
            "gguf_header.bin",
            "manifest.json",
            "tied_output_weight.bin",
            "hf_source.json",
            "notes.txt",
        ] {
            std::fs::write(model_dir.join(name), b"old").unwrap();
        }
        for name in [
            "shard_000.bin",
            "shard_001.bin",
            "gguf_header.bin",
            "rope_freqs.bin",
        ] {
            std::fs::write(staging.join(name), b"new").unwrap();
        }
        swap_files(&model_dir, &staging, &[0, 1]).unwrap();
        let read = |n: &str| std::fs::read(model_dir.join(n)).ok();
        assert_eq!(read("shard_000.bin").as_deref(), Some(&b"new"[..]));
        assert_eq!(read("shard_001.bin").as_deref(), Some(&b"new"[..]));
        assert_eq!(read("gguf_header.bin").as_deref(), Some(&b"new"[..]));
        assert_eq!(read("rope_freqs.bin").as_deref(), Some(&b"new"[..]));
        assert!(
            read("shard_002.bin").is_none(),
            "a part of the old upload outside the new set"
        );
        assert!(read("shard_002.bin.tmp").is_none());
        assert!(read("manifest.json").is_none(), "rewritten by the caller");
        assert!(
            read("tied_output_weight.bin").is_none(),
            "the old upload's side file"
        );
        assert_eq!(read("hf_source.json").as_deref(), Some(&b"old"[..]));
        assert_eq!(read("notes.txt").as_deref(), Some(&b"old"[..]));
    }

    fn m(id: &str) -> ModelId {
        ModelId(id.into())
    }

    /// The field shape (2026-10-03): the first model in name order cannot be
    /// switched. The models behind it must still get their turn in the SAME
    /// pass — before, the turn was taken before the attempt and the first
    /// failure blocked the rest on every pass.
    #[test]
    fn a_switch_that_cannot_go_ahead_does_not_hold_up_the_ones_behind_it() {
        let now = tokio::time::Instant::now();
        let mut q = SwitchQueue::default();
        q.new_pass();
        let (stuck, small, big) = (
            m("llama-xlam-2-8b-fc-r-q4-k-m"),
            m("mistral-7b-instruct-v0.3-q4-k-m"),
            m("thudm-glm-4-9b-0414-q4-k-m"),
        );
        assert_eq!(q.turn_for(&stuck, now), Turn::Go);
        q.record(&stuck, SwitchTurn::HuggingFaceFailed, now);
        assert_eq!(
            q.turn_for(&small, now),
            Turn::Go,
            "the next model is not blocked"
        );
        q.record(&small, SwitchTurn::Fetched, now);
        assert_eq!(
            q.turn_for(&big, now),
            Turn::Queued,
            "one switch FETCHES per pass — the connection is still shared out"
        );
    }

    /// A switch with no room on disk asked nobody anything: it holds no turn
    /// and is re-checked next pass, so freeing space takes effect at once.
    #[test]
    fn no_room_on_disk_holds_no_turn_and_waits_out_no_backoff() {
        let now = tokio::time::Instant::now();
        let mut q = SwitchQueue::default();
        let big = m("qwen2.5-14b-instruct-q4-k-m");
        q.new_pass();
        q.record(&big, SwitchTurn::NothingFetched, now);
        assert_eq!(q.turn_for(&m("other"), now), Turn::Go);
        q.new_pass();
        assert_eq!(q.turn_for(&big, now + PASS_EVERY), Turn::Go);
    }

    /// HuggingFace failing a switch is not asked about it again every two
    /// minutes: the wait doubles from ten minutes to six hours, and a success
    /// (or the model no longer needing a switch) clears it.
    #[test]
    fn a_switch_huggingface_failed_backs_off_doubling_up_to_a_ceiling() {
        let t0 = tokio::time::Instant::now();
        let mut q = SwitchQueue::default();
        let x = m("x-q4-k-m");
        q.record(&x, SwitchTurn::HuggingFaceFailed, t0);
        assert_eq!(q.turn_for(&x, t0 + PASS_EVERY), Turn::BackedOff);
        assert_eq!(q.turn_for(&x, t0 + SWITCH_BACKOFF_FIRST), Turn::Go);

        let t1 = t0 + SWITCH_BACKOFF_FIRST;
        q.record(&x, SwitchTurn::HuggingFaceFailed, t1);
        assert_eq!(
            q.turn_for(&x, t1 + SWITCH_BACKOFF_FIRST),
            Turn::BackedOff,
            "the second wait is twice the first"
        );
        assert_eq!(q.turn_for(&x, t1 + 2 * SWITCH_BACKOFF_FIRST), Turn::Go);

        let mut t = t1;
        for _ in 0..20 {
            q.record(&x, SwitchTurn::HuggingFaceFailed, t);
            t += SWITCH_BACKOFF_MAX;
        }
        assert_eq!(q.backoff[&x].1, SWITCH_BACKOFF_MAX, "capped");

        q.record(&x, SwitchTurn::Fetched, t);
        q.new_pass();
        assert_eq!(q.turn_for(&x, t), Turn::Go, "a success clears the backoff");
        q.record(&x, SwitchTurn::HuggingFaceFailed, t);
        q.forget(&x);
        assert_eq!(q.turn_for(&x, t), Turn::Go, "and so does being canonical");
    }

    /// Parts staged for one upload are never swapped in for another: switching
    /// the upload empties the staging directory.
    #[test]
    fn staging_for_another_upload_starts_empty() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("canonical/m");
        let a = HfSource {
            repo_id: "a/M".into(),
            filename: "M-Q4_K_M.gguf".into(),
            mmproj_filename: None,
        };
        let b = HfSource {
            repo_id: "b/M".into(),
            ..a.clone()
        };
        prepare_staging(&staging, &a).unwrap();
        std::fs::write(staging.join("shard_000.bin"), b"a's").unwrap();
        prepare_staging(&staging, &a).unwrap();
        assert!(
            staging.join("shard_000.bin").exists(),
            "same upload: kept, resumes"
        );
        prepare_staging(&staging, &b).unwrap();
        assert!(
            !staging.join("shard_000.bin").exists(),
            "another upload: emptied"
        );
    }
}
