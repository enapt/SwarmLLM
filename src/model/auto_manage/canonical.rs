//! Keeping this node on the swarm's upload of every model it holds or may
//! fetch — and healing it onto that upload when it holds anything else.
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
//!    differ everywhere — and again for every part that arrives or falls into
//!    dispute later ([`CheckedParts`]). The header beside the parts is checked
//!    by hash: a node that took its parts from peers took its header from
//!    whatever source it heard of first, which could be another upload's.
//! 3. **Prune and fetch again.** A part that is not the upload's bytes — a
//!    copy of another upload's layout, a part cut from one upload at another's
//!    offsets, a part in dispute with the swarm — is DELETED, and the upload's
//!    own parts covering the same layers are fetched in its place through the
//!    repair queue: from a peer when this node knows the part's hash to check
//!    it against, from HuggingFace otherwise ([`replace_parts`]). Nothing is
//!    deleted until HuggingFace has answered in the same pass, so whatever
//!    goes can come back. Bytes that are not their own table's upload go at
//!    once, in use or not (a request using them computes garbage); the rest
//!    wait until nothing is using the model, withheld from the swarm meanwhile.
//! 4. **Ask the holders that checked theirs.** A node whose copy this heal has
//!    checked says so in its announcements (`origin_checked_models`). The 64 KB
//!    check reads only a part's first tensor; a part every checked holder
//!    disagrees with is in dispute, and the upload's own bytes settle it. A
//!    node with NO origin to ask — offline mode, or HuggingFace never
//!    answering it — settles its parts by those holders instead
//!    ([`settle_by_checked_holders`]): a part whose bytes differ from what
//!    [`CHECKED_QUORUM`] of them agree on, none of them holding these bytes,
//!    is fetched again from them. Before v0.3.224 such a node never judged its
//!    copy at all (`9594e1ff`, FUTURE_WORK #160).

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
/// Records which upload a staging directory holds files of (its header, its
/// side files), so what was staged for one upload is never used for another.
const STAGED_UPLOAD_FILENAME: &str = "upload.json";
/// How many connected holders that checked their copy against the upload must
/// agree on a part before a node that cannot ask the upload itself deletes its
/// own different bytes of it. Two, because one is a single stranger's word —
/// the bar `ModelRegistry::mismatch_policy` exists to hold — and because the
/// replacement must exist somewhere other than one peer before ours goes:
/// HDFS invalidates a corrupt replica only while enough good replicas remain
/// (HDFS-3493). Ceph's repair, which picks an "authoritative" copy among
/// replicas, warns the pick cannot be proven without a recorded checksum; here
/// every vote comes from a holder that checked against the upload, and the
/// fetched bytes must hash to what the votes name.
const CHECKED_QUORUM: usize = 2;

/// For a node with no origin to ask: the tag its part should have instead of
/// `ours`, when at least [`CHECKED_QUORUM`] checked holders announce one and
/// the same tag and none of them announces ours. Checked holders that disagree
/// with each other settle nothing.
fn outvoted_by_checked_holders(ours: u64, checked: &[u64]) -> Option<u64> {
    let first = *checked.first()?;
    (checked.len() >= CHECKED_QUORUM
        && !checked.contains(&ours)
        && checked.iter().all(|t| *t == first))
    .then_some(first)
}

/// For a node that CAN ask the origin: does every checked holder of this part
/// hold other bytes than ours? One is enough — the dispute is settled by the
/// upload's own bytes, never by the peer, so a wrong vote costs one download.
fn disputed_by_checked_holders(ours: u64, checked: &[u64]) -> bool {
    !checked.is_empty() && !checked.contains(&ours)
}

/// Is this model served from a whole GGUF its owner gave this node (`-m`)?
/// While that file is there — or while there are no part files beside the
/// marker to judge (an owner's drive unplugged for a while is not a reason to
/// fetch the model again as parts). The marker outlives the flag: a node that
/// once ran with `-m`, or whose data folder moved to another machine or into a
/// container, keeps a `source_path` naming a file that is gone, and the heal
/// used to exempt the PARTS beside it for ever. The marker itself is left
/// alone — `-m` writes it only when it first builds the manifest, so removing
/// it would end the exemption for good once the file came back.
fn owner_serves_own_file(model_dir: &Path) -> bool {
    let Ok(named) = std::fs::read_to_string(model_dir.join("source_path")) else {
        return false;
    };
    Path::new(named.trim()).is_file() || !holds_part_files(model_dir)
}

fn holds_part_files(model_dir: &Path) -> bool {
    std::fs::read_dir(model_dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin"))
        })
    })
}

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
    let mut checked = CheckedParts::default();
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    return;
                }
            }
            _ = tokio::time::sleep_until(next) => {
                pass(&state, &network_tx, &mut checked).await;
                next = tokio::time::Instant::now() + PASS_EVERY;
            }
        }
    }
}

/// Which of a model's parts this run has checked against the canonical
/// upload, and whether each was in dispute when it was.
///
/// Before this the check ran once per run — "what is on disk changes only
/// through a switch" — and that was not so: a part fetched later from a peer is
/// verified only against the hash this node's manifest carries, which for a
/// part it did not hold came from gossip, and gossip can carry another
/// upload's. Such a part was never checked, and the copy was reported as the
/// swarm's for as long as the node ran.
#[derive(Default)]
struct CheckedParts {
    by_model: HashMap<ModelId, HashMap<u32, bool>>,
    /// Parts this run deleted and fetched again from the upload itself. The
    /// upload's own bytes have settled them: checked holders that still
    /// disagree are the ones holding something else, and asking the origin
    /// again would only fetch the same bytes again, every pass.
    from_origin: HashSet<ShardId>,
}

impl CheckedParts {
    /// The held parts never checked this run, plus any checked while
    /// undisputed that are disputed now.
    fn to_check(&self, model: &ModelId, held: &[u32], disputed: impl Fn(u32) -> bool) -> Vec<u32> {
        let seen = self.by_model.get(model);
        held.iter()
            .copied()
            .filter(|&i| match seen.and_then(|m| m.get(&i)) {
                None => true,
                Some(&was_disputed) => !was_disputed && disputed(i),
            })
            .collect()
    }

    fn record(&mut self, model: &ModelId, parts: &[u32], disputed: impl Fn(u32) -> bool) {
        let seen = self.by_model.entry(model.clone()).or_default();
        for &i in parts {
            seen.insert(i, disputed(i));
        }
    }

    /// Its parts are being replaced, or it is no longer this node's to judge.
    /// What came from the origin stays known.
    fn forget(&mut self, model: &ModelId) {
        self.by_model.remove(model);
    }

    fn note_from_origin(&mut self, model: &ModelId, parts: &[u32]) {
        for &index in parts {
            self.from_origin.insert(ShardId {
                model_id: model.clone(),
                index,
            });
        }
    }

    fn came_from_origin(&self, model: &ModelId, index: u32) -> bool {
        self.from_origin.contains(&ShardId {
            model_id: model.clone(),
            index,
        })
    }
}

async fn pass(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    checked: &mut CheckedParts,
) {
    if !canonical::canonical_uploads_enabled() {
        return;
    }
    // Offline mode never reaches HuggingFace — but its copy is still judged,
    // by the holders that checked theirs. It used to skip the heal entirely.
    let origin = if state
        .credits
        .offline_mode
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        Origin::Unreachable
    } else {
        Origin::Ask
    };
    let mut origin = origin;
    let models = models_to_settle(state);
    if origin == Origin::Ask {
        let mut asked = 0usize;
        for model in &models {
            if asked >= RESOLVES_PER_PASS {
                break;
            }
            match resolve(state, model).await {
                Asked::No => {}
                Asked::Answered => asked += 1,
                Asked::NoAnswer => {
                    origin = Origin::Unreachable;
                    break;
                }
            }
        }
    }
    for model in &models {
        settle(state, net_tx, model, checked, &mut origin).await;
    }
}

/// Whether this pass may ask HuggingFace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Ask,
    /// Offline mode — or HuggingFace did not answer earlier in this pass. Every
    /// call retries for ~155 s before it gives up (`NETWORK_RETRY_DELAYS`), so
    /// asking again for each model would hold a node with no route for many
    /// minutes a pass before any of its copies was judged at all.
    Unreachable,
}

/// What [`resolve`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asked {
    /// Nothing to ask about.
    No,
    /// HuggingFace answered — the upload checked out, or was refused for good.
    Answered,
    /// No answer that says anything about the upload: HuggingFace (or the way
    /// to it) failed.
    NoAnswer,
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
/// uses now.
async fn resolve(state: &Arc<SharedState>, model: &ModelId) -> Asked {
    let current = state.canonical_build(model);
    let Some(best) = state
        .origin_candidates(model)
        .into_iter()
        .find(|c| !state.origin_refused(model, c))
    else {
        return Asked::No;
    };
    if let Some(cur) = &current {
        if canonical::same_origin(&best, &cur.source)
            || !canonical::outranks(model, &best, &cur.source)
        {
            return Asked::No;
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
            state.note_canonical_holding(model, None);
            Asked::Answered
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
            if permanent {
                Asked::Answered
            } else {
                Asked::NoAnswer
            }
        }
    }
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
/// upload is emptied first, so nothing of it is ever installed for this one.
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
    state.note_canonical_holding(model, Some(holding));
}

fn holding(state: &SharedState, model: &ModelId) -> Option<Holding> {
    state
        .models
        .canonical_holding
        .get(model)
        .map(|h| h.value().clone())
}

/// Bring what this node holds of `model` onto its canonical upload: keep the
/// parts that are its bytes, delete the ones that are not, fetch the upload's
/// own in their place.
async fn settle(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    checked: &mut CheckedParts,
    origin: &mut Origin,
) {
    let model_dir = state.model_dir(&model.0);
    // A model this node serves from a whole GGUF its owner gave it (`-m`):
    // that file is the owner's, and is never replaced on their behalf.
    if owner_serves_own_file(&model_dir) {
        set_holding(state, model, Holding::OwnFile);
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

    // No upload verified here — offline mode, or HuggingFace has never
    // answered this node about this model. Its parts are judged by the holders
    // that checked theirs; before, they were not judged at all.
    let build = match *origin {
        Origin::Ask => state.canonical_build(model),
        Origin::Unreachable => None,
    };
    let Some(build) = build else {
        settle_by_checked_holders(state, net_tx, model, &held, manifest.as_ref()).await;
        return;
    };

    if held.is_empty() {
        set_holding(state, model, Holding::Nothing);
        checked.forget(model);
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

    let disputed = |index: u32| {
        state.shard_is_disputed(&ShardId {
            model_id: model.clone(),
            index,
        })
    };
    let same_shape = manifest.as_ref().is_some_and(|m| build.describes(m));
    if !same_shape {
        // Another upload's layout: not one of its parts is a part of this
        // upload. Before deleting anything, make sure the upload can be
        // fetched in its place — its header, from HuggingFace, now.
        if let Err(e) = staged_header(state, model, &build).await {
            tracing::info!(model = %model, error = %e, "This node holds another upload of this model and cannot reach the swarm's yet — keeping it, withheld, until it can");
            set_holding(
                state,
                model,
                Holding::Replacing {
                    parts: held.len() as u32,
                },
            );
            return;
        }
        checked.forget(model);
        let _ = replace_parts(
            state,
            net_tx,
            model,
            Some(&build),
            &held,
            Doomed::AnotherLayout,
        )
        .await;
        return;
    }

    // A copy already found canonical is checked again only where it has
    // changed since: a part that arrived after the check (a peer's transfer
    // verifies against whatever hash the manifest carries, and a hash this
    // node took from gossip can be another upload's), or one that has since
    // fallen into dispute. The rest was checked this run.
    let to_check = if holding(state, model) == Some(Holding::Canonical) {
        checked.to_check(model, &held, disputed)
    } else {
        held.clone()
    };
    // And any part every holder that checked its own copy disagrees with. The
    // 64 KB check reads only a part's first tensor, so bytes that differ past
    // it pass; the upload's own bytes settle it (`Doomed::InDispute`). Not a
    // part the upload has already settled this run.
    let connected = |n: &crate::types::NodeId| state.peer_registry.contains_key(n);
    let outvoted: Vec<u32> = held
        .iter()
        .copied()
        .filter(|&index| {
            let sid = ShardId {
                model_id: model.clone(),
                index,
            };
            !checked.came_from_origin(model, index)
                && !state.models.shards_pending_verification.contains(&sid)
                && disputed_by_checked_holders(
                    state.model_registry.announced_build_tag(&sid),
                    &state.model_registry.checked_holder_tags(&sid, connected),
                )
        })
        .collect();
    let mut asking = to_check.clone();
    asking.extend(outvoted.iter().copied());
    asking.sort_unstable();
    asking.dedup();
    if asking.is_empty() && holding(state, model) == Some(Holding::Canonical) {
        return;
    }
    let failed = match parts_not_from(&build, &model_dir, &asking).await {
        Ok(failed) => failed,
        Err(e) => {
            // No verdict from HuggingFace this pass. A node that once reached
            // it keeps the upload it verified (restored at startup), and a node
            // that has lost its route keeps landing here — so its parts are
            // judged by the checked holders instead of by nobody.
            tracing::debug!(model = %model, error = %e, "Could not compare parts with HuggingFace — judging them by the holders that checked theirs");
            *origin = Origin::Unreachable;
            settle_by_checked_holders(state, net_tx, model, &held, manifest.as_ref()).await;
            return;
        }
    };
    let doomed = doomed_parts(&asking, &failed, |i| disputed(i) || outvoted.contains(&i));
    if !doomed.is_empty() {
        tracing::warn!(
            model = %model,
            not_the_upload = ?failed,
            in_dispute = ?doomed.iter().filter(|i| !failed.contains(i) && !outvoted.contains(i)).collect::<Vec<_>>(),
            checked_holders_disagree = ?doomed.iter().filter(|i| !failed.contains(i) && outvoted.contains(i)).collect::<Vec<_>>(),
            "DIAG: parts on this node are not the canonical upload's bytes — deleting them and fetching the upload's"
        );
        checked.forget(model);
        let why = if failed.is_empty() {
            Doomed::InDispute
        } else {
            Doomed::WrongBytes
        };
        // What was deleted is fetched again from the upload itself. Only that
        // is settled: a replacement that waits for the model to be idle, or a
        // file that would not go, is judged again next pass.
        let deleted = replace_parts(state, net_tx, model, Some(&build), &doomed, why).await;
        checked.note_from_origin(model, &deleted);
        return;
    }
    checked.record(model, &to_check, disputed);
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
    if holding(state, model) != Some(Holding::Canonical) {
        tracing::info!(model = %model, parts = held.len(), "DIAG: this node's parts are the canonical upload's");
    }
    set_holding(state, model, Holding::Canonical);
    let _ = std::fs::remove_dir_all(staging_dir(state, model));
}

/// The heal for a node with no origin to ask — offline mode, or HuggingFace
/// never answering it. Its parts are judged by the holders that DID check
/// theirs against the upload (`ShardAnnounce::origin_checked_models`): a part
/// whose bytes differ from what at least [`CHECKED_QUORUM`] connected checked
/// holders agree on, none of them holding these bytes, is deleted and fetched
/// from them, checked against the full hash their tags name
/// (`ModelRegistry::heard_hash_with_tag`). Nothing goes unless the agreed
/// copy is held by enough peers connected NOW, so whatever goes can come back.
///
/// Only the same layout is judged (the agreed hash must come from a manifest
/// giving the part the size ours has): a copy of another layout is one upload's
/// header, table and bytes, computes correctly, and without HuggingFace there
/// is no way to fetch the swarm's header in its place — it stays, announced as
/// the build it is, so no peer routes to it.
///
/// The tag a part is announced under can lag its bytes for as long as a
/// re-check is pending, so those parts wait, and the bytes of every part about
/// to go are hashed first: a part that already IS the agreed bytes stays.
async fn settle_by_checked_holders(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    held: &[u32],
    manifest: Option<&crate::types::ModelManifest>,
) {
    let Some(manifest) = manifest else {
        return;
    };
    let connected = |n: &crate::types::NodeId| state.peer_registry.contains_key(n);
    let mut outvoted: Vec<(u32, crate::types::Blake3Hash)> = Vec::new();
    for &index in held {
        let sid = ShardId {
            model_id: model.clone(),
            index,
        };
        if state.models.shards_pending_verification.contains(&sid) {
            continue;
        }
        let ours = state.model_registry.announced_build_tag(&sid);
        let checked = state.model_registry.checked_holder_tags(&sid, connected);
        let Some(theirs) = outvoted_by_checked_holders(ours, &checked) else {
            continue;
        };
        let Some(size) = manifest
            .shards
            .iter()
            .find(|s| s.index == index)
            .map(|s| s.size_bytes)
        else {
            continue;
        };
        match state.model_registry.heard_hash_with_tag(&sid, theirs, size) {
            Some(hash) => outvoted.push((index, hash)),
            None => tracing::debug!(
                model = %model,
                shard = index,
                holders = checked.len(),
                "The holders that checked this part agree on other bytes than ours, and no \
                 manifest of the same layout has named their hash yet — waiting for one"
            ),
        }
    }
    if outvoted.is_empty() {
        return;
    }
    let model_dir = state.model_dir(&model.0);
    let confirmed: HashMap<u32, crate::types::Blake3Hash> =
        tokio::task::spawn_blocking(move || {
            outvoted
                .into_iter()
                .filter(|(index, agreed)| {
                    crate::model::shard::hash_file_blake3(
                        &model_dir.join(crate::model::shard::shard_filename(*index)),
                    )
                    .map_or(true, |ours| &ours != agreed)
                })
                .collect()
        })
        .await
        .unwrap_or_default();
    if confirmed.is_empty() {
        return;
    }
    let mut doomed: Vec<u32> = confirmed.keys().copied().collect();
    doomed.sort_unstable();
    tracing::warn!(
        model = %model,
        parts = ?doomed,
        "DIAG: parts on this node differ from what the holders that checked theirs against the upload agree on — this node cannot ask HuggingFace itself, so it is fetching them from those holders"
    );
    let _ = replace_parts(
        state,
        net_tx,
        model,
        None,
        &doomed,
        Doomed::Outvoted(confirmed),
    )
    .await;
}

/// Why parts of a copy are replaced — which decides whether they may wait for
/// the model to be idle, and where their replacements come from.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Doomed {
    /// A copy of another upload's LAYOUT: every part goes. Its header, table and
    /// bytes are one upload's, so it computes correctly — just not with the
    /// swarm's copy — and it waits until nothing is using it.
    AnotherLayout,
    /// Parts whose bytes are not the upload their own table (the upload's)
    /// describes: every read lands on the wrong bytes — the leftovers of the
    /// pre-v0.3.221 splice, one upload's part in three byte versions on three
    /// peers (2026-10-03). They go NOW, in use or not: a request using them is
    /// computing garbage, and failing it is what lets it be re-routed. On
    /// v0.3.222 such a copy answered `给给给…` for as long as its switch took.
    WrongBytes,
    /// Parts whose bytes passed the check but disagree with the swarm's hash:
    /// settled by the upload's own bytes once nothing is using the model.
    InDispute,
    /// On a node with no origin to ask: parts whose bytes differ from what
    /// [`CHECKED_QUORUM`] connected holders that checked theirs agree on, with
    /// the hash each replacement must have. Same layout, so — like
    /// `WrongBytes` — every read of them lands on bytes that are not the
    /// upload's: they go at once, and come back from those holders.
    Outvoted(HashMap<u32, crate::types::Blake3Hash>),
}

/// Which of the checked parts must go: every one whose bytes are not the
/// upload's, and every one in dispute with the swarm — a part whose bytes
/// disagree with the hash the swarm reports is settled by fetching the
/// upload's own, which records the upload's hash and ends the argument for
/// good (`docs/FUTURE_WORK.md` #61).
fn doomed_parts(checked: &[u32], failed: &[u32], disputed: impl Fn(u32) -> bool) -> Vec<u32> {
    checked
        .iter()
        .copied()
        .filter(|i| failed.contains(i) || disputed(*i))
        .collect()
}

/// The canonical upload's parts to fetch in place of `doomed` — those covering
/// the layers the doomed parts held. Of the same layout, that is the same
/// parts; of another, whichever of the upload's parts overlap them.
fn replacement_targets(
    build: &CanonicalBuild,
    manifest: Option<&crate::types::ModelManifest>,
    doomed: &[u32],
    why: &Doomed,
) -> Vec<u32> {
    if *why != Doomed::AnotherLayout {
        return doomed.to_vec();
    }
    let ranges: Vec<(u32, u32)> = manifest
        .map(|m| {
            m.shards
                .iter()
                .filter(|s| doomed.contains(&s.index) && s.layer_range.1 > s.layer_range.0)
                .map(|s| s.layer_range)
                .collect()
        })
        .unwrap_or_default();
    let targets = build.indices_covering(&ranges);
    if targets.is_empty() {
        // A manifest that does not say which layers its parts held: the
        // whole model, rather than a guess.
        (0..build.shard_count()).collect()
    } else {
        targets
    }
}

/// Delete this node's `doomed` parts of `model` — bytes that are not the
/// canonical upload's — and fetch the upload's parts covering the same layers
/// in their place, through the repair queue (`mark_shard_for_repair`, which
/// runs whether or not auto-manage is on): from a peer when this node knows the
/// part's hash to check it against, from HuggingFace otherwise
/// (`trigger_download`). A part that arrives is checked against its hash AND
/// against the upload's bytes before it is kept (the accept path).
///
/// This replaced a staged switch that fetched the whole copy from HuggingFace
/// beside the old one and kept serving the old parts until the swap — which
/// held another upload's bytes on a node for as long as a switch could not go
/// ahead (13 holdings on 4 peers 13 h after v0.3.221, gotcha #780), fetched
/// only from HuggingFace (#157) and needed room for two copies. Pruned first,
/// the space is there, and a peer already holding the upload can supply it.
///
/// Waits while the model is in use — except for bytes that are not their own
/// table's upload ([`Doomed::WrongBytes`], [`Doomed::Outvoted`]), which go at
/// once — and the copy is withheld from the swarm meanwhile
/// (`Holding::Replacing`).
///
/// `build` is the upload as HuggingFace described it this pass; `None` only for
/// [`Doomed::Outvoted`], on a node with no origin to ask, whose replacements
/// come from the checked holders against the hashes they agree on.
///
/// Returns the parts it actually deleted — none when it waits for the model to
/// be idle, and not one whose file would not go.
async fn replace_parts(
    state: &Arc<SharedState>,
    net_tx: &mpsc::Sender<NetworkCommand>,
    model: &ModelId,
    build: Option<&CanonicalBuild>,
    doomed: &[u32],
    why: Doomed,
) -> Vec<u32> {
    let same_shape = why != Doomed::AnotherLayout;
    let waits_for_idle = matches!(why, Doomed::AnotherLayout | Doomed::InDispute);
    if waits_for_idle && state.model_is_in_use(model) {
        set_holding(
            state,
            model,
            Holding::Replacing {
                parts: doomed.len() as u32,
            },
        );
        return Vec::new();
    }
    let me = state.identity.node_id().clone();
    let held: Vec<u32> = state
        .model_registry
        .local_shard_indices(model, &me)
        .into_iter()
        .filter(|&i| i != crate::types::MMPROJ_SHARD_INDEX)
        .collect();
    let manifest = state.model_registry.get_manifest(model);
    state.evict_and_unload(model).await;
    let model_dir = state.model_dir(&model.0);

    // The bytes.
    let removed = {
        let dir = model_dir.clone();
        let parts = doomed.to_vec();
        tokio::task::spawn_blocking(move || {
            if same_shape {
                remove_parts(&dir, &parts)
            } else {
                remove_upload_files(&dir)
            }
        })
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r)
    };
    if let Err(e) = removed {
        tracing::warn!(model = %model, error = %e, "Could not delete every part of this node's copy that is not the canonical upload's");
    }
    // The records follow the DISK: a part whose file is still there was not
    // deleted (Windows refuses to delete a file a just-stopped worker still
    // holds open) and stays recorded as held, to be judged again next pass.
    // Dropping it anyway left the file on disk with the registry saying it was
    // gone, and the repair finding it "already on disk" — the wrong bytes kept.
    let doomed: Vec<u32> = if same_shape {
        doomed
            .iter()
            .copied()
            .filter(|&i| {
                !model_dir
                    .join(crate::model::shard::shard_filename(i))
                    .exists()
            })
            .collect()
    } else {
        doomed.to_vec()
    };
    if doomed.is_empty() {
        return doomed;
    }
    let deleted = doomed.clone();
    let doomed = doomed.as_slice();
    let targets = match build {
        Some(b) => replacement_targets(b, manifest.as_ref(), doomed, &why),
        None => doomed.to_vec(),
    };

    // What this node says about them, and what it checks the replacements by.
    let kept: HashSet<u32> = held
        .iter()
        .copied()
        .filter(|i| !doomed.contains(i))
        .collect();
    state
        .model_registry
        .retain_node_shards_for_model(model, &me, &kept);
    for &index in doomed {
        state.clear_shard_dispute(&ShardId {
            model_id: model.clone(),
            index,
        });
    }
    if same_shape {
        // Their hashes are of the bytes just deleted: a replacement is checked
        // against the upload's — from a peer's gossip, or written by the
        // download from HuggingFace, or the one the checked holders agree on —
        // never against those.
        state.forget_origin_verified_for_parts(model, doomed);
        if let Some(mut m) = manifest {
            for s in m.shards.iter_mut() {
                if doomed.contains(&s.index) {
                    s.hash = match &why {
                        Doomed::Outvoted(agreed) => {
                            agreed.get(&s.index).copied().unwrap_or([0u8; 32])
                        }
                        _ => [0u8; 32],
                    };
                }
            }
            m.manifest_hash = m.compute_hash();
            // Removed first, or the merge would keep the hashes it is told to
            // forget (`merge_known_shard_hashes`).
            state.model_registry.remove_manifest(model);
            state.model_registry.register_manifest(m.clone());
            persist(state, &m, &model_dir);
        }
    } else if let Some(build) = build {
        state.forget_origin_verified_for_model(model);
        state.model_registry.remove_manifest(model);
        register_for_fetching(state, model, build).await;
    }
    // The side files are cut from part 0: when it goes, or the whole layout
    // does, the upload's own replace them. With no origin to ask they are
    // deleted instead — a node holding part 0 never reads them, and startup
    // cuts them again from the part 0 that comes back.
    if !same_shape || doomed.contains(&0) {
        match build {
            Some(build) => {
                if let Err(e) = refresh_side_files(state, model, build).await {
                    tracing::info!(model = %model, error = %e, "Could not fetch the canonical side files yet — the header check fetches them again");
                }
            }
            None => {
                for side in side_files() {
                    let _ = std::fs::remove_file(model_dir.join(side));
                }
            }
        }
    }
    state.gguf_meta.remove(model);
    state.standalone_tokenizers.remove(model);
    state.models.hf_probe_cache.remove(model);

    // Tell the swarm, then fetch again.
    let gone: Vec<ShardId> = doomed
        .iter()
        .map(|&index| ShardId {
            model_id: model.clone(),
            index,
        })
        .collect();
    let _ = net_tx.try_send(NetworkCommand::StopProviding(gone));
    let remaining: Vec<ShardId> = kept
        .iter()
        .map(|&index| ShardId {
            model_id: model.clone(),
            index,
        })
        .collect();
    let announce = crate::model::manifest::shard_announce(
        &state.model_registry,
        me,
        remaining,
        vec![model.clone()],
    );
    let _ = net_tx.try_send(NetworkCommand::Broadcast(
        crate::types::SwarmMessage::ShardAnnounce(announce),
    ));
    for &index in &targets {
        let sid = ShardId {
            model_id: model.clone(),
            index,
        };
        // A dispute is settled by the upload itself, never by a peer: the
        // hash gossip would hand the replacement is the very claim in
        // dispute, and 64 KB of agreement says nothing of the rest. The
        // origin download records the upload's hash, which ends it.
        if why == Doomed::InDispute {
            state.models.shard_p2p_failed.insert(sid.clone());
        }
        state.mark_shard_for_repair(&sid);
    }
    // Judged afresh next pass, once the replacements are in.
    state.note_canonical_holding(model, None);
    let from = match build {
        Some(b) => format!("other computers or {}", b.source.repo_id),
        None => "the other computers that checked theirs".to_string(),
    };
    tracing::warn!(
        model = %model,
        deleted = doomed.len(),
        fetching = ?targets,
        from = %from,
        "DIAG: deleted this node's parts that are not the canonical upload's — fetching the upload's in their place"
    );
    state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "download",
            "model_copy_replaced",
            format!(
                "{}: {} part(s) on this computer were not the copy the rest of the swarm uses — \
                 deleted, and the right ones are being fetched from {from}",
                state.model_registry.display_name(model),
                doomed.len(),
            ),
        )
        .with_model(model.0.clone())
        .with_toast("info", 8000),
    );
    crate::model::auto_manage::spawn_check_and_load(state.clone(), model.clone());
    deleted
}

/// Delete these parts' files (a part already gone is fine). A file Windows
/// will not delete yet — still open in a worker that was just stopped — is
/// tried again for up to two seconds; what is left after that, the caller
/// finds still on disk.
fn remove_parts(model_dir: &Path, parts: &[u32]) -> Result<(), String> {
    let mut failures = Vec::new();
    for &index in parts {
        let path = model_dir.join(crate::model::shard::shard_filename(index));
        let mut attempt = 0;
        loop {
            match std::fs::remove_file(&path) {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(e) if attempt < 10 => {
                    attempt += 1;
                    tracing::debug!(path = %path.display(), error = %e, attempt, "Could not delete a part yet — trying again");
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => {
                    failures.push(format!("remove {}: {e}", path.display()));
                    break;
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// Delete every file of the upload a model directory holds — every part,
/// partial part, header, manifest and side file — and nothing of the owner's.
fn remove_upload_files(model_dir: &Path) -> Result<(), String> {
    let leaving: Vec<PathBuf> = match std::fs::read_dir(model_dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with("shard_")
                    || name == crate::model::shard::HEADER_FILENAME
                    || name == crate::model::shard::MANIFEST_FILENAME
                    || side_files().contains(&name)
            })
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.to_string()),
    };
    for path in leaving {
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

/// The held parts whose first tensor is NOT byte-identical to the upload's.
/// `Err` when HuggingFace (or the disk) could not answer for one: no verdict,
/// and nothing may be deleted on it.
async fn parts_not_from(
    build: &CanonicalBuild,
    model_dir: &Path,
    held: &[u32],
) -> Result<Vec<u32>, String> {
    let mut failed = Vec::new();
    for &index in held {
        let path = model_dir.join(crate::model::shard::shard_filename(index));
        if !part_is_from(build, path, index).await? {
            failed.push(index);
        }
    }
    Ok(failed)
}

/// Does the part file at `path` begin with the bytes part `index` of the
/// canonical upload begins with? Its first tensor's first 64 KB, read from
/// HuggingFace anonymously — uploads of one model differ in tensor bytes
/// everywhere, so this tells them apart for 64 KB a part, where the hash in a
/// manifest only says what some node's bytes were. `Err` when either side
/// could not be read (HuggingFace unreachable, say): no verdict.
///
/// The one byte check, for a copy held here (`settle`) and for a part a peer
/// has just sent (the network manager's accept path). A part file that cannot
/// be read in full is not the upload's part.
pub(crate) async fn part_is_from(
    build: &CanonicalBuild,
    path: PathBuf,
    index: u32,
) -> Result<bool, String> {
    let Some(&(offset, size)) = build.shard_first_tensor.get(index as usize) else {
        return Ok(false);
    };
    let len = size.min(SPOT_CHECK_BYTES);
    let ours = tokio::task::spawn_blocking(move || -> Option<Vec<u8>> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path).ok()?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    })
    .await
    .map_err(|e| e.to_string())?;
    let Some(ours) = ours else {
        return Ok(false);
    };
    let theirs = crate::model::huggingface::read_public_range(
        &build.source.repo_id,
        &build.source.filename,
        offset,
        len,
    )
    .await?;
    Ok(ours == theirs)
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
    if let Err(e) = stage_side_files(state, model, build, &header).await {
        tracing::debug!(model = %model, error = %e, "Could not fetch the canonical side files yet");
        return;
    }
    state.evict_and_unload(model).await;
    let moved = move_side_files(&staging_dir(state, model), &model_dir).and_then(|()| {
        std::fs::copy(&header, &ours)
            .map(|_| ())
            .map_err(|e| e.to_string())
    });
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

/// Fetch the canonical upload's side files (`side_files`) into staging, from
/// the upload itself, as its `header` lists them.
async fn stage_side_files(
    state: &SharedState,
    model: &ModelId,
    build: &CanonicalBuild,
    header: &Path,
) -> Result<(), String> {
    let staging = staging_dir(state, model);
    let meta = crate::inference::split::GgufTensorMeta::from_gguf_file(header)
        .map_err(|e| format!("the canonical header does not parse: {e}"))?;
    for side in side_files() {
        let _ = std::fs::remove_file(staging.join(side));
    }
    crate::model::huggingface::download_sidecar_tensors(
        &build.source.repo_id,
        &build.source.filename,
        &staging,
        &meta,
    )
    .await
    .map(|_| ())
}

/// Replace the model directory's side files with the staged ones — an old one
/// the upload has no counterpart for is removed too.
fn move_side_files(staging: &Path, model_dir: &Path) -> Result<(), String> {
    for side in side_files() {
        let _ = std::fs::remove_file(model_dir.join(side));
        let staged = staging.join(side);
        if staged.exists() {
            std::fs::rename(&staged, model_dir.join(side)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// The upload's side files in place of whatever this node had.
async fn refresh_side_files(
    state: &SharedState,
    model: &ModelId,
    build: &CanonicalBuild,
) -> Result<(), String> {
    let header = staged_header(state, model, build).await?;
    stage_side_files(state, model, build, &header).await?;
    move_side_files(&staging_dir(state, model), &state.model_dir(&model.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruning another upload's copy leaves nothing of it — not a part, a
    /// partial part, its header, manifest or side files — and
    /// leaves the owner's other files alone.
    #[test]
    fn pruning_a_copy_removes_every_file_of_its_upload_and_nothing_else() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("models/m");
        std::fs::create_dir_all(&model_dir).unwrap();
        for name in [
            "shard_000.bin",
            "shard_002.bin.tmp",
            "gguf_header.bin",
            "manifest.json",
            "tied_output_weight.bin",
            "hf_source.json",
            "notes.txt",
        ] {
            std::fs::write(model_dir.join(name), b"old").unwrap();
        }
        remove_upload_files(&model_dir).unwrap();
        let left: std::collections::BTreeSet<String> = std::fs::read_dir(&model_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            left,
            ["hf_source.json", "notes.txt"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        );
        // And pruning single parts leaves the rest.
        for name in ["shard_000.bin", "shard_001.bin"] {
            std::fs::write(model_dir.join(name), b"x").unwrap();
        }
        remove_parts(&model_dir, &[1, 7]).unwrap();
        assert!(model_dir.join("shard_000.bin").exists());
        assert!(!model_dir.join("shard_001.bin").exists());
    }

    fn build3() -> CanonicalBuild {
        CanonicalBuild {
            source: HfSource {
                repo_id: "Org/X".into(),
                filename: "x-q4_k_m.gguf".into(),
                mmproj_filename: None,
            },
            total_size: 1_000,
            header_size: 100,
            shard_sizes: vec![300, 300, 300],
            shard_layers: vec![(0, 4), (4, 8), (8, 12)],
            shard_first_tensor: vec![(100, 300), (400, 300), (700, 300)],
            header_hash: [1; 32],
            resolved_at_ms: 0,
        }
    }

    /// A part goes when its bytes are not the upload's — and when it is in
    /// dispute with the swarm, which only the upload's own bytes can settle
    /// (#61). A part that passes and is not disputed stays.
    #[test]
    fn a_part_goes_when_it_is_not_the_upload_or_in_dispute() {
        let none = |_: u32| false;
        assert_eq!(doomed_parts(&[0, 1, 2], &[], none), Vec::<u32>::new());
        assert_eq!(doomed_parts(&[0, 1, 2], &[1], none), vec![1]);
        assert_eq!(doomed_parts(&[0, 1, 2], &[1], |i| i == 2), vec![1, 2]);
        // Only what was checked this pass is judged.
        assert_eq!(doomed_parts(&[2], &[], |i| i == 0), Vec::<u32>::new());
    }

    /// What is fetched in place of what goes: the same parts of the same
    /// layout; of another layout, the upload's parts overlapping the layers
    /// the old ones held — and the whole model when those are unknown.
    #[test]
    fn the_parts_fetched_again_cover_the_layers_the_deleted_ones_held() {
        let b = build3();
        assert_eq!(
            replacement_targets(&b, None, &[1], &Doomed::WrongBytes),
            vec![1]
        );
        assert_eq!(
            replacement_targets(&b, None, &[2], &Doomed::InDispute),
            vec![2]
        );
        let theirs = crate::model::manifest::build_manifest_from_gguf(
            crate::model::manifest::ManifestFromGguf {
                id: ModelId("x-q4-k-m".into()),
                name: "x".into(),
                architecture: crate::types::ModelArchitecture::Llama,
                num_layers: 12,
                total_size_bytes: 1_200,
                shard_count: 2,
                shards: vec![
                    crate::types::ShardInfo {
                        index: 0,
                        layer_range: (0, 6),
                        size_bytes: 550,
                        hash: [0; 32],
                        tensors: Vec::new(),
                    },
                    crate::types::ShardInfo {
                        index: 1,
                        layer_range: (6, 12),
                        size_bytes: 550,
                        hash: [0; 32],
                        tensors: Vec::new(),
                    },
                ],
                publisher: crate::types::NodeId([0; 32]),
            },
        );
        assert_eq!(
            replacement_targets(&b, Some(&theirs), &[0], &Doomed::AnotherLayout),
            vec![0, 1]
        );
        assert_eq!(
            replacement_targets(&b, Some(&theirs), &[1], &Doomed::AnotherLayout),
            vec![1, 2]
        );
        assert_eq!(
            replacement_targets(&b, None, &[0], &Doomed::AnotherLayout),
            vec![0, 1, 2]
        );
    }

    fn m(id: &str) -> ModelId {
        ModelId(id.into())
    }

    /// The field shape (2026-10-03): a copy found canonical was never checked
    /// again in that run, so a part added later — from a peer, verified only
    /// against a hash gossip supplied — went unchecked for as long as the node
    /// ran. Only what changed is checked again: a new part, or one that fell
    /// into dispute after it was checked.
    #[test]
    fn a_part_that_arrives_after_the_check_is_checked_too() {
        let mut c = CheckedParts::default();
        let x = m("x-q4-k-m");
        let none = |_: u32| false;
        assert_eq!(c.to_check(&x, &[0, 1, 2], none), vec![0, 1, 2]);
        c.record(&x, &[0, 1, 2], none);
        assert!(c.to_check(&x, &[0, 1, 2], none).is_empty(), "all checked");

        assert_eq!(c.to_check(&x, &[0, 1, 2, 5], none), vec![5], "a new part");

        let two_disputed = |i: u32| i == 2;
        assert_eq!(
            c.to_check(&x, &[0, 1, 2], two_disputed),
            vec![2],
            "a part in dispute since its check"
        );
        c.record(&x, &[2], two_disputed);
        assert!(
            c.to_check(&x, &[0, 1, 2], two_disputed).is_empty(),
            "checked while disputed: not asked about every pass"
        );

        c.forget(&x);
        assert_eq!(
            c.to_check(&x, &[0, 1], none),
            vec![0, 1],
            "a replacement starts over"
        );
    }

    /// Files staged for one upload are never used for another: staging for
    /// another upload empties the directory first.
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

    /// A node with no origin to ask replaces a part only on the word of
    /// [`CHECKED_QUORUM`] checked holders that agree with each other, while
    /// none of them holds its bytes. The 2026-10-03 shape: `9594e1ff` held
    /// parts whose tags no checked holder shared.
    #[test]
    fn a_part_is_outvoted_only_by_enough_checked_holders_that_agree() {
        let (ours, right, other) = (11u64, 22u64, 33u64);
        assert_eq!(
            outvoted_by_checked_holders(ours, &[right, right]),
            Some(right)
        );
        assert_eq!(
            outvoted_by_checked_holders(ours, &[right, right, right]),
            Some(right)
        );
        assert_eq!(
            outvoted_by_checked_holders(ours, &[right]),
            None,
            "one stranger's word is not enough to delete bytes"
        );
        assert_eq!(
            outvoted_by_checked_holders(ours, &[right, right, ours]),
            None,
            "a checked holder of OUR bytes means the swarm is not agreed"
        );
        assert_eq!(
            outvoted_by_checked_holders(ours, &[right, other]),
            None,
            "checked holders that disagree with each other settle nothing"
        );
        assert_eq!(outvoted_by_checked_holders(ours, &[]), None);
    }

    /// A node that CAN ask the origin takes one checked holder's disagreement
    /// as a dispute — the upload's bytes settle it, so a wrong vote costs one
    /// download, never the bytes.
    #[test]
    fn one_checked_holder_disagreeing_puts_a_part_in_dispute_where_the_origin_answers() {
        assert!(disputed_by_checked_holders(1, &[2]));
        assert!(!disputed_by_checked_holders(1, &[2, 1]));
        assert!(!disputed_by_checked_holders(1, &[]));
    }

    /// A part the upload has settled this run is not asked about again because
    /// peers still disagree — and that knowledge survives `forget`, which every
    /// replacement calls.
    #[test]
    fn a_part_fetched_again_from_the_origin_is_not_judged_by_peers_again() {
        let mut c = CheckedParts::default();
        let x = m("x-q4-k-m");
        assert!(!c.came_from_origin(&x, 3));
        c.note_from_origin(&x, &[3]);
        c.forget(&x);
        assert!(c.came_from_origin(&x, 3));
        assert!(!c.came_from_origin(&x, 2));
    }

    /// The owner's own GGUF exempts a copy only while it is there, or while
    /// there are no parts beside the marker to judge — and the marker is never
    /// removed, since `-m` writes it only once.
    #[test]
    fn an_owners_file_exempts_its_parts_only_while_it_is_there() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("models/m");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!owner_serves_own_file(&dir), "no marker");

        let gguf = root.path().join("own.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        std::fs::write(dir.join("source_path"), gguf.display().to_string()).unwrap();
        std::fs::write(dir.join("shard_000.bin"), b"x").unwrap();
        assert!(owner_serves_own_file(&dir), "the file is there");

        std::fs::remove_file(&gguf).unwrap();
        assert!(
            !owner_serves_own_file(&dir),
            "the file is gone and parts are here: they are judged"
        );
        assert!(dir.join("source_path").exists(), "the marker stays");

        std::fs::remove_file(dir.join("shard_000.bin")).unwrap();
        assert!(
            owner_serves_own_file(&dir),
            "nothing beside the marker to judge — an unplugged drive is not a reason to fetch the model"
        );
    }
}
