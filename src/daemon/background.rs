//! Post-spawn background tasks kicked off by the daemon's run loop.
//!
//! Each `spawn_*` function adds a task to a shared `JoinSet<&'static str>`
//! (see [`BackgroundTasks`]) that runs until the shutdown channel fires
//! (or the task completes naturally). These are best-effort background
//! chores: shard verification, IP geolocation, startup broadcasts, key
//! rotation, opening a browser, auto-loading models, and handling SIGHUP
//! config reloads.
//!
//! Pre-2026-04-24 these used bare `tokio::spawn`, which silently swallowed
//! panics and left no way to drain the tasks during shutdown. Routing
//! through a `JoinSet` lets the supervisor surface panics in the drain
//! phase and ensures pending background work has a chance to land before
//! the daemon exits.

/// Max time `drain` waits for background tasks before forcefully aborting.
const DRAIN_TIMEOUT_SECS: u64 = 5;
/// Period for the housekeeping sweep that prunes stored responses,
/// settled pool_forwards, applied credit_txns, and expired pool invites.
const SWEEP_INTERVAL_SECS: u64 = 3600;

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;

use crate::config::Config;
use crate::types::{NetworkCommand, SwarmMessage};

use super::helpers::{detect_region_from_ip, open_browser};
use super::state::SharedState;

/// Shared `JoinSet` for the daemon's best-effort background tasks. Each
/// task returns a static name string, used by the drain phase to attribute
/// panics in logs.
pub(super) type BackgroundTasks = JoinSet<&'static str>;

/// Drain the background `JoinSet` after the supervisor has signaled
/// shutdown. Logs any task that panicked (otherwise silent under bare
/// `tokio::spawn`) and gives in-flight chores a brief window to land
/// before the daemon exits.
pub(super) async fn drain(mut tasks: BackgroundTasks) {
    if tasks.is_empty() {
        return;
    }
    tracing::debug!(
        remaining = tasks.len(),
        timeout_secs = DRAIN_TIMEOUT_SECS,
        "Draining background tasks"
    );
    let deadline = tokio::time::sleep(std::time::Duration::from_secs(DRAIN_TIMEOUT_SECS));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => {
                if !tasks.is_empty() {
                    tracing::debug!(
                        remaining = tasks.len(),
                        "Background drain timeout — aborting remaining tasks"
                    );
                    tasks.abort_all();
                }
                break;
            }
            result = tasks.join_next() => {
                match result {
                    None => break,
                    Some(Ok(name)) => {
                        tracing::debug!(task = name, "Background task exited");
                    }
                    Some(Err(e)) if e.is_panic() => {
                        tracing::error!(error = %e, "Background task panicked");
                    }
                    Some(Err(_)) => {
                        // cancellation during shutdown — expected
                    }
                }
            }
        }
    }
}

/// How long a quarantined shard is kept before its disk is reclaimed.
///
/// A shard that fails BLAKE3 verification is renamed `shard_NNN.bin.quarantine`
/// rather than deleted, so the bytes survive for inspection instead of a
/// corruption being silently erased. Nothing ever removed them, and nothing
/// surfaces them either — so in practice they were pure, invisible disk loss.
///
/// **Measured on the development node 2026-08-04: 5.5 GB across two models**,
/// including an entire 8-shard model quarantined by the v0.3.44 accept-gate bug
/// on 28 July and still resident a week later. The bytes are not counted as
/// held — the storage budget prices the registry, not the directory — but
/// they are real on the disk, and the budget's free-space clamp sees a
/// fuller disk for as long as they stay.
///
/// A day is long enough that an operator who notices a verification failure can
/// still look, and short enough that a bad batch does not squeeze out real
/// shards.
const QUARANTINE_RETENTION_SECS: u64 = 24 * 60 * 60;

/// The two names a shard is moved aside under: `.quarantine` for a hash
/// mismatch, `.mismatched` for a size one (`model::shard`). The sweep matched
/// only the first for five weeks after the second was introduced (2026-07-27),
/// so a truncated transfer sat on disk for ever — a tester's node carried
/// ~1 GB of `.mismatched` files from downloads that had failed weeks earlier
/// (gotcha #448).
const QUARANTINE_EXTENSIONS: [&str; 2] = ["quarantine", "mismatched"];

/// Delete quarantined shards older than [`QUARANTINE_RETENTION_SECS`].
///
/// Returns `(files_removed, bytes_reclaimed)`. Errors are logged and skipped —
/// failing to reclaim disk must never take the verification pass down with it.
fn sweep_expired_quarantine(models_dir: &std::path::Path) -> (u32, u64) {
    let mut files = 0u32;
    let mut bytes = 0u64;
    let Ok(model_dirs) = std::fs::read_dir(models_dir) else {
        return (0, 0);
    };
    for model_dir in model_dirs.flatten() {
        if !model_dir.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(model_dir.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let quarantined = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| QUARANTINE_EXTENSIONS.contains(&e));
            if !quarantined {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            // `modified()` is unavailable on some filesystems; treat an unknown
            // age as "not yet expired" rather than deleting something we cannot
            // date.
            let expired = meta
                .modified()
                .ok()
                .and_then(|m| m.elapsed().ok())
                .map(|age| age.as_secs() >= QUARANTINE_RETENTION_SECS)
                .unwrap_or(false);
            if !expired {
                continue;
            }
            let size = meta.len();
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    files += 1;
                    bytes = bytes.saturating_add(size);
                    tracing::info!(
                        path = %path.display(),
                        bytes = size,
                        "Reclaimed an expired quarantined shard"
                    );
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Could not remove quarantined shard");
                }
            }
        }
    }
    (files, bytes)
}

/// BLAKE3 hash check runs after API is up so the dashboard is responsive
/// immediately. Bad shards are quarantined.
pub(super) fn spawn_shard_verification(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    data_dir: PathBuf,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        // Small delay to let the API server bind and first WS clients connect
        tokio::select! {
            _ = shutdown_rx.changed() => { return "shard_verification_aborted"; }
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
        }
        let shard_store = crate::model::shard::ShardStore::new(&data_dir);
        let mut verified = 0u32;
        let mut quarantined = 0u32;
        let mut unchecked = 0u32;
        for manifest in shared_state.model_registry.models() {
            for shard_info in &manifest.shards {
                // Only verify shards we registered (i.e., we are a holder)
                let sid = crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: shard_info.index,
                };
                if !shared_state
                    .model_registry
                    .shard_holders(&sid)
                    .contains(shared_state.identity.node_id())
                {
                    continue;
                }
                let shard_path = shard_store.shard_path(&manifest.id, shard_info.index);
                if !shard_path.exists() {
                    continue;
                }
                // A shard we hold no hash for is UNCHECKED, not verified.
                //
                // Counting it as verified is what kept this failure mode
                // invisible: on 2026-08-24 this pass twice reported "all shards
                // OK verified=21" over a set including five freshly downloaded
                // shards it had never hashed — one of which was corrupt and
                // surfaced hours later, after a restart, once a manifest
                // carrying the real hash arrived. A verifier that reports work
                // it did not do is worse than none, because it reads as
                // assurance.
                //
                // There is nothing to quarantine here: the bytes may be
                // perfectly good and we simply have nothing to compare them
                // against. So it is counted separately, reported, and re-checked
                // on a later pass once the hash is known — which is what
                // `manifest::merge_known_shard_hashes` now makes possible, by
                // stopping a placeholder overwriting a hash we already held.
                if shard_info.hash == [0u8; 32] {
                    unchecked += 1;
                    continue;
                }
                // Run BLAKE3 verification in a blocking thread
                let mid = manifest.id.clone();
                let si = shard_info.clone();
                let store = crate::model::shard::ShardStore::new(&data_dir);
                let result =
                    tokio::task::spawn_blocking(move || store.verify_shard(&mid, &si)).await;
                match result {
                    Ok(Ok(())) => {
                        verified += 1;
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(
                            model = %manifest.id,
                            shard = shard_info.index,
                            error = %e,
                            "Background shard verification failed — quarantined"
                        );
                        shared_state
                            .model_registry
                            .remove_shard_holder(&sid, shared_state.identity.node_id());
                        // Removing the bad bytes is only half of it — ask for a
                        // good copy, or the model stays permanently short of a
                        // shard on any node that is not auto-managing.
                        shared_state.mark_shard_for_repair(&sid);
                        shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "model",
                                "shard_verify_failed",
                                format!(
                                    "Shard {} of {} failed verification — quarantined",
                                    crate::types::ShardId::display_index_short(shard_info.index),
                                    manifest.name
                                ),
                            )
                            .with_model(manifest.id.0.clone())
                            .with_model_name(manifest.name.clone())
                            .with_detail_num(shard_info.index as i64)
                            .with_detail_str(format!("{e}"))
                            .with_toast("warning", 5000),
                        );
                        quarantined += 1;
                    }
                    Err(_) => {
                        // JoinError — spawn_blocking panicked
                        quarantined += 1;
                    }
                }
            }
        }
        // Reclaim disk from quarantined shards old enough that nobody is going
        // to inspect them. Done here because this is the pass that creates
        // them, so the two halves of their lifetime stay in one place.
        let models_dir = shard_store.models_dir();
        let (swept_files, swept_bytes) =
            tokio::task::spawn_blocking(move || sweep_expired_quarantine(&models_dir))
                .await
                .unwrap_or_default();
        if swept_files > 0 {
            let mb = swept_bytes / (1024 * 1024);
            tracing::info!(
                files = swept_files,
                mb,
                "Reclaimed disk from expired quarantined shards"
            );
            shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "system",
                    "quarantine_reclaimed",
                    format!("Recovered {mb} MB from {swept_files} quarantined shard files"),
                )
                .with_detail_num(mb as i64),
            );
        }

        if quarantined > 0 {
            tracing::warn!(
                verified,
                quarantined,
                unchecked,
                "Background shard verification complete — some shards quarantined"
            );
        } else if unchecked > 0 {
            // Deliberately NOT "all shards OK": some were never hashed. Said
            // plainly so the gap is visible while it lasts, rather than being
            // rounded up into the success line.
            tracing::info!(
                verified,
                unchecked,
                "Background shard verification complete — some shards have no \
                 hash to check against yet"
            );
        } else {
            tracing::info!(
                verified,
                "Background shard verification complete — all shards OK"
            );
        }
        shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "shard_verified",
                format!(
                    "Verified {} shards{}{}",
                    verified,
                    if quarantined > 0 {
                        format!(" ({quarantined} quarantined)")
                    } else {
                        String::new()
                    },
                    if unchecked > 0 {
                        format!(" ({unchecked} not yet checkable)")
                    } else {
                        String::new()
                    }
                ),
            )
            .with_detail_num(verified as i64),
        );
        "shard_verification"
    });
}

/// Auto-detect region via IP geolocation (non-blocking, best-effort). If the
/// user configured a region explicitly, apply that instead.
pub(super) fn spawn_region_detection(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    if shared_state.config.identity.region.is_none() {
        let geo_state = shared_state.clone();
        tasks.spawn(async move {
            tokio::select! {
                result = detect_region_from_ip() => {
                    match result {
                        Some(code) => {
                            tracing::info!(region = %code, "Auto-detected region via IP geolocation");
                            *geo_state.detected_region.write().await = Some(code);
                        }
                        None => {
                            tracing::debug!(
                                "IP geolocation unavailable — network map will show unknown region"
                            );
                        }
                    }
                }
                _ = async {
                    loop {
                        if shutdown_rx.changed().await.is_err() { return; }
                        if *shutdown_rx.borrow() { return; }
                    }
                } => {}
            }
            "region_detection_geo"
        });
    } else {
        let state = shared_state.clone();
        tasks.spawn(async move {
            *state.detected_region.write().await = state.config.identity.region.clone();
            "region_detection_configured"
        });
    }
}

/// Broadcast shard announcements and manifests shortly after startup so peers
/// discover our shards quickly (don't wait for the 30s health tick).
pub(super) fn spawn_initial_announcements(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        // Wait for peer connections to establish, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
            _ = shutdown_rx.changed() => { return "initial_announcements_aborted"; }
        }

        let node_id = shared_state.identity.node_id().clone();

        // Broadcast shard announcements
        let hosted_shards = shared_state.model_registry.shards_for_node(&node_id);

        if !hosted_shards.is_empty() {
            // S5: Register as DHT provider for local shards
            let _ = network_tx
                .send(NetworkCommand::StartProviding(hosted_shards.clone()))
                .await;

            // Complete for every model represented, so a receiver converges
            // even if it missed a targeted retraction — the anti-entropy half
            // of the design (a Kademlia provider record is soft state kept
            // alive by republish, not by explicit delete).
            let complete_for_models: Vec<crate::types::ModelId> = hosted_shards
                .iter()
                .map(|s| s.model_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let announce = crate::types::ShardAnnounce {
                node_id: node_id.clone(),
                shards: hosted_shards,
                timestamp: chrono::Utc::now(),
                complete_for_models,
            };
            tracing::info!(
                shards = announce.shards.len(),
                "Broadcasting initial shard announcement"
            );
            let _ = network_tx
                .send(NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(
                    announce,
                )))
                .await;
        }

        // Models we published OR hold at least one shard of, so shard-holding
        // nodes propagate manifests to pure-consumer peers. Shared with the
        // health monitor's periodic broadcast — see
        // `ModelRegistry::manifests_to_gossip`.
        for manifest in shared_state.model_registry.manifests_to_gossip(&node_id) {
            let _ = network_tx
                .send(NetworkCommand::Broadcast(SwarmMessage::ModelManifest(
                    manifest,
                )))
                .await;
        }
        "initial_announcements"
    });
}

/// Spawn key rotation task (evicts stale sessions + ephemeral re-keying).
pub(super) fn spawn_key_rotation(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
) {
    let sm = shared_state.session_manager.clone();
    let node_id = shared_state.identity.node_id().clone();
    tasks.spawn(async move {
        crate::crypto::key_rotation::run_key_rotation(
            sm,
            network_tx,
            node_id,
            shared_state,
            shutdown_rx,
        )
        .await;
        "key_rotation"
    });
}

/// Open browser on first start if configured.
pub(super) fn spawn_browser_open(
    tasks: &mut BackgroundTasks,
    config: &Config,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    if !config.ui.open_browser_on_start {
        return;
    }
    let url = format!("http://localhost:{}", config.node.listen_port);
    // Check if config file exists — if not, open setup wizard
    let config_path = config.node.data_dir.join("config.toml");
    let target = if config_path.exists() {
        format!("{url}/admin")
    } else {
        format!("{url}/setup")
    };
    tasks.spawn(async move {
        // Small delay to let the server bind, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                if let Err(e) = open_browser(&target) {
                    tracing::debug!(error = %e, "Could not open browser automatically");
                }
            }
            _ = shutdown_rx.changed() => {}
        }
        "browser_open"
    });
}

/// Item 8 Phase 1: drain the model-process pool's prefix-cache manifest
/// channel and (a) gossip each update as `SwarmMessage::PrefixCacheAnnounce`
/// so peers can index our cached blocks, (b) fold our own blocks into the
/// loopback view by recording them under our `node_id` in the cross-node
/// index. This lets a single-node setup verify the wire path end-to-end —
/// after sending a prompt we should see our own NodeId returned by
/// `cross_node_prefix_holders` on the matching block hashes.
/// Drains worker progress reports onto the matching in-flight `RequestTrace`.
///
/// `active_traces` is the single place every surface already reads from — the
/// admin API, the DIAG line, the dashboard stats payload — so stamping progress
/// there means none of them need their own plumbing, and a request that
/// finishes drops its progress with the trace it belongs to.
///
/// Reports for unknown request ids are dropped without complaint: the request
/// finished (or was cancelled) between the worker sending and this draining,
/// which is normal and not worth a log line per chunk.
pub(super) fn spawn_progress_forwarder(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    mut rx: mpsc::Receiver<crate::inference::process_pool::ProgressEvent>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => break,
                msg = rx.recv() => {
                    let Some(event) = msg else { break };
                    let Some(trace) = shared_state.active_traces.get(&event.request_id) else {
                        continue;
                    };
                    trace.set_progress(event.phase.as_str(), event.done, event.total);
                    if let Some(p) = trace.progress() {
                        tracing::debug!(
                            request_id = %event.request_id,
                            phase = p.phase,
                            done = p.done,
                            total = p.total,
                            percent = ?p.percent,
                            eta_ms = ?p.eta_ms,
                            "DIAG: request progress"
                        );
                    }
                }
            }
        }
        "progress_forwarder"
    });
}

pub(super) fn spawn_prefix_announce_forwarder(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut rx: mpsc::Receiver<crate::inference::process_pool::PrefixManifestEvent>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        let our_id = shared_state.identity.node_id().clone();
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => break,
                msg = rx.recv() => {
                    let Some(event) = msg else { break };
                    let block_count = event.blocks.len();
                    // Loopback verification: record our own blocks under our
                    // NodeId so a single-node test can `cross_node_prefix_holders`
                    // and observe end-to-end indexing without needing a peer.
                    let block_hashes: Vec<[u8; 32]> = event
                        .blocks
                        .iter()
                        .map(|b| b.block_hash)
                        .collect();
                    let (added, removed) = shared_state.models.replace_peer_prefix_blocks(
                        our_id.clone(),
                        event.model_id.clone(),
                        block_hashes,
                    );
                    tracing::debug!(
                        model = %event.model_id,
                        blocks = block_count,
                        added,
                        removed,
                        "DIAG: prefix-cache loopback indexed (self)"
                    );
                    // The loopback index above is local and always runs — it is
                    // how THIS node finds its own cached prefixes. What follows
                    // leaves the machine, and a block hash is a commitment to
                    // the prompt it was taken from, so it is opt-in.
                    if !shared_state.cfg().inference.share_prefix_cache_with_peers {
                        continue;
                    }
                    let announce = crate::types::PrefixCacheAnnounce {
                        node_id: our_id.clone(),
                        model_id: event.model_id,
                        blocks: event.blocks,
                        timestamp: chrono::Utc::now(),
                    };
                    let cmd = NetworkCommand::Broadcast(SwarmMessage::PrefixCacheAnnounce(announce));
                    if let Err(e) = network_tx.send(cmd).await {
                        tracing::debug!(error = %e, "prefix-cache announce: network_tx send failed");
                    }
                }
            }
        }
        "prefix_announce_forwarder"
    });
}

/// Item 8 Phase 3: outcome of a post-fetch sanity check. `Reject` carries a
/// short reason string used for logging + informing trust-penalty decisions.
#[derive(Debug, Clone)]
enum SnapshotVerdict {
    Ok,
    Reject(&'static str),
}

/// Item 8 Phase 3: sanity-check a freshly-fetched KV snapshot against the
/// requested `block_hash` before handing the bytes to the worker. Three
/// layers of check, cheapest first:
///   1. Deserialize succeeds (magic, version, framing all valid).
///   2. BLAKE3 chain over the declared tokens at the configured block
///      size matches `requested_hash` at `snap.token_count`. This is the
///      same check the worker would do — we just short-circuit so a bad
///      peer never even gets its KV into our worker process.
///   3. Every populated layer's K/V tensors contain only finite values.
///
/// `block_size` matches the sender's `prefix_cache_block_tokens`; for
/// Phase 3 we use the local `block_tokens` as a reasonable swarm
/// convention. When the sender used a different block size the chain
/// hash check will fail (correctly) — Phase 4 will normalize block size
/// as part of the announce protocol.
fn verify_fetched_snapshot(bytes: &[u8], requested_hash: &[u8; 32]) -> SnapshotVerdict {
    use crate::inference::split;
    let device = candle_core::Device::Cpu;
    let (snap, tokens, block_size_opt) = match split::deserialize_snapshot_full(bytes, &device) {
        Ok(x) => x,
        Err(e) => {
            tracing::debug!(error = %e, "verify_fetched_snapshot: deserialize failed");
            return SnapshotVerdict::Reject("deserialize_failed");
        }
    };
    // BLAKE3 chain check — the sender claims these tokens produce
    // `requested_hash`; verify. Phase 3 senders record `block_size` in the
    // header so we know exactly what to hash against. Older snapshots fall
    // back to trying the common defaults (32, 64, 128).
    let matched = match block_size_opt {
        Some(bs) => split::verify_token_hash_chain(&tokens, bs, snap.token_count, requested_hash),
        None => [32usize, 64, 128].iter().any(|&bs| {
            split::verify_token_hash_chain(&tokens, bs, snap.token_count, requested_hash)
        }),
    };
    if !matched {
        return SnapshotVerdict::Reject("hash_chain_mismatch");
    }
    if !split::snapshot_is_finite(&snap) {
        return SnapshotVerdict::Reject("non_finite_tensors");
    }
    SnapshotVerdict::Ok
}

/// Item 8 Phase 2b: drain cross-node fetch probes from worker subprocesses,
/// resolve each via the cross-node index + remote fetch, and deliver the
/// result (hit payload or miss) back to the originating worker via the
/// pool's `send_prefix_fetch_result` IPC. Spawned once per daemon.
pub(super) fn spawn_prefix_probe_handler(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut rx: mpsc::Receiver<crate::inference::process_pool::PrefixProbeEvent>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => break,
                msg = rx.recv() => {
                    let Some(event) = msg else { break };
                    // Walk the probe's chained-hash manifest longest-first and
                    // find the best remote peer holding ANY block of it. We
                    // rebuild the manifest-to-prompt mapping from the blocks
                    // the worker shipped, rather than re-tokenizing the prompt
                    // in the daemon (which doesn't load the tokenizer).
                    //
                    // The blocks have increasing `token_count` and chained
                    // hashes that are valid iff computed over the same token
                    // sequence. We trust the worker's manifest here and let
                    // BLAKE3 verification happen in the *peer's* response path
                    // (not ours) — because WE receive the peer's tokens +
                    // snapshot and re-hash against the block hash we asked for.
                    let our_id = shared_state.identity.node_id();
                    // Item 8 Phase 3: trust-gate candidate peers. Peers below
                    // the threshold are locked out entirely — no wire round
                    // trip, no chance to poison our cache. Default threshold
                    // is DEFAULT_TRUST (0.5), which means any peer that has
                    // incurred a `SpotCheckFail` is excluded until they
                    // decay/repair their score.
                    let trust_min = shared_state
                        .config
                        .inference
                        .cross_node_prefix_trust_min;
                    let mut best: Option<(crate::types::NodeId, [u8; 32], u32)> = None;
                    if let Some(model_index) = shared_state
                        .models
                        .cross_node_prefix_index
                        .get(&event.model_id)
                    {
                        for entry in event.blocks.iter().rev() {
                            if let Some(holders) = model_index.get(&entry.block_hash) {
                                let candidates: Vec<crate::types::NodeId> = holders
                                    .iter()
                                    .map(|r| r.clone())
                                    .filter(|n| n != our_id)
                                    .filter(|n| {
                                        shared_state
                                            .credits
                                            .trust_manager
                                            .get_trust(n)
                                            >= trust_min
                                    })
                                    .collect();
                                if candidates.is_empty() {
                                    continue;
                                }
                                // Pick lowest observed per-layer latency, NodeId tiebreak.
                                let pick = candidates
                                    .into_iter()
                                    .min_by(|a, b| {
                                        let la = shared_state.observed_latency_ms_per_layer(a).unwrap_or(f32::INFINITY);
                                        let lb = shared_state.observed_latency_ms_per_layer(b).unwrap_or(f32::INFINITY);
                                        la.partial_cmp(&lb)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                            .then_with(|| a.0.cmp(&b.0))
                                    });
                                if let Some(peer) = pick {
                                    best = Some((peer, entry.block_hash, entry.token_count));
                                    break;
                                }
                            }
                        }
                    }
                    let (matched_tokens, payload) = match best {
                        Some((peer, block_hash, token_count)) => {
                            let peer_bytes_opt = shared_state
                                .peer_id_map
                                .get(&peer)
                                .map(|r| r.clone())
                                .filter(|b| !b.is_empty());
                            if let Some(peer_bytes) = peer_bytes_opt {
                                // Install the oneshot + dispatch the fetch.
                                let fetch_id = uuid::Uuid::new_v4();
                                let (tx, rx) = tokio::sync::oneshot::channel::<Option<Vec<u8>>>();
                                shared_state
                                    .pending_prefix_kv_fetches
                                    .insert(fetch_id, tx);
                                let cleanup_state = shared_state.clone();
                                struct FetchGuard {
                                    state: Arc<SharedState>,
                                    fetch_id: uuid::Uuid,
                                }
                                impl Drop for FetchGuard {
                                    fn drop(&mut self) {
                                        self.state
                                            .pending_prefix_kv_fetches
                                            .remove(&self.fetch_id);
                                    }
                                }
                                let _guard = FetchGuard {
                                    state: cleanup_state,
                                    fetch_id,
                                };
                                let cmd = NetworkCommand::SendPrefixKvFetch {
                                    target_peer_bytes: peer_bytes,
                                    request_id: fetch_id,
                                    model_id: event.model_id.clone(),
                                    block_hash,
                                };
                                if let Err(e) = network_tx.send(cmd).await {
                                    tracing::debug!(
                                        error = %e,
                                        "prefix-probe: network_tx send failed"
                                    );
                                    (0u32, None)
                                } else {
                                    // Sized for 7B-class model snapshots over loopback:
                                    // the serving peer has to route through worker IPC,
                                    // pull the snapshot, f32-serialize it, and ship it
                                    // back. Measured Qwen-7B round trip on loopback is
                                    // ~500 ms. Kept under the worker-side probe timeout
                                    // (model_worker::PREFIX_FETCH_TIMEOUT_MS) so the
                                    // worker still sees a miss verdict before its own
                                    // window closes.
                                    match tokio::time::timeout(
                                        std::time::Duration::from_millis(2500),
                                        rx,
                                    )
                                    .await
                                    {
                                        Ok(Ok(Some(bytes))) => {
                                            // Item 8 Phase 3: verify the
                                            // returned snapshot's tensors are
                                            // finite before handing the bytes
                                            // to the worker. A malicious peer
                                            // could supply BLAKE3-valid tokens
                                            // with poisoned KV tensors that
                                            // produce NaN/Inf on forward; we
                                            // reject those and penalize trust.
                                            let verdict = verify_fetched_snapshot(
                                                &bytes,
                                                &block_hash,
                                            );
                                            match verdict {
                                                SnapshotVerdict::Ok => (token_count, Some(bytes)),
                                                SnapshotVerdict::Reject(reason) => {
                                                    tracing::warn!(
                                                        %peer,
                                                        reason = reason,
                                                        bytes = bytes.len(),
                                                        "prefix-probe: rejected KV snapshot — penalizing peer trust"
                                                    );
                                                    shared_state
                                                        .credits
                                                        .trust_manager
                                                        .update_trust(
                                                            &shared_state.peer_registry,
                                                            &peer,
                                                            crate::credit::trust::TrustEvent::SpotCheckFail,
                                                        );
                                                    (0, None)
                                                }
                                            }
                                        }
                                        Ok(Ok(None)) => (0, None),
                                        Ok(Err(_)) => (0, None),
                                        Err(_) => {
                                            tracing::debug!(
                                                %peer,
                                                "prefix-probe: fetch timed out"
                                            );
                                            (0, None)
                                        }
                                    }
                                }
                            } else {
                                (0u32, None)
                            }
                        }
                        None => (0u32, None),
                    };
                    if let Err(e) = shared_state
                        .model_process_pool
                        .send_prefix_fetch_result(
                            &event.model_id,
                            event.request_id,
                            matched_tokens,
                            payload,
                        )
                        .await
                    {
                        tracing::debug!(error = %e, "prefix-probe: send_prefix_fetch_result failed");
                    }
                }
            }
        }
        "prefix_probe_handler"
    });
}

/// Auto-load models that have local shards available. Popular models
/// (by historical request count) are loaded first so they get VRAM priority
/// on restart.
pub(super) fn spawn_model_autoload(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        // Brief delay to let shard announcements propagate, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
            _ = shutdown_rx.changed() => { return "model_autoload_aborted"; }
        }
        let mut manifests = shared_state.model_registry.models();
        manifests.sort_by(|a, b| {
            let count_a = shared_state
                .models
                .model_request_counts
                .get(&a.id)
                .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
            let count_b = shared_state
                .models
                .model_request_counts
                .get(&b.id)
                .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
            count_b.cmp(&count_a)
        });
        for m in &manifests {
            if shared_state.has_split_model(&m.id) {
                continue;
            }
            // Recompute VRAM budget each iteration since loading a model consumes VRAM
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state);
            crate::model::auto_manage::check_and_load_model(&shared_state, &m.id, vram_budget)
                .await;
        }
        "model_autoload"
    });
}

/// SIGHUP config reload handler. No-op on non-Unix platforms.
#[cfg(unix)]
pub(super) fn spawn_sighup_handler(
    tasks: &mut BackgroundTasks,
    shared_state: Arc<SharedState>,
    config: Config,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        let mut sighup = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
        {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to register SIGHUP handler — config reload via signal disabled");
                return "sighup_handler_failed_init";
            }
        };
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = sighup.recv() => {
                    let config_path = config.node.data_dir.join("config.toml");
                    tracing::info!(
                        "SIGHUP received — reloading config from {}",
                        config_path.display()
                    );
                    match crate::config::reload_operational_params(&config_path) {
                        Ok(params) => {
                            let old = crate::config::OperationalParams::from_config(&config);
                            if params != old {
                                tracing::info!(
                                    ?params,
                                    "Config reloaded with changes"
                                );
                            } else {
                                tracing::info!(
                                    "Config reloaded — no changes detected"
                                );
                            }
                            shared_state.apply_config_reload(params);
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "Failed to reload config on SIGHUP"
                            );
                        }
                    }
                }
            }
        }
        "sighup_handler"
    });
}

#[cfg(not(unix))]
pub(super) fn spawn_sighup_handler(
    _tasks: &mut BackgroundTasks,
    _shared_state: Arc<SharedState>,
    _config: Config,
    _shutdown_rx: watch::Receiver<bool>,
) {
}

/// Periodically prune `/v1/responses` records whose 30-day retention has
/// elapsed. Runs hourly; the TTL is coarse so there's no value in
/// sweeping faster.
pub(super) fn spawn_responses_sweep(
    tasks: &mut BackgroundTasks,
    db: crate::storage::db::Database,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tasks.spawn(async move {
        let mut tick =
            tokio::time::interval(std::time::Duration::from_secs(SWEEP_INTERVAL_SECS));
        // If a tick gets missed (executor starvation, blocking spawn_blocking
        // contention), skip the catch-up burst — at 1h granularity a back-to-
        // back double-fire is harmless but inconsistent with every other
        // periodic loop in the codebase. Match dispatch::DRAIN_TICK_INTERVAL
        // and health::monitor::HEALTH_TICK_SECS which both set Skip.
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Skip the first tick (runs immediately) — the daemon just started
        // and there's nothing to clean up yet.
        tick.tick().await;
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = tick.tick() => {
                    let now = chrono::Utc::now().timestamp();
                    match crate::api::openai::responses::store::sweep_expired(&db, now) {
                        Ok(0) => {}
                        Ok(n) => tracing::info!(count = n, "responses sweep: pruned expired records"),
                        Err(e) => tracing::warn!(error = %e, "responses sweep failed"),
                    }
                    // Defense in depth: prune BACKGROUND_CANCEL /
                    // BACKGROUND_STATE entries whose owning task was
                    // cancelled externally (e.g. shutdown mid-flight)
                    // before its cleanup path could run. Sized at 2 h —
                    // generously above any real background-inference run.
                    let stale = crate::api::openai::responses::prune_stale_background_state();
                    if stale > 0 {
                        tracing::warn!(
                            count = stale,
                            "responses sweep: pruned stale background state \
                             (likely cancelled-without-cleanup)"
                        );
                    }
                    // Prune pool_forwards dedup audit log. Entries serve only
                    // as a replay block for a per-member rate-limit-window
                    // (60s); 30 days is generous protection while bounding
                    // disk growth. Without this, a busy pool owner accrues
                    // ~hundreds of MB/day of append-only entries forever.
                    let cutoff = chrono::Utc::now() - chrono::Duration::days(30);
                    match prune_pool_forwards_older_than(&db, cutoff) {
                        Ok(0) => {}
                        Ok(n) => tracing::info!(
                            count = n,
                            cutoff = %cutoff.to_rfc3339(),
                            "pool_forwards sweep: pruned old entries"
                        ),
                        Err(e) => tracing::warn!(error = %e, "pool_forwards sweep failed"),
                    }
                    // SEC: prune credit_txns dedup table. Entries serve only
                    // as the UUID-replay block. Replay protection requires
                    // keeping records within `BALANCE_REPORT_MAX_AGE_SECS`
                    // (5 minutes — the staleness window that gates new tx
                    // acceptance); 30 days is generous defense-in-depth
                    // while bounding disk growth. Without this, every valid
                    // gossiped CreditTransaction (~1/peer/window) writes a
                    // permanent entry — at 200-peer cap the table grows
                    // ~MB/day with no eviction.
                    match prune_credit_txns_older_than(&db, cutoff) {
                        Ok(0) => {}
                        Ok(n) => tracing::info!(
                            count = n,
                            cutoff = %cutoff.to_rfc3339(),
                            "credit_txns sweep: pruned old entries"
                        ),
                        Err(e) => tracing::warn!(error = %e, "credit_txns sweep failed"),
                    }
                    // SEC: prune expired pool_invitations from disk. The
                    // in-memory map skips expired entries on rehydration,
                    // but the DB grew unbounded. Same 30-day cutoff —
                    // invitations have a max 24h TTL so any entry past 30
                    // days is long-expired regardless of `expires_at`.
                    match prune_pool_invitations_older_than(&db, cutoff) {
                        Ok(0) => {}
                        Ok(n) => tracing::info!(
                            count = n,
                            cutoff = %cutoff.to_rfc3339(),
                            "pool_invitations sweep: pruned old entries"
                        ),
                        Err(e) => tracing::warn!(error = %e, "pool_invitations sweep failed"),
                    }
                }
            }
        }
        "responses_sweep"
    });
}

/// TREE_POOL_FORWARDS audit/dedup log pruner. Returns count of removed
/// entries. Anything older than `cutoff` is deleted.
fn prune_pool_forwards_older_than(
    db: &crate::storage::db::Database,
    cutoff: chrono::DateTime<chrono::Utc>,
) -> Result<usize, crate::error::SwarmError> {
    let entries = db.iter_json::<crate::types::PoolCreditForward>("pool_forwards")?;
    let mut removed = 0usize;
    for entry in entries {
        if entry.timestamp < cutoff {
            // Use the UUID-string key (same format the writer used at the
            // put_json call site in pool/manager/mod.rs).
            db.remove("pool_forwards", &entry.id.to_string())?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// TREE_TRANSACTIONS (credit_txns) dedup log pruner. Returns count of
/// removed entries. Mirrors `prune_pool_forwards_older_than`.
fn prune_credit_txns_older_than(
    db: &crate::storage::db::Database,
    cutoff: chrono::DateTime<chrono::Utc>,
) -> Result<usize, crate::error::SwarmError> {
    let entries =
        db.iter_json::<crate::types::CreditTransaction>(crate::credit::ledger::TREE_TRANSACTIONS)?;
    let mut removed = 0usize;
    for entry in entries {
        if entry.timestamp < cutoff {
            db.remove(
                crate::credit::ledger::TREE_TRANSACTIONS,
                &entry.id.to_string(),
            )?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// TREE_POOL_INVITATIONS pruner. Removes entries whose `expires_at` is
/// older than `cutoff`. The in-memory rehydrate path already skips
/// expired entries, but the DB grew unbounded.
fn prune_pool_invitations_older_than(
    db: &crate::storage::db::Database,
    cutoff: chrono::DateTime<chrono::Utc>,
) -> Result<usize, crate::error::SwarmError> {
    let entries = db.iter_json::<swarmllm_types::pool::PoolInvitation>("pool_invitations")?;
    let mut removed = 0usize;
    for entry in entries {
        if entry.expires_at < cutoff {
            db.remove("pool_invitations", &entry.id.to_string())?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::{verify_fetched_snapshot, SnapshotVerdict};
    use crate::inference::split::{
        compute_block_hashes, serialize_snapshot_with_block_size, KvSnapshot, KV_SNAPSHOT_MAGIC,
    };
    use candle_core::{Device, Tensor};

    fn make_snapshot(device: &Device, token_count: usize) -> KvSnapshot {
        let shape = (1usize, 1, token_count, 2);
        let n = token_count * 2;
        let k =
            Tensor::from_vec((0..n).map(|i| i as f32).collect::<Vec<_>>(), shape, device).unwrap();
        let v = Tensor::from_vec(
            (0..n).map(|i| (i + 100) as f32).collect::<Vec<_>>(),
            shape,
            device,
        )
        .unwrap();
        KvSnapshot {
            token_count,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 4096,
        }
    }

    #[test]
    fn verify_ok_when_hash_matches_and_tensors_finite() {
        let device = Device::Cpu;
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4);
        let snap = make_snapshot(&device, 8);
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, Some(4)).unwrap();
        let last_hash = hashes.last().unwrap().block_hash;
        assert!(matches!(
            verify_fetched_snapshot(&bytes, &last_hash),
            SnapshotVerdict::Ok
        ));
    }

    #[test]
    fn verify_rejects_hash_mismatch() {
        let device = Device::Cpu;
        let tokens: Vec<u32> = (1..=8).collect();
        let snap = make_snapshot(&device, 8);
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, Some(4)).unwrap();
        let bogus = [0xAB; 32];
        match verify_fetched_snapshot(&bytes, &bogus) {
            SnapshotVerdict::Reject("hash_chain_mismatch") => {}
            v => panic!("unexpected verdict: {:?}", v),
        }
    }

    #[test]
    fn verify_rejects_non_finite_tensors() {
        let device = Device::Cpu;
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4);
        // Build a snapshot with NaN inside.
        let k = Tensor::from_vec(
            vec![
                1.0f32,
                f32::NAN,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
            ],
            (1usize, 1, 8, 2),
            &device,
        )
        .unwrap();
        let v = Tensor::from_vec(
            (0..16).map(|i| (i + 100) as f32).collect::<Vec<_>>(),
            (1usize, 1, 8, 2),
            &device,
        )
        .unwrap();
        let snap = KvSnapshot {
            token_count: 8,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 4096,
        };
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, Some(4)).unwrap();
        let last_hash = hashes.last().unwrap().block_hash;
        match verify_fetched_snapshot(&bytes, &last_hash) {
            SnapshotVerdict::Reject("non_finite_tensors") => {}
            v => panic!("unexpected verdict: {:?}", v),
        }
    }

    #[test]
    fn verify_rejects_bad_magic() {
        let mut bad = vec![0u8; 32];
        bad[..4].copy_from_slice(b"NOPE");
        // Magic mismatch → deserialize_failed.
        match verify_fetched_snapshot(&bad, &[0; 32]) {
            SnapshotVerdict::Reject("deserialize_failed") => {}
            v => panic!("unexpected verdict: {:?}", v),
        }
        // Sanity: the real magic is 4 bytes.
        assert_eq!(KV_SNAPSHOT_MAGIC.len(), 4);
    }
}

#[cfg(test)]
mod quarantine_sweep_tests {
    use super::{sweep_expired_quarantine, QUARANTINE_RETENTION_SECS};

    /// Build a models dir with one model directory containing `files`, each
    /// given an age in seconds.
    fn models_dir_with(files: &[(&str, u64, u64)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("some-model");
        std::fs::create_dir_all(&model).unwrap();
        for (name, size, age_secs) in files {
            let p = model.join(name);
            std::fs::write(&p, vec![0u8; *size as usize]).unwrap();
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(*age_secs);
            // std::fs::FileTimes (stable since 1.75; MSRV here is 1.80) — no
            // extra dependency just to age a fixture.
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_times(std::fs::FileTimes::new().set_modified(when))
                .unwrap();
        }
        root
    }

    /// The disk leak: nothing ever removed quarantined shards. 5.5 GB had
    /// accumulated on the development node, including a whole model quarantined
    /// a week earlier — and because `dir_size` counts every file, those bytes
    /// counted toward `max_disk_mb`, so the node would prune LIVE shards to
    /// stay under a budget dead files were consuming.
    #[test]
    fn an_expired_quarantined_shard_is_reclaimed() {
        let root = models_dir_with(&[(
            "shard_000.bin.quarantine",
            4096,
            QUARANTINE_RETENTION_SECS + 60,
        )]);
        let (files, bytes) = sweep_expired_quarantine(root.path());
        assert_eq!(files, 1);
        assert_eq!(bytes, 4096);
        assert!(!root
            .path()
            .join("some-model/shard_000.bin.quarantine")
            .exists());
    }

    /// The retention window is the whole point — a shard quarantined moments
    /// ago must still be there for an operator who noticed the failure.
    #[test]
    fn a_recent_quarantined_shard_is_kept() {
        let root = models_dir_with(&[("shard_000.bin.quarantine", 4096, 60)]);
        let (files, bytes) = sweep_expired_quarantine(root.path());
        assert_eq!((files, bytes), (0, 0), "recent quarantine must be kept");
        assert!(root
            .path()
            .join("some-model/shard_000.bin.quarantine")
            .exists());
    }

    /// A size-mismatch quarantine (`.mismatched`) is reclaimed on the same
    /// clock as a hash-mismatch one. It was not: the sweep named one
    /// extension and the other had been introduced later (gotcha #448).
    #[test]
    fn an_expired_size_mismatch_quarantine_is_reclaimed_too() {
        let root = models_dir_with(&[
            (
                "shard_001.bin.mismatched",
                4096,
                QUARANTINE_RETENTION_SECS + 60,
            ),
            ("shard_012.bin.mismatched", 4096, 60),
        ]);
        let (files, bytes) = sweep_expired_quarantine(root.path());
        assert_eq!(
            (files, bytes),
            (1, 4096),
            "the expired one, not the recent one"
        );
        let m = root.path().join("some-model");
        assert!(!m.join("shard_001.bin.mismatched").exists());
        assert!(m.join("shard_012.bin.mismatched").exists());
    }

    /// Only quarantine files. Deleting a live shard, a header or a manifest
    /// here would be far worse than the leak this fixes.
    #[test]
    fn nothing_but_quarantine_files_is_touched() {
        let old = QUARANTINE_RETENTION_SECS + 600;
        let root = models_dir_with(&[
            ("shard_000.bin", 2048, old),
            ("gguf_header.bin", 1024, old),
            ("manifest.json", 512, old),
            ("hf_source.json", 64, old),
            ("shard_001.bin.quarantine", 999, old),
        ]);
        let (files, bytes) = sweep_expired_quarantine(root.path());
        assert_eq!((files, bytes), (1, 999), "only the quarantine file");
        let m = root.path().join("some-model");
        for keep in [
            "shard_000.bin",
            "gguf_header.bin",
            "manifest.json",
            "hf_source.json",
        ] {
            assert!(m.join(keep).exists(), "{keep} must survive");
        }
    }

    /// An empty or absent models dir must be a no-op, not an error.
    #[test]
    fn an_empty_models_dir_is_harmless() {
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(sweep_expired_quarantine(empty.path()), (0, 0));
        assert_eq!(
            sweep_expired_quarantine(&empty.path().join("does-not-exist")),
            (0, 0)
        );
    }
}
