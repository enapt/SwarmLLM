use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::types::NodeId;

use super::hf::{HfProbeInfo, HfSource};

/// Model management: shard acquisition, auto-manage, trust gating, pruning.
pub struct ModelMgmt {
    pub acquisition_progress:
        DashMap<crate::types::ModelId, crate::model::acquisition::AcquisitionStatus>,
    pub hf_sources: DashMap<crate::types::ModelId, HfSource>,
    pub auto_manage_notify: Arc<tokio::sync::Notify>,
    pub auto_manage_enabled: std::sync::atomic::AtomicBool,
    pub auto_manage_default_model_cap: AtomicU32,
    pub model_auto_manage_policies:
        DashMap<crate::types::ModelId, crate::config::ModelAutoManagePolicy>,
    pub hf_probe_cache: DashMap<crate::types::ModelId, HfProbeInfo>,
    pub peer_shard_downloads: DashMap<crate::types::ShardId, Vec<(NodeId, u32)>>,
    pub download_cancel_flags: DashMap<crate::types::ModelId, Arc<AtomicBool>>,
    pub model_trust: DashMap<crate::types::ModelId, crate::types::ModelTrustInfo>,
    pub loading_models: DashMap<crate::types::ModelId, Arc<tokio::sync::Notify>>,
    pub locked_shards: DashMap<crate::types::ShardId, bool>,
    /// Shards the operator removed by hand (dashboard / API delete), persisted
    /// in the `removed_shards` DB tree. Auto-manage must not bring one of these
    /// back on its own — see `SharedState::shard_removed_by_user` and the
    /// scorer skip in `model/auto_manage/scoring.rs`. Cleared by an explicit
    /// request for the shard again (HF shard download, single-shard download,
    /// a pool pin to this device).
    pub removed_by_user: DashMap<crate::types::ShardId, bool>,
    /// Shards where P2P download has exhausted all peer attempts in this session.
    /// Signals auto_manage to force the HF path even when peer holders are registered.
    /// Cleared when a download for the shard successfully completes.
    pub shard_p2p_failed: dashmap::DashSet<crate::types::ShardId>,
    /// Shards whose bytes were found WRONG and which need a fresh, verified copy.
    ///
    /// Written only by `SharedState::mark_shard_for_repair`, from the three
    /// places that can catch a bad shard: the P2P accept path, the background
    /// verification sweep, and the auto-manage rescan. Drained by
    /// `AutoShardManager::complete_pending_shard_fetches`, which runs OUTSIDE
    /// the `auto_manage.enabled` gate — a shard this node already held and has
    /// just lost to corruption is not a new acquisition decision, so repairing
    /// it is not "managing your disk on your behalf".
    ///
    /// Deliberately NOT `shard_p2p_failed`: that flag forces the HuggingFace
    /// path, and a repair should be free to fetch from a peer. Detecting the
    /// corruption at all means we HAVE the real hash, so a peer copy will be
    /// verified against it — and if that copy is bad too, the accept path
    /// quarantines it and docks the sender, which is the behaviour we want.
    /// An entry is dropped once the shard is back on disk.
    pub shards_needing_repair: dashmap::DashSet<crate::types::ShardId>,
    /// Shards this node HOLDS whose expected hash has just changed, and whose
    /// bytes must therefore be re-checked against the new reference.
    ///
    /// Written by the registry's manifest-update hook; drained by
    /// `AutoShardManager::verify_pending_shards`, which quarantines and requests
    /// a replacement on mismatch.
    ///
    /// **This is how a node learns from the swarm that what it is serving is
    /// wrong.** The only other re-check of an already-held shard is the startup
    /// sweep, which runs seconds after boot against whatever the database held —
    /// i.e. before any corrected hash arrives by gossip — so without this it took
    /// a further restart to notice. Measured on the live swarm: a shard corrupt
    /// on at least two peers, each of which could have been told the right hash
    /// (gotcha #382).
    pub shards_pending_verification: dashmap::DashSet<crate::types::ShardId>,
    /// Shards this node HOLDS whose bytes disagree with the hash the swarm
    /// reports — and which are kept and served anyway, because that hash has
    /// no origin backing.
    ///
    /// The third of the trio, and the one where nothing happens next.
    /// `shards_needing_repair` says "these bytes are wrong, get new ones";
    /// `shards_pending_verification` says "re-check these against a hash that
    /// just changed"; this one says **"we checked, we disagree, and we are
    /// standing by our copy"** — see `ModelRegistry::mismatch_policy` for why
    /// destroying it needs better evidence than a stranger's claim.
    ///
    /// It exists because the disagreement was otherwise a `u32` local to one
    /// startup task, logged once and dropped. That made it invisible to the
    /// dashboard, to the diagnostics report a reporter pastes, and therefore to
    /// us — while the open question about this whole path is *how often it
    /// fires in the field* (`docs/FUTURE_WORK.md` § "A disputed shard is kept
    /// but the disagreement is never settled", whose stated precondition is
    /// "count disputes first"). An instrument nobody can read is not an
    /// instrument.
    ///
    /// Written and cleared ONLY by `SharedState::note_shard_disputed` /
    /// `clear_shard_dispute`, from the two paths that re-check bytes already on
    /// disk: the startup verification sweep and the auto-manage rescan. Both
    /// clear on a later successful verify, which is how a dispute resolves once
    /// the origin's hash arrives — no separate expiry, because the only thing
    /// that can settle it is another check.
    pub disputed_shards: dashmap::DashSet<crate::types::ShardId>,
    /// Per-shard download backoff. A shard whose download fails (hard HF error,
    /// GGUF-probe failure, P2P give-up with no HF fallback, or stall-
    /// reconciliation in `health/monitor.rs`) records an exponentially-growing
    /// cooldown here. `gather_candidates` skips a shard while it's in backoff,
    /// so one persistently-stuck download can't monopolize a
    /// `max_concurrent_downloads` slot and starve every other candidate —
    /// including a fresh model a user is actively waiting on (external report,
    /// 2026-07-23). Distinct from `shard_p2p_failed` (which only *forces* the HF
    /// path, without throttling re-selection). Cleared on successful completion;
    /// expired-and-idle entries self-evict on read to keep the map bounded.
    pub shard_download_backoff: DashMap<crate::types::ShardId, ShardDownloadBackoff>,
    pub model_request_counts: DashMap<crate::types::ModelId, AtomicU64>,
    pub resource_schedule: RwLock<crate::config::ResourceSchedule>,
    pub prune_history: RwLock<VecDeque<crate::types::PruneEvent>>,
    /// Parallax Phase C.2 stability counter per shard. Positive values mean
    /// the allocator has recommended this node hold the shard for N
    /// consecutive auto-manage ticks; negative values mean the allocator
    /// wants it off this node. Score biases trigger once the magnitude
    /// crosses `PARALLAX_STABILITY_THRESHOLD`. Clamped to `[-10, 10]` so a
    /// long-stable recommendation can't be flipped by a single noisy tick.
    pub parallax_stability: DashMap<crate::types::ShardId, i32>,
    /// Item 8 Phase 1: cross-node prefix-cache index. Outer key = `model_id`,
    /// inner key = chained BLAKE3 block hash, value = the set of remote
    /// peers known to hold a KV snapshot ending at that block. Updated from
    /// `SwarmMessage::PrefixCacheAnnounce` and consulted by Phase 2's
    /// remote-KV fetch path. We never insert ourselves here — local cache
    /// hits are served by the in-process `PrefixCache` directly.
    pub cross_node_prefix_index:
        DashMap<crate::types::ModelId, DashMap<[u8; 32], dashmap::DashSet<NodeId>>>,
    /// Reverse index from `peer_id → list of (model_id, block_hash)` so that
    /// when a peer disconnects (or sends a fresh announce that supersedes
    /// the previous one) we can remove every entry attributed to them
    /// without rescanning the per-model maps. Wrapped in an `RwLock` rather
    /// than `DashMap<_, Mutex<_>>` because peer announces are rare relative
    /// to lookups; a single short write lock per announce is cheaper than
    /// per-bucket locking.
    pub peer_prefix_blocks:
        DashMap<NodeId, DashMap<crate::types::ModelId, dashmap::DashSet<[u8; 32]>>>,
    /// OwnedSemaphorePermits for in-flight P2P shard downloads, paired with
    /// the time the permit was parked. The P2P path queues a
    /// `NetworkCommand::SendShardRequest` and returns immediately, so the
    /// permit can't be held on `trigger_download`'s stack — it's parked
    /// here keyed by `ShardId` and released from the network event loop
    /// (success path + retry-fallback give-up + stall watchdog). HF and
    /// mmproj paths hold their permits in-task and don't touch this map.
    /// Without this, `max_concurrent_downloads` only bounded HF — P2P
    /// permits dropped the moment the request was queued, so the
    /// semaphore had no effect on P2P load.
    ///
    /// SEC: the `Instant` is consumed by `AutoShardManager::sweep_stalled_p2p_permits`
    /// — without that periodic sweep, a silent network drop (libp2p
    /// dispatch loop missed event, peer disconnected before request
    /// landed) parks the permit forever and `max_concurrent_downloads`
    /// silently freezes after enough silent drops.
    pub p2p_download_permits: DashMap<crate::types::ShardId, P2pDownloadSlot>,
    /// Shards a download task is *actually writing right now*, one entry per
    /// live writer, held by an RAII `ShardDownloadClaim`.
    ///
    /// The thing being protected is a FILE: every fetch of shard N writes
    /// `shard_NNN.bin.tmp` in the model directory, and the HuggingFace path and
    /// the P2P path write it in different formats (HF packs tensor bytes and
    /// pins the layout with a `.tmp.layout` sidecar; P2P writes raw shard bytes
    /// at a chunk offset and resumes from `tmp_size`). Two writers on that file
    /// — two HF attempts, two P2P attempts, or one of each — corrupt it, and the
    /// coalesced-range cleanup of whichever finishes first deletes the other's
    /// `.tmp` and sidecar out from under it.
    ///
    /// Exclusion used to rest entirely on `acquisition_progress` carrying a
    /// `Downloading` mark. That map is a PROGRESS structure: several subsystems
    /// write it, and `schedule_acquisition_cleanup` deletes whole models from it
    /// on a timer. Coupling a concurrency guard to it has now failed twice in
    /// the field — 2026-09-11 (a second request erased the marks of shards
    /// already in flight) and 2026-09-13 (one shard completing deleted the whole
    /// entry while three others were still downloading, so the next auto-manage
    /// tick started duplicates and the same ~512 MB shard re-downloaded from
    /// byte zero every five minutes for hours). This set exists so the answer
    /// does not depend on that map surviving: it is written only by
    /// `claim_shard_download` and cleared only by dropping the claim, so a task
    /// that ends — returns, errors, is aborted, or is dropped mid-await —
    /// releases it with no cleanup call to forget.
    ///
    /// Read through `is_shard_in_progress`, never directly, so the ~4 consumers
    /// of that predicate cannot disagree about what "in progress" means.
    /// `huggingface_hub` guards its own `.incomplete` blobs with a per-blob lock
    /// file for exactly this reason; in-process is enough here because a data
    /// directory already has a single daemon (redb holds it).
    pub shard_download_claims: Arc<dashmap::DashSet<crate::types::ShardId>>,
    /// R111: latest computed Wishlist snapshot. ArcSwap so the dashboard +
    /// REST + future HfWatcher all read a lock-free snapshot. Refreshed on
    /// every WS stats build (cheap pass over the model registry) and on
    /// every auto-manage tick.
    pub wishlist: arc_swap::ArcSwap<crate::model::auto_manage::wishlist::Wishlist>,
    /// R112: cache of the most recent HuggingFace trending-GGUF poll.
    /// Populated by `HfWatcher` once an hour; consumed by the wishlist
    /// scorer (boosts trending models) and by the future task-filter
    /// view in the HF browser. Empty until the first successful fetch.
    pub hf_trending_cache: arc_swap::ArcSwap<crate::model::huggingface::HfTrendingSnapshot>,
    /// R130: foreign wishlist interest, populated from inbound
    /// `WishlistAnnouncement` gossip. Key = (publisher node id, model id);
    /// value = (publisher's score 0..100, timestamp_ms received). Capped
    /// at `MAX_FOREIGN_WISHLIST_ENTRIES`. Pruned on read in
    /// `compute_wishlist` (drops entries older than
    /// `FOREIGN_WISHLIST_MAX_AGE_MS`). Empty when wishlist gossip is
    /// disabled — the publisher gate also blocks ingest unrelated to
    /// the publish flag, so a node that opts out of *publishing* still
    /// accepts inbound boosts.
    pub foreign_wishlist: DashMap<(NodeId, crate::types::ModelId), (u32, u64)>,
    /// R133: latest cached quant recommendations. Refreshed on every
    /// auto-manage tick alongside the wishlist. `ArcSwap` so the
    /// dashboard + REST handler can read a lock-free snapshot.
    pub quant_recommendations:
        arc_swap::ArcSwap<crate::model::auto_manage::quant::QuantRecommendations>,
}

/// Maximum number of `(publisher, model_id)` entries we retain from inbound
/// wishlist gossip. ~10K entries × ~30 models per top-K announce → ~333
/// publishers worth of state; well within memory budget and protects
/// against unbounded growth if a small set of nodes churn wildly.
pub const MAX_FOREIGN_WISHLIST_ENTRIES: usize = 10_000;

/// Maximum age (ms) for a foreign wishlist entry before it's ignored by
/// the scoring pass. Two hours covers ~240× the 30s republish cadence —
/// plenty of room for missed broadcasts under network churn, while
/// keeping the boost responsive to opt-out / pool dissolution.
pub const FOREIGN_WISHLIST_MAX_AGE_MS: u64 = 2 * 60 * 60 * 1000;

/// Per-shard download backoff state. See `ModelMgmt::shard_download_backoff`.
#[derive(Debug, Clone)]
pub struct ShardDownloadBackoff {
    /// Number of consecutive failed download attempts for this shard.
    pub fail_count: u32,
    /// The earliest instant the shard becomes eligible for re-download.
    pub retry_after: std::time::Instant,
}

/// First-failure cooldown, doubled on each subsequent consecutive failure.
pub const SHARD_BACKOFF_BASE_SECS: u64 = 30;
/// Ceiling on the exponential backoff — a chronically-broken shard is retried
/// at most once per this window rather than every auto-manage tick.
pub const SHARD_BACKOFF_MAX_SECS: u64 = 300;
/// Once an entry has been *expired* (past `retry_after`) for longer than this,
/// forget it entirely. Keeps the escalating `fail_count` alive across an active
/// failing window while bounding the map for shards that later succeed or stop
/// being candidates (e.g. a deleted model).
pub const SHARD_BACKOFF_FORGET_SECS: u64 = 3600;

/// Exponential backoff schedule: 30, 60, 120, 240, 300, 300… (capped).
/// `fail_count` is 1-based (the first failure yields the base delay).
pub fn shard_backoff_delay_secs(fail_count: u32) -> u64 {
    if fail_count == 0 {
        return SHARD_BACKOFF_BASE_SECS;
    }
    // Saturating shift so a large fail_count can't overflow; cap at MAX.
    let shifted = SHARD_BACKOFF_BASE_SECS.checked_shl(fail_count - 1);
    shifted
        .unwrap_or(SHARD_BACKOFF_MAX_SECS)
        .min(SHARD_BACKOFF_MAX_SECS)
}

/// Is this per-shard progress state one where a fetch is still outstanding?
///
/// The single expression of it: `is_shard_in_progress` asks it of one shard and
/// `model_has_shard_in_flight` of every shard of a model, and the two deciding
/// differently is how a model's progress entry gets deleted while a shard it
/// still describes is downloading.
pub(crate) fn shard_state_is_in_flight(state: &crate::model::acquisition::ShardState) -> bool {
    matches!(
        state,
        crate::model::acquisition::ShardState::Downloading
            | crate::model::acquisition::ShardState::Pending
            | crate::model::acquisition::ShardState::Verifying
    )
}

/// One in-flight P2P shard transfer's parked state.
///
/// A P2P transfer is a chain of request/response hops across the network event
/// loop rather than a loop on a stack, so everything it holds for its lifetime
/// is parked here and released together when the entry is removed — on
/// completion, on give-up, on cancel, or by the stall watchdog. Adding
/// something a transfer owns means adding a field here, not another map with
/// its own release sites to keep in step.
pub struct P2pDownloadSlot {
    /// Bounds `max_concurrent_downloads`. Released by dropping this slot.
    pub _permit: tokio::sync::OwnedSemaphorePermit,
    /// Exclusive right to write this shard's `.tmp`. Released the same way.
    pub _claim: ShardDownloadClaim,
    /// The cancel flag this transfer STARTED under, held rather than looked up.
    ///
    /// `download_cancel_flags` is keyed by model and its entry is replaced when
    /// a download starts after a cancel (`live_cancel_flag` refuses to hand out
    /// a flag that is already set). A transfer that consulted the map could
    /// therefore be shown a newer, unset flag belonging to a different download
    /// and sail straight through the cancel that was meant for it.
    pub cancel: Arc<AtomicBool>,
    /// When this transfer last made progress — refreshed on every chunk that
    /// arrives, read by `sweep_stalled_p2p_permits`.
    ///
    /// **It measures last progress, not start, and the distinction is the
    /// whole point.** The sweep's own comment describes "180s of no progress",
    /// but the field it read was stamped once when the transfer began and never
    /// touched again, so the sweep fired on any transfer whose TOTAL wall clock
    /// passed 180 s however healthy it was. A 512 MB shard is 64 chunks of
    /// `SHARD_CHUNK_SIZE` (8 MiB), so any link under roughly 2.8 MB/s — 23
    /// Mbit/s — had every P2P shard transfer declared stalled and handed to
    /// HuggingFace while it was progressing perfectly. The report that began
    /// this round was from a 10-15 Mbit/s connection, squarely inside that.
    pub last_progress_at: std::time::Instant,
}

impl P2pDownloadSlot {
    /// Has the download this transfer belongs to been cancelled?
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Exclusive right to write one shard's `.tmp` file, released on drop.
///
/// Obtained from `ModelMgmt::claim_shard_download`. Hold it for as long as the
/// download task runs: move it into the spawned task (HuggingFace), or park it
/// beside the semaphore permit in `p2p_download_permits` (P2P), so it is
/// released by the same code that releases the permit. Dropping it is the ONLY
/// way to release the claim — there is no explicit `release`, so an early
/// return, an error, a panic or an aborted task cannot leave one behind.
pub struct ShardDownloadClaim {
    claims: Arc<dashmap::DashSet<crate::types::ShardId>>,
    shard: crate::types::ShardId,
}

impl ShardDownloadClaim {
    /// The shard this claim covers.
    pub fn shard(&self) -> &crate::types::ShardId {
        &self.shard
    }
}

impl Drop for ShardDownloadClaim {
    fn drop(&mut self) {
        self.claims.remove(&self.shard);
    }
}

impl std::fmt::Debug for ShardDownloadClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardDownloadClaim")
            .field("shard", &self.shard)
            .finish()
    }
}

impl ModelMgmt {
    /// Take the exclusive right to write this shard's `.tmp`, or `None` if a
    /// download of it is already running.
    ///
    /// This is a check and a claim in ONE atomic step (`DashSet::insert`
    /// reports whether the entry is new), which is the point: the callers that
    /// used to ask `is_shard_in_progress` and then spawn had a window between
    /// the two, and the map they asked could be cleared inside it.
    pub fn claim_shard_download(
        &self,
        shard: &crate::types::ShardId,
    ) -> Option<ShardDownloadClaim> {
        if self.shard_download_claims.insert(shard.clone()) {
            Some(ShardDownloadClaim {
                claims: self.shard_download_claims.clone(),
                shard: shard.clone(),
            })
        } else {
            None
        }
    }

    /// A chunk arrived for this transfer, so it is not stalled.
    ///
    /// Called from the chunk-receive path. Without it `last_progress_at` is a
    /// start time wearing the name of a progress time, and the stall sweep
    /// measures the wrong quantity.
    pub fn note_p2p_transfer_progress(&self, shard: &crate::types::ShardId) {
        if let Some(mut slot) = self.p2p_download_permits.get_mut(shard) {
            slot.last_progress_at = std::time::Instant::now();
        }
    }

    /// Does a P2P transfer of this shard still own the file it would write?
    ///
    /// A transfer owns the file while its slot is parked in
    /// `p2p_download_permits` — the slot holds the writer claim. If the claim
    /// is held and the slot is gone, the claim has passed to somebody else
    /// (in practice the HuggingFace fallback, after the stall sweep released
    /// this transfer's slot) and the transfer is over.
    ///
    /// An unclaimed shard answers `true`: the P2P paths that take no claim at
    /// all (a user-initiated acquisition) are left exactly as they were.
    pub fn p2p_transfer_still_owns_the_file(&self, shard: &crate::types::ShardId) -> bool {
        !self.shard_download_claimed(shard) || self.p2p_download_permits.contains_key(shard)
    }

    /// Whether a download task is writing this shard's `.tmp` right now.
    /// Prefer `is_shard_in_progress`, which also covers work that is queued
    /// but has not reached a writer yet.
    pub fn shard_download_claimed(&self, shard: &crate::types::ShardId) -> bool {
        self.shard_download_claims.contains(shard)
    }

    /// Whether a download task for ANY shard of this model is running right now.
    ///
    /// Asked before removing the model's `acquisition_progress` entry, because
    /// that entry is what a live download's progress is written into and what
    /// `is_shard_in_progress` reads.
    ///
    /// Deliberately reads ONLY the claims, not the per-shard progress marks.
    /// A claim is exact — it exists for exactly as long as a writer does,
    /// because only a `Drop` can release it. A progress mark is advisory: a
    /// path that gives up without clearing its own mark leaves a `Downloading`
    /// behind, and reading those here would make such a leak permanent (the
    /// entry could never be collected, so the shard could never be fetched
    /// again). `is_shard_in_progress` still reads both, because refusing to
    /// start a second download on stale evidence is the safe direction and
    /// deleting state is not.
    pub fn model_has_live_shard_download(&self, model_id: &crate::types::ModelId) -> bool {
        self.shard_download_claims
            .iter()
            .any(|s| &s.model_id == model_id)
    }

    /// Check if a shard is currently being downloaded, pending, or verifying.
    /// Prevents races where multiple subsystems try to download the same shard.
    ///
    /// Answers from the live writer claim as well as the progress map, so the
    /// answer survives the progress entry being cleaned up under a download
    /// that is still running (see `shard_download_claims`).
    pub fn is_shard_in_progress(&self, model_id: &crate::types::ModelId, shard_index: u32) -> bool {
        if self.shard_download_claimed(&crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_index,
        }) {
            return true;
        }
        self.shard_marked_in_progress(model_id, shard_index)
    }

    /// Whether the progress map marks this shard as outstanding — queued,
    /// downloading or verifying — ignoring the writer claims.
    ///
    /// Use it where you already HOLD the claim for this shard and so cannot ask
    /// `is_shard_in_progress` without seeing yourself; everywhere else wants
    /// `is_shard_in_progress`, which covers a live writer as well as the map.
    pub fn shard_marked_in_progress(
        &self,
        model_id: &crate::types::ModelId,
        shard_index: u32,
    ) -> bool {
        self.acquisition_progress
            .get(model_id)
            .map(|entry| {
                entry
                    .shard_progress
                    .get(&shard_index)
                    .map(|sp| shard_state_is_in_flight(&sp.state))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Record a failed download attempt for a shard, growing its cooldown
    /// exponentially. Returns `(fail_count, delay_secs)` for the caller to log.
    /// Called from every terminal, transient download-failure site (HF error,
    /// GGUF-probe failure, P2P give-up without an HF fallback, stall
    /// reconciliation).
    pub fn record_shard_download_failure(&self, shard_id: &crate::types::ShardId) -> (u32, u64) {
        let mut entry = self
            .shard_download_backoff
            .entry(shard_id.clone())
            .or_insert(ShardDownloadBackoff {
                fail_count: 0,
                retry_after: std::time::Instant::now(),
            });
        entry.fail_count = entry.fail_count.saturating_add(1);
        let delay = shard_backoff_delay_secs(entry.fail_count);
        entry.retry_after = std::time::Instant::now() + std::time::Duration::from_secs(delay);
        (entry.fail_count, delay)
    }

    /// Whether a shard is currently in download backoff (still cooling down).
    /// Opportunistically forgets an entry that expired long ago
    /// (`SHARD_BACKOFF_FORGET_SECS`) so the map stays bounded — while keeping
    /// the escalating `fail_count` for a shard that is still actively failing.
    pub fn shard_in_backoff(&self, shard_id: &crate::types::ShardId) -> bool {
        let now = std::time::Instant::now();
        if let Some(entry) = self.shard_download_backoff.get(shard_id) {
            if entry.retry_after > now {
                return true;
            }
            // Expired. Forget it only once it's been expired for a good while,
            // so a shard that keeps failing across successive ticks still
            // escalates rather than resetting to the base delay each time.
            let expired_for = now.duration_since(entry.retry_after);
            drop(entry);
            if expired_for >= std::time::Duration::from_secs(SHARD_BACKOFF_FORGET_SECS) {
                self.shard_download_backoff.remove(shard_id);
            }
        }
        false
    }

    /// Clear a shard's download backoff after a successful completion.
    pub fn clear_shard_download_backoff(&self, shard_id: &crate::types::ShardId) {
        self.shard_download_backoff.remove(shard_id);
    }

    /// Mutate a model's AcquisitionStatus if present. No-op if the model has
    /// no acquisition entry. Locks `acquisition_progress` only for the body
    /// of the closure — do NOT hold the closure across `.await`.
    pub fn update_acquisition<F>(&self, model_id: &crate::types::ModelId, f: F)
    where
        F: FnOnce(&mut crate::model::acquisition::AcquisitionStatus),
    {
        if let Some(mut entry) = self.acquisition_progress.get_mut(model_id) {
            f(&mut entry);
        }
    }

    /// Mark an acquisition as failed — sets state, increments failed_shards,
    /// and pushes a log line. Safe to call if the model has no acquisition
    /// entry (no-op).
    pub fn set_acquisition_failed(
        &self,
        model_id: &crate::types::ModelId,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        self.update_acquisition(model_id, |s| {
            s.state = crate::model::acquisition::AcquisitionState::Failed {
                reason: reason.clone(),
            };
            s.failed_shards += 1;
            s.log_push(format!("Failed: {reason}"));
        });
    }

    /// Mark an acquisition as complete for single-file downloads (e.g., full
    /// GGUF). Sets state + treats the single file as 1 downloaded and verified
    /// shard. For multi-shard downloads, use `update_acquisition` directly.
    pub fn set_acquisition_complete_single(
        &self,
        model_id: &crate::types::ModelId,
        log_msg: impl Into<String>,
    ) {
        let msg = log_msg.into();
        self.update_acquisition(model_id, |s| {
            s.state = crate::model::acquisition::AcquisitionState::Complete;
            s.downloaded_shards = 1;
            s.verified_shards = 1;
            s.log_push(msg);
        });
    }

    /// Register a new download job: insert the initial AcquisitionStatus and
    /// a cancel flag atomically from the caller's perspective, so subsystems
    /// that observe one but not the other (auto-manage scan vs hf download)
    /// don't race. Returns the cancel flag Arc.
    /// Register a download, returning its cancel flag and **the requested
    /// shards that were already being fetched**.
    ///
    /// The per-shard marks in `acquisition_progress` are how a second fetch of
    /// one shard is avoided — auto-manage skips any shard marked `Downloading`.
    /// This used to `insert` the new status wholesale, which erased those marks
    /// for every shard the new request did not mention, and told the caller
    /// nothing about the ones it did. So asking for a shard that was already on
    /// its way started a SECOND download of it: two writers on one `.tmp`, one
    /// of them finishing and registering the shard while the other kept going
    /// and then reported `size mismatch: expected N bytes but wrote 0 bytes`.
    /// Reported from the field 2026-09-11.
    ///
    /// Existing marks are therefore carried over, and the shards already in
    /// flight are handed back so the caller can leave them alone.
    pub fn begin_download(
        &self,
        model_id: crate::types::ModelId,
        mut status: crate::model::acquisition::AcquisitionStatus,
    ) -> (Arc<AtomicBool>, Vec<u32>) {
        use crate::model::acquisition::{AcquisitionState, ShardState};
        let mut already_in_flight = Vec::new();
        if let Some(existing) = self.acquisition_progress.get(&model_id) {
            if matches!(existing.state, AcquisitionState::Downloading) {
                for (&index, progress) in &existing.shard_progress {
                    if matches!(progress.state, ShardState::Downloading) {
                        if status.shard_progress.contains_key(&index) {
                            already_in_flight.push(index);
                        }
                        // The live download owns this shard's progress.
                        status.shard_progress.insert(index, progress.clone());
                    } else {
                        status
                            .shard_progress
                            .entry(index)
                            .or_insert_with(|| progress.clone());
                    }
                }
            }
        }
        already_in_flight.sort_unstable();
        let flag = self.live_cancel_flag(&model_id);
        self.acquisition_progress.insert(model_id.clone(), status);
        (flag, already_in_flight)
    }

    /// The cancel flag every download of this model watches, creating it if
    /// this is the first.
    ///
    /// One flag per model, because one press of Cancel is asking for everything
    /// being fetched for that model to stop. Replacing it orphaned the running
    /// download's — it keeps its own `Arc` and carries on, so the map no longer
    /// points at what is running and a later cancel reached only the newest
    /// registration.
    ///
    /// **A flag that is already SET is not reused**: it belongs to a cancel in
    /// progress, and handing it to a fresh download would cancel that download
    /// the instant it started.
    ///
    /// Every path that starts a download must take its flag from here. The
    /// auto-manage path did not take one at all, so it passed `None` to
    /// `download_shard` and there was nothing in the map for `cancel_download`
    /// to set: pressing Cancel on an auto-managed download reported success and
    /// stopped nothing, while the bytes kept arriving.
    pub fn live_cancel_flag(&self, model_id: &crate::types::ModelId) -> Arc<AtomicBool> {
        if let Some(existing) = self
            .download_cancel_flags
            .get(model_id)
            .map(|f| f.value().clone())
            .filter(|f| !f.load(std::sync::atomic::Ordering::Acquire))
        {
            return existing;
        }
        let fresh = Arc::new(AtomicBool::new(false));
        self.download_cancel_flags
            .insert(model_id.clone(), fresh.clone());
        fresh
    }

    /// Item 8 Phase 1: replace this peer's known set of prefix-cache block
    /// hashes for `model_id` with `new_blocks`. Drops any previously-recorded
    /// blocks for the same `(peer, model)` pair, preserving entries from
    /// other peers in the per-block holder set. A peer announcing an empty
    /// set is treated as "I no longer hold any blocks for this model".
    ///
    /// SEC: drops any input list larger than `MAX_BLOCKS_PER_PEER_MODEL`.
    /// Without this cap a single misbehaving peer can announce millions of
    /// distinct block hashes per model and exhaust memory in the per-block
    /// holder sets. The cap matches the worst-case prefix-cache snapshot
    /// count for typical 8K-context workloads (256-token blocks → ~32 blocks
    /// per request × a few hundred concurrent sessions).
    ///
    /// Returns `(added, removed)` counts for logging — strictly diagnostic.
    pub fn replace_peer_prefix_blocks(
        &self,
        peer: NodeId,
        model_id: crate::types::ModelId,
        mut new_blocks: Vec<[u8; 32]>,
    ) -> (usize, usize) {
        const MAX_BLOCKS_PER_PEER_MODEL: usize = 16_384;
        if new_blocks.len() > MAX_BLOCKS_PER_PEER_MODEL {
            tracing::warn!(
                %peer,
                model = %model_id,
                announced = new_blocks.len(),
                cap = MAX_BLOCKS_PER_PEER_MODEL,
                "Truncating prefix-block announce: exceeds per-peer-per-model cap"
            );
            new_blocks.truncate(MAX_BLOCKS_PER_PEER_MODEL);
        }
        // Snapshot the previous block set for this (peer, model) pair, then
        // compute the diff so we only touch the per-block holder sets that
        // actually changed.
        let new_set: std::collections::HashSet<[u8; 32]> = new_blocks.iter().copied().collect();
        let peer_models = self.peer_prefix_blocks.entry(peer.clone()).or_default();
        let prev: Vec<[u8; 32]> = peer_models
            .get(&model_id)
            .map(|s| s.iter().map(|r| *r.key()).collect())
            .unwrap_or_default();

        let model_index = self
            .cross_node_prefix_index
            .entry(model_id.clone())
            .or_default();

        let mut removed = 0usize;
        for h in &prev {
            if !new_set.contains(h) {
                if let Some(holders) = model_index.get(h) {
                    holders.remove(&peer);
                    let now_empty = holders.is_empty();
                    drop(holders);
                    if now_empty {
                        model_index.remove(h);
                    }
                }
                removed += 1;
            }
        }

        let prev_set: std::collections::HashSet<[u8; 32]> = prev.iter().copied().collect();
        let mut added = 0usize;
        for h in &new_blocks {
            if !prev_set.contains(h) {
                let holders = model_index.entry(*h).or_default();
                holders.insert(peer.clone());
                added += 1;
            }
        }

        // Refresh the reverse index for this (peer, model) pair.
        let new_per_peer = dashmap::DashSet::new();
        for h in &new_blocks {
            new_per_peer.insert(*h);
        }
        if new_per_peer.is_empty() {
            peer_models.remove(&model_id);
        } else {
            peer_models.insert(model_id, new_per_peer);
        }

        (added, removed)
    }

    /// Drop every entry attributed to `peer` from the cross-node prefix-cache
    /// index. Called when a peer disconnects or is evicted — leaving stale
    /// entries would point Phase 2's KV-fetch path at unreachable peers.
    pub fn forget_peer_prefix_blocks(&self, peer: &NodeId) -> usize {
        let Some((_, models)) = self.peer_prefix_blocks.remove(peer) else {
            return 0;
        };
        let mut removed = 0usize;
        for entry in models.iter() {
            let model_id = entry.key();
            let blocks = entry.value();
            if let Some(model_index) = self.cross_node_prefix_index.get(model_id) {
                for hash in blocks.iter() {
                    if let Some(holders) = model_index.get(hash.key()) {
                        holders.remove(peer);
                        let now_empty = holders.is_empty();
                        drop(holders);
                        if now_empty {
                            model_index.remove(hash.key());
                        }
                        removed += 1;
                    }
                }
            }
        }
        removed
    }

    /// R130: apply an inbound `WishlistAnnouncement` to the foreign wishlist
    /// index. Replaces this publisher's entire slice (drops models the
    /// publisher no longer mentions), then inserts the new entries.
    /// Caller is expected to have already validated authentication,
    /// freshness, and entry-count cap.
    ///
    /// Returns `(added, removed)` counts for logging.
    pub fn apply_wishlist_announcement(
        &self,
        publisher: NodeId,
        entries: &[(crate::types::ModelId, u32)],
        timestamp_ms: u64,
    ) -> (usize, usize) {
        let new_models: std::collections::HashSet<crate::types::ModelId> =
            entries.iter().map(|(m, _)| m.clone()).collect();
        let mut removed = 0usize;
        self.foreign_wishlist.retain(|(p, m), _| {
            if p == &publisher && !new_models.contains(m) {
                removed += 1;
                false
            } else {
                true
            }
        });
        let mut added = 0usize;
        for (model_id, score) in entries {
            if model_id.0.len() > 256 {
                continue;
            }
            let bounded = (*score).min(100);
            let prev = self.foreign_wishlist.insert(
                (publisher.clone(), model_id.clone()),
                (bounded, timestamp_ms),
            );
            if prev.is_none() {
                added += 1;
            }
        }
        (added, removed)
    }

    /// Lookup the set of remote peers that announced holding a KV snapshot
    /// for this `(model_id, block_hash)` pair. Empty when no peer has it.
    /// Phase 2 will use this to decide where to fetch from; Phase 1 only
    /// exposes it for tests + diagnostics.
    #[cfg(test)]
    pub fn cross_node_prefix_holders(
        &self,
        model_id: &crate::types::ModelId,
        block_hash: &[u8; 32],
    ) -> Vec<NodeId> {
        self.cross_node_prefix_index
            .get(model_id)
            .and_then(|m| {
                m.get(block_hash)
                    .map(|s| s.iter().map(|r| r.clone()).collect())
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelId;

    pub(super) fn make_mgmt() -> ModelMgmt {
        ModelMgmt {
            acquisition_progress: DashMap::new(),
            hf_sources: DashMap::new(),
            auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
            auto_manage_enabled: AtomicBool::new(false),
            auto_manage_default_model_cap: AtomicU32::new(0),
            model_auto_manage_policies: DashMap::new(),
            hf_probe_cache: DashMap::new(),
            peer_shard_downloads: DashMap::new(),
            download_cancel_flags: DashMap::new(),
            model_trust: DashMap::new(),
            loading_models: DashMap::new(),
            locked_shards: DashMap::new(),
            removed_by_user: DashMap::new(),
            shard_p2p_failed: dashmap::DashSet::new(),
            shards_needing_repair: dashmap::DashSet::new(),
            shards_pending_verification: dashmap::DashSet::new(),
            disputed_shards: dashmap::DashSet::new(),
            shard_download_backoff: DashMap::new(),
            model_request_counts: DashMap::new(),
            resource_schedule: RwLock::new(Default::default()),
            prune_history: RwLock::new(VecDeque::new()),
            parallax_stability: DashMap::new(),
            cross_node_prefix_index: DashMap::new(),
            peer_prefix_blocks: DashMap::new(),
            p2p_download_permits: DashMap::new(),
            shard_download_claims: Arc::new(dashmap::DashSet::new()),
            wishlist: arc_swap::ArcSwap::from_pointee(
                crate::model::auto_manage::wishlist::Wishlist::default(),
            ),
            hf_trending_cache: arc_swap::ArcSwap::from_pointee(
                crate::model::huggingface::HfTrendingSnapshot::default(),
            ),
            foreign_wishlist: DashMap::new(),
            quant_recommendations: arc_swap::ArcSwap::from_pointee(
                crate::model::auto_manage::quant::QuantRecommendations::default(),
            ),
        }
    }

    fn h(b: u8) -> [u8; 32] {
        [b; 32]
    }

    #[test]
    fn replace_inserts_announced_blocks() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let (added, removed) =
            m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2), h(3)]);
        assert_eq!(added, 3);
        assert_eq!(removed, 0);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(2)), vec![peer]);
    }

    #[test]
    fn replace_supersedes_previous_announcement_from_same_peer() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2), h(3)]);
        // Second announcement drops blocks 2/3 and adds 4.
        let (added, removed) =
            m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(4)]);
        assert_eq!(added, 1);
        assert_eq!(removed, 2);
        assert_eq!(
            m.cross_node_prefix_holders(&model, &h(1)),
            vec![peer.clone()]
        );
        assert!(m.cross_node_prefix_holders(&model, &h(2)).is_empty());
        assert_eq!(m.cross_node_prefix_holders(&model, &h(4)), vec![peer]);
    }

    #[test]
    fn multiple_peers_share_block_holders() {
        let m = make_mgmt();
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(1), h(2)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(2), h(3)]);
        let mut holders = m.cross_node_prefix_holders(&model, &h(2));
        holders.sort_by_key(|n| n.0);
        assert_eq!(holders, vec![p1.clone(), p2.clone()]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(1)), vec![p1]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(3)), vec![p2]);
    }

    #[test]
    fn forget_peer_drops_only_their_entries() {
        let m = make_mgmt();
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(1), h(2)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(2), h(3)]);
        let removed = m.forget_peer_prefix_blocks(&p1);
        assert_eq!(removed, 2);
        assert!(m.cross_node_prefix_holders(&model, &h(1)).is_empty());
        assert_eq!(m.cross_node_prefix_holders(&model, &h(2)), vec![p2.clone()]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(3)), vec![p2]);
    }

    #[test]
    fn empty_announce_drops_all_entries_for_peer_model() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2)]);
        let (added, removed) = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![]);
        assert_eq!(added, 0);
        assert_eq!(removed, 2);
        assert!(m.cross_node_prefix_holders(&model, &h(1)).is_empty());
        assert!(m.cross_node_prefix_holders(&model, &h(2)).is_empty());
    }

    /// Item 8 Phase 4: simulates the core resolver logic inside
    /// `spawn_prefix_probe_handler` against a scenario with three peers
    /// holding progressively longer prefix matches. Validates that the
    /// longest-prefix hit wins AND that a trust-gate filter correctly
    /// excludes low-trust peers even when they hold a longer match.
    /// We mirror the probe-handler's inline logic here rather than
    /// constructing a SharedState (which needs a full runtime harness).
    #[test]
    fn probe_resolver_picks_longest_prefix_above_trust_floor() {
        use crate::types::PrefixBlockEntry;
        let m = make_mgmt();
        let model = ModelId("m".into());
        let high_trust_peer = NodeId([1u8; 32]);
        let med_trust_peer = NodeId([2u8; 32]);
        let low_trust_peer = NodeId([3u8; 32]);

        // Manifest over a prompt with 3 blocks. The low-trust peer has
        // ALL three blocks (longest match); the medium-trust peer has the
        // first two; the high-trust peer only the first.
        let blocks = [
            PrefixBlockEntry {
                block_hash: h(10),
                token_count: 64,
            },
            PrefixBlockEntry {
                block_hash: h(20),
                token_count: 128,
            },
            PrefixBlockEntry {
                block_hash: h(30),
                token_count: 192,
            },
        ];
        let _ = m.replace_peer_prefix_blocks(high_trust_peer.clone(), model.clone(), vec![h(10)]);
        let _ =
            m.replace_peer_prefix_blocks(med_trust_peer.clone(), model.clone(), vec![h(10), h(20)]);
        let _ = m.replace_peer_prefix_blocks(
            low_trust_peer.clone(),
            model.clone(),
            vec![h(10), h(20), h(30)],
        );

        // Trust scores: low-trust peer is below threshold 0.4.
        let trust = |peer: &NodeId| -> f32 {
            if peer == &high_trust_peer {
                0.9
            } else if peer == &med_trust_peer {
                0.6
            } else {
                0.2
            }
        };
        let trust_min: f32 = 0.4;

        // Resolver mirror: walk manifest longest-first, pick a peer above
        // the trust floor.
        let mut best: Option<(NodeId, [u8; 32], u32)> = None;
        if let Some(model_index) = m.cross_node_prefix_index.get(&model) {
            for entry in blocks.iter().rev() {
                if let Some(holders) = model_index.get(&entry.block_hash) {
                    let candidates: Vec<NodeId> = holders
                        .iter()
                        .map(|r| r.clone())
                        .filter(|n| trust(n) >= trust_min)
                        .collect();
                    if !candidates.is_empty() {
                        // Sort by NodeId for determinism — the actual
                        // resolver uses latency EMA as tiebreak, absent
                        // here, so pick first sorted.
                        let mut c = candidates;
                        c.sort_by_key(|n| n.0);
                        best = Some((c[0].clone(), entry.block_hash, entry.token_count));
                        break;
                    }
                }
            }
        }
        // Low-trust peer holds the longest match (h(30) at 192 tokens)
        // but is below the floor — so we should fall back to h(20) at
        // 128 tokens, served by the medium-trust peer.
        let (peer, hash, token_count) = best.expect("should find a match");
        assert_eq!(peer, med_trust_peer);
        assert_eq!(hash, h(20));
        assert_eq!(token_count, 128);
    }

    /// R130: applying a WishlistAnnouncement replaces the publisher's
    /// existing slice — models the publisher no longer mentions should
    /// be dropped, new models inserted, and scores updated for shared
    /// keys. Other publishers' entries must be left alone.
    #[test]
    fn wishlist_announce_replaces_publisher_slice() {
        let m = make_mgmt();
        let alice = NodeId([1u8; 32]);
        let bob = NodeId([2u8; 32]);
        let a = ModelId("alpha".into());
        let b = ModelId("beta".into());
        let g = ModelId("gamma".into());

        // Seed: alice wants alpha+beta; bob wants alpha.
        m.apply_wishlist_announcement(alice.clone(), &[(a.clone(), 80), (b.clone(), 60)], 1000);
        m.apply_wishlist_announcement(bob.clone(), &[(a.clone(), 50)], 1000);
        assert_eq!(m.foreign_wishlist.len(), 3);

        // Alice's next announcement drops beta and adds gamma; alpha
        // score updated. Bob's entry must survive untouched.
        let (added, removed) =
            m.apply_wishlist_announcement(alice.clone(), &[(a.clone(), 95), (g.clone(), 40)], 2000);
        assert_eq!(added, 1, "gamma is new");
        assert_eq!(removed, 1, "beta was dropped");
        assert_eq!(
            m.foreign_wishlist
                .get(&(alice.clone(), a.clone()))
                .map(|v| *v),
            Some((95, 2000))
        );
        assert!(m.foreign_wishlist.get(&(alice.clone(), b)).is_none());
        assert_eq!(
            m.foreign_wishlist
                .get(&(alice.clone(), g.clone()))
                .map(|v| *v),
            Some((40, 2000))
        );
        assert_eq!(
            m.foreign_wishlist.get(&(bob, a)).map(|v| *v),
            Some((50, 1000)),
            "bob's entry should not be touched"
        );
    }

    /// R130: scores >100 are clamped on insert. Long model_ids are
    /// dropped silently (the caller is expected to have already enforced
    /// a hard wire-side cap, but the helper defends in depth).
    #[test]
    fn wishlist_announce_clamps_scores_and_drops_oversized_ids() {
        let m = make_mgmt();
        let p = NodeId([7u8; 32]);
        let normal = ModelId("normal".into());
        let oversized = ModelId("x".repeat(257));
        m.apply_wishlist_announcement(p.clone(), &[(normal.clone(), 250), (oversized, 50)], 1000);
        let entry = m.foreign_wishlist.get(&(p, normal)).map(|v| *v);
        assert_eq!(entry, Some((100, 1000)));
        assert_eq!(m.foreign_wishlist.len(), 1);
    }

    /// R130: empty announcement from a known publisher drops all of
    /// their entries — modelling "I no longer want any of these".
    #[test]
    fn wishlist_announce_empty_clears_publisher_slice() {
        let m = make_mgmt();
        let p = NodeId([3u8; 32]);
        let a = ModelId("alpha".into());
        let b = ModelId("beta".into());
        m.apply_wishlist_announcement(p.clone(), &[(a, 50), (b, 30)], 1000);
        assert_eq!(m.foreign_wishlist.len(), 2);
        let (added, removed) = m.apply_wishlist_announcement(p, &[], 2000);
        assert_eq!(added, 0);
        assert_eq!(removed, 2);
        assert_eq!(m.foreign_wishlist.len(), 0);
    }

    /// Phase 4: when ALL candidate peers are below the trust threshold,
    /// the resolver must return no match — the fetcher falls through to
    /// a full local prefill instead of risking a poisoned KV.
    #[test]
    fn probe_resolver_returns_none_when_all_peers_below_trust_floor() {
        use crate::types::PrefixBlockEntry;
        let m = make_mgmt();
        let model = ModelId("m".into());
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(10)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(10)]);
        let blocks = [PrefixBlockEntry {
            block_hash: h(10),
            token_count: 64,
        }];
        let trust = |_: &NodeId| -> f32 { 0.1 };
        let trust_min: f32 = 0.5;

        let mut best: Option<(NodeId, [u8; 32], u32)> = None;
        if let Some(model_index) = m.cross_node_prefix_index.get(&model) {
            for entry in blocks.iter().rev() {
                if let Some(holders) = model_index.get(&entry.block_hash) {
                    let candidates: Vec<NodeId> = holders
                        .iter()
                        .map(|r| r.clone())
                        .filter(|n| trust(n) >= trust_min)
                        .collect();
                    if !candidates.is_empty() {
                        let mut c = candidates;
                        c.sort_by_key(|n| n.0);
                        best = Some((c[0].clone(), entry.block_hash, entry.token_count));
                        break;
                    }
                }
            }
        }
        assert!(best.is_none());
    }

    // --- Shard download backoff (external report, 2026-07-23) ---

    fn sid(model: &str, index: u32) -> crate::types::ShardId {
        crate::types::ShardId {
            model_id: ModelId(model.into()),
            index,
        }
    }

    #[test]
    fn backoff_delay_schedule_doubles_and_caps() {
        // 1-based fail_count: 30, 60, 120, 240, then capped at 300.
        assert_eq!(shard_backoff_delay_secs(1), 30);
        assert_eq!(shard_backoff_delay_secs(2), 60);
        assert_eq!(shard_backoff_delay_secs(3), 120);
        assert_eq!(shard_backoff_delay_secs(4), 240);
        assert_eq!(shard_backoff_delay_secs(5), 300);
        assert_eq!(shard_backoff_delay_secs(6), 300);
        // A pathologically large count can't overflow the shift.
        assert_eq!(shard_backoff_delay_secs(1000), 300);
    }

    #[test]
    fn record_failure_escalates_and_marks_in_backoff() {
        let m = make_mgmt();
        let s = sid("m1", 6);

        assert!(!m.shard_in_backoff(&s), "clean shard is not in backoff");

        let (fails1, delay1) = m.record_shard_download_failure(&s);
        assert_eq!(fails1, 1);
        assert_eq!(delay1, 30);
        assert!(m.shard_in_backoff(&s), "in backoff right after a failure");

        let (fails2, delay2) = m.record_shard_download_failure(&s);
        assert_eq!(fails2, 2);
        assert_eq!(delay2, 60);

        let (fails3, delay3) = m.record_shard_download_failure(&s);
        assert_eq!(fails3, 3);
        assert_eq!(delay3, 120);
    }

    #[test]
    fn clear_backoff_removes_the_entry() {
        let m = make_mgmt();
        let s = sid("m1", 0);
        m.record_shard_download_failure(&s);
        assert!(m.shard_in_backoff(&s));
        m.clear_shard_download_backoff(&s);
        assert!(!m.shard_in_backoff(&s));
        assert!(m.shard_download_backoff.get(&s).is_none());
    }

    #[test]
    fn backoff_is_per_shard() {
        let m = make_mgmt();
        let a = sid("m1", 1);
        let b = sid("m1", 2);
        m.record_shard_download_failure(&a);
        assert!(m.shard_in_backoff(&a));
        assert!(!m.shard_in_backoff(&b), "sibling shard is unaffected");
    }

    #[test]
    fn expired_entry_is_kept_until_forget_window() {
        let m = make_mgmt();
        let s = sid("m1", 3);
        // Simulate an entry that expired 10s ago (well under the forget window):
        // shard_in_backoff must return false but retain the fail_count so the
        // next failure escalates rather than resetting to the base delay.
        m.shard_download_backoff.insert(
            s.clone(),
            ShardDownloadBackoff {
                fail_count: 2,
                retry_after: std::time::Instant::now() - std::time::Duration::from_secs(10),
            },
        );
        assert!(!m.shard_in_backoff(&s), "expired entry is not blocking");
        assert!(
            m.shard_download_backoff.get(&s).is_some(),
            "entry retained within the forget window"
        );
        let (fails, delay) = m.record_shard_download_failure(&s);
        assert_eq!(fails, 3, "escalates from the retained count");
        assert_eq!(delay, 120);
    }

    #[test]
    fn long_expired_entry_is_forgotten_on_read() {
        let m = make_mgmt();
        let s = sid("m1", 4);
        m.shard_download_backoff.insert(
            s.clone(),
            ShardDownloadBackoff {
                fail_count: 3,
                retry_after: std::time::Instant::now()
                    - std::time::Duration::from_secs(SHARD_BACKOFF_FORGET_SECS + 60),
            },
        );
        assert!(!m.shard_in_backoff(&s));
        assert!(
            m.shard_download_backoff.get(&s).is_none(),
            "entry idle past the forget window self-evicts on read"
        );
    }

    /// Registering a download must not erase the in-flight marks of shards it
    /// is not asking for, and must say which of the ones it IS asking for are
    /// already on their way.
    ///
    /// The reported failure: a shard already downloading was requested again,
    /// a second fetch started, and the two wrote the same `.tmp` — one
    /// finished and registered the shard while the other carried on and then
    /// reported `size mismatch: expected N bytes but wrote 0 bytes`.
    #[test]
    fn a_second_request_for_a_downloading_shard_is_reported_not_restarted() {
        use crate::model::acquisition::{AcquisitionStatus, ShardProgress};
        let mgmt = make_mgmt();
        let mid = ModelId("m".into());

        let mut first = AcquisitionStatus::new_downloading(
            mid.clone(),
            2,
            0,
            "huggingface",
            "auto",
            "first".to_string(),
        );
        first
            .shard_progress
            .insert(0, ShardProgress::new_downloading(0, 0));
        first
            .shard_progress
            .insert(1, ShardProgress::new_downloading(1, 0));
        let (_flag, already) = mgmt.begin_download(mid.clone(), first);
        assert!(already.is_empty(), "nothing was in flight yet");

        // A user asks for shard 0 — already going — while shard 1 is untouched.
        let mut second = AcquisitionStatus::new_downloading(
            mid.clone(),
            1,
            0,
            "huggingface",
            "user",
            "second".to_string(),
        );
        second
            .shard_progress
            .insert(0, ShardProgress::new_downloading(0, 0));
        let (_flag, already) = mgmt.begin_download(mid.clone(), second);
        assert_eq!(already, vec![0], "shard 0 was already being fetched");

        let entry = mgmt
            .acquisition_progress
            .get(&mid)
            .expect("the model keeps an entry");
        assert!(
            entry.shard_progress.contains_key(&1),
            "shard 1 was not mentioned by the second request and must keep its \
             in-flight mark — erasing it is what let a duplicate start"
        );
        assert_eq!(entry.shard_progress.len(), 2);
    }

    /// Registering a second download must not orphan the first one's cancel
    /// flag: the running download keeps its own `Arc` and carries on, so a
    /// cancel that reaches only the newest registration cannot stop it.
    #[test]
    fn a_second_registration_shares_the_running_downloads_cancel_flag() {
        use crate::model::acquisition::AcquisitionStatus;
        let mgmt = make_mgmt();
        let mid = ModelId("m".into());
        let status = || {
            AcquisitionStatus::new_downloading(
                mid.clone(),
                1,
                0,
                "huggingface",
                "user",
                "x".to_string(),
            )
        };

        let (first, _) = mgmt.begin_download(mid.clone(), status());
        let (second, _) = mgmt.begin_download(mid.clone(), status());
        assert!(
            Arc::ptr_eq(&first, &second),
            "both downloads must watch one flag, or one of them cannot be cancelled"
        );

        // Cancelling reaches the download that registered first.
        second.store(true, std::sync::atomic::Ordering::Release);
        assert!(first.load(std::sync::atomic::Ordering::Acquire));

        // A flag already cancelled is NOT handed to a fresh download — that
        // would cancel it before it began.
        let (third, _) = mgmt.begin_download(mid.clone(), status());
        assert!(!third.load(std::sync::atomic::Ordering::Acquire));
        assert!(!Arc::ptr_eq(&third, &first));
    }

    /// A model with no download in progress registers exactly as before.
    #[test]
    fn registering_a_fresh_download_reports_nothing_in_flight() {
        use crate::model::acquisition::{AcquisitionStatus, ShardProgress};
        let mgmt = make_mgmt();
        let mid = ModelId("m".into());
        let mut status = AcquisitionStatus::new_downloading(
            mid.clone(),
            1,
            0,
            "huggingface",
            "user",
            "only".to_string(),
        );
        status
            .shard_progress
            .insert(3, ShardProgress::new_downloading(3, 0));
        let (_flag, already) = mgmt.begin_download(mid.clone(), status);
        assert!(already.is_empty());
        assert_eq!(
            mgmt.acquisition_progress
                .get(&mid)
                .map(|e| e.shard_progress.len()),
            Some(1)
        );
    }
}

#[cfg(test)]
mod shard_download_claim_tests {
    use super::*;
    use crate::model::acquisition::{
        AcquisitionState, AcquisitionStatus, ShardProgress, ShardState,
    };
    use crate::types::{ModelId, ShardId};

    fn mgmt() -> ModelMgmt {
        super::tests::make_mgmt()
    }

    fn shard(model: &str, index: u32) -> ShardId {
        ShardId {
            model_id: ModelId(model.to_string()),
            index,
        }
    }

    /// Every path that starts a download must find the SAME flag, or pressing
    /// Cancel reaches some downloads and not others. Auto-manage took none at
    /// all: `cancel_download` found nothing in the map, set nothing, and
    /// reported success while the bytes kept arriving.
    #[test]
    fn every_download_of_a_model_watches_one_cancel_flag() {
        let m = mgmt();
        let mid = ModelId("glm-4-9b".to_string());

        let first = m.live_cancel_flag(&mid);
        let second = m.live_cancel_flag(&mid);
        assert!(
            Arc::ptr_eq(&first, &second),
            "one press of Cancel is asking for everything on this model to stop"
        );

        // What the cancel endpoint does.
        m.live_cancel_flag(&mid)
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(
            first.load(std::sync::atomic::Ordering::Acquire),
            "the flag a running download is watching is the one Cancel sets"
        );
    }

    /// A flag that is already set belongs to a cancel in progress. Handing it
    /// to a fresh download would cancel that download the instant it started.
    #[test]
    fn a_download_starting_after_a_cancel_gets_a_clean_flag() {
        let m = mgmt();
        let mid = ModelId("glm-4-9b".to_string());
        let cancelled = m.live_cancel_flag(&mid);
        cancelled.store(true, std::sync::atomic::Ordering::Release);

        let fresh = m.live_cancel_flag(&mid);
        assert!(!fresh.load(std::sync::atomic::Ordering::Acquire));
        assert!(!Arc::ptr_eq(&cancelled, &fresh));
    }

    #[test]
    fn cancel_flags_are_per_model() {
        let m = mgmt();
        let a = m.live_cancel_flag(&ModelId("glm-4-9b".to_string()));
        a.store(true, std::sync::atomic::Ordering::Release);
        assert!(
            !m.live_cancel_flag(&ModelId("qwen3-1.7b".to_string()))
                .load(std::sync::atomic::Ordering::Acquire),
            "cancelling one model must not stop another model's download"
        );
    }

    #[test]
    fn a_claim_is_exclusive_and_is_released_only_by_dropping_it() {
        let m = mgmt();
        let sid = shard("glm-4-9b", 3);

        let first = m.claim_shard_download(&sid).expect("first claim is free");
        assert!(
            m.claim_shard_download(&sid).is_none(),
            "a second writer must not be handed the same shard's .tmp"
        );
        assert!(m.shard_download_claimed(&sid));

        drop(first);
        assert!(
            !m.shard_download_claimed(&sid),
            "dropping the claim is what releases it — there is no release call"
        );
        assert!(
            m.claim_shard_download(&sid).is_some(),
            "the shard can be fetched again once nothing is writing it"
        );
    }

    fn park_slot(m: &ModelMgmt, sid: &ShardId, claim: ShardDownloadClaim) {
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        m.p2p_download_permits.insert(
            sid.clone(),
            crate::daemon::state::P2pDownloadSlot {
                _permit: sem.try_acquire_owned().unwrap(),
                _claim: claim,
                cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                last_progress_at: std::time::Instant::now(),
            },
        );
    }

    /// Confirmed on this project's own node 2026-09-12: a stalled P2P transfer
    /// had its slot swept, the HuggingFace fallback completed the shard, and
    /// the P2P chain was still asking peers for it a minute later — writing to
    /// the same file, which `write_chunk` truncates at offset 0.
    #[test]
    fn a_p2p_transfer_loses_its_file_when_its_slot_is_swept_and_another_download_claims_it() {
        let m = mgmt();
        let sid = shard("llama-3.2-3b", 1);

        let p2p_claim = m.claim_shard_download(&sid).unwrap();
        park_slot(&m, &sid, p2p_claim);
        assert!(
            m.p2p_transfer_still_owns_the_file(&sid),
            "while its slot is parked the transfer owns the file"
        );

        // The 180s stall sweep releases the slot, which drops the claim...
        m.p2p_download_permits.remove(&sid);
        // ...and the HuggingFace fallback takes it.
        let _hf_claim = m.claim_shard_download(&sid).unwrap();

        assert!(
            !m.p2p_transfer_still_owns_the_file(&sid),
            "the claim has passed to the HF download — the P2P chain must stop"
        );
    }

    /// The stall sweep asks "how long since PROGRESS", and the field must
    /// answer that question rather than "how long since the transfer started".
    /// A 512 MB shard is 64 chunks of 8 MiB, so on a 10-15 Mbit/s link — the
    /// connection the report behind this round came from — an honest transfer
    /// takes 4-7 minutes and a start-time clock declares it stalled at 3.
    #[test]
    fn a_chunk_arriving_keeps_a_slow_transfer_from_reading_as_stalled() {
        let m = mgmt();
        let sid = shard("glm-4-9b", 3);
        let claim = m.claim_shard_download(&sid).unwrap();
        park_slot(&m, &sid, claim);

        let parked_at = m.p2p_download_permits.get(&sid).unwrap().last_progress_at;
        std::thread::sleep(std::time::Duration::from_millis(15));
        m.note_p2p_transfer_progress(&sid);
        let after_chunk = m.p2p_download_permits.get(&sid).unwrap().last_progress_at;

        assert!(
            after_chunk > parked_at,
            "a chunk arriving must move the clock the stall sweep reads, or a \
             healthy slow download is killed on total elapsed time"
        );
    }

    /// A shard with no parked transfer must not be resurrected by a stray
    /// chunk — the entry is gone because the transfer is over.
    #[test]
    fn progress_on_an_unparked_shard_creates_nothing() {
        let m = mgmt();
        let sid = shard("glm-4-9b", 3);
        m.note_p2p_transfer_progress(&sid);
        assert!(m.p2p_download_permits.is_empty());
    }

    /// The P2P paths that take no claim at all (a user-initiated acquisition)
    /// must be left exactly as they were.
    #[test]
    fn an_unclaimed_shard_leaves_a_p2p_transfer_alone() {
        let m = mgmt();
        let sid = shard("llama-3.2-3b", 1);
        assert!(m.p2p_transfer_still_owns_the_file(&sid));
    }

    #[test]
    fn claims_are_per_shard_not_per_model() {
        let m = mgmt();
        let _three = m.claim_shard_download(&shard("glm-4-9b", 3)).unwrap();
        assert!(
            m.claim_shard_download(&shard("glm-4-9b", 7)).is_some(),
            "a different shard of the same model is a different file"
        );
        assert!(
            m.claim_shard_download(&shard("qwen3-1.7b", 3)).is_some(),
            "the same index of a different model is a different file"
        );
    }

    /// The point of the claim: the guard against a second writer must not
    /// depend on the progress map, which is cleaned up on a timer.
    #[test]
    fn a_live_download_is_in_progress_even_with_no_entry_in_the_progress_map() {
        let m = mgmt();
        let sid = shard("glm-4-9b", 3);
        let _claim = m.claim_shard_download(&sid).unwrap();

        assert!(
            m.acquisition_progress.get(&sid.model_id).is_none(),
            "precondition: the progress entry has been cleaned up"
        );
        assert!(
            m.is_shard_in_progress(&sid.model_id, sid.index),
            "a shard being written right now must read as in progress even \
             after its progress entry has been removed"
        );
    }

    /// `trigger_download` holds the claim itself from the moment it decides to
    /// act, so it needs a predicate that does not see its own claim.
    #[test]
    fn the_map_only_predicate_does_not_see_a_claim() {
        let m = mgmt();
        let sid = shard("glm-4-9b", 3);
        let _claim = m.claim_shard_download(&sid).unwrap();

        assert!(m.is_shard_in_progress(&sid.model_id, sid.index));
        assert!(
            !m.shard_marked_in_progress(&sid.model_id, sid.index),
            "the map-only predicate answers about the progress map alone"
        );
    }

    fn downloading_status(model: &str, shards: &[(u32, ShardState)]) -> AcquisitionStatus {
        let mut status = AcquisitionStatus::new_downloading(
            ModelId(model.to_string()),
            shards.len() as u32,
            1024,
            "huggingface",
            "auto_manage",
            "test".to_string(),
        );
        status.state = AcquisitionState::Downloading;
        for (idx, state) in shards {
            let mut sp = ShardProgress::new_downloading(*idx, 512);
            sp.state = state.clone();
            status.shard_progress.insert(*idx, sp);
        }
        status
    }

    #[test]
    fn a_model_has_a_live_download_while_any_of_its_shards_is_claimed() {
        let m = mgmt();
        let mid = ModelId("glm-4-9b".to_string());
        assert!(!m.model_has_live_shard_download(&mid));

        let claim = m.claim_shard_download(&shard("glm-4-9b", 7)).unwrap();
        assert!(m.model_has_live_shard_download(&mid));
        assert!(
            !m.model_has_live_shard_download(&ModelId("qwen3-1.7b".to_string())),
            "another model's download says nothing about this one"
        );

        drop(claim);
        assert!(!m.model_has_live_shard_download(&mid));
    }

    /// A stale `Downloading` mark left by a path that gave up without clearing
    /// its own progress entry must NOT hold the entry alive for ever — that
    /// would make the shard unfetchable, which is worse than the tidy-up it is
    /// protecting. Liveness is answered by the claims, which cannot go stale.
    #[test]
    fn a_stale_progress_mark_is_not_mistaken_for_a_live_download() {
        let m = mgmt();
        let mid = ModelId("glm-4-9b".to_string());
        m.acquisition_progress.insert(
            mid.clone(),
            downloading_status("glm-4-9b", &[(3, ShardState::Downloading)]),
        );

        assert!(
            m.shard_marked_in_progress(&mid, 3),
            "precondition: the map still carries the mark"
        );
        assert!(
            !m.model_has_live_shard_download(&mid),
            "no writer holds a claim, so nothing is actually downloading"
        );
    }
}
