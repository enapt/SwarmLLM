//! Model process pool — manages one subprocess per loaded ModelId.
//!
//! When a model is unloaded, its worker process is killed and the OS/CUDA
//! driver reclaims all GPU memory immediately — no restart required.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::process::Child;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::error::SwarmError;
use crate::inference::router::StreamingTokenEvent;
use crate::inference::worker_ipc::*;
use crate::types::{ModelId, PrefixBlockEntry, SamplingParams};

/// Cross-platform IPC stream halves between daemon and model worker.
/// Unix: AF_UNIX socket with 0o600 filesystem perms (current user only).
/// Windows: named pipe with default DACL (current logon session only).
pub(crate) type IpcReader = interprocess::local_socket::tokio::RecvHalf;
pub(crate) type IpcWriter = interprocess::local_socket::tokio::SendHalf;

/// Item 8 Phase 1: each prefix-cache insert in a worker emits one of these
/// over the pool's `prefix_manifest_tx`. The daemon-side forwarder drains
/// the channel, broadcasts a `SwarmMessage::PrefixCacheAnnounce`, and folds
/// the blocks into the local cross-node index (so a single-node loopback
/// test sees the wire path end-to-end).
#[derive(Clone, Debug)]
pub struct PrefixManifestEvent {
    pub model_id: ModelId,
    pub blocks: Vec<PrefixBlockEntry>,
}

/// Progress on a request that has not produced its first token yet. Workers
/// emit one per prefill chunk; the daemon-side forwarder stamps it onto the
/// request's `RequestTrace`, which is what every surface reads from.
#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub request_id: Uuid,
    pub phase: crate::inference::worker_ipc::ProgressPhase,
    pub done: u32,
    pub total: u32,
}

/// Item 8 Phase 2b: worker-initiated cross-node prefix-KV probe. The
/// daemon's probe handler drains these off the channel, runs
/// `SharedState::try_fetch_cross_node_prefix`, and sends the (possibly
/// empty) result back via `ModelProcessPool::send_prefix_fetch_result`.
#[derive(Clone, Debug)]
pub struct PrefixProbeEvent {
    pub model_id: ModelId,
    pub request_id: Uuid,
    pub blocks: Vec<PrefixBlockEntry>,
}

const WORKER_CONNECT_TIMEOUT_SECS: u64 = 30;

/// Exponential backoff for repeat worker-spawn failures: 1s, 2s, 4s, 8s,
/// 16s, 32s, capped at 60s. The arriving request gets `ModelNotAvailable`
/// during the cooldown window, so a permanently-broken model can't drown
/// the inference path in 30-second connect timeouts.
fn spawn_failure_cooldown(consecutive_failures: u32) -> std::time::Duration {
    let secs = match consecutive_failures {
        0 | 1 => 1,
        2 => 2,
        3 => 4,
        4 => 8,
        5 => 16,
        6 => 32,
        _ => 60,
    };
    std::time::Duration::from_secs(secs)
}
/// Default KV-cache TTL in seconds (10 minutes). Overridden by config at startup.
pub const DEFAULT_KV_CACHE_TTL_SECS: u64 = 600;

/// Per-request buffered channel capacity for multiplexed worker responses.
/// Long decode streams emit one WorkerMsg::Token per generated token; 256 gives
/// plenty of headroom for a caller that's slow to consume without stalling the
/// reader actor.
const RESPONSE_CHANNEL_CAPACITY: usize = 256;

/// Response channel entry: a bounded mpsc sender carrying `(WorkerMsg, payload_bytes)`.
type ResponseTx = mpsc::Sender<(WorkerMsg, Vec<u8>)>;

/// Shared map from `request_id` to the caller's response channel, tagged with
/// a unique per-attempt token. The reader actor looks up each inbound message's
/// `request_id` here to route the reply.
///
/// The token exists because a `request_id` is NOT unique across concurrent
/// attempts: a router retry (and a hedge race) re-sends the *same* id while the
/// original is still in flight. Keyed by id alone, the retry's insert silently
/// dropped the original's sender and the original's cleanup then removed the
/// retry's — killing both. See `WorkerHandle::register_response`.
type ResponseMap = Arc<DashMap<Uuid, (u64, ResponseTx)>>;

/// Source of per-attempt tokens for `ResponseMap`. Process-wide; only equality
/// matters, never ordering.
/// One GPU-resident model considered for reclaim, as the planner sees it.
#[derive(Debug, Clone)]
struct VramReclaimCandidate {
    model: ModelId,
    /// What admission has charged for it — the figure the budget is spent in.
    charge_mb: u64,
    idle_secs: u64,
    /// A request is in flight against it right now.
    busy: bool,
}

/// Choose which resident models to unload so `needed_mb` fits in `budget`.
///
/// Pure so the policy can be tested without spawning a worker: the wiring is
/// verified by running the node, this decides what the wiring does.
///
/// Returns an EMPTY plan when the request cannot be satisfied even by
/// reclaiming everything eligible. That is the load-bearing case: unloading
/// models and still not fitting costs the user a cold start and buys nothing,
/// because the model was going to the CPU either way.
fn plan_vram_reclaim(
    budget_mb: u64,
    committed_mb: u64,
    needed_mb: u64,
    candidates: Vec<VramReclaimCandidate>,
) -> Vec<(ModelId, u64)> {
    if committed_mb.saturating_add(needed_mb) <= budget_mb {
        return Vec::new();
    }
    let mut eligible: Vec<VramReclaimCandidate> = candidates
        .into_iter()
        .filter(|c| !c.busy && c.charge_mb > 0 && c.idle_secs >= VRAM_MAKE_ROOM_MIN_IDLE_SECS)
        .collect();
    // Most idle first: `spawned_at` cannot tell a worker answering steadily for
    // an hour from one loaded an hour ago and never used since.
    eligible.sort_by_key(|a| std::cmp::Reverse(a.idle_secs));

    let mut plan = Vec::new();
    let mut freed = 0u64;
    for c in eligible {
        if committed_mb.saturating_sub(freed).saturating_add(needed_mb) <= budget_mb {
            break;
        }
        freed = freed.saturating_add(c.charge_mb);
        plan.push((c.model, c.charge_mb));
    }
    if committed_mb.saturating_sub(freed).saturating_add(needed_mb) > budget_mb {
        return Vec::new();
    }
    plan
}

/// How long a GPU-resident model must have gone unused before another model
/// may take the card from it.
///
/// Exists only to stop two models alternating faster than they load from
/// evicting each other on every request; below it the pre-existing behaviour
/// (the newcomer runs on the CPU) is kept, so this can only ever make the
/// placement better than it was.
const VRAM_MAKE_ROOM_MIN_IDLE_SECS: u64 = 60;

static RESPONSE_ATTEMPT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A handle to a running model worker subprocess.
///
/// The socket is split into a shared writer (one-at-a-time under `Mutex`) and
/// a reader owned by a dedicated actor task. The reader dispatches each
/// incoming `WorkerMsg` to the per-request channel keyed by `request_id`,
/// allowing N concurrent `forward()` / `generate()` callers to interleave
/// their requests through the same worker without serializing on a single
/// full-request mutex.
struct WorkerHandle {
    /// The worker subprocess.
    ///
    /// `Option` so `Drop` can move it into a reaper task. Killing a process and
    /// the OS reclaiming its device memory are separate events, and the budget
    /// must not be handed back between them — see `exited`.
    child: Option<Child>,
    /// Set once the subprocess has actually been reaped, not merely signalled.
    ///
    /// `Child::start_kill` sends the signal and returns; the kernel tears the
    /// process down and frees its CUDA allocations afterwards. Releasing the
    /// VRAM charge on the signal made `vram_committed_mb` report zero while the
    /// card was still full, so an admission landing in that window passed
    /// against memory that did not exist and the new worker died with
    /// `CUDA_ERROR_OUT_OF_MEMORY`. Reported as an out-of-memory crash that
    /// "landed exactly when auto-manage was mid-eviction of a different model".
    exited: Arc<AtomicBool>,
    /// Write half of the IPC socket. Brief lock held only for the duration of
    /// one outbound framed message (header + optional binary payload).
    writer: Mutex<IpcWriter>,
    /// Per-request response channels. The reader actor inserts `(msg, payload)`
    /// tuples keyed by `request_id`; callers register a channel before sending
    /// their request and drain it until they get a terminal message.
    responses: ResponseMap,
    /// Set to true when the reader actor observes a socket error. Subsequent
    /// callers short-circuit with an error + trigger worker eviction.
    dead: Arc<AtomicBool>,
    /// Socket path used to connect, kept for drop-time cleanup.
    ///
    /// Unix only: the filesystem entry has to be unlinked when the worker goes
    /// away, whereas a Windows named pipe is reclaimed by the kernel once every
    /// handle closes. Carrying the field on Windows would be storing a value
    /// nothing reads, which is what the Windows lint build reports.
    #[cfg(unix)]
    socket_name: String,
    /// Handle to the reader actor task. Aborted on drop so the task doesn't
    /// outlive its worker; also unblocks any pending `recv_worker` in tests.
    reader_handle: tokio::task::JoinHandle<()>,
    /// When this worker was spawned.
    ///
    /// Idle-unload needs a floor on how long a model *could* have been idle. A
    /// model with no recorded request history is not "idle forever" — it cannot
    /// have been idle longer than it has existed. Without this, a model loaded
    /// seconds ago was evicted as though it had sat unused for the whole
    /// configured window, killing the request that had just loaded it.
    spawned_at: std::time::Instant,
    /// Seconds since this worker last had a request registered against it.
    ///
    /// `spawned_at` is only an upper bound on idleness — it cannot tell a model
    /// answering steadily for an hour from one loaded an hour ago and never
    /// used since, and those deserve opposite treatment when something has to
    /// give up the card. Stamped in `register_response`, which is the ONE place
    /// every execution path (local, distributed, peer-served) passes through,
    /// so no caller can forget it.
    ///
    /// Stored as seconds since `spawned_at` rather than a wall-clock stamp, so
    /// a clock step cannot make a busy worker look idle for hours.
    last_used: AtomicU64,
}

/// Pull the `request_id` field out of any `WorkerMsg` variant that carries one.
/// `Ready` / `Bye` have no request_id and are dropped by the reader actor
/// (they're only relevant during spawn, which handshakes synchronously).
/// `BatchResult` carries N request_ids and is handled specially by the reader
/// actor — this helper returns `None` for it. `PrefixManifestUpdate` also
/// carries no request_id; the reader actor routes it through the dedicated
/// `prefix_manifest_tx` channel instead.
fn worker_msg_request_id(msg: &WorkerMsg) -> Option<Uuid> {
    match msg {
        WorkerMsg::LayerResult(r) => Some(r.request_id),
        WorkerMsg::Token { request_id, .. }
        | WorkerMsg::GenerateDone { request_id, .. }
        | WorkerMsg::Error { request_id, .. } => Some(*request_id),
        // `PrefixSnapshotResponse` is correlated by `request_id` via the
        // normal response-routing channel — `fetch_local_snapshot`
        // registers a receiver up-front and waits for the reply.
        WorkerMsg::PrefixSnapshotResponse { request_id, .. } => Some(*request_id),
        // `Progress` carries a request_id but is explicitly NOT a response:
        // routing it to the per-request channel would hand the waiting caller a
        // progress note where it expects a result. Side-band, like the prefix
        // messages above.
        WorkerMsg::BatchResult { .. }
        | WorkerMsg::PrefixManifestUpdate { .. }
        | WorkerMsg::PrefixFetchProbe { .. }
        | WorkerMsg::Progress { .. }
        | WorkerMsg::Ready
        | WorkerMsg::Bye => None,
    }
}

/// Reader actor: owns the read half of the worker socket, dispatches each
/// inbound message to the right per-request channel. Exits when the socket
/// errors out (worker died, IPC corrupted); sets `dead` and drops all
/// in-flight response senders to wake waiting callers with `None`.
async fn reader_actor(
    mut reader: IpcReader,
    responses: ResponseMap,
    dead: Arc<AtomicBool>,
    model_id: ModelId,
    prefix_manifest_tx: Option<mpsc::Sender<PrefixManifestEvent>>,
    prefix_probe_tx: Option<mpsc::Sender<PrefixProbeEvent>>,
    progress_tx: Option<mpsc::Sender<ProgressEvent>>,
) {
    loop {
        match recv_worker(&mut reader).await {
            Ok((msg, payload)) => {
                // BatchResult carries N inner results + a concatenated payload.
                // Split it so each caller sees a synthesized LayerResult on
                // their individual response channel — callers using the batch
                // API register one channel per request_id up front.
                if let WorkerMsg::BatchResult {
                    results,
                    activation_lens,
                } = msg
                {
                    dispatch_batch_result(&responses, results, activation_lens, payload).await;
                    continue;
                }
                // Cross-node prefix-cache announcement (Item 8 Phase 1).
                // No request_id; route through the daemon-installed
                // forwarder channel. Drop silently when no forwarder is
                // installed (unit tests / pre-startup).
                if let WorkerMsg::PrefixManifestUpdate {
                    model_id: announce_model,
                    blocks,
                } = msg
                {
                    if let Some(tx) = prefix_manifest_tx.as_ref() {
                        // try_send: never block the IPC reader. Daemon being
                        // slow or absent must not stall worker responses.
                        if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) =
                            tx.try_send(PrefixManifestEvent {
                                model_id: announce_model,
                                blocks,
                            })
                        {
                            tracing::debug!("prefix manifest channel full — dropping announce");
                        }
                    }
                    continue;
                }
                // Prefill/load progress. Best-effort by construction: a full
                // channel means the display lags, which is strictly better than
                // stalling the worker's result path to deliver a status note.
                if let WorkerMsg::Progress {
                    request_id,
                    phase,
                    done,
                    total,
                } = msg
                {
                    if let Some(tx) = progress_tx.as_ref() {
                        let _ = tx.try_send(ProgressEvent {
                            request_id,
                            phase,
                            done,
                            total,
                        });
                    }
                    continue;
                }
                // Worker-initiated cross-node probe (Item 8 Phase 2b).
                if let WorkerMsg::PrefixFetchProbe {
                    request_id,
                    model_id: probe_model,
                    blocks,
                } = msg
                {
                    if let Some(tx) = prefix_probe_tx.as_ref() {
                        let probe_model_for_log = probe_model.clone();
                        if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) =
                            tx.try_send(PrefixProbeEvent {
                                model_id: probe_model,
                                request_id,
                                blocks,
                            })
                        {
                            // Dropped probes look indistinguishable from cross-node
                            // misses on the worker side — log so operators can
                            // distinguish channel-saturation from actual misses
                            // when chasing prefix-cache hit-rate regressions.
                            tracing::debug!(
                                model = %probe_model_for_log,
                                %request_id,
                                "prefix probe channel full — dropping probe"
                            );
                        }
                    }
                    continue;
                }
                if let Some(rid) = worker_msg_request_id(&msg) {
                    // `get` returns a Ref, which holds a shard lock. Clone the
                    // Sender and drop the Ref *before* awaiting `send` so we
                    // don't hold a DashMap shard across an await point (that
                    // would risk deadlock on a concurrent insert/remove).
                    if let Some(tx) = responses.get(&rid).map(|r| r.value().1.clone()) {
                        // Send best-effort; if the caller has already hung up
                        // we just drop the message.
                        let _ = tx.send((msg, payload)).await;
                    } else {
                        tracing::debug!(
                            request_id = %rid,
                            "Worker response for unknown request_id (caller dropped?)"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(model = %model_id, error = %e, "Worker reader exiting — evicting");
                // R107: Release ordering pairs with Acquire loads at every
                // dead-check call site. On weakly-ordered CPUs (ARM), a
                // Relaxed load could legally observe `dead == false` after
                // this store and the subsequent `responses.clear()`,
                // letting a concurrent caller insert into `responses` after
                // the clear — that channel would never receive anything,
                // and the caller's `recv()` loop has no timeout, so it
                // would hang forever. Release/Acquire prevents this race.
                dead.store(true, Ordering::Release);
                // Clear all pending response channels; dropping the Senders
                // makes each caller's `recv()` return `None`, which we map
                // to a "worker died" error.
                responses.clear();
                return;
            }
        }
    }
}

/// Split an incoming IpcLayerResult's binary payload back into the two
/// mutually exclusive fields it can carry (activations XOR spec_logits).
/// The worker guarantees only one of `has_activations` / `has_spec_logits`
/// is true; if the invariant is violated we preserve activations and drop
/// spec_logits with a warning (safer than panicking on the hot path).
fn reconstruct_layer_payload(
    has_activations: bool,
    has_spec_logits: bool,
    spec_logits_dims: Option<(u32, u32)>,
    payload: Vec<u8>,
) -> Result<(Vec<u8>, Vec<Vec<f32>>), SwarmError> {
    if has_activations && has_spec_logits {
        tracing::warn!(
            "IpcLayerResult has both has_activations and has_spec_logits — treating as activations"
        );
        return Ok((payload, Vec::new()));
    }
    if has_spec_logits {
        let dims = spec_logits_dims
            .ok_or_else(|| SwarmError::Internal("spec_logits flagged but dims missing".into()))?;
        let logits = crate::inference::worker_ipc::decode_spec_logits(&payload, dims)
            .map_err(SwarmError::Internal)?;
        Ok((Vec::new(), logits))
    } else if has_activations {
        Ok((payload, Vec::new()))
    } else {
        Ok((Vec::new(), Vec::new()))
    }
}

/// Fan out a BatchResult to N per-request channels. Splits the concatenated
/// payload into per-slot byte slices by `activation_lens`, wraps each inner
/// `IpcLayerResult` in a `WorkerMsg::LayerResult`, and delivers to the caller
/// registered under each `request_id`.
async fn dispatch_batch_result(
    responses: &ResponseMap,
    results: Vec<IpcLayerResult>,
    activation_lens: Vec<u32>,
    payload: Vec<u8>,
) {
    if results.len() != activation_lens.len() {
        tracing::warn!(
            results = results.len(),
            lens = activation_lens.len(),
            "BatchResult len mismatch — dropping batch"
        );
        return;
    }
    let mut cursor = 0usize;
    for (r, len) in results.into_iter().zip(activation_lens) {
        let len = len as usize;
        // SEC: only consume payload bytes for slots that actually carry an
        // activation tensor. A malformed worker response with len > 0 on a
        // has_activations=false slot would otherwise silently shift the
        // cursor past valid bytes, corrupting EVERY subsequent slot's
        // payload without any error returned.
        let (slot_payload, advance) = if r.has_activations {
            let end = cursor.saturating_add(len);
            if end <= payload.len() {
                (payload[cursor..end].to_vec(), len)
            } else {
                tracing::warn!(
                    request_id = %r.request_id,
                    cursor, len, payload_len = payload.len(),
                    "BatchResult activation length exceeds payload"
                );
                (Vec::new(), 0)
            }
        } else {
            if len != 0 {
                tracing::warn!(
                    request_id = %r.request_id,
                    len,
                    "BatchResult non-activation slot has len > 0 — ignoring (potential malformed worker response)"
                );
            }
            (Vec::new(), 0)
        };
        cursor = cursor.saturating_add(advance);
        let rid = r.request_id;
        let tx_opt = responses.get(&rid).map(|e| e.value().1.clone());
        if let Some(tx) = tx_opt {
            let _ = tx.send((WorkerMsg::LayerResult(r), slot_payload)).await;
        } else {
            tracing::debug!(
                request_id = %rid,
                "Batch slot reply for unknown request_id (caller dropped?)"
            );
        }
    }
}

/// How long an eviction waits for the subprocess to actually be reaped before
/// handing its memory budget back.
///
/// Bounds the wait on what varies — process teardown, which is fast (the
/// kernel is freeing already-allocated pages, not doing work proportional to
/// the model) but not instant. Long enough that the normal case always
/// completes inside it; short enough that a wedged process cannot stall an
/// eviction triggered from the request path.
const WORKER_EXIT_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Poll interval while waiting for a worker to be reaped.
const WORKER_EXIT_POLL: std::time::Duration = std::time::Duration::from_millis(20);

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Stop the reader actor so it doesn't outlive the socket.
        self.reader_handle.abort();
        // Kill the child process if still running, then reap it in the
        // background and publish the fact. `start_kill` only signals; whoever
        // is waiting to reuse this worker's memory needs to know when the
        // process is actually gone, which is when the OS has taken its device
        // allocations back.
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            let exited = self.exited.clone();
            // No runtime in some unit tests; there the flag simply stays false
            // and the waiter falls through on its timeout, which is the same
            // conservative outcome as before this existed.
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    let _ = child.wait().await;
                    exited.store(true, Ordering::Release);
                });
            } else {
                self.exited.store(true, Ordering::Release);
            }
        }
        // Clean up the socket file (Unix only — Windows named pipes are
        // reclaimed by the kernel when all handles close).
        #[cfg(unix)]
        let _ = std::fs::remove_file(&self.socket_name);
    }
}

/// RAII guard: unregister a request's response channel when the caller drops
/// it (whether the request finished, errored, or the caller was cancelled).
/// Without this, a cancelled request leaks its entry in `responses` forever.
struct ResponseGuard {
    responses: ResponseMap,
    request_id: Uuid,
    /// Identifies THIS attempt's entry. `Drop` removes the map entry only when
    /// it still carries this token, so a superseded attempt's cleanup cannot
    /// tear down the attempt that displaced it.
    token: u64,
    /// Worker to notify if this guard drops without being disarmed. `None`
    /// suppresses the cancel entirely (used where the request is known to have
    /// already reached the worker's terminal state).
    worker: Option<Arc<WorkerHandle>>,
}

/// Claim `request_id` in `map` for a fresh attempt. Returns the new attempt's
/// token plus the token it displaced, if another attempt was already
/// registered under the same id.
fn claim_response_slot(map: &ResponseMap, request_id: Uuid, tx: ResponseTx) -> (u64, Option<u64>) {
    let token = RESPONSE_ATTEMPT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let displaced = map.insert(request_id, (token, tx)).map(|(prev, _)| prev);
    (token, displaced)
}

/// True when `request_id` is registered to an attempt other than `token` —
/// i.e. a later attempt has taken the slot over.
fn response_slot_superseded(map: &ResponseMap, request_id: Uuid, token: u64) -> bool {
    map.get(&request_id).is_some_and(|e| e.value().0 != token)
}

impl WorkerHandle {
    /// Register `tx` as the response channel for `request_id` and return the
    /// guard that owns the entry. This is the ONLY supported way to populate
    /// `responses`: inserting directly cannot produce a correctly-tokened
    /// guard, so a superseded attempt would tear down its successor.
    ///
    /// `cancel_on_drop` sends the worker a cancel if the guard drops while
    /// still armed; pass `false` where there is no compute worth abandoning.
    ///
    /// Returns the guard plus the token displaced from the map, if any. A
    /// `Some` displaced token means another attempt is in flight under the
    /// same `request_id` — the caller has superseded it.
    fn register_response(
        self: &Arc<Self>,
        request_id: Uuid,
        tx: ResponseTx,
        cancel_on_drop: bool,
    ) -> (ResponseGuard, Option<u64>) {
        self.last_used
            .store(self.spawned_at.elapsed().as_secs(), Ordering::Relaxed);
        let (token, displaced) = claim_response_slot(&self.responses, request_id, tx);
        if displaced.is_some() {
            tracing::warn!(
                %request_id,
                "Second in-flight generate for this request_id — superseding \
                 the earlier attempt (router retry or hedge race)"
            );
        }
        (
            ResponseGuard {
                responses: self.responses.clone(),
                request_id,
                token,
                worker: cancel_on_drop.then(|| self.clone()),
            },
            displaced,
        )
    }

    /// Seconds since this worker last had a request registered against it.
    ///
    /// A worker that has never been used reports its full residency, which is
    /// the honest answer: it has been idle for as long as it has existed.
    fn idle_secs(&self) -> u64 {
        self.spawned_at
            .elapsed()
            .as_secs()
            .saturating_sub(self.last_used.load(Ordering::Relaxed))
    }

    /// True when `request_id`'s entry has been taken over by a later attempt.
    /// Distinguishes "a retry displaced me" from "the worker really died",
    /// which present identically as a closed response channel.
    fn response_superseded(&self, request_id: Uuid, token: u64) -> bool {
        response_slot_superseded(&self.responses, request_id, token)
    }
}

impl ResponseGuard {
    /// Mark the request as completed so dropping this guard does NOT cancel it.
    /// Every path that returns a terminal result must call this — otherwise a
    /// normal completion sends a spurious cancel for a request the worker has
    /// already finished. (Harmless, since the worker treats unmatched cancels
    /// as no-ops and sweeps them, but it's pure noise on the IPC socket.)
    fn disarm(&mut self) {
        self.worker = None;
    }
}

impl Drop for ResponseGuard {
    fn drop(&mut self) {
        // Identity-checked: if a retry with the same request_id has already
        // replaced our entry, that entry belongs to the retry and removing it
        // would fail a request that is still perfectly healthy.
        self.responses
            .remove_if(&self.request_id, |_, (tok, _)| *tok == self.token);

        // Still armed → the caller went away before a terminal reply arrived:
        // client disconnected, `tokio::select!` timeout fired, or a hedge race
        // resolved and this was the loser. Tell the worker to stop; otherwise
        // it computes to completion and the reader actor silently discards the
        // result. For a `Generate` that is the whole remaining token budget of
        // GPU time spent on output nobody will read.
        let Some(worker) = self.worker.take() else {
            return;
        };
        // Drop is sync, so the IPC write is handed to a task. `try_current`
        // rather than `Handle::current` because a guard can be dropped outside
        // a runtime in tests, where losing a best-effort cancel is fine.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let request_id = self.request_id;
        handle.spawn(async move {
            if worker.dead.load(Ordering::Acquire) {
                return;
            }
            let mut writer = worker.writer.lock().await;
            let _ = send_daemon(&mut *writer, &DaemonMsg::CancelRequest { request_id }, &[]).await;
        });
    }
}

/// Auto-coalescing batch scheduler loop. One per `ModelProcessPool`. Collects
/// `Forward` requests into time-windowed batches grouped by `model_id`, then
/// dispatches each group via `pool.forward_batch(...)`. Responses are fanned
/// out to the per-request `oneshot::Sender`s.
///
/// Worker-side CPU fallback (`run_fused_batch_forward` errors out on CPU, which
/// `handle_batch_forward` catches and runs sequentially) means this is safe on
/// every device — GPU workers run the fused `SplitModel::forward_batch` path,
/// CPU workers run the sequential path with zero regression.
async fn batch_scheduler_loop(
    pool: Arc<ModelProcessPool>,
    mut rx: mpsc::Receiver<BatchSchedulerMsg>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    loop {
        let first = tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                tracing::debug!("batch_scheduler_loop: shutdown observed");
                return;
            }
            msg = rx.recv() => match msg {
                Some(m) => m,
                None => return, // sender side dropped
            },
        };
        let collection_ms = pool
            .batch_collection_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        let max_batch = pool
            .max_concurrent_decode_batch
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        let max_batch = max_batch.max(1);

        let mut pending: Vec<BatchSchedulerMsg> = vec![first];
        if collection_ms > 0 {
            let deadline = Instant::now() + Duration::from_millis(collection_ms);
            while pending.len() < max_batch {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match tokio::time::timeout(deadline - now, rx.recv()).await {
                    Ok(Some(msg)) => pending.push(msg),
                    Ok(None) => {
                        // Sender dropped — process what we have and exit.
                        dispatch_scheduler_pending(&pool, pending).await;
                        return;
                    }
                    Err(_) => break, // deadline elapsed
                }
            }
        }

        // Group by model_id — forward_batch requires a single model.
        let mut by_model: HashMap<ModelId, Vec<BatchSchedulerMsg>> = HashMap::new();
        for msg in pending {
            let BatchSchedulerMsg::Forward { ref fwd, .. } = msg;
            by_model.entry(fwd.model_id.clone()).or_default().push(msg);
        }
        for (_, msgs) in by_model {
            dispatch_scheduler_group(&pool, msgs).await;
        }
    }
}

async fn dispatch_scheduler_pending(pool: &Arc<ModelProcessPool>, pending: Vec<BatchSchedulerMsg>) {
    use std::collections::HashMap;
    let mut by_model: HashMap<ModelId, Vec<BatchSchedulerMsg>> = HashMap::new();
    for msg in pending {
        let BatchSchedulerMsg::Forward { ref fwd, .. } = msg;
        by_model.entry(fwd.model_id.clone()).or_default().push(msg);
    }
    for (_, msgs) in by_model {
        dispatch_scheduler_group(pool, msgs).await;
    }
}

/// Dispatch a same-model group: 1 request → `pool.forward_direct`, ≥2 →
/// `pool.forward_batch`. Fan out to each `resp_tx`.
async fn dispatch_scheduler_group(pool: &Arc<ModelProcessPool>, msgs: Vec<BatchSchedulerMsg>) {
    if msgs.is_empty() {
        return;
    }
    if msgs.len() == 1 {
        let BatchSchedulerMsg::Forward { fwd, resp_tx } =
            msgs.into_iter().next().expect("len == 1 checked above");
        let result = pool.forward_direct(fwd).await;
        let _ = resp_tx.send(result);
        return;
    }
    let (forwards, resp_txs): (Vec<_>, Vec<_>) = msgs
        .into_iter()
        .map(|BatchSchedulerMsg::Forward { fwd, resp_tx }| (fwd, resp_tx))
        .unzip();
    let n = forwards.len();
    match pool.forward_batch(forwards).await {
        Ok(results) => {
            if results.len() != n {
                for tx in resp_txs {
                    let _ = tx.send(Err(SwarmError::Internal(format!(
                        "batch result count mismatch: expected {n}, got {}",
                        results.len()
                    ))));
                }
                return;
            }
            // Fan out in sender order (forward_batch preserves input order).
            for (tx, result) in resp_txs.into_iter().zip(results) {
                let _ = tx.send(Ok(result));
            }
        }
        Err(e) => {
            // SwarmError isn't Clone, so we can't propagate `e` to each
            // tx. The vast-majority batch failure cause is worker death
            // (subprocess closed IPC) — ServiceUnavailable is correct
            // for that. Was previously Internal which mapped to 500 and
            // misled operators about the failure class.
            let err_str = e.to_string();
            for tx in resp_txs {
                let _ = tx.send(Err(SwarmError::ServiceUnavailable(format!(
                    "batch failed: {err_str}"
                ))));
            }
        }
    }
}

/// Manages one worker subprocess per loaded ModelId.
///
/// When a model is unloaded, its worker process is killed and the OS/CUDA
/// driver reclaims all GPU memory immediately — no restart required.
///
/// ## Concurrency model
///
/// Each WorkerHandle has a `Mutex<write_half>` and a reader-actor task that
/// multiplexes inbound `WorkerMsg`s by `request_id` to per-request channels.
/// `forward()` and `generate()` only hold the write mutex long enough to send
/// one framed IPC message; waiting for the response happens off-lock on the
/// per-request channel. **Multiple concurrent `forward()` / `generate()` calls
/// against the same model no longer block each other**, as long as the worker
/// itself can make progress on them.
///
/// Compute-side serialization still applies: the worker subprocess handles one
/// forward call at a time internally (until Item 7 BatchGenerate lands proper
/// slot batching). So two concurrent requests share the worker in a "fair"
/// interleaved fashion — each request's message arrives at the worker in the
/// order it was sent, and responses flow back in whatever order the worker
/// emits them.
/// Why a model is running on the CPU.
///
/// Kept as three distinct values rather than a bool because they call for three
/// different actions from the user: change a setting, free some memory, or
/// accept that this machine's GPU is too old. Downstream they were all just
/// `--gpu-layers 0`, which is how a correct setting and an override came to look
/// identical — see [`ModelProcessPool::cpu_reason`].
/// Does a model being spawned count against the system-RAM budget (and get a
/// KV budget handed to its worker)? Yes when it is being sent to the CPU, when
/// no GPU was detected on this node, or when this build cannot drive one.
/// Pure, so the truth table is pinned by `ram_is_charged_wherever_the_model_
/// can_only_land_in_ram`.
pub(crate) fn charges_ram(going_to_cpu: bool, gpu_detected: bool, build_has_cuda: bool) -> bool {
    going_to_cpu || !gpu_detected || !build_has_cuda
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// `inference.gpu_layers = 0` — the user asked for CPU.
    Configured,
    /// This build's CUDA kernels need a newer card than the one present.
    GpuTooOld,
    /// The model's estimated footprint exceeds the free VRAM budget. Unlike the
    /// other two this clears itself once memory frees up.
    NotEnoughVram,
}

impl CpuReason {
    /// Stable machine-readable tag. Used in logs and in the admin API, so keep
    /// these values stable — something will grep for them.
    pub fn as_str(self) -> &'static str {
        match self {
            CpuReason::Configured => "configured_cpu_only",
            CpuReason::GpuTooOld => "gpu_too_old_for_this_build",
            CpuReason::NotEnoughVram => "not_enough_vram",
        }
    }
}

pub struct ModelProcessPool {
    workers: DashMap<ModelId, Arc<WorkerHandle>>,
    /// Serializes worker spawning to prevent TOCTOU races where two concurrent
    /// callers both miss the DashMap lookup and each spawn a subprocess.
    spawn_lock: Mutex<()>,
    /// Crash-loop backoff: per-model `(last_failure_at, consecutive_failures)`.
    /// A spawn that fails (or crashes immediately) bumps the counter; further
    /// `get_or_spawn` calls within the cooldown window short-circuit with
    /// `ModelNotAvailable` instead of repeatedly burning the 30 s
    /// `WORKER_CONNECT_TIMEOUT_SECS` on every arriving request. First
    /// successful spawn clears the entry.
    spawn_failures: DashMap<ModelId, (std::time::Instant, u32)>,
    data_dir: PathBuf,
    /// Active shard windows: which shards each model worker should load.
    /// If absent, the worker loads all on-disk shards (default behavior).
    active_shard_windows: DashMap<ModelId, Vec<u32>>,
    /// Device placement passed to future-spawned workers, mirroring
    /// `InferenceConfig::gpu_layers`: `-1` auto, `0` CPU only, `>0` GPU.
    gpu_layers: std::sync::atomic::AtomicI32,
    // (see `CpuReason` for why the three CPU causes are kept distinct)
    /// Models forced onto the CPU for the rest of this daemon's life because
    /// a worker died of a GPU OOM while serving them. Without this, the
    /// respawned worker makes the identical allocation and dies the same way,
    /// and the user sees an unbroken run of 500s with no path out.
    cpu_pinned_models: dashmap::DashSet<ModelId>,
    /// GPU memory budget in MB, 0 = unset (no admission control).
    ///
    /// Mirror of the effective `resources` budget, set at startup like the other
    /// knobs here. The pool cannot reach `SharedState`, and this has to be
    /// readable from inside `spawn_lock` where the admission decision happens.
    vram_budget_mb: std::sync::atomic::AtomicU64,
    /// CPU threads each worker's rayon pool may use. 0 until set at startup,
    /// which the spawn path reads as "not configured" and leaves alone.
    cpu_threads: std::sync::atomic::AtomicUsize,
    /// GPU memory charged to each live worker, in MB.
    ///
    /// Admission needs to know what is already committed, and it cannot ask the
    /// device: `nvidia-smi` reports the whole machine (so it counts other
    /// programs, and on WSL per-process figures are unavailable), and a worker
    /// that has been admitted but has not finished loading has allocated
    /// nothing yet while still owing its footprint. Charging at admission and
    /// crediting on unload is the only view that is correct at the moment the
    /// decision is made.
    vram_reserved_mb: dashmap::DashMap<ModelId, u64>,
    /// System RAM budget in MB, 0 = unset (no admission control).
    ///
    /// The CPU-side sibling of `vram_budget_mb`, and it matters most exactly
    /// where that one gives up: refusing a model the GPU cannot hold loads it
    /// on the CPU instead, so the busier the GPU admission control is, the more
    /// weight lands in system RAM. With no ceiling there, the fallback that
    /// keeps a node answering is also what drives a small machine into swap.
    ram_budget_mb: std::sync::atomic::AtomicU64,
    /// How `ram_budget_mb` was arrived at, in one sentence, for the refusal
    /// message — the number alone surprised a user whose `max_ram_mb` was
    /// larger (external report, 2026-08-21).
    ram_budget_note: std::sync::Mutex<String>,
    /// System RAM charged to each live CPU worker, in MB. Same
    /// charge-at-admission / credit-on-unload discipline as
    /// `vram_reserved_mb`, and for the same reason: a worker owes its
    /// footprint from the moment it is admitted, long before it has loaded
    /// anything to measure.
    ram_reserved_mb: dashmap::DashMap<ModelId, u64>,
    /// Live RAM budget source, installed once by the daemon
    /// (`set_ram_budget_provider`): returns the cap from the CURRENT config and
    /// the anti-swap headroom from memory free NOW. Without one (tests), the
    /// stored `ram_budget_mb` cap is used alone.
    #[allow(clippy::type_complexity)]
    ram_budget_provider: std::sync::OnceLock<
        Box<dyn Fn() -> Option<crate::model::auto_manage::vram::RamBudget> + Send + Sync>,
    >,
    /// Whether this node detected a GPU at startup (`SharedState::gpu_info`).
    /// Defaults to `true` so nothing changes for a pool nobody told; the daemon
    /// sets it once. See `charges_ram`.
    gpu_detected: std::sync::atomic::AtomicBool,
    /// KV budget handed to each CPU worker at spawn (`--kv-budget-bytes`):
    /// the model's typical-context KV charge plus the RAM budget still
    /// uncommitted at admission. Set by `record_cpu_kv_budget`, cleared with
    /// the reservation.
    cpu_kv_budget_bytes: dashmap::DashMap<ModelId, u64>,
    /// Activity event sender for dashboard notifications.
    activity_tx:
        std::sync::OnceLock<tokio::sync::broadcast::Sender<crate::daemon::state::ActivityEvent>>,
    /// KV-cache session TTL passed to worker subprocesses (from config).
    kv_cache_ttl_secs: std::sync::atomic::AtomicU64,
    /// Prefix-cache config snapshot applied to future-spawned workers.
    /// Reading/writing is Relaxed — workers are spawned rarely enough that
    /// we don't care about cross-thread immediacy.
    prefix_cache_enabled: std::sync::atomic::AtomicBool,
    prefix_cache_max_entries: std::sync::atomic::AtomicU32,
    prefix_cache_max_prompt_tokens: std::sync::atomic::AtomicU32,
    prefix_cache_max_mb: std::sync::atomic::AtomicU32,
    prefix_cache_block_tokens: std::sync::atomic::AtomicU32,
    prefix_cache_min_tokens: std::sync::atomic::AtomicU32,
    /// SWIFT (arxiv 2410.06916) self-speculative decoding settings applied
    /// to future-spawned workers.
    swift_self_speculative: std::sync::atomic::AtomicBool,
    swift_calibration_tokens: std::sync::atomic::AtomicU32,
    swift_gamma: std::sync::atomic::AtomicU32,
    /// Stored as parts-per-thousand to fit into AtomicU32 (e.g. 0.45 → 450).
    swift_skip_ratio_milli: std::sync::atomic::AtomicU32,
    /// Force `standard_attention` everywhere (baseline + speculative paths).
    /// Required for SWIFT correctness; optional for benchmarking baselines.
    force_standard_attn: std::sync::atomic::AtomicBool,
    /// 0 means use the GGUF context_length verbatim. >0 caps it for KV-cache
    /// pre-allocation, so 128K-context models fit on small VRAM.
    max_seq_len_override: std::sync::atomic::AtomicU32,
    /// Quantize intermediate-segment hidden state activations to Q8_0 before
    /// returning them to the daemon (which forwards to the next pipeline peer).
    /// Receivers auto-dispatch on the dtype tag. See Item 13 in
    /// `docs/plans/archive/distributed_inference_speedup.md`.
    activation_compression: std::sync::atomic::AtomicBool,
    /// Continuous batching: when on, `forward()` routes through an
    /// auto-coalescing scheduler that collects concurrent arrivals into a
    /// time-windowed batch and dispatches via `forward_batch` (one
    /// `DaemonMsg::BatchForward` IPC message, one fused `SplitModel::forward_batch`
    /// call on GPU, falls through to sequential on CPU). Off → `forward()`
    /// bypasses the scheduler and sends a single `DaemonMsg::Forward`.
    continuous_batching: std::sync::atomic::AtomicBool,
    /// Time-window in milliseconds for the scheduler to wait for additional
    /// arrivals after the first request lands in an empty batch. Matches
    /// `InferenceConfig::batch_collection_ms`.
    batch_collection_ms: std::sync::atomic::AtomicU64,
    /// Maximum batch size. Matches `InferenceConfig::max_concurrent_decode_batch`.
    max_concurrent_decode_batch: std::sync::atomic::AtomicU32,
    /// Item 7 Phase 2: Sarathi prefill chunk size (in prompt tokens). Passed
    /// into spawned workers as `--prefill-chunk-tokens`. Matches
    /// `InferenceConfig::prefill_chunk_tokens`.
    prefill_chunk_tokens: std::sync::atomic::AtomicU32,
    /// Runtime mirror of `InferenceConfig::prefill_target_ms`.
    prefill_target_ms: std::sync::atomic::AtomicU64,
    /// Longest n-gram the local draft-free speculator will try to match.
    /// Matches `InferenceConfig::ngram_max_size`.
    ngram_max_size: std::sync::atomic::AtomicU32,
    /// How many tokens a local speculative round drafts. **Zero means the
    /// speculator is off**, which is how `InferenceConfig::ngram_lookup_enabled`
    /// reaches the worker — one value carries both the switch and the width, so
    /// a worker cannot be spawned with them disagreeing.
    ngram_pred_tokens: std::sync::atomic::AtomicU32,
    /// Item 7 Phase 4: fuse concurrent same-shape Prefilling slots into one
    /// `forward_batch` call. Passed into spawned workers as
    /// `--batched-prefill-forward`. Matches
    /// `InferenceConfig::batched_prefill_forward`. When false, Phase A always
    /// runs singleton forwards (useful for A/B benchmarks).
    batched_prefill_forward: std::sync::atomic::AtomicBool,
    /// Global batch scheduler. Created once by `start_batch_scheduler` from
    /// daemon setup (where `Arc<Self>` is available). When unset, `forward()`
    /// bypasses batching entirely regardless of the `continuous_batching` flag.
    batch_scheduler: std::sync::OnceLock<mpsc::Sender<BatchSchedulerMsg>>,
    /// Item 8 Phase 1: cross-node prefix-cache announcement sink. Set by
    /// `SharedState::new` after the daemon spawns its forwarder task; reader
    /// actors call `try_send` so a slow daemon never backpressures the worker
    /// IPC reader. When unset (e.g. unit tests constructing a bare pool),
    /// inbound `PrefixManifestUpdate` messages are dropped silently.
    prefix_manifest_tx: std::sync::OnceLock<mpsc::Sender<PrefixManifestEvent>>,
    progress_tx: std::sync::OnceLock<mpsc::Sender<ProgressEvent>>,
    /// Item 8 Phase 2b: worker-initiated fetch probes land here. Daemon
    /// drains and responds via `send_prefix_fetch_result`. Unset → drop.
    prefix_probe_tx: std::sync::OnceLock<mpsc::Sender<PrefixProbeEvent>>,
}

/// Command into the batch scheduler task.
enum BatchSchedulerMsg {
    Forward {
        fwd: crate::types::LayerForward,
        resp_tx: tokio::sync::oneshot::Sender<Result<crate::types::LayerResult, SwarmError>>,
    },
}

/// Can this `LayerForward` share a `BatchForward` message with others? Matches
/// the worker-side `batch_eligible` check: same layer_range + decode-only + no
/// vision/LoRA/spec/TP/pre-embedded/truncate. If any of these trip, we send a
/// single `DaemonMsg::Forward` instead.
fn forward_is_schedulable(f: &crate::types::LayerForward) -> bool {
    if f.sequence_num == 0 || f.index_pos == 0 {
        return false;
    }
    if f.tp_meta.is_some() {
        return false;
    }
    if f.vision_embeddings.is_some() {
        return false;
    }
    if f.adapter_id.is_some() {
        return false;
    }
    if !f.draft_tokens.is_empty() {
        return false;
    }
    if f.spec_logits_requested {
        return false;
    }
    if f.pre_embedded {
        return false;
    }
    if f.truncate_kv_to.is_some() {
        return false;
    }
    true
}

impl ModelProcessPool {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            workers: DashMap::new(),
            spawn_lock: Mutex::new(()),
            spawn_failures: DashMap::new(),
            data_dir,
            active_shard_windows: DashMap::new(),
            gpu_layers: std::sync::atomic::AtomicI32::new(-1),
            cpu_pinned_models: dashmap::DashSet::new(),
            vram_budget_mb: std::sync::atomic::AtomicU64::new(0),
            cpu_threads: std::sync::atomic::AtomicUsize::new(0),
            vram_reserved_mb: dashmap::DashMap::new(),
            ram_budget_mb: std::sync::atomic::AtomicU64::new(0),
            ram_budget_provider: std::sync::OnceLock::new(),
            gpu_detected: std::sync::atomic::AtomicBool::new(true),
            cpu_kv_budget_bytes: dashmap::DashMap::new(),
            ram_budget_note: std::sync::Mutex::new(String::new()),
            ram_reserved_mb: dashmap::DashMap::new(),
            activity_tx: std::sync::OnceLock::new(),
            kv_cache_ttl_secs: std::sync::atomic::AtomicU64::new(DEFAULT_KV_CACHE_TTL_SECS),
            prefix_cache_enabled: std::sync::atomic::AtomicBool::new(true),
            prefix_cache_max_entries: std::sync::atomic::AtomicU32::new(16),
            prefix_cache_max_prompt_tokens: std::sync::atomic::AtomicU32::new(8192),
            prefix_cache_max_mb: std::sync::atomic::AtomicU32::new(2048),
            prefix_cache_block_tokens: std::sync::atomic::AtomicU32::new(64),
            prefix_cache_min_tokens: std::sync::atomic::AtomicU32::new(32),
            swift_self_speculative: std::sync::atomic::AtomicBool::new(false),
            swift_calibration_tokens: std::sync::atomic::AtomicU32::new(32),
            swift_gamma: std::sync::atomic::AtomicU32::new(4),
            swift_skip_ratio_milli: std::sync::atomic::AtomicU32::new(450),
            force_standard_attn: std::sync::atomic::AtomicBool::new(false),
            max_seq_len_override: std::sync::atomic::AtomicU32::new(0),
            activation_compression: std::sync::atomic::AtomicBool::new(false),
            continuous_batching: std::sync::atomic::AtomicBool::new(false),
            batch_collection_ms: std::sync::atomic::AtomicU64::new(5),
            max_concurrent_decode_batch: std::sync::atomic::AtomicU32::new(8),
            prefill_chunk_tokens: std::sync::atomic::AtomicU32::new(128),
            ngram_max_size: std::sync::atomic::AtomicU32::new(0),
            ngram_pred_tokens: std::sync::atomic::AtomicU32::new(0),
            prefill_target_ms: std::sync::atomic::AtomicU64::new(200),
            batched_prefill_forward: std::sync::atomic::AtomicBool::new(true),
            batch_scheduler: std::sync::OnceLock::new(),
            prefix_manifest_tx: std::sync::OnceLock::new(),
            progress_tx: std::sync::OnceLock::new(),
            prefix_probe_tx: std::sync::OnceLock::new(),
        }
    }

    /// Install the prefix-manifest sink. The daemon spawns a forwarder task
    /// that owns the receiver and turns each event into a gossip broadcast +
    /// local-index update. Idempotent — a second call is a no-op.
    pub fn set_progress_tx(&self, tx: mpsc::Sender<ProgressEvent>) {
        let _ = self.progress_tx.set(tx);
    }

    pub fn set_prefix_manifest_tx(&self, tx: mpsc::Sender<PrefixManifestEvent>) {
        let _ = self.prefix_manifest_tx.set(tx);
    }

    /// Install the prefix-probe sink. Daemon owns the receiver and answers
    /// each probe via `send_prefix_fetch_result`.
    pub fn set_prefix_probe_tx(&self, tx: mpsc::Sender<PrefixProbeEvent>) {
        let _ = self.prefix_probe_tx.set(tx);
    }

    /// Item 8 Phase 2b: deliver a cross-node-fetch result back to the
    /// worker that emitted the probe. Worker correlates by `request_id`
    /// via its `pending_fetches` map. Sends `None` when the daemon
    /// couldn't resolve a hit (no index match, peer miss, BLAKE3 fail,
    /// timeout) — caller falls through to normal prefill.
    pub async fn send_prefix_fetch_result(
        &self,
        model_id: &ModelId,
        request_id: Uuid,
        matched_tokens: u32,
        payload: Option<Vec<u8>>,
    ) -> Result<(), SwarmError> {
        let handle = self.get_existing(model_id).ok_or_else(|| {
            SwarmError::Internal(format!(
                "send_prefix_fetch_result: no worker for model {model_id}"
            ))
        })?;
        if handle.dead.load(Ordering::Acquire) {
            return Err(SwarmError::ServiceUnavailable("worker dead".into()));
        }
        let mut writer = handle.writer.lock().await;
        let present = payload.is_some();
        let payload_slice: &[u8] = payload.as_deref().unwrap_or(&[]);
        send_daemon(
            &mut *writer,
            &DaemonMsg::PrefixFetchResult {
                request_id,
                matched_tokens,
                present,
            },
            payload_slice,
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("prefix fetch result send: {e}")))
    }

    /// Item 8 Phase 2b: serving side. The daemon received an inbound
    /// `SwarmRequest::PrefixKvFetch` — ask the local worker to extract a
    /// matching snapshot from its `PrefixCache`. Returns `Some(bytes)` on
    /// hit, `None` on miss / worker-unreachable (caller replies
    /// `PrefixKvData { payload: None }`).
    pub async fn fetch_local_snapshot(
        &self,
        model_id: &ModelId,
        block_hash: [u8; 32],
    ) -> Option<Vec<u8>> {
        let handle = self.get_existing(model_id)?;
        if handle.dead.load(Ordering::Acquire) {
            return None;
        }
        let request_id = Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel::<(WorkerMsg, Vec<u8>)>(2);
        // Snapshot export is a cheap metadata read, not a decode loop — there
        // is no meaningful compute to abandon.
        let (_guard, _) = handle.register_response(request_id, tx, false);
        {
            let mut writer = handle.writer.lock().await;
            let msg = DaemonMsg::ExportPrefixSnapshot {
                request_id,
                model_id: model_id.clone(),
                block_hash,
            };
            if let Err(e) = send_daemon(&mut *writer, &msg, &[]).await {
                tracing::debug!(error = %e, "fetch_local_snapshot: send failed");
                return None;
            }
        }
        // Wait for the worker's reply. Bounded so a stalled worker can't
        // block the manager — serving-side misses are fine, and the network
        // peer will get a `None` reply. Sized for 7B-class model snapshots:
        // 28 MB (TinyLlama) serializes in <200 ms on CPU; 73 MB (Qwen-7B
        // GQA @ 640 tokens) measured at ~500 ms. Kept under the daemon's
        // network-probe window (~2.5 s) so a stuck worker still lets the
        // network peer see a clean miss.
        match tokio::time::timeout(std::time::Duration::from_millis(2000), rx.recv()).await {
            Ok(Some((WorkerMsg::PrefixSnapshotResponse { present, .. }, payload))) => {
                if present {
                    Some(payload)
                } else {
                    None
                }
            }
            Ok(Some(_)) => None,
            Ok(None) => None,
            Err(_) => {
                tracing::debug!("fetch_local_snapshot: timed out");
                None
            }
        }
    }

    fn get_existing(&self, model_id: &ModelId) -> Option<Arc<WorkerHandle>> {
        self.workers.get(model_id).map(|r| r.clone())
    }

    /// Start the global auto-coalescing batch scheduler task. Must be called
    /// from within a Tokio runtime; no-op if no runtime is available (sync
    /// tests constructing `SharedState` directly). Safe to call more than
    /// once (second call is a no-op via `OnceLock::set`).
    pub fn start_batch_scheduler(
        self: &Arc<Self>,
        shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) {
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::debug!("start_batch_scheduler called outside a tokio runtime — skipping");
            return;
        }
        let pool = self.clone();
        let (tx, rx) = mpsc::channel::<BatchSchedulerMsg>(1024);
        if self.batch_scheduler.set(tx).is_err() {
            return;
        }
        tokio::spawn(async move {
            batch_scheduler_loop(pool, rx, shutdown_rx).await;
        });
    }

    /// Toggle auto-coalescing batch scheduler. When on, concurrent `forward()`
    /// calls for the same model are collected into one `BatchForward` IPC
    /// message. CPU workers fall through to sequential automatically; GPU
    /// workers run the fused `SplitModel::forward_batch` path. See Item 3
    /// Phase 2b in `docs/plans/archive/distributed_inference_speedup.md`.
    pub fn set_continuous_batching(&self, enabled: bool) {
        self.continuous_batching
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set continuous-batching scheduler parameters. Takes effect on the next
    /// scheduler task creation (existing per-model scheduler loops keep the
    /// window they were spawned with).
    pub fn set_batch_params(&self, collection_ms: u64, max_batch: u32) {
        self.batch_collection_ms
            .store(collection_ms, std::sync::atomic::Ordering::Relaxed);
        self.max_concurrent_decode_batch
            .store(max_batch.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Sarathi prefill chunk size for future-spawned workers. Existing
    /// workers retain whatever chunk size they were spawned with.
    /// Set the local speculator's shape. `pred_tokens == 0` disables it.
    ///
    /// Takes effect for workers spawned afterwards, like every other spawn-time
    /// option here — a running worker holds its model and cache and is not
    /// recycled to change a setting (see `settings.contribution_restart_note`).
    pub fn set_ngram_params(&self, max_size: u32, pred_tokens: u32) {
        self.ngram_max_size
            .store(max_size, std::sync::atomic::Ordering::Relaxed);
        self.ngram_pred_tokens
            .store(pred_tokens, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_prefill_chunk_tokens(&self, chunk_tokens: u32) {
        self.prefill_chunk_tokens
            .store(chunk_tokens.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Per-tick prefill wall-time budget for future-spawned workers. Existing
    /// workers retain whatever budget they were spawned with.
    pub fn set_prefill_target_ms(&self, target_ms: u64) {
        self.prefill_target_ms
            .store(target_ms.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Item 7 Phase 4: toggle prefill-chunk fusion inside the worker's
    /// `step_decode_pool` Phase A. Takes effect on the next spawned worker.
    pub fn set_batched_prefill_forward(&self, enabled: bool) {
        self.batched_prefill_forward
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Toggle Q8_0 quantization of intermediate-segment hidden state activations
    /// for future-spawned workers. Existing workers retain whatever flag they
    /// were spawned with — restart the worker to apply changes.
    pub fn set_activation_compression(&self, enabled: bool) {
        self.activation_compression
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the device placement for future-spawned workers.
    /// `-1` = auto (GPU when available), `0` = CPU only, `>0` = GPU.
    pub fn set_gpu_layers(&self, gpu_layers: i32) {
        self.gpu_layers
            .store(gpu_layers, std::sync::atomic::Ordering::Relaxed);
    }

    /// The `--gpu-layers` value a worker for `model_id` should be spawned with.
    /// A model pinned to the CPU by a previous OOM overrides the config.
    fn effective_gpu_layers(&self, model_id: &ModelId) -> i32 {
        match self.cpu_reason(model_id) {
            Some(_) => 0,
            None => self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Why this model is going to the CPU, when it is — `None` means the GPU.
    ///
    /// **The three causes are indistinguishable downstream and mean completely
    /// different things.** All of them spawn `swarmllm model-worker
    /// --gpu-layers 0` and log `requested: gpu_layers = 0`: one is the user's
    /// own setting, the other two are this node overriding it. A tester on
    /// 2026-08-10 concluded from exactly those two signals that
    /// `inference.gpu_layers` was being ignored on a node that was honouring it
    /// and then refusing the model for VRAM — and checking the worker's real
    /// command line, which is the correct way to rule out a mere logging bug,
    /// could not separate them either. The information existed; nothing carried
    /// it to where the question gets asked.
    fn cpu_reason(&self, model_id: &ModelId) -> Option<CpuReason> {
        // Checked first: an OOM pin overrides whatever the config says, and is
        // the only one of the three that clears itself when memory frees up.
        if self.cpu_pinned_models.contains(model_id) {
            return Some(CpuReason::NotEnoughVram);
        }
        // A GPU older than this build's kernel floor cannot run a single
        // forward, so sending a worker there produces a fatal CUDA error, a
        // killed worker and a spawn-backoff loop — for every model, forever.
        // Placing it on the CPU up front costs speed and keeps the node
        // answering. `local_gpu_is_supported` is cached, so this stays a plain
        // atomic load after the first call.
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return Some(CpuReason::GpuTooOld);
        }
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return Some(CpuReason::Configured);
        }
        None
    }

    /// Public form of [`ModelProcessPool::cpu_reason`], for the admin API — so
    /// "why is this model not on my GPU" is answerable without reading the log
    /// at the moment the decision was taken.
    pub fn cpu_placement_reason(&self, model_id: &ModelId) -> Option<&'static str> {
        self.cpu_reason(model_id).map(CpuReason::as_str)
    }

    /// Is this model currently forced onto the CPU after a GPU OOM?
    pub fn is_cpu_pinned(&self, model_id: &ModelId) -> bool {
        self.cpu_pinned_models.contains(model_id)
    }

    /// The configured value, before any override — so a log line can show what
    /// the user asked for next to what actually happened.
    pub fn configured_gpu_layers(&self) -> i32 {
        self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Estimate this model's GPU footprint from its own GGUF header and shards.
    ///
    /// Reads `gguf_header.bin` from the model directory; a spawn happens once per
    /// model, so the read is not on any hot path. Returns 0 when the header or
    /// geometry cannot be read, which `admit_to_gpu` treats as "do not judge" —
    /// refusing the GPU because a file was unreadable would be a worse failure
    /// than the one being prevented.
    fn estimate_gpu_footprint_mb(&self, model_id: &ModelId) -> u64 {
        use crate::model::auto_manage::vram::estimate_worker_vram_mb;
        self.footprint_inputs(model_id)
            .map(|i| estimate_worker_vram_mb(&i))
            .unwrap_or(0)
    }

    /// Estimate this model's system-RAM footprint from the same geometry.
    ///
    /// Returns 0 on an unreadable header, which `admit_to_cpu` treats as "do
    /// not judge" — refusing to load because a file could not be read would be
    /// a worse failure than the one being prevented, and matches how the GPU
    /// side handles the same gap.
    fn estimate_cpu_footprint_mb(&self, model_id: &ModelId) -> u64 {
        use crate::model::auto_manage::vram::estimate_worker_ram_mb;
        self.footprint_inputs(model_id)
            .map(|i| estimate_worker_ram_mb(&i))
            .unwrap_or(0)
    }

    /// The itemised CPU estimate and where its context came from — what the
    /// refusal message is built from.
    fn cpu_footprint_detail(
        &self,
        model_id: &ModelId,
    ) -> Option<(
        crate::model::auto_manage::vram::ResidentFootprint,
        u64,
        crate::model::auto_manage::vram::ContextSource,
    )> {
        use crate::model::auto_manage::vram::{cpu_footprint, ContextSource};
        let inputs = self.footprint_inputs(model_id)?;
        let source = if self
            .max_seq_len_override
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            ContextSource::Override
        } else {
            ContextSource::DeclaredOrDefault
        };
        Some((cpu_footprint(&inputs), inputs.effective_context, source))
    }

    /// Read a model's real geometry from its GGUF header and on-disk shards.
    ///
    /// Shared by both footprint estimators so the GPU and CPU figures can never
    /// disagree about the model's shape — only about the per-process overhead
    /// that is genuinely device-specific. Returns `None` when the header or
    /// geometry cannot be read.
    fn footprint_inputs(
        &self,
        model_id: &ModelId,
    ) -> Option<crate::model::auto_manage::vram::VramFootprintInputs> {
        use crate::model::auto_manage::vram::VramFootprintInputs;
        let model_dir = crate::model::shard::model_dir(&self.data_dir, &model_id.0);
        let header = model_dir.join(crate::model::shard::HEADER_FILENAME);
        let meta = crate::inference::split::GgufTokenizerMeta::from_gguf_file(&header).ok()?;
        let file = std::fs::File::open(&header).ok()?;
        let mut reader = std::io::BufReader::new(file);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut reader).ok()?;
        let tensor_meta = crate::inference::split::GgufTensorMeta::from_content(&ct).ok()?;
        let arch = crate::inference::split::gguf_arch_str(&ct);
        let md_u32 = |suffix: &str| -> Option<u64> {
            ct.metadata
                .get(&format!("{arch}.{suffix}"))
                .and_then(|v| v.to_u32().ok())
                .map(u64::from)
        };
        let declared_ctx = md_u32("context_length").unwrap_or(4096);
        // The KV cache is sized to the EFFECTIVE context, which is the override
        // when set and otherwise the shipped default cap — not the GGUF value.
        // The daemon holds the override in its own atomic (the worker's global
        // is set at spawn), so apply the shared rule against that value rather
        // than re-deriving the arithmetic here.
        let override_ctx = self
            .max_seq_len_override
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        let effective_ctx =
            crate::inference::split::effective_context_length_with(declared_ctx as usize, {
                if override_ctx > 0 {
                    Some(override_ctx)
                } else {
                    None
                }
            }) as u64;
        let vocab = md_u32("vocab_size").unwrap_or(meta.vocab.len() as u64);

        // Only the shards actually on disk will be mapped.
        let shard_bytes: u64 = std::fs::read_dir(&model_dir)
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin"))
                    })
                    .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                    .sum()
            })
            .unwrap_or(0);

        // Is this an UNQUANTIZED checkpoint? Read it off the largest tensor,
        // which on any architecture is one of the big linear weights: in a
        // quantized GGUF that is a Q* block type, and in an unquantized one it
        // is F16/BF16/F32. Norm and bias tensors are f32 even in a quantized
        // file, so "does any f32 tensor exist" would answer the wrong question.
        let unquantized_bytes_per_element = ct
            .tensor_infos
            .values()
            .max_by_key(|t| t.shape.elem_count())
            .and_then(|t| match t.ggml_dtype {
                candle_core::quantized::GgmlDType::F16
                | candle_core::quantized::GgmlDType::BF16 => Some(2),
                candle_core::quantized::GgmlDType::F32 => Some(4),
                _ => None,
            });

        // Shape-and-dtype half of "can the loader read embedding rows on
        // demand"; each estimator applies the device half itself.
        let embedding_gatherable = ct
            .tensor_infos
            .get("token_embd.weight")
            .is_some_and(|info| {
                crate::inference::split::table_supports_row_gather(
                    info.ggml_dtype,
                    info.shape.dims(),
                )
            });

        Some(VramFootprintInputs {
            quantized_weight_bytes: shard_bytes,
            unquantized_bytes_per_element,
            embedding_gatherable,
            vocab_size: vocab,
            embedding_length: tensor_meta.embedding_length as u64,
            segment_layers: tensor_meta.block_count as u64,
            head_count_kv: tensor_meta.head_count_kv as u64,
            head_dim: tensor_meta.head_dim as u64,
            rope_dim: tensor_meta.rope_dim as u64,
            effective_context: effective_ctx,
            is_first: true,
        })
    }

    /// Set the GPU memory budget used for admission. 0 disables the check.
    pub fn set_vram_budget_mb(&self, budget_mb: u64) {
        self.vram_budget_mb
            .store(budget_mb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the CPU thread count handed to each worker's rayon pool.
    ///
    /// Resolved from `resources.max_cpu_threads` / `node.contribution` at
    /// startup, like the other mirrors here — the pool cannot reach
    /// `SharedState`, and this has to be readable from inside the spawn path.
    pub fn set_cpu_threads(&self, threads: usize) {
        self.cpu_threads
            .store(threads.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// GPU memory already committed to live workers, in MB.
    pub(crate) fn vram_committed_mb(&self) -> u64 {
        self.vram_reserved_mb.iter().map(|e| *e.value()).sum()
    }

    /// Would this model fit on the GPU right now, without charging anything?
    ///
    /// The read-only half of [`Self::admit_to_gpu`], for callers that need to
    /// *decide* rather than *load* — the scheduler asking whether this node can
    /// serve a request well before it commits to serving it alone.
    ///
    /// Returns:
    /// - `Some(true)`  — it fits, or a worker for it is already live and paid for
    /// - `Some(false)` — it would be refused and fall back to the CPU
    /// - `None`        — no budget configured, or the geometry could not be read,
    ///   which must NOT be read as "no". Refusing to route on an unreadable file
    ///   would be a worse failure than the one being avoided, and matches how
    ///   `admit_to_gpu` treats the same gap.
    ///
    /// Deliberately shares `estimate_gpu_footprint_mb` and `vram_committed_mb`
    /// with the admission gate, so the scheduler's view and the loader's view
    /// cannot drift apart and disagree about whether a request could have run.
    pub fn would_fit_on_gpu(&self, model_id: &ModelId) -> Option<bool> {
        // Already resident ON THE GPU means already charged: running it costs
        // no new memory, whatever the budget currently says.
        //
        // **The residency check alone is not enough**, and answering it that
        // way made this function contradict itself. A worker that was refused
        // the GPU still lives in `workers` — it is running on the CPU — and
        // holds no VRAM at all, so "already charged" is false for it. Observed
        // live 2026-08-18: a node with a 400 MB budget reported `fits_on_gpu:
        // true` for a 3138 MB model in the same breath as
        // `cpu_placement_reason: not_enough_vram`, and a request that should
        // have gone to a peer with room ran on that node's CPU instead.
        //
        // A CPU-resident worker therefore falls through to the estimate below,
        // which is the right question for it: the VRAM is genuinely free.
        if self.workers.contains_key(model_id) && self.cpu_reason(model_id).is_none() {
            return Some(true);
        }
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 {
            return None;
        }
        let estimated = self.estimate_gpu_footprint_mb(model_id);
        if estimated == 0 {
            return None;
        }
        Some(self.vram_committed_mb().saturating_add(estimated) <= budget)
    }

    /// This model's real GPU footprint in MB, or `None` when its geometry
    /// cannot be read (no local shards, unreadable header).
    ///
    /// The figure the LOADER decides with. Exposed because the admin API used to
    /// report `estimate_model_vram_mb` — `file_size * 1.15`, whose own doc says
    /// it is 56% low on phi-3.5-mini-q4 and "useless as an admission decision" —
    /// next to `cpu_placement_reason: not_enough_vram`, so the dashboard showed
    /// a model comfortably fitting a card the daemon had just refused it on.
    pub fn estimated_gpu_mb(&self, model_id: &ModelId) -> Option<u64> {
        match self.estimate_gpu_footprint_mb(model_id) {
            0 => None,
            mb => Some(mb),
        }
    }

    /// Is this model going to run on our CPU *because it does not fit our GPU*,
    /// on a node whose GPU otherwise works?
    ///
    /// The trigger for handing a whole model to a peer instead
    /// (`scheduler::delegation_target`), and deliberately narrower than "is
    /// this on the CPU". Two of the three ways a model lands on the CPU are NOT
    /// degradations and must not cause work to be sent away:
    ///
    /// - `inference.gpu_layers = 0` is the user saying they want the CPU. A
    ///   node cannot honour that by quietly using someone else's GPU.
    /// - A GPU below this build's kernel floor means EVERY model runs on the
    ///   CPU here. That is how the node works, not a fault of this model, and
    ///   treating it as degradation would turn a serving node into a proxy for
    ///   all of its traffic — a much larger change than this, and one the owner
    ///   has not asked for.
    ///
    /// What is left is the case reported on 2026-08-17: a working GPU that this
    /// particular model does not fit in.
    pub fn is_cpu_bound_for_lack_of_vram(&self, model_id: &ModelId) -> bool {
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return false;
        }
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return false;
        }
        matches!(self.would_fit_on_gpu(model_id), Some(false))
    }

    /// The configured GPU memory budget in MB, or `None` when unset.
    ///
    /// Admission compares against THIS, not against the card's total VRAM — so
    /// anything reporting "will it fit" has to use the same number or it
    /// contradicts the daemon.
    pub fn vram_budget_mb(&self) -> Option<u64> {
        match self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            mb => Some(mb),
        }
    }

    /// Decide whether `model_id` may be loaded onto the GPU, and charge it if so.
    ///
    /// Called inside `spawn_lock`, which already serializes spawns, so the
    /// read-decide-charge sequence cannot interleave with another admission.
    /// That matters because a worker is admitted well before it allocates: the
    /// process starts here, and the model is loaded lazily on its first message.
    /// Two concurrent spawns that both consulted the *device* would each see
    /// plenty free and both proceed.
    ///
    /// Returning `false` means "spawn on the CPU instead" — deliberately slow
    /// rather than a `CUDA_ERROR_OUT_OF_MEMORY` that kills the worker and (until
    /// the pin became recoverable) cost the model its GPU for the whole run.
    fn admit_to_gpu(&self, model_id: &ModelId, estimated_mb: u64) -> bool {
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 || estimated_mb == 0 {
            // No budget configured, or nothing to weigh: preserve the previous
            // behaviour rather than inventing a limit.
            self.vram_reserved_mb.insert(model_id.clone(), estimated_mb);
            return true;
        }
        let committed = self.vram_committed_mb();
        if committed.saturating_add(estimated_mb) <= budget {
            // Record what we believed BEFORE the worker allocates anything.
            //
            // Admission only ever logged when it refused, so a model that was
            // admitted and then died with CUDA_ERROR_OUT_OF_MEMORY left no trace
            // of what this gate had expected it to cost. That is precisely the
            // case that needs explaining, and without this line it cannot be
            // told apart from the outside: an estimate that was too low, a
            // budget already spent by a model still being evicted, and a
            // genuinely unbounded allocation all look identical in the logs.
            // Compare `estimated_mb` here against the worker's own
            // `vram_after_load_mb` — a large gap is an under-estimate, a close
            // match with an OOM anyway points at the eviction timing instead.
            tracing::info!(
                model = %model_id,
                estimated_mb,
                committed_mb = committed,
                budget_mb = budget,
                headroom_mb = budget.saturating_sub(committed.saturating_add(estimated_mb)),
                "DIAG: admitting model to GPU"
            );
            self.vram_reserved_mb.insert(model_id.clone(), estimated_mb);
            return true;
        }
        // Deliberately DEBUG, and deliberately silent about what happens next.
        // This used to be a WARN asserting "loading it on the CPU instead" — and
        // once `free_vram_for_admission` was added that became a statement the
        // very next line could falsify. Verified on this machine 2026-08-25: the
        // line was logged, an idle model was reclaimed 0.3 ms later, and the
        // model loaded on the GPU. An operator reading the log would have
        // concluded the opposite of what happened. A refusal here is one step in
        // a decision, not the decision; the outcome is announced by the caller,
        // where it is known.
        tracing::debug!(
            model = %model_id,
            estimated_mb,
            committed_mb = committed,
            budget_mb = budget,
            "DIAG: GPU admission refused — not enough budget at this moment"
        );
        false
    }

    /// Wait for a killed worker to actually be reaped, up to `limit`.
    ///
    /// Returns `true` if the process is confirmed gone. `false` means the wait
    /// expired — the caller must decide, and today it frees the budget anyway
    /// rather than strand the device forever on one stuck process.
    async fn await_worker_exit(exited: &Arc<AtomicBool>, limit: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + limit;
        loop {
            if exited.load(Ordering::Acquire) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(WORKER_EXIT_POLL).await;
        }
    }

    /// Reclaim graphics memory from models nothing is using, so `model_id` can
    /// have the card instead of being demoted to the CPU.
    ///
    /// Called when [`Self::admit_to_gpu`] refuses, BEFORE the refusal reaches
    /// the caller — the same shape as [`Self::free_ram_for_admission`], which
    /// has done exactly this for the RAM budget since v0.3.111. The GPU side
    /// never had it, so a model that happened to load first kept the card for
    /// as long as it stayed resident and everything asked for afterwards ran on
    /// the processor.
    ///
    /// Observed on this development machine 2026-08-25: an 8B loaded at 14:17
    /// and untouched since held a 6033 MB reservation against a 6722 MB budget;
    /// a 3B requested fifteen minutes later needed 3138 MB, was refused the 689
    /// MB that remained, and answered at 7 tok/s on the CPU while the card sat
    /// idle. The background idle-unload could not help: it runs on a timer, and
    /// its regional-demand clause deliberately keeps a model the swarm wants
    /// warm for up to an hour. Neither can act on the request arriving *now*.
    ///
    /// Three properties this must keep.
    ///
    /// **Plan before destroying.** Unloading models and *still* not fitting
    /// costs the user a reload for nothing, so the whole plan is costed first
    /// and abandoned whole if it cannot succeed. The RAM sibling can unload
    /// opportunistically because its alternative is failing the request
    /// outright; here the alternative is a slower answer, so a wasted eviction
    /// is a real regression rather than a lesser evil.
    ///
    /// **Take the card only from a model that has stopped being used.** A
    /// worker with a request in flight is never a candidate, and neither is one
    /// used within `VRAM_MAKE_ROOM_MIN_IDLE_SECS`. Without that, two models
    /// alternating faster than they load would evict each other on every
    /// request and each answer would pay a cold start. With it, that case
    /// simply keeps today's behaviour — one of them runs on the CPU — while a
    /// genuinely stale occupant is displaced.
    ///
    /// **Least-recently-used first**, by `idle_secs` rather than by residency:
    /// a worker answering steadily for an hour and one loaded an hour ago and
    /// never used since are indistinguishable by `spawned_at`, and deserve
    /// opposite treatment.
    ///
    /// Returns the megabytes reclaimed.
    async fn free_vram_for_admission(&self, exclude: &ModelId, needed_mb: u64) -> u64 {
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 || needed_mb == 0 {
            return 0;
        }
        let committed = self.vram_committed_mb();
        if committed.saturating_add(needed_mb) <= budget {
            return 0;
        }

        // Candidates: workers holding a VRAM charge that are neither the model
        // being loaded, nor busy, nor recently used.
        let candidates: Vec<VramReclaimCandidate> = self
            .vram_reserved_mb
            .iter()
            .filter(|e| e.key() != exclude && *e.value() > 0)
            .filter_map(|e| {
                let worker = self.workers.get(e.key())?;
                Some(VramReclaimCandidate {
                    model: e.key().clone(),
                    charge_mb: *e.value(),
                    idle_secs: worker.idle_secs(),
                    busy: !worker.responses.is_empty(),
                })
            })
            .collect();

        let plan = plan_vram_reclaim(budget, committed, needed_mb, candidates);
        if plan.is_empty() {
            tracing::debug!(
                model = %exclude,
                needed_mb,
                committed_mb = committed,
                budget_mb = budget,
                "No idle model could be reclaimed to fit this one — leaving the GPU as it is"
            );
            return 0;
        }

        let mut reclaimed = 0u64;
        for (victim, charge) in plan {
            tracing::info!(
                model = %victim,
                reclaimed_mb = charge,
                for_model = %exclude,
                "Freeing graphics memory from an idle model so the requested one can use the GPU"
            );
            self.unload_model(&victim).await;
            reclaimed = reclaimed.saturating_add(charge);
        }
        reclaimed
    }

    /// Release a worker's charge. Must pair with every `admit_to_gpu`.
    fn release_vram_charge(&self, model_id: &ModelId) {
        self.vram_reserved_mb.remove(model_id);
    }

    /// Set the system RAM budget used for CPU admission. 0 disables the check.
    pub fn set_ram_budget_mb(&self, budget_mb: u64) {
        self.ram_budget_mb
            .store(budget_mb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Install the live budget source. Installed once; later calls are ignored.
    #[allow(clippy::type_complexity)]
    pub fn set_ram_budget_provider(
        &self,
        provider: Box<dyn Fn() -> Option<crate::model::auto_manage::vram::RamBudget> + Send + Sync>,
    ) {
        let _ = self.ram_budget_provider.set(provider);
    }

    /// The RAM budget to judge an admission against, RIGHT NOW. `None` = do
    /// not judge (no cap derivable, and none stored).
    fn ram_budget_now(&self) -> Option<crate::model::auto_manage::vram::RamBudget> {
        if let Some(p) = self.ram_budget_provider.get() {
            return p();
        }
        match self
            .ram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            cap => Some(crate::model::auto_manage::vram::RamBudget::cap_only(cap)),
        }
    }

    /// See `ram_budget_note`.
    /// Tell the pool whether a GPU was detected on this node at all. A node
    /// without one never takes the VRAM-refusal branch (there is no budget to
    /// refuse against), so without this its models landed in system RAM with
    /// nothing charged and no KV budget handed to the worker — every CPU-only
    /// machine, which is exactly where swapping hurts most (2026-08-21).
    pub fn set_gpu_detected(&self, detected: bool) {
        self.gpu_detected
            .store(detected, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_ram_budget_note(&self, note: String) {
        if let Ok(mut n) = self.ram_budget_note.lock() {
            *n = note;
        }
    }

    fn ram_budget_note(&self) -> String {
        self.ram_budget_note
            .lock()
            .map(|n| n.clone())
            .unwrap_or_default()
    }

    /// System RAM already committed to live CPU workers, in MB.
    fn ram_committed_mb(&self) -> u64 {
        self.ram_reserved_mb.iter().map(|e| *e.value()).sum()
    }

    /// After a CPU admission: how much KV this model's worker may hold — its
    /// typical-context charge (already inside the reservation) plus whatever of
    /// the RAM budget nothing else has claimed. Handed to the worker at spawn
    /// so its runtime guard refuses a conversation that would outgrow the
    /// room, with a 503 that re-routes, instead of swapping the machine.
    /// No budget configured → no guard (the pre-2026-08-21 behaviour).
    fn record_cpu_kv_budget(&self, model_id: &ModelId) {
        let Some(budget) = self.ram_budget_now() else {
            self.cpu_kv_budget_bytes.remove(model_id);
            return;
        };
        let Some(inputs) = self.footprint_inputs(model_id) else {
            self.cpu_kv_budget_bytes.remove(model_id);
            return;
        };
        let typical_kv = crate::model::auto_manage::vram::cpu_footprint(&inputs).kv_bytes;
        let this_mb = self.ram_reserved_mb.get(model_id).map(|v| *v).unwrap_or(0);
        let headroom_mb = budget.headroom_after(self.ram_committed_mb(), this_mb);
        let kv_budget = typical_kv.saturating_add(headroom_mb.saturating_mul(1024 * 1024));
        tracing::info!(
            model = %model_id,
            kv_budget_mb = kv_budget / (1024 * 1024),
            typical_kv_mb = typical_kv / (1024 * 1024),
            headroom_mb,
            "CPU worker KV budget: typical-context charge plus uncommitted RAM budget"
        );
        self.cpu_kv_budget_bytes.insert(model_id.clone(), kv_budget);
    }

    /// Decide whether `model_id` may be loaded into system RAM, and charge it
    /// if so. Called inside `spawn_lock` for the same read-decide-charge
    /// atomicity as [`Self::admit_to_gpu`].
    ///
    /// Unlike the GPU case there is **no further fallback**: the CPU already is
    /// the fallback. So returning `false` fails the spawn rather than demoting
    /// it, and the caller surfaces `ServiceUnavailable`. That is deliberate —
    /// the alternative is swapping, which does not merely slow this model down
    /// but degrades every other request on the machine, and does so without
    /// anything in the API response explaining why.
    fn admit_to_cpu(&self, model_id: &ModelId, estimated_mb: u64) -> bool {
        let Some(budget) = self.ram_budget_now() else {
            // No budget derivable: preserve the previous behaviour rather than
            // inventing a limit.
            self.ram_reserved_mb.insert(model_id.clone(), estimated_mb);
            return true;
        };
        if estimated_mb == 0 {
            // The model's geometry could not be read: nothing to weigh.
            self.ram_reserved_mb.insert(model_id.clone(), estimated_mb);
            return true;
        }
        let committed = self.ram_committed_mb();
        if budget.allows(committed, estimated_mb) {
            self.ram_reserved_mb.insert(model_id.clone(), estimated_mb);
            return true;
        }
        tracing::warn!(
            model = %model_id,
            estimated_mb,
            committed_mb = committed,
            cap_mb = budget.cap_mb,
            available_mb = budget.available_mb,
            live_headroom_mb = budget.live_headroom_mb,
            "Not enough system memory for this model right now — refusing to load it. \
             Loading it anyway would swap, which slows down every other request \
             on this machine, not just this one"
        );
        false
    }

    /// Make room in the RAM budget by unloading models nothing is using.
    ///
    /// Called when [`Self::admit_to_cpu`] refuses, BEFORE the refusal reaches
    /// the user. Without this, a node whose budget is already spent refuses the
    /// next model outright even when the resident one is doing nothing —
    /// reported from a live node: 6210 MB resident, 5986 MB wanted, 8000 MB
    /// budget, refused. The graphics-memory path has had pressure-based
    /// eviction for exactly this; system memory had none, so the request failed
    /// and only the next background cleanup pass fixed it.
    ///
    /// Only workers with **no in-flight requests** are candidates, read from
    /// the pool's own response channels, so this can never take memory from a
    /// request that is mid-generation. Longest-resident goes first.
    ///
    /// Deliberately does not consult the pin / reference-model lists that guard
    /// the background idle-unload. Those exist to keep a wanted model *warm*,
    /// which is a preference; this runs only when the alternative is failing
    /// the request outright, and a worker respawns on its next use. Returns the
    /// megabytes reclaimed.
    async fn free_ram_for_admission(&self, exclude: &ModelId, needed_mb: u64) -> u64 {
        if self.ram_budget_now().is_none() {
            return 0;
        }
        let mut freed = 0u64;
        // Re-read each round: unloading a model frees real memory, which the
        // live headroom sees.
        while let Some(budget) = self.ram_budget_now() {
            let committed = self.ram_committed_mb();
            if budget.allows(committed, needed_mb) {
                break;
            }
            // Longest-resident worker that is not serving anything.
            let victim = self
                .workers
                .iter()
                .filter(|e| e.key() != exclude && e.value().responses.is_empty())
                .min_by_key(|e| e.value().spawned_at)
                .map(|e| e.key().clone());
            let Some(victim) = victim else {
                break; // everything left is busy — refuse rather than interrupt it
            };
            let reclaimed = self.ram_reserved_mb.get(&victim).map(|v| *v).unwrap_or(0);
            tracing::info!(
                model = %victim,
                reclaimed_mb = reclaimed,
                for_model = %exclude,
                "Unloading an unused model to make room in the memory budget"
            );
            self.unload_model(&victim).await;
            freed = freed.saturating_add(reclaimed);
            if reclaimed == 0 {
                break; // nothing accounted to it; avoid spinning
            }
        }
        freed
    }

    /// Release a worker's RAM charge. Must pair with every `admit_to_cpu`.
    fn release_ram_charge(&self, model_id: &ModelId) {
        self.ram_reserved_mb.remove(model_id);
        self.cpu_kv_budget_bytes.remove(model_id);
    }

    /// Models currently forced onto the CPU after a GPU OOM.
    ///
    /// Reported by `GET /api/admin/stats` because `inference_backend` describes
    /// the BUILD, not what any model is actually running on — it kept saying
    /// "CUDA" while every model had been pinned to the CPU, so an operator
    /// losing ~10x throughput had nothing to see. Reported 2026-07-30.
    pub fn cpu_pinned_model_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .cpu_pinned_models
            .iter()
            .map(|m| m.key().0.clone())
            .collect();
        v.sort();
        v
    }

    /// Clear a CPU pin so the next worker spawn may use the GPU again.
    /// Exposed for the admin "retry on GPU" path — VRAM pressure is usually
    /// transient (another model unloaded, another process exited).
    pub fn clear_cpu_pin(&self, model_id: &ModelId) -> bool {
        self.cpu_pinned_models.remove(model_id).is_some()
    }

    /// Force every attention call through `standard_attention` on
    /// future-spawned workers (auto-enabled while SWIFT is on).
    pub fn set_force_standard_attn(&self, force: bool) {
        self.force_standard_attn
            .store(force, std::sync::atomic::Ordering::Relaxed);
    }

    /// Cap the GGUF context_length when constructing the KV cache. Pass `None`
    /// to use the GGUF value verbatim.
    pub fn set_max_seq_len_override(&self, override_val: Option<u32>) {
        self.max_seq_len_override.store(
            override_val.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Apply SWIFT self-speculative decoding settings to future-spawned workers.
    pub fn set_swift_config(
        &self,
        enabled: bool,
        calibration_tokens: u32,
        gamma: u32,
        skip_ratio: f32,
    ) {
        use std::sync::atomic::Ordering;
        self.swift_self_speculative
            .store(enabled, Ordering::Relaxed);
        self.swift_calibration_tokens
            .store(calibration_tokens, Ordering::Relaxed);
        self.swift_gamma.store(gamma.max(1), Ordering::Relaxed);
        let milli = (skip_ratio.clamp(0.0, 0.95) * 1000.0).round() as u32;
        self.swift_skip_ratio_milli.store(milli, Ordering::Relaxed);
    }

    /// Set the KV-cache TTL for worker subprocesses (called once after config load).
    pub fn set_kv_cache_ttl(&self, ttl_secs: u64) {
        self.kv_cache_ttl_secs
            .store(ttl_secs, std::sync::atomic::Ordering::Relaxed);
    }

    /// Apply the prefix-cache section of inference config to future-spawned workers.
    pub fn set_prefix_cache_config(
        &self,
        enabled: bool,
        max_entries: u32,
        max_prompt_tokens: u32,
        block_tokens: u32,
        min_tokens: u32,
        max_mb: u32,
    ) {
        use std::sync::atomic::Ordering;
        self.prefix_cache_enabled.store(enabled, Ordering::Relaxed);
        self.prefix_cache_max_entries
            .store(max_entries, Ordering::Relaxed);
        self.prefix_cache_max_mb.store(max_mb, Ordering::Relaxed);
        self.prefix_cache_max_prompt_tokens
            .store(max_prompt_tokens, Ordering::Relaxed);
        self.prefix_cache_block_tokens
            .store(block_tokens, Ordering::Relaxed);
        self.prefix_cache_min_tokens
            .store(min_tokens, Ordering::Relaxed);
    }

    /// Set the activity event sender (called once after SharedState is created).
    pub fn set_activity_tx(
        &self,
        tx: tokio::sync::broadcast::Sender<crate::daemon::state::ActivityEvent>,
    ) {
        let _ = self.activity_tx.set(tx);
    }

    /// Get or spawn a worker for this model.
    async fn get_or_spawn(&self, model_id: &ModelId) -> Result<Arc<WorkerHandle>, SwarmError> {
        // Fast path: worker already exists
        if let Some(handle) = self.workers.get(model_id) {
            return Ok(handle.clone());
        }

        // Slow path: serialize spawns to prevent duplicate workers
        let _guard = self.spawn_lock.lock().await;
        // Re-check after acquiring lock (another task may have spawned it)
        if let Some(handle) = self.workers.get(model_id) {
            return Ok(handle.clone());
        }
        // Crash-loop backoff: refuse to re-spawn while a recent failure is
        // still inside its cooldown window. Without this a permanently-broken
        // model (corrupt shards, GPU OOM on load) burns one
        // WORKER_CONNECT_TIMEOUT_SECS = 30s spawn attempt per arriving
        // request and saturates the whole inference path.
        // Check INSIDE the spawn_lock so a concurrent failure that records
        // itself between our entry and our spawn cannot be bypassed.
        if let Some(entry) = self.spawn_failures.get(model_id) {
            let (at, count) = *entry;
            let cooldown = spawn_failure_cooldown(count);
            if at.elapsed() < cooldown {
                let remaining = cooldown.saturating_sub(at.elapsed());
                return Err(SwarmError::ServiceUnavailable(format!(
                    "Worker spawn for {} failing repeatedly — backing off for {:?} (attempt #{})",
                    model_id.0, remaining, count
                )));
            }
        }
        // Admission control. Inside `spawn_lock`, so read-decide-charge cannot
        // interleave with another spawn — which matters because a worker is
        // admitted long before it allocates (the model loads lazily on its first
        // message), so two spawns that both asked the device would each see
        // plenty free and both proceed. Refusing here means the model loads on
        // the CPU: slower, but not a dead worker and a lost GPU.
        let mut going_to_cpu = self.effective_gpu_layers(model_id) == 0;
        if !going_to_cpu {
            let estimated = self.estimate_gpu_footprint_mb(model_id);
            // Demoting to the CPU is the last resort, not the first answer:
            // reclaim the card from models nothing is using, then ask again.
            // Ask ONCE unless the answer was no — `admit_to_gpu` charges the
            // reservation when it succeeds, so a second call would weigh the
            // model against its own charge and refuse one already let in.
            let mut admitted = self.admit_to_gpu(model_id, estimated);
            if !admitted {
                let freed = self.free_vram_for_admission(model_id, estimated).await;
                if freed > 0 {
                    tracing::info!(
                        model = %model_id,
                        freed_mb = freed,
                        "Reclaimed graphics memory from idle models; retrying admission"
                    );
                    admitted = self.admit_to_gpu(model_id, estimated);
                }
            }
            if !admitted {
                // Now it IS true: the budget could not be made to fit, even
                // after reclaiming every idle model that could be spared.
                tracing::warn!(
                    model = %model_id,
                    estimated_mb = estimated,
                    committed_mb = self.vram_committed_mb(),
                    budget_mb = self
                        .vram_budget_mb
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "Not enough GPU memory budget for this model — loading it on the CPU \
                     instead (slower, but it answers). It will use the GPU again once \
                     memory frees up"
                );
                going_to_cpu = true;
                self.cpu_pinned_models.insert(model_id.clone());
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "inference",
                            "model_cpu_fallback",
                            format!(
                                "{} needs more graphics memory than is free — running it on the CPU for now",
                                model_id.0
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("warning", 8000),
                    );
                }
            }
        }
        // Say WHICH of the three reasons put this model on the CPU, every time
        // a worker spawns — not only at the moment a VRAM pin is taken. The
        // worker's command line and the loader's own log line are identical for
        // all three, so a node inspected later cannot otherwise be asked
        // "is this my setting, or did you override it?". Reported 2026-08-10 by
        // a tester who reasonably concluded the setting was being ignored.
        if going_to_cpu {
            let reason = self
                .cpu_reason(model_id)
                .map(CpuReason::as_str)
                .unwrap_or("not_enough_vram");
            tracing::info!(
                model = %model_id,
                reason,
                configured_gpu_layers = self.configured_gpu_layers(),
                estimated_vram_mb = self.estimate_gpu_footprint_mb(model_id),
                vram_budget_mb = self.vram_budget_mb.load(std::sync::atomic::Ordering::Relaxed),
                "Model will run on the CPU"
            );
        }
        // Anything landing in system RAM — a CPU-only node, or the fallback
        // just taken above — is charged against the RAM budget. There is no
        // further device to demote to, so this refuses rather than degrades:
        // swapping would slow every other request on the machine, not just
        // this model, and nothing in the response would say so.
        //
        // `going_to_cpu` alone missed every CPU-only node: with no GPU there is
        // no VRAM budget, `admit_to_gpu` admits everything, and the model lands
        // in RAM uncharged. `charges_ram` adds "no GPU detected" and "this build
        // has no CUDA"; placement is deliberately NOT changed by it — the worker
        // falls back to the CPU on its own, so a working card whose probe failed
        // is never sent to the CPU by this (unreadable is unknown, not absent).
        let charge_ram = charges_ram(
            going_to_cpu,
            self.gpu_detected.load(std::sync::atomic::Ordering::Relaxed),
            cfg!(feature = "candle-cuda"),
        );
        if charge_ram && !going_to_cpu {
            tracing::debug!(
                model = %model_id,
                "No usable GPU on this node — charging the model against the RAM budget"
            );
        }
        if charge_ram {
            let estimated = self.estimate_cpu_footprint_mb(model_id);
            // Refusing is the last resort: first reclaim memory from models
            // nothing is using, then ask again. Only then does the user see an
            // error.
            //
            // Ask ONCE unless the answer was no. `admit_to_cpu` charges the
            // reservation when it succeeds, so calling it a second time after a
            // success weighs the model against its own charge and refuses a
            // model that had already been let in.
            let mut admitted = self.admit_to_cpu(model_id, estimated);
            if !admitted {
                let freed = self.free_ram_for_admission(model_id, estimated).await;
                if freed > 0 {
                    tracing::info!(
                        model = %model_id,
                        freed_mb = freed,
                        "Reclaimed memory from unused models; retrying admission"
                    );
                }
                admitted = self.admit_to_cpu(model_id, estimated);
            }
            if !admitted {
                self.release_vram_charge(model_id);
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "inference",
                            "model_ram_refused",
                            format!(
                                "{} needs more memory than this node is allowed to use — not loading it",
                                model_id.0
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("warning", 8000),
                    );
                }
                let in_use = self.ram_committed_mb();
                // The figure the model was judged against RIGHT NOW, and why —
                // the cap, or the live anti-swap headroom.
                let (budget, note) = self
                    .ram_budget_now()
                    .map(|b| b.limiting_figure(in_use, estimated))
                    .unwrap_or_else(|| {
                        (
                            self.ram_budget_mb
                                .load(std::sync::atomic::Ordering::Relaxed),
                            self.ram_budget_note(),
                        )
                    });
                let message = match self.cpu_footprint_detail(model_id) {
                    Some((footprint, effective_context, source)) => {
                        crate::model::auto_manage::vram::describe_cpu_refusal(
                            &model_id.0,
                            &footprint,
                            effective_context,
                            source,
                            budget,
                            &note,
                            in_use,
                        )
                    }
                    None => format!(
                        "{} needs about {} MB of memory but this node's budget allows {} MB \
                         and {} MB is already in use. Raise `resources.max_ram_mb`, or free \
                         memory by unloading another model.",
                        model_id.0, estimated, budget, in_use,
                    ),
                };
                return Err(SwarmError::ServiceUnavailable(message));
            }
            self.record_cpu_kv_budget(model_id);
        }

        match self.spawn_worker(model_id).await {
            Ok(handle) => {
                // Reset the failure counter on first success.
                self.spawn_failures.remove(model_id);
                let handle = Arc::new(handle);
                self.workers.insert(model_id.clone(), handle.clone());
                Ok(handle)
            }
            Err(e) => {
                // The worker never started, so it owes nothing. Leaving the
                // charge would shrink the budget permanently on every failure.
                // Both devices: a CPU-bound spawn was charged RAM above.
                self.release_vram_charge(model_id);
                self.release_ram_charge(model_id);
                let count = self
                    .spawn_failures
                    .entry(model_id.clone())
                    .and_modify(|v| *v = (std::time::Instant::now(), v.1.saturating_add(1)))
                    .or_insert((std::time::Instant::now(), 1))
                    .1;
                let cooldown = spawn_failure_cooldown(count);
                // Severity from the classification, not from the call site:
                // most of `spawn_worker`'s arms are `ServiceUnavailable`
                // (socket bind, spawn, accept, init timeout), which is a 503
                // and a WARN. Logging every one at ERROR reports ordinary
                // transient spawn contention as this node being broken.
                crate::log_failure!(
                    &e,
                    model = %model_id,
                    consecutive_failures = count,
                    cooldown_secs = cooldown.as_secs(),
                    error = %e,
                    "Worker spawn failed — backing off"
                );
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "model",
                            "worker_spawn_failed",
                            format!(
                                "Worker spawn failed for {} (attempt #{}, cooldown {}s)",
                                model_id.0,
                                count,
                                cooldown.as_secs()
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("error", 6000),
                    );
                }
                Err(e)
            }
        }
    }

    async fn spawn_worker(&self, model_id: &ModelId) -> Result<WorkerHandle, SwarmError> {
        use interprocess::local_socket::{tokio::prelude::*, ListenerOptions};

        // Cross-platform socket naming:
        //  * Unix: filesystem path under `$TMPDIR/swarmllm-worker-<uuid>.sock`.
        //    `chmod 0o600` below restricts connect() to the current user.
        //  * Windows: namespace name `swarmllm-worker-<uuid>` (becomes
        //    `\\.\pipe\swarmllm-worker-<uuid>`). The default DACL on a named
        //    pipe grants access only to the current logon session — the
        //    equivalent of 0o600 for cross-user isolation.
        let uuid_str = uuid::Uuid::new_v4().to_string();
        #[cfg(unix)]
        let socket_name: String = std::env::temp_dir()
            .join(format!("swarmllm-worker-{uuid_str}.sock"))
            .to_str()
            .ok_or_else(|| SwarmError::Internal("socket path not UTF-8".into()))?
            .to_string();
        #[cfg(windows)]
        let socket_name: String = format!("swarmllm-worker-{uuid_str}");

        // RAII guard: remove the Unix socket file if spawn errors out partway.
        // Defused on success; WorkerHandle's Drop then owns the cleanup.
        // Windows named pipes are kernel-reclaimed — no guard needed.
        #[cfg(unix)]
        let socket_guard = {
            struct SocketCleanup(String);
            impl Drop for SocketCleanup {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(&self.0);
                }
            }
            SocketCleanup(socket_name.clone())
        };

        // Build the interprocess `Name` — filesystem path on Unix, namespace
        // name on Windows. The Name borrows from socket_name, which outlives
        // create_tokio() below.
        #[cfg(unix)]
        let ipc_name = socket_name
            .as_str()
            .to_fs_name::<interprocess::local_socket::GenericFilePath>()
            .map_err(|e| SwarmError::Internal(format!("ipc name: {e}")))?;
        #[cfg(windows)]
        let ipc_name = socket_name
            .as_str()
            .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
            .map_err(|e| SwarmError::Internal(format!("ipc name: {e}")))?;

        // SEC: On Unix, set umask(0o177) BEFORE bind so the socket file is
        // created with mode 0o600 atomically. The previous approach
        // (post-bind set_permissions) leaves a window where a local attacker
        // racing inotify on /tmp could connect first and impersonate the
        // worker, receiving plaintext prompts.
        //
        // umask is process-global; we save and restore it. Other threads
        // creating files during this brief window would see 0o600 perms
        // applied — that's strictly more restrictive, not less, so safe.
        // libc::umask is async-signal-safe and not blocking.
        #[cfg(unix)]
        let prev_umask = unsafe { libc::umask(0o177) };

        // Start listening before spawning so the worker can connect immediately.
        let listener = ListenerOptions::new()
            .name(ipc_name)
            .create_tokio()
            .map_err(|e| {
                #[cfg(unix)]
                unsafe {
                    libc::umask(prev_umask);
                }
                SwarmError::ServiceUnavailable(format!("socket bind: {e}"))
            })?;

        #[cfg(unix)]
        unsafe {
            libc::umask(prev_umask);
        }

        // Defense-in-depth chmod — covers any platform/filesystem where the
        // umask path didn't kick in (some FUSE mounts, weird umasks).
        // On Windows the default named-pipe DACL already scopes to the
        // current logon session — equivalent isolation, no extra call.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&socket_name, std::fs::Permissions::from_mode(0o600));
        }

        // Spawn the worker subprocess (same binary, model-worker subcommand)
        let exe = crate::current_exe_path()
            .map_err(|e| SwarmError::ServiceUnavailable(format!("current_exe: {e}")))?;
        let socket_str = socket_name.as_str();
        let data_dir_str = self
            .data_dir
            .to_str()
            .ok_or_else(|| SwarmError::Internal("data dir path is not valid UTF-8".into()))?;
        let mut args = vec![
            "model-worker".to_string(),
            "--socket".to_string(),
            socket_str.to_string(),
            "--data-dir".to_string(),
            data_dir_str.to_string(),
        ];

        // Give the worker the verbosity the daemon was started with. Inference
        // happens in here, so a daemon run with `-v` that leaves this process at
        // INFO is blind exactly where it matters most.
        let verbosity = crate::DAEMON_VERBOSITY.load(Ordering::Relaxed);
        for _ in 0..verbosity {
            args.push("-v".to_string());
        }

        // Pass KV-cache TTL from config
        let ttl = self
            .kv_cache_ttl_secs
            .load(std::sync::atomic::Ordering::Relaxed);
        args.push("--kv-cache-ttl".to_string());
        args.push(ttl.to_string());

        // Pass prefix-cache config from the active inference settings.
        {
            use std::sync::atomic::Ordering;
            args.push("--prefix-cache-enabled".to_string());
            args.push(
                self.prefix_cache_enabled
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-entries".to_string());
            args.push(
                self.prefix_cache_max_entries
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-prompt-tokens".to_string());
            args.push(
                self.prefix_cache_max_prompt_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-mb".to_string());
            args.push(self.prefix_cache_max_mb.load(Ordering::Relaxed).to_string());
            args.push("--prefix-cache-block-tokens".to_string());
            args.push(
                self.prefix_cache_block_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-min-tokens".to_string());
            args.push(
                self.prefix_cache_min_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-self-speculative".to_string());
            args.push(
                self.swift_self_speculative
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-calibration-tokens".to_string());
            args.push(
                self.swift_calibration_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-gamma".to_string());
            args.push(self.swift_gamma.load(Ordering::Relaxed).to_string());
            args.push("--swift-skip-ratio".to_string());
            let ratio = self.swift_skip_ratio_milli.load(Ordering::Relaxed) as f32 / 1000.0;
            args.push(format!("{ratio}"));
            args.push("--force-standard-attn".to_string());
            args.push(self.force_standard_attn.load(Ordering::Relaxed).to_string());
            let cap = self.max_seq_len_override.load(Ordering::Relaxed);
            if cap > 0 {
                args.push("--max-seq-len-override".to_string());
                args.push(cap.to_string());
            }
            args.push("--activation-compression".to_string());
            args.push(
                self.activation_compression
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            // BatchGenerate (Item 7): enabled iff the same `continuous_batching`
            // flag that gates daemon-side `forward()` coalescing is on.
            args.push("--batch-generate".to_string());
            args.push(self.continuous_batching.load(Ordering::Relaxed).to_string());
            args.push("--batch-generate-max-slots".to_string());
            args.push(
                self.max_concurrent_decode_batch
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefill-chunk-tokens".to_string());
            args.push(
                self.prefill_chunk_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--ngram-max-size".to_string());
            args.push(self.ngram_max_size.load(Ordering::Relaxed).to_string());
            args.push("--ngram-pred-tokens".to_string());
            args.push(self.ngram_pred_tokens.load(Ordering::Relaxed).to_string());
            args.push("--prefill-target-ms".to_string());
            args.push(self.prefill_target_ms.load(Ordering::Relaxed).to_string());
            args.push("--batched-prefill-forward".to_string());
            args.push(
                self.batched_prefill_forward
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
        }

        // Device placement. Before R146 this never reached the worker at all:
        // the split loader called `Device::cuda_if_available(0)` unconditionally
        // and `inference.gpu_layers` was read only by the legacy llama.cpp
        // executor, so a CUDA build ignored the setting entirely.
        args.push("--gpu-layers".to_string());
        args.push(self.effective_gpu_layers(model_id).to_string());
        if let Some(b) = self.cpu_kv_budget_bytes.get(model_id) {
            args.push("--kv-budget-bytes".to_string());
            args.push(b.value().to_string());
        }

        // If a shard window is set for this model, pass it to the worker
        if let Some(window) = self.active_shard_windows.get(model_id) {
            let window_str = window
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push("--shard-window".to_string());
            args.push(window_str);
        }

        // Bound the worker's CPU parallelism to what the owner agreed to give.
        //
        // candle parallelises CPU tensor ops through rayon, whose global pool
        // defaults to every logical core, and nothing narrowed it — so a single
        // request took the whole machine regardless of contribution level
        // (measured: 529-534% of 600% on a 6-core node set to Minimal). The
        // worker is a separate process, so setting this in its environment is
        // enough to size its rayon pool and cannot affect the daemon's own
        // runtime.
        //
        // Deliberately does NOT override an operator-set RAYON_NUM_THREADS:
        // someone who exported it has made exactly this decision already.
        let cpu_threads = self.cpu_threads.load(std::sync::atomic::Ordering::Relaxed);
        let mut command = tokio::process::Command::new(&exe);
        command.args(&args).kill_on_drop(true);
        if cpu_threads > 0 && std::env::var_os("RAYON_NUM_THREADS").is_none() {
            command.env("RAYON_NUM_THREADS", cpu_threads.to_string());
        }
        let child = command
            .spawn()
            .map_err(|e| SwarmError::ServiceUnavailable(format!("spawn worker: {e}")))?;

        // Wait for worker to connect
        let conn = tokio::time::timeout(
            std::time::Duration::from_secs(WORKER_CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .map_err(|_| SwarmError::ServiceUnavailable("worker connect timeout".into()))?
        .map_err(|e| SwarmError::ServiceUnavailable(format!("accept: {e}")))?;

        let (mut read_half, write_half) = conn.split();

        // Read Ready message
        let (ready_msg, _) = recv_worker(&mut read_half)
            .await
            .map_err(|e| SwarmError::ServiceUnavailable(format!("read ready: {e}")))?;
        match ready_msg {
            WorkerMsg::Ready => {}
            other => {
                return Err(SwarmError::ServiceUnavailable(format!(
                    "expected Ready, got {other:?}"
                )))
            }
        }

        tracing::info!(model_id = %model_id.0, "DIAG: model worker subprocess started");

        // Build a descriptive message including shard window if available
        let shard_info = self.active_shard_windows.get(model_id).map(|w| {
            let indices: Vec<_> = w.iter().map(|i| (i + 1).to_string()).collect();
            if indices.len() == 1 {
                format!("shard {}", indices[0])
            } else {
                format!("shards {}", indices.join(", "))
            }
        });
        match shard_info {
            Some(shards) => tracing::info!(
                model_id = %model_id,
                shards = %shards,
                "spawning worker"
            ),
            None => tracing::info!(model_id = %model_id, "spawning worker"),
        };

        // Success — defuse the cleanup guard on Unix; WorkerHandle now owns
        // the socket file and its Drop will unlink on process exit.
        #[cfg(unix)]
        std::mem::forget(socket_guard);
        let responses: ResponseMap = Arc::new(DashMap::new());
        let dead = Arc::new(AtomicBool::new(false));
        let reader_handle = tokio::spawn(reader_actor(
            read_half,
            responses.clone(),
            dead.clone(),
            model_id.clone(),
            self.prefix_manifest_tx.get().cloned(),
            self.prefix_probe_tx.get().cloned(),
            self.progress_tx.get().cloned(),
        ));
        Ok(WorkerHandle {
            child: Some(child),
            exited: Arc::new(AtomicBool::new(false)),
            writer: Mutex::new(write_half),
            responses,
            dead,
            #[cfg(unix)]
            socket_name,
            reader_handle,
            spawned_at: std::time::Instant::now(),
            last_used: AtomicU64::new(0),
        })
    }

    /// Send a LayerForward to the worker, get a LayerResult back.
    /// Send a single forward. When `continuous_batching` is on and the request
    /// is schedulable (decode-only, no vision/LoRA/spec/TP/pre-embedded), routes
    /// through the auto-coalescing scheduler which collects concurrent
    /// arrivals within `batch_collection_ms` and dispatches via
    /// `forward_batch`. Otherwise goes direct.
    pub async fn forward(
        &self,
        forward: crate::types::LayerForward,
    ) -> Result<crate::types::LayerResult, SwarmError> {
        if self
            .continuous_batching
            .load(std::sync::atomic::Ordering::Relaxed)
            && forward_is_schedulable(&forward)
        {
            if let Some(tx) = self.batch_scheduler.get() {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                if tx
                    .send(BatchSchedulerMsg::Forward {
                        fwd: forward,
                        resp_tx,
                    })
                    .await
                    .is_ok()
                {
                    return resp_rx.await.unwrap_or_else(|_| {
                        Err(SwarmError::Internal(
                            "batch scheduler dropped response".into(),
                        ))
                    });
                }
                // If the scheduler channel is closed, fall through to direct.
                return Err(SwarmError::Internal(
                    "batch scheduler channel closed".into(),
                ));
            }
        }
        self.forward_direct(forward).await
    }

    async fn forward_direct(
        &self,
        forward: crate::types::LayerForward,
    ) -> Result<crate::types::LayerResult, SwarmError> {
        // `None` means the caller had no request context — a segment served
        // for a REMOTE coordinator, whose parameters are not on the wire.
        // Defaulting there preserves the previous behaviour; see the field's
        // documentation on `LayerForward`.
        let forward_sampling = forward.sampling.clone().unwrap_or_default();
        let model_id = forward.model_id.clone();
        let handle = self.get_or_spawn(&model_id).await?;

        // Destructure to avoid cloning activations (can be large tensor data)
        let crate::types::LayerForward {
            request_id,
            sequence_num,
            index_pos,
            activations,
            format,
            model_id: fwd_model_id,
            layer_range,
            tp_meta,
            vision_embeddings,
            chain: _,
            sender_peer_bytes: _,
            requester_node_id,
            pre_embedded,
            generated_ids,
            adapter_id,
            draft_tokens,
            spec_logits_requested,
            truncate_kv_to,
            chunk_meta: _,
            sampling: _,
        } = forward;

        // Split vision embeddings out of the JSON header into the binary
        // payload prefix — serde_json encodes `Vec<u8>` as a JSON array of
        // integers (~5× bloat) and can push the header past `MAX_HEADER`.
        // Layout: `[vision_bytes][activation_bytes]` with
        // `vision_embeddings_len` recording the prefix length.
        let (vision_prefix, vision_len) = match vision_embeddings {
            Some(bytes) => {
                let len = u32::try_from(bytes.len()).map_err(|_| {
                    SwarmError::Internal("vision embeddings > u32::MAX bytes".into())
                })?;
                (bytes, len)
            }
            None => (Vec::new(), 0u32),
        };

        let ipc_fwd = IpcForward {
            request_id,
            sequence_num,
            index_pos,
            format,
            model_id: fwd_model_id,
            layer_range,
            tp_meta,
            vision_embeddings_len: vision_len,
            requester_node_id,
            pre_embedded,
            sampling: forward_sampling,
            adapter_id,
            draft_tokens,
            generated_ids,
            spec_logits_requested,
            truncate_kv_to,
        };

        if handle.dead.load(Ordering::Acquire) {
            self.workers.remove(&model_id);
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        // Register a response channel BEFORE sending so the reader actor can
        // route any early error/reply. Unregistered on drop via ResponseGuard.
        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let (mut guard, _) = handle.register_response(request_id, resp_tx, true);

        let payload_buf: Vec<u8> = if vision_len == 0 {
            activations
        } else {
            let mut buf = Vec::with_capacity(vision_prefix.len() + activations.len());
            buf.extend_from_slice(&vision_prefix);
            buf.extend_from_slice(&activations);
            buf
        };

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) =
                send_daemon(&mut *writer, &DaemonMsg::Forward(ipc_fwd), &payload_buf).await
            {
                drop(writer);
                self.workers.remove(&model_id);
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                // The worker never received the request; nothing to cancel.
                guard.disarm();
                return Err(SwarmError::ServiceUnavailable(format!("send Forward: {e}")));
            }
        }

        loop {
            match resp_rx.recv().await {
                Some((msg, payload)) => match msg {
                    WorkerMsg::LayerResult(r) if r.request_id == request_id => {
                        let (activations, spec_logits) = reconstruct_layer_payload(
                            r.has_activations,
                            r.has_spec_logits,
                            r.spec_logits_dims,
                            payload,
                        )?;
                        guard.disarm();
                        return Ok(crate::types::LayerResult {
                            request_id: r.request_id,
                            token_ids: r.token_ids,
                            finish_reason: r.finish_reason,
                            activations,
                            sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                            spec_logits,
                            matched_stop_sequence: r.matched_stop_sequence,
                            token_logprobs: r.logprobs.unwrap_or_default(),
                        });
                    }
                    WorkerMsg::Error {
                        request_id: rid,
                        message,
                        fatal,
                    } if rid == request_id => {
                        // Worker already reached a terminal state for this id.
                        guard.disarm();
                        return Err(self.classify_worker_error(&model_id, message, fatal));
                    }
                    _ => continue,
                },
                None => {
                    // Reader actor closed the channel — worker died while we were waiting.
                    // Subprocess lifecycle failure → ServiceUnavailable (per
                    // .claude/rules/completeness.md); Internal is for code bugs.
                    self.workers.remove(&model_id);
                    guard.disarm();
                    return Err(SwarmError::ServiceUnavailable(
                        "worker closed connection before reply".into(),
                    ));
                }
            }
        }
    }

    /// Batched forward: send N forwards in a single `DaemonMsg::BatchForward`
    /// message. Worker runs them through the fused `SplitModel::forward_batch`
    /// when they're compatible (same model/layer_range, decode-only, no vision
    /// /LoRA/spec/TP) for a real matmul-fusion speedup, otherwise falls back
    /// to the sequential path internally. Either way, callers get one
    /// `LayerResult` per input in the same order.
    ///
    /// All inputs must target the same `model_id`. Returns an error if the
    /// batch is empty or targets mixed models.
    pub async fn forward_batch(
        &self,
        forwards: Vec<crate::types::LayerForward>,
    ) -> Result<Vec<crate::types::LayerResult>, SwarmError> {
        if forwards.is_empty() {
            return Err(SwarmError::Validation("forward_batch: empty input".into()));
        }
        let model_id = forwards[0].model_id.clone();
        if forwards.iter().any(|f| f.model_id != model_id) {
            return Err(SwarmError::Validation(
                "forward_batch: all inputs must share model_id".into(),
            ));
        }
        let handle = self.get_or_spawn(&model_id).await?;
        if handle.dead.load(Ordering::Acquire) {
            self.workers.remove(&model_id);
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        // Register one response channel per request_id BEFORE sending.
        type SlotRx = mpsc::Receiver<(WorkerMsg, Vec<u8>)>;
        let n = forwards.len();
        let mut receivers: Vec<(Uuid, SlotRx)> = Vec::with_capacity(n);
        let mut guards: Vec<ResponseGuard> = Vec::with_capacity(n);
        for f in &forwards {
            let (tx, rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            // A BatchForward is one fused matmul over N requests; the worker
            // cannot skip a single member, so cancelling one is meaningless.
            // Matches `cancelled_request_id`'s exclusion.
            let (g, _) = handle.register_response(f.request_id, tx, false);
            guards.push(g);
            receivers.push((f.request_id, rx));
        }

        // Build IPC requests + concatenated activation payload.
        let mut ipc_requests: Vec<IpcForward> = Vec::with_capacity(n);
        let mut activation_lens: Vec<u32> = Vec::with_capacity(n);
        let mut concat_payload: Vec<u8> = Vec::new();
        for f in forwards {
            let crate::types::LayerForward {
                request_id,
                sequence_num,
                index_pos,
                activations,
                format,
                model_id,
                layer_range,
                tp_meta,
                // `forward_is_schedulable` / `batch_eligible` reject vision
                // forwards before they reach this path, so we assert here
                // rather than forward anything — any non-None value would
                // imply a scheduler bug.
                vision_embeddings: _,
                chain: _,
                sender_peer_bytes: _,
                requester_node_id,
                pre_embedded,
                generated_ids,
                adapter_id,
                draft_tokens,
                spec_logits_requested,
                truncate_kv_to,
                chunk_meta: _,
                // Per-item: a batch can carry forwards from different requests,
                // so the parameters travel with each one rather than being
                // taken from the batch head.
                sampling,
            } = f;
            let forward_sampling = sampling.unwrap_or_default();
            activation_lens.push(activations.len() as u32);
            concat_payload.extend_from_slice(&activations);
            ipc_requests.push(IpcForward {
                request_id,
                sequence_num,
                index_pos,
                format,
                model_id,
                layer_range,
                tp_meta,
                vision_embeddings_len: 0,
                requester_node_id,
                pre_embedded,
                sampling: forward_sampling,
                adapter_id,
                draft_tokens,
                generated_ids,
                spec_logits_requested,
                truncate_kv_to,
            });
        }

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) = send_daemon(
                &mut *writer,
                &DaemonMsg::BatchForward {
                    requests: ipc_requests,
                    activation_lens,
                },
                &concat_payload,
            )
            .await
            {
                drop(writer);
                self.workers.remove(&model_id);
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                return Err(SwarmError::ServiceUnavailable(format!(
                    "send BatchForward: {e}"
                )));
            }
        }

        // Collect results in the original request order. Each receiver fires
        // exactly once (LayerResult or Error) before being dropped.
        let mut results: Vec<crate::types::LayerResult> = Vec::with_capacity(n);
        for (rid, mut rx) in receivers {
            loop {
                match rx.recv().await {
                    Some((WorkerMsg::LayerResult(r), payload)) if r.request_id == rid => {
                        let (activations, spec_logits) = reconstruct_layer_payload(
                            r.has_activations,
                            r.has_spec_logits,
                            r.spec_logits_dims,
                            payload,
                        )?;
                        results.push(crate::types::LayerResult {
                            request_id: r.request_id,
                            token_ids: r.token_ids,
                            finish_reason: r.finish_reason,
                            activations,
                            sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                            spec_logits,
                            matched_stop_sequence: r.matched_stop_sequence,
                            token_logprobs: r.logprobs.unwrap_or_default(),
                        });
                        break;
                    }
                    Some((
                        WorkerMsg::Error {
                            request_id: err_rid,
                            message,
                            fatal,
                        },
                        _,
                    )) if err_rid == rid => {
                        return Err(self.classify_worker_error(&model_id, message, fatal));
                    }
                    Some(_) => continue,
                    None => {
                        // Subprocess lifecycle failure → ServiceUnavailable
                        // (mirrors the single-forward arm above).
                        self.workers.remove(&model_id);
                        return Err(SwarmError::ServiceUnavailable(
                            "worker closed connection during batch forward".into(),
                        ));
                    }
                }
            }
        }
        drop(guards);
        Ok(results)
    }

    /// Run full generation in the worker, streaming tokens back.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
        prompt: String,
        sampling: SamplingParams,
        request_id: uuid::Uuid,
        session_id: Option<String>,
        token_tx: Option<crate::inference::router::StreamingTokenTx>,
    ) -> Result<crate::inference::router::InferenceOutput, SwarmError> {
        // Track whether any token has been streamed to the caller, so a retry
        // can never duplicate output the client already saw.
        let emitted = std::sync::atomic::AtomicBool::new(false);
        let was_pinned = self.is_cpu_pinned(model_id);
        let first = self
            .generate_attempt(
                model_id,
                layer_range,
                prompt.clone(),
                sampling.clone(),
                request_id,
                session_id.clone(),
                token_tx.clone(),
                &emitted,
            )
            .await;
        match first {
            Ok(out) => Ok(out),
            Err(e) => {
                // A GPU OOM on the first (cold) request kills the worker and pins
                // the model to CPU. The request that TRIGGERED the OOM used to eat
                // the failure (0 tokens) while the very next request succeeded on
                // CPU. Retry that same request once — now it loads on CPU. The OOM
                // happens at load/prefill, so nothing was streamed yet (guarded by
                // `emitted`), making the retry duplicate-free (field report,
                // 2026-07-23, RTX 4050 6 GB).
                let freshly_pinned = !was_pinned && self.is_cpu_pinned(model_id);
                if freshly_pinned
                    && !emitted.load(std::sync::atomic::Ordering::Relaxed)
                    && matches!(e, SwarmError::ServiceUnavailable(_))
                {
                    tracing::warn!(
                        model = %model_id,
                        "Retrying request on CPU after a GPU out-of-memory auto-pinned the model"
                    );
                    let emitted_retry = std::sync::atomic::AtomicBool::new(false);
                    self.generate_attempt(
                        model_id,
                        layer_range,
                        prompt,
                        sampling,
                        request_id,
                        session_id,
                        token_tx,
                        &emitted_retry,
                    )
                    .await
                } else {
                    Err(e)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_attempt(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
        prompt: String,
        sampling: SamplingParams,
        request_id: uuid::Uuid,
        session_id: Option<String>,
        token_tx: Option<crate::inference::router::StreamingTokenTx>,
        emitted: &std::sync::atomic::AtomicBool,
    ) -> Result<crate::inference::router::InferenceOutput, SwarmError> {
        let handle = self.get_or_spawn(model_id).await?;

        // Kept for the post-generation stop-marker trim below; `sampling` is
        // moved into the IPC message.
        let stop_sequences = sampling.stop.clone();
        // Same reason as `stop_sequences`: `sampling` is moved below, and the
        // short-reply report needs to tell "asked for one token" from "stopped
        // after one".
        let requested_max_tokens = sampling.max_tokens;

        let gen = IpcGenerate {
            request_id,
            model_id: model_id.clone(),
            layer_range,
            prompt,
            sampling,
            session_id,
        };

        if handle.dead.load(Ordering::Acquire) {
            self.workers.remove(model_id);
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let (mut guard, _) = handle.register_response(request_id, resp_tx, true);
        let attempt_token = guard.token;

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) = send_daemon(&mut *writer, &DaemonMsg::Generate(gen), &[]).await {
                drop(writer);
                self.workers.remove(model_id);
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                // The worker never received the request; nothing to cancel.
                guard.disarm();
                return Err(SwarmError::ServiceUnavailable(format!(
                    "send Generate: {e}"
                )));
            }
        }

        let mut content = String::new();
        #[allow(unused_assignments)]
        let mut prompt_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut completion_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut finish_reason = String::new();
        let mut token_logprobs: Vec<crate::inference::router::TokenLogProbEntry> = Vec::new();
        #[allow(unused_assignments)]
        let mut matched_stop_sequence: Option<String> = None;

        loop {
            let (msg, _) = match resp_rx.recv().await {
                Some(v) => v,
                None => {
                    guard.disarm();
                    // A closed channel has two very different causes. If a
                    // later attempt has taken over this request_id, the worker
                    // is fine and is still computing for that attempt —
                    // evicting it here would destroy a healthy worker (forcing
                    // a full model reload) and abort the attempt that replaced
                    // us.
                    if handle.response_superseded(request_id, attempt_token) {
                        return Err(SwarmError::ServiceUnavailable(
                            "superseded by a retry of the same request".into(),
                        ));
                    }
                    // Subprocess lifecycle failure → ServiceUnavailable.
                    // Was Internal; operators saw 500s here and misattributed
                    // them to code bugs rather than worker crashes.
                    self.workers.remove(model_id);
                    return Err(SwarmError::ServiceUnavailable(
                        "worker closed connection mid-generate".into(),
                    ));
                }
            };
            match msg {
                WorkerMsg::Token {
                    request_id: rid,
                    text,
                    is_eos,
                    logprob,
                    token_id: _,
                } if rid == request_id => {
                    if !is_eos {
                        content.push_str(&text);
                        if let Some(lp) = logprob {
                            token_logprobs.push(crate::inference::router::TokenLogProbEntry {
                                token: text.clone(),
                                logprob: lp,
                                top_logprobs: Vec::new(),
                            });
                        }
                        if let Some(ref tx) = token_tx {
                            // Mark output as delivered so the OOM CPU-retry in the
                            // `generate` wrapper can't re-run and duplicate tokens.
                            emitted.store(true, std::sync::atomic::Ordering::Relaxed);
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: text.clone(),
                                    finish_reason: None,
                                    matched_stop_sequence: None,
                                })
                                .await;
                        }
                    }
                }
                WorkerMsg::GenerateDone {
                    request_id: rid,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    finish_reason: fr,
                    matched_stop_sequence: ms,
                } if rid == request_id => {
                    prompt_tokens = pt as u32;
                    completion_tokens = ct as u32;
                    finish_reason = fr;
                    matched_stop_sequence = ms;
                    // GenerateDone is terminal — the worker is finished.
                    guard.disarm();
                    break;
                }
                WorkerMsg::Error {
                    request_id: rid,
                    message,
                    fatal,
                } if rid == request_id => {
                    guard.disarm();
                    return Err(self.classify_worker_error(model_id, message, fatal));
                }
                _ => continue,
            }
        }

        if let Some(ref tx) = token_tx {
            let _ = tx
                .send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some(finish_reason.clone()),
                    matched_stop_sequence: matched_stop_sequence.clone(),
                })
                .await;
        }

        // Strip a trailing PARTIAL stop marker before returning.
        //
        // The worker checks stop strings against its accumulated text and
        // withholds the token that completes a match — but a marker like
        // `<|im_end|>` is decoded across several tokens, and the earlier pieces
        // (`<|im_end`, `|`) match nothing yet, so they were already sent. The
        // consumer therefore ends up holding a partial marker.
        //
        // `local_exec.rs` and `distributed.rs` already do this; the split path
        // — everything routed through a worker subprocess, which is every
        // locally-held model — did not, which is why a Llama-3.2 GGUF emitting
        // ChatML markers returned a bare `<|im_end|` as its entire reply across
        // three releases that each claimed to fix it (external report
        // 2026-07-25).
        //
        // Also trims a COMPLETE marker: `find_stop_sequence` uses `contains`,
        // so a match can sit anywhere in the accumulated text, and the token
        // that completed it is withheld rather than the marker being removed.
        crate::inference::finalize_reply_text(&mut content, &stop_sequences);
        crate::inference::report_short_reply(
            &request_id,
            completion_tokens,
            requested_max_tokens,
            matched_stop_sequence.as_deref(),
        );

        Ok(crate::inference::router::InferenceOutput {
            request_id,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            session_id: None,
            token_logprobs,
            matched_stop_sequence,
            trace: None,
        })
    }

    /// Turn a `WorkerMsg::Error` into a `SwarmError`, evicting the worker first
    /// when the failure was fatal to its device state.
    ///
    /// Before this existed, a worker that hit a CUDA OOM mid-forward reported
    /// the error and stayed resident holding its full VRAM allocation — the
    /// only two code paths that ever killed a worker were explicit unloads.
    /// Each retry then had *less* free VRAM than the last, so a single OOM
    /// reliably cascaded into permanent failure for that model until someone
    /// killed the process by hand.
    ///
    /// Eviction drops the last `Arc<WorkerHandle>` once the caller's clone goes
    /// out of scope, and `WorkerHandle::Drop` kills the child, which is what
    /// actually returns the VRAM to the OS.
    fn classify_worker_error(
        &self,
        model_id: &ModelId,
        message: String,
        fatal_flag: bool,
    ) -> SwarmError {
        // Trust the worker's own verdict, but re-derive it from the message as
        // well: a worker binary older than the `fatal` field always sends
        // `false`, and a stranded worker is much worse than a needless respawn.
        if fatal_flag || crate::inference::worker_ipc::worker_error_is_fatal(&message) {
            self.workers.remove(model_id);
            tracing::warn!(
                model = %model_id,
                error = %message,
                "Fatal worker error — killing worker to reclaim its device memory"
            );

            // Some GPU failures repeat verbatim on the respawn — same model,
            // same device, same outcome — so retrying on the GPU just burns
            // another load. Pin the model to the CPU: slower, but it answers.
            let was_on_gpu = self.effective_gpu_layers(model_id) != 0;
            let permanent = crate::inference::worker_ipc::permanent_gpu_failure(&message);
            if let (true, Some(kind)) = (was_on_gpu, permanent) {
                use crate::inference::worker_ipc::PermanentGpuFailure;
                self.cpu_pinned_models.insert(model_id.clone());
                let reason = match kind {
                    PermanentGpuFailure::OutOfMemory => "GPU out of memory",
                    PermanentGpuFailure::ArchitectureTooOld => {
                        "GPU is older than this build's kernels"
                    }
                    PermanentGpuFailure::NoKvCacheRoom => {
                        "no GPU memory left for the conversation cache"
                    }
                };
                tracing::warn!(
                    model = %model_id,
                    reason,
                    "Pinning this model to CPU for the rest of this run"
                );
                if let Some(tx) = self.activity_tx.get() {
                    // Distinct `kind` per cause, NOT one kind with two
                    // messages: the frontend translates by kind
                    // (`I18n.t('activity.' + kind)`) and only falls back to
                    // this English text when the key is missing. Reusing
                    // `model_cpu_fallback` would have told everyone outside
                    // English that they had run out of GPU memory — sending
                    // them to free VRAM that was never the problem.
                    let (event_kind, text) = match kind {
                        // `NoKvCacheRoom` deliberately SHARES this kind rather
                        // than getting its own. The rule above is that a
                        // distinct cause needs a distinct kind because the
                        // frontend translates by kind — but the reason it
                        // exists is that the arch case sends people to free
                        // VRAM that was never the problem. Here VRAM IS the
                        // problem and freeing it IS the remedy, so this text is
                        // already correct and already translated; a second key
                        // saying the same thing in 21 locales would be
                        // duplication, not precision.
                        PermanentGpuFailure::OutOfMemory | PermanentGpuFailure::NoKvCacheRoom => (
                            "model_cpu_fallback",
                            format!(
                                "{} ran out of GPU memory — switched to CPU (slower, but working)",
                                model_id.0
                            ),
                        ),
                        // Phrased for someone who will not know what a compute
                        // capability is, and who needs to know the node is
                        // still working rather than what CUDA reported.
                        PermanentGpuFailure::ArchitectureTooOld => (
                            "model_cpu_fallback_gpu_too_old",
                            format!(
                                "This graphics card is too old for this version's GPU \
                                 support — {} switched to CPU (slower, but working)",
                                model_id.0
                            ),
                        ),
                    };
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new("inference", event_kind, text)
                            .with_model(model_id.0.clone())
                            .with_toast("warning", 8000),
                    );
                }
            }

            // Subprocess lifecycle failure → ServiceUnavailable, not Internal:
            // this server can't serve right now, but it isn't a code bug.
            return SwarmError::ServiceUnavailable(format!("worker fatal error: {message}"));
        }

        // A worker-side VALIDATION failure is the caller's input, not our bug.
        //
        // The worker raises a real `SwarmError::Validation` for things like a
        // prompt longer than the model's context window, but crossing the IPC
        // boundary flattens it to a string — so everything arrived here as
        // `Inference`, which maps to HTTP 500 `server_error`. A prompt that is
        // too long is the one thing the user CAN fix, and telling them the
        // server broke is both wrong and actively harmful: any client with
        // retry-on-5xx will re-send a request that cannot ever succeed.
        //
        // Re-deriving the class from the message is the same approach
        // `worker_error_is_fatal` already takes for the same reason (the typed
        // information does not survive the hop). Observed against a 2176-token
        // prompt on a 2048-context model, 2026-07-29.
        // One recovery, shared with the network boundary, which has the same
        // problem for the same reason (`crate::error::reclassify_flattened_error`).
        if let Some(recovered) = crate::error::reclassify_flattened_error(&message) {
            return recovered;
        }
        SwarmError::Inference(message)
    }

    /// Tell every live worker to abandon `request_id`.
    ///
    /// Used for cancellations that arrive over the network
    /// (`SwarmMessage::CancelInference`), where the sender knows only the
    /// request id — the coordinator's id space is global, ours is per-worker.
    /// Workers are few (one per loaded model) and a cancel for an unknown id
    /// is a no-op on the worker side, so the fan-out is cheap and safe.
    ///
    /// Locally-originated cancels don't need this: `ResponseGuard` knows its
    /// own worker and messages it directly on drop.
    pub async fn cancel_request(&self, request_id: Uuid) {
        // Collect handles before awaiting — never hold a DashMap ref across an
        // await point.
        let workers: Vec<Arc<WorkerHandle>> = self
            .workers
            .iter()
            .filter(|e| !e.value().dead.load(Ordering::Acquire))
            .map(|e| e.value().clone())
            .collect();
        if workers.is_empty() {
            return;
        }
        for worker in workers {
            let mut writer = worker.writer.lock().await;
            let _ = send_daemon(&mut *writer, &DaemonMsg::CancelRequest { request_id }, &[]).await;
        }
    }

    /// Unload all segments for a model (kills the worker subprocess).
    pub async fn unload_model(&self, model_id: &ModelId) {
        // Was this worker holding GPU memory? Only then does killing it change
        // the pressure that caused any outstanding CPU pins.
        let freed_gpu_memory = self.effective_gpu_layers(model_id) != 0;
        if let Some((_, handle)) = self.workers.remove(model_id) {
            // Try graceful shutdown first
            if let Ok(mut writer) = handle.writer.try_lock() {
                let _ = send_daemon(&mut *writer, &DaemonMsg::Shutdown, &[]).await;
            }
            // Dropping the handle signals the child; it does NOT wait for the
            // OS to reclaim its device memory. Hold the budget until the
            // process is genuinely gone, or the next admission decides against
            // memory that is still occupied and its worker dies with
            // CUDA_ERROR_OUT_OF_MEMORY — the exact failure admission exists to
            // prevent.
            let exited = handle.exited.clone();
            drop(handle);
            let freed_cleanly = Self::await_worker_exit(&exited, WORKER_EXIT_WAIT).await;
            if !freed_cleanly {
                // Release anyway. Holding a charge forever for a process that
                // will not die would refuse every later load on this device,
                // which is a worse failure than the race being closed here.
                tracing::warn!(
                    model_id = %model_id,
                    waited_ms = WORKER_EXIT_WAIT.as_millis(),
                    "Worker did not exit in time — freeing its memory budget anyway; \
                     a load starting now could still find the device occupied"
                );
            }
            self.release_vram_charge(model_id);
            // The process is gone, so it holds neither device's memory. A CPU
            // worker never had a VRAM charge and vice versa; releasing both is
            // correct and keeps the two budgets from drifting on churn.
            self.release_ram_charge(model_id);
            tracing::info!(model_id = %model_id, "Model worker killed, GPU memory freed");

            // A GPU OOM pins its model to the CPU, and that pin used to last for
            // the life of the process — a ~10x throughput loss that no API
            // response mentioned, triggered by nothing more than the daemon's
            // own background model churn. The pin's reasoning ("the OOM will
            // repeat verbatim on respawn") only held while VRAM pressure was
            // static, which it was, because eviction never actually freed
            // anything. Now that it does, releasing memory is exactly the event
            // that makes a retry worth it — the same condition
            // `clear_cpu_pin`'s own documentation names.
            //
            // Worst case a pinned model retries the GPU, OOMs again and re-pins,
            // costing one model load. That is a far better trade than staying on
            // the CPU for ever.
            if freed_gpu_memory && !self.cpu_pinned_models.is_empty() {
                let lifted: Vec<String> = self
                    .cpu_pinned_models
                    .iter()
                    .map(|m| m.key().0.clone())
                    .collect();
                self.cpu_pinned_models.clear();
                tracing::info!(
                    freed_by = %model_id,
                    lifted = ?lifted,
                    "GPU memory freed — clearing CPU pins so these models may use the GPU again"
                );
            }
        }
    }

    /// Unload a model and clear its shard window so next spawn uses defaults.
    pub async fn unload_and_clear_window(&self, model_id: &ModelId) {
        self.unload_model(model_id).await;
        self.clear_shard_window(model_id);
    }

    /// Check if a worker is running for a model.
    pub fn is_loaded(&self, model_id: &ModelId) -> bool {
        self.workers.contains_key(model_id)
    }

    /// List all currently loaded model IDs.
    pub fn loaded_model_ids(&self) -> Vec<ModelId> {
        self.workers.iter().map(|e| e.key().clone()).collect()
    }

    /// Models with at least one request in flight *right now*.
    ///
    /// Read from the pool's own per-request response channels, which is the one
    /// place EVERY execution path passes through — local, distributed, and
    /// peer-served alike. The callers' own bookkeeping does not have that
    /// property: `active_pipelines` is populated only by the distributed path
    /// (`distributed_exec` inserts; `local_exec` only removes) and
    /// `serving_models` only by peer-served work, so a node answering its own
    /// client locally appeared in NEITHER. That is how a worker was killed
    /// mid-generation seven seconds after loading (reported 2026-07-31).
    ///
    /// Anything asking "is this model busy?" should use this rather than
    /// re-deriving it from a caller-side map that covers one path.
    pub fn models_with_inflight_requests(&self) -> Vec<ModelId> {
        self.workers
            .iter()
            .filter(|e| !e.value().responses.is_empty())
            .map(|e| e.key().clone())
            .collect()
    }

    /// How long each loaded model's worker has existed. Upper bound on how long
    /// it can possibly have been idle.
    pub fn model_residency_secs(&self) -> Vec<(ModelId, u64)> {
        self.workers
            .iter()
            .map(|e| (e.key().clone(), e.value().spawned_at.elapsed().as_secs()))
            .collect()
    }

    /// Restart a model's worker with a new shard window.
    /// Kills the current worker → OS/CUDA frees VRAM → next inference request
    /// triggers `get_or_spawn` which reads the new window.
    pub async fn restart_with_window(&self, model_id: &ModelId, window: Vec<u32>) {
        tracing::info!(
            model_id = %model_id,
            window = ?window,
            "Restarting worker with narrower shard window"
        );
        self.active_shard_windows.insert(model_id.clone(), window);
        // Kill the existing worker — next request will re-spawn with new window
        self.unload_model(model_id).await;
    }

    /// Clear a shard window (revert to loading all on-disk shards).
    fn clear_shard_window(&self, model_id: &ModelId) {
        self.active_shard_windows.remove(model_id);
    }

    /// Get the current shard window for a model, if any.
    pub fn get_shard_window(&self, model_id: &ModelId) -> Option<Vec<u32>> {
        self.active_shard_windows.get(model_id).map(|v| v.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a guard the way `register_response` does, for map-level tests.
    fn guard_for(map: &ResponseMap, request_id: Uuid, token: u64) -> ResponseGuard {
        ResponseGuard {
            responses: map.clone(),
            request_id,
            token,
            worker: None,
        }
    }

    fn dummy_tx() -> ResponseTx {
        mpsc::channel(1).0
    }

    #[test]
    fn second_attempt_on_same_request_id_displaces_the_first() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, d1) = claim_response_slot(&map, rid, dummy_tx());
        assert_eq!(d1, None, "first claim displaces nothing");
        let (t2, d2) = claim_response_slot(&map, rid, dummy_tx());
        assert_eq!(d2, Some(t1), "retry reports the attempt it superseded");
        assert_ne!(t1, t2, "attempts get distinct tokens");
    }

    #[test]
    fn superseded_attempt_is_distinguishable_from_a_dead_worker() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        // Nobody has taken over yet: a closed channel here really is a dead worker.
        assert!(!response_slot_superseded(&map, rid, t1));
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());
        // Now attempt 1 is superseded, attempt 2 is not.
        assert!(response_slot_superseded(&map, rid, t1));
        assert!(!response_slot_superseded(&map, rid, t2));
    }

    /// The live failure: a retry displaced the original, then the original's
    /// cleanup removed the *retry's* channel, so both attempts died and a
    /// healthy worker was evicted.
    #[test]
    fn superseded_attempt_cleanup_does_not_evict_its_successor() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        let g1 = guard_for(&map, rid, t1);
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());

        drop(g1); // attempt 1 gives up after being displaced

        let held = map.get(&rid).map(|e| e.value().0);
        assert_eq!(held, Some(t2), "the retry's channel must survive");
    }

    #[test]
    fn last_attempt_cleanup_clears_the_slot() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());
        drop(guard_for(&map, rid, t1));
        drop(guard_for(&map, rid, t2));
        assert!(map.get(&rid).is_none(), "no entry may be leaked");
    }

    #[test]
    fn unrelated_request_ids_are_untouched() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let (ta, _) = claim_response_slot(&map, a, dummy_tx());
        let (tb, _) = claim_response_slot(&map, b, dummy_tx());
        drop(guard_for(&map, a, ta));
        assert!(map.get(&a).is_none());
        assert_eq!(map.get(&b).map(|e| e.value().0), Some(tb));
    }

    fn test_pool() -> ModelProcessPool {
        ModelProcessPool::new(std::path::PathBuf::from("/tmp/swarmllm-test-pool"))
    }

    #[test]
    fn ram_is_charged_wherever_the_model_can_only_land_in_ram() {
        // Sent to the CPU explicitly or by VRAM refusal: charged.
        assert!(charges_ram(true, true, true));
        // GPU present and usable, model going to it: not charged.
        assert!(!charges_ram(false, true, true));
        // No GPU detected: charged — this is every CPU-only node, which used
        // to skip RAM admission entirely (2026-08-21).
        assert!(charges_ram(false, false, true));
        // A build without CUDA cannot put the model anywhere else.
        assert!(charges_ram(false, true, false));
    }

    #[test]
    fn effective_gpu_layers_defaults_to_auto() {
        let pool = test_pool();
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), -1);
    }

    #[test]
    fn effective_gpu_layers_follows_config() {
        let pool = test_pool();
        pool.set_gpu_layers(0);
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), 0);
        pool.set_gpu_layers(20);
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), 20);
    }

    #[test]
    fn cpu_pin_overrides_config_per_model() {
        // After a GPU OOM the pinned model must respawn on CPU even though the
        // config says otherwise — otherwise the respawned worker makes the
        // identical allocation and dies the same way, forever.
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let pinned = ModelId("oom-model".into());
        let other = ModelId("healthy-model".into());
        pool.cpu_pinned_models.insert(pinned.clone());

        assert_eq!(pool.effective_gpu_layers(&pinned), 0);
        assert_eq!(pool.effective_gpu_layers(&other), -1);
        assert!(pool.is_cpu_pinned(&pinned));
        assert!(!pool.is_cpu_pinned(&other));

        assert!(pool.clear_cpu_pin(&pinned));
        assert_eq!(pool.effective_gpu_layers(&pinned), -1);
        assert!(!pool.clear_cpu_pin(&pinned), "second clear is a no-op");
    }

    /// A pin lasted the life of the process because `clear_cpu_pin` had no
    /// caller — a ~10x throughput loss triggered by the daemon's own background
    /// churn, with nothing in the API to show it. Freeing GPU memory is the
    /// event that makes a retry worthwhile.
    #[tokio::test]
    async fn unloading_a_gpu_worker_lifts_cpu_pins() {
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let pinned = ModelId("oom-model".into());
        pool.cpu_pinned_models.insert(pinned.clone());
        assert!(pool.is_cpu_pinned(&pinned));

        // No worker registered for this id, but it is not CPU-pinned itself, so
        // it counts as a GPU tenant whose removal frees memory.
        pool.unload_model(&ModelId("some-other-gpu-model".into()))
            .await;

        assert!(
            pool.is_cpu_pinned(&pinned),
            "no worker existed, so nothing was actually freed and the pin stands"
        );
    }

    /// Unloading a model that was ITSELF pinned to the CPU frees no GPU memory,
    /// so it must not trigger a retry storm for everything else.
    #[tokio::test]
    async fn unloading_a_cpu_pinned_worker_does_not_lift_other_pins() {
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let a = ModelId("cpu-bound".into());
        let b = ModelId("also-pinned".into());
        pool.cpu_pinned_models.insert(a.clone());
        pool.cpu_pinned_models.insert(b.clone());

        pool.unload_model(&a).await;
        assert!(
            pool.is_cpu_pinned(&b),
            "unloading a CPU worker frees no VRAM, so other pins must stand"
        );
    }

    /// `resources.max_ram_mb` shipped documented as "0 = auto (50% of system
    /// RAM)" while nothing read it, so a node had no memory ceiling at all.
    /// Admission now behaves like its GPU sibling.
    /// The reporter's machine: cap 18000 MB, 14773 MB free at the time, a
    /// 13149 MB model. The cap allows it; the live headroom (70% of free) does
    /// not — and the same pool admits it once memory frees up, because the
    /// provider is asked every time rather than once at startup.
    #[test]
    fn a_live_headroom_refuses_what_the_cap_would_allow_and_relents_when_memory_frees() {
        use crate::model::auto_manage::vram::RamBudget;
        let free = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(14773));
        let p = test_pool();
        let f = free.clone();
        p.set_ram_budget_provider(Box::new(move || {
            Some(RamBudget::from_machine(
                18000,
                18000,
                32000,
                f.load(std::sync::atomic::Ordering::Relaxed),
            ))
        }));
        let m = ModelId("llama-8b".into());
        assert!(
            !p.admit_to_cpu(&m, 13149),
            "10341 MB of headroom cannot take 13149 MB"
        );
        free.store(26000, std::sync::atomic::Ordering::Relaxed);
        assert!(p.admit_to_cpu(&m, 13149), "18200 MB of headroom can");
        p.release_ram_charge(&m);
    }

    #[test]
    fn ram_admission_refuses_once_the_budget_is_committed() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        assert!(p.admit_to_cpu(&ModelId("first".into()), 4000));
        assert!(
            !p.admit_to_cpu(&ModelId("second".into()), 2500),
            "4000 + 2500 exceeds 6000 — must refuse rather than swap"
        );
        // Something that does fit in the remainder is still admitted.
        assert!(p.admit_to_cpu(&ModelId("small".into()), 1500));
    }

    /// Releasing must credit the budget back, or churn shrinks it permanently.
    #[test]
    fn releasing_a_ram_charge_frees_the_budget_again() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        let a = ModelId("a".into());
        assert!(p.admit_to_cpu(&a, 5000));
        assert!(!p.admit_to_cpu(&ModelId("b".into()), 2000));
        p.release_ram_charge(&a);
        assert!(
            p.admit_to_cpu(&ModelId("b".into()), 2000),
            "the freed charge must be usable again"
        );
    }

    /// No budget configured, or an unreadable header (estimate 0), must not
    /// start refusing loads — that would be a worse failure than the one being
    /// prevented, and mirrors how `admit_to_gpu` treats the same gap.
    #[test]
    fn ram_admission_does_not_judge_what_it_cannot_measure() {
        let p = test_pool();
        // Budget unset.
        assert!(p.admit_to_cpu(&ModelId("nobudget".into()), 99_999));

        let p2 = test_pool();
        p2.set_ram_budget_mb(1);
        assert!(
            p2.admit_to_cpu(&ModelId("mystery".into()), 0),
            "an unreadable model must not be refused on a guess"
        );
    }

    #[test]
    fn cpu_pinned_ids_are_reported_for_the_api() {
        let pool = test_pool();
        pool.cpu_pinned_models.insert(ModelId("zeta".into()));
        pool.cpu_pinned_models.insert(ModelId("alpha".into()));
        assert_eq!(pool.cpu_pinned_model_ids(), vec!["alpha", "zeta"]);
        pool.clear_cpu_pin(&ModelId("alpha".into()));
        assert_eq!(pool.cpu_pinned_model_ids(), vec!["zeta"]);
    }

    /// The full classification, not just the string helper: a GPU with no room
    /// left must come back as `ServiceUnavailable` so the router can hand the
    /// request to a peer. As `Inference` it was HTTP 500, which a coordinator
    /// reads as "this node is broken" rather than "ask someone else" — costing
    /// an answer another peer could have given.
    ///
    /// Not marked fatal: the worker is fine, the GPU is merely full, so it must
    /// keep its worker rather than being evicted and reloaded.
    #[test]
    fn a_full_gpu_is_classified_as_service_unavailable_not_an_inference_fault() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "prefill forward: Service unavailable: Not enough free GPU memory to continue \
             this conversation (168 MB of KV cache already in use, budget 323 MB)."
                .into(),
            false,
        );
        assert!(
            matches!(err, SwarmError::ServiceUnavailable(_)),
            "a full GPU must be 503 so the router re-routes, got {err:?}"
        );
    }

    #[test]
    fn fatal_error_evicts_worker_and_returns_service_unavailable() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))".into(),
            true,
        );
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));
        // OOM on a GPU-eligible model pins it to CPU for the next spawn.
        assert!(pool.is_cpu_pinned(&model));
    }

    #[test]
    fn non_fatal_error_stays_inference_and_leaves_model_gpu_eligible() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(&model, "Tokenize: bad prompt".into(), false);
        assert!(matches!(err, SwarmError::Inference(_)));
        assert!(!pool.is_cpu_pinned(&model));
    }

    #[test]
    fn fatal_non_oom_error_does_not_pin_to_cpu() {
        // An illegal memory access is fatal to the worker but is not evidence
        // that the GPU is too small — retrying on the GPU is right here.
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "cuda error: an illegal memory access was encountered".into(),
            true,
        );
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));
        assert!(!pool.is_cpu_pinned(&model));
    }
}

#[cfg(test)]
mod validation_detail_tests {
    use crate::error::{reclassify_flattened_error, SwarmError};

    /// These live here because this is the boundary they were written against,
    /// but the recovery is now shared with the network boundary — see
    /// `crate::error::reclassify_flattened_error`.
    fn model_unavailable_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::ModelNotAvailable(id)) => Some(id.0),
            _ => None,
        }
    }
    fn service_unavailable_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::ServiceUnavailable(d)) => Some(d),
            _ => None,
        }
    }
    fn validation_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::Validation(d)) => Some(d),
            _ => None,
        }
    }

    /// A model whose files are gone must reach the caller as 404, not 500.
    /// Reported with a real trace: the message plainly said "Model not
    /// available" and the client still got `server_error`.
    #[test]
    fn a_missing_model_is_recovered_from_a_flattened_message() {
        let msg = "Inference error: Model not available: Manifest not found: \
                   /home/u/.local/share/swarmllm/models/qwen2.5-0.5b/manifest.json";
        let got = model_unavailable_detail(msg).expect("should be recognised");
        assert!(got.starts_with("Manifest not found:"));
    }

    /// Doubly-wrapped messages happen (worker → pipeline → router). Take the
    /// innermost marker so the id survives rather than a wrapper fragment.
    #[test]
    fn the_innermost_marker_wins() {
        let msg = "Worker: Model not available: outer: Model not available: real-model-id";
        assert_eq!(
            model_unavailable_detail(msg).as_deref(),
            Some("real-model-id")
        );
    }

    /// A GPU that cannot fit another conversation must reach the caller as 503,
    /// not 500 — a coordinator re-routes on 503 and gives up on 500, so the
    /// wrong status costs an answer another peer could have given.
    ///
    /// Observed with two processes sharing one GPU: `Inference error: prefill
    /// forward: Service unavailable: Not enough free GPU memory to continue
    /// this conversation (168 MB of KV cache already in use, budget 323 MB)` —
    /// served as HTTP 500 with the words "Service unavailable" inside it.
    #[test]
    fn a_gpu_out_of_room_is_recovered_as_service_unavailable() {
        let msg = "Inference error: prefill forward: Service unavailable: Not enough free \
                   GPU memory to continue this conversation (168 MB of KV cache already in \
                   use, budget 323 MB).";
        let got = service_unavailable_detail(msg).expect("should be recognised");
        assert!(
            got.starts_with("Not enough free GPU memory"),
            "must keep the actionable part, got {got:?}"
        );
    }

    /// The innermost marker wins, as with its two siblings — a doubly-wrapped
    /// message must still yield the original reason rather than a wrapper.
    #[test]
    fn the_innermost_service_unavailable_marker_wins() {
        let msg = "outer: Service unavailable: mid: Service unavailable: the real reason";
        assert_eq!(
            service_unavailable_detail(msg).as_deref(),
            Some("the real reason")
        );
    }

    /// An ordinary failure must NOT be reclassified as unavailable — that would
    /// turn a genuine bug into a "try again later" and hide it.
    #[test]
    fn an_ordinary_failure_is_not_reclassified_as_unavailable() {
        assert!(service_unavailable_detail("Inference error: tensor shape mismatch").is_none());
        assert!(service_unavailable_detail("Service unavailable: ").is_none());
    }

    /// An ordinary inference failure must NOT be reclassified as a missing
    /// model — that would turn a real server fault into a 404 and hide it.
    #[test]
    fn an_unrelated_failure_is_left_alone() {
        assert!(model_unavailable_detail("CUDA out of memory").is_none());
        assert!(model_unavailable_detail("Model not available: ").is_none());
    }

    /// The case this exists for: an over-long prompt must reach the user as
    /// their problem, with the sentence that says how to fix it intact.
    #[test]
    fn a_context_overflow_is_recognised_and_unwrapped() {
        let msg = "prefill forward: Validation error: Sequence length (2176) exceeds \
                   model context window (2048). Reduce your prompt or max_tokens.";
        let got = validation_detail(msg).expect("should be recognised as validation");
        assert!(got.starts_with("Sequence length (2176) exceeds"));
        assert!(got.ends_with("Reduce your prompt or max_tokens."));
        assert!(
            !got.contains("prefill forward"),
            "call-site plumbing must be dropped"
        );
    }

    /// Genuine faults must keep mapping to a server error — this must not
    /// become a way to relabel our own bugs as the caller's fault.
    #[test]
    fn other_worker_errors_are_not_treated_as_validation() {
        assert!(validation_detail("Internal error: blk.0.attn_q: missing region").is_none());
        assert!(validation_detail("CUDA_ERROR_OUT_OF_MEMORY").is_none());
        assert!(validation_detail("worker closed connection mid-generate").is_none());
    }

    /// A marker with nothing after it carries no information for the user, so
    /// it stays an inference error rather than becoming an empty 400.
    #[test]
    fn an_empty_detail_is_not_a_validation_error() {
        assert!(validation_detail("prefill forward: Validation error: ").is_none());
    }

    /// Nested prefixes: take the LAST marker so the innermost detail wins.
    #[test]
    fn the_innermost_detail_wins() {
        let got = validation_detail("a: Validation error: b: Validation error: real detail");
        assert_eq!(got.as_deref(), Some("real detail"));
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    fn pool() -> ModelProcessPool {
        ModelProcessPool::new(std::path::PathBuf::from("/tmp/swarmllm-admission-test"))
    }

    /// With no budget configured, behaviour must be exactly as before — this
    /// gate must not start refusing GPU loads on nodes that never asked for a
    /// limit.
    #[test]
    fn no_budget_admits_everything() {
        let p = pool();
        assert!(p.admit_to_gpu(&ModelId("a".into()), 99_999));
        assert!(p.admit_to_gpu(&ModelId("b".into()), 99_999));
    }

    /// A model that fits is admitted and charged; the next one is judged against
    /// what is already committed, not against an empty device.
    #[test]
    fn a_second_model_is_judged_against_the_first() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        assert!(p.admit_to_gpu(&ModelId("first".into()), 4000));
        assert_eq!(p.vram_committed_mb(), 4000);
        // 4000 + 2500 > 6000 → refused, and NOT charged.
        assert!(!p.admit_to_gpu(&ModelId("second".into()), 2500));
        assert_eq!(p.vram_committed_mb(), 4000, "a refusal must not charge");
        // Something that does fit still gets in.
        assert!(p.admit_to_gpu(&ModelId("small".into()), 1500));
        assert_eq!(p.vram_committed_mb(), 5500);
    }

    /// The reported failure in miniature: two models that each fit alone but not
    /// together. Before admission control both proceeded and the second worker
    /// died with a real CUDA OOM.
    #[test]
    fn two_models_that_do_not_fit_together_are_not_both_admitted() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        let a = ModelId("phi-3.5".into());
        let b = ModelId("llama-3b".into());
        assert!(p.admit_to_gpu(&a, 5900), "fits on its own");
        assert!(
            !p.admit_to_gpu(&b, 3000),
            "must be refused rather than left to OOM"
        );
    }

    fn cand(model: &str, charge_mb: u64, idle_secs: u64) -> VramReclaimCandidate {
        VramReclaimCandidate {
            model: ModelId(model.into()),
            charge_mb,
            idle_secs,
            busy: false,
        }
    }

    /// The reported case, in miniature. An 8B loaded fifteen minutes ago holds
    /// 6033 MB of a 6722 MB budget; a 3B needing 3138 MB arrives. Before this,
    /// admission simply refused and the 3B answered at 7 tok/s on the CPU while
    /// the card sat idle.
    #[test]
    fn an_idle_model_gives_up_the_card_to_the_one_being_asked_for() {
        let plan = plan_vram_reclaim(6722, 6033, 3138, vec![cand("llama-8b", 6033, 900)]);
        assert_eq!(
            plan,
            vec![(ModelId("llama-8b".into()), 6033)],
            "the idle occupant must be reclaimed rather than the newcomer demoted"
        );
    }

    /// Reclaiming must stop as soon as the model fits: taking more than needed
    /// costs cold starts nobody asked for.
    #[test]
    fn only_as_many_models_are_freed_as_the_new_one_needs() {
        let plan = plan_vram_reclaim(
            8000,
            7500,
            1000,
            vec![cand("stale", 600, 3600), cand("older", 900, 1800)],
        );
        assert_eq!(plan.len(), 1, "one is enough: 7500-600+1000 <= 8000");
        assert_eq!(plan[0].0, ModelId("stale".into()), "most idle goes first");
    }

    /// The property that makes this safe to run on every refused admission:
    /// when reclaiming everything eligible would STILL not fit, nothing is
    /// unloaded. The model is going to the CPU either way, so a cold start
    /// bought nothing.
    #[test]
    fn nothing_is_unloaded_when_it_would_still_not_fit() {
        let plan = plan_vram_reclaim(6000, 5000, 5500, vec![cand("small", 1000, 3600)]);
        assert!(
            plan.is_empty(),
            "must not pay a reload for a model that cannot fit anyway"
        );
    }

    /// A worker with a request in flight is never taken, however idle it looks
    /// by the clock — that is killing an answer mid-generation.
    #[test]
    fn a_busy_worker_is_never_reclaimed() {
        let mut busy = cand("serving", 6033, 9999);
        busy.busy = true;
        assert!(plan_vram_reclaim(6722, 6033, 3138, vec![busy]).is_empty());
    }

    /// Two models alternating faster than they load must NOT evict each other
    /// on every request: below the idle floor the previous behaviour (the
    /// newcomer runs on the CPU) is kept, so this can only improve placement.
    #[test]
    fn a_recently_used_model_keeps_the_card() {
        let plan = plan_vram_reclaim(
            6722,
            6033,
            3138,
            vec![cand(
                "just-answered",
                6033,
                VRAM_MAKE_ROOM_MIN_IDLE_SECS - 1,
            )],
        );
        assert!(plan.is_empty(), "displacing it would thrash both models");
    }

    /// Nothing to do when it already fits — this must not unload models to make
    /// room that is already there.
    #[test]
    fn a_model_that_already_fits_reclaims_nothing() {
        assert!(plan_vram_reclaim(8000, 1000, 2000, vec![cand("idle", 1000, 3600)]).is_empty());
    }

    /// Releasing a charge frees the budget for the next admission — otherwise
    /// unloading a model would not actually make room.
    #[test]
    fn releasing_a_charge_frees_the_budget() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        let a = ModelId("a".into());
        assert!(p.admit_to_gpu(&a, 5000));
        assert!(!p.admit_to_gpu(&ModelId("b".into()), 2000));
        p.release_vram_charge(&a);
        assert_eq!(p.vram_committed_mb(), 0);
        assert!(p.admit_to_gpu(&ModelId("b".into()), 2000));
    }

    /// The budget must not come back before the memory does.
    ///
    /// `Child::start_kill` only signals; the OS frees the process's device
    /// allocations afterwards. Releasing the charge on the signal made the
    /// budget read as free while the card was still full, so the next admission
    /// passed against memory that did not exist and its worker died with
    /// CUDA_ERROR_OUT_OF_MEMORY.
    #[tokio::test]
    async fn waiting_for_exit_returns_as_soon_as_the_process_is_reaped() {
        let exited = Arc::new(AtomicBool::new(false));
        let flag = exited.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            flag.store(true, Ordering::Release);
        });

        let started = std::time::Instant::now();
        let confirmed =
            ModelProcessPool::await_worker_exit(&exited, std::time::Duration::from_secs(5)).await;

        assert!(confirmed, "must report the process confirmed gone");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "must return when the process exits, not sit out the whole limit"
        );
    }

    /// A process that never dies must not strand the device forever. The wait
    /// expires and says so, and the caller frees the budget anyway — a worse
    /// outcome than the race, but better than refusing every later load.
    #[tokio::test]
    async fn waiting_for_exit_gives_up_rather_than_stranding_the_device() {
        let never = Arc::new(AtomicBool::new(false));
        let confirmed =
            ModelProcessPool::await_worker_exit(&never, std::time::Duration::from_millis(60)).await;
        assert!(!confirmed, "an expired wait must be reported, not hidden");
    }

    /// An already-reaped worker costs nothing to wait for.
    #[tokio::test]
    async fn waiting_for_an_already_exited_worker_is_immediate() {
        let done = Arc::new(AtomicBool::new(true));
        let started = std::time::Instant::now();
        assert!(
            ModelProcessPool::await_worker_exit(&done, std::time::Duration::from_secs(5)).await
        );
        assert!(started.elapsed() < std::time::Duration::from_millis(100));
    }

    /// An unreadable header yields a 0 estimate, which must NOT be read as "it
    /// fits in nothing" nor as a refusal — failing to read a file is a worse
    /// reason to lose the GPU than the OOM this prevents.
    #[test]
    fn an_unknown_footprint_does_not_refuse() {
        let p = pool();
        p.set_vram_budget_mb(1);
        assert!(p.admit_to_gpu(&ModelId("mystery".into()), 0));
    }

    /// A missing model directory must yield 0 rather than panicking.
    #[test]
    fn estimating_a_missing_model_is_zero() {
        let p = pool();
        assert_eq!(p.estimate_gpu_footprint_mb(&ModelId("nope".into())), 0);
    }
}
