//! Model process pool — manages one subprocess per loaded ModelId.
//!
//! When a model is unloaded, its worker process is killed and the OS/CUDA
//! driver reclaims all GPU memory immediately — no restart required.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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
    child: Child,
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

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Stop the reader actor so it doesn't outlive the socket.
        self.reader_handle.abort();
        // Kill the child process if still running
        let _ = self.child.start_kill();
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
    /// Models forced onto the CPU for the rest of this daemon's life because
    /// a worker died of a GPU OOM while serving them. Without this, the
    /// respawned worker makes the identical allocation and dies the same way,
    /// and the user sees an unbroken run of 500s with no path out.
    cpu_pinned_models: dashmap::DashSet<ModelId>,
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
            activity_tx: std::sync::OnceLock::new(),
            kv_cache_ttl_secs: std::sync::atomic::AtomicU64::new(DEFAULT_KV_CACHE_TTL_SECS),
            prefix_cache_enabled: std::sync::atomic::AtomicBool::new(true),
            prefix_cache_max_entries: std::sync::atomic::AtomicU32::new(16),
            prefix_cache_max_prompt_tokens: std::sync::atomic::AtomicU32::new(8192),
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
        if self.cpu_pinned_models.contains(model_id) {
            return 0;
        }
        self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Is this model currently forced onto the CPU after a GPU OOM?
    pub fn is_cpu_pinned(&self, model_id: &ModelId) -> bool {
        self.cpu_pinned_models.contains(model_id)
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
    ) {
        use std::sync::atomic::Ordering;
        self.prefix_cache_enabled.store(enabled, Ordering::Relaxed);
        self.prefix_cache_max_entries
            .store(max_entries, Ordering::Relaxed);
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
        match self.spawn_worker(model_id).await {
            Ok(handle) => {
                // Reset the failure counter on first success.
                self.spawn_failures.remove(model_id);
                let handle = Arc::new(handle);
                self.workers.insert(model_id.clone(), handle.clone());
                Ok(handle)
            }
            Err(e) => {
                let count = self
                    .spawn_failures
                    .entry(model_id.clone())
                    .and_modify(|v| *v = (std::time::Instant::now(), v.1.saturating_add(1)))
                    .or_insert((std::time::Instant::now(), 1))
                    .1;
                let cooldown = spawn_failure_cooldown(count);
                tracing::error!(
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

        let child = tokio::process::Command::new(&exe)
            .args(&args)
            .kill_on_drop(true)
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
            child,
            writer: Mutex::new(write_half),
            responses,
            dead,
            #[cfg(unix)]
            socket_name,
            reader_handle,
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
            sender_peer_bytes: _,
            requester_node_id,
            pre_embedded,
            generated_ids,
            adapter_id,
            draft_tokens,
            spec_logits_requested,
            truncate_kv_to,
            chunk_meta: _,
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
            sampling: Default::default(),
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
                sender_peer_bytes: _,
                requester_node_id,
                pre_embedded,
                generated_ids,
                adapter_id,
                draft_tokens,
                spec_logits_requested,
                truncate_kv_to,
                chunk_meta: _,
            } = f;
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
                sampling: Default::default(),
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

            // An OOM will repeat verbatim on the respawn — same model, same
            // device, same allocation — so retrying on the GPU just burns
            // another load. Pin the model to the CPU: slower, but it answers.
            let was_on_gpu = self.effective_gpu_layers(model_id) != 0;
            if was_on_gpu && message.to_ascii_lowercase().contains("out of memory") {
                self.cpu_pinned_models.insert(model_id.clone());
                tracing::warn!(
                    model = %model_id,
                    "GPU out of memory — pinning this model to CPU for the rest of this run"
                );
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "inference",
                            "model_cpu_fallback",
                            format!(
                                "{} ran out of GPU memory — switched to CPU (slower, but working)",
                                model_id.0
                            ),
                        )
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
        if let Some(detail) = validation_detail(&message) {
            return SwarmError::Validation(detail);
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
        if let Some((_, handle)) = self.workers.remove(model_id) {
            // Try graceful shutdown first
            if let Ok(mut writer) = handle.writer.try_lock() {
                let _ = send_daemon(&mut *writer, &DaemonMsg::Shutdown, &[]).await;
            }
            // Drop handle → aborts reader, kills child process → OS frees all CUDA memory
            drop(handle);
            tracing::info!(model_id = %model_id, "Model worker killed, GPU memory freed");
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

/// Pull the user-facing part out of a worker message that wrapped a validation
/// failure, or `None` when it is not one.
///
/// Worker messages arrive with call-site context prepended
/// (`"prefill forward: Validation error: <detail>"`). Taking the text after the
/// LAST marker drops that plumbing, which the caller can neither act on nor
/// understand, and keeps the part that tells them what to change.
fn validation_detail(message: &str) -> Option<String> {
    const MARKER: &str = "Validation error: ";
    let idx = message.rfind(MARKER)?;
    let detail = message[idx + MARKER.len()..].trim();
    if detail.is_empty() {
        None
    } else {
        Some(detail.to_string())
    }
}

#[cfg(test)]
mod validation_detail_tests {
    use super::validation_detail;

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
