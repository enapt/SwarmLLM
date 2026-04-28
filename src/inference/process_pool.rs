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
/// Default KV-cache TTL in seconds (10 minutes). Overridden by config at startup.
pub const DEFAULT_KV_CACHE_TTL_SECS: u64 = 600;

/// Per-request buffered channel capacity for multiplexed worker responses.
/// Long decode streams emit one WorkerMsg::Token per generated token; 256 gives
/// plenty of headroom for a caller that's slow to consume without stalling the
/// reader actor.
const RESPONSE_CHANNEL_CAPACITY: usize = 256;

/// Response channel entry: a bounded mpsc sender carrying `(WorkerMsg, payload_bytes)`.
type ResponseTx = mpsc::Sender<(WorkerMsg, Vec<u8>)>;

/// Shared map from `request_id` to the caller's response channel. The reader
/// actor looks up each inbound message's `request_id` here to route the reply.
type ResponseMap = Arc<DashMap<Uuid, ResponseTx>>;

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
    /// Socket name used to connect (Unix filesystem path / Windows namespace
    /// name). Only the Unix filesystem variant requires drop-time cleanup.
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
        WorkerMsg::BatchResult { .. }
        | WorkerMsg::PrefixManifestUpdate { .. }
        | WorkerMsg::PrefixFetchProbe { .. }
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
                        let _ = tx.try_send(PrefixManifestEvent {
                            model_id: announce_model,
                            blocks,
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
                        let _ = tx.try_send(PrefixProbeEvent {
                            model_id: probe_model,
                            request_id,
                            blocks,
                        });
                    }
                    continue;
                }
                if let Some(rid) = worker_msg_request_id(&msg) {
                    // `get` returns a Ref, which holds a shard lock. Clone the
                    // Sender and drop the Ref *before* awaiting `send` so we
                    // don't hold a DashMap shard across an await point (that
                    // would risk deadlock on a concurrent insert/remove).
                    if let Some(tx) = responses.get(&rid).map(|r| r.value().clone()) {
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
                dead.store(true, Ordering::Relaxed);
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
    for (r, len) in results.into_iter().zip(activation_lens.into_iter()) {
        let len = len as usize;
        let end = cursor.saturating_add(len);
        let slot_payload = if r.has_activations && end <= payload.len() {
            payload[cursor..end].to_vec()
        } else {
            Vec::new()
        };
        cursor = end;
        let rid = r.request_id;
        let tx_opt = responses.get(&rid).map(|e| e.value().clone());
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
}

impl Drop for ResponseGuard {
    fn drop(&mut self) {
        self.responses.remove(&self.request_id);
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
) {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    while let Some(first) = rx.recv().await {
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
        let BatchSchedulerMsg::Forward { fwd, resp_tx } = msgs.into_iter().next().unwrap();
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
            for (tx, result) in resp_txs.into_iter().zip(results.into_iter()) {
                let _ = tx.send(Ok(result));
            }
        }
        Err(e) => {
            let msg = e.to_string();
            for tx in resp_txs {
                let _ = tx.send(Err(SwarmError::Internal(format!("batch failed: {msg}"))));
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
    data_dir: PathBuf,
    /// Active shard windows: which shards each model worker should load.
    /// If absent, the worker loads all on-disk shards (default behavior).
    active_shard_windows: DashMap<ModelId, Vec<u32>>,
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
            data_dir,
            active_shard_windows: DashMap::new(),
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
            batched_prefill_forward: std::sync::atomic::AtomicBool::new(true),
            batch_scheduler: std::sync::OnceLock::new(),
            prefix_manifest_tx: std::sync::OnceLock::new(),
            prefix_probe_tx: std::sync::OnceLock::new(),
        }
    }

    /// Install the prefix-manifest sink. The daemon spawns a forwarder task
    /// that owns the receiver and turns each event into a gossip broadcast +
    /// local-index update. Idempotent — a second call is a no-op.
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
        if handle.dead.load(Ordering::Relaxed) {
            return Err(SwarmError::Internal("worker dead".into()));
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
        if handle.dead.load(Ordering::Relaxed) {
            return None;
        }
        let request_id = Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel::<(WorkerMsg, Vec<u8>)>(2);
        handle.responses.insert(request_id, tx);
        let _guard = ResponseGuard {
            responses: handle.responses.clone(),
            request_id,
        };
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
    pub fn start_batch_scheduler(self: &Arc<Self>) {
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
            batch_scheduler_loop(pool, rx).await;
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

    fn emit_activity(&self, event: crate::daemon::state::ActivityEvent) {
        if let Some(tx) = self.activity_tx.get() {
            let _ = tx.send(event);
        }
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
        let handle = self.spawn_worker(model_id).await?;
        let handle = Arc::new(handle);
        self.workers.insert(model_id.clone(), handle.clone());
        Ok(handle)
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
                SwarmError::Internal(format!("socket bind: {e}"))
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
        let exe = std::env::current_exe()
            .map_err(|e| SwarmError::Internal(format!("current_exe: {e}")))?;
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
            args.push("--batched-prefill-forward".to_string());
            args.push(
                self.batched_prefill_forward
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
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

        let child = tokio::process::Command::new(&exe)
            .args(&args)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| SwarmError::Internal(format!("spawn worker: {e}")))?;

        // Wait for worker to connect
        let conn = tokio::time::timeout(
            std::time::Duration::from_secs(WORKER_CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .map_err(|_| SwarmError::Internal("worker connect timeout".into()))?
        .map_err(|e| SwarmError::Internal(format!("accept: {e}")))?;

        let (mut read_half, write_half) = conn.split();

        // Read Ready message
        let (ready_msg, _) = recv_worker(&mut read_half)
            .await
            .map_err(|e| SwarmError::Internal(format!("read ready: {e}")))?;
        match ready_msg {
            WorkerMsg::Ready => {}
            other => {
                return Err(SwarmError::Internal(format!(
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
        let msg = match shard_info {
            Some(shards) => format!("Spawning worker for {} ({})", model_id.0, shards),
            None => format!("Spawning worker for {}", model_id.0),
        };
        self.emit_activity(
            crate::daemon::state::ActivityEvent::new("model", "worker_spawned", msg)
                .with_model(model_id.0.clone()),
        );

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
        ));
        Ok(WorkerHandle {
            child,
            writer: Mutex::new(write_half),
            responses,
            dead,
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
            adapter_id,
            draft_tokens,
            spec_logits_requested,
            truncate_kv_to,
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
            spec_logits_requested,
            truncate_kv_to,
        };

        if handle.dead.load(Ordering::Relaxed) {
            self.workers.remove(&model_id);
            return Err(SwarmError::Internal("worker is dead".into()));
        }

        // Register a response channel BEFORE sending so the reader actor can
        // route any early error/reply. Unregistered on drop via ResponseGuard.
        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        handle.responses.insert(request_id, resp_tx);
        let _guard = ResponseGuard {
            responses: handle.responses.clone(),
            request_id,
        };

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
                return Err(SwarmError::Internal(format!("send Forward: {e}")));
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
                        return Ok(crate::types::LayerResult {
                            request_id: r.request_id,
                            token_ids: r.token_ids,
                            finish_reason: r.finish_reason,
                            activations,
                            sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                            spec_logits,
                        });
                    }
                    WorkerMsg::Error {
                        request_id: rid,
                        message,
                    } if rid == request_id => {
                        return Err(SwarmError::Inference(message));
                    }
                    _ => continue,
                },
                None => {
                    // Reader actor closed the channel — worker died while we were waiting.
                    self.workers.remove(&model_id);
                    return Err(SwarmError::Internal(
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
        if handle.dead.load(Ordering::Relaxed) {
            self.workers.remove(&model_id);
            return Err(SwarmError::Internal("worker is dead".into()));
        }

        // Register one response channel per request_id BEFORE sending.
        type SlotRx = mpsc::Receiver<(WorkerMsg, Vec<u8>)>;
        let n = forwards.len();
        let mut receivers: Vec<(Uuid, SlotRx)> = Vec::with_capacity(n);
        let mut guards: Vec<ResponseGuard> = Vec::with_capacity(n);
        for f in &forwards {
            let (tx, rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            handle.responses.insert(f.request_id, tx);
            guards.push(ResponseGuard {
                responses: handle.responses.clone(),
                request_id: f.request_id,
            });
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
                adapter_id,
                draft_tokens,
                spec_logits_requested,
                truncate_kv_to,
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
                return Err(SwarmError::Internal(format!("send BatchForward: {e}")));
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
                        });
                        break;
                    }
                    Some((
                        WorkerMsg::Error {
                            request_id: err_rid,
                            message,
                        },
                        _,
                    )) if err_rid == rid => {
                        return Err(SwarmError::Inference(message));
                    }
                    Some(_) => continue,
                    None => {
                        self.workers.remove(&model_id);
                        return Err(SwarmError::Internal(
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
        token_tx: Option<tokio::sync::mpsc::Sender<StreamingTokenEvent>>,
    ) -> Result<crate::inference::router::InferenceOutput, SwarmError> {
        let handle = self.get_or_spawn(model_id).await?;

        let gen = IpcGenerate {
            request_id,
            model_id: model_id.clone(),
            layer_range,
            prompt,
            sampling,
            session_id,
        };

        if handle.dead.load(Ordering::Relaxed) {
            self.workers.remove(model_id);
            return Err(SwarmError::Internal("worker is dead".into()));
        }

        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        handle.responses.insert(request_id, resp_tx);
        let _guard = ResponseGuard {
            responses: handle.responses.clone(),
            request_id,
        };

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) = send_daemon(&mut *writer, &DaemonMsg::Generate(gen), &[]).await {
                drop(writer);
                self.workers.remove(model_id);
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                return Err(SwarmError::Internal(format!("send Generate: {e}")));
            }
        }

        let mut content = String::new();
        #[allow(unused_assignments)]
        let mut prompt_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut completion_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut finish_reason = String::new();

        loop {
            let (msg, _) = match resp_rx.recv().await {
                Some(v) => v,
                None => {
                    self.workers.remove(model_id);
                    return Err(SwarmError::Internal(
                        "worker closed connection mid-generate".into(),
                    ));
                }
            };
            match msg {
                WorkerMsg::Token {
                    request_id: rid,
                    text,
                    is_eos,
                    ..
                } if rid == request_id => {
                    if !is_eos {
                        content.push_str(&text);
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: text.clone(),
                                    finish_reason: None,
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
                } if rid == request_id => {
                    prompt_tokens = pt as u32;
                    completion_tokens = ct as u32;
                    finish_reason = fr;
                    break;
                }
                WorkerMsg::Error {
                    request_id: rid,
                    message,
                } if rid == request_id => {
                    return Err(SwarmError::Inference(message));
                }
                _ => continue,
            }
        }

        if let Some(ref tx) = token_tx {
            let _ = tx
                .send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some(finish_reason.clone()),
                })
                .await;
        }

        Ok(crate::inference::router::InferenceOutput {
            request_id,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            session_id: None,
            token_logprobs: vec![],
        })
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
            self.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "model",
                    "worker_unloaded",
                    format!(
                        "Unloaded {} from memory (worker killed, GPU memory freed)",
                        model_id.0
                    ),
                )
                .with_model(model_id.0.clone()),
            );
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
