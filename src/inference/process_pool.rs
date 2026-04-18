//! Model process pool — manages one subprocess per loaded ModelId.
//!
//! When a model is unloaded, its worker process is killed and the OS/CUDA
//! driver reclaims all GPU memory immediately — no restart required.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::net::UnixListener;
use tokio::process::Child;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::error::SwarmError;
use crate::inference::router::StreamingTokenEvent;
use crate::inference::worker_ipc::*;
use crate::types::{ModelId, SamplingParams};

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
    writer: Mutex<tokio::net::unix::OwnedWriteHalf>,
    /// Per-request response channels. The reader actor inserts `(msg, payload)`
    /// tuples keyed by `request_id`; callers register a channel before sending
    /// their request and drain it until they get a terminal message.
    responses: ResponseMap,
    /// Set to true when the reader actor observes a socket error. Subsequent
    /// callers short-circuit with an error + trigger worker eviction.
    dead: Arc<AtomicBool>,
    /// Socket file to clean up on drop.
    socket_path: PathBuf,
    /// Handle to the reader actor task. Aborted on drop so the task doesn't
    /// outlive its worker; also unblocks any pending `recv_worker` in tests.
    reader_handle: tokio::task::JoinHandle<()>,
}

/// Pull the `request_id` field out of any `WorkerMsg` variant that carries one.
/// `Ready` / `Bye` have no request_id and are dropped by the reader actor
/// (they're only relevant during spawn, which handshakes synchronously).
fn worker_msg_request_id(msg: &WorkerMsg) -> Option<Uuid> {
    match msg {
        WorkerMsg::LayerResult(r) => Some(r.request_id),
        WorkerMsg::BatchResult { results, .. } => results.first().map(|r| r.request_id),
        WorkerMsg::Token { request_id, .. }
        | WorkerMsg::GenerateDone { request_id, .. }
        | WorkerMsg::Error { request_id, .. } => Some(*request_id),
        WorkerMsg::Ready | WorkerMsg::Bye => None,
    }
}

/// Reader actor: owns the read half of the worker socket, dispatches each
/// inbound message to the right per-request channel. Exits when the socket
/// errors out (worker died, IPC corrupted); sets `dead` and drops all
/// in-flight response senders to wake waiting callers with `None`.
async fn reader_actor(
    mut reader: tokio::net::unix::OwnedReadHalf,
    responses: ResponseMap,
    dead: Arc<AtomicBool>,
    model_id: ModelId,
) {
    loop {
        match recv_worker(&mut reader).await {
            Ok((msg, payload)) => {
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

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Stop the reader actor so it doesn't outlive the socket.
        self.reader_handle.abort();
        // Kill the child process if still running
        let _ = self.child.start_kill();
        // Clean up the socket file
        let _ = std::fs::remove_file(&self.socket_path);
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
    /// `docs/plans/distributed_inference_speedup.md`.
    activation_compression: std::sync::atomic::AtomicBool,
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
        }
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
        // Create a unique socket path
        let socket_name = format!("swarmllm-worker-{}.sock", uuid::Uuid::new_v4());
        let socket_path = std::env::temp_dir().join(&socket_name);

        // RAII guard: clean up socket file if spawn fails at any step.
        // Defused (forgotten) on success when WorkerHandle takes ownership.
        struct SocketCleanup(std::path::PathBuf);
        impl Drop for SocketCleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }

        // Start listening before spawning so the worker can connect immediately
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| SwarmError::Internal(format!("socket bind: {e}")))?;
        let socket_guard = SocketCleanup(socket_path.clone());

        // SEC: Restrict socket permissions so only the current user can connect.
        // Without this, any local process can impersonate the worker and intercept
        // inference data (prompts, activations).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600));
        }

        // Spawn the worker subprocess (same binary, model-worker subcommand)
        let exe = std::env::current_exe()
            .map_err(|e| SwarmError::Internal(format!("current_exe: {e}")))?;
        let socket_str = socket_path
            .to_str()
            .ok_or_else(|| SwarmError::Internal("socket path is not valid UTF-8".into()))?;
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
        .map_err(|e| SwarmError::Internal(format!("accept: {e}")))?
        .0;

        let (mut read_half, write_half) = conn.into_split();

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

        // Success — defuse the cleanup guard; WorkerHandle now owns the socket file
        std::mem::forget(socket_guard);
        let responses: ResponseMap = Arc::new(DashMap::new());
        let dead = Arc::new(AtomicBool::new(false));
        let reader_handle = tokio::spawn(reader_actor(
            read_half,
            responses.clone(),
            dead.clone(),
            model_id.clone(),
        ));
        Ok(WorkerHandle {
            child,
            writer: Mutex::new(write_half),
            responses,
            dead,
            socket_path,
            reader_handle,
        })
    }

    /// Send a LayerForward to the worker, get a LayerResult back.
    pub async fn forward(
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

        let ipc_fwd = IpcForward {
            request_id,
            sequence_num,
            index_pos,
            format,
            model_id: fwd_model_id,
            layer_range,
            tp_meta,
            vision_embeddings,
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

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) =
                send_daemon(&mut *writer, &DaemonMsg::Forward(ipc_fwd), &activations).await
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
                        let activations = if r.has_activations { payload } else { vec![] };
                        return Ok(crate::types::LayerResult {
                            request_id: r.request_id,
                            token_ids: r.token_ids,
                            finish_reason: r.finish_reason,
                            activations,
                            sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                            spec_logits: r.spec_logits,
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
