//! Model process pool — manages one subprocess per loaded ModelId.
//!
//! When a model is unloaded, its worker process is killed and the OS/CUDA
//! driver reclaims all GPU memory immediately — no restart required.

use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::net::UnixListener;
use tokio::process::Child;
use tokio::sync::Mutex;

use crate::error::SwarmError;
use crate::inference::router::StreamingTokenEvent;
use crate::inference::worker_ipc::*;
use crate::types::{ModelId, SamplingParams};

/// A handle to a running model worker subprocess.
struct WorkerHandle {
    /// The worker subprocess.
    child: Child,
    /// Socket to communicate with the worker. Locked to serialize requests.
    socket: Mutex<(
        tokio::net::unix::OwnedReadHalf,
        tokio::net::unix::OwnedWriteHalf,
    )>,
    /// Socket file to clean up on drop.
    socket_path: PathBuf,
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Kill the child process if still running
        let _ = self.child.start_kill();
        // Clean up the socket file
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Manages one worker subprocess per loaded ModelId.
///
/// When a model is unloaded, its worker process is killed and the OS/CUDA
/// driver reclaims all GPU memory immediately — no restart required.
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
}

impl ModelProcessPool {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            workers: DashMap::new(),
            spawn_lock: Mutex::new(()),
            data_dir,
            active_shard_windows: DashMap::new(),
            activity_tx: std::sync::OnceLock::new(),
        }
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

        // Start listening before spawning so the worker can connect immediately
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| SwarmError::Internal(format!("socket bind: {e}")))?;

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

        // Wait for worker to connect (timeout 30s)
        let conn = tokio::time::timeout(std::time::Duration::from_secs(30), listener.accept())
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

        tracing::info!("Model worker subprocess started");

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
        self.emit_activity(crate::daemon::state::ActivityEvent {
            category: "model",
            kind: "worker_spawned",
            message: msg,
            model_id: Some(model_id.0.clone()),
            model_name: None,
            node_id: None,
            detail_num: None,
            detail_str: None,
            toast_level: None,
            toast_duration_ms: None,
            shard_index: None,
            freed_bytes: None,
            holder_count_before: None,
            holder_count_after: None,
            remaining_local_shards: None,
            timestamp: None,
        });

        Ok(WorkerHandle {
            child,
            socket: Mutex::new((read_half, write_half)),
            socket_path,
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
        };

        let mut sock = handle.socket.lock().await;
        let (ref mut reader, ref mut writer) = *sock;

        send_daemon(writer, &DaemonMsg::Forward(ipc_fwd), &activations)
            .await
            .map_err(|e| SwarmError::Internal(format!("send Forward: {e}")))?;

        loop {
            let (msg, payload) = recv_worker(reader)
                .await
                .map_err(|e| SwarmError::Internal(format!("recv worker: {e}")))?;
            match msg {
                WorkerMsg::LayerResult(r) if r.request_id == request_id => {
                    let activations = if r.has_activations { payload } else { vec![] };
                    return Ok(crate::types::LayerResult {
                        request_id: r.request_id,
                        token_ids: r.token_ids,
                        finish_reason: r.finish_reason,
                        activations,
                        sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                    });
                }
                WorkerMsg::Error {
                    request_id: rid,
                    message,
                } if rid == request_id => {
                    return Err(SwarmError::Inference(message));
                }
                _ => continue, // unexpected message, skip
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

        let mut sock = handle.socket.lock().await;
        let (ref mut reader, ref mut writer) = *sock;

        send_daemon(writer, &DaemonMsg::Generate(gen), &[])
            .await
            .map_err(|e| SwarmError::Internal(format!("send Generate: {e}")))?;

        let mut content = String::new();
        #[allow(unused_assignments)]
        let mut prompt_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut completion_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut finish_reason = String::new();

        loop {
            let (msg, _) = recv_worker(reader)
                .await
                .map_err(|e| SwarmError::Internal(format!("recv generate: {e}")))?;
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
            if let Ok(mut sock) = handle.socket.try_lock() {
                let (_, ref mut writer) = *sock;
                let _ = send_daemon(writer, &DaemonMsg::Shutdown, &[]).await;
            }
            // Drop handle → kills child process → OS frees all CUDA memory
            drop(handle);
            tracing::info!(model_id = %model_id, "Model worker killed, GPU memory freed");
            self.emit_activity(crate::daemon::state::ActivityEvent {
                category: "model",
                kind: "worker_unloaded",
                message: format!(
                    "Unloaded {} from memory (worker killed, GPU memory freed)",
                    model_id.0
                ),
                model_id: Some(model_id.0.clone()),
                model_name: None,
                node_id: None,
                detail_num: None,
                detail_str: None,
                toast_level: None,
                toast_duration_ms: None,
                shard_index: None,
                freed_bytes: None,
                holder_count_before: None,
                holder_count_after: None,
                remaining_local_shards: None,
                timestamp: None,
            });
        }
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
    pub fn clear_shard_window(&self, model_id: &ModelId) {
        self.active_shard_windows.remove(model_id);
    }

    /// Get the current shard window for a model, if any.
    pub fn get_shard_window(&self, model_id: &ModelId) -> Option<Vec<u32>> {
        self.active_shard_windows.get(model_id).map(|v| v.clone())
    }
}
