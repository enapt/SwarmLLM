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
    data_dir: PathBuf,
}

impl ModelProcessPool {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            workers: DashMap::new(),
            data_dir,
        }
    }

    /// Get or spawn a worker for this model.
    async fn get_or_spawn(&self, model_id: &ModelId) -> Result<Arc<WorkerHandle>, SwarmError> {
        if let Some(handle) = self.workers.get(model_id) {
            return Ok(handle.clone());
        }
        // Spawn new worker
        let handle = self.spawn_worker(model_id).await?;
        let handle = Arc::new(handle);
        self.workers.insert(model_id.clone(), handle.clone());
        Ok(handle)
    }

    async fn spawn_worker(&self, _model_id: &ModelId) -> Result<WorkerHandle, SwarmError> {
        // Create a unique socket path
        let socket_name = format!("swarmllm-worker-{}.sock", uuid::Uuid::new_v4());
        let socket_path = std::env::temp_dir().join(&socket_name);

        // Start listening before spawning so the worker can connect immediately
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| SwarmError::Internal(format!("socket bind: {e}")))?;

        // Spawn the worker subprocess (same binary, model-worker subcommand)
        let exe = std::env::current_exe()
            .map_err(|e| SwarmError::Internal(format!("current_exe: {e}")))?;
        let child = tokio::process::Command::new(&exe)
            .args([
                "model-worker",
                "--socket",
                socket_path.to_str().unwrap_or(""),
                "--data-dir",
                self.data_dir.to_str().unwrap_or(""),
            ])
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

        let request_id = forward.request_id;
        let activations = forward.activations.clone();

        let ipc_fwd = IpcForward {
            request_id,
            sequence_num: forward.sequence_num,
            index_pos: forward.index_pos,
            format: forward.format.clone(),
            model_id: forward.model_id.clone(),
            layer_range: forward.layer_range,
            tp_meta: forward.tp_meta.clone(),
            vision_embeddings: forward.vision_embeddings.clone(),
            requester_node_id: forward.requester_node_id,
            pre_embedded: forward.pre_embedded,
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
}
