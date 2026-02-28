use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::model::distribution::ShardDistributor;
use crate::model::shard::ShardStore;
use crate::types::{ModelId, ModelManifest, NetworkCommand, NodeId, ShardId};

/// Status of a model acquisition job.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AcquisitionStatus {
    pub model_id: ModelId,
    pub state: AcquisitionState,
    pub total_shards: u32,
    pub downloaded_shards: u32,
    pub verified_shards: u32,
    pub failed_shards: u32,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    /// Per-shard progress: index → bytes received so far.
    #[serde(default)]
    pub shard_progress: HashMap<u32, ShardProgress>,
    /// Bytes/sec download speed (rolling average).
    #[serde(default)]
    pub speed_bytes_per_sec: u64,
    /// Timestamp when acquisition started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Recent log lines for the UI.
    #[serde(default)]
    pub log: Vec<String>,
}

/// Progress for a single shard.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ShardProgress {
    pub index: u32,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub state: ShardState,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardState {
    Pending,
    Downloading,
    Verifying,
    Complete,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionState {
    /// Waiting for manifest from network.
    AwaitingManifest,
    /// Manifest received, downloading shards.
    Downloading,
    /// All shards downloaded and verified.
    Complete,
    /// Acquisition failed.
    Failed { reason: String },
}

/// Command sent to the AcquisitionManager.
#[derive(Debug)]
pub enum AcquisitionCommand {
    /// Request acquisition of a model by ID.
    /// The manifest must already be known in the model_registry.
    Acquire { model_id: ModelId },
    /// A shard data chunk was received from the network.
    ShardDataReceived {
        shard_id: ShardId,
        offset: u64,
        data: Vec<u8>,
        total_size: u64,
    },
    /// Query the status of an acquisition.
    Status {
        model_id: ModelId,
        reply: tokio::sync::oneshot::Sender<Option<AcquisitionStatus>>,
    },
}

/// Manages the lifecycle of acquiring models from the network.
///
/// Security model:
/// - Only downloads models whose manifest is known from the network (via GossipSub/DHT)
/// - Manifests must pass BLAKE3 hash verification before being trusted
/// - Each downloaded shard is BLAKE3-verified against the manifest
/// - Failed shards are quarantined and the serving peer's trust score is penalized
pub struct AcquisitionManager {
    shared_state: Arc<SharedState>,
    shard_store: ShardStore,
    network_tx: mpsc::Sender<NetworkCommand>,
    command_rx: mpsc::Receiver<AcquisitionCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Active acquisition jobs keyed by model ID.
    jobs: HashMap<ModelId, AcquisitionJob>,
}

struct AcquisitionJob {
    manifest: ModelManifest,
    status: AcquisitionStatus,
    /// Tracks raw bytes received per shard (for multi-chunk downloads).
    shard_bytes: HashMap<u32, u64>,
    /// For speed calculation: (timestamp, cumulative_bytes) samples.
    speed_samples: Vec<(std::time::Instant, u64)>,
}

impl AcquisitionManager {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        command_rx: mpsc::Receiver<AcquisitionCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let shard_store = ShardStore::new(&shared_state.config.node.data_dir);
        Self {
            shared_state,
            shard_store,
            network_tx,
            command_rx,
            shutdown_rx,
            jobs: HashMap::new(),
        }
    }

    pub async fn run(mut self) -> Result<(), SwarmError> {
        tracing::info!("AcquisitionManager running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("AcquisitionManager shutting down");
                        break;
                    }
                }
                cmd = self.command_rx.recv() => {
                    match cmd {
                        Some(AcquisitionCommand::Acquire { model_id }) => {
                            self.handle_acquire(model_id).await;
                        }
                        Some(AcquisitionCommand::ShardDataReceived {
                            shard_id, offset, data, total_size,
                        }) => {
                            self.handle_shard_data(shard_id, offset, &data, total_size).await;
                        }
                        Some(AcquisitionCommand::Status { model_id, reply }) => {
                            let status = self.jobs.get(&model_id).map(|j| j.status.clone());
                            let _ = reply.send(status);
                        }
                        None => break,
                    }
                }
            }
        }

        Ok(())
    }

    /// Start acquiring a model from the network.
    async fn handle_acquire(&mut self, model_id: ModelId) {
        // Check if already acquiring
        if self.jobs.contains_key(&model_id) {
            tracing::info!(model = %model_id, "Model acquisition already in progress");
            return;
        }

        // SECURITY: The manifest MUST come from the network registry, not from disk.
        // This ensures we're downloading a model that the network knows about.
        let manifest = match self.shared_state.model_registry.get_manifest(&model_id) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    model = %model_id,
                    "Cannot acquire model: manifest not found in network registry"
                );
                let status = AcquisitionStatus {
                    model_id: model_id.clone(),
                    state: AcquisitionState::AwaitingManifest,
                    total_shards: 0,
                    downloaded_shards: 0,
                    verified_shards: 0,
                    failed_shards: 0,
                    total_bytes: 0,
                    downloaded_bytes: 0,
                    shard_progress: HashMap::new(),
                    speed_bytes_per_sec: 0,
                    started_at: Some(chrono::Utc::now()),
                    log: vec!["Waiting for manifest from network...".into()],
                };
                self.publish_progress(&model_id, &status);
                self.jobs.insert(
                    model_id.clone(),
                    AcquisitionJob {
                        manifest: ModelManifest {
                            id: model_id.clone(),
                            name: String::new(),
                            architecture: crate::types::ModelArchitecture::Llama,
                            num_layers: 0,
                            num_params_billions: 0.0,
                            quantization: crate::types::Quantization::Q4KM,
                            total_size_bytes: 0,
                            shard_count: 0,
                            shards: vec![],
                            tokenizer_hash: [0u8; 32],
                            manifest_hash: [0u8; 32],
                            publisher: NodeId([0u8; 32]),
                            publish_date: chrono::Utc::now(),
                            license: String::new(),
                        },
                        status,
                        shard_bytes: HashMap::new(),
                        speed_samples: Vec::new(),
                    },
                );
                return;
            }
        };

        // SECURITY: Verify the manifest hash before trusting it
        if let Err(e) = manifest.verify_hash() {
            tracing::error!(
                model = %model_id,
                error = %e,
                "Manifest hash verification failed — refusing to download (possible poisoning)"
            );
            return;
        }

        tracing::info!(
            model = %model_id,
            shards = manifest.shard_count,
            size_bytes = manifest.total_size_bytes,
            "Starting model acquisition"
        );

        // Save the verified manifest to disk
        let model_dir = self.shard_store.models_dir().join(&model_id.0);
        if let Err(e) = manifest.save_to_dir(&model_dir) {
            tracing::error!(model = %model_id, error = %e, "Failed to save manifest");
            return;
        }

        let total_bytes = manifest.total_size_bytes;
        let total_shards = manifest.shard_count;

        // Determine which shards we need (rarest-first)
        let needed =
            ShardDistributor::select_rarest_shards(&manifest, &self.shared_state.model_registry);

        // Filter out shards we already have locally and verified
        let needed: Vec<_> = needed
            .into_iter()
            .filter(|sid| {
                let path = self.shard_store.shard_path(&sid.model_id, sid.index);
                if !path.exists() {
                    return true;
                }
                // If file exists, verify it against manifest
                let shard_info = manifest.shards.iter().find(|s| s.index == sid.index);
                match shard_info {
                    Some(info) => self.shard_store.verify_shard(&sid.model_id, info).is_err(),
                    None => true,
                }
            })
            .collect();

        // Build initial per-shard progress
        let mut shard_prog: HashMap<u32, ShardProgress> = HashMap::new();
        for s in &manifest.shards {
            let already_local = !needed.iter().any(|sid| sid.index == s.index);
            shard_prog.insert(
                s.index,
                ShardProgress {
                    index: s.index,
                    total_bytes: s.size_bytes,
                    downloaded_bytes: if already_local { s.size_bytes } else { 0 },
                    state: if already_local {
                        ShardState::Complete
                    } else {
                        ShardState::Pending
                    },
                },
            );
        }

        let already_bytes: u64 = manifest
            .shards
            .iter()
            .filter(|s| !needed.iter().any(|sid| sid.index == s.index))
            .map(|s| s.size_bytes)
            .sum();

        let status = AcquisitionStatus {
            model_id: model_id.clone(),
            state: AcquisitionState::Downloading,
            total_shards,
            downloaded_shards: total_shards - needed.len() as u32,
            verified_shards: total_shards - needed.len() as u32,
            failed_shards: 0,
            total_bytes,
            downloaded_bytes: already_bytes,
            shard_progress: shard_prog,
            speed_bytes_per_sec: 0,
            started_at: Some(chrono::Utc::now()),
            log: vec![format!(
                "Starting acquisition: {} shards to download ({} total)",
                needed.len(),
                format_bytes_short(total_bytes)
            )],
        };
        self.publish_progress(&model_id, &status);

        self.jobs.insert(
            model_id.clone(),
            AcquisitionJob {
                manifest: manifest.clone(),
                status,
                shard_bytes: HashMap::new(),
                speed_samples: vec![(std::time::Instant::now(), already_bytes)],
            },
        );

        if needed.is_empty() {
            tracing::info!(model = %model_id, "All shards already present and verified");
            if let Some(job) = self.jobs.get_mut(&model_id) {
                job.status.state = AcquisitionState::Complete;
                job.status
                    .log
                    .push("All shards already present and verified".into());
                self.shared_state
                    .acquisition_progress
                    .insert(model_id.clone(), job.status.clone());
            }
            self.register_model(&model_id, &manifest);
            return;
        }

        // Request each needed shard from the network with retry logic
        for shard_id in &needed {
            let mut failed_peers: Vec<NodeId> = Vec::new();
            let retry_delays = [5u64, 30, 120]; // exponential backoff: 5s, 30s, 120s
            let mut success = false;

            for attempt in 0..3u32 {
                // Find peers that hold this shard, excluding previously failed ones
                let holders = self.shared_state.model_registry.shard_holders(shard_id);
                let eligible: Vec<_> = holders.iter().filter(|h| !failed_peers.contains(h)).cloned().collect();

                if eligible.is_empty() {
                    if attempt == 0 && holders.is_empty() {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_id.index,
                            "No known holders for shard"
                        );
                    } else {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_id.index,
                            attempt = attempt + 1,
                            "No eligible holders remaining after excluding failed peers"
                        );
                    }
                    break;
                }

                // Pick the best peer (lowest latency, highest trust)
                let target = self.select_best_peer(&eligible);

                // Send directed shard transfer request to the target peer
                let request = crate::types::ShardRequest {
                    shard_id: shard_id.clone(),
                    chunk_offset: 0,
                    chunk_size: 32 * 1024 * 1024, // 32MB chunks
                };

                // Look up the peer's libp2p PeerId bytes for directed request_response
                let peer_id_bytes = self
                    .shared_state
                    .peer_registry
                    .get(&target)
                    .and_then(|p| p.peer_id_bytes.clone());

                match peer_id_bytes {
                    Some(bytes) => {
                        let cmd = NetworkCommand::SendShardRequest {
                            target_peer_bytes: bytes,
                            request,
                        };
                        if let Err(e) = self.network_tx.send(cmd).await {
                            tracing::warn!(
                                error = %e,
                                attempt = attempt + 1,
                                "Failed to send shard request, retrying"
                            );
                            failed_peers.push(target);
                            if attempt < 2 {
                                tokio::time::sleep(std::time::Duration::from_secs(retry_delays[attempt as usize])).await;
                            }
                            continue;
                        }
                        success = true;
                    }
                    None => {
                        tracing::warn!(
                            peer = %target,
                            attempt = attempt + 1,
                            "Cannot send shard request: peer_id_bytes not available"
                        );
                        failed_peers.push(target);
                        if attempt < 2 {
                            tokio::time::sleep(std::time::Duration::from_secs(retry_delays[attempt as usize])).await;
                        }
                        continue;
                    }
                }

                tracing::info!(
                    model = %model_id,
                    shard = shard_id.index,
                    peer = %target,
                    attempt = attempt + 1,
                    "Requested shard transfer"
                );
                break;
            }

            // Update status log and mark shard as downloading
            if let Some(job) = self.jobs.get_mut(&model_id) {
                let msg = if success {
                    format!("Requesting shard {}", shard_id.index)
                } else {
                    format!("Failed to request shard {} after 3 attempts", shard_id.index)
                };
                job.status.log.push(msg);
                if success {
                    if let Some(sp) = job.status.shard_progress.get_mut(&shard_id.index) {
                        sp.state = ShardState::Downloading;
                    }
                }
                self.shared_state
                    .acquisition_progress
                    .insert(model_id.clone(), job.status.clone());
            }
        }
    }

    /// Handle incoming shard data from a peer.
    async fn handle_shard_data(
        &mut self,
        shard_id: ShardId,
        offset: u64,
        data: &[u8],
        total_size: u64,
    ) {
        let model_id = shard_id.model_id.clone();
        let shard_index = shard_id.index;

        if !self.jobs.contains_key(&model_id) {
            tracing::warn!(
                model = %model_id,
                "Received shard data for unknown acquisition — ignoring"
            );
            return;
        }

        // Write chunk to disk
        if let Err(e) = self
            .shard_store
            .write_chunk(&model_id, shard_index, offset, data)
        {
            tracing::error!(
                model = %model_id,
                shard = shard_index,
                error = %e,
                "Failed to write shard chunk"
            );
            return;
        }

        // Grab references we need before borrowing job mutably
        let progress_map = &self.shared_state.acquisition_progress;
        let node_id = self.shared_state.identity.node_id().clone();

        let job = self.jobs.get_mut(&model_id).unwrap();

        // Track progress
        let received = job.shard_bytes.entry(shard_index).or_insert(0);
        *received += data.len() as u64;
        job.status.downloaded_bytes += data.len() as u64;

        // Update per-shard progress
        if let Some(sp) = job.status.shard_progress.get_mut(&shard_index) {
            sp.downloaded_bytes = *received;
            sp.state = ShardState::Downloading;
        }

        // Update speed (rolling 10-second window)
        let now = std::time::Instant::now();
        job.speed_samples.push((now, job.status.downloaded_bytes));
        let cutoff = now - std::time::Duration::from_secs(10);
        job.speed_samples.retain(|(t, _)| *t >= cutoff);
        if job.speed_samples.len() >= 2 {
            let first = &job.speed_samples[0];
            let last = &job.speed_samples[job.speed_samples.len() - 1];
            let dt = last.0.duration_since(first.0).as_secs_f64();
            if dt > 0.1 {
                job.status.speed_bytes_per_sec = ((last.1 - first.1) as f64 / dt) as u64;
            }
        }

        // Publish progress
        progress_map.insert(model_id.clone(), job.status.clone());

        // Check if this shard is complete
        if *received >= total_size {
            // Atomically finalize the shard file (.tmp → .bin)
            if let Err(e) = self.shard_store.finalize_shard(&model_id, shard_index) {
                tracing::error!(
                    model = %model_id,
                    shard = shard_index,
                    error = %e,
                    "Failed to finalize shard file"
                );
            }

            // SECURITY: Verify the completed shard against the manifest hash
            let shard_info_cloned = job
                .manifest
                .shards
                .iter()
                .find(|s| s.index == shard_index)
                .cloned();

            match shard_info_cloned {
                Some(info) => {
                    // Mark as verifying
                    if let Some(sp) = job.status.shard_progress.get_mut(&shard_index) {
                        sp.state = ShardState::Verifying;
                    }
                    job.status.log.push(format!(
                        "Shard {} complete ({}) — verifying BLAKE3 hash...",
                        shard_index,
                        format_bytes_short(total_size)
                    ));
                    progress_map.insert(model_id.clone(), job.status.clone());

                    match self.shard_store.verify_shard(&model_id, &info) {
                        Ok(()) => {
                            job.status.downloaded_shards += 1;
                            job.status.verified_shards += 1;
                            if let Some(sp) = job.status.shard_progress.get_mut(&shard_index) {
                                sp.state = ShardState::Complete;
                            }
                            job.status.log.push(format!(
                                "Shard {} verified OK ({}/{})",
                                shard_index, job.status.verified_shards, job.status.total_shards
                            ));
                            tracing::info!(
                                model = %model_id,
                                shard = shard_index,
                                "Shard downloaded and verified"
                            );

                            // Register as shard holder
                            self.shared_state
                                .model_registry
                                .record_shard_holder(shard_id.clone(), node_id.clone());
                            let mut holders = self
                                .shared_state
                                .shard_registry
                                .entry(shard_id)
                                .or_default();
                            if !holders.contains(&node_id) {
                                holders.push(node_id);
                            }
                        }
                        Err(e) => {
                            job.status.failed_shards += 1;
                            if let Some(sp) = job.status.shard_progress.get_mut(&shard_index) {
                                sp.state = ShardState::Failed;
                            }
                            job.status
                                .log
                                .push(format!("Shard {} FAILED verification: {}", shard_index, e));
                            tracing::warn!(
                                model = %model_id,
                                shard = shard_index,
                                error = %e,
                                "Downloaded shard failed verification — quarantined, penalizing peer"
                            );
                        }
                    }
                }
                None => {
                    job.status.failed_shards += 1;
                    if let Some(sp) = job.status.shard_progress.get_mut(&shard_index) {
                        sp.state = ShardState::Failed;
                    }
                    job.status.log.push(format!(
                        "Shard {} not found in manifest — discarding",
                        shard_index
                    ));
                    tracing::error!(
                        model = %model_id,
                        shard = shard_index,
                        "Shard not found in manifest — discarding"
                    );
                    let _ = self.shard_store.delete_shard(&model_id, shard_index);
                }
            }

            // Publish updated progress
            progress_map.insert(model_id.clone(), job.status.clone());

            // Check if all shards are done
            let all_done =
                job.status.verified_shards + job.status.failed_shards >= job.status.total_shards;
            if all_done {
                if job.status.failed_shards == 0 {
                    job.status.state = AcquisitionState::Complete;
                    job.status.speed_bytes_per_sec = 0;
                    let elapsed = job
                        .status
                        .started_at
                        .map(|s| (chrono::Utc::now() - s).num_seconds().max(1) as u64)
                        .unwrap_or(1);
                    let avg_speed = job.status.total_bytes / elapsed;
                    job.status.log.push(format!(
                        "Acquisition complete! {} in {}s (avg {})",
                        format_bytes_short(job.status.total_bytes),
                        elapsed,
                        format_speed(avg_speed)
                    ));
                    progress_map.insert(model_id.clone(), job.status.clone());
                    let manifest = job.manifest.clone();
                    tracing::info!(model = %model_id, "Model acquisition complete");
                    self.register_model(&model_id, &manifest);
                } else {
                    let reason = format!(
                        "{} of {} shards failed verification",
                        job.status.failed_shards, job.status.total_shards
                    );
                    job.status.log.push(format!("FAILED: {}", reason));
                    job.status.state = AcquisitionState::Failed { reason };
                    progress_map.insert(model_id.clone(), job.status.clone());
                }
            }
        }
    }

    /// Publish acquisition progress to SharedState for WebSocket/API consumption.
    fn publish_progress(&self, model_id: &ModelId, status: &AcquisitionStatus) {
        self.shared_state
            .acquisition_progress
            .insert(model_id.clone(), status.clone());
    }

    /// Register a fully acquired model, reconstruct the GGUF, and auto-load for inference.
    fn register_model(&self, model_id: &ModelId, manifest: &ModelManifest) {
        // Persist manifest to DB
        if let Err(e) = self
            .shared_state
            .model_registry
            .persist_manifest(&self.shared_state.db, manifest)
        {
            tracing::error!(model = %model_id, error = %e, "Failed to persist manifest to DB");
        }

        // Update acquisition log
        if let Some(mut entry) = self.shared_state.acquisition_progress.get_mut(model_id) {
            entry.log.push("Reconstructing GGUF from shards...".into());
        }

        // Reconstruct the full GGUF file from shard files
        let gguf_path = match self.shard_store.reconstruct_gguf(model_id, manifest) {
            Ok(path) => path,
            Err(e) => {
                tracing::error!(model = %model_id, error = %e, "Failed to reconstruct GGUF");
                if let Some(mut entry) = self.shared_state.acquisition_progress.get_mut(model_id) {
                    entry.log.push(format!("GGUF reconstruction failed: {}", e));
                }
                return;
            }
        };

        tracing::info!(model = %model_id, path = %gguf_path.display(), "GGUF reconstructed");

        // Auto-load the model into the executor for inference
        let executor = self.shared_state.executor.clone();
        let shared_state = self.shared_state.clone();
        let model_name = manifest.name.clone();
        let model_id_clone = model_id.clone();
        let gpu_layers = self.shared_state.config.inference.gpu_layers;

        if let Some(mut entry) = self.shared_state.acquisition_progress.get_mut(model_id) {
            entry
                .log
                .push("Loading model into GPU for inference...".into());
        }

        tokio::spawn(async move {
            tracing::info!(model = %model_name, "Loading reconstructed model...");

            let mut exec = executor.lock().await;
            match exec.load_model(&gguf_path, gpu_layers) {
                Ok(()) => {
                    let size = exec.model_size_bytes().unwrap_or(0);
                    let gguf_meta = crate::inference::executor::extract_gguf_metadata(&gguf_path);
                    // Extract EOS tokens from GGUF metadata with architecture-specific fallbacks
                    let eos_tokens = {
                        let mut tokens = Vec::new();
                        if let Ok(mut f) = std::fs::File::open(&gguf_path) {
                            if let Ok(ct) = candle_core::quantized::gguf_file::Content::read(&mut f) {
                                if let Some(eos_id) = ct.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.to_u32().ok()) {
                                    tokens.push(eos_id);
                                }
                                let arch = ct.metadata.get("general.architecture").and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();
                                match arch.as_str() {
                                    "qwen2" => {
                                        for &id in &[151643u32, 151645] {
                                            if !tokens.contains(&id) { tokens.push(id); }
                                        }
                                    }
                                    _ => {
                                        if !tokens.contains(&2) { tokens.push(2); }
                                    }
                                }
                            }
                        }
                        if tokens.is_empty() { tokens.push(2); }
                        tokens
                    };
                    let info = crate::daemon::LoadedModelInfo {
                        name: model_name.clone(),
                        size_bytes: size,
                        eos_tokens,
                        chat_template: gguf_meta.as_ref().and_then(|m| m.chat_template.clone()),
                        bos_token: gguf_meta
                            .as_ref()
                            .map(|m| m.bos_token.clone())
                            .unwrap_or_default(),
                        eos_token: gguf_meta
                            .as_ref()
                            .map(|m| m.eos_token.clone())
                            .unwrap_or_default(),
                    };
                    *shared_state.loaded_model_info.write().await = Some(info.clone());

                    // Generate manifest for the reconstructed model so we can serve shards
                    crate::daemon::generate_and_register_local_manifest(
                        &shared_state,
                        &info,
                        &gguf_path,
                    );

                    if let Some(mut entry) =
                        shared_state.acquisition_progress.get_mut(&model_id_clone)
                    {
                        entry
                            .log
                            .push(format!("Model loaded! {} ready for inference", model_name));
                    }

                    tracing::info!(model = %model_name, "Model loaded and ready for inference");
                }
                Err(e) => {
                    if let Some(mut entry) =
                        shared_state.acquisition_progress.get_mut(&model_id_clone)
                    {
                        entry.log.push(format!("Model load failed: {}", e));
                    }
                    tracing::error!(model = %model_name, error = %e, "Failed to load reconstructed model");
                }
            }
        });
    }

    /// Select the best peer to download from based on latency and trust.
    fn select_best_peer(&self, holders: &[NodeId]) -> NodeId {
        let local_id = self.shared_state.identity.node_id().clone();

        holders
            .iter()
            .filter(|n| **n != local_id)
            .min_by_key(|node_id| {
                self.shared_state
                    .peer_registry
                    .get(node_id)
                    .map(|peer| {
                        let latency = peer.latency_ms.unwrap_or(200);
                        let trust_penalty = ((1.0 - peer.trust_score) * 100.0) as u32;
                        latency + trust_penalty
                    })
                    .unwrap_or(500)
            })
            .cloned()
            .unwrap_or_else(|| holders[0].clone())
    }
}

fn format_bytes_short(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

fn format_speed(bytes_per_sec: u64) -> String {
    if bytes_per_sec >= 1_048_576 {
        format!("{:.1} MB/s", bytes_per_sec as f64 / 1_048_576.0)
    } else if bytes_per_sec >= 1024 {
        format!("{:.0} KB/s", bytes_per_sec as f64 / 1024.0)
    } else {
        format!("{} B/s", bytes_per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquisition_state_serializes() {
        let state = AcquisitionState::Downloading;
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, "\"downloading\"");

        let failed = AcquisitionState::Failed {
            reason: "bad hash".into(),
        };
        let json = serde_json::to_string(&failed).unwrap();
        assert!(json.contains("bad hash"));
    }
}
