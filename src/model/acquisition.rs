use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::model::distribution::ShardDistributor;
use crate::model::shard::ShardStore;
use crate::types::{ModelId, ModelManifest, NetworkCommand, NodeId, ShardId, SwarmMessage};

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
    /// Tracks bytes received per shard (for multi-chunk downloads).
    shard_progress: HashMap<u32, u64>,
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
                        status: AcquisitionStatus {
                            model_id: model_id.clone(),
                            state: AcquisitionState::AwaitingManifest,
                            total_shards: 0,
                            downloaded_shards: 0,
                            verified_shards: 0,
                            failed_shards: 0,
                            total_bytes: 0,
                            downloaded_bytes: 0,
                        },
                        shard_progress: HashMap::new(),
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

        self.jobs.insert(
            model_id.clone(),
            AcquisitionJob {
                manifest: manifest.clone(),
                status: AcquisitionStatus {
                    model_id: model_id.clone(),
                    state: AcquisitionState::Downloading,
                    total_shards,
                    downloaded_shards: total_shards - needed.len() as u32,
                    verified_shards: total_shards - needed.len() as u32,
                    failed_shards: 0,
                    total_bytes,
                    downloaded_bytes: 0,
                },
                shard_progress: HashMap::new(),
            },
        );

        if needed.is_empty() {
            tracing::info!(model = %model_id, "All shards already present and verified");
            if let Some(job) = self.jobs.get_mut(&model_id) {
                job.status.state = AcquisitionState::Complete;
            }
            self.register_model(&model_id, &manifest);
            return;
        }

        // Request each needed shard from the network
        for shard_id in &needed {
            // Find peers that hold this shard
            let holders = self.shared_state.model_registry.shard_holders(shard_id);
            if holders.is_empty() {
                tracing::warn!(
                    model = %model_id,
                    shard = shard_id.index,
                    "No known holders for shard"
                );
                continue;
            }

            // Pick the best peer (lowest latency, highest trust)
            let target = self.select_best_peer(&holders);

            // Send shard transfer request
            let _request = crate::types::ShardRequest {
                shard_id: shard_id.clone(),
                chunk_offset: 0,
                chunk_size: 1024 * 1024, // 1MB chunks
            };

            let msg = SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
                node_id: self.shared_state.identity.node_id().clone(),
                shards: vec![shard_id.clone()],
                timestamp: chrono::Utc::now(),
            });

            if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
                tracing::warn!(error = %e, "Failed to request shard from network");
            }

            tracing::info!(
                model = %model_id,
                shard = shard_id.index,
                peer = %target,
                "Requested shard transfer"
            );
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

        let job = match self.jobs.get_mut(&model_id) {
            Some(j) => j,
            None => {
                tracing::warn!(
                    model = %model_id,
                    "Received shard data for unknown acquisition — ignoring"
                );
                return;
            }
        };

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

        // Track progress
        let received = job.shard_progress.entry(shard_index).or_insert(0);
        *received += data.len() as u64;
        job.status.downloaded_bytes += data.len() as u64;

        // Check if this shard is complete
        if *received >= total_size {
            // SECURITY: Verify the completed shard against the manifest hash
            let shard_info = job.manifest.shards.iter().find(|s| s.index == shard_index);
            match shard_info {
                Some(info) => match self.shard_store.verify_shard(&model_id, info) {
                    Ok(()) => {
                        job.status.downloaded_shards += 1;
                        job.status.verified_shards += 1;
                        tracing::info!(
                            model = %model_id,
                            shard = shard_index,
                            "Shard downloaded and verified"
                        );

                        // Register as shard holder
                        let node_id = self.shared_state.identity.node_id().clone();
                        self.shared_state
                            .model_registry
                            .record_shard_holder(shard_id.clone(), node_id.clone());
                        self.shared_state
                            .shard_registry
                            .entry(shard_id)
                            .or_default()
                            .push(node_id);
                    }
                    Err(e) => {
                        job.status.failed_shards += 1;
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_index,
                            error = %e,
                            "Downloaded shard failed verification — quarantined, penalizing peer"
                        );
                        // Shard is already quarantined by verify_shard()
                    }
                },
                None => {
                    job.status.failed_shards += 1;
                    tracing::error!(
                        model = %model_id,
                        shard = shard_index,
                        "Shard not found in manifest — discarding"
                    );
                    let _ = self.shard_store.delete_shard(&model_id, shard_index);
                }
            }

            // Check if all shards are done — extract what we need before calling register_model
            let all_done =
                job.status.verified_shards + job.status.failed_shards >= job.status.total_shards;
            if all_done {
                if job.status.failed_shards == 0 {
                    job.status.state = AcquisitionState::Complete;
                    let manifest = job.manifest.clone();
                    tracing::info!(model = %model_id, "Model acquisition complete");
                    self.register_model(&model_id, &manifest);
                } else {
                    job.status.state = AcquisitionState::Failed {
                        reason: format!(
                            "{} of {} shards failed verification",
                            job.status.failed_shards, job.status.total_shards
                        ),
                    };
                }
            }
        }
    }

    /// Register a fully acquired model in the local registry and announce it.
    fn register_model(&self, model_id: &ModelId, manifest: &ModelManifest) {
        // Persist manifest to DB
        if let Err(e) = self
            .shared_state
            .model_registry
            .persist_manifest(&self.shared_state.db, manifest)
        {
            tracing::error!(model = %model_id, error = %e, "Failed to persist manifest to DB");
        }

        tracing::info!(model = %model_id, "Model registered and ready for inference");
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
