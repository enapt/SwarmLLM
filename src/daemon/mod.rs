use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::config::Config;
use crate::credit::ledger::CreditLedger;
use crate::health::monitor::HealthMonitor;
use crate::health::rebalancer::ShardRebalancer;
use crate::identity::Identity;
use crate::inference::router::{InferenceRouter, RouterCommand};
use crate::model::acquisition::{AcquisitionCommand, AcquisitionManager};
use crate::model::manifest::ModelManifestExt;
use crate::model::shard::ShardStore;
use crate::network::manager::NetworkManager;
use crate::storage::db::Database;
use crate::types::{AuthenticatedMessage, NetworkCommand, RebalanceEvent, ShardId, SwarmMessage};
use tokio::sync::RwLock;

mod dispatch;
pub mod manifest;
pub mod shard_loader;
pub mod state;

// Re-export public types so callers use crate::daemon::SharedState etc.
pub use dispatch::estimate_vram_from_shard_dir;
pub use manifest::generate_and_register_local_manifest;
pub use shard_loader::{try_load_from_shards, ShardLoadParams};
pub use state::*;

use dispatch::dispatch_network_messages;
use manifest::{extract_tied_output_weight, regenerate_manifest_from_header};

/// Maximum restart attempts before a subsystem is considered permanently failed.
const MAX_RESTART_ATTEMPTS: u32 = 5;

/// Whether a subsystem is critical to daemon operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemCriticality {
    /// Daemon must shut down if this subsystem permanently fails.
    Critical,
    /// Daemon can continue without this subsystem.
    NonCritical,
}

/// Top-level daemon orchestrating all SwarmLLM subsystems.
pub struct Daemon {
    config: Config,
    identity: Identity,
    db: Database,
}

impl Daemon {
    pub fn new(config: Config, identity: Identity, db: Database) -> Self {
        Self {
            config,
            identity,
            db,
        }
    }

    /// Run the daemon — spawns all subsystems and waits for shutdown.
    pub async fn run(self) -> anyhow::Result<()> {
        // Load .env file from data dir (or cwd) into process environment
        crate::config::load_dotenv(&self.config.node.data_dir);

        // Log detected provider API keys from environment
        let env_keys = crate::config::ProvidersConfig::detect_env_keys();
        if !env_keys.is_empty() {
            let names: Vec<&str> = env_keys.iter().map(|(_, name)| *name).collect();
            tracing::info!(
                providers = ?names,
                count = env_keys.len(),
                "Detected provider API keys in environment"
            );
        }

        // Log resolved configuration at startup
        let auto_interval = self
            .config
            .auto_manage
            .interval_seconds
            .map(|s| format!("{s}s"))
            .unwrap_or_else(|| format!("{}m", self.config.auto_manage.interval_minutes));
        tracing::info!(
            port = self.config.node.listen_port,
            data_dir = %self.config.node.data_dir.display(),
            bootstrap_peers = self.config.network.bootstrap_peers.len(),
            auto_manage = self.config.auto_manage.enabled,
            "SwarmLLM daemon starting with resolved config"
        );
        tracing::debug!(
            port = self.config.node.listen_port,
            data_dir = %self.config.node.data_dir.display(),
            bootstrap_peers = self.config.network.bootstrap_peers.len(),
            auto_manage_enabled = self.config.auto_manage.enabled,
            auto_manage_interval = %auto_interval,
            max_concurrent_requests = self.config.inference.max_concurrent_requests,
            shard_size_mb = self.config.model.shard_size_mb,
            log_level = %self.config.logging.level,
            max_peers = self.config.network.max_peers,
            session_timeout_secs = self.config.inference.session_timeout_seconds,
            relay_enabled = self.config.network.enable_relay,
            "Full resolved configuration"
        );

        // Run database integrity check before spawning subsystems
        let integrity_report = self.db.check_integrity();
        if integrity_report.total_corrupt > 0 {
            tracing::warn!(
                corrupt_entries = integrity_report.total_corrupt,
                "Database integrity issues detected — some entries may be skipped"
            );
        }

        // Initialize model executor
        let mut executor = crate::inference::executor::ModelExecutor::new();
        if let Some(ref model_path) = self.config.inference.model_path {
            match executor.load_model(model_path, self.config.inference.gpu_layers) {
                Ok(()) => tracing::info!("Model ready"),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load model — running without inference")
                }
            }
        }

        // Gather model info for manifest generation and admin display.
        // Extract GGUF metadata (chat template, special tokens) if available.
        let gguf_meta = self
            .config
            .inference
            .model_path
            .as_ref()
            .and_then(|p| crate::inference::executor::extract_gguf_metadata(p));

        let model_info = if executor.is_loaded() {
            Some(LoadedModelInfo {
                name: executor.model_name().to_string(),
                size_bytes: executor.model_size_bytes().unwrap_or(0),
                eos_tokens: vec![2], // Default; updated when split model loads with GGUF metadata
                chat_template: gguf_meta.as_ref().and_then(|m| m.chat_template.clone()),
                bos_token: gguf_meta
                    .as_ref()
                    .map(|m| m.bos_token.clone())
                    .unwrap_or_default(),
                eos_token: gguf_meta
                    .as_ref()
                    .map(|m| m.eos_token.clone())
                    .unwrap_or_default(),
            })
        } else {
            None
        };

        // When --shards is set, the node only holds part of the model — don't
        // report a fully loaded model, which would cause the API to serve
        // requests through the (incomplete) local executor.
        let cached_info = if self.config.inference.shard_range.is_some() {
            if let Some(ref info) = model_info {
                tracing::info!(
                    model = %info.name,
                    "Model available for split inference (not full-model serving)"
                );
            }
            None
        } else {
            model_info.clone()
        };

        // Detect GPU via llama.cpp backend; fall back to candle CUDA probe
        let gpu_info = {
            let llama_gpu = crate::inference::executor::detect_gpu();
            #[cfg(feature = "candle-cuda")]
            let gpu_info = llama_gpu.or_else(|| {
                let cuda_ok = candle_core::Device::cuda_if_available(0)
                    .map(|d| d.is_cuda())
                    .unwrap_or(false);
                if cuda_ok {
                    let (name, vram_mb) = crate::api::admin::detect_gpu_nvidia_smi();
                    Some(crate::inference::executor::GpuInfo {
                        name: name.unwrap_or_else(|| "NVIDIA GPU".to_string()),
                        vram_total_mb: vram_mb.unwrap_or(0),
                        vram_free_mb: 0,
                        backend: "CUDA".to_string(),
                    })
                } else {
                    None
                }
            });
            #[cfg(not(feature = "candle-cuda"))]
            let gpu_info = llama_gpu;
            gpu_info
        };
        if let Some(ref gpu) = gpu_info {
            tracing::info!(gpu = %gpu.name, vram_mb = gpu.vram_total_mb, backend = %gpu.backend, "GPU detected");
        }

        let executor = Arc::new(tokio::sync::Mutex::new(executor));

        // Create shared state
        let (shared_state, mut shutdown_rx) = SharedState::new(
            self.config.clone(),
            self.identity.clone(),
            self.db.clone(),
            executor,
            gpu_info,
        );

        *shared_state.loaded_model_info.write().await = cached_info;

        // Not set in shard/split mode — those nodes use split_models instead.
        if model_info.is_some() && self.config.inference.shard_range.is_none() {
            shared_state
                .model_loaded
                .store(true, std::sync::atomic::Ordering::Release);
        }

        // Load draft model for speculative decoding if configured
        if self.config.inference.speculative_decoding {
            if let Some(ref draft_path) = self.config.inference.draft_model_path {
                let draft_gpu_layers = self
                    .config
                    .inference
                    .draft_gpu_layers
                    .unwrap_or(self.config.inference.gpu_layers);
                let mut draft = shared_state.draft_executor.lock().await;
                match draft.load_model(draft_path, draft_gpu_layers) {
                    Ok(()) => tracing::info!(
                        draft_model = %draft.model_name(),
                        gamma = self.config.inference.speculative_gamma,
                        "Draft model loaded for speculative decoding"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Failed to load draft model — falling back to standard decoding"
                    ),
                }
            } else {
                tracing::info!("Speculative decoding enabled but no draft_model_path configured");
            }
        }

        // Generate a ModelManifest for the locally loaded model so peers can discover it.
        // This is needed even in split mode so the shard registry gets populated.
        if let Some(ref info) = model_info {
            if let Some(ref model_path) = self.config.inference.model_path {
                generate_and_register_local_manifest(&shared_state, info, model_path);
            }
        }

        // Restore persisted manifests from the DB and register shard holders.
        // This handles the case where a node restarts with --shards but no --model:
        // the manifest was generated in a previous run and persisted, so we restore
        // it and re-register ourselves as holder of our shard range.
        {
            let node_id = shared_state.identity.node_id().clone();
            let shard_range = self.config.inference.shard_range;
            if let Ok(manifests) = self
                .db
                .iter_json::<crate::types::ModelManifest>("model_meta")
            {
                for manifest in manifests {
                    let model_id = manifest.id.clone();
                    // Verify manifest hash before trusting DB data (MOD-I2)
                    if manifest.verify_hash().is_err() {
                        tracing::warn!(
                            model = %model_id,
                            "Manifest from DB failed hash verification — skipping"
                        );
                        continue;
                    }
                    // Register the manifest if not already in-memory
                    if shared_state
                        .model_registry
                        .get_manifest(&model_id)
                        .is_none()
                    {
                        shared_state
                            .model_registry
                            .register_manifest(manifest.clone());
                        tracing::info!(
                            model = %model_id,
                            shards = manifest.shard_count,
                            "Restored manifest from DB"
                        );
                    }
                    // Register ourselves as holder of our shard range
                    let shard_store_reg = ShardStore::new(&self.config.node.data_dir);
                    for shard_info in &manifest.shards {
                        let in_range = match shard_range {
                            Some((start, end)) => {
                                shard_info.index >= start && shard_info.index <= end
                            }
                            None => true,
                        };
                        if in_range {
                            // Verify the shard file actually exists on disk before registering
                            let shard_path =
                                shard_store_reg.shard_path(&model_id, shard_info.index);
                            if !shard_path.exists() {
                                tracing::warn!(
                                    model = %model_id,
                                    shard = shard_info.index,
                                    path = %shard_path.display(),
                                    "Shard file missing on disk — skipping registration"
                                );
                                continue;
                            }
                            let shard_id = crate::types::ShardId {
                                model_id: model_id.clone(),
                                index: shard_info.index,
                            };
                            shared_state
                                .model_registry
                                .record_shard_holder(shard_id, node_id.clone());
                        }
                    }
                    // Load GGUF metadata for the model if we have a source path
                    if !shared_state.gguf_meta.contains_key(&model_id) {
                        let shard_store_tmp = ShardStore::new(&self.config.node.data_dir);
                        let model_dir = shard_store_tmp.models_dir().join(&model_id.0);
                        let source_path_file = model_dir.join("source_path");
                        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
                            let path = std::path::PathBuf::from(path_str.trim());
                            // SEC: Containment check — source_path must be within data directory
                            let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                            let data_models = shard_store_tmp.models_dir();
                            if !canonical.starts_with(&data_models) {
                                tracing::warn!(
                                    model = %model_id,
                                    path = %path.display(),
                                    "source_path outside data directory — ignoring"
                                );
                            } else if let Ok(meta) =
                                crate::inference::split::GgufTensorMeta::from_gguf_file(&path)
                            {
                                tracing::info!(
                                    model = %model_id,
                                    layers = meta.block_count,
                                    "Loaded GGUF metadata from source path"
                                );
                                shared_state.gguf_meta.insert(model_id.clone(), meta);
                            }
                        }
                    }
                }
            }
        }

        // Pre-pass: regenerate any missing manifests from GGUF headers + shard files.
        // load_all_local() requires a manifest to exist (security check), so we must
        // create one first if gguf_header.bin + shard files are present.
        let shard_store = ShardStore::new(&self.config.node.data_dir);
        {
            let models_dir = shard_store.models_dir();
            if models_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    for entry in entries.flatten() {
                        let model_dir = entry.path();
                        if !model_dir.is_dir() {
                            continue;
                        }
                        let manifest_path = model_dir.join("manifest.json");
                        let header_path = model_dir.join("gguf_header.bin");
                        // If header is missing, try to extract it from shard_000.bin
                        if !header_path.exists() {
                            let _ = crate::inference::split::ensure_gguf_header(&model_dir);
                        }
                        if !manifest_path.exists() && header_path.exists() {
                            let model_id_str = model_dir
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("")
                                .to_string();
                            let model_id = crate::types::ModelId(model_id_str);
                            if let Ok(meta) =
                                crate::inference::split::GgufTensorMeta::from_gguf_file(
                                    &header_path,
                                )
                            {
                                tracing::info!(
                                    model = %model_id,
                                    "Regenerating missing manifest from GGUF header"
                                );
                                if regenerate_manifest_from_header(
                                    &model_id,
                                    &model_dir,
                                    &meta,
                                    &self.config,
                                )
                                .is_some()
                                {
                                    shared_state.gguf_meta.insert(model_id, meta);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clean up leftover .tmp files from interrupted downloads
        shard_store.cleanup_tmp_files();

        // Scan local shards and register them + their manifests
        match shard_store.load_all_local() {
            Ok(shards) => {
                // Track which model manifests we've already registered
                let mut registered_manifests = std::collections::HashSet::new();

                for (model_id, shard_info) in &shards {
                    // Register the manifest if we haven't yet
                    if registered_manifests.insert(model_id.clone()) {
                        let model_dir = shard_store.models_dir().join(&model_id.0);

                        // Ensure GGUF header exists (extract from shard_000 if available)
                        // and load GGUF metadata for split inference.
                        // Do this BEFORE loading manifest so we can regenerate if needed.
                        if !shared_state.gguf_meta.contains_key(model_id) {
                            if let Ok(()) = crate::inference::split::ensure_gguf_header(&model_dir)
                            {
                                let header_path = model_dir.join("gguf_header.bin");
                                if let Ok(meta) =
                                    crate::inference::split::GgufTensorMeta::from_gguf_file(
                                        &header_path,
                                    )
                                {
                                    tracing::info!(
                                        model = %model_id,
                                        layers = meta.block_count,
                                        "Loaded GGUF metadata from shard header"
                                    );
                                    shared_state.gguf_meta.insert(model_id.clone(), meta);
                                }
                            }
                        }

                        let manifest_loaded = if let Ok(manifest) =
                            crate::types::ModelManifest::load_from_dir(&model_dir)
                        {
                            if manifest.verify_hash().is_ok() {
                                shared_state
                                    .model_registry
                                    .register_manifest(manifest.clone());
                                if let Err(e) = shared_state
                                    .model_registry
                                    .persist_manifest(&shared_state.db, &manifest)
                                {
                                    tracing::warn!(error = %e, "Failed to persist manifest to DB");
                                }
                                tracing::info!(
                                    model = %model_id,
                                    shards = manifest.shard_count,
                                    "Registered manifest from local shard directory"
                                );
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Regenerate manifest if missing/invalid and GGUF header available
                        if !manifest_loaded {
                            if let Some(meta) = shared_state.gguf_meta.get(model_id) {
                                tracing::info!(
                                    model = %model_id,
                                    "Regenerating manifest from GGUF header + shard files"
                                );
                                if let Some(manifest) = regenerate_manifest_from_header(
                                    model_id,
                                    &model_dir,
                                    &meta,
                                    &shared_state.config,
                                ) {
                                    shared_state
                                        .model_registry
                                        .register_manifest(manifest.clone());
                                    let _ = shared_state
                                        .model_registry
                                        .persist_manifest(&shared_state.db, &manifest);
                                }
                            }
                        }

                        // Auto-extract tied_output_weight.bin for weight-tied models.
                        // If shard_000 is available locally, extract token_embd.weight
                        // so nodes with the last segment can project logits even without
                        // shard_000 (in distributed inference, another node may have it).
                        let tied_path = model_dir.join("tied_output_weight.bin");
                        if !tied_path.exists() {
                            if let Some(meta) = shared_state.gguf_meta.get(model_id) {
                                let has_output = meta.tensors.contains_key("output.weight");
                                let has_embd = meta.tensors.contains_key("token_embd.weight");
                                if !has_output && has_embd {
                                    // Weight-tied model — try to extract from shard_000
                                    let shard0_path = model_dir.join("shard_000.bin");
                                    if shard0_path.exists() {
                                        if let Err(e) = extract_tied_output_weight(
                                            &shard0_path,
                                            &model_dir,
                                            &meta,
                                        ) {
                                            tracing::warn!(
                                                model = %model_id,
                                                error = %e,
                                                "Failed to extract tied_output_weight.bin from shard_000"
                                            );
                                        }
                                    }
                                }
                            }

                            // Local embedding privacy: load embedding table from shard_000
                            // so the requesting node can embed locally before sending to peers.
                            if shared_state.config.inference.local_embedding_privacy
                                && !shared_state.local_embedders.contains_key(model_id)
                            {
                                let shard0_path = model_dir.join("shard_000.bin");
                                if shard0_path.exists() {
                                    match crate::inference::local_embedder::LocalEmbedder::load(
                                        &shard0_path,
                                    ) {
                                        Ok(embedder) => {
                                            shared_state.local_embedders.insert(
                                                model_id.clone(),
                                                std::sync::Arc::new(embedder),
                                            );
                                            tracing::info!(
                                                model = %model_id,
                                                "Loaded local embedding table for privacy mode"
                                            );
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                model = %model_id,
                                                error = %e,
                                                "Failed to load local embedder from shard_000"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let shard_id = ShardId {
                        model_id: model_id.clone(),
                        index: shard_info.index,
                    };
                    let node_id = shared_state.identity.node_id().clone();
                    shared_state
                        .model_registry
                        .record_shard_holder(shard_id, node_id);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
            }
        }

        // Register local mmproj files as sentinel shards.
        {
            let models_dir = self.config.node.data_dir.join("models");
            if models_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    let node_id = shared_state.identity.node_id().clone();
                    for entry in entries.flatten() {
                        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let mmproj_path = entry.path().join("mmproj.gguf");
                        if mmproj_path.exists() {
                            let model_id_str = entry.file_name().to_string_lossy().to_string();
                            let model_id = crate::types::ModelId(model_id_str.clone());
                            let mmproj_sid = ShardId::mmproj_for(model_id);
                            shared_state
                                .model_registry
                                .record_shard_holder(mmproj_sid, node_id.clone());
                            tracing::info!(
                                model = %model_id_str,
                                "Registered local mmproj.gguf as vision encoder shard"
                            );
                        }
                    }
                }
            }
        }

        // Discover HF sources from hf_source.json files alongside manifests.
        // Models always originate from HuggingFace, so this ensures the source
        // is known even after a DB wipe or fresh node with pre-seeded shards.
        {
            let models_dir = self.config.node.data_dir.join("models");
            if models_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    for entry in entries.flatten() {
                        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let model_id_str = entry.file_name().to_string_lossy().to_string();
                        let mid = crate::types::ModelId(model_id_str.clone());
                        if shared_state.hf_sources.contains_key(&mid) {
                            continue;
                        }
                        let hf_path = entry.path().join("hf_source.json");
                        if hf_path.exists() {
                            if let Ok(data) = std::fs::read_to_string(&hf_path) {
                                if let Ok(source) = serde_json::from_str::<HfSource>(&data) {
                                    tracing::info!(
                                        model = %model_id_str,
                                        repo = %source.repo_id,
                                        file = %source.filename,
                                        "Loaded HF source from disk"
                                    );
                                    shared_state.hf_sources.insert(mid.clone(), source.clone());
                                    let _ = self.db.put_json("hf_sources", &model_id_str, &source);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Auto-download missing GGUF headers from HuggingFace.
        // If a model directory has shard files but no gguf_header.bin (and no shard_000
        // to extract it from), try to download the header using hf_source.json.
        {
            let models_dir = self.config.node.data_dir.join("models");
            if models_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    for entry in entries.flatten() {
                        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let model_dir = entry.path();
                        let header_path = model_dir.join("gguf_header.bin");
                        if header_path.exists() {
                            continue; // Already have header
                        }
                        // Check if we have any shard files
                        let has_shards = model_dir.join("shard_000.bin").exists()
                            || model_dir.join("shard_001.bin").exists();
                        if !has_shards {
                            continue;
                        }
                        // Try local extraction from shard_000 first
                        if model_dir.join("shard_000.bin").exists()
                            && crate::inference::split::ensure_gguf_header(&model_dir).is_ok()
                        {
                            continue;
                        }
                        // Download from HF if source is known
                        let model_id_str = entry.file_name().to_string_lossy().to_string();
                        let mid = crate::types::ModelId(model_id_str.clone());
                        if let Some(hf_src) = shared_state.hf_sources.get(&mid) {
                            tracing::info!(
                                model = %model_id_str,
                                repo = %hf_src.repo_id,
                                "Downloading GGUF header from HuggingFace (no local shard_000)"
                            );
                            let shard_size = self.config.model.shard_size_bytes();
                            match crate::model::huggingface::probe_gguf_file(
                                &hf_src.repo_id,
                                &hf_src.filename,
                                shard_size,
                            )
                            .await
                            {
                                Ok(info) => {
                                    if let Ok(hp) = crate::model::huggingface::download_gguf_header(
                                        &hf_src.repo_id,
                                        &hf_src.filename,
                                        &model_dir,
                                        info.header_size,
                                    )
                                    .await
                                    {
                                        tracing::info!(
                                            model = %model_id_str,
                                            path = %hp.display(),
                                            "Downloaded GGUF header from HuggingFace"
                                        );
                                        // Load GGUF metadata
                                        if let Ok(meta) =
                                            crate::inference::split::GgufTensorMeta::from_gguf_file(
                                                &hp,
                                            )
                                        {
                                            shared_state
                                                .gguf_meta
                                                .insert(mid.clone(), meta.clone());
                                            // Regenerate manifest if missing
                                            let manifest_path = model_dir.join("manifest.json");
                                            if !manifest_path.exists() {
                                                regenerate_manifest_from_header(
                                                    &mid,
                                                    &model_dir,
                                                    &meta,
                                                    &self.config,
                                                );
                                            }
                                            // Extract tied output weight if needed
                                            let tied_path =
                                                model_dir.join("tied_output_weight.bin");
                                            if !tied_path.exists() {
                                                let has_output =
                                                    meta.tensors.contains_key("output.weight");
                                                let has_embd =
                                                    meta.tensors.contains_key("token_embd.weight");
                                                if !has_output && has_embd {
                                                    if let Err(e) = crate::model::huggingface::download_tied_output_weight(
                                                        &hf_src.repo_id,
                                                        &hf_src.filename,
                                                        &model_dir,
                                                        &meta,
                                                    ).await {
                                                        tracing::warn!(
                                                            model = %model_id_str,
                                                            error = %e,
                                                            "Failed to download tied_output_weight"
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        model = %model_id_str,
                                        error = %e,
                                        "Failed to probe GGUF on HuggingFace for header download"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Channel Architecture ──
        //
        // network_tx      → NetworkManager (outbound commands: broadcast, send tensor)
        // network_out_tx  → from NetworkManager (inbound decoded messages)
        // router_cmd_tx   → InferenceRouter (commands from API + network)
        // rebalance_tx    → ShardRebalancer (events from HealthMonitor)
        // acquisition_tx  → AcquisitionManager (model download commands from API)
        //
        let (network_tx, network_rx) = mpsc::channel::<NetworkCommand>(1024);
        let (network_out_tx, mut network_out_rx) = mpsc::channel::<AuthenticatedMessage>(1024);
        let (router_cmd_tx, router_cmd_rx) = mpsc::channel::<RouterCommand>(256);
        let (rebalance_tx, rebalance_rx) = mpsc::channel::<RebalanceEvent>(64);
        let (acquisition_tx, acquisition_rx) = mpsc::channel::<AcquisitionCommand>(64);

        // ── Subsystem Supervisor (JoinSet) ──
        //
        // All 10 subsystem tasks are spawned into a JoinSet for unified monitoring.
        // Each task returns (name, criticality, result) so the supervisor loop
        // can decide whether to trigger shutdown or continue degraded.
        //
        let mut subsystems: JoinSet<(&'static str, SubsystemCriticality, Result<(), String>)> =
            JoinSet::new();

        // Spawn NetworkManager (acquisition_tx wired after channel creation below)
        let network_manager = NetworkManager::new(
            shared_state.clone(),
            &self.identity,
            &self.config,
            network_rx,
            network_out_tx,
            shutdown_rx.clone(),
            Some(acquisition_tx.clone()),
        )?;

        subsystems.spawn(async move {
            let result = network_manager.run().await.map_err(|e| e.to_string());
            ("NetworkManager", SubsystemCriticality::Critical, result)
        });

        let inference_router = InferenceRouter::new(
            shared_state.clone(),
            router_cmd_rx,
            router_cmd_tx.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = inference_router.run().await.map_err(|e| e.to_string());
            ("InferenceRouter", SubsystemCriticality::Critical, result)
        });

        // Spawn message dispatcher: routes network inbound messages to the right subsystem
        let dispatcher_credit_balances: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
        let dispatcher_router_tx = router_cmd_tx.clone();
        let dispatcher_shutdown = shutdown_rx.clone();
        let dispatcher_credit_ref = dispatcher_credit_balances.clone();
        let dispatcher_state = shared_state.clone();
        let dispatcher_network_tx = network_tx.clone();
        subsystems.spawn(async move {
            dispatch_network_messages(
                &mut network_out_rx,
                &dispatcher_router_tx,
                dispatcher_credit_ref,
                &dispatcher_state,
                dispatcher_network_tx,
                dispatcher_shutdown,
            )
            .await;
            ("MessageDispatcher", SubsystemCriticality::Critical, Ok(()))
        });

        let health_monitor = HealthMonitor::new(
            shared_state.clone(),
            network_tx.clone(),
            rebalance_tx,
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = health_monitor.run().await.map_err(|e| e.to_string());
            ("HealthMonitor", SubsystemCriticality::NonCritical, result)
        });

        let shard_rebalancer = ShardRebalancer::new(
            shared_state.clone(),
            rebalance_rx,
            network_tx.clone(),
            acquisition_tx.clone(),
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = shard_rebalancer.run().await.map_err(|e| e.to_string());
            ("ShardRebalancer", SubsystemCriticality::NonCritical, result)
        });

        // Spawn CreditLedger — shares the same Arc<RwLock<CreditBalance>> as SharedState
        let mut credit_ledger = CreditLedger::new(
            shared_state.identity.node_id().clone(),
            shared_state.credit_balance.clone(),
            self.db.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
            dispatcher_credit_balances.clone(),
        );
        credit_ledger.set_shared_state(shared_state.clone());
        credit_ledger.set_identity(shared_state.identity.clone());

        subsystems.spawn(async move {
            let result = credit_ledger.run().await.map_err(|e| e.to_string());
            ("CreditLedger", SubsystemCriticality::NonCritical, result)
        });

        let acquisition_manager = AcquisitionManager::new(
            shared_state.clone(),
            network_tx.clone(),
            acquisition_rx,
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = acquisition_manager.run().await.map_err(|e| e.to_string());
            (
                "AcquisitionManager",
                SubsystemCriticality::NonCritical,
                result,
            )
        });

        let (pool_cmd_tx, pool_cmd_rx) = mpsc::channel::<crate::pool::types::PoolCommand>(64);
        {
            *shared_state.pool_tx.write().await = Some(pool_cmd_tx);
        }
        let pool_manager = crate::pool::manager::PoolManager::new(
            shared_state.clone(),
            pool_cmd_rx,
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        subsystems.spawn(async move {
            let result = pool_manager.run().await.map_err(|e| e.to_string());
            ("PoolManager", SubsystemCriticality::NonCritical, result)
        });

        let auto_manage = crate::model::auto_manage::AutoShardManager::new(
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        subsystems.spawn(async move {
            auto_manage.run().await;
            (
                "AutoShardManager",
                SubsystemCriticality::NonCritical,
                Ok(()),
            )
        });

        // Spawn UpdateChecker (11th subsystem task — optional, runs only if not disabled)
        {
            let update_config = self.config.updates.clone();
            let update_state = shared_state.update_state.clone();
            let update_tx = shared_state.update_tx.clone();
            let update_shutdown = shutdown_rx.clone();
            let checker = crate::update::UpdateChecker::new(
                update_config,
                "enapt/SwarmLLM".to_string(),
                update_state,
                update_tx,
            );
            subsystems.spawn(async move {
                checker.run(update_shutdown).await;
                ("UpdateChecker", SubsystemCriticality::NonCritical, Ok(()))
            });
        }

        // Spawn API server (pass router_cmd_tx + acquisition_tx + network_tx so API can submit requests)
        let api_shared_state = shared_state.clone();
        let api_router_tx = router_cmd_tx.clone();
        let api_acquisition_tx = acquisition_tx.clone();
        let api_network_tx = network_tx.clone();
        subsystems.spawn(async move {
            let result = crate::api::server::run_server_with_state(
                api_shared_state,
                api_router_tx,
                api_acquisition_tx,
                api_network_tx,
            )
            .await
            .map_err(|e| e.to_string());
            ("ApiServer", SubsystemCriticality::Critical, result)
        });

        // All subsystems spawned — mark node as ready for health probes
        shared_state
            .is_ready
            .store(true, std::sync::atomic::Ordering::Release);

        tracing::info!(
            node_id = %self.identity.node_id(),
            port = self.config.node.listen_port,
            "SwarmLLM daemon running"
        );

        // Auto-detect region via IP geolocation (non-blocking, best-effort)
        if shared_state.config.identity.region.is_none() {
            let geo_state = shared_state.clone();
            let mut geo_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
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
                    _ = geo_shutdown.changed() => {}
                }
            });
        } else {
            // User configured a region explicitly — use it
            *shared_state.detected_region.write().await =
                shared_state.config.identity.region.clone();
        }

        // Broadcast shard announcements and manifests shortly after startup
        // so peers discover our shards quickly (don't wait for the 30s health tick).
        {
            let announce_state = shared_state.clone();
            let announce_tx = network_tx.clone();
            let mut announce_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Wait for peer connections to establish, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                    _ = announce_shutdown.changed() => { return; }
                }

                let node_id = announce_state.identity.node_id().clone();

                // Broadcast shard announcements
                let mut hosted_shards = Vec::new();
                for entry in announce_state.model_registry.all_shard_entries() {
                    let (shard_id, holders) = entry;
                    if holders.contains(&node_id) {
                        hosted_shards.push(shard_id);
                    }
                }

                if !hosted_shards.is_empty() {
                    let announce = crate::types::ShardAnnounce {
                        node_id: node_id.clone(),
                        shards: hosted_shards,
                        timestamp: chrono::Utc::now(),
                    };
                    tracing::info!(
                        shards = announce.shards.len(),
                        "Broadcasting initial shard announcement"
                    );
                    let _ = announce_tx
                        .send(NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(
                            announce,
                        )))
                        .await;
                }

                // Broadcast manifests for models where we hold at least one shard
                // (not just models we originally published). This allows shard-holding
                // nodes to propagate manifests to pure-consumer peers.
                let hosted_models: std::collections::HashSet<String> = announce_state
                    .model_registry
                    .all_shard_entries()
                    .into_iter()
                    .filter_map(|(shard_id, holders)| {
                        if holders.contains(&node_id) {
                            Some(shard_id.model_id.0.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                for manifest in announce_state.model_registry.models() {
                    if manifest.publisher == node_id || hosted_models.contains(&manifest.id.0) {
                        let _ = announce_tx
                            .send(NetworkCommand::Broadcast(SwarmMessage::ModelManifest(
                                manifest,
                            )))
                            .await;
                    }
                }
            });
        }

        // Spawn key rotation task (evicts stale sessions + ephemeral re-keying)
        {
            let rotation_sm = shared_state.session_manager.clone();
            let rotation_shutdown = shutdown_rx.clone();
            let rotation_network_tx = network_tx.clone();
            let rotation_node_id = shared_state.identity.node_id().clone();
            let rotation_shared_state = shared_state.clone();
            tokio::spawn(async move {
                crate::crypto::key_rotation::run_key_rotation(
                    rotation_sm,
                    rotation_network_tx,
                    rotation_node_id,
                    rotation_shared_state,
                    rotation_shutdown,
                )
                .await;
            });
        }

        // Open browser on first start if configured
        if self.config.ui.open_browser_on_start {
            let url = format!("http://localhost:{}", self.config.node.listen_port);
            // Check if config file exists — if not, open setup wizard
            let config_path = self.config.node.data_dir.join("config.toml");
            let target = if config_path.exists() {
                format!("{url}/admin")
            } else {
                format!("{url}/setup")
            };
            let mut browser_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Small delay to let the server bind, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                        if let Err(e) = open_browser(&target) {
                            tracing::debug!(error = %e, "Could not open browser automatically");
                        }
                    }
                    _ = browser_shutdown.changed() => {}
                }
            });
        }

        // Auto-load models that have local shards available
        {
            let sm = shared_state.clone();
            let mut autoload_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Brief delay to let shard announcements propagate, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
                    _ = autoload_shutdown.changed() => { return; }
                }
                let mut manifests = sm.model_registry.models();
                // Sort by request count descending so popular models get VRAM priority on restart
                manifests.sort_by(|a, b| {
                    let count_a = sm
                        .model_request_counts
                        .get(&a.id)
                        .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                        .unwrap_or(0);
                    let count_b = sm
                        .model_request_counts
                        .get(&b.id)
                        .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                        .unwrap_or(0);
                    count_b.cmp(&count_a)
                });
                let vram_budget = crate::model::auto_manage::compute_vram_budget(&sm);
                for m in &manifests {
                    if sm.split_models.iter().any(|e| e.key().0 == m.id) {
                        continue;
                    }
                    crate::model::auto_manage::check_and_load_model(&sm, &m.id, vram_budget).await;
                }
            });
        }

        // ── SIGHUP Config Reload Handler (Unix only) ──
        #[cfg(unix)]
        {
            let sighup_state = shared_state.clone();
            let sighup_config = self.config.clone();
            let mut sighup_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                let mut sighup = match tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::hangup(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to register SIGHUP handler — config reload via signal disabled");
                        return;
                    }
                };
                loop {
                    tokio::select! {
                        _ = sighup_shutdown.changed() => {
                            if *sighup_shutdown.borrow() {
                                break;
                            }
                        }
                        _ = sighup.recv() => {
                            let config_path = sighup_config.node.data_dir.join("config.toml");
                            tracing::info!(
                                "SIGHUP received — reloading config from {}",
                                config_path.display()
                            );
                            match crate::config::reload_operational_params(&config_path) {
                                Ok(params) => {
                                    let old = crate::config::OperationalParams::from_config(
                                        &sighup_config,
                                    );
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
                                    sighup_state.apply_config_reload(params);
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
            });
        }

        // ── Supervisor Loop ──
        //
        // Monitors all subsystem tasks via JoinSet. When a task exits:
        // - Due to shutdown signal: expected, no action needed
        // - Non-critical subsystem: log error and continue running
        // - Critical subsystem: trigger graceful shutdown
        // - Panic: treated as unexpected exit with same criticality rules
        //
        // Track restart attempts per subsystem name
        let mut restart_counts: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();

        loop {
            tokio::select! {
                // Handle OS shutdown signals
                _ = async {
                    let ctrl_c = tokio::signal::ctrl_c();
                    #[cfg(unix)]
                    {
                        match tokio::signal::unix::signal(
                            tokio::signal::unix::SignalKind::terminate(),
                        ) {
                            Ok(mut sigterm) => {
                                tokio::select! {
                                    _ = ctrl_c => {
                                        tracing::info!("Shutdown signal received (Ctrl+C)");
                                    }
                                    _ = sigterm.recv() => {
                                        tracing::info!("Shutdown signal received (SIGTERM)");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to register SIGTERM handler — using Ctrl+C only");
                                ctrl_c.await.ok();
                                tracing::info!("Shutdown signal received (Ctrl+C)");
                            }
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        ctrl_c.await.ok();
                        tracing::info!("Shutdown signal received (Ctrl+C)");
                    }
                } => {
                    break;
                }
                // Handle API-triggered shutdown (watch channel)
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        tracing::info!("Shutdown requested via API — draining subsystems");
                        break;
                    }
                }
                // Handle subsystem task exits
                result = subsystems.join_next() => {
                    match result {
                        None => {
                            // All tasks finished — shouldn't happen during normal operation
                            tracing::error!("All subsystem tasks have exited");
                            break;
                        }
                        Some(Ok((name, criticality, task_result))) => {
                            // Check if this is a shutdown-induced exit (expected)
                            if *shutdown_rx.borrow() {
                                tracing::debug!(subsystem = name, "Subsystem exited during shutdown");
                                continue;
                            }

                            match task_result {
                                Ok(()) => {
                                    tracing::warn!(
                                        subsystem = name,
                                        "Subsystem exited unexpectedly with Ok"
                                    );
                                }
                                Err(ref e) => {
                                    tracing::error!(
                                        subsystem = name,
                                        error = %e,
                                        "Subsystem exited with error"
                                    );
                                }
                            }

                            let count = restart_counts.entry(name).or_insert(0);
                            *count += 1;

                            if criticality == SubsystemCriticality::Critical {
                                tracing::error!(
                                    subsystem = name,
                                    "Critical subsystem failed — triggering graceful shutdown"
                                );
                                break;
                            } else if *count >= MAX_RESTART_ATTEMPTS {
                                tracing::error!(
                                    subsystem = name,
                                    restart_count = *count,
                                    max_restarts = MAX_RESTART_ATTEMPTS,
                                    "Non-critical subsystem exceeded max restarts — triggering shutdown"
                                );
                                break;
                            } else {
                                // Non-critical: log and continue
                                tracing::warn!(
                                    subsystem = name,
                                    restart_count = *count,
                                    max_restarts = MAX_RESTART_ATTEMPTS,
                                    "Non-critical subsystem failed — daemon continues without it"
                                );
                            }
                        }
                        Some(Err(join_error)) => {
                            // Task panicked or was cancelled
                            if join_error.is_panic() {
                                tracing::error!(
                                    error = %join_error,
                                    "Subsystem task panicked — triggering shutdown"
                                );
                                break;
                            } else {
                                tracing::warn!(
                                    error = %join_error,
                                    "Subsystem task cancelled"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Signal graceful shutdown to all subsystems
        shared_state.shutdown();

        // Drain the JoinSet with a timeout so subsystems can run their cleanup
        // (e.g., save peer cache, close connections, flush data).
        tracing::info!("Waiting for subsystems to shut down (10s timeout)...");
        let drain_deadline = tokio::time::sleep(std::time::Duration::from_secs(10));
        tokio::pin!(drain_deadline);
        loop {
            tokio::select! {
                _ = &mut drain_deadline => {
                    tracing::warn!("Shutdown timeout — aborting remaining subsystems");
                    break;
                }
                result = subsystems.join_next() => {
                    match result {
                        Some(Ok((name, _, _))) => {
                            tracing::debug!(subsystem = name, "Subsystem exited cleanly");
                        }
                        Some(Err(e)) => {
                            tracing::debug!(error = %e, "Subsystem join error during shutdown");
                        }
                        None => {
                            tracing::info!("All subsystems shut down cleanly");
                            break;
                        }
                    }
                }
            }
        }

        // redb writes are durable on commit — no flush needed

        tracing::info!("Daemon shutdown complete");

        Ok(())
    }
}

fn resolve_api_key(config: &Config, db: &Database) -> String {
    let key;

    // 1. Explicit key in config takes priority
    if let Some(ref k) = config.api.api_key {
        if !k.is_empty() {
            tracing::info!("Using API key from configuration");
            key = k.clone();
            write_api_key_file(&config.node.data_dir, &key);
            return key;
        }
    }

    // 2. Check persisted key in database
    if let Ok(Some(k)) = db.get_json::<String>("config", "api_key") {
        if !k.is_empty() {
            tracing::info!("Using persisted API key from database");
            write_api_key_file(&config.node.data_dir, &k);
            return k;
        }
    }

    // 3. Generate a new 32-byte hex key
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    key = hex::encode(bytes);

    // Persist to DB
    if let Err(e) = db.put_json("config", "api_key", &key) {
        tracing::warn!(error = %e, "Failed to persist API key to database");
    }

    // Write to file so CLI `status` can read it without opening the database
    write_api_key_file(&config.node.data_dir, &key);

    // Print API key to stderr only (not to tracing logs which may be persisted/shipped)
    eprintln!("Generated new API key (save this for API access): {key}");

    key
}

/// Write the API key to a plain file so the CLI can read it while the daemon holds the DB lock.
fn write_api_key_file(data_dir: &std::path::Path, key: &str) {
    let path = data_dir.join("api_key");
    if let Err(e) = std::fs::write(&path, key) {
        tracing::warn!(error = %e, "Failed to write api_key file");
    }
    // Restrict permissions on Unix (owner read/write only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
}

fn map_gguf_architecture(path: &std::path::Path) -> crate::types::ModelArchitecture {
    let arch_str = match std::fs::File::open(path) {
        Ok(mut f) => match candle_core::quantized::gguf_file::Content::read(&mut f) {
            Ok(ct) => ct
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| "llama".to_string()),
            Err(_) => "llama".to_string(),
        },
        Err(_) => "llama".to_string(),
    };
    match arch_str.as_str() {
        "qwen2" | "qwen3" | "qwen2moe" => crate::types::ModelArchitecture::Qwen2,
        "qwen35" => crate::types::ModelArchitecture::Qwen35,
        "qwen35moe" | "qwen3_5moe" => crate::types::ModelArchitecture::Qwen35Moe {
            num_experts: 0,
            experts_per_token: 0,
        },
        "mistral" => crate::types::ModelArchitecture::Mistral,
        "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
        // All remaining supported transformer architectures map to Llama
        // (they share the same manifest structure).
        "llama" | "gemma" | "gemma2" | "starcoder2" | "deepseek2" | "glm4" | "llama4" => {
            crate::types::ModelArchitecture::Llama
        }
        other => {
            tracing::warn!(
                arch = other,
                "Unknown model architecture, defaulting to Llama"
            );
            crate::types::ModelArchitecture::Llama
        }
    }
}

/// Try to open a URL in the default browser.
fn open_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    {
        // On Windows, use `cmd /C start` for opening URLs
        return std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
            .map(|_| ());
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return Err("Unsupported platform".into());

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        std::process::Command::new(cmd)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

/// Best-effort IP geolocation using a free API (ip-api.com).
/// Returns an ISO 3166-1 alpha-2 country code (e.g. "US", "DE") or None on failure.
/// Timeout: 5 seconds. No API key required.
async fn detect_region_from_ip() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    // ip-api.com returns JSON with a "countryCode" field for free, no key needed.
    // Rate limit: 45 requests/min (we only call once at startup).
    let resp = client
        .get("http://ip-api.com/json/?fields=status,countryCode")
        .send()
        .await
        .ok()?;

    let json: serde_json::Value = resp.json().await.ok()?;
    if json.get("status")?.as_str()? == "success" {
        json.get("countryCode")?.as_str().map(|s| s.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn subsystem_criticality_variants_are_distinct() {
        assert_ne!(
            SubsystemCriticality::Critical,
            SubsystemCriticality::NonCritical
        );
    }

    #[test]
    fn max_restart_attempts_is_five() {
        assert_eq!(MAX_RESTART_ATTEMPTS, 5);
    }

    #[tokio::test]
    async fn joinset_catches_task_panic() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            panic!("simulated subsystem panic");
        });

        let result = set.join_next().await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_panic());
    }

    #[tokio::test]
    async fn joinset_returns_task_error() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            (
                "TestSubsystem",
                SubsystemCriticality::NonCritical,
                Err("boom".to_string()),
            )
        });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::NonCritical);
        assert!(task_result.is_err());
        assert_eq!(task_result.unwrap_err(), "boom");
    }

    #[tokio::test]
    async fn joinset_returns_task_success() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async { ("TestSubsystem", SubsystemCriticality::Critical, Ok(())) });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::Critical);
        assert!(task_result.is_ok());
    }

    #[tokio::test]
    async fn supervisor_non_critical_failure_does_not_drain_set() {
        // Simulate: one non-critical task fails, others keep running
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();

        // Task that fails immediately
        set.spawn(async {
            (
                "HealthMonitor",
                SubsystemCriticality::NonCritical,
                Err("test error".to_string()),
            )
        });

        // Task that runs until cancelled
        set.spawn(async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            ("ApiServer", SubsystemCriticality::Critical, Ok(()))
        });

        // First join: get the failed task
        let result = set.join_next().await.unwrap();
        let (name, crit, _) = result.unwrap();
        assert_eq!(name, "HealthMonitor");
        assert_eq!(crit, SubsystemCriticality::NonCritical);

        // The other task is still running — set is not empty
        assert_eq!(set.len(), 1);

        // Clean up
        set.abort_all();
    }

    #[tokio::test]
    async fn supervisor_restart_counting() {
        // Simulate the restart counting logic from the supervisor loop
        let mut restart_counts: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();

        // Simulate 5 failures of a non-critical subsystem
        for i in 1..=5 {
            let count = restart_counts.entry("HealthMonitor").or_insert(0);
            *count += 1;
            assert_eq!(*count, i);
        }

        // After 5 failures, count should be 5 (at the limit)
        assert_eq!(
            *restart_counts.get("HealthMonitor").unwrap(),
            MAX_RESTART_ATTEMPTS
        );

        // One more would exceed
        let count = restart_counts.entry("HealthMonitor").or_insert(0);
        *count += 1;
        assert!(*count > MAX_RESTART_ATTEMPTS);
    }
}
