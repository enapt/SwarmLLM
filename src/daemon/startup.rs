//! Daemon startup helpers: rebuild in-memory state from persistent storage.
//!
//! Runs once before subsystems are spawned. Pulls manifests from the DB,
//! scans local shard directories, registers shard holders for this node,
//! restores HF sources, and (if needed) re-downloads missing GGUF headers
//! from HuggingFace. This is the most side-effect-heavy startup phase.

use std::sync::Arc;

use crate::config::Config;
use crate::model::manifest::ModelManifestExt;
use crate::storage::db::Database;
use crate::types::ShardId;

use super::manifest::{extract_tied_output_weight, regenerate_manifest_from_header};
use super::state::{HfSource, SharedState};

/// Rehydrate SharedState from on-disk artifacts: DB-persisted manifests,
/// local shard files, mmproj sentinel shards, HF source sidecars, and any
/// needed HF header downloads.
pub(super) async fn restore_persistent_state(
    shared_state: &Arc<SharedState>,
    config: &Config,
    db: &Database,
) {
    // Restore persisted manifests from the DB and register shard holders.
    // This handles the case where a node restarts with --shards but no --model:
    // the manifest was generated in a previous run and persisted, so we restore
    // it and re-register ourselves as holder of our shard range.
    {
        let node_id = shared_state.identity.node_id().clone();
        if let Ok(manifests) = db.iter_json::<crate::types::ModelManifest>("model_meta") {
            for manifest in manifests {
                let model_id = manifest.id.clone();
                if manifest.verify_hash().is_err() {
                    tracing::warn!(
                        model = %model_id,
                        "Manifest from DB failed hash verification — skipping"
                    );
                    continue;
                }
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
                let shard_store_reg = shared_state.shard_store();
                for shard_info in &manifest.shards {
                    let in_range = config.inference.claims_shard(shard_info.index);
                    if in_range {
                        let shard_path = shard_store_reg.shard_path(&model_id, shard_info.index);
                        if !shard_path.exists() {
                            // Not a problem, and not worth a warning: this loop
                            // walks every shard of every manifest the node KNOWS
                            // about, and a node is only ever expected to hold
                            // some of them. Not holding a piece another machine
                            // holds is the normal state of a node in a swarm.
                            //
                            // It was logged at WARN, which made it the loudest
                            // line at every startup — a dozen alarming "missing
                            // on disk" warnings about models the user had
                            // deliberately removed, or never had. Noise at WARN
                            // is worse than silence: it is what a real problem
                            // has to be spotted among.
                            //
                            // A shard that IS held but has gone bad is caught
                            // just below by the size check, which quarantines it
                            // so the ordinary acquisition path repairs it.
                            tracing::debug!(
                                model = %model_id,
                                shard = shard_info.index,
                                path = %shard_path.display(),
                                "Shard not held by this node — skipping registration"
                            );
                            continue;
                        }
                        // Existence alone is not enough, and this was the path
                        // that mattered: it is what records us as a HOLDER.
                        // Registering on existence while every loader rejects
                        // the file on size left the node believing it held a
                        // shard it could not serve — so nothing ever
                        // re-downloaded it, and peers could still be routed to
                        // us for it. An external report tracked exactly that for
                        // 16 releases. Quarantine frees the name so the ordinary
                        // acquisition path repairs it.
                        if crate::model::shard::quarantine_shard_if_size_mismatch(
                            &shard_path,
                            shard_info.size_bytes,
                        )
                        .is_some()
                        {
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
                    let shard_store_tmp = shared_state.shard_store();
                    let model_dir = shard_store_tmp.model_dir(&model_id);
                    let source_path_file = model_dir.join("source_path");
                    if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
                        let path = std::path::PathBuf::from(path_str.trim());
                        // SEC: Containment check — source_path must be within data directory.
                        // If canonicalize fails (path doesn't exist), skip entirely — never
                        // fall back to the raw path which could contain ".." traversal.
                        let canonical = match path.canonicalize() {
                            Ok(c) => c,
                            Err(e) => {
                                tracing::warn!(
                                    model = %model_id,
                                    path = %path.display(),
                                    error = %e,
                                    "source_path canonicalize failed — skipping"
                                );
                                continue;
                            }
                        };
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
    let shard_store = shared_state.shard_store();
    {
        let models_dir = shard_store.models_dir();
        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    let model_dir = entry.path();
                    if !model_dir.is_dir() {
                        continue;
                    }
                    let manifest_path = model_dir.join(crate::model::shard::MANIFEST_FILENAME);
                    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
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
                            crate::inference::split::GgufTensorMeta::from_gguf_file(&header_path)
                        {
                            tracing::info!(
                                model = %model_id,
                                "Regenerating missing manifest from GGUF header"
                            );
                            if regenerate_manifest_from_header(&model_id, &model_dir, &meta, config)
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
            let mut registered_manifests = std::collections::HashSet::new();
            let mut model_shard_counts: std::collections::HashMap<crate::types::ModelId, u32> =
                std::collections::HashMap::new();

            for (model_id, shard_info) in &shards {
                // `inference.shard_range` says which shard indices this node
                // claims. It was applied ONLY to the manifests restored from the
                // database a few hundred lines above, and not here — where a
                // node registers what it actually found on disk.
                //
                // So on any machine that has the files, the setting did nothing:
                // it registered every shard, announced every shard, and kept
                // serving the whole model. No error, no warning, and the config
                // key parsed fine, so the only way to notice was to look at the
                // registry and find four shards claimed after asking for two —
                // which is how this was found (2026-08-09).
                //
                // That matters most for the thing the setting exists to do:
                // deliberately splitting one model across two machines. Anyone
                // setting it up had it silently not happen.
                if !config.inference.claims_shard(shard_info.index) {
                    tracing::debug!(
                        model = %model_id,
                        shard = shard_info.index,
                        "Not claiming shard — outside inference.shard_range"
                    );
                    continue;
                }
                if registered_manifests.insert(model_id.clone()) {
                    let model_dir = shard_store.model_dir(model_id);

                    // Materialise the header (extracting it from shard_000 if
                    // that is all we have), then let `gguf_meta_for` learn the
                    // geometry from it — the one place that does, so this and
                    // the runtime shard-landing path cannot disagree.
                    if let Ok(()) = crate::inference::split::ensure_gguf_header(&model_dir) {
                        let _ = shared_state.gguf_meta_for(model_id);
                    }

                    let manifest_loaded = if let Ok(manifest) =
                        crate::types::ModelManifest::load_from_dir(&model_dir)
                    {
                        // A manifest states its own model id, but shard scanning
                        // and manifest regeneration both key off the directory
                        // name. Nothing reconciled the two, so copying a model
                        // directory — `<model>.FULLBACKUP`, `<model>.old` —
                        // produced a model the swarm was told about under a name
                        // that resolves to nothing: peers record shard holders
                        // for it, replica counts double, and no one can ever
                        // acquire it because the identity is local invention
                        // rather than anything upstream.
                        //
                        // A model's identity has to come from the model, not from
                        // whatever a directory happens to be called. Mismatches
                        // are skipped rather than renamed: the copy is almost
                        // always a deliberate local backup, and silently
                        // adopting it under the real id would let a stale copy
                        // race the live one.
                        if manifest.id.0 != model_id.0 {
                            tracing::warn!(
                                dir = %model_id,
                                manifest_id = %manifest.id,
                                "Skipping model: directory name does not match the \
                                 manifest's own id. Rename the directory to match \
                                 if this is a real model; a copy kept as a backup \
                                 is being ignored on purpose."
                            );
                            false
                        } else if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
                            // The dir name and the manifest id agree — but both
                            // are a backup-copy name (an older build regenerated
                            // the manifest from the copied folder). Skip so it is
                            // neither registered nor persisted and re-gossiped.
                            tracing::warn!(
                                dir = %model_id,
                                "Skipping model: name looks like a local backup copy \
                                 (`.FULLBACKUP`, `.old`, …), not a real model. Rename \
                                 to the real model id if this is genuine."
                            );
                            false
                        } else if manifest.verify_hash().is_ok() {
                            // Deliberately does NOT rewrite `publisher` to this
                            // node. That claim existed only to earn broadcast
                            // rights, and broadcasting is now driven by what we
                            // HOLD (`ModelRegistry::manifests_to_gossip`), so the
                            // rewrite bought nothing and cost a great deal: every
                            // holder claimed the same model, `register_manifest`
                            // overwrites unconditionally, so holders erased each
                            // other's claim until none of them broadcast at all.
                            // It also changed `manifest_hash` on every restart,
                            // making an unchanged model look changed to every
                            // peer — 81 registrations under 50 publishers for one
                            // model on a 5-node swarm. `publisher` now means what
                            // it says: who published it.
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
                        if let Some(meta) = shared_state.gguf_meta_for(model_id) {
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
                    let tied_path = model_dir.join(crate::inference::split::TIED_OUTPUT_FILENAME);
                    if !tied_path.exists() {
                        if let Some(meta) = shared_state.gguf_meta_for(model_id) {
                            let has_output = meta.tensors.contains_key("output.weight");
                            let has_embd = meta.tensors.contains_key("token_embd.weight");
                            if !has_output && has_embd {
                                let shard0_path = model_dir.join("shard_000.bin");
                                if shard0_path.exists() {
                                    if let Err(e) =
                                        extract_tied_output_weight(&shard0_path, &model_dir, &meta)
                                    {
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
                *model_shard_counts.entry(model_id.clone()).or_insert(0) += 1;
            }

            // Emit startup activity events so the dashboard shows them
            for (mid, count) in &model_shard_counts {
                let name = shared_state
                    .model_registry
                    .get_manifest(mid)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| mid.0.clone());
                shared_state.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "model",
                        "shards_loaded",
                        format!("Loaded {} shards for {}", count, name),
                    )
                    .with_model(mid.0.clone()),
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to scan local shards");
        }
    }

    // Register local mmproj files as sentinel shards.
    {
        let models_dir = shard_store.models_dir();
        if models_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                let node_id = shared_state.identity.node_id().clone();
                for entry in entries.flatten() {
                    if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        continue;
                    }
                    let mmproj_path = entry.path().join(crate::model::shard::MMPROJ_FILENAME);
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
        let models_dir = shard_store.models_dir();
        if models_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        continue;
                    }
                    let model_id_str = entry.file_name().to_string_lossy().to_string();
                    let mid = crate::types::ModelId(model_id_str.clone());
                    if shared_state.models.hf_sources.contains_key(&mid) {
                        continue;
                    }
                    let hf_path = entry.path().join(crate::model::shard::HF_SOURCE_FILENAME);
                    if hf_path.exists() {
                        if let Ok(data) = std::fs::read_to_string(&hf_path) {
                            if let Ok(source) = serde_json::from_str::<HfSource>(&data) {
                                tracing::info!(
                                    model = %model_id_str,
                                    repo = %source.repo_id,
                                    file = %source.filename,
                                    "Loaded HF source from disk"
                                );
                                shared_state
                                    .models
                                    .hf_sources
                                    .insert(mid.clone(), source.clone());
                                let _ = db.put_json("hf_sources", &model_id_str, &source);
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
        let models_dir = shard_store.models_dir();
        if models_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        continue;
                    }
                    let model_dir = entry.path();
                    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
                    if header_path.exists() {
                        continue;
                    }
                    let has_shards = model_dir.join("shard_000.bin").exists()
                        || model_dir.join("shard_001.bin").exists();
                    if !has_shards {
                        continue;
                    }
                    if model_dir.join("shard_000.bin").exists()
                        && crate::inference::split::ensure_gguf_header(&model_dir).is_ok()
                    {
                        continue;
                    }
                    let model_id_str = entry.file_name().to_string_lossy().to_string();
                    let mid = crate::types::ModelId(model_id_str.clone());
                    if let Some(hf_src) = shared_state.models.hf_sources.get(&mid) {
                        tracing::info!(
                            model = %model_id_str,
                            repo = %hf_src.repo_id,
                            "Downloading GGUF header from HuggingFace (no local shard_000)"
                        );
                        let shard_size = config.model.shard_size_bytes();
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
                                    if let Ok(meta) =
                                        crate::inference::split::GgufTensorMeta::from_gguf_file(&hp)
                                    {
                                        shared_state.gguf_meta.insert(mid.clone(), meta.clone());
                                        let manifest_path =
                                            model_dir.join(crate::model::shard::MANIFEST_FILENAME);
                                        if !manifest_path.exists() {
                                            regenerate_manifest_from_header(
                                                &mid, &model_dir, &meta, config,
                                            );
                                        }
                                        let tied_path = model_dir
                                            .join(crate::inference::split::TIED_OUTPUT_FILENAME);
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
}
