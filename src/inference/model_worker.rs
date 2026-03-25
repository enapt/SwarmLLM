//! Model worker subprocess for SwarmLLM.
//!
//! Each model runs in its own process. When killed, the OS/CUDA driver
//! reclaims ALL GPU memory immediately — solving the "memory doesn't drop
//! on unload" problem and keeping inference off the main daemon's Tokio runtime.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::UnixStream;

use crate::daemon::shard_loader::{try_load_from_shards, ShardLoadParams};
use crate::error::SwarmError;
use crate::inference::split::{self, KvCacheStore, SplitModel};
use crate::inference::worker_ipc::*;
use crate::types::NetworkFinishReason;

/// Run the model worker subprocess.
/// Called from main.rs when the binary is invoked with `model-worker` subcommand.
/// `shard_window`: if Some, only load these shard indices (VRAM-saving mode).
pub async fn run_worker(
    socket_path: PathBuf,
    data_dir: PathBuf,
    shard_window: Option<Vec<u32>>,
    kv_cache_ttl_secs: u64,
) {
    // Connect to the daemon's Unix socket
    let stream = match UnixStream::connect(&socket_path).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "model-worker: failed to connect to {}: {e}",
                socket_path.display()
            );
            std::process::exit(1);
        }
    };
    let (mut reader, mut writer) = stream.into_split();

    // Send Ready
    if let Err(e) = send_worker(&mut writer, &WorkerMsg::Ready, &[]).await {
        eprintln!("model-worker: failed to send Ready: {e}");
        std::process::exit(1);
    }

    // Per-model state: (layer_start, layer_end) → SplitModel
    let mut models: HashMap<(usize, usize), SplitModel> = HashMap::new();
    let kv_store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(
        kv_cache_ttl_secs,
    )));

    if let Some(ref w) = shard_window {
        tracing::info!(window = ?w, "model-worker: shard window active — only loading specified shards");
    }

    loop {
        let (msg, payload) = match recv_daemon(&mut reader).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("model-worker: socket read error: {e}");
                break;
            }
        };

        match msg {
            DaemonMsg::Forward(fwd) => {
                let request_id = fwd.request_id;
                match handle_forward(
                    &mut writer,
                    &mut models,
                    &kv_store,
                    &data_dir,
                    fwd,
                    payload,
                    &shard_window,
                )
                .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = send_worker(
                            &mut writer,
                            &WorkerMsg::Error {
                                request_id,
                                message: e.to_string(),
                            },
                            &[],
                        )
                        .await;
                    }
                }
            }
            DaemonMsg::Generate(gen) => {
                let request_id = gen.request_id;
                match handle_generate(
                    &mut writer,
                    &mut models,
                    &kv_store,
                    &data_dir,
                    gen,
                    &shard_window,
                )
                .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = send_worker(
                            &mut writer,
                            &WorkerMsg::Error {
                                request_id,
                                message: e.to_string(),
                            },
                            &[],
                        )
                        .await;
                    }
                }
            }
            DaemonMsg::Unload {
                layer_start,
                layer_end,
            } => {
                models.remove(&(layer_start, layer_end));
                tracing::info!("model-worker: unloaded [{layer_start}..{layer_end})");
            }
            DaemonMsg::Shutdown => {
                let _ = send_worker(&mut writer, &WorkerMsg::Bye, &[]).await;
                break;
            }
        }
    }

    // Explicitly drop all models before exiting — CUDA contexts will be freed
    drop(models);
    tracing::info!("model-worker: exiting cleanly");
}

/// Ensure a SplitModel is loaded for the given model_id and layer range.
/// `shard_window`: if Some, only load shards in this set (VRAM-saving mode).
fn ensure_model_loaded(
    models: &mut HashMap<(usize, usize), SplitModel>,
    data_dir: &std::path::Path,
    model_id: &crate::types::ModelId,
    layer_start: usize,
    layer_end: usize,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    let key = (layer_start, layer_end);
    if models.contains_key(&key) {
        return Ok(());
    }

    let model_dir = data_dir.join("models").join(&model_id.0);
    let manifest_path = model_dir.join("manifest.json");
    let manifest: crate::types::ModelManifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| {
            SwarmError::Internal(format!("Read manifest {}: {e}", manifest_path.display()))
        })?)
        .map_err(|e| SwarmError::Internal(format!("Parse manifest: {e}")))?;

    let total_layers = manifest.num_layers as usize;
    // Determine which shards we have on disk
    let shard_store = crate::model::shard::ShardStore::new(data_dir);
    let mut local_shard_indices: Vec<u32> = Vec::new();
    let scan_limit = manifest.shard_count.max(1);
    for i in 0u32..scan_limit {
        let path = shard_store.shard_path(model_id, i);
        if path.exists() {
            local_shard_indices.push(i);
        }
    }

    // Filter by shard window if active — only load allowed shards into VRAM
    if let Some(window) = shard_window {
        let before = local_shard_indices.len();
        local_shard_indices.retain(|i| window.contains(i));
        if local_shard_indices.len() < before {
            tracing::info!(
                model = %model_id,
                before_count = before,
                after_count = local_shard_indices.len(),
                window = ?window,
                "Shard window active — loading subset of on-disk shards"
            );
        }
    }

    let has_shard_0 = local_shard_indices.contains(&0);
    let last_shard_idx = manifest.shard_count.saturating_sub(1);
    let has_last_shard = local_shard_indices.contains(&last_shard_idx);
    let is_first = layer_start == 0 && has_shard_0;
    let is_last = layer_end >= total_layers && has_last_shard;

    // Try loading the split model from available sources
    let gguf_path = model_dir.join("model.gguf");
    let source_path_file = model_dir.join("source_path");

    let model = if gguf_path.exists() {
        tracing::info!(
            model = %model_id,
            layers = format!("[{layer_start}..{layer_end})"),
            "model-worker: Loading from reconstructed GGUF"
        );
        SplitModel::load_from_gguf(&gguf_path, layer_start, layer_end, is_first, is_last)?
    } else if source_path_file.exists() {
        match std::fs::read_to_string(&source_path_file) {
            Ok(p) => {
                let p = std::path::PathBuf::from(p.trim());
                let data_models = shard_store.models_dir();
                let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
                if !canonical.starts_with(&data_models) {
                    return Err(SwarmError::Internal(
                        "source_path outside data directory".into(),
                    ));
                }
                if canonical.exists() {
                    SplitModel::load_from_gguf(
                        &canonical,
                        layer_start,
                        layer_end,
                        is_first,
                        is_last,
                    )?
                } else {
                    try_load_from_shards(&ShardLoadParams {
                        model_dir: &model_dir,
                        shard_store: &shard_store,
                        model_id,
                        layer_start,
                        layer_end,
                        is_first,
                        is_last,
                        manifest: &manifest,
                    })?
                }
            }
            Err(e) => return Err(SwarmError::Io(e)),
        }
    } else {
        try_load_from_shards(&ShardLoadParams {
            model_dir: &model_dir,
            shard_store: &shard_store,
            model_id,
            layer_start,
            layer_end,
            is_first,
            is_last,
            manifest: &manifest,
        })?
    };

    // Initialize paged KV cache pool when the feature is enabled and model is on GPU
    #[cfg(feature = "paged-attn")]
    if model.device().is_cuda() {
        let n_kv_heads = model.n_kv_head();
        let head_dim = model.head_dim();
        // Allocate 20% of VRAM (256 MB default) for the paged KV pool
        let budget_mb = 256u64;
        let num_blocks = crate::inference::paged_kv::PagedKvPool::auto_size(
            budget_mb,
            n_kv_heads,
            head_dim,
            candle_core::DType::F16,
        );
        if num_blocks > 0 {
            match crate::inference::paged_kv::PagedKvPool::new(
                num_blocks,
                n_kv_heads,
                head_dim,
                candle_core::DType::F16,
                model.device(),
            ) {
                Ok(pool) => {
                    tracing::info!(
                        num_blocks,
                        n_kv_heads,
                        head_dim,
                        budget_mb,
                        "model-worker: Initialized PagedKvPool"
                    );
                    // Store the pool — it will be used when the forward path supports paged attention
                    let _ = pool; // TODO: wire into forward path when paged attention kernels are integrated
                }
                Err(e) => {
                    tracing::warn!(error = %e, "model-worker: Failed to create PagedKvPool, using standard KV cache");
                }
            }
        }
    }

    tracing::info!(
        model = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        device = ?model.device(),
        "model-worker: Model loaded"
    );
    models.insert(key, model);
    Ok(())
}

/// Handle a Forward IPC message — run a single-step forward pass.
async fn handle_forward(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    models: &mut HashMap<(usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    fwd: IpcForward,
    activation_bytes: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    let request_id = fwd.request_id;
    let model_id = fwd.model_id.clone();
    let (layer_start, layer_end) = (fwd.layer_range.0 as usize, fwd.layer_range.1 as usize);

    // Ensure model is loaded
    ensure_model_loaded(
        models,
        data_dir,
        &model_id,
        layer_start,
        layer_end,
        shard_window,
    )?;

    let model = models
        .get_mut(&(layer_start, layer_end))
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;

    let is_first = model.is_first();
    let is_last = model.is_last();
    let total_layers = model.total_layers;
    let req_id_str = request_id.to_string();
    let pre_embedded = fwd.pre_embedded;

    // Clear per-request KV-cache at the start of a new request (prefill)
    if fwd.sequence_num == 0 {
        let model_key = format!("{}-{}-{}", layer_start, layer_end, total_layers);
        kv_store.clear_request(&model_key, &req_id_str);
    }

    // Convert activation bytes to a candle Tensor
    let input_tensor = if pre_embedded {
        split::bytes_to_tensor(&activation_bytes)?
    } else if is_first {
        if fwd.index_pos == 0 {
            // Prefill: activations are the prompt text → tokenize
            let prompt = String::from_utf8_lossy(&activation_bytes);
            let token_ids: Vec<i64> = if let Some(tokenizer) = model.tokenizer() {
                tokenizer.encode(&prompt)
            } else {
                prompt.bytes().map(|b| b as i64).collect()
            };
            candle_core::Tensor::from_vec(
                token_ids.clone(),
                &[1, token_ids.len()],
                &candle_core::Device::Cpu,
            )
            .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?
        } else {
            // Decode step: single i64 token ID (8 bytes LE)
            let token_id = if activation_bytes.len() >= 8 {
                let bytes: [u8; 8] = activation_bytes[..8]
                    .try_into()
                    .map_err(|_| SwarmError::Internal("Invalid activation data".into()))?;
                i64::from_le_bytes(bytes)
            } else {
                return Err(SwarmError::Internal(format!(
                    "Decode step activation payload too short: {} bytes (need 8)",
                    activation_bytes.len()
                )));
            };
            candle_core::Tensor::from_vec(vec![token_id], &[1, 1], &candle_core::Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?
        }
    } else {
        split::bytes_to_tensor(&activation_bytes)?
    };

    // Decompress vision embeddings if present
    let vision_tensor: Option<candle_core::Tensor> =
        if let Some(ref compressed) = fwd.vision_embeddings {
            match zstd::decode_all(std::io::Cursor::new(compressed)) {
                Ok(raw_bytes) => {
                    const MAX_VISION_EMBEDDING_BYTES: usize = 50 * 1024 * 1024;
                    if raw_bytes.len() > MAX_VISION_EMBEDDING_BYTES {
                        None
                    } else {
                        let num_f16 = raw_bytes.len() / 2;
                        const COMMON_HIDDEN_DIMS: &[usize] =
                            &[5120, 4096, 3584, 3072, 2560, 2048, 1536, 1024];
                        let hidden_dim = COMMON_HIDDEN_DIMS
                            .iter()
                            .copied()
                            .find(|&d| num_f16 % d == 0 && (1..2048).contains(&(num_f16 / d)))
                            .unwrap_or(1024);
                        let num_tokens = num_f16 / hidden_dim;
                        let f32_values: Vec<f32> = raw_bytes
                            .chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect();
                        candle_core::Tensor::from_vec(
                            f32_values,
                            &[num_tokens, hidden_dim],
                            &candle_core::Device::Cpu,
                        )
                        .ok()
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to decompress vision embeddings");
                    None
                }
            }
        } else {
            None
        };

    // Load LoRA adapter if requested
    let lora_adapter = if let Some(ref adapter_id) = fwd.adapter_id {
        let adapter_dir = data_dir.join("adapters").join(adapter_id);
        if adapter_dir.exists() {
            match crate::model::lora::load_adapter_from_dir(&adapter_dir) {
                Ok(adapter) => {
                    tracing::debug!(adapter_id, "Loaded LoRA adapter for inference");
                    Some(adapter)
                }
                Err(e) => {
                    tracing::warn!(adapter_id, error = %e, "Failed to load LoRA adapter, proceeding without");
                    None
                }
            }
        } else {
            tracing::warn!(adapter_id, "LoRA adapter directory not found");
            None
        }
    } else {
        None
    };

    // Run forward pass — CPU-bound, use block_in_place
    let compute_result =
        tokio::task::block_in_place(|| -> Result<crate::types::LayerResult, String> {
            let output = if pre_embedded {
                model
                    .forward_pre_embedded(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                    )
                    .map_err(|e| format!("Forward pre-embedded: {e}"))?
            } else if let Some(ref vis_emb) = vision_tensor {
                model
                    .forward_multimodal(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                        Some(vis_emb),
                    )
                    .map_err(|e| format!("Forward multimodal: {e}"))?
            } else {
                model
                    .forward_with_lora(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                        lora_adapter.as_ref(),
                    )
                    .map_err(|e| format!("Forward: {e}"))?
            };

            if is_last {
                let token_id =
                    split::sample_token(&output, fwd.sampling.temperature, fwd.sampling.top_p)
                        .map_err(|e| format!("Sample: {e}"))?;
                let eos_tokens = model.eos_tokens();
                let finish = if eos_tokens.contains(&token_id) {
                    Some(NetworkFinishReason::Stop)
                } else {
                    None
                };
                Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![token_id],
                    finish_reason: finish,
                    activations: vec![],
                    sealed_token_ids: None,
                })
            } else {
                let activation_bytes =
                    split::tensor_to_bytes(&output).map_err(|e| format!("Encode: {e}"))?;
                Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: activation_bytes,
                    sealed_token_ids: None,
                })
            }
        });

    let result = compute_result.map_err(SwarmError::Internal)?;

    // Build IPC response
    let has_activations = !result.activations.is_empty();
    let activation_payload = if has_activations {
        result.activations.clone()
    } else {
        vec![]
    };

    let ipc_result = IpcLayerResult {
        request_id: result.request_id,
        token_ids: result.token_ids,
        finish_reason: result.finish_reason,
        format: None,
        sealed: result.sealed_token_ids.is_some(),
        sealed_payload: result.sealed_token_ids,
        logprobs: None,
        has_activations,
    };

    send_worker(
        writer,
        &WorkerMsg::LayerResult(ipc_result),
        &activation_payload,
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send LayerResult: {e}")))?;

    Ok(())
}

/// Handle a Generate IPC message — run a full tokenize+decode loop.
async fn handle_generate(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    models: &mut HashMap<(usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    gen: IpcGenerate,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    let request_id = gen.request_id;
    let model_id = gen.model_id.clone();
    let (layer_start, layer_end) = (gen.layer_range.0 as usize, gen.layer_range.1 as usize);

    // Ensure model is loaded
    ensure_model_loaded(
        models,
        data_dir,
        &model_id,
        layer_start,
        layer_end,
        shard_window,
    )?;

    let model = models
        .get_mut(&(layer_start, layer_end))
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;

    let req_id_str = request_id.to_string();

    // Tokenize the prompt
    let (input, prompt_tokens) = model.tokenize(&gen.prompt)?;

    // Prefill — block_in_place for CPU-bound inference
    let logits = tokio::task::block_in_place(|| model.forward(&input, 0, kv_store, &req_id_str))?;

    let use_logprobs = gen.sampling.logprobs;
    let (mut next_token, mut token_logprob) =
        crate::inference::tensor_util::sample_token_with_logprob(&logits, &gen.sampling)?;

    let eos = model.eos_tokens().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut index_pos = prompt_tokens;
    let mut finish_reason = "length".to_string();

    for _ in 0..gen.sampling.max_tokens {
        if eos.contains(&next_token) {
            finish_reason = "stop".to_string();
            break;
        }

        generated.push(next_token);

        let text = decode_token(model, next_token);

        send_worker(
            writer,
            &WorkerMsg::Token {
                request_id,
                token_id: next_token,
                text,
                is_eos: false,
                logprob: if use_logprobs { token_logprob } else { None },
            },
            &[],
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("send Token: {e}")))?;

        let input = model.token_tensor(next_token)?;
        let logits = tokio::task::block_in_place(|| {
            model.forward(&input, index_pos, kv_store, &req_id_str)
        })?;
        let (tok, lp) =
            crate::inference::tensor_util::sample_token_with_logprob(&logits, &gen.sampling)?;
        next_token = tok;
        token_logprob = lp;
        index_pos += 1;
    }

    // Send GenerateDone
    send_worker(
        writer,
        &WorkerMsg::GenerateDone {
            request_id,
            prompt_tokens,
            completion_tokens: generated.len(),
            finish_reason,
        },
        &[],
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send GenerateDone: {e}")))?;

    Ok(())
}

/// Decode a single token to text using the model's vocabulary.
fn decode_token(model: &SplitModel, token_id: u32) -> String {
    if let Some(vocab) = model.vocab() {
        if let Some(token_str) = vocab.get(token_id as usize) {
            if let Some(tokenizer) = model.tokenizer() {
                let bytes = tokenizer.decode_token(token_str);
                return String::from_utf8_lossy(&bytes).to_string();
            }
            return token_str.clone();
        }
    }
    String::new()
}
