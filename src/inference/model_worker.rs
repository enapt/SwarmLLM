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
                tracing::warn!(error = %e, "model-worker: socket read error");
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
                tracing::info!(layer_start, layer_end, "model-worker: unloaded shard range");
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

    let model_dir = crate::model::shard::model_dir(data_dir, &model_id.0);
    use crate::model::manifest::ModelManifestExt;
    let manifest = crate::types::ModelManifest::load_from_dir(&model_dir)?;

    let total_layers = manifest.num_layers as usize;
    let shard_store = crate::model::shard::ShardStore::new(data_dir);
    let mut local_shard_indices: Vec<u32> = shard_store
        .scan_local_shards(model_id, manifest.shard_count)
        .iter()
        .map(|(i, _)| *i)
        .collect();

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

    let (is_first, is_last) = crate::model::shard::compute_first_last(
        &local_shard_indices,
        manifest.shard_count,
        layer_start,
        layer_end,
        total_layers,
    );

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

    // Pre-split weights for tensor parallelism on first TP forward
    if let Some(ref tp) = fwd.tp_meta {
        let model = models
            .get_mut(&(layer_start, layer_end))
            .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;
        // Only split once — check if n_head already reflects tp splitting
        let current_heads = model.n_kv_head();
        let expected_heads = current_heads / (tp.tp_size as usize);
        if expected_heads > 0 && current_heads > expected_heads {
            model.pre_split_for_tp(tp.tp_rank as usize, tp.tp_size as usize)?;
        }
    }

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

    // Decompress vision embeddings if present.
    // Wire format: 8-byte header (num_tokens u32 LE + hidden_dim u32 LE) + zstd(FP16 data)
    let vision_tensor: Option<candle_core::Tensor> = if let Some(ref compressed) =
        fwd.vision_embeddings
    {
        if compressed.len() < 8 {
            tracing::warn!(request_id = %fwd.request_id, bytes = compressed.len(), "Vision embedding too short — dropping vision tensor");
            None
        } else {
            // Read shape header
            let num_tokens =
                u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]])
                    as usize;
            let hidden_dim =
                u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]])
                    as usize;
            let zstd_data = &compressed[8..];
            match zstd::decode_all(std::io::Cursor::new(zstd_data)) {
                Ok(raw_bytes) => {
                    const MAX_VISION_EMBEDDING_BYTES: usize = 50 * 1024 * 1024;
                    if raw_bytes.len() > MAX_VISION_EMBEDDING_BYTES || raw_bytes.len() % 2 != 0 {
                        None
                    } else {
                        let f32_values: Vec<f32> = raw_bytes
                            .chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect();
                        if f32_values.len() != num_tokens * hidden_dim {
                            tracing::warn!(
                                expected = num_tokens * hidden_dim,
                                actual = f32_values.len(),
                                "Vision embedding shape mismatch"
                            );
                            None
                        } else {
                            candle_core::Tensor::from_vec(
                                f32_values,
                                &[num_tokens, hidden_dim],
                                &candle_core::Device::Cpu,
                            )
                            .ok()
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to decompress vision embeddings");
                    None
                }
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
                let token_id = split::sample_token_with_params(&output, &fwd.sampling)
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

    // SYNC: token loop logic must match executor.rs generate_stream_inner.
    // Changes to EOS/stop handling must be applied to both.
    let eos = model.eos_tokens().to_vec();
    let stop_sequences = &gen.sampling.stop;
    let mut generated: Vec<u32> = Vec::new();
    let mut accumulated_text = String::new();
    let mut index_pos = prompt_tokens;
    let mut finish_reason = "length".to_string();

    for _ in 0..gen.sampling.max_tokens {
        if eos.contains(&next_token) {
            finish_reason = "stop".to_string();
            break;
        }

        let text = decode_token(model, next_token);
        accumulated_text.push_str(&text);

        // Check user-provided stop sequences
        if crate::inference::sampling::find_stop_sequence(&accumulated_text, stop_sequences)
            .is_some()
        {
            finish_reason = "stop".to_string();
            break;
        }

        generated.push(next_token);

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

    // If the loop exhausted max_tokens (not EOS/stop), the last sampled token
    // was never sent. Emit it now to avoid the off-by-one.
    if finish_reason == "length" && !eos.contains(&next_token) {
        let text = decode_token(model, next_token);
        generated.push(next_token);
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
        .map_err(|e| SwarmError::Internal(format!("send final Token: {e}")))?;
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
                return String::from_utf8_lossy(&bytes).into_owned();
            }
            return token_str.clone();
        }
    }
    String::new()
}
