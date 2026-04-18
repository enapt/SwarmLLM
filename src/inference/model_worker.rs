//! Model worker subprocess for SwarmLLM.
//!
//! Each model runs in its own process. When killed, the OS/CUDA driver
//! reclaims ALL GPU memory immediately — solving the "memory doesn't drop
//! on unload" problem and keeping inference off the main daemon's Tokio runtime.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::UnixStream;

use candle_core::IndexOp;

use crate::daemon::shard_loader::{try_load_from_shards, ShardLoadParams};
use crate::error::SwarmError;
use crate::inference::split::{self, KvCacheStore, PrefixCache, SplitModel};
use crate::inference::swift::{SwiftCalibrator, SwiftConfig};
use crate::inference::worker_ipc::*;
use crate::types::NetworkFinishReason;

/// Configuration for the worker's cross-request prefix KV-cache.
#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_prompt_tokens: usize,
    pub block_tokens: usize,
    pub min_tokens: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 16,
            max_prompt_tokens: 8192,
            block_tokens: 64,
            min_tokens: 32,
        }
    }
}

/// Run the model worker subprocess.
/// Called from main.rs when the binary is invoked with `model-worker` subcommand.
/// `shard_window`: if Some, only load these shard indices (VRAM-saving mode).
#[allow(clippy::too_many_arguments)]
pub async fn run_worker(
    socket_path: PathBuf,
    data_dir: PathBuf,
    shard_window: Option<Vec<u32>>,
    kv_cache_ttl_secs: u64,
    prefix_cfg: PrefixCacheConfig,
    swift_cfg: SwiftConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
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

    // Per-model state: (layer_start, layer_end, tp_rank, tp_size) → SplitModel
    // Non-TP models use (start, end, 0, 1). TP-split models use (start, end, rank, size).
    // This allows TP and non-TP inference to coexist without corrupting shared weights.
    let mut models: HashMap<(usize, usize, usize, usize), SplitModel> = HashMap::new();
    let kv_store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(
        kv_cache_ttl_secs,
    )));
    let prefix_cache = Arc::new(PrefixCache::new(
        prefix_cfg.enabled,
        prefix_cfg.max_entries,
        prefix_cfg.block_tokens,
        prefix_cfg.min_tokens,
        prefix_cfg.max_prompt_tokens,
    ));
    tracing::info!(
        enabled = prefix_cfg.enabled,
        max_entries = prefix_cfg.max_entries,
        block_tokens = prefix_cfg.block_tokens,
        min_tokens = prefix_cfg.min_tokens,
        "model-worker: prefix-cache configured"
    );
    tracing::info!(
        enabled = swift_cfg.enabled,
        gamma = swift_cfg.gamma,
        skip_ratio = swift_cfg.skip_ratio,
        calibration_tokens = swift_cfg.calibration_tokens,
        "model-worker: SWIFT self-speculative decoding configured"
    );
    tracing::info!(
        force_standard_attn,
        max_seq_len_override = ?max_seq_len_override,
        "model-worker: attention-kernel + KV-budget overrides"
    );

    // Apply context-length override (process-global, read by the loader on
    // every model construction). Setting once at startup is fine — the worker
    // is single-process and the override doesn't change per request.
    if let Some(cap) = max_seq_len_override {
        crate::inference::split::MAX_SEQ_LEN_OVERRIDE
            .store(cap, std::sync::atomic::Ordering::Relaxed);
    }

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
                if let Err(e) = handle_forward(
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
                    send_worker_error(&mut writer, request_id, e).await;
                }
            }
            DaemonMsg::BatchForward {
                requests,
                activation_lens,
            } => {
                if let Err(e) = handle_batch_forward(
                    &mut writer,
                    &mut models,
                    &kv_store,
                    &data_dir,
                    requests,
                    activation_lens,
                    payload,
                    &shard_window,
                )
                .await
                {
                    // Batch-wide error: reply to first request's id so caller can log.
                    tracing::warn!(error = %e, "model-worker: BatchForward failed");
                }
            }
            DaemonMsg::Generate(gen) => {
                let request_id = gen.request_id;
                if let Err(e) = handle_generate(
                    &mut writer,
                    &mut models,
                    &kv_store,
                    &prefix_cache,
                    &data_dir,
                    gen,
                    &shard_window,
                    &swift_cfg,
                    force_standard_attn,
                    max_seq_len_override,
                )
                .await
                {
                    send_worker_error(&mut writer, request_id, e).await;
                }
            }
            DaemonMsg::Unload {
                layer_start,
                layer_end,
            } => {
                // Remove all entries for this layer range (both TP and non-TP variants)
                models.retain(|&(ls, le, _, _), _| !(ls == layer_start && le == layer_end));
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

/// Send a `WorkerMsg::Error` back to the daemon. Used by the `run_worker`
/// dispatch loop to report handler failures without crashing the subprocess.
async fn send_worker_error(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    request_id: uuid::Uuid,
    err: SwarmError,
) {
    let _ = send_worker(
        writer,
        &WorkerMsg::Error {
            request_id,
            message: err.to_string(),
        },
        &[],
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
/// Ensure a SplitModel is loaded for the given model_id, layer range, and TP config.
/// Non-TP uses (0, 1). TP uses the actual (rank, size).
/// `shard_window`: if Some, only load shards in this set (VRAM-saving mode).
fn ensure_model_loaded(
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    data_dir: &std::path::Path,
    model_id: &crate::types::ModelId,
    layer_start: usize,
    layer_end: usize,
    tp_rank: usize,
    tp_size: usize,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    let key = (layer_start, layer_end, tp_rank, tp_size);
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

    let mut model = if gguf_path.exists() {
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
                    return Err(SwarmError::Validation(
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

    // Apply TP weight splitting if this is a TP variant (tp_size > 1)
    if tp_size > 1 {
        model.pre_split_for_tp(tp_rank, tp_size)?;
    }

    tracing::info!(
        model = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        tp_rank,
        tp_size,
        device = ?model.device(),
        "model-worker: Model loaded"
    );
    models.insert(key, model);
    Ok(())
}

/// Handle a Forward IPC message — run a single-step forward pass.
/// Handle a `DaemonMsg::BatchForward` — multiple forward requests folded into
/// one IPC call. Item 3 Phase 1 (wire protocol only): dispatches each request
/// through the existing `handle_forward` path sequentially, emitting one
/// `WorkerMsg::LayerResult` per request. This eliminates the daemon-side IPC
/// mutex contention (callers no longer serialize on `WorkerHandle.socket`)
/// but does NOT yet give the compute-side tensor-batching speedup. A future
/// phase will replace this with a true `forward_batch` in `SplitModel` that
/// stacks the per-request inputs into a single forward pass and returns a
/// `WorkerMsg::BatchResult`.
#[allow(clippy::too_many_arguments)]
async fn handle_batch_forward(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    requests: Vec<IpcForward>,
    activation_lens: Vec<u32>,
    payload: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    if activation_lens.len() != requests.len() {
        return Err(SwarmError::Internal(format!(
            "BatchForward len mismatch: requests={} activation_lens={}",
            requests.len(),
            activation_lens.len()
        )));
    }
    let mut cursor = 0usize;
    for (fwd, &act_len) in requests.into_iter().zip(activation_lens.iter()) {
        let act_len = act_len as usize;
        let slice = payload
            .get(cursor..cursor + act_len)
            .ok_or_else(|| {
                SwarmError::Internal(format!(
                    "BatchForward payload slice out of range: cursor={cursor} act_len={act_len} total={}",
                    payload.len()
                ))
            })?
            .to_vec();
        cursor += act_len;
        let request_id = fwd.request_id;
        if let Err(e) =
            handle_forward(writer, models, kv_store, data_dir, fwd, slice, shard_window).await
        {
            send_worker_error(writer, request_id, e).await;
        }
    }
    Ok(())
}

async fn handle_forward(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    fwd: IpcForward,
    activation_bytes: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
) -> Result<(), SwarmError> {
    let request_id = fwd.request_id;
    let model_id = fwd.model_id.clone();
    let (layer_start, layer_end) = (fwd.layer_range.0 as usize, fwd.layer_range.1 as usize);

    // Determine TP config for the cache key
    let (tp_rank, tp_size) = fwd
        .tp_meta
        .as_ref()
        .map(|tp| (tp.tp_rank as usize, tp.tp_size as usize))
        .unwrap_or((0, 1));

    // Ensure the TP-specific model variant is loaded and split
    ensure_model_loaded(
        models,
        data_dir,
        &model_id,
        layer_start,
        layer_end,
        tp_rank,
        tp_size,
        shard_window,
    )?;

    let model = models
        .get_mut(&(layer_start, layer_end, tp_rank, tp_size))
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

    // Speculative partial-accept KV fixup: coordinator may request truncation
    // of this request's KV cache to a specific length before the forward runs.
    // Discards trailing stale entries written during a prior verify round.
    if let Some(target_len) = fwd.truncate_kv_to {
        let model_key = format!("{}-{}-{}", layer_start, layer_end, total_layers);
        if let Err(e) = kv_store.truncate_request_to(&model_key, &req_id_str, target_len as usize) {
            tracing::warn!(
                request_id = %request_id,
                target_len,
                error = %e,
                "truncate_request_to failed — proceeding without truncation"
            );
        } else {
            tracing::debug!(
                request_id = %request_id,
                target_len,
                "DIAG: truncated KV cache for speculative partial accept"
            );
        }
    }

    // Speculative verify path: draft_tokens carries γ candidate IDs to verify
    // in a single multi-position forward. We build the input tensor from
    // draft_tokens directly (ignoring activation_bytes) and return γ logit
    // vectors via spec_logits. Only valid when the current segment is both
    // `is_first` (takes token IDs) AND `is_last` (produces logits) — i.e. the
    // full model is on this peer.
    let speculative_verify = fwd.spec_logits_requested && !fwd.draft_tokens.is_empty();
    let input_tensor = if speculative_verify {
        if !is_first || !is_last {
            return Err(SwarmError::Internal(
                "Speculative verify requested on a partial segment (needs both first & last layers)"
                    .into(),
            ));
        }
        let token_ids: Vec<i64> = fwd.draft_tokens.iter().map(|&t| t as i64).collect();
        let seq_len = token_ids.len();
        candle_core::Tensor::from_vec(token_ids, &[1, seq_len], &candle_core::Device::Cpu)
            .map_err(|e| SwarmError::Internal(format!("spec verify tensor: {e}")))?
    } else if pre_embedded {
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
    let tp_meta = fwd.tp_meta.clone();
    let compute_result =
        tokio::task::block_in_place(|| -> Result<crate::types::LayerResult, String> {
            // Speculative verify: multi-position forward returning per-position logits.
            if speculative_verify {
                let output_t = model
                    .forward_verify_all_positions(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                    )
                    .map_err(|e| format!("Forward speculative verify: {e}"))?;
                // output_t shape is [1, seq_len, vocab_size]
                let dims = output_t.dims();
                if dims.len() != 3 {
                    return Err(format!("spec verify unexpected shape: {dims:?}"));
                }
                let seq_len = dims[1];
                let mut spec_logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
                for pos in 0..seq_len {
                    let row = output_t
                        .i((0, pos, ..))
                        .map_err(|e| format!("spec verify slice: {e}"))?;
                    let row = row
                        .to_dtype(candle_core::DType::F32)
                        .map_err(|e| format!("spec verify dtype: {e}"))?;
                    let v: Vec<f32> = row
                        .to_vec1::<f32>()
                        .map_err(|e| format!("spec verify to_vec1: {e}"))?;
                    spec_logits.push(v);
                }
                return Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: vec![],
                    sealed_token_ids: None,
                    spec_logits,
                });
            }

            // TP single-layer forward: process one layer in AttnOnly or FfnOnly phase
            let output = if let Some(ref tp) = tp_meta {
                model
                    .forward_tp_phase(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                        tp.single_layer as usize,
                        &tp.phase,
                    )
                    .map_err(|e| format!("Forward TP phase: {e}"))?
            } else if pre_embedded {
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

            if is_last && tp_meta.is_none() {
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
                    spec_logits: Vec::new(),
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
                    spec_logits: Vec::new(),
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
        spec_logits: result.spec_logits,
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
#[allow(clippy::too_many_arguments)]
async fn handle_generate(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    gen: IpcGenerate,
    shard_window: &Option<Vec<u32>>,
    swift_cfg: &SwiftConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
) -> Result<(), SwarmError> {
    let _ = max_seq_len_override; // applied at worker startup (process-global)
    let request_id = gen.request_id;
    let model_id = gen.model_id.clone();
    let (layer_start, layer_end) = (gen.layer_range.0 as usize, gen.layer_range.1 as usize);

    // Generate is always non-TP (full local inference)
    ensure_model_loaded(
        models,
        data_dir,
        &model_id,
        layer_start,
        layer_end,
        0,
        1,
        shard_window,
    )?;

    let model = models
        .get_mut(&(layer_start, layer_end, 0, 1))
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;

    let req_id_str = request_id.to_string();
    let model_key_string = model.kv_model_key().to_string();

    // Tokenize the prompt to u32 IDs first so we can probe the prefix cache.
    // Probe BEFORE building the input tensor — on hit we forward only the suffix.
    let prompt_ids = model.encode_ids(&gen.prompt);
    let prompt_tokens = prompt_ids.len();
    if prompt_tokens == 0 {
        return Err(SwarmError::Internal(
            "empty prompt after tokenization".into(),
        ));
    }

    // Prefix-cache lookup: if a cached prefix is a strict prefix of this
    // prompt, hydrate the request's KV with the snapshot and only forward
    // the suffix.
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    let prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(kv_store, &model_key_string, &req_id_str, snap)
            .unwrap_or(0),
        None => 0,
    };
    // Guard: must have at least one token left to run a forward pass.
    let prefix_len = prefix_len.min(prompt_tokens.saturating_sub(1));

    let (input, index_pos_start) = if prefix_len > 0 {
        tracing::info!(
            %request_id,
            matched_tokens = prefix_len,
            total_tokens = prompt_tokens,
            "DIAG: handle_generate prefix-cache HIT — prefilling suffix only"
        );
        (
            model.tensor_from_ids(&prompt_ids[prefix_len..])?,
            prefix_len,
        )
    } else {
        (model.tensor_from_ids(&prompt_ids)?, 0)
    };

    // SWIFT eligibility decided up-front so the prefill, baseline decode, and
    // verify all run under the same attention kernel when speculative work is
    // active. `force_standard_attn` is the manual override; SWIFT auto-enables
    // it because draft and verify must produce identical logits.
    let swift_active = swift_cfg.enabled
        && gen.sampling.temperature == 0.0
        && model.total_layers >= 8
        && gen.sampling.max_tokens >= (swift_cfg.gamma + 1);
    let force_attn = force_standard_attn || swift_active;

    // Prefill — block_in_place for CPU-bound inference
    let logits = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
        model.forward(&input, index_pos_start, kv_store, &req_id_str)
    })?;

    // After prefill the KV cache holds exactly `prompt_tokens` positions.
    // Snapshot it into the prefix cache so future prompts sharing this
    // prefix skip the prefill work. insert_from_kv is a no-op when the
    // prompt is shorter than the configured floor or the cache is off.
    prefix_cache.insert_from_kv(&model_key_string, &req_id_str, kv_store, &prompt_ids);

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

    if swift_active {
        let calibrator = SwiftCalibrator::new(
            model.total_layers,
            swift_cfg.skip_ratio,
            swift_cfg.calibration_tokens,
        );
        let outcome = swift_decode_loop(
            writer,
            model,
            kv_store,
            &req_id_str,
            request_id,
            &gen,
            &eos,
            stop_sequences,
            use_logprobs,
            swift_cfg.gamma as usize,
            &calibrator,
            &mut next_token,
            token_logprob,
            &mut index_pos,
            &mut generated,
            &mut accumulated_text,
        )
        .await?;
        finish_reason = outcome;
        tracing::info!(
            request_id = %request_id,
            rounds = calibrator.rounds(),
            acceptance_rate = calibrator.acceptance_rate(),
            num_candidates = calibrator.num_candidates(),
            selected = ?calibrator.selected_candidate(),
            "DIAG: SWIFT session complete"
        );
    } else {
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
                let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
                model.forward(&input, index_pos, kv_store, &req_id_str)
            })?;
            let (tok, lp) =
                crate::inference::tensor_util::sample_token_with_logprob(&logits, &gen.sampling)?;
            next_token = tok;
            token_logprob = lp;
            index_pos += 1;
        }

        // If the loop exhausted max_tokens (not EOS/stop), the last sampled
        // token was never sent. Emit it now to avoid the off-by-one.
        // Skip when max_tokens == 0 — user explicitly requested no completion
        // tokens.
        if finish_reason == "length" && gen.sampling.max_tokens > 0 && !eos.contains(&next_token) {
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

    // Free KV cache for this request to prevent VRAM leak across requests
    kv_store.clear_request(model.kv_model_key(), &req_id_str);

    Ok(())
}

/// SWIFT (arxiv 2410.06916) decode loop. Greedy-only v1.
///
/// Each round:
/// 1. **Draft**: feed the current `next_token` plus γ-1 just-sampled draft
///    tokens through `forward_with_skip_mask` one position at a time. Skipped
///    layers don't write KV — only layers in the kept set do.
/// 2. **KV truncate**: roll non-skipped layers' KV back to the round-start
///    position so the verify pass writes positions p..p+γ correctly. Skipped
///    layers are no-ops here.
/// 3. **Verify**: one full forward over `[next_token, d_0, .., d_{γ-1}]` at
///    `index_pos = p`, returning logits at all γ+1 positions.
/// 4. **Greedy accept-reject**: longest matching prefix where draft argmax
///    equals target argmax. Bonus = target's argmax at the rejection point or
///    at position γ if all accepted.
/// 5. **Final truncate**: shrink KV to `p + accepted_count + 1` so rejected
///    draft positions are discarded.
///
/// Falls through cleanly if any forward fails — the caller will report.
#[allow(clippy::too_many_arguments)]
async fn swift_decode_loop(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    model: &mut SplitModel,
    kv_store: &Arc<KvCacheStore>,
    req_id_str: &str,
    request_id: uuid::Uuid,
    gen: &IpcGenerate,
    eos: &[u32],
    stop_sequences: &[String],
    use_logprobs: bool,
    gamma: usize,
    calibrator: &SwiftCalibrator,
    next_token: &mut u32,
    initial_logprob: Option<f32>,
    index_pos: &mut usize,
    generated: &mut Vec<u32>,
    accumulated_text: &mut String,
) -> Result<String, SwarmError> {
    let model_key = model.kv_model_key().to_string();

    // Local helper: emit a single committed token and update bookkeeping.
    // Returns Ok(true) when the caller should break (EOS, stop, or budget
    // exhausted). The token IS pushed to `generated` and sent.
    async fn emit_token(
        writer: &mut tokio::net::unix::OwnedWriteHalf,
        model: &SplitModel,
        request_id: uuid::Uuid,
        eos: &[u32],
        stop_sequences: &[String],
        use_logprobs: bool,
        logprob: Option<f32>,
        max_tokens: u32,
        token: u32,
        generated: &mut Vec<u32>,
        accumulated_text: &mut String,
    ) -> Result<EmitOutcome, SwarmError> {
        if eos.contains(&token) {
            return Ok(EmitOutcome::Stop);
        }
        if generated.len() as u32 >= max_tokens {
            return Ok(EmitOutcome::Length);
        }
        let text = decode_token(model, token);
        accumulated_text.push_str(&text);
        if crate::inference::sampling::find_stop_sequence(accumulated_text, stop_sequences)
            .is_some()
        {
            return Ok(EmitOutcome::Stop);
        }
        generated.push(token);
        send_worker(
            writer,
            &WorkerMsg::Token {
                request_id,
                token_id: token,
                text,
                is_eos: false,
                logprob: if use_logprobs { logprob } else { None },
            },
            &[],
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("send Token: {e}")))?;
        Ok(EmitOutcome::Continue)
    }

    // Track logprob of the most recent committed token (we only have a real
    // logprob for the prefill-sampled token; subsequent tokens are reported
    // without logprobs in v1 since SWIFT's verify path skips the logprob
    // sampler bookkeeping).
    let mut pending_logprob = initial_logprob;

    loop {
        // Stop conditions on the carried `next_token` BEFORE running a round.
        if eos.contains(next_token) {
            return Ok("stop".into());
        }
        if generated.len() as u32 >= gen.sampling.max_tokens {
            return Ok("length".into());
        }

        let p_start = *index_pos;
        let remaining_budget = gen.sampling.max_tokens - generated.len() as u32;
        // Need budget for at least 1 emitted token from this round; if the
        // budget can't cover next_token alone, just emit it via the per-token
        // fallback.
        if remaining_budget == 0 {
            return Ok("length".into());
        }

        // ── Phase 1: draft γ tokens with the skip mask ──
        // The calibrator picks the candidate pattern: round-robin during the
        // calibration window, pinned best-accept after.
        let cand_idx = calibrator.next_candidate();
        let skip_mask = calibrator.pattern(cand_idx);
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(gamma);
        let mut current_token = *next_token;
        for k_offset in 0..gamma {
            let input = model.token_tensor(current_token)?;
            let logits = tokio::task::block_in_place(|| {
                let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(true);
                model.forward_with_skip_mask(
                    &input,
                    p_start + k_offset,
                    kv_store,
                    req_id_str,
                    skip_mask,
                )
            })?;
            // Greedy argmax — SWIFT v1 is greedy-only.
            let sampled = sample_argmax(&logits)?;
            draft_tokens.push(sampled);
            current_token = sampled;
        }

        // ── Phase 2: roll the KV cache back to p_start so verify writes the
        // canonical positions for ALL layers (including the ones we skipped).
        kv_store.truncate_request_to(&model_key, req_id_str, p_start)?;

        // ── Phase 3: verify with one full forward over γ+1 inputs ──
        let mut verify_ids = Vec::with_capacity(gamma + 1);
        verify_ids.push(*next_token);
        verify_ids.extend(draft_tokens.iter().copied());
        let verify_input = model.tensor_from_ids(&verify_ids)?;
        let verify_logits = tokio::task::block_in_place(|| {
            let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(true);
            model.forward_verify_all_positions(&verify_input, p_start, kv_store, req_id_str)
        })?;
        // Shape: [1, γ+1, vocab].

        // ── Phase 4: greedy accept-reject ──
        let mut accepted_count = 0usize;
        let mut rejection_logits_idx: Option<usize> = None;
        for (k_idx, &draft_tok) in draft_tokens.iter().enumerate() {
            let pos_logits = slice_position_logits(&verify_logits, k_idx)?;
            let target_argmax = sample_argmax(&pos_logits)?;
            if target_argmax == draft_tok {
                accepted_count += 1;
            } else {
                rejection_logits_idx = Some(k_idx);
                break;
            }
        }
        let bonus_logits_idx = rejection_logits_idx.unwrap_or(gamma);
        let bonus_logits = slice_position_logits(&verify_logits, bonus_logits_idx)?;
        let bonus_token = sample_argmax(&bonus_logits)?;

        calibrator.record(cand_idx, gamma as u32, accepted_count as u32);

        // ── Phase 5: emit committed tokens. The carried next_token is
        // committed at p_start, then accepted draft tokens, then bonus.
        let mut break_outcome: Option<String> = None;

        // Commit next_token at p_start (this is the token we sampled at the
        // end of the previous round — or from prefill on the first round).
        match emit_token(
            writer,
            model,
            request_id,
            eos,
            stop_sequences,
            use_logprobs,
            pending_logprob.take(),
            gen.sampling.max_tokens,
            *next_token,
            generated,
            accumulated_text,
        )
        .await?
        {
            EmitOutcome::Continue => {}
            EmitOutcome::Stop => break_outcome = Some("stop".into()),
            EmitOutcome::Length => break_outcome = Some("length".into()),
        }

        if break_outcome.is_none() {
            for tok in draft_tokens.iter().take(accepted_count) {
                match emit_token(
                    writer,
                    model,
                    request_id,
                    eos,
                    stop_sequences,
                    use_logprobs,
                    None,
                    gen.sampling.max_tokens,
                    *tok,
                    generated,
                    accumulated_text,
                )
                .await?
                {
                    EmitOutcome::Continue => {}
                    EmitOutcome::Stop => {
                        break_outcome = Some("stop".into());
                        break;
                    }
                    EmitOutcome::Length => {
                        break_outcome = Some("length".into());
                        break;
                    }
                }
            }
        }

        // Truncate KV cache to the accepted prefix so rejected positions are
        // discarded. The bonus token is NOT in the cache yet — it becomes
        // the new `next_token` and will be committed next round.
        kv_store.truncate_request_to(&model_key, req_id_str, p_start + accepted_count + 1)?;

        if let Some(reason) = break_outcome {
            return Ok(reason);
        }

        *next_token = bonus_token;
        *index_pos = p_start + accepted_count + 1;
    }
}

#[derive(Debug, PartialEq)]
enum EmitOutcome {
    Continue,
    Stop,
    Length,
}

/// Greedy argmax over a logits tensor. Accepts shapes `[1, vocab]` (decode
/// step output) or `[vocab]` (already squeezed). Returns the argmax token id.
fn sample_argmax(logits: &candle_core::Tensor) -> Result<u32, SwarmError> {
    use candle_core::DType;
    let l = if logits.dims().len() == 2 {
        logits.squeeze(0).map_err(SwarmError::internal)?
    } else {
        logits.clone()
    };
    let l = l.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let v: Vec<f32> = l.to_vec1::<f32>().map_err(SwarmError::internal)?;
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    Ok(best as u32)
}

/// Slice the per-position logits out of a verify-pass output of shape
/// `[1, seq_len, vocab]`, returning `[1, vocab]`.
fn slice_position_logits(
    verify_logits: &candle_core::Tensor,
    pos: usize,
) -> Result<candle_core::Tensor, SwarmError> {
    use candle_core::IndexOp;
    verify_logits
        .i((.., pos, ..))
        .map_err(|e| SwarmError::Internal(format!("slice verify logits at pos {pos}: {e}")))
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
