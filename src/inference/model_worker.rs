//! Model worker subprocess for SwarmLLM.
//!
//! Each model runs in its own process. When killed, the OS/CUDA driver
//! reclaims ALL GPU memory immediately — solving the "memory doesn't drop
//! on unload" problem and keeping inference off the main daemon's Tokio runtime.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::inference::process_pool::IpcWriter;

use candle_core::IndexOp;

use crate::daemon::shard_loader::{try_load_from_shards, ShardLoadParams};

/// Item 8 Phase 2b: per-probe waiter. `handle_generate` /
/// `try_register_generate_slot` register a oneshot keyed by a fresh probe
/// `Uuid` and await the daemon's response. The reader task fulfils
/// matching `DaemonMsg::PrefixFetchResult` arrivals inline.
type PrefixFetchWaiterMap = Arc<DashMap<Uuid, oneshot::Sender<(u32, Option<Vec<u8>>)>>>;
use crate::error::SwarmError;
use crate::inference::slot_table::{Slot, SlotTable};
use crate::inference::split::{self, BatchItem, KvCacheStore, PrefixCache, SplitModel};
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
/// `batch_generate`: Item 7 — multiple concurrent `Generate` IPC requests
///   share one `forward_batch` per decode tick instead of running serially.
/// `batch_generate_max_slots`: cap on the slot table; admissions beyond this
///   fall through to sequential `handle_generate`.
/// `prefill_chunk_tokens`: Sarathi-style chunked prefill (Phase 2). Each
///   admitted slot starts in `Prefilling` state — every decode tick advances
///   it by this many prompt tokens before the batched decode runs, so a long
///   admission can no longer block decode for more than one chunk's worth of
///   compute.
#[allow(clippy::too_many_arguments)]
pub async fn run_worker(
    socket_name: String,
    data_dir: PathBuf,
    shard_window: Option<Vec<u32>>,
    kv_cache_ttl_secs: u64,
    prefix_cfg: PrefixCacheConfig,
    swift_cfg: SwiftConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
    activation_compression: bool,
    batch_generate: bool,
    batch_generate_max_slots: u32,
    prefill_chunk_tokens: u32,
    batched_prefill_forward: bool,
) {
    // Connect to the daemon's IPC socket. The name matches what the daemon
    // bound: a filesystem path on Unix, a namespace name on Windows.
    use interprocess::local_socket::tokio::{prelude::*, Stream};

    #[cfg(unix)]
    let ipc_name = match socket_name
        .as_str()
        .to_fs_name::<interprocess::local_socket::GenericFilePath>()
    {
        Ok(n) => n,
        Err(e) => {
            eprintln!("model-worker: invalid socket name {socket_name:?}: {e}");
            std::process::exit(1);
        }
    };
    #[cfg(windows)]
    let ipc_name = match socket_name
        .as_str()
        .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
    {
        Ok(n) => n,
        Err(e) => {
            eprintln!("model-worker: invalid socket name {socket_name:?}: {e}");
            std::process::exit(1);
        }
    };
    let stream = match Stream::connect(ipc_name).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("model-worker: failed to connect to {socket_name:?}: {e}");
            std::process::exit(1);
        }
    };
    let (mut reader, mut writer) = stream.split();

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
        activation_compression,
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

    tracing::info!(
        enabled = batch_generate,
        max_slots = batch_generate_max_slots,
        prefill_chunk_tokens,
        batched_prefill_forward,
        "model-worker: BatchGenerate (Item 7) configured"
    );

    let mut slot_table = SlotTable::new(batch_generate_max_slots as usize);
    let chunk_size_tokens = prefill_chunk_tokens.max(1) as usize;

    // Item 8 Phase 2b: cross-node prefix-KV probe waiters. `handle_generate`
    // and `try_register_generate_slot` register a oneshot keyed by the
    // probe's `request_id` before sending `WorkerMsg::PrefixFetchProbe`;
    // the reader task intercepts matching `DaemonMsg::PrefixFetchResult`
    // and fulfils the oneshot inline (short-circuiting the main loop).
    let pending_fetches: PrefixFetchWaiterMap = Arc::new(DashMap::new());

    // Spawn a reader task that pushes framed IPC messages onto an mpsc.
    // Decoupling read-from-socket from the main select! loop keeps frame
    // alignment safe under cancellation (recv_framed itself is not cancel-safe).
    let (ipc_tx, mut ipc_rx) = mpsc::channel::<(DaemonMsg, Vec<u8>)>(16);
    let reader_pending = pending_fetches.clone();
    let reader_task = tokio::spawn(async move {
        loop {
            match recv_daemon(&mut reader).await {
                Ok((msg, payload)) => {
                    // Short-circuit cross-node prefix fetch results so
                    // `handle_generate` can await its oneshot without
                    // relying on the main loop to pump ipc_rx (which it
                    // can't, since it's blocked inside handle_generate).
                    if let DaemonMsg::PrefixFetchResult {
                        request_id,
                        matched_tokens,
                        present,
                    } = msg
                    {
                        if let Some((_, tx)) = reader_pending.remove(&request_id) {
                            let kv_bytes = if present { Some(payload) } else { None };
                            let _ = tx.send((matched_tokens, kv_bytes));
                        } else {
                            tracing::debug!(
                                %request_id,
                                "prefix-fetch result without waiting probe (timed out?)"
                            );
                        }
                        continue;
                    }
                    if ipc_tx.send((msg, payload)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "model-worker: socket read error");
                    break;
                }
            }
        }
    });

    loop {
        // Either block on the next IPC message (no slots are decoding) OR race
        // a fresh IPC arrival against an immediate decode tick (slots active).
        let next_msg: Option<(DaemonMsg, Vec<u8>)> = if slot_table.is_empty() {
            ipc_rx.recv().await
        } else {
            tokio::select! {
                biased;
                m = ipc_rx.recv() => m,
                _ = tokio::task::yield_now() => None,
            }
        };

        let mut shutdown = false;
        if let Some((msg, payload)) = next_msg {
            shutdown |= handle_daemon_msg(
                msg,
                payload,
                &mut writer,
                &mut models,
                &kv_store,
                &prefix_cache,
                &data_dir,
                &shard_window,
                &swift_cfg,
                force_standard_attn,
                max_seq_len_override,
                activation_compression,
                batch_generate,
                &mut slot_table,
                &pending_fetches,
            )
            .await;

            // Admit-coalescing: drain any further messages already queued on
            // the mpsc before the next decode tick runs. This lets Phase 4
            // (Item 7) batch concurrent admits whose Generates arrive in a
            // cluster — without this drain, each admit gets its own tick,
            // and prefill chunks at different `(chunk_len, index_pos)` never
            // group together. Bounded by the channel capacity (16) so a
            // rogue sender can't starve the tick.
            for _ in 0..16 {
                match ipc_rx.try_recv() {
                    Ok((m2, p2)) => {
                        shutdown |= handle_daemon_msg(
                            m2,
                            p2,
                            &mut writer,
                            &mut models,
                            &kv_store,
                            &prefix_cache,
                            &data_dir,
                            &shard_window,
                            &swift_cfg,
                            force_standard_attn,
                            max_seq_len_override,
                            activation_compression,
                            batch_generate,
                            &mut slot_table,
                            &pending_fetches,
                        )
                        .await;
                    }
                    Err(_) => break,
                }
            }
        } else if slot_table.is_empty() {
            // ipc_rx returned None and no slots — daemon socket closed, exit cleanly.
            break;
        }
        if shutdown {
            break;
        }

        // Decode tick: per-slot prefill chunk (Phase A) + one batched forward
        // across all decoding slots (Phase B). Marks finished slots which the
        // drain step then collects.
        if !slot_table.is_empty() {
            if let Err(e) = step_decode_pool(
                &mut writer,
                &mut models,
                &kv_store,
                &prefix_cache,
                &mut slot_table,
                chunk_size_tokens,
                force_standard_attn,
                batched_prefill_forward,
            )
            .await
            {
                tracing::warn!(error = %e, "model-worker: decode tick failed — finishing all slots with error");
                let drained = std::mem::replace(
                    &mut slot_table,
                    SlotTable::new(batch_generate_max_slots as usize),
                );
                for slot in drained.into_active() {
                    send_worker_error(
                        &mut writer,
                        slot.request_id,
                        SwarmError::Internal(format!("BatchGenerate decode failed: {e}")),
                    )
                    .await;
                }
            }
            for finished in slot_table.drain_finished() {
                if let Err(e) = finalize_slot(&mut writer, &kv_store, finished).await {
                    tracing::warn!(error = %e, "model-worker: finalize_slot failed");
                }
            }
        }
    }

    reader_task.abort();
    drop(models);
    tracing::info!("model-worker: exiting cleanly");
}

/// Send a `WorkerMsg::Error` back to the daemon. Used by the `run_worker`
/// dispatch loop to report handler failures without crashing the subprocess.
async fn send_worker_error(writer: &mut IpcWriter, request_id: uuid::Uuid, err: SwarmError) {
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
/// True if every request in a batch can go through `SplitModel::forward_batch`.
///
/// Restrictive on purpose for v1: any special feature (vision, LoRA, spec
/// decoding, TP, pre-embedded input, prefill) falls back to the sequential
/// per-request path. Decode-only same-model-same-layer-range is the 90% case.
fn batch_eligible(requests: &[IpcForward]) -> bool {
    if requests.len() < 2 {
        return false;
    }
    let first = &requests[0];
    for r in requests {
        if r.layer_range != first.layer_range {
            return false;
        }
        if r.tp_meta.is_some() {
            return false;
        }
        if r.vision_embeddings_len != 0 {
            return false;
        }
        if r.adapter_id.is_some() {
            return false;
        }
        if !r.draft_tokens.is_empty() {
            return false;
        }
        if r.spec_logits_requested {
            return false;
        }
        if r.pre_embedded {
            return false;
        }
        if r.truncate_kv_to.is_some() {
            return false;
        }
        // Prefill (index_pos == 0 or sequence_num == 0) is ineligible: input is
        // raw prompt bytes with caller-variable length. Decode has `seq_num > 0`
        // and one i64 token ID per request (8 bytes LE).
        if r.sequence_num == 0 || r.index_pos == 0 {
            return false;
        }
    }
    true
}

/// Handle a `DaemonMsg::BatchForward` — multiple forward requests folded into
/// one IPC call. When the batch passes `batch_eligible`, runs one fused
/// `SplitModel::forward_batch` pass (batched QKV + FFN matmuls, per-slot
/// attention) and emits a single `WorkerMsg::BatchResult`. Otherwise falls
/// through to the sequential `handle_forward` path and emits individual
/// `LayerResult` messages (preserves wire compatibility).
#[allow(clippy::too_many_arguments)]
async fn handle_batch_forward(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    requests: Vec<IpcForward>,
    activation_lens: Vec<u32>,
    payload: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
    activation_compression: bool,
) -> Result<(), SwarmError> {
    if activation_lens.len() != requests.len() {
        return Err(SwarmError::Internal(format!(
            "BatchForward len mismatch: requests={} activation_lens={}",
            requests.len(),
            activation_lens.len()
        )));
    }

    if batch_eligible(&requests) {
        match run_fused_batch_forward(
            writer,
            models,
            kv_store,
            data_dir,
            &requests,
            &activation_lens,
            &payload,
            shard_window,
            activation_compression,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(e) => {
                let msg = e.to_string();
                // Expected fall-through cases (CPU device, etc.) log at debug;
                // real failures log at warn. Either way, callers still get a
                // response via the sequential path below.
                if msg.contains("not profitable") {
                    tracing::debug!(reason = %msg, "Skipping fused batch (CPU)");
                } else {
                    tracing::warn!(error = %msg, "Fused batch forward failed — falling back to sequential");
                }
            }
        }
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
        if let Err(e) = handle_forward(
            writer,
            models,
            kv_store,
            data_dir,
            fwd,
            slice,
            shard_window,
            activation_compression,
        )
        .await
        {
            send_worker_error(writer, request_id, e).await;
        }
    }
    Ok(())
}

/// Run a fused `SplitModel::forward_batch` pass over N eligible requests and
/// emit a single `WorkerMsg::BatchResult`. Caller must have verified
/// `batch_eligible(&requests)`.
#[allow(clippy::too_many_arguments)]
async fn run_fused_batch_forward(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    requests: &[IpcForward],
    activation_lens: &[u32],
    payload: &[u8],
    shard_window: &Option<Vec<u32>>,
    activation_compression: bool,
) -> Result<(), SwarmError> {
    let first = &requests[0];
    let (layer_start, layer_end) = (first.layer_range.0 as usize, first.layer_range.1 as usize);
    let model_id = first.model_id.clone();

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

    // Build per-request input tensors + decode BatchItems.
    let model = models
        .get_mut(&(layer_start, layer_end, 0, 1))
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;

    // CPU fused batching is a net loss at typical decode batch sizes (1-8):
    // candle's CPU matmul is memory-bandwidth-bound so the batched QKV/FFN
    // doesn't amortize a per-call cost, and per-layer Tensor::cat / narrow
    // adds real wall-clock. Measured on a 22-layer 1024-hidden model, batch
    // 2-8 runs 0.5–1.0× sequential (see docs/plans/benchmarks/round3.md).
    // Return an error here so the caller falls through to the sequential
    // handle_forward loop — keeps continuous_batching safe to enable on
    // CPU-only nodes.
    if matches!(model.device(), candle_core::Device::Cpu) {
        return Err(SwarmError::Internal(
            "fused batch not profitable on CPU — falling back to sequential".into(),
        ));
    }

    let is_first = model.is_first();
    let is_last = model.is_last();
    let total_layers = model.total_layers;

    // Slice payload per request and build tensor inputs.
    let mut input_tensors: Vec<candle_core::Tensor> = Vec::with_capacity(requests.len());
    let mut request_id_strings: Vec<String> = Vec::with_capacity(requests.len());
    let mut cursor = 0usize;
    for (r, &act_len) in requests.iter().zip(activation_lens.iter()) {
        let act_len = act_len as usize;
        let slice = payload.get(cursor..cursor + act_len).ok_or_else(|| {
            SwarmError::Internal(format!(
                "Fused batch payload slice out of range: cursor={cursor} act_len={act_len}"
            ))
        })?;
        cursor += act_len;

        let tensor = if is_first {
            // Decode step on first segment: i64 token IDs (8 bytes LE each).
            if slice.is_empty() || slice.len() % 8 != 0 {
                return Err(SwarmError::Internal(format!(
                    "Fused batch decode payload must be a non-empty multiple of 8 bytes (got {})",
                    slice.len()
                )));
            }
            let token_ids: Vec<i64> = slice
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let seq_len = token_ids.len();
            candle_core::Tensor::from_vec(token_ids, &[1, seq_len], &candle_core::Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?
        } else {
            split::bytes_to_tensor(slice)?
        };
        input_tensors.push(tensor);
        request_id_strings.push(r.request_id.to_string());
    }

    // Clear prefill KV if sequence_num == 0 (shouldn't happen for eligible
    // batches, but defensively). Eligibility guarantees sequence_num > 0.
    let model_key = format!("{layer_start}-{layer_end}-{total_layers}");
    let _ = model_key;

    // Build BatchItems (references to our owned tensors + strings).
    let items: Vec<split::BatchItem<'_>> = requests
        .iter()
        .zip(input_tensors.iter())
        .zip(request_id_strings.iter())
        .map(|((r, t), rid)| split::BatchItem {
            input: t,
            index_pos: r.index_pos as usize,
            request_id: rid.as_str(),
        })
        .collect();

    // Fused forward — CPU-heavy, use block_in_place so we don't starve
    // tokio reactors waiting on other IPC messages.
    let results: Vec<candle_core::Tensor> =
        tokio::task::block_in_place(|| model.forward_batch(&items, kv_store))?;

    if results.len() != requests.len() {
        return Err(SwarmError::Internal(format!(
            "forward_batch returned {} outputs for {} requests",
            results.len(),
            requests.len()
        )));
    }

    // Convert per-request output tensors to IpcLayerResult + payload bytes.
    let mut ipc_results: Vec<IpcLayerResult> = Vec::with_capacity(requests.len());
    let mut result_lens: Vec<u32> = Vec::with_capacity(requests.len());
    let mut concat_payload: Vec<u8> = Vec::new();

    for (r, output_t) in requests.iter().zip(results.iter()) {
        if is_last {
            // Logits [1, vocab] → sample + EOS check.
            let token_id = split::sample_token_with_params(output_t, &r.sampling)
                .map_err(|e| SwarmError::Internal(format!("Sample: {e}")))?;
            let finish = if model.eos_tokens().contains(&token_id) {
                Some(crate::types::NetworkFinishReason::Stop)
            } else {
                None
            };
            ipc_results.push(IpcLayerResult {
                request_id: r.request_id,
                token_ids: vec![token_id],
                finish_reason: finish,
                format: None,
                sealed: false,
                sealed_payload: None,
                logprobs: None,
                has_activations: false,
                has_spec_logits: false,
                spec_logits_dims: None,
            });
            result_lens.push(0);
        } else {
            // Hidden state [1, 1, hidden] → serialize.
            let bytes = if activation_compression {
                split::tensor_to_bytes_q8_0(output_t)
                    .map_err(|e| SwarmError::Internal(format!("Encode Q8_0: {e}")))?
            } else {
                split::tensor_to_bytes(output_t)
                    .map_err(|e| SwarmError::Internal(format!("Encode: {e}")))?
            };
            let len = bytes.len() as u32;
            concat_payload.extend_from_slice(&bytes);
            ipc_results.push(IpcLayerResult {
                request_id: r.request_id,
                token_ids: Vec::new(),
                finish_reason: None,
                format: None,
                sealed: false,
                sealed_payload: None,
                logprobs: None,
                has_activations: true,
                has_spec_logits: false,
                spec_logits_dims: None,
            });
            result_lens.push(len);
        }
    }

    send_worker(
        writer,
        &WorkerMsg::BatchResult {
            results: ipc_results,
            activation_lens: result_lens,
        },
        &concat_payload,
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send BatchResult: {e}")))?;

    tracing::debug!(
        batch_size = requests.len(),
        is_last,
        "DIAG: fused batch forward complete"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn handle_forward(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    data_dir: &std::path::Path,
    fwd: IpcForward,
    mut activation_bytes: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
    activation_compression: bool,
) -> Result<(), SwarmError> {
    let request_id = fwd.request_id;
    let model_id = fwd.model_id.clone();
    let (layer_start, layer_end) = (fwd.layer_range.0 as usize, fwd.layer_range.1 as usize);

    // Split the compound payload: daemon sends `[vision_bytes][activation_bytes]`
    // with `vision_embeddings_len` giving the prefix boundary. Before
    // gotcha #24's spec_logits fix generalized to vision, these lived
    // inside the JSON header as `Vec<u8>` and bloated ~5× through serde_json.
    let vision_bytes: Vec<u8> = if fwd.vision_embeddings_len == 0 {
        Vec::new()
    } else {
        let vlen = fwd.vision_embeddings_len as usize;
        if vlen > activation_bytes.len() {
            return Err(SwarmError::Internal(format!(
                "vision_embeddings_len={vlen} exceeds payload len={}",
                activation_bytes.len()
            )));
        }
        let rest = activation_bytes.split_off(vlen);
        std::mem::replace(&mut activation_bytes, rest)
    };

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
    let model_key = model.kv_model_key().to_string();
    let req_id_str = request_id.to_string();
    let pre_embedded = fwd.pre_embedded;

    // Clear per-request KV-cache at the start of a new request (prefill)
    if fwd.sequence_num == 0 {
        kv_store.clear_request(&model_key, &req_id_str);
    }

    // Speculative partial-accept KV fixup: coordinator may request truncation
    // of this request's KV cache to a specific length before the forward runs.
    // Discards trailing stale entries written during a prior verify round.
    if let Some(target_len) = fwd.truncate_kv_to {
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

    // Speculative output emission is gated on `is_last` only. Input
    // construction goes through the same branches as a non-speculative
    // forward — for the single-segment Item 2 case (`is_first && is_last`)
    // the coordinator now packs all γ token IDs into `activations` (γ × 8
    // bytes LE) so the standard first-segment multi-token decode branch
    // produces the same `[1, γ]` tensor that the old dedicated code path did.
    // For DSD multi-segment (`!is_first && is_last`) the input is a
    // `[1, γ, hidden]` tensor from the previous pipeline segment.
    //
    // `draft_tokens` and `spec_logits_requested` propagate forward unchanged
    // through intermediate segments so the coordinator can still trace the
    // verify request, but only the LAST segment computes spec_logits.
    let want_spec_output = fwd.spec_logits_requested && is_last;
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
            // Decode step: one or more i64 token IDs (8 bytes each, LE).
            //
            // The single-token case (8 bytes) is the standard per-token decode
            // round trip. The multi-token case (γ × 8 bytes, γ ≥ 2) is the
            // distributed-speculative entry point (Item 12 / DSD): the
            // coordinator drafts γ tokens locally and pushes the entire window
            // through the pipeline in one round. Candle's transformer forward
            // is shape-polymorphic in the seq_len dim and will write KV at
            // positions `[index_pos..index_pos+γ]`, so no other layer changes
            // are needed.
            if activation_bytes.is_empty() || activation_bytes.len() % 8 != 0 {
                return Err(SwarmError::Internal(format!(
                    "Decode step activation payload must be a non-empty multiple of 8 bytes (got {})",
                    activation_bytes.len()
                )));
            }
            let token_ids: Vec<i64> = activation_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let seq_len = token_ids.len();
            candle_core::Tensor::from_vec(token_ids, &[1, seq_len], &candle_core::Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?
        }
    } else {
        split::bytes_to_tensor(&activation_bytes)?
    };

    // Decompress vision embeddings if present.
    // Wire format: 8-byte header (num_tokens u32 LE + hidden_dim u32 LE) + zstd(FP16 data)
    let vision_tensor: Option<candle_core::Tensor> = if vision_bytes.is_empty() {
        None
    } else {
        let compressed = &vision_bytes;
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
            // Speculative verify: multi-position forward returning per-position
            // logits. In the single-segment Item 2 case (is_first && is_last)
            // the input is raw token IDs and we use the embedding-applying
            // path. In the DSD multi-segment case (!is_first && is_last) the
            // input is hidden state from the previous segment and we skip
            // embedding via the *_pre_embedded variant.
            if want_spec_output {
                let output_t = if is_first {
                    model
                        .forward_verify_all_positions(
                            &input_tensor,
                            fwd.index_pos as usize,
                            kv_store,
                            &req_id_str,
                        )
                        .map_err(|e| format!("Forward speculative verify: {e}"))?
                } else {
                    model
                        .forward_verify_all_positions_pre_embedded(
                            &input_tensor,
                            fwd.index_pos as usize,
                            kv_store,
                            &req_id_str,
                        )
                        .map_err(|e| format!("Forward speculative verify (pre-embedded): {e}"))?
                };
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
                let activation_bytes = if activation_compression {
                    split::tensor_to_bytes_q8_0(&output).map_err(|e| format!("Encode Q8_0: {e}"))?
                } else {
                    split::tensor_to_bytes(&output).map_err(|e| format!("Encode: {e}"))?
                };
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

    let mut result = compute_result.map_err(SwarmError::Internal)?;

    // Build IPC response. The payload slot is single-use: activations and
    // spec_logits are mutually exclusive (spec fires only on the last
    // segment; activations only on non-last). Take ownership of whichever
    // is populated.
    let has_activations = !result.activations.is_empty();
    let has_spec_logits = !result.spec_logits.is_empty();
    let (payload, spec_logits_dims): (Vec<u8>, Option<(u32, u32)>) = if has_spec_logits {
        let (bytes, dims) = crate::inference::worker_ipc::encode_spec_logits(&result.spec_logits);
        (bytes, Some(dims))
    } else if has_activations {
        (std::mem::take(&mut result.activations), None)
    } else {
        (Vec::new(), None)
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
        has_spec_logits,
        spec_logits_dims,
    };

    send_worker(writer, &WorkerMsg::LayerResult(ipc_result), &payload)
        .await
        .map_err(|e| SwarmError::Internal(format!("send LayerResult: {e}")))?;

    Ok(())
}

/// Item 8 Phase 2b: runtime timeout (in milliseconds) for the cross-node
/// prefix-KV probe. Picked to be short relative to prefill latency —
/// missing the window means the local prefill runs, which is no worse
/// than not having the feature at all.
///
/// Sized for 7B-class models on commodity hardware: the serving peer
/// has to route through worker IPC, pull the snapshot from its local
/// PrefixCache, serialize f32 tensors (~70–150 MB for a 7B at
/// 500-token prefix), and ship them back. 500 ms was enough for
/// TinyLlama (28 MB snapshot, Round 6 bench) but too tight for larger
/// models — the Qwen-7B two-daemon bench saw the fetch complete in
/// ~1400 ms while the probe timed out at 500 ms, forcing full local
/// re-prefill. 3000 ms keeps the fallback window reasonable without
/// starving decode on peers that never respond.
const PREFIX_FETCH_TIMEOUT_MS: u64 = 3000;

/// Item 8 Phase 2b: probe the daemon for a cross-node prefix KV hit and,
/// if one arrives inside the timeout, hydrate the request's KV entry from
/// the returned snapshot bytes. Returns the number of tokens seeded (0
/// when no hit or any step fails — caller unconditionally falls through
/// to normal prefill). Non-fatal: any error here is a degraded path, not
/// a correctness issue.
#[allow(clippy::too_many_arguments)]
async fn try_remote_prefix_hydrate(
    writer: &mut IpcWriter,
    model: &SplitModel,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    pending_fetches: &PrefixFetchWaiterMap,
    model_id: &crate::types::ModelId,
    model_key: &str,
    req_id_str: &str,
    prompt_ids: &[u32],
    prompt_tokens: usize,
) -> usize {
    let block_size = prefix_cache.block_tokens();
    if block_size == 0 {
        return 0;
    }
    let blocks = crate::inference::split::compute_block_hashes(prompt_ids, block_size);
    if blocks.is_empty() {
        return 0;
    }
    let probe_id = Uuid::new_v4();
    let (tx, rx) = oneshot::channel::<(u32, Option<Vec<u8>>)>();
    pending_fetches.insert(probe_id, tx);
    struct ProbeGuard<'a> {
        map: &'a PrefixFetchWaiterMap,
        id: Uuid,
    }
    impl<'a> Drop for ProbeGuard<'a> {
        fn drop(&mut self) {
            self.map.remove(&self.id);
        }
    }
    let _guard = ProbeGuard {
        map: pending_fetches,
        id: probe_id,
    };
    if let Err(e) = send_worker(
        writer,
        &WorkerMsg::PrefixFetchProbe {
            request_id: probe_id,
            model_id: model_id.clone(),
            blocks,
        },
        &[],
    )
    .await
    {
        tracing::debug!(error = %e, "prefix-fetch probe: send failed");
        return 0;
    }
    let (matched_tokens, payload) = match tokio::time::timeout(
        std::time::Duration::from_millis(PREFIX_FETCH_TIMEOUT_MS),
        rx,
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(_)) => {
            tracing::debug!(%probe_id, "prefix-fetch probe: oneshot dropped");
            return 0;
        }
        Err(_) => {
            tracing::debug!(%probe_id, "prefix-fetch probe: timed out");
            return 0;
        }
    };
    let Some(bytes) = payload else { return 0 };
    // Leave at least one token for the forward pass — same rule as the
    // local-hit path, keeps sampling-logits invariants intact.
    let usable = (matched_tokens as usize).min(prompt_tokens.saturating_sub(1));
    if usable == 0 {
        return 0;
    }
    match prefix_cache.hydrate_request_from_bytes(
        kv_store,
        model_key,
        req_id_str,
        &bytes,
        model.device(),
    ) {
        Ok(n) => {
            let n = n.min(usable);
            tracing::info!(
                matched_tokens = n,
                total_tokens = prompt_tokens,
                bytes = bytes.len(),
                "DIAG: cross-node prefix HIT — hydrated KV"
            );
            n
        }
        Err(e) => {
            tracing::warn!(error = %e, "prefix-fetch: hydrate from bytes failed");
            0
        }
    }
}

/// Handle a Generate IPC message — run a full tokenize+decode loop.
#[allow(clippy::too_many_arguments)]
async fn handle_generate(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    gen: IpcGenerate,
    shard_window: &Option<Vec<u32>>,
    swift_cfg: &SwiftConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
    pending_fetches: &PrefixFetchWaiterMap,
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
    // the suffix. Try local first (free); on miss, probe cross-node (Item 8
    // Phase 2b).
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(kv_store, &model_key_string, &req_id_str, snap)
            .unwrap_or(0),
        None => 0,
    };
    // Clamp to keep at least one token for the forward pass.
    prefix_len = prefix_len.min(prompt_tokens.saturating_sub(1));
    if prefix_len == 0 {
        // Only probe when the local cache missed — avoids wasting a round
        // trip when we already have a good hydration source.
        prefix_len = try_remote_prefix_hydrate(
            writer,
            model,
            kv_store,
            prefix_cache,
            pending_fetches,
            &model_id,
            &model_key_string,
            &req_id_str,
            &prompt_ids,
            prompt_tokens,
        )
        .await;
    }

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
    let manifest =
        prefix_cache.insert_from_kv(&model_key_string, &req_id_str, kv_store, &prompt_ids);
    if !manifest.is_empty() {
        let _ = send_worker(
            writer,
            &WorkerMsg::PrefixManifestUpdate {
                model_id: model_id.clone(),
                blocks: manifest,
            },
            &[],
        )
        .await;
    }

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
    writer: &mut IpcWriter,
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
        writer: &mut IpcWriter,
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

/// Reasons a slot admission can fail.
enum SlotAdmitError {
    /// Caller should fall back to sequential `handle_generate` with the
    /// original `IpcGenerate`. Boxed to keep the `Err` variant small (the
    /// `Result<(), SlotAdmitError>` happy path wins on layout).
    FallThrough(Box<IpcGenerate>),
    /// Unrecoverable error from admission; emit Error to the daemon.
    Fatal(SwarmError),
}

/// Cheap admission gate. Mirrors the carve-outs in `handle_generate`: SWIFT
/// gets its own decode loop, so SWIFT-eligible requests always go sequential.
/// Anything that doesn't fit inside one batched `forward_batch` per tick also
/// goes sequential.
fn slot_admission_eligible(
    gen: &IpcGenerate,
    swift_cfg: &SwiftConfig,
    slot_table: &SlotTable,
) -> bool {
    if gen.sampling.max_tokens == 0 {
        return false;
    }
    // Layer range must match if anything is already in the table.
    let lr = (gen.layer_range.0 as usize, gen.layer_range.1 as usize);
    if !slot_table.can_admit(lr) {
        return false;
    }
    // SWIFT decoding has its own self-speculative loop; not batchable v1.
    if swift_cfg.enabled && gen.sampling.temperature == 0.0 {
        return false;
    }
    true
}

/// Lightweight slot registration — Phase 2 chunked-prefill version.
///
/// Tokenizes the prompt, performs the prefix-cache lookup + KV hydration if
/// applicable, and pushes a `Slot` in `Prefilling` state into the table. Does
/// NO compute — the chunked prefill happens inside `step_decode_pool`'s
/// Phase A on subsequent ticks. `async` because Item 8 Phase 2b may issue a
/// cross-node prefix-KV probe when the local cache misses; the probe
/// round-trip is bounded by `PREFIX_FETCH_TIMEOUT_MS`.
#[allow(clippy::too_many_arguments)]
async fn try_register_generate_slot(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    gen: IpcGenerate,
    shard_window: &Option<Vec<u32>>,
    slot_table: &mut SlotTable,
    pending_fetches: &PrefixFetchWaiterMap,
) -> Result<(), SlotAdmitError> {
    let request_id = gen.request_id;
    let model_id = gen.model_id.clone();
    let (layer_start, layer_end) = (gen.layer_range.0 as usize, gen.layer_range.1 as usize);

    if let Err(e) = ensure_model_loaded(
        models,
        data_dir,
        &model_id,
        layer_start,
        layer_end,
        0,
        1,
        shard_window,
    ) {
        return Err(SlotAdmitError::Fatal(e));
    }

    let model = match models.get_mut(&(layer_start, layer_end, 0, 1)) {
        Some(m) => m,
        None => {
            return Err(SlotAdmitError::Fatal(SwarmError::Internal(
                "Model vanished after load".into(),
            )))
        }
    };

    let req_id_str = request_id.to_string();
    let model_key_string = model.kv_model_key().to_string();

    let prompt_ids = model.encode_ids(&gen.prompt);
    let prompt_tokens = prompt_ids.len();
    if prompt_tokens == 0 {
        return Err(SlotAdmitError::Fatal(SwarmError::Internal(
            "empty prompt after tokenization".into(),
        )));
    }

    // Prefix-cache lookup + per-request KV hydration if we hit. Cheap clone of
    // K/V tensors — no compute.
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(kv_store, &model_key_string, &req_id_str, snap)
            .unwrap_or(0),
        None => 0,
    };
    // Always leave at least one prompt token for the first chunk's forward —
    // we need that forward to produce logits for the first sample.
    prefix_len = prefix_len.min(prompt_tokens.saturating_sub(1));
    if prefix_len == 0 {
        // Item 8 Phase 2b: probe cross-node only when local missed.
        prefix_len = try_remote_prefix_hydrate(
            writer,
            model,
            kv_store,
            prefix_cache,
            pending_fetches,
            &model_id,
            &model_key_string,
            &req_id_str,
            &prompt_ids,
            prompt_tokens,
        )
        .await;
    }
    if prefix_len > 0 {
        tracing::info!(
            %request_id,
            matched_tokens = prefix_len,
            total_tokens = prompt_tokens,
            "DIAG: try_register_generate_slot prefix-cache HIT"
        );
    }

    // Re-check capacity — admission gate already checked, but a second admit
    // may have raced ahead. If so, drop the hydrated KV state and fall back
    // to sequential handling.
    let lr = (layer_start, layer_end);
    if !slot_table.can_admit(lr) {
        kv_store.clear_request(model.kv_model_key(), &req_id_str);
        return Err(SlotAdmitError::FallThrough(Box::new(gen)));
    }

    let remaining_ids: Vec<u32> = prompt_ids[prefix_len..].to_vec();
    let max_tokens = gen.sampling.max_tokens;
    let use_logprobs = gen.sampling.logprobs;
    let stop_sequences = gen.sampling.stop.clone();
    let sampling = gen.sampling.clone();
    let eos = model.eos_tokens().to_vec();

    let slot = Slot {
        request_id,
        req_id_str,
        model_key: model_key_string,
        model_id: model_id.clone(),
        layer_range: lr,
        state: crate::inference::slot_table::SlotState::Prefilling {
            remaining_ids,
            next_chunk_index_pos: prefix_len,
        },
        max_tokens,
        use_logprobs,
        eos,
        stop_sequences,
        accumulated_text: String::new(),
        sampling,
        prompt_tokens,
        prompt_ids,
        finish_reason: None,
        error_message: None,
    };
    let prompt_len = slot.prompt_tokens;
    let remaining_after_prefix = prompt_len.saturating_sub(prefix_len);
    slot_table.admit(slot);
    tracing::debug!(
        %request_id,
        prompt_tokens = prompt_len,
        prefix_matched = prefix_len,
        remaining_to_prefill = remaining_after_prefix,
        slots_active = slot_table.len(),
        "DIAG: BatchGenerate slot registered"
    );
    Ok(())
}

/// One decode tick across every active slot in the table.
///
/// **Phase A — Sarathi-style chunked prefill.** Every `Prefilling` slot
/// advances by up to `chunk_size` prompt tokens. When a slot's final chunk
/// runs, we sample its first decode token, snapshot its KV into the prefix
/// cache, and transition the slot to `Decoding` so it joins this same
/// tick's batched decode (Phase B).
///
/// **Phase B — batched decode.** Every `Decoding` slot's `last_token` is
/// fed through `forward_batch`. Per-slot sampling, EOS / stop-string gate
/// (mirrors `handle_generate` byte-for-byte), and Token emit happen inline.
/// Slots that hit `max_tokens` emit the off-by-one Token in the same tick
/// and get marked `length`.
#[allow(clippy::too_many_arguments)]
async fn step_decode_pool(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    slot_table: &mut SlotTable,
    chunk_size: usize,
    force_standard_attn: bool,
    batched_prefill_forward: bool,
) -> Result<(), SwarmError> {
    if slot_table.is_empty() {
        return Ok(());
    }
    let layer_range = slot_table
        .current_layer_range()
        .ok_or_else(|| SwarmError::Internal("slot table non-empty but layer_range unset".into()))?;
    let (layer_start, layer_end) = layer_range;
    let model = models
        .get_mut(&(layer_start, layer_end, 0, 1))
        .ok_or_else(|| {
            SwarmError::Internal("model variant evicted between admit and tick".into())
        })?;

    // ---- PHASE A: chunked prefill (Item 7 Phase 4 batched variant) ----
    //
    // Every `Prefilling` slot contributes one `PrefillStep` (a tensor +
    // index_pos + remaining_after). Steps sharing `(chunk_len, index_pos)`
    // are fused into one `forward_batch` call; singletons fall through to
    // sequential `forward`. Per-slot error containment: tensor build failure
    // marks THAT slot; a batched-forward failure errors every slot in the
    // group (strict fall-back would duplicate work and rarely helps in
    // practice — catastrophic OOM / kernel failures apply to every slot).
    {
        struct PrefillStep {
            slot_idx: usize,
            request_id: uuid::Uuid,
            req_id_str: String,
            input: candle_core::Tensor,
            pos: usize,
            chunk_len: usize,
            remaining_after: usize,
        }

        let active = slot_table.active();

        // Stage 1: collect chunks + build input tensors.
        let mut steps: Vec<PrefillStep> = Vec::new();
        for (slot_idx, slot) in active.iter_mut().enumerate() {
            if !slot.is_prefilling() || slot.is_finished() {
                continue;
            }
            let (chunk, pos, remaining_after) = match slot.take_prefill_chunk(chunk_size) {
                Some(t) => t,
                None => continue,
            };
            let request_id = slot.request_id;
            let req_id_str = slot.req_id_str.clone();
            let chunk_len = chunk.len();
            match model.tensor_from_ids(&chunk) {
                Ok(input) => steps.push(PrefillStep {
                    slot_idx,
                    request_id,
                    req_id_str,
                    input,
                    pos,
                    chunk_len,
                    remaining_after,
                }),
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "DIAG: BatchGenerate prefill chunk tensor build failed — slot errored");
                    slot.finish_error(format!("prefill tensor build: {e}"));
                }
            }
        }

        if !steps.is_empty() {
            // Stage 2: group by (chunk_len, index_pos). BTreeMap keeps it
            // deterministic so fused calls run in a stable order.
            // When the flag is on, group by `(chunk_len, index_pos)` so
            // same-shape chunks fuse into one `forward_batch` call. When off,
            // every step is its own singleton (forces Phase A to run
            // sequentially — useful for Phase 4 A/B benchmarks).
            let groups: Vec<Vec<usize>> = if batched_prefill_forward {
                use std::collections::BTreeMap;
                let mut map: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
                for (i, s) in steps.iter().enumerate() {
                    map.entry((s.chunk_len, s.pos)).or_default().push(i);
                }
                map.into_values().collect()
            } else {
                (0..steps.len()).map(|i| vec![i]).collect()
            };

            // Stage 3: forward each group. `logits_per_step[i]` ends up Some
            // on success, None on error (slot already marked errored).
            let mut logits_per_step: Vec<Option<candle_core::Tensor>> =
                std::iter::repeat_with(|| None).take(steps.len()).collect();

            for indices in groups {
                if indices.len() == 1 {
                    // Singleton: sequential forward, cheapest path.
                    let i = indices[0];
                    let step = &steps[i];
                    let forward_result = tokio::task::block_in_place(|| {
                        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(
                            force_standard_attn,
                        );
                        model.forward(&step.input, step.pos, kv_store, &step.req_id_str)
                    });
                    match forward_result {
                        Ok(l) => logits_per_step[i] = Some(l),
                        Err(e) => {
                            let request_id = step.request_id;
                            tracing::warn!(%request_id, error = %e, "DIAG: BatchGenerate prefill chunk forward failed — slot errored");
                            active[step.slot_idx].finish_error(format!("prefill forward: {e}"));
                        }
                    }
                } else {
                    // Fused forward over same-shape, same-position chunks.
                    let chunk_len = steps[indices[0]].chunk_len;
                    let pos = steps[indices[0]].pos;
                    let items: Vec<BatchItem<'_>> = indices
                        .iter()
                        .map(|&i| BatchItem {
                            input: &steps[i].input,
                            index_pos: steps[i].pos,
                            request_id: steps[i].req_id_str.as_str(),
                        })
                        .collect();
                    let forward_result = tokio::task::block_in_place(|| {
                        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(
                            force_standard_attn,
                        );
                        model.forward_batch(&items, kv_store)
                    });
                    match forward_result {
                        Ok(outs) if outs.len() == indices.len() => {
                            tracing::debug!(
                                batch_size = indices.len(),
                                chunk_tokens = chunk_len,
                                index_pos = pos,
                                "DIAG: BatchGenerate prefill chunk fused"
                            );
                            for (j, &i) in indices.iter().enumerate() {
                                logits_per_step[i] = Some(outs[j].clone());
                            }
                        }
                        Ok(outs) => {
                            let err = format!(
                                "forward_batch returned {} outputs for {} prefill chunks",
                                outs.len(),
                                indices.len()
                            );
                            for &i in &indices {
                                let slot_idx = steps[i].slot_idx;
                                let request_id = steps[i].request_id;
                                tracing::warn!(%request_id, %err, "DIAG: BatchGenerate fused prefill output mismatch — slot errored");
                                active[slot_idx].finish_error(err.clone());
                            }
                        }
                        Err(e) => {
                            for &i in &indices {
                                let slot_idx = steps[i].slot_idx;
                                let request_id = steps[i].request_id;
                                tracing::warn!(%request_id, error = %e, "DIAG: BatchGenerate fused prefill forward failed — slot errored");
                                active[slot_idx].finish_error(format!("prefill forward: {e}"));
                            }
                        }
                    }
                }
            }

            // Stage 4: per-step finalize (DIAG trace; first-token sample +
            // prefix-cache insert + promote-to-decoding on the final chunk).
            // Announces are accumulated here and dispatched after the
            // mutable borrow on `active` drops below.
            let mut pending_announces: Vec<(
                crate::types::ModelId,
                Vec<crate::types::PrefixBlockEntry>,
            )> = Vec::new();
            for (i, step) in steps.iter().enumerate() {
                let Some(logits) = logits_per_step[i].take() else {
                    continue;
                };
                let slot = &mut active[step.slot_idx];
                let request_id = step.request_id;
                tracing::debug!(
                    %request_id,
                    chunk_tokens = step.chunk_len,
                    index_pos = step.pos,
                    remaining_after = step.remaining_after,
                    "DIAG: BatchGenerate prefill chunk ran"
                );
                if step.remaining_after == 0 {
                    let (first_token, first_logprob) =
                        match crate::inference::tensor_util::sample_token_with_logprob(
                            &logits,
                            &slot.sampling,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(%request_id, error = %e, "DIAG: BatchGenerate first-token sample failed — slot errored");
                                slot.finish_error(format!("first-token sample: {e}"));
                                continue;
                            }
                        };
                    let manifest = prefix_cache.insert_from_kv(
                        &slot.model_key,
                        &slot.req_id_str,
                        kv_store,
                        &slot.prompt_ids,
                    );
                    if !manifest.is_empty() {
                        pending_announces.push((slot.model_id.clone(), manifest));
                    }
                    let is_eos_first = slot.eos.contains(&first_token);
                    slot.promote_to_decoding(first_token, first_logprob);
                    if is_eos_first {
                        slot.finish_stop();
                    }
                }
            }
            // The `active` borrow ends here (NLL) — `pending_announces` is
            // owned, so we can hop onto the writer without re-borrowing.
            for (model_id, blocks) in pending_announces {
                let _ = send_worker(
                    writer,
                    &WorkerMsg::PrefixManifestUpdate { model_id, blocks },
                    &[],
                )
                .await;
            }
        }
    }

    // ---- PHASE B: batched decode (gate + emit + forward + sample) ----
    {
        let active = slot_table.active();
        for slot in active.iter_mut() {
            if !slot.is_decoding() || slot.is_finished() {
                continue;
            }
            let (last_token, last_logprob) = match &slot.state {
                crate::inference::slot_table::SlotState::Decoding {
                    last_token,
                    last_token_logprob,
                    ..
                } => (*last_token, *last_token_logprob),
                _ => continue,
            };
            if slot.eos.contains(&last_token) {
                slot.finish_stop();
                continue;
            }
            let text = decode_token(model, last_token);
            slot.accumulated_text.push_str(&text);
            if crate::inference::sampling::find_stop_sequence(
                &slot.accumulated_text,
                &slot.stop_sequences,
            )
            .is_some()
            {
                slot.finish_stop();
                continue;
            }
            let logprob = if slot.use_logprobs {
                last_logprob
            } else {
                None
            };
            let request_id = slot.request_id;
            send_worker(
                writer,
                &WorkerMsg::Token {
                    request_id,
                    token_id: last_token,
                    text,
                    is_eos: false,
                    logprob,
                },
                &[],
            )
            .await
            .map_err(|e| SwarmError::Internal(format!("send Token: {e}")))?;
            if let crate::inference::slot_table::SlotState::Decoding {
                generated_count, ..
            } = &mut slot.state
            {
                *generated_count += 1;
            }
        }
    }

    let active = slot_table.active();
    let still_active_indices: Vec<usize> = active
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_decoding() && !s.is_finished())
        .map(|(i, _)| i)
        .collect();
    if still_active_indices.is_empty() {
        return Ok(());
    }

    let mut input_tensors: Vec<candle_core::Tensor> =
        Vec::with_capacity(still_active_indices.len());
    let mut req_id_strs: Vec<String> = Vec::with_capacity(still_active_indices.len());
    let mut sampling_clones: Vec<crate::types::SamplingParams> =
        Vec::with_capacity(still_active_indices.len());
    let mut index_positions: Vec<usize> = Vec::with_capacity(still_active_indices.len());
    // Per-slot tensor build with error containment. If one slot's
    // token_tensor fails (e.g. impossibly large token id), mark it errored
    // and skip — keeping the rest of the batch intact.
    let mut keep_indices: Vec<usize> = Vec::with_capacity(still_active_indices.len());
    for &i in &still_active_indices {
        let slot = &active[i];
        let (last_token, index_pos) = match &slot.state {
            crate::inference::slot_table::SlotState::Decoding {
                last_token,
                index_pos,
                ..
            } => (*last_token, *index_pos),
            _ => unreachable!("filtered to is_decoding above"),
        };
        match model.token_tensor(last_token) {
            Ok(t) => {
                input_tensors.push(t);
                req_id_strs.push(slot.req_id_str.clone());
                sampling_clones.push(slot.sampling.clone());
                index_positions.push(index_pos);
                keep_indices.push(i);
            }
            Err(e) => {
                tracing::warn!(request_id = %slot.request_id, error = %e, "DIAG: BatchGenerate decode token_tensor failed — slot errored");
                // Re-borrow as mut for finish_error.
                let slot_mut = &mut active[i];
                slot_mut.finish_error(format!("decode token_tensor: {e}"));
            }
        }
    }
    let still_active_indices = keep_indices;
    if still_active_indices.is_empty() {
        return Ok(());
    }

    let items: Vec<BatchItem<'_>> = still_active_indices
        .iter()
        .enumerate()
        .map(|(j, _)| BatchItem {
            input: &input_tensors[j],
            index_pos: index_positions[j],
            request_id: req_id_strs[j].as_str(),
        })
        .collect();

    let outputs: Vec<candle_core::Tensor> = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_standard_attn);
        model.forward_batch(&items, kv_store)
    })?;

    if outputs.len() != still_active_indices.len() {
        return Err(SwarmError::Internal(format!(
            "forward_batch returned {} outputs for {} active slots",
            outputs.len(),
            still_active_indices.len()
        )));
    }

    for (j, &i) in still_active_indices.iter().enumerate() {
        let slot = &mut active[i];
        let (next_tok, next_logprob) =
            match crate::inference::tensor_util::sample_token_with_logprob(
                &outputs[j],
                &sampling_clones[j],
            ) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(request_id = %slot.request_id, error = %e, "DIAG: BatchGenerate decode sample failed — slot errored");
                    slot.finish_error(format!("decode sample: {e}"));
                    continue;
                }
            };
        let new_generated_count = match &mut slot.state {
            crate::inference::slot_table::SlotState::Decoding {
                last_token,
                last_token_logprob,
                index_pos,
                generated_count,
            } => {
                *last_token = next_tok;
                *last_token_logprob = next_logprob;
                *index_pos += 1;
                *generated_count
            }
            _ => unreachable!("filtered to is_decoding above"),
        };
        if new_generated_count >= slot.max_tokens as usize {
            if slot.max_tokens > 0 && !slot.eos.contains(&next_tok) {
                let text = decode_token(model, next_tok);
                let logprob = if slot.use_logprobs {
                    next_logprob
                } else {
                    None
                };
                send_worker(
                    writer,
                    &WorkerMsg::Token {
                        request_id: slot.request_id,
                        token_id: next_tok,
                        text,
                        is_eos: false,
                        logprob,
                    },
                    &[],
                )
                .await
                .map_err(|e| SwarmError::Internal(format!("send final Token: {e}")))?;
                if let crate::inference::slot_table::SlotState::Decoding {
                    generated_count, ..
                } = &mut slot.state
                {
                    *generated_count += 1;
                }
            }
            slot.finish_length();
        }
    }
    Ok(())
}

/// Wrap up a finished slot: send GenerateDone + free its per-request KV.
/// The off-by-one Token (when finish="length") is already emitted inside
/// `step_decode_pool` so this is purely bookkeeping.
async fn finalize_slot(
    writer: &mut IpcWriter,
    kv_store: &Arc<KvCacheStore>,
    slot: Slot,
) -> Result<(), SwarmError> {
    let generated_count = slot.generated_count();
    let Slot {
        request_id,
        req_id_str,
        model_key,
        prompt_tokens,
        finish_reason,
        error_message,
        ..
    } = slot;
    let finish_label = finish_reason.unwrap_or("length").to_string();

    if finish_label == "error" {
        let message = error_message
            .unwrap_or_else(|| "BatchGenerate slot failed without a recorded message".to_string());
        send_worker(
            writer,
            &WorkerMsg::Error {
                request_id,
                message,
            },
            &[],
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("send Error: {e}")))?;
        kv_store.clear_request(&model_key, &req_id_str);
        return Ok(());
    }

    send_worker(
        writer,
        &WorkerMsg::GenerateDone {
            request_id,
            prompt_tokens,
            completion_tokens: generated_count,
            finish_reason: finish_label,
        },
        &[],
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send GenerateDone: {e}")))?;

    kv_store.clear_request(&model_key, &req_id_str);
    Ok(())
}

/// Handle a single `DaemonMsg` from the daemon. Extracted from the main select
/// loop so it can be reused inside the admit-coalescing drain loop (drain any
/// messages already queued on the mpsc before running the next decode tick).
///
/// Returns `true` iff the message was `Shutdown` (caller should break).
#[allow(clippy::too_many_arguments)]
async fn handle_daemon_msg(
    msg: DaemonMsg,
    payload: Vec<u8>,
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    shard_window: &Option<Vec<u32>>,
    swift_cfg: &SwiftConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
    activation_compression: bool,
    batch_generate: bool,
    slot_table: &mut SlotTable,
    pending_fetches: &PrefixFetchWaiterMap,
) -> bool {
    match msg {
        DaemonMsg::Forward(fwd) => {
            let request_id = fwd.request_id;
            if let Err(e) = handle_forward(
                writer,
                models,
                kv_store,
                data_dir,
                fwd,
                payload,
                shard_window,
                activation_compression,
            )
            .await
            {
                send_worker_error(writer, request_id, e).await;
            }
        }
        DaemonMsg::BatchForward {
            requests,
            activation_lens,
        } => {
            if let Err(e) = handle_batch_forward(
                writer,
                models,
                kv_store,
                data_dir,
                requests,
                activation_lens,
                payload,
                shard_window,
                activation_compression,
            )
            .await
            {
                tracing::warn!(error = %e, "model-worker: BatchForward failed");
            }
        }
        DaemonMsg::Generate(gen) => {
            let request_id = gen.request_id;
            let mut pending: Option<IpcGenerate> = Some(gen);
            if batch_generate
                && pending
                    .as_ref()
                    .map(|g| slot_admission_eligible(g, swift_cfg, slot_table))
                    .unwrap_or(false)
            {
                let g = pending.take().expect("checked above");
                match try_register_generate_slot(
                    writer,
                    models,
                    kv_store,
                    prefix_cache,
                    data_dir,
                    g,
                    shard_window,
                    slot_table,
                    pending_fetches,
                )
                .await
                {
                    Ok(_) => { /* registered — chunked prefill happens in step_decode_pool */ }
                    Err(SlotAdmitError::Fatal(e)) => {
                        send_worker_error(writer, request_id, e).await;
                    }
                    Err(SlotAdmitError::FallThrough(g)) => {
                        pending = Some(*g);
                    }
                }
            }
            if let Some(g) = pending {
                if let Err(e) = handle_generate(
                    writer,
                    models,
                    kv_store,
                    prefix_cache,
                    data_dir,
                    g,
                    shard_window,
                    swift_cfg,
                    force_standard_attn,
                    max_seq_len_override,
                    pending_fetches,
                )
                .await
                {
                    send_worker_error(writer, request_id, e).await;
                }
            }
        }
        DaemonMsg::Unload {
            layer_start,
            layer_end,
        } => {
            models.retain(|&(ls, le, _, _), _| !(ls == layer_start && le == layer_end));
            tracing::info!(layer_start, layer_end, "model-worker: unloaded shard range");
        }
        DaemonMsg::ExportPrefixSnapshot {
            request_id,
            model_id,
            block_hash,
        } => {
            // Item 8 Phase 2b: serving-side handler. The daemon received an
            // inbound PrefixKvFetch from a peer; look up the hash in our
            // local PrefixCache (the `model_key` depends on the model's
            // layer range, which we can't know without loading — but the
            // cache is keyed by the kv_model_key the SplitModel already
            // uses, so we iterate every loaded model's key to find it).
            let payload = export_snapshot_for_hash(models, prefix_cache, &model_id, &block_hash);
            let present = payload.is_some();
            let payload_slice: &[u8] = payload.as_deref().unwrap_or(&[]);
            let _ = send_worker(
                writer,
                &WorkerMsg::PrefixSnapshotResponse {
                    request_id,
                    present,
                },
                payload_slice,
            )
            .await;
        }
        DaemonMsg::PrefixFetchResult { request_id, .. } => {
            // The reader task short-circuits these and routes them through
            // `pending_fetches` before they reach this path. If we ever see
            // one here it's a routing bug or a late arrival — drop it.
            tracing::debug!(
                %request_id,
                "model-worker: unexpected PrefixFetchResult in main loop (reader short-circuit missed?)"
            );
            let _ = pending_fetches; // quiet unused-param warning when the reader short-circuit is comprehensive
        }
        DaemonMsg::Shutdown => {
            let _ = send_worker(writer, &WorkerMsg::Bye, &[]).await;
            return true;
        }
    }
    false
}

/// Item 8 Phase 2b: find a cached prefix snapshot matching `block_hash`
/// across every loaded model's `kv_model_key`. `kv_model_key` is layer-range
/// scoped (format `{start}-{end}-{block_count}`), NOT model-id scoped, so
/// we try every loaded key and return the first hit. The incoming
/// `model_id` arrives for future trust / rate-limiting but doesn't narrow
/// the cache bucket today.
fn export_snapshot_for_hash(
    models: &HashMap<(usize, usize, usize, usize), SplitModel>,
    prefix_cache: &Arc<PrefixCache>,
    _model_id: &crate::types::ModelId,
    block_hash: &[u8; 32],
) -> Option<Vec<u8>> {
    for model in models.values() {
        let key = model.kv_model_key().to_string();
        if let Some(bytes) = prefix_cache.export_snapshot_bytes(&key, block_hash) {
            return Some(bytes);
        }
    }
    None
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
