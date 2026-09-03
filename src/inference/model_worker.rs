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

use crate::daemon::shard_loader::{try_load_from_shards, ShardLoadParams};

/// Item 8 Phase 2b: per-probe waiter. `handle_generate` /
/// `try_register_generate_slot` register a oneshot keyed by a fresh probe
/// `Uuid` and await the daemon's response. The reader task fulfils
/// matching `DaemonMsg::PrefixFetchResult` arrivals inline.
type PrefixFetchWaiterMap = Arc<DashMap<Uuid, oneshot::Sender<(u32, Option<Vec<u8>>)>>>;

/// Requests the daemon has abandoned, with the instant each cancel arrived.
///
/// Populated by the reader task rather than the main loop, for the same reason
/// `PrefixFetchResult` is short-circuited there: while `handle_generate` is
/// running its decode loop it owns the main loop, so nothing drains `ipc_rx`
/// and a cancel sitting in the channel would not be seen until the generation
/// it was meant to stop had already finished.
///
/// Entries are removed when consumed and swept by age otherwise — a cancel can
/// legitimately arrive for a request that already finished, and those must not
/// accumulate for the life of the worker.
type CancelledSet = Arc<DashMap<Uuid, std::time::Instant>>;

/// How long an unconsumed cancellation is remembered. Covers the window where a
/// cancel overtakes its own request on the IPC channel; well beyond that a
/// lingering entry is just an unmatched cancel for a finished request.
const CANCEL_RETENTION_SECS: u64 = 60;

/// How often the worker reports what its KV cache is holding.
///
/// **This is where the KV cache actually lives for local inference.** The
/// daemon has a `KvCacheStore` too, but that one serves the distributed
/// tensor-forward path; a single node answering its own requests fills the
/// worker's and leaves the daemon's empty, so occupancy logged only in the
/// daemon reports zero on the most common path — which is exactly what the
/// first version of this instrumentation did.
const KV_OCCUPANCY_REPORT_SECS: u64 = 30;
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
    /// Maximum bytes of KV retained per model. 0 disables the byte bound.
    pub max_bytes: usize,
    pub block_tokens: usize,
    pub min_tokens: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 16,
            max_prompt_tokens: 8192,
            max_bytes: 2048 * 1024 * 1024,
            block_tokens: 64,
            min_tokens: 32,
        }
    }
}

/// Runtime knobs passed from the daemon's CLI/config down through the worker.
/// Bundles the seven scalars that previously rode the entry-point signature
/// individually and forced `#[allow(clippy::too_many_arguments)]` on every
/// caller in the chain.
#[derive(Debug, Clone, Copy)]
pub struct WorkerOptions {
    /// Force every attention call through `standard_attention` (matmul) instead
    /// of the fused `run_flash_attn_cpu` path. Diagnostic flag.
    pub force_standard_attn: bool,
    /// Cap GGUF `context_length` when allocating KV cache. None = use the GGUF
    /// metadata value.
    pub max_seq_len_override: Option<usize>,
    /// See `inference::split::CPU_KV_BUDGET_BYTES`: the KV-cache budget a CPU
    /// worker enforces at run time. None = no guard.
    pub kv_budget_bytes: Option<u64>,
    /// Quantize intermediate-segment hidden state activations to Q8_0 before
    /// returning them to the daemon. Off by default; receivers always
    /// auto-dispatch on the dtype tag.
    pub activation_compression: bool,
    /// Item 7 BatchGenerate: multiple concurrent `Generate` requests interleave
    /// through one `forward_batch` per decode tick.
    pub batch_generate: bool,
    /// Maximum number of concurrent decode slots when `batch_generate` is on.
    pub batch_generate_max_slots: u32,
    /// Item 7 Phase 2 chunked prefill chunk size (in prompt tokens). A CEILING
    /// — `prefill_target_ms` picks the operating quantum.
    pub prefill_chunk_tokens: u32,
    /// Longest n-gram the local draft-free speculator tries to match.
    pub ngram_max_size: u32,
    /// Tokens a local speculative round drafts; zero disables the speculator.
    pub ngram_pred_tokens: u32,
    /// Wall-time budget (ms) for one tick's prefill work while slots are
    /// shared. See `inference::prefill_pacer`.
    pub prefill_target_ms: u64,
    /// Item 7 Phase 4: fuse concurrent same-shape Prefilling slots into one
    /// `forward_batch` call inside `step_decode_pool`'s Phase A.
    pub batched_prefill_forward: bool,
    /// Device placement, mirroring `InferenceConfig::gpu_layers`: `-1` auto
    /// (GPU when available), `0` CPU only, `>0` GPU. The split engine places a
    /// worker's whole layer window on one device, so any positive value means
    /// the same thing as auto — see `force_cpu_for` for the mapping.
    pub gpu_layers: i32,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            force_standard_attn: false,
            max_seq_len_override: None,
            kv_budget_bytes: None,
            activation_compression: false,
            batch_generate: false,
            batch_generate_max_slots: 8,
            gpu_layers: -1,
            prefill_chunk_tokens: 128,
            prefill_target_ms: 200,
            // Off unless a caller asks. The daemon always does, from
            // `inference.ngram_lookup_enabled`; a bare `WorkerOptions::default()`
            // (tests, tools) gets the plain decode loop.
            ngram_max_size: 0,
            ngram_pred_tokens: 0,
            batched_prefill_forward: true,
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
pub async fn run_worker(
    socket_name: String,
    data_dir: PathBuf,
    shard_window: Option<Vec<u32>>,
    kv_cache_ttl_secs: u64,
    prefix_cfg: PrefixCacheConfig,
    swift_cfg: SwiftConfig,
    options: WorkerOptions,
) {
    let WorkerOptions {
        force_standard_attn,
        max_seq_len_override,
        kv_budget_bytes,
        activation_compression,
        batch_generate,
        batch_generate_max_slots,
        prefill_chunk_tokens,
        prefill_target_ms,
        ngram_max_size,
        ngram_pred_tokens,
        batched_prefill_forward,
        gpu_layers,
    } = options;
    set_worker_force_cpu(gpu_layers);
    // The local draft-free speculator's shape, built once. `num_pred_tokens ==
    // 0` is how `inference.ngram_lookup_enabled = false` arrives, so there is
    // no separate switch to disagree with the width.
    let ngram_cfg = crate::inference::ngram_lookup::NgramLookupConfig {
        max_ngram_size: ngram_max_size as usize,
        num_pred_tokens: ngram_pred_tokens as usize,
        ..Default::default()
    };
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
        prefix_cfg.max_bytes,
    ));
    // The per-chunk head-room guard can shrink the prefix cache when a
    // request's own KV needs the room (gotcha #440). Weak on both sides:
    // the store must not keep itself alive through its own closure.
    {
        let pc = Arc::downgrade(&prefix_cache);
        let ks = Arc::downgrade(&kv_store);
        kv_store.set_external_evictor(Box::new(move |needed: u64| {
            let (Some(pc), Some(ks)) = (pc.upgrade(), ks.upgrade()) else {
                return 0;
            };
            let freed = pc.release(needed as usize) as u64;
            ks.set_external_reserved(pc.bytes_total() as u64);
            freed
        }));
    }
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
    // KV budget for a CPU worker: same lifetime and scope as the override.
    if let Some(b) = kv_budget_bytes {
        crate::inference::split::CPU_KV_BUDGET_BYTES.store(b, std::sync::atomic::Ordering::Relaxed);
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
    // `prefill_chunk_tokens` is the ceiling; the pacer picks the operating
    // quantum from measured wall time whenever more than one slot is active,
    // so a long prompt cannot starve a co-scheduled request on a slow machine
    // (gotcha #191). One pacer per worker: the cost of a prompt token is a
    // property of this machine and this loaded model.
    let mut last_kv_report = std::time::Instant::now();
    let mut prefill_pacer = crate::inference::prefill_pacer::PrefillPacer::new(
        prefill_chunk_tokens.max(1) as usize,
        prefill_target_ms,
    );

    // Item 8 Phase 2b: cross-node prefix-KV probe waiters. `handle_generate`
    // and `try_register_generate_slot` register a oneshot keyed by the
    // probe's `request_id` before sending `WorkerMsg::PrefixFetchProbe`;
    // the reader task intercepts matching `DaemonMsg::PrefixFetchResult`
    // and fulfils the oneshot inline (short-circuiting the main loop).
    let pending_fetches: PrefixFetchWaiterMap = Arc::new(DashMap::new());
    let cancelled: CancelledSet = Arc::new(DashMap::new());
    // A forward pass asks between layers whether its request was cancelled
    // (gotcha #441): the daemon's cancel reaches the reader task, which
    // writes `cancelled`, and this is how that reaches a prompt pass already
    // running. The set is shared, not cloned, so a cancel is seen at once.
    {
        let cancelled = cancelled.clone();
        kv_store.set_cancel_oracle(Box::new(move |request_id: &str| {
            uuid::Uuid::parse_str(request_id)
                .map(|id| cancelled.contains_key(&id))
                .unwrap_or(false)
        }));
    }

    // Temperature reporting. Polled in the worker because this is the process
    // actually doing the arithmetic, so it is the one whose heat is worth
    // reporting. Every other resource this node spends has a ceiling; heat had
    // nothing at all until a reporter's laptop went 71 °C → 88 °C in five
    // minutes on a model that had silently fallen back to the CPU (2026-08-10),
    // and they only knew because they were watching `k10temp` themselves.
    //
    // Reports; does not act. An automatic thread reduction was built and
    // measured to change nothing — see `inference::thermal`. Dormant wherever
    // no sensor is readable, which is most containers and VMs.
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(15));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            // The read touches a few small sysfs files; keep it off the runtime
            // in case a flaky sensor driver blocks.
            let _ = tokio::task::spawn_blocking(crate::inference::thermal::poll_and_report).await;
        }
    });

    // Spawn a reader task that pushes framed IPC messages onto an mpsc.
    // Decoupling read-from-socket from the main select! loop keeps frame
    // alignment safe under cancellation (recv_framed itself is not cancel-safe).
    //
    // Capacity 64: the admit-coalescing drain loop pulls up to 16 messages
    // per tick, but a single decode tick can be 100-500ms on CPU for a 7B
    // model. During that window the reader task may receive a burst of
    // Generate requests + cross-node PrefixFetchResult replies. Capacity 16
    // would block the reader on send, delaying the fast-path PrefixFetchResult
    // short-circuit (see lines below) and degrading cross-node prefix-cache
    // efficiency. 64 gives 4× the drain budget without bounding socket-buffer
    // backpressure too loosely.
    let (ipc_tx, mut ipc_rx) = mpsc::channel::<(DaemonMsg, Vec<u8>)>(64);
    let reader_pending = pending_fetches.clone();
    let reader_cancelled = cancelled.clone();
    let reader_task = tokio::spawn(async move {
        loop {
            match recv_daemon(&mut reader).await {
                Ok((msg, payload)) => {
                    // Short-circuit cancellations for the same reason as
                    // prefix-fetch results below: `handle_generate` owns the
                    // main loop while it decodes, so a cancel queued on
                    // `ipc_rx` would only be seen after the work it was meant
                    // to stop had already been done.
                    if let DaemonMsg::CancelRequest { request_id } = msg {
                        reader_cancelled.insert(request_id, std::time::Instant::now());
                        tracing::debug!(%request_id, "model-worker: request cancelled by daemon");
                        continue;
                    }
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
                &ngram_cfg,
                &options,
                &mut slot_table,
                &pending_fetches,
                &cancelled,
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
                            &ngram_cfg,
                            &options,
                            &mut slot_table,
                            &pending_fetches,
                            &cancelled,
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

        // Drop any decoding slot the daemon has abandoned before spending
        // another tick on it, and free its KV. No reply is sent: the daemon
        // already unregistered the response channel, so anything we emit would
        // be discarded by the reader actor.
        if !cancelled.is_empty() {
            if !slot_table.is_empty() {
                for slot in slot_table.take_matching(|s| cancelled.contains_key(&s.request_id)) {
                    tracing::info!(
                        request_id = %slot.request_id,
                        generated = slot.generated_count(),
                        "model-worker: dropping cancelled decode slot"
                    );
                    cancelled.remove(&slot.request_id);
                    kv_store.clear_request(&slot.model_key, &slot.req_id_str);
                }
            }
            // Sweep cancels that never matched a request (the request had
            // already finished when the cancel arrived).
            cancelled.retain(|_, at| at.elapsed().as_secs() < CANCEL_RETENTION_SECS);
        }

        // What the KV cache is holding, in bytes.
        //
        // Process RSS cannot answer this: the reservation is zeroed pages the
        // OS backs lazily, so a 4x change in reserved bytes moved RSS ~5% when
        // measured, and in both directions. See `KvCacheStore::occupancy`.
        // Cheap enough at this cadence — it walks live entries only, and an
        // idle worker blocks on IPC so it does not run at all.
        if last_kv_report.elapsed().as_secs() >= KV_OCCUPANCY_REPORT_SECS {
            last_kv_report = std::time::Instant::now();
            let occ = kv_store.occupancy();
            if occ.entries > 0 {
                tracing::debug!(
                    entries = occ.entries,
                    allocated_mb = occ.allocated_bytes / 1_000_000,
                    used_mb = occ.used_bytes / 1_000_000,
                    utilisation_pct = (occ.utilisation() * 100.0).round() as u64,
                    tokens = occ.tokens,
                    "DIAG: worker KV-cache occupancy"
                );
            }
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
                &mut prefill_pacer,
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
    let message = err.to_string();
    let fatal = crate::inference::worker_ipc::worker_error_is_fatal(&message);
    if fatal {
        // The daemon will kill us on receipt. Say so in our own log too — an
        // operator reading the worker log should not have to infer the
        // subsequent respawn from the daemon side.
        tracing::error!(
            request_id = %request_id,
            error = %message,
            "model-worker: fatal device error — daemon will recycle this worker"
        );
    }
    let _ = send_worker(
        writer,
        &WorkerMsg::Error {
            request_id,
            message,
            fatal,
        },
        &[],
    )
    .await;
}

/// Device placement for this worker process, decided once at startup from
/// `--gpu-layers` and never changed.
///
/// A process-global rather than another parameter: every model this worker
/// loads goes on the same device, and the alternative is threading an
/// immutable `bool` through six call layers alongside `shard_window`. The
/// daemon respawns the worker to change placement (see
/// `ModelProcessPool::cpu_pinned_models`), so there is no in-process
/// transition to reason about.
static WORKER_FORCE_CPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Apply `--gpu-layers` to this process. Called once from `run_worker`.
fn set_worker_force_cpu(gpu_layers: i32) {
    let force_cpu = crate::daemon::shard_loader::force_cpu_for(gpu_layers);
    WORKER_FORCE_CPU.store(force_cpu, std::sync::atomic::Ordering::Relaxed);
    if force_cpu {
        tracing::info!("model-worker: gpu_layers = 0 — loading models on CPU");
    } else if gpu_layers > 0 {
        // A positive count is now a real instruction, not something to warn
        // about and ignore: the first `gpu_layers` layers of this worker's
        // window go on the card and the rest on the processor. Until hybrid
        // placement existed this could only be honoured as "all of them",
        // which is why it used to warn.
        crate::inference::split::GPU_LAYER_LIMIT
            .store(gpu_layers as usize, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            gpu_layers,
            "model-worker: placing the first {gpu_layers} layers of this window on the \
             graphics card and the rest on the processor"
        );
    }
}

/// Should models load on the CPU regardless of GPU availability?
fn worker_force_cpu() -> bool {
    WORKER_FORCE_CPU.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whole-device GPU memory in use, in MB, or `None` when there is no GPU to ask.
///
/// Whole-device rather than per-process because WSL reports `[N/A]` for
/// `--query-compute-apps` memory, so process attribution is simply unavailable
/// there. Sound enough to bracket a single load in a process that does one at a
/// time; not sound as a live admission input.
fn current_vram_used_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// GPU memory this load appears to have cost, in MB.
///
/// `None` when there is no GPU, when either sample failed, or when the device
/// went DOWN across the load — that last case means another process freed more
/// than we allocated, so the delta says nothing about us and a negative number
/// dressed up as zero would be worse than admitting we do not know.
fn measured_vram_delta_mb(before: Option<u64>) -> Option<u64> {
    let before = before?;
    let after = current_vram_used_mb()?;
    after.checked_sub(before)
}

/// The request id a cancellation would apply to, for message kinds worth
/// checking before starting work.
///
/// `BatchForward` is deliberately excluded: it carries N request ids and the
/// batch is dispatched as one fused matmul, so skipping it wholesale on one
/// cancelled member would drop work the other members still want. The waste
/// there is bounded by the batch, and the per-slot decode path already handles
/// cancellation at a finer grain.
fn cancelled_request_id(msg: &DaemonMsg) -> Option<Uuid> {
    match msg {
        DaemonMsg::Forward(f) => Some(f.request_id),
        DaemonMsg::Generate(g) => Some(g.request_id),
        _ => None,
    }
}

/// A `Generate` is a WHOLE-model operation: the worker is handed the raw prompt
/// and must return sampled tokens, so it needs the embedding table at the front
/// and the output head at the back. A partial layer range has neither, and the
/// failure it produces is unreadable — without the embedding table the prompt's
/// token ids are fed straight into the first block
/// (`attn_norm: shape mismatch in rms-norm [1, 128] [3072]`, i.e.
/// `[batch, seq_len]` ids where hidden states belong), and without the output
/// head the sampler is handed hidden states
/// (`unexpected rank, expected: 1, got: 2 ([20, 3072])`). Both were reported as
/// separate crashes before they were recognised as one caller mistake.
///
/// `Forward` is the operation for a partial range; this guard is deliberately
/// NOT in `ensure_model_loaded`, which both kinds share.
fn ensure_whole_model_for_generate(
    models: &HashMap<(usize, usize, usize, usize), SplitModel>,
    key: (usize, usize, usize, usize),
    model_id: &crate::types::ModelId,
) -> Result<(), SwarmError> {
    let Some(model) = models.get(&key) else {
        return Ok(()); // caller reports the missing model itself
    };
    if model.is_first() && model.is_last() {
        return Ok(());
    }
    let missing = match (model.is_first(), model.is_last()) {
        (false, false) => "neither the start nor the end",
        (false, true) => "not the start",
        (true, false) => "not the end",
        (true, true) => unreachable!("returned above"),
    };
    Err(SwarmError::ServiceUnavailable(format!(
        "this node holds layers {}..{} of {model_id}, which is {missing} of the model — \
         a full generation needs every layer, so this request must go through the pipeline",
        key.0, key.1
    )))
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

    // Sampled before the load so the delta is attributable to it.
    let vram_before_mb = current_vram_used_mb();

    // Try loading the split model from available sources
    let gguf_path = model_dir.join("model.gguf");
    let source_path_file = model_dir.join("source_path");

    let mut model = if gguf_path.exists() {
        tracing::info!(
            model = %model_id,
            layers = format!("[{layer_start}..{layer_end})"),
            "model-worker: Loading from reconstructed GGUF"
        );
        SplitModel::load_from_gguf(
            &gguf_path,
            layer_start,
            layer_end,
            is_first,
            is_last,
            worker_force_cpu(),
        )?
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
                        worker_force_cpu(),
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
                        force_cpu: worker_force_cpu(),
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
            force_cpu: worker_force_cpu(),
        })?
    };

    // Apply TP weight splitting if this is a TP variant (tp_size > 1)
    if tp_size > 1 {
        model.pre_split_for_tp(tp_rank, tp_size)?;
    }

    // Report what this load ACTUALLY cost on the GPU, not what we guessed.
    //
    // This is the footprint AT LOAD, which is NOT the steady state: candle
    // allocates the KV cache lazily on the first append, i.e. during the first
    // forward. Measured on phi-3.5-mini-q4, load was 2772 MB and steady state
    // 6037 MB — the KV cache is over half the total. Do not compare this figure
    // to an admission estimate without adding the KV term.
    //
    // Admission control needs a number it can trust, and the estimate is built
    // from GGUF geometry with a provisional constant for driver overhead. On
    // WSL, `nvidia-smi --query-compute-apps` reports `[N/A]` for per-process
    // memory, so whole-device sampling around the load is the only attribution
    // available — and it is only sound because this process does one load at a
    // time. Treat a sample taken while another process is allocating as noise;
    // that is why this is logged rather than fed straight into a decision.
    let vram_after_load_mb = measured_vram_delta_mb(vram_before_mb);
    tracing::info!(
        model = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        tp_rank,
        tp_size,
        device = ?model.device(),
        vram_after_load_mb,
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
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
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
            // Logits [1, vocab] → sample + EOS check. Pass generated_ids
            // (populated by the daemon coordinator on the final segment
            // when penalties are non-zero) so frequency_penalty /
            // presence_penalty are honored on the distributed batched
            // path.
            let token_id =
                split::sample_token_with_params_history(output_t, &r.sampling, &r.generated_ids)
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
                matched_stop_sequence: None,
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
                matched_stop_sequence: None,
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
            if activation_bytes.is_empty() || !activation_bytes.len().is_multiple_of(8) {
                return Err(SwarmError::Internal(format!(
                    "Decode step activation payload must be a non-empty multiple of 8 bytes (got {})",
                    activation_bytes.len()
                )));
            }
            let token_ids: Vec<i64> = activation_bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
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
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect();
                        // SEC: reject NaN/Inf from peer-supplied vision embeddings.
                        // Without this, a poisoned tensor propagates through the KV
                        // cache for the session, contaminating every subsequent
                        // decode step. Same convention as `spec_logits` decode and
                        // AllReduce sums.
                        if f32_values.iter().any(|v| !v.is_finite()) {
                            tracing::warn!(
                                request_id = %fwd.request_id,
                                "Vision embeddings contain NaN/Inf — dropping vision tensor"
                            );
                            return Err(SwarmError::Validation(
                                "Vision embeddings contain non-finite values".into(),
                            ));
                        }
                        // SEC: checked_mul guards against integer overflow
                        // for adversarial shape headers. On 32-bit platforms
                        // num_tokens * hidden_dim can wrap if both are near
                        // u32::MAX, bypassing the length-equality check and
                        // letting Tensor::from_vec receive a mismatched len.
                        match num_tokens.checked_mul(hidden_dim) {
                            None => {
                                tracing::warn!(
                                    num_tokens,
                                    hidden_dim,
                                    "Vision embedding shape multiply overflow — dropping tensor"
                                );
                                None
                            }
                            Some(expected_len) if f32_values.len() != expected_len => {
                                tracing::warn!(
                                    expected = expected_len,
                                    actual = f32_values.len(),
                                    "Vision embedding shape mismatch"
                                );
                                None
                            }
                            Some(_) => candle_core::Tensor::from_vec(
                                f32_values,
                                &[num_tokens, hidden_dim],
                                &candle_core::Device::Cpu,
                            )
                            .ok(),
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

    // Load LoRA adapter if requested.
    //
    // SEC: `adapter_id` rides peer-controlled `LayerForward.adapter_id`. A naive
    // `data_dir.join("adapters").join(adapter_id)` lets a malicious peer escape
    // the adapters/ directory via `..`, absolute paths, or NUL bytes. Reject
    // anything that isn't a single, plain filename component before joining.
    let lora_adapter = if let Some(ref adapter_id) = fwd.adapter_id {
        let is_safe_id = !adapter_id.is_empty()
            && !adapter_id.contains('/')
            && !adapter_id.contains('\\')
            && !adapter_id.contains('\0')
            && adapter_id != "."
            && adapter_id != ".."
            && !adapter_id.starts_with('.')
            && std::path::Path::new(adapter_id).components().count() == 1;
        if !is_safe_id {
            tracing::warn!(
                adapter_id,
                "Rejecting LoRA adapter_id with unsafe path components"
            );
            return Err(SwarmError::Validation(
                "adapter_id must be a single safe filename component".into(),
            ));
        }
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
                // output_t shape is [1, seq_len, vocab_size]. Flatten + dtype-cast +
                // to_vec1 in one shot (rather than per-position tensor slicing) so a
                // 151K-vocab model at γ=4 doesn't pay seq_len intermediate tensor
                // views + casts. Then split into per-position rows by vocab_size.
                let dims = output_t.dims();
                if dims.len() != 3 {
                    return Err(format!("spec verify unexpected shape: {dims:?}"));
                }
                let seq_len = dims[1];
                let vocab_size = dims[2];
                let flat: Vec<f32> = output_t
                    .flatten_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::F32))
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| format!("spec verify flatten/to_vec1: {e}"))?;
                if flat.len() != seq_len * vocab_size {
                    return Err(format!(
                        "spec verify flat len {} ≠ seq_len({seq_len}) * vocab({vocab_size})",
                        flat.len()
                    ));
                }
                let spec_logits: Vec<Vec<f32>> =
                    flat.chunks_exact(vocab_size).map(<[f32]>::to_vec).collect();
                return Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: vec![],
                    sealed_token_ids: None,
                    spec_logits,
                    matched_stop_sequence: None,
                    token_logprobs: Vec::new(),
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
                let (token_id, token_logprob) = if fwd.sampling.logprobs {
                    crate::inference::tensor_util::sample_token_with_logprob_history(
                        &output,
                        &fwd.sampling,
                        &fwd.generated_ids,
                    )
                    .map_err(|e| format!("Sample: {e}"))?
                } else {
                    let tid = split::sample_token_with_params_history(
                        &output,
                        &fwd.sampling,
                        &fwd.generated_ids,
                    )
                    .map_err(|e| format!("Sample: {e}"))?;
                    (tid, None)
                };
                let eos_tokens = model.eos_tokens();
                let finish = if eos_tokens.contains(&token_id) {
                    Some(NetworkFinishReason::Stop)
                } else {
                    None
                };
                // When the request asked for logprobs, package the per-token
                // entry so the coordinator's `collected_logprobs` accumulates
                // it and the final `InferenceOutput.token_logprobs` is
                // populated for the distributed-pipeline path too.
                let token_logprobs = match token_logprob {
                    Some(lp) => vec![swarmllm_types::TokenLogProbEntry {
                        // Per-token text decoding happens at the coordinator
                        // (which holds the tokenizer); leave empty here and
                        // let the API layer fill it if needed. OpenAI clients
                        // expect the `token` string but distributed-pipeline
                        // logprobs are an opt-in compatibility feature.
                        token: String::new(),
                        logprob: lp,
                        top_logprobs: Vec::new(),
                    }],
                    None => Vec::new(),
                };
                Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![token_id],
                    finish_reason: finish,
                    activations: vec![],
                    sealed_token_ids: None,
                    spec_logits: Vec::new(),
                    matched_stop_sequence: None,
                    token_logprobs,
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
                    matched_stop_sequence: None,
                    token_logprobs: Vec::new(),
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
        logprobs: if result.token_logprobs.is_empty() {
            None
        } else {
            Some(result.token_logprobs)
        },
        matched_stop_sequence: result.matched_stop_sequence,
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

/// Clamp a hydrated prefix to leave one token for the forward, and truncate the
/// KV cache so it holds exactly that many positions.
///
/// Hydration writes however many tokens the cached snapshot covers. The callers
/// then clamped only the *number they carry forward*, leaving the cache holding
/// more positions than the clamp claimed. Nothing crashed, because a 1-token
/// forward takes the `seq_len == 1` path where no mask is built — but
/// `forward_inner_impl` reads `kv_offset` from the **cache**, not from
/// `index_pos`, so the final prompt token ended up in the cache twice and the
/// model attended to a duplicate.
///
/// **Only the cross-node path can reach this.** `PrefixCache::lookup` clamps its
/// own result to `prompt_tokens - 1` and filters out any entry longer than that,
/// so a LOCAL hit can never over-fill the cache and the clamp below is inert for
/// it. `hydrate_request_from_bytes` takes a snapshot sent by a PEER and applies
/// no such bound — the peer returns whatever blocks it matched — so that is the
/// path where the cache can end up holding the whole prompt.
///
/// Established 2026-07-30 by a tester who deduced it from black-box behaviour
/// (`matched_tokens` never exceeding half a 128-token prompt across three
/// repeats) without source access, and asked which cache path was meant.
///
/// `KvCacheStore::truncate_request_to` already existed for the speculative
/// partial-accept fixup; this is the same need.
fn reconcile_hydrated_prefix(
    kv_store: &Arc<KvCacheStore>,
    model_key: &str,
    req_id_str: &str,
    hydrated: usize,
    prompt_tokens: usize,
) -> usize {
    let clamped = hydrated.min(prompt_tokens.saturating_sub(1));
    if clamped < hydrated {
        if let Err(e) = kv_store.truncate_request_to(model_key, req_id_str, clamped) {
            // Truncation failed, so the cache still holds `hydrated` positions
            // and we cannot describe it honestly. Drop it and prefill in full:
            // slower, but correct.
            tracing::warn!(
                error = %e, hydrated, clamped,
                "prefix hydrate: could not truncate KV to the clamped length — \
                 discarding the hydrated cache and prefilling the whole prompt"
            );
            kv_store.clear_request(model_key, req_id_str);
            return 0;
        }
        tracing::debug!(
            hydrated,
            clamped,
            prompt_tokens,
            "prefix hydrate: truncated KV cache to leave a token for the forward"
        );
    }
    clamped
}

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
            // Same reconciliation as the local-hit path: the clamp must move
            // the cache, not just the number we report.
            // `n` is what the cache actually holds; reconcile does the clamp
            // (to `prompt_tokens - 1`, the same bound `usable` applies) AND
            // truncates the cache to match, so pass `n` through unmodified.
            let n = reconcile_hydrated_prefix(
                kv_store,
                model.kv_model_key(),
                req_id_str,
                n,
                prompt_tokens,
            );
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

/// May this request use the local draft-free speculator?
///
/// The admission gate and the decode loop BOTH ask this. They used to be two
/// conditions written in two places, which is the shape this codebase gets
/// caught by most often: the gate would divert a request to the sequential loop
/// and the loop would then decline to speculate it, so it would lose batching
/// and gain nothing.
///
/// **Temperature is deliberately NOT one of the clauses**, and it used to be.
/// The reasoning for excluding sampled requests was that a draft is kept by
/// comparing it against what the sampler returned, which "is a verification
/// only while the sampler is deterministic". That is wrong, and it mattered:
/// the default temperature is 0.7 on the OpenAI surface and 1.0 on the
/// Anthropic one, so the gate left the feature inert for essentially all real
/// traffic — including Claude Code and MCP tool use, the workload
/// `ngram_lookup`'s own documentation names as the reason it exists.
///
/// Speculative sampling with a DETERMINISTIC draft `x` (an n-gram guess carries
/// no distribution, so `q = δ_x`) accepts with probability
/// `min(1, p(x)/q(x)) = p(x)` and otherwise draws from the residual
/// `norm((p − q)₊)`, i.e. `p` with `x` removed and renormalised. "Draw `t ~ p`;
/// keep the draft iff `t == x`" has exactly those two branches — it accepts
/// with probability `p(x)`, and conditioned on rejection `t` is distributed as
/// `p` restricted to `≠ x`. Sampling each position with the real sampler and
/// keeping a draft only on a match therefore IS the rejection rule, at any
/// temperature, and needs no separate implementation.
/// `accepting_only_on_a_match_preserves_the_sampled_distribution` pins that
/// empirically, with a control that fails if the metric could not see a bias.
///
/// Acceptance is naturally lower when sampling is diffuse, which is correct
/// rather than unfortunate — and the case speculation exists for, copying text
/// already in the context, is exactly where the distribution is sharply peaked
/// and `p(draft)` is near 1. `SpecBackoff` handles the rest.
///
/// The remaining clauses are correctness conditions, not tuning choices:
///
/// * `!logprobs` — accepted tokens come out of a verify forward and this path
///   does not carry their per-token logprobs back. Better to decline than to
///   answer `null` where a client asked for numbers.
/// * SWIFT off — it is already speculating; two schemes drafting for one
///   request would fight.
/// * a non-zero draft width, which is how `inference.ngram_lookup_enabled`
///   arrives here.
pub(crate) fn ngram_spec_eligible(
    sampling: &crate::types::SamplingParams,
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    swift_cfg: &SwiftConfig,
) -> bool {
    ngram_cfg.num_pred_tokens > 0
        && ngram_cfg.max_ngram_size >= ngram_cfg.min_ngram_size
        && !sampling.logprobs
        && !(swift_cfg.enabled && sampling.temperature == 0.0)
}

/// Recent tokens-per-round the local speculator has been achieving, ×100, or 0
/// for "not measured yet". One value for the worker, which serves one model set.
static SPEC_PAYOFF_X100: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Tokens per round below which diverting a request out of the batch to
/// speculate is not worth what it gives up. A round that lands one token is
/// doing exactly what a plain decode step does.
const SPEC_PAYOFF_WORTH_DIVERTING_X100: u32 = 150;

/// Record what a finished speculative generation achieved, so the next
/// admission decision has something better than a guess. EWMA at 1/4 weight —
/// enough to follow a workload change within a few requests, slow enough that
/// one unusual reply does not flip the policy.
fn record_spec_payoff(tokens_per_round: f64) {
    use std::sync::atomic::Ordering;
    let prev = SPEC_PAYOFF_X100.load(Ordering::Relaxed);
    SPEC_PAYOFF_X100.store(blend_spec_payoff(prev, tokens_per_round), Ordering::Relaxed);
}

/// The EWMA step, separated from the global it updates so the arithmetic can be
/// tested without a shared static — tests that mutate one only pass under
/// `--test-threads=1`, and this project runs them in parallel.
///
/// Never returns 0 once a sample has been taken: 0 means "not measured yet", and
/// letting a measured-as-poor workload look unmeasured would quietly re-enable
/// the diversion it had just been measured out of.
fn blend_spec_payoff(prev_x100: u32, tokens_per_round: f64) -> u32 {
    let sample = (tokens_per_round * 100.0).clamp(0.0, 100_000.0) as u32;
    let next = if prev_x100 == 0 {
        sample
    } else {
        (prev_x100 * 3 + sample) / 4
    };
    next.max(1)
}

/// Is speculation paying well enough to justify taking a request OUT of the
/// batched decode path to get it?
///
/// **This exists because the trade is not free, and measuring the workload
/// speculation CANNOT help is what showed it.** A diverted request runs on the
/// sequential loop, which owns the worker for its whole duration, so every other
/// request waits rather than sharing a batched decode tick. When speculation is
/// landing ~9 tokens a round that is still a clear win — measured on an RTX
/// 3070, 8 concurrent copy-heavy requests finished in 3.76 s against 5.52 s
/// batched. When it is landing ~1, the node gives up batching and gets nothing:
/// the same 8 requests on an open-ended prompt took 29.07 s against 12.48 s,
/// and aggregate throughput fell from 77 to 33 tok/s.
///
/// The earlier justification for diverting unconditionally cited this project's
/// own measurement that batching is worth ~3% (gotcha #348). That figure is from
/// a PROCESSOR. On a graphics card batching amortises kernel launches across
/// requests — the same launch-bound property speculation exploits — so it is
/// worth far more there, and a CPU-derived number had no business being applied
/// to it.
///
/// Unknown means "let one request find out": the first generation after start
/// diverts, measures, and the answer steers everything after it.
fn spec_payoff_justifies_diverting() -> bool {
    payoff_justifies_diverting(SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed))
}

/// The decision itself. Pure, for the same reason `blend_spec_payoff` is.
fn payoff_justifies_diverting(seen_x100: u32) -> bool {
    seen_x100 == 0 || seen_x100 >= SPEC_PAYOFF_WORTH_DIVERTING_X100
}

/// Whether the local speculator should draft this round.
///
/// A draft that is found and then rejected is not free: the round still pays a
/// multi-token forward to return the one token a plain step would have
/// returned. On a reply with nothing to copy that is a standing tax — measured
/// at ~5% on an open-ended prompt, where 70 rounds produced 80 tokens and only
/// 10 drafts were accepted.
///
/// So a run of useless rounds pauses drafting, doubling the pause each time it
/// recurs, and ANY acceptance clears it entirely. The lookup itself is a
/// hash-table probe and is not what costs; the forward it provokes is.
///
/// Extracted rather than left inline because it is a policy with an arithmetic
/// that is easy to get subtly wrong — and because a decision made inside a hot
/// loop is otherwise only observable by timing a whole request.
#[derive(Debug, Default)]
struct SpecBackoff {
    miss_streak: u32,
    pause_left: u32,
    pause_len: u32,
    paused_rounds: u64,
}

impl SpecBackoff {
    /// Consecutive drafted-and-rejected rounds before pausing. Small, because
    /// each one buys nothing and costs a wider forward.
    const MISSES_BEFORE_PAUSE: u32 = 3;
    /// Longest pause, in rounds. Bounded so a reply that becomes copy-heavy
    /// partway through — a preamble and then the model quoting the prompt back,
    /// which is the common agentic shape — resumes speculating promptly rather
    /// than staying switched off for the rest of a long generation.
    const MAX_PAUSE_ROUNDS: u32 = 64;

    /// Call once per round. Consumes one round of any active pause.
    fn should_draft(&mut self) -> bool {
        if self.pause_left == 0 {
            return true;
        }
        self.pause_left -= 1;
        self.paused_rounds += 1;
        false
    }

    /// Report what a drafted round achieved. Only call when a draft was made —
    /// a paused round proves nothing about whether drafting would have worked.
    fn record(&mut self, accepted_any: bool) {
        if accepted_any {
            self.miss_streak = 0;
            self.pause_len = 0;
            return;
        }
        self.miss_streak += 1;
        if self.miss_streak >= Self::MISSES_BEFORE_PAUSE {
            self.miss_streak = 0;
            self.pause_len = (self.pause_len.max(1) * 2).min(Self::MAX_PAUSE_ROUNDS);
            self.pause_left = self.pause_len;
        }
    }
}

/// Window of recent generation the local speculator searches for self-matches,
/// matching the distributed path's choice in `pipeline::speculative`.
const NGRAM_RECENT_GEN_WINDOW: usize = 500;

/// One draft-free speculative round on the local decode path.
///
/// Drafts from an n-gram match against the prompt and the generation tail, then
/// verifies the whole draft in ONE forward. There is no draft model: the guess
/// is that text already in the context will recur, which is what SwarmLLM's own
/// workload does constantly (tool schemas, code, retrieved passages).
///
/// Returns the tokens produced and how many KV positions were committed. The
/// two are deliberately NOT the same number: a round commits `next_token` plus
/// every accepted draft token, and the last token it produces is not in the
/// cache yet. That is the same invariant the plain loop keeps, where one
/// forward commits one position and yields one not-yet-cached token — which is
/// why the caller can drain the extras without forwarding again.
///
/// **Acceptance goes through the real sampler, not a private argmax.** Every
/// position is sampled by the same `sample_token_with_logprob_history` the plain
/// loop uses, with the history it would have had at that point, and a draft
/// token is kept only when it equals what that call returned. Re-deriving
/// "argmax" here would have been a second sampler to keep in step with the
/// first — this codebase's most repeated defect.
///
/// **What that does and does not guarantee.** In exact arithmetic the result is
/// the sequence greedy decoding would have produced. In floating point it is
/// *almost* that: a verify forward computes its logits with a different matmul
/// shape than a one-token forward, so the reduction order differs and a
/// near-tie between two tokens can land the other way. Measured on
/// llama-3.2-3b: an input-grounded reply of 53 tokens came out byte-identical,
/// while an open-ended one diverged at one token ("which refers to" against
/// "which means") and then, as any single token flip does, went its own way.
/// Both runs are deterministic — repeating either reproduces it exactly — so
/// this is reassociation, not a race.
///
/// That is inherent to speculative decoding on real hardware rather than a
/// defect here, but it must not be described as bit-identical, because someone
/// will one day diff two replies and need to know whether they have found a
/// bug. They have not; they have found fp addition being non-associative.
#[allow(clippy::too_many_arguments)]
fn ngram_spec_round(
    model: &mut crate::inference::split::SplitModel,
    kv_store: &crate::inference::split::KvCacheStore,
    model_key: &str,
    req_id_str: &str,
    index_pos: usize,
    next_token: u32,
    ctx: &[u32],
    prompt_len: usize,
    generated: &[u32],
    sampling: &crate::types::SamplingParams,
    cfg: crate::inference::ngram_lookup::NgramLookupConfig,
    draft_allowed: bool,
    force_attn: bool,
) -> Result<(Vec<u32>, usize), SwarmError> {
    let (draft, _source) = if draft_allowed {
        crate::inference::ngram_lookup::cascade_find_candidate(
            ctx,
            prompt_len,
            NGRAM_RECENT_GEN_WINDOW,
            cfg,
        )
    } else {
        (
            Vec::new(),
            crate::inference::ngram_lookup::NgramHitSource::None,
        )
    };

    // No match: do exactly what the plain loop does. A miss must cost nothing
    // beyond the lookup itself, or speculation becomes a tax on the workloads
    // it cannot help.
    if draft.is_empty() {
        let input = model.token_tensor(next_token)?;
        let logits = tokio::task::block_in_place(|| {
            let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
            model.forward(&input, index_pos, kv_store, req_id_str)
        })?;
        let (tok, _) = crate::inference::tensor_util::sample_token_with_logprob_history(
            &logits, sampling, generated,
        )?;
        return Ok((vec![tok], 1));
    }

    let ids: Vec<u32> = std::iter::once(next_token)
        .chain(draft.iter().copied())
        .collect();
    let input = model.tensor_from_ids(&ids)?;
    let all = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
        model.forward_verify_all_positions(&input, index_pos, kv_store, req_id_str)
    })?;

    use candle_core::IndexOp;
    let mut hist: Vec<u32> = generated.to_vec();
    let mut produced: Vec<u32> = Vec::with_capacity(draft.len() + 1);
    let mut accepted = 0usize;
    for (i, _) in ids.iter().enumerate() {
        let row = all
            .i((0, i))
            .and_then(|t| t.unsqueeze(0))
            .map_err(SwarmError::internal)?;
        let (tok, _) = crate::inference::tensor_util::sample_token_with_logprob_history(
            &row, sampling, &hist,
        )?;
        produced.push(tok);
        // The last position has no draft to check — its token is the round's
        // bonus and always ends it.
        if i < draft.len() && tok == draft[i] {
            accepted += 1;
            hist.push(tok);
        } else {
            break;
        }
    }

    // Drop the cache entries written for drafts that were not accepted. Without
    // this the next forward would attend over positions holding tokens the
    // reply does not contain — a silent corruption, not an error.
    let committed = 1 + accepted;
    kv_store.truncate_request_to(model_key, req_id_str, index_pos + committed)?;
    Ok((produced, committed))
}

/// KV positions reserved for the reply when a prompt is admitted and when its
/// snapshot is sized: one growth quantum, the least a decode needs to claim.
const REPLY_RESERVE_POSITIONS: usize = crate::inference::layers::KV_CACHE_GROWTH_TOKENS;

/// `SWARMLLM_KV_PREFIX_CHARGE=0` turns off the whole-prompt admission below,
/// so the two arms can be compared inside ONE binary. Read once.
fn prefix_cache_charged() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_KV_PREFIX_CHARGE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// How many positions of a finished prompt to snapshot into the prefix cache
/// so the copy fits BESIDE the request's live cache (gotcha #440). Evicts
/// older cached prompts first. `usize::MAX` when the budget is unknown or
/// charging is switched off.
fn snapshot_positions_that_fit(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    prompt_tokens: usize,
) -> usize {
    if !prefix_cache_charged() {
        return usize::MAX;
    }
    let allocated = kv_store.occupancy().allocated_bytes;
    let cached = prefix_cache.bytes_total() as u64;
    // Against the card as it stands, not the budget predicted at load: a
    // snapshot is a device allocation the size of the prompt, taken on a card
    // that may hold tenants the load-time figure never saw.
    let (Some(budget), per_token) = model.kv_budget_now(allocated, cached) else {
        return usize::MAX;
    };
    if per_token == 0 {
        return usize::MAX;
    }
    // The live figure carries one growth quantum of reserve for the reply
    // about to be decoded: a snapshot sized to the last byte left this
    // request unable to claim its next quantum mid-reply (measured: the
    // reply cut at token 32, the quantum boundary after a 7649-token prompt).
    let live = allocated.saturating_add(per_token.saturating_mul(REPLY_RESERVE_POSITIONS as u64));
    let plan = crate::inference::split::kv_budget::plan_snapshot(
        budget,
        live,
        cached,
        per_token,
        prompt_tokens,
    );
    if plan.evict_bytes > 0 {
        prefix_cache.release(plan.evict_bytes as usize);
    }
    if plan.positions < prompt_tokens {
        tracing::info!(
            prompt_tokens,
            keeping = plan.positions,
            live_mb = live / (1024 * 1024),
            cached_mb = cached / (1024 * 1024),
            budget_mb = budget / (1024 * 1024),
            "DIAG: prefix-cache snapshot cut to the room beside the live cache"
        );
    }
    plan.positions
}

/// Make sure the whole prompt's KV can live on the device BEFORE prefill
/// starts — evicting cached prompts first, refusing second (gotcha #440).
///
/// Called after the prefix-cache lookup and before hydration, at both entry
/// points (sequential and batched), because hydration is itself an
/// allocation of the matched prefix. The prefix cache's snapshots are the
/// same device memory as the live caches and were charged nowhere; on an
/// 8 GB card the second long prompt's cache landed in host-backed memory
/// and decoded at 3-5 tok/s where an empty card did 19-33. A refusal here
/// is a 503 the coordinator can route elsewhere, at token 0, with nothing
/// half-built.
fn ensure_room_for_prompt(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    request_id: &str,
    prompt_tokens: usize,
) -> Result<(), SwarmError> {
    if !prefix_cache_charged() {
        return Ok(());
    }
    let occupancy = kv_store.occupancy();
    let live = occupancy.allocated_bytes;
    let cached = prefix_cache.bytes_total() as u64;
    // Nothing of THIS request is in the store yet, so anything live belongs to
    // an earlier one — a request still decoding, or one whose cache outlived
    // its reply. The store's figure was seen to wander by ~1 GB across
    // identical sequential prompts (2856 → 4200 MB, 2026-09-02) and this is
    // the line that says whose bytes those are.
    if live > 0 {
        tracing::debug!(
            request_id,
            store_entries = occupancy.entries,
            live_mb = live / (1024 * 1024),
            live_tokens = occupancy.tokens,
            "DIAG: KV admission — the store already holds caches from earlier requests"
        );
    }
    // The budget as the card can honour it NOW, not as predicted at load. The
    // load-time figure cannot see a tenant that arrived after it — another
    // model's worker, the full build's second CUDA context — and on the
    // released v0.3.149 it admitted a prompt into ~300 MB of real room, which
    // WSL2 served from host memory at 1.95 tok/s rather than refusing.
    let (Some(budget), per_token) = model.kv_budget_now(live, cached) else {
        return Ok(());
    };
    if per_token == 0 {
        return Ok(());
    }
    let load_time_budget = model.kv_budget().0.unwrap_or(budget);
    // The prompt plus one growth quantum for the reply, so an admitted
    // request can at least begin decoding without meeting the per-chunk
    // guard at its first quantum boundary. A very long reply may still meet
    // it later, which is the lazy charge that guard exists for.
    let positions =
        crate::inference::layers::kv_cache_reservation(prompt_tokens) + REPLY_RESERVE_POSITIONS;
    let mb = |b: u64| b / (1024 * 1024);
    use crate::inference::split::kv_budget::{admit_prompt, PromptAdmission};
    let verdict = admit_prompt(budget, live, cached, per_token, positions);
    let short_by = match verdict {
        PromptAdmission::Fits => return Ok(()),
        PromptAdmission::EvictBytes(needed) => {
            let freed = prefix_cache.release(needed as usize) as u64;
            // Charging is on here (checked above), so keep the guard's figure current.
            kv_store.set_external_reserved(prefix_cache.bytes_total() as u64);
            tracing::info!(
                request_id,
                prompt_tokens,
                positions,
                live_mb = mb(live),
                cached_mb = mb(cached),
                budget_mb = mb(budget),
                load_time_budget_mb = mb(load_time_budget),
                needed_mb = mb(needed),
                freed_mb = mb(freed),
                "DIAG: KV admission — evicted cached prompts so this prompt's cache fits on the device"
            );
            if freed >= needed {
                return Ok(());
            }
            needed - freed
        }
        PromptAdmission::Refuse { short_by } => short_by,
    };
    tracing::warn!(
        request_id,
        prompt_tokens,
        positions,
        live_mb = mb(live),
        cached_mb = mb(cached),
        budget_mb = mb(budget),
        load_time_budget_mb = mb(load_time_budget),
        short_by_mb = mb(short_by),
        "DIAG: KV admission — refusing this prompt before prefill: it would not fit on the device"
    );
    Err(SwarmError::ServiceUnavailable(format!(
        "Not enough free memory on this node for a {prompt_tokens}-token prompt ({} MB of KV \
         cache in use, budget {} MB, short by {} MB). Shorter conversations still work; free \
         memory on this node (close other programs, or raise its memory budget) to raise this.",
        mb(live),
        mb(budget),
        mb(short_by),
    )))
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
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    force_standard_attn: bool,
    max_seq_len_override: Option<usize>,
    pending_fetches: &PrefixFetchWaiterMap,
    cancelled: &CancelledSet,
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

    ensure_whole_model_for_generate(models, (layer_start, layer_end, 0, 1), &model_id)?;

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

    prompt_fits_window(
        prompt_tokens,
        gen.sampling.max_tokens,
        model.context_window(),
        gen.model_id.0.as_str(),
    )?;

    // Prefix-cache lookup: if a cached entry shares a long-enough prefix
    // with this prompt (the entry is narrowed to the shared length when it
    // is longer), hydrate the request's KV with the snapshot and only
    // forward the suffix. Try local first (free); on miss, probe cross-node
    // (Item 8 Phase 2b).
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    ensure_room_for_prompt(model, kv_store, prefix_cache, &req_id_str, prompt_ids.len())?;
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(kv_store, &model_key_string, &req_id_str, snap)
            .unwrap_or(0),
        None => 0,
    };
    // Clamp to keep at least one token for the forward pass, and make the KV
    // cache AGREE with the clamp — see `reconcile_hydrated_prefix`.
    prefix_len = reconcile_hydrated_prefix(
        kv_store,
        model.kv_model_key(),
        &req_id_str,
        prefix_len,
        prompt_tokens,
    );
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
    let prefill = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
        model.forward(&input, index_pos_start, kv_store, &req_id_str)
    });
    let logits = match prefill {
        Ok(l) => l,
        Err(e) if crate::inference::split::kv_cache::forward_was_cancelled(&e) => {
            // The client left during the prompt pass. Not an error: the
            // reply was never wanted. Clean up as a finished request does.
            cancelled.remove(&request_id);
            tracing::info!(
                %request_id,
                prompt_tokens = prompt_ids.len(),
                "model-worker: prompt pass abandoned — request cancelled by daemon"
            );
            kv_store.clear_request(model.kv_model_key(), &req_id_str);
            send_worker(
                writer,
                &WorkerMsg::GenerateDone {
                    request_id,
                    prompt_tokens: prompt_ids.len(),
                    completion_tokens: 0,
                    finish_reason: "cancelled".to_string(),
                    matched_stop_sequence: None,
                },
                &[],
            )
            .await
            .map_err(|e| SwarmError::Internal(format!("send GenerateDone: {e}")))?;
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // After prefill the KV cache holds exactly `prompt_tokens` positions.
    // Snapshot it into the prefix cache so future prompts sharing this
    // prefix skip the prefill work. insert_from_kv is a no-op when the
    // prompt is shorter than the configured floor or the cache is off.
    let keep = snapshot_positions_that_fit(model, kv_store, prefix_cache, prompt_ids.len());
    let manifest =
        prefix_cache.insert_from_kv(&model_key_string, &req_id_str, kv_store, &prompt_ids, keep);
    // The snapshot just taken is device memory the head-room guard must see.
    if prefix_cache_charged() {
        kv_store.set_external_reserved(prefix_cache.bytes_total() as u64);
    }
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
    // Bytes of a codepoint the previous token did not finish. Same lifetime as
    // `accumulated_text`: it belongs to THIS generation and must not be shared.
    let mut utf8_carry: Vec<u8> = Vec::new();
    let mut index_pos = prompt_tokens;
    let mut finish_reason = "length".to_string();
    let mut matched_stop_sequence: Option<String> = None;

    if swift_active {
        let calibrator = SwiftCalibrator::new(
            model.total_layers,
            swift_cfg.skip_ratio,
            swift_cfg.calibration_tokens,
        );
        let (outcome, matched) = swift_decode_loop(
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
            &mut utf8_carry,
        )
        .await?;
        finish_reason = outcome;
        if matched.is_some() {
            matched_stop_sequence = matched;
        }
        tracing::info!(
            request_id = %request_id,
            rounds = calibrator.rounds(),
            acceptance_rate = calibrator.acceptance_rate(),
            num_candidates = calibrator.num_candidates(),
            selected = ?calibrator.selected_candidate(),
            "DIAG: SWIFT session complete"
        );
    } else {
        // Draft-free speculation on the local path. `ngram_spec_eligible` owns
        // the conditions, and the admission gate asks it the same question, so
        // the two cannot disagree about which requests get here.
        //
        // Works at ANY temperature: sampling each position with the real
        // sampler and keeping a draft only when it matches is the
        // speculative-sampling rejection rule itself, not an approximation of
        // it. See the note on `ngram_spec_eligible`.
        let spec_cfg = *ngram_cfg;
        let spec_active = ngram_spec_eligible(&gen.sampling, ngram_cfg, swift_cfg) && !swift_active;
        // Context the lookup searches: prompt then generation, in order, so its
        // tail is always the most recent token. Only built when it will be used
        // — it is a copy of the whole prompt.
        let mut spec_ctx: Vec<u32> = if spec_active {
            prompt_ids.clone()
        } else {
            Vec::new()
        };
        // Verified tokens a round produced beyond the one being emitted. They
        // are already in the KV cache, so draining them costs no forward.
        let mut pending: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let (mut spec_rounds, mut spec_drafted, mut spec_accepted) = (0u64, 0u64, 0u64);
        let mut spec_backoff = SpecBackoff::default();

        for _ in 0..gen.sampling.max_tokens {
            // This loop owns the worker's main loop for its whole duration, so
            // the cancelled set (written by the reader task) is the only way a
            // daemon-side abandonment reaches us. Checking per token bounds
            // wasted compute to one forward instead of the full max_tokens.
            if cancelled.remove(&request_id).is_some() {
                tracing::info!(
                    %request_id,
                    generated = generated.len(),
                    "model-worker: generation cancelled by daemon"
                );
                finish_reason = "cancelled".to_string();
                break;
            }
            if eos.contains(&next_token) {
                finish_reason = "stop".to_string();
                break;
            }

            let text = decode_token(model, next_token, &mut utf8_carry);
            accumulated_text.push_str(&text);

            // Check user-provided stop sequences
            if let Some(matched) =
                crate::inference::sampling::find_stop_sequence(&accumulated_text, stop_sequences)
            {
                matched_stop_sequence = Some(matched.to_string());
                finish_reason = "stop".to_string();
                break;
            }

            generated.push(next_token);
            if spec_active {
                spec_ctx.push(next_token);
            }

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

            if let Some(tok) = pending.pop_front() {
                // Already verified and already in the cache: no forward at all.
                // This is where speculation actually pays.
                next_token = tok;
                token_logprob = None;
            } else if spec_active {
                let draft_allowed = spec_backoff.should_draft();
                let (produced, committed) = ngram_spec_round(
                    model,
                    kv_store,
                    &model_key_string,
                    &req_id_str,
                    index_pos,
                    next_token,
                    &spec_ctx,
                    prompt_ids.len(),
                    &generated,
                    &gen.sampling,
                    spec_cfg,
                    draft_allowed,
                    force_attn,
                )?;
                index_pos += committed;
                spec_rounds += 1;
                spec_drafted += produced.len() as u64;
                spec_accepted += (committed - 1) as u64;
                if draft_allowed {
                    spec_backoff.record(committed > 1);
                }
                let mut it = produced.into_iter();
                // A round always yields at least the token that follows the one
                // just forwarded, so this cannot be empty.
                next_token = it.next().ok_or_else(|| {
                    SwarmError::Internal("speculative round produced no token".into())
                })?;
                pending.extend(it);
                token_logprob = None;
            } else {
                let input = model.token_tensor(next_token)?;
                let logits = tokio::task::block_in_place(|| {
                    let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
                    model.forward(&input, index_pos, kv_store, &req_id_str)
                })?;
                // Pass `generated` so frequency_penalty / presence_penalty
                // are honored against the completion-so-far per OpenAI spec.
                let (tok, lp) = crate::inference::tensor_util::sample_token_with_logprob_history(
                    &logits,
                    &gen.sampling,
                    &generated,
                )?;
                next_token = tok;
                token_logprob = lp;
                index_pos += 1;
            }
        }

        if spec_rounds > 0 {
            let tokens_per_round = spec_drafted as f64 / spec_rounds as f64;
            // Steers the next admission decision — see
            // `spec_payoff_justifies_diverting`.
            record_spec_payoff(tokens_per_round);
            tracing::debug!(
                request_id = %request_id,
                rounds = spec_rounds,
                drafted = spec_drafted,
                accepted = spec_accepted,
                paused_rounds = spec_backoff.paused_rounds,
                tokens_per_round,
                "DIAG: local n-gram speculation complete"
            );
        }

        // The loop samples one token ahead: when it exits on `length`,
        // `next_token` holds a candidate that was never sent.
        //
        // That candidate is the (max_tokens + 1)-th token and sending it
        // unconditionally — as this did, to "avoid the off-by-one" — created
        // one instead. Measured 2026-08-05: EVERY request returned exactly
        // `max_tokens + 1` completion tokens (1->2, 2->3, 5->6, 8->9, 16->17).
        // `max_tokens` is a hard limit people size cost and latency against, so
        // exceeding it is not a rounding detail; on a cloud-proxied request it
        // is billable.
        //
        // Kept, but bounded: it may only ever top the reply UP TO the limit,
        // never past it. If a future change does leave the loop one short, this
        // still covers it; if it does not, this does nothing.
        //
        // Both decode paths carried this pattern and both are verified exact
        // (`continuous_batching` false and true, confirmed from the worker's own
        // `BatchGenerate configured enabled=` line rather than assumed).
        if finish_reason == "length"
            && (generated.len() as u32) < gen.sampling.max_tokens
            && gen.sampling.max_tokens > 0
            && !eos.contains(&next_token)
        {
            let text = decode_token(model, next_token, &mut utf8_carry);
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
            matched_stop_sequence,
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
    // Threaded rather than owned here: SWIFT decodes into the SAME reply as the
    // caller, so a codepoint can straddle the boundary between them.
    utf8_carry: &mut Vec<u8>,
) -> Result<(String, Option<String>), SwarmError> {
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
        // Paired with `accumulated_text` — see `decode_token`. Passing a fresh
        // buffer per call would defeat the point: the carry only works if it
        // survives from one token to the next.
        utf8_carry: &mut Vec<u8>,
    ) -> Result<EmitOutcome, SwarmError> {
        if eos.contains(&token) {
            return Ok(EmitOutcome::Stop);
        }
        if generated.len() as u32 >= max_tokens {
            return Ok(EmitOutcome::Length);
        }
        let text = decode_token(model, token, utf8_carry);
        accumulated_text.push_str(&text);
        if let Some(matched) =
            crate::inference::sampling::find_stop_sequence(accumulated_text, stop_sequences)
        {
            return Ok(EmitOutcome::StopMatch(matched.to_string()));
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
            return Ok(("stop".into(), None));
        }
        if generated.len() as u32 >= gen.sampling.max_tokens {
            return Ok(("length".into(), None));
        }

        let p_start = *index_pos;
        let remaining_budget = gen.sampling.max_tokens - generated.len() as u32;
        // Need budget for at least 1 emitted token from this round; if the
        // budget can't cover next_token alone, just emit it via the per-token
        // fallback.
        if remaining_budget == 0 {
            return Ok(("length".into(), None));
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
        let mut break_outcome: Option<(String, Option<String>)> = None;

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
            utf8_carry,
        )
        .await?
        {
            EmitOutcome::Continue => {}
            EmitOutcome::Stop => break_outcome = Some(("stop".into(), None)),
            EmitOutcome::StopMatch(matched) => break_outcome = Some(("stop".into(), Some(matched))),
            EmitOutcome::Length => break_outcome = Some(("length".into(), None)),
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
                    utf8_carry,
                )
                .await?
                {
                    EmitOutcome::Continue => {}
                    EmitOutcome::Stop => {
                        break_outcome = Some(("stop".into(), None));
                        break;
                    }
                    EmitOutcome::StopMatch(matched) => {
                        break_outcome = Some(("stop".into(), Some(matched)));
                        break;
                    }
                    EmitOutcome::Length => {
                        break_outcome = Some(("length".into(), None));
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
    /// EOS-token-triggered stop (no user stop sequence matched).
    Stop,
    /// User-provided stop sequence matched. The string is carried so the
    /// caller can record `matched_stop_sequence` for the response.
    StopMatch(String),
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
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    slot_table: &SlotTable,
) -> bool {
    if gen.sampling.max_tokens == 0 {
        tracing::debug!(request_id = %gen.request_id, "slot admission refused: max_tokens=0");
        return false;
    }
    // Layer range must match if anything is already in the table.
    let lr = (gen.layer_range.0 as usize, gen.layer_range.1 as usize);
    if !slot_table.can_admit(lr) {
        tracing::debug!(
            request_id = %gen.request_id,
            requested = ?lr,
            table_range = ?slot_table.layer_range(),
            occupied = slot_table.len(),
            "slot admission refused: layer range mismatch or table full"
        );
        return false;
    }
    // SWIFT decoding has its own self-speculative loop; not batchable v1.
    if swift_cfg.enabled && gen.sampling.temperature == 0.0 {
        tracing::debug!(request_id = %gen.request_id, "slot admission refused: SWIFT-eligible");
        return false;
    }
    // A request that is ALONE and can be speculated takes the sequential loop
    // instead, where the n-gram speculator lives.
    //
    // **Only when the table is empty.** Refusing while others are decoding
    // would send this request to a loop that owns the worker for its whole
    // duration, stalling everyone already in the batch to speed up the
    // newcomer. Joining the batch is strictly better there; speculation is
    // simply not available to it.
    //
    // Trading batching for speculation when solo is safe on this project's own
    // numbers: batching was measured at ~3% for 8 concurrent requests and
    // NEUTRAL on a processor, after a claimed 40% was retracted as a
    // measurement artefact (gotcha #348). A verify round wins far more than
    // that, and against an empty table batching has nothing to amortise
    // anyway — there is no second request to share the weight read with.
    if slot_table.is_empty()
        && ngram_spec_eligible(&gen.sampling, ngram_cfg, swift_cfg)
        && spec_payoff_justifies_diverting()
    {
        tracing::debug!(
            request_id = %gen.request_id,
            "slot admission refused: solo and speculatable — taking the n-gram path"
        );
        return false;
    }
    tracing::debug!(
        request_id = %gen.request_id,
        occupied = slot_table.len(),
        "slot admission accepted — request will decode in a shared batch"
    );
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

    if let Err(e) =
        ensure_whole_model_for_generate(models, (layer_start, layer_end, 0, 1), &model_id)
    {
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
    // Same check as the non-batched path — and this is the one that runs by
    // default, since `continuous_batching` is on.
    prompt_fits_window(
        prompt_tokens,
        gen.sampling.max_tokens,
        model.context_window(),
        gen.model_id.0.as_str(),
    )
    .map_err(SlotAdmitError::Fatal)?;

    // Prefix-cache lookup + per-request KV hydration if we hit. Cheap clone of
    // K/V tensors — no compute.
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    ensure_room_for_prompt(model, kv_store, prefix_cache, &req_id_str, prompt_ids.len())
        .map_err(SlotAdmitError::Fatal)?;
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(kv_store, &model_key_string, &req_id_str, snap)
            .unwrap_or(0),
        None => 0,
    };
    // Always leave at least one prompt token for the first chunk's forward —
    // we need that forward to produce logits for the first sample — and make
    // the KV cache agree with that count. See `reconcile_hydrated_prefix`.
    prefix_len = reconcile_hydrated_prefix(
        kv_store,
        model.kv_model_key(),
        &req_id_str,
        prefix_len,
        prompt_tokens,
    );
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
        utf8_carry: Vec::new(),
        sampling,
        prompt_tokens,
        prompt_ids,
        generated_ids: Vec::new(),
        finish_reason: None,
        error_message: None,
        matched_stop_sequence: None,
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
    prefill_pacer: &mut crate::inference::prefill_pacer::PrefillPacer,
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

        // Size this tick's quantum from measured wall time. `is_sharing` gates
        // on there being someone to starve — a solo prompt keeps the full
        // configured chunk so its throughput is untouched.
        let sharing = crate::inference::prefill_pacer::PrefillPacer::is_sharing(active.len());
        let chunk_size = prefill_pacer.chunk_size(sharing);
        let phase_a_started = std::time::Instant::now();

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
                        Err(e) if crate::inference::split::kv_cache::forward_was_cancelled(&e) => {
                            // The cancel landed mid-chunk; the drain step
                            // collects the slot like any other cancelled one.
                            tracing::debug!(request_id = %step.request_id, "BatchGenerate prefill chunk abandoned — request cancelled");
                        }
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
            let mut pending_progress: Vec<(uuid::Uuid, u32, u32)> = Vec::new();
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
                pending_progress.push((
                    request_id,
                    (step.pos + step.chunk_len) as u32,
                    (step.pos + step.chunk_len + step.remaining_after) as u32,
                ));
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
                    let keep = snapshot_positions_that_fit(
                        model,
                        kv_store,
                        prefix_cache,
                        slot.prompt_ids.len(),
                    );
                    let manifest = prefix_cache.insert_from_kv(
                        &slot.model_key,
                        &slot.req_id_str,
                        kv_store,
                        &slot.prompt_ids,
                        keep,
                    );
                    if prefix_cache_charged() {
                        kv_store.set_external_reserved(prefix_cache.bytes_total() as u64);
                    }
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

            // Tell the daemon how far each still-prefilling request has got, so
            // a long prompt reads as progress rather than as a hang. Same
            // collect-then-send shape as the announces above: the `active`
            // borrow has to end before we can touch the writer.
            for (request_id, done, total) in pending_progress {
                let _ = send_worker(
                    writer,
                    &WorkerMsg::Progress {
                        request_id,
                        phase: crate::inference::worker_ipc::ProgressPhase::Prefill,
                        done,
                        total,
                    },
                    &[],
                )
                .await;
            }

            // Feed the measured cost back into the pacer. Timed across the whole
            // of Phase A rather than per group, because that is the quantity the
            // budget is about: what a co-scheduled decode slot waits through
            // before Phase B runs.
            let prefill_tokens: usize = steps.iter().map(|s| s.chunk_len).sum();
            prefill_pacer.observe(prefill_tokens, phase_a_started.elapsed());
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
            let text = decode_token(model, last_token, &mut slot.utf8_carry);
            slot.accumulated_text.push_str(&text);
            if let Some(matched) = crate::inference::sampling::find_stop_sequence(
                &slot.accumulated_text,
                &slot.stop_sequences,
            ) {
                let matched = matched.to_string();
                slot.finish_stop_with_match(matched);
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
    tracing::debug!(
        total_slots = active.len(),
        decoding = still_active_indices.len(),
        "DIAG: decode tick slot census"
    );
    if still_active_indices.is_empty() {
        return Ok(());
    }

    let mut input_tensors: Vec<candle_core::Tensor> =
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

    // BatchItem borrows req_id_str directly from `active` — no per-slot
    // clone needed. items goes out of scope before any mutable borrow of
    // active resumes (line 2479's `&mut active[i]`).
    let items: Vec<BatchItem<'_>> = still_active_indices
        .iter()
        .enumerate()
        .map(|(j, &i)| BatchItem {
            input: &input_tensors[j],
            index_pos: index_positions[j],
            request_id: active[i].req_id_str.as_str(),
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
        // Pass slot.generated_ids so frequency_penalty / presence_penalty
        // see the completion-so-far for THIS slot (each batch slot has
        // its own decoded history).
        let (next_tok, next_logprob) =
            match crate::inference::tensor_util::sample_token_with_logprob_history(
                &outputs[j],
                &slot.sampling,
                &slot.generated_ids,
            ) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(request_id = %slot.request_id, error = %e, "DIAG: BatchGenerate decode sample failed — slot errored");
                    slot.finish_error(format!("decode sample: {e}"));
                    continue;
                }
            };
        slot.generated_ids.push(next_tok);
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
            // The limit is already reached, so `next_tok` is the
            // (max_tokens + 1)-th token and must NOT be sent.
            //
            // This used to send it, which made every request on the batched
            // path return exactly one token too many. Measured 2026-08-05
            // against a streamed reply, counting the deltas on the wire rather
            // than trusting the usage field: max_tokens=3 produced 4 chunks,
            // max_tokens=8 produced 9. So the text genuinely exceeded the
            // limit — this was not a mis-count. `max_tokens` is what people
            // size cost and latency against, and on a cloud-proxied request the
            // extra token is billable.
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
        matched_stop_sequence,
        ..
    } = slot;
    let finish_label = finish_reason.unwrap_or("length").to_string();

    if finish_label == "error" {
        let message = error_message
            .unwrap_or_else(|| "BatchGenerate slot failed without a recorded message".to_string());
        let fatal = crate::inference::worker_ipc::worker_error_is_fatal(&message);
        send_worker(
            writer,
            &WorkerMsg::Error {
                request_id,
                message,
                fatal,
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
            matched_stop_sequence,
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
#[allow(clippy::too_many_arguments)] // distinct concerns: writer, model state, ctx, options, mailbox
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
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    options: &WorkerOptions,
    slot_table: &mut SlotTable,
    pending_fetches: &PrefixFetchWaiterMap,
    cancelled: &CancelledSet,
) -> bool {
    let activation_compression = options.activation_compression;
    let force_standard_attn = options.force_standard_attn;
    let max_seq_len_override = options.max_seq_len_override;
    let batch_generate = options.batch_generate;

    // A cancel can overtake the request it cancels — the reader task
    // short-circuits cancels while requests queue behind `ipc_rx`. Drop the
    // work before starting it rather than computing a reply nobody reads.
    if let Some(request_id) = cancelled_request_id(&msg) {
        if cancelled.remove(&request_id).is_some() {
            tracing::debug!(%request_id, "model-worker: skipping already-cancelled request");
            return false;
        }
    }

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
                    .map(|g| slot_admission_eligible(g, swift_cfg, ngram_cfg, slot_table))
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
                    ngram_cfg,
                    force_standard_attn,
                    max_seq_len_override,
                    pending_fetches,
                    cancelled,
                )
                .await
                {
                    send_worker_error(writer, request_id, e).await;
                }
            }
        }
        DaemonMsg::CancelRequest { request_id } => {
            // The reader task intercepts these before they reach the main
            // loop; arriving here means the short-circuit was bypassed
            // (a direct-dispatch test harness). Record it anyway so the
            // semantics hold on either path.
            cancelled.insert(request_id, std::time::Instant::now());
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

/// Emit the longest complete UTF-8 prefix of `carry`, keeping any incomplete
/// trailing sequence for the next token.
///
/// **A codepoint can span several tokens.** Emoji and most non-Latin scripts are
/// emitted as byte-fallback tokens — one token per BYTE — so converting each
/// token to text on its own turns every one of those bytes into U+FFFD. Asking
/// llama-3.2-3b for three emoji returned nine replacement characters,
/// deterministically, 3 runs out of 3 (2026-08-05).
///
/// Buffering the tail is what every streaming detokenizer does for this reason
/// (llama.cpp's examples accumulate bytes; HuggingFace `tokenizers` ships a
/// `DecodeStream` for it).
fn take_complete_utf8(carry: &mut Vec<u8>) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(carry) {
            Ok(s) => {
                out.push_str(s);
                carry.clear();
                return out;
            }
            Err(e) => {
                let good = e.valid_up_to();
                if good > 0 {
                    // Valid by construction — `valid_up_to` is a UTF-8 boundary.
                    out.push_str(std::str::from_utf8(&carry[..good]).unwrap_or_default());
                }
                match e.error_len() {
                    // Truncated at the end: the rest of this codepoint is in the
                    // next token. Keep it rather than corrupting it.
                    None => {
                        carry.drain(..good);
                        return out;
                    }
                    // Genuinely invalid bytes. Emit one replacement and skip
                    // them, or we would spin on the same bytes forever.
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        carry.drain(..good + bad);
                    }
                }
            }
        }
    }
}

/// Reject a prompt that cannot fit the model's context window, with numbers the
/// caller can act on.
///
/// **Both tokenization sites must call this.** Chunked prefill discovers the
/// overflow one 128-token chunk at a time, so the executor's own guard can only
/// report `index_pos + chunk_len` — a value just past the limit whatever the
/// prompt actually was. Every over-long prompt therefore got the SAME number: a
/// 600-word and a 1500-word prompt were both told "Sequence length (4224)
/// exceeds model context window (4096)", so "reduce your prompt" gave no hint
/// whether to cut a little or three quarters of it (measured 2026-08-05). It
/// also burned a full prefill before failing.
///
/// The executor guard stays as the backstop for paths that do not come through
/// here; this exists to make the message actionable.
fn prompt_fits_window(
    prompt_tokens: usize,
    max_new_tokens: u32,
    window: usize,
    model_id: &str,
) -> Result<(), SwarmError> {
    let needed = prompt_tokens.saturating_add(max_new_tokens as usize);
    if needed <= window {
        return Ok(());
    }
    let over = needed - window;
    Err(SwarmError::Validation(format!(
        "This conversation is too long for {model_id}: {prompt_tokens} tokens of prompt \
         plus {max_new_tokens} reserved for the reply is {needed}, and the model's limit \
         is {window}. Shorten it by about {over} tokens (roughly {words} words), or ask \
         for a shorter reply.",
        words = (over * 3) / 4,
    )))
}

/// Decode a single token to text using the model's vocabulary.
///
/// `carry` holds bytes left over from a codepoint that was not finished by the
/// previous token; it is a required parameter rather than internal state so a
/// caller cannot decode a stream without one and silently mangle every
/// multi-byte character. Flush it with [`take_complete_utf8`] when generation
/// ends so a trailing partial sequence is not dropped in silence.
fn decode_token(model: &SplitModel, token_id: u32, carry: &mut Vec<u8>) -> String {
    if let Some(vocab) = model.vocab() {
        if let Some(token_str) = vocab.get(token_id as usize) {
            if let Some(tokenizer) = model.tokenizer() {
                carry.extend_from_slice(&tokenizer.decode_token(token_str));
                return take_complete_utf8(carry);
            }
            // No tokenizer: the raw vocab entry is NOT user-facing text.
            //
            // Returning it verbatim — as this used to — puts the vocabulary's
            // own notation into the reply: every space becomes `▁` (U+2581) and
            // a byte-fallback token appears literally as `<0x0A>`. Both were
            // seen in served replies (2026-07-28/29): Phi-3.5 answered
            // `A▁distributed▁system▁is…` through the local API while the very
            // same node returned clean text for the same prompt over the
            // network, because that path had a tokenizer and this one did not.
            //
            // We cannot know the vocabulary's family without the tokenizer, but
            // these two conventions are never legitimate output in ANY family,
            // so undoing them is strictly better than emitting them. Warn once
            // so a missing tokenizer is visible rather than silently degrading
            // every reply this node produces.
            warn_missing_tokenizer_once();
            return decode_raw_vocab_entry(token_str);
        }
    }
    String::new()
}

/// Best-effort cleanup of a raw vocabulary entry when no tokenizer is loaded.
///
/// Handles the two notations that are always artefacts rather than content:
/// SentencePiece's `▁` word-boundary marker, and `<0xNN>` byte fallback.
fn decode_raw_vocab_entry(token_str: &str) -> String {
    // Byte fallback: `<0x0A>` is one byte, not six characters.
    if token_str.len() == 6 && token_str.starts_with("<0x") && token_str.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
            return String::from_utf8_lossy(&[byte]).into_owned();
        }
    }
    token_str.replace('\u{2581}', " ")
}

/// One warning per process — this fires per TOKEN, so logging every time would
/// bury the log in exactly the situation where it is needed.
fn warn_missing_tokenizer_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "No tokenizer available for this model — replies are being assembled \
             from raw vocabulary entries. Text will be approximate; check that \
             gguf_header.bin is present for this model."
        );
    });
}

#[cfg(test)]
mod decode_raw_vocab_tests {
    use super::decode_raw_vocab_entry;

    /// The observed defect: SentencePiece word boundaries reaching the user.
    #[test]
    fn word_boundary_markers_become_spaces() {
        assert_eq!(
            decode_raw_vocab_entry("\u{2581}distributed"),
            " distributed"
        );
        assert_eq!(decode_raw_vocab_entry("\u{2581}a\u{2581}b"), " a b");
    }

    /// The other half of the same defect — `<0x0A>` is a newline, not six
    /// characters of literal text.
    #[test]
    fn byte_fallback_tokens_become_their_byte() {
        assert_eq!(decode_raw_vocab_entry("<0x0A>"), "\n");
        assert_eq!(decode_raw_vocab_entry("<0x20>"), " ");
    }

    /// Ordinary text is passed through untouched.
    #[test]
    fn ordinary_tokens_are_unchanged() {
        assert_eq!(decode_raw_vocab_entry("hello"), "hello");
        assert_eq!(decode_raw_vocab_entry("."), ".");
        // Not a byte-fallback token despite the shape — left alone.
        assert_eq!(decode_raw_vocab_entry("<0xZZ>"), "<0xZZ>");
        assert_eq!(decode_raw_vocab_entry("<s>"), "<s>");
    }
}

#[cfg(test)]
mod utf8_stream_tests {
    use super::take_complete_utf8;

    use super::prompt_fits_window;

    /// **The message has to say how much to cut.** Chunked prefill discovers
    /// the overflow one chunk at a time, so the executor could only ever report
    /// a position just past the limit — a 600-word prompt and a 1500-word
    /// prompt were both told "Sequence length (4224)", which gives no idea
    /// whether to trim a sentence or three quarters of the conversation.
    #[test]
    fn an_overlong_prompt_is_told_its_real_size_and_overage() {
        let err = prompt_fits_window(9000, 100, 4096, "llama-3.2-3b").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("9000"),
            "must state the real prompt size: {msg}"
        );
        assert!(msg.contains("4096"), "must state the limit: {msg}");
        // 9000 + 100 - 4096 = 5004 over.
        assert!(msg.contains("5004"), "must state how much to cut: {msg}");
    }

    /// The reservation for the reply counts — a prompt that fits alone can
    /// still not leave room to answer.
    #[test]
    fn the_reply_reservation_is_included() {
        assert!(prompt_fits_window(4000, 96, 4096, "m").is_ok());
        assert!(prompt_fits_window(4000, 97, 4096, "m").is_err());
    }

    /// A prompt that fits is not refused.
    #[test]
    fn a_prompt_that_fits_passes() {
        assert!(prompt_fits_window(10, 10, 4096, "m").is_ok());
        assert!(prompt_fits_window(4096, 0, 4096, "m").is_ok());
    }

    /// **A codepoint can span several tokens.** Emoji and most non-Latin
    /// scripts arrive as byte-fallback tokens — one token per BYTE — so
    /// converting each token to text on its own turns every byte into U+FFFD.
    /// Asking llama-3.2-3b for three emoji returned nine replacement
    /// characters, deterministically, 3 runs out of 3 (2026-08-05).
    #[test]
    fn a_codepoint_split_across_tokens_survives() {
        let mut carry = Vec::new();
        let mut out = String::new();
        // "\u{1F389}" is F0 9F 8E 89 — four separate byte-fallback tokens.
        for b in [0xF0u8, 0x9F, 0x8E, 0x89] {
            carry.push(b);
            out.push_str(&take_complete_utf8(&mut carry));
        }
        assert_eq!(out, "\u{1F389}");
        assert!(carry.is_empty(), "nothing should be left pending");
        assert!(!out.contains('\u{FFFD}'), "no replacement characters");
    }

    /// Nothing is emitted early: a partial codepoint must be held back rather
    /// than flushed as garbage the caller has already streamed to the user.
    #[test]
    fn a_partial_codepoint_is_held_not_emitted() {
        let mut carry = vec![0xF0u8, 0x9F];
        assert_eq!(take_complete_utf8(&mut carry), "");
        assert_eq!(
            carry,
            vec![0xF0, 0x9F],
            "the tail must be kept for next time"
        );
    }

    /// Text before an incomplete tail still flows immediately — buffering must
    /// not stall a stream waiting for a codepoint that has not started.
    #[test]
    fn complete_text_before_a_partial_tail_is_emitted_at_once() {
        let mut carry = b"hello \xF0\x9F".to_vec();
        assert_eq!(take_complete_utf8(&mut carry), "hello ");
        assert_eq!(carry, vec![0xF0, 0x9F]);
    }

    /// Genuinely invalid bytes must be consumed, not retried forever — a lone
    /// continuation byte can never become valid however many follow it.
    #[test]
    fn invalid_bytes_terminate_instead_of_spinning() {
        let mut carry = vec![0x80u8, b'o', b'k'];
        let out = take_complete_utf8(&mut carry);
        assert_eq!(out, "\u{FFFD}ok");
        assert!(carry.is_empty());
    }

    /// Multi-byte text that arrives whole in one token is unaffected.
    #[test]
    fn whole_codepoints_pass_through_unchanged() {
        let mut carry = "\u{65E5}\u{672C}\u{8A9E}".as_bytes().to_vec();
        assert_eq!(take_complete_utf8(&mut carry), "\u{65E5}\u{672C}\u{8A9E}");
        assert!(carry.is_empty());
    }
}

#[cfg(test)]
mod prefix_reconcile_tests {
    use super::reconcile_hydrated_prefix;
    use crate::inference::split::KvCacheStore;
    use candle_core::{Device, Tensor};
    use std::sync::Arc;

    const MODEL_KEY: &str = "m";
    const REQ: &str = "r";

    /// Put `n` sequence positions into every layer of a request's KV cache,
    /// the way hydration does.
    fn hydrate(store: &Arc<KvCacheStore>, n: usize, layers: usize) {
        let mut entry = store.get_or_create(MODEL_KEY, REQ, layers);
        for slot in entry.layers.iter_mut() {
            let mut kv = crate::inference::split::kv_cache::LayerKv::with_dim(2, 64);
            // [batch, heads, seq, head_dim]
            let k =
                Tensor::zeros((1usize, 2, n, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
            let v = k.clone();
            kv.append(&k, &v).unwrap();
            *slot = Some(kv);
        }
    }

    fn cached_len(store: &Arc<KvCacheStore>) -> usize {
        let key = KvCacheStore::cache_key(MODEL_KEY, REQ);
        store
            .get_entry(&key)
            .and_then(|e| {
                e.layers
                    .iter()
                    .find_map(|c| c.as_ref().map(|kv| kv.k_cache().current_seq_len()))
            })
            .unwrap_or(0)
    }

    /// The reported defect's precondition: an EXACT repeat of a prompt hydrates
    /// every token, so the clamp to `prompt_tokens - 1` has real work to do.
    /// Before this, the cache kept all 8 positions while the caller was told 7
    /// — `forward_inner_impl` reads `kv_offset` from the cache, so the last
    /// prompt token was attended to twice.
    #[test]
    fn a_full_prompt_hit_leaves_the_cache_agreeing_with_the_clamp() {
        let store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(60)));
        hydrate(&store, 8, 3);
        assert_eq!(cached_len(&store), 8);

        let got = reconcile_hydrated_prefix(&store, MODEL_KEY, REQ, 8, 8);
        assert_eq!(got, 7, "must leave one token for the forward");
        assert_eq!(
            cached_len(&store),
            7,
            "the cache must be truncated to match the number reported, not just the number"
        );
    }

    /// A partial hit needs no truncation and must be left exactly as hydrated.
    #[test]
    fn a_partial_hit_is_untouched() {
        let store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(60)));
        hydrate(&store, 4, 3);
        let got = reconcile_hydrated_prefix(&store, MODEL_KEY, REQ, 4, 10);
        assert_eq!(got, 4);
        assert_eq!(cached_len(&store), 4);
    }

    /// A single-token prompt cannot keep any prefix — the forward needs it.
    #[test]
    fn a_one_token_prompt_yields_no_prefix() {
        let store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(60)));
        hydrate(&store, 1, 2);
        assert_eq!(reconcile_hydrated_prefix(&store, MODEL_KEY, REQ, 1, 1), 0);
        assert_eq!(cached_len(&store), 0);
    }

    /// Nothing hydrated is a no-op, not a truncation attempt.
    #[test]
    fn zero_hydrated_is_inert() {
        let store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(60)));
        assert_eq!(reconcile_hydrated_prefix(&store, MODEL_KEY, REQ, 0, 10), 0);
    }
}

#[cfg(test)]
mod local_speculation_tests {
    use super::*;
    use crate::inference::ngram_lookup::NgramLookupConfig;
    use crate::types::SamplingParams;

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            logprobs: false,
            ..Default::default()
        }
    }

    fn on() -> NgramLookupConfig {
        NgramLookupConfig::default()
    }

    fn swift_off() -> SwiftConfig {
        SwiftConfig {
            enabled: false,
            ..Default::default()
        }
    }

    /// The admission gate and the decode loop must agree about which requests
    /// are speculated. They were two conditions in two places before this
    /// helper existed, and the failure mode is silent: the gate diverts a
    /// request off the batched path and the loop then declines to speculate it,
    /// so it loses batching and gains nothing.
    #[test]
    fn eligibility_is_one_predicate_both_callers_share() {
        assert!(ngram_spec_eligible(&greedy(), &on(), &swift_off()));

        // Sampling on is FINE, and this is the case that matters: 0.7 and 1.0
        // are the two API defaults, so gating on greedy left the feature inert
        // for almost all real traffic. Keeping a draft only when the sampler
        // independently produced it is the speculative-sampling rejection rule
        // itself — see the note on `ngram_spec_eligible` and
        // `accepting_only_on_a_match_preserves_the_sampled_distribution`.
        let mut warm = greedy();
        warm.temperature = 0.7;
        assert!(ngram_spec_eligible(&warm, &on(), &swift_off()));

        // Logprobs asked for: accepted tokens carry none back out, and
        // answering `null` where a client asked for numbers is worse than
        // declining to speculate.
        let mut lp = greedy();
        lp.logprobs = true;
        assert!(!ngram_spec_eligible(&lp, &on(), &swift_off()));

        // SWIFT is already speculating for this request.
        let swift_on = SwiftConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(!ngram_spec_eligible(&greedy(), &on(), &swift_on));

        // `inference.ngram_lookup_enabled = false` arrives as a zero width, so
        // the switch cannot disagree with the shape.
        let off = NgramLookupConfig {
            num_pred_tokens: 0,
            ..NgramLookupConfig::default()
        };
        assert!(!ngram_spec_eligible(&greedy(), &off, &swift_off()));
    }

    #[test]
    fn backoff_pauses_after_a_run_of_useless_rounds() {
        let mut b = SpecBackoff::default();
        // Drafting is free to start.
        for _ in 0..SpecBackoff::MISSES_BEFORE_PAUSE {
            assert!(b.should_draft());
            b.record(false);
        }
        // Having missed three times, it now sits out a couple of rounds rather
        // than paying a wider forward for each of them.
        assert!(!b.should_draft());
        assert!(b.paused_rounds > 0);
    }

    #[test]
    fn any_acceptance_clears_the_whole_backoff() {
        let mut b = SpecBackoff::default();
        for _ in 0..(SpecBackoff::MISSES_BEFORE_PAUSE * 4) {
            if b.should_draft() {
                b.record(false);
            }
        }
        assert!(b.pause_len > 0, "should have backed off by now");

        // Drain the pause, then land one acceptance.
        while !b.should_draft() {}
        b.record(true);

        // The next round drafts immediately, and a fresh run of misses starts
        // from the shortest pause again rather than resuming where it left off.
        // A reply that turns copy-heavy halfway through is the case this
        // protects: staying switched off for the rest of it would forfeit
        // exactly the part speculation is best at.
        assert!(b.should_draft());
        assert_eq!(b.pause_len, 0);
    }

    #[test]
    fn the_pause_is_bounded() {
        let mut b = SpecBackoff::default();
        for _ in 0..10_000 {
            if b.should_draft() {
                b.record(false);
            }
        }
        assert!(
            b.pause_len <= SpecBackoff::MAX_PAUSE_ROUNDS,
            "pause grew past its bound: {}",
            b.pause_len
        );
    }
}

#[cfg(test)]
mod spec_payoff_tests {
    use super::*;

    /// Run the EWMA from "unmeasured" through `n` samples of the same value,
    /// the way a run of similar requests would.
    fn settle(tokens_per_round: f64, n: usize) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = blend_spec_payoff(v, tokens_per_round);
        }
        v
    }

    /// Serialising a request out of the batched path costs every other request
    /// in flight. Measured on an RTX 3070 with 8 concurrent requests: worth it
    /// when speculation lands ~8.8 tokens a round (3.76 s against 5.52 s
    /// batched), badly wrong when it lands ~1 (29.07 s against 12.48 s, and
    /// aggregate throughput 33 against 77 tok/s). The threshold has to sit
    /// between the two, and these are the two measured workloads.
    #[test]
    fn the_threshold_separates_the_two_measured_workloads() {
        assert!(payoff_justifies_diverting(settle(8.83, 8)), "copy-heavy");
        assert!(!payoff_justifies_diverting(settle(1.04, 8)), "open-ended");
    }

    /// The first generation after start has nothing to go on and must be allowed
    /// to find out — otherwise the policy latches off and nothing ever measures.
    #[test]
    fn unknown_payoff_lets_one_request_find_out() {
        assert!(payoff_justifies_diverting(0));
    }

    /// It has to be able to change its mind. A session that turns copy-heavy
    /// after a discursive opening is the common agentic shape, and latching off
    /// would forfeit exactly the part speculation is best at.
    #[test]
    fn it_recovers_when_the_workload_changes() {
        let mut v = settle(1.0, 8);
        assert!(!payoff_justifies_diverting(v));
        for _ in 0..8 {
            v = blend_spec_payoff(v, 9.0);
        }
        assert!(
            payoff_justifies_diverting(v),
            "must follow the workload back up, not latch off"
        );
    }

    /// 0 means "not measured". A measured-as-poor workload must never be
    /// mistaken for one, or it silently regains the diversion it just lost.
    #[test]
    fn a_measured_payoff_is_never_mistaken_for_unmeasured() {
        assert_ne!(blend_spec_payoff(0, 0.0), 0);
        assert_ne!(blend_spec_payoff(1, 0.0), 0);
    }
}
