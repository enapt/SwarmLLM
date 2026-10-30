//! Model worker subprocess for SwarmLLM.
//!
//! Each model runs in its own process. When killed, the OS/CUDA driver
//! reclaims ALL GPU memory immediately — solving the "memory doesn't drop
//! on unload" problem and keeping inference off the main daemon's Tokio runtime.

use crate::inference::tokenizer::take_complete_utf8;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::inference::process_pool::IpcWriter;

use crate::daemon::shard_loader::{load_model_range, ShardLoadParams};

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
use crate::inference::split::kv_budget::KvGrowth;
use crate::inference::split::kv_cache::KvHome;
use crate::inference::split::{
    self, BatchItem, KvCacheStore, PrefixCache, SnapshotHome, SnapshotRoom, SplitModel,
};
use crate::inference::swift::{SwiftCalibrator, SwiftConfig};
use crate::inference::worker_ipc::*;
use crate::types::{KeepAsked, NetworkFinishReason, SamplingParams};

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
            max_prompt_tokens: 0,
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
    /// How many of the card's mixture-of-experts layers, from the first, keep
    /// their routed experts in RAM (`--cpu-moe-layers`; the pool's
    /// `CardSplit::ExpertsInRam`). 0 = none.
    pub cpu_moe_layers: usize,
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
            cpu_moe_layers: 0,
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
        cpu_moe_layers,
    } = options;
    set_worker_force_cpu(gpu_layers);
    set_worker_cpu_moe_layers(cpu_moe_layers);
    crate::inference::split::PROMPT_CHUNK_TOKENS.store(
        prefill_chunk_tokens as usize,
        std::sync::atomic::Ordering::Relaxed,
    );
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
            let freed = pc.release(needed as usize, SnapshotHome::BesideLive) as u64;
            ks.set_external_reserved(pc.bytes_in(SnapshotHome::BesideLive) as u64);
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
    // How many generations the card may run at once, by what the card DOES:
    // halved on a stall, one back per quiet minute (`docs/FUTURE_WORK.md` #146).
    let mut card_pace =
        crate::inference::card_pace::CardPace::new(batch_generate_max_slots as usize);
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

    // Read once: the daemon that spawned this worker decodes the card probe's
    // report (`worker_ipc::DAEMON_READS_CARD_PROBE`).
    let daemon_reads_card_probe = crate::inference::worker_ipc::daemon_reads_card_probe(
        std::env::var_os(crate::inference::worker_ipc::DAEMON_READS_CARD_PROBE).as_deref(),
    );
    let daemon_reads_load_timing = crate::inference::worker_ipc::daemon_reads_card_probe(
        std::env::var_os(crate::inference::worker_ipc::DAEMON_READS_LOAD_TIMING).as_deref(),
    );
    let daemon_reads_context_fits = crate::inference::worker_ipc::daemon_reads_card_probe(
        std::env::var_os(crate::inference::worker_ipc::DAEMON_READS_CONTEXT_FITS).as_deref(),
    );
    // The `WorkerMsg::ContextFits` figure last sent (`None` = never sent), the
    // loaded ranges it was read for, and when.
    let mut context_fits_sent: Option<Option<u32>> = None;
    let mut context_fits_ranges: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut context_fits_at = std::time::Instant::now();

    // Whether the card's pool may hold memory freed since the last hand-back:
    // set by any message or tick, cleared by `cuda_pool::trim` once the worker
    // has had nothing to do for `cuda_pool::IDLE_TRIM` (#146).
    let mut pool_may_hold_freed = false;
    // Cancelled requests whose caches this loop has already released, so the
    // release runs once per cancel rather than on every tick it is retained.
    let mut kv_released_for_cancel: std::collections::HashSet<uuid::Uuid> =
        std::collections::HashSet::new();
    // The drafter's trees that flow, kept between a reply's calls (#134 (b)).
    let mut flow_sessions = FlowSessions::default();
    // Chains whose two ends this worker holds, learning their far run.
    let mut far_runs = FarRuns::default();

    loop {
        // Either block on the next IPC message (no slots are decoding) OR race
        // a fresh IPC arrival against an immediate decode tick (slots active).
        let next_msg: Option<(DaemonMsg, Vec<u8>)> = if slot_table.is_empty() {
            // Nothing decoding and nothing waiting: fit what the last replies
            // taught the card's stand-in, stopping the moment a message comes.
            if ipc_rx.is_empty() && refit_stale_priors(&mut models, &|| ipc_rx.is_empty()) {
                pool_may_hold_freed = true;
            }
            if pool_may_hold_freed && crate::inference::cuda_pool::keep_enabled() {
                match tokio::time::timeout(crate::inference::cuda_pool::IDLE_TRIM, ipc_rx.recv())
                    .await
                {
                    Ok(m) => m,
                    Err(_idle) => {
                        pool_may_hold_freed = false;
                        hand_back_idle_card_memory(&models);
                        continue;
                    }
                }
            } else {
                ipc_rx.recv().await
            }
        } else {
            tokio::select! {
                biased;
                m = ipc_rx.recv() => m,
                _ = tokio::task::yield_now() => None,
            }
        };

        let mut shutdown = false;
        if let Some((msg, payload)) = next_msg {
            pool_may_hold_freed = true;
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
                &mut card_pace,
                &pending_fetches,
                &cancelled,
                &mut flow_sessions,
                &mut far_runs,
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
                            &mut card_pace,
                            &pending_fetches,
                            &cancelled,
                            &mut flow_sessions,
                            &mut far_runs,
                        )
                        .await;
                    }
                    Err(_) => break,
                }
            }
            // The card allocation probe ran inside the first load onto the
            // card (`split::loader::load_device`); hand its figure to the
            // daemon once (#762) — only to a daemon that said it reads it.
            if let Some(took) = daemon_reads_card_probe
                .then(crate::inference::cuda_pool::take_probe_report)
                .flatten()
            {
                let took_ms = u32::try_from(took.as_millis()).unwrap_or(u32::MAX);
                if let Err(e) = send_worker(
                    &mut writer,
                    &WorkerMsg::CardAllocationProbe { took_ms },
                    &[],
                )
                .await
                {
                    tracing::debug!(error = %e, "model-worker: could not report the card probe");
                }
            }
            // Every load this batch ran, for the rate this node advertises
            // (#129) — the same drain point, the same gate.
            if daemon_reads_load_timing {
                for (bytes, ms) in take_load_timings() {
                    if let Err(e) =
                        send_worker(&mut writer, &WorkerMsg::ModelLoadTimed { bytes, ms }, &[])
                            .await
                    {
                        tracing::debug!(error = %e, "model-worker: could not report a load timing");
                    }
                }
            }
            // The context the whole model here holds (#252 (1)): read when a
            // load or an unload in this batch changed the ranges, and again
            // now and then while no batched decode is under way.
            if daemon_reads_context_fits {
                let mut ranges: Vec<_> = models.keys().copied().collect();
                ranges.sort_unstable();
                let due = ranges != context_fits_ranges
                    || (slot_table.is_empty() && context_fits_at.elapsed() >= CONTEXT_FITS_EVERY);
                if due {
                    context_fits_ranges = ranges;
                    context_fits_at = std::time::Instant::now();
                    let tokens = whole_model_context_fits(
                        &models,
                        &kv_store,
                        &prefix_cache,
                        draft_margin_positions(&ngram_cfg, &swift_cfg),
                    );
                    if context_fits_sent != Some(tokens) {
                        let msg = WorkerMsg::ContextFits { tokens };
                        match send_worker(&mut writer, &msg, &[]).await {
                            Ok(()) => context_fits_sent = Some(tokens),
                            Err(e) => tracing::debug!(
                                error = %e,
                                "model-worker: could not report the context that fits"
                            ),
                        }
                    }
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
                let range = slot_table.current_layer_range();
                for slot in slot_table.take_matching(|s| cancelled.contains_key(&s.request_id)) {
                    tracing::info!(
                        request_id = %slot.request_id,
                        generated = slot.generated_count(),
                        "model-worker: dropping cancelled decode slot"
                    );
                    cancelled.remove(&slot.request_id);
                    // A retried request still finds its prompt.
                    let model = range.and_then(|(ls, le)| models.get(&(ls, le, 0, 1)));
                    save_slot_prompt_when_done(&mut writer, &kv_store, &prefix_cache, model, &slot)
                        .await;
                    kv_store.clear_request(&slot.model_key, &slot.req_id_str);
                }
            }
            // Here, in the main loop and between messages, no forward of a
            // cancelled request can be running in this worker.
            release_caches_of_cancelled(&cancelled, &mut kv_released_for_cancel, &kv_store);
            flow_sessions.forget_cancelled(&cancelled);
            far_runs.forget_cancelled(&cancelled);
            // Sweep cancels that never matched a request (the request had
            // already finished when the cancel arrived).
            cancelled.retain(|_, at| at.elapsed().as_secs() < CANCEL_RETENTION_SECS);
            kv_released_for_cancel.retain(|id| cancelled.contains_key(id));
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
            // The store's TTL sweep. It was configured (`--kv-cache-ttl`)
            // and never run in a worker — only the daemon's own store was
            // swept — so a conversation nobody released (a segment served for
            // another computer, whose coordinator releases only its own
            // workers) stayed for the worker's whole life (report #005).
            kv_store.cleanup_expired();
            let occ = kv_store.occupancy();
            if occ.entries > 0 {
                tracing::debug!(
                    entries = occ.entries,
                    allocated_mb = occ.allocated_bytes / 1_000_000,
                    of_which_host_mb = occ.host_bytes / 1_000_000,
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
            let running = slot_table.len();
            let on_card = slot_table
                .current_layer_range()
                .and_then(|(ls, le)| models.get(&(ls, le, 0, 1)))
                .is_some_and(SplitModel::runs_entirely_on_card);
            let tick_started = std::time::Instant::now();
            let tick = step_decode_pool(
                &mut writer,
                &mut models,
                &kv_store,
                &prefix_cache,
                &mut slot_table,
                &mut prefill_pacer,
                force_standard_attn,
                batched_prefill_forward,
            )
            .await;
            if let Some(stalled) = card_pace.observe(
                tick_started.elapsed(),
                running,
                on_card,
                std::time::Instant::now(),
            ) {
                stalled.log("decode tick");
            }
            if let Err(e) = tick {
                tracing::warn!(error = %e, "model-worker: decode tick failed — finishing all slots with error");
                let drained = std::mem::replace(
                    &mut slot_table,
                    SlotTable::new(batch_generate_max_slots as usize),
                );
                // Each slot's prompt is still whole (`prompt_outlives`).
                let range = drained.current_layer_range();
                let failure = e.to_string();
                for slot in drained.into_active() {
                    if prompt_outlives(Some(failure.as_str())) {
                        let model = range.and_then(|(ls, le)| models.get(&(ls, le, 0, 1)));
                        save_slot_prompt_when_done(
                            &mut writer,
                            &kv_store,
                            &prefix_cache,
                            model,
                            &slot,
                        )
                        .await;
                    }
                    send_worker_error(
                        &mut writer,
                        slot.request_id,
                        SwarmError::Internal(format!("BatchGenerate decode failed: {e}")),
                    )
                    .await;
                }
            }
            // Read before the drain: an emptied table forgets its range.
            let range = slot_table.current_layer_range();
            for finished in slot_table.drain_finished() {
                let model = range.and_then(|(ls, le)| models.get(&(ls, le, 0, 1)));
                if let Err(e) =
                    finalize_slot(&mut writer, &kv_store, &prefix_cache, model, finished).await
                {
                    tracing::warn!(error = %e, "model-worker: finalize_slot failed");
                }
            }
        }
    }

    reader_task.abort();
    drop(models);
    tracing::info!("model-worker: exiting cleanly");
}

/// A pure-argmax verify's walk, from each row's argmax (`picks`, γ+1 of them):
/// the drafts kept while each equals its row's pick, then the pick where they
/// part — or the last row's, the bonus, after all of them. What
/// `sampling::walk_verified_positions` returns for a greedy request with no
/// penalties, from the picks alone (`the_argmax_walk_is_the_greedy_walk`).
fn argmax_walk(drafts: &[u32], picks: &[u32]) -> Vec<u32> {
    let mut kept = Vec::with_capacity(drafts.len() + 1);
    for (i, &q) in drafts.iter().enumerate() {
        if picks[i] != q {
            kept.push(picks[i]);
            return kept;
        }
        kept.push(q);
    }
    kept.push(picks[drafts.len()]);
    kept
}

/// Hand the card memory this worker's pool keeps unused back to the driver,
/// once it has been idle for `cuda_pool::IDLE_TRIM` — so another worker, or a
/// model the daemon wants to load, sees the room (`docs/FUTURE_WORK.md` #146).
/// The pool is the device's, shared by every model this process holds on the
/// card, so one trim covers them all.
fn hand_back_idle_card_memory(models: &HashMap<(usize, usize, usize, usize), SplitModel>) {
    let Some(device) = models
        .values()
        .map(SplitModel::device)
        .find(|d| d.is_cuda())
    else {
        return;
    };
    let started = std::time::Instant::now();
    if let Some((before, after)) = crate::inference::cuda_pool::trim(device) {
        tracing::info!(
            reserved_before_mb = before.reserved >> 20,
            reserved_after_mb = after.reserved >> 20,
            in_use_mb = after.used >> 20,
            took_ms = started.elapsed().as_millis() as u64,
            idle_secs = crate::inference::cuda_pool::IDLE_TRIM.as_secs(),
            "DIAG: card memory pool handed its unused memory back after idle"
        );
    }
    if let Some((before_mb, after_mb)) =
        crate::inference::cuda_graph::trim_idle_graph_memory(device)
    {
        if before_mb > after_mb {
            tracing::info!(
                before_mb,
                after_mb,
                "DIAG: CUDA graphs handed their idle card memory back"
            );
        }
    }
}

/// Send a `WorkerMsg::Error` back to the daemon. Used by the `run_worker`
/// dispatch loop to report handler failures without crashing the subprocess.
/// Loaded ranges that `incoming` entirely covers, and which are therefore
/// about to be redundant.
///
/// Strict subsumption only, and within the same tensor-parallel shape: a range
/// equal to `incoming` is the caller's own cache hit, and a partial overlap
/// describes layers each range still needs on its own.
fn subsumed_segment_keys<'a>(
    loaded: impl Iterator<Item = &'a (usize, usize, usize, usize)>,
    incoming: (usize, usize, usize, usize),
) -> Vec<(usize, usize, usize, usize)> {
    let (start, end, tp_rank, tp_size) = incoming;
    loaded
        .filter(|(ls, le, tr, ts)| {
            *tr == tp_rank
                && *ts == tp_size
                && *ls >= start
                && *le <= end
                && (*ls, *le) != (start, end)
        })
        .copied()
        .collect()
}

async fn send_worker_error(writer: &mut IpcWriter, request_id: uuid::Uuid, err: SwarmError) {
    let message = err.to_string();
    let fatal = crate::inference::worker_ipc::worker_error_is_fatal(&message);
    // Taken from the VARIANT, here, while it still exists. One line further on
    // it is a string, and the string is deliberately indistinguishable from a
    // peer's memory refusal.
    let local_memory_refusal = matches!(err, SwarmError::LocalMemoryUnavailable(_));
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
            local_memory_refusal,
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
        // about and ignore: `gpu_layers` layers of this worker's window go on
        // the card and the rest on the processor — which ones,
        // `hybrid::processor_run` says, and the loader logs. Until hybrid
        // placement existed this could only be honoured as "all of them",
        // which is why it used to warn.
        crate::inference::split::GPU_LAYER_LIMIT
            .store(gpu_layers as usize, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            gpu_layers,
            "model-worker: placing {gpu_layers} layers of this window on the graphics card \
             and the rest on the processor"
        );
    }
}

/// Apply `--cpu-moe-layers` to this process. Called once from `run_worker`;
/// the loader reads it through `split::cpu_expert_layers`, as it reads
/// `--gpu-layers` through `GPU_LAYER_LIMIT` — the daemon decides, the worker
/// obeys.
fn set_worker_cpu_moe_layers(layers: usize) {
    crate::inference::split::CPU_EXPERT_LAYERS.store(layers, std::sync::atomic::Ordering::Relaxed);
    if layers > 0 {
        tracing::info!(
            layers,
            "model-worker: keeping the routed experts of the first {layers} card-held layers \
             in RAM"
        );
    }
}

/// Should models load on the CPU regardless of GPU availability?
fn worker_force_cpu() -> bool {
    WORKER_FORCE_CPU.load(std::sync::atomic::Ordering::Relaxed)
}

/// Drop the cache of every cancelled request this worker has not released yet.
///
/// A cancelled request's conversation is never read again, so its cache goes
/// now rather than at the TTL. This is also how a node serving a SEGMENT
/// learns that a request ended: its coordinator cancels it on every peer of the
/// request once it is over (#238). Called from the main loop between messages,
/// where no forward of the request can be running; a forward still queued is
/// skipped by the cancel mark, which outlives this release. Once per cancel:
/// `released` remembers which marks have been acted on.
fn release_caches_of_cancelled(
    cancelled: &CancelledSet,
    released: &mut std::collections::HashSet<Uuid>,
    kv_store: &KvCacheStore,
) {
    for request_id in cancelled.iter().map(|e| *e.key()).collect::<Vec<_>>() {
        if released.insert(request_id) {
            kv_store.cleanup_request_id(&request_id.to_string());
        }
    }
}

/// Whole-device GPU memory in use, in MB, or `None` when there is no GPU to ask.
///
/// Whole-device rather than per-process because WSL reports `[N/A]` for
/// `--query-compute-apps` memory, so process attribution is simply unavailable
/// there. Sound enough to bracket a single load in a process that does one at a
/// time; not sound as a live admission input.
fn current_vram_used_mb() -> Option<u64> {
    crate::model::auto_manage::vram::nvidia_smi(&[
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])?
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
        DaemonMsg::Draft(d) => Some(d.request_id),
        DaemonMsg::HeadGuess(g) => Some(g.request_id),
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

/// `SWARMLLM_FAULT_LOAD_DELAY_SECS=<n>`: every model load in this node's
/// workers first waits `n` seconds — a cold load as slow as a busy machine's,
/// on demand. The network-level test for a requester waiting out a peer's load
/// (`examples/cold_load_test.sh`, FUTURE_WORK #129). Unset in production; read
/// once.
fn fault_load_delay_secs() -> Option<u64> {
    use std::sync::OnceLock;
    static SECS: OnceLock<Option<u64>> = OnceLock::new();
    *SECS.get_or_init(|| {
        std::env::var("SWARMLLM_FAULT_LOAD_DELAY_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|s| *s > 0)
    })
}

/// Loads this worker has timed and not yet reported, as `(bytes, ms)`.
/// Filled by [`ensure_model_loaded`] — the one place a model is loaded — and
/// drained by the main loop beside the card probe (`WorkerMsg::ModelLoadTimed`).
static LOAD_TIMINGS: std::sync::Mutex<Vec<(u64, u32)>> = std::sync::Mutex::new(Vec::new());

/// Cap on unreported timings, so a daemon that never drains (one that does
/// not read them) cannot grow this without bound.
const LOAD_TIMINGS_MAX: usize = 16;

fn record_load_timing(bytes: u64, took: std::time::Duration) {
    let ms = u32::try_from(took.as_millis()).unwrap_or(u32::MAX);
    if let Ok(mut q) = LOAD_TIMINGS.lock() {
        if q.len() < LOAD_TIMINGS_MAX {
            q.push((bytes, ms));
        }
    }
}

fn take_load_timings() -> Vec<(u64, u32)> {
    LOAD_TIMINGS
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
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
    // From here on this IS a load; the fault delay below stands in for a slow
    // disk, so it is timed like one.
    let load_started = std::time::Instant::now();
    if let Some(secs) = fault_load_delay_secs() {
        tracing::warn!(
            model = %model_id,
            secs,
            "FAULT INJECTION: delaying this model load (SWARMLLM_FAULT_LOAD_DELAY_SECS)"
        );
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }

    // A range that SUBSUMES ranges we already hold makes them redundant, and
    // holding both means holding those layers' weights twice.
    //
    // The map is keyed by the exact `(start, end, tp_rank, tp_size)`, so a plan
    // restating coverage this worker already has — [16..48) plus [0..16)
    // becoming [0..48) — is a miss, and the whole model is read from disk again
    // beside the copy already resident. Reported on a 16 GB processor-only node
    // (#010): one worker, never restarted, loaded [16..48), then [0..16), then
    // the full [0..48) sixty-three seconds later, ending in sustained swap.
    //
    // Dropped BEFORE the load, so the process holds `max(old, new)` rather than
    // their sum — the point of the fix is the peak, not just the steady state.
    // Only strict subsumption: a partial overlap ([0..20) against [16..48)) is
    // two ranges that each genuinely need their own layers, and dropping either
    // would cost a reload for nothing.
    for stale in subsumed_segment_keys(models.keys(), key) {
        models.remove(&stale);
        tracing::info!(
            model = %model_id,
            dropped = format!("[{}..{})", stale.0, stale.1),
            superseded_by = format!("[{layer_start}..{layer_end})"),
            "model-worker: dropping a layer range the incoming one already covers"
        );
    }

    // The reverse of the case above, which this cannot fix and must not hide.
    // A range CONTAINED IN one already resident is a miss on the exact key, so
    // the layers it shares with the resident copy are about to be read from
    // disk and held a second time — 19 of 48 layers in the reported case,
    // roughly 3.4 GB of a 14B on a machine with 16 GB (report #021).
    //
    // Serving it from the resident superset is not available: the layer loop
    // walks all of `self.layers` with no offset, and `is_first`/`is_last` are
    // baked in at load time from which components were loaded, so a resident
    // whole-model instance would run 48 layers and apply the output head to a
    // middle segment's hidden states. Dropping the superset instead is worse —
    // it is very likely serving other conversations whose KV entries are keyed
    // by its range, and they would silently read an empty cache.
    //
    // So this is reported rather than repaired. It is real memory, the charge
    // the daemon makes for it is accurate, and until now the only evidence was
    // two "Model loaded" lines a minute apart that looked like a retry.
    if let Some((rs, re)) = models
        .keys()
        .find(|(ls, le, tr, ts)| {
            *tr == tp_rank && *ts == tp_size && *ls <= layer_start && *le >= layer_end
        })
        .map(|(ls, le, _, _)| (*ls, *le))
    {
        tracing::warn!(
            model = %model_id,
            loading = format!("[{layer_start}..{layer_end})"),
            already_resident = format!("[{rs}..{re})"),
            duplicated_layers = layer_end.saturating_sub(layer_start),
            "model-worker: loading a layer range this worker already covers — its \
             layers will be held twice until one of the two is released"
        );
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

    let mut model = load_model_range(&ShardLoadParams {
        model_dir: &model_dir,
        shard_store: &shard_store,
        model_id,
        layer_start,
        layer_end,
        is_first,
        is_last,
        manifest: &manifest,
        force_cpu: worker_force_cpu(),
    })?;

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
        load_ms = load_started.elapsed().as_millis() as u64,
        "model-worker: Model loaded"
    );
    models.insert(key, model);
    // On the planner's basis — the model's size × the share of its layers —
    // so a rate measured here prices a load there in the same units.
    let bytes = manifest
        .total_size_bytes
        .saturating_mul(layer_end.saturating_sub(layer_start) as u64)
        / (total_layers.max(1) as u64);
    record_load_timing(bytes, load_started.elapsed());
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
        if !r.draft_tokens.is_empty() || !r.draft_parents.is_empty() || r.draft_tree_from.is_some()
        {
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
    prefix_cache: &Arc<PrefixCache>,
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
            prefix_cache,
            data_dir,
            fwd,
            slice,
            shard_window,
            activation_compression,
            // A batched forward is a decode step — one position per request
            // (`forward_is_schedulable` refuses a prompt pass) — so there is
            // nothing to chunk.
            0,
            // Nor does it carry a far run: a request learning one is never
            // batched (`ModelProcessPool::far_runs`).
            &mut FarRuns::default(),
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

    // One request in the batch with no conversation here must not be decoded
    // from nothing alongside the others. Declining the whole batch sends every
    // request through `handle_forward`, which refuses that one and runs the
    // rest — the batch is an optimisation, never a reason to skip the check.
    let model_key = model.kv_model_key();
    if let Some(lost) = requests.iter().find(|r| {
        forward_lacks_its_conversation(
            r.sequence_num,
            r.index_pos,
            kv_store.request_holds_state(model_key, &r.request_id.to_string()),
        )
    }) {
        return Err(SwarmError::Internal(format!(
            "fused batch declined — request {} has no conversation on this worker; \
             running the batch one request at a time",
            lost.request_id
        )));
    }

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
            let token_ids = wire_token_ids(slice, model.embedding_rows(), "Fused batch decode")?;
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
                prompt_blocks_stored: None,
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
                prompt_blocks_stored: None,
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

/// Is this forward being asked to continue a conversation this worker does not
/// hold?
///
/// Every forward after the prompt pass extends a request's cache, and its
/// attention reads everything the earlier forwards wrote here. If nothing was
/// written — the worker was restarted mid-reply, the cache expired, or the
/// prompt pass never ran on this machine — running it anyway attends over an
/// empty cache and returns a well-formed, plausible result that is GARBAGE.
/// Nothing downstream can tell: a split reply read `使用命令命令命令…` for 96
/// tokens with no error in any log, because the daemon had retired this
/// worker 14 times in one reply (`docs/FUTURE_WORK.md` #93, gotcha #690).
///
/// So it is refused. vLLM's rule for a preempted sequence is the same: its
/// cache is swapped back in or recomputed, never decoded without. Both the
/// single and the fused batch path ask this, and only this.
fn forward_lacks_its_conversation(sequence_num: u32, index_pos: u32, holds_state: bool) -> bool {
    sequence_num > 0 && index_pos > 0 && !holds_state
}

/// A first segment's token ids, read off a decode or check forward's bytes (8
/// bytes little-endian each) — the ONE reading, for the single and the fused
/// batch forward. `rows` is the embedding table's (`SplitModel::embedding_rows`).
///
/// An id the table has no row for is refused HERE, on the host. Whoever sent
/// the forward chose these ids — a peer coordinating a split, a requester's
/// guesses — and a card's row gather does not check them (the vendored
/// `quantized/cuda.rs` gather; its kernel's assert is a sticky device error,
/// which ends the worker for every request on it). The processor's gather
/// refused them already. Security review of #318, 2026-10-20.
fn wire_token_ids(bytes: &[u8], rows: Option<usize>, what: &str) -> Result<Vec<i64>, SwarmError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(8) {
        return Err(SwarmError::Internal(format!(
            "{what} payload must be a non-empty multiple of 8 bytes (got {})",
            bytes.len()
        )));
    }
    let rows = rows.ok_or_else(|| {
        SwarmError::Internal(format!("{what}: this segment holds no embedding table"))
    })?;
    let ids: Vec<i64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect();
    if let Some(bad) = ids
        .iter()
        .find(|&&id| usize::try_from(id).map_or(true, |id| id >= rows))
    {
        return Err(SwarmError::Internal(format!(
            "{what}: token id {bad} is outside the model's {rows} embedding rows"
        )));
    }
    Ok(ids)
}

/// The tree of guesses a forward applies (#134), checked: a decode-side check
/// (never a prompt pass), well formed (`DraftTree::from_wire`: parents before
/// children, at most `draft_tree::MAX_TREE_ROWS`), one entry per row this
/// forward reads, and — on the segment that answers — answered with token ids
/// (`unwalked_answer`): a tree's every row back as a vocabulary would be tens
/// of megabytes, so a coordinator asks for one only from a tail that walks or
/// draws. Anything else is the coordinator's mistake, a 400 — a segment whose
/// layers cannot read a tree (`layers_refuse`, `SplitModel::draft_tree_refusal`)
/// included, refused before anything runs so the coordinator can check the
/// same guesses as a chain.
///
/// A tree that FLOWS (`tree_from`, #134 (b)) starts at or before `index_pos`:
/// its rows before `index_pos` are in the cache from earlier forwards, so the
/// shape names `index_pos - tree_from` rows more than this forward reads — and
/// a tree's rows must end where the cache does, which the layers check.
fn draft_tree_to_apply(
    wire: &[u16],
    tree_from: Option<u32>,
    index_pos: usize,
    positions: usize,
    sequence_num: u32,
    unwalked_answer: bool,
    layers_refuse: Option<&'static str>,
) -> Result<Option<crate::inference::draft_tree::DraftTree>, SwarmError> {
    if wire.is_empty() {
        if tree_from.is_some() {
            return Err(SwarmError::Validation(
                "a tree that flows was sent with no shape".into(),
            ));
        }
        return Ok(None);
    }
    if let Some(why) = layers_refuse {
        return Err(SwarmError::Validation(format!(
            "this model cannot check a tree of guesses: {why}"
        )));
    }
    if sequence_num == 0 {
        return Err(SwarmError::Validation(
            "a prompt pass cannot carry a tree of guesses".into(),
        ));
    }
    if unwalked_answer {
        return Err(SwarmError::Validation(
            "a tree of guesses is walked by the segment that answers it".into(),
        ));
    }
    let base = match tree_from {
        None => index_pos,
        Some(from) if from as usize <= index_pos => from as usize,
        Some(from) => {
            return Err(SwarmError::Validation(format!(
                "a tree that flows from position {from} continued at {index_pos}"
            )));
        }
    };
    let tree = crate::inference::draft_tree::DraftTree::from_wire(base, wire)
        .map_err(SwarmError::Validation)?;
    let read_before = index_pos - base;
    if tree.len() != read_before + positions {
        return Err(SwarmError::Validation(format!(
            "a tree of {} rows sent with {positions} positions to read after {read_before} of it",
            tree.len()
        )));
    }
    Ok(Some(tree))
}

/// A tree that flows is answered with one draw per row (#134 (b),
/// `sampling::draw_each_row`), and a draw there reads no path's tokens — so a
/// request whose sampling penalises its own history is refused, a 400: its
/// requester checks such a request in rounds, where the tail walks each path
/// with its tokens.
fn refuse_draws_with_penalties(
    sampling: &crate::types::SamplingParams,
    generated_ids: &[u32],
) -> Result<(), SwarmError> {
    if sampling.frequency_penalty != 0.0
        || sampling.presence_penalty != 0.0
        || !generated_ids.is_empty()
    {
        return Err(SwarmError::Validation(
            "a tree that flows is drawn row by row, and penalties need each path's own tokens"
                .into(),
        ));
    }
    Ok(())
}

/// The rows a check's answer is made from, checked before the forward: the
/// tail computes the vocabulary over every row it is sent (608 KB a row at a
/// 152K vocabulary), so a coordinator naming thousands of positions had a
/// holder build gigabytes before the length was looked at, after the work
/// (security review of #134). A chain walked here reads one row per guess and
/// the row after the last; one answered with logits carries no more rows than
/// a coordinator reads back (`MAX_SPEC_LOGITS_POSITIONS`) — an honest check is
/// at most `config::MAX_GUESSES_PER_CHECK` + 1. A tree is checked by
/// [`draft_tree_to_apply`]. Anything else is the coordinator's mistake, a 400.
fn spec_rows_fit_the_answer(
    positions: usize,
    drafts: usize,
    walked_here: bool,
) -> Result<(), SwarmError> {
    let most = crate::network::protocol::MAX_SPEC_LOGITS_POSITIONS;
    if walked_here && positions != drafts + 1 {
        return Err(SwarmError::Validation(format!(
            "a check of {drafts} guesses sent with {positions} positions to read"
        )));
    }
    if !walked_here && positions > most {
        return Err(SwarmError::Validation(format!(
            "a check answered with logits reads at most {most} positions, not {positions}"
        )));
    }
    Ok(())
}

/// The picture spans a forward applies (#115 (2b)), checked: well formed
/// (`vision::spans_are_well_formed`), none ending past `forward_end` — the
/// position after this forward's last — none starting before `forward_start`
/// and ending after it (a picture cut by the forward's start; one wholly
/// before it is not read), and, where this is the first segment
/// splicing the rows itself (`first_segment`: its token ids and every
/// picture's rows together), exactly the spans its own markers give
/// (`vision::image_spans`). Anything else is the coordinator's mistake, a 400.
fn image_spans_to_apply(
    sent: &[crate::types::PromptSpan],
    forward_start: usize,
    forward_end: usize,
    first_segment: Option<(&candle_core::Tensor, usize)>,
) -> Result<Vec<std::ops::Range<usize>>, SwarmError> {
    let spans: Vec<std::ops::Range<usize>> = sent
        .iter()
        .map(|s| s.start as usize..s.end as usize)
        .collect();
    if spans.is_empty() {
        return Ok(spans);
    }
    if !crate::inference::vision::spans_are_well_formed(&spans)
        || spans.last().is_some_and(|s| s.end > forward_end)
        || spans
            .iter()
            .any(|s| s.start < forward_start && s.end > forward_start)
    {
        return Err(SwarmError::Validation(format!(
            "the picture positions sent with this prompt ({} of them) do not fit a \
             forward of positions {forward_start}..{forward_end}",
            spans.len()
        )));
    }
    if let Some((ids, rows)) = first_segment {
        let ids: Vec<i64> = ids
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(|e| SwarmError::Internal(format!("image prompt ids: {e}")))?;
        if crate::inference::vision::image_spans(&ids, rows) != spans {
            return Err(SwarmError::Validation(
                "the picture positions sent with this prompt are not where its pictures are".into(),
            ));
        }
    }
    Ok(spans)
}
/// The refusal for [`forward_lacks_its_conversation`].
///
/// `ServiceUnavailable`, because it is this machine that cannot serve the
/// step: with a remote segment in the plan the router re-plans the request
/// from the prompt while nothing has reached the client, and otherwise the
/// reply ends with an error — either is right, and an answer made from nothing
/// is not.
fn lost_conversation_error(
    request_id: uuid::Uuid,
    layers: (usize, usize),
    index_pos: u32,
) -> SwarmError {
    SwarmError::ServiceUnavailable(format!(
        "this computer no longer holds the conversation for request {request_id} \
         (layers {}..{}, position {index_pos}) — its model process was restarted or its \
         cache expired mid-reply, so it refuses rather than answer from nothing",
        layers.0, layers.1
    ))
}

#[allow(clippy::too_many_arguments)]
async fn handle_forward(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    fwd: IpcForward,
    mut activation_bytes: Vec<u8>,
    shard_window: &Option<Vec<u32>>,
    activation_compression: bool,
    // Ceiling on the positions a prompt pass runs per forward; a prompt
    // longer than this is run in chunks so a cancel can land between them
    // (gotcha #445). 0 disables chunking.
    prefill_chunk_tokens: usize,
    far_runs: &mut FarRuns,
) -> Result<(), SwarmError> {
    let request_id = fwd.request_id;
    let model_id = fwd.model_id.clone();
    let (layer_start, layer_end) = (fwd.layer_range.0 as usize, fwd.layer_range.1 as usize);
    // Where a split's per-token time goes outside the layers: this line's
    // timestamp against the daemon's send, the executor's `forward_ms`, and
    // `DIAG: worker forward answered` below (split overhead, 2026-09-29).
    let received = std::time::Instant::now();
    tracing::debug!(%request_id, index_pos = fwd.index_pos, "DIAG: worker forward received");

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

    // A chain whose two ends this worker holds (`FarRuns`): a decode forward
    // of its first segment starts past every position the last check kept,
    // so what the forwards before it kept is learned now — the prompt after
    // its first token, never before.
    let far_run = fwd
        .far_run
        .map(|(a, b)| (a as usize, b as usize))
        .filter(|_| tp_size == 1 && fwd.draft_parents.is_empty() && fwd.draft_tree_from.is_none());
    if far_run.is_some_and(|(a, _)| layer_start == 0 && layer_end == a) && fwd.sequence_num > 0 {
        far_runs.commit(&request_id, fwd.index_pos as usize, models);
    }

    let model = models
        .get_mut(&(layer_start, layer_end, tp_rank, tp_size))
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;

    let is_first = model.is_first();
    let is_last = model.is_last();
    // Which end of the far run this forward is, if either: the first
    // segment's output is kept, or the last segment's input.
    let far_side = far_run.and_then(|(a, b)| {
        if is_first && !is_last && layer_end == a {
            Some(true)
        } else if is_last && !is_first && layer_start == b {
            Some(false)
        } else {
            None
        }
    });
    let model_key = model.kv_model_key().to_string();
    let req_id_str = request_id.to_string();
    let pre_embedded = fwd.pre_embedded;

    if forward_lacks_its_conversation(
        fwd.sequence_num,
        fwd.index_pos,
        kv_store.request_holds_state(&model_key, &req_id_str),
    ) {
        return Err(lost_conversation_error(
            request_id,
            (layer_start, layer_end),
            fwd.index_pos,
        ));
    }

    // A prompt pass read in pieces (FUTURE_WORK #171): only its FIRST piece
    // starts the pass — clears, restores, is admitted — and every later piece
    // continues the cache the pieces before it built. Without a span the
    // forward is the whole pass, as it always was.
    let prompt_span = fwd.prompt_span.filter(|_| fwd.sequence_num == 0);
    let first_piece = prompt_span.is_none_or(|s| s.starts_at(fwd.index_pos));

    // Clear per-request KV-cache at the start of a new request (prefill). A
    // range admitted up front with the request's others keeps that admission:
    // it was made for the whole request (`DaemonMsg::AdmitPrompt`, #263).
    let admitted_up_front =
        fwd.sequence_num == 0 && kv_store.admitted_up_front(&req_id_str, fwd.layer_range);
    if fwd.sequence_num == 0 && first_piece {
        if admitted_up_front {
            kv_store.clear_request_caches(&model_key, &req_id_str);
        } else {
            kv_store.clear_request(&model_key, &req_id_str);
        }
    }

    // The owner's own segment of a request this node coordinates is the
    // owner's, as its `Generate` is: marked before any layer of it picks a
    // thread pool, released with the rest of the request's bookkeeping. Only
    // the daemon sets the flag; a peer's segment never carries it.
    //
    // AFTER the clear above, never before it: `clear_request` releases the
    // request's bookkeeping, owner mark included, so a mark made first was
    // wiped at every prompt pass — the only pass the owner's width is for.
    // Found by running a split request, not by the tests (2026-09-26).
    if fwd.for_the_owner {
        kv_store.mark_owner_request(&req_id_str);
    }

    // Speculative partial-accept KV fixup: coordinator may request truncation
    // of this request's KV cache to a specific length before the forward runs.
    // Discards trailing stale entries written during a prior verify round. A
    // truncate refused is the forward refused: the cache would hold positions
    // the reply does not, and the forward would write after them (it read
    // "proceeding without truncation" until 2026-10-25) — a model with
    // recurrent state refuses one past its last check (`SsmState::rewind_to`).
    if let Some(target_len) = fwd.truncate_kv_to {
        kv_store
            .truncate_request_to(&model_key, &req_id_str, target_len as usize)
            .map_err(|e| {
                SwarmError::ServiceUnavailable(format!(
                    "this computer cannot take request {request_id} back to position \
                     {target_len}: {e}"
                ))
            })?;
        tracing::debug!(
            request_id = %request_id,
            target_len,
            "DIAG: truncated KV cache for speculative partial accept"
        );
    }

    // A streamed verify must continue exactly where the cache ends: the cache
    // writes at its OWN length while RoPE rotates by `index_pos`, so a chunk
    // that arrived out of its stream's order would be read into the wrong
    // positions with nothing failing. The serving daemon runs a stream in
    // order (`daemon::dispatch::forward_stream`); this is the check that it
    // did, refused like the drafter's own (`SplitModel::draft_after`).
    //
    // A later piece of a prompt pass must continue exactly where the cache
    // ends too, streamed or not (this node's own segment is driven without a
    // stream): a worker that lost the conversation mid-pass, or a piece left
    // over from a pass that was abandoned and started again, holds something
    // else. The FIRST piece is exempt — it has just cleared the cache, and a
    // resumed one restores its opening below.
    let continues = if fwd.sequence_num == 0 {
        !first_piece
    } else {
        fwd.stream_seq.is_some()
    };
    if continues {
        let held = kv_store.request_positions(&model_key, &req_id_str);
        if held != fwd.index_pos as usize {
            return Err(SwarmError::ServiceUnavailable(if fwd.sequence_num == 0 {
                format!(
                    "this computer holds {held} positions of request {request_id}, not the {} \
                     the next piece of its prompt continues from",
                    fwd.index_pos
                )
            } else {
                format!(
                    "this computer holds {held} positions of request {request_id}, not the {} \
                     its next streamed check continues from — the stream arrived out of order",
                    fwd.index_pos
                )
            }));
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
        // A prompt pass carrying a split prompt-cache hint is sent the prompt's
        // TOKEN IDS from `resume_at` (FUTURE_WORK #10): the coordinator keyed
        // the stored blocks by those ids, and text re-tokenized here could
        // split differently. It reads below as a multi-position decode input.
        if first_segment_input_is_text(
            fwd.index_pos,
            fwd.prompt_cache.is_some(),
            prompt_span.is_some(),
        ) {
            // Prefill: activations are the prompt text → tokenize. A prompt
            // sent with image rows is tokenized with its markers, one id per
            // image for `forward_multimodal` to replace — the coordinator
            // counted its positions the same way (`vision::encode_prompt_with_images`).
            let prompt = String::from_utf8_lossy(&activation_bytes);
            let carries_images = !vision_bytes.is_empty();
            let token_ids: Vec<i64> = match model.tokenizer() {
                Some(tokenizer) if carries_images => {
                    crate::inference::vision::encode_prompt_with_images(tokenizer, &prompt)
                }
                Some(tokenizer) => tokenizer.encode(&prompt),
                None if carries_images => {
                    return Err(SwarmError::Inference(
                        "this computer cannot read the model's vocabulary, so it cannot \
                         place the picture in the prompt"
                            .into(),
                    ))
                }
                None => prompt.bytes().map(|b| b as i64).collect(),
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
            let token_ids = wire_token_ids(
                &activation_bytes,
                model.embedding_rows(),
                "Decode step activation",
            )?;
            let seq_len = token_ids.len();
            candle_core::Tensor::from_vec(token_ids, &[1, seq_len], &candle_core::Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?
        }
    } else {
        split::bytes_to_tensor(&activation_bytes)?
    };
    // The far run's far end: what the last segment is handed.
    let far_input = far_side.filter(|near| !near).map(|_| input_tensor.clone());

    // Decompress vision embeddings if present (`vision::decode_wire_embeddings`).
    // An unreadable payload fails the forward: it used to be dropped with a
    // warning, and the reply then described a picture the model never saw.
    let vision_tensor: Option<candle_core::Tensor> = if vision_bytes.is_empty() {
        None
    } else {
        Some(crate::inference::vision::decode_wire_embeddings(
            &vision_bytes,
        )?)
    };

    // The LoRA adapter this request asked for, if any — loaded from its
    // registration (`<data_dir>/adapters/<id>.adapter.json`) and cached for
    // this worker's life, so a reply does not re-read the file per token.
    //
    // A failure is the REQUEST's failure, never a quiet fallback: this used to
    // log "proceeding without" and answer from the base model, which is the
    // exact outcome of `docs/FUTURE_WORK.md` #110. The id is checked as a safe
    // file name inside `lora` (gotcha #94) — a peer's forward never carries
    // one (both binary decoders set `adapter_id: None`), but the check does
    // not rely on that.
    let lora_adapter = match fwd.adapter_id.as_deref() {
        Some(adapter_id) => Some(
            crate::model::lora::cached_registered_adapter(
                &data_dir.join("adapters"),
                adapter_id,
                model.device(),
                model.lora_qk_row_order(),
            )
            .map_err(|e| SwarmError::Validation(format!("LoRA adapter '{adapter_id}': {e}")))?,
        ),
        None => None,
    };

    // Whole-prompt admission for a SEGMENT's prompt pass (gotcha #447).
    //
    // `ensure_room_for_prompt` guarded the `Generate` path only; a segment's
    // prompt pass was admitted chunk by chunk (`claim_room` at each growth
    // boundary), so the card filled up until attention's transient
    // allocation — not the cache — hit the wall: a 6 GB card handed 24 layers
    // of an 8,111-token prompt died 22 s in with `CUDA_ERROR_OUT_OF_MEMORY`
    // and no standby. Decided here, before any layer runs, the refusal is a
    // 503 at token 0 that the coordinator fails over from.
    //
    // The prompt pass is `sequence_num == 0` (the work KIND, gotcha #434); a
    // decode step of the same segment is one position and skips this. The
    // positions are the input's sequence axis whichever form it arrived in —
    // `[1, tokens]` for a first segment, `[1, positions, hidden]` for hidden
    // states. Tensor-parallel phases and speculative verify rounds keep their
    // own shape and are left alone.
    let tp_meta = fwd.tp_meta.clone();
    // A split's prompt pass that resumes from what this segment stored on an
    // earlier turn (FUTURE_WORK #10): the stored opening is looked up BEFORE
    // admission and copied in AFTER it, the generate path's order — the copy is
    // itself an allocation the admission must have room for. A miss is
    // refused: the forward carries only the positions after `resume_at`, so
    // computing it without them would be silently wrong.
    let prompt_cache = fwd.prompt_cache.clone();
    let restored = match &prompt_cache {
        Some(hint) if fwd.sequence_num == 0 && first_piece && hint.resume_at > 0 => {
            if fwd.index_pos != hint.resume_at {
                return Err(SwarmError::Internal(format!(
                    "a prompt pass resuming at {} was sent from position {}",
                    hint.resume_at, fwd.index_pos
                )));
            }
            Some(
                prefix_cache
                    .lookup_chain(&model_key, hint.block_tokens, &hint.keys, hint.resume_at as usize)
                    .ok_or_else(|| {
                        SwarmError::PromptCacheMiss(format!(
                            "this computer no longer holds the first {} positions of this conversation",
                            hint.resume_at
                        ))
                    })?,
            )
        }
        _ => None,
    };
    // An image prompt's ids hold one marker per image, each of which becomes
    // that image's rows: the positions the pass will hold are the merged
    // sequence's, the count the coordinator made (`vision::positions_with_images`).
    let input_positions = match &vision_tensor {
        Some(rows) if is_first && !pre_embedded => {
            let ids: Vec<i64> = input_tensor
                .flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| SwarmError::Internal(format!("image prompt ids: {e}")))?;
            crate::inference::vision::positions_with_images(&ids, rows.dims()[0])
        }
        _ => input_tensor.dims().get(1).copied().unwrap_or(0),
    };
    if let Some(span) = prompt_span {
        if !span.holds(fwd.index_pos, input_positions as u32) {
            return Err(SwarmError::Validation(format!(
                "a piece of {input_positions} positions at {} lies outside its prompt pass {}..{}",
                fwd.index_pos, span.start, span.end
            )));
        }
    }
    // Where each picture's rows sit, for a family that reads them both ways
    // (#115 (2b)). They come from the coordinator, so they are checked before
    // a mask is built from them: well formed, none past the positions this
    // forward ends at, and on the first segment exactly where this prompt's
    // own markers put the rows it splices — a coordinator that tokenized
    // differently would otherwise have one picture's rows read with another's.
    let image_spans = image_spans_to_apply(
        &fwd.image_spans,
        fwd.index_pos as usize,
        fwd.index_pos as usize + input_positions,
        match &vision_tensor {
            Some(rows) if is_first && !pre_embedded => Some((&input_tensor, rows.dims()[0])),
            _ => None,
        },
    )?;
    // A tree of guesses (#134): its shape comes from the coordinator, so it is
    // checked against the rows it shapes before any forward reads it.
    let draft_tree = draft_tree_to_apply(
        &fwd.draft_parents,
        fwd.draft_tree_from,
        fwd.index_pos as usize,
        input_positions,
        fwd.sequence_num,
        is_last && !fwd.spec_walk_at_tail,
        model.draft_tree_refusal(),
    )?;
    // A tree that flows is answered with one draw per row (#134 (b)), which
    // reads no path's tokens: a request whose sampling penalises its own
    // history cannot be drawn so, and its requester keeps it on rounds.
    let draws_each_row = want_spec_output && fwd.draft_tree_from.is_some();
    if draws_each_row {
        refuse_draws_with_penalties(&fwd.sampling, &fwd.generated_ids)?;
    }
    // The rows of this forward's tree read in earlier forwards — answered then.
    let read_before = draft_tree
        .as_ref()
        .map_or(0, |tree| tree.len().saturating_sub(input_positions));
    if want_spec_output && draft_tree.is_none() {
        spec_rows_fit_the_answer(
            input_positions,
            fwd.draft_tokens.len(),
            fwd.spec_walk_at_tail,
        )?;
    }
    // The positions this pass will hold once every piece has run: the whole
    // span, admitted at its first piece (gotcha #447 — a pass admitted piece by
    // piece fills a card until attention's transient allocation fails
    // mid-pass); a later piece was admitted with it and asks nothing.
    let pass_positions = match prompt_span {
        Some(span) if first_piece => span.end.saturating_sub(fwd.index_pos) as usize,
        Some(_) => 0,
        None => input_positions,
    };
    if fwd.sequence_num == 0 && tp_meta.is_none() && !want_spec_output {
        let positions = pass_positions;
        if positions > 0 {
            refuse_a_prompt_past_the_served_context(
                model.context_window(),
                fwd.index_pos as usize,
                positions,
            )?;
            // A segment never sees the reply's budget, so it reserves the
            // whole quantum it always did. A resumed prompt holds its restored
            // opening as well as the positions it computes.
            let restoring = restored.as_ref().map_or(0, |s| s.token_count);
            // Admitted up front, for every range of this request at once, before
            // the pipeline sent anything: asking again would weigh the request
            // against its own promise.
            if !admitted_up_front {
                ensure_room_for_prompt(
                    model,
                    kv_store,
                    prefix_cache,
                    &req_id_str,
                    restoring + positions,
                    REPLY_RESERVE_POSITIONS,
                    width_handed_to_the_caches(
                        tp_meta.is_some(),
                        pre_embedded,
                        vision_tensor.is_some(),
                        input_positions,
                    ),
                )?;
            }
        }
    }
    if let Some(snapshot) = &restored {
        let held = prefix_cache
            .hydrate_request_from_snapshot(
                kv_store,
                &model_key,
                &req_id_str,
                snapshot,
                &model.cache_layouts(),
            )
            .map_err(|e| {
                // The detail stays here: this message crosses to the
                // coordinator through `sanitize_peer_facing_error`, which reads
                // any mention of a shard as missing shards — a retraction, not
                // the retry a miss gets.
                tracing::warn!(request_id = %req_id_str, error = %e,
                    "restoring a split's stored prompt failed — the coordinator reads it again from the start");
                SwarmError::PromptCacheMiss("restoring the stored opening failed".into())
            })?;
        if held != fwd.index_pos as usize {
            return Err(SwarmError::PromptCacheMiss(format!(
                "restored {held} positions, not the {} the prompt resumes from",
                fwd.index_pos
            )));
        }
    }

    // The far run's near end: what the first segment hands on.
    let far_output: std::cell::RefCell<Option<candle_core::Tensor>> = Default::default();
    // Run forward pass — CPU-bound, use block_in_place. A tree of guesses
    // shapes every forward the pass makes (`SplitModel::with_draft_tree`).
    let compute = |model: &mut SplitModel| -> Result<crate::types::LayerResult, String> {
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
            // A pure-argmax walk — greedy, no penalties, no shared noise —
            // is chosen on the card: the row's argmax IS what the sampler
            // would pick (`sample_token_with_ctx`, temperature 0, penalties
            // off), so only γ+1 token ids come back instead of every row
            // (2.4 MB for four at a 152K vocabulary, plus a copy per row
            // and a scan of each on the processor). The non-finite guard
            // stays: one summed scalar, which any NaN or ±inf poisons.
            if fwd.spec_walk_at_tail
                && output_t.device().is_cuda()
                && fwd.sampling.temperature <= 0.0
                && fwd.sampling.frequency_penalty == 0.0
                && fwd.sampling.presence_penalty == 0.0
                && fwd.coupling_seed.is_none()
            {
                let total: f32 = output_t
                    .sum_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::F32))
                    .and_then(|t| t.to_scalar())
                    .map_err(|e| format!("spec walk on the card: {e}"))?;
                if !total.is_finite() {
                    return Err("spec walk: non-finite logits".into());
                }
                // The first of equal maxima, as every other greedy path picks —
                // never candle's `argmax`, whose card kernel breaks a tie by
                // thread order (`tree_rank::greedy_picks`).
                let rows = output_t
                    .squeeze(0)
                    .map_err(|e| format!("spec walk on the card: {e}"))?;
                #[cfg(feature = "candle-cuda")]
                let picks: Vec<u32> = crate::inference::tree_rank::greedy_picks(&rows)
                    .map_err(|e| format!("spec walk on the card: {e}"))?
                    .ok_or_else(|| "spec walk: non-finite logits".to_string())?;
                // A build without the card kernels has no card tensor to reach
                // this branch; the rows are picked on the host all the same.
                #[cfg(not(feature = "candle-cuda"))]
                let picks: Vec<u32> = rows
                    .to_dtype(candle_core::DType::F32)
                    .and_then(|t| t.to_vec2::<f32>())
                    .map_err(|e| format!("spec walk on the card: {e}"))?
                    .iter()
                    .map(|row| crate::inference::sampling::argmax(row))
                    .collect();
                let token_ids = match draft_tree.as_ref() {
                    // A tree that flows: every row's own argmax, walked by the
                    // requester.
                    Some(_) if draws_each_row => picks,
                    Some(tree) => crate::inference::sampling::argmax_walk_tree(
                        tree,
                        &fwd.draft_tokens,
                        &picks,
                    )?,
                    None => {
                        if picks.len() != fwd.draft_tokens.len() + 1 {
                            return Err(format!(
                                "spec walk: {} positions verified for {} drafts",
                                picks.len(),
                                fwd.draft_tokens.len()
                            ));
                        }
                        argmax_walk(&fwd.draft_tokens, &picks)
                    }
                };
                return Ok(crate::types::LayerResult {
                    request_id,
                    token_ids,
                    finish_reason: None,
                    activations: vec![],
                    sealed_token_ids: None,
                    spec_logits: Vec::new(),
                    matched_stop_sequence: None,
                    token_logprobs: Vec::new(),
                    locally_constructed: false,
                    refusal: None,
                    answers_step: None,
                    prompt_blocks_stored: None,
                });
            }
            // Row i predicts the token at `index_pos + 1 + i` (the verify's
            // first token is written at `index_pos`) — the absolute position
            // the shared noise is keyed by. A tree that flows starts before
            // this forward: its rows' draws are keyed from the tree's start,
            // as the drafter grew them.
            let noise = fwd
                .coupling_seed
                .map(crate::inference::coupled_noise::CoupledNoise::new);
            let tree_start = fwd.draft_tree_from.unwrap_or(fwd.index_pos);
            let coupling = noise.as_ref().map(|n| (n, u64::from(tree_start) + 1));
            // A sampled tree that flows: each row's draw reads only its top k,
            // which the card ranks, so only those come back (2026-10-23).
            #[cfg(feature = "candle-cuda")]
            if let (true, Some(tree), Some(coupling)) = (
                fwd.spec_walk_at_tail && draws_each_row && output_t.device().is_cuda(),
                draft_tree.as_ref(),
                coupling,
            ) {
                let rows = output_t
                    .squeeze(0)
                    .map_err(|e| format!("spec draws on the card: {e}"))?;
                if let Some(token_ids) = crate::inference::tree_rank::draw_each_row_on_the_card(
                    &rows,
                    tree,
                    read_before,
                    &fwd.sampling,
                    coupling,
                )? {
                    return Ok(crate::types::LayerResult {
                        request_id,
                        token_ids,
                        finish_reason: None,
                        activations: vec![],
                        sealed_token_ids: None,
                        spec_logits: Vec::new(),
                        matched_stop_sequence: None,
                        token_logprobs: Vec::new(),
                        locally_constructed: false,
                        refusal: None,
                        answers_step: None,
                        prompt_blocks_stored: None,
                    });
                }
            }
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
            // Walk the drafts HERE when asked: the coordinator gets the
            // tokens it would have derived — by the same function — for a
            // few bytes instead of a vocabulary per position (608 KB each at
            // 152K), which on a long link cost more than the round trips
            // speculation saved (`docs/plans/split_speculation.md`).
            let token_ids = if fwd.spec_walk_at_tail {
                match draft_tree.as_ref() {
                    Some(tree) if draws_each_row => crate::inference::sampling::draw_each_row(
                        tree,
                        read_before,
                        &spec_logits,
                        &fwd.sampling,
                        coupling,
                    )?,
                    Some(tree) => crate::inference::sampling::walk_verified_tree(
                        tree,
                        &fwd.draft_tokens,
                        &spec_logits,
                        &fwd.sampling,
                        &fwd.generated_ids,
                        coupling,
                    )?,
                    None => crate::inference::sampling::walk_verified_positions(
                        &fwd.draft_tokens,
                        &spec_logits,
                        &fwd.sampling,
                        &fwd.generated_ids,
                        coupling,
                    )?,
                }
            } else {
                Vec::new()
            };
            let spec_logits = if fwd.spec_walk_at_tail {
                Vec::new()
            } else {
                spec_logits
            };
            return Ok(crate::types::LayerResult {
                request_id,
                token_ids,
                finish_reason: None,
                activations: vec![],
                sealed_token_ids: None,
                spec_logits,
                matched_stop_sequence: None,
                token_logprobs: Vec::new(),
                locally_constructed: false,
                refusal: None,
                answers_step: None,
                prompt_blocks_stored: None,
            });
        }

        // TP single-layer forward: process one layer in AttnOnly or FfnOnly
        // phase. Every arm reads a picture's rows as a whole where the
        // forward carries its span (#115 (2b)); a coordinator never sends
        // one to a tensor-parallel group.
        let output = model.with_image_spans(&image_spans, |model| -> Result<_, String> {
            Ok(if let Some(ref tp) = tp_meta {
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
                // The adapter as on the token path: an adapter changes the
                // layers' projections, never the embedding table
                // (`model::lora`), so an input embedded elsewhere is the one
                // it expects. `None` here answered a LoRA request from the
                // base model, without a word (#316's review).
                model
                    .forward_prompt_in_chunks(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                        lora_adapter.as_deref(),
                        true,
                        prefill_chunk_tokens,
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
                // In chunks when the prompt is longer than one, so a cancelled
                // request stops at the next boundary even on a one-layer
                // segment; a decode step is one position and takes the
                // one-shot path unchanged.
                model
                    .forward_prompt_in_chunks(
                        &input_tensor,
                        fwd.index_pos as usize,
                        kv_store,
                        &req_id_str,
                        lora_adapter.as_deref(),
                        false,
                        prefill_chunk_tokens,
                    )
                    .map_err(|e| format!("Forward: {e}"))?
            })
        })?;

        if is_last && tp_meta.is_none() {
            let (token_id, token_logprob) = if fwd.sampling.logprobs {
                crate::inference::tensor_util::sample_token_with_logprob_history(
                    &output,
                    &fwd.sampling,
                    &fwd.generated_ids,
                    // A forward's sampling crosses the IPC without a grammar:
                    // a split is not held to one yet (#336).
                    None,
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
                locally_constructed: false,
                refusal: None,
                answers_step: None,
                prompt_blocks_stored: None,
            })
        } else {
            if far_side == Some(true) {
                *far_output.borrow_mut() = Some(output.clone());
            }
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
                locally_constructed: false,
                refusal: None,
                answers_step: None,
                prompt_blocks_stored: None,
            })
        }
    };
    // A chain-shaped tree — a check of no guesses after kept tokens read again
    // — reads what an ordinary forward does, and runs as one (captured, flash);
    // its walk still starts where its guesses would have.
    let compute_result = tokio::task::block_in_place(|| match draft_tree.clone() {
        Some(tree) if !tree.is_chain() => model
            .with_draft_tree(tree, compute)
            .map_err(|e| e.to_string())
            .and_then(|r| r),
        _ => compute(model),
    });

    let mut result = compute_result.map_err(SwarmError::Internal)?;
    if let (Some(run), Some(near)) = (far_run, far_side) {
        let kept = if near {
            far_output.into_inner()
        } else {
            far_input
        };
        if let Some(h) = kept {
            let range = (layer_start, layer_end, tp_rank, tp_size);
            far_runs.keep(
                request_id,
                run,
                (near, range),
                fwd.index_pos as usize,
                h,
                model.device(),
            );
        }
    }
    // Keep this segment's part of a split's prompt for the next turn
    // (FUTURE_WORK #10): whole blocks of what the prompt pass left in the
    // cache, sized to the room beside the live cache as the generate path's
    // snapshot is (gotcha #440), and answer how many were kept — the most the
    // coordinator may resume from here next turn.
    // A pass read in pieces is stored once, at its FINAL piece: earlier ones hold
    // only part of the prompt.
    let final_piece =
        prompt_span.is_none_or(|s| s.ends_with(fwd.index_pos, input_positions as u32));
    if let Some(hint) = prompt_cache
        .as_ref()
        .filter(|_| fwd.sequence_num == 0 && final_piece)
    {
        let held = kv_store.request_positions(&model_key, &req_id_str);
        let room = snapshot_room(model, kv_store, prefix_cache, held, ReplyAhead::ToDecode);
        let blocks = prefix_cache.insert_chain_from_kv(
            &model_key,
            &req_id_str,
            kv_store,
            hint.block_tokens,
            &hint.keys,
            room,
        );
        // Stored blocks kept beside the live cache are device memory the
        // head-room guard must see.
        if prefix_cache_charged() {
            kv_store.set_external_reserved(prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64);
        }
        result.prompt_blocks_stored = Some(blocks);
    }
    let computed = std::time::Instant::now();

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
        prompt_blocks_stored: result.prompt_blocks_stored,
    };

    send_worker(writer, &WorkerMsg::LayerResult(ipc_result), &payload)
        .await
        .map_err(|e| SwarmError::Internal(format!("send LayerResult: {e}")))?;
    tracing::debug!(
        %request_id,
        received_to_computed_ms = format_args!("{:.2}", (computed - received).as_secs_f64() * 1000.0),
        computed_to_sent_ms = format_args!("{:.2}", computed.elapsed().as_secs_f64() * 1000.0),
        "DIAG: worker forward answered"
    );

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
    // The request's caller asked to keep it on this computer (#251 (2b)):
    // the probe says so, and the daemon then asks no peer.
    keep: KeepAsked,
) -> usize {
    let block_size = prefix_cache.block_tokens();
    // A peer's snapshot carries attention layers only (the wire format), and
    // a recurrent layer cannot be restored beside them (#228) — no peer
    // announces one for such a model, so asking would only wait.
    if block_size == 0 || model.carries_recurrent_state() {
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
            keep_here: keep.is_kept(),
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
        &model.cache_layouts(),
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
///
/// Every model qualifies, on either device. One with recurrent state (Qwen
/// 3.5) takes a rejected guess back by replaying the check's accepted rows
/// (`SsmState::rewind_to`, #228 (5)); on a card its checks are captured as a
/// dense model's are (`SplitModel::mask_is_read`).
pub(crate) fn ngram_spec_eligible(
    sampling: &crate::types::SamplingParams,
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    swift_cfg: &SwiftConfig,
) -> bool {
    ngram_cfg.num_pred_tokens > 0
        && ngram_cfg.max_ngram_size >= ngram_cfg.min_ngram_size
        && !sampling.logprobs
        && !(swift_cfg.enabled && sampling.temperature == 0.0)
        // A guess is drawn without the reply's grammar, and one taken back
        // would leave its matcher ahead of the reply (#336).
        && sampling.reply_grammar.is_none()
}

/// The reply's grammar compiled for `model` (#336), or `None` for a reply held
/// to nothing — built by both generate paths before the prompt is read, so a
/// grammar this model cannot hold costs no prompt pass.
fn held_reply_for(
    model: &SplitModel,
    sampling: &crate::types::SamplingParams,
) -> Result<Option<crate::inference::reply_grammar::HeldReply>, SwarmError> {
    let Some(grammar) = sampling.reply_grammar.as_deref() else {
        return Ok(None);
    };
    let vocab = model.grammar_vocab()?;
    crate::inference::reply_grammar::HeldReply::new(&vocab, grammar).map(Some)
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
/// diverts, measures, and the answer steers everything after it — but not for
/// good. Only a diverted request records a payoff, so a refusal was a latch:
/// a 14B split's first reply, prose at 1.37 tokens a round, turned speculation
/// (and the card's guesses with it) off for every later request of that
/// worker, code at 1.4x included (2026-10-28, gotcha #985). Each refused solo
/// request now moves the payoff back toward the bar ([`drift_toward_retrying`]),
/// so a poor stretch is measured again by ONE request rather than kept for the
/// worker's life. vLLM decides by concurrency instead
/// (`speculative_disable_by_batch_size`), never by a latched payoff.
fn spec_payoff_justifies_diverting() -> bool {
    use std::sync::atomic::Ordering;
    let seen = SPEC_PAYOFF_X100.load(Ordering::Relaxed);
    if payoff_justifies_diverting(seen) {
        return true;
    }
    SPEC_PAYOFF_X100.store(drift_toward_retrying(seen), Ordering::Relaxed);
    false
}

/// How far one refused solo request moves the payoff back up, ×100: from 1.37
/// tokens a round, every 7th solo request measures again; from 1.05 (nothing to
/// copy, no card to guess), every 23rd.
const SPEC_PAYOFF_DRIFT_X100: u32 = 2;

/// The payoff after one more refusal — never past the bar, so the request that
/// reaches it measures, and its own result decides what follows.
fn drift_toward_retrying(seen_x100: u32) -> u32 {
    seen_x100
        .saturating_add(SPEC_PAYOFF_DRIFT_X100)
        .min(SPEC_PAYOFF_WORTH_DIVERTING_X100)
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
/// Verifies `draft` — the guesses [`LocalSpec::next`] found in an n-gram match
/// against the prompt and the generation tail, and judged worth checking — in
/// ONE forward. There is no draft model: the guess is that text already in the
/// context will recur, which is what SwarmLLM's own workload does constantly
/// (tool schemas, code, retrieved passages).
///
/// Returns the tokens produced and how many KV positions were committed. The
/// two are deliberately NOT the same number: a round commits `next_token` plus
/// every accepted draft token, and the last token it produces is not in the
/// cache yet. That is the same invariant the plain loop keeps, where one
/// forward commits one position and yields one not-yet-cached token — which is
/// why the caller can drain the extras without forwarding again.
///
/// **A model with recurrent state is taken back the same way** (#228 (5)):
/// its DeltaNet layers summarise every token the check read, and a truncate
/// into the check replays the accepted rows from the state the check read
/// (`SsmState::rewind_to` — only the convolution and the delta rule, from the
/// inputs the check kept). Until 2026-10-25 such a check was taken back WHOLE
/// and its accepted tokens read again at the head of the next round, which a
/// split's peers could not do.
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
    draft: &[u32],
    generated: &[u32],
    sampling: &crate::types::SamplingParams,
    force_attn: bool,
    capture: Option<&std::collections::HashSet<usize>>,
) -> Result<SpecRound, SwarmError> {
    let started = std::time::Instant::now();
    let capture = capture.map(|layers| crate::inference::split::Capture {
        layers,
        last: usize::MAX,
    });
    // No match: do exactly what the plain loop does. A miss must cost nothing
    // beyond the lookup itself, or speculation becomes a tax on the workloads
    // it cannot help.
    if draft.is_empty() {
        let input = model.token_tensor(next_token)?;
        let (logits, caught) = tokio::task::block_in_place(|| {
            let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
            match capture {
                Some(capture) => model
                    .forward_keeping(&input, index_pos, kv_store, req_id_str, capture, false)
                    .map(|(logits, caught)| (logits, Some(caught))),
                None => model
                    .forward(&input, index_pos, kv_store, req_id_str)
                    .map(|logits| (logits, None)),
            }
        })?;
        // `None`: a held reply is never speculated (`ngram_spec_eligible`).
        let (tok, _) = crate::inference::tensor_util::sample_token_with_logprob_history(
            &logits, sampling, generated, None,
        )?;
        return Ok(SpecRound {
            produced: vec![tok],
            committed: 1,
            positions: 1,
            ms: started.elapsed().as_secs_f64() * 1000.0,
            proposed: 0,
            accepted: 0,
            caught,
        });
    }

    let mut ids = Vec::with_capacity(draft.len() + 1);
    ids.push(next_token);
    ids.extend_from_slice(draft);
    let input = model.tensor_from_ids(&ids)?;
    let (all, caught) = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
        match capture {
            Some(capture) => model
                .forward_keeping(&input, index_pos, kv_store, req_id_str, capture, true)
                .map(|(all, caught)| (all, Some(caught))),
            None => model
                .forward_verify_all_positions(&input, index_pos, kv_store, req_id_str)
                .map(|all| (all, None)),
        }
    })?;

    use candle_core::IndexOp;
    let mut hist: Vec<u32> = generated.to_vec();
    let mut produced: Vec<u32> = Vec::with_capacity(draft.len() + 1);
    let mut accepted = 0usize;
    for i in 0..=draft.len() {
        let row = all
            .i((0, i))
            .and_then(|t| t.unsqueeze(0))
            .map_err(SwarmError::internal)?;
        let (tok, _) = crate::inference::tensor_util::sample_token_with_logprob_history(
            &row, sampling, &hist, None,
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

    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let kept = 1 + accepted;
    // Drop the cache entries written for drafts that were not accepted. Without
    // this the next forward would attend over positions holding tokens the
    // reply does not contain — a silent corruption, not an error.
    kv_store.truncate_request_to(model_key, req_id_str, index_pos + kept)?;
    Ok(SpecRound {
        produced,
        committed: kept,
        positions: ids.len(),
        ms,
        proposed: draft.len(),
        accepted,
        caught,
    })
}

/// What one [`ngram_spec_round`] did.
#[derive(Debug)]
struct SpecRound {
    /// The accepted guesses and the token after them, in order.
    produced: Vec<u32>,
    /// Cache positions it added.
    committed: usize,
    /// Positions its forward read, and how long the forward and the sampling
    /// took — what [`LocalSpec`] prices the next round's guesses from.
    positions: usize,
    ms: f64,
    /// Guesses checked; 0 for a round that found none, was paused, or found
    /// some not worth checking — which proves nothing about whether guessing
    /// works ([`SpecBackoff::record`]).
    proposed: usize,
    /// Guesses accepted.
    accepted: usize,
    /// The hidden states the forward was asked to keep, every position it
    /// read (`[1, positions, hidden]`); the first `committed` are the reply's.
    caught: Option<std::collections::HashMap<usize, candle_core::Tensor>>,
}

/// One request's local speculation on the decode loop: what the lookup
/// searches and the tokens a round verified beyond the one being emitted
/// ([`ngram_spec_round`]). The loop and `a_guessed_reply_is_the_reply_read_one_token_at_a_time` both
/// drive it through [`Self::next`].
///
/// **It checks only as many guesses as pay** ([`Self::guesses`], 2026-10-25).
/// It checked every guess it found, up to `num_pred_tokens`, which made replies
/// SLOWER on a card: a step there is one captured CUDA graph, a check of 11
/// positions is past what a graph carries (`cuda_graph::MAX_POSITIONS`) and
/// runs op by op, and Qwen 3.5's checks run op by op at any width. Measured on
/// an RTX 3070, 96 greedy tokens: qwen3-1.7B 120 → 110 tok/s on code and
/// 121 → 65-86 on a story; Llama-3.2-3B 107 → 65-71 on code; Qwen3.5-0.8B
/// 127-132 → 87-97. Each round is now timed — a step of one position on its
/// own, every wider forward as a check (`dsd_controller::CheckCost`) — and a
/// round checks as many of the guesses it FOUND as give the most tokens per
/// millisecond (`dsd_controller::best_gamma_beside_a_step`, zero included:
/// SmartSpec, as every other speculating path chooses), at the recent
/// acceptance (`AcceptanceEstimate::recent`): two guesses found can be worth
/// less than the check that reads them, where seven are not. On a card a check
/// is held to a graph's width.
///
/// **A one-off is not a price** ([`Self::one_offs`]). On a card a worker's
/// first forward of each shape runs without a graph (`cuda_graph`: a capture
/// follows an ordinary step of the same shape; 25-49 ms beside a 7.7-13.5 ms
/// step) and its next one makes the graph (a Qwen3.5-4B check: 259-277 ms
/// against 16-30 replayed). Timed as checks, either priced every check of the
/// reply; neither is timed.
///
/// **A refusal is tried again** ([`Self::probe_after`]). Fitted from its first
/// cold check, the count went to zero and, checking
/// nothing more, never learned that the reply had turned to repeating itself —
/// Qwen3.5-4B's story, where checking every guess ran 102-104 tok/s against 72
/// one at a time (α 0.93), stayed at 71. So after `probe_after` rounds of
/// guesses declined, the next guesses found are checked anyway; one that pays
/// (it wrote more than a step would have in its time) starts the count over at
/// [`Self::FIRST_PROBE_AFTER`], one that does not doubles it, up to
/// [`Self::MOST_PROBE_AFTER`] — `SpecBackoff`'s doubling. The split path's
/// counterpart re-tries a remembered zero after `dsd_controller::REPROBE_AFTER`.
/// `SWARMLLM_LOCAL_GUESS_GAMMA=fixed` checks every guess as before.
///
/// **Where the conversation offers nothing to copy, a split's card guesses on
/// its own** ([`CardGuesses`], 2026-10-28).
struct LocalSpec {
    cfg: crate::inference::ngram_lookup::NgramLookupConfig,
    /// Prompt then reply, in order, so its tail is always the most recent token.
    ctx: Vec<u32>,
    prompt_len: usize,
    /// Verified tokens a round produced beyond the one being emitted. Already
    /// in the cache, so draining them costs no forward. This is where
    /// speculation actually pays.
    pending: std::collections::VecDeque<u32>,
    backoff: SpecBackoff,
    /// The most positions a round may read: a captured graph's on a card, and
    /// for a model with recurrent state no more than the rows a rewind can
    /// replay (`SsmState::MOST_ROWS_KEPT`).
    widest: usize,
    /// Every guess checked, as before 2026-10-25 (`SWARMLLM_LOCAL_GUESS_GAMMA=fixed`).
    fixed: bool,
    step_ms: crate::inference::dsd_controller::RecentMedian,
    check: crate::inference::dsd_controller::CheckCost,
    acceptance: crate::inference::dsd_controller::AcceptanceEstimate,
    rounds: u64,
    produced: u64,
    accepted: u64,
    /// Checks that refused a guess, and so were taken back into.
    rejected: u64,
    /// Rounds that found guesses and checked none of them.
    declined: u64,
    /// Rounds whose forward paid a one-off on the card — the first of its
    /// shape, a graph made, a refused capture (`SplitModel::one_off_forwards`):
    /// a Qwen3.5-4B check that made its graph took 259-277 ms against 16-30
    /// replayed. Left out of the timings, or one such round prices every
    /// check of the reply.
    one_offs: u64,
    /// Rounds of guesses declined since the last check.
    declined_run: u32,
    /// Declined rounds after which found guesses are checked anyway.
    probe_after: u32,
    /// Checks made that way.
    probes: u64,
    /// The card's own guesses, for a whole model split between the card and
    /// the processor; `None` elsewhere.
    card: Option<CardGuesses>,
    /// Every round, as the test prints it.
    #[cfg(test)]
    trace: Vec<Traced>,
    /// The guesses each round checks, in place of the count's choice: a
    /// failing card run replayed exactly, on either device.
    #[cfg(test)]
    script: Option<std::collections::VecDeque<usize>>,
}

/// One round of [`LocalSpec`], for the test's trace.
#[cfg(test)]
#[derive(Debug, Clone)]
struct Traced {
    /// The cache's length when it began.
    at: usize,
    /// Guesses it checked.
    checked: usize,
    ms: f64,
    /// Its forward paid a one-off on the card.
    one_off: bool,
    accepted: usize,
    produced: Vec<u32>,
}

/// A split's card guessing the next tokens on its own, for [`LocalSpec`]
/// (2026-10-28).
///
/// A whole model split between the card and the processor leaves the card
/// idle for most of every step: a 14B with 27 of its 48 layers on an RTX 3070
/// spends ~20 ms of a 150 ms step there, the rest reading the processor's 21.
/// Run the card's layers alone — the token table, the card's layers, the
/// output head, skipping the processor's run (`SplitModel::card_guess_skips`)
/// — and the card writes a fair guess at the next token in that 20 ms; the
/// whole model then checks several such guesses in one step, and the
/// processor reads its weights once for every row it checks (the vendored
/// k-quant matmul is column-outer). Skipping ~44% of the layers, the card's
/// guess agreed with the whole model's next token 77-83% of the time on Qwen
/// 2.5 (7B, 14B), 50-67% on Llama 3.1 8B and GLM-4 9B, ~28% on Mistral 7B —
/// and 0-14% on all of them with the skipped layers at the END, which is why
/// a whole model's processor run sits early in the middle
/// (`hybrid::processor_run`; probe `examples/skip_agreement_probe.rs`).
///
/// Self-speculation by skipping layers is LayerSkip's, SWIFT's and Draft &
/// Verify's idea; S3D (arxiv 2405.20314) skips middle layers for low-memory
/// cards. What is ours is choosing the skipped layers as the ones the
/// processor holds, so the guess runs where the split leaves capacity idle.
///
/// The count is the step-beside one ([`LocalSpec::guesses`]'s), with each
/// guess's card time charged (`best_gamma_beside_a_step`'s `draft_ms_each`)
/// at this source's own acceptance — never the n-gram's, whose copied runs
/// are right far more often. A count of zero fades the acceptance toward its
/// prior, so a reply that turns guessable is tried again.
/// `SWARMLLM_CARD_GUESSES=0` turns it off.
///
/// **The skipped run is stood in for** ([`split::RunStandIn`], 2026-10-28):
/// every real forward of the reply keeps the hidden state where the run
/// starts and where it ends — the prompt pass its last
/// [`split::STAND_IN_WINDOW`] positions, each step and check the positions it
/// commits — and a guess adds the run's effect estimated from them, where it
/// passed the hidden state on as if the run wrote nothing. A stand-in that
/// fails is dropped for the reply, never the reply.
/// `SWARMLLM_CARD_GUESS_STAND_IN=none` guesses without one.
///
/// **It starts from what the model's earlier replies learned**
/// ([`split::StandInPrior`], 2026-10-29): each reply's positions are kept
/// with the model once the reply is out ([`absorb_the_reply`]), the window is
/// fitted to what their fit leaves, and a prompt longer than the window joins
/// that fit for its own reply ([`learn_the_prompt`]).
/// `SWARMLLM_CARD_GUESS_PRIOR=0` fits each reply alone.
///
/// [`split::RunStandIn`]: crate::inference::split::RunStandIn
/// [`split::STAND_IN_WINDOW`]: crate::inference::split::STAND_IN_WINDOW
/// [`split::StandInPrior`]: crate::inference::split::StandInPrior
struct CardGuesses {
    /// The layers a guess skips, over absolute layer indices.
    skips: Vec<bool>,
    /// What the skipped run did at this reply's real positions; `None` where
    /// it is off, or failed.
    stand_in: Option<crate::inference::split::RunStandIn>,
    /// The prompt pass's positions at the run's ends, and how many — learned
    /// on the reply's first step ([`Self::take_the_prompt`]), not before its
    /// first token.
    prompt: Option<(HashMap<usize, candle_core::Tensor>, usize)>,
    acceptance: crate::inference::dsd_controller::AcceptanceEstimate,
    /// One guess's time on the card, read back with the round's guesses.
    guess_ms: crate::inference::dsd_controller::RecentMedian,
    rounds: u64,
    proposed: u64,
    accepted: u64,
    /// Rounds that found guessing on the card not worth its time.
    declined: u64,
}

impl CardGuesses {
    /// Guesses made before any check has been timed: enough to fit the
    /// check's line from, few enough to cost little where they miss.
    const FIRST_GUESSES: usize = 3;

    fn for_model(model: &crate::inference::split::SplitModel) -> Option<Self> {
        if !card_guesses_on() {
            return None;
        }
        Some(Self {
            skips: model.card_guess_skips()?,
            stand_in: None,
            prompt: None,
            acceptance: crate::inference::dsd_controller::AcceptanceEstimate::recent(),
            guess_ms: Default::default(),
            rounds: 0,
            proposed: 0,
            accepted: 0,
            declined: 0,
        })
    }

    /// The prompt's positions into the stand-in, once, on the reply's first
    /// step. A guess is the first thing that reads the stand-in and the first
    /// token is the prompt pass's own, so the fit — the window, and for a long
    /// prompt the model's prior refitted with it — waits for that token to be
    /// out instead of holding it back (round 14: 1,759 tokens' first token
    /// 2.93-3.03 s without the prior, 3.55-3.87 s with it fitted before).
    /// A reply that ends before its first step never learns its prompt, which
    /// only its own guesses would have read.
    fn take_the_prompt(&mut self, model: &mut crate::inference::split::SplitModel) {
        let Some((caught, rows)) = self.prompt.take() else {
            return;
        };
        let Some(stand_in) = self.stand_in.as_mut() else {
            return;
        };
        let learned =
            tokio::task::block_in_place(|| learn_the_prompt(model, stand_in, &caught, rows));
        if let Err(e) = learned {
            tracing::warn!(error = %e, "card guesses: stand-in dropped for this reply");
            self.stand_in = None;
        }
    }
}

/// `SWARMLLM_CARD_GUESSES=0`: a split's card makes no guesses of its own
/// ([`CardGuesses`]), for A/B inside one binary.
fn card_guesses_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CARD_GUESSES").as_deref() != Ok("0"))
}

/// `SWARMLLM_CARD_GUESS_STAND_IN=none`: a card's guess passes the skipped
/// run's input on unchanged, as before the stand-in ([`CardGuesses`]), for
/// A/B inside one binary.
fn card_stand_in_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CARD_GUESS_STAND_IN").as_deref() != Ok("none"))
}

/// `SWARMLLM_CARD_GUESS_PRIOR=0`: each reply's stand-in is fitted from that
/// reply alone, as before the model's prior ([`CardGuesses`]), for A/B inside
/// one binary.
fn card_prior_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CARD_GUESS_PRIOR").as_deref() != Ok("0"))
}

/// The share of the stand-in's last miss its next guesses carry
/// (`RunStandIn::with_carry`): `SWARMLLM_CARD_GUESS_CARRY` (0 to 1; `0` is
/// the stand-in as before), else `split::STAND_IN_CARRY`.
fn card_carry_share() -> f64 {
    static SHARE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| {
        std::env::var("SWARMLLM_CARD_GUESS_CARRY")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|s| (0.0..=1.0).contains(s))
            .unwrap_or(crate::inference::split::STAND_IN_CARRY)
    })
}

/// The stand-in a reply's card guesses will read, made before its prompt
/// pass so the pass can keep what it is fitted from: where the reply will be
/// speculated (`speculating`), its card can guess and the stand-in is on.
fn card_stand_in_for(
    model: &crate::inference::split::SplitModel,
    speculating: bool,
) -> Option<crate::inference::split::RunStandIn> {
    if !speculating || local_guess_gamma_fixed() || !card_guesses_on() || !card_stand_in_on() {
        return None;
    }
    let stand_in =
        crate::inference::split::RunStandIn::for_run(&model.card_guess_skips()?, model.device())?
            .with_carry(
                card_carry_share(),
                crate::inference::split::STAND_IN_CARRY_POSITIONS,
            );
    Some(if card_prior_on() {
        stand_in.collecting()
    } else {
        stand_in
    })
}

/// The prompt's positions into the reply's stand-in: measured against the
/// model's prior where the stand-in collects for one — refitted with the
/// prompt first where it holds more positions than the window
/// (`StandInPrior::fit_for_prompt`). A prior that cannot be fitted leaves the
/// reply's window to itself; only the window failing drops the stand-in.
fn learn_the_prompt(
    model: &mut crate::inference::split::SplitModel,
    stand_in: &mut crate::inference::split::RunStandIn,
    caught: &HashMap<usize, candle_core::Tensor>,
    rows: usize,
) -> Result<(), SwarmError> {
    let (phi, delta) = stand_in.rows_of(caught, rows)?;
    learn_prompt_rows(model, stand_in, phi, delta)
}

/// [`learn_the_prompt`] from the prompt's `(φ, Δ)` however they were caught —
/// a captured pass, or a chain's two ends ([`FarRuns`]).
fn learn_prompt_rows(
    model: &mut crate::inference::split::SplitModel,
    stand_in: &mut crate::inference::split::RunStandIn,
    phi: candle_core::Tensor,
    delta: candle_core::Tensor,
) -> Result<(), SwarmError> {
    if stand_in.collects() {
        let window = stand_in.window();
        let prior = model.stand_in_prior_for(stand_in);
        let fits_before = prior.fits;
        match prior.fit_for_prompt(&phi, &delta, window) {
            Ok(fit) => {
                if prior.fits > fits_before {
                    tracing::debug!(
                        rows = fit.as_ref().map_or(0, |f| f.rows()),
                        fits = prior.fits,
                        fit_ms_total = prior.ms,
                        "card guesses: the prior fitted on the reply's first step"
                    );
                }
                stand_in.set_prior(fit)?
            }
            Err(e) => {
                // Dropped, as the idle refit drops one: kept stale, every
                // later reply would pay for the same failing fit.
                tracing::warn!(error = %e, "card guesses: no prior for this reply; it is dropped");
                *prior = crate::inference::split::StandInPrior::for_stand_in(stand_in);
            }
        }
    }
    stand_in.learn_rows(phi, delta)
}

/// Once a reply is out, what its stand-in learned joins the model's prior
/// (`StandInPrior::take_in`) — unfitted: the worker fits it while nothing
/// waits ([`refit_stale_priors`]), else the next reply's first step does.
/// A failure costs the prior, never the worker.
fn absorb_the_reply(
    model: &mut crate::inference::split::SplitModel,
    mut stand_in: crate::inference::split::RunStandIn,
) {
    if !stand_in.collects() {
        return;
    }
    let prior = model.stand_in_prior_for(&stand_in);
    if let Err(e) = prior.take_in(&mut stand_in) {
        tracing::warn!(error = %e, "card guesses: the model's prior is dropped");
        *prior = crate::inference::split::StandInPrior::for_stand_in(&stand_in);
    }
}

/// Fit every model's prior that replies have joined since its last fit, for
/// as long as no message waits (`keep_going`): a request arriving meanwhile
/// waits one of the inverse's steps, not the 175-280 ms fit (round 15). True
/// where anything was fitted, so the card's pool may hold what it freed.
fn refit_stale_priors(
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    keep_going: &dyn Fn() -> bool,
) -> bool {
    let mut any = false;
    for model in models.values_mut() {
        if !keep_going() {
            break;
        }
        match tokio::task::block_in_place(|| model.refit_stale_prior(keep_going)) {
            Ok(true) => {
                any = true;
                if let Some(prior) = model.stand_in_prior() {
                    tracing::debug!(
                        rows = prior.rows(),
                        fits = prior.fits,
                        fit_ms_total = prior.ms,
                        "card guesses: the model's prior refitted"
                    );
                }
            }
            Ok(false) => {}
            Err(e) => tracing::warn!(error = %e, "card guesses: the model's prior is dropped"),
        }
    }
    any
}

impl LocalSpec {
    /// `fixed`: check every guess found, as before 2026-10-25 — the loop
    /// passes [`local_guess_gamma_fixed`], the test runs both.
    fn new(
        cfg: crate::inference::ngram_lookup::NgramLookupConfig,
        prompt_ids: &[u32],
        model: &crate::inference::split::SplitModel,
        fixed: bool,
    ) -> Self {
        let widest = if model.device().is_cuda() && !fixed {
            crate::inference::cuda_graph::MAX_POSITIONS
        } else {
            usize::MAX
        };
        let widest = if model.carries_recurrent_state() {
            widest.min(crate::inference::layers::SsmState::MOST_ROWS_KEPT)
        } else {
            widest
        };
        Self {
            cfg,
            ctx: prompt_ids.to_vec(),
            prompt_len: prompt_ids.len(),
            pending: std::collections::VecDeque::new(),
            backoff: SpecBackoff::default(),
            widest,
            fixed,
            step_ms: Default::default(),
            check: crate::inference::dsd_controller::CheckCost::for_a_plan(None),
            acceptance: crate::inference::dsd_controller::AcceptanceEstimate::recent(),
            rounds: 0,
            produced: 0,
            accepted: 0,
            rejected: 0,
            declined: 0,
            one_offs: 0,
            declined_run: 0,
            probe_after: Self::FIRST_PROBE_AFTER,
            probes: 0,
            card: (!fixed).then(|| CardGuesses::for_model(model)).flatten(),
            #[cfg(test)]
            trace: Vec::new(),
            #[cfg(test)]
            script: None,
        }
    }

    const FIRST_PROBE_AFTER: u32 = 8;
    const MOST_PROBE_AFTER: u32 = 64;

    /// Hand the card's guesses the stand-in the prompt pass fed
    /// ([`card_stand_in_for`]) and what the pass kept for it, learned on the
    /// first step ([`CardGuesses::take_the_prompt`]); dropped where the card
    /// makes none.
    fn with_stand_in(
        mut self,
        stand_in: Option<crate::inference::split::RunStandIn>,
        prompt: Option<(HashMap<usize, candle_core::Tensor>, usize)>,
    ) -> Self {
        if let Some(card) = self.card.as_mut() {
            card.stand_in = stand_in;
            card.prompt = prompt;
        }
        self
    }

    /// Did a check made against the count's advice pay — write more tokens
    /// than steps would have in its time?
    fn probe_paid(&self, round: &SpecRound) -> bool {
        let Some(step) = self.step_ms.median() else {
            return true;
        };
        round.produced.len() as f64 * step >= round.ms
    }

    /// How many guesses the next round's lookup may return: a round reads the
    /// token, then the guesses.
    fn room(&self) -> usize {
        self.cfg.num_pred_tokens.min(self.widest.saturating_sub(1))
    }

    /// How many of the `most` guesses found the round checks: all of them
    /// until a step and a check have both been timed, then the count that
    /// pays — zero included.
    fn guesses(&self, most: usize) -> usize {
        if self.fixed {
            return most;
        }
        crate::inference::dsd_controller::best_gamma_beside_a_step(
            self.acceptance.alpha(),
            self.step_ms.median(),
            &self.check,
            u32::try_from(most).unwrap_or(u32::MAX),
            0.0,
        )
        .map_or(most, |g| g as usize)
    }

    /// How many guesses the card makes this round ([`CardGuesses`]): none
    /// before a plain step has been timed, [`CardGuesses::FIRST_GUESSES`]
    /// before a check has, then the count that pays with each guess's card
    /// time charged — zero included.
    fn card_guess_count(&self, room: usize) -> usize {
        let Some(card) = self.card.as_ref() else {
            return 0;
        };
        let Some(step) = self.step_ms.median() else {
            return 0;
        };
        crate::inference::dsd_controller::best_gamma_beside_a_step(
            card.acceptance.alpha(),
            Some(step),
            &self.check,
            u32::try_from(room).unwrap_or(u32::MAX),
            card.guess_ms.median().unwrap_or(0.0),
        )
        .map_or(CardGuesses::FIRST_GUESSES.min(room), |g| g as usize)
    }

    /// What the guessing learned, for the request's closing line.
    fn summary(&self) -> String {
        let fit = self.check.fit().map_or("-".to_string(), |(f, s)| {
            format!("{f:.1} + {s:.2}/position")
        });
        let step = self
            .step_ms
            .median()
            .map_or("-".to_string(), |m| format!("{m:.1}"));
        let card = self.card.as_ref().map_or(String::new(), |c| {
            let stand_in = c.stand_in.as_ref().map_or(" stand-in none".to_string(), |s| {
                format!(
                    " stand-in rows {} prior {} rebuilds {} ms {:.1} not-finite {}",
                    s.rows(),
                    s.prior_rows(),
                    s.rebuilds,
                    s.ms,
                    s.not_finite
                )
            });
            format!(
                " card rounds {} guessed {} kept {} declined {} guess {} ms alpha {:.2} next {}{stand_in}",
                c.rounds,
                c.proposed,
                c.accepted,
                c.declined,
                c.guess_ms
                    .median()
                    .map_or("-".to_string(), |m| format!("{m:.1}")),
                c.acceptance.alpha(),
                self.card_guess_count(self.room()),
            )
        });
        format!(
            "rounds {} accepted {} rejected {} declined {} probes {} one-off {} step {step} ms check {fit} ms alpha {:.2} next {}{card}",
            self.rounds,
            self.accepted,
            self.rejected,
            self.declined,
            self.probes,
            self.one_offs,
            self.acceptance.alpha(),
            self.guesses(self.room()),
        )
    }

    /// The token after `next_token`, which the caller has just emitted and
    /// pushed onto `generated`: one already verified, else the first of a new
    /// round. `index_pos` is the cache's length, moved by what the round adds.
    #[allow(clippy::too_many_arguments)]
    fn next(
        &mut self,
        model: &mut crate::inference::split::SplitModel,
        kv_store: &crate::inference::split::KvCacheStore,
        model_key: &str,
        req_id_str: &str,
        index_pos: &mut usize,
        next_token: u32,
        generated: &[u32],
        sampling: &crate::types::SamplingParams,
        force_attn: bool,
    ) -> Result<u32, SwarmError> {
        self.ctx.push(next_token);
        if let Some(card) = self.card.as_mut() {
            card.take_the_prompt(model);
        }
        if let Some(tok) = self.pending.pop_front() {
            return Ok(tok);
        }
        // Until the processor's decode width has settled, step plainly: the
        // calibration learns only from plain steps, and a step taken at a width
        // under trial is not the step the guesses are priced against
        // (`cpu_pools::decode_width_settled`).
        let settling = !self.fixed && !model.decode_width_settled();
        #[cfg(test)]
        let settling = settling && self.script.is_none();
        let room = if settling { 0 } else { self.room() };
        // `room` first: a settling step is no paused round.
        let mut found = if room > 0 && self.backoff.should_draft() {
            let (mut found, _source) = crate::inference::ngram_lookup::cascade_find_candidate(
                &self.ctx,
                self.prompt_len,
                NGRAM_RECENT_GEN_WINDOW,
                self.cfg,
            );
            found.truncate(room);
            found
        } else {
            Vec::new()
        };
        let mut worth = self.guesses(found.len());
        let mut probe = false;
        if !found.is_empty() {
            if worth > 0 {
                self.declined_run = 0;
            } else {
                self.declined_run += 1;
                if self.declined_run >= self.probe_after {
                    (worth, probe, self.declined_run) = (found.len(), true, 0);
                }
            }
        }
        // Guesses found and none worth their check.
        #[cfg(test)]
        if let Some(script) = self.script.as_mut() {
            (worth, probe) = (script.pop_front().unwrap_or(0).min(found.len()), false);
        }
        let declined = !found.is_empty() && worth == 0;
        let one_offs_before = model.one_off_forwards();
        #[cfg(test)]
        let began_at = *index_pos;
        // Nothing to copy: the card guesses on its own, where it can.
        let mut card_round = false;
        let mut card_declined = false;
        let card_due = found.is_empty() && self.card.is_some() && !settling;
        #[cfg(test)]
        let card_due = card_due && self.script.is_none();
        if card_due {
            let gamma = self.card_guess_count(room);
            if gamma > 0 {
                if let Some(card) = self.card.as_mut() {
                    let started = std::time::Instant::now();
                    if let Some(Err(e)) = card.stand_in.as_mut().map(|s| s.refresh()) {
                        tracing::warn!(error = %e, "card guesses: stand-in dropped for this reply");
                        card.stand_in = None;
                    }
                    let bridge = card
                        .stand_in
                        .as_ref()
                        .and_then(crate::inference::split::RunStandIn::bridge);
                    found = tokio::task::block_in_place(|| {
                        let _g =
                            crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
                        model.card_guesses(
                            kv_store,
                            req_id_str,
                            *index_pos,
                            next_token,
                            gamma,
                            &card.skips,
                            bridge,
                        )
                    })?;
                    found.truncate(gamma);
                    worth = found.len();
                    if worth > 0 {
                        card.guess_ms
                            .record(started.elapsed().as_secs_f64() * 1000.0 / worth as f64);
                    }
                    card_round = worth > 0;
                }
            } else {
                card_declined = self.step_ms.median().is_some() && self.check.fit().is_some();
            }
        }
        let round = ngram_spec_round(
            model,
            kv_store,
            model_key,
            req_id_str,
            *index_pos,
            next_token,
            &found[..worth],
            generated,
            sampling,
            force_attn,
            self.card
                .as_ref()
                .and_then(|c| c.stand_in.as_ref())
                .map(crate::inference::split::RunStandIn::capture_layers),
        )?;
        *index_pos += round.committed;
        if let (Some(caught), Some(card)) = (round.caught.as_ref(), self.card.as_mut()) {
            if let Some(stand_in) = card.stand_in.as_mut() {
                if let Err(e) = stand_in.learn(caught, round.committed) {
                    tracing::warn!(error = %e, "card guesses: stand-in dropped for this reply");
                    card.stand_in = None;
                }
            }
        }
        let one_off = model.one_off_forwards() > one_offs_before;
        self.one_offs += u64::from(one_off);
        #[cfg(test)]
        self.trace.push(Traced {
            at: began_at,
            checked: worth,
            ms: round.ms,
            one_off,
            accepted: round.accepted,
            produced: round.produced.clone(),
        });
        self.rounds += 1;
        self.produced += round.produced.len() as u64;
        self.accepted += round.accepted as u64;
        self.rejected += u64::from(round.accepted < round.proposed);
        match round.positions {
            _ if one_off || settling => {}
            1 => self.step_ms.record(round.ms),
            p => self
                .check
                .record(u32::try_from(p).unwrap_or(u32::MAX), round.ms),
        }
        if let Some(card) = self.card.as_mut().filter(|_| card_round || card_declined) {
            if card_round {
                card.rounds += 1;
                card.proposed += round.proposed as u64;
                card.accepted += round.accepted as u64;
                card.acceptance
                    .record(round.accepted as u32, round.proposed as u32);
            } else {
                card.declined += 1;
                card.acceptance.fade();
            }
        } else if round.proposed > 0 {
            self.backoff.record(round.accepted > 0);
            self.acceptance
                .record(round.accepted as u32, round.proposed as u32);
        } else if declined {
            self.declined += 1;
            self.acceptance.fade();
        }
        if probe {
            self.probes += 1;
            // A probe that paid a one-off said nothing of a check's price:
            // the next is soon, and cheap.
            self.probe_after = if one_off || self.probe_paid(&round) {
                Self::FIRST_PROBE_AFTER
            } else {
                (self.probe_after * 2).min(Self::MOST_PROBE_AFTER)
            };
        }
        let mut it = round.produced.into_iter();
        // A round always yields at least the token that follows the one just
        // forwarded, so this cannot be empty.
        let first = it
            .next()
            .ok_or_else(|| SwarmError::Internal("speculative round produced no token".into()))?;
        self.pending.extend(it);
        Ok(first)
    }
}

/// `SWARMLLM_LOCAL_GUESS_GAMMA=fixed`: the local n-gram checks every guess it
/// finds, up to `num_pred_tokens` and at any width — the rule before
/// [`LocalSpec::guesses`] — for A/B inside one binary.
fn local_guess_gamma_fixed() -> bool {
    static FIXED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FIXED.get_or_init(|| std::env::var("SWARMLLM_LOCAL_GUESS_GAMMA").as_deref() == Ok("fixed"))
}

/// KV positions reserved for the reply when a prompt is admitted and when its
/// snapshot is sized: one growth quantum, the least a decode needs to claim.
/// The most a reply is reserved, and all of it when the reply's length is not
/// known — a segment's prompt pass, which never sees `max_tokens`.
const REPLY_RESERVE_POSITIONS: usize = crate::inference::layers::KV_CACHE_GROWTH_TOKENS;

/// Whether a reply is still to be decoded beside a prompt snapshot being
/// sized — a REQUIRED argument of every sizing ([`save_prompt_opening`],
/// [`snapshot_room`]). A snapshot copied before the reply leaves its live
/// cache a growth quantum to claim; one copied once the reply is out
/// ([`prompt_saved_when_done`]) needs none, its live cache about to be freed —
/// kept, the reserve shortened that snapshot on a tight card or evicted
/// another prompt's to make room for a reply already written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplyAhead {
    ToDecode,
    Done,
}

impl ReplyAhead {
    /// Positions kept beside the live cache for the reply.
    fn reserve_positions(self) -> usize {
        match self {
            Self::ToDecode => REPLY_RESERVE_POSITIONS,
            Self::Done => 0,
        }
    }
}

/// The least a reply's reserve leaves for a speculative round to write past
/// the reply before it is truncated back — n-gram drafts
/// (`--ngram-pred-tokens`, 10 by default) and SWIFT's gamma (4) fit with room.
/// [`draft_margin_positions`] raises it to whatever this worker is configured
/// to draft.
const DRAFT_MARGIN_POSITIONS: usize = 32;

/// Positions this worker's speculative rounds may write past a reply: the
/// larger of the configured draft lengths (plus the bonus token) and
/// [`DRAFT_MARGIN_POSITIONS`].
///
/// `--ngram-pred-tokens` is an unbounded setting, and a fixed margin would let
/// an operator who raised it draft past a short chat's reservation, growing
/// the cache mid-reply — where the guard can refuse — at exactly the
/// concurrency a smaller reserve admits (review of #122, 2026-09-26).
fn draft_margin_positions(
    ngram_cfg: &crate::inference::ngram_lookup::NgramLookupConfig,
    swift_cfg: &SwiftConfig,
) -> usize {
    DRAFT_MARGIN_POSITIONS
        .max(ngram_cfg.num_pred_tokens.saturating_add(1))
        .max((swift_cfg.gamma as usize).saturating_add(1))
}

/// The reply reserve for a request whose granted budget is `max_new_tokens`
/// (`resolve_max_new_tokens`, already assigned back), plus `draft_margin` for
/// speculation ([`draft_margin_positions`]): what the reply can actually
/// occupy, never more than [`REPLY_RESERVE_POSITIONS`] — so never more than
/// the reservation before this existed, whatever the draft settings.
///
/// A reply cannot outgrow its budget, so a quantum reserved for a 128-token
/// answer was 384 positions nothing could ever write. It is not free: the
/// reservation is each layer's FIRST allocation, so it is held from the first
/// token. Measured 2026-09-26 on llama-3.1-8b on an 8 GB card: every short
/// chat claimed 1,024 positions (384 MB with its f16 mirror), three of them
/// took 1,152 of a 1,336 MB conversation budget, and a fourth 45-token chat
/// was refused (`docs/FUTURE_WORK.md` #122). Sized from the budget, a
/// twenty-token chat asking for 128 tokens reserves one quantum instead of two.
fn reply_reserve_positions(max_new_tokens: u32, draft_margin: usize) -> usize {
    (max_new_tokens as usize)
        .saturating_add(draft_margin)
        .min(REPLY_RESERVE_POSITIONS)
}

/// The positions a drafter's cache is admitted for when it reads a reply's
/// prompt ([`handle_draft`]'s first call): the prompt and the reply's reserve
/// in whole quanta — the target's own per-request figure
/// ([`admit_prompt_over`]). A drafter's LOAD is charged this, not a typical
/// conversation (`vram::KvCharge::Conversation`, #275), so the room a split
/// leaves for it is the room that call takes. `max_new_tokens` is the reply's
/// granted budget, `None` where it is not known (the planner): the whole
/// reserve. The margin covers every guess a call can ask for, so the charge is
/// never below that call's admission (asserted below).
pub(crate) fn drafter_cache_positions(prompt_tokens: usize, max_new_tokens: Option<u32>) -> u64 {
    let reply = max_new_tokens.map_or(REPLY_RESERVE_POSITIONS, |m| {
        reply_reserve_positions(m, DRAFT_MARGIN_POSITIONS)
    });
    crate::inference::layers::kv_cache_reservation(prompt_tokens.saturating_add(reply)) as u64
}

/// [`handle_draft`] admits `gamma + 1` positions past the reply; the charge's
/// margin must hold the largest.
const _: () = assert!(DRAFT_MARGIN_POSITIONS > crate::config::MAX_GUESSES_PER_CHECK as usize);

/// `SWARMLLM_KV_PREFIX_CHARGE=0` turns off the whole-prompt admission below,
/// so the two arms can be compared inside ONE binary. Read once.
#[cfg(test)]
mod subsumption_tests {
    use super::subsumed_segment_keys;

    /// The reported sequence (#010): a worker holding [16..48) and [0..16) is
    /// asked for [0..48). Both are covered, so both are redundant — before
    /// this, the map missed on the exact key and read the whole model from
    /// disk a second time beside the copy already resident.
    #[test]
    fn a_full_range_supersedes_the_pieces_that_already_cover_it() {
        let loaded = [(16, 48, 0, 1), (0, 16, 0, 1)];
        let mut got = subsumed_segment_keys(loaded.iter(), (0, 48, 0, 1));
        got.sort();
        assert_eq!(got, vec![(0, 16, 0, 1), (16, 48, 0, 1)]);
    }

    /// A partial overlap is NOT subsumption: each range still needs layers the
    /// other does not have, so dropping either would cost a reload for nothing.
    #[test]
    fn a_partial_overlap_is_left_alone() {
        let loaded = [(0, 20, 0, 1)];
        assert!(subsumed_segment_keys(loaded.iter(), (16, 48, 0, 1)).is_empty());
    }

    /// The identical range is the caller's own cache hit, not something to drop
    /// — returning it here would have the loader remove the entry it is about
    /// to find.
    #[test]
    fn the_same_range_is_never_dropped() {
        let loaded = [(0, 48, 0, 1)];
        assert!(subsumed_segment_keys(loaded.iter(), (0, 48, 0, 1)).is_empty());
    }

    /// A narrower incoming range supersedes nothing — this is the direction
    /// that must NOT thrash: asking for a piece of what we hold keeps both.
    #[test]
    fn a_narrower_range_supersedes_nothing() {
        let loaded = [(0, 48, 0, 1)];
        assert!(subsumed_segment_keys(loaded.iter(), (0, 16, 0, 1)).is_empty());
    }

    /// Tensor-parallel shape is part of the identity: a rank's range says
    /// nothing about another rank's, and dropping across ranks would take away
    /// weights a different worker slot is using.
    #[test]
    fn a_different_tensor_parallel_shape_is_untouched() {
        let loaded = [(0, 16, 1, 2)];
        assert!(subsumed_segment_keys(loaded.iter(), (0, 48, 0, 1)).is_empty());
        let loaded = [(0, 16, 0, 2)];
        assert!(subsumed_segment_keys(loaded.iter(), (0, 48, 0, 1)).is_empty());
    }
}

fn prefix_cache_charged() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_KV_PREFIX_CHARGE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// Where a finished prompt's prefix-cache snapshot is kept and how many of its
/// positions fit there — the ONE plan both inserts take (`SnapshotRoom`).
/// Evicts older cached prompts of the same home first.
///
/// A card's snapshot goes to host RAM (#257) and is sized against the
/// machine (`kv_budget::host_snapshot_budget`); one kept beside the live
/// cache is sized against that device's KV budget (gotcha #440).
///
/// **A card with no room for the copy beside the live cache keeps it in host
/// RAM instead of keeping less of it** (field report 2026-10-19): a 12 GB card
/// holding two segments of a 30B had ~190 MB left for caches, the next turn's
/// prompt opening was not kept whole beside its live cache, and the whole
/// split read the 3,718-token prompt again — 103 s on the peer abroad that had
/// kept its part. The copy off the card costs 59-145 ms per 264 MB (#257's
/// measurement); the turn it saves costs the whole prompt on every segment.
fn snapshot_room(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    prompt_tokens: usize,
    reply: ReplyAhead,
) -> SnapshotRoom {
    let home = SnapshotHome::for_device(model.device());
    if home == SnapshotHome::HostRam {
        return SnapshotRoom {
            home,
            max_positions: host_ram_positions(model, prefix_cache, prompt_tokens),
        };
    }
    let beside = positions_beside_live(model, kv_store, prefix_cache, prompt_tokens, reply);
    let room = room_off_the_card_if_more(
        beside,
        prompt_tokens,
        SnapshotHome::may_leave_the_card(model.device()),
        || host_ram_positions(model, prefix_cache, prompt_tokens),
    );
    if room.home == SnapshotHome::HostRam {
        tracing::info!(
            prompt_tokens,
            beside_live = beside,
            in_host_memory = room.max_positions.min(prompt_tokens),
            "DIAG: prefix-cache snapshot kept in host memory — the card has no room for it \
             beside the live cache"
        );
    }
    room
}

/// [`snapshot_room`]'s choice for a card's snapshot: beside the live cache
/// when `beside` holds the whole prompt or the copy may not leave the card,
/// else host RAM where it holds more there (`host`, asked only then — it may
/// evict older host copies).
fn room_off_the_card_if_more(
    beside: usize,
    prompt_tokens: usize,
    may_leave_the_card: bool,
    host: impl FnOnce() -> usize,
) -> SnapshotRoom {
    let on_the_card = SnapshotRoom {
        home: SnapshotHome::BesideLive,
        max_positions: beside,
    };
    if beside >= prompt_tokens || !may_leave_the_card {
        return on_the_card;
    }
    let host = host();
    if host > beside {
        SnapshotRoom {
            home: SnapshotHome::HostRam,
            max_positions: host,
        }
    } else {
        on_the_card
    }
}

/// Saves the opening a request's caches hold now — `tokens`, every position
/// read so far — to the prefix cache, where its pass reaches
/// `PrefixCache::save_point` (#228). Both generate paths call it, so the plan
/// of where the copy goes and the charge for it are one decision. Returns the
/// blocks to announce (none for a kept request or a recurrent model's state).
#[allow(clippy::too_many_arguments)]
fn save_prompt_opening(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    model_key: &str,
    req_id: &str,
    tokens: &[u32],
    keep: KeepAsked,
    reply: ReplyAhead,
) -> Vec<crate::types::PrefixBlockEntry> {
    let room = snapshot_room(model, kv_store, prefix_cache, tokens.len(), reply);
    let manifest = prefix_cache.insert_from_kv(model_key, req_id, kv_store, tokens, room, keep);
    // A snapshot kept beside the live cache is device memory the head-room
    // guard must see.
    if prefix_cache_charged() {
        kv_store.set_external_reserved(prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64);
    }
    manifest
}

/// Is a prompt's end-of-pass snapshot copied once the REPLY is done, rather
/// than between the prompt pass and the first token? llama.cpp's server saves
/// a slot's prompt to its host cache only when the slot is next used
/// (`tools/server/server-context.cpp:1647` at 4b1a27f), so neither the first
/// token nor the reply waits on the copy. Here the copy sat before the first
/// token on both generate paths: 0.5-0.9 s of a 14B's 1,800-token prompt on
/// a card/processor split. Deferred, the first token came 0.12 s sooner at the
/// median — the next prompt pass ran 0.25 s slower, cause open (2026-10-28,
/// round 8 — `docs/invariants/inference.md`). It can wait wherever the prompt's
/// positions are still whole at the end: decode only appends, and a rejected
/// guess is cut back no further than the prompt's end — except a windowed
/// cache that slides ([`SplitModel::any_cache_slides`]), which may have
/// dropped the prompt's opening by then. A recurrent model saves short of the
/// prompt's end, inside the pass (`PrefixCache::save_point`), and is not
/// asked. `SWARMLLM_PROMPT_SAVE=at-prompt-end` is the old order, the A/B.
fn prompt_saved_when_done(model: &SplitModel) -> bool {
    static AT_PROMPT_END: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let at_prompt_end = *AT_PROMPT_END
        .get_or_init(|| std::env::var("SWARMLLM_PROMPT_SAVE").as_deref() == Ok("at-prompt-end"));
    !at_prompt_end && !model.any_cache_slides()
}

/// Are a request's caches still whole after its reply ended with `error`, so
/// a prompt snapshot left for the reply's end ([`prompt_saved_when_done`]) is
/// still taken? Yes for every error this worker survives — a cancel landing
/// inside a step included: a failed step puts the caches back where it began,
/// so the prompt's positions are the prompt pass's, and a retried request
/// finds its prompt as it did when the copy came before the first token. No
/// after an error that ruined the device ([`worker_error_is_fatal`]). The ONE
/// answer for both generate paths: the sequential reply, a failed decode tick
/// and a batched slot that ended in error.
///
/// [`worker_error_is_fatal`]: crate::inference::worker_ipc::worker_error_is_fatal
fn prompt_outlives(error: Option<&str>) -> bool {
    error.is_none_or(|e| !crate::inference::worker_ipc::worker_error_is_fatal(e))
}

/// Positions of a card's snapshot that fit in host RAM beside the
/// snapshots already there, evicting the oldest of those first. `usize::MAX`
/// when the machine's memory or a position's weight cannot be read. Not
/// switched off with `SWARMLLM_KV_PREFIX_CHARGE=0`: that A/B is about the
/// card's budget, and this is the only bound host RAM has.
fn host_ram_positions(
    model: &SplitModel,
    prefix_cache: &PrefixCache,
    prompt_tokens: usize,
) -> usize {
    use crate::inference::split::kv_budget::{
        host_free_and_total_bytes, host_snapshot_budget, plan_snapshot, KvGrowth,
    };
    // A snapshot copies every layer's cache, card-held and processor-held.
    let growth = model
        .kv_homes()
        .map(|home| model.kv_budget(home).1)
        .fold(KvGrowth::default(), KvGrowth::plus);
    let Some((available, total)) = host_free_and_total_bytes() else {
        return usize::MAX;
    };
    if growth.per_position() == 0 {
        return usize::MAX;
    }
    let held = prefix_cache.bytes_in(SnapshotHome::HostRam) as u64;
    let budget = host_snapshot_budget(available, total, held);
    let plan = plan_snapshot(budget, 0, held, growth, prompt_tokens);
    if plan.evict_bytes > 0 {
        prefix_cache.release(plan.evict_bytes as usize, SnapshotHome::HostRam);
    }
    if plan.positions < prompt_tokens {
        tracing::info!(
            prompt_tokens,
            keeping = plan.positions,
            held_mb = held / (1024 * 1024),
            available_mb = available / (1024 * 1024),
            budget_mb = budget / (1024 * 1024),
            "DIAG: prefix-cache snapshot cut to the host memory the machine can spare"
        );
    }
    plan.positions
}

/// How many positions of a finished prompt to snapshot into the prefix cache
/// so the copy fits BESIDE the request's live cache (gotcha #440). Evicts
/// older cached prompts first. `usize::MAX` when the budget is unknown or
/// charging is switched off.
fn positions_beside_live(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    prompt_tokens: usize,
    reply: ReplyAhead,
) -> usize {
    if !prefix_cache_charged() {
        return usize::MAX;
    }
    let home = model.kv_home();
    let allocated = kv_store.occupancy().allocated_in(home);
    let cached = prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64;
    // Against the card as it stands, not the budget predicted at load: a
    // snapshot is a device allocation the size of the prompt, taken on a card
    // that may hold tenants the load-time figure never saw.
    let (Some(budget), growth) = model.kv_budget_now(home, allocated, cached) else {
        return usize::MAX;
    };
    if growth.per_position() == 0 {
        return usize::MAX;
    }
    // The live figure carries one growth quantum of reserve for a reply
    // still to be decoded: a snapshot sized to the last byte left this
    // request unable to claim its next quantum mid-reply (measured: the
    // reply cut at token 32, the quantum boundary after a 7649-token prompt).
    // None once the reply is out (`ReplyAhead::Done`).
    let live = allocated.saturating_add(
        growth.bytes_between(prompt_tokens, prompt_tokens + reply.reserve_positions()),
    );
    let plan = crate::inference::split::kv_budget::plan_snapshot(
        budget,
        live,
        cached,
        growth,
        prompt_tokens,
    );
    if plan.evict_bytes > 0 {
        prefix_cache.release(plan.evict_bytes as usize, SnapshotHome::BesideLive);
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
/// Refuse a segment's prompt pass that is longer than this node serves —
/// the WHOLE prompt, before any layer runs.
///
/// The executor's own guard sees one chunk at a time (`forward_prompt_in_chunks`,
/// 128 positions), so an over-long prompt was computed through every layer of
/// this segment up to the limit and only then refused, naming the chunk
/// boundary rather than the prompt: measured on the live swarm 2026-09-25, an
/// 8560-token prompt on a peer serving 8192 was refused as "8320 tokens" 177 s
/// after it was sent, with 8192 positions of work thrown away. Here it costs
/// nothing and names the real length, which the coordinator reads to decide
/// whether another machine could serve it (`error::served_context_refusal`).
fn refuse_a_prompt_past_the_served_context(
    context_window: usize,
    index_pos: usize,
    positions: usize,
) -> Result<(), SwarmError> {
    let total = index_pos.saturating_add(positions);
    if total > context_window {
        return Err(crate::error::longer_than_served(total, context_window));
    }
    Ok(())
}

/// `reply_positions` is the reply's reserve — [`reply_reserve_positions`] of
/// the granted budget where the worker knows it, [`REPLY_RESERVE_POSITIONS`]
/// where it cannot (a segment's prompt pass). A required argument, so no
/// entry point can fall back to the fixed quantum by omission.
/// Does a FIRST segment's input carry the prompt as TEXT, to be tokenized here,
/// rather than token ids? Only a whole prompt pass from position 0 that neither
/// resumes a kept prompt (#10) nor is a piece of a pass (#171): those, and every
/// later step, carry ids (8 bytes each). One rule for the worker that reads the
/// input and the serving node that counts it
/// (`tensor_util::forward_positions`).
pub(crate) fn first_segment_input_is_text(
    index_pos: u32,
    resumes_a_kept_prompt: bool,
    is_a_piece: bool,
) -> bool {
    index_pos == 0 && !resumes_a_kept_prompt && !is_a_piece
}

/// How many positions a coordinator's forward of `input_positions` hands a
/// windowed layer's cache at once — what that cache must hold beside its
/// window (FUTURE_WORK #281) — where that is wider than this worker's own
/// prompt chunk; 0 where it is not (`SplitModel::widest_prompt_forward`,
/// which the model's budget already allows for).
///
/// `handle_forward` runs a prompt in the worker's own chunks
/// (`forward_prompt_in_chunks`, cut at `prompt_chunk_ceiling` — the rule
/// `widest_prompt_forward` reads) unless it is a tensor-parallel phase or an
/// image's prompt (`forward_multimodal`), which run the whole input in ONE
/// pass — the same three facts, in the same order, its run branches on.
/// Widened to every forward's width, as it was, a peer handed a long prompt
/// was admitted as if its windowed layers kept every position of it.
fn width_handed_to_the_caches(
    tensor_parallel: bool,
    pre_embedded: bool,
    has_image: bool,
    input_positions: usize,
) -> usize {
    let one_pass = tensor_parallel || (!pre_embedded && has_image);
    if one_pass {
        input_positions
    } else {
        0
    }
}

/// `forward_width` is how many positions the forward asking hands a windowed
/// layer's cache at once where that is wider than this worker's own chunk
/// ([`width_handed_to_the_caches`]); 0 where it is not.
fn ensure_room_for_prompt(
    model: &SplitModel,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    request_id: &str,
    prompt_tokens: usize,
    reply_positions: usize,
    forward_width: usize,
) -> Result<(), SwarmError> {
    let width = model.layer_end.saturating_sub(model.layer_start) as u32;
    admit_prompt_over(
        &[(model, width)],
        KvGrowth::default(),
        kv_store,
        prefix_cache,
        request_id,
        prompt_tokens,
        reply_positions,
        forward_width,
        Admitting::Record,
    )
    .map(|_| ())
    .map_err(|refused| refused.error)
}

/// What an admission may change. A REQUIRED argument of [`admit_prompt_over`]:
/// the one decision answers all three, and a peer's question writes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admitting {
    /// This worker will run the prompt: record its reservation and promise,
    /// end finished conversations and evict saved prompts to make the room.
    Record,
    /// Up front, for every range at once, BEFORE any forward has looked up the
    /// prompt opening it will restore (`DaemonMsg::AdmitPrompt`, #263): as
    /// `Record`, except that a prompt fitting only once saved prompts go is
    /// left to its forwards' own admissions (`Admitted::LeftToTheForwards`).
    /// Evicting here took the very opening the pass was about to resume from —
    /// it had not been looked up, so it was the least recently used — and the
    /// miss sent the whole prompt to every segment again (field report
    /// 2026-10-19: 103 s on a peer that had kept its part). A forward looks its
    /// opening up first and holds it, so its eviction cannot take it.
    UpFront,
    /// A PEER asks whether it would fit (`PromptAdmissionAsk`, FUTURE_WORK
    /// #252 (b)): the same arithmetic, counting the room the release and the
    /// eviction WOULD free, and nothing written — vLLM's `can_allocate`, not an
    /// allocation. Room held for a machine that may never send its forwards
    /// was a reservation any peer could take for the cache TTL (security
    /// review, 2026-10-18).
    OnlyWeigh,
}

impl Admitting {
    /// Does this admission record what it admits (a reservation, a promise)
    /// and make room (finished conversations, saved prompts)?
    fn writes(self) -> bool {
        self != Self::OnlyWeigh
    }
}

/// What [`admit_prompt_over`] did with a prompt it did not refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admitted {
    /// Weighed, and — for a writing admission — promised and recorded.
    Recorded,
    /// Up front (`Admitting::UpFront`), a prompt that fits only once saved
    /// prompts go: nothing promised, nothing evicted, and each forward admits
    /// itself after looking its opening up.
    LeftToTheForwards,
}

/// [`admit_prompt_in`]'s answer in one memory.
enum Room {
    /// Fits: the bytes promised there.
    Promised(u64),
    /// No range keeps a budget there.
    Unbudgeted,
    /// Fits once saved prompts go — answered only up front, which evicts none.
    AfterEvicting,
}

/// A prompt [`admit_prompt_over`] would not take: the cache it `needed`, in
/// bytes over every range, and how far the room fell `short_by` — what
/// [`layers_that_fit`] turns into the figure a re-plan holds this node to
/// (FUTURE_WORK #263).
struct PromptRefusal {
    error: SwarmError,
    needed: u64,
    short_by: u64,
}

/// How many of `layers` a prompt's cache fits in, when the room holds
/// `needed - short_by` of the `needed` bytes it asked for: every layer costs
/// the prompt the same, so the same share of them, rounded DOWN — a re-plan
/// held to this must fit, never fall one layer short again.
fn layers_that_fit(layers: u32, needed: u64, short_by: u64) -> u32 {
    if needed == 0 {
        return layers;
    }
    let room = needed.saturating_sub(short_by);
    // In u128: a peer's question names the layers (#252 (b)), and a width near
    // `u32::MAX` times a room near `u64::MAX` overflows u64 — a panic in a
    // debug build, the whole worker with it.
    u32::try_from(u128::from(layers) * u128::from(room) / u128::from(needed)).unwrap_or(layers)
}

/// The layers `ranges` name, together. Saturating: the ranges may be a peer's
/// (#252 (b)), each only checked `start < end`.
fn layers_named(ranges: &[(u32, u32)]) -> u32 {
    ranges.iter().fold(0u32, |sum, (start, end)| {
        sum.saturating_add(end.saturating_sub(*start))
    })
}

/// The ONE admission decision for a prompt's cache, over every range of this
/// worker the prompt will run through: `ranges` pairs each loaded model with
/// the layers of it the prompt uses (its width, or fewer where a plan's range
/// sits inside a loaded one), and `unloaded` charges ranges not loaded
/// yet. One range is a segment's forward or a `Generate`, exactly as
/// before; several are an up-front admission (`DaemonMsg::AdmitPrompt`). The
/// ranges share one card and one store, so the tightest of their budgets binds.
///
/// Decided in EVERY memory the ranges keep caches in (`SplitModel::kv_homes`,
/// #262): a card/processor split's card-held layers against the card, its
/// processor-held layers against the machine, and what it promises is recorded
/// only once every one has room.
#[allow(clippy::too_many_arguments)]
fn admit_prompt_over(
    ranges: &[(&SplitModel, u32)],
    unloaded: KvGrowth,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    request_id: &str,
    prompt_tokens: usize,
    reply_positions: usize,
    forward_width: usize,
    admitting: Admitting,
) -> Result<Admitted, PromptRefusal> {
    let Some(context_window) = ranges.iter().map(|(m, _)| m.context_window()).min() else {
        return Ok(Admitted::Recorded);
    };
    // The prompt plus the reply's reserve, rounded to whole quanta, so an
    // admitted request can decode its whole reply (or its first quantum, for
    // a long one) without meeting the per-chunk guard. A reply longer than a
    // quantum may still meet it later, which is the lazy charge that guard
    // exists for. With the full quantum as the reserve this is exactly the
    // figure it always was: rounding `p + 512` up equals rounding `p` up,
    // plus 512.
    let positions = crate::inference::layers::kv_cache_reservation(
        prompt_tokens.saturating_add(reply_positions),
    );
    // This is where the prompt's length is first known, so this is where its
    // caches learn how big to be born: every layer's first allocation covers
    // these positions, instead of being grown into a quantum at a time with
    // a copy of everything so far at each step (FUTURE_WORK #32). Recorded
    // before any budget question, because a node with no budget to check
    // still pays for the growth.
    if admitting.writes() {
        kv_store.set_reserved_positions(request_id, positions.min(context_window));
    }
    if !prefix_cache_charged() {
        return Ok(Admitted::Recorded);
    }
    let mut homes: Vec<KvHome> = Vec::new();
    for &(model, _) in ranges {
        for home in model.kv_homes() {
            if !homes.contains(&home) {
                homes.push(home);
            }
        }
    }
    let mut promises: Vec<(KvHome, u64)> = Vec::new();
    let mut after_evicting = false;
    for (i, &home) in homes.iter().enumerate() {
        // A range not loaded yet is charged where the loaded ones keep their
        // primary caches — the first memory asked.
        let unloaded = if i == 0 {
            unloaded
        } else {
            KvGrowth::default()
        };
        let promise = admit_prompt_in(
            home,
            ranges,
            unloaded,
            kv_store,
            prefix_cache,
            request_id,
            prompt_tokens,
            reply_positions,
            positions,
            forward_width,
            admitting,
        )?;
        match promise {
            Room::Promised(bytes) => promises.push((home, bytes)),
            Room::Unbudgeted => {}
            Room::AfterEvicting => after_evicting = true,
        }
    }
    if !admitting.writes() {
        return Ok(Admitted::Recorded);
    }
    if after_evicting {
        return Ok(Admitted::LeftToTheForwards);
    }
    // Record what this prompt has been promised, so the NEXT one is weighed
    // against it rather than against memory it is about to take. Released by
    // `clear_request`, which every path that finishes or abandons a request
    // already calls.
    for (home, bytes) in promises {
        kv_store.record_prompt_admission(request_id, home, bytes);
    }
    Ok(Admitted::Recorded)
}

/// [`admit_prompt_over`]'s decision in ONE memory: the room there, else the
/// refusal.
#[allow(clippy::too_many_arguments)]
fn admit_prompt_in(
    home: KvHome,
    ranges: &[(&SplitModel, u32)],
    unloaded: KvGrowth,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    request_id: &str,
    prompt_tokens: usize,
    reply_positions: usize,
    positions: usize,
    forward_width: usize,
    admitting: Admitting,
) -> Result<Room, PromptRefusal> {
    let occupancy = kv_store.occupancy();
    let in_home = occupancy.allocated_in(home);
    // Prompts already admitted whose prefill has not written its cache yet.
    // Without this the decision is made against an occupancy that does not
    // contain them: on the batched path `admit_slot` marks a slot `Prefilling`
    // and returns to the message loop, so the next prompt is weighed against
    // memory the previous one has already been promised. Drawn down by what
    // each has actually allocated, so nothing is charged twice.
    let promised = kv_store.outstanding_admission_bytes(home);
    let mut live = in_home.saturating_add(promised);
    let cached = prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64;
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
            memory = ?home,
            "DIAG: KV admission — the store already holds caches from earlier requests"
        );
    }
    // The budget as the card can honour it NOW, not as predicted at load. The
    // load-time figure cannot see a tenant that arrived after it — another
    // model's worker, the full build's second CUDA context — and on the
    // released v0.3.149 it admitted a prompt into ~300 MB of real room, which
    // WSL2 served from host memory at 1.95 tok/s rather than refusing.
    let mut budget: Option<u64> = None;
    let mut load_time_budget: Option<u64> = None;
    let mut growth = unloaded;
    for &(model, used) in ranges {
        let (now, model_growth) = model.kv_budget_now(home, live, cached);
        // A range of which the prompt uses part pays for that part.
        let width = model.layer_end.saturating_sub(model.layer_start).max(1) as u64;
        growth = growth.plus(model_growth.share(u64::from(used).min(width), width));
        if let Some(b) = now {
            budget = Some(budget.map_or(b, |kept| kept.min(b)));
        }
        if let Some(b) = model.kv_budget(home).0 {
            load_time_budget = Some(load_time_budget.map_or(b, |kept| kept.min(b)));
        }
    }
    let growth = growth.widened_to(forward_width);
    let Some(budget) = budget else {
        return Ok(Room::Unbudgeted);
    };
    if growth.per_position() == 0 {
        return Ok(Room::Unbudgeted);
    }
    let load_time_budget = load_time_budget.unwrap_or(budget);
    let mb = |b: u64| b / (1024 * 1024);
    use crate::inference::split::kv_budget::{admit_prompt, PromptAdmission};
    // What the prompt's caches will hold — a sliding window's no more than its
    // window (FUTURE_WORK #281).
    let claim = growth.bytes_for(positions);
    let mut verdict = admit_prompt(budget, live, cached, claim);
    // Conversations that are over give their room up before a cached prompt
    // does, and before this one is refused: a node serving segments for other
    // computers is never told their replies ended (report #005).
    if !matches!(verdict, PromptAdmission::Fits) {
        let idle = crate::inference::split::kv_cache::FINISHED_CONVERSATION_AFTER;
        if admitting == Admitting::OnlyWeigh {
            // The room the release would give, released by nobody: a peer's
            // question never ends another conversation.
            let would_free = kv_store.finished_conversations_bytes_in(home, idle, request_id);
            if would_free > 0 {
                live = live.saturating_sub(would_free);
                verdict = admit_prompt(budget, live, cached, claim);
            }
        } else if kv_store.release_finished_conversations(idle, request_id) > 0 {
            // What was freed in THIS memory: the release counts every device.
            live = kv_store
                .occupancy()
                .allocated_in(home)
                .saturating_add(kv_store.outstanding_admission_bytes(home));
            verdict = admit_prompt(budget, live, cached, claim);
        }
    }
    // What this prompt is promised here, recorded by the caller once every
    // memory has room.
    let promise = claim;
    let short_by = match verdict {
        PromptAdmission::Fits => {
            return Ok(Room::Promised(promise));
        }
        // `admit_prompt` asks for an eviction only when the saved prompts cover
        // the excess, and `release` can free every one of them.
        PromptAdmission::EvictBytes(_) if admitting == Admitting::OnlyWeigh => {
            return Ok(Room::Promised(promise));
        }
        PromptAdmission::EvictBytes(_) if admitting == Admitting::UpFront => {
            return Ok(Room::AfterEvicting);
        }
        PromptAdmission::EvictBytes(needed) => {
            let freed = prefix_cache.release(needed as usize, SnapshotHome::BesideLive) as u64;
            // Charging is on here (checked above), so keep the guard's figure current.
            kv_store.set_external_reserved(prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64);
            tracing::info!(
                request_id,
                prompt_tokens,
                positions,
                live_mb = mb(live),
                promised_mb = mb(promised),
                cached_mb = mb(cached),
                budget_mb = mb(budget),
                load_time_budget_mb = mb(load_time_budget),
                needed_mb = mb(needed),
                freed_mb = mb(freed),
                "DIAG: KV admission — evicted cached prompts so this prompt's cache fits on the device"
            );
            if freed >= needed {
                return Ok(Room::Promised(promise));
            }
            needed - freed
        }
        PromptAdmission::Refuse { short_by } => short_by,
    };
    if admitting.writes() {
        tracing::warn!(
            request_id,
            prompt_tokens,
            positions,
            live_mb = mb(live),
            promised_mb = mb(promised),
            cached_mb = mb(cached),
            budget_mb = mb(budget),
            load_time_budget_mb = mb(load_time_budget),
            short_by_mb = mb(short_by),
            memory = ?home,
            "DIAG: KV admission — refusing this prompt before prefill: it would not fit on the device"
        );
    }
    // "budget" unqualified is the wrong word here, and the sibling refusal in
    // `auto_manage::vram` proves it: that one says "This node's budget allows
    // N MB (`resources.max_ram_mb` is N MB)" about the WHOLE node, while this
    // figure is only what is left for conversations once the weights and
    // overhead are paid. A reader who has set `max_ram_mb = 4000` and is told
    // "budget 1822 MB" reasonably concludes their setting is being ignored.
    // Name which budget this is, and say the weights are already accounted for.
    // `LocalMemoryUnavailable`, not `ServiceUnavailable`, and the difference is
    // not cosmetic: it is the one local failure the router re-plans, and a
    // re-plan is what this refusal needs. Reported live with a five-segment
    // route across peers already priced in the same scheduling pass, discarded
    // on estimated compute cost, and never reconsidered when the plan it chose
    // turned out not to fit (report #019). The wire wording is deliberately
    // identical to the peer-side refusal — see the variant's own doc — so the
    // class travels on `WorkerMsg::Error::local_memory_refusal` rather than in
    // the text.
    let needed = promise;
    let error = SwarmError::LocalMemoryUnavailable(prompt_refusal_message(
        prompt_tokens,
        mb(live),
        mb(budget),
        mb(short_by),
        other_conversations_hold_the_room(live, budget, needed),
        longest_prompt_that_fits(budget, growth, reply_positions),
    ));
    Err(PromptRefusal {
        error,
        needed,
        short_by,
    })
}

/// Answer a `DaemonMsg::AdmitPrompt`: admit the request's prompt over every
/// range the plan gives this worker, at once, before the pipeline sends its
/// first forward (FUTURE_WORK #263). `Ok` with nothing recorded when none of
/// the ranges is loaded yet — there is nothing to weigh, and each forward then
/// admits itself as it always has. A refusal says how many of the ranges'
/// layers the room holds at this length: the fact the coordinator's re-plan
/// needs, which a forward's own refusal never carried.
fn admit_prompt_up_front(
    models: &HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    admit: &crate::inference::worker_ipc::IpcAdmitPrompt,
) -> Result<(), (SwarmError, u32)> {
    let request_id = admit.request_id.to_string();
    let admitting = if admit.only_weigh {
        Admitting::OnlyWeigh
    } else {
        Admitting::UpFront
    };
    // Each range: the loaded model holding it (a plan's range may sit inside a
    // loaded one) and the layers of it the prompt uses.
    let mut loaded: Vec<(&SplitModel, u32)> = Vec::new();
    let mut unloaded_layers = 0u32;
    for &(start, end) in &admit.layer_ranges {
        let width = end.saturating_sub(start);
        let holder = models
            .iter()
            .filter(|((_, _, tp_rank, tp_size), _)| *tp_rank == 0 && *tp_size == 1)
            .find(|((ls, le, _, _), _)| *ls <= start as usize && end as usize <= *le)
            .map(|(_, m)| m);
        match holder {
            Some(model) => loaded.push((model, width)),
            None => unloaded_layers = unloaded_layers.saturating_add(width),
        }
    }
    if loaded.is_empty() {
        return Ok(());
    }
    // A range not loaded yet costs what the loaded ones cost a layer: the same
    // model's layers on the same device.
    let (loaded_growth, loaded_layers) = loaded.iter().fold(
        (KvGrowth::default(), 0u64),
        |(growth, layers), &(model, used)| {
            let width = model.layer_end.saturating_sub(model.layer_start).max(1) as u64;
            let used = u64::from(used).min(width);
            (
                growth.plus(model.kv_budget(model.kv_home()).1.share(used, width)),
                layers + used,
            )
        },
    );
    let unloaded = loaded_growth.share(u64::from(unloaded_layers), loaded_layers.max(1));
    // The pass starts every range's cache afresh, so an earlier attempt's
    // caches for this request (a re-plan keeps the request's id) are not
    // weighed against it; the forwards would clear them anyway. A peer's
    // question names a fresh id and clears nothing.
    if admitting.writes() {
        kv_store.cleanup_request_id(&request_id);
    }
    match admit_prompt_over(
        &loaded,
        unloaded,
        kv_store,
        prefix_cache,
        &request_id,
        admit.prompt_tokens as usize,
        REPLY_RESERVE_POSITIONS,
        0,
        admitting,
    ) {
        Ok(Admitted::Recorded) => {
            if admitting.writes() {
                kv_store.record_ranges_admitted(&request_id, admit.layer_ranges.clone());
            }
            Ok(())
        }
        Ok(Admitted::LeftToTheForwards) => {
            tracing::info!(
                request_id,
                prompt_tokens = admit.prompt_tokens,
                "DIAG: KV admission up front — the prompt fits once saved prompts go; each \
                 segment admits itself after looking up the opening it restores"
            );
            Ok(())
        }
        Err(refused) => {
            // Nothing of this attempt runs here now: the reservation the
            // admission recorded before it weighed would hold room against
            // every other conversation until the re-plan's own admission.
            if admitting.writes() {
                kv_store.cleanup_request_id(&request_id);
            }
            Err((
                refused.error,
                layers_that_fit(
                    layers_named(&admit.layer_ranges),
                    refused.needed,
                    refused.short_by,
                ),
            ))
        }
    }
}

/// The longest prompt this worker admits with no other conversation held: the
/// whole reservation quanta `budget` covers, less the reply's reserve —
/// admission charges `kv_cache_reservation(prompt + reply_positions)`, a
/// multiple of the quantum. A figure the owner can act on: told only "short by
/// 259 MB", nobody learns that this card takes about 4,600 tokens of this model
/// while `/v1/models` says 8,192 (field report #035, FUTURE_WORK #252).
fn longest_prompt_that_fits(budget: u64, growth: KvGrowth, reply_positions: usize) -> usize {
    use crate::inference::layers::KV_CACHE_GROWTH_TOKENS;
    if growth.per_position() == 0 {
        return 0;
    }
    let positions = growth.positions_within(budget);
    (positions / KV_CACHE_GROWTH_TOKENS * KV_CACHE_GROWTH_TOKENS).saturating_sub(reply_positions)
}

/// The longest conversation, prompt and reply together, that the model this
/// worker holds WHOLE admits with nothing else held, in each memory its caches
/// are charged to (`SplitModel::kv_homes`), the least of them. `None` when no
/// loaded range covers every layer (the node's limit then depends on its
/// peers), when no memory's budget is known (an unknown budget admits
/// everything), or when nothing fits at all (the refusal says so). What
/// `/v1/models` reports while this worker lives (`WorkerMsg::ContextFits`,
/// FUTURE_WORK #252 (1)).
///
/// The budget as admission reads it (`admit_prompt_in`: `kv_budget_now` over
/// the caches live in that memory, the promises not yet written, the cached
/// prompts beside them), which is the load-time budget capped by what the
/// device holds for caches plus what it has free — invariant under whatever
/// the conversations hold at the moment, so it describes the device, not the
/// load. The positions it holds in whole growth quanta, less `draft_margin`
/// ([`draft_margin_positions`]): admission charges
/// `kv_cache_reservation(prompt + reserve)` with the reserve
/// `min(max_tokens + draft_margin, REPLY_RESERVE_POSITIONS)`
/// ([`reply_reserve_positions`]), so EVERY request whose prompt and reply fit
/// in the figure is admitted, however they divide it — a reply under the cap
/// reserves itself plus the margin, one at the cap leaves its prompt shorter
/// than the reserve — and its reply, a speculative round's drafts included,
/// grows inside the budget. A refusal names the same quanta less ITS reserve.
/// (Less the whole cap instead, as first written, it read 4,608 where a
/// 14B's card admitted 5,088 — measured 2026-10-14.)
///
/// ⚠ Reads the device's free memory after a synchronize: asked when the
/// loaded ranges change and at most every [`CONTEXT_FITS_EVERY`], never per
/// message.
fn whole_model_context_fits(
    models: &HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    draft_margin: usize,
) -> Option<u32> {
    let model = models
        .iter()
        .find(|((_, _, _, tp_size), m)| {
            *tp_size == 1
                && m.total_layers > 0
                && m.layer_start == 0
                && m.layer_end >= m.total_layers
        })
        .map(|(_, m)| m)?;
    let occupancy = kv_store.occupancy();
    let cached = prefix_cache.bytes_in(SnapshotHome::BesideLive) as u64;
    context_fits_in(
        model.kv_homes().map(|home| {
            let live = occupancy
                .allocated_in(home)
                .saturating_add(kv_store.outstanding_admission_bytes(home));
            model.kv_budget_now(home, live, cached)
        }),
        draft_margin,
    )
}

/// How often an active worker re-reads [`whole_model_context_fits`] while the
/// loaded ranges stay the same: another program, or another model's worker,
/// can take the card's room after the load.
const CONTEXT_FITS_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// [`whole_model_context_fits`] over each memory's `(budget, growth)`, as
/// `SplitModel::kv_budget` answers it.
fn context_fits_in(
    budgets: impl Iterator<Item = (Option<u64>, KvGrowth)>,
    draft_margin: usize,
) -> Option<u32> {
    let tokens = budgets
        .filter_map(|budget| match budget {
            (Some(budget), growth) if growth.per_position() > 0 => {
                Some(longest_prompt_that_fits(budget, growth, draft_margin))
            }
            _ => None,
        })
        .min()?;
    (tokens > 0).then(|| u32::try_from(tokens).unwrap_or(u32::MAX))
}

/// Is this refusal only because OTHER requests hold the memory — would the
/// prompt fit if the conversations already live had finished?
///
/// Two situations share one refusal and need opposite advice (the #309
/// shape): a prompt too long for this card is helped by a shorter one, while
/// a short prompt that met three live chats is helped by nothing but waiting.
/// Told "shorter conversations still work", the second reader shortens a
/// 45-token prompt and is refused again (`docs/FUTURE_WORK.md` #122).
fn other_conversations_hold_the_room(live: u64, budget: u64, needed: u64) -> bool {
    live > 0 && needed <= budget
}

#[cfg(test)]
mod reply_reserve_tests {
    use super::*;
    use crate::inference::layers::kv_cache_reservation;

    /// What admission charges and every layer's first allocation covers.
    fn admitted_positions(prompt: usize, max_new_tokens: u32) -> usize {
        kv_cache_reservation(
            prompt + reply_reserve_positions(max_new_tokens, DRAFT_MARGIN_POSITIONS),
        )
    }

    /// #122's case: a 45-token chat asking for 128 tokens held two quanta,
    /// 384 of them unreachable. It holds one now — half the memory, so an
    /// 8 GB card with an 8B takes six such chats instead of three.
    #[test]
    fn a_short_chat_reserves_what_its_reply_can_reach() {
        assert_eq!(admitted_positions(45, 128), 512);
        assert_eq!(admitted_positions(20, 1), 512);
    }

    /// A prompt snapshot sized before its reply keeps the reply a quantum
    /// beside the live cache; one copied once the reply is out
    /// (`prompt_saved_when_done`) keeps none for a reply already written.
    #[test]
    fn a_snapshot_taken_after_the_reply_keeps_no_room_for_it() {
        assert_eq!(
            ReplyAhead::ToDecode.reserve_positions(),
            REPLY_RESERVE_POSITIONS
        );
        assert_eq!(ReplyAhead::Done.reserve_positions(), 0);
    }

    /// With a budget of a quantum or more the reserve is the quantum it always
    /// was, so a long reply is admitted exactly as before — `round_up(p +
    /// 512) == round_up(p) + 512` for every prompt length.
    #[test]
    fn a_long_reply_is_reserved_exactly_as_before() {
        for prompt in [1, 45, 511, 512, 513, 1835, 4035] {
            for budget in [480u32, 512, 4096, u32::MAX] {
                assert_eq!(
                    admitted_positions(prompt, budget),
                    kv_cache_reservation(prompt) + REPLY_RESERVE_POSITIONS,
                    "prompt {prompt}, budget {budget}"
                );
            }
        }
    }

    /// A reply ending just short of a quantum boundary keeps room for a
    /// speculative draft past it, rather than growing the cache mid-reply.
    #[test]
    fn the_reserve_leaves_room_for_a_speculative_draft() {
        assert_eq!(
            reply_reserve_positions(100, DRAFT_MARGIN_POSITIONS),
            100 + DRAFT_MARGIN_POSITIONS
        );
        assert_eq!(admitted_positions(400, 100), 1024);
    }

    /// The draft margin follows what this worker is configured to draft, and
    /// however large that is the reserve never exceeds the quantum the
    /// reservation held before it was sized by the reply.
    #[test]
    fn the_draft_margin_follows_the_configured_draft_length() {
        let swift = SwiftConfig::default();
        let mut ngram = crate::inference::ngram_lookup::NgramLookupConfig::default();
        assert_eq!(
            draft_margin_positions(&ngram, &swift),
            DRAFT_MARGIN_POSITIONS
        );
        ngram.num_pred_tokens = 64;
        assert_eq!(draft_margin_positions(&ngram, &swift), 65);
        assert_eq!(reply_reserve_positions(128, 65), 193);
        ngram.num_pred_tokens = 100_000;
        let huge = draft_margin_positions(&ngram, &swift);
        assert_eq!(reply_reserve_positions(128, huge), REPLY_RESERVE_POSITIONS);
    }

    /// Opposite advice for opposite causes. A prompt that fits the budget on
    /// its own, refused because other chats are live, must never be told to
    /// shorten itself; a prompt that does not fit even an empty cache must.
    #[test]
    fn a_refusal_names_the_cause_it_can_do_something_about() {
        assert!(other_conversations_hold_the_room(1152, 1336, 192));
        assert!(!other_conversations_hold_the_room(0, 683, 1000));
        assert!(!other_conversations_hold_the_room(500, 1336, 2000));

        let busy = prompt_refusal_message(45, 1152, 1336, 8, true, 2048);
        assert!(busy.starts_with("Not enough free memory on this node for a 45-token prompt"));
        assert!(busy.contains("try again in a moment"), "{busy}");
        assert!(!busy.contains("Shorter conversations still work"), "{busy}");

        let too_long = prompt_refusal_message(4035, 0, 1336, 400, false, 2944);
        assert!(too_long.starts_with("Not enough free memory on this node for a 4035-token prompt"));
        assert!(
            too_long.contains("Shorter conversations still work"),
            "{too_long}"
        );
        assert!(too_long.contains("up to about 2944 tokens"), "{too_long}");
        let none_fit = prompt_refusal_message(4035, 0, 100, 1300, false, 0);
        assert!(!none_fit.contains("up to about"), "{none_fit}");
    }

    /// Field report #035's figures: an 860 MB budget, a Qwen3-14B's 160 KiB a
    /// position (40 layers x 8 KV heads x 128 x K and V x f16), the default
    /// reply reserve — 4,608 tokens, where `/v1/models` said 8,192 and a
    /// 6,228-token prompt was refused.
    #[test]
    fn a_refusal_says_how_long_a_prompt_fits() {
        let per_token = 40 * 8 * 128 * 2 * 2;
        let budget = 860 * 1024 * 1024;
        let reserve = REPLY_RESERVE_POSITIONS;
        let fits = longest_prompt_that_fits(budget, KvGrowth::flat(per_token), reserve);
        assert_eq!(fits, 5120 - reserve);
        // A prompt that long is admitted: its reservation is within budget.
        let charged = kv_cache_reservation(fits + reserve) as u64 * per_token;
        assert!(charged <= budget, "{charged} > {budget}");
        // One token more is not.
        let over = kv_cache_reservation(fits + 1 + reserve) as u64 * per_token;
        assert!(over > budget, "{over} <= {budget}");
        assert_eq!(
            longest_prompt_that_fits(budget, KvGrowth::default(), reserve),
            0
        );
        assert_eq!(
            longest_prompt_that_fits(100, KvGrowth::flat(per_token), reserve),
            0
        );
    }

    /// A model whose sliding-window layers keep only their window (#281)
    /// holds a far longer conversation in the same budget — Gemma 3's shape,
    /// one global layer in six — and every prompt inside the figure is
    /// admitted: the figure is the inverse of what admission charges.
    #[test]
    fn a_windowed_model_fits_the_conversation_its_windows_leave_room_for() {
        let per_layer: u64 = 8 * 256 * 2 * 2;
        let flat = KvGrowth::flat(per_layer * 6);
        let windowed = KvGrowth::with_window(per_layer, per_layer * 5, 1024, 512).widened_to(512);
        let budget = 860u64 * 1024 * 1024;
        let reserve = REPLY_RESERVE_POSITIONS;
        let flat_fits = longest_prompt_that_fits(budget, flat, reserve);
        let fits = longest_prompt_that_fits(budget, windowed, reserve);
        assert!(fits > 4 * flat_fits, "{fits} vs {flat_fits}");
        let charged = |prompt: usize| windowed.bytes_for(kv_cache_reservation(prompt + reserve));
        assert!(charged(fits) <= budget);
        assert!(charged(fits + REPLY_RESERVE_POSITIONS) > budget);
    }

    /// The context reported for a whole model here (#252 (1)) is the longest
    /// prompt + reply that admission takes HOWEVER the two divide it, and not
    /// one position more; in the TIGHTEST memory its caches live in — a split
    /// keeps its card layers' caches on the card and the rest in RAM. A memory
    /// whose budget is unknown admits everything and bounds nothing; a model
    /// with no room at all reports no figure, never a context of 0.
    #[test]
    fn the_context_a_whole_model_holds_is_its_tightest_memorys() {
        let margin = DRAFT_MARGIN_POSITIONS;
        // Field report #035's card: 860 MB, Qwen3-14B's 160 KiB a position —
        // 5,120 positions in whole quanta.
        let per_token: u64 = 40 * 8 * 128 * 2 * 2;
        let budget = 860u64 * 1024 * 1024;
        let card = (Some(budget), KvGrowth::flat(per_token));
        let fits = context_fits_in([card].into_iter(), margin).unwrap() as usize;
        assert_eq!(fits, 5120 - margin);
        let admitted = |prompt: usize, reply: u32| {
            kv_cache_reservation(prompt + reply_reserve_positions(reply, margin)) as u64 * per_token
                <= budget
        };
        for reply in [1, 32, 100, 480, 481, 512, 600, 2000, 4000] {
            assert!(admitted(fits - reply as usize, reply), "reply {reply}");
        }
        // A prompt of `fits` and a one-token reply: one position more.
        assert!(!admitted(fits, 1), "one position more is refused");
        // Ten of the forty layers' caches in RAM: roomy RAM leaves the card's
        // figure, tight RAM sets its own (2,560 positions less the margin).
        let roomy_ram = (Some(16u64 << 30), KvGrowth::flat(per_token / 4));
        assert_eq!(
            context_fits_in([card, roomy_ram].into_iter(), margin),
            Some(5088)
        );
        let tight_ram = (Some(100u64 << 20), KvGrowth::flat(per_token / 4));
        assert_eq!(
            context_fits_in([card, tight_ram].into_iter(), margin),
            Some(2528)
        );
        assert_eq!(
            context_fits_in(
                [card, (None, KvGrowth::flat(per_token))].into_iter(),
                margin
            ),
            Some(5088)
        );
        assert_eq!(
            context_fits_in([(None, KvGrowth::flat(per_token))].into_iter(), margin),
            None
        );
        assert_eq!(
            context_fits_in([(Some(100), KvGrowth::flat(per_token))].into_iter(), margin),
            None
        );
    }

    /// **A refused prompt says how many of its layers the room holds, and a
    /// plan held to that many fits** (#263). The tester's shape: 30 of a 48-layer
    /// model's layers here, ~750 MB of cache asked for at 6,656 positions, 388 MB
    /// of room. The re-plan this figure bounds must not fall short again, so it
    /// rounds down.
    #[test]
    fn a_refused_prompt_says_how_many_of_its_layers_fit() {
        let mb = 1024 * 1024u64;
        let (needed, room) = (750 * mb, 388 * mb);
        let fit = layers_that_fit(30, needed, needed - room);
        assert_eq!(fit, 15);
        assert!(
            u64::from(fit) * needed / 30 <= room,
            "the re-plan's cache must fit"
        );
        assert!(
            u64::from(fit + 1) * needed / 30 > room,
            "and no more layers than that"
        );
        // No room at all is no layers; nothing asked for is every layer.
        assert_eq!(layers_that_fit(30, needed, needed), 0);
        assert_eq!(layers_that_fit(30, needed, 2 * needed), 0);
        assert_eq!(layers_that_fit(30, 0, 0), 30);
    }

    /// A peer names the ranges it asks about (#252 (b)), each only checked
    /// `start < end`: the arithmetic over them saturates rather than overflow,
    /// which a debug build turns into a dead worker.
    #[test]
    fn the_layers_a_peer_names_cannot_overflow_the_refusal() {
        assert_eq!(layers_named(&[(0, 4), (10, 12)]), 6);
        assert_eq!(layers_named(&[(0, u32::MAX), (1, u32::MAX)]), u32::MAX);
        assert_eq!(
            layers_that_fit(u32::MAX, u64::MAX, u64::MAX / 2),
            u32::MAX / 2
        );
    }
}

/// The words of a whole-prompt refusal, chosen by its cause. The opening —
/// "Not enough free memory on this node for a N-token prompt" — is the same
/// either way: it is how this refusal has always read in logs and on the
/// wire, and the class travels as a flag, not in the text.
fn prompt_refusal_message(
    prompt_tokens: usize,
    live_mb: u64,
    budget_mb: u64,
    short_by_mb: u64,
    busy: bool,
    fits_tokens: usize,
) -> String {
    if busy {
        format!(
            "Not enough free memory on this node for a {prompt_tokens}-token prompt right now: \
             other conversations on this node are using {live_mb} MB of the {budget_mb} MB \
             available for conversations once this model's weights are accounted for, short by \
             {short_by_mb} MB. It fits once they finish, so try again in a moment; a shorter \
             prompt will not help."
        )
    } else {
        format!(
            "Not enough free memory on this node for a {prompt_tokens}-token prompt ({live_mb} MB \
             of conversation memory in use, {budget_mb} MB available for conversations once this \
             model's weights are accounted for, short by {short_by_mb} MB). Shorter conversations \
             still work{fits}; free memory on this node (close other programs, or raise its memory \
             budget) to raise this.",
            fits = if fits_tokens > 0 {
                format!(" — up to about {fits_tokens} tokens with this model here")
            } else {
                String::new()
            }
        )
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
    mut gen: IpcGenerate,
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
    let mut held = held_reply_for(model, &gen.sampling)?;

    // Assign back, so every downstream reader of `gen.sampling.max_tokens` —
    // the generation loop, the off-by-one guard, the finish-reason check — sees
    // the budget that was actually granted rather than the one that was asked
    // for.
    gen.sampling.max_tokens = resolve_max_new_tokens(
        prompt_tokens,
        &gen.sampling,
        model.context_window(),
        gen.model_id.0.as_str(),
    )?;

    // Prefix-cache lookup: if a cached entry shares a long-enough prefix
    // with this prompt (the entry is narrowed to the shared length when it
    // is longer), hydrate the request's KV with the snapshot and only
    // forward the suffix. Try local first (free); on miss, probe cross-node
    // (Item 8 Phase 2b).
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    ensure_room_for_prompt(
        model,
        kv_store,
        prefix_cache,
        &req_id_str,
        prompt_ids.len(),
        reply_reserve_positions(
            gen.sampling.max_tokens,
            draft_margin_positions(ngram_cfg, swift_cfg),
        ),
        0,
    )?;
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(
                kv_store,
                &model_key_string,
                &req_id_str,
                snap,
                &model.cache_layouts(),
            )
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
            KeepAsked::kept_if(gen.keep_here),
        )
        .await;
    }

    if prefix_len > 0 {
        tracing::info!(
            %request_id,
            matched_tokens = prefix_len,
            total_tokens = prompt_tokens,
            "DIAG: handle_generate prefix-cache HIT — prefilling suffix only"
        );
    }
    let index_pos_start = prefix_len;
    // Where this pass saves its opening: the prompt's end, or short of it for
    // a model with recurrent layers, whose pass stops there to save and goes
    // on (`PrefixCache::save_point`, #228).
    let save_at = PrefixCache::save_point(prompt_tokens, model.carries_recurrent_state());
    let stop_to_save = (index_pos_start < save_at && save_at < prompt_tokens).then_some(save_at);
    let keep = KeepAsked::kept_if(gen.keep_here);

    // SWIFT eligibility decided up-front so the prefill, baseline decode, and
    // verify all run under the same attention kernel when speculative work is
    // active. `force_standard_attn` is the manual override; SWIFT auto-enables
    // it because draft and verify must produce identical logits.
    let swift_active = swift_cfg.enabled
        // SWIFT verifies drafts and rolls rejected ones back — impossible over
        // recurrent state (`SplitModel::carries_recurrent_state`).
        && !model.carries_recurrent_state()
        // Its drafts are drawn without the reply's grammar (#336).
        && held.is_none()
        && gen.sampling.temperature == 0.0
        && model.total_layers >= 8
        && gen.sampling.max_tokens >= (swift_cfg.gamma + 1);
    let force_attn = force_standard_attn || swift_active;

    // Draft-free speculation on the local path. `ngram_spec_eligible` owns
    // the conditions, and the admission gate asks it the same question, so
    // the two cannot disagree about which requests get here.
    let speculating = ngram_spec_eligible(&gen.sampling, ngram_cfg, swift_cfg) && !swift_active;
    // The card's stand-in for the run its guesses skip is fitted from the
    // prompt's last positions first (`CardGuesses`).
    let mut stand_in = card_stand_in_for(model, speculating);

    // Prefill — block_in_place for CPU-bound inference
    let mut manifest = Vec::new();
    let prefill = tokio::task::block_in_place(|| {
        let _g = crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
        let mut from = index_pos_start;
        if let Some(at) = stop_to_save {
            let head = model.tensor_from_ids(&prompt_ids[from..at])?;
            model.forward(&head, from, kv_store, &req_id_str)?;
            manifest = save_prompt_opening(
                model,
                kv_store,
                prefix_cache,
                &model_key_string,
                &req_id_str,
                &prompt_ids[..at],
                keep,
                ReplyAhead::ToDecode,
            );
            from = at;
        }
        let rest = model.tensor_from_ids(&prompt_ids[from..])?;
        match stand_in.as_ref() {
            Some(s) => {
                let capture = crate::inference::split::Capture {
                    layers: s.capture_layers(),
                    last: s.prompt_capture(),
                };
                model
                    .forward_keeping(&rest, from, kv_store, &req_id_str, capture, false)
                    .map(|(logits, caught)| (logits, Some(caught)))
            }
            None => model
                .forward(&rest, from, kv_store, &req_id_str)
                .map(|logits| (logits, None)),
        }
    });
    // What the pass kept for the stand-in, learned once the first token is out
    // (`CardGuesses::take_the_prompt`).
    let mut prompt_caught: Option<(HashMap<usize, candle_core::Tensor>, usize)> = None;
    let logits = match prefill {
        Ok((l, caught)) => {
            if let (Some(_), Some(caught)) = (stand_in.as_ref(), caught) {
                let rows = caught.values().next().map_or(0, |t| t.dim(1).unwrap_or(0));
                prompt_caught = Some((caught, rows));
            }
            l
        }
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
    // prefix skip the prefill work — unless the pass saved short of the end
    // above, or the copy waits for the reply (`prompt_saved_when_done`).
    // insert_from_kv is a no-op when the prompt is shorter than the
    // configured floor or the cache is off.
    let save_when_done = save_at == prompt_tokens && prompt_saved_when_done(model);
    if save_at == prompt_tokens && !save_when_done {
        manifest = save_prompt_opening(
            model,
            kv_store,
            prefix_cache,
            &model_key_string,
            &req_id_str,
            &prompt_ids,
            keep,
            ReplyAhead::ToDecode,
        );
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

    // The reply, as one block: however it ends, the prompt's snapshot left
    // for its end is copied below before the error, if any, goes back.
    // What the reply's stand-in learned, for the model's prior once the reply
    // is out (`absorb_the_reply`).
    let mut finished_stand_in: Option<crate::inference::split::RunStandIn> = None;
    let replied: Result<(), SwarmError> = async {
        let use_logprobs = gen.sampling.logprobs;
        let (mut next_token, mut token_logprob) =
            crate::inference::tensor_util::sample_token_with_logprob(
                &logits,
                &gen.sampling,
                held.as_mut(),
            )?;

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
            // Draft-free speculation on the local path (`speculating`, above).
            //
            // Works at ANY temperature: sampling each position with the real
            // sampler and keeping a draft only when it matches is the
            // speculative-sampling rejection rule itself, not an approximation of
            // it. See the note on `ngram_spec_eligible`.
            // Only built when it will be used — it copies the whole prompt.
            let mut spec = speculating.then(|| {
                LocalSpec::new(*ngram_cfg, &prompt_ids, model, local_guess_gamma_fixed())
                    .with_stand_in(stand_in.take(), prompt_caught.take())
            });

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
                if let Some(matched) = crate::inference::sampling::find_stop_sequence(
                    &accumulated_text,
                    stop_sequences,
                ) {
                    matched_stop_sequence = Some(matched.to_string());
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

                if let Some(spec) = spec.as_mut() {
                    next_token = spec.next(
                        model,
                        kv_store,
                        &model_key_string,
                        &req_id_str,
                        &mut index_pos,
                        next_token,
                        &generated,
                        &gen.sampling,
                        force_attn,
                    )?;
                    token_logprob = None;
                } else {
                    let input = model.token_tensor(next_token)?;
                    let logits = tokio::task::block_in_place(|| {
                        let _g =
                            crate::inference::attn_kernel::ForceStandardAttnGuard::new(force_attn);
                        model.forward(&input, index_pos, kv_store, &req_id_str)
                    })?;
                    // Pass `generated` so frequency_penalty / presence_penalty
                    // are honored against the completion-so-far per OpenAI spec.
                    let (tok, lp) =
                        crate::inference::tensor_util::sample_token_with_logprob_history(
                            &logits,
                            &gen.sampling,
                            &generated,
                            held.as_mut(),
                        )?;
                    next_token = tok;
                    token_logprob = lp;
                    index_pos += 1;
                }
            }

            if let Some(spec) = spec.as_ref().filter(|s| s.rounds > 0) {
                let tokens_per_round = spec.produced as f64 / spec.rounds as f64;
                // Steers the next admission decision — see
                // `spec_payoff_justifies_diverting`.
                record_spec_payoff(tokens_per_round);
                tracing::debug!(
                    request_id = %request_id,
                    rounds = spec.rounds,
                    drafted = spec.produced,
                    accepted = spec.accepted,
                    rejected = spec.rejected,
                    paused_rounds = spec.backoff.paused_rounds,
                    tokens_per_round,
                    learned = %spec.summary(),
                    "DIAG: local n-gram speculation complete"
                );
            }
            finished_stand_in = spec.and_then(|s| s.card).and_then(|c| c.stand_in);

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
        Ok(())
    }
    .await;

    // The prompt's snapshot, copied now that the reply is out — or has ended
    // any way this worker survives ([`prompt_outlives`]). Its first
    // `prompt_tokens` positions are the prompt pass's: decode only appends
    // and a rejected guess is cut back no further than the prompt's end.
    let failure = replied.as_ref().err().map(ToString::to_string);
    if save_when_done && prompt_outlives(failure.as_deref()) {
        let blocks = save_prompt_opening(
            model,
            kv_store,
            prefix_cache,
            &model_key_string,
            &req_id_str,
            &prompt_ids,
            keep,
            ReplyAhead::Done,
        );
        if !blocks.is_empty() {
            let _ = send_worker(
                writer,
                &WorkerMsg::PrefixManifestUpdate {
                    model_id: model_id.clone(),
                    blocks,
                },
                &[],
            )
            .await;
        }
    }

    replied?;
    // Free KV cache for this request to prevent VRAM leak across requests
    kv_store.clear_request(model.kv_model_key(), &req_id_str);
    if let Some(stand_in) = finished_stand_in {
        absorb_the_reply(model, stand_in);
    }

    Ok(())
}

/// Guess `gamma` tokens with this model for a bigger one's request
/// (`DaemonMsg::Draft`) — the drafter of speculation across computers, run in
/// this engine from the model's shards instead of in llama.cpp from a whole
/// model file (which a node must never assemble from shards, CLAUDE.md).
///
/// The cache lives on between calls, keyed by the speculating ATTEMPT's own id
/// (`IpcDraft::request_id`, `engine_drafter::draft_key`) in this worker's own
/// store, and released when that attempt ends (`ReleaseRequestKv` for that id).
/// See `IpcDraft` for what each call reads and leaves.
/// A guess is drawn with the shared noise when the request carries a seed —
/// the draw the target's sampler will make at that position — and is the
/// argmax otherwise, exactly as the llama.cpp drafter
/// (`pipeline::speculative::draft_next_gamma`) picks.
#[allow(clippy::too_many_arguments)]
async fn handle_draft(
    writer: &mut IpcWriter,
    models: &mut HashMap<(usize, usize, usize, usize), SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    data_dir: &std::path::Path,
    shard_window: &Option<Vec<u32>>,
    prefill_chunk_tokens: usize,
    flow_sessions: &mut FlowSessions,
    d: IpcDraft,
) -> Result<(), SwarmError> {
    let (layer_start, layer_end) = (d.layer_range.0 as usize, d.layer_range.1 as usize);
    ensure_model_loaded(
        models,
        data_dir,
        &d.model_id,
        layer_start,
        layer_end,
        0,
        1,
        shard_window,
    )?;
    let key = (layer_start, layer_end, 0, 1);
    ensure_whole_model_for_generate(models, key, &d.model_id)?;
    let model = models
        .get_mut(&key)
        .ok_or_else(|| SwarmError::Internal("Model vanished after load".into()))?;
    let req = d.request_id.to_string();
    if let Some(ask) = &d.flow {
        if d.for_the_owner {
            kv_store.mark_owner_request(&req);
        }
        let proposed = tokio::task::block_in_place(|| {
            grow_flow_call(
                model,
                kv_store,
                prefix_cache,
                flow_sessions,
                &req,
                &d,
                ask,
                prefill_chunk_tokens,
            )
        })?;
        let (parents, tokens) = proposed
            .into_iter()
            .map(|(parent, token)| (parent + 1, token))
            .unzip();
        return send_worker(
            writer,
            &WorkerMsg::Drafted {
                request_id: d.request_id,
                tokens,
                parents,
                scores: Vec::new(),
            },
            &[],
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("send Drafted: {e}")));
    }
    if d.append.is_empty() {
        return Err(SwarmError::Validation(
            "a draft call must name the token its first guess follows".into(),
        ));
    }
    if d.for_the_owner {
        kv_store.mark_owner_request(&req);
    }
    let keep = d.keep as usize;
    let gamma = d.gamma as usize;
    if keep == 0 {
        // The first call reads the whole prompt: admit it as a prompt, with
        // room for the reply's guesses, before any memory is claimed. A later
        // call is cut back to `keep` by `draft_after`, which refuses a cache
        // holding less.
        ensure_room_for_prompt(
            model,
            kv_store,
            prefix_cache,
            &req,
            d.append.len(),
            // A tree's guesses sit in the cache while it grows, a chain's
            // after the call.
            reply_reserve_positions(
                d.sampling.max_tokens,
                gamma
                    .max(
                        d.tree
                            .as_ref()
                            .map_or(0, |t| (t.rows as usize).saturating_add(t.seed.tokens.len())),
                    )
                    .saturating_add(1),
            ),
            0,
        )?;
    }
    let noise = d
        .coupling_seed
        .map(crate::inference::coupled_noise::CoupledNoise::new);
    let (tokens, parents, scores) = tokio::task::block_in_place(|| match &d.tree {
        Some(tree) => model
            .draft_tree_after(
                kv_store,
                &req,
                keep,
                &d.append,
                crate::inference::split::TreeGrowth {
                    seed: &tree.seed.grown(),
                    rows: tree.rows as usize,
                    per_step: tree.per_step as usize,
                    grow_for: tree
                        .grow_for_ms
                        .map(|ms| std::time::Duration::from_millis(u64::from(ms))),
                },
                &d.sampling,
                &d.history,
                noise.as_ref(),
                prefill_chunk_tokens,
            )
            .map(|grown| {
                let wire = grown
                    .parents
                    .iter()
                    .map(|p| p.map_or(0, |p| p + 1))
                    .collect();
                (grown.tokens, wire, grown.path_logp)
            }),
        None => model
            .draft_after(
                kv_store,
                &req,
                keep,
                &d.append,
                gamma,
                &d.sampling,
                &d.history,
                noise.as_ref(),
                prefill_chunk_tokens,
                d.stop_below,
            )
            .map(|tokens| (tokens, Vec::new(), Vec::new())),
    })?;
    send_worker(
        writer,
        &WorkerMsg::Drafted {
            request_id: d.request_id,
            tokens,
            parents,
            scores,
        },
        &[],
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send Drafted: {e}")))
}

/// The drafter's trees that flow (#134 (b), `SplitModel::grow_flow`), each
/// kept between its reply's calls under the attempt's draft key — the key its
/// cache is kept under — and gone with it: `ReleaseRequestKv`, a cancel, or
/// any error (a session that failed is never used again). At most
/// [`MOST_FLOW_SESSIONS`]; the one used longest ago goes first, and its next
/// call is refused as a cache that expired, which the drafter's caller
/// already answers by starting again.
#[derive(Default)]
struct FlowSessions(HashMap<Uuid, (std::time::Instant, crate::inference::split::FlowGrowth)>);

/// How many replies' trees one drafter worker keeps: a drafter serves the
/// replies of the splits this node drives, a handful at most.
const MOST_FLOW_SESSIONS: usize = 16;

impl FlowSessions {
    fn take(&mut self, id: &Uuid) -> Option<crate::inference::split::FlowGrowth> {
        self.0.remove(id).map(|(_, g)| g)
    }

    fn keep(&mut self, id: Uuid, growth: crate::inference::split::FlowGrowth) {
        while self.0.len() >= MOST_FLOW_SESSIONS {
            let Some(oldest) = self
                .0
                .iter()
                .min_by_key(|(_, (at, _))| *at)
                .map(|(id, _)| *id)
            else {
                break;
            };
            self.0.remove(&oldest);
        }
        self.0.insert(id, (std::time::Instant::now(), growth));
    }

    fn release(&mut self, id: &Uuid) {
        self.0.remove(id);
    }

    fn forget_cancelled(&mut self, cancelled: &CancelledSet) {
        if !self.0.is_empty() {
            self.0.retain(|id, _| !cancelled.contains_key(id));
        }
    }
}

/// The far runs of chains whose first and last segments this worker both
/// holds — one per request its coordinator guesses at the head of
/// (`IpcForward::far_run`, `DaemonMsg::HeadGuess`; plan
/// `docs/plans/split_speculation.md` § "Guess at the head through a stand-in
/// for the far layers"). Each learns what the far layers did to the hidden
/// state between this worker's two segments, from the positions the chain's
/// checks KEPT, as a split's card learns its processor run ([`CardGuesses`]):
/// a forward's rows wait until a later forward of the first segment (or a
/// guess) starts past them. Gone with the request — `ReleaseRequestKv` folds
/// it into the model's prior; a cancel or an error drops it, never the reply.
/// At most [`MOST_FLOW_SESSIONS`], the one used longest ago going first.
#[derive(Default)]
struct FarRuns(HashMap<Uuid, FarRun>);

type RangeKey = (usize, usize, usize, usize);

struct FarRun {
    run: (usize, usize),
    /// The last segment, whose model keeps the run's prior.
    last: Option<RangeKey>,
    stand_in: crate::inference::split::RunStandIn,
    /// The first segment's outputs and the last segment's inputs at positions
    /// not yet known kept, each `(first position, [1, rows, hidden])`.
    out_of_first: Vec<(usize, candle_core::Tensor)>,
    into_last: Vec<(usize, candle_core::Tensor)>,
    /// The prompt is learned beside the model's prior, once, at the first
    /// commit — after the first token, never before it.
    prompt_learned: bool,
    used: std::time::Instant,
    asked: u64,
    guessed: u64,
    guess_ms: f64,
    rows: u64,
}

/// Rows a side of a far run holds unkept before the oldest go: a prompt
/// read in pieces, or decode forwards no commit followed.
const MOST_UNKEPT_BLOCKS: usize = 64;

impl FarRuns {
    /// A forward's hidden state at its request's far run: the first
    /// segment's output (`near`) or the last segment's input, from position
    /// `start`. A request seen for the first time gets a stand-in.
    fn keep(
        &mut self,
        request_id: Uuid,
        run: (usize, usize),
        side: (bool, RangeKey),
        start: usize,
        h: candle_core::Tensor,
        device: &candle_core::Device,
    ) {
        let (near, range) = side;
        if !self.0.contains_key(&request_id) {
            while self.0.len() >= MOST_FLOW_SESSIONS {
                let Some(oldest) = self.0.iter().min_by_key(|(_, f)| f.used).map(|(id, _)| *id)
                else {
                    break;
                };
                self.0.remove(&oldest);
            }
            let Some(stand_in) =
                crate::inference::split::RunStandIn::for_far_run(run.0, run.1, device)
            else {
                return;
            };
            let stand_in = stand_in.with_carry(
                card_carry_share(),
                crate::inference::split::STAND_IN_CARRY_POSITIONS,
            );
            let stand_in = if card_prior_on() {
                stand_in.collecting()
            } else {
                stand_in
            };
            self.0.insert(
                request_id,
                FarRun {
                    run,
                    last: None,
                    stand_in,
                    out_of_first: Vec::new(),
                    into_last: Vec::new(),
                    prompt_learned: false,
                    used: std::time::Instant::now(),
                    asked: 0,
                    guessed: 0,
                    guess_ms: 0.0,
                    rows: 0,
                },
            );
        }
        let Some(far) = self.0.get_mut(&request_id).filter(|f| f.run == run) else {
            return;
        };
        far.used = std::time::Instant::now();
        if !near {
            far.last = Some(range);
        }
        // A prompt keeps only the positions its fit reads, COPIED: a narrowed
        // view would hold the whole pass's output.
        let rows = h.dim(1).unwrap_or(0);
        let most = far.stand_in.prompt_capture();
        let (start, h) = if rows > most {
            match h.narrow(1, rows - most, most).and_then(|t| t.copy()) {
                Ok(t) => (start + rows - most, t),
                Err(_) => return,
            }
        } else {
            (start, h)
        };
        let side = if near {
            &mut far.out_of_first
        } else {
            &mut far.into_last
        };
        if side.len() >= MOST_UNKEPT_BLOCKS {
            side.remove(0);
        }
        side.push((start, h));
    }

    /// Fold a finished request's far run into its model's prior.
    fn release(&mut self, request_id: &Uuid, models: &mut HashMap<RangeKey, SplitModel>) {
        let Some(far) = self.0.remove(request_id) else {
            return;
        };
        // Once a reply, and only on a chain that starts and finishes here.
        tracing::info!(
            %request_id,
            run = ?far.run,
            asked = far.asked,
            guessed = far.guessed,
            guess_ms = far.guess_ms.round(),
            learned_rows = far.rows,
            learn_ms = far.stand_in.ms.round(),
            "DIAG: head guesses complete"
        );
        if let Some(model) = far.last.and_then(|k| models.get_mut(&k)) {
            absorb_the_reply(model, far.stand_in);
        }
    }

    fn forget_cancelled(&mut self, cancelled: &CancelledSet) {
        if !self.0.is_empty() {
            self.0.retain(|id, _| !cancelled.contains_key(id));
        }
    }

    /// What the request's forwards kept below `upto` into its stand-in. A
    /// failure drops the far run — its guesses stop; the reply goes on.
    fn commit(
        &mut self,
        request_id: &Uuid,
        upto: usize,
        models: &mut HashMap<RangeKey, SplitModel>,
    ) {
        let Some(far) = self.0.get_mut(request_id) else {
            return;
        };
        let Some(last) = far.last.and_then(|k| models.get_mut(&k)) else {
            return;
        };
        if let Err(e) = tokio::task::block_in_place(|| far.commit(upto, last)) {
            tracing::warn!(%request_id, error = %e, "head guesses: stand-in dropped for this reply");
            self.0.remove(request_id);
        }
    }
}

impl FarRun {
    fn commit(&mut self, upto: usize, last: &mut SplitModel) -> Result<(), SwarmError> {
        let Some((near, far)) = kept_pairs(
            std::mem::take(&mut self.out_of_first),
            std::mem::take(&mut self.into_last),
            upto,
        )?
        else {
            return Ok(());
        };
        self.rows += near.dim(1).map_err(SwarmError::internal)? as u64;
        if self.prompt_learned {
            return self.stand_in.learn_pairs(&near, &far);
        }
        self.prompt_learned = true;
        let (phi, delta) = self.stand_in.pair_rows(&near, &far)?;
        learn_prompt_rows(last, &mut self.stand_in, phi, delta)
    }
}

/// A far run's two ends paired, `[1, kept, hidden]` each, in order: every
/// first-segment block with the last-segment block of the same positions,
/// cut to those below `upto` — the positions the reply kept. A block whose
/// other end never came (its forward failed, or the chain was replanned) is
/// dropped; so is everything at or past `upto`, which a check rejected.
fn kept_pairs(
    out_of_first: Vec<(usize, candle_core::Tensor)>,
    into_last: Vec<(usize, candle_core::Tensor)>,
    upto: usize,
) -> Result<Option<(candle_core::Tensor, candle_core::Tensor)>, SwarmError> {
    let mut near = Vec::new();
    let mut far = Vec::new();
    for (start, a) in out_of_first {
        let Some((_, b)) = into_last
            .iter()
            .find(|(s, b)| *s == start && b.dims() == a.dims())
        else {
            continue;
        };
        let kept = upto.saturating_sub(start).min(a.dim(1).unwrap_or(0));
        if kept == 0 {
            continue;
        }
        near.push(a.narrow(1, 0, kept).map_err(SwarmError::internal)?);
        far.push(b.narrow(1, 0, kept).map_err(SwarmError::internal)?);
    }
    if near.is_empty() {
        return Ok(None);
    }
    let cat = |parts: &[candle_core::Tensor]| {
        candle_core::Tensor::cat(parts, 1).map_err(SwarmError::internal)
    };
    Ok(Some((cat(&near)?, cat(&far)?)))
}

#[cfg(test)]
mod far_run_tests {
    use super::kept_pairs;
    use candle_core::{Device, Tensor};

    /// `rows` positions from `start`, position p's row `[2p, 2p + 1] + side`.
    fn block(start: usize, rows: usize, side: f32) -> (usize, Tensor) {
        let v: Vec<f32> = (0..rows * 2)
            .map(|i| side + (start * 2 + i) as f32)
            .collect();
        (
            start,
            Tensor::from_vec(v, (1, rows, 2), &Device::Cpu).unwrap(),
        )
    }

    /// A prompt of positions 4-9, then a check of four positions from 10 that
    /// kept two, beside a step at 14 whose last segment never answered: the
    /// pairs are positions 4-11, each end its own, in order — nothing a check
    /// rejected, nothing with one end missing.
    #[test]
    fn only_kept_positions_with_both_ends_are_paired() {
        let near = vec![block(4, 6, 0.0), block(10, 4, 0.0), block(14, 1, 0.0)];
        let far = vec![block(4, 6, 1000.0), block(10, 4, 1000.0)];
        let (a, b) = kept_pairs(near, far, 12).unwrap().expect("pairs");
        assert_eq!(a.dims(), &[1, 8, 2]);
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let want: Vec<f32> = (4..12)
            .flat_map(|p| [(2 * p) as f32, (2 * p + 1) as f32])
            .collect();
        assert_eq!(a, want);
        assert_eq!(b, want.iter().map(|x| x + 1000.0).collect::<Vec<_>>());
        let none_kept = kept_pairs(vec![block(10, 4, 0.0)], vec![block(10, 4, 1.0)], 10);
        assert!(none_kept.unwrap().is_none(), "nothing kept below its start");
    }
}

/// `DaemonMsg::HeadGuess`: commit what the request's forwards kept below the
/// guess's position, then guess through its far run's stand-in
/// (`split::draft::guesses_through_far_run`). No guesses — never an error —
/// where the stand-in has not learned enough or failed; an error only where
/// the guess itself failed, which the coordinator answers by cutting the
/// caches back with its next check.
async fn handle_head_guess(
    writer: &mut IpcWriter,
    models: &mut HashMap<RangeKey, SplitModel>,
    kv_store: &Arc<KvCacheStore>,
    far_runs: &mut FarRuns,
    g: crate::inference::worker_ipc::IpcHeadGuess,
) -> Result<(), SwarmError> {
    let key = |r: (u32, u32)| (r.0 as usize, r.1 as usize, 0usize, 1usize);
    let (first_key, last_key) = (key(g.first_range), key(g.last_range));
    let run = (first_key.1, last_key.0);
    let index_pos = g.index_pos as usize;
    far_runs.commit(&g.request_id, index_pos, models);
    let req = g.request_id.to_string();
    let guessed = tokio::task::block_in_place(|| -> Result<Vec<u32>, SwarmError> {
        let refreshed = match far_runs.0.get_mut(&g.request_id).filter(|f| f.run == run) {
            Some(far) => far.stand_in.refresh(),
            None => return Ok(Vec::new()),
        };
        if let Err(e) = refreshed {
            tracing::warn!(request_id = %g.request_id, error = %e, "head guesses: stand-in dropped for this reply");
            far_runs.0.remove(&g.request_id);
            return Ok(Vec::new());
        }
        let Some(far) = far_runs.0.get_mut(&g.request_id) else {
            return Ok(Vec::new());
        };
        let Some(bridge) = far.stand_in.bridge().cloned() else {
            return Ok(Vec::new());
        };
        if first_key == last_key {
            return Err(SwarmError::Validation(
                "a guess at the head needs two different segments".into(),
            ));
        }
        let [Some(first), Some(last)] = models.get_disjoint_mut([&first_key, &last_key]) else {
            return Err(SwarmError::ServiceUnavailable(format!(
                "this computer no longer holds both ends of request {}",
                g.request_id
            )));
        };
        let started = std::time::Instant::now();
        let guesses = crate::inference::split::guesses_through_far_run(
            first,
            last,
            &bridge,
            kv_store,
            &req,
            index_pos,
            g.token,
            g.gamma as usize,
        )?;
        far.asked += 1;
        far.guessed += guesses.len() as u64;
        far.guess_ms += started.elapsed().as_secs_f64() * 1000.0;
        Ok(guesses)
    })?;
    send_worker(
        writer,
        &WorkerMsg::Drafted {
            request_id: g.request_id,
            tokens: guessed,
            parents: Vec::new(),
            scores: Vec::new(),
        },
        &[],
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send Drafted: {e}")))
}

/// One drafting call of a tree that flows: start the session (`ask.start`,
/// after reading the context since `keep`) or take it, grow it, and keep it
/// again only if the call succeeded.
#[allow(clippy::too_many_arguments)]
fn grow_flow_call(
    model: &mut SplitModel,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &Arc<PrefixCache>,
    flow_sessions: &mut FlowSessions,
    req: &str,
    d: &IpcDraft,
    ask: &FlowAsk,
    prefill_chunk_tokens: usize,
) -> Result<Vec<(u16, u32)>, SwarmError> {
    let mut session = match ask.start {
        Some(root) => {
            flow_sessions.release(&d.request_id);
            if d.keep == 0 {
                if d.append.is_empty() {
                    return Err(SwarmError::Validation(
                        "a tree that flows starts after the context it follows".into(),
                    ));
                }
                // The first call reads the whole prompt: admit it with room
                // for the tree, which sits in the cache while it flows.
                ensure_room_for_prompt(
                    model,
                    kv_store,
                    prefix_cache,
                    req,
                    d.append.len(),
                    reply_reserve_positions(
                        d.sampling.max_tokens,
                        crate::inference::draft_tree::MAX_TREE_ROWS + 1,
                    ),
                    0,
                )?;
            }
            model.start_flow(
                kv_store,
                req,
                d.keep as usize,
                &d.append,
                root,
                prefill_chunk_tokens,
            )?
        }
        None => flow_sessions.take(&d.request_id).ok_or_else(|| {
            SwarmError::ServiceUnavailable(
                "the guessing model holds no tree for this reply — it expired or its worker \
                 was replaced"
                    .into(),
            )
        })?,
    };
    let noise = d
        .coupling_seed
        .map(crate::inference::coupled_noise::CoupledNoise::new);
    let proposed = model.grow_flow(
        kv_store,
        req,
        &mut session,
        crate::inference::split::FlowStep {
            ops: &ask.ops,
            rows: ask.rows as usize,
            levels: ask.levels as usize,
            live_most: ask.live_most as usize,
        },
        &d.sampling,
        noise.as_ref(),
        prefill_chunk_tokens,
    )?;
    flow_sessions.keep(d.request_id, session);
    Ok(proposed)
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
    Ok(crate::inference::sampling::argmax(&v))
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
    // SWIFT decoding has its own self-speculative loop; not batchable v1. A
    // reply held to JSON never runs it (#336), so it may batch.
    if swift_cfg.enabled && gen.sampling.temperature == 0.0 && gen.sampling.reply_grammar.is_none()
    {
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
    mut gen: IpcGenerate,
    shard_window: &Option<Vec<u32>>,
    slot_table: &mut SlotTable,
    card_pace: &mut crate::inference::card_pace::CardPace,
    pending_fetches: &PrefixFetchWaiterMap,
    // `draft_margin_positions` of this worker's settings. A batched slot is
    // not speculated today (`slot_admission_eligible` sends those requests
    // down the sequential path), but the reserve must not depend on that.
    draft_margin: usize,
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
    let held = held_reply_for(model, &gen.sampling).map_err(SlotAdmitError::Fatal)?;
    // Same resolution as the non-batched path — and this is the one that runs
    // by default, since `continuous_batching` is on. The slot copies
    // `gen.sampling.max_tokens` into `slot.max_tokens` below, so the assignment
    // has to happen before that, not after.
    gen.sampling.max_tokens = resolve_max_new_tokens(
        prompt_tokens,
        &gen.sampling,
        model.context_window(),
        gen.model_id.0.as_str(),
    )
    .map_err(SlotAdmitError::Fatal)?;

    // Prefix-cache lookup + per-request KV hydration if we hit. Cheap clone of
    // K/V tensors — no compute.
    let matched = prefix_cache.lookup(&model_key_string, &prompt_ids);
    // The device-bound part of admission: the budget's synchronize and
    // free-memory read, then hydration's allocations and copies. Healthy it is
    // well under 0.1 s; on 2026-09-28 it took 2-60 s, and every running slot
    // waited with it, because admission and decode share this one loop
    // (`docs/FUTURE_WORK.md` #146). Timed after the model load and before the
    // remote-prefix probe, which wait on disk and network, not on the card.
    let running = slot_table.len();
    let on_card = model.runs_entirely_on_card();
    let admission_started = std::time::Instant::now();
    let room = ensure_room_for_prompt(
        model,
        kv_store,
        prefix_cache,
        &req_id_str,
        prompt_ids.len(),
        reply_reserve_positions(gen.sampling.max_tokens, draft_margin),
        0,
    );
    // A refusal is timed too: the stall can be in the synchronize before it.
    let admission_done = |card_pace: &mut crate::inference::card_pace::CardPace| {
        if let Some(stalled) = card_pace.observe(
            admission_started.elapsed(),
            running,
            on_card,
            std::time::Instant::now(),
        ) {
            stalled.log("admission");
        }
    };
    if let Err(e) = room {
        admission_done(card_pace);
        return Err(SlotAdmitError::Fatal(e));
    }
    let mut prefix_len = match matched.as_ref() {
        Some(snap) => prefix_cache
            .hydrate_request_from_snapshot(
                kv_store,
                &model_key_string,
                &req_id_str,
                snap,
                &model.cache_layouts(),
            )
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
    admission_done(card_pace);
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
            KeepAsked::kept_if(gen.keep_here),
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
        keep_here: gen.keep_here,
        save_at: PrefixCache::save_point(prompt_tokens, model.carries_recurrent_state()),
        save_when_done: false,
        held,
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
        let chunk_size = prefill_pacer.chunk_size_for(sharing, || {
            let at = active
                .iter()
                .filter(|s| !s.is_finished())
                .filter_map(|s| s.next_prefill_pos())
                .max()
                .unwrap_or(0);
            model.prompt_reader(at)
        });
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
                // The chunk that ends at the slot's save point — the prompt's
                // end, or short of it for a recurrent model (#228) — saves the
                // opening the caches now hold.
                let read_to = step.pos + step.chunk_len;
                if read_to == slot.save_at
                    && read_to == slot.prompt_tokens
                    && prompt_saved_when_done(model)
                {
                    // Copied by the slot's finish, once its reply is out.
                    slot.save_when_done = true;
                } else if read_to == slot.save_at {
                    let manifest = save_prompt_opening(
                        model,
                        kv_store,
                        prefix_cache,
                        &slot.model_key,
                        &slot.req_id_str,
                        &slot.prompt_ids[..read_to],
                        KeepAsked::kept_if(slot.keep_here),
                        ReplyAhead::ToDecode,
                    );
                    if !manifest.is_empty() {
                        pending_announces.push((slot.model_id.clone(), manifest));
                    }
                }
                if step.remaining_after == 0 {
                    let (first_token, first_logprob) =
                        match crate::inference::tensor_util::sample_token_with_logprob(
                            &logits,
                            &slot.sampling,
                            slot.held.as_mut(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(%request_id, error = %e, "DIAG: BatchGenerate first-token sample failed — slot errored");
                                slot.finish_error(format!("first-token sample: {e}"));
                                continue;
                            }
                        };
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
                slot.held.as_mut(),
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

/// Copy the prompt snapshot a slot's pass left for when its reply is done
/// (`Slot::save_when_done`, `prompt_saved_when_done`) and announce its blocks
/// — for a finished slot and an abandoned one alike, before its caches are
/// freed. `model` is the table's (`None` once nothing names it: no copy).
async fn save_slot_prompt_when_done(
    writer: &mut IpcWriter,
    kv_store: &KvCacheStore,
    prefix_cache: &PrefixCache,
    model: Option<&SplitModel>,
    slot: &Slot,
) {
    let Some(model) = model.filter(|_| slot.save_when_done) else {
        return;
    };
    let blocks = save_prompt_opening(
        model,
        kv_store,
        prefix_cache,
        &slot.model_key,
        &slot.req_id_str,
        &slot.prompt_ids[..slot.prompt_tokens],
        KeepAsked::kept_if(slot.keep_here),
        ReplyAhead::Done,
    );
    if !blocks.is_empty() {
        let _ = send_worker(
            writer,
            &WorkerMsg::PrefixManifestUpdate {
                model_id: slot.model_id.clone(),
                blocks,
            },
            &[],
        )
        .await;
    }
}

/// Wrap up a finished slot: send GenerateDone, copy a prompt snapshot left
/// for the reply's end, free its per-request KV. The off-by-one Token (when
/// finish="length") is already emitted inside `step_decode_pool`.
async fn finalize_slot(
    writer: &mut IpcWriter,
    kv_store: &Arc<KvCacheStore>,
    prefix_cache: &PrefixCache,
    model: Option<&SplitModel>,
    slot: Slot,
) -> Result<(), SwarmError> {
    let generated_count = slot.generated_count();
    let request_id = slot.request_id;
    let finish_label = slot.finish_reason.unwrap_or("length").to_string();

    if finish_label == "error" {
        let message = slot
            .error_message
            .clone()
            .unwrap_or_else(|| "BatchGenerate slot failed without a recorded message".to_string());
        let fatal = crate::inference::worker_ipc::worker_error_is_fatal(&message);
        if prompt_outlives(Some(message.as_str())) {
            save_slot_prompt_when_done(writer, kv_store, prefix_cache, model, &slot).await;
        }
        send_worker(
            writer,
            &WorkerMsg::Error {
                request_id,
                message,
                fatal,
                // A slot that failed mid-batch carries its reason as a string
                // and nothing else; a memory refusal is decided before
                // admission, never here.
                local_memory_refusal: false,
            },
            &[],
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("send Error: {e}")))?;
        kv_store.clear_request(&slot.model_key, &slot.req_id_str);
        return Ok(());
    }

    send_worker(
        writer,
        &WorkerMsg::GenerateDone {
            request_id,
            prompt_tokens: slot.prompt_tokens,
            completion_tokens: generated_count,
            finish_reason: finish_label,
            matched_stop_sequence: slot.matched_stop_sequence.clone(),
        },
        &[],
    )
    .await
    .map_err(|e| SwarmError::Internal(format!("send GenerateDone: {e}")))?;

    save_slot_prompt_when_done(writer, kv_store, prefix_cache, model, &slot).await;
    kv_store.clear_request(&slot.model_key, &slot.req_id_str);
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
    card_pace: &mut crate::inference::card_pace::CardPace,
    pending_fetches: &PrefixFetchWaiterMap,
    cancelled: &CancelledSet,
    flow_sessions: &mut FlowSessions,
    far_runs: &mut FarRuns,
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
                prefix_cache,
                data_dir,
                fwd,
                payload,
                shard_window,
                activation_compression,
                options.prefill_chunk_tokens as usize,
                far_runs,
            )
            .await
            {
                send_worker_error(writer, request_id, e).await;
            }
        }
        DaemonMsg::HeadGuess(g) => {
            let request_id = g.request_id;
            if let Err(e) = handle_head_guess(writer, models, kv_store, far_runs, g).await {
                send_worker_error(writer, request_id, e).await;
            }
        }
        DaemonMsg::Draft(d) => {
            let request_id = d.request_id;
            if let Err(e) = handle_draft(
                writer,
                models,
                kv_store,
                prefix_cache,
                data_dir,
                shard_window,
                options.prefill_chunk_tokens as usize,
                flow_sessions,
                d,
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
                prefix_cache,
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
        DaemonMsg::Generate(mut gen) => {
            gen.rejoin_grammar();
            let request_id = gen.request_id;
            // A card that has been stalling takes no more than its pace allows,
            // on EITHER path: this sits before the choice between the batched
            // table and `handle_generate`, because a request the full table
            // turns away falls through to the sequential path and runs on the
            // card all the same (`docs/FUTURE_WORK.md` #146). Refused as the
            // busy 503 the router re-plans with this node barred from the
            // whole model — before anything is marked, so nothing is left to
            // release.
            let running = slot_table.len();
            if !card_pace.admits(running, std::time::Instant::now()) {
                let ceiling = card_pace.ceiling();
                tracing::warn!(
                    %request_id,
                    running,
                    ceiling,
                    "DIAG: card pace — refusing a new generation while the graphics card is stalling"
                );
                send_worker_error(
                    writer,
                    request_id,
                    SwarmError::LocalMemoryUnavailable(
                        crate::inference::card_pace::refusal_message(running, ceiling),
                    ),
                )
                .await;
                return false;
            }
            // Recorded once, here, before either admission path: every forward
            // of this request reads it where its thread pool is chosen
            // (`cpu_pools::in_phase_pool`), and it is released with the rest
            // of the request's bookkeeping.
            if gen.for_the_owner {
                kv_store.mark_owner_request(&request_id.to_string());
            }
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
                    card_pace,
                    pending_fetches,
                    draft_margin_positions(ngram_cfg, swift_cfg),
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
        DaemonMsg::AdmitPrompt(admit) => {
            let request_id = admit.request_id;
            let reply = match admit_prompt_up_front(models, kv_store, prefix_cache, &admit) {
                Ok(()) => WorkerMsg::PromptAdmitted { request_id },
                Err((error, layers_that_fit)) => WorkerMsg::PromptRefused {
                    request_id,
                    message: error.to_string(),
                    layers_that_fit,
                },
            };
            let _ = send_worker(writer, &reply, &[]).await;
        }
        DaemonMsg::CancelRequest { request_id } => {
            // The reader task intercepts these before they reach the main
            // loop; arriving here means the short-circuit was bypassed
            // (a direct-dispatch test harness). Record it anyway so the
            // semantics hold on either path.
            cancelled.insert(request_id, std::time::Instant::now());
        }
        DaemonMsg::ReleaseRequestKv { request_id } => {
            // The request is over. Its KV can never be read again — the entry
            // is keyed by request id, and the next turn of the same
            // conversation arrives under a new one — so holding it only
            // charges the shared budget against prompts that could otherwise
            // be admitted. The `Generate` handlers clear their own; this is
            // how a SEGMENT's entry gets released, which nothing did
            // (report #019).
            let before = kv_store.occupancy().allocated_bytes;
            kv_store.cleanup_request_id(&request_id.to_string());
            flow_sessions.release(&request_id);
            far_runs.release(&request_id, models);
            let after = kv_store.occupancy().allocated_bytes;
            if after < before {
                tracing::debug!(
                    %request_id,
                    freed_mb = (before - after) / (1024 * 1024),
                    "model-worker: released a finished request's KV cache"
                );
            }
        }
        DaemonMsg::Unload {
            layer_start,
            layer_end,
        } => {
            models.retain(|&(ls, le, _, _), _| !(ls == layer_start && le == layer_end));
            tracing::info!(layer_start, layer_end, "model-worker: unloaded shard range");
        }
        DaemonMsg::PlaceRange {
            layer_start,
            layer_end,
            placement,
        } => {
            // Recorded before the range's first forward, which this loop takes
            // after it (#265); the loader asks `split::placement_of`.
            crate::inference::split::place_range(layer_start, layer_end, placement);
            tracing::info!(
                layers = format!("[{layer_start}..{layer_end})"),
                ?placement,
                "model-worker: placing a later range where the daemon found room for it"
            );
        }
        DaemonMsg::ReadCardFree => {
            // Before anything loads: the spawn of a card/processor split sizes
            // it by what the load will read (#365).
            let free_bytes = crate::inference::split::card_free_for_a_load();
            tracing::info!(
                free_mb = free_bytes.map(|b| b / (1024 * 1024)),
                "model-worker: read the card's free memory for the daemon's split"
            );
            if let Err(e) = send_worker(writer, &WorkerMsg::CardFree { free_bytes }, &[]).await {
                tracing::warn!(error = %e, "model-worker: could not report the card's free memory");
            }
        }
        DaemonMsg::LoadRange {
            model_id,
            layer_start,
            layer_end,
        } => {
            // A peer's split will send this range its forwards; read it now so
            // the machines before this one in the split are not loading first
            // and this one after. The forward loads it the same way when it
            // was not asked for ahead (`ensure_model_loaded` is a no-op on a
            // range already held).
            let began = std::time::Instant::now();
            match ensure_model_loaded(
                models,
                data_dir,
                &model_id,
                layer_start,
                layer_end,
                0,
                1,
                shard_window,
            ) {
                Ok(()) => tracing::info!(
                    model = %model_id,
                    layers = format!("[{layer_start}..{layer_end})"),
                    load_ms = began.elapsed().as_millis() as u64,
                    "DIAG: model-worker: read a range ahead of its first forward"
                ),
                Err(e) => tracing::warn!(
                    model = %model_id,
                    layers = format!("[{layer_start}..{layer_end})"),
                    error = %e,
                    "model-worker: could not read a range ahead — its first forward \
                     tries the load again and answers with the error"
                ),
            }
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

/// Decide how many tokens this reply may actually use, against the model's real
/// context window — or refuse, with numbers the caller can act on.
///
/// **Both tokenization sites must call this**, and must use what it returns.
/// Chunked prefill discovers an overflow one 128-token chunk at a time, so the
/// executor's own guard can only report `index_pos + chunk_len` — a value just
/// past the limit whatever the prompt actually was. Every over-long prompt
/// therefore got the SAME number: a 600-word and a 1500-word prompt were both
/// told "Sequence length (4224) exceeds model context window (4096)", so
/// "reduce your prompt" gave no hint whether to cut a little or three quarters
/// of it (measured 2026-08-05). It also burned a full prefill before failing.
///
/// The executor guard stays as the backstop for paths that do not come through
/// here; this exists to make the message actionable.
///
/// # A budget the caller did not choose is a ceiling, not a demand
///
/// This used to check `params.max_tokens` and refuse whenever the sum did not
/// fit, with no regard for where that number came from. For an ABSENT
/// `max_tokens` the number was our own flat fallback, chosen with no knowledge
/// of the model — so on any model whose context window is the fallback (2048:
/// TinyLlama-1.1B-Chat, a shipped reference model) every non-empty prompt
/// whatsoever was refused, a 34-token one included. The advice compounded it:
/// "shorten it by about 34 tokens" was the whole prompt, so following it left
/// nothing to ask (report #001, reproduced on v0.3.182-alpha).
///
/// The llama.cpp executor already had this right —
/// `params.max_tokens.min(n_ctx.saturating_sub(prompt_tokens))` at
/// `executor.rs:571` — so this was one invariant implemented twice, and only
/// the candle path refused. Same rule, one helper, both paths.
fn resolve_max_new_tokens(
    prompt_tokens: usize,
    params: &SamplingParams,
    window: usize,
    model_id: &str,
) -> Result<u32, SwarmError> {
    // The prompt alone leaves no room to answer. Nothing to negotiate: this is
    // a genuine "too long", whoever picked the reply budget.
    let room = window.saturating_sub(prompt_tokens);
    if room == 0 {
        let over = prompt_tokens.saturating_sub(window).max(1);
        return Err(SwarmError::Validation(format!(
            "This conversation is too long for {model_id}: {prompt_tokens} tokens of prompt \
             against a limit of {window}, which leaves no room for a reply. Shorten it by at \
             least {over} tokens (roughly {words} words), or start a new conversation.",
            words = (over * 3) / 4,
        )));
    }

    let asked = params.max_tokens;
    if !params.max_tokens_explicit {
        // Our own default. Lower it to what the window leaves — never raise it,
        // so a long-context model does not start generating to the horizon.
        return Ok(asked.min(room as u32));
    }

    // The caller named this budget. Honour it or say why not; silently serving
    // a quarter of what was asked for is the failure mode that cannot be
    // debugged from the outside.
    let needed = prompt_tokens.saturating_add(asked as usize);
    if needed <= window {
        return Ok(asked);
    }
    Err(SwarmError::Validation(format!(
        "This conversation is too long for {model_id}: {prompt_tokens} tokens of prompt plus \
         {asked} you asked to reserve for the reply is {needed}, and the model's limit is \
         {window}. Ask for at most {room} reply tokens, or shorten the conversation."
    )))
}

/// [`resolve_max_new_tokens`] for tests outside this module — the coordinator
/// reads this refusal back (`error::served_context_refusal`), so its wording is
/// checked against the constructor that writes it, not only a frozen copy.
#[cfg(test)]
pub(crate) fn resolve_max_new_tokens_for_test(
    prompt_tokens: usize,
    params: &SamplingParams,
    window: usize,
    model_id: &str,
) -> Result<u32, SwarmError> {
    resolve_max_new_tokens(prompt_tokens, params, window, model_id)
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
mod served_context_tests {
    use super::refuse_a_prompt_past_the_served_context;

    /// The whole prompt is weighed, and the refusal names ITS length — not the
    /// chunk boundary the executor's per-chunk guard would have reported.
    #[test]
    fn a_prompt_past_the_window_is_refused_whole_with_its_real_length() {
        let err = refuse_a_prompt_past_the_served_context(8192, 0, 8560)
            .expect_err("8560 positions do not fit 8192");
        assert_eq!(
            crate::error::served_context_refusal(&err.to_string()),
            Some(crate::error::ServedContextRefusal {
                tokens: 8560,
                limit: 8192
            })
        );
    }

    /// The boundary: a prompt that exactly fills the window is served, as the
    /// executor's `total_seq > max_seq_len` guard serves it.
    #[test]
    fn a_prompt_that_fills_the_window_exactly_is_served() {
        assert!(refuse_a_prompt_past_the_served_context(8192, 0, 8192).is_ok());
        assert!(refuse_a_prompt_past_the_served_context(8192, 100, 8092).is_ok());
        assert!(refuse_a_prompt_past_the_served_context(8192, 100, 8093).is_err());
    }
}

#[cfg(test)]
mod image_span_check_tests {
    use super::{draft_tree_to_apply, image_spans_to_apply, spec_rows_fit_the_answer};
    use crate::types::PromptSpan;

    fn span(start: u32, end: u32) -> PromptSpan {
        PromptSpan { start, end }
    }

    /// The first segment takes the spans only where its own markers put the
    /// rows; a later segment takes well-formed spans inside its forward; every
    /// other list is the coordinator's mistake, refused as one (a 400).
    #[test]
    fn picture_spans_are_taken_only_where_the_prompt_puts_its_pictures() {
        // BOS, text, a marker, text, a marker: 2 pictures of 3 rows each.
        let ids = candle_core::Tensor::new(
            &[[
                2i64,
                10,
                crate::inference::vision::IMAGE_TOKEN_ID,
                11,
                crate::inference::vision::IMAGE_TOKEN_ID,
            ]],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let right = [span(2, 5), span(6, 9)];
        assert_eq!(
            image_spans_to_apply(&right, 0, 9, Some((&ids, 6))).unwrap(),
            vec![2..5, 6..9]
        );
        let shifted = [span(3, 6), span(6, 9)];
        let err = image_spans_to_apply(&shifted, 0, 9, Some((&ids, 6))).unwrap_err();
        assert!(
            matches!(err, crate::error::SwarmError::Validation(_)),
            "{err}"
        );

        // A later segment cannot check the markers, only the shape.
        assert!(image_spans_to_apply(&shifted, 0, 9, None).is_ok());
        // A forward starting after a picture does not read it; one starting
        // inside it would read half.
        assert!(image_spans_to_apply(&right, 5, 12, None).is_ok());
        let err = image_spans_to_apply(&right, 3, 12, None).unwrap_err();
        assert!(
            matches!(err, crate::error::SwarmError::Validation(_)),
            "{err}"
        );
        for bad in [
            vec![span(6, 9), span(2, 5)],
            vec![span(4, 4)],
            vec![span(2, 10)],
        ] {
            let err = image_spans_to_apply(&bad, 0, 9, None).unwrap_err();
            assert!(
                matches!(err, crate::error::SwarmError::Validation(_)),
                "{bad:?}: {err}"
            );
        }
        assert!(image_spans_to_apply(&[], 0, 1, None).unwrap().is_empty());
    }

    /// A tree of guesses (#134) is applied only to a decode-side check it fits
    /// and that this segment walks or hands on; any other shape is the
    /// coordinator's mistake (a 400), and so is a tree sent to a segment whose
    /// layers cannot read one — refused before anything runs, so the
    /// coordinator can check the same guesses as a chain.
    #[test]
    fn a_tree_of_guesses_is_applied_only_to_a_check_it_fits() {
        // Row 0 follows the cache; rows 1 and 2 follow row 0; row 3 row 1.
        let wire = [0u16, 1, 1, 2];
        let tree = draft_tree_to_apply(&wire, None, 10, 4, 3, false, None)
            .unwrap()
            .unwrap();
        assert_eq!(tree.len(), 4);
        assert!(draft_tree_to_apply(&[], None, 10, 4, 0, true, None)
            .unwrap()
            .is_none());
        // (wire, positions, seq, unwalked, the layers' refusal)
        type Shape<'a> = (&'a [u16], usize, u32, bool, Option<&'static str>);
        let refusals: [Shape<'_>; 5] = [
            (&wire, 4, 0, false, None),             // a prompt pass
            (&wire, 4, 3, true, None),              // an answer nobody walks
            (&wire, 5, 3, false, None),             // rows and positions disagree
            (&[0, 2], 2, 3, false, None),           // a row its own parent
            (&wire, 4, 3, false, Some("a window")), // layers that cannot read it
        ];
        for (wire, positions, seq, unwalked, layers_refuse) in refusals {
            let err = draft_tree_to_apply(wire, None, 10, positions, seq, unwalked, layers_refuse)
                .unwrap_err();
            assert!(
                matches!(err, crate::error::SwarmError::Validation(_)),
                "{wire:?} {positions} {seq}: {err}"
            );
        }
    }

    /// A tree that flows (#134 (b)) is its whole shape from where it starts:
    /// this forward reads its last rows, the ones before already cached, so a
    /// forward of 2 rows at position 12 of a tree from 10 carries 4 parents.
    /// A start after the forward, a shape that does not end with the forward's
    /// rows, or a start with no shape, is the requester's mistake (a 400).
    #[test]
    fn a_tree_that_flows_is_applied_from_where_it_starts() {
        let wire = [0u16, 1, 1, 2];
        let tree = draft_tree_to_apply(&wire, Some(10), 12, 2, 3, false, None)
            .unwrap()
            .unwrap();
        assert_eq!(tree.len(), 4);
        // Row 3 follows row 1, at depth 2: rotated as position 12, whatever
        // forward carries it.
        assert_eq!(tree.depth(3), 2);
        let (positions, _) = tree.rows_for(12, 2).unwrap();
        assert_eq!(positions, vec![11, 12]);
        // Starting where the forward does is the tree of one forward.
        assert_eq!(
            draft_tree_to_apply(&wire, Some(10), 10, 4, 3, false, None).unwrap(),
            draft_tree_to_apply(&wire, None, 10, 4, 3, false, None).unwrap()
        );
        type Shape<'a> = (&'a [u16], Option<u32>, usize, usize);
        let refusals: [Shape<'_>; 4] = [
            (&wire, Some(13), 12, 2), // starts after the forward
            (&wire, Some(10), 12, 3), // more rows than the shape has left
            (&wire, Some(9), 12, 2),  // a shape one row short
            (&[], Some(10), 12, 2),   // a start with no shape
        ];
        for (wire, from, index_pos, positions) in refusals {
            let err =
                draft_tree_to_apply(wire, from, index_pos, positions, 3, false, None).unwrap_err();
            assert!(
                matches!(err, crate::error::SwarmError::Validation(_)),
                "{wire:?} {from:?} {index_pos} {positions}: {err}"
            );
        }
    }

    /// A check's rows are bounded before the tail computes a vocabulary over
    /// each: a chain walked here reads one row per guess and one after; one
    /// answered with logits no more than a coordinator reads back.
    #[test]
    fn a_check_names_no_more_rows_than_its_answer_reads() {
        let most = crate::network::protocol::MAX_SPEC_LOGITS_POSITIONS;
        assert!(spec_rows_fit_the_answer(5, 4, true).is_ok());
        assert!(
            spec_rows_fit_the_answer(1, 0, true).is_ok(),
            "a one-token check"
        );
        assert!(spec_rows_fit_the_answer(17, 0, false).is_ok());
        assert!(spec_rows_fit_the_answer(most, 0, false).is_ok());
        for (positions, drafts, walked) in [(8192, 4, true), (4, 4, true), (most + 1, 0, false)] {
            let err = spec_rows_fit_the_answer(positions, drafts, walked).unwrap_err();
            assert!(
                matches!(err, crate::error::SwarmError::Validation(_)),
                "{positions} {drafts} {walked}: {err}"
            );
        }
    }
}

#[cfg(test)]
mod utf8_stream_tests {
    use super::take_complete_utf8;

    use super::resolve_max_new_tokens;
    use crate::types::SamplingParams;

    fn budget(max_tokens: u32, explicit: bool) -> SamplingParams {
        SamplingParams {
            max_tokens,
            max_tokens_explicit: explicit,
            ..Default::default()
        }
    }

    /// **The regression from report #001.** A 2048-token window with our own
    /// 2048-token fallback refused EVERY non-empty prompt: 39 + 2048 > 2048.
    /// TinyLlama-1.1B-Chat is a shipped reference model and has exactly that
    /// window, so it could not answer a single chat turn at default settings.
    ///
    /// Toggle `max_tokens_explicit` to true below and this test goes red —
    /// which is the check that it is testing the fix and not the weather.
    #[test]
    fn a_default_budget_is_lowered_to_fit_rather_than_refused() {
        let got = resolve_max_new_tokens(39, &budget(2048, false), 2048, "tinyllama")
            .expect("a 39-token prompt must be answerable on a 2048-token model");
        assert_eq!(got, 2009, "the reply gets what the prompt leaves free");
    }

    /// A budget the caller did not choose is a CEILING — lowered to fit, never
    /// raised to fill. MCP asks for 512 on a 128k model and must still get 512,
    /// not 128k: vLLM's `max_model_len - prompt_tokens` default is what causes
    /// the KV over-reservation in verl#5504.
    #[test]
    fn a_default_budget_is_never_raised_to_fill_a_large_window() {
        let got = resolve_max_new_tokens(100, &budget(512, false), 131_072, "qwen").unwrap();
        assert_eq!(got, 512);
    }

    /// A budget the caller DID choose is honoured exactly, or refused — never
    /// silently shortened, which cannot be debugged from the outside.
    #[test]
    fn an_explicit_budget_is_honoured_or_refused_but_never_shortened() {
        assert_eq!(
            resolve_max_new_tokens(4000, &budget(96, true), 4096, "m").unwrap(),
            96
        );
        let err = resolve_max_new_tokens(4000, &budget(97, true), 4096, "m").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("97"), "must name what was asked for: {msg}");
        assert!(
            msg.contains("at most 96"),
            "must name the budget that WOULD fit — advice the caller can act on: {msg}"
        );
    }

    /// **The message has to say how much to cut.** Chunked prefill discovers
    /// the overflow one chunk at a time, so the executor could only ever report
    /// a position just past the limit — a 600-word prompt and a 1500-word
    /// prompt were both told "Sequence length (4224)", which gives no idea
    /// whether to trim a sentence or three quarters of the conversation.
    #[test]
    fn an_overlong_prompt_is_told_its_real_size_and_overage() {
        let err =
            resolve_max_new_tokens(9000, &budget(100, true), 4096, "llama-3.2-3b").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("9000"),
            "must state the real prompt size: {msg}"
        );
        assert!(msg.contains("4096"), "must state the limit: {msg}");
    }

    /// **The advice must be followable.** The old message told a 39-token
    /// prompt to shorten itself by 39 tokens, which leaves nothing to ask —
    /// the #295 trap, advice that cannot work. A prompt that genuinely fills
    /// the window is a real refusal, but it must not quote the whole prompt
    /// back as the overage.
    #[test]
    fn a_prompt_that_fills_the_window_is_refused_with_advice_that_can_be_followed() {
        let err = resolve_max_new_tokens(4096, &budget(2048, false), 4096, "m").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no room for a reply"),
            "must say why, not just that it is too long: {msg}"
        );
        assert!(
            !msg.contains("ask for a shorter reply"),
            "there is no shorter reply to ask for at zero room: {msg}"
        );
    }

    /// A prompt that fits is not refused.
    #[test]
    fn a_prompt_that_fits_passes() {
        assert!(resolve_max_new_tokens(10, &budget(10, true), 4096, "m").is_ok());
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
            let mut kv = crate::inference::split::kv_cache::LayerKv::with_capacity(2, 64, 64);
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

    /// A reply guessed ahead is the reply written one token at a time — and,
    /// for Qwen 3.5, the rewind that makes it so (#228 (5)). Its DeltaNet
    /// layers summarise every token a check read, so a check that rejects a
    /// guess is taken back into by replaying the rows it accepted
    /// (`SsmState::rewind_to`). On the processor each row of a check is computed as a lone
    /// step computes it (`attn_rows`, the row-blocked quantized matmul, the
    /// delta rule's per-token loop), so the ids must match EXACTLY; on a card
    /// a forward of several positions rounds otherwise than a step of one, so
    /// there they may come apart only where the card cannot decide: at a near
    /// tie ([`CARD_NEAR_TIE_LOGITS`]) or at a token its own prompt pass does
    /// not pick either (printed with the top two either way). That a row reads
    /// nothing after it is [`a_checks_row_reads_nothing_after_it`], bit for
    /// bit on both devices. On a model with recurrent state checks must have
    /// been both accepted and rejected, or it proved nothing. Red if a check
    /// keeps no rows (the truncate into it is refused and the request fails),
    /// or if the replay reads the wrong rows or window (the reply goes its own
    /// way). It prints both speeds: read them off a release build on an idle
    /// machine.
    ///
    /// Needs a real file: `SWARMLLM_GUESS_TEST_GGUF=~/swarmllm-ref/qwen35/Qwen3.5-0.8B-Q8_0.gguf
    /// cargo dev-test --lib guessed_reply -- --ignored --nocapture`
    /// (`SWARMLLM_GUESS_TEST_ON_CARD=1` on a card build).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs a GGUF in SWARMLLM_GUESS_TEST_GGUF"]
    async fn a_guessed_reply_is_the_reply_read_one_token_at_a_time() {
        let Some(mut model) = guess_test_model() else {
            return;
        };
        let recurrent = model.carries_recurrent_state();
        // `SWARMLLM_GUESS_TEST_PROMPT` measures another reply (an open-ended
        // one guesses little), and then only the ids are checked.
        let asked = std::env::var("SWARMLLM_GUESS_TEST_PROMPT").ok();
        let prompt = model.encode_ids(asked.as_deref().unwrap_or(GUESS_TEST_CODE));
        let kv = crate::inference::split::KvCacheStore::new(std::time::Duration::from_secs(600));
        // One token at a time, every guess checked (the rule before
        // 2026-10-25, and the one that exercises the rewind most), the count
        // that pays — each twice, the second timed, so graphs captured on a
        // card's first use of a shape are not in the figure.
        let mut every = None;
        let mut chosen = None;
        let mut plain = (Vec::new(), 0.0, None);
        for pass in 0..2 {
            plain = guessed_reply(&mut model, &kv, &prompt, &format!("plain-{pass}"), None);
            every = Some(guessed_reply(
                &mut model,
                &kv,
                &prompt,
                &format!("every-{pass}"),
                Some(true),
            ));
            chosen = Some(guessed_reply(
                &mut model,
                &kv,
                &prompt,
                &format!("chosen-{pass}"),
                Some(false),
            ));
        }
        let (every, chosen) = (every.unwrap(), chosen.unwrap());
        // Where an arm came apart: the logits there, read one token at a time
        // and in one prompt pass. A card's
        // forward of several positions rounds otherwise than its step of one
        // (other matmul kernels, flash against the decode kernel), so there a
        // near tie can fall either way; a wide gap is a fault.
        let mut apart = Vec::new();
        for (name, (ids, _, _)) in [("every guess", &every), ("guesses that pay", &chosen)] {
            if let Some(at) = plain.0.iter().zip(ids).position(|(a, b)| a != b) {
                let opened = &plain.0[..at];
                let stepped = logits_after(
                    &mut model,
                    &kv,
                    &prompt,
                    opened,
                    &format!("why-step-{name}"),
                    true,
                );
                let passed = logits_after(
                    &mut model,
                    &kv,
                    &prompt,
                    opened,
                    &format!("why-pass-{name}"),
                    false,
                );
                apart.push((name, at, ids, stepped, passed));
            }
        }
        let speed = |secs: f64| GUESSED_REPLY_STEPS as f64 / secs;
        // Speed is read off a release build on an idle machine, never this
        // test's debug one.
        eprintln!(
            "one token at a time {:.1} tok/s, every guess {:.1} tok/s, guesses that pay {:.1} tok/s",
            speed(plain.1),
            speed(every.1),
            speed(chosen.1)
        );
        let text = |ids: &[u32]| {
            let mut carry = Vec::new();
            ids.iter()
                .map(|&t| decode_token(&model, t, &mut carry))
                .collect::<String>()
        };
        eprintln!("reply {:?}", text(&plain.0));
        let arms = [("every guess", &every), ("guesses that pay", &chosen)];
        for (name, (ids, _, spec)) in arms {
            let apart = plain.0.iter().zip(ids).position(|(a, b)| a != b);
            let spec = spec.as_ref().unwrap();
            eprintln!("{name}: {}; first apart {apart:?}", spec.summary());
            if let Some(at) = apart {
                // Replayed with `SWARMLLM_GUESS_TEST_SCRIPT`.
                let checked: Vec<String> =
                    spec.trace.iter().map(|t| t.checked.to_string()).collect();
                eprintln!("{name} checked: {}", checked.join(","));
                // The rounds before it: where each began, what it checked
                // and accepted, and the reply positions it wrote.
                let mut wrote = 1;
                for (j, t) in spec.trace.iter().enumerate() {
                    let from = wrote;
                    wrote += t.produced.len();
                    if wrote + 12 > at && from <= at {
                        eprintln!(
                            "{name} round {j}: cache {} checked {} accepted {} wrote {from}..{wrote} {:?}",
                            t.at, t.checked, t.accepted, t.produced
                        );
                    }
                }
            }
            // `SWARMLLM_GUESS_TEST_TRACE=1`: every round, positions:ms, `*`
            // where its forward paid a one-off on the card.
            if std::env::var("SWARMLLM_GUESS_TEST_TRACE").is_ok() {
                let rounds: Vec<String> = spec
                    .trace
                    .iter()
                    .map(|t| {
                        let p = 1 + t.checked;
                        format!("{p}:{:.1}{}", t.ms, if t.one_off { "*" } else { "" })
                    })
                    .collect();
                eprintln!("{name} rounds: {}", rounds.join(" "));
            }
        }
        for (name, at, ids, stepped, passed) in &apart {
            let (wrote, guessed) = (plain.0[*at] as usize, ids[*at] as usize);
            eprintln!(
                "{name}: apart at {at} — one at a time wrote {wrote}, guessing {guessed}; \
                 one at a time {:.3} against {:.3}, top two {:?}; in one prompt pass {:.3} against {:.3}, top two {:?}",
                stepped[wrote],
                stepped[guessed],
                top_two(stepped),
                passed[wrote],
                passed[guessed],
                top_two(passed)
            );
        }
        // On the processor the ids must match EXACTLY. On a card an arm may
        // come apart only where the card cannot decide: a near tie (one token
        // at a time wrote its likeliest token and the guessed one was within
        // `CARD_NEAR_TIE_LOGITS` of it), or a token the card's own prompt pass
        // over the same text does not pick — where llama.cpp's card build
        // parts from its processor build too (`CARD_NEAR_TIE_LOGITS`'s note).
        // A fault in a check shows where the steps and the pass agree.
        let on_card = model.device().is_cuda();
        for (name, at, ids, stepped, passed) in &apart {
            let (wrote, guessed) = (plain.0[*at] as usize, ids[*at] as usize);
            let near_tie = top_two(stepped)[0].0 as usize == wrote
                && stepped[wrote] - stepped[guessed] < CARD_NEAR_TIE_LOGITS;
            let undecided = top_two(passed)[0].0 as usize != wrote;
            assert!(
                on_card && (near_tie || undecided),
                "{name} came apart at {at}{}\none token at a time {:?}\nguessed ahead     {:?}",
                if on_card {
                    ", not at a near tie, where the card's steps and its prompt pass agree"
                } else {
                    ""
                },
                text(&plain.0),
                text(ids)
            );
        }
        if asked.is_none() {
            let spec = every.2.as_ref().unwrap();
            assert!(
                spec.accepted > 0,
                "no guess was accepted — nothing was checked"
            );
            // A rejected check's truncate into it can only succeed through
            // the replay (`SsmState::rewind_to`), so one is the mechanism.
            assert!(
                !recurrent || spec.rejected > 0,
                "no check was taken back into — the replay never ran"
            );
        }
    }

    const GUESSED_REPLY_STEPS: usize = 96;

    /// How close two logits must be for a card to pick either (2026-10-25,
    /// Qwen3.5-0.8B and 4B on an RTX 3070): where guessing came apart from
    /// one token at a time the top two were 0.013, 0.09 and 0.11 apart, and
    /// in two of the three a plain prompt pass over the same text picked the
    /// guessed token too; that pass moved a logit 0.43 from where steps put it.
    /// A tie can be three-way (Llama-3.2-3B: 18.703, 18.678 and the guessed
    /// token's), so the guessed token is weighed against the top, not second.
    ///
    /// Past a near tie, a card can be unable to decide at all. Qwen3.5-4B's
    /// story reply, position 58 ("Once upon a way"): steps — ours on either
    /// device, llama.cpp master's processor build — write "way" (1560), and
    /// guessing wrote "hope" (3769), 1.3 below it. Every several-position
    /// forward on the card picks otherwise there: our prompt pass picks 854,
    /// 8026 with `SWARMLLM_DELTA_RULE=steps` (16.568 / 15.001 for its top two)
    /// and 3769 with `SWARMLLM_QMATMUL_MMQ=dp4a`, and llama.cpp's own card
    /// build (`dump_logits_cuda2`, `DUMP_GPU_LAYERS=99`) picks 8026 (16.107 /
    /// 14.951), its last row up to 10.2 logits from its processor build's.
    const CARD_NEAR_TIE_LOGITS: f32 = 0.5;

    /// [`GUESSED_REPLY_STEPS`] greedy tokens after `prompt` under `request`:
    /// one at a time (`None`), or through [`LocalSpec`] (`Some(fixed)`).
    /// Returns the ids, the seconds the reply took, and the guessing's state.
    fn guessed_reply(
        model: &mut SplitModel,
        kv: &crate::inference::split::KvCacheStore,
        prompt: &[u32],
        request: &str,
        guessing: Option<bool>,
    ) -> (Vec<u32>, f64, Option<LocalSpec>) {
        let sampling = greedy();
        let key = model.kv_model_key().to_string();
        let pick = |logits: &candle_core::Tensor, history: &[u32]| {
            crate::inference::tensor_util::sample_token_with_logprob_history(
                logits, &sampling, history, None,
            )
            .unwrap()
            .0
        };
        let input = model.tensor_from_ids(prompt).unwrap();
        let mut next = pick(&model.forward(&input, 0, kv, request).unwrap(), &[]);
        // `SWARMLLM_GUESS_TEST_MOST=7`: at most that many guesses a check, so
        // on a card every check of the fixed arm is captured
        // (`cuda_graph::MAX_POSITIONS`) and its choices do not turn on timings.
        let cfg = NgramLookupConfig {
            num_pred_tokens: std::env::var("SWARMLLM_GUESS_TEST_MOST")
                .ok()
                .and_then(|most| most.parse().ok())
                .unwrap_or(NgramLookupConfig::default().num_pred_tokens),
            ..NgramLookupConfig::default()
        };
        let mut spec = guessing.map(|fixed| LocalSpec::new(cfg, prompt, model, fixed));
        // `SWARMLLM_GUESS_TEST_SCRIPT=3,0,7,…`: the guesses each round of the
        // guesses that pay checks, as a failing run printed them.
        if let (Some(spec), Some(false), Ok(script)) = (
            spec.as_mut(),
            guessing,
            std::env::var("SWARMLLM_GUESS_TEST_SCRIPT"),
        ) {
            spec.script = Some(
                script
                    .split(',')
                    .filter_map(|w| w.trim().parse().ok())
                    .collect(),
            );
        }
        let (mut reply, mut pos) = (Vec::new(), prompt.len());
        let started = std::time::Instant::now();
        for _ in 0..GUESSED_REPLY_STEPS {
            reply.push(next);
            next = match spec.as_mut() {
                Some(spec) => spec
                    .next(
                        model, kv, &key, request, &mut pos, next, &reply, &sampling, false,
                    )
                    .unwrap(),
                None => {
                    let input = model.token_tensor(next).unwrap();
                    let logits = model.forward(&input, pos, kv, request).unwrap();
                    pos += 1;
                    pick(&logits, &reply)
                }
            };
        }
        (reply, started.elapsed().as_secs_f64(), spec)
    }

    /// The logits after `prompt` then `reply`: read one token at a time
    /// (`stepped`), or in one prompt pass.
    fn logits_after(
        model: &mut SplitModel,
        kv: &crate::inference::split::KvCacheStore,
        prompt: &[u32],
        reply: &[u32],
        request: &str,
        stepped: bool,
    ) -> Vec<f32> {
        let logits = if stepped {
            let input = model.tensor_from_ids(prompt).unwrap();
            let mut logits = model.forward(&input, 0, kv, request).unwrap();
            for (i, &token) in reply.iter().enumerate() {
                let input = model.token_tensor(token).unwrap();
                logits = model
                    .forward(&input, prompt.len() + i, kv, request)
                    .unwrap();
            }
            logits
        } else {
            let ids: Vec<u32> = prompt.iter().chain(reply).copied().collect();
            let input = model.tensor_from_ids(&ids).unwrap();
            model.forward(&input, 0, kv, request).unwrap()
        };
        logits
            .flatten_all()
            .and_then(|l| l.to_dtype(candle_core::DType::F32))
            .and_then(|l| l.to_vec1())
            .unwrap()
    }

    /// Code that repeats its shape: guesses land on the shape and miss on each
    /// new name.
    const GUESS_TEST_CODE: &str =
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\n\
         def mul(a, b):\n    return a * b\n\ndef div(a, b):\n";

    /// The file the guessing tests read (`SWARMLLM_GUESS_TEST_GGUF`), on the
    /// card with `SWARMLLM_GUESS_TEST_ON_CARD=1`; `None` skips the test.
    /// `SWARMLLM_GUESS_TEST_LOG=swarmllm::inference::cuda_graph=info` prints
    /// the decode graph's DIAG lines (why a capture was rebuilt or refused).
    fn guess_test_model() -> Option<SplitModel> {
        let Ok(path) = std::env::var("SWARMLLM_GUESS_TEST_GGUF") else {
            eprintln!("SWARMLLM_GUESS_TEST_GGUF not set — skipped");
            return None;
        };
        if let Ok(filter) = std::env::var("SWARMLLM_GUESS_TEST_LOG") {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
                .with_writer(std::io::stderr)
                .try_init();
        }
        let model = SplitModel::load_from_gguf(
            std::path::Path::new(&path),
            0,
            999,
            true,
            true,
            std::env::var("SWARMLLM_GUESS_TEST_ON_CARD").is_err(),
        )
        .expect("load the file");
        Some(model)
    }

    /// A row of a check reads nothing after it: the same opening checked with
    /// other guesses after it gives that row bit for bit, on either device —
    /// every kernel a check runs (the projections, the convolution, the delta
    /// rule, attention) computes a row from its own position and those before
    /// it. Across WIDTHS a card's kernels may round otherwise (printed); on the
    /// processor every width gives the row bit for bit too. The check opens on
    /// the reply's own tokens: `SWARMLLM_ROW_TEST_AT` (default 8) of the reply
    /// read one token at a time, then a check of the next three and a tail.
    /// Needs a real file, as [`a_guessed_reply_is_the_reply_read_one_token_at_a_time`].
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs a GGUF in SWARMLLM_GUESS_TEST_GGUF"]
    async fn a_checks_row_reads_nothing_after_it() {
        use candle_core::IndexOp;
        let Some(mut model) = guess_test_model() else {
            return;
        };
        let asked = std::env::var("SWARMLLM_GUESS_TEST_PROMPT").ok();
        let prompt = model.encode_ids(asked.as_deref().unwrap_or(GUESS_TEST_CODE));
        let kv = crate::inference::split::KvCacheStore::new(std::time::Duration::from_secs(600));
        let reply = guessed_reply(&mut model, &kv, &prompt, "row-plain", None).0;
        let at: usize = std::env::var("SWARMLLM_ROW_TEST_AT")
            .ok()
            .and_then(|at| at.parse().ok())
            .unwrap_or(8);
        const OPENING: usize = 3;
        let (opening, after) = (&reply[at..at + OPENING], &reply[at + OPENING..]);
        // Tokens the reply does not continue with: the prompt's.
        let other = &prompt[1..];
        let tails: [(&str, &[u32]); 5] = [
            ("width 5, the reply's own", &after[..2]),
            ("width 5, other tokens", &other[..2]),
            ("width 8, the reply's own", &after[..5]),
            ("width 8, other tokens", &other[..5]),
            ("width 3", &[]),
        ];
        let next = after[0] as usize;
        let mut rows = Vec::new();
        for (i, (name, tail)) in tails.iter().enumerate() {
            let request = format!("row-{i}");
            logits_after(&mut model, &kv, &prompt, &reply[..at], &request, true);
            let ids: Vec<u32> = opening.iter().chain(tail.iter()).copied().collect();
            let input = model.tensor_from_ids(&ids).unwrap();
            let all = model
                .forward_verify_all_positions(&input, prompt.len() + at, &kv, &request)
                .unwrap();
            let row: Vec<f32> = all
                .i((0, OPENING - 1))
                .and_then(|r| r.to_dtype(candle_core::DType::F32))
                .and_then(|r| r.to_vec1())
                .unwrap();
            eprintln!(
                "{name}: top two {:?}, the reply's next token {next} at {:.3}",
                top_two(&row),
                row[next]
            );
            rows.push(row);
        }
        assert!(
            rows[0] == rows[1],
            "a row of a 5-wide check read the guesses after it"
        );
        assert!(
            rows[2] == rows[3],
            "a row of an 8-wide check read the guesses after it"
        );
        if model.device().is_cpu() {
            assert!(
                rows[0] == rows[2] && rows[0] == rows[4],
                "on the processor a row depends on the check's width"
            );
        }
    }

    /// The two likeliest tokens of a row and their logits, the first of equal
    /// maxima first (as the greedy pick takes it).
    fn top_two(row: &[f32]) -> [(u32, f32); 2] {
        let mut best = [(0u32, f32::NEG_INFINITY); 2];
        for (id, &logit) in row.iter().enumerate() {
            if logit > best[0].1 {
                best[1] = best[0];
                best[0] = (id as u32, logit);
            } else if logit > best[1].1 {
                best[1] = (id as u32, logit);
            }
        }
        best
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

    /// The test above feeds the blend samples, but a refused worker records
    /// none: only a diverted request measures. Refusals alone must bring ONE
    /// request back to measure (gotcha #985 — a 14B split's prose reply at
    /// 1.37 a round kept every later request off the card's guesses), and no
    /// further than the bar, so that request's own result decides.
    #[test]
    fn refusals_alone_bring_one_request_back_to_measure() {
        for (start, at_most) in [(137, 7), (105, 23), (1, 75)] {
            let mut v = start;
            let mut refused = 0;
            while !payoff_justifies_diverting(v) {
                v = drift_toward_retrying(v);
                refused += 1;
                assert!(
                    refused <= at_most,
                    "from {start}, still refused after {refused} solo requests"
                );
            }
            assert_eq!(v, SPEC_PAYOFF_WORTH_DIVERTING_X100, "drifted past the bar");
        }
        // A measured poor reply puts it back below the bar, so the next
        // stretch is refused again.
        assert!(!payoff_justifies_diverting(blend_spec_payoff(
            SPEC_PAYOFF_WORTH_DIVERTING_X100,
            1.0
        )));
    }

    /// 0 means "not measured". A measured-as-poor workload must never be
    /// mistaken for one, or it silently regains the diversion it just lost.
    #[test]
    fn a_measured_payoff_is_never_mistaken_for_unmeasured() {
        assert_ne!(blend_spec_payoff(0, 0.0), 0);
        assert_ne!(blend_spec_payoff(1, 0.0), 0);
    }
}

#[cfg(test)]
mod lost_conversation_tests {
    use super::{forward_lacks_its_conversation, lost_conversation_error, wire_token_ids};
    use crate::error::{reclassify_flattened_error, SwarmError};

    /// The rule that stops a reply being decoded from nothing
    /// (`docs/FUTURE_WORK.md` #93): a forward that continues a conversation
    /// runs only where that conversation is.
    #[test]
    fn a_forward_past_the_prompt_is_refused_where_its_conversation_is_gone() {
        // The case that produced 96 tokens of garbage: a decode step reaching a
        // worker restarted since the prompt pass.
        assert!(forward_lacks_its_conversation(5, 41, false));
        // The same step where the cache is intact runs as always.
        assert!(!forward_lacks_its_conversation(5, 41, true));
        // A prompt pass starts the conversation, so it never needs one — and
        // the worker clears whatever it holds for that id right after.
        assert!(!forward_lacks_its_conversation(0, 0, false));
        // A replay to a stand-in is one forward at position 0 by design
        // (`pipeline::distributed::assemble_replay`): it rebuilds the cache.
        assert!(!forward_lacks_its_conversation(3, 0, false));
    }

    #[test]
    fn a_forward_naming_a_token_the_table_has_no_row_for_is_refused() {
        let bytes = |ids: &[i64]| {
            ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        // Every row of a 100-row table is an id it can look up.
        assert_eq!(
            wire_token_ids(&bytes(&[0, 5, 99]), Some(100), "t").unwrap(),
            vec![0, 5, 99]
        );
        // One past the last row, a negative id, and one far past are refused
        // before anything reaches the gather.
        for bad in [100, -1, i64::MAX] {
            let err = wire_token_ids(&bytes(&[5, bad]), Some(100), "t").unwrap_err();
            assert!(
                err.to_string()
                    .contains("outside the model's 100 embedding rows"),
                "{err}"
            );
        }
        // A payload that is not whole ids, and a segment with no table.
        assert!(wire_token_ids(&[], Some(100), "t").is_err());
        assert!(wire_token_ids(&[0; 7], Some(100), "t").is_err());
        assert!(wire_token_ids(&bytes(&[1]), None, "t").is_err());
    }

    /// The refusal crosses the worker IPC hop and the network hop as TEXT, and
    /// is re-typed from its prefix on the far side (`reclassify_flattened_error`).
    /// It must come back as `ServiceUnavailable`: that is what lets the router
    /// re-plan the request, where `Inference` would report a crash.
    #[test]
    fn the_refusal_keeps_its_type_across_the_process_boundary() {
        let err = lost_conversation_error(uuid::Uuid::nil(), (14, 40), 41);
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));
        assert!(
            matches!(
                reclassify_flattened_error(&err.to_string()),
                Some(SwarmError::ServiceUnavailable(_))
            ),
            "flattened to text, it must still read as ServiceUnavailable: {err}"
        );
    }
}

#[cfg(test)]
mod argmax_walk_tests {
    use super::argmax_walk;

    /// The card's walk from per-row argmaxes keeps what the processor's greedy
    /// walk keeps, for every way a round can end: all kept, a mismatch at each
    /// position, no drafts at all.
    #[test]
    fn the_argmax_walk_is_the_greedy_walk() {
        let greedy = crate::types::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let rows: Vec<Vec<f32>> = vec![
            vec![0.1, 2.0, -1.0, 0.5],
            vec![3.0, 0.0, 0.2, 0.1],
            vec![-2.0, -1.0, 0.0, 4.0],
            vec![0.0, 0.3, 1.5, 0.2],
        ];
        let picks: Vec<u32> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                        if x > b.1 {
                            (i, x)
                        } else {
                            b
                        }
                    })
                    .0 as u32
            })
            .collect();
        for drafts in [
            vec![1u32, 0, 3],
            vec![2, 0, 3],
            vec![1, 1, 3],
            vec![1, 0, 0],
        ] {
            let host = crate::inference::sampling::walk_verified_positions(
                &drafts,
                &rows,
                &greedy,
                &[],
                None,
            )
            .unwrap();
            assert_eq!(argmax_walk(&drafts, &picks), host, "drafts {drafts:?}");
        }
        let one = crate::inference::sampling::walk_verified_positions(
            &[],
            &rows[..1],
            &greedy,
            &[],
            None,
        )
        .unwrap();
        assert_eq!(argmax_walk(&[], &picks[..1]), one);
    }
}

#[cfg(test)]
mod cancel_release_tests {
    use super::release_caches_of_cancelled;
    use crate::inference::split::KvCacheStore;
    use candle_core::{Device, Tensor};
    use std::sync::Arc;

    fn conversation(store: &KvCacheStore, request: &uuid::Uuid) {
        let mut entry = store.get_or_create("m", &request.to_string(), 1);
        let mut kv = crate::inference::split::kv_cache::LayerKv::with_capacity(2, 64, 64);
        let k = Tensor::zeros((1usize, 2, 4, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        kv.append(&k, &k.clone()).unwrap();
        entry.layers[0] = Some(kv);
    }

    /// A cancel is also how a node serving a segment learns the request ended
    /// (#238): the cancelled request's cache goes at once, another request's
    /// stays, and the release runs once per cancel however long the mark is
    /// kept to skip a forward still queued.
    #[test]
    fn a_cancelled_requests_cache_is_released_once() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let (ended, live) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        conversation(&store, &ended);
        conversation(&store, &live);
        let cancelled: super::CancelledSet = Arc::new(dashmap::DashMap::new());
        cancelled.insert(ended, std::time::Instant::now());
        let mut released = std::collections::HashSet::new();

        release_caches_of_cancelled(&cancelled, &mut released, &store);
        assert!(!store.request_holds_state("m", &ended.to_string()));
        assert!(store.request_holds_state("m", &live.to_string()));
        assert!(
            cancelled.contains_key(&ended),
            "the mark stays, to skip a queued forward"
        );

        // Once per cancel: a cache the same id builds afterwards is not
        // touched again by the same mark.
        conversation(&store, &ended);
        release_caches_of_cancelled(&cancelled, &mut released, &store);
        assert!(store.request_holds_state("m", &ended.to_string()));
    }
}

#[cfg(test)]
mod forward_width_tests {
    use super::*;
    use crate::inference::split::kv_budget::KvGrowth;
    use crate::inference::split::tests::gemma2::windowed_model_with_budget;

    /// **A coordinator's long forward is read in this worker's own chunks**,
    /// so a windowed layer is admitted at its window and one chunk, never the
    /// whole forward (#281 (4)): a peer handed a 4,096-position prompt in one
    /// forward was admitted as if its windowed layers kept every position.
    /// An image's prompt and a tensor-parallel phase run in ONE pass, and are
    /// charged their whole width.
    #[test]
    fn a_long_forward_read_in_chunks_is_admitted_at_its_window() {
        let prompt = 4096;
        assert_eq!(width_handed_to_the_caches(false, false, false, prompt), 0);
        assert_eq!(width_handed_to_the_caches(false, true, true, prompt), 0);
        assert_eq!(
            width_handed_to_the_caches(false, false, true, prompt),
            prompt
        );
        assert_eq!(
            width_handed_to_the_caches(true, false, false, prompt),
            prompt
        );

        let admitted = |width: usize| {
            let quantum = crate::inference::layers::kv_growth_quantum(8192);
            // Room for the global layer's every position and the windowed
            // one's window and a chunk — not for both keeping every position.
            let model = windowed_model_with_budget(
                8192,
                &[Some(64), None],
                KvGrowth::with_window(1000, 1000, 64, quantum),
                7_000_000,
            );
            let kv = KvCacheStore::new(std::time::Duration::from_secs(600));
            let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
            ensure_room_for_prompt(&model, &kv, &pc, "r", prompt, 0, width).is_ok()
        };
        let read_in_chunks = width_handed_to_the_caches(false, false, false, prompt);
        assert!(
            admitted(read_in_chunks),
            "a text prompt read in chunks fits"
        );
        assert!(
            !admitted(prompt),
            "control: charged the whole forward, the same prompt does not"
        );
    }
}

#[cfg(test)]
mod up_front_admission_tests {
    use super::*;
    use crate::inference::split::kv_budget::KvGrowth;
    use crate::inference::split::kv_cache::LayerKv;
    use crate::inference::split::tests::gemma2::windowed_model_with_budget;
    use crate::inference::split::SnapshotRoom;
    use candle_core::{DType, Device, Tensor};

    /// A saved split prompt of `positions` positions, 1 KiB each, kept beside
    /// the live cache — the opening a pass resumes from next turn.
    fn saved_prompt(kv: &KvCacheStore, pc: &PrefixCache, positions: usize) -> usize {
        let mut entry = kv.get_or_create("m", "earlier", 2);
        for slot in entry.layers.iter_mut() {
            let k = Tensor::zeros((1usize, 1, positions, 64), DType::F32, &Device::Cpu).unwrap();
            let mut layer = LayerKv::with_capacity(2, 4096, 4096);
            layer.append(&k, &k).unwrap();
            *slot = Some(layer);
        }
        drop(entry);
        let room = SnapshotRoom {
            home: SnapshotHome::BesideLive,
            max_positions: usize::MAX,
        };
        let keys = vec![[7u8; 32]; positions / 8];
        assert_eq!(
            pc.insert_chain_from_kv("m", "earlier", kv, 8, &keys, room) as usize,
            positions / 8
        );
        kv.cleanup_request_id("earlier");
        pc.bytes_in(SnapshotHome::BesideLive)
    }

    /// **Up front, a prompt that fits only once saved prompts go evicts none
    /// of them** (field report 2026-10-19): the eviction took the opening the
    /// pass was about to resume from — not looked up yet, so the least
    /// recently used — and the miss sent a 3,718-token prompt to every segment
    /// again, 103 s on a peer abroad. Each forward then admits itself after
    /// its lookup. Neither a refusal nor a fit changes.
    #[test]
    fn an_up_front_admission_leaves_saved_prompts_to_the_forwards() {
        let prompt = 100;
        let per_position = 1000;
        let claim = per_position
            * crate::inference::layers::kv_cache_reservation(prompt + REPLY_RESERVE_POSITIONS)
                as u64;
        let admit = |budget: u64, admitting: Admitting| {
            let model = windowed_model_with_budget(
                8192,
                &[None, None],
                KvGrowth::flat(per_position),
                budget,
            );
            let kv = KvCacheStore::new(std::time::Duration::from_secs(600));
            let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
            let saved = saved_prompt(&kv, &pc, 128);
            assert!(saved > 0, "fixture: a saved prompt beside the live cache");
            let width = model.layer_end.saturating_sub(model.layer_start) as u32;
            let outcome = admit_prompt_over(
                &[(&model, width)],
                KvGrowth::default(),
                &kv,
                &pc,
                "r",
                prompt,
                REPLY_RESERVE_POSITIONS,
                0,
                admitting,
            )
            .map_err(|refused| refused.short_by);
            (outcome, pc.bytes_in(SnapshotHome::BesideLive) == saved)
        };
        // Room for this prompt's cache, but not beside the saved one.
        let tight = claim + 1024;
        let (outcome, kept) = admit(tight, Admitting::UpFront);
        assert_eq!(outcome, Ok(Admitted::LeftToTheForwards));
        assert!(
            kept,
            "up front, the saved prompt stays for the forward to restore"
        );
        let (outcome, kept) = admit(tight, Admitting::Record);
        assert_eq!(outcome, Ok(Admitted::Recorded));
        assert!(
            !kept,
            "control: a forward's own admission makes the room, as it always has"
        );
        // Room for both: promised up front, nothing evicted.
        let (outcome, kept) = admit(claim * 4, Admitting::UpFront);
        assert_eq!(outcome, Ok(Admitted::Recorded));
        assert!(kept);
        // No room even with the saved prompt gone: refused up front, as before.
        let (outcome, kept) = admit(claim / 2, Admitting::UpFront);
        assert!(
            outcome.is_err(),
            "a prompt that cannot fit is refused up front"
        );
        assert!(kept, "and its refusal evicted nothing");
    }

    /// **A card with no room for a prompt's copy beside its live cache keeps
    /// the copy in host RAM, not a cut of it** (field report 2026-10-19: ~190 MB
    /// left for caches on a 12 GB card holding two segments of a 30B, and the
    /// next turn read the whole prompt again on every segment). Host memory is
    /// asked only when the card falls short, and never under
    /// `SWARMLLM_PREFIX_CACHE_HOME=card`.
    #[test]
    fn a_snapshot_the_card_cannot_hold_whole_goes_to_host_memory() {
        let asked = std::cell::Cell::new(false);
        let host = |positions: usize| {
            let asked = &asked;
            move || {
                asked.set(true);
                positions
            }
        };
        let room = room_off_the_card_if_more(3718, 3718, true, host(9000));
        assert_eq!(room.home, SnapshotHome::BesideLive);
        assert_eq!(room.max_positions, 3718);
        assert!(
            !asked.get(),
            "a card holding the whole copy never asks host memory"
        );

        let room = room_off_the_card_if_more(640, 3718, true, host(9000));
        assert_eq!(room.home, SnapshotHome::HostRam);
        assert_eq!(room.max_positions, 9000);

        let room = room_off_the_card_if_more(640, 3718, true, host(100));
        assert_eq!(
            (room.home, room.max_positions),
            (SnapshotHome::BesideLive, 640),
            "host memory holding less than the card keeps the card's cut"
        );
        asked.set(false);
        let room = room_off_the_card_if_more(640, 3718, false, host(9000));
        assert_eq!(
            (room.home, room.max_positions),
            (SnapshotHome::BesideLive, 640)
        );
        assert!(
            !asked.get(),
            "a copy that may not leave the card never asks"
        );
    }
}

#[cfg(test)]
mod prompt_outlives_tests {
    use super::prompt_outlives;
    use crate::inference::split::kv_cache::CANCELLED_MID_FORWARD;

    /// A prompt snapshot left for the reply's end is taken however the reply
    /// ends while the worker survives — above all a cancel landing inside a
    /// step, the user who stops a reply and asks again — and never after an
    /// error that ruined the device. Copied before the first token, as it was
    /// until 2026-10-28, the snapshot outlived every later failure; moved to
    /// the reply's end, it was taken only by a reply that finished cleanly.
    #[test]
    fn a_prompt_left_for_the_replys_end_outlives_every_error_the_worker_survives() {
        assert!(prompt_outlives(None));
        let cancelled =
            crate::error::SwarmError::Inference(CANCELLED_MID_FORWARD.to_string()).to_string();
        assert!(prompt_outlives(Some(&cancelled)), "{cancelled}");
        assert!(prompt_outlives(Some(
            "send Token: Broken pipe (os error 32)"
        )));
        assert!(!prompt_outlives(Some(
            "Forward: Cuda(DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\"))"
        )));
        assert!(!prompt_outlives(Some(
            "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))"
        )));
    }
}
