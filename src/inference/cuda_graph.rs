//! A decoded token's forward pass sent to the graphics card as ONE CUDA graph
//! (`docs/plans/local_decode_submissions.md` Stage 4).
//!
//! **Why.** Local decode is bound by submission COUNT, not arithmetic: several
//! hundred driver calls a token — kernel launches, plus candle's per-op
//! allocations and frees — each a trip through WSL2's GPU channel on the one
//! thread that drives the card. `examples/cuda_graph_cost_probe.cu` put 1,000
//! such operations at 10.4 ms submitted one at a time, against 4.9 ms captured
//! into a graph, updated in place and launched once — measured on the dev
//! machine, where the open question was whether RECORDING a launch costs as
//! much as making one. It does not.
//!
//! **How: llama.cpp's way.** Re-capture the forward every token, update the
//! instantiated graph with `cuGraphExecUpdate`, launch it. Every token changes
//! kernel arguments (the KV length, the RoPE row, the append offset) and the
//! attention scores' sizes; the update accepts both — 199/199 updates with
//! every tenth allocation growing each token and its launch grid with it (the
//! same probe, extended 2026-09-29). So no length classes (HuggingFace's
//! grout) and no stable-buffer arena: candle's per-op allocations become the
//! graph's own memory nodes.
//!
//! **What makes a capture safe, and where each condition is held:**
//! 1. The device has its own stream — CUDA refuses to capture the legacy one
//!    (`SWARMLLM_CUDA_OWN_STREAM=1`), and with event tracking off, so no
//!    allocation records an event mid-capture. [`DecodeGraphSlot::decide`].
//! 2. No host→device copy inside. CUDA ACCEPTS a copy from pageable host
//!    memory in a capture and replays it from that host ADDRESS at every
//!    launch — measured: a value changed between capture and launch is the one
//!    that arrives (`examples/cuda_graph_capture_rules.cu`). Candle copies
//!    from temporaries freed long before the launch, so that is silent garbage,
//!    not an error. The count of copies is read either side of every capture,
//!    and a capture that made one is thrown away (`capture`).
//! 3. Nothing allocated outside is freed inside, and nothing allocated inside
//!    outlives it. The caller (`SplitModel::forward_decode_as_graph`) captures
//!    only a step AFTER an ordinary step of the same conversation (at any
//!    position since 2026-09-30), whose KV buffers already hold the new position
//!    (`KvCacheStore::every_cache_holds`), and copies the result into a tensor
//!    made BEFORE the capture.
//! 4. No synchronisation inside — CUDA refuses that loudly, which ends the
//!    capture and falls to the next rule.
//!
//! **A refused capture never fails the request.** The caller puts the KV cache
//! back where it was and runs the step the ordinary way — the same kernels in
//! the same order, so the same answer.
//!
//! **A graph pays only when it is UPDATED.** A shape the driver keeps refusing
//! to update is rebuilt every launch at 10-100 ms, against the ~3 ms a graph
//! saves; after [`REBUILDS_BEFORE_RESTING`] among its last [`CHURN_WINDOW`]
//! launches it rests — runs the ordinary way — for [`REST_STEPS`], then is
//! tried again. The first refusal of each kind is logged with the driver's
//! reason.
//!
//! ON by default since 2026-09-30, after the gate ran on a `--features cuda`
//! build (v0.3.199 shipped a stream change that every test and a cheaper build
//! passed while every reply was garbage, gotcha #683): byte-identical replies on
//! five models, a 7B split across two card nodes, `failover_mid`, 700-token
//! replies across a KV growth step, split speculation with its drafter.
//! `SWARMLLM_CUDA_GRAPH=0` turns it off; `SWARMLLM_CUDA_OWN_STREAM=0` does too,
//! since the legacy stream cannot be captured.

use crate::error::SwarmError;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// On unless `SWARMLLM_CUDA_GRAPH=0`. Read once: the answer must not change
/// between two models in one worker.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CUDA_GRAPH").as_deref() != Ok("0"))
}

/// The most positions one captured forward may carry: a decode step (1) or a
/// speculative check (the token plus its guesses). Up to 8 rows the quantized
/// matmuls stay on the vector kernel; past it they switch to the prompt path.
pub(crate) const MAX_POSITIONS: usize = 8;

/// How many conversations' last decode step a model remembers. A worker
/// serves a handful at once; past this the map is simply cleared, which costs
/// each live conversation one uncaptured step.
const LAST_STEP_CAPACITY: usize = 64;

/// How often a model that is capturing says how it is going.
const REPORT_EVERY: Duration = Duration::from_secs(60);

/// Refusals that point at the PROGRAM — a copy, a failed call, a broken
/// capture — after which a model stops capturing. They recur on every token
/// once they happen at all, and one of them costs more than a refusal: a free
/// of memory made before the capture fails inside it and that buffer is never
/// freed (`docs/plans/local_decode_submissions.md` § Stage 4b item 1), so a
/// systematic one would leak a buffer a token. A forward that fails on its own
/// (a cancelled request, the end of the context) is not counted.
const DEFECTS_BEFORE_GIVING_UP: u32 = 3;

/// Launches, among the last [`CHURN_WINDOW`] of one number of positions, whose
/// graph the driver would not update in place and had to be built again. At
/// this many, forwards of that many positions REST — run the ordinary way — for
/// [`REST_STEPS`]. A window rather than a run in a row: a shape rebuilt on half
/// its launches loses as much as one rebuilt on all of them, and clean launches
/// in between would keep breaking a run.
///
/// A graph pays only when it is updated: an update is ~1 ms of recording on a
/// 7B step that saves ~3 ms of submissions, while a rebuild
/// (`cuGraphInstantiate`) costs 10-100 ms. Measured 2026-10-04/05: the 7B's
/// speculative checks on the .225 gate's split rig rebuilt 105 of 135 launches
/// at 65-108 ms of recording each (the drafter beside it: 1 of 404, ~6 ms), and
/// a Qwen2.5-14B segment served for peers rebuilt 177 of 183 at 12.7 ms. A shape
/// that keeps changing is never worth capturing — llama.cpp's own rule
/// (`ggml_backend_cuda_graph_compute`, llama-cpp-sys-2 0.1.156) runs a graph
/// whose node properties changed UNcaptured until two calls in a row agree.
const REBUILDS_BEFORE_RESTING: u32 = 3;

/// How many recent launches of one number of positions [`REBUILDS_BEFORE_RESTING`]
/// counts over. A rebuild now and then — one in 64 when a step grows its KV
/// chunk count — never reaches it.
const CHURN_WINDOW: u32 = 8;

/// Clean launches in a row after which a number of positions' rests start
/// again from [`REST_STEPS`]: a shape that has settled is not punished for
/// churn it showed hours ago.
const CLEAN_LAUNCHES_TO_FORGET: u32 = 64;

/// Steps of one number of positions run the ordinary way once it rests, before
/// capture is tried again — doubled each time it comes back to rest, up to
/// 16× (forgotten after [`CLEAN_LAUNCHES_TO_FORGET`]). Bounds what a shape that
/// always churns costs (three rebuilds per rest)
/// while a shape that churned for a moment — a burst of 50 rebuilds in one
/// minute among 980 updates, live node 2026-10-01 — gets its graphs back.
const REST_STEPS: u32 = 256;

/// The refusal that is the forward's own failure, not the capture's.
pub(crate) const FORWARD_FAILED: &str = "the forward failed inside the capture";

/// The error a forward returns from [`Cutter::cut`] when a group was refused —
/// it unwinds the forward; [`DecodeGraph::capture`] reports the refusal itself.
#[cfg(feature = "candle-cuda")]
pub(crate) const GROUP_REFUSED: &str = "a CUDA graph group was refused";

/// Layers per graph group: the card waits only for the FIRST group to be
/// recorded, then runs each group while the next is recorded. Recording a 7B
/// step whole took 4.5 ms of a 20.8 ms token, all of it with the card idle.
/// Two was best or tied on every model measured (RTX 3070 Laptop, 2026-09-30,
/// tok/s by layers per group 1 / 2 / 4 / whole step): TinyLlama 226 / 228 /
/// 220, Llama 3.2 3B 105 / 106 / 103, Qwen2.5-Coder-7B 56.6 / 56.7 / 56.1 /
/// 47.3, Llama 3.1 8B 55.0 / 54.9 / 54.7. `SWARMLLM_CUDA_GRAPH_GROUP=N` sets
/// it; `0` records the step as one graph.
pub(crate) fn group_layers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var("SWARMLLM_CUDA_GRAPH_GROUP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(0) => usize::MAX,
            Some(n) => n,
            None => 2,
        }
    })
}

/// Handed to a forward being captured: [`Cutter::cut`] closes the group
/// recorded so far, launches it, and begins the next.
pub(crate) struct Cutter<'a> {
    #[cfg(feature = "candle-cuda")]
    session: cuda::Session<'a>,
    #[cfg(not(feature = "candle-cuda"))]
    _never: std::marker::PhantomData<&'a ()>,
}

impl Cutter<'_> {
    /// Launch what has been recorded and begin recording the rest. Everything
    /// the rest reads that the launched part made must already have been
    /// copied into memory made OUTSIDE the capture ([`DecodeGraph::boundaries`]).
    /// An `Err` (`GROUP_REFUSED`) means stop the forward.
    pub(crate) fn cut(&mut self) -> Result<(), SwarmError> {
        #[cfg(feature = "candle-cuda")]
        {
            self.session.cut()
        }
        #[cfg(not(feature = "candle-cuda"))]
        {
            Ok(())
        }
    }
}

/// Whether a model's decode steps are captured — decided at its first decode
/// step, once, because nothing it depends on changes while a model is loaded.
#[derive(Default)]
pub(crate) enum DecodeGraphSlot {
    #[default]
    Unchecked,
    Off,
    On(Box<DecodeGraph>),
}

impl DecodeGraphSlot {
    /// `model_says_no` is the model's own reason, if it has one (a layer type
    /// the capture has not been checked on, a card/processor split).
    pub(crate) fn decide(device: &Device, model_says_no: Option<&'static str>) -> Self {
        if !requested() {
            return Self::Off;
        }
        match model_says_no.or_else(|| device_says_no(device)) {
            Some(reason) => {
                tracing::info!(reason, "DIAG: decode steps on this model stay uncaptured");
                Self::Off
            }
            None => {
                tracing::info!(
                    off_switch = "SWARMLLM_CUDA_GRAPH=0",
                    "DIAG: decode steps on this model go to the card as one CUDA graph"
                );
                Self::On(Box::default())
            }
        }
    }
}

/// Hand the card memory graphs hold between steps back to the driver, once a
/// worker has been idle (`model_worker::hand_back_idle_card_memory`, beside the
/// memory pool's own trim). Graph allocations come from a pool of their own that
/// `cuda_pool::trim` never touches; each group's graph keeps its step's
/// temporaries mapped so the next launch need not map them again. Returns the
/// megabytes held before and after, or `None` off a card / when the driver
/// will not say. The caller has synchronised, so no graph is running.
pub(crate) fn trim_idle_graph_memory(device: &Device) -> Option<(u64, u64)> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::trim_graph_memory(dev);
    }
    let _ = device;
    None
}

fn device_says_no(device: &Device) -> Option<&'static str> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::device_says_no(dev);
    }
    let _ = device;
    Some("not on a graphics card")
}

/// Why a capture was thrown away. `kind` is fixed so the first of each is
/// logged once at info; `detail` goes to debug every time.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub(crate) kind: &'static str,
    pub(crate) detail: String,
}

/// What a launched capture did: whether every group's graph was updated in
/// place, and — for each existing graph the driver would NOT update — why: its
/// result code and the type of the node it named (`cuGraphExecUpdate`'s
/// `resultInfo`), as one line. A group with no graph yet is built without a
/// refusal. Logged once per kind, so the next churn names its cause.
type Launched = (bool, Vec<String>);

/// One number of positions' run of rebuilds, and its rest.
#[derive(Default)]
struct Churn {
    /// The last [`CHURN_WINDOW`] launches, newest in the lowest bit: 1 where at
    /// least one group's update was refused.
    recent: u8,
    /// Launches in a row with no refused update.
    clean_in_a_row: u32,
    /// Steps still to run the ordinary way before capture is tried again.
    rest_left: u32,
    /// How many rests it has taken — doubles the next.
    rests: u32,
}

#[derive(Default)]
struct Stats {
    launched: u64,
    updated: u64,
    instantiated: u64,
    uncaptured: u64,
    refused: u64,
    /// Launches where the driver refused to update an existing graph.
    rebuilt: u64,
    /// Rests begun ([`REBUILDS_BEFORE_RESTING`]).
    rested: u64,
    update_refusals_seen: Vec<String>,
    /// Host time from the start of a capture to its launch, summed over the
    /// launched ones. The card waits for all of it — a graph starts nothing
    /// until the step is recorded — so it is the cost to set against the
    /// submissions the graph saves.
    recording: Duration,
    reasons_seen: Vec<&'static str>,
    reported_at: Option<Instant>,
}

/// One model's decode graph: the instantiated graph every capture updates,
/// and what the caller needs to decide whether a step may be captured.
#[derive(Default)]
pub(crate) struct DecodeGraph {
    /// One instantiated graph per layer GROUP ([`group_layers`]), per number
    /// of positions: the same group covers the same layers every token, so
    /// each updates in place. Keyed by positions because a speculative check
    /// of 3 runs other kernels than a step of 1 — one shared graph would be
    /// rebuilt every time a round alternated between them.
    #[cfg(feature = "candle-cuda")]
    execs: HashMap<usize, Vec<Option<cuda::Exec>>>,
    /// Where the residual stream crosses from one group's graph to the next,
    /// per number of positions: made OUTSIDE every capture and reused every
    /// token (two, alternating, so a group never writes the buffer its own
    /// first layer read). Only clones are handed out, so no drop inside a
    /// capture frees them.
    boundaries: HashMap<usize, [Tensor; 2]>,
    /// Each conversation's last decode position. A capture needs the step
    /// before it to have run the ordinary way: that step allocated whatever a
    /// conversation's first decode step allocates, loaded every kernel module
    /// the step uses, and left the output shape in `templates`.
    last_step: HashMap<String, usize>,
    /// The output's shape and type, by (`all_positions`, positions) — the
    /// tensor a capture copies its result into has to exist before the capture
    /// starts.
    templates: HashMap<(bool, usize), (Vec<usize>, DType)>,
    /// Refusals that were the capture's fault, per number of positions — see
    /// [`DEFECTS_BEFORE_GIVING_UP`]. Per positions because a defect of the
    /// several-position check (22 host copies, 2026-09-30) said nothing about
    /// the one-position step, and giving the whole model up threw away the
    /// decode graphs with it.
    defects: HashMap<usize, u32>,
    /// Rebuilds and rests, per number of positions — see
    /// [`REBUILDS_BEFORE_RESTING`]. Per positions for the same reason as
    /// `defects`: the checks churned while the one-position step updated.
    churn: HashMap<usize, Churn>,
    stats: Stats,
}

impl DecodeGraph {
    /// Record a decode step at `index_pos`; true when this conversation has
    /// already taken a decode step here. That step is the one that allocated
    /// what a conversation's first step allocates, shed a hydrated snapshot's
    /// mirror, and loaded every kernel module a step uses; a later step at ANY
    /// position — after a rejected guess cut the cache back, or after a
    /// several-position pass moved it on — has none of that left to do, and
    /// the caller still checks every KV buffer holds the new position. Requiring
    /// the very next position turned every speculation round's first step into
    /// an uncaptured one (2026-09-30).
    pub(crate) fn follows_previous_step(&mut self, request_id: &str, index_pos: usize) -> bool {
        if let Some(last) = self.last_step.get_mut(request_id) {
            *last = index_pos;
            return true;
        }
        if self.last_step.len() >= LAST_STEP_CAPACITY {
            self.last_step.clear();
        }
        self.last_step.insert(request_id.to_string(), index_pos);
        false
    }

    /// True when forwards of `positions` are not to be captured now: they were
    /// refused often enough, for reasons of the program's making, to be given
    /// up for good — or their graphs kept having to be rebuilt and they are
    /// resting ([`REBUILDS_BEFORE_RESTING`]).
    pub(crate) fn declines(&self, positions: usize) -> bool {
        self.defects.get(&positions).copied().unwrap_or(0) >= DEFECTS_BEFORE_GIVING_UP
            || self.churn.get(&positions).is_some_and(|c| c.rest_left > 0)
    }

    /// The two boundary buffers for a `[1, positions, hidden]` residual
    /// stream, made on first use — which must be OUTSIDE a capture, so the
    /// caller asks before it begins one.
    pub(crate) fn boundaries(
        &mut self,
        device: &Device,
        positions: usize,
        hidden: usize,
    ) -> Result<[Tensor; 2], SwarmError> {
        let make = || {
            Tensor::zeros((1, positions, hidden), DType::F32, device).map_err(SwarmError::internal)
        };
        match self.boundaries.entry(positions) {
            std::collections::hash_map::Entry::Occupied(e) => Ok(e.get().clone()),
            std::collections::hash_map::Entry::Vacant(e) => {
                Ok(e.insert([make()?, make()?]).clone())
            }
        }
    }

    pub(crate) fn template(
        &self,
        all_positions: bool,
        positions: usize,
    ) -> Option<&(Vec<usize>, DType)> {
        self.templates.get(&(all_positions, positions))
    }

    /// A step that ran the ordinary way: remember its output's shape, and count
    /// it off a rest.
    pub(crate) fn note_uncaptured(&mut self, all_positions: bool, positions: usize, out: &Tensor) {
        self.templates.insert(
            (all_positions, positions),
            (out.dims().to_vec(), out.dtype()),
        );
        if let Some(churn) = self.churn.get_mut(&positions) {
            churn.rest_left = churn.rest_left.saturating_sub(1);
        }
        self.stats.uncaptured += 1;
        self.report();
    }

    /// A launched capture: count its rebuilds, and put its number of positions
    /// to rest once the driver has refused to update it
    /// [`REBUILDS_BEFORE_RESTING`] times among its last [`CHURN_WINDOW`] launches.
    fn note_launched(&mut self, positions: usize, refused_updates: &[String]) {
        let churn = self.churn.entry(positions).or_default();
        let rebuilt = !refused_updates.is_empty();
        let window = (1u16 << CHURN_WINDOW) - 1;
        churn.recent = (((u16::from(churn.recent) << 1) | u16::from(rebuilt)) & window) as u8;
        if !rebuilt {
            churn.clean_in_a_row += 1;
            if churn.clean_in_a_row >= CLEAN_LAUNCHES_TO_FORGET {
                churn.rests = 0;
            }
            return;
        }
        churn.clean_in_a_row = 0;
        self.stats.rebuilt += 1;
        for refused in refused_updates {
            if !self.stats.update_refusals_seen.contains(refused) {
                tracing::info!(
                    positions,
                    why = %refused,
                    "DIAG: decode graph could not be updated in place — rebuilt (first of this kind)"
                );
                self.stats.update_refusals_seen.push(refused.clone());
            }
        }
        if churn.recent.count_ones() < REBUILDS_BEFORE_RESTING {
            return;
        }
        churn.recent = 0;
        churn.rest_left = REST_STEPS << churn.rests.min(4);
        churn.rests += 1;
        self.stats.rested += 1;
        if churn.rests == 1 {
            tracing::info!(
                positions,
                rest_steps = churn.rest_left,
                "DIAG: decode graph: forwards of this many positions keep being rebuilt — they \
                 run the ordinary way for a while, then capture is tried again"
            );
        } else {
            tracing::debug!(
                positions,
                rest_steps = churn.rest_left,
                rests = churn.rests,
                "DIAG: decode graph: still rebuilt every launch — resting again"
            );
        }
    }

    /// Capture `forward` on `device`'s stream and launch it — in GROUPS: each
    /// [`Cutter::cut`] the forward makes ends the group recorded so far,
    /// updates (or builds) that group's graph and launches it, and begins the
    /// next, so the card runs one group while the next is recorded. A forward
    /// that never cuts is one graph. On `Ok` the step's work is queued on the
    /// stream exactly as if issued op by op. On `Err`, groups launched before
    /// the refusal HAVE run (their layers wrote this position's KV), nothing
    /// after it will, and the caller must undo what `forward` did on the host
    /// and run the step the ordinary way — which rewrites the same positions.
    pub(crate) fn capture(
        &mut self,
        device: &Device,
        positions: usize,
        forward: impl FnOnce(&mut Cutter<'_>) -> Result<(), SwarmError>,
    ) -> Result<(), Refusal> {
        let started = Instant::now();
        #[cfg(feature = "candle-cuda")]
        let outcome = match device {
            Device::Cuda(dev) => {
                cuda::capture_in_groups(dev, self.execs.entry(positions).or_default(), forward)
            }
            _ => Err(Refusal {
                kind: "not on a graphics card",
                detail: String::new(),
            }),
        };
        #[cfg(not(feature = "candle-cuda"))]
        let outcome: Result<Launched, Refusal> = {
            let _ = (
                device,
                positions,
                forward,
                Cutter {
                    _never: std::marker::PhantomData,
                },
            );
            Err(Refusal {
                kind: "built without CUDA",
                detail: String::new(),
            })
        };
        match &outcome {
            Ok((all_updated, refused_updates)) => {
                self.stats.launched += 1;
                self.stats.recording += started.elapsed();
                if *all_updated {
                    self.stats.updated += 1;
                } else {
                    self.stats.instantiated += 1;
                }
                self.note_launched(positions, refused_updates);
            }
            Err(refusal) => {
                self.stats.refused += 1;
                if refusal.kind != FORWARD_FAILED {
                    let defects = self.defects.entry(positions).or_insert(0);
                    *defects += 1;
                    if *defects == DEFECTS_BEFORE_GIVING_UP {
                        tracing::warn!(
                            positions,
                            refused = self.stats.refused,
                            kinds = ?self.stats.reasons_seen,
                            last = refusal.kind,
                            "decode graph: this model's forwards of this many positions kept \
                             being refused — they run the ordinary way from now on"
                        );
                    }
                }
                if self.stats.reasons_seen.contains(&refusal.kind) {
                    tracing::debug!(kind = refusal.kind, detail = %refusal.detail,
                        "DIAG: decode graph capture refused — step ran uncaptured");
                } else {
                    self.stats.reasons_seen.push(refusal.kind);
                    tracing::info!(kind = refusal.kind, detail = %refusal.detail,
                        "DIAG: decode graph capture refused — step ran uncaptured (first of this kind)");
                }
            }
        }
        self.report();
        outcome.map(|_| ())
    }

    fn report(&mut self) {
        let now = Instant::now();
        match self.stats.reported_at {
            Some(at) if now.duration_since(at) < REPORT_EVERY => return,
            Some(_) => {}
            // The first report waits a full period, so a short chat says nothing.
            None => {
                self.stats.reported_at = Some(now);
                return;
            }
        }
        self.stats.reported_at = Some(now);
        let s = &self.stats;
        tracing::info!(
            launched = s.launched,
            recording_ms_per_launch = format!(
                "{:.2}",
                s.recording.as_secs_f64() * 1e3 / s.launched.max(1) as f64
            ),
            updated_in_place = s.updated,
            instantiated = s.instantiated,
            rebuilt = s.rebuilt,
            rested = s.rested,
            uncaptured = s.uncaptured,
            refused = s.refused,
            "DIAG: decode graph"
        );
    }
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use super::Refusal;
    use crate::error::SwarmError;
    use candle_core::cuda::htod_copies_so_far;
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    /// An instantiated graph, destroyed once.
    pub(super) struct Exec(sys::CUgraphExec);

    // SAFETY: a graph exec may be used from any thread with the context bound,
    // but not from two at once. It lives inside a `SplitModel`, which is only
    // ever driven through `&mut self`, one forward at a time.
    unsafe impl Send for Exec {}

    impl Drop for Exec {
        fn drop(&mut self) {
            // SAFETY: made by `cuGraphInstantiateWithFlags`, destroyed only here.
            let _ = unsafe { result::graph::exec_destroy(self.0) };
        }
    }

    /// A captured graph, destroyed once it has been instantiated or refused.
    struct Graph(sys::CUgraph);

    impl Drop for Graph {
        fn drop(&mut self) {
            // SAFETY: returned non-null by `cuStreamEndCapture`, destroyed only here.
            let _ = unsafe { result::graph::destroy(self.0) };
        }
    }

    fn graph_reserved(device: sys::CUdevice) -> Option<u64> {
        let mut value: u64 = 0;
        // SAFETY: RESERVED_MEM_CURRENT is a `cuuint64_t`.
        unsafe {
            sys::cuDeviceGetGraphMemAttribute(
                device,
                sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
                (&mut value as *mut u64).cast::<core::ffi::c_void>(),
            )
        }
        .result()
        .ok()
        .map(|()| value)
    }

    pub(super) fn trim_graph_memory(dev: &candle_core::CudaDevice) -> Option<(u64, u64)> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        ctx.bind_to_thread().ok()?;
        let device = ctx.cu_device();
        let before = graph_reserved(device)?;
        // SAFETY: a live device; the caller has synchronised the stream.
        unsafe { sys::cuDeviceGraphMemTrim(device) }.result().ok()?;
        let after = graph_reserved(device)?;
        Some((before >> 20, after >> 20))
    }

    pub(super) fn device_says_no(dev: &candle_core::CudaDevice) -> Option<&'static str> {
        if dev.cuda_stream().cu_stream().is_null() {
            return Some(
                "the card is driven on CUDA's legacy stream, which cannot be captured \
                 (SWARMLLM_CUDA_OWN_STREAM=0 is set)",
            );
        }
        if dev.is_event_tracking() {
            return Some("per-allocation event tracking is on");
        }
        None
    }

    fn refused(kind: &'static str, detail: impl std::fmt::Display) -> Refusal {
        Refusal {
            kind,
            detail: detail.to_string(),
        }
    }

    /// The driver's reason for refusing an update: the call's own error, its
    /// update result, and the type of the node it names (a node of the NEW
    /// graph, still alive here). The call's error leads because a call that
    /// failed before judging the graph leaves the zeroed result reading
    /// SUCCESS. `cuGraphNodeGetType` needs no context and exists since CUDA 10.
    fn why_not_updated(
        error: &impl std::fmt::Display,
        info: &sys::CUgraphExecUpdateResultInfo,
    ) -> String {
        let node = if info.errorNode.is_null() {
            "none named".to_string()
        } else {
            let mut kind = std::mem::MaybeUninit::<sys::CUgraphNodeType>::uninit();
            // SAFETY: a node of the graph the update was asked about.
            match unsafe { sys::cuGraphNodeGetType(info.errorNode, kind.as_mut_ptr()) }.result() {
                // SAFETY: the driver wrote it.
                Ok(()) => format!("{:?}", unsafe { kind.assume_init() }),
                Err(e) => format!("unknown ({e})"),
            }
        };
        format!("{error}: {:?} at a node of type {node}", info.result)
    }

    /// One step's capture, group by group. Dropping it mid-capture (an error
    /// or a panic in the forward) ends the capture and discards it: a stream
    /// left capturing refuses everything after.
    pub(crate) struct Session<'a> {
        dev: &'a candle_core::CudaDevice,
        execs: &'a mut Vec<Option<Exec>>,
        group: usize,
        copies_before: u64,
        capturing: bool,
        refusal: Option<Refusal>,
        all_updated: bool,
        refused_updates: Vec<String>,
    }

    impl Session<'_> {
        fn begin(&mut self) -> Result<(), Refusal> {
            let stream = self.dev.cuda_stream();
            let ctx = stream.context();
            ctx.bind_to_thread()
                .map_err(|e| refused("could not bind the context", e))?;
            // An error some earlier drop recorded is not this capture's: take
            // it now, so the check at the end sees only what happened inside.
            if let Err(e) = ctx.check_err() {
                tracing::debug!(error = %e, "a CUDA error recorded before the capture began");
            }
            self.copies_before = htod_copies_so_far();
            // THREAD_LOCAL: only this thread's unsafe calls break the capture,
            // so another thread reading the card's free memory does not.
            // SAFETY: a live stream cudarc created for this device.
            unsafe {
                result::stream::begin_capture(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            }
            .map_err(|e| refused("could not begin a capture", e))?;
            self.capturing = true;
            Ok(())
        }

        /// End the current group's capture, update (or build) its graph, launch it.
        fn end_and_launch(&mut self) -> Result<(), Refusal> {
            let stream = self.dev.cuda_stream();
            let ctx = stream.context();
            self.capturing = false;
            // SAFETY: the stream this thread began capturing.
            let ended = unsafe { result::stream::end_capture(stream.cu_stream()) };
            let copies = htod_copies_so_far().wrapping_sub(self.copies_before);
            // A buffer dropped inside a broken capture records its failed free
            // on the context, where it would surface as the error of some
            // later, unrelated call. Take it here, where it belongs.
            let recorded = ctx.check_err();
            let graph = match ended {
                Ok(g) if !g.is_null() => Graph(g),
                Ok(_) => return Err(refused("the capture was invalidated", "")),
                Err(e) => return Err(refused("the capture ended with an error", e)),
            };
            if copies != 0 {
                return Err(refused(
                    "a host-to-device copy inside the capture",
                    format!("{copies} copies"),
                ));
            }
            if let Err(e) = recorded {
                return Err(refused("a CUDA call failed inside the capture", e));
            }
            if self.execs.len() <= self.group {
                self.execs.resize_with(self.group + 1, || None);
            }
            let slot = &mut self.execs[self.group];
            let updated = slot.as_ref().is_some_and(|x| {
                // SAFETY: zeroed is a valid bit pattern for this plain C
                // struct, and the driver writes it before returning.
                let mut info: sys::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
                // SAFETY: both handles are live.
                match unsafe { sys::cuGraphExecUpdate_v2(x.0, graph.0, &mut info) }.result() {
                    Ok(()) => true,
                    Err(e) => {
                        self.refused_updates.push(why_not_updated(&e, &info));
                        false
                    }
                }
            });
            if !updated {
                // A refused update leaves the old graph in an unspecified
                // state: replace it, as llama.cpp does.
                *slot = None;
                let mut raw: sys::CUgraphExec = std::ptr::null_mut();
                // SAFETY: `graph` is a complete captured graph; flags 0.
                unsafe { sys::cuGraphInstantiateWithFlags(&mut raw, graph.0, 0) }
                    .result()
                    .map_err(|e| refused("could not instantiate the graph", e))?;
                *slot = Some(Exec(raw));
                self.all_updated = false;
            }
            let x = slot.as_ref().expect("set just above");
            // SAFETY: a live exec, launched on the stream it was captured from.
            if let Err(e) = unsafe { result::graph::launch(x.0, stream.cu_stream()) } {
                *slot = None;
                return Err(refused("the graph would not launch", e));
            }
            Ok(())
        }

        /// End a capture without launching it.
        fn abandon(&mut self) {
            if !self.capturing {
                return;
            }
            self.capturing = false;
            let stream = self.dev.cuda_stream();
            // SAFETY: the stream this thread began capturing.
            if let Ok(g) = unsafe { result::stream::end_capture(stream.cu_stream()) } {
                if !g.is_null() {
                    drop(Graph(g));
                }
            }
            let _ = stream.context().check_err();
        }

        pub(super) fn cut(&mut self) -> Result<(), SwarmError> {
            if self.refusal.is_none() {
                let next = self.end_and_launch().and_then(|()| {
                    self.group += 1;
                    self.begin()
                });
                match next {
                    Ok(()) => return Ok(()),
                    Err(r) => {
                        self.abandon();
                        self.refusal = Some(r);
                    }
                }
            }
            Err(SwarmError::Internal(super::GROUP_REFUSED.to_string()))
        }
    }

    impl Drop for Session<'_> {
        fn drop(&mut self) {
            self.abandon();
        }
    }

    /// On `Ok`, whether every group's graph was updated in place, and why any
    /// existing one could not be.
    pub(super) fn capture_in_groups(
        dev: &candle_core::CudaDevice,
        execs: &mut Vec<Option<Exec>>,
        forward: impl FnOnce(&mut super::Cutter<'_>) -> Result<(), SwarmError>,
    ) -> Result<super::Launched, Refusal> {
        let mut session = Session {
            dev,
            execs,
            group: 0,
            copies_before: 0,
            capturing: false,
            refusal: None,
            all_updated: true,
            refused_updates: Vec::new(),
        };
        session.begin()?;
        let mut cutter = super::Cutter { session };
        let ran = forward(&mut cutter);
        let mut session = cutter.session;
        if let Some(r) = session.refusal.take() {
            return Err(r);
        }
        if let Err(e) = ran {
            session.abandon();
            return Err(refused(super::FORWARD_FAILED, e));
        }
        session.end_and_launch()?;
        Ok((
            session.all_updated,
            std::mem::take(&mut session.refused_updates),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capture needs a decode step of the SAME conversation to have run the
    /// ordinary way first: that step did whatever a first decode step
    /// allocates and left the output's shape behind. After that, any position
    /// follows — a rolled-back guess or a jump past a several-position pass.
    #[test]
    fn a_conversations_steps_follow_once_it_has_taken_one() {
        let mut graph = DecodeGraph::default();
        assert!(
            !graph.follows_previous_step("a", 10),
            "a conversation's first decode step"
        );
        assert!(graph.follows_previous_step("a", 11));
        assert!(
            !graph.follows_previous_step("b", 12),
            "another conversation's first step"
        );
        assert!(
            graph.follows_previous_step("a", 12),
            "interleaving with another conversation does not break a run"
        );
        assert!(
            graph.follows_previous_step("a", 9),
            "a rolled-back cache (a rejected guess) still follows"
        );
        assert!(
            graph.follows_previous_step("a", 15),
            "so does a step after a several-position pass"
        );
    }

    /// A shape the driver keeps refusing to update rests — runs the ordinary
    /// way — once REBUILDS_BEFORE_RESTING of its last CHURN_WINDOW launches
    /// were rebuilds, for REST_STEPS of its own steps, then is tried again; a
    /// second rest is twice as long, and a long clean stretch forgets that.
    /// Rebuilds spread wider than the window never rest it; clean launches in
    /// between do not hide a shape rebuilt on half its launches; another
    /// shape is untouched.
    #[test]
    fn a_shape_rebuilt_every_launch_rests_then_is_tried_again() {
        let mut graph = DecodeGraph::default();
        let refused = vec!["CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED".to_string()];
        let step = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let clean = |graph: &mut DecodeGraph, n: u32| {
            for _ in 0..n {
                graph.note_launched(3, &[]);
            }
        };

        // Two rebuilds, then one more after the window has moved past them.
        graph.note_launched(3, &refused);
        graph.note_launched(3, &refused);
        clean(&mut graph, CHURN_WINDOW - 2);
        graph.note_launched(3, &refused);
        assert!(!graph.declines(3), "rebuilds spread wider than the window");
        clean(&mut graph, CHURN_WINDOW);

        // Every other launch rebuilt: rests on the third.
        for _ in 0..REBUILDS_BEFORE_RESTING {
            assert!(!graph.declines(3));
            graph.note_launched(3, &refused);
            clean(&mut graph, 1);
        }
        assert!(graph.declines(3), "rebuilt on half its launches: resting");
        assert!(
            !graph.declines(1),
            "another number of positions is untouched"
        );

        for _ in 0..REST_STEPS {
            assert!(graph.declines(3));
            graph.note_uncaptured(false, 3, &step);
        }
        assert!(
            !graph.declines(3),
            "the rest is over: capture is tried again"
        );

        for _ in 0..REBUILDS_BEFORE_RESTING {
            graph.note_launched(3, &refused);
        }
        for _ in 0..2 * REST_STEPS - 1 {
            graph.note_uncaptured(false, 3, &step);
        }
        assert!(graph.declines(3), "the second rest is twice as long");
        graph.note_uncaptured(false, 3, &step);
        assert!(!graph.declines(3));
        assert_eq!(graph.stats.rested, 2);

        // A long clean stretch: the next rest is the first length again.
        clean(&mut graph, CLEAN_LAUNCHES_TO_FORGET);
        for _ in 0..REBUILDS_BEFORE_RESTING {
            graph.note_launched(3, &refused);
        }
        for _ in 0..REST_STEPS {
            graph.note_uncaptured(false, 3, &step);
        }
        assert!(!graph.declines(3), "a settled shape's rests start over");
    }

    /// The switch is off unless asked for, and a model that is not on a card
    /// never captures.
    #[test]
    fn a_model_off_the_card_never_captures() {
        assert!(matches!(
            DecodeGraphSlot::decide(&Device::Cpu, None),
            DecodeGraphSlot::Off
        ));
    }
}
