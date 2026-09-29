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
//!    only a step that directly follows an ordinary step of the same
//!    conversation, whose KV buffers already hold the new position
//!    (`KvCacheStore::every_cache_holds`), and copies the result into a tensor
//!    made BEFORE the capture.
//! 4. No synchronisation inside — CUDA refuses that loudly, which ends the
//!    capture and falls to the next rule.
//!
//! **A refused capture never fails the request.** The caller puts the KV cache
//! back where it was and runs the step the ordinary way — the same kernels in
//! the same order, so the same answer.
//!
//! `SWARMLLM_CUDA_GRAPH=1` turns it on. OFF by default until it has been
//! measured and gated on a `--features cuda` build: v0.3.199 shipped a stream
//! change that every test and a cheaper build passed while every reply was
//! garbage (gotcha #683).

use crate::error::SwarmError;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// `SWARMLLM_CUDA_GRAPH=1`. Read once: the answer must not change between
/// two models in one worker.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CUDA_GRAPH").as_deref() == Ok("1"))
}

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

/// The refusal that is the forward's own failure, not the capture's.
pub(crate) const FORWARD_FAILED: &str = "the forward failed inside the capture";

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
                    switch = "SWARMLLM_CUDA_GRAPH=1",
                    "DIAG: decode steps on this model go to the card as one CUDA graph"
                );
                Self::On(Box::default())
            }
        }
    }
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

#[derive(Default)]
struct Stats {
    launched: u64,
    updated: u64,
    instantiated: u64,
    uncaptured: u64,
    refused: u64,
    reasons_seen: Vec<&'static str>,
    reported_at: Option<Instant>,
}

/// One model's decode graph: the instantiated graph every capture updates,
/// and what the caller needs to decide whether a step may be captured.
#[derive(Default)]
pub(crate) struct DecodeGraph {
    #[cfg(feature = "candle-cuda")]
    exec: Option<cuda::Exec>,
    /// Each conversation's last decode position. A capture needs the step
    /// before it to have run the ordinary way: that step allocated whatever a
    /// conversation's first decode step allocates, loaded every kernel module
    /// the step uses, and left the output shape in `templates`.
    last_step: HashMap<String, usize>,
    /// The decode output's shape and type, by `all_positions` — the tensor a
    /// capture copies its result into has to exist before the capture starts.
    templates: [Option<(Vec<usize>, DType)>; 2],
    /// Refusals that were the capture's fault — see [`DEFECTS_BEFORE_GIVING_UP`].
    defects: u32,
    stats: Stats,
}

impl DecodeGraph {
    /// Record a decode step at `index_pos`; true when it directly follows
    /// this conversation's previous one.
    pub(crate) fn follows_previous_step(&mut self, request_id: &str, index_pos: usize) -> bool {
        if let Some(last) = self.last_step.get_mut(request_id) {
            let follows = *last + 1 == index_pos;
            *last = index_pos;
            return follows;
        }
        if self.last_step.len() >= LAST_STEP_CAPACITY {
            self.last_step.clear();
        }
        self.last_step.insert(request_id.to_string(), index_pos);
        false
    }

    /// True once this model has been refused often enough, for reasons of the
    /// program's making, that it should stop capturing.
    pub(crate) fn gave_up(&self) -> bool {
        self.defects >= DEFECTS_BEFORE_GIVING_UP
    }

    pub(crate) fn template(&self, all_positions: bool) -> Option<&(Vec<usize>, DType)> {
        self.templates[usize::from(all_positions)].as_ref()
    }

    /// A step that ran the ordinary way: remember its output's shape.
    pub(crate) fn note_uncaptured(&mut self, all_positions: bool, out: &Tensor) {
        self.templates[usize::from(all_positions)] = Some((out.dims().to_vec(), out.dtype()));
        self.stats.uncaptured += 1;
        self.report();
    }

    /// Capture `forward` on `device`'s stream, update (or build) the graph
    /// from it, and launch it. On `Ok` the step's work is queued on the stream
    /// exactly as if it had been issued op by op; on `Err` nothing the capture
    /// recorded will ever run, and the caller must undo what `forward` did on
    /// the host and run the step the ordinary way.
    pub(crate) fn capture(
        &mut self,
        device: &Device,
        forward: impl FnOnce() -> Result<(), SwarmError>,
    ) -> Result<(), Refusal> {
        #[cfg(feature = "candle-cuda")]
        let outcome = match device {
            Device::Cuda(dev) => cuda::capture_and_launch(dev, &mut self.exec, forward),
            _ => Err(Refusal {
                kind: "not on a graphics card",
                detail: String::new(),
            }),
        };
        #[cfg(not(feature = "candle-cuda"))]
        let outcome = {
            let _ = (device, forward);
            Err(Refusal {
                kind: "built without CUDA",
                detail: String::new(),
            })
        };
        match &outcome {
            Ok(updated) => {
                self.stats.launched += 1;
                if *updated {
                    self.stats.updated += 1;
                } else {
                    self.stats.instantiated += 1;
                }
            }
            Err(refusal) => {
                self.stats.refused += 1;
                if refusal.kind != FORWARD_FAILED {
                    self.defects += 1;
                    if self.gave_up() {
                        tracing::warn!(
                            refused = self.stats.refused,
                            kinds = ?self.stats.reasons_seen,
                            last = refusal.kind,
                            "decode graph: this model's steps kept being refused — it stops \
                             capturing and decodes the ordinary way"
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
            updated_in_place = s.updated,
            instantiated = s.instantiated,
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

    pub(super) fn device_says_no(dev: &candle_core::CudaDevice) -> Option<&'static str> {
        if dev.cuda_stream().cu_stream().is_null() {
            return Some(
                "the card is driven on CUDA's legacy stream, which cannot be captured \
                 (SWARMLLM_CUDA_OWN_STREAM=1)",
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

    /// `Ok(true)` when the previous graph was updated in place, `Ok(false)`
    /// when one had to be built.
    pub(super) fn capture_and_launch(
        dev: &candle_core::CudaDevice,
        exec: &mut Option<Exec>,
        forward: impl FnOnce() -> Result<(), SwarmError>,
    ) -> Result<bool, Refusal> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        ctx.bind_to_thread()
            .map_err(|e| refused("could not bind the context", e))?;
        // An error some earlier drop recorded is not this capture's: take it
        // now, so the check after the capture sees only what happened inside.
        if let Err(e) = ctx.check_err() {
            tracing::debug!(error = %e, "a CUDA error recorded before the capture began");
        }
        let copies_before = htod_copies_so_far();
        // THREAD_LOCAL: only this thread's unsafe calls break the capture, so
        // another thread reading the card's free memory does not.
        // SAFETY: a live stream cudarc created for this device.
        unsafe {
            result::stream::begin_capture(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .map_err(|e| refused("could not begin a capture", e))?;
        let ran = forward();
        // ALWAYS end it: a stream left capturing refuses everything after.
        // SAFETY: the stream this thread began capturing just above.
        let ended = unsafe { result::stream::end_capture(stream.cu_stream()) };
        let copies = htod_copies_so_far().wrapping_sub(copies_before);
        // A buffer dropped inside a broken capture records its failed free on
        // the context, where it would surface as the error of some later,
        // unrelated call. Take it here, where it belongs.
        let recorded = ctx.check_err();
        let graph = match ended {
            Ok(g) if !g.is_null() => Graph(g),
            Ok(_) => return Err(refused("the capture was invalidated", "")),
            Err(e) => return Err(refused("the capture ended with an error", e)),
        };
        if let Err(e) = ran {
            return Err(refused(super::FORWARD_FAILED, e));
        }
        if copies != 0 {
            return Err(refused(
                "a host-to-device copy inside the capture",
                format!("{copies} copies"),
            ));
        }
        if let Err(e) = recorded {
            return Err(refused("a CUDA call failed inside the capture", e));
        }

        let updated = exec.as_ref().is_some_and(|x| {
            // SAFETY: zeroed is a valid bit pattern for this plain C struct,
            // and the driver writes it before returning.
            let mut info: sys::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
            // SAFETY: both handles are live.
            unsafe { sys::cuGraphExecUpdate_v2(x.0, graph.0, &mut info) }
                .result()
                .is_ok()
        });
        if !updated {
            // A refused update leaves the old graph in an unspecified state:
            // replace it, as llama.cpp does.
            *exec = None;
            let mut raw: sys::CUgraphExec = std::ptr::null_mut();
            // SAFETY: `graph` is a complete captured graph; flags 0.
            unsafe { sys::cuGraphInstantiateWithFlags(&mut raw, graph.0, 0) }
                .result()
                .map_err(|e| refused("could not instantiate the graph", e))?;
            *exec = Some(Exec(raw));
        }
        let x = exec.as_ref().expect("set just above");
        // SAFETY: a live exec, launched on the stream it was captured from.
        if let Err(e) = unsafe { result::graph::launch(x.0, stream.cu_stream()) } {
            *exec = None;
            return Err(refused("the graph would not launch", e));
        }
        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capture needs the step before it to have run the ordinary way for
    /// the SAME conversation: that step did whatever a first decode step
    /// allocates and left the output's shape behind.
    #[test]
    fn only_the_step_right_after_a_conversations_last_one_follows_it() {
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
            !graph.follows_previous_step("a", 9),
            "a rolled-back cache (a rejected guess) does not follow"
        );
        assert!(graph.follows_previous_step("a", 10));
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
