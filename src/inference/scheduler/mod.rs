use std::collections::HashMap;
use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{
    ModelId, ModelManifest, NodeId, PipelineAssignment, PipelineSegment, ShardId,
    TensorParallelGroup,
};

/// PipelineScheduler assembles a distributed inference pipeline
/// by selecting the best nodes for each layer range.
#[derive(Clone)]
pub struct PipelineScheduler {
    shared_state: Arc<SharedState>,
    /// What THIS node's processor would manage on a 7B, pinned by a test.
    ///
    /// The real figure is measured on the machine the code runs on — which in
    /// a test is whatever CI runner happens to be executing, under whatever
    /// load. A loaded runner measured itself slow enough that a control arm
    /// that must stay local (short prompt, cards 900 ms away) went to the
    /// cards on paper, while the same test passed on a workstation
    /// (2026-09-03). A routing test that depends on the tester's memory
    /// bandwidth is not a test of routing.
    #[cfg(test)]
    local_processor_tokens_per_sec: Option<f32>,
}

/// A candidate node for layer ranges, with scoring metadata.
/// A single node may advertise multiple non-contiguous layer ranges (e.g.,
/// layers [0,2) and [10,14)) when the GGUF's alphabetical tensor ordering
/// scatters layers across byte-range shards.
#[derive(Debug, Clone)]
struct NodeCandidate {
    node_id: NodeId,
    shard_id: ShardId,
    /// All contiguous layer ranges this node can serve for the model.
    available_ranges: Vec<(u32, u32)>,
    /// How this node is reachable. Ranked before latency so the
    /// direct-beats-relayed guarantee holds for any latency values.
    reach: ReachTier,
    latency_ms: u32,
    load: f32,
    trust_score: f32,
    /// True if this node has shard 0 (token_embd.weight, needed for is_first).
    can_be_first: bool,
    /// True if this node has the final shard (output head, needed for is_last).
    can_be_last: bool,
    /// Region proximity score: 1.0 same region, 0.5 adjacent, 0.2 distant, 0.7 unknown.
    region_score: f32,
    /// Estimated tokens/s for a 7B Q4 model (from NodeCapability). 0 = unknown.
    est_tokens_per_sec: f32,
    /// Observed per-layer latency EMA (ms/layer) for remote segments this peer
    /// served for us. None = no samples yet, use `est_tokens_per_sec` as proxy.
    /// Populated by `state.record_peer_segment_latency` after successful
    /// `forward_through_segments` hops. Used by the Parallax routing DP to
    /// replace the static capability estimate with live signal when available.
    observed_latency_ms_per_layer: Option<f32>,
    /// Observed per-layer latency EMA (ms/layer) for whole models this peer has
    /// run for us end to end — the *delegated* shape, carrying no per-token
    /// round trip. `None` for the local node and for any peer we have never
    /// delegated to. Populated from `state.observed_delegated_ms_per_layer`.
    ///
    /// Kept apart from `observed_latency_ms_per_layer` because substituting one
    /// for the other is a measured mis-pricing, not a rounding error; see
    /// `parallax::vertex_cost`.
    observed_delegated_ms_per_layer: Option<f32>,
    /// Expected attempts to get ONE intact reply out of this peer, from the
    /// measured fraction that arrive whole. 1.0 for the local node and for any
    /// peer whose path has been reliable or is unmeasured.
    ///
    /// A property of the path, not the hardware — see
    /// `PeerSpeed::delivery_intact_ratio`.
    expected_attempts: f32,
    /// True if this node is in our device pool (preferred for routing — free, trusted, low latency).
    is_pool_member: bool,
    /// Free GPU memory this node last advertised, in MB. `None` when it has no
    /// GPU, or has told us nothing.
    ///
    /// Self-reported, so it is only ever used to answer "could this peer
    /// plausibly run the whole model on its GPU" — never to rank peers against
    /// each other. See [`delegation_target`].
    gpu_vram_available_mb: Option<u64>,
    /// How many of this model's layers this peer could plausibly hold at once,
    /// from the free memory it last advertised. `None` means WE CANNOT TELL —
    /// the peer gossiped no capability, or gossiped zero (which is what every
    /// node before v0.3.103 sent, gotcha #330). See
    /// [`max_hostable_layers`] for why unknown must never exclude.
    max_hostable_layers: Option<u32>,
    /// MEASURED prefill coefficient for this peer, ms per (layer x activation
    /// byte). `None` until this node has prefilled through it — see
    /// `parallax::vertex_cost` for what stands in meanwhile.
    observed_prefill_ms_per_layer_byte: Option<f32>,
    /// Whether this peer has told us it has a graphics card. Used ONLY to pick
    /// which prefill-to-decode prior applies before a real measurement exists;
    /// never to rank peers against each other, which is what the peer's own
    /// speed figures are for.
    has_gpu: bool,
}

/// How many layers of a model a peer could plausibly hold, from the free
/// memory it last advertised — GPU first, else RAM.
///
/// **`None` means unknown and must never be read as zero.** A peer that
/// gossips no capability, or gossips a zero (every node before v0.3.103 sent
/// `vram_available_mb: 0`, gotcha #330), tells us nothing; refusing to route to
/// it would empty the candidate set during any rollout, and the swarm is always
/// mixed-version while one is in progress.
///
/// Carries [`DELEGATE_VRAM_MARGIN`] for the same reason the delegation path
/// does: the advertised figure is free memory at the last health tick, not a
/// reservation, and the worker needs room for activations and KV cache beyond
/// the weights.
///
/// Why it exists: observed on the live swarm 2026-08-21, a request for
/// llama-3.1-8b was routed whole to a node that holds all 32 layers and is the
/// fastest advertised — and which refused it in 3 s, because its 6 GB card
/// cannot take an 8 GB model and its processor budget was already full. The
/// information to route around it was on the wire and nothing consulted it.
fn max_hostable_layers(
    capability: Option<&swarmllm_types::NodeCapability>,
    bytes_per_layer: u64,
    already_warm: bool,
    // What THIS prompt's KV cache costs per layer on that peer — positions ×
    // bytes per position per layer, including the f16 mirror where its card
    // keeps one. 0 when the prompt length or the model's geometry is unknown.
    prompt_kv_bytes_per_layer: u64,
    // What THIS node has already committed to that peer and not yet seen
    // reported back — see `SharedState::peer_vram_commitments`. The advertised
    // figure is a snapshot up to 30 s old, so without this every request
    // scheduled inside one gossip window sees the same room and books it.
    committed_mb: u64,
) -> Option<u32> {
    // A peer that is already serving this model has already paid for its
    // WEIGHTS. The advertised figure is free memory RIGHT NOW (`health/monitor`
    // queries the card on every broadcast), so it EXCLUDES the weights of
    // anything resident — meaning the one node that certainly can hold the
    // model reports the least room for it, and charging it for the weights
    // would route around the best-placed machine in the swarm.
    //
    // This is gotcha #329 from the other side: "is it loaded?" and "would it
    // fit?" are different questions, and free memory only answers the second.
    //
    // It has NOT paid for this prompt's KV cache, which is why the prompt term
    // applies to a warm peer too. The capacity bound used to be weights-only:
    // a warm 6 GB card 500 ms away was handed 24 layers of an 8,111-token
    // prompt — ~2.4 GB of KV it did not have — and its worker died in
    // attention with `CUDA_ERROR_OUT_OF_MEMORY` 22 s in, with no standby
    // (gotcha #447).
    let per_layer = if already_warm {
        prompt_kv_bytes_per_layer
    } else {
        bytes_per_layer.saturating_add(prompt_kv_bytes_per_layer)
    };
    if per_layer == 0 || (!already_warm && bytes_per_layer == 0) {
        // Nothing known to charge: unknown never excludes (see the doc comment).
        return None;
    }
    let cap = capability?;
    // A GPU node is judged on its card; a node with no card, on its RAM. This
    // mirrors `allocate_offline`, which asks the same question for capacity
    // planning — the worker subprocess can host layers in either.
    let free_mb = match &cap.gpu {
        Some(g) => g.vram_available_mb,
        None => cap.ram_available_mb,
    };
    if free_mb == 0 {
        // Not "no room": no information. See the doc comment.
        return None;
    }
    // Subtract AFTER the zero test, because zero means "told us nothing" and
    // our own commitments cannot turn no information into a refusal. Saturating,
    // so an over-commitment reads as no room rather than wrapping to all of it.
    let free_mb = free_mb.saturating_sub(committed_mb);
    let usable_bytes = (free_mb as f64 * 1_048_576.0 / DELEGATE_VRAM_MARGIN) as u64;
    Some((usable_bytes / per_layer) as u32)
}

/// KV-cache bytes ONE prompt position costs across ONE layer of `meta`'s model
/// on a peer with (`on_gpu`) or without a graphics card.
///
/// The same arithmetic the worker charges at admission
/// (`kv_budget::kv_bytes_per_token` over `standard_kv_elems`), so the
/// coordinator's bound and the peer's refusal agree about the shape. A CUDA
/// worker keeps an f16 mirror of a GQA model's cache for the flash kernel
/// (`layers::model_wants_kv_mirror`), which is the same elements again at half
/// the width — 295 KB per position over Gemma-2's 24 middle layers, mirror
/// included, which is the figure the #447 worker ran out of memory against.
/// DeepSeek-style MLA caches wider decompressed heads and is priced LOW here;
/// the peer's own admission remains the backstop.
fn kv_bytes_per_position_per_layer(
    meta: &crate::inference::split::GgufTensorMeta,
    on_gpu: bool,
) -> u64 {
    let (k, v) =
        crate::inference::split::kv_budget::standard_kv_elems(meta.head_count_kv, meta.head_dim);
    let mirrored = on_gpu
        && crate::inference::layers::model_wants_kv_mirror(meta.head_count, meta.head_count_kv);
    crate::inference::split::kv_budget::kv_bytes_per_token(1, k, v, mirrored)
}

/// How far away a peer may be and still be handed a whole model, in ms.
///
/// The number that matters most here. A previous attempt at this (2026-08-03,
/// reverted in `cbbed678`) sent a request to a machine in another country while
/// one five milliseconds away was available: five minutes, then failure. This
/// bounds the damage a wrong decision can do — a peer inside this budget is on
/// the same LAN or metro, so being wrong about it costs a little latency rather
/// than the request.
///
/// **Calibrated against measured values, 2026-08-18, not from network
/// intuition.** This is `peer_registry.latency_ms`, an application-level health
/// round trip, so it carries queueing and processing time and is far larger and
/// noisier than a raw ping. Sampled on a live node:
///
/// | peers | observed |
/// |---|---|
/// | same machine / LAN | 2-134 ms (2-3 ms once idle) |
/// | other continent | 447-484 ms |
///
/// 200 ms sat 1.5x above the worst local reading and 2.2x below the best
/// remote one. The first attempt at this constant was 50 ms, which read as
/// obviously generous for a LAN and in fact **excluded a peer on the same
/// machine** whenever either node was busy — the feature would have shipped
/// inert on exactly the loaded nodes that need it.
///
/// **Raised to 1000 ms on 2026-08-31, because the premise behind 200 changed.**
/// That value was chosen to admit LAN and metro peers and exclude another
/// continent, on the reasoning that a nearby peer bounds the damage of a wrong
/// decision. In a swarm this size the nearest GPU frequently IS a continent
/// away, and the measurement says delegating to it is not the damage — it is
/// the win:
///
/// | route | measured |
/// |---|---|
/// | GPU peer at 585-611 ms, whole model (the delegation shape) | **21-25 tok/s** |
/// | this node's own processor fallback, same class of model | 9-10 tok/s |
///
/// So a peer 3x outside the old bound served **~2.3x faster than not
/// delegating at all**, and 200 ms made the feature unreachable for every peer
/// in the fleet — the only one inside it had no GPU. That is the same failure
/// the 50 ms value had, one order of magnitude out.
///
/// **Raising the ceiling cannot make a near peer lose to a far one.**
/// `candidates` arrives sorted pool-first, then reachability, then latency, so
/// the first survivor is still the nearest qualifying peer; a wider bound only
/// adds fallbacks where there were none. The compute advantage is gated
/// separately and unchanged ([`DELEGATE_MIN_CPU_SPEEDUP`]), so this bounds the
/// network cost only.
///
/// 1000 ms covers the observed fleet (worst peer 665 ms) with room for the
/// queueing this figure carries — that headroom is #331's lesson, not padding.
/// **This is still a threshold on a proxy.** The honest version compares
/// predicted delegated time against local processor time and needs no constant
/// at all; see `docs/FUTURE_WORK.md`.
const DELEGATE_MAX_LATENCY_MS: u32 = 1000;

/// How recently a peer must have served a model for us to treat it as still
/// holding it. Matches `pipeline::local::PEER_MODEL_WARM_TTL_SECS`, which asks
/// the same question to size a forward's budget.
const PEER_MODEL_WARM_TTL_SECS: u64 = 900;

/// Minimum trust before this node will hand a peer a whole prompt.
///
/// **Deliberately equal to `credit::trust::DEFAULT_TRUST`, and compared with
/// `>=`.** A peer we have merely met sits exactly at the default, so a fresh
/// pair of machines on one LAN — the case this whole path exists for — is
/// eligible immediately. Anything stricter would make the feature inert on the
/// setups that need it, which is a failure mode this codebase has shipped
/// before. What it does exclude is a peer whose record has actually gone bad:
/// a failed spot check costs 0.1, a signature violation 0.2.
///
/// A peer with no `peer_registry` entry at all scores 0.3 (`get_peer_metrics`)
/// and is correctly refused — we would be sending a plaintext prompt to
/// something we know nothing about.
const DELEGATE_MIN_TRUST: f32 = crate::credit::trust::DEFAULT_TRUST;

/// How much faster a peer must look before it is handed a model it will run on
/// its PROCESSOR, as a multiple of what this node would manage.
///
/// A peer with a graphics card that fits the model is a clear improvement over
/// our own processor fallback and needs no speed comparison. A peer that will
/// also use its processor is not obviously better at all, so it has to prove a
/// wide margin — wide enough that being wrong about it still leaves the request
/// no worse off than staying here.
///
/// **This became possible to check honestly only on 2026-08-18.** Until then
/// every processor-only node advertised `estimate_tokens_per_sec_7b(50.0,
/// false)` — a hardcoded bandwidth assumption, so an eight-channel server and a
/// fanless mini-PC both claimed exactly 1.70 tokens/s. Comparing those numbers
/// would have been comparing a constant with itself.
/// `inference::mem_bandwidth` measures the machine instead.
const DELEGATE_MIN_CPU_SPEEDUP: f32 = 2.0;

/// Headroom required on top of the model's estimated size before believing a
/// peer can host it, as a multiplier.
///
/// The peer's free VRAM is self-reported and a moment out of date, and our size
/// estimate is for OUR placement of the model. Requiring a clear margin rather
/// than a bare fit keeps a borderline case on the local node, where the outcome
/// is merely slow instead of a failed hand-off.
const DELEGATE_VRAM_MARGIN: f64 = 1.2;
/// Does this standby cover the whole of `range`, and so stand a chance of
/// taking that segment over?
///
/// The one predicate behind both the plan's coverage report and the failover
/// search in `pipeline::distributed::failover_segment`. A standby is chosen per
/// segment, so its range is normally an exact match — but the failover search
/// scans the whole list, and a standby for a DIFFERENT segment covers nothing
/// it needs.
pub(crate) fn standby_covers(standby: &PipelineSegment, range: (u32, u32)) -> bool {
    standby.layer_range.0 <= range.0 && standby.layer_range.1 >= range.1
}

/// Could this candidate actually STAND IN for `segment_layers` more layers, on
/// top of the `already_committed` layers this same plan has already given it?
///
/// [`standby_covers`] asks whether a node HOLDS the range. This asks whether it
/// could RUN it — the two questions that #452 (the planner) and #454
/// (delegation) each had to learn to separate, on the third and last path that
/// assigns layers to a machine.
///
/// **Why the running total, and not the bound alone.** A standby is chosen per
/// segment, and one node can be picked for several — the local node especially,
/// since `find_standbys` sorts it first on the sound reasoning that a node
/// holding everything is the most reliable fallback there is. Each choice was
/// weighed against nothing at all, so a 16 GB processor-only Mac mini holding
/// 12 of a 48-layer 14B became the standby for all four remote segments. Three
/// of them failed over to it in turn during one request, it was charged +9 and
/// +8 layers on top of its own 12, and at 29 of 48 the worker process was
/// killed — losing a reply that had already streamed 238 tokens over ~10
/// minutes (gotcha #462).
///
/// The plan said `standbys=4`. It had capacity to be one. HA practice has a
/// name for that: a cluster at high utilisation is "HA capable on paper" and
/// cannot actually fail over, because nothing ever checked that the spare
/// capacity was really spare.
///
/// The accounting is the one Kubernetes' scheduler uses for the same hazard —
/// a node's resources are decremented as each assignment is *decided*, not when
/// it is bound, so later decisions in the same pass see the reduced capacity
/// ("assumed pods"). Ours is simpler because a plan is built synchronously in
/// one pass, so a running tally is all it takes.
///
/// **Primary duty counts.** A node that is primary for one segment and standby
/// for another must be able to run both at once, which is exactly what happens
/// when the second segment's holder dies.
///
/// **Standbys are charged at FULL weight, not discounted by the chance of
/// being used.** Two reasons. `charge_additional_segment` never gives a range
/// back — a worker charged for a failed-over segment holds it for its life —
/// so the memory genuinely accumulates. And this cannot lose a standby that
/// would have worked: one that does not fit is refused by that same charge with
/// a 503, and `find_standbys` picks only one candidate per segment, so before
/// this the request simply died. Choosing a candidate that fits instead is a
/// strict improvement; where none fits there was never a standby, and
/// `segments_without_standby` now says so honestly rather than counting a
/// fiction (gotcha #451's lesson, on the other side).
///
/// `None` still means unknowable and never excludes — the contract
/// [`max_hostable_layers`] sets, and the reason a mixed-version swarm keeps
/// working during a rollout.
pub(crate) fn standby_has_room(
    max_hostable_layers: Option<u32>,
    already_committed: u32,
    segment_layers: u32,
) -> bool {
    match max_hostable_layers {
        None => true,
        Some(cap) => already_committed.saturating_add(segment_layers) <= cap,
    }
}

/// Could anything take over the segment(s) `node_id` is serving in this plan?
///
/// **The question the ACK fast-fail actually needs**, as opposed to "does this
/// request have any standby at all". Standbys are chosen PER SEGMENT, so a
/// five-segment plan with one standby reports `standbys=1` while four segments
/// have no backup whatsoever — and abandoning a silent peer is only justified
/// by the failover it enables (Dean & Barroso, *The Tail at Scale*: the point
/// of a second copy is that it goes to a DIFFERENT replica). Without one,
/// giving up converts a slow success into a hard 503; measured on the live
/// swarm, a peer's result arrived 1.6 s after it had been abandoned and was
/// discarded (gotcha #386).
///
/// This became more exposed, not less, once standbys stopped being assigned to
/// machines that could not run them (`standby_has_room`, gotcha #464): segments
/// that used to carry a fictional standby now honestly carry none, and the
/// per-request count would have gone on fast-failing every one of them.
///
/// **Unknown keeps the OLD behaviour, which is to fast-fail.** A peer we cannot
/// place in the plan is a path this check cannot see — a chain hop, a
/// tensor-parallel member — and those behaved this way before the standby gate
/// existed. Same reasoning as the caller treating a missing pipeline entry as
/// "yes".
pub(crate) fn peer_segment_has_standby(
    segments: &[PipelineSegment],
    standbys: &[PipelineSegment],
    node_id: &NodeId,
) -> bool {
    let mut serves_something = false;
    for seg in segments.iter().filter(|s| s.node_id == *node_id) {
        serves_something = true;
        if standbys.iter().any(|s| standby_covers(s, seg.layer_range)) {
            return true;
        }
    }
    // Not a segment holder we can see: keep the pre-gate behaviour.
    !serves_something
}

/// Layers each node is already on the hook for as a PRIMARY in this plan.
///
/// Seeds the running tally in [`find_standbys`]. A node can appear more than
/// once: under prompt privacy the local node holds both ends of a boomerang,
/// so it is primary twice before any standby is considered.
fn primary_layer_commitments(segments: &[PipelineSegment]) -> HashMap<NodeId, u32> {
    let mut m: HashMap<NodeId, u32> = HashMap::new();
    for seg in segments {
        let layers = seg.layer_range.1.saturating_sub(seg.layer_range.0);
        *m.entry(seg.node_id.clone()).or_insert(0) += layers;
    }
    m
}

/// Indices of the segments no standby covers — the ones whose holder failing
/// takes the whole request down.
///
/// Reported alongside the plan because a plain count could not answer the
/// question anyone asks of it. A three-segment boomerang whose middle is held
/// by the only peer that has those layers logs `standbys=1`, and that one
/// standby covers an end segment: the count says "there is a backup", the
/// request then fails with "NO standby available", and both lines are true.
/// A tester read the pair as a contradiction and was right to (gotcha #451).
pub(crate) fn segments_without_standby(
    segments: &[PipelineSegment],
    standbys: &[PipelineSegment],
) -> Vec<usize> {
    segments
        .iter()
        .enumerate()
        .filter(|(_, seg)| !standbys.iter().any(|s| standby_covers(s, seg.layer_range)))
        .map(|(idx, _)| idx)
        .collect()
}

/// Pick a peer to hand this whole model to, or `None` to run it here.
///
/// **This exists because holding every layer is not the same as being able to
/// run them well.** The local fast path below takes any node with full coverage
/// and runs the request there, whatever that costs — so a laptop whose GPU is
/// too small for a model runs it on the CPU even with an idle GPU machine
/// beside it on the same LAN. Measured by an external report on 2026-08-17: six
/// and a half minutes of prompt processing, and the machine reaching its
/// thermal warning, for a request a peer could have answered in seconds.
///
/// **How this differs from the attempt that was reverted**, which matters more
/// than the conditions themselves. That version priced a full local node at
/// `OUT_OF_ROOM_COST_PENALTY = 10_000` per layer and fell through to the
/// general routing search. The penalty did not merely discourage running
/// locally — it made local layers unusable, so the *split* that would have been
/// best (some layers here, the rest on a peer 5 ms away) was priced out too, and
/// the search picked a distant node holding everything. The failure was the
/// consequence, not the trigger.
///
/// So this returns a peer or nothing. It never falls through to the search, and
/// it never changes any cost the search sees. Both outcomes are a single
/// segment: run the whole model here, or hand the whole model to one named peer.
/// If nothing qualifies, the local fast path runs exactly as before.
///
/// Conditions, all required:
///
/// - **The local route is genuinely degraded** — we have a working GPU and
///   this model does not fit it, OR no usable card at all — either way the
///   request would run on this node's processor. An unreadable estimate or no
///   configured budget on a node WITH a card is NOT degraded; see
///   `ModelProcessPool::serves_on_cpu`, which owns that distinction (a
///   processor-only node was excluded until 2026-09-02, gotcha #442, and ran
///   a freshly completed model itself with GPU peers idle). Declining to serve
///   over a file we could not read would be worse than the problem being
///   solved.
/// - **The peer covers every layer.** This is a delegation, not a split. A
///   split pays a network round trip per token and measured slower than a
///   single remote segment every time it was tried (see `docs/FUTURE_WORK.md`).
/// - **The peer has room for the layers it is about to be given**, judged by
///   the same [`max_hostable_layers`] bound the routing search uses — which
///   charges this prompt's KV cache per layer, not a nominal context. This is
///   the capacity authority for the whole function; the two reasons below only
///   decide WHICH KIND of improvement a surviving peer offers. Unknown never
///   excludes, per that bound's own contract.
/// - **The peer can plausibly do better**, one of two ways: it advertises a GPU
///   with room for the model plus [`DELEGATE_VRAM_MARGIN`], or it is at least
///   [`DELEGATE_MIN_CPU_SPEEDUP`] times faster than this node's own processor.
///   Both figures are self-reported, which is why each is only ever a yes/no
///   gate paired with the locality and trust bounds below, never a ranking
///   signal — and why the processor comparison demands a wide margin rather
///   than a nose ahead.
/// - **The peer is close and directly reachable** — see
///   [`DELEGATE_MAX_LATENCY_MS`]. A relayed peer is excluded outright: relaying
///   a whole generation is not what the relay path is sized for.
/// - **The peer is trusted enough to be shown the prompt**
///   ([`DELEGATE_MIN_TRUST`]).
///
/// **Prompt privacy does not disqualify a peer here — it changes what is sent.**
/// This function answers "is there a peer worth involving"; the caller decides
/// the shape. With privacy off that is a whole-model hand-off. With privacy on
/// it is the boomerang: embedding and sampling stay local, the peer runs the
/// middle layers on encrypted activations and never sees the prompt or the
/// sampled tokens. Refusing to involve a peer at all under privacy would leave
/// the node on its CPU for no privacy gain, since the boomerang is exactly the
/// mode `encrypted_pipeline` exists to provide.
/// What [`delegation_target`] needs to know about this request.
///
/// A struct rather than a parameter list because there are enough of these
/// that a new one has to be DECIDED about rather than defaulted by position —
/// the same reason `FailoverInput` exists.
pub(crate) struct DelegationInput<'a> {
    pub local_node_id: &'a NodeId,
    pub num_layers: u32,
    /// How many of those layers the peer would ACTUALLY be given — every layer
    /// for a whole-model hand-off, the middle for a boomerang. The peer must
    /// still HOLD all `num_layers`; this is what it must have room to RUN. See
    /// [`delegated_layer_span`], which both this and the boomerang builder read
    /// so the bound and the assignment cannot disagree.
    pub layers_to_assign: u32,
    pub local_serves_on_cpu: bool,
    pub model_vram_mb: u64,
    pub local_cpu_tokens_per_sec: f32,
    /// How long the prompt is, so the peer can be priced on the work it would
    /// actually be given. `None` means unknown and the price gate stands aside.
    pub prompt_tokens: Option<u32>,
}

fn delegation_target<'a>(
    candidates: &'a [NodeCandidate],
    input: &DelegationInput<'_>,
) -> Option<&'a NodeCandidate> {
    let DelegationInput {
        local_node_id,
        num_layers,
        layers_to_assign,
        local_serves_on_cpu,
        model_vram_mb,
        local_cpu_tokens_per_sec,
        prompt_tokens,
    } = *input;
    // Only a request that would run on this node's PROCESSOR is worth handing
    // away: a card the model does not fit, or no usable card at all.
    // `ModelProcessPool::serves_on_cpu` owns that distinction (gotcha #442).
    if !local_serves_on_cpu {
        return None;
    }
    // Without a size for the model we cannot judge whether a peer has room,
    // and guessing is how the previous attempt went wrong.
    if model_vram_mb == 0 {
        return None;
    }
    let needed = (model_vram_mb as f64 * DELEGATE_VRAM_MARGIN) as u64;

    // `candidates` is already sorted pool-first, then reachability, then
    // latency, so the first survivor is the nearest trusted one.
    //
    // Every rejection is logged. This decision has a lot of conditions, all of
    // them invisible from outside, and "my fast machine is sitting idle" is
    // exactly the question an operator will need answered — as will the next
    // person to change this.
    for c in candidates.iter().filter(|c| c.node_id != *local_node_id) {
        let reason = if !c
            .available_ranges
            .iter()
            .any(|r| r.0 == 0 && r.1 >= num_layers)
        {
            "does not hold every layer"
        } else if !matches!(c.reach, ReachTier::DirectMeasured) {
            "not directly reachable with a measured latency"
        } else if c.latency_ms > DELEGATE_MAX_LATENCY_MS {
            "too far away"
        } else if c.trust_score < DELEGATE_MIN_TRUST {
            "not trusted enough to be shown the prompt"
        } else if c
            .max_hostable_layers
            .is_some_and(|cap| cap < layers_to_assign)
        {
            // The capacity authority, and the only term here that knows how
            // long THIS prompt is. Applied before either accept branch so both
            // inherit it: the branches below decide what kind of improvement a
            // peer offers, not whether it has the memory to deliver one.
            //
            // The two failures this closes were the same omission seen from
            // opposite ends (gotchas #454, #455). The boomerang branch had no
            // memory check whatsoever, so a peer whose own bound read 2-15
            // layers was handed 34 and answered with
            // `CUDA_ERROR_OUT_OF_MEMORY`. And the whole-model branch's
            // `needed` is priced at `ADMISSION_KV_CONTEXT` (4096 tokens)
            // however long the prompt really is, so an 18,000-token request
            // was accepted by a card that could not hold its KV cache and
            // returned nothing at all for the full 600 s deadline.
            //
            // `needed` stays below because it still answers a different
            // question — "is there a card here worth preferring to our
            // processor" — and it can no longer admit anything this bound
            // refuses.
            "not enough free memory for the layers this request needs"
        } else if costs_more_than_staying_here(
            c,
            candidates,
            local_node_id,
            num_layers,
            layers_to_assign,
            prompt_tokens,
        ) {
            // Priced on THIS prompt, by the same cost model the routing search
            // uses — the only term here that knows how long reading it takes.
            //
            // Everything above compares decode: `est_tokens_per_sec` is
            // `bandwidth / 4.4 * efficiency`, a memory-bandwidth estimate of
            // how fast a machine WRITES tokens. Prefill is compute-bound, and
            // the hardware spread on it is ~55x against the ~6x on decode
            // (see `parallax::vertex_cost`) — so on a long prompt the gate was
            // comparing machines on the wrong axis entirely.
            //
            // Measured live on v0.3.156: an Apple M4 advertising 14.82 tok/s
            // against a local 6.46 cleared the processor branch twice in a row
            // and took 5-6 MINUTES to the first token on ~11-12k-token
            // prompts, while the routing search had priced that same peer at
            // ~234 minutes of prefill — an order of magnitude above every
            // other candidate — and correctly avoided it. Two paths, one cost
            // model between them, and only one was consulting it.
            "priced slower for this prompt than running it here"
        } else if c.gpu_vram_available_mb.is_some_and(|free| free >= needed) {
            // A graphics card with room beats our processor fallback outright.
            return Some(c);
        } else if local_cpu_tokens_per_sec > 0.0
            && c.est_tokens_per_sec >= local_cpu_tokens_per_sec * DELEGATE_MIN_CPU_SPEEDUP
        {
            // No card, but a machine measurably faster than ours at the thing
            // that limits generation — reading memory. Both figures come from
            // `mem_bandwidth`, so this compares like with like.
            return Some(c);
        } else {
            "no graphics card with room, and not clearly faster than our own processor"
        };
        // `info`, not `debug`: this only runs when the request would run on
        // this node's processor, once per candidate per assembly, and "why is
        // my fast machine idle" is the question an operator at the default log
        // level needs answered — a tester grepping for it found nothing.
        tracing::info!(
            peer = %c.node_id,
            reach = ?c.reach,
            latency_ms = c.latency_ms,
            trust = c.trust_score,
            free_vram_mb = ?c.gpu_vram_available_mb,
            needed_vram_mb = needed,
            max_hostable_layers = ?c.max_hostable_layers,
            layers_to_assign,
            peer_tokens_per_sec = c.est_tokens_per_sec,
            local_cpu_tokens_per_sec,
            // The two figures the price gate compared, so a reader can check
            // the verdict instead of inferring a mechanism from its absence.
            // The shape this peer would actually be given, not the whole
            // model — or the line contradicts the verdict above it.
            peer_cost_ms = prompt_tokens.and_then(|_| {
                delegated_shape_cost_ms(
                    c,
                    candidates,
                    local_node_id,
                    num_layers,
                    layers_to_assign,
                    prompt_tokens,
                )
            }),
            local_cost_ms = prompt_tokens.and_then(|_| {
                candidates
                    .iter()
                    .find(|x| x.node_id == *local_node_id)
                    .map(|l| parallax::vertex_cost(l, (0, num_layers), local_node_id, num_layers, prompt_tokens).total())
            }),
            prompt_tokens = ?prompt_tokens,
            "Not handing this model to peer: {reason}"
        );
    }
    None
}

/// Would handing the whole model to `c` cost MORE than running it here?
///
/// Priced with `parallax::vertex_cost`, the same function the routing search
/// uses, so the delegation gate and the search cannot come to different
/// conclusions about the same peer — which is exactly what was observed
/// (report of 2026-09-05: the search priced a peer at ~234 minutes of prefill
/// and avoided it; this gate, which never looked at a price, chose it twice).
///
/// **Answers `false` whenever it cannot say.** No prompt length, no local
/// candidate to compare against, or a peer standing at the shared unknown
/// prior — none of those is evidence the peer is bad, and refusing on missing
/// information would strand a node on its processor beside a genuinely faster
/// machine, which is the failure `delegation_target` exists to fix. A peer
/// never measured is still tried; the measurement that first request produces
/// is what stops the second one, and the live report showed the same bad peer
/// being picked twice precisely because nothing learned.
fn costs_more_than_staying_here(
    c: &NodeCandidate,
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    layers_to_assign: u32,
    prompt_tokens: Option<u32>,
) -> bool {
    if !price_gate_enabled()
        || prompt_tokens.is_none_or(|t| t == 0)
        || !priced_from_a_measurement(c)
    {
        return false;
    }
    let Some(local) = candidates.iter().find(|x| x.node_id == *local_node_id) else {
        return false;
    };
    let Some(peer_ms) = delegated_shape_cost_ms(
        c,
        candidates,
        local_node_id,
        num_layers,
        layers_to_assign,
        prompt_tokens,
    ) else {
        return false;
    };
    let local_ms = parallax::vertex_cost(
        local,
        (0, num_layers),
        local_node_id,
        num_layers,
        prompt_tokens,
    )
    .total();
    peer_ms > local_ms
}

/// What this request costs if `peer` is given `layers_to_assign` of a
/// `num_layers` model — priced as the SHAPE that will actually be assigned.
///
/// **The whole point is that those are two different prices.**
/// `parallax::vertex_cost` exempts exactly one shape from per-token network —
/// a remote candidate covering the WHOLE model, which is entered once for the
/// entire request and streams its tokens back. Every other remote range is
/// entered once per token. So a boomerang's middle on a peer `d` away pays
/// `2 * d` per token where the same peer given the whole model pays it once,
/// and pricing the second while assigning the first is not a small error: it
/// is the difference between a round trip and a round trip per token.
///
/// Measured on the release pair 2026-09-06 (gotcha #478): a processor-only
/// node holding llama-3.2-1b whole handed the middle to a card 496 ms away and
/// took **9.1 s to return a single token**, against a local processor that
/// decodes at 4.28 tok/s. The gate had priced that peer at `(0, num_layers)`.
///
/// Same class as gotcha #434 — the work being budgeted was not the work being
/// done — and the reason this is one helper rather than an expression at each
/// site is that the gate, the line that logs its verdict, and
/// [`privacy_cost_ms`] must not be able to disagree about the same peer.
///
/// `None` when the shape cannot be priced: a boomerang needs a local candidate
/// to hold its two ends.
fn delegated_shape_cost_ms(
    peer: &NodeCandidate,
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    layers_to_assign: u32,
    prompt_tokens: Option<u32>,
) -> Option<f32> {
    let price = |c: &NodeCandidate, range: (u32, u32)| {
        parallax::vertex_cost(c, range, local_node_id, num_layers, prompt_tokens).total()
    };
    // The whole model, in one message, decoded there. Note this is also the
    // right answer for a model too short to cut a middle from: then
    // `delegated_layer_span` hands over everything.
    if layers_to_assign >= num_layers || num_layers < BOOMERANG_MIN_LAYERS || !shape_price_enabled()
    {
        return Some(price(peer, (0, num_layers)));
    }
    // The boomerang: embedding here, the middle there, sampling back here —
    // the shape `boomerang_assignment` builds.
    let local = candidates.iter().find(|c| c.node_id == *local_node_id)?;
    Some(
        price(local, (0, 1))
            + price(peer, (1, num_layers - 1))
            + price(local, (num_layers - 1, num_layers)),
    )
}

/// Privacy must add at least this long, and at least this multiple of the
/// alternative, before it is worth interrupting anyone about. Below either, the
/// guarantee is essentially free and saying so is noise.
const PRIVACY_COST_REPORT_MS: f32 = 5_000.0;
const PRIVACY_COST_REPORT_RATIO: f32 = 1.0;
/// How often the same model may say it, at most.
const PRIVACY_COST_REPORT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(600);

/// What prompt privacy is expected to ADD to this request, in milliseconds, by
/// keeping the first and last layers on this machine instead of handing the
/// whole model to `peer`.
///
/// Priced with `parallax::vertex_cost`, the same function everything else in
/// this module uses. `None` when it cannot be priced, which is not a claim that
/// privacy is free.
///
/// **This figure is REPORTED, never acted on.** See
/// `docs/FUTURE_WORK.md` § "Auto-enabled prompt privacy" for why the option to
/// drop the guarantee automatically was researched and rejected: an automatic
/// downgrade triggered by slowness is one an adversary can trigger by being
/// slow, which is the failure RFC 7507 exists to prevent for TLS. The user is
/// told what it costs and left to decide.
fn privacy_cost_ms(
    peer: &NodeCandidate,
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    prompt_tokens: Option<u32>,
) -> Option<f32> {
    if num_layers < 3 || prompt_tokens.is_none_or(|t| t == 0) {
        return None;
    }
    // Both shapes through the one helper, so the figure reported to the user
    // and the figure the gate decides on cannot drift apart.
    let shape = |assigned: u32| {
        delegated_shape_cost_ms(
            peer,
            candidates,
            local_node_id,
            num_layers,
            assigned,
            prompt_tokens,
        )
    };
    let boomerang = shape(delegated_layer_span(num_layers, true))?;
    let whole_on_peer = shape(num_layers)?;
    Some(boomerang - whole_on_peer)
}

/// Is the delegation price gate switched on?
///
/// `SWARMLLM_DELEGATE_PRICE_GATE=0` restores the pre-2026-09-05 behaviour of
/// deciding without a price. This is the one change in its release that can
/// DECLINE work a node previously handed away, and the cost model it consults
/// falls back to device-class priors for a peer whose prefill has never been
/// measured — so if it turns out to be over-cautious in the field, the symptom
/// is a node grinding on its own processor beside an idle peer, which is
/// exactly what `delegation_target` exists to prevent. The hatch is for
/// diagnosing that without a downgrade, the role `SWARMLLM_KV_RECONCILE=0`
/// plays for gotcha #462 and `SWARMLLM_BUILD_FILTER=0` for #406.
fn price_gate_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_DELEGATE_PRICE_GATE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// Is a hand-off priced as the shape it will actually be given?
///
/// `SWARMLLM_DELEGATE_SHAPE_PRICE=0` restores the pre-2026-09-06 behaviour of
/// pricing every hand-off as a whole-model delegation, whatever shape is about
/// to be assigned. That is the defect gotcha #478 records, and the hatch exists
/// for the same reason `SWARMLLM_DELEGATE_PRICE_GATE=0` does: this change can
/// DECLINE work a node previously handed away, so if it turns out to be
/// over-cautious in the field the symptom is a node on its processor beside an
/// idle peer — and the two behaviours need to be comparable inside one binary
/// to tell that from the bug it fixes.
fn shape_price_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_DELEGATE_SHAPE_PRICE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// Is this candidate's speed KNOWN — measured by us, or advertised by it — as
/// opposed to standing at the shared `UNKNOWN_COMPUTE_MS` prior?
fn priced_from_a_measurement(c: &NodeCandidate) -> bool {
    c.observed_latency_ms_per_layer.is_some()
        || c.observed_delegated_ms_per_layer.is_some()
        || c.est_tokens_per_sec > 0.0
}

/// The cheapest remote candidate for the whole model, and what it was priced
/// at — the option a reader looking at a slow local decision will ask about.
///
/// The candidate list is logged with a cost per node, so when this node keeps a
/// request that a peer was priced far cheaper for, the numbers sit in the log
/// side by side and the DECISION does not mention either of them. Three reports
/// in one day reduced to "a cheaper option was right there and nothing says why
/// it was not used" — twice with the reporter reasonably inferring a penalty
/// that does not exist. The reason was always logged; the thing it was a reason
/// ABOUT was not.
fn cheapest_whole_model_peer<'a>(
    candidates: &'a [NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    prompt_tokens: Option<u32>,
) -> Option<(&'a NodeCandidate, f32)> {
    candidates
        .iter()
        .filter(|c| {
            c.node_id != *local_node_id
                && c.available_ranges
                    .iter()
                    .any(|r| r.0 == 0 && r.1 >= num_layers)
        })
        .map(|c| {
            (
                c,
                parallax::vertex_cost(c, (0, num_layers), local_node_id, num_layers, prompt_tokens)
                    .total(),
            )
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// May the chain the search produced replace this node's own processor route?
/// `Err` names why not.
///
/// Asked only when the local node holds every layer, would run the model on its
/// processor (`ModelProcessPool::serves_on_cpu`), no single peer qualified for a
/// whole-model hand-off, and the priced search has come back cheaper than
/// running here. Two grounds to run here regardless:
///
/// - **The chain is all local.** The search agrees with the fast path, and the
///   fast path's shape — no standby, no tensor-parallel group — is the right
///   one for a request nobody else is involved in.
/// - **This node's own speed is not yet measured.** The question is whether to
///   give up running the model here, so "here" must have a price for the
///   comparison to mean anything. Home also has no network term and no peer to
///   be wrong about, so it is the safe answer under no information.
///
/// **What used to be here, and why it went** (2026-09-06, gotcha #479). The
/// second ground was the reverse: a chain containing any peer priced at
/// `UNKNOWN_COMPUTE_MS` was refused, on the rule "a route this node can price
/// is never given up for one it cannot". Three things were wrong with it.
///
/// It asserted a symmetry with `delegation_target` that does not exist. Both
/// consult `priced_from_a_measurement`, and they read it oppositely: there an
/// unmeasured peer makes the price gate STAND ASIDE, so the peer may still be
/// handed the whole model, on the stated grounds that "a peer never measured is
/// still tried; the measurement that first request produces is what stops the
/// second". Here the same fact disqualified. The multi-hop path was strictly
/// stricter than the single-hop one, and nothing said why.
///
/// It also duplicated a conservatism the cost model already applies.
/// `UNKNOWN_COMPUTE_MS` exists precisely so an unknown candidate can compete on
/// a pessimistic footing — its own doc says it is "deliberately nearer the
/// pessimistic end so an unmeasured node does not outrank a measured good one",
/// and it was raised from 0 for exactly that purpose. Refusing the outcome
/// afterwards means the prior can never do the job it was raised to do.
///
/// And the arithmetic shows what the veto actually blocked. An unpriced peer
/// taking `L` layers is charged `UNKNOWN_COMPUTE_MS * L * ASSUMED_FORWARD_PASSES`
/// = 1600·L ms, plus `2 * latency * ASSUMED_FORWARD_PASSES` of network; a local
/// processor at `e` tokens/sec is charged `2000·L/e`. Ignoring network the peer
/// wins only below **e = 1.25 tok/s**, and a peer 500 ms away needs the local
/// node under ~0.5 tok/s before it wins at all. So the search reaches for an
/// unmeasured peer only when running here would take many minutes — which is
/// the one case where trying the unknown is warranted, and the only case the
/// veto had any effect on. The threshold is derived from the cost model rather
/// than invented, which is why none is written down here.
///
/// The exploration is bounded by machinery that already exists rather than by
/// the veto: the ACK fast-fail abandons a silent peer in 10-90 s,
/// `find_standbys` sorts the local node FIRST so home is the fallback, and
/// `is_transient_remote_failure` re-routes. That is the abandonability half of
/// hedged requests (Dean & Barroso, *The Tail at Scale*) without the cost of
/// running both — an LLM decode is expensive and stateful, so a genuine
/// duplicate is not affordable here. The framing is Weitzman's Pandora's Box
/// (1979): inspecting a box is worth it when the known alternative is bad
/// enough, and it is safe because you are never forced to keep what you find.
fn pipeline_may_replace_processor_route(
    chain: &[PipelineSegment],
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
) -> Result<(), &'static str> {
    let remote: Vec<&PipelineSegment> = chain
        .iter()
        .filter(|s| s.node_id != *local_node_id)
        .collect();
    if remote.is_empty() {
        return Err("no pipeline across peers is priced faster than the processor");
    }
    // The BASELINE must be known, because that is the whole comparison: this
    // function is asked whether to give up running the model here, and "here"
    // has to have a price for that to mean anything. A local candidate still
    // standing at the prior tells us nothing, and staying home is then the
    // safe answer — home has no network term and no peer to be wrong about.
    if !candidates
        .iter()
        .find(|c| c.node_id == *local_node_id)
        .is_some_and(priced_from_a_measurement)
    {
        return Err("this node's own speed is not yet measured, so there is nothing to compare");
    }
    Ok(())
}

/// How many layers a delegated peer actually RUNS, given the shape the caller
/// will build.
///
/// The whole model when privacy is off; the middle when it is on, because
/// [`boomerang_assignment`] keeps one layer at each end here. Both that builder
/// and [`delegation_target`]'s capacity gate read this, so the number the peer
/// is checked against is by construction the number it is handed — the
/// arithmetic being written out twice is precisely how the boomerang came to be
/// sized by a check that had never been given its span.
fn delegated_layer_span(num_layers: u32, encrypted: bool) -> u32 {
    if encrypted && num_layers >= BOOMERANG_MIN_LAYERS {
        num_layers - 2
    } else {
        num_layers
    }
}

/// Fewest layers a boomerang can be cut from: one at each end and at least one
/// in the middle.
const BOOMERANG_MIN_LAYERS: u32 = 3;

/// Build the boomerang: embedding here, the middle layers on `peer`, sampling
/// back here.
///
/// **Constructed rather than searched, for the same reason the whole-model
/// hand-off is.** Asked to route this, the general search legitimately answers
/// "run all of it locally": that satisfies the encrypted constraint (first and
/// last segments are local) at zero network cost, and nothing in its cost model
/// knows the local node is about to fall back to its CPU. Verified on two nodes
/// on 2026-08-18 — skipping the local fast path alone produced
/// `segments=1 node=<local> layer_start=0 layer_end=28`, which is not a
/// boomerang. Teaching the search that local compute is expensive here is what
/// the reverted `cbbed678` did, and it distorted every other route.
///
/// The split is deliberately lopsided: one layer at each end, everything else on
/// the peer. The local segments exist to satisfy privacy — the first does the
/// token embedding, the last the norm and output head — and every layer kept
/// here is a layer running on the CPU we are trying to get off.
///
/// `None` when the model is too short to split three ways, or either side does
/// not cover what it needs; the caller then keeps the request local.
fn boomerang_assignment(
    local: &NodeCandidate,
    peer: &NodeCandidate,
    num_layers: u32,
) -> Option<Vec<PipelineSegment>> {
    // Need a layer at each end and at least one in the middle.
    if num_layers < BOOMERANG_MIN_LAYERS {
        return None;
    }
    let covers = |c: &NodeCandidate, from: u32, to: u32| {
        c.available_ranges.iter().any(|r| r.0 <= from && r.1 >= to)
    };
    // The local node must own both ends — that IS prompt privacy — and the peer
    // must cover the middle it is being given.
    if !local.can_be_first || !local.can_be_last {
        return None;
    }
    if !covers(local, 0, 1) || !covers(local, num_layers - 1, num_layers) {
        return None;
    }
    if !covers(peer, 1, num_layers - 1) {
        return None;
    }
    Some(vec![
        PipelineSegment {
            node_id: local.node_id.clone(),
            shard_id: local.shard_id.clone(),
            layer_range: (0, 1),
        },
        PipelineSegment {
            node_id: peer.node_id.clone(),
            shard_id: peer.shard_id.clone(),
            layer_range: (1, num_layers - 1),
        },
        PipelineSegment {
            node_id: local.node_id.clone(),
            shard_id: local.shard_id.clone(),
            layer_range: (num_layers - 1, num_layers),
        },
    ])
}

/// Maximum number of GPUs in a tensor-parallel group. AllReduce communication
/// between layers requires low latency, so groups are bounded to LAN-class
/// peers; 4 keeps the all-reduce ring small enough for sub-millisecond
/// per-token sync on a single switch.
const MAX_TP_GROUP_SIZE: usize = 4;

/// Latency charged to a holder reachable only through an application-level
/// relay (NETWORKING_PLAN §4 "reachable-via-relay" tier).
///
/// A relayed forward is us → relay → target instead of us → target, so it costs
/// roughly one extra RTT each way. This is a *cost* adjustment used for ranking
/// within a reachability tier; it is NOT what keeps direct ahead of relayed —
/// see [`ReachTier`] for that. An additive penalty cannot enforce an ordering,
/// which is exactly how the old arrangement failed.
const RELAY_HOP_LATENCY_PENALTY_MS: u32 = 150;

/// Latency assumed for a peer we have never successfully timed.
///
/// Deliberately pessimistic. The previous default was 100 ms, which is *better
/// than most real peers* — so a peer we knew nothing about outranked one we had
/// measured and found merely mediocre. Unknown is not the same as good.
/// [`ReachTier`] already sorts unmeasured peers behind measured ones in the
/// same tier; this value only affects cost arithmetic.
const UNMEASURED_PEER_LATENCY_MS: u32 = 300;

/// Latency assumed for a peer that is not in the registry at all.
const UNKNOWN_PEER_LATENCY_MS: u32 = 400;

/// Per-layer compute cost assumed for a peer with neither a measurement nor an
/// advertised throughput. Chosen to sit in the middle of the range real peers
/// occupy, so an unrated peer is neither favoured nor disqualified — the cost
/// is then decided by the terms we *do* know, principally network latency.
const DEFAULT_COMPUTE_MS_PER_LAYER: f32 = 20.0;

/// How a candidate is reachable, and whether its latency is a measurement or a
/// guess. **Ordered best-first, and compared before any cost.**
///
/// This exists because the documented guarantee — "a directly-connected holder
/// always outranks a relayed one" — was implemented as an additive 150 ms
/// penalty, and an additive penalty cannot guarantee an ordering. A relay-only
/// peer that had never been timed scored `100 (default) + 150 = 250` and so
/// beat a *measured* direct peer at 570 ms. Both halves of that were wrong: the
/// unknown peer was flattered by the optimistic default, and the penalty was
/// too small to dominate a real-world latency spread. Observed live on
/// 2026-08-01, where the relay-only peer was also the one whose forward timed
/// out with no standby.
///
/// Making the tier a separate, higher-priority sort key means the guarantee
/// holds for any latency values whatsoever. Relayed holders remain *usable* —
/// they simply rank behind direct ones, which is what the tier was always
/// meant to express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReachTier {
    /// This node. No network at all.
    Local,
    /// Directly connected, and we have a real latency sample.
    DirectMeasured,
    /// Directly connected, never successfully timed.
    DirectUnmeasured,
    /// Reachable only through a relay, with a latency sample.
    RelayedMeasured,
    /// Reachable only through a relay, never timed. The weakest evidence there
    /// is: we know neither that we can reach it directly nor how far it is.
    RelayedUnmeasured,
}

/// Static adjacency table for adjacent regions (0.5 score).
/// These pairs represent geographically close countries where cross-region
/// latency is typically acceptable for inference.
const ADJACENT_REGIONS: &[(&str, &str)] = &[
    ("US", "CA"),
    ("US", "MX"),
    ("DE", "FR"),
    ("DE", "NL"),
    ("DE", "AT"),
    ("DE", "CH"),
    ("DE", "PL"),
    ("FR", "ES"),
    ("FR", "IT"),
    ("FR", "BE"),
    ("GB", "IE"),
    ("GB", "FR"),
    ("GB", "NL"),
    ("JP", "KR"),
    ("JP", "TW"),
    ("AU", "NZ"),
    ("SE", "NO"),
    ("SE", "FI"),
    ("SE", "DK"),
    ("BR", "AR"),
    ("SG", "MY"),
    ("IN", "BD"),
];

/// Check if two regions are adjacent.
fn regions_adjacent(a: &str, b: &str) -> bool {
    ADJACENT_REGIONS.iter().any(|(x, y)| {
        (x.eq_ignore_ascii_case(a) && y.eq_ignore_ascii_case(b))
            || (x.eq_ignore_ascii_case(b) && y.eq_ignore_ascii_case(a))
    })
}

impl PipelineScheduler {
    /// Say what prompt privacy is costing this request, when it is material.
    ///
    /// Rate-limited per model: a boomerang is chosen on EVERY request to that
    /// model, and anything repeated per request is repeated at the user for
    /// ever — the lesson the manifest-rejection limiter already carries.
    fn report_privacy_cost(
        &self,
        model_id: &ModelId,
        peer: &NodeCandidate,
        candidates: &[NodeCandidate],
        local_node_id: &NodeId,
        num_layers: u32,
        prompt_tokens: Option<u32>,
    ) {
        let Some(extra_ms) =
            privacy_cost_ms(peer, candidates, local_node_id, num_layers, prompt_tokens)
        else {
            return;
        };
        let whole_on_peer = parallax::vertex_cost(
            peer,
            (0, num_layers),
            local_node_id,
            num_layers,
            prompt_tokens,
        )
        .total();
        if extra_ms < PRIVACY_COST_REPORT_MS
            || whole_on_peer <= 0.0
            || extra_ms / whole_on_peer < PRIVACY_COST_REPORT_RATIO
        {
            return;
        }
        if !self
            .shared_state
            .note_privacy_cost_reported(model_id, PRIVACY_COST_REPORT_INTERVAL)
        {
            return;
        }
        let seconds = (extra_ms / 1000.0).round() as i64;
        tracing::info!(
            model = %model_id.0,
            peer = %peer.node_id,
            privacy_extra_ms = extra_ms,
            whole_on_peer_ms = whole_on_peer,
            prompt_tokens = ?prompt_tokens,
            "Prompt privacy is keeping the first and last layers on this node, \
             which is what most of this request's wait will be"
        );
        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "inference",
                "privacy_cost",
                format!(
                    "Keeping your prompt private is adding about {seconds}s to replies from \
                     {}. The first and last steps stay on this computer so no other machine \
                     sees your words, and this computer is the slow part. You can turn this \
                     off for this model in Settings if you would rather have speed.",
                    model_id.0
                ),
            )
            .with_model(model_id.0.clone())
            .with_detail_num(seconds),
        );
    }

    pub fn new(shared_state: Arc<SharedState>) -> Self {
        Self {
            shared_state,
            #[cfg(test)]
            local_processor_tokens_per_sec: None,
        }
    }

    /// A scheduler whose local node's PROCESSOR speed is `tokens_per_sec`
    /// rather than whatever this machine measures — see the field.
    #[cfg(test)]
    pub(super) fn with_local_processor_speed(
        shared_state: Arc<SharedState>,
        tokens_per_sec: f32,
    ) -> Self {
        Self {
            shared_state,
            local_processor_tokens_per_sec: Some(tokens_per_sec),
        }
    }

    /// The pinned processor speed, when a test set one; `None` in production.
    fn pinned_local_processor_speed(&self) -> Option<f32> {
        #[cfg(test)]
        {
            self.local_processor_tokens_per_sec
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    /// Assemble a pipeline for the given model.
    ///
    /// Algorithm (from spec):
    /// 1. Fetch model manifest from registry
    /// 2. Determine required layer ranges (0..num_layers)
    /// 3. Query model_registry.shard_holders for all nodes hosting shards of this model
    /// 4. For each node, fetch current load and latency from peer_registry
    /// 5. Greedy assignment: sort candidates by (latency ASC, load ASC, trust DESC),
    ///    assign the best available node covering the widest contiguous layer range
    /// 6. If any layer range has no available node -> fail
    /// 7. Identify standby nodes for each segment
    /// 8. Return PipelineAssignment
    #[cfg(test)]
    pub(crate) fn assemble_pipeline(
        &self,
        model_id: &ModelId,
        local_node_id: &NodeId,
    ) -> Result<PipelineAssignment, SwarmError> {
        self.assemble_pipeline_for(model_id, local_node_id, uuid::Uuid::new_v4(), None)
    }

    /// Assemble a pipeline for the given model with a specific request ID.
    pub fn assemble_pipeline_for(
        &self,
        model_id: &ModelId,
        local_node_id: &NodeId,
        request_id: uuid::Uuid,
        // Roughly how many tokens of prompt this request carries, when the
        // caller knows. `None` prices the request exactly as this scheduler did
        // before prompt length was threaded through, so a caller with no prompt
        // in hand loses nothing.
        prompt_tokens: Option<u32>,
    ) -> Result<PipelineAssignment, SwarmError> {
        let manifest = self
            .shared_state
            .model_registry
            .get_manifest(model_id)
            .ok_or_else(|| SwarmError::ModelNotAvailable(model_id.clone()))?;

        let num_layers = manifest.num_layers;
        if num_layers == 0 {
            return Err(SwarmError::PipelineError(
                "Model has zero layers".to_string(),
            ));
        }

        let start = std::time::Instant::now();

        // Per-model choice, then explicit global, then on automatically when this
        // node holds both ends — see `encrypted_pipeline_for`.
        let encrypted = self.shared_state.encrypted_pipeline_for(model_id);
        if encrypted {
            tracing::info!(
                model = %model_id,
                "Encrypted pipeline active — forcing first+last segments to local node"
            );
        }

        // Where would THIS request run on this node — the card, or the
        // processor? Asked at most once per assembly, and only if the local
        // node turns out to hold any of the model: `serves_on_cpu` prices the
        // model against the graphics budget, which reads its header off disk
        // for a model with no resident worker, and that cost belongs on the
        // path that needs the answer rather than on every assembly.
        //
        // The answer feeds two things. The local candidate's own speed
        // (`gather_candidates`), so the search prices this node by the device
        // the request would actually get — a node whose card is too small for
        // this model was priced at its card's speed, which is a speed it was
        // not going to deliver. And the whole-model hand-off below.
        let local_runs_on_processor = std::cell::OnceCell::new();
        let local_on_processor = || {
            *local_runs_on_processor
                .get_or_init(|| self.shared_state.model_process_pool.serves_on_cpu(model_id))
        };

        // Gather all candidates: nodes that have shards for this model
        let candidates = self.gather_candidates(
            &manifest,
            local_node_id,
            request_id,
            prompt_tokens,
            &local_on_processor,
        );
        if candidates.is_empty() {
            // In private mode, give a specific error showing which shards are missing
            if self
                .shared_state
                .credits
                .private_mode
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                // Find which shards no allowed node holds. R134.7: also fold in
                // cross-pool extras so the error message matches what
                // gather_candidates considered eligible.
                let mut allowed =
                    crate::pool::scope::allowed_node_set(&self.shared_state).unwrap_or_default();
                allowed.extend(crate::pool::scope::cross_pool_extras(
                    &self.shared_state,
                    &manifest.id,
                ));
                let missing: Vec<u32> = manifest
                    .shards
                    .iter()
                    .filter(|s| {
                        let sid = ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        };
                        let holders = self.shared_state.model_registry.shard_holders(&sid);
                        !holders.iter().any(|h| allowed.contains(h))
                    })
                    .map(|s| s.index)
                    .collect();
                return Err(SwarmError::PrivateModeUnavailable {
                    model_id: manifest.name.clone(),
                    missing_shards: missing,
                });
            }
            return Err(SwarmError::InsufficientCapacity(model_id.clone()));
        }

        // Holding every layer is not the same as being able to run them well.
        // Before taking the local fast path, check whether this node is about
        // to serve the model from its CPU because the model does not fit its
        // GPU — and whether a nearby peer could do better.
        //
        // Two ways it can, and prompt privacy decides which:
        //
        // - Privacy OFF: hand the peer the whole model. One segment, no
        //   per-token network, the `remote_generate` fast path.
        // - Privacy ON: the boomerang this node already knows how to build —
        //   embedding and sampling stay here, the middle layers go to the peer
        //   as encrypted activations. The peer never sees the prompt or the
        //   sampled tokens, so the guarantee is kept in full.
        //
        // The second is what `encrypted_pipeline` is FOR, and the routing for
        // it is already wired: `route_shortest_path` is passed
        // `parallax_partial_ranges || encrypted`, which lets a peer holding the
        // whole model be cut down to a middle segment. All that was missing is
        // getting there — the local fast path below returns first, so a
        // privacy-on node ran everything on its own CPU with an idle GPU peer
        // beside it and no way to use it.
        let local_covers_everything = candidates.iter().any(|c| {
            c.node_id == *local_node_id
                && c.available_ranges
                    .iter()
                    .any(|r| r.0 == 0 && r.1 >= num_layers)
        });
        // Set below when this node holds every layer but would run the model on
        // its processor and no single peer could take the whole of it: then the
        // routing search is allowed to compete with the local fast path, on the
        // strength of the local candidate now being priced at processor speed.
        let mut pipeline_may_beat_local = false;
        if local_covers_everything {
            let pool = &self.shared_state.model_process_pool;
            // Ask the cheap question first and only price the model if the
            // answer was yes. `delegation_target` returns `None` on a node that
            // is not degraded WITHOUT reading `model_vram_mb`, but Rust
            // evaluates every argument before the call — and
            // `estimated_gpu_mb` re-reads `gguf_header.bin` and scans the model
            // directory. On a healthy node that whole reading was performed and
            // thrown away, once per assembly, on the pipeline path — which is
            // the COLD-START path a first reply waits on.
            //
            // This is exactly equivalent, because the flag being false is the
            // first thing `delegation_target` tests.
            //
            // "Would this request run on our processor" — a node with no card
            // at all included, since 2026-09-02 (gotcha #442). It used to ask
            // only whether a card we HAVE was too small, so a processor-only
            // node holding every shard ran the model itself with GPU peers
            // idle beside it.
            let local_is_degraded = local_on_processor();
            let delegate_to = if local_is_degraded {
                delegation_target(
                    &candidates,
                    &DelegationInput {
                        local_node_id,
                        num_layers,
                        // Privacy decides the shape, and the shape decides how
                        // many layers the peer must have room for — the middle
                        // only, when the two ends stay here.
                        layers_to_assign: delegated_layer_span(num_layers, encrypted),
                        local_serves_on_cpu: true,
                        model_vram_mb: pool.estimated_gpu_mb(model_id).unwrap_or(0),
                        // OUR processor speed, not our graphics card's: this
                        // only runs when the model does not fit the card, so
                        // the processor is what the request would actually get
                        // here.
                        local_cpu_tokens_per_sec:
                            crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(
                                crate::inference::mem_bandwidth::measured_gbps().unwrap_or(0.0),
                                false,
                            ),
                        prompt_tokens,
                    },
                )
            } else {
                None
            };
            if let Some(peer) = delegate_to {
                if encrypted {
                    // Tell the user what privacy is costing, when it is
                    // material — and do nothing else about it.
                    //
                    // Researched and decided 2026-09-05 (report: 73.8s with the
                    // boomerang against 12.4s without, on a ~10k-token prompt;
                    // the two layers kept here were 55.3s of the 73.8). The
                    // option of dropping the guarantee automatically past some
                    // ratio was rejected: a downgrade triggered by slowness is
                    // one a hostile peer can trigger BY BEING SLOW, and what it
                    // would win is the plaintext prompt — precisely the thing
                    // the boomerang exists to withhold. That is the failure
                    // RFC 7507 exists to prevent for TLS, where a transient
                    // network fault was enough to induce a permanent-feeling
                    // downgrade. Chrome's HTTPS-First warns before falling
                    // back; Apple's Private Relay tells you when a network has
                    // turned it off; the one system that does downgrade
                    // silently, Firefox's DoH fallback, is criticised for
                    // exactly that silence.
                    self.report_privacy_cost(
                        model_id,
                        peer,
                        &candidates,
                        local_node_id,
                        num_layers,
                        prompt_tokens,
                    );
                    // Boomerang. Skipping the local fast path is the whole
                    // change: the distributed assembly below already forces the
                    // first and last segments onto this node and already
                    // enables partial ranges when `encrypted`, so it can cut
                    // this peer's whole-model range down to the middle. The
                    // peer sees encrypted hidden states and nothing else.
                    //
                    // Privacy is not traded away for speed here, and it must
                    // never be: handing over the WHOLE model would let the peer
                    // read the prompt, which is the one thing this setting
                    // promises will not happen.
                    let local_cand = candidates
                        .iter()
                        .find(|c| c.node_id == *local_node_id)
                        .and_then(|l| boomerang_assignment(l, peer, num_layers));
                    if let Some(segments) = local_cand {
                        tracing::info!(
                            model = %model_id,
                            peer = %peer.node_id,
                            peer_latency_ms = peer.latency_ms,
                            middle = ?(1, num_layers - 1),
                            // The two numbers that decide whether this peer can
                            // hold what it is being given. The sibling
                            // whole-model line has always carried its memory
                            // figure and this one carried none, so the one log
                            // written to explain "why this peer" omitted the
                            // only field that would have shown the mismatch
                            // (gotcha #454).
                            peer_free_vram_mb = ?peer.gpu_vram_available_mb,
                            peer_max_hostable_layers = ?peer.max_hostable_layers,
                            "This model does not fit our GPU. Prompt privacy is on, so the \
                             first and last layers stay here and a nearby peer runs the \
                             middle — it sees encrypted activations, never the prompt"
                        );
                        return Ok(PipelineAssignment {
                            request_id,
                            segments,
                            standbys: vec![],
                            tp_groups: vec![],
                            supports_speculative: true,
                        });
                    }
                } else {
                    tracing::info!(
                        model = %model_id,
                        peer = %peer.node_id,
                        peer_latency_ms = peer.latency_ms,
                        peer_free_vram_mb = ?peer.gpu_vram_available_mb,
                        peer_max_hostable_layers = ?peer.max_hostable_layers,
                        "This model does not fit our GPU, so a nearby peer runs the whole \
                         of it instead of falling back to our CPU"
                    );
                    return Ok(PipelineAssignment {
                        request_id,
                        segments: vec![PipelineSegment {
                            node_id: peer.node_id.clone(),
                            shard_id: peer.shard_id.clone(),
                            layer_range: (0, num_layers),
                        }],
                        // No standby. If this peer fails, the retry in
                        // `dispatch_single` re-routes — and this node still holds
                        // every layer, so the request can always come home.
                        //
                        // That retry is `segment_ran_out_of_machines`, added
                        // 2026-09-04. Until then this comment described an
                        // intention rather than a behaviour: the error a lost
                        // peer produces here is `SegmentFailoverExhausted`,
                        // which was on neither of the retry gate's two lists,
                        // so a delegated request whose peer disconnected simply
                        // failed with the local node idle beside it holding
                        // every layer (gotcha #456).
                        standbys: vec![],
                        tp_groups: vec![],
                        supports_speculative: true,
                    });
                }
            }
            // No single peer could take the whole model — but the request
            // would still run on this node's processor, and a PIPELINE over
            // several peers' cards may be faster than that (gotcha #444; the
            // tester's case in #442: a 14B no one card holds, on a
            // processor-only node that had just finished acquiring it, where
            // the day before a three-segment pipeline across two GPU nodes
            // answered in seconds). Let the search below compete with the
            // fast path. It can, now, because the local candidate is priced
            // at the processor's measured speed rather than the card's — the
            // honest input the reverted `cbbed678` lacked when it priced local
            // layers at a constant 10,000 and sent a request abroad past a
            // peer 5 ms away. The search still charges every remote hop per
            // token, so a short prompt with only distant cards on offer stays
            // here; a long prompt on a processor does not.
            //
            // Only the priced search may make this call: greedy has no cost
            // to compare, so with parallax routing off the fast path stands.
            pipeline_may_beat_local = local_is_degraded
                && self.shared_state.config.inference.parallax_routing
                && candidates.len() > 1;
        }

        // Fast path: if the local node has full layer coverage (0..num_layers),
        // run entirely locally without involving remote peers.  This prevents
        // "Segment N failed with no standby" errors caused by remote peers that
        // hold overlapping shards being pulled into the pipeline unnecessarily.
        let local_cand = candidates.iter().find(|c| {
            c.node_id == *local_node_id
                && c.available_ranges
                    .iter()
                    .any(|r| r.0 == 0 && r.1 >= num_layers)
        });
        if let Some(local_cand) = local_cand.filter(|_| !pipeline_may_beat_local) {
            // Why this node kept the request, in the same line that says it
            // did. Reaching here with peers in the list means the priced search
            // was not allowed to compete — this node is not on its processor,
            // or parallax routing is off, or nothing else was a candidate — and
            // that is the question a slow local answer provokes.
            let passed_over =
                cheapest_whole_model_peer(&candidates, local_node_id, num_layers, prompt_tokens);
            tracing::info!(
                model = %model_id,
                num_layers,
                candidates = candidates.len(),
                cheapest_peer = ?passed_over.map(|(c, _)| c.node_id.to_string()),
                cheapest_peer_cost_ms = ?passed_over.map(|(_, ms)| ms),
                local_runs_on_processor = pipeline_may_beat_local,
                parallax_routing = self.shared_state.config.inference.parallax_routing,
                "Local node has full layer coverage — single local segment"
            );
            return Ok(Self::local_only_assignment(
                request_id,
                local_node_id,
                local_cand,
                num_layers,
            ));
        }

        // Distributed layer assignment: prefer Parallax shortest-path DP when
        // enabled; fall back to greedy on any failure (disjoint ranges, no
        // valid source/sink, etc.) so routing never regresses below greedy.
        let raw_segments = if self.shared_state.config.inference.parallax_routing {
            // Encryption forces the first and last segments onto this node,
            // so an encrypted distributed pipeline is multi-segment by
            // construction — there is no single-delegation alternative to
            // lose. The per-token cost that keeps partial ranges off by
            // default therefore does not apply, and without them a peer
            // holding a SUPERSET of the middle (very commonly the whole
            // model) offers only one indivisible range, which can be neither
            // a middle segment nor a remote encrypted end. That produced a
            // hard "No node available" for a perfectly valid boomerang.
            let partial = self.shared_state.config.inference.parallax_partial_ranges || encrypted;
            // Route twice at most. The first pass holds every peer to the layer
            // count its advertised free memory can take; the second drops that
            // bound entirely. A peer's self-reported figure is therefore allowed
            // to make a route BETTER and never to make a routable request fail —
            // which matters because the figure is stale by up to a health tick,
            // is zero on any node older than v0.3.103, and is absent for a peer
            // that has gossiped no capability at all.
            let routed = parallax::route_shortest_path(
                num_layers,
                &candidates,
                local_node_id,
                encrypted,
                partial,
                true,
                prompt_tokens,
            )
            .or_else(|first_err| {
                let relaxed = parallax::route_shortest_path(
                    num_layers,
                    &candidates,
                    local_node_id,
                    encrypted,
                    partial,
                    false,
                    prompt_tokens,
                );
                if relaxed.is_ok() {
                    tracing::info!(
                        model = %model_id,
                        constrained_err = %first_err,
                        "DIAG: no route fits the peers' advertised memory — routing without \
                         that bound, a holder may refuse and the request will re-plan"
                    );
                }
                relaxed
            });
            match routed {
                // Both arms log at `info`, deliberately. Nodes run at `info`, so
                // at `debug` which router actually chose a route was invisible in
                // every real log — and reading that absence as "parallax never
                // runs" produced a wrong diagnosis on 2026-08-03. This is once
                // per pipeline assembly, not per token, so it is affordable.
                Ok(segs) => {
                    tracing::info!(
                        model = %model_id,
                        segments = segs.len(),
                        "DIAG: parallax routing selected chain"
                    );
                    // The search ran against a local node that holds every
                    // layer and would run the model on its processor. Its
                    // answer displaces the fast path only on a chain that is
                    // genuinely remote AND priced from real figures; the log
                    // carries both costs so the choice can be checked.
                    if let (true, Some(local_cand)) = (pipeline_may_beat_local, local_cand) {
                        let local_ms = parallax::vertex_cost(
                            local_cand,
                            (0, num_layers),
                            local_node_id,
                            num_layers,
                            prompt_tokens,
                        )
                        .total();
                        let chain_ms = parallax::chain_cost_ms(
                            &segs,
                            &candidates,
                            local_node_id,
                            num_layers,
                            prompt_tokens,
                        );
                        match pipeline_may_replace_processor_route(
                            &segs,
                            &candidates,
                            local_node_id,
                        ) {
                            Ok(()) => tracing::info!(
                                model = %model_id,
                                segments = segs.len(),
                                local_processor_cost_ms = local_ms,
                                pipeline_cost_ms = chain_ms,
                                prompt_tokens = ?prompt_tokens,
                                "This node holds the whole model but would run it on its \
                                 processor; a pipeline across peers' cards is priced faster, \
                                 so the request goes there"
                            ),
                            Err(reason) => {
                                // Name the option a reader will ask about. The
                                // candidate list already carries a cost per
                                // node, so without this the log shows a peer
                                // priced 55x cheaper and a decision that never
                                // mentions it.
                                let passed_over = cheapest_whole_model_peer(
                                    &candidates,
                                    local_node_id,
                                    num_layers,
                                    prompt_tokens,
                                );
                                tracing::info!(
                                    model = %model_id,
                                    local_processor_cost_ms = local_ms,
                                    pipeline_cost_ms = chain_ms,
                                    cheapest_peer = ?passed_over.map(|(c, _)| c.node_id.to_string()),
                                    cheapest_peer_cost_ms = ?passed_over.map(|(_, ms)| ms),
                                    prompt_tokens = ?prompt_tokens,
                                    "This node holds the whole model and runs it on its \
                                     processor: {reason}"
                                );
                                return Ok(Self::local_only_assignment(
                                    request_id,
                                    local_node_id,
                                    local_cand,
                                    num_layers,
                                ));
                            }
                        }
                    }
                    segs
                }
                Err(e) => {
                    // A local node holding every layer needs no fallback route:
                    // greedy has no cost to compare against the processor, so
                    // the fast path it would have taken is the answer.
                    if let (true, Some(local_cand)) = (pipeline_may_beat_local, local_cand) {
                        let passed_over = cheapest_whole_model_peer(
                            &candidates,
                            local_node_id,
                            num_layers,
                            prompt_tokens,
                        );
                        tracing::info!(
                            model = %model_id,
                            err = %e,
                            cheapest_peer = ?passed_over.map(|(c, _)| c.node_id.to_string()),
                            cheapest_peer_cost_ms = ?passed_over.map(|(_, ms)| ms),
                            "DIAG: parallax routing unavailable — this node holds the whole \
                             model, so it runs here"
                        );
                        return Ok(Self::local_only_assignment(
                            request_id,
                            local_node_id,
                            local_cand,
                            num_layers,
                        ));
                    }
                    tracing::info!(
                        model = %model_id,
                        err = %e,
                        "DIAG: parallax routing unavailable — falling back to greedy"
                    );
                    self.greedy_assign(num_layers, &candidates, encrypted)?
                }
            }
        } else {
            self.greedy_assign(num_layers, &candidates, encrypted)?
        };

        // Merge contiguous segments on the same node into a single segment.
        // This avoids sending multiple LayerForward messages to the same node
        // when it handles its full layer range in one forward pass.
        let mut segments = Self::merge_contiguous(raw_segments);

        // Re-point each segment's `shard_id` at the first shard its layer range
        // actually covers. Candidates carry only their FIRST shard id, so a
        // segment serving a later part of a range would otherwise be labelled
        // with shard 0 — which it may not even hold. Applied here, after both
        // the parallax and greedy paths have converged, so neither can skip it.
        // Consumers needing the full span still go through
        // `ModelRegistry::shards_spanned_by_segment`.
        for seg in &mut segments {
            if let Some(first) = self
                .shared_state
                .model_registry
                .shards_overlapping_layers(&seg.shard_id.model_id, seg.layer_range)
                .into_iter()
                .min_by_key(|s| s.index)
            {
                seg.shard_id = first;
            }
        }

        // Identify standby nodes for each segment
        let standbys = self.find_standbys(&segments, &candidates, prompt_tokens, num_layers);

        // Detect tensor-parallel opportunities: LAN peers sharing the same layer range.
        // Opt-in only (`inference.tensor_parallel`, default false) — per-layer
        // AllReduce over Ethernet costs more than the compute it splits for
        // anything but a large model on a very fast LAN.
        // Skip TP when encrypted pipeline is active — no remote node should process
        // tensor data in encrypted mode (defeats the purpose of local-only embedding/sampling).
        let tp_groups = if encrypted || !self.shared_state.config.inference.tensor_parallel {
            vec![]
        } else {
            self.detect_tp_groups(&segments, &candidates)
        };

        let uncovered = segments_without_standby(&segments, &standbys);
        tracing::info!(
            request_id = %request_id,
            model = %model_id,
            candidates_count = candidates.len(),
            segments = segments.len(),
            standbys = standbys.len(),
            // Which segments a failure would be FATAL for. The bare standby
            // count says nothing about that — see `segments_without_standby`.
            segments_without_standby = ?uncovered,
            tp_groups = tp_groups.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "DIAG: assemble_pipeline_for completed"
        );

        Ok(PipelineAssignment {
            request_id,
            segments,
            standbys,
            tp_groups,
            // All current nodes advertise speculative verify-batch support. Will
            // flip to a per-peer capability check once we gate on version.
            supports_speculative: true,
        })
    }

    /// Book what `assignment` is about to ask of each peer's graphics memory,
    /// so a request scheduled moments later can see it.
    ///
    /// Called by the ROUTER, on the path a request actually executes — not from
    /// `assemble_pipeline_for`, which the dashboard also calls to preview a
    /// route. A preview that booked memory would never release it, because
    /// nothing ever calls `release_request_state` for a request that does not
    /// exist.
    ///
    /// The charge per layer is exactly what [`max_hostable_layers`] weighs, so
    /// the reservation and the bound cannot describe different quantities: a
    /// cold peer pays weights plus this prompt's KV, a warm one pays the KV
    /// alone. Segments on this node are skipped — our own loader tracks that.
    pub fn record_peer_commitments(
        &self,
        assignment: &PipelineAssignment,
        local_node_id: &NodeId,
        prompt_tokens: Option<u32>,
    ) {
        let Some(first) = assignment.segments.first() else {
            return;
        };
        let model_id = &first.shard_id.model_id;
        let Some(manifest) = self.shared_state.model_registry.get_manifest(model_id) else {
            return;
        };
        let bytes_per_layer = manifest.total_size_bytes / manifest.num_layers.max(1) as u64;
        let meta = self.shared_state.gguf_meta_for(model_id);
        let mut charges: Vec<(NodeId, u64)> = Vec::new();
        for seg in &assignment.segments {
            if seg.node_id == *local_node_id {
                continue;
            }
            let layers = u64::from(seg.layer_range.1.saturating_sub(seg.layer_range.0));
            if layers == 0 {
                continue;
            }
            let has_gpu = self
                .shared_state
                .peer_registry
                .get(&seg.node_id)
                .is_some_and(|p| p.capability.as_ref().is_some_and(|c| c.gpu.is_some()));
            let kv_per_layer = match (prompt_tokens, meta.as_ref()) {
                (Some(tokens), Some(m)) => {
                    kv_bytes_per_position_per_layer(m, has_gpu).saturating_mul(u64::from(tokens))
                }
                _ => 0,
            };
            let warm = self.shared_state.peer_model_is_warm(
                &seg.node_id,
                model_id,
                std::time::Duration::from_secs(PEER_MODEL_WARM_TTL_SECS),
            );
            let per_layer = if warm {
                kv_per_layer
            } else {
                bytes_per_layer.saturating_add(kv_per_layer)
            };
            let mb = layers.saturating_mul(per_layer) / 1_048_576;
            if mb > 0 {
                // Several segments can land on one peer; charge each.
                if let Some(entry) = charges.iter_mut().find(|(n, _)| *n == seg.node_id) {
                    entry.1 = entry.1.saturating_add(mb);
                } else {
                    charges.push((seg.node_id.clone(), mb));
                }
            }
        }
        self.shared_state
            .record_peer_vram_commitments(assignment.request_id, charges);
    }

    /// Gather all candidate nodes for the given model's shards.
    ///
    /// Groups shards by node and computes combined layer ranges using actual GGUF
    /// tensor metadata when available, falling back to manifest layer_range otherwise.
    /// `request_id` is used only to honour the per-request holder blacklist —
    /// holders that already told us, during THIS request, that they do not have
    /// the data they advertise. Without it, retracting a stale claim is futile:
    /// the DHT still lists the holder, so the retry re-learns it and picks the
    /// same dead peer (observed live 2026-07-26).
    fn gather_candidates(
        &self,
        manifest: &ModelManifest,
        local_node_id: &NodeId,
        request_id: uuid::Uuid,
        prompt_tokens: Option<u32>,
        // Would a request for this model run on the local node's PROCESSOR?
        // Consulted only if the local node holds any of the model, hence a
        // closure: the answer prices the model against the graphics budget,
        // which reads its header off disk for a model with no worker resident.
        local_runs_on_processor: &dyn Fn() -> bool,
    ) -> Vec<NodeCandidate> {
        // Private mode: compute allowed node set (None = unrestricted).
        // R134.7: when `allow_cross_pool_inference` is on and the local pool
        // can't serve this model, union the cross-pool extras into the
        // allowed set. No-op when both flags aren't on or when a local pool
        // member already holds the model.
        let allowed_set = {
            let base = crate::pool::scope::allowed_node_set(&self.shared_state);
            let extras = crate::pool::scope::cross_pool_extras(&self.shared_state, &manifest.id);
            match (base, extras.is_empty()) {
                (None, _) => None,
                (Some(set), true) => Some(set),
                (Some(set), false) => {
                    let mut merged = set;
                    merged.extend(extras);
                    Some(merged)
                }
            }
        };

        // Average bytes a layer occupies, for the per-peer capacity bound
        // below. The manifest is the only size the coordinator has; it is the
        // on-disk quantized figure, which is what the peer's loader charges
        // against its budget too.
        let bytes_per_layer = manifest.total_size_bytes / manifest.num_layers.max(1) as u64;
        // What THIS prompt's KV cache costs per layer, on a card and on a
        // processor, from the model's geometry when this node holds its header.
        // `gguf_meta_for` is the only way to ask — it reads the header on a
        // miss, because nothing filled the map when a model's shards arrived
        // from the swarm mid-run and this bound was therefore inert on exactly
        // the fresh-distribution case it was written for (gotcha #451).
        // Unknown → 0 → the bound charges weights only, as it always did.
        let (prompt_kv_per_layer_gpu, prompt_kv_per_layer_cpu) =
            match (prompt_tokens, self.shared_state.gguf_meta_for(&manifest.id)) {
                (Some(tokens), Some(meta)) => (
                    kv_bytes_per_position_per_layer(&meta, true).saturating_mul(u64::from(tokens)),
                    kv_bytes_per_position_per_layer(&meta, false).saturating_mul(u64::from(tokens)),
                ),
                _ => (0, 0),
            };

        // Build set of pool member NodeIds for preferred routing.
        // Pool devices are trusted, free (no credit cost), and usually low latency.
        let pool_member_ids: std::collections::HashSet<NodeId> = {
            if let Ok(ps) = self.shared_state.credits.pool_state.try_read() {
                ps.as_ref()
                    .map(|s| s.members.iter().map(|m| m.node_id.clone()).collect())
                    .unwrap_or_default()
            } else {
                std::collections::HashSet::new()
            }
        };

        // First, collect which shard indices each node holds
        let mut node_shards: std::collections::HashMap<NodeId, Vec<u32>> =
            std::collections::HashMap::new();

        for shard in &manifest.shards {
            let shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = self.shared_state.model_registry.shard_holders(&shard_id);
            for node_id in holders {
                // Private mode: skip nodes outside the allowed set
                if let Some(ref allowed) = allowed_set {
                    if !allowed.contains(&node_id) {
                        continue;
                    }
                }
                // Skip peers we can't currently reach. Two stale-source paths:
                // (1) The DHT periodically re-injects stale providers (peers
                //     that recently disconnected but whose Kademlia provider
                //     records haven't expired yet) into shard_holders.
                // (2) When a peer disconnects mid-pipeline, peer_registry is
                //     intentionally preserved for reconnect attempts (see
                //     handle_connection_closed `in_active_pipeline` branch),
                //     but the libp2p `connected_node_ids` set is cleared
                //     unconditionally — making it the right liveness oracle.
                // Without this filter, the scheduler picks a dead peer,
                // remote-generate sends to it, and the request hangs until
                // the 120s first-token timeout.
                // NETWORKING_PLAN §4 Phase 1 "reachable-via-relay" tier: a peer
                // we hold no libp2p connection to is still usable if a relay we
                // share can carry the inference to it. Without this the
                // app-level relay could only substitute the data path for an
                // already-connected peer — the both-NAT'd case it exists for
                // never reached the scheduler at all. Ranked below direct peers
                // via a latency penalty in `get_peer_metrics`, so direct is
                // always preferred when both are available.
                let is_local = node_id == *local_node_id;
                // Already failed us on this request with "I don't have that
                // data" — skip regardless of what the registry or DHT says.
                if !is_local
                    && self
                        .shared_state
                        .holder_blacklisted_for_request(request_id, &node_id)
                {
                    continue;
                }
                if !is_local
                    && !self.shared_state.connected_node_ids.contains(&node_id)
                    && !self.shared_state.peer_reachable_via_relay(&node_id)
                {
                    continue;
                }
                node_shards.entry(node_id).or_default().push(shard.index);
            }
        }

        let mut candidates = Vec::new();

        for (node_id, mut shard_indices) in node_shards {
            shard_indices.sort();

            // Compute ALL contiguous layer ranges for this node's shards
            // from manifest layer_range data.
            let ranges = {
                let manifest_ranges = crate::inference::split::available_layer_ranges_from_manifest(
                    manifest,
                    &shard_indices,
                );
                if !manifest_ranges.is_empty() {
                    manifest_ranges
                        .into_iter()
                        .map(|(s, e)| (s as u32, e as u32))
                        .collect::<Vec<_>>()
                } else {
                    // Fallback: use manifest layer ranges (approximate, single range)
                    let mut ls = manifest.num_layers as usize;
                    let mut le = 0usize;
                    for &idx in &shard_indices {
                        if let Some(shard) = manifest.shards.iter().find(|s| s.index == idx) {
                            ls = ls.min(shard.layer_range.0 as usize);
                            le = le.max(shard.layer_range.1 as usize);
                        }
                    }
                    if ls < le {
                        vec![(ls as u32, le as u32)]
                    } else {
                        vec![]
                    }
                }
            };

            if ranges.is_empty() {
                continue; // No complete layers on this node
            }

            let first_shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard_indices[0],
            };
            let (reach, latency_ms, trust_score) = self.get_peer_metrics(&node_id, local_node_id);

            // Determine if this node can serve as first/last segment
            let can_be_first = shard_indices.contains(&0);
            let last_shard_idx = manifest.shard_count.saturating_sub(1);
            let can_be_last = shard_indices.contains(&last_shard_idx);

            // Use the most up-to-date load info: for local node, use active_pipelines
            // directly. For remote nodes, take the max of health-ping report and local
            // pipeline tracking (health pings can be stale by up to ~5s).
            let active_load = if &node_id == local_node_id {
                // Everything this node is doing, not just what the router
                // assembled — a local split fast-path reply and a segment
                // served for a peer both occupy the same GPU. See
                // `SharedState::active_inference_load`.
                self.shared_state.active_inference_load() as f32
            } else {
                let health_ping_load = self
                    .shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| p.active_request_count as f32)
                    .unwrap_or(0.0);
                let local_pipeline_load = self
                    .shared_state
                    .active_pipelines
                    .iter()
                    .filter(|entry| entry.value().segments.iter().any(|s| s.node_id == node_id))
                    .count() as f32;
                health_ping_load.max(local_pipeline_load)
            };

            // Compute region_score: 1.0 same, 0.5 adjacent, 0.2 distant, 0.7 unknown.
            let region_score = if &node_id == local_node_id {
                1.0 // Local node is always "same region"
            } else {
                self.compute_region_score(&node_id, local_node_id)
            };

            // The device THIS request would run on here. A node with a card
            // that this model does not fit runs it on the processor (or a
            // hybrid split, which is nearer the processor than the card), so
            // pricing the local candidate at the card's speed described a speed
            // it was not going to deliver — and made the local route look
            // unbeatable to the search (gotcha #444). Asked once, lazily, and
            // only for the local node.
            let is_local = &node_id == local_node_id;
            let local_on_processor = is_local && local_runs_on_processor();
            let local_device_name = if local_on_processor {
                None
            } else {
                self.shared_state.gpu_info.as_ref().map(|g| g.name.as_str())
            };
            // Look up speed estimation from capability gossip
            let est_tokens_per_sec = if is_local {
                // Derived the same way every peer derives its own, so this
                // compares like with like — and so a processor-only node states
                // a real figure instead of the zero its consumers read as
                // "unknown". 0.0 is still the answer when the machine's
                // bandwidth genuinely could not be measured.
                match (local_on_processor, self.pinned_local_processor_speed()) {
                    (true, Some(pinned)) => pinned,
                    _ => crate::model::auto_manage::vram::node_tokens_per_sec_7b(local_device_name)
                        .unwrap_or(0.0),
                }
            } else {
                self.shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| {
                        p.capability
                            .as_ref()
                            .map(|c| c.est_tokens_per_sec_7b)
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0)
            };

            let is_pool = pool_member_ids.contains(&node_id);
            // Includes the local node. It used to be excluded, which combined
            // with `UNKNOWN_COMPUTE_MS = 0` meant local compute was free at any
            // width — so the router would happily pile every layer onto a slow
            // local CPU rather than hand work to a faster peer. A local sample
            // carries no network component, which is correct: there isn't one.
            //
            // EXCEPT when this node has a card and this request would not use
            // it: the figure is per node, not per device, so it was measured on
            // whatever the card served for someone else and says nothing about
            // the processor about to do this work. A node with no card at all
            // keeps its samples — all its work is processor work.
            let observed_latency_ms_per_layer =
                if local_on_processor && self.shared_state.gpu_info.is_some() {
                    None
                } else {
                    self.shared_state.observed_latency_ms_per_layer(&node_id)
                };
            // Deliberately `None` for the local node: there is no such thing as
            // delegating to ourselves, and a local segment pays no network at
            // all, so the whole distinction this figure exists to draw is moot.
            let observed_delegated_ms_per_layer = if &node_id == local_node_id {
                None
            } else {
                self.shared_state.observed_delegated_ms_per_layer(&node_id)
            };
            // No path to ourselves, so nothing can be lost on the way.
            let expected_attempts = if &node_id == local_node_id {
                1.0
            } else {
                self.shared_state.peer_expected_attempts(&node_id)
            };
            let observed_prefill_ms_per_layer_byte = if &node_id == local_node_id {
                // Our own prefill cost is not measured through this path (there
                // is no segment round trip to sample), so the local node prices
                // prefill from its capability prior like any unmeasured peer.
                None
            } else {
                self.shared_state
                    .observed_prefill_ms_per_layer_byte(&node_id)
            };
            // Which prefill prior applies — a card reads a prompt ~40x faster
            // than it writes a reply, a processor ~4.5x. It follows the device
            // the request would use, not the device the node owns.
            let has_gpu = if is_local {
                local_device_name.is_some()
            } else {
                self.shared_state
                    .peer_registry
                    .get(&node_id)
                    .is_some_and(|p| p.capability.as_ref().is_some_and(|c| c.gpu.is_some()))
            };
            // The local node is bounded by asking OUR OWN loader, which is the
            // authority on what we can fit and knows what is already committed
            // to live workers — information no gossiped figure carries. It used
            // to be left unconstrained on that same reasoning, which is right
            // about who decides and wrong about when: admission runs at load
            // time, after the plan is committed and too late to reshape it. So
            // the one candidate with the best information was the only one the
            // search priced as having no memory ceiling at all, and a
            // processor-only node holding every shard was handed 36 of a 14B's
            // 48 layers, refused them, retried and produced the same plan
            // (gotcha #452).
            // What other in-flight requests have already booked on this peer
            // since its last capability broadcast. Zero for the local node,
            // whose loader knows exactly what it has committed.
            let committed_mb = if node_id == *local_node_id {
                0
            } else {
                self.shared_state
                    .committed_peer_vram_mb(&node_id, request_id)
            };
            let max_hostable_layers = if node_id == *local_node_id {
                self.shared_state
                    .model_process_pool
                    .max_local_hostable_layers(&manifest.id, has_gpu)
            } else {
                let warm = self.shared_state.peer_model_is_warm(
                    &node_id,
                    &manifest.id,
                    std::time::Duration::from_secs(PEER_MODEL_WARM_TTL_SECS),
                );
                let prompt_kv_per_layer = if has_gpu {
                    prompt_kv_per_layer_gpu
                } else {
                    prompt_kv_per_layer_cpu
                };
                self.shared_state.peer_registry.get(&node_id).and_then(|p| {
                    max_hostable_layers(
                        p.capability.as_ref(),
                        bytes_per_layer,
                        warm,
                        prompt_kv_per_layer,
                        committed_mb,
                    )
                })
            };
            let gpu_vram_available_mb = if node_id == *local_node_id {
                // Never used for the local node — the loader's own admission
                // check is the authority on whether WE can fit a model, and it
                // knows what is already committed to live workers.
                None
            } else {
                self.shared_state.peer_registry.get(&node_id).and_then(|p| {
                    p.capability.as_ref().and_then(|c| {
                        c.gpu.as_ref().map(|g| {
                            // Same deduction as the bound above, for the same
                            // reason: this figure is a snapshot the peer sends
                            // every 30 s, and it cannot know what this node
                            // booked onto it since.
                            g.vram_available_mb.saturating_sub(committed_mb)
                        })
                    })
                })
            };
            candidates.push(NodeCandidate {
                node_id,
                shard_id: first_shard_id,
                available_ranges: ranges,
                reach,
                latency_ms,
                load: active_load,
                trust_score,
                can_be_first,
                can_be_last,
                region_score,
                est_tokens_per_sec,
                observed_latency_ms_per_layer,
                observed_delegated_ms_per_layer,
                expected_attempts,
                is_pool_member: is_pool,
                gpu_vram_available_mb,
                max_hostable_layers,
                observed_prefill_ms_per_layer_byte,
                has_gpu,
            });
        }

        // At `info`, and for the same reason the parallax/greedy choice above is
        // logged at `info`: nodes run at `info`, so anything at `debug` is
        // invisible in every real log, and reasoning from its absence has
        // already produced one wrong diagnosis (2026-08-03).
        //
        // This is the line that decides a routing question nothing else can
        // answer. Measured 2026-08-19: with three holders of one model — 0.82
        // tok/s at 75 ms, 20.45 tok/s at 455 ms, 1.26 tok/s at 637 ms — the
        // router picked the last of those five times running, and the request
        // came back at 0.23 tok/s against a 36 tok/s local baseline. Replaying
        // those numbers through `route_shortest_path` picks the GPU, so the
        // divergence is in these inputs; without them being visible there is no
        // way to tell which one, and no admin endpoint exposes them either.
        //
        // Once per pipeline assembly, not per token, so it is affordable — the
        // same cost argument the router-choice line already makes.
        //
        // The cost is DECOMPOSED, not just totalled. Prefill and decode are
        // priced separately and scale with different things, so a single number
        // cannot say why one candidate beat another — and "the winner flipped"
        // is the only assertion that proves prompt-length routing fired at all.
        for c in &candidates {
            let whole = parallax::vertex_cost(
                c,
                (0, manifest.num_layers),
                local_node_id,
                manifest.num_layers,
                prompt_tokens,
            );
            tracing::info!(
                node = %c.node_id,
                ranges = ?c.available_ranges,
                can_be_first = c.can_be_first,
                can_be_last = c.can_be_last,
                region_score = c.region_score,
                latency_ms = c.latency_ms,
                est_tokens_per_sec = c.est_tokens_per_sec,
                observed_ms_per_layer = ?c.observed_latency_ms_per_layer,
                observed_delegated_ms_per_layer = ?c.observed_delegated_ms_per_layer,
                observed_prefill_ms_per_layer_byte = ?c.observed_prefill_ms_per_layer_byte,
                has_gpu = c.has_gpu,
                max_hostable_layers = ?c.max_hostable_layers,
                expected_attempts = c.expected_attempts,
                load = c.load,
                prompt_tokens = ?prompt_tokens,
                // Priced over the WHOLE model, so candidates are comparable to
                // each other even when they offer different ranges.
                cost_prefill_ms = whole.prefill_ms,
                cost_compute_ms = whole.compute_ms,
                cost_network_ms = whole.network_ms,
                cost_total_ms = whole.total(),
                "DIAG: pipeline candidate"
            );
        }

        // Sort: pool members first (free + trusted), then reachability tier,
        // then latency ASC, region DESC, load ASC, trust DESC, speed DESC.
        //
        // The reachability tier sits above latency so a directly-connected
        // holder always outranks a relayed one, and a measured peer always
        // outranks one we have merely assumed a latency for. Latency alone
        // could not express either guarantee.
        candidates.sort_by(|a, b| {
            b.is_pool_member
                .cmp(&a.is_pool_member) // true (1) > false (0) → pool members first
                .then_with(|| a.reach.cmp(&b.reach))
                .then_with(|| a.latency_ms.cmp(&b.latency_ms))
                .then_with(|| {
                    b.region_score
                        .partial_cmp(&a.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    a.load
                        .partial_cmp(&b.load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.trust_score
                        .partial_cmp(&a.trust_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    // Speed tie-breaker: faster nodes (higher tokens/s) preferred
                    b.est_tokens_per_sec
                        .partial_cmp(&a.est_tokens_per_sec)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        tracing::debug!(
            candidates_count = candidates.len(),
            model = %manifest.id,
            latency_range = ?candidates.first().map(|c| c.latency_ms)
                ..=candidates.last().map(|c| c.latency_ms),
            "DIAG: gather_candidates complete"
        );

        candidates
    }

    /// Compute region proximity score for a remote node relative to us.
    /// Returns 1.0 same, 0.5 adjacent, 0.2 distant, 0.7 unknown.
    fn compute_region_score(&self, node_id: &NodeId, _local_node_id: &NodeId) -> f32 {
        // Get our region — canonical resolver (configured wins, else detected),
        // shared with auto-manage so the two never disagree on a node's region.
        let our_region = self
            .shared_state
            .effective_region_sync()
            .map(|r| r.to_uppercase());
        let our_region = match our_region {
            Some(r) => r,
            None => return 0.7, // Our region unknown
        };

        // Get the candidate node's region
        let peer_region = self.shared_state.peer_registry.get(node_id).and_then(|p| {
            p.capability
                .as_ref()
                .and_then(|c| c.region.as_ref().map(|r| r.to_uppercase()))
        });

        match peer_region {
            Some(ref r) if *r == our_region => 1.0,
            Some(ref r) if regions_adjacent(&our_region, r) => 0.5,
            Some(_) => 0.2,
            None => 0.7, // Unknown region — treat as neutral
        }
    }

    /// Get reachability tier, latency and trust for a peer.
    ///
    /// The tier is the primary ranking key and encodes both how we reach the
    /// peer and whether its latency is measured — see [`ReachTier`]. The
    /// latency is a cost input only; it never has to carry the direct-beats-
    /// relayed guarantee on its own, which is what used to break.
    fn get_peer_metrics(&self, node_id: &NodeId, local_node_id: &NodeId) -> (ReachTier, u32, f32) {
        if node_id == local_node_id {
            return (ReachTier::Local, 0, 1.0);
        }

        let entry = self.shared_state.peer_registry.get(node_id);
        // What forwarding to this peer has actually cost, when we have sent
        // it any; the health ping is the fallback for a peer we have not.
        // The ping is a request_response round trip too, but taken when the
        // peer is idle — it cannot see the queueing a loaded event loop adds
        // to every forward (gotcha #386), which is the figure routing prices.
        let measured = entry.as_ref().and_then(|p| p.ack_srtt_ms.or(p.latency_ms));
        let trust = entry.as_ref().map(|p| p.trust_score).unwrap_or(0.3);
        let direct = self.shared_state.connected_node_ids.contains(node_id);

        let tier = match (direct, measured.is_some()) {
            (true, true) => ReachTier::DirectMeasured,
            (true, false) => ReachTier::DirectUnmeasured,
            (false, true) => ReachTier::RelayedMeasured,
            (false, false) => ReachTier::RelayedUnmeasured,
        };

        let base = match (measured, entry.is_some()) {
            (Some(ms), _) => ms,
            (None, true) => UNMEASURED_PEER_LATENCY_MS,
            (None, false) => UNKNOWN_PEER_LATENCY_MS,
        };
        // A relayed forward is us → relay → target, so it really does cost an
        // extra hop. Charging it keeps the cost arithmetic honest within the
        // relayed tier; the tier itself is what orders relayed against direct.
        let latency = if direct {
            base
        } else {
            base.saturating_add(RELAY_HOP_LATENCY_PENALTY_MS)
        };
        (tier, latency, trust)
    }

    /// Estimated milliseconds *per layer covered* of handing `range` to
    /// `candidate`, starting at `current_layer`. Lower is better.
    ///
    /// Three quantities have to be traded off, and ranking them one after
    /// another gets it wrong:
    ///
    /// - **Network.** One round trip is paid per segment per token, no matter
    ///   how many layers the segment covers. Dividing it by the coverage is
    ///   what lets a wide segment amortise a distant peer — and stops a narrow
    ///   one from pretending it is cheap.
    /// - **Compute.** The peer's measured per-layer cost where we have one,
    ///   otherwise its advertised throughput, otherwise a neutral default.
    /// - **Load.** Scales the compute term rather than being a separate,
    ///   higher-priority key. `load` counts requests in flight, so a peer
    ///   already serving one is treated as roughly twice as expensive — a real
    ///   penalty, but one a 100x latency difference can still outweigh.
    fn estimated_cost_per_layer(
        candidate: &NodeCandidate,
        range: (u32, u32),
        current_layer: u32,
    ) -> f32 {
        let covered = range.1.saturating_sub(current_layer).max(1) as f32;

        let compute_per_layer = candidate
            .observed_latency_ms_per_layer
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or_else(|| {
                // No measurement. Fall back to the advertised capability, which
                // is quoted for a ~32-layer 7B model, then to a neutral figure
                // so an unrated peer is neither favoured nor disqualified.
                if candidate.est_tokens_per_sec > 0.0 {
                    1000.0 / (candidate.est_tokens_per_sec * 32.0)
                } else {
                    DEFAULT_COMPUTE_MS_PER_LAYER
                }
            });

        let load_multiplier = 1.0 + candidate.load.max(0.0);
        candidate.latency_ms as f32 / covered + compute_per_layer * load_multiplier
    }

    /// Greedy layer assignment: cover all layers 0..num_layers using sorted candidates.
    ///
    /// Starting from layer 0, find the best candidate that covers at least
    /// the current layer, preferring those that cover the widest contiguous range.
    /// A single node may appear multiple times in the pipeline if it has
    /// non-contiguous layer ranges (e.g., layers [0,2) and [10,14)).
    ///
    /// Constraints:
    /// - The first segment (layer 0) must be assigned to a node with `can_be_first`
    ///   (has shard 0 for token_embd.weight)
    /// - The last segment (ending at num_layers) must be assigned to a node with
    ///   `can_be_last` (has the final shard for output.weight)
    /// - When `encrypted_pipeline` is true, both first AND last segments must be
    ///   the local (requesting) node — ensures no remote node sees plaintext.
    ///
    /// Each node's memory bound shapes the assignment where it can, and is
    /// dropped rather than refusing the request when it cannot.
    ///
    /// **Trying beats refusing here, and that is `parallax_assign`'s judgement
    /// too** — it logs "no route fits the peers' advertised memory — routing
    /// without that bound" and proceeds. A model whose holders' advertised
    /// memory admits no complete route is exactly the case a swarm exists for;
    /// answering slowly, or even failing on one segment, beats declining to
    /// try. So the bound shapes the assignment when it can and is dropped when
    /// it cannot, which is the difference between a preference and a gate.
    fn greedy_assign(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
        encrypted_pipeline: bool,
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        match self.greedy_assign_inner(num_layers, candidates, encrypted_pipeline, true) {
            Ok(segments) => Ok(segments),
            Err(capped_err) => {
                // Only worth a second pass if a cap could have been what
                // stopped it; with no caps in play the two runs are identical.
                if !candidates.iter().any(|c| c.max_hostable_layers.is_some()) {
                    return Err(capped_err);
                }
                tracing::info!(
                    num_layers,
                    "DIAG: no greedy route fits the peers' advertised memory — \
                     routing without that bound, a holder may be overcommitted"
                );
                self.greedy_assign_inner(num_layers, candidates, encrypted_pipeline, false)
            }
        }
    }

    fn greedy_assign_inner(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
        encrypted_pipeline: bool,
        respect_capacity: bool,
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        let mut segments = Vec::new();
        let mut current_layer = 0u32;
        let local_node_id = self.shared_state.identity.node_id();
        // **A memory bound is per NODE, not per segment**, and this loop can
        // return to a candidate it has already used — node A for 0..10, B for
        // 10..15, A again for 15..20. The cap below was applied fresh each
        // time, so a node capped at 36 could be handed 35 + 1 (gotcha #485).
        //
        // That is exactly the shape the #452 field report showed, and the DP
        // has guarded the same invariant since it was written ("a capped
        // candidate may appear at most ONCE"). This is the fallback beneath it,
        // which runs precisely when the model is awkward enough for the DP to
        // give up — so the guard was missing exactly where it was needed, the
        // same way the per-segment cap itself was until 2026-08-31.
        //
        // Decide-time accounting, the shape `find_standbys` and
        // `peer_vram_commitments` already use: what a plan has COMMITTED counts
        // against what the next segment may ask for.
        let mut assigned: std::collections::HashMap<NodeId, u32> = std::collections::HashMap::new();

        while current_layer < num_layers {
            let is_first_segment = current_layer == 0;

            // Find all (candidate, range) pairs that cover current_layer
            let mut options: Vec<(&NodeCandidate, (u32, u32))> = candidates
                .iter()
                .flat_map(|c| {
                    c.available_ranges
                        .iter()
                        .filter(|r| r.0 <= current_layer && r.1 > current_layer)
                        .map(move |r| (c, *r))
                })
                .collect();

            // Encrypted pipeline: first segment MUST be the local (requesting) node
            // so that token embedding happens locally (no remote sees raw tokens).
            if is_first_segment && encrypted_pipeline {
                let local_only: Vec<_> = options
                    .iter()
                    .filter(|(c, _)| c.node_id == *local_node_id && c.can_be_first)
                    .cloned()
                    .collect();
                if local_only.is_empty() {
                    // Fires when prompt privacy is on for a model whose first
                    // shard this node does not have — commonly because the
                    // setting was enabled while the shards were present and
                    // outlived them.
                    //
                    // This is its OWN error rather than a `PipelineError` for a
                    // reason: as the latter it was answered 500 "server_error"
                    // with the generic pipeline hint, which told the user a peer
                    // had gone offline and to try again. Nothing about retrying
                    // can help here — the setting and the shards on disk
                    // disagree until one of them changes.
                    let model = candidates
                        .first()
                        .map(|c| c.shard_id.model_id.0.as_str())
                        .unwrap_or("this model");
                    return Err(SwarmError::PromptPrivacyUnavailable {
                        model_id: model.to_string(),
                    });
                }
                options = local_only;
            }
            // First segment must be assigned to a node that can serve as first
            else if is_first_segment {
                let first_capable: Vec<_> = options
                    .iter()
                    .filter(|(c, _)| c.can_be_first)
                    .cloned()
                    .collect();
                if !first_capable.is_empty() {
                    options = first_capable;
                }
                // If no can_be_first candidates, fall through (best-effort)
            }

            // Prefer a node that still has room, before any other preference:
            // one already given everything it can hold should not be picked
            // again while another candidate could take this layer. Only a
            // preference — if NOTHING has room left, the options stand and the
            // segment goes somewhere rather than the request being refused,
            // which is what the relaxed pass and the holder's own admission
            // check are for.
            if respect_capacity {
                let with_room: Vec<_> = options
                    .iter()
                    .filter(|(c, _)| match c.max_hostable_layers {
                        Some(cap) => {
                            cap.saturating_sub(assigned.get(&c.node_id).copied().unwrap_or(0)) > 0
                        }
                        // Unknown capacity never excludes — the standing
                        // contract of `max_hostable_layers`.
                        None => true,
                    })
                    .cloned()
                    .collect();
                if !with_room.is_empty() {
                    options = with_room;
                }
            }

            // If this range could reach the end, prefer nodes that can be last.
            // But ALWAYS keep the local node as an option — distributed inference
            // should use locally-hosted shards first, forwarding the remainder.
            let any_reaches_end = options.iter().any(|(_, r)| r.1 >= num_layers);
            if any_reaches_end {
                // Encrypted pipeline: last segment MUST be the local node
                // so that token sampling happens locally (no remote sees output).
                if encrypted_pipeline {
                    let local_last: Vec<_> = options
                        .iter()
                        .filter(|(c, r)| {
                            c.node_id == *local_node_id && r.1 >= num_layers && c.can_be_last
                        })
                        .cloned()
                        .collect();
                    if !local_last.is_empty() {
                        options = local_last;
                    } else {
                        // Local node can't finish from this layer, but may have a later
                        // range that reaches the end (A→B→A bounce-back).
                        // Check if the local node has ANY range that finishes the model.
                        let local_can_finish_later = candidates.iter().any(|c| {
                            c.node_id == *local_node_id
                                && c.can_be_last
                                && c.available_ranges.iter().any(|r| r.1 >= num_layers)
                        });
                        if local_can_finish_later {
                            // Find where the local node's finishing range starts, and cap
                            // remote nodes to stop before that so A can take over.
                            let local_finish_start = candidates
                                .iter()
                                .filter(|c| c.node_id == *local_node_id)
                                .flat_map(|c| c.available_ranges.iter())
                                .filter(|r| r.1 >= num_layers)
                                .map(|r| r.0)
                                .min()
                                .unwrap_or(num_layers);
                            // Cap all remote options to end before the local finishing range
                            let capped: Vec<_> = options
                                .iter()
                                .map(|(c, r)| {
                                    if c.node_id != *local_node_id && r.1 > local_finish_start {
                                        (*c, (r.0, local_finish_start))
                                    } else {
                                        (*c, *r)
                                    }
                                })
                                .filter(|(_, r)| r.1 > r.0) // drop zero-width ranges
                                .collect();
                            if !capped.is_empty() {
                                options = capped;
                            } else {
                                return Err(SwarmError::PipelineError(
                                    "Encrypted pipeline requires the requesting node to hold \
                                     the final shard (output head). Download the last shard \
                                     to enable this mode."
                                        .to_string(),
                                ));
                            }
                        } else {
                            // Local node truly can't finish — no range reaches the end
                            let not_reaching_end: Vec<_> = options
                                .iter()
                                .filter(|(_, r)| r.1 < num_layers)
                                .cloned()
                                .collect();
                            if !not_reaching_end.is_empty() {
                                options = not_reaching_end;
                            } else {
                                return Err(SwarmError::PipelineError(
                                    "Encrypted pipeline requires the requesting node to hold \
                                     the final shard (output head). Download the last shard \
                                     to enable this mode."
                                        .to_string(),
                                ));
                            }
                        }
                    }
                } else {
                    let last_capable: Vec<_> = options
                        .iter()
                        .filter(|(c, r)| {
                            (r.1 >= num_layers && c.can_be_last) || c.node_id == *local_node_id
                        })
                        .cloned()
                        .collect();
                    if !last_capable.is_empty() {
                        options = last_capable;
                    }
                    // If no can_be_last candidates reach the end, let others that DON'T
                    // reach the end take over so a can_be_last node can finish later
                    else {
                        let not_reaching_end: Vec<_> = options
                            .iter()
                            .filter(|(c, r)| r.1 < num_layers || c.node_id == *local_node_id)
                            .cloned()
                            .collect();
                        if !not_reaching_end.is_empty() {
                            options = not_reaching_end;
                        }
                    }
                }
            }

            // Pick the best candidate: local first, then reachability tier,
            // then lowest estimated cost per layer covered.
            //
            // This replaced a lexicographic chain of local → coverage → load →
            // latency. Because `load` is a whole-request integer, it changed
            // more often than it tied, so latency was effectively never
            // reached: ONE in-flight request on a 4 ms LAN peer was enough to
            // hand the segment to a peer 100x further away. Coverage and
            // latency are not comparable quantities and cannot be ranked one
            // after the other — they have to be priced against each other,
            // which is what `estimated_cost_per_layer` does.
            let best = options
                .into_iter()
                .map(|(c, r)| {
                    let key = (
                        // Local always wins: its shards are already here and
                        // there is no network hop to price.
                        c.node_id != *local_node_id,
                        c.reach,
                        Self::estimated_cost_per_layer(c, r, current_layer),
                        // Only reached on a genuine tie (e.g. two local ranges,
                        // which have identical zero-network cost). Wider is
                        // better, so negate for a min-comparison.
                        -((r.1 - current_layer) as i64),
                    );
                    (key, c, r)
                })
                .min_by(|(ka, _, _), (kb, _, _)| {
                    ka.0.cmp(&kb.0)
                        .then_with(|| ka.1.cmp(&kb.1))
                        .then_with(|| ka.2.partial_cmp(&kb.2).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| ka.3.cmp(&kb.3))
                })
                .map(|(_, c, r)| (c, r));

            match best {
                Some((candidate, range)) => {
                    // **Cap the span at what this node can actually hold.**
                    // Taking `range.1` outright asks a node for every layer it
                    // HAS, which is a different question from how many it can
                    // fit in memory at once — and for a model that fits nobody,
                    // one node declaring the whole range means one node being
                    // handed the whole model.
                    //
                    // Measured 2026-08-31 on the live swarm: a 6 GB card was
                    // assigned all 48 layers of an 8,571 MB model as a single
                    // segment and failed at segment 0, roughly one request in
                    // four. `parallax_assign` has honoured this cap since it
                    // was introduced; the greedy fallback beneath it never did,
                    // and the fallback runs precisely when the model is awkward
                    // enough for parallax to give up — so the guard was absent
                    // exactly where it was needed. Same shape as the prefill
                    // budget in `.claude/rules/architecture.md`: enforced on the
                    // sophisticated path, missing from the crude one beneath it.
                    //
                    // `None` means UNKNOWN and must never exclude — an
                    // unreadable capability is not evidence a node is small
                    // (see [`max_hostable_layers`]). At least one layer always
                    // moves, or the loop cannot terminate.
                    let layer_end = match candidate.max_hostable_layers {
                        Some(cap) if respect_capacity => {
                            // What this node may still take, after everything
                            // this plan has already given it.
                            let already = assigned.get(&candidate.node_id).copied().unwrap_or(0);
                            let remaining = cap.saturating_sub(already);
                            if remaining == 0 {
                                // Nothing left, and the filter above already
                                // preferred anyone who had room — so no node
                                // can take this layer within its bound.
                                //
                                // FAIL rather than hand it over anyway. Giving
                                // it a layer regardless would not enforce the
                                // bound, it would only FRAGMENT the overage
                                // into one-layer segments and still exceed it,
                                // which is worse than both alternatives. The
                                // caller answers this by re-running without the
                                // bound (`greedy_assign`), which is the same
                                // constrained-then-relaxed shape the DP uses,
                                // and the holder's own admission check is the
                                // backstop after that.
                                return Err(SwarmError::PipelineError(format!(
                                    "greedy: no node can take layer {current_layer} \
                                     within the memory it advertises"
                                )));
                            }
                            range.1.min(current_layer.saturating_add(remaining))
                        }
                        _ => range.1,
                    };
                    segments.push(PipelineSegment {
                        node_id: candidate.node_id.clone(),
                        shard_id: candidate.shard_id.clone(),
                        layer_range: (current_layer, layer_end),
                    });
                    *assigned.entry(candidate.node_id.clone()).or_insert(0) +=
                        layer_end.saturating_sub(current_layer);
                    current_layer = layer_end;
                }
                None => {
                    // Name the model and say what is missing. "No node available
                    // for layer 0" was reported (2026-08-10) against a model the
                    // node's own status called "loaded", which reads as a
                    // contradiction: loaded describes what was started, this
                    // describes who can serve the piece containing that layer
                    // right now, and nobody currently can.
                    let model = candidates
                        .first()
                        .map(|c| c.shard_id.model_id.0.as_str())
                        .unwrap_or("this model");
                    return Err(SwarmError::ModelIncompleteInSwarm {
                        model_id: model.to_string(),
                        layer: current_layer,
                    });
                }
            }
        }

        Ok(segments)
    }

    /// Merge contiguous segments assigned to the same node into one segment.
    /// The assignment for a node that runs the whole model itself: one local
    /// segment and nobody else involved.
    ///
    /// No TP groups here — deliberately. When the local node already holds
    /// every layer, pulling a LAN peer into a tensor-parallel group can only
    /// make the request slower (2 × num_layers AllReduce round trips replacing
    /// compute we were about to do anyway) and adds a hard dependency on a peer
    /// we did not need. A peer that stalls then fails the whole request with an
    /// AllReduce timeout, even though this node could have answered alone. No
    /// standby for the same reason: there is no remote segment to fail over.
    fn local_only_assignment(
        request_id: uuid::Uuid,
        local_node_id: &NodeId,
        local_cand: &NodeCandidate,
        num_layers: u32,
    ) -> PipelineAssignment {
        PipelineAssignment {
            request_id,
            segments: vec![PipelineSegment {
                node_id: local_node_id.clone(),
                shard_id: local_cand.shard_id.clone(),
                layer_range: (0, num_layers),
            }],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: true,
        }
    }

    fn merge_contiguous(segments: Vec<PipelineSegment>) -> Vec<PipelineSegment> {
        let mut merged: Vec<PipelineSegment> = Vec::new();
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if last.node_id == seg.node_id && last.layer_range.1 == seg.layer_range.0 {
                    // Extend the previous segment
                    last.layer_range.1 = seg.layer_range.1;
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }

    /// Detect tensor-parallel opportunities among LAN peers.
    ///
    /// For each pipeline segment, check if there are additional LAN peers that
    /// could serve the same layer range. If so, form a TensorParallelGroup
    /// containing the primary node plus LAN peers (up to 4 nodes per group).
    ///
    /// Tensor parallelism is only beneficial on LAN (<5ms latency) because the
    /// AllReduce communication between layers requires low latency.
    fn detect_tp_groups(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
    ) -> Vec<TensorParallelGroup> {
        let local_id = self.shared_state.identity.node_id().clone();

        let mut groups = Vec::new();

        for segment in segments {
            // Only form TP groups for segments assigned to us (local node)
            if segment.node_id != local_id {
                continue;
            }

            // Find LAN peers that can serve the same layer range
            let mut tp_nodes = vec![local_id.clone()];

            for candidate in candidates {
                if candidate.node_id == local_id {
                    continue;
                }
                if tp_nodes.len() >= MAX_TP_GROUP_SIZE {
                    break;
                }

                // Must cover the same layer range
                let covers = candidate
                    .available_ranges
                    .iter()
                    .any(|r| r.0 <= segment.layer_range.0 && r.1 >= segment.layer_range.1);
                if !covers {
                    continue;
                }

                // Must be a LAN peer with low latency for AllReduce.
                // Accept peers that are either mDNS-discovered (is_lan_peer) or
                // have measured RTT ≤ 10ms (auto-detected via rr_ping).
                let (is_lan, measured_latency) = self
                    .shared_state
                    .peer_registry
                    .get(&candidate.node_id)
                    .map(|p| (p.is_lan_peer, p.latency_ms))
                    .unwrap_or((false, None));
                let tp_max_ms = self.shared_state.config.inference.tp_max_latency_ms;
                let low_latency = measured_latency.is_some_and(|ms| ms <= tp_max_ms);
                if !is_lan && !low_latency {
                    continue;
                }

                tp_nodes.push(candidate.node_id.clone());
            }

            // Need at least 2 nodes for tensor parallelism
            if tp_nodes.len() >= 2 {
                let shard_ids: Vec<_> = candidates
                    .iter()
                    .filter(|c| tp_nodes.contains(&c.node_id))
                    .map(|c| c.shard_id.clone())
                    .collect();

                tracing::info!(
                    layers = ?(segment.layer_range.0..segment.layer_range.1),
                    tp_size = tp_nodes.len(),
                    nodes = ?tp_nodes.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
                    "Formed tensor-parallel group"
                );

                groups.push(TensorParallelGroup {
                    nodes: tp_nodes,
                    layer_range: segment.layer_range,
                    shard_ids,
                });
            }
        }

        groups
    }

    /// Produce a Parallax-style offline allocation recommendation for the
    /// given model. Snapshots the current peer registry + local node, derives
    /// per-peer layer capacity from known VRAM or capability signals, and
    /// runs the greedy multi-pipeline packer.
    ///
    /// Returns `None` when the manifest is missing or the cluster can't cover
    /// the model's layer count even once. Callers should treat the result as
    /// advisory — in v1 this is not auto-applied to `ShardRebalancer`.
    pub fn allocate_offline(
        &self,
        model_id: &crate::types::ModelId,
        max_pipelines: u32,
    ) -> Option<parallax_allocator::AllocationPlan> {
        let manifest = self.shared_state.model_registry.get_manifest(model_id)?;
        let num_layers = manifest.num_layers;
        if num_layers == 0 {
            return None;
        }
        let local_node_id = self.shared_state.identity.node_id();
        let bytes_per_layer = manifest.total_size_bytes / manifest.num_layers.max(1) as u64;

        let mut peers: Vec<parallax_allocator::PeerCapacity> = Vec::new();
        // Local node. If we have on-disk shards for this model, treat our
        // capacity as the union of their layer ranges; otherwise assume no
        // local capacity (Phase C won't recommend putting layers here).
        // Same derivation every peer uses for itself; without it a
        // processor-only node offered the allocator a zero, which it documents
        // as "unknown" and replaces with an average.
        let local_tps = crate::model::auto_manage::vram::node_tokens_per_sec_7b(
            self.shared_state.gpu_info.as_ref().map(|g| g.name.as_str()),
        )
        .unwrap_or(0.0);
        let local_layer_capacity = manifest_layer_capacity_for_local(&manifest, &self.shared_state);
        peers.push(parallax_allocator::PeerCapacity {
            node_id: local_node_id.clone(),
            layer_capacity: local_layer_capacity,
            tokens_per_sec: local_tps,
            latency_ms: 0,
        });

        for entry in self.shared_state.peer_registry.iter() {
            let peer = entry.value();
            let node_id = peer.node_id.clone();
            if &node_id == local_node_id {
                continue;
            }
            // Scheduler Liveness Oracle: peer_registry retains disconnected
            // peers (for reconnect attempts); allocate against currently
            // connected nodes only.
            //
            // Deliberately STRICTER than `gather_candidates`, which also admits
            // relay-reachable holders (NETWORKING_PLAN §4 Phase 1 tier). The two
            // answer different questions: this plans a capacity allocation we
            // intend to hold, and a plan built around an extra relay hop per
            // layer is a bad plan — whereas routing a single request through a
            // relay to the only holder of a shard is strictly better than
            // failing. So a relay-only peer is usable on demand but is not
            // allocated pipeline capacity.
            if !self.shared_state.connected_node_ids.contains(&node_id) {
                continue;
            }
            // Prefer VRAM when the peer has a GPU, else fall back to RAM —
            // the worker subprocess can host layers in either.
            let available_mb = peer
                .capability
                .as_ref()
                .map(|c| match &c.gpu {
                    Some(g) => g.vram_available_mb,
                    None => c.ram_available_mb,
                })
                .unwrap_or(0);
            let available_bytes = available_mb.saturating_mul(1_048_576);
            let layer_capacity = available_bytes.checked_div(bytes_per_layer).unwrap_or(0) as u32;
            let tps = peer
                .capability
                .as_ref()
                .map(|c| c.est_tokens_per_sec_7b)
                .unwrap_or(0.0);
            let latency_ms = peer.latency_ms.unwrap_or(200);
            peers.push(parallax_allocator::PeerCapacity {
                node_id,
                layer_capacity,
                tokens_per_sec: tps,
                latency_ms,
            });
        }

        parallax_allocator::recommend_allocation(&peers, num_layers, max_pipelines)
    }

    /// Find standby (backup) nodes for each pipeline segment.
    fn find_standbys(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
        prompt_tokens: Option<u32>,
        num_layers: u32,
    ) -> Vec<PipelineSegment> {
        let mut standbys = Vec::new();

        let local_node_id = self.shared_state.identity.node_id();

        // What this plan has already asked of each node. Seeded with primary
        // duty, then grown as standbys are chosen, so the fourth segment's
        // standby search sees what the first three already committed. See
        // `standby_has_room`.
        let mut committed = primary_layer_commitments(segments);

        for segment in segments {
            let segment_layers = segment.layer_range.1.saturating_sub(segment.layer_range.0);
            // Collect all eligible standbys, then pick the local node first.
            // Preferring local prevents "no standby available" when a remote
            // primary returns an inference error — the local node can always
            // execute the segment if it has full coverage.
            let mut eligible: Vec<&NodeCandidate> = candidates
                .iter()
                .filter(|c| {
                    c.node_id != segment.node_id
                        && c.available_ranges
                            .iter()
                            .any(|r| r.0 <= segment.layer_range.0 && r.1 >= segment.layer_range.1)
                        // Holding the range is not being able to run it.
                        && standby_has_room(
                            c.max_hostable_layers,
                            committed.get(&c.node_id).copied().unwrap_or(0),
                            segment_layers,
                        )
                })
                .collect();
            // For standby anti-affinity: prefer DIFFERENT regions from primary
            // so a regional outage doesn't kill both primary and standby.
            // Local node first (most reliable), then different-region, then by latency.
            let primary_region_score = candidates
                .iter()
                .find(|c| c.node_id == segment.node_id)
                .map(|c| c.region_score)
                .unwrap_or(0.7);
            eligible.sort_by(|a, b| {
                let la = u32::from(a.node_id != *local_node_id);
                let lb = u32::from(b.node_id != *local_node_id);
                // Anti-affinity: if primary is same-region (1.0), prefer standbys
                // with lower region_score (different region). If primary is distant,
                // prefer same-region standbys for faster failover.
                let region_cmp = if primary_region_score > 0.8 {
                    // Primary is same-region — prefer different-region standbys
                    a.region_score
                        .partial_cmp(&b.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    // Primary is distant — prefer same-region standbys
                    b.region_score
                        .partial_cmp(&a.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                };
                // Last tiebreak used to be raw `latency_ms`, i.e. whichever
                // node answered a health ping fastest. That is the wrong
                // question once the primary was chosen by a cost model: the
                // request that failed over is the SAME request, so the standby
                // has to be priced the same way, prompt length included.
                //
                // Ranking on ping alone routinely picked the slowest machine in
                // the swarm — nearness and speed are unrelated here, and the
                // measured spread is 0.37 to 20.45 tok/s. A long prompt
                // correctly steered to a graphics card would fail over to a
                // processor-only box and take minutes instead of seconds.
                //
                // Local-first and region anti-affinity still come first: a
                // standby is about surviving a failure, and those two answer
                // that. This only replaces the tiebreak among equals.
                let cost = |c: &NodeCandidate| {
                    parallax::vertex_cost(
                        c,
                        segment.layer_range,
                        local_node_id,
                        num_layers,
                        prompt_tokens,
                    )
                    .total()
                };
                la.cmp(&lb).then(region_cmp).then_with(|| {
                    cost(a)
                        .partial_cmp(&cost(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
            if let Some(backup) = eligible.first() {
                // Decide-time accounting: this node is now on the hook for
                // these layers as far as the rest of this plan is concerned.
                *committed.entry(backup.node_id.clone()).or_insert(0) += segment_layers;
                standbys.push(PipelineSegment {
                    node_id: backup.node_id.clone(),
                    shard_id: backup.shard_id.clone(),
                    layer_range: segment.layer_range,
                });
            }
        }

        tracing::debug!(
            segment_count = segments.len(),
            standby_count = standbys.len(),
            segments = ?segments.iter().map(|s| format!("{}:{}-{}", s.node_id, s.layer_range.0, s.layer_range.1)).collect::<Vec<_>>(),
            "DIAG: find_standbys complete"
        );

        standbys
    }
}

mod parallax;
pub mod parallax_allocator;

/// Estimate how many layers the local node can reasonably host for `manifest`.
/// Uses the shards already on disk for this model as the primary signal — the
/// union of their GGUF tensor layer ranges — falling back to 0 when the node
/// holds none. This keeps Phase C's recommendations aligned with what the
/// local node is ACTUALLY ready to serve, rather than an aspirational VRAM
/// estimate.
fn manifest_layer_capacity_for_local(manifest: &ModelManifest, shared_state: &SharedState) -> u32 {
    let local_node = shared_state.identity.node_id();
    let mut shard_indices: Vec<u32> = Vec::new();
    for shard in &manifest.shards {
        let shard_id = ShardId {
            model_id: manifest.id.clone(),
            index: shard.index,
        };
        if shared_state
            .model_registry
            .shard_holders(&shard_id)
            .iter()
            .any(|n| n == local_node)
        {
            shard_indices.push(shard.index);
        }
    }
    let ranges =
        crate::inference::split::available_layer_ranges_from_manifest(manifest, &shard_indices);
    ranges
        .iter()
        .map(|(s, e)| (e - s) as u32)
        .sum::<u32>()
        .min(manifest.num_layers)
}

#[cfg(test)]
mod tests;
