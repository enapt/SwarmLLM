//! Parallax-inspired shortest-path routing over (node, layer_range) vertices.
//!
//! Adapted from Parallax Phase 2 (arxiv 2509.26182 / GradientHQ/parallax MIT).
//! Parallax was designed for peer-to-peer pipeline data flow where transition
//! edge cost is `rtt(peer_A, peer_B)`. SwarmLLM pipelines are coordinator-relayed:
//! every hop routes through the local node. So edge weight collapses into a
//! per-vertex cost: `2 * rtt_local_to_peer + compute_time + load_compensator`
//! (local node: just compute_time).
//!
//! Two departures from the paper, both because it targets GPU clusters where
//! compute dominates rather than heterogeneous nodes on home connections:
//! communication is charged **per token** for a segment entered mid-chain (the
//! paper sums it once per layer transition), and an unmeasured candidate gets a
//! non-zero compute prior (the paper does not discuss cold start at all).
//!
//! DAG: vertex = (candidate_idx, range_idx). Edge v → w iff
//! `ranges[v].end == ranges[w].start`. Sources have `start == 0` (and may have
//! first-segment constraints). Sinks have `end == num_layers` (and may have
//! last-segment constraints). DP finds the minimum-cost chain.
//!
//! Falls back to `greedy_assign` on any configuration the DP can't cover
//! (e.g., no valid source, no valid sink, disconnected layers).
//!
//! Gated by `InferenceConfig::parallax_routing` (default on). Set to `false`
//! to revert to pure greedy assembly — falls back automatically anyway when
//! the DP has no valid source→sink path.

use crate::error::SwarmError;
use crate::types::{NodeId, PipelineSegment};

use super::NodeCandidate;

/// Per-vertex cost components. All in milliseconds.
#[derive(Debug, Clone, Copy)]
pub(super) struct VertexCost {
    /// `2 * latency_ms` for remote peers, 0 for local — multiplied by
    /// `ASSUMED_FORWARD_PASSES` when the segment is entered mid-chain, since the
    /// coordinator round-trips into it per token.
    pub(super) network_ms: f32,
    /// Per-layer cost × layers × `ASSUMED_FORWARD_PASSES`.
    pub(super) compute_ms: f32,
    /// active_request_count * LOAD_COMPENSATOR_MS.
    pub(super) load_ms: f32,
    /// Reading the prompt: one pass over `prompt_tokens`, linear in them.
    /// Zero when the caller did not say how long the prompt is.
    pub(super) prefill_ms: f32,
}

impl VertexCost {
    pub(super) fn total(self) -> f32 {
        self.network_ms + self.compute_ms + self.load_ms + self.prefill_ms
    }
}

/// Penalty (ms) per active concurrent request on a candidate.
/// Parallax uses `load_compensator=0.05` as a fraction of baseline latency;
/// we use a fixed ms penalty so it's tunable independently of a baseline estimate.
const LOAD_COMPENSATOR_MS: f32 = 25.0;

/// Per-layer compute cost assumed for a candidate we have neither observed nor
/// received a throughput estimate for.
///
/// **This used to be 0, and zero is the one value guaranteed to be wrong.** It
/// made every unmeasured candidate free, so on a freshly started node — where
/// nothing has been observed yet — competing chains tied at their network term
/// alone and the winner was decided by vertex iteration order rather than by
/// merit. Routing quality silently depended on how long the node had been up.
///
/// A single shared constant does not discriminate between candidates, which is
/// the honest position when we know nothing about them: it makes cost scale
/// with the number of layers a candidate would take on, so wide segments on
/// unknown nodes are no longer free, while measured candidates still win or
/// lose on their real numbers. Roughly a mid-range CPU node's per-layer decode
/// cost, deliberately nearer the pessimistic end so an unmeasured node does not
/// outrank a measured good one.
pub(super) const UNKNOWN_COMPUTE_MS: f32 = 25.0;

/// Forward passes assumed per request when charging per-token costs.
///
/// A pipeline split across nodes exchanges activations **once per token**,
/// whereas a single remote segment covering the whole model is delegated in one
/// message and decodes remotely with no per-token network at all. Charging a
/// remote hop once per *segment* — as this model did — cannot see that
/// difference, and measured it wrong: on a LAN pair the split was chosen and
/// ran 11.2s → 17.8s (16-token replies), because the per-token round trips cost
/// more than the faster node saved.
///
/// A fixed estimate rather than the request's `max_tokens`, which is a loose
/// upper bound (commonly 2048 for a reply of 50) and would over-penalise
/// splitting. Order of magnitude is what matters: the term must be large enough
/// that a boundary is not free, and it scales with the same units as the rest of
/// the model.
pub(super) const ASSUMED_FORWARD_PASSES: f32 = 64.0;

/// How much faster a machine reads a prompt than it writes a reply, before we
/// have measured that machine doing it.
///
/// Prefill and decode are bounded by different things — prefill by arithmetic,
/// decode by memory bandwidth — so the ratio is a property of the DEVICE CLASS
/// and is the entire reason routing needs to know how long a prompt is. Both
/// figures are measured on the hardware this was written against,
/// llama-3.2-3b Q4_K:
///
/// | machine | prefill | decode | ratio |
/// |---|---|---|---|
/// | i5-10500T, no GPU | 36 tok/s | 8 tok/s | 4.5x |
/// | Ryzen 5800H, no GPU | 37 | 6.6 | 5.6x |
/// | RTX 3070 laptop | ~2000 | 47 | 42x |
///
/// They are a PRIOR and nothing more: the moment this node has actually
/// prefilled through a peer, `observed_prefill_ms_per_layer_byte` replaces them
/// (`peer_speed` has tracked prefill separately since 2026-08-01). They exist so
/// that a peer we have never measured is not priced as though reading a prompt
/// were free, which is what the model did before — and the error that produces
/// is much larger on prefill than the 6x decode spread the model already knew
/// about.
pub(super) const PREFILL_SPEEDUP_GPU: f32 = 40.0;
/// See [`PREFILL_SPEEDUP_GPU`]. Deliberately the low end of the two CPU
/// measurements: under-crediting a processor's prefill errs toward keeping a
/// long prompt off it, which is the safe direction.
pub(super) const PREFILL_SPEEDUP_CPU: f32 = 4.5;

/// Bytes of activation one token of prompt carries, at a typical hidden width.
/// Shared with `peer_speed` so the measured coefficient and the estimate below
/// are expressed in the same units.
pub(super) const ACTIVATION_BYTES_PER_TOKEN: usize =
    crate::daemon::state::peer_speed::NOMINAL_DECODE_ACTIVATION_BYTES;

/// Cap on how many partial-range vertices the DP will consider, summed across
/// candidates. Keeps vertex generation bounded when a popular model has many
/// holders; past it, only whole ranges are emitted.
pub(super) const MAX_SUBRANGE_VERTICES: usize = 4096;

/// Baseline transformer layer count used to scale a whole-model throughput
/// estimate down to a per-segment contribution. 32 matches Llama-7B and most
/// 7B Q4 models we benchmark against; arch-aware scaling would replace this
/// with the actual layer count from the GGUF metadata. Shared with
/// `parallax_allocator.rs` so the cost model stays consistent across the
/// scheduler and the offline allocator.
pub(super) const BASELINE_LAYER_COUNT: f32 = 32.0;

/// Compute per-vertex cost for a (candidate, range) pair.
///
/// Cost priority for `compute_ms`:
/// 1. Observed per-layer latency EMA if the candidate has any samples (Phase B).
///    This folds the remote peer's per-segment wall-clock (already includes both
///    compute and any peer-side queuing/load) into the DP objective, so the
///    `network_ms` term doesn't double-count load here.
/// 2. Static `est_tokens_per_sec` capability estimate when no observations exist.
/// 3. `UNKNOWN_COMPUTE_MS` when neither is available — deliberately non-zero,
///    because a free unmeasured candidate outranks every measured one.
pub(super) fn vertex_cost(
    c: &NodeCandidate,
    range: (u32, u32),
    local: &NodeId,
    num_layers: u32,
    // How long the prompt is, when the caller knows. `None` leaves the prefill
    // term at zero, which is exactly the cost model as it stood before prompt
    // length was threaded through — so an offline allocation or a test that
    // does not have a prompt prices requests exactly as it always did.
    prompt_tokens: Option<u32>,
) -> VertexCost {
    let is_local = &c.node_id == local;
    let layers = (range.1 - range.0) as f32;
    // The one shape that escapes per-token network: a remote peer covering the
    // WHOLE model. That is delegated in a single message and decodes remotely,
    // streaming tokens back, so the coordinator round-trips into it once for the
    // entire request.
    //
    // Every other remote segment is entered once per token, including one that
    // starts at layer 0. The coordinator drives each hop, so after sampling
    // token t it must hand the new token back to the first segment — a remote
    // 0..k of a split pays the round trip exactly as a mid-chain segment does.
    // This used to be keyed on `range.0 != 0`, which charged a remote first
    // segment its network ONCE and so quietly subsidised splitting: the routing
    // model could not see most of the cost of the boundary it was choosing.
    let covers_whole_model = range.0 == 0 && range.1 == num_layers;
    let delegated_whole_model = !is_local && covers_whole_model;
    let entered_per_token = !is_local && !covers_whole_model;

    // Pick the observation measured on the shape being priced, and never
    // substitute one for the other.
    //
    // A mid-chain decode sample carries the coordinator's round trip amortised
    // over whatever layers that segment owned; a delegated sample carries none.
    // Reusing the mid-chain figure for a delegated vertex therefore charges a
    // round trip several times over for a trip that vertex never makes —
    // measured at ~2.7x for a 16-layer delegation priced from a 6-layer
    // observation, and the reason the router kept choosing a split that ran
    // 11.2s where delegating ran 11.2s → 17.8s the other way.
    //
    // Where the delegated figure is missing, the mid-chain one is still used —
    // as an UPPER BOUND, which is what it honestly is. It contains everything
    // the delegated cost contains plus a round trip, so a peer it says is slow
    // really is slow, and the error is in the safe direction.
    //
    // Discarding it and falling through to the static estimate was tried and is
    // worse: a peer measured at 107 ms/layer would be re-priced at the
    // `UNKNOWN_COMPUTE_MS` prior of 25, so a candidate we have measured and
    // found slow would outrank one we know nothing about. That is the exact
    // failure that constant was raised from 0 to prevent, reintroduced through
    // a different door — four existing tests caught it.
    //
    // The overcharge is therefore bounded and self-correcting: it applies only
    // until this peer serves one delegated request, after which the
    // right-shaped figure takes over. `partial_ranges` is off by default, so a
    // single holder is routed the whole model and earns that sample on its
    // first request.
    let observation = if delegated_whole_model {
        c.observed_delegated_ms_per_layer
            .or(c.observed_latency_ms_per_layer)
    } else {
        c.observed_latency_ms_per_layer
    };

    // When we have an observed per-layer latency, it already includes the peer's
    // segment wall-clock round-trip (compute + peer-side load). Fold the whole
    // `segment_ms` into `compute_ms` and skip the static `2 * latency_ms` network
    // term to avoid double-counting. When we don't have an observation yet, use
    // the traditional two-part cost (network + static compute estimate).
    let (base_network_ms, per_layer_ms) = if let Some(obs_per_layer) = observation {
        (0.0, obs_per_layer)
    } else {
        let network = if is_local {
            0.0
        } else {
            2.0 * c.latency_ms as f32
        };
        let per_layer = if c.est_tokens_per_sec > 0.0 {
            // Very rough: layer_compute_ms ≈ layers / (est_tokens_per_sec * some_constant).
            // est_tokens_per_sec is a whole-model throughput estimate for a 7B Q4 model;
            // per-layer contribution is 1/num_layers of that. We conservatively use
            // `1000.0 / est_tokens_per_sec` as the per-token-whole-model compute cost,
            // then scale by the fraction of layers this segment owns. Assumes 32 layers
            // as the baseline; adjust here if we want arch-aware scaling later.
            let whole_model_ms = 1000.0 / c.est_tokens_per_sec;
            whole_model_ms / BASELINE_LAYER_COUNT
        } else {
            UNKNOWN_COMPUTE_MS
        };
        (network, per_layer)
    };

    // Compute is per forward pass, and a request is many passes. Scaling every
    // candidate by the same factor leaves their relative order untouched, but it
    // puts compute in the same units as the per-token network term below, so the
    // two can be compared at all.
    let compute_ms = per_layer_ms * layers * ASSUMED_FORWARD_PASSES;

    // The asymmetry the model was missing. Every remote segment short of the
    // whole model is round-tripped into for every token; only a delegated whole
    // model pays its network once, because it decodes remotely on its own.
    let network_ms = if entered_per_token {
        base_network_ms * ASSUMED_FORWARD_PASSES
    } else {
        base_network_ms
    };

    let load_ms = c.load * LOAD_COMPENSATOR_MS;

    // Reading the prompt. Nothing in this cost model used to price it at all:
    // every candidate was charged a fixed `ASSUMED_FORWARD_PASSES` of DECODE
    // and a ten-token request was routed identically to a ten-thousand-token
    // one. That is not a rounding error — prefill is linear in prompt length,
    // and the hardware spread on prefill (~55x between a graphics card and a
    // processor here) is an order of magnitude wider than the ~6x on decode
    // that the model did price. A 5000-token agentic system prompt is ~140s of
    // prefill on a CPU node against ~2.5s on a GPU one.
    //
    // Kept as a separate term rather than folded into `compute_ms` so the two
    // can be reported and reasoned about apart — they scale with different
    // things and a future measurement will want to check them separately.
    let prefill_ms = match prompt_tokens {
        None => 0.0,
        Some(0) => 0.0,
        Some(tokens) => {
            let per_layer_prefill_ms = match c.observed_prefill_ms_per_layer_byte {
                // Measured on this peer, for exactly this work. The coefficient
                // is normalised by layers x activation bytes, so it carries
                // across models.
                Some(coeff) => coeff * tokens as f32 * ACTIVATION_BYTES_PER_TOKEN as f32,
                // Never measured: derive from whatever prices its decode, using
                // the device-class prior. `per_layer_ms` is a per-TOKEN decode
                // cost, so a prompt of N tokens costs N of them, divided by how
                // much faster this class of machine reads than writes.
                None => {
                    let speedup = if c.has_gpu {
                        PREFILL_SPEEDUP_GPU
                    } else {
                        PREFILL_SPEEDUP_CPU
                    };
                    per_layer_ms * tokens as f32 / speedup
                }
            };
            per_layer_prefill_ms * layers
        }
    };

    // Scale the whole vertex by how many attempts it takes to get one intact
    // answer out of this peer.
    //
    // Not a tuning knob: if a fraction `p` of replies from a peer arrive whole,
    // then one whole reply costs `1/p` of everything — network, compute and
    // queueing alike, since a lost reply wastes all three. A reliable or
    // unmeasured path multiplies by 1 and nothing changes.
    //
    // This exists because the previous thing steering traffic away from a lossy
    // link was an accident: a truncated stream poisoned the peer's SPEED, which
    // was wrong about why (a network fault read as slow hardware) and right
    // about where not to send work. Removing the mis-attribution removed the
    // avoidance with it, so the avoidance is now stated deliberately, in the
    // one term that can express it honestly.
    let attempts = if c.expected_attempts.is_finite() && c.expected_attempts >= 1.0 {
        c.expected_attempts
    } else {
        1.0
    };
    VertexCost {
        network_ms: network_ms * attempts,
        compute_ms: compute_ms * attempts,
        load_ms: load_ms * attempts,
        prefill_ms: prefill_ms * attempts,
    }
}

/// What a whole chain is priced at: the sum of its segments' vertex costs, in
/// the same milliseconds `vertex_cost` speaks. A segment whose node is not
/// among `candidates` contributes nothing, which cannot happen for a chain the
/// DP produced from those candidates.
///
/// Exists so the caller can put the number it is about to act on in the log
/// beside the alternative it is giving up — the search itself returns only the
/// winning segments, and "the pipeline was cheaper" is unverifiable without
/// both figures.
pub(super) fn chain_cost_ms(
    segments: &[PipelineSegment],
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    prompt_tokens: Option<u32>,
) -> f32 {
    segments
        .iter()
        .filter_map(|seg| {
            candidates
                .iter()
                .find(|c| c.node_id == seg.node_id)
                .map(|c| {
                    vertex_cost(c, seg.layer_range, local_node_id, num_layers, prompt_tokens)
                        .total()
                })
        })
        .sum()
}

/// Route layers [0, num_layers) across candidates using shortest-path DP.
///
/// Returns segments covering [0, num_layers) in order, or the SwarmError from
/// an empty search space if the DAG has no source/sink path. Callers should
/// fall back to the greedy assigner on error.
pub(super) fn route_shortest_path(
    num_layers: u32,
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    encrypted_pipeline: bool,
    // Allow a candidate's range to be used in part. Off by default: see
    // `config.inference.parallax_partial_ranges` for the measured reason.
    partial_ranges: bool,
    // Hold each peer to the layer count its advertised free memory can take
    // (`NodeCandidate::max_hostable_layers`). The caller runs this ON first and
    // retries with it OFF if no route exists, so a self-reported figure can
    // never turn a routable request into a failure — see
    // `assemble_pipeline_for`.
    respect_capacity: bool,
    // Prompt length, when known. See `vertex_cost`; `None` reproduces the cost
    // model exactly as it stood before prefill was priced.
    prompt_tokens: Option<u32>,
) -> Result<Vec<PipelineSegment>, SwarmError> {
    if num_layers == 0 {
        return Err(SwarmError::PipelineError("num_layers=0".into()));
    }

    // Build the vertex list: one vertex per (candidate, range).
    // Also precompute each vertex's cost.
    #[derive(Clone)]
    struct Vertex {
        cand_idx: usize,
        range: (u32, u32),
        cost_ms: f32,
    }
    // Split points at which a segment boundary can usefully fall: the ends of
    // every candidate's ranges, plus the model's own ends. A sub-range is only
    // worth emitting if some other candidate could begin or end there.
    let mut split_points: Vec<u32> = vec![0, num_layers];
    for c in candidates {
        for &(a, b) in &c.available_ranges {
            if a < num_layers {
                split_points.push(a);
            }
            let b = b.min(num_layers);
            if b > 0 {
                split_points.push(b);
            }
        }
    }
    // A local candidate holding the WHOLE model offers the search no boundary
    // to cut at: its one range starts at 0 and ends at `num_layers`, and under
    // encryption it is the only legal source and sink, so every other chain
    // must begin and end on it — at a layer the points above do not contain.
    // Add the boomerang's own ends, one layer each: exactly the shape
    // `boomerang_assignment` builds by hand for a single peer, now available
    // to the priced search across several. Without them a node holding every
    // shard of a model it would run on its processor could never be routed as
    // a boomerang over its peers' cards, however much faster they were priced
    // (gotcha #444). Only when such a candidate exists, so every other
    // topology's split set — and route — is exactly what it was.
    if encrypted_pipeline && num_layers >= 3 {
        let local_holds_whole = candidates.iter().any(|c| {
            &c.node_id == local_node_id
                && c.available_ranges
                    .iter()
                    .any(|&(a, b)| a == 0 && b >= num_layers)
        });
        if local_holds_whole {
            split_points.push(1);
            split_points.push(num_layers - 1);
        }
    }
    split_points.sort_unstable();
    split_points.dedup();

    // Usable (clamped, non-empty) ranges, paired with the split points falling
    // inside each.
    let clamped: Vec<(usize, (u32, u32))> = candidates
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            c.available_ranges
                .iter()
                .map(move |&(a, b)| (i, (a, b.min(num_layers))))
        })
        .filter(|(_, (a, b))| a < b)
        .collect();

    // Emitting every sub-range is O(k^2) per range in the number of interior
    // split points. That is trivial for a handful of holders and unbounded at
    // swarm scale, so it is budgeted: past the cap we emit whole ranges only,
    // which is exactly the pre-split behaviour rather than a degraded one.
    let subrange_cost: usize = clamped
        .iter()
        .map(|(_, (lo, hi))| {
            let k = split_points
                .iter()
                .filter(|&&p| p >= *lo && p <= *hi)
                .count();
            k.saturating_mul(k.saturating_sub(1)) / 2
        })
        .sum();
    let split_enabled = partial_ranges && subrange_cost <= MAX_SUBRANGE_VERTICES;
    if partial_ranges && !split_enabled {
        tracing::debug!(
            subrange_cost,
            cap = MAX_SUBRANGE_VERTICES,
            candidates = candidates.len(),
            "parallax: too many candidates to consider partial ranges — whole ranges only"
        );
    }

    let mut vertices: Vec<Vertex> = Vec::new();
    for (cand_idx, (lo, hi)) in clamped {
        let c = &candidates[cand_idx];
        // `None` = we cannot tell what this peer can hold, which must never be
        // read as "nothing" (gotcha #330: every node before v0.3.103 gossiped
        // zero free VRAM).
        let cap = if respect_capacity {
            c.max_hostable_layers
        } else {
            None
        };
        let over_capacity = cap.is_some_and(|k| hi - lo > k);
        let mut push = |range: (u32, u32)| {
            if let Some(k) = cap {
                if range.1 - range.0 > k {
                    return;
                }
            }
            vertices.push(Vertex {
                cand_idx,
                range,
                cost_ms: vertex_cost(c, range, local_node_id, num_layers, prompt_tokens).total(),
            });
        };
        // The whole range is always available, so a chain that was routable
        // before stays routable — unless this peer has told us it cannot hold
        // that many layers, in which case `push` drops it and the sub-ranges
        // below are its only way to contribute.
        push((lo, hi));
        // A peer that cannot take its whole range is split even when partial
        // ranges are off: this is a correctness bound, not the throughput
        // optimisation that flag governs. Holding all 32 layers and being able
        // to load 32 layers are different claims, and routing the whole model
        // to a node that can only take 20 of them costs a round trip and a
        // refusal.
        if !split_enabled && !over_capacity {
            continue;
        }
        // Partial use of a range. Without these a candidate could only ever
        // serve its range in full, so a node holding the entire model was the
        // only representable route and no other node's shards could contribute
        // — regardless of how much slower that node was.
        for (i, &p) in split_points.iter().enumerate() {
            if p < lo || p >= hi {
                continue;
            }
            for &q in &split_points[i + 1..] {
                if q > hi {
                    break;
                }
                if (p, q) != (lo, hi) {
                    push((p, q));
                }
            }
        }
    }

    if vertices.is_empty() {
        return Err(SwarmError::PipelineError(
            "parallax: no candidate ranges".into(),
        ));
    }

    // Source filter: start==0. Must have can_be_first. Encrypted: must be local.
    let is_source = |v: &Vertex| -> bool {
        if v.range.0 != 0 {
            return false;
        }
        let c = &candidates[v.cand_idx];
        if !c.can_be_first {
            return false;
        }
        if encrypted_pipeline && &c.node_id != local_node_id {
            return false;
        }
        true
    };

    // Sink filter: end==num_layers. Must have can_be_last. Encrypted: must be local.
    let is_sink = |v: &Vertex| -> bool {
        if v.range.1 != num_layers {
            return false;
        }
        let c = &candidates[v.cand_idx];
        if !c.can_be_last {
            return false;
        }
        if encrypted_pipeline && &c.node_id != local_node_id {
            return false;
        }
        true
    };

    if !vertices.iter().any(is_source) {
        return Err(SwarmError::PipelineError(
            "parallax: no valid source vertex (starts at layer 0, can_be_first)".into(),
        ));
    }
    if !vertices.iter().any(is_sink) {
        return Err(SwarmError::PipelineError(
            "parallax: no valid sink vertex (ends at num_layers, can_be_last)".into(),
        ));
    }

    // Topological order: sort by (range.0 asc, range.1 asc). DAG edges always go
    // from smaller end to larger start, and within the same start we want smaller
    // spans processed first so later vertices can relax against them.
    let mut order: Vec<usize> = (0..vertices.len()).collect();
    order.sort_by(|&a, &b| {
        vertices[a]
            .range
            .0
            .cmp(&vertices[b].range.0)
            .then(vertices[a].range.1.cmp(&vertices[b].range.1))
    });

    // DP: best_cost[v] = min total cost reaching vertex v as a sink of some source.
    // parent[v] = predecessor vertex for path reconstruction.
    let n = vertices.len();
    let mut best_cost = vec![f32::INFINITY; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];

    // A memory bound is per NODE, not per segment. Capping each vertex alone is
    // not enough: the DP will happily give a capped peer two sub-ranges that
    // each fit and together do not, and `merge_contiguous` then hands it back
    // the whole model. (Caught by
    // `a_peer_is_not_handed_more_layers_than_it_says_it_can_hold` — the first
    // cut of this bound shipped that hole.)
    //
    // So a capped candidate may appear at most ONCE in a chain. That is also
    // the shape worth having on its own merits: a node appearing twice means
    // the activations leave it and come back, paying its network hop twice.
    // Tracked as a bitmask over capped candidates carried along the best path;
    // past 64 of them the bound is dropped rather than approximated, and the
    // caller's relaxed pass is the backstop either way.
    let mut capped_bit: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    if respect_capacity {
        for v in &vertices {
            let c = &candidates[v.cand_idx];
            // **The local node is exempt, and prompt privacy is why**
            // (gotcha #481). Both halves of the argument above are about a
            // REMOTE node: it pays its network hop twice, and its memory is
            // known only from what it advertises. This node pays no hop, and
            // an encrypted route REQUIRES it at both ends — so once #452 gave
            // the local node a bound it became a capped candidate, and the
            // default privacy shape stopped being routable. The DP then
            // returned the all-local chain instead, which is `Ok`, so the
            // caller's relaxed pass never ran and the request silently took a
            // route priced 11.8x worse with nothing logged.
            //
            // The memory half still applies and is enforced after
            // reconstruction, where the summed span can actually be measured.
            if &c.node_id == local_node_id {
                continue;
            }
            if c.max_hostable_layers.is_some() && !capped_bit.contains_key(&v.cand_idx) {
                let next = capped_bit.len() as u32;
                if next >= 64 {
                    capped_bit.clear();
                    break;
                }
                capped_bit.insert(v.cand_idx, next);
            }
        }
    }
    let bit_of = |vi: usize| -> u64 {
        capped_bit
            .get(&vertices[vi].cand_idx)
            .map(|b| 1u64 << b)
            .unwrap_or(0)
    };
    let mut used_capped = vec![0u64; n];

    // Initialize sources.
    for i in 0..n {
        if is_source(&vertices[i]) {
            best_cost[i] = vertices[i].cost_ms;
            used_capped[i] = bit_of(i);
        }
    }

    // Forward DP.
    for &v_idx in &order {
        if !best_cost[v_idx].is_finite() {
            continue;
        }
        let v_end = vertices[v_idx].range.1;
        if v_end >= num_layers {
            continue; // nothing to extend
        }
        // Find successors: vertices whose start equals v_end.
        // Scan is O(n); with many candidates we could bucket by start_layer but
        // vertex counts stay small (bounded by peer_count * ranges_per_peer).
        for &w_idx in &order {
            if vertices[w_idx].range.0 != v_end {
                continue;
            }
            let w_bit = bit_of(w_idx);
            if w_bit != 0 && used_capped[v_idx] & w_bit != 0 {
                // This capped node is already carrying part of the chain.
                continue;
            }
            let new_cost = best_cost[v_idx] + vertices[w_idx].cost_ms;
            if new_cost < best_cost[w_idx] {
                best_cost[w_idx] = new_cost;
                parent[w_idx] = Some(v_idx);
                used_capped[w_idx] = used_capped[v_idx] | w_bit;
            }
        }
    }

    // Pick the best sink.
    let best_sink = (0..n).filter(|&i| is_sink(&vertices[i])).min_by(|&a, &b| {
        best_cost[a]
            .partial_cmp(&best_cost[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sink_idx = match best_sink {
        Some(i) if best_cost[i].is_finite() => i,
        _ => {
            return Err(SwarmError::PipelineError(
                "parallax: no reachable source→sink path".into(),
            ));
        }
    };

    // Reconstruct path.
    let mut chain: Vec<usize> = Vec::new();
    let mut cur = Some(sink_idx);
    while let Some(i) = cur {
        chain.push(i);
        cur = parent[i];
    }
    chain.reverse();

    // Convert to PipelineSegments.
    let segments: Vec<PipelineSegment> = chain
        .into_iter()
        .map(|i| {
            let v = &vertices[i];
            let c = &candidates[v.cand_idx];
            PipelineSegment {
                node_id: c.node_id.clone(),
                shard_id: c.shard_id.clone(),
                layer_range: v.range,
            }
        })
        .collect();

    // The memory half of the rule the local node was exempted from above.
    // Exempting it from "appears at most once" must not exempt it from "does
    // not take on more layers than it can hold" — and here, unlike inside the
    // relaxation, the summed span can simply be measured. Failing means the
    // caller's relaxed pass runs, which is the same backstop every other
    // capacity refusal in this function uses.
    if respect_capacity {
        if let Some(cap) = candidates
            .iter()
            .find(|c| &c.node_id == local_node_id)
            .and_then(|c| c.max_hostable_layers)
        {
            let ours: u32 = segments
                .iter()
                .filter(|s| &s.node_id == local_node_id)
                .map(|s| s.layer_range.1.saturating_sub(s.layer_range.0))
                .sum();
            if ours > cap.max(1) {
                return Err(SwarmError::PipelineError(format!(
                    "parallax: this node would take {ours} layers across its segments, \
                     more than the {cap} it can hold"
                )));
            }
        }
    }

    // Sanity: segments must contiguously cover [0, num_layers).
    let mut expect = 0u32;
    for s in &segments {
        if s.layer_range.0 != expect {
            return Err(SwarmError::PipelineError(format!(
                "parallax: non-contiguous chain at layer {expect}"
            )));
        }
        expect = s.layer_range.1;
    }
    if expect != num_layers {
        return Err(SwarmError::PipelineError(format!(
            "parallax: chain ends at {expect}, want {num_layers}"
        )));
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelId, ShardId};

    fn cand(
        byte: u8,
        ranges: Vec<(u32, u32)>,
        latency_ms: u32,
        load: f32,
        can_be_first: bool,
        can_be_last: bool,
        est_tokens_per_sec: f32,
    ) -> NodeCandidate {
        NodeCandidate {
            node_id: NodeId([byte; 32]),
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            available_ranges: ranges,
            reach: crate::inference::scheduler::ReachTier::DirectMeasured,
            latency_ms,
            load,
            trust_score: 1.0,
            can_be_first,
            can_be_last,
            region_score: 1.0,
            est_tokens_per_sec,
            observed_latency_ms_per_layer: None,
            observed_delegated_ms_per_layer: None,
            expected_attempts: 1.0,
            is_pool_member: false,
            gpu_vram_available_mb: None,
            max_hostable_layers: None,
            observed_prefill_ms_per_layer_byte: None,
            has_gpu: false,
        }
    }

    /// The whole point of pricing prefill: a loaded GPU node against an idle
    /// CPU one is the right call for a short prompt and the wrong one for a
    /// long prompt, and before this the scheduler could not tell those two
    /// requests apart.
    ///
    /// The CPU peer here is idle and the GPU peer is carrying load, so on
    /// decode alone the CPU peer wins. Prefill is where the ~55x hardware gap
    /// lives, so a long prompt has to reverse that.
    #[test]
    fn a_long_prompt_moves_the_choice_to_the_machine_that_can_read_it() {
        let local = NodeId([9u8; 32]);
        let gpu_id = NodeId([2u8; 32]);
        let cpu_id = NodeId([3u8; 32]);

        // Both hold the whole model and both ADVERTISE THE SAME DECODE SPEED,
        // so the only thing separating them is the card — which is exactly the
        // variable under test. The GPU peer is carrying load and is further
        // away, so on decode alone it loses.
        let mut gpu = cand(2, vec![(0, 32)], 40, 12.0, true, true, 8.0);
        gpu.has_gpu = true;
        let cpu = cand(3, vec![(0, 32)], 5, 0.0, true, true, 8.0);

        let pick = |tokens: Option<u32>| -> NodeId {
            route_shortest_path(
                32,
                &[gpu.clone(), cpu.clone()],
                &local,
                false,
                false,
                false,
                tokens,
            )
            .expect("a route must exist")[0]
                .node_id
                .clone()
        };

        assert_eq!(
            pick(Some(8)),
            cpu_id,
            "a short prompt should still go to the idle, closer node — its \
             prefill disadvantage is a few tokens' worth and does not outweigh \
             the busy node's queue"
        );
        assert_eq!(
            pick(Some(6000)),
            gpu_id,
            "a 6000-token prompt is minutes of prefill on a processor — it must \
             go to the graphics card even though that node is busier"
        );
    }

    /// `None` must reproduce the cost model exactly as it stood before prefill
    /// was priced, so a caller with no prompt in hand loses nothing. Without
    /// this the change would silently alter the offline allocator and every
    /// non-request routing path.
    #[test]
    fn an_unknown_prompt_length_prices_exactly_as_before() {
        let local = NodeId([9u8; 32]);
        let mut gpu = cand(2, vec![(0, 32)], 20, 0.0, true, true, 20.0);
        gpu.has_gpu = true;
        let cpu = cand(3, vec![(0, 32)], 15, 0.0, true, true, 8.0);
        for c in [&gpu, &cpu] {
            let none = vertex_cost(c, (0, 32), &local, 32, None);
            let zero = vertex_cost(c, (0, 32), &local, 32, Some(0));
            assert_eq!(none.prefill_ms, 0.0, "unknown prompt must cost no prefill");
            assert_eq!(
                none.total(),
                zero.total(),
                "an empty prompt and an unknown one must price identically"
            );
        }
    }

    /// A measured coefficient must override the device-class prior — the prior
    /// exists only until this node has actually prefilled through a peer.
    #[test]
    fn a_measured_prefill_coefficient_outranks_the_prior() {
        let local = NodeId([9u8; 32]);
        // A peer with no GPU, so the prior would price it at the slow CPU
        // ratio; but we have measured it prefilling very fast.
        let mut measured = cand(2, vec![(0, 32)], 20, 0.0, true, true, 8.0);
        measured.observed_prefill_ms_per_layer_byte = Some(1.0e-9);
        let prior_only = cand(3, vec![(0, 32)], 20, 0.0, true, true, 8.0);

        let with_measurement = vertex_cost(&measured, (0, 32), &local, 32, Some(4000)).prefill_ms;
        let with_prior = vertex_cost(&prior_only, (0, 32), &local, 32, Some(4000)).prefill_ms;
        assert!(
            with_measurement < with_prior,
            "the measured peer should be priced cheaper than the prior: \
             {with_measurement} vs {with_prior}"
        );
    }

    /// The live case from 2026-08-21: a node holding 4 of 9 shards asked for
    /// llama-3.1-8b, and every request went whole to the fastest advertised
    /// holder — a 6 GB card that cannot take an 8 GB model. It refused in 3 s.
    ///
    /// The peer's own gossip said it could not hold that many layers, and
    /// nothing consulted it. With the bound on, the peer takes only what it
    /// can and the local node covers the rest.
    #[test]
    fn a_peer_is_not_handed_more_layers_than_it_says_it_can_hold() {
        let local = NodeId([1u8; 32]);
        let peer_id = NodeId([2u8; 32]);
        // As observed: we hold a prefix only (4 of 9 shards), the peer holds
        // every layer and is by far the faster machine.
        let peer = cand(2, vec![(0, 32)], 20, 0.0, true, true, 20.0);
        let mut local_prefix = cand(1, vec![(0, 12)], 0, 0.0, true, false, 0.8);
        local_prefix.node_id = local.clone();

        let peer_layers = |segs: &[PipelineSegment]| -> u32 {
            segs.iter()
                .filter(|s| s.node_id == peer_id)
                .map(|s| s.layer_range.1 - s.layer_range.0)
                .sum()
        };

        // CONTROL: with no capacity figure gossiped, the fast peer is handed
        // more than it could hold. This is the behaviour that was observed, and
        // it remains the behaviour for any peer that tells us nothing.
        let unbounded = route_shortest_path(
            32,
            &[peer.clone(), local_prefix.clone()],
            &local,
            false,
            true,
            true,
            None,
        )
        .expect("a route must exist");
        assert!(
            peer_layers(&unbounded) > 20,
            "control: without the bound the peer should be over-committed, got {unbounded:?}"
        );

        // Now the peer advertises room for 20 of the 32 layers.
        let mut bounded_peer = peer;
        bounded_peer.max_hostable_layers = Some(20);
        let segs = route_shortest_path(
            32,
            &[bounded_peer, local_prefix],
            &local,
            false,
            true,
            true,
            None,
        )
        .expect("a route must still exist");
        assert!(
            peer_layers(&segs) <= 20,
            "peer was handed {} layers but advertised room for 20: {segs:?}",
            peer_layers(&segs)
        );
        assert_eq!(
            segs.iter()
                .map(|s| s.layer_range.1 - s.layer_range.0)
                .sum::<u32>(),
            32,
            "the whole model must still be covered: {segs:?}"
        );
    }

    /// A peer that cannot take its whole range must still be split even when
    /// `parallax_partial_ranges` is OFF — the bound is a correctness
    /// constraint, not the throughput optimisation that flag governs.
    #[test]
    fn the_capacity_bound_splits_a_range_even_with_partial_ranges_off() {
        let local = NodeId([1u8; 32]);
        let mut peer = cand(2, vec![(0, 32)], 20, 0.0, true, true, 20.0);
        peer.max_hostable_layers = Some(16);
        let mut local_head = cand(1, vec![(0, 32)], 0, 0.0, true, true, 0.8);
        local_head.node_id = local.clone();

        // partial_ranges = false, respect_capacity = true
        let segs = route_shortest_path(32, &[peer, local_head], &local, false, false, true, None)
            .expect("a route must exist");
        for seg in &segs {
            if seg.node_id == NodeId([2u8; 32]) {
                assert!(
                    seg.layer_range.1 - seg.layer_range.0 <= 16,
                    "over-capacity segment survived with partial ranges off: {segs:?}"
                );
            }
        }
    }

    /// The bound must never turn a routable request into a failure. When the
    /// ONLY holder cannot take the whole model and nothing else can cover the
    /// remainder, the constrained pass fails and the caller's relaxed pass is
    /// what keeps the request alive.
    #[test]
    fn an_impossible_bound_fails_closed_so_the_caller_can_relax_it() {
        let local = NodeId([9u8; 32]);
        let mut only_holder = cand(2, vec![(0, 32)], 20, 0.0, true, true, 20.0);
        only_holder.max_hostable_layers = Some(4);

        // Constrained: no route (no split points to cut against, nothing else
        // holds these layers).
        assert!(
            route_shortest_path(32, &[only_holder.clone()], &local, false, false, true, None)
                .is_err(),
            "the constrained pass should refuse rather than over-commit the peer"
        );
        // Relaxed: the same call that `assemble_pipeline_for` makes second.
        assert!(
            route_shortest_path(32, &[only_holder], &local, false, false, false, None).is_ok(),
            "the relaxed pass must still route — a self-reported figure may not \
             make a request unservable"
        );
    }

    /// The live v0.3.34 configuration that could not be routed: a fast local
    /// node holding layers 0-10 and a slow remote holding the whole model. The
    /// split needs a vertex starting at layer 10, which the remote's single
    /// (0,16) range did not provide, so all 16 layers went to the slow node.
    #[test]
    fn partial_range_lets_a_local_prefix_pair_with_a_whole_model_holder() {
        let local = NodeId([1u8; 32]);
        let remote_slow = cand_with_obs(cand(2, vec![(0, 16)], 5, 0.0, true, true, 0.0), 107.0);
        let mut local_fast = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_fast.node_id = local.clone();

        let segs = route_shortest_path(
            16,
            &[remote_slow, local_fast],
            &local,
            false,
            true,
            true,
            None,
        )
        .expect("a route must exist");

        assert_eq!(segs.len(), 2, "expected a split, got {segs:?}");
        assert_eq!(segs[0].node_id, local, "local prefix must run locally");
        assert_eq!(segs[0].layer_range, (0, 10));
        assert_eq!(
            segs[1].layer_range,
            (10, 16),
            "remote must serve the suffix"
        );
        assert_ne!(segs[1].node_id, local);
    }

    /// An unmeasured candidate used to cost nothing, so on a freshly started
    /// node every chain tied on its network term and the winner was decided by
    /// vertex iteration order. A wider unmeasured segment must now cost more
    /// than a narrower one.
    #[test]
    fn unmeasured_candidates_are_not_free() {
        let local = NodeId([9u8; 32]);
        let wide = cand(1, vec![(0, 16)], 0, 0.0, true, true, 0.0);
        let narrow = cand(2, vec![(0, 4)], 0, 0.0, true, true, 0.0);
        let cw = vertex_cost(&wide, (0, 16), &local, 16, None).total();
        let cn = vertex_cost(&narrow, (0, 4), &local, 16, None).total();
        assert!(cw > 0.0, "an unmeasured candidate must not be free");
        assert!(
            cw > cn,
            "cost must scale with layers taken on: {cw} vs {cn}"
        );
    }

    /// A measured-fast candidate must beat an unmeasured one. Previously the
    /// unmeasured node looked free and won every time, so the router preferred
    /// nodes it knew nothing about over nodes it had measured and liked.
    #[test]
    fn a_measured_fast_node_beats_an_unmeasured_one() {
        let local = NodeId([9u8; 32]);
        let measured_fast = cand_with_obs(cand(1, vec![(0, 16)], 0, 0.0, true, true, 0.0), 1.0);
        let unmeasured = cand(2, vec![(0, 16)], 0, 0.0, true, true, 0.0);
        let cm = vertex_cost(&measured_fast, (0, 16), &local, 16, None).total();
        let cu = vertex_cost(&unmeasured, (0, 16), &local, 16, None).total();
        assert!(cm < cu, "measured-fast {cm} should beat unmeasured {cu}");
    }

    /// A remote segment is round-tripped into once per token, while a remote
    /// peer running the WHOLE model is delegated in one message and pays its
    /// network once. Charging both the same is what made the router pick a split
    /// that measured 11.2s -> 17.8s on a real LAN pair.
    #[test]
    fn a_mid_chain_segment_pays_network_per_token() {
        let local = NodeId([9u8; 32]);
        let remote = cand(2, vec![(0, 16)], 20, 0.0, true, true, 0.0);
        // Same candidate, differing only in whether it runs the whole model.
        let delegated = vertex_cost(&remote, (0, 16), &local, 16, None);
        let as_mid_chain = vertex_cost(&remote, (8, 16), &local, 16, None);
        assert!(
            as_mid_chain.network_ms > delegated.network_ms,
            "mid-chain network {} must exceed delegated {}",
            as_mid_chain.network_ms,
            delegated.network_ms
        );
        assert_eq!(
            as_mid_chain.network_ms,
            delegated.network_ms * ASSUMED_FORWARD_PASSES,
            "mid-chain network should scale by the assumed pass count"
        );
    }

    /// A remote segment starting at layer 0 but NOT covering the whole model is
    /// still entered once per token: the coordinator samples each token and
    /// hands it back to the first segment. Charging that segment its network
    /// once — which keying on `range.0 != 0` did — subsidised splitting by
    /// hiding most of the cost of the boundary being chosen.
    #[test]
    fn a_remote_first_segment_of_a_split_still_pays_per_token() {
        let local = NodeId([9u8; 32]);
        let remote = cand(2, vec![(0, 16)], 20, 0.0, true, true, 0.0);
        let first_of_split = vertex_cost(&remote, (0, 8), &local, 16, None);
        let whole_model = vertex_cost(&remote, (0, 16), &local, 16, None);
        assert_eq!(
            first_of_split.network_ms,
            whole_model.network_ms * ASSUMED_FORWARD_PASSES,
            "a remote 0..8 of a 16-layer model is not a delegation and must pay \
             the round trip per token"
        );
    }

    /// The delegated shape is priced by the delegated observation, and a
    /// mid-chain shape by the mid-chain one. Substituting either for the other
    /// is a measured mis-pricing: a mid-chain figure carries a round trip
    /// amortised over that segment's layers, and reusing it for a whole-model
    /// delegation charges that trip several times over for a trip the
    /// delegation never makes.
    #[test]
    fn each_shape_is_priced_by_the_observation_measured_on_it() {
        let local = NodeId([9u8; 32]);
        let mut c = cand(2, vec![(0, 16)], 20, 0.0, true, true, 0.0);
        // Expensive mid-chain (carries a per-token round trip), cheap delegated.
        c.observed_latency_ms_per_layer = Some(100.0);
        c.observed_delegated_ms_per_layer = Some(10.0);

        let delegated = vertex_cost(&c, (0, 16), &local, 16, None);
        let mid_chain = vertex_cost(&c, (8, 16), &local, 16, None);

        assert_eq!(
            delegated.compute_ms,
            10.0 * 16.0 * ASSUMED_FORWARD_PASSES,
            "the whole-model vertex must use the delegated coefficient"
        );
        assert_eq!(
            mid_chain.compute_ms,
            100.0 * 8.0 * ASSUMED_FORWARD_PASSES,
            "the mid-chain vertex must use the mid-chain coefficient"
        );
    }

    /// Earning a delegated sample must actually change the price. Until one
    /// exists the mid-chain figure stands in as an upper bound — deliberately,
    /// since dropping it would re-price a known-slow peer at the unknown prior
    /// — and the first delegated request replaces it with the right-shaped
    /// number. Without this the coefficient would be collected and ignored,
    /// which is the state this fix ends.
    #[test]
    fn a_delegated_sample_corrects_the_mid_chain_upper_bound() {
        let local = NodeId([9u8; 32]);
        // The live RTX 4050 shape: a mid-chain observation an order of
        // magnitude worse than what the card actually delivers delegated.
        let mut c = cand(2, vec![(0, 16)], 500, 0.0, true, true, 20.0);
        c.observed_latency_ms_per_layer = Some(1063.0);

        c.observed_delegated_ms_per_layer = None;
        let bounded = vertex_cost(&c, (0, 16), &local, 16, None);
        assert_eq!(
            bounded.compute_ms,
            1063.0 * 16.0 * ASSUMED_FORWARD_PASSES,
            "with no delegated sample the mid-chain figure stands in"
        );

        c.observed_delegated_ms_per_layer = Some(60.0);
        let corrected = vertex_cost(&c, (0, 16), &local, 16, None);
        assert_eq!(
            corrected.compute_ms,
            60.0 * 16.0 * ASSUMED_FORWARD_PASSES,
            "a delegated sample must take over from the upper bound"
        );
        assert!(
            corrected.total() < bounded.total(),
            "correcting the overcharge must make delegation cheaper: {} vs {}",
            corrected.total(),
            bounded.total()
        );
    }

    /// A path that loses replies must cost more, in proportion to how often it
    /// loses them — and a reliable or unmeasured path must cost exactly what it
    /// did before.
    ///
    /// This is the deliberate replacement for an accident. A truncated stream
    /// used to poison the peer's SPEED figure, which was wrong about why (a
    /// network fault recorded as slow hardware) but right about where not to
    /// send work. Removing the mis-attribution removed the avoidance with it.
    #[test]
    fn a_lossy_path_costs_more_in_proportion_to_its_losses() {
        let local = NodeId([9u8; 32]);
        let mut reliable = cand(2, vec![(0, 16)], 20, 0.0, true, true, 5.0);
        reliable.expected_attempts = 1.0;
        let baseline = vertex_cost(&reliable, (0, 16), &local, 16, None).total();

        // Half the replies arrive whole: two attempts per usable answer.
        let mut lossy = reliable.clone();
        lossy.expected_attempts = 2.0;
        let doubled = vertex_cost(&lossy, (0, 16), &local, 16, None).total();
        assert_eq!(
            doubled,
            baseline * 2.0,
            "a path delivering half the replies must cost twice as much"
        );

        // The measured case: 3 tokens of 60 arriving, clamped at 20x.
        let mut terrible = reliable.clone();
        terrible.expected_attempts = 20.0;
        assert_eq!(
            vertex_cost(&terrible, (0, 16), &local, 16, None).total(),
            baseline * 20.0
        );
    }

    /// A nonsensical multiplier must not corrupt the route. Below 1.0 would
    /// make a lossy peer CHEAPER than a perfect one, which is the opposite of
    /// the point, and a non-finite value would poison the whole DP.
    #[test]
    fn an_impossible_reliability_figure_is_ignored() {
        let local = NodeId([9u8; 32]);
        let mut c = cand(2, vec![(0, 16)], 20, 0.0, true, true, 5.0);
        c.expected_attempts = 1.0;
        let baseline = vertex_cost(&c, (0, 16), &local, 16, None).total();

        for bad in [0.0f32, 0.5, -3.0, f32::NAN, f32::INFINITY] {
            c.expected_attempts = bad;
            let got = vertex_cost(&c, (0, 16), &local, 16, None).total();
            assert_eq!(got, baseline, "multiplier {bad} must be ignored, got {got}");
        }
    }

    /// End to end: a fast peer on a path that eats most of its replies must
    /// lose to a slower peer that actually delivers. This is the live case —
    /// an RTX 4050 returning 3 tokens of 60 while a LAN CPU returned all of
    /// them.
    #[test]
    fn a_reliable_slow_peer_beats_a_fast_one_that_loses_replies() {
        let local = NodeId([1u8; 32]);
        // Fast, distant, and losing 95% of replies.
        let mut fast_lossy = cand(2, vec![(0, 16)], 500, 0.0, true, true, 20.0);
        fast_lossy.expected_attempts = 20.0;
        // A quarter the throughput, on the LAN, delivering everything.
        let mut slow_reliable = cand(3, vec![(0, 16)], 1, 0.0, true, true, 5.0);
        slow_reliable.expected_attempts = 1.0;

        let segs = route_shortest_path(
            16,
            &[fast_lossy, slow_reliable],
            &local,
            false,
            false,
            true,
            None,
        )
        .expect("a route must exist");
        assert_eq!(segs.len(), 1);
        assert_eq!(
            segs[0].node_id,
            NodeId([3u8; 32]),
            "a peer that delivers must beat a faster one that does not"
        );
    }

    /// The local node carries no network term wherever it sits in the chain.
    #[test]
    fn the_local_node_never_pays_network() {
        let local = NodeId([1u8; 32]);
        let mut c = cand(1, vec![(0, 16)], 50, 0.0, true, true, 0.0);
        c.node_id = local.clone();
        assert_eq!(vertex_cost(&c, (0, 8), &local, 16, None).network_ms, 0.0);
        assert_eq!(vertex_cost(&c, (8, 16), &local, 16, None).network_ms, 0.0);
    }

    /// With the per-token term in place, a split onto a peer that is only
    /// slightly faster must NOT be chosen — the round trips outweigh it. This is
    /// the regression the live measurement exposed.
    #[test]
    fn a_marginal_speedup_does_not_justify_per_token_round_trips() {
        let local = NodeId([1u8; 32]);
        // Remote is somewhat faster per layer but sits behind a real RTT.
        let remote = cand_with_obs(cand(2, vec![(0, 16)], 20, 0.0, true, true, 0.0), 8.0);
        let mut local_node = cand_with_obs(cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0), 10.0);
        local_node.node_id = local.clone();

        let segs = route_shortest_path(16, &[remote, local_node], &local, false, true, true, None)
            .expect("route");
        assert_eq!(
            segs.len(),
            1,
            "a marginal per-layer gain must not buy a per-token boundary: {segs:?}"
        );
    }

    /// External report 2026-07-27, Finding 4: `encrypted_pipeline = true` could
    /// not assemble a pipeline in EITHER tested topology, including the nominal
    /// boomerang (local holds first and last, peer holds the middle).
    ///
    /// It is the same root cause as the whole-model monopolisation above.
    /// Encryption forces source and sink to be local, so the middle must come
    /// from a peer — but a peer holding the ENTIRE model has only the vertex
    /// (0, N), which can be neither a middle segment nor (being remote) a source
    /// or sink. With ranges indivisible there is no chain at all.
    #[test]
    fn encrypted_boomerang_is_unroutable_without_partial_ranges() {
        let local = NodeId([1u8; 32]);
        // Local holds the head and the tail, not the middle.
        let mut head = cand(1, vec![(0, 3)], 0, 0.0, true, false, 0.0);
        head.node_id = local.clone();
        let mut tail = cand(1, vec![(21, 28)], 0, 0.0, false, true, 0.0);
        tail.node_id = local.clone();
        // A peer holds the whole model.
        let peer = cand(2, vec![(0, 28)], 5, 0.0, true, true, 0.0);

        let cands = vec![head, tail, peer];
        assert!(
            route_shortest_path(28, &cands, &local, true, false, true, None).is_err(),
            "reproduces the reported failure: no route with ranges indivisible"
        );

        // And the fix: let the peer serve part of its range.
        let segs = route_shortest_path(28, &cands, &local, true, true, true, None)
            .expect("partial ranges must make the boomerang routable");
        assert_eq!(
            segs.len(),
            3,
            "expected local head, peer middle, local tail: {segs:?}"
        );
        assert_eq!(segs[0].node_id, local);
        assert_eq!(segs[0].layer_range, (0, 3));
        assert_ne!(segs[1].node_id, local, "the middle must be the peer");
        assert_eq!(segs[1].layer_range, (3, 21));
        assert_eq!(segs[2].node_id, local);
        assert_eq!(segs[2].layer_range, (21, 28));
    }

    /// Does encrypted_pipeline work when a peer holds EXACTLY the middle range?
    #[test]
    fn encrypted_boomerang_routes_when_a_peer_holds_exactly_the_middle() {
        let local = NodeId([1u8; 32]);
        let mut head = cand(1, vec![(0, 3)], 0, 0.0, true, false, 0.0);
        head.node_id = local.clone();
        let mut tail = cand(1, vec![(21, 28)], 0, 0.0, false, true, 0.0);
        tail.node_id = local.clone();
        // Peer holds ONLY the middle — the aligned case.
        let peer = cand(2, vec![(3, 21)], 5, 0.0, false, false, 0.0);

        let segs = route_shortest_path(28, &[head, tail, peer], &local, true, false, true, None)
            .expect("aligned middle must route with partial ranges OFF");
        assert_eq!(segs.len(), 3, "{segs:?}");
        assert_eq!(segs[0].layer_range, (0, 3));
        assert_eq!(segs[1].layer_range, (3, 21));
        assert_ne!(segs[1].node_id, local);
        assert_eq!(segs[2].layer_range, (21, 28));
    }

    /// The tester's Topology B, which is what the scheduler now enables partial
    /// ranges for automatically: a valid boomerang where the only peer holds a
    /// SUPERSET of the middle. Encrypted pipelines are multi-segment by
    /// construction, so allowing a partial range costs nothing they were not
    /// already paying.
    #[test]
    fn encrypted_boomerang_routes_against_a_whole_model_peer() {
        let local = NodeId([1u8; 32]);
        let mut head = cand(1, vec![(0, 3)], 0, 0.0, true, false, 0.0);
        head.node_id = local.clone();
        let mut tail = cand(1, vec![(21, 28)], 0, 0.0, false, true, 0.0);
        tail.node_id = local.clone();
        let whole_model_peer = cand(2, vec![(0, 28)], 5, 0.0, true, true, 0.0);

        let segs = route_shortest_path(
            28,
            &[head, tail, whole_model_peer],
            &local,
            true,
            true,
            true,
            None,
        )
        .expect("must route: this is the topology encryption is designed for");
        assert_eq!(segs.len(), 3, "{segs:?}");
        assert_eq!(segs[0].node_id, local, "first segment must stay local");
        assert_ne!(segs[1].node_id, local, "middle must be the peer");
        assert_eq!(segs[2].node_id, local, "last segment must stay local");
        // The privacy guarantee: the peer never sees layer 0 or the final layer.
        assert!(segs[1].layer_range.0 > 0 && segs[1].layer_range.1 < 28);
    }

    /// Gotcha #481. The boomerang as PRODUCTION builds it: `gather_candidates`
    /// emits ONE candidate per node with its ranges merged, so the local node
    /// is a single entry that has to supply both ends. The sibling test above
    /// splits it into two entries, which gives them two `cand_idx` values and
    /// so two different bits — sidestepping the "a capped candidate appears at
    /// most once" rule entirely.
    ///
    /// That rule is right about a REMOTE node (appearing twice means the
    /// activations leave and come back, paying its hop twice) and wrong about
    /// this one: the local node pays no hop, and prompt privacy REQUIRES it at
    /// both ends. Since #452 gave the local node a capacity bound, it became a
    /// capped candidate — so the constrained pass could no longer route the
    /// default privacy shape at all, and every such request fell through to the
    /// relaxed pass, which drops the memory bound for EVERY peer.
    #[test]
    fn the_local_node_may_hold_both_ends_of_a_boomerang_while_capped() {
        let local = NodeId([1u8; 32]);
        // A slow processor beside a fast card: the shape the boomerang exists
        // for, and the only one where an all-local chain is NOT the answer.
        let mut me = cand(1, vec![(0, 28)], 0, 0.0, true, true, 0.5);
        me.node_id = local.clone();
        // The bound #452 introduced. Comfortably more than the two layers the
        // two ends actually need.
        me.max_hostable_layers = Some(28);
        let peer = cand(2, vec![(0, 28)], 5, 0.0, true, true, 60.0);

        // THE CONTROL, and it is the whole point: the identical fixture with
        // the cap removed routes as a boomerang. So the cap — not the cost
        // model — is what decides, which is exactly the mechanism claim.
        let mut uncapped = me.clone();
        uncapped.max_hostable_layers = None;
        let control = route_shortest_path(
            28,
            &[uncapped, peer.clone()],
            &local,
            true,
            true,
            true,
            None,
        )
        .expect("uncapped, this topology routes as a boomerang");
        assert_eq!(
            control.len(),
            3,
            "control: an uncapped local node uses the fast peer: {control:?}"
        );

        let segs = route_shortest_path(28, &[me, peer], &local, true, true, true, None)
            .expect("the constrained pass must route the shape privacy is on by default for");
        assert_eq!(segs.len(), 3, "{segs:?}");
        assert_eq!(segs[0].node_id, local);
        assert_ne!(segs[1].node_id, local);
        assert_eq!(segs[2].node_id, local);
    }

    /// Exempting the local node from "appears at most once" must not exempt it
    /// from its memory bound — the summed span is checked after
    /// reconstruction, where it can actually be measured.
    #[test]
    fn the_local_nodes_segments_are_still_summed_against_its_bound() {
        let local = NodeId([1u8; 32]);
        let mut me = cand(1, vec![(0, 28)], 0, 0.0, true, true, 0.5);
        me.node_id = local.clone();
        let peer = cand(2, vec![(0, 28)], 5, 0.0, true, true, 60.0);

        // Room for both ends: routes.
        let mut fits = me.clone();
        fits.max_hostable_layers = Some(2);
        assert_eq!(
            route_shortest_path(28, &[fits, peer.clone()], &local, true, true, true, None)
                .expect("two one-layer ends fit a bound of two")
                .len(),
            3
        );

        // Room for one end only: the constrained pass must refuse rather than
        // hand this node more than it can hold, leaving the relaxed pass to
        // decide — the same backstop every other capacity refusal here uses.
        let mut too_small = me;
        too_small.max_hostable_layers = Some(1);
        let routed = route_shortest_path(28, &[too_small, peer], &local, true, true, true, None);
        match routed {
            Err(_) => {}
            Ok(segs) => {
                let ours: u32 = segs
                    .iter()
                    .filter(|s| s.node_id == local)
                    .map(|s| s.layer_range.1 - s.layer_range.0)
                    .sum();
                assert!(
                    ours <= 1,
                    "this node took {ours} layers against a bound of 1"
                );
            }
        }
    }

    /// The half of the rule that must survive: a capped REMOTE peer still may
    /// not be handed two slices of one chain, which is what
    /// `a_peer_is_not_handed_more_layers_than_it_says_it_can_hold` was written
    /// for. Exempting the local node must not exempt anyone else.
    #[test]
    fn a_capped_peer_still_may_not_take_two_slices_of_one_chain() {
        let local = NodeId([1u8; 32]);
        let mut me = cand(1, vec![(0, 4)], 0, 0.0, true, false, 8.0);
        me.node_id = local.clone();
        // Holds everything, but only has room for a few layers at a time, and
        // is the ONLY other candidate — so a chain covering 0..28 would have to
        // use it twice.
        let mut small = cand(2, vec![(0, 28)], 5, 0.0, true, true, 20.0);
        small.max_hostable_layers = Some(6);

        let routed = route_shortest_path(28, &[me, small], &local, false, true, true, None);
        if let Ok(segs) = routed {
            let slices = segs
                .iter()
                .filter(|s| s.node_id == NodeId([2u8; 32]))
                .count();
            assert!(
                slices <= 1,
                "a capped peer must not carry two slices of one chain: {segs:?}"
            );
        }
    }

    /// Encryption must still refuse a topology it cannot make private — local
    /// not holding the tail means the sink would have to be remote, which would
    /// expose the sampled tokens. The tester's Topology A.
    #[test]
    fn encrypted_refuses_when_this_node_cannot_hold_the_tail() {
        let local = NodeId([1u8; 32]);
        let mut head = cand(1, vec![(0, 12)], 0, 0.0, true, false, 0.0);
        head.node_id = local.clone();
        let peer = cand(2, vec![(0, 28)], 5, 0.0, true, true, 0.0);
        assert!(
            route_shortest_path(28, &[head, peer], &local, true, true, true, None).is_err(),
            "must refuse rather than leak the tail to a peer"
        );
    }

    /// With partial ranges OFF — the shipped default — routing must be exactly
    /// what it was before they existed: the whole-model holder takes everything.
    /// This is the behaviour the default protects, because a multi-segment chain
    /// exchanges activations per token and measured slower on a LAN pair.
    #[test]
    fn partial_ranges_off_keeps_the_single_segment_route() {
        let local = NodeId([1u8; 32]);
        let remote = cand_with_obs(cand(2, vec![(0, 16)], 5, 0.0, true, true, 0.0), 107.0);
        let mut local_fast = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_fast.node_id = local.clone();

        let segs = route_shortest_path(16, &[remote, local_fast], &local, false, false, true, None)
            .expect("a route must exist");
        assert_eq!(segs.len(), 1, "default must not split: {segs:?}");
        assert_eq!(segs[0].layer_range, (0, 16));
    }

    /// The cost model must still be what decides. With the remote fast and the
    /// local expensive, splitting is not worth it — so a partial range being
    /// *available* must not make it *mandatory*.
    #[test]
    fn partial_ranges_do_not_force_a_split_when_whole_is_cheaper() {
        let local = NodeId([1u8; 32]);
        // A remote with an observed per-layer cost of ~0 is essentially free.
        let remote_fast = cand_with_obs(cand(2, vec![(0, 16)], 0, 0.0, true, true, 0.0), 0.01);
        // Local looks free too (local always does), but splitting adds a second
        // segment whose cost cannot beat one nearly-free whole-model segment.
        let mut local_node = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_node.node_id = local.clone();

        let segs = route_shortest_path(
            16,
            &[remote_fast, local_node],
            &local,
            false,
            true,
            true,
            None,
        )
        .expect("route");
        let total: u32 = segs.iter().map(|s| s.layer_range.1 - s.layer_range.0).sum();
        assert_eq!(total, 16, "coverage must be complete either way");
    }

    #[test]
    fn partial_ranges_still_cover_every_layer_contiguously() {
        let local = NodeId([9u8; 32]);
        let a = cand(1, vec![(0, 12)], 10, 0.0, true, false, 0.0);
        let b = cand(2, vec![(4, 20)], 10, 0.0, false, true, 0.0);
        let segs =
            route_shortest_path(20, &[a, b], &local, false, true, true, None).expect("route");
        let mut expect = 0;
        for s in &segs {
            assert_eq!(s.layer_range.0, expect, "gap or overlap in {segs:?}");
            expect = s.layer_range.1;
        }
        assert_eq!(expect, 20);
    }

    /// A sink must still hold the final shard: partial ranges must not let a
    /// `can_be_last = false` candidate end the chain.
    #[test]
    fn partial_ranges_respect_can_be_last() {
        let local = NodeId([9u8; 32]);
        // Covers everything but is not allowed to be last.
        let not_last = cand(1, vec![(0, 16)], 1, 0.0, true, false, 0.0);
        assert!(route_shortest_path(16, &[not_last], &local, false, true, true, None).is_err());
    }

    /// And a source must still be able to be first.
    #[test]
    fn partial_ranges_respect_can_be_first() {
        let local = NodeId([9u8; 32]);
        let not_first = cand(1, vec![(0, 16)], 1, 0.0, false, true, 0.0);
        assert!(route_shortest_path(16, &[not_first], &local, false, true, true, None).is_err());
    }

    /// Encrypted pipelines require the local node at both ends; partial ranges
    /// must not open a path around that.
    #[test]
    fn partial_ranges_do_not_bypass_encrypted_pipeline_ends() {
        let local = NodeId([9u8; 32]);
        let remote = cand(2, vec![(0, 16)], 1, 0.0, true, true, 0.0);
        assert!(
            route_shortest_path(16, &[remote], &local, true, true, true, None).is_err(),
            "a remote-only chain must be refused when encrypted_pipeline is on"
        );
    }

    /// Above the budget the DP falls back to whole ranges — which must still
    /// route, not fail.
    #[test]
    fn many_candidates_fall_back_to_whole_ranges_and_still_route() {
        let local = NodeId([200u8; 32]);
        let mut cands = Vec::new();
        for i in 0..90u8 {
            // Overlapping ranges across many holders: enough split points to
            // blow the sub-range budget.
            let lo = (i as u32) % 40;
            cands.push(cand(i, vec![(lo, 80)], 5, 0.0, true, true, 0.0));
        }
        let segs = route_shortest_path(80, &cands, &local, false, true, true, None)
            .expect("must still route");
        let mut expect = 0;
        for s in &segs {
            assert_eq!(s.layer_range.0, expect);
            expect = s.layer_range.1;
        }
        assert_eq!(expect, 80);
    }

    fn cand_with_obs(mut c: NodeCandidate, observed_ms_per_layer: f32) -> NodeCandidate {
        c.observed_latency_ms_per_layer = Some(observed_ms_per_layer);
        c
    }

    #[test]
    fn single_node_covers_all() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 32)], 0, 0.0, true, true, 0.0)];
        let segs = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].layer_range, (0, 32));
        assert_eq!(segs[0].node_id, local);
    }

    #[test]
    fn picks_low_latency_chain() {
        let local = NodeId([1u8; 32]);
        // Local holds only [0,8); two remotes cover [8,32): slow (latency 200) vs fast (latency 10).
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(8, 32)], 200, 0.0, false, true, 0.0),
            cand(3, vec![(8, 32)], 10, 0.0, false, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].node_id, local);
        assert_eq!(segs[1].node_id, NodeId([3u8; 32]));
    }

    #[test]
    fn load_penalty_shifts_choice() {
        let local = NodeId([99u8; 32]);
        // Two remotes, both [0,32), same latency. One is heavily loaded.
        let cands = vec![
            cand(1, vec![(0, 32)], 50, 10.0, true, true, 0.0),
            cand(2, vec![(0, 32)], 50, 0.0, true, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].node_id, NodeId([2u8; 32]));
    }

    #[test]
    fn encrypted_requires_local_first_and_last() {
        let local = NodeId([1u8; 32]);
        // Local holds [0,4) and [28,32); one remote covers middle.
        let cands = vec![
            cand(1, vec![(0, 4), (28, 32)], 0, 0.0, true, true, 0.0),
            cand(2, vec![(4, 28)], 5, 0.0, false, false, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, true, false, true, None).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].node_id, local);
        assert_eq!(segs[0].layer_range, (0, 4));
        assert_eq!(segs[2].node_id, local);
        assert_eq!(segs[2].layer_range, (28, 32));
    }

    #[test]
    fn no_first_capable_errors() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(2, vec![(0, 32)], 50, 0.0, false, true, 0.0)];
        let err = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn no_sink_errors() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 16)], 0, 0.0, true, false, 0.0)];
        let err = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn disjoint_ranges_fail_cleanly() {
        let local = NodeId([1u8; 32]);
        // Gap between [0,8) and [16,32) — layers 8-15 missing.
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(16, 32)], 10, 0.0, false, true, 0.0),
        ];
        let err = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn observed_latency_overrides_static_estimate() {
        // Two remotes, identical static latency (50 ms) and est_tokens_per_sec (0).
        // Without observations they tie. Add an observed-latency EMA showing one
        // is actually slow (20 ms/layer) and the other fast (2 ms/layer); DP picks
        // the fast one.
        let local = NodeId([99u8; 32]);
        let slow = cand_with_obs(cand(1, vec![(0, 32)], 50, 0.0, true, true, 0.0), 20.0);
        let fast = cand_with_obs(cand(2, vec![(0, 32)], 50, 0.0, true, true, 0.0), 2.0);
        let cands = vec![slow, fast];
        let segs = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].node_id, NodeId([2u8; 32]));
    }

    #[test]
    fn multi_hop_chain_minimizes_total_latency() {
        let local = NodeId([1u8; 32]);
        // Two paths possible: 1→2→3 (latencies 0+50+50) vs 1→4 (local alone can't finish)
        // Simpler: ensure 3-hop chain with lower total beats 2-hop chain.
        // Local [0,8), A[8,16)@10ms, B[16,32)@10ms  vs  Local[0,8), C[8,32)@100ms
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(8, 16)], 10, 0.0, false, false, 0.0),
            cand(3, vec![(16, 32)], 10, 0.0, false, true, 0.0),
            cand(4, vec![(8, 32)], 100, 0.0, false, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false, true, None).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].node_id, NodeId([2u8; 32]));
        assert_eq!(segs[2].node_id, NodeId([3u8; 32]));
    }

    /// Replay of the three real candidates from the live swarm, 2026-08-19.
    ///
    /// Measured: the scheduler chose the node that is LAST on both latency and
    /// throughput, five times out of five, and the request came back at 0.23
    /// tok/s against a 36 tok/s local baseline. This pins which term decides it.
    /// A much faster remote must win over closer but slower ones.
    ///
    /// The three real candidates from the live swarm on 2026-08-19, the first
    /// day this swarm had a second GPU: a 20.45 tok/s RTX 4050 at 455 ms
    /// against an 0.82 tok/s node at 75 ms and a 1.26 tok/s node at 637 ms, all
    /// three holding the whole 16-layer model.
    ///
    /// The DP gets this right, which is what makes it worth pinning: the live
    /// router chose the 1.26 tok/s node five times out of five and the request
    /// came back at 0.23 tok/s against a 36 tok/s local baseline. Since the cost
    /// model prefers the GPU on these numbers, the divergence is in the inputs
    /// it was given, not in this function — see `docs/FUTURE_WORK.md`.
    #[test]
    fn a_much_faster_remote_beats_closer_slower_ones() {
        let local = crate::types::NodeId([9u8; 32]);
        let near_slow = cand(1, vec![(0, 16)], 75, 0.0, true, true, 0.82);
        let far_fast = cand(2, vec![(0, 16)], 455, 0.0, true, true, 20.45);
        let far_slow = cand(3, vec![(0, 16)], 637, 0.0, true, true, 1.26);
        let segs = route_shortest_path(
            16,
            &[near_slow, far_fast, far_slow],
            &local,
            false,
            true,
            true,
            None,
        )
        .expect("must route");
        assert_eq!(segs.len(), 1, "one holder can serve the whole model");
        assert_eq!(
            segs[0].node_id.0[0], 2,
            "a 25x faster peer must win despite being 6x further away"
        );
    }
    /// A chain is priced as the sum of its segments, in the same milliseconds
    /// `vertex_cost` speaks — so the log can put the pipeline's figure beside
    /// the local route it displaced.
    #[test]
    fn a_chain_costs_the_sum_of_its_segments() {
        let local = NodeId([0xAA; 32]);
        let a = cand(1, vec![(0, 16)], 20, 0.0, true, false, 10.0);
        let b = cand(2, vec![(16, 32)], 30, 0.0, false, true, 10.0);
        let cands = vec![a.clone(), b.clone()];
        let segs = vec![
            PipelineSegment {
                node_id: a.node_id.clone(),
                shard_id: a.shard_id.clone(),
                layer_range: (0, 16),
            },
            PipelineSegment {
                node_id: b.node_id.clone(),
                shard_id: b.shard_id.clone(),
                layer_range: (16, 32),
            },
        ];
        let expected = vertex_cost(&a, (0, 16), &local, 32, Some(1000)).total()
            + vertex_cost(&b, (16, 32), &local, 32, Some(1000)).total();
        let got = chain_cost_ms(&segs, &cands, &local, 32, Some(1000));
        assert!((got - expected).abs() < 1e-3, "{got} vs {expected}");
        assert!(got > 0.0);
    }
    /// A node holding EVERY layer of a model it would run on its processor,
    /// with prompt privacy on: the only private route is a boomerang, and until
    /// the boomerang's ends were split points the search could not build one
    /// across two peers each holding half — its ranges had no boundary at layer
    /// 1 or 31 — so the whole model stayed on the processor however cheap the
    /// cards were priced (gotcha #444). Control: privacy off routes straight to
    /// the two cards.
    #[test]
    fn an_encrypted_whole_model_holder_is_routed_as_a_boomerang_across_two_cards() {
        let local = NodeId([1u8; 32]);
        let mut slow_local = cand(1, vec![(0, 32)], 0, 0.0, true, true, 1.0);
        slow_local.node_id = local.clone();
        let mut a = cand(2, vec![(0, 16)], 5, 0.0, true, false, 20.0);
        a.has_gpu = true;
        let mut b = cand(3, vec![(16, 32)], 5, 0.0, false, true, 20.0);
        b.has_gpu = true;
        let cands = vec![slow_local, a, b];

        let segs = route_shortest_path(32, &cands, &local, true, true, true, Some(14_000))
            .expect("the boomerang across two cards must be routable");
        let shape: Vec<(bool, (u32, u32))> = segs
            .iter()
            .map(|s| (s.node_id == local, s.layer_range))
            .collect();
        assert_eq!(
            shape,
            vec![
                (true, (0, 1)),
                (false, (1, 16)),
                (false, (16, 31)),
                (true, (31, 32))
            ],
            "{segs:?}"
        );

        // Control: with privacy off nothing forces the ends home, and the two
        // cards take a half each.
        let segs =
            route_shortest_path(32, &cands, &local, false, false, true, Some(14_000)).unwrap();
        let nodes: Vec<u8> = segs.iter().map(|s| s.node_id.0[0]).collect();
        assert_eq!(nodes, vec![2, 3], "{segs:?}");
    }
}
