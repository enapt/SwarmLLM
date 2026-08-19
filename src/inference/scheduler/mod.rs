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
    /// True if this node is in our device pool (preferred for routing — free, trusted, low latency).
    is_pool_member: bool,
    /// Free GPU memory this node last advertised, in MB. `None` when it has no
    /// GPU, or has told us nothing.
    ///
    /// Self-reported, so it is only ever used to answer "could this peer
    /// plausibly run the whole model on its GPU" — never to rank peers against
    /// each other. See [`delegation_target`].
    gpu_vram_available_mb: Option<u64>,
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
/// 200 ms sits 1.5x above the worst local reading and 2.2x below the best
/// remote one. The first attempt at this constant was 50 ms, which read as
/// obviously generous for a LAN and in fact **excluded a peer on the same
/// machine** whenever either node was busy — the feature would have shipped
/// inert on exactly the loaded nodes that need it.
const DELEGATE_MAX_LATENCY_MS: u32 = 200;

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
///   this model does not fit it, so serving here means the CPU fallback. An
///   unreadable estimate, no configured budget, a node told to use its CPU and
///   a node with no usable GPU are all NOT degraded; see
///   `ModelProcessPool::is_cpu_bound_for_lack_of_vram`, which owns that
///   distinction. Declining to serve over a file we could not read would be
///   worse than the problem being solved.
/// - **The peer covers every layer.** This is a delegation, not a split. A
///   split pays a network round trip per token and measured slower than a
///   single remote segment every time it was tried (see `docs/FUTURE_WORK.md`).
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
fn delegation_target<'a>(
    candidates: &'a [NodeCandidate],
    local_node_id: &NodeId,
    num_layers: u32,
    local_is_cpu_bound_for_lack_of_vram: bool,
    model_vram_mb: u64,
    local_cpu_tokens_per_sec: f32,
) -> Option<&'a NodeCandidate> {
    // Only a node with a working GPU that this model does not fit is degraded.
    // `ModelProcessPool::is_cpu_bound_for_lack_of_vram` owns that distinction —
    // a node told to use its CPU, or without a usable GPU, is working normally.
    if !local_is_cpu_bound_for_lack_of_vram {
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
        tracing::debug!(
            peer = %c.node_id,
            reach = ?c.reach,
            latency_ms = c.latency_ms,
            trust = c.trust_score,
            free_vram_mb = ?c.gpu_vram_available_mb,
            needed_vram_mb = needed,
            "Not handing this model to peer: {reason}"
        );
    }
    None
}

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
    if num_layers < 3 {
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
    pub fn new(shared_state: Arc<SharedState>) -> Self {
        Self { shared_state }
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
        self.assemble_pipeline_for(model_id, local_node_id, uuid::Uuid::new_v4())
    }

    /// Assemble a pipeline for the given model with a specific request ID.
    pub fn assemble_pipeline_for(
        &self,
        model_id: &ModelId,
        local_node_id: &NodeId,
        request_id: uuid::Uuid,
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

        // Gather all candidates: nodes that have shards for this model
        let candidates = self.gather_candidates(&manifest, local_node_id, request_id);
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
        if local_covers_everything {
            let pool = &self.shared_state.model_process_pool;
            if let Some(peer) = delegation_target(
                &candidates,
                local_node_id,
                num_layers,
                pool.is_cpu_bound_for_lack_of_vram(model_id),
                pool.estimated_gpu_mb(model_id).unwrap_or(0),
                // OUR processor speed, not our graphics card's: this only runs
                // when the model does not fit the card, so the processor is
                // what the request would actually get here.
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(
                    crate::inference::mem_bandwidth::measured_gbps().unwrap_or(0.0),
                    false,
                ),
            ) {
                if encrypted {
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
                        standbys: vec![],
                        tp_groups: vec![],
                        supports_speculative: true,
                    });
                }
            }
        }

        // Fast path: if the local node has full layer coverage (0..num_layers),
        // run entirely locally without involving remote peers.  This prevents
        // "Segment N failed with no standby" errors caused by remote peers that
        // hold overlapping shards being pulled into the pipeline unnecessarily.
        if let Some(local_cand) = candidates.iter().find(|c| {
            c.node_id == *local_node_id
                && c.available_ranges
                    .iter()
                    .any(|r| r.0 == 0 && r.1 >= num_layers)
        }) {
            tracing::info!(
                model = %model_id,
                num_layers,
                "Local node has full layer coverage — single local segment (no remote peers)"
            );
            let segment = PipelineSegment {
                node_id: local_node_id.clone(),
                shard_id: local_cand.shard_id.clone(),
                layer_range: (0, num_layers),
            };
            // No TP groups here — deliberately. When the local node already
            // holds every layer, pulling a LAN peer into a tensor-parallel
            // group can only make the request slower (2 × num_layers AllReduce
            // round trips replacing compute we were about to do anyway) and
            // adds a hard dependency on a peer we did not need. A peer that
            // stalls then fails the whole request with an AllReduce timeout,
            // even though this node could have answered alone.
            return Ok(PipelineAssignment {
                request_id,
                segments: vec![segment],
                standbys: vec![],
                tp_groups: vec![],
                supports_speculative: true,
            });
        }

        // Distributed layer assignment: prefer Parallax shortest-path DP when
        // enabled; fall back to greedy on any failure (disjoint ranges, no
        // valid source/sink, etc.) so routing never regresses below greedy.
        let raw_segments = if self.shared_state.config.inference.parallax_routing {
            match parallax::route_shortest_path(
                num_layers,
                &candidates,
                local_node_id,
                encrypted,
                // Encryption forces the first and last segments onto this node,
                // so an encrypted distributed pipeline is multi-segment by
                // construction — there is no single-delegation alternative to
                // lose. The per-token cost that keeps partial ranges off by
                // default therefore does not apply, and without them a peer
                // holding a SUPERSET of the middle (very commonly the whole
                // model) offers only one indivisible range, which can be neither
                // a middle segment nor a remote encrypted end. That produced a
                // hard "No node available" for a perfectly valid boomerang.
                self.shared_state.config.inference.parallax_partial_ranges || encrypted,
            ) {
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
                    segs
                }
                Err(e) => {
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
        let standbys = self.find_standbys(&segments, &candidates);

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

        tracing::info!(
            request_id = %request_id,
            model = %model_id,
            candidates_count = candidates.len(),
            segments = segments.len(),
            standbys = standbys.len(),
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
                self.shared_state.active_pipelines.len() as f32
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

            // Look up speed estimation from capability gossip
            let est_tokens_per_sec = if &node_id == local_node_id {
                // Local: compute directly from our GPU info
                self.shared_state
                    .gpu_info
                    .as_ref()
                    .map(|g| {
                        let bw =
                            crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
                        crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true)
                    })
                    .unwrap_or(0.0)
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
            let observed_latency_ms_per_layer =
                self.shared_state.observed_latency_ms_per_layer(&node_id);
            let gpu_vram_available_mb = if node_id == *local_node_id {
                // Never used for the local node — the loader's own admission
                // check is the authority on whether WE can fit a model, and it
                // knows what is already committed to live workers.
                None
            } else {
                self.shared_state.peer_registry.get(&node_id).and_then(|p| {
                    p.capability
                        .as_ref()
                        .and_then(|c| c.gpu.as_ref().map(|g| g.vram_available_mb))
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
                is_pool_member: is_pool,
                gpu_vram_available_mb,
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
        for c in &candidates {
            tracing::info!(
                node = %c.node_id,
                ranges = ?c.available_ranges,
                can_be_first = c.can_be_first,
                can_be_last = c.can_be_last,
                region_score = c.region_score,
                latency_ms = c.latency_ms,
                est_tokens_per_sec = c.est_tokens_per_sec,
                observed_ms_per_layer = ?c.observed_latency_ms_per_layer,
                load = c.load,
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
        let measured = entry.as_ref().and_then(|p| p.latency_ms);
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
    fn greedy_assign(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
        encrypted_pipeline: bool,
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        let mut segments = Vec::new();
        let mut current_layer = 0u32;
        let local_node_id = self.shared_state.identity.node_id();

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
                    segments.push(PipelineSegment {
                        node_id: candidate.node_id.clone(),
                        shard_id: candidate.shard_id.clone(),
                        layer_range: (current_layer, range.1),
                    });
                    current_layer = range.1;
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
        let local_tps = self
            .shared_state
            .gpu_info
            .as_ref()
            .map(|g| {
                let bw = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true)
            })
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
    ) -> Vec<PipelineSegment> {
        let mut standbys = Vec::new();

        let local_node_id = self.shared_state.identity.node_id();

        for segment in segments {
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
                la.cmp(&lb)
                    .then(region_cmp)
                    .then_with(|| a.latency_ms.cmp(&b.latency_ms))
            });
            if let Some(backup) = eligible.first() {
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
