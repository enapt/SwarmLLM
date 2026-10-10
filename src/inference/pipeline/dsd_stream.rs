//! Split speculation's continuous stream (`docs/plans/split_speculation.md`
//! § 4b): the next chunk of guesses is drafted, run through this node's own
//! layers and sent while the chunks before it are still being checked.
//!
//! A round of split speculation is strictly serial — the drafter guesses, this
//! node runs its layers over the guesses, the far segment checks them, and only
//! then does the next round start — so the drafting and the near half sit inside
//! every round trip. Measured on the emulated 24 ms link (2026-09-30, 7B split
//! 14/14 on one card): ~17 ms of drafting and 11 ms of near half in a ~75 ms
//! round that kept 2.35 tokens.
//!
//! **How a chunk is built on chunks not yet checked.** Each carries one guess
//! past the rows it sends — the LOOK-AHEAD — and the next chunk starts from it.
//! The far segment walks a chunk's rows exactly as it walks a round (keep the
//! guesses while its own sample agrees, answer with them and the token it
//! sampled where they parted, or after the last), so no new rule runs there:
//! the answer's last token IS its choice for the look-ahead's position. A chunk
//! whose guesses were all kept and whose look-ahead that token matches leaves
//! the chunks after it valid. Any other answer ends them: they are dropped
//! unanswered, the drafter is rewound to what was kept, and the next chunk
//! truncates both halves' caches back to it — the far segment runs the dropped
//! chunks first, since it runs a stream in order (`daemon::state::forward_streams`),
//! and they cost it work, never a wrong cache.
//!
//! **Where the guesses are walked.** A plan streams when ONE of its segments is
//! a peer's and every other is this node's ([`stream_shape`]). Where the peer
//! holds the model's last layers it walks each chunk, as above. Where this node
//! does — the boomerang a node holding both ends of a model runs by default
//! ("Start and finish on this computer"), or a peer holding the first layers —
//! the peer answers each chunk with hidden states and this node's last segment
//! walks it, as each answer is taken in order (FUTURE_WORK #152); the peer is
//! sent no guess, history or sampler, as on any other step of such a reply. A
//! chunk a restart drops is then never run on this side at all. PipeInfer's
//! head node samples the same way: the logits come back to it.
//!
//! PipeInfer (arXiv 2407.11798, SC'24) runs this scheme across MPI nodes; it
//! found small chunks (1-4 tokens) better than large ones once several are in
//! flight, which [`guesses_per_chunk`]'s default follows, and needs runs executed
//! in the order sent, which the far segment's gate gives. Its early cancellation
//! of runs already known to be wrong is not built: it pays where the far node
//! is saturated, and here a dropped chunk is computed while that node would
//! otherwise wait for the restarted one.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use futures::stream::{FuturesOrdered, StreamExt};
use futures::FutureExt;

use crate::daemon::state::{ExpectedStep, PendingLayerResult, SharedState, WaiterKey};
use crate::error::SwarmError;
use crate::inference::coupled_noise::CoupledNoise;
use crate::inference::dsd_controller::{AcceptanceEstimate, CheckCost, RecentMedian};
use crate::inference::router::StreamingTokenTx;
use crate::types::inference::stream_seq;
use crate::types::{LayerResult, NetworkCommand, NetworkFinishReason, PipelineSegment};

use super::engine_drafter::EngineDrafter;
use super::local::{ActivationUnits, ResendOnRefusal, SegmentBudget};
use super::{prompt, PipelineExecutor, TailWalk, VerifyReply};

/// Guesses per chunk, the look-ahead included: 3 checks two rows past the
/// bootstrap and guesses one more. `SWARMLLM_SPEC_STREAM_GUESSES` (1-8).
fn guesses_per_chunk() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_SPEC_STREAM_GUESSES", 3, 8))
}

/// Chunks out at once. Enough to cover a round trip with chunks being built
/// (~20 ms each on the rig against ~45-70 ms to an answer); each one past that
/// is only more work thrown away at a refusal. `SWARMLLM_SPEC_STREAM_WINDOW`
/// (1-8 — `forward_streams::MAX_STREAM_WINDOW`, which the serving side sizes
/// its per-stream admission from).
fn chunks_in_flight() -> usize {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        knob(
            "SWARMLLM_SPEC_STREAM_WINDOW",
            3,
            crate::daemon::state::forward_streams::MAX_STREAM_WINDOW,
        )
    }) as usize
}

fn knob(name: &str, default: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(default, |v| v.clamp(1, max))
}

/// Whether a coordinator streams its checks: ON unless `SWARMLLM_SPEC_STREAM=0`
/// (or `off` / `false`), which keeps the rounds. Measured over a real link on
/// 2026-10-01 — this node's card holding the near half, a processor in Italy
/// the far half, ~270 ms apart, the release binary with the switch as the only
/// difference: +20% / +51% / +57% decode over rounds on prose, an explanation
/// and code (+25-133% at temperature 0.7), replies scored against llama.cpp
/// like the rounds' and plain decoding's. A rig with both halves on ONE card
/// reads it ~15% slower (gotcha #759) — two processes contending for one
/// device, not a deployment. Where guessing does not pay at all, a streamed
/// request remembers the rounds' verdict of zero and the next one steps aside
/// (`dsd_controller::best_gamma_overall`). A far node is streamed to only from
/// v0.3.216 (`features::STREAM_AS_ONE_WORK`); an older one gets rounds — see
/// [`stream_shape`].
pub(super) fn stream_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| stream_switch(std::env::var("SWARMLLM_SPEC_STREAM").ok().as_deref()))
}

/// The switch's reading, apart from its `OnceLock` so a test can ask it twice.
fn stream_switch(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// A plan a stream can run on: ONE segment on a peer, every other one this
/// node's own. Their forwards then leave here one at a time, in order, and the
/// one peer runs the stream in that order (`daemon::state::forward_streams`).
pub(super) struct StreamShape {
    /// This node's segments before the peer's — none when the peer holds the
    /// model's first layers.
    near: Vec<PipelineSegment>,
    /// The peer's segment.
    far: PipelineSegment,
    /// This node's segments after the peer's — the boomerang's tail. When there
    /// are any, the guesses are walked HERE, by the last of them, and the peer
    /// answers each chunk with hidden states (FUTURE_WORK #152).
    after: Vec<PipelineSegment>,
}

impl StreamShape {
    /// Whether the peer samples, and so walks each chunk's guesses: only when
    /// it holds the model's last layers.
    fn far_walks(&self) -> bool {
        self.after.is_empty()
    }

    /// For the log line that says which shape a request streamed on.
    fn name(&self) -> &'static str {
        match (self.near.is_empty(), self.after.is_empty()) {
            (false, true) => "peer-last",
            (false, false) => "boomerang",
            (true, false) => "peer-first",
            (true, true) => "peer-only",
        }
    }
}

/// The shape a request's checks stream on, or `None` to keep the rounds.
///
/// The peer must serve a stream as one piece of work: `STREAMED_VERIFY` alone
/// is not enough — v0.3.213-v0.3.215 serve a stream but count each chunk
/// against their per-peer cap of 4, and a chunk refused there stalled the rest
/// of the stream for 60 s (`features::STREAM_AS_ONE_WORK`). And where the peer
/// holds the last layers it must walk a check; where this node does, the walk
/// is ours and the peer only computes its layers, which every streaming peer
/// does — the serving side has never cared which segment a stream is
/// (`forward_streams` is keyed by request, layer range and attempt).
pub(super) fn stream_shape(
    state: &SharedState,
    segments: &[PipelineSegment],
) -> Option<StreamShape> {
    use swarmllm_types::node::features::{STREAMED_VERIFY, STREAM_AS_ONE_WORK};
    shape_of(segments, state.identity.node_id(), |peer, last| {
        state.peer_advertises_feature(peer, STREAMED_VERIFY | STREAM_AS_ONE_WORK)
            && (!last || super::peer_walks_at_tail(state, peer))
    })
}

/// [`stream_shape`]'s decision, apart from the state it asks: `serves(peer,
/// last)` — does that peer serve a stream, and walk one when it holds the last
/// layers.
fn shape_of(
    segments: &[PipelineSegment],
    me: &crate::types::NodeId,
    serves: impl Fn(&crate::types::NodeId, bool) -> bool,
) -> Option<StreamShape> {
    let mut peers = segments
        .iter()
        .enumerate()
        .filter(|(_, s)| s.node_id != *me);
    let (at, far) = peers.next()?;
    // Two peers' segments would be two streams to keep in step, and a chunk
    // forwarded from one to the other would leave this node's order.
    if peers.next().is_some() {
        return None;
    }
    let shape = StreamShape {
        near: segments[..at].to_vec(),
        far: far.clone(),
        after: segments[at + 1..].to_vec(),
    };
    // A plan that is the peer's alone is a hand-off, never driven from here.
    let ours = !shape.near.is_empty() || !shape.after.is_empty();
    (ours && serves(&far.node_id, shape.far_walks())).then_some(shape)
}

/// Each stream's attempt tag: the high bits of its numbers
/// (`types::inference::stream_seq`). From a counter, not the request id, so a
/// router retry under the same id is a different stream (gotcha #749).
pub(super) static NEXT_ATTEMPT: AtomicU32 = AtomicU32::new(0);

/// What a request's reply is made of, and what the next request on the same
/// machines learns from this one (`dsd_controller::remember`).
pub(super) struct StreamReply<'a> {
    pub generated: &'a mut Vec<u32>,
    pub finish_reason: &'a mut String,
    pub acceptance: &'a mut AcceptanceEstimate,
    pub check: &'a mut CheckCost,
    pub draft_cost: &'a mut RecentMedian,
    pub proposed: &'a mut u32,
    pub accepted: &'a mut u32,
}

/// Where emitted tokens go and when the reply ends.
pub(super) struct StreamIo<'a> {
    pub token_tx: &'a Option<StreamingTokenTx>,
    pub decoder: &'a prompt::CachedDecoder,
    pub eos: &'a HashSet<u32>,
    pub noise: Option<CoupledNoise>,
    pub max_tokens: u32,
}

/// A chunk out being checked.
struct Sent {
    seq: u32,
    index_pos: u32,
    rows: usize,
    /// The guesses the far segment walks: the rows after the bootstrap.
    drafts: Vec<u32>,
    /// The guess past the rows, which the far segment's own next token
    /// confirms or refutes; `None` once the drafter has stopped.
    lookahead: Option<u32>,
    /// The reply as the walk read it, through the bootstrap.
    history: Vec<u32>,
    /// The cut it carried — applied by the peer and this node's near segments
    /// as it was sent, and by this node's segments after the peer's as its
    /// answer is taken.
    truncate: Option<u32>,
    /// What the peer was sent, where its answer is fed on to this node's
    /// segments after it: the answer is checked against it before anything
    /// here runs on it.
    far_input: Option<Vec<u8>>,
    /// Nothing else was out when it was built, so its time is a clean sample
    /// of what a check costs (`CheckCost`).
    alone: bool,
    started: std::time::Instant,
}

type Answer = (Sent, Result<LayerResult, SwarmError>);
type Waiting = Pin<Box<dyn Future<Output = Answer> + Send>>;

/// A streamed chunk's wait on its answer, withdrawn when dropped — answered,
/// abandoned by a restart, or the reply ending — so a restart drops every
/// chunk after the refused one with nothing left behind in the map.
pub(super) struct StreamWaiter {
    pub(super) state: Arc<SharedState>,
    pub(super) key: WaiterKey,
}

impl Drop for StreamWaiter {
    fn drop(&mut self) {
        self.state.pending_layer_results.remove(&self.key);
    }
}

/// Where the next chunk starts.
struct Frontier {
    /// The position of its bootstrap — the reply's last token so far, kept or
    /// guessed.
    index_pos: u32,
    /// The reply through that bootstrap, guesses taken as kept included.
    history: Vec<u32>,
    /// A restart's cut, for both halves' caches, on the next chunk.
    truncate: Option<u32>,
    /// Nothing is built past this chunk until it is answered: it has no
    /// look-ahead (the drafter stopped), or its look-ahead is a guess the
    /// drafter was unsure of — it stopped short of the guesses asked for
    /// (`draft_confidence_floor`), so a chunk built on it would most likely be
    /// thrown away. PipeInfer's "reactive speculation" draws the same line.
    held_by: Option<u32>,
}

enum Taken {
    /// The chunks out are still good.
    Continue,
    /// Every chunk out was built on a refused guess.
    Restart,
    /// The reply is over.
    Done,
}

struct StreamRun<'a, 'r> {
    exec: &'a PipelineExecutor,
    state: Arc<SharedState>,
    shape: StreamShape,
    attempt: u32,
    turn: u32,
    frontier: Frontier,
    drafter: &'a mut EngineDrafter,
    drafting_off: bool,
    drafter_warm: bool,
    /// The shared noise for the walk — this node's own, or the peer's where it
    /// walks and samples with it.
    coupling: Option<u64>,
    io: StreamIo<'a>,
    reply: StreamReply<'r>,
    chunks: u32,
    kept_whole: u32,
    restarts: u32,
}

impl PipelineExecutor {
    /// Check the reply's guesses as a stream (module doc) until it ends, from
    /// the prompt pass's token at `start_pos`, the last of `reply.generated`.
    pub(super) async fn stream_checks(
        &self,
        shape: StreamShape,
        drafter: &mut EngineDrafter,
        start_pos: u32,
        io: StreamIo<'_>,
        reply: StreamReply<'_>,
    ) -> Result<(), SwarmError> {
        let state = self.shared_state.clone();
        // Our own worker always samples with the shared noise; a peer only
        // when it says so.
        let coupling = io.noise.map(|n| n.seed()).filter(|_| {
            !shape.far_walks()
                || state.peer_advertises_feature(
                    &shape.far.node_id,
                    swarmllm_types::node::features::COUPLED_SAMPLING,
                )
        });
        let shape_name = shape.name();
        let history = reply.generated.clone();
        let mut run = StreamRun {
            exec: self,
            state,
            shape,
            attempt: NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed) & stream_seq::MAX_ATTEMPT,
            turn: 0,
            frontier: Frontier {
                index_pos: start_pos,
                history,
                truncate: None,
                held_by: None,
            },
            drafter,
            drafting_off: false,
            drafter_warm: false,
            coupling,
            io,
            reply,
            chunks: 0,
            kept_whole: 0,
            restarts: 0,
        };
        let outcome = run.run().await;
        tracing::info!(
            request_id = %self.request.id,
            shape = shape_name,
            chunks = run.chunks,
            kept_whole = run.kept_whole,
            restarts = run.restarts,
            guesses_per_chunk = guesses_per_chunk(),
            window = chunks_in_flight(),
            "DSD: streamed checks complete"
        );
        outcome
    }
}

impl StreamRun<'_, '_> {
    async fn run(&mut self) -> Result<(), SwarmError> {
        let window = chunks_in_flight();
        let mut out: FuturesOrdered<Waiting> = FuturesOrdered::new();
        loop {
            if self.exec.request.is_cancelled() {
                tracing::info!(
                    request_id = %self.exec.request.id,
                    "DIAG: DSD inference cancelled externally"
                );
                *self.reply.finish_reason = "stop".to_string();
                return Ok(());
            }
            // An answer already in goes first: it may end the chunks the next
            // one would be built on.
            let ready = match out.next().now_or_never() {
                Some(Some(answer)) => Some(answer),
                _ => None,
            };
            let answer = match ready {
                Some(answer) => answer,
                None => {
                    let may_build = self.frontier.held_by.is_none()
                        && out.len() < window
                        && (self.frontier.history.len() as u32) < self.io.max_tokens;
                    if may_build {
                        let chunk = self.send_chunk(out.is_empty()).await?;
                        out.push_back(chunk);
                        continue;
                    }
                    match out.next().await {
                        Some(answer) => answer,
                        // Nothing out and nothing to build: the reply reached
                        // its length.
                        None => return Ok(()),
                    }
                }
            };
            match self.take_answer(answer).await? {
                Taken::Continue => {}
                // Dropping them withdraws their waits; their late answers
                // find none.
                Taken::Restart => out = FuturesOrdered::new(),
                Taken::Done => return Ok(()),
            }
        }
    }

    /// Guess, run this node's layers over the chunk, send it to the far
    /// segment, and hand back the wait on its answer.
    async fn send_chunk(&mut self, alone: bool) -> Result<Waiting, SwarmError> {
        let exec = self.exec;
        let request_id = exec.request.id;
        let sampling = &exec.request.sampling_params;
        let started = std::time::Instant::now();
        let room = self
            .io
            .max_tokens
            .saturating_sub(self.frontier.history.len() as u32);
        let want = guesses_per_chunk().min(room);
        let guesses: Vec<u32> = if self.drafting_off || want == 0 {
            Vec::new()
        } else {
            match self
                .drafter
                .draft(
                    &self.state,
                    want,
                    sampling,
                    &self.frontier.history,
                    self.io.noise.map(|n| n.seed()),
                    exec.request.cancel.clone(),
                )
                .await
            {
                Ok(guesses) => {
                    // Taken as kept: the next chunk is drafted on top of them,
                    // and `rewind` undoes it at a refusal.
                    self.drafter.settle(guesses.len());
                    self.drafter.push(&guesses);
                    // The first call reads the prompt too, when the read-ahead
                    // did not, and is not a guess's cost (`dsd`'s rule).
                    if self.drafter_warm && !guesses.is_empty() {
                        self.reply.draft_cost.record(
                            started.elapsed().as_secs_f64() * 1000.0 / guesses.len() as f64,
                        );
                    }
                    self.drafter_warm = true;
                    guesses
                }
                Err(e) => {
                    tracing::warn!(
                        %request_id,
                        error = %e,
                        "DSD: the drafter failed — finishing this reply without guessing ahead"
                    );
                    self.drafting_off = true;
                    Vec::new()
                }
            }
        };
        let (drafts, lookahead) = match guesses.split_last() {
            Some((&last, rest)) => (rest.to_vec(), Some(last)),
            None => (Vec::new(), None),
        };
        let bootstrap = *self
            .frontier
            .history
            .last()
            .expect("a reply holds its first token before any check");
        let rows: Vec<u32> = std::iter::once(bootstrap)
            .chain(drafts.iter().copied())
            .collect();
        let index_pos = self.frontier.index_pos;
        let truncate = self.frontier.truncate.take();
        // A check's cost is timed from here, drafting excluded — the rounds'
        // `CheckCost` measures the same span.
        let checking = std::time::Instant::now();
        let seq = stream_seq::compose(self.attempt, self.turn).ok_or_else(|| {
            SwarmError::Internal("a streamed reply ran past its checks' numbering".into())
        })?;
        self.turn += 1;

        // This node's own layers before the peer's, one segment after another.
        let mut activations = super::pack_verify_tokens_to_le_bytes(&rows);
        for segment in &self.shape.near {
            let forward = super::build_spec_verify_forward(
                request_id,
                index_pos,
                activations,
                segment,
                truncate,
                None,
                Some(seq),
            );
            let result = self
                .state
                .model_process_pool
                .forward_for_request(forward, None, super::worker_requester(&exec.request))
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(crate::error::reclassify_flattened_error(msg)
                    .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
            }
            activations = result.activations;
        }

        // The walk goes to the peer only where it samples. In the boomerang it
        // is sent no guess, no history and no sampler — hidden states alone,
        // as on every other step of a reply that starts and finishes here.
        let walk = TailWalk {
            drafts: &drafts,
            sampling,
            generated: &self.frontier.history,
            coupling: self.coupling,
        };
        let far = &self.shape.far;
        let far_input = (!self.shape.far_walks()).then(|| activations.clone());
        let forward = super::build_spec_verify_forward(
            request_id,
            index_pos,
            activations,
            far,
            truncate,
            self.shape.far_walks().then_some(&walk),
            Some(seq),
        );
        let Some(peer) = self.state.resolve_peer_id_bytes(&far.node_id) else {
            return Err(SwarmError::Inference(format!(
                "streamed check: no route to segment holder {} — it left mid-request",
                far.node_id
            )));
        };
        if self.state.pending_layer_results.len() >= super::MAX_PENDING_LAYER_RESULTS {
            return Err(SwarmError::ServiceUnavailable(
                "Pipeline overloaded — too many pending layer results".into(),
            ));
        }
        let key = WaiterKey::streamed(request_id, seq);
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.state.pending_layer_results.insert(
            key,
            PendingLayerResult {
                tx,
                awaiting: Some(far.node_id.clone()),
                chain_members: Vec::new(),
                expects_step: Some(ExpectedStep::one(index_pos, far.layer_range)),
            },
        );
        let waiter = StreamWaiter {
            state: self.state.clone(),
            key,
        };
        let activation_bytes = forward.activations.len();
        if exec
            .network_tx
            .send(NetworkCommand::SendTensor {
                target_peer_bytes: peer.clone(),
                forward: forward.clone(),
            })
            .await
            .is_err()
        {
            return Err(SwarmError::Network("streamed check: send dropped".into()));
        }
        let num_layers = far.layer_range.1 - far.layer_range.0;
        let budget = SegmentBudget::for_forward(
            &self.state,
            &far.node_id,
            &far.shard_id.model_id,
            crate::daemon::state::WorkKind::Decode,
            num_layers,
            activation_bytes,
            // A peer holding the first layers is sent the packed guesses.
            if self.shape.near.is_empty() {
                ActivationUnits::PromptBytes
            } else {
                ActivationUnits::HiddenStates
            },
        );

        let history = self.frontier.history.clone();
        if lookahead.is_some() {
            self.frontier.index_pos += rows.len() as u32;
            self.frontier.history.extend_from_slice(&guesses);
        }
        if lookahead.is_none() || (guesses.len() as u32) < want {
            self.frontier.held_by = Some(seq);
        }
        let sent = Sent {
            seq,
            index_pos,
            rows: rows.len(),
            drafts,
            lookahead,
            history,
            truncate,
            far_input,
            alone,
            started: checking,
        };
        let state = self.state.clone();
        let network_tx = exec.network_tx.clone();
        let node = far.node_id.clone();
        let cancel = exec.request.cancel.clone();
        let segment_idx = self.shape.near.len();
        Ok(Box::pin(async move {
            let _waiter = waiter;
            let rebuild = move || forward.clone();
            let result = PipelineExecutor::wait_for_result(
                &state,
                rx,
                request_id,
                segment_idx,
                &node,
                num_layers,
                activation_bytes,
                budget,
                cancel.as_ref(),
                ResendOnRefusal::SameForward {
                    network_tx: &network_tx,
                    target_peer_bytes: &peer,
                    rebuild: &rebuild,
                },
            )
            .await;
            (sent, result)
        }))
    }

    /// The boomerang's last leg: this node's segments after the peer's, over
    /// the peer's answer to `sent`, the last of them walking its guesses.
    ///
    /// Run here, as each answer is taken in order — never as answers arrive.
    /// Two forwards of one request at a worker at once would cross their
    /// replies (gotcha #180), and a chunk whose answer is dropped by a restart
    /// is then never run on this side at all, so nothing but the cut the next
    /// chunk carries has to be undone here.
    ///
    /// `&mut self` only so the future is `Send`: the drafter this run borrows
    /// is not `Sync`, so a shared borrow of the run cannot cross an await.
    async fn finish_here(
        &mut self,
        sent: &Sent,
        far_input: &[u8],
        far: LayerResult,
    ) -> Result<LayerResult, SwarmError> {
        let exec = self.exec;
        super::check_intermediate_activations(self.shape.near.len(), far_input, &far.activations)
            .map_err(|e| SwarmError::Inference(format!("streamed check: {e}")))?;
        let walk = TailWalk {
            drafts: &sent.drafts,
            sampling: &exec.request.sampling_params,
            generated: &sent.history,
            coupling: self.coupling,
        };
        let mut activations = far.activations;
        // Not empty: only a shape with segments after the peer's feeds on.
        let last = self.shape.after.len() - 1;
        for (i, segment) in self.shape.after.iter().enumerate() {
            let is_last = i == last;
            let forward = super::build_spec_verify_forward(
                exec.request.id,
                sent.index_pos,
                activations,
                segment,
                sent.truncate,
                is_last.then_some(&walk),
                Some(sent.seq),
            );
            let result = self
                .state
                .model_process_pool
                .forward_for_request(forward, None, super::worker_requester(&exec.request))
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(crate::error::reclassify_flattened_error(msg)
                    .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
            }
            if is_last {
                return Ok(result);
            }
            activations = result.activations;
        }
        unreachable!("the loop returns at the last segment after the peer's")
    }

    /// Emit what a chunk's check kept and decide what the chunks after it
    /// are worth.
    async fn take_answer(&mut self, (sent, result): Answer) -> Result<Taken, SwarmError> {
        let result = result?;
        if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
            // The class does not survive the wire (gotcha #304).
            return Err(crate::error::reclassify_flattened_error(msg)
                .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
        }
        let result = match &sent.far_input {
            Some(far_input) => self.finish_here(&sent, far_input, result).await?,
            None => result,
        };
        let answer = if !result.spec_logits.is_empty() {
            VerifyReply::Logits(result.spec_logits)
        } else if !result.token_ids.is_empty() {
            VerifyReply::Walked(result.token_ids)
        } else {
            return Err(SwarmError::Inference(
                "streamed check: the last segment returned neither logits nor a walk".into(),
            ));
        };
        let (kept, next, all) = answer.accept(
            &sent.drafts,
            &self.exec.request.sampling_params,
            &sent.history,
            self.io
                .noise
                .as_ref()
                .map(|n| (n, u64::from(sent.index_pos) + 1)),
        )?;
        // The far segment's own token at the look-ahead's position is `next`
        // when it kept every row's guess.
        let whole = all && sent.lookahead == Some(next);

        let guessed = sent.drafts.len() + usize::from(sent.lookahead.is_some());
        if guessed > 0 {
            let kept_guesses = kept.len() + usize::from(whole);
            *self.reply.proposed += guessed as u32;
            *self.reply.accepted += kept_guesses as u32;
            self.reply
                .acceptance
                .record(kept_guesses as u32, guessed as u32);
        }
        if sent.alone {
            self.reply.check.record(
                sent.rows as u32,
                sent.started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        self.chunks += 1;
        if whole {
            self.kept_whole += 1;
        }

        // Nothing past the first end-of-reply token reaches anyone (R105), and
        // nothing past the budget.
        let emitted = super::round_tokens_for_reply(
            &kept,
            next,
            self.io.eos,
            (self.io.max_tokens as usize).saturating_sub(self.reply.generated.len()),
        );
        super::emit_streaming_batch(
            &self.exec.partial_reply,
            self.io.token_tx,
            self.io.decoder,
            &emitted,
            self.io.eos,
            self.reply.finish_reason,
        )
        .await;
        if !self.reply.finish_reason.is_empty() {
            return Ok(Taken::Done);
        }
        self.reply.generated.extend_from_slice(&emitted);
        let last = *emitted.last().expect("a check answers at least one token");
        if self.io.eos.contains(&last) {
            *self.reply.finish_reason = "stop".to_string();
            return Ok(Taken::Done);
        }
        if self.reply.generated.len() as u32 >= self.io.max_tokens {
            return Ok(Taken::Done);
        }
        if whole {
            // Its look-ahead was right: the frontier already stands on it.
            if self.frontier.held_by == Some(sent.seq) {
                self.frontier.held_by = None;
            }
            return Ok(Taken::Continue);
        }
        // Everything built after this chunk stood on a guess that was not
        // kept. `last` sits at `next_pos`, not yet in any cache: the next
        // chunk starts from it and cuts both halves back to it.
        let next_pos = sent.index_pos + emitted.len() as u32;
        self.drafter.rewind(next_pos as usize);
        self.drafter.push(&[last]);
        self.frontier = Frontier {
            index_pos: next_pos,
            history: self.reply.generated.clone(),
            truncate: Some(next_pos),
            held_by: None,
        };
        Ok(match sent.lookahead {
            Some(_) => {
                self.restarts += 1;
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    turn = stream_seq::turn(sent.seq),
                    kept = kept.len(),
                    of = guessed,
                    "DSD: a streamed check refused a guess — restarting after it"
                );
                Taken::Restart
            }
            // A chunk without a look-ahead had nothing built after it.
            None => Taken::Continue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{ModelId, NodeId, ShardId};

    const ME: NodeId = NodeId([1; 32]);
    const PEER: NodeId = NodeId([2; 32]);
    const OTHER: NodeId = NodeId([3; 32]);

    fn plan(nodes: &[NodeId]) -> Vec<PipelineSegment> {
        nodes
            .iter()
            .enumerate()
            .map(|(i, n)| PipelineSegment {
                node_id: n.clone(),
                shard_id: ShardId {
                    model_id: ModelId("m".into()),
                    index: i as u32,
                },
                layer_range: (i as u32 * 4, i as u32 * 4 + 4),
            })
            .collect()
    }

    /// A peer that serves a stream but cannot walk a check.
    fn serves_without_walking(_: &NodeId, last: bool) -> bool {
        !last
    }

    fn shape(nodes: &[NodeId], serves: impl Fn(&NodeId, bool) -> bool) -> Option<&'static str> {
        shape_of(&plan(nodes), &ME, serves).map(|s| s.name())
    }

    /// FUTURE_WORK #152: the shapes the swarm makes stream — the boomerang
    /// above all (58 of 78 holdings held both ends, 2026-10-01) — and each
    /// keeps its layers where the plan put them.
    #[test]
    fn every_plan_with_one_peer_segment_streams() {
        let every = |_: &NodeId, _: bool| true;
        assert_eq!(shape(&[ME, PEER], every), Some("peer-last"));
        assert_eq!(shape(&[ME, PEER, ME], every), Some("boomerang"));
        assert_eq!(shape(&[PEER, ME], every), Some("peer-first"));
        assert_eq!(shape(&[ME, ME, PEER, ME], every), Some("boomerang"));
        let s = shape_of(&plan(&[ME, ME, PEER, ME]), &ME, every).unwrap();
        assert_eq!(s.near.len(), 2);
        assert_eq!(s.far.layer_range, (8, 12));
        assert_eq!(s.after.len(), 1);
        assert!(
            !s.far_walks(),
            "this node holds the last layers: the walk is ours"
        );
    }

    /// The walk is asked of the peer only where it holds the last layers: a
    /// peer that cannot walk still serves the boomerang's middle, and a peer
    /// that serves no stream serves none.
    #[test]
    fn a_peer_is_asked_to_walk_only_where_it_holds_the_last_layers() {
        assert_eq!(shape(&[ME, PEER], serves_without_walking), None);
        assert_eq!(
            shape(&[ME, PEER, ME], serves_without_walking),
            Some("boomerang")
        );
        assert_eq!(
            shape(&[PEER, ME], serves_without_walking),
            Some("peer-first")
        );
        assert_eq!(shape(&[ME, PEER, ME], |_, _| false), None);
    }

    /// Two peers' segments, a plan that is one peer's alone, or no peer at all:
    /// the rounds (or the hand-off, or local generation) run it.
    #[test]
    fn a_plan_that_is_not_one_peer_segment_beside_ours_does_not_stream() {
        let every = |_: &NodeId, _: bool| true;
        assert_eq!(shape(&[ME, PEER, OTHER], every), None);
        assert_eq!(shape(&[PEER, ME, PEER], every), None);
        assert_eq!(shape(&[PEER], every), None);
        assert_eq!(shape(&[ME, ME], every), None);
    }

    #[test]
    fn the_stream_is_on_unless_switched_off() {
        assert!(stream_switch(None));
        assert!(stream_switch(Some("1")));
        assert!(!stream_switch(Some("0")));
        assert!(!stream_switch(Some("off")));
        assert!(!stream_switch(Some("false")));
    }
}
