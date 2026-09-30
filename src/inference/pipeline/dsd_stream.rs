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
/// (1-8).
fn chunks_in_flight() -> usize {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_SPEC_STREAM_WINDOW", 3, 8)) as usize
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
/// (`dsd_controller::best_gamma_overall`). The SERVING side has been on since
/// v0.3.213; an older far node advertises no `STREAMED_VERIFY` and gets rounds.
pub(super) fn stream_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| stream_switch(std::env::var("SWARMLLM_SPEC_STREAM").ok().as_deref()))
}

/// The switch's reading, apart from its `OnceLock` so a test can ask it twice.
fn stream_switch(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// The segment a stream's chunks go to: the request's LAST segment, when every
/// segment before it is this node's own — their forwards then leave here one
/// at a time, in order — and it is a peer that walks a check and serves a
/// stream (`features::STREAMED_VERIFY`). `None` keeps the rounds.
pub(super) fn stream_tail<'a>(
    state: &SharedState,
    segments: &'a [PipelineSegment],
) -> Option<&'a PipelineSegment> {
    let (tail, head) = segments.split_last()?;
    let me = state.identity.node_id();
    let head_is_ours = !head.is_empty() && head.iter().all(|s| s.node_id == *me);
    (head_is_ours
        && tail.node_id != *me
        && super::peer_walks_at_tail(state, &tail.node_id)
        && state.peer_advertises_feature(
            &tail.node_id,
            swarmllm_types::node::features::STREAMED_VERIFY,
        ))
    .then_some(tail)
}

/// Each stream's attempt tag: the high bits of its numbers
/// (`types::inference::stream_seq`). From a counter, not the request id, so a
/// router retry under the same id is a different stream (gotcha #749).
static NEXT_ATTEMPT: AtomicU32 = AtomicU32::new(0);

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
struct StreamWaiter {
    state: Arc<SharedState>,
    key: WaiterKey,
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
    head: Vec<PipelineSegment>,
    tail: PipelineSegment,
    attempt: u32,
    turn: u32,
    frontier: Frontier,
    drafter: &'a mut EngineDrafter,
    drafting_off: bool,
    drafter_warm: bool,
    /// The shared noise for the far segment's walk, where it samples with it.
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
        tail: PipelineSegment,
        drafter: &mut EngineDrafter,
        start_pos: u32,
        io: StreamIo<'_>,
        reply: StreamReply<'_>,
    ) -> Result<(), SwarmError> {
        let state = self.shared_state.clone();
        let head = self.assignment.segments[..self.assignment.segments.len() - 1].to_vec();
        let coupling = io.noise.map(|n| n.seed()).filter(|_| {
            state.peer_advertises_feature(
                &tail.node_id,
                swarmllm_types::node::features::COUPLED_SAMPLING,
            )
        });
        let history = reply.generated.clone();
        let mut run = StreamRun {
            exec: self,
            state,
            head,
            tail,
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

        // This node's own layers, one segment after another.
        let mut activations = super::pack_verify_tokens_to_le_bytes(&rows);
        for segment in &self.head {
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
                .forward_for_request(
                    forward,
                    None,
                    crate::inference::process_pool::Requester::Owner,
                )
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(crate::error::reclassify_flattened_error(msg)
                    .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
            }
            activations = result.activations;
        }

        let walk = TailWalk {
            drafts: &drafts,
            sampling,
            generated: &self.frontier.history,
            coupling: self.coupling,
        };
        let forward = super::build_spec_verify_forward(
            request_id,
            index_pos,
            activations,
            &self.tail,
            truncate,
            Some(&walk),
            Some(seq),
        );
        let Some(peer) = self.state.resolve_peer_id_bytes(&self.tail.node_id) else {
            return Err(SwarmError::Inference(format!(
                "streamed check: no route to segment holder {} — it left mid-request",
                self.tail.node_id
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
                awaiting: Some(self.tail.node_id.clone()),
                chain_members: Vec::new(),
                expects_step: Some(ExpectedStep::one(index_pos, self.tail.layer_range)),
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
        let num_layers = self.tail.layer_range.1 - self.tail.layer_range.0;
        let budget = SegmentBudget::for_forward(
            &self.state,
            &self.tail.node_id,
            &self.tail.shard_id.model_id,
            crate::daemon::state::WorkKind::Decode,
            num_layers,
            activation_bytes,
            ActivationUnits::HiddenStates,
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
            alone,
            started: checking,
        };
        let state = self.state.clone();
        let network_tx = exec.network_tx.clone();
        let node = self.tail.node_id.clone();
        let cancel = exec.request.cancel.clone();
        let segment_idx = self.head.len();
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

    /// Emit what a chunk's check kept and decide what the chunks after it
    /// are worth.
    async fn take_answer(&mut self, (sent, result): Answer) -> Result<Taken, SwarmError> {
        let result = result?;
        if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
            // The class does not survive the wire (gotcha #304).
            return Err(crate::error::reclassify_flattened_error(msg)
                .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
        }
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

        let mut emitted: Vec<u32> = kept.iter().copied().chain(std::iter::once(next)).collect();
        // Nothing past the first end-of-reply token reaches anyone (R105).
        if let Some(at) = emitted.iter().position(|t| self.io.eos.contains(t)) {
            emitted.truncate(at + 1);
        }
        let room = (self.io.max_tokens as usize).saturating_sub(self.reply.generated.len());
        emitted.truncate(room.max(1));
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

    #[test]
    fn the_stream_is_on_unless_switched_off() {
        assert!(stream_switch(None));
        assert!(stream_switch(Some("1")));
        assert!(!stream_switch(Some("0")));
        assert!(!stream_switch(Some("off")));
        assert!(!stream_switch(Some("false")));
    }
}
