//! A split reads its prompt in pieces, every machine at once (FUTURE_WORK #171,
//! `docs/plans/pipelined_prompt_pass.md`) — the coordinator's half.
//!
//! A split's prompt pass ran one machine at a time: the head read every
//! position through its layers, the hidden states crossed, the next segment
//! read them all. Here the positions are cut into pieces and every segment has
//! a driver of its own, run concurrently: a driver takes its inputs in order
//! from the segment before, runs its segment on each, and hands each output on
//! the moment it exists — so the peer reads piece k while the head reads piece
//! k+1, and the pass approaches its slowest stage instead of the sum. This is
//! sequence pipeline parallelism (Medha/Mnemosyne, arXiv 2409.17264), and
//! llama.cpp's micro-batched prompt across GPUs (#6017); their measured point
//! that carries over is that a piece costs little even when small, so the size
//! is chosen for overlap.
//!
//! Each piece is a forward of the prompt pass (`sequence_num` 0) carrying the
//! pass's span (`LayerForward::prompt_span`, the `0x0E` trailer): the receiving
//! worker starts the pass at the first piece — clears, restores what #10 kept,
//! is admitted for the whole span — and continues it at every later one. A
//! peer's pieces travel as a numbered stream with an attempt tag of their own
//! (`forward_streams` runs them there in order; this node's waits never meet),
//! at most [`PEER_WINDOW`] out at once so the peer never idles a round trip
//! between pieces. Unchained: every piece comes back here.
//!
//! **A failure is the old pass, once** — the caller's decision: any piece
//! failing stops new pieces, the ones already out are waited for (so none
//! reaches a worker beside the pass that follows), and the caller runs the
//! prompt pass whole, from its first position, with its failover.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use futures::stream::{FuturesOrdered, StreamExt};
use tokio::sync::mpsc;

use crate::daemon::state::{ExpectedStep, PendingLayerResult, WaiterKey};
use crate::error::SwarmError;
use crate::types::inference::stream_seq;
use crate::types::{
    LayerForward, LayerResult, NetworkCommand, NodeId, PipelineSegment, PromptSpan, TensorFormat,
};

use super::dsd_stream::{StreamWaiter, NEXT_ATTEMPT};
use super::local::{ActivationUnits, ResendOnRefusal, SegmentBudget};
use super::PipelineExecutor;

/// Positions in a piece unless `SWARMLLM_PROMPT_CHUNK_TOKENS` says otherwise.
/// Large enough that a piece is several of the worker's own prompt chunks
/// (`prefill_chunk_tokens`, 128) and its round trip is small beside its
/// compute; small enough that a 2,000-token prompt is four pieces.
const DEFAULT_PIECE_TOKENS: u32 = 512;

/// The smallest piece the setting may ask for — Medha's efficient floor is ~40
/// positions with grouped-query attention; past it a piece is mostly overhead.
const MIN_PIECE_TOKENS: u32 = 64;

/// The fewest pieces worth the machinery: one piece is the old pass.
const MIN_PIECES: u32 = 2;

/// The most pieces a pass is cut into: a longer prompt gets longer pieces, so
/// the round trips a pass pays stay bounded.
const MAX_PIECES: u32 = 16;

/// Pieces of one pass out to a peer at once. Two keep it busy — the next is
/// queued behind the one running (`forward_streams`) — without queueing
/// hidden states there that a failure would leave behind.
const PEER_WINDOW: usize = 2;

/// `SWARMLLM_PROMPT_CHUNKS=0` reads every prompt pass whole, as before — the
/// A/B arm. Read once.
pub(super) fn switched_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_PROMPT_CHUNKS").as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        )
    })
}

/// Positions in a piece: `SWARMLLM_PROMPT_CHUNK_TOKENS`, never below
/// [`MIN_PIECE_TOKENS`]. Read once.
fn piece_tokens() -> u32 {
    static TOKENS: OnceLock<u32> = OnceLock::new();
    *TOKENS.get_or_init(|| {
        std::env::var("SWARMLLM_PROMPT_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(DEFAULT_PIECE_TOKENS)
            .max(MIN_PIECE_TOKENS)
    })
}

/// The pieces of a pass over `start..end`, as `(from, to)` position ranges —
/// empty when the pass is shorter than [`MIN_PIECES`] pieces of `piece_tokens`.
/// At most [`MAX_PIECES`]: a longer pass gets longer pieces.
pub(super) fn pieces(start: u32, end: u32, piece_tokens: u32) -> Vec<(u32, u32)> {
    let positions = end.saturating_sub(start);
    let piece = piece_tokens.max(MIN_PIECE_TOKENS);
    if positions < MIN_PIECES.saturating_mul(piece) {
        return Vec::new();
    }
    let size = piece.max(positions.div_ceil(MAX_PIECES));
    let mut out = Vec::new();
    let mut at = start;
    while at < end {
        let to = at.saturating_add(size).min(end);
        out.push((at, to));
        at = to;
    }
    out
}

/// Can a plan's prompt pass be read in pieces? Two or more segments, no
/// machine twice — a worker holding two segments of one request would cross
/// their replies (gotcha #180), so the boomerang stays whole — every peer reads
/// pieces (`serves`; this node's own segments always do), and this node is on
/// every boundary between two segments.
///
/// The last condition because pieces come back HERE: between two peers a whole
/// pass is chained straight from one to the other, and a coordinator far from
/// two close peers would relay every piece across the long link twice. The
/// shapes this leaves are this node's segment beside a peer's — every split it
/// leads, the delegated head of #143 included.
fn shape_reads_in_pieces(
    segments: &[PipelineSegment],
    me: &NodeId,
    serves: impl Fn(&NodeId) -> bool,
) -> bool {
    if segments.len() < 2 {
        return false;
    }
    let mut seen = std::collections::HashSet::new();
    segments
        .iter()
        .all(|s| seen.insert(&s.node_id) && (s.node_id == *me || serves(&s.node_id)))
        && segments
            .windows(2)
            .all(|pair| pair[0].node_id == *me || pair[1].node_id == *me)
}

/// What one segment's driver gives back: the final piece's answer where the
/// segment samples, nothing where it hands its output on.
type DriverResult = Result<Option<LayerResult>, SwarmError>;

/// A piece out to a peer: which piece, and its answer.
type PieceWait<'a> =
    Pin<Box<dyn Future<Output = (usize, Result<LayerResult, SwarmError>)> + Send + 'a>>;

impl PipelineExecutor {
    /// May this plan's prompt pass be read in pieces — the plan's half of the
    /// question, before any prompt is tokenized.
    fn plan_reads_in_pieces(&self) -> bool {
        use swarmllm_types::node::features::{PROMPT_CHUNKS, STREAMED_VERIFY, STREAM_AS_ONE_WORK};
        switched_on()
            && self.assignment.tp_groups.is_empty()
            && shape_reads_in_pieces(
                &self.assignment.segments,
                self.shared_state.identity.node_id(),
                |peer| {
                    self.shared_state.peer_advertises_feature(
                        peer,
                        PROMPT_CHUNKS | STREAMED_VERIFY | STREAM_AS_ONE_WORK,
                    )
                },
            )
    }

    /// Should a pass computing `positions` positions be read in pieces?
    pub(super) fn reads_in_pieces(&self, positions: u32) -> bool {
        self.plan_reads_in_pieces() && !pieces(0, positions, piece_tokens()).is_empty()
    }

    /// The prompt's token ids, tokenized as a first segment would tokenize the
    /// text it is sent (`SharedState::standalone_tokenizer`, the same header);
    /// `None` without a tokenizer here or for bytes that are not text.
    pub(super) fn tokenize_prompt(&self, prompt_bytes: &[u8]) -> Option<Vec<u32>> {
        let tokenizer = self
            .shared_state
            .standalone_tokenizer(&self.assignment.segments.first()?.shard_id.model_id)?;
        let text = std::str::from_utf8(prompt_bytes).ok()?;
        Some(
            tokenizer
                .encode(text)
                .into_iter()
                .map(|t| t as u32)
                .collect(),
        )
    }

    /// The prompt's token ids, when this pass should be read in pieces: a
    /// prompt pass of plain text (no image, nothing pre-embedded) on a plan
    /// that reads pieces, long enough for two of them.
    pub(super) fn prompt_to_read_in_pieces(
        &self,
        sequence_num: u32,
        prompt_bytes: &[u8],
        vision: bool,
        pre_embedded: bool,
    ) -> Option<Vec<u32>> {
        if sequence_num != 0 || vision || pre_embedded || !self.plan_reads_in_pieces() {
            return None;
        }
        // A token is at least a byte: a prompt shorter in bytes than two pieces
        // is never tokenized just to learn that.
        if pieces(
            0,
            u32::try_from(prompt_bytes.len()).unwrap_or(u32::MAX),
            piece_tokens(),
        )
        .is_empty()
        {
            return None;
        }
        let ids = self.tokenize_prompt(prompt_bytes)?;
        self.reads_in_pieces(ids.len() as u32).then_some(ids)
    }

    /// Run the prompt pass over positions `start..ids.len()` in pieces, every
    /// segment at once, and answer with the last segment's answer to the final
    /// piece — what the whole pass would have answered.
    ///
    /// An `Err` leaves every segment's cache for this request in an unknown
    /// state, and every piece already out has been answered or given up on:
    /// the caller reads the pass whole, which starts each segment over.
    pub(super) async fn prompt_pass_in_pieces(
        &self,
        request_id: uuid::Uuid,
        ids: &[u32],
        start: u32,
        generated_ids: &[u32],
    ) -> Result<LayerResult, SwarmError> {
        let end = u32::try_from(ids.len())
            .map_err(|_| SwarmError::Validation("a prompt past u32 positions".into()))?;
        let cuts = pieces(start, end, piece_tokens());
        if cuts.is_empty() {
            return Err(SwarmError::Internal(
                "a prompt too short for pieces was sent to be read in pieces".into(),
            ));
        }
        let span = PromptSpan { start, end };
        let segments = &self.assignment.segments;
        let started = std::time::Instant::now();
        tracing::info!(
            %request_id,
            pieces = cuts.len(),
            prompt_tokens = end,
            from = start,
            segments = segments.len(),
            "DIAG: a prompt pass in pieces"
        );
        // A piece's input is not kept for a stand-in's replay: a failure
        // mid-reply is continued by the router (#236), as after #10's resume.
        for s in segments {
            self.shared_state
                .retained_activations
                .mark_unrestorable(request_id, s.layer_range);
        }

        // One input channel per segment; the first is filled with the pieces'
        // token ids, each later one by the driver before it.
        let mut senders = Vec::with_capacity(segments.len());
        let mut receivers = Vec::with_capacity(segments.len());
        for _ in segments {
            let (tx, rx) = mpsc::channel::<Vec<u8>>(cuts.len());
            senders.push(Some(tx));
            receivers.push(Some(rx));
        }
        if let Some(first) = senders[0].take() {
            for &(from, to) in &cuts {
                first
                    .try_send(super::pack_verify_tokens_to_le_bytes(
                        &ids[from as usize..to as usize],
                    ))
                    .map_err(|_| SwarmError::Internal("a piece did not fit its channel".into()))?;
            }
        }

        let failed = AtomicBool::new(false);
        let mut drivers = Vec::with_capacity(segments.len());
        for (idx, segment) in segments.iter().enumerate() {
            let input = receivers[idx]
                .take()
                .expect("each segment's receiver is taken once");
            let output = senders.get_mut(idx + 1).and_then(Option::take);
            drivers.push(self.drive_segment(
                request_id,
                idx,
                segment,
                &cuts,
                span,
                input,
                output,
                generated_ids,
                &failed,
            ));
        }
        // Every driver runs to its end — a failed one's neighbours drain what
        // they have out rather than being dropped with it in flight.
        let outcomes = futures::future::join_all(drivers).await;
        let mut answer = None;
        let mut first_error = None;
        for outcome in outcomes {
            match outcome {
                Ok(Some(result)) => answer = Some(result),
                Ok(None) => {}
                Err(e) => {
                    // A driver that stopped because another failed reports
                    // that; the cause is the one to keep.
                    let keep = first_error
                        .as_ref()
                        .is_none_or(|prev| is_knock_on(prev) && !is_knock_on(&e));
                    if keep {
                        first_error = Some(e);
                    }
                }
            }
        }
        if let Some(e) = first_error {
            return Err(e);
        }
        let result = answer.ok_or_else(|| {
            SwarmError::Internal("a prompt pass in pieces ended with no answer".into())
        })?;
        tracing::info!(
            %request_id,
            pieces = cuts.len(),
            pipeline_ms = started.elapsed().as_millis() as u64,
            "DIAG: a prompt pass in pieces completed"
        );
        Ok(result)
    }

    /// One segment's driver: its inputs in order, its segment run on each, each
    /// output handed on. The last segment's answer to the final piece is the
    /// pass's.
    #[allow(clippy::too_many_arguments)]
    async fn drive_segment(
        &self,
        request_id: uuid::Uuid,
        idx: usize,
        segment: &PipelineSegment,
        cuts: &[(u32, u32)],
        span: PromptSpan,
        input: mpsc::Receiver<Vec<u8>>,
        output: Option<mpsc::Sender<Vec<u8>>>,
        generated_ids: &[u32],
        failed: &AtomicBool,
    ) -> DriverResult {
        let outcome = if segment.node_id == *self.shared_state.identity.node_id() {
            self.drive_local(
                idx,
                segment,
                cuts,
                span,
                input,
                &output,
                generated_ids,
                failed,
            )
            .await
        } else {
            self.drive_peer(
                request_id,
                idx,
                segment,
                cuts,
                span,
                input,
                &output,
                generated_ids,
                failed,
            )
            .await
        };
        if outcome.is_err() {
            failed.store(true, Ordering::Release);
        }
        // Dropping `output` closes the next segment's input, which is how a
        // failure here reaches it.
        drop(output);
        outcome
    }

    /// This node's own segment, one piece after another.
    #[allow(clippy::too_many_arguments)]
    async fn drive_local(
        &self,
        idx: usize,
        segment: &PipelineSegment,
        cuts: &[(u32, u32)],
        span: PromptSpan,
        mut input: mpsc::Receiver<Vec<u8>>,
        output: &Option<mpsc::Sender<Vec<u8>>>,
        generated_ids: &[u32],
        failed: &AtomicBool,
    ) -> DriverResult {
        let mut answer = None;
        for (k, &(from, _)) in cuts.iter().enumerate() {
            let Some(activations) = input.recv().await else {
                return Err(knock_on());
            };
            if failed.load(Ordering::Acquire) {
                return Err(knock_on());
            }
            let final_piece = k + 1 == cuts.len();
            let result = self
                .run_local_forward(
                    segment,
                    0,
                    from as usize,
                    activations,
                    None,
                    false,
                    if output.is_none() && final_piece {
                        generated_ids
                    } else {
                        &[]
                    },
                    Some(span),
                )
                .await?;
            if let Some(err) = super::peer_error_from_result(&result) {
                return Err(err);
            }
            if final_piece {
                self.note_prompt_blocks_stored(idx, 0, &result);
            }
            match output {
                Some(next) => next
                    .send(result.activations)
                    .await
                    .map_err(|_| knock_on())?,
                None if final_piece => answer = Some(result),
                None => {}
            }
        }
        Ok(answer)
    }

    /// A peer's segment: its pieces as a numbered stream, at most
    /// [`PEER_WINDOW`] out, the answers taken in order. On a failure no new
    /// piece is sent and the ones out are waited for before returning.
    #[allow(clippy::too_many_arguments)]
    async fn drive_peer(
        &self,
        request_id: uuid::Uuid,
        idx: usize,
        segment: &PipelineSegment,
        cuts: &[(u32, u32)],
        span: PromptSpan,
        mut input: mpsc::Receiver<Vec<u8>>,
        output: &Option<mpsc::Sender<Vec<u8>>>,
        generated_ids: &[u32],
        failed: &AtomicBool,
    ) -> DriverResult {
        let peer = self
            .shared_state
            .resolve_peer_id_bytes(&segment.node_id)
            .ok_or_else(|| {
                SwarmError::Network(format!("No peer_id_bytes for node {}", segment.node_id))
            })?;
        let attempt = NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed) & stream_seq::MAX_ATTEMPT;
        let mut out: FuturesOrdered<PieceWait<'_>> = FuturesOrdered::new();
        // The bytes of each piece out, in order: a piece queued behind another
        // waits for that one's compute too, and its deadline must cover it.
        let mut out_bytes: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut sent = 0usize;
        let mut answered = 0usize;
        let mut input_open = true;
        let mut error: Option<SwarmError> = None;
        let mut answer = None;
        loop {
            if answered == cuts.len() || (error.is_some() && out.is_empty()) {
                break;
            }
            let may_send = error.is_none()
                && input_open
                && sent < cuts.len()
                && out.len() < PEER_WINDOW
                && !failed.load(Ordering::Acquire);
            if !may_send && out.is_empty() {
                // Nothing out and nothing more to send: the input stopped, or
                // another segment failed.
                error.get_or_insert_with(knock_on);
                break;
            }
            tokio::select! {
                biased;
                Some((k, result)) = out.next(), if !out.is_empty() => {
                    out_bytes.pop_front();
                    if error.is_some() {
                        continue;
                    }
                    let result = result.and_then(|r| match super::peer_error_from_result(&r) {
                        Some(err) => Err(err),
                        None => Ok(r),
                    });
                    match result {
                        Ok(result) => {
                            answered += 1;
                            let final_piece = k + 1 == cuts.len();
                            if final_piece {
                                self.note_prompt_blocks_stored(idx, 0, &result);
                            }
                            match output {
                                Some(next) => {
                                    if next.send(result.activations).await.is_err() {
                                        error = Some(knock_on());
                                    }
                                }
                                None if final_piece => answer = Some(result),
                                None => {}
                            }
                        }
                        Err(e) => {
                            failed.store(true, Ordering::Release);
                            error = Some(e);
                        }
                    }
                }
                next = input.recv(), if may_send => {
                    match next {
                        None => input_open = false,
                        Some(activations) => {
                            let k = sent;
                            let queued: usize = out_bytes.iter().sum();
                            let bytes = activations.len();
                            match self
                                .send_piece(
                                    request_id,
                                    idx,
                                    segment,
                                    &peer,
                                    attempt,
                                    k,
                                    cuts,
                                    span,
                                    activations,
                                    output.is_none(),
                                    generated_ids,
                                    queued,
                                )
                                .await
                            {
                                Ok(wait) => {
                                    out.push_back(wait);
                                    out_bytes.push_back(bytes);
                                    sent += 1;
                                }
                                Err(e) => {
                                    failed.store(true, Ordering::Release);
                                    error = Some(e);
                                }
                            }
                        }
                    }
                }
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(answer),
        }
    }

    /// Send piece `k` of the pass to the peer holding `segment` and return the
    /// wait on its answer. The waiter is registered BEFORE the send so a fast
    /// answer is never missed, and removed when the wait is dropped.
    #[allow(clippy::too_many_arguments)]
    async fn send_piece<'a>(
        &'a self,
        request_id: uuid::Uuid,
        idx: usize,
        segment: &'a PipelineSegment,
        peer: &'a [u8],
        attempt: u32,
        k: usize,
        cuts: &[(u32, u32)],
        span: PromptSpan,
        activations: Vec<u8>,
        samples: bool,
        generated_ids: &[u32],
        queued_bytes: usize,
    ) -> Result<PieceWait<'a>, SwarmError> {
        use swarmllm_types::node::features::{FORWARD_GENERATED_IDS, FORWARD_SAMPLING};
        let turn = u32::try_from(k).unwrap_or(u32::MAX);
        let seq = stream_seq::compose(attempt, turn).ok_or_else(|| {
            SwarmError::Internal("a prompt pass ran past its pieces' numbering".into())
        })?;
        let (from, _) = cuts[k];
        let final_piece = k + 1 == cuts.len();
        let state = &self.shared_state;
        let forward = LayerForward {
            request_id,
            sequence_num: 0,
            index_pos: from,
            activations,
            format: TensorFormat::FP32,
            model_id: segment.shard_id.model_id.clone(),
            layer_range: segment.layer_range,
            tp_meta: None,
            vision_embeddings: None,
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            // The same rules as an unpieced pass: the history only to the
            // segment that samples, only on the piece whose token is kept, and
            // only to a peer that reads it.
            generated_ids: if samples
                && final_piece
                && !generated_ids.is_empty()
                && state.peer_advertises_feature(&segment.node_id, FORWARD_GENERATED_IDS)
            {
                generated_ids.to_vec()
            } else {
                Vec::new()
            },
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            spec_walk_at_tail: false,
            coupling_seed: None,
            stream_seq: Some(seq),
            truncate_kv_to: None,
            // A kept pass's hint rides every piece: the receiver restores at
            // the first and stores at the final one.
            prompt_cache: self.prompt_cache_hint.clone(),
            chunk_meta: None,
            sampling: if samples
                && state.peer_advertises_feature(&segment.node_id, FORWARD_SAMPLING)
            {
                Some(self.request.sampling_params.clone())
            } else {
                None
            },
            prompt_span: Some(span),
        };
        if state.pending_layer_results.len() >= super::MAX_PENDING_LAYER_RESULTS {
            return Err(SwarmError::ServiceUnavailable(
                "Pipeline overloaded — too many pending layer results".into(),
            ));
        }
        let key = WaiterKey::streamed(request_id, seq);
        let (tx, rx) = tokio::sync::oneshot::channel();
        state.pending_layer_results.insert(
            key,
            PendingLayerResult {
                tx,
                awaiting: Some(segment.node_id.clone()),
                chain_members: Vec::new(),
                expects_step: Some(ExpectedStep::one(from, segment.layer_range)),
            },
        );
        let waiter = StreamWaiter {
            state: state.clone(),
            key,
        };
        let bytes = forward.activations.len();
        self.network_tx
            .send(NetworkCommand::SendTensor {
                target_peer_bytes: peer.to_vec(),
                forward: forward.clone(),
            })
            .await
            .map_err(|_| SwarmError::Network("Failed to send a piece of the prompt".into()))?;
        let num_layers = segment.layer_range.1 - segment.layer_range.0;
        let budget = SegmentBudget::for_forward(
            state,
            &segment.node_id,
            &segment.shard_id.model_id,
            super::work_kind_for(0),
            num_layers,
            bytes.saturating_add(queued_bytes),
            if idx == 0 {
                ActivationUnits::PromptBytes
            } else {
                ActivationUnits::HiddenStates
            },
        );
        let cancel = self.request.cancel.clone();
        let network_tx = &self.network_tx;
        Ok(Box::pin(async move {
            let _waiter = waiter;
            let rebuild = move || forward.clone();
            let result = PipelineExecutor::wait_for_result(
                state,
                rx,
                request_id,
                idx,
                &segment.node_id,
                num_layers,
                bytes,
                budget,
                cancel.as_ref(),
                ResendOnRefusal::SameForward {
                    network_tx,
                    target_peer_bytes: peer,
                    rebuild: &rebuild,
                },
            )
            .await;
            (k, result)
        }))
    }
}

/// What a driver reports when it stopped because ANOTHER segment's driver
/// failed — never the cause.
fn knock_on() -> SwarmError {
    SwarmError::Inference(KNOCK_ON.into())
}

const KNOCK_ON: &str = "a prompt pass in pieces stopped: another segment's piece failed";

fn is_knock_on(e: &SwarmError) -> bool {
    matches!(e, SwarmError::Inference(m) if m == KNOCK_ON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelId, ShardId};

    fn seg(node: u8, range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id: NodeId([node; 32]),
            shard_id: ShardId {
                model_id: ModelId("prompt-chunks-test".into()),
                index: 0,
            },
            layer_range: range,
        }
    }

    /// Pieces cover the pass exactly, in order, with no gap and no overlap; a
    /// pass shorter than two pieces is not cut; a long one gets longer pieces,
    /// never more than the cap.
    #[test]
    fn pieces_cover_the_pass_and_no_more() {
        assert!(pieces(0, 1000, 512).is_empty(), "under two pieces: whole");
        let p = pieces(0, 2274, 512);
        assert_eq!(p.first().unwrap().0, 0);
        assert_eq!(p.last().unwrap().1, 2274);
        assert!(p.windows(2).all(|w| w[0].1 == w[1].0));
        assert_eq!(p.len(), 5);
        let resumed = pieces(2240, 2240 + 1100, 512);
        assert_eq!(
            resumed.first().unwrap().0,
            2240,
            "a resumed pass starts at its resume point"
        );
        assert_eq!(resumed.last().unwrap().1, 3340);
        let long = pieces(0, 100_000, 512);
        assert!(long.len() as u32 <= MAX_PIECES);
        assert_eq!(long.last().unwrap().1, 100_000);
        assert_eq!(
            pieces(0, 200, 10),
            vec![(0, 64), (64, 128), (128, 192), (192, 200)],
            "the floor holds a tiny setting at 64 positions a piece"
        );
    }

    /// The boomerang (this node twice) and a single segment stay whole; a peer
    /// that does not read pieces keeps the whole plan whole; and so does a
    /// boundary between two peers, which a whole pass chains straight across.
    #[test]
    fn only_a_plan_of_distinct_machines_that_all_read_pieces_is_cut() {
        let me = NodeId([1; 32]);
        let yes = |_: &NodeId| true;
        assert!(shape_reads_in_pieces(
            &[seg(1, (0, 14)), seg(2, (14, 28))],
            &me,
            yes
        ));
        assert!(shape_reads_in_pieces(
            &[seg(2, (0, 14)), seg(1, (14, 28))],
            &me,
            yes
        ));
        assert!(shape_reads_in_pieces(
            &[seg(2, (0, 10)), seg(1, (10, 20)), seg(3, (20, 28))],
            &me,
            yes
        ));
        assert!(
            !shape_reads_in_pieces(&[seg(2, (0, 14)), seg(3, (14, 28))], &me, yes),
            "two peers only: pieces would come back here between them"
        );
        assert!(!shape_reads_in_pieces(
            &[seg(1, (0, 10)), seg(2, (10, 20)), seg(3, (20, 28))],
            &me,
            yes
        ));
        assert!(!shape_reads_in_pieces(&[seg(1, (0, 28))], &me, yes));
        assert!(!shape_reads_in_pieces(
            &[seg(1, (0, 1)), seg(2, (1, 27)), seg(1, (27, 28))],
            &me,
            yes
        ));
        let not_three = |n: &NodeId| *n != NodeId([3; 32]);
        assert!(!shape_reads_in_pieces(
            &[seg(1, (0, 10)), seg(2, (10, 20)), seg(3, (20, 28))],
            &me,
            not_three
        ));
    }

    /// The cause of a failed pass is kept over the drivers that stopped
    /// because of it.
    #[test]
    fn a_knock_on_is_told_from_a_cause() {
        assert!(is_knock_on(&knock_on()));
        assert!(!is_knock_on(&SwarmError::PeerUnresponsive("x".into())));
    }
}
