//! A split's block drafter (#428 stage 3): the DFlash drafter beside the
//! split's FIRST segment, on the computer that coordinates the reply, fed the
//! outputs of the model's layers it reads — rows every segment running one of
//! them sends back with its forward (`LayerForward::block_rows_wanted`,
//! `LayerResult::block_rows`).
//!
//! Prior art: vLLM relays an EAGLE-3 drafter's auxiliary hidden states
//! (`aux_hidden_states`) along its pipeline-parallel hand-off to the rank that
//! drafts; SGLang drafts at the last rank. Here the drafter sits with the token
//! table, at the head — the tail holds no table to embed the block's next token
//! — so no peer keeps a drafter's cache for another computer's reply, and the
//! computer that asked pays for its own guesses, as `FarRunGuesses` does. The
//! rows travel only as protected as a result's activations, which are hidden
//! states too.

use crate::error::SwarmError;
use crate::types::{BlockRows, BlockRowsWanted, LayerResult, NodeId};

use super::PipelineExecutor;

/// Rows of the prompt's end each segment sends after the prompt pass: over one
/// fixed reply the drafter kept 92-98% of its whole-prompt acceptance with the
/// last 64 (#428 stage 1, `SWARMLLM_GUESS_TEST_CONTEXT`).
pub(crate) const PROMPT_WINDOW: u32 = 64;

/// What a reply that drafts by block asks of its segments, and the rows they
/// sent that the drafter has not yet taken in.
pub(crate) struct BlockRowsAsk {
    layers: Vec<u32>,
    /// A row's width per layer: the model's hidden size.
    hidden: usize,
    got: std::sync::Mutex<Vec<BlockRows>>,
}

impl BlockRowsAsk {
    fn new(layers: Vec<u32>, hidden: usize) -> Self {
        Self {
            layers,
            hidden,
            got: Default::default(),
        }
    }

    /// The ask a forward carries: at most `most_rows` of its last rows.
    pub(super) fn wanted(&self, most_rows: u32) -> BlockRowsWanted {
        BlockRowsWanted {
            layers: self.layers.clone(),
            most_rows,
        }
    }

    /// Keep the rows a segment running `runs` answered with, to an ask for at
    /// most `most_rows` — only where they are what was asked of it: layers it
    /// runs that this reply asked for, each once, and no more rows than asked.
    /// Anything else is dropped before it is read; the drafter then finds a
    /// gap and stops for the reply (security review, 2026-11-08: a peer could
    /// send 64 MiB where 16 rows were asked, or rows of layers it does not run
    /// in place of another segment's).
    pub(super) fn note(&self, result: &LayerResult, runs: (u32, u32), most_rows: u32) {
        if let Some(rows) = result.block_rows.as_ref() {
            self.keep(rows, runs, most_rows);
        }
    }

    fn keep(&self, rows: &BlockRows, runs: (u32, u32), most_rows: u32) {
        let mut seen = std::collections::HashSet::new();
        let asked = !rows.layers.is_empty()
            && rows.layers.iter().all(|&l| {
                (runs.0..runs.1).contains(&l) && self.layers.contains(&l) && seen.insert(l)
            });
        let most_bytes = (most_rows as usize)
            .saturating_mul(rows.layers.len())
            .saturating_mul(self.hidden)
            .saturating_mul(2);
        if !asked || rows.data.len() > most_bytes {
            tracing::debug!(
                layers = ?rows.layers,
                bytes = rows.data.len(),
                most_rows,
                ?runs,
                "block drafter: rows a segment was not asked for, dropped"
            );
            return;
        }
        if let Ok(mut got) = self.got.lock() {
            got.push(rows.clone());
        }
    }

    fn take(&self) -> Vec<BlockRows> {
        self.got
            .lock()
            .map(|mut got| std::mem::take(&mut *got))
            .unwrap_or_default()
    }
}

impl PipelineExecutor {
    /// The ask a forward to `node` carries for this reply's block drafter,
    /// for at most `most_rows` rows: `None` unless the reply drafts by block,
    /// and for a peer only where it advertises `features::SENDS_BLOCK_ROWS` —
    /// gated at the sender, as every trailer is.
    pub(super) fn block_rows_wanted_for(
        &self,
        node: &NodeId,
        most_rows: u32,
    ) -> Option<BlockRowsWanted> {
        let ask = self.block_rows.as_ref()?;
        let here = node == self.shared_state.identity.node_id();
        (here
            || self
                .shared_state
                .peer_advertises_feature(node, swarmllm_types::node::features::SENDS_BLOCK_ROWS))
        .then(|| ask.wanted(most_rows))
    }

    /// Keep the rows `segment`'s answer carries, for the next draft — held to
    /// `most_rows`, what its forward asked ([`BlockRowsAsk::note`]).
    pub(super) fn note_block_rows(
        &self,
        result: &LayerResult,
        segment: &crate::types::PipelineSegment,
        most_rows: u32,
    ) {
        if let Some(ask) = self.block_rows.as_ref() {
            ask.note(result, segment.layer_range, most_rows);
        }
    }
}

/// `features::SENDS_BLOCK_ROWS` asked of every peer whose segment runs one of
/// `layers`.
fn every_peer_sends_rows(exec: &PipelineExecutor, layers: &[usize]) -> bool {
    let local = exec.shared_state.identity.node_id();
    exec.assignment.segments.iter().all(|s| {
        let (a, b) = (s.layer_range.0 as usize, s.layer_range.1 as usize);
        s.node_id == *local
            || !layers.iter().any(|l| (a..b).contains(l))
            || exec.shared_state.peer_advertises_feature(
                &s.node_id,
                swarmllm_types::node::features::SENDS_BLOCK_ROWS,
            )
    })
}

/// Models whose first-segment worker refused to draft by block lately — a
/// drafter on a card target, or no output head beside it: the next reply does
/// not ask its segments for rows it cannot use.
fn refused_lately(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>> {
    static REFUSED: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
    > = std::sync::OnceLock::new();
    REFUSED.get_or_init(Default::default)
}

/// How long a refusal keeps a model's replies from drafting by block.
const REFUSAL_HOLDS: std::time::Duration = std::time::Duration::from_secs(600);

/// The coordinator's side of a split's block drafter: when to ask it, how
/// many guesses, and how it has done this reply.
///
/// **Its count is priced as one computer's is** (`model_worker::LocalSpec`):
/// a step — a one-token round over the split — timed on its own, a line
/// fitted from checks only, and a check of a width not yet timed priced
/// through the step ([`Step::OverTheChecksRoute`]). Priced from the round
/// loop's one fit instead, whose one-token rounds and 16-row checks each
/// faded from it while the other ran, the count read a flat line in turn
/// through each and chose 15 or 0 and nothing between: on a story it guessed
/// 15 in 54 of 99 rounds, kept ~1.1 a round, and ran at 0.56× the reply
/// without it (rig, 2026-11-08). A refusal is checked anyway after
/// [`FIRST_PROBE_AFTER`] declined rounds, doubling while that does not pay.
///
/// [`Step::OverTheChecksRoute`]: crate::inference::dsd_controller::Step::OverTheChecksRoute
/// [`FIRST_PROBE_AFTER`]: crate::inference::dsd_controller::FIRST_PROBE_AFTER
pub(super) struct SplitBlockDrafter {
    first: (u32, u32),
    /// The most guesses a draft makes: its block less the next token.
    most: u32,
    acceptance: crate::inference::dsd_controller::AcceptanceEstimate,
    /// One draft's time from here, rows taken in included — one forward over
    /// its block whatever it guesses.
    draft_ms: crate::inference::dsd_controller::RecentMedian,
    /// The last draft's time, for the probe's verdict.
    last_draft_ms: f64,
    /// A one-token round's time: what a round of guesses must beat.
    step_ms: crate::inference::dsd_controller::RecentMedian,
    /// Checks of more than one position — every guess source's, over the
    /// same segments — and never a one-token round.
    check: crate::inference::dsd_controller::CheckCost,
    /// Rounds the count declined since the last it guessed.
    declined_run: u32,
    /// Declined rounds after which the next guesses anyway.
    probe_after: u32,
    /// This round guesses as a probe.
    probing: bool,
    /// Rounds declined and probes made this reply, for the log.
    pub(super) declined: u32,
    pub(super) probes: u32,
    /// Calls made this attempt, rows-only ones included.
    calls: u32,
    /// The prompt's and the reply's words the worker has been told of.
    words_sent: usize,
}

impl SplitBlockDrafter {
    /// A drafter for this reply, and the ask its segments' forwards carry:
    /// this node runs the split's first segment from layer 0 with a drafter
    /// file beside the model, a peer runs other layers, and every peer whose
    /// segment runs a layer the drafter reads sends its rows. Not for a
    /// tensor-parallel plan, nor a model whose worker refused lately.
    pub(super) fn for_plan(exec: &PipelineExecutor) -> Option<(Self, BlockRowsAsk)> {
        if !exec.assignment.tp_groups.is_empty() {
            return None;
        }
        let local = exec.shared_state.identity.node_id();
        let segments = &exec.assignment.segments;
        let first = segments.first()?;
        if first.node_id != *local
            || first.layer_range.0 != 0
            || segments.iter().all(|s| s.node_id == *local)
        {
            return None;
        }
        let model = exec.request.model_id.0.as_str();
        if refused_lately()
            .lock()
            .ok()?
            .get(model)
            .is_some_and(|at| at.elapsed() < REFUSAL_HOLDS)
        {
            return None;
        }
        let plan =
            crate::inference::split::block_drafter_plan_in(&exec.shared_state.model_dir(model))?;
        if !every_peer_sends_rows(exec, &plan.target_layers) {
            return None;
        }
        let hidden = exec
            .shared_state
            .gguf_meta_for(&exec.request.model_id)?
            .embedding_length;
        let layers = plan.target_layers.iter().map(|&l| l as u32).collect();
        Some((
            Self::new(
                first.layer_range,
                plan.block_size.saturating_sub(1) as u32,
                super::check_reach(&exec.shared_state, segments),
            ),
            BlockRowsAsk::new(layers, hidden),
        ))
    }

    fn new(
        first: (u32, u32),
        most: u32,
        reach: crate::inference::dsd_controller::CheckReach,
    ) -> Self {
        Self {
            first,
            most,
            acceptance: crate::inference::dsd_controller::AcceptanceEstimate::recent(),
            draft_ms: Default::default(),
            last_draft_ms: 0.0,
            step_ms: Default::default(),
            check: crate::inference::dsd_controller::CheckCost::for_a_plan(reach),
            declined_run: 0,
            probe_after: crate::inference::dsd_controller::FIRST_PROBE_AFTER,
            probing: false,
            declined: 0,
            probes: 0,
            calls: 0,
            words_sent: 0,
        }
    }

    /// Hand the drafter every row the segments sent since the last call — up
    /// to `upto`, the position after the last one committed — and ask for up
    /// to `gamma` guesses after `token`, the token at `upto`. `gamma` 0 only
    /// keeps its rows level with the reply. An error ends it for the reply.
    /// `prompt` and `generated` are the conversation's words so far; the
    /// worker is told those it has not been.
    pub(super) async fn round(
        &mut self,
        exec: &PipelineExecutor,
        upto: usize,
        token: u32,
        gamma: u32,
        prompt: &[u32],
        generated: &[u32],
    ) -> Result<Vec<u32>, SwarmError> {
        let words: Vec<u32> = prompt
            .iter()
            .chain(generated)
            .skip(self.words_sent)
            .copied()
            .collect();
        self.words_sent += words.len();
        let rows = exec
            .block_rows
            .as_ref()
            .map(BlockRowsAsk::take)
            .unwrap_or_default();
        let mut payload = Vec::with_capacity(rows.iter().map(|r| r.data.len()).sum());
        let mut metas = Vec::with_capacity(rows.len());
        for r in rows {
            metas.push(crate::inference::worker_ipc::IpcBlockRows {
                layers: r.layers,
                from: r.from,
                bytes: r.data.len() as u32,
            });
            payload.extend_from_slice(&r.data);
        }
        let started = std::time::Instant::now();
        let guessed = exec
            .shared_state
            .model_process_pool
            .block_draft(
                crate::inference::worker_ipc::IpcBlockDraft {
                    request_id: exec.request.id,
                    model_id: exec.request.model_id.clone(),
                    first_range: self.first,
                    rows: metas,
                    upto: upto as u32,
                    index_pos: upto as u32,
                    token,
                    gamma: gamma.min(self.most),
                    first: self.calls == 0,
                    words,
                },
                &payload,
            )
            .await;
        let first = self.calls == 0;
        self.calls += 1;
        match &guessed {
            Ok(guesses) if !guesses.is_empty() => {
                self.last_draft_ms = started.elapsed().as_secs_f64() * 1000.0;
                self.draft_ms.record(self.last_draft_ms);
            }
            Ok(_) => {}
            Err(SwarmError::ServiceUnavailable(_)) if first => {
                if let Ok(mut refused) = refused_lately().lock() {
                    refused.insert(exec.request.model_id.0.clone(), std::time::Instant::now());
                }
            }
            Err(_) => {}
        }
        guessed
    }

    /// The next draft's count, for a round with nothing else to check: one
    /// forward whatever it guesses, against a step. Before a draft is timed,
    /// as many as it may (how the check's line is learned); then one plain
    /// step, the price a guess must beat; then the count that pays — zero
    /// included, and after `probe_after` zeros running, as many as it may.
    pub(super) fn gamma(&mut self, remaining: u32) -> u32 {
        use crate::inference::dsd_controller::{best_gamma_beside_a_step_per_round, Step};
        let most = self.most.min(remaining);
        self.probing = false;
        let Some(draft_ms) = self.draft_ms.median() else {
            return most;
        };
        let Some(step) = self.step_ms.median() else {
            // One plain step first. A round another source fills (the far
            // run's guesses) times no step, so the wait counts as a decline
            // and is probed past the same way.
            return self.decline_or_probe(most);
        };
        let gamma = best_gamma_beside_a_step_per_round(
            self.acceptance.alpha(),
            Some(Step::OverTheChecksRoute(step)),
            &self.check,
            most,
            draft_ms,
        )
        .unwrap_or(most);
        if gamma > 0 {
            self.declined_run = 0;
            return gamma;
        }
        let gamma = self.decline_or_probe(most);
        if gamma == 0 {
            self.declined += 1;
            self.acceptance.fade();
        }
        gamma
    }

    /// Zero, or `most` as a probe once `probe_after` rounds have declined.
    fn decline_or_probe(&mut self, most: u32) -> u32 {
        self.declined_run += 1;
        if self.declined_run
            < self
                .probe_after
                .min(crate::inference::dsd_controller::MOST_PROBE_AFTER)
        {
            return 0;
        }
        self.declined_run = 0;
        self.probing = true;
        self.probes += 1;
        most
    }

    /// A one-token round took `ms`: the step. The first ends the wait for
    /// one, which is no decline of a price.
    pub(super) fn record_step(&mut self, ms: f64) {
        if self.step_ms.median().is_none() {
            self.declined_run = 0;
        }
        self.step_ms.record(ms);
    }

    /// A check of `positions` took `ms`, whatever guessed it — the segments
    /// it crossed are this drafter's.
    pub(super) fn record_check(&mut self, positions: u32, ms: f64) {
        if positions > 1 {
            self.check.record(positions, ms);
        }
    }

    /// This drafter's round kept `kept` of `guessed` and wrote `wrote` tokens
    /// in a check of `check_ms`. A probe paid if steps would have written
    /// fewer in its draft's and check's time: the count starts over, else the
    /// wait for the next doubles.
    pub(super) fn record_round(&mut self, kept: u32, guessed: u32, wrote: usize, check_ms: f64) {
        use crate::inference::dsd_controller::{FIRST_PROBE_AFTER, MOST_PROBE_AFTER};
        self.acceptance.record(kept, guessed);
        if std::mem::take(&mut self.probing) {
            let paid = self
                .step_ms
                .median()
                .is_none_or(|step| wrote as f64 * step >= check_ms + self.last_draft_ms);
            self.probe_after = if paid {
                FIRST_PROBE_AFTER
            } else {
                (self.probe_after * 2).min(MOST_PROBE_AFTER)
            };
        }
    }

    pub(super) fn draft_ms(&self) -> Option<f64> {
        self.draft_ms.median()
    }

    pub(super) fn step_ms(&self) -> Option<f64> {
        self.step_ms.median()
    }

    pub(super) fn check_fit(&self) -> Option<(f64, f64)> {
        self.check.fit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::dsd_controller::{CheckReach, FIRST_PROBE_AFTER};

    /// A segment's rows are kept only as asked of it: layers it runs and the
    /// reply asked for, each once, and no more rows than its forward asked.
    #[test]
    fn rows_a_segment_was_not_asked_for_are_dropped() {
        let hidden = 8;
        let ask = BlockRowsAsk::new(vec![1, 5, 17, 21], hidden);
        let answer = |layers: Vec<u32>, rows: usize| BlockRows {
            data: vec![0; rows * layers.len() * hidden * 2],
            layers,
            from: 40,
        };
        let b = (17, 31);
        ask.keep(&answer(vec![17, 21], 16), b, 16);
        assert_eq!(ask.take().len(), 1, "as asked");
        ask.keep(&answer(vec![17, 21], 17), b, 16);
        assert!(ask.take().is_empty(), "a row more than asked");
        ask.keep(&answer(vec![1, 5], 16), b, 16);
        assert!(ask.take().is_empty(), "layers another segment runs");
        ask.keep(&answer(vec![17, 19], 16), b, 16);
        assert!(ask.take().is_empty(), "a layer the reply did not ask for");
        ask.keep(&answer(vec![17, 17], 8), b, 16);
        assert!(ask.take().is_empty(), "a layer twice");
        ask.keep(&answer(vec![17], 4), b, 0);
        assert!(ask.take().is_empty(), "a forward that asked for none");
    }

    /// The rig's story (#428 stage 3, 2026-11-08: Qwen3.5-9B split over
    /// 130 ms, processors): a 16-row check of 1,262 ms beside a one-token
    /// round of 302, a draft of 247, one guess in fifteen kept. The count
    /// learns the step before it prices, then declines, and probes after
    /// `FIRST_PROBE_AFTER` declines — a probe that wrote too little doubles
    /// the wait. Every round guessed 15 before, priced by the loop's fit.
    #[test]
    fn a_split_block_drafter_steps_aside_on_a_reply_it_does_not_guess() {
        let mut b = SplitBlockDrafter::new(
            (0, 17),
            15,
            CheckReach::Peer {
                round_trip_ms: Some(130),
            },
        );
        assert_eq!(b.gamma(200), 15, "untimed: as many as it may");
        b.draft_ms.record(247.0);
        b.last_draft_ms = 247.0;
        b.record_check(16, 1_262.0);
        b.record_round(1, 15, 2, 1_262.0);
        assert_eq!(b.gamma(200), 0, "a step first");
        b.record_step(302.0);
        b.record_check(1, 302.0);
        for _ in 0..3 {
            b.record_check(16, 1_262.0);
            b.record_round(1, 15, 2, 1_262.0);
        }
        for _ in 1..FIRST_PROBE_AFTER {
            assert_eq!(b.gamma(200), 0);
        }
        assert_eq!(b.gamma(200), 15, "a probe");
        b.record_check(16, 1_262.0);
        b.record_round(1, 15, 2, 1_262.0);
        assert_eq!(b.probe_after, FIRST_PROBE_AFTER * 2, "it did not pay");
        assert_eq!((b.declined, b.probes), (FIRST_PROBE_AFTER - 1, 1));
        // Guesses kept again — the reply turned to copying — and the count
        // buys them.
        for _ in 0..8 {
            b.record_round(14, 15, 15, 1_262.0);
        }
        assert!(b.gamma(200) > 0);
        assert!(b.gamma(3) <= 3, "never past what is left");
    }

    /// Where no one-token round ever comes — another source fills every
    /// round the drafter declines — the wait for a step is probed past as a
    /// decline is, never kept for the reply.
    #[test]
    fn a_drafter_waiting_for_a_step_still_probes() {
        let mut b = SplitBlockDrafter::new(
            (0, 17),
            15,
            CheckReach::Peer {
                round_trip_ms: None,
            },
        );
        b.draft_ms.record(247.0);
        for _ in 1..FIRST_PROBE_AFTER {
            assert_eq!(b.gamma(200), 0);
        }
        assert_eq!(b.gamma(200), 15);
        assert_eq!((b.declined, b.probes), (0, 1));
    }
}
