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

/// Guessers whose first-segment worker refused lately — a drafter on a card
/// target, no output head beside it, no guess head loaded: the next reply
/// does not ask its segments for rows it cannot use. Keyed by
/// [`Guesser::refusal_key`].
fn refused_lately(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>> {
    static REFUSED: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
    > = std::sync::OnceLock::new();
    REFUSED.get_or_init(Default::default)
}

/// How long a refusal keeps a model's replies from asking that guesser.
const REFUSAL_HOLDS: std::time::Duration = std::time::Duration::from_secs(600);

fn refused(key: &str) -> bool {
    refused_lately()
        .lock()
        .ok()
        .and_then(|r| r.get(key).copied())
        .is_some_and(|at| at.elapsed() < REFUSAL_HOLDS)
}

/// The most a guess head's draft makes — the worker's `MtpGuesses::MOST_GUESSES`.
pub(crate) const HEAD_MOST_GUESSES: u32 = 4;
/// A guess head's first draft, before its guesses are timed — the worker's
/// `MtpGuesses::FIRST_GUESSES`, llama.cpp's default.
const HEAD_FIRST_GUESSES: u32 = 3;
/// Rounds a guesser that would have guessed sits out while the other is
/// chosen before it is given one: its price is its own record, which moves
/// only when it runs.
const MOST_ROUNDS_SAT_OUT: u32 = 16;

/// Which guesser beside the split's first segment a round asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum By {
    /// The block drafter (#428): one forward over its block, whatever it guesses.
    Block,
    /// The model's own guess head (#425 (a)): one forward a guess.
    Head,
}

impl By {
    pub(super) fn name(self) -> &'static str {
        match self {
            By::Block => "block drafter",
            By::Head => "guess head",
        }
    }
}

/// One guesser's price and record this reply.
pub(super) struct Guesser {
    by: By,
    /// The most guesses a draft makes: a block less the next token, or the
    /// head's [`HEAD_MOST_GUESSES`].
    most: u32,
    acceptance: crate::inference::dsd_controller::AcceptanceEstimate,
    /// A draft's time from here, rows taken in included: a block's whole
    /// draft, one forward whatever it guesses; the head's per guess.
    draft_ms: crate::inference::dsd_controller::RecentMedian,
    /// The last draft's whole time, for the probe's verdict.
    last_draft_ms: f64,
    /// Rounds the count declined since the last it guessed.
    declined_run: u32,
    /// Declined rounds after which the next guesses anyway.
    probe_after: u32,
    /// This round guesses as a probe.
    probing: bool,
    /// Rounds it would have guessed in that the other guesser took.
    sat_out: u32,
    /// Drafts asked of it this reply.
    asked: u32,
    /// Rounds declined and probes made this reply, for the log.
    pub(super) declined: u32,
    pub(super) probes: u32,
}

impl Guesser {
    fn new(by: By, most: u32) -> Self {
        Self {
            by,
            most,
            acceptance: crate::inference::dsd_controller::AcceptanceEstimate::recent(),
            draft_ms: Default::default(),
            last_draft_ms: 0.0,
            declined_run: 0,
            probe_after: crate::inference::dsd_controller::FIRST_PROBE_AFTER,
            probing: false,
            sat_out: 0,
            asked: 0,
            declined: 0,
            probes: 0,
        }
    }

    fn refusal_key(by: By, model: &str) -> String {
        match by {
            By::Block => model.to_owned(),
            By::Head => format!("{model}:guess-head"),
        }
    }

    /// The next draft's count and the rate it is priced to write at (tokens
    /// a millisecond, `None` before it is priced): before a draft is timed, a
    /// first draft (how the check's line is learned); before a plain step is,
    /// a decline; then the count that pays — zero included, and after
    /// `probe_after` zeros running, as many as it may.
    fn gamma(
        &mut self,
        remaining: u32,
        step: Option<f64>,
        check: &crate::inference::dsd_controller::CheckCost,
    ) -> (u32, Option<f64>) {
        use crate::inference::dsd_controller::{
            rated_beside_a_step_per_guess, rated_beside_a_step_per_round, Step,
        };
        let most = self.most.min(remaining);
        self.probing = false;
        let Some(draft_ms) = self.draft_ms.median() else {
            return match self.by {
                By::Block => (most, None),
                By::Head => (HEAD_FIRST_GUESSES.min(most), None),
            };
        };
        let Some(step) = step else {
            // One plain step first. A round another source fills (the far
            // run's guesses) times no step, so the wait counts as a decline
            // and is probed past the same way.
            return (self.decline_or_probe(most), None);
        };
        let alpha = self.acceptance.alpha();
        let step = Some(Step::OverTheChecksRoute(step));
        let rated = match self.by {
            By::Block => rated_beside_a_step_per_round(alpha, step, check, most, draft_ms),
            By::Head => rated_beside_a_step_per_guess(alpha, step, check, most, draft_ms),
        };
        let (gamma, rate) = match rated {
            Some((g, rate)) => (g, Some(rate)),
            None => (most, None),
        };
        if gamma > 0 {
            self.declined_run = 0;
            return (gamma, rate);
        }
        let gamma = self.decline_or_probe(most);
        if gamma == 0 {
            self.declined += 1;
            self.acceptance.fade();
        }
        (gamma, None)
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

    /// A draft of `guessed` guesses took `ms` from here.
    fn record_draft(&mut self, guessed: usize, ms: f64) {
        self.last_draft_ms = ms;
        match self.by {
            By::Block => self.draft_ms.record(ms),
            By::Head => self.draft_ms.record(ms / guessed.max(1) as f64),
        }
    }

    /// Its round kept `kept` of `guessed` and wrote `wrote` tokens in a check
    /// of `check_ms`. A probe paid if steps would have written fewer in its
    /// draft's and check's time: the count starts over, else the wait for the
    /// next doubles.
    fn record_round(
        &mut self,
        kept: u32,
        guessed: u32,
        wrote: usize,
        check_ms: f64,
        step: Option<f64>,
    ) {
        use crate::inference::dsd_controller::{FIRST_PROBE_AFTER, MOST_PROBE_AFTER};
        self.acceptance.record(kept, guessed);
        if std::mem::take(&mut self.probing) {
            let paid = step.is_none_or(|step| wrote as f64 * step >= check_ms + self.last_draft_ms);
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
}

/// The coordinator's side of a split's guessers beside its FIRST segment —
/// the block drafter (#428 stage 3) and the model's own guess head (#425
/// (a)): which one a round asks, how many guesses, and how each has done this
/// reply.
///
/// **Each count is priced as one computer's is** (`model_worker::LocalSpec`):
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
/// **With both, a round goes to the one priced to write faster** over the
/// same step and line — the block's draft a round, the head's a guess — a
/// probe goes to its guesser, one that has not drafted yet drafts once, and
/// one that would have guessed and sat out [`MOST_ROUNDS_SAT_OUT`] rounds is
/// given one: its price is its own record, which moves only when it runs.
/// Estimated before building: on field prose the head's ~17 tok/s against
/// the block's ~9-10 (`docs/FUTURE_WORK.md` #425 (a)).
///
/// [`Step::OverTheChecksRoute`]: crate::inference::dsd_controller::Step::OverTheChecksRoute
/// [`FIRST_PROBE_AFTER`]: crate::inference::dsd_controller::FIRST_PROBE_AFTER
pub(super) struct SplitBlockDrafter {
    first: (u32, u32),
    pub(super) block: Option<Guesser>,
    pub(super) head: Option<Guesser>,
    /// A one-token round's time: what a round of guesses must beat.
    step_ms: crate::inference::dsd_controller::RecentMedian,
    /// Checks of more than one position — every guess source's, over the
    /// same segments — and never a one-token round.
    check: crate::inference::dsd_controller::CheckCost,
    /// Calls made this attempt, rows-only ones included.
    calls: u32,
    /// The prompt's and the reply's words the worker has been told of.
    words_sent: usize,
}

impl SplitBlockDrafter {
    /// The guessers for this reply, and the ask its segments' forwards carry:
    /// this node runs the split's first segment from layer 0, a peer runs
    /// other layers, and every peer whose segment runs a layer a guesser
    /// reads sends its rows — the block drafter's where its file is beside the
    /// model, the guess head's (the trunk's last layer) where the file
    /// carries one in the shape it runs. Not for a tensor-parallel plan, nor
    /// a guesser whose worker refused lately.
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
        let meta = exec.shared_state.gguf_meta_for(&exec.request.model_id)?;
        let block = (!refused(&Guesser::refusal_key(By::Block, model)))
            .then(|| {
                crate::inference::split::block_drafter_plan_in(&exec.shared_state.model_dir(model))
            })
            .flatten()
            .filter(|plan| every_peer_sends_rows(exec, &plan.target_layers));
        let head_layer = (!refused(&Guesser::refusal_key(By::Head, model)) && head_planned(&meta))
            .then(|| meta.block_count.checked_sub(1))
            .flatten()
            .filter(|&layer| every_peer_sends_rows(exec, &[layer]));
        if block.is_none() && head_layer.is_none() {
            return None;
        }
        let mut layers: Vec<u32> = block
            .iter()
            .flat_map(|plan| plan.target_layers.iter().map(|&l| l as u32))
            .chain(head_layer.map(|l| l as u32))
            .collect();
        layers.sort_unstable();
        layers.dedup();
        Some((
            Self::new(
                first.layer_range,
                block
                    .as_ref()
                    .map(|plan| plan.block_size.saturating_sub(1) as u32),
                head_layer.is_some(),
                super::check_reach(&exec.shared_state, segments),
            ),
            BlockRowsAsk::new(layers, meta.embedding_length),
        ))
    }

    fn new(
        first: (u32, u32),
        block_most: Option<u32>,
        head: bool,
        reach: crate::inference::dsd_controller::CheckReach,
    ) -> Self {
        Self {
            first,
            block: block_most.map(|most| Guesser::new(By::Block, most)),
            head: head.then(|| Guesser::new(By::Head, HEAD_MOST_GUESSES)),
            step_ms: Default::default(),
            check: crate::inference::dsd_controller::CheckCost::for_a_plan(reach),
            calls: 0,
            words_sent: 0,
        }
    }

    /// No guesser is left: the reply goes on without asking.
    pub(super) fn is_empty(&self) -> bool {
        self.block.is_none() && self.head.is_none()
    }

    /// Hand the worker every row the segments sent since the last call — up
    /// to `upto`, the position after the last one committed — and ask `by`
    /// for up to `gamma` guesses after `token`, the token at `upto`
    /// ([`Self::gamma`]'s answer). `gamma` 0 only keeps the guessers' rows
    /// level with the reply. An error ends the guesser asked for the reply.
    /// `prompt` and `generated` are the conversation's words so far; the
    /// worker is told those it has not been.
    pub(super) async fn round(
        &mut self,
        exec: &PipelineExecutor,
        upto: usize,
        token: u32,
        (by, gamma): (By, u32),
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
        let most = match by {
            By::Block => self.block.as_ref().map_or(0, |g| g.most),
            By::Head => self.head.as_ref().map_or(0, |g| g.most),
        };
        let gamma = gamma.min(most);
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
                    gamma,
                    first: self.calls == 0,
                    words,
                    by_head: by == By::Head,
                },
                &payload,
            )
            .await;
        self.calls += 1;
        let asked = (gamma > 0).then_some(by);
        let guesser = match asked {
            Some(By::Block) => self.block.as_mut(),
            Some(By::Head) => self.head.as_mut(),
            None => None,
        };
        let first_ask = guesser.as_ref().is_some_and(|g| g.asked == 0);
        if let Some(g) = guesser {
            g.asked += 1;
        }
        match (&guessed, asked) {
            (Ok(guesses), Some(by)) if !guesses.is_empty() => {
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                if let Some(g) = self.guesser_mut(by) {
                    g.record_draft(guesses.len(), ms);
                }
            }
            (Ok(_), _) => {}
            (Err(e), Some(by)) => {
                // Absent on its first ask: not loaded beside the segment, so
                // the next replies ask none of its rows for a while.
                if first_ask && matches!(e, SwarmError::ServiceUnavailable(_)) {
                    if let Ok(mut refused) = refused_lately().lock() {
                        refused.insert(
                            Guesser::refusal_key(by, &exec.request.model_id.0),
                            std::time::Instant::now(),
                        );
                    }
                }
                *self.slot(by) = None;
            }
            // A rows-only call fails where nothing beside the segment guesses
            // any more.
            (Err(_), None) => {
                self.block = None;
                self.head = None;
            }
        }
        guessed
    }

    fn slot(&mut self, by: By) -> &mut Option<Guesser> {
        match by {
            By::Block => &mut self.block,
            By::Head => &mut self.head,
        }
    }

    fn guesser_mut(&mut self, by: By) -> Option<&mut Guesser> {
        self.slot(by).as_mut()
    }

    /// The next round's guesser and count, for a round with nothing else to
    /// check ([`SplitBlockDrafter`]'s rule for two). Zero is an answer.
    pub(super) fn gamma(&mut self, remaining: u32) -> (By, u32) {
        let step = self.step_ms.median();
        let check = &self.check;
        let block = self.block.as_mut().map(|g| {
            let (gamma, rate) = g.gamma(remaining, step, check);
            (gamma, rate, g.probing, g.draft_ms.median().is_none())
        });
        // A block probing takes its round: the head is not asked, so its
        // record does not move.
        if let Some((gamma, _, true, _)) = block {
            return (By::Block, gamma);
        }
        let head = self.head.as_mut().map(|g| {
            let (gamma, rate) = g.gamma(remaining, step, check);
            (gamma, rate, g.probing, g.draft_ms.median().is_none())
        });
        let by = match (block, head) {
            (Some((b, ..)), None) => return (By::Block, b),
            (None, Some((h, ..))) => return (By::Head, h),
            (None, None) => return (By::Block, 0),
            (Some((b, b_rate, _, b_untimed)), Some((h, h_rate, h_probing, h_untimed))) => {
                let b_sat = self.block.as_ref().map_or(0, |g| g.sat_out);
                let h_sat = self.head.as_ref().map_or(0, |g| g.sat_out);
                choose(
                    Choice {
                        gamma: b,
                        rate: b_rate,
                        untimed: b_untimed,
                        probing: false,
                        sat_out: b_sat,
                    },
                    Choice {
                        gamma: h,
                        rate: h_rate,
                        untimed: h_untimed,
                        probing: h_probing,
                        sat_out: h_sat,
                    },
                )
            }
        };
        let (gamma_b, gamma_h) = (block.map_or(0, |c| c.0), head.map_or(0, |c| c.0));
        let (chosen, other, other_gamma) = match by {
            By::Block => (By::Block, By::Head, gamma_h),
            By::Head => (By::Head, By::Block, gamma_b),
        };
        if let Some(g) = self.guesser_mut(chosen) {
            g.sat_out = 0;
        }
        // A probe always takes its round, so the one sitting out guessed by
        // its price.
        if other_gamma > 0 {
            if let Some(g) = self.guesser_mut(other) {
                g.sat_out += 1;
            }
        }
        (
            by,
            match by {
                By::Block => gamma_b,
                By::Head => gamma_h,
            },
        )
    }

    /// A one-token round took `ms`: the step. The first ends the wait for
    /// one, which is no decline of a price.
    pub(super) fn record_step(&mut self, ms: f64) {
        if self.step_ms.median().is_none() {
            for g in self.block.iter_mut().chain(self.head.iter_mut()) {
                g.declined_run = 0;
            }
        }
        self.step_ms.record(ms);
    }

    /// A check of `positions` took `ms`, whatever guessed it — the segments
    /// it crossed are these guessers'.
    pub(super) fn record_check(&mut self, positions: u32, ms: f64) {
        if positions > 1 {
            self.check.record(positions, ms);
        }
    }

    /// `by`'s round kept `kept` of `guessed` and wrote `wrote` tokens in a
    /// check of `check_ms` ([`Guesser::record_round`]).
    pub(super) fn record_round(
        &mut self,
        by: By,
        kept: u32,
        guessed: u32,
        wrote: usize,
        check_ms: f64,
    ) {
        let step = self.step_ms.median();
        if let Some(g) = self.guesser_mut(by) {
            g.record_round(kept, guessed, wrote, check_ms, step);
        }
    }

    pub(super) fn step_ms(&self) -> Option<f64> {
        self.step_ms.median()
    }

    pub(super) fn check_fit(&self) -> Option<(f64, f64)> {
        self.check.fit()
    }
}

/// Does this model's file carry a guess head a split's first segment loads
/// (#425 (a)): the switches on, Qwen 3.5's families, and one next-token block
/// in the shape the head runs (`split::mtp::head_in_shape`, the check the
/// load runs). Whether this node holds its parts is the worker's answer — a
/// head not loaded is refused on its first ask, and held off a while.
fn head_planned(meta: &crate::inference::split::GgufTensorMeta) -> bool {
    use crate::inference::split::{head_in_shape, mtp_on, split_guess_head_on};
    let has = |name: &str| meta.tensors.contains_key(name);
    mtp_on()
        && split_guess_head_on()
        && matches!(meta.architecture.as_str(), "qwen35" | "qwen35moe")
        && head_in_shape(meta.block_count, has)
        && !has(&format!(
            "blk.{}.nextn.eh_proj.weight",
            meta.block_count + 1
        ))
}

/// One guesser's case for a round.
#[derive(Clone, Copy, Debug)]
struct Choice {
    gamma: u32,
    rate: Option<f64>,
    /// It has not drafted yet.
    untimed: bool,
    probing: bool,
    sat_out: u32,
}

/// Which of two guessers a round asks ([`SplitBlockDrafter`]): one that would
/// not guess loses to one that would; a probe takes its round; one not yet
/// timed drafts once, the block first; one that sat out
/// [`MOST_ROUNDS_SAT_OUT`] rounds gets one; else the faster priced, the block
/// on a tie or where either is unpriced.
fn choose(block: Choice, head: Choice) -> By {
    match (block.gamma > 0, head.gamma > 0) {
        (false, true) => return By::Head,
        (_, false) => return By::Block,
        (true, true) => {}
    }
    if block.probing {
        return By::Block;
    }
    if head.probing {
        return By::Head;
    }
    if block.untimed {
        return By::Block;
    }
    if head.untimed {
        return By::Head;
    }
    if head.sat_out >= MOST_ROUNDS_SAT_OUT {
        return By::Head;
    }
    if block.sat_out >= MOST_ROUNDS_SAT_OUT {
        return By::Block;
    }
    match (block.rate, head.rate) {
        (Some(b), Some(h)) if h > b => By::Head,
        _ => By::Block,
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

    fn block_only(round_trip_ms: Option<u32>) -> SplitBlockDrafter {
        SplitBlockDrafter::new((0, 17), Some(15), false, CheckReach::Peer { round_trip_ms })
    }

    fn block(b: &mut SplitBlockDrafter) -> &mut Guesser {
        b.block.as_mut().expect("a block drafter")
    }

    /// The count alone: a block-only drafter's rounds are the block's.
    fn count(b: &mut SplitBlockDrafter, remaining: u32) -> u32 {
        let (by, gamma) = b.gamma(remaining);
        assert_eq!(by, By::Block);
        gamma
    }

    /// The rig's story (#428 stage 3, 2026-11-08: Qwen3.5-9B split over
    /// 130 ms, processors): a 16-row check of 1,262 ms beside a one-token
    /// round of 302, a draft of 247, one guess in fifteen kept. The count
    /// learns the step before it prices, then declines, and probes after
    /// `FIRST_PROBE_AFTER` declines — a probe that wrote too little doubles
    /// the wait. Every round guessed 15 before, priced by the loop's fit.
    #[test]
    fn a_split_block_drafter_steps_aside_on_a_reply_it_does_not_guess() {
        let mut b = block_only(Some(130));
        assert_eq!(count(&mut b, 200), 15, "untimed: as many as it may");
        block(&mut b).record_draft(15, 247.0);
        b.record_check(16, 1_262.0);
        b.record_round(By::Block, 1, 15, 2, 1_262.0);
        assert_eq!(count(&mut b, 200), 0, "a step first");
        b.record_step(302.0);
        b.record_check(1, 302.0);
        for _ in 0..3 {
            b.record_check(16, 1_262.0);
            b.record_round(By::Block, 1, 15, 2, 1_262.0);
        }
        for _ in 1..FIRST_PROBE_AFTER {
            assert_eq!(count(&mut b, 200), 0);
        }
        assert_eq!(count(&mut b, 200), 15, "a probe");
        b.record_check(16, 1_262.0);
        b.record_round(By::Block, 1, 15, 2, 1_262.0);
        assert_eq!(
            block(&mut b).probe_after,
            FIRST_PROBE_AFTER * 2,
            "it did not pay"
        );
        assert_eq!(
            (block(&mut b).declined, block(&mut b).probes),
            (FIRST_PROBE_AFTER - 1, 1)
        );
        // Guesses kept again — the reply turned to copying — and the count
        // buys them.
        for _ in 0..8 {
            b.record_round(By::Block, 14, 15, 15, 1_262.0);
        }
        assert!(count(&mut b, 200) > 0);
        assert!(count(&mut b, 3) <= 3, "never past what is left");
    }

    /// Where no one-token round ever comes — another source fills every
    /// round the drafter declines — the wait for a step is probed past as a
    /// decline is, never kept for the reply.
    #[test]
    fn a_drafter_waiting_for_a_step_still_probes() {
        let mut b = block_only(None);
        block(&mut b).record_draft(15, 247.0);
        for _ in 1..FIRST_PROBE_AFTER {
            assert_eq!(count(&mut b, 200), 0);
        }
        assert_eq!(count(&mut b, 200), 15);
        assert_eq!((block(&mut b).declined, block(&mut b).probes), (0, 1));
    }

    /// Both guessers over a ~130 ms link, a step ~150 ms: the block's 60 ms
    /// draft keeps two of 15 a round, the head's 4 ms guesses three of four. Each drafts once
    /// before it is priced — the block first — then the head, priced faster,
    /// takes the rounds, and the block is given one after sitting out
    /// `MOST_ROUNDS_SAT_OUT`.
    #[test]
    fn the_faster_priced_guesser_takes_the_round_and_the_other_is_tried_again() {
        let mut b = SplitBlockDrafter::new(
            (0, 1),
            Some(15),
            true,
            CheckReach::Peer {
                round_trip_ms: Some(130),
            },
        );
        assert_eq!(b.gamma(200), (By::Block, 15), "the block drafts first");
        b.block.as_mut().unwrap().record_draft(15, 60.0);
        b.record_check(16, 260.0);
        for _ in 0..4 {
            b.record_round(By::Block, 2, 15, 3, 260.0);
        }
        assert_eq!(
            b.gamma(200),
            (By::Head, HEAD_FIRST_GUESSES),
            "then the head once"
        );
        b.head.as_mut().unwrap().record_draft(3, 12.0);
        b.record_check(4, 175.0);
        b.record_round(By::Head, 2, 3, 3, 175.0);
        b.record_step(150.0);
        for _ in 0..6 {
            b.record_round(By::Head, 3, 4, 4, 180.0);
            b.record_check(5, 180.0);
        }
        let mut heads = 0;
        let mut blocks = 0;
        for _ in 0..=MOST_ROUNDS_SAT_OUT {
            let (by, gamma) = b.gamma(200);
            assert!(gamma > 0);
            match by {
                By::Head => {
                    heads += 1;
                    b.record_round(By::Head, 3, gamma, gamma as usize, 180.0);
                }
                By::Block => blocks += 1,
            }
        }
        assert_eq!(heads, MOST_ROUNDS_SAT_OUT, "the head writes faster");
        assert_eq!(blocks, 1, "the block is given a round after sitting out");
    }

    /// One that would not guess never takes the round from one that would,
    /// and a probe takes its own.
    #[test]
    fn a_guesser_that_declines_or_probes_is_chosen_by_that() {
        let c = |gamma, rate, probing| Choice {
            gamma,
            rate,
            untimed: false,
            probing,
            sat_out: 0,
        };
        assert_eq!(choose(c(0, None, false), c(3, Some(0.01), false)), By::Head);
        assert_eq!(
            choose(c(15, Some(0.01), false), c(0, None, false)),
            By::Block
        );
        assert_eq!(
            choose(c(15, Some(0.01), false), c(4, Some(0.02), false)),
            By::Head
        );
        assert_eq!(
            choose(c(15, Some(0.02), false), c(4, Some(0.01), false)),
            By::Block
        );
        assert_eq!(
            choose(c(15, Some(0.02), false), c(4, None, true)),
            By::Head,
            "a probe"
        );
        assert_eq!(
            choose(c(15, None, true), c(4, Some(0.05), false)),
            By::Block,
            "a probe"
        );
        assert_eq!(
            choose(c(15, Some(0.01), false), c(4, None, false)),
            By::Block,
            "unpriced"
        );
    }
}
