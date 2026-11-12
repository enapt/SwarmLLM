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

    /// Ask only `layers` from now on — the head's, once the block is set
    /// aside ([`SplitBlockDrafter::rows_now_read`]). Rows already kept stay:
    /// each guesser in the worker reads only its own layers of them.
    pub(super) fn narrow(&mut self, layers: Vec<u32>) {
        self.layers = layers;
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

    /// After a round's tokens are committed: where the reply's block drafter
    /// was just set aside ([`SplitBlockDrafter::reply_committed`]), its
    /// forwards ask only the head's rows from now on — or none, and the
    /// reply goes on without asking a guesser.
    pub(super) fn block_drafter_set_aside(
        &mut self,
        drafter: &mut Option<SplitBlockDrafter>,
        reply_len: usize,
    ) {
        let Some(d) = drafter.as_mut() else {
            return;
        };
        let Some(layers) = d.rows_now_read() else {
            return;
        };
        // A round the block took judges it before the reply has its tokens.
        d.set_aside_at.get_or_insert(reply_len);
        tracing::info!(
            request_id = %self.request.id,
            reply_tokens = reply_len,
            judged = d.shadows,
            judged_kept = d.shadow_kept,
            head = d.head.is_some(),
            "block drafter set aside for this reply: the guess head or a plain step writes faster"
        );
        if d.is_empty() || layers.is_empty() {
            *drafter = None;
            self.block_rows = None;
        } else if let Some(ask) = self.block_rows.as_mut() {
            ask.narrow(layers);
        }
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
/// only when it runs. Not the block's while it is judged by the reply
/// ([`judged_by_the_reply`]): its record moves on every round.
const MOST_ROUNDS_SAT_OUT: u32 = 16;
/// Judged rounds in a row in which the block would not have written faster
/// than the round the reply took — the head's, or a plain step — after which
/// it is set aside for the reply, and its rows no longer cross the split.
const SET_ASIDE_AFTER_JUDGED: u32 = 8;

/// The block drafter is judged by the reply on every round it does not take
/// (`dsd_controller::BlockShadow`), each guesser is priced by its own depth
/// curve (`DepthAcceptance`), and a block that keeps losing is set aside for
/// the reply — on unless `SWARMLLM_SPLIT_GUESS_PRICE=geometric`, which keeps
/// the rule before 2026-11-09 (one α, the block measured only when checked),
/// for A/B inside one binary.
pub(super) fn judged_by_the_reply() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_SPLIT_GUESS_PRICE").as_deref() != Ok("geometric"))
}

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
    acceptance: crate::inference::dsd_controller::DepthAcceptance,
    /// Its record moves on every round, whether it takes the round or not
    /// (the block, [`judged_by_the_reply`]): it never probes, never fades
    /// on a decline and never sits rounds out.
    judged: bool,
    /// Priced by its own depth curve, else by one α (the rule before).
    by_depth: bool,
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
    /// `rule`: [`judged_by_the_reply`], read once for the reply.
    fn new(by: By, most: u32, rule: bool) -> Self {
        Self {
            by,
            most,
            acceptance: crate::inference::dsd_controller::DepthAcceptance::recent(),
            judged: by == By::Block && rule,
            by_depth: rule,
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
        let most = self.most.min(remaining);
        self.probing = false;
        if self.draft_ms.median().is_none() {
            return match self.by {
                By::Block => (most, None),
                By::Head => (HEAD_FIRST_GUESSES.min(most), None),
            };
        }
        let Some(step) = step else {
            // One plain step first. A round another source fills (the far
            // run's guesses) times no step, so the wait counts as a decline
            // and is probed past the same way.
            return (self.decline_or_probe(most), None);
        };
        let (gamma, rate) = match self.priced(most, step, check) {
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
            if !self.judged {
                self.acceptance.fade();
            }
        }
        (gamma, None)
    }

    /// The count of at most `most` that writes fastest over the checks'
    /// route, and that rate (tokens a millisecond) — zero guesses, a plain
    /// step, among them. `None` before a draft and a check are timed.
    fn priced(
        &self,
        most: u32,
        step: f64,
        check: &crate::inference::dsd_controller::CheckCost,
    ) -> Option<(u32, f64)> {
        use crate::inference::dsd_controller::{
            expected_tokens_per_round, rated_beside_a_step_per_guess,
            rated_beside_a_step_per_round, Step,
        };
        let draft_ms = self.draft_ms.median()?;
        let step = Some(Step::OverTheChecksRoute(step));
        let alpha = self.acceptance.alpha();
        let expected = |g| {
            if self.by_depth {
                self.acceptance.expected_tokens(g)
            } else {
                expected_tokens_per_round(alpha, g)
            }
        };
        match self.by {
            By::Block => rated_beside_a_step_per_round(expected, step, check, most, draft_ms),
            By::Head => rated_beside_a_step_per_guess(expected, step, check, most, draft_ms),
        }
    }

    /// Zero, or `most` as a probe once `probe_after` rounds have declined —
    /// never for a guesser judged by the reply, whose record does not wait
    /// for a round of its own.
    fn decline_or_probe(&mut self, most: u32) -> u32 {
        if self.judged {
            return 0;
        }
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
/// a head that would have guessed and sat out [`MOST_ROUNDS_SAT_OUT`] rounds
/// is given one: its price is its own record, which moves only when it runs.
///
/// **The block is judged on every round it does not take** (2026-11-09,
/// [`judged_by_the_reply`]): it drafts beside the head's draft, the n-gram's
/// or a plain step, one shadow at a time, and the reply's next tokens say
/// how many of its guesses a check would have kept (`BlockShadow` — exact
/// under greedy decoding; HedgeSpec's full information, where the rule
/// before had a bandit's). Both are priced by their own depth curves
/// (`DepthAcceptance`), never one α. Measured only when it ran, a few rounds
/// set the block's price for the reply: on the field (35B-A3B here ↔
/// Australia, `~/rig428/fieldb1`) its three rounds of a class kept 12 of 24
/// while the head's 28 kept 80 of 101 — and the block alone wrote that
/// class 20-25% faster (9.7-10.4 tok/s against the head's 7.9-8.4; the pair
/// 6.5) — and a story's block took 23-33 rounds keeping one guess in five
/// (the pair 4.3-5.2, the head alone 6.9). After
/// [`SET_ASIDE_AFTER_JUDGED`] judged rounds in a row that it would not
/// have written faster than the round the reply took, it is set aside for
/// the reply ([`Self::rows_now_read`]): its rows — eight layers a position,
/// a 5-position answer 133,858 bytes against 10,944 — stop crossing.
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
    /// The block's draft beside a round it did not take, waiting for the
    /// reply's tokens to judge it ([`judged_by_the_reply`]).
    shadow: Option<crate::inference::dsd_controller::BlockShadow>,
    /// Shadows judged this reply, and the guesses of theirs the reply kept.
    pub(super) shadows: u32,
    pub(super) shadow_kept: u32,
    /// Judged in a row that the block would not have written faster.
    losing_run: u32,
    /// The trunk layer the head reads, and whether the block was set aside
    /// since the caller last asked ([`Self::rows_now_read`]).
    head_layer: Option<u32>,
    set_aside: bool,
    /// The reply's length when the block was set aside, for the log.
    pub(super) set_aside_at: Option<usize>,
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
                head_layer.map(|l| l as u32),
                super::check_reach(&exec.shared_state, segments),
                judged_by_the_reply(),
            ),
            BlockRowsAsk::new(layers, meta.embedding_length),
        ))
    }

    fn new(
        first: (u32, u32),
        block_most: Option<u32>,
        head_layer: Option<u32>,
        reach: crate::inference::dsd_controller::CheckReach,
        rule: bool,
    ) -> Self {
        Self {
            first,
            block: block_most.map(|most| Guesser::new(By::Block, most, rule)),
            head: head_layer.map(|_| Guesser::new(By::Head, HEAD_MOST_GUESSES, rule)),
            step_ms: Default::default(),
            check: crate::inference::dsd_controller::CheckCost::for_a_plan(reach),
            calls: 0,
            words_sent: 0,
            shadow: None,
            shadows: 0,
            shadow_kept: 0,
            losing_run: 0,
            head_layer,
            set_aside: false,
            set_aside_at: None,
        }
    }

    /// No guesser is left: the reply goes on without asking.
    pub(super) fn is_empty(&self) -> bool {
        self.block.is_none() && self.head.is_none()
    }

    /// This reply asks the block drafter beside the first segment nothing
    /// more — set aside, stopped, or never here — and a block is never taken
    /// back within a reply, so the worker may let its own go
    /// (`IpcBlockDraft::block_aside`).
    pub(super) fn block_aside_for_the_worker(&self) -> bool {
        self.block.is_none()
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
                    block_aside: self.block_aside_for_the_worker(),
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
        if asked != Some(By::Block) {
            self.shadow_the_block(exec, upto, token).await;
        }
        guessed
    }

    /// Draft the block beside a round it does not take, for the reply to
    /// judge ([`judged_by_the_reply`]): one shadow at a time, once it has
    /// drafted and been timed, with every guess it may make — one forward
    /// whatever it guesses. Its rows were taken in by the round's own call.
    async fn shadow_the_block(&mut self, exec: &PipelineExecutor, upto: usize, token: u32) {
        if self.shadow.is_some() {
            return;
        }
        let Some(most) = self
            .block
            .as_ref()
            .filter(|g| g.judged && g.draft_ms.median().is_some())
            .map(|g| g.most)
        else {
            return;
        };
        let started = std::time::Instant::now();
        let guessed = exec
            .shared_state
            .model_process_pool
            .block_draft(
                crate::inference::worker_ipc::IpcBlockDraft {
                    request_id: exec.request.id,
                    model_id: exec.request.model_id.clone(),
                    first_range: self.first,
                    rows: Vec::new(),
                    upto: upto as u32,
                    index_pos: upto as u32,
                    token,
                    gamma: most,
                    first: false,
                    words: Vec::new(),
                    by_head: false,
                    block_aside: self.block_aside_for_the_worker(),
                },
                &[],
            )
            .await;
        self.calls += 1;
        match guessed {
            Ok(guesses) if !guesses.is_empty() => {
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                if let Some(g) = self.block.as_mut() {
                    g.asked += 1;
                    g.record_draft(guesses.len(), ms);
                }
                self.shadow = Some(crate::inference::dsd_controller::BlockShadow {
                    from: upto + 1,
                    guesses,
                    followed: 0,
                    draft_ms: ms,
                });
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    request_id = %exec.request.id,
                    error = %e,
                    "the block drafter beside the first segment stopped for this reply"
                );
                self.block = None;
            }
        }
    }

    /// The reply committed `committed` from position `at` (`reply_len`
    /// tokens so far): a waiting shadow is judged by them, and the block is
    /// set aside once [`SET_ASIDE_AFTER_JUDGED`] judged rounds in a row say
    /// it would not have written faster — it could not beat a plain step,
    /// or the head was priced faster.
    pub(super) fn reply_committed(&mut self, at: usize, committed: &[u32], reply_len: usize) {
        let Some(shadow) = self.shadow.as_mut() else {
            return;
        };
        let Some(kept) = shadow.follow(at, committed) else {
            return;
        };
        let guessed = shadow.guesses.len();
        self.shadow = None;
        self.shadows += 1;
        self.shadow_kept += kept as u32;
        let Some(block) = self.block.as_mut() else {
            return;
        };
        block.acceptance.record(kept as u32, guessed as u32);
        let Some(step) = self.step_ms.median() else {
            return;
        };
        let head = self
            .head
            .as_ref()
            .and_then(|h| h.priced(h.most, step, &self.check));
        let losing = match block.priced(block.most, step, &self.check) {
            Some((0, _)) => true,
            Some((_, b)) => head.is_some_and(|(g, h)| g > 0 && h > b),
            None => false,
        };
        self.note_judgement(losing, Some(reply_len));
    }

    /// One judgement of the block: `losing` when it would not have written
    /// faster than a plain step or the head. Only a judgement it WON starts
    /// the count over — being chosen is not one: a story chose a block that
    /// kept ~1 guess in 5 15-19 times, each choice restarted the count, and it
    /// was set aside only at token 64 (`~/rig428/fieldc1`, 2026-11-09). A
    /// round it took is judged by what it wrote ([`Self::record_round`]).
    fn note_judgement(&mut self, losing: bool, reply_len: Option<usize>) {
        self.losing_run = if losing { self.losing_run + 1 } else { 0 };
        if self.losing_run >= SET_ASIDE_AFTER_JUDGED && self.block.is_some() {
            self.block = None;
            self.set_aside = true;
            self.set_aside_at = reply_len;
        }
    }

    /// Once, after the block was set aside: the layers the reply's forwards
    /// still ask rows of — the head's trunk layer, or none.
    pub(super) fn rows_now_read(&mut self) -> Option<Vec<u32>> {
        std::mem::take(&mut self.set_aside).then(|| self.head_layer.into_iter().collect())
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
        // its price. One judged by the reply never sits a round out.
        if other_gamma > 0 {
            if let Some(g) = self.guesser_mut(other).filter(|g| !g.judged) {
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
        // A judged block's own round is a judgement by what it WROTE — the
        // tokens a ms over its draft and check, against the best the others
        // are priced at (the head's price counts a plain step) — as one
        // computer sets a guesser aside (`dsd_controller::Delivered`).
        let judged = self
            .block
            .as_ref()
            .filter(|g| by == By::Block && g.judged)
            .map(|g| g.last_draft_ms);
        if let (Some(draft_ms), Some(step)) = (judged, step) {
            let wrote_per_ms = wrote as f64 / (check_ms + draft_ms).max(0.1);
            let head = self
                .head
                .as_ref()
                .and_then(|h| h.priced(h.most, step, &self.check))
                .map_or(0.0, |(_, rate)| rate);
            self.note_judgement(wrote_per_ms < head.max(1.0 / step), None);
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

    /// The rule before 2026-11-09 (`SWARMLLM_SPLIT_GUESS_PRICE=geometric`):
    /// the block measured only when checked, probed after declines.
    fn block_only(round_trip_ms: Option<u32>) -> SplitBlockDrafter {
        SplitBlockDrafter::new(
            (0, 17),
            Some(15),
            None,
            CheckReach::Peer { round_trip_ms },
            false,
        )
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
            Some(40),
            CheckReach::Peer {
                round_trip_ms: Some(130),
            },
            false,
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

    /// Both guessers under the rule of 2026-11-09, timed as on the field: a
    /// round over the split ~190 ms, checks on a line, the block's draft
    /// 20 ms, the head's guesses 3 ms.
    fn judged_pair() -> SplitBlockDrafter {
        let mut b = SplitBlockDrafter::new(
            (0, 1),
            Some(15),
            Some(40),
            CheckReach::Peer {
                round_trip_ms: Some(130),
            },
            true,
        );
        assert_eq!(b.gamma(200), (By::Block, 15), "the block drafts first");
        b.block.as_mut().unwrap().record_draft(15, 20.0);
        b.record_check(16, 600.0);
        b.record_round(By::Block, 1, 15, 2, 600.0);
        assert_eq!(
            b.gamma(200),
            (By::Head, HEAD_FIRST_GUESSES),
            "then the head"
        );
        b.head.as_mut().unwrap().record_draft(3, 9.0);
        b.record_check(4, 220.0);
        b.record_round(By::Head, 3, 3, 4, 220.0);
        b.record_step(190.0);
        for _ in 0..6 {
            b.record_check(5, 230.0);
            b.record_round(By::Head, 3, 4, 4, 230.0);
        }
        b
    }

    /// The reply's next tokens judge a shadow drafted at `from`: `kept` of its
    /// 15 guesses followed.
    fn judge(b: &mut SplitBlockDrafter, from: usize, kept: usize) {
        b.shadow = Some(crate::inference::dsd_controller::BlockShadow {
            from,
            guesses: (1..=15).collect(),
            followed: 0,
            draft_ms: 20.0,
        });
        let mut committed: Vec<u32> = (1..=kept as u32).collect();
        committed.push(99);
        b.reply_committed(from, &committed, from);
        assert!(b.shadow.is_none(), "decided");
    }

    /// A block the reply never follows is judged on the head's rounds — it
    /// never probes and never sits rounds out to be measured — and after
    /// `SET_ASIDE_AFTER_JUDGED` judged rounds that it would not have paid,
    /// it is set aside: the forwards ask the head's layer alone, once.
    #[test]
    fn a_judged_block_that_loses_is_set_aside_and_its_rows_stop() {
        let mut b = judged_pair();
        for i in 0..SET_ASIDE_AFTER_JUDGED as usize {
            assert!(b.block.is_some(), "still guessing after {i} judged");
            assert!(!b.block_aside_for_the_worker(), "the worker keeps it");
            judge(&mut b, 100 + 20 * i, 0);
            for _ in 0..FIRST_PROBE_AFTER + MOST_ROUNDS_SAT_OUT {
                let (by, gamma) = b.gamma(200);
                assert!(by == By::Head || gamma == 0, "the block took a round");
                if by == By::Head && gamma > 0 {
                    b.record_round(By::Head, 3, gamma, gamma as usize, 230.0);
                }
            }
            if let Some(block) = b.block.as_ref() {
                assert_eq!((block.probes, block.sat_out), (0, 0));
            }
        }
        assert!(b.block.is_none(), "set aside");
        assert!(
            b.block_aside_for_the_worker(),
            "and the worker is told, before rows that carry only the head's layer reach it"
        );
        assert_eq!(b.set_aside_at, Some(100 + 20 * 7));
        assert_eq!(b.rows_now_read(), Some(vec![40]), "the head's layer");
        assert_eq!(b.rows_now_read(), None, "once");
        assert_eq!((b.shadows, b.shadow_kept), (SET_ASIDE_AFTER_JUDGED, 0));
    }

    /// A block whose judged runs the reply follows twelve deep is priced
    /// over the head and takes the round, and a judged win starts the
    /// count towards setting it aside over.
    #[test]
    fn a_judged_block_the_reply_follows_takes_the_round() {
        let mut b = judged_pair();
        for i in 0..2 {
            judge(&mut b, 100 + 20 * i, 0);
        }
        for i in 0..6 {
            judge(&mut b, 300 + 20 * i, 12);
        }
        assert_eq!(b.losing_run, 0);
        let (by, gamma) = b.gamma(200);
        assert_eq!(by, By::Block, "the block writes faster");
        assert!(gamma >= 8, "deep enough to keep its run: {gamma}");
        assert!(b.block.is_some());
    }

    /// The story of 2026-11-09 (`~/rig428/fieldc1`): a block the reply
    /// followed early is priced over the head and chosen, then keeps 3 of 15
    /// in its own rounds. Each is judged by what it wrote and being chosen
    /// starts nothing over, so it is set aside after `SET_ASIDE_AFTER_JUDGED`
    /// losing judgements — its own rounds and the head's shadows together.
    #[test]
    fn a_block_chosen_often_while_keeping_little_is_set_aside() {
        let mut b = judged_pair();
        for i in 0..6 {
            judge(&mut b, 100 + 20 * i, 12);
        }
        let (mut judged, mut chosen, mut at) = (0, 0, 300);
        while b.block.is_some() && judged < 3 * SET_ASIDE_AFTER_JUDGED {
            match b.gamma(200) {
                (By::Block, gamma) if gamma > 0 => {
                    chosen += 1;
                    block(&mut b).record_draft(gamma as usize, 20.0);
                    b.record_check(gamma + 1, 600.0);
                    b.record_round(By::Block, 3, gamma, 4, 600.0);
                }
                (by, gamma) => {
                    if by == By::Head && gamma > 0 {
                        b.record_round(By::Head, 3, gamma, gamma as usize, 230.0);
                    }
                    judge(&mut b, at, 3);
                }
            }
            judged += 1;
            at += 20;
        }
        assert!(chosen >= 2, "chosen while it lost: {chosen}");
        assert!(b.block.is_none(), "set aside");
        assert_eq!(judged, SET_ASIDE_AFTER_JUDGED);
        assert_eq!(b.rows_now_read(), Some(vec![40]), "the head's layer");
    }

    /// A block that wins its own rounds — more tokens a ms over its draft and
    /// check than the head is priced at — starts the count over, so losing
    /// shadows on the head's rounds in between never add up to setting it
    /// aside.
    #[test]
    fn a_block_winning_its_own_rounds_survives_the_heads_in_between() {
        let mut b = judged_pair();
        let mut at = 100;
        for _ in 0..3 {
            for _ in 1..SET_ASIDE_AFTER_JUDGED {
                judge(&mut b, at, 0);
                at += 20;
            }
            block(&mut b).record_draft(15, 20.0);
            b.record_check(16, 600.0);
            b.record_round(By::Block, 14, 15, 15, 600.0);
            assert_eq!(b.losing_run, 0, "a round it won");
        }
        assert!(b.block.is_some());
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
