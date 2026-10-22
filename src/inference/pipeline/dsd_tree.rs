//! Rounds that check a TREE of guesses (#134, `docs/plans/large_model_splits.md`
//! § "Draft trees"): the bookkeeping `dsd`'s round loop does around one.
//!
//! A chain's check keeps the run of guesses before the first miss; a tree's
//! keeps the longest run along whichever branch the target takes, for the same
//! round trip. Where a split's machines are far apart that round trip is most
//! of a token's cost, so the tokens a check keeps are what sets the speed.
//!
//! A check's rows are, in order: the kept guesses of the last round whose
//! cache rows sat out of place (read again), the check's own last token, then
//! the guesses worth their row ([`worth_checking`]). The tail walks
//! the tree where the logits are and answers the tokens it kept; the rows of
//! the kept path that follow the check's own token in place stay in every
//! segment's cache, and the rest are read again at the head of the next check
//! — never gathered into place, which no peer would need to learn.

use crate::error::SwarmError;
use crate::inference::draft_tree::{
    keep_rows, kept_rows, worth_checking, CheckPrice, DraftTree, GrownTree,
};

/// May a reply's rounds check trees? Where every segment runs its layers on a
/// card (`pipeline::every_segment_runs_on_a_card`) AND so does the drafter
/// (`ModelProcessPool::runs_whole_on_the_card`) AND the farthest peer segment
/// is a measured [`TREES_LEAST_ROUND_TRIP_MS`] or more away
/// (`pipeline::farthest_segment_round_trip_ms`), and anywhere with
/// `SWARMLLM_DRAFT_TREES=1`; never with `=0`.
///
/// A tree's rows are what it costs a check, so it needs them cheap: with the
/// head's part on its processor (the 385 ms rig, 2026-10-20) rows cost 4-18
/// ms, trees kept 2.9-3.3 and ran 4.75-5.06 tok/s against the stream's
/// 5.03-5.09. And it needs the drafter fast, because a check whose outcome the
/// tree grown beside it did not hold (most of them: 40-52 of 59-78 rounds)
/// waits for a fresh tree, eight drafter forwards. On the TH↔AU link
/// (2026-10-21, a 7B split, the head's layers on this machine's card, the tail
/// an RTX 3060 ~410 ms away) with the Q4 drafter kept on the PROCESSOR that
/// wait was 289-331 ms of a ~850 ms round, and trees won at 160 tokens
/// (5.33-5.74 against the stream's 5.01-5.05, four replies each) but lost at
/// 400 (4.82/5.05/6.40 against 5.97/6.04/6.18) and at 28. On a card the same
/// eight forwards take ~40 ms, and on the same link with the 0.5B drafter on
/// this machine's card (2026-10-22, `cardrig247.sh`: arms alternating by node
/// start, four replies a length each) trees ran 8.41 tok/s mean at 160 tokens
/// against the stream's 5.95 and 9.36 at 400 against 6.48 (+41-44%; 26-70
/// rounds, 4.7-5.3 kept a round), every reply rank-1 against llama.cpp at
/// 394-397 of 400 tokens with near-ties only (largest gap 0.21). A plan that
/// fills the head's card leaves the drafter room there for this (#275 (4),
/// `ModelProcessPool::max_hostable_layers_beside_guest`). Every node
/// advertises `features::DRAFT_TREES` and checks a tree it is sent, so a peer
/// needs no update for this to change.
///
/// The round trip: trees were measured winning only over 385-425 ms links,
/// and a short link is where they lose most. There a check costs ~30 ms, and
/// a missed seed's regrow (eight drafter forwards, ~155 ms on a card) is most
/// of the round. The schedule model (`examples/tree_schedule_sim.py`,
/// 2026-10-21, calibrated to TH↔AU within ~10%) puts tree rounds at 21.7
/// tok/s against the chain stream's 47.6 at 20 ms with the drafter on a card.
/// Sizing the regrow ([`TreeRounds::grow_size`]) brings that to 41.0. Unknown
/// is short: the chain stream was measured on every link it runs on.
pub(super) fn trees_wanted(
    every_segment_on_a_card: bool,
    drafter_on_a_card: bool,
    farthest_round_trip_ms: Option<u32>,
) -> bool {
    trees_wanted_given(
        std::env::var("SWARMLLM_DRAFT_TREES").ok().as_deref(),
        every_segment_on_a_card,
        drafter_on_a_card,
        farthest_round_trip_ms,
    )
}

/// The shortest round trip to the farthest peer segment at which trees are
/// the default. Between ~150 and ~400 ms the schedule model puts tree rounds
/// within its own error of the chain stream (−6% at 300 ms, against a chain
/// stream it prices 25-50% fast on a long link, where it leaves out each
/// chunk's fixed cost and a restart's thrown-away work). So the line is at
/// 300 ms, under the shortest link trees were measured winning on (385 ms),
/// until a rig at 20 and 300 ms with the drafter on a card moves it.
pub(super) const TREES_LEAST_ROUND_TRIP_MS: u32 = 300;

fn trees_wanted_given(
    setting: Option<&str>,
    every_segment_on_a_card: bool,
    drafter_on_a_card: bool,
    farthest_round_trip_ms: Option<u32>,
) -> bool {
    match setting.map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => {
            every_segment_on_a_card
                && drafter_on_a_card
                && farthest_round_trip_ms.is_some_and(|ms| ms >= TREES_LEAST_ROUND_TRIP_MS)
        }
    }
}

/// Does the drafter keep growing a tree while its check is on the way?
/// `SWARMLLM_TREE_GROW_BESIDE=0` grows each tree only before its check, for an
/// A/B inside one binary.
pub(super) fn grow_beside_requested() -> bool {
    std::env::var("SWARMLLM_TREE_GROW_BESIDE").as_deref() != Ok("0")
}

/// Guesses a call grown beside a check may add, and read a forward. The time
/// given (`TreeRounds::beside_budget_ms`) is what usually stops it; this caps
/// the cache it takes on a fast drafter.
pub(super) const BESIDE_ROWS: u32 = 128;
pub(super) const BESIDE_PER_STEP: u32 = 8;
/// Rows of a grown tree carried into the next round as its seed: each is read
/// again by the call that grows on it, so the likeliest are kept.
const SEED_MOST: usize = 96;
/// Of the time a check is predicted to take, the share a call beside it is
/// given: past its answer the call holds the round up, short of it only
/// grows less.
const BESIDE_SHARE: f64 = 0.85;
/// A call beside a check given less than this would read the seed and stop.
const BESIDE_LEAST_MS: f64 = 15.0;

/// Rows of the tree a drafter grows, at most and at least: past 64 a check
/// gains well under a token per doubling (`docs/plans/large_model_splits.md` §
/// "Projections"); the check keeps only the rows that pay.
const MOST_ROWS_GROWN: u32 = 64;
const FEWEST_ROWS_GROWN: u32 = 16;
/// Drafter forwards a tree is grown in at most — the deepest path it can
/// hold. Each forward reads `rows / STEPS` guesses at once, which costs a card
/// little more than one.
const STEPS: u32 = 8;
/// The depths a tree grown before its check may be given
/// ([`TreeRounds::grow_size`]): fewer forwards where the check is short.
const STEP_CHOICES: [u32; 4] = [1, 2, 4, STEPS];

/// One reply's tree rounds.
#[derive(Debug)]
pub(super) struct TreeRounds {
    /// How far the drafter's ratings have matched what checks kept (kept ÷
    /// the ratings' sum): a drafter's confidence approximates its acceptance,
    /// but not exactly, and the price of a row depends on it.
    calibration: f64,
    /// Kept guesses whose cache rows sat out of place: read again at the head
    /// of the next check, which must then be a tree's.
    pub reread: Vec<u32>,
    /// A segment refused to check a tree: chains for the rest of the reply.
    pub refused: bool,
    /// Guesses checked a round, recently — what the next tree is grown for.
    recent_sent: f64,
    /// Guesses kept a round, recently.
    recent_kept: f64,
    /// The next check's tree, grown while the last check was on its way and
    /// cut to the part under that check's outcome (`GrownTree::after`).
    seed: Option<GrownTree>,
    /// What one drafter forward of a tree grown before its check took,
    /// recently (ms): the growth's time over its steps and one more, the
    /// call's read of what came before.
    forward_ms: Option<f64>,
    /// Levels of the checked trees that held the reply's token, and walks that
    /// stopped where the tree had followers and none was the reply's, both
    /// recently: [`Self::level_acceptance`]'s two counts.
    level_hits: f64,
    level_misses: f64,
    /// Is a growth before the check sized by what it buys, or always
    /// [`STEPS`] deep? `SWARMLLM_TREE_REGROW=fixed` for an A/B inside one
    /// binary.
    sized_growth: bool,
    /// How far a call grown beside a check ran past the check's answer,
    /// recently (ms) — taken off the next one's time.
    beside_overrun_ms: f64,
    pub rounds: u32,
    pub guesses_sent: u32,
    pub guesses_kept: u32,
    pub rows_carried: u32,
    /// Rounds whose tree was grown beside the check before, with no drafting
    /// in their way.
    pub rounds_seeded: u32,
    /// Checks whose outcome the tree grown beside them held no path for.
    pub seed_misses: u32,
    /// Drafting the rounds waited for, and drafting beside their checks (ms).
    pub draft_ms_waited: f64,
    pub draft_ms_beside: f64,
    /// Trees grown before their check, and the drafter forwards they were
    /// given in all ([`Self::grow_size`]).
    pub grown_before: u32,
    pub steps_grown_before: u32,
}

/// One tree check, ready to send.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TreeCheck {
    /// Every row's token: the kept tokens read again, the check's own last
    /// token, then the guesses.
    pub verify_tokens: Vec<u32>,
    /// Every row's parent, in `LayerForward::draft_parents`' wire form.
    pub parents: Vec<u16>,
    /// The guesses alone — what the tail walks (`draft_tokens`).
    pub guesses: Vec<u32>,
    /// The first guess's row.
    guessed_from: usize,
    /// The guesses' ratings summed: how many the drafter expects kept.
    rated: f64,
}

/// What a walked tree check kept.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TreeOutcome {
    /// The guesses kept, in order.
    pub kept: Vec<u32>,
    /// The token the tail sampled where its walk stopped.
    pub next: u32,
    /// Rows of the forward that stay in the cache: every row before the
    /// guesses, then the kept guesses that sit in place right after them.
    pub rows_in_place: usize,
}

impl TreeRounds {
    pub(super) fn new() -> Self {
        Self {
            calibration: 1.0,
            reread: Vec::new(),
            refused: false,
            recent_sent: f64::from(MOST_ROWS_GROWN),
            recent_kept: 0.0,
            seed: None,
            forward_ms: None,
            // A drafter's likeliest guess is the reply's about three times in
            // four (Sequoia's positional figures, as
            // `examples/tree_schedule_sim.py` takes them): a = 0.75 until
            // walks say otherwise.
            level_hits: 3.0,
            level_misses: 1.0,
            sized_growth: std::env::var("SWARMLLM_TREE_REGROW").as_deref() != Ok("fixed"),
            beside_overrun_ms: 0.0,
            rounds: 0,
            guesses_sent: 0,
            guesses_kept: 0,
            rows_carried: 0,
            rounds_seeded: 0,
            seed_misses: 0,
            draft_ms_waited: 0.0,
            draft_ms_beside: 0.0,
            grown_before: 0,
            steps_grown_before: 0,
        }
    }

    pub(super) fn calibration(&self) -> f64 {
        self.calibration
    }

    /// The tree grown beside the last check for this one, if any — empty
    /// where none was, or the check's outcome left nothing of it.
    pub(super) fn take_seed(&mut self) -> GrownTree {
        self.seed.take().unwrap_or_default()
    }

    /// Forget the seed: the next check does not follow the last one's
    /// outcome (a refused check, a drafter that failed).
    pub(super) fn drop_seed(&mut self) {
        self.seed = None;
    }

    /// Should the drafter grow `seed` further before this check goes out, or
    /// is the seed checked as it is? Growing pays where the tokens it adds a
    /// round outweigh the time it adds to the round: `(1 + grown) / (check +
    /// grow) > (1 + seed) / check`, each side's tokens the check's sample plus
    /// the guesses expected kept (the calibrated ratings' sum, the same
    /// reading [`worth_checking`] prices rows by; a grown tree is expected to
    /// keep what rounds have recently kept), the growth the one
    /// [`Self::grow_size`] would give. An empty seed, or one with no measured
    /// check or growth to weigh it by, is grown.
    pub(super) fn grow_before_the_check(
        &self,
        seed: &GrownTree,
        price: Option<(f64, f64)>,
    ) -> bool {
        let (rows, per_step) = self.grow_size(price);
        let (Some((fixed_ms, per_row_ms)), Some(grow_ms)) =
            (price, self.growth_ms(rows.div_ceil(per_step)))
        else {
            return true;
        };
        if seed.tokens.is_empty() {
            return true;
        }
        let from_seed = self.calibration
            * seed
                .path_logp
                .iter()
                .map(|&r| f64::from(r).exp())
                .sum::<f64>();
        let from_growth = from_seed.max(self.recent_kept);
        let rows = (self.reread.len() + 1) as f64 + self.recent_sent;
        let check_ms = (fixed_ms + per_row_ms * rows).max(1.0);
        (1.0 + from_growth) / (check_ms + grow_ms) > (1.0 + from_seed) / check_ms
    }

    /// Record a tree grown before its check in `steps` drafter forwards
    /// (`ms` waited for it).
    pub(super) fn note_grown_before(&mut self, ms: f64, steps: u32) {
        self.draft_ms_waited += ms;
        self.grown_before += 1;
        self.steps_grown_before += steps;
        let per_forward = ms / f64::from(steps + 1);
        self.forward_ms = Some(
            self.forward_ms
                .map_or(per_forward, |f| 0.7 * f + 0.3 * per_forward),
        );
    }

    /// What growing a tree in `steps` forwards should take (ms), from what
    /// the drafter's recent forwards took: `steps` and one more, the call's
    /// read of what came before. `None` before any growth was timed.
    fn growth_ms(&self, steps: u32) -> Option<f64> {
        self.forward_ms.map(|f| f * f64::from(steps + 1))
    }

    /// How often a level of a checked tree held the reply's token, recently:
    /// kept guesses over kept guesses plus the walks that stopped with
    /// followers to choose from (a geometric run's estimate where some runs
    /// are cut short — a walk that reached a leaf saw the tree end, not a
    /// miss, so a shallow tree does not teach that the drafter is poor).
    pub(super) fn level_acceptance(&self) -> f64 {
        (self.level_hits / (self.level_hits + self.level_misses)).clamp(0.0, 0.99)
    }

    /// The time a call growing beside a check of `rows` rows is given (ms):
    /// most of what the check is predicted to take, less what such calls
    /// recently ran past their checks' answers — or, before any check was
    /// timed, what growing a tree before one took. `None` where that leaves
    /// too little to grow anything.
    pub(super) fn beside_budget_ms(&self, price: Option<(f64, f64)>, rows: usize) -> Option<u32> {
        let predicted = match price {
            Some((fixed_ms, per_row_ms)) => fixed_ms + per_row_ms * rows as f64,
            None => self.growth_ms(STEPS)?,
        };
        let given = BESIDE_SHARE * predicted - self.beside_overrun_ms;
        (given >= BESIDE_LEAST_MS).then_some(given as u32)
    }

    /// A call grown beside a check came back `call_ms` after it was sent; the
    /// check's answer, `check_ms` after. The tree it grew, cut to the part
    /// under what the round emitted, is the next check's seed.
    pub(super) fn note_grown_beside(
        &mut self,
        grown: &GrownTree,
        emitted: &[u32],
        call_ms: f64,
        check_ms: f64,
    ) {
        self.draft_ms_beside += call_ms;
        let overrun = (call_ms - check_ms).max(0.0);
        self.beside_overrun_ms = 0.7 * self.beside_overrun_ms + 0.3 * overrun;
        let Some((&next, kept)) = emitted.split_last() else {
            self.seed = None;
            return;
        };
        self.seed = grown
            .after(kept, next, SEED_MOST)
            .filter(|seed| !seed.tokens.is_empty());
        if self.seed.is_none() {
            self.seed_misses += 1;
        }
    }

    /// This round's tree was the seed alone, with no drafting in its way.
    pub(super) fn note_seeded(&mut self) {
        self.rounds_seeded += 1;
    }

    /// The tree the drafter grows before a check, `(rows, per_step)`: as wide
    /// a forward as twice what checks have recently kept a place for, between
    /// [`FEWEST_ROWS_GROWN`] and [`MOST_ROWS_GROWN`] over [`STEPS`] forwards,
    /// and as many forwards of the [`STEP_CHOICES`] as buy the most tokens per
    /// ms of the round (#134 (a)): `(1 + E(s)) / (check(s) + grow(s))`, with
    /// E(s) = a + a² + … + aˢ at [`Self::level_acceptance`]'s a, the check at
    /// its fitted `price` for the rows grown, the growth at the drafter's
    /// measured forward. Sequoia's size rule (arXiv 2402.12374) over the
    /// round's measured terms. On a long link the check outweighs any growth
    /// and the deepest tree is grown; on a short one a missed seed waited for
    /// eight forwards before a ~30 ms check, and tree rounds ran at half the
    /// chain stream (`examples/tree_schedule_sim.py`, 2026-10-21: 21.7 tok/s
    /// against 47.6 at 20 ms with the drafter on a card; sized, 41.0). With
    /// no check or growth measured yet, or `SWARMLLM_TREE_REGROW=fixed`,
    /// [`STEPS`].
    pub(super) fn grow_size(&self, price: Option<(f64, f64)>) -> (u32, u32) {
        let wanted =
            ((2.0 * self.recent_sent).ceil() as u32).clamp(FEWEST_ROWS_GROWN, MOST_ROWS_GROWN);
        let per_step = wanted.div_ceil(STEPS).max(1);
        let steps = match (price, self.forward_ms) {
            (Some((fixed_ms, per_row_ms)), Some(forward_ms)) if self.sized_growth => {
                let a = self.level_acceptance();
                let lead = (self.reread.len() + 1) as f64;
                let value = |s: u32| {
                    let kept: f64 = (1..=s).map(|i| a.powi(i as i32)).sum();
                    let check_ms =
                        (fixed_ms + per_row_ms * (lead + f64::from(per_step * s))).max(1.0);
                    (1.0 + kept) / (check_ms + forward_ms * f64::from(s + 1))
                };
                // Deepest first, so a tie keeps the deeper tree.
                STEP_CHOICES
                    .iter()
                    .rev()
                    .map(|&s| (s, value(s)))
                    .fold(
                        (STEPS, f64::MIN),
                        |best, (s, v)| if v > best.1 { (s, v) } else { best },
                    )
                    .0
            }
            _ => STEPS,
        };
        (per_step * steps, per_step)
    }

    /// The check for this round: the rows read again, `last`, then the guesses
    /// of `grown` worth their row at `price` (the check's measured
    /// `(fixed, per_position)`; `None` before any check, when every guess is
    /// sent). No path deeper than `room` tokens is sent: the reply ends first.
    ///
    /// Where the price has no slope yet — every check so far carried as many
    /// rows — every other round is held to a quarter of the rows, so the next
    /// fit can tell what a row costs.
    pub(super) fn check(
        &self,
        last: u32,
        grown: &GrownTree,
        price: Option<(f64, f64)>,
        room: usize,
    ) -> Result<TreeCheck, SwarmError> {
        let lead = self.reread.len() + 1;
        let n = grown.tokens.len();
        if grown.parents.len() != n || grown.path_logp.len() != n {
            return Err(SwarmError::Internal(
                "a grown tree's tokens, parents and ratings disagree in length".into(),
            ));
        }
        // Depth of each guess (0 = follows `last`); a parent comes first.
        let mut depth = vec![0usize; n];
        for i in 0..n {
            depth[i] = match grown.parents[i] {
                None => 0,
                Some(p) if usize::from(p) < i => depth[usize::from(p)] + 1,
                Some(_) => {
                    return Err(SwarmError::Internal(
                        "a grown tree names a parent after its child".into(),
                    ))
                }
            };
        }
        // A guess past the room left can never be emitted: rated as never kept.
        let rated: Vec<f32> = grown
            .path_logp
            .iter()
            .zip(&depth)
            .map(|(&r, &d)| if d + 1 >= room { f32::NEG_INFINITY } else { r })
            .collect();
        let kept = match price {
            Some((fixed_ms, per_row_ms)) => {
                let most = if per_row_ms <= 0.0 && self.rounds % 2 == 1 {
                    n / 4
                } else {
                    n
                };
                worth_checking(
                    &rated,
                    lead,
                    CheckPrice {
                        fixed_ms,
                        per_row_ms,
                    },
                    self.calibration,
                    most,
                )
            }
            None => (0..n).filter(|&i| rated[i].is_finite()).collect(),
        };
        let parents = keep_rows(&grown.parents, &kept).ok_or_else(|| {
            SwarmError::Internal("the guesses chosen for a check are not a tree".into())
        })?;
        let mut verify_tokens = self.reread.clone();
        verify_tokens.push(last);
        let guesses: Vec<u32> = kept.iter().map(|&i| grown.tokens[i]).collect();
        verify_tokens.extend_from_slice(&guesses);
        // Wire form: 0 follows the cache, k follows row k − 1. The lead rows
        // are a chain from the cache; a guess with no parent follows `last`.
        let wire: Vec<u16> = (0..lead as u16)
            .chain(parents.iter().map(|p| match p {
                None => lead as u16,
                Some(p) => lead as u16 + 1 + p,
            }))
            .collect();
        Ok(TreeCheck {
            verify_tokens,
            parents: wire,
            guesses,
            guessed_from: lead,
            rated: kept.iter().map(|&i| f64::from(rated[i]).exp()).sum(),
        })
    }

    /// Read a walked check back (`walked`: the guesses the tail kept, then its
    /// sample), refusing a walk no honest tail sends, and learn from it: which
    /// kept tokens must be read again, and how far the ratings held.
    pub(super) fn settle(
        &mut self,
        check: &TreeCheck,
        walked: &[u32],
    ) -> Result<TreeOutcome, SwarmError> {
        let Some((&next, kept)) = walked.split_last() else {
            return Err(SwarmError::Inference(
                "tree check: the last segment walked and returned no token".into(),
            ));
        };
        // The base only places rows; any position reads the shape the same.
        let tree = DraftTree::from_wire(1, &check.parents).map_err(SwarmError::Internal)?;
        let rows =
            kept_rows(&tree, check.guessed_from, &check.verify_tokens, kept).ok_or_else(|| {
                SwarmError::Inference(format!(
                    "tree check: the last segment kept {} tokens no path of the tree carries",
                    kept.len()
                ))
            })?;
        let in_place = rows
            .iter()
            .enumerate()
            .take_while(|&(i, &r)| r == check.guessed_from + i)
            .count();
        // Each kept guess is a level that held the reply's token; a walk that
        // stopped where the tree had followers is one that did not. One that
        // stopped at a leaf saw the tree end, and says nothing.
        let stopped_at = rows
            .last()
            .copied()
            .unwrap_or(check.guessed_from.saturating_sub(1));
        let missed = tree
            .children(Some(stopped_at))
            .any(|c| c >= check.guessed_from);
        self.level_hits = 0.7 * self.level_hits + 0.3 * kept.len() as f64;
        self.level_misses = 0.7 * self.level_misses + 0.3 * f64::from(u8::from(missed));
        self.reread = kept[in_place..].to_vec();
        if check.rated > 0.05 {
            let held = kept.len() as f64 / check.rated;
            self.calibration = (0.7 * self.calibration + 0.3 * held).clamp(0.05, 2.0);
        }
        let sent = check.guesses.len();
        self.recent_sent = 0.7 * self.recent_sent + 0.3 * sent as f64;
        self.recent_kept = if self.rounds == 0 {
            kept.len() as f64
        } else {
            0.7 * self.recent_kept + 0.3 * kept.len() as f64
        };
        self.rounds += 1;
        self.guesses_sent += sent as u32;
        self.guesses_kept += kept.len() as u32;
        self.rows_carried += check.verify_tokens.len() as u32;
        Ok(TreeOutcome {
            kept: kept.to_vec(),
            next,
            rows_in_place: check.guessed_from + in_place,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Laid out trunk first, as a drafter grows it: 10 → 12 → 13 is the
    /// likeliest path; 11 also follows the last token.
    fn grown() -> GrownTree {
        GrownTree {
            tokens: vec![10, 12, 13, 11],
            parents: vec![None, Some(0), Some(1), None],
            path_logp: [0.8f64, 0.6, 0.5, 0.15]
                .iter()
                .map(|p| p.ln() as f32)
                .collect(),
        }
    }

    const FAR: Option<u32> = Some(420);

    /// Trees by default where every segment runs on a card; the setting
    /// forces them on (`1`) or off (`0`) whatever the plan, and anything else
    /// leaves the choice to the plan.
    #[test]
    fn trees_are_wanted_on_cards_unless_the_setting_says_otherwise() {
        assert!(trees_wanted_given(None, true, true, FAR));
        assert!(!trees_wanted_given(None, false, true, FAR));
        assert!(trees_wanted_given(Some("1"), false, false, None));
        assert!(trees_wanted_given(Some(" 1 "), false, false, None));
        assert!(!trees_wanted_given(Some("0"), true, true, FAR));
        assert!(trees_wanted_given(Some("yes"), true, true, FAR));
        assert!(!trees_wanted_given(Some("yes"), false, true, FAR));
    }

    /// The segments on cards are not enough: a drafter on the processor
    /// regrows a missed tree in ~300 ms, and trees lost to the stream that way
    /// over 400-token replies on the real link (2026-10-21).
    #[test]
    fn trees_are_not_the_default_where_the_drafter_runs_on_the_processor() {
        assert!(!trees_wanted_given(None, true, false, FAR));
        assert!(
            trees_wanted_given(Some("1"), true, false, FAR),
            "the switch still forces them"
        );
    }

    /// Cards everywhere are not enough either: over a short link a check
    /// costs ~30 ms and the chain stream beat tree rounds two to one in the
    /// schedule model, so trees are the default only where the farthest
    /// segment is measured far away, and an unmeasured one is not.
    #[test]
    fn trees_are_the_default_only_over_a_long_measured_link() {
        assert!(!trees_wanted_given(None, true, true, Some(20)));
        assert!(!trees_wanted_given(
            None,
            true,
            true,
            Some(TREES_LEAST_ROUND_TRIP_MS - 1)
        ));
        assert!(trees_wanted_given(
            None,
            true,
            true,
            Some(TREES_LEAST_ROUND_TRIP_MS)
        ));
        assert!(!trees_wanted_given(None, true, true, None), "unmeasured");
        assert!(
            trees_wanted_given(Some("1"), true, true, Some(20)),
            "the switch still forces them"
        );
    }

    #[test]
    fn a_check_reads_the_kept_rows_again_then_its_token_then_the_guesses() {
        let mut t = TreeRounds::new();
        t.reread = vec![5, 6];
        let c = t.check(7, &grown(), None, 100).unwrap();
        assert_eq!(c.verify_tokens, vec![5, 6, 7, 10, 12, 13, 11]);
        // 5 follows the cache, 6 follows 5, 7 follows 6; 10 follows 7, 12
        // follows 10, 13 follows 12; 11 follows 7.
        assert_eq!(c.parents, vec![0, 1, 2, 3, 4, 5, 3]);
        assert_eq!(c.guesses, vec![10, 12, 13, 11]);
        let tree = DraftTree::from_wire(40, &c.parents).unwrap();
        assert_eq!(tree.depth(5), 5, "13 is the third guess after 7");
        assert_eq!(tree.depth(6), 3, "11 is the first");
    }

    #[test]
    fn a_path_past_the_room_left_is_not_sent() {
        let t = TreeRounds::new();
        // Room for 2 more tokens: one guess and the sample.
        let c = t.check(7, &grown(), None, 2).unwrap();
        assert_eq!(c.guesses, vec![10, 11]);
        assert_eq!(c.parents, vec![0, 1, 1]);
    }

    #[test]
    fn the_kept_path_in_place_stays_and_the_rest_is_read_again() {
        let mut t = TreeRounds::new();
        let c = t.check(7, &grown(), None, 100).unwrap();
        // The trunk 10 → 12 → 13 sits in place: all three stay.
        let o = t.settle(&c, &[10, 12, 13, 99]).unwrap();
        assert_eq!(
            (o.kept.clone(), o.next, o.rows_in_place),
            (vec![10, 12, 13], 99, 1 + 3)
        );
        assert!(t.reread.is_empty());
        // 11 sits after 10's row: nothing after the check's token stays in place.
        let c = t.check(99, &grown(), None, 100).unwrap();
        let o = t.settle(&c, &[11, 98]).unwrap();
        assert_eq!(o.rows_in_place, 1);
        assert_eq!(t.reread, vec![11]);
        // And the next check opens with it.
        let c = t.check(98, &grown(), None, 100).unwrap();
        assert_eq!(&c.verify_tokens[..2], &[11, 98]);
    }

    #[test]
    fn a_walk_that_no_path_carries_is_refused() {
        let mut t = TreeRounds::new();
        let c = t.check(7, &grown(), None, 100).unwrap();
        assert!(t.settle(&c, &[10, 13, 1]).is_err(), "13 does not follow 10");
        assert!(t.settle(&c, &[]).is_err());
    }

    /// A tree grown beside a check becomes the next check's seed: the part
    /// under what the round emitted, re-rated from its new root; an outcome
    /// it holds no path for is a miss and leaves no seed.
    #[test]
    fn the_tree_grown_beside_a_check_seeds_the_next_by_its_outcome() {
        let mut t = TreeRounds::new();
        // Kept 10, sampled 12: what follows is 13.
        t.note_grown_beside(&grown(), &[10, 12], 300.0, 400.0);
        let seed = t.take_seed();
        assert_eq!(seed.tokens, vec![13]);
        assert_eq!(seed.parents, vec![None]);
        assert!((seed.path_logp[0] - (0.5f32.ln() - 0.6f32.ln())).abs() < 1e-5);
        assert_eq!(t.seed_misses, 0);
        assert!(t.take_seed().tokens.is_empty(), "a seed is taken once");
        // Sampled 99 after the check's own token: nothing grown follows it.
        t.note_grown_beside(&grown(), &[99], 300.0, 400.0);
        assert!(t.take_seed().tokens.is_empty());
        assert_eq!(t.seed_misses, 1);
    }

    /// A call beside a check is given most of the check's predicted time,
    /// less what such calls have run past their checks; too little time is no
    /// call at all; before any check was timed, the time growing one took.
    #[test]
    fn a_call_beside_a_check_is_given_the_time_the_check_takes() {
        let mut t = TreeRounds::new();
        assert_eq!(t.beside_budget_ms(None, 20), None, "nothing measured yet");
        t.note_grown_before(200.0, STEPS);
        assert_eq!(t.beside_budget_ms(None, 20), Some(170));
        assert_eq!(t.beside_budget_ms(Some((400.0, 5.0)), 20), Some(425));
        assert_eq!(t.beside_budget_ms(Some((10.0, 0.0)), 20), None);
        // A call that ran 200 ms past its check's answer shortens the next.
        t.note_grown_beside(&grown(), &[10, 12], 600.0, 400.0);
        assert_eq!(t.beside_budget_ms(Some((400.0, 5.0)), 20), Some(365));
    }

    /// Growing a seed before its check pays only where the tokens it adds
    /// outweigh the time it adds to the round.
    #[test]
    fn a_seed_is_grown_before_its_check_only_where_that_pays() {
        let mut t = TreeRounds::new();
        let seed = grown();
        assert!(t.grow_before_the_check(&seed, None), "nothing measured");
        assert!(t.grow_before_the_check(&GrownTree::default(), Some((400.0, 1.0))));
        // Rounds have kept 3 guesses; the seed's ratings promise ~2.
        for _ in 0..5 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[10, 12, 13, 1]).unwrap();
        }
        t.calibration = 1.0;
        // A drafter on a card grows in 50 ms beside a 900 ms check: grow.
        t.note_grown_before(50.0, STEPS);
        assert!(t.grow_before_the_check(&seed, Some((900.0, 0.0))));
        // One on the processor takes 1500 ms: check the seed as it is.
        t.forward_ms = Some(1500.0 / f64::from(STEPS + 1));
        assert!(!t.grow_before_the_check(&seed, Some((900.0, 0.0))));
    }

    /// A missed seed is regrown only as deep as pays (#134 (a)). A card
    /// drafter's forward is ~19 ms (a 64-row tree from nothing in 155-171
    /// ms): before a ~30 ms check on a short link one forward buys the most
    /// tokens per ms of the round, and eight cost the round six times the
    /// check; before a ~425 ms check all eight do. Unmeasured, eight.
    #[test]
    fn a_missed_seed_is_regrown_only_as_deep_as_pays() {
        let mut t = TreeRounds::new();
        assert_eq!(t.grow_size(Some((10.0, 0.3))), (64, 8), "no growth timed");
        t.note_grown_before(171.0, STEPS);
        assert_eq!(t.grow_size(None), (64, 8), "no check priced");
        assert_eq!(t.grow_size(Some((10.0, 0.3))), (8, 8), "a short link");
        assert_eq!(t.grow_size(Some((425.0, 0.3))), (64, 8), "a long link");
        assert_eq!(t.grown_before, 1);
        assert_eq!(t.steps_grown_before, STEPS);
    }

    /// A level's acceptance is learned from where walks stopped: at a miss
    /// (the tree had followers there, none the reply's) it falls; at a leaf
    /// it does not, or a shallow tree would teach that the drafter is poor
    /// and the next tree be grown shallower still.
    #[test]
    fn a_walk_that_reaches_a_leaf_is_not_a_miss() {
        let prior = TreeRounds::new().level_acceptance();
        assert!((prior - 0.75).abs() < 1e-9);
        // Kept the whole trunk 10 → 12 → 13, a leaf: no miss, three levels held.
        let mut t = TreeRounds::new();
        for _ in 0..20 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[10, 12, 13, 1]).unwrap();
        }
        assert!(t.level_acceptance() > 0.9, "{}", t.level_acceptance());
        // Kept 11, a leaf too.
        let mut t = TreeRounds::new();
        for _ in 0..20 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[11, 1]).unwrap();
        }
        assert!(t.level_acceptance() > prior, "{}", t.level_acceptance());
        // Kept 10, then 12 was not the reply's though 10 had followers: a miss.
        let mut t = TreeRounds::new();
        for _ in 0..20 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[10, 1]).unwrap();
        }
        assert!(
            (t.level_acceptance() - 0.5).abs() < 0.05,
            "{}",
            t.level_acceptance()
        );
        // Nothing kept with followers to choose from.
        let mut t = TreeRounds::new();
        for _ in 0..20 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[1]).unwrap();
        }
        assert!(t.level_acceptance() < 0.3, "{}", t.level_acceptance());
    }

    #[test]
    fn a_drafter_that_promises_more_than_checks_keep_is_trusted_less() {
        let mut t = TreeRounds::new();
        for _ in 0..5 {
            let c = t.check(7, &grown(), None, 100).unwrap();
            t.settle(&c, &[1]).unwrap();
        }
        assert!(t.calibration() < 0.3, "{}", t.calibration());
        // And a check at a price then sends fewer guesses.
        let priced = t.check(7, &grown(), Some((400.0, 60.0)), 100).unwrap();
        let fresh = TreeRounds::new()
            .check(7, &grown(), Some((400.0, 60.0)), 100)
            .unwrap();
        assert!(priced.guesses.len() < fresh.guesses.len());
    }
}
