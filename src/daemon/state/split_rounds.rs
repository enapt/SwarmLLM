//! How many tokens each round of a split this node drives has been yielding,
//! per model — what the planner divides a split's per-token costs by
//! (FUTURE_WORK #3 (d)).
//!
//! Routing charges a split's per-token costs (every segment's compute, every
//! hop) once per pass of the reply. A split driven from here rarely makes one
//! pass per token: the guess-check paths (`pipeline::dsd`, its continuous
//! stream, the n-gram loop) send several guessed tokens through the segments
//! in ONE round and keep those the model agrees with. On this node's own rigs
//! the splits with a drafter ran in 0.39 of their price (n=25, 2026-10-15) —
//! the gain the price never saw.
//!
//! **What is counted is what happened**, whichever path ran: the rounds a
//! reply's decode sent through the first segment (`RequestTrace::
//! note_decode_round`, at the places a round leaves) and the tokens it
//! produced, recorded once per finished attempt by `PipelineExecutor::execute`.
//! A path whose forwards OVERLAP — the flowing tree (`pipeline::dsd_flow`) and
//! the continuous stream (`pipeline::dsd_stream`) — counts, when it ends, the
//! round trips it waited ([`round_trips_waited`]), not one a forward.
//! The prompt pass is counted as a round too, because it yields the reply's
//! first token and the price counts that token among its passes — so a plain
//! loop reads exactly 1 token per round, never a little more. vLLM reports the
//! same quantity as its "mean acceptance length" (tokens per verification step,
//! the bonus token included); SmartSpec (Liu et al. 2024) and SGLang's adaptive
//! speculation steer by a running average of it.
//!
//! **A ratio of sums, not a mean of ratios**: the price is an expected time, a
//! round costs the same wherever it falls, so the expected rounds of a reply
//! are its tokens over the tokens a round yields across ALL recent tokens — a
//! two-token reply's ratio must not weigh as much as a 300-token one's.
//!
//! What this does not see, each named so it can be checked:
//! - a round that checks several guesses costs more than a plain step where a
//!   segment computes on a processor (SmartSpec's linear batch-time model) —
//!   the γ controller (`dsd_controller::best_gamma_for_check`) holds such a
//!   check short, and stops guessing where it does not pay;
//! - the figure is per model and TAIL (below), every split of that shape
//!   together: a flow whose far card is saturated (a lopsided split) yields
//!   fewer tokens a round than one at an even split, and the price of either
//!   reads both;
//! - the drafter's own time per guess, small beside a hop.
//!
//! Kept across restarts (`recent_by_model`): a restart priced every split at
//! one round a token until four had finished.
//!
//! **Before four have finished, a split with a drafter starts from a prior**
//! (FUTURE_WORK #3 (j), 2026-11-13): one reply's worth of rounds at the γ
//! controller's own opening figures ([`YieldPrior::Drafter`]), blended with
//! whatever splits there are, dropped at [`BEFORE_TRUSTED`]. Priced at one
//! round a token, the 14B's split with a card far away (TH ↔ AU, 34-37 tok/s
//! on code) lost every plan to a whole hand-off (17-18) — and a split that
//! never runs is never measured, so it lost for good. Prior art: a prior
//! counted as pseudo-observations (Bayes-UCBVI's fake data; this crate's
//! `AcceptanceEstimate`, α 0.7 over ten trials); a bandit's untried arm must
//! not be priced as if it had been tried and failed.
//!
//! **And now and then it is priced OPTIMISTICALLY, so it runs** (2026-11-13):
//! the prior is the low end of what drafted splits measured, so where a split
//! would win only at a better yield — a tree flowing to a card far away — it
//! still lost every plan, and still was never measured: three 160-token
//! requests on a fresh node over Thailand ↔ Australia, the prior on, all went
//! to a whole hand-off (15.6-20 tok/s) beside a split measured at 34-37 on code
//! (`real_link.sh` arms `flowall`). [`SplitRounds::prior_for_request`] prices
//! the third request priced from the prior, then the sixth after it, the
//! twelfth, … at most every [`MOST_EXPLORE_AFTER`]th, at
//! [`EXPLORED_TOKENS_A_ROUND`] ([`YieldPrior::Explore`]) — every candidate,
//! every gate and the logged prediction reading the one figure, so the plan is
//! simply the one that would win were the split that good. A split that
//! finishes starts the count again; four that have finished are trusted, and
//! the prior — the optimistic one with it — is gone. What an exploration can
//! cost is bounded twice: it is taken only where the split would then win,
//! and a split that runs is measured, so the next price is its own. Prior art:
//! optimism in the face of uncertainty (UCB — an arm not yet pulled is priced
//! at what it might be, Auer et al. 2002), with forced exploration on a
//! doubling schedule as ParetoBandit starts a new arm and as this crate's
//! `model_worker::Probing` and `SpecBackoff` probe a declined guesser; Envoy's
//! slow start and gRPC's weighted round robin keep an unmeasured endpoint in
//! rotation for the same reason — one never tried is never measured (#178's
//! ratchet, at the scale of a whole split).
//!
//! **Each shape of a split is counted apart: its tail here, or on another
//! computer** ([`SplitTail`], FUTURE_WORK #3 (j5), 2026-11-14). Which machine
//! runs a split's LAST layers decides how its guesses travel: a tail on a card
//! far away draws the flowing tree there (`dsd_tree::FlowDraws::AtTheFarEnd`,
//! the 120 ms line), a tail here keeps the stream (300 ms). On the real
//! Thailand ↔ Australia link the 14B's here-first split (flow) ran 15.8 / 39.0
//! tok/s on prose / code where the same two halves turned round (stream) ran
//! 11.5 / 27.4 — and 7.6 / 17.5 against 6.8 / 13.2 with the far card slowed,
//! here-first faster in four pairs of four (`real_link.sh` `TAG=j7`,
//! 2026-11-13). One figure for both priced them alike, so the hop charges
//! decided — and they lean to the shape that cannot flow: the one split the
//! real link explored ran peer-first. Each tail is now its own arm, measured,
//! trusted and explored on its own (the planner prices each,
//! `parallax::route_shortest_path`); an exploration tries the shape measured
//! least, the flow's first on a tie — every arm pulled before any is trusted,
//! as UCB starts (Auer et al. 2002), since an estimate pooled over two arms
//! hides the better one.
//!
//! [`BEFORE_TRUSTED`]: super::recent_by_model::BEFORE_TRUSTED

use super::recent_by_model::RecentByModel;
use crate::storage::db::Database;

/// What a stored round means. 1: the prompt pass is a round, and a flow or a
/// stream counts the round trips it waited (`round_trips_waited`, v0.3.261).
/// 2: kept per [`SplitTail`] (#3 (j5)) — a figure of 1 is both shapes'
/// together, which priced the slower like the faster.
/// Bump it where a round comes to be counted another way, so a figure
/// counted the old way is never priced.
const ROUNDS_COUNTED_AS: u32 = 2;

/// Where a split's LAST segment runs — the shape its guesses take
/// (`pipeline::dsd`: a tail far away lets the tree flow there, a tail here
/// keeps the stream), so the yield a round is measured and priced per tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SplitTail {
    /// This node runs the model's last layers.
    Here,
    /// Another computer does.
    Far,
}

impl SplitTail {
    /// Both, in the order an exploration breaks a tie: the flow's shape first —
    /// the one the hop charges alone would never try (#3 (j5)).
    pub const BOTH: [SplitTail; 2] = [SplitTail::Far, SplitTail::Here];

    /// The tail of a plan whose last segment runs on `last`.
    pub fn of(last: &crate::types::NodeId, here: &crate::types::NodeId) -> Self {
        if last == here {
            Self::Here
        } else {
            Self::Far
        }
    }

    /// The key its figures are kept under.
    fn key(self, model: &str) -> String {
        match self {
            Self::Here => format!("{model}\u{1f}here"),
            Self::Far => format!("{model}\u{1f}far"),
        }
    }
}

/// The rounds a reply whose forwards overlap waited: its decode time over the
/// least time any of its forwards took to be answered, at most the forwards it
/// sent — `0` until one has been answered (none then waited a round).
///
/// The price charges a split ROUND by round, a round being one crossing of
/// every hop and one pass of each segment (`NodeCandidate::rounds_entered`). A
/// path that waits for each answer before sending again spends one of those a
/// round and counts each as it leaves. One that keeps forwards in flight does
/// not: the 14B's flow over Thailand ↔ Australia sent 252-322 forwards for a
/// 160-token prose reply (2026-10-29), counted one a forward they read 0.5
/// tokens a round — floored to 1, so a split that ran at 7-9 tokens a second
/// was priced at about 2, and, the figure being per model, every other split
/// of the model with it (one such reply among four tree-round replies took
/// their 4 tokens a round to 1.7). Its decode over its least answer is the
/// rounds it waited in the price's own unit: 37-50 for those replies (3.2-4.3
/// tokens a round, 7.7-10.3 on code), and 90-98 (1.6-1.8) where the far card
/// was saturated (36 of 48 layers there, answers queued up to 2.2 s), so a
/// queue is charged, not hidden. The least answer, not a mean: a queued one is
/// the queue's time, which the count must then show as rounds. At most one a
/// forward, so no reply is priced slower than that count priced it.
pub fn round_trips_waited(decode_ms: f64, least_answer_ms: Option<f64>, forwards: u32) -> u32 {
    let Some(least) = least_answer_ms.filter(|ms| ms.is_finite() && *ms > 0.0) else {
        return 0;
    };
    if forwards == 0 || !decode_ms.is_finite() {
        return 0;
    }
    let waited = (decode_ms.max(0.0) / least).ceil();
    // `as` saturates a float past `u32::MAX`; the clamp then holds it.
    (waited as u32).clamp(1, forwards)
}

pub struct SplitRounds {
    /// (tokens, rounds) of each recent split, oldest first.
    recent: RecentByModel<(u32, u32)>,
    /// Each model's exploration schedule ([`Self::prior_for_request`]);
    /// memory only — a restart starts it over.
    exploring: dashmap::DashMap<String, Exploring>,
}

/// What a split of a model is expected to yield a round before this node has
/// measured [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED) of
/// them — a REQUIRED argument of [`SplitRounds::tokens_per_round`], chosen by
/// `pipeline::split_yield_prior` and carried by the plan
/// (`PromptPlan::with_guessing_ahead`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YieldPrior {
    /// A drafter held here would guess for it: one reply of
    /// [`PRIOR_REPLY_TOKENS`] at Leviathan's tokens a round for the γ
    /// controller's opening acceptance and its largest step
    /// (`AcceptanceEstimate::new`'s α 0.7, `GAMMA_STEP` 2: 2.19). The low end
    /// of what drafted splits measured here — 1.64 on two processors
    /// (2026-10-19), 2.2 on the 70B, 3.2-4.3 prose and 7.7-10.3 code where a
    /// tree flowed (2026-10-29) — so it lets a split run, never oversells one.
    Drafter,
    /// [`Self::Drafter`] for the request [`SplitRounds::prior_for_request`]
    /// explores with: one reply at [`EXPLORED_TOKENS_A_ROUND`], so a split
    /// that would win at that yield is planned, run and measured.
    Explore,
    /// Nothing guesses but the n-gram loop's copies, or the plan has no
    /// request behind it: one round a token until measured, as before.
    #[default]
    None,
}

/// The tokens a round an explored split is priced at: the low end of the
/// code replies measured where a tree flowed to a card far away (7.7-10.3 a
/// round trip waited, 14B over Thailand ↔ Australia, 2026-10-29) — above
/// every prose figure (3.2-4.3) and every chain's (1.6-2.2), which is the
/// point: an arm not yet pulled is priced at what it might be. A split that
/// cannot win even there is not tried.
pub const EXPLORED_TOKENS_A_ROUND: f64 = 8.0;

/// Requests priced from the prior before the first is explored: the third.
/// The first two are priced honestly — a newcomer's first replies are not an
/// experiment — and give the reply length (`ReplyLengths`) and a whole
/// hand-off's speed their first samples.
pub const FIRST_EXPLORE_AFTER: u32 = 3;

/// The longest wait between two explorations, in requests priced from the
/// prior: each doubles the wait, so a split that keeps losing is tried ever
/// less often, never never.
pub const MOST_EXPLORE_AFTER: u32 = 64;

/// When a model's split is next priced at [`YieldPrior::Explore`].
#[derive(Clone, Copy, Debug)]
struct Exploring {
    /// Requests priced from the prior since a split of the tail the next
    /// exploration would try finished here, or since the last exploration.
    since: u32,
    /// How many of them the next exploration waits for.
    after: u32,
    /// The last request answered, and its answer: a re-plan of the same
    /// request (`router::another_replan_is_due`) is answered the same and
    /// moves nothing.
    last: Option<(uuid::Uuid, YieldPriors)>,
}

/// The tokens one reply of the prior counts for: the planner's assumed reply
/// (`parallax::ASSUMED_FORWARD_PASSES`), so a first split of a typical
/// length already moves the figure as far as the prior holds it.
pub const PRIOR_REPLY_TOKENS: f64 = 64.0;

impl YieldPrior {
    /// The prior as one pseudo-reply, (tokens, rounds); `None` for no prior.
    fn pseudo_reply(self) -> Option<(f64, f64)> {
        match self {
            Self::Drafter => {
                let alpha = crate::inference::dsd_controller::AcceptanceEstimate::new().alpha();
                let per_round = crate::inference::dsd_controller::expected_tokens_per_round(
                    alpha,
                    crate::inference::dsd_controller::GAMMA_STEP,
                );
                Some((PRIOR_REPLY_TOKENS, PRIOR_REPLY_TOKENS / per_round))
            }
            Self::Explore => Some((
                PRIOR_REPLY_TOKENS,
                PRIOR_REPLY_TOKENS / EXPLORED_TOKENS_A_ROUND,
            )),
            Self::None => None,
        }
    }
}

/// The prior each tail of a split is priced from (#3 (j5)): the same for both,
/// but where [`SplitRounds::prior_for_request`] explores one of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct YieldPriors {
    pub here: YieldPrior,
    pub far: YieldPrior,
}

impl YieldPriors {
    /// `prior` for both tails.
    pub fn both(prior: YieldPrior) -> Self {
        Self {
            here: prior,
            far: prior,
        }
    }

    /// The prior a split with `tail` is priced from.
    pub fn of(self, tail: SplitTail) -> YieldPrior {
        match tail {
            SplitTail::Here => self.here,
            SplitTail::Far => self.far,
        }
    }

    /// Does either tail explore?
    pub fn explores(self) -> bool {
        self.here == YieldPrior::Explore || self.far == YieldPrior::Explore
    }
}

impl From<YieldPrior> for YieldPriors {
    fn from(prior: YieldPrior) -> Self {
        Self::both(prior)
    }
}

/// Where a split's tokens a round came from, logged as `split_yield_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YieldFrom {
    /// The prior alone: no split of the model has finished here.
    Prior,
    /// The prior beside fewer splits than are trusted alone.
    PriorAndSplits,
    /// This node's own recent splits, nothing else.
    Splits,
}

/// What a round of a model's split is expected to yield, and from what.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitYield {
    pub tokens_per_round: f32,
    pub from: YieldFrom,
}

impl Default for SplitRounds {
    fn default() -> Self {
        Self {
            recent: RecentByModel::new("split_rounds", ROUNDS_COUNTED_AS),
            exploring: dashmap::DashMap::new(),
        }
    }
}

impl SplitRounds {
    /// A split of `model` this node drove, its last segment at `tail`,
    /// produced `tokens` in `rounds`, the prompt pass included. Either at zero
    /// tells nothing and is not recorded.
    pub fn record(&self, model: &str, tail: SplitTail, tokens: u32, rounds: u32) {
        if usable(&(tokens, rounds)) {
            // A split of the tail the next exploration would try has measured
            // it: the count to that exploration starts again. A split of the
            // OTHER tail measures nothing about it — counted so, a model whose
            // every request ran a split here-last never tried the far tail
            // (the real link's second arm, 2026-11-13: five streams at
            // ~10 tok/s, the flow that ran 16.7 never priced off the prior).
            // UCB keeps a count per arm for the same reason.
            let measures_the_next = self.tail_to_explore(model) == Some(tail);
            self.recent.push(&tail.key(model), (tokens, rounds));
            if measures_the_next {
                if let Some(mut e) = self.exploring.get_mut(model) {
                    e.since = 0;
                }
            }
        }
    }

    /// The tail an exploration of `model` would try now: the untrusted one
    /// with the fewest splits recorded, [`SplitTail::BOTH`]'s order on a tie;
    /// `None` once both are trusted.
    fn tail_to_explore(&self, model: &str) -> Option<SplitTail> {
        SplitTail::BOTH
            .into_iter()
            .filter(|tail| self.recent.with_trusted(&tail.key(model), |_| ()).is_none())
            .min_by_key(|tail| {
                self.recent
                    .with_any(&tail.key(model), |r| r.len())
                    .unwrap_or(0)
            })
    }

    /// The priors `request` for `model` is priced with: `prior` for both
    /// tails, but [`YieldPrior::Explore`] for ONE where `prior` is the
    /// drafter's, a tail's splits are not yet trusted, and this is the
    /// [`FIRST_EXPLORE_AFTER`]th request so priced since a split of the tail
    /// it would explore finished here — then twice as many after each exploration, up to
    /// [`MOST_EXPLORE_AFTER`]. The tail explored is the untrusted one with the
    /// fewest splits recorded, [`SplitTail::BOTH`]'s order on a tie (#3 (j5)).
    /// Asked once a planning attempt (`pipeline::split_yield_prior`); the same
    /// request asked again gets the same answer.
    pub fn prior_for_request(
        &self,
        model: &str,
        request: uuid::Uuid,
        prior: YieldPrior,
    ) -> YieldPriors {
        let untried = self.tail_to_explore(model);
        let (YieldPrior::Drafter, Some(untried)) = (prior, untried) else {
            return YieldPriors::both(prior);
        };
        let mut e = self
            .exploring
            .entry(model.to_string())
            .or_insert(Exploring {
                since: 0,
                after: FIRST_EXPLORE_AFTER,
                last: None,
            });
        if let Some((asked, answer)) = e.last {
            if asked == request {
                return answer;
            }
        }
        e.since += 1;
        let mut answer = YieldPriors::both(YieldPrior::Drafter);
        if e.since >= e.after {
            e.since = 0;
            e.after = e.after.saturating_mul(2).min(MOST_EXPLORE_AFTER);
            match untried {
                SplitTail::Here => answer.here = YieldPrior::Explore,
                SplitTail::Far => answer.far = YieldPrior::Explore,
            }
        }
        e.last = Some((request, answer));
        answer
    }

    /// The tokens a round of `model`'s recent splits with `tail` yielded, once
    /// [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED) have
    /// finished — measured, never derived. Before then the `prior`'s
    /// pseudo-reply joins whatever splits there are in the same ratio of sums,
    /// or `None` where there is no prior. Never below one: a round that
    /// yielded nothing (a stream's chunk sent on guesses that were then
    /// refused) overlapped another, and must not price a split above the one
    /// round a token it was priced at before.
    pub fn tokens_per_round(
        &self,
        model: &str,
        tail: SplitTail,
        prior: YieldPrior,
    ) -> Option<SplitYield> {
        let model = &tail.key(model);
        let sums = |recent: &std::collections::VecDeque<(u32, u32)>| {
            recent
                .iter()
                .fold((0u64, 0u64), |(t, r), &(tokens, rounds)| {
                    (t + u64::from(tokens), r + u64::from(rounds))
                })
        };
        let at_least_one = |tokens: f64, rounds: f64| (tokens / rounds.max(1.0)).max(1.0) as f32;
        if let Some((tokens, rounds)) = self.recent.with_trusted(model, sums) {
            return Some(SplitYield {
                tokens_per_round: at_least_one(tokens as f64, rounds as f64),
                from: YieldFrom::Splits,
            });
        }
        let (prior_tokens, prior_rounds) = prior.pseudo_reply()?;
        let (tokens, rounds) = self.recent.with_any(model, sums).unwrap_or((0, 0));
        Some(SplitYield {
            tokens_per_round: at_least_one(
                prior_tokens + tokens as f64,
                prior_rounds + rounds as f64,
            ),
            from: if rounds == 0 {
                YieldFrom::Prior
            } else {
                YieldFrom::PriorAndSplits
            },
        })
    }

    /// The splits an earlier run stored (`recent_by_model`).
    pub(super) fn restore(&self, db: &Database) {
        self.recent.restore(db, usable);
    }

    /// Store the recent splits if one finished since the last write.
    pub(super) fn persist(&self, db: &Database) {
        self.recent.persist(db);
    }
}

/// A split with either count at zero tells nothing.
fn usable(&(tokens, rounds): &(u32, u32)) -> bool {
    tokens > 0 && rounds > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::state::recent_by_model::{BEFORE_TRUSTED, RECENT};
    use SplitTail::{Far, Here};

    /// The figure with no prior: measured, or `None`.
    fn measured(rounds: &SplitRounds, model: &str) -> Option<f32> {
        rounds
            .tokens_per_round(model, Here, YieldPrior::None)
            .map(|y| y.tokens_per_round)
    }

    /// The figure is the recent splits' tokens over their rounds — a long
    /// reply weighs as its tokens do — once enough have finished.
    #[test]
    fn a_models_splits_yield_their_tokens_over_their_rounds() {
        let rounds = SplitRounds::default();
        for _ in 0..BEFORE_TRUSTED - 1 {
            rounds.record("m", Here, 300, 100);
        }
        assert_eq!(measured(&rounds, "m"), None, "too few to trust");
        // A two-token reply in two rounds (the prompt pass and one step).
        rounds.record("m", Here, 2, 2);
        let each = measured(&rounds, "m").expect("trusted");
        assert!((each - 902.0 / 302.0).abs() < 1e-4, "{each}");
        assert_eq!(measured(&rounds, "other"), None);
        // Nothing to learn from an empty reply or an uncounted one.
        rounds.record("m", Here, 0, 5);
        rounds.record("m", Here, 5, 0);
        assert!((measured(&rounds, "m").unwrap() - each).abs() < 1e-6);
    }

    /// A plain loop reads exactly one token a round, and a stream whose
    /// chunks were refused never reads below it.
    #[test]
    fn a_round_is_never_priced_at_less_than_a_token() {
        let rounds = SplitRounds::default();
        for _ in 0..BEFORE_TRUSTED {
            rounds.record("plain", Here, 40, 40);
            rounds.record("wasted", Here, 10, 14);
        }
        assert_eq!(measured(&rounds, "plain"), Some(1.0));
        assert_eq!(measured(&rounds, "wasted"), Some(1.0));
    }

    /// A reply whose forwards overlapped is counted the round trips it waited:
    /// the 14B's flows on the real TH ↔ AU link (2026-10-29), at an even split
    /// and with the far card saturated, against their own lines.
    #[test]
    fn overlapping_forwards_are_counted_the_round_trips_they_waited() {
        // flowed 19.8 s, least answer 445 ms, 292 forwards, 160 tokens.
        let even = round_trips_waited(19_800.0, Some(445.0), 292);
        assert_eq!(even, 45);
        assert!((160.0 / even as f64 - 3.56).abs() < 0.01);
        // Saturated (12 of 48 here): 53.7 s, least 548 ms, 409 forwards.
        let saturated = round_trips_waited(53_700.0, Some(548.0), 409);
        assert_eq!(saturated, 98);
        assert!(160.0 / (saturated as f64) < 160.0 / (even as f64) / 2.0);
        // Never more than a round a forward: one sent at a time, each waited on
        // behind this node's own slow work, is one round each.
        assert_eq!(round_trips_waited(10_000.0, Some(100.0), 12), 12);
        // Nothing answered, nothing sent, nothing measurable: no round.
        assert_eq!(round_trips_waited(5_000.0, None, 10), 0);
        assert_eq!(round_trips_waited(5_000.0, Some(400.0), 0), 0);
        assert_eq!(round_trips_waited(5_000.0, Some(0.0), 10), 0);
        assert_eq!(round_trips_waited(f64::NAN, Some(400.0), 10), 0);
        // A decode shorter than a round trip still waited one.
        assert_eq!(round_trips_waited(10.0, Some(400.0), 3), 1);
    }

    /// **A split with a drafter is priced from a prior until its own splits
    /// are trusted** (#3 (j)): one reply of 64 tokens at Leviathan's 2.19 a
    /// round (α 0.7, γ 2) before any has run; that reply beside the splits
    /// that have, in one ratio of sums; then the splits alone, exactly as
    /// without a prior. Without a drafter there is no prior.
    #[test]
    fn a_split_with_a_drafter_starts_from_the_prior_until_its_own_are_trusted() {
        let rounds = SplitRounds::default();
        let cold = rounds
            .tokens_per_round("m", Here, YieldPrior::Drafter)
            .expect("a prior");
        assert!((cold.tokens_per_round - 2.19).abs() < 0.005, "{cold:?}");
        assert_eq!(cold.from, YieldFrom::Prior);
        assert_eq!(
            measured(&rounds, "m"),
            None,
            "no drafter: one round a token"
        );

        // A first split that kept 4 a round pulls it up, one that kept ~1 down.
        let prior_rounds = PRIOR_REPLY_TOKENS / f64::from(cold.tokens_per_round);
        rounds.record("good", Here, 160, 40);
        let good = rounds
            .tokens_per_round("good", Here, YieldPrior::Drafter)
            .unwrap();
        assert_eq!(good.from, YieldFrom::PriorAndSplits);
        let expect = (PRIOR_REPLY_TOKENS + 160.0) / (prior_rounds + 40.0);
        assert!(
            (f64::from(good.tokens_per_round) - expect).abs() < 0.01,
            "{good:?}"
        );
        assert!(good.tokens_per_round > cold.tokens_per_round);
        rounds.record("poor", Here, 120, 118);
        let poor = rounds
            .tokens_per_round("poor", Here, YieldPrior::Drafter)
            .unwrap();
        assert!(poor.tokens_per_round < cold.tokens_per_round, "{poor:?}");
        assert!(poor.tokens_per_round >= 1.0);
        assert_eq!(
            measured(&rounds, "good"),
            None,
            "still too few to trust alone"
        );

        // Trusted: the splits alone, the prior gone.
        for _ in 1..BEFORE_TRUSTED {
            rounds.record("good", Here, 160, 40);
        }
        let trusted = rounds
            .tokens_per_round("good", Here, YieldPrior::Drafter)
            .unwrap();
        assert_eq!(trusted.from, YieldFrom::Splits);
        assert_eq!(trusted.tokens_per_round, 4.0);
        assert_eq!(measured(&rounds, "good"), Some(4.0));
    }

    /// **An untried split is priced optimistically now and then** (#3 (j)):
    /// the third request priced from the drafter's prior explores, then the
    /// sixth after it, doubling to the cap; a re-plan of one request keeps its
    /// answer; a split of the tail it would explore starts the count again, a
    /// split of the other tail does not; four finished of each end it, and no
    /// other prior is ever explored.
    #[test]
    fn an_untried_split_is_explored_on_a_doubling_schedule() {
        let rounds = SplitRounds::default();
        let ask = |model: &str| {
            rounds.prior_for_request(model, uuid::Uuid::new_v4(), YieldPrior::Drafter)
        };
        let answers: Vec<_> = (0..9).map(|_| ask("m")).collect();
        use YieldPrior::{Drafter as D, Explore as X};
        let dd = YieldPriors::both(D);
        // Nothing measured: the flow's tail first (#3 (j5)), then the stream's.
        let xf = YieldPriors { here: D, far: X };
        let xh = YieldPriors { here: X, far: D };
        assert_eq!(
            answers,
            [dd, dd, xf, dd, dd, dd, dd, dd, xf],
            "3rd, then 6 later"
        );

        // Priced at the optimistic yield, beside nothing measured.
        let explored = rounds.tokens_per_round("m", Here, X).unwrap();
        assert!((f64::from(explored.tokens_per_round) - EXPLORED_TOKENS_A_ROUND).abs() < 1e-4);

        // A re-plan of the same request is answered the same, and moves nothing.
        let fresh = SplitRounds::default();
        let first = uuid::Uuid::new_v4();
        let second = uuid::Uuid::new_v4();
        let third = uuid::Uuid::new_v4();
        assert_eq!(fresh.prior_for_request("m", first, D), dd);
        assert_eq!(fresh.prior_for_request("m", second, D), dd);
        assert_eq!(fresh.prior_for_request("m", second, D), dd, "a re-plan");
        assert_eq!(fresh.prior_for_request("m", third, D), xf);
        assert_eq!(
            fresh.prior_for_request("m", third, D),
            xf,
            "the re-plan explores too"
        );

        // A split of the OTHER tail does not start the count again: the
        // real link's second arm ran a stream (tail here) on every request and
        // the far tail was never tried.
        let ran = SplitRounds::default();
        ran.prior_for_request("m", uuid::Uuid::new_v4(), D);
        ran.prior_for_request("m", uuid::Uuid::new_v4(), D);
        ran.record("m", Here, 160, 40);
        // The tail with fewer splits is the one explored, on time.
        assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), xf);
        ran.record("m", Here, 160, 40);
        ran.record("m", Here, 160, 40);
        // (The wait has doubled to six, and splits here-last never move it.)
        for _ in 0..5 {
            assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), dd);
            ran.record("m", Here, 160, 40);
        }
        assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), xf);
        // A split of the tail it would explore does start it again.
        ran.record("m", Far, 160, 40);
        for _ in 0..11 {
            assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), dd);
        }
        assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), xf);
        // The tail with fewer splits is the one explored, Here too.
        let other = SplitRounds::default();
        other.record("m", Far, 160, 40);
        assert_eq!(other.prior_for_request("m", uuid::Uuid::new_v4(), D), dd);
        assert_eq!(other.prior_for_request("m", uuid::Uuid::new_v4(), D), dd);
        assert_eq!(other.prior_for_request("m", uuid::Uuid::new_v4(), D), xh);

        // Both trusted: no prior, nothing explored.
        for _ in 1..BEFORE_TRUSTED {
            ran.record("m", Here, 160, 40);
            ran.record("m", Far, 160, 40);
        }
        for _ in 0..2 * MOST_EXPLORE_AFTER {
            assert_eq!(ran.prior_for_request("m", uuid::Uuid::new_v4(), D), dd);
        }
        // No drafter, no prior to explore.
        for _ in 0..2 * MOST_EXPLORE_AFTER {
            assert_eq!(
                rounds.prior_for_request("n", uuid::Uuid::new_v4(), YieldPrior::None),
                YieldPriors::both(YieldPrior::None)
            );
        }

        // The wait doubles to the cap and stays there.
        let long = SplitRounds::default();
        let mut gaps = Vec::new();
        let mut since = 0;
        for _ in 0..400 {
            since += 1;
            if long
                .prior_for_request("m", uuid::Uuid::new_v4(), D)
                .explores()
            {
                gaps.push(since);
                since = 0;
            }
        }
        assert_eq!(gaps[..6], [3, 6, 12, 24, 48, 64]);
        assert!(gaps[6..].iter().all(|&g| g == MOST_EXPLORE_AFTER));
    }

    /// **Each tail is its own arm** (#3 (j5)): a split whose last layers run
    /// here and one whose last layers run far away are measured, trusted and
    /// explored apart — on the real link the flow (tail far) ran 1.1-1.4x the
    /// stream (tail here) and one figure for both priced them alike.
    #[test]
    fn each_tail_of_a_split_is_measured_and_explored_apart() {
        let rounds = SplitRounds::default();
        for _ in 0..BEFORE_TRUSTED {
            rounds.record("m", Here, 160, 40);
        }
        let here = rounds
            .tokens_per_round("m", Here, YieldPrior::Drafter)
            .unwrap();
        assert_eq!((here.tokens_per_round, here.from), (4.0, YieldFrom::Splits));
        let far = rounds
            .tokens_per_round("m", Far, YieldPrior::Drafter)
            .unwrap();
        assert_eq!(far.from, YieldFrom::Prior, "the other tail has not run");
        assert!((far.tokens_per_round - 2.19).abs() < 0.005, "{far:?}");
        // Only the untrusted tail is ever explored.
        let mut explored = 0;
        for _ in 0..200 {
            let p = rounds.prior_for_request("m", uuid::Uuid::new_v4(), YieldPrior::Drafter);
            assert_eq!(p.here, YieldPrior::Drafter, "{p:?}");
            explored += usize::from(p.far == YieldPrior::Explore);
        }
        assert!(explored >= 3, "{explored}");
        // A plan's tail is its last segment's machine.
        let (a, b) = (crate::types::NodeId([1; 32]), crate::types::NodeId([2; 32]));
        assert_eq!(SplitTail::of(&a, &a), Here);
        assert_eq!(SplitTail::of(&b, &a), Far);
    }

    #[test]
    fn only_the_most_recent_splits_count() {
        let rounds = SplitRounds::default();
        for _ in 0..RECENT {
            rounds.record("m", Here, 400, 100);
        }
        for _ in 0..RECENT {
            rounds.record("m", Here, 60, 60);
        }
        assert_eq!(measured(&rounds, "m"), Some(1.0));
    }
}
