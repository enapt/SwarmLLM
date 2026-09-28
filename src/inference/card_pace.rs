//! How many generations a worker runs on its graphics card at once, decided by
//! what the card actually DOES rather than by memory arithmetic alone.
//!
//! Every other admission gate on the card is arithmetic: the slot count and the
//! KV budget (`split::kv_budget`). The budget's own doc says why that is not
//! enough on this platform — "WSL2 hands out shared memory rather than failing,
//! so the accounting is the only guard there is". On 2026-09-28 the live node
//! (an 8 GB laptop card that also drives the Windows desktop) admitted up to six
//! simultaneous chats of an 8B, all within its budget, while each admission took
//! 2-60 s of device calls where it had always taken under 0.1 s. Because the
//! worker is ONE loop, every running chat froze for as long, and the node kept
//! taking more for an hour — until the whole PC hung (gotcha #754,
//! `docs/FUTURE_WORK.md` #146).
//!
//! The cause of the stalls is not known, and nothing here needs it. This is the
//! signal every cause shares: **one device-bound step taking longer than the
//! operating system itself allows a graphics engine to ignore it.**
//!
//! It is congestion control applied to concurrency — the pattern of Netflix's
//! `concurrency-limits` and Envoy's adaptive-concurrency filter, reduced to
//! AIMD on a single, unambiguous signal:
//!
//! - **Stall** = one step of at least [`STALL`] on a model running entirely on
//!   the card. 2 s is Windows' `TdrDelay` default — "the number of seconds that
//!   the GPU can delay the preempt request from the GPU scheduler"
//!   (learn.microsoft.com, TDR registry keys). A healthy admission here took
//!   under 0.1 s and a decode tick tens of milliseconds.
//! - **Decrease**: the ceiling becomes half of what was running, at least one,
//!   never higher than it already was.
//! - **Increase**: one more after each [`HOLD`] without a stall, up to the
//!   configured capacity. 60 s is Windows' `TdrLimitTime`, the window in which
//!   repeated engine resets escalate to a crash.
//! - **Never below one**: a lone generation is always admitted, so the owner is
//!   never locked out of their own card.
//!
//! **Processor steps never count.** A CPU tick can legitimately take seconds
//! (gotcha #191), so the caller reports only steps whose model runs entirely on
//! the card; a card/processor split is excluded for the same reason. The first
//! [`WARM_STEPS`] card steps are excluded too: a fresh worker's first forwards
//! load kernels and library handles and are slow for reasons that pass.
//!
//! `SWARMLLM_CARD_PACE=0` turns it off, for an A/B inside one binary.

use std::time::{Duration, Instant};

/// One device-bound step at least this long is a stall. Windows' `TdrDelay`
/// default; see the module doc.
pub(crate) const STALL: Duration = Duration::from_secs(2);

/// How long the card must go without a stall before the ceiling rises by one.
/// Windows' `TdrLimitTime` default; see the module doc.
pub(crate) const HOLD: Duration = Duration::from_secs(60);

/// Card steps at the start of a worker's life that cannot count as a stall.
pub(crate) const WARM_STEPS: u32 = 16;

/// What a stall did to the ceiling, for the caller to log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stalled {
    pub took: Duration,
    pub running: usize,
    pub ceiling_before: usize,
    pub ceiling_after: usize,
}

impl Stalled {
    /// The one line a stall writes, whichever step it was.
    pub(crate) fn log(&self, step: &'static str) {
        tracing::warn!(
            step,
            took_ms = self.took.as_millis() as u64,
            running = self.running,
            ceiling_before = self.ceiling_before,
            ceiling_after = self.ceiling_after,
            "DIAG: card pace — the graphics card stalled; running fewer generations at once \
             until it keeps up"
        );
    }
}

/// The ceiling on simultaneous generations for one worker's card.
#[derive(Debug)]
pub(crate) struct CardPace {
    enabled: bool,
    capacity: usize,
    ceiling: usize,
    card_steps: u32,
    last_stall: Option<Instant>,
    last_raise: Option<Instant>,
}

impl CardPace {
    /// `capacity` is the worker's configured slot count; the ceiling starts
    /// there and never exceeds it.
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_switch(
            capacity,
            enabled_from(std::env::var("SWARMLLM_CARD_PACE").ok().as_deref()),
        )
    }

    pub(crate) fn with_switch(capacity: usize, enabled: bool) -> Self {
        let capacity = capacity.max(1);
        Self {
            enabled,
            capacity,
            ceiling: capacity,
            card_steps: 0,
            last_stall: None,
            last_raise: None,
        }
    }

    /// The ceiling as it stands, without letting it recover. Diagnostic.
    pub(crate) fn ceiling(&self) -> usize {
        self.ceiling
    }

    /// A device-bound step finished. `on_card` is whether its model runs
    /// entirely on the graphics card; `running` is how many generations the
    /// worker held while it ran. Returns what a stall did, if this was one.
    pub(crate) fn observe(
        &mut self,
        took: Duration,
        running: usize,
        on_card: bool,
        now: Instant,
    ) -> Option<Stalled> {
        if !self.enabled || !on_card {
            return None;
        }
        self.card_steps = self.card_steps.saturating_add(1);
        if self.card_steps <= WARM_STEPS || took < STALL {
            return None;
        }
        let ceiling_before = self.ceiling;
        self.ceiling = self.ceiling.min((running / 2).max(1));
        self.last_stall = Some(now);
        Some(Stalled {
            took,
            running,
            ceiling_before,
            ceiling_after: self.ceiling,
        })
    }

    /// May the worker, already running `running` generations, start another?
    /// Recovers the ceiling first: one step per [`HOLD`] since the later of the
    /// last stall and the last rise.
    ///
    /// **Only a lowered ceiling refuses anything.** At capacity the pace is not
    /// limiting and says yes whatever is running: what a full table does with
    /// one more request is the table's decision, as it was before this existed,
    /// and a refusal saying the card stalled would be untrue.
    pub(crate) fn admits(&mut self, running: usize, now: Instant) -> bool {
        if !self.enabled {
            return true;
        }
        self.recover(now);
        self.ceiling >= self.capacity || running < self.ceiling
    }

    fn recover(&mut self, now: Instant) {
        let Some(stall) = self.last_stall else {
            return;
        };
        while self.ceiling < self.capacity {
            let since = self.last_raise.map_or(stall, |raise| raise.max(stall));
            if now.saturating_duration_since(since) < HOLD {
                return;
            }
            self.ceiling += 1;
            self.last_raise = Some(since + HOLD);
        }
    }
}

/// `SWARMLLM_CARD_PACE=0` (or `false`/`off`) disables the guard; anything else,
/// or nothing, leaves it on.
fn enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off")
    )
}

/// What a caller refused by the guard is told. Carried as
/// `SwarmError::LocalMemoryUnavailable` — the busy 503 the router re-plans with
/// this node barred from the whole model — because the next step is the same:
/// another machine now, or this one again in a moment.
///
/// ⚠ Must never contain a `worker_ipc::worker_error_is_fatal` pattern ("out of
/// memory", "cuda error", …): a refusal that read as fatal would have the pool
/// kill a healthy worker. Pinned by a test.
pub(crate) fn refusal_message(running: usize, ceiling: usize) -> String {
    let conversations = if ceiling == 1 {
        "conversation"
    } else {
        "conversations"
    };
    format!(
        "This node's graphics card has been stalling for seconds at a time, so for now it runs at \
         most {ceiling} {conversations} at once and is already running {running}. Try again in a \
         moment; it takes more again once the card keeps up."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: fn(u64) -> Duration = Duration::from_millis;

    /// A pace past its warm-up, so the test is about the rule and not the
    /// first steps of a worker's life.
    fn warm(capacity: usize, now: Instant) -> CardPace {
        let mut pace = CardPace::with_switch(capacity, true);
        for _ in 0..WARM_STEPS {
            assert_eq!(pace.observe(MS(10), 1, true, now), None);
        }
        pace
    }

    #[test]
    fn a_stall_halves_what_was_running_and_refuses_past_it() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        assert!(pace.admits(5, t0));
        let s = pace
            .observe(Duration::from_secs(14), 6, true, t0)
            .expect("a 14 s step is a stall");
        assert_eq!((s.ceiling_before, s.ceiling_after), (8, 3));
        assert!(pace.admits(2, t0));
        assert!(
            !pace.admits(3, t0),
            "at the ceiling a new generation is refused"
        );
    }

    #[test]
    fn a_step_just_under_the_threshold_is_not_a_stall() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        assert_eq!(pace.observe(STALL.saturating_sub(MS(1)), 6, true, t0), None);
        assert_eq!(pace.ceiling(), 8);
        assert!(pace.observe(STALL, 6, true, t0).is_some());
    }

    #[test]
    fn the_ceiling_never_falls_below_one_so_a_lone_chat_always_runs() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        pace.observe(Duration::from_secs(9), 1, true, t0);
        assert_eq!(pace.ceiling(), 1);
        pace.observe(Duration::from_secs(9), 0, true, t0);
        assert_eq!(pace.ceiling(), 1);
        assert!(pace.admits(0, t0), "an idle worker always takes one");
        assert!(!pace.admits(1, t0));
    }

    #[test]
    fn a_stall_never_raises_the_ceiling() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        pace.observe(Duration::from_secs(5), 2, true, t0);
        assert_eq!(pace.ceiling(), 1);
        // Six running (admitted before the trip) and stalling again: half of
        // six is three, above the ceiling already set — it must stay at one.
        let s = pace.observe(Duration::from_secs(5), 6, true, t0).unwrap();
        assert_eq!((s.ceiling_before, s.ceiling_after), (1, 1));
    }

    #[test]
    fn the_ceiling_comes_back_one_per_quiet_hold_and_no_further_than_capacity() {
        let t0 = Instant::now();
        let mut pace = warm(3, t0);
        pace.observe(Duration::from_secs(5), 2, true, t0);
        assert_eq!(pace.ceiling(), 1);
        assert!(
            !pace.admits(1, t0 + HOLD.saturating_sub(MS(1))),
            "held for a full HOLD"
        );
        assert!(pace.admits(1, t0 + HOLD), "one more after one quiet HOLD");
        assert!(!pace.admits(2, t0 + HOLD + MS(1)));
        assert!(pace.admits(2, t0 + HOLD * 2));
        // Long idle: back to capacity, never past it.
        assert!(pace.admits(2, t0 + HOLD * 50));
        assert_eq!(pace.ceiling(), 3);
    }

    /// A full table with a healthy card is the TABLE's business: before this
    /// existed the extra request fell through to the sequential path, and a
    /// refusal claiming the card stalled would be untrue.
    #[test]
    fn a_card_that_never_stalled_refuses_nothing_however_full() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        assert!(pace.admits(8, t0));
        assert!(pace.admits(20, t0));
        // And once recovered to capacity after a stall, the same.
        pace.observe(Duration::from_secs(3), 4, true, t0);
        assert!(!pace.admits(2, t0));
        assert!(pace.admits(8, t0 + HOLD * 10));
        assert_eq!(pace.ceiling(), 8);
    }

    #[test]
    fn a_fresh_stall_restarts_the_hold() {
        let t0 = Instant::now();
        let mut pace = warm(4, t0);
        pace.observe(Duration::from_secs(5), 2, true, t0);
        assert!(pace.admits(1, t0 + HOLD)); // back to 2
        pace.observe(Duration::from_secs(5), 2, true, t0 + HOLD + MS(10));
        assert_eq!(pace.ceiling(), 1);
        assert!(
            !pace.admits(1, t0 + HOLD * 2),
            "the hold counts from the NEW stall"
        );
        assert!(pace.admits(1, t0 + HOLD * 2 + MS(10)));
    }

    #[test]
    fn processor_steps_never_count_however_long() {
        let t0 = Instant::now();
        let mut pace = warm(8, t0);
        assert_eq!(pace.observe(Duration::from_secs(59), 6, false, t0), None);
        assert_eq!(pace.ceiling(), 8);
    }

    #[test]
    fn the_first_card_steps_of_a_worker_cannot_trip_it() {
        let t0 = Instant::now();
        let mut pace = CardPace::with_switch(8, true);
        for _ in 0..WARM_STEPS {
            assert_eq!(pace.observe(Duration::from_secs(8), 4, true, t0), None);
        }
        assert!(pace.observe(Duration::from_secs(8), 4, true, t0).is_some());
        // Processor steps do not use up the warm-up.
        let mut pace = CardPace::with_switch(8, true);
        for _ in 0..100 {
            pace.observe(MS(10), 1, false, t0);
        }
        assert_eq!(pace.observe(Duration::from_secs(8), 4, true, t0), None);
    }

    #[test]
    fn switched_off_it_admits_everything_and_never_trips() {
        let t0 = Instant::now();
        let mut pace = CardPace::with_switch(2, false);
        for _ in 0..(WARM_STEPS * 2) {
            assert_eq!(pace.observe(Duration::from_secs(30), 2, true, t0), None);
        }
        assert!(pace.admits(50, t0));
        assert!(!enabled_from(Some("0")));
        assert!(!enabled_from(Some(" OFF ")));
        assert!(!enabled_from(Some("false")));
        assert!(enabled_from(None));
        assert!(enabled_from(Some("1")));
    }

    /// A refusal the pool read as fatal would kill a healthy worker — the
    /// opposite of backing off.
    #[test]
    fn the_refusal_is_never_mistaken_for_a_broken_worker() {
        for (running, ceiling) in [(1, 1), (3, 3), (6, 1)] {
            let msg = refusal_message(running, ceiling);
            assert!(
                !crate::inference::worker_ipc::worker_error_is_fatal(&msg),
                "{msg}"
            );
            assert!(msg.contains("Try again in a moment"));
        }
    }
}
