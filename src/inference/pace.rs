//! A hot computer's own work, paced (#440 step 4).
//!
//! While the daemon holds back for heat (`daemon::host_heat`), the work for
//! other computers stops, but the owner's own chats are never refused. They
//! are SLOWED instead: each step a worker runs — a forward of the model, a
//! prompt chunk, a batch, a block drafter's draft — owes a pause as long as
//! itself (`HOT_PAUSE_PERMILLE`), so the card and the processor work half the
//! time, and the reply still comes, at half speed.
//!
//! **The pause is slept in pieces of at least [`PAUSE_PIECE`] (250 ms)**, not
//! after every step. Measured on a laptop card (RTX 3070 Laptop, Llama-3.1-8B
//! whole, 2026-11-11): paused after every 16 ms decode step, the card was busy
//! 46% of the time but drew 82.5 W against 94.7 W cool, because it filled the
//! gaps by raising its clock (1150 → 1610 MHz); in 250 ms pieces 67.5 W, in
//! 1000 ms pieces 66.5 W, at the same half speed. A card's power limit is an
//! AVERAGE (NVIDIA's TensorRT guide: gaps between inferences lower average
//! power, so the clock is throttled less), so only gaps long enough for the
//! card to idle lower it.
//!
//! Prior art: BOINC's throttle runs and suspends work in seconds (5 s of every
//! 10 at 50%); a frame-rate cap paces a card with gaps between frames (AMD's
//! Frame Rate Target Control). A pause owed in PROPORTION to the step keeps a
//! step's share of the time fixed whatever its length; slept in pieces, most
//! timed steps carry none, and one that does reads to the speculation
//! controllers as one slow check, which they hold back (`CheckCost::record`).
//! Fewer processor threads was tried in 2026-08 and changed nothing
//! (`FUTURE_WORK_ARCHIVE.md` § "Thermal throttling had no measurable effect"),
//! so the lever here is time, not width.
//!
//! The pause is taken once per OUTERMOST step on a thread ([`Step`]): a batch
//! of one runs through `forward`, which does not pause again. On a card the
//! step synchronizes first, so the time measured is the card's work, not the
//! launch. A pause short of a piece is carried into the next step's, and an
//! overshoot is credited back, so the duty cycle holds on average however
//! coarse the timer is; time the thread rested between steps (between two
//! replies) pays what is owed, so a reply never pays for the one before it.
//! The time paused is counted
//! ([`paused_total`]) so a timer that judges a step's length — `CardPace`'s
//! stall — never reads a pause as a stall.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The pause a step is followed by, in thousandths of the step's own time.
/// `0`: full speed. Set by the daemon (`DaemonMsg::Pace`, or
/// [`WORKER_PACE_ENV`] at spawn).
static PAUSE_PERMILLE: AtomicU32 = AtomicU32::new(0);

/// Every pause this process has taken, in nanoseconds.
static PAUSED_NANOS: AtomicU64 = AtomicU64::new(0);

/// The worker's pace at spawn, in thousandths (`DaemonMsg::Pace` for a change
/// afterwards). Absent: full speed.
pub(crate) const WORKER_PACE_ENV: &str = "SWARMLLM_WORKER_PACE_PERMILLE";

/// While hot: pause as long as the step took — half the duty cycle (the
/// design's starting value, #440).
pub(crate) const HOT_PAUSE_PERMILLE: u32 = 1000;

/// The most a pace may ask: a pause four times the step.
const MOST_PERMILLE: u32 = 4000;

/// The longest single pause. A step longer than this (a long prompt read on
/// the processor in one forward) is still paced, only not fully: the reply's
/// first word is not held back past it.
const MOST_PAUSE: Duration = Duration::from_secs(5);

/// A pause is slept in slices this long, so a cancel lands within one.
const PAUSE_SLICE: Duration = Duration::from_millis(50);

/// A pause shorter than this is carried into the next step's: the system's
/// timer may sleep a millisecond or more whatever it is asked.
const SHORTEST_SLEEP: Duration = Duration::from_millis(2);

/// The pause owed is carried until it reaches this much, then slept in one
/// piece (#440 step 4 (a)). Paused after every 16 ms step, a card filled the
/// gaps by boosting its clock and drew 87% of its cool power at half the work;
/// in 250 ms pieces, 71% (1000 ms: 70%). Prior art: NVIDIA's TensorRT
/// measurement guide (gaps between inferences lower average power, so the
/// clock is throttled less — the power limit is an average); BOINC's throttle
/// runs and suspends work in seconds (5 s of 10 at 50%), never per step.
const PAUSE_PIECE: Duration = Duration::from_millis(250);

/// A/B knob: the piece in milliseconds (`2` = after every step, as before).
pub(crate) const PAUSE_FROM_ENV: &str = "SWARMLLM_PACE_PAUSE_FROM_MS";

/// The piece in nanoseconds once read ([`pause_from`]); 0 = not read yet.
static PIECE_NANOS: AtomicU64 = AtomicU64::new(0);

/// The shortest pause actually slept: [`PAUSE_PIECE`], or [`PAUSE_FROM_ENV`].
fn pause_from() -> Duration {
    let read = PIECE_NANOS.load(Ordering::Relaxed);
    if read != 0 {
        return Duration::from_nanos(read);
    }
    let piece = std::env::var(PAUSE_FROM_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .map_or(PAUSE_PIECE, |d| d.clamp(SHORTEST_SLEEP, MOST_PAUSE));
    PIECE_NANOS.store(
        u64::try_from(piece.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    piece
}

thread_local! {
    /// How many [`Step`]s are open on this thread; only the outermost pauses.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Pause owed (positive) or overslept (negative), in nanoseconds.
    static DEBT_NANOS: Cell<i64> = const { Cell::new(0) };
    /// When the last paced step on this thread ended, its pause included.
    static LAST_END: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Set the pace; answers the one it replaces. Logged on a change.
pub(crate) fn set_pause_permille(permille: u32) -> u32 {
    let permille = permille.min(MOST_PERMILLE);
    let before = PAUSE_PERMILLE.swap(permille, Ordering::Relaxed);
    if before != permille {
        if permille == 0 {
            tracing::info!(
                "model-worker: the computer has cooled down — its steps run at full speed again"
            );
        } else {
            tracing::info!(
                pause_per_step_pct = permille / 10,
                "model-worker: the computer is running hot — each step is followed by a pause \
                 until it cools"
            );
        }
    }
    before
}

/// The pace as it stands, in thousandths of a step.
pub(crate) fn pause_permille() -> u32 {
    PAUSE_PERMILLE.load(Ordering::Relaxed)
}

/// May a worker do work no reply is waiting for — a guess head guessing on
/// while its check is abroad (`model_worker`'s `guess_ahead`, #425 (f))? Not
/// while the computer is paced: that work only runs where the device would
/// otherwise rest, and resting is what a hot computer needs (#440 step 4).
pub(crate) fn work_ahead_may_run() -> bool {
    pause_permille() == 0
}

/// The pace a worker's environment names at spawn ([`WORKER_PACE_ENV`]).
pub(crate) fn pace_at_spawn(value: Option<&std::ffi::OsStr>) -> u32 {
    value
        .and_then(|v| v.to_str())
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0)
}

/// Every pause this process has taken. A timer that judges a step's length
/// subtracts what was paused while it ran.
pub(crate) fn paused_total() -> Duration {
    Duration::from_nanos(PAUSED_NANOS.load(Ordering::Relaxed))
}

/// A stopwatch that leaves out this process's pauses: the time a section
/// WORKED, for a timer that judges a step's length (`CardPace` — a paused
/// tick is not a stalled card). Pauses on any thread count, so a section
/// overlapping another thread's pause reads short by it; a worker's steps run
/// on its main loop.
pub(crate) struct Stopwatch {
    started: Instant,
    paused_before: Duration,
}

impl Stopwatch {
    pub(crate) fn start() -> Self {
        Self {
            started: Instant::now(),
            paused_before: paused_total(),
        }
    }

    pub(crate) fn worked(&self) -> Duration {
        self.started
            .elapsed()
            .saturating_sub(paused_total().saturating_sub(self.paused_before))
    }
}

/// The pause a step of `busy` is owed at `permille`, before any carried debt.
fn owed(busy: Duration, permille: u32) -> Duration {
    busy.saturating_mul(permille) / 1000
}

/// A pause still owed, less the time the thread rested since its last step
/// (between two replies, or while the next step waited for work): rest is
/// rest, so a reply never pays for the one before it. An overshoot paid
/// ahead (negative) is kept.
fn after_rest(debt_nanos: i64, rested: Duration) -> i64 {
    if debt_nanos <= 0 {
        return debt_nanos;
    }
    let rested = i64::try_from(rested.as_nanos()).unwrap_or(i64::MAX);
    debt_nanos.saturating_sub(rested).max(0)
}

/// What to sleep now, and the debt carried after it, from the debt before,
/// what this step owes and the shortest pause slept ([`pause_from`]). Pure:
/// [`Step::end`] sleeps it.
fn plan(debt_nanos: i64, owed: Duration, from: Duration) -> (Duration, i64) {
    let owed = i64::try_from(owed.as_nanos()).unwrap_or(i64::MAX);
    let total = debt_nanos.saturating_add(owed);
    if total < from.max(SHORTEST_SLEEP).as_nanos() as i64 {
        // Too short to sleep (or paid ahead by an overshoot): carried.
        return (Duration::ZERO, total);
    }
    let sleep = Duration::from_nanos(total as u64).min(MOST_PAUSE);
    // A pause cut at the cap is not owed again: a step that long is paced
    // as far as the cap allows, never by the next steps.
    (sleep, 0)
}

/// One step a worker runs, paced at its end while the computer is hot. Open
/// one at the top of every forward entry point; only the outermost on a
/// thread pauses.
#[must_use = "a step is paced only when it is ended"]
pub(crate) struct Step {
    started: Instant,
    outermost: bool,
    open: bool,
}

impl Step {
    pub(crate) fn begin() -> Self {
        let outermost = DEPTH.with(|d| {
            let depth = d.get();
            d.set(depth + 1);
            depth == 0
        });
        Self {
            started: Instant::now(),
            outermost,
            open: true,
        }
    }

    /// End the step. While the computer is hot, an outermost step first lets
    /// the device finish (`synchronize`, so the time is the device's work),
    /// then pauses for its share — stopping early once `cancelled` says so.
    pub(crate) fn end(mut self, synchronize: impl FnOnce(), cancelled: impl Fn() -> bool) {
        self.close();
        if !self.outermost {
            return;
        }
        let permille = pause_permille();
        if permille == 0 {
            DEBT_NANOS.with(|d| d.set(0));
            LAST_END.with(|e| e.set(None));
            return;
        }
        synchronize();
        let busy = self.started.elapsed();
        let rested = LAST_END.with(Cell::get).map_or(Duration::ZERO, |e| {
            self.started.saturating_duration_since(e)
        });
        let debt = after_rest(DEBT_NANOS.with(Cell::get), rested);
        let (sleep, debt) = plan(debt, owed(busy, permille), pause_from());
        if sleep.is_zero() {
            DEBT_NANOS.with(|d| d.set(debt));
            LAST_END.with(|e| e.set(Some(Instant::now())));
            return;
        }
        let began = Instant::now();
        while began.elapsed() < sleep && !cancelled() {
            std::thread::sleep(PAUSE_SLICE.min(sleep.saturating_sub(began.elapsed())));
        }
        let slept = began.elapsed();
        PAUSED_NANOS.fetch_add(
            u64::try_from(slept.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        // An overshoot is paid ahead; a pause a cancel cut short is not owed.
        let over = slept.as_nanos() as i64 - sleep.as_nanos() as i64;
        DEBT_NANOS.with(|d| d.set(debt - over.max(0)));
        LAST_END.with(|e| e.set(Some(Instant::now())));
    }

    fn close(&mut self) {
        if std::mem::replace(&mut self.open, false) {
            DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
}

impl Drop for Step {
    /// A step that returned early (an error, an unwind) still closes, so the
    /// next one on the thread is outermost again.
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pace is process-wide; tests that set it run one at a time.
    static PACE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// One test failing while it holds the lock must not fail the others.
    /// Each starts pausing after every step; a test of the piece sets its own.
    fn pace_lock() -> std::sync::MutexGuard<'static, ()> {
        let lock = PACE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PIECE_NANOS.store(SHORTEST_SLEEP.as_nanos() as u64, Ordering::Relaxed);
        lock
    }

    fn busy_for(d: Duration) {
        let t = Instant::now();
        while t.elapsed() < d {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_hot_step_is_followed_by_a_pause_as_long_as_itself() {
        let _lock = pace_lock();
        set_pause_permille(HOT_PAUSE_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        let paused_before = paused_total();
        let t = Instant::now();
        let step = Step::begin();
        busy_for(Duration::from_millis(40));
        step.end(|| {}, || false);
        let took = t.elapsed();
        let paused = paused_total().saturating_sub(paused_before);
        set_pause_permille(0);
        assert!(
            paused >= Duration::from_millis(38) && paused < Duration::from_millis(250),
            "paused {paused:?} after a 40 ms step"
        );
        assert!(
            took >= Duration::from_millis(78),
            "the step and its pause took {took:?}"
        );
    }

    #[test]
    fn a_stopwatch_leaves_out_the_pauses() {
        // `CardPace` times a tick with it: a paused tick is not a stalled card.
        let _lock = pace_lock();
        set_pause_permille(HOT_PAUSE_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        let watch = Stopwatch::start();
        let step = Step::begin();
        busy_for(Duration::from_millis(30));
        step.end(|| {}, || false);
        set_pause_permille(0);
        let worked = watch.worked();
        let elapsed = watch.started.elapsed();
        assert!(elapsed >= Duration::from_millis(58), "{elapsed:?}");
        assert!(
            worked >= Duration::from_millis(29) && worked < Duration::from_millis(50),
            "worked {worked:?} of {elapsed:?}"
        );
    }

    #[test]
    fn a_cool_computer_never_pauses() {
        let _lock = pace_lock();
        set_pause_permille(0);
        let paused_before = paused_total();
        let synchronized = Cell::new(false);
        let step = Step::begin();
        busy_for(Duration::from_millis(10));
        step.end(|| synchronized.set(true), || false);
        assert_eq!(paused_total(), paused_before);
        assert!(!synchronized.get(), "nor waits for the device");
    }

    #[test]
    fn only_the_outermost_step_pauses() {
        let _lock = pace_lock();
        set_pause_permille(HOT_PAUSE_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        let outer = Step::begin();
        let paused_before = paused_total();
        let inner = Step::begin();
        busy_for(Duration::from_millis(20));
        inner.end(|| {}, || false);
        assert_eq!(
            paused_total(),
            paused_before,
            "a batch of one does not pause twice"
        );
        busy_for(Duration::from_millis(5));
        outer.end(|| {}, || false);
        let paused = paused_total().saturating_sub(paused_before);
        set_pause_permille(0);
        assert!(
            paused >= Duration::from_millis(24),
            "the outer step pays for both: {paused:?}"
        );
    }

    #[test]
    fn a_step_that_returns_early_leaves_the_next_one_outermost() {
        let _lock = pace_lock();
        set_pause_permille(HOT_PAUSE_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        drop(Step::begin()); // a `?` out of a forward
        let paused_before = paused_total();
        let step = Step::begin();
        busy_for(Duration::from_millis(10));
        step.end(|| {}, || false);
        set_pause_permille(0);
        assert!(paused_total() > paused_before);
    }

    #[test]
    fn a_cancel_cuts_the_pause_short() {
        let _lock = pace_lock();
        set_pause_permille(MOST_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        let t = Instant::now();
        let step = Step::begin();
        busy_for(Duration::from_millis(100));
        step.end(|| {}, || t.elapsed() > Duration::from_millis(150));
        set_pause_permille(0);
        assert!(
            t.elapsed() < Duration::from_millis(450),
            "{:?}",
            t.elapsed()
        );
    }

    #[test]
    fn short_pauses_are_carried_and_an_overshoot_is_paid_ahead() {
        let from = SHORTEST_SLEEP;
        // A 1 ms step owes 1 ms: too short to sleep, carried.
        let (sleep, debt) = plan(0, owed(Duration::from_millis(1), 1000), from);
        assert_eq!(sleep, Duration::ZERO);
        assert_eq!(debt, 1_000_000);
        // The next 1 ms step brings the debt to a sleep of 2 ms.
        let (sleep, debt) = plan(debt, owed(Duration::from_millis(1), 1000), from);
        assert_eq!(sleep, Duration::from_millis(2));
        assert_eq!(debt, 0);
        // A debt paid ahead by an overshoot absorbs what the next steps owe.
        let (sleep, debt) = plan(-3_000_000, owed(Duration::from_millis(1), 1000), from);
        assert_eq!((sleep, debt), (Duration::ZERO, -2_000_000));
        // A step longer than the cap is paused the cap, and owes nothing on.
        let (sleep, debt) = plan(0, owed(Duration::from_secs(20), 1000), from);
        assert_eq!((sleep, debt), (MOST_PAUSE, 0));
    }

    /// Paused in longer pieces: a card's 16 ms decode steps owe 16 ms each,
    /// carried until a second is owed, then slept at once.
    #[test]
    fn a_pause_is_carried_until_it_reaches_its_piece() {
        let from = Duration::from_secs(1);
        let mut debt = 0;
        for _ in 0..62 {
            let (sleep, d) = plan(debt, owed(Duration::from_millis(16), 1000), from);
            assert_eq!(sleep, Duration::ZERO);
            debt = d;
        }
        let (sleep, debt) = plan(debt, owed(Duration::from_millis(16), 1000), from);
        assert_eq!((sleep, debt), (Duration::from_millis(1008), 0));
    }

    /// The piece on a real thread: two 40 ms steps owe 80 ms, under a 100 ms
    /// piece, and run on unpaused; the third brings it to 120 ms, slept at once.
    #[test]
    fn steps_run_on_until_a_piece_is_owed() {
        let _lock = pace_lock();
        PIECE_NANOS.store(100_000_000, Ordering::Relaxed);
        set_pause_permille(HOT_PAUSE_PERMILLE);
        DEBT_NANOS.with(|d| d.set(0));
        LAST_END.with(|e| e.set(None));
        let paused_before = paused_total();
        for _ in 0..2 {
            let step = Step::begin();
            busy_for(Duration::from_millis(40));
            step.end(|| {}, || false);
        }
        let after_two = paused_total().saturating_sub(paused_before);
        let step = Step::begin();
        busy_for(Duration::from_millis(40));
        step.end(|| {}, || false);
        let after_three = paused_total().saturating_sub(paused_before);
        set_pause_permille(0);
        assert_eq!(after_two, Duration::ZERO);
        assert!(
            after_three >= Duration::from_millis(115) && after_three < Duration::from_millis(400),
            "{after_three:?}"
        );
    }

    /// The default piece is what was measured on the card, unless the A/B
    /// knob names another.
    #[test]
    fn the_piece_defaults_to_a_quarter_second() {
        let _lock = pace_lock();
        PIECE_NANOS.store(0, Ordering::Relaxed);
        if std::env::var_os(PAUSE_FROM_ENV).is_none() {
            assert_eq!(pause_from(), PAUSE_PIECE);
        }
    }

    /// Time the thread rested since its last step pays what it owes, never
    /// below nothing, and leaves an overshoot paid ahead alone.
    #[test]
    fn rest_between_steps_pays_the_pause_owed() {
        let ms = |m: i64| m * 1_000_000;
        assert_eq!(after_rest(ms(900), Duration::from_millis(300)), ms(600));
        assert_eq!(after_rest(ms(900), Duration::from_secs(60)), 0);
        assert_eq!(after_rest(ms(-3), Duration::from_secs(60)), ms(-3));
    }

    #[test]
    fn a_pace_is_read_from_the_spawn_environment_and_bounded() {
        assert_eq!(pace_at_spawn(None), 0);
        assert_eq!(pace_at_spawn(Some(std::ffi::OsStr::new("1000"))), 1000);
        assert_eq!(pace_at_spawn(Some(std::ffi::OsStr::new("hot"))), 0);
        let _lock = pace_lock();
        set_pause_permille(1_000_000);
        assert_eq!(pause_permille(), MOST_PERMILLE);
        set_pause_permille(0);
    }

    /// Guessing on while a check is abroad waits while the computer is hot,
    /// and comes back when it cools.
    #[test]
    fn work_ahead_waits_while_the_computer_is_paced() {
        let _lock = pace_lock();
        set_pause_permille(0);
        assert!(work_ahead_may_run());
        set_pause_permille(HOT_PAUSE_PERMILLE);
        assert!(!work_ahead_may_run());
        set_pause_permille(0);
        assert!(work_ahead_may_run());
    }
}
