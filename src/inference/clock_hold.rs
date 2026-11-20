//! A card between a split's forwards keeps its clocks up (2026-11-20).
//!
//! A split's segment works for a few tens of milliseconds and then waits for the
//! far end — ~400 ms a round between this computer and a laptop in the US. Over
//! that wait the card's own governor reads it as idle and drops its clocks, so
//! the next forward runs at a fraction of the card's speed. Through a
//! Qwen3.8-27B split's replies (`~/swarmllm-gate/split27_prof`, the first 21
//! layers on an RTX 3070 Laptop) the card's clock read 210-420 MHz against 1755
//! at full boost, and the same five-row forward took 26 ms right after a slow
//! one, then 49, 58, 65, 77, 88 and 180 ms as the clocks fell — median 71 ms.
//!
//! Prior art: NVIDIA's forum ("First batch after idle / between workloads is
//! much slower": 13 → 160 ms) answers with locked clocks (`nvidia-smi -lgc`),
//! which need administrator rights no node has; the user-space answer is filler
//! work while waiting, which costs up to one filler's length of latency
//! ("Reducing GPU idle time") — so a filler is short, one at a time, and on the
//! card's ONE stream (`arch-inference.md`). Measured before building
//! (`~/swarmllm-gate/clock_probe`): a ~27 ms forward every 410 ms on that card
//! slid to 50-56 ms with the gaps idle; 2 ms pieces of one block at half the
//! time held the clock at 1755 MHz and the forward at 27.4 ms, at a quarter the
//! clock slid to 555 MHz. A piece costs a forward that arrives during it at
//! most [`PIECE`].
//!
//! The hold starts at every step of a reply shared with other computers a
//! worker is handed ([`starts_a_hold`]: this computer's segments, those it
//! serves for others, and the guesses beside them) and ends [`HOLD_FOR`] after
//! the last, pauses at a message's arrival (it waits only between pieces), and
//! ends while the computer is paced for heat
//! (`pace::work_ahead_may_run`), or for good on a launch that fails. A/B
//! `SWARMLLM_CARD_CLOCK_HOLD=0`.

use std::time::{Duration, Instant};

use crate::inference::worker_ipc::DaemonMsg;

/// How long one piece keeps the card busy.
pub(crate) const PIECE: Duration = Duration::from_millis(2);

/// A piece starts every `PERIOD` while a hold lasts: busy half the time, the
/// share the probe found holds the clock (a quarter let it slide).
pub(crate) const PERIOD: Duration = PIECE.saturating_mul(2);

/// How long a hold lasts after the last forward: past the longest round a split
/// has been measured to wait (~440 ms here ↔ the US), short enough that a reply
/// that has ended stops holding the card within a second.
pub(crate) const HOLD_FOR: Duration = Duration::from_secs(1);

/// `SWARMLLM_CARD_CLOCK_HOLD=0` turns the hold off, for A/B inside one binary.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CARD_CLOCK_HOLD").as_deref() != Ok("0"))
}

/// One worker's hold on its card's clocks.
#[derive(Debug, Default)]
pub(crate) struct ClockHold {
    until: Option<Instant>,
    since: Option<Instant>,
    pieces: u64,
    failed: bool,
}

impl ClockHold {
    /// A message was handed to this worker: one that is a step of a reply
    /// shared with other computers ([`starts_a_hold`]) holds the card's
    /// clocks until [`HOLD_FOR`] after it.
    pub(crate) fn note_message(&mut self, msg: &DaemonMsg, now: Instant) {
        if starts_a_hold(msg) {
            self.note_forward(now);
        }
    }

    /// A step of such a reply was handed to this worker: hold the card's
    /// clocks until [`HOLD_FOR`] after it.
    fn note_forward(&mut self, now: Instant) {
        if !enabled() || self.failed {
            return;
        }
        self.until = Some(now + HOLD_FOR);
        self.since.get_or_insert(now);
    }

    /// Starts one piece on `card` where a hold lasts and the computer is not
    /// paced for heat, and answers how long to wait for a message before the
    /// next; `None` where no hold lasts (nothing started).
    pub(crate) fn start_piece(
        &mut self,
        now: Instant,
        card: Option<&candle_core::Device>,
    ) -> Option<Duration> {
        let until = self.until?;
        let paced = !crate::inference::pace::work_ahead_may_run();
        if now >= until || paced || self.failed {
            self.end(
                now,
                if paced {
                    "paced for heat"
                } else {
                    "no forward since"
                },
            );
            return None;
        }
        match start_piece_on(card?) {
            Ok(true) => {
                self.pieces += 1;
                Some(PERIOD)
            }
            Ok(false) => {
                self.end(now, "no card");
                None
            }
            Err(e) => {
                tracing::warn!(error = %e,
                    "a piece of work to keep the card's clocks up could not start; the card is no longer held");
                self.failed = true;
                self.end(now, "launch failed");
                None
            }
        }
    }

    fn end(&mut self, now: Instant, why: &str) {
        if let Some(since) = self.since.take() {
            tracing::debug!(
                pieces = self.pieces,
                held_ms = now.saturating_duration_since(since).as_millis() as u64,
                why,
                "DIAG: the card's clocks were held between a split's forwards"
            );
        }
        self.until = None;
        self.pieces = 0;
    }
}

/// Whether `msg` is a step of a reply whose next step comes after a round
/// trip to another computer — the work a card idles between: a segment's
/// forward (alone or batched), and the guesses made beside one (a drafter
/// model's, the file's guess head or block drafter, a far run's stand-in). A
/// local reply (`Generate`) keeps its card busy itself (a card/processor run
/// held 1440-1755 MHz, 2026-11-20), and housekeeping is no reply's step. A new
/// variant is weighed here: the match names every one.
pub(crate) fn starts_a_hold(msg: &DaemonMsg) -> bool {
    match msg {
        DaemonMsg::Forward(_)
        | DaemonMsg::BatchForward { .. }
        | DaemonMsg::Draft(_)
        | DaemonMsg::HeadGuess(_)
        | DaemonMsg::BlockDraft(_) => true,
        DaemonMsg::Generate(_)
        | DaemonMsg::Unload { .. }
        | DaemonMsg::PlaceRange { .. }
        | DaemonMsg::ReadCardFree
        | DaemonMsg::LoadRange { .. }
        | DaemonMsg::PrefixFetchResult { .. }
        | DaemonMsg::ExportPrefixSnapshot { .. }
        | DaemonMsg::AdmitPrompt(_)
        | DaemonMsg::CancelRequest { .. }
        | DaemonMsg::ReleaseRequestKv { .. }
        | DaemonMsg::Pace { .. }
        | DaemonMsg::Shutdown => false,
    }
}

/// PTX for `kernels/clock_hold.cu`, compiled by `build.rs`.
#[cfg(feature = "candle-cuda")]
const CLOCK_HOLD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/clock_hold.ptx"));

/// One [`PIECE`] on `device`'s stream; `Ok(false)` where it is no card.
fn start_piece_on(device: &candle_core::Device) -> candle_core::Result<bool> {
    #[cfg(feature = "candle-cuda")]
    if let candle_core::Device::Cuda(dev) = device {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let func =
            dev.get_or_load_custom_func("clock_hold_piece", "swarmllm_clock_hold", CLOCK_HOLD_PTX)?;
        let ns = PIECE.as_nanos() as u64;
        // Three cycles a nanosecond is past any card's clock: this bound ends
        // the piece only where the card's timer stood still.
        let max_cycles = ns.saturating_mul(3);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&ns);
        builder.arg(&max_cycles);
        // SAFETY: ffi. The kernel reads its two arguments and touches no memory.
        unsafe { builder.launch(cfg) }.w()?;
        return Ok(true);
    }
    let _ = device;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hold lasts [`HOLD_FOR`] after the forward that started it, starts no
    /// piece where the model is on no card, and a later forward renews it.
    #[test]
    fn a_hold_ends_a_second_after_the_last_forward_and_never_without_a_card() {
        let t0 = Instant::now();
        let mut hold = ClockHold::default();
        assert_eq!(hold.start_piece(t0, Some(&candle_core::Device::Cpu)), None);
        hold.note_forward(t0);
        assert!(hold.until.is_some());
        // The processor is no card: the hold ends at once, nothing started.
        assert_eq!(hold.start_piece(t0, Some(&candle_core::Device::Cpu)), None);
        assert!(hold.until.is_none());
        hold.note_forward(t0);
        assert_eq!(hold.start_piece(t0, None), None);
        hold.note_forward(t0);
        hold.note_forward(t0 + Duration::from_millis(600));
        assert_eq!(hold.until, Some(t0 + Duration::from_millis(600) + HOLD_FOR));
        assert_eq!(
            hold.start_piece(
                t0 + Duration::from_millis(1700),
                Some(&candle_core::Device::Cpu)
            ),
            None
        );
        assert!(hold.until.is_none() && hold.since.is_none());
    }

    /// A split's step starts a hold; housekeeping between replies does not.
    #[test]
    fn a_split_step_starts_a_hold_and_housekeeping_does_not() {
        let batch = DaemonMsg::BatchForward {
            requests: Vec::new(),
            activation_lens: Vec::new(),
        };
        assert!(starts_a_hold(&batch));
        for quiet in [
            DaemonMsg::ReadCardFree,
            DaemonMsg::Shutdown,
            DaemonMsg::Pace { pause_permille: 0 },
            DaemonMsg::CancelRequest {
                request_id: uuid::Uuid::nil(),
            },
        ] {
            assert!(!starts_a_hold(&quiet));
        }
        let mut hold = ClockHold::default();
        hold.note_message(&DaemonMsg::ReadCardFree, Instant::now());
        assert!(hold.until.is_none());
        hold.note_message(&batch, Instant::now());
        assert_eq!(hold.until.is_some(), enabled());
    }
}
