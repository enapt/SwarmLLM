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
//! the last has been HANDLED — timed from its end, so a prompt pass longer
//! than the hold still holds the round after it — pauses at a message's
//! arrival (it waits only between pieces), and ends while the computer is
//! paced for heat (`pace::work_ahead_may_run`), or for good on a launch that
//! fails. A piece starts only on an idle stream (`cuStreamQuery`, as
//! PyTorch's `Stream.query()` asks): work still queued there — a piece of
//! ours held up by another program sharing the card — already keeps the card
//! busy, so at most one piece of ours is ever queued.
//!
//! **On unless `SWARMLLM_CARD_CLOCK_HOLD=0`.** It was off from 2026-11-20 to
//! 11-21 (FUTURE_WORK #489): a piece is bounded by its length only on its OWN
//! stream, and a second process on the card — a split's drafter model beside its target, or the owner's
//! own reply beside a slice served for a peer — is time-sliced against it.
//! The 14B split with its 0.5B drafter on this card
//! (`~/swarmllm-gate/deep/fr/real_link_readme303`, hold on/off in one build)
//! took ~38 ms a forward here held against ~27 ms off, its code replies
//! 32.1-33.6 tok/s against 36.8-37.2.
//!
//! **So a hold pauses while another process of ours works on the card**
//! (2026-11-21, #489): a worker cannot see another process's stream, the
//! daemon sees every worker's replies in flight, and tells each card worker
//! when that changes (`DaemonMsg::CardBusyBeside`,
//! `ModelProcessPool::tell_cards_who_is_busy`); [`ClockHold::start_piece`]
//! starts nothing while it says busy. And a drafter model never holds
//! ([`starts_a_hold`] leaves out `Draft`): its target, waiting on the same
//! round, holds the card for both. Prior art: none found for holding clocks
//! only while no other process uses the card — NVML cannot say "busy now"
//! under WDDM, stream priority acts inside one process, consumer cards have
//! no MPS (FUTURE_WORK #489) — so the signal is ours to send.
//!
//! That work counts for a second past its last step (`CARD_WORK_LINGERS_FOR`
//! in `process_pool`), so no piece slips in between two of a drafter's
//! guesses. Measured with all three (2026-11-21, `2271a6762`): the 27B split
//! 24 ms a forward held against 51 off, replies 8-15% faster; the 14B split's
//! drafting the same a forward held and not (0-29 pieces a reply); the owner's
//! 1B reply on this card beside a held 27B 204 tok/s held against 198 off.

use std::sync::atomic::{AtomicBool, Ordering};
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

/// `SWARMLLM_CARD_CLOCK_HOLD=0` turns the hold off; it is on otherwise (it
/// pauses while another process of ours works on the card: the module's notes).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_by(std::env::var("SWARMLLM_CARD_CLOCK_HOLD").ok().as_deref()))
}

/// The switch's rule: on unless it is set to `0`, as `SWARMLLM_HEAT_GUARD` reads.
fn on_by(setting: Option<&str>) -> bool {
    setting.is_none_or(|v| v.trim() != "0")
}

/// Whether the daemon last said another of its model processes has work in
/// flight on the card (`DaemonMsg::CardBusyBeside`). A worker process's one
/// reading, set by its reader task the moment the message arrives.
static BUSY_BESIDE: AtomicBool = AtomicBool::new(false);

/// The daemon's word on the card ([`BUSY_BESIDE`]).
pub(crate) fn set_busy_beside(busy: bool) {
    BUSY_BESIDE.store(busy, Ordering::Release);
}

/// Is another process of this daemon working on the card now, as last told?
pub(crate) fn busy_beside() -> bool {
    BUSY_BESIDE.load(Ordering::Acquire)
}

/// One worker's hold on its card's clocks.
#[derive(Debug, Default)]
pub(crate) struct ClockHold {
    until: Option<Instant>,
    since: Option<Instant>,
    pieces: u64,
    /// Turns a piece was not started because the stream was still working.
    busy: u64,
    /// Turns a piece was not started because another process of ours was
    /// working on the card ([`busy_beside`]).
    beside: u64,
    failed: bool,
}

impl ClockHold {
    /// A message has been HANDLED at `now`: one that was a step of a reply
    /// shared with other computers (`step`, [`starts_a_hold`] asked before the
    /// message was handed over) holds the card's clocks until [`HOLD_FOR`]
    /// after its end — never after its arrival, which a long forward would
    /// leave behind with the hold already spent.
    pub(crate) fn note_handled(&mut self, step: bool, now: Instant) {
        if step && enabled() {
            self.note_forward(now);
        }
    }

    /// A step of such a reply was handed to this worker: hold the card's
    /// clocks until [`HOLD_FOR`] after it.
    fn note_forward(&mut self, now: Instant) {
        if self.failed {
            return;
        }
        self.until = Some(now + HOLD_FOR);
        self.since.get_or_insert(now);
    }

    /// Starts one piece on `card` where a hold lasts, the computer is not
    /// paced for heat and no other process of ours works on the card
    /// (`beside`, [`busy_beside`] — while it does, the hold lasts and nothing
    /// starts), and answers how long to wait for a message before the next;
    /// `None` where no hold lasts (nothing started).
    pub(crate) fn start_piece(
        &mut self,
        now: Instant,
        card: Option<&candle_core::Device>,
        beside: bool,
    ) -> Option<Duration> {
        let until = self.until?;
        let Some(card) = card else {
            // No model here runs on a card: nothing to hold, nothing to report.
            *self = Self {
                failed: self.failed,
                ..Self::default()
            };
            return None;
        };
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
        if beside {
            self.beside += 1;
            return Some(PERIOD);
        }
        match start_piece_on(card) {
            Ok(Launch::Started) => {
                self.pieces += 1;
                Some(PERIOD)
            }
            Ok(Launch::StreamBusy) => {
                self.busy += 1;
                Some(PERIOD)
            }
            Ok(Launch::NoCard) => {
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
                stream_busy = self.busy,
                busy_beside = self.beside,
                held_ms = now.saturating_duration_since(since).as_millis() as u64,
                why,
                "DIAG: the card's clocks were held between a split's forwards"
            );
        }
        self.until = None;
        self.pieces = 0;
        self.busy = 0;
        self.beside = 0;
    }
}

/// Whether `msg` is a step of a reply whose next step comes after a round
/// trip to another computer — the work a card idles between: a segment's
/// forward (alone or batched), and the guesses made inside its own process
/// (the file's guess head or block drafter, a far run's stand-in). A
/// drafter MODEL's `Draft` does not: it runs in a process of its own beside
/// its target, whose hold keeps the card up for both — two holding processes
/// would time-slice their pieces against each other's work (#489). A local
/// reply (`Generate`) keeps its card busy itself (a card/processor run held
/// 1440-1755 MHz, 2026-11-20), and housekeeping is no reply's step. A new
/// variant is weighed here: the match names every one.
pub(crate) fn starts_a_hold(msg: &DaemonMsg) -> bool {
    match msg {
        DaemonMsg::Forward(_)
        | DaemonMsg::BatchForward { .. }
        | DaemonMsg::HeadGuess(_)
        | DaemonMsg::BlockDraft(_) => true,
        DaemonMsg::Draft(_)
        | DaemonMsg::Generate(_)
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
        | DaemonMsg::CardBusyBeside { .. }
        | DaemonMsg::Shutdown => false,
    }
}

/// PTX for `kernels/clock_hold.cu`, compiled by `build.rs`.
#[cfg(feature = "candle-cuda")]
const CLOCK_HOLD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/clock_hold.ptx"));

/// What [`start_piece_on`] did. Only a card build starts or finds a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "candle-cuda"), allow(dead_code))]
enum Launch {
    /// A piece was queued on the card's stream.
    Started,
    /// The stream was still working (a piece of ours not yet run, or other
    /// work): nothing was queued, the card is busy already.
    StreamBusy,
    /// The device is no card.
    NoCard,
}

/// One [`PIECE`] on `device`'s stream, where that stream is idle.
fn start_piece_on(device: &candle_core::Device) -> candle_core::Result<Launch> {
    #[cfg(feature = "candle-cuda")]
    if let candle_core::Device::Cuda(dev) = device {
        use candle_core::cuda_backend::cudarc::driver::{sys, LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        // Never a second piece behind one not yet run: a stream held up (by
        // another program time-slicing the card) would otherwise gather one
        // piece every PERIOD, and a forward arriving after them would wait
        // for all of them.
        let stream = dev.cuda_stream();
        stream.context().bind_to_thread().w()?;
        // SAFETY: ffi on the device's own live stream; the query waits for nothing.
        let state = unsafe { sys::cuStreamQuery(stream.cu_stream()) };
        if state == sys::CUresult::CUDA_ERROR_NOT_READY {
            return Ok(Launch::StreamBusy);
        }
        state.result().w()?;
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
        return Ok(Launch::Started);
    }
    let _ = device;
    Ok(Launch::NoCard)
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
        assert_eq!(
            hold.start_piece(t0, Some(&candle_core::Device::Cpu), false),
            None
        );
        hold.note_forward(t0);
        assert!(hold.until.is_some());
        // The processor is no card: the hold ends at once, nothing started.
        assert_eq!(
            hold.start_piece(t0, Some(&candle_core::Device::Cpu), false),
            None
        );
        assert!(hold.until.is_none());
        hold.note_forward(t0);
        assert_eq!(hold.start_piece(t0, None, false), None);
        // A worker with no model on a card drops the hold without a report.
        assert!(hold.until.is_none() && hold.since.is_none());
        hold.note_forward(t0);
        hold.note_forward(t0 + Duration::from_millis(600));
        assert_eq!(hold.until, Some(t0 + Duration::from_millis(600) + HOLD_FOR));
        assert_eq!(
            hold.start_piece(
                t0 + Duration::from_millis(1700),
                Some(&candle_core::Device::Cpu),
                false
            ),
            None
        );
        assert!(hold.until.is_none() && hold.since.is_none());
    }

    /// #489: while another process of ours works on the card, a hold lasts
    /// and starts nothing — it asks again a period later — and once that
    /// work ends the next turn reaches the card (here the processor, which
    /// is no card, so the hold then ends).
    #[test]
    fn a_hold_starts_nothing_while_another_process_works_on_the_card() {
        let t0 = Instant::now();
        let mut hold = ClockHold::default();
        hold.note_forward(t0);
        let cpu = candle_core::Device::Cpu;
        for turn in 1..=3u64 {
            assert_eq!(
                hold.start_piece(t0 + PERIOD * turn as u32, Some(&cpu), true),
                Some(PERIOD),
                "busy beside: wait a period, start nothing"
            );
            assert_eq!((hold.beside, hold.pieces), (turn, 0));
            assert!(hold.until.is_some(), "the hold lasts through it");
        }
        // Past the hold's end it ends, busy beside or not.
        assert_eq!(hold.start_piece(t0 + HOLD_FOR, Some(&cpu), true), None);
        assert!(hold.until.is_none() && hold.beside == 0);
        // The other process done: the turn reaches the device.
        hold.note_forward(t0);
        assert_eq!(hold.start_piece(t0, Some(&cpu), false), None);
        assert!(hold.until.is_none(), "the processor is no card");
    }

    /// The daemon's word is the worker's one reading.
    #[test]
    fn the_daemons_word_on_the_card_is_kept() {
        set_busy_beside(true);
        assert!(busy_beside());
        set_busy_beside(false);
        assert!(!busy_beside());
    }

    /// On unless turned off: a hold pauses beside our other work on the card
    /// (FUTURE_WORK #489).
    #[test]
    fn the_hold_is_on_unless_it_is_turned_off() {
        assert!(on_by(None));
        assert!(on_by(Some("1")));
        assert!(on_by(Some("")));
        assert!(!on_by(Some("0")));
        assert!(!on_by(Some(" 0 ")));
    }

    /// A split's step starts a hold; housekeeping between replies does not.
    #[test]
    fn a_split_step_starts_a_hold_and_housekeeping_does_not() {
        let batch = DaemonMsg::BatchForward {
            requests: Vec::new(),
            activation_lens: Vec::new(),
        };
        assert!(starts_a_hold(&batch));
        // A drafter model's guess runs in a process of its own beside its
        // target, which holds the card for both (#489).
        let draft = DaemonMsg::Draft(crate::inference::worker_ipc::IpcDraft {
            request_id: uuid::Uuid::nil(),
            model_id: crate::types::ModelId("drafter".into()),
            layer_range: (0, 24),
            keep: 0,
            append: vec![1, 2],
            gamma: 4,
            sampling: crate::types::SamplingParams::default(),
            history: Vec::new(),
            coupling_seed: None,
            for_the_owner: false,
            runner_ups: false,
            stop_below: None,
            tree: None,
            flow: None,
        });
        for quiet in [
            draft,
            DaemonMsg::ReadCardFree,
            DaemonMsg::Shutdown,
            DaemonMsg::Pace { pause_permille: 0 },
            DaemonMsg::CardBusyBeside { busy: true },
            DaemonMsg::CancelRequest {
                request_id: uuid::Uuid::nil(),
            },
        ] {
            assert!(!starts_a_hold(&quiet), "{quiet:?}");
        }
        let mut hold = ClockHold::default();
        hold.note_handled(starts_a_hold(&DaemonMsg::ReadCardFree), Instant::now());
        assert!(hold.until.is_none());
        hold.note_handled(starts_a_hold(&batch), Instant::now());
        assert_eq!(hold.until.is_some(), enabled());
    }
}
