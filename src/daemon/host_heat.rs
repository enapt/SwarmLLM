//! The host heat guard (FUTURE_WORK #440): this node takes no new work for
//! other computers while the computer it runs on is overheating, or while its
//! graphics driver has stopped answering.
//!
//! # Why
//!
//! On 2026-11-11 the development laptop (Ryzen 7 5800H + RTX 3070 Laptop) hard-froze
//! under twenty minutes of sustained card load and then refused to power on
//! until it had cooled (gotcha #1081). The card sat at its 87 °C TARGET with its
//! clocks cut to 780 MHz, the processor's zone at 93-96 °C, and `nvidia-smi`
//! blocked for minutes at a time long before Windows froze. A node serving the
//! swarm puts that load on a volunteer's card for as long as peers ask, and
//! nothing watched it: `inference::thermal` reads only the processor, only
//! warns, and only in the model worker. The user: "swarmllm cant be locking up
//! peoples systems like that. it discourages use."
//!
//! Prior art: GPU miners stop a card at a temperature and start it again below
//! a lower one (NBMiner: start = limit − 5; lolMiner `--tstop`/`--tstart`); BOINC
//! suspends GPU work while the user is active and duty-cycles the processor;
//! Apple's `ProcessInfo.thermalState` asks for less work from `fair` on. None
//! of Ollama, LM Studio, llama.cpp, vLLM, Petals or exo has a guard (searched
//! 2026-11-11, FUTURE_WORK #440). Taken here: stop and start lines with a
//! hold-off, so the state never flaps; the card as the first signal; and the
//! driver's OWN word that it is cooling the card (`clocks_event_reasons.
//! sw_thermal_slowdown`), because a card at its target is not yet throttled by
//! the hardware (that is 98 °C here).
//!
//! # What it does
//!
//! [`SharedState::peer_work_hold_now`] — with the metered hold, the one answer
//! the dispatcher reads before it starts a peer's request (`dispatch::
//! admits_a_new_request`), and the capability reads for `can_serve_inference`
//! — so planners leave this node out, and a peer that asks anyway is refused
//! out loud, with no penalty to anyone. Work already running, the owner's own
//! requests and serving the model pieces it holds all go on (the next steps of
//! #440 slow those too). The owner is told once per change, in their words.
//!
//! `SWARMLLM_HEAT_GUARD=0` turns the hold off (rigs and gates, whose safety
//! kit guards the test machine); `SWARMLLM_FAULT_HOST_HEAT` fakes a reading.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::daemon::state::{ActivityEvent, SharedState};

/// A card this close to its target temperature counts as AT it: the driver
/// holds it there by cutting clocks, so a card at its target under load is a
/// card whose cooling is already spent.
const CARD_AT_TARGET_WITHIN_C: f32 = 2.0;
/// The target assumed where the card names none (an older driver, a card that
/// reports N/A): NVIDIA's laptop figure.
const CARD_TARGET_UNKNOWN_C: f32 = 87.0;
/// A card is cool again this far under its target, and not being cooled by
/// its driver.
const CARD_COOL_UNDER_TARGET_C: f32 = 7.0;
/// How long a card stays at its target before this node stops taking work
/// for others: a reply's burst reaches it for seconds, minutes of strangers'
/// work keep it there. Healthy release checks touched 85.9-86.5 °C for one
/// reading at most (gotcha #1081).
const CARD_HOT_FOR: Duration = Duration::from_secs(120);
/// Processor package at or past this counts as hot. Laptop firmware holds
/// ~95 °C under load BY DESIGN, so seconds there are normal; the guard is for
/// minutes of it.
const PROCESSOR_HOT_C: f32 = 93.0;
const PROCESSOR_COOL_C: f32 = 88.0;
const PROCESSOR_HOT_FOR: Duration = Duration::from_secs(300);
/// Nothing warm for this long before work for others resumes: a hold-off, so
/// a machine cooling across the line does not take a burst of work and heat
/// straight back up.
const COOL_FOR: Duration = Duration::from_secs(60);

const _: () = assert!(PROCESSOR_COOL_C < PROCESSOR_HOT_C);
const _: () = assert!(CARD_AT_TARGET_WITHIN_C < CARD_COOL_UNDER_TARGET_C);

/// The card's heat, as its driver reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CardHeat {
    pub temp_c: f32,
    /// The driver is cutting clocks to hold the card at its target
    /// ("SW Thermal Slowdown").
    pub driver_cooling: bool,
    /// The hardware's own protection is on ("HW Thermal Slowdown": clocks at
    /// least halved, past the slowdown temperature).
    pub hardware_protecting: bool,
    /// The temperature the driver holds the card at ("GPU Target Temperature
    /// Specification"); `None` where the card does not say.
    pub target_c: Option<f32>,
}

impl CardHeat {
    fn target(&self) -> f32 {
        self.target_c.unwrap_or(CARD_TARGET_UNKNOWN_C)
    }
    fn at_target(&self) -> bool {
        self.driver_cooling || self.temp_c >= self.target() - CARD_AT_TARGET_WITHIN_C
    }
    fn cool(&self) -> bool {
        !self.driver_cooling
            && !self.hardware_protecting
            && self.temp_c <= self.target() - CARD_COOL_UNDER_TARGET_C
    }
}

/// One reading of the computer's heat. Anything unknown is `None`/`false`,
/// and unknown is never hot: most machines expose some of this and not the
/// rest (`inference::thermal`'s rule).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HeatReading {
    pub card: Option<CardHeat>,
    pub processor_c: Option<f32>,
    /// An `nvidia-smi` has waited on the driver for a minute or more
    /// (`vram::graphics_driver_not_answering`).
    pub driver_not_answering: bool,
}

/// Why this node is holding back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatCause {
    /// The card has stayed at its target temperature.
    Card,
    /// The card's hardware protection is on.
    CardProtecting,
    /// The processor has stayed hot.
    Processor,
    /// The graphics driver has stopped answering.
    DriverNotAnswering,
}

impl HeatCause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Card => "card_at_target",
            Self::CardProtecting => "card_protecting",
            Self::Processor => "processor_hot",
            Self::DriverNotAnswering => "driver_not_answering",
        }
    }

    fn to_u8(c: Option<Self>) -> u8 {
        match c {
            None => 0,
            Some(Self::Card) => 1,
            Some(Self::CardProtecting) => 2,
            Some(Self::Processor) => 3,
            Some(Self::DriverNotAnswering) => 4,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Card),
            2 => Some(Self::CardProtecting),
            3 => Some(Self::Processor),
            4 => Some(Self::DriverNotAnswering),
            _ => None,
        }
    }
}

/// The stop/start decision over successive readings. Pure but for the clock
/// it is handed, so every rule is tested without owning a hot computer.
#[derive(Debug, Default)]
pub(crate) struct HeatTracker {
    card_hot_since: Option<Instant>,
    processor_hot_since: Option<Instant>,
    cool_since: Option<Instant>,
    holding: Option<HeatCause>,
}

impl HeatTracker {
    /// Take one reading at `now`; the cause held back for afterwards, if any.
    pub(crate) fn observe(&mut self, r: &HeatReading, now: Instant) -> Option<HeatCause> {
        // Between the hot and cool lines — and an unknown reading — keep
        // whatever was running: a blocked `nvidia-smi` is the driver watch's
        // to judge, not a sign the card cooled.
        match r.card {
            Some(c) if c.at_target() || c.hardware_protecting => {
                self.card_hot_since.get_or_insert(now);
            }
            Some(c) if c.cool() => self.card_hot_since = None,
            _ => {}
        }
        match r.processor_c {
            Some(t) if t >= PROCESSOR_HOT_C => {
                self.processor_hot_since.get_or_insert(now);
            }
            Some(t) if t <= PROCESSOR_COOL_C => self.processor_hot_since = None,
            _ => {}
        }
        let held_for = |since: Option<Instant>, d: Duration| {
            since.is_some_and(|s| now.saturating_duration_since(s) >= d)
        };
        let cause = if r.driver_not_answering {
            Some(HeatCause::DriverNotAnswering)
        } else if r.card.is_some_and(|c| c.hardware_protecting) {
            Some(HeatCause::CardProtecting)
        } else if held_for(self.card_hot_since, CARD_HOT_FOR) {
            Some(HeatCause::Card)
        } else if held_for(self.processor_hot_since, PROCESSOR_HOT_FOR) {
            Some(HeatCause::Processor)
        } else {
            None
        };
        match (self.holding, cause) {
            (_, Some(c)) => {
                self.holding = Some(c);
                self.cool_since = None;
            }
            (Some(_), None) => {
                let warm = self.card_hot_since.is_some() || self.processor_hot_since.is_some();
                if warm {
                    self.cool_since = None;
                } else if held_for(Some(*self.cool_since.get_or_insert(now)), COOL_FOR) {
                    self.holding = None;
                    self.cool_since = None;
                }
            }
            (None, None) => {}
        }
        self.holding
    }
}

/// What this node last learned about its heat (`state.metrics.host_heat`).
#[derive(Debug, Default)]
pub struct HostHeatWatch {
    tracker: parking_lot::Mutex<HeatTracker>,
    /// The cause held back for ([`HeatCause::to_u8`]), read without a lock by
    /// the dispatcher and the capability.
    holding: AtomicU8,
    /// The last reading, for the diagnostics report.
    last: parking_lot::Mutex<Option<HeatReading>>,
    /// The card's target temperature once learned (it never changes).
    card_target_c: parking_lot::Mutex<Option<f32>>,
    /// A refresh is running: the health tick never starts a second.
    refreshing: AtomicBool,
}

impl HostHeatWatch {
    /// The last reading, for display.
    pub fn last_reading(&self) -> Option<HeatReading> {
        *self.last.lock()
    }
}

/// Clears [`HostHeatWatch::refreshing`] however the refresh ends.
struct Refreshing<'a>(&'a AtomicBool);

impl Drop for Refreshing<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Why this node takes no new work for other computers right now. One answer
/// for the dispatcher, the prompt-admission ask and the capability, so the
/// two reasons can never be applied at different places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerWorkHold {
    /// The connection is metered (#277, `daemon::metered`).
    Metered,
    /// The computer is too hot, or its graphics driver is not answering.
    Heat,
}

impl SharedState {
    /// Is this node holding back new work for other computers because of its
    /// heat, and why? Read without asking anything — for the dispatch loop and
    /// the capability; [`Self::refresh_host_heat`] keeps it fresh.
    pub fn host_heat_hold_now(&self) -> Option<HeatCause> {
        if !heat_guard_on() {
            return None;
        }
        HeatCause::from_u8(self.metrics.host_heat.holding.load(Ordering::Relaxed))
    }

    /// The one answer to "does this node take no new work for other computers
    /// now, and why" — the metered connection first, then the heat.
    pub fn peer_work_hold_now(&self) -> Option<PeerWorkHold> {
        if self.metered_hold_now() {
            Some(PeerWorkHold::Metered)
        } else if self.host_heat_hold_now().is_some() {
            Some(PeerWorkHold::Heat)
        } else {
            None
        }
    }

    /// Read the computer's heat (off the calling task: `nvidia-smi` and the
    /// sensors run on the blocking pool), update the hold, and tell the owner
    /// when it changes. From the health monitor's tick; one at a time.
    pub async fn refresh_host_heat(self: &std::sync::Arc<Self>) {
        let watch = &self.metrics.host_heat;
        if watch.refreshing.swap(true, Ordering::AcqRel) {
            return;
        }
        let _refreshing = Refreshing(&watch.refreshing);
        let reads_card = self
            .gpu_info
            .as_ref()
            .is_some_and(|g| g.backend.eq_ignore_ascii_case("cuda"));
        let known_target = *watch.card_target_c.lock();
        let reading = tokio::task::spawn_blocking(move || read_heat(reads_card, known_target))
            .await
            .unwrap_or_default();
        if let Some(t) = reading.card.and_then(|c| c.target_c) {
            *watch.card_target_c.lock() = Some(t);
        }
        *watch.last.lock() = Some(reading);
        let cause = watch.tracker.lock().observe(&reading, Instant::now());
        let before = HeatCause::from_u8(
            watch
                .holding
                .swap(HeatCause::to_u8(cause), Ordering::AcqRel),
        );
        if before.is_some() != cause.is_some() {
            self.tell_the_owner_about_heat(cause, &reading);
        }
    }

    fn tell_the_owner_about_heat(&self, cause: Option<HeatCause>, reading: &HeatReading) {
        let card_c = reading.card.map(|c| c.temp_c);
        let (kind, message) = match cause {
            // The words of `activity.<kind>` in `frontend/i18n/en.json`, which
            // the dashboard shows translated.
            Some(HeatCause::DriverNotAnswering) => (
                "card_unresponsive_paused",
                "Your graphics card has stopped responding, so work for other computers \
                 is paused until it responds again.",
            ),
            Some(_) => (
                "heat_paused",
                "Your computer is running hot, so work for other computers is paused \
                 until it cools down. Your own chats still work.",
            ),
            None => (
                "heat_resumed",
                "Your computer has cooled down, so work for other computers has resumed.",
            ),
        };
        if !heat_guard_on() {
            tracing::info!(
                cause = cause.map(HeatCause::as_str),
                card_c,
                processor_c = reading.processor_c,
                "heat guard is off (SWARMLLM_HEAT_GUARD=0): it would have {}",
                if cause.is_some() {
                    "held back work for other computers"
                } else {
                    "resumed it"
                }
            );
            return;
        }
        if cause.is_some() {
            tracing::warn!(
                cause = cause.map(HeatCause::as_str),
                card_c,
                card_driver_cooling = reading.card.map(|c| c.driver_cooling),
                card_target_c = reading.card.and_then(|c| c.target_c),
                processor_c = reading.processor_c,
                "{message}"
            );
        } else {
            tracing::info!(card_c, processor_c = reading.processor_c, "{message}");
        }
        self.emit_activity(
            ActivityEvent::new("system", kind, message.to_string()).with_toast("info", 10000),
        );
    }
}

/// `SWARMLLM_HEAT_GUARD=0`: never hold back for heat (rigs and gates; their
/// safety kit guards the test machine — gotcha #1081). Read once.
fn heat_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_HEAT_GUARD").map_or(true, |v| v.trim() != "0"))
}

/// One reading, on a blocking thread.
fn read_heat(reads_card: bool, known_target: Option<f32>) -> HeatReading {
    if let Some(faked) = faked_reading() {
        return faked;
    }
    HeatReading {
        card: if reads_card {
            read_card(known_target)
        } else {
            None
        },
        processor_c: crate::inference::thermal::read_package_temp_c(),
        driver_not_answering: driver_not_answering(),
    }
}

#[cfg(feature = "candle-cuda")]
fn driver_not_answering() -> bool {
    crate::model::auto_manage::vram::graphics_driver_not_answering()
}

#[cfg(not(feature = "candle-cuda"))]
fn driver_not_answering() -> bool {
    false
}

/// The card's temperature and the driver's cooling flags, through the one
/// bounded `nvidia-smi` (a driver that does not answer reads as unknown, and
/// is the driver watch's to judge). Its target is asked once.
fn read_card(known_target: Option<f32>) -> Option<CardHeat> {
    use crate::model::auto_manage::vram::nvidia_smi;
    // `clocks_throttle_reasons` is the older spelling, which current drivers
    // still take (621.01 answers both, 2026-11-11).
    let fields = |prefix: &str| {
        format!(
            "--query-gpu=temperature.gpu,{prefix}.sw_thermal_slowdown,{prefix}.hw_thermal_slowdown"
        )
    };
    let out = nvidia_smi(&[
        &fields("clocks_event_reasons"),
        "--format=csv,noheader,nounits",
    ])
    .or_else(|| {
        nvidia_smi(&[
            &fields("clocks_throttle_reasons"),
            "--format=csv,noheader,nounits",
        ])
    })?;
    let target = known_target
        .or_else(|| nvidia_smi(&["-q", "-d", "TEMPERATURE"]).and_then(|q| parse_card_target(&q)));
    parse_card_heat(&out, target)
}

/// `temperature, sw_thermal, hw_thermal` per card, one line each: the hottest
/// card's temperature, and a flag set if any card's is.
pub(crate) fn parse_card_heat(out: &str, target_c: Option<f32>) -> Option<CardHeat> {
    let mut heat: Option<CardHeat> = None;
    for line in out.lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 3 {
            continue;
        }
        let Ok(temp_c) = f[0].parse::<f32>() else {
            continue;
        };
        if !(0.0..150.0).contains(&temp_c) {
            continue;
        }
        let active = |s: &str| s.eq_ignore_ascii_case("active");
        let card = CardHeat {
            temp_c,
            driver_cooling: active(f[1]),
            hardware_protecting: active(f[2]),
            target_c,
        };
        heat = Some(match heat {
            None => card,
            Some(h) => CardHeat {
                temp_c: h.temp_c.max(card.temp_c),
                driver_cooling: h.driver_cooling || card.driver_cooling,
                hardware_protecting: h.hardware_protecting || card.hardware_protecting,
                target_c,
            },
        });
    }
    heat
}

/// "GPU Target Temperature Specification : 87 C" from `nvidia-smi -q -d
/// TEMPERATURE` — the lowest if several cards name one; `None` for N/A.
pub(crate) fn parse_card_target(out: &str) -> Option<f32> {
    out.lines()
        .filter(|l| l.contains("Target Temperature"))
        .filter_map(|l| l.split(':').nth(1))
        .filter_map(|v| v.trim().trim_end_matches('C').trim().parse::<f32>().ok())
        .filter(|t| (40.0..120.0).contains(t))
        .reduce(f32::min)
}

/// `SWARMLLM_FAULT_HOST_HEAT=card|protecting|processor|driver|cool`: a reading
/// that says so, for proving the hold on a computer that is not hot. `card`
/// and `processor` hold back only after their minutes, as a real one would.
fn faked_reading() -> Option<HeatReading> {
    let fault = std::env::var("SWARMLLM_FAULT_HOST_HEAT").ok()?;
    let card = |temp_c, driver_cooling, hardware_protecting| CardHeat {
        temp_c,
        driver_cooling,
        hardware_protecting,
        target_c: Some(87.0),
    };
    Some(match fault.trim() {
        "card" => HeatReading {
            card: Some(card(87.5, true, false)),
            ..Default::default()
        },
        "protecting" => HeatReading {
            card: Some(card(98.0, true, true)),
            ..Default::default()
        },
        "processor" => HeatReading {
            processor_c: Some(95.0),
            ..Default::default()
        },
        "driver" => HeatReading {
            driver_not_answering: true,
            ..Default::default()
        },
        "cool" => HeatReading {
            card: Some(card(50.0, false, false)),
            processor_c: Some(60.0),
            ..Default::default()
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(temp_c: f32, driver_cooling: bool) -> HeatReading {
        HeatReading {
            card: Some(CardHeat {
                temp_c,
                driver_cooling,
                hardware_protecting: false,
                target_c: Some(87.0),
            }),
            ..Default::default()
        }
    }

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    /// The lock-up's own readings (gotcha #1081): the card at 87.7 °C with the
    /// driver cooling it. Not at once — a reply's burst reaches its target for
    /// seconds — but once it has held there for two minutes.
    #[test]
    fn a_card_held_at_its_target_stops_work_for_others_after_two_minutes() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let hot = card(87.7, true);
        assert_eq!(
            t.observe(&hot, t0),
            None,
            "the first reading only starts the clock"
        );
        assert_eq!(t.observe(&hot, t0 + secs(90)), None, "90 s is a burst");
        assert_eq!(t.observe(&hot, t0 + secs(120)), Some(HeatCause::Card));
    }

    /// A healthy release check's hottest reading (86.5 °C, one reading) never
    /// holds anything back: the card dropped before two minutes passed.
    #[test]
    fn a_short_burst_at_the_target_holds_nothing_back() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        assert_eq!(t.observe(&card(86.5, false), t0), None);
        assert_eq!(t.observe(&card(78.0, false), t0 + secs(30)), None);
        assert_eq!(t.observe(&card(86.5, false), t0 + secs(100)), None);
        assert_eq!(
            t.observe(&card(86.5, false), t0 + secs(200)),
            None,
            "the clock restarted when the card cooled at 30 s"
        );
    }

    /// Stop and start lines: once held back, a card hovering just under its
    /// target keeps it held; only well under it, for a minute, lets work back.
    #[test]
    fn work_resumes_only_once_the_card_is_well_under_its_target_for_a_minute() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&card(88.0, true), t0);
        assert_eq!(
            t.observe(&card(88.0, true), t0 + secs(120)),
            Some(HeatCause::Card)
        );
        assert_eq!(
            t.observe(&card(83.0, false), t0 + secs(200)),
            Some(HeatCause::Card),
            "83 °C is under the target but not under the cool line (80)"
        );
        assert_eq!(
            t.observe(&card(79.0, false), t0 + secs(230)),
            Some(HeatCause::Card),
            "cool now, but the hold-off has only started"
        );
        assert_eq!(t.observe(&card(79.0, false), t0 + secs(290)), None);
    }

    #[test]
    fn a_driver_that_stopped_answering_holds_back_at_once_and_its_return_lets_work_back() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let stuck = HeatReading {
            driver_not_answering: true,
            ..Default::default()
        };
        assert_eq!(t.observe(&stuck, t0), Some(HeatCause::DriverNotAnswering));
        let back = card(60.0, false);
        assert!(t.observe(&back, t0 + secs(30)).is_some(), "the hold-off");
        assert_eq!(t.observe(&back, t0 + secs(90)), None);
    }

    #[test]
    fn the_cards_hardware_protection_holds_back_at_once() {
        let mut r = card(98.0, true);
        r.card.as_mut().unwrap().hardware_protecting = true;
        let mut t = HeatTracker::default();
        assert_eq!(
            t.observe(&r, Instant::now()),
            Some(HeatCause::CardProtecting)
        );
    }

    /// A laptop processor sits at ~95 °C under load by its firmware's design:
    /// minutes of it hold back, seconds do not, and 90 °C never starts it.
    #[test]
    fn a_processor_holds_back_only_after_five_minutes_hot() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let p = |c| HeatReading {
            processor_c: Some(c),
            ..Default::default()
        };
        assert_eq!(t.observe(&p(90.0), t0), None);
        assert_eq!(t.observe(&p(95.0), t0 + secs(10)), None);
        assert_eq!(t.observe(&p(95.0), t0 + secs(250)), None);
        assert_eq!(
            t.observe(&p(95.0), t0 + secs(310)),
            Some(HeatCause::Processor)
        );
    }

    /// Unknown is never hot, and an unknown reading in the middle of a hot
    /// spell neither starts nor ends anything (a blocked `nvidia-smi`).
    #[test]
    fn an_unknown_reading_is_never_hot_and_never_cools_a_hot_card() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        assert_eq!(t.observe(&HeatReading::default(), t0), None);
        t.observe(&card(88.0, true), t0);
        assert_eq!(
            t.observe(&HeatReading::default(), t0 + secs(60)),
            None,
            "an unknown reading does not start the hold"
        );
        assert_eq!(
            t.observe(&card(88.0, true), t0 + secs(121)),
            Some(HeatCause::Card),
            "and did not reset the card's clock either"
        );
    }

    #[test]
    fn a_card_naming_no_target_is_judged_against_eighty_seven() {
        let r = HeatReading {
            card: Some(CardHeat {
                temp_c: 85.5,
                driver_cooling: false,
                hardware_protecting: false,
                target_c: None,
            }),
            ..Default::default()
        };
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&r, t0);
        assert_eq!(t.observe(&r, t0 + secs(120)), Some(HeatCause::Card));
    }

    /// The fields as this laptop's driver prints them (621.01, RTX 3070 Laptop,
    /// read 2026-11-11), and two cards on one line each.
    #[test]
    fn nvidia_smis_answers_are_read() {
        assert_eq!(
            parse_card_heat("52.60, Not Active, Not Active\n", Some(87.0)),
            Some(CardHeat {
                temp_c: 52.6,
                driver_cooling: false,
                hardware_protecting: false,
                target_c: Some(87.0),
            })
        );
        let two = parse_card_heat("61, Not Active, Not Active\n87, Active, Not Active\n", None)
            .expect("two cards");
        assert_eq!(two.temp_c, 87.0);
        assert!(two.driver_cooling && !two.hardware_protecting);
        assert_eq!(parse_card_heat("[N/A], [N/A], [N/A]\n", None), None);
        assert_eq!(parse_card_heat("", None), None);

        let q = "    Temperature\n        GPU Current Temp                               : 52.60 C\n        GPU Current Max Temp                           : N/A\n        GPU Current T.Limit Temp                       : N/A\n        GPU Shutdown Temp                              : 101 C\n        GPU Slowdown Temp                              : 98 C\n        GPU Max Operating Temp                         : 105 C\n        GPU Target Temperature Specification           : 87 C\n        Memory Current Temp                            : N/A\n";
        assert_eq!(parse_card_target(q), Some(87.0));
        assert_eq!(
            parse_card_target("        GPU Target Temperature Specification : N/A\n"),
            None
        );
    }

    /// The control for the tests above: a tracker that held back on the first
    /// hot reading would hold back during a release check's bursts.
    #[test]
    fn the_first_hot_reading_alone_never_holds_back() {
        let mut t = HeatTracker::default();
        assert_eq!(t.observe(&card(95.0, true), Instant::now()), None);
    }

    #[test]
    fn a_cause_survives_its_atomic_round_trip() {
        for c in [
            None,
            Some(HeatCause::Card),
            Some(HeatCause::CardProtecting),
            Some(HeatCause::Processor),
            Some(HeatCause::DriverNotAnswering),
        ] {
            assert_eq!(HeatCause::from_u8(HeatCause::to_u8(c)), c);
        }
    }
}
