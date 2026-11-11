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
//! out loud, with no penalty to anyone. The owner's own requests and serving
//! the model pieces it holds go on (pacing the owner's is the next step of
//! #440). The owner is told once per change, in their words.
//!
//! **Work already running for other computers stops too** once the hold has
//! lasted [`RUNNING_WORK_STOPS_AFTER`], and at once when the driver stops
//! answering or the card's hardware protection is on
//! ([`PeerWorkHold::Heat`]'s `stops_running`): a split's next steps are
//! refused (`dispatch::admit_own_forward`, `admit_stream_chunk`) and a whole
//! reply running here is ended in words its coordinator reads as "continue
//! elsewhere" ([`SharedState::stop_running_peer_work`]). Kubernetes' node-
//! pressure eviction draws the same two lines — a soft threshold evicts after
//! its grace period, a hard one at once — and BOINC's suspended GPU task
//! loses only what came after its last checkpoint; ours is what the
//! coordinator already received, which its continuation resumes from
//! (`router::continuation_after`, #236).
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
/// How long a hold lasts before the work for other computers ALREADY running
/// here stops too — the grace period of a soft threshold (Kubernetes' node-
/// pressure eviction). Most replies end inside it, so a computer that only
/// touched its line finishes what it took on; one that stays hot stops being
/// driven by strangers four minutes after its card reached its target (the
/// laptop that froze held its card there for ~20 minutes, gotcha #1081). A
/// driver that stopped answering or the card's own protection stops it at
/// once — the hard thresholds.
pub(crate) const RUNNING_WORK_STOPS_AFTER: Duration = Duration::from_secs(120);

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

    /// A hard threshold: the work already running for other computers stops
    /// as soon as this is read, not after [`RUNNING_WORK_STOPS_AFTER`]. A
    /// driver that does not answer is how the lock-up began (gotcha #1081),
    /// and the hardware's protection means the card is past its slowdown
    /// temperature.
    fn stops_running_at_once(self) -> bool {
        matches!(self, Self::DriverNotAnswering | Self::CardProtecting)
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
    /// When the current hold began, whatever its cause since.
    holding_since: Option<Instant>,
    /// The work already running for other computers stops too. Latched for
    /// the rest of the hold: a driver answering again inside the hold-off
    /// does not hand a stopped split its next steps back.
    stops_running: bool,
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
                self.holding_since.get_or_insert(now);
                self.holding = Some(c);
                self.cool_since = None;
            }
            (Some(_), None) => {
                let warm = self.card_hot_since.is_some() || self.processor_hot_since.is_some();
                if warm {
                    self.cool_since = None;
                } else if held_for(Some(*self.cool_since.get_or_insert(now)), COOL_FOR) {
                    self.holding = None;
                    self.holding_since = None;
                    self.cool_since = None;
                }
            }
            (None, None) => {}
        }
        self.stops_running = self.holding.is_some()
            && (self.stops_running
                || cause.is_some_and(HeatCause::stops_running_at_once)
                || held_for(self.holding_since, RUNNING_WORK_STOPS_AFTER));
        self.holding
    }

    /// Does the work already running here for other computers stop too, as
    /// of the last [`Self::observe`]? Only ever while it holds back.
    pub(crate) fn stops_running(&self) -> bool {
        self.stops_running
    }
}

/// What this node last learned about its heat (`state.metrics.host_heat`).
#[derive(Debug, Default)]
pub struct HostHeatWatch {
    tracker: parking_lot::Mutex<HeatTracker>,
    /// The cause held back for ([`HeatCause::to_u8`]), read without a lock by
    /// the dispatcher and the capability.
    holding: AtomicU8,
    /// [`HeatTracker::stops_running`], read without a lock by the dispatcher.
    stops_running: AtomicBool,
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
    /// The connection is metered (#277, `daemon::metered`). Work already
    /// running goes on, as a download already under way does.
    Metered,
    /// The computer is too hot, or its graphics driver is not answering.
    /// `stops_running`: the work already running here for other computers
    /// stops too ([`RUNNING_WORK_STOPS_AFTER`], or at once on a hard
    /// threshold) — a split's next step is refused like a new request.
    Heat { stops_running: bool },
}

impl PeerWorkHold {
    /// Does work a peer ALREADY has running here stop?
    pub fn stops_running_work(self) -> bool {
        matches!(
            self,
            Self::Heat {
                stops_running: true
            }
        )
    }
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

    /// Does the work already running here for other computers stop, because
    /// this computer has stayed hot or its driver stopped answering? Read
    /// without asking anything; only ever while [`Self::host_heat_hold_now`]
    /// holds.
    pub fn running_peer_work_stops_now(&self) -> bool {
        self.host_heat_hold_now().is_some()
            && self.metrics.host_heat.stops_running.load(Ordering::Relaxed)
    }

    /// The one answer to "does this node take no new work for other computers
    /// now, and why" — and whether the work already running stops too.
    ///
    /// A heat hold that stops running work comes first, whatever the
    /// connection: a metered answer would let a split's next steps through
    /// on a computer that must stop them. Otherwise the metered connection,
    /// then the heat.
    pub fn peer_work_hold_now(&self) -> Option<PeerWorkHold> {
        let heat = self
            .host_heat_hold_now()
            .map(|_| self.running_peer_work_stops_now());
        peer_work_hold(heat, self.metered_hold_now())
    }

    /// End every whole reply this node is producing for another computer —
    /// a whole model handed here, a split led here — while running work stops
    /// for heat. Aborting the reply's task is what `CancelInference` does
    /// (`inbound_generate_aborts`); its last frame then carries the cooling
    /// words (`dispatch::remote_generate::ended_reply_words`), so the
    /// coordinator continues the reply elsewhere from what it received. A
    /// split's steps are refused at the dispatcher instead: each is its own
    /// forward. How many were ended.
    pub(crate) fn stop_running_peer_work(&self) -> usize {
        // Collected first: no map guard is held while a task is aborted.
        let running: Vec<(uuid::Uuid, tokio::task::AbortHandle)> = self
            .inbound_generate_aborts
            .iter()
            .map(|e| (*e.key(), e.value().0.clone()))
            .collect();
        for (request_id, handle) in &running {
            tracing::warn!(
                %request_id,
                "ending a reply this computer was writing for another computer — it is cooling down"
            );
            handle.abort();
        }
        running.len()
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
        let (cause, stops_running) = {
            let mut tracker = watch.tracker.lock();
            let cause = tracker.observe(&reading, Instant::now());
            (cause, tracker.stops_running())
        };
        let before = HeatCause::from_u8(
            watch
                .holding
                .swap(HeatCause::to_u8(cause), Ordering::AcqRel),
        );
        let stopped_before = watch.stops_running.swap(stops_running, Ordering::AcqRel);
        if before.is_some() != cause.is_some() {
            self.tell_the_owner_about_heat(cause, &reading);
        }
        // The owner's own work is never refused, only paced (step 4): every
        // worker pauses after each step as long as the step took, while the
        // hold lasts. Every round — the pool tells only the workers not yet
        // at it. Off with the guard (`host_heat_hold_now` is then `None`).
        let pace = if self.host_heat_hold_now().is_some() {
            crate::inference::pace::HOT_PAUSE_PERMILLE
        } else {
            0
        };
        self.model_process_pool.pace_steps(pace).await;
        if stops_running && heat_guard_on() {
            // Every round while it holds, not only on the edge: a reply that
            // slipped in as the hold began is ended at the next.
            let ended = self.stop_running_peer_work();
            if !stopped_before {
                tracing::warn!(
                    cause = cause.map(HeatCause::as_str),
                    replies_ended = ended,
                    "this computer is still too hot (or its graphics driver is not answering), \
                     so the work it was running for other computers stops too — their replies \
                     continue elsewhere"
                );
            }
        } else if stops_running && !stopped_before {
            tracing::info!(
                cause = cause.map(HeatCause::as_str),
                "heat guard is off (SWARMLLM_HEAT_GUARD=0): it would have stopped the work \
                 running for other computers"
            );
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
                 and your own chats run more slowly until it cools down.",
            ),
            None => (
                "heat_resumed",
                "Your computer has cooled down, so work for other computers has resumed \
                 and your own chats run at full speed again.",
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

/// [`SharedState::peer_work_hold_now`]'s precedence: `heat` is `None` when
/// the computer is not held back for heat, else whether its running work
/// stops too.
fn peer_work_hold(heat: Option<bool>, metered: bool) -> Option<PeerWorkHold> {
    match (heat, metered) {
        (Some(true), _) => Some(PeerWorkHold::Heat {
            stops_running: true,
        }),
        (_, true) => Some(PeerWorkHold::Metered),
        (Some(false), false) => Some(PeerWorkHold::Heat {
            stops_running: false,
        }),
        (None, false) => None,
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
/// `file:<path>` reads the fault from that file at every reading, and the
/// real reading while the file is absent — so a rig can make a node hot in
/// the middle of a reply it is serving.
fn faked_reading() -> Option<HeatReading> {
    let fault = std::env::var("SWARMLLM_FAULT_HOST_HEAT").ok()?;
    match fault.trim().strip_prefix("file:") {
        Some(path) => faked(&std::fs::read_to_string(path).ok()?),
        None => faked(&fault),
    }
}

fn faked(fault: &str) -> Option<HeatReading> {
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

    /// The soft threshold: a hold that lasts stops the work already running
    /// for other computers too, two minutes after it began — never with it.
    #[test]
    fn a_hold_that_lasts_stops_the_running_work_too() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let hot = card(87.7, true);
        t.observe(&hot, t0);
        assert_eq!(t.observe(&hot, t0 + secs(120)), Some(HeatCause::Card));
        assert!(
            !t.stops_running(),
            "new work is held back; what runs finishes"
        );
        t.observe(&hot, t0 + secs(239));
        assert!(!t.stops_running(), "119 s into the hold");
        t.observe(&hot, t0 + secs(240));
        assert!(t.stops_running(), "two minutes into the hold");
    }

    /// A hold that ends before its grace never stops running work, and the
    /// next hold's grace starts again from nothing.
    #[test]
    fn a_short_hold_lets_running_work_finish() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&card(88.0, true), t0);
        t.observe(&card(88.0, true), t0 + secs(120));
        t.observe(&card(79.0, false), t0 + secs(150));
        assert_eq!(t.observe(&card(79.0, false), t0 + secs(210)), None);
        assert!(!t.stops_running());
        t.observe(&card(88.0, true), t0 + secs(300));
        assert!(t.observe(&card(88.0, true), t0 + secs(420)).is_some());
        t.observe(&card(88.0, true), t0 + secs(500));
        assert!(
            !t.stops_running(),
            "80 s into the second hold, not 200 s into the first"
        );
    }

    /// The hard thresholds stop running work at once, and the stop lasts the
    /// whole hold — through the hold-off after the driver answers again — and
    /// ends with it.
    #[test]
    fn a_stuck_driver_or_the_cards_protection_stops_running_work_at_once() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let stuck = HeatReading {
            driver_not_answering: true,
            ..Default::default()
        };
        t.observe(&stuck, t0);
        assert!(t.stops_running(), "a stuck driver: at once");
        t.observe(&card(60.0, false), t0 + secs(30));
        assert!(t.stops_running(), "the hold-off keeps it stopped");
        assert_eq!(t.observe(&card(60.0, false), t0 + secs(90)), None);
        assert!(!t.stops_running(), "it ends with the hold");

        let mut protecting = card(98.0, true);
        protecting.card.as_mut().unwrap().hardware_protecting = true;
        let mut t = HeatTracker::default();
        t.observe(&protecting, t0);
        assert!(t.stops_running(), "the card's own protection: at once");
    }

    /// The control: nothing hot, nothing stopped — and the processor's slow
    /// hold is soft like the card's.
    #[test]
    fn running_work_stops_only_inside_a_hold() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&card(50.0, false), t0);
        assert!(!t.stops_running());
        let p = HeatReading {
            processor_c: Some(95.0),
            ..Default::default()
        };
        t.observe(&p, t0);
        assert!(t.observe(&p, t0 + secs(300)).is_some());
        assert!(!t.stops_running(), "the processor's hold is soft");
        t.observe(&p, t0 + secs(420));
        assert!(t.stops_running());
    }

    /// A heat hold that stops running work wins over a metered connection,
    /// whose answer would let a split's next steps through.
    #[test]
    fn a_hold_that_stops_running_work_is_never_hidden_by_a_metered_connection() {
        assert_eq!(
            peer_work_hold(Some(true), true),
            Some(PeerWorkHold::Heat {
                stops_running: true
            })
        );
        assert_eq!(
            peer_work_hold(Some(false), true),
            Some(PeerWorkHold::Metered)
        );
        assert_eq!(
            peer_work_hold(Some(false), false),
            Some(PeerWorkHold::Heat {
                stops_running: false
            })
        );
        assert_eq!(peer_work_hold(None, true), Some(PeerWorkHold::Metered));
        assert_eq!(peer_work_hold(None, false), None);
        assert!(PeerWorkHold::Heat {
            stops_running: true
        }
        .stops_running_work());
        assert!(!PeerWorkHold::Metered.stops_running_work());
    }

    /// `file:<path>`: the fault in the file while it exists, the real reading
    /// while it does not.
    #[test]
    fn a_fault_read_from_a_file_follows_the_file() {
        assert_eq!(
            faked("driver\n").map(|r| r.driver_not_answering),
            Some(true)
        );
        assert_eq!(faked("nonsense"), None);
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
