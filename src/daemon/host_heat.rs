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
//! The processor is read by its package sensor, and on Windows — where that
//! sensor needs an administrator — by the ACPI thermal zone
//! (`inference::thermal::read_thermal_zone_c`, #440 stage B); a computer with
//! neither is guarded by its card alone.
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
/// A sensor that has given no reading this long is forgotten — its clock and
/// its warm flag — so a hold it started can end: unknown is never hot. A
/// driver that stopped answering is the driver watch's, which holds back on its
/// own. Kubernetes keeps a node's pressure condition only while it was observed
/// within the pressure-transition period (`nodeConditionsObservedSince`,
/// kubelet's default 5 min).
const READING_GONE_AFTER: Duration = Duration::from_secs(300);
/// How long one reading of every sensor may take. `nvidia-smi` is bounded
/// (`vram::nvidia_smi`: 10 s a call, three calls at most); the processor's
/// sensors are not (sysinfo's WMI, PDH), and one that never returned froze the
/// guard in whatever state it was in.
const SENSOR_READ_LIMIT: Duration = Duration::from_secs(40);

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
    /// The last known reading was above the cool line (or the driver was
    /// cooling the card): a hold does not end while either is.
    card_warm: bool,
    processor_warm: bool,
    /// The last KNOWN reading of each sensor ([`READING_GONE_AFTER`]).
    card_read_at: Option<Instant>,
    processor_read_at: Option<Instant>,
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
        // The clock to a hold counts readings AT the hot line, and a known
        // reading under it starts it again: a card that touched its target
        // once and then sat a few degrees under it, its driver not cutting
        // clocks, never spent its cooling. Prior art: Kubernetes keeps a
        // soft threshold's first-observed time only while the threshold is
        // met (`thresholdsFirstObservedAt`, pkg/kubelet/eviction/helpers.go),
        // and the gate kit's sustained stops do the same. The cool line only
        // ENDS a hold. An unknown reading neither starts nor resets the clock
        // (a blocked `nvidia-smi` is the driver watch's to judge, not a sign of
        // cooling) — but a hold needs a reading at the line NOW, and a sensor
        // silent for `READING_GONE_AFTER` is forgotten: one hot reading
        // followed by a sensor that stopped answering (a driver updated under
        // a running system, a broken WMI) held for good, and stopped the work
        // running for others two minutes later (the pre-release review,
        // 2026-11-12). Kubernetes the same way: a threshold whose signal has
        // no observation is not met (`thresholdsMet`, "no observation found
        // for eviction signal").
        let held_for = |since: Option<Instant>, d: Duration| {
            since.is_some_and(|s| now.saturating_duration_since(s) >= d)
        };
        match r.card {
            Some(c) if c.at_target() || c.hardware_protecting => {
                self.card_hot_since.get_or_insert(now);
                self.card_warm = true;
                self.card_read_at = Some(now);
            }
            Some(c) => {
                self.card_hot_since = None;
                self.card_warm = !c.cool();
                self.card_read_at = Some(now);
            }
            None if held_for(self.card_read_at, READING_GONE_AFTER) => {
                self.card_hot_since = None;
                self.card_warm = false;
                self.card_read_at = None;
            }
            None => {}
        }
        match r.processor_c {
            Some(t) if t >= PROCESSOR_HOT_C => {
                self.processor_hot_since.get_or_insert(now);
                self.processor_warm = true;
                self.processor_read_at = Some(now);
            }
            Some(t) => {
                self.processor_hot_since = None;
                self.processor_warm = t > PROCESSOR_COOL_C;
                self.processor_read_at = Some(now);
            }
            None if held_for(self.processor_read_at, READING_GONE_AFTER) => {
                self.processor_hot_since = None;
                self.processor_warm = false;
                self.processor_read_at = None;
            }
            None => {}
        }
        let card_at_line = r
            .card
            .is_some_and(|c| c.at_target() || c.hardware_protecting);
        let processor_at_line = r.processor_c.is_some_and(|t| t >= PROCESSOR_HOT_C);
        let cause = if r.driver_not_answering {
            Some(HeatCause::DriverNotAnswering)
        } else if r.card.is_some_and(|c| c.hardware_protecting) {
            Some(HeatCause::CardProtecting)
        } else if card_at_line && held_for(self.card_hot_since, CARD_HOT_FOR) {
            Some(HeatCause::Card)
        } else if processor_at_line && held_for(self.processor_hot_since, PROCESSOR_HOT_FOR) {
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
                if self.card_warm || self.processor_warm {
                    self.cool_since = None;
                } else if held_for(Some(*self.cool_since.get_or_insert(now)), COOL_FOR) {
                    self.holding = None;
                    self.holding_since = None;
                    self.cool_since = None;
                }
            }
            (None, None) => {}
        }
        // The grace ends on a computer that is STILL hot: a hold in its cool
        // hold-off, or one kept only by a warm reading under the hot line, had
        // its running work stopped 120 s in although it had cooled (the
        // pre-release review, 2026-11-12). Kubernetes' soft threshold evicts
        // only once it has been MET for its grace period
        // (`thresholdsMetGracePeriod`).
        self.stops_running = self.holding.is_some()
            && (self.stops_running
                || cause.is_some_and(HeatCause::stops_running_at_once)
                || (cause.is_some() && held_for(self.holding_since, RUNNING_WORK_STOPS_AFTER)));
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
    /// The card's target temperature once the driver has ANSWERED the
    /// question (it never changes): `Some(None)` for a card that names none,
    /// so a target of N/A is not asked again every round.
    card_target_c: parking_lot::Mutex<Option<Option<f32>>>,
    /// A refresh is running: the health tick never starts a second.
    refreshing: AtomicBool,
    /// A reading of the sensors is still out on a blocking thread, past
    /// [`SENSOR_READ_LIMIT`] or not: the next round never starts a second
    /// (one stuck reader at a time, as `vram::nvidia_smi` keeps one stuck copy).
    sensors_out: AtomicBool,
}

impl HostHeatWatch {
    /// The last reading, for display.
    pub fn last_reading(&self) -> Option<HeatReading> {
        *self.last.lock()
    }
}

/// Clears its flag ([`HostHeatWatch::refreshing`], `sensors_out`) however
/// what set it ends — a panic included.
struct Refreshing<'a>(&'a AtomicBool);

impl Drop for Refreshing<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Why this node takes no new work for other computers right now. One answer
/// for the dispatcher, the prompt-admission ask and the capability, so the
/// reasons can never be applied at different places.
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
    /// An update is draining this node before it restarts (#444,
    /// `update_restart::drain`): work already running finishes, nothing new
    /// starts — Kubernetes' not-ready pod, Envoy's failed health check.
    Updating,
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
    /// connection or an update: either answer would let a split's next steps
    /// through on a computer that must stop them. Then an update draining this
    /// node (#444: the restart takes it off the swarm, the clearest reason to
    /// give), then the metered connection, then the heat.
    pub fn peer_work_hold_now(&self) -> Option<PeerWorkHold> {
        let heat = self
            .host_heat_hold_now()
            .map(|_| self.running_peer_work_stops_now());
        peer_work_hold(
            heat,
            self.events.update_drain.now(),
            self.metered_hold_now(),
        )
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
        let (reading, target) = self
            .read_sensors_within(SENSOR_READ_LIMIT, move || {
                read_heat(reads_card, known_target)
            })
            .await;
        if target.is_some() {
            *watch.card_target_c.lock() = target;
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

    /// `read` ([`read_heat`]) on a blocking thread, waited for at most `limit`
    /// ([`SENSOR_READ_LIMIT`]). A reading that took longer, or one still out
    /// from an earlier round, is unknown — but for the driver watch, which asks
    /// nothing and is read every round: a sensor call that never returned had
    /// left the guard frozen, a stuck driver included (the pre-release review,
    /// 2026-11-12).
    async fn read_sensors_within<F>(
        self: &std::sync::Arc<Self>,
        limit: Duration,
        read: F,
    ) -> (HeatReading, Option<Option<f32>>)
    where
        F: FnOnce() -> (HeatReading, Option<Option<f32>>) + Send + 'static,
    {
        let unknown = || {
            let reading = HeatReading {
                driver_not_answering: driver_not_answering(),
                ..Default::default()
            };
            (reading, None)
        };
        if self
            .metrics
            .host_heat
            .sensors_out
            .swap(true, Ordering::AcqRel)
        {
            return unknown();
        }
        let state = std::sync::Arc::clone(self);
        let reading = tokio::task::spawn_blocking(move || {
            let _back = Refreshing(&state.metrics.host_heat.sensors_out);
            read()
        });
        match tokio::time::timeout(limit, reading).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => unknown(),
            Err(_) => {
                tracing::warn!(
                    limit_s = limit.as_secs_f32(),
                    "reading this computer's temperature has not returned; it counts as unknown \
                     until it does"
                );
                unknown()
            }
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
/// stops too; `updating` while an update drains this node.
fn peer_work_hold(heat: Option<bool>, updating: bool, metered: bool) -> Option<PeerWorkHold> {
    match (heat, updating, metered) {
        (Some(true), _, _) => Some(PeerWorkHold::Heat {
            stops_running: true,
        }),
        (_, true, _) => Some(PeerWorkHold::Updating),
        (_, false, true) => Some(PeerWorkHold::Metered),
        (Some(false), false, false) => Some(PeerWorkHold::Heat {
            stops_running: false,
        }),
        (None, false, false) => None,
    }
}

/// `SWARMLLM_HEAT_GUARD=0`: never hold back for heat (rigs and gates; their
/// safety kit guards the test machine — gotcha #1081). Read once.
fn heat_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_HEAT_GUARD").map_or(true, |v| v.trim() != "0"))
}

/// One reading, on a blocking thread.
/// The reading, and the card's target as the driver answered it (`None` while
/// it has not; see [`HostHeatWatch::card_target_c`]).
fn read_heat(
    reads_card: bool,
    known_target: Option<Option<f32>>,
) -> (HeatReading, Option<Option<f32>>) {
    if let Some(faked) = faked_reading() {
        return (faked, known_target);
    }
    let mut target = known_target;
    let reading = HeatReading {
        card: if reads_card {
            read_card(&mut target)
        } else {
            None
        },
        // On Windows the package sensor needs an administrator; the thermal
        // zone does not (#440 stage B).
        processor_c: crate::inference::thermal::read_package_temp_c()
            .or_else(crate::inference::thermal::read_thermal_zone_c),
        driver_not_answering: driver_not_answering(),
    };
    (reading, target)
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
/// is the driver watch's to judge). Its target is asked until the driver
/// answers once, N/A included: `target` holds the answer.
fn read_card(target: &mut Option<Option<f32>>) -> Option<CardHeat> {
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
    if target.is_none() {
        *target = nvidia_smi(&["-q", "-d", "TEMPERATURE"]).map(|q| parse_card_target(&q));
    }
    parse_card_heat(&out, target.flatten())
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

    /// A card that touched its target once and then sat a few degrees under
    /// it, its driver not cooling it (a laptop card under a long reply), never
    /// held anything back: 83 °C is under the hot line (85), so the clock
    /// starts again there — even though it is above the cool line (80).
    #[test]
    fn a_card_just_under_its_target_restarts_the_clock() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&card(86.5, false), t0);
        for s in [30, 60, 90, 120, 150, 180] {
            assert_eq!(t.observe(&card(83.0, false), t0 + secs(s)), None, "{s} s");
        }
        assert_eq!(t.observe(&card(86.5, false), t0 + secs(210)), None);
        assert_eq!(
            t.observe(&card(86.5, false), t0 + secs(300)),
            None,
            "the clock began again at 210 s"
        );
        assert_eq!(
            t.observe(&card(83.0, true), t0 + secs(330)),
            Some(HeatCause::Card),
            "83 °C with the driver cutting clocks IS at its target: the clock from 210 s ran on"
        );
    }

    /// The processor's clock the same way: one reading at 91 °C between two
    /// at 95 starts it again.
    #[test]
    fn a_processor_just_under_its_hot_line_restarts_the_clock() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let p = |c| HeatReading {
            processor_c: Some(c),
            ..Default::default()
        };
        t.observe(&p(95.0), t0);
        t.observe(&p(91.0), t0 + secs(150));
        t.observe(&p(95.0), t0 + secs(180));
        assert_eq!(t.observe(&p(95.0), t0 + secs(310)), None);
        assert_eq!(
            t.observe(&p(95.0), t0 + secs(480)),
            Some(HeatCause::Processor)
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

    /// One hot reading, then a sensor that stopped answering (a driver
    /// updated under a running system answers every `nvidia-smi` with an
    /// error at once, never a stuck call): nothing is known to be hot, so
    /// nothing is held back and nothing running is stopped. It held for good,
    /// and stopped the work for others at 240 s (the pre-release review,
    /// 2026-11-12).
    #[test]
    fn a_hot_reading_followed_by_silence_holds_nothing_back() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        t.observe(&card(88.0, true), t0);
        for s in (30..=900).step_by(30) {
            assert_eq!(
                t.observe(&HeatReading::default(), t0 + secs(s)),
                None,
                "{s} s"
            );
            assert!(!t.stops_running(), "{s} s");
        }
    }

    /// A hold whose sensor then says nothing is kept while what was last known
    /// stands — new work stays refused — but stops nothing running (nothing
    /// is known hot), and ends once the sensor has been silent five minutes.
    #[test]
    fn a_hold_ends_once_its_sensor_has_been_silent_five_minutes() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let hot = card(88.0, true);
        t.observe(&hot, t0);
        assert_eq!(t.observe(&hot, t0 + secs(120)), Some(HeatCause::Card));
        let silent = HeatReading::default();
        for s in (150..=390).step_by(30) {
            assert_eq!(
                t.observe(&silent, t0 + secs(s)),
                Some(HeatCause::Card),
                "{s} s: the last reading known stands"
            );
            assert!(!t.stops_running(), "{s} s: nothing is known hot");
        }
        assert!(
            t.observe(&silent, t0 + secs(420)).is_some(),
            "silent five minutes: forgotten, and the hold-off starts"
        );
        assert_eq!(t.observe(&silent, t0 + secs(480)), None);
    }

    /// The grace ends on a computer that is still hot: a hold whose card has
    /// cooled, inside its hold-off at 120 s into the hold, lets what runs
    /// finish. It was stopped there (the pre-release review, 2026-11-12).
    #[test]
    fn running_work_is_not_stopped_on_a_computer_that_has_cooled() {
        let t0 = Instant::now();
        let mut t = HeatTracker::default();
        let hot = card(88.0, true);
        t.observe(&hot, t0);
        assert_eq!(t.observe(&hot, t0 + secs(120)), Some(HeatCause::Card));
        t.observe(&hot, t0 + secs(180));
        t.observe(&card(79.0, false), t0 + secs(210));
        assert!(
            t.observe(&card(79.0, false), t0 + secs(240)).is_some(),
            "the hold-off still runs"
        );
        assert!(
            !t.stops_running(),
            "120 s into a hold whose card has cooled"
        );
        assert_eq!(t.observe(&card(79.0, false), t0 + secs(270)), None);
        assert!(!t.stops_running());
    }

    fn test_state() -> std::sync::Arc<SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;

        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    /// A sensor call that does not return in time is unknown, and while it is
    /// still out the next round asks nothing of the sensors — one stuck reader
    /// at a time — until it returns. A call that never returned had frozen the
    /// guard (the pre-release review, 2026-11-12).
    #[tokio::test]
    async fn a_sensor_that_does_not_return_is_unknown_and_never_asked_twice() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        let state = test_state();
        let asked = Arc::new(AtomicUsize::new(0));
        let reader = |asked: Arc<AtomicUsize>, takes: Duration| {
            move || {
                asked.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(takes);
                (card(95.0, true), None)
            }
        };
        let (r, _) = state
            .read_sensors_within(
                Duration::from_millis(50),
                reader(asked.clone(), Duration::from_millis(800)),
            )
            .await;
        assert_eq!(r.card, None, "past its limit: unknown");
        let (r, _) = state
            .read_sensors_within(secs(5), reader(asked.clone(), Duration::ZERO))
            .await;
        assert_eq!(r.card, None, "the first is still out: nothing is asked");
        assert_eq!(asked.load(Ordering::SeqCst), 1);
        let deadline = Instant::now() + secs(10);
        while state.metrics.host_heat.sensors_out.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "the slow reading never returned");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let (r, _) = state
            .read_sensors_within(secs(5), reader(asked.clone(), Duration::ZERO))
            .await;
        assert_eq!(
            r.card.map(|c| c.temp_c),
            Some(95.0),
            "asked again once back"
        );
        assert_eq!(asked.load(Ordering::SeqCst), 2);
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

    /// A heat hold that stops running work wins over a metered connection and
    /// an update's drain, whose answers would let a split's next steps through.
    /// A drain names itself over the metered connection and a soft heat hold.
    #[test]
    fn a_hold_that_stops_running_work_is_never_hidden_by_a_metered_connection() {
        for updating in [false, true] {
            assert_eq!(
                peer_work_hold(Some(true), updating, true),
                Some(PeerWorkHold::Heat {
                    stops_running: true
                })
            );
        }
        assert_eq!(
            peer_work_hold(Some(false), false, true),
            Some(PeerWorkHold::Metered)
        );
        assert_eq!(
            peer_work_hold(Some(false), false, false),
            Some(PeerWorkHold::Heat {
                stops_running: false
            })
        );
        assert_eq!(
            peer_work_hold(None, false, true),
            Some(PeerWorkHold::Metered)
        );
        assert_eq!(peer_work_hold(None, false, false), None);
        for (heat, metered) in [(None, false), (None, true), (Some(false), true)] {
            assert_eq!(
                peer_work_hold(heat, true, metered),
                Some(PeerWorkHold::Updating)
            );
        }
        assert!(PeerWorkHold::Heat {
            stops_running: true
        }
        .stops_running_work());
        assert!(!PeerWorkHold::Metered.stops_running_work());
        assert!(!PeerWorkHold::Updating.stops_running_work());
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
