//! Thermal backstop for CPU inference.
//!
//! # Why
//!
//! Reported 2026-08-10: a 7B model fell back to the CPU (the model did not fit
//! the reporter's 6 GB card), a real coding-agent prompt pegged ~487% CPU, and
//! the package went from 71 °C to 88 °C in about five minutes without the
//! request finishing. They killed it by hand. Their note is the point:
//!
//! > nothing would have stopped this on its own [...] config.toml has ceilings
//! > for VRAM, RAM, disk, bandwidth, concurrent requests and rate limits, but
//! > nothing tied to CPU load or temperature.
//!
//! Every other resource this daemon spends has a ceiling. Heat did not.
//!
//! # What it does — and what it does NOT do
//!
//! It **tells you**, once, when the machine crosses a temperature that a node
//! should not sit at, and again when it recovers. That is all. It never refuses
//! work, never stops serving, and does not change how inference runs.
//!
//! **It deliberately does not throttle, because throttling was tried here and
//! measured to do nothing.** Routing both phases through a half-width thread
//! pool while hot left CPU usage unchanged: 744% peak against 741%, wall 118 s
//! against 115 s (llama-3.2-3b Q4_K_M, ~700-token prompt, `contribution =
//! "maximum"`, one binary, `SWARMLLM_THERMAL_FORCE` flipping the arm). The pool
//! was built and installed — its `swarm-cool-*` threads were visible in
//! `/proc/<pid>/task` — and the work still ran ~8 threads wide. Shipping it
//! would have claimed a protection that does not exist, which is worse than
//! shipping nothing. The unresolved question and the numbers are in
//! `docs/FUTURE_WORK.md` § "Thermal throttling had no measurable effect".
//!
//! So the CPU's own throttle remains the real protection. This exists so that a
//! laptop sitting at 88 °C is at least SAID so, which is what nothing did
//! before — the reporter found out by watching `k10temp` themselves.
//!
//! # Verification status
//!
//! The decision rule is unit-tested, and the warning path was exercised
//! end-to-end via `SWARMLLM_THERMAL_FORCE`. **The sensor read itself could not
//! be tested on the development box** — WSL2 exposes no CPU temperature (only
//! `AC1`/`BAT1` under hwmon), so `read_package_temp_c` returns `None` there and
//! the feature stays dormant. That is the safe direction, and it is why an
//! unreadable sensor is treated as "not hot" rather than as a reason to act.
//! A machine with a real sensor (the reporter's `k10temp`) is where the
//! threshold numbers still need confirming.

use std::sync::atomic::{AtomicU32, Ordering};

/// Warn at or above this package temperature.
///
/// 85 °C sits below the ~95-100 °C `TJ_MAX` at which consumer parts throttle
/// themselves, so this engages before the hardware has to, and above the
/// 70-80 °C a busy laptop reaches legitimately — the reporter's idle baseline
/// under load was 71 °C, which must not trip it.
pub const WARN_ON_C: f32 = 85.0;

/// Consider it recovered once back under this. The gap is hysteresis: without
/// it a machine sitting exactly at the threshold logs a transition every tick.
pub const WARN_OFF_C: f32 = 78.0;

const _: () = assert!(WARN_OFF_C < WARN_ON_C);

/// Last observed package temperature in milli-°C, or `u32::MAX` for "unknown".
/// Surfaced in the diagnostics report so the number behind a warning is
/// visible to whoever is reading it.
static LAST_TEMP_MC: AtomicU32 = AtomicU32::new(u32::MAX);

/// The hot/not-hot decision, given the current temperature and current state.
///
/// Pure so the policy is testable without owning a hot CPU. `None` for the
/// temperature means the sensor could not be read, which must NOT read as hot:
/// most machines this runs on expose nothing readable, and warning every one of
/// them because of a missing file would be a far worse bug than the one this
/// reports.
pub fn is_hot(temp_c: Option<f32>, currently_hot: bool) -> bool {
    match temp_c {
        None => false,
        Some(t) if t >= WARN_ON_C => true,
        Some(t) if t <= WARN_OFF_C => false,
        // Between the two thresholds: hold whatever we were doing.
        Some(_) => currently_hot,
    }
}

/// Is this sensor the CPU, as opposed to some other thing on the board?
///
/// **The whole defect this replaced was the absence of this question**
/// (gotcha #480). `sysinfo` composes a Linux label as `"{chip} {label}"` when
/// hwmon supplies one and `"{chip} temp{N}"` when it does not, so the chip name
/// is always present and is what identifies the sensor.
///
/// Matching is on the chip names that mean "this is the processor" plus the
/// labels that mean it on platforms where the chip name does not
/// (`Package id 0`, `Tctl`/`Tdie` on AMD, the SMC's `CPU`-prefixed keys on
/// macOS). Deliberately a list of things we RECOGNISE: anything unrecognised
/// is not read, because the failure this fixes was reading something
/// unrecognised and believing it.
fn sensor_is_cpu_package(label: &str) -> bool {
    let l = label.to_ascii_lowercase();
    const CPU_CHIPS: &[&str] = &[
        "coretemp",    // Intel
        "k10temp",     // AMD
        "zenpower",    // AMD, third-party driver
        "cpu_thermal", // ARM SoCs, Raspberry Pi
        "soc_thermal", // other ARM SoCs
    ];
    if CPU_CHIPS.iter().any(|chip| l.starts_with(chip)) {
        return true;
    }
    // Label-side names, for platforms whose chip name says nothing.
    l.contains("package id") || l.contains("tctl") || l.contains("tdie") || l.contains("cpu")
}

/// The hottest CPU-package reading among `sensors`, in °C, ignoring everything
/// that is not the processor. Pure, so the selection can be tested against
/// sensor sets from real machines without owning them.
fn hottest_cpu_package_c<'a>(sensors: impl Iterator<Item = (&'a str, f32)>) -> Option<f32> {
    sensors
        .filter(|(label, _)| sensor_is_cpu_package(label))
        .map(|(_, t)| t)
        .filter(|t| t.is_finite() && *t > 0.0 && *t < 150.0)
        .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.max(t))))
}

/// Highest CPU-package temperature the OS will tell us about, in °C.
///
/// Returns `None` when nothing readable is exposed — a container, a VM, WSL2,
/// a machine whose sensors need a driver that is not loaded, **or one whose CPU
/// sensor we do not recognise**. `sysinfo` enumerates hwmon on Linux, the SMC
/// on macOS and WMI on Windows.
///
/// **This used to take the maximum across EVERY component**, on the stated
/// grounds that "the hottest sensor is the one that matters and the naming
/// differs per platform". Both halves are wrong and the second explains the
/// first: because the naming differs, the code declined to name anything, and
/// so read every temperature on the board — the NVMe drive, the chipset, the
/// wifi card, and any unpopulated header the Super I/O chip reports garbage
/// for. Measured on a Dell OptiPlex 3090 (gotcha #480): `dell_smm temp8`
/// reported **126000** m°C with nothing wired to it, while `coretemp Package
/// id 0` — the actual processor — read **38 °C**. The node told its owner it
/// was running hot and to consider a smaller model, at 38 degrees.
pub fn read_package_temp_c() -> Option<f32> {
    let components = sysinfo::Components::new_with_refreshed_list();
    let hottest = hottest_cpu_package_c(
        components
            .iter()
            .filter_map(|c| c.temperature().map(|t| (c.label(), t))),
    );
    match hottest {
        Some(t) => {
            LAST_TEMP_MC.store((t * 1000.0) as u32, Ordering::Relaxed);
            Some(t)
        }
        None => {
            LAST_TEMP_MC.store(u32::MAX, Ordering::Relaxed);
            None
        }
    }
}

/// Last temperature observed, for reporting. `None` when never read or unknown.
pub fn last_temp_c() -> Option<f32> {
    match LAST_TEMP_MC.load(Ordering::Relaxed) {
        u32::MAX => None,
        mc => Some(mc as f32 / 1000.0),
    }
}

/// Test/measurement override, mirroring `SWARMLLM_DECODE_THREADS` and
/// `SWARMLLM_FORCE_STANDARD_ATTN`: it makes this exercisable inside ONE binary
/// on a machine with no usable sensor, which is the only way it could be
/// verified at all here — and is what showed the throttle attempt did nothing.
///
/// `1` forces hot, `0` forces not-hot, unset reads the sensor.
pub fn forced_state() -> Option<bool> {
    match std::env::var("SWARMLLM_THERMAL_FORCE").ok()?.trim() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// One poll: read, decide, report. Returns the new hot state.
///
/// Call from a timer. Cheap enough for that — a hwmon read is a handful of
/// small file reads — and deliberately not on any per-request path.
pub fn poll_and_report() -> bool {
    let currently = super::cpu_pools::machine_is_hot();
    let hot = match forced_state() {
        Some(forced) => forced,
        None => is_hot(read_package_temp_c(), currently),
    };
    if super::cpu_pools::set_machine_is_hot(hot) {
        if hot {
            tracing::warn!(
                temp_c = last_temp_c(),
                threshold_c = WARN_ON_C,
                "This machine is running hot while doing inference on its processor. Nothing has \
                 been changed automatically — if this continues, consider using a smaller model, \
                 lowering the contribution level, or checking why the model is not on the GPU."
            );
        } else {
            tracing::info!(
                temp_c = last_temp_c(),
                "Processor temperature back to normal"
            );
        }
    }
    hot
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unreadable sensor must never read as hot. Most machines expose
    /// nothing useful, and the development box for this feature is one of them
    /// — so getting this backwards would warn every such node forever.
    #[test]
    fn an_unreadable_sensor_never_reads_as_hot() {
        assert!(!is_hot(None, false));
        assert!(
            !is_hot(None, true),
            "and it clears an existing hot state rather than latching it"
        );
    }

    /// The reporter's figures: 71 °C under load is normal and must not trip it;
    /// 88 °C is what they killed the process at and must.
    #[test]
    fn the_reported_temperatures_land_on_the_right_side() {
        assert!(!is_hot(Some(71.0), false), "busy but fine");
        assert!(is_hot(Some(88.0), false), "what they saw at 5 min");
    }

    /// Hysteresis: between the thresholds the state is held, so a machine
    /// hovering at the trip point does not log a transition every tick.
    #[test]
    fn the_band_between_thresholds_holds_state() {
        let mid = (WARN_ON_C + WARN_OFF_C) / 2.0;
        assert!(is_hot(Some(mid), true), "stays hot while cooling");
        assert!(!is_hot(Some(mid), false), "stays cool while heating");
    }

    /// And it does eventually clear, or a single spike would leave the node
    /// reporting itself hot for the rest of its life.
    #[test]
    fn it_clears_once_actually_cool() {
        assert!(!is_hot(Some(WARN_OFF_C - 1.0), true));
    }
}

#[cfg(test)]
mod cpu_sensor_tests {
    use super::*;

    /// The sensor set read off the Dell OptiPlex 3090 that produced gotcha
    /// #480, verbatim. `dell_smm temp8` is an unpopulated header reporting the
    /// classic ~127 °C garbage; the processor is at 38.
    fn optiplex_3090() -> Vec<(&'static str, f32)> {
        vec![
            ("acpitz temp1", 27.8),
            ("nvme Composite", 32.85),
            ("nvme Sensor 1", 32.85),
            ("pch_cometlake temp1", 33.0),
            ("dell_smm temp1", 37.0),
            ("dell_smm temp7", 15.0),
            ("dell_smm temp8", 126.0),
            ("coretemp Package id 0", 38.0),
            ("coretemp Core 0", 36.0),
            ("coretemp Core 5", 37.0),
            ("iwlwifi_1 temp1", 40.0),
        ]
    }

    #[test]
    fn a_board_sensor_is_not_mistaken_for_the_processor() {
        let read = hottest_cpu_package_c(optiplex_3090().into_iter());
        assert_eq!(
            read,
            Some(38.0),
            "the processor is at 38 °C; 126 °C is an unpopulated dell_smm header"
        );
        assert!(
            !is_hot(read, false),
            "a machine at 38 °C must not be told it is running hot"
        );
    }

    /// The control: without the filter this set really does read as hot, so the
    /// test above is detecting the fix rather than passing vacuously.
    #[test]
    fn the_old_unfiltered_maximum_would_have_warned() {
        let unfiltered = optiplex_3090()
            .into_iter()
            .map(|(_, t)| t)
            .filter(|t| t.is_finite() && *t > 0.0 && *t < 150.0)
            .fold(f32::MIN, f32::max);
        assert_eq!(unfiltered, 126.0);
        assert!(
            is_hot(Some(unfiltered), false),
            "the old rule warned at 126 °C — this is the behaviour being removed"
        );
    }

    #[test]
    fn a_real_hot_processor_is_still_reported() {
        let hot = vec![("coretemp Package id 0", 91.0), ("nvme Composite", 40.0)];
        assert_eq!(hottest_cpu_package_c(hot.into_iter()), Some(91.0));
    }

    #[test]
    fn the_amd_and_arm_spellings_are_recognised() {
        for label in [
            "k10temp Tctl",
            "k10temp Tdie",
            "zenpower Tdie",
            "cpu_thermal temp1",
            "soc_thermal temp1",
            "CPU proximity", // macOS SMC
        ] {
            assert!(
                sensor_is_cpu_package(label),
                "{label} names a processor sensor"
            );
        }
    }

    #[test]
    fn nothing_on_the_board_is_taken_for_the_processor() {
        for label in [
            "dell_smm temp8",
            "nvme Composite",
            "pch_cometlake temp1",
            "acpitz temp1",
            "iwlwifi_1 temp1",
            "amdgpu edge",
        ] {
            assert!(
                !sensor_is_cpu_package(label),
                "{label} is not the processor"
            );
        }
    }

    /// A machine exposing nothing we recognise stays dormant, which is the
    /// direction this module has always chosen for an unreadable sensor.
    #[test]
    fn an_unrecognised_machine_reports_nothing_rather_than_guessing() {
        let unknown = vec![("acpitz temp1", 55.0), ("nvme Composite", 40.0)];
        assert_eq!(hottest_cpu_package_c(unknown.into_iter()), None);
        assert!(!is_hot(None, false));
    }
}
