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
//! `docs/FUTURE_WORK_ARCHIVE.md` § "Thermal throttling had no measurable effect".
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

/// The temperature Windows' ACPI thermal zones report, in °C — what the heat
/// guard (`daemon::host_heat`) reads where [`read_package_temp_c`] has nothing.
/// `None` off Windows, and on a Windows machine whose firmware feeds no zone.
///
/// **Why a second reading**: on Windows `sysinfo` asks WMI's
/// `MSAcpi_ThermalZoneTemperature`, which refuses an ordinary user ("Access
/// denied", measured on this laptop 2026-11-11), so the processor read
/// "unknown" on every Windows computer the node does not run as administrator —
/// and the heat guard watched the card alone while the laptop froze at 93-96 °C
/// (gotcha #1081). The same zones are open to any user through the Performance
/// Data Helper: `\Thermal Zone Information(*)\High Precision Temperature`,
/// tenths of a kelvin, read 3492 → 76 °C here as an ordinary user, valid on
/// the first sample, one instance (`\_TZ.THRM`).
///
/// A zone is a sensor the firmware names, never marked as the processor, so
/// this is the heat guard's reading only — the processor warning
/// ([`poll_and_report`]) stays on the package sensor it was tuned for.
///
/// Prior art: windows_exporter's `thermalzone` collector (v0.30.0
/// `thermalzone.go`: `High Precision Temperature / 10 - 273.15`, fallback
/// `Temperature` in whole kelvin) and Netdata's (`Temperature - 273.15`) read
/// these counters; windows_exporter removed it 2026-10-10 (#2659) because many
/// vendors feed no zone — hence `None` there, never "cool". The English path
/// (`PdhAddEnglishCounterW`) because the counter names are localised.
/// LibreHardwareMonitor's package reading needs a ring-0 driver (administrator)
/// and was not taken.
pub fn read_thermal_zone_c() -> Option<f32> {
    #[cfg(windows)]
    {
        let precise =
            windows_zone::values(r"\Thermal Zone Information(*)\High Precision Temperature");
        hottest_zone_c(
            precise
                .into_iter()
                .flatten()
                .map(|decikelvin| decikelvin / 10.0),
        )
        .or_else(|| {
            hottest_zone_c(windows_zone::values(
                r"\Thermal Zone Information(*)\Temperature",
            )?)
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// A zone colder than this or hotter than [`ZONE_MOST_C`] is not believed: a
/// zone the firmware never fills reads 0 K (−273 °C), and nothing still running
/// reads 125 °C.
#[cfg(any(windows, test))]
const ZONE_LEAST_C: f64 = 5.0;
#[cfg(any(windows, test))]
const ZONE_MOST_C: f64 = 125.0;

/// The hottest believable zone, in °C, from readings in kelvin. Pure, so the
/// rule is tested off Windows.
#[cfg(any(windows, test))]
fn hottest_zone_c(kelvin: impl IntoIterator<Item = f64>) -> Option<f32> {
    kelvin
        .into_iter()
        .map(|k| k - 273.15)
        .filter(|c| c.is_finite() && (ZONE_LEAST_C..=ZONE_MOST_C).contains(c))
        .fold(None::<f64>, |hottest, c| {
            Some(hottest.map_or(c, |h| h.max(c)))
        })
        .map(|c| c as f32)
}

/// Windows' thermal zones through the Performance Data Helper (`pdh.dll`), the
/// interface `typeperf` and Performance Monitor read, open to an ordinary user.
#[cfg(windows)]
mod windows_zone {
    use windows_sys::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW, PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W,
        PDH_FMT_DOUBLE, PDH_HQUERY, PDH_MORE_DATA,
    };

    /// pdh.h's `PDH_FMT_NOCAP100`, which windows-sys does not carry: a value
    /// over 100 is not cut to 100 (a zone reads ~3,500 tenths of a kelvin).
    const PDH_FMT_NOCAP100: u32 = 0x0000_8000;

    /// The query, closed however the reading ends.
    struct Query(PDH_HQUERY);

    impl Drop for Query {
        fn drop(&mut self) {
            // SAFETY: the handle came from a successful `PdhOpenQueryW`, and
            // this is the only place it is closed.
            unsafe { PdhCloseQuery(self.0) };
        }
    }

    /// Every valid instance of the English counter `path`, in the counter's
    /// own unit; `None` on any PDH failure (no such counter, no instance, a
    /// damaged performance registry).
    pub(super) fn values(path: &str) -> Option<Vec<f64>> {
        let path: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
        let mut handle: PDH_HQUERY = std::ptr::null_mut();
        // SAFETY: a null data source is this machine's live data; `handle` is
        // written only on success.
        if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut handle) } != 0 {
            return None;
        }
        let query = Query(handle);
        let mut counter = std::ptr::null_mut();
        // SAFETY: `path` is NUL-terminated and outlives the call; the query is open.
        if unsafe { PdhAddEnglishCounterW(query.0, path.as_ptr(), 0, &mut counter) } != 0 {
            return None;
        }
        // SAFETY: the query is open and holds the counter. A temperature is an
        // instantaneous counter, so one collection gives a value.
        if unsafe { PdhCollectQueryData(query.0) } != 0 {
            return None;
        }
        let format = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;
        let (mut size, mut count) = (0u32, 0u32);
        // SAFETY: a null buffer with a size of 0 asks only for the size needed.
        let asked = unsafe {
            PdhGetFormattedCounterArrayW(
                counter,
                format,
                &mut size,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if asked != PDH_MORE_DATA || size == 0 {
            return None;
        }
        // The items and their names share one buffer; `u64`s keep it aligned
        // for the `f64` inside each item.
        let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
        let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
        // SAFETY: `buffer` holds at least `size` bytes, aligned for an item.
        if unsafe { PdhGetFormattedCounterArrayW(counter, format, &mut size, &mut count, items) }
            != 0
        {
            return None;
        }
        let bytes =
            (count as usize).checked_mul(std::mem::size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>())?;
        if bytes > buffer.len() * 8 {
            return None;
        }
        // SAFETY: PDH wrote `count` items at the start of `buffer`, which
        // holds them (checked above) and outlives the slice.
        let items = unsafe { std::slice::from_raw_parts(items, count as usize) };
        Some(
            items
                .iter()
                .filter(|item| {
                    matches!(
                        item.FmtValue.CStatus,
                        PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA
                    )
                })
                // SAFETY: `PDH_FMT_DOUBLE` fills the union's `doubleValue`.
                .map(|item| unsafe { item.FmtValue.Anonymous.doubleValue })
                .collect(),
        )
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

    /// This laptop's zone as an ordinary user read it (2026-11-11): 3492
    /// tenths of a kelvin. The reading is in kelvin by the time it gets here.
    #[test]
    fn a_zone_reading_is_kelvin_less_273() {
        let c = hottest_zone_c([3492.0 / 10.0]).expect("a real zone is believed");
        assert!((c - 76.05).abs() < 0.01, "{c}");
    }

    /// Several zones, no flag saying which is the processor: the hottest.
    #[test]
    fn the_hottest_zone_is_taken() {
        assert_eq!(hottest_zone_c([310.15, 366.15, 330.15]), Some(93.0));
    }

    /// A zone the firmware never fills reads 0 K; one out of reason must not
    /// trip the guard or hide a real zone beside it.
    #[test]
    fn an_unfilled_or_impossible_zone_is_not_believed() {
        assert_eq!(hottest_zone_c([0.0]), None);
        assert_eq!(hottest_zone_c([273.15 + 200.0]), None);
        assert_eq!(hottest_zone_c([f64::NAN, 0.0, 340.15]), Some(67.0));
    }

    /// No zone at all (a desktop or a VM) is "unknown", never "cool".
    #[test]
    fn no_zone_is_no_reading() {
        assert_eq!(hottest_zone_c(std::iter::empty()), None);
    }

    /// Off Windows there is nothing to read; the guard keeps its package sensor.
    #[cfg(not(windows))]
    #[test]
    fn off_windows_the_zone_reads_nothing() {
        assert_eq!(read_thermal_zone_c(), None);
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
