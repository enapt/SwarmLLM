//! Can this build's CUDA kernels actually run on the GPU that is present?
//!
//! **Why this exists.** `Device::cuda_if_available(0)` answers "is there a
//! CUDA device", which is not the same question. Creating a context succeeds on
//! any NVIDIA card with a driver; the architecture mismatch only surfaces later,
//! when the driver tries to load a module whose PTX targets an arch newer than
//! the device. So a card below this build's floor would start the daemon
//! cleanly, log `GPU detected`, advertise itself to the swarm as a GPU node, and
//! then fail *every* inference request with
//! `DriverError(CUDA_ERROR_NO_BINARY_FOR_GPU)` — a message that tells a
//! non-technical owner nothing about what to do.
//!
//! That became reachable when flash-attention-2 was re-enabled in the `cuda`
//! feature: every `candle-flash-attn` kernel source is `_sm80` and uses Ampere's
//! async-copy instructions, so the CUDA builds compile at compute capability 8.0
//! and pre-Ampere cards (GTX 16-series, RTX 20-series) are outside it.
//!
//! The floor is a property of the BUILD, not of the machine, so it is a
//! compile-time constant here and must be kept equal to `CUDA_COMPUTE_CAP` in
//! `.github/workflows/release.yml`. `compute_cap_matches_release_workflow` in
//! `tests/repo_consistency.rs` fails the build if they drift.

/// Minimum CUDA compute capability this build's kernels are compiled for.
///
/// Must equal `CUDA_COMPUTE_CAP` in `.github/workflows/release.yml`, expressed
/// as (major, minor) — the workflow writes it as the two digits concatenated,
/// so `80` here is `(8, 0)`.
pub const MIN_COMPUTE_CAP: (u32, u32) = (8, 0);

/// Is a card with this compute capability able to run our kernels?
///
/// Forward compatibility is real: PTX compiled for `compute_80` is JIT-compiled
/// by the driver onto any newer architecture, so Ada, Hopper and Blackwell all
/// pass. Only *older* cards fail, and they fail hard rather than degrading.
pub fn compute_cap_supported(cap: (u32, u32)) -> bool {
    cap >= MIN_COMPUTE_CAP
}

/// Parse the `major.minor` string `nvidia-smi --query-gpu=compute_cap` prints.
///
/// Returns `None` for anything unexpected, which every caller must treat as
/// "unknown", never as "unsupported" — refusing the GPU because a subprocess
/// printed something surprising would break working cards, which is a worse
/// failure than the one this module prevents.
pub fn parse_compute_cap(s: &str) -> Option<(u32, u32)> {
    let (major, minor) = s.trim().split_once('.')?;
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

/// The message shown to someone whose card this build has left behind.
///
/// Names the card, the requirement, and what actually happens next — the node
/// keeps working on the CPU. Written for someone who does not know what a
/// compute capability is, which is why it translates the number into a card
/// generation they can check against the box.
///
/// It deliberately does NOT point at "a build for older cards": no pre-Ampere
/// CUDA asset is published (see docs/FUTURE_WORK.md). Telling someone to go and
/// find a download that does not exist is worse than telling them nothing, and
/// this text is the only thing most people will ever read about it.
pub fn unsupported_gpu_message(gpu_name: &str, cap: (u32, u32)) -> String {
    format!(
        "{} is too old for GPU acceleration in this version (your card is NVIDIA compute \
         capability {}.{}; {}.{} or newer is needed, which means an RTX 30-series or newer). \
         Running on the processor instead — everything still works, just slower. \
         Nothing to change; this message is for information only.",
        gpu_name, cap.0, cap.1, MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1
    )
}

/// Is the local GPU usable by this build's kernels? Probed once, then cached.
///
/// Returns `true` when there is no GPU at all, or when the capability could not
/// be read — this answers "should we STOP using the GPU", and the only `false`
/// is a card we positively know is too old. Callers that have no GPU are
/// already on the CPU for other reasons.
///
/// Cached because it shells out to nvidia-smi and the answer cannot change
/// while the process runs: it is a property of the card and of this binary.
#[cfg(feature = "candle-cuda")]
pub fn local_gpu_is_supported() -> bool {
    use std::sync::OnceLock;
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(
        || match crate::model::auto_manage::vram::detect_gpu_compute_cap() {
            Some(cap) if !compute_cap_supported(cap) => {
                let (name, _) = crate::model::auto_manage::vram::detect_gpu_nvidia_smi();
                let name = name.unwrap_or_else(|| "This NVIDIA GPU".to_string());
                tracing::warn!(
                    gpu = %name,
                    compute_cap = format!("{}.{}", cap.0, cap.1),
                    required = format!("{}.{}", MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1),
                    "{}",
                    unsupported_gpu_message(&name, cap)
                );
                false
            }
            _ => true,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ampere_and_newer_are_supported() {
        assert!(compute_cap_supported((8, 0)), "A100 / the floor itself");
        assert!(compute_cap_supported((8, 6)), "RTX 3070 — the dev machine");
        assert!(compute_cap_supported((8, 9)), "RTX 4090 (Ada)");
        assert!(compute_cap_supported((9, 0)), "H100 (Hopper)");
        assert!(compute_cap_supported((12, 0)), "RTX 50-series (Blackwell)");
    }

    #[test]
    fn pre_ampere_is_not_supported() {
        assert!(
            !compute_cap_supported((7, 5)),
            "RTX 2080 / GTX 1660 (Turing)"
        );
        assert!(!compute_cap_supported((7, 0)), "V100 (Volta)");
        assert!(!compute_cap_supported((6, 1)), "GTX 1080 (Pascal)");
        assert!(!compute_cap_supported((5, 2)), "GTX 970 (Maxwell)");
    }

    #[test]
    fn minor_version_is_compared_within_a_major() {
        // The comparison is on the tuple, so this must not degrade to a
        // major-only check: 7.5 is below 8.0 despite being the highest 7.x.
        assert!(!compute_cap_supported((7, 9)));
        assert!(compute_cap_supported((8, 1)));
    }

    #[test]
    fn parses_what_nvidia_smi_prints() {
        assert_eq!(parse_compute_cap("8.6"), Some((8, 6)));
        assert_eq!(parse_compute_cap("7.5"), Some((7, 5)));
        // --format=csv,noheader leaves the trailing newline on.
        assert_eq!(parse_compute_cap("8.9\n"), Some((8, 9)));
        assert_eq!(parse_compute_cap(" 12.0 "), Some((12, 0)));
    }

    #[test]
    fn unparseable_output_is_unknown_not_unsupported() {
        // Every one of these must reach the caller as None so it treats the
        // capability as unknown and leaves the GPU alone. A card that works
        // must never be sent to the CPU because nvidia-smi was odd.
        for junk in [
            "",
            "N/A",
            "8",
            "eight.six",
            "Failed to initialize NVML: Driver/library version mismatch",
        ] {
            assert_eq!(parse_compute_cap(junk), None, "input {junk:?}");
        }
    }

    #[test]
    fn the_message_names_the_card_and_says_what_happens_next() {
        let msg = unsupported_gpu_message("NVIDIA GeForce RTX 2060", (7, 5));
        assert!(msg.contains("RTX 2060"), "must name the card: {msg}");
        assert!(msg.contains("7.5"), "must state what it has: {msg}");
        assert!(msg.contains("8.0"), "must state what is needed: {msg}");
        assert!(
            msg.contains("processor"),
            "must say the node keeps working, not just that it failed: {msg}"
        );
        assert!(
            msg.contains("RTX 30-series"),
            "must translate the number into something checkable against the box: {msg}"
        );
        // No pre-Ampere CUDA asset is published, so this must not send anyone
        // looking for one. Cheap to assert, and the kind of promise that rots
        // silently once someone edits the copy.
        for absent in ["releases page", "download", "older cards"] {
            assert!(
                !msg.contains(absent),
                "must not promise a build that does not exist ({absent:?}): {msg}"
            );
        }
    }
}
