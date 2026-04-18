//! SWIFT (arxiv 2410.06916) — On-the-Fly Self-Speculative Decoding.
//!
//! The target LLM acts as its own draft by skipping a contiguous range of
//! intermediate layers during the proposal phase. Verification still uses the
//! full target. No external draft weights are required, which makes SWIFT a
//! plug-and-play decode speedup for any loaded model.
//!
//! v1 here uses a fixed contiguous skip pattern centered in the middle of
//! the layer stack. The first and last two layers are always preserved
//! because outer layers carry the most distribution-shaping signal. A
//! follow-up version will add online calibration that picks the best skip
//! window during a warmup phase.

use std::sync::atomic::{AtomicU32, Ordering};

/// Per-request SWIFT runtime configuration.
#[derive(Debug, Clone, Copy)]
pub struct SwiftConfig {
    pub enabled: bool,
    /// Number of warmup tokens during which calibration runs. v1: still full
    /// forwards; v2 will use this to pick the best skip pattern.
    pub calibration_tokens: u32,
    /// Number of draft tokens proposed per verification round.
    pub gamma: u32,
    /// Fraction of layers to skip in the draft pass.
    pub skip_ratio: f32,
}

impl Default for SwiftConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            calibration_tokens: 32,
            gamma: 4,
            skip_ratio: 0.45,
        }
    }
}

/// Build a fixed skip mask: a contiguous middle block of layers is skipped.
/// First two and last two layers are always preserved (outer layers are most
/// sensitive to perturbation per the SWIFT paper).
///
/// `num_layers`: total transformer layer count (absolute layer indices
/// `0..num_layers`). `skip_ratio` is clamped to `[0.0, 0.95]`.
pub fn build_skip_mask(num_layers: usize, skip_ratio: f32) -> Vec<bool> {
    let mut mask = vec![false; num_layers];
    if num_layers < 5 || skip_ratio <= 0.0 {
        return mask;
    }
    let ratio = skip_ratio.clamp(0.0, 0.95) as f64;
    let target_skip = (num_layers as f64 * ratio).round() as usize;
    // Reserve at least the outer 2 layers on each side.
    let max_skip = num_layers.saturating_sub(4);
    let skip = target_skip.min(max_skip);
    if skip == 0 {
        return mask;
    }
    // Center the skip block.
    let start = 2 + (num_layers - 4 - skip) / 2;
    let end = (start + skip).min(num_layers.saturating_sub(2));
    for slot in mask.iter_mut().take(end).skip(start) {
        *slot = true;
    }
    mask
}

/// Tracks acceptance statistics for a SWIFT session. v1 uses a single fixed
/// pattern so the calibrator just records aggregate accept/total counts for
/// observability. v2 will rotate candidate patterns and pick the best.
#[derive(Debug, Default)]
pub struct SwiftCalibrator {
    pub total_proposed: AtomicU32,
    pub total_accepted: AtomicU32,
    pub rounds: AtomicU32,
}

impl SwiftCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, proposed: u32, accepted: u32) {
        self.total_proposed.fetch_add(proposed, Ordering::Relaxed);
        self.total_accepted.fetch_add(accepted, Ordering::Relaxed);
        self.rounds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn acceptance_rate(&self) -> f32 {
        let total = self.total_proposed.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_accepted.load(Ordering::Relaxed) as f32 / total as f32
    }

    pub fn rounds(&self) -> u32 {
        self.rounds.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_mask_zero_ratio_no_skip() {
        let mask = build_skip_mask(22, 0.0);
        assert!(mask.iter().all(|&b| !b));
    }

    #[test]
    fn skip_mask_preserves_outer_layers() {
        let mask = build_skip_mask(22, 0.5);
        assert!(!mask[0] && !mask[1], "first two layers must be preserved");
        assert!(!mask[20] && !mask[21], "last two layers must be preserved");
        let skipped: usize = mask.iter().filter(|&&b| b).count();
        assert!(skipped > 0 && skipped <= 18);
    }

    #[test]
    fn skip_mask_contiguous_middle() {
        let mask = build_skip_mask(32, 0.45);
        // Find first true and last true — all in between should be true.
        let first = mask.iter().position(|&b| b).unwrap();
        let last = mask.iter().rposition(|&b| b).unwrap();
        for (i, slot) in mask.iter().enumerate().take(last + 1).skip(first) {
            assert!(*slot, "skip block must be contiguous, gap at {i}");
        }
        assert!(first >= 2);
        assert!(last < 30);
    }

    #[test]
    fn skip_mask_tiny_model_falls_back_to_no_skip() {
        let mask = build_skip_mask(4, 0.5);
        assert!(mask.iter().all(|&b| !b));
    }

    #[test]
    fn calibrator_records_acceptance() {
        let cal = SwiftCalibrator::new();
        cal.record(4, 3);
        cal.record(4, 2);
        assert_eq!(cal.rounds(), 2);
        assert!((cal.acceptance_rate() - 5.0 / 8.0).abs() < f32::EPSILON);
    }
}
