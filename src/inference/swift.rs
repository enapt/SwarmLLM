//! SWIFT (arxiv 2410.06916) — On-the-Fly Self-Speculative Decoding.
//!
//! The target LLM acts as its own draft by skipping a contiguous range of
//! intermediate layers during the proposal phase. Verification still uses the
//! full target. No external draft weights are required, which makes SWIFT a
//! plug-and-play decode speedup for any loaded model.
//!
//! v2 calibration: instead of a single fixed skip pattern, we generate a
//! handful of candidate patterns (varying the start position of the skip
//! block while keeping width = `skip_ratio * num_layers`) and round-robin
//! through them during the calibration window. After the window closes the
//! candidate with the highest empirical acceptance rate is selected and used
//! for the rest of the request. Falls back to the v1 fixed pattern when only
//! one candidate is feasible (very small models or extreme skip ratios).

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Per-request SWIFT runtime configuration.
#[derive(Debug, Clone, Copy)]
pub struct SwiftConfig {
    pub enabled: bool,
    /// Number of warmup rounds during which calibration rotates candidate
    /// skip patterns. After this many rounds the best-accepting pattern is
    /// pinned.
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

/// Maximum number of candidate skip patterns generated per session.
const MAX_CANDIDATES: usize = 5;

/// Build the v1 single-pattern skip mask: a contiguous middle block of
/// layers is skipped, with the first two and last two layers always
/// preserved (per the SWIFT paper, outer layers carry the most distribution-
/// shaping signal). Kept for tests and as a fallback when only one candidate
/// is feasible.
pub fn build_skip_mask(num_layers: usize, skip_ratio: f32) -> Vec<bool> {
    let mut mask = vec![false; num_layers];
    if num_layers < 5 || skip_ratio <= 0.0 {
        return mask;
    }
    let ratio = skip_ratio.clamp(0.0, 0.95) as f64;
    let target_skip = (num_layers as f64 * ratio).round() as usize;
    let max_skip = num_layers.saturating_sub(4);
    let skip = target_skip.min(max_skip);
    if skip == 0 {
        return mask;
    }
    let start = 2 + (num_layers - 4 - skip) / 2;
    let end = (start + skip).min(num_layers.saturating_sub(2));
    for slot in mask.iter_mut().take(end).skip(start) {
        *slot = true;
    }
    mask
}

/// Generate up to `MAX_CANDIDATES` candidate skip patterns for SWIFT v2
/// calibration. Each candidate is a contiguous skip block of width
/// `target_skip = round(num_layers * skip_ratio)`, varying only the start
/// position. The outer two layers on each side are always preserved.
pub fn generate_candidates(num_layers: usize, skip_ratio: f32) -> Vec<Vec<bool>> {
    if num_layers < 5 || skip_ratio <= 0.0 {
        return vec![vec![false; num_layers]];
    }
    let ratio = skip_ratio.clamp(0.0, 0.95) as f64;
    let target_skip = (num_layers as f64 * ratio).round() as usize;
    let max_skip = num_layers.saturating_sub(4);
    let skip = target_skip.min(max_skip);
    if skip == 0 {
        return vec![vec![false; num_layers]];
    }
    // Possible start positions: 2..=(num_layers - 2 - skip), inclusive count.
    let max_start = num_layers - 2 - skip;
    if max_start < 2 {
        // Only one feasible position — fall back to v1.
        return vec![build_skip_mask(num_layers, skip_ratio)];
    }
    let inner_count = max_start - 2 + 1;
    let n = MAX_CANDIDATES.min(inner_count);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Even spread across the feasible range.
        let start = if n == 1 {
            2 + (inner_count - 1) / 2
        } else {
            2 + (i * (inner_count - 1)) / (n - 1)
        };
        let mut mask = vec![false; num_layers];
        for slot in mask.iter_mut().take(start + skip).skip(start) {
            *slot = true;
        }
        out.push(mask);
    }
    out
}

/// SWIFT v2 calibrator: rotates candidate skip patterns during the warmup
/// window, tracks per-candidate acceptance, then pins the best-accepting
/// candidate for the remainder of the request.
///
/// Designed for single-request use (one calibrator per `swift_decode_loop`
/// invocation). All counters are atomic so `record()` could later be called
/// from multiple worker threads without locking, but for v1 we run rounds
/// serially.
pub struct SwiftCalibrator {
    candidates: Vec<Vec<bool>>,
    accept_per_candidate: Vec<AtomicU32>,
    propose_per_candidate: Vec<AtomicU32>,
    /// Round counter (0-indexed). Increments on every `record()` call.
    rounds: AtomicU32,
    /// After this many rounds, switch from rotation to the pinned best.
    calibration_target: u32,
    /// Index of the chosen candidate after calibration. `usize::MAX` means
    /// not yet selected.
    selected: AtomicUsize,
}

impl SwiftCalibrator {
    pub fn new(num_layers: usize, skip_ratio: f32, calibration_target: u32) -> Self {
        let candidates = generate_candidates(num_layers, skip_ratio);
        let n = candidates.len();
        let mut accept = Vec::with_capacity(n);
        let mut propose = Vec::with_capacity(n);
        for _ in 0..n {
            accept.push(AtomicU32::new(0));
            propose.push(AtomicU32::new(0));
        }
        // When only one candidate exists, no calibration is possible —
        // pin it immediately.
        let selected = if n <= 1 {
            AtomicUsize::new(0)
        } else {
            AtomicUsize::new(usize::MAX)
        };
        Self {
            candidates,
            accept_per_candidate: accept,
            propose_per_candidate: propose,
            rounds: AtomicU32::new(0),
            calibration_target,
            selected,
        }
    }

    /// Pick the candidate index for the next round. During calibration
    /// rotates round-robin; after calibration returns the pinned best.
    pub fn next_candidate(&self) -> usize {
        let chosen = self.selected.load(Ordering::Relaxed);
        if chosen != usize::MAX {
            return chosen;
        }
        let r = self.rounds.load(Ordering::Relaxed) as usize;
        r % self.candidates.len()
    }

    /// Borrow the skip mask for a candidate index.
    pub fn pattern(&self, idx: usize) -> &[bool] {
        &self.candidates[idx.min(self.candidates.len() - 1)]
    }

    /// Record one round's outcome. When the calibration window closes,
    /// picks and pins the best-accepting candidate.
    pub fn record(&self, candidate_idx: usize, proposed: u32, accepted: u32) {
        if candidate_idx < self.candidates.len() {
            self.propose_per_candidate[candidate_idx].fetch_add(proposed, Ordering::Relaxed);
            self.accept_per_candidate[candidate_idx].fetch_add(accepted, Ordering::Relaxed);
        }
        let r = self.rounds.fetch_add(1, Ordering::Relaxed) + 1;
        // Once calibration window closes, pick the best (only if not already
        // pinned, which can happen for single-candidate calibrators).
        if r == self.calibration_target && self.selected.load(Ordering::Relaxed) == usize::MAX {
            let best = self.best_candidate();
            self.selected.store(best, Ordering::Relaxed);
            tracing::info!(
                best_idx = best,
                accept_rate = self.candidate_accept_rate(best),
                num_candidates = self.candidates.len(),
                "DIAG: SWIFT calibration complete — pinned best skip pattern"
            );
        }
    }

    /// Pick the candidate with the highest accept rate so far. Ties broken
    /// by lowest index (so we prefer earlier-in-rotation candidates).
    fn best_candidate(&self) -> usize {
        let mut best_idx = 0usize;
        let mut best_score: f32 = -1.0;
        for (i, _) in self.candidates.iter().enumerate() {
            let p = self.propose_per_candidate[i].load(Ordering::Relaxed);
            if p == 0 {
                continue;
            }
            let a = self.accept_per_candidate[i].load(Ordering::Relaxed) as f32;
            let score = a / p as f32;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    fn candidate_accept_rate(&self, idx: usize) -> f32 {
        let p = self.propose_per_candidate[idx].load(Ordering::Relaxed);
        if p == 0 {
            return 0.0;
        }
        self.accept_per_candidate[idx].load(Ordering::Relaxed) as f32 / p as f32
    }

    /// Aggregate accept rate across all candidates (for end-of-request log).
    pub fn acceptance_rate(&self) -> f32 {
        let total_p: u32 = self
            .propose_per_candidate
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        if total_p == 0 {
            return 0.0;
        }
        let total_a: u32 = self
            .accept_per_candidate
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        total_a as f32 / total_p as f32
    }

    pub fn rounds(&self) -> u32 {
        self.rounds.load(Ordering::Relaxed)
    }

    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    pub fn selected_candidate(&self) -> Option<usize> {
        let s = self.selected.load(Ordering::Relaxed);
        if s == usize::MAX {
            None
        } else {
            Some(s)
        }
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
    fn candidates_spread_across_feasible_range() {
        let cands = generate_candidates(32, 0.45);
        assert!(!cands.is_empty() && cands.len() <= MAX_CANDIDATES);
        // Each candidate should skip the SAME number of layers
        let widths: Vec<usize> = cands
            .iter()
            .map(|c| c.iter().filter(|&&b| b).count())
            .collect();
        let first_width = widths[0];
        for w in &widths {
            assert_eq!(
                *w, first_width,
                "all candidates should have same skip width"
            );
        }
        // First candidate starts earlier than last
        let first_start = cands.first().unwrap().iter().position(|&b| b).unwrap();
        let last_start = cands.last().unwrap().iter().position(|&b| b).unwrap();
        assert!(
            last_start > first_start,
            "candidates should vary start position"
        );
    }

    #[test]
    fn candidates_outer_layers_always_safe() {
        let cands = generate_candidates(22, 0.45);
        for (i, c) in cands.iter().enumerate() {
            assert!(!c[0] && !c[1], "cand {i}: outer-low must be preserved");
            assert!(!c[20] && !c[21], "cand {i}: outer-high must be preserved");
        }
    }

    #[test]
    fn calibrator_rotates_then_pins_best() {
        let cal = SwiftCalibrator::new(22, 0.45, 10);
        let n = cal.num_candidates();
        assert!(n > 1, "test assumes multiple candidates available");
        // Verify that during the warmup window selected_candidate() is None.
        let mut last_idx = usize::MAX;
        for _ in 0..n {
            let idx = cal.next_candidate();
            // round-robin should never repeat within one cycle
            assert_ne!(idx, last_idx);
            last_idx = idx;
            // Make candidate 0 win by giving it perfect acceptance.
            let accepted = if idx == 0 { 4 } else { 1 };
            cal.record(idx, 4, accepted);
        }
        assert!(cal.selected_candidate().is_none(), "still calibrating");
        // Burn through the rest of the calibration window.
        while cal.rounds() < 10 {
            let idx = cal.next_candidate();
            let accepted = if idx == 0 { 4 } else { 1 };
            cal.record(idx, 4, accepted);
        }
        // After calibration, candidate 0 should be pinned and stay pinned.
        assert_eq!(cal.selected_candidate(), Some(0));
        for _ in 0..5 {
            assert_eq!(cal.next_candidate(), 0);
        }
    }

    #[test]
    fn calibrator_single_candidate_pinned_immediately() {
        // Tiny model collapses to one candidate.
        let cal = SwiftCalibrator::new(5, 0.45, 32);
        assert_eq!(cal.num_candidates(), 1);
        assert_eq!(cal.selected_candidate(), Some(0));
    }

    #[test]
    fn calibrator_aggregate_rate() {
        let cal = SwiftCalibrator::new(22, 0.45, 32);
        cal.record(0, 4, 3);
        cal.record(1, 4, 2);
        let total = cal.acceptance_rate();
        assert!((total - 5.0 / 8.0).abs() < f32::EPSILON);
    }
}
