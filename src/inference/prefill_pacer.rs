//! Adaptive sizing for the chunked-prefill quantum.
//!
//! Chunked prefill exists so a long admission cannot stall the decode slots
//! already running: each tick advances every `Prefilling` slot by at most one
//! chunk before the batched decode runs. `slot_table.rs` states the guarantee
//! as "a long admission can no longer block decode for more than
//! `prefill_chunk_tokens` of compute", and that holds exactly as written.
//!
//! **The trap is the unit.** The bound is expressed in *tokens of compute*, and
//! a token is not a fixed amount of time — 128 prompt tokens is milliseconds on
//! a GPU and 45–59 seconds on a modest CPU (measured 2026-07-28, gotcha #191).
//! On the slow machine the per-tick bound is honoured while a co-scheduled slot
//! decodes **one token per tick** for the whole of the long prefill: 8 tokens in
//! 5.5 minutes beside a 3 968-token prompt, which a client reasonably reads as a
//! hang.
//!
//! So the quantum is sized here from *measured wall time* instead: keep an EWMA
//! of ms-per-prompt-token and pick the token count that lands near
//! `target_ms`. That self-calibrates across a GPU and a CPU node instead of
//! asking an operator to pick a number whose meaning depends on their hardware.
//!
//! Two properties worth not undoing:
//!
//! - **Only paces when it would matter.** With a single slot in the table there
//!   is nobody to starve, so the full configured chunk is used and throughput is
//!   untouched. Smaller chunks mean more `forward` calls over the same total
//!   work, so shrinking unconditionally would slow every solo long prompt to buy
//!   fairness nobody needed.
//! - **A floor, because the trade is bounded.** On a machine slow enough that
//!   even one token overruns `target_ms`, no chunk size meets the target and
//!   driving it to 1 would pay maximum per-call overhead for the last sliver of
//!   responsiveness. `MIN_CHUNK_TOKENS` stops there and the target is simply not
//!   met — the honest outcome, and `PrefillPacer::meeting_target` reports it so
//!   the caller can say so rather than implying otherwise.

use std::time::Duration;

/// Smallest quantum the pacer will choose. Below this, per-call overhead
/// dominates: the same prompt is split into 16x more `forward` invocations for
/// a share of a second's extra responsiveness.
pub const MIN_CHUNK_TOKENS: usize = 8;

/// Weight of each new observation in the ms-per-token EWMA. Deliberately
/// sluggish: prompt-token cost climbs with `index_pos` (attention is quadratic
/// in context), so a single late chunk should nudge the estimate rather than
/// redefine it.
const EWMA_ALPHA: f64 = 0.3;

/// A shrink must bring tick time below this fraction of the previous tick to
/// count as having worked. Above it, the reduction is within noise of "no
/// change" and the machine's cost is dominated by fixed per-call overhead
/// rather than per-token work.
const SHRINK_MUST_BEAT: f64 = 0.9;

/// Chooses the chunked-prefill quantum from measured wall time.
///
/// Construct one per worker (the cost of a prompt token is a property of the
/// machine and the loaded model, both fixed for a worker's lifetime).
#[derive(Debug)]
pub struct PrefillPacer {
    /// Configured ceiling — `inference.prefill_chunk_tokens`. Also the value
    /// used verbatim whenever no other slot could be starved.
    max_tokens: usize,
    /// Wall-time budget for one tick's prefill work while slots are shared.
    target_ms: u64,
    /// EWMA of milliseconds per prompt token; `None` until the first
    /// observation, when `max_tokens` is used as the opening guess.
    ms_per_token: Option<f64>,
    /// Previous observation, as (tokens, elapsed_ms). Used to check whether a
    /// shrink actually bought anything.
    last: Option<(usize, f64)>,
    /// Set once shrinking has been shown NOT to reduce tick time on this
    /// machine. See [`Self::observe`].
    shrink_ineffective: bool,
}

impl PrefillPacer {
    pub fn new(max_tokens: usize, target_ms: u64) -> Self {
        Self {
            max_tokens: max_tokens.max(1),
            target_ms: target_ms.max(1),
            ms_per_token: None,
            last: None,
            shrink_ineffective: false,
        }
    }

    /// The quantum to use for this tick.
    ///
    /// `sharing` is whether any OTHER slot is waiting on this tick — i.e.
    /// whether shrinking the chunk buys anyone anything. Pass the count of
    /// active slots via [`Self::is_sharing`] rather than deriving it here, so
    /// the caller's definition of "active" stays the single one.
    pub fn chunk_size(&self, sharing: bool) -> usize {
        if !sharing || self.shrink_ineffective {
            return self.max_tokens;
        }
        match self.ms_per_token {
            // No measurement yet: start at the ceiling. The first chunk pays the
            // un-paced cost once, which is also what produces the measurement.
            None => self.max_tokens,
            Some(ms) if ms <= 0.0 => self.max_tokens,
            Some(ms) => {
                let budgeted = (self.target_ms as f64 / ms).floor() as usize;
                budgeted.clamp(MIN_CHUNK_TOKENS.min(self.max_tokens), self.max_tokens)
            }
        }
    }

    /// Record how long a chunk of `tokens` prompt tokens actually took.
    ///
    /// Ignores empty or zero-duration observations: a zero would drive
    /// ms-per-token to 0 and re-inflate the quantum to the ceiling on the next
    /// tick, which is the oscillation this is meant to damp.
    pub fn observe(&mut self, tokens: usize, elapsed: Duration) {
        if tokens == 0 {
            return;
        }
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let ms = total_ms / tokens as f64;
        if !ms.is_finite() || ms <= 0.0 {
            return;
        }

        // **Does shrinking actually buy anything on this machine?**
        //
        // The whole model above assumes tick time is proportional to tokens.
        // On a GPU it is not: a forward pass is dominated by fixed per-call
        // cost, so an 8-token chunk costs nearly what a 128-token chunk does.
        // Measured on an RTX 3070 (2026-07-28): 128 tokens/tick ≈ 130ms
        // (1.0 ms/token) but 8 tokens/tick ≈ 790ms (99 ms/token).
        //
        // Left unchecked that is a feedback loop pointing the wrong way —
        // dividing a near-constant tick time by fewer tokens *raises* the
        // apparent ms-per-token, which shrinks the chunk further, which raises
        // it again, pinning the quantum at the floor while making every tick
        // slower than doing nothing. It made the co-scheduled request WORSE,
        // which is the opposite of the point.
        //
        // So: if we shrank and the tick did not get meaningfully cheaper, stop
        // pacing on this worker. The check is one-way and permanent for the
        // worker's life — a machine's cost structure does not change, and
        // re-probing would re-enter the loop it exists to break.
        if let Some((prev_tokens, prev_ms)) = self.last {
            let shrank = tokens < prev_tokens;
            let barely_faster = total_ms > prev_ms * SHRINK_MUST_BEAT;
            if shrank && barely_faster {
                self.shrink_ineffective = true;
            }
        }
        self.last = Some((tokens, total_ms));

        self.ms_per_token = Some(match self.ms_per_token {
            None => ms,
            Some(prev) => prev * (1.0 - EWMA_ALPHA) + ms * EWMA_ALPHA,
        });
    }

    /// Whether the chosen quantum can actually meet `target_ms`, i.e. the floor
    /// is not binding. False means this machine is slow enough that a single
    /// prompt token overruns the per-tick budget.
    pub fn meeting_target(&self) -> bool {
        match self.ms_per_token {
            None => true,
            Some(ms) => ms * MIN_CHUNK_TOKENS as f64 <= self.target_ms as f64,
        }
    }

    /// Current cost estimate in milliseconds per prompt token; `None` before the
    /// first observation. Used to derive a prefill ETA.
    pub fn ms_per_token(&self) -> Option<f64> {
        self.ms_per_token
    }

    /// Whether this tick has anyone to starve. `active_slots` is the number of
    /// slots in the table, prefilling and decoding alike — a second *prefilling*
    /// slot is starved by a big chunk just as a decoding one is.
    pub fn is_sharing(active_slots: usize) -> bool {
        active_slots > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_request_always_gets_the_full_configured_chunk() {
        let mut p = PrefillPacer::new(128, 150);
        // Even after observing a punishingly slow machine, a solo request is
        // not paced — there is nobody to starve and smaller chunks would only
        // add per-call overhead.
        p.observe(128, Duration::from_secs(50));
        assert_eq!(p.chunk_size(false), 128);
    }

    #[test]
    fn a_slow_machine_shrinks_the_quantum_when_sharing() {
        let mut p = PrefillPacer::new(128, 150);
        // The measured CPU case: 128 tokens in ~46s => ~360ms per token.
        p.observe(128, Duration::from_millis(46_000));
        let c = p.chunk_size(true);
        assert_eq!(
            c, MIN_CHUNK_TOKENS,
            "360ms/token cannot meet a 150ms budget at any size above the floor"
        );
        assert!(
            !p.meeting_target(),
            "the floor is binding here and the pacer should say so"
        );
    }

    #[test]
    fn a_fast_machine_keeps_the_full_chunk_even_when_sharing() {
        let mut p = PrefillPacer::new(128, 150);
        // GPU case: 128 tokens in 4ms => 0.03ms/token, budget allows far more
        // than the ceiling, so the ceiling wins.
        p.observe(128, Duration::from_millis(4));
        assert_eq!(p.chunk_size(true), 128);
        assert!(p.meeting_target());
    }

    #[test]
    fn quantum_lands_near_the_target_on_a_middling_machine() {
        let mut p = PrefillPacer::new(512, 200);
        // 10ms per token => a 200ms budget should buy about 20 tokens.
        p.observe(100, Duration::from_millis(1000));
        assert_eq!(p.chunk_size(true), 20);
        assert!(p.meeting_target());
    }

    #[test]
    fn zero_and_empty_observations_do_not_reset_the_estimate() {
        let mut p = PrefillPacer::new(128, 150);
        p.observe(100, Duration::from_millis(1000)); // 10ms/token
        let before = p.chunk_size(true);
        p.observe(0, Duration::from_millis(5)); // no tokens — ignored
        p.observe(10, Duration::ZERO); // zero duration — ignored
        assert_eq!(
            p.chunk_size(true),
            before,
            "a degenerate observation must not re-inflate the quantum"
        );
    }

    #[test]
    fn the_floor_never_exceeds_a_small_configured_ceiling() {
        // An operator who deliberately sets a tiny chunk must not have it
        // silently raised to the floor.
        let mut p = PrefillPacer::new(4, 150);
        p.observe(4, Duration::from_secs(10));
        assert_eq!(p.chunk_size(true), 4);
    }

    #[test]
    fn estimate_moves_gradually_rather_than_jumping() {
        let mut p = PrefillPacer::new(128, 150);
        p.observe(100, Duration::from_millis(1000)); // 10ms/token
        p.observe(100, Duration::from_millis(2000)); // 20ms/token
        let ms = p.ms_per_token().unwrap();
        assert!(
            ms > 10.0 && ms < 20.0,
            "EWMA should sit between the two observations, got {ms}"
        );
    }

    #[test]
    fn stops_pacing_when_shrinking_does_not_make_ticks_cheaper() {
        // The measured GPU case (RTX 3070, 2026-07-28): tick time barely moves
        // with chunk size because fixed per-call cost dominates. Pacing there
        // is pure loss — 128 tokens in 130ms became 8 tokens in 790ms.
        let mut p = PrefillPacer::new(128, 200);
        p.observe(128, Duration::from_millis(400)); // 3.1ms/token -> wants ~64
        let shrunk = p.chunk_size(true);
        assert!(shrunk < 128, "should try shrinking first, got {shrunk}");

        // The smaller chunk costs essentially the same wall time.
        p.observe(shrunk, Duration::from_millis(390));
        assert_eq!(
            p.chunk_size(true),
            128,
            "shrinking bought nothing, so pacing must switch itself off"
        );
    }

    #[test]
    fn keeps_pacing_when_shrinking_genuinely_helps() {
        // The CPU case: time really is roughly proportional to tokens, so a
        // smaller chunk really is a shorter tick. Pacing must stay on.
        let mut p = PrefillPacer::new(128, 200);
        p.observe(128, Duration::from_millis(46_000));
        let shrunk = p.chunk_size(true);
        assert_eq!(shrunk, MIN_CHUNK_TOKENS);

        // 8 tokens costs ~1/16th of what 128 did — proportional, so it worked.
        p.observe(shrunk, Duration::from_millis(2_875));
        assert_eq!(
            p.chunk_size(true),
            MIN_CHUNK_TOKENS,
            "shrinking worked, so pacing must stay engaged"
        );
    }

    #[test]
    fn a_disabled_pacer_still_reports_full_chunks_when_solo() {
        let mut p = PrefillPacer::new(128, 200);
        p.observe(128, Duration::from_millis(400));
        let shrunk = p.chunk_size(true);
        p.observe(shrunk, Duration::from_millis(395));
        assert_eq!(p.chunk_size(false), 128);
        assert_eq!(p.chunk_size(true), 128);
    }

    #[test]
    fn growing_the_chunk_never_trips_the_ineffective_check() {
        // Only a SHRINK that failed is evidence. A chunk getting bigger and
        // slower is exactly what should happen and must not disable pacing.
        let mut p = PrefillPacer::new(128, 200);
        p.observe(8, Duration::from_millis(50));
        p.observe(128, Duration::from_millis(800));
        assert!(
            p.chunk_size(true) < 128,
            "a larger-and-slower observation is normal, not proof pacing fails"
        );
    }

    #[test]
    fn is_sharing_only_counts_more_than_one_slot() {
        assert!(!PrefillPacer::is_sharing(0));
        assert!(!PrefillPacer::is_sharing(1));
        assert!(PrefillPacer::is_sharing(2));
    }
}
