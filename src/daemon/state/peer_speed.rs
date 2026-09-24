//! What we have learned about how fast each peer actually computes, and how
//! reliably the path to it delivers.
//!
//! Those are two different questions and the answers must not be mixed. Speed
//! is a property of the peer's hardware; delivery is a property of the link
//! between here and there. Folding a delivery failure into a speed figure is
//! what recorded an RTX 4050 at 345 ms/layer against the 3.1 it had measured
//! minutes earlier — see `delivery_intact_ratio`.
//!
//! # Why prefill and decode are tracked separately
//!
//! They differ by roughly two orders of magnitude on the *same* hardware, and
//! they scale with different things:
//!
//! - **Prefill** processes the whole prompt at once. Its cost is linear in
//!   both the layer count and the prompt size, so it is normalised by
//!   `layers × activation_bytes`.
//! - **Decode** processes a single token. Its cost is linear in layers only.
//!
//! Measured live on 2026-08-01 against one CPU-only peer serving an 8B model
//! over an 8-layer segment: prefill ran at **1275 ms/layer** (10.2s for 213 KB
//! of activations) while decode ran at **18.75 ms/layer** (150 ms). The single
//! blended EMA that preceded this module sat at **239 ms/layer** — a figure
//! that predicts neither, and which is simply an artefact of whatever mix of
//! prefill and decode samples happened to arrive.
//!
//! # Why normalising by activation bytes makes this model-independent
//!
//! `activation_bytes = tokens × hidden_dim × 4`, and prefill work is
//! proportional to `layers × tokens × hidden_dim`. Dividing by
//! `layers × activation_bytes` therefore cancels the model's width as well as
//! the prompt length, so one coefficient per peer transfers across models. It
//! is a first-order model: attention is quadratic in prompt length and
//! quantisation varies, so the coefficient drifts somewhat with very long
//! prompts. It is used to size a *timeout* with a safety factor, not to make a
//! precise promise.

use std::time::{Duration, Instant};

/// Which half of inference a sample came from. Prefill and decode have
/// separate coefficients — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    Prefill,
    Decode,
    /// A whole model run on the peer, one decode step's share of it.
    ///
    /// **Kept apart from [`WorkKind::Decode`] because the two measure different
    /// things.** A mid-chain decode sample is a coordinator round trip: we send
    /// activations, the peer computes its layers, it sends them back, once per
    /// token. A delegated sample has no per-token round trip at all — the peer
    /// holds the whole model and streams tokens out. Folding them into one EMA
    /// would price a delegated run using a figure that carries a round trip it
    /// will never pay, which is exactly how the delegated option came to be
    /// systematically overcharged (see `docs/FUTURE_WORK.md`).
    Delegated,
}

/// Weight given to the newest sample in each EMA. Responsive enough to follow
/// a peer that has genuinely changed (another workload arriving, thermal
/// throttling) without letting one outlier dominate.
const ALPHA: f32 = 0.3;

/// How long a measured speed keeps influencing ranking.
///
/// Long enough that a peer in steady use is always measured — an active
/// coordinator re-measures a peer on every request it routes there — and short
/// enough that a peer which fell out of rotation returns to a neutral price
/// rather than staying frozen at whatever it happened to score once.
///
/// Ten minutes was picked to sit well above a normal request's gap and well
/// below the timescale over which hardware actually changes.
const RANKING_STALE_AFTER: Duration = Duration::from_secs(600);

/// How much of a reply may go missing before a peer is priced as unreliable.
///
/// Bounds the penalty rather than letting a run of losses price a peer out
/// permanently: at this floor the multiplier is 20x, which is already decisive
/// against any competitor, and a peer that recovers climbs back out. Without a
/// floor a ratio of zero is an infinite cost, and infinity is not a number a
/// peer can ever come back from.
const MIN_INTACT_DELIVERY_RATIO: f32 = 0.05;

/// Least samples before a two-term fit is trusted over the proportional EMA.
///
/// Two points define a line, so three is the first count that can disagree with
/// them — and a fit from two samples is an interpolation dressed as a model.
const TWO_TERM_MIN_SAMPLES: u32 = 4;

/// Least weighted variance in the layer count before the two terms can be told
/// apart at all.
///
/// A peer always given the SAME segment width gives a singular system: every
/// `(fixed, slope)` pair through that one point fits it equally, and solving
/// anyway returns whatever the floating-point noise decides. A peer holding one
/// shard of one model is exactly that case, which is why the proportional EMA
/// stays as the fallback rather than being replaced.
const TWO_TERM_MIN_LAYER_VARIANCE: f32 = 1.0;

/// One peer's `segment_ms ≈ fixed + slope × layers`, fitted online.
///
/// **Why a second term at all.** Measured 2026-09-20 (gotcha #659): a peer given
/// **2 of 32 layers** took a chain's time-per-token from 152 ms to 3768 ms —
/// the other 30 layers cost less than those 2. A model with only a per-layer
/// coefficient cannot express that, in either direction: it under-prices a
/// small segment on an expensive peer and over-prices a large one on a cheap
/// peer, and the first mistake is the one that routes work to the machine that
/// will ruin the request.
///
/// ⚠ **The fixed cost is NOT the round trip and must not be derived from one.**
/// Same session, solo on one model: 105 ms RTT → 5.39 tok/s, 1043 ms → 2.66,
/// 643 ms → **0.48**. The peer 1043 ms away is five times faster than the one
/// 643 ms away, so ping predicts nothing about what a peer does to a chain. It
/// is whatever that peer spends per visit — scheduling, worker IPC, cache
/// handling, its own load — and the only honest way to know it is to measure it.
///
/// Decayed least squares, weighted like the EMAs beside it so one sample moves
/// it by [`ALPHA`]: every accumulator is scaled by `1 - ALPHA` before the new
/// sample is added at `ALPHA`.
#[derive(Debug, Clone, Default)]
struct LinearFit {
    /// Σw, Σwx, Σwx², Σwy, Σwxy with x = layers, y = ms.
    w: f64,
    wx: f64,
    wxx: f64,
    wy: f64,
    wxy: f64,
    samples: u32,
}

impl LinearFit {
    fn observe(&mut self, layers: u32, ms: f64) {
        let decay = (1.0 - ALPHA) as f64;
        let a = ALPHA as f64;
        let x = layers as f64;
        self.w = self.w * decay + a;
        self.wx = self.wx * decay + a * x;
        self.wxx = self.wxx * decay + a * x * x;
        self.wy = self.wy * decay + a * ms;
        self.wxy = self.wxy * decay + a * x * ms;
        self.samples = self.samples.saturating_add(1);
    }

    /// `(fixed_ms, ms_per_layer)`, or `None` when the samples cannot tell the
    /// two apart — too few, or all at one width.
    ///
    /// **A negative SLOPE declines.** Cost falling as a segment gets wider is
    /// not something a peer does; it is what one stall on a narrow segment
    /// looks like, and the proportional EMA is the better answer about those
    /// samples.
    ///
    /// **A negative FIXED term is clamped to zero, not declined.** It is an
    /// estimate of a quantity that cannot be negative, so a peer whose cost
    /// really is proportional fits it at zero plus noise and lands below it
    /// about half the time. Declining there would make this fire only for
    /// expensive peers and never for cheap ones, which is a bias, not a
    /// safeguard. Clamping leaves the slope carrying the whole cost — the
    /// pessimistic direction, and the one this codebase already prefers for a
    /// candidate it is unsure about.
    fn terms(&self) -> Option<(f32, f32)> {
        if self.samples < TWO_TERM_MIN_SAMPLES || self.w <= 0.0 {
            return None;
        }
        // w · Σwx² − (Σwx)² is the weighted variance of x, times w². Zero when
        // every sample had the same layer count.
        let denom = self.w * self.wxx - self.wx * self.wx;
        let variance = denom / (self.w * self.w);
        if !variance.is_finite() || variance < TWO_TERM_MIN_LAYER_VARIANCE as f64 {
            return None;
        }
        let slope = (self.w * self.wxy - self.wx * self.wy) / denom;
        let fixed = (self.wy - slope * self.wx) / self.w;
        if !slope.is_finite() || !fixed.is_finite() || slope < 0.0 {
            return None;
        }
        Some((fixed.max(0.0) as f32, slope as f32))
    }
}

/// Observed compute speed of one peer, plus how reliably the path to it
/// actually delivers.
#[derive(Debug, Clone)]
pub struct PeerSpeed {
    /// EMA of ms per (layer × activation byte) during prefill.
    prefill_ms_per_layer_byte: Option<f32>,
    /// EMA of ms per layer for a single-token decode step.
    decode_ms_per_layer: Option<f32>,
    /// EMA of ms per layer for one decode step of a WHOLE model run on the
    /// peer — no per-token round trip. See [`WorkKind::Delegated`].
    delegated_ms_per_layer: Option<f32>,
    /// The same decode samples, fitted as `fixed + slope × layers` instead of
    /// forced through the origin. See [`LinearFit`] for the measurement that
    /// makes the second term necessary. Decode only: it is the token hot path,
    /// and the one where a mispriced small segment ruins a whole reply.
    decode_fit: LinearFit,
    prefill_samples: u32,
    decode_samples: u32,
    delegated_samples: u32,
    updated_at: Instant,
    /// EMA over "did this reply arrive intact", 1.0 = nothing lost.
    ///
    /// **This measures the PATH, not the hardware.** Every `StreamingToken` is
    /// an independent fire-and-forget send with no acknowledgement and no
    /// retransmission, so one drop truncates a reply permanently — the
    /// reassembler may only release the consecutive run. Measured 2026-08-20:
    /// all six losses in a session were on peers 450-650 ms away and none on
    /// the peer at 1 ms, despite that one serving more requests.
    ///
    /// Kept apart from the speed coefficients because a lossy link says nothing
    /// about how fast a peer computes — conflating them is what recorded an RTX
    /// 4050 at 345 ms/layer against its real 3.1. Kept apart from trust because
    /// **nobody is at fault**: the peer generated the whole answer and we asked
    /// for it correctly; the path between lost it. Docking a peer's reputation
    /// for that is the wrong-culprit mistake this codebase has made twice.
    ///
    /// Starts optimistic at 1.0 and needs no minimum sample count: with
    /// `ALPHA` at 0.3 a single loss moves it to 0.7, a 1.4x penalty, which is a
    /// nudge rather than a verdict, and a genuinely lossy path converges down
    /// fast.
    delivery_intact_ratio: f32,
    delivery_samples: u32,
    /// Separate from `updated_at` on purpose. A truncated stream records a
    /// delivery sample but deliberately records NO speed sample, so sharing one
    /// timestamp would let losses keep a stale speed figure alive for as long
    /// as they kept arriving.
    delivery_updated_at: Instant,
}

impl Default for PeerSpeed {
    fn default() -> Self {
        Self {
            prefill_ms_per_layer_byte: None,
            decode_ms_per_layer: None,
            delegated_ms_per_layer: None,
            decode_fit: LinearFit::default(),
            prefill_samples: 0,
            decode_samples: 0,
            delegated_samples: 0,
            updated_at: Instant::now(),
            delivery_intact_ratio: 1.0,
            delivery_samples: 0,
            delivery_updated_at: Instant::now(),
        }
    }
}

impl PeerSpeed {
    /// Fold one completed segment into the matching EMA.
    ///
    /// Samples that cannot be normalised are ignored rather than poisoning the
    /// average: a zero layer count, a prefill with no activation bytes, or a
    /// non-finite duration. `segment_ms` is the wall-clock round trip, so it
    /// includes network time — which is what we want, since the number is used
    /// to decide how long to wait for this peer.
    /// `warm` is whether this peer already had the model resident when the
    /// segment ran. A COLD segment paid for loading the model as well as
    /// running it, so it says almost nothing about how fast the peer computes —
    /// at a few hundred ms of round trip it can be an order of magnitude out.
    ///
    /// A cold sample still updates the PREFILL coefficient, because that sizes
    /// how long we are willing to wait and being pessimistic there is safe. It
    /// must not update the DECODE coefficient, which is what ranks the peer:
    /// ranking on a load time is how a fast machine gets classified as slow and
    /// then never gets another chance to prove otherwise.
    pub fn observe(
        &mut self,
        kind: WorkKind,
        segment_ms: u64,
        layers: u32,
        activation_bytes: usize,
        warm: bool,
    ) {
        if layers == 0 {
            return;
        }
        let sample = match kind {
            WorkKind::Prefill => {
                if activation_bytes == 0 {
                    return;
                }
                segment_ms as f64 / (layers as f64 * activation_bytes as f64)
            }
            WorkKind::Decode | WorkKind::Delegated => segment_ms as f64 / layers as f64,
        } as f32;
        if !sample.is_finite() {
            return;
        }

        // A cold decode sample is a load time wearing a compute figure's clothes.
        if !warm && matches!(kind, WorkKind::Decode | WorkKind::Delegated) {
            return;
        }
        // The same sample, unnormalised, for the two-term fit. Fed the RAW
        // wall-clock and layer count rather than the per-layer figure above,
        // because dividing by `layers` is precisely what makes the fixed term
        // unrecoverable.
        if matches!(kind, WorkKind::Decode) {
            self.decode_fit.observe(layers, segment_ms as f64);
        }
        let (slot, count) = match kind {
            WorkKind::Prefill => (
                &mut self.prefill_ms_per_layer_byte,
                &mut self.prefill_samples,
            ),
            WorkKind::Decode => (&mut self.decode_ms_per_layer, &mut self.decode_samples),
            WorkKind::Delegated => (
                &mut self.delegated_ms_per_layer,
                &mut self.delegated_samples,
            ),
        };
        *slot = Some(match *slot {
            Some(prev) => ALPHA * sample + (1.0 - ALPHA) * prev,
            None => sample,
        });
        *count = count.saturating_add(1);
        self.updated_at = Instant::now();
    }

    /// Predicted milliseconds for a segment of this shape on this peer, or
    /// `None` when we have never seen the relevant kind of work from it.
    pub fn predict_ms(&self, kind: WorkKind, layers: u32, activation_bytes: usize) -> Option<f32> {
        let predicted = match kind {
            WorkKind::Prefill => {
                self.prefill_ms_per_layer_byte? * layers as f32 * activation_bytes as f32
            }
            // Two-term where the samples can support one. This also sizes the
            // per-segment deadline, where the proportional model was wrong in
            // the dangerous direction: a small segment on a peer with a large
            // fixed cost got a correspondingly small budget and was failed over
            // before it could possibly have answered.
            WorkKind::Decode => match self.decode_terms() {
                Some((fixed, per_layer)) => fixed + per_layer * layers as f32,
                None => self.decode_ms_per_layer? * layers as f32,
            },
            WorkKind::Delegated => self.delegated_ms_per_layer? * layers as f32,
        };
        predicted.is_finite().then_some(predicted)
    }

    /// `(fixed_ms_per_visit, ms_per_layer)` for a decode step, when the samples
    /// can tell the two apart. `None` falls the caller back to the proportional
    /// EMA, which is what every peer used before this existed.
    ///
    /// **Expires with the ranking figure**, for the same reason: a fit nobody
    /// has refreshed in [`RANKING_STALE_AFTER`] describes a peer as it was, and
    /// a stale price is how the routing ratchet starts.
    pub fn decode_terms(&self) -> Option<(f32, f32)> {
        if self.updated_at.elapsed() > RANKING_STALE_AFTER {
            return None;
        }
        self.decode_fit.terms()
    }

    /// Per-layer cost used for *ranking* peers against each other, in ms.
    ///
    /// Ranking wants one comparable number per peer, and decode dominates a
    /// generated answer (one prefill, then one decode step per token), so the
    /// decode coefficient is the honest choice where we have it. Prefill is
    /// converted at a nominal activation width when decode is unseen, so a
    /// peer we have only ever prefilled through still ranks.
    /// **An observation expires.** Past [`RANKING_STALE_AFTER`] this returns
    /// `None`, so the scheduler prices the peer from its advertised capability
    /// instead — the same path a peer we have never measured already takes.
    ///
    /// Without an expiry the EMA has no decay and is only ever updated when we
    /// route to a peer, so a single bad sample is permanent: a peer measured
    /// slow once — during a cold model load, or a momentary load spike — keeps
    /// that number for the life of the process, is priced badly, is therefore
    /// not routed to, and so is never re-measured. That is a ratchet, and it
    /// falls hardest on modest hardware, which is also the hardware most likely
    /// to produce one slow sample while loading.
    ///
    /// Expiring rather than decaying is deliberate: there is no prior stored
    /// here to decay *toward*, and the capability estimate the caller already
    /// falls back to is exactly that prior. Falling back cannot price a peer
    /// worse than one that was never measured at all, which bounds the risk.
    pub fn ranking_ms_per_layer(&self) -> Option<f32> {
        if self.updated_at.elapsed() >= RANKING_STALE_AFTER {
            return None;
        }
        // Only a figure THIS node measured may rank a peer. `merge_ranking_sample`
        // seeds `decode_ms_per_layer` from gossip without touching the sample
        // count, so `decode_samples == 0` means "a stranger's number", and a
        // stranger's number describes the path from THEM to that peer, not from
        // here.
        //
        // This is the same rule the prefill coefficient has always followed —
        // gossip deliberately cannot seed it, because "a figure we did not
        // measure ourselves must not be able to shorten how long we are willing
        // to wait for a peer". Ranking had the identical exposure and no such
        // guard, and the consequence was worse than a bad timeout: an inherited
        // figure OUTRANKS the peer's own advertised speed, because
        // `parallax::vertex_cost` prefers any observation over the static
        // estimate and drops the network term when it has one.
        //
        // Measured 2026-08-19. A probe node that had been alive for minutes and
        // had never sent that peer a single segment already held 1063 ms/layer
        // for an RTX 4050 advertising 20.45 tok/s, against 83 ms/layer for an
        // i5-10500T advertising 0.82. Over 16 layers that priced the GPU at
        // 17,008 ms against 1,328 — a 13x margin against hardware 25x faster —
        // and the request it lost came back at 0.23 tok/s where the same box
        // serves a larger model locally at 36. One node's bad sample had become
        // every node's opinion, because observations travel in
        // `NodeCapability.observed_latencies`.
        if self.decode_samples > 0 {
            if let Some(d) = self.decode_ms_per_layer {
                return Some(d);
            }
        }
        // Prefill is local by construction, so it can always rank.
        self.prefill_ms_per_layer_byte
            .map(|c| c * NOMINAL_DECODE_ACTIVATION_BYTES as f32)
    }

    /// Measured prefill coefficient, in ms per (layer x activation byte), or
    /// `None` when this node has never prefilled through this peer.
    ///
    /// Separate from [`Self::ranking_ms_per_layer`] because prefill and decode
    /// are not the same work and do not scale with the same thing: prefill is
    /// linear in prompt length, decode is not. Measured on the same hardware
    /// they differ by roughly two orders of magnitude (see the module docs).
    ///
    /// Expires on the same clock as ranking, and carries the same rule: only a
    /// figure THIS node measured may price a peer, because a stranger's number
    /// describes the path from them to that peer rather than from here.
    /// `merge_ranking_sample` deliberately cannot seed this coefficient, so the
    /// sample check is belt-and-braces rather than the only guard.
    pub fn prefill_ms_per_layer_byte(&self) -> Option<f32> {
        if self.updated_at.elapsed() >= RANKING_STALE_AFTER {
            return None;
        }
        if self.prefill_samples == 0 {
            return None;
        }
        self.prefill_ms_per_layer_byte
    }

    /// Per-layer cost of running a **whole model** on this peer, in ms — the
    /// delegated shape, with no per-token round trip in it.
    ///
    /// Separate from [`Self::ranking_ms_per_layer`] on purpose, and the two must
    /// not substitute for each other. A mid-chain decode sample carries the
    /// coordinator's round trip amortised over however many layers that segment
    /// happened to own; reusing it to price a delegated run charges that round
    /// trip several times over for a trip the delegated run never makes. Measured
    /// at roughly 2.7x for a 16-layer delegation priced from a 6-layer mid-chain
    /// observation, which is why the router kept preferring a split that ran
    /// 11.2s → 17.8s.
    ///
    /// Expires on the same clock as ranking, and for the same reason: an EMA with
    /// no decay that is only updated when we route to a peer would freeze a peer
    /// at one bad sample and then never re-measure it.
    ///
    /// Like ranking, this returns only a figure **this node measured**. Gossip
    /// cannot seed it — `merge_ranking_sample` writes the decode coefficient and
    /// never touches this one — but the sample count is checked anyway so the
    /// rule holds by construction rather than by where a caller happens to write.
    pub fn delegated_ms_per_layer(&self) -> Option<f32> {
        if self.updated_at.elapsed() >= RANKING_STALE_AFTER {
            return None;
        }
        if self.delegated_samples == 0 {
            return None;
        }
        self.delegated_ms_per_layer
    }

    pub fn delegated_samples(&self) -> u32 {
        self.delegated_samples
    }

    /// Record whether one completed reply from this peer arrived intact.
    ///
    /// Binary rather than the fraction delivered, because a truncated answer is
    /// a failed answer whether it lost two tokens or fifty-seven — the caller
    /// cannot use either.
    pub fn observe_delivery(&mut self, intact: bool) {
        let sample = if intact { 1.0 } else { 0.0 };
        self.delivery_intact_ratio = ALPHA * sample + (1.0 - ALPHA) * self.delivery_intact_ratio;
        self.delivery_samples = self.delivery_samples.saturating_add(1);
        self.delivery_updated_at = Instant::now();
    }

    /// Fraction of recent replies from this peer that arrived intact, or `None`
    /// when we have not delivered from it recently enough to say.
    ///
    /// Expires on the same clock as the speed figures, for the same reason: a
    /// link that was lossy an hour ago may be fine now, and a figure that never
    /// decays is a ratchet. Falling back to "unknown" prices the peer exactly
    /// like one we have never used, which cannot be worse than never having
    /// tried it.
    pub fn intact_delivery_ratio(&self) -> Option<f32> {
        if self.delivery_samples == 0 || self.delivery_updated_at.elapsed() >= RANKING_STALE_AFTER {
            return None;
        }
        Some(self.delivery_intact_ratio)
    }

    pub fn delivery_samples(&self) -> u32 {
        self.delivery_samples
    }

    /// How much more work it takes, in expectation, to get one intact answer
    /// out of this peer.
    ///
    /// This is not a tuning knob: if a fraction `p` of replies arrive intact,
    /// getting one intact reply costs `1/p` attempts in expectation. A perfect
    /// path multiplies by 1 and changes nothing; a path delivering 5% of
    /// replies multiplies by 20. Clamped by [`MIN_INTACT_DELIVERY_RATIO`] so a
    /// bad run cannot price a peer out beyond recovery.
    pub fn expected_attempts_multiplier(&self) -> f32 {
        match self.intact_delivery_ratio() {
            None => 1.0,
            Some(p) => 1.0 / p.clamp(MIN_INTACT_DELIVERY_RATIO, 1.0),
        }
    }

    /// Fold in a per-layer figure that another node gossiped to us, weighted
    /// by how much we trust the reporter.
    ///
    /// Gossip carries "this peer computes a layer in about X ms", which is the
    /// ranking-scale quantity, so it merges into the decode coefficient. It
    /// deliberately cannot seed the *prefill* coefficient: prefill sizes a
    /// timeout, and a figure we did not measure ourselves must not be able to
    /// shorten how long we are willing to wait for a peer.
    ///
    /// An unseen peer is only seeded by a sufficiently trusted reporter
    /// (`seed_threshold`); below that the sample refines an existing estimate
    /// but cannot create one.
    pub fn merge_ranking_sample(
        &mut self,
        sample_ms_per_layer: f32,
        weight: f32,
        seed_threshold: f32,
    ) {
        if !sample_ms_per_layer.is_finite() || sample_ms_per_layer <= 0.0 || weight <= 0.0 {
            return;
        }
        let weight = weight.clamp(0.0, 1.0);
        let effective_alpha = ALPHA * weight;
        match self.decode_ms_per_layer {
            Some(prev) => {
                self.decode_ms_per_layer =
                    Some(effective_alpha * sample_ms_per_layer + (1.0 - effective_alpha) * prev);
                self.updated_at = Instant::now();
            }
            None => {
                if weight >= seed_threshold {
                    self.decode_ms_per_layer = Some(sample_ms_per_layer);
                    self.updated_at = Instant::now();
                }
            }
        }
    }

    pub fn prefill_samples(&self) -> u32 {
        self.prefill_samples
    }

    pub fn decode_samples(&self) -> u32 {
        self.decode_samples
    }

    pub fn updated_at(&self) -> Instant {
        self.updated_at
    }

    /// Has this peer gone quiet for longer than `ttl`? Stale entries are
    /// evicted rather than decayed: a peer we have not routed to in an hour
    /// tells us nothing useful, and keeping the old figure is what let slow
    /// nodes stay permanently de-ranked ("the routing ratchet") while departed
    /// peers accumulated in the map.
    pub fn is_stale(&self, now: Instant, ttl: std::time::Duration) -> bool {
        now.duration_since(self.updated_at) > ttl
    }
}

/// Activation size of a single-token decode step at a typical hidden width
/// (4096 × 4 bytes). Used only to put a prefill-only coefficient on the same
/// scale as a decode one for ranking.
pub(crate) const NOMINAL_DECODE_ACTIVATION_BYTES: usize = 4096 * 4;

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers here are the ones measured live on 2026-08-01 — the case
    /// that motivated splitting the EMA.
    #[test]
    fn prefill_and_decode_do_not_contaminate_each_other() {
        let mut s = PeerSpeed::default();
        // 8-layer segment, 213268 bytes of activations, 10199 ms.
        s.observe(WorkKind::Prefill, 10_199, 8, 213_268, true);
        // Same peer, same segment, single-token decode: 150 ms.
        s.observe(WorkKind::Decode, 150, 8, 16_384, true);

        let p = s.predict_ms(WorkKind::Prefill, 8, 213_268).unwrap();
        let d = s.predict_ms(WorkKind::Decode, 8, 16_384).unwrap();
        assert!(
            (p - 10_199.0).abs() < 1.0,
            "prefill prediction should reproduce its own sample, got {p}"
        );
        assert!(
            (d - 150.0).abs() < 1.0,
            "decode prediction should reproduce its own sample, got {d}"
        );
        // The blended figure this replaced was 239 ms/layer, which is ~1900ms
        // for this segment — 5x short of the real prefill.
        assert!(p > 5.0 * d, "prefill and decode must stay distinguishable");
    }

    /// Normalising by activation bytes means a longer prompt scales the
    /// prediction, which is the whole point — a flat per-layer budget is what
    /// cut off a legitimately slow prefill.
    #[test]
    fn prediction_scales_with_prompt_size_and_layers() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Prefill, 1_000, 4, 100_000, true);

        let base = s.predict_ms(WorkKind::Prefill, 4, 100_000).unwrap();
        let double_tokens = s.predict_ms(WorkKind::Prefill, 4, 200_000).unwrap();
        let double_layers = s.predict_ms(WorkKind::Prefill, 8, 100_000).unwrap();

        assert!((base - 1_000.0).abs() < 1.0);
        assert!((double_tokens - 2_000.0).abs() < 1.0);
        assert!((double_layers - 2_000.0).abs() < 1.0);
    }

    #[test]
    fn unseen_work_kinds_predict_nothing_rather_than_guessing() {
        let mut s = PeerSpeed::default();
        assert_eq!(s.predict_ms(WorkKind::Prefill, 4, 1000), None);
        assert_eq!(s.predict_ms(WorkKind::Decode, 4, 1000), None);

        s.observe(WorkKind::Decode, 100, 4, 16_384, true);
        assert!(s.predict_ms(WorkKind::Decode, 4, 16_384).is_some());
        assert_eq!(
            s.predict_ms(WorkKind::Prefill, 4, 1000),
            None,
            "a decode sample must not be used to predict prefill"
        );
    }

    #[test]
    fn unnormalisable_samples_are_ignored() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Prefill, 500, 0, 1000, true); // zero layers
        s.observe(WorkKind::Prefill, 500, 4, 0, true); // zero bytes
        assert_eq!(s.prefill_samples(), 0);
        assert_eq!(s.predict_ms(WorkKind::Prefill, 4, 1000), None);

        s.observe(WorkKind::Decode, 500, 0, 1000, true);
        assert_eq!(s.decode_samples(), 0);
    }

    #[test]
    fn the_ema_follows_a_peer_that_changes_speed() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 100, 1, 0, true);
        let first = s.predict_ms(WorkKind::Decode, 1, 0).unwrap();
        assert!((first - 100.0).abs() < 0.01, "first sample seeds directly");

        for _ in 0..20 {
            s.observe(WorkKind::Decode, 400, 1, 0, true);
        }
        let settled = s.predict_ms(WorkKind::Decode, 1, 0).unwrap();
        assert!(
            settled > 380.0,
            "EMA should converge toward the new speed, got {settled}"
        );
    }

    #[test]
    fn ranking_prefers_decode_but_falls_back_to_prefill() {
        let mut only_prefill = PeerSpeed::default();
        only_prefill.observe(WorkKind::Prefill, 10_199, 8, 213_268, true);
        assert!(
            only_prefill.ranking_ms_per_layer().is_some(),
            "a prefill-only peer must still be rankable"
        );

        let mut both = PeerSpeed::default();
        both.observe(WorkKind::Prefill, 10_199, 8, 213_268, true);
        both.observe(WorkKind::Decode, 150, 8, 16_384, true);
        assert!((both.ranking_ms_per_layer().unwrap() - 18.75).abs() < 0.1);
    }

    /// Gossip must not be able to shorten a timeout we size from our own
    /// measurements — it can rank a peer, never claim how fast it prefills.
    #[test]
    fn gossip_cannot_seed_the_prefill_coefficient() {
        let mut s = PeerSpeed::default();
        s.merge_ranking_sample(500.0, 1.0, 0.3);
        assert!(
            s.predict_ms(WorkKind::Prefill, 8, 213_268).is_none(),
            "a gossiped figure must never size a prefill timeout"
        );
        // Asserted on the seeded field rather than through
        // `ranking_ms_per_layer`, which deliberately no longer surfaces a figure
        // this node did not measure — see
        // `ranking_trusts_only_what_it_measured`. The seeding behaviour under
        // test here is unchanged.
        assert!(s.decode_ms_per_layer.is_some());
    }

    #[test]
    fn gossip_only_seeds_an_unseen_peer_when_trusted_enough() {
        let mut low = PeerSpeed::default();
        low.merge_ranking_sample(500.0, 0.29, 0.3);
        assert_eq!(low.decode_ms_per_layer, None);

        let mut high = PeerSpeed::default();
        high.merge_ranking_sample(500.0, 0.3, 0.3);
        assert_eq!(high.decode_ms_per_layer, Some(500.0));
    }

    /// A direct observation must dominate hearsay about the same peer.
    #[test]
    fn gossip_refines_but_does_not_overwrite_a_measurement() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 100, 1, 0, true);
        s.merge_ranking_sample(10_000.0, 1.0, 0.3);
        let after = s.ranking_ms_per_layer().unwrap();
        assert!(
            after < 3_100.0,
            "one gossiped outlier should not swamp a measurement, got {after}"
        );
        assert!(after > 100.0, "but it should still move the estimate");
    }

    #[test]
    fn staleness_is_measured_from_the_last_observation() {
        let s = PeerSpeed::default();
        let now = Instant::now();
        assert!(!s.is_stale(now, std::time::Duration::from_secs(60)));
        assert!(s.is_stale(
            now + std::time::Duration::from_secs(120),
            std::time::Duration::from_secs(60)
        ));
    }
}

#[cfg(test)]
mod ranking_staleness_tests {
    use super::*;

    fn measured_slow() -> PeerSpeed {
        let mut s = PeerSpeed::default();
        // One slow decode sample, as a cold model load would produce.
        s.observe(WorkKind::Decode, 2_200, 22, 4096, true);
        s
    }

    /// A fresh measurement must still rank — expiry must not simply disable
    /// observed latency, which is the whole point of measuring peers.
    #[test]
    fn a_fresh_observation_still_ranks() {
        let s = measured_slow();
        assert!(
            s.ranking_ms_per_layer().is_some(),
            "a just-taken measurement must be used"
        );
    }

    /// **The ratchet.** The EMA is only updated when we route to a peer, and it
    /// has no decay — so one slow sample (a cold load, a load spike) used to
    /// price that peer badly forever, which stopped it being routed to, which
    /// stopped it ever being re-measured. Expiring the observation returns it
    /// to the neutral capability-based price a never-measured peer gets.
    #[test]
    fn a_stale_observation_stops_ranking_so_the_peer_is_repriced() {
        let mut s = measured_slow();
        s.updated_at = Instant::now()
            .checked_sub(RANKING_STALE_AFTER + Duration::from_secs(1))
            .expect("up for longer than the staleness window");
        assert!(
            s.ranking_ms_per_layer().is_none(),
            "a stale measurement must stop pricing the peer — otherwise one bad \
             sample is permanent and the peer can never earn its way back"
        );
    }

    /// Re-measuring must lift the expiry, or a peer that came back would still
    /// be treated as unknown despite fresh evidence.
    #[test]
    fn re_measuring_restores_ranking() {
        let mut s = measured_slow();
        s.updated_at = Instant::now()
            .checked_sub(RANKING_STALE_AFTER + Duration::from_secs(1))
            .expect("up for longer than the staleness window");
        assert!(s.ranking_ms_per_layer().is_none());
        s.observe(WorkKind::Decode, 30, 22, 4096, true);
        assert!(
            s.ranking_ms_per_layer().is_some(),
            "a fresh sample must make the peer measurable again"
        );
    }
}

#[cfg(test)]
mod ranking_trusts_only_what_it_measured {
    use super::*;

    /// A figure that arrived by gossip must not rank a peer.
    ///
    /// The live case, 2026-08-19: a probe node that had never sent an RTX 4050
    /// a single segment already held 1063 ms/layer for it, inherited through
    /// `NodeCapability.observed_latencies`. Because `vertex_cost` prefers any
    /// observation over the peer's advertised speed, that number outranked a
    /// self-reported 20.45 tok/s and the work went to an 0.82 tok/s processor
    /// instead — 0.23 tok/s end to end, against 36 for the same box locally.
    #[test]
    fn a_gossiped_figure_does_not_rank_a_peer_we_have_never_used() {
        let mut s = PeerSpeed::default();
        s.merge_ranking_sample(1063.0, 1.0, 0.0);
        assert_eq!(
            s.ranking_ms_per_layer(),
            None,
            "a stranger's measurement describes THEIR path to that peer, not ours"
        );
        // It is still recorded, so anything that wants the hint can have it.
        assert_eq!(s.decode_ms_per_layer, Some(1063.0));
    }

    /// ...but once we have measured the peer ourselves, ranking uses it, and
    /// gossip may refine it.
    #[test]
    fn our_own_measurement_ranks_and_gossip_may_then_refine_it() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 160, 16, 16_384, true);
        let mine = s.ranking_ms_per_layer().expect("our own sample must rank");
        assert!(
            (mine - 10.0).abs() < 0.01,
            "160ms / 16 layers = 10ms, got {mine}"
        );
        s.merge_ranking_sample(1063.0, 1.0, 0.0);
        let blended = s.ranking_ms_per_layer().expect("still ranks");
        assert!(blended > mine, "gossip should move it");
        assert!(blended < 1063.0, "but must not replace it wholesale");
    }

    /// The first segment to a peer pays to LOAD the model. Ranking on that is
    /// how a fast machine is classified as slow and then never gets another
    /// chance — the sample that ranks it is the one it can never repeat.
    #[test]
    fn a_cold_segment_does_not_set_the_ranking_figure() {
        let mut cold = PeerSpeed::default();
        cold.observe(WorkKind::Decode, 17_000, 16, 16_384, false);
        assert_eq!(
            cold.ranking_ms_per_layer(),
            None,
            "a load time must not become this peer's compute speed"
        );

        // The same segment once the model is resident is a real measurement.
        let mut warm = PeerSpeed::default();
        warm.observe(WorkKind::Decode, 160, 16, 16_384, true);
        assert!(warm.ranking_ms_per_layer().is_some());
    }

    /// Being pessimistic about how long to WAIT is safe, so a cold sample still
    /// feeds the prefill coefficient that sizes timeouts.
    #[test]
    fn a_cold_segment_still_sizes_the_timeout() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Prefill, 10_000, 8, 200_000, false);
        assert!(
            s.predict_ms(WorkKind::Prefill, 8, 200_000).is_some(),
            "timeout sizing must keep the cold sample — waiting too long is safe, \
             waiting too little is not"
        );
    }

    /// The reliability figure must start optimistic, fall with losses, and
    /// recover — and it must never turn a peer into a permanent outcast.
    #[test]
    fn delivery_reliability_falls_with_losses_and_recovers() {
        let mut s = PeerSpeed::default();

        // Never used: unmeasured, and priced exactly like any other unknown.
        assert_eq!(s.intact_delivery_ratio(), None);
        assert_eq!(s.expected_attempts_multiplier(), 1.0);

        // A clean reply keeps it at 1.0 — no penalty for working correctly.
        s.observe_delivery(true);
        assert_eq!(s.expected_attempts_multiplier(), 1.0);

        // One loss is a nudge, not a verdict.
        s.observe_delivery(false);
        let after_one = s.expected_attempts_multiplier();
        assert!(
            after_one > 1.0 && after_one < 2.0,
            "one loss should nudge, got {after_one}"
        );

        // A persistently lossy path converges to a decisive penalty, and the
        // clamp stops it running away to infinity.
        for _ in 0..50 {
            s.observe_delivery(false);
        }
        let settled = s.expected_attempts_multiplier();
        assert_eq!(
            settled,
            1.0 / MIN_INTACT_DELIVERY_RATIO,
            "a path that loses everything must be clamped, not infinite"
        );

        // ...and it can climb back out once the path recovers.
        for _ in 0..50 {
            s.observe_delivery(true);
        }
        let recovered = s.expected_attempts_multiplier();
        assert!(
            recovered < 1.05,
            "a recovered path must stop being penalised, got {recovered}"
        );
    }

    /// Losing replies must not silently keep a stale SPEED figure alive. A
    /// truncated stream records a delivery sample and deliberately records no
    /// speed sample, so the two timestamps have to be separate.
    #[test]
    fn a_delivery_sample_does_not_refresh_the_speed_clock() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 160, 16, 16_384, true);
        let speed_clock = s.updated_at();

        s.observe_delivery(false);

        assert_eq!(
            s.updated_at(),
            speed_clock,
            "a delivery observation must leave the speed clock untouched"
        );
        assert!(s.intact_delivery_ratio().is_some());
    }
}

/// The second term, against the numbers that showed it was missing.
#[cfg(test)]
mod two_term_tests {
    use super::*;

    /// Feed a peer whose real behaviour is `fixed + per_layer × layers`, at
    /// widths a router would actually give it, and check both terms come back.
    fn observe_synthetic(s: &mut PeerSpeed, fixed: u64, per_layer: u64, widths: &[u32]) {
        for &layers in widths {
            s.observe(
                WorkKind::Decode,
                fixed + per_layer * layers as u64,
                layers,
                4096,
                true,
            );
        }
    }

    /// **The defect, in the shape it was measured in (gotcha #659).**
    ///
    /// A peer whose cost is nearly all fixed — 1000 ms per visit, 5 ms per
    /// layer — is given 2 layers. The proportional model divides its 32-layer
    /// observation by 32 and multiplies back by 2, pricing it at a sixteenth of
    /// what it will actually cost, which is exactly the mistake that put 2 of
    /// 32 layers on the peer that took a chain from 152 ms/token to 3768.
    #[test]
    fn a_small_segment_on_an_expensive_peer_is_not_priced_as_a_small_cost() {
        let mut s = PeerSpeed::default();
        observe_synthetic(&mut s, 1000, 5, &[32, 16, 24, 8, 32, 16]);

        let (fixed, per_layer) = s
            .decode_terms()
            .expect("widths vary, so the fit identifies");
        assert!(
            (fixed - 1000.0).abs() < 60.0,
            "fixed term {fixed} should recover ~1000 ms"
        );
        assert!(
            (per_layer - 5.0).abs() < 3.0,
            "slope {per_layer} should recover ~5 ms/layer"
        );

        let predicted = s.predict_ms(WorkKind::Decode, 2, 4096).unwrap();
        assert!(
            predicted > 900.0,
            "2 layers on this peer costs ~1010 ms, predicted {predicted}"
        );

        // What the proportional model says about the same peer, for contrast:
        // its EMA sits near the mean of segment_ms/layers, and 2 layers priced
        // through it is an order of magnitude cheaper than the truth.
        let proportional = s.ranking_ms_per_layer().unwrap() * 2.0;
        assert!(
            proportional < predicted / 4.0,
            "the proportional price ({proportional}) must be the one that is \
             badly wrong here, or this test is not about the defect"
        );
    }

    /// **The fallback is the whole safety argument.** A peer holding one shard
    /// of one model is always given the same width, so the two terms are not
    /// separable and nothing about its pricing may change.
    #[test]
    fn a_peer_always_given_the_same_width_keeps_the_proportional_price() {
        let mut s = PeerSpeed::default();
        observe_synthetic(&mut s, 1000, 5, &[16, 16, 16, 16, 16, 16]);
        assert!(
            s.decode_terms().is_none(),
            "one width cannot identify two terms; every (fixed, slope) through \
             that point fits it equally"
        );
        // And the prediction is exactly what it always was.
        let expected = s.ranking_ms_per_layer().unwrap() * 16.0;
        let got = s.predict_ms(WorkKind::Decode, 16, 4096).unwrap();
        assert!((got - expected).abs() < 0.01);
    }

    /// Too few samples is the other way the fit must decline. Two points define
    /// a line and cannot disagree with it.
    #[test]
    fn a_fit_needs_more_samples_than_it_has_parameters() {
        let mut s = PeerSpeed::default();
        observe_synthetic(&mut s, 1000, 5, &[32, 8]);
        assert!(s.decode_terms().is_none());
        observe_synthetic(&mut s, 1000, 5, &[24, 16]);
        assert!(
            s.decode_terms().is_some(),
            "four samples across four widths"
        );
    }

    /// A peer whose cost really is proportional must fit a fixed term near
    /// zero, not invent one — otherwise this makes every peer look expensive to
    /// visit and suppresses splitting everywhere.
    #[test]
    fn a_genuinely_proportional_peer_fits_almost_no_fixed_cost() {
        let mut s = PeerSpeed::default();
        observe_synthetic(&mut s, 0, 20, &[32, 16, 24, 8, 32, 16]);
        let (fixed, per_layer) = s.decode_terms().expect("widths vary");
        assert!(fixed < 40.0, "fixed term {fixed} should be ~0");
        assert!((per_layer - 20.0).abs() < 3.0, "slope {per_layer} ~20");
    }

    /// Samples the shape cannot explain — a cold load, a load spike — push the
    /// fit negative. That is not a rounding error to clamp: it means the model
    /// does not describe these samples, and the EMA is the better answer.
    #[test]
    fn a_fit_that_comes_out_negative_declines_rather_than_clamping() {
        let mut s = PeerSpeed::default();
        // Cost FALLING with width is not something a peer does; it is what a
        // one-off stall on a narrow segment looks like.
        for (layers, ms) in [(4u32, 5_000u64), (8, 3_000), (16, 1_500), (32, 400)] {
            s.observe(WorkKind::Decode, ms, layers, 4096, true);
        }
        assert!(
            s.decode_terms().is_none(),
            "a negative slope means the samples are about something else"
        );
    }

    /// The fit expires with the ranking figure it sits beside, or a peer that
    /// fell out of rotation keeps its price for ever — the routing ratchet.
    #[test]
    fn the_fit_goes_stale_with_the_rest_of_the_measurement() {
        let mut s = PeerSpeed::default();
        observe_synthetic(&mut s, 1000, 5, &[32, 16, 24, 8]);
        assert!(s.decode_terms().is_some());
        s.updated_at = Instant::now()
            .checked_sub(RANKING_STALE_AFTER + Duration::from_secs(1))
            .expect("up for longer than the staleness window");
        assert!(
            s.decode_terms().is_none(),
            "a stale fit must fall back like a stale EMA does"
        );
    }
}
