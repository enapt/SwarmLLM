//! What we have learned about how fast each peer actually computes.
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

/// Observed compute speed of one peer, as two separately-normalised EMAs.
#[derive(Debug, Clone)]
pub struct PeerSpeed {
    /// EMA of ms per (layer × activation byte) during prefill.
    prefill_ms_per_layer_byte: Option<f32>,
    /// EMA of ms per layer for a single-token decode step.
    decode_ms_per_layer: Option<f32>,
    prefill_samples: u32,
    decode_samples: u32,
    updated_at: Instant,
}

impl Default for PeerSpeed {
    fn default() -> Self {
        Self {
            prefill_ms_per_layer_byte: None,
            decode_ms_per_layer: None,
            prefill_samples: 0,
            decode_samples: 0,
            updated_at: Instant::now(),
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
    pub fn observe(
        &mut self,
        kind: WorkKind,
        segment_ms: u64,
        layers: u32,
        activation_bytes: usize,
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
            WorkKind::Decode => segment_ms as f64 / layers as f64,
        } as f32;
        if !sample.is_finite() {
            return;
        }

        let (slot, count) = match kind {
            WorkKind::Prefill => (
                &mut self.prefill_ms_per_layer_byte,
                &mut self.prefill_samples,
            ),
            WorkKind::Decode => (&mut self.decode_ms_per_layer, &mut self.decode_samples),
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
            WorkKind::Decode => self.decode_ms_per_layer? * layers as f32,
        };
        predicted.is_finite().then_some(predicted)
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
        if let Some(d) = self.decode_ms_per_layer {
            return Some(d);
        }
        self.prefill_ms_per_layer_byte
            .map(|c| c * NOMINAL_DECODE_ACTIVATION_BYTES as f32)
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
const NOMINAL_DECODE_ACTIVATION_BYTES: usize = 4096 * 4;

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers here are the ones measured live on 2026-08-01 — the case
    /// that motivated splitting the EMA.
    #[test]
    fn prefill_and_decode_do_not_contaminate_each_other() {
        let mut s = PeerSpeed::default();
        // 8-layer segment, 213268 bytes of activations, 10199 ms.
        s.observe(WorkKind::Prefill, 10_199, 8, 213_268);
        // Same peer, same segment, single-token decode: 150 ms.
        s.observe(WorkKind::Decode, 150, 8, 16_384);

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
        s.observe(WorkKind::Prefill, 1_000, 4, 100_000);

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

        s.observe(WorkKind::Decode, 100, 4, 16_384);
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
        s.observe(WorkKind::Prefill, 500, 0, 1000); // zero layers
        s.observe(WorkKind::Prefill, 500, 4, 0); // zero bytes
        assert_eq!(s.prefill_samples(), 0);
        assert_eq!(s.predict_ms(WorkKind::Prefill, 4, 1000), None);

        s.observe(WorkKind::Decode, 500, 0, 1000);
        assert_eq!(s.decode_samples(), 0);
    }

    #[test]
    fn the_ema_follows_a_peer_that_changes_speed() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 100, 1, 0);
        let first = s.predict_ms(WorkKind::Decode, 1, 0).unwrap();
        assert!((first - 100.0).abs() < 0.01, "first sample seeds directly");

        for _ in 0..20 {
            s.observe(WorkKind::Decode, 400, 1, 0);
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
        only_prefill.observe(WorkKind::Prefill, 10_199, 8, 213_268);
        assert!(
            only_prefill.ranking_ms_per_layer().is_some(),
            "a prefill-only peer must still be rankable"
        );

        let mut both = PeerSpeed::default();
        both.observe(WorkKind::Prefill, 10_199, 8, 213_268);
        both.observe(WorkKind::Decode, 150, 8, 16_384);
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
        assert!(s.ranking_ms_per_layer().is_some());
    }

    #[test]
    fn gossip_only_seeds_an_unseen_peer_when_trusted_enough() {
        let mut low = PeerSpeed::default();
        low.merge_ranking_sample(500.0, 0.29, 0.3);
        assert_eq!(low.ranking_ms_per_layer(), None);

        let mut high = PeerSpeed::default();
        high.merge_ranking_sample(500.0, 0.3, 0.3);
        assert_eq!(high.ranking_ms_per_layer(), Some(500.0));
    }

    /// A direct observation must dominate hearsay about the same peer.
    #[test]
    fn gossip_refines_but_does_not_overwrite_a_measurement() {
        let mut s = PeerSpeed::default();
        s.observe(WorkKind::Decode, 100, 1, 0);
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
        s.observe(WorkKind::Decode, 2_200, 22, 4096);
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
        s.updated_at = Instant::now() - (RANKING_STALE_AFTER + Duration::from_secs(1));
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
        s.updated_at = Instant::now() - (RANKING_STALE_AFTER + Duration::from_secs(1));
        assert!(s.ranking_ms_per_layer().is_none());
        s.observe(WorkKind::Decode, 30, 22, 4096);
        assert!(
            s.ranking_ms_per_layer().is_some(),
            "a fresh sample must make the peer measurable again"
        );
    }
}
