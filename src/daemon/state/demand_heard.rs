//! What each computer last said its demand for a model was, so a region's
//! demand does not depend on how OFTEN they say it (FUTURE_WORK #91 (2m)).
//!
//! A computer publishes the requests it served for peers, per model, as a rate
//! that changes only at its 10-minute decay tick. It used to publish it every
//! gossip round (30 s) for as long as the rate stood above 0.01 — ~4.7 h after
//! a single request, ~560 messages per computer and model — and on 2026-11-18
//! that was the largest gossip an idle node received (4.6 of 10.3 MB/h). The
//! sender now repeats an unmoved rate every 10 minutes (`health::monitor`'s
//! `demand_send_due`), so the receiver can no longer let the send cadence be
//! its averaging time: it blended `0.8 old + 0.2 new` per DELIVERED message,
//! which at 30 s a message is a time constant of ~134 s and at 10 minutes would
//! have been ~45 minutes. And with several computers in one region, a blend
//! per message is a mean of their rates only while their messages interleave
//! densely; sparse ones would have swung the value to whichever spoke last.
//!
//! So the receiver keeps each publisher's LATEST rate per (model, region) and
//! blends the MEAN of the fresh ones into `SharedState::region_demand` with a
//! weight that grows with the time since the key's last blend:
//! `1 − 0.8^(Δt / 30 s)` — exactly 0.2 at the old cadence, so a computer one
//! release behind that still sends every 30 s is read as before.
//!
//! Prior art: the irregular-interval EMA (a weight of `1 − e^(−Δt/τ)`, or a
//! half-life form such as Polars' `ewm_mean_by` and Vector's `TimeEwma`; one
//! form only, since mixing them skews τ by ln 2) — here the base-0.8-per-30-s
//! form, which is `e^(−Δt/τ)` with τ = 30 s / ln 1.25. Cassandra's gossip keeps
//! state PER ENDPOINT and lets the reader combine it, so a stable cluster moves
//! little data and a reader never depends on who spoke last; that is the shape
//! of `by_publisher` below.

use crate::types::{ModelId, NodeId};
use dashmap::DashMap;
use std::collections::HashMap;

/// The old cadence and what a message kept of the old value then: the blend's
/// weight is `1 − RETAINED_PER_STEP^(Δt / STEP_MS)`.
const STEP_MS: f64 = 30_000.0;
const RETAINED_PER_STEP: f64 = 0.8;

/// A publisher's rate counts toward its key's mean this long after it was
/// heard: 2.5 of the sender's longest gap (10 min, plus a gossip round of up
/// to 2 min on a large swarm) — the rule `pool::types::member_is_online` uses
/// for reports. A computer whose rate fell under the publish line stops
/// sending, and drops out of the mean once this has passed.
pub(crate) const HEARD_FRESH_FOR_MS: u64 = 25 * 60 * 1000;

/// Publishers kept per (model, region): the most recently heard. A mean over
/// eight is a region's demand; more would only be a peer's lever on memory.
pub(crate) const MOST_PUBLISHERS_A_KEY: usize = 8;

/// Keys kept in all — the bound `region_demand` has always had.
pub(crate) const MOST_KEYS: usize = 10_000;

/// The weight a blend gives the new mean after `elapsed_ms` without one.
pub(crate) fn blend_weight(elapsed_ms: u64) -> f64 {
    1.0 - RETAINED_PER_STEP.powf(elapsed_ms as f64 / STEP_MS)
}

#[derive(Default)]
struct Heard {
    /// Each publisher's latest rate and when it arrived (ms since the epoch).
    by_publisher: HashMap<NodeId, (f64, u64)>,
    /// When this key was last blended into `region_demand` from gossip.
    blended_at_ms: Option<u64>,
}

/// What a message adds: the mean of the fresh publishers' rates, and the
/// weight to blend it with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Blend {
    pub mean: f64,
    pub weight: f64,
}

#[derive(Default)]
pub struct DemandHeard {
    keys: DashMap<(ModelId, String), Heard>,
}

impl DemandHeard {
    /// Notes `publisher`'s `rate` for `key` and returns what to blend; `None`
    /// when the table is full of keys still being heard and this one is new.
    /// The caller has checked the rate is finite and the publisher is the
    /// authenticated sender.
    pub(crate) fn note(
        &self,
        key: (ModelId, String),
        publisher: NodeId,
        rate: f64,
        now_ms: u64,
    ) -> Option<Blend> {
        if !self.keys.contains_key(&key) && self.keys.len() >= MOST_KEYS {
            self.keys.retain(|_, heard| {
                heard
                    .by_publisher
                    .values()
                    .any(|(_, at)| now_ms.saturating_sub(*at) < HEARD_FRESH_FOR_MS)
            });
            if self.keys.len() >= MOST_KEYS {
                return None;
            }
        }
        let mut heard = self.keys.entry(key).or_default();
        heard
            .by_publisher
            .retain(|_, (_, at)| now_ms.saturating_sub(*at) < HEARD_FRESH_FOR_MS);
        if !heard.by_publisher.contains_key(&publisher)
            && heard.by_publisher.len() >= MOST_PUBLISHERS_A_KEY
        {
            let longest_quiet = heard
                .by_publisher
                .iter()
                .min_by_key(|(_, (_, at))| *at)
                .map(|(id, _)| id.clone());
            if let Some(id) = longest_quiet {
                heard.by_publisher.remove(&id);
            }
        }
        heard.by_publisher.insert(publisher, (rate, now_ms));
        let mean = heard.by_publisher.values().map(|(r, _)| r).sum::<f64>()
            / heard.by_publisher.len() as f64;
        // A key never blended from gossip is blended as one old-cadence message
        // was: a computer's own measurement of its own region is not replaced
        // by the first peer it hears.
        let weight = blend_weight(
            heard
                .blended_at_ms
                .map_or(STEP_MS as u64, |at| now_ms.saturating_sub(at)),
        );
        heard.blended_at_ms = Some(now_ms);
        Some(Blend { mean, weight })
    }
}

impl super::SharedState {
    /// The one way a peer's demand message reaches `region_demand`: the
    /// publisher's latest rate is kept, and the mean of the fresh ones is
    /// blended in by the time since the key's last blend. Returns the region's
    /// new figure, or `None` when the message was not taken (the tables are
    /// full).
    pub(crate) fn note_region_demand_heard(
        &self,
        model_id: &ModelId,
        region: &str,
        publisher: &NodeId,
        rate: f64,
        now_ms: u64,
    ) -> Option<f64> {
        let key = (model_id.clone(), region.to_string());
        if self.region_demand.len() >= MOST_KEYS && !self.region_demand.contains_key(&key) {
            return None;
        }
        let blend = self
            .models
            .demand_heard
            .note(key.clone(), publisher.clone(), rate, now_ms)?;
        // SEC: a stale NaN in the merged map (a pre-R102 rehydrate, a decay
        // race) would poison every later blend: `NaN * x = NaN`. Gotcha #98.
        let existing = self.region_demand.get(&key).map(|v| *v);
        let new_rate = match existing {
            Some(prev) => {
                let prev = if prev.is_finite() { prev } else { 0.0 };
                prev + (blend.mean - prev) * blend.weight
            }
            None => blend.mean,
        };
        self.region_demand.insert(key, new_rate);
        Some(new_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn key() -> (ModelId, String) {
        (ModelId("m".into()), "AU".into())
    }

    #[test]
    fn the_old_cadence_keeps_its_old_weight() {
        assert!((blend_weight(30_000) - 0.2).abs() < 1e-12);
        assert!((blend_weight(60_000) - 0.36).abs() < 1e-12);
        assert!(blend_weight(600_000) > 0.98);
        assert_eq!(blend_weight(0), 0.0);
    }

    #[test]
    fn a_regions_demand_is_the_mean_of_its_publishers_whoever_spoke_last() {
        let heard = DemandHeard::default();
        let t = 1_000_000;
        heard.note(key(), node(1), 1.0, t);
        let b = heard.note(key(), node(2), 3.0, t + 300_000).unwrap();
        assert_eq!(b.mean, 2.0);
        // Node 1 speaks again with the same rate: still the mean of both.
        let b = heard.note(key(), node(1), 1.0, t + 600_000).unwrap();
        assert_eq!(b.mean, 2.0);
    }

    #[test]
    fn a_publisher_that_stopped_drops_out_of_the_mean() {
        let heard = DemandHeard::default();
        let t = 1_000_000;
        heard.note(key(), node(1), 5.0, t);
        let b = heard
            .note(key(), node(2), 1.0, t + HEARD_FRESH_FOR_MS)
            .unwrap();
        assert_eq!(b.mean, 1.0, "node 1 was last heard a fresh-window ago");
    }

    #[test]
    fn a_key_holds_at_most_its_publishers_and_the_longest_quiet_goes() {
        let heard = DemandHeard::default();
        let t = 1_000_000;
        for i in 0..MOST_PUBLISHERS_A_KEY as u8 {
            heard.note(key(), node(i), 1.0, t + u64::from(i));
        }
        // A newcomer at 9.0 replaces node 0 (heard first): mean of seven 1.0s and 9.0.
        let b = heard.note(key(), node(200), 9.0, t + 100).unwrap();
        let n = MOST_PUBLISHERS_A_KEY as f64;
        assert!((b.mean - ((n - 1.0) + 9.0) / n).abs() < 1e-12);
        assert_eq!(
            heard.keys.get(&key()).unwrap().by_publisher.len(),
            MOST_PUBLISHERS_A_KEY
        );
        assert!(!heard
            .keys
            .get(&key())
            .unwrap()
            .by_publisher
            .contains_key(&node(0)));
    }

    /// The point of the change: a region's figure follows its publishers in
    /// TIME, whatever their cadence. Ten minutes of one publisher at 4.0 over a
    /// figure of 0 reaches the same place sent every 30 s (the old cadence) as
    /// sent once at the end (the new one). Weighting per message instead (the
    /// old 0.2) leaves the once-sent figure at 0.8.
    #[test]
    fn ten_minutes_of_demand_reach_the_same_figure_whatever_the_cadence() {
        let run = |every_ms: u64| {
            let heard = DemandHeard::default();
            let mut figure = 0.0_f64;
            // The key was blended at t=0 (to a figure of 0).
            heard.note(key(), node(9), 0.0, 0);
            heard.keys.get_mut(&key()).unwrap().by_publisher.clear();
            let mut t = 0;
            while t < 600_000 {
                t += every_ms;
                let b = heard.note(key(), node(1), 4.0, t).unwrap();
                figure += (b.mean - figure) * b.weight;
            }
            figure
        };
        let dense = run(30_000);
        let sparse = run(600_000);
        assert!(
            (dense - sparse).abs() < 1e-9,
            "dense {dense} vs sparse {sparse}"
        );
        assert!(
            sparse > 3.9,
            "ten minutes nearly reach the publisher's rate: {sparse}"
        );
    }

    #[test]
    fn a_full_table_makes_room_from_keys_nobody_is_still_heard_on() {
        let heard = DemandHeard::default();
        for i in 0..MOST_KEYS {
            heard.note((ModelId(format!("m{i}")), "AU".into()), node(1), 1.0, 0);
        }
        let fresh_key = (ModelId("new".into()), "AU".into());
        assert!(
            heard.note(fresh_key.clone(), node(1), 1.0, 1_000).is_none(),
            "all still heard"
        );
        assert!(heard
            .note(fresh_key, node(1), 1.0, HEARD_FRESH_FOR_MS + 1)
            .is_some());
        assert_eq!(heard.keys.len(), 1);
    }
}
