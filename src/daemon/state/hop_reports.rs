//! The chains this node drove, each waiting for its hops' own reports of how
//! long they held the reply's decode steps (#427, `SwarmMessage::HopTimings`).
//!
//! A chained hop's answer goes on to the next hop, and the tail's to this node,
//! which did not send the tail its forward: no hop's own time comes back with
//! an answer, so a chain was priced from its peers' stated speeds or from
//! prompt passes that hold none of a visit's fixed cost (545 ms predicted a
//! pass against ~600 measured). Each hop now tells this node once the reply is
//! over, in answer to its `CancelInference` (`network::manager::hop_timings`).
//!
//! **A report is believed only as an answer to a question this node asked**:
//! from a peer that ran a segment of that request's LAST plan here, of the
//! width it ran, once, within [`WAIT_FOR`] of the reply's end — never more
//! than the reply's whole time over its rounds, and over at least
//! [`LEAST_STEPS`] steps. What it may move is that peer's own price, and only
//! slower than its stated speed (`ModelSpeed::note_held_steps`).
//!
//! Bounded: [`MOST_REQUESTS`] replies at once, the oldest dropped first.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::{HopTimings, ModelId, NodeId};

/// Replies waiting for reports at once.
pub(crate) const MOST_REQUESTS: usize = 256;

/// How long after a reply ends its hops' reports are taken — they answer the
/// cancel this node sends as it ends, within a round trip or two.
pub(crate) const WAIT_FOR: Duration = Duration::from_secs(120);

/// The fewest steps a report's median may stand on.
pub(crate) const LEAST_STEPS: u32 = 4;

struct Awaited {
    model: ModelId,
    /// (peer, layers of its segment) — each taken once.
    hops: Vec<(NodeId, u32)>,
    /// The reply's whole time over its rounds: no step held one hop longer.
    round_us_at_most: u64,
    layer_work: f32,
    until: Instant,
}

/// Lives on `state.metrics`. Written ONLY by `PipelineExecutor::execute`
/// (`SharedState::expect_hop_reports`), taken ONLY by dispatch's `HopTimings`
/// arm (`SharedState::note_hop_report`).
#[derive(Default)]
pub struct HopReportsAwaited {
    map: parking_lot::Mutex<HashMap<uuid::Uuid, Awaited>>,
}

/// What a report came to.
#[derive(Debug, PartialEq)]
pub(crate) enum HopReport {
    /// Price `model` on the reporting peer at this many ms a layer.
    Priced {
        model: ModelId,
        ms_per_layer: f32,
        layer_work: f32,
    },
    /// Not an answer to anything this node asked (or asked already).
    Unasked,
    /// An answer, but not one a step could have taken.
    Implausible(&'static str),
}

impl HopReportsAwaited {
    /// A reply driven here ended: its plan's peer segments may report.
    pub(crate) fn expect(
        &self,
        request_id: uuid::Uuid,
        model: ModelId,
        hops: Vec<(NodeId, u32)>,
        round_us_at_most: u64,
        layer_work: f32,
        now: Instant,
    ) {
        let hops: Vec<_> = hops.into_iter().filter(|(_, layers)| *layers > 0).collect();
        if hops.is_empty() || round_us_at_most == 0 {
            return;
        }
        let mut map = self.map.lock();
        map.retain(|_, a| a.until > now);
        if !map.contains_key(&request_id) && map.len() >= MOST_REQUESTS {
            let oldest = map.iter().min_by_key(|(_, a)| a.until).map(|(id, _)| *id);
            if let Some(oldest) = oldest {
                map.remove(&oldest);
            }
        }
        map.insert(
            request_id,
            Awaited {
                model,
                hops,
                round_us_at_most,
                layer_work,
                until: now + WAIT_FOR,
            },
        );
    }

    /// `from`'s report: what it comes to. Taken once, whatever it says.
    pub(crate) fn take(&self, from: &NodeId, report: &HopTimings, now: Instant) -> HopReport {
        let mut map = self.map.lock();
        let Some(awaited) = map.get_mut(&report.request_id).filter(|a| a.until > now) else {
            return HopReport::Unasked;
        };
        let Some(at) = awaited
            .hops
            .iter()
            .position(|(node, layers)| node == from && *layers == report.layers)
        else {
            return HopReport::Unasked;
        };
        let (_, layers) = awaited.hops.swap_remove(at);
        let (model, layer_work, round_us_at_most) = (
            awaited.model.clone(),
            awaited.layer_work,
            awaited.round_us_at_most,
        );
        if awaited.hops.is_empty() {
            map.remove(&report.request_id);
        }
        drop(map);
        if report.steps < LEAST_STEPS {
            return HopReport::Implausible("too few steps");
        }
        if report.median_held_us == 0 || u64::from(report.median_held_us) > round_us_at_most {
            return HopReport::Implausible("longer than a round took");
        }
        HopReport::Priced {
            model,
            ms_per_layer: report.median_held_us as f32 / 1000.0 / layers as f32,
            layer_work,
        }
    }

    #[cfg(test)]
    fn waiting(&self) -> usize {
        self.map.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn report(request_id: uuid::Uuid, layers: u32, steps: u32, us: u32) -> HopTimings {
        HopTimings {
            request_id,
            layers,
            steps,
            median_held_us: us,
        }
    }

    /// A hop of the plan, of the width it ran, is believed once; a peer that
    /// was not in it, or names another width, is not.
    #[test]
    fn only_a_hop_of_the_plan_is_believed_and_once() {
        let awaited = HopReportsAwaited::default();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        let model = ModelId("m".into());
        awaited.expect(
            rid,
            model.clone(),
            vec![(node(1), 16), (node(2), 8)],
            300_000,
            1.5,
            t0,
        );
        assert_eq!(
            awaited.take(&node(9), &report(rid, 16, 50, 80_000), t0),
            HopReport::Unasked
        );
        assert_eq!(
            awaited.take(&node(1), &report(rid, 8, 50, 80_000), t0),
            HopReport::Unasked
        );
        assert_eq!(
            awaited.take(&node(1), &report(rid, 16, 50, 80_000), t0),
            HopReport::Priced {
                model,
                ms_per_layer: 5.0,
                layer_work: 1.5
            }
        );
        assert_eq!(
            awaited.take(&node(1), &report(rid, 16, 50, 80_000), t0),
            HopReport::Unasked
        );
        assert_eq!(awaited.waiting(), 1, "node 2 may still report");
        assert!(matches!(
            awaited.take(&node(2), &report(rid, 8, 50, 9_000), t0),
            HopReport::Priced { .. }
        ));
        assert_eq!(awaited.waiting(), 0, "every hop answered");
    }

    /// A held time longer than the reply's rounds took, or one over too few
    /// steps, is no price — and the hop's turn is spent.
    #[test]
    fn an_implausible_report_prices_nothing() {
        let awaited = HopReportsAwaited::default();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        awaited.expect(
            rid,
            ModelId("m".into()),
            vec![(node(1), 16), (node(2), 8)],
            300_000,
            1.0,
            t0,
        );
        assert_eq!(
            awaited.take(&node(1), &report(rid, 16, 50, 300_001), t0),
            HopReport::Implausible("longer than a round took")
        );
        assert_eq!(
            awaited.take(&node(1), &report(rid, 16, 50, 1_000), t0),
            HopReport::Unasked
        );
        assert_eq!(
            awaited.take(&node(2), &report(rid, 8, LEAST_STEPS - 1, 1_000), t0),
            HopReport::Implausible("too few steps")
        );
    }

    /// Through `SharedState`: a hop of a chain this node drove reports, and
    /// the scheduler's reading of that peer moves — the figure that is
    /// gossiped does not, and a peer that was not asked moves nothing.
    #[test]
    fn a_report_reaches_the_price_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut config = crate::config::Config::default();
        config.node.data_dir = dir.clone();
        let db = crate::storage::db::Database::open(&dir).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = crate::daemon::SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        let model = ModelId("chain-model".into());
        let rid = uuid::Uuid::new_v4();
        state.expect_hop_reports(
            rid,
            &model,
            vec![(node(1), 20), (node(2), 20)],
            50,
            Duration::from_secs(10),
        );
        state.note_hop_report(&node(3), &report(rid, 20, 50, 60_000));
        assert_eq!(state.observed_ranking(&node(3), &model), None, "not asked");
        state.note_hop_report(&node(1), &report(rid, 20, 50, 60_000));
        let r = state.observed_ranking(&node(1), &model).expect("priced");
        assert_eq!(
            r.timed_on,
            crate::daemon::state::TimedOn::HeldSteps { reports: 1 }
        );
        assert!((r.ms_per_layer - 3.0).abs() < 1e-3, "{}", r.ms_per_layer);
        assert_eq!(
            state.observed_latency_ms_per_layer(&node(1), &model),
            None,
            "the gossiped figure is what this node timed"
        );
    }

    /// Reports are taken only for a while after the reply, and only so many
    /// replies wait at once.
    #[test]
    fn the_wait_is_bounded_in_time_and_count() {
        let awaited = HopReportsAwaited::default();
        let t0 = Instant::now();
        let late = uuid::Uuid::new_v4();
        awaited.expect(
            late,
            ModelId("m".into()),
            vec![(node(1), 4)],
            100_000,
            1.0,
            t0,
        );
        assert_eq!(
            awaited.take(&node(1), &report(late, 4, 10, 1_000), t0 + WAIT_FOR),
            HopReport::Unasked
        );
        let first = uuid::Uuid::new_v4();
        awaited.expect(
            first,
            ModelId("m".into()),
            vec![(node(1), 4)],
            100_000,
            1.0,
            t0,
        );
        for i in 0..MOST_REQUESTS {
            awaited.expect(
                uuid::Uuid::new_v4(),
                ModelId("m".into()),
                vec![(node(1), 4)],
                100_000,
                1.0,
                t0 + Duration::from_millis(1 + i as u64),
            );
        }
        assert_eq!(awaited.waiting(), MOST_REQUESTS);
        assert_eq!(
            awaited.take(&node(1), &report(first, 4, 10, 1_000), t0),
            HopReport::Unasked,
            "the oldest went"
        );
    }
}
