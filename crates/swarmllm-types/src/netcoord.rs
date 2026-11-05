//! Vivaldi network coordinates — estimating the round trip between two peers
//! **neither of which is us**.
//!
//! The routing cost model prices each candidate by `NodeCandidate::latency_ms`,
//! which is OUR round trip to that peer. A pipeline's real cost includes the
//! A→B and B→C legs between the peers themselves, and nothing in the scheduler
//! could see them — so a chain of three peers in one city and a chain spanning
//! three continents were priced the same way. Measured 2026-09-20: a 4-segment
//! chain that crossed Thailand↔Italy three times per token ran at 0.35 tok/s
//! while the same split within 18 ms ran at 6.76.
//!
//! Vivaldi (Dabek et al., SIGCOMM'04) gives every node a coordinate such that
//! the distance between two coordinates predicts the RTT between those two
//! nodes. It is fully decentralised, needs no landmarks, and — the property
//! that matters here — it is fed by round trips a node **already measures**, so
//! adopting it costs no extra probing. The paper reports a median relative
//! error of 11% for the 2-D-plus-height model implemented below.
//!
//! # The model
//!
//! A 2-D Euclidean position models the Internet core, where latency is roughly
//! proportional to geographic distance, plus a **height** modelling the access
//! link from the node to that core — queuing on an oversubscribed line, a slow
//! last mile, or simply a long haul to the nearest exchange. Height is not a
//! third dimension: a packet climbs the sender's height, crosses the plane, and
//! descends the receiver's height, so the two heights **add** where the planar
//! coordinates subtract. The paper found this beats both 2-D and 3-D Euclidean.
//!
//! # What this is not
//!
//! ⚠ A coordinate system is poor at picking the single *closest* node — the
//! Azureus study found exactly that, and Pharos answers it with a second,
//! local-cluster coordinate. We do not need that: the routing question is
//! whether a hop costs 20 ms or 600 ms, a distinction this predicts well. If
//! fine ordering is ever needed, add the local tier rather than sharpening this.

use serde::{Deserialize, Serialize};

/// How long a round-trip sample stays eligible to be the window's minimum.
///
/// Bounded in TIME, not just in count, so the estimate can RISE again when a
/// path genuinely degrades. An all-time minimum would latch the best moment the
/// network ever had and never let go.
pub const LATENCY_WINDOW_MS: u64 = 300_000;

/// Hard cap on retained samples per peer, so a chatty peer cannot grow the
/// window without bound between evictions.
pub const LATENCY_WINDOW_MAX_SAMPLES: usize = 64;

/// Never age a sample out while this few remain, however old it is.
///
/// **Found 2026-09-21, from a live reading**: every peer on the release node
/// reported `rtt_samples: 3`, against a cap of 64. The only sample source that
/// runs regardless of load is the PEX ping at `RR_PING_INTERVAL_SECS` (120 s,
/// `network/manager/mod.rs`), so a merely-connected peer can put **at most 3**
/// samples in a 5-minute window — the other source, an acknowledged tensor
/// forward, exists only while this node is serving distributed work.
///
/// A minimum over 3 draws is not the filter documented on [`LatencyFilter`].
/// On the real sample kept in this file's tests, **10 of 14 observations are in
/// the slow mode**, so three draws miss the fast mode outright about a third of
/// the time and two draws about half — and what Vivaldi is then taught as the
/// distance to that peer is the remote node's own event-loop delay. The two
/// constants were each reasonable and were never read against each other; the
/// window was sized for a stream only a busy node produces.
///
/// The cost is paid only on a QUIET link, and it is a slower reaction to
/// genuine degradation: the estimate rises once this many newer samples exist,
/// ~16 min at the ping rate rather than 5. At that rate there is nothing better
/// to be had — an estimate needs samples. On a busy link the window fills many
/// times over and this floor never binds.
pub const LATENCY_WINDOW_MIN_SAMPLES: usize = 8;

/// Age past which a sample is dropped whatever the floor says.
///
/// The floor exists so a quiet peer still has an estimate, not so a peer that
/// went silent for an hour can answer with the minimum it had back then.
/// Without this, a peer that came back WORSE would report its old best until
/// the floor had been refilled one ping at a time; with it, the first new
/// sample flushes everything stale at once.
pub const LATENCY_SAMPLE_MAX_AGE_MS: u64 = 1_800_000;

/// The round trips seen to ONE peer recently, answering with the **minimum**.
///
/// # Why a minimum, when the reference implementation uses a median
///
/// HashiCorp's Serf/Consul — the most battle-tested Vivaldi deployment — keeps
/// `LatencyFilterSamples` per node and takes their MEDIAN. That is right for
/// their input: a lightweight UDP gossip probe, where noise is modest and
/// roughly symmetric, and a median rejects the occasional outlier without
/// biasing the estimate.
///
/// **Our input is not that.** The round trip we can measure is an
/// application-level request/response that queues behind the node's own event
/// loop, and measured on this fleet 2026-09-20 the contamination was large and
/// one-sided: against a peer whose ICMP round trip was 0.563-1.539 ms, the
/// application figure came back **bimodal — 3-8 ms or 118-158 ms**, with 10 of
/// 14 samples in the slow mode. A median of that is 120 ms. **The median would
/// be the contamination**, faithfully encoded as distance, because a median
/// filter assumes most samples are near-clean and here most are not.
///
/// A minimum is right whenever the corruption only ever ADDS, which queueing
/// and scheduling delay do: the smallest round trip recently observed is the
/// best available estimate of the propagation delay underneath them. It is the
/// same argument BBR makes for min-RTT, and the same "min-of-N" discipline this
/// repo already applies to benchmarking (gotcha #367).
///
/// ⚠ **If the measurement source is ever changed to a true network-level round
/// trip, revisit this** — with a clean signal the median becomes the better
/// estimator again, because a minimum over clean samples chases the low tail.
#[derive(Clone, Debug, Default)]
pub struct LatencyFilter {
    /// `(observed_at_ms, rtt_ms)`, oldest first.
    samples: std::collections::VecDeque<(u64, f32)>,
}

impl LatencyFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sample and return the window's minimum — the figure to feed
    /// the coordinate. `None` when the sample carries no information.
    ///
    /// `now_ms` is passed in rather than read from a clock so this stays pure
    /// and its window behaviour is testable without sleeping.
    pub fn observe(&mut self, now_ms: u64, rtt_ms: f32) -> Option<f32> {
        if !(rtt_ms.is_finite() && rtt_ms > 0.0) {
            return self.min();
        }
        self.samples.push_back((now_ms, rtt_ms));
        // Too old to describe the path at all, floor or no floor.
        let hard_cutoff = now_ms.saturating_sub(LATENCY_SAMPLE_MAX_AGE_MS);
        while self
            .samples
            .front()
            .is_some_and(|(at, _)| *at < hard_cutoff)
        {
            self.samples.pop_front();
        }
        // Then the ordinary window — but never down to a count too small to be
        // a minimum of anything. See `LATENCY_WINDOW_MIN_SAMPLES`.
        let cutoff = now_ms.saturating_sub(LATENCY_WINDOW_MS);
        while self.samples.len() > LATENCY_WINDOW_MIN_SAMPLES
            && self.samples.front().is_some_and(|(at, _)| *at < cutoff)
        {
            self.samples.pop_front();
        }
        while self.samples.len() > LATENCY_WINDOW_MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.min()
    }

    /// The smallest round trip still in the window.
    pub fn min(&self) -> Option<f32> {
        self.samples
            .iter()
            .map(|(_, ms)| *ms)
            .fold(None, |acc: Option<f32>, ms| {
                Some(acc.map_or(ms, |a| a.min(ms)))
            })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Vivaldi's `c_c`: how far a node moves toward where a sample says it belongs.
/// 0.25 is the paper's value — large enough to converge in a few samples, small
/// enough that one bad measurement cannot throw the coordinate across the map.
const C_C: f32 = 0.25;

/// Vivaldi's `c_e`: how fast the local error estimate tracks observed error.
const C_E: f32 = 0.25;

/// Height floor. The paper requires a strictly positive height so it can always
/// be scaled up or down; at exactly zero a node can never climb again, because
/// every update multiplies it.
const MIN_HEIGHT_MS: f32 = 0.01;

/// A fresh coordinate knows nothing, and says so. Error is a RELATIVE figure, so
/// 1.0 means "expect this to be as wrong as the value itself" — which makes the
/// sample weight `w` favour the better-informed party until this node settles.
const MAX_ERROR: f32 = 1.0;

/// Coordinates beyond this are nonsense on any real network and are almost
/// certainly a runaway from bad samples. Clamping keeps one pathological peer
/// from dragging a node somewhere it can never walk back from.
const MAX_COORD_MS: f32 = 60_000.0;

/// A node's position in the latency space, as published to peers.
///
/// Wire-carried inside [`crate::node::NodeCapability`], so every field is
/// `#[serde(default)]`-friendly and the whole struct is optional there: a node
/// that has never published one is simply priced the old way.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkCoord {
    /// Planar position, milliseconds.
    pub x: f32,
    pub y: f32,
    /// Access-link cost, milliseconds. Always `>= MIN_HEIGHT_MS`.
    pub height: f32,
    /// This node's own estimate of how wrong its coordinate is, RELATIVE — 0.0
    /// is perfect, 1.0 is "no better than a guess". Published so the receiver
    /// can weigh a sample from us against what it already believes, which is
    /// what stops a freshly-joined node dragging settled ones around.
    pub error: f32,
}

impl Default for NetworkCoord {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkCoord {
    /// A node that has measured nothing: at the origin, minimum height, and
    /// honest about knowing nothing.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            height: MIN_HEIGHT_MS,
            error: MAX_ERROR,
        }
    }

    /// Predicted round trip to `other`, in milliseconds.
    ///
    /// Planar distance plus BOTH heights — a packet climbs out of this node's
    /// access link and descends into the other's, so heights add even when the
    /// two nodes sit at the same planar point.
    pub fn distance_ms(&self, other: &NetworkCoord) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt() + self.height + other.height
    }

    /// Has this coordinate seen enough to be worth believing?
    ///
    /// A caller deciding a route should fall back to whatever it did before
    /// rather than act on a coordinate still at its starting error. The
    /// threshold is deliberately generous — Vivaldi's own reported median is
    /// 0.11, so 0.5 admits coordinates that are rough but still separate "same
    /// city" from "other hemisphere", which is the only question asked of it.
    ///
    /// It must also be a coordinate this code could have produced: a height
    /// under `MIN_HEIGHT_MS` or a position past `MAX_COORD_MS` never comes out
    /// of [`NetworkCoord::observe`], and a peer publishing a negative height
    /// would make every hop to it look cheaper than free — a router or a
    /// chain planner would then prefer it for nothing (#318).
    pub fn is_usable(&self) -> bool {
        self.error < 0.5
            && self.x.is_finite()
            && self.y.is_finite()
            && self.height.is_finite()
            && self.height >= MIN_HEIGHT_MS
            && self.x.abs() <= MAX_COORD_MS
            && self.y.abs() <= MAX_COORD_MS
    }

    /// Fold in one measured round trip to a peer whose coordinate we know.
    ///
    /// This is Figure 3 of the Vivaldi paper — the adaptive-timestep form,
    /// which is the one to use: with a constant timestep, settled nodes place
    /// "too much faith in young high-error nodes" and a wave of joiners
    /// destroys the existing structure. The weight `w` is what prevents that.
    ///
    /// ```text
    /// w   = e_i / (e_i + e_j)
    /// e_s = |‖x_i − x_j‖ − rtt| / rtt
    /// e_i = e_s·c_e·w + e_i·(1 − c_e·w)
    /// x_i = x_i + c_c·w·(rtt − ‖x_i − x_j‖)·u(x_i − x_j)
    /// ```
    ///
    /// `rtt_ms` must be a real measurement; a zero or negative sample carries no
    /// information (the relative error would divide by it) and is dropped.
    pub fn observe(&mut self, rtt_ms: f32, remote: &NetworkCoord) {
        if !(rtt_ms.is_finite() && rtt_ms > 0.0) {
            return;
        }
        if !(remote.x.is_finite() && remote.y.is_finite() && remote.height.is_finite()) {
            return;
        }
        let remote_err = remote.error.clamp(0.0, MAX_ERROR);

        // Sample weight: how much of this disagreement is likely OURS. Two
        // nodes that both know nothing split it evenly rather than dividing by
        // zero.
        let err_sum = self.error + remote_err;
        let w = if err_sum > 0.0 {
            self.error / err_sum
        } else {
            0.5
        };

        let predicted = self.distance_ms(remote);
        let e_s = ((predicted - rtt_ms).abs()) / rtt_ms;

        // Weighted moving average of our own error.
        self.error = (e_s * C_E * w + self.error * (1.0 - C_E * w)).clamp(0.0, MAX_ERROR);

        // Direction to push. `u(0)` must be a unit vector in SOME direction or
        // two nodes at the same point can never separate — the paper says
        // "randomly chosen". We derive it from the sample instead of drawing a
        // random number: it separates them just as well, it keeps this function
        // pure and testable, and different pairs produce different samples so
        // nodes do not all march the same way.
        let dx = self.x - remote.x;
        let dy = self.y - remote.y;
        let planar = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = if planar > f32::EPSILON {
            (dx / planar, dy / planar)
        } else {
            let angle = (rtt_ms.to_bits() % 628) as f32 / 100.0;
            (angle.cos(), angle.sin())
        };

        let force = C_C * w * (rtt_ms - predicted);
        self.x = (self.x + force * ux).clamp(-MAX_COORD_MS, MAX_COORD_MS);
        self.y = (self.y + force * uy).clamp(-MAX_COORD_MS, MAX_COORD_MS);

        // Height moves under the SAME update, because the paper redefines the
        // vector operations rather than adding a special case: the difference
        // of two height vectors ADDS their heights, `[x,xh] − [y,yh] =
        // ((x−y), xh+yh)`, so the unit vector's height component is
        // `(xh + yh) / ‖diff‖` — both heights, not just ours. Using only ours
        // pinned every node to the floor, because a node starting at the
        // minimum has almost no height to scale (caught by
        // `a_slow_access_link_lands_in_height_not_position`).
        //
        // This is what lets a node too far from EVERYONE — the signature of a
        // slow access link, where the planar forces cancel out — rise off the
        // plane instead of sitting still.
        let norm = (planar + self.height + remote.height).max(1e-6);
        let height_force = force * ((self.height + remote.height) / norm);
        self.height = (self.height + height_force).max(MIN_HEIGHT_MS);
        if !self.height.is_finite() {
            self.height = MIN_HEIGHT_MS;
        }
    }

    /// How far this coordinate is from `other` as a POSITION, in milliseconds:
    /// planar distance plus the change in height. Not [`Self::distance_ms`],
    /// which adds both heights because it predicts a round trip between two
    /// nodes; a node compared with its own earlier position has one height.
    pub fn moved_from(&self, other: &NetworkCoord) -> f32 {
        (self.x - other.x).hypot(self.y - other.y) + (self.height - other.height).abs()
    }

    fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.height.is_finite()
            && self.error.is_finite()
    }
}

/// How many of this node's own coordinates each window of [`UpdateFilter`]
/// holds. Ledlie, Pietzuch and Seltzer ("Stable and Accurate Network
/// Coordinates", ICDCS 2006, § 5) found every window from 32 to 512 improved
/// both stability and accuracy; 32 is the smallest of them and the one they
/// ran on PlanetLab. At the PEX ping's rate (one per peer every 2 min) seven
/// peers fill it in ~9 minutes; a split's acknowledged forwards, in seconds.
pub const UPDATE_WINDOW: usize = 32;

/// How far the recent centroid must have moved from the earlier one, as a
/// fraction of the predicted round trip to the nearest neighbour, before the
/// published coordinate follows — the paper's RELATIVE heuristic and its
/// threshold: accuracy began to fall past 0.3, the most conservative value
/// that still bought stability.
pub const UPDATE_RELATIVE_THRESHOLD: f32 = 0.3;

/// The smallest move ever worth publishing, whatever the neighbour's distance:
/// a coordinate learned from application-level round trips cannot place a
/// neighbour 2 ms away to within 0.6 ms. Also the capability's own deadband
/// for the coordinate (`health::monitor::capability_news`), which reads it
/// from here.
pub const UPDATE_MIN_MOVE_MS: f32 = 5.0;

/// How many of the newest coordinates say where the node now IS, whenever the
/// published coordinate changes.
///
/// A change fires as soon as enough of the recent window has moved, so that
/// window's centroid still holds positions from before the move, and a node's
/// first window holds its coordinate converging from wherever it began. The
/// paper publishes the whole window's centroid; on the live node (2026-11-01,
/// ~2 samples a minute, minutes 16-25 after a start) that held the published
/// coordinate's predictions at a 10% median error beside the live
/// coordinate's 5%. The newest quarter is past the move; whether there WAS a
/// move is still decided on whole windows, so a few odd samples cannot fire it.
pub const UPDATE_SETTLED_FROM_NEWEST: usize = UPDATE_WINDOW / 4;

/// The coordinate this node PUBLISHES, kept apart from the one Vivaldi moves on
/// every sample (#91).
///
/// A Vivaldi coordinate never stops moving: each sample nudges it, and a node
/// that has settled orbits a point rather than resting on it. Published as it
/// stands, every orbit past a fixed deadband is news, and on 2026-10-31 16 of
/// a live node's 18 capability broadcasts in 18 minutes were that — each one a
/// whole capability (~5 KB, and ~12 KB from a peer listing ~300 parts)
/// forwarded across the mesh.
/// A fixed threshold cannot fix it: "a single threshold ('react if moved more
/// than 50ms') did not work for all nodes because the volume through which
/// each coordinate moved was node-dependent" (Ledlie, Gardner and Seltzer,
/// "Network Coordinates in the Wild", NSDI 2007, § 4.1), and the ICDCS paper
/// measured the windowless thresholds as a trade with a narrow sweet spot.
///
/// So this is the paper's RELATIVE heuristic, which separates a coordinate
/// orbiting one point from one that has MOVED: a start window of the first
/// [`UPDATE_WINDOW`] coordinates after the last change, frozen once full, and a
/// window of the latest ones sliding behind it (Ben-David, Gehrke and Kifer's
/// change detection). When the centroids differ by more than
/// [`UPDATE_RELATIVE_THRESHOLD`] of the round trip to the nearest neighbour,
/// the centroid of the newest [`UPDATE_SETTLED_FROM_NEWEST`] is published and
/// both windows start again; the first full window after that re-centres it
/// when it has moved past [`UPDATE_MIN_MOVE_MS`] since. Whole windows decide
/// WHEN the published coordinate changes; the newest quarter says WHERE to.
/// Until the first window fills, the coordinate as it stands is published
/// whenever it has moved past the same threshold since it was last published,
/// or has become usable (or stopped being) since — from the first sample on,
/// since a node starting up must still publish, however rough
/// (`a_brand_new_node_still_publishes_its_coordinate`) — and the
/// first full window counts as a change: it holds the coordinate's convergence
/// from wherever it started, not a place it settled.
///
/// What a peer TIMES against us is our coordinate as it stands when it asks —
/// the peer-exchange answer carries that one (`PeerExchangeResponse::coord`).
/// This one is for everything else: the scheduler's distance between two
/// peers, and an older peer's own updates.
#[derive(Clone, Debug, Default)]
pub struct UpdateFilter {
    start: Vec<NetworkCoord>,
    recent: std::collections::VecDeque<NetworkCoord>,
    followed: Option<NetworkCoord>,
    settled: Option<NetworkCoord>,
    nearest: Option<Neighbour>,
}

/// The neighbour a move is measured against: the peer with the smallest
/// windowed round trip, as each sample teaches it (the paper's "approximate
/// nearest neighbor is learned through a comparison with each latency sample").
#[derive(Clone, Debug)]
struct Neighbour {
    id: crate::NodeId,
    rtt_ms: f32,
    coord: NetworkCoord,
    at_ms: u64,
}

/// What a change point published, for the log that proves one fired.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Migration {
    /// How far the recent centroid was from the earlier one.
    pub moved_ms: f32,
    /// The move it had to exceed.
    pub threshold_ms: f32,
}

impl UpdateFilter {
    /// The coordinate to publish: the settled one once a window has filled;
    /// before that, the coordinate as it stood when it last moved past the
    /// threshold — from the first sample on.
    pub fn published(&self) -> Option<NetworkCoord> {
        self.settled.or(self.followed)
    }

    /// The move that counts, judged from `at`: [`UPDATE_RELATIVE_THRESHOLD`]
    /// of the round trip from there to the nearest neighbour, never under
    /// [`UPDATE_MIN_MOVE_MS`].
    fn threshold_ms(&self, at: &NetworkCoord) -> f32 {
        let scale = self
            .nearest
            .as_ref()
            .map_or(0.0, |n| at.distance_ms(&n.coord));
        (UPDATE_RELATIVE_THRESHOLD * scale).max(UPDATE_MIN_MOVE_MS)
    }

    /// Note that a sample to `peer` — its windowed minimum `rtt_ms`, its
    /// coordinate `theirs`, at `at_ms` — has moved this node's coordinate to
    /// `now`. Returns what was published when the sample completed a change.
    pub fn note(
        &mut self,
        now: NetworkCoord,
        peer: &crate::NodeId,
        rtt_ms: f32,
        theirs: &NetworkCoord,
        at_ms: u64,
    ) -> Option<Migration> {
        if !now.is_finite() {
            return None;
        }
        if rtt_ms.is_finite() && rtt_ms > 0.0 && theirs.is_finite() {
            let replace = match &self.nearest {
                None => true,
                Some(n) => {
                    n.id == *peer
                        || rtt_ms < n.rtt_ms
                        || at_ms.saturating_sub(n.at_ms) > LATENCY_SAMPLE_MAX_AGE_MS
                }
            };
            if replace {
                self.nearest = Some(Neighbour {
                    id: peer.clone(),
                    rtt_ms,
                    coord: *theirs,
                    at_ms,
                });
            }
        }

        if self.settled.is_none() {
            // Before the first window fills there is no centroid to compare, so
            // the coordinate as it stands is published when it has moved past
            // the same threshold — the paper measured only after start-up
            // effects and says nothing of this phase. Following every sample
            // cost ~14 broadcasts in the 16 minutes a live node with 4
            // coordinate-publishing peers took to fill its first window
            // (2026-11-01).
            //
            // …or when a reader's verdict on it changed: routing acts only on a
            // coordinate that `is_usable`, and a starting node's error can fall
            // under the bar on a sample that moves it less than the threshold —
            // held, it was published as unusable until its first window filled
            // (`a_node_starting_up_is_published_usable_as_soon_as_it_is`).
            let moved = self.followed.is_none_or(|f| {
                now.moved_from(&f) > self.threshold_ms(&f) || now.is_usable() != f.is_usable()
            });
            if moved {
                self.followed = Some(now);
            }
        }
        let mut refilled = false;
        if self.start.len() < UPDATE_WINDOW {
            self.start.push(now);
            refilled = self.start.len() == UPDATE_WINDOW;
        }
        self.recent.push_back(now);
        if self.recent.len() > UPDATE_WINDOW {
            self.recent.pop_front();
        }
        // Both windows refill together after a change, so the recent one is
        // full exactly when the start one is.
        if self.recent.len() < UPDATE_WINDOW {
            return None;
        }
        let recent = centroid(self.recent.iter());
        if refilled {
            let Some(settled) = self.settled else {
                // The first full window since this node started holds its
                // coordinate converging from wherever it began (the live node
                // was still ~20 ms on its way after 16 minutes, 2026-11-01): a
                // change from nothing, so the newest quarter is published and a
                // start window frozen now would describe the trip, not a place.
                self.settled = Some(self.newest());
                self.start.clear();
                self.recent.clear();
                return None;
            };
            // The first full window since a change: publish where the node
            // SETTLED, if what was published then missed it. In the paper's
            // form (the whole recent window published) the windows then refill
            // identically and never differ again, and the published coordinate
            // stayed part of the way there for as long as the node stayed put;
            // a move still under way at the change does the same to the newest
            // quarter.
            let moved_ms = recent.moved_from(&settled);
            if moved_ms <= UPDATE_MIN_MOVE_MS {
                return None;
            }
            self.settled = Some(self.newest());
            return Some(Migration {
                moved_ms,
                threshold_ms: UPDATE_MIN_MOVE_MS,
            });
        }
        let start = centroid(self.start.iter());
        let threshold_ms = self.threshold_ms(&start);
        let moved_ms = recent.moved_from(&start);
        if moved_ms <= threshold_ms {
            return None;
        }
        self.settled = Some(self.newest());
        self.start.clear();
        self.recent.clear();
        Some(Migration {
            moved_ms,
            threshold_ms,
        })
    }

    /// The centroid of the newest [`UPDATE_SETTLED_FROM_NEWEST`] coordinates.
    fn newest(&self) -> NetworkCoord {
        let skip = self.recent.len().saturating_sub(UPDATE_SETTLED_FROM_NEWEST);
        centroid(self.recent.iter().skip(skip))
    }
}

/// The mean position (and error) of a non-empty window.
fn centroid<'a>(window: impl ExactSizeIterator<Item = &'a NetworkCoord>) -> NetworkCoord {
    let n = window.len().max(1) as f32;
    let (mut x, mut y, mut h, mut e) = (0.0, 0.0, 0.0, 0.0);
    for c in window {
        x += c.x;
        y += c.y;
        h += c.height;
        e += c.error;
    }
    NetworkCoord {
        x: x / n,
        y: y / n,
        height: (h / n).max(MIN_HEIGHT_MS),
        error: (e / n).clamp(0.0, MAX_ERROR),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a set of coordinates toward a known truth and report the worst
    /// relative prediction error left. The truth is a plain 2-D layout plus a
    /// per-node access cost, which is exactly the shape the model claims to
    /// fit — so this measures whether the ALGORITHM converges, not whether the
    /// Internet happens to be Euclidean.
    fn converge(truth: &[(f32, f32, f32)], rounds: usize) -> (Vec<NetworkCoord>, f32) {
        let true_rtt = |a: usize, b: usize| -> f32 {
            let (ax, ay, ah) = truth[a];
            let (bx, by, bh) = truth[b];
            ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() + ah + bh
        };
        let mut coords = vec![NetworkCoord::new(); truth.len()];
        for r in 0..rounds {
            for i in 0..truth.len() {
                for j in 0..truth.len() {
                    if i == j {
                        continue;
                    }
                    // Alternate direction so neither node is always the one
                    // being pushed; real gossip is symmetric over time.
                    let (a, b) = if r % 2 == 0 { (i, j) } else { (j, i) };
                    let remote = coords[b];
                    coords[a].observe(true_rtt(a, b), &remote);
                }
            }
        }
        let mut worst: f32 = 0.0;
        for i in 0..truth.len() {
            for j in 0..truth.len() {
                if i == j {
                    continue;
                }
                let t = true_rtt(i, j);
                let p = coords[i].distance_ms(&coords[j]);
                worst = worst.max((p - t).abs() / t);
            }
        }
        (coords, worst)
    }

    /// The 14 consecutive samples measured against the LAN peer on
    /// 2026-09-20, whose true round trip was 0.563-1.539 ms by ICMP. Kept
    /// verbatim because the SHAPE is the whole argument for a minimum: it is
    /// bimodal, and the slow mode is the majority.
    const FIELD_SAMPLES: [f32; 14] = [
        4.0, 118.0, 3.0, 3.0, 121.0, 132.0, 120.0, 123.0, 120.0, 120.0, 158.0, 132.0, 8.0, 125.0,
    ];

    #[test]
    fn the_filter_recovers_the_real_link_from_a_contaminated_majority() {
        let mut f = LatencyFilter::new();
        let mut last = None;
        for (i, s) in FIELD_SAMPLES.iter().enumerate() {
            last = f.observe(i as u64 * 1000, *s);
        }
        let got = last.expect("a minimum after 14 samples");
        assert_eq!(
            got, 3.0,
            "the window must answer with the real link (~1-3 ms by ICMP), not \
             the queueing delay that dominates the samples"
        );

        // The comparison that justifies diverging from Serf's median filter.
        let mut sorted = FIELD_SAMPLES;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = (sorted[6] + sorted[7]) / 2.0;
        assert!(
            median > 100.0,
            "sanity: the median of this real sample really is the \
             contamination ({median} ms), which is why a median filter is \
             wrong for THIS input"
        );
    }

    /// Bounded in time, so a path that genuinely gets worse is not masked for
    /// ever by one good moment.
    ///
    /// Takes enough newer samples to clear `LATENCY_WINDOW_MIN_SAMPLES`: the
    /// floor is what stops a quiet peer's window holding three draws, and the
    /// price of it is exactly this — the estimate rises after that many newer
    /// samples rather than at the window edge.
    #[test]
    fn a_stale_good_sample_stops_counting() {
        let mut f = LatencyFilter::new();
        assert_eq!(f.observe(0, 3.0), Some(3.0));
        assert_eq!(f.observe(1_000, 200.0), Some(3.0), "still in window");
        let mut last = None;
        for i in 0..=LATENCY_WINDOW_MIN_SAMPLES {
            last = f.observe(LATENCY_WINDOW_MS + 2_000 + i as u64 * 1_000, 200.0);
        }
        assert_eq!(
            last.expect("a value"),
            200.0,
            "once the good sample ages out the estimate must rise; an all-time \
             minimum would latch the best moment the network ever had"
        );
    }

    /// **The defect the floor exists for, at the cadence that produces it.**
    ///
    /// The PEX ping is the only sample source that runs regardless of load, at
    /// `RR_PING_INTERVAL_SECS` = 120 s, so a merely-connected peer offers three
    /// samples per 5-minute window — which is exactly what every peer on the
    /// release node reported on 2026-09-21 (`rtt_samples: 3`).
    ///
    /// Scored over EVERY prefix of the real sample rather than its end, because
    /// a single end-state is cherry-picking: `FIELD_SAMPLES` happens to finish
    /// beside a fast observation, so asserting on the last answer alone passes
    /// with the floor removed and proves nothing (gotcha #502).
    ///
    /// The floor does not make a slow answer impossible and this does not claim
    /// it does — the sample contains a run of eight consecutive slow
    /// observations, which no window of eight can see past. It makes it rare:
    /// **8 of 14 prefixes answer in the fast mode at three samples, 13 of 14
    /// with the floor.** Remove the floor and this goes red.
    #[test]
    fn a_quiet_peers_window_usually_finds_the_fast_mode() {
        const PING_INTERVAL_MS: u64 = 120_000;
        /// Above the measured fast mode (3-8 ms), far below the slow one
        /// (118-158 ms). Nothing in this sample lands between them.
        const FAST_MODE_CEILING_MS: f32 = 20.0;

        let mut f = LatencyFilter::new();
        let mut fast_answers = 0;
        for (i, s) in FIELD_SAMPLES.iter().enumerate() {
            let min = f
                .observe(i as u64 * PING_INTERVAL_MS, *s)
                .expect("a minimum after any valid sample");
            if min < FAST_MODE_CEILING_MS {
                fast_answers += 1;
            }
        }
        assert!(
            fast_answers >= 13,
            "at the real ping rate only {fast_answers} of {} prefixes answered \
             in the fast mode; three samples per window scores 8, so the \
             window is not holding what the floor promises",
            FIELD_SAMPLES.len()
        );
        assert!(
            f.len() >= LATENCY_WINDOW_MIN_SAMPLES,
            "a peer pinged every {}s must still hold {} samples, held {}",
            PING_INTERVAL_MS / 1000,
            LATENCY_WINDOW_MIN_SAMPLES,
            f.len()
        );
    }

    /// The floor keeps a quiet peer's estimate alive; it must not resurrect one
    /// from an hour ago. A peer that goes silent and comes back WORSE answers
    /// honestly on its first new sample, not after the floor is refilled.
    #[test]
    fn a_sample_too_old_to_mean_anything_goes_whatever_the_floor_says() {
        let mut f = LatencyFilter::new();
        for i in 0..LATENCY_WINDOW_MIN_SAMPLES {
            f.observe(i as u64 * 1_000, 3.0);
        }
        assert_eq!(f.min(), Some(3.0));

        let much_later = LATENCY_SAMPLE_MAX_AGE_MS + 60_000;
        assert_eq!(
            f.observe(much_later, 200.0),
            Some(200.0),
            "one sample after a long silence must flush the stale window, not \
             report the minimum the path had before it"
        );
        assert_eq!(f.len(), 1);
    }

    /// The floor must not defeat the cap: a busy peer is still bounded.
    #[test]
    fn the_floor_does_not_lift_the_hard_cap() {
        let mut f = LatencyFilter::new();
        for i in 0..(LATENCY_WINDOW_MAX_SAMPLES * 3) {
            f.observe(i as u64 * 10, 50.0);
        }
        assert!(f.len() <= LATENCY_WINDOW_MAX_SAMPLES);
    }

    #[test]
    fn the_window_is_bounded_and_ignores_junk() {
        let mut f = LatencyFilter::new();
        for i in 0..(LATENCY_WINDOW_MAX_SAMPLES * 2) {
            f.observe(i as u64, 50.0);
        }
        assert!(
            f.len() <= LATENCY_WINDOW_MAX_SAMPLES,
            "retained {} samples, cap is {}",
            f.len(),
            LATENCY_WINDOW_MAX_SAMPLES
        );
        let before = f.min();
        for bad in [0.0, -5.0, f32::NAN, f32::INFINITY] {
            f.observe(1_000, bad);
        }
        assert_eq!(before, f.min(), "a junk sample must not enter the window");
    }

    #[test]
    fn a_fresh_coordinate_admits_it_knows_nothing() {
        let c = NetworkCoord::new();
        assert_eq!(c.error, MAX_ERROR);
        assert!(
            !c.is_usable(),
            "a coordinate that has seen no samples must not be routed on"
        );
        assert!(
            c.height >= MIN_HEIGHT_MS,
            "height must start strictly positive"
        );
    }

    /// A peer publishes its own coordinate. One this code could never produce —
    /// a height under the floor (a negative one prices every hop to the peer
    /// below free) or a position past the clamp — is not routed or planned on.
    #[test]
    fn a_coordinate_this_code_could_never_produce_is_not_usable() {
        let honest = NetworkCoord {
            x: 100.0,
            y: -40.0,
            height: 2.0,
            error: 0.1,
        };
        assert!(honest.is_usable());
        for forged in [
            NetworkCoord {
                height: -500.0,
                ..honest
            },
            NetworkCoord {
                height: 0.0,
                ..honest
            },
            NetworkCoord {
                x: MAX_COORD_MS * 2.0,
                ..honest
            },
            NetworkCoord {
                y: -MAX_COORD_MS * 2.0,
                ..honest
            },
        ] {
            assert!(!forged.is_usable(), "{forged:?} must not be usable");
        }
    }

    /// The property the whole thing exists for: after gossip, the distance
    /// between two coordinates predicts the round trip between those nodes.
    #[test]
    fn coordinates_converge_to_predict_the_real_round_trips() {
        let truth = [
            (0.0, 0.0, 5.0),
            (10.0, 5.0, 3.0),
            (5.0, 8.0, 20.0),
            (300.0, 120.0, 8.0),
            (140.0, 400.0, 40.0),
        ];
        let (coords, worst) = converge(&truth, 300);
        assert!(
            worst < 0.20,
            "worst relative prediction error {worst:.3} — Vivaldi's own reported \
             median is 0.11, so anything near 0.2 across a clean synthetic \
             layout means the update rule is wrong"
        );
        for (i, c) in coords.iter().enumerate() {
            assert!(c.is_usable(), "node {i} never became usable: {c:?}");
        }
    }

    /// The distinction the router actually needs: near and far must come out
    /// different, by a lot.
    #[test]
    fn a_nearby_peer_and_a_distant_one_are_told_apart() {
        let truth = [(0.0, 0.0, 2.0), (8.0, 0.0, 2.0), (600.0, 0.0, 2.0)];
        let (c, _) = converge(&truth, 300);
        let near = c[0].distance_ms(&c[1]);
        let far = c[0].distance_ms(&c[2]);
        assert!(
            near < 40.0 && far > 400.0,
            "near={near:.1}ms far={far:.1}ms — the whole point is telling a \
             20ms hop from a 600ms one"
        );
    }

    /// Two nodes at the same point must be able to separate, or every node that
    /// starts at the origin stays there for ever.
    #[test]
    fn nodes_at_the_same_point_can_still_separate() {
        let mut a = NetworkCoord::new();
        let b = NetworkCoord::new();
        assert_eq!(a.distance_ms(&b), a.height + b.height, "start coincident");
        a.observe(120.0, &b);
        assert!(
            a.x.abs() + a.y.abs() > 0.0,
            "a coincident sample produced no movement at all — u(0) is not \
             returning a direction, and every node would stay stuck at the origin"
        );
    }

    /// A settled node must not be dragged around by one that has just joined —
    /// what the adaptive timestep buys over a constant one.
    #[test]
    fn a_settled_node_is_barely_moved_by_a_newcomer() {
        let truth = [(0.0, 0.0, 2.0), (50.0, 0.0, 2.0), (0.0, 50.0, 2.0)];
        let (coords, _) = converge(&truth, 300);
        let mut settled = coords[0];
        let before = settled;
        assert!(settled.is_usable());

        let newcomer = NetworkCoord::new();
        settled.observe(900.0, &newcomer);

        let moved = ((settled.x - before.x).powi(2) + (settled.y - before.y).powi(2)).sqrt();
        assert!(
            moved < 40.0,
            "a settled node moved {moved:.1}ms on one sample from a node that \
             knows nothing — the weight w is not damping the update"
        );
    }

    #[test]
    fn a_sample_that_carries_no_information_is_dropped() {
        let before = {
            let mut c = NetworkCoord::new();
            c.observe(50.0, &NetworkCoord::new());
            c
        };
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut c = before;
            c.observe(bad, &NetworkCoord::new());
            assert_eq!(
                c, before,
                "rtt {bad} must not move the coordinate — the relative error \
                 divides by it"
            );
        }
    }

    /// A peer publishing nonsense must not be able to throw us off the map.
    #[test]
    fn a_peer_with_a_broken_coordinate_cannot_drag_us_away() {
        let mut c = NetworkCoord::new();
        c.observe(50.0, &NetworkCoord::new());
        let before = c;
        for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY] {
            let mut victim = before;
            let junk = NetworkCoord {
                x: bad,
                y: bad,
                height: bad,
                error: 0.0,
            };
            victim.observe(50.0, &junk);
            assert_eq!(
                victim, before,
                "a non-finite peer coordinate must be ignored"
            );
        }
        let mut victim = before;
        let liar = NetworkCoord {
            x: 1e9,
            y: 1e9,
            height: 1e9,
            error: 0.0,
        };
        victim.observe(50.0, &liar);
        assert!(
            victim.x.is_finite() && victim.y.is_finite() && victim.height.is_finite(),
            "one hostile sample left the coordinate non-finite: {victim:?}"
        );
        assert!(victim.x.abs() <= MAX_COORD_MS && victim.y.abs() <= MAX_COORD_MS);
    }

    /// Height models the access link, and the case it exists for is a node
    /// that is further from EVERYONE by a constant.
    ///
    /// ⚠ The layout has to make height NECESSARY, which a tight cluster does
    /// not: if the other nodes sit within 30 ms of each other, "200 ms further
    /// from all of them" is just a point 200 ms away in the plane, and Vivaldi
    /// is right to use it — the first version of this test asserted height on
    /// exactly that layout and was wrong to. Here the others are spread 600 ms
    /// apart, so no planar point is equidistantly-further from all four and the
    /// offset can ONLY be expressed as height.
    #[test]
    fn a_slow_access_link_lands_in_height_not_position() {
        let truth = [
            (0.0, 0.0, 1.0),
            (600.0, 0.0, 1.0),
            (0.0, 600.0, 1.0),
            (600.0, 600.0, 1.0),
            // Geometric centre of the four, plus a 200 ms access link.
            (300.0, 300.0, 200.0),
        ];
        let (coords, worst) = converge(&truth, 400);
        assert!(worst < 0.25, "layout did not converge: worst {worst:.3}");
        let slow = coords[4];
        let neighbours: f32 = coords[..4].iter().map(|c| c.height).sum::<f32>() / 4.0;
        // Measured 2026-09-20: slow 214.1 ms against a true 200 ms link, with
        // neighbours averaging 55.7 ms. The neighbours' heights are NOT their
        // true 1 ms and are not expected to be — heights absorb whatever slack
        // the plane leaves, and the system is judged on its predictions, which
        // the convergence check above already pins. What must hold is that the
        // slow node recovers its own link and is plainly distinguished.
        assert!(
            (150.0..350.0).contains(&slow.height),
            "the 200ms access link should be recovered as height; got {:.1}ms",
            slow.height
        );
        assert!(
            slow.height > neighbours * 2.0,
            "the slow node's height ({:.1}ms) must stand out from its \
             neighbours' ({:.1}ms), or height is not separating access cost \
             from position at all",
            slow.height,
            neighbours
        );
    }

    fn at(x: f32, y: f32) -> NetworkCoord {
        NetworkCoord {
            x,
            y,
            height: 2.0,
            error: 0.2,
        }
    }

    /// Feed `filter` one sample per coordinate, all against one neighbour at
    /// `neighbour`, and return every published value seen after each sample.
    fn feed(
        filter: &mut UpdateFilter,
        coords: impl IntoIterator<Item = NetworkCoord>,
        neighbour: NetworkCoord,
    ) -> Vec<NetworkCoord> {
        let peer = crate::NodeId([7; 32]);
        coords
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                filter.note(c, &peer, 40.0, &neighbour, i as u64 * 1_000);
                filter.published().unwrap()
            })
            .collect()
    }

    /// A settled coordinate orbiting one point: the old rule (publish whenever
    /// the coordinate is 5 ms from the one last sent) republished it again and
    /// again; the filter publishes once, when its first window fills.
    #[test]
    fn a_coordinate_orbiting_one_point_is_published_once() {
        let orbit: Vec<NetworkCoord> = (0..400)
            .map(|i| {
                let a = i as f32 * 2.4;
                at(100.0 + 6.0 * a.cos(), 50.0 + 6.0 * a.sin())
            })
            .collect();

        let mut sent = orbit[0];
        let old_rule = orbit
            .iter()
            .filter(|c| {
                let news = c.moved_from(&sent) >= UPDATE_MIN_MOVE_MS;
                if news {
                    sent = **c;
                }
                news
            })
            .count();
        assert!(
            old_rule > 100,
            "the orbit must be news to the old rule ({old_rule})"
        );

        let mut filter = UpdateFilter::default();
        let published = feed(&mut filter, orbit, at(0.0, 0.0));
        let settled = &published[UPDATE_WINDOW - 1..];
        assert!(
            settled.iter().all(|c| *c == settled[0]),
            "an orbit is not a move: the published coordinate changed after its first window"
        );
        assert!(
            settled[0].moved_from(&at(100.0, 50.0)) < 1.0,
            "published at the orbit's centre, got {:?}",
            settled[0]
        );
    }

    /// Before its first window fills, a node publishes its coordinate from its
    /// first sample, and again only once it has moved past the threshold — a
    /// node starting up must publish, however rough, but not every wobble.
    #[test]
    fn a_node_starting_up_publishes_from_its_first_sample_and_follows_only_a_move() {
        let mut filter = UpdateFilter::default();
        assert_eq!(filter.published(), None);
        // The neighbour ~504 ms away: a move must pass ~150 ms.
        let neighbour = at(500.0, 0.0);
        let wobble: Vec<NetworkCoord> = (0..10).map(|i| at((i % 4) as f32 * 3.0, 0.0)).collect();
        let published = feed(&mut filter, wobble.clone(), neighbour);
        assert!(
            published.iter().all(|c| *c == wobble[0]),
            "a wobble of a few ms is not a move: {published:?}"
        );
        let published = feed(&mut filter, [at(200.0, 0.0)], neighbour);
        assert_eq!(
            published,
            vec![at(200.0, 0.0)],
            "a move past the threshold is published at once"
        );
    }

    /// The true round trip between two nodes of a `converge` layout.
    fn layout_rtt(truth: &[(f32, f32, f32)], a: usize, b: usize) -> f32 {
        let (ax, ay, ah) = truth[a];
        let (bx, by, bh) = truth[b];
        ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() + ah + bh
    }

    /// Start node 0 of a settled layout afresh (the others stay settled) and
    /// feed it `samples` round trips, one peer at a time, through a fresh
    /// filter. Returns the sample after which its coordinate as it stands was
    /// first usable, and the sample after which the PUBLISHED one first was.
    fn start_node_0(
        truth: &[(f32, f32, f32)],
        settled: &[NetworkCoord],
        samples: usize,
    ) -> (Option<usize>, Option<usize>) {
        let mut me = NetworkCoord::new();
        let mut filter = UpdateFilter::default();
        let (mut live_after, mut published_after) = (None, None);
        for s in 0..samples {
            let j = 1 + s % (truth.len() - 1);
            let rtt = layout_rtt(truth, 0, j);
            me.observe(rtt, &settled[j]);
            filter.note(
                me,
                &crate::NodeId([j as u8; 32]),
                rtt,
                &settled[j],
                s as u64 * 30_000,
            );
            if live_after.is_none() && me.is_usable() {
                live_after = Some(s + 1);
            }
            if published_after.is_none() && filter.published().is_some_and(|c| c.is_usable()) {
                published_after = Some(s + 1);
            }
        }
        (live_after, published_after)
    }

    const LAYOUT: [(f32, f32, f32); 5] = [
        (0.0, 0.0, 2.0),
        (120.0, 10.0, 3.0),
        (60.0, 200.0, 5.0),
        (-150.0, 80.0, 2.0),
        (30.0, -170.0, 4.0),
    ];

    /// A node starting up is published usable at the sample its coordinate
    /// becomes usable, not when its first window fills: routing acts only on a
    /// coordinate that `is_usable`, and one whose last publication was made
    /// while it was still unusable would otherwise be routed around until the
    /// window filled (~16 minutes on the live node) unless it happened to move
    /// past the threshold again first.
    #[test]
    fn a_node_starting_up_is_published_usable_as_soon_as_it_is() {
        let (settled, worst) = converge(&LAYOUT, 300);
        assert!(worst < 0.2, "the layout must settle first ({worst})");
        let (live, published) = start_node_0(&LAYOUT, &settled, UPDATE_WINDOW - 1);
        let live = live.expect("the coordinate becomes usable inside the first window");
        assert_eq!(
            published,
            Some(live),
            "published usable after {published:?} samples, usable itself after {live}"
        );
    }

    /// A node that moves is published where it went, at the change: the newest
    /// quarter is past the move, where the recent window's centroid (the
    /// paper's choice) is still part of the way there.
    #[test]
    fn a_coordinate_that_moves_is_published_where_it_went() {
        let here = at(100.0, 0.0);
        let there = at(150.0, 0.0);
        // The neighbour 100 ms from here: a move must pass 30 ms.
        let neighbour = at(0.0, 0.0);
        let wobble = |c: NetworkCoord, i: usize| at(c.x + (i % 3) as f32 - 1.0, c.y);

        let mut filter = UpdateFilter::default();
        let peer = crate::NodeId([7; 32]);
        let mut changes = Vec::new();
        for i in 0..UPDATE_WINDOW * 2 {
            changes.extend(filter.note(wobble(here, i), &peer, 40.0, &neighbour, 0));
        }
        assert!(changes.is_empty());
        let mut at_the_change = None;
        for i in 0..UPDATE_WINDOW * 3 {
            changes.extend(filter.note(wobble(there, i), &peer, 40.0, &neighbour, 0));
            if !changes.is_empty() {
                at_the_change.get_or_insert(filter.published().unwrap());
            }
        }
        assert_eq!(changes.len(), 1, "one move, one change: {changes:?}");
        assert!(changes[0].moved_ms > changes[0].threshold_ms);
        assert!(
            (changes[0].threshold_ms - 30.0).abs() < 1.5,
            "0.3 of the ~102 ms round trip to the neighbour, got {}",
            changes[0].threshold_ms
        );
        let at_the_change = at_the_change.unwrap();
        assert!(
            at_the_change.moved_from(&there) < 1.0,
            "published part of the way there: {at_the_change:?}"
        );
        assert_eq!(filter.published(), Some(at_the_change));
    }

    /// A move still under way when the change fires is published where it
    /// ended once the windows have refilled — the re-centring, without which
    /// the coordinate stays where the move was at the change for good.
    #[test]
    fn a_move_still_under_way_at_the_change_is_published_where_it_ended() {
        let neighbour = at(0.0, 0.0);
        let mut filter = UpdateFilter::default();
        let peer = crate::NodeId([7; 32]);
        let mut changes = Vec::new();
        for i in 0..UPDATE_WINDOW * 2 {
            changes.extend(filter.note(at(100.0, 0.0), &peer, 40.0, &neighbour, i as u64));
        }
        // 60 ms over 48 samples, then still: the change fires mid-way.
        let ramp = (1..=48).map(|j| at(100.0 + 1.25 * j as f32, 0.0));
        let still = std::iter::repeat_n(at(160.0, 0.0), UPDATE_WINDOW * 3);
        let mut mid_way = None;
        for c in ramp.chain(still) {
            changes.extend(filter.note(c, &peer, 40.0, &neighbour, 0));
            if !changes.is_empty() {
                mid_way.get_or_insert(filter.published().unwrap());
            }
        }
        assert_eq!(changes.len(), 2, "a change and a re-centring: {changes:?}");
        assert!(
            mid_way.unwrap().x < 150.0,
            "the change fired mid-way: {mid_way:?}"
        );
        let published = filter.published().unwrap();
        assert!(
            published.moved_from(&at(160.0, 0.0)) < 0.01,
            "published where the move ended: {published:?}"
        );
    }

    /// A node starting up converges from wherever its coordinate began, and
    /// its first full window holds that trip: published as that window's
    /// centroid, the coordinate went BACK toward the start. On the live node
    /// the first change came two minutes after the first window filled, a
    /// 21.8 ms move (2026-11-01).
    #[test]
    fn a_node_converging_after_start_up_is_never_published_behind_where_it_was() {
        let target = at(100.0, 0.0);
        let neighbour = at(200.0, 0.0);
        let converging =
            (0..UPDATE_WINDOW * 8).map(|k| at(target.x * (1.0 - (-(k as f32) / 20.0).exp()), 0.0));
        let mut filter = UpdateFilter::default();
        let published = feed(&mut filter, converging, neighbour);
        let mut distinct = published.clone();
        distinct.dedup();
        assert!(
            distinct.windows(2).all(|w| w[1].x > w[0].x),
            "every publication is nearer where the node is going: {distinct:?}"
        );
        assert!(
            distinct.len() <= 4,
            "a start-up is a handful of publications: {distinct:?}"
        );
        let last = published.last().unwrap();
        assert!(
            last.moved_from(&target) < 6.0,
            "published within the threshold of where it settled: {last:?}"
        );
    }

    /// The same move is news beside a near neighbour and not beside a far one:
    /// the threshold is relative to the node's locale, which is why a fixed
    /// one "did not work for all nodes" — and never under the floor. Two
    /// windows still first: the window after a node's first is a re-centring,
    /// judged on the floor alone.
    #[test]
    fn the_move_that_counts_scales_with_the_nearest_neighbour() {
        let moves = |neighbour: NetworkCoord, by: f32| {
            let mut filter = UpdateFilter::default();
            let peer = crate::NodeId([7; 32]);
            let mut changes = 0;
            for i in 0..UPDATE_WINDOW * 2 {
                changes += filter
                    .note(at(0.0, 0.0), &peer, 40.0, &neighbour, i as u64)
                    .is_some() as usize;
            }
            for i in 0..UPDATE_WINDOW {
                changes += filter
                    .note(at(by, 0.0), &peer, 40.0, &neighbour, i as u64)
                    .is_some() as usize;
            }
            changes
        };
        assert!(
            moves(at(30.0, 0.0), 20.0) > 0,
            "20 ms beside a neighbour ~34 ms away"
        );
        assert_eq!(
            moves(at(200.0, 0.0), 20.0),
            0,
            "20 ms beside a neighbour ~204 ms away"
        );
        assert_eq!(
            moves(at(1.0, 0.0), 4.0),
            0,
            "under the 5 ms floor, however near the neighbour"
        );
    }

    /// A nonsense coordinate teaches the filter nothing.
    #[test]
    fn a_coordinate_that_is_not_a_number_is_not_published() {
        let mut filter = UpdateFilter::default();
        let peer = crate::NodeId([7; 32]);
        filter.note(at(1.0, 2.0), &peer, 40.0, &at(0.0, 0.0), 0);
        assert_eq!(
            filter.note(at(f32::NAN, 2.0), &peer, 40.0, &at(0.0, 0.0), 1),
            None
        );
        assert_eq!(filter.published(), Some(at(1.0, 2.0)));
    }
}
