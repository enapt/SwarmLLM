use std::sync::atomic::{AtomicI64, AtomicU64};
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::types::NodeStats;

use super::capacity::SwarmCapacity;

/// Metrics, stats, and provider configuration.
pub struct MetricsProviders {
    /// This node's Vivaldi network coordinate — the position that lets ANY
    /// reader estimate the round trip between two nodes, including two peers
    /// neither of which is the reader.
    ///
    /// Written only by `SharedState::observe_network_coord`, from round trips
    /// the tensor path already measures (`AckRttEstimator`'s samples, which are
    /// gated to small forwards where the time is the peer's rather than the
    /// payload's). Read by the capability builder that publishes it and — once
    /// routing consumes it — by the scheduler.
    ///
    /// A `std::sync::RwLock` rather than an async one because both readers are
    /// synchronous and the critical section is a struct copy.
    pub network_coord: std::sync::RwLock<swarmllm_types::netcoord::NetworkCoord>,
    /// Recent round trips per peer, answering with the windowed MINIMUM — the
    /// only figure fed to `network_coord`.
    ///
    /// The round trip we can measure is application-level and queues behind
    /// the remote node's event loop, so it is contaminated one-sidedly and, on
    /// this fleet, bimodally: 3-8 ms or 118-158 ms against a peer whose ICMP
    /// round trip is ~1 ms. Feeding raw samples taught the coordinate the
    /// queueing delay. See `LatencyFilter` for why this is a minimum where
    /// Serf's equivalent is a median.
    pub network_coord_samples:
        DashMap<crate::types::NodeId, swarmllm_types::netcoord::LatencyFilter>,
    pub inference_requests_total: AtomicU64,
    /// Mirror of node_stats.requests_served as an AtomicU64 — written from
    /// multiple async contexts. The RwLock-guarded field on NodeStats was
    /// updated via `try_write()` which silently drops on contention,
    /// undercounting served requests on busy dashboards. Serialization sites
    /// snapshot this counter into the displayed NodeStats. Same pattern as
    /// inference_requests_total.
    pub requests_served_atomic: AtomicU64,
    /// Mirror of node_stats.forwards_served — same try_write→atomic story.
    pub forwards_served_atomic: AtomicU64,
    /// Bounded ring of recent inference latencies for percentile computation.
    /// Entries are `(observed_at_instant, latency_seconds)`; the timestamp lets
    /// `compute_latency_stats` and the Prometheus histogram emitter drop entries
    /// older than `LATENCY_SAMPLE_MAX_AGE` (10 min) so a lightly-loaded node
    /// doesn't keep showing a p99 from yesterday's spike. The 1000-entry cap
    /// remains the memory bound; the age window is a freshness bound on top.
    /// R137 (closes R105 deferral).
    pub inference_latency_samples:
        std::sync::RwLock<std::collections::VecDeque<(std::time::Instant, f64)>>,
    /// Monotonic total count of latency samples ever recorded. The
    /// `inference_latency_samples` ring buffer caps at a fixed size and
    /// would otherwise produce a non-monotonic Prometheus histogram
    /// `_count` (it falls when the ring wraps), breaking `rate()` and
    /// `increase()` queries. This counter is the canonical histogram
    /// `_count`. Same idea for sum.
    pub inference_latency_total_count: AtomicU64,
    /// Monotonic total sum of latency samples (ms × 1000 to keep an
    /// integer; divide by 1e6 when emitting as seconds).
    pub inference_latency_total_micros: AtomicU64,
    /// Time-to-first-token and time-per-output-token samples, seconds.
    ///
    /// OTel's `gen_ai.server.time_to_first_token` and
    /// `gen_ai.server.time_per_output_token`. Neither existed server-side
    /// before — TTFT lived only in the bench CLI, measured client-side — and
    /// they are the two numbers that separate "the queue is backed up" from
    /// "decode is slow", which wall-clock total cannot.
    ///
    /// Same ring + monotonic-counter shape as `inference_latency_samples`: the
    /// ring gives the bucket distribution, the atomics give a `_count`/`_sum`
    /// that never falls when the ring wraps (R105).
    pub ttft_samples: std::sync::RwLock<std::collections::VecDeque<(std::time::Instant, f64)>>,
    pub ttft_total_count: AtomicU64,
    pub ttft_total_micros: AtomicU64,
    pub tpot_samples: std::sync::RwLock<std::collections::VecDeque<(std::time::Instant, f64)>>,
    pub tpot_total_count: AtomicU64,
    pub tpot_total_micros: AtomicU64,
    /// Completed requests by `(route, outcome)`.
    ///
    /// Deliberately the ONLY labelled request counter. Both label values come
    /// from closed sets (5 routes × 4 outcomes = 20 series max), so this cannot
    /// grow with the swarm. Per-peer, per-model and per-shard breakdowns are
    /// unbounded and live in `GET /api/admin/diagnostics`, which is pulled on
    /// demand and never retained — see `docs/FUTURE_WORK.md` § Observability on
    /// why an unbounded label set takes down the scrape.
    pub requests_by_route: DashMap<(&'static str, &'static str), u64>,
    /// Serving-side totals: segments this node computed FOR OTHER PEERS.
    ///
    /// Every other counter here is requester-side. Without these an operator
    /// cannot answer "is my node actually contributing, and how well", and a
    /// node whose segments everyone times out on looks identical to a healthy
    /// one. Plain atomics rather than a labelled map — the useful question is
    /// the node's own throughput, and per-requester breakdown would be
    /// unbounded.
    pub segments_served: AtomicU64,
    pub layers_served: AtomicU64,
    pub segment_serve_micros: AtomicU64,
    pub segment_bytes_out: AtomicU64,
    /// Tokens this node has produced for other people.
    ///
    /// The panel used to describe served work only as segments, layers and an
    /// average ms-per-layer. None of those is a rate a person can act on, and
    /// the average in particular CANNOT be turned into one: a prompt pass and a
    /// single decode step each count as one segment, and they differ by orders
    /// of magnitude, so the mean is over two different quantities. An operator
    /// asked us directly what tokens per second his machine was managing —
    /// the number was not on his screen and could not be derived from what was.
    /// It was already in hand at the one place serving is recorded, and thrown
    /// away.
    pub tokens_served: AtomicU64,
    pub channel_metrics: ChannelMetricsSet,
    /// Message-dispatcher liveness: when it last took a message off
    /// `network_out`, in epoch millis, and which variant that was.
    ///
    /// **Written by the dispatcher, read by a DIFFERENT task**, and that split
    /// is the whole point. On 2026-09-18 the dispatcher stopped consuming for
    /// 45 minutes and nothing noticed: `daemon::supervisor` reacts only when
    /// `JoinSet::join_next()` returns, i.e. to a panic or a clean exit, so a
    /// task parked for ever inside an `.await` produces no signal at all. A
    /// heartbeat emitted BY the dispatcher would be just as silent — it never
    /// gets back to the top of its own loop to emit one. So the dispatcher
    /// writes a marker the instant a message arrives, before it decides what to
    /// do with it, and `HealthMonitor` (already ticking on its own timer) is
    /// what complains. `docs/FUTURE_WORK.md` #90.
    ///
    /// `0` means nothing has been dispatched yet.
    pub last_dispatch_at_ms: AtomicI64,
    /// The `SwarmMessage` variant behind `last_dispatch_at_ms` — the one thing
    /// that says WHICH arm to look at. A `Mutex<&'static str>` rather than an
    /// atomic index into a table: the name comes from
    /// `SwarmMessage::kind_name`, which the compiler forces to stay exhaustive,
    /// and a second table to keep in step would be the thing that goes stale.
    /// Uncontended, never held across an await, ~15 ns on a path that peaks
    /// around 100 messages a second.
    pub last_dispatch_kind: parking_lot::Mutex<&'static str>,
    pub ws_connection_count: std::sync::atomic::AtomicUsize,
    pub node_stats: RwLock<NodeStats>,
    pub providers_config: RwLock<crate::config::ProvidersConfig>,
    pub provider_model_map: DashMap<String, String>,
    pub provider_models_cache: RwLock<(Vec<serde_json::Value>, std::time::Instant)>,
    /// Cached `/api/admin/provider-health` results, `(providers, built_at)`.
    ///
    /// Building this costs one billable request per configured provider, so it
    /// must not be rebuilt per dashboard poll: the budget is per-IP and several
    /// open tabs share it, which drove ~60 outbound paid probes/min on a live
    /// node until this existed.
    pub provider_health_cache: RwLock<(Vec<serde_json::Value>, std::time::Instant)>,
    /// Cached WebSocket stats JSON, shared across all connected clients.
    /// (built_at, message). Built on demand by the first WS client to tick
    /// with a stale cache; subsequent clients within TTL reuse the string.
    /// Eliminates O(n) shard/peer registry scans per client per 2s tick.
    pub stats_cache: parking_lot::Mutex<Option<(std::time::Instant, std::sync::Arc<String>)>>,
    /// Stampede guard for stats_cache: when the cache expires and 100 clients
    /// tick simultaneously, they would all observe a miss and rebuild in
    /// parallel. CAS this flag to ensure only one rebuilder runs; the rest
    /// return the stale value. Rebuilder clears the flag after writing the
    /// new cache entry.
    pub stats_building: std::sync::atomic::AtomicBool,
    /// Measured compute speed per remote peer — see `state::peer_speed`.
    /// Updated after every successful remote segment in
    /// `forward_through_segments`, and merged (trust-weighted) from gossip.
    ///
    /// Prefill and decode are held as SEPARATE, differently-normalised EMAs
    /// because they differ by ~2 orders of magnitude on the same peer. This
    /// replaced a single blended `ms_per_layer` figure that could predict
    /// neither and was consequently useless for sizing a timeout.
    ///
    /// Consumers: segment-timeout sizing (`pipeline::local`), remote-candidate
    /// ranking (`inference::scheduler`), the Parallax routing DP, and
    /// `GET /api/admin/performance`. Swept by the HealthMonitor tick via
    /// `evict_stale_peer_speed` so departed peers do not linger.
    pub peer_speed: DashMap<crate::types::NodeId, super::PeerSpeed>,
    /// Last time a segment forward for `(peer, model)` completed successfully.
    ///
    /// Its only job is to answer "might this peer have to LOAD the model
    /// before it can answer?". A cold peer legitimately takes minutes: the
    /// 2026-08-01 failure was a peer needing ~120s to load an 8B model, cut
    /// off by a flat 120s deadline. A first forward therefore gets a much
    /// larger budget than a warm one. Unreachability is NOT covered by that
    /// generosity — `RR_ACK_TIMEOUT_SECS` still fails a silently-dropped send
    /// in 10s.
    pub peer_model_warm_at:
        DashMap<(crate::types::NodeId, crate::types::ModelId), std::time::Instant>,
    /// Cached snapshot of swarm-wide capacity (online nodes, total VRAM,
    /// serveable models, ...). Refreshed on gossip ticks via
    /// `capacity::refresh_swarm_capacity`. ArcSwap so dashboard / WS / REST
    /// readers all see a lock-free snapshot — capacity is read on every
    /// dashboard render and we don't want to gate it behind the same lock
    /// tree the writers contend for. R110.
    pub swarm_capacity: ArcSwap<SwarmCapacity>,
    /// Per-(model, segment, holder) forward latency, as a moving average —
    /// what the peer performance table reports per computer. Written by
    /// `SharedState::record_segment_latency`, swept by the health monitor.
    pub segment_latency: Arc<crate::inference::segment_latency::SegmentLatencyTracker>,
    /// SWARM-SPEC Layer 3: conversation-level prefetch orchestrator.
    /// Tracks per-session first-token histograms + idle time; emits
    /// candidate first-tokens to prefetch when the predicted next
    /// request becomes likely. The decision-and-history surface lives
    /// here; the actual prefetch dispatch (running activations
    /// forward, gossiping warming) is a follow-up integration point
    /// per docs/FUTURE_WORK.md § R136 Layer 3.
    pub prefetch_orchestrator: crate::inference::prefetch::PrefetchHandle,
    /// SWARM-SPEC Layer 1: lifetime counters for n-gram-cascade
    /// hits / misses across all spec paths (`speculative.rs` draft+ngram
    /// path AND `ngram_only_spec.rs` draft-free path). Surfaced in
    /// `GET /api/admin/stats → swarm_spec.ngram` so operators can see
    /// whether L1 is actually firing on their workload mix. R137.
    pub ngram_hits: AtomicU64,
    pub ngram_misses: AtomicU64,
    /// Bytes this node has put on, and taken off, the wire.
    ///
    /// Counted at the transport, so it covers every protocol — gossip, DHT
    /// maintenance, shard transfers, inference — and not merely the traffic
    /// this code writes itself. Read through `BandwidthMeter::totals`, which
    /// answers `None` when nothing is counting rather than a zero that cannot
    /// be told apart from a silent node.
    ///
    /// It exists because a user chasing a hammered home connection had no way
    /// to ask their node what it was sending, and had to stop the daemon and
    /// diff the interface counters to find out (2026-09-11 suggestion).
    pub bandwidth: Arc<crate::network::bandwidth::BandwidthMeter>,
    /// The same question, split by WHICH traffic — GossipSub's own per-topic
    /// byte counters, armed when the behaviour is built.
    ///
    /// Separate from `bandwidth` because the two are read from different
    /// registries and one can be present while the other is absent: the
    /// transport counters appear on the first byte of any protocol, these on
    /// the first gossip message. Both answer `None` rather than zero when
    /// nothing is counting, for the reason in `BandwidthMeter::totals`.
    pub gossip: Arc<crate::network::bandwidth::GossipMeter>,
    /// Shard bytes this node has SERVED to peers, and received from them.
    ///
    /// The one category an operator can already act on: `shard_upload_mbps` —
    /// what `resources.max_bandwidth_mbps` sets — throttles exactly this and
    /// nothing else. Counting it makes the cap's scope checkable instead of
    /// documented, which is the whole complaint behind field reports
    /// 2026-09-11 and 2026-09-20; the second bounded it by counting a log line
    /// because there was no counter.
    ///
    /// Plain atomics rather than a meter: unlike gossip and the transport
    /// totals, these are OUR OWN bytes at a choke point we control, so there is
    /// no registry to parse and no "is anything counting?" ambiguity. Zero
    /// here genuinely means zero.
    ///
    /// Written off the swarm event loop, in the spawned task that already does
    /// the disk read and the throttle sleep (gotcha #11), and on the inbound
    /// response path. Never from the tensor-forward path, which is deliberately
    /// outside the cap and outside this.
    pub shard_bytes_out: std::sync::atomic::AtomicU64,
    pub shard_bytes_in: std::sync::atomic::AtomicU64,
    /// Distributed-inference bytes on the wire: tensor forwards and the results
    /// that answer them, plus the tokens a fast-path reply streams back.
    ///
    /// The category a user actually asks about — "is my line being used for
    /// WORK, or for chatter?" — and the one that was hardest to answer by
    /// elimination, because it is bursty and looks like nothing at all when the
    /// node is idle. Gossip and shard serving between them left it inside
    /// `other_*` with the DHT, identify and ping, which are constant and tiny;
    /// a user watching a 4 Mbps remainder could not tell which it was.
    ///
    /// Counted in the CODEC, not at the send sites, because
    /// `network.tensor_compression` defaults on and the send site holds the
    /// uncompressed activation — see [`InferenceTraffic`] for why an over-count
    /// here would corrupt `other_*` rather than merely misreport itself.
    ///
    /// ⚠ **`resources.max_bandwidth_mbps` does NOT cover this.** That cap is
    /// shard serving only, and the reason to count these separately is so that
    /// stays visible rather than implied.
    ///
    /// [`InferenceTraffic`]: crate::network::bandwidth::InferenceTraffic
    pub inference: Arc<crate::network::bandwidth::InferenceTraffic>,
    /// What a gossip topic is actually made OF, by message variant.
    ///
    /// GossipSub's own counters stop at the topic, and `swarm/models` carries
    /// six variants — `ModelManifest`, `NodeCapabilityUpdate`, `ShardAnnounce`,
    /// `ShardDownloadProgress`, `HfSourceGossip`, `PrefixCacheAnnounce`. That
    /// topic is **82% of an idle node's upload** (measured 2026-09-21), and
    /// `docs/FUTURE_WORK.md` #91's two remaining fixes — change-gating the
    /// capability broadcast, and moving manifests off the broadcast path —
    /// sit on that same topic. Nothing said which of the six was paying for
    /// it, so the two could be told apart only by estimating from sizes and
    /// intervals.
    ///
    /// **That estimate is the one #91 records getting wrong by 10-100x, three
    /// times.** Gotcha #673 is the rule it left: before costing the parts of
    /// an expensive total, find the counter you are not reading.
    ///
    /// Counted on RECEIVE, which is the figure that decides this: an idle node
    /// publishes ~0.2% of what it sends, so its upload is relaying, and what it
    /// relays is what it received. Bytes are the sealed frame as it arrived, so
    /// they are comparable with the per-topic counters beside them.
    pub gossip_by_kind: Arc<crate::network::bandwidth::GossipKindMeter>,
}

/// How long a channel may keep refusing messages before the next drop is
/// reported again.
///
/// 30 s is the shortest interval at which a genuinely stalled consumer is still
/// obvious in a log read at human speed: a 45-minute stall becomes ~91 lines
/// instead of 230,402.
const DROP_REPORT_INTERVAL_MS: u64 = 30_000;

/// Sentinel for `last_sent_ms`: this channel has never accepted a message.
const NEVER_SENT: u64 = u64::MAX;

/// A drop that is worth a log line, and the context that makes it readable.
///
/// The count alone cannot tell a momentary burst from a dead consumer, and that
/// is the whole question: one is normal under load, the other means this node
/// has silently stopped taking part. `nothing_accepted_for` is what separates
/// them.
pub struct DropBurst {
    /// Drops on this channel swallowed since the last reported one.
    pub suppressed: u64,
    /// How long this channel has gone without accepting ANYTHING — measured
    /// from its creation when it has never accepted a message, which is the
    /// honest reading of "nothing has got through".
    pub nothing_accepted_for: std::time::Duration,
    /// Lifetime drops on this channel, this one included.
    pub total: u64,
}

impl DropBurst {
    /// `nothing_accepted_for` as whole seconds, for a `tracing` field.
    pub fn stalled_secs(&self) -> u64 {
        self.nothing_accepted_for.as_secs()
    }
}

/// The "has this burst already been reported recently?" decision.
#[derive(Default)]
struct DropReportWindow {
    /// `created.elapsed()` in ms at the last drop that was reported.
    last_report_ms: Option<u64>,
    /// Drops swallowed since then.
    suppressed: u64,
}

impl MetricsProviders {
    /// Record that the message dispatcher has just taken a message off its
    /// channel. Called BEFORE the message is acted on, at the one point every
    /// message passes through whatever arm it takes.
    pub fn note_dispatch(&self, kind: &'static str) {
        self.last_dispatch_at_ms.store(
            chrono::Utc::now().timestamp_millis(),
            std::sync::atomic::Ordering::Relaxed,
        );
        *self.last_dispatch_kind.lock() = kind;
    }

    /// How long the dispatcher has gone without taking a message, and what the
    /// last one was. `None` before the first message — a node that has received
    /// nothing yet is not a stalled one.
    pub fn dispatch_idle_for(&self) -> Option<(std::time::Duration, &'static str)> {
        let at = self
            .last_dispatch_at_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if at == 0 {
            return None;
        }
        let idle_ms = chrono::Utc::now().timestamp_millis().saturating_sub(at);
        Some((
            std::time::Duration::from_millis(idle_ms.max(0) as u64),
            *self.last_dispatch_kind.lock(),
        ))
    }
}

/// Atomic counters for a single mpsc channel.
///
/// `record_sent` is on the hot path and touches only atomics. A drop is by
/// definition the exceptional path, so it may take a lock to decide whether
/// this one is worth a log line.
pub struct ChannelCounters {
    pub capacity: u32,
    pub sent: AtomicU64,
    pub dropped: AtomicU64,
    /// Base for the millisecond clocks below.
    created: std::time::Instant,
    /// `created.elapsed()` in ms at the last message this channel ACCEPTED, or
    /// `NEVER_SENT`.
    last_sent_ms: AtomicU64,
    report: std::sync::Mutex<DropReportWindow>,
    /// Test-only: added to every clock reading, so a test can reach the far side
    /// of `DROP_REPORT_INTERVAL_MS` or simulate a 45-minute stall without
    /// sleeping. The field does not exist in a release build.
    ///
    /// The clocks measure time SINCE `created`, so a test cannot move them into
    /// the past — at `t ≈ 0` there is no past to move into. Advancing "now" is
    /// the same fiction from the other end, and the one that works.
    #[cfg(test)]
    test_clock_advance_ms: AtomicU64,
}

impl ChannelCounters {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            created: std::time::Instant::now(),
            last_sent_ms: AtomicU64::new(NEVER_SENT),
            report: std::sync::Mutex::new(DropReportWindow::default()),
            #[cfg(test)]
            test_clock_advance_ms: AtomicU64::new(0),
        }
    }

    /// Milliseconds since this counter was created.
    #[inline]
    fn now_ms(&self) -> u64 {
        let elapsed = self.created.elapsed().as_millis() as u64;
        #[cfg(test)]
        let elapsed = elapsed
            + self
                .test_clock_advance_ms
                .load(std::sync::atomic::Ordering::Relaxed);
        elapsed
    }

    /// Pretend `ms` milliseconds have passed.
    #[cfg(test)]
    fn advance_clock_for_test(&self, ms: u64) {
        self.test_clock_advance_ms
            .fetch_add(ms, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_sent(&self) {
        self.sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last_sent_ms
            .store(self.now_ms(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Count a dropped message, and answer whether this one should be logged.
    ///
    /// **There is no way to count a drop without being handed this decision**,
    /// which is deliberate. The `network_out` channel reported every single
    /// drop unconditionally, and when its consumer wedged for 45 minutes on
    /// 2026-09-18 the node wrote 230,402 identical WARN lines — 74% of the
    /// whole log file — burying the one fact that mattered: that nothing had
    /// been accepted since 11:36 and the node had stopped taking part in the
    /// swarm. A per-message line is not a smaller version of that signal; it is
    /// what hides it. Gotcha #648.
    #[must_use = "a dropped message that is never reported is a silent failure — log the returned burst"]
    pub fn note_dropped(&self) -> Option<DropBurst> {
        let total = self
            .dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let now_ms = self.now_ms();
        // A poisoned lock here must not cost the report — the guarded state is
        // two counters used only for rate limiting, and nothing reads them back
        // for a decision that matters.
        let mut window = self
            .report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let due = match window.last_report_ms {
            None => true,
            Some(prev) => now_ms.saturating_sub(prev) >= DROP_REPORT_INTERVAL_MS,
        };
        if !due {
            window.suppressed += 1;
            return None;
        }
        let suppressed = std::mem::take(&mut window.suppressed);
        window.last_report_ms = Some(now_ms);
        drop(window);
        let last_sent = self.last_sent_ms.load(std::sync::atomic::Ordering::Relaxed);
        let since_ms = if last_sent == NEVER_SENT {
            now_ms
        } else {
            now_ms.saturating_sub(last_sent)
        };
        Some(DropBurst {
            suppressed,
            nothing_accepted_for: std::time::Duration::from_millis(since_ms),
            total,
        })
    }
}

/// Backpressure metrics for all daemon mpsc channels.
pub struct ChannelMetricsSet {
    pub network_cmd: Arc<ChannelCounters>,
    pub network_out: Arc<ChannelCounters>,
    pub router_cmd: Arc<ChannelCounters>,
    pub rebalance: Arc<ChannelCounters>,
    pub acquisition: Arc<ChannelCounters>,
    pub pool_cmd: Arc<ChannelCounters>,
}

impl ChannelMetricsSet {
    pub(super) fn new() -> Self {
        Self {
            network_cmd: Arc::new(ChannelCounters::new(1024)),
            network_out: Arc::new(ChannelCounters::new(1024)),
            router_cmd: Arc::new(ChannelCounters::new(256)),
            rebalance: Arc::new(ChannelCounters::new(64)),
            acquisition: Arc::new(ChannelCounters::new(64)),
            pool_cmd: Arc::new(ChannelCounters::new(64)),
        }
    }
}

#[cfg(test)]
mod channel_counter_tests {
    use super::*;

    /// The first drop on a channel is always reported — a burst that is never
    /// announced is exactly the silence this rate limiter exists to avoid.
    #[test]
    fn the_first_drop_of_a_burst_is_reported() {
        let c = ChannelCounters::new(8);
        let burst = c.note_dropped().expect("first drop must be reported");
        assert_eq!(burst.suppressed, 0);
        assert_eq!(burst.total, 1);
    }

    /// The property the live node needed and did not have: 230,402 drops in one
    /// stall produced 230,402 log lines. Inside the window, every drop after the
    /// first is counted and swallowed.
    #[test]
    fn a_burst_inside_the_window_is_reported_once() {
        let c = ChannelCounters::new(8);
        let reported = (0..5_000).filter(|_| c.note_dropped().is_some()).count();
        assert_eq!(
            reported, 1,
            "5000 drops inside one window must produce exactly one report"
        );
        assert_eq!(c.dropped.load(std::sync::atomic::Ordering::Relaxed), 5_000);
    }

    /// And the swallowed ones are not lost — the next report carries them, so
    /// the log still says how bad it got.
    #[test]
    fn the_next_report_carries_what_was_swallowed() {
        let c = ChannelCounters::new(8);
        assert!(c.note_dropped().is_some());
        for _ in 0..41 {
            assert!(c.note_dropped().is_none());
        }
        c.advance_clock_for_test(DROP_REPORT_INTERVAL_MS + 1);
        let burst = c.note_dropped().expect("window has elapsed");
        assert_eq!(burst.suppressed, 41);
        assert_eq!(burst.total, 43);
        // And the count starts again rather than accumulating for ever.
        c.advance_clock_for_test(DROP_REPORT_INTERVAL_MS + 1);
        let next = c.note_dropped().expect("window has elapsed again");
        assert_eq!(next.suppressed, 0);
        assert_eq!(next.total, 44);
    }

    /// The figure that separates a momentary burst from a dead consumer.
    #[test]
    fn a_report_says_how_long_nothing_has_got_through() {
        let c = ChannelCounters::new(8);
        c.record_sent();
        c.advance_clock_for_test(45 * 60 * 1000);
        let burst = c.note_dropped().expect("first drop");
        assert!(
            burst.stalled_secs() >= 45 * 60,
            "a consumer that has accepted nothing for 45 minutes must say so, got {}s",
            burst.stalled_secs()
        );
    }

    /// A channel that has never accepted anything reports its whole life, not a
    /// zero that reads like a healthy channel having one bad moment.
    #[test]
    fn a_channel_that_never_accepted_anything_does_not_report_zero_stall() {
        let c = ChannelCounters::new(8);
        let burst = c.note_dropped().expect("first drop");
        assert_eq!(
            burst.nothing_accepted_for,
            std::time::Duration::from_millis(0),
            "at creation the stall is genuinely zero"
        );
        c.advance_clock_for_test(10_000);
        c.advance_clock_for_test(DROP_REPORT_INTERVAL_MS + 1);
        let later = c.note_dropped().expect("window has elapsed");
        assert!(
            later.stalled_secs() > 0,
            "a channel that has still accepted nothing must report the elapsed time"
        );
    }

    /// A message that gets through ends the stall, so the next burst is measured
    /// from the last thing the consumer actually took.
    #[test]
    fn an_accepted_message_resets_the_stall_clock() {
        let c = ChannelCounters::new(8);
        c.record_sent();
        c.advance_clock_for_test(60_000);
        assert!(c.note_dropped().expect("first drop").stalled_secs() >= 60);
        c.record_sent();
        c.advance_clock_for_test(DROP_REPORT_INTERVAL_MS + 1);
        let burst = c.note_dropped().expect("window has elapsed");
        assert!(
            burst.stalled_secs() <= DROP_REPORT_INTERVAL_MS / 1000 + 1,
            "the clock must run from the last ACCEPTED message, got {}s",
            burst.stalled_secs()
        );
    }
}

#[cfg(test)]
mod dispatch_liveness_tests {
    fn test_state() -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(
            crate::config::Config::default(),
            Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    /// A node that has received nothing yet is not a stalled one — reporting a
    /// stall at boot would train everyone to ignore the line.
    #[test]
    fn a_node_that_has_dispatched_nothing_is_not_reported_as_stalled() {
        let state = test_state();
        assert!(state.metrics.dispatch_idle_for().is_none());
    }

    /// And once a message has been taken, the marker carries WHICH one — the
    /// only thing that says which handler to suspect.
    #[test]
    fn the_marker_names_the_last_message_taken() {
        let state = test_state();
        state.metrics.note_dispatch("LayerForward");
        let (idle, kind) = state
            .metrics
            .dispatch_idle_for()
            .expect("a dispatched message must be recorded");
        assert_eq!(kind, "LayerForward");
        assert!(
            idle < std::time::Duration::from_secs(5),
            "a marker written just now must read as fresh, got {idle:?}"
        );

        state.metrics.note_dispatch("ShardAnnounce");
        assert_eq!(
            state.metrics.dispatch_idle_for().unwrap().1,
            "ShardAnnounce"
        );
    }

    /// The name comes from the message itself, so a new variant cannot reach
    /// the marker as "unknown" — `kind_name` has no catch-all arm and the
    /// compiler enforces it.
    #[test]
    fn a_messages_kind_name_is_its_variant_name() {
        use swarmllm_types::SwarmMessage;
        assert_eq!(
            SwarmMessage::PeerExchangeRequest.kind_name(),
            "PeerExchangeRequest"
        );
        let ping = SwarmMessage::HealthPing {
            nonce: 1,
            timestamp: 0,
            node_id: None,
            active_request_count: 0,
        };
        assert_eq!(ping.kind_name(), "HealthPing");
    }
}
