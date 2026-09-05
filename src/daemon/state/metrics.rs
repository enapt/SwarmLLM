use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::types::NodeStats;

use super::capacity::SwarmCapacity;

/// Metrics, stats, and provider configuration.
pub struct MetricsProviders {
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
    /// SWARM-SPEC Layer 2: per-(model, segment, holder) latency EWMA
    /// tracker, used to decide when to fire a duplicate forward to
    /// the second-best holder. Lock-free reads/writes via DashMap.
    /// Always present; `hedge_enabled` config flag gates whether the
    /// decision actually fires a hedge.
    pub hedge_tracker: Arc<crate::inference::hedging::HedgeTracker>,
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
}

/// Atomic counters for a single mpsc channel.
pub struct ChannelCounters {
    pub capacity: u32,
    pub sent: AtomicU64,
    pub dropped: AtomicU64,
}

impl ChannelCounters {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    pub fn record_sent(&self) {
        self.sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_dropped(&self) {
        self.dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
