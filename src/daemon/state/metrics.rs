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
    pub channel_metrics: ChannelMetricsSet,
    pub ws_connection_count: std::sync::atomic::AtomicUsize,
    pub node_stats: RwLock<NodeStats>,
    pub providers_config: RwLock<crate::config::ProvidersConfig>,
    pub provider_model_map: DashMap<String, String>,
    pub provider_models_cache: RwLock<(Vec<serde_json::Value>, std::time::Instant)>,
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
    /// Observed per-layer latency EMA (ms per layer) per remote peer. Updated
    /// after every successful remote segment in `forward_through_segments`.
    /// Consumed by the Parallax routing DP to replace the static
    /// `est_tokens_per_sec` capability estimate when a live signal is
    /// available. Per-layer normalisation makes the signal comparable across
    /// segment widths (e.g. a 4-layer segment vs a 16-layer segment on the
    /// same peer).
    pub peer_segment_latency_ms_per_layer: DashMap<crate::types::NodeId, f32>,
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
