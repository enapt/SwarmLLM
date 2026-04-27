use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::types::NodeStats;

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
    pub inference_latency_samples: std::sync::RwLock<std::collections::VecDeque<f64>>,
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
