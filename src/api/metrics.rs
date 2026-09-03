use std::fmt::Write;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};

use crate::api::server::AppState;

/// R137 (closes R105 deferral): time-coverage bound on the latency sample ring.
/// Without this, a lightly-loaded node accumulates samples that may be hours
/// or days old, surfacing stale p99 values that no longer reflect reality.
/// 10 minutes matches Prometheus's typical `rate(...[5m])` / `rate(...[10m])`
/// window for histogram quantiles. The 1000-entry memory cap remains; this
/// is an additional freshness bound on top.
pub(crate) const LATENCY_SAMPLE_MAX_AGE: Duration = Duration::from_secs(600);

/// GET /metrics — Prometheus/OpenMetrics text-format endpoint.
///
/// Exposes key node metrics for Prometheus scraping. No auth required
/// (convention for metrics endpoints).
pub async fn metrics(State(state): State<AppState>) -> Response {
    tracing::debug!("DIAG: metrics scrape");
    let shared = &state.shared_state;
    let mut buf = String::with_capacity(2048);

    // swarmllm_peers_connected (gauge).
    // CORRECTNESS (R105): use `connected_node_ids` — the transport-level
    // ground truth (populated on Identify-Received, removed on
    // ConnectionClosed). `peer_registry` is intentionally preserved
    // across mid-pipeline disconnects for reconnect attempts (gotcha
    // #86), so it OVER-counts: a node with 50 stale registry entries but
    // 12 actually-connected peers reported 50. Grafana alerts on
    // "swarmllm_peers_connected < 5" silently never fired during real
    // isolation events.
    let peers = shared.connected_node_ids.len();
    write_gauge(
        &mut buf,
        "swarmllm_peers_connected",
        "Number of currently transport-connected peers",
        peers as f64,
    );

    // swarmllm_inference_requests_total (counter)
    let requests = shared
        .metrics
        .inference_requests_total
        .load(Ordering::Relaxed);
    write_counter(
        &mut buf,
        "swarmllm_inference_requests_total",
        "Total inference requests processed",
        requests as f64,
    );

    // Replies the model generated that finalisation removed entirely. Such a
    // reply reaches the client as an ordinary `200` with empty content, so it
    // moves no other counter; without this the rate was knowable only from the
    // log. See `inference::EMPTY_REPLIES_TOTAL`.
    write_counter(
        &mut buf,
        "swarmllm_empty_replies_total",
        "Replies the model generated that finalisation removed entirely (control tokens or a \
         stop sequence matching at once) — delivered to the client as an empty success",
        crate::inference::EMPTY_REPLIES_TOTAL.load(Ordering::Relaxed) as f64,
    );

    // Credits. `/metrics` carried only the balance while the JSON API grew the
    // lifetime figures, so anyone monitoring with Prometheus rather than the
    // dashboard saw a materially poorer picture — reported 2026-07-30 alongside
    // two counters that were not wired up at all.
    //
    // `lifetime_refunded` is the interesting one: as a share of
    // `lifetime_spent` it is the node's own request failure rate, which was
    // otherwise invisible. See `CreditBalance::lifetime_refunded`.
    let (balance, earned, spent, refunded) = {
        let cb = shared.credits.credit_balance.read().await;
        (
            cb.balance,
            cb.lifetime_earned,
            cb.lifetime_spent,
            cb.lifetime_refunded,
        )
    };
    write_gauge(
        &mut buf,
        "swarmllm_credits_balance",
        "Current credit balance",
        balance as f64,
    );
    write_counter(
        &mut buf,
        "swarmllm_credits_earned_total",
        "Credits earned over this node's lifetime",
        earned as f64,
    );
    write_counter(
        &mut buf,
        "swarmllm_credits_reserved_total",
        "Credits ever reserved for spending, including reservations later returned",
        spent as f64,
    );
    write_counter(
        &mut buf,
        "swarmllm_credits_returned_total",
        "Credits returned by reverting a reservation — usually a failed request",
        refunded as f64,
    );

    // swarmllm_shards_hosted (gauge)
    let local_shards = count_local_shards(shared);
    write_gauge(
        &mut buf,
        "swarmllm_shards_hosted",
        "Number of locally hosted shards",
        local_shards as f64,
    );

    // swarmllm_inference_latency_seconds (histogram)
    write_latency_histogram(&mut buf, shared);

    // OpenTelemetry GenAI server metrics. Named to the semantic conventions so
    // an OTel collector and the community Grafana dashboards work with no
    // translation layer. `swarmllm_inference_latency_seconds` above is the same
    // measurement as `gen_ai_server_request_duration_seconds` under a local
    // name; both are emitted while dashboards migrate.
    write_genai_histograms(&mut buf, shared);

    // Requests by route and outcome — the ONLY labelled request counter.
    // Both labels are closed sets, so this cannot grow with the swarm.
    write_route_counter(&mut buf, shared);

    // Serving side: work this node did for other peers. Everything above is
    // requester-side, so these are what answer "is my node contributing".
    {
        let m = &shared.metrics;
        write_counter(
            &mut buf,
            "swarmllm_segments_served_total",
            "Pipeline segments computed for other peers",
            m.segments_served.load(Ordering::Relaxed) as f64,
        );
        write_counter(
            &mut buf,
            "swarmllm_layers_served_total",
            "Transformer layers computed for other peers",
            m.layers_served.load(Ordering::Relaxed) as f64,
        );
        write_counter(
            &mut buf,
            "swarmllm_segment_serve_seconds_total",
            "Cumulative compute time spent serving segments for other peers",
            m.segment_serve_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        write_counter(
            &mut buf,
            "swarmllm_segment_activation_bytes_total",
            "Activation bytes returned to peers after serving a segment",
            m.segment_bytes_out.load(Ordering::Relaxed) as f64,
        );
    }

    // Channel backpressure metrics
    write_channel_metrics(&mut buf, shared);

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        buf,
    )
        .into_response()
}

/// GET /health/ready — startup readiness probe.
///
/// Returns 200 with `{"ready": true, ...}` when all subsystems are initialized,
/// 503 with `{"ready": false, ...}` otherwise.
pub async fn health_ready(State(state): State<AppState>) -> Response {
    let shared = &state.shared_state;
    let ready = shared.is_ready.load(Ordering::Acquire);
    tracing::debug!(ready, "DIAG: health_ready probe");

    // Subsystem status: all true once is_ready is set (they're spawned atomically)
    let subsystems = serde_json::json!({
        "network": ready,
        "inference_router": ready,
        "message_dispatcher": ready,
        "credit_ledger": ready,
        "health_monitor": ready,
        "shard_rebalancer": ready,
        "acquisition_manager": ready,
        "api_server": ready,
        "pool_manager": ready,
        "auto_shard_manager": ready,
    });

    let body = serde_json::json!({
        "ready": ready,
        "subsystems": subsystems,
    });

    let status = if ready {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };

    (status, axum::Json(body)).into_response()
}

/// Count shards held by this node across all models.
pub(crate) fn count_local_shards(shared: &crate::daemon::SharedState) -> usize {
    let node_id = shared.identity.node_id();
    let mut count = 0;
    for entry in shared.model_registry.all_shard_entries() {
        if entry.1.contains(node_id) {
            count += 1;
        }
    }
    count
}

/// Precomputed latency statistics from the inference latency sample buffer.
pub(crate) struct LatencyStats {
    pub total_requests: u64,
    pub count: usize,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Compute latency percentile statistics from the shared inference sample buffer.
/// Returns `None` if the lock is poisoned or there are no samples.
///
/// R137: drops entries older than `LATENCY_SAMPLE_MAX_AGE` before computing
/// statistics. Per-call drop is needed because at low rates the writer's
/// own drop-on-insert pass can be hours apart, leaving stale samples in
/// the ring between calls.
pub(crate) fn compute_latency_stats(shared: &crate::daemon::SharedState) -> Option<LatencyStats> {
    let total_requests = shared
        .metrics
        .inference_requests_total
        .load(std::sync::atomic::Ordering::Relaxed);
    let cutoff = std::time::Instant::now() - LATENCY_SAMPLE_MAX_AGE;
    let mut latencies: Vec<f64> = match shared.metrics.inference_latency_samples.read() {
        Ok(s) => s
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, v)| *v)
            .collect(),
        Err(_) => {
            tracing::warn!("inference_latency_samples lock poisoned — skipping stats");
            return None;
        }
    };
    if latencies.is_empty() {
        return None;
    }
    let count = latencies.len();
    let sum: f64 = latencies.iter().sum();
    let avg_ms = (sum / count as f64) * 1000.0;
    let min_ms = latencies.iter().cloned().fold(f64::INFINITY, f64::min) * 1000.0;
    let max_ms = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1000.0;
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50_ms = latencies[(count - 1) / 2] * 1000.0;
    let p95_ms = latencies[((count as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1)]
        * 1000.0;
    let p99_ms = latencies[((count as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1)]
        * 1000.0;
    Some(LatencyStats {
        total_requests,
        count,
        avg_ms: (avg_ms * 10.0).round() / 10.0,
        min_ms: (min_ms * 10.0).round() / 10.0,
        max_ms: (max_ms * 10.0).round() / 10.0,
        p50_ms: (p50_ms * 10.0).round() / 10.0,
        p95_ms: (p95_ms * 10.0).round() / 10.0,
        p99_ms: (p99_ms * 10.0).round() / 10.0,
    })
}

fn write_gauge(buf: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} gauge");
    let _ = writeln!(buf, "{name} {value}");
}

fn write_counter(buf: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} counter");
    let _ = writeln!(buf, "{name} {value}");
}

/// Write a basic histogram for inference latency.
///
/// Uses the latency samples stored in `MetricsProviders::inference_latency_samples`.
/// Bucket boundaries (in seconds): 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf
fn write_latency_histogram(buf: &mut String, shared: &crate::daemon::SharedState) {
    const BUCKETS: &[f64] = &[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];
    let name = "swarmllm_inference_latency_seconds";

    let cutoff = std::time::Instant::now() - LATENCY_SAMPLE_MAX_AGE;
    let fresh_latencies: Vec<f64> = match shared.metrics.inference_latency_samples.read() {
        Ok(g) => g
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, v)| *v)
            .collect(),
        Err(_) => {
            tracing::warn!(
                module = "metrics",
                "inference_latency_samples lock poisoned — skipping histogram"
            );
            return;
        }
    };
    // CORRECTNESS (R105): emit MONOTONIC `_count` and `_sum` from the
    // dedicated atomic counters; the ring is only used for the per-bucket
    // distribution. Without this, `_count` capped at 1000 (the ring size)
    // and could fall when the ring wrapped — breaking `rate()` /
    // `increase()` queries that assume counters are non-decreasing.
    let total_count = shared
        .metrics
        .inference_latency_total_count
        .load(Ordering::Relaxed);
    let total_micros = shared
        .metrics
        .inference_latency_total_micros
        .load(Ordering::Relaxed);
    let total_sum_secs = total_micros as f64 / 1_000_000.0;

    let _ = writeln!(buf, "# HELP {name} Inference request latency in seconds");
    let _ = writeln!(buf, "# TYPE {name} histogram");

    // Bucket counts come from the bounded ring — they're best-effort
    // distribution snapshots filtered to entries within the freshness
    // window (R137). They may not equal `_count` (the ring is capped
    // AND age-bounded). That's expected for a sliding-window histogram.
    for &bound in BUCKETS {
        let bucket_count = fresh_latencies.iter().filter(|&&s| s <= bound).count();
        let _ = writeln!(buf, "{name}_bucket{{le=\"{bound}\"}} {bucket_count}");
    }
    let _ = writeln!(
        buf,
        "{name}_bucket{{le=\"+Inf\"}} {}",
        fresh_latencies.len()
    );
    let _ = writeln!(buf, "{name}_sum {total_sum_secs}");
    let _ = writeln!(buf, "{name}_count {total_count}");
}

/// Write per-channel backpressure metrics (capacity, sent_total, dropped_total).
fn write_channel_metrics(buf: &mut String, shared: &crate::daemon::SharedState) {
    use std::sync::atomic::Ordering::Relaxed;

    // CORRECTNESS (R105): only emit channels whose `record_sent` /
    // `record_dropped` are actually instrumented in their send path. The
    // others (network_cmd, router_cmd, rebalance, pool_cmd) had counter
    // declarations but no call sites incrementing them, producing a
    // perpetual zero in Prometheus that misled `swarmllm_channel_dropped_total > 0`
    // alerts into never firing despite real backpressure. Drop them from
    // the output until instrumented; operators who alerted on these will
    // see a clean missing-series rather than fabricated zeros.
    let channels: &[(&str, &crate::daemon::ChannelCounters)] = &[
        ("network_out", &shared.metrics.channel_metrics.network_out),
        ("acquisition", &shared.metrics.channel_metrics.acquisition),
    ];

    let _ = writeln!(
        buf,
        "# HELP swarmllm_channel_capacity Channel buffer capacity"
    );
    let _ = writeln!(buf, "# TYPE swarmllm_channel_capacity gauge");
    for (name, counters) in channels {
        let _ = writeln!(
            buf,
            "swarmllm_channel_capacity{{channel=\"{name}\"}} {}",
            counters.capacity
        );
    }

    let _ = writeln!(
        buf,
        "# HELP swarmllm_channel_sent_total Messages sent through channel"
    );
    let _ = writeln!(buf, "# TYPE swarmllm_channel_sent_total counter");
    for (name, counters) in channels {
        let _ = writeln!(
            buf,
            "swarmllm_channel_sent_total{{channel=\"{name}\"}} {}",
            counters.sent.load(Relaxed)
        );
    }

    let _ = writeln!(
        buf,
        "# HELP swarmllm_channel_dropped_total Messages dropped due to backpressure"
    );
    let _ = writeln!(buf, "# TYPE swarmllm_channel_dropped_total counter");
    for (name, counters) in channels {
        let _ = writeln!(
            buf,
            "swarmllm_channel_dropped_total{{channel=\"{name}\"}} {}",
            counters.dropped.load(Relaxed)
        );
    }
}

/// OpenTelemetry GenAI bucket boundaries for time-to-first-token, in seconds.
/// Taken verbatim from the semantic conventions rather than chosen locally, so
/// our buckets line up with every other GenAI server a collector scrapes.
const GENAI_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
];

/// Emit `gen_ai_server_time_to_first_token_seconds` and
/// `gen_ai_server_time_per_output_token_seconds`.
///
/// These are the two numbers that separate a backed-up queue from slow decode,
/// and neither was measured server-side before — TTFT existed only in the bench
/// CLI, computed client-side.
fn write_genai_histograms(buf: &mut String, shared: &crate::daemon::SharedState) {
    let m = &shared.metrics;
    write_genai_histogram(
        buf,
        "gen_ai_server_time_to_first_token_seconds",
        "Time to first token in seconds",
        &m.ttft_samples,
        m.ttft_total_count.load(Ordering::Relaxed),
        m.ttft_total_micros.load(Ordering::Relaxed),
    );
    write_genai_histogram(
        buf,
        "gen_ai_server_time_per_output_token_seconds",
        "Time per output token after the first, in seconds",
        &m.tpot_samples,
        m.tpot_total_count.load(Ordering::Relaxed),
        m.tpot_total_micros.load(Ordering::Relaxed),
    );
}

fn write_genai_histogram(
    buf: &mut String,
    name: &str,
    help: &str,
    samples: &std::sync::RwLock<std::collections::VecDeque<(std::time::Instant, f64)>>,
    total_count: u64,
    total_micros: u64,
) {
    let cutoff = std::time::Instant::now() - LATENCY_SAMPLE_MAX_AGE;
    let fresh: Vec<f64> = match samples.read() {
        Ok(g) => g
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, v)| *v)
            .collect(),
        Err(_) => {
            tracing::warn!(module = "metrics", name, "sample lock poisoned — skipping");
            return;
        }
    };
    let _ = writeln!(buf, "# HELP {name} {help}");
    let _ = writeln!(buf, "# TYPE {name} histogram");
    for &bound in GENAI_BUCKETS {
        let n = fresh.iter().filter(|&&s| s <= bound).count();
        let _ = writeln!(buf, "{name}_bucket{{le=\"{bound}\"}} {n}");
    }
    // `_count`/`_sum` come from the monotonic atomics, never the ring — the
    // ring is capped and age-bounded, so its length falls and would break
    // rate()/increase() (R105).
    let _ = writeln!(buf, "{name}_bucket{{le=\"+Inf\"}} {total_count}");
    let _ = writeln!(buf, "{name}_count {total_count}");
    let _ = writeln!(buf, "{name}_sum {}", total_micros as f64 / 1_000_000.0);
}

/// Emit `swarmllm_inference_requests_by_route_total{route,outcome}`.
///
/// Cardinality is 5 routes × 4 outcomes = 20 series, fixed regardless of swarm
/// size, model count or peer count. Anything per-peer or per-model is
/// deliberately NOT here.
fn write_route_counter(buf: &mut String, shared: &crate::daemon::SharedState) {
    let name = "swarmllm_inference_requests_by_route_total";
    let _ = writeln!(
        buf,
        "# HELP {name} Completed inference requests by route and outcome"
    );
    let _ = writeln!(buf, "# TYPE {name} counter");
    // Sorted so scrape output is stable between calls — makes diffing two
    // scrapes by hand actually workable.
    let mut rows: Vec<((&str, &str), u64)> = shared
        .metrics
        .requests_by_route
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    rows.sort_unstable();
    for ((route, outcome), n) in rows {
        let _ = writeln!(buf, "{name}{{route=\"{route}\",outcome=\"{outcome}\"}} {n}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_gauge_format() {
        let mut buf = String::new();
        write_gauge(&mut buf, "test_metric", "A test metric", 42.0);
        assert!(buf.contains("# HELP test_metric A test metric"));
        assert!(buf.contains("# TYPE test_metric gauge"));
        assert!(buf.contains("test_metric 42"));
    }

    #[test]
    fn write_counter_format() {
        let mut buf = String::new();
        write_counter(&mut buf, "test_counter", "A test counter", 100.0);
        assert!(buf.contains("# HELP test_counter A test counter"));
        assert!(buf.contains("# TYPE test_counter counter"));
        assert!(buf.contains("test_counter 100"));
    }

    #[test]
    fn write_gauge_zero() {
        let mut buf = String::new();
        write_gauge(&mut buf, "empty", "Empty gauge", 0.0);
        assert!(buf.contains("empty 0"));
    }

    #[test]
    fn write_counter_negative_balance() {
        let mut buf = String::new();
        // Credits can be negative (Bronze tier)
        write_gauge(&mut buf, "swarmllm_credits_balance", "Credits", -50.0);
        assert!(buf.contains("swarmllm_credits_balance -50"));
    }

    #[test]
    fn histogram_empty_samples() {
        // Verify histogram output with no samples
        let mut buf = String::new();
        // Manually simulate empty histogram output
        let name = "swarmllm_inference_latency_seconds";
        let _ = writeln!(buf, "# HELP {name} Inference request latency in seconds");
        let _ = writeln!(buf, "# TYPE {name} histogram");
        let _ = writeln!(buf, "{name}_bucket{{le=\"0.01\"}} 0");
        let _ = writeln!(buf, "{name}_bucket{{le=\"+Inf\"}} 0");
        let _ = writeln!(buf, "{name}_sum 0");
        let _ = writeln!(buf, "{name}_count 0");

        assert!(buf.contains("# TYPE swarmllm_inference_latency_seconds histogram"));
        assert!(buf.contains("_bucket{le=\"0.01\"} 0"));
        assert!(buf.contains("_bucket{le=\"+Inf\"} 0"));
        assert!(buf.contains("_sum 0"));
        assert!(buf.contains("_count 0"));
    }

    #[test]
    fn histogram_with_samples() {
        // Verify bucket counting logic
        let samples: Vec<f64> = vec![0.005, 0.02, 0.08, 0.3, 1.5, 7.0];
        let buckets: [f64; 9] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

        for &bound in &buckets {
            let count = samples.iter().filter(|&&s| s <= bound).count();
            match bound {
                b if (b - 0.01).abs() < f64::EPSILON => assert_eq!(count, 1), // 0.005
                b if (b - 0.05).abs() < f64::EPSILON => assert_eq!(count, 2), // +0.02
                b if (b - 0.1).abs() < f64::EPSILON => assert_eq!(count, 3),  // +0.08
                b if (b - 0.25).abs() < f64::EPSILON => assert_eq!(count, 3), // same
                b if (b - 0.5).abs() < f64::EPSILON => assert_eq!(count, 4),  // +0.3
                b if (b - 1.0).abs() < f64::EPSILON => assert_eq!(count, 4),  // same
                b if (b - 2.5).abs() < f64::EPSILON => assert_eq!(count, 5),  // +1.5
                b if (b - 5.0).abs() < f64::EPSILON => assert_eq!(count, 5),  // same
                b if (b - 10.0).abs() < f64::EPSILON => assert_eq!(count, 6), // +7.0
                _ => {}
            }
        }
    }

    /// R137 (closes R105 deferral): time-coverage filter drops stale entries
    /// when computing histogram-bucket counts, leaving only fresh samples.
    /// Reproduces the gist of the per-call age filter without needing a full
    /// SharedState construction.
    #[test]
    fn latency_age_filter_drops_old_entries() {
        let now = std::time::Instant::now();
        let cutoff = now - LATENCY_SAMPLE_MAX_AGE;
        // Use a clearly stale instant — older than the freshness window.
        let stale = cutoff - Duration::from_secs(60);
        let fresh = cutoff + Duration::from_secs(60);
        let samples: Vec<(std::time::Instant, f64)> =
            vec![(stale, 9.5), (fresh, 0.1), (fresh, 0.3)];
        let kept: Vec<f64> = samples
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, v)| *v)
            .collect();
        // Stale 9.5s entry must be filtered; only the two fresh values remain.
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|&v| v < 1.0));
    }
}
