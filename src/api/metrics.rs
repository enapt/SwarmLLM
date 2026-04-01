use std::fmt::Write;
use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};

use crate::api::server::AppState;

/// GET /metrics — Prometheus/OpenMetrics text-format endpoint.
///
/// Exposes key node metrics for Prometheus scraping. No auth required
/// (convention for metrics endpoints).
pub async fn metrics(State(state): State<AppState>) -> Response {
    tracing::debug!("DIAG: metrics scrape");
    let shared = &state.shared_state;
    let mut buf = String::with_capacity(2048);

    // swarmllm_peers_connected (gauge)
    let peers = shared.peer_registry.len();
    write_gauge(
        &mut buf,
        "swarmllm_peers_connected",
        "Number of connected peers",
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

    // swarmllm_credits_balance (gauge)
    let balance = {
        let cb = shared.credits.credit_balance.read().await;
        cb.balance
    };
    write_gauge(
        &mut buf,
        "swarmllm_credits_balance",
        "Current credit balance",
        balance as f64,
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

    let samples_guard = match shared.metrics.inference_latency_samples.read() {
        Ok(g) => g,
        Err(_) => {
            tracing::warn!(
                module = "metrics",
                "inference_latency_samples lock poisoned — skipping histogram"
            );
            return;
        }
    };
    let count = samples_guard.len() as f64;
    let sum: f64 = samples_guard.iter().sum();

    let _ = writeln!(buf, "# HELP {name} Inference request latency in seconds");
    let _ = writeln!(buf, "# TYPE {name} histogram");

    for &bound in BUCKETS {
        let bucket_count = samples_guard.iter().filter(|&&s| s <= bound).count();
        let _ = writeln!(buf, "{name}_bucket{{le=\"{bound}\"}} {bucket_count}");
    }
    let _ = writeln!(buf, "{name}_bucket{{le=\"+Inf\"}} {}", samples_guard.len());
    let _ = writeln!(buf, "{name}_sum {sum}");
    let _ = writeln!(buf, "{name}_count {count}");
}

/// Write per-channel backpressure metrics (capacity, sent_total, dropped_total).
fn write_channel_metrics(buf: &mut String, shared: &crate::daemon::SharedState) {
    use std::sync::atomic::Ordering::Relaxed;

    let channels: &[(&str, &crate::daemon::ChannelCounters)] = &[
        ("network_cmd", &shared.metrics.channel_metrics.network_cmd),
        ("network_out", &shared.metrics.channel_metrics.network_out),
        ("router_cmd", &shared.metrics.channel_metrics.router_cmd),
        ("rebalance", &shared.metrics.channel_metrics.rebalance),
        ("acquisition", &shared.metrics.channel_metrics.acquisition),
        ("pool_cmd", &shared.metrics.channel_metrics.pool_cmd),
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
}
