# Prometheus Metrics

SwarmLLM exposes a Prometheus-compatible metrics endpoint at `GET /metrics`. No authentication required (standard convention for metrics endpoints).

## Available Metrics

### Core Metrics

| Metric | Type | Description |
|---|---|---|
| `swarmllm_peers_connected` | gauge | Number of connected peers |
| `swarmllm_inference_requests_total` | counter | Total inference requests processed |
| `swarmllm_credits_balance` | gauge | Current credit balance |
| `swarmllm_shards_hosted` | gauge | Number of locally hosted shards |
| `swarmllm_inference_latency_seconds` | histogram | Inference request latency |
| `swarmllm_inference_requests_by_route_total{route,outcome}` | counter | Completed requests by route and outcome |

`route` is one of `local`, `split`, `distributed`, `relayed`, `cloud`; `outcome`
is `ok`, `error` or `cancelled`. Both are closed sets, so this metric is 20
series regardless of how large the swarm grows. **Per-peer, per-model and
per-shard breakdowns are deliberately not exported here** — that label set grows
with the swarm and would eventually break the scrape. Fetch them from
`GET /api/admin/performance` instead, which is served on request and retains
nothing.

### OpenTelemetry GenAI Metrics

Named to the [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
so an OTel collector and the community Grafana dashboards work without a
translation layer.

| Metric | Type | Description |
|---|---|---|
| `gen_ai_server_time_to_first_token_seconds` | histogram | Queue + prefill: how long until the first token |
| `gen_ai_server_time_per_output_token_seconds` | histogram | Decode cost per token **after** the first |

These are the two figures that separate a backed-up queue from slow generation;
end-to-end latency alone cannot. `swarmllm_inference_latency_seconds` is the same
measurement as the conventions' `gen_ai.server.request.duration` under a local
name, and both are exported while dashboards migrate.

Only requests that emitted an incremental token contribute to these histograms.
A non-streaming path never stamps a first token, so there is no honest way to
split decode out of its total and it is omitted rather than counted as zero.

### Serving-Side Metrics

Work this node performed **for other peers**. Every metric above measures
requests this node *made*; these measure what it *gave*.

| Metric | Type | Description |
|---|---|---|
| `swarmllm_segments_served_total` | counter | Pipeline segments computed for other peers |
| `swarmllm_layers_served_total` | counter | Transformer layers computed for other peers |
| `swarmllm_segment_serve_seconds_total` | counter | Cumulative compute time spent serving |
| `swarmllm_segment_activation_bytes_total` | counter | Activation bytes returned to peers |

`rate(swarmllm_segment_serve_seconds_total[5m]) / rate(swarmllm_layers_served_total[5m])`
gives seconds per layer served — the figure other peers' schedulers actually rank
this node on.

### Channel Metrics

Internal channel health metrics for monitoring backpressure:

| Metric | Type | Description |
|---|---|---|
| `swarmllm_channel_capacity{channel="..."}` | gauge | Channel buffer capacity |
| `swarmllm_channel_sent_total{channel="..."}` | counter | Messages sent through channel |
| `swarmllm_channel_dropped_total{channel="..."}` | counter | Messages dropped due to backpressure |

### Histogram Buckets

`swarmllm_inference_latency_seconds` uses these bucket boundaries (in seconds):
`0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf`

The two `gen_ai_server_*` histograms use the boundaries published in the
OpenTelemetry GenAI conventions rather than locally chosen ones, so buckets line
up with every other GenAI server a collector scrapes:
`0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, +Inf`

For every histogram, `_count` and `_sum` come from monotonic counters rather than
the in-memory sample ring. The ring is both size- and age-bounded, so its length
falls when it wraps, which would break `rate()` and `increase()`.

## Scraping Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "swarmllm"
    static_configs:
      - targets: ["localhost:8800"]
```

## Example Queries

```promql
# Request rate (requests per second over 5 minutes)
rate(swarmllm_inference_requests_total[5m])

# P50 latency
histogram_quantile(0.50, rate(swarmllm_inference_latency_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(swarmllm_inference_latency_seconds_bucket[5m]))

# Average latency
rate(swarmllm_inference_latency_seconds_sum[5m]) / rate(swarmllm_inference_latency_seconds_count[5m])

# P95 time to first token — is the queue backed up, or is generation slow?
histogram_quantile(0.95, rate(gen_ai_server_time_to_first_token_seconds_bucket[5m]))

# P95 per-token decode cost
histogram_quantile(0.95, rate(gen_ai_server_time_per_output_token_seconds_bucket[5m]))

# Share of requests that left this machine
sum(rate(swarmllm_inference_requests_by_route_total{route=~"distributed|relayed"}[5m]))
  / sum(rate(swarmllm_inference_requests_by_route_total[5m]))

# Error rate by route
sum by (route) (rate(swarmllm_inference_requests_by_route_total{outcome="error"}[5m]))

# Seconds of compute contributed per layer served
rate(swarmllm_segment_serve_seconds_total[5m]) / rate(swarmllm_layers_served_total[5m])
```

## Health Check

### GET /health/ready

Readiness probe returning subsystem status. Returns 200 when ready, 503 otherwise. No auth required.

```json
{
  "ready": true,
  "subsystems": {
    "network": true,
    "inference_router": true,
    "api_server": true,
    ...
  }
}
```
