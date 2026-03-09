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

### Channel Metrics

Internal channel health metrics for monitoring backpressure:

| Metric | Type | Description |
|---|---|---|
| `swarmllm_channel_capacity{channel="..."}` | gauge | Channel buffer capacity |
| `swarmllm_channel_sent_total{channel="..."}` | counter | Messages sent through channel |
| `swarmllm_channel_dropped_total{channel="..."}` | counter | Messages dropped due to backpressure |

### Histogram Buckets

The latency histogram uses these bucket boundaries (in seconds):
`0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf`

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
