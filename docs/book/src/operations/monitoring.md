# Monitoring with Grafana

SwarmLLM ships with a pre-built Grafana dashboard and Prometheus configuration in the `monitoring/` directory.

## Quick Start

```bash
cd monitoring/
cp .env.example .env   # required: sets the Grafana admin login
docker compose up -d
```

This starts:
- **Prometheus** at `http://localhost:9090` — scrapes SwarmLLM metrics
- **Grafana** at `http://localhost:3000` — visualizes metrics (log in with the
  user and password in `.env`; compose refuses to start without them)

> ⚠ **Known issue:** the bundled `prometheus.yml` scrapes `localhost:8800`,
> which inside the Prometheus container is the container itself, so this stack
> currently shows empty panels. Until that is fixed, run Prometheus directly on
> the machine that runs the node (see Manual Setup below).

The SwarmLLM dashboard is auto-provisioned on first start.

## Dashboard Panels

The Grafana dashboard includes:

### Node Overview
- Connected Peers (stat)
- Total Inference Requests (stat)
- Credit Balance (stat)
- Shards Hosted (stat)

### Inference
- Request Rate (req/s over time)
- Latency Percentiles (p50, p90, p99)
- Latency Distribution (histogram)
- Average Inference Latency (gauge)

### Network & Peers
- Connected Peers Over Time

### Storage & Shards
- Hosted Shards Over Time

### Credits
- Credit Balance Over Time

## Manual Setup

If you already have Prometheus and Grafana running:

### 1. Configure Prometheus

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: "swarmllm"
    static_configs:
      - targets: ["localhost:8800"]
```

`/metrics` answers without a key only to a request made **on the node's own
machine** (and not even then if `api.metrics_auth_required = true`). A
Prometheus running anywhere else — another computer, or a container — must send
the node's API key, or every scrape is refused with 401:

```yaml
scrape_configs:
  - job_name: "swarmllm"
    authorization:
      type: Bearer
      credentials_file: /path/to/api_key   # a copy of the node's `api_key` file
    static_configs:
      - targets: ["node1:8800"]
```

### 2. Import Dashboard

1. Open Grafana → Dashboards → Import
2. Upload `monitoring/grafana-dashboard.json`
3. Select your Prometheus data source
4. Click Import

## Multi-Node Monitoring

For monitoring multiple SwarmLLM nodes, add all targets. Every node has its
own API key, so a Prometheus scraping several nodes from elsewhere needs one
scrape job per node, each with that node's `credentials_file` (see above):

```yaml
scrape_configs:
  - job_name: "swarmllm"
    static_configs:
      - targets:
          - "node1:8800"
          - "node2:8800"
          - "node3:8800"
```

Or use file-based service discovery:
```yaml
scrape_configs:
  - job_name: "swarmllm"
    file_sd_configs:
      - files: ["swarmllm-targets.json"]
        refresh_interval: 30s
```

## Alerting

Example alert rules for Prometheus:

```yaml
groups:
  - name: swarmllm
    rules:
      - alert: NoPeersConnected
        expr: swarmllm_peers_connected == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "SwarmLLM node has no connected peers"

      # More than 1% of requests slower than 10 s, i.e. p99 > 10 s.
      - alert: HighInferenceLatency
        expr: |
          (
            sum(rate(swarmllm_inference_latency_seconds_bucket{le="+Inf"}[5m]))
            - sum(rate(swarmllm_inference_latency_seconds_bucket{le=~"10|10.0"}[5m]))
          ) / sum(rate(swarmllm_inference_latency_seconds_bucket{le="+Inf"}[5m])) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "More than 1% of inference requests take longer than 10 seconds"
```

Two traps this rule avoids:

- **`histogram_quantile(...) > 10` can never fire.** The histogram's highest
  finite bucket is 10 s, and when a quantile lands in the `+Inf` bucket
  Prometheus returns that bucket's lower edge — 10 — however slow requests
  really are. Counting the requests above the bucket is the reliable form.
- **Prometheus 3 rewrites `le="10"` as `le="10.0"`** on ingestion, so a rule
  matching the integer spelling silently matches nothing there. The regex
  matches both.
