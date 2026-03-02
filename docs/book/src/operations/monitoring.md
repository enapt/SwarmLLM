# Monitoring with Grafana

SwarmLLM ships with a pre-built Grafana dashboard and Prometheus configuration in the `monitoring/` directory.

## Quick Start

```bash
cd monitoring/
docker compose up -d
```

This starts:
- **Prometheus** at `http://localhost:9090` — scrapes SwarmLLM metrics
- **Grafana** at `http://localhost:3000` — visualizes metrics (login: `admin`/`admin`)

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

### 2. Import Dashboard

1. Open Grafana → Dashboards → Import
2. Upload `monitoring/grafana-dashboard.json`
3. Select your Prometheus data source
4. Click Import

## Multi-Node Monitoring

For monitoring multiple SwarmLLM nodes, add all targets:

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

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, rate(swarmllm_inference_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 inference latency exceeds 10 seconds"

      - alert: NegativeCreditBalance
        expr: swarmllm_credits_balance < 0
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "Node has negative credit balance (Bronze tier)"
```
