# Deployment Guide

## Single Node

The simplest deployment — just run the binary:

```bash
./swarmllm run
```

This starts the daemon on port 8800 with default settings.

## Production Configuration

For production use, create a config file:

```toml
[node]
listen_port = 8800
contribution = "maximum"

[resources]
max_gpu_vram_mb = 0        # Auto-detect
max_disk_mb = 100000       # 100 GB

[inference]
gpu_layers = 99            # Offload all layers to GPU
max_concurrent_requests = 20
max_batch_size = 4
session_timeout_seconds = 600

[auto_manage]
enabled = true
max_storage_mb = 50000
max_concurrent_downloads = 5

[logging]
level = "info"
format = "json"            # Structured logs for production
file = "/var/log/swarmllm.log"

[ui]
open_browser_on_start = false

[identity]
region = "US"
```

## Systemd Service

Create `/etc/systemd/system/swarmllm.service`:

```ini
[Unit]
Description=SwarmLLM P2P Inference Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=swarmllm
ExecStart=/usr/local/bin/swarmllm run --config /etc/swarmllm/config.toml
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/swarmllm /var/log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now swarmllm
```

## Docker

### Quick Start (Recommended)

```bash
# Download compose file and env template
curl -LO https://raw.githubusercontent.com/enapt/SwarmLLM/main/docker-compose.yml
curl -LO https://raw.githubusercontent.com/enapt/SwarmLLM/main/.env.example
cp .env.example .env

# CPU
docker compose up -d

# GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up -d
```

### Pre-built Images

| Image | Description |
|---|---|
| `ghcr.io/enapt/swarmllm:latest` | CPU-only (Debian bookworm-slim) |
| `ghcr.io/enapt/swarmllm:latest-cuda` | NVIDIA GPU (CUDA 12.4 runtime) |

Versioned tags follow semver — e.g. `0.3.9-alpha`, `0.3.9-alpha-cuda` (pre-releases publish the full version; stable releases additionally get the `0.3` / `0.3-cuda` minor tags).

### Manual Docker Run

```bash
# CPU
docker run -d \
  --name swarmllm \
  --restart unless-stopped \
  -p 8800:8800/tcp \
  -p 8810:8810/tcp \
  -p 8800:8800/udp \
  -v swarmllm-data:/data \
  -v /path/to/models:/data/models \
  --env-file .env \
  ghcr.io/enapt/swarmllm:latest

# GPU
docker run -d \
  --gpus all \
  --name swarmllm \
  --restart unless-stopped \
  -p 8800:8800/tcp \
  -p 8810:8810/tcp \
  -p 8800:8800/udp \
  -v swarmllm-data:/data \
  -v /path/to/models:/data/models \
  --env-file .env \
  ghcr.io/enapt/swarmllm:latest-cuda
```

### Build from Source

```bash
# CPU
docker build -t swarmllm .

# CUDA
docker build -f Dockerfile.cuda -t swarmllm:cuda .
```

### Multi-Node Dev Cluster

For development and testing, a 3-node compose file is available:

```bash
docker compose -f docker-compose.dev.yml up
```

Nodes are at `localhost:8800`, `localhost:8801`, `localhost:8802`. Add GPU support:

```bash
docker compose -f docker-compose.dev.yml -f docker-compose.cuda.dev.yml up
```

## Multi-Node Cluster

### Same LAN

Nodes on the same network discover each other automatically via mDNS. Just start multiple instances on different ports:

```bash
# Node 1
./swarmllm run -p 8800

# Node 2
./swarmllm run -p 8801 -d ~/.local/share/swarmllm-node2
```

### Across Networks

Use bootstrap peers or invite codes:

```bash
# Node 1 (get its address from the dashboard or logs)
./swarmllm run

# Node 2 (connect to Node 1)
./swarmllm run --bootstrap "/ip4/NODE1_IP/udp/8800/quic-v1/p2p/PEER_ID"
```

### Split Inference Cluster

For a dedicated split-inference setup across multiple machines:

```bash
# Machine A: shards 0-3
./swarmllm run --shards "0-3" --bootstrap "/ip4/MACHINE_B/udp/8800/quic-v1/p2p/..."

# Machine B: shards 4-7
./swarmllm run --shards "4-7" --bootstrap "/ip4/MACHINE_A/udp/8800/quic-v1/p2p/..."
```

## Firewall

Open TCP port 8800 (HTTP API), TCP port 8810 (P2P), and optionally UDP port 8800 (QUIC):

```bash
# Linux (ufw)
sudo ufw allow 8800/tcp    # HTTP API
sudo ufw allow 8810/tcp    # P2P (Noise+Yamux, primary transport)
sudo ufw allow 8800/udp    # P2P (QUIC, optional)

# Linux (iptables)
sudo iptables -A INPUT -p tcp --dport 8800 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8810 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 8800 -j ACCEPT
```

## Reverse Proxy (Optional)

If you want to put the HTTP API behind nginx:

```nginx
server {
    listen 443 ssl;
    server_name swarmllm.example.com;

    location / {
        proxy_pass http://127.0.0.1:8800;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Note: The reverse proxy only handles HTTP traffic. P2P (QUIC/UDP) must still be accessible directly on port 8800.

## Cloud Provider API Keys

To use cloud model fallback, configure provider API keys via:

1. **Dashboard:** Settings page in the web UI
2. **Environment file:** Place a `.env` file in the data directory with standard variable names:

```bash
# ~/.local/share/swarmllm/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
CEREBRAS_API_KEY=...
SAMBANOVA_API_KEY=...
FIREWORKS_API_KEY=...
TOGETHER_API_KEY=...
DEEPINFRA_API_KEY=...
MOONSHOT_API_KEY=...
```

3. **Shell environment:** Export the same variables before starting the daemon
