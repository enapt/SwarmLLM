# Admin API

Admin endpoints are CORS-protected. Most read-only endpoints don't require Bearer auth; write operations do.

## Node Management

### GET /api/admin/stats
Node statistics and hardware info.

### GET /api/admin/peers
Connected peers with latency, trust scores, and hosted models.

### GET /api/admin/credits
Credit balance and tier info.

### GET/PUT /api/admin/config
Read or update daemon configuration. PUT requires Bearer auth.

### POST /api/admin/config/reload
Hot-reload operational parameters without restart. Bearer auth required.

### POST /api/admin/shutdown
Gracefully shut down the node. Localhost only, Bearer auth required.

## Model Management

### GET /api/admin/models
List models with shard status, VRAM estimates, and acquisition state.

### POST /api/admin/models/:id/add
Trigger model acquisition from the network.

### GET /api/admin/models/:id/status
Check model acquisition progress.

### DELETE /api/admin/models/:model_id
Remove model (shards + manifest + state).

### DELETE /api/admin/models/:id/shards/:index
Delete a single shard.

### GET/PUT /api/admin/models/:id/auto-manage
Per-model auto-manage policy (including prune toggle).

### PUT /api/admin/models/:id/shards/:index/lock
Lock/unlock a shard to prevent auto-pruning.

## Storage & Shards

### GET /api/admin/shard-storage
Per-model storage breakdown, disk and VRAM usage.

### GET /api/admin/prune-history
Recent auto-prune events.

### GET/PUT /api/admin/schedule
Resource schedule management.

## HuggingFace Integration

### GET /api/admin/hf/search?q=...
Search HuggingFace for GGUF models.

### GET /api/admin/hf/probe?repo_id=...&filename=...
Probe a remote GGUF file (size, shard layout).

### POST /api/admin/hf/download-shards
Download specific shard indices from HuggingFace. Bearer auth required.

```bash
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"}'
```

### POST /api/admin/downloads/:model_id/cancel
Cancel an in-progress download.

## Discovery

### GET /api/admin/network-code
Get a shareable invite code and multiaddr.

### POST /api/admin/join-network
Join the network via invite code or multiaddr.

## Authentication

### GET /api/admin/api-key
Retrieve the API key. Bearer auth required.

## WebSocket

### GET /api/admin/ws
WebSocket for live updates (peer count, shard changes, prune events, download progress).
