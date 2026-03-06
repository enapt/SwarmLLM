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
List models with shard status, VRAM estimates, and acquisition state. Each model includes an `mmproj` field with `available` (bool), `local` (bool), and `holders` (count) for VLM vision encoder status.

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
Search HuggingFace for GGUF models. Returns results grouped by repository with quantization variants, recommended variant, and VRAM fitness indicator.

Response format:
```json
[{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "downloads": 50000,
  "likes": 120,
  "variants": [
    { "filename": "...Q4_K_M.gguf", "size_bytes": 668000000, "quant": "Q4_K_M" },
    { "filename": "...Q8_0.gguf", "size_bytes": 1100000000, "quant": "Q8_0" }
  ],
  "recommended_variant": "Q4_K_M",
  "fits_vram": true
}]
```

### GET /api/admin/hf/probe?repo_id=...&filename=...
Probe a remote GGUF file (size, shard layout).

### POST /api/admin/hf/download-shards
Download specific shard indices from HuggingFace. Bearer auth required.

Supports `peer_fair_share: true` for smart distribution — the backend computes a deterministic fair share of shards using BLAKE3(node_id || model_id), and peers with auto-manage enabled auto-acquire the rest.

```bash
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct.Q4_K_M.gguf", "peer_fair_share": true}'
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
WebSocket for live updates. Pushes the following event types:

| Event | Trigger | Data |
|-------|---------|------|
| `stats_update` | Every 2s | Peer count, credits, acquisitions, shard registry |
| `prune_event` | Shard auto-pruned | Model ID, shard index, freed bytes, holder counts |
| `models_changed` | Shard download/load/prune | (none — signals dashboard to refresh) |
| `lan_peer_discovered` | mDNS peer found | Peer count |
| `update_available` | New version detected | Version info, changelog |
