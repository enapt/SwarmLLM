# Admin API

Admin endpoints are CORS-protected. Most read-only endpoints don't require Bearer auth; write operations do.

## Node Management

### GET /api/admin/stats
Node statistics and hardware info.

### GET /api/admin/peers
Connected peers with latency, trust scores, and hosted models.

### GET /api/admin/credits
Credit balance and tier info.

### GET /api/admin/network-map
Geographic distribution of peers and shards across regions. Each entry includes the total peer count for that region, per-model shard-holder counts, per-model request demand rates, coverage gaps (models with zero holders in the region), and per-model replication targets derived from pool size and demand. Includes the local node in its auto-detected or configured region.

Response:
```json
{
  "regions": {
    "US": {
      "total": 3,
      "models": { "tinyllama-1.1b-q4-k-m": 2 },
      "demand": { "tinyllama-1.1b-q4-k-m": 5 },
      "coverage_gaps": [],
      "replication_target": { "tinyllama-1.1b-q4-k-m": 2 }
    }
  }
}
```

### GET/PUT /api/admin/config
Read or update daemon configuration. PUT requires Bearer auth.

### POST /api/admin/config/reload
Hot-reload operational parameters without restart. Bearer auth required.

### POST /api/admin/shutdown
Gracefully shut down the node. Localhost only, Bearer auth required.

## Model Management

### GET /api/admin/models
List models with shard status, VRAM estimates, and acquisition state. Each model includes:
- `mmproj` field with `available` (bool), `local` (bool), and `holders` (count) for VLM vision encoder status
- `trust_level` field: one of `"Discovered"`, `"Pinned"`, `"DemandVerified"`, or `"NetworkPopular"` indicating the model's trust status (auto-manage only downloads shards for DemandVerified+ or Pinned models, OR for any model where this node already hosts at least one shard — gap-filling exemption). R141: trusted curator publishers (meta-llama, Qwen, mistralai, bartowski, unsloth, etc.) promote at 10k HF downloads instead of 100k

### POST /api/admin/models/{id}/add
Trigger model acquisition from the network.

### GET /api/admin/models/{id}/status
Check model acquisition progress.

### DELETE /api/admin/models/{id}
Remove model (shards + manifest + state).

### DELETE /api/admin/models/{id}/shards/{index}
Delete a single shard.

### GET/PUT /api/admin/models/{id}/auto-manage
Per-model auto-manage policy (including prune toggle).

### GET/PUT /api/admin/models/{id}/encrypted-pipeline
Per-model encrypted pipeline toggle. GET returns current status, readiness (whether local node holds first + last shard), and overhead note. PUT enables/disables with body `{"enabled": true}`. Requires the local node to hold shard 0 and the final shard. Returns a warning for 2-shard models (fully local, no distribution benefit). Setting is persisted to the database and survives restarts. Falls back to global `encrypted_pipeline` config if no per-model override is set.

### PUT /api/admin/models/{id}/shards/{index}/lock
Lock/unlock a shard to prevent auto-pruning.

## Storage & Shards

### POST /api/admin/rescan-shards
Rescan local shard files on disk and update the model registry and network announcements without restarting the daemon. Useful after manually placing shard files in the data directory. Bearer auth required.

Response:
```json
{ "status": "ok", "models_updated": ["model-id-1"], "count": 1 }
```

### GET /api/admin/models/{id}/metadata
Read parsed GGUF metadata from a locally-stored model header (`gguf_header.bin`). Returns architecture parameters, tokenizer settings, quantization type, and all raw metadata key/value pairs (tokenizer vocabulary arrays are excluded). Returns 400 if no header file exists for the model.

Response shape:
```json
{
  "model_id": "...",
  "general": { "name": "...", "architecture": "llama", "architecture_supported": true, "file_type": 11, "quantization": "Q4_K_M" },
  "model": { "context_length": 4096, "block_count": 32, "embedding_length": 4096, "head_count": 32, "head_count_kv": 8, "rope_dimension_count": 128, "rope_freq_base": 500000.0, "layer_norm_rms_epsilon": 1e-5, "vocab_size": 32000 },
  "tokenizer": { "model": "llama", "pre": "...", "eos_token_id": 2, "bos_token_id": 1, "padding_token_id": null },
  "tensors": { "count": 291, "data_offset": 131072 },
  "raw": [{ "key": "general.architecture", "value": "llama" }, ...]
}
```

### POST /api/admin/models/{id}/shards/{index}/download
Trigger a P2P download of a specific shard that is not yet held locally. The daemon first checks for P2P peers that hold the shard (picking the best peer by LAN-proximity, latency, and trust), then falls back to returning HuggingFace source info if no peers are available. Bearer auth required.

Responses:
- `{ "status": "already_local", ... }` — shard is already on disk
- `{ "status": "downloading", "source": "p2p", "peer": "...", ... }` — P2P download started
- `{ "status": "use_hf", "source": "huggingface", "repo_id": "...", "filename": "...", ... }` — no P2P peers, use `hf/download-shards` instead
- 400 if no peers and no HuggingFace source known

### POST /api/admin/models/{id}/shards/{index}/unload
Unload a single shard from memory (VRAM/RAM) without deleting the file from disk. Narrows the model's shard window to exclude this shard and restarts the worker subprocess. If this is the last loaded shard, the model is fully unloaded. Bearer auth required.

Response:
```json
{ "status": "unloaded", "model_id": "...", "shard_index": 0, "remaining_loaded": [1, 2] }
```

### POST /api/admin/models/{id}/shards/{index}/load
Load a shard that is on disk into memory. The shard must already be present locally (use `/download` first if not). Expands the model's shard window to include the shard and restarts the worker subprocess. Bearer auth required.

Response:
```json
{ "status": "loaded", "model_id": "...", "shard_index": 0, "loaded_shards": [0, 1, 2] }
```

### POST /api/admin/models/{id}/unload
Unload an entire model from memory (VRAM/RAM) without deleting any files from disk. Evicts all split-model entries, kills the worker subprocess, clears GGUF metadata cache, and clears the loaded-model record. Bearer auth required.

Response:
```json
{ "status": "unloaded", "model_id": "...", "model_name": "...", "segments_removed": 2, "estimated_freed_mb": 4096 }
```

### GET /api/admin/shard-storage
Per-model storage breakdown, disk and VRAM usage.

### GET /api/admin/prune-history
Recent auto-prune events.

### GET/PUT /api/admin/schedule
Resource schedule management.

## HuggingFace Integration

### GET /api/admin/hf/search?query=...
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

### GET /api/admin/hf/source/{model_id}
Look up the HuggingFace source (repo + filename) for a locally-known model. First checks the in-memory source cache and the probe cache, then auto-discovers by searching HuggingFace if neither has an entry. If found via auto-discovery the result is cached to the database and `hf_source.json` in the model directory.

Response:
```json
{ "model_id": "...", "repo_id": "TheBloke/TinyLlama-...-GGUF", "filename": "tinyllama-...Q4_K_M.gguf" }
```

### GET /api/admin/downloads
List the download queue with per-shard progress, speed, and source.

### POST /api/admin/downloads/{model_id}/cancel
Cancel an in-progress download.

## LoRA Adapters

### GET /api/admin/adapters
List all registered LoRA adapters with their metadata (id, name, base model, rank, alpha, path).

Response: `{ "adapters": [ { "id": "...", "name": "...", "base_model": "...", "rank": 16, "alpha": 32.0, "path": "..." } ] }`

### POST /api/admin/adapters
Register a LoRA adapter from a safetensors file. Bearer auth required. Path traversal is blocked. If `id` is omitted, a UUID is generated.

Request body:
```json
{ "id": "my-adapter", "name": "My Adapter", "base_model": "tinyllama-...", "rank": 16, "alpha": 32.0, "path": "adapters/my-adapter.safetensors" }
```

`path` may be absolute or relative to `<data_dir>/adapters/`.

Response: `{ "status": "ok", "adapter": { ... } }`

### DELETE /api/admin/adapters/{id}
Unregister a LoRA adapter. Does not delete the file from disk. Bearer auth required. Returns 400 if the id is not found.

Response: `{ "status": "ok", "message": "Adapter 'my-adapter' removed" }`

## Cloud Providers

### GET /api/admin/providers
List configured cloud providers (name + configured flag, no keys exposed).

### PUT /api/admin/providers
Update cloud provider API keys. Bearer auth required. Keys are encrypted at rest.

### GET /api/admin/provider-models
List available models from all configured cloud providers. Results are cached for 60 seconds; stale results are returned immediately and refreshed in the background. Includes models from OpenAI, Anthropic (static list), DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together AI, DeepInfra, and Moonshot/Kimi.

Response: `{ "models": [ { "id": "gpt-4o", "name": "GPT-4o", "provider": "openai" } ] }`

### GET /api/admin/provider-health
Probe each configured provider by sending a tiny `max_tokens=1` inference request (using a suitable test model per provider). All probes run in parallel with a connect timeout.

Response:
```json
{ "providers": [ { "provider": "openai", "status": "up", "latency_ms": 320, "detail": "" } ] }
```

Status values: `up`, `rate_limited`, `overloaded`, `timeout`, `unreachable`, `error_<code>`.

### POST /api/admin/provider-model-status
Probe availability and latency for a list of specific cloud model IDs (up to 20 per request). Sends a `max_tokens=1` request to each model's provider endpoint. Anthropic models are skipped (no cloud proxy probing). Bearer auth not required.

Request body: `{ "models": ["gpt-5.4", "claude-sonnet-5", "deepseek-v4-flash"] }`

Response:
```json
{ "models": [ { "model": "gpt-4o", "status": "up", "latency_ms": 210 } ] }
```

Status values: `up`, `rate_limited`, `not_found`, `unavailable`, `timeout`, `error`.

## Claude Subscription (feature-gated)

> Requires building with `--features claude-subscription`. When the feature is not enabled, these endpoints return `{"error": "claude-subscription feature not enabled"}`.

### GET /api/admin/claude-subscription/status

Detect whether the `claude` CLI is installed and authenticated on this machine. Reads version from `claude --version` and subscription info from `~/.claude/.credentials.json` (read-only).

Response:
```json
{
  "cli_installed": true,
  "cli_version": "2.1.92 (Claude Code)",
  "authenticated": true,
  "subscription_type": "max",
  "rate_limit_tier": "default_claude_max_5x"
}
```

### PUT /api/admin/providers (claude_subscription_enabled field)

Enable or disable the Claude subscription provider. Pass `claude_subscription_enabled` alongside other provider key updates.

```json
{ "claude_subscription_enabled": true }
```

When enabled, `claude-*` model requests are routed through the local CLI subprocess instead of the Anthropic API key. The Anthropic API key (if configured) is used as fallback when disabled.

## Updates

### GET /api/admin/version
Current binary version info.

### POST /api/admin/update/check
Check for available updates. Returns version info and changelog if update available.

### POST /api/admin/update/apply
Download and apply an update. Bearer auth required.

## Discovery

### GET /api/admin/network-code
Get an encrypted shareable invite code and network phase. The code embeds the node's TCP listening address encrypted with ChaCha20Poly1305 — the IP is not visible in the code.

### POST /api/admin/join-network
Join the network via encrypted invite code (`swarm://...`) or raw multiaddr. Immediately dials the peer and saves the address to the peer cache.

## Responses API listing

### GET /api/admin/responses
List stored Responses-API records (backs the dashboard's Responses tab).
Optional query params: `?limit=N` (cap on returned records, default 100,
max 500) and `?status=...` (filter by `completed` / `in_progress` /
`cancelled` / `failed` / `queued`). See [Responses API](./responses.md)
for the user-facing surface.

## Auto-Manage Surfaces

### GET /api/admin/wishlist
R111 — ranked list of models the swarm wants. Triggers a wishlist
refresh on every call so the response always reflects current state.
Response shape:

```json
{
  "computed_at": 1727712345,
  "entries": [
    {
      "model_id": "mistral-7b-instruct-v0.3-Q4_K_M",
      "display_name": "Mistral 7B Instruct v0.3",
      "status": "serveable",
      "score": 78,
      "swarm_replicas": 4,
      "target_replicas": 3,
      "size_mb": 4368,
      "vram_required_mb": 5023,
      "shards_covered": 8,
      "total_shards": 8,
      "hosted_by_us": false,
      "why_tags": ["wishlist.why.popular_on_swarm", "wishlist.why.fits_your_memory"]
    },
    {
      "model_id": "hf-candidate:bartowski/Llama-3.2-3B-Instruct-GGUF",
      "display_name": "Llama-3.2-3B-Instruct-GGUF",
      "status": "candidate",
      "score": 65,
      "hf_repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
      "task_tags": ["chat"],
      "why_tags": ["wishlist.why.popular_on_hf", "wishlist.why.trusted_publisher", "wishlist.why.candidate_one_click"]
    }
  ]
}
```

`status` is one of `hosting` / `serveable` / `aspirational` / `candidate`
(**R141**) / `unreachable` / `blocked`. Candidate entries carry an
`hf_repo_id` field and a synthetic `model_id` (prefix `hf-candidate:`)
so the frontend can route the click to the HF browse without colliding
with real model_ids. See [Wishlist architecture](../architecture.md) for the
score formula and trust-promotion path.

### GET /api/admin/hf/trending
R112 — most recent HF trending GGUF snapshot from the background
HfWatcher (refreshed hourly). Response shape mirrors `HfTrendingSnapshot`
with `entries: [{ repo_id, downloads, likes, task_tags, created_at_secs }]`
and `fetched_at: <unix-secs>`. Empty `entries` means the watcher hasn't
completed its first poll yet (30s startup delay) OR the daemon is
configured with `auto_manage.hf_watcher_enabled = false`.

### GET /api/admin/quant-recommendations
R133 — per-family quant-choice recommendations. Returns the
highest-quality variant that fits the swarm's aggregate VRAM with
reasonable replication for each model family. R141 flipped
`auto_manage.auto_switch_quants` to default `true`, so this surface
now drives automatic trust promotion of the recommended variant — the
UI uses it as a transparency view rather than a manual action surface.

## Authentication

### GET /api/admin/api-key
Retrieve the API key. Bearer auth required.

## WebSocket

### GET /api/admin/ws
WebSocket for live updates. Pushes the following event types:

| Event | Trigger | Data |
|-------|---------|------|
| `activity_event` | Any subsystem event | kind, model_id, message, timestamp, toast_level |
| `stats_update` | Every 2s | Peer count, credits, acquisitions, shard registry, **swarm_capacity** (R110), **wishlist** (R111) |
| `peer_list` | Peer connect/disconnect | Full peer snapshot |
| `models_changed` | Shard download/load/prune | (none — signals dashboard to refresh) |
| `update_available` | New version detected | Version info, changelog |
