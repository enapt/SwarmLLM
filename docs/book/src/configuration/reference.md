# Config File Reference

Every configuration option, organized by section.

## `[node]` — Basic Node Settings

| Option | Type | Default | Description |
|---|---|---|---|
| `listen_port` | integer | `8800` | Port for web dashboard and P2P networking |
| `data_dir` | path | Platform-specific | Where SwarmLLM stores data |
| `contribution` | string | `"minimal"` | Resource contribution: `"minimal"`, `"moderate"`, `"maximum"` |

## `[resources]` — Resource Limits

| Option | Type | Default | Description |
|---|---|---|---|
| `max_gpu_vram_mb` | integer | `0` | Max GPU memory in MB. `0` = auto-detect |
| `max_ram_mb` | integer | `0` | Max system RAM in MB. `0` = auto |
| `max_disk_mb` | integer | `50000` | Max disk space in MB for model storage |
| `max_bandwidth_mbps` | integer | `0` | Max upload bandwidth. `0` = unlimited |

## `[resources.schedule]` — Usage Schedule

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Enable scheduled resource reduction |
| `reduced_hours_start` | integer | `22` | Hour (0-23) to start reduced mode |
| `reduced_hours_end` | integer | `8` | Hour (0-23) to end reduced mode |
| `reduced_contribution` | string | `"minimal"` | Contribution level during reduced hours |
| `prune_aggressiveness` | string | `"normal"` | Shard pruning during reduced hours: `"normal"`, `"aggressive"`, `"conservative"` |

## `[network]` — Networking

| Option | Type | Default | Description |
|---|---|---|---|
| `bootstrap_peers` | list | `[]` | Peer addresses to connect on startup |
| `enable_mdns` | boolean | `true` | LAN peer discovery |
| `gossip_network_id` | string | none | Custom network ID for private networks |
| `peer_exchange` | boolean | `true` | Share peer lists with connected nodes |
| `enable_relay` | boolean | `true` | Act as relay for peers behind firewalls |
| `enable_relay_client` | boolean | `true` | Use relays when behind a firewall |
| `max_peers` | integer | `200` | Max simultaneous peer connections |
| `auto_relay` | boolean | `true` | Auto-use relay when NAT detected |
| `relay_max_circuit_duration_secs` | integer | `3600` | Max relay circuit duration |
| `relay_max_circuits` | integer | `16` | Max relay circuits to serve |
| `enable_encryption` | boolean | `true` | E2E encryption for tensor forwards and control messages |
| `enable_autonat` | boolean | `true` | NAT detection. Disable on WSL2 to reduce noise |
| `enable_dcutr` | boolean | `true` | Hole punching. Disable on WSL2 to reduce noise |
| `tensor_compression` | boolean | `true` | Zstd compression for tensor payloads |
| `prefix_kv_compression` | boolean | `false` | Zstd compression for cross-node prefix-KV snapshot wire frames. Default off — meaningful win on WAN where wire size is the bottleneck; roughly neutral on localhost. Receivers always decompress regardless of this flag. |
| `tensor_compress_level` | integer | `1` | Zstd compression level (1-22, 1 = fastest). Shared between tensor and prefix-KV. |
| `tensor_compress_threshold` | integer | `1024` | Min payload bytes before compression. Shared between tensor and prefix-KV. |

## `[inference]` — AI Model Inference

| Option | Type | Default | Description |
|---|---|---|---|
| `default_model` | string | `""` | Default model. Empty = first available |
| `session_timeout_seconds` | integer | `600` | Chat session memory lifetime (10 min) |
| `max_concurrent_requests` | integer | `10` | Max parallel requests |
| `model_path` | path | none | Path to a GGUF model file |
| `gpu_layers` | integer | `0` | Layers to offload to GPU. `0` = CPU only |
| `kv_cache_ttl_secs` | integer | `600` | KV-cache lifetime |
| `max_batch_size` | integer | `1` | Max request batch size. `1` = no batching. When `> 1`, both local and remote forward requests batch together via `BatchForwarder`, filling pipeline bubbles in distributed inference |
| `batch_timeout_ms` | integer | `50` | Ms to wait for additional requests before dispatching a partial batch. `0` = dispatch immediately (purely opportunistic batching) |
| `speculative_decoding` | boolean | `false` | Enable speculative decoding |
| `speculative_gamma` | integer | `4` | Draft tokens per verification step |
| `draft_model_path` | path | none | Path to draft model |
| `max_split_model_memory_mb` | integer | none | Max GPU memory for split model cache |
| `tp_max_latency_ms` | integer | `10` | Max peer latency (ms) for tensor parallelism groups |
| `local_embedding_privacy` | boolean | `false` | Embed tokens locally before sending to first segment. Remote nodes never see raw token IDs |
| `encrypted_pipeline` | boolean | `false` | Force first+last segment to local node (boomerang topology). No remote sees plaintext. Adds ~1 RTT/token. Per-model override via API. Requires shard 0 + final shard locally |

## `[logging]` — Log Output

| Option | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"info"` | Log level: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"` |
| `format` | string | `"pretty"` | Log format: `"pretty"` or `"json"` |
| `file` | path | none | Write logs to file |

## `[ui]` — Web Interface

| Option | Type | Default | Description |
|---|---|---|---|
| `open_browser_on_start` | boolean | `true` | Open dashboard on launch |
| `theme` | string | `"dark"` | Color theme: `"dark"` or `"light"` |

## `[api]` — API Authentication

| Option | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | none | Bearer token. Empty = auto-generated |
| `rate_limit_rpm` | integer | `60` | Rate limit for `/v1/` endpoints (requests/min) |
| `rate_limit_admin_rpm` | integer | `200` | Rate limit for `/api/admin/` endpoints (requests/min) |

## `[model]` — Model Storage

| Option | Type | Default | Description |
|---|---|---|---|
| `shard_size_mb` | integer | `512` | Shard size in MB. Range: 64-2048 |

## `[auto_manage]` — Automatic Shard Management

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `true` | Auto-download popular shards (only for models at DemandVerified+ or Pinned trust level) |
| `max_storage_mb` | integer | `0` | Max disk for auto-downloads. `0` = 50% of max_disk_mb |
| `interval_minutes` | integer | `5` | Check interval for new shards |
| `max_shards` | integer | `0` | Max shards. `0` = unlimited |
| `max_concurrent_downloads` | integer | `3` | Max parallel downloads |
| `prune_enabled` | boolean | `true` | Auto-remove over-replicated shards |
| `min_replicas` | integer | `2` | Min network replicas before pruning |
| `prune_cooldown_secs` | integer | `300` | Seconds between prune actions per model |
| `max_holder_load_for_prune` | integer | `3` | Block pruning if holders are busy |

## `[pool]` — Device Pool

| Option | Type | Default | Description |
|---|---|---|---|
| `max_pool_size` | integer | `10` | Max devices in a pool |
| `invitation_ttl_hours` | integer | `24` | Invitation validity period |
| `rate_limit_per_hour` | integer | `10` | Max pool operations per hour |
| `gossip_interval_secs` | integer | `600` | Pool state gossip interval |
| `private_mode` | bool | `false` | Restrict inference to pool members only. Toggleable at runtime via API/UI |
| `private_mode_allow_lan` | bool | `true` | Also allow LAN peers (mDNS-discovered) when private mode is on |
| `offline_mode` | bool | `false` | Air-gapped: no bootstrap peers, no HF downloads, mDNS-only discovery |

## `[pool.credit_rates]` — Credit Rates

| Option | Type | Default | Description |
|---|---|---|---|
| `inference_serve` | integer | `10` | Credits earned per layer per token served |
| `inference_consume` | integer | `10` | Credits spent per layer per token consumed |
| `shard_hosting` | integer | `1` | Credits per GB per hour hosting |
| `shard_seeding` | integer | `5` | Credits per GB seeding |
| `relay_service` | integer | `2` | Credits per connection hour relaying |
| `penalty_serve_failure` | integer | `50` | Credits deducted per failure |

## `[updates]` — Auto-Update

| Option | Type | Default | Description |
|---|---|---|---|
| `auto_update` | string | `"stable"` | Policy: `"disabled"`, `"stable"`, `"all"` |
| `check_interval_hours` | integer | `6` | Update check frequency |

## `[identity]` — Your Identity

| Option | Type | Default | Description |
|---|---|---|---|
| `region` | string | none | Country code for network map (e.g., `"US"`) |

## `[providers.claude_subscription]` — Claude Subscription (feature-gated)

> Requires `--features claude-subscription` at build time. Managed via the dashboard or `PUT /api/admin/providers`.

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Route `claude-*` model requests through the local CLI |
| `claude_binary` | string | `"claude"` | Path to the `claude` binary |
| `default_model` | string | none | Override model for all requests |
| `max_concurrent` | integer | `3` | Maximum concurrent subprocess invocations |
| `timeout_secs` | integer | `300` | Per-request timeout in seconds |
| `working_dir` | string | *(temp dir)* | Working directory for the subprocess. Empty or `"none"` uses system temp dir (recommended for API proxy use). Set to a project path for context-aware responses. |
