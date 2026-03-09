# Storage & Data

## Data Directory Layout

```
~/.local/share/swarmllm/
├── config.toml          # User configuration
├── identity.key         # Ed25519 keypair
├── api_key              # Bearer token (auto-generated)
├── db.redb              # redb database (migrated from sled db/ directory)
└── models/
    ├── qwen2.5-coder-7b/
    │   ├── manifest.json
    │   ├── gguf_header.bin
    │   ├── shard_000.bin
    │   └── shard_001.bin
    └── tinyllama-1.1b/
        └── ...
```

## Database Tables (redb)

| Table | Key | Value |
|---|---|---|
| config | `"config"` | Config |
| config | `"api_key"` | Bearer token string |
| identity | `"keypair"` | Encrypted Ed25519 key |
| credits | `"balance"` | CreditBalance |
| credit_txns | `{uuid}` | CreditTransaction |
| peer_trust | `{node_id_hex}` | TrustScore |
| peer_cache | `{multiaddr}` | () presence key |
| shard_meta | `{model_id}/{index}` | ShardInfo + path |
| model_meta | `{model_id}` | ModelManifest |
| sessions | `{session_id}` | KV-cache metadata |
| nicknames | `{node_id_hex}` | NicknameRecord |
| pool_state | `"pool"` | PoolState |
| trust_scores | `{node_id_hex}` | f64 trust score |
| escrow | `{escrow_id}` | EscrowEntry |
| hf_sources | `{model_id}` | HfSource metadata |
| locked_shards | `{shard_id_json}` | bool |
| resource_schedule | `"current"` | ResourceSchedule |
| model_trust | `{model_id}` | ModelTrustEntry (level, request count, last seen) |

## Model Acquisition Pipeline

```
Network Registry (GossipSub/DHT)
        │
        ▼
  Manifest Check ──► Reject if BLAKE3 mismatch
        │
        ▼
  Shard Selection ──► Rarest-first (BitTorrent-style)
        │
        ▼
  Download Loop ──► Atomic write to .tmp, rename to .bin
        │
        ▼
  Shard Verify ──► BLAKE3 vs manifest hash
        │
        ▼
  Model Ready
```

**Integrity guarantees:**
- Manifests verified via BLAKE3 self-hash
- Each shard verified against manifest hash
- Failed shards renamed `.bin.quarantine`, serving peer penalized
- Downloads retried (3 attempts, exponential backoff)
- Atomic writes prevent corrupt partial files
- Stale `.tmp` files cleaned on startup
