# Storage & Data

## Data Directory Layout

```
~/.local/share/swarmllm/
├── config.toml          # User configuration
├── identity.key         # Ed25519 secret key: 32 raw bytes, owner-only (mode 0600)
├── api_key              # Bearer token (auto-generated)
├── db.redb              # redb database (migrated from sled db/ directory)
├── canonical/           # the swarm's copy's header and side files, staged while
│   └── <model>/         # this computer checks its own copy against it
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

Every tree below lives in ONE redb table named `data`, keyed `"{tree}\0{key}"`.
The node's identity key is not in the database: it is the `identity.key` file
above. Which parts a node holds is read from the `models/` directory, not stored.

| Tree | Key | Value |
|---|---|---|
| config | `"config"` | Config |
| config | `"api_key"` | Bearer token string |
| credits | `"balance"` | CreditBalance |
| credit_txns | `{uuid}` | CreditTransaction |
| peer_cache | `{multiaddr}` | () presence key |
| model_meta | `{model_id}` | ModelManifest |
| kv_sessions | `{session_id}` | KV-cache metadata |
| nicknames | `{node_id_hex}` | NicknameRecord |
| identity_prefs | — | this node's own identity preferences |
| pool_state | `"my_pool"` | PoolState |
| pool_invitations, pool_invite_codes, pool_forwards, pool_removal_replays | — | device-pool bookkeeping; invite codes are keyed by their hash, so a pending join survives the owner restarting |
| node_modes | `"private_mode"`, `"offline_mode"` | bool |
| trust_scores | `{node_id_hex}` | f32 trust score |
| escrow | `{escrow_id}` | EscrowEntry |
| hf_sources | `{model_id}` | HfSource — the upload this node fetches the model from; since v0.3.221 always the canonical one once verified |
| canonical_builds | `{model_id}` | CanonicalBuild — the upload the whole swarm uses (size, part sizes and layers, first-tensor offsets, header BLAKE3) |
| origin_verified_hashes | `{shard_id_json}` | BLAKE3 of a part fetched from the origin itself; forgotten when the node switches upload |
| locked_shards | `{shard_id_json}` | bool |
| removed_shards | `{shard_id_json}` | bool — the user deleted this shard; auto-manage leaves it alone until it is asked for again |
| resource_schedule | `"current"` | ResourceSchedule |
| model_trust | `{model_id}` | ModelTrustEntry (level, request count, last seen) |
| responses | `{response_id}` | stored `/v1/responses` records (30-day TTL) |
| network | — | whether a remote computer has ever dialled this node in, kept across restarts |
| update | — | update-check bookkeeping |

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

## One upload per model (v0.3.221)

A model's id comes from its file name, and several people publish their own copy
of the same model and quantisation on HuggingFace — byte-different files under one
id. Parts of two copies cannot be combined in a split, so **every node uses the same
upload of each model**:

- **One ranking everywhere.** A pinned reference model first, then the publisher's
  place in the trusted list (the model's own author, then well-known curators, in a
  fixed order), then anyone else, ties by name. Every node uses the best upload anyone
  has claimed — after checking it on HuggingFace *without* a login token, so every
  node sees the same answer.
- **Nothing is fetched until the choice is made**, and a peer's manifest of another
  upload is ignored.
- **Self-healing.** A node holding another upload fetches the chosen upload's parts
  for the layers it holds (by byte range — never the whole file) into `canonical/`,
  keeps serving its old parts meanwhile, and swaps when every part is in and the
  model is idle. A node whose parts are right but whose header or manifest describe
  another upload gets the right ones and reloads.
- `GET /api/admin/models` reports it per model in `shared_copy`;
  `SWARMLLM_CANONICAL_UPLOADS=0` turns the mechanism off (test rigs).

Details and evidence: `docs/invariants/network.md` § "One upload per model id".

