# Identity & Device Pool API

## Identity

### GET /api/identity/nickname
Get the current node's nickname.

### PUT /api/identity/nickname
Set a nickname. Body: `{"nickname": "my-node"}`

### DELETE /api/identity/nickname
Clear the nickname.

### GET /api/identity/leaderboard
Network-wide credit leaderboard.

### GET /api/identity/peers
Peer identity directory (nicknames, regions, tiers).

## Device Pools ("My Devices")

Link multiple devices owned by the same user. Credits earned by all linked devices are combined into one balance on the main (owner) device.

> **Terminology**: "Linked Devices" in the UI. This is different from connecting to the SwarmLLM network — linking devices groups your own hardware, while the network connects you with other people.

### Quick Start (CLI)

```bash
# On your main device:
swarmllm pool create --name "My Devices"
swarmllm pool invite-code
# → A3F7K2M9

# On each other device:
swarmllm pool join A3F7K2M9

# Check status:
swarmllm pool status
```

### Invite Code System

Instead of exchanging raw 64-character node IDs, device pools use **8-character invite codes** (e.g., `A3F7K2M9`):

1. Owner generates a code → `POST /api/pool/generate-code`
2. Code shared verbally, via QR, or copy-paste
3. Member enters code → `POST /api/pool/join` → broadcasts join request over gossip
4. Owner's node auto-validates code and creates invitation
5. Member auto-accepts → pool established

**Security**: Codes use a 32-character alphabet (no 0/O/1/I), are one-time use, expire in 24h, and the code itself is never transmitted over the network — only its BLAKE3 hash.

### API Endpoints

#### GET /api/pool/state
Current pool membership state. Returns `in_pool`, member list with device names, online status, per-device stats, credit split percentage.

#### POST /api/pool/create
Create a new device pool. Body: `{"name": "My Devices"}`

#### POST /api/pool/generate-code
Generate an invite code (owner only). Returns: `{"code": "A3F7K2M9"}`. Max 5 active codes.

#### POST /api/pool/join
Join a pool using an invite code. Body: `{"code": "A3F7K2M9"}`

#### POST /api/pool/invite
Invite a specific node by ID (advanced). Body: `{"node_id": "abc123..."}`

#### POST /api/pool/accept
Accept a pool invitation. Body: `{"invitation_id": "..."}`

#### POST /api/pool/remove
Remove a member (owner only). Body: `{"node_id": "..."}`

#### POST /api/pool/leave
Leave the current pool.

#### POST /api/pool/device-name
Set this device's nickname. Body: `{"name": "Gaming PC"}`

#### PUT /api/pool/credit-split
Set credit split percentage (owner only). Body: `{"pct": 20}` (0-50)

#### PUT /api/pool/contribution
Set per-member contribution level override. Body: `{"node_id": "...", "level": 75}` (integer 0–100)

#### GET /api/pool/invitations
List pending invitations for this node.

#### GET /api/pool/leaderboard
Pool member contribution rankings.

#### GET/PUT /api/admin/pools/:id/rates
Per-pool credit rate overrides.

## Private Mode

Restrict inference to your device pool for maximum privacy. Your prompts never leave your devices.

### GET /api/pool/private-mode
Current state + coverage summary. Returns `enabled`, `allow_lan`, `offline_mode`, and `coverage` object.

### PUT /api/pool/private-mode
Toggle private mode. Body: `{"enabled": true}` or `{"enabled": true, "offline_mode": true}`.
Returns coverage summary so the UI can show trade-offs immediately.

### GET /api/pool/coverage
Per-model coverage breakdown: `total_shards`, `pool_shards`, `coverage_pct`, `missing` indices, `est_download_mb`. Also returns `disk_budget_mb` and `disk_used_mb`.

### Shard Pinning

### GET /api/pool/pins
List current shard pins.

### POST /api/pool/pin
Pin a model to a specific device (owner only). Body: `{"model_id": "...", "target_node_id": "hex..."}`.
Optional `shard_indices` array for specific shards (empty = all shards).

### DELETE /api/pool/pin
Remove a shard pin. Same body format as POST.

## Pool Features

- **Device nicknames**: Name each device for easy identification
- **Online/offline status**: Tracked via health pings, displayed with last-seen timestamps
- **Per-device stats**: VRAM, shards hosted, forwards served, uptime, models hosted
- **Combined VRAM**: Aggregate GPU memory across all linked devices
- **Credit split**: Owner configures what percentage (0-50%) members keep vs forward
- **Private Mode**: Restrict inference to pool devices only. Toggle via UI or API
- **Shard Pinning**: Assign specific models to specific devices. Auto-manage respects pins
- **Offline Mode**: Air-gapped LAN operation with mDNS-only discovery
- **Coverage Dashboard**: Per-model availability bars showing pool shard coverage
- **Max 10 devices** per pool (configurable), 10 pool operations per hour rate limit

## Pool Security

- Invite codes: 32^8 ≈ 1.1 trillion combos, one-time use, 24h expiry
- Join requests signed with Ed25519 (transport-layer sender authentication)
- Credit forwarding uses dual-signed `PoolCreditForward` (member + owner)
- Member removal requires Ed25519-signed removal notice with replay protection
- Pool state gossip verifies each member's acceptance signature
- Blinded invitation broadcast (SEC-M18): network observers can't see who's invited
