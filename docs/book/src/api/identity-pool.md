# Identity & Pool API

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

## Device Pools

Pools let multiple devices share credit earnings and aggregate resources.

### GET /api/pool/state
Current pool membership state.

### POST /api/pool/create
Create a new device pool. Body: `{"name": "my-pool"}`

### POST /api/pool/invite
Invite a node to the pool. Body: `{"node_id": "abc123..."}`

### POST /api/pool/accept
Accept a pool invitation. Body: `{"invitation_id": "..."}`

### POST /api/pool/remove
Remove a member from the pool. Body: `{"node_id": "..."}`

### POST /api/pool/leave
Leave the current pool.

### GET /api/pool/invitations
List pending invitations.

### GET /api/pool/leaderboard
Pool member contribution rankings.

## Pool Security

- Invitations require dual signatures (owner creates, member accepts)
- Pool state gossip verifies each member's acceptance signature
- Member removal requires Ed25519-signed leave notice
- Credit forwarding uses dual-signed `PoolCreditForward` records
