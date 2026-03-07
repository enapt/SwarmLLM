# Networking & Discovery

## Transport Stack

```
libp2p Swarm
├── Kademlia (DHT) — distributed hash table for peer/shard/model lookup
├── GossipSub — pub/sub for shard announcements, governance, identity, pools
├── request_response — unified protocol (/swarmllm/1.0.0, 300s timeout)
├── mDNS — optional LAN peer discovery
├── connection_limits — max 2/peer, 500 total
├── Identify — protocol identification
├── AutoNAT — NAT detection
├── DCUtR — hole punching
└── relay::client — circuit relay
```

## Protocol Format

The unified protocol uses a type-tag byte:
- `0x00` — JSON control message (SwarmMessage, ShardRequest/Response)
- `0x01` — Binary tensor payload (LayerForward, LayerResult)

## Discovery Stack

SwarmLLM uses 5 independent discovery layers:

1. **mDNS** — Discovers LAN peers in seconds. Config: `enable_mdns = true`
2. **Persistent Peer Cache** — Saves up to 200 peers every 5 min + on shutdown. Fastest reconnect.
3. **Invite Codes** — Format: `swarm://<base64url(key‖nonce‖encrypted_multiaddr)>`. Encrypted with ChaCha20Poly1305.
4. **Peer Exchange (PEX)** — On each connection, exchanges up to 20 known peers.
5. **Kademlia DHT** — Bootstrap flag + periodic re-bootstrap every 60s.

## GossipSub Topics

| Topic | Content |
|---|---|
| `swarm/models/{model_id}` | ShardAnnounce, capacity |
| `swarm/governance` | ModelVote |
| `swarm/health` | Trust summaries |
| `swarm/identity` | NicknameRecord (signed) |
| `swarm/pools` | PoolState, PoolInvitation |

Messages older than 5 minutes are rejected (replay protection).

## Anti-Gaming

- Subnet clustering detection: >5 nodes per /24 triggers 25% spot-check rate (up from 5%)
- SubnetClustering trust penalty (-0.03 per cycle)
- Signed balance reports with timestamp freshness (5 min window)
