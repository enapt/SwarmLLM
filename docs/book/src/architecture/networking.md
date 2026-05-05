# Networking & Discovery

## Transport Stack

```
libp2p Swarm
├── Kademlia (DHT) — distributed hash table for peer/shard/model lookup
├── GossipSub — pub/sub for shard/health/credits/identity/pools/regions
├── request_response — unified protocol (/swarmllm/1.0.0, 600s timeout)
├── mDNS — optional LAN peer discovery
├── connection_limits — max 1/peer (>1 causes rr round-robin to dead connections), 500 total
├── Identify — protocol identification
├── AutoNAT — NAT detection
├── DCUtR — hole punching
└── relay::client — circuit relay
```

## Protocol Format

The unified protocol uses a type-tag byte on every frame
(`src/network/protocol/mod.rs`):

| Tag | Constant | Use |
|---|---|---|
| `0x00` | `WIRE_TAG_JSON` | JSON control message (`SwarmMessage`, `ShardRequest`/`ShardResponse`) |
| `0x01` | `WIRE_TAG_TENSOR` | Binary tensor payload (`LayerForward`, `LayerResult`), f16 |
| `0x02` | `WIRE_TAG_TENSOR_COMPRESSED` | Q8_0 activation frame (flag-gated `activation_compression`) — ~3.76× smaller than `0x01` |
| `0x03` | `WIRE_TAG_SHARD` | Raw shard bytes (ShardResponse payload, 32 MB max — bypasses the 4 MB JSON cap) |
| `0x04` | `WIRE_TAG_PREFIX_KV` | Cross-node prefix-KV snapshot. Frame body's flag byte: `0` = miss, `1` = raw f32, `2` = zstd-compressed f32 (gated on `NetworkConfig::prefix_kv_compression`, default off). Receivers always decompress regardless of the send-side flag. |

Receivers auto-dispatch on the leading byte; senders choose based on
config + request kind. Only the `0x00` frame carries a JSON body; the
rest use binary framing with length prefixes.

## Discovery Stack

SwarmLLM uses 5 independent discovery layers:

1. **mDNS** — Discovers LAN peers in seconds. Config: `enable_mdns = true`
2. **Persistent Peer Cache** — Saves up to 200 peers every 5 min + on shutdown. Fastest reconnect.
3. **Invite Codes** — Format: `swarm://<base64url(key‖nonce‖encrypted_multiaddr)>`. Encrypted with ChaCha20Poly1305.
4. **Peer Exchange (PEX)** — On each connection, exchanges up to 20 known peers.
5. **Kademlia DHT** — Bootstrap flag + periodic re-bootstrap every 60s.

## GossipSub Topics

Six topics, all subscribed at startup in `discovery::subscribe_topics`:

| Topic | Constant | Content |
|---|---|---|
| `swarm/models` | `TOPIC_MODELS` | `ShardAnnounce`, `ModelManifest`, `PrefixCacheAnnounce` (cross-node prefix-KV index) |
| `swarm/health` | `TOPIC_HEALTH` | `HealthPing`, `NodeCapability` (includes observed per-layer latencies for the Parallax scheduler), `TpAllReduceResponse` |
| `swarm/credits` | `TOPIC_CREDITS` | `CreditGossip`, `CreditTransaction` |
| `swarm/identity` | `TOPIC_IDENTITY` | `NicknameGossip` (signed) |
| `swarm/pools` | `TOPIC_POOLS` | `PoolMessage` (PoolState, PoolInvitation, CreditForward) |
| `swarm/regions` | `TOPIC_REGIONS` | `RegionShardSummary` (per-region shard availability for routing locality) |

The topic match in `NetworkManager::handle_broadcast` is
contract-not-default: a `SwarmMessage` variant with no topic arm falls
through `_ => return` and silently drops at the wire. Adding a new
gossip variant requires updating the match — an early multi-node test
caught `PrefixCacheAnnounce` missing from the `TOPIC_MODELS` arm, which
had silently dropped every cross-node prefix-cache announce at the
network layer until a two-daemon run flushed it out.

Messages older than 5 minutes are rejected (replay protection).

## Cross-Node Prefix KV Sharing Dispatch

The cross-node prefix-cache fetch path uses the `request_response`
protocol, not gossip. The gossip layer only broadcasts which blocks
each peer holds (`PrefixCacheAnnounce` on `swarm/models`); the actual
snapshot transfer is a direct bilateral exchange:

1. Requesting daemon sends `SwarmRequest::PrefixKvFetch` to the peer
   chosen by the probe resolver (trust-gated by
   `cross_node_prefix_trust_min`, default 0.5)
2. Serving daemon runs `fetch_local_snapshot` against its own worker
   over IPC (2000 ms timeout) and gets the serialized bytes or `None`
3. Serving daemon returns `SwarmResponse::PrefixKvData { present, payload }`
   with the bytes wrapped in the `WIRE_TAG_PREFIX_KV` frame on the
   binary payload slot (not in the JSON header — `serde_json` inflates
   `Vec<u8>` ~5× and blows past the 64 MiB IPC cap)
4. Requesting daemon BLAKE3-reverifies + NaN/Inf-scans, hands bytes to
   its worker to hydrate a `KvCacheEntry`

See [Inference > Prefix-Cache KV Sharing](./inference.md#prefix-cache-kv-sharing-cross-node)
for the full pipeline and measured numbers.

## Anti-Gaming

- Subnet clustering detection: >5 nodes per /24 triggers 25% spot-check rate (up from 5%)
- `SubnetClustering` trust penalty (-0.03 per cycle)
- Signed balance reports with timestamp freshness (5 min window)
- Gossip replay rejection (5 min window)
- `cross_node_prefix_trust_min` gates fetch peers at a minimum trust
  score (default 0.5, equal to `DEFAULT_TRUST`; set to 2.0 to disable
  cross-node fetch entirely)
