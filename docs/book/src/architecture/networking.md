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
3. **Invite Codes** — two formats, see below.
4. **Peer Exchange (PEX)** — On each connection, exchanges up to 20 known peers.
5. **Kademlia DHT** — Bootstrap flag + periodic re-bootstrap every 60s.

### Invite code formats

Two codes exist for two different jobs. Both are ChaCha20-Poly1305 sealed with
a randomly generated key embedded in the blob itself — that is deliberately
*not* confidentiality against someone holding the code, only a guard against
casual harvesting of node addresses from screenshots and chat logs.

**`swarm://...` — network invite.** `base64url(key ‖ nonce ‖ encrypted_multiaddr)`.
Carries one reachable address for the node that minted it. Used to bring a
machine onto the swarm.

**`swarmpool://...` — pool invite (v2, R140).** Wraps the 8-character pool code
with everything a fresh node needs to find the inviter *before* any shared
discovery exists. Inner payload, JSON-serialised then sealed then base64url'd:

```json
{
  "version": 2,
  "pool_id": "...",
  "pool_name": "...",
  "multiaddrs": ["/ip4/…/tcp/8810/p2p/12D3KooW…"],
  "code": "A3F7K2M9",
  "expires_at_unix": 1750000000
}
```

Roughly 300–500 characters — long, but it fits a copy-paste, which the 8-char
code could not do for this purpose. `multiaddrs` is the node's live
`listen_multiaddrs` snapshot: bound sockets **unioned with** confirmed external
addresses (UPnP-mapped, AutoNAT-confirmed, relay-circuit, or manually declared
via `network.external_addresses`), each suffixed with `/p2p/<peer_id>` so the
dialer can verify identity. Without that union a NAT'd node silently minted a
LAN-only code that worked on the LAN and died over the internet.

The legacy bare 8-character code (`A3F7K2M9`) still works and still means
"broadcast a join request over the existing swarm". It only ever worked when
both nodes were already on the same swarm — which is exactly the situation an
invite code is least needed for. `pool::invite::looks_like_v2` is the prefix
sniff that routes between the two paths; `decode_invite_code` normalises every
decode failure to `Validation`, because the overwhelmingly likely cause is a
truncated paste rather than a daemon bug.

Generation fails with `ServiceUnavailable` when `listen_multiaddrs` is empty
(the daemon hasn't bound yet). When it has entries but none pass the stricter
`any_internet_reachable` check — public IP, DNS name, or relay circuit; LAN and
CGNAT ranges excluded — generation still succeeds but emits an
`invite_lan_only` warning, so nobody is handed a code that cannot survive the
trip it was made for.

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
