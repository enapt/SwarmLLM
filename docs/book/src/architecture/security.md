# Security & Encryption

## Three Encryption Tiers

### Tier 1: Pairwise Sessions (Unicast)

For direct peer-to-peer communication:
- Ed25519 → X25519 → ECDH → ChaCha20-Poly1305
- Forward secrecy via ephemeral X25519 re-keying every 10 minutes
- Nonce reuse prevented by session clearing on disconnect (`remove_session()`)
- Replay protection: RFC 6479 sliding window (128-bit bitmap) — allows packet reordering within window while rejecting duplicates
- Nonce state updated only after successful decryption (prevents DoS)
- Pending ephemeral keys expire after 60 seconds (prevents memory exhaustion from unanswered re-keys)

### Tier 2: Pipeline Sealing (Inference)

For inference prompts and responses:
- Per-request ephemeral key
- Sealed prompt/response
- Wire tag: `TENSOR_TAG_ENCRYPTED = 0x10`

> Pipeline sealing is active: the final segment encrypts output token IDs for the requester's X25519 public key. Intermediate nodes process activation tensors (protected by Tier 1 in transit) but never see the final plaintext output. See [Known Limitations](#known-limitations) for activation inference risks.

### Tier 3: Sealed Gossip (Broadcasts)

For GossipSub messages:
- Epoch-based group key + **mandatory** Ed25519 origin signature
- All gossip messages MUST be `seal_signed()` — unsigned messages are rejected
- Verifies sender authenticity before processing
- 1-hour rotation cycle

## Transport-Authenticated Dispatch

All inbound network messages carry transport-authenticated sender identity:

- libp2p Noise protocol authenticates peers at the transport layer
- `AuthenticatedMessage` wrapper carries the verified `NodeId` of the sender
- MessageDispatcher validates sender identity against message claims:
  - ShardAnnounce: sender must match `announce.node_id`
  - CreditTransaction: sender must be a party (from or to)
  - CreditGossip, NicknameGossip: sender must match claimed `node_id`
  - HealthPing/Pong: sender must match claimed `node_id`
  - EphemeralKeyExchange: sender must match `exchange.node_id`
- Mismatched messages are logged and dropped

## Signed DHT Records

Kademlia DHT records are Ed25519-signed to prevent poisoning:
- Format: `[32B pubkey][64B signature][payload]`
- `announce_capability()` and `announce_shards()` sign records with node identity
- Consumers verify signatures with `verify_dht_value()` before trusting records
- Records expire after 1 hour with automatic re-publication

## Identity

- Ed25519 keypair generated on first run, stored in `identity.key`
- Private key never leaves the machine
- Public key = Node ID (first 8 bytes hex for display)
- Nickname system: Ed25519-signed records with timestamp-wins conflict resolution
- Nickname registry capped at 10,000 entries (requires peer_registry membership)

## Trust & Reputation

`TrustManager` tracks per-peer scores (0.0-1.0, default 0.5):

| Event | Score Change |
|---|---|
| InferenceSuccess | +0.01 |
| ValidTransaction | +0.02 |
| SpotCheckFail | -0.10 |
| InvalidGossip | -0.05 |
| SignatureViolation | -0.20 |

Scores decay toward 0.5 over time (1% per health cycle, default 30 seconds). Trust factors into pipeline scheduling and credit tier weighting.

## Sybil Resistance

- Subnet clustering detection: >5 nodes per /24 → elevated spot-check rate
- Signed-only balance reports
- Timestamp freshness checks on gossip (5 min window, rejects >5 min old)

## API Authentication

- Auto-generated 32-byte hex Bearer token (constant-time comparison)
- Protected: `/v1/*`, `/api/admin/provider-models`, config PUT, shutdown, HF downloads, API key endpoint
- Exempt: `/`, `/health`, `/admin` (read-only dashboard), static assets
- Request body limit: 2 MB
- Content-Security-Policy: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Referrer-Policy: no-referrer
- WebSocket Origin validation (rejects cross-site WebSocket hijacking)

## Input Validation

- Model field length: max 256 chars in OpenAI + Anthropic handlers
- Tools array: max 128 entries
- Stop sequences: max 16 entries
- HuggingFace repo_id: validated `owner/repo` format (alphanumeric, hyphens, dots, underscores, max 96 chars)
- HuggingFace filename: must end in `.gguf`, no `..`, no URL metacharacters
- Path traversal: `sanitize_path_component()` on all network-provided model IDs before filesystem operations
- Update URLs: only GitHub download URLs accepted
- Update binaries: SHA256 checksum verification mandatory

## Rate Limiting & DoS Protection

- Per-IP rate limiter with periodic cleanup (5 min intervals)
- Inference queue depth cap: 512 requests
- HTTP timeout: 5 minutes (Slowloris protection via tower-http TimeoutLayer)
- Credit transaction signature verification before ledger apply

## Known Limitations

These are architectural properties that cannot be fully mitigated with code changes:

- **Gossip epoch key is publicly derivable** — derived from "swarmllm-mainnet-v1". Gossip encryption is defense-in-depth; Ed25519 signing is the primary security mechanism.
- **Prompt inference via intermediate activations** — peers hosting pipeline segments can theoretically reconstruct input from embeddings. Mitigation: first-segment-local scheduling preference, pipeline sealing encrypts output tokens for the requester's X25519 key.
- **Byzantine tensor manipulation** — malicious peers can send garbage activations. Mitigation: probabilistic spot-check validation (5% rate, 25% for subnet-clustered peers) with trust score reduction on failure.
- **Sybil credit farming** — Ed25519 keys are free. Anti-gaming heuristics help but are not bulletproof.
- **GGUF parser vulnerabilities** — llama.cpp CVEs. BLAKE3 content hash gates shard loading but parser bugs remain upstream.
- **Kademlia eclipse attacks** — strategic Sybil node IDs can control DHT routing. K-bucket eviction policies help.
