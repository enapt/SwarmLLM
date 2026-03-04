# Security & Encryption

## Three Encryption Tiers

### Tier 1: Pairwise Sessions (Unicast)

For direct peer-to-peer communication:
- Ed25519 → X25519 → ECDH → ChaCha20-Poly1305
- Forward secrecy via ephemeral X25519 re-keying every 10 minutes
- Nonce reuse prevented by session clearing on disconnect (`remove_session()`)
- Replay protection: atomic `fetch_max` on receive nonce
- Nonce state updated only after successful decryption (prevents DoS)

### Tier 2: Pipeline Sealing (Inference)

For inference prompts and responses:
- Per-request ephemeral key
- Sealed prompt/response
- Wire tag: `TENSOR_TAG_ENCRYPTED = 0x10`

### Tier 3: Sealed Gossip (Broadcasts)

For GossipSub messages:
- Epoch-based group key + Ed25519 origin signature
- Verifies sender authenticity before processing
- 1-hour rotation cycle

## Identity

- Ed25519 keypair generated on first run, stored in `identity.key`
- Private key never leaves the machine
- Public key = Node ID (first 8 bytes hex for display)
- Nickname system: Ed25519-signed records with timestamp-wins conflict resolution

## Trust & Reputation

`TrustManager` tracks per-peer scores (0.0-1.0, default 0.5):

| Event | Score Change |
|---|---|
| InferenceSuccess | +0.01 |
| ValidTransaction | +0.02 |
| SpotCheckFail | -0.10 |
| InvalidGossip | -0.05 |
| SignatureViolation | -0.20 |

Scores decay toward 0.5 over time (1% per health cycle). Trust factors into pipeline scheduling and credit tier weighting.

## Sybil Resistance

- Subnet clustering detection: >5 nodes per /24 → elevated spot-check rate
- Signed-only balance reports
- Timestamp freshness checks on gossip (5 min window, rejects >5 min old)

## API Authentication

- Auto-generated 32-byte hex Bearer token
- Protected: `/v1/*`, config PUT, shutdown, HF downloads, API key endpoint
- Exempt: `/`, `/health`, `/admin` (read-only dashboard), static assets
- Request body limit: 2 MB
- Content-Security-Policy header on all responses
