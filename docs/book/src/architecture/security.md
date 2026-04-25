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

> Pipeline sealing is active: the final segment encrypts output token IDs with the requester's X25519 public key. The final-segment node can see the sampled tokens before encryption — this is inherent to the architecture since sampling happens on that node. Intermediate nodes process activation tensors (protected by Tier 1 in transit) but never see the final plaintext output. See [Pipeline Privacy Model](#pipeline-privacy-model) for a full breakdown of what each node can see.

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
- `start_providing_shards()` signs records with node identity
- **Active verification**: `verify_dht_value()` is called on all `GetRecordOk` results in NetworkManager — records with invalid or missing signatures are logged and discarded
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
- Request body limit: 32 MB (raised from 2 MB to support VLM image payloads)
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

## Pipeline Privacy Model

Distributed inference splits a model across multiple nodes. This creates inherent privacy trade-offs — each node in the pipeline must process data to do its job. This section documents exactly what each node can see.

### What each node sees during inference

Consider a 3-node pipeline: **Requester** → **Node A** (layers 0-10) → **Node B** (layers 11-21) → **Node C** (layers 22-27, final):

| Data | Requester | Node A (first) | Node B (middle) | Node C (last) |
|---|---|---|---|---|
| **Plaintext prompt** | Yes (author) | See below* | No | No |
| **Raw token IDs** | Yes | See below* | No | No |
| **Input activations** | — | Yes | Yes | Yes |
| **Output activations** | — | Yes | Yes | — |
| **Generated token IDs** | Yes (decrypted) | No | No | Yes (samples them) |
| **Final plaintext response** | Yes (decrypted) | No | No | Yes (before sealing) |

*\*Node A's visibility depends on the `local_embedding_privacy` setting — see below.*

### Risk: First-segment node sees raw tokens (default)

**Without `local_embedding_privacy`** (default): The first-segment node (Node A) receives the raw prompt text or token IDs to perform the embedding lookup. This means Node A can read the user's prompt in plaintext.

**With `local_embedding_privacy: true`**: The requesting node performs the embedding lookup locally and sends pre-embedded activation tensors. Node A receives floating-point vectors instead of token IDs. This is a significant privacy improvement, but not absolute — see [Activation Inversion Risk](#activation-inversion-risk) below.

### Risk: Final-segment node sees generated output

The final-segment node (Node C) **must** sample tokens from the logit distribution. This is fundamental — sampling is the act of choosing the next word, and it can only happen where the final layer's output logits exist. Node C therefore sees every generated token before encrypting them via Tier 2 pipeline sealing.

**This cannot be mitigated architecturally.** The node that runs the last transformer layer and samples tokens will always know what tokens were sampled. Pipeline sealing ensures the tokens are encrypted before being sent back over the network, so intermediate nodes and eavesdroppers cannot read the response — but the final-segment node itself can.

### Risk: Activation inversion attacks

All intermediate nodes see hidden-state activation tensors (floating-point matrices). Research has shown that activations from early transformer layers can sometimes be partially inverted to recover input tokens, especially:

- **Embedding-layer activations** (layer 0 output) — most vulnerable, essentially a lookup table that can be reversed
- **Early layers (1-4)** — progressively harder to invert as information mixes across token positions
- **Deep layers (5+)** — extremely difficult to invert in practice; activations encode abstract features, not token identity

**Mitigations in SwarmLLM:**
1. `local_embedding_privacy: true` — the requesting node performs embedding locally, so the first segment never receives the trivially-invertible embedding output. It receives post-layer-0 activations at earliest.
2. Tier 1 encryption — all inter-node tensor transfers are encrypted with ChaCha20-Poly1305, preventing network-level eavesdropping
3. Pipeline scheduling preference — the scheduler prefers local segments for the first layers when possible

### Risk: Byzantine tensor manipulation

A malicious node can send garbage activations instead of computing the actual transformer layers. This produces incorrect output without detection unless spot-checked. Mitigations: probabilistic spot-check validation (5% rate, 25% for subnet-clustered peers) with trust score reduction on failure.

### Summary of privacy guarantees

| Configuration | Prompt privacy | Response privacy | Activation risk |
|---|---|---|---|
| Default (no privacy flags) | First segment sees plaintext | Final segment sees plaintext | Intermediate nodes see activations |
| `local_embedding_privacy: true` | No remote node sees raw tokens | Final segment sees plaintext | Reduced — no trivial embedding inversion |
| `encrypted_pipeline: true` | No remote node sees raw tokens | No remote node sees output | Only intermediate activations visible to remote nodes |
| + Tier 2 pipeline sealing | No remote node sees raw tokens | Encrypted on the wire | Reduced — no trivial embedding inversion |
| All protections enabled | Best available | Best available | Remote nodes only see intermediate activations; inversion theoretically possible but computationally expensive |

> **Bottom line:** With `encrypted_pipeline`, no remote node sees plaintext input or output — the pipeline "boomerangs" through remote nodes and returns to the requester. This is the strongest privacy mode. Without it, `local_embedding_privacy` still protects raw token IDs but the final-segment node sees generated output.

## Local Embedding Privacy

When `local_embedding_privacy: true` is set in `[inference]` config, the requesting node performs token→embedding lookup locally before sending activations to the first pipeline segment. Remote nodes never see raw token IDs — only hidden-state activation tensors.

**How it works:**
1. On startup, `LocalEmbedder` loads `token_embd.weight` from `shard_000.bin` (~64MB for a 7B Q4 model)
2. The requesting node tokenizes the prompt and performs the embedding lookup locally (~1ms)
3. The resulting hidden-state tensor (`[1, seq_len, hidden_dim]`, FP32) is sent as `LayerForward.activations` with `pre_embedded: true`
4. The receiving first-segment node skips its embedding lookup and processes the pre-embedded activations directly

**Wire format:** The `pre_embedded` flag on `LayerForward` is `#[serde(default)]`, so old nodes receiving new-format messages default to `false` (backward compatible).

**Trade-off:** Pre-embedded activations are larger than raw text (e.g., 512 tokens × 4096 hidden × 4 bytes = 8MB vs ~2KB text). This matches the existing inter-segment activation sizes, so it does not change the bandwidth profile of distributed inference.

Relevant code: `src/inference/local_embedder.rs`, `src/inference/pipeline/`, `src/daemon/state/mod.rs` (`local_embedders` DashMap).

## Encrypted Pipeline

When `encrypted_pipeline: true` is enabled (globally or per-model), the pipeline scheduler forces the requesting node to handle both the **first** and **last** segments. This creates a "boomerang" topology:

```
Requester (shard 0, embed) → Remote A (middle shards) → ... → Requester (final shard, decode)
```

**No remote node ever sees plaintext** — neither the raw prompt tokens nor the generated output. Remote nodes only process intermediate hidden-state activations.

**Requirements:**
- The requesting node must hold **shard 0** (embedding table) AND the **final shard** (output head)
- `local_embedding_privacy` is auto-enabled when encrypted pipeline is active
- Only useful for models with **3+ shards** (2-shard models = fully local, no distribution)

**Overhead:**
- Adds ~1 extra network RTT per generated token (activations must return to the requester for final decoding)
- Latency increase depends on distance to the furthest remote segment
- No bandwidth overhead vs normal distributed inference (activation sizes are the same)

**Per-model configuration:**
- API: `GET/PUT /api/admin/models/{id}/encrypted-pipeline`
- Dashboard: gear icon on model card → "Encrypted pipeline" checkbox
- Global fallback: `encrypted_pipeline = true` in `[inference]` config
- Per-model overrides are persisted to the database

Relevant code: `src/inference/scheduler/mod.rs` (greedy_assign), `src/inference/pipeline/` (auto-enable local embedding), `src/api/admin_models/mod.rs` (API endpoints), `src/daemon/state/mod.rs` (`encrypted_pipeline_models` DashMap).

## Known Limitations

These are architectural properties that cannot be fully mitigated with code changes:

- **Gossip epoch key is publicly derivable** — derived from "swarmllm-mainnet-v1". Gossip encryption is defense-in-depth; Ed25519 signing is the primary security mechanism.
- **Final-segment output visibility** — the node running the last transformer layers sees all generated tokens before pipeline sealing encrypts them. This is inherent to the architecture (see [Pipeline Privacy Model](#pipeline-privacy-model)).
- **Activation inversion** — hidden-state tensors passed between nodes can theoretically be inverted to recover input, especially from early layers. `local_embedding_privacy` eliminates the trivial case (embedding lookup reversal). Deep-layer inversion remains an open research problem.
- **Byzantine tensor manipulation** — malicious peers can send garbage activations. Mitigation: probabilistic spot-check validation (5% rate, 25% for subnet-clustered peers) with trust score reduction on failure.
- **Sybil credit farming** — Ed25519 keys are free. Anti-gaming heuristics help but are not bulletproof.
- **GGUF parser vulnerabilities** — llama.cpp CVEs. BLAKE3 content hash gates shard loading but parser bugs remain upstream.
- **Kademlia eclipse attacks** — strategic Sybil node IDs can control DHT routing. K-bucket eviction policies help.
