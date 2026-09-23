# Security & Encryption

## Two Operator-Facing Encryption Layers

SwarmLLM ships with two distinct encryption layers that solve different
problems. The first is on by default; the second is an opt-in
stronger privacy mode, which also switches on by itself for any model
whose first and last shard this node holds
(`inference.encrypted_pipeline_auto`, default on).

| Layer | Config flag | Default | What it protects |
|-------|-------------|---------|------------------|
| **Layer 1 — pairwise session encryption** | `network.enable_encryption` | **`true`** (on) | Every inter-node tensor forward is ChaCha20-Poly1305 sealed on the wire. Eavesdroppers on the network see ciphertext only. Entry/exit nodes still see plaintext at their boundary (the first segment receives the prompt itself — text or token IDs — unless `local_embedding_privacy` is on; the last segment sees the sampled tokens). |
| **Layer 2 — encrypted (boomerang) pipeline** | `inference.encrypted_pipeline` | `false` (opt-in, per-model override available), but `inference.encrypted_pipeline_auto` (default **`true`**) turns it on automatically for any model whose first and last shard this node holds. A per-model setting wins over both. | Forces the requesting node to handle BOTH the first segment (embedding) AND the last segment (sampling), so no remote node receives the prompt text or samples the reply. Remote nodes still compute on the intermediate hidden states **in plaintext** (they are sealed only in transit), and those are partially invertible back to text. Requires shard 0 + final shard locally. Adds ~1 RTT/token. |

Layer 1 is "encryption in transit." Layer 2 keeps the two ENDS of the
pipeline on your machine — a **structural** guarantee, not a cryptographic
one: the nodes doing the work in the middle can still recover much of your
text from what they compute on (see [Activation
inversion](#risk-activation-inversion-attacks)). **For anything sensitive,
use Private Mode or run the model entirely on your own machine.**
Disabling Layer 1 is only sensible for local-loopback debugging — there
is NO plaintext fallback on `seal()` failure (forwards are dropped).

R139 hardened Layer 1 by offloading the ChaCha20-Poly1305 seal/open
operations from the NetworkManager event loop via `tokio::spawn`, so
encryption no longer adds jitter to libp2p ping / gossip / connection
handling under concurrent decode load.

## Three Encryption Tiers (internal mechanisms)

### Tier 1: Pairwise Sessions (Unicast)

Underlying mechanism for the operator-facing Layer 1 above. For direct
peer-to-peer communication:
- Ed25519 → X25519 → ECDH → ChaCha20-Poly1305
- Forward secrecy via ephemeral X25519 re-keying every 10 minutes
- Nonce reuse prevented by session clearing on disconnect (`remove_session()`)
- Replay protection: RFC 6479 sliding window (128-bit bitmap) — allows packet reordering within window while rejecting duplicates
- Nonce state updated only after successful decryption (prevents DoS)
- Pending ephemeral keys expire after 60 seconds (prevents memory exhaustion from unanswered re-keys)
- AAD covers the cleartext header AND every optional trailer it emits (tensor-parallel, spec, kv-truncate, chunk-meta, chain next-hop/reply-to, generated-ids, pre-embedded — markers `0x02`–`0x09`) via `build_layer_forward_aad` — flipping cleartext metadata on the wire fails Poly1305
- Wire tag: `TENSOR_TAG_ENCRYPTED = 0x10` — a session-sealed `LayerForward` (activations sealed; header and trailers travel in cleartext, bound as AAD)

### Tier 2: Pipeline Sealing (Inference)

Designed to seal the final segment's output token IDs to the requester's X25519 key (per-request ephemeral key, `crypto::pipeline_seal`), so that only the requester could read them.

> **Not active today.** The final segment calls the seal without the requester's key (`src/daemon/dispatch/layer_forward.rs`, "Deliberately NOT fed the requester id"), so it never runs: every result path on the coordinator would first have to unseal. Returned results travel under libp2p's transport encryption (Noise on TCP, TLS 1.3 on QUIC), not under this seal. Either way the final-segment node sees the sampled tokens, because sampling happens on that node. See [Pipeline Privacy Model](#pipeline-privacy-model) for a full breakdown of what each node can see.

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

## DHT Provider Records

The Kademlia DHT holds only **provider records** (`start_providing` / `get_providers`) saying which peers hold which shard. They are not signed payloads:
- `start_providing_shards()` announces a shard key; it signs nothing
- A provider record is only as trustworthy as the peer that announced it — which is why every shard fetched from a provider is BLAKE3-verified on arrival, and a failed check costs that peer 0.20 trust (`ShardVerificationFail`)
- Provider records expire after 1 hour and are republished every 20 minutes
- `verify_dht_value()` would check `[32B pubkey][64B signature][payload]` values on a `GetRecord` result, but this build issues no `GetRecord` queries, so it never runs

## Identity

- Ed25519 keypair generated on first run, stored in `identity.key`
- Private key never leaves the machine
- Public key = Node ID (first 8 bytes hex for display)
- Nickname system: Ed25519-signed records with timestamp-wins conflict resolution
- Nickname registry capped at 10,000 entries (requires peer_registry membership)

## Trust & Reputation

`TrustManager` tracks per-peer scores (0.0-1.0, default 0.5):

| Event | Score Change | When it fires |
|---|---|---|
| InferenceSuccess | +0.01 | Each peer that served a well-formed distributed result |
| SpotCheckFail | -0.10 | A malformed result served by a single peer, or a rejected prefix-KV snapshot |
| ShardVerificationFail | -0.20 | A shard from that peer failed its BLAKE3 check |

`ValidTransaction`, `InvalidGossip`, `SignatureViolation` and `SubnetClustering` are defined in `src/credit/trust.rs` but nothing applies them.

Scores decay toward 0.5 over time (1% per health cycle, default 30 seconds). Trust orders pipeline candidates and gates cross-node prefix-KV fetches (`cross_node_prefix_trust_min`).

## Sybil Resistance

- Subnet clustering (>5 nodes per /24) is tracked; the elevated check rate it triggers applies only to inbound credit transactions, which no node sends while credits are dormant, so today it has no effect
- Signed-only balance reports
- Timestamp freshness checks on gossip (5 min window, rejects >5 min old)

## API Authentication

- Auto-generated 32-byte hex Bearer token (constant-time comparison)
- Protected: every route not listed as exempt below — including all of `/v1/*`, `/api/*` (reads as well as writes) and `/mcp`
- Exempt: `/`, `/health`, `/health/ready`, `/admin`, `/chat`, `/setup` and their sub-paths, `/static/*` and `/favicon.ico`
- `/metrics` is exempt from loopback only — and not even then when `api.metrics_auth_required = true`
- `/api/admin/ws` is exempt from Bearer auth but needs a single-use, short-lived ticket from `POST /api/admin/ws-ticket` (which is Bearer-authed)
- `GET /api/admin/api-key` answers without a key only to a network this node trusts (see `api::dashboard_trust`) and only with the dashboard's one-time page nonce
- Request body limit: 32 MB (raised from 2 MB to support VLM image payloads)
- Content-Security-Policy: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`
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
- Update binaries: SHA-256 checksum verification is mandatory, and the checksum file must carry a valid minisign signature from the offline release key compiled into the binary, with the asset name and version bound in the signature's trusted comment. An unsigned release is refused (`src/update_signature.rs`)

## Rate Limiting & DoS Protection

- Per-IP rate limiter with periodic cleanup (5 min intervals)
- Inference queue depth cap: 512 requests
- HTTP timeout: 5 minutes (Slowloris protection via tower-http TimeoutLayer).
  Model-running routes sit outside it (generation is unbounded in time) but
  still inside authentication and rate limiting.
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
| **Generated token IDs** | Yes | No | No | Yes (samples them) |
| **Final plaintext response** | Yes | No | No | Yes |

*\*Node A's visibility depends on the `local_embedding_privacy` setting — see below.*

### Risk: First-segment node sees raw tokens (default)

**Without `local_embedding_privacy`** (default): The first-segment node (Node A) receives the raw prompt text or token IDs to perform the embedding lookup. This means Node A can read the user's prompt in plaintext.

**With `local_embedding_privacy: true`**: The requesting node performs the embedding lookup locally and sends pre-embedded activation tensors. Node A receives floating-point vectors instead of token IDs. This is a significant privacy improvement, but not absolute — see [Activation Inversion Risk](#risk-activation-inversion-attacks) below.

### Risk: Final-segment node sees generated output

The final-segment node (Node C) **must** sample tokens from the logit distribution. This is fundamental — sampling is the act of choosing the next word, and it can only happen where the final layer's output logits exist. Node C therefore sees every generated token. (Tier 2 pipeline sealing, which would seal them to the requester's key, is not active — see above.)

**This cannot be mitigated architecturally.** The node that runs the last transformer layer and samples tokens will always know what tokens were sampled. The results travel back under the connection's transport encryption, which stops eavesdroppers on the network — but not the final-segment node itself.

### Risk: Activation inversion attacks

All intermediate nodes see hidden-state activation tensors (floating-point matrices), **in plaintext**: the wire encryption is removed on arrival, because a matrix multiply cannot run on ciphertext. And hidden states are **not** safe to treat as meaningless numbers:

- **Embedding-layer activations** (layer 0 output) are essentially a lookup table and can be reversed trivially.
- **Deeper layers are harder, but still leak a great deal.** Published inversion attacks (2025) recover roughly **81% of the input text even from final-layer hidden states**. Keeping more layers on your own machine therefore does not make a remote node blind — which is why there is no "keep N layers local" privacy setting.

**Treat a remote node doing work on your request as able to read it.** Truly hiding a prompt from the computer that processes it needs encrypted computation (homomorphic encryption or multi-party computation), which is currently about three orders of magnitude too slow to use here.

**What SwarmLLM does, and what each measure protects:**
1. Encryption in transit — every peer connection is encrypted (Noise / TLS), and activation forwards are additionally sealed with ChaCha20-Poly1305 (Tier 1). This stops **eavesdroppers on the network**, not the nodes doing the work.
2. `local_embedding_privacy: true` — the first remote node never receives the raw tokens or the trivially reversible embedding output. It removes the easy case, not the risk.
3. **Private Mode** — the only setting that keeps a request away from strangers' machines: your requests are served by your own devices (and, optionally, your local network) only.

### Risk: Byzantine tensor manipulation

A malicious node can send garbage activations instead of computing the actual transformer layers. Mitigation is limited: every distributed result gets a well-formedness check (tokens reported with no text, text with no tokens, or a reply that ran to the token cap repeating one token). A malformed result earns no participant any trust, and when a single peer served it, that peer loses 0.10 trust. Fluent but wrong output passes the check, because there is no replicated or known-answer verification (`src/inference/router/spot_check.rs`).

### Summary of privacy guarantees

All rows below assume **`network.enable_encryption = true`** (the
default) — every peer connection is encrypted in transport, and activation
forwards are additionally ChaCha20-Poly1305 sealed, regardless of which row
you're in. The columns describe what the
*endpoints* see; the wire is always encrypted.

| Configuration | Prompt privacy | Response privacy | Activation risk |
|---|---|---|---|
| **Default** (Layer 1 only, no privacy flags) | First segment receives the plaintext prompt | Final segment sees plaintext sampled tokens | Intermediate nodes see encrypted-on-wire activations; can decrypt at their boundary |
| `local_embedding_privacy: true` | No remote node sees raw token IDs | Final segment sees plaintext | Reduced — no trivial embedding inversion |
| `encrypted_pipeline: true` ("boomerang") | No remote node receives the prompt text or token IDs | No remote node samples the output | Remote nodes see intermediate activations in plaintext — **partially invertible to your text** |
| All protections enabled | Best available structurally | Best available structurally | Remote nodes still see activations that published attacks invert to much of the input |

> **Bottom line:** Layer 1 (basic wire encryption) is on by default —
> eavesdroppers on the network see ciphertext. The "boomerang" mode
> (`encrypted_pipeline`) keeps the two ENDS of the pipeline on your
> machine, so no remote node handles the prompt text or picks the reply's
> words. It is a **structural** guarantee, not a cryptographic one: the
> middle nodes still compute on your hidden states in plaintext, and those
> can be partly turned back into text. **For anything sensitive, use Private
> Mode or run the model entirely on your own machine.**

## Local Embedding Privacy

When `local_embedding_privacy: true` is set in `[inference]` config, the requesting node performs token→embedding lookup locally before sending activations to the first pipeline segment. Remote nodes never see raw token IDs — only hidden-state activation tensors.

**How it works:**
1. On startup, `LocalEmbedder` loads `token_embd.weight` from `shard_000.bin` and dequantizes it to f32 in RAM (vocab × hidden × 4 bytes — far larger than the quantized table on disk)
2. The requesting node tokenizes the prompt and performs the embedding lookup locally (~1ms)
3. The resulting hidden-state tensor (`[1, seq_len, hidden_dim]`, FP32) is sent as `LayerForward.activations` with `pre_embedded: true`
4. The receiving first-segment node skips its embedding lookup and processes the pre-embedded activations directly

**Wire format:** `pre_embedded` travels as its own `LayerForward` trailer (`0x09`), bound into the seal's AAD. It is sent only to a first-segment peer that advertises `features::FORWARD_PRE_EMBEDDED` (v0.3.195 and later). If the peer holding segment 0 does not advertise it, the request is refused (`PromptPrivacyUnavailable`) rather than sent as raw tokens.

**Trade-off:** Pre-embedded activations are larger than raw text (e.g., 512 tokens × 4096 hidden × 4 bytes = 8MB vs ~2KB text). This matches the existing inter-segment activation sizes, so it does not change the bandwidth profile of distributed inference.

Relevant code: `src/inference/local_embedder.rs`, `src/inference/pipeline/`, `src/daemon/state/mod.rs` (`local_embedders` DashMap).

## Encrypted Pipeline

When `encrypted_pipeline: true` is enabled (globally or per-model), the pipeline scheduler forces the requesting node to handle both the **first** and **last** segments. This creates a "boomerang" topology:

```
Requester (shard 0, embed) → Remote A (middle shards) → ... → Requester (final shard, decode)
```

No remote node handles the raw prompt tokens or samples the generated output — those two ends stay on your machine. Remote nodes process intermediate hidden-state activations, **in plaintext and partially invertible back to your text** (see [Activation inversion](#risk-activation-inversion-attacks)), so this is a structural guarantee, not a cryptographic one.

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
- Dashboard: gear icon on model card → "Start and finish on this computer" checkbox
- Global fallback: `encrypted_pipeline = true` in `[inference]` config
- Automatic: `encrypted_pipeline_auto = true` (default) turns it on for any model whose first and last shard this node holds, unless a per-model setting says otherwise
- Fetch the two ends so it can engage: `swarmllm privacy <model>` or `POST /api/admin/models/{id}/enable-privacy`
- Per-model overrides are persisted to the database

Relevant code: `src/inference/scheduler/mod.rs` (greedy_assign), `src/inference/pipeline/` (auto-enable local embedding), `src/api/admin_models/lifecycle.rs` (API endpoints), `src/daemon/state/mod.rs` (`encrypted_pipeline_models` DashMap).

## Known Limitations

These are architectural properties that cannot be fully mitigated with code changes:

- **Gossip epoch key is publicly derivable** — derived from "swarmllm-mainnet-v1". Gossip encryption is defense-in-depth; Ed25519 signing is the primary security mechanism.
- **Final-segment output visibility** — the node running the last transformer layers sees all generated tokens. This is inherent to the architecture (see [Pipeline Privacy Model](#pipeline-privacy-model)).
- **Activation inversion** — each computing node decrypts the hidden states it works on, and published 2025 attacks recover roughly 81% of the input text even from final-layer hidden states (see [Activation inversion](#risk-activation-inversion-attacks)). `local_embedding_privacy` removes only the trivial case (reversing the embedding lookup). Treat any remote node computing on your request as able to read it.
- **Byzantine tensor manipulation** — malicious peers can send garbage activations. Mitigation is limited to a well-formedness check on every distributed result; fluent but wrong output is not detected (see [Byzantine tensor manipulation](#risk-byzantine-tensor-manipulation)).
- **Sybil credit farming** — Ed25519 keys are free. Anti-gaming heuristics help but are not bulletproof.
- **GGUF parser vulnerabilities** — llama.cpp CVEs. BLAKE3 content hash gates shard loading but parser bugs remain upstream.
- **Kademlia eclipse attacks** — strategic Sybil node IDs can control DHT routing. K-bucket eviction policies help.
