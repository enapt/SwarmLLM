# Credit System

Credits are SwarmLLM's fairness mechanism — no blockchain, no token, just local accounting with dual-signed transactions. The system ensures contributors are rewarded and free-riders are deprioritized.

## Earning & Spending

| Action | Credits | Notes |
|---|---|---|
| Serve inference (per token) | +10 | Balanced with consume side |
| Host shard (per GB per hour) | +1 | Hourly tick in CreditLedger |
| Seed shard data (per GB transferred) | +5 | Atomic counter, periodic drain |
| Relay traffic (per connection hour) | +2 | Circuit open/close tracking |
| Consume inference (per token) | -10 | Balanced with earn side |
| Distributed inference failure | -50 | Automatic penalty |

**Balanced rates**: Both earn and spend use `rate × tokens` — no layer multiplier. A 22-layer model serving 100 tokens earns the same as it costs to consume, preventing credit inflation.

All rates are configurable per pool via `[pool.credit_rates]` in config.

## Minimum Balance Enforcement

Nodes with balance below **-1000 credits** have remote inference requests rejected. They receive a clear error message telling them to contribute (host shards, serve inference, seed data).

- Local API requests (from localhost) are always allowed regardless of balance
- This prevents free-riders from endlessly consuming without contributing
- The floor is configurable via `MIN_BALANCE_FOR_INFERENCE` constant

## Priority Tiers

Tiers are calculated from your credit balance relative to the network:

| Tier | Requirement | Concurrent Limit |
|---|---|---|
| Platinum | ≥90th percentile **and** balance > 0 | 2× base max |
| Gold | ≥70th percentile **and** balance > 0 | base max |
| Silver | Positive balance | ½ base max |
| Bronze | Zero or negative | ¼ base max (min 1) |

**How it works:** On each inference request, the router computes your network percentile from peer credit gossip data (deduplicated by NodeId to prevent Sybil stuffing) and calls `calculate_tier()`. Higher tiers dequeue first. Bronze nodes are never fully blocked — they get deprioritized but always get at least 1 concurrent slot.

## Anti-Abuse Mechanisms

- **Anti-Sybil deduplication**: Peer balance gossip is deduplicated by NodeId — a single peer can't stuff the percentile distribution by re-gossiping
- **Atomic accumulation**: Forward participation credits use `AtomicI64` accumulator, flushed every 60s — no credits lost under high concurrency
- **AntiGaming rate limiter**: Max 100 credit transactions per node per 5-minute window
- **Self-dealing rejection**: Transactions from/to same node are rejected
- **Signed balance reports**: Ed25519 signatures with 5-minute freshness window

## Failure Penalties

When distributed inference fails:
- The requesting node is penalized (configurable `penalty_serve_failure`, default 50 credits)
- A `broadcast_pipeline_error()` message is sent to all pipeline participants

## Transaction Security

- Every transaction requires dual Ed25519 signatures (serving node + requesting node)
- UUID deduplication prevents replay attacks (checked against DB)
- Balance arithmetic uses `saturating_add` (no overflow panics)
- Peer balance gossip rejects implausible values (abs > 100M)

## Escrow

For large requests (above configurable threshold), credits are held in escrow:
- `create_escrow()` → `release_escrow()` (success) or `refund_escrow()` (failure)
- Balance deducted BEFORE escrow persisted (crash-safe: lose credits > create free credits)
- Refunds are persisted to DB immediately
- Entries expire after 10 minutes with automatic refund
- Escrow and direct charge are mutually exclusive (no double-billing)

## Device Pool Credit Forwarding

When devices are linked in a pool, member devices forward their earnings to the owner:
- Credit split configurable: 0-50% kept by member, rest forwarded
- Dual-signed `PoolCreditForward` (member signature + owner co-signature)
- Forwarded amount deducted from member balance before persisting
- Owner's `PoolManager` validates and applies credits atomically
