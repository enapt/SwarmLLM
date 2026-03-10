# Credit System

Credits are SwarmLLM's fairness mechanism — no blockchain, no token, just local accounting with dual-signed transactions.

## Earning & Spending

| Action | Credits | Status |
|---|---|---|
| Serve inference (per layer per token) | +10 | Active |
| Forward activations (per layer processed) | +10 | Active |
| Host shard (per GB per hour) | +1 | Active (hourly tick in CreditLedger) |
| Seed shard data (per GB transferred) | +5 | Active (atomic counter, periodic drain) |
| Relay traffic (per connection hour) | +2 | Active (circuit open/close tracking) |
| Consume inference (per layer per token) | -10 | Active |
| Distributed inference failure | -50 | Active (automatic penalty) |

All rates are configurable per pool via `[pool.credit_rates]` in config.

## Priority Tiers

Tiers are calculated from your credit balance relative to the network:

| Tier | Requirement | Queue Priority | Concurrent Limit |
|---|---|---|---|
| Platinum | ≥90th percentile balance | Immediate | 2× base max |
| Gold | ≥70th percentile | 1-3s | base max |
| Silver | Positive balance | 5-15s | ½ base max |
| Bronze | Zero or negative | 30s+ | ¼ base max |

**How it works:** On each inference request, the router computes your network percentile from peer credit gossip data and calls `calculate_tier()`. The tier determines both queue ordering (higher tiers dequeue first via `tier_weight()`) and concurrent execution slots (via `max_concurrent_for_tier()`). Bronze nodes are never blocked — they get deprioritized but always served.

Local inference (single-node) never costs credits.

## Failure Penalties

When distributed inference fails:
- The requesting node is penalized (configurable `penalty_serve_failure`, default 50 credits)
- A `broadcast_pipeline_error()` message is sent to all pipeline participants
- Remote peers can update their shard availability in response

## Transaction Security

- Every transaction requires dual Ed25519 signatures (serving node + requesting node)
- UUID deduplication prevents replay attacks (checked against DB)
- Balance arithmetic uses `saturating_add` (no overflow panics)
- Peer balance gossip rejects implausible values (abs > 100M)
- Signed balance reports with 5-minute timestamp freshness window

## Escrow

For large requests (above configurable threshold), credits are held in escrow:
- `create_escrow()` → `release_escrow()` (success) or `refund_escrow()` (failure)
- Entries expire after 10 minutes with automatic refund
- Persisted in redb `escrow` table
