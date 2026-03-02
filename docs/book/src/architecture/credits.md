# Credit System

Credits are SwarmLLM's fairness mechanism — no blockchain, no token, just local accounting with dual-signed transactions.

## Earning & Spending

| Action | Credits |
|---|---|
| Serve inference (per layer per token) | +10 |
| Host shard (per GB per hour) | +1 |
| Seed shard data (per GB transferred) | +5 |
| Relay traffic (per connection hour) | +2 |
| Consume inference (per layer per token) | -10 |
| Serve failure (timeout) | -50 |

Rates are configurable per pool via `[pool.credit_rates]` in config.

## Priority Tiers

| Tier | Requirement | Queue Priority |
|---|---|---|
| Platinum | ≥90th percentile balance | Immediate |
| Gold | ≥70th percentile | 1-3s |
| Silver | Positive balance | 5-15s |
| Bronze | Zero or negative | 30s+ |

Local inference (single-node) never costs credits.

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
- Persisted in sled `escrow` tree
