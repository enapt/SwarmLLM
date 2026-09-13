# Credit System

> ## ⚠ Credits are switched off, and gate nothing
>
> Since 2026-08-17 the credit economy is **dormant**. `MIN_BALANCE_FOR_INFERENCE`
> is `0` and `calculate_tier` returns the same tier whatever balance it is
> given, so **no balance affects who is served, how fast, or what any surface
> shows**. Nothing displays a balance either — not the dashboard, not the
> leaderboard, not the command line.
>
> The reason is simple and worth stating: credit has never moved between two
> machines as payment for work. Each node mints its own figure, and the one real
> transfer — pool forwarding — just concentrates self-minted numbers. Showing
> someone a number like that as earnings is a claim the project cannot stand
> behind; acting on it is worse.
>
> **The accounting below still runs and the rates below are real** — it costs
> nothing and the recorded figures are the only honest data about what traffic
> actually looks like, which is what a re-enabled economy would have to be
> designed from. But read every "ensures", "enforced" and "deprioritized" on
> this page as *what the design does when switched on*, not as what your node
> does today.
>
> The full picture — what is actually true now, why the obvious fix is wrong,
> the bilateral-settlement design, and the exit criteria that must hold before
> any of it comes back — is in
> [`docs/CREDITS_DESIGN.md`](https://github.com/enapt/SwarmLLM/blob/main/docs/CREDITS_DESIGN.md).

Credits are SwarmLLM's fairness mechanism — no blockchain, no token, just local accounting with dual-signed transactions. The design rewards contributors and deprioritizes free-riders.

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

## Minimum Balance Enforcement (switched OFF)

**`MIN_BALANCE_FOR_INFERENCE` is `0`, which the enforcement check reads as "no
floor". No requester is refused over credits today.** It was `-1000`, and the
paragraph below describes what that did. It was switched off because refusing
somebody service over a self-minted figure is not a small inaccuracy — it is
denying them the product on the strength of a number nobody reconciled, and a
node driven negative by a bug stayed refused with no recovery path.

Nodes with balance below **-1000 credits** had remote inference requests rejected. They receive a clear error message telling them to contribute (host shards, serve inference, seed data).

- Local API requests (from localhost) are always allowed regardless of balance
- This prevents free-riders from endlessly consuming without contributing
- The floor is configurable via `MIN_BALANCE_FOR_INFERENCE` constant

## Priority Tiers (switched OFF)

**`calculate_tier` returns the same tier for everyone whatever balance it is
given**, so the table below is what the design does when switched on, not what
your node does. Per-requester concurrency isolation is preserved — one peer
still cannot monopolise the queue — while the *advantage* from a minted number
is gone.

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
