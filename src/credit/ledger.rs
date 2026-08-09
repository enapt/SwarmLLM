use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use ed25519_dalek::VerifyingKey;
use tokio::sync::{mpsc, watch, RwLock};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, CreditGossip, NetworkCommand, NodeId, PriorityTier, ShardId, SwarmMessage,
};

/// Earning/spending rates (credits per unit).
/// IMPORTANT: Both earn and spend use `rate * tokens` (no layer multiplier) to prevent
/// credit inflation. A 22-layer model serving 100 tokens earns 10*100=1000 credits,
/// and the consumer spends 10*100=1000 credits — balanced.
/// These constants are default values. The `CreditLedger` methods (`earn_inference`,
/// `spend_inference`) resolve overrides from `[pool.credit_rates]` in config.toml,
/// but some callers (dispatch.rs, router.rs) reference these constants directly.
pub const RATE_INFERENCE_SERVE: i64 = 10; // per token served (not per layer)
pub const RATE_INFERENCE_CONSUME: i64 = 10; // per token consumed — balanced with serve

/// Minimum credit balance required to submit inference requests.
/// Nodes below this floor have their requests rejected (not just deprioritized).
/// Set to 0 to disable enforcement (permissive mode for small networks).
/// Nodes can earn credits by hosting shards, serving inference, or seeding data.
pub const MIN_BALANCE_FOR_INFERENCE: i64 = -1000;

/// Database tree name for credit data.
pub const TREE_CREDITS: &str = "credits";
pub const KEY_BALANCE: &str = "balance";
/// Tree name for credit transaction history (spec: `credit_txns`).
pub const TREE_TRANSACTIONS: &str = "credit_txns";

/// Interval between bucketed-balance gossip messages used for percentile estimation.
const LEDGER_GOSSIP_INTERVAL_SECS: u64 = 300;
/// Interval between on-disk balance persistence ticks.
const LEDGER_PERSIST_INTERVAL_SECS: u64 = 60;
/// Interval between shard-hosting credit payouts.
const LEDGER_HOSTING_INTERVAL_SECS: u64 = 3600;
/// Interval between expired-escrow cleanup sweeps.
const LEDGER_ESCROW_CLEANUP_INTERVAL_SECS: u64 = 300;
/// Interval between negative-balance decay ticks (walks a negative balance
/// back toward zero — see `negative_balance_decay_amount`).
const LEDGER_DECAY_INTERVAL_SECS: u64 = 3600;
/// Fraction of the outstanding deficit forgiven on each decay tick.
const NEGATIVE_BALANCE_DECAY_FRACTION: f64 = 0.05;
/// Minimum credits forgiven per decay tick, so a modest deficit clears in a
/// bounded number of ticks instead of shrinking asymptotically forever.
const NEGATIVE_BALANCE_DECAY_FLOOR: i64 = 500;

/// Credits to forgive this tick for a node sitting at a negative `balance`.
///
/// Credit balances gate *scaling and priority*, not correctness — a node's own
/// local inference is always served, and the `MIN_BALANCE_FOR_INFERENCE` floor
/// only affects requests to/from other nodes. That makes a deeply negative
/// balance a soft penalty, but a *permanent* one: without any recovery path a
/// node driven far below zero (e.g. by a framework bug that over-charged it, or
/// a burst of failed distributed requests) stays stuck at Bronze and below the
/// pool-participation floor forever, even after the cause is fixed.
///
/// This walks a negative balance back toward zero — never past it (decay must
/// not mint positive credits) and never touching the monotonic
/// `lifetime_earned`/`lifetime_spent` counters (applied via `CreditDelta::Refund`).
/// The percentage term makes a huge deficit recover in a bounded number of
/// ticks; the flat floor clears the long tail. Positive balances are left
/// entirely alone — good contributors are never decayed.
///
/// Returns `0` for a non-negative balance (no-op).
pub(crate) fn negative_balance_decay_amount(balance: i64) -> i64 {
    if balance >= 0 {
        return 0;
    }
    // Positive magnitude of the deficit; `saturating_neg` handles `i64::MIN`.
    let deficit = balance.saturating_neg();
    let pct = (deficit as f64 * NEGATIVE_BALANCE_DECAY_FRACTION).ceil() as i64;
    // At least the flat floor, but never more than the deficit itself (so we
    // land exactly on zero rather than overshooting into positive territory).
    pct.max(NEGATIVE_BALANCE_DECAY_FLOOR).min(deficit)
}

/// Attribute a pre-existing earned/spent/balance gap to refunds, once, on load.
///
/// `lifetime_refunded` was added after nodes had been running for weeks. It
/// defaults to 0 on an old persisted record, but the refunds it should have
/// counted are already folded into `balance` — so the identity
/// `earned - spent + refunded == balance` could never close on an existing
/// install, and `books_balance` reported `false` for ever. That is precisely the
/// false alarm the field was added to remove: observed on both test nodes
/// immediately after shipping it (gaps of 650 and 1680).
///
/// A refund is the only mechanism that makes `balance` exceed
/// `earned - spent`, so the gap IS historical refunds and labelling it as such
/// is accurate rather than a fudge.
///
/// Fires on the REMAINING gap, whatever `lifetime_refunded` already reads.
///
/// The first version keyed off `lifetime_refunded == 0`, reasoning that a
/// node already recording refunds needed no help. That conflates two
/// different quantities: refunds recorded *since the counter shipped* (real,
/// already reflected in `balance`, must not be double-counted) and the
/// *historical* gap from before the field existed (never recorded, needs
/// attributing). A node has both the moment it takes a single refund between
/// the counter shipping and the migration running — and the `!= 0` guard saw
/// only the first, skipped the node, and left its historical gap permanently
/// unexplained. Reported from a live node whose ~905k gap survived the
/// migration because a deliberately-provoked timeout the day before had put
/// 640 in the counter.
///
/// Working from the remaining gap instead is both a strict generalisation —
/// at `lifetime_refunded == 0` it computes exactly what the old version did —
/// and idempotent by construction, since applying it drives the gap to 0 and
/// a second call is a no-op. A correctly-recorded refund raises `balance` and
/// `lifetime_refunded` together, so it cancels out of the gap and cannot be
/// double-counted; only unrecorded history shows up here.
///
/// A fresh node has all four at 0 and is left alone. A NEGATIVE gap is left
/// alone too — that direction cannot be explained by refunds, so inventing a
/// number would hide a genuine inconsistency instead of surfacing it.
pub(crate) fn backfill_historical_refunds(bal: &mut CreditBalance) {
    let gap = (bal.balance as i128) - (bal.lifetime_earned as i128) + (bal.lifetime_spent as i128)
        - (bal.lifetime_refunded as i128);
    if gap <= 0 {
        return;
    }
    let gap = gap.min(u64::MAX as i128) as u64;
    let already_recorded = bal.lifetime_refunded;
    bal.lifetime_refunded = bal.lifetime_refunded.saturating_add(gap);
    tracing::info!(
        attributed = gap,
        already_recorded,
        lifetime_refunded = bal.lifetime_refunded,
        balance = bal.balance,
        lifetime_earned = bal.lifetime_earned,
        lifetime_spent = bal.lifetime_spent,
        "Attributed a pre-existing credit gap to historical refunds so the ledger reconciles"
    );
}

/// CreditLedger tracks the local node's credit balance and transaction history.
///
/// It persists balance and transactions to redb, and gossips balance buckets
/// for network-wide percentile estimation.
pub struct CreditLedger {
    node_id: NodeId,
    balance: Arc<RwLock<CreditBalance>>,
    db: Database,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bucketed balances from other nodes, used for percentile estimation.
    ///
    /// `ArcSwap` gives lock-free reads on the inference credit-gate hot
    /// path: `estimate_percentile` → `.load_full()` is a single atomic Arc
    /// clone with no await, even while `process_balance_gossip` is storing
    /// a fresh snapshot on the dispatcher task. Previously this was an
    /// `Arc<RwLock<Vec<i64>>>` — writes (one per inbound `CreditGossip`
    /// in the dispatcher) and reads (inference credit-gate check) held
    /// the same RwLock, so a busy swarm could serialize the two.
    peer_balances: Arc<ArcSwap<Vec<i64>>>,
    /// Reference to SharedState for pool credit forwarding.
    shared_state: Option<std::sync::Arc<crate::daemon::SharedState>>,
    /// Node identity for signing balance reports (Sybil resistance).
    identity: Option<crate::identity::Identity>,
    /// Fractional shard-hosting credit carried between hourly accruals, in
    /// millionths of a credit.
    ///
    /// Hosting pays `rate * size_gb * hours` and the result was truncated to an
    /// integer **per shard**. With the shipped defaults — `shard_size_mb = 512`
    /// and `shard_hosting = 1` — that is `1 * 0.5 * 1.0 = 0.5`, which truncates
    /// to **zero**. Every shard smaller than 1 GB therefore earned nothing, on
    /// every node, for ever, while spending worked normally: a node that hosted
    /// more got monotonically poorer, inverting the whole incentive. Reported
    /// 2026-07-30 by an operator who went from 5 to 13 shards and watched
    /// `credits_earned=0` at every accrual.
    ///
    /// Integer micro-credits rather than an `f64` accumulator so repeated ticks
    /// cannot drift. In memory only — losing it on restart forfeits less than
    /// one credit, which is not worth a write on every tick.
    hosting_remainder_micro: std::sync::atomic::AtomicI64,
}

impl CreditLedger {
    pub fn new(
        node_id: NodeId,
        balance: Arc<RwLock<CreditBalance>>,
        db: Database,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
        peer_balances: Arc<ArcSwap<Vec<i64>>>,
    ) -> Self {
        // Restore persisted balance synchronously to avoid race condition.
        // redb reads are fast, so this is safe in constructor.
        let restored = match db.get_json::<CreditBalance>(TREE_CREDITS, KEY_BALANCE) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "DIAG: failed to read credit balance from database — starting at zero"
                );
                None
            }
        };

        if let Some(mut restored_balance) = restored {
            if restored_balance.node_id == node_id {
                backfill_historical_refunds(&mut restored_balance);
                // Use try_write — should always succeed at startup before concurrent access.
                if let Ok(mut bal) = balance.try_write() {
                    tracing::info!(
                        balance = restored_balance.balance,
                        lifetime_earned = restored_balance.lifetime_earned,
                        "Restored credit balance from database"
                    );
                    *bal = restored_balance;
                } else {
                    tracing::error!(node_id = %node_id, "CRITICAL: Failed to restore credit balance — lock unavailable at startup. Balance may be zero.");
                }
            }
        }

        Self {
            node_id,
            balance,
            db,
            network_tx,
            shutdown_rx,
            peer_balances,
            shared_state: None,
            identity: None,
            hosting_remainder_micro: std::sync::atomic::AtomicI64::new(0),
        }
    }

    /// Set a shared state reference for pool credit forwarding.
    pub fn set_shared_state(&mut self, shared_state: std::sync::Arc<crate::daemon::SharedState>) {
        self.shared_state = Some(shared_state);
    }

    /// Set the node identity for signing balance reports (Sybil resistance).
    pub fn set_identity(&mut self, identity: crate::identity::Identity) {
        self.identity = Some(identity);
    }

    pub async fn get_balance(&self) -> CreditBalance {
        self.balance.read().await.clone()
    }

    /// Resolve effective credit rates: per-pool override > global config > compile-time defaults.
    fn credit_rates(&self) -> crate::config::CreditRateConfig {
        if let Some(ref ss) = self.shared_state {
            resolve_credit_rates(ss)
        } else {
            crate::config::CreditRateConfig::default()
        }
    }

    /// Earn credits for serving inference.
    ///
    /// Formula: `rate * tokens` — no layer multiplier to stay balanced with the
    /// spend side (`rate * tokens`). Each node that participates in a distributed
    /// pipeline earns for the tokens it helped produce, regardless of how many
    /// layers it processed. This prevents credit inflation on deep models.
    ///
    /// If this node is a pool member (not owner), forwards credits to the pool owner.
    pub async fn earn_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
    ) -> Result<i64, SwarmError> {
        let rates = self.credit_rates();
        let amount = rates.inference_serve.saturating_mul(tokens as i64);

        tracing::info!(
            amount,
            tokens,
            request_id = %request_id,
            "Earned credits for inference serving"
        );

        // Forward (a portion of) earned credits to the pool owner per
        // PoolState::member_credit_split_pct. The forwarder returns the
        // member's local-credit share — full amount if not in pool / is
        // owner / forward channel unavailable; member_keeps if a partial
        // split is configured; zero if 100% forwarded. We then apply
        // exactly that to the local balance — no inflation, no theft of
        // the configured split.
        let local_credit = if let Some(ref ss) = self.shared_state {
            match crate::pool::forward::forward_credits_to_owner(ss, amount).await {
                Ok(local) => local,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        amount,
                        "Pool credit forwarding failed — crediting locally. Owner will not receive these credits."
                    );
                    amount
                }
            }
        } else {
            amount
        };

        if local_credit > 0 {
            self.apply_credit(local_credit, CreditDelta::Earning)
                .await?;
            self.persist_balance().await?;
        }
        if local_credit < amount {
            tracing::info!(
                forwarded = amount - local_credit,
                kept = local_credit,
                "Forwarded share of earned credits to pool owner"
            );
        }

        Ok(local_credit)
    }

    /// Spend credits for consuming inference.
    /// Formula: `rate * tokens` — balanced with earn side.
    pub async fn spend_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
    ) -> Result<i64, SwarmError> {
        let rates = self.credit_rates();
        let amount = rates.inference_consume.saturating_mul(tokens as i64);
        self.apply_credit(-amount, CreditDelta::Spending).await?;
        self.persist_balance().await?;

        tracing::info!(
            amount,
            tokens,
            request_id = %request_id,
            "Spent credits for inference consumption"
        );

        Ok(amount)
    }

    /// Saturate a raw f64 credit amount into a non-negative i64, capped at `max`.
    ///
    /// SEC: every earn_* path takes peer-influenceable inputs (shard size,
    /// bytes transferred, relay duration) and multiplies by an f64 rate.
    /// Without this guard a hostile or buggy upstream could feed `raw =
    /// f64::INFINITY` or a value beyond `i64::MAX`, and the `as i64` cast
    /// would saturate to `i64::MAX` — an instant Platinum-tier credit mint.
    /// Reject non-finite, clamp negatives to zero, and cap to a per-call
    /// ceiling that's implausibly generous for honest inputs (1M credits).
    #[inline]
    fn safe_f64_credits(raw: f64, max: f64) -> i64 {
        if raw.is_finite() && raw >= 0.0 {
            raw.min(max) as i64
        } else {
            0
        }
    }

    /// Earn credits for hosting a shard.
    /// Accrue hosting credit for the WHOLE set of shards held this tick.
    ///
    /// Batched, not per-shard, and carries the sub-credit remainder forward.
    /// Paying each shard separately truncated `0.5` to `0` for every 512 MB
    /// shard — see `hosting_remainder_micro`. Summing first means a node with
    /// two half-gigabyte shards earns 1/hour instead of 0, and the carry means
    /// even a single small shard eventually pays rather than rounding away for
    /// ever.
    ///
    /// One `apply_credit` + one persist per tick rather than per shard, so a
    /// node hosting hundreds of shards no longer does hundreds of DB writes an
    /// hour either.
    pub async fn earn_shard_hosting_total(
        &self,
        total_gb: f64,
        hours: f32,
    ) -> Result<i64, SwarmError> {
        use std::sync::atomic::Ordering;
        let rates = self.credit_rates();
        let raw = rates.shard_hosting as f64 * total_gb * hours as f64;
        if !raw.is_finite() || raw < 0.0 {
            return Ok(0);
        }
        // Cap before converting so a preposterous rate cannot overflow.
        const MAX_HOSTING_PER_TICK: f64 = 1_000_000.0;
        let micro = (raw.min(MAX_HOSTING_PER_TICK) * 1_000_000.0).round() as i64;
        let carried = self.hosting_remainder_micro.load(Ordering::Relaxed);
        let total_micro = micro.saturating_add(carried);
        let amount = total_micro / 1_000_000;
        self.hosting_remainder_micro
            .store(total_micro % 1_000_000, Ordering::Relaxed);
        if amount > 0 {
            self.apply_credit(amount, CreditDelta::Earning).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for seeding shard data.
    pub async fn earn_shard_seeding(
        &self,
        _shard_id: &ShardId,
        bytes_transferred: u64,
    ) -> Result<i64, SwarmError> {
        let gb = bytes_transferred as f64 / (1024.0 * 1024.0 * 1024.0);
        let rates = self.credit_rates();
        let raw = rates.shard_seeding as f64 * gb;
        const MAX_SEEDING_PER_CALL: f64 = 1_000_000.0;
        let amount = Self::safe_f64_credits(raw, MAX_SEEDING_PER_CALL);
        if amount > 0 {
            self.apply_credit(amount, CreditDelta::Earning).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for forwarding application-level inference relay traffic
    /// (NETWORKING_PLAN Phase 3). Priced at the same per-byte rate as shard
    /// seeding — relaying a GB of inference traffic is treated as the same
    /// contribution as seeding a GB of shard data. Purely informational /
    /// priority-affecting today (credits are not enforced for correctness); this
    /// keeps donated relay capacity counting as a contribution so the incentive
    /// story holds if credits are ever hardened.
    pub async fn earn_relay_forwarding(&self, bytes_forwarded: u64) -> Result<i64, SwarmError> {
        let gb = bytes_forwarded as f64 / (1024.0 * 1024.0 * 1024.0);
        let rates = self.credit_rates();
        let raw = rates.shard_seeding as f64 * gb;
        const MAX_RELAY_FWD_PER_CALL: f64 = 1_000_000.0;
        let amount = Self::safe_f64_credits(raw, MAX_RELAY_FWD_PER_CALL);
        if amount > 0 {
            self.apply_credit(amount, CreditDelta::Earning).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for relay service.
    pub async fn earn_relay_service(&self, duration_seconds: u64) -> Result<i64, SwarmError> {
        let hours = duration_seconds as f64 / 3600.0;
        let rates = self.credit_rates();
        let raw = rates.relay_service as f64 * hours;
        const MAX_RELAY_PER_CALL: f64 = 1_000_000.0;
        let amount = Self::safe_f64_credits(raw, MAX_RELAY_PER_CALL);
        if amount > 0 {
            self.apply_credit(amount, CreditDelta::Earning).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Calculate the current priority tier based on balance and network percentile.
    pub async fn calculate_tier(&self) -> PriorityTier {
        let bal = self.balance.read().await;
        let percentile = self.estimate_percentile(bal.balance);
        super::priority::calculate_tier(bal.balance, percentile)
    }

    #[cfg(test)]
    pub fn peer_balances(&self) -> &Arc<ArcSwap<Vec<i64>>> {
        &self.peer_balances
    }

    /// Estimate this node's percentile in the network.
    fn estimate_percentile(&self, balance: i64) -> f32 {
        let balances = self.peer_balances.load_full();
        if balances.is_empty() {
            // With no network data, use balance sign as a proxy
            return if balance > 0 { 0.5 } else { 0.1 };
        }

        let below = balances.iter().filter(|&&b| b < balance).count();
        below as f32 / balances.len() as f32
    }

    /// Apply a credit delta to the balance.
    /// SEC-I1: Uses saturating arithmetic to prevent overflow.
    async fn apply_credit(&self, delta: i64, kind: CreditDelta) -> Result<(), SwarmError> {
        let mut bal = self.balance.write().await;
        bal.balance = bal.balance.saturating_add(delta);
        bal.last_updated = chrono::Utc::now();

        // Counters take |delta|, so the KIND and the SIGN must agree or the
        // books stop closing. Taking `CreditDelta` rather than a bool means the
        // two accounting paths share one vocabulary and a refund is
        // expressible here too, instead of being silently counted as a spend.
        match kind {
            CreditDelta::Earning => {
                bal.lifetime_earned = bal.lifetime_earned.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Spending => {
                bal.lifetime_spent = bal.lifetime_spent.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Refund => {
                bal.lifetime_refunded = bal.lifetime_refunded.saturating_add(delta.unsigned_abs());
            }
        }

        tracing::debug!(
            balance = bal.balance,
            delta,
            lifetime_earned = bal.lifetime_earned,
            lifetime_spent = bal.lifetime_spent,
            "Credit balance updated"
        );

        Ok(())
    }

    /// Persist the current balance to the database.
    /// Snapshots out of the lock before the synchronous redb write so a
    /// queued writer (apply_credit on the inference hot path) doesn't park
    /// on the read lock for the disk write duration.
    async fn persist_balance(&self) -> Result<(), SwarmError> {
        let snapshot = self.balance.read().await.clone();
        self.db.put_json(TREE_CREDITS, KEY_BALANCE, &snapshot)?;
        Ok(())
    }

    /// Run the credit ledger background task.
    ///
    /// This periodically:
    /// - Persists the balance to disk
    /// - Gossips bucketed balance for percentile estimation
    /// - Calculates and logs the current tier
    pub async fn run(self) -> Result<(), SwarmError> {
        tracing::info!(target: "swarmllm::credit::ledger", "CreditLedger running");

        // Gossip bucketed balance for percentile estimation.
        let mut gossip_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_GOSSIP_INTERVAL_SECS));
        gossip_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Persist balance to disk.
        let mut persist_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_PERSIST_INTERVAL_SECS));
        persist_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Earn shard hosting credits.
        let mut hosting_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_HOSTING_INTERVAL_SECS));
        hosting_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Clean up expired escrows.
        let mut escrow_cleanup_interval = tokio::time::interval(std::time::Duration::from_secs(
            LEDGER_ESCROW_CLEANUP_INTERVAL_SECS,
        ));
        escrow_cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Walk a negative balance back toward zero.
        let mut decay_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_DECAY_INTERVAL_SECS));
        decay_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Consume the immediate t=0 tick so decay only applies after a full
        // interval of uptime — otherwise a node could restart-farm the decay,
        // clearing its deficit faster than the intended recovery rate.
        decay_interval.tick().await;

        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        // Final persist on shutdown
                        let _ = self.persist_balance().await;
                        tracing::info!(target: "swarmllm::credit::ledger", "CreditLedger shutting down");
                        break;
                    }
                }
                _ = gossip_interval.tick() => {
                    // Read once, derive both. Two sequential read-locks let a
                    // concurrent apply_credit slip a write between them, so the
                    // gossiped (bucket, tier) pair could reflect different
                    // balance values.
                    let (bucket, tier) = {
                        let balance_now = {
                            let bal = self.balance.read().await;
                            bal.balance
                        };
                        let percentile = self.estimate_percentile(balance_now);
                        (
                            bucket_balance(balance_now),
                            super::priority::calculate_tier(balance_now, percentile),
                        )
                    };

                    tracing::info!(
                        balance_bucket = bucket,
                        tier = ?tier,
                        "Gossiping credit balance"
                    );

                    // Sign the balance report if identity is available
                    let timestamp = chrono::Utc::now();
                    let signature = if let Some(ref identity) = self.identity {
                        sign_balance_report(&self.node_id, bucket, timestamp, identity)
                    } else {
                        Vec::new()
                    };

                    let msg = SwarmMessage::CreditGossip(CreditGossip {
                        node_id: self.node_id.clone(),
                        balance_bucket: bucket,
                        timestamp,
                        signature,
                    });

                    if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
                        tracing::warn!(error = %e, "Failed to send credit gossip");
                    }
                }
                _ = persist_interval.tick() => {
                    // Flush pending credit earn accumulator from forward participation.
                    // This prevents credit loss from try_write() contention in the hot path.
                    if let Some(ref ss) = self.shared_state {
                        let pending = ss.credits.pending_credit_earn.swap(0, std::sync::atomic::Ordering::AcqRel);
                        if pending != 0 {
                            if let Err(e) = apply_credit_direct_noted(
                                &self.balance,
                                &self.db,
                                pending,
                                if pending > 0 { CreditDelta::Earning } else { CreditDelta::Spending },
                                // Tagged, not "unspecified". This is the one
                                // place inference serving reaches the balance,
                                // and an untagged +440 in the transaction log is
                                // exactly the movement nobody can account for
                                // that the log exists to prevent.
                                "inference_serve_earning",
                            ).await {
                                // Restore the credits so they aren't lost. Use
                                // compare_exchange so we don't double-count: if
                                // a concurrent earn_inference call landed
                                // increments while we were flushing, the field
                                // is no longer 0 and a naive fetch_add would
                                // add `pending` on top of those increments.
                                // On the contended path we fall back to
                                // fetch_add anyway — losing the credits is
                                // worse than the rare overcount, and the next
                                // flush tick will reconcile against the DB.
                                match ss.credits.pending_credit_earn.compare_exchange(
                                    0,
                                    pending,
                                    std::sync::atomic::Ordering::AcqRel,
                                    std::sync::atomic::Ordering::Acquire,
                                ) {
                                    Ok(_) => tracing::warn!(
                                        error = %e,
                                        pending,
                                        "Failed to flush pending credit earn — restored cleanly"
                                    ),
                                    Err(observed) => {
                                        ss.credits.pending_credit_earn.fetch_add(
                                            pending,
                                            std::sync::atomic::Ordering::AcqRel,
                                        );
                                        tracing::warn!(
                                            error = %e,
                                            pending,
                                            observed,
                                            "Failed to flush pending credit earn — concurrent earn detected, restored with potential overcount"
                                        );
                                    }
                                }
                            } else {
                                tracing::debug!(pending, "Flushed pending forward participation credits");
                            }
                        }
                    }
                    if let Err(e) = self.persist_balance().await {
                        tracing::warn!(error = %e, "Failed to persist credit balance");
                    }
                }
                _ = hosting_interval.tick() => {
                    // Earn credits for each shard we're hosting
                    if let Some(ref ss) = self.shared_state {
                        // Sum the whole set FIRST, then pay once. Paying per
                        // shard truncated every sub-1 GB shard to zero credits
                        // — see `hosting_remainder_micro`.
                        let mut total_gb: f64 = 0.0;
                        let mut shard_count: u32 = 0;
                        for manifest in ss.model_registry.models() {
                            for shard in &manifest.shards {
                                let shard_id = ShardId {
                                    model_id: manifest.id.clone(),
                                    index: shard.index,
                                };
                                let holders = ss.model_registry.shard_holders(&shard_id);
                                if holders.contains(&self.node_id) {
                                    total_gb += shard.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                                    shard_count += 1;
                                }
                            }
                        }
                        let total_earned = match self.earn_shard_hosting_total(total_gb, 1.0).await {
                            Ok(earned) => earned,
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to earn shard hosting credit");
                                0
                            }
                        };
                        if shard_count > 0 {
                            tracing::info!(
                                shards = shard_count,
                                total_gb = format!("{total_gb:.3}"),
                                credits_earned = total_earned,
                                carry_micro = self
                                    .hosting_remainder_micro
                                    .load(std::sync::atomic::Ordering::Relaxed),
                                "Earned shard hosting credits"
                            );
                        }

                        // Earn credits for seeding (serving shard data to peers)
                        let bytes_served = ss.shard_bytes_served.swap(
                            0,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if bytes_served > 0 {
                            let dummy_shard = ShardId {
                                model_id: crate::types::ModelId("__seeding__".into()),
                                index: 0,
                            };
                            match self.earn_shard_seeding(&dummy_shard, bytes_served).await {
                                Ok(earned) if earned > 0 => {
                                    tracing::info!(
                                        bytes_served,
                                        credits_earned = earned,
                                        "Earned shard seeding credits"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to earn shard seeding credit"
                                    );
                                }
                                _ => {}
                            }
                        }

                        // NETWORKING_PLAN Phase 3 — drain app-level inference
                        // relay bytes forwarded and earn at the seeding rate.
                        let relay_fwd_bytes = ss
                            .relay_inference_bytes
                            .swap(0, std::sync::atomic::Ordering::Relaxed);
                        if relay_fwd_bytes > 0 {
                            match self.earn_relay_forwarding(relay_fwd_bytes).await {
                                Ok(earned) if earned > 0 => {
                                    tracing::info!(
                                        bytes_forwarded = relay_fwd_bytes,
                                        credits_earned = earned,
                                        "Earned inference-relay forwarding credits"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to earn relay forwarding credit"
                                    );
                                }
                                _ => {}
                            }
                        }

                        // Drain accumulated relay service seconds and earn credits
                        let relay_secs = ss.relay_seconds_served.swap(
                            0,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if relay_secs > 0 {
                            match self.earn_relay_service(relay_secs).await {
                                Ok(earned) if earned > 0 => {
                                    tracing::info!(
                                        relay_seconds = relay_secs,
                                        credits_earned = earned,
                                        "Earned relay service credits"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to earn relay service credit"
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ = escrow_cleanup_interval.tick() => {
                    // Clean up expired escrows (refund credits for timed-out requests)
                    if let Some(ref ss) = self.shared_state {
                        let cleaned = ss.credits.escrow_manager.cleanup_expired(&self.balance).await;
                        if cleaned > 0 {
                            tracing::info!(
                                cleaned,
                                "Cleaned up expired escrows"
                            );
                        }
                    }
                }
                _ = decay_interval.tick() => {
                    // Walk a negative balance back toward zero so a soft credit
                    // penalty (or bug-induced over-charge) isn't permanent.
                    // Refund semantics: adjust balance, leave lifetime counters alone.
                    let balance_now = { self.balance.read().await.balance };
                    let forgiven = negative_balance_decay_amount(balance_now);
                    if forgiven > 0 {
                        match apply_credit_direct(
                            &self.balance,
                            &self.db,
                            forgiven,
                            CreditDelta::Refund,
                        ).await {
                            Ok(()) => tracing::info!(
                                previous = balance_now,
                                forgiven,
                                new_balance = balance_now.saturating_add(forgiven),
                                "Negative-balance decay applied"
                            ),
                            Err(e) => tracing::warn!(
                                error = %e,
                                forgiven,
                                "Negative-balance decay: DB persist failed, balance unchanged"
                            ),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Apply credit operations directly on shared state (for use outside the CreditLedger task).
///
/// Resolve effective credit rates from shared state: per-pool override > global config.
pub(crate) fn resolve_credit_rates(
    state: &crate::daemon::state::SharedState,
) -> crate::config::CreditRateConfig {
    if let Ok(pool_state) = state.credits.pool_state.try_read() {
        if let Some(ref ps) = *pool_state {
            if let Some(rates) = state.credits.pool_credit_rates.get(&ps.pool_id) {
                return rates.value().clone();
            }
        }
    }
    state.config.pool.credit_rates.clone()
}

/// Kind of credit movement applied by `apply_credit_direct`. Determines
/// which monotonic lifetime counter is updated.
///
/// - `Earning`  → `lifetime_earned += |delta|`
/// - `Spending` → `lifetime_spent  += |delta|`
/// - `Refund`   → neither counter is touched (used when reverting a
///   prior `Spending`, e.g. an escrow refund — `lifetime_spent` must stay
///   monotonic)
#[derive(Debug, Clone, Copy)]
pub enum CreditDelta {
    Earning,
    Spending,
    Refund,
}

/// This replicates what `CreditLedger::apply_credit` + `persist_balance` do, so that
/// callers like `InferenceRouter`, `PipelineExecutor`, and `EscrowManager`
/// don't bypass persistence and proper accounting.
pub async fn apply_credit_direct(
    balance: &Arc<RwLock<CreditBalance>>,
    db: &crate::storage::db::Database,
    delta: i64,
    kind: CreditDelta,
) -> Result<(), SwarmError> {
    apply_credit_direct_noted(balance, db, delta, kind, "unspecified").await
}

/// As [`apply_credit_direct`], but records WHY in the audit log.
///
/// **Every movement of a node's balance goes through here, and until now none
/// of them was written down.** A node reported 205,170 spent and 204,880
/// refunded against zero requests made or served, and there was no way for
/// anyone — its operator or us — to find out what any of it was. The totals
/// reconciled, but that proves only that the arithmetic held: unexplained gaps
/// are attributed to refunds by `backfill_historical_refunds`, so the books
/// close by construction.
///
/// `note` is a short stable tag for the reason, not a message. It is the
/// difference between "205k moved" and "205k of escrow reservations that were
/// released again", which is the whole question an operator is asking.
pub async fn apply_credit_direct_noted(
    balance: &Arc<RwLock<CreditBalance>>,
    db: &crate::storage::db::Database,
    delta: i64,
    kind: CreditDelta,
    note: &str,
) -> Result<(), SwarmError> {
    // Update in-memory balance under write lock, then release before DB write.
    // The small crash window (memory updated, process dies before persist) is
    // acceptable — the same pattern is used by CreditLedger::apply_credit.
    let kind_str = match kind {
        CreditDelta::Earning => "earning",
        CreditDelta::Spending => "spending",
        CreditDelta::Refund => "refund",
    };
    let snapshot = {
        let mut bal = balance.write().await;
        // SEC-I1: saturating arithmetic to prevent overflow
        bal.balance = bal.balance.saturating_add(delta);
        bal.last_updated = chrono::Utc::now();

        match kind {
            CreditDelta::Earning => {
                bal.lifetime_earned = bal.lifetime_earned.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Spending => {
                bal.lifetime_spent = bal.lifetime_spent.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Refund => {
                // Reverting a prior spend. `lifetime_spent` stays monotonic,
                // so record the return separately — otherwise the books cannot
                // be closed from outside and the node looks broken (see
                // `CreditBalance::lifetime_refunded`).
                bal.lifetime_refunded = bal.lifetime_refunded.saturating_add(delta.unsigned_abs());
            }
        }

        tracing::debug!(
            balance = bal.balance,
            delta,
            kind = ?kind,
            lifetime_earned = bal.lifetime_earned,
            lifetime_spent = bal.lifetime_spent,
            "Credit balance updated (direct)"
        );

        bal.clone()
    };
    // Persist outside write lock to avoid blocking inference hot path.
    // On DB failure, REVERT the in-memory mutation so the caller can retry
    // (e.g. via pending_credit_earn restore in ledger persist_interval)
    // without double-counting in memory. Without this, a failed flush plus
    // restored-pending plus successful next flush applies the same delta
    // to in-memory balance twice — the second flush then writes the
    // doubled value to DB, persisting the divergence.
    if let Err(e) = db.put_json(TREE_CREDITS, KEY_BALANCE, &snapshot) {
        let mut bal = balance.write().await;
        bal.balance = bal.balance.saturating_sub(delta);
        match kind {
            CreditDelta::Earning => {
                bal.lifetime_earned = bal.lifetime_earned.saturating_sub(delta.unsigned_abs());
            }
            CreditDelta::Spending => {
                bal.lifetime_spent = bal.lifetime_spent.saturating_sub(delta.unsigned_abs());
            }
            CreditDelta::Refund => {
                // Mirror of the forward path — a reverted refund must give
                // back its counter increment too, or the books stop closing.
                bal.lifetime_refunded = bal.lifetime_refunded.saturating_sub(delta.unsigned_abs());
            }
        }
        tracing::warn!(
            error = %e,
            delta,
            "apply_credit_direct: DB persist failed — reverted in-memory mutation"
        );
        return Err(e);
    }

    // Audit line, written only after the balance is durable so the log never
    // claims a movement that did not persist. Best-effort: a node that cannot
    // write its diagnostic log must still be able to transact.
    let _ = db.append_credit_log(&serde_json::json!({
        "seq": chrono::Utc::now().timestamp_millis(),
        "at": chrono::Utc::now().to_rfc3339(),
        "delta": delta,
        "kind": kind_str,
        "note": note,
        "balance_after": snapshot.balance,
    }));

    Ok(())
}

/// Maximum staleness for a signed balance report (5 minutes). Shared with
/// the credit-transaction freshness check in `daemon::dispatch` (every signed
/// credit-typed gossip uses the same window per gotcha #32 / #44).
pub(crate) const BALANCE_REPORT_MAX_AGE_SECS: i64 = 300;

/// Allowable clock skew tolerance for a balance report timestamped in the
/// future. Honest cross-node clocks drift by single-digit seconds; anything
/// larger is rejected so an attacker can't pre-sign with a future timestamp
/// to extend the effective replay window.
pub(crate) const CLOCK_SKEW_TOLERANCE_SECS: i64 = 30;

/// One-sided staleness check shared across signed credit-typed messages
/// (balance reports, credit transactions, future signed types). Returns
/// `Ok(())` if `timestamp` is within `[now - max_age_secs, now + skew_secs]`,
/// `Err(SwarmError::CreditError)` otherwise. Per gotcha #32 / #44 the future
/// side MUST be one-sided — `(now - ts).abs() > MAX` doubles the effective
/// replay window. Centralising here pins the invariant.
pub(crate) fn check_signed_freshness(
    timestamp: chrono::DateTime<chrono::Utc>,
    skew_secs: i64,
    max_age_secs: i64,
    kind: &'static str,
) -> Result<(), SwarmError> {
    let age_secs = (chrono::Utc::now() - timestamp).num_seconds();
    // SEC: `chrono::Duration::num_seconds()` returns `i64::MIN` on overflow
    // when `timestamp` is e.g. `DateTime::<Utc>::MAX_UTC` (year 262143 CE,
    // which serde_json will happily round-trip from a wire field). `-i64::MIN`
    // is signed-negation overflow — panic in debug, wrap to MIN in release.
    // `saturating_neg` clamps to `i64::MAX` for the formatted message.
    if age_secs < -skew_secs {
        return Err(SwarmError::CreditError(format!(
            "Future-dated {kind}: {}s ahead (skew tolerance {skew_secs}s)",
            age_secs.saturating_neg(),
        )));
    }
    if age_secs > max_age_secs {
        return Err(SwarmError::CreditError(format!(
            "Stale {kind}: {age_secs}s old (max {max_age_secs}s)"
        )));
    }
    Ok(())
}

/// Build the deterministic signing payload for a balance report.
/// Format: "swarmllm-balance-v1" || node_id(32) || balance_bucket(8) || timestamp_secs(8)
fn build_balance_report_payload(
    node_id: &NodeId,
    balance_bucket: i64,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(56);
    payload.extend_from_slice(b"swarmllm-balance-v1");
    payload.extend_from_slice(&node_id.0);
    payload.extend_from_slice(&balance_bucket.to_le_bytes());
    payload.extend_from_slice(&timestamp.timestamp().to_le_bytes());
    payload
}

/// Sign a balance report with the node's Ed25519 identity.
pub fn sign_balance_report(
    node_id: &NodeId,
    balance_bucket: i64,
    timestamp: chrono::DateTime<chrono::Utc>,
    identity: &crate::identity::Identity,
) -> Vec<u8> {
    let payload = build_balance_report_payload(node_id, balance_bucket, timestamp);
    identity.sign(&payload)
}

/// Verify a signed balance report.
///
/// Checks:
/// 1. Signature is valid Ed25519 over the canonical payload
/// 2. The signing key matches the claimed node_id (node_id == verifying_key bytes)
/// 3. Timestamp is within `BALANCE_REPORT_MAX_AGE_SECS` of `now`
///
/// Returns `Ok(())` for valid signed reports, `Err` for invalid/unsigned/stale.
pub fn verify_balance_report(gossip: &CreditGossip) -> Result<(), SwarmError> {
    // Reject unsigned reports
    if gossip.signature.is_empty() {
        return Err(SwarmError::InvalidSignature);
    }

    check_signed_freshness(
        gossip.timestamp,
        CLOCK_SKEW_TOLERANCE_SECS,
        BALANCE_REPORT_MAX_AGE_SECS,
        "balance report",
    )?;

    // Verify Ed25519 signature
    let verifying_key =
        VerifyingKey::from_bytes(&gossip.node_id.0).map_err(|_| SwarmError::InvalidSignature)?;

    let payload =
        build_balance_report_payload(&gossip.node_id, gossip.balance_bucket, gossip.timestamp);

    crate::crypto::verify_ed25519_sig(&gossip.signature, &payload, &verifying_key)?;

    Ok(())
}

/// Process a received balance gossip message with Sybil resistance.
///
/// Only signed reports are accepted. Invalid signatures, unsigned reports,
/// and stale reports are rejected entirely.
///
/// Deduplicates by node_id: each peer gets exactly one entry in the rolling window.
/// This prevents a single peer from dominating the percentile distribution by
/// re-gossiping frequently (Sybil percentile stuffing).
pub async fn process_balance_gossip(
    peer_balances: &Arc<ArcSwap<Vec<i64>>>,
    gossip: &CreditGossip,
    peer_balance_map: Option<&DashMap<NodeId, i64>>,
) {
    // Reject implausible balance buckets.
    // SEC: `.abs()` panics in debug builds on `i64::MIN`. A peer can send
    // `balance_bucket: -9_223_372_036_854_775_808` over GossipSub — and this
    // check runs BEFORE signature verification, so it's pre-auth. Use
    // `saturating_abs()` (clamps to `i64::MAX`) to never panic.
    const MAX_PLAUSIBLE_BALANCE: i64 = 100_000_000;
    if gossip.balance_bucket.saturating_abs() > MAX_PLAUSIBLE_BALANCE {
        tracing::debug!(
            balance_bucket = gossip.balance_bucket,
            "Ignoring implausible peer balance gossip"
        );
        return;
    }

    match verify_balance_report(gossip) {
        Ok(()) => {
            // If a peer_balance_map is provided, use it for deduplication.
            // The Vec snapshot is rebuilt from the map's values for
            // percentile estimation and swapped atomically via ArcSwap —
            // readers on the inference hot path never block.
            if let Some(map) = peer_balance_map {
                // Cap the map to prevent unbounded growth from departed peers
                const MAX_PEERS: usize = 10_000;
                if map.len() < MAX_PEERS || map.contains_key(&gossip.node_id) {
                    map.insert(gossip.node_id.clone(), gossip.balance_bucket);
                }

                let new_values: Vec<i64> = map.iter().map(|e| *e.value()).collect();
                peer_balances.store(Arc::new(new_values));
            } else {
                // Fallback: raw push (used in tests without SharedState).
                // ArcSwap has no read-modify-write primitive, so do the
                // update by load + clone + store. Contention is a non-
                // issue in this test-only branch.
                const MAX_BALANCE_VEC_PEERS: usize = 1000;
                let prev = peer_balances.load_full();
                let mut balances: Vec<i64> = (*prev).clone();
                balances.push(gossip.balance_bucket);
                if balances.len() > MAX_BALANCE_VEC_PEERS {
                    let excess = balances.len() - MAX_BALANCE_VEC_PEERS;
                    balances.drain(..excess);
                }
                peer_balances.store(Arc::new(balances));
            }

            tracing::debug!(
                peer = %gossip.node_id,
                bucket = gossip.balance_bucket,
                "Processed signed credit gossip"
            );
        }
        Err(e) => {
            tracing::warn!(
                peer = %gossip.node_id,
                error = %e,
                "Rejected invalid balance report"
            );
        }
    }
}

/// Bucket a balance value for privacy-preserving gossip.
/// Rounds toward negative infinity (floor division) to nearest 100.
/// This ensures negative balances near zero are not confused with positive.
fn bucket_balance(balance: i64) -> i64 {
    balance.div_euclid(100) * 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_balance_rounds_down() {
        assert_eq!(bucket_balance(0), 0);
        assert_eq!(bucket_balance(99), 0);
        assert_eq!(bucket_balance(100), 100);
        assert_eq!(bucket_balance(150), 100);
        assert_eq!(bucket_balance(250), 200);
        assert_eq!(bucket_balance(-150), -200);
        assert_eq!(bucket_balance(-50), -100);
    }

    #[tokio::test]
    async fn earn_inference_increases_balance() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([1u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 0,
            lifetime_earned: 0,
            lifetime_spent: 0,
            lifetime_refunded: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        let earned = ledger
            .earn_inference(uuid::Uuid::new_v4(), 10)
            .await
            .unwrap();

        // 10 credits/token * 10 tokens = 100
        assert_eq!(earned, 100);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 100);
        assert_eq!(bal.lifetime_earned, 100);
        assert_eq!(bal.lifetime_spent, 0);
    }

    #[tokio::test]
    async fn spend_inference_decreases_balance() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([2u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 1000,
            lifetime_earned: 1000,
            lifetime_spent: 0,
            lifetime_refunded: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        let spent = ledger
            .spend_inference(uuid::Uuid::new_v4(), 5)
            .await
            .unwrap();

        // 10 credits/token * 5 tokens = 50
        assert_eq!(spent, 50);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 950); // 1000 - 50
        assert_eq!(bal.lifetime_spent, 50);
    }

    #[tokio::test]
    async fn percentile_estimation() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([4u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 500,
            lifetime_earned: 500,
            lifetime_spent: 0,
            lifetime_refunded: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        // No peers: default percentile
        let tier = ledger.calculate_tier().await;
        assert_eq!(tier, PriorityTier::Silver); // balance > 0, percentile 0.5

        // Add peer balances: our 500 should be above most
        ledger.peer_balances().store(Arc::new(vec![
            100, 200, 300, 150, 250, 50, 400, 350, 180, 220,
        ]));

        let tier = ledger.calculate_tier().await;
        // 500 > all 10 peers, percentile = 1.0, so Platinum
        assert_eq!(tier, PriorityTier::Platinum);
    }

    #[tokio::test]
    async fn balance_persists_to_db() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([5u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 0,
            lifetime_earned: 0,
            lifetime_spent: 0,
            lifetime_refunded: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id.clone(),
            balance,
            db.clone(),
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        ledger
            .earn_inference(uuid::Uuid::new_v4(), 10)
            .await
            .unwrap();

        // Check it was persisted
        let stored: CreditBalance = db.get_json(TREE_CREDITS, KEY_BALANCE).unwrap().unwrap();
        assert_eq!(stored.balance, 100); // 10 credits/token * 10 tokens
        assert_eq!(stored.node_id, node_id);
    }

    // ---- Signed balance report tests ----

    #[test]
    fn sign_and_verify_balance_report() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let bucket = 500;
        let timestamp = chrono::Utc::now();

        let signature = sign_balance_report(&node_id, bucket, timestamp, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: bucket,
            timestamp,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_ok());
    }

    #[test]
    fn unsigned_report_rejected() {
        let identity = crate::identity::Identity::generate();
        let gossip = CreditGossip {
            node_id: identity.node_id().clone(),
            balance_bucket: 300,
            timestamp: chrono::Utc::now(),
            signature: Vec::new(), // unsigned
        };

        assert!(verify_balance_report(&gossip).is_err());
    }

    #[test]
    fn wrong_signer_rejected() {
        let real_identity = crate::identity::Identity::generate();
        let imposter = crate::identity::Identity::generate();
        let bucket = 500;
        let timestamp = chrono::Utc::now();

        // Imposter signs but claims to be real_identity
        let signature = sign_balance_report(real_identity.node_id(), bucket, timestamp, &imposter);

        let gossip = CreditGossip {
            node_id: real_identity.node_id().clone(),
            balance_bucket: bucket,
            timestamp,
            signature,
        };

        let result = verify_balance_report(&gossip);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_balance_rejected() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();

        let signature = sign_balance_report(&node_id, 500, timestamp, &identity);

        // Tamper: change the balance_bucket
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 999,
            timestamp,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_err());
    }

    #[test]
    fn stale_report_rejected() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        // Timestamp 10 minutes ago — beyond the 5 min window
        let old_timestamp = chrono::Utc::now() - chrono::Duration::seconds(600);

        let signature = sign_balance_report(&node_id, 500, old_timestamp, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: old_timestamp,
            signature,
        };

        let result = verify_balance_report(&gossip);
        assert!(result.is_err());
    }

    #[test]
    fn fresh_report_accepted() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        // Timestamp 2 minutes ago — within the 5 min window
        let recent = chrono::Utc::now() - chrono::Duration::seconds(120);

        let signature = sign_balance_report(&node_id, 500, recent, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: recent,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_ok());
    }

    #[test]
    fn future_dated_report_rejected() {
        // Pre-signed reports with a timestamp in the future would extend the
        // effective replay window if `.abs()` was used on the staleness
        // check. R65 fixed this with a one-sided bound + 30s clock-skew
        // tolerance — verify a report 60s in the future (well beyond
        // tolerance) is rejected.
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let future = chrono::Utc::now() + chrono::Duration::seconds(60);
        let signature = sign_balance_report(&node_id, 500, future, &identity);
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: future,
            signature,
        };
        assert!(
            verify_balance_report(&gossip).is_err(),
            "Future-dated balance report should be rejected"
        );
    }

    #[test]
    fn slightly_future_report_accepted_within_skew_tolerance() {
        // Honest cross-node clock skew of 5s should still accept the report.
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let slight_future = chrono::Utc::now() + chrono::Duration::seconds(5);
        let signature = sign_balance_report(&node_id, 500, slight_future, &identity);
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: slight_future,
            signature,
        };
        assert!(
            verify_balance_report(&gossip).is_ok(),
            "Report 5s in future (within 30s skew tolerance) should be accepted"
        );
    }

    #[tokio::test]
    async fn process_signed_gossip_adds_balance() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 500, timestamp, &identity);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 1);
        assert_eq!(balances[0], 500);
    }

    #[tokio::test]
    async fn process_invalid_gossip_rejected() {
        let identity = crate::identity::Identity::generate();
        let imposter = crate::identity::Identity::generate();
        let timestamp = chrono::Utc::now();

        // Imposter signs a report claiming to be identity
        let signature = sign_balance_report(identity.node_id(), 500, timestamp, &imposter);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id: identity.node_id().clone(),
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        // Should have been rejected — no balance added
        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 0);
    }

    #[tokio::test]
    async fn process_implausible_balance_ignored() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 200_000_000, timestamp, &identity);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 200_000_000,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 0);
    }

    // --- Negative-balance decay (external report follow-up, 2026-07-23) ---

    #[test]
    fn decay_leaves_non_negative_balances_untouched() {
        assert_eq!(negative_balance_decay_amount(0), 0);
        assert_eq!(negative_balance_decay_amount(1), 0);
        assert_eq!(negative_balance_decay_amount(1_000_000), 0);
    }

    #[test]
    fn decay_uses_the_flat_floor_for_small_deficits() {
        // 5% of these is below the 500 floor, so the floor applies…
        assert_eq!(
            negative_balance_decay_amount(-9_000),
            NEGATIVE_BALANCE_DECAY_FLOOR
        );
        assert_eq!(
            negative_balance_decay_amount(-2_000),
            NEGATIVE_BALANCE_DECAY_FLOOR
        );
        // …but never more than the deficit itself (lands exactly on zero).
        assert_eq!(negative_balance_decay_amount(-300), 300);
        assert_eq!(negative_balance_decay_amount(-1), 1);
    }

    #[test]
    fn decay_uses_the_percentage_for_large_deficits() {
        // 5% of 41,400 = 2,070, above the floor.
        assert_eq!(negative_balance_decay_amount(-41_400), 2_070);
        // 5% of 100,000 = 5,000.
        assert_eq!(negative_balance_decay_amount(-100_000), 5_000);
    }

    #[test]
    fn decay_never_overshoots_zero() {
        // Repeatedly applying decay walks toward zero and stops there, never
        // crossing into a positive (minted) balance.
        let mut balance: i64 = -41_400;
        for _ in 0..1_000 {
            let forgiven = negative_balance_decay_amount(balance);
            if forgiven == 0 {
                break;
            }
            assert!(forgiven > 0 && forgiven <= -balance);
            balance += forgiven;
            assert!(balance <= 0, "decay must never push the balance positive");
        }
        assert_eq!(balance, 0, "decay converges exactly to zero");
    }

    #[test]
    fn decay_handles_i64_min_without_panicking() {
        // saturating_neg + f64 path must not overflow on the extreme.
        let forgiven = negative_balance_decay_amount(i64::MIN);
        assert!(forgiven > 0);
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::backfill_historical_refunds;
    use crate::types::{CreditBalance, NodeId};

    fn bal(balance: i64, earned: u64, spent: u64, refunded: u64) -> CreditBalance {
        CreditBalance {
            node_id: NodeId([3u8; 32]),
            balance,
            lifetime_earned: earned,
            lifetime_spent: spent,
            lifetime_refunded: refunded,
            last_updated: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
        }
    }

    /// Both live test nodes immediately after the field shipped: a real gap,
    /// zero recorded refunds, `books_balance` false for ever without this.
    #[test]
    fn a_pre_existing_gap_is_attributed_and_the_books_close() {
        for (b, e, s, expect) in [
            (850_467i64, 881_047u64, 31_230u64, 650u64),
            (1_772_870, 1_809_850, 38_660, 1_680),
            // A tester's node, reported 2026-07-30: the ~905k gap their two
            // separate reports both cite. `lifetime_refunded` read 0 after
            // updating despite weeks of deliberately-provoked failures, and
            // they correctly concluded the fix was forward-only.
            (220_135, 260_350, 945_340, 905_125),
        ] {
            let mut cb = bal(b, e, s, 0);
            assert!(!cb.books_balance(), "precondition: must not close yet");
            backfill_historical_refunds(&mut cb);
            assert_eq!(cb.lifetime_refunded, expect);
            assert!(cb.books_balance(), "must reconcile after backfill");
        }
    }

    /// Idempotent: a second load must not double-count.
    #[test]
    fn backfill_is_idempotent() {
        let mut cb = bal(850_467, 881_047, 31_230, 0);
        backfill_historical_refunds(&mut cb);
        let once = cb.lifetime_refunded;
        backfill_historical_refunds(&mut cb);
        assert_eq!(cb.lifetime_refunded, once);
    }

    /// A node whose books already close must never be rewritten — note this
    /// holds because its gap is 0, NOT because its refund counter is non-zero.
    #[test]
    fn a_node_with_real_refunds_is_untouched() {
        let mut cb = bal(146_065, 174_330, 933_290, 905_025);
        assert!(cb.books_balance(), "precondition: already reconciles");
        backfill_historical_refunds(&mut cb);
        assert_eq!(cb.lifetime_refunded, 905_025);
    }

    /// The case the first version stranded permanently, reported from a live
    /// node 2026-07-31. It had recorded a real 640-credit refund before the
    /// migration ran, so keying off `lifetime_refunded == 0` skipped it and
    /// left a ~905k historical gap unexplained for ever. Any node that took a
    /// single refund between the counter shipping and this migration running
    /// was in the same position.
    #[test]
    fn a_gap_is_still_attributed_when_some_refunds_are_already_recorded() {
        let mut cb = bal(216_702, 263_517, 952_580, 640);
        assert!(!cb.books_balance(), "precondition: the reported gap");

        backfill_historical_refunds(&mut cb);

        // The pre-existing 640 is preserved and the remaining gap added to it,
        // rather than the counter being overwritten.
        assert_eq!(cb.lifetime_refunded, 640 + 905_125);
        assert!(cb.books_balance(), "must reconcile after backfill");
    }

    /// Idempotent for that case too — the generalised form drives the gap to
    /// zero, so a second load is a no-op without needing a "have I run?" flag.
    #[test]
    fn backfill_is_idempotent_with_pre_existing_refunds() {
        let mut cb = bal(216_702, 263_517, 952_580, 640);
        backfill_historical_refunds(&mut cb);
        let once = cb.lifetime_refunded;
        backfill_historical_refunds(&mut cb);
        assert_eq!(cb.lifetime_refunded, once);
    }

    /// A refund recorded correctly raises `balance` and `lifetime_refunded`
    /// together, so it cancels out of the gap. Nothing to attribute, and in
    /// particular nothing double-counted.
    #[test]
    fn a_correctly_recorded_refund_is_not_double_counted() {
        let mut cb = bal(850_467, 881_047, 31_230, 0);
        backfill_historical_refunds(&mut cb);
        let after_migration = cb.lifetime_refunded;

        // A 100-credit refund lands: balance and the counter both move.
        cb.balance += 100;
        cb.lifetime_refunded += 100;
        assert!(cb.books_balance());

        backfill_historical_refunds(&mut cb);
        assert_eq!(
            cb.lifetime_refunded,
            after_migration + 100,
            "a recorded refund must not be attributed a second time"
        );
    }

    /// A fresh node has nothing to attribute.
    #[test]
    fn a_fresh_node_is_untouched() {
        let mut cb = bal(0, 0, 0, 0);
        backfill_historical_refunds(&mut cb);
        assert_eq!(cb.lifetime_refunded, 0);
        assert!(cb.books_balance());
    }

    /// A gap in the OTHER direction cannot be refunds. Leave it visible rather
    /// than inventing a number that hides a real inconsistency.
    #[test]
    fn a_negative_gap_is_left_alone() {
        let mut cb = bal(100, 500, 100, 0);
        backfill_historical_refunds(&mut cb);
        assert_eq!(cb.lifetime_refunded, 0);
        assert!(!cb.books_balance(), "the inconsistency must stay visible");
    }
}

#[cfg(test)]
mod hosting_accrual_tests {
    /// Pure re-implementation of the accrual arithmetic in
    /// `earn_shard_hosting_total`, so the rounding rule can be exercised
    /// without standing up a ledger, a database and a network channel.
    fn accrue(rate: i64, total_gb: f64, hours: f32, carry: &mut i64) -> i64 {
        let raw = rate as f64 * total_gb * hours as f64;
        if !raw.is_finite() || raw < 0.0 {
            return 0;
        }
        let micro = (raw.min(1_000_000.0) * 1_000_000.0).round() as i64;
        let total = micro.saturating_add(*carry);
        let amount = total / 1_000_000;
        *carry = total % 1_000_000;
        amount
    }

    /// The reported defect, with the shipped defaults: `shard_size_mb = 512`
    /// and `shard_hosting = 1`. Paid per shard, `1 * 0.5 * 1.0` truncated to
    /// zero — on every node, for ever.
    #[test]
    fn a_single_half_gigabyte_shard_no_longer_rounds_away_forever() {
        let mut carry = 0i64;
        // Per-shard truncation would give 0 here and 0 on every later tick.
        assert_eq!(accrue(1, 0.5, 1.0, &mut carry), 0, "first hour is still 0");
        assert_eq!(
            accrue(1, 0.5, 1.0, &mut carry),
            1,
            "but the carry pays out on the second hour instead of vanishing"
        );
    }

    /// The operator's actual case: 13 shards at 512 MB. Per-shard truncation
    /// paid 0; summing first pays for the 6.5 GB actually hosted.
    #[test]
    fn thirteen_half_gigabyte_shards_pay() {
        let mut carry = 0i64;
        let earned = accrue(1, 13.0 * 0.5, 1.0, &mut carry);
        assert_eq!(earned, 6);
        assert_eq!(carry, 500_000, "half a credit carried, not discarded");
    }

    /// Over many ticks the total must track the exact rate with no drift —
    /// the reason for integer micro-credits rather than an f64 accumulator.
    #[test]
    fn carry_does_not_drift_over_many_ticks() {
        let mut carry = 0i64;
        let mut total = 0i64;
        for _ in 0..100 {
            total += accrue(1, 0.5, 1.0, &mut carry);
        }
        assert_eq!(total, 50, "100 hours at 0.5 credits/hour is exactly 50");
    }

    /// Hosting nothing pays nothing and must not disturb the carry.
    #[test]
    fn zero_shards_pays_zero() {
        let mut carry = 250_000i64;
        assert_eq!(accrue(1, 0.0, 1.0, &mut carry), 0);
        assert_eq!(carry, 250_000);
    }

    /// Degenerate rates must not panic or mint absurd credit.
    #[test]
    fn nonsense_inputs_are_inert() {
        let mut carry = 0i64;
        assert_eq!(accrue(1, f64::NAN, 1.0, &mut carry), 0);
        assert_eq!(accrue(1, -5.0, 1.0, &mut carry), 0);
        assert_eq!(accrue(1, f64::INFINITY, 1.0, &mut carry), 0);
        // Enormous but finite: capped, never overflowing.
        let huge = accrue(i64::MAX, 1e9, 1.0, &mut carry);
        assert!((0..=1_000_000).contains(&huge), "capped, got {huge}");
    }
}
