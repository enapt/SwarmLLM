use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::pool::crypto;
use crate::pool::types::PoolCommand;

/// Forward earned credits to the pool owner, honouring the pool's
/// `member_credit_split_pct`.
///
/// Called from the credit earning path when this node is a pool member
/// (not owner). The member keeps `member_credit_split_pct` percent of
/// `amount`; the remainder is forwarded to the owner via a member-signed
/// `PoolCreditForward`.
///
/// Returns `Ok(member_keeps)` — the credits the caller should still apply
/// to its own balance:
/// - `member_keeps == amount`: not in a pool / is the owner / forwarding
///   declined (caller credits the full amount locally)
/// - `0 < member_keeps < amount`: partial split — caller credits the
///   `member_keeps` share and the rest was forwarded
/// - `member_keeps == 0`: 100% forwarded to owner
///
/// This is a behaviour change from the legacy `Ok(bool)` contract, which
/// silently routed 100% to the owner regardless of the configured split.
pub async fn forward_credits_to_owner(
    shared_state: &Arc<SharedState>,
    amount: i64,
) -> Result<i64, SwarmError> {
    if amount <= 0 {
        return Ok(amount);
    }

    // Extract pool info while holding the read lock, then release it.
    let (pool_id, owner_id, forward_amount, member_keeps) = {
        let guard = shared_state.credits.pool_state.read().await;
        let ps = match guard.as_ref() {
            Some(ps) => ps,
            None => return Ok(amount), // Not in a pool — caller credits full amount
        };

        let my_id = shared_state.identity.node_id();

        // Only forward if we're a member (not the owner)
        if ps.pool_id == *my_id {
            return Ok(amount); // Owner keeps their own credits
        }

        // SEC: honour `member_credit_split_pct` from PoolState. Earlier code
        // hard-coded 100% forwarding, silently overriding the split the pool
        // owner advertised in gossip and on the dashboard.
        let split_pct = ps.member_credit_split_pct.clamp(0, 100) as i64;
        let member_keeps = amount.saturating_mul(split_pct) / 100;
        let forward = amount.saturating_sub(member_keeps);
        if forward <= 0 {
            // Member keeps 100% — caller credits the full amount locally.
            return Ok(amount);
        }

        // pool_id is the owner's NodeId by design — credit forwards go to the owner
        let id = ps.pool_id.clone();
        (id.clone(), id, forward, member_keeps)
    };

    let amount = forward_amount;

    // Create member-signed credit forward
    let my_id = shared_state.identity.node_id();
    tracing::debug!(
        pool = %pool_id,
        amount,
        from = %my_id,
        to = %owner_id,
        "DIAG: forwarding credits to pool owner"
    );
    let forward =
        crypto::create_credit_forward(&shared_state.identity, &pool_id, my_id, &owner_id, amount);

    // Send to pool manager for processing + broadcasting. Clone the
    // sender out of the RwLock guard before awaiting on `tx.send` so a
    // concurrent PoolManager start/stop (which takes the write lock)
    // doesn't stall behind channel backpressure on this send.
    let pool_tx = shared_state.credits.pool_tx.read().await.clone();
    if let Some(tx) = pool_tx {
        if tx
            .send(PoolCommand::ProcessCreditForward { forward })
            .await
            .is_err()
        {
            tracing::warn!(
                subsystem = "pool",
                "forward channel unavailable — keeping credits locally"
            );
            // On forward failure, caller should credit the FULL original
            // amount locally (forward_amount + member_keeps) — owner won't
            // receive their share, but the member shouldn't lose theirs too.
            return Ok(member_keeps.saturating_add(forward_amount));
        }
    } else {
        tracing::warn!(
            subsystem = "pool",
            "pool manager not running — keeping credits locally"
        );
        return Ok(member_keeps.saturating_add(forward_amount));
    }

    // Forward dispatched. Caller credits only the member's share locally;
    // the owner's share lands when their pool manager processes the forward.
    Ok(member_keeps)
}
