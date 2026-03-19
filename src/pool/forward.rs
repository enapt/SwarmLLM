use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::pool::crypto;
use crate::pool::types::PoolCommand;

/// Forward earned credits to the pool owner.
///
/// Called from the credit earning path when this node is a pool member (not owner).
/// Creates a member-signed PoolCreditForward and sends it to the pool manager.
/// Respects the pool's `member_credit_split_pct` — member keeps that percentage,
/// forwards the remainder to the owner.
/// Returns `Ok(true)` if credits were forwarded, `Ok(false)` if not in a pool or is owner.
pub async fn forward_credits_to_owner(
    shared_state: &Arc<SharedState>,
    amount: i64,
) -> Result<bool, SwarmError> {
    if amount <= 0 {
        return Ok(false);
    }

    // Extract pool info while holding the read lock, then release it.
    let (pool_id, owner_id, forward_amount) = {
        let guard = shared_state.pool_state.read().await;
        let ps = match guard.as_ref() {
            Some(ps) => ps,
            None => return Ok(false), // Not in a pool
        };

        let my_id = shared_state.identity.node_id();

        // Only forward if we're a member (not the owner)
        if ps.pool_id == *my_id {
            return Ok(false); // Owner keeps their own credits
        }

        // Forward 100% of member earnings to the owner (main device).
        // This is a personal multi-device feature — all devices belong to the same
        // user, so all credits consolidate on the main device's balance.
        let forward = amount;
        if forward <= 0 {
            return Ok(false); // Member keeps 100%
        }

        // pool_id is the owner's NodeId by design — credit forwards go to the owner
        let id = ps.pool_id.clone();
        (id.clone(), id, forward)
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

    // Send to pool manager for processing + broadcasting
    if let Some(ref tx) = *shared_state.pool_tx.read().await {
        if tx
            .send(PoolCommand::ProcessCreditForward { forward })
            .await
            .is_err()
        {
            tracing::warn!("Pool forward channel unavailable — keeping credits locally");
            return Ok(false);
        }
    } else {
        tracing::warn!("Pool manager not running — keeping credits locally");
        return Ok(false);
    }

    Ok(true)
}
