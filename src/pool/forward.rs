use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::pool::crypto;
use crate::pool::types::PoolCommand;

/// Forward earned credits to the pool owner.
///
/// Called from the credit earning path when this node is a pool member (not owner).
/// Creates a member-signed PoolCreditForward and sends it to the pool manager.
pub async fn forward_credits_to_owner(
    shared_state: &Arc<SharedState>,
    amount: i64,
) -> Result<(), SwarmError> {
    if amount <= 0 {
        return Ok(());
    }

    // Extract pool info while holding the read lock, then release it.
    let (pool_id, owner_id) = {
        let guard = shared_state.pool_state.read().await;
        let ps = match guard.as_ref() {
            Some(ps) => ps,
            None => return Ok(()), // Not in a pool
        };

        let my_id = shared_state.identity.node_id();

        // Only forward if we're a member (not the owner)
        if ps.pool_id == *my_id {
            return Ok(()); // Owner keeps their own credits
        }

        (ps.pool_id.clone(), ps.pool_id.clone())
    };

    // Create member-signed credit forward
    let my_id = shared_state.identity.node_id();
    let forward = crypto::create_credit_forward(
        &shared_state.identity,
        &pool_id,
        my_id,
        &owner_id,
        amount,
    );

    // Send to pool manager for processing + broadcasting
    if let Some(ref tx) = *shared_state.pool_tx.read().await {
        let _ = tx.send(PoolCommand::ProcessCreditForward { forward }).await;
    }

    Ok(())
}
