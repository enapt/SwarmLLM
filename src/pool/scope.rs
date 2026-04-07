use std::collections::HashSet;
use std::sync::atomic::Ordering::Relaxed;

use crate::daemon::state::SharedState;
use crate::types::NodeId;

/// Returns the set of NodeIds allowed for inference and shard management.
///
/// - `None` → unrestricted (normal swarm mode, all peers allowed)
/// - `Some(set)` → only these nodes may participate (private mode)
///
/// The allowed set always includes the local node. When private mode is on,
/// it includes pool members and optionally LAN peers (mDNS-discovered or low-latency).
pub fn allowed_node_set(shared: &SharedState) -> Option<HashSet<NodeId>> {
    if !shared.credits.private_mode.load(Relaxed) {
        return None;
    }

    let mut allowed = HashSet::new();

    // Always include ourselves
    allowed.insert(shared.identity.node_id().clone());

    // Add all pool members
    // Use try_read to avoid blocking in sync contexts; fall back to just self if contended
    if let Ok(guard) = shared.credits.pool_state.try_read() {
        if let Some(ref ps) = *guard {
            for m in &ps.members {
                allowed.insert(m.node_id.clone());
            }
        }
    }

    // Optionally include LAN peers
    if shared.config.pool.private_mode_allow_lan {
        for entry in shared.peer_registry.iter() {
            if entry.value().is_lan_peer {
                allowed.insert(entry.key().clone());
            }
        }
    }

    Some(allowed)
}

/// Returns the effective pool size for auto-manage calculations.
/// In private mode, this is the allowed set size. Otherwise, it's peer_registry + 1 (self).
pub fn effective_pool_size(shared: &SharedState) -> usize {
    match allowed_node_set(shared) {
        Some(allowed) => allowed.len(),
        None => shared.peer_registry.len() + 1,
    }
}
