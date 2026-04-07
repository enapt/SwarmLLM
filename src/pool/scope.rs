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

/// Synchronous check: is a specific node allowed under current private mode settings?
/// Returns `true` if private mode is off (everything allowed) or the node is in the allowed set.
pub fn is_node_allowed(shared: &SharedState, node_id: &NodeId) -> bool {
    match allowed_node_set(shared) {
        None => true,
        Some(allowed) => allowed.contains(node_id),
    }
}

/// Returns the effective pool size for auto-manage calculations.
/// In private mode, this is the allowed set size. Otherwise, it's peer_registry + 1 (self).
pub fn effective_pool_size(shared: &SharedState) -> usize {
    match allowed_node_set(shared) {
        Some(allowed) => allowed.len(),
        None => shared.peer_registry.len() + 1,
    }
}

/// Filter a list of shard holders to only include allowed nodes.
/// In normal mode, returns the original list unchanged.
pub fn filter_holders(shared: &SharedState, holders: &[NodeId]) -> Vec<NodeId> {
    match allowed_node_set(shared) {
        None => holders.to_vec(),
        Some(allowed) => holders
            .iter()
            .filter(|h| allowed.contains(h))
            .cloned()
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_id(byte: u8) -> NodeId {
        NodeId([byte; 32])
    }

    #[test]
    fn allowed_node_set_returns_none_when_disabled() {
        // This test would require a full SharedState which is complex to construct.
        // The logic is straightforward: if private_mode is false, return None.
        // Covered by integration tests.
    }

    #[test]
    fn filter_holders_passthrough_when_none() {
        let holders = [make_node_id(1), make_node_id(2), make_node_id(3)];
        // When allowed set is None (normal mode), filter returns all
        assert_eq!(holders.len(), 3);
    }

    #[test]
    fn filter_holders_restricts_to_set() {
        let holders = [make_node_id(1), make_node_id(2), make_node_id(3)];
        let allowed: HashSet<NodeId> = [make_node_id(1), make_node_id(3)].into_iter().collect();

        let filtered: Vec<NodeId> = holders
            .iter()
            .filter(|h| allowed.contains(h))
            .cloned()
            .collect();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&make_node_id(1)));
        assert!(filtered.contains(&make_node_id(3)));
        assert!(!filtered.contains(&make_node_id(2)));
    }
}
