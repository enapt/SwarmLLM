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

    // Add all pool members.
    // `try_read` avoids blocking in sync callers; on contention we'd silently
    // fall back to the self-only set (too-restrictive, not a bypass) so at
    // least log it so recurring contention is visible.
    match shared.credits.pool_state.try_read() {
        Ok(guard) => {
            if let Some(ref ps) = *guard {
                for m in &ps.members {
                    allowed.insert(m.node_id.clone());
                }
            }
        }
        Err(_) => {
            tracing::warn!(
                "pool_state contended during allowed_node_set() — falling back to self-only; \
                 inference scope may be briefly too restrictive"
            );
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

/// Filter a holder list to the allowed set (private mode). Returns the input
/// unchanged when `allowed_set` is `None`. Used by auto-manage scoring and
/// pruning so both paths compute replica counts against the same node set.
pub fn filter_allowed_holders(
    holders: Vec<NodeId>,
    allowed_set: &Option<HashSet<NodeId>>,
) -> Vec<NodeId> {
    match allowed_set {
        Some(allowed) => holders
            .into_iter()
            .filter(|h| allowed.contains(h))
            .collect(),
        None => holders,
    }
}

/// Count holders that fall within the allowed set without allocating.
/// Returns the full slice length when `allowed_set` is `None`.
pub fn count_allowed_holders(holders: &[NodeId], allowed_set: &Option<HashSet<NodeId>>) -> usize {
    match allowed_set {
        Some(allowed) => holders.iter().filter(|h| allowed.contains(h)).count(),
        None => holders.len(),
    }
}

/// R134.7: opt-in cross-pool routing. When `private_mode` AND
/// `allow_cross_pool_inference` are both on AND no member of the local
/// pool holds any shard of `model_id`, returns the set of NodeIds in
/// foreign pools that have advertised this model via the
/// `foreign_pool_catalog`. The scheduler unions this with the local
/// `allowed_node_set` so cross-pool inference can fall through when
/// (and only when) the local pool genuinely can't serve.
///
/// Returns an empty set when:
/// - private_mode is off (allowed_node_set is None — global routing
///   already applies, no fallback needed);
/// - allow_cross_pool_inference is off (user has not opted into the
///   contract change);
/// - the local pool has at least one shard of the model;
/// - no foreign pool has advertised the model.
pub fn cross_pool_extras(
    shared: &SharedState,
    model_id: &crate::types::ModelId,
) -> HashSet<NodeId> {
    use std::sync::atomic::Ordering::Relaxed;
    if !shared.credits.private_mode.load(Relaxed) || !shared.config.pool.allow_cross_pool_inference
    {
        return HashSet::new();
    }
    // Bail when the local pool already holds at least one shard of the
    // model — keeps the existing "stays in pool" contract for models
    // the pool serves itself.
    let manifest = shared.model_registry.get_manifest(model_id);
    let local_pool_members: HashSet<NodeId> = match shared.credits.pool_state.try_read() {
        Ok(ps) => ps
            .as_ref()
            .map(|s| s.members.iter().map(|m| m.node_id.clone()).collect())
            .unwrap_or_default(),
        Err(_) => {
            // R135: surface the degradation — silently returning an
            // empty set causes the local-holder-exists check below to
            // think the pool can't serve, which triggers the cross-pool
            // fallback even when the local pool does host the model.
            // Operators should know this is happening so they can size
            // the pool_state write-lock holders.
            tracing::warn!(
                model_id = %model_id.0,
                "cross_pool_extras: pool_state write-locked — local member set unknown, falling back to empty"
            );
            HashSet::new()
        }
    };
    if let Some(ref m) = manifest {
        let any_local_pool_holder = m.shards.iter().any(|s| {
            let sid = crate::types::ShardId {
                model_id: model_id.clone(),
                index: s.index,
            };
            shared
                .model_registry
                .shard_holders(&sid)
                .iter()
                .any(|h| local_pool_members.contains(h))
        });
        if any_local_pool_holder {
            return HashSet::new();
        }
    }

    // Find foreign pools that advertised this model, then expand each
    // pool's members from the pool_registry. Skip our own pool.
    let mut extras = HashSet::new();
    let my_id = shared.identity.node_id().clone();
    for entry in shared.credits.foreign_pool_catalog.iter() {
        let (pool_id, advertised_model) = entry.key();
        if advertised_model != model_id {
            continue;
        }
        if pool_id.0 == my_id.0 {
            continue;
        }
        if let Some(pool_state) = shared.credits.pool_registry.get(pool_id) {
            for m in &pool_state.value().members {
                if m.node_id != my_id {
                    extras.insert(m.node_id.clone());
                }
            }
        }
    }
    extras
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::identity::Identity;
    use crate::inference::executor::ModelExecutor;
    use crate::storage::db::Database;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn make_state(config: Config) -> Arc<SharedState> {
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(config, identity, db, executor, None);
        state
    }

    /// R134.7: cross-pool extras are empty when the user has not opted in,
    /// regardless of whether private_mode is on.
    #[test]
    fn cross_pool_extras_empty_when_flag_off() {
        let mut config = Config::default();
        config.pool.private_mode = true;
        assert!(!config.pool.allow_cross_pool_inference);
        let state = make_state(config);
        let model_id = crate::types::ModelId("anything".into());
        let extras = cross_pool_extras(&state, &model_id);
        assert!(extras.is_empty());
    }

    /// R134.7: cross-pool extras are also empty when private_mode is off —
    /// in that case the normal global routing already applies; no fallback
    /// needed.
    #[test]
    fn cross_pool_extras_empty_when_private_mode_off() {
        let mut config = Config::default();
        config.pool.private_mode = false;
        config.pool.allow_cross_pool_inference = true;
        let state = make_state(config);
        let model_id = crate::types::ModelId("anything".into());
        let extras = cross_pool_extras(&state, &model_id);
        assert!(extras.is_empty());
    }
}
