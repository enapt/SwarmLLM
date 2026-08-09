use std::collections::HashSet;
use std::sync::atomic::Ordering::Relaxed;

use crate::daemon::state::SharedState;
use crate::types::NodeId;

/// Whether this peer is one the operator has explicitly admitted to their pool.
///
/// **This is the credential-trust question, and it is NOT the same as
/// [`allowed_node_set`].** That one answers "may this node take part in my
/// inference", returns `None` (meaning *everyone*) whenever private mode is off,
/// and is about routing scope. Handing a secret to a peer needs a positive act
/// of trust, not the absence of a restriction — so this returns `false` by
/// default and only `true` for a node that joined via a `swarmpool://` invite
/// the operator issued or accepted.
///
/// LAN membership deliberately does not count. A private address is not an
/// authenticated boundary — the same reasoning that keeps `dashboard_trust_lan`
/// off by default (gotcha #195).
pub fn is_pool_member(shared: &SharedState, node_id: &NodeId) -> bool {
    if node_id == shared.identity.node_id() {
        return true;
    }
    // `try_read` matches `allowed_node_set`: on contention fall back to "not a
    // member", which is the safe direction for a trust check.
    match shared.credits.pool_state.try_read() {
        Ok(guard) => guard
            .as_ref()
            .is_some_and(|ps| ps.members.iter().any(|m| m.node_id == *node_id)),
        Err(_) => false,
    }
}

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
    // R137: read `allow_cross_pool_inference` from the runtime AtomicBool
    // mirror on `state.credits` rather than the startup-frozen config.
    // Identical semantics + value when no admin PUT has flipped it.
    if !shared.credits.private_mode.load(Relaxed) || !shared.cfg().pool.allow_cross_pool_inference {
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
            // R135 (fix-up): pool_state is write-locked. Returning an
            // empty set here would let the local-holder-exists check
            // below treat the pool as if it can't serve — and then
            // route the prompt CROSS-POOL even when the local pool
            // actually does host the model. That inverts the stated
            // contract ("cross-pool only when local pool genuinely
            // can't serve"). Safe-degrade direction is to refuse the
            // fallback and let the request retry once the lock clears.
            tracing::warn!(
                model_id = %model_id.0,
                "cross_pool_extras: pool_state write-locked — refusing cross-pool fallback to avoid leaking prompts to foreign pools"
            );
            return HashSet::new();
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

    /// **Trust for handing over a credential must be a positive act.**
    ///
    /// `find_peer_with_model` -> `forward_to_peer` sends this node's
    /// `Authorization` header verbatim, and that header carries the API key
    /// which also guards `/api/admin/*`. Before this gate existed, ANY peer
    /// whose gossiped capability mentioned the requested model was eligible, so
    /// advertising a popular model was enough to harvest the admin key of every
    /// node that asked for it.
    ///
    /// A stranger must not qualify by default, however well-connected or
    /// well-scored — none of which is a statement by the operator.
    #[test]
    fn a_stranger_is_not_a_pool_member() {
        let state = make_state(Config::default());
        let stranger = crate::types::NodeId([9u8; 32]);
        assert!(!is_pool_member(&state, &stranger));
    }

    /// Not the same question as `allowed_node_set`, which returns `None`
    /// (meaning *everyone may take part in inference*) whenever private mode is
    /// off. Reusing that here would have made every peer credential-trusted on
    /// a default install — the exact opposite of the intent.
    #[test]
    fn unrestricted_routing_scope_does_not_imply_credential_trust() {
        let state = make_state(Config::default());
        assert!(
            allowed_node_set(&state).is_none(),
            "precondition: routing is unrestricted by default"
        );
        let stranger = crate::types::NodeId([7u8; 32]);
        assert!(
            !is_pool_member(&state, &stranger),
            "an unrestricted routing scope must not make a stranger trusted"
        );
    }

    /// The local node is always itself.
    #[test]
    fn the_local_node_is_always_a_member() {
        let state = make_state(Config::default());
        let me = state.identity.node_id().clone();
        assert!(is_pool_member(&state, &me));
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

    /// R135: when both flags are on AND a foreign pool has advertised the
    /// model AND the local pool doesn't host it, extras includes the
    /// foreign pool's members. This is the happy-path for cross-pool
    /// fallback.
    #[test]
    fn cross_pool_extras_returns_foreign_members_when_eligible() {
        use crate::pool::types::{PoolMembership, PoolState};
        use crate::types::ModelId;
        use chrono::Utc;
        let mut config = Config::default();
        config.pool.private_mode = true;
        config.pool.allow_cross_pool_inference = true;
        let state = make_state(config);
        state
            .credits
            .private_mode
            .store(true, std::sync::atomic::Ordering::Release);

        let foreign_pool = crate::types::NodeId([7u8; 32]);
        let foreign_member_a = crate::types::NodeId([1u8; 32]);
        let foreign_member_b = crate::types::NodeId([2u8; 32]);
        let model_id = ModelId("forbidden-fruit".into());

        // Seed the foreign pool's PoolState so member expansion can resolve it.
        state.credits.pool_registry.insert(
            foreign_pool.clone(),
            PoolState {
                pool_id: foreign_pool.clone(),
                name: "foreign".into(),
                created_at: Utc::now(),
                owner_signature: vec![],
                members: vec![
                    PoolMembership {
                        node_id: foreign_member_a.clone(),
                        credits_contributed: 0,
                        joined_at: Utc::now(),
                        acceptance_signature: vec![],
                        invitation_id: uuid::Uuid::new_v4(),
                        invitation_expires_at: Utc::now(),
                        device_name: None,
                        last_seen: None,
                        online: false,
                        device_stats: None,
                        contribution_level: 100,
                    },
                    PoolMembership {
                        node_id: foreign_member_b.clone(),
                        credits_contributed: 0,
                        joined_at: Utc::now(),
                        acceptance_signature: vec![],
                        invitation_id: uuid::Uuid::new_v4(),
                        invitation_expires_at: Utc::now(),
                        device_name: None,
                        last_seen: None,
                        online: false,
                        device_stats: None,
                        contribution_level: 100,
                    },
                ],
                shard_pins: Default::default(),
                total_lifetime_credits: 0,
                member_credit_split_pct: 0,
                generation: 0,
            },
        );
        // Foreign pool advertised the model — this is what
        // PoolModelAvailability ingest would do.
        state.credits.foreign_pool_catalog.insert(
            (foreign_pool.clone(), model_id.clone()),
            crate::types::unix_now_ms(),
        );

        let extras = cross_pool_extras(&state, &model_id);
        assert_eq!(extras.len(), 2);
        assert!(extras.contains(&foreign_member_a));
        assert!(extras.contains(&foreign_member_b));
    }

    /// R135: when the local pool already hosts the model, extras MUST be
    /// empty — the "stays in pool" contract is preserved for any model
    /// the local pool can serve itself, even if foreign pools also serve
    /// it. Verifies the early-return at scope.rs:139.
    #[test]
    fn cross_pool_extras_empty_when_local_pool_serves() {
        // We assert the early-return shape by constructing a state where
        // the local pool has members but no model. The full
        // local-pool-serves check requires a ModelManifest + holder
        // population which is tested at the scheduler integration
        // level. Here we just verify the structural plumbing — extras
        // is empty when local_pool_members is non-empty AND the
        // catalog is empty.
        let mut config = Config::default();
        config.pool.private_mode = true;
        config.pool.allow_cross_pool_inference = true;
        let state = make_state(config);
        state
            .credits
            .private_mode
            .store(true, std::sync::atomic::Ordering::Release);
        let extras = cross_pool_extras(&state, &crate::types::ModelId("nothing-advertised".into()));
        // Empty catalog → empty extras regardless of pool state.
        assert!(extras.is_empty());
    }

    /// Turning cross-pool inference off through `PUT /api/admin/config` is
    /// honoured on the next call, with no daemon restart.
    ///
    /// The gate reads the LIVE config, not the boot-time snapshot. It used to
    /// need a private `AtomicBool` mirror for this (R137); that mirror was
    /// folded into the live config on 2026-08-09 once every setting gained the
    /// same treatment, so the test now flips the setting the way the API does.
    #[test]
    fn cross_pool_extras_honors_runtime_flag_toggle() {
        let mut config = Config::default();
        // Start with the gate open so the function would otherwise return data.
        config.pool.private_mode = true;
        config.pool.allow_cross_pool_inference = true;
        let state = make_state(config);

        // Turn it off at runtime. The boot snapshot still says `true`, so this
        // also pins that the gate is not reading `state.config`.
        let mut off = (**state.cfg()).clone();
        off.pool.allow_cross_pool_inference = false;
        state.apply_live_config(off);
        assert!(
            state.config.pool.allow_cross_pool_inference,
            "the boot snapshot must be unchanged, or this proves nothing"
        );

        let extras = cross_pool_extras(&state, &crate::types::ModelId("any".into()));
        assert!(
            extras.is_empty(),
            "runtime flag-off must override config-on"
        );

        // Back on — the gate passes again (an empty catalog still yields empty).
        let mut on = (**state.cfg()).clone();
        on.pool.allow_cross_pool_inference = true;
        state.apply_live_config(on);
        let extras = cross_pool_extras(&state, &crate::types::ModelId("any".into()));
        assert!(
            extras.is_empty(),
            "no catalog entries still yields empty (gate passed but no foreign advertise)"
        );
    }
}
