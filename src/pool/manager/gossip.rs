//! Pool gossip + device stats reporting.
//!
//! Handles pool-state broadcast (owner → members), DeviceStatsReport
//! (member → owner), inbound ingestion, device-stat collection from the
//! shared node stats, and leaderboard computation.

use ed25519_dalek::Verifier;

use crate::pool::types::*;
use crate::types::{NetworkCommand, SwarmMessage};

use super::PoolManager;

impl PoolManager {
    pub(super) async fn handle_pool_state_gossip(&mut self, state: PoolState) {
        // Verify owner signature before inserting into registry
        let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&state.pool_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!(pool_id = %hex::encode(&state.pool_id.0[..8]), "Invalid owner key in pool state gossip");
                return;
            }
        };
        // Reconstruct the pool creation signing payload and verify
        let payload = crate::pool::crypto::pool_create_payload(
            &state.pool_id,
            &state.name,
            &state.created_at,
        );
        let sig_bytes: &[u8; 64] = match state.owner_signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                tracing::warn!(pool_id = %state.pool_id, "Pool state gossip has invalid signature length");
                return;
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
        if owner_key.verify(&payload, &sig).is_err() {
            tracing::warn!(pool_id = %state.pool_id, "Invalid owner signature in pool state gossip");
            return;
        }

        // Enforce max pool size to prevent Ed25519 verification DoS
        let max_pool_size = self.shared_state.config.pool.max_pool_size;
        if state.members.len() > max_pool_size as usize {
            tracing::warn!(
                pool_id = %state.pool_id,
                members = state.members.len(),
                max = max_pool_size,
                "Rejecting pool state gossip: exceeds max pool size"
            );
            return;
        }

        // Reject duplicate node_ids in gossiped pool state (prevents inflated stats)
        {
            let mut seen_nodes = std::collections::HashSet::new();
            for member in &state.members {
                if !seen_nodes.insert(member.node_id.clone()) {
                    tracing::warn!(
                        pool_id = %state.pool_id,
                        dup_node = %member.node_id,
                        "Duplicate node_id in gossiped pool state — rejecting"
                    );
                    return;
                }
            }
        }

        // SEC-C7: Verify acceptance_signature of each member
        for member in &state.members {
            // The pool owner's own membership uses the pool creation signature, skip it
            if member.node_id == state.pool_id {
                continue;
            }
            let member_key = match ed25519_dalek::VerifyingKey::from_bytes(&member.node_id.0) {
                Ok(k) => k,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid member key in pool state gossip");
                    return;
                }
            };
            // Verify the acceptance signature using the acceptance payload
            let acceptance_payload = crate::pool::crypto::acceptance_payload(
                &member.invitation_id,
                &state.pool_id,
                &member.node_id,
            );
            let sig_bytes: &[u8; 64] = match member.acceptance_signature.as_slice().try_into() {
                Ok(b) => b,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid acceptance signature length in pool state gossip");
                    return;
                }
            };
            let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
            if member_key.verify(&acceptance_payload, &sig).is_err() {
                tracing::warn!(member = %member.node_id, "Invalid acceptance signature in pool state gossip");
                return;
            }
        }

        // If this gossip is for our own pool, update local pool_state with full member list
        // (preserving our locally-set device_name and device_stats)
        {
            let my_id = self.shared_state.identity.node_id().clone();
            let mut local_ps = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut local) = *local_ps {
                if local.pool_id == state.pool_id && state.pool_id != my_id {
                    // We're a member (not owner) of this pool — merge gossip into our local state
                    // Preserve our local device_name and device_stats
                    let my_device_name = local
                        .members
                        .iter()
                        .find(|m| m.node_id == my_id)
                        .and_then(|m| m.device_name.clone());
                    let my_device_stats = local
                        .members
                        .iter()
                        .find(|m| m.node_id == my_id)
                        .and_then(|m| m.device_stats.clone());

                    // Replace local state with gossip (full member list from leader)
                    local.name = state.name.clone();
                    local.members = state.members.clone();
                    local.total_lifetime_credits = state.total_lifetime_credits;
                    local.member_credit_split_pct = state.member_credit_split_pct;

                    // Restore our locally-set fields
                    if let Some(me) = local.members.iter_mut().find(|m| m.node_id == my_id) {
                        if my_device_name.is_some() && me.device_name.is_none() {
                            me.device_name = my_device_name;
                        }
                        if my_device_stats.is_some() && me.device_stats.is_none() {
                            me.device_stats = my_device_stats;
                        }
                    }

                    if let Err(e) = self.persist_pool_state(local) {
                        tracing::warn!(error = %e, "Failed to persist pool state from gossip");
                    }
                    tracing::debug!(
                        pool_id = %state.pool_id,
                        members = local.members.len(),
                        "Updated local pool state from leader gossip"
                    );
                }
            }
        }

        // Store in registry for network-wide visibility (cap to prevent unbounded growth from gossip)
        const MAX_POOL_REGISTRY: usize = 1_000;
        if self.shared_state.credits.pool_registry.len() < MAX_POOL_REGISTRY
            || self
                .shared_state
                .credits
                .pool_registry
                .contains_key(&state.pool_id)
        {
            self.shared_state
                .credits
                .pool_registry
                .insert(state.pool_id.clone(), state);
        }
    }

    pub(super) async fn gossip_pool_state(&self) {
        let my_id = self.shared_state.identity.node_id().clone();
        // If we're the owner, update our own stats before gossiping
        {
            let mut state = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut ps) = *state {
                if ps.pool_id == my_id {
                    let stats = self.collect_device_stats().await;
                    if let Some(me) = ps.members.iter_mut().find(|m| m.node_id == my_id) {
                        me.device_stats = Some(stats);
                        me.last_seen = Some(chrono::Utc::now());
                        me.online = true;
                    }
                }
            }
        }
        let state = self.shared_state.credits.pool_state.read().await;
        if let Some(ref ps) = *state {
            let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::StateGossip(ps.clone()));
            let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;
        }
    }

    /// Send a device stats report to the pool leader (members only).
    /// Called on each gossip tick so the leader has up-to-date stats + nickname.
    pub(super) async fn send_device_stats_report(&self) {
        let my_id = self.shared_state.identity.node_id().clone();
        let pool_id = {
            let state = self.shared_state.credits.pool_state.read().await;
            match state.as_ref() {
                Some(ps) if ps.pool_id != my_id => ps.pool_id.clone(),
                _ => return, // We're the owner or not in a pool
            }
        };

        // Resolve device name: pool device_name first, then identity nickname
        let device_name = {
            let state = self.shared_state.credits.pool_state.read().await;
            let pool_name = state
                .as_ref()
                .and_then(|ps| ps.members.iter().find(|m| m.node_id == my_id))
                .and_then(|m| m.device_name.clone());
            pool_name.or_else(|| {
                let store =
                    crate::identity::nickname::NicknameStore::new(self.shared_state.db.clone());
                store.get_prefs().ok().and_then(|p| p.nickname)
            })
        };

        let stats = self.collect_device_stats().await;

        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::DeviceStatsReport {
            pool_id,
            node_id: my_id,
            device_name,
            stats,
        });
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;
    }

    /// Collect real device stats from shared state.
    pub(super) async fn collect_device_stats(&self) -> crate::types::PoolDeviceStats {
        let node_stats = self.shared_state.metrics.node_stats.read().await;

        // Uptime from start time
        let uptime_secs = (chrono::Utc::now() - node_stats.uptime_start)
            .num_seconds()
            .max(0) as u64;

        // GPU VRAM
        let vram_mb = self
            .shared_state
            .gpu_info
            .as_ref()
            .map(|g| g.vram_total_mb)
            .unwrap_or(0);

        // RAM: use sysinfo (blocking but lightweight)
        let ram_mb = tokio::task::block_in_place(|| {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            sys.total_memory() / (1024 * 1024)
        });

        // Count hosted shards and collect model names
        let my_node_id = self.shared_state.identity.node_id();
        let mut shards_hosted: u32 = 0;
        let mut models_hosted: Vec<String> = Vec::new();
        for manifest in self.shared_state.model_registry.models() {
            let local = self
                .shared_state
                .model_registry
                .local_shard_indices(&manifest.id, my_node_id);
            if !local.is_empty() {
                shards_hosted += local.len() as u32;
                models_hosted.push(manifest.name.clone());
            }
        }

        crate::types::PoolDeviceStats {
            forwards_served: self
                .shared_state
                .metrics
                .forwards_served_atomic
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_served: self
                .shared_state
                .metrics
                .requests_served_atomic
                .load(std::sync::atomic::Ordering::Relaxed),
            shards_hosted,
            vram_mb,
            ram_mb,
            uptime_secs,
            models_hosted,
        }
    }

    /// Handle an inbound device stats report from a pool member (leader only).
    pub(super) async fn handle_inbound_device_stats_report(
        &mut self,
        pool_id: crate::types::NodeId,
        node_id: crate::types::NodeId,
        device_name: Option<String>,
        stats: crate::types::PoolDeviceStats,
    ) {
        let my_id = self.shared_state.identity.node_id().clone();
        // Only the pool owner processes these reports
        if pool_id != my_id {
            return;
        }

        let mut ps_guard = self.shared_state.credits.pool_state.write().await;
        if let Some(ref mut ps) = *ps_guard {
            match ps.members.iter_mut().find(|m| m.node_id == node_id) {
                Some(member) => {
                    if device_name.is_some() {
                        member.device_name = device_name;
                    }
                    member.device_stats = Some(stats);
                    member.last_seen = Some(chrono::Utc::now());
                    member.online = true;
                }
                None => {
                    tracing::warn!(
                        %node_id,
                        "Dropping DeviceStatsReport from non-member (possibly stale after removal)"
                    );
                }
            }
        }
        drop(ps_guard);
    }

    pub(super) async fn build_leaderboard(&self) -> Vec<LeaderboardEntry> {
        let state = self.shared_state.credits.pool_state.read().await;
        let mut entries = Vec::new();

        if let Some(ref ps) = *state {
            let mut members: Vec<_> = ps
                .members
                .iter()
                .map(|m| (m.node_id.clone(), m.credits_contributed))
                .collect();

            members.sort_by(|a, b| b.1.cmp(&a.1));

            for (rank, (node_id, credits)) in members.into_iter().enumerate() {
                entries.push(LeaderboardEntry {
                    node_id,
                    credits_contributed: credits,
                    rank: (rank + 1) as u32,
                });
            }
        }

        entries
    }
}
