//! Pool gossip + device stats reporting.
//!
//! Handles pool-state broadcast (owner → members), DeviceStatsReport
//! (member → owner), inbound ingestion, device-stat collection from the
//! shared node stats, and leaderboard computation.

use ed25519_dalek::Verifier;

use crate::daemon::SharedState;
use crate::pool::types::*;
use crate::types::{NetworkCommand, SwarmMessage};

use super::PoolManager;

/// R134: compute the diff between a `prev` snapshot and a `current`
/// snapshot. Returns `None` when the diff would be empty (no member /
/// pin / scalar changes). Caller is the pool owner — signs the result
/// with their identity key before broadcasting.
fn build_pool_state_diff(
    prev: &PoolState,
    current: &PoolState,
    shared_state: &SharedState,
) -> Option<swarmllm_types::PoolStateDiff> {
    use std::collections::HashSet;

    let prev_ids: HashSet<_> = prev.members.iter().map(|m| m.node_id.clone()).collect();
    let curr_ids: HashSet<_> = current.members.iter().map(|m| m.node_id.clone()).collect();

    let added_members: Vec<_> = current
        .members
        .iter()
        .filter(|m| !prev_ids.contains(&m.node_id))
        .cloned()
        .collect();
    let removed_node_ids: Vec<_> = prev
        .members
        .iter()
        .map(|m| m.node_id.clone())
        .filter(|id| !curr_ids.contains(id))
        .collect();

    let shard_pins_changed = prev.shard_pins != current.shard_pins;
    let credits_changed = prev.total_lifetime_credits != current.total_lifetime_credits;
    let split_changed = prev.member_credit_split_pct != current.member_credit_split_pct;

    if added_members.is_empty()
        && removed_node_ids.is_empty()
        && !shard_pins_changed
        && !credits_changed
        && !split_changed
    {
        return None;
    }

    let state_checksum = crate::pool::crypto::pool_state_checksum(current);
    let timestamp_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
    let payload = crate::pool::crypto::pool_state_diff_payload(
        &current.pool_id,
        prev.generation,
        current.generation,
        &state_checksum,
        timestamp_ms,
    );
    let owner_signature = shared_state.identity.sign(&payload);

    Some(swarmllm_types::PoolStateDiff {
        pool_id: current.pool_id.clone(),
        parent_generation: prev.generation,
        new_generation: current.generation,
        added_members,
        removed_node_ids,
        shard_pins: if shard_pins_changed {
            Some(current.shard_pins.clone())
        } else {
            None
        },
        total_lifetime_credits: if credits_changed {
            Some(current.total_lifetime_credits)
        } else {
            None
        },
        member_credit_split_pct: if split_changed {
            Some(current.member_credit_split_pct)
        } else {
            None
        },
        state_checksum,
        timestamp_ms,
        owner_signature,
    })
}

// SEC: cap inbound DeviceStatsReport payload sizes. The local-write path
/// enforces a 32-char `device_name` cap, but inbound gossip is what gets
/// persisted to redb AND broadcast to all pool members — a malicious member
/// can otherwise smuggle multi-MB strings/Vecs into every peer's pool state.
const MAX_DEVICE_NAME_BYTES: usize = 64;
const MAX_MODELS_HOSTED: usize = 64;
const MAX_MODEL_NAME_LEN: usize = 256;

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
                &member.invitation_expires_at,
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

    /// Force a PoolState broadcast immediately. Updates owner stats first
    /// if we're the owner. Sets `last_pool_gossip_at` and clears the
    /// `pool_gossip_dirty` flag so the coalescer doesn't double-fire.
    /// Use `maybe_gossip_pool_state` instead from event-driven sites —
    /// this method bypasses the debounce window and should only be called
    /// from the periodic interval and from `maybe_gossip_pool_state` itself.
    ///
    /// R134: when `pool.state_diff_gossip` is on AND a prior full broadcast
    /// is cached AND the cap `MAX_DIFFS_BEFORE_FULL` hasn't been hit, emits
    /// a `PoolStateDiff` instead of a full `StateGossip`. Periodic broadcasts
    /// and the first broadcast after restart always go full to bound
    /// recovery time for late joiners.
    pub(super) async fn gossip_pool_state(&mut self) {
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

        let diff_enabled = self.shared_state.config.pool.state_diff_gossip;

        // Snapshot the current state for both diff computation and broadcast.
        let ps_snapshot: Option<PoolState> = {
            let state = self.shared_state.credits.pool_state.read().await;
            state.clone()
        };

        let Some(mut current) = ps_snapshot else {
            return; // No pool yet — nothing to gossip.
        };

        // Only the pool owner emits gossip (anti-forgery — a non-owner's
        // diff signature would fail verification anyway).
        if current.pool_id != my_id {
            // We're a member, not owner. Don't broadcast pool state.
            self.last_pool_gossip_at = Some(std::time::Instant::now());
            self.pool_gossip_dirty = false;
            return;
        }

        // Decide diff vs full. We need a baseline AND must not exceed the
        // diff cap; the cap forces a fresh full broadcast every
        // MAX_DIFFS_BEFORE_FULL diffs so receivers that missed an earlier
        // diff recover within a bounded window.
        let can_diff = diff_enabled
            && self.last_broadcast_state.is_some()
            && self.diffs_since_full < super::MAX_DIFFS_BEFORE_FULL;

        // Determine generation transition. State checksum hash is
        // generation-independent — we compare the membership/pin bytes
        // by hashing both sides at the same nominal generation.
        let prior_gen = self
            .last_broadcast_state
            .as_ref()
            .map(|s| s.generation)
            .unwrap_or(0);
        let state_changed = self
            .last_broadcast_state
            .as_ref()
            .map(|prev| {
                crate::pool::crypto::pool_state_checksum_at(prev, 0)
                    != crate::pool::crypto::pool_state_checksum_at(&current, 0)
            })
            .unwrap_or(true);
        if state_changed {
            current.generation = prior_gen.saturating_add(1);
            let mut state = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut ps) = *state {
                if ps.pool_id == my_id {
                    ps.generation = current.generation;
                }
            }
        } else {
            current.generation = prior_gen;
        }

        let mut sent_diff = false;
        if can_diff && state_changed {
            if let Some(prev) = self.last_broadcast_state.as_ref() {
                if let Some(diff) = build_pool_state_diff(prev, &current, &self.shared_state) {
                    let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::StateDiff(diff));
                    let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;
                    self.diffs_since_full = self.diffs_since_full.saturating_add(1);
                    self.last_broadcast_state = Some(current.clone());
                    sent_diff = true;
                }
            }
        }

        if !sent_diff {
            let msg =
                SwarmMessage::PoolMessage(crate::types::PoolMessage::StateGossip(current.clone()));
            let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;
            self.last_broadcast_state = Some(current);
            self.diffs_since_full = 0;
        }

        self.last_pool_gossip_at = Some(std::time::Instant::now());
        self.pool_gossip_dirty = false;
    }

    /// R134: handle an inbound `PoolStateDiff`. Verifies the owner
    /// signature, applies the diff to the cached state, verifies the
    /// post-apply checksum matches the owner's intent, and updates
    /// `state.credits.pool_state` + `pool_registry`.
    pub(super) async fn handle_pool_state_diff_gossip(
        &mut self,
        diff: swarmllm_types::PoolStateDiff,
    ) {
        use ed25519_dalek::Verifier;

        // Freshness: same one-sided staleness window as other timestamped gossip.
        let now_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
        if !crate::daemon::dispatch::timestamp_fresh_one_sided(
            diff.timestamp_ms,
            now_ms,
            300_000,
            5_000,
            "pool_state_diff",
        ) {
            tracing::debug!(
                pool_id = %diff.pool_id,
                "Dropping stale PoolStateDiff"
            );
            return;
        }

        // Verify owner signature over the diff payload.
        let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&diff.pool_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!(pool_id = %hex::encode(&diff.pool_id.0[..8]), "Invalid owner key in PoolStateDiff");
                return;
            }
        };
        let payload = crate::pool::crypto::pool_state_diff_payload(
            &diff.pool_id,
            diff.parent_generation,
            diff.new_generation,
            &diff.state_checksum,
            diff.timestamp_ms,
        );
        let sig_bytes: &[u8; 64] = match diff.owner_signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                tracing::warn!(pool_id = %diff.pool_id, "Invalid signature length in PoolStateDiff");
                return;
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
        if owner_key.verify(&payload, &sig).is_err() {
            tracing::warn!(pool_id = %diff.pool_id, "Invalid owner signature in PoolStateDiff");
            return;
        }

        if diff.new_generation <= diff.parent_generation {
            tracing::debug!(
                pool_id = %diff.pool_id,
                parent = diff.parent_generation,
                new = diff.new_generation,
                "Dropping PoolStateDiff with non-advancing generation"
            );
            return;
        }

        // SEC: cap the diff size before doing any further work.
        let max_pool_size = self.shared_state.config.pool.max_pool_size as usize;
        if diff.added_members.len() > max_pool_size
            || diff.removed_node_ids.len() > max_pool_size
            || diff
                .shard_pins
                .as_ref()
                .map(|p| p.len() > max_pool_size * 8)
                .unwrap_or(false)
        {
            tracing::warn!(pool_id = %diff.pool_id, "Rejecting oversized PoolStateDiff");
            return;
        }

        // Look up the cached state. Source of truth is `pool_state` when this
        // is our own pool, else `pool_registry`.
        let my_id = self.shared_state.identity.node_id().clone();
        let mut base_state: Option<PoolState> = {
            let local = self.shared_state.credits.pool_state.read().await;
            if let Some(ref ps) = *local {
                if ps.pool_id == diff.pool_id {
                    Some(ps.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };
        if base_state.is_none() {
            base_state = self
                .shared_state
                .credits
                .pool_registry
                .get(&diff.pool_id)
                .map(|e| e.value().clone());
        }

        let Some(mut new_state) = base_state else {
            // Receiver has no cached state — drop. Next full broadcast will
            // resync (within `gossip_interval_secs` worst case).
            tracing::debug!(pool_id = %diff.pool_id, "Dropping PoolStateDiff — no cached baseline");
            return;
        };

        if new_state.generation != diff.parent_generation {
            tracing::debug!(
                pool_id = %diff.pool_id,
                cached = new_state.generation,
                expected = diff.parent_generation,
                "Dropping PoolStateDiff — parent generation mismatch"
            );
            return;
        }

        // Apply removals first, then additions; matches FIFO semantics of a
        // sequence of operations.
        new_state
            .members
            .retain(|m| !diff.removed_node_ids.contains(&m.node_id));
        for added in &diff.added_members {
            if new_state.members.iter().any(|m| m.node_id == added.node_id) {
                continue;
            }
            new_state.members.push(added.clone());
        }
        if let Some(ref pins) = diff.shard_pins {
            new_state.shard_pins = pins.clone();
        }
        if let Some(c) = diff.total_lifetime_credits {
            new_state.total_lifetime_credits = c;
        }
        if let Some(pct) = diff.member_credit_split_pct {
            new_state.member_credit_split_pct = pct;
        }
        new_state.generation = diff.new_generation;

        // Cap inbound size (matches the StateGossip handler).
        if new_state.members.len() > max_pool_size {
            tracing::warn!(
                pool_id = %diff.pool_id,
                members = new_state.members.len(),
                "Rejecting PoolStateDiff — post-apply members exceed max"
            );
            return;
        }

        // Verify the checksum. If our local computation disagrees with the
        // owner's intent, drop and wait for the next full broadcast.
        let local_checksum = crate::pool::crypto::pool_state_checksum(&new_state);
        if local_checksum != diff.state_checksum {
            tracing::warn!(
                pool_id = %diff.pool_id,
                "PoolStateDiff checksum mismatch — dropping (next full broadcast will resync)"
            );
            return;
        }

        // Verify each added member's acceptance_signature (matches the
        // full-state handler's per-member verification).
        for member in &diff.added_members {
            if member.node_id == diff.pool_id {
                continue;
            }
            let member_key = match ed25519_dalek::VerifyingKey::from_bytes(&member.node_id.0) {
                Ok(k) => k,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid member key in PoolStateDiff");
                    return;
                }
            };
            let acceptance_payload = crate::pool::crypto::acceptance_payload(
                &member.invitation_id,
                &diff.pool_id,
                &member.node_id,
                &member.invitation_expires_at,
            );
            let sig_bytes: &[u8; 64] = match member.acceptance_signature.as_slice().try_into() {
                Ok(b) => b,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid acceptance signature length in PoolStateDiff");
                    return;
                }
            };
            let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
            if member_key.verify(&acceptance_payload, &sig).is_err() {
                tracing::warn!(member = %member.node_id, "Invalid acceptance signature in PoolStateDiff");
                return;
            }
        }

        // Update local pool_state if this is our pool — preserve our local
        // device_name / device_stats just like the StateGossip handler.
        {
            let mut local_ps = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut local) = *local_ps {
                if local.pool_id == diff.pool_id && diff.pool_id != my_id {
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

                    local.members = new_state.members.clone();
                    local.total_lifetime_credits = new_state.total_lifetime_credits;
                    local.member_credit_split_pct = new_state.member_credit_split_pct;
                    local.shard_pins = new_state.shard_pins.clone();
                    local.generation = new_state.generation;

                    if let Some(me) = local.members.iter_mut().find(|m| m.node_id == my_id) {
                        if my_device_name.is_some() && me.device_name.is_none() {
                            me.device_name = my_device_name;
                        }
                        if my_device_stats.is_some() && me.device_stats.is_none() {
                            me.device_stats = my_device_stats;
                        }
                    }
                    if let Err(e) = self.persist_pool_state(local) {
                        tracing::warn!(error = %e, "Failed to persist pool state from diff");
                    }
                }
            }
        }

        self.shared_state
            .credits
            .pool_registry
            .insert(diff.pool_id.clone(), new_state);
    }

    /// R131: rate-limited event-driven gossip entrypoint. If the last
    /// broadcast was longer than `POOL_GOSSIP_MIN_INTERVAL` ago (or this
    /// is the first broadcast), fires immediately. Otherwise sets
    /// `pool_gossip_dirty` so the pool-coalesce timer fires a trailing
    /// broadcast once the cooldown expires — collapsing N bursty member
    /// changes into ≤ 2 broadcasts (one immediate, one trailing) instead
    /// of N. The periodic full-broadcast tick bypasses this gate.
    pub(super) async fn maybe_gossip_pool_state(&mut self) {
        let ready = self
            .last_pool_gossip_at
            .map(|t| t.elapsed() >= super::POOL_GOSSIP_MIN_INTERVAL)
            .unwrap_or(true);
        if ready {
            self.gossip_pool_state().await;
        } else {
            self.pool_gossip_dirty = true;
            tracing::debug!("PoolState gossip debounced; trailing broadcast pending");
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

        let device_name = device_name.map(|n| {
            if n.len() > MAX_DEVICE_NAME_BYTES {
                tracing::warn!(
                    %node_id,
                    len = n.len(),
                    "Truncating oversized inbound device_name"
                );
                // R107: byte-based truncation to honor the byte cap. The
                // earlier `chars().take(MAX_DEVICE_NAME_BYTES)` confused
                // bytes with chars — 64 multi-byte CJK chars is 256 bytes,
                // letting an attacker bypass the documented byte limit.
                // Cut at the last valid char boundary <= the cap so we
                // never split mid-codepoint.
                let mut end = MAX_DEVICE_NAME_BYTES.min(n.len());
                while end > 0 && !n.is_char_boundary(end) {
                    end -= 1;
                }
                let mut s = n;
                s.truncate(end);
                s
            } else {
                n
            }
        });

        let mut stats = stats;
        if stats.models_hosted.len() > MAX_MODELS_HOSTED
            || stats
                .models_hosted
                .iter()
                .any(|m| m.len() > MAX_MODEL_NAME_LEN)
        {
            tracing::warn!(
                %node_id,
                count = stats.models_hosted.len(),
                "Truncating oversized inbound models_hosted in DeviceStatsReport"
            );
            stats.models_hosted.truncate(MAX_MODELS_HOSTED);
            for m in stats.models_hosted.iter_mut() {
                if m.len() > MAX_MODEL_NAME_LEN {
                    *m = m.chars().take(MAX_MODEL_NAME_LEN).collect();
                }
            }
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

            members.sort_by_key(|(_, credits)| std::cmp::Reverse(*credits));

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
