use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::types::NodeId;

use super::hf::{HfProbeInfo, HfSource};

/// Model management: shard acquisition, auto-manage, trust gating, pruning.
pub struct ModelMgmt {
    pub acquisition_progress:
        DashMap<crate::types::ModelId, crate::model::acquisition::AcquisitionStatus>,
    pub hf_sources: DashMap<crate::types::ModelId, HfSource>,
    pub auto_manage_notify: Arc<tokio::sync::Notify>,
    pub auto_manage_enabled: std::sync::atomic::AtomicBool,
    pub auto_manage_default_model_cap: AtomicU32,
    pub model_auto_manage_policies:
        DashMap<crate::types::ModelId, crate::config::ModelAutoManagePolicy>,
    pub hf_probe_cache: DashMap<crate::types::ModelId, HfProbeInfo>,
    pub peer_shard_downloads: DashMap<crate::types::ShardId, Vec<(NodeId, u32)>>,
    pub download_cancel_flags: DashMap<crate::types::ModelId, Arc<AtomicBool>>,
    pub model_trust: DashMap<crate::types::ModelId, crate::types::ModelTrustInfo>,
    pub loading_models: DashMap<crate::types::ModelId, Arc<tokio::sync::Notify>>,
    pub locked_shards: DashMap<crate::types::ShardId, bool>,
    /// Shards where P2P download has exhausted all peer attempts in this session.
    /// Signals auto_manage to force the HF path even when peer holders are registered.
    /// Cleared when a download for the shard successfully completes.
    pub shard_p2p_failed: dashmap::DashSet<crate::types::ShardId>,
    pub model_request_counts: DashMap<crate::types::ModelId, AtomicU64>,
    pub resource_schedule: RwLock<crate::config::ResourceSchedule>,
    pub prune_history: RwLock<VecDeque<crate::types::PruneEvent>>,
    /// Parallax Phase C.2 stability counter per shard. Positive values mean
    /// the allocator has recommended this node hold the shard for N
    /// consecutive auto-manage ticks; negative values mean the allocator
    /// wants it off this node. Score biases trigger once the magnitude
    /// crosses `PARALLAX_STABILITY_THRESHOLD`. Clamped to `[-10, 10]` so a
    /// long-stable recommendation can't be flipped by a single noisy tick.
    pub parallax_stability: DashMap<crate::types::ShardId, i32>,
    /// Item 8 Phase 1: cross-node prefix-cache index. Outer key = `model_id`,
    /// inner key = chained BLAKE3 block hash, value = the set of remote
    /// peers known to hold a KV snapshot ending at that block. Updated from
    /// `SwarmMessage::PrefixCacheAnnounce` and consulted by Phase 2's
    /// remote-KV fetch path. We never insert ourselves here — local cache
    /// hits are served by the in-process `PrefixCache` directly.
    pub cross_node_prefix_index:
        DashMap<crate::types::ModelId, DashMap<[u8; 32], dashmap::DashSet<NodeId>>>,
    /// Reverse index from `peer_id → list of (model_id, block_hash)` so that
    /// when a peer disconnects (or sends a fresh announce that supersedes
    /// the previous one) we can remove every entry attributed to them
    /// without rescanning the per-model maps. Wrapped in an `RwLock` rather
    /// than `DashMap<_, Mutex<_>>` because peer announces are rare relative
    /// to lookups; a single short write lock per announce is cheaper than
    /// per-bucket locking.
    pub peer_prefix_blocks:
        DashMap<NodeId, DashMap<crate::types::ModelId, dashmap::DashSet<[u8; 32]>>>,
    /// OwnedSemaphorePermits for in-flight P2P shard downloads, paired with
    /// the time the permit was parked. The P2P path queues a
    /// `NetworkCommand::SendShardRequest` and returns immediately, so the
    /// permit can't be held on `trigger_download`'s stack — it's parked
    /// here keyed by `ShardId` and released from the network event loop
    /// (success path + retry-fallback give-up + stall watchdog). HF and
    /// mmproj paths hold their permits in-task and don't touch this map.
    /// Without this, `max_concurrent_downloads` only bounded HF — P2P
    /// permits dropped the moment the request was queued, so the
    /// semaphore had no effect on P2P load.
    ///
    /// SEC: the `Instant` is consumed by `AutoShardManager::sweep_stalled_p2p_permits`
    /// — without that periodic sweep, a silent network drop (libp2p
    /// dispatch loop missed event, peer disconnected before request
    /// landed) parks the permit forever and `max_concurrent_downloads`
    /// silently freezes after enough silent drops.
    pub p2p_download_permits:
        DashMap<crate::types::ShardId, (tokio::sync::OwnedSemaphorePermit, std::time::Instant)>,
    /// R111: latest computed Wishlist snapshot. ArcSwap so the dashboard +
    /// REST + future HfWatcher all read a lock-free snapshot. Refreshed on
    /// every WS stats build (cheap pass over the model registry) and on
    /// every auto-manage tick.
    pub wishlist: arc_swap::ArcSwap<crate::model::auto_manage::wishlist::Wishlist>,
}

impl ModelMgmt {
    /// Check if a shard is currently being downloaded, pending, or verifying.
    /// Prevents races where multiple subsystems try to download the same shard.
    pub fn is_shard_in_progress(&self, model_id: &crate::types::ModelId, shard_index: u32) -> bool {
        self.acquisition_progress
            .get(model_id)
            .map(|entry| {
                entry
                    .shard_progress
                    .get(&shard_index)
                    .map(|sp| {
                        matches!(
                            sp.state,
                            crate::model::acquisition::ShardState::Downloading
                                | crate::model::acquisition::ShardState::Pending
                                | crate::model::acquisition::ShardState::Verifying
                        )
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Mutate a model's AcquisitionStatus if present. No-op if the model has
    /// no acquisition entry. Locks `acquisition_progress` only for the body
    /// of the closure — do NOT hold the closure across `.await`.
    pub fn update_acquisition<F>(&self, model_id: &crate::types::ModelId, f: F)
    where
        F: FnOnce(&mut crate::model::acquisition::AcquisitionStatus),
    {
        if let Some(mut entry) = self.acquisition_progress.get_mut(model_id) {
            f(&mut entry);
        }
    }

    /// Mark an acquisition as failed — sets state, increments failed_shards,
    /// and pushes a log line. Safe to call if the model has no acquisition
    /// entry (no-op).
    pub fn set_acquisition_failed(
        &self,
        model_id: &crate::types::ModelId,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        self.update_acquisition(model_id, |s| {
            s.state = crate::model::acquisition::AcquisitionState::Failed {
                reason: reason.clone(),
            };
            s.failed_shards += 1;
            s.log_push(format!("Failed: {reason}"));
        });
    }

    /// Mark an acquisition as complete for single-file downloads (e.g., full
    /// GGUF). Sets state + treats the single file as 1 downloaded and verified
    /// shard. For multi-shard downloads, use `update_acquisition` directly.
    pub fn set_acquisition_complete_single(
        &self,
        model_id: &crate::types::ModelId,
        log_msg: impl Into<String>,
    ) {
        let msg = log_msg.into();
        self.update_acquisition(model_id, |s| {
            s.state = crate::model::acquisition::AcquisitionState::Complete;
            s.downloaded_shards = 1;
            s.verified_shards = 1;
            s.log_push(msg);
        });
    }

    /// Register a new download job: insert the initial AcquisitionStatus and
    /// a cancel flag atomically from the caller's perspective, so subsystems
    /// that observe one but not the other (auto-manage scan vs hf download)
    /// don't race. Returns the cancel flag Arc.
    pub fn begin_download(
        &self,
        model_id: crate::types::ModelId,
        status: crate::model::acquisition::AcquisitionStatus,
    ) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.acquisition_progress.insert(model_id.clone(), status);
        self.download_cancel_flags.insert(model_id, flag.clone());
        flag
    }

    /// Item 8 Phase 1: replace this peer's known set of prefix-cache block
    /// hashes for `model_id` with `new_blocks`. Drops any previously-recorded
    /// blocks for the same `(peer, model)` pair, preserving entries from
    /// other peers in the per-block holder set. A peer announcing an empty
    /// set is treated as "I no longer hold any blocks for this model".
    ///
    /// SEC: drops any input list larger than `MAX_BLOCKS_PER_PEER_MODEL`.
    /// Without this cap a single misbehaving peer can announce millions of
    /// distinct block hashes per model and exhaust memory in the per-block
    /// holder sets. The cap matches the worst-case prefix-cache snapshot
    /// count for typical 8K-context workloads (256-token blocks → ~32 blocks
    /// per request × a few hundred concurrent sessions).
    ///
    /// Returns `(added, removed)` counts for logging — strictly diagnostic.
    pub fn replace_peer_prefix_blocks(
        &self,
        peer: NodeId,
        model_id: crate::types::ModelId,
        mut new_blocks: Vec<[u8; 32]>,
    ) -> (usize, usize) {
        const MAX_BLOCKS_PER_PEER_MODEL: usize = 16_384;
        if new_blocks.len() > MAX_BLOCKS_PER_PEER_MODEL {
            tracing::warn!(
                %peer,
                model = %model_id,
                announced = new_blocks.len(),
                cap = MAX_BLOCKS_PER_PEER_MODEL,
                "Truncating prefix-block announce: exceeds per-peer-per-model cap"
            );
            new_blocks.truncate(MAX_BLOCKS_PER_PEER_MODEL);
        }
        // Snapshot the previous block set for this (peer, model) pair, then
        // compute the diff so we only touch the per-block holder sets that
        // actually changed.
        let new_set: std::collections::HashSet<[u8; 32]> = new_blocks.iter().copied().collect();
        let peer_models = self.peer_prefix_blocks.entry(peer.clone()).or_default();
        let prev: Vec<[u8; 32]> = peer_models
            .get(&model_id)
            .map(|s| s.iter().map(|r| *r.key()).collect())
            .unwrap_or_default();

        let model_index = self
            .cross_node_prefix_index
            .entry(model_id.clone())
            .or_default();

        let mut removed = 0usize;
        for h in &prev {
            if !new_set.contains(h) {
                if let Some(holders) = model_index.get(h) {
                    holders.remove(&peer);
                    let now_empty = holders.is_empty();
                    drop(holders);
                    if now_empty {
                        model_index.remove(h);
                    }
                }
                removed += 1;
            }
        }

        let prev_set: std::collections::HashSet<[u8; 32]> = prev.iter().copied().collect();
        let mut added = 0usize;
        for h in &new_blocks {
            if !prev_set.contains(h) {
                let holders = model_index.entry(*h).or_default();
                holders.insert(peer.clone());
                added += 1;
            }
        }

        // Refresh the reverse index for this (peer, model) pair.
        let new_per_peer = dashmap::DashSet::new();
        for h in &new_blocks {
            new_per_peer.insert(*h);
        }
        if new_per_peer.is_empty() {
            peer_models.remove(&model_id);
        } else {
            peer_models.insert(model_id, new_per_peer);
        }

        (added, removed)
    }

    /// Drop every entry attributed to `peer` from the cross-node prefix-cache
    /// index. Called when a peer disconnects or is evicted — leaving stale
    /// entries would point Phase 2's KV-fetch path at unreachable peers.
    pub fn forget_peer_prefix_blocks(&self, peer: &NodeId) -> usize {
        let Some((_, models)) = self.peer_prefix_blocks.remove(peer) else {
            return 0;
        };
        let mut removed = 0usize;
        for entry in models.iter() {
            let model_id = entry.key();
            let blocks = entry.value();
            if let Some(model_index) = self.cross_node_prefix_index.get(model_id) {
                for hash in blocks.iter() {
                    if let Some(holders) = model_index.get(hash.key()) {
                        holders.remove(peer);
                        let now_empty = holders.is_empty();
                        drop(holders);
                        if now_empty {
                            model_index.remove(hash.key());
                        }
                        removed += 1;
                    }
                }
            }
        }
        removed
    }

    /// Lookup the set of remote peers that announced holding a KV snapshot
    /// for this `(model_id, block_hash)` pair. Empty when no peer has it.
    /// Phase 2 will use this to decide where to fetch from; Phase 1 only
    /// exposes it for tests + diagnostics.
    pub fn cross_node_prefix_holders(
        &self,
        model_id: &crate::types::ModelId,
        block_hash: &[u8; 32],
    ) -> Vec<NodeId> {
        self.cross_node_prefix_index
            .get(model_id)
            .and_then(|m| {
                m.get(block_hash)
                    .map(|s| s.iter().map(|r| r.clone()).collect())
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelId;

    fn make_mgmt() -> ModelMgmt {
        ModelMgmt {
            acquisition_progress: DashMap::new(),
            hf_sources: DashMap::new(),
            auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
            auto_manage_enabled: AtomicBool::new(false),
            auto_manage_default_model_cap: AtomicU32::new(0),
            model_auto_manage_policies: DashMap::new(),
            hf_probe_cache: DashMap::new(),
            peer_shard_downloads: DashMap::new(),
            download_cancel_flags: DashMap::new(),
            model_trust: DashMap::new(),
            loading_models: DashMap::new(),
            locked_shards: DashMap::new(),
            shard_p2p_failed: dashmap::DashSet::new(),
            model_request_counts: DashMap::new(),
            resource_schedule: RwLock::new(Default::default()),
            prune_history: RwLock::new(VecDeque::new()),
            parallax_stability: DashMap::new(),
            cross_node_prefix_index: DashMap::new(),
            peer_prefix_blocks: DashMap::new(),
            p2p_download_permits: DashMap::new(),
            wishlist: arc_swap::ArcSwap::from_pointee(
                crate::model::auto_manage::wishlist::Wishlist::default(),
            ),
        }
    }

    fn h(b: u8) -> [u8; 32] {
        [b; 32]
    }

    #[test]
    fn replace_inserts_announced_blocks() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let (added, removed) =
            m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2), h(3)]);
        assert_eq!(added, 3);
        assert_eq!(removed, 0);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(2)), vec![peer]);
    }

    #[test]
    fn replace_supersedes_previous_announcement_from_same_peer() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2), h(3)]);
        // Second announcement drops blocks 2/3 and adds 4.
        let (added, removed) =
            m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(4)]);
        assert_eq!(added, 1);
        assert_eq!(removed, 2);
        assert_eq!(
            m.cross_node_prefix_holders(&model, &h(1)),
            vec![peer.clone()]
        );
        assert!(m.cross_node_prefix_holders(&model, &h(2)).is_empty());
        assert_eq!(m.cross_node_prefix_holders(&model, &h(4)), vec![peer]);
    }

    #[test]
    fn multiple_peers_share_block_holders() {
        let m = make_mgmt();
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(1), h(2)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(2), h(3)]);
        let mut holders = m.cross_node_prefix_holders(&model, &h(2));
        holders.sort_by_key(|n| n.0);
        assert_eq!(holders, vec![p1.clone(), p2.clone()]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(1)), vec![p1]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(3)), vec![p2]);
    }

    #[test]
    fn forget_peer_drops_only_their_entries() {
        let m = make_mgmt();
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(1), h(2)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(2), h(3)]);
        let removed = m.forget_peer_prefix_blocks(&p1);
        assert_eq!(removed, 2);
        assert!(m.cross_node_prefix_holders(&model, &h(1)).is_empty());
        assert_eq!(m.cross_node_prefix_holders(&model, &h(2)), vec![p2.clone()]);
        assert_eq!(m.cross_node_prefix_holders(&model, &h(3)), vec![p2]);
    }

    #[test]
    fn empty_announce_drops_all_entries_for_peer_model() {
        let m = make_mgmt();
        let peer = NodeId([1u8; 32]);
        let model = ModelId("m".into());
        let _ = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![h(1), h(2)]);
        let (added, removed) = m.replace_peer_prefix_blocks(peer.clone(), model.clone(), vec![]);
        assert_eq!(added, 0);
        assert_eq!(removed, 2);
        assert!(m.cross_node_prefix_holders(&model, &h(1)).is_empty());
        assert!(m.cross_node_prefix_holders(&model, &h(2)).is_empty());
    }

    /// Item 8 Phase 4: simulates the core resolver logic inside
    /// `spawn_prefix_probe_handler` against a scenario with three peers
    /// holding progressively longer prefix matches. Validates that the
    /// longest-prefix hit wins AND that a trust-gate filter correctly
    /// excludes low-trust peers even when they hold a longer match.
    /// We mirror the probe-handler's inline logic here rather than
    /// constructing a SharedState (which needs a full runtime harness).
    #[test]
    fn probe_resolver_picks_longest_prefix_above_trust_floor() {
        use crate::types::PrefixBlockEntry;
        let m = make_mgmt();
        let model = ModelId("m".into());
        let high_trust_peer = NodeId([1u8; 32]);
        let med_trust_peer = NodeId([2u8; 32]);
        let low_trust_peer = NodeId([3u8; 32]);

        // Manifest over a prompt with 3 blocks. The low-trust peer has
        // ALL three blocks (longest match); the medium-trust peer has the
        // first two; the high-trust peer only the first.
        let blocks = [
            PrefixBlockEntry {
                block_hash: h(10),
                token_count: 64,
            },
            PrefixBlockEntry {
                block_hash: h(20),
                token_count: 128,
            },
            PrefixBlockEntry {
                block_hash: h(30),
                token_count: 192,
            },
        ];
        let _ = m.replace_peer_prefix_blocks(high_trust_peer.clone(), model.clone(), vec![h(10)]);
        let _ =
            m.replace_peer_prefix_blocks(med_trust_peer.clone(), model.clone(), vec![h(10), h(20)]);
        let _ = m.replace_peer_prefix_blocks(
            low_trust_peer.clone(),
            model.clone(),
            vec![h(10), h(20), h(30)],
        );

        // Trust scores: low-trust peer is below threshold 0.4.
        let trust = |peer: &NodeId| -> f32 {
            if peer == &high_trust_peer {
                0.9
            } else if peer == &med_trust_peer {
                0.6
            } else {
                0.2
            }
        };
        let trust_min: f32 = 0.4;

        // Resolver mirror: walk manifest longest-first, pick a peer above
        // the trust floor.
        let mut best: Option<(NodeId, [u8; 32], u32)> = None;
        if let Some(model_index) = m.cross_node_prefix_index.get(&model) {
            for entry in blocks.iter().rev() {
                if let Some(holders) = model_index.get(&entry.block_hash) {
                    let candidates: Vec<NodeId> = holders
                        .iter()
                        .map(|r| r.clone())
                        .filter(|n| trust(n) >= trust_min)
                        .collect();
                    if !candidates.is_empty() {
                        // Sort by NodeId for determinism — the actual
                        // resolver uses latency EMA as tiebreak, absent
                        // here, so pick first sorted.
                        let mut c = candidates;
                        c.sort_by_key(|n| n.0);
                        best = Some((c[0].clone(), entry.block_hash, entry.token_count));
                        break;
                    }
                }
            }
        }
        // Low-trust peer holds the longest match (h(30) at 192 tokens)
        // but is below the floor — so we should fall back to h(20) at
        // 128 tokens, served by the medium-trust peer.
        let (peer, hash, token_count) = best.expect("should find a match");
        assert_eq!(peer, med_trust_peer);
        assert_eq!(hash, h(20));
        assert_eq!(token_count, 128);
    }

    /// Phase 4: when ALL candidate peers are below the trust threshold,
    /// the resolver must return no match — the fetcher falls through to
    /// a full local prefill instead of risking a poisoned KV.
    #[test]
    fn probe_resolver_returns_none_when_all_peers_below_trust_floor() {
        use crate::types::PrefixBlockEntry;
        let m = make_mgmt();
        let model = ModelId("m".into());
        let p1 = NodeId([1u8; 32]);
        let p2 = NodeId([2u8; 32]);
        let _ = m.replace_peer_prefix_blocks(p1.clone(), model.clone(), vec![h(10)]);
        let _ = m.replace_peer_prefix_blocks(p2.clone(), model.clone(), vec![h(10)]);
        let blocks = [PrefixBlockEntry {
            block_hash: h(10),
            token_count: 64,
        }];
        let trust = |_: &NodeId| -> f32 { 0.1 };
        let trust_min: f32 = 0.5;

        let mut best: Option<(NodeId, [u8; 32], u32)> = None;
        if let Some(model_index) = m.cross_node_prefix_index.get(&model) {
            for entry in blocks.iter().rev() {
                if let Some(holders) = model_index.get(&entry.block_hash) {
                    let candidates: Vec<NodeId> = holders
                        .iter()
                        .map(|r| r.clone())
                        .filter(|n| trust(n) >= trust_min)
                        .collect();
                    if !candidates.is_empty() {
                        let mut c = candidates;
                        c.sort_by_key(|n| n.0);
                        best = Some((c[0].clone(), entry.block_hash, entry.token_count));
                        break;
                    }
                }
            }
        }
        assert!(best.is_none());
    }
}
