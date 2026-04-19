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
}
