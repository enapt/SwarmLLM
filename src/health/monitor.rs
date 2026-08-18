use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{NetworkCommand, RebalanceEvent, SwarmMessage};

/// Periodic health monitoring for the node and its peers.
///
/// Sends health pings to known peers and tracks their response latencies.
/// Peers that fail to respond after multiple missed intervals are considered offline.
/// When peers are detected as stale, a `RebalanceEvent::PeerLeft` is emitted.
pub struct HealthMonitor {
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    rebalance_tx: mpsc::Sender<RebalanceEvent>,
    shutdown_rx: watch::Receiver<bool>,
    /// Track last broadcast shard set for delta compression.
    /// Full re-announce only when set changes or every FULL_REANNOUNCE ticks.
    last_announced_shards: std::collections::HashSet<crate::types::ShardId>,
    /// Counter for periodic full re-announce (ensures late-joining peers get data).
    shard_announce_counter: u64,
    /// Per-acquisition liveness tracker: model_id → (last bytes seen, when seen).
    /// If bytes don't advance for STALL_THRESHOLD, the acquisition is reconciled
    /// against disk (mark Complete if shards present, Failed otherwise).
    acq_liveness: std::collections::HashMap<crate::types::ModelId, (u64, std::time::Instant)>,
    /// Per-peer-shard-download liveness tracker: (ShardId, NodeId) → (last pct, when seen).
    /// Removed if pct doesn't change for STALL_THRESHOLD — peer's gossip stopped.
    peer_dl_liveness: std::collections::HashMap<
        (crate::types::ShardId, crate::types::NodeId),
        (u32, std::time::Instant),
    >,
    /// When this monitor started, for the inbound-reachability grace period.
    started_at: std::time::Instant,
    /// Latch so the WSL firewall warning is emitted at most once per run, and
    /// the check stops costing anything once it has resolved either way.
    wsl_firewall_warned: bool,
}

/// How long to wait before concluding that nothing can dial in.
///
/// Long enough that a node which has simply not been found yet is never
/// accused of a firewall problem: mDNS answers in seconds, but a peer that has
/// to come via the DHT or a bootstrap round can take minutes to dial back.
const WSL_FIREWALL_GRACE: std::time::Duration = std::time::Duration::from_secs(600);

/// How long a download tracking entry can sit unchanged before being treated
/// as stalled. Generous enough to tolerate slow peers and HF rate limiting.
const DOWNLOAD_STALL_THRESHOLD: Duration = Duration::from_secs(90);

/// How often to send health pings.
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// Number of missed pings before a peer is considered dead.
const MAX_MISSED_PINGS: u32 = 3;

/// Drop `HedgeTracker.stats` entries that haven't seen a new observation
/// in this many ms. Departed peers leave dead entries that would otherwise
/// accumulate one per (model × segment) they ever served. 1h matches the
/// scale at which a peer being gone is treated as "really gone" by the
/// scheduler — short reconnects don't lose useful latency history.
const HEDGE_STATS_MAX_AGE_MS: u64 = 3_600_000;

/// Drop a peer's measured speed after this long without a fresh observation.
///
/// Serves two purposes: departed peers stop accumulating (three dead entries
/// were live on 2026-08-01), and a peer whose estimate made it un-routable
/// gets a clean slate rather than being permanently de-ranked by a figure that
/// can only be refreshed by routing to it. Matches the hedge-tracker horizon.
const PEER_SPEED_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(3_600);

/// Drop `PrefetchOrchestrator.histories` entries whose last activity is
/// older than this. Matches the KV-cache session expiry (10 min): a
/// session that has been idle longer than the KV-cache TTL has no usable
/// cache to prefetch against anyway.
const PREFETCH_HISTORY_MAX_IDLE_MS: u64 = 600_000;

/// Verdict of the "can anything reach this node" check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InboundCheck {
    /// A remote peer has dialled us — inbound demonstrably works.
    Reachable,
    /// Not enough evidence yet: still inside the grace period, or we have not
    /// met anyone who *could* dial back.
    KeepWaiting,
    /// Connected to peers for a good while and nothing has ever dialled in.
    Blocked,
}

/// Decide whether inbound connections are reaching this node.
///
/// Split out from `maybe_warn_wsl_firewall` because the decision is the part
/// worth pinning: the first version of this warning had no decision at all — it
/// fired unconditionally at config-load on every WSL2 mirrored node, including
/// ones whose ports were already open and verified working.
///
/// Both negative arms matter. Outbound connections prove nothing about inbound,
/// so peers alone are not evidence of reachability; and *no* peers is a
/// discovery problem, where a firewall message would send someone off after the
/// wrong thing entirely.
pub(crate) fn inbound_warning_decision(
    observed_inbound: bool,
    uptime: std::time::Duration,
    connected_peers: usize,
) -> InboundCheck {
    if observed_inbound {
        return InboundCheck::Reachable;
    }
    if uptime < WSL_FIREWALL_GRACE || connected_peers == 0 {
        return InboundCheck::KeepWaiting;
    }
    InboundCheck::Blocked
}

impl HealthMonitor {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        rebalance_tx: mpsc::Sender<RebalanceEvent>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            shared_state,
            network_tx,
            rebalance_tx,
            shutdown_rx,
            last_announced_shards: std::collections::HashSet::new(),
            shard_announce_counter: 0,
            acq_liveness: std::collections::HashMap::new(),
            peer_dl_liveness: std::collections::HashMap::new(),
            started_at: std::time::Instant::now(),
            wsl_firewall_warned: false,
        }
    }

    /// Compute how many base ticks (30s each) between heavy gossip broadcasts.
    /// Scales with log(peer_count) to reduce bandwidth at large network sizes.
    ///   ≤10 peers:  every tick   (30s)
    ///   ~100 peers: every 2 ticks (60s)
    ///   ~1K peers:  every 4 ticks (120s)
    ///   ~10K peers: every 8 ticks (240s)
    fn gossip_broadcast_interval(&self) -> u64 {
        let peer_count = self.shared_state.peer_registry.len();
        if peer_count <= 10 {
            1
        } else {
            // floor(log2(peer_count / 5)), clamped to [1, 10]
            let ratio = peer_count / 5;
            // checked_ilog2 returns None for 0 — protects against an
            // upstream guard-change accidentally letting ratio=0 reach
            // here, where the bit_length-1 expression would underflow.
            let log2 = ratio.checked_ilog2().unwrap_or(0) as u64;
            log2.clamp(1, 10)
        }
    }

    /// Run the health monitoring loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        let mut interval = tokio::time::interval(PING_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut nonce: u64 = 0;

        tracing::info!(target: "swarmllm::health::monitor", "HealthMonitor running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!(target: "swarmllm::health::monitor", "HealthMonitor shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    nonce = nonce.wrapping_add(1);

                    // Health pings and peer liveness: always run every 30s (critical)
                    self.send_health_ping(nonce).await;
                    self.check_peer_health().await;

                    // Heavy gossip broadcasts: scale frequency with network size.
                    // At ≤10 peers, every 30s (same as before).
                    // At 1K+ peers, every ~120s to reduce bandwidth.
                    let broadcast_every = self.gossip_broadcast_interval();
                    if nonce.is_multiple_of(broadcast_every) {
                        self.broadcast_capabilities().await;
                        self.broadcast_manifests().await;
                        self.broadcast_region_summary().await;
                        self.broadcast_wishlist_announcement().await;
                        self.broadcast_pool_model_availability().await;
                    }

                    self.maybe_warn_wsl_firewall();

                    // Cleanup tasks: run every tick (cheap, local-only)
                    self.cleanup_acquisition_progress();
                    self.cleanup_stale_peer_shard_downloads();
                    self.cleanup_stale_channels();
                    self.cleanup_stale_peer_id_map();
                    // Cleanup expired anti-gaming rate limit entries.
                    // try_lock so a contending API/credit-ledger call can't stall the
                    // health monitor; cleanup is idempotent and the next tick retries.
                    if let Ok(mut ag) = self.shared_state.credits.anti_gaming.try_lock() {
                        ag.cleanup();
                    }
                    // Decay trust scores toward default (0.5) on each health ping cycle
                    self.shared_state.credits.trust_manager.decay_all(&self.shared_state.peer_registry);
                    // Clean up stale AllReduce/RingChunk entries (receiver dropped/timed out)
                    self.shared_state.allreduce_registry.cleanup_stale();
                    self.shared_state.ring_chunk_registry.cleanup_stale();
                    // R139 Tier 4K — evict incomplete chunk assemblies whose last
                    // chunk arrived past the TTL. Without this a stuck or abandoned
                    // sender would leak `pending_activation_chunks` slots.
                    let chunk_ttl = self.shared_state.config.inference.streaming_chunk_assembly_ttl_secs;
                    let evicted = self.shared_state.sweep_stale_chunk_assemblies(chunk_ttl);
                    if evicted > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            evicted, ttl_secs = chunk_ttl,
                            "Swept stale chunk assemblies"
                        );
                    }
                    // NETWORKING_PLAN Phase 1 — bound the learned relay-route
                    // table + relay-forward rate counters so they can't grow
                    // unbounded under peer churn.
                    self.shared_state.sweep_stale_relay_state();
                    // R142 — bound SWARM-SPEC Layer 2/3 in-memory state.
                    // `HedgeTracker.stats` accumulates one entry per
                    // (model × segment × holder) triple ever observed; peers
                    // that have left the swarm stop receiving observations
                    // but their entries stick. `PrefetchOrchestrator.histories`
                    // grows one entry per unique session id (UUID — unbounded
                    // cardinality). Both have eviction methods; wire them here.
                    let now_ms = crate::types::unix_now_secs().saturating_mul(1000);
                    let hedge_evicted = self
                        .shared_state
                        .metrics
                        .hedge_tracker
                        .evict_stale(now_ms, HEDGE_STATS_MAX_AGE_MS);
                    if hedge_evicted > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            evicted = hedge_evicted,
                            max_age_ms = HEDGE_STATS_MAX_AGE_MS,
                            "Evicted stale hedge-tracker entries"
                        );
                    }
                    // Peer speed estimates go stale the same way, with an extra
                    // twist: the estimate is only refreshed when we route to a
                    // peer, and we stop routing to peers that look slow — so a
                    // bad figure can never be corrected ("the routing
                    // ratchet"). Dropping it lets the peer be tried afresh.
                    let speed_evicted = self
                        .shared_state
                        .evict_stale_peer_speed(PEER_SPEED_MAX_AGE);
                    if speed_evicted > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            evicted = speed_evicted,
                            max_age_secs = PEER_SPEED_MAX_AGE.as_secs(),
                            "Evicted stale peer-speed entries"
                        );
                    }
                    let prefetch_evicted = self
                        .shared_state
                        .metrics
                        .prefetch_orchestrator
                        .evict_idle(now_ms, PREFETCH_HISTORY_MAX_IDLE_MS);
                    if prefetch_evicted > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            evicted = prefetch_evicted,
                            max_idle_ms = PREFETCH_HISTORY_MAX_IDLE_MS,
                            "Evicted idle prefetch session histories"
                        );
                    }
                    // Suspend idle Claude Code sessions and warn about upcoming timeouts
                    #[cfg(feature = "claude-subscription")]
                    crate::api::claude_session::SessionManager::global()
                        .cleanup_stale(&self.shared_state)
                        .await;
                }
            }
        }

        Ok(())
    }

    async fn send_health_ping(&self, nonce: u64) {
        let timestamp = crate::types::unix_now_secs();

        let active_request_count = self.shared_state.active_pipelines.len() as u32;
        let node_id = Some(self.shared_state.identity.node_id().clone());
        let msg = SwarmMessage::HealthPing {
            nonce,
            timestamp,
            node_id,
            active_request_count,
        };

        // Record the send so the pong can be turned into a live RTT. Stored
        // before the send: a pong cannot arrive before the ping leaves, but the
        // reverse ordering would race under a fast LAN peer.
        *self.shared_state.last_health_ping.lock() = Some((nonce, std::time::Instant::now()));

        if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
            tracing::warn!(error = %e, "Failed to send health ping");
        }
    }

    async fn broadcast_capabilities(&mut self) {
        let node_id = self.shared_state.identity.node_id().clone();

        // Gather hosted shards using the reverse-index for O(1) lookup.
        let mut hosted_shards = self.shared_state.model_registry.shards_for_node(&node_id);
        // Never report a backup-copy model as hosted — it drives DHT provider
        // announcements (StartProviding) and shows up in peers' hosted_models
        // lists under a name no one can resolve. Both were reported still
        // leaking on v0.3.15 despite the manifest guard.
        hosted_shards.retain(|s| !crate::model::manifest::is_backup_artifact_id(&s.model_id.0));

        // Drop any shard whose file is no longer on disk, and stop claiming it.
        //
        // The registry is populated once at startup and then updated by events;
        // nothing watches the filesystem. A user who frees space the only way
        // the software offers — deleting the folder, since there is no remove
        // command — left the registry asserting shards that no longer exist.
        // That is not merely cosmetic: THIS list is what the swarm announce
        // below advertises, so the node kept offering peers work it could not
        // do, and every surface reading the registry (the `privacy` command,
        // the dashboard, the scheduler) reported it as held. Reported
        // 2026-08-02; the reporter found it via `swarmllm privacy` still saying
        // "both ends are already on this machine" for a deleted model.
        //
        // Correcting the registry here rather than at each reader is what makes
        // one check fix all of them, and it feeds the existing retraction path:
        // `shards_changed` fires below and re-announces with
        // `complete_for_models`, which is what actually removes the claim from
        // peers.
        //
        // Existence only, deliberately. A shard being downloaded is written at
        // its final path and is legitimately incomplete for a while; size and
        // hash are the accept gate's job (`verify_shard`), not this one. Absent
        // is the only signal that cannot be a transient.
        let store = self.shared_state.shard_store();
        let mut vanished = Vec::new();
        hosted_shards.retain(|s| {
            if store.shard_file_present(s) {
                true
            } else {
                vanished.push(s.clone());
                false
            }
        });
        for shard_id in vanished {
            tracing::warn!(
                model = %shard_id.model_id,
                index = shard_id.index,
                "Shard file is gone from disk — no longer claiming it to the swarm"
            );
            self.shared_state
                .model_registry
                .remove_shard_holder(&shard_id, &node_id);
            // The split-model cache asserts the same thing this reconcile just
            // disproved — "we can serve this model locally" — and nothing else
            // clears it when files vanish (`auto_manage::scan` only evicts for
            // models that gained shards). Left behind, it wins the local fast
            // path in the API layer, spawns a worker, and turns a servable
            // request into a 404 while peers hold every shard. Observed live
            // 2026-08-04, with this very warning already in the log.
            self.shared_state.evict_split_models(&shard_id.model_id);
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "model",
                    "shard_vanished",
                    format!(
                        "A piece of {} was removed from disk — no longer offering it",
                        shard_id.model_id
                    ),
                )
                .with_model(shard_id.model_id.0.clone()),
            );
        }

        // If no shards from registry but we have a loaded model (and no shard_range),
        // represent the full model as shard index 0.
        if hosted_shards.is_empty() && self.shared_state.config.inference.shard_range.is_none() {
            if let Some(info) = self.shared_state.loaded_model_info.read().await.as_ref() {
                // Slugified, NOT the raw display name. This announcement is
                // how peers learn what we hold, and they match it against a
                // manifest id — which `generate_and_register_local_manifest`
                // derives from this same field via the same helper. Sending
                // "Llama 3.2 3B Instruct" where the manifest says
                // "llama-3.2-3b-instruct" made this node invisible as a holder
                // and put a phantom entry in every peer's model list (#310).
                hosted_shards.push(crate::types::ShardId {
                    model_id: crate::types::ModelId(crate::types::slugify_model_name(&info.name)),
                    index: 0,
                });
            }
        }

        let gpu_info = self.shared_state.gpu_info.as_ref().map(|g| {
            let bandwidth = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
            // Ask the card, here, every broadcast.
            //
            // `SharedState::gpu_info.vram_free_mb` is set ONCE at startup and
            // hardcoded to 0 there (`daemon/mod.rs`), so every node in the swarm
            // advertised zero free VRAM for as long as this field has existed.
            // Nothing read it, so nothing went wrong — until something did, and
            // then it silently answered "no room" for every peer, everywhere.
            //
            // Free VRAM is the one figure here that is meaningless stale: it is
            // exactly the quantity that changes as models load and unload. This
            // capability is rebuilt on every broadcast, so querying it now costs
            // one `nvidia-smi` per cycle and is the only way the number can be
            // true. `None` (unreadable) advertises 0, which reads as "no room"
            // — the safe direction for anyone deciding whether to send us work.
            let free = crate::model::auto_manage::vram::query_gpu_vram_free_mb().unwrap_or(0);
            crate::types::GpuInfo {
                name: g.name.clone(),
                vram_total_mb: g.vram_total_mb,
                vram_available_mb: free,
                compute_capability: None,
                memory_bandwidth_gbps: bandwidth,
            }
        });

        // Use real uptime so message content changes each broadcast (avoids GossipSub dedup)
        let uptime_seconds = {
            let stats = self.shared_state.metrics.node_stats.read().await;
            (chrono::Utc::now() - stats.uptime_start)
                .num_seconds()
                .max(0) as u64
        };

        // Populate real system metrics. sysinfo does blocking filesystem
        // reads (/proc/*); route them to the dedicated blocking pool via
        // spawn_blocking instead of block_in_place — block_in_place parks
        // a Tokio worker thread for the duration of the syscalls, which
        // forces the runtime to spin up a replacement on every 30s tick.
        let data_dir = self.shared_state.config.node.data_dir.clone();
        let (ram_total_mb, ram_available_mb, disk_available_mb) =
            tokio::task::spawn_blocking(move || {
                let mut sys = sysinfo::System::new();
                sys.refresh_memory();
                let ram_total = sys.total_memory() / (1024 * 1024);
                let ram_avail = sys.available_memory() / (1024 * 1024);

                let disks = sysinfo::Disks::new_with_refreshed_list();
                let disk_avail: u64 = disks
                    .list()
                    .iter()
                    .filter(|d| data_dir.starts_with(d.mount_point()))
                    .max_by_key(|d| d.mount_point().as_os_str().len())
                    .map(|d| d.available_space() / (1024 * 1024))
                    .unwrap_or_else(|| {
                        disks
                            .list()
                            .iter()
                            .map(|d| d.available_space() / (1024 * 1024))
                            .sum()
                    });
                (ram_total, ram_avail, disk_avail)
            })
            .await
            .unwrap_or_else(|e| {
                // sysinfo block can panic on malformed /proc on certain
                // container environments. Log so a recurring zero-broadcast
                // doesn't look like genuine resource exhaustion to peers.
                tracing::warn!(error = %e, "Hardware-detection task failed; broadcasting zeros for this tick");
                (0, 0, 0)
            });

        let est_tokens_per_sec_7b = gpu_info
            .as_ref()
            .map(|g| {
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(
                    g.memory_bandwidth_gbps,
                    true,
                )
            })
            .unwrap_or_else(|| {
                // CPU-only: assume ~50 GB/s DDR4/5 bandwidth
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(50.0, false)
            });

        // Top-N observed-latency snapshot, ordered by the trust we have in
        // each *observed peer* (not the sender). Gives receivers a pre-warm
        // Parallax DP signal so newly-joining nodes don't need to route
        // requests through a peer to price it. Kept to 32 entries → ≈1.2 KB
        // extra per broadcast, well under the 4 MB gossip cap.
        const MAX_OBSERVED: usize = 32;
        let observed_latencies = {
            let mut entries: Vec<(crate::types::NodeId, f32, f32)> = self
                .shared_state
                .metrics
                .peer_speed
                .iter()
                // Gossip carries the ranking-scale per-layer figure; a peer we
                // have only ever prefilled through still has one.
                .filter_map(|r| {
                    r.value()
                        .ranking_ms_per_layer()
                        .map(|ms| (r.key().clone(), ms))
                })
                .map(|(peer_id, ms)| {
                    let trust = self
                        .shared_state
                        .peer_registry
                        .get(&peer_id)
                        .map(|p| p.trust_score)
                        .unwrap_or(0.5);
                    (peer_id, ms, trust)
                })
                .collect();
            // Higher trust first. Stable ordering (partial_cmp handles NaN by
            // treating as Less — but trust_scores are clamped [0,1]).
            entries.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(MAX_OBSERVED);
            entries
                .into_iter()
                .map(|(peer, ms_per_layer, _)| crate::types::LatencyObservation {
                    peer,
                    ms_per_layer,
                })
                .collect::<Vec<_>>()
        };

        let cap = crate::types::NodeCapability {
            // OS family only — see the field docs on why not a version string.
            os: Some(std::env::consts::OS.to_string()),
            node_id: node_id.clone(),
            gpu: gpu_info,
            ram_total_mb,
            ram_available_mb,
            disk_available_mb,
            bandwidth_mbps: 0.0,
            hosted_shards: hosted_shards.clone(),
            // From the runtime mirror: this is what we tell the swarm we are
            // willing to do, so a level the user changed an hour ago must not
            // keep being advertised as the one we booted with.
            max_contribution: self.shared_state.contribution().into(),
            uptime_seconds,
            version: env!("CARGO_PKG_VERSION").to_string(),
            region: self.shared_state.effective_region().await,
            est_tokens_per_sec_7b,
            observed_latencies,
            // Advertise willingness to relay inference for un-connectable peer
            // pairs (NETWORKING_PLAN Phase 1). `--anchor` implies it; any node
            // can opt in via `network.relay_forwarding`.
            relay_capable: self.shared_state.relay_forwarding_enabled(),
            // Label ourselves an anchor so peers can show it. An anchor holds
            // nothing and serves nothing by design, which looks identical to a
            // broken node in a peer list.
            anchor_mode: self.shared_state.config.node.anchor_mode,
            // Advertise the protocol epoch + the optional features this build
            // implements, so peers negotiate new message types additively.
            protocol_version: swarmllm_types::PROTOCOL_VERSION,
            features: swarmllm_types::features::ALL,
            // NETWORKING_PLAN Phase 3 — advertise the relay-capable peers we are
            // connected to, so a peer that can't reach us directly can pick a
            // relay we share. Bounded to keep the capability gossip small.
            relay_reservations: {
                const MAX_RELAY_RESERVATIONS: usize = 8;
                self.shared_state
                    .connected_node_ids
                    .iter()
                    .filter(|n| {
                        self.shared_state
                            .peer_registry
                            .get(n.key())
                            .and_then(|p| p.capability.as_ref().map(|c| c.relay_capable))
                            .unwrap_or(false)
                    })
                    .take(MAX_RELAY_RESERVATIONS)
                    .map(|n| n.key().clone())
                    .collect()
            },
        };

        // Keep a copy before it goes on the wire, so local surfaces can show
        // this node using exactly what peers are told about it.
        self.shared_state
            .local_capability
            .store(Some(std::sync::Arc::new(cap.clone())));
        let msg = NetworkCommand::Broadcast(SwarmMessage::NodeCapabilityUpdate(cap));
        if let Err(e) = self.network_tx.send(msg).await {
            tracing::debug!(error = %e, "DIAG: failed to broadcast capability update");
        }

        // Delta-compressed shard announcements: only broadcast when shard set
        // changes, or every 10 broadcast cycles as a full re-announce (ensures
        // late-joining peers get the full picture). At 10K peers with scaled
        // gossip interval (~240s), full re-announce happens every ~40 min.
        if !hosted_shards.is_empty() {
            let current_set: std::collections::HashSet<_> = hosted_shards.iter().cloned().collect();
            let shards_changed = current_set != self.last_announced_shards;
            self.shard_announce_counter += 1;
            // Full re-announce every 10 broadcast cycles so late-joining peers
            // eventually discover our shards even if nothing changed.
            let periodic_reannounce = self.shard_announce_counter.is_multiple_of(10);

            if shards_changed || periodic_reannounce {
                let shard_count = hosted_shards.len();
                // Complete for every model represented. `shards_changed` above
                // fires when shards are deleted, so this is what actually
                // retracts them on peers — previously it re-sent the smaller
                // set and receivers merged it, keeping the deleted shards.
                let complete_for_models: Vec<crate::types::ModelId> = hosted_shards
                    .iter()
                    .map(|s| s.model_id.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let announce = crate::types::ShardAnnounce {
                    node_id,
                    shards: hosted_shards,
                    timestamp: chrono::Utc::now(),
                    complete_for_models,
                };
                let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
                if let Err(e) = self.network_tx.send(msg).await {
                    tracing::debug!(error = %e, shard_count, "DIAG: failed to broadcast shard announce");
                }
                self.last_announced_shards = current_set;
            }
        }
    }

    /// Broadcast model manifests and HF sources so peers can discover and acquire models.
    async fn broadcast_manifests(&self) {
        let our_id = self.shared_state.identity.node_id().clone();
        // Models we published OR hold a shard of. A publisher-only filter here
        // silently stopped model discovery for the whole swarm — see
        // `ModelRegistry::manifests_to_gossip`.
        for manifest in self
            .shared_state
            .model_registry
            .manifests_to_gossip(&our_id)
        {
            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelManifest(manifest.clone()));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast manifest");
            }

            // Also broadcast HfSourceGossip so late-joining peers discover the HF source
            if let Some(hf_source) = self.shared_state.models.hf_sources.get(&manifest.id) {
                let gossip = crate::types::HfSourceGossip {
                    model_id: manifest.id.clone(),
                    repo_id: hf_source.repo_id.clone(),
                    filename: hf_source.filename.clone(),
                    publisher: our_id.clone(),
                    mmproj_filename: hf_source.mmproj_filename.clone(),
                };
                let msg = NetworkCommand::Broadcast(SwarmMessage::HfSourceGossip(gossip));
                if let Err(e) = self.network_tx.send(msg).await {
                    tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast HF source");
                }
            }
        }
    }

    /// Broadcast compact per-region shard summaries and demand gossip.
    /// Published on every 30s tick to `swarm/regions` topic.
    async fn broadcast_region_summary(&self) {
        // Determine our region — skip if unknown. Canonical resolver (configured
        // region wins, else IP-detected) so this gossip agrees with the capacity
        // announcement and WS region counts.
        let our_region = match self.shared_state.effective_region().await {
            Some(r) => r.to_uppercase(),
            None => return, // No region — nothing to broadcast
        };

        let our_id = self.shared_state.identity.node_id().clone();
        let now_ms = crate::types::unix_now_ms();

        // Count same-region peers (including self)
        let mut region_node_count: u32 = 1; // self
        for peer in self.shared_state.peer_registry.iter() {
            if let Some(ref cap) = peer.value().capability {
                if let Some(ref r) = cap.region {
                    if r.to_uppercase() == our_region {
                        region_node_count = region_node_count.saturating_add(1);
                    }
                }
            }
        }

        // Build a set of same-region node IDs once to avoid O(holders) peer_registry
        // lookups per shard (was O(models × shards × holders), now O(models × shards)).
        let same_region_nodes: std::collections::HashSet<crate::types::NodeId> = {
            let mut set = std::collections::HashSet::new();
            set.insert(our_id.clone());
            for entry in self.shared_state.peer_registry.iter() {
                if let Some(ref cap) = entry.capability {
                    if let Some(ref r) = cap.region {
                        if r.to_uppercase() == our_region {
                            set.insert(entry.key().clone());
                        }
                    }
                }
            }
            set
        };

        // For each model, count same-region shard holders
        for manifest in self.shared_state.model_registry.models() {
            let mut shard_counts: Vec<(u32, u32)> = Vec::new();
            for shard_info in &manifest.shards {
                let sid = crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: shard_info.index,
                };
                let holders = self.shared_state.model_registry.shard_holders(&sid);
                let regional_count = holders
                    .iter()
                    .filter(|h| same_region_nodes.contains(h))
                    .count() as u32;
                shard_counts.push((shard_info.index, regional_count));
            }

            if shard_counts.is_empty() {
                continue;
            }

            let summary = crate::types::RegionShardSummary {
                region: our_region.clone(),
                model_id: manifest.id.clone(),
                shard_counts,
                region_node_count,
                publisher: our_id.clone(),
                timestamp_ms: now_ms,
            };

            // Also update our own shared state
            let key = (our_region.clone(), manifest.id.clone());
            self.shared_state
                .region_shard_summaries
                .insert(key, summary.clone());

            let msg = NetworkCommand::Broadcast(SwarmMessage::RegionShardSummary(summary));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "Failed to broadcast region summary");
            }
        }

        // Broadcast demand gossip for models with recent requests
        for entry in self.shared_state.region_demand.iter() {
            let (model_id, region) = entry.key();
            let rate = *entry.value();
            if rate < 0.01 {
                continue; // Don't gossip negligible demand
            }
            let demand = crate::types::ModelDemandGossip {
                model_id: model_id.clone(),
                region: region.clone(),
                decayed_rate: rate,
                window_requests: 0, // Raw count already decayed into rate
                publisher: our_id.clone(),
                timestamp_ms: now_ms,
            };
            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelDemandGossip(demand));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, "Failed to broadcast demand gossip");
            }
        }
    }

    /// R130: cross-pool wishlist gossip publisher. Only runs when
    /// `config.inference.auto_manage.wishlist_gossip_publish` is on.
    /// Pulls the top-K entries from the current local wishlist snapshot
    /// (already capped at `MAX_WISHLIST_ENTRIES`) and broadcasts them.
    /// The receive side is always on — opt-out is publish only, so
    /// privacy-conscious operators still benefit from inbound boost.
    /// R134: cross-pool model availability publisher. Only the pool owner
    /// emits; only fires when `pool.share_model_catalog` is on AND the
    /// pool has at least `share_model_catalog_min_members` members
    /// (k-anonymity floor). Carries the model IDs the pool can currently
    /// serve — derived from the local model registry — at the gossip
    /// granularity that the wishlist announcement already operates at.
    /// Pure discovery; routing across pool boundaries is NOT enabled.
    async fn broadcast_pool_model_availability(&self) {
        // Live config, not the boot snapshot: the operator can turn catalog
        // sharing off, or raise the anonymity floor, and have it hold from the
        // next gossip tick.
        let live = self.shared_state.cfg();
        if !live.pool.share_model_catalog {
            return;
        }
        let min_members = live.pool.share_model_catalog_min_members.max(1) as usize;
        let my_id = self.shared_state.identity.node_id().clone();
        let pool_id = {
            let ps = self.shared_state.credits.pool_state.read().await;
            match ps.as_ref() {
                Some(ps) if ps.pool_id == my_id && ps.members.len() >= min_members => {
                    ps.pool_id.clone()
                }
                _ => return, // not owner, no pool, or below k-anonymity floor
            }
        };

        // The pool serves any model whose shards are locally hosted by
        // any pool member. For privacy + simplicity we use the owner's
        // local model registry as the catalog source — distributing the
        // per-member catalog would expose composition signals.
        let mut model_ids: Vec<crate::types::ModelId> = self
            .shared_state
            .model_registry
            .models()
            .into_iter()
            .map(|m| m.id)
            .collect();
        model_ids.sort_by(|a, b| a.0.cmp(&b.0));
        model_ids.dedup_by(|a, b| a.0 == b.0);
        model_ids.truncate(crate::daemon::dispatch::MAX_POOL_MODEL_ANNOUNCE_ENTRIES);
        if model_ids.is_empty() {
            return;
        }

        let timestamp_ms = crate::types::unix_now_ms();
        let payload = crate::pool::crypto::pool_model_availability_payload(
            &pool_id,
            &model_ids,
            timestamp_ms,
        );
        let owner_signature = self.shared_state.identity.sign(&payload);
        let announce = crate::types::PoolModelAvailability {
            pool_id,
            model_ids,
            timestamp_ms,
            owner_signature,
        };
        let msg = NetworkCommand::Broadcast(SwarmMessage::PoolModelAvailability(announce));
        if let Err(e) = self.network_tx.send(msg).await {
            tracing::debug!(error = %e, "Failed to broadcast pool model availability");
        }
    }

    async fn broadcast_wishlist_announcement(&self) {
        if !self.shared_state.cfg().auto_manage.wishlist_gossip_publish {
            return;
        }
        let snapshot = self.shared_state.models.wishlist.load_full();
        if snapshot.entries.is_empty() {
            return;
        }
        // Cap the announcement to the wire limit; entries are already
        // sorted by score descending in `compute_wishlist`, so a simple
        // truncate gives us the top-K.
        const ANNOUNCE_CAP: usize = 64;
        let entries: Vec<crate::types::WishlistAnnouncementEntry> = snapshot
            .entries
            .iter()
            .take(ANNOUNCE_CAP)
            .filter(|e| e.score > 0)
            .map(|e| crate::types::WishlistAnnouncementEntry {
                model_id: crate::types::ModelId(e.model_id.clone()),
                score: e.score,
            })
            .collect();
        if entries.is_empty() {
            return;
        }
        let announce = crate::types::WishlistAnnouncement {
            publisher: self.shared_state.identity.node_id().clone(),
            entries,
            timestamp_ms: crate::types::unix_now_ms(),
        };
        let msg = NetworkCommand::Broadcast(SwarmMessage::WishlistAnnouncement(announce));
        if let Err(e) = self.network_tx.send(msg).await {
            tracing::debug!(error = %e, "Failed to broadcast wishlist announcement");
        }
    }

    async fn check_peer_health(&self) {
        let now = chrono::Utc::now();
        let timeout =
            chrono::Duration::seconds((PING_INTERVAL.as_secs() * MAX_MISSED_PINGS as u64) as i64);

        // Collect node IDs participating in active inference pipelines —
        // these must not be removed even if they appear stale (long forward passes).
        let mut active_nodes = std::collections::HashSet::new();
        for entry in self.shared_state.active_pipelines.iter() {
            for seg in &entry.value().segments {
                active_nodes.insert(seg.node_id.clone());
            }
        }

        let mut stale_peers = Vec::new();
        // NodeIds whose registry entry looks stale but whose libp2p connection
        // is still live — bumped back to `now` after the iteration completes
        // (can't take a write lock on a DashMap entry while we're holding a
        // read ref via .iter()).
        let mut refresh_peers = Vec::new();

        for entry in self.shared_state.peer_registry.iter() {
            let peer = entry.value();
            let age = now
                .signed_duration_since(peer.last_seen)
                .max(chrono::Duration::zero());
            if age > timeout {
                if active_nodes.contains(entry.key()) {
                    tracing::debug!(
                        peer = %entry.key(),
                        "Peer appears stale but is active in inference pipeline, skipping removal"
                    );
                    continue;
                }
                // Libp2p still has a live connection to this peer. Registry
                // staleness is just silence in the application-level protocol
                // (no recent PEX/gossip), not an actual disconnect — refresh
                // last_seen so it doesn't keep flagging every tick.
                if self.shared_state.connected_node_ids.contains(entry.key()) {
                    tracing::debug!(
                        peer = %entry.key(),
                        age_secs = age.num_seconds(),
                        "Peer registry entry is stale but libp2p connection is live — refreshing last_seen, skipping eviction"
                    );
                    refresh_peers.push(entry.key().clone());
                    continue;
                }
                stale_peers.push(entry.key().clone());
            }
        }

        for nid in refresh_peers {
            if let Some(mut p) = self.shared_state.peer_registry.get_mut(&nid) {
                p.last_seen = now;
            }
        }

        if !stale_peers.is_empty() {
            tracing::warn!(
                stale_count = stale_peers.len(),
                total_peers = self.shared_state.peer_registry.len(),
                active_pipelines = self.shared_state.active_pipelines.len(),
                "DIAG: removing stale peers"
            );
        }
        for peer_id in stale_peers {
            self.shared_state.peer_registry.remove(&peer_id);
            // Clean up stale peer from model_registry shard holders
            self.shared_state
                .model_registry
                .remove_peer_from_all_shards(&peer_id);
            tracing::info!(peer = %peer_id, "Removed stale peer (and shard registry entries)");
            // Signal the rebalancer that a peer has left
            if self
                .rebalance_tx
                .try_send(RebalanceEvent::PeerLeft(peer_id))
                .is_err()
            {
                self.shared_state
                    .metrics
                    .channel_metrics
                    .rebalance
                    .record_dropped();
            } else {
                self.shared_state
                    .metrics
                    .channel_metrics
                    .rebalance
                    .record_sent();
            }
        }
    }

    /// Remove stale pending_layer_results (closed oneshot channels) and
    /// streaming_token_txs (closed mpsc channels) to prevent memory leaks.
    fn cleanup_stale_channels(&self) {
        // pending_layer_results: remove entries where the receiver has been dropped
        let stale_layer: Vec<_> = self
            .shared_state
            .pending_layer_results
            .iter()
            .filter(|entry| entry.value().tx.is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_layer.is_empty() {
            tracing::info!(
                count = stale_layer.len(),
                total_pending = self.shared_state.pending_layer_results.len(),
                request_ids = ?stale_layer.iter().take(5).map(|u| u.to_string()).collect::<Vec<_>>(),
                "DIAG: cleaning up stale pending_layer_results"
            );
            for key in stale_layer {
                self.shared_state.pending_layer_results.remove(&key);
            }
        }

        // pending_tp_partials: remove entries older than TP_PARTIALS_STALE_SECS
        // (stale AllReduce collectors — protocol timeout long since elapsed).
        const TP_PARTIALS_STALE_SECS: u64 = 60;
        let stale_tp: Vec<_> = self
            .shared_state
            .pending_tp_partials
            .iter()
            .filter(|entry| entry.value().created_at.elapsed().as_secs() > TP_PARTIALS_STALE_SECS)
            .map(|entry| *entry.key())
            .collect();
        if !stale_tp.is_empty() {
            tracing::info!(
                count = stale_tp.len(),
                "DIAG: cleaning up stale pending_tp_partials"
            );
            for key in stale_tp {
                self.shared_state.pending_tp_partials.remove(&key);
            }
        }

        // pending_vision_results: remove entries where the oneshot receiver has been dropped
        let stale_vision: Vec<_> = self
            .shared_state
            .pending_vision_results
            .iter()
            .filter(|entry| entry.value().1.is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_vision.is_empty() {
            tracing::info!(
                count = stale_vision.len(),
                "DIAG: cleaning up stale pending_vision_results"
            );
            for key in stale_vision {
                self.shared_state.pending_vision_results.remove(&key);
            }
        }

        // streaming_token_txs: remove entries where the receiver has been dropped
        let stale_stream: Vec<_> = self
            .shared_state
            .streaming_token_txs
            .iter()
            .filter(|entry| entry.value().is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_stream.is_empty() {
            tracing::info!(
                count = stale_stream.len(),
                total_streaming = self.shared_state.streaming_token_txs.len(),
                "DIAG: cleaning up stale streaming_token_txs"
            );
            for key in stale_stream {
                self.shared_state.streaming_token_txs.remove(&key);
            }
        }

        // region_shard_summaries: evict entries older than 10 minutes
        const REGION_SUMMARY_TTL_MS: u64 = 600_000;
        let now_ms = crate::types::unix_now_ms();
        let stale_region: Vec<_> = self
            .shared_state
            .region_shard_summaries
            .iter()
            .filter(|entry| {
                now_ms.saturating_sub(entry.value().timestamp_ms) > REGION_SUMMARY_TTL_MS
            })
            .map(|entry| entry.key().clone())
            .collect();
        if !stale_region.is_empty() {
            tracing::debug!(
                count = stale_region.len(),
                total = self.shared_state.region_shard_summaries.len(),
                "DIAG: cleaning up stale region_shard_summaries"
            );
            for key in stale_region {
                self.shared_state.region_shard_summaries.remove(&key);
            }
        }

        // active_relay_circuits: remove entries older than 1 hour (abnormally terminated)
        const RELAY_CIRCUIT_TTL_SECS: u64 = 3600;
        let stale_relay: Vec<_> = self
            .shared_state
            .active_relay_circuits
            .iter()
            .filter(|entry| entry.value().elapsed().as_secs() > RELAY_CIRCUIT_TTL_SECS)
            .map(|entry| *entry.key())
            .collect();
        if !stale_relay.is_empty() {
            tracing::debug!(
                count = stale_relay.len(),
                "DIAG: cleaning up stale active_relay_circuits"
            );
            for key in stale_relay {
                self.shared_state.active_relay_circuits.remove(&key);
            }
        }
    }

    /// Clean stale peer_id_map entries for peers no longer in peer_registry.
    /// Only runs when the map exceeds 1000 entries to avoid removing entries
    /// that are intentionally kept across disconnects for short periods.
    fn cleanup_stale_peer_id_map(&self) {
        const SOFT_CAP: usize = 8_000;
        const EVICT_TO: usize = 6_000;

        if self.shared_state.peer_id_map.len() <= SOFT_CAP {
            return;
        }
        // First pass: evict entries not in peer_registry (stale)
        let stale_peers: Vec<_> = self
            .shared_state
            .peer_id_map
            .iter()
            .filter(|entry| !self.shared_state.peer_registry.contains_key(entry.key()))
            .map(|entry| entry.key().clone())
            .collect();
        let removed = stale_peers.len();
        for nid in stale_peers {
            self.shared_state.peer_id_map.remove(&nid);
        }
        // Second pass: if still over target, evict oldest (arbitrary order from DashMap)
        let mut removed2 = 0;
        if self.shared_state.peer_id_map.len() > EVICT_TO {
            let excess = self.shared_state.peer_id_map.len() - EVICT_TO;
            let to_evict: Vec<_> = self
                .shared_state
                .peer_id_map
                .iter()
                .filter(|entry| !self.shared_state.peer_registry.contains_key(entry.key()))
                .take(excess)
                .map(|entry| entry.key().clone())
                .collect();
            removed2 = to_evict.len();
            for nid in &to_evict {
                self.shared_state.peer_id_map.remove(nid);
            }
        }
        let total_removed = removed + removed2;
        if total_removed > 0 {
            tracing::debug!(
                removed = total_removed,
                remaining = self.shared_state.peer_id_map.len(),
                "Cleaned stale peer_id_map entries"
            );
        }
    }

    /// Reconcile acquisition_progress against reality each tick.
    ///
    /// 1. Stalled Downloading entries (no byte progress for >STALL_THRESHOLD)
    ///    are reconciled against the shard registry: if all shards are now
    ///    locally held, mark Complete; otherwise mark Failed. Per-shard
    ///    ShardProgress states are flipped to Failed so the dashboard's
    ///    per-shard progress bars disappear.
    /// 2. Completed/Failed entries older than 5 minutes are evicted (was 1h —
    ///    too long, kept stale UI around long after the user cared).
    ///
    /// Warn, once, when a WSL2 mirrored-mode node is demonstrably unreachable.
    ///
    /// Under WSL2 mirrored networking the node is a first-class LAN citizen — a
    /// real address, working QUIC/mDNS/UPnP — but the **Windows** firewall still
    /// governs inbound, and Windows only prompts to allow apps it launches
    /// itself. A Linux binary under WSL gets no prompt at all, so inbound is
    /// silently dropped while the node looks perfectly healthy from the inside:
    /// it dials out fine, holds peers, and advertises its address correctly.
    /// Only the other machine sees it, as sends that never complete. Measured
    /// 2026-08-04: a peer 2ms away on the same subnet could not open TCP 8810 or
    /// UDP 8800, and cross-machine requests died on the segment timeout at 284s.
    ///
    /// **The condition has to be observed, not assumed.** This warning first
    /// shipped as an unconditional line at config-load time, which meant every
    /// mirrored-mode node saw it on every start — including this development
    /// machine after the ports had been opened and verified working. Telling
    /// someone to fix a problem they have already fixed is how warnings stop
    /// being read.
    ///
    /// `observed_inbound_connection` is the evidence: any non-loopback peer
    /// dialling us proves inbound arrives. Outbound proves nothing, which is why
    /// having peers is not enough on its own — this node had three.
    ///
    /// Deliberately gated on having tried for a while AND having outbound peers.
    /// A node that simply has not met anyone yet is not evidence of a firewall,
    /// and warning during startup would reintroduce the false positive with
    /// extra steps.
    fn maybe_warn_wsl_firewall(&mut self) {
        use std::sync::atomic::Ordering;
        if self.wsl_firewall_warned {
            return;
        }
        let decision = inbound_warning_decision(
            self.shared_state
                .observed_inbound_connection
                .load(Ordering::Relaxed),
            self.started_at.elapsed(),
            self.shared_state.connected_node_ids.len(),
        );
        match decision {
            InboundCheck::KeepWaiting => return,
            InboundCheck::Reachable => {
                // Proven reachable — stop paying for the check for this run.
                self.wsl_firewall_warned = true;
                return;
            }
            InboundCheck::Blocked => {}
        }
        self.wsl_firewall_warned = true;
        // The remedy is Windows-specific, so only say it where it applies. The
        // *observation* above is platform-neutral and worth keeping general if
        // another unreachable-but-healthy case turns up.
        if !(crate::config::network::is_wsl2()
            && crate::config::network::wsl_networking_is_mirrored())
        {
            return;
        }
        let port = self.shared_state.config.node.listen_port;
        tracing::warn!(
            p2p_tcp = port + 10,
            quic_udp = port,
            peers = self.shared_state.connected_node_ids.len(),
            "Other machines cannot reach this node — it has been connected for a while \
             and nothing has ever dialled IN. Windows only asks to allow the firewall \
             for apps it launches, never for a Linux program under WSL, so inbound is \
             being dropped even though this node is on the LAN. Run this once in an \
             Administrator PowerShell: \
             New-NetFirewallRule -DisplayName 'SwarmLLM P2P TCP' -Direction Inbound \
             -Protocol TCP -LocalPort {} -Action Allow ; \
             New-NetFirewallRule -DisplayName 'SwarmLLM P2P QUIC' -Direction Inbound \
             -Protocol UDP -LocalPort {} -Action Allow",
            port + 10,
            port
        );
        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "network",
                "inbound_blocked",
                "Other machines can't reach this node. Its ports need allowing through \
                 the Windows firewall — see the log for the exact command."
                    .to_string(),
            )
            .with_toast("warning", 12000),
        );
    }

    /// This is the single source of truth for download liveness — replaces the
    /// scattered cleanup logic that left both `acquisition_progress` and
    /// `peer_shard_downloads` drifting when a download task died silently.
    fn cleanup_acquisition_progress(&mut self) {
        use crate::model::acquisition::{AcquisitionState, ShardState};

        let now = std::time::Instant::now();
        let chrono_now = chrono::Utc::now();
        let cutoff = chrono_now - chrono::Duration::minutes(5);
        let local_node_id = self.shared_state.identity.node_id().clone();
        let mut to_remove: Vec<crate::types::ModelId> = Vec::new();
        let mut to_fail: Vec<(crate::types::ModelId, String, bool)> = Vec::new();
        let mut seen: std::collections::HashSet<crate::types::ModelId> =
            std::collections::HashSet::new();

        for entry in self.shared_state.models.acquisition_progress.iter() {
            let key = entry.key();
            let status = entry.value();
            seen.insert(key.clone());

            match &status.state {
                AcquisitionState::Complete | AcquisitionState::Failed { .. } => {
                    if status.started_at.is_none_or(|s| s < cutoff) {
                        to_remove.push(key.clone());
                    }
                    self.acq_liveness.remove(key);
                }
                AcquisitionState::Downloading | AcquisitionState::AwaitingManifest => {
                    let bytes = status.downloaded_bytes;
                    let prev = self.acq_liveness.get(key).copied();
                    let stalled = match prev {
                        None => false,
                        Some((prev_bytes, last_change)) => {
                            if bytes != prev_bytes {
                                false
                            } else {
                                now.duration_since(last_change) > DOWNLOAD_STALL_THRESHOLD
                            }
                        }
                    };
                    if stalled {
                        // Reconcile against disk — if every shard the model
                        // needs is now locally held, the download actually
                        // completed but the completion event was lost.
                        let manifest = self.shared_state.model_registry.get_manifest(key);
                        let all_local = manifest
                            .as_ref()
                            .map(|m| {
                                m.shards.iter().all(|s| {
                                    let sid = crate::types::ShardId {
                                        model_id: key.clone(),
                                        index: s.index,
                                    };
                                    self.shared_state
                                        .model_registry
                                        .shard_holders(&sid)
                                        .contains(&local_node_id)
                                })
                            })
                            .unwrap_or(false);
                        let secs = now
                            .duration_since(prev.map(|p| p.1).unwrap_or(now))
                            .as_secs();
                        let reason = format!("Stalled — no progress for {}s", secs);
                        to_fail.push((key.clone(), reason, all_local));
                        self.acq_liveness.remove(key);
                    } else if prev.is_none_or(|(b, _)| b != bytes) {
                        self.acq_liveness.insert(key.clone(), (bytes, now));
                    }
                }
            }
        }

        // Drop tracker entries for acquisitions that vanished from the map.
        self.acq_liveness.retain(|k, _| seen.contains(k));

        for (mid, reason, all_local) in to_fail {
            if let Some(mut entry) = self.shared_state.models.acquisition_progress.get_mut(&mid) {
                if all_local {
                    entry.state = AcquisitionState::Complete;
                    entry.log_push("Reconciled: all shards present on disk".into());
                    tracing::info!(model = %mid, "Reconciled stalled acquisition → Complete");
                } else {
                    entry.state = AcquisitionState::Failed {
                        reason: reason.clone(),
                    };
                    entry.log_push(format!("Reconciliation: {}", reason));
                    tracing::warn!(model = %mid, %reason, "Reconciled stalled acquisition → Failed");
                }
                // Flip in-flight per-shard progress to terminal so the
                // dashboard's per-shard bars stop rendering.
                for (idx, sp) in entry.shard_progress.iter_mut() {
                    if matches!(
                        sp.state,
                        ShardState::Downloading | ShardState::Verifying | ShardState::Pending
                    ) {
                        if all_local {
                            sp.state = ShardState::Complete;
                        } else {
                            sp.state = ShardState::Failed;
                            // Back off this shard so the next auto-manage cycle
                            // doesn't immediately re-select the exact same
                            // stalled shard and monopolize the download slot
                            // (external report, 2026-07-23). Touches a distinct
                            // DashMap, so it's safe while `entry` is held.
                            let sid = crate::types::ShardId {
                                model_id: mid.clone(),
                                index: *idx,
                            };
                            let (fails, delay) =
                                self.shared_state.models.record_shard_download_failure(&sid);
                            tracing::debug!(
                                model = %mid,
                                shard = *idx,
                                fails,
                                backoff_secs = delay,
                                "Backing off stalled shard after reconciliation"
                            );
                        }
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            tracing::debug!(
                count = to_remove.len(),
                "Cleaning up stale acquisition progress entries"
            );
            for key in to_remove {
                self.shared_state.models.acquisition_progress.remove(&key);
            }
        }
    }

    /// Sweep peer_shard_downloads — drop entries whose pct hasn't moved for
    /// STALL_THRESHOLD (peer's progress gossip stopped, likely crashed) and
    /// entries whose peer is no longer in peer_registry (already covered by
    /// disconnect handler, but defensive). Single sweep keeps the per-shard
    /// peer-progress dots in the dashboard from going stale.
    fn cleanup_stale_peer_shard_downloads(&mut self) {
        let now = std::time::Instant::now();
        let mut seen: std::collections::HashSet<(crate::types::ShardId, crate::types::NodeId)> =
            std::collections::HashSet::new();
        let mut total_stripped = 0usize;

        self.shared_state
            .models
            .peer_shard_downloads
            .retain(|shard_id, downloaders| {
                downloaders.retain(|(node_id, pct)| {
                    let key = (shard_id.clone(), node_id.clone());
                    seen.insert(key.clone());
                    // Drop if peer is no longer known
                    if !self.shared_state.peer_registry.contains_key(node_id) {
                        self.peer_dl_liveness.remove(&key);
                        total_stripped += 1;
                        return false;
                    }
                    let prev = self.peer_dl_liveness.get(&key).copied();
                    match prev {
                        None => {
                            self.peer_dl_liveness.insert(key, (*pct, now));
                            true
                        }
                        Some((prev_pct, last_change)) => {
                            if *pct != prev_pct {
                                self.peer_dl_liveness.insert(key, (*pct, now));
                                true
                            } else if now.duration_since(last_change) > DOWNLOAD_STALL_THRESHOLD {
                                self.peer_dl_liveness.remove(&key);
                                total_stripped += 1;
                                false
                            } else {
                                true
                            }
                        }
                    }
                });
                !downloaders.is_empty()
            });

        // GC liveness entries for downloaders that vanished from the map.
        self.peer_dl_liveness.retain(|k, _| seen.contains(k));

        if total_stripped > 0 {
            tracing::debug!(
                count = total_stripped,
                "Stripped stale peer_shard_downloads entries"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the check. This warning first shipped unconditional at
    /// config-load, so it fired on every WSL2 mirrored node on every start —
    /// including one whose ports had been opened and verified working, telling
    /// its owner to fix an already-fixed problem.
    #[test]
    fn a_node_that_has_been_dialled_is_never_warned() {
        let long = WSL_FIREWALL_GRACE * 10;
        assert_eq!(
            inbound_warning_decision(true, long, 5),
            InboundCheck::Reachable
        );
        // Still reachable even with no peers connected right now — inbound was
        // observed at some point, which is what the firewall question asks.
        assert_eq!(
            inbound_warning_decision(true, long, 0),
            InboundCheck::Reachable
        );
    }

    /// Outbound connections prove nothing about inbound, so "we have peers" is
    /// not evidence of reachability — the machine that produced this bug had
    /// three peers while dropping every inbound packet.
    #[test]
    fn peers_without_any_inbound_is_the_blocked_case() {
        assert_eq!(
            inbound_warning_decision(false, WSL_FIREWALL_GRACE * 2, 3),
            InboundCheck::Blocked
        );
    }

    /// Two ways to have no evidence yet, neither of which is a firewall.
    #[test]
    fn no_evidence_yet_does_not_accuse_the_firewall() {
        // Inside the grace period: a peer reached via the DHT or a bootstrap
        // round can take minutes to dial back.
        assert_eq!(
            inbound_warning_decision(false, std::time::Duration::from_secs(5), 3),
            InboundCheck::KeepWaiting
        );
        // No peers at all: nobody has had the chance to dial in, so this is a
        // discovery problem and a firewall message would misdirect entirely.
        assert_eq!(
            inbound_warning_decision(false, WSL_FIREWALL_GRACE * 2, 0),
            InboundCheck::KeepWaiting
        );
    }

    /// Long enough that a slow-to-be-discovered node is never accused.
    #[test]
    fn inbound_grace_period_is_generous() {
        assert!(WSL_FIREWALL_GRACE >= std::time::Duration::from_secs(300));
    }

    #[test]
    fn ping_interval_is_30s() {
        assert_eq!(PING_INTERVAL, Duration::from_secs(30));
    }

    #[test]
    fn max_missed_pings_is_3() {
        assert_eq!(MAX_MISSED_PINGS, 3);
    }
}
