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
    /// `manifest_hash` of each model as this node last put it on the wire, so a
    /// manifest is re-broadcast when it CHANGES rather than every tick. See
    /// `broadcast_manifests` for what that was costing.
    last_announced_manifests: std::collections::HashMap<crate::types::ModelId, [u8; 32]>,
    /// Counter for the periodic full manifest re-announce, the anti-entropy half
    /// of the same scheme.
    manifest_announce_counter: u64,
    /// Peers that were connected when manifests last went out. A peer outside
    /// this set has never been sent our picture, and gets a full round.
    peers_told_about_manifests: std::collections::HashSet<crate::types::NodeId>,
    /// What this node last said about each `(region, model)`, so a regional
    /// summary is re-broadcast when it CHANGES rather than once per model per
    /// tick. Measured 2026-09-21: an idle node holding no models published 114
    /// gossip messages every 30 s, of which 20 were these — about models it
    /// does not have, to say a number that had not moved.
    last_announced_region_summaries:
        std::collections::HashMap<(String, crate::types::ModelId), u64>,
    /// Counter for the periodic full regional re-announce, the anti-entropy
    /// half of the same scheme.
    region_summary_counter: u64,
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

/// How long to wait before remarking that nothing has dialled in.
///
/// Long enough that a node which has simply not been found yet is never
/// accused of a firewall problem: mDNS answers in seconds, but a peer that has
/// to come via the DHT or a bootstrap round can take minutes to dial back.
///
/// **Ten minutes was far too short, and the machine's own log said so.** On a
/// run measured 2026-08-18 this check fired at 06:47 and a remote machine
/// dialled in successfully at 07:41 — 64 minutes in, on a node whose inbound
/// ports were open the whole time. An hour is scaled to that observation rather
/// than to intuition. It cannot be scaled to cover the case entirely, because
/// there is no length of silence that proves unreachability: a node dials every
/// peer it already knows within the first second of starting, so it is the
/// dialer on every link and may never be dialled back at all. That is what the
/// persisted observation in `SharedState::observed_inbound_connection` is for,
/// and it is why the message below reports what was seen instead of asserting a
/// cause.
const WSL_FIREWALL_GRACE: std::time::Duration = std::time::Duration::from_secs(3600);

/// How many gossip-broadcast rounds between full re-announcements.
///
/// Shard announcements and model manifests are both delta-broadcast — sent when
/// they change — with a full re-send every this many rounds so a peer that
/// joined during a quiet spell still converges. **One constant for both**,
/// because they answer the same question about the same tick and a reader
/// comparing them should not have to check whether two `10`s mean the same
/// thing. At the ≤10-peer cadence of one round per 30 s that is a full picture
/// every 5 minutes; a node with many peers broadcasts less often and the
/// interval scales with it.
///
/// Each holder's counter is phased independently, so a model held by several
/// nodes is re-announced several times per cycle. A newly connected peer does
/// NOT force one: it is sent our picture directly instead, because a broadcast
/// cannot be addressed to the one node that needs it.
const FULL_REANNOUNCE_EVERY_TICKS: u64 = 10;

/// Is this a round where every manifest is BROADCAST, changed or not?
///
/// Pure so the truth table can be asserted directly — the expensive half of
/// `broadcast_manifests` is one `if` and it is the whole fix.
///
/// This used to take `new_peer_arrived` and return true for it. That answered a
/// question about one peer with a message to every peer, which is the whole
/// cost being removed here; the newcomer now gets a direct catch-up.
fn manifest_round_is_full(counter: u64) -> bool {
    counter.is_multiple_of(FULL_REANNOUNCE_EVERY_TICKS)
}

/// What this node is ASSERTING about a `(region, model)`, independent of when
/// it says it.
///
/// `timestamp_ms` is deliberately excluded: it moves every tick by
/// construction, so folding it in would make every summary look changed and
/// suppress nothing — the shape of a change-gate that silently does not gate.
fn region_summary_digest(summary: &crate::types::RegionShardSummary) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(summary.region.as_bytes());
    hasher.update(summary.model_id.0.as_bytes());
    hasher.update(&summary.region_node_count.to_le_bytes());
    for (index, count) in &summary.shard_counts {
        hasher.update(&index.to_le_bytes());
        hasher.update(&count.to_le_bytes());
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("a blake3 digest is 32 bytes, so its first 8 are always there"),
    )
}

/// Does this one manifest go on the wire this round?
///
/// `last_announced` is the `manifest_hash` this node last broadcast for the
/// model, or `None` if it never has. A manifest nobody has been told about is
/// always sent — "unchanged" is only meaningful against something previously
/// announced.
fn manifest_needs_broadcast(
    last_announced: Option<&[u8; 32]>,
    current_hash: &[u8; 32],
    full_round: bool,
) -> bool {
    full_round || last_announced.is_none_or(|prev| prev != current_hash)
}

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

/// This machine's processor, read once.
///
/// Cached because the answer cannot change while the process runs, and the
/// capability it feeds is rebuilt every thirty seconds — `sysinfo` refreshing a
/// CPU list on each of those would be pure waste.
pub(crate) fn local_cpu_info() -> Option<crate::types::CpuInfo> {
    static CPU: std::sync::OnceLock<Option<crate::types::CpuInfo>> = std::sync::OnceLock::new();
    CPU.get_or_init(|| {
        let sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_cpu(sysinfo::CpuRefreshKind::nothing()),
        );
        let cpus = sys.cpus();
        let name = cpus.first()?.brand().trim().to_string();
        if name.is_empty() {
            return None;
        }
        Some(crate::types::CpuInfo {
            name,
            cores: cpus.len() as u32,
        })
    })
    .clone()
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
            last_announced_manifests: std::collections::HashMap::new(),
            manifest_announce_counter: 0,
            peers_told_about_manifests: std::collections::HashSet::new(),
            last_announced_region_summaries: std::collections::HashMap::new(),
            region_summary_counter: 0,
            acq_liveness: std::collections::HashMap::new(),
            peer_dl_liveness: std::collections::HashMap::new(),
            started_at: std::time::Instant::now(),
            wsl_firewall_warned: false,
        }
    }

    /// Compute how many base ticks (30s each) between heavy gossip broadcasts.
    /// Re-measure this machine's memory bandwidth while it is idle, keeping the
    /// higher figure.
    ///
    /// The first measurement is taken on the first capability broadcast,
    /// whatever else the machine is doing at that moment, and since gotcha
    /// #428 it is the speed a processor-only node advertises — i.e. how much
    /// work the swarm offers it. A node that booted while a build or a browser
    /// was streaming memory carried a low figure for its whole run (Proxmox
    /// read 23.5 GB/s one boot and 25.9 the next). Bandwidth is a hardware
    /// ceiling, so the best observation is the least contaminated one: this
    /// re-measures at ten minutes and then hourly, only when no inference is
    /// in flight (a measurement taken under a decode measures the decode), on
    /// a blocking thread so the 250 ms read does not stall the health loop.
    /// GPU nodes are priced by their card's name and never measure.
    async fn maybe_remeasure_memory_bandwidth(&self, nonce: u64) {
        /// Ticks (30 s each) between re-measurements: hourly.
        const REMEASURE_EVERY_TICKS: u64 = 120;
        /// The first re-measurement, once the boot-time noise has settled.
        const FIRST_REMEASURE_TICK: u64 = 20;
        if self.shared_state.gpu_info.is_some() {
            return;
        }
        if nonce != FIRST_REMEASURE_TICK && !nonce.is_multiple_of(REMEASURE_EVERY_TICKS) {
            return;
        }
        if self.shared_state.active_inference_load() > 0
            || !self
                .shared_state
                .model_process_pool
                .models_with_inflight_requests()
                .is_empty()
        {
            tracing::debug!(
                target: "swarmllm::health::monitor",
                "memory bandwidth re-measure skipped: inference in flight"
            );
            return;
        }
        let before = crate::inference::mem_bandwidth::measured_gbps();
        let after = tokio::task::spawn_blocking(
            crate::inference::mem_bandwidth::remeasure_keeping_the_best,
        )
        .await
        .ok()
        .flatten();
        match (before, after) {
            (Some(b), Some(a)) if a > b => tracing::info!(
                target: "swarmllm::health::monitor",
                before_gbps = format!("{b:.1}"),
                after_gbps = format!("{a:.1}"),
                "Memory bandwidth re-measured higher on an idle machine — the advertised \
                 speed rises with it"
            ),
            _ => tracing::debug!(
                target: "swarmllm::health::monitor",
                before_gbps = ?before,
                after_gbps = ?after,
                "memory bandwidth re-measured; best figure unchanged"
            ),
        }
    }

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

                    // Is the message dispatcher still consuming?
                    //
                    // FIRST, before anything that can itself wait: this is the
                    // one check whose whole value is that it runs in a task the
                    // failure cannot reach. `daemon::supervisor` cannot do it —
                    // `JoinSet::join_next()` fires on a panic or a clean exit,
                    // and a task parked for ever inside an `.await` does
                    // neither, so 45 minutes of total silence produced not one
                    // supervisor line (`docs/FUTURE_WORK.md` #90).
                    self.report_dispatcher_stall();

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

                    self.maybe_remeasure_memory_bandwidth(nonce).await;

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
                    // A worker that exited any way other than a graceful unload
                    // leaves its memory charge behind, and that charge is a
                    // single shared budget — so a dead 14B keeps refusing every
                    // OTHER model too. The request-path check only fires when
                    // that same model is asked for again; this catches the rest.
                    let reaped = self.shared_state.model_process_pool.reap_dead_workers().await;
                    if reaped > 0 {
                        tracing::info!(
                            target: "swarmllm::health::monitor",
                            reaped,
                            "Released the memory budget of workers that had exited"
                        );
                    }
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
                    // Replies kept for `ResendTokens` (gotcha #438) — bounded by
                    // count at insert, and by age here.
                    let swept = self.shared_state.retained_replies.sweep(
                        crate::daemon::state::retained_replies::RETAINED_REPLY_TTL,
                    );
                    if swept > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            swept,
                            "Swept retained fast-path replies"
                        );
                    }
                    // Boundary activations kept so a stand-in can take a
                    // segment over mid-reply. Normally released by
                    // `release_request_state`; this catches a request that
                    // ended without reaching it, so the bytes cannot be held
                    // for the life of the daemon.
                    self.shared_state.retained_activations.sweep_stale(
                        crate::daemon::state::retained_activations::RETAINED_ACTIVATION_TTL,
                    );
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
                    // Take a traffic reading. Here rather than at the stats
                    // build, because a RATE needs two readings taken at a
                    // known cadence and this loop has one — a reader that
                    // sampled whenever someone opened the dashboard would
                    // divide by whatever interval that happened to be.
                    self.shared_state.metrics.bandwidth.refresh();

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

    /// Say so, loudly, when the message dispatcher has stopped consuming.
    ///
    /// A stalled dispatcher is a total outage that reports itself as healthy:
    /// that one channel carries gossip AND every inbound `LayerForward`,
    /// `LayerResult`, `StreamingToken` and `RemoteGenerateRequest`, so the node
    /// stops taking part in the swarm entirely while `/health/ready` still
    /// answers `true` and locally-served requests still work. On 2026-09-18 it
    /// lasted 45 minutes and ended only because the node was restarted for an
    /// unrelated deploy.
    ///
    /// The threshold is deliberately several ping intervals: a quiet node is
    /// not a stalled one, and `HealthPing`/`HealthPong` alone keep this moving
    /// on any node with a peer. Naming the last message's KIND is the point —
    /// it says which arm to look at, which is the question a recurrence has to
    /// answer.
    fn report_dispatcher_stall(&self) {
        let Some((idle, kind)) = self.shared_state.metrics.dispatch_idle_for() else {
            return;
        };
        if idle < crate::daemon::state::DISPATCH_STALL_AFTER {
            return;
        }
        tracing::error!(
            target: "swarmllm::health::monitor",
            idle_secs = idle.as_secs(),
            last_message = kind,
            peers = self.shared_state.peer_registry.len(),
            "The message dispatcher has taken nothing off its channel for \
             {}s — this node is not receiving from the swarm at all, whatever \
             its health endpoint says. The kind above is the last message it \
             accepted, and so the handler to suspect. Inference has been \
             withdrawn from what this node advertises so peers stop routing \
             work it cannot receive; shard serving continues. Restarting the \
             node clears it; see docs/FUTURE_WORK.md #90.",
            idle.as_secs()
        );
    }

    async fn send_health_ping(&self, nonce: u64) {
        let timestamp = crate::types::unix_now_secs();

        let active_request_count = self.shared_state.active_inference_load();
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
        hosted_shards.retain(|s| match store.missing_shard_reason(s) {
            None => true,
            Some(why) => {
                vanished.push((s.clone(), why));
                false
            }
        });
        for (shard_id, why) in vanished {
            // Say WHICH of the two this is. Both end the claim, but a
            // quarantine is a verdict this node reached about bytes that are
            // still on disk, while an absence is storage loss — and they send
            // an operator in opposite directions. Reporting a quarantine as
            // "gone from disk" is how a correct copy, moved aside under a
            // reference that may itself have been wrong, left no trace anyone
            // could follow (2026-08-26).
            tracing::warn!(
                model = %shard_id.model_id,
                index = shard_id.index,
                reason = ?why,
                "No longer claiming a shard to the swarm: {}",
                why.explanation()
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

        // Can this node actually run a request right now? Two failures say no —
        // the graphics stack dying under us, and the dispatcher going deaf —
        // and both report themselves as health, so peers keep routing work that
        // can only fail. `SharedState::inference_outage` is the one answer;
        // withdrawing here rather than per cause is what stops the two
        // conditions advertising different things about the same node.
        //
        // Shard serving is deliberately untouched. A byte-range read needs no
        // worker and no inbound dispatch, so a node in this state can still be
        // the copy that keeps a model reachable for everyone else — which is
        // the whole reason this is a withdrawal of inference and not of the
        // node (docs/FUTURE_WORK.md #89, #90).
        let outage = self.shared_state.inference_outage();
        if let Some(reason) = outage {
            tracing::warn!(
                target: "swarmllm::health::monitor",
                reason = reason.as_str(),
                "Telling the swarm this node cannot take inference work for now \
                 — {}. It is still serving the model pieces it holds, and will \
                 offer inference again by itself if the problem clears.",
                reason.as_str()
            );
        }

        // A card we can no longer reach is not capacity, and advertising it
        // does active harm: peers route work here by what we claim, so a node
        // whose graphics stack has died would keep being sent GPU-sized
        // segments and would keep failing them. Withdrawing the claim leaves
        // the node advertising what it can still honour — its processor.
        //
        // Kept beside the inference withdrawal above rather than folded into
        // it: this one is about the CARD specifically, and a node can lose its
        // card's capacity for reasons that still leave it able to serve.
        //
        // Done here rather than by clearing `gpu_info` because this capability
        // is rebuilt on every broadcast, so the withdrawal takes effect on the
        // next cycle and reverses itself if a worker starts again.
        let gpu_info = self
            .shared_state
            .gpu_info
            .as_ref()
            .filter(|_| !crate::daemon::gpu_support::gpu_runtime_has_failed())
            .map(|g| {
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

        // **Keyed on where models will actually run, not on whether a card
        // exists.** This is the exact ordering `ram_model_budget_mb` above uses
        // and for the same reason: a node that HAS a card and has been told not
        // to use it (`inference.gpu_layers = 0`) still gossips that card,
        // because the card is really there — but its models load into system
        // memory, and generating a token is bandwidth-bound on whichever memory
        // holds the weights.
        //
        // The memory field was given this ordering deliberately; the speed field
        // never had the equivalent, so such a node advertised its card's
        // throughput. Measured on this machine: 35.6 tok/s broadcast against the
        // 4.95 its own scheduler was pricing the local candidate at. That figure
        // is GOSSIPED, so peers rank the node by a speed it will not deliver,
        // and `delegation_target`'s `DELEGATE_MIN_CPU_SPEEDUP` on the far side
        // is compared against it — a seven-fold overstatement of the one number
        // that decides whether work is handed over.
        //
        // `models_go_to_the_card` is the single existing answer to the question,
        // so this field and the memory field cannot come to disagree about the
        // same machine.
        //
        // Deliberately NOT touched here: `gpu`, which keeps reporting the card
        // and its bandwidth. The card is really present, `gpu_inference` is
        // defensible as "this build and machine can do GPU inference", and the
        // memory field draws the line the same way.
        let models_on_card =
            crate::model::auto_manage::vram::models_go_to_the_card(&self.shared_state);
        let est_tokens_per_sec_7b = gpu_info
            .as_ref()
            .filter(|_| models_on_card)
            .map(|g| {
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(
                    g.memory_bandwidth_gbps,
                    true,
                )
            })
            .unwrap_or_else(|| {
                // Models run on the processor here — either there is no card,
                // or there is one and it is not being used. Measure what this
                // machine's memory actually delivers, because generating a
                // token is bandwidth-bound.
                //
                // This was a flat 50 GB/s for every machine, so every CPU node
                // in the swarm advertised the identical 1.70 tokens/s whether it
                // was an eight-channel server or a fanless mini-PC — nothing
                // could tell them apart and nothing could route on the
                // difference. Measured on the machine this was written on:
                // 29.9 GB/s against the 50 assumed, i.e. the guess was 67% high
                // for a perfectly ordinary laptop.
                //
                // Falls back to a deliberately modest nominal when the
                // measurement cannot be taken, which keeps a memory-starved node
                // advertising something rather than nothing — without letting it
                // advertise itself as capable. See
                // `mem_bandwidth::UNMEASURABLE_FALLBACK_GBPS` for why the figure
                // is small and why it had to move when the efficiency did.
                let gbps = crate::model::auto_manage::vram::node_memory_bandwidth_gbps(None)
                    .unwrap_or(crate::inference::mem_bandwidth::UNMEASURABLE_FALLBACK_GBPS);
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(gbps, false)
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
            // `None` until it has settled — see `network_coord_for_publication`.
            coord: self.shared_state.network_coord_for_publication(),
            node_id: node_id.clone(),
            gpu: gpu_info,
            cpu: local_cpu_info(),
            ram_total_mb,
            ram_available_mb,
            // What a model may ACTUALLY take here, which is the figure a peer
            // scheduling work onto this node needs — `ram_available_mb` is the
            // operating system's reading and is roughly twice it on a default
            // install, so a peer routing on that offers segments this node's
            // own admission then refuses (report #022). `None` on a node whose
            // models go to a graphics card: `gpu.vram_available_mb` answers for
            // those, and it already excludes what is resident.
            ram_model_budget_mb: crate::model::auto_manage::vram::node_ram_model_budget_mb(
                &self.shared_state,
            ),
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
            // Whether we can run a request at all, from the one predicate.
            // Rebuilt every broadcast like the rest of this struct, so the
            // withdrawal reverses itself the moment the condition clears — a
            // worker starting again, or the dispatcher resuming.
            can_serve_inference: outage.is_none(),
            // What is LOADED, as distinct from what is on disk above. A peer
            // pricing our spare capacity needs to know how much of a model we
            // have already paid for; without it the only signal is "did this
            // node serve the model recently", which cannot say how much of it
            // is resident. See `NodeCapability::resident_layers`.
            resident_layers: self.shared_state.model_process_pool.resident_model_layers(),
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
            // Full re-announce every `FULL_REANNOUNCE_EVERY_TICKS` broadcast
            // cycles so late-joining peers eventually discover our shards even
            // if nothing changed.
            let periodic_reannounce = self
                .shard_announce_counter
                .is_multiple_of(FULL_REANNOUNCE_EVERY_TICKS);

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
                let announce = crate::model::manifest::shard_announce(
                    &self.shared_state.model_registry,
                    node_id,
                    hosted_shards,
                    complete_for_models,
                );
                let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
                if let Err(e) = self.network_tx.send(msg).await {
                    tracing::debug!(error = %e, shard_count, "DIAG: failed to broadcast shard announce");
                }
                self.last_announced_shards = current_set;
            }
        }
    }

    /// Broadcast model manifests and HF sources so peers can discover and acquire models.
    ///
    /// **A manifest goes out when it CHANGES, not on every tick.** It used to go
    /// out on every tick, and a manifest is not a small message: it carries the
    /// full tensor table of every shard — name, GGUF offset, shard offset and
    /// size per tensor — because `daemon::shard_loader` needs that table to load
    /// a split model from shard files. Measured on the release node 2026-09-20:
    /// 15 manifests totalling **825 KB**, republished every 30 s to a mesh of 6,
    /// i.e. **1.31 Mbps of egress from one node doing nothing at all** — before
    /// counting the copies of every other node's manifests this one forwards.
    /// The node measured 4.8 Mbps out with zero inference and no relaying, which
    /// is the shape two separate field reports described (2026-09-11 and
    /// 2026-09-20) and neither could be answered from config.
    ///
    /// Each republish is sealed with a fresh nonce, so the bytes differ every
    /// time, so the message id differs, so **GossipSub's duplicate cache never
    /// suppressed any of it**. Pushing a large payload to everyone on a timer is
    /// the pattern GossipSub's own IHAVE/IWANT layer exists to avoid: metadata
    /// is disseminated periodically and full messages are sent on request
    /// (libp2p/specs, pubsub/gossipsub). We cannot lazy-pull a manifest yet —
    /// there is no on-demand manifest fetch, gossip is the only way one arrives
    /// (gotcha #296) — so the cheap half of the same idea is taken here: send it
    /// when it changes, and re-send the lot rarely.
    ///
    /// Two things keep discovery working, because a newcomer that cannot see a
    /// model cannot run it and that failure has happened before (gotcha #296):
    ///
    /// 1. **A full round every `FULL_REANNOUNCE_EVERY_TICKS` broadcasts**, the
    ///    same anti-entropy the shard announce beside this already does, and for
    ///    the same reason. Holders' counters are independently phased, so a
    ///    model held by *k* nodes is re-announced roughly *k* times per cycle.
    /// 2. **A full round whenever a peer we have not announced to appears.**
    ///    That is precisely the case the per-tick flood was paying for — someone
    ///    new who needs the whole picture — and it costs one round per join
    ///    instead of one round per 30 seconds for ever.
    async fn broadcast_manifests(&mut self) {
        let our_id = self.shared_state.identity.node_id().clone();

        self.manifest_announce_counter += 1;

        // Anyone connected that the last round did not reach. Compared as a set
        // rather than a count so a simultaneous join and leave is still a join:
        // the newcomer is the whole reason this branch exists.
        let connected: std::collections::HashSet<crate::types::NodeId> = self
            .shared_state
            .connected_node_ids
            .iter()
            .map(|p| p.key().clone())
            .collect();
        // A newcomer no longer forces a BROADCAST round — it is caught up
        // directly below. Gossip has no way to address one peer, so answering
        // "someone new is here" with a topic-wide re-announce made a single
        // join cost every node in the swarm a full copy of every manifest.
        // Measured 2026-09-21: `swarm/models` inbound went from 98.8 KB/s
        // settled to 398.3 KB/s for minutes after ONE node joined, and with
        // peers reconnecting about 7 times an hour on an 8-peer node, the
        // trigger fired roughly as often as the 5-minute periodic round. (An
        // earlier note here said once every 80 s and was wrong — it counted
        // `connection established` lines, which include the post-restart dial
        // burst and libp2p's several connections per peer.)
        // BitTorrent draws this line in the same place — BEP 3's
        // bitfield goes to the peer that connected, over that connection, and
        // only per-piece `have` deltas are sent to everyone afterwards.
        let full_round = manifest_round_is_full(self.manifest_announce_counter);

        // Models we published OR hold a shard of. A publisher-only filter here
        // silently stopped model discovery for the whole swarm — see
        // `ModelRegistry::manifests_to_gossip`.
        let manifests = self
            .shared_state
            .model_registry
            .manifests_to_gossip(&our_id);

        // Anything we no longer gossip stops being remembered, so a model that
        // is deleted and later re-acquired is announced again as the change it
        // is. Without this the map only ever grows and a re-added model looks
        // unchanged.
        let still_gossiped: std::collections::HashSet<crate::types::ModelId> =
            manifests.iter().map(|m| m.id.clone()).collect();
        self.last_announced_manifests
            .retain(|id, _| still_gossiped.contains(id));

        let mut sent = 0usize;
        for manifest in &manifests {
            // `manifest_hash` covers the shard table AND every tensor entry
            // (`ModelManifest::compute_hash`), so it is the right answer to
            // "would a peer see anything new?".
            if !manifest_needs_broadcast(
                self.last_announced_manifests.get(&manifest.id),
                &manifest.manifest_hash,
                full_round,
            ) {
                continue;
            }

            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelManifest(manifest.clone()));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast manifest");
                // Not recorded as announced — a send that failed must be retried
                // on the next tick, not suppressed as already done.
                continue;
            }
            self.last_announced_manifests
                .insert(manifest.id.clone(), manifest.manifest_hash);
            sent += 1;

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

        // Catch up anyone we have never announced to, point to point.
        //
        // This is the anti-entropy the per-tick flood used to pay for, aimed at
        // the node that actually needs it. `SwarmRequest::Message` carries any
        // `SwarmMessage` over request_response and the receiver dispatches it
        // exactly as it would a gossiped one, so an older peer needs no new
        // message type and no feature bit to understand this.
        let mut caught_up = std::collections::HashSet::new();
        for node_id in connected.difference(&self.peers_told_about_manifests) {
            let Some(target_peer_bytes) =
                self.shared_state.resolve_connected_peer_id_bytes(node_id)
            else {
                // Not resolvable right now — leave it out of the told set so
                // the next tick tries again.
                continue;
            };
            let mut delivered = 0usize;
            for manifest in &manifests {
                let msg = NetworkCommand::SendDirectMessage {
                    target_peer_bytes: target_peer_bytes.clone(),
                    message: SwarmMessage::ModelManifest(manifest.clone()),
                    delivery_request_id: None,
                };
                if self.network_tx.send(msg).await.is_err() {
                    break;
                }
                delivered += 1;
            }
            if delivered == manifests.len() {
                caught_up.insert(node_id.clone());
            }
            tracing::debug!(
                peer = %node_id,
                delivered,
                of = manifests.len(),
                "DIAG: caught a new peer up on manifests directly"
            );
        }

        // Only peers actually reached may be recorded as told; a peer left out
        // is retried next tick rather than latched as done.
        if full_round {
            self.peers_told_about_manifests = connected;
        } else {
            self.peers_told_about_manifests.extend(caught_up);
            self.peers_told_about_manifests
                .retain(|node_id| connected.contains(node_id));
        }
        if sent > 0 {
            tracing::debug!(sent, full_round, "DIAG: broadcast model manifests");
        }
    }

    /// Broadcast per-region shard summaries and demand gossip to `swarm/regions`.
    ///
    /// **A summary goes out when it CHANGES**, with a full re-announce every
    /// `FULL_REANNOUNCE_EVERY_TICKS` rounds — the same anti-entropy
    /// `broadcast_manifests` uses, and for the same reason. Publishing one
    /// message per known model per tick made `swarm/regions` the busiest topic
    /// on the swarm by message count: measured 2026-09-21 on a node holding no
    /// models at all, 114 published messages every 30 s and 87 arriving per
    /// SECOND, at 557 bytes each. That is a message-rate problem, not a payload
    /// one, and the duplicate factor was 1.31 — GossipSub's deduplication was
    /// working; there was simply that much being said.
    ///
    /// **Demand comes from `local_region_demand`, never `region_demand`** — see
    /// the note on those fields. The merged map holds what every peer told us,
    /// and re-publishing it re-originated the whole swarm's demand table under
    /// this node's id every 30 s.
    async fn broadcast_region_summary(&mut self) {
        // Determine our region — skip if unknown. Canonical resolver (configured
        // region wins, else IP-detected) so this gossip agrees with the capacity
        // announcement and WS region counts.
        let our_region = match self.shared_state.effective_region().await {
            Some(r) => r.to_uppercase(),
            None => return, // No region — nothing to broadcast
        };

        let our_id = self.shared_state.identity.node_id().clone();
        let now_ms = crate::types::unix_now_ms();

        self.region_summary_counter += 1;
        let full_round = self
            .region_summary_counter
            .is_multiple_of(FULL_REANNOUNCE_EVERY_TICKS);

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

            // Our own shared state is updated every tick regardless: it is a
            // local map, it costs nothing, and only the BROADCAST is rationed.
            let key = (our_region.clone(), manifest.id.clone());
            self.shared_state
                .region_shard_summaries
                .insert(key.clone(), summary.clone());

            // Has anything we would be ASSERTING moved? The digest deliberately
            // excludes `timestamp_ms`, which changes every tick by construction
            // and would suppress nothing.
            let digest = region_summary_digest(&summary);
            if !full_round && self.last_announced_region_summaries.get(&key) == Some(&digest) {
                continue;
            }

            let msg = NetworkCommand::Broadcast(SwarmMessage::RegionShardSummary(summary));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "Failed to broadcast region summary");
                // A send that failed is not a thing the swarm has been told, so
                // it must be retried next tick rather than remembered as done.
                continue;
            }
            self.last_announced_region_summaries.insert(key, digest);
        }

        // Anything we no longer summarise stops being remembered, so a model
        // that goes away and comes back is announced as the change it is.
        self.last_announced_region_summaries
            .retain(|(region, _), _| region == &our_region);

        // Demand gossip for models THIS node has served recently.
        //
        // `local_region_demand` rather than `region_demand`: the latter is the
        // merged view, so iterating it re-published every peer's demand under
        // our own id — 93 messages every 30 s on a node that had served nothing
        // at all, each round refreshing timestamps that should have been
        // ageing out. GossipSub already carries the originator's message to
        // every node; re-originating it was never what made it travel.
        for entry in self.shared_state.local_region_demand.iter() {
            let model_id = entry.key();
            let rate = *entry.value();
            if rate < 0.01 {
                continue; // Don't gossip negligible demand
            }
            let demand = crate::types::ModelDemandGossip {
                model_id: model_id.clone(),
                region: our_region.clone(),
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
                .try_send(RebalanceEvent::PeerLeft(peer_id.clone()))
                .is_err()
            {
                // Silent until 2026-09-18. A departed peer that the rebalancer
                // is never told about leaves its shards looking replicated when
                // they are not, and the INFO line above still claims the
                // cleanup happened.
                if let Some(burst) = self
                    .shared_state
                    .metrics
                    .channel_metrics
                    .rebalance
                    .note_dropped()
                {
                    tracing::warn!(
                        peer = %peer_id,
                        dropped_since_last = burst.suppressed,
                        nothing_accepted_for_secs = burst.stalled_secs(),
                        total_dropped = burst.total,
                        "Rebalance channel full — the rebalancer was not told this peer left"
                    );
                }
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
            .filter(|entry| entry.value().tx.is_closed())
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
    ///
    /// **And it did reintroduce it, which is why the evidence is now persisted
    /// and the message no longer names a cause.** Making the observation
    /// per-process left the check deciding a question about the machine's
    /// firewall from whatever happened in one ten-minute window, and a reachable
    /// node routinely sees nothing in that window: it dials every peer it
    /// already knows within the first second of starting, so it is the dialer on
    /// every link. Measured on this development machine 2026-08-18 — inbound
    /// TCP open and verified by hand from a peer on the same subnet, 181 inbound
    /// connections in the log's history, zero in a 9-hour run, and a run that
    /// warned at 06:47 contradicted by its own inbound connection at 07:41.
    /// Three of the four most recent runs warned; all three were wrong.
    ///
    /// So: silence is reported as silence. The remedy is still offered, because
    /// when it IS a firewall this is the only thing that tells the user, but it
    /// is offered against the condition they can actually check — whether other
    /// machines say they cannot reach this one.
    ///
    /// What that trades away, deliberately: a machine that was reachable and is
    /// later blocked again — a rebuilt Windows install, a new security suite —
    /// is never warned, because the observation is kept forever. That is the
    /// right way round. This is an onboarding aid, and its own history says a
    /// warning that cries wolf is worth less than one that occasionally misses:
    /// it had already been narrowed once for exactly that reason.
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
            "No other machine has ever opened a connection TO this node — every link it \
             has was dialled outwards. That can simply mean nobody has needed to reach \
             it yet, but it is also exactly what a blocked Windows firewall looks like: \
             Windows only asks to allow apps it launches itself, never a Linux program \
             under WSL, so inbound can be dropped silently while the node looks healthy \
             from the inside. If other machines report they cannot reach this one, run \
             this once in an Administrator PowerShell: \
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
                "No other machine has connected to this node yet. If they can't reach \
                 it, its ports need allowing through the Windows firewall — see the log \
                 for the exact command."
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
                // `Cancelled` is terminal too — without it here a cancelled
                // download's entry is never collected and sits in the queue
                // for the life of the daemon.
                AcquisitionState::Complete
                | AcquisitionState::Failed { .. }
                | AcquisitionState::Cancelled => {
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
                // Through the shared helper, not a bare remove: a model whose
                // entry has gone terminal can still have shards downloading —
                // an eleven-shard model that fails one shard is marked Failed
                // while the other two slots keep working — and the entry is
                // what those downloads write their progress into.
                self.shared_state.remove_acquisition_if_idle(&key);
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

    /// The regression this whole scheme exists for. A manifest carries the full
    /// tensor table of every shard, and 15 of them measured 825 KB on the
    /// release node — republished to a mesh of 6 every 30 s, i.e. 1.31 Mbps of
    /// egress from a node doing nothing. An unchanged manifest on an ordinary
    /// round must not go on the wire.
    #[test]
    fn an_unchanged_manifest_is_not_rebroadcast() {
        let hash = [7u8; 32];
        assert!(
            !manifest_needs_broadcast(Some(&hash), &hash, false),
            "a manifest a peer has already been sent, unchanged, on an ordinary \
             round, is the 825 KB this node must stop republishing every 30 s"
        );
    }

    /// The three reasons it still goes out. Each is a way a peer could otherwise
    /// be left without a model it needs, and a node that cannot see a model
    /// cannot run it — that failure has shipped before (gotcha #296).
    #[test]
    fn a_manifest_still_goes_out_when_anyone_could_be_missing_it() {
        let hash = [7u8; 32];
        let other = [9u8; 32];
        assert!(
            manifest_needs_broadcast(None, &hash, false),
            "never announced — nobody has it"
        );
        assert!(
            manifest_needs_broadcast(Some(&other), &hash, false),
            "changed since we announced it — peers hold a stale one"
        );
        assert!(
            manifest_needs_broadcast(Some(&hash), &hash, true),
            "a full round sends everything, which is what makes a late joiner converge"
        );
    }

    fn summary_for_test(
        model: &str,
        counts: &[(u32, u32)],
        nodes: u32,
        timestamp_ms: u64,
    ) -> crate::types::RegionShardSummary {
        crate::types::RegionShardSummary {
            region: "TH".to_string(),
            model_id: crate::types::ModelId(model.to_string()),
            shard_counts: counts.to_vec(),
            region_node_count: nodes,
            publisher: crate::types::NodeId([3u8; 32]),
            timestamp_ms,
        }
    }

    /// The sibling regression, measured on the live swarm 2026-09-21: one
    /// summary per KNOWN model per 30 s tick, from every node, whether or not
    /// anything had moved. `swarm/regions` was carrying 87 messages a second
    /// inbound at 557 bytes each — a message-rate problem that no payload
    /// shrink would have touched.
    ///
    /// ⚠ The digest must ignore `timestamp_ms`. Including it is the failure
    /// this asserts against: every summary would read as changed, the gate
    /// would suppress nothing, and the bug would look fixed in review.
    #[test]
    fn an_unchanged_region_summary_is_not_rebroadcast() {
        let first = summary_for_test("llama-3.2-3b", &[(0, 2), (1, 2)], 3, 1_000);
        let later = summary_for_test("llama-3.2-3b", &[(0, 2), (1, 2)], 3, 9_999_999);
        assert_eq!(
            region_summary_digest(&first),
            region_summary_digest(&later),
            "the same claim made at a later moment is the same claim — if the \
             clock moves the digest, the change-gate suppresses nothing"
        );
    }

    /// Everything the summary actually asserts must move the digest, or a real
    /// regional change would be silently withheld from the swarm — which is
    /// worse than the traffic it saves.
    #[test]
    fn a_changed_region_summary_still_goes_out() {
        let base = summary_for_test("llama-3.2-3b", &[(0, 2), (1, 2)], 3, 1_000);
        for (label, other) in [
            (
                "a shard gained a holder",
                summary_for_test("llama-3.2-3b", &[(0, 2), (1, 3)], 3, 1_000),
            ),
            (
                "a shard lost its last holder",
                summary_for_test("llama-3.2-3b", &[(0, 2), (1, 0)], 3, 1_000),
            ),
            (
                "the region gained a node",
                summary_for_test("llama-3.2-3b", &[(0, 2), (1, 2)], 4, 1_000),
            ),
            (
                "a different model entirely",
                summary_for_test("qwen3-1.7b", &[(0, 2), (1, 2)], 3, 1_000),
            ),
        ] {
            assert_ne!(
                region_summary_digest(&base),
                region_summary_digest(&other),
                "{label} — this is information the swarm needs"
            );
        }
    }

    /// A newcomer must NOT force a broadcast round any more.
    ///
    /// It used to, and that is what made a single join cost every node in the
    /// swarm a full copy of every manifest — measured 2026-09-21 as
    /// `swarm/models` inbound going 98.8 → 398.3 KB/s after one node joined,
    /// with peers reconnecting about 7 times an hour on an 8-peer node — often
    /// enough to roughly double the rate of full rounds. The newcomer is caught
    /// up point to point instead, which is where BitTorrent puts the bitfield.
    #[test]
    fn only_the_timer_forces_a_full_broadcast_round() {
        assert!(
            !manifest_round_is_full(FULL_REANNOUNCE_EVERY_TICKS + 1),
            "an ordinary tick between full rounds sends only what changed"
        );
        assert!(
            manifest_round_is_full(FULL_REANNOUNCE_EVERY_TICKS),
            "the anti-entropy round still fires on its own, so a peer that \
             missed a direct catch-up converges within one interval"
        );
        // The bound that makes the direct catch-up safe to rely on: a peer the
        // point-to-point send never reached still gets everything within one
        // full-round interval, and nothing about a join changes that.
        assert!(
            (1..FULL_REANNOUNCE_EVERY_TICKS).all(|counter| !manifest_round_is_full(counter)),
            "no tick inside the interval broadcasts everything"
        );
    }

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
