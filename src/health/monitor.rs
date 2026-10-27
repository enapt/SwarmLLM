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
    /// Track last broadcast shard set for delta compression — each part with
    /// the build tag it went out under, so a tag that changes (a part falling
    /// into dispute, or out of it) is re-announced on the next tick rather
    /// than left standing until the full re-announce — and whether its model
    /// went out as checked against the origin, so a copy the heal has just
    /// checked tells peers on the next tick too.
    /// Full re-announce only when set changes or every FULL_REANNOUNCE ticks.
    last_announced_shards: std::collections::HashSet<(crate::types::ShardId, u64, bool)>,
    /// Models withheld from the swarm at the last broadcast
    /// (`ModelRegistry::model_is_withheld`), so the DHT is told when one stops
    /// or starts being offered.
    last_withheld_models: std::collections::HashSet<crate::types::ModelId>,
    /// Counter for periodic full re-announce (ensures late-joining peers get data).
    shard_announce_counter: u64,
    /// `manifest_hash` of each model as this node last put it on the wire (or
    /// let stand as already heard), and since when, so a manifest is
    /// re-broadcast when it CHANGES rather than every tick. See
    /// `broadcast_manifests` for what that was costing, and
    /// [`manifest_needs_broadcast`] for what the time is for.
    last_announced_manifests:
        std::collections::HashMap<crate::types::ModelId, ([u8; 32], std::time::Instant)>,
    /// Counter for the periodic full manifest re-announce, the anti-entropy half
    /// of the same scheme.
    manifest_announce_counter: u64,
    /// Peers caught up on our manifests, each with the number of the connection
    /// it was told on (`ManifestSync::connection_of`). A peer not here has never
    /// been sent our picture; a digest-taking peer on a NEWER connection is
    /// caught up again (BEP 3's bitfield, per connection) — a peer that
    /// restarted between two rounds was otherwise still "told" and heard
    /// nothing for up to `MANIFEST_QUIET_WINDOW` (#91 (2), 2026-10-20).
    peers_told_about_manifests: std::collections::HashMap<crate::types::NodeId, Option<u64>>,
    /// When each connected peer advertising `features::MANIFEST_DIGESTS` was
    /// last sent our `ManifestDigest` — at its catch-up, then every
    /// `MANIFEST_QUIET_WINDOW` (FUTURE_WORK #91 (2)).
    digest_sent_at: std::collections::HashMap<crate::types::NodeId, std::time::Instant>,
    /// `models.manifest_heard_changes` as the last stored copy of
    /// `manifest_heard` saw it (`SharedState::persist_manifests_heard`). Zero
    /// at boot: what was just read back is what is stored.
    manifests_heard_stored: u64,
    /// When the capability last went on the wire and what it said, so it is
    /// published when it says something new or `CAPABILITY_HEARTBEAT` has
    /// passed, not every tick (`capability_needs_broadcast`).
    last_published_capability: Option<(std::time::Instant, crate::types::NodeCapability)>,
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
/// cannot be addressed to the one node that needs it — the manifests by this
/// round's catch-up (`broadcast_manifests`), the parts at identify time
/// (`ModelMgmt::own_shard_announce`). Until 2026-10-27 the parts had no such
/// catch-up and this sentence was true only of the manifests: a restarted node
/// learned what a peer held only from this full round, up to 5 minutes later.
const FULL_REANNOUNCE_EVERY_TICKS: u64 = 10;

/// How long a manifest the WHOLE SWARM was just handed stays said.
///
/// Within this window of a gossiped copy of exactly our manifest — anyone's,
/// ours included — this node does not re-announce it, on a full round or
/// otherwise. That is RFC 6206's Trickle suppression: a node that has
/// recently heard exactly what it was about to say says nothing. Without it
/// every holder re-announced on its own full round, so a model held by *k*
/// nodes went out *k* times every 5 minutes; measured on the release node
/// 2026-09-23 at 0.22 manifests a second, ~35 KB each, **87% of all the
/// gossip it received**, none of it news.
///
/// Why two hours is safe: nothing else leans on the periodic round any more.
/// A change of our own is published the tick it happens (only a version heard
/// AFTER we left the one before is suppressed — `manifest_needs_broadcast`,
/// #224); a newcomer, or a peer that dropped and came back, is caught up point
/// to point until delivered; the registry never expires a manifest that stops
/// being repeated. What is left for the periodic round is repairing a gossip
/// message lost in flight, and that now takes at most this window plus one
/// round. It was 30 minutes until the capability broadcast was change-gated
/// (#91): measured 2026-10-11 on the release node (.234, 4.1 h up, 7 peers,
/// idle, 0 messages dropped), the ~22 versions in the swarm went out 44 times
/// an hour, each received ~4.2 times — 59% of the gossip it sent, all of it
/// repeats. Two hours sits under RPL's default Trickle ceiling for the same job
/// (RFC 6550 § 17: Imin 2^3 ms doubled 20 times, ~2.3 h).
const MANIFEST_QUIET_WINDOW: Duration = Duration::from_secs(2 * 60 * 60);

/// The quiet window once EVERY connected peer advertises
/// `features::MANIFEST_DIGESTS` (FUTURE_WORK #91 (2)).
///
/// Then the repair the gossip repeat was for is made point to point: each such
/// peer is sent our `ManifestDigest` at its catch-up and every
/// `MANIFEST_QUIET_WINDOW`, and pulls only a version it lacks. What the repeat
/// is still for is a node connected to no holder of a model, which learns of it
/// only from gossip relayed across the mesh — so it is kept, once a day, rather
/// than dropped. Measured 2026-10-16 (release node, fleet on .240, idle):
/// `swarm/models` 84% of the gossip sent, every manifest a repeat delivered
/// ~3 times; a twelfth of the repeats is a twelfth of that.
pub(crate) const MANIFEST_QUIET_WINDOW_WITH_DIGESTS: Duration = Duration::from_secs(24 * 60 * 60);

/// How long the swarm's hearing of a manifest stays quiet this round:
/// [`MANIFEST_QUIET_WINDOW_WITH_DIGESTS`] when every connected peer takes
/// digests, [`MANIFEST_QUIET_WINDOW`] otherwise — a peer without the bit is
/// repaired only by the gossip repeat, as it always was. No peer at all is the
/// short window: nothing has been told anything (and such a round gossips no
/// manifest at all — `broadcast_manifests`, #91 (2c)).
fn manifest_quiet_window(connected: usize, taking_digests: usize) -> Duration {
    if connected > 0 && taking_digests == connected {
        MANIFEST_QUIET_WINDOW_WITH_DIGESTS
    } else {
        MANIFEST_QUIET_WINDOW
    }
}

/// Is a peer that takes digests owed our versions again? Every
/// `MANIFEST_QUIET_WINDOW` since the last digest or catch-up it was sent; one
/// with no record is owed one now.
fn digest_due(last_sent: Option<std::time::Instant>, now: std::time::Instant) -> bool {
    last_sent.is_none_or(|at| now.saturating_duration_since(at) >= MANIFEST_QUIET_WINDOW)
}

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

/// How long an UNCHANGED capability may go without being published again.
///
/// It used to go out on every broadcast round (30 s on a small swarm) whether
/// or not anything in it had moved. Once Trickle took manifests from 86% of an
/// idle node's gossip down to a few hundred bytes a second, the capability was
/// the largest thing left: measured 2026-10-10 on the release node, 0.2
/// messages a second at ~4.9 KB, about 60% of the gossip bytes it received as
/// first copies (FUTURE_WORK #91).
///
/// Kubernetes met the same cost and named the fix — KEP-589, "efficient node
/// heartbeats": liveness on a small frequent beat, the full status only on a
/// meaningful change or every 5 minutes. The small beat exists here already:
/// `HealthPing` every tick, any other message, and the network's own liveness
/// tick all refresh a peer's `last_seen`, and no version ages a capability out
/// (checked 2026-10-10 — the only readers of how long ago a peer spoke are the
/// 90 s eviction and the dashboard's healthy flag, both on `last_seen`). So a
/// slower repeat drops no peer; it only bounds how stale a figure that moved
/// less than its deadband may get. A peer's displayed uptime lags by up to
/// this much.
const CAPABILITY_HEARTBEAT: Duration = Duration::from_secs(5 * 60);

/// How far a memory figure (free card memory, the RAM budget, free RAM where a
/// peer reads it) must move from what was last PUBLISHED before it is news.
///
/// Measured from the last value SENT, not snapped to a grid: a figure sitting
/// on a grid line would flip across it every tick and republish every tick
/// (deadband reporting, as data-change filters in industrial telemetry do it).
/// 256 MB is under two layers of a 7B at Q4 (~157 MB each), and every peer
/// reading these already discounts them for staleness (`DELEGATE_VRAM_MARGIN`).
const CAPABILITY_MEMORY_DEADBAND_MB: u64 = 256;

/// The same for free disk, which nothing routes a request on — it sizes the
/// swarm's storage pool and a peer's coverage plan.
const CAPABILITY_DISK_DEADBAND_MB: u64 = 1024;

/// The same for the network coordinate, as the distance it predicts between
/// the old and new position. Nothing routes on coordinates yet; a receiver
/// folds ours into its own estimate.
const CAPABILITY_COORD_DEADBAND_MS: f32 = 5.0;

/// The same for a worker's own decode speed (`ResidentModelLayers::
/// decode_us_per_layer`, FUTURE_WORK #3 (f)), as a FACTOR: it is news once the
/// larger of the last-published and current figures is this many times the
/// smaller, or when one appears or goes. A peer reads it only as a floor
/// under the stated speed it prices by, and the error it corrects was 8x
/// (2.3 s priced, 19.4 s taken); a figure within 1.5x of the truth leaves
/// little of that, and the fastest-of-32 it is drawn from moves only when a
/// faster step arrives or the fastest ages out.
pub(crate) const CAPABILITY_SPEED_DEADBAND: f64 = 1.5;

/// Is `now` within [`CAPABILITY_SPEED_DEADBAND`] of `sent`?
fn speed_within_deadband(now: u32, sent: u32) -> bool {
    let (low, high) = (now.min(sent).max(1), now.max(sent));
    f64::from(high) < f64::from(low) * CAPABILITY_SPEED_DEADBAND
}

/// The feature bits this node advertises: what this build implements
/// (`features::ALL`), plus the statements only its settings make true — it
/// takes a requester's guesses (#318: none is offered by a head that would
/// check none), it keeps the half KV cache on its card, and only a window in a
/// sliding-window layer's cache (so a coordinator prices its room for a prompt
/// as it will be held).
fn advertised_features(shared: &SharedState) -> u64 {
    use swarmllm_types::features;
    let cfg = shared.cfg();
    let mut bits = features::ALL;
    if crate::daemon::state::requester_guesses::takes_guesses(&cfg.inference) {
        bits |= features::REQUESTER_GUESSES;
    }
    if crate::inference::layers::kv_half_on_card_enabled() {
        bits |= features::KV_HALF_ON_CARD;
    }
    if crate::inference::split::kv_budget::windows_advertised(
        cfg.inference.prefill_chunk_tokens as usize,
    ) {
        bits |= features::KV_WINDOW_SLIDES;
    }
    bits
}

/// Does `now` tell the swarm anything `sent` — the capability as last
/// PUBLISHED — did not?
///
/// Compares the two WHOLE, after carrying over from `sent` only what is not
/// news: `uptime_seconds` (moves every tick by construction — fold it in and
/// the gate suppresses nothing while looking right in review, the region
/// summary's lesson), the observed-latency snapshot (a pre-warm hint that
/// drifts with every request), and figures that moved less than their
/// deadband. Every other field, including any added later, is a change the
/// moment it differs — the safe direction for a field nobody thought about.
/// Sets are compared as sets.
///
/// Returns the names of the fields that are news (empty: nothing is), so the
/// publish can say WHY it went — the only way to tell a gate that works from
/// one a wobbling figure keeps open.
fn capability_news(
    sent: &crate::types::NodeCapability,
    now: &crate::types::NodeCapability,
) -> Vec<String> {
    fn as_sets(mut cap: crate::types::NodeCapability) -> crate::types::NodeCapability {
        cap.hosted_shards.sort_by(|a, b| {
            (a.model_id.0.as_str(), a.index).cmp(&(b.model_id.0.as_str(), b.index))
        });
        cap.relay_reservations.sort_by_key(|n| n.0);
        cap.resident_layers
            .sort_by(|a, b| a.model_id.cmp(&b.model_id));
        cap
    }
    fn hold(now: &mut u64, sent: u64, deadband: u64) {
        if now.abs_diff(sent) < deadband {
            *now = sent;
        }
    }
    let sent = as_sets(sent.clone());
    let mut now = as_sets(now.clone());
    now.uptime_seconds = sent.uptime_seconds;
    now.observed_latencies = sent.observed_latencies.clone();
    // Free RAM is the host's reading, so it moves with every other program on
    // the machine; on the two-node rig (2026-10-10) it was the only field that
    // republished an idle capability before its heartbeat. Its one reader is `memory_for_model_layers_mb`, which takes it
    // only from a node stating neither a RAM budget nor a card — never this
    // build. Where that reader would not see it, it is not news at any size.
    let mut unread = now.clone();
    unread.ram_available_mb = sent.ram_available_mb;
    if unread.memory_for_model_layers_mb() == now.memory_for_model_layers_mb() {
        now.ram_available_mb = sent.ram_available_mb;
    } else {
        hold(
            &mut now.ram_available_mb,
            sent.ram_available_mb,
            CAPABILITY_MEMORY_DEADBAND_MB,
        );
    }
    hold(
        &mut now.disk_available_mb,
        sent.disk_available_mb,
        CAPABILITY_DISK_DEADBAND_MB,
    );
    if let (Some(n), Some(s)) = (now.ram_model_budget_mb.as_mut(), sent.ram_model_budget_mb) {
        hold(n, s, CAPABILITY_MEMORY_DEADBAND_MB);
    }
    // The machine's room beside a card moves with every other program's
    // memory, as the RAM budget does.
    if let (Some(n), Some(s)) = (
        now.machine_room_beside_card_mb.as_mut(),
        sent.machine_room_beside_card_mb,
    ) {
        hold(n, s, CAPABILITY_MEMORY_DEADBAND_MB);
    }
    if let (Some(n), Some(s)) = (now.gpu.as_mut(), sent.gpu.as_ref()) {
        hold(
            &mut n.vram_available_mb,
            s.vram_available_mb,
            CAPABILITY_MEMORY_DEADBAND_MB,
        );
    }
    if let (Some(n), Some(s)) = (now.coord, sent.coord) {
        let moved = (n.x - s.x).hypot(n.y - s.y) + (n.height - s.height).abs();
        if moved < CAPABILITY_COORD_DEADBAND_MS {
            now.coord = sent.coord;
        }
    }
    // Each worker's own decode speed moves with every step it times; the rest
    // of its residency entry is compared exactly, as before.
    for entry in &mut now.resident_layers {
        let published = sent
            .resident_layers
            .iter()
            .find(|s| s.model_id == entry.model_id)
            .and_then(|s| s.decode_us_per_layer);
        if let (Some(n), Some(s)) = (entry.decode_us_per_layer, published) {
            if speed_within_deadband(n, s) {
                entry.decode_us_per_layer = Some(s);
            }
        }
    }
    // A capability that cannot be serialized cannot be compared: say it is
    // new, so the failure costs a message rather than a stale peer view.
    match (serde_json::to_value(&now), serde_json::to_value(&sent)) {
        (Ok(serde_json::Value::Object(now)), Ok(serde_json::Value::Object(sent))) => now
            .iter()
            .filter(|(field, value)| sent.get(*field) != Some(*value))
            .map(|(field, _)| field.clone())
            .chain(sent.keys().filter(|f| !now.contains_key(*f)).cloned())
            .collect(),
        _ => vec!["(not comparable)".to_string()],
    }
}

/// Does the capability go on the wire this round, and why? `last` is when it
/// was last published and what it said; `None` before the first, which always
/// goes. `None` back means stay quiet.
fn capability_needs_broadcast(
    last: Option<&(std::time::Instant, crate::types::NodeCapability)>,
    now: &crate::types::NodeCapability,
    at: std::time::Instant,
) -> Option<String> {
    let Some((sent_at, sent)) = last else {
        return Some("first".to_string());
    };
    let news = capability_news(sent, now);
    if !news.is_empty() {
        return Some(news.join(","));
    }
    (at.saturating_duration_since(*sent_at) >= CAPABILITY_HEARTBEAT)
        .then(|| "heartbeat".to_string())
}

/// This node's claim of which HuggingFace upload `model` is — what
/// `model::canonical` chooses from — as the message that carries it, or `None`
/// when it knows no source. The ONE builder, for the broadcast round and the
/// newcomer catch-up alike — and the answer to a peer's `ManifestPull` — so
/// they cannot disagree about what a node claims.
///
/// Cloned out of `hf_sources` before anything is sent: `network_tx.send` waits
/// whenever the network loop is busy, and a `Ref` held across it blocks the
/// dispatcher's `hf_sources.insert` on the same shard (clippy.toml).
pub(crate) fn upload_claim(
    state: &crate::daemon::SharedState,
    model: &crate::types::ModelId,
    our_id: &crate::types::NodeId,
) -> Option<SwarmMessage> {
    let source = state
        .models
        .hf_sources
        .get(model)
        .map(|s| s.value().clone())?;
    Some(SwarmMessage::HfSourceGossip(crate::types::HfSourceGossip {
        model_id: model.clone(),
        repo_id: source.repo_id,
        filename: source.filename,
        publisher: our_id.clone(),
        mmproj_filename: source.mmproj_filename,
    }))
}

/// Does this one manifest go on the wire this round?
///
/// `last_announced` is the `manifest_hash` this node last broadcast for the
/// model, or `None` if it never has. A manifest nobody has been told about is
/// always sent — "unchanged" is only meaningful against something previously
/// announced.
///
/// `swarm_heard_at` — when the swarm was last gossiped exactly this hash,
/// within `MANIFEST_QUIET_WINDOW` — overrides both. That is the Trickle
/// suppression, and on the "changed" arm it counts only when heard AFTER we
/// announced the version we are leaving: a change we LEARNED from gossip was
/// heard then, and the swarm has it; a change we made ourselves BACK to a
/// version heard before that is news to every peer that has since taken ours
/// (#224: a node adopted a stranger's hash, caught a newcomer up with it, then
/// re-fetched its part and returned to the version it had broadcast minutes
/// earlier — suppressed for the quiet window (then 30 minutes), so the newcomer never learned
/// the agreed hash and its heal waited for one). RFC 6206 §4.2 draws the same
/// line: a node's own change of state starts a new interval, and only
/// consistent transmissions heard within the current interval suppress.
fn manifest_needs_broadcast(
    last_announced: Option<&([u8; 32], std::time::Instant)>,
    current_hash: &[u8; 32],
    full_round: bool,
    swarm_heard_at: Option<std::time::Instant>,
) -> bool {
    let swarm_heard_it = swarm_heard_at.is_some_and(|heard| match last_announced {
        Some((prev, since)) if prev != current_hash => heard > *since,
        _ => true,
    });
    !swarm_heard_it && (full_round || last_announced.is_none_or(|(prev, _)| prev != current_hash))
}

/// How long a download tracking entry can sit unchanged before being treated
/// as stalled. Generous enough to tolerate slow peers and HF rate limiting.
const DOWNLOAD_STALL_THRESHOLD: Duration = Duration::from_secs(90);

/// How often to send health pings.
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// Number of missed pings before a peer is considered dead.
const MAX_MISSED_PINGS: u32 = 3;

/// Drop `SegmentLatencyTracker` entries that haven't seen a new observation
/// in this many ms. Departed peers leave dead entries that would otherwise
/// accumulate one per (model × segment) they ever served. 1h matches the
/// scale at which a peer being gone is treated as "really gone" by the
/// scheduler — short reconnects don't lose useful latency history.
const SEGMENT_LATENCY_MAX_AGE_MS: u64 = 3_600_000;

/// Drop a peer's measured speed after this long without a fresh observation.
///
/// Serves two purposes: departed peers stop accumulating (three dead entries
/// were live on 2026-08-01), and a peer whose estimate made it un-routable
/// gets a clean slate rather than being permanently de-ranked by a figure that
/// can only be refreshed by routing to it. Matches the segment-latency horizon.
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
            last_withheld_models: std::collections::HashSet::new(),
            shard_announce_counter: 0,
            last_announced_manifests: std::collections::HashMap::new(),
            manifest_announce_counter: 0,
            peers_told_about_manifests: std::collections::HashMap::new(),
            digest_sent_at: std::collections::HashMap::new(),
            manifests_heard_stored: 0,
            last_published_capability: None,
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
                        self.broadcast_region_demand().await;
                        self.broadcast_wishlist_announcement().await;
                        self.broadcast_pool_model_availability().await;
                    }

                    self.maybe_warn_wsl_firewall();
                    self.maybe_warn_slow_card();

                    self.maybe_remeasure_memory_bandwidth(nonce).await;

                    // What this computer may give a model, read here — off the
                    // network loop — for the capacity snapshot's "Runs locally"
                    // marks; the loop's refreshes at every peer connect carry it
                    // over (`refresh_swarm_capacity`). An `nvidia-smi` on a card.
                    crate::daemon::state::refresh_swarm_capacity_reading_the_budget(
                        &self.shared_state,
                    )
                    .await;

                    // Cleanup tasks: run every tick (cheap, local-only)
                    self.cleanup_acquisition_progress();
                    self.cleanup_stale_peer_shard_downloads();
                    self.cleanup_stale_channels();
                    self.cleanup_stale_peer_id_map();
                    // Holder records of keys that never became peers, gone
                    // unrefreshed past every node's re-announce period (#339).
                    if nonce.is_multiple_of(FULL_REANNOUNCE_EVERY_TICKS) {
                        let aged = self.shared_state.model_registry.age_out_holders();
                        if aged > 0 {
                            tracing::debug!(
                                aged,
                                "Forgot holder records no peer of this node has refreshed"
                            );
                        }
                    }
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
                    // Streamed verifies whose request went quiet (a stream is
                    // numbered per request, so an entry per request served).
                    self.shared_state.forward_streams.sweep(
                        crate::daemon::state::forward_streams::FORWARD_STREAM_IDLE,
                    );
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
                    // `SegmentLatencyTracker` accumulates one entry per
                    // (model × segment × holder) triple ever observed; peers
                    // that have left the swarm stop receiving observations
                    // but their entries stick. `PrefetchOrchestrator.histories`
                    // grows one entry per unique session id (UUID — unbounded
                    // cardinality). Both have eviction methods; wire them here.
                    let now_ms = crate::types::unix_now_secs().saturating_mul(1000);
                    let latency_evicted = self
                        .shared_state
                        .metrics
                        .segment_latency
                        .evict_stale(now_ms, SEGMENT_LATENCY_MAX_AGE_MS);
                    if latency_evicted > 0 {
                        tracing::debug!(
                            target: "swarmllm::health::monitor",
                            evicted = latency_evicted,
                            max_age_ms = SEGMENT_LATENCY_MAX_AGE_MS,
                            "Evicted stale segment-latency entries"
                        );
                    }
                    // Take a traffic reading. Here rather than at the stats
                    // build, because a RATE needs two readings taken at a
                    // known cadence and this loop has one — a reader that
                    // sampled whenever someone opened the dashboard would
                    // divide by whatever interval that happened to be.
                    self.shared_state.metrics.bandwidth.refresh();
                    // What this node has served, kept across restarts (#226):
                    // written only when it changed, so an idle node writes
                    // nothing, and a crash loses at most one tick of it.
                    self.shared_state.persist_served_lifetime();
                    // …and how fast it loads a model, so a restart does not
                    // price its first cold load at the prior.
                    self.shared_state.persist_load_rate();

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
    ///
    /// "Stopped consuming" means messages WAITED and were not taken — not that
    /// none arrived. A node with no peers has nothing to take; on 2026-09-25 a
    /// 52-minute host network outage fired this every 30 s, telling the user to
    /// restart a node that recovered all seven peers within five seconds of the
    /// address coming back (`MetricsProviders::dispatch_stalled_for`).
    fn report_dispatcher_stall(&self) {
        let Some((idle, kind)) = self.shared_state.metrics.dispatch_stalled_for() else {
            return;
        };
        if idle < crate::daemon::state::DISPATCH_STALL_AFTER {
            return;
        }
        tracing::error!(
            target: "swarmllm::health::monitor",
            idle_secs = idle.as_secs(),
            last_message = kind,
            waiting = self.shared_state.metrics.dispatch_backlog(),
            peers = self.shared_state.peer_registry.len(),
            "The message dispatcher has taken nothing off its channel for \
             {}s while messages waited in it — this node is not receiving from \
             the swarm at all, whatever \
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

        // A model whose copy here is not the swarm's upload is not offered —
        // not in the capability, not in the announcement — while
        // `auto_manage::canonical` switches it (`SharedState::
        // note_canonical_holding`). `held_models` keeps it in the
        // announcement's `complete_for_models`, which is what makes peers
        // retract the parts they had from us; and the DHT stops naming us as
        // its provider, since a provider record is believed with no build at
        // all.
        let held_models: std::collections::HashSet<crate::types::ModelId> =
            hosted_shards.iter().map(|s| s.model_id.clone()).collect();
        let registry = &self.shared_state.model_registry;
        let withheld_now: std::collections::HashSet<crate::types::ModelId> = held_models
            .iter()
            .filter(|m| registry.model_is_withheld(m))
            .cloned()
            .collect();
        let provide_change = |models: &std::collections::HashSet<crate::types::ModelId>| {
            hosted_shards
                .iter()
                .filter(|s| models.contains(&s.model_id))
                .cloned()
                .collect::<Vec<_>>()
        };
        let stop = provide_change(
            &withheld_now
                .difference(&self.last_withheld_models)
                .cloned()
                .collect(),
        );
        let start = provide_change(
            &self
                .last_withheld_models
                .difference(&withheld_now)
                .cloned()
                .collect(),
        );
        if !stop.is_empty() {
            let _ = self
                .network_tx
                .send(NetworkCommand::StopProviding(stop))
                .await;
        }
        if !start.is_empty() {
            let _ = self
                .network_tx
                .send(NetworkCommand::StartProviding(start))
                .await;
        }
        self.last_withheld_models = withheld_now;
        hosted_shards.retain(|s| !registry.model_is_withheld(&s.model_id));

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
        // A metered connection (#277, `daemon::metered`): no new work for
        // other computers, so say so and planners skip this node the way they
        // skip one that cannot run a request — a peer that asked anyway would
        // be refused. Read without waiting; the system is asked off this loop,
        // and a changed answer is news at the next tick (`capability_news`).
        let metered_hold = self.shared_state.metered_hold_now();
        {
            let state = self.shared_state.clone();
            tokio::spawn(async move {
                state.metered_hold().await;
            });
        }
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
                // `SharedState::gpu_info` used to carry a `vram_free_mb` set ONCE at
                // startup and hardcoded to 0 there (`daemon/mod.rs`), so every node
                // in the swarm advertised zero free VRAM for as long as that field
                // existed. Nothing read it, so nothing went wrong — until something
                // did, and then it silently answered "no room" for every peer,
                // everywhere. The field is gone (2026-10-05) so it cannot be read.
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

        // Real uptime, for the peer lists that show it. It is not news on its
        // own: `capability_news` leaves it out of the comparison.
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
                let (ram_total, ram_avail) =
                    crate::daemon::machine_memory::machine_memory_mb();

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
                crate::model::auto_manage::vram::processor_tokens_per_sec_7b()
            });

        // Top-N observed-latency snapshot, ordered by the trust we have in
        // each *observed peer* (not the sender). Gives receivers a pre-warm
        // Parallax DP signal so newly-joining nodes don't need to route
        // requests through a peer to price it. Kept to 32 entries → ≈1.2 KB
        // extra per broadcast, well under the 4 MB gossip cap.
        const MAX_OBSERVED: usize = 32;
        let observed_latencies = {
            let mut entries: Vec<(crate::types::NodeId, f32, f32, f32)> = self
                .shared_state
                .metrics
                .peer_speed
                .iter()
                // Gossip carries the ranking-scale per-layer figure of the
                // model we last measured the peer on, with that model's
                // weight so a receiver pricing another model can scale it
                // (#250); a peer we have only ever prefilled through still
                // has one.
                .filter_map(|r| {
                    r.value()
                        .ranking_for_gossip()
                        .map(|(ms, work)| (r.key().clone(), ms, work))
                })
                .map(|(peer_id, ms, work)| {
                    let trust = self
                        .shared_state
                        .peer_registry
                        .get(&peer_id)
                        .map(|p| p.trust_score)
                        .unwrap_or(0.5);
                    (peer_id, ms, work, trust)
                })
                .collect();
            // Higher trust first. Stable ordering (partial_cmp handles NaN by
            // treating as Less — but trust_scores are clamped [0,1]).
            entries.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(MAX_OBSERVED);
            entries
                .into_iter()
                .map(
                    |(peer, ms_per_layer, layer_work, _)| crate::types::LatencyObservation {
                        peer,
                        ms_per_layer,
                        layer_work: Some(layer_work),
                    },
                )
                .collect::<Vec<_>>()
        };
        let machine_room_beside_card_mb =
            crate::model::auto_manage::vram::node_machine_room_beside_card_mb(&self.shared_state);

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
            // Whether we can run a request at all, from the one predicate —
            // and whether we will, on a metered connection. Rebuilt every
            // broadcast like the rest of this struct, so the withdrawal
            // reverses itself the moment the condition clears — a worker
            // starting again, the dispatcher resuming, the connection no
            // longer metered.
            can_serve_inference: outage.is_none() && !metered_hold,
            // What is LOADED, as distinct from what is on disk above. A peer
            // pricing our spare capacity needs to know how much of a model we
            // have already paid for; without it the only signal is "did this
            // node serve the model recently", which cannot say how much of it
            // is resident. See `NodeCapability::resident_layers`.
            resident_layers: self.shared_state.model_process_pool.resident_model_layers(),
            // How long a conversation we serve, so a coordinator can skip us
            // for a prompt we would refuse instead of finding out minutes into
            // its prompt pass. See `NodeCapability::context_ceiling_tokens`.
            context_ceiling_tokens: Some(
                self.shared_state
                    .model_process_pool
                    .served_context_ceiling(),
            ),
            // How fast this node's own loads have been (#129), so a
            // coordinator can price waiting for it to load a model.
            model_load_ms_per_gib: self.shared_state.model_process_pool.load_ms_per_gib(),
            // The most a model could ever be given here, which a coordinator
            // never plans past (a tester's report, 2026-10-07).
            model_memory_ceiling_mb: crate::model::auto_manage::vram::node_model_memory_ceiling_mb(
                &self.shared_state,
            ),
            // On a card node: what a segment wider than the card may put in
            // system memory, and how fast those layers run — without them a
            // card that was full offered almost nothing, with the machine's
            // memory free beside it (2026-10-25).
            machine_room_beside_card_mb,
            processor_tokens_per_sec_7b: machine_room_beside_card_mb
                .map(|_| crate::model::auto_manage::vram::processor_tokens_per_sec_7b()),
            // Advertise the protocol epoch + the optional features this build
            // implements, so peers negotiate new message types additively.
            protocol_version: swarmllm_types::PROTOCOL_VERSION,
            features: advertised_features(&self.shared_state),
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

        // Keep a copy every round, published or not: local surfaces show this
        // node from it, and a newly connected peer is caught up with it
        // directly (`identify.rs`), so it must be today's figures.
        self.shared_state
            .local_capability
            .store(Some(std::sync::Arc::new(cap.clone())));
        let at = std::time::Instant::now();
        if let Some(why) =
            capability_needs_broadcast(self.last_published_capability.as_ref(), &cap, at)
        {
            let msg = NetworkCommand::Broadcast(SwarmMessage::NodeCapabilityUpdate(cap.clone()));
            match self.network_tx.send(msg).await {
                Ok(()) => {
                    tracing::debug!(
                        target: "swarmllm::health::monitor",
                        why = %why,
                        "DIAG: capability published"
                    );
                    self.last_published_capability = Some((at, cap));
                }
                Err(e) => {
                    tracing::debug!(error = %e, "DIAG: failed to broadcast capability update")
                }
            }
        }

        // Delta-compressed shard announcements: only broadcast when shard set
        // changes, or every 10 broadcast cycles as a full re-announce (ensures
        // late-joining peers get the full picture). At 10K peers with scaled
        // gossip interval (~240s), full re-announce happens every ~40 min.
        if let Some(announce) = self.shard_announce_round(node_id, hosted_shards, &held_models) {
            let shard_count = announce.shards.len();
            let msg = NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(announce));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, shard_count, "DIAG: failed to broadcast shard announce");
            }
        }
    }

    /// This round's announcement of the parts held here, returned when it goes
    /// on gossip: when the set changed, or every `FULL_REANNOUNCE_EVERY_TICKS`
    /// rounds. Built EVERY round and kept in `ModelMgmt::own_shard_announce`,
    /// whether it goes or not, so a peer connecting between full rounds is sent
    /// this picture point to point by the identify handler — one builder, so
    /// the copy a newcomer gets is the one the swarm gets. `None` is kept while
    /// nothing is held.
    fn shard_announce_round(
        &mut self,
        node_id: crate::types::NodeId,
        hosted_shards: Vec<crate::types::ShardId>,
        held_models: &std::collections::HashSet<crate::types::ModelId>,
    ) -> Option<crate::types::ShardAnnounce> {
        if held_models.is_empty() {
            self.shared_state.models.own_shard_announce.store(None);
            return None;
        }
        let registry = &self.shared_state.model_registry;
        let current_set: std::collections::HashSet<_> = hosted_shards
            .iter()
            .map(|s| {
                (
                    s.clone(),
                    registry.announced_build_tag(s),
                    registry.model_is_origin_checked(&s.model_id),
                )
            })
            .collect();
        let shards_changed = current_set != self.last_announced_shards;
        self.shard_announce_counter += 1;
        // Full re-announce every `FULL_REANNOUNCE_EVERY_TICKS` broadcast
        // cycles so late-joining peers eventually discover our shards even
        // if nothing changed.
        let periodic_reannounce = self
            .shard_announce_counter
            .is_multiple_of(FULL_REANNOUNCE_EVERY_TICKS);
        // Complete for every model represented. `shards_changed` above
        // fires when shards are deleted, so this is what actually
        // retracts them on peers — previously it re-sent the smaller
        // set and receivers merged it, keeping the deleted shards.
        //
        // Every model this node HOLDS, offered or not: a withheld one
        // goes out with none of its parts, which retracts them.
        let complete_for_models: Vec<crate::types::ModelId> = held_models.iter().cloned().collect();
        let announce = crate::model::manifest::shard_announce(
            registry,
            node_id,
            hosted_shards,
            complete_for_models,
        );
        self.shared_state
            .models
            .own_shard_announce
            .store(Some(std::sync::Arc::new(announce.clone())));
        if !(shards_changed || periodic_reannounce) {
            return None;
        }
        self.last_announced_shards = current_set;
        Some(announce)
    }

    /// `hash` is this node's announced version of `model` — broadcast, or let
    /// stand as already heard. The time is when it BECAME ours, kept while it
    /// stays so: [`manifest_needs_broadcast`] asks whether a version we return
    /// to was heard after we left it.
    fn note_announced(&mut self, model: &crate::types::ModelId, hash: [u8; 32]) {
        if self
            .last_announced_manifests
            .get(model)
            .is_none_or(|(prev, _)| *prev != hash)
        {
            self.last_announced_manifests
                .insert(model.clone(), (hash, std::time::Instant::now()));
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
    /// 1. **A newcomer is caught up point to point**, every manifest sent
    ///    directly and retried each tick until delivered. A broadcast cannot be
    ///    addressed to the one node that needs it. A newcomer advertising
    ///    `features::MANIFEST_DIGESTS` is sent only the versions
    ///    (`ManifestDigest`) and pulls what it lacks, and is sent them again
    ///    every `MANIFEST_QUIET_WINDOW` (#91 (2)).
    /// 2. **A full round every `FULL_REANNOUNCE_EVERY_TICKS` broadcasts**, the
    ///    anti-entropy that repairs a gossip message lost in flight — minus any
    ///    manifest the whole swarm was gossiped within `MANIFEST_QUIET_WINDOW`
    ///    (RFC 6206's Trickle suppression). Holders' counters are phased
    ///    independently, so without that a model held by *k* nodes went out *k*
    ///    times per round; with it, once per window from whichever holder's
    ///    round comes up first.
    async fn broadcast_manifests(&mut self) {
        let our_id = self.shared_state.identity.node_id().clone();

        self.manifest_announce_counter += 1;
        // Kept as long as the longer of the two windows can ask about it.
        self.shared_state
            .models
            .forget_manifests_heard_before(MANIFEST_QUIET_WINDOW_WITH_DIGESTS);

        // Anyone connected that the last round did not reach. Compared as a set
        // rather than a count so a simultaneous join and leave is still a join:
        // the newcomer is the whole reason this branch exists.
        let connected: std::collections::HashSet<crate::types::NodeId> = self
            .shared_state
            .connected_node_ids
            .iter()
            .map(|p| p.key().clone())
            .collect();
        // Those told the versions in place of the manifests (#91 (2)). Unknown
        // is no: a peer whose capability has not arrived is sent manifests.
        let takes_digests: std::collections::HashSet<crate::types::NodeId> = connected
            .iter()
            .filter(|node_id| {
                self.shared_state
                    .peer_advertises_feature(node_id, crate::types::features::MANIFEST_DIGESTS)
            })
            .cloned()
            .collect();
        let quiet = manifest_quiet_window(connected.len(), takes_digests.len());
        self.retry_overdue_manifest_pulls(&takes_digests).await;
        self.shared_state.models.manifest_sync.forget_stale();
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
        // Each manifest carries the context its model declares, so a node
        // holding none of it can say how long a conversation it takes (#189).
        self.shared_state.fill_declared_contexts();
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
        let mut suppressed = 0usize;
        // With no peer connected nobody hears a broadcast: GossipSub keeps it
        // for the first peer to subscribe, which the catch-up below tells
        // point to point anyway, and this round's window cannot know whether
        // peers take digests. The monitor's first round runs before any peer is
        // back, so after a restart it gossiped every manifest the swarm had
        // heard within the day again (#91 (2c)). It waits for a peer.
        let to_gossip: &[crate::types::ModelManifest] = if connected.is_empty() {
            &[]
        } else {
            &manifests
        };
        for manifest in to_gossip {
            // `manifest_hash` covers the shard table AND every tensor entry
            // (`ModelManifest::compute_hash`), so it is the right answer to
            // "would a peer see anything new?".
            let swarm_heard_at = self.shared_state.models.manifest_heard_at(
                &manifest.id,
                &manifest.manifest_hash,
                quiet,
            );
            if !manifest_needs_broadcast(
                self.last_announced_manifests.get(&manifest.id),
                &manifest.manifest_hash,
                full_round,
                swarm_heard_at,
            ) {
                if swarm_heard_at.is_some() {
                    // The swarm has this hash, so it no longer counts as a
                    // change still owed — otherwise it would go out as one the
                    // moment the window lapsed, on an ordinary round.
                    self.note_announced(&manifest.id, manifest.manifest_hash);
                    suppressed += 1;
                }
                continue;
            }

            let msg = NetworkCommand::Broadcast(SwarmMessage::ModelManifest(manifest.clone()));
            if let Err(e) = self.network_tx.send(msg).await {
                tracing::debug!(error = %e, model = %manifest.id, "DIAG: failed to broadcast manifest");
                // Not recorded as announced — a send that failed must be retried
                // on the next tick, not suppressed as already done.
                continue;
            }
            self.note_announced(&manifest.id, manifest.manifest_hash);
            // Our own broadcast is gossip the swarm has now heard — GossipSub
            // never hands a node its own message back, so without this we would
            // re-announce on our next full round what we said a minute ago.
            self.shared_state.models.note_manifest_heard(
                &manifest.id,
                manifest.manifest_hash,
                crate::types::MessageTransport::Gossip,
            );
            sent += 1;

            // Also broadcast HfSourceGossip so late-joining peers discover the HF source.
            if let Some(claim) = upload_claim(&self.shared_state, &manifest.id, &our_id) {
                let msg = NetworkCommand::Broadcast(claim);
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
        //
        // A peer that takes digests is caught up again once it is on a newer
        // connection than the one it was told on: ~200 bytes a model, and it
        // pulls only what it lacks. An older peer is sent every manifest in
        // full, so it keeps the rule of a peer a round found gone.
        let shared = self.shared_state.clone();
        let sync = &shared.models.manifest_sync;
        let to_catch_up: Vec<crate::types::NodeId> = connected
            .iter()
            .filter(
                |node_id| match self.peers_told_about_manifests.get(*node_id) {
                    None => true,
                    Some(told_on) => {
                        takes_digests.contains(*node_id) && *told_on != sync.connection_of(node_id)
                    }
                },
            )
            .cloned()
            .collect();
        let mut caught_up = std::collections::HashSet::new();
        for node_id in &to_catch_up {
            let Some(target_peer_bytes) =
                self.shared_state.resolve_connected_peer_id_bytes(node_id)
            else {
                // Not resolvable right now — leave it out of the told set so
                // the next tick tries again.
                continue;
            };
            if takes_digests.contains(node_id) {
                // The versions alone, ~200 bytes a model: the peer pulls only
                // what it lacks, so one reconnecting with every version pulls
                // nothing. Every manifest in full was ~2.9 MB from six peers
                // to one probe node in 20 s (#91 (2), 2026-10-16).
                if self
                    .send_manifest_digest(&target_peer_bytes, &manifests, &our_id)
                    .await
                {
                    caught_up.insert(node_id.clone());
                }
                tracing::debug!(
                    peer = %node_id,
                    versions = manifests.len(),
                    "DIAG: caught a new peer up on manifests with a digest"
                );
                continue;
            }
            let mut delivered = 0usize;
            for manifest in &manifests {
                // The claim travels WITH the manifest. Without it a newcomer
                // learned which upload a model is only from a broadcast, and
                // those are suppressed while the swarm has heard the manifest
                // (up to `MANIFEST_QUIET_WINDOW`): meanwhile it could register
                // the first manifest it was sent and fetch against it. One
                // upload per model is a max-register (`model::canonical`), and
                // a max-register converges only on claims every node is
                // eventually DELIVERED — this is that delivery's anti-entropy
                // half, as Demers et al.'s epidemic algorithms pair rumour
                // mongering with it.
                let mut messages = vec![SwarmMessage::ModelManifest(manifest.clone())];
                messages.extend(upload_claim(&self.shared_state, &manifest.id, &our_id));
                let mut sent_all = true;
                for message in messages {
                    let msg = NetworkCommand::SendDirectMessage {
                        target_peer_bytes: target_peer_bytes.clone(),
                        message,
                        delivery_request_id: None,
                    };
                    if self.network_tx.send(msg).await.is_err() {
                        sent_all = false;
                        break;
                    }
                }
                if !sent_all {
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
        // is retried next tick rather than latched as done. A full round used
        // to mark EVERY connected peer told, on the grounds that the broadcast
        // had just reached them — no longer true once a full round can be
        // suppressed, and a newcomer whose direct catch-up failed would then
        // be latched as told with nothing sent to it at all.
        let now = std::time::Instant::now();
        for node_id in &caught_up {
            // A peer just caught up — in full or by digest — holds every
            // version as of now, so its first repeat digest is a window away.
            self.digest_sent_at.insert(node_id.clone(), now);
        }
        for node_id in caught_up {
            let told_on = sync.connection_of(&node_id);
            self.peers_told_about_manifests.insert(node_id, told_on);
        }
        self.peers_told_about_manifests
            .retain(|node_id, _| connected.contains(node_id));
        self.digest_sent_at
            .retain(|node_id, _| connected.contains(node_id));

        // The anti-entropy for a peer that takes digests: the versions again
        // every `MANIFEST_QUIET_WINDOW`, the interval the gossiped repeat had
        // before it was stretched for such peers — so a change lost in flight
        // reaches one no later than it did, at ~200 bytes a model, not ~35 KB.
        let mut repeated = 0usize;
        for node_id in takes_digests
            .iter()
            .filter(|node_id| self.peers_told_about_manifests.contains_key(*node_id))
        {
            if !digest_due(self.digest_sent_at.get(node_id).copied(), now) {
                continue;
            }
            let Some(target_peer_bytes) =
                self.shared_state.resolve_connected_peer_id_bytes(node_id)
            else {
                continue;
            };
            if self
                .send_manifest_digest(&target_peer_bytes, &manifests, &our_id)
                .await
            {
                self.digest_sent_at.insert(node_id.clone(), now);
                repeated += 1;
            }
        }
        // Our own broadcasts above are among what the swarm has heard; stored,
        // a restart does not gossip them again (#91 (2c)).
        self.manifests_heard_stored = self
            .shared_state
            .persist_manifests_heard(self.manifests_heard_stored);
        if sent > 0 || suppressed > 0 || repeated > 0 {
            tracing::debug!(
                sent,
                suppressed,
                repeated_digests = repeated,
                full_round,
                quiet_window_secs = quiet.as_secs(),
                "DIAG: broadcast model manifests (suppressed = the swarm was gossiped \
                 that exact manifest within the quiet window)"
            );
        }
    }

    /// Ask every connected peer that takes digests for each manifest version
    /// a pull asked of one that has not answered within `PULL_PATIENCE` —
    /// slow, gone or hostile — rather than leave it to a digest up to
    /// `MANIFEST_QUIET_WINDOW` later (`ManifestSync::retry_overdue_pulls`,
    /// #91 (2)). Every such peer, not a remembered few: a hostile peer could
    /// fill any list it is put on, and a peer holding no part of the model
    /// answers nothing.
    async fn retry_overdue_manifest_pulls(
        &self,
        takes_digests: &std::collections::HashSet<crate::types::NodeId>,
    ) {
        let sync = &self.shared_state.models.manifest_sync;
        let versions = sync.retry_overdue_pulls(std::time::Instant::now());
        if versions.is_empty() {
            return;
        }
        let mut asked = 0usize;
        for peer in takes_digests {
            let Some(target_peer_bytes) = self.shared_state.resolve_connected_peer_id_bytes(peer)
            else {
                continue;
            };
            let mut sent = true;
            for chunk in versions.chunks(crate::daemon::state::manifest_sync::MAX_PULL_MODELS) {
                let message = SwarmMessage::ManifestPull(crate::types::ManifestPull {
                    models: chunk.iter().map(|(model, _)| model.clone()).collect(),
                });
                let msg = NetworkCommand::SendDirectMessage {
                    target_peer_bytes: target_peer_bytes.clone(),
                    message,
                    delivery_request_id: None,
                };
                if self.network_tx.send(msg).await.is_err() {
                    sent = false;
                    break;
                }
            }
            asked += usize::from(sent);
        }
        if asked == 0 {
            sync.pull_not_sent(&versions);
        }
        tracing::debug!(
            versions = versions.len(),
            peers = asked,
            "DIAG: asked every peer that takes digests for manifest versions a pull left unanswered"
        );
    }

    /// Tell one peer which version of each manifest this node gossips, with
    /// the upload claim each carries — the peer pulls (`ManifestPull`) only the
    /// versions it lacks, answered by `daemon::dispatch::manifest_sync::on_pull`.
    /// The claims go whether or not it pulls: they are a max-register that
    /// converges only on claims every node is eventually delivered
    /// (`model::canonical`), and they are small. True when every message was
    /// handed to the network loop.
    async fn send_manifest_digest(
        &self,
        target_peer_bytes: &[u8],
        manifests: &[crate::types::ModelManifest],
        our_id: &crate::types::NodeId,
    ) -> bool {
        let mut messages: Vec<SwarmMessage> = manifests
            // At the size of one pull, so each digest can be pulled whole.
            .chunks(crate::daemon::state::manifest_sync::MAX_PULL_MODELS)
            .map(|chunk| {
                SwarmMessage::ManifestDigest(crate::daemon::state::manifest_sync::digest_of(chunk))
            })
            .collect();
        for manifest in manifests {
            messages.extend(upload_claim(&self.shared_state, &manifest.id, our_id));
        }
        for message in messages {
            let msg = NetworkCommand::SendDirectMessage {
                target_peer_bytes: target_peer_bytes.to_vec(),
                message,
                delivery_request_id: None,
            };
            if self.network_tx.send(msg).await.is_err() {
                return false;
            }
        }
        true
    }

    /// Broadcast this node's demand to `swarm/regions`.
    ///
    /// **Per-region shard summaries are no longer sent (2026-10-15, #91).**
    /// Every node re-announced one `RegionShardSummary` per KNOWN model every 5
    /// minutes — measured on the release node, 0.11 messages a second from a
    /// node knowing 31 models, and with each message arriving ~6.5 times, a
    /// third of the gossip it sent — and nothing used them: auto-manage scoring
    /// asked only about this node's OWN region, which it counts from the holder
    /// records it already has (`count_regional_holders`), and the dashboard's
    /// map builds its coverage from peers' capabilities (`api::admin`). Older
    /// nodes still send them; dispatch drops them.
    ///
    /// **Demand comes from `local_region_demand`, never `region_demand`** — see
    /// the note on those fields. The merged map holds what every peer told us,
    /// and re-publishing it re-originated the whole swarm's demand table under
    /// this node's id every 30 s.
    async fn broadcast_region_demand(&self) {
        let Some(our_region) = self.shared_state.effective_region().await else {
            return; // No region — nothing to file demand under
        };
        let our_region = our_region.to_uppercase();
        let our_id = self.shared_state.identity.node_id().clone();
        let now_ms = crate::types::unix_now_ms();

        // Demand gossip for models THIS node has served recently.
        //
        // `local_region_demand` rather than `region_demand`: the latter is the
        // merged view, so iterating it re-published every peer's demand under
        // our own id — 93 messages every 30 s on a node that had served nothing
        // at all, each round refreshing timestamps that should have been
        // ageing out. GossipSub already carries the originator's message to
        // every node; re-originating it was never what made it travel.
        //
        // Snapshotted before sending: an iterator holds its shard's read lock,
        // and each send below can wait on the network loop.
        let local_demand: Vec<(crate::types::ModelId, f64)> = self
            .shared_state
            .local_region_demand
            .iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect();
        for (model_id, rate) in local_demand {
            let model_id = &model_id;
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
                request_ids = ?stale_layer.iter().take(5).map(|k| k.request_id.to_string()).collect::<Vec<_>>(),
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
    /// Tell the owner, at most every `cuda_pool::SLOW_CARD_NOTICE_EVERY`, that
    /// the graphics card has become slow to hand out fresh memory — what the
    /// workers' allocation probes measure (`cuda_pool::probe_once`). On WSL2
    /// that grows with Windows uptime and only a Windows restart clears it
    /// (#146's deep dive, microsoft/WSL#41701); this PC crashed twice at 52-54 h
    /// of uptime with it (#754, #762). The log line is written everywhere; the
    /// dashboard notice only where its advice — restart Windows — applies.
    fn maybe_warn_slow_card(&self) {
        let Some(took) = self
            .shared_state
            .model_process_pool
            .slow_card_notice()
            .take(std::time::Instant::now())
        else {
            return;
        };
        let wsl = crate::config::network::is_wsl2();
        tracing::warn!(
            took_ms = took.as_millis() as u64,
            wsl,
            "The graphics card has become slow to hand out memory — a fresh allocation \
             probe took {:.1} s where a healthy card takes a few milliseconds. Under WSL2 \
             this builds up the longer Windows runs, and restarting Windows restores it \
             (microsoft/WSL#41701); the node keeps working, but more slowly, and this PC \
             has crashed under load in this state",
            took.as_secs_f64()
        );
        if !wsl {
            return;
        }
        let secs = format!("{:.1}", took.as_secs_f64());
        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "card_slow_restart_windows",
                format!(
                    "Your graphics card has become slow to set aside memory ({secs} s, \
                     normally a few thousandths of a second). This builds up the longer \
                     Windows runs without a restart. Restart Windows to bring back full \
                     speed and stability."
                ),
            )
            .with_detail_str(secs)
            .with_toast("warning", 20000),
        );
    }

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
        let since = std::time::Instant::now();
        assert!(
            !manifest_needs_broadcast(Some(&(hash, since)), &hash, false, None),
            "a manifest a peer has already been sent, unchanged, on an ordinary \
             round, is the 825 KB this node must stop republishing every 30 s"
        );
    }

    /// The three reasons it still goes out when the swarm has NOT just heard
    /// it. Each is a way a peer could otherwise be left without a model it
    /// needs, and a node that cannot see a model cannot run it — that failure
    /// has shipped before (gotcha #296).
    #[test]
    fn a_manifest_still_goes_out_when_anyone_could_be_missing_it() {
        let hash = [7u8; 32];
        let other = [9u8; 32];
        let since = std::time::Instant::now();
        assert!(
            manifest_needs_broadcast(None, &hash, false, None),
            "never announced — nobody has it"
        );
        assert!(
            manifest_needs_broadcast(Some(&(other, since)), &hash, false, None),
            "changed since we announced it — peers hold a stale one"
        );
        assert!(
            manifest_needs_broadcast(Some(&(hash, since)), &hash, true, None),
            "a full round repairs a lost message, so it sends what nobody has \
             repeated within the quiet window"
        );
    }

    /// Trickle suppression (RFC 6206): a manifest the whole swarm was gossiped
    /// within `MANIFEST_QUIET_WINDOW` is not repeated — on a full round, and
    /// not as a "change" we merely learned from that same gossip.
    ///
    /// Without it every holder re-announced on its own full round: measured on
    /// the release node 2026-09-23, 0.22 manifests a second at ~35 KB each
    /// were 87% of the gossip it received, and none of them was news.
    #[test]
    fn a_manifest_the_swarm_just_heard_is_not_repeated() {
        let hash = [7u8; 32];
        let other = [9u8; 32];
        let since = std::time::Instant::now();
        let heard = since + std::time::Duration::from_secs(1);
        assert!(
            !manifest_needs_broadcast(Some(&(hash, since)), &hash, true, Some(heard)),
            "a full round must not repeat what another holder said minutes ago \
             — that repetition, from every holder, was 87% of received gossip"
        );
        assert!(
            !manifest_needs_broadcast(Some(&(other, since)), &hash, false, Some(heard)),
            "a change we learned from gossip is one the swarm already has"
        );
        assert!(
            !manifest_needs_broadcast(None, &hash, true, Some(heard)),
            "a model we just picked up, whose manifest the swarm just heard"
        );
    }

    /// **A change of our own BACK to a version heard before we left it is
    /// news** (#224). The rig's shape: a holder broadcast the agreed manifest,
    /// adopted a stranger's contradicting hash (and caught a newcomer up with
    /// it), deleted its part and announced that, then re-fetched the part and
    /// returned to the agreed version — which the swarm had heard minutes
    /// earlier, so it was suppressed for the quiet window (then 30 minutes) and the newcomer
    /// never learned the agreed hash. Treating any heard record as "the swarm
    /// has it" makes the first assert fail.
    #[test]
    fn returning_to_a_version_heard_before_we_left_it_is_announced() {
        let agreed = [7u8; 32];
        let deleted = [9u8; 32];
        let heard_agreed = std::time::Instant::now();
        let left_it = heard_agreed + std::time::Duration::from_secs(60);
        assert!(
            manifest_needs_broadcast(
                Some(&(deleted, left_it)),
                &agreed,
                false,
                Some(heard_agreed)
            ),
            "peers that took the version we announced since have not heard this one"
        );
        assert!(
            !manifest_needs_broadcast(
                Some(&(deleted, left_it)),
                &agreed,
                false,
                Some(left_it + std::time::Duration::from_secs(1))
            ),
            "heard again after we left it — the swarm has it"
        );
    }

    /// An idle card node's capability as one round builds it.
    fn capability_for_test() -> crate::types::NodeCapability {
        crate::types::NodeCapability {
            coord: Some(swarmllm_types::netcoord::NetworkCoord {
                x: 10.0,
                y: -4.0,
                height: 2.0,
                error: 0.3,
            }),
            node_id: crate::types::NodeId([7u8; 32]),
            gpu: Some(crate::types::GpuInfo {
                name: "test card".into(),
                vram_total_mb: 8192,
                vram_available_mb: 5_000,
                compute_capability: None,
                memory_bandwidth_gbps: 448.0,
            }),
            cpu: None,
            ram_total_mb: 16_000,
            ram_available_mb: 12_000,
            ram_model_budget_mb: Some(6_000),
            disk_available_mb: 900_000,
            bandwidth_mbps: 0.0,
            hosted_shards: vec![
                crate::types::ShardId {
                    model_id: crate::types::ModelId("llama-3.2-3b".into()),
                    index: 0,
                },
                crate::types::ShardId {
                    model_id: crate::types::ModelId("llama-3.2-3b".into()),
                    index: 1,
                },
            ],
            max_contribution: crate::types::ContributionLevel::Moderate,
            uptime_seconds: 600,
            version: "0.3.233-alpha".into(),
            region: Some("asia".into()),
            est_tokens_per_sec_7b: 56.0,
            os: Some("linux".into()),
            observed_latencies: vec![crate::types::LatencyObservation {
                peer: crate::types::NodeId([9u8; 32]),
                ms_per_layer: 1.5,
                layer_work: None,
            }],
            relay_capable: false,
            protocol_version: 1,
            features: 7,
            relay_reservations: vec![
                crate::types::NodeId([1u8; 32]),
                crate::types::NodeId([2u8; 32]),
            ],
            anchor_mode: false,
            can_serve_inference: true,
            resident_layers: Vec::new(),
            context_ceiling_tokens: Some(8192),
            model_load_ms_per_gib: Some(900),
            model_memory_ceiling_mb: Some(24_000),
            machine_room_beside_card_mb: None,
            processor_tokens_per_sec_7b: None,
        }
    }

    /// The same capability a few rounds later, on a node where nothing
    /// happened, is not news. Every line here moves on an idle node by itself
    /// — `uptime_seconds` by construction — so if any of them is folded into
    /// the comparison the gate republishes every tick and suppresses nothing,
    /// which is exactly how it would silently stop working.
    #[test]
    fn an_idle_nodes_capability_is_not_news_round_after_round() {
        let sent = capability_for_test();
        let mut now = capability_for_test();
        now.uptime_seconds += 90;
        now.ram_available_mb -= 200;
        now.ram_model_budget_mb = Some(6_000 + 150);
        now.disk_available_mb -= 700;
        now.gpu.as_mut().unwrap().vram_available_mb -= 40;
        now.observed_latencies[0].ms_per_layer = 1.7;
        now.coord.as_mut().unwrap().x += 1.5;
        now.coord.as_mut().unwrap().error = 0.28;
        now.hosted_shards.reverse();
        now.relay_reservations.reverse();
        assert!(
            capability_news(&sent, &now).is_empty(),
            "uptime, small memory/disk/coordinate drift, the latency hint and \
             set order are not news"
        );
        let at = std::time::Instant::now();
        assert!(
            capability_needs_broadcast(Some(&(at, sent.clone())), &now, at + PING_INTERVAL)
                .is_none(),
            "one round after the last publish, an unchanged capability stays quiet"
        );
    }

    /// A worker's own decode speed (FUTURE_WORK #3 (f)) is timed afresh on
    /// every step it runs, so only a move past `CAPABILITY_SPEED_DEADBAND` — or
    /// the figure appearing or going — is news; otherwise every reply a node
    /// served would gossip its capability to the whole swarm (#91).
    #[test]
    fn a_workers_own_speed_is_news_only_past_its_deadband() {
        let with = |speed: Option<u32>| {
            let mut c = capability_for_test();
            c.resident_layers = vec![crate::types::ResidentModelLayers {
                model_id: "llama-3.2-3b".into(),
                layers: 28,
                ranges: Vec::new(),
                decode_us_per_layer: speed,
                card_layers: None,
            }];
            c
        };
        let sent = with(Some(1_000));
        for drift in [1_000, 700, 1_400, 1_499, 667] {
            assert!(
                capability_news(&sent, &with(Some(drift))).is_empty(),
                "{drift} µs a layer against 1000 is inside the band"
            );
        }
        for moved in [1_500, 2_000, 666, 300] {
            assert_eq!(
                capability_news(&sent, &with(Some(moved))),
                ["resident_layers"],
                "{moved} µs a layer against 1000 is news"
            );
        }
        assert_eq!(
            capability_news(&sent, &with(None)),
            ["resident_layers"],
            "a figure that went is news"
        );
        assert_eq!(
            capability_news(&with(None), &with(Some(1_000))),
            ["resident_layers"],
            "a figure that appeared is news"
        );
        // The rest of the entry is still compared exactly.
        let mut grown = with(Some(1_100));
        grown.resident_layers[0].layers = 30;
        assert_eq!(capability_news(&sent, &grown), ["resident_layers"]);
    }

    /// Free RAM moves with everything else running on the host, and a peer
    /// reads it only from a node stating neither a RAM budget nor a card. So
    /// it is never news on a node that states one — the rig found it holding
    /// the gate open — and is still news, past its deadband, where it is read.
    #[test]
    fn free_ram_is_news_only_where_a_peer_reads_it() {
        let sent = capability_for_test();
        let mut now = capability_for_test();
        now.ram_available_mb -= 3_000;
        assert!(capability_news(&sent, &now).is_empty(), "budget stated");
        let (mut sent, mut now) = (sent, now);
        for cap in [&mut sent, &mut now] {
            cap.ram_model_budget_mb = None;
        }
        assert!(capability_news(&sent, &now).is_empty(), "card stated");
        for cap in [&mut sent, &mut now] {
            cap.gpu = None;
        }
        assert_eq!(capability_news(&sent, &now), ["ram_available_mb"]);
        now.ram_available_mb = sent.ram_available_mb - 200;
        assert!(
            capability_news(&sent, &now).is_empty(),
            "inside the deadband"
        );
    }

    /// Everything that changes what a peer would DECIDE about this node goes
    /// out on the next round, not at the heartbeat — a withdrawal above all.
    #[test]
    fn a_capability_change_that_matters_goes_out_at_once() {
        type Change = Box<dyn Fn(&mut crate::types::NodeCapability)>;
        let sent = capability_for_test();
        let cases: Vec<(&str, Change)> = vec![
            (
                "it stopped serving",
                Box::new(|c| c.can_serve_inference = false),
            ),
            (
                "it lost a part",
                Box::new(|c| {
                    c.hosted_shards.pop();
                }),
            ),
            (
                "a model loaded",
                Box::new(|c| {
                    c.resident_layers.push(crate::types::ResidentModelLayers {
                        model_id: "llama-3.2-3b".into(),
                        layers: 28,
                        ranges: Vec::new(),
                        decode_us_per_layer: None,
                        card_layers: None,
                    })
                }),
            ),
            (
                "the card filled",
                Box::new(|c| c.gpu.as_mut().unwrap().vram_available_mb -= 300),
            ),
            (
                "the RAM budget shrank",
                Box::new(|c| c.ram_model_budget_mb = Some(5_600)),
            ),
            (
                "the disk filled",
                Box::new(|c| c.disk_available_mb -= 2_000),
            ),
            (
                "the coordinate moved",
                Box::new(|c| c.coord.as_mut().unwrap().y += 8.0),
            ),
            (
                "the contribution level changed",
                Box::new(|c| c.max_contribution = crate::types::ContributionLevel::Minimal),
            ),
            (
                "a relay was lost",
                Box::new(|c| {
                    c.relay_reservations.pop();
                }),
            ),
            (
                "a load was timed",
                Box::new(|c| c.model_load_ms_per_gib = Some(1_400)),
            ),
            (
                "it updated",
                Box::new(|c| c.version = "0.3.234-alpha".into()),
            ),
        ];
        let at = std::time::Instant::now();
        for (label, change) in cases {
            let mut now = capability_for_test();
            change(&mut now);
            assert!(!capability_news(&sent, &now).is_empty(), "{label} is news");
            assert!(
                capability_needs_broadcast(Some(&(at, sent.clone())), &now, at + PING_INTERVAL)
                    .is_some(),
                "{label} goes out on the next round"
            );
        }
    }

    /// The first capability always goes, and an unchanged one goes again once
    /// the heartbeat has passed: the repeat is what heals a lost message and
    /// moves the figures that stayed inside their deadband.
    #[test]
    fn the_capability_goes_out_first_and_then_at_the_heartbeat() {
        let cap = capability_for_test();
        let at = std::time::Instant::now();
        assert_eq!(
            capability_needs_broadcast(None, &cap, at).as_deref(),
            Some("first")
        );
        let sent = Some((at, cap.clone()));
        assert_eq!(
            capability_needs_broadcast(
                sent.as_ref(),
                &cap,
                at + CAPABILITY_HEARTBEAT.saturating_sub(PING_INTERVAL)
            ),
            None
        );
        assert_eq!(
            capability_needs_broadcast(sent.as_ref(), &cap, at + CAPABILITY_HEARTBEAT).as_deref(),
            Some("heartbeat")
        );
    }

    /// A publish names what changed, so a gate held open by a wobbling figure
    /// can be told from one that works by reading the log.
    #[test]
    fn a_published_capability_names_the_fields_that_changed() {
        let sent = capability_for_test();
        let mut now = capability_for_test();
        now.can_serve_inference = false;
        now.gpu.as_mut().unwrap().vram_available_mb -= 1_000;
        let at = std::time::Instant::now();
        let mut why: Vec<String> = capability_needs_broadcast(Some(&(at, sent)), &now, at)
            .expect("both are news")
            .split(',')
            .map(str::to_string)
            .collect();
        why.sort();
        assert_eq!(why, ["can_serve_inference", "gpu"]);
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
            "the anti-entropy round still fires on its own, to repair a gossip \
             message lost in flight"
        );
        // A newcomer does not depend on it: its direct catch-up is retried every
        // tick until delivered, and it is never marked told without one. What
        // the full round repairs is a LOST gossip message, within
        // `MANIFEST_QUIET_WINDOW` plus one interval.
        assert!(
            (1..FULL_REANNOUNCE_EVERY_TICKS).all(|counter| !manifest_round_is_full(counter)),
            "no tick inside the interval broadcasts everything"
        );
    }

    /// The gossiped repeat stretches to a day only when EVERY connected peer is
    /// repaired another way — one peer without the bit, or one whose capability
    /// has not arrived, keeps the window it had (#91 (2)).
    #[test]
    fn the_repeat_stretches_only_when_every_peer_takes_digests() {
        assert_eq!(
            manifest_quiet_window(8, 8),
            MANIFEST_QUIET_WINDOW_WITH_DIGESTS
        );
        assert_eq!(
            manifest_quiet_window(8, 7),
            MANIFEST_QUIET_WINDOW,
            "a peer repaired only by the gossip repeat still needs it every 2 h"
        );
        assert_eq!(
            manifest_quiet_window(0, 0),
            MANIFEST_QUIET_WINDOW,
            "no peer has been told anything"
        );
    }

    #[test]
    fn a_digest_is_owed_once_a_window() {
        let at = std::time::Instant::now();
        assert!(digest_due(None, at), "a peer with no record is owed one");
        assert!(!digest_due(
            Some(at),
            at + MANIFEST_QUIET_WINDOW.saturating_sub(Duration::from_secs(1))
        ));
        assert!(digest_due(Some(at), at + MANIFEST_QUIET_WINDOW));
    }

    fn monitor_for_test() -> (
        Arc<SharedState>,
        HealthMonitor,
        mpsc::Receiver<NetworkCommand>,
    ) {
        let temp = tempfile::tempdir().unwrap();
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            crate::identity::Identity::generate(),
            crate::storage::db::Database::open(temp.path()).unwrap(),
            Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        let (network_tx, network_rx) = mpsc::channel(1024);
        let (rebalance_tx, _) = mpsc::channel(16);
        let (_, shutdown_rx) = watch::channel(false);
        let monitor = HealthMonitor::new(state.clone(), network_tx, rebalance_tx, shutdown_rx);
        (state, monitor, network_rx)
    }

    /// A connected peer advertising `features`, reachable at `[byte; 4]`.
    fn connect_peer(state: &SharedState, byte: u8, features: u64) -> crate::types::NodeId {
        let node = crate::types::NodeId([byte; 32]);
        let mut capability = capability_for_test();
        capability.node_id = node.clone();
        capability.features = features;
        state.peer_registry.insert(
            node.clone(),
            crate::types::PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability: Some(capability),
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: None,
                ack_srtt_ms: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
                goodput_bytes_per_sec: None,
                goodput_samples: 0,
            },
        );
        state.peer_id_map.insert(node.clone(), vec![byte; 4]);
        state.connected_node_ids.insert(node.clone());
        node
    }

    /// What each peer was sent point to point, by the bytes it is reached at.
    fn direct_sends(
        rx: &mut mpsc::Receiver<NetworkCommand>,
    ) -> std::collections::HashMap<Vec<u8>, Vec<SwarmMessage>> {
        let mut sent: std::collections::HashMap<Vec<u8>, Vec<SwarmMessage>> = Default::default();
        while let Ok(command) = rx.try_recv() {
            if let NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message,
                ..
            } = command
            {
                sent.entry(target_peer_bytes).or_default().push(message);
            }
        }
        sent
    }

    /// How many manifests went on gossip since the last look.
    fn gossiped_manifests(rx: &mut mpsc::Receiver<NetworkCommand>) -> usize {
        let mut gossiped = 0;
        while let Ok(command) = rx.try_recv() {
            if matches!(
                command,
                NetworkCommand::Broadcast(SwarmMessage::ModelManifest(_))
            ) {
                gossiped += 1;
            }
        }
        gossiped
    }

    /// A monitor over the node whose data directory is `dir`, built as a boot
    /// builds it — twice over one directory is a restart.
    fn monitor_over(
        dir: &std::path::Path,
    ) -> (
        Arc<SharedState>,
        HealthMonitor,
        mpsc::Receiver<NetworkCommand>,
    ) {
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            crate::identity::Identity::generate(),
            crate::storage::db::Database::open(dir).unwrap(),
            Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        let (network_tx, network_rx) = mpsc::channel(1024);
        let (rebalance_tx, _) = mpsc::channel(16);
        let (_, shutdown_rx) = watch::channel(false);
        let monitor = HealthMonitor::new(state.clone(), network_tx, rebalance_tx, shutdown_rx);
        (state, monitor, network_rx)
    }

    /// Register a manifest this node published, at version `hash`.
    fn publish_test_manifest(state: &SharedState, hash: [u8; 32]) {
        state
            .model_registry
            .register_manifest(crate::types::ModelManifest {
                id: crate::types::ModelId("restart-model".into()),
                name: "restart-model".into(),
                architecture: crate::types::ModelArchitecture::Llama,
                num_layers: 2,
                num_params_billions: 0.0,
                quantization: crate::types::Quantization::Q4KM,
                total_size_bytes: 0,
                shard_count: 0,
                shards: Vec::new(),
                tokenizer_hash: [0; 32],
                manifest_hash: hash,
                publisher: state.identity.node_id().clone(),
                publish_date: chrono::Utc::now(),
                license: String::new(),
                mmproj: None,
                context_length: Some(4096),
            });
    }

    /// The parts list a newly connected peer is sent at identify time
    /// (`own_shard_announce`) is this round's, kept on EVERY round — a quiet one
    /// that gossips nothing included — and follows a change at once. A node
    /// that had just started, or a peer back from a disconnect, otherwise knew
    /// nothing of what its peers held until their next full round, ~5 minutes
    /// away, and split requests in between were planned without them
    /// (2026-10-27, a 70B refused twice).
    #[tokio::test]
    async fn the_parts_list_a_newcomer_is_sent_is_this_rounds() {
        let (state, mut monitor, _rx) = monitor_for_test();
        let node = state.identity.node_id().clone();
        let model = crate::types::ModelId("m".into());
        let part = |index| crate::types::ShardId {
            model_id: model.clone(),
            index,
        };
        let held: std::collections::HashSet<_> = [model.clone()].into();
        let kept = |state: &SharedState| {
            let announce = state
                .models
                .own_shard_announce
                .load_full()
                .expect("a node holding parts keeps a list to hand a newcomer");
            let mut indices: Vec<u32> = announce.shards.iter().map(|s| s.index).collect();
            indices.sort_unstable();
            assert_eq!(
                announce.complete_for_models,
                vec![model.clone()],
                "the list is complete for the model, so a newcomer drops parts we no longer hold"
            );
            indices
        };

        assert!(
            monitor
                .shard_announce_round(node.clone(), vec![part(0), part(1)], &held)
                .is_some(),
            "a first round is a change, and goes on gossip"
        );
        assert_eq!(kept(&state), vec![0, 1]);

        assert!(
            monitor
                .shard_announce_round(node.clone(), vec![part(0), part(1)], &held)
                .is_none(),
            "an unchanged set between full rounds is not gossiped again"
        );
        assert_eq!(
            kept(&state),
            vec![0, 1],
            "but a newcomer is still handed it"
        );

        assert!(monitor
            .shard_announce_round(node.clone(), vec![part(1)], &held)
            .is_some());
        assert_eq!(
            kept(&state),
            vec![1],
            "a change replaces the kept list at once"
        );

        assert!(monitor
            .shard_announce_round(node, Vec::new(), &Default::default())
            .is_none());
        assert!(
            state.models.own_shard_announce.load_full().is_none(),
            "a node holding nothing hands a newcomer nothing"
        );
    }

    /// #91 (2c): a restart gossips no manifest the swarm was gossiped within
    /// the window. Before, the first round after a restart sent each again in
    /// full (and the boot a second copy, `daemon::background`), because what
    /// the swarm had heard lived only in memory.
    #[tokio::test]
    async fn a_restart_does_not_gossip_what_the_swarm_already_heard() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (state, mut monitor, mut rx) = monitor_over(dir.path());
            publish_test_manifest(&state, [5; 32]);
            connect_peer(&state, 1, crate::types::features::ALL);
            monitor.broadcast_manifests().await;
            assert_eq!(
                gossiped_manifests(&mut rx),
                1,
                "a version the swarm never heard goes on gossip"
            );
        }
        let (state, mut monitor, mut rx) = monitor_over(dir.path());
        publish_test_manifest(&state, [5; 32]);
        connect_peer(&state, 1, crate::types::features::ALL);
        monitor.broadcast_manifests().await;
        assert_eq!(
            gossiped_manifests(&mut rx),
            0,
            "restarted, the node still knows the swarm heard that version"
        );
        publish_test_manifest(&state, [6; 32]);
        monitor.broadcast_manifests().await;
        assert_eq!(
            gossiped_manifests(&mut rx),
            1,
            "a change still goes on gossip"
        );
    }

    /// #91 (2c): a round with no peer connected gossips no manifest — nobody
    /// would hear it, and its window cannot know whether peers take digests —
    /// and the first round with a peer does.
    #[tokio::test]
    async fn a_round_with_no_peer_connected_gossips_no_manifest() {
        let (state, mut monitor, mut rx) = monitor_for_test();
        publish_test_manifest(&state, [5; 32]);
        monitor.broadcast_manifests().await;
        assert_eq!(gossiped_manifests(&mut rx), 0, "no peer, nothing gossiped");
        connect_peer(&state, 1, crate::types::features::ALL);
        monitor.broadcast_manifests().await;
        assert_eq!(
            gossiped_manifests(&mut rx),
            1,
            "the first round with a peer announces the version"
        );
    }

    /// The catch-up that was ~2.9 MB to one newcomer: a peer advertising
    /// `features::MANIFEST_DIGESTS` is sent the versions, never a manifest, and
    /// is sent them again once its record lapses; a peer without the bit is
    /// sent the manifests exactly as before (#91 (2)).
    #[tokio::test]
    async fn a_newcomer_that_takes_digests_is_sent_the_versions_not_the_manifests() {
        let (state, mut monitor, mut rx) = monitor_for_test();
        let our_id = state.identity.node_id().clone();
        let model = crate::types::ModelId("digest-model".into());
        state
            .model_registry
            .register_manifest(crate::types::ModelManifest {
                id: model.clone(),
                name: "digest-model".into(),
                architecture: crate::types::ModelArchitecture::Llama,
                num_layers: 2,
                num_params_billions: 0.0,
                quantization: crate::types::Quantization::Q4KM,
                total_size_bytes: 0,
                shard_count: 0,
                shards: Vec::new(),
                tokenizer_hash: [0; 32],
                manifest_hash: [5; 32],
                publisher: our_id,
                publish_date: chrono::Utc::now(),
                license: String::new(),
                mmproj: None,
                context_length: Some(4096),
            });
        connect_peer(&state, 1, crate::types::features::ALL);
        connect_peer(&state, 2, 0);

        monitor.broadcast_manifests().await;
        let sent = direct_sends(&mut rx);
        let to_new = &sent[&vec![1u8; 4]];
        assert!(
            to_new
                .iter()
                .all(|m| !matches!(m, SwarmMessage::ModelManifest(_))),
            "a peer that pulls is never pushed a manifest it may already hold"
        );
        let digest = to_new
            .iter()
            .find_map(|m| match m {
                SwarmMessage::ManifestDigest(d) => Some(d),
                _ => None,
            })
            .expect("the newcomer is told the versions");
        assert_eq!(digest.versions.len(), 1);
        assert_eq!(digest.versions[0].model_id, model);
        assert_eq!(digest.versions[0].manifest_hash, [5; 32]);
        assert_eq!(digest.versions[0].context_length, Some(4096));
        assert!(
            sent[&vec![2u8; 4]]
                .iter()
                .any(|m| matches!(m, SwarmMessage::ModelManifest(m) if m.id == model)),
            "a peer without the bit is caught up in full, as it always was"
        );

        monitor.broadcast_manifests().await;
        assert!(
            direct_sends(&mut rx).is_empty(),
            "a peer told this round is owed nothing the next"
        );

        monitor
            .digest_sent_at
            .remove(&crate::types::NodeId([1; 32]));
        monitor.broadcast_manifests().await;
        let sent = direct_sends(&mut rx);
        assert!(
            sent[&vec![1u8; 4]]
                .iter()
                .any(|m| matches!(m, SwarmMessage::ManifestDigest(_))),
            "once its record lapses, a told peer is sent the versions again"
        );
        assert!(
            !sent.contains_key(&vec![2u8; 4]),
            "a peer without the bit is repaired by the gossip repeat, not a digest"
        );
    }

    /// A head advertises that it takes a requester's guesses only where its
    /// owner leaves speculation on (#318) — the same answer that opens a
    /// hand-off's slot — so a requester never loads a drafter and reads the
    /// prompt ahead for a head that would check none.
    #[tokio::test]
    async fn a_head_with_speculation_off_does_not_advertise_taking_guesses() {
        use swarmllm_types::features;
        let temp = tempfile::tempdir().unwrap();
        let state_with = |speculate: bool, dir: &std::path::Path| {
            std::fs::create_dir_all(dir).unwrap();
            let mut config = crate::config::Config::default();
            config.inference.speculative_decoding = speculate;
            SharedState::new(
                config,
                crate::identity::Identity::generate(),
                crate::storage::db::Database::open(dir).unwrap(),
                Arc::new(tokio::sync::Mutex::new(
                    crate::inference::executor::ModelExecutor::new(),
                )),
                None,
            )
            .0
        };
        let on = state_with(true, &temp.path().join("on"));
        let off = state_with(false, &temp.path().join("off"));
        assert!(features::supports(
            advertised_features(&on),
            features::REQUESTER_GUESSES
        ));
        assert!(!features::supports(
            advertised_features(&off),
            features::REQUESTER_GUESSES
        ));
        // Everything this build implements goes out either way.
        assert!(features::supports(advertised_features(&off), features::ALL));
    }

    /// A peer that restarted between two rounds is back on a new connection
    /// while still in the connected set, so a set of peers told never saw it
    /// leave: it heard nothing until its repeat digest, up to 2 h later, and
    /// knew no model it held no part of (a rig's requester, 2026-10-20). A
    /// digest-taking peer on a newer connection than it was told on is told
    /// again (BEP 3's bitfield, per connection); a peer sent manifests in full
    /// keeps the old rule.
    #[tokio::test]
    async fn a_peer_back_on_a_new_connection_is_told_the_versions_again() {
        let (state, mut monitor, mut rx) = monitor_for_test();
        publish_test_manifest(&state, [5; 32]);
        let sync = &state.models.manifest_sync;
        for byte in [1u8, 2] {
            let features = if byte == 1 {
                crate::types::features::ALL
            } else {
                0
            };
            connect_peer(&state, byte, features);
            sync.note_new_connection(&crate::types::NodeId([byte; 32]));
        }
        monitor.broadcast_manifests().await;
        let sent = direct_sends(&mut rx);
        assert!(sent.contains_key(&vec![1u8; 4]) && sent.contains_key(&vec![2u8; 4]));
        monitor.broadcast_manifests().await;
        assert!(
            direct_sends(&mut rx).is_empty(),
            "control: told, owed nothing"
        );

        // Both restart between rounds: a new connection each, never absent.
        for byte in [1u8, 2] {
            sync.note_new_connection(&crate::types::NodeId([byte; 32]));
        }
        monitor.broadcast_manifests().await;
        let sent = direct_sends(&mut rx);
        assert!(
            sent.get(&vec![1u8; 4]).is_some_and(|m| m
                .iter()
                .any(|m| matches!(m, SwarmMessage::ManifestDigest(_)))),
            "a peer back on a new connection is told the versions again"
        );
        assert!(
            !sent.contains_key(&vec![2u8; 4]),
            "a peer sent manifests in full is not sent them again per connection"
        );
        monitor.broadcast_manifests().await;
        assert!(
            direct_sends(&mut rx).is_empty(),
            "told on this connection, it is owed nothing more"
        );
    }

    /// A pull its peer did not answer is asked, on the next round, of every
    /// connected peer that takes digests — the one first asked too, which may
    /// only have been slow — and never of a peer without the bit, which could
    /// not decode it (#91 (2)). Marked unsent here so the round need not wait
    /// out `PULL_PATIENCE`.
    #[tokio::test]
    async fn a_pull_left_unanswered_is_asked_of_every_peer_that_takes_digests() {
        let (state, mut monitor, mut rx) = monitor_for_test();
        let first = connect_peer(&state, 1, crate::types::features::ALL);
        connect_peer(&state, 2, crate::types::features::ALL);
        connect_peer(&state, 3, 0);
        let sync = &state.models.manifest_sync;
        let digest = crate::types::ManifestDigest {
            versions: vec![crate::types::ManifestVersion {
                model_id: crate::types::ModelId("m".into()),
                manifest_hash: [7; 32],
                context_length: None,
            }],
        };
        let pulled = sync.versions_to_pull(&digest, &first, |_| None);
        assert_eq!(pulled.len(), 1);
        sync.pull_not_sent(&pulled);

        monitor.broadcast_manifests().await;
        let sent = direct_sends(&mut rx);
        let asked = |byte: u8| {
            sent.get(&vec![byte; 4]).is_some_and(|messages| {
                messages.iter().any(|m| {
                    matches!(m, SwarmMessage::ManifestPull(p)
                        if p.models == [crate::types::ModelId("m".into())])
                })
            })
        };
        assert!(
            asked(1) && asked(2),
            "every peer that takes digests is asked"
        );
        assert!(!asked(3), "never a peer that could not decode it");
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
