//! Node capability, peer info, and node-level stats.

use serde::{Deserialize, Serialize};

use crate::ids::{NodeId, ShardId};
use crate::pool::ContributionMode;

/// Wire-protocol epoch. Bumped ONLY on a genuinely breaking change to the
/// `SwarmMessage` wire format (a variant repurposed or removed — which the
/// project rule forbids without a negotiated fallback). Additive changes (a new
/// variant behind a `features` bit) do NOT bump this. A receiver that sees a
/// higher epoch than it knows treats the peer as "newer, some features I don't
/// speak" and keeps interoperating on the common subset, rather than failing.
pub const PROTOCOL_VERSION: u16 = 1;

/// Optional, additively-negotiated protocol features. A node advertises the set
/// it implements in `NodeCapability::features`; a sender gates an optional
/// message type on the recipient advertising the corresponding bit, so an older
/// node is never handed a variant it can't decode. This is the mechanism that
/// makes network evolution backward-compatible — new features are extensions,
/// never a hard cutover (see `.claude/rules/architecture.md`).
pub mod features {
    /// Understands the NETWORKING_PLAN Phase 1 `RelayedEnvelope` (can receive an
    /// inference message routed through a relay).
    pub const RELAY: u64 = 1 << 0;

    /// The full feature set THIS build implements. Advertised by every node.
    pub const ALL: u64 = RELAY;

    /// Does `advertised` include every bit in `needed`?
    pub fn supports(advertised: u64, needed: u64) -> bool {
        advertised & needed == needed
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub gpu: Option<GpuInfo>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub disk_available_mb: u64,
    pub bandwidth_mbps: f32,
    pub hosted_shards: Vec<ShardId>,
    pub max_contribution: ContributionLevel,
    pub uptime_seconds: u64,
    pub version: String,
    /// Voluntary ISO 3166-1 alpha-2 country code (e.g. "US", "DE").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Estimated tokens/s for a 7B Q4 model based on GPU memory bandwidth.
    /// Used by the scheduler as a speed tie-breaker.
    #[serde(default)]
    pub est_tokens_per_sec_7b: f32,
    /// Snapshot of the sender's observed per-layer latency EMA for other
    /// peers. Lets newly-joining nodes bootstrap Parallax routing from
    /// gossiped foreign observations instead of waiting for their own
    /// direct samples. Receivers merge each entry with a trust-weighted
    /// discount so low-trust senders can't poison routing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observed_latencies: Vec<LatencyObservation>,
    /// NETWORKING_PLAN Phase 1 — this node will forward inference messages
    /// between two peers that cannot reach each other directly (it is
    /// publicly reachable and opted into relaying, e.g. an `--anchor`). Peers
    /// gate relay use on this flag before wrapping traffic for it, so the
    /// capability is negotiated, never assumed. `#[serde(default)]` (false)
    /// means an older node advertising no flag is simply never used as a relay.
    #[serde(default)]
    pub relay_capable: bool,
    /// Wire-protocol epoch this node speaks (see [`PROTOCOL_VERSION`]).
    /// `#[serde(default)]` (0) marks a pre-negotiation node.
    #[serde(default)]
    pub protocol_version: u16,
    /// Bitfield of optional protocol features this node implements (see
    /// [`features`]). A sender gates an optional/new message type on the
    /// recipient advertising the matching bit, so evolution stays additive and
    /// an older node is never handed a variant it can't decode. `0` (default)
    /// means "advertises no optional features" — the safe pre-negotiation base.
    #[serde(default)]
    pub features: u64,
}

/// One entry in `NodeCapability::observed_latencies`: the sender observed
/// this `peer` takes `ms_per_layer` to serve a distributed-inference
/// segment (averaged via the sender's local EMA).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyObservation {
    pub peer: NodeId,
    pub ms_per_layer: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
    /// Memory bandwidth in GB/s, looked up from GPU name.
    #[serde(default)]
    pub memory_bandwidth_gbps: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,
    Moderate,
    Maximum,
}

impl From<ContributionMode> for ContributionLevel {
    fn from(mode: ContributionMode) -> Self {
        match mode {
            ContributionMode::Minimal => Self::Minimal,
            ContributionMode::Moderate => Self::Moderate,
            ContributionMode::Maximum => Self::Maximum,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeStats {
    pub peers_connected: u32,
    pub requests_made: u64,
    pub uptime_start: chrono::DateTime<chrono::Utc>,
    /// NAT status detected by AutoNAT ("Public", "Private", "Unknown").
    #[serde(default)]
    pub nat_status: Option<String>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            peers_connected: 0,
            requests_made: 0,
            uptime_start: chrono::Utc::now(),
            nat_status: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub addresses: Vec<String>,
    pub capability: Option<NodeCapability>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub latency_ms: Option<u32>,
    pub trust_score: f32,
    /// Raw libp2p PeerId bytes for directed request_response messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_id_bytes: Option<Vec<u8>>,
    /// Active inference request count reported by this peer's last health ping/pong.
    #[serde(default)]
    pub active_request_count: u32,
    /// When this peer was first discovered (Unix timestamp).
    /// Used for leaderboard eligibility: peers must be at least `min_lifetime_days` old.
    #[serde(default)]
    pub first_seen: u64,
    /// Number of verified dual-signed credit transactions from this peer.
    /// Used for leaderboard eligibility: peers need `min_verified_transactions`.
    #[serde(default)]
    pub verified_transaction_count: u32,
    /// Whether this peer was discovered via mDNS (on the same LAN).
    /// LAN peers have ~1ms latency and are automatically preferred by the scheduler.
    #[serde(default)]
    pub is_lan_peer: bool,
}

/// Peer Exchange response — a list of known peer multiaddrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerExchangeResponse {
    /// Up to 20 known peer multiaddrs.
    pub peers: Vec<String>,
}
