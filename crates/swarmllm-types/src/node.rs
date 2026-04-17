//! Node capability, peer info, and node-level stats.

use serde::{Deserialize, Serialize};

use crate::ids::{NodeId, ShardId};
use crate::pool::ContributionMode;

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
    pub requests_served: u64,
    pub requests_made: u64,
    /// Layer forwards processed for other nodes in distributed inference.
    pub forwards_served: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub uptime_start: chrono::DateTime<chrono::Utc>,
    /// NAT status detected by AutoNAT ("Public", "Private", "Unknown").
    #[serde(default)]
    pub nat_status: Option<String>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            peers_connected: 0,
            requests_served: 0,
            requests_made: 0,
            forwards_served: 0,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
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
