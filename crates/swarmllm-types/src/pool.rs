//! Device-pool types: state, memberships, invitations, and gossip messages.

use serde::{Deserialize, Serialize};

use crate::ids::NodeId;

/// Contribution mode from node config — maps to ContributionLevel.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContributionMode {
    #[default]
    Minimal,
    Moderate,
    Maximum,
}

/// Pool identity is the owner's NodeId.
pub type PoolId = NodeId;

/// A single pool member record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolMembership {
    pub node_id: NodeId,
    pub credits_contributed: i64,
    pub joined_at: chrono::DateTime<chrono::Utc>,
    pub acceptance_signature: Vec<u8>,
    pub invitation_id: uuid::Uuid,
    /// User-chosen device nickname (e.g., "Gaming PC", "Laptop")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    /// Last time this device was seen on the network (updated via health pings)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen: Option<chrono::DateTime<chrono::Utc>>,
    /// Whether the device is currently online (derived from last_seen < 2 min)
    #[serde(default)]
    pub online: bool,
    /// Per-device stats reported via pool state gossip
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_stats: Option<PoolDeviceStats>,
    /// Contribution level set by the pool owner (0-100%).
    /// Controls how much of this device's resources are dedicated to the network.
    /// 100 = full contribution (default), 50 = half speed/bandwidth, 0 = paused.
    #[serde(default = "default_contribution_level")]
    pub contribution_level: u8,
}

fn default_contribution_level() -> u8 {
    100
}

/// Per-device performance stats within a pool.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PoolDeviceStats {
    /// Forwards served (inference segments processed)
    pub forwards_served: u64,
    /// Total inference requests served
    pub requests_served: u64,
    /// Number of model shards hosted
    pub shards_hosted: u32,
    /// GPU VRAM in MB (0 if CPU-only)
    pub vram_mb: u64,
    /// RAM in MB
    pub ram_mb: u64,
    /// Node uptime in seconds
    pub uptime_secs: u64,
    /// Model IDs currently loaded/hosted
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models_hosted: Vec<String>,
}

/// State of a device pool — owner + list of members.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolState {
    pub pool_id: PoolId,
    pub name: String,
    pub members: Vec<PoolMembership>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub total_lifetime_credits: i64,
    /// Credit split: percentage (0-100) of earnings kept by the member.
    /// The remainder is forwarded to the owner. Default: 0 (all to owner).
    #[serde(default)]
    pub member_credit_split_pct: u8,
    /// Shard pins: owner assigns specific models/shards to specific devices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_pins: Vec<ShardPin>,
}

/// A shard pinning assignment: a model (or specific shards) pinned to a target device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardPin {
    /// Model ID to pin.
    pub model_id: String,
    /// Specific shard indices to pin, or empty for all shards.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_indices: Vec<u32>,
    /// Target device NodeId.
    pub target_node_id: NodeId,
}

impl ShardPin {
    /// Check whether this pin applies to a given model/node/shard combination.
    /// Empty `shard_indices` means "all shards".
    pub fn matches(&self, model_id: &str, node_id: &NodeId, shard_index: u32) -> bool {
        self.model_id == model_id
            && self.target_node_id == *node_id
            && (self.shard_indices.is_empty() || self.shard_indices.contains(&shard_index))
    }

    /// Check whether this pin applies to a given model and shard (any node).
    pub fn matches_shard(&self, model_id: &str, shard_index: u32) -> bool {
        self.model_id == model_id
            && (self.shard_indices.is_empty() || self.shard_indices.contains(&shard_index))
    }
}

/// Invitation to join a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Privacy-preserving blinded invitation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindedPoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    /// H("pool_invitee_commit_v1" || invitee_node_id || invitation_id)
    pub invitee_commitment: [u8; 32],
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// SEC: BLAKE3 hash of the invite code that triggered this invitation, if any.
    /// `Some(hash)` for code-based JoinRequest auto-invites; `None` for direct
    /// owner-initiated invites. The receiver's auto-accept gate verifies that
    /// `code_hash == auto_accept_code_hash` to prevent an attacker from
    /// hijacking a `JoinRequest` (broadcast in cleartext) by issuing their own
    /// invitation under a pool they control. Without this binding, the
    /// invitee would auto-accept ANY pool's invitation that arrives within
    /// the 5-minute auto-accept window after a code-based join request.
    #[serde(default)]
    pub code_hash: Option<[u8; 32]>,
}

/// Acceptance of a pool invitation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolAcceptance {
    pub invitation_id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub invitee_signature: Vec<u8>,
    pub accepted_at: chrono::DateTime<chrono::Utc>,
}

/// Credit forwarding within a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolCreditForward {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    pub from_node_id: NodeId,
    pub to_node_id: NodeId,
    pub amount: i64,
    pub member_signature: Vec<u8>,
    pub owner_signature: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Removal of a member from a pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolRemoval {
    pub pool_id: PoolId,
    pub removed_node_id: NodeId,
    pub owner_signature: Vec<u8>,
    pub removed_at: chrono::DateTime<chrono::Utc>,
    /// Unique ID to prevent replay attacks (new field, defaults to nil for old messages)
    #[serde(default = "uuid::Uuid::nil")]
    pub removal_id: uuid::Uuid,
}

/// Messages related to device pool management, sent over GossipSub.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolMessage {
    /// SEC-M18: Privacy-preserving blinded invitation broadcast.
    BlindedInvitation(BlindedPoolInvitation),
    Acceptance(PoolAcceptance),
    StateGossip(PoolState),
    CreditForward(PoolCreditForward),
    Removal(PoolRemoval),
    MemberLeft {
        pool_id: NodeId,
        node_id: NodeId,
        /// Unix timestamp (seconds) when the leave notice was created.
        /// Receivers MUST reject notices more than ~5 minutes out of range,
        /// and dedup on the UUID below to prevent replay.
        #[serde(default)]
        left_at: i64,
        #[serde(default)]
        nonce: uuid::Uuid,
        signature: Vec<u8>,
    },
    /// Join request from a device that has an invite code.
    /// The code_hash is BLAKE3(code) — the code itself is never sent over the network.
    JoinRequest {
        code_hash: [u8; 32],
        requester: NodeId,
        /// Ed25519 signature over BLAKE3("pool_join_request_v1" || code_hash || requester)
        signature: Vec<u8>,
    },
    /// Periodic stats + nickname report from a pool member to the leader.
    DeviceStatsReport {
        pool_id: NodeId,
        node_id: NodeId,
        device_name: Option<String>,
        stats: PoolDeviceStats,
    },
}
