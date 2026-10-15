use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// Re-export pool types that moved to swarmllm-types crate
use crate::types::NodeId;
pub use crate::types::{
    BlindedPoolInvitation, PoolAcceptance, PoolCreditForward, PoolId, PoolInvitation,
    PoolMembership, PoolRemoval, PoolState, ShardPin,
};

/// A short, human-readable invite code for easy device pool setup.
/// Format: 8 uppercase alphanumeric characters (e.g., "A3F7K2M9").
/// One-time use, expires after `invitation_ttl_hours`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolInviteCode {
    pub code: String,
    pub pool_id: crate::types::NodeId,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    /// BLAKE3 hash of the code — stored instead of plaintext for anti-brute-force
    pub code_hash: [u8; 32],
    /// Set to true once the code has been used (one-time)
    pub consumed: bool,
    /// Who used it, and the invitation it was answered with — so a repeat of
    /// the same device's join request is answered again rather than ignored
    /// (`PoolManager::handle_inbound_join_request`). In memory only: the
    /// stored copy of a code is deleted when it is used.
    #[serde(skip)]
    pub claimed_by: Option<(crate::types::NodeId, uuid::Uuid)>,
}

impl PoolInviteCode {
    /// Generate a new random invite code.
    pub fn generate(pool_id: &crate::types::NodeId, ttl_hours: u32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let chars: Vec<char> = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789".chars().collect(); // No 0/O/1/I to avoid ambiguity
        let code: String = (0..8)
            .map(|_| chars[rng.gen_range(0..chars.len())])
            .collect();
        let code_hash = *blake3::hash(code.as_bytes()).as_bytes();
        let now = chrono::Utc::now();
        Self {
            code: code.clone(),
            pool_id: pool_id.clone(),
            created_at: now,
            expires_at: now + chrono::Duration::hours(ttl_hours as i64),
            code_hash,
            consumed: false,
            claimed_by: None,
        }
    }

    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }
}

/// Is `member` on line now, as this node can tell? The ONE reading of a pool
/// member's liveness: it is this device, it is connected to this one right
/// now, or its last report arrived within two and a half report intervals
/// (`report_every_secs`, the pool's `gossip_interval_secs` — one missed report
/// is not an absence).
///
/// `PoolMembership::online` is not an answer: it is set when a member joins
/// or reports and nothing ever cleared it, so `swarmllm pool status` showed a
/// device stopped two days earlier as on line (field report 2026-10-15). It
/// stays on the wire for older nodes; no reader trusts it.
pub fn member_is_online(
    member: &PoolMembership,
    me: &NodeId,
    connected_now: impl Fn(&NodeId) -> bool,
    now: chrono::DateTime<chrono::Utc>,
    report_every_secs: u64,
) -> bool {
    if member.node_id == *me || connected_now(&member.node_id) {
        return true;
    }
    let window = chrono::Duration::seconds(
        i64::try_from(report_every_secs.saturating_mul(5) / 2).unwrap_or(i64::MAX),
    );
    member
        .last_seen
        .is_some_and(|seen| now.signed_duration_since(seen) <= window)
}

/// Commands sent to the PoolManager task.
#[derive(Debug)]
pub enum PoolCommand {
    CreatePool {
        name: String,
        reply: tokio::sync::oneshot::Sender<Result<PoolState, crate::error::SwarmError>>,
    },
    CreateInvitation {
        invitee: NodeId,
        reply: tokio::sync::oneshot::Sender<Result<PoolInvitation, crate::error::SwarmError>>,
    },
    /// Generate a short invite code that any device can use to join.
    GenerateInviteCode {
        reply: tokio::sync::oneshot::Sender<Result<String, crate::error::SwarmError>>,
    },
    /// Join a pool using an invite code (from the joining device).
    JoinWithCode {
        code: String,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    /// Inbound join request from a peer who has an invite code.
    InboundJoinRequest {
        code_hash: [u8; 32],
        requester: NodeId,
    },
    /// Set the device nickname for this node within the pool.
    SetDeviceName {
        name: String,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    /// Set the credit split percentage (owner only).
    SetCreditSplit {
        pct: u8,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    /// Set contribution level for a member device (owner only). 0-100%.
    SetContributionLevel {
        node_id: NodeId,
        level: u8,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    AcceptInvitation {
        invitation: PoolInvitation,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    RemoveMember {
        node_id: NodeId,
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    LeavePool {
        reply: tokio::sync::oneshot::Sender<Result<(), crate::error::SwarmError>>,
    },
    ProcessCreditForward {
        forward: PoolCreditForward,
    },
    /// Received pool state gossip from the network.
    PoolStateGossip {
        state: PoolState,
    },
    /// R134: received incremental pool state diff from the network. Applied
    /// only when the cached state's `generation` matches the diff's
    /// `parent_generation`; otherwise dropped (next full broadcast resyncs).
    PoolStateDiffGossip {
        diff: swarmllm_types::PoolStateDiff,
    },
    /// Received blinded invitation from the network (SEC-M18).
    InboundBlindedInvitation {
        blinded: BlindedPoolInvitation,
    },
    /// Received acceptance from the network.
    InboundAcceptance {
        acceptance: PoolAcceptance,
    },
    /// Received removal notice from the network.
    InboundRemoval {
        removal: PoolRemoval,
    },
    /// Received member-left notice from the network.
    InboundMemberLeft {
        pool_id: PoolId,
        node_id: NodeId,
        left_at: i64,
        nonce: uuid::Uuid,
        signature: Vec<u8>,
    },
    GetState {
        reply: tokio::sync::oneshot::Sender<Option<PoolState>>,
    },
    GetInvitations {
        reply: tokio::sync::oneshot::Sender<Vec<PoolInvitation>>,
    },
    GetLeaderboard {
        reply: tokio::sync::oneshot::Sender<Vec<LeaderboardEntry>>,
    },
    /// Received device stats report from a pool member.
    InboundDeviceStatsReport {
        pool_id: crate::types::NodeId,
        node_id: crate::types::NodeId,
        device_name: Option<String>,
        stats: crate::types::PoolDeviceStats,
    },
}

/// Sliding-window rate limiter: max `limit` events per `window`.
pub struct PoolRateLimiter {
    events: VecDeque<chrono::DateTime<chrono::Utc>>,
    limit: usize,
    window: chrono::Duration,
}

impl PoolRateLimiter {
    pub fn new(limit: usize, window_hours: u32) -> Self {
        Self {
            events: VecDeque::new(),
            limit,
            window: chrono::Duration::hours(window_hours as i64),
        }
    }

    /// Returns `true` if the action is allowed (under rate limit).
    pub fn check_and_record(&mut self) -> bool {
        let now = chrono::Utc::now();
        let cutoff = now - self.window;

        // Evict expired entries
        while let Some(front) = self.events.front() {
            if *front < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }

        if self.events.len() >= self.limit {
            return false;
        }

        self.events.push_back(now);
        true
    }
}

/// Leaderboard entry for pool credit contributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub node_id: NodeId,
    pub credits_contributed: i64,
    pub rank: u32,
}

// ---- Privacy-preserving blinded broadcast invitation (SEC-M18) ----

/// Compute the invitee commitment for a blinded invitation.
pub fn compute_invitee_commitment(
    invitee_node_id: &NodeId,
    invitation_id: &uuid::Uuid,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"pool_invitee_commit_v1");
    h.update(&invitee_node_id.0);
    h.update(invitation_id.as_bytes());
    *h.finalize().as_bytes()
}

/// Extension methods for BlindedPoolInvitation.
pub trait BlindedPoolInvitationExt {
    fn from_invitation(inv: &PoolInvitation, code_hash: Option<[u8; 32]>) -> BlindedPoolInvitation;
}

impl BlindedPoolInvitationExt for BlindedPoolInvitation {
    fn from_invitation(inv: &PoolInvitation, code_hash: Option<[u8; 32]>) -> BlindedPoolInvitation {
        BlindedPoolInvitation {
            id: inv.id,
            pool_id: inv.pool_id.clone(),
            invitee_commitment: compute_invitee_commitment(&inv.invitee_node_id, &inv.id),
            expires_at: inv.expires_at,
            owner_signature: inv.owner_signature.clone(),
            created_at: inv.created_at,
            code_hash,
        }
    }
}

// ---- Privacy-preserving blind invitation types (test-only until protocol is wired) ----

/// A random blinding factor generated by the invitee.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindingFactor(pub [u8; 32]);

/// A blinded token sent from invitee to pool creator.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindedToken {
    /// H(invitation_id || blinding_factor) — the blinded commitment
    pub commitment: [u8; 32],
    /// The pool this invitation is for
    pub pool_id: PoolId,
    /// When this blinded invite expires
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// The pool creator's signature over a blinded token.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindSignature {
    /// Ed25519 signature over the blind token payload
    pub signature: Vec<u8>,
    /// The commitment that was signed
    pub commitment: [u8; 32],
    /// The pool that issued the signature
    pub pool_id: PoolId,
}

/// The unblinded token held by the invitee to prove membership.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnblindedToken {
    /// The original invitation_id
    pub invitation_id: uuid::Uuid,
    /// The blinding factor (needed for verification)
    pub blinding_factor: BlindingFactor,
    /// The pool creator's signature over the commitment
    pub signature: Vec<u8>,
    /// The pool this token belongs to
    pub pool_id: PoolId,
    /// Expiry time bound cryptographically to the signature
    #[serde(default = "default_expiry")]
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
fn default_expiry() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now() + chrono::Duration::hours(24)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blinded_invitation_commitment_matches_invitee() {
        let invitee = NodeId([42u8; 32]);
        let inv_id = uuid::Uuid::new_v4();
        let commitment = compute_invitee_commitment(&invitee, &inv_id);

        // Same inputs produce the same commitment
        let commitment2 = compute_invitee_commitment(&invitee, &inv_id);
        assert_eq!(commitment, commitment2);

        // Different invitee produces different commitment
        let other = NodeId([99u8; 32]);
        let other_commitment = compute_invitee_commitment(&other, &inv_id);
        assert_ne!(commitment, other_commitment);
    }

    #[test]
    fn blinded_invitation_from_invitation() {
        let pool_id = NodeId([1u8; 32]);
        let invitee = NodeId([2u8; 32]);
        let inv = PoolInvitation {
            id: uuid::Uuid::new_v4(),
            pool_id: pool_id.clone(),
            invitee_node_id: invitee.clone(),
            expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
            owner_signature: vec![0u8; 64],
            created_at: chrono::Utc::now(),
        };

        let blinded = BlindedPoolInvitation::from_invitation(&inv, None);
        assert_eq!(blinded.id, inv.id);
        assert_eq!(blinded.pool_id, pool_id);
        assert_eq!(
            blinded.invitee_commitment,
            compute_invitee_commitment(&invitee, &inv.id)
        );
        assert!(blinded.code_hash.is_none());

        let bound_hash = [7u8; 32];
        let bound = BlindedPoolInvitation::from_invitation(&inv, Some(bound_hash));
        assert_eq!(bound.code_hash, Some(bound_hash));
    }

    /// **A member that stopped reporting is off line, whatever its stored
    /// flag says** (field report 2026-10-15: a device stopped two days earlier
    /// showed `[on]`). Connected now, or reported within the window, it is on.
    #[test]
    fn a_member_that_stopped_reporting_is_off_line() {
        let now = chrono::Utc::now();
        let me = NodeId([1u8; 32]);
        let member = |seen_secs_ago: Option<i64>| PoolMembership {
            node_id: NodeId([2u8; 32]),
            credits_contributed: 0,
            joined_at: now,
            acceptance_signature: vec![],
            invitation_id: uuid::Uuid::new_v4(),
            invitation_expires_at: now,
            device_name: None,
            last_seen: seen_secs_ago.map(|s| now - chrono::Duration::seconds(s)),
            online: true,
            device_stats: None,
            contribution_level: 100,
        };
        let not_connected = |_: &NodeId| false;
        let two_days = Some(2 * 24 * 3600);
        assert!(!member_is_online(
            &member(two_days),
            &me,
            not_connected,
            now,
            600
        ));
        assert!(!member_is_online(
            &member(None),
            &me,
            not_connected,
            now,
            600
        ));
        assert!(
            member_is_online(&member(Some(1_400)), &me, not_connected, now, 600),
            "one report missed"
        );
        assert!(!member_is_online(
            &member(Some(1_600)),
            &me,
            not_connected,
            now,
            600
        ));
        assert!(
            member_is_online(&member(two_days), &me, |_| true, now, 600),
            "connected now"
        );
        let mut myself = member(two_days);
        myself.node_id = me.clone();
        assert!(
            member_is_online(&myself, &me, not_connected, now, 600),
            "this device"
        );
    }

    #[test]
    fn rate_limiter_allows_within_limit() {
        let mut rl = PoolRateLimiter::new(3, 1);
        assert!(rl.check_and_record());
        assert!(rl.check_and_record());
        assert!(rl.check_and_record());
        assert!(!rl.check_and_record()); // 4th should be denied
    }

    #[test]
    fn rate_limiter_evicts_expired() {
        let mut rl = PoolRateLimiter::new(2, 1);
        assert!(rl.check_and_record());
        assert!(rl.check_and_record());
        assert!(!rl.check_and_record());

        // Simulate time passing by manually inserting old timestamps
        rl.events.clear();
        let old = chrono::Utc::now() - chrono::Duration::hours(2);
        rl.events.push_back(old);
        rl.events.push_back(old);

        // Should allow now since old events expire
        assert!(rl.check_and_record());
    }
}
