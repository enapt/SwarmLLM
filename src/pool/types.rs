use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::types::NodeId;

/// Pool identity is the owner's NodeId.
pub type PoolId = NodeId;

/// State of a device pool — owner + list of members.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolState {
    pub pool_id: PoolId,
    pub name: String,
    pub members: Vec<PoolMembership>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub total_lifetime_credits: i64,
}

/// A single pool member record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolMembership {
    pub node_id: NodeId,
    pub credits_contributed: i64,
    pub joined_at: chrono::DateTime<chrono::Utc>,
    pub acceptance_signature: Vec<u8>,
    pub invitation_id: uuid::Uuid,
}

/// Owner-signed invitation targeting a specific invitee.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Invitee-signed acceptance of an invitation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolAcceptance {
    pub invitation_id: uuid::Uuid,
    pub pool_id: PoolId,
    pub invitee_node_id: NodeId,
    pub invitee_signature: Vec<u8>,
    pub accepted_at: chrono::DateTime<chrono::Utc>,
}

/// Owner-signed removal of a member.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolRemoval {
    pub pool_id: PoolId,
    pub removed_node_id: NodeId,
    pub owner_signature: Vec<u8>,
    pub removed_at: chrono::DateTime<chrono::Utc>,
}

/// Dual-signed credit forwarding transaction from member to pool owner.
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
    /// Received invitation from the network.
    InboundInvitation {
        invitation: PoolInvitation,
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
        signature: Vec<u8>,
    },
    GetState {
        reply: tokio::sync::oneshot::Sender<Option<PoolState>>,
    },
    GetInvitations {
        reply: tokio::sync::oneshot::Sender<Vec<PoolInvitation>>,
    },
    GetMembership {
        reply: tokio::sync::oneshot::Sender<Option<PoolMembership>>,
    },
    GetLeaderboard {
        reply: tokio::sync::oneshot::Sender<Vec<LeaderboardEntry>>,
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

/// A pool invitation broadcast that hides the invitee's identity.
/// Instead of broadcasting `invitee_node_id`, we broadcast a BLAKE3
/// commitment: H("pool_invitee_commit_v1" || invitee_node_id || invitation_id).
/// Only the actual invitee can recognize the invitation by recomputing the hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindedPoolInvitation {
    pub id: uuid::Uuid,
    pub pool_id: PoolId,
    /// H("pool_invitee_commit_v1" || invitee_node_id || invitation_id)
    pub invitee_commitment: [u8; 32],
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub owner_signature: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

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

impl BlindedPoolInvitation {
    /// Create a blinded invitation from a full invitation.
    pub fn from_invitation(inv: &PoolInvitation) -> Self {
        Self {
            id: inv.id,
            pool_id: inv.pool_id.clone(),
            invitee_commitment: compute_invitee_commitment(&inv.invitee_node_id, &inv.id),
            expires_at: inv.expires_at,
            owner_signature: inv.owner_signature.clone(),
            created_at: inv.created_at,
        }
    }
}

// ---- Privacy-preserving blind invitation types ----

/// A random blinding factor generated by the invitee.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindingFactor(pub [u8; 32]);

/// A blinded token sent from invitee to pool creator.
/// The pool creator cannot see the underlying invitation identity.
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
/// The invitee reveals this to verifiers who can check the signature.
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

        let blinded = BlindedPoolInvitation::from_invitation(&inv);
        assert_eq!(blinded.id, inv.id);
        assert_eq!(blinded.pool_id, pool_id);
        assert_eq!(
            blinded.invitee_commitment,
            compute_invitee_commitment(&invitee, &inv.id)
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
