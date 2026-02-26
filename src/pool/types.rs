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

#[cfg(test)]
mod tests {
    use super::*;

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
