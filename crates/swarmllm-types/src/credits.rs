//! Credit ledger types: balance, priority tiers, transactions, gossip.

use serde::{Deserialize, Serialize};

use crate::ids::{NodeId, ShardId};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: NodeId,
    pub balance: i64,
    pub lifetime_earned: u64,
    pub lifetime_spent: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityTier {
    Bronze = 0,
    Silver = 1,
    Gold = 2,
    Platinum = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditTransaction {
    pub id: uuid::Uuid,
    pub from: NodeId,
    pub to: NodeId,
    pub amount: i64,
    pub reason: TransactionReason,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature_from: Vec<u8>,
    pub signature_to: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransactionReason {
    InferenceServed { request_id: uuid::Uuid, tokens: u32 },
    ShardSeeding { shard_id: ShardId, bytes: u64 },
}

/// Bucketed credit balance gossip for network-wide percentile estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditGossip {
    pub node_id: NodeId,
    pub balance_bucket: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Ed25519 signature over (node_id || balance_bucket || timestamp_secs).
    /// Required — unsigned gossip is rejected.
    #[serde(default)]
    pub signature: Vec<u8>,
}
