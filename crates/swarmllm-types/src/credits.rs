//! Credit ledger types: balance, priority tiers, transactions, gossip.

use serde::{Deserialize, Serialize};

use crate::ids::{NodeId, ShardId};

/// Per-node credit ledger snapshot.
///
/// **Persistence contract.** `CreditBalance` is persisted under
/// `TREE_CREDITS / KEY_BALANCE` and restored at daemon startup
/// ([`crate::credit::ledger::CreditLedger::new`] in the main crate). If
/// deserialization fails the node silently starts at zero — which means
/// *adding a new field without `#[serde(default)]` is a credit-loss bug*:
/// old persisted records lack the new field, deserialization rejects them,
/// and the node forgets its lifetime balance on the next start.
///
/// The numeric + timestamp fields below all carry `#[serde(default)]` so
/// that a future field addition is forward-compatible. **Every new field
/// added here MUST also carry `#[serde(default)]`** (and its absence
/// from a persisted record must be a meaningful zero/empty/epoch
/// equivalent — not a security signal — mirroring the
/// `CreditGossip::signature` comment below). `node_id` is intentionally
/// not defaulted: a missing identity is a data-corruption signal, not a
/// schema upgrade, and should fail to deserialize loudly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: NodeId,
    #[serde(default)]
    pub balance: i64,
    #[serde(default)]
    pub lifetime_earned: u64,
    #[serde(default)]
    pub lifetime_spent: u64,
    #[serde(default)]
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
    ///
    /// SEC: NO `#[serde(default)]` here. A permissive default would let any
    /// future handler that forgets to call `verify_balance_report` silently
    /// accept unsigned credit gossip. Forcing the field to be present at
    /// the deserializer means the failure mode is "discard malformed
    /// message" rather than "accept an unsigned report".
    pub signature: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_balance_round_trip_preserves_fields() {
        let cb = CreditBalance {
            node_id: NodeId([7u8; 32]),
            balance: 12_345,
            lifetime_earned: 99_999,
            lifetime_spent: 100,
            last_updated: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH
                + chrono::Duration::seconds(1_700_000_000),
        };
        let s = serde_json::to_string(&cb).unwrap();
        let round: CreditBalance = serde_json::from_str(&s).unwrap();
        assert_eq!(round.node_id, cb.node_id);
        assert_eq!(round.balance, cb.balance);
        assert_eq!(round.lifetime_earned, cb.lifetime_earned);
        assert_eq!(round.lifetime_spent, cb.lifetime_spent);
        assert_eq!(round.last_updated, cb.last_updated);
    }

    #[test]
    fn credit_balance_missing_numeric_fields_use_defaults() {
        // Simulates a future code path reading an older persisted record where
        // one or more of balance / lifetime_earned / lifetime_spent / last_updated
        // were added in a later release. Without #[serde(default)] this
        // deserialization would fail and the daemon would silently start at
        // zero on every restart — losing the lifetime balance. With defaults,
        // missing fields take 0 / epoch and the node keeps participating.
        let json =
            r#"{"node_id":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}"#;
        let cb: CreditBalance =
            serde_json::from_str(json).expect("missing numeric/timestamp fields must default");
        assert_eq!(cb.node_id, NodeId([1u8; 32]));
        assert_eq!(cb.balance, 0);
        assert_eq!(cb.lifetime_earned, 0);
        assert_eq!(cb.lifetime_spent, 0);
        assert_eq!(cb.last_updated, chrono::DateTime::<chrono::Utc>::UNIX_EPOCH);
    }

    #[test]
    fn credit_balance_unknown_fields_are_ignored() {
        // serde_json's default behaviour is to ignore unknown fields; this
        // test pins that behaviour so a deny_unknown_fields attribute is
        // never silently added to CreditBalance. A future-version record
        // with extra fields must still deserialize against current code.
        let json = r#"{
            "node_id":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            "balance":5,
            "lifetime_earned":6,
            "lifetime_spent":1,
            "last_updated":"1970-01-01T00:00:00Z",
            "future_field_added_in_v2":"hello"
        }"#;
        let cb: CreditBalance = serde_json::from_str(json).expect("unknown fields must be ignored");
        assert_eq!(cb.balance, 5);
    }

    #[test]
    fn credit_balance_missing_node_id_fails_loudly() {
        // node_id is intentionally NOT defaulted — a missing identity is
        // a data-corruption signal, not a schema upgrade, and should fail
        // to deserialize loudly so the operator notices.
        let json = r#"{"balance":1,"lifetime_earned":1,"lifetime_spent":0,"last_updated":"1970-01-01T00:00:00Z"}"#;
        let r: Result<CreditBalance, _> = serde_json::from_str(json);
        assert!(
            r.is_err(),
            "missing node_id must error, not silently default"
        );
    }
}
