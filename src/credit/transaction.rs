use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::SwarmError;
use crate::identity::Identity;
use crate::types::{CreditTransaction, NodeId, TransactionReason};

/// Create a new credit transaction initiated by the serving node.
///
/// The serving node creates the transaction and signs it first.
/// The resulting transaction has `signature_from` populated
/// but `signature_to` empty (to be co-signed by the requesting node).
pub fn create_transaction(
    identity: &Identity,
    from: NodeId,
    to: NodeId,
    amount: i64,
    reason: TransactionReason,
) -> Result<CreditTransaction, SwarmError> {
    let id = uuid::Uuid::new_v4();
    let timestamp = chrono::Utc::now();

    // Build the signing payload (deterministic)
    let payload = build_signing_payload(&id, &from, &to, amount, &reason, &timestamp);

    // Serving node signs first
    let signature_from = identity.sign(&payload);

    Ok(CreditTransaction {
        id,
        from,
        to,
        amount,
        reason,
        timestamp,
        signature_from,
        signature_to: Vec::new(), // To be filled by co-signer
    })
}

/// Co-sign a transaction as the requesting node.
///
/// Verifies the serving node's signature first, then adds our own.
pub fn cosign_transaction(
    identity: &Identity,
    tx: &mut CreditTransaction,
    from_verifying_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    // Verify the serving node's signature
    verify_single_signature(tx, from_verifying_key, true)?;

    // Build the same payload and sign
    let payload = build_signing_payload(
        &tx.id,
        &tx.from,
        &tx.to,
        tx.amount,
        &tx.reason,
        &tx.timestamp,
    );

    tx.signature_to = identity.sign(&payload);

    Ok(())
}

/// Verify both signatures on a fully-signed transaction.
/// Also checks for UUID replay: rejects transactions already recorded in the database.
pub fn verify_transaction(
    tx: &CreditTransaction,
    from_key: &VerifyingKey,
    to_key: &VerifyingKey,
    db: &crate::storage::db::Database,
) -> Result<(), SwarmError> {
    // SEC-C3: Check for transaction replay via UUID deduplication
    if let Ok(Some(_)) = db
        .get_json::<CreditTransaction>(crate::credit::ledger::TREE_TRANSACTIONS, &tx.id.to_string())
    {
        return Err(SwarmError::Internal(format!(
            "Duplicate transaction: {}",
            tx.id
        )));
    }

    verify_single_signature(tx, from_key, true)?;
    verify_single_signature(tx, to_key, false)?;
    tracing::debug!(
        tx_id = %tx.id,
        amount = tx.amount,
        "DIAG: verify_transaction OK — dual signatures valid"
    );
    Ok(())
}

/// Verify both signatures on a credit transaction without the DB replay check.
/// Used by the gossip dispatch handler which does its own replay check separately.
pub fn verify_single_signatures(
    tx: &CreditTransaction,
    from_key: &VerifyingKey,
    to_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    verify_single_signature(tx, from_key, true)?;
    verify_single_signature(tx, to_key, false)?;
    Ok(())
}

/// Verify a single signature on the transaction.
fn verify_single_signature(
    tx: &CreditTransaction,
    key: &VerifyingKey,
    is_from: bool,
) -> Result<(), SwarmError> {
    let sig_bytes = if is_from {
        &tx.signature_from
    } else {
        &tx.signature_to
    };

    if sig_bytes.len() != 64 {
        return Err(SwarmError::InvalidSignature);
    }

    let sig = Signature::from_bytes(
        sig_bytes
            .as_slice()
            .try_into()
            .map_err(|_| SwarmError::Internal("Invalid signature length".into()))?,
    );

    let payload = build_signing_payload(
        &tx.id,
        &tx.from,
        &tx.to,
        tx.amount,
        &tx.reason,
        &tx.timestamp,
    );

    key.verify(&payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)
}

/// Build a deterministic signing payload from transaction fields.
///
/// We hash the concatenation of all fields to produce a fixed-size payload.
fn build_signing_payload(
    id: &uuid::Uuid,
    from: &NodeId,
    to: &NodeId,
    amount: i64,
    reason: &TransactionReason,
    timestamp: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(id.as_bytes());
    hasher.update(&from.0);
    hasher.update(&to.0);
    hasher.update(&amount.to_le_bytes());
    hasher.update(&serde_json::to_vec(reason).unwrap_or_default());
    hasher.update(timestamp.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    #[test]
    fn create_and_cosign_transaction() {
        let server = Identity::generate();
        let client = Identity::generate();

        let mut tx = create_transaction(
            &server,
            server.node_id().clone(),
            client.node_id().clone(),
            100,
            TransactionReason::InferenceServed {
                request_id: uuid::Uuid::new_v4(),
                tokens: 10,
            },
        )
        .unwrap();

        // Verify server's signature
        assert!(verify_single_signature(&tx, &server.verifying_key(), true).is_ok());

        // Client hasn't signed yet
        assert!(tx.signature_to.is_empty());

        // Client co-signs
        cosign_transaction(&client, &mut tx, &server.verifying_key()).unwrap();

        // Now both signatures should be valid
        assert!(!tx.signature_to.is_empty());
        let tmp = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(tmp.path()).unwrap();
        verify_transaction(&tx, &server.verifying_key(), &client.verifying_key(), &db).unwrap();
    }

    #[test]
    fn wrong_key_fails_verification() {
        let server = Identity::generate();
        let client = Identity::generate();
        let imposter = Identity::generate();

        let tx = create_transaction(
            &server,
            server.node_id().clone(),
            client.node_id().clone(),
            50,
            TransactionReason::ShardSeeding {
                shard_id: crate::types::ShardId {
                    model_id: crate::types::ModelId("test".into()),
                    index: 0,
                },
                bytes: 1024,
            },
        )
        .unwrap();

        // Imposter's key should not verify server's signature
        assert!(verify_single_signature(&tx, &imposter.verifying_key(), true).is_err());
    }

    #[test]
    fn tampered_amount_fails_verification() {
        let server = Identity::generate();
        let client = Identity::generate();

        let mut tx = create_transaction(
            &server,
            server.node_id().clone(),
            client.node_id().clone(),
            100,
            TransactionReason::InferenceServed {
                request_id: uuid::Uuid::new_v4(),
                tokens: 10,
            },
        )
        .unwrap();

        // Tamper with the amount
        tx.amount = 999;

        // Signature should fail because payload has changed
        assert!(verify_single_signature(&tx, &server.verifying_key(), true).is_err());
    }

    #[test]
    fn cosign_rejects_invalid_from_signature() {
        let server = Identity::generate();
        let client = Identity::generate();
        let imposter = Identity::generate();

        // Imposter creates a transaction pretending to be server
        let mut tx = create_transaction(
            &imposter,
            server.node_id().clone(), // claims to be server
            client.node_id().clone(),
            100,
            TransactionReason::InferenceServed {
                request_id: uuid::Uuid::new_v4(),
                tokens: 10,
            },
        )
        .unwrap();

        // Client should reject: the from signature was made by imposter, not server
        assert!(cosign_transaction(&client, &mut tx, &server.verifying_key()).is_err());
    }

    #[test]
    fn signing_payload_is_deterministic() {
        let id = uuid::Uuid::new_v4();
        let from = NodeId([1u8; 32]);
        let to = NodeId([2u8; 32]);
        let amount = 42i64;
        let reason = TransactionReason::InferenceServed {
            request_id: uuid::Uuid::nil(),
            tokens: 5,
        };
        let timestamp = chrono::Utc::now();

        let p1 = build_signing_payload(&id, &from, &to, amount, &reason, &timestamp);
        let p2 = build_signing_payload(&id, &from, &to, amount, &reason, &timestamp);

        assert_eq!(p1, p2);
    }
}
