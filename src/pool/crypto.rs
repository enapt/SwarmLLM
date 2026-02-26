use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::SwarmError;
use crate::identity::Identity;
use crate::pool::types::{PoolAcceptance, PoolCreditForward, PoolId, PoolInvitation, PoolRemoval};
use crate::types::NodeId;

// Domain-separated BLAKE3 prefixes per the plan.
const PREFIX_INVITATION: &[u8] = b"pool_invitation_v1";
const PREFIX_ACCEPTANCE: &[u8] = b"pool_acceptance_v1";
const PREFIX_REMOVAL: &[u8] = b"pool_removal_v1";
const PREFIX_CREDIT_FORWARD: &[u8] = b"pool_credit_forward_v1";

/// Create a pool invitation signed by the pool owner.
pub fn create_invitation(
    identity: &Identity,
    pool_id: &PoolId,
    invitee: &NodeId,
    ttl_hours: u32,
) -> PoolInvitation {
    let id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    let expires_at = now + chrono::Duration::hours(ttl_hours as i64);

    let payload = invitation_payload(&id, pool_id, invitee, &expires_at);
    let signature = identity.sign(&payload);

    PoolInvitation {
        id,
        pool_id: pool_id.clone(),
        invitee_node_id: invitee.clone(),
        expires_at,
        owner_signature: signature,
        created_at: now,
    }
}

/// Verify an invitation's owner signature.
pub fn verify_invitation(
    invitation: &PoolInvitation,
    owner_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    let payload = invitation_payload(
        &invitation.id,
        &invitation.pool_id,
        &invitation.invitee_node_id,
        &invitation.expires_at,
    );
    verify_sig(&invitation.owner_signature, &payload, owner_key)
}

/// Create an acceptance signed by the invitee.
pub fn create_acceptance(identity: &Identity, invitation: &PoolInvitation) -> PoolAcceptance {
    let now = chrono::Utc::now();
    let payload = acceptance_payload(
        &invitation.id,
        &invitation.pool_id,
        &invitation.invitee_node_id,
    );
    let signature = identity.sign(&payload);

    PoolAcceptance {
        invitation_id: invitation.id,
        pool_id: invitation.pool_id.clone(),
        invitee_node_id: invitation.invitee_node_id.clone(),
        invitee_signature: signature,
        accepted_at: now,
    }
}

/// Verify an acceptance's invitee signature.
pub fn verify_acceptance(
    acceptance: &PoolAcceptance,
    invitee_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    let payload = acceptance_payload(
        &acceptance.invitation_id,
        &acceptance.pool_id,
        &acceptance.invitee_node_id,
    );
    verify_sig(&acceptance.invitee_signature, &payload, invitee_key)
}

/// Create a removal notice signed by the pool owner.
pub fn create_removal(identity: &Identity, pool_id: &PoolId, removed_node: &NodeId) -> PoolRemoval {
    let now = chrono::Utc::now();
    let payload = removal_payload(pool_id, removed_node, &now);
    let signature = identity.sign(&payload);

    PoolRemoval {
        pool_id: pool_id.clone(),
        removed_node_id: removed_node.clone(),
        owner_signature: signature,
        removed_at: now,
    }
}

/// Verify a removal notice's owner signature.
pub fn verify_removal(removal: &PoolRemoval, owner_key: &VerifyingKey) -> Result<(), SwarmError> {
    let payload = removal_payload(
        &removal.pool_id,
        &removal.removed_node_id,
        &removal.removed_at,
    );
    verify_sig(&removal.owner_signature, &payload, owner_key)
}

/// Create a credit forward transaction signed by the member (first signature).
pub fn create_credit_forward(
    identity: &Identity,
    pool_id: &PoolId,
    from: &NodeId,
    to: &NodeId,
    amount: i64,
) -> PoolCreditForward {
    let id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    let payload = credit_forward_payload(&id, pool_id, from, to, amount, &now);
    let signature = identity.sign(&payload);

    PoolCreditForward {
        id,
        pool_id: pool_id.clone(),
        from_node_id: from.clone(),
        to_node_id: to.clone(),
        amount,
        member_signature: signature,
        owner_signature: Vec::new(),
        timestamp: now,
    }
}

/// Co-sign a credit forward as the pool owner (second signature).
pub fn cosign_credit_forward(
    identity: &Identity,
    forward: &mut PoolCreditForward,
    member_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    // Verify member's signature first
    let payload = credit_forward_payload(
        &forward.id,
        &forward.pool_id,
        &forward.from_node_id,
        &forward.to_node_id,
        forward.amount,
        &forward.timestamp,
    );
    verify_sig(&forward.member_signature, &payload, member_key)?;

    // Owner co-signs
    forward.owner_signature = identity.sign(&payload);
    Ok(())
}

// ---- Payload builders ----

fn invitation_payload(
    id: &uuid::Uuid,
    pool_id: &PoolId,
    invitee: &NodeId,
    expires_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_INVITATION);
    hasher.update(id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&invitee.0);
    hasher.update(expires_at.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn acceptance_payload(invitation_id: &uuid::Uuid, pool_id: &PoolId, invitee: &NodeId) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_ACCEPTANCE);
    hasher.update(invitation_id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&invitee.0);
    hasher.finalize().as_bytes().to_vec()
}

fn removal_payload(
    pool_id: &PoolId,
    removed_node: &NodeId,
    removed_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_REMOVAL);
    hasher.update(&pool_id.0);
    hasher.update(&removed_node.0);
    hasher.update(removed_at.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn credit_forward_payload(
    id: &uuid::Uuid,
    pool_id: &PoolId,
    from: &NodeId,
    to: &NodeId,
    amount: i64,
    timestamp: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_CREDIT_FORWARD);
    hasher.update(id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&from.0);
    hasher.update(&to.0);
    hasher.update(&amount.to_le_bytes());
    hasher.update(timestamp.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn verify_sig(sig_bytes: &[u8], payload: &[u8], key: &VerifyingKey) -> Result<(), SwarmError> {
    if sig_bytes.len() != 64 {
        return Err(SwarmError::InvalidSignature);
    }
    let sig = Signature::from_bytes(
        sig_bytes
            .try_into()
            .map_err(|_| SwarmError::Internal("Invalid signature length".into()))?,
    );
    key.verify(payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    #[test]
    fn invitation_create_and_verify() {
        let owner = Identity::generate();
        let invitee = Identity::generate();
        let pool_id = owner.node_id().clone();

        let invitation = create_invitation(&owner, &pool_id, invitee.node_id(), 24);

        assert!(verify_invitation(&invitation, &owner.verifying_key()).is_ok());
        // Wrong key should fail
        assert!(verify_invitation(&invitation, &invitee.verifying_key()).is_err());
    }

    #[test]
    fn acceptance_create_and_verify() {
        let owner = Identity::generate();
        let invitee = Identity::generate();
        let pool_id = owner.node_id().clone();

        let invitation = create_invitation(&owner, &pool_id, invitee.node_id(), 24);
        let acceptance = create_acceptance(&invitee, &invitation);

        assert!(verify_acceptance(&acceptance, &invitee.verifying_key()).is_ok());
        // Wrong key should fail
        assert!(verify_acceptance(&acceptance, &owner.verifying_key()).is_err());
    }

    #[test]
    fn removal_create_and_verify() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let pool_id = owner.node_id().clone();

        let removal = create_removal(&owner, &pool_id, member.node_id());

        assert!(verify_removal(&removal, &owner.verifying_key()).is_ok());
        assert!(verify_removal(&removal, &member.verifying_key()).is_err());
    }

    #[test]
    fn credit_forward_create_and_cosign() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let pool_id = owner.node_id().clone();

        let mut forward =
            create_credit_forward(&member, &pool_id, member.node_id(), owner.node_id(), 100);

        assert!(forward.owner_signature.is_empty());

        // Owner co-signs
        cosign_credit_forward(&owner, &mut forward, &member.verifying_key()).unwrap();
        assert!(!forward.owner_signature.is_empty());
    }

    #[test]
    fn credit_forward_rejects_wrong_member_key() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let imposter = Identity::generate();
        let pool_id = owner.node_id().clone();

        let mut forward =
            create_credit_forward(&member, &pool_id, member.node_id(), owner.node_id(), 100);

        // Owner tries to co-sign but uses wrong member key
        assert!(cosign_credit_forward(&owner, &mut forward, &imposter.verifying_key()).is_err());
    }

    #[test]
    fn invitation_payload_is_deterministic() {
        let id = uuid::Uuid::new_v4();
        let pool_id = NodeId([1u8; 32]);
        let invitee = NodeId([2u8; 32]);
        let expires = chrono::Utc::now();

        let p1 = invitation_payload(&id, &pool_id, &invitee, &expires);
        let p2 = invitation_payload(&id, &pool_id, &invitee, &expires);
        assert_eq!(p1, p2);
    }
}
