use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use dashmap::DashMap;
use hkdf::Hkdf;
use sha2::Sha256;
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

use crate::error::SwarmError;
use crate::types::NodeId;

/// A cached pairwise session derived from X25519 ECDH.
pub struct CachedSession {
    cipher_key: [u8; 32],
    send_nonce: AtomicU64,
    created_at: Instant,
}

impl CachedSession {
    fn new(cipher_key: [u8; 32]) -> Self {
        Self {
            cipher_key,
            send_nonce: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    fn next_nonce(&self) -> Result<[u8; 12], SwarmError> {
        let counter = self.send_nonce.fetch_add(1, Ordering::SeqCst);
        if counter == u64::MAX {
            return Err(SwarmError::NonceOverflow);
        }
        let mut nonce = [0u8; 12];
        nonce[4..12].copy_from_slice(&counter.to_le_bytes());
        Ok(nonce)
    }
}

/// Manages pairwise encryption sessions with peers.
pub struct SessionManager {
    local_secret: StaticSecret,
    local_public: PublicKey,
    sessions: DashMap<NodeId, CachedSession>,
}

impl SessionManager {
    /// Create a new SessionManager from an Ed25519 signing key.
    pub fn from_ed25519_key(signing_key_bytes: &[u8; 32]) -> Self {
        let secret = ed25519_to_x25519_secret(signing_key_bytes);
        let public = PublicKey::from(&secret);
        Self {
            local_secret: secret,
            local_public: public,
            sessions: DashMap::new(),
        }
    }

    /// Get this node's X25519 public key.
    pub fn local_public_key(&self) -> &PublicKey {
        &self.local_public
    }

    /// Establish a session with a peer given their X25519 public key.
    pub fn establish_session(&self, peer: &NodeId, peer_x25519_pub: PublicKey) {
        let shared_secret = self.local_secret.diffie_hellman(&peer_x25519_pub);
        let cipher_key = derive_cipher_key(
            shared_secret.as_bytes(),
            &self.local_public,
            &peer_x25519_pub,
        );
        self.sessions
            .insert(peer.clone(), CachedSession::new(cipher_key));
        tracing::debug!(peer = %peer, "Established encryption session");
    }

    /// Check if a session exists for the given peer.
    pub fn has_session(&self, peer: &NodeId) -> bool {
        self.sessions.contains_key(peer)
    }

    /// Seal (encrypt) data for a specific peer.
    /// `aad` is additional authenticated data (e.g., the cleartext header).
    pub fn seal(&self, peer: &NodeId, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, SwarmError> {
        let session = self
            .sessions
            .get(peer)
            .ok_or_else(|| SwarmError::NoSession(peer.clone()))?;
        let nonce_bytes = session.next_nonce()?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let cipher = ChaCha20Poly1305::new_from_slice(&session.cipher_key)
            .map_err(|e| SwarmError::Encryption(format!("Cipher init failed: {e}")))?;

        let payload = chacha20poly1305::aead::Payload {
            msg: plaintext,
            aad,
        };
        let ciphertext = cipher
            .encrypt(nonce, payload)
            .map_err(|e| SwarmError::Encryption(format!("Seal failed: {e}")))?;

        // Prepend nonce to ciphertext: [12B nonce][ciphertext+tag]
        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Open (decrypt) data from a specific peer.
    pub fn open(&self, peer: &NodeId, sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>, SwarmError> {
        if sealed.len() < 12 {
            return Err(SwarmError::DecryptionFailed);
        }
        let session = self
            .sessions
            .get(peer)
            .ok_or_else(|| SwarmError::NoSession(peer.clone()))?;

        let nonce = Nonce::from_slice(&sealed[..12]);
        let ciphertext = &sealed[12..];

        let cipher = ChaCha20Poly1305::new_from_slice(&session.cipher_key)
            .map_err(|e| SwarmError::Encryption(format!("Cipher init failed: {e}")))?;

        let payload = chacha20poly1305::aead::Payload {
            msg: ciphertext,
            aad,
        };
        cipher
            .decrypt(nonce, payload)
            .map_err(|_| SwarmError::DecryptionFailed)
    }

    /// Evict sessions older than `max_age`.
    pub fn evict_stale(&self, max_age: std::time::Duration) {
        let now = Instant::now();
        self.sessions.retain(|peer, session| {
            let keep = now.duration_since(session.created_at) < max_age;
            if !keep {
                tracing::debug!(peer = %peer, "Evicted stale encryption session");
            }
            keep
        });
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Seal with an ephemeral X25519 keypair for forward secrecy.
/// Returns `(ephemeral_public_key, sealed_data)`.
pub fn ephemeral_seal(
    recipient_pub: &PublicKey,
    plaintext: &[u8],
    aad: &[u8],
) -> Result<([u8; 32], Vec<u8>), SwarmError> {
    let ephemeral_secret = EphemeralSecret::random_from_rng(rand::rngs::OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);
    let shared_secret = ephemeral_secret.diffie_hellman(recipient_pub);

    let cipher_key = derive_cipher_key(shared_secret.as_bytes(), &ephemeral_public, recipient_pub);
    let cipher = ChaCha20Poly1305::new_from_slice(&cipher_key)
        .map_err(|e| SwarmError::Encryption(format!("Cipher init failed: {e}")))?;

    let nonce_bytes = [0u8; 12]; // Single-use key, nonce=0 is safe
    let nonce = Nonce::from_slice(&nonce_bytes);

    let payload = chacha20poly1305::aead::Payload {
        msg: plaintext,
        aad,
    };
    let ciphertext = cipher
        .encrypt(nonce, payload)
        .map_err(|e| SwarmError::Encryption(format!("Ephemeral seal failed: {e}")))?;

    Ok((ephemeral_public.to_bytes(), ciphertext))
}

/// Open data sealed with an ephemeral keypair.
pub fn ephemeral_open(
    local_secret: &StaticSecret,
    ephemeral_pub_bytes: &[u8; 32],
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, SwarmError> {
    let ephemeral_public = PublicKey::from(*ephemeral_pub_bytes);
    let local_public = PublicKey::from(local_secret);
    let shared_secret = local_secret.diffie_hellman(&ephemeral_public);

    let cipher_key = derive_cipher_key(shared_secret.as_bytes(), &ephemeral_public, &local_public);
    let cipher = ChaCha20Poly1305::new_from_slice(&cipher_key)
        .map_err(|e| SwarmError::Encryption(format!("Cipher init failed: {e}")))?;

    let nonce = Nonce::from_slice(&[0u8; 12]);
    let payload = chacha20poly1305::aead::Payload {
        msg: ciphertext,
        aad,
    };
    cipher
        .decrypt(nonce, payload)
        .map_err(|_| SwarmError::DecryptionFailed)
}

/// Convert an Ed25519 signing key to an X25519 static secret.
/// Per RFC 7748: SHA-512 the Ed25519 key, take low 32 bytes, apply clamping.
pub fn ed25519_to_x25519_secret(signing_key_bytes: &[u8; 32]) -> StaticSecret {
    use sha2::{Digest, Sha512};
    let hash = Sha512::digest(signing_key_bytes);
    let mut x25519_bytes = [0u8; 32];
    x25519_bytes.copy_from_slice(&hash[..32]);
    // RFC 7748 clamping
    x25519_bytes[0] &= 248;
    x25519_bytes[31] &= 127;
    x25519_bytes[31] |= 64;
    StaticSecret::from(x25519_bytes)
}

/// Convert an Ed25519 public key (verifying key bytes) to an X25519 public key.
/// Uses curve25519-dalek's birational map from Edwards to Montgomery form.
pub fn ed25519_pubkey_to_x25519(ed_pub_bytes: &[u8; 32]) -> Option<PublicKey> {
    use curve25519_dalek::edwards::CompressedEdwardsY;
    let compressed = CompressedEdwardsY(*ed_pub_bytes);
    let edwards_point = compressed.decompress()?;
    let montgomery = edwards_point.to_montgomery();
    Some(PublicKey::from(montgomery.to_bytes()))
}

/// Derive a symmetric cipher key from an ECDH shared secret using HKDF-SHA256.
/// Public keys are sorted to ensure both sides derive the same key regardless of
/// who initiates the session.
fn derive_cipher_key(shared_secret: &[u8], pub_a: &PublicKey, pub_b: &PublicKey) -> [u8; 32] {
    // Sort public keys lexicographically for deterministic salt
    let (first, second) = if pub_a.as_bytes() < pub_b.as_bytes() {
        (pub_a.as_bytes(), pub_b.as_bytes())
    } else {
        (pub_b.as_bytes(), pub_a.as_bytes())
    };
    let mut salt = Vec::with_capacity(64);
    salt.extend_from_slice(first);
    salt.extend_from_slice(second);

    let hk = Hkdf::<Sha256>::new(Some(&salt), shared_secret);
    let mut okm = [0u8; 32];
    hk.expand(b"swarmllm-session-v1", &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    fn make_session_pair() -> (SessionManager, SessionManager, NodeId, NodeId) {
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm_a = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let sm_b = SessionManager::from_ed25519_key(&id_b.signing_key_bytes());
        let node_a = id_a.node_id().clone();
        let node_b = id_b.node_id().clone();

        // Establish sessions in both directions
        sm_a.establish_session(&node_b, *sm_b.local_public_key());
        sm_b.establish_session(&node_a, *sm_a.local_public_key());

        (sm_a, sm_b, node_a, node_b)
    }

    #[test]
    fn seal_open_roundtrip() {
        let (sm_a, sm_b, node_a, node_b) = make_session_pair();
        let plaintext = b"hello world from node A";
        let aad = b"request-id-123";

        let sealed = sm_a.seal(&node_b, plaintext, aad).unwrap();
        let opened = sm_b.open(&node_a, &sealed, aad).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn wrong_peer_fails() {
        let (sm_a, _sm_b, node_a, node_b) = make_session_pair();
        let id_c = Identity::generate();
        let sm_c = SessionManager::from_ed25519_key(&id_c.signing_key_bytes());

        // A seals for B
        let sealed = sm_a.seal(&node_b, b"secret", b"aad").unwrap();

        // C tries to open — no session
        assert!(sm_c.open(&node_b, &sealed, b"aad").is_err());

        // Even if C has a session with A, the derived key is different
        sm_c.establish_session(&node_a, *sm_a.local_public_key());
        // C opening with node_a as peer should fail because the key is derived differently
        // (A sealed for B, not for C)
        let result = sm_c.open(&node_a, &sealed, b"aad");
        assert!(result.is_err());
    }

    #[test]
    fn nonce_monotonicity() {
        let (sm_a, _sm_b, _node_a, node_b) = make_session_pair();
        let aad = b"";

        // Seal multiple messages and verify nonces increment
        let sealed1 = sm_a.seal(&node_b, b"msg1", aad).unwrap();
        let sealed2 = sm_a.seal(&node_b, b"msg2", aad).unwrap();
        let sealed3 = sm_a.seal(&node_b, b"msg3", aad).unwrap();

        // Nonces are in first 12 bytes — counter is bytes [4..12]
        let nonce1 = u64::from_le_bytes(sealed1[4..12].try_into().unwrap());
        let nonce2 = u64::from_le_bytes(sealed2[4..12].try_into().unwrap());
        let nonce3 = u64::from_le_bytes(sealed3[4..12].try_into().unwrap());

        assert_eq!(nonce1, 0);
        assert_eq!(nonce2, 1);
        assert_eq!(nonce3, 2);
    }

    #[test]
    fn stale_eviction() {
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let node_b = id_b.node_id().clone();
        let sm_b = SessionManager::from_ed25519_key(&id_b.signing_key_bytes());

        sm.establish_session(&node_b, *sm_b.local_public_key());
        assert_eq!(sm.session_count(), 1);

        // Evicting with a large max_age should keep the session
        sm.evict_stale(std::time::Duration::from_secs(3600));
        assert_eq!(sm.session_count(), 1);

        // Evicting with zero max_age should remove it
        sm.evict_stale(std::time::Duration::ZERO);
        assert_eq!(sm.session_count(), 0);
    }

    #[test]
    fn ephemeral_seal_open_roundtrip() {
        let id = Identity::generate();
        let secret = ed25519_to_x25519_secret(&id.signing_key_bytes());
        let public = PublicKey::from(&secret);

        let plaintext = b"ephemeral secret message";
        let aad = b"req-456";

        let (eph_pub_bytes, ciphertext) = ephemeral_seal(&public, plaintext, aad).unwrap();
        let opened = ephemeral_open(&secret, &eph_pub_bytes, &ciphertext, aad).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn ed25519_to_x25519_deterministic() {
        let id = Identity::generate();
        let s1 = ed25519_to_x25519_secret(&id.signing_key_bytes());
        let s2 = ed25519_to_x25519_secret(&id.signing_key_bytes());
        let p1 = PublicKey::from(&s1);
        let p2 = PublicKey::from(&s2);
        assert_eq!(p1.as_bytes(), p2.as_bytes());
    }

    #[test]
    fn ed25519_pubkey_to_x25519_works() {
        let id = Identity::generate();
        let x25519_pub = ed25519_pubkey_to_x25519(&id.node_id().0);
        assert!(x25519_pub.is_some());
        // The derived public key should match what we get from the secret
        let secret = ed25519_to_x25519_secret(&id.signing_key_bytes());
        let expected = PublicKey::from(&secret);
        assert_eq!(x25519_pub.unwrap().as_bytes(), expected.as_bytes());
    }
}
