use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// SEC-I5: Tracks the highest received nonce to prevent replay attacks.
    last_seen_recv_nonce: AtomicU64,
    /// Whether nonce=0 has been seen (prevents replay of the first message).
    nonce_zero_seen: AtomicBool,
    created_at: Instant,
}

impl CachedSession {
    fn new(cipher_key: [u8; 32]) -> Self {
        Self {
            cipher_key,
            send_nonce: AtomicU64::new(0),
            last_seen_recv_nonce: AtomicU64::new(0),
            nonce_zero_seen: AtomicBool::new(false),
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
    /// Monotonic session epoch counter to prevent key reuse across re-established sessions.
    session_epoch: AtomicU64,
    /// Pending ephemeral secrets for in-progress key exchanges (initiator side).
    /// Removed and consumed when the peer responds with their ephemeral key.
    pending_ephemeral: DashMap<NodeId, EphemeralSecret>,
    /// Our ephemeral public keys for pending exchanges (used in key derivation).
    pending_ephemeral_pub: DashMap<NodeId, [u8; 32]>,
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
            session_epoch: AtomicU64::new(0),
            pending_ephemeral: DashMap::new(),
            pending_ephemeral_pub: DashMap::new(),
        }
    }

    /// Get this node's X25519 public key.
    pub fn local_public_key(&self) -> &PublicKey {
        &self.local_public
    }

    /// Establish a session with a peer given their X25519 public key.
    /// SEC-C5: Each session gets a unique epoch counter mixed into key derivation
    /// to prevent nonce reuse when sessions are re-established with the same peer.
    pub fn establish_session(&self, peer: &NodeId, peer_x25519_pub: PublicKey) {
        let epoch = self.session_epoch.fetch_add(1, Ordering::SeqCst);
        let shared_secret = self.local_secret.diffie_hellman(&peer_x25519_pub);
        let cipher_key = derive_cipher_key_with_epoch(
            shared_secret.as_bytes(),
            &self.local_public,
            &peer_x25519_pub,
            epoch,
        );
        self.sessions
            .insert(peer.clone(), CachedSession::new(cipher_key));
        tracing::debug!(peer = %peer, epoch, "Established encryption session");
    }

    /// Initiate an ephemeral ECDH key exchange for forward secrecy.
    ///
    /// Generates a fresh ephemeral X25519 keypair and returns the public key
    /// to be sent to the peer. The ephemeral secret is stored temporarily
    /// until the peer responds with their ephemeral public key.
    ///
    /// Call `complete_ephemeral_session` when the peer's response arrives.
    pub fn initiate_ephemeral_exchange(&self, peer: &NodeId) -> [u8; 32] {
        let ephemeral_secret = EphemeralSecret::random_from_rng(rand::rngs::OsRng);
        let ephemeral_public = PublicKey::from(&ephemeral_secret);
        let pub_bytes = *ephemeral_public.as_bytes();

        // Store the ephemeral secret and public key temporarily, keyed by peer
        self.pending_ephemeral_pub.insert(peer.clone(), pub_bytes);
        self.pending_ephemeral
            .insert(peer.clone(), ephemeral_secret);

        tracing::debug!(peer = %peer, "Initiated ephemeral ECDH exchange");
        pub_bytes
    }

    /// Complete an ephemeral ECDH exchange as the initiator.
    ///
    /// Called when the peer responds with their ephemeral public key.
    /// Derives the session key from the ephemeral DH and installs the session.
    /// The ephemeral secret is consumed (dropped/zeroized) after derivation.
    pub fn complete_ephemeral_session(
        &self,
        peer: &NodeId,
        peer_ephemeral_pub_bytes: &[u8; 32],
    ) -> bool {
        let ephemeral_secret = match self.pending_ephemeral.remove(peer) {
            Some((_, secret)) => secret,
            None => {
                tracing::warn!(peer = %peer, "No pending ephemeral exchange to complete");
                return false;
            }
        };

        let peer_ephemeral_pub = PublicKey::from(*peer_ephemeral_pub_bytes);
        let shared_secret = ephemeral_secret.diffie_hellman(&peer_ephemeral_pub);
        // ephemeral_secret is consumed by diffie_hellman — dropped here

        let epoch = self.session_epoch.fetch_add(1, Ordering::SeqCst);
        let our_ephemeral_pub = PublicKey::from(
            self.pending_ephemeral_pub
                .remove(peer)
                .map(|(_, b)| b)
                .unwrap_or(*self.local_public.as_bytes()),
        );
        let cipher_key = derive_cipher_key_with_epoch(
            shared_secret.as_bytes(),
            &our_ephemeral_pub,
            &peer_ephemeral_pub,
            epoch,
        );

        self.sessions
            .insert(peer.clone(), CachedSession::new(cipher_key));
        tracing::debug!(peer = %peer, epoch, "Established ephemeral forward-secret session");
        true
    }

    /// Handle an incoming ephemeral key exchange request (responder side).
    ///
    /// Generates a fresh ephemeral keypair, computes the shared secret with
    /// the initiator's ephemeral public key, installs the session, and
    /// returns our ephemeral public key for the response message.
    /// The ephemeral secret is consumed (dropped/zeroized) after derivation.
    pub fn accept_ephemeral_exchange(
        &self,
        peer: &NodeId,
        peer_ephemeral_pub_bytes: &[u8; 32],
    ) -> [u8; 32] {
        let our_ephemeral_secret = EphemeralSecret::random_from_rng(rand::rngs::OsRng);
        let our_ephemeral_public = PublicKey::from(&our_ephemeral_secret);
        let our_pub_bytes = *our_ephemeral_public.as_bytes();

        let peer_ephemeral_pub = PublicKey::from(*peer_ephemeral_pub_bytes);
        let shared_secret = our_ephemeral_secret.diffie_hellman(&peer_ephemeral_pub);
        // our_ephemeral_secret is consumed — dropped here

        let epoch = self.session_epoch.fetch_add(1, Ordering::SeqCst);
        let cipher_key = derive_cipher_key_with_epoch(
            shared_secret.as_bytes(),
            &our_ephemeral_public,
            &peer_ephemeral_pub,
            epoch,
        );

        self.sessions
            .insert(peer.clone(), CachedSession::new(cipher_key));
        tracing::debug!(peer = %peer, epoch, "Accepted ephemeral forward-secret session (responder)");

        our_pub_bytes
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
    /// SEC-I5: Rejects replayed messages by enforcing monotonic nonce ordering.
    pub fn open(&self, peer: &NodeId, sealed: &[u8], aad: &[u8]) -> Result<Vec<u8>, SwarmError> {
        if sealed.len() < 12 {
            return Err(SwarmError::DecryptionFailed);
        }
        let session = self
            .sessions
            .get(peer)
            .ok_or_else(|| SwarmError::NoSession(peer.clone()))?;

        // Extract and check the nonce counter for replay protection.
        // Atomic check-and-update: fetch_max atomically sets last_seen to
        // max(last_seen, recv_nonce) and returns the previous value. If the
        // previous value >= recv_nonce, this nonce was already accepted.
        // This eliminates the TOCTOU race of separate load + check + store.
        let mut nonce_counter_bytes = [0u8; 8];
        nonce_counter_bytes.copy_from_slice(&sealed[4..12]);
        let recv_nonce = u64::from_le_bytes(nonce_counter_bytes);

        // Handle nonce=0 specially (fetch_max(0) on a fresh 0 returns 0,
        // which looks like "already seen" even though it's the first message).
        if recv_nonce == 0 {
            if session.nonce_zero_seen.swap(true, Ordering::SeqCst) {
                tracing::warn!(peer = %peer, "Rejecting replayed nonce=0");
                return Err(SwarmError::DecryptionFailed);
            }
        } else {
            // For nonce > 0: atomically claim this nonce. If prev >= recv_nonce,
            // this nonce was already processed (replay or out-of-order).
            let prev = session
                .last_seen_recv_nonce
                .fetch_max(recv_nonce, Ordering::SeqCst);
            if prev >= recv_nonce {
                tracing::warn!(peer = %peer, recv_nonce, prev, "Rejecting replayed nonce");
                return Err(SwarmError::DecryptionFailed);
            }
        }

        // Decrypt after replay check passes
        let nonce = Nonce::from_slice(&sealed[..12]);
        let ciphertext = &sealed[12..];

        let cipher = ChaCha20Poly1305::new_from_slice(&session.cipher_key)
            .map_err(|e| SwarmError::Encryption(format!("Cipher init failed: {e}")))?;

        let payload = chacha20poly1305::aead::Payload {
            msg: ciphertext,
            aad,
        };
        let plaintext = cipher
            .decrypt(nonce, payload)
            .map_err(|_| SwarmError::DecryptionFailed)?;

        Ok(plaintext)
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

    /// Get the NodeIds of all peers with active sessions (for key rotation).
    pub fn active_peers(&self) -> Vec<NodeId> {
        self.sessions.iter().map(|e| e.key().clone()).collect()
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

/// SEC-C5: Derive a cipher key with session epoch mixed in to prevent key reuse.
fn derive_cipher_key_with_epoch(
    shared_secret: &[u8],
    pub_a: &PublicKey,
    pub_b: &PublicKey,
    epoch: u64,
) -> [u8; 32] {
    let (first, second) = if pub_a.as_bytes() < pub_b.as_bytes() {
        (pub_a.as_bytes(), pub_b.as_bytes())
    } else {
        (pub_b.as_bytes(), pub_a.as_bytes())
    };
    let mut salt = Vec::with_capacity(72);
    salt.extend_from_slice(first);
    salt.extend_from_slice(second);
    salt.extend_from_slice(&epoch.to_le_bytes());

    let hk = Hkdf::<Sha256>::new(Some(&salt), shared_secret);
    let mut okm = [0u8; 32];
    hk.expand(b"swarmllm-session-v1", &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
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

    // ---- Ephemeral forward secrecy tests ----

    #[test]
    fn ephemeral_exchange_seal_open_roundtrip() {
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm_a = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let sm_b = SessionManager::from_ed25519_key(&id_b.signing_key_bytes());
        let node_a = id_a.node_id().clone();
        let node_b = id_b.node_id().clone();

        // A initiates ephemeral exchange
        let a_eph_pub = sm_a.initiate_ephemeral_exchange(&node_b);

        // B accepts (responder side): gets A's ephemeral pub, generates its own
        let b_eph_pub = sm_b.accept_ephemeral_exchange(&node_a, &a_eph_pub);

        // A completes: uses B's ephemeral pub response
        assert!(sm_a.complete_ephemeral_session(&node_b, &b_eph_pub));

        // Now both should have forward-secret sessions
        let plaintext = b"forward secret message";
        let aad = b"test-aad";

        let sealed = sm_a.seal(&node_b, plaintext, aad).unwrap();
        let opened = sm_b.open(&node_a, &sealed, aad).unwrap();
        assert_eq!(opened, plaintext);

        // And reverse direction
        let sealed_b = sm_b.seal(&node_a, b"reply", aad).unwrap();
        let opened_b = sm_a.open(&node_b, &sealed_b, aad).unwrap();
        assert_eq!(opened_b, b"reply");
    }

    #[test]
    fn different_sessions_get_different_keys() {
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm_a = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let sm_b = SessionManager::from_ed25519_key(&id_b.signing_key_bytes());
        let node_a = id_a.node_id().clone();
        let node_b = id_b.node_id().clone();

        // First ephemeral session
        let a_eph1 = sm_a.initiate_ephemeral_exchange(&node_b);
        let b_eph1 = sm_b.accept_ephemeral_exchange(&node_a, &a_eph1);
        sm_a.complete_ephemeral_session(&node_b, &b_eph1);

        let sealed1 = sm_a.seal(&node_b, b"msg1", b"").unwrap();

        // Second ephemeral session (re-key)
        let a_eph2 = sm_a.initiate_ephemeral_exchange(&node_b);
        let b_eph2 = sm_b.accept_ephemeral_exchange(&node_a, &a_eph2);
        sm_a.complete_ephemeral_session(&node_b, &b_eph2);

        let sealed2 = sm_a.seal(&node_b, b"msg1", b"").unwrap();

        // Ephemeral keys should be different each time
        assert_ne!(a_eph1, a_eph2);
        assert_ne!(b_eph1, b_eph2);

        // Even with the same plaintext, sealed data should differ
        // (different keys + different nonces = different ciphertext)
        assert_ne!(sealed1, sealed2);
    }

    #[test]
    fn complete_without_initiation_fails() {
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm_a = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let node_b = id_b.node_id().clone();

        // Try to complete without initiating
        let fake_pub = [42u8; 32];
        assert!(!sm_a.complete_ephemeral_session(&node_b, &fake_pub));
    }

    #[test]
    fn static_session_still_works_as_fallback() {
        // Verify the existing static-key session path still works
        // (backward compatibility when peers don't support ephemeral)
        let (sm_a, sm_b, node_a, node_b) = make_session_pair();

        let plaintext = b"static key message";
        let aad = b"aad";

        let sealed = sm_a.seal(&node_b, plaintext, aad).unwrap();
        let opened = sm_b.open(&node_a, &sealed, aad).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn ephemeral_sessions_independent_of_static_keys() {
        // Two nodes with identical static keys should get different
        // ephemeral session keys
        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let sm_a = SessionManager::from_ed25519_key(&id_a.signing_key_bytes());
        let sm_b = SessionManager::from_ed25519_key(&id_b.signing_key_bytes());
        let node_a = id_a.node_id().clone();
        let node_b = id_b.node_id().clone();

        // Ephemeral exchange
        let a_eph = sm_a.initiate_ephemeral_exchange(&node_b);
        let b_eph = sm_b.accept_ephemeral_exchange(&node_a, &a_eph);
        sm_a.complete_ephemeral_session(&node_b, &b_eph);

        // The ephemeral public keys should not equal the static public keys
        assert_ne!(&a_eph, sm_a.local_public_key().as_bytes());
        assert_ne!(&b_eph, sm_b.local_public_key().as_bytes());
    }
}
