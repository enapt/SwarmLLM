use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;

use crate::error::SwarmError;
use crate::identity::Identity;

/// Seals and opens GossipSub messages with an epoch-based group key.
/// The group key rotates every hour (epoch = unix_timestamp / 3600).
pub struct GossipSealer {
    network_id: Vec<u8>,
}

impl GossipSealer {
    pub fn new(network_id: &[u8]) -> Self {
        Self {
            network_id: network_id.to_vec(),
        }
    }

    /// Derive the group key for a given epoch.
    fn derive_epoch_key(&self, epoch: u32) -> [u8; 32] {
        let mut ikm = self.network_id.clone();
        ikm.extend_from_slice(&epoch.to_le_bytes());
        let hk = Hkdf::<Sha256>::new(None, &ikm);
        let mut okm = [0u8; 32];
        hk.expand(b"swarmllm-gossip-v1", &mut okm)
            .expect("32 bytes is valid HKDF output length");
        okm
    }

    /// Get the current epoch (unix timestamp / 3600).
    fn current_epoch() -> u32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (now / 3600) as u32
    }

    /// Seal a gossip message (without Ed25519 signature — use `seal_signed` for authenticated gossip).
    /// Output format: `[4B epoch_tag][12B nonce][ciphertext+tag]`
    pub fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>, SwarmError> {
        let epoch = Self::current_epoch();
        let key = self.derive_epoch_key(epoch);

        let cipher = ChaCha20Poly1305::new_from_slice(&key)
            .map_err(|e| SwarmError::Encryption(format!("Gossip cipher init: {e}")))?;

        let mut nonce_bytes = [0u8; 12];
        rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| SwarmError::Encryption(format!("Gossip seal: {e}")))?;

        let mut out = Vec::with_capacity(4 + 12 + ciphertext.len());
        out.extend_from_slice(&epoch.to_le_bytes());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Open a sealed gossip message.
    /// Tries the epoch from the message tag, then the previous epoch for clock skew tolerance.
    pub fn open(&self, sealed: &[u8]) -> Result<Vec<u8>, SwarmError> {
        if sealed.len() < 16 {
            return Err(SwarmError::DecryptionFailed);
        }

        let epoch_tag = u32::from_le_bytes(
            sealed[..4]
                .try_into()
                .map_err(|_| SwarmError::DecryptionFailed)?,
        );
        let nonce = Nonce::from_slice(&sealed[4..16]);
        let ciphertext = &sealed[16..];

        // Try the tagged epoch first
        let key = self.derive_epoch_key(epoch_tag);
        let cipher = ChaCha20Poly1305::new_from_slice(&key)
            .map_err(|e| SwarmError::Encryption(format!("Gossip cipher init: {e}")))?;

        if let Ok(plaintext) = cipher.decrypt(nonce, ciphertext) {
            return Ok(plaintext);
        }

        // Try adjacent epochs for clock skew tolerance
        for delta in [1u32, u32::MAX] {
            // u32::MAX wraps to epoch_tag - 1
            let alt_epoch = epoch_tag.wrapping_add(delta);
            let alt_key = self.derive_epoch_key(alt_epoch);
            let alt_cipher = ChaCha20Poly1305::new_from_slice(&alt_key)
                .map_err(|e| SwarmError::Encryption(format!("Gossip cipher init: {e}")))?;
            if let Ok(plaintext) = alt_cipher.decrypt(nonce, ciphertext) {
                return Ok(plaintext);
            }
        }

        Err(SwarmError::DecryptionFailed)
    }

    /// SEC-C6: Seal a gossip message with Ed25519 signature from the originating node.
    /// Output format: `[32B sender_pubkey][64B ed25519_signature][4B epoch_tag][12B nonce][ciphertext+tag]`
    pub fn seal_signed(
        &self,
        plaintext: &[u8],
        identity: &Identity,
    ) -> Result<Vec<u8>, SwarmError> {
        let inner_sealed = self.seal(plaintext)?;

        // Sign the sealed payload (epoch+nonce+ciphertext)
        let signature = identity.sign(&inner_sealed);
        let pubkey = identity.node_id().0;

        let mut out = Vec::with_capacity(32 + 64 + inner_sealed.len());
        out.extend_from_slice(&pubkey);
        out.extend_from_slice(&signature);
        out.extend_from_slice(&inner_sealed);
        Ok(out)
    }

    /// SEC-C6: Open a signed gossip message, verifying the Ed25519 signature.
    /// Returns `(sender_node_id_bytes, plaintext)`.
    pub fn open_signed(&self, sealed: &[u8]) -> Result<([u8; 32], Vec<u8>), SwarmError> {
        if sealed.len() < 32 + 64 + 16 {
            return Err(SwarmError::DecryptionFailed);
        }

        let sender_pub_bytes: [u8; 32] = sealed[..32]
            .try_into()
            .map_err(|_| SwarmError::DecryptionFailed)?;
        let sig_bytes: [u8; 64] = sealed[32..96]
            .try_into()
            .map_err(|_| SwarmError::DecryptionFailed)?;
        let inner_sealed = &sealed[96..];

        // Verify Ed25519 signature
        let vk = VerifyingKey::from_bytes(&sender_pub_bytes)
            .map_err(|_| SwarmError::InvalidSignature)?;
        let sig = Signature::from_bytes(&sig_bytes);
        vk.verify(inner_sealed, &sig)
            .map_err(|_| SwarmError::InvalidSignature)?;

        // Decrypt the inner payload
        let plaintext = self.open(inner_sealed)?;
        Ok((sender_pub_bytes, plaintext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_open_roundtrip() {
        let sealer = GossipSealer::new(b"test-network-id");
        let plaintext = b"hello gossip world";

        let sealed = sealer.seal(plaintext).unwrap();
        let opened = sealer.open(&sealed).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn deterministic_key_derivation() {
        let s1 = GossipSealer::new(b"net-a");
        let s2 = GossipSealer::new(b"net-a");
        assert_eq!(s1.derive_epoch_key(100), s2.derive_epoch_key(100));
    }

    #[test]
    fn different_network_different_key() {
        let s1 = GossipSealer::new(b"net-a");
        let s2 = GossipSealer::new(b"net-b");
        assert_ne!(s1.derive_epoch_key(100), s2.derive_epoch_key(100));
    }

    #[test]
    fn different_epoch_different_key() {
        let sealer = GossipSealer::new(b"test");
        assert_ne!(sealer.derive_epoch_key(1), sealer.derive_epoch_key(2));
    }

    #[test]
    fn open_too_short_fails() {
        let sealer = GossipSealer::new(b"test");
        assert!(sealer.open(&[0u8; 10]).is_err());
    }
}
