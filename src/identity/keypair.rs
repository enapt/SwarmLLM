use std::path::Path;

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;

use crate::error::SwarmError;
use crate::types::NodeId;

/// Node identity wrapping an Ed25519 keypair.
#[derive(Clone)]
pub struct Identity {
    signing_key: SigningKey,
    node_id: NodeId,
}

impl Identity {
    /// Generate a new random identity.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = NodeId(signing_key.verifying_key().to_bytes());
        Self {
            signing_key,
            node_id,
        }
    }

    /// Load identity from disk, or generate and save if it doesn't exist.
    pub fn load_or_generate(data_dir: &Path) -> Result<Self, SwarmError> {
        let key_path = data_dir.join("identity.key");

        if key_path.exists() {
            let bytes = std::fs::read(&key_path).map_err(SwarmError::Io)?;
            if bytes.len() != 32 {
                return Err(SwarmError::Keystore(format!(
                    "Invalid key file size: {} bytes (expected 32)",
                    bytes.len()
                )));
            }
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(&bytes);
            let signing_key = SigningKey::from_bytes(&key_bytes);
            let node_id = NodeId(signing_key.verifying_key().to_bytes());

            tracing::info!(node_id = %node_id, "Loaded identity");
            Ok(Self {
                signing_key,
                node_id,
            })
        } else {
            let identity = Self::generate();

            // Ensure parent directory exists
            if let Some(parent) = key_path.parent() {
                std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
            }

            // SECURITY: Write key file with restricted permissions (owner-only)
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&key_path)
                    .map_err(SwarmError::Io)?;
                file.write_all(&identity.signing_key.to_bytes()).map_err(SwarmError::Io)?;
            }
            #[cfg(not(unix))]
            {
                std::fs::write(&key_path, identity.signing_key.to_bytes()).map_err(SwarmError::Io)?;
            }
            tracing::info!(node_id = %identity.node_id, "Generated new identity");
            Ok(identity)
        }
    }

    /// Create identity from an existing signing key.
    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        let node_id = NodeId(signing_key.verifying_key().to_bytes());
        Self {
            signing_key,
            node_id,
        }
    }

    /// Sign a message with this node's private key.
    pub fn sign(&self, msg: &[u8]) -> Vec<u8> {
        self.signing_key.sign(msg).to_bytes().to_vec()
    }

    /// Get this node's public identity.
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Get the verifying (public) key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Get the raw signing key bytes (for libp2p keypair conversion).
    pub(crate) fn signing_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Derive an X25519 static secret from this identity's Ed25519 signing key.
    pub fn x25519_static_secret(&self) -> x25519_dalek::StaticSecret {
        crate::crypto::session::ed25519_to_x25519_secret(&self.signing_key.to_bytes())
    }

    /// Derive the X25519 public key for this identity.
    pub fn x25519_public_key(&self) -> x25519_dalek::PublicKey {
        x25519_dalek::PublicKey::from(&self.x25519_static_secret())
    }
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Identity")
            .field("node_id", &self.node_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Verifier;

    #[test]
    fn generate_produces_valid_identity() {
        let id = Identity::generate();
        assert_ne!(id.node_id().0, [0u8; 32]);
    }

    #[test]
    fn sign_and_verify() {
        let id = Identity::generate();
        let msg = b"hello world";
        let sig_bytes = id.sign(msg);
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes.as_slice().try_into().unwrap());
        assert!(id.verifying_key().verify(msg, &sig).is_ok());
    }

    #[test]
    fn load_or_generate_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let id1 = Identity::load_or_generate(dir.path()).unwrap();
        let id2 = Identity::load_or_generate(dir.path()).unwrap();
        assert_eq!(id1.node_id(), id2.node_id());
    }

    #[test]
    fn node_id_from_verifying_key() {
        let id = Identity::generate();
        assert_eq!(id.node_id().0, id.verifying_key().to_bytes());
    }
}
