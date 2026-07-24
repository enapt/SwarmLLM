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

            tracing::debug!(node_id = %node_id, "DIAG: identity key loaded from disk");
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
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&key_path)
                    .map_err(SwarmError::Io)?;
                file.write_all(&identity.signing_key.to_bytes())
                    .map_err(SwarmError::Io)?;
                // fsync to disk before declaring the identity persisted. A
                // kernel panic / power loss between write_all returning and
                // the page cache flushing can leave a 0-byte file on disk;
                // load_or_generate then errors with "Invalid key file size:
                // 0 bytes" and the node permanently can't start under its
                // existing identity. Cheap insurance for a 32-byte write.
                file.sync_all().map_err(SwarmError::Io)?;
            }
            #[cfg(not(unix))]
            {
                std::fs::write(&key_path, identity.signing_key.to_bytes())
                    .map_err(SwarmError::Io)?;
                // SEC-I6: On Windows, restrict key file permissions to current user only.
                // Uses icacls to remove inherited permissions and grant only the current user.
                #[cfg(target_os = "windows")]
                {
                    if let Ok(username) = std::env::var("USERNAME") {
                        // SEC: Validate USERNAME to prevent argument injection via crafted env var
                        let is_safe = !username.is_empty()
                            && username.len() <= 256
                            && username.chars().all(|c| {
                                c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ' '
                            });
                        if is_safe {
                            let path_str = key_path.display().to_string();
                            let _ = std::process::Command::new("icacls")
                                .args([
                                    &path_str,
                                    "/inheritance:r",
                                    "/grant:r",
                                    &format!("{username}:F"),
                                ])
                                .output();
                        }
                    }
                }
            }
            tracing::info!(node_id = %identity.node_id, "Generated new identity");
            Ok(identity)
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
    ///
    /// SEC: callers that hold the returned array beyond a single immediate
    /// consume MUST wrap in `zeroize::Zeroizing::new(...)` so the heap/stack
    /// copy is scrubbed on drop. `SigningKey` zeroizes itself, but once
    /// bytes leave via `to_bytes()` that guarantee doesn't propagate. The
    /// public signature stays `[u8; 32]` because most consumers pass the
    /// array straight into libp2p / candle APIs that take ownership; forcing
    /// `Zeroizing` here just produces an unzeroized copy on the way in.
    pub(crate) fn signing_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// This node's X25519 static secret, derived from its Ed25519 signing key
    /// (RFC 7748). The single source of truth for opening anything sealed to
    /// this node's key — pipeline prompts and relay envelopes both need it.
    /// Returning the secret is no greater exposure than holding the `Identity`
    /// itself (which already holds the root signing key).
    pub fn x25519_secret(&self) -> x25519_dalek::StaticSecret {
        crate::crypto::session::ed25519_to_x25519_secret(&self.signing_key_bytes())
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
