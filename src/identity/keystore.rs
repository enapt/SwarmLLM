use std::path::Path;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use argon2::Argon2;
use ed25519_dalek::SigningKey;
use rand::RngCore;

use crate::error::SwarmError;

/// Binary format version for encrypted keystore files.
const KEYSTORE_VERSION: u8 = 1;

/// Encrypted keystore for Ed25519 signing keys.
/// Format: [version(1B)][salt(16B)][nonce(12B)][ciphertext(32B)][tag(16B)] = 77 bytes
pub struct Keystore;

impl Keystore {
    /// Save a signing key to disk, optionally encrypted with a passphrase.
    pub fn save(key: &SigningKey, passphrase: Option<&str>, path: &Path) -> Result<(), SwarmError> {
        match passphrase {
            Some(pass) => Self::save_encrypted(key, pass, path),
            None => {
                // Store raw 32-byte key
                std::fs::write(path, key.to_bytes()).map_err(SwarmError::Io)
            }
        }
    }

    /// Load a signing key from disk, decrypting if necessary.
    pub fn load(path: &Path, passphrase: Option<&str>) -> Result<SigningKey, SwarmError> {
        let data = std::fs::read(path).map_err(SwarmError::Io)?;

        if data.len() == 32 {
            // Unencrypted raw key
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(&data);
            Ok(SigningKey::from_bytes(&bytes))
        } else if data.len() == 77 && data[0] == KEYSTORE_VERSION {
            // Encrypted keystore
            let pass = passphrase.ok_or(SwarmError::Keystore(
                "Encrypted keystore requires a passphrase".into(),
            ))?;
            Self::load_encrypted(&data, pass)
        } else {
            Err(SwarmError::Keystore(format!(
                "Invalid keystore format: {} bytes",
                data.len()
            )))
        }
    }

    fn save_encrypted(key: &SigningKey, passphrase: &str, path: &Path) -> Result<(), SwarmError> {
        let mut rng = rand::thread_rng();

        // Generate salt and nonce
        let mut salt = [0u8; 16];
        rng.fill_bytes(&mut salt);
        let mut nonce_bytes = [0u8; 12];
        rng.fill_bytes(&mut nonce_bytes);

        // Derive encryption key with Argon2id
        let mut derived_key = [0u8; 32];
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(65536, 3, 4, Some(32))
                .map_err(|e| SwarmError::Keystore(e.to_string()))?,
        );
        argon2
            .hash_password_into(passphrase.as_bytes(), &salt, &mut derived_key)
            .map_err(|e| SwarmError::Keystore(e.to_string()))?;

        // Encrypt the signing key bytes
        let cipher = Aes256Gcm::new_from_slice(&derived_key)
            .map_err(|e| SwarmError::Keystore(e.to_string()))?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher
            .encrypt(nonce, key.to_bytes().as_slice())
            .map_err(|e| SwarmError::Keystore(e.to_string()))?;

        // Assemble: version + salt + nonce + ciphertext(includes tag)
        let mut output = Vec::with_capacity(77);
        output.push(KEYSTORE_VERSION);
        output.extend_from_slice(&salt);
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);

        std::fs::write(path, &output).map_err(SwarmError::Io)
    }

    fn load_encrypted(data: &[u8], passphrase: &str) -> Result<SigningKey, SwarmError> {
        // Parse fields from the binary format
        let salt = &data[1..17];
        let nonce_bytes = &data[17..29];
        let ciphertext = &data[29..];

        // Derive key
        let mut derived_key = [0u8; 32];
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(65536, 3, 4, Some(32))
                .map_err(|e| SwarmError::Keystore(e.to_string()))?,
        );
        argon2
            .hash_password_into(passphrase.as_bytes(), salt, &mut derived_key)
            .map_err(|e| SwarmError::Keystore(e.to_string()))?;

        // Decrypt
        let cipher = Aes256Gcm::new_from_slice(&derived_key)
            .map_err(|e| SwarmError::Keystore(e.to_string()))?;
        let nonce = Nonce::from_slice(nonce_bytes);
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| SwarmError::WrongPassphrase)?;

        if plaintext.len() != 32 {
            return Err(SwarmError::Keystore(format!(
                "Decrypted key has wrong size: {}",
                plaintext.len()
            )));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&plaintext);
        Ok(SigningKey::from_bytes(&key_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_unencrypted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.key");
        let key = SigningKey::generate(&mut rand::rngs::OsRng);

        Keystore::save(&key, None, &path).unwrap();
        let loaded = Keystore::load(&path, None).unwrap();
        assert_eq!(key.to_bytes(), loaded.to_bytes());
    }

    #[test]
    fn save_and_load_encrypted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.key");
        let key = SigningKey::generate(&mut rand::rngs::OsRng);

        Keystore::save(&key, Some("mypassword"), &path).unwrap();

        // Verify file size
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 77);
        assert_eq!(data[0], KEYSTORE_VERSION);

        let loaded = Keystore::load(&path, Some("mypassword")).unwrap();
        assert_eq!(key.to_bytes(), loaded.to_bytes());
    }

    #[test]
    fn wrong_passphrase_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.key");
        let key = SigningKey::generate(&mut rand::rngs::OsRng);

        Keystore::save(&key, Some("correct"), &path).unwrap();
        let result = Keystore::load(&path, Some("wrong"));
        assert!(result.is_err());
    }

    #[test]
    fn encrypted_needs_passphrase() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.key");
        let key = SigningKey::generate(&mut rand::rngs::OsRng);

        Keystore::save(&key, Some("pass"), &path).unwrap();
        let result = Keystore::load(&path, None);
        assert!(result.is_err());
    }
}
