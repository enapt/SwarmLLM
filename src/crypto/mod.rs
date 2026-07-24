pub mod gossip_seal;
pub mod key_rotation;
pub mod pipeline_seal;
pub mod provider_keys;
pub mod relay_seal;
pub mod session;

use crate::error::SwarmError;
use ed25519_dalek::Verifier;
use hkdf::Hkdf;
use sha2::Sha256;

/// Derive a 32-byte symmetric key via HKDF-SHA256. Used by every subsystem
/// that needs a symmetric key from a higher-entropy secret (provider key
/// encryption, gossip epoch keys, session ChaCha keys). Pre-existing call
/// sites duplicated the 5-line `Hkdf::new + expand` pattern with the same
/// `.expect("32 bytes is a valid HKDF-SHA256 output length")` string —
/// consolidated here.
///
/// `salt = None` is HKDF's spec-correct null-salt path (semantically
/// equivalent to a zero-byte salt of the hash output length).
pub fn hkdf_sha256_derive_32(ikm: &[u8], salt: Option<&[u8]>, info: &[u8]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(salt, ikm);
    let mut okm = [0u8; 32];
    hk.expand(info, &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Verify an Ed25519 signature from raw bytes.
/// Reusable primitive for all subsystems that verify signatures.
pub fn verify_ed25519_sig(
    sig_bytes: &[u8],
    payload: &[u8],
    key: &ed25519_dalek::VerifyingKey,
) -> Result<(), SwarmError> {
    if sig_bytes.len() != 64 {
        return Err(SwarmError::InvalidSignature);
    }
    let sig = ed25519_dalek::Signature::from_bytes(
        sig_bytes
            .try_into()
            .map_err(|_| SwarmError::InvalidSignature)?,
    );
    key.verify(payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)
}

pub use gossip_seal::GossipSealer;
pub use pipeline_seal::{open_prompt, seal_prompt, SealedPrompt};
pub use provider_keys::{decrypt_config, encrypt_config, scrub_api_keys, validate_api_key};
pub use session::SessionManager;
