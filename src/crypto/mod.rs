pub mod gossip_seal;
pub mod key_rotation;
pub mod pipeline_seal;
pub mod provider_keys;
pub mod session;

use crate::error::SwarmError;
use ed25519_dalek::Verifier;

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
