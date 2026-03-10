use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;
use x25519_dalek::PublicKey;

use crate::error::SwarmError;
// Re-export SealedPrompt from swarmllm-types crate
pub use crate::types::SealedPrompt;

/// Seal a prompt for a specific pipeline node.
/// The prompt is encrypted with a random symmetric key, and that key is
/// ephemeral-sealed for the target node's X25519 public key.
pub fn seal_prompt(
    request_id: uuid::Uuid,
    prompt: &[u8],
    target_x25519_pub: &PublicKey,
) -> Result<SealedPrompt, SwarmError> {
    // Generate random request key
    let mut request_key = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut request_key);

    // Generate random nonce
    let mut nonce_bytes = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);

    // Encrypt prompt with request key
    let cipher = ChaCha20Poly1305::new_from_slice(&request_key)
        .map_err(|e| SwarmError::Encryption(format!("Cipher init: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let aad = request_id.as_bytes().as_slice();
    let payload = chacha20poly1305::aead::Payload { msg: prompt, aad };
    let encrypted_prompt = cipher
        .encrypt(nonce, payload)
        .map_err(|e| SwarmError::Encryption(format!("Prompt seal: {e}")))?;

    // Wrap request key for the target node
    let (ephemeral_pub, key_envelope) =
        crate::crypto::session::ephemeral_seal(target_x25519_pub, &request_key, aad)?;

    tracing::debug!(
        request_id = %request_id,
        prompt_len = prompt.len(),
        "DIAG: pipeline prompt sealed"
    );
    Ok(SealedPrompt {
        request_id,
        encrypted_prompt,
        nonce: nonce_bytes,
        ephemeral_pub,
        key_envelope,
    })
}

/// Open a sealed prompt using this node's X25519 static secret.
pub fn open_prompt(
    sealed: &SealedPrompt,
    local_secret: &x25519_dalek::StaticSecret,
) -> Result<Vec<u8>, SwarmError> {
    let aad = sealed.request_id.as_bytes().as_slice();

    // Unwrap the request key
    let request_key_bytes = crate::crypto::session::ephemeral_open(
        local_secret,
        &sealed.ephemeral_pub,
        &sealed.key_envelope,
        aad,
    )?;

    if request_key_bytes.len() != 32 {
        return Err(SwarmError::DecryptionFailed);
    }

    // Decrypt the prompt
    let cipher = ChaCha20Poly1305::new_from_slice(&request_key_bytes)
        .map_err(|e| SwarmError::Encryption(format!("Cipher init: {e}")))?;
    let nonce = Nonce::from_slice(&sealed.nonce);
    let payload = chacha20poly1305::aead::Payload {
        msg: sealed.encrypted_prompt.as_slice(),
        aad,
    };
    cipher
        .decrypt(nonce, payload)
        .map_err(|_| SwarmError::DecryptionFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::session::ed25519_to_x25519_secret;
    use crate::identity::Identity;

    #[test]
    fn seal_open_prompt_roundtrip() {
        let id = Identity::generate();
        let secret = ed25519_to_x25519_secret(&id.signing_key_bytes());
        let public = x25519_dalek::PublicKey::from(&secret);

        let request_id = uuid::Uuid::new_v4();
        let prompt = b"What is the meaning of life?";

        let sealed = seal_prompt(request_id, prompt, &public).unwrap();
        assert_eq!(sealed.request_id, request_id);
        assert_ne!(sealed.encrypted_prompt, prompt.as_slice());

        let opened = open_prompt(&sealed, &secret).unwrap();
        assert_eq!(opened, prompt.as_slice());
    }
}
