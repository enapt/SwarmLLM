//! At-rest encryption for provider API keys using the node's Ed25519 identity.
//!
//! Keys are encrypted with ChaCha20-Poly1305 using a symmetric key derived from
//! the node's Ed25519 signing key via HKDF-SHA256. A random nonce is generated
//! per encryption operation.
//!
//! Encrypted format: `$SWARM_ENC$` prefix + base64(nonce(12) || ciphertext+tag)

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;

use crate::config::{ProviderEntry, ProvidersConfig};
use crate::error::SwarmError;

/// Prefix that marks an encrypted API key in the database.
const ENC_PREFIX: &str = "$SWARM_ENC$";

/// Maximum allowed length for an API key.
const MAX_KEY_LENGTH: usize = 256;

/// Derive a 32-byte symmetric key from the node's Ed25519 signing key for provider key encryption.
fn derive_encryption_key(signing_key_bytes: &[u8; 32]) -> [u8; 32] {
    super::hkdf_sha256_derive_32(
        signing_key_bytes,
        None,
        b"swarmllm-provider-key-encryption-v1",
    )
}

/// Encrypt a plaintext API key string. Returns a string with the `$SWARM_ENC$` prefix.
fn encrypt_key(plaintext: &str, signing_key_bytes: &[u8; 32]) -> Result<String, SwarmError> {
    if plaintext.is_empty() {
        return Ok(String::new());
    }
    let sym_key = derive_encryption_key(signing_key_bytes);
    let cipher = ChaCha20Poly1305::new_from_slice(&sym_key)
        .map_err(|e| SwarmError::Internal(format!("cipher init: {e}")))?;

    let mut nonce_bytes = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|e| SwarmError::Internal(format!("encrypt: {e}")))?;

    // nonce || ciphertext+tag
    let mut blob = Vec::with_capacity(12 + ciphertext.len());
    blob.extend_from_slice(&nonce_bytes);
    blob.extend_from_slice(&ciphertext);

    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&blob);
    Ok(format!("{ENC_PREFIX}{encoded}"))
}

/// Decrypt an API key string. All stored keys must be encrypted.
fn decrypt_key(stored: &str, signing_key_bytes: &[u8; 32]) -> Result<String, SwarmError> {
    if stored.is_empty() {
        return Ok(String::new());
    }
    let encoded = match stored.strip_prefix(ENC_PREFIX) {
        Some(e) => e,
        None => {
            return Err(SwarmError::DecryptionFailed);
        }
    };

    use base64::Engine;
    let blob = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| SwarmError::DecryptionFailed)?;

    if blob.len() < 12 {
        return Err(SwarmError::DecryptionFailed);
    }

    let sym_key = derive_encryption_key(signing_key_bytes);
    let cipher = ChaCha20Poly1305::new_from_slice(&sym_key)
        .map_err(|e| SwarmError::Internal(format!("cipher init: {e}")))?;

    let nonce = Nonce::from_slice(&blob[..12]);
    let plaintext = cipher
        .decrypt(nonce, &blob[12..])
        .map_err(|_| SwarmError::DecryptionFailed)?;

    String::from_utf8(plaintext).map_err(|_| SwarmError::DecryptionFailed)
}

/// Apply a transformation to all provider entries in a ProvidersConfig.
/// Used by encrypt_config and decrypt_config to avoid duplicating the field list.
fn transform_config(
    config: &ProvidersConfig,
    entry_fn: fn(&mut Option<ProviderEntry>, &[u8; 32]) -> Result<(), SwarmError>,
    key_fn: fn(&str, &[u8; 32]) -> Result<String, SwarmError>,
    signing_key_bytes: &[u8; 32],
) -> Result<ProvidersConfig, SwarmError> {
    let mut out = config.clone();
    entry_fn(&mut out.anthropic, signing_key_bytes)?;
    entry_fn(&mut out.openai, signing_key_bytes)?;
    entry_fn(&mut out.deepseek, signing_key_bytes)?;
    entry_fn(&mut out.mistral, signing_key_bytes)?;
    entry_fn(&mut out.groq, signing_key_bytes)?;
    entry_fn(&mut out.nvidia_nim, signing_key_bytes)?;
    entry_fn(&mut out.cerebras, signing_key_bytes)?;
    entry_fn(&mut out.sambanova, signing_key_bytes)?;
    entry_fn(&mut out.fireworks, signing_key_bytes)?;
    entry_fn(&mut out.together, signing_key_bytes)?;
    entry_fn(&mut out.deepinfra, signing_key_bytes)?;
    entry_fn(&mut out.moonshot, signing_key_bytes)?;
    for custom in &mut out.custom {
        custom.api_key = key_fn(&custom.api_key, signing_key_bytes)?;
    }
    Ok(out)
}

/// Encrypt all API keys in a ProvidersConfig before persisting to database.
pub fn encrypt_config(
    config: &ProvidersConfig,
    signing_key_bytes: &[u8; 32],
) -> Result<ProvidersConfig, SwarmError> {
    transform_config(config, encrypt_entry, encrypt_key, signing_key_bytes)
}

/// Decrypt all API keys in a ProvidersConfig loaded from database.
pub fn decrypt_config(
    config: &ProvidersConfig,
    signing_key_bytes: &[u8; 32],
) -> Result<ProvidersConfig, SwarmError> {
    transform_config(config, decrypt_entry, decrypt_key, signing_key_bytes)
}

fn encrypt_entry(
    entry: &mut Option<ProviderEntry>,
    signing_key_bytes: &[u8; 32],
) -> Result<(), SwarmError> {
    if let Some(e) = entry {
        e.api_key = encrypt_key(&e.api_key, signing_key_bytes)?;
    }
    Ok(())
}

fn decrypt_entry(
    entry: &mut Option<ProviderEntry>,
    signing_key_bytes: &[u8; 32],
) -> Result<(), SwarmError> {
    if let Some(e) = entry {
        e.api_key = decrypt_key(&e.api_key, signing_key_bytes)?;
    }
    Ok(())
}

/// Validate an API key string. Returns an error message if invalid.
pub fn validate_api_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Ok(()); // empty means "remove"
    }
    if key.len() > MAX_KEY_LENGTH {
        return Err(format!(
            "API key too long ({} chars, max {})",
            key.len(),
            MAX_KEY_LENGTH
        ));
    }
    // Reject control characters, HTML/script injection, null bytes
    for (i, ch) in key.chars().enumerate() {
        if ch == '\0' {
            return Err(format!("API key contains null byte at position {i}"));
        }
        if ch == '\n' || ch == '\r' {
            return Err("API key contains newline characters".to_string());
        }
        if ch == '<' || ch == '>' {
            return Err("API key contains HTML characters (< or >)".to_string());
        }
        if ch == '{' || ch == '}' {
            return Err("API key contains brace characters".to_string());
        }
        if ch.is_control() {
            return Err(format!(
                "API key contains control character at position {i}"
            ));
        }
    }
    // Must be printable ASCII (API keys are always ASCII)
    if !key.is_ascii() {
        return Err("API key contains non-ASCII characters".to_string());
    }
    Ok(())
}

/// Scrub potential API key patterns from a string for safe logging.
/// Replaces known patterns with redacted versions.
pub fn scrub_api_keys(input: &str) -> String {
    // Patterns: sk-ant-*, sk-*, nvapi-*, gsk_*, csk-*, key-*, tok-*
    // Also catch generic long alphanumeric tokens (32+ chars)
    let mut output = input.to_string();

    // Named prefixes — redact everything after the prefix
    let prefixes = [
        "sk-ant-", "sk-", "nvapi-", "gsk_", "csk-", "key-", "tok-", "xai-", "fw_", "di_", "sn_",
    ];
    for prefix in prefixes {
        let mut search_start = 0usize;
        while search_start < output.len() {
            let Some(rel) = output[search_start..].find(prefix) else {
                break;
            };
            let start = search_start + rel;
            let rest = &output[start + prefix.len()..];
            let suffix_len = rest
                .find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                .unwrap_or(rest.len());
            let token_len = prefix.len() + suffix_len;
            let token_end = start + token_len;
            if token_len > prefix.len() + 4 {
                let redacted = format!("{}***REDACTED***", prefix);
                let redacted_len = redacted.len();
                output.replace_range(start..token_end, &redacted);
                search_start = start + redacted_len;
            } else {
                search_start = start + prefix.len();
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CustomProvider;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let signing_key = [42u8; 32];
        let original = "sk-ant-api03-test-key-12345";
        let encrypted = encrypt_key(original, &signing_key).unwrap();
        assert!(encrypted.starts_with(ENC_PREFIX));
        assert_ne!(encrypted, original);
        let decrypted = decrypt_key(&encrypted, &signing_key).unwrap();
        assert_eq!(decrypted, original);
    }

    #[test]
    fn decrypt_plaintext_rejected() {
        let signing_key = [42u8; 32];
        let plaintext = "sk-ant-legacy-key";
        // Unencrypted keys must be rejected — no plaintext migration fallback
        assert!(decrypt_key(plaintext, &signing_key).is_err());
    }

    #[test]
    fn encrypt_empty_key() {
        let signing_key = [42u8; 32];
        let encrypted = encrypt_key("", &signing_key).unwrap();
        assert_eq!(encrypted, "");
        let decrypted = decrypt_key("", &signing_key).unwrap();
        assert_eq!(decrypted, "");
    }

    #[test]
    fn wrong_key_fails_decrypt() {
        let key_a = [1u8; 32];
        let key_b = [2u8; 32];
        let encrypted = encrypt_key("secret", &key_a).unwrap();
        assert!(decrypt_key(&encrypted, &key_b).is_err());
    }

    #[test]
    fn config_encrypt_decrypt_roundtrip() {
        let signing_key = [99u8; 32];
        let config = ProvidersConfig {
            anthropic: Some(ProviderEntry {
                api_key: "sk-ant-test123".into(),
                default_model: None,
            }),
            openai: Some(ProviderEntry {
                api_key: "sk-openai-test".into(),
                default_model: Some("gpt-4o".into()),
            }),
            custom: vec![CustomProvider {
                name: "mycloud".into(),
                base_url: "https://api.example.com".into(),
                api_key: "tok-custom".into(),
                default_model: None,
            }],
            ..Default::default()
        };

        let encrypted = encrypt_config(&config, &signing_key).unwrap();
        // Encrypted keys should have the prefix
        assert!(encrypted
            .anthropic
            .as_ref()
            .unwrap()
            .api_key
            .starts_with(ENC_PREFIX));
        assert!(encrypted
            .openai
            .as_ref()
            .unwrap()
            .api_key
            .starts_with(ENC_PREFIX));
        assert!(encrypted.custom[0].api_key.starts_with(ENC_PREFIX));
        // default_model should be preserved
        assert_eq!(
            encrypted.openai.as_ref().unwrap().default_model,
            Some("gpt-4o".into())
        );

        let decrypted = decrypt_config(&encrypted, &signing_key).unwrap();
        assert_eq!(
            decrypted.anthropic.as_ref().unwrap().api_key,
            "sk-ant-test123"
        );
        assert_eq!(decrypted.openai.as_ref().unwrap().api_key, "sk-openai-test");
        assert_eq!(decrypted.custom[0].api_key, "tok-custom");
    }

    #[test]
    fn validate_good_keys() {
        assert!(validate_api_key("").is_ok());
        assert!(validate_api_key("sk-ant-api03-abc123").is_ok());
        assert!(validate_api_key("nvapi-12345-abcde").is_ok());
        assert!(validate_api_key("gsk_test_key_12345").is_ok());
    }

    #[test]
    fn validate_rejects_too_long() {
        let long_key = "a".repeat(257);
        assert!(validate_api_key(&long_key).is_err());
    }

    #[test]
    fn validate_rejects_control_chars() {
        assert!(validate_api_key("sk-\0test").is_err());
        assert!(validate_api_key("sk-test\n").is_err());
        assert!(validate_api_key("sk-test\r").is_err());
    }

    #[test]
    fn validate_rejects_html() {
        assert!(validate_api_key("<script>alert(1)</script>").is_err());
        assert!(validate_api_key("sk-test<img>").is_err());
        assert!(validate_api_key("sk-{test}").is_err());
    }

    #[test]
    fn validate_rejects_non_ascii() {
        assert!(validate_api_key("sk-tëst-üñîcödé").is_err());
    }

    #[test]
    fn scrub_known_patterns() {
        let input = "Error with key sk-ant-api03-abcdefg12345 failed";
        let scrubbed = scrub_api_keys(input);
        assert!(!scrubbed.contains("abcdefg12345"));
        assert!(scrubbed.contains("sk-ant-***REDACTED***"));
    }

    #[test]
    fn scrub_nvapi_pattern() {
        let input = "Using nvapi-abc123def456 for request";
        let scrubbed = scrub_api_keys(input);
        assert!(!scrubbed.contains("abc123def456"));
        assert!(scrubbed.contains("nvapi-***REDACTED***"));
    }

    #[test]
    fn scrub_gsk_pattern() {
        let input = "Key gsk_test_abcdefghijklm is invalid";
        let scrubbed = scrub_api_keys(input);
        assert!(!scrubbed.contains("abcdefghijklm"));
        assert!(scrubbed.contains("gsk_***REDACTED***"));
    }

    #[test]
    fn scrub_preserves_non_key_text() {
        let input = "Connection to api.openai.com succeeded";
        let scrubbed = scrub_api_keys(input);
        assert_eq!(scrubbed, input);
    }
}
