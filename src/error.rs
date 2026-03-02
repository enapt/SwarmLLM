use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use thiserror::Error;

use crate::types::{ModelId, NodeId, ShardId};

#[derive(Error, Debug)]
pub enum SwarmError {
    // Network
    #[error("Network error: {0}")]
    Network(String),
    #[error("Peer not found: {0}")]
    PeerNotFound(NodeId),
    #[error("Connection failed to {peer}: {reason}")]
    ConnectionFailed { peer: NodeId, reason: String },

    // Inference
    #[error("Model not available: {0}")]
    ModelNotAvailable(ModelId),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Insufficient network capacity for model {0}")]
    InsufficientCapacity(ModelId),
    #[error("Pipeline assembly failed: {0}")]
    PipelineError(String),
    #[error("Inference timeout after {0}s")]
    InferenceTimeout(u64),
    #[error("No model loaded")]
    NoModelLoaded,

    // Shards
    #[error("Shard verification failed: expected {expected}, got {actual}")]
    ShardIntegrity { expected: String, actual: String },
    #[error("Shard not found: {0:?}")]
    ShardNotFound(ShardId),

    // Credits
    #[error("Insufficient credits: balance={balance}, required={required}")]
    InsufficientCredits { balance: i64, required: i64 },
    #[error("Invalid transaction signature")]
    InvalidSignature,
    #[error("Credit error: {0}")]
    CreditError(String),

    // Encryption
    #[error("Encryption error: {0}")]
    Encryption(String),
    #[error("Decryption failed (invalid key, corrupted data, or tampered ciphertext)")]
    DecryptionFailed,
    #[error("No encryption session for peer: {0}")]
    NoSession(NodeId),
    #[error("Nonce counter overflow — session must be re-established")]
    NonceOverflow,

    // Identity
    #[error("Keystore error: {0}")]
    Keystore(String),
    #[error("Wrong passphrase")]
    WrongPassphrase,
    #[error("Invalid nickname: {0}")]
    InvalidNickname(String),

    // Storage
    #[error("Database error: {0}")]
    Database(String),
    #[error("Insufficient disk space: need {need_mb}MB, have {have_mb}MB")]
    InsufficientDisk { need_mb: u64, have_mb: u64 },

    // Auth
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    // Config
    #[error("Configuration error: {0}")]
    Config(String),

    // Generic
    #[error("Internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

/// API-facing error that maps SwarmError to HTTP status codes.
pub struct ApiError(pub SwarmError);

impl From<SwarmError> for ApiError {
    fn from(err: SwarmError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message, error_type) = match &self.0 {
            SwarmError::ModelNotAvailable(_) => {
                (StatusCode::NOT_FOUND, self.0.to_string(), "not_found_error")
            }
            SwarmError::NoModelLoaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                self.0.to_string(),
                "not_found_error",
            ),
            SwarmError::InferenceTimeout(_) => (
                StatusCode::GATEWAY_TIMEOUT,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::InsufficientCredits { .. } => (
                StatusCode::PAYMENT_REQUIRED,
                self.0.to_string(),
                "rate_limit_error",
            ),
            SwarmError::InsufficientCapacity(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::InsufficientDisk { .. } => (
                StatusCode::INSUFFICIENT_STORAGE,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::ShardIntegrity { .. } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::PipelineError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::PeerNotFound(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::Unauthorized(_) => (
                StatusCode::UNAUTHORIZED,
                self.0.to_string(),
                "authentication_error",
            ),
            SwarmError::Config(_) => (
                StatusCode::BAD_REQUEST,
                self.0.to_string(),
                "invalid_request_error",
            ),
            SwarmError::InvalidNickname(_) => (
                StatusCode::BAD_REQUEST,
                self.0.to_string(),
                "invalid_request_error",
            ),
            _ => {
                // Log the full error internally but return a generic message
                // to avoid leaking internal paths, peer errors, or DB details.
                tracing::error!(
                    error = %self.0,
                    "Internal server error"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "An internal error occurred".to_string(),
                    "server_error",
                )
            }
        };
        // Log 5xx errors so they appear in tracing output (catch non-catch-all 5xx)
        if status.is_server_error() && error_type != "server_error" {
            tracing::error!(
                status = status.as_u16(),
                error = %message,
                "Server error"
            );
        }

        let hint = error_hint(&self.0);

        let mut error_obj = serde_json::json!({
            "message": message,
            "type": error_type,
            "code": status.as_u16()
        });
        if let Some(hint_text) = hint {
            error_obj["hint"] = serde_json::Value::String(hint_text.to_string());
        }

        (status, Json(serde_json::json!({ "error": error_obj }))).into_response()
    }
}

/// Return an actionable hint for common error variants.
/// These help users understand what to do when they encounter an error.
pub fn error_hint(err: &SwarmError) -> Option<&'static str> {
    match err {
        SwarmError::ModelNotAvailable(_) => Some(
            "Download shards via the admin dashboard or: \
             curl -X POST http://localhost:8800/api/admin/hf/download-shards -H 'Authorization: Bearer <key>' \
             -d '{\"model_id\": \"<model>\"}'",
        ),
        SwarmError::NoModelLoaded => Some(
            "No model is loaded yet. Download model shards via the admin dashboard \
             or POST /api/admin/hf/download-shards with a model_id.",
        ),
        SwarmError::InsufficientCredits { .. } => Some(
            "Earn credits by hosting shards, serving inference, or seeding data to peers. \
             Check your balance at GET /api/admin/credits.",
        ),
        SwarmError::InsufficientCapacity(_) => Some(
            "Not enough nodes have the required shards. Wait for more peers to join \
             or download additional shards locally via POST /api/admin/hf/download-shards.",
        ),
        SwarmError::InsufficientDisk { .. } => Some(
            "Free up disk space or increase max_disk_mb in your config.toml under [resources].",
        ),
        SwarmError::Unauthorized(_) => Some(
            "Include your API key in the Authorization header: \
             -H 'Authorization: Bearer <your-api-key>'. \
             Find your key in the daemon startup logs or GET /api/admin/api-key.",
        ),
        SwarmError::PeerNotFound(_) => Some(
            "The target peer is offline or unreachable. Check your network connection \
             and ensure bootstrap peers are configured in config.toml.",
        ),
        SwarmError::ShardIntegrity { .. } => Some(
            "The shard file is corrupted. It will be quarantined and re-downloaded automatically. \
             You can also manually re-download via POST /api/admin/hf/download-shards.",
        ),
        SwarmError::PipelineError(_) => Some(
            "Pipeline assembly failed — this often means required shards are missing or \
             nodes holding them went offline. Try again or download more shards locally.",
        ),
        SwarmError::InferenceTimeout(_) => Some(
            "The request took too long. Try a shorter prompt, reduce max_tokens, \
             or check if serving nodes are overloaded.",
        ),
        SwarmError::Config(_) => Some(
            "Check your config.toml for syntax errors. See config/default.toml for valid options.",
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_for_model_not_available() {
        let err = SwarmError::ModelNotAvailable(ModelId("test-model".into()));
        assert!(error_hint(&err).unwrap().contains("download-shards"));
    }

    #[test]
    fn hint_for_no_model_loaded() {
        let err = SwarmError::NoModelLoaded;
        assert!(error_hint(&err).unwrap().contains("Download model shards"));
    }

    #[test]
    fn hint_for_insufficient_credits() {
        let err = SwarmError::InsufficientCredits {
            balance: 10,
            required: 100,
        };
        assert!(error_hint(&err).unwrap().contains("Earn credits"));
    }

    #[test]
    fn hint_for_unauthorized() {
        let err = SwarmError::Unauthorized("missing token".into());
        assert!(error_hint(&err).unwrap().contains("Authorization"));
    }

    #[test]
    fn hint_for_insufficient_disk() {
        let err = SwarmError::InsufficientDisk {
            need_mb: 1000,
            have_mb: 100,
        };
        assert!(error_hint(&err).unwrap().contains("max_disk_mb"));
    }

    #[test]
    fn hint_for_shard_integrity() {
        let err = SwarmError::ShardIntegrity {
            expected: "abc".into(),
            actual: "def".into(),
        };
        assert!(error_hint(&err).unwrap().contains("corrupted"));
    }

    #[test]
    fn no_hint_for_generic_error() {
        let err = SwarmError::Internal("something broke".into());
        assert!(error_hint(&err).is_none());
    }
}
