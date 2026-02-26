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
    Database(#[from] sled::Error),
    #[error("Insufficient disk space: need {need_mb}MB, have {have_mb}MB")]
    InsufficientDisk { need_mb: u64, have_mb: u64 },

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
        let (status, message) = match &self.0 {
            SwarmError::ModelNotAvailable(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            SwarmError::NoModelLoaded => (StatusCode::SERVICE_UNAVAILABLE, self.0.to_string()),
            SwarmError::InferenceTimeout(_) => (StatusCode::GATEWAY_TIMEOUT, self.0.to_string()),
            SwarmError::InsufficientCredits { .. } => {
                (StatusCode::TOO_MANY_REQUESTS, self.0.to_string())
            }
            SwarmError::Config(_) => (StatusCode::BAD_REQUEST, self.0.to_string()),
            SwarmError::InvalidNickname(_) => (StatusCode::BAD_REQUEST, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };
        (
            status,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "swarm_error",
                    "code": status.as_u16()
                }
            })),
        )
            .into_response()
    }
}
