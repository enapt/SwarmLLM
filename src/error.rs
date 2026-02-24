use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use thiserror::Error;

use crate::types::ModelId;

#[derive(Error, Debug)]
pub enum SwarmError {
    // Network (stubs for Phase 1, populated in Phase 2)
    #[error("Network error: {0}")]
    Network(String),

    // Inference
    #[error("Model not available: {0}")]
    ModelNotAvailable(ModelId),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Inference timeout after {0}s")]
    InferenceTimeout(u64),
    #[error("No model loaded")]
    NoModelLoaded,

    // Storage
    #[error("Database error: {0}")]
    Database(#[from] sled::Error),

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
            SwarmError::Config(_) => (StatusCode::BAD_REQUEST, self.0.to_string()),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".into(),
            ),
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
