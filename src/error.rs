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
    #[error("No vision encoder (mmproj) available for model {0}")]
    VisionEncoderUnavailable(ModelId),

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

    // Validation
    #[error("Validation error: {0}")]
    Validation(String),

    // Resource lookup miss (generic 404 — distinct from Validation 400). Use
    // for endpoints where the request shape is fine but the named resource
    // (response id, session id, etc.) doesn't exist in the store.
    #[error("Not found: {0}")]
    NotFound(String),

    // Config
    #[error("Configuration error: {0}")]
    Config(String),

    // Provider proxy
    #[error("Provider error ({status}): {body}")]
    ProviderError { status: u16, body: String },

    // Private mode
    #[error("Private mode: model {model_id} not fully available in your device pool (missing shards: {missing_shards:?})")]
    PrivateModeUnavailable {
        model_id: String,
        missing_shards: Vec<u32>,
    },

    // Overload
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    // Generic
    #[error("Internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

impl SwarmError {
    /// Build `SwarmError::Internal` from any `Display`-able error. Use as
    /// `.map_err(SwarmError::internal)` to avoid the closure boilerplate.
    pub fn internal<E: std::fmt::Display>(e: E) -> Self {
        SwarmError::Internal(e.to_string())
    }
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
                "server_error",
            ),
            SwarmError::InferenceTimeout(_) => (
                StatusCode::GATEWAY_TIMEOUT,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::InsufficientCredits { .. } => (
                StatusCode::PAYMENT_REQUIRED,
                self.0.to_string(),
                "insufficient_credits",
            ),
            SwarmError::InsufficientCapacity(_) | SwarmError::ServiceUnavailable(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::PrivateModeUnavailable { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                self.0.to_string(),
                "private_mode_error",
            ),
            SwarmError::VisionEncoderUnavailable(_) => (
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
            SwarmError::Inference(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::ProviderError { status, ref body } => {
                let http_status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
                // Truncate provider body to avoid leaking upstream internals
                let safe_body: String = body.chars().take(512).collect();
                (
                    http_status,
                    format!("Provider error: {safe_body}"),
                    "server_error",
                )
            }
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
            // SwarmError::Config is for daemon startup / config-file errors per
            // .claude/rules/completeness.md. If it surfaces in an HTTP response
            // path, the daemon has shipped misconfigured — that's a 500, not a
            // 400 (the user did not send invalid input).
            SwarmError::Config(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.0.to_string(),
                "server_error",
            ),
            SwarmError::InvalidNickname(_) | SwarmError::Validation(_) => (
                StatusCode::BAD_REQUEST,
                self.0.to_string(),
                "invalid_request_error",
            ),
            SwarmError::ShardNotFound(_) => {
                (StatusCode::NOT_FOUND, self.0.to_string(), "not_found_error")
            }
            SwarmError::NotFound(_) => {
                (StatusCode::NOT_FOUND, self.0.to_string(), "not_found_error")
            }
            // Network upstream-unreachable: 502 + a distinct error_type so
            // SDK retry logic can distinguish 'try later' from a 500 bug.
            SwarmError::Network(_) => {
                (StatusCode::BAD_GATEWAY, self.0.to_string(), "network_error")
            }
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
            "param": null,
            "code": error_type
        });
        if let Some(hint_text) = hint {
            error_obj["hint"] = serde_json::Value::String(hint_text.to_string());
        }

        (status, Json(serde_json::json!({ "error": error_obj }))).into_response()
    }
}

/// Return an actionable hint for common error variants.
/// These are user-facing — no curl commands or API paths.
pub fn error_hint(err: &SwarmError) -> Option<&'static str> {
    match err {
        SwarmError::ModelNotAvailable(_) => Some(
            "This model isn't available yet. Open the Models tab in the dashboard to browse \
             and download models, or wait for auto-manage to acquire it from the network.",
        ),
        SwarmError::NoModelLoaded => Some(
            "No model is loaded yet. Go to the dashboard and select a model to download, \
             or connect to more peers so models can be served from the network.",
        ),
        SwarmError::InsufficientCredits { .. } => Some(
            "Your credit balance is too low. Earn credits by hosting model shards \
             (happens automatically) or serving inference for other users. \
             Check your balance on the dashboard.",
        ),
        SwarmError::InsufficientCapacity(_) => Some(
            "Not enough peers have the shards needed for this model. \
             Try again later as more peers come online, or download the model \
             shards yourself from the Models tab.",
        ),
        SwarmError::InsufficientDisk { .. } => Some(
            "Not enough disk space. Free up space or increase the storage limit \
             in Settings → Advanced → max_disk_mb.",
        ),
        SwarmError::Unauthorized(_) => Some(
            "Authentication required. Your API key can be found on the dashboard \
             Settings page. Include it as a Bearer token in the Authorization header.",
        ),
        SwarmError::PeerNotFound(_) => Some(
            "That peer is offline or unreachable. Check your internet connection \
             and try again later.",
        ),
        SwarmError::ShardIntegrity { .. } => Some(
            "A model file was corrupted and will be re-downloaded automatically. \
             This is usually caused by an interrupted download — try again in a moment.",
        ),
        SwarmError::PipelineError(_) => Some(
            "Something went wrong assembling the inference pipeline. This usually means \
             a peer went offline mid-request. Try again — a different route will be used.",
        ),
        SwarmError::InferenceTimeout(_) => Some(
            "The request took too long. Try a shorter prompt, reduce the max tokens, \
             or wait for a less busy time.",
        ),
        SwarmError::Config(_) => Some(
            "There's a configuration issue. Check Settings in the dashboard \
             or review your config.toml file.",
        ),
        SwarmError::ProviderError { status, ref body } => {
            let lower = body.to_lowercase();
            let is_quota = *status == 402
                || *status == 429
                || lower.contains("quota")
                || lower.contains("billing")
                || lower.contains("insufficient")
                || lower.contains("balance")
                || lower.contains("exceeded")
                || lower.contains("limit")
                || lower.contains("credits")
                || lower.contains("payment");
            if is_quota {
                Some(
                    "Your cloud provider credits may be exhausted or rate-limited. \
                     Top up your account on the provider's website, switch to a free-tier provider \
                     (DeepSeek, Groq, NVIDIA NIM), or use a local swarm model instead.",
                )
            } else if *status == 401 || *status == 403 {
                Some(
                    "Your API key appears to be invalid or revoked. \
                     Update it in Settings → Cloud Providers.",
                )
            } else {
                Some(
                    "The cloud provider returned an error. Try again, \
                     switch to a different model, or use a local swarm model.",
                )
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_for_model_not_available() {
        let err = SwarmError::ModelNotAvailable(ModelId("test-model".into()));
        assert!(error_hint(&err).unwrap().contains("Models tab"));
    }

    #[test]
    fn hint_for_no_model_loaded() {
        let err = SwarmError::NoModelLoaded;
        assert!(error_hint(&err).unwrap().contains("dashboard"));
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
