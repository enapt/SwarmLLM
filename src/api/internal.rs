//! Internal research API — hidden state inspection.
//!
//! POST /v1/internal/hidden-states
//!
//! Runs a prompt through the locally loaded model and returns intermediate
//! hidden-state activations at the requested transformer layers. Gated behind
//! `api.expose_hidden_states = true` in config.

use axum::extract::State;
use axum::Json;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::types::{HiddenStateRequest, HiddenStateResponse};

/// Maximum number of layers that can be requested at once.
const MAX_RETURN_LAYERS: usize = 128;

/// Maximum prompt length in characters.
const MAX_PROMPT_LENGTH: usize = 100_000;

/// POST /v1/internal/hidden-states
///
/// Returns hidden-state tensors at the requested layers. Requires:
/// - `api.expose_hidden_states = true` in config (returns 404 otherwise)
/// - Bearer auth (handled by middleware)
/// - A locally loaded model (returns 503 otherwise)
pub async fn hidden_states(
    State(state): State<AppState>,
    Json(req): Json<HiddenStateRequest>,
) -> Result<Json<HiddenStateResponse>, ApiError> {
    tracing::debug!(
        layers = ?req.return_layers,
        prompt_len = req.prompt.len(),
        "DIAG: hidden_states request"
    );

    // Gate: feature must be enabled in config
    if !state.config.api.expose_hidden_states {
        tracing::debug!("DIAG: hidden_states gate denied — endpoint disabled");
        return Err(ApiError(crate::error::SwarmError::Inference(
            "Hidden states endpoint is disabled. Set api.expose_hidden_states = true in config."
                .to_string(),
        )));
    }

    // Validate request
    if req.return_layers.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "return_layers must not be empty".to_string(),
        )));
    }
    if req.return_layers.len() > MAX_RETURN_LAYERS {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "return_layers has {} entries (max {})",
            req.return_layers.len(),
            MAX_RETURN_LAYERS
        ))));
    }
    if req.prompt.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "prompt must not be empty".to_string(),
        )));
    }
    if req.prompt.len() > MAX_PROMPT_LENGTH {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "prompt too long: {} chars (max {})",
            req.prompt.len(),
            MAX_PROMPT_LENGTH
        ))));
    }

    // Check for duplicate layer indices
    let mut unique_layers = req.return_layers.clone();
    unique_layers.sort_unstable();
    unique_layers.dedup();
    if unique_layers.len() != req.return_layers.len() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "return_layers contains duplicate layer indices".to_string(),
        )));
    }

    // Find a loaded split model to use for hidden state extraction
    let split_model_key = state
        .shared_state
        .split_models
        .iter()
        .next()
        .map(|entry| entry.key().clone());

    let split_model_key =
        split_model_key.ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;

    tracing::info!(
        model_key = ?split_model_key,
        requested_model = %req.model,
        layers = ?req.return_layers,
        prompt_len = req.prompt.len(),
        "Hidden states request"
    );

    // Hidden state extraction requires in-process model access.
    // The model now lives in a worker subprocess for GPU memory isolation.
    // This endpoint is not yet supported with subprocess inference.
    Err(ApiError(crate::error::SwarmError::Internal(
        "Hidden states API not yet supported with subprocess inference".into(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_return_layers_is_reasonable() {
        let val = MAX_RETURN_LAYERS;
        assert!(val >= 64);
        assert!(val <= 256);
    }

    #[test]
    fn hidden_state_tensor_base64_encode() {
        use base64::Engine;
        let data = vec![0u8; 16];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&data);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, data);
    }
}
