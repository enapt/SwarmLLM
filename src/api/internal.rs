//! Internal research API — hidden state inspection.
//!
//! POST /v1/internal/hidden-states
//!
//! Runs a prompt through the locally loaded model and returns intermediate
//! hidden-state activations at the requested transformer layers. Gated behind
//! `api.expose_hidden_states = true` in config.

use axum::extract::State;
use axum::Json;
use std::collections::HashMap;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::types::{HiddenStateRequest, HiddenStateResponse, HiddenStateTensor};

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

    let capture_set: std::collections::HashSet<usize> = req.return_layers.iter().copied().collect();

    // Run the forward pass with hidden state capture in a blocking task
    let shared_state = state.shared_state.clone();
    let prompt = req.prompt.clone();
    let return_layers = req.return_layers.clone();

    // Get the model entry and lock it
    let model_entry = shared_state
        .split_models
        .get(&split_model_key)
        .ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;
    let model_arc = model_entry.model.clone();
    drop(model_entry); // Release DashMap guard

    let mut model = model_arc.lock().await;

    // Tokenize the prompt
    let token_ids: Vec<i64> = if let Some(tokenizer) = model.tokenizer() {
        tokenizer.encode(&prompt)
    } else {
        prompt.bytes().map(|b| b as i64).collect()
    };
    let tokens_processed = token_ids.len();

    // Build input tensor
    let input =
        candle_core::Tensor::from_vec(token_ids, &[1, tokens_processed], &candle_core::Device::Cpu)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "tensor create: {e}"
                )))
            })?;

    let kv_store = crate::inference::split::KvCacheStore::new(std::time::Duration::from_secs(60));
    let request_id = format!("hidden-states-{}", uuid::Uuid::new_v4());

    // Run the forward pass with hidden state capture (CPU-bound, use block_in_place)
    let (_output, captured) = tokio::task::block_in_place(|| {
        model.forward_with_hidden_capture(&input, 0, &kv_store, &request_id, &capture_set)
    })
    .map_err(ApiError)?;

    // Convert captured tensors to API response format
    use base64::Engine;
    let mut hidden_states = HashMap::new();
    for &layer_idx in &return_layers {
        if let Some(tensor) = captured.get(&layer_idx) {
            let shape: Vec<usize> = tensor.dims().to_vec();
            // Flatten to f32 bytes
            let flat = tensor
                .flatten_all()
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| {
                    ApiError(crate::error::SwarmError::Internal(format!(
                        "tensor serialize: {e}"
                    )))
                })?;
            let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
            let data_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

            hidden_states.insert(
                layer_idx,
                HiddenStateTensor {
                    shape,
                    dtype: "f32".to_string(),
                    data_base64,
                },
            );
        }
    }

    Ok(Json(HiddenStateResponse {
        hidden_states,
        tokens_processed,
    }))
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
