use crate::error::ApiError;

mod adapters;
mod helpers;
mod lifecycle;
mod listing;
mod shards;

pub use adapters::{delete_adapter, list_adapters, register_adapter, RegisterAdapterRequest};
pub use helpers::serialize_acquisition_to_json;
pub use lifecycle::{
    delete_model, get_model_auto_manage, get_model_encrypted_pipeline, set_model_auto_manage,
    set_model_encrypted_pipeline, unload_model, EncryptedPipelineUpdate, ModelAutoManageUpdate,
};
pub use listing::{
    add_model_interest, download_queue, list_models, model_acquisition_status, model_metadata,
    pipeline_plan, prune_history, shard_storage,
};
pub use shards::{
    delete_shard, download_shard, load_shard, lock_shard, unload_shard, ShardLockUpdate,
};

/// Validate that a model ID from a URL path param is within length bounds.
pub(crate) fn validate_model_id(model_id: &str) -> Result<(), ApiError> {
    // SEC: reject empty model_id. `model_dir("")` resolves to
    // `data_dir/models/_/` (sanitize_path_component maps empty/NUL to `_`),
    // colliding with any other model whose sanitized name is `_`. A peer
    // gossiping a `ModelManifest` with `id = ""` (or `"\0"`) stomps the
    // local file layout. The HTTP path extractor blocks empty path
    // segments, but this validator is also used as a defensive gate
    // wherever ModelId enters the system.
    if model_id.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Model ID must not be empty".into(),
        )));
    }
    if model_id.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Model ID must be 256 characters or fewer".into(),
        )));
    }
    if model_id.contains("..")
        || model_id.contains('/')
        || model_id.contains('\\')
        || model_id.contains('\0')
    {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Model ID contains invalid characters".into(),
        )));
    }
    Ok(())
}

/// Validate model ID path param and reject the mmproj sentinel shard index.
pub(crate) fn validate_shard_params(model_id: &str, shard_index: u32) -> Result<(), ApiError> {
    validate_model_id(model_id)?;
    if shard_index == u32::MAX {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Reserved shard index".into(),
        )));
    }
    Ok(())
}
