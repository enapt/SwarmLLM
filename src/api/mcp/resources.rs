use serde_json::{json, Value};

use super::tools::enumerate_models;
use super::types::{JsonRpcResponse, RESOURCE_NOT_FOUND};
use crate::api::server::AppState;

pub(super) fn handle_resources_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "resources": [
                {
                    "uri": "swarmllm://status",
                    "name": "Node Status",
                    "description": "Current SwarmLLM node status including loaded models and peer count",
                    "mimeType": "application/json"
                },
                {
                    "uri": "swarmllm://models",
                    "name": "Available Models",
                    "description": "All models available for inference: local, network, and cloud providers with capabilities and status",
                    "mimeType": "application/json"
                },
                {
                    "uri": "swarmllm://peers",
                    "name": "Connected Peers",
                    "description": "Currently connected P2P peers with latency, trust, load, and shard info",
                    "mimeType": "application/json"
                }
            ]
        }),
    )
}

pub(super) async fn handle_resources_read(
    state: &AppState,
    id: Option<Value>,
    params: Value,
) -> JsonRpcResponse {
    let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");

    match uri {
        "swarmllm://status" => resource_status(state, id).await,
        "swarmllm://models" => resource_models(state, id).await,
        "swarmllm://peers" => resource_peers(state, id).await,
        _ => JsonRpcResponse::error(id, RESOURCE_NOT_FOUND, format!("Unknown resource: {uri}")),
    }
}

// ---- Tool implementations ----
async fn resource_status(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let info = state.shared_state.loaded_model_info.read().await;
    let model_name = info.as_ref().map(|i| i.name.clone()).unwrap_or_default();
    let model_loaded = info.is_some();
    drop(info);

    let peer_count = state.shared_state.peer_registry.len();

    let status = json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": model_loaded,
        "model_name": model_name,
        "peers": peer_count,
    });

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://status",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&status).unwrap_or_default()
                }
            ]
        }),
    )
}

async fn resource_models(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let base = enumerate_models(state).await;
    let mut models = Vec::with_capacity(base.len());

    for (model_id, name, source) in base {
        let mut entry = json!({ "id": model_id, "name": name, "source": source });
        match source {
            "local" => {
                if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("size_bytes".to_string(), json!(info.size_bytes));
                        obj.insert("status".to_string(), json!("loaded"));
                    }
                }
            }
            "network" => {
                if let Some(manifest) = state
                    .shared_state
                    .model_registry
                    .get_manifest(&crate::types::ModelId(model_id))
                {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("shards".to_string(), json!(manifest.shards.len()));
                        obj.insert(
                            "architecture".to_string(),
                            json!(format!("{:?}", manifest.architecture)),
                        );
                    }
                }
            }
            _ => {}
        }
        models.push(entry);
    }

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://models",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&models).unwrap_or_default()
                }
            ]
        }),
    )
}

async fn resource_peers(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let mut peers = Vec::new();
    for entry in state.shared_state.peer_registry.iter() {
        let mut p = mcp_peer_json(&entry);
        let region = entry
            .value()
            .capability
            .as_ref()
            .and_then(|c| c.region.as_deref())
            .unwrap_or("unknown");
        if let Some(obj) = p.as_object_mut() {
            obj.insert("region".into(), json!(region));
        }
        peers.push(p);
    }

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://peers",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&peers).unwrap_or_default()
                }
            ]
        }),
    )
}

/// Build a compact MCP peer summary JSON object.
pub(super) fn mcp_peer_json(
    entry: &dashmap::mapref::multiple::RefMulti<'_, crate::types::NodeId, crate::types::PeerInfo>,
) -> Value {
    let peer = entry.value();
    json!({
        "node_id_short": entry.key().to_string(),
        "latency_ms": peer.latency_ms,
        "is_lan": peer.is_lan_peer,
        "trust_score": peer.trust_score,
        "active_requests": peer.active_request_count,
    })
}
