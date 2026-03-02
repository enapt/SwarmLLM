use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::StreamExt;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::chat_template;
use crate::inference::router::{RouterCommand, StreamingTokenEvent};
use crate::types::{
    ChatMessage, ImageData, InferenceRequest, ModelId, NodeId, PriorityTier, SamplingParams,
};

/// Timeout for peer-forwarded inference requests (seconds).
const INFERENCE_FORWARD_TIMEOUT_SECS: u64 = 120;

/// SSE keep-alive interval for streaming responses (seconds).
const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Maximum cold-start wait time before returning 503 (seconds).
const COLD_START_WAIT_SECS: u32 = 10;

// ---- Request types ----

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    /// Optional session ID for multi-turn KV-cache reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional LoRA adapter ID for per-request fine-tuned inference (SwarmLLM extension).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
}

/// API-layer chat message supporting OpenAI's multimodal content format.
///
/// The `content` field can be either:
/// - A plain string: `"content": "Hello"`
/// - An array of content parts: `"content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]`
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiChatMessage {
    pub role: crate::types::Role,
    #[serde(default)]
    pub content: MessageContent,
}

/// OpenAI-compatible content field: either a plain string or an array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A single content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlRef },
}

/// Image URL reference — supports base64 data URIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlRef {
    pub url: String,
}

/// Maximum allowed base64-encoded image size (20MB decoded).
const MAX_IMAGE_BASE64_BYTES: usize = 20 * 1024 * 1024;

impl ApiChatMessage {
    /// Convert to the internal `ChatMessage` type, decoding any base64 images.
    pub fn to_chat_message(&self) -> Result<ChatMessage, crate::error::SwarmError> {
        let (text, images) = match &self.content {
            MessageContent::Text(s) => (s.clone(), vec![]),
            MessageContent::Parts(parts) => {
                let mut text_parts = Vec::new();
                let mut images = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => text_parts.push(text.clone()),
                        ContentPart::ImageUrl { image_url } => {
                            let img = decode_image_url(&image_url.url)?;
                            images.push(img);
                        }
                    }
                }
                (text_parts.join("\n"), images)
            }
        };

        Ok(ChatMessage {
            role: self.role.clone(),
            content: text,
            images,
        })
    }
}

impl ChatCompletionRequest {
    /// Convert API messages to internal ChatMessage format, decoding images.
    pub fn to_internal_messages(&self) -> Result<Vec<ChatMessage>, crate::error::SwarmError> {
        self.messages.iter().map(|m| m.to_chat_message()).collect()
    }
}

/// Decode a base64 data URI image to raw RGB pixels.
fn decode_image_url(url: &str) -> Result<ImageData, crate::error::SwarmError> {
    let base64_data = if let Some(rest) = url.strip_prefix("data:") {
        if let Some(comma_pos) = rest.find(',') {
            let header = &rest[..comma_pos];
            if !header.contains("base64") {
                return Err(crate::error::SwarmError::Config(
                    "Only base64 data URIs are supported for image_url".into(),
                ));
            }
            &rest[comma_pos + 1..]
        } else {
            return Err(crate::error::SwarmError::Config(
                "Invalid data URI format".into(),
            ));
        }
    } else {
        return Err(crate::error::SwarmError::Config(
            "Only data: URIs are supported for image_url (not remote URLs)".into(),
        ));
    };

    if base64_data.len() > MAX_IMAGE_BASE64_BYTES * 4 / 3 + 4 {
        return Err(crate::error::SwarmError::Config(format!(
            "Image too large (max {}MB)",
            MAX_IMAGE_BASE64_BYTES / (1024 * 1024)
        )));
    }

    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| crate::error::SwarmError::Config(format!("Invalid base64 image: {e}")))?;

    if bytes.len() > MAX_IMAGE_BASE64_BYTES {
        return Err(crate::error::SwarmError::Config(format!(
            "Decoded image too large: {}MB (max {}MB)",
            bytes.len() / (1024 * 1024),
            MAX_IMAGE_BASE64_BYTES / (1024 * 1024)
        )));
    }

    let img = image::load_from_memory(&bytes)
        .map_err(|e| crate::error::SwarmError::Config(format!("Failed to decode image: {e}")))?;

    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());

    Ok(ImageData {
        rgb_bytes: rgb.into_raw(),
        width,
        height,
    })
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_max_tokens() -> u32 {
    2048
}

impl ChatCompletionRequest {
    fn to_sampling_params(&self) -> SamplingParams {
        let stop = match &self.stop {
            Some(StopSequence::Single(s)) => vec![s.clone()],
            Some(StopSequence::Multiple(v)) => v.clone(),
            None => vec![],
        };
        SamplingParams {
            // Clamp temperature to [0.0, 2.0] to prevent invalid values
            temperature: self.temperature.clamp(0.0, 2.0),
            // Clamp top_p to (0.0, 1.0]
            top_p: self.top_p.clamp(f32::EPSILON, 1.0),
            top_k: 40,
            // Clamp max_tokens to a reasonable range
            max_tokens: self.max_tokens.min(32768),
            stop,
            frequency_penalty: self.frequency_penalty.clamp(-2.0, 2.0),
            presence_penalty: self.presence_penalty.clamp(-2.0, 2.0),
        }
    }
}

// ---- Response types (OpenAI-compatible) ----

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Session ID for multi-turn KV-cache reuse. Echoed from the request
    /// or auto-generated. Clients should pass this back in subsequent
    /// requests to reuse cached KV state and skip redundant prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageResponse,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---- Model list types ----

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

// ---- Handlers ----

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    // Validate session_id length to prevent memory abuse
    if let Some(ref sid) = req.session_id {
        if sid.len() > 256 {
            return Err(ApiError(crate::error::SwarmError::Config(
                "session_id too long (max 256 chars)".into(),
            )));
        }
    }

    // Convert API messages to internal format (decode base64 images if present)
    let internal_messages = req.to_internal_messages().map_err(ApiError)?;

    let request_id = format!("swarm-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    // Track requests made by this node
    if let Ok(mut stats) = state.shared_state.node_stats.try_write() {
        stats.requests_made += 1;
    }

    // Resolve "auto" model alias to the first available model.
    // Check loaded_model_info first (local split model), then fall back to registry ID.
    if req.model == "auto" {
        let resolved = {
            let info = state.shared_state.loaded_model_info.read().await;
            info.as_ref().and_then(|i| {
                // Find the registry key for this loaded model (may differ from display name)
                let slug = i
                    .name
                    .to_lowercase()
                    .replace(' ', "-")
                    .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "");
                let registry_id = state
                    .shared_state
                    .model_registry
                    .get_manifest(&crate::types::ModelId(slug.clone()))
                    .map(|m| m.id.0.clone())
                    .or_else(|| {
                        state
                            .shared_state
                            .model_registry
                            .models()
                            .into_iter()
                            .find(|m| m.name == i.name)
                            .map(|m| m.id.0.clone())
                    });
                registry_id.or(Some(slug))
            })
        };
        // Fall back to the first model in the registry if nothing loaded locally
        let resolved = resolved.or_else(|| {
            state
                .shared_state
                .model_registry
                .models()
                .into_iter()
                .next()
                .map(|m| m.id.0.clone())
        });
        if let Some(id) = resolved {
            tracing::info!(resolved_model = %id, "Resolved 'auto' model alias");
            req.model = id;
        }
    }

    let image_count: usize = internal_messages.iter().map(|m| m.images.len()).sum();
    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        messages = internal_messages.len(),
        images = image_count,
        stream = req.stream,
        "Chat completion request"
    );

    // Get model name from lock-free cache
    let model_name = {
        let info = state.shared_state.loaded_model_info.read().await;
        info.as_ref().map(|i| i.name.clone())
    };

    // No local full-model executor — use distributed inference or forward.
    // Nodes are NOT required to have all shards. Any node can initiate inference
    // as long as the network collectively covers all layers.
    // The `x-swarm-forwarded` header prevents infinite forwarding loops between nodes.
    let is_forwarded = headers.get("x-swarm-forwarded").is_some();

    if model_name.is_none() {
        // Priority 1: Check if all layers are covered across the network for
        // distributed inference. The local node may have zero, some, or all shards —
        // it doesn't matter as long as the network covers every layer.
        if all_shards_available(&state, &req.model) {
            tracing::info!(
                request_id = %request_id,
                model = %req.model,
                stream = req.stream,
                "All layers covered across network — using distributed inference"
            );

            if let Some(router_tx) = &state.router_tx {
                if req.stream {
                    return router_inference_stream(
                        router_tx.clone(),
                        &state,
                        &req,
                        internal_messages.clone(),
                        request_id,
                        created,
                    )
                    .await;
                } else {
                    return router_inference(
                        router_tx.clone(),
                        &req,
                        internal_messages.clone(),
                        request_id,
                        created,
                    )
                    .await;
                }
            } else {
                return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
            }
        }

        // Priority 2: Forward to a peer that hosts shards for this model.
        // That peer can handle inference locally or build its own pipeline.
        if !is_forwarded {
            if let Some(peer_url) = find_peer_with_model(&state, &req.model) {
                tracing::info!(
                    request_id = %request_id,
                    peer_url = %peer_url,
                    "Forwarding request to peer"
                );
                return forward_to_peer(&peer_url, &req, req.stream).await;
            }
        }

        // Cold-start wait: shard announcements may still be propagating.
        // Poll for up to 10 seconds before giving up.
        let max_polls = COLD_START_WAIT_SECS * 2; // 500ms intervals
        for attempt in 1..=max_polls {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            // Re-check distributed inference availability
            if all_shards_available(&state, &req.model) {
                tracing::info!(
                    request_id = %request_id,
                    model = %req.model,
                    wait_ms = attempt * 500,
                    "Model became available after cold-start wait"
                );
                if let Some(router_tx) = &state.router_tx {
                    if req.stream {
                        return router_inference_stream(
                            router_tx.clone(),
                            &state,
                            &req,
                            internal_messages.clone(),
                            request_id,
                            created,
                        )
                        .await;
                    } else {
                        return router_inference(
                            router_tx.clone(),
                            &req,
                            internal_messages.clone(),
                            request_id,
                            created,
                        )
                        .await;
                    }
                }
                break;
            }
            // Re-check peer forwarding
            if !is_forwarded {
                if let Some(peer_url) = find_peer_with_model(&state, &req.model) {
                    tracing::info!(
                        request_id = %request_id,
                        peer_url = %peer_url,
                        wait_ms = attempt * 500,
                        "Found peer after cold-start wait"
                    );
                    return forward_to_peer(&peer_url, &req, req.stream).await;
                }
            }
        }

        // Cloud provider fallback: proxy to configured cloud provider if model matches
        let body = serde_json::to_value(&req).unwrap_or_default();
        if let Some(response) =
            crate::api::providers::try_proxy_openai(&state, &body, req.stream).await?
        {
            return Ok(response);
        }

        return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
    }

    let model_name = model_name.unwrap();

    // Build prompt using chat template from GGUF if available, else ChatML fallback.
    let (tmpl, bos, eos) = {
        let info = state.shared_state.loaded_model_info.read().await;
        match info.as_ref() {
            Some(i) => (
                i.chat_template.clone(),
                i.bos_token.clone(),
                i.eos_token.clone(),
            ),
            None => (None, String::new(), String::new()),
        }
    };
    let prompt = chat_template::build_prompt(&internal_messages, tmpl.as_deref(), &bos, &eos);
    let params = req.to_sampling_params();

    // Fast path: if we have a complete local split model (all layers), generate directly.
    // This avoids the distributed pipeline overhead (per-token segment coordination,
    // activation serialization, mutex per token). ~5-10x faster for local inference.
    // Uses the pre-computed is_complete flag — no model mutex needed.
    let has_local_split_model = state
        .shared_state
        .split_models
        .iter()
        .next()
        .map(|e| e.value().is_complete)
        .unwrap_or(false);

    if has_local_split_model {
        if req.stream {
            return Ok(split_stream_response(
                state, request_id, created, model_name, prompt, params,
            )
            .await
            .into_response());
        } else {
            return split_non_stream_response(
                state, request_id, created, model_name, prompt, params,
            )
            .await;
        }
    }

    // Distributed inference: network covers all layers across multiple nodes.
    let peers_have_shards = all_shards_available(&state, &req.model)
        || state.shared_state.config.inference.shard_range.is_some();
    if peers_have_shards {
        if let Some(router_tx) = &state.router_tx {
            if req.stream {
                return router_inference_stream(
                    router_tx.clone(),
                    &state,
                    &req,
                    internal_messages.clone(),
                    request_id,
                    created,
                )
                .await;
            } else {
                return router_inference(
                    router_tx.clone(),
                    &req,
                    internal_messages.clone(),
                    request_id,
                    created,
                )
                .await;
            }
        }
    }

    if req.stream {
        // Streaming: use direct executor path for real token-by-token SSE
        Ok(
            stream_response(state, request_id, created, model_name, prompt, params)
                .await
                .into_response(),
        )
    } else if let Some(router_tx) = &state.router_tx {
        // Non-streaming: route through InferenceRouter for priority queueing
        router_inference(
            router_tx.clone(),
            &req,
            internal_messages,
            request_id,
            created,
        )
        .await
    } else {
        // Fallback: direct executor path
        Ok(
            non_stream_response(state, request_id, created, model_name, prompt, params)
                .await?
                .into_response(),
        )
    }
}

/// Find a peer that hosts shards for this model and return its HTTP base URL.
/// This is a fallback for when not all layers are covered network-wide — the
/// peer may be able to handle the request directly or assemble its own pipeline.
fn find_peer_with_model(state: &AppState, model: &str) -> Option<String> {
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            let has_model = cap.hosted_shards.iter().any(|s| s.model_id.0 == model);
            if has_model {
                if let Some(url) = peer_http_url(peer) {
                    return Some(url);
                }
            }
        }
    }
    None
}

/// Check if all layers for a model are covered across the network (for distributed inference).
/// This does NOT require any single node to have all shards — it only requires that every
/// shard has at least one holder somewhere in the network so the pipeline scheduler can
/// assemble a complete pipeline across multiple nodes.
pub fn all_shards_available(state: &AppState, model_name: &str) -> bool {
    let model_id = ModelId(model_name.to_string());

    let manifest = match state.shared_state.model_registry.get_manifest(&model_id) {
        Some(m) => m,
        None => {
            tracing::debug!(model = %model_name, "all_shards_available: no manifest");
            return false;
        }
    };

    // Need a valid layer count for the scheduler to work
    if manifest.num_layers == 0 {
        tracing::debug!(model = %model_name, "all_shards_available: num_layers=0");
        return false;
    }

    let total = manifest.shards.len();
    let mut covered = 0;
    for shard_info in &manifest.shards {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        let holders = state.shared_state.model_registry.shard_holders(&shard_id);
        if holders.is_empty() {
            tracing::debug!(
                model = %model_name,
                shard = shard_info.index,
                "all_shards_available: no node in network holds this shard"
            );
            return false;
        }
        covered += 1;
    }

    tracing::info!(
        model = %model_name,
        shards = total,
        covered,
        num_layers = manifest.num_layers,
        "all_shards_available: all layers covered across network"
    );
    true
}

/// Extract an HTTP base URL from a peer's known addresses.
/// Multiaddrs look like `/ip4/127.0.0.1/udp/8800/quic-v1` — the peer runs
/// HTTP on the same port as QUIC.
/// Decode token IDs to text using the split model's tokenizer.
///
/// Uses BPE tokenizer byte decoding for proper UTF-8 handling (GPT-2 byte
/// encoding, SentencePiece byte fallbacks, etc).
pub(crate) fn decode_split_tokens(
    model: &crate::inference::split::SplitModel,
    token_ids: &[u32],
) -> String {
    if let Some(vocab) = model.vocab() {
        if let Some(tokenizer) = model.tokenizer() {
            let mut bytes = Vec::new();
            for &id in token_ids {
                if let Some(token_str) = vocab.get(id as usize) {
                    bytes.extend(tokenizer.decode_token(token_str));
                }
            }
            return String::from_utf8_lossy(&bytes).to_string();
        }
        // Fallback: raw vocab concatenation
        token_ids
            .iter()
            .filter_map(|&id| vocab.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    } else {
        String::new()
    }
}

fn peer_http_url(peer: &crate::types::PeerInfo) -> Option<String> {
    for addr in &peer.addresses {
        // Parse multiaddr: /ip4/<ip>/udp/<port>/quic-v1
        let parts: Vec<&str> = addr.split('/').collect();
        let mut ip = None;
        let mut port = None;
        for i in 0..parts.len() {
            if parts[i] == "ip4" && i + 1 < parts.len() {
                ip = Some(parts[i + 1]);
            }
            if parts[i] == "udp" && i + 1 < parts.len() {
                port = Some(parts[i + 1]);
            }
        }
        if let (Some(ip_str), Some(port_str)) = (ip, port) {
            // SECURITY: Block RFC-1918 private ranges, loopback, and link-local
            // to prevent SSRF. Only allow routable peer addresses.
            if is_private_ip(ip_str) {
                continue;
            }
            return Some(format!("http://{}:{}", ip_str, port_str));
        }
    }
    None
}

/// Check if an IP address string is in a private/reserved range (SSRF protection).
fn is_private_ip(ip: &str) -> bool {
    if let Ok(addr) = ip.parse::<std::net::Ipv4Addr>() {
        let octets = addr.octets();
        return octets[0] == 10                                              // 10.0.0.0/8
            || (octets[0] == 172 && (16..=31).contains(&octets[1]))         // 172.16.0.0/12
            || (octets[0] == 192 && octets[1] == 168)                       // 192.168.0.0/16
            || octets[0] == 127                                             // 127.0.0.0/8
            || (octets[0] == 169 && octets[1] == 254)                       // 169.254.0.0/16
            || octets[0] == 0; // 0.0.0.0/8
    }
    if let Ok(addr) = ip.parse::<std::net::Ipv6Addr>() {
        let segs = addr.segments();
        return addr.is_loopback()                                           // ::1
            || (segs[0] & 0xfe00) == 0xfc00                                 // fc00::/7 (ULA)
            || (segs[0] & 0xffc0) == 0xfe80                                 // fe80::/10 (link-local)
            || (segs[0] == 0 && segs[1] == 0 && segs[2] == 0
                && segs[3] == 0 && segs[4] == 0 && segs[5] == 0xffff); // ::ffff:0:0/96 (IPv4-mapped)
    }
    true // block unparseable addresses
}

/// Forward a chat completion request to a peer's HTTP API.
/// Lazily-initialized shared reqwest client for peer forwarding.
/// Avoids creating a new TLS + connection pool on every request.
static PEER_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();

fn get_peer_client() -> &'static reqwest::Client {
    PEER_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                INFERENCE_FORWARD_TIMEOUT_SECS,
            ))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

async fn forward_to_peer(
    peer_url: &str,
    req: &ChatCompletionRequest,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    let client = get_peer_client();
    let url = format!("{}/v1/chat/completions", peer_url);

    let peer_resp = client
        .post(&url)
        .header("x-swarm-forwarded", "true")
        .json(req)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, url = %url, "Failed to forward to peer");
            ApiError(crate::error::SwarmError::Internal(format!(
                "Peer forwarding failed: {e}"
            )))
        })?;

    if !peer_resp.status().is_success() {
        let status = peer_resp.status();
        let body = peer_resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "Peer returned error");
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Peer returned error status {status}"
        ))));
    }

    if stream {
        // Forward the SSE stream from the peer
        let byte_stream = peer_resp.bytes_stream();
        let body = axum::body::Body::from_stream(byte_stream);
        let response = axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("x-swarm-forwarded", "true")
            .body(body)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    } else {
        // Forward JSON response
        let body = peer_resp.text().await.unwrap_or_default();
        let response = axum::response::Response::builder()
            .header("content-type", "application/json")
            .header("x-swarm-forwarded", "true")
            .body(axum::body::Body::from(body))
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    }
}

/// Route inference through the InferenceRouter (non-streaming).
async fn router_inference(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
) -> Result<axum::response::Response, ApiError> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();

    let inference_req = InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: ModelId(req.model.clone()),
        messages,
        sampling_params: req.to_sampling_params(),
        stream: false,
        requester: NodeId([0u8; 32]), // Local API request
        priority: PriorityTier::Silver,
        created_at: chrono::Utc::now(),
        session_id: req.session_id.clone(),
        lora_adapter: req.lora_adapter.clone(),
    };

    router_tx
        .send(RouterCommand::Submit {
            request: inference_req,
            result_tx,
        })
        .await
        .map_err(|_| {
            ApiError(crate::error::SwarmError::Internal(
                "Router unavailable".into(),
            ))
        })?;

    let output = result_rx.await.map_err(|_| {
        ApiError(crate::error::SwarmError::Internal(
            "Router dropped the request".into(),
        ))
    })??;

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: req.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: output.content,
            },
            finish_reason: output.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
        session_id: output.session_id,
    };

    Ok(Json(response).into_response())
}

/// Route streaming inference through the InferenceRouter.
///
/// Submits the request via `StreamSubmit` so the pipeline executor sends decoded
/// tokens incrementally. Each token is forwarded as an SSE chunk, providing true
/// token-by-token streaming for distributed inference.
async fn router_inference_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    _state: &AppState,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
) -> Result<axum::response::Response, ApiError> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<StreamingTokenEvent>(64);

    let inference_req = InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: ModelId(req.model.clone()),
        messages,
        sampling_params: req.to_sampling_params(),
        stream: true,
        requester: NodeId([0u8; 32]),
        priority: PriorityTier::Silver,
        created_at: chrono::Utc::now(),
        session_id: req.session_id.clone(),
        lora_adapter: req.lora_adapter.clone(),
    };

    router_tx
        .send(RouterCommand::StreamSubmit {
            request: inference_req,
            result_tx,
            token_tx,
        })
        .await
        .map_err(|_| {
            ApiError(crate::error::SwarmError::Internal(
                "Router unavailable".into(),
            ))
        })?;

    let model_name = req.model.clone();
    let rid = request_id.clone();

    // Bridge the streaming token channel into SSE events
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    tokio::spawn(async move {
        // Send initial role delta
        let _ = sse_tx
            .send(StreamEvent::Delta {
                content: None,
                role: Some("assistant".into()),
                finish_reason: None,
            })
            .await;

        // Read tokens from the pipeline as they arrive
        let mut got_finish = false;
        while let Some(event) = token_rx.recv().await {
            if let Some(ref reason) = event.finish_reason {
                got_finish = true;
                if !event.text.is_empty() {
                    let _ = sse_tx
                        .send(StreamEvent::Delta {
                            content: Some(event.text),
                            role: None,
                            finish_reason: None,
                        })
                        .await;
                }
                let _ = sse_tx
                    .send(StreamEvent::Delta {
                        content: None,
                        role: None,
                        finish_reason: Some(reason.clone()),
                    })
                    .await;
                break;
            }
            if !event.text.is_empty()
                && sse_tx
                    .send(StreamEvent::Delta {
                        content: Some(event.text),
                        role: None,
                        finish_reason: None,
                    })
                    .await
                    .is_err()
            {
                break;
            }
        }

        // Fallback: if pipeline finished without sending streaming events
        // (e.g., local-only path or error), read the final result.
        if !got_finish {
            match result_rx.await {
                Ok(Ok(output)) => {
                    if !output.content.is_empty() {
                        let _ = sse_tx
                            .send(StreamEvent::Delta {
                                content: Some(output.content),
                                role: None,
                                finish_reason: None,
                            })
                            .await;
                    }
                    let _ = sse_tx
                        .send(StreamEvent::Delta {
                            content: None,
                            role: None,
                            finish_reason: Some(output.finish_reason),
                        })
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = sse_tx
                        .send(StreamEvent::Delta {
                            content: Some(format!("Error: {e}")),
                            role: None,
                            finish_reason: Some("stop".into()),
                        })
                        .await;
                }
                Err(_) => {
                    let _ = sse_tx
                        .send(StreamEvent::Delta {
                            content: None,
                            role: None,
                            finish_reason: Some("stop".into()),
                        })
                        .await;
                }
            }
        }
        let _ = sse_tx.send(StreamEvent::Done).await;
    });

    let stream =
        tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| match event {
            StreamEvent::Delta {
                content,
                role,
                finish_reason,
            } => {
                let chunk = ChatCompletionChunk {
                    id: rid.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta { role, content },
                        finish_reason,
                    }],
                };
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                Ok::<_, Infallible>(Event::default().data(json))
            }
            StreamEvent::Done => Ok(Event::default().data("[DONE]")),
        });

    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response())
}

async fn non_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    prompt: String,
    params: SamplingParams,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let mut executor = state.executor.lock().await;
    let (content, result) = executor.generate(&prompt, &params).map_err(ApiError)?;

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content,
            },
            finish_reason: result.finish_reason.as_str().into(),
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
        session_id: None, // Direct executor path doesn't use multi-turn sessions
    };

    Ok(Json(response))
}

/// Direct split-model generation (non-streaming).
///
/// Bypasses the distributed pipeline executor and generates directly from
/// the locally loaded SplitModel. Much faster for single-node inference
/// because it avoids per-token pipeline coordination overhead.
async fn split_non_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    prompt: String,
    params: SamplingParams,
) -> Result<axum::response::Response, ApiError> {
    use crate::inference::split::sample_token;

    let model_entry = state.shared_state.split_models.iter().next();
    let model_ref = match model_entry {
        Some(entry) => entry,
        None => return Err(ApiError(crate::error::SwarmError::NoModelLoaded)),
    };
    let entry = model_ref.value();
    let kv_store = state.shared_state.kv_cache_store.clone();
    let mut model = entry.model.lock().await;

    // Tokenize the prompt — forward() handles embedding internally
    let (input, prompt_tokens) = model.tokenize(&prompt)?;

    // First forward pass (prefill) — process entire prompt at once
    let logits = model.forward(&input, 0, &kv_store, &request_id)?;
    // logits shape: (1, seq_len, vocab) — take last token's logits
    let last_logits = logits
        .narrow(1, prompt_tokens - 1, 1)
        .map_err(|e| ApiError(crate::error::SwarmError::Internal(e.to_string())))?;
    let mut next_token = sample_token(&last_logits, params.temperature, params.top_p)?;

    let eos = model.eos_tokens().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut index_pos = prompt_tokens;

    for _ in 0..params.max_tokens {
        if eos.contains(&next_token) {
            break;
        }
        generated.push(next_token);

        // Create single-token tensor — forward() handles embedding
        let input = model.token_tensor(next_token)?;
        let logits = model.forward(&input, index_pos, &kv_store, &request_id)?;
        next_token = sample_token(&logits, params.temperature, params.top_p)?;
        index_pos += 1;
    }

    let finish_reason = if eos.contains(&next_token) {
        "stop"
    } else {
        "length"
    };

    // Decode tokens using BPE tokenizer for proper byte handling
    let content = decode_split_tokens(&model, &generated);

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content,
            },
            finish_reason: finish_reason.into(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: generated.len() as u32,
            total_tokens: (prompt_tokens + generated.len()) as u32,
        },
        session_id: None,
    };

    Ok(Json(response).into_response())
}

/// Direct split-model streaming generation.
///
/// Same fast path as split_non_stream_response but streams tokens via SSE.
async fn split_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    prompt: String,
    params: SamplingParams,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    use crate::inference::split::sample_token;

    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
    let request_id_inner = request_id.clone();

    tokio::spawn(async move {
        let request_id = request_id_inner;
        // Send initial role delta
        let _ = tx
            .send(StreamEvent::Delta {
                content: None,
                role: Some("assistant".into()),
                finish_reason: None,
            })
            .await;

        let model_entry = state.shared_state.split_models.iter().next();
        let model_ref = match model_entry {
            Some(entry) => entry,
            None => {
                let _ = tx.send(StreamEvent::Done).await;
                return;
            }
        };
        let entry = model_ref.value();
        let kv_store = state.shared_state.kv_cache_store.clone();
        let mut model = entry.model.lock().await;

        // Tokenize — forward() handles embedding internally
        let (input, prompt_tokens) = match model.tokenize(&prompt) {
            Ok(r) => r,
            Err(_) => {
                let _ = tx.send(StreamEvent::Done).await;
                return;
            }
        };

        // Prefill
        let logits = match model.forward(&input, 0, &kv_store, &request_id) {
            Ok(l) => l,
            Err(_) => {
                let _ = tx.send(StreamEvent::Done).await;
                return;
            }
        };
        let last_logits = match logits.narrow(1, prompt_tokens - 1, 1) {
            Ok(l) => l,
            Err(_) => {
                let _ = tx.send(StreamEvent::Done).await;
                return;
            }
        };
        let mut next_token = match sample_token(&last_logits, params.temperature, params.top_p) {
            Ok(t) => t,
            Err(_) => {
                let _ = tx.send(StreamEvent::Done).await;
                return;
            }
        };

        let eos = model.eos_tokens().to_vec();
        let mut index_pos = prompt_tokens;
        let mut finish = "length".to_string();

        for _ in 0..params.max_tokens {
            if eos.contains(&next_token) {
                finish = "stop".to_string();
                break;
            }

            // Decode and stream token
            let text = decode_split_tokens(&model, &[next_token]);

            if tx
                .send(StreamEvent::Delta {
                    content: Some(text),
                    role: None,
                    finish_reason: None,
                })
                .await
                .is_err()
            {
                return; // Client disconnected
            }

            // Create single-token tensor — forward() handles embedding
            let input = match model.token_tensor(next_token) {
                Ok(h) => h,
                Err(_) => break,
            };
            let logits = match model.forward(&input, index_pos, &kv_store, &request_id) {
                Ok(l) => l,
                Err(_) => break,
            };
            next_token = match sample_token(&logits, params.temperature, params.top_p) {
                Ok(t) => t,
                Err(_) => break,
            };
            index_pos += 1;
        }

        // Send finish
        let _ = tx
            .send(StreamEvent::Delta {
                content: None,
                role: None,
                finish_reason: Some(finish),
            })
            .await;
        let _ = tx.send(StreamEvent::Done).await;
    });

    // Convert channel to SSE stream (reuse existing stream mapping)
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |event| match event {
        StreamEvent::Delta {
            content,
            role,
            finish_reason,
        } => {
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta { role, content },
                    finish_reason,
                }],
            };
            Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
        }
        StreamEvent::Done => Ok(Event::default().data("[DONE]")),
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    )
}

async fn stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    prompt: String,
    params: SamplingParams,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    // Spawn generation in background
    tokio::spawn(async move {
        // Send initial role delta
        let _ = tx
            .send(StreamEvent::Delta {
                content: None,
                role: Some("assistant".into()),
                finish_reason: None,
            })
            .await;

        let mut executor = state.executor.lock().await;
        let result = executor.generate_stream(&prompt, &params, |token| {
            let send_result = tx.try_send(StreamEvent::Delta {
                content: Some(token.to_string()),
                role: None,
                finish_reason: None,
            });
            send_result.is_ok()
        });

        // Send finish reason
        let finish = match result {
            Ok(r) => r.finish_reason.as_str().to_string(),
            Err(_) => "error".to_string(),
        };
        let _ = tx
            .send(StreamEvent::Delta {
                content: None,
                role: None,
                finish_reason: Some(finish),
            })
            .await;
        let _ = tx.send(StreamEvent::Done).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |event| match event {
        StreamEvent::Delta {
            content,
            role,
            finish_reason,
        } => {
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta { role, content },
                    finish_reason,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        }
        StreamEvent::Done => Ok(Event::default().data("[DONE]")),
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
    )
}

enum StreamEvent {
    Delta {
        content: Option<String>,
        role: Option<String>,
        finish_reason: Option<String>,
    },
    Done,
}

/// GET /v1/models
///
/// Lists models usable for inference. A model is usable when all its layers
/// are covered by at least one node in the network — no single node needs
/// the full shard set. Models still propagating across the network (some
/// layers uncovered) are excluded here but visible in the admin dashboard.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let mut data = vec![];
    let mut seen = std::collections::HashSet::new();

    // Use cached model info (lock-free, no executor contention)
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        seen.insert(info.name.clone());
        let slug = info
            .name
            .to_lowercase()
            .replace(' ', "-")
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "");
        seen.insert(slug.clone());

        // Find the registry manifest for this model so we can use its canonical ID
        // and mark it as seen (prevents duplicates in section 2).
        let manifest = state
            .shared_state
            .model_registry
            .get_manifest(&crate::types::ModelId(slug.clone()))
            .or_else(|| {
                state
                    .shared_state
                    .model_registry
                    .get_manifest(&crate::types::ModelId(info.name.clone()))
            })
            .or_else(|| {
                // Match by manifest name field (auto-manage sets loaded_model_info.name
                // from manifest.name, but registry key is manifest.id)
                state
                    .shared_state
                    .model_registry
                    .models()
                    .into_iter()
                    .find(|m| m.name == info.name)
            });

        let model_id = if let Some(ref m) = manifest {
            seen.insert(m.id.0.clone());
            m.id.0.clone()
        } else {
            slug
        };

        data.push(ModelInfo {
            id: model_id,
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".into(),
        });
    }

    // Include models from the registry if all layers are covered network-wide
    for manifest in state.shared_state.model_registry.models() {
        let id = manifest.id.0.clone();
        if seen.contains(&id) {
            continue;
        }
        // Check that every shard has at least one holder somewhere in the network
        let all_covered = (0..manifest.shard_count).all(|idx| {
            let shard_id = crate::types::ShardId {
                model_id: manifest.id.clone(),
                index: idx,
            };
            !state
                .shared_state
                .model_registry
                .shard_holders(&shard_id)
                .is_empty()
        });
        if all_covered && manifest.num_layers > 0 {
            seen.insert(id.clone());
            data.push(ModelInfo {
                id,
                object: "model",
                created: manifest.publish_date.timestamp(),
                owned_by: "network".into(),
            });
        }
    }

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

// ---- Embeddings ----

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

fn default_encoding_format() -> String {
    "float".into()
}

/// POST /v1/embeddings — OpenAI-compatible embeddings endpoint.
///
/// Returns mean-pooled token embeddings from the loaded model's embedding layer.
/// This is a best-effort embedding — dedicated embedding models (e.g. text-embedding-3)
/// will produce better results for retrieval tasks.
pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Check if we have a loaded split model with tok_embeddings
    let model_entry = state.shared_state.split_models.iter().next();

    let model_ref = match model_entry {
        Some(entry) => entry,
        None => {
            return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
        }
    };
    let model_entry = model_ref.value();

    let inputs = match &req.input {
        EmbeddingInput::Single(s) => vec![s.clone()],
        EmbeddingInput::Batch(v) => v.clone(),
    };

    if inputs.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "Input must not be empty".into(),
        )));
    }

    // Lock the model to compute embeddings
    let model = model_entry.model.lock().await;

    let mut data = Vec::new();
    let mut total_tokens = 0usize;

    for (idx, text) in inputs.iter().enumerate() {
        // tokenize_and_embed returns (1, seq_len, hidden_dim) tensor
        let embedding = model.tokenize_and_embed(text).map_err(ApiError)?;

        // Count tokens from the embedding's seq dimension
        let seq_len = embedding.dim(1).map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Dim error: {e}"
            )))
        })?;
        total_tokens += seq_len;

        // Mean pool: average across the sequence dimension → (1, hidden_dim)
        let mean = embedding.mean(1).map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Mean pooling failed: {e}"
            )))
        })?;
        let mean = mean.squeeze(0).map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Squeeze failed: {e}"
            )))
        })?;

        let values: Vec<f32> = mean.to_vec1().map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Tensor conversion failed: {e}"
            )))
        })?;

        data.push(serde_json::json!({
            "object": "embedding",
            "index": idx,
            "embedding": values,
        }));
    }

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }
    })))
}

/// GET /v1/status — SwarmLLM extension endpoint
pub async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.shared_state.loaded_model_info.read().await;
    let local_model = info.is_some();
    let model_name = info.as_ref().map(|i| i.name.clone()).unwrap_or_default();
    drop(info);

    // Count network-available models from peers
    let mut network_models = Vec::new();
    for entry in state.shared_state.peer_registry.iter() {
        if let Some(ref cap) = entry.value().capability {
            for shard in &cap.hosted_shards {
                network_models.push(shard.model_id.0.clone());
            }
        }
    }
    network_models.sort();
    network_models.dedup();

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": local_model,
        "model_name": model_name,
        "network_models": network_models,
        "peers": state.shared_state.peer_registry.len(),
    }))
}
