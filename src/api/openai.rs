use axum::extract::State;
use axum::response::sse::{Event, Sse};
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
use crate::types::{ChatMessage, InferenceRequest, ModelId, NodeId, PriorityTier, SamplingParams};

// ---- Request types ----

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    let request_id = format!("swarm-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    // Track requests made by this node
    if let Ok(mut stats) = state.shared_state.node_stats.try_write() {
        stats.requests_made += 1;
    }

    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        messages = req.messages.len(),
        stream = req.stream,
        "Chat completion request"
    );

    // Get model name from lock-free cache
    let model_name = {
        let info = state.shared_state.loaded_model_info.read().await;
        info.as_ref().map(|i| i.name.clone())
    };

    // If no model loaded locally, try to forward to a peer or auto-assemble.
    // The `x-swarm-forwarded` header prevents infinite forwarding loops between nodes.
    let is_forwarded = headers.get("x-swarm-forwarded").is_some();

    if model_name.is_none() {
        // Priority 1: Check if all shards exist across the pool for split/distributed
        // inference. This is preferred over forwarding because it uses the pipeline
        // scheduler to coordinate multi-node layer processing.
        if all_shards_available(&state, &req.model) {
            tracing::info!(
                request_id = %request_id,
                model = %req.model,
                stream = req.stream,
                "All shards available across pool — using distributed inference"
            );

            if let Some(router_tx) = &state.router_tx {
                if req.stream {
                    return router_inference_stream(
                        router_tx.clone(),
                        &state,
                        &req,
                        request_id,
                        created,
                    )
                    .await;
                } else {
                    return router_inference(router_tx.clone(), &req, request_id, created).await;
                }
            } else {
                return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
            }
        }

        // Priority 2: Forward to a peer that has the full model loaded.
        // The x-swarm-forwarded header prevents infinite forwarding loops.
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
    let prompt = chat_template::build_prompt(&req.messages, tmpl.as_deref(), &bos, &eos);
    let params = req.to_sampling_params();

    if req.stream {
        // Streaming: use direct executor path for real token-by-token SSE
        Ok(
            stream_response(state, request_id, created, model_name, prompt, params)
                .await
                .into_response(),
        )
    } else if let Some(router_tx) = &state.router_tx {
        // Non-streaming: route through InferenceRouter for priority queueing
        router_inference(router_tx.clone(), &req, request_id, created).await
    } else {
        // Fallback: direct executor path
        Ok(
            non_stream_response(state, request_id, created, model_name, prompt, params)
                .await?
                .into_response(),
        )
    }
}

/// Find a peer that has the full model loaded (not just some shards) and return its HTTP base URL.
/// This is a fallback for when distributed inference isn't available.
fn find_peer_with_model(state: &AppState, model: &str) -> Option<String> {
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            // Check if this peer advertises shards matching the requested model.
            // A peer with `loaded_model_info` broadcasts shard_id index=0 for
            // the full model — look for that as a signal of full model loaded.
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

/// Check if all shards for a model exist across the network (for split inference).
fn all_shards_available(state: &AppState, model_name: &str) -> bool {
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
                "all_shards_available: missing holders"
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
        "all_shards_available: all shards covered"
    );
    true
}

/// Extract an HTTP base URL from a peer's known addresses.
/// Multiaddrs look like `/ip4/127.0.0.1/udp/8800/quic-v1` — the peer runs
/// HTTP on the same port as QUIC.
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
        return addr.is_loopback()                                           // ::1
            || (addr.segments()[0] & 0xfe00) == 0xfc00; // fc00::/7
    }
    true // block unparseable addresses
}

/// Forward a chat completion request to a peer's HTTP API.
async fn forward_to_peer(
    peer_url: &str,
    req: &ChatCompletionRequest,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    let client = reqwest::Client::new();
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
            "Peer error ({status}): {body}"
        ))));
    }

    if stream {
        // Forward the SSE stream from the peer
        let byte_stream = peer_resp.bytes_stream();
        let body = axum::body::Body::from_stream(byte_stream);
        Ok(axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("x-swarm-forwarded", "true")
            .body(body)
            .unwrap()
            .into_response())
    } else {
        // Forward JSON response
        let body = peer_resp.text().await.unwrap_or_default();
        Ok(axum::response::Response::builder()
            .header("content-type", "application/json")
            .header("x-swarm-forwarded", "true")
            .body(axum::body::Body::from(body))
            .unwrap()
            .into_response())
    }
}

/// Route inference through the InferenceRouter (non-streaming).
async fn router_inference(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    request_id: String,
    created: i64,
) -> Result<axum::response::Response, ApiError> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();

    let inference_req = InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: ModelId(req.model.clone()),
        messages: req.messages.clone(),
        sampling_params: req.to_sampling_params(),
        stream: false,
        requester: NodeId([0u8; 32]), // Local API request
        priority: PriorityTier::Silver,
        created_at: chrono::Utc::now(),
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
    request_id: String,
    created: i64,
) -> Result<axum::response::Response, ApiError> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<StreamingTokenEvent>(64);

    let inference_req = InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: ModelId(req.model.clone()),
        messages: req.messages.clone(),
        sampling_params: req.to_sampling_params(),
        stream: true,
        requester: NodeId([0u8; 32]),
        priority: PriorityTier::Silver,
        created_at: chrono::Utc::now(),
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
                            finish_reason: Some("error".into()),
                        })
                        .await;
                }
                Err(_) => {
                    let _ = sse_tx
                        .send(StreamEvent::Delta {
                            content: None,
                            role: None,
                            finish_reason: Some("error".into()),
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

    Ok(Sse::new(stream).into_response())
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
    };

    Ok(Json(response))
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

    Sse::new(stream)
}

enum StreamEvent {
    Delta {
        content: Option<String>,
        role: Option<String>,
        finish_reason: Option<String>,
    },
    Done,
}

// ---- Completions (non-chat) ----

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
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
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

/// POST /v1/completions — OpenAI-compatible text completions endpoint.
pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let request_id = format!("swarm-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        "Completion request"
    );

    let stop = match &req.stop {
        Some(StopSequence::Single(s)) => vec![s.clone()],
        Some(StopSequence::Multiple(v)) => v.clone(),
        None => vec![],
    };

    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: 40,
        max_tokens: req.max_tokens,
        stop,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    let mut executor = state.executor.lock().await;
    if !executor.is_loaded() {
        return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
    }

    let (content, result) = executor.generate(&req.prompt, &params).map_err(ApiError)?;

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created,
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: content,
            finish_reason: result.finish_reason.as_str().into(),
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
    }))
}

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let mut data = vec![];
    let mut seen = std::collections::HashSet::new();

    // Use cached model info (lock-free, no executor contention)
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        seen.insert(info.name.clone());
        data.push(ModelInfo {
            id: info.name.clone(),
            object: "model",
            created: 0,
            owned_by: "local".into(),
        });
    }

    // Include models from the model registry (restored from DB or received via gossip)
    // Use the model ID (slug) as the primary identifier so inference routing works.
    for manifest in state.shared_state.model_registry.models() {
        let id = manifest.id.0.clone();
        if seen.insert(id.clone()) {
            data.push(ModelInfo {
                id,
                object: "model",
                created: manifest.publish_date.timestamp(),
                owned_by: "network".into(),
            });
        }
    }

    // Include models available from peers (so chat can offer them)
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            for shard in &cap.hosted_shards {
                let name = shard.model_id.0.clone();
                if seen.insert(name.clone()) {
                    data.push(ModelInfo {
                        id: name,
                        object: "model",
                        created: 0,
                        owned_by: "network".into(),
                    });
                }
            }
        }
    }

    Json(ModelListResponse {
        object: "list",
        data,
    })
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
