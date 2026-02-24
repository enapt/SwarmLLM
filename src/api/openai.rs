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
use crate::inference::executor::build_chat_prompt;
use crate::inference::router::RouterCommand;
use crate::types::{ChatMessage, InferenceRequest, ModelId, NodeId, PriorityTier, SamplingParams};

// ---- Request types ----

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: 40,
            max_tokens: self.max_tokens,
            stop,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    let request_id = format!("swarm-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        messages = req.messages.len(),
        stream = req.stream,
        "Chat completion request"
    );

    // Route all requests through InferenceRouter when available
    if let Some(router_tx) = &state.router_tx {
        if req.stream {
            return router_inference_stream(router_tx.clone(), &state, &req, request_id, created)
                .await;
        } else {
            return router_inference(router_tx.clone(), &req, request_id, created).await;
        }
    }

    // Fallback: direct executor path (no router available — standalone mode)
    let prompt = build_chat_prompt(&req.messages);
    let params = req.to_sampling_params();
    let model_name = {
        let executor = state.executor.lock().await;
        if !executor.is_loaded() {
            return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
        }
        executor.model_name().to_string()
    };

    if req.stream {
        Ok(
            stream_response(state, request_id, created, model_name, prompt, params)
                .await
                .into_response(),
        )
    } else {
        Ok(
            non_stream_response(state, request_id, created, model_name, prompt, params)
                .await?
                .into_response(),
        )
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
/// Submits the request to the router for priority queueing and pipeline assembly,
/// then streams the result back via SSE.
async fn router_inference_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    _state: &AppState,
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
        stream: true,
        requester: NodeId([0u8; 32]),
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

    let model_name = req.model.clone();
    let rid = request_id.clone();

    // Spawn a task that waits for the router result and streams it as SSE chunks
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    tokio::spawn(async move {
        // Send initial role delta
        let _ = tx
            .send(StreamEvent::Delta {
                content: None,
                role: Some("assistant".into()),
                finish_reason: None,
            })
            .await;

        match result_rx.await {
            Ok(Ok(output)) => {
                // Stream the content word by word for SSE compatibility
                for word in output.content.split_inclusive(' ') {
                    if tx
                        .send(StreamEvent::Delta {
                            content: Some(word.to_string()),
                            role: None,
                            finish_reason: None,
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = tx
                    .send(StreamEvent::Delta {
                        content: None,
                        role: None,
                        finish_reason: Some(output.finish_reason),
                    })
                    .await;
            }
            Ok(Err(e)) => {
                let _ = tx
                    .send(StreamEvent::Delta {
                        content: Some(format!("Error: {e}")),
                        role: None,
                        finish_reason: Some("error".into()),
                    })
                    .await;
            }
            Err(_) => {
                let _ = tx
                    .send(StreamEvent::Delta {
                        content: None,
                        role: None,
                        finish_reason: Some("error".into()),
                    })
                    .await;
            }
        }
        let _ = tx.send(StreamEvent::Done).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |event| match event {
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
    let executor = state.executor.lock().await;
    let mut data = vec![];

    if executor.is_loaded() {
        data.push(ModelInfo {
            id: executor.model_name().to_string(),
            object: "model",
            created: 0,
            owned_by: "local".into(),
        });
    }

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

/// GET /v1/status — SwarmLLM extension endpoint
pub async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let executor = state.executor.lock().await;
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": executor.is_loaded(),
        "model_name": executor.model_name(),
    }))
}
