//! Bidirectional translation between OpenAI Responses (this module) and
//! Chat Completions (the existing `crate::api::openai::types`).
//!
//! Milestone scope:
//! - **M3**: plain-text input ↔ plain-text output.
//! - **M4**: function tools and tool_choice in both directions plus
//!   function_call / function_call_output input items and assistant
//!   tool_calls → function_call output items in the response.
//! - **M5**: cloud-proxy verbatim — no translation runs at all.
//! - **M6**: streaming token map.
//! - **V2 (v2 plan)**: multimodal input parts. `input_image{image_url}`
//!   maps to chat's `image_url` ContentPart (base64 data URIs pass
//!   through); `input_file{file_data}` inlines UTF-8 payloads as text;
//!   `input_image{file_id}`, `input_file{file_id}`, `input_audio`, and
//!   non-UTF-8 files are rejected with explicit errors pointing at the
//!   supported alternatives.

use std::collections::HashMap;

use crate::api::openai::responses::store::ResponsesRecord;
use crate::api::openai::responses::types::*;
use crate::api::openai::types::{
    ApiChatMessage, ChatCompletionRequest, ContentPart, FunctionCall as ChatFunctionCall,
    FunctionDefinition, ImageUrlRef, MessageContent, StopSequence, ToolCall as ChatToolCall,
    ToolDefinition,
};
use crate::error::SwarmError;
use crate::types::Role;

/// Default `max_tokens` when the caller did not set `max_output_tokens`.
/// Re-export of the shared constant from `super` so the chat-translation
/// path uses the same default as the local + Anthropic skeleton sites.
const DEFAULT_MAX_TOKENS: u32 = super::DEFAULT_MAX_OUTPUT_TOKENS;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.9;

// ============================================================================
// Request: Responses → Chat
// ============================================================================

/// Convert a Responses API request into a Chat Completions request the
/// existing `chat_completions` handler can consume.
///
/// `prior` is an optional previous-turn record from redb. When set, its
/// request input + response output are flattened into messages and
/// prepended to the current request's input — that's how
/// `previous_response_id` chaining (M8) feeds prior context to the local
/// chat path.
pub fn request_to_chat(
    req: &ResponsesRequest,
    prior: Option<&ResponsesRecord>,
) -> Result<ChatCompletionRequest, SwarmError> {
    let mut messages = Vec::new();

    // `instructions` becomes a leading system message.
    if let Some(instructions) = req.instructions.as_ref() {
        if !instructions.is_empty() {
            messages.push(ApiChatMessage {
                role: Role::System,
                content: MessageContent::Text(instructions.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                cache_control: None,
            });
        }
    }

    // M8: prior turn (if any) goes before the current input, in order:
    // prior request → prior response → current input.
    if let Some(prior_record) = prior {
        append_prior_turn(prior_record, &mut messages)?;
    }

    match &req.input {
        ResponsesInput::Text(s) => {
            messages.push(ApiChatMessage {
                role: Role::User,
                content: MessageContent::Text(s.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                cache_control: None,
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                push_input_item(item, &mut messages)?;
            }
        }
    }

    let stop = req.stop.as_ref().map(|s| match s {
        StopField::One(s) => StopSequence::Single(s.clone()),
        StopField::Many(v) => StopSequence::Multiple(v.clone()),
    });

    let tools = match req.tools.as_deref() {
        Some(t) if !t.is_empty() => Some(translate_tools(t)?),
        _ => None,
    };
    let tool_choice = req.tool_choice.as_ref().map(translate_tool_choice);

    Ok(ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        top_p: req.top_p.unwrap_or(DEFAULT_TOP_P),
        max_tokens: req.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        // M6 will replace the `false` here with a streaming pipeline.
        stream: false,
        stop,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        tools,
        tool_choice,
        logprobs: false,
        top_logprobs: None,
        response_format: None,
        session_id: None,
        lora_adapter: None,
        cache_control: None,
        extras: HashMap::new(),
    })
}

/// Translate Responses tool definitions to Chat tool definitions. Only
/// `function` tools translate; built-in tools are rejected upstream by the
/// handler, and any remaining `Raw` variant is an unmodeled type we can't
/// translate to the Chat format.
fn translate_tools(tools: &[ToolDef]) -> Result<Vec<ToolDefinition>, SwarmError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        match t {
            ToolDef::Typed(TypedToolDef::Function {
                name,
                description,
                parameters,
                strict: _,
                extras: _,
            }) => {
                // `strict` is an OpenAI-cloud-side schema-enforcement flag;
                // local inference doesn't honor it (the existing chat tool
                // definition has no slot for it).
                out.push(ToolDefinition {
                    tool_type: "function".into(),
                    function: FunctionDefinition {
                        name: name.clone(),
                        description: description.clone(),
                        parameters: parameters.clone(),
                    },
                });
            }
            ToolDef::Raw(value) => {
                let kind = value
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>");
                return Err(SwarmError::Validation(format!(
                    "Tool type `{kind}` is not supported by /v1/responses on \
                     this server. Only `function` tools are translated for \
                     local inference."
                )));
            }
        }
    }
    Ok(out)
}

/// Translate Responses `tool_choice` to the Chat-shaped value.
///
/// - `"auto" | "none" | "required"` → same string (Chat accepts these
///   directly).
/// - `{type:"function", name:"x"}` → `{type:"function", function:{name:"x"}}`
///   (Chat's nested form).
/// - Any other object form is forwarded as-is so future Responses-only
///   tool_choice shapes don't get silently dropped.
fn translate_tool_choice(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Mode(s) => serde_json::Value::String(s.clone()),
        ToolChoice::Object(obj) if obj.kind == "function" => serde_json::json!({
            "type": "function",
            "function": { "name": obj.name.clone().unwrap_or_default() },
        }),
        ToolChoice::Object(obj) => {
            // Pass-through for unmodeled object forms (allowed_tools,
            // mcp tool selectors, etc.). Local inference will likely
            // ignore them, but cloud-proxy paths benefit from the
            // verbatim shape.
            serde_json::to_value(obj).unwrap_or(serde_json::Value::String("auto".into()))
        }
    }
}

/// Flatten a previously-stored Responses turn into chat messages.
///
/// The record's `request.input` re-creates the inputs that preceded the
/// prior call; the record's `response.output` becomes the assistant's
/// reply (including any tool calls). Reasoning items and unknown output
/// types are dropped — they only matter on the cloud-proxy path, which
/// doesn't reach this helper.
fn append_prior_turn(
    prior: &ResponsesRecord,
    messages: &mut Vec<ApiChatMessage>,
) -> Result<(), SwarmError> {
    // Prior request input.
    match &prior.request.input {
        ResponsesInput::Text(s) => {
            messages.push(ApiChatMessage {
                role: Role::User,
                content: MessageContent::Text(s.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                cache_control: None,
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                push_input_item(item, messages)?;
            }
        }
    }

    // Prior response output — turn assistant messages and function_calls
    // back into chat messages.
    for item in &prior.response.output {
        match item {
            OutputItem::Typed(TypedOutputItem::Message(m)) => {
                let mut text = String::new();
                for part in &m.content {
                    if let OutputContentPart::Typed(TypedOutputContentPart::Text {
                        text: t, ..
                    }) = part
                    {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                }
                messages.push(ApiChatMessage {
                    role: Role::Assistant,
                    content: MessageContent::Text(text),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    cache_control: None,
                });
            }
            OutputItem::Typed(TypedOutputItem::FunctionCall(fc)) => {
                messages.push(ApiChatMessage {
                    role: Role::Assistant,
                    content: MessageContent::Text(String::new()),
                    tool_calls: Some(vec![ChatToolCall {
                        id: fc.call_id.clone(),
                        tool_type: "function".into(),
                        function: ChatFunctionCall {
                            name: fc.name.clone(),
                            arguments: fc.arguments.clone(),
                        },
                    }]),
                    tool_call_id: None,
                    name: None,
                    cache_control: None,
                });
            }
            OutputItem::Typed(TypedOutputItem::Reasoning(_)) => {
                // Local inference can't consume reasoning items — they
                // only matter for cloud-provider chaining.
            }
            OutputItem::Raw(_) => {
                // Unknown output item types (future / cloud-only) drop
                // silently; the current call will run without them.
            }
        }
    }

    Ok(())
}

fn push_input_item(item: &InputItem, messages: &mut Vec<ApiChatMessage>) -> Result<(), SwarmError> {
    match item {
        InputItem::Typed(TypedInputItem::Message(m)) => {
            let role = parse_role(&m.role)?;
            let content = match &m.content {
                InputMessageContent::Text(s) => MessageContent::Text(s.clone()),
                InputMessageContent::Parts(parts) => collect_content_parts(parts)?,
            };
            messages.push(ApiChatMessage {
                role,
                content,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                cache_control: None,
            });
        }
        InputItem::Typed(TypedInputItem::FunctionCall(fc)) => {
            // A prior assistant tool call being re-fed. Chat models it as
            // an assistant message with `content: null` and a single
            // `tool_calls` entry. Multiple consecutive Responses
            // function_call items become multiple assistant messages
            // each with one tool_call — chat models handle this either
            // way, and merging them would silently change the wire shape
            // for callers that care about parallelism semantics.
            messages.push(ApiChatMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: Some(vec![ChatToolCall {
                    id: fc.call_id.clone(),
                    tool_type: "function".into(),
                    function: ChatFunctionCall {
                        name: fc.name.clone(),
                        arguments: fc.arguments.clone(),
                    },
                }]),
                tool_call_id: None,
                name: None,
                cache_control: None,
            });
        }
        InputItem::Typed(TypedInputItem::FunctionCallOutput(out)) => {
            messages.push(ApiChatMessage {
                role: Role::Tool,
                content: MessageContent::Text(out.output.clone()),
                tool_calls: None,
                tool_call_id: Some(out.call_id.clone()),
                name: None,
                cache_control: None,
            });
        }
        InputItem::Typed(TypedInputItem::Reasoning(_)) => {
            // Drop silently. Local inference does not consume reasoning items
            // (gpt-5 / o-series chaining is a cloud-proxy concern, M5).
            tracing::debug!("Dropped reasoning input item on local inference path");
        }
        InputItem::Raw(value) => {
            let kind = value
                .get("type")
                .and_then(|x| x.as_str())
                .unwrap_or("<unknown>");
            return Err(SwarmError::Validation(format!(
                "Input item type `{kind}` is not supported by /v1/responses on this server. \
                 Only `message` items are translated for local inference."
            )));
        }
    }
    Ok(())
}

fn parse_role(role: &str) -> Result<Role, SwarmError> {
    match role {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "system" | "developer" => Ok(Role::System),
        "tool" => Ok(Role::Tool),
        other => Err(SwarmError::Validation(format!(
            "Unsupported message role `{other}` in /v1/responses input"
        ))),
    }
}

/// Cap on a single decoded `input_file.file_data` payload. Matches the
/// 20 MB image cap in `crate::api::openai::types`; the Responses plan
/// (V2) explicitly preserves the same cap for files.
const MAX_INPUT_FILE_BYTES: usize = 20 * 1024 * 1024;

/// V2: multimodal translation for message.content parts. Maps Responses
/// `input_text` → chat `text`, `input_image{image_url}` → chat
/// `image_url`, and `input_file{file_data}` → inlined-as-text (UTF-8
/// decodes only; binary formats are rejected with a clear 400).
///
/// Returns `MessageContent::Text` when every part collapsed to text, and
/// `MessageContent::Parts` when any non-text part survived. The chat
/// handler's `to_chat_message` already handles base64 image decode + size
/// cap + format validation on the chat-side ContentPart, so this
/// translator just forwards the data URI string as-is.
///
/// Rejections:
/// - `input_image { file_id }` (no uploads API on this server)
/// - `input_file { file_id }` (same)
/// - `input_file` with non-UTF-8 payload (PDF / docx / binary formats)
/// - `input_audio` (no Whisper plumbing yet)
/// - `InputContentPart::Raw` with an unknown type tag
fn collect_content_parts(parts: &[InputContentPart]) -> Result<MessageContent, SwarmError> {
    use base64::Engine;

    let mut chat_parts: Vec<ContentPart> = Vec::new();

    for part in parts {
        let typed = match part {
            InputContentPart::Typed(t) => t,
            InputContentPart::Raw(v) => {
                let kind = v
                    .get("type")
                    .and_then(|x| x.as_str())
                    .unwrap_or("<unknown>");
                return Err(SwarmError::Validation(format!(
                    "Input content part type `{kind}` is not supported on /v1/responses. \
                     Supported types: input_text, input_image (image_url), input_file (file_data)."
                )));
            }
        };

        match typed {
            TypedInputContentPart::Text { text, .. } => {
                if !text.is_empty() {
                    chat_parts.push(ContentPart::Text { text: text.clone() });
                }
            }
            TypedInputContentPart::Image {
                image_url, file_id, ..
            } => {
                if file_id.is_some() {
                    return Err(SwarmError::Validation(
                        "input_image file_id references are not supported on this server \
                         (no uploads API). Inline the image via image_url as a base64 \
                         data URI instead, e.g. `data:image/png;base64,<...>`."
                            .into(),
                    ));
                }
                let url = image_url.as_ref().ok_or_else(|| {
                    SwarmError::Validation(
                        "input_image requires either image_url (base64 data URI) or file_id".into(),
                    )
                })?;
                chat_parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrlRef { url: url.clone() },
                });
            }
            TypedInputContentPart::File {
                file_id,
                file_data,
                filename,
                ..
            } => {
                if file_id.is_some() {
                    return Err(SwarmError::Validation(
                        "input_file file_id references are not supported on this server \
                         (no uploads API). Inline the file contents via file_data (base64) \
                         instead."
                            .into(),
                    ));
                }
                let data_b64 = file_data.as_ref().ok_or_else(|| {
                    SwarmError::Validation("input_file requires either file_id or file_data".into())
                })?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data_b64)
                    .map_err(|e| {
                        SwarmError::Validation(format!(
                            "input_file.file_data base64 decode failed: {e}"
                        ))
                    })?;
                if bytes.len() > MAX_INPUT_FILE_BYTES {
                    return Err(SwarmError::Validation(format!(
                        "input_file.file_data decoded size {} exceeds the {}-byte cap",
                        bytes.len(),
                        MAX_INPUT_FILE_BYTES
                    )));
                }
                let text = std::str::from_utf8(&bytes).map_err(|_| {
                    SwarmError::Validation(format!(
                        "input_file `{}` is not UTF-8 text. Binary file formats (PDF, \
                         docx, images, etc.) are not yet supported on /v1/responses — \
                         either decode server-side and pass the extracted text as \
                         input_text, or send images via input_image with a base64 data URI.",
                        filename.as_deref().unwrap_or("<no filename>"),
                    ))
                })?;
                let with_header = match filename {
                    Some(name) if !name.is_empty() => format!("[File: {name}]\n{text}"),
                    _ => text.to_string(),
                };
                chat_parts.push(ContentPart::Text { text: with_header });
            }
            TypedInputContentPart::Audio { .. } => {
                return Err(SwarmError::Validation(
                    "input_audio is not supported on /v1/responses yet. Audio input \
                     requires a Whisper-class transcription model that SwarmLLM does \
                     not currently expose."
                        .into(),
                ));
            }
        }
    }

    // If every surviving part is text, collapse to MessageContent::Text so
    // the chat handler doesn't pay the array-form overhead for text-only
    // turns (and keeps the wire shape identical to what it would produce
    // before V2 for backwards compatibility).
    let all_text = chat_parts
        .iter()
        .all(|p| matches!(p, ContentPart::Text { .. }));
    if all_text {
        let joined = chat_parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(MessageContent::Text(joined))
    } else {
        Ok(MessageContent::Parts(chat_parts))
    }
}

// ============================================================================
// Response: Chat → Responses
// ============================================================================

/// Translate a Chat Completions response (parsed as `serde_json::Value`,
/// since `ChatCompletionResponse` is `Serialize`-only) into a
/// `ResponsesResponse`.
///
/// Output item ordering when the model both emitted text and called tools:
/// the message item comes first, followed by one `function_call` item per
/// chat `tool_call`. When `finish_reason` is `tool_calls` and `content` is
/// empty, no message item is emitted — just the function_call items.
/// `output_text` always reflects only the text content, never the
/// arguments JSON (matches OpenAI's wire format).
pub fn chat_response_to_responses(
    chat: &serde_json::Value,
    original: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> Result<ResponsesResponse, SwarmError> {
    let model = chat
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&original.model)
        .to_string();

    let choice = chat
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| SwarmError::Internal("Chat response has no choices".into()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| SwarmError::Internal("Chat response choice missing `message`".into()))?;

    let text = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap_or("stop");

    let status = match finish_reason {
        "stop" | "tool_calls" => ResponseStatus::Completed,
        "length" => ResponseStatus::Incomplete,
        "content_filter" | "error" => ResponseStatus::Failed,
        _ => ResponseStatus::Completed,
    };

    let incomplete_details = (status == ResponseStatus::Incomplete).then(|| IncompleteDetails {
        reason: "max_output_tokens".into(),
        extras: HashMap::new(),
    });

    let usage = chat
        .get("usage")
        .map(|u| ResponsesUsage {
            input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            input_tokens_details: None,
            output_tokens_details: None,
        })
        .unwrap_or_default();

    let mut output: Vec<OutputItem> = Vec::new();

    // Emit a message item only when there's actual text content. Empty
    // assistant content during a tool_calls finish should NOT show up as
    // an empty `output_text` item; OpenAI's spec keeps the output array
    // pure (function_calls only) in that case.
    if !text.is_empty() {
        let output_message = OutputMessageItem {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            role: "assistant".into(),
            status: Some("completed".into()),
            content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                text: text.clone(),
                annotations: Vec::new(),
                logprobs: None,
                extras: HashMap::new(),
            })],
            extras: HashMap::new(),
        };
        output.push(OutputItem::Typed(TypedOutputItem::Message(output_message)));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
        for tc in tool_calls {
            let call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| SwarmError::Internal("tool_call missing `id`".into()))?
                .to_string();
            let func = tc
                .get("function")
                .ok_or_else(|| SwarmError::Internal("tool_call missing `function`".into()))?;
            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| SwarmError::Internal("tool_call function missing `name`".into()))?
                .to_string();
            let arguments = func
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            output.push(OutputItem::Typed(TypedOutputItem::FunctionCall(
                FunctionCallItem {
                    call_id: call_id.clone(),
                    name,
                    arguments,
                    id: Some(format!("fc_{call_id}")),
                    status: Some("completed".into()),
                    extras: HashMap::new(),
                },
            )));
        }
    }

    // If neither text nor tool_calls produced an item, surface an empty
    // assistant message so the response shape stays valid.
    if output.is_empty() {
        output.push(OutputItem::Typed(TypedOutputItem::Message(
            OutputMessageItem {
                id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
                role: "assistant".into(),
                status: Some("completed".into()),
                content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                    text: String::new(),
                    annotations: Vec::new(),
                    logprobs: None,
                    extras: HashMap::new(),
                })],
                extras: HashMap::new(),
            },
        )));
    }

    Ok(ResponsesResponse {
        id: response_id.into(),
        object: "response".into(),
        created_at,
        status,
        model,
        output,
        output_text: Some(text),
        usage,
        error: None,
        incomplete_details,
        previous_response_id: original.previous_response_id.clone(),
        instructions: original.instructions.clone(),
        tools: original.tools.clone(),
        tool_choice: original.tool_choice.clone(),
        parallel_tool_calls: original.parallel_tool_calls,
        temperature: Some(original.temperature.unwrap_or(DEFAULT_TEMPERATURE)),
        top_p: Some(original.top_p.unwrap_or(DEFAULT_TOP_P)),
        max_output_tokens: Some(original.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        truncation: original.truncation.clone(),
        metadata: original.metadata.clone(),
        user: original.user.clone(),
        reasoning: original.reasoning.clone(),
        text: original.text.clone(),
        modalities: original.modalities.clone(),
        service_tier: original.service_tier.clone(),
        background: original.background,
        extras: HashMap::new(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req_text(input: &str) -> ResponsesRequest {
        ResponsesRequest {
            model: "test-model".into(),
            input: ResponsesInput::Text(input.into()),
            instructions: None,
            previous_response_id: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            seed: None,
            user: None,
            metadata: None,
            stream: None,
            store: None,
            background: None,
            parallel_tool_calls: None,
            truncation: None,
            service_tier: None,
            modalities: None,
            include: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            text: None,
            conversation: None,
            context_management: None,
            extras: HashMap::new(),
        }
    }

    #[test]
    fn text_input_becomes_user_message() {
        let req = req_text("Hello, world");
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.model, "test-model");
        assert_eq!(chat.messages.len(), 1);
        assert!(matches!(chat.messages[0].role, Role::User));
        match &chat.messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "Hello, world"),
            _ => panic!(),
        }
        assert!(!chat.stream);
    }

    #[test]
    fn instructions_become_leading_system_message() {
        let mut req = req_text("Hi");
        req.instructions = Some("You are helpful.".into());
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert!(matches!(chat.messages[0].role, Role::System));
        assert!(matches!(chat.messages[1].role, Role::User));
    }

    #[test]
    fn array_input_message_with_text_parts() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
            ]},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
        match &chat.messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "first\nsecond"),
            _ => panic!(),
        }
    }

    // ------------------------------------------------------------------
    // V2: multimodal input parts
    // ------------------------------------------------------------------

    #[test]
    fn input_image_with_base64_data_uri_becomes_chat_image_part() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="},
            ]},
        ])).unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
        let parts = match &chat.messages[0].content {
            MessageContent::Parts(p) => p.clone(),
            _ => panic!("expected Parts form for mixed text+image input"),
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "describe"));
        let url = match &parts[1] {
            ContentPart::ImageUrl { image_url } => image_url.url.clone(),
            _ => panic!("expected image_url part"),
        };
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn input_image_file_id_is_rejected_with_clear_message() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_image", "file_id": "file-123"},
            ]},
        ]))
        .unwrap();
        let err = request_to_chat(&req, None).unwrap_err().to_string();
        assert!(err.contains("file_id"), "error should name file_id: {err}");
        assert!(
            err.contains("image_url") || err.contains("base64"),
            "error should point to the alternative: {err}"
        );
    }

    #[test]
    fn input_audio_is_rejected() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "AQID", "format": "wav"}},
            ]},
        ]))
        .unwrap();
        let err = request_to_chat(&req, None).unwrap_err().to_string();
        assert!(
            err.contains("audio"),
            "expected audio-specific rejection: {err}"
        );
    }

    #[test]
    fn input_file_file_id_is_rejected() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_file", "file_id": "file-123"},
            ]},
        ]))
        .unwrap();
        let err = request_to_chat(&req, None).unwrap_err().to_string();
        assert!(
            err.contains("file_id") && err.contains("file_data"),
            "error should name both file_id and file_data: {err}"
        );
    }

    #[test]
    fn input_file_with_utf8_file_data_inlines_as_text_with_filename_header() {
        use base64::Engine;
        let payload = "line one\nline two";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "summarize"},
                {"type": "input_file", "file_data": b64, "filename": "notes.txt"},
            ]},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
        match &chat.messages[0].content {
            MessageContent::Text(s) => {
                assert!(
                    s.contains("summarize"),
                    "chat text should include prompt: {s}"
                );
                assert!(
                    s.contains("[File: notes.txt]"),
                    "chat text should include filename header: {s}"
                );
                assert!(
                    s.contains("line one") && s.contains("line two"),
                    "chat text should include file body: {s}"
                );
            }
            _ => panic!("all-text parts should collapse to MessageContent::Text"),
        }
    }

    #[test]
    fn input_file_binary_is_rejected() {
        use base64::Engine;
        // Invalid UTF-8 bytes (0x80 0xff are lone continuation/invalid).
        let bytes: &[u8] = &[0xff, 0xfe, 0x00, 0x01, 0x02];
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_file", "file_data": b64, "filename": "scan.pdf"},
            ]},
        ]))
        .unwrap();
        let err = request_to_chat(&req, None).unwrap_err().to_string();
        assert!(
            err.contains("scan.pdf"),
            "error should include filename: {err}"
        );
        assert!(
            err.contains("UTF-8") || err.contains("binary") || err.contains("Binary"),
            "error should flag the binary/UTF-8 issue: {err}"
        );
    }

    #[test]
    fn mixed_text_and_image_stays_as_parts_form() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what's in this?"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4A=="},
                {"type": "input_text", "text": "be terse"},
            ]},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        let parts = match &chat.messages[0].content {
            MessageContent::Parts(p) => p,
            _ => panic!("expected Parts"),
        };
        // Text parts preserved in order around the image.
        assert_eq!(parts.len(), 3);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "what's in this?"));
        assert!(matches!(&parts[1], ContentPart::ImageUrl { .. }));
        assert!(matches!(&parts[2], ContentPart::Text { text } if text == "be terse"));
    }

    #[test]
    fn text_only_parts_collapse_to_messagecontent_text() {
        // Regression: when every part is text, we should still collapse to
        // the string shape (matches v1 behavior and is cheaper on chat).
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"},
            ]},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        match &chat.messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "a\nb"),
            _ => panic!("expected MessageContent::Text, got Parts"),
        }
    }

    #[test]
    fn developer_role_collapses_to_system() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "developer", "content": "be terse"},
            {"type": "message", "role": "user", "content": "hi"},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert!(matches!(chat.messages[0].role, Role::System));
        assert!(matches!(chat.messages[1].role, Role::User));
    }

    #[test]
    fn function_call_input_translates_to_assistant_with_tool_calls() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}"},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert!(matches!(chat.messages[0].role, Role::Assistant));
        let tcs = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "c1");
        assert_eq!(tcs[0].tool_type, "function");
        assert_eq!(tcs[0].function.name, "f");
        assert_eq!(tcs[0].function.arguments, "{\"x\":1}");
    }

    #[test]
    fn function_call_output_input_translates_to_tool_message() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "function_call_output", "call_id": "c1", "output": "{\"result\":42}"},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert!(matches!(chat.messages[0].role, Role::Tool));
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("c1"));
        match &chat.messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "{\"result\":42}"),
            _ => panic!(),
        }
    }

    #[test]
    fn full_tool_calling_round_trip_request() {
        // user → assistant tool_call → tool result → user — the standard
        // multi-turn function calling shape Responses callers re-feed.
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "weather in NYC?"},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "{\"temp\":72}"},
            {"type": "message", "role": "user", "content": "and tomorrow?"},
        ])).unwrap();
        req.tools = Some(serde_json::from_value(json!([
            {"type": "function", "name": "get_weather", "description": "Look up weather", "parameters": {"type": "object"}, "strict": true},
        ])).unwrap());
        req.tool_choice = Some(ToolChoice::Mode("auto".into()));

        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 4);
        assert!(matches!(chat.messages[0].role, Role::User));
        assert!(matches!(chat.messages[1].role, Role::Assistant));
        assert!(chat.messages[1].tool_calls.is_some());
        assert!(matches!(chat.messages[2].role, Role::Tool));
        assert!(matches!(chat.messages[3].role, Role::User));

        let tools = chat.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");

        assert_eq!(chat.tool_choice, Some(json!("auto")));
    }

    #[test]
    fn tool_choice_function_object_nests_for_chat() {
        let mut req = req_text("hi");
        req.tools = Some(
            serde_json::from_value(json!([
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
            ]))
            .unwrap(),
        );
        req.tool_choice = Some(ToolChoice::Object(ToolChoiceObject {
            kind: "function".into(),
            name: Some("lookup".into()),
            extras: HashMap::new(),
        }));
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(
            chat.tool_choice,
            Some(json!({"type": "function", "function": {"name": "lookup"}}))
        );
    }

    #[test]
    fn unsupported_tool_type_is_rejected() {
        // Built-in tools are stopped at the handler before reaching
        // request_to_chat. But a synthetic Raw variant (unmodeled type)
        // landing here should also produce a clear validation error.
        let mut req = req_text("hi");
        req.tools = Some(vec![ToolDef::Raw(json!({"type": "future_kind"}))]);
        let err = request_to_chat(&req, None).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("future_kind")),
            _ => panic!(),
        }
    }

    #[test]
    fn chat_response_with_tool_calls_emits_function_call_items() {
        let chat = json!({
            "id": "chatcmpl_1",
            "model": "tinyllama",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "f1", "arguments": "{\"x\":1}"}},
                        {"id": "call_b", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        });
        let original = req_text("x");
        let resp = chat_response_to_responses(&chat, &original, "resp_x", 0).unwrap();
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 2);
        for (i, expected_call) in [("call_a", "f1", "{\"x\":1}"), ("call_b", "f2", "{}")]
            .iter()
            .enumerate()
        {
            match &resp.output[i] {
                OutputItem::Typed(TypedOutputItem::FunctionCall(fc)) => {
                    assert_eq!(fc.call_id, expected_call.0);
                    assert_eq!(fc.name, expected_call.1);
                    assert_eq!(fc.arguments, expected_call.2);
                    assert_eq!(
                        fc.id.as_deref(),
                        Some(format!("fc_{}", expected_call.0).as_str())
                    );
                }
                _ => panic!("expected function_call at index {i}"),
            }
        }
    }

    fn sample_record(prior_text_in: &str, prior_text_out: &str) -> ResponsesRecord {
        let mut req = req_text(prior_text_in);
        req.instructions = Some("You are helpful.".into());
        let resp = ResponsesResponse {
            id: "resp_prior".into(),
            object: "response".into(),
            created_at: 1_700_000_000,
            status: ResponseStatus::Completed,
            model: "test".into(),
            output: vec![OutputItem::Typed(TypedOutputItem::Message(
                OutputMessageItem {
                    id: "msg_1".into(),
                    role: "assistant".into(),
                    status: Some("completed".into()),
                    content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                        text: prior_text_out.into(),
                        annotations: Vec::new(),
                        logprobs: None,
                        extras: HashMap::new(),
                    })],
                    extras: HashMap::new(),
                },
            ))],
            output_text: Some(prior_text_out.into()),
            usage: ResponsesUsage::default(),
            error: None,
            incomplete_details: None,
            previous_response_id: None,
            instructions: req.instructions.clone(),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_output_tokens: Some(2048),
            truncation: None,
            metadata: None,
            user: None,
            reasoning: None,
            text: None,
            modalities: None,
            service_tier: None,
            background: None,
            extras: HashMap::new(),
        };
        ResponsesRecord {
            id: "resp_prior".into(),
            created_at: resp.created_at,
            expires_at: resp.created_at + 1_000_000,
            request: req,
            response: resp,
        }
    }

    #[test]
    fn prior_turn_prepends_user_and_assistant_messages() {
        let prior = sample_record("What's 2+2?", "4");
        let current = req_text("And 3+3?");
        let chat = request_to_chat(&current, Some(&prior)).unwrap();
        // System (from current.instructions — None here) +
        // (from prior: user "What's 2+2?" + assistant "4") +
        // current user "And 3+3?"
        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(chat.messages[0].role, Role::User));
        if let MessageContent::Text(s) = &chat.messages[0].content {
            assert_eq!(s, "What's 2+2?");
        } else {
            panic!()
        };
        assert!(matches!(chat.messages[1].role, Role::Assistant));
        if let MessageContent::Text(s) = &chat.messages[1].content {
            assert_eq!(s, "4");
        } else {
            panic!()
        };
        assert!(matches!(chat.messages[2].role, Role::User));
        if let MessageContent::Text(s) = &chat.messages[2].content {
            assert_eq!(s, "And 3+3?");
        } else {
            panic!()
        };
    }

    #[test]
    fn prior_turn_with_function_call_output() {
        // Prior: user → assistant function_call → tool result output.
        // Current: user follow-up. Full chain should land in chat messages.
        let mut prior = sample_record("weather?", "unused");
        prior.response.output = vec![OutputItem::Typed(TypedOutputItem::FunctionCall(
            FunctionCallItem {
                call_id: "c1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
                id: Some("fc_c1".into()),
                status: Some("completed".into()),
                extras: HashMap::new(),
            },
        ))];
        let current = req_text("and tomorrow?");
        let chat = request_to_chat(&current, Some(&prior)).unwrap();
        // prior user + prior assistant(tool_calls) + current user.
        assert_eq!(chat.messages.len(), 3);
        assert!(chat.messages[1].tool_calls.is_some());
        let tc = &chat.messages[1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.function.name, "get_weather");
    }

    #[test]
    fn chat_response_with_text_and_tool_calls_emits_both() {
        let chat = json!({
            "id": "x",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Looking that up.",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        });
        let original = req_text("x");
        let resp = chat_response_to_responses(&chat, &original, "r", 0).unwrap();
        assert_eq!(resp.output.len(), 2);
        assert!(matches!(
            &resp.output[0],
            OutputItem::Typed(TypedOutputItem::Message(_))
        ));
        assert!(matches!(
            &resp.output[1],
            OutputItem::Typed(TypedOutputItem::FunctionCall(_))
        ));
        assert_eq!(resp.output_text.as_deref(), Some("Looking that up."));
    }

    #[test]
    fn reasoning_input_dropped_silently() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "x"},
            {"type": "reasoning", "id": "rs", "summary": []},
        ]))
        .unwrap();
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    // ------------------------------------------------------------------
    // V7: reasoning item propagation
    // ------------------------------------------------------------------
    //
    // Two guarantees:
    //   (1) `include` round-trips through the cloud-proxy path because
    //       it's an explicit ResponsesRequest field — an o-series caller
    //       passing include:["reasoning.encrypted_content"] expects the
    //       upstream to echo those blocks back.
    //   (2) When a stored record from a prior turn contains reasoning
    //       output items (e.g. from an o-series cloud response), M8
    //       flatten MUST leave them in the stored record (so a second
    //       GET returns the same data) but MUST NOT re-inject them as
    //       chat messages on the next local turn — chat can't consume
    //       them and adding empty assistant stubs would confuse the
    //       prompt.

    #[test]
    fn include_field_round_trips_through_request_serialization() {
        // The cloud-proxy code path is `serde_json::to_value(&req)`
        // inside create_response. Verify `include` survives.
        let mut req = req_text("check");
        req.include = Some(vec![
            "reasoning.encrypted_content".into(),
            "reasoning.summary".into(),
        ]);
        let v = serde_json::to_value(&req).unwrap();
        let arr = v["include"].as_array().expect("include present");
        let strings: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
        assert_eq!(
            strings,
            vec!["reasoning.encrypted_content", "reasoning.summary"]
        );
    }

    #[test]
    fn prior_turn_with_reasoning_items_preserves_record_but_skips_chat_injection() {
        // Simulate a prior record that a cloud o-series call produced:
        // a reasoning item plus a final assistant message.
        let mut prior = sample_record("question", "the answer");
        prior.response.output = vec![
            OutputItem::Typed(TypedOutputItem::Reasoning(ReasoningItem {
                id: Some("rs_1".into()),
                summary: Some(vec![ReasoningSummaryPart::SummaryText {
                    text: "private chain of thought".into(),
                    extras: HashMap::new(),
                }]),
                encrypted_content: Some("opaque-cloud-blob-xyz".into()),
                status: Some("completed".into()),
                extras: HashMap::new(),
            })),
            OutputItem::Typed(TypedOutputItem::Message(OutputMessageItem {
                id: "msg_1".into(),
                role: "assistant".into(),
                status: Some("completed".into()),
                content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                    text: "the answer".into(),
                    annotations: Vec::new(),
                    logprobs: None,
                    extras: HashMap::new(),
                })],
                extras: HashMap::new(),
            })),
        ];

        // Sanity: the stored record still carries the reasoning item
        // (so a subsequent GET /v1/responses/:id sees encrypted_content
        // round-trip).
        let reasoning_item_count = prior
            .response
            .output
            .iter()
            .filter(|o| matches!(o, OutputItem::Typed(TypedOutputItem::Reasoning(_))))
            .count();
        assert_eq!(reasoning_item_count, 1);
        let roundtrip: ResponsesResponse =
            serde_json::from_value(serde_json::to_value(&prior.response).unwrap()).unwrap();
        let enc = roundtrip
            .output
            .iter()
            .find_map(|o| match o {
                OutputItem::Typed(TypedOutputItem::Reasoning(r)) => r.encrypted_content.clone(),
                _ => None,
            })
            .expect("encrypted_content round-trips through serde");
        assert_eq!(enc, "opaque-cloud-blob-xyz");

        // Now the local chat_completions flatten: reasoning must be
        // dropped, the assistant message must survive.
        let current = req_text("follow-up");
        let chat = request_to_chat(&current, Some(&prior)).unwrap();
        // prior_user + prior_assistant_message + current_user = 3.
        // If reasoning leaked in we'd see 4.
        assert_eq!(
            chat.messages.len(),
            3,
            "reasoning should not be re-injected as chat messages"
        );
        // The assistant message content must be the text output item's
        // content, NOT reasoning summary text.
        if let MessageContent::Text(s) = &chat.messages[1].content {
            assert_eq!(s, "the answer");
            assert!(
                !s.contains("private chain of thought"),
                "reasoning summary must not leak into chat history"
            );
        } else {
            panic!("assistant message should be text content");
        }
    }

    #[test]
    fn unknown_input_item_rejected_with_named_type() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "computer_call_output", "call_id": "x", "output": {}},
        ]))
        .unwrap();
        let err = request_to_chat(&req, None).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("computer_call_output")),
            _ => panic!(),
        }
    }

    #[test]
    fn sampling_params_propagate() {
        let mut req = req_text("hi");
        req.temperature = Some(0.3);
        req.top_p = Some(0.5);
        req.max_output_tokens = Some(100);
        req.frequency_penalty = Some(0.1);
        req.presence_penalty = Some(-0.2);
        req.stop = Some(StopField::Many(vec!["END".into(), "STOP".into()]));
        let chat = request_to_chat(&req, None).unwrap();
        assert_eq!(chat.temperature, 0.3);
        assert_eq!(chat.top_p, 0.5);
        assert_eq!(chat.max_tokens, 100);
        assert_eq!(chat.frequency_penalty, 0.1);
        assert_eq!(chat.presence_penalty, -0.2);
        match chat.stop {
            Some(StopSequence::Multiple(v)) => assert_eq!(v, vec!["END", "STOP"]),
            _ => panic!(),
        }
    }

    #[test]
    fn chat_response_to_responses_extracts_text_and_usage() {
        let chat = json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "tinyllama",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        });
        let original = req_text("hello");
        let resp =
            chat_response_to_responses(&chat, &original, "resp_test", 1_700_000_000).unwrap();
        assert_eq!(resp.id, "resp_test");
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.model, "tinyllama");
        assert_eq!(resp.output_text.as_deref(), Some("Hi there!"));
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::Typed(TypedOutputItem::Message(m)) => {
                assert_eq!(m.role, "assistant");
                match &m.content[0] {
                    OutputContentPart::Typed(TypedOutputContentPart::Text { text, .. }) => {
                        assert_eq!(text, "Hi there!");
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
        let u = &resp.usage;
        assert_eq!(u.input_tokens, 12);
        assert_eq!(u.output_tokens, 4);
        assert_eq!(u.total_tokens, 16);
    }

    #[test]
    fn chat_response_length_finish_reason_becomes_incomplete() {
        let chat = json!({
            "id": "x",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "trunc..."},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 100, "total_tokens": 101},
        });
        let original = req_text("hi");
        let resp = chat_response_to_responses(&chat, &original, "r", 0).unwrap();
        assert_eq!(resp.status, ResponseStatus::Incomplete);
        assert_eq!(
            resp.incomplete_details.as_ref().map(|d| d.reason.as_str()),
            Some("max_output_tokens")
        );
    }

    #[test]
    fn chat_response_content_filter_becomes_failed() {
        let chat = json!({
            "id": "x",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        });
        let original = req_text("hi");
        let resp = chat_response_to_responses(&chat, &original, "r", 0).unwrap();
        assert_eq!(resp.status, ResponseStatus::Failed);
    }
}
