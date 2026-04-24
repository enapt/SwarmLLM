//! Bidirectional translation between OpenAI Responses (this module) and
//! Chat Completions (the existing `crate::api::openai::types`).
//!
//! Milestone scope:
//! - **M3**: plain-text input ↔ plain-text output.
//! - **M4 (current)**: function tools and tool_choice in both directions
//!   plus function_call / function_call_output input items and assistant
//!   tool_calls → function_call output items in the response.
//! - **M5**: cloud-proxy verbatim — no translation runs at all.
//! - **M6**: streaming token map.

use std::collections::HashMap;

use crate::api::openai::responses::store::ResponsesRecord;
use crate::api::openai::responses::types::*;
use crate::api::openai::types::{
    ApiChatMessage, ChatCompletionRequest, FunctionCall as ChatFunctionCall, FunctionDefinition,
    MessageContent, StopSequence, ToolCall as ChatToolCall, ToolDefinition,
};
use crate::error::SwarmError;
use crate::types::Role;

/// Default `max_tokens` when the caller did not set `max_output_tokens`.
/// Matches `crate::api::openai::types::default_max_tokens` so behaviour
/// lines up between the two endpoints.
const DEFAULT_MAX_TOKENS: u32 = 2048;
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
                InputMessageContent::Parts(parts) => {
                    let text = collect_text_from_parts(parts);
                    MessageContent::Text(text)
                }
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

fn collect_text_from_parts(parts: &[InputContentPart]) -> String {
    let mut buf = String::new();
    for part in parts {
        if let InputContentPart::Typed(TypedInputContentPart::Text { text, .. }) = part {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(text);
        }
        // Image / file / audio parts: not yet translated for local
        // inference. Vision support exists in chat_completions; routing
        // multimodal Responses requests through it lands in a later
        // milestone.
    }
    buf
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
