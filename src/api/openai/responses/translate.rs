//! Bidirectional translation between OpenAI Responses (this module) and
//! Chat Completions (the existing `crate::api::openai::types`).
//!
//! Milestone scope:
//! - **M3 (current)**: plain-text input ↔ plain-text output. Tools and
//!   streaming explicitly returned as 501 by the handler (M4 / M6).
//! - **M4**: function tools and tool_choice in both directions.
//! - **M5**: cloud-proxy verbatim — no translation runs at all.
//! - **M6**: streaming token map.

use std::collections::HashMap;

use crate::api::openai::responses::types::*;
use crate::api::openai::types::{
    ApiChatMessage, ChatCompletionRequest, MessageContent, StopSequence,
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
/// M3 plain-text scope. Returns `Validation` errors for inputs M3 doesn't
/// translate yet (caller should emit a clear "not yet implemented" message
/// pointing to the milestone that adds support).
pub fn request_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, SwarmError> {
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
        // M4 wires tools/tool_choice. The handler rejects requests that
        // would land here with tools set, so leaving these as None is safe.
        tools: None,
        tool_choice: None,
        logprobs: false,
        top_logprobs: None,
        response_format: None,
        session_id: None,
        lora_adapter: None,
        cache_control: None,
        extras: HashMap::new(),
    })
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
        InputItem::Typed(TypedInputItem::FunctionCall(_))
        | InputItem::Typed(TypedInputItem::FunctionCallOutput(_)) => {
            return Err(SwarmError::Validation(
                "Function-tool input items are not yet supported by /v1/responses on this \
                 server (planned for M4). Use plain message items for now."
                    .into(),
            ));
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
/// Plain-text path only — `tool_calls` in the chat response are surfaced as
/// the message text (M4 will replace this with a `function_call` output
/// item).
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

    Ok(ResponsesResponse {
        id: response_id.into(),
        object: "response".into(),
        created_at,
        status,
        model,
        output: vec![OutputItem::Typed(TypedOutputItem::Message(output_message))],
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
        let chat = request_to_chat(&req).unwrap();
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
        let chat = request_to_chat(&req).unwrap();
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
        let chat = request_to_chat(&req).unwrap();
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
        let chat = request_to_chat(&req).unwrap();
        assert!(matches!(chat.messages[0].role, Role::System));
        assert!(matches!(chat.messages[1].role, Role::User));
    }

    #[test]
    fn function_call_input_items_rejected_for_now() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        ]))
        .unwrap();
        let err = request_to_chat(&req).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("M4")),
            _ => panic!("expected Validation"),
        }
    }

    #[test]
    fn reasoning_input_dropped_silently() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "x"},
            {"type": "reasoning", "id": "rs", "summary": []},
        ]))
        .unwrap();
        let chat = request_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn unknown_input_item_rejected_with_named_type() {
        let mut req = req_text("");
        req.input = serde_json::from_value(json!([
            {"type": "computer_call_output", "call_id": "x", "output": {}},
        ]))
        .unwrap();
        let err = request_to_chat(&req).unwrap_err();
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
        let chat = request_to_chat(&req).unwrap();
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
