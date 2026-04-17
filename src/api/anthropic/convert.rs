//! Conversion helpers between Anthropic API wire types and internal types.

use crate::types::{ChatMessage, Role, SamplingParams};

use super::types::{AnthropicContent, ContentBlock, MessagesRequest, SystemContent};

/// Check if a request is a connectivity probe (Claude Code sends these to test the endpoint).
/// Narrowed to also check message content to avoid false-positiving on legitimate max_tokens=1 requests.
pub(super) fn is_connectivity_probe(req: &MessagesRequest) -> bool {
    if req.max_tokens != 1 || req.messages.len() != 1 || req.stream {
        return false;
    }
    // Check if the single message has very short content (probes are typically <20 chars)
    let content_len = match &req.messages[0].content {
        AnthropicContent::Text(s) => s.len(),
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => text.len(),
                _ => 100, // non-text content = not a probe
            })
            .sum(),
    };
    content_len <= 20
}

/// Convert Anthropic messages to internal ChatMessage format.
pub(super) fn to_internal_messages(req: &MessagesRequest) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // System prompt → System role
    if let Some(ref system) = req.system {
        let text = match system {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: Role::System,
                content: text,
                images: vec![],
            });
        }
    }

    for msg in &req.messages {
        let role = match msg.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        let text = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => {
                let mut texts = Vec::new();
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => texts.push(text.clone()),
                        ContentBlock::Image { .. } => {
                            // Image handling: VLM images handled via openai.rs path
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            // Include tool result text in conversation for local inference
                            if let Some(c) = content {
                                if let Some(s) = c.as_str() {
                                    texts.push(s.to_string());
                                } else if let Some(arr) = c.as_array() {
                                    for item in arr {
                                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                                            texts.push(t.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            // For local inference, represent tool calls as text
                            texts.push(format!("[Tool call: {name}({input})]"));
                        }
                        ContentBlock::Thinking { thinking } => {
                            // Include thinking in context for local models
                            texts.push(format!("<thinking>{thinking}</thinking>"));
                        }
                        ContentBlock::RedactedThinking { .. } => {}
                    }
                }
                texts.join("\n")
            }
        };
        messages.push(ChatMessage {
            role,
            content: text,
            images: vec![],
        });
    }

    messages
}

/// Convert Anthropic request to SamplingParams.
pub(super) fn to_sampling_params(req: &MessagesRequest) -> SamplingParams {
    crate::api::build_sampling_params(
        req.temperature.unwrap_or(1.0),
        req.top_p.unwrap_or(0.9),
        req.top_k.unwrap_or(crate::api::DEFAULT_TOP_K),
        req.max_tokens,
        req.stop_sequences.clone().unwrap_or_default(),
        0.0,
        0.0,
        false,
        0,
    )
}

/// Map internal finish reason to Anthropic stop_reason.
pub(super) fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// Resolve model name: strip `provider:` prefix if present.
pub(super) fn resolve_model(model: &str) -> &str {
    crate::api::strip_provider_prefix(model)
}
