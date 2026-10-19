//! Conversion helpers between Anthropic API wire types and internal types.

use crate::error::SwarmError;
use crate::types::{
    ChatMessage, ImageData, Role, SamplingParams, ToolCallFunction, ToolCallTurn, ToolTurn,
};

use super::types::{AnthropicContent, ContentBlock, MessagesRequest};

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

/// An Anthropic image block's `source` as the image it carries, through the
/// OpenAI surface's decoder (one size cap, one format check). `base64` is a
/// data URI in all but name; `url` is refused like an OpenAI remote URL.
fn decode_image_source(source: &serde_json::Value) -> Result<ImageData, SwarmError> {
    let field = |name: &str| source.get(name).and_then(|v| v.as_str());
    match field("type") {
        Some("base64") => {
            let media_type = field("media_type").unwrap_or("image/png");
            let data = field("data").ok_or_else(|| {
                SwarmError::Validation("image source of type base64 has no `data`".into())
            })?;
            crate::api::openai::decode_image_url(&format!("data:{media_type};base64,{data}"))
        }
        Some("url") => crate::api::openai::decode_image_url(field("url").unwrap_or("")),
        other => Err(SwarmError::Validation(format!(
            "image source type {} is not supported here; send the picture as base64",
            other.unwrap_or("<none>")
        ))),
    }
}

/// Convert Anthropic messages to internal ChatMessage format.
///
/// A user's image blocks become images at their place in the text
/// (`vision::join_content`, the OpenAI surface's rule). They were dropped
/// here until #115 (2), under a comment saying the OpenAI path handled them.
pub(super) fn to_internal_messages(req: &MessagesRequest) -> Result<Vec<ChatMessage>, SwarmError> {
    let mut messages = Vec::new();

    // System prompt → System role
    if let Some(ref system) = req.system {
        let text = system.to_plain_text();
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: Role::System,
                content: text,
                images: vec![],
                tool: Default::default(),
            });
        }
    }

    for msg in &req.messages {
        let role = match msg.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        // A call and its answer travel as the structure a chat template renders
        // them from (#283), never as made-up text: `tool_use` becomes this
        // turn's `tool_calls`, and each `tool_result` a tool turn of its own,
        // ahead of whatever the user wrote beside it (the API puts results
        // first). The template's own call and response syntax — or, for one
        // that has none, the shape the tool description asked the model for —
        // is chosen where the prompt is built (`chat_template::tool_turns`).
        let mut calls = Vec::new();
        let mut images = Vec::new();
        let text = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => {
                // `None` marks where an image was (`vision::ContentPiece::Image`).
                let mut texts: Vec<Option<String>> = Vec::new();
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => texts.push(Some(text.clone())),
                        ContentBlock::Image { source } => {
                            images.push(decode_image_source(source)?);
                            texts.push(None);
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            let mut answer = Vec::new();
                            if let Some(c) = content {
                                if let Some(s) = c.as_str() {
                                    answer.push(s.to_string());
                                } else if let Some(arr) = c.as_array() {
                                    for item in arr {
                                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                                            answer.push(t.to_string());
                                        }
                                    }
                                }
                            }
                            messages.push(ChatMessage {
                                role: Role::Tool,
                                content: answer.join("\n"),
                                images: vec![],
                                tool: ToolTurn {
                                    tool_call_id: Some(tool_use_id.clone()),
                                    ..Default::default()
                                },
                            });
                        }
                        ContentBlock::ToolUse { id, name, input }
                            if matches!(role, Role::Assistant) =>
                        {
                            calls.push(ToolCallTurn {
                                id: id.clone(),
                                kind: "function".to_string(),
                                function: ToolCallFunction {
                                    name: name.clone(),
                                    arguments: input.clone(),
                                },
                            });
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            texts.push(Some(format!("[Tool call: {name}({input})]")));
                        }
                        ContentBlock::Thinking { thinking } => {
                            // Include thinking in context for local models
                            texts.push(Some(format!("<thinking>{thinking}</thinking>")));
                        }
                        ContentBlock::RedactedThinking { .. } => {}
                        // Server-tool & citation blocks: the local-inference path
                        // doesn't execute server tools, so flatten them to text
                        // hints that preserve conversational continuity. The
                        // proxy path never takes this branch (it re-serializes
                        // the raw request before forwarding).
                        ContentBlock::ServerToolUse { name, input, .. } => {
                            texts.push(Some(format!("[Server tool call: {name}({input})]")));
                        }
                        ContentBlock::WebSearchToolResult { content, .. }
                        | ContentBlock::CodeExecutionToolResult { content, .. }
                        | ContentBlock::BashToolResult { content, .. }
                        | ContentBlock::TextEditorToolResult { content, .. } => {
                            texts.push(Some(format!("[Tool result: {content}]")));
                        }
                        ContentBlock::Document { .. } | ContentBlock::SearchResult { .. } => {
                            // Citations sources — local models can't index into them
                            // meaningfully, skip.
                        }
                    }
                }
                crate::inference::vision::join_content(texts.iter().map(|t| match t {
                    Some(t) => crate::inference::vision::ContentPiece::Text(t),
                    None => crate::inference::vision::ContentPiece::Image,
                }))
            }
        };
        // A user turn that only carried results has said everything already.
        let only_results = matches!(&msg.content, AnthropicContent::Blocks(blocks)
            if !blocks.is_empty()
                && blocks.iter().all(|b| matches!(b, ContentBlock::ToolResult { .. })));
        if only_results {
            continue;
        }
        messages.push(ChatMessage {
            role,
            content: text,
            images,
            tool: ToolTurn {
                tool_calls: calls,
                ..Default::default()
            },
        });
    }

    Ok(messages)
}

/// Convert Anthropic request to SamplingParams.
pub(super) fn to_sampling_params(req: &MessagesRequest) -> SamplingParams {
    crate::api::build_sampling_params(
        req.temperature.unwrap_or(1.0),
        req.top_p.unwrap_or(0.9),
        req.top_k.unwrap_or(crate::api::DEFAULT_TOP_K),
        // Always explicit: the Messages API makes `max_tokens` a REQUIRED
        // field, and `anthropic::validate` has already rejected 0 and anything
        // over MAX_TOKENS_HARD_CAP by the time we get here.
        Some(req.max_tokens),
        req.stop_sequences.clone().unwrap_or_default(),
        0.0,
        0.0,
        false,
        0,
    )
}

/// Map internal finish reason to Anthropic stop_reason.
///
/// The catch-all is `end_turn`, and that is right for an unrecognised value —
/// but it must never be reached by a reply that did NOT finish, because
/// `end_turn` is one of only two values Anthropic defines as "the turn
/// completed naturally". Reporting a failure as the model choosing to stop is
/// the defect gotcha #433 was filed for, on the other surface.
///
/// So [`crate::inference::FINISH_REASON_INTERRUPTED`] gets an explicit arm.
/// Anthropic's vocabulary has no member for a turn cut short by the
/// infrastructure — the set is `end_turn`, `max_tokens`, `stop_sequence`,
/// `tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded` — and
/// inventing one is not open to us either: this surface shipped an undefined
/// `stop_reason: "error"` once and it was removed for exactly that reason
/// (gotcha #300). Of the defined values, `max_tokens` is the only one whose
/// contract is "this reply is incomplete because it was cut off", which is the
/// fact a client has to act on; every Anthropic client already handles it as
/// truncation and may continue from it. That is a translation into the target
/// vocabulary — the same thing this function does for `stop` — and it is the
/// nearest true statement available, not the nearest-sounding one.
///
/// `pause_turn` was considered and rejected: it means a server-tool loop hit
/// its iteration limit and the caller should resend to continue, so a client
/// obeying it would silently retry into the failure with nothing said.
pub(super) fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        crate::inference::FINISH_REASON_INTERRUPTED => "max_tokens",
        _ => "end_turn",
    }
}

/// Map internal finish reason + matched-stop-string to the Anthropic
/// `stop_reason`. When a user-provided custom stop sequence matched (rather
/// than EOS), the spec requires `stop_sequence` instead of `end_turn`.
pub(super) fn map_finish_reason_with_match(
    reason: &str,
    matched_stop: Option<&str>,
) -> &'static str {
    match (reason, matched_stop) {
        ("stop", Some(_)) => "stop_sequence",
        _ => map_finish_reason(reason),
    }
}

/// Resolve model name: strip `provider:` prefix if present, then expand the
/// bare family aliases `opus` / `sonnet` / `haiku` / `fable` to the current
/// full IDs.
///
/// Claude Code + the official SDK often send the full model ID (e.g.
/// `claude-opus-4-8`), but users / `.claude/settings.json` configs
/// sometimes use the shorthand — and Claude Code 2.1's default is now the
/// bare `sonnet` alias (Sonnet 5). Without alias expansion the router
/// at mod.rs:241 (`.starts_with("claude")`) drops the bare alias to the
/// non-Claude path, which then fails to find a provider. Keep the bump point
/// here — if/when Anthropic ships new aliases we only edit this table.
pub(super) fn resolve_model(model: &str) -> &str {
    let stripped = crate::api::strip_provider_prefix(model);
    match stripped {
        "opus" => "claude-opus-4-8",
        "sonnet" => "claude-sonnet-5",
        "haiku" => "claude-haiku-4-5",
        "fable" => "claude-fable-5",
        _ => stripped,
    }
}

/// The tools this request should be served with, translated into the shape a
/// chat template expects, or `None` when the model must not use any.
///
/// Anthropic describes a tool as `{"name", "description", "input_schema"}`;
/// every HuggingFace chat template is written against the OpenAI/transformers
/// shape, `{"type": "function", "function": {"name", "description",
/// "parameters"}}`, and dumps it verbatim — Qwen3 does `{{- tool | tojson }}`
/// straight into its `<tools>` block. Handing it Anthropic's shape would put a
/// tool with no `parameters` key in front of the model.
///
/// Tools deliberately do NOT become a system message here. See
/// `ChatCompletionRequest::tool_definitions` on the OpenAI surface for why that
/// choice belongs to `chat_template::build_prompt` instead; a local model would
/// otherwise be told about Anthropic's tools in a format its own template
/// already knows how to write.
///
/// `tool_choice: {"type": "none"}` returns `None`: not describing the tools is
/// the only way to hold a local model to it.
pub(super) fn tool_definitions_for_template(
    req: &MessagesRequest,
) -> Option<Vec<serde_json::Value>> {
    let tools = req.tools.as_ref()?;
    if tools.is_empty() || crate::api::tool_parse::tool_choice_forbids_tools(&req.tool_choice) {
        return None;
    }
    let converted: Vec<serde_json::Value> = tools
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let mut function = serde_json::Map::new();
            function.insert("name".into(), serde_json::Value::String(name.to_string()));
            if let Some(desc) = t.get("description") {
                function.insert("description".into(), desc.clone());
            }
            if let Some(schema) = t.get("input_schema") {
                function.insert("parameters".into(), schema.clone());
            }
            Some(serde_json::json!({"type": "function", "function": function}))
        })
        .collect();
    (!converted.is_empty()).then_some(converted)
}
