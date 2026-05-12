//! Anthropic Messages API request/response types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemContent>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions for function calling (Claude Code sends these).
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tool choice: "auto", "any", "none", or {"type":"tool","name":"..."}
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Request metadata (e.g. user_id for abuse tracking).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Extended thinking configuration: {"type":"enabled","budget_tokens":N}
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    /// Catch-all for Anthropic request fields we don't explicitly model
    /// (`service_tier`, `container`, future extensions). `#[serde(flatten)]`
    /// preserves them verbatim so the proxy round-trip
    /// (mod.rs → ProxyMessagesRequest → upstream) forwards the original
    /// caller's knobs instead of silently dropping anything new.
    #[serde(flatten)]
    pub extras: std::collections::HashMap<String, serde_json::Value>,
}

fn default_temperature() -> Option<f32> {
    Some(1.0)
}

/// System content: either a plain string or an array of blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

impl SystemContent {
    /// Flatten to a single string; multi-block input joins with newlines.
    pub(super) fn to_plain_text(&self) -> String {
        match self {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
    /// Cache control hint (Anthropic prompt caching).
    #[serde(default)]
    pub cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Content field: either a plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// An Anthropic conversation content block.
///
/// The blocks SwarmLLM reasons about locally (text, image, tool_use,
/// tool_result, thinking, redacted_thinking) are modeled explicitly.
/// Server-tool variants and any future type are accepted as `Unknown`,
/// storing the raw JSON for verbatim proxy forwarding and treating them
/// as empty content on the local-inference path. Without this fallback,
/// echoing a Claude-server-tool conversation history back through
/// `/v1/messages` would fail deserialization at the `JsonBody` extractor.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: serde_json::Value },
    /// Tool use request from assistant (Claude Code tool calls).
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result from user (response to tool call).
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<serde_json::Value>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    /// Thinking block (extended thinking / chain-of-thought).
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    /// Redacted thinking block.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// Server-side tool use initiated by an Anthropic-hosted tool
    /// (web_search, code_execution, bash, text_editor, tool_search).
    /// Parsed so conversation echoes don't fail; treated as empty by
    /// the local-inference path.
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    /// Result of an Anthropic server tool (web search, code exec, bash,
    /// text editor, tool search). One variant per tool, collapsed here
    /// into a single shape — the `type` string is preserved via
    /// `ServerToolResultKind`.
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    #[serde(rename = "bash_tool_result")]
    BashToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    #[serde(rename = "text_editor_tool_result")]
    TextEditorToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    /// Document source for citations API (PDF / plain-text / custom).
    #[serde(rename = "document")]
    Document {
        source: serde_json::Value,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        citations: Option<serde_json::Value>,
    },
    /// Search result source for citations.
    #[serde(rename = "search_result")]
    SearchResult {
        #[serde(default)]
        source: Option<serde_json::Value>,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        citations: Option<serde_json::Value>,
    },
}

// ---- Response types ----

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

impl MessagesResponse {
    /// Build a simple text-only response.
    pub(super) fn text(
        id: String,
        model: String,
        text: String,
        stop_reason: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Self {
        Self::text_with_stop(
            id,
            model,
            text,
            stop_reason,
            None,
            input_tokens,
            output_tokens,
        )
    }

    /// Like `text` but also fills the `stop_sequence` field per Anthropic
    /// spec. When `stop_reason == "stop_sequence"` the matched custom stop
    /// string MUST be reported here; clients route on it for multi-stop
    /// agent scaffolds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn text_with_stop(
        id: String,
        model: String,
        text: String,
        stop_reason: &str,
        stop_sequence: Option<String>,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Self {
        Self {
            id,
            response_type: "message",
            role: "assistant",
            content: vec![ResponseContentBlock::Text { text }],
            model,
            stop_reason: Some(stop_reason.into()),
            stop_sequence,
            usage: AnthropicUsage {
                input_tokens,
                output_tokens,
                ..Default::default()
            },
        }
    }
}

/// Response content block — supports text, tool_use, and thinking.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Default, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens written to the prompt cache on this request (Anthropic extension).
    /// Only emitted when non-zero so existing clients / snapshot tests are
    /// unaffected.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_creation_input_tokens: u32,
    /// Tokens served from the prompt cache on this request. Same policy —
    /// elided when zero.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_read_input_tokens: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_content_block_serialization() {
        let text = ResponseContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_value(&text).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");

        let tool = ResponseContentBlock::ToolUse {
            id: "t1".into(),
            name: "Read".into(),
            input: serde_json::json!({"path": "/tmp"}),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "Read");

        let think = ResponseContentBlock::Thinking {
            thinking: "hmm".into(),
        };
        let json = serde_json::to_value(&think).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "hmm");
    }

    #[test]
    fn server_tool_content_blocks_parse() {
        // Echoing Claude-server-tool conversation history through our
        // /v1/messages endpoint used to fail here because these content-
        // block types were missing from the enum.
        let cases = [
            (
                r#"{"type":"server_tool_use","id":"t1","name":"web_search","input":{"query":"rust"}}"#,
                "ServerToolUse",
            ),
            (
                r#"{"type":"web_search_tool_result","tool_use_id":"t1","content":[{"type":"web_search_result","url":"https://example.com"}]}"#,
                "WebSearchToolResult",
            ),
            (
                r#"{"type":"code_execution_tool_result","tool_use_id":"t1","content":{"stdout":"hi"}}"#,
                "CodeExecutionToolResult",
            ),
            (
                r#"{"type":"bash_tool_result","tool_use_id":"t1","content":"ok"}"#,
                "BashToolResult",
            ),
            (
                r#"{"type":"text_editor_tool_result","tool_use_id":"t1","content":"patched"}"#,
                "TextEditorToolResult",
            ),
            (
                r#"{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0..."}}"#,
                "Document",
            ),
        ];
        for (json, label) in cases {
            let parsed: Result<ContentBlock, _> = serde_json::from_str(json);
            assert!(
                parsed.is_ok(),
                "expected {label} to parse but got: {:?}",
                parsed.err()
            );
        }
    }
}
