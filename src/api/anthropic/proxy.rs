use serde::Serialize;

use super::types::{AnthropicContent, AnthropicMessage, ContentBlock, SystemBlock, SystemContent};

/// Serializable proxy request (borrows from the original request).
/// Includes all Claude Code fields for full pass-through to Anthropic cloud.
#[derive(Serialize)]
pub(super) struct ProxyMessagesRequest<'a> {
    pub(super) model: &'a str,
    pub(super) max_tokens: u32,
    pub(super) messages: &'a [AnthropicMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) system: &'a Option<SystemContent>,
    pub(super) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stop_sequences: &'a Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tools: &'a Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_choice: &'a Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) metadata: &'a Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) thinking: &'a Option<serde_json::Value>,
    /// Forwarded verbatim — preserves `service_tier`, `container`, and any
    /// future Anthropic request fields the caller supplied that we don't
    /// explicitly model. Captured by `MessagesRequest::extras` via flatten.
    #[serde(flatten)]
    pub(super) extras: &'a std::collections::HashMap<String, serde_json::Value>,
}

// We need Serialize for AnthropicMessage/Content to proxy them
impl Serialize for AnthropicMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry("content", &self.content)?;
        map.end()
    }
}

impl Serialize for AnthropicContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            AnthropicContent::Text(s) => serializer.serialize_str(s),
            AnthropicContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl Serialize for ContentBlock {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ContentBlock::Text { text } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            ContentBlock::Image { source } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
                map.end()
            }
            ContentBlock::ToolUse { id, name, input } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_use")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("input", input)?;
                map.end()
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                if let Some(c) = content {
                    map.serialize_entry("content", c)?;
                }
                if let Some(e) = is_error {
                    map.serialize_entry("is_error", e)?;
                }
                map.end()
            }
            ContentBlock::Thinking { thinking } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "thinking")?;
                map.serialize_entry("thinking", thinking)?;
                map.end()
            }
            ContentBlock::RedactedThinking { data } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "redacted_thinking")?;
                map.serialize_entry("data", data)?;
                map.end()
            }
            ContentBlock::ServerToolUse { id, name, input } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "server_tool_use")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("input", input)?;
                map.end()
            }
            ContentBlock::WebSearchToolResult {
                tool_use_id,
                content,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "web_search_tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            ContentBlock::CodeExecutionToolResult {
                tool_use_id,
                content,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "code_execution_tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            ContentBlock::BashToolResult {
                tool_use_id,
                content,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "bash_tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            ContentBlock::TextEditorToolResult {
                tool_use_id,
                content,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "text_editor_tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                map.end()
            }
            ContentBlock::Document {
                source,
                title,
                citations,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "document")?;
                map.serialize_entry("source", source)?;
                if let Some(t) = title {
                    map.serialize_entry("title", t)?;
                }
                if let Some(c) = citations {
                    map.serialize_entry("citations", c)?;
                }
                map.end()
            }
            ContentBlock::SearchResult {
                source,
                title,
                citations,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "search_result")?;
                if let Some(s) = source {
                    map.serialize_entry("source", s)?;
                }
                if let Some(t) = title {
                    map.serialize_entry("title", t)?;
                }
                if let Some(c) = citations {
                    map.serialize_entry("citations", c)?;
                }
                map.end()
            }
        }
    }
}

impl Serialize for SystemContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            SystemContent::Text(s) => serializer.serialize_str(s),
            SystemContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl Serialize for SystemBlock {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", &self.block_type)?;
        if let Some(ref text) = self.text {
            map.serialize_entry("text", text)?;
        }
        if let Some(ref cc) = self.cache_control {
            map.serialize_entry("cache_control", cc)?;
        }
        map.end()
    }
}
