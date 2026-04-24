//! Request, response, and streaming-event types for `/v1/responses`.
//!
//! Design notes:
//! - Every struct OpenAI is likely to extend in-place carries
//!   `#[serde(flatten)] extras: HashMap<String, serde_json::Value>` so the
//!   cloud-proxy path round-trips unknown fields verbatim. The Chat
//!   Completions equivalent (commit `0ecd38e`) caught this the hard way.
//! - Polymorphic "type"-tagged shapes (input items, content parts, tool
//!   defs, output items) use an `untagged` enum with two arms: `Typed(...)`
//!   for variants we model and translate, and `Raw(serde_json::Value)` as
//!   the fallback so unknown discriminants still round-trip. Without the
//!   `Raw` arm a single new tool type from OpenAI would 400 on parse.
//! - Numeric and option fields use `Option<T>` rather than defaults so we
//!   can tell "caller did not set this" from "caller set it to zero" — the
//!   cloud-proxy path needs to forward only what the caller actually sent.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Request
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOpts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextFormat>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationRef>,

    /// 2026 Q1 addition. Verbatim-forwarded for now; CRUD not implemented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<serde_json::Value>,

    /// Catch-all for fields OpenAI may add. Required for cloud-proxy
    /// forward compatibility (gpt-5, o-series ship new params constantly).
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

/// `input` is either a single text prompt or an array of structured items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

impl Default for ResponsesInput {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

// ---- Input items ----

/// A single item in array-form `input`. Known item types deserialize into
/// `Typed`; everything else (including future variants) round-trips as
/// `Raw(Value)` so the cloud-proxy path stays forward-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    Typed(TypedInputItem),
    Raw(serde_json::Value),
}

impl InputItem {
    /// Discriminator string — useful for handler-side validation
    /// (e.g. rejecting `mcp_approval_response` for local inference).
    pub fn type_str(&self) -> Option<&str> {
        match self {
            Self::Typed(t) => Some(t.type_str()),
            Self::Raw(v) => v.get("type").and_then(|x| x.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TypedInputItem {
    #[serde(rename = "message")]
    Message(InputMessageItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionCallOutputItem),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
}

impl TypedInputItem {
    pub fn type_str(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::FunctionCall(_) => "function_call",
            Self::FunctionCallOutput(_) => "function_call_output",
            Self::Reasoning(_) => "reasoning",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessageItem {
    pub role: String,
    pub content: InputMessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<InputContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContentPart {
    Typed(TypedInputContentPart),
    Raw(serde_json::Value),
}

impl InputContentPart {
    pub fn type_str(&self) -> Option<&str> {
        match self {
            Self::Typed(t) => Some(t.type_str()),
            Self::Raw(v) => v.get("type").and_then(|x| x.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TypedInputContentPart {
    #[serde(rename = "input_text")]
    Text {
        text: String,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "input_image")]
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "input_file")]
    File {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "input_audio")]
    Audio {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_audio: Option<serde_json::Value>,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
}

impl TypedInputContentPart {
    pub fn type_str(&self) -> &'static str {
        match self {
            Self::Text { .. } => "input_text",
            Self::Image { .. } => "input_image",
            Self::File { .. } => "input_file",
            Self::Audio { .. } => "input_audio",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallItem {
    pub call_id: String,
    pub name: String,
    /// JSON-encoded string (matches OpenAI's wire format).
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallOutputItem {
    pub call_id: String,
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<ReasoningSummaryPart>>,
    /// Opaque blob the o-series / gpt-5 chain uses to thread reasoning
    /// across calls. Must round-trip byte-for-byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReasoningSummaryPart {
    #[serde(rename = "summary_text")]
    SummaryText {
        text: String,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
}

// ---- Tool definitions ----

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolDef {
    Typed(TypedToolDef),
    Raw(serde_json::Value),
}

impl ToolDef {
    /// Discriminator string. Used by M2's built-in-tool rejection path.
    pub fn type_str(&self) -> Option<&str> {
        match self {
            Self::Typed(t) => Some(t.type_str()),
            Self::Raw(v) => v.get("type").and_then(|x| x.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TypedToolDef {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameters: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
}

impl TypedToolDef {
    pub fn type_str(&self) -> &'static str {
        match self {
            Self::Function { .. } => "function",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "auto" | "none" | "required"
    Mode(String),
    Object(ToolChoiceObject),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceObject {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

// ---- Reasoning / text format / conversation ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOpts {
    /// "minimal" | "low" | "medium" | "high"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFormat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormatSpec>,
    /// "low" | "medium" | "high"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TextFormatSpec {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConversationRef {
    Id(String),
    Object(ConversationRefObject),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRefObject {
    pub id: String,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

// ============================================================================
// Response
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    /// Always "response". Stored as `String` (not `&'static str`) so the
    /// type round-trips through `serde_json::from_value` for cached/proxy
    /// paths.
    pub object: String,
    pub created_at: i64,
    pub status: ResponseStatus,
    pub model: String,
    pub output: Vec<OutputItem>,

    /// Convenience concatenation of all `output_text` deltas. Optional
    /// because some terminal states (failed, cancelled) emit nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,

    pub usage: ResponsesUsage,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningOpts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Incomplete,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OutputItem {
    Typed(TypedOutputItem),
    Raw(serde_json::Value),
}

impl OutputItem {
    pub fn type_str(&self) -> Option<&str> {
        match self {
            Self::Typed(t) => Some(t.type_str()),
            Self::Raw(v) => v.get("type").and_then(|x| x.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TypedOutputItem {
    #[serde(rename = "message")]
    Message(OutputMessageItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
}

impl TypedOutputItem {
    pub fn type_str(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::FunctionCall(_) => "function_call",
            Self::Reasoning(_) => "reasoning",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessageItem {
    pub id: String,
    pub role: String,
    pub content: Vec<OutputContentPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OutputContentPart {
    Typed(TypedOutputContentPart),
    Raw(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TypedOutputContentPart {
    #[serde(rename = "output_text")]
    Text {
        text: String,
        /// Always emitted (per OpenAI wire format), even when empty.
        #[serde(default)]
        annotations: Vec<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<serde_json::Value>,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
    #[serde(rename = "refusal")]
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extras: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

impl ResponsesUsage {
    pub fn from_counts(input: u32, output: u32) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            input_tokens_details: None,
            output_tokens_details: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: String,
    #[serde(flatten)]
    pub extras: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Tests — serde round-trip
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper: parse value, re-serialize, parse again, assert equal.
    /// Catches any field that doesn't round-trip (silent drops, key reorders,
    /// type coercions).
    fn roundtrip<T: Serialize + for<'de> Deserialize<'de>>(input: serde_json::Value) -> T {
        let parsed: T = serde_json::from_value(input.clone()).expect("parse");
        let reserialized = serde_json::to_value(&parsed).expect("serialize");
        // Deep compare via re-parse: HashMap key order isn't stable, so we
        // can't compare the two `Value`s structurally. Re-parsing into T
        // and serializing again would loop; instead we assert that all keys
        // present in the original survive.
        assert_eq_json_subset(&input, &reserialized);
        parsed
    }

    /// Assert every field in `original` appears in `roundtripped` with the
    /// same value. (Roundtripped may have extra null fields that serde
    /// elided in the original.)
    fn assert_eq_json_subset(original: &serde_json::Value, roundtripped: &serde_json::Value) {
        match (original, roundtripped) {
            (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                for (k, v) in a {
                    let other = b
                        .get(k)
                        .unwrap_or_else(|| panic!("missing key `{k}` after roundtrip"));
                    assert_eq_json_subset(v, other);
                }
            }
            (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "array length changed: {a:?} vs {b:?}");
                for (av, bv) in a.iter().zip(b.iter()) {
                    assert_eq_json_subset(av, bv);
                }
            }
            (a, b) => assert_eq!(a, b, "value mismatch"),
        }
    }

    #[test]
    fn request_string_input() {
        let v = json!({
            "model": "gpt-5",
            "input": "Hello",
        });
        let req: ResponsesRequest = roundtrip(v);
        match req.input {
            ResponsesInput::Text(s) => assert_eq!(s, "Hello"),
            _ => panic!("expected Text input"),
        }
    }

    #[test]
    fn request_array_input_with_message() {
        let v = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "Hi"},
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Typed(TypedInputItem::Message(m)) => assert_eq!(m.role, "user"),
            other => panic!("expected typed message, got {other:?}"),
        }
    }

    #[test]
    fn request_array_input_mixed_message_function_call_reasoning() {
        let v = json!({
            "model": "o4-mini",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "search"},
                ]},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "OPAQUE_BLOB",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": "{\"q\":\"rust\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"results\":[]}"
                },
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].type_str(), Some("message"));
        assert_eq!(items[1].type_str(), Some("reasoning"));
        assert_eq!(items[2].type_str(), Some("function_call"));
        assert_eq!(items[3].type_str(), Some("function_call_output"));

        // Verify the reasoning item's encrypted_content survives byte-for-byte.
        if let InputItem::Typed(TypedInputItem::Reasoning(r)) = &items[1] {
            assert_eq!(r.encrypted_content.as_deref(), Some("OPAQUE_BLOB"));
        } else {
            panic!("expected typed reasoning");
        }
    }

    #[test]
    fn nested_input_content_parts() {
        let v = json!({
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "high"},
                        {"type": "input_file", "file_id": "file-123"},
                        {"type": "input_audio", "input_audio": {"data": "AQID", "format": "wav"}},
                    ],
                },
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            _ => panic!(),
        };
        let parts = match &items[0] {
            InputItem::Typed(TypedInputItem::Message(m)) => match &m.content {
                InputMessageContent::Parts(p) => p,
                _ => panic!("expected Parts"),
            },
            _ => panic!(),
        };
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].type_str(), Some("input_text"));
        assert_eq!(parts[1].type_str(), Some("input_image"));
        assert_eq!(parts[2].type_str(), Some("input_file"));
        assert_eq!(parts[3].type_str(), Some("input_audio"));
    }

    #[test]
    fn unknown_input_item_falls_back_to_raw() {
        // `mcp_approval_response` is a real input item we don't model
        // explicitly. It must still parse and round-trip via Raw.
        let v = json!({
            "model": "gpt-5",
            "input": [
                {
                    "type": "mcp_approval_response",
                    "approve": true,
                    "approval_request_id": "mcpr_1",
                },
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            _ => panic!(),
        };
        assert_eq!(items[0].type_str(), Some("mcp_approval_response"));
        assert!(matches!(items[0], InputItem::Raw(_)));
    }

    #[test]
    fn tools_function_plus_unknown() {
        let v = json!({
            "model": "gpt-5",
            "input": "x",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Look up current weather.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": true,
                },
                {"type": "web_search"},
                {"type": "file_search", "vector_store_ids": ["vs_1"]},
                {"type": "computer_use_preview", "display_width": 1024, "display_height": 768},
                {"type": "code_interpreter"},
                {"type": "image_generation"},
                {"type": "mcp", "server_label": "x", "server_url": "https://example.com"},
                {"type": "custom", "name": "lark_grammar", "format": {"type": "grammar"}},
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 8);
        assert_eq!(tools[0].type_str(), Some("function"));
        assert_eq!(tools[1].type_str(), Some("web_search"));
        assert_eq!(tools[2].type_str(), Some("file_search"));
        assert_eq!(tools[3].type_str(), Some("computer_use_preview"));
        assert_eq!(tools[4].type_str(), Some("code_interpreter"));
        assert_eq!(tools[5].type_str(), Some("image_generation"));
        assert_eq!(tools[6].type_str(), Some("mcp"));
        assert_eq!(tools[7].type_str(), Some("custom"));
        // Function deserialized as Typed; everything else as Raw.
        assert!(matches!(tools[0], ToolDef::Typed(_)));
        for t in &tools[1..] {
            assert!(matches!(t, ToolDef::Raw(_)), "expected Raw, got {t:?}");
        }
    }

    #[test]
    fn tool_choice_string_and_object() {
        for (input, expect_mode) in [
            (json!("auto"), Some("auto")),
            (json!("none"), Some("none")),
            (json!("required"), Some("required")),
        ] {
            let parsed: ToolChoice = serde_json::from_value(input.clone()).unwrap();
            match parsed {
                ToolChoice::Mode(m) => assert_eq!(m, expect_mode.unwrap()),
                _ => panic!("expected Mode for {input}"),
            }
        }
        let obj = json!({"type": "function", "name": "lookup"});
        let parsed: ToolChoice = serde_json::from_value(obj.clone()).unwrap();
        match parsed {
            ToolChoice::Object(o) => {
                assert_eq!(o.kind, "function");
                assert_eq!(o.name.as_deref(), Some("lookup"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reasoning_text_and_extras() {
        // ReasoningOpts must preserve `effort:"minimal"` (Q1 2026 addition)
        // and any future field via extras.
        let v = json!({
            "model": "o4-mini",
            "input": "x",
            "reasoning": {
                "effort": "minimal",
                "summary": "auto",
                "future_field": "preserved",
            },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Result",
                    "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "strict": true,
                },
                "verbosity": "high",
            },
        });
        let req: ResponsesRequest = roundtrip(v);
        let r = req.reasoning.as_ref().unwrap();
        assert_eq!(r.effort.as_deref(), Some("minimal"));
        assert!(r.extras.contains_key("future_field"));

        let t = req.text.as_ref().unwrap();
        assert_eq!(t.verbosity.as_deref(), Some("high"));
        match t.format.as_ref().unwrap() {
            TextFormatSpec::JsonSchema { name, strict, .. } => {
                assert_eq!(name, "Result");
                assert_eq!(*strict, Some(true));
            }
            _ => panic!("expected JsonSchema"),
        }
    }

    #[test]
    fn top_level_extras_preserved() {
        // SDKs that pass through `service_tier`, `seed`, and any unknown
        // future param must round-trip. The cloud-proxy path depends on it.
        let v = json!({
            "model": "gpt-5",
            "input": "x",
            "service_tier": "priority",
            "seed": 42,
            "future_unknown_param": {"nested": [1, 2, 3]},
        });
        let req: ResponsesRequest = roundtrip(v);
        assert_eq!(req.service_tier.as_deref(), Some("priority"));
        assert_eq!(req.seed, Some(42));
        assert!(req.extras.contains_key("future_unknown_param"));
    }

    #[test]
    fn stop_field_string_and_array() {
        let s = json!({"model": "x", "input": "y", "stop": "END"});
        let req: ResponsesRequest = serde_json::from_value(s).unwrap();
        assert!(matches!(req.stop, Some(StopField::One(ref s)) if s == "END"));

        let a = json!({"model": "x", "input": "y", "stop": ["A", "B"]});
        let req: ResponsesRequest = serde_json::from_value(a).unwrap();
        assert!(
            matches!(req.stop, Some(StopField::Many(ref v)) if v == &vec!["A".to_string(), "B".into()])
        );
    }

    #[test]
    fn conversation_ref_string_and_object() {
        let s = json!({"model": "x", "input": "y", "conversation": "conv_123"});
        let req: ResponsesRequest = serde_json::from_value(s).unwrap();
        assert!(matches!(req.conversation, Some(ConversationRef::Id(ref s)) if s == "conv_123"));

        let o = json!({"model": "x", "input": "y", "conversation": {"id": "conv_456"}});
        let req: ResponsesRequest = serde_json::from_value(o).unwrap();
        assert!(
            matches!(req.conversation, Some(ConversationRef::Object(ref o)) if o.id == "conv_456")
        );
    }

    #[test]
    fn numeric_edge_cases() {
        let v = json!({
            "model": "x",
            "input": "y",
            "max_output_tokens": u32::MAX,
            "seed": u64::MAX,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": -2.0,
            "presence_penalty": 2.0,
        });
        let req: ResponsesRequest = roundtrip(v);
        assert_eq!(req.max_output_tokens, Some(u32::MAX));
        assert_eq!(req.seed, Some(u64::MAX));
        assert_eq!(req.temperature, Some(0.0));
        assert_eq!(req.top_p, Some(1.0));
        assert_eq!(req.frequency_penalty, Some(-2.0));
        assert_eq!(req.presence_penalty, Some(2.0));
    }

    #[test]
    fn response_full_shape() {
        let v = json!({
            "id": "resp_abc",
            "object": "response",
            "created_at": 1_700_000_000,
            "status": "completed",
            "model": "gpt-5",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "I considered..."}],
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Hello!", "annotations": []},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": "{\"q\":\"x\"}",
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
            "metadata": {"trace_id": "abc"},
        });
        let resp: ResponsesResponse = roundtrip(v);
        assert_eq!(resp.id, "resp_abc");
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 3);
        assert_eq!(resp.output[0].type_str(), Some("reasoning"));
        assert_eq!(resp.output[1].type_str(), Some("message"));
        assert_eq!(resp.output[2].type_str(), Some("function_call"));
        let u = &resp.usage;
        assert_eq!(u.total_tokens, 30);
        assert_eq!(
            u.output_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            Some(5)
        );
    }

    #[test]
    fn response_status_serialization_uses_snake_case() {
        for (status, expected) in [
            (ResponseStatus::Queued, "queued"),
            (ResponseStatus::InProgress, "in_progress"),
            (ResponseStatus::Completed, "completed"),
            (ResponseStatus::Failed, "failed"),
            (ResponseStatus::Incomplete, "incomplete"),
            (ResponseStatus::Cancelled, "cancelled"),
        ] {
            let s = serde_json::to_value(status).unwrap();
            assert_eq!(s, json!(expected), "status {status:?}");
            let back: ResponseStatus = serde_json::from_value(s).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn output_unknown_item_falls_back_to_raw() {
        // A future / cloud-only output type like `web_search_call` must
        // round-trip even though we don't model it.
        let v = json!({
            "id": "resp_x",
            "object": "response",
            "created_at": 1_700_000_000,
            "status": "completed",
            "model": "gpt-5",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        });
        let resp: ResponsesResponse = roundtrip(v);
        assert_eq!(resp.output[0].type_str(), Some("web_search_call"));
        assert!(matches!(resp.output[0], OutputItem::Raw(_)));
    }

    #[test]
    fn function_tool_strict_and_extras_preserved() {
        let v = json!({
            "model": "gpt-5",
            "input": "x",
            "tools": [
                {
                    "type": "function",
                    "name": "f",
                    "parameters": {"type": "object"},
                    "strict": false,
                    "future_field": 99,
                }
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let tools = req.tools.unwrap();
        match &tools[0] {
            ToolDef::Typed(TypedToolDef::Function {
                name,
                strict,
                extras,
                ..
            }) => {
                assert_eq!(name, "f");
                assert_eq!(*strict, Some(false));
                assert!(extras.contains_key("future_field"));
            }
            _ => panic!("expected typed function"),
        }
    }

    #[test]
    fn message_content_string_form() {
        // The spec allows message content as a plain string (not an array
        // of parts). Cover the untagged InputMessageContent fallback.
        let v = json!({
            "model": "x",
            "input": [
                {"type": "message", "role": "user", "content": "raw string content"}
            ],
        });
        let req: ResponsesRequest = roundtrip(v);
        let items = match req.input {
            ResponsesInput::Items(i) => i,
            _ => panic!(),
        };
        match &items[0] {
            InputItem::Typed(TypedInputItem::Message(m)) => match &m.content {
                InputMessageContent::Text(s) => assert_eq!(s, "raw string content"),
                InputMessageContent::Parts(_) => panic!("expected Text content"),
            },
            _ => panic!(),
        }
    }
}
