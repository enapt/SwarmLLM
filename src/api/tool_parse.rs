//! Extracting tool calls from a local model's raw text output.
//!
//! A cloud provider returns tool calls as structured fields. A local GGUF just
//! emits text, so somebody has to parse it — and until now nobody did: we
//! injected a system prompt asking for `{"tool_calls": [...]}` (see
//! `openai::types::format_tool_system_prompt`), the model complied, and the JSON
//! was handed back to the client verbatim as assistant content with
//! `finish_reason: "length"`. Any standard OpenAI or Anthropic client saw a
//! model that ignored its tools.
//!
//! ## Approach
//!
//! This mirrors llama.cpp's design: per-family native formats, plus a *generic*
//! fallback for a model whose template we don't recognise, where we ask for a
//! known shape and parse that back. The generic path is the one we prompt for,
//! so it is tried first; the native formats exist because an instruction-tuned
//! model will often emit its *trained* format regardless of what the system
//! prompt asked for.
//!
//! Formats are tried in order, each independent:
//!
//! 1. **Generic** — `{"tool_calls": [{"function": {"name", "arguments"}}]}`.
//!    What `format_tool_system_prompt` requests.
//! 2. **Hermes / Qwen-Instruct** — `<tool_call>{"name", "arguments"}</tool_call>`,
//!    one block per call. The most common native format in circulating GGUFs.
//! 3. **Mistral** — `[TOOL_CALLS][{"name", "arguments"}]`.
//! 4. **Llama 3.x** — a bare `{"name", "parameters"}` object, optionally behind
//!    a `<|python_tag|>` marker.
//!
//! Adding a family means adding one `try_*` function and one line in
//! [`parse_tool_calls`]. Deliberately NOT a trait or registry: four small
//! functions with a fixed order are easier to read and to reason about than an
//! abstraction over four things.
//!
//! ## What this does not do
//!
//! No attempt is made to repair truncated JSON. A generation that hits
//! `max_tokens` mid-object is reported as text, not as a silently-wrong tool
//! call — inventing arguments a user never approved would be worse than
//! surfacing the raw output. (The report that prompted this work included
//! exactly such a truncated response, `finish_reason: "length"`.)

use serde_json::Value;

/// Build the system-prompt text that tells a local model how to call a tool.
///
/// Shared by both API layers deliberately. The OpenAI path had its own copy and
/// the Anthropic path had none — so `/v1/messages` never told a local model its
/// tools existed at all, and the model would answer "I'm unable to access
/// external tools" (external report 2026-07-25). Sharing one formatter also
/// keeps the requested shape identical to what [`parse_tool_calls`] looks for
/// first; two wordings would mean two formats to parse.
///
/// `tools` is `(name, description, schema)` so each endpoint can map its own
/// wire shape — OpenAI nests under `function`, Anthropic uses `input_schema`.
pub fn format_tool_prompt(tools: &[(String, Option<String>, Option<String>)]) -> String {
    let mut prompt = String::from(
        "You have access to the following tools. To call a tool, respond with a JSON object \
         in the following format:\n\
         {\"tool_calls\": [{\"id\": \"call_<unique_id>\", \"type\": \"function\", \
         \"function\": {\"name\": \"<function_name>\", \"arguments\": \"<json_args>\"}}]}\n\n\
         Available tools:\n",
    );
    for (name, description, schema) in tools {
        prompt.push_str(&format!("- {name}"));
        if let Some(desc) = description {
            prompt.push_str(&format!(": {desc}"));
        }
        prompt.push('\n');
        if let Some(schema) = schema {
            prompt.push_str(&format!("  Parameters: {schema}\n"));
        }
    }
    prompt
}

/// One tool call recovered from model output, in the shape both API layers need.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// Call id. Taken from the model when it supplies one, else synthesised —
    /// clients use it to correlate a result, so it only has to be unique.
    pub id: String,
    pub name: String,
    /// Arguments as a JSON **string**, which is what the OpenAI wire format
    /// uses. Anthropic wants an object, so its handler parses this back; keeping
    /// the string canonical avoids a lossy round-trip for a model that emitted
    /// arguments as a string in the first place.
    pub arguments: String,
}

/// Try every known format against `text`, returning the first that yields at
/// least one call. `None` means "this is ordinary prose" — the overwhelmingly
/// common case, so every path must be cheap to reject.
pub fn parse_tool_calls(text: &str) -> Option<Vec<ParsedToolCall>> {
    let trimmed = strip_code_fences(text.trim());

    // Cheap pre-filter: every supported format contains one of these. Avoids
    // running four parsers over normal chat output.
    let looks_like_tool_call = trimmed.contains("tool_call")
        || trimmed.contains("[TOOL_CALLS]")
        || trimmed.contains("<|python_tag|>")
        || (trimmed.starts_with('{') && trimmed.contains("\"name\""));
    if !looks_like_tool_call {
        return None;
    }

    try_generic(trimmed)
        .or_else(|| try_hermes(trimmed))
        .or_else(|| try_mistral(trimmed))
        .or_else(|| try_llama3(trimmed))
        .filter(|calls| !calls.is_empty())
}

/// Strip a markdown code fence, with or without a language tag.
///
/// Models wrap JSON in ``` constantly — the report that prompted this included a
/// response ending in a stray fence. An unterminated opening fence is also
/// handled, since that is what truncation at `max_tokens` produces.
fn strip_code_fences(text: &str) -> &str {
    let t = text.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    // Drop an optional language tag on the first line.
    let rest = match rest.find('\n') {
        Some(nl) => &rest[nl + 1..],
        None => rest,
    };
    rest.trim_end().strip_suffix("```").unwrap_or(rest).trim()
}

/// `{"tool_calls": [{"id"?, "function": {"name", "arguments"}}]}` — the shape
/// `format_tool_system_prompt` asks for. Also accepts a flattened
/// `{"name", "arguments"}` entry, which models produce about as often as the
/// nested form even when shown the nested one.
fn try_generic(text: &str) -> Option<Vec<ParsedToolCall>> {
    let v: Value = serde_json::from_str(text).ok()?;
    let arr = v.get("tool_calls")?.as_array()?;
    let calls: Vec<ParsedToolCall> = arr
        .iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| synth_id(i));
            // Nested under "function", or flattened onto the entry.
            let f = entry.get("function").unwrap_or(entry);
            let name = f.get("name")?.as_str()?.to_string();
            Some(ParsedToolCall {
                id,
                name,
                arguments: arguments_to_string(f),
            })
        })
        .collect();
    Some(calls)
}

/// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, repeated.
/// Hermes 2/3 and Qwen-Instruct. A trailing unterminated block is skipped
/// rather than guessed at.
fn try_hermes(text: &str) -> Option<Vec<ParsedToolCall>> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if !text.contains(OPEN) {
        return None;
    }
    let mut calls = Vec::new();
    for (i, chunk) in text.split(OPEN).skip(1).enumerate() {
        // No close tag → truncated mid-call; refuse it rather than invent.
        if !chunk.contains(CLOSE) {
            continue;
        }
        let Some(body) = chunk.split(CLOSE).next() else {
            continue;
        };
        if let Some(call) = single_object_call(body.trim(), i) {
            calls.push(call);
        }
    }
    Some(calls)
}

/// `[TOOL_CALLS][{"name": ..., "arguments": {...}}]` — Mistral.
fn try_mistral(text: &str) -> Option<Vec<ParsedToolCall>> {
    let rest = text.split("[TOOL_CALLS]").nth(1)?.trim();
    let v: Value = serde_json::from_str(rest).ok()?;
    let arr = v.as_array()?;
    Some(
        arr.iter()
            .enumerate()
            .filter_map(|(i, e)| single_value_call(e, i))
            .collect(),
    )
}

/// A bare `{"name": ..., "parameters": {...}}`, optionally behind
/// `<|python_tag|>` — Llama 3.x. Tried last because it is the loosest pattern
/// and would otherwise shadow the more specific formats.
fn try_llama3(text: &str) -> Option<Vec<ParsedToolCall>> {
    let body = text
        .split("<|python_tag|>")
        .nth(1)
        .map(str::trim)
        .unwrap_or(text);
    let v: Value = serde_json::from_str(body).ok()?;
    // A single object, or a list of them.
    if let Some(arr) = v.as_array() {
        return Some(
            arr.iter()
                .enumerate()
                .filter_map(|(i, e)| single_value_call(e, i))
                .collect(),
        );
    }
    single_value_call(&v, 0).map(|c| vec![c])
}

/// Parse one `{"name", "arguments"|"parameters"}` object from a JSON string.
fn single_object_call(body: &str, index: usize) -> Option<ParsedToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    single_value_call(&v, index)
}

/// Parse one already-decoded `{"name", ...}` value.
fn single_value_call(v: &Value, index: usize) -> Option<ParsedToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let id = v
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| synth_id(index));
    Some(ParsedToolCall {
        id,
        name,
        arguments: arguments_to_string(v),
    })
}

/// Normalise an arguments field to a JSON string.
///
/// Accepts `arguments` or `parameters` (families differ), and either an object
/// or an already-serialised string (models emit both). Missing arguments become
/// `{}` rather than failing the whole call — a zero-argument tool is legitimate.
fn arguments_to_string(v: &Value) -> String {
    let raw = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("input"));
    match raw {
        // Already a string: the model serialised it itself. Pass through so we
        // don't double-encode.
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    }
}

fn synth_id(index: usize) -> String {
    format!("call_{}", index + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ordinary prose must never be mistaken for a tool call — this is the
    /// common path and a false positive would replace a real answer with an
    /// empty tool call.
    #[test]
    fn plain_text_is_not_a_tool_call() {
        for s in [
            "Hello friend.",
            "The weather in Paris is mild.",
            "",
            "   ",
            "I could call a tool_call but I won't", // mentions the word only
            "{\"answer\": 42}",                     // JSON without a name field
        ] {
            assert_eq!(parse_tool_calls(s), None, "false positive on {s:?}");
        }
    }

    /// The format our own system prompt requests.
    #[test]
    fn parses_the_generic_format_we_prompt_for() {
        let text = r#"{"tool_calls": [{"id": "call_abc", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    /// Models wrap JSON in markdown fences constantly; the reported response
    /// even ended in a stray one.
    #[test]
    fn strips_markdown_fences_including_a_stray_trailing_one() {
        let fenced = "```json\n{\"tool_calls\": [{\"function\": \
                      {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}}]}\n```";
        let calls = parse_tool_calls(fenced).expect("fenced JSON should parse");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);

        // Unterminated fence — what truncation at max_tokens produces.
        let unterminated = "```\n{\"tool_calls\": [{\"function\": \
                            {\"name\": \"ping\", \"arguments\": {}}}]}";
        assert_eq!(parse_tool_calls(unterminated).unwrap()[0].name, "ping");
    }

    /// Hermes / Qwen-Instruct — the most common native format in the wild.
    #[test]
    fn parses_hermes_tool_call_tags_including_multiple() {
        let text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>\n\
                    <tool_call>{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}</tool_call>";
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
        // Synthesised ids must be distinct or clients cannot correlate results.
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn parses_mistral_and_llama3_formats() {
        let mistral = r#"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;
        assert_eq!(parse_tool_calls(mistral).unwrap()[0].name, "get_weather");

        // Llama 3.x uses "parameters" rather than "arguments".
        let llama = r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#;
        let calls = parse_tool_calls(llama).unwrap();
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);

        let tagged =
            "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
        assert_eq!(parse_tool_calls(tagged).unwrap()[0].name, "get_weather");
    }

    /// Truncated output must stay text. Fabricating a call from half an object
    /// would have a client act on arguments the model never finished stating.
    #[test]
    fn truncated_tool_calls_are_refused_not_guessed() {
        // Cut mid-arguments, no closing brace — the reported failure shape.
        let truncated = r#"{"tool_calls": [{"function": {"name": "get_weather", "argum"#;
        assert_eq!(parse_tool_calls(truncated), None);

        // Hermes block with no closing tag.
        let unclosed = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\"";
        assert_eq!(parse_tool_calls(unclosed), None);
    }

    /// A zero-argument tool is legitimate and must not be dropped.
    #[test]
    fn missing_arguments_become_an_empty_object() {
        let text = r#"{"tool_calls": [{"function": {"name": "list_files"}}]}"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].name, "list_files");
        assert_eq!(calls[0].arguments, "{}");
    }

    /// Some models flatten name/arguments onto the entry instead of nesting
    /// them under "function", even when shown the nested form.
    #[test]
    fn accepts_flattened_entries() {
        let text = r#"{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }
}
