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
         `arguments` is a JSON object whose keys are the argument names listed below. \
         Do not wrap it in `properties`, `parameters`, or a type declaration.\n\n\
         Available tools:\n",
    );
    for (name, description, schema) in tools {
        prompt.push_str(&format!("- {name}"));
        if let Some(desc) = description {
            prompt.push_str(&format!(": {desc}"));
        }
        prompt.push('\n');
        if let Some(schema) = schema {
            match describe_arguments(schema) {
                Some(rendered) => prompt.push_str(&rendered),
                // Unrecognised schema shape — fall back to the raw text rather
                // than describing it wrongly.
                None => prompt.push_str(&format!("  Parameters: {schema}\n")),
            }
        }
    }
    prompt
}

/// Render a JSON-Schema parameter object as the ARGUMENT shape a model should
/// produce, rather than as the schema itself.
///
/// Handing a model the raw schema and asking it for `<json_args>` invites it to
/// copy the schema's own structure: `llama-3.2-3b` reproducibly answered
/// `{"properties":{"city":"Paris"}}` instead of `{"city":"Paris"}` (live
/// 2026-07-26, 2/2 runs). That is the worst kind of failure — a tool call that
/// parses cleanly and passes validation while carrying arguments the caller
/// cannot read, so `args.city` is silently undefined.
///
/// Emits a concrete example object plus one line per argument, and never uses
/// the word `properties`. Returns `None` for anything that is not a plain
/// object-with-properties schema (nested objects, `$ref`, unusual shapes), so
/// the caller keeps the raw schema instead of a lossy paraphrase.
fn describe_arguments(schema: &str) -> Option<String> {
    let v: Value = serde_json::from_str(schema).ok()?;
    let props = v.get("properties")?.as_object()?;
    if props.is_empty() {
        return None;
    }
    let required: Vec<&str> = v
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();

    // Example object: {"city": <string>} — shows the exact key placement.
    let example = props
        .iter()
        .map(|(k, spec)| {
            let ty = spec.get("type").and_then(|t| t.as_str()).unwrap_or("value");
            format!("\"{k}\": <{ty}>")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = format!("  arguments: {{{example}}}\n");

    for (k, spec) in props {
        let ty = spec.get("type").and_then(|t| t.as_str()).unwrap_or("value");
        let req = if required.contains(&k.as_str()) {
            ", required"
        } else {
            ", optional"
        };
        out.push_str(&format!("    {k} ({ty}{req})"));
        if let Some(d) = spec.get("description").and_then(|d| d.as_str()) {
            out.push_str(&format!(": {d}"));
        }
        out.push('\n');
    }
    Some(out)
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
        // Not `starts_with('{')`: a model often prefixes prose before the
        // object, and requiring the object to be first re-introduces the bug
        // `parse_embedded_json` exists to fix.
        || (trimmed.contains('{') && trimmed.contains("\"name\""));
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

/// Parse the first complete JSON object embedded in `text`.
///
/// Models rarely emit a bare JSON object. They wrap it in prose ("Here is the
/// tool call:"), append an explanation after it, or both — and requiring the
/// WHOLE string to parse means all of those are reported as ordinary text with
/// the JSON visible to the user. Reported live 2026-07-25 against
/// qwen2.5-coder-7b, which produced a correct tool call the parser then
/// rejected.
///
/// Scans for the first `{` and returns the substring up to its balanced close,
/// tracking string literals and escapes so a brace inside a string value does
/// not end the object early. Returns `None` if no balanced object exists, which
/// is also what a truncated generation produces — deliberately left as text
/// rather than repaired.
fn parse_embedded_json(text: &str) -> Option<Value> {
    // Fast path: the whole thing is already JSON.
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return Some(v);
    }
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if escaped {
            escaped = false;
            continue;
        }
        match b {
            b'\\' if in_string => escaped = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(text.get(start..=i)?).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// `{"tool_calls": [{"id"?, "function": {"name", "arguments"}}]}` — the shape
/// `format_tool_system_prompt` asks for. Also accepts a flattened
/// `{"name", "arguments"}` entry, which models produce about as often as the
/// nested form even when shown the nested one.
fn try_generic(text: &str) -> Option<Vec<ParsedToolCall>> {
    let v: Value = parse_embedded_json(text)?;
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
        // don't double-encode — but re-parse first so a schema-echoed wrapper
        // is unwrapped here too (models serialise arguments both ways).
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => unwrap_schema_echo(&parsed).to_string(),
            Err(_) => s.clone(),
        },
        Some(other) => unwrap_schema_echo(other).to_string(),
        None => "{}".to_string(),
    }
}

/// Undo a model copying the JSON-Schema wrapper into its arguments.
///
/// Asked for arguments while being shown a schema, a small model may answer
/// `{"properties":{"city":"Paris"}}` instead of `{"city":"Paris"}` —
/// `llama-3.2-3b` did so reproducibly (live 2026-07-26). The call parses, looks
/// valid, and passes straight to the caller, who reads `args.city` and gets
/// nothing. A silently-wrong tool call is worse than a rejected one, and the
/// caller has no way to tell.
///
/// `format_tool_prompt` no longer shows the raw schema, which removes most of
/// the temptation; this is the backstop for models that do it anyway.
///
/// Deliberately narrow: unwraps ONLY an object whose single key is `properties`
/// or `parameters` and whose value is itself an object. A tool whose one and
/// only argument is an object literally named `properties` would be unwrapped
/// wrongly — accepted, because that shape is vanishingly rare next to the
/// schema echo, which is reproducible on a current model.
fn unwrap_schema_echo(v: &Value) -> &Value {
    let Some(obj) = v.as_object() else { return v };
    if obj.len() != 1 {
        return v;
    }
    match obj.iter().next() {
        Some((k, inner)) if (k == "properties" || k == "parameters") && inner.is_object() => inner,
        _ => v,
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

    /// A model rarely emits a bare object — it explains itself first, or adds a
    /// note after. Requiring the WHOLE string to parse meant a correct tool call
    /// was reported as text with the JSON visible to the user (reported live
    /// 2026-07-25 against qwen2.5-coder-7b).
    #[test]
    fn finds_a_tool_call_wrapped_in_prose() {
        let prefixed = "Sure! Here is the tool call:\n\
            {\"tool_calls\": [{\"id\": \"call_1\", \"function\": \
            {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}}]}";
        let calls = parse_tool_calls(prefixed).expect("prose-prefixed call should parse");
        assert_eq!(calls[0].name, "get_weather");

        let suffixed = "{\"tool_calls\": [{\"function\": {\"name\": \"ping\", \
            \"arguments\": {}}}]}\n\nI'll call that for you.";
        assert_eq!(parse_tool_calls(suffixed).unwrap()[0].name, "ping");
    }

    /// A brace inside a string value must not end the object early, or the
    /// extracted substring is invalid JSON and a correct call is dropped.
    #[test]
    fn braces_inside_strings_do_not_end_the_object() {
        let tricky = "Here you go: {\"tool_calls\": [{\"function\": \
            {\"name\": \"echo\", \"arguments\": {\"text\": \"a } brace\"}}}]} done";
        let calls = parse_tool_calls(tricky).expect("should parse past the inner brace");
        assert_eq!(calls[0].name, "echo");
        assert!(calls[0].arguments.contains("a } brace"));
    }

    /// Truncation must still be refused even with the looser extraction — an
    /// unbalanced object has no close, so there is nothing safe to act on.
    #[test]
    fn prose_wrapped_but_truncated_is_still_refused() {
        let truncated = "Here is the call: {\"tool_calls\": [{\"function\": \
            {\"name\": \"get_weather\", \"argum";
        assert_eq!(parse_tool_calls(truncated), None);
    }

    /// A GPU-heavy caller path could pass megabytes of prose; make sure a
    /// no-brace input exits immediately rather than scanning.
    #[test]
    fn text_without_any_object_is_rejected_cheaply() {
        assert_eq!(
            parse_tool_calls("I will use the tool_call soon, promise"),
            None
        );
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

#[cfg(test)]
mod schema_echo_tests {
    use super::*;

    /// The live failure: `llama-3.2-3b` answered with the schema's own wrapper
    /// around its arguments. The call parsed and looked valid, but the caller
    /// reading `args.city` got nothing.
    #[test]
    fn schema_echo_is_unwrapped() {
        let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"properties":{"city":"Paris"}}}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    /// Same echo, but the model serialised the arguments as a string.
    #[test]
    fn schema_echo_is_unwrapped_when_stringified() {
        let text = r#"{"tool_calls":[{"function":{"name":"f","arguments":"{\"properties\":{\"city\":\"Paris\"}}"}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    /// Correct arguments must pass through untouched.
    #[test]
    fn correct_arguments_are_left_alone() {
        let text = r#"{"tool_calls":[{"function":{"name":"f","arguments":{"city":"Paris","units":"c"}}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        let v: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["city"], "Paris");
        assert_eq!(v["units"], "c");
    }

    /// Only a LONE wrapper key is unwrapped — an argument that happens to be
    /// called `properties` alongside others is real data.
    #[test]
    fn properties_alongside_other_arguments_is_not_unwrapped() {
        let text = r#"{"tool_calls":[{"function":{"name":"f","arguments":{"properties":{"a":1},"name":"x"}}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        let v: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(v.get("properties").is_some(), "must not unwrap");
        assert_eq!(v["name"], "x");
    }

    /// A non-object value under `properties` is not a schema echo.
    #[test]
    fn scalar_properties_value_is_not_unwrapped() {
        let text =
            r#"{"tool_calls":[{"function":{"name":"f","arguments":{"properties":"blue"}}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls[0].arguments, r#"{"properties":"blue"}"#);
    }

    /// The prompt must describe the ARGUMENT shape, and must not hand the model
    /// the word it was copying.
    #[test]
    fn prompt_describes_arguments_not_schema() {
        let tools = vec![(
            "get_weather".to_string(),
            Some("Get current weather".to_string()),
            Some(
                r#"{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}"#
                    .to_string(),
            ),
        )];
        let p = format_tool_prompt(&tools);
        assert!(p.contains(r#"arguments: {"city": <string>}"#), "got:\n{p}");
        assert!(
            p.contains("city (string, required): City name"),
            "got:\n{p}"
        );
        assert!(
            !p.contains(r#""properties":{"city""#),
            "raw schema must not be shown:\n{p}"
        );
    }

    /// An exotic schema keeps its raw form rather than a lossy paraphrase.
    #[test]
    fn unrenderable_schema_falls_back_to_raw() {
        let tools = vec![(
            "f".to_string(),
            None,
            Some(r##"{"$ref":"#/defs/Thing"}"##.to_string()),
        )];
        let p = format_tool_prompt(&tools);
        assert!(p.contains("Parameters: {\"$ref\""), "got:\n{p}");
    }
}
