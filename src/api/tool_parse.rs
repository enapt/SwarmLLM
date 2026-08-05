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

/// Whether `tool_choice` forbids the model from calling a tool at all.
///
/// A cloud provider enforces this itself. A local model only knows about its
/// tools because [`format_tool_prompt`] puts them in the prompt, so honouring
/// "none" means not describing them in the first place — otherwise the model
/// is told "here are your tools" and told nothing about being forbidden, and
/// calls one. Measured on llama-3.2-3b: `tool_choice: "none"` produced a tool
/// call every time, because neither API layer read the field.
///
/// Accepts both spellings, since both layers store the raw JSON: OpenAI sends
/// the string `"none"`, Anthropic an object `{"type": "none"}`.
///
/// Anything else — `"auto"`, `"required"`, a named function, or absent — leaves
/// the tools described. "required" is not enforced here: a local model cannot be
/// compelled, and refusing the request would be worse than letting it answer.
pub fn tool_choice_forbids_tools(tool_choice: &Option<serde_json::Value>) -> bool {
    match tool_choice {
        Some(Value::String(s)) => s == "none",
        Some(Value::Object(o)) => o.get("type").and_then(|t| t.as_str()) == Some("none"),
        _ => false,
    }
}

/// One tool call recovered from model output, in the shape both API layers need.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// Call id, always assigned here — never the one the model wrote.
    ///
    /// A client correlates a tool result back to its call by this id, so it has
    /// to be unique across everything the client is tracking, which is a whole
    /// conversation and not one response. Models do not do that: llama-3.2-3b
    /// at temperature 0 emits `call_1`, `call_2`, `call_3` in that order for
    /// EVERY tool-calling response it produces, so a conversation with three
    /// such turns contains three different calls all claiming to be `call_1`.
    /// Nothing stops a model reusing one id twice inside a single response
    /// either, which breaks correlation outright.
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
        .map(assign_unique_ids)
}

/// Replace whatever id the model wrote with one that is actually unique.
///
/// Done here rather than in each parser or each API layer because this is the
/// single point every tool call crosses on its way out — all four callers
/// (Anthropic streaming and not, OpenAI streaming and not) go through
/// `parse_tool_calls`, so a new caller inherits this instead of having to
/// remember it. See [`ParsedToolCall::id`] for why the model's own id will not
/// do.
fn assign_unique_ids(mut calls: Vec<ParsedToolCall>) -> Vec<ParsedToolCall> {
    for call in &mut calls {
        call.id = format!("call_{}", uuid::Uuid::new_v4().simple());
    }
    calls
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
///
/// One near-miss IS repaired: a mismatched closer where an array is still open.
/// The brace scan below deliberately ignores `[`/`]`, so a model that closes its
/// array with `}` instead of `]` yields a brace-balanced slice that `serde_json`
/// then rejects, and a perfectly good tool call is handed to the user as raw
/// JSON. Observed live on llama-3.2-3b, **1 run in 3** at default temperature:
///
/// ```text
/// {"tool_calls": [{"id": …, "function": {"name": "get_weather",
///                  "arguments": {"city": "Paris"}}}}
///                                                 ^ should be ]}
/// ```
///
/// [`repair_unbalanced_close`] only ever INSERTS the closer the open delimiter
/// requires. It never invents a key, a value or a name, so it cannot turn
/// garbage into a plausible-but-wrong call — the failure mode the argument
/// renderer above exists to prevent. Everything it produces still has to satisfy
/// `try_generic`'s requirement of a string `name`.
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
                    let slice = text.get(start..=i)?;
                    return match serde_json::from_str(slice) {
                        Ok(v) => Some(v),
                        // Brace-balanced but still invalid — the array case above.
                        Err(_) => repair_unbalanced_close(slice)
                            .and_then(|fixed| serde_json::from_str(&fixed).ok()),
                    };
                }
            }
            _ => {}
        }
    }
    // Never reached brace depth 0 — the generation stopped before closing up.
    // If the ONLY thing missing is closing delimiters, completing them invents
    // nothing, so the call is recoverable; anything else stays text.
    close_open_delimiters(text.get(start..)?).and_then(|fixed| serde_json::from_str(&fixed).ok())
}

/// Complete a candidate that ends with delimiters still open, e.g. a model that
/// stopped one `}` short of finishing.
///
/// Observed live on llama-3.2-3b alongside the mismatched-closer case, and it is
/// the same defect from the user's side — a complete, correct tool call shown as
/// raw text because of one absent character:
///
/// ```text
/// {"tool_calls": [{"id": …, "function": {"name": "get_weather",
///                  "arguments": {"city": "Paris"}}}]
///                                                  ^ missing final }
/// ```
///
/// **This is NOT general truncation repair.** It refuses anything where the cut
/// landed mid-value, because completing THOSE would fabricate an argument the
/// model never finished saying — a tool call that parses cleanly and does the
/// wrong thing, which is worse than the leak. Specifically it refuses when the
/// text ends inside a string literal, or on a dangling `,` or `:`.
fn close_open_delimiters(slice: &str) -> Option<String> {
    /// Same near-miss budget as [`repair_unbalanced_close`].
    const MAX_CLOSERS: usize = 4;
    let mut stack: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for &b in slice.as_bytes() {
        if escaped {
            escaped = false;
            continue;
        }
        match b {
            b'\\' if in_string => escaped = true,
            b'"' => in_string = !in_string,
            b'{' | b'[' if !in_string => stack.push(b),
            b'}' if !in_string => {
                if stack.last() != Some(&b'{') {
                    return None;
                }
                stack.pop();
            }
            b']' if !in_string => {
                if stack.last() != Some(&b'[') {
                    return None;
                }
                stack.pop();
            }
            _ => {}
        }
    }
    // Cut mid-string: the next characters were part of a value we do not have.
    if in_string || stack.is_empty() || stack.len() > MAX_CLOSERS {
        return None;
    }
    // Cut immediately after a separator: a key or value was about to follow.
    match slice.trim_end().chars().last()? {
        ',' | ':' => return None,
        _ => {}
    }
    let mut out = String::with_capacity(slice.len() + stack.len());
    out.push_str(slice.trim_end());
    for open in stack.iter().rev() {
        out.push(if *open == b'[' { ']' } else { '}' });
    }
    Some(out)
}

/// Rewrite a JSON candidate whose closers are the wrong KIND, e.g. `}` used to
/// close an array.
///
/// Tracks the real delimiter stack (`{` and `[`) and, on a closer that does not
/// match the innermost open delimiter, emits the closers that are actually
/// required. Purely structural: no key, value or name is ever invented, so a
/// repaired string cannot carry arguments the model did not produce.
///
/// Returns `None` when nothing needed fixing, when the input is unsalvageable,
/// or when more than [`MAX_REPAIRS`] corrections would be required — a long run
/// of mismatches means the output is not a near-miss tool call and guessing at
/// it is worse than showing it.
fn repair_unbalanced_close(slice: &str) -> Option<String> {
    /// Beyond a couple of corrections this stops being a near miss.
    const MAX_REPAIRS: usize = 4;
    let mut out = String::with_capacity(slice.len() + MAX_REPAIRS);
    let mut stack: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut repairs = 0usize;

    for &b in slice.as_bytes() {
        if escaped {
            escaped = false;
            out.push(b as char);
            continue;
        }
        match b {
            b'\\' if in_string => {
                escaped = true;
                out.push(b as char);
            }
            b'"' => {
                in_string = !in_string;
                out.push('"');
            }
            b'{' | b'[' if !in_string => {
                stack.push(b);
                out.push(b as char);
            }
            b'}' | b']' if !in_string => {
                let want = if b == b'}' { b'{' } else { b'[' };
                // Close any delimiters the model skipped over.
                while let Some(&top) = stack.last() {
                    if top == want {
                        break;
                    }
                    repairs += 1;
                    if repairs > MAX_REPAIRS {
                        return None;
                    }
                    stack.pop();
                    out.push(if top == b'[' { ']' } else { '}' });
                }
                stack.pop()?;
                out.push(b as char);
            }
            _ => out.push(b as char),
        }
    }
    // Anything still open was never closed at all; that is truncation, which
    // stays text by design.
    if !stack.is_empty() || repairs == 0 {
        return None;
    }
    Some(out)
}

/// `{"tool_calls": [{"id"?, "function": {"name", "arguments"}}]}` — the shape
/// `format_tool_system_prompt` asks for. Also accepts a flattened
/// `{"name", "arguments"}` entry, which models produce about as often as the
/// nested form even when shown the nested one.
fn try_generic(text: &str) -> Option<Vec<ParsedToolCall>> {
    let v: Value = parse_embedded_json(text)?;
    // A model often drops the wrapper and emits the ARRAY ELEMENT on its own:
    // `{"id": …, "type": "function", "function": {"name": …, "arguments": …}}`.
    // That is unambiguous, and refusing it returned a perfectly good call to the
    // user as raw JSON text (observed live on qwen2.5-0.5b, 2026-07-26). Only a
    // `function` object carrying a string `name` qualifies, so ordinary prose
    // containing JSON is not swept up.
    let single = v
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .map(|_| std::slice::from_ref(&v));
    let arr = match v.get("tool_calls").and_then(Value::as_array) {
        Some(a) => a.as_slice(),
        None => single?,
    };
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

    /// **The exact string a live model produced**, llama-3.2-3b at default
    /// temperature, 1 run in 3: the array is closed with `}` instead of `]`.
    ///
    /// Before the repair this was handed to the user as the assistant's reply —
    /// raw JSON instead of a tool call, which for a Claude Code user means the
    /// tool simply does not work.
    #[test]
    fn an_array_closed_with_a_brace_is_still_a_tool_call() {
        let observed = r#"{"tool_calls": [{"id": "call_7c5f4d3f", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}}"#;
        let calls = parse_tool_calls(observed).expect("should recover the call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        // The ARGUMENTS must survive untouched — a repair that changed them
        // would be worse than the leak it replaces.
        assert!(
            calls[0].arguments.contains("Paris"),
            "arguments lost or altered: {}",
            calls[0].arguments
        );
    }

    /// The repair is structural only. It must never invent a name, so text that
    /// merely looks bracket-ish cannot become a call.
    #[test]
    fn repair_never_invents_a_tool_call() {
        assert!(parse_tool_calls(r#"{"tool_calls": [{"id": "x"}}"#).is_none());
        assert!(parse_tool_calls(r#"{"tool_calls": [{"function": {}}}"#).is_none());
        // Ordinary prose that happens to contain braces.
        assert!(parse_tool_calls("I would use {the tool} if I could.").is_none());
    }

    /// **The other string a live model produced**: the array closes correctly
    /// but the final `}` never arrives. Everything semantic is present, so
    /// completing the delimiter invents nothing.
    #[test]
    fn a_call_missing_only_its_final_brace_is_recovered() {
        let observed = r#"{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]"#;
        let calls = parse_tool_calls(observed).expect("should recover the call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].arguments.contains("Paris"));
    }

    /// A cut that landed MID-VALUE must stay text. Completing it would fabricate
    /// an argument the model never finished saying — a call that parses cleanly
    /// and does the wrong thing, which is worse than showing the raw text.
    #[test]
    fn a_cut_inside_a_value_is_never_completed() {
        // Ends inside the city string: "Par… could have been Paris or Parma.
        assert!(parse_tool_calls(
            r#"{"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Par"#
        )
        .is_none());
        // Ends on a dangling separator: a value was about to follow.
        assert!(parse_tool_calls(
            r#"{"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city":"#
        )
        .is_none());
        // Ends before the name is known at all.
        assert!(
            parse_tool_calls(r#"{"tool_calls": [{"function": {"nam"#).is_none(),
            "a call with no recoverable name must not be invented"
        );
    }

    /// A long run of mismatches is not a near miss; guessing is worse than
    /// showing the text.
    #[test]
    fn wildly_unbalanced_input_is_refused() {
        let junk = r#"{"tool_calls": [[[[{"name": "x"}}}}}}}}"#;
        // Either None, or at minimum never a confident call with a bad name.
        if let Some(calls) = parse_tool_calls(junk) {
            assert!(calls.iter().all(|c| !c.name.is_empty()));
        }
    }

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

    /// A client matches a tool result back to its call by id, across a whole
    /// conversation. The model's own id cannot carry that: llama-3.2-3b at
    /// temperature 0 emits `call_1`, `call_2`, `call_3` for EVERY tool-calling
    /// response, so three such turns leave three distinct calls all claiming to
    /// be `call_1` (measured live 2026-08-06, 4 runs of 4).
    #[test]
    fn a_models_own_call_id_is_never_used() {
        // Two responses a model could plausibly produce back to back, the
        // second reusing an id inside one response.
        let first = r#"{"tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
            {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}]}"#;
        let second = r#"{"tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Cairo\"}"}},
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lima\"}"}}]}"#;

        let a = parse_tool_calls(first).expect("parses");
        let b = parse_tool_calls(second).expect("parses");

        let mut ids: Vec<&str> = a.iter().chain(b.iter()).map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), 4);
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            4,
            "ids must be unique across responses, and within one even when the \
             model repeats itself"
        );
        for c in a.iter().chain(b.iter()) {
            assert!(!matches!(c.id.as_str(), "call_1" | "call_2"));
        }
        // The arguments still belong to the right call.
        assert!(a[0].arguments.contains("Paris"));
        assert!(b[1].arguments.contains("Lima"));
    }

    /// The format our own system prompt requests.
    #[test]
    fn parses_the_generic_format_we_prompt_for() {
        let text = r#"{"tool_calls": [{"id": "call_abc", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}"#;
        let calls = parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        // The model's own id is deliberately discarded — see
        // `ParsedToolCall::id` and `a_models_own_call_id_is_never_used`.
        assert_ne!(calls[0].id, "call_abc");
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

#[cfg(test)]
mod bare_call_tests {
    use super::*;

    /// Models frequently drop the `tool_calls` wrapper and emit the array
    /// element on its own. Refusing it handed the user raw JSON instead of a
    /// tool call (observed live on qwen2.5-0.5b, 2026-07-26).
    #[test]
    fn bare_single_call_object_is_accepted() {
        let text = r#"{"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}"#;
        let calls = parse_tool_calls(text).expect("bare call should parse");
        assert_eq!(calls.len(), 1);
        assert_ne!(
            calls[0].id, "call_weather",
            "the model's id must be replaced"
        );
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    /// The wrapper form still works and still wins.
    #[test]
    fn wrapped_form_still_parses() {
        let text = r#"{"tool_calls":[{"function":{"name":"a","arguments":{"x":1}}},{"function":{"name":"b","arguments":{}}}]}"#;
        let calls = parse_tool_calls(text).expect("wrapped should parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    /// Truncation stays text — we never invent arguments the user did not
    /// approve. This is the boundary the bare-object support must not cross.
    #[test]
    fn truncated_call_is_still_refused() {
        let text =
            r#"{"tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weat"#;
        assert!(parse_tool_calls(text).is_none());
    }

    /// Prose that merely mentions a function must not become a tool call.
    #[test]
    fn prose_with_a_function_word_is_not_a_call() {
        for s in [
            "The function name is stored in the registry.",
            r#"Here is some config: {"function": "enabled"}"#,
            r#"{"function": {"description": "no name field here"}}"#,
        ] {
            assert!(parse_tool_calls(s).is_none(), "false positive on: {s}");
        }
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::tool_choice_forbids_tools;
    use serde_json::json;

    /// A local model only learns about its tools from the prompt, so "none"
    /// has to mean "do not describe them" — there is nowhere else to enforce
    /// it. Neither API layer read the field, and measured on llama-3.2-3b,
    /// `tool_choice: "none"` produced a tool call every time.
    #[test]
    fn none_forbids_tools_in_both_spellings() {
        assert!(tool_choice_forbids_tools(&Some(json!("none"))));
        assert!(tool_choice_forbids_tools(&Some(json!({"type": "none"}))));
    }

    /// Everything else leaves the tools described. "required" in particular is
    /// NOT treated as forbidding them — an easy thing to invert by accident.
    #[test]
    fn every_other_choice_leaves_tools_available() {
        for tc in [
            json!("auto"),
            json!("required"),
            json!({"type": "auto"}),
            json!({"type": "any"}),
            json!({"type": "tool", "name": "get_weather"}),
            json!({"type": "function", "function": {"name": "get_weather"}}),
            json!("nonsense"),
            json!(42),
        ] {
            assert!(
                !tool_choice_forbids_tools(&Some(tc.clone())),
                "{tc} must not disable tools"
            );
        }
        assert!(!tool_choice_forbids_tools(&None));
    }
}
