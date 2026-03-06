//! Minimal Jinja2-style chat template engine for GGUF models.
//!
//! GGUF files store a `tokenizer.chat_template` metadata field containing a
//! Jinja2-format template string. Rather than pulling in a full Jinja2 parser,
//! this module implements just enough to handle the common patterns used by
//! popular model families (ChatML, Llama, Mistral, Qwen, Gemma, Phi, etc.).
//!
//! Supported constructs:
//! - `{% for message in messages %}...{% endfor %}` — message iteration
//! - `{% if message['role'] == 'system' %}` / `{% elif %}` / `{% else %}` / `{% endif %}`
//! - `{% if add_generation_prompt %}...{% endif %}`
//! - `{% if not loop.last %}...{% endif %}` (partial)
//! - `{{ message['role'] }}`, `{{ message['content'] }}`, `{{ message.role }}`, `{{ message.content }}`
//! - `{{ bos_token }}`, `{{ eos_token }}`
//! - String concatenation with `+` inside `{{ }}`
//! - String literals `'...'` and `"..."`

use crate::types::{ChatMessage, Role};

/// Apply a Jinja2-style chat template to a list of messages.
///
/// `bos_token` and `eos_token` are the special token strings (e.g. `<s>`, `</s>`).
/// `add_generation_prompt` controls whether to append the assistant turn prefix.
///
/// Returns `None` if the template cannot be parsed, so callers can fall back to ChatML.
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
) -> Option<String> {
    let tokens = tokenize(template)?;
    let mut output = String::new();
    let ctx = EvalCtx {
        tokens: &tokens,
        messages,
        bos_token,
        eos_token,
        add_generation_prompt,
    };
    eval_block(&ctx, 0, &mut output, &mut EvalState::TopLevel)?;
    Some(output)
}

// ── Token types ──

#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// Raw text between template tags
    Text(String),
    /// `{{ expr }}` — expression to evaluate and output
    Expr(String),
    /// `{% tag %}` — control flow statement
    Tag(String),
    /// `{%- tag %}` or `{% tag -%}` — tag with whitespace trimming
    TagTrimLeft(String),
    TagTrimRight(String),
    TagTrimBoth(String),
}

impl Token {
    fn tag_content(&self) -> Option<&str> {
        match self {
            Token::Tag(s)
            | Token::TagTrimLeft(s)
            | Token::TagTrimRight(s)
            | Token::TagTrimBoth(s) => Some(s.trim()),
            _ => None,
        }
    }
}

// ── Tokenizer ──

fn tokenize(template: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut rest = template;

    while !rest.is_empty() {
        // Find the next template marker
        if let Some(pos) = rest.find('{') {
            if pos > 0 {
                tokens.push(Token::Text(rest[..pos].to_string()));
                rest = &rest[pos..];
            }

            if rest.starts_with("{{") {
                // Expression: {{ ... }}
                let end = rest.find("}}")?;
                let expr = &rest[2..end];
                tokens.push(Token::Expr(expr.trim().to_string()));
                rest = &rest[end + 2..];
            } else if rest.starts_with("{%") {
                // Tag: {% ... %}
                let end = find_tag_end(rest)?;
                let inner = &rest[2..end];

                // Check for whitespace trimming markers
                let trim_left = inner.starts_with('-');
                let trim_right = inner.ends_with('-');
                let content = inner
                    .trim_start_matches('-')
                    .trim_end_matches('-')
                    .to_string();

                let token = match (trim_left, trim_right) {
                    (true, true) => Token::TagTrimBoth(content),
                    (true, false) => Token::TagTrimLeft(content),
                    (false, true) => Token::TagTrimRight(content),
                    (false, false) => Token::Tag(content),
                };

                // Apply left trim (explicit `{%-`) or lstrip_blocks (default in HF Jinja2):
                // strip leading whitespace on the same line before a block tag.
                if trim_left {
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                } else {
                    // lstrip_blocks: strip spaces/tabs from start of line to the tag
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        if let Some(nl_pos) = t.rfind('\n') {
                            let after_nl = &t[nl_pos + 1..];
                            if after_nl.chars().all(|c| c == ' ' || c == '\t') {
                                t.truncate(nl_pos + 1);
                            }
                        } else if t.chars().all(|c| c == ' ' || c == '\t') {
                            // Entire text token is whitespace at start of template
                            t.clear();
                        }
                    }
                }

                tokens.push(token);
                rest = &rest[end + 2..];

                // Apply right trim (explicit `-%}`) OR trim_blocks (default in HF Jinja2):
                // strip the first newline after a block tag. HuggingFace renders chat
                // templates with `trim_blocks=True, lstrip_blocks=True`, so we always
                // strip the trailing newline after `{% %}` tags.
                if trim_right {
                    rest = rest.trim_start_matches([' ', '\t']);
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    }
                } else {
                    // trim_blocks: strip exactly one newline after a block tag
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    } else if rest.starts_with("\r\n") {
                        rest = &rest[2..];
                    }
                }
            } else if rest.starts_with("{#") {
                // Comment: {# ... #}
                let end = rest.find("#}")?;
                rest = &rest[end + 2..];
            } else {
                // Lone '{' — treat as text
                tokens.push(Token::Text("{".to_string()));
                rest = &rest[1..];
            }
        } else {
            tokens.push(Token::Text(rest.to_string()));
            break;
        }
    }

    Some(tokens)
}

fn find_tag_end(s: &str) -> Option<usize> {
    // s starts with "{%", find matching "%}"
    let inner = &s[2..];
    inner.find("%}").map(|pos| pos + 2)
}

// ── Evaluator ──

/// Bundled evaluation context to avoid passing many parameters through recursive calls.
struct EvalCtx<'a> {
    tokens: &'a [Token],
    messages: &'a [ChatMessage],
    bos_token: &'a str,
    eos_token: &'a str,
    add_generation_prompt: bool,
}

#[derive(Debug)]
enum EvalState<'a> {
    TopLevel,
    InLoop {
        messages: &'a [ChatMessage],
        current_index: usize,
    },
}

/// Evaluate a block of tokens starting at `start`, returning the index after the consumed tokens.
fn eval_block(
    ctx: &EvalCtx,
    start: usize,
    output: &mut String,
    state: &mut EvalState,
) -> Option<usize> {
    let mut i = start;
    while i < ctx.tokens.len() {
        match &ctx.tokens[i] {
            Token::Text(t) => {
                output.push_str(t);
                i += 1;
            }
            Token::Expr(expr) => {
                let val = eval_expr(expr, state, ctx.bos_token, ctx.eos_token)?;
                output.push_str(&val);
                i += 1;
            }
            tok if tok.tag_content().is_some() => {
                let content = tok.tag_content().unwrap();

                if content.starts_with("for ") && content.contains(" in messages") {
                    let body_start = i + 1;
                    let end = find_endfor(ctx.tokens, body_start)?;

                    for (idx, _msg) in ctx.messages.iter().enumerate() {
                        let mut loop_state = EvalState::InLoop {
                            messages: ctx.messages,
                            current_index: idx,
                        };
                        eval_block(ctx, body_start, output, &mut loop_state)?;
                    }
                    i = end + 1;
                } else if content.starts_with("if ") {
                    i = eval_if_chain(ctx, i, output, state)?;
                } else if content == "endfor"
                    || content == "endif"
                    || content.starts_with("elif ")
                    || content == "else"
                {
                    return Some(i);
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    Some(i)
}

/// Evaluate an if/elif/else/endif chain. Returns the token index after the endif.
fn eval_if_chain(
    ctx: &EvalCtx,
    start: usize,
    output: &mut String,
    state: &mut EvalState,
) -> Option<usize> {
    let content = ctx.tokens[start].tag_content()?;
    let condition = content.strip_prefix("if ")?.trim();
    let matched = eval_condition(condition, state, ctx.add_generation_prompt);

    let body_start = start + 1;

    if matched {
        let end = eval_block(ctx, body_start, output, state)?;
        return skip_to_endif(ctx, end, output, state, true);
    }

    let end = skip_block_content(ctx.tokens, body_start)?;
    skip_to_endif(ctx, end, output, state, matched)
}

/// Skip forward through elif/else/endif chain.
fn skip_to_endif(
    ctx: &EvalCtx,
    mut i: usize,
    output: &mut String,
    state: &mut EvalState,
    mut already_matched: bool,
) -> Option<usize> {
    while i < ctx.tokens.len() {
        let content = ctx.tokens[i].tag_content()?;

        if content == "endif" {
            return Some(i + 1);
        } else if content.starts_with("elif ") {
            let cond = content.strip_prefix("elif ")?.trim();
            let cond_result = eval_condition(cond, state, ctx.add_generation_prompt);

            if !already_matched && cond_result {
                already_matched = true;
                i = eval_block(ctx, i + 1, output, state)?;
            } else {
                i = skip_block_content(ctx.tokens, i + 1)?;
            }
        } else if content == "else" {
            if !already_matched {
                i = eval_block(ctx, i + 1, output, state)?;
                if i < ctx.tokens.len() {
                    if let Some(c) = ctx.tokens[i].tag_content() {
                        if c == "endif" {
                            return Some(i + 1);
                        }
                    }
                }
                return Some(i);
            } else {
                i = skip_block_content(ctx.tokens, i + 1)?;
            }
        } else {
            return Some(i + 1);
        }
    }
    Some(i)
}

/// Skip tokens until we hit a sibling elif/else/endif at the same nesting level.
fn skip_block_content(tokens: &[Token], start: usize) -> Option<usize> {
    let mut i = start;
    let mut depth = 0;
    while i < tokens.len() {
        if let Some(content) = tokens[i].tag_content() {
            if content.starts_with("if ") || content.starts_with("for ") {
                depth += 1;
            } else if content == "endif" || content == "endfor" {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            } else if depth == 0 && (content.starts_with("elif ") || content == "else") {
                return Some(i);
            }
        }
        i += 1;
    }
    Some(i)
}

/// Find the matching {% endfor %} for a for loop body starting at `start`.
fn find_endfor(tokens: &[Token], start: usize) -> Option<usize> {
    let mut depth = 0;
    let mut i = start;
    while i < tokens.len() {
        if let Some(content) = tokens[i].tag_content() {
            if content.starts_with("for ") {
                depth += 1;
            } else if content == "endfor" {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            }
        }
        i += 1;
    }
    None
}

// ── Expression evaluator ──

/// Evaluate a `{{ ... }}` expression and return the string value.
fn eval_expr(expr: &str, state: &EvalState, bos_token: &str, eos_token: &str) -> Option<String> {
    let expr = expr.trim();

    // Handle string concatenation with +
    if contains_plus_outside_strings(expr) {
        let parts = split_on_plus(expr);
        let mut result = String::new();
        for part in parts {
            result.push_str(&eval_expr(part.trim(), state, bos_token, eos_token)?);
        }
        return Some(result);
    }

    // String literal: 'foo' or "foo"
    if let Some(s) = parse_string_literal(expr) {
        return Some(s);
    }

    // Special variables
    match expr {
        "bos_token" => return Some(bos_token.to_string()),
        "eos_token" => return Some(eos_token.to_string()),
        _ => {}
    }

    // Message field access: message['role'], message['content'], message.role, message.content
    if let EvalState::InLoop {
        messages,
        current_index,
    } = state
    {
        let msg = messages.get(*current_index)?;

        if expr == "message['role']" || expr == "message[\"role\"]" || expr == "message.role" {
            return Some(role_str(&msg.role).to_string());
        }
        if expr == "message['content']"
            || expr == "message[\"content\"]"
            || expr == "message.content"
        {
            return Some(msg.content.clone());
        }
    }

    // If nothing matched, return the expr as-is (best effort)
    None
}

/// Evaluate a condition expression (for {% if %} / {% elif %}).
fn eval_condition(condition: &str, state: &EvalState, add_generation_prompt: bool) -> bool {
    let condition = condition.trim();

    // Compound conditions: `A and B`, `A or B`
    // Split on ` and ` / ` or ` (space-delimited to avoid matching inside strings)
    if let Some(pos) = condition.find(" and ") {
        let left = &condition[..pos];
        let right = &condition[pos + 5..];
        return eval_condition(left, state, add_generation_prompt)
            && eval_condition(right, state, add_generation_prompt);
    }
    if let Some(pos) = condition.find(" or ") {
        let left = &condition[..pos];
        let right = &condition[pos + 4..];
        return eval_condition(left, state, add_generation_prompt)
            || eval_condition(right, state, add_generation_prompt);
    }

    // add_generation_prompt
    if condition == "add_generation_prompt" {
        return add_generation_prompt;
    }
    if condition == "not add_generation_prompt" {
        return !add_generation_prompt;
    }

    // loop.last / not loop.last / loop.first
    if let EvalState::InLoop {
        messages,
        current_index,
    } = state
    {
        if condition == "not loop.last" {
            return *current_index < messages.len().saturating_sub(1);
        }
        if condition == "loop.last" {
            return *current_index == messages.len().saturating_sub(1);
        }
        if condition == "loop.first" {
            return *current_index == 0;
        }

        // message['role'] == 'system', message.role == 'system', etc.
        if let Some(role_check) = parse_role_comparison(condition) {
            let msg = messages.get(*current_index);
            if let Some(msg) = msg {
                return role_str(&msg.role) == role_check;
            }
        }

        // message['role'] != 'system'
        if let Some(role_check) = parse_role_not_equal(condition) {
            let msg = messages.get(*current_index);
            if let Some(msg) = msg {
                return role_str(&msg.role) != role_check;
            }
        }
    }

    // messages[0]['role'] == 'system' (without loop context)
    if condition.starts_with("messages[0]") {
        // Not inside a loop, but checking the first message
        return false; // Can't evaluate without message context
    }

    false
}

fn parse_role_comparison(condition: &str) -> Option<&str> {
    // Patterns: message['role'] == 'system', message.role == 'system', etc.
    let patterns = [
        "message['role'] == ",
        "message[\"role\"] == ",
        "message.role == ",
    ];
    for pat in &patterns {
        if let Some(rest) = condition.strip_prefix(pat) {
            return parse_string_literal_ref(rest.trim());
        }
    }
    None
}

fn parse_role_not_equal(condition: &str) -> Option<&str> {
    let patterns = [
        "message['role'] != ",
        "message[\"role\"] != ",
        "message.role != ",
    ];
    for pat in &patterns {
        if let Some(rest) = condition.strip_prefix(pat) {
            return parse_string_literal_ref(rest.trim());
        }
    }
    None
}

fn parse_string_literal(s: &str) -> Option<String> {
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

fn parse_string_literal_ref(s: &str) -> Option<&str> {
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        Some(&s[1..s.len() - 1])
    } else {
        None
    }
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Check if the expression contains a `+` operator outside of string literals.
fn contains_plus_outside_strings(expr: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    for ch in expr.chars() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '+' if !in_single && !in_double => return true,
            _ => {}
        }
    }
    false
}

/// Split an expression on `+` outside of string literals.
fn split_on_plus(expr: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_single = false;
    let mut in_double = false;

    for (i, ch) in expr.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '+' if !in_single && !in_double => {
                parts.push(&expr[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&expr[start..]);
    parts
}

/// Build a ChatML-formatted prompt (default fallback when no template is available).
pub fn chatml_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = role_str(&msg.role);
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// The `<image>` placeholder token used by LLaVA to mark where vision embeddings go.
pub const IMAGE_PLACEHOLDER: &str = "<image>";

/// Build a Vicuna v1.1 formatted prompt (used by LLaVA and other Vicuna-based models).
///
/// When a user message contains images, `<image>\n` is prepended to the user content
/// so that the vision encoder embeddings can replace the `<image>` token embedding
/// at the correct position in the sequence.
pub fn vicuna_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    // System message
    let sys = messages.iter().find(|m| matches!(m.role, Role::System));
    if let Some(s) = sys {
        prompt.push_str(&s.content);
        prompt.push(' ');
    }
    for msg in messages {
        match msg.role {
            Role::System => {} // already handled
            Role::User => {
                prompt.push_str("USER: ");
                if !msg.images.is_empty() {
                    prompt.push_str(IMAGE_PLACEHOLDER);
                    prompt.push('\n');
                }
                prompt.push_str(&msg.content);
                prompt.push(' ');
            }
            Role::Assistant => {
                prompt.push_str("ASSISTANT: ");
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            }
            Role::Tool => {
                prompt.push_str("TOOL: ");
                prompt.push_str(&msg.content);
                prompt.push(' ');
            }
        }
    }
    prompt.push_str("ASSISTANT:");
    prompt
}

/// Extract stop strings from a chat template.
///
/// These are role markers that, when generated by the model, indicate the start
/// of a new turn and should terminate generation. Common patterns:
/// - Zephyr/TinyLlama: `<|user|>`, `<|system|>`
/// - ChatML: `<|im_end|>`, `<|im_start|>`
/// - Llama: `[INST]`
pub fn extract_stop_strings(template: Option<&str>) -> Vec<String> {
    let tmpl = match template {
        Some(t) => t,
        None => return vec!["<|im_end|>".to_string()], // ChatML fallback
    };

    let mut stops = Vec::new();

    // Scan template for known role marker patterns
    for marker in &[
        "<|user|>",
        "<|system|>",
        "<|im_end|>",
        "<|im_start|>",
        "[INST]",
        "<|eot_id|>",
    ] {
        if tmpl.contains(marker) {
            stops.push(marker.to_string());
        }
    }

    // For Zephyr-style templates, `<|assistant|>` in the middle of generation
    // means the model is hallucinating a new assistant turn after ending its own.
    if tmpl.contains("<|assistant|>") && tmpl.contains("<|user|>") {
        // Already have <|user|>, also stop on <|assistant|> if model re-emits it
        if !stops.contains(&"<|assistant|>".to_string()) {
            stops.push("<|assistant|>".to_string());
        }
    }

    stops
}

/// Build a chat prompt using the given template, falling back to ChatML.
///
/// This is the main entry point — all call sites should use this instead of
/// the old hardcoded `build_chat_prompt`.
pub fn build_prompt(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
) -> String {
    build_prompt_with_model(messages, template, bos_token, eos_token, None)
}

/// Build prompt with optional model name hint for fallback template selection.
pub fn build_prompt_with_model(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
    model_name: Option<&str>,
) -> String {
    if let Some(tmpl) = template {
        if let Some(result) = apply_chat_template(tmpl, messages, bos_token, eos_token, true) {
            tracing::debug!(template_matched = true, "DIAG: chat template applied");
            return result;
        }
        tracing::warn!(
            fallback = "chatml",
            "DIAG: chat template failed, using fallback"
        );
    } else {
        // No template: pick fallback based on model name heuristic
        if let Some(name) = model_name {
            let name_lower = name.to_lowercase();
            if name_lower.contains("llava") || name_lower.contains("vicuna") {
                tracing::debug!(
                    model_name = name,
                    fallback = "vicuna",
                    "DIAG: no chat template, using vicuna fallback"
                );
                return vicuna_fallback(messages);
            }
        }
        tracing::debug!(
            template_matched = false,
            fallback = "chatml",
            "DIAG: no chat template, using fallback"
        );
    }
    chatml_fallback(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role};

    fn test_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".into(),
                images: vec![],
            },
        ]
    }

    #[test]
    fn chatml_template_roundtrip() {
        // Standard ChatML template used by Qwen2, many OpenHermes models, etc.
        let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_fallback_matches_original() {
        let msgs = test_messages();
        let result = chatml_fallback(&msgs);
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_style_template() {
        // Simplified Llama 3 / Llama 3.1 style template
        let template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";
        let msgs = test_messages();
        let result =
            apply_chat_template(template, &msgs, "<|begin_of_text|>", "<|eot_id|>", true).unwrap();
        assert!(result
            .contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn mistral_style_template() {
        // Simplified Mistral Instruct template
        let template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}";
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "<s>", "</s>", true).unwrap();
        assert_eq!(result, "<s>[INST] Hello [/INST]");
    }

    #[test]
    fn build_prompt_with_template() {
        let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
        let msgs = test_messages();
        let result = build_prompt(&msgs, Some(template), "", "");
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn build_prompt_without_template_falls_back() {
        let msgs = test_messages();
        let result = build_prompt(&msgs, None, "", "");
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn dot_notation_works() {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert!(result.contains("system: You are helpful.\n"));
        assert!(result.contains("user: Hello\n"));
        assert!(result.ends_with("assistant: "));
    }

    #[test]
    fn empty_messages() {
        let template = "{% for message in messages %}{{ message.content }}{% endfor %}";
        let result = apply_chat_template(template, &[], "", "", true).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn no_generation_prompt() {
        let template = "{% for message in messages %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}ASSIST{% endif %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert!(!result.contains("ASSIST"));
    }

    #[test]
    fn bos_eos_tokens() {
        let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}";
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "Hi".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "<s>", "</s>", true).unwrap();
        assert_eq!(result, "<s>Hi</s>");
    }

    #[test]
    fn zephyr_tinyllama_template() {
        // TinyLlama / Zephyr uses `loop.last and add_generation_prompt`
        let template = r#"{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>
' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"#;
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "<s>", "</s>", true).unwrap();
        assert!(result.contains("<|user|>\nHello</s>"));
        assert!(
            result.ends_with("<|assistant|>"),
            "Expected prompt to end with <|assistant|>, got: {:?}",
            &result[result.len().saturating_sub(30)..]
        );
    }

    #[test]
    fn compound_and_condition() {
        // Verify `and` compound conditions work
        let template = "{% for message in messages %}{{ message.content }}{% if loop.last and add_generation_prompt %}ASSIST{% endif %}{% endfor %}";
        let msgs = vec![
            ChatMessage {
                role: Role::User,
                content: "A".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::User,
                content: "B".into(),
                images: vec![],
            },
        ];
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert_eq!(result, "ABASSIST");
        // Without generation prompt, ASSIST should NOT appear
        let result2 = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result2, "AB");
    }

    #[test]
    fn else_branch() {
        let template = "{% for message in messages %}{% if message['role'] == 'system' %}SYS:{{ message['content'] }}{% else %}OTHER:{{ message['content'] }}{% endif %}{% endfor %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert!(result.contains("SYS:You are helpful."));
        assert!(result.contains("OTHER:Hello"));
    }

    #[test]
    fn zephyr_tinyllama_multiline_template() {
        // The ACTUAL template from the TinyLlama GGUF header (with newlines between tags).
        // HuggingFace renders with trim_blocks=True, lstrip_blocks=True.
        let template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}\n";
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "<s>", "</s>", true).unwrap();
        assert!(
            result.contains("<|user|>\nHello</s>"),
            "Expected user message, got: {:?}",
            result
        );
        assert!(
            result.trim_end().ends_with("<|assistant|>"),
            "Expected prompt to end with <|assistant|>, got: {:?}",
            result
        );
        // Should NOT have excessive newlines
        assert!(
            !result.contains("\n\n\n"),
            "Too many consecutive newlines: {:?}",
            result
        );
    }
}
