//! Jinja2-style chat template engine for GGUF models.
//!
//! GGUF files store a `tokenizer.chat_template` metadata field containing a
//! Jinja2-format template string. This module implements enough of Jinja2 to
//! handle the common patterns used by popular model families:
//! ChatML, Llama, Mistral, Qwen, Gemma, Phi, TinyLlama, etc.
//!
//! Supported constructs:
//! - `{% for message in messages %}...{% endfor %}` — message iteration
//! - `{% if %}` / `{% elif %}` / `{% else %}` / `{% endif %}` — conditionals
//! - `{% set var = expr %}` — variable assignment
//! - `{{ expr }}` with `{{- expr -}}` whitespace trimming
//! - `{%- tag -%}` whitespace trimming on tags
//! - `message['role']`, `message.role`, `message['content']`, `message.content`
//! - `messages[N]['role']`, `messages[N]['content']` — indexed message access
//! - `loop.index0`, `loop.first`, `loop.last`, `not loop.last`
//! - `bos_token`, `eos_token`
//! - String concatenation with `+`
//! - String literals with `\n`, `\t` escape sequences
//! - `| trim`, `| tojson` filters
//! - `raise_exception(...)` — silently ignored (no-op)
//! - `and`, `or`, `not` operators with proper precedence
//! - Parenthesized conditions
//! - Undefined variables evaluate as falsy

use crate::types::{ChatMessage, Role};

/// Apply a Jinja2-style chat template to a list of messages.
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
    let mut state = EvalState::new(messages);
    eval_block(&ctx, 0, &mut output, &mut state)?;
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
                // Expression: {{ ... }} with optional {{- / -}} trimming
                let end = rest.find("}}")?;
                let raw = &rest[2..end];

                // Check for {{- (trim left)
                let (trim_left, after_left) = if let Some(stripped) = raw.strip_prefix('-') {
                    (true, stripped)
                } else {
                    (false, raw)
                };
                // Check for -}} (trim right)
                let (trim_right, content) = if let Some(stripped) = after_left.strip_suffix('-') {
                    (true, stripped)
                } else {
                    (false, after_left)
                };

                if trim_left {
                    if let Some(Token::Text(ref mut t)) = tokens.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                }

                tokens.push(Token::Expr(content.trim().to_string()));
                rest = &rest[end + 2..];

                if trim_right {
                    rest = rest.trim_start_matches([' ', '\t']);
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    } else if rest.starts_with("\r\n") {
                        rest = &rest[2..];
                    }
                }
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

/// Bundled evaluation context (immutable across the entire template).
struct EvalCtx<'a> {
    tokens: &'a [Token],
    messages: &'a [ChatMessage],
    bos_token: &'a str,
    eos_token: &'a str,
    add_generation_prompt: bool,
}

/// Mutable evaluation state (changes per loop iteration, accumulates variables).
struct EvalState<'a> {
    messages: &'a [ChatMessage],
    loop_index: Option<usize>,
    vars: std::collections::HashMap<String, String>,
}

impl<'a> EvalState<'a> {
    fn new(messages: &'a [ChatMessage]) -> Self {
        Self {
            messages,
            loop_index: None,
            vars: std::collections::HashMap::new(),
        }
    }

    fn for_loop(messages: &'a [ChatMessage], index: usize) -> Self {
        Self {
            messages,
            loop_index: Some(index),
            vars: std::collections::HashMap::new(),
        }
    }

    fn current_msg(&self) -> Option<&'a ChatMessage> {
        self.loop_index.and_then(|i| self.messages.get(i))
    }

    fn is_last(&self) -> bool {
        self.loop_index
            .is_some_and(|i| i == self.messages.len().saturating_sub(1))
    }

    fn is_first(&self) -> bool {
        self.loop_index.is_some_and(|i| i == 0)
    }
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
                if let Some(val) = eval_expr(expr, state, ctx) {
                    output.push_str(&val);
                }
                // None (e.g. raise_exception) → silently skip
                i += 1;
            }
            tok if tok.tag_content().is_some() => {
                let content = tok.tag_content().unwrap();

                if content.starts_with("for ") && content.contains(" in messages") {
                    let body_start = i + 1;
                    let end = find_endfor(ctx.tokens, body_start)?;

                    for (idx, _msg) in ctx.messages.iter().enumerate() {
                        let mut loop_state = EvalState::for_loop(ctx.messages, idx);
                        eval_block(ctx, body_start, output, &mut loop_state)?;
                    }
                    i = end + 1;
                } else if content.starts_with("set ") {
                    // {% set var = expr %}
                    if let Some((var, val_expr)) = parse_set_tag(content) {
                        if let Some(val) = eval_expr(val_expr, state, ctx) {
                            state.vars.insert(var.to_string(), val);
                        }
                    }
                    i += 1;
                } else if content.starts_with("if ") {
                    i = eval_if_chain(ctx, i, output, state)?;
                } else if content == "endfor"
                    || content == "endif"
                    || content.starts_with("elif ")
                    || content == "else"
                {
                    return Some(i);
                } else {
                    // Unknown tag — skip silently
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

/// Parse `set var = expr` from a {% set %} tag.
fn parse_set_tag(content: &str) -> Option<(&str, &str)> {
    let rest = content.strip_prefix("set ")?.trim();
    let eq_pos = rest.find('=')?;
    let var = rest[..eq_pos].trim();
    let expr = rest[eq_pos + 1..].trim();
    Some((var, expr))
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
    let matched = eval_condition(condition, state, ctx);

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
            let cond_result = eval_condition(cond, state, ctx);

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
/// Returns `None` for undefined values or raise_exception (silent skip).
fn eval_expr(expr: &str, state: &EvalState, ctx: &EvalCtx) -> Option<String> {
    let expr = expr.trim();

    // raise_exception(...) — silently skip
    if expr.starts_with("raise_exception(") {
        return None;
    }

    // Handle string concatenation with +
    if contains_plus_outside_strings(expr) {
        let parts = split_on_plus(expr);
        let mut result = String::new();
        for part in parts {
            result.push_str(&eval_expr(part.trim(), state, ctx)?);
        }
        return Some(result);
    }

    // Handle | filter (trim, tojson, etc.) — filter binds tighter than +
    if let Some((base, filter)) = split_filter(expr) {
        let val = eval_expr(base, state, ctx)?;
        return Some(apply_filter(&val, filter));
    }

    // String literal: 'foo' or "foo" (with escape sequence support)
    if let Some(s) = parse_string_literal(expr) {
        return Some(s);
    }

    // Numeric literal
    if !expr.is_empty() && expr.bytes().all(|b| b.is_ascii_digit()) {
        return Some(expr.to_string());
    }

    // Special tokens
    match expr {
        "bos_token" => return Some(ctx.bos_token.to_string()),
        "eos_token" => return Some(ctx.eos_token.to_string()),
        _ => {}
    }

    // Loop variables
    if expr == "loop.index0" {
        return state.loop_index.map(|i| i.to_string());
    }

    // Modulo: X % N
    if let Some((left, right)) = split_outside_strings(expr, '%') {
        let lv = eval_expr(left, state, ctx)?.parse::<i64>().ok()?;
        let rv = right.parse::<i64>().ok()?;
        return Some((lv % rv).to_string());
    }

    // messages[N]['field'] or messages[N].field — indexed message access
    if let Some(val) = eval_messages_index(expr, ctx.messages) {
        return Some(val);
    }

    // Current message field access (inside for loop)
    if let Some(msg) = state.current_msg() {
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

    // Variable lookup (from {% set %})
    if let Some(val) = state.vars.get(expr) {
        return Some(val.clone());
    }

    // Parenthesized condition used as expression: (A == B) → "true"/"false"
    if expr.starts_with('(') && expr.ends_with(')') {
        if let Some(inner) = strip_balanced_parens(expr) {
            if inner.contains(" == ")
                || inner.contains(" != ")
                || inner.contains(" and ")
                || inner.contains(" or ")
                || inner.starts_with("not ")
            {
                let result = eval_condition(inner, state, ctx);
                return Some(if result { "true" } else { "false" }.to_string());
            }
        }
    }

    // Undefined → None (falsy in conditions)
    None
}

/// Evaluate messages[N]['field'] or messages[N].field
fn eval_messages_index(expr: &str, messages: &[ChatMessage]) -> Option<String> {
    let rest = expr.strip_prefix("messages[")?;
    let bracket_end = rest.find(']')?;
    let idx: usize = rest[..bracket_end].trim().parse().ok()?;
    let msg = messages.get(idx)?;
    let after = rest[bracket_end + 1..].trim();

    if after == "['role']" || after == "[\"role\"]" || after == ".role" {
        return Some(role_str(&msg.role).to_string());
    }
    if after == "['content']" || after == "[\"content\"]" || after == ".content" {
        return Some(msg.content.clone());
    }

    None
}

/// Split on `| filter_name` outside strings (first occurrence).
fn split_filter(expr: &str) -> Option<(&str, &str)> {
    let mut in_single = false;
    let mut in_double = false;
    for (i, ch) in expr.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '|' if !in_single && !in_double => {
                return Some((expr[..i].trim(), expr[i + 1..].trim()));
            }
            _ => {}
        }
    }
    None
}

fn apply_filter(val: &str, filter: &str) -> String {
    match filter {
        "trim" => val.trim().to_string(),
        "tojson" => {
            use std::fmt::Write;
            let mut out = String::with_capacity(val.len() + 2);
            out.push('"');
            for c in val.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => {
                        write!(out, "\\u{:04x}", c as u32).unwrap();
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        "upper" => val.to_uppercase(),
        "lower" => val.to_lowercase(),
        _ => val.to_string(),
    }
}

/// Split on a character outside strings (returns trimmed halves).
fn split_outside_strings(expr: &str, sep: char) -> Option<(&str, &str)> {
    let mut in_single = false;
    let mut in_double = false;
    for (i, ch) in expr.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            c if c == sep && !in_single && !in_double => {
                return Some((expr[..i].trim(), expr[i + 1..].trim()));
            }
            _ => {}
        }
    }
    None
}

// ── Condition evaluator ──

/// Evaluate a condition expression (for {% if %} / {% elif %}).
fn eval_condition(condition: &str, state: &EvalState, ctx: &EvalCtx) -> bool {
    let condition = condition.trim();

    if condition.is_empty() {
        return false;
    }

    // Strip balanced outer parentheses: (X) → X
    if condition.starts_with('(') {
        if let Some(inner) = strip_balanced_parens(condition) {
            return eval_condition(inner, state, ctx);
        }
    }

    // OR (lowest precedence)
    if let Some((left, right)) = split_condition_op(condition, " or ") {
        return eval_condition(left, state, ctx) || eval_condition(right, state, ctx);
    }

    // AND
    if let Some((left, right)) = split_condition_op(condition, " and ") {
        return eval_condition(left, state, ctx) && eval_condition(right, state, ctx);
    }

    // NOT prefix (must come after or/and splitting)
    if let Some(inner) = condition.strip_prefix("not ") {
        return !eval_condition(inner.trim(), state, ctx);
    }

    // Comparisons: ==, !=
    if let Some((left, right)) = split_condition_op(condition, " == ") {
        let lv = eval_expr(left.trim(), state, ctx).unwrap_or_default();
        let rv = eval_expr(right.trim(), state, ctx).unwrap_or_default();
        return lv == rv;
    }
    if let Some((left, right)) = split_condition_op(condition, " != ") {
        let lv = eval_expr(left.trim(), state, ctx).unwrap_or_default();
        let rv = eval_expr(right.trim(), state, ctx).unwrap_or_default();
        return lv != rv;
    }

    // Special boolean names
    if condition == "add_generation_prompt" {
        return ctx.add_generation_prompt;
    }
    if condition == "loop.last" {
        return state.is_last();
    }
    if condition == "loop.first" {
        return state.is_first();
    }

    // Truthiness of arbitrary expression
    match eval_expr(condition, state, ctx) {
        Some(val) => !val.is_empty() && val != "false" && val != "0",
        None => false, // undefined → falsy
    }
}

/// Split a condition string on an operator, respecting strings and parentheses.
fn split_condition_op<'a>(condition: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut depth: i32 = 0;
    let mut in_single = false;
    let mut in_double = false;
    let bytes = condition.as_bytes();
    let op_bytes = op.as_bytes();

    if bytes.len() < op_bytes.len() {
        return None;
    }

    for i in 0..=bytes.len() - op_bytes.len() {
        let ch = bytes[i];
        match ch {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'(' if !in_single && !in_double => depth += 1,
            b')' if !in_single && !in_double => depth -= 1,
            _ => {}
        }
        if depth == 0 && !in_single && !in_double && bytes[i..].starts_with(op_bytes) {
            return Some((&condition[..i], &condition[i + op.len()..]));
        }
    }
    None
}

/// Strip balanced outer parentheses: "(X)" → "X" if parens match.
fn strip_balanced_parens(s: &str) -> Option<&str> {
    if !s.starts_with('(') || !s.ends_with(')') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let mut depth: i32 = 0;
    let mut in_single = false;
    let mut in_double = false;
    for ch in inner.chars() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' if !in_single && !in_double => depth += 1,
            ')' if !in_single && !in_double => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        Some(inner)
    } else {
        None
    }
}

// ── String helpers ──

fn parse_string_literal(s: &str) -> Option<String> {
    if (s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')) {
        let inner = &s[1..s.len() - 1];
        Some(unescape_string(inner))
    } else {
        None
    }
}

/// Unescape Jinja2 string escape sequences: \n, \t, \\, \', \"
fn unescape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('\'') => result.push('\''),
                Some('"') => result.push('"'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }
    result
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

// ── Public utility functions ──

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

/// Build a Gemma-formatted prompt (for Gemma 1/2 models).
///
/// Gemma uses `<start_of_turn>role\ncontent<end_of_turn>` format.
pub fn gemma_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<bos>");
    for msg in messages {
        let role = match msg.role {
            Role::System | Role::User => "user",
            Role::Assistant => "model",
            Role::Tool => "user",
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }
    prompt.push_str("<start_of_turn>model\n");
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
        // Template failed — try architecture-specific fallbacks before ChatML
        if tmpl.contains("start_of_turn") {
            tracing::warn!(
                fallback = "gemma",
                "DIAG: chat template failed, using gemma fallback"
            );
            return gemma_fallback(messages);
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
            if name_lower.contains("gemma") {
                tracing::debug!(
                    model_name = name,
                    fallback = "gemma",
                    "DIAG: no chat template, using gemma fallback"
                );
                return gemma_fallback(messages);
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

    fn user_only_messages() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
        }]
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
        let msgs = user_only_messages();
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
        let msgs = user_only_messages();
        let result = apply_chat_template(template, &msgs, "<s>", "</s>", true).unwrap();
        assert_eq!(result, "<s>Hi</s>".replace("Hi", "Hello"));
    }

    #[test]
    fn zephyr_tinyllama_template() {
        // TinyLlama / Zephyr uses `loop.last and add_generation_prompt`
        let template = r#"{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>
' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"#;
        let msgs = user_only_messages();
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
        let msgs = user_only_messages();
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

    // ── New tests for enhanced parser ──

    #[test]
    fn set_variable() {
        let template = "{% for message in messages %}{% if message['role'] == 'assistant' %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ role }}: {{ message.content }}\n{% endfor %}";
        let msgs = vec![
            ChatMessage {
                role: Role::User,
                content: "Hi".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hey".into(),
                images: vec![],
            },
        ];
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert!(result.contains("user: Hi"), "Got: {:?}", result);
        assert!(result.contains("model: Hey"), "Got: {:?}", result);
    }

    #[test]
    fn trim_filter() {
        let template = "{% for message in messages %}{{ message.content | trim }}{% endfor %}";
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "  Hello  ".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn messages_index_access() {
        // Access messages[0] outside of loop
        let template =
            "{% if messages[0]['role'] == 'system' %}SYS:{{ messages[0]['content'] }}{% endif %}DONE";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert!(result.contains("SYS:You are helpful."), "Got: {:?}", result);
    }

    #[test]
    fn messages_index_no_system() {
        let template = "{% if messages[0]['role'] == 'system' %}SYS{% else %}NO_SYS{% endif %}";
        let msgs = user_only_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "NO_SYS");
    }

    #[test]
    fn undefined_variable_is_falsy() {
        // `tools` is undefined, should be falsy
        let template = "{% if tools %}TOOLS{% else %}NO_TOOLS{% endif %}";
        let msgs = user_only_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "NO_TOOLS");
    }

    #[test]
    fn or_and_precedence() {
        // or has lower precedence than and
        // true or (false and false) → true
        let template = "{% for message in messages %}{% if message.role == 'user' or message.role == 'system' and not loop.first %}MATCH{% else %}SKIP{% endif %}{% endfor %}";
        let msgs = user_only_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "MATCH");
    }

    #[test]
    fn raise_exception_ignored() {
        let template = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('no system') }}{% endif %}OK";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        // raise_exception produces no output but doesn't abort
        assert!(result.ends_with("OK"), "Got: {:?}", result);
    }

    #[test]
    fn expression_trim_markers() {
        // {{- trims whitespace before, -}} trims whitespace after
        let template = "  hello  {{- ' world' }}  ";
        let result = apply_chat_template(template, &[], "", "", false).unwrap();
        assert_eq!(result, "  hello world  ");
    }

    #[test]
    fn string_escape_sequences() {
        let template = "{{ 'hello\\nworld' }}";
        let result = apply_chat_template(template, &[], "", "", false).unwrap();
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn loop_index0() {
        let template = "{% for message in messages %}{{ loop.index0 }}{% endfor %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "01");
    }

    #[test]
    fn not_loop_first() {
        let template = "{% for message in messages %}{% if not loop.first %},{% endif %}{{ message.content }}{% endfor %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "", false).unwrap();
        assert_eq!(result, "You are helpful.,Hello");
    }

    #[test]
    fn gemma2_actual_template() {
        // The actual Gemma-2 template from GGUF (simplified — no raise_exception assertions)
        let template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}";
        let msgs = vec![ChatMessage {
            role: Role::User,
            content: "What is 2+2?".into(),
            images: vec![],
        }];
        let result = apply_chat_template(template, &msgs, "<bos>", "<eos>", true).unwrap();
        assert!(
            result.starts_with("<bos>"),
            "Should start with bos_token, got: {:?}",
            result
        );
        assert!(
            result.contains("<start_of_turn>user\nWhat is 2+2?<end_of_turn>"),
            "Should contain user turn, got: {:?}",
            result
        );
        assert!(
            result.ends_with("<start_of_turn>model\n"),
            "Should end with model turn, got: {:?}",
            result
        );
    }

    #[test]
    fn gemma2_user_assistant_alternation() {
        let template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('roles must alternate') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}";
        let msgs = vec![
            ChatMessage {
                role: Role::User,
                content: "Hi".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hey".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::User,
                content: "How are you?".into(),
                images: vec![],
            },
        ];
        let result = apply_chat_template(template, &msgs, "<bos>", "<eos>", true).unwrap();
        assert!(
            result.contains("<start_of_turn>user\nHi<end_of_turn>"),
            "Got: {:?}",
            result
        );
        assert!(
            result.contains("<start_of_turn>model\nHey<end_of_turn>"),
            "Got: {:?}",
            result
        );
        assert!(
            result.contains("<start_of_turn>user\nHow are you?<end_of_turn>"),
            "Got: {:?}",
            result
        );
    }

    #[test]
    fn qwen25_actual_template_no_tools() {
        // Qwen2.5's template (the non-tools path). Uses {{- -}} trim, messages[0] access,
        // and `not message.tool_calls` (undefined → true).
        let template = concat!(
            "{%- if tools %}\n",
            "    {{- 'TOOLS_BLOCK' }}\n",
            "{%- else %}\n",
            "    {%- if messages[0]['role'] == 'system' %}\n",
            "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n",
            "    {%- else %}\n",
            "        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endif %}\n",
            "{%- for message in messages %}\n",
            "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n",
            "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endfor %}\n",
            "{%- if add_generation_prompt %}\n",
            "    {{- '<|im_start|>assistant\\n' }}\n",
            "{%- endif %}\n",
        );
        let msgs = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".into(),
                images: vec![],
            },
            ChatMessage {
                role: Role::User,
                content: "What is 2+2?".into(),
                images: vec![],
            },
        ];
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert!(
            result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"),
            "Should contain system message, got: {:?}",
            result
        );
        assert!(
            result.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"),
            "Should contain user message, got: {:?}",
            result
        );
        assert!(
            result.ends_with("<|im_start|>assistant\n"),
            "Should end with assistant prompt, got: {:?}",
            result
        );
        // Should NOT contain the tools block
        assert!(
            !result.contains("TOOLS_BLOCK"),
            "Should not have tools block"
        );
    }

    #[test]
    fn qwen25_no_system_message() {
        // When there's no system message, Qwen injects a default
        let template = concat!(
            "{%- if tools %}\n",
            "    {{- 'TOOLS' }}\n",
            "{%- else %}\n",
            "    {%- if messages[0]['role'] == 'system' %}\n",
            "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n",
            "    {%- else %}\n",
            "        {{- '<|im_start|>system\\nDefault system.<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endif %}\n",
            "{%- for message in messages %}\n",
            "    {%- if (message.role == \"user\") or (message.role == \"assistant\" and not message.tool_calls) %}\n",
            "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endfor %}\n",
            "{%- if add_generation_prompt %}\n",
            "    {{- '<|im_start|>assistant\\n' }}\n",
            "{%- endif %}\n",
        );
        let msgs = user_only_messages();
        let result = apply_chat_template(template, &msgs, "", "", true).unwrap();
        assert!(
            result.contains("<|im_start|>system\nDefault system.<|im_end|>"),
            "Should contain default system message, got: {:?}",
            result
        );
        assert!(
            result.contains("<|im_start|>user\nHello<|im_end|>"),
            "Got: {:?}",
            result
        );
    }

    #[test]
    fn phi35_actual_template() {
        // Phi-3.5's actual template
        let template = "{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}";
        let msgs = test_messages();
        let result = apply_chat_template(template, &msgs, "", "<|endoftext|>", true).unwrap();
        assert!(
            result.contains("<|system|>\nYou are helpful.<|end|>"),
            "Got: {:?}",
            result
        );
        assert!(
            result.contains("<|user|>\nHello<|end|>"),
            "Got: {:?}",
            result
        );
        assert!(result.ends_with("<|assistant|>\n"), "Got: {:?}", result);
    }
}
