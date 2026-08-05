use std::cell::Cell;

use crate::types::{ChatMessage, Role};

use super::parser::Token;

/// Maximum depth for recursive template evaluation. Chat templates come from
/// peer-supplied GGUF metadata; without a depth cap, a crafted template with
/// deeply-nested parens / for-in-for / if-in-if blocks can overflow the worker
/// thread's stack. 256 is generous for any reasonable chat template (real-world
/// templates rarely exceed depth 5).
pub(super) const MAX_TEMPLATE_DEPTH: u32 = 256;

/// Hard cap on the rendered template output length. The depth guard alone
/// stops infinite recursion but doesn't stop heap amplification: e.g.
/// a chain of `{% set x = x + x %}` (or any string-doubling expression)
/// allocates 2^depth-sized strings before the depth cap fires. Real
/// chat templates render to a few KB at most. Any output past this cap
/// is a footgun (or an attack via a poisoned model's tokenizer.json).
pub(super) const MAX_TEMPLATE_OUTPUT: usize = 4 * 1024 * 1024;

pub(super) struct EvalCtx<'a> {
    pub(super) tokens: &'a [Token],
    pub(super) messages: &'a [ChatMessage],
    pub(super) bos_token: &'a str,
    pub(super) eos_token: &'a str,
    pub(super) add_generation_prompt: bool,
    /// Current recursion depth — `Cell` so we can borrow `&EvalCtx`
    /// throughout while still mutating the counter.
    pub(super) depth: Cell<u32>,
}

impl EvalCtx<'_> {
    /// RAII helper: increments depth, returns a guard that decrements on drop.
    /// Returns `None` if `MAX_TEMPLATE_DEPTH` would be exceeded.
    pub(super) fn enter(&self) -> Option<DepthGuard<'_>> {
        let cur = self.depth.get();
        if cur >= MAX_TEMPLATE_DEPTH {
            return None;
        }
        self.depth.set(cur + 1);
        Some(DepthGuard { cell: &self.depth })
    }
}

pub(super) struct DepthGuard<'a> {
    cell: &'a Cell<u32>,
}

impl Drop for DepthGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.cell.get().saturating_sub(1));
    }
}

/// Mutable evaluation state (changes per loop iteration, accumulates variables).
pub(super) struct EvalState<'a> {
    messages: &'a [ChatMessage],
    loop_index: Option<usize>,
    vars: std::collections::HashMap<String, String>,
    /// Names bound to the message list itself via `{% set x = messages %}`.
    ///
    /// These can't live in `vars`, which holds strings — the message list has
    /// no string value, so `eval_expr("messages")` returns `None` and the
    /// binding would simply be dropped.
    ///
    /// The value is the index the list starts at, so
    /// `{% set messages = messages[1:] %}` records 1 and a later loop skips the
    /// message the template already emitted itself. Dropping that offset is how
    /// every Llama-3 system prompt came to be rendered TWICE.
    msg_aliases: std::collections::HashMap<String, usize>,
}

impl<'a> EvalState<'a> {
    pub(super) fn new(messages: &'a [ChatMessage]) -> Self {
        Self {
            messages,
            loop_index: None,
            vars: std::collections::HashMap::new(),
            msg_aliases: std::collections::HashMap::new(),
        }
    }

    fn for_loop(messages: &'a [ChatMessage], index: usize) -> Self {
        Self {
            messages,
            loop_index: Some(index),
            vars: std::collections::HashMap::new(),
            msg_aliases: std::collections::HashMap::new(),
        }
    }

    /// Resolve an expression to a message-list slice, returning the index the
    /// slice starts at. `None` means the expression is not the message list.
    ///
    /// Handles the literal name, any alias bound by `{% set %}`, and a trailing
    /// `[N:]` on either — offsets compose, so binding `x = messages[1:]` and
    /// then looping `x[1:]` starts at 2.
    fn messages_offset(&self, expr: &str) -> Option<usize> {
        let (base, skip) = split_message_ref(expr)?;
        // The alias map wins over the literal name: Llama-3's template rebinds
        // `messages` to `messages[1:]`, so consulting the literal first would
        // hand back offset 0 and undo the slice.
        let base_offset = match self.msg_aliases.get(base) {
            Some(offset) => *offset,
            None if base == "messages" => 0,
            None => return None,
        };
        Some(base_offset + skip)
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
pub(super) fn eval_block(
    ctx: &EvalCtx,
    start: usize,
    output: &mut String,
    state: &mut EvalState,
) -> Option<usize> {
    let _depth_guard = ctx.enter()?;
    let mut i = start;
    while i < ctx.tokens.len() {
        // SEC: bail out if output exceeds the hard cap. A chain of
        // `{% set x = x + x %}` doublings allocates exponentially-sized
        // strings before MAX_TEMPLATE_DEPTH fires — return early so the
        // worker can't be coerced into multi-GB allocations by a poisoned
        // model's tokenizer.json.
        if output.len() > MAX_TEMPLATE_OUTPUT {
            tracing::warn!(
                output_len = output.len(),
                cap = MAX_TEMPLATE_OUTPUT,
                "Chat template output exceeded hard cap — truncating"
            );
            return Some(ctx.tokens.len());
        }
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

                if let Some(offset) =
                    parse_for_iterable(content).and_then(|it| state.messages_offset(it))
                {
                    let body_start = i + 1;
                    let end = find_endfor(ctx.tokens, body_start)?;

                    // Iterate the SLICE, and give the body a state scoped to it
                    // so `loop.first` / `loop.last` are relative to what is
                    // actually being walked.
                    let slice = ctx.messages.get(offset..).unwrap_or(&[]);
                    for idx in 0..slice.len() {
                        let mut loop_state = EvalState::for_loop(slice, idx);
                        eval_block(ctx, body_start, output, &mut loop_state)?;
                    }
                    i = end + 1;
                } else if content.starts_with("set ") {
                    // {% set var = expr %}
                    if let Some((var, val_expr)) = parse_set_tag(content) {
                        // `{% set x = messages %}` binds a name to the message
                        // list, which has no string value — record the alias so
                        // a later `{% for message in x %}` is recognised as a
                        // message loop, along with where that list starts.
                        if let Some(offset) = state.messages_offset(val_expr) {
                            state.msg_aliases.insert(var.to_string(), offset);
                        } else if let Some(val) = eval_expr(val_expr, state, ctx) {
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

/// Extract the iterable name from a `{% for <var> in <iterable> %}` tag.
///
/// Returns the leading identifier only, so a slice or filter (`messages[1:]`,
/// `messages | reverse`) still resolves to `messages` and drives the loop as it
/// did before this helper existed. Neither is honoured — the loop always walks
/// every message — but that is pre-existing behaviour, and iterating all
/// messages beats emitting nothing.
///
/// This exists because matching the iterable by substring (`content.contains(
/// " in messages")`) silently failed on the aliased form every official
/// Llama-3.x GGUF ships:
///
/// ```jinja
/// {% set loop_messages = messages %}{% for message in loop_messages %}
/// ```
///
/// `" in loop_messages"` does not contain `" in messages"`, so the loop was
/// treated as an unknown tag, the body evaluated once with no current message,
/// and `apply_chat_template` returned `None`. Callers then fell back to
/// ChatML — the wrong prompt format for Llama-3 — which is why those models
/// emitted `<|im_end|>` and other ChatML markers into replies
/// (external report 2026-07-25, live-confirmed 2026-07-26).
fn parse_for_iterable(content: &str) -> Option<&str> {
    let rest = content.strip_prefix("for ")?;
    let (_var, iterable) = rest.split_once(" in ")?;
    let iterable = iterable.trim();
    (!iterable.is_empty()).then_some(iterable)
}

/// Split a message-list reference into its base name and how many leading
/// messages a `[N:]` slice drops.
///
/// `messages` → `("messages", 0)`, `messages[1:]` → `("messages", 1)`,
/// `loop_messages | reverse` → `("loop_messages", 0)`.
///
/// Anything other than a plain `[N:]` suffix yields an offset of 0: a filter we
/// do not implement is better applied as identity than not recognised at all,
/// since failing to recognise the loop drops every message (the ChatML fallback
/// this parser exists to prevent). `[N:]` specifically must be honoured because
/// templates use it to remove a message they have already placed by hand —
/// ignoring it renders that message twice.
fn split_message_ref(expr: &str) -> Option<(&str, usize)> {
    let expr = expr.trim();
    let end = expr
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(expr.len());
    let base = &expr[..end];
    if base.is_empty() {
        return None;
    }
    let mut rest = expr[end..].trim();
    let mut skip = 0usize;
    if let Some(after) = rest.strip_prefix('[') {
        // Only a whole-tail `[N:]`. `messages[0]['content']` indexes a single
        // message and must NOT be mistaken for the list, or the expression is
        // bound as an alias instead of being evaluated to its string value.
        let (start, tail) = after.split_once(':')?;
        rest = tail.trim_start().strip_prefix(']')?.trim();
        skip = start.trim().parse::<usize>().ok()?;
    }
    // Only a filter chain may follow the name or slice.
    if !rest.is_empty() && !rest.starts_with('|') {
        return None;
    }
    Some((base, skip))
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
    if let Some((base, filter)) = split_outside_strings(expr, '|') {
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

    // Modulo: X % N. The chat template is peer-supplied (GGUF tokenizer.chat_template
    // field), so guard against `% 0` which would panic in debug and is undefined in
    // release for i64.
    if let Some((left, right)) = split_outside_strings(expr, '%') {
        let lv = eval_expr(left, state, ctx)?.parse::<i64>().ok()?;
        let rv = right.parse::<i64>().ok()?;
        if rv == 0 {
            return None;
        }
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
    let _depth_guard = match ctx.enter() {
        Some(g) => g,
        None => return false,
    };
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
