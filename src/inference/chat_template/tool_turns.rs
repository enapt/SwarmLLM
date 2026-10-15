//! A conversation's tool turns, as the template in hand can show them (#283).
//!
//! An assistant turn's calls and the tool turns answering them reach
//! [`super::build_prompt`] as structure (`ChatMessage::tool`, HuggingFace's
//! chat format). A template that renders them is handed them as they are — the
//! shape its author wrote against. One that does not would show the model an
//! empty assistant turn where its call was, and a tool turn it may refuse
//! outright (Gemma 2 and Mistral raise on any role but user and assistant). For
//! it, the calls are written into the turn as the `{"tool_calls": …}` JSON the
//! prose tool description asks the model to write, and each answer becomes a
//! user turn — llama.cpp's polyfills for the same gap (minja's
//! `chat_template::apply`: `polyfill_tool_calls`, `polyfill_tool_responses`).
//!
//! Which template does what is found by RENDERING two probe conversations, once
//! per template, as minja's `chat_template_caps` does: a template is a program,
//! and whether its text mentions `tool_calls` says nothing about whether a call
//! reaches the prompt. The needle is checked without quotes, because Gemma 4
//! writes its argument keys bare (`argument_needle:`), where minja's quoted
//! check would call it unable.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::types::{ChatMessage, Role, ToolCallFunction, ToolCallTurn, ToolTurn};

/// What a renderer does with tool turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ToolTurnSupport {
    /// It renders an assistant turn's `tool_calls`.
    pub calls: bool,
    /// It renders a `tool` turn's content.
    pub responses: bool,
}

impl ToolTurnSupport {
    /// The built-in fallback renderers: they read no calls, and map a tool turn
    /// themselves.
    pub(super) const FALLBACK: Self = Self {
        calls: false,
        responses: true,
    };
}

/// An argument KEY only a rendered call can put in the prompt.
const ARGUMENT_NEEDLE: &str = "argument_needle";
/// A tool answer only a rendered tool turn can put in the prompt.
const RESPONSE_NEEDLE: &str = "response_needle";
/// Templates whose answer is kept; a node serves a handful of models.
const KEPT_TEMPLATES: usize = 64;

/// What `template` does with tool turns — probed once per template text.
pub(super) fn support_of(template: &str) -> ToolTurnSupport {
    static KNOWN: OnceLock<Mutex<HashMap<blake3::Hash, ToolTurnSupport>>> = OnceLock::new();
    let key = blake3::hash(template.as_bytes());
    let known = KNOWN.get_or_init(Default::default);
    if let Some(support) = known.lock().ok().and_then(|k| k.get(&key).copied()) {
        return support;
    }
    let support = probe(template);
    if let Ok(mut known) = known.lock() {
        if known.len() >= KEPT_TEMPLATES {
            known.clear();
        }
        known.insert(key, support);
    }
    support
}

fn probe(template: &str) -> ToolTurnSupport {
    let turn = |role: Role, content: &str, tool: ToolTurn| ChatMessage {
        role,
        content: content.to_string(),
        images: Vec::new(),
        tool,
    };
    let user = turn(Role::User, "Hey", ToolTurn::default());
    let call = turn(
        Role::Assistant,
        "",
        ToolTurn {
            tool_calls: vec![ToolCallTurn {
                id: "call_1___".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "ipython".to_string(),
                    arguments: serde_json::json!({ ARGUMENT_NEEDLE: "print('Hello, World!')" }),
                },
            }],
            ..ToolTurn::default()
        },
    );
    let answer = turn(
        Role::Tool,
        RESPONSE_NEEDLE,
        ToolTurn {
            tool_call_id: Some("call_1___".to_string()),
            name: Some("ipython".to_string()),
            ..ToolTurn::default()
        },
    );
    let renders = |messages: &[ChatMessage], needle: &str| {
        super::apply_chat_template(template, messages, "", "", false, None)
            .is_some_and(|prompt| prompt.contains(needle))
    };
    ToolTurnSupport {
        calls: renders(&[user.clone(), call.clone()], ARGUMENT_NEEDLE),
        responses: renders(&[user, call, answer], RESPONSE_NEEDLE),
    }
}

/// `messages` as a renderer with `support` can show them, or `None` when
/// nothing changes.
///
/// Whatever the renderer, a tool turn that names its call but not its tool
/// gets the tool's name from that call: Anthropic's results carry none, an
/// OpenAI client may omit it, and templates print it (Gemma 4's says
/// `unknown` without it).
pub(super) fn for_renderer(
    messages: &[ChatMessage],
    support: ToolTurnSupport,
) -> Option<Vec<ChatMessage>> {
    if !messages
        .iter()
        .any(|m| matches!(m.role, Role::Tool) || !m.tool.tool_calls.is_empty())
    {
        return None;
    }
    let mut named: Vec<ChatMessage> = messages.to_vec();
    for i in 0..named.len() {
        if matches!(named[i].role, Role::Tool) && named[i].tool.name.is_none() {
            let id = named[i].tool.tool_call_id.clone();
            named[i].tool.name = id.and_then(|id| name_of_call(&named[..i], &id));
        }
    }
    let mut out: Vec<ChatMessage> = Vec::with_capacity(named.len());
    // Whether `out`'s last turn is an answer this function made a user turn.
    let mut last_was_converted = false;
    for mut m in named {
        if !support.calls && !m.tool.tool_calls.is_empty() {
            let calls = std::mem::take(&mut m.tool.tool_calls);
            m.content = calls_as_text(&m.content, &calls);
        }
        let converted = !support.responses && matches!(m.role, Role::Tool);
        if converted {
            m.content = answer_as_text(&m);
            m.role = Role::User;
            m.tool = ToolTurn::default();
        }
        // Two user turns in a row, one of them made here: one turn, so a
        // template that insists on alternation still renders.
        let joins = (converted || last_was_converted)
            && matches!(m.role, Role::User)
            && out.last().is_some_and(|p| matches!(p.role, Role::User));
        if joins {
            let previous = out.last_mut().expect("checked above");
            previous.content = format!("{}\n\n{}", previous.content, m.content);
            previous.images.append(&mut m.images);
        } else {
            out.push(m);
        }
        last_was_converted = converted || (joins && last_was_converted);
    }
    Some(out)
}

/// The tool an earlier call with this id named.
fn name_of_call(earlier: &[ChatMessage], id: &str) -> Option<String> {
    earlier.iter().rev().find_map(|m| {
        m.tool
            .tool_calls
            .iter()
            .find(|call| call.id == id)
            .map(|call| call.function.name.clone())
    })
}

/// An assistant turn's calls written as the prose description asks a model
/// to write them (`api::tool_parse::format_tool_prompt`), after its text.
fn calls_as_text(content: &str, calls: &[ToolCallTurn]) -> String {
    let calls: Vec<serde_json::Value> = calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "name": call.function.name,
                "arguments": call.function.arguments,
            })
        })
        .collect();
    let json = serde_json::json!({ "tool_calls": calls }).to_string();
    if content.trim().is_empty() {
        json
    } else {
        format!("{}\n{json}", content.trim_end())
    }
}

/// A tool's answer as a user turn can carry it — minja's shape.
fn answer_as_text(answer: &ChatMessage) -> String {
    let mut response = serde_json::Map::new();
    if let Some(name) = &answer.tool.name {
        response.insert("tool".into(), name.clone().into());
    }
    response.insert("content".into(), answer.content.clone().into());
    serde_json::json!({ "tool_response": response }).to_string()
}
