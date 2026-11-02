//! A caller's chat-template variables (#391): settled ONCE at each API's door,
//! carried with the conversation ([`TemplateVars`] on `InferenceRequest` and
//! `DelegatedSplit`), and given to the template UNDER the renderer's own
//! variables ([`for_render`], asked by `apply_chat_template`).
//!
//! The families people run switch their reasoning in the template, by the
//! variable [`ENABLE_THINKING`] (Qwen 3 / 3.5 / 3.6 / 3.8 think unless it is
//! false; Gemma 4 thinks only when it is true). Until 2026-11-01 nothing set
//! it, so a caller asking for a quick answer got the upload's default and paid
//! for the reasoning — on a split, minutes.
//!
//! How the reference servers take it (docs.vllm.ai `features/reasoning_outputs`,
//! read 2026-11-01): a request's `chat_template_kwargs` over the server's
//! defaults; `reasoning_effort` (the Responses API's `reasoning.effort`) sets
//! `enable_thinking` — `"none"` false, any other effort true, unset nothing —
//! and an explicit `enable_thinking` in `chat_template_kwargs` wins. llama.cpp's
//! server has `--chat-template-kwargs` and `--reasoning on|off|auto`, and hands
//! the effort itself to the template as [`REASONING_EFFORT`] — `"none"` leaves
//! it out (`tools/server/server-common.cpp`, master, read 2026-11-02): the
//! variable gpt-oss's template reads (#390).
//!
//! vLLM's two advisories on the field (GHSA-69j4-grxj-j64p, CVE-2025-61620)
//! were kwargs reaching the RENDER CALL's own parameters — `tokenize`,
//! `chat_template` — through a `**kwargs` unpack. Here they reach only the
//! template's variables, and never one the renderer sets itself ([`RESERVED`]).

use serde_json::Value;

use crate::error::SwarmError;
use crate::types::TemplateVars;

/// The variable the families people run read to switch their reasoning.
pub const ENABLE_THINKING: &str = "enable_thinking";

/// The variable gpt-oss's template reads for how hard to think ("Reasoning:
/// low / medium / high" in its system turn; medium when unset).
pub const REASONING_EFFORT: &str = "reasoning_effort";

/// The renderer's own variables and the functions it gives every template: a
/// caller's variable of the same name is left out, so no request replaces the
/// conversation, its tools or the model's turn markers. `context!` would keep
/// ours anyway (the first value a key has wins); this does not lean on that.
const RESERVED: [&str; 7] = [
    "messages",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
    "tools",
    "strftime_now",
    "raise_exception",
];

/// The OpenAI chat door (`validate_chat_request`; the Responses API reaches it
/// with `reasoning.effort` as `reasoning_effort`): `chat_template_kwargs` and
/// `reasoning_effort`, read from the request's catch-all fields — left there,
/// so a cloud provider is still handed them as the caller sent them.
///
/// A `chat_template_kwargs` that is not an object, or a `reasoning_effort`
/// that is not a string, is the caller's error (400), as on vLLM.
pub fn from_openai(
    chat_template_kwargs: Option<&Value>,
    reasoning_effort: Option<&Value>,
) -> Result<Option<TemplateVars>, SwarmError> {
    let mut vars = match chat_template_kwargs {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(_) => {
            return Err(SwarmError::Validation(
                "chat_template_kwargs must be an object, e.g. {\"enable_thinking\": false}".into(),
            ))
        }
    };
    match reasoning_effort {
        None | Some(Value::Null) => {}
        Some(Value::String(effort)) => {
            vars.entry(ENABLE_THINKING)
                .or_insert(Value::Bool(effort != "none"));
            if effort != "none" {
                vars.entry(REASONING_EFFORT)
                    .or_insert(Value::String(effort.clone()));
            }
        }
        Some(_) => {
            return Err(SwarmError::Validation(
                "reasoning_effort must be a string: \"none\", \"low\", \"medium\" or \"high\""
                    .into(),
            ))
        }
    }
    Ok((!vars.is_empty()).then_some(TemplateVars(vars)))
}

/// The Anthropic door: `thinking` — `{"type": "disabled"}` turns reasoning
/// off, `"enabled"` (with its budget) or `"adaptive"` on. Absent, the template
/// decides, as on the OpenAI surface. A shape this does not know asks nothing
/// here; the proxy paths forward it as sent, to be judged where it lands.
pub fn from_anthropic(thinking: Option<&Value>) -> Option<TemplateVars> {
    let on = match thinking?.get("type")?.as_str()? {
        "disabled" => false,
        "enabled" | "adaptive" => true,
        _ => return None,
    };
    let mut vars = serde_json::Map::new();
    vars.insert(ENABLE_THINKING.into(), Value::Bool(on));
    Some(TemplateVars(vars))
}

/// The caller's variables as the render takes them, the reserved names left
/// out; `None` when nothing is left.
/// A name the environment itself defines is left out too — minijinja's own
/// `range`, `dict` and `namespace` among them: a context value is looked up
/// before a global, so `{"namespace": 1}` would fail every template that
/// calls `namespace()` (security review of #391).
pub(super) fn for_render(
    vars: &TemplateVars,
    env: &minijinja::Environment<'_>,
) -> Option<minijinja::Value> {
    let theirs: serde_json::Map<String, Value> = vars
        .0
        .iter()
        .filter(|(name, _)| !RESERVED.contains(&name.as_str()))
        .filter(|(name, _)| !env.globals().any(|(global, _)| global == name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    (!theirs.is_empty()).then(|| minijinja::Value::from_serialize(&theirs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thinking(vars: &Option<TemplateVars>) -> Option<&Value> {
        vars.as_ref().and_then(|v| v.0.get(ENABLE_THINKING))
    }

    /// vLLM's rule, case by case: "none" turns it off, any other effort on,
    /// no effort says nothing, and an explicit `enable_thinking` wins.
    #[test]
    fn reasoning_effort_sets_thinking_unless_the_caller_named_it() {
        let none = from_openai(None, Some(&json!("none"))).unwrap();
        assert_eq!(thinking(&none), Some(&json!(false)));
        let low = from_openai(None, Some(&json!("low"))).unwrap();
        assert_eq!(thinking(&low), Some(&json!(true)));
        // The effort itself, as llama.cpp's server passes it — and never "none".
        let effort =
            |v: &Option<TemplateVars>| v.as_ref().and_then(|v| v.0.get(REASONING_EFFORT).cloned());
        assert_eq!(effort(&low), Some(json!("low")));
        assert_eq!(effort(&none), None);
        assert_eq!(from_openai(None, None).unwrap(), None);
        assert_eq!(from_openai(Some(&json!({})), None).unwrap(), None);

        let explicit = from_openai(
            Some(&json!({"enable_thinking": true, "custom": 3})),
            Some(&json!("none")),
        )
        .unwrap();
        assert_eq!(thinking(&explicit), Some(&json!(true)));
        let named = from_openai(
            Some(&json!({"reasoning_effort": "low"})),
            Some(&json!("high")),
        );
        assert_eq!(
            effort(&named.unwrap()),
            Some(json!("low")),
            "the caller's own kwarg wins"
        );
        assert_eq!(explicit.unwrap().0.get("custom"), Some(&json!(3)));
    }

    #[test]
    fn a_malformed_field_is_the_callers_error() {
        for (kwargs, effort) in [
            (Some(json!([1])), None),
            (Some(json!("enable_thinking=false")), None),
            (None, Some(json!(false))),
        ] {
            let err = from_openai(kwargs.as_ref(), effort.as_ref()).unwrap_err();
            assert!(matches!(err, SwarmError::Validation(_)), "{err:?}");
        }
    }

    #[test]
    fn anthropic_thinking_maps_to_the_same_variable() {
        let off = from_anthropic(Some(&json!({"type": "disabled"})));
        assert_eq!(thinking(&off), Some(&json!(false)));
        let on = from_anthropic(Some(&json!({"type": "enabled", "budget_tokens": 1024})));
        assert_eq!(thinking(&on), Some(&json!(true)));
        assert_eq!(from_anthropic(None), None);
        assert_eq!(
            from_anthropic(Some(&json!({"type": "something-new"}))),
            None
        );
    }

    #[test]
    fn the_renderers_own_names_are_never_a_callers() {
        let mut map = serde_json::Map::new();
        for name in RESERVED {
            map.insert(name.into(), json!("from the caller"));
        }
        let env = minijinja::Environment::new();
        assert!(for_render(&TemplateVars(map.clone()), &env).is_none());
        map.insert(ENABLE_THINKING.into(), json!(false));
        let value = for_render(&TemplateVars(map), &env).unwrap();
        assert_eq!(value.len(), Some(1));
    }

    /// A caller's variable never hides a function the environment gives every
    /// template (the real Qwen 3 template's check is in `tests.rs`).
    #[test]
    fn a_callers_variable_never_hides_an_environment_function() {
        let env = minijinja::Environment::new();
        let mut map = serde_json::Map::new();
        for name in ["namespace", "range", "dict"] {
            map.insert(name.into(), json!(1));
        }
        assert!(for_render(&TemplateVars(map.clone()), &env).is_none());
        map.insert(ENABLE_THINKING.into(), json!(false));
        assert_eq!(for_render(&TemplateVars(map), &env).unwrap().len(), Some(1));
    }
}
