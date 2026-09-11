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
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
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
    let result = apply_chat_template(
        template,
        &msgs,
        "<|begin_of_text|>",
        "<|eot_id|>",
        true,
        None,
    )
    .unwrap();
    assert!(
        result.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>")
    );
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"));
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn mistral_style_template() {
    // Simplified Mistral Instruct template
    let template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "<s>", "</s>", true, None).unwrap();
    assert_eq!(result, "<s>[INST] Hello [/INST]");
}

#[test]
fn build_prompt_with_template() {
    let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
    let msgs = test_messages();
    let result = build_prompt(&msgs, Some(template), "", "", None, None);
    assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn build_prompt_without_template_falls_back() {
    let msgs = test_messages();
    let result = build_prompt(&msgs, None, "", "", None, None);
    assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn dot_notation_works() {
    let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
    assert!(result.contains("system: You are helpful.\n"));
    assert!(result.contains("user: Hello\n"));
    assert!(result.ends_with("assistant: "));
}

#[test]
fn empty_messages() {
    let template = "{% for message in messages %}{{ message.content }}{% endfor %}";
    let result = apply_chat_template(template, &[], "", "", true, None).unwrap();
    assert_eq!(result, "");
}

#[test]
fn no_generation_prompt() {
    let template = "{% for message in messages %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}ASSIST{% endif %}";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert!(!result.contains("ASSIST"));
}

#[test]
fn bos_eos_tokens() {
    let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "<s>", "</s>", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "<s>", "</s>", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
    assert_eq!(result, "ABASSIST");
    // Without generation prompt, ASSIST should NOT appear
    let result2 = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result2, "AB");
}

#[test]
fn else_branch() {
    let template = "{% for message in messages %}{% if message['role'] == 'system' %}SYS:{{ message['content'] }}{% else %}OTHER:{{ message['content'] }}{% endif %}{% endfor %}";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
    assert!(result.contains("SYS:You are helpful."));
    assert!(result.contains("OTHER:Hello"));
}

#[test]
fn zephyr_tinyllama_multiline_template() {
    // The ACTUAL template from the TinyLlama GGUF header (with newlines between tags).
    // HuggingFace renders with trim_blocks=True, lstrip_blocks=True.
    let template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}\n";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "<s>", "</s>", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result, "Hello");
}

#[test]
fn messages_index_access() {
    // Access messages[0] outside of loop
    let template =
        "{% if messages[0]['role'] == 'system' %}SYS:{{ messages[0]['content'] }}{% endif %}DONE";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert!(result.contains("SYS:You are helpful."), "Got: {:?}", result);
}

#[test]
fn messages_index_no_system() {
    let template = "{% if messages[0]['role'] == 'system' %}SYS{% else %}NO_SYS{% endif %}";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result, "NO_SYS");
}

#[test]
fn undefined_variable_is_falsy() {
    // `tools` is undefined, should be falsy
    let template = "{% if tools %}TOOLS{% else %}NO_TOOLS{% endif %}";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result, "NO_TOOLS");
}

#[test]
fn or_and_precedence() {
    // or has lower precedence than and
    // true or (false and false) → true
    let template = "{% for message in messages %}{% if message.role == 'user' or message.role == 'system' and not loop.first %}MATCH{% else %}SKIP{% endif %}{% endfor %}";
    let msgs = user_only_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result, "MATCH");
}

/// `raise_exception` is how Gemma and Mistral templates say they do not support
/// the message they were handed, and it is honoured: the render FAILS and the
/// caller falls back.
///
/// The engine this replaced treated it as a silent skip and rendered on, which
/// produced a turn the model was never trained on — Gemma emitting
/// `<start_of_turn>system`. Declining is the answer the template asked for.
#[test]
fn raise_exception_declines_the_render() {
    let template =
        "{% if messages[0]['role'] == 'system' %}{{ raise_exception('no system') }}{% endif %}OK";
    let msgs = test_messages();
    assert!(
        apply_chat_template(template, &msgs, "", "", false, None).is_none(),
        "a template that raises must decline, so the caller can fall back"
    );
}

#[test]
fn expression_trim_markers() {
    // {{- trims whitespace before, -}} trims whitespace after
    let template = "  hello  {{- ' world' }}  ";
    let result = apply_chat_template(template, &[], "", "", false, None).unwrap();
    assert_eq!(result, "  hello world  ");
}

#[test]
fn string_escape_sequences() {
    let template = "{{ 'hello\\nworld' }}";
    let result = apply_chat_template(template, &[], "", "", false, None).unwrap();
    assert_eq!(result, "hello\nworld");
}

#[test]
fn loop_index0() {
    let template = "{% for message in messages %}{{ loop.index0 }}{% endfor %}";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
    assert_eq!(result, "01");
}

#[test]
fn not_loop_first() {
    let template = "{% for message in messages %}{% if not loop.first %},{% endif %}{{ message.content }}{% endfor %}";
    let msgs = test_messages();
    let result = apply_chat_template(template, &msgs, "", "", false, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "<bos>", "<eos>", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "<bos>", "<eos>", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "", true, None).unwrap();
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
    let result = apply_chat_template(template, &msgs, "", "<|endoftext|>", true, None).unwrap();
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

// ── Fallback selection when a template exists but fails to evaluate ──

/// A template the engine cannot usefully apply. LLaVA GGUFs in the wild ship
/// templates referencing filters/objects our engine doesn't implement; the
/// failure path has to behave like the no-template path, not drop to ChatML.
///
/// An unknown filter renders empty rather than failing evaluation outright,
/// which `apply_chat_template` reports as `None` — see its doc comment.
const UNEVALUABLE_TEMPLATE: &str = "{{ messages | this_filter_does_not_exist }}";

#[test]
fn failed_template_falls_back_to_vicuna_for_llava() {
    let msgs = user_only_messages();
    let result = build_prompt_with_model(
        &msgs,
        Some(UNEVALUABLE_TEMPLATE),
        "",
        "</s>",
        Some("llava-v1.5-7b"),
        None,
    );
    assert!(
        result.contains("USER: ") && result.contains("ASSISTANT:"),
        "expected vicuna format, got: {result:?}"
    );
    assert!(
        !result.contains("<|im_start|>"),
        "must not silently drop to ChatML, got: {result:?}"
    );
}

#[test]
fn failed_template_falls_back_to_gemma_by_model_name() {
    let msgs = user_only_messages();
    let result = build_prompt_with_model(
        &msgs,
        Some(UNEVALUABLE_TEMPLATE),
        "",
        "<eos>",
        Some("gemma-2-2b-it"),
        None,
    );
    assert!(
        result.contains("<start_of_turn>"),
        "expected gemma format, got: {result:?}"
    );
}

#[test]
fn failed_template_body_evidence_beats_model_name() {
    // A gemma-shaped template that fails to evaluate should pick gemma from the
    // template body even when the model name says otherwise. Unclosed `{% for %}`
    // is a genuine structural failure, so nothing renders from the body itself.
    let msgs = user_only_messages();
    let broken_gemma = "{% for m in messages %}<start_of_turn>user";
    assert!(
        apply_chat_template(broken_gemma, &msgs, "", "<eos>", true, None).is_none(),
        "fixture must actually fail to evaluate"
    );
    let result = build_prompt_with_model(
        &msgs,
        Some(broken_gemma),
        "",
        "<eos>",
        Some("llava-7b"),
        None,
    );
    assert!(
        result.contains("<start_of_turn>"),
        "template body should win over model name, got: {result:?}"
    );
}

// ── Empty renders are failures, not success ──

#[test]
fn template_rendering_nothing_is_reported_as_failure() {
    let msgs = user_only_messages();
    // Each of these parses and evaluates cleanly but emits nothing.
    for tmpl in [
        "{{ messages | this_filter_does_not_exist }}",
        "{{ nonexistent_var }}",
        "{% endfor %}",
        "{% if true %}",
        "   ",
    ] {
        assert!(
            apply_chat_template(tmpl, &msgs, "", "</s>", true, None).is_none(),
            "empty render should be a failure: {tmpl:?}"
        );
    }
}

#[test]
fn empty_message_list_may_legitimately_render_empty() {
    // With nothing to render, an empty result is not evidence of a broken
    // template — don't turn it into a failure.
    assert_eq!(
        apply_chat_template(
            "{% for m in messages %}x{% endfor %}",
            &[],
            "",
            "</s>",
            false,
            None,
        ),
        Some(String::new())
    );
}

#[test]
fn failed_template_with_unknown_model_name_still_uses_chatml() {
    let msgs = user_only_messages();
    let result = build_prompt_with_model(
        &msgs,
        Some(UNEVALUABLE_TEMPLATE),
        "",
        "</s>",
        Some("some-unknown-model-7b"),
        None,
    );
    assert!(
        result.contains("<|im_start|>"),
        "expected ChatML for an unrecognised name, got: {result:?}"
    );
}

/// A model's own turn markers must be stop strings, or they reach the user as
/// visible text. Observed live 2026-07-25: a Llama-3.2 model emitted
/// `<|eom_id|><|start_header_id|>assistant<|end_header_id|>` into its reply
/// because only `<|eot_id|>` was recognised.
#[test]
fn extract_stop_strings_covers_llama3_message_and_header_markers() {
    // Representative Llama-3.x template fragment.
    let llama3 = "<|start_header_id|>user<|end_header_id|>\n\n{{ content }}<|eot_id|>\
                  <|start_header_id|>assistant<|end_header_id|>";
    let stops = super::extract_stop_strings(Some(llama3));

    assert!(stops.contains(&"<|eot_id|>".to_string()), "got {stops:?}");
    assert!(
        stops.contains(&"<|start_header_id|>".to_string()),
        "a header marker mid-generation means a hallucinated turn: {stops:?}"
    );

    // `<|eom_id|>` is what Llama 3.1+ emits when it thinks it is calling a
    // tool. Only picked up when the template mentions it.
    let with_eom = format!("{llama3}<|eom_id|>");
    let stops = super::extract_stop_strings(Some(&with_eom));
    assert!(stops.contains(&"<|eom_id|>".to_string()), "got {stops:?}");
}

/// Other families' boundary markers, so one model's fix doesn't regress others.
/// Every path that renders a chat template must also carry that template's stop
/// markers into the sampling params.
///
/// A model ends its turn with a marker, not only the tokenizer's EOS id, so a
/// path that renders the template and forwards params untouched lets the model
/// emit `<|user|>` as visible text and then invent the next turn. It has been
/// missed three times; the third was `remote_generate` — the fast path used
/// whenever ONE peer holds the whole model, which is the normal case for a node
/// that stores nothing itself. Observed on TinyLlama over the network:
/// `'Count from six to ten.\n<|user|> Can you give me a summary…'`.
///
/// Asserted on the PARAMS rather than on generated text: the leak only appears
/// when the model happens to emit a marker, so a test driving a real model
/// passes most runs while the defect is fully present.
#[test]
fn template_stops_reach_the_sampling_params() {
    let tinyllama = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n        {{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n        {{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% endfor %}";

    let bare = swarmllm_types::inference::SamplingParams::default();
    assert!(
        !bare.stop.iter().any(|s| s == "<|user|>"),
        "precondition: default params carry no template stops"
    );

    let with_stops = super::with_template_stops(bare, Some(tinyllama));
    assert!(
        with_stops.stop.iter().any(|s| s == "<|user|>"),
        "the template's own role marker must become a stop, got {:?}",
        with_stops.stop
    );

    // Idempotent: paths that already applied it must not accumulate duplicates.
    let twice = super::with_template_stops(with_stops.clone(), Some(tinyllama));
    assert_eq!(
        twice.stop.len(),
        with_stops.stop.len(),
        "applying twice must not duplicate stops"
    );

    // A caller's own stops survive.
    let mut custom = swarmllm_types::inference::SamplingParams::default();
    custom.stop.push("###".to_string());
    let merged = super::with_template_stops(custom, Some(tinyllama));
    assert!(
        merged.stop.iter().any(|s| s == "###") && merged.stop.iter().any(|s| s == "<|user|>"),
        "caller stops and template stops must coexist, got {:?}",
        merged.stop
    );
}

#[test]
fn extract_stop_strings_covers_other_families() {
    let cases: &[(&str, &str)] = &[
        ("<|im_start|>user\n{{ c }}<|im_end|>", "<|im_end|>"),
        ("<start_of_turn>user\n{{ c }}<end_of_turn>", "<end_of_turn>"),
        ("[INST] {{ c }} [/INST]", "[INST]"),
        ("{{ bos }}{{ c }}</s>", "</s>"),
        ("{{ c }}<|endoftext|>", "<|endoftext|>"),
    ];
    for (template, expected) in cases {
        let stops = super::extract_stop_strings(Some(template));
        assert!(
            stops.contains(&expected.to_string()),
            "template {template:?} should yield {expected:?}, got {stops:?}"
        );
    }
}

/// A marker that could appear in genuine prose must not become a stop string
/// unless this model's template uses it — stopping on one the model legitimately
/// emits would truncate real answers.
///
/// Note the deliberate split: `<|...|>`-style SPECIAL TOKENS are stop strings
/// universally (see `universal_special_tokens_stop_even_when_absent_from_the_template`)
/// because no model emits them as content. `[INST]` and `</s>` are not, because
/// a reply about code or XML plausibly contains them.
#[test]
fn extract_stop_strings_does_not_invent_ambiguous_markers() {
    let chatml_only = "<|im_start|>user\n{{ c }}<|im_end|>";
    let stops = super::extract_stop_strings(Some(chatml_only));
    assert!(!stops.contains(&"</s>".to_string()), "got {stops:?}");
    assert!(!stops.contains(&"[INST]".to_string()), "got {stops:?}");
    assert!(!stops.contains(&"<|user|>".to_string()), "got {stops:?}");
}

/// Phi-4-mini-instruct's tool branch tests `'tools' in message and
/// message['tools'] is not none` — a per-MESSAGE field, not the top-level
/// variable it is handed. So the substring check in `template_renders_tools`
/// passes, the prose description is suppressed as redundant, the branch never
/// fires, and the model is told NOTHING about its tools. Reported from the field
/// 2026-09-10 as "no engagement with the tools array whatsoever".
#[test]
fn a_template_that_ignores_the_tools_variable_still_gets_them_described() {
    let phi4 = "{% for message in messages %}\
        {% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}\
        {{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}\
        {% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}\
        {% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    })];
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "What time is it?".to_string(),
        images: Vec::new(),
    }];
    let prompt = super::build_prompt(&msgs, Some(phi4), "", "", Some("phi-4-mini"), Some(&tools));
    assert!(
        prompt.contains("terminal"),
        "the model must be told its tool exists, got:\n{prompt}"
    );
    assert!(
        prompt.contains("What time is it?"),
        "the question must survive the retry, got:\n{prompt}"
    );
}

/// The retry must not fire for a template that DOES use the variable — Qwen3
/// renders its own `<tool_call>` framing and must keep it, not be handed our
/// prose format on top.
#[test]
fn a_template_that_renders_tools_is_not_second_guessed() {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {"name": "terminal", "parameters": {"type": "object", "properties": {}}},
    })];
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "hi".to_string(),
        images: Vec::new(),
    }];
    let prompt = super::build_prompt(
        &msgs,
        Some(QWEN3_OFFICIAL),
        "",
        "",
        Some("qwen3-8b"),
        Some(&tools),
    );
    assert!(prompt.contains("<tools>"), "got:\n{prompt}");
    assert!(
        !prompt.contains("respond with a JSON object"),
        "the prose fallback must not be added on top:\n{prompt}"
    );
}

/// Phi-3/3.5/4 close every turn with `<|end|>` while their GGUFs declare
/// `<|endoftext|>` as EOS, so nothing stopped the reply there. Reproduced on
/// v0.3.171: Phi-3.5 asked "Say exactly: hello" ran the full 120 tokens,
/// `finish_reason: "length"`, having invented a second assistant turn and a
/// fabricated user turn. The end-of-generation token id is the primary fix; this
/// is what also removes a LEAKED marker from the text.
#[test]
fn a_phi_turn_closer_is_a_stop_string() {
    let phi = "<|system|>\n{{ s }}<|end|>\n<|user|>\n{{ c }}<|end|>\n<|assistant|>\n";
    let stops = super::extract_stop_strings(Some(phi));
    assert!(
        stops.contains(&"<|end|>".to_string()),
        "<|end|> ends every phi turn, got {stops:?}"
    );
}

/// The exclusion that makes the fix above safe. In harmony (gpt-oss) and
/// solar-open, `<|end|>` separates messages inside a reply that is still being
/// written, so stopping on it truncates every such reply at its first message.
#[test]
fn end_is_not_a_stop_where_it_separates_messages() {
    let harmony = "<|start|>system<|message|>{{ s }}<|end|>\
                   <|start|>assistant<|channel|>final<|message|>{{ c }}<|return|>";
    let stops = super::extract_stop_strings(Some(harmony));
    assert!(
        !stops.contains(&"<|end|>".to_string()),
        "<|end|> only separates harmony messages, got {stops:?}"
    );
}

/// No template does NOT mean ChatML. `build_prompt_inner`'s no-template branch
/// asks `fallback_by_model_name` first, so the prompt may be zephyr, llama3,
/// gemma, mistral or vicuna — and this returned ChatML's single marker, leaving
/// every one of those with no stop for the marker it will actually emit.
#[test]
fn no_template_still_stops_on_every_unsafe_marker() {
    let stops = super::extract_stop_strings(None);
    for expected in [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<end_of_turn>",
        "<|endoftext|>",
    ] {
        assert!(
            stops.contains(&expected.to_string()),
            "{expected} must stop even with no template, got {stops:?}"
        );
    }
}

/// Special tokens are never legitimate assistant output, so they must be stop
/// strings even when this model's template doesn't mention them. Reported live
/// 2026-07-25: a Llama-3.2 q8_0 returned `<|im_end|>hello</im_start>` — ChatML
/// markers from a model whose template is not ChatML, which scanning the
/// template alone could never catch.
#[test]
fn universal_special_tokens_stop_even_when_absent_from_the_template() {
    let llama3_only = "<|start_header_id|>user<|end_header_id|>{{ c }}<|eot_id|>";
    let stops = super::extract_stop_strings(Some(llama3_only));
    for expected in [
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<end_of_turn>",
        "<|endoftext|>",
    ] {
        assert!(
            stops.contains(&expected.to_string()),
            "{expected} must stop regardless of template, got {stops:?}"
        );
    }
    // No duplicates: the template scan must not re-add a universal marker.
    let mut sorted = stops.clone();
    sorted.sort();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(before, sorted.len(), "duplicate stop strings: {stops:?}");
}

/// Markers that CAN appear in real prose stay template-gated — stopping on
/// `[INST]` or `</s>` in a model that never emits them would truncate genuine
/// answers about code or XML.
#[test]
fn ambiguous_markers_remain_template_gated() {
    let chatml = "<|im_start|>user\n{{ c }}<|im_end|>";
    let stops = super::extract_stop_strings(Some(chatml));
    assert!(!stops.contains(&"[INST]".to_string()), "got {stops:?}");
    assert!(!stops.contains(&"</s>".to_string()), "got {stops:?}");
}

/// The chat template every official Llama-3.x Instruct GGUF ships, verbatim
/// from `gguf_header.bin` on a live node (2026-07-26).
///
/// It binds the message list to a name first (`{% set loop_messages = messages
/// %}`) and iterates that. Matching the iterable by substring missed the
/// aliased form, so this template evaluated to nothing and callers fell back to
/// ChatML — the wrong format for Llama-3, which is why these models emitted
/// `<|im_end|>` into replies.
#[test]
fn llama3_aliased_message_loop_renders() {
    let tmpl = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}";
    let msgs = user_only_messages();
    let out = apply_chat_template(tmpl, &msgs, "<|begin_of_text|>", "<|eot_id|>", true, None)
        .expect("aliased message loop must render, not fall back to ChatML");

    assert_eq!(
        out,
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    // The ChatML fallback is what this template used to produce by failing.
    assert!(!out.contains("<|im_start|>"));
}

/// An alias is only a message list when it was bound to one. An unrelated
/// `{% set %}` must not turn a foreign loop into a message loop.
///
/// An unrecognised loop has always rendered its body once, inline — that is
/// unchanged here. What must not happen is the body repeating once per
/// message, which is what treating `tools` as a message list would do.
#[test]
fn unrelated_set_does_not_alias_messages() {
    let tmpl = "{% set tools = 'x' %}{% for t in tools %}BODY;{% endfor %}";
    let msgs = test_messages(); // two messages
    let out = apply_chat_template(tmpl, &msgs, "", "", true, None).unwrap();
    assert_eq!(out, "BODY;", "must not repeat per message");
}

/// A `[N:]` slice on the iterable is honoured, and a filter still drives the
/// loop over everything.
///
/// The offset used to be discarded — the loop always walked every message. That
/// is not a harmless approximation: templates slice precisely to drop a message
/// they have already placed by hand, so ignoring it emitted that message TWICE.
/// Every Llama-3 system prompt was duplicated for exactly this reason.
#[test]
fn a_sliced_message_loop_skips_what_the_slice_drops() {
    let msgs = test_messages(); // system, then user
    let sliced = "{% for message in messages[1:] %}{{ message['content'] }};{% endfor %}";
    assert_eq!(
        apply_chat_template(sliced, &msgs, "", "", true, None).unwrap(),
        "Hello;",
        "messages[1:] must skip the first message"
    );

    // `| reverse` is now genuinely applied. The hand-rolled evaluator this
    // replaced treated every filter it did not implement as identity, so this
    // read back in message order; a real engine reverses it.
    let filtered = "{% for message in messages | reverse %}{{ message['content'] }};{% endfor %}";
    assert_eq!(
        apply_chat_template(filtered, &msgs, "", "", true, None).unwrap(),
        "Hello;You are helpful.;"
    );
}

/// Rebinding `messages` to a slice of itself — what every official Llama-3.x
/// template does — must take effect for the loop that follows.
#[test]
fn rebinding_messages_to_its_own_tail_is_honoured() {
    let tmpl = "{%- if messages[0]['role'] == 'system' %}\
                {%- set system_message = messages[0]['content'] %}\
                {%- set messages = messages[1:] %}{%- endif %}\
                SYS:{{ system_message }};\
                {%- for message in messages %}{{ message['content'] }};{%- endfor %}";
    let msgs = test_messages();
    assert_eq!(
        apply_chat_template(tmpl, &msgs, "", "", true, None).unwrap(),
        "SYS:You are helpful.;Hello;",
        "the system message belongs in the header only, not again in the loop"
    );
}

/// The REAL Llama-3.x template, shipped verbatim in every Llama-3.1/3.2 GGUF,
/// rendered against the exact output jinja2 produces for it.
///
/// This is the integration guard for the whole prompt builder: the pieces below
/// are each unit-tested, but only rendering the real thing catches them
/// interacting. Every one of these was wrong at once, and none produced an
/// error — the system message came out twice, comments left blank lines
/// scattered through the prompt, and the date fell back to the hardcoded
/// 26 Jul 2024 written into the template.
///
/// Expected strings were taken from jinja2 (trim_blocks + lstrip_blocks, as
/// HuggingFace renders chat templates), not derived by reading our evaluator.
#[test]
fn the_official_llama3_template_renders_exactly_as_jinja2_does() {
    let tmpl = include_str!("fixtures/llama3_official.jinja");
    let today = chrono::Local::now().format("%d %b %Y").to_string();
    let header =
        format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: {today}\n\n");

    let render = |msgs: &[ChatMessage]| {
        apply_chat_template(tmpl, msgs, "<|begin_of_text|>", "<|eot_id|>", true, None)
            .expect("renders")
    };

    let user = |c: &str| ChatMessage {
        role: Role::User,
        content: c.into(),
        images: vec![],
    };
    let assistant = |c: &str| ChatMessage {
        role: Role::Assistant,
        content: c.into(),
        images: vec![],
    };
    let system = |c: &str| ChatMessage {
        role: Role::System,
        content: c.into(),
        images: vec![],
    };

    assert_eq!(
        render(&[user("Hi there")]),
        format!(
            "{header}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi there<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    );

    // The system message belongs in the header block and NOWHERE else.
    let with_system = render(&[system("You are terse."), user("Hi there")]);
    assert_eq!(
        with_system,
        format!(
            "{header}You are terse.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             Hi there<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    );
    assert_eq!(
        with_system.matches("You are terse.").count(),
        1,
        "the system message was rendered more than once"
    );

    assert_eq!(
        render(&[user("a"), assistant("b"), user("c")]),
        format!(
            "{header}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\na<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\nb<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nc<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    );

    // The template's own hardcoded fallback date must never reach a model.
    assert!(
        !with_system.contains("26 Jul 2024") || today == "26 Jul 2024",
        "rendered the template's fallback date instead of today's"
    );
}

/// A comment's body is dropped, but the whitespace around it must go too.
///
/// Comments were skipped without honouring their trim markers or the
/// lstrip_blocks / trim_blocks defaults, so every `{#- ... #}` a template used
/// to document itself left a blank line behind in the model's prompt.
#[test]
fn a_comment_does_not_leave_its_whitespace_behind() {
    // Expectations taken from jinja2 itself, with the trim_blocks and
    // lstrip_blocks defaults HuggingFace renders chat templates under.
    let msgs = test_messages();
    for (tmpl, want) in [
        ("A\n{#- a comment #}\nB", "AB"),
        ("A\n    {#- a comment -#}\n    B", "AB"),
        // No trim markers: lstrip_blocks removes the indent before the comment
        // and trim_blocks the newline after it, but the newline BEFORE stays.
        ("A\n{# a comment #}\nB", "A\nB"),
        ("A\n  {# c #}\n  B", "A\n  B"),
    ] {
        assert_eq!(
            apply_chat_template(tmpl, &msgs, "", "", true, None).unwrap(),
            want,
            "wrong whitespace around comment in {tmpl:?}"
        );
    }
}

/// Llama-3.x templates ask for `strftime_now` and fall back to a HARDCODED
/// date when it is missing, so reporting it undefined told every Llama-3 model
/// that today was 26 Jul 2024 forever.
#[test]
fn strftime_now_reports_todays_date_not_the_templates_fallback() {
    let msgs = test_messages();
    let tmpl = "{%- if strftime_now is defined %}\
                {{- strftime_now(\"%Y\") }}\
                {%- else %}FALLBACK{%- endif %}";
    let out = apply_chat_template(tmpl, &msgs, "", "", true, None).unwrap();
    assert_ne!(
        out, "FALLBACK",
        "the guard must report strftime_now present"
    );
    let year: i32 = out.parse().expect("a four-digit year");
    assert!((2025..=2100).contains(&year), "implausible year {year}");

    // A format we cannot render must not panic — chrono's Display panics on an
    // unknown specifier, and the format string comes from model metadata.
    let bad = "{{ strftime_now(\"%Q\") }}X";
    assert_eq!(
        apply_chat_template(bad, &msgs, "", "", true, None).unwrap(),
        "X"
    );
}

/// Indexing a single message must still evaluate to that message's field, not
/// be mistaken for a binding to the whole list.
#[test]
fn indexing_one_message_is_not_a_list_binding() {
    let tmpl = "{%- set first = messages[0]['content'] %}{{ first }}";
    let msgs = test_messages();
    assert_eq!(
        apply_chat_template(tmpl, &msgs, "", "", true, None).unwrap(),
        "You are helpful."
    );
}

/// Mistral's official template uses `namespace()`, `selectattr`, and slicing
/// that our evaluator does not implement, so it fails whenever a system message
/// is present. Falling through to ChatML would ask a Mistral model to speak
/// ChatML — the exact failure that leaked `<|im_end|>` markers from Llama-3.
/// It must degrade to Mistral's own format instead.
#[test]
fn mistral_name_fallback_uses_inst_format_not_chatml() {
    let msgs = test_messages(); // system + user
    let (prompt, kind) = super::fallback_by_model_name(&msgs, Some("Mistral-7B-Instruct-v0.3"))
        .expect("mistral must be recognised by name");
    assert_eq!(kind, "mistral");
    // System text folds into the last user turn, as the official template does.
    assert_eq!(prompt, "<s>[INST] You are helpful.\n\nHello[/INST]");
    assert!(!prompt.contains("<|im_start|>"));
}

/// A LLaVA build named after its Mistral base is still a LLaVA model — the
/// vicuna check must win, or vision prompts lose their `<image>` placement.
#[test]
fn llava_mistral_build_still_uses_vicuna() {
    let msgs = user_only_messages();
    let (_, kind) = super::fallback_by_model_name(&msgs, Some("llava-v1.6-mistral-7b"))
        .expect("llava must be recognised");
    assert_eq!(kind, "vicuna");
}

/// Llama-3 gets its own header format rather than ChatML if its template ever
/// fails to evaluate again.
#[test]
fn llama3_name_fallback_uses_header_format() {
    let msgs = test_messages();
    let (prompt, kind) = super::fallback_by_model_name(&msgs, Some("Llama-3.2-1B-Instruct"))
        .expect("llama3 must be recognised by name");
    assert_eq!(kind, "llama3");
    assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"));
    assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    assert!(!prompt.contains("<|im_start|>"));
}

/// Mistral alternation: an assistant turn closes with `</s>` and the next user
/// turn opens a fresh `[INST]`.
#[test]
fn mistral_fallback_multi_turn_alternates() {
    let msgs = vec![
        ChatMessage {
            role: Role::User,
            content: "one".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::Assistant,
            content: "two".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "three".into(),
            images: vec![],
        },
    ];
    let (prompt, _) = super::fallback_by_model_name(&msgs, Some("mistral-7b")).unwrap();
    assert_eq!(prompt, "<s>[INST] one[/INST] two</s>[INST] three[/INST]");
}

/// End-to-end through the real entry point: a Mistral model whose official
/// template our evaluator cannot run must still be prompted in Mistral format.
///
/// This is the assertion that was impossible before the model name was plumbed
/// through — `build_prompt` hardcoded `None`, so every fallback decision on the
/// OpenAI, Anthropic, streaming and router paths collapsed to ChatML.
#[test]
fn real_mistral_template_failure_degrades_to_mistral_not_chatml() {
    // A template that cannot render at all. `namespace()` and slicing USED to
    // be the examples here and are both supported now, so this reaches for the
    // one thing that will always fail: a syntax error. What is under test is
    // the FALLBACK CHOICE, not which construct broke.
    let tmpl = "{%- for message in messages %}{{ message['content'] }}{%- endnope %}";
    let msgs = test_messages();

    // Confirm the premise: this really does fail to render.
    assert!(
        apply_chat_template(tmpl, &msgs, "<s>", "</s>", true, None).is_none(),
        "premise changed — template now renders, revisit this test"
    );

    let prompt = build_prompt(
        &msgs,
        Some(tmpl),
        "<s>",
        "</s>",
        Some("Mistral-7B-Instruct-v0.3"),
        None,
    );
    assert_eq!(prompt, "<s>[INST] You are helpful.\n\nHello[/INST]");
    assert!(
        !prompt.contains("<|im_start|>"),
        "must not fall through to ChatML"
    );
}

// ── Default system message injection ──

fn bare_user() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: Role::User,
        content: "Hi".to_string(),
        images: vec![],
    }]
}

const TINYLLAMA_TMPL: &str = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";

/// TinyLlama answers a bare user question with nothing but a `<|user|>` turn
/// marker; the same question with a system message is answered normally.
#[test]
fn system_message_injected_for_zephyr_template() {
    let out = build_prompt_with_model(
        &bare_user(),
        Some(TINYLLAMA_TMPL),
        "<s>",
        "</s>",
        None,
        None,
    );
    assert!(
        out.contains("<|system|>"),
        "expected an injected system turn, got: {out:?}"
    );
}

/// Gemma and Mistral declare no system role via `raise_exception`. Our
/// evaluator treats that as a silent skip, so injecting would quietly render a
/// turn the model was never trained on rather than failing loudly.
#[test]
fn system_message_never_injected_when_template_raises() {
    let gemma = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}{% endfor %}";
    let out = build_prompt_with_model(&bare_user(), Some(gemma), "<bos>", "<eos>", None, None);
    assert!(
        !out.contains("system"),
        "must not inject into a template that raises on system: {out:?}"
    );
}

/// A template that refuses a system role must still render the model's own
/// prompt, with the system text moved into the first user turn.
///
/// Gemma-2 is the reported case, found by `examples/family_conformance.sh`
/// 2026-09-11: the tool description IS a system message, so every Gemma-2
/// request carrying tools rendered through the `gemma_fallback` instead of the
/// template that shipped with the model.
#[test]
fn a_template_that_raises_on_system_renders_with_it_folded_into_the_user_turn() {
    // The `OWN-TEMPLATE` marker is what makes the last assertion mean anything:
    // `gemma_fallback` emits the same turn markers AND the same generation
    // prompt, so without something only this template can produce, a test
    // asserting those would pass on the fallback it exists to rule out.
    let gemma = "{{ 'OWN-TEMPLATE ' }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}";
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "You have a tool called get_time.".to_string(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "What time is it?".to_string(),
            images: vec![],
        },
    ];
    let out = build_prompt_with_model(&msgs, Some(gemma), "<bos>", "<eos>", None, None);
    assert!(
        out.contains("<start_of_turn>user\nYou have a tool called get_time.\n\nWhat time is it?"),
        "the system text must ride in the first user turn: {out:?}"
    );
    assert!(
        !out.contains("<start_of_turn>system"),
        "a refused role must not be rendered anyway: {out:?}"
    );
    // The distinguishing property: this is the MODEL'S template, not the
    // fallback.
    assert!(
        out.starts_with("OWN-TEMPLATE "),
        "must be the model's own template, not gemma_fallback: {out:?}"
    );
    assert!(
        out.ends_with("<start_of_turn>model\n"),
        "the generation prompt must survive the retry: {out:?}"
    );
}

/// The retry only fires when the template actually refused. A template that
/// renders a system turn keeps rendering one.
#[test]
fn a_template_that_accepts_system_still_gets_its_own_system_turn() {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "You are a pirate.".to_string(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "Hi".to_string(),
            images: vec![],
        },
    ];
    let out = build_prompt_with_model(&msgs, Some(TINYLLAMA_TMPL), "<s>", "</s>", None, None);
    assert!(out.contains("<|system|>"), "got: {out:?}");
    assert!(
        !out.contains("You are a pirate.\n\nHi"),
        "must not fold a system turn the template renders perfectly well: {out:?}"
    );
}

/// A conversation with no user turn has nowhere to put the system text, and
/// losing it silently is worse than the fallback.
#[test]
fn folding_declines_when_there_is_no_user_turn_to_fold_into() {
    let msgs = vec![ChatMessage {
        role: Role::System,
        content: "You are helpful.".to_string(),
        images: vec![],
    }];
    assert!(super::fold_system_into_first_user(&msgs).is_none());
}

/// A caller-supplied system message must never be overridden.
#[test]
fn caller_system_message_is_preserved() {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "You are a pirate.".to_string(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "Hi".to_string(),
            images: vec![],
        },
    ];
    let out = build_prompt_with_model(&msgs, Some(TINYLLAMA_TMPL), "<s>", "</s>", None, None);
    assert!(out.contains("You are a pirate."), "got: {out:?}");
    assert!(!out.contains(DEFAULT_SYSTEM_PROMPT), "got: {out:?}");
}

/// A blank system message renders an empty system turn, which reproduces the
/// original failure — treat it as absent.
#[test]
fn blank_system_message_is_replaced() {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "   ".to_string(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "Hi".to_string(),
            images: vec![],
        },
    ];
    let out = build_prompt_with_model(&msgs, Some(TINYLLAMA_TMPL), "<s>", "</s>", None, None);
    assert!(out.contains(DEFAULT_SYSTEM_PROMPT), "got: {out:?}");
}

/// TinyLlama's name matches none of the other families, so without a Zephyr
/// entry it reached ChatML and was asked a ChatML question — which it answered
/// with a stray `<|user|>` marker and an unrelated question.
#[test]
fn tinyllama_without_a_template_gets_zephyr_not_chatml() {
    let out = build_prompt_with_model(
        &bare_user(),
        None,
        "<s>",
        "</s>",
        Some("tinyllama-1.1b-chat-v1.0.q4-k-m"),
        None,
    );
    assert!(
        out.contains("<|user|>"),
        "expected Zephyr format, got: {out:?}"
    );
    assert!(
        out.trim_end().ends_with("<|assistant|>"),
        "must end with the generation prompt: {out:?}"
    );
    assert!(
        !out.contains("<|im_start|>"),
        "must NOT fall through to ChatML: {out:?}"
    );
    assert!(
        out.contains("<|system|>"),
        "Zephyr models are trained with a system turn: {out:?}"
    );
}

/// A Llama-3 model must keep its own format — "tinyllama" must not be matched
/// by a broad "llama" substring, and vice versa.
#[test]
fn llama3_still_gets_llama3_format() {
    let out = build_prompt_with_model(
        &bare_user(),
        None,
        "<s>",
        "</s>",
        Some("meta-llama-3.1-8b"),
        None,
    );
    assert!(
        out.contains("<|start_header_id|>"),
        "Llama-3 must keep its own format: {out:?}"
    );
    assert!(!out.contains("<|user|>\n"), "must not be Zephyr: {out:?}");
}

/// The prompt shapes a correct template produces MUST pass. A false positive
/// here would put a warning on every healthy request, which is worse than the
/// silence it replaces.
#[test]
fn a_correctly_rendered_prompt_hands_over_to_the_model() {
    use super::prompt_hands_over_to_the_model;
    for good in [
        // ChatML / Qwen
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        // Llama-3
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n",
        // Gemma
        "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
        // Mistral / Llama-2
        "[INST] hi [/INST]",
        // A plain completion prompt has no turn structure at all.
        "Once upon a time",
    ] {
        assert!(
            prompt_hands_over_to_the_model(good),
            "flagged a healthy prompt: {good:?}"
        );
    }
}

/// The failure this exists to name: the generation prompt was not appended, so
/// the model is shown a finished conversation and ends its turn at once.
#[test]
fn a_prompt_that_closes_the_turn_is_caught() {
    use super::prompt_hands_over_to_the_model;
    for bad in [
        "<|im_start|>user\nhi<|im_end|>",
        // Trailing whitespace must not hide it — templates commonly emit a
        // newline after the closing marker.
        "<|im_start|>user\nhi<|im_end|>\n",
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
        "<start_of_turn>user\nhi<end_of_turn>\n",
        "[INST] hi [/INST] answer</s>",
    ] {
        assert!(
            !prompt_hands_over_to_the_model(bad),
            "missed a turn-closing prompt: {bad:?}"
        );
    }
}

/// Both entry points must apply the turn-closing check.
///
/// `build_prompt` is the local fast path; `build_prompt_with_model` is what the
/// router and distributed paths reach through
/// `pipeline::prompt::build_prompt_with_header`. The check lived in the
/// `build_prompt` wrapper for one commit and the router path — which does not
/// call it — was silently exempt. A shared rule implemented in one of two paths
/// is this repo's most repeated defect, so both are pinned here rather than
/// trusting the call graph to stay as it is.
#[test]
fn both_prompt_entry_points_go_through_the_same_renderer() {
    // A template with no generation prompt: renders a closed turn either way.
    // The same template the passing tests in this file use, with ONLY the
    // `{% if add_generation_prompt %}` block removed — so it evaluates
    // successfully and still leaves the turn closed. Using an invalid template
    // instead makes the renderer fall back to ChatML, which appends the opener
    // and quietly stops the test exercising anything.
    //
    // It closes on `</s>` rather than `<|im_end|>` because a ChatML closer is
    // now REPAIRED (`open_the_models_turn_if_the_prompt_closed_it`) — which
    // would leave this test asserting nothing, exactly as its own guard below
    // says. `</s>` names three families at once, so it is deliberately left
    // alone and still reaches the check this test is about.
    let closing = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' \
                   + message['content'] + '</s>' + '\n'}}{% endfor %}";
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "hi".into(),
        images: vec![],
    }];

    let via_wrapper =
        super::build_prompt(&msgs, Some(closing), "<s>", "</s>", Some("qwen2.5"), None);
    let via_inner =
        super::build_prompt_with_model(&msgs, Some(closing), "<s>", "</s>", Some("qwen2.5"), None);

    assert_eq!(
        via_wrapper, via_inner,
        "the two entry points must render identically — a divergence here is what \
         gotchas #169 and #171 were"
    );
    assert!(
        !super::prompt_hands_over_to_the_model(&via_inner),
        "this fixture is supposed to produce a turn-closing prompt; if it does not, \
         the test is no longer exercising the check"
    );
}

/// The official Qwen3 template must at minimum OPEN the assistant's turn.
///
/// Reported against a Qwen3-8B: every request logged "the rendered prompt ends
/// by CLOSING a turn rather than opening the model's", which is our own
/// warning for a prompt that shows the model a finished conversation — it then
/// answers with one token and `finish_reason: "stop"`.
///
/// This template is the most demanding in common use: `namespace()`, a reversed
/// slice `messages[::-1]`, `loop.index0` / `loop.first` / `loop.last`, the
/// `tojson` and `length` filters, `is string` / `is defined` tests, and string
/// methods (`startswith`, `split`, `rstrip`). The renderer is deliberately a
/// subset of Jinja, so the question this pins is not "do we implement all of
/// that" — it is that whatever we DO produce must still open the turn, because
/// a prompt without it cannot get an answer out of any model.
#[test]
fn the_official_qwen3_template_opens_the_assistants_turn() {
    let tmpl = include_str!("fixtures/qwen3_official.jinja");
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "Say OK".into(),
        images: vec![],
    }];

    let rendered = apply_chat_template(tmpl, &msgs, "", "<|im_end|>", true, None);

    // Declining is an acceptable outcome — `build_prompt_with_model` then falls
    // back, and the turn-opening repair covers what the fallback leaves closed.
    // What must not happen is a HALF-render that silently drops the generation
    // prompt, because that reaches the model as a finished conversation.
    if let Some(out) = rendered {
        assert!(
            out.trim_end().ends_with("<|im_start|>assistant"),
            "rendered prompt must open the model's turn, got tail: {:?}",
            &out[out.len().saturating_sub(80)..]
        );
    }
}

/// Opening the model's turn is necessary and not sufficient: the prompt must
/// also still CONTAIN the question.
///
/// The sibling test above accepts any render that ends on
/// `<|im_start|>assistant`, which a render that dropped every message also
/// does. Field-reported on a Qwen3-8B and reproduced here on a 1.7B: three
/// different prompts all reached the model as the same 14 tokens and produced
/// the same reply, because the user's text was not in any of them.
#[test]
fn the_official_qwen3_template_keeps_the_users_question_in_the_prompt() {
    let tmpl = include_str!("fixtures/qwen3_official.jinja");
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "BANANE VIOLETTE MYSTERE 42".into(),
        images: vec![],
    }];

    // Whatever path the prompt takes — the template rendering, or the fallback
    // it declines into — the question has to survive it.
    let prompt = build_prompt_with_model(&msgs, Some(tmpl), "", "<|im_end|>", None, None);
    assert!(
        prompt.contains("BANANE VIOLETTE MYSTERE 42"),
        "the user's question is not in the prompt that would be sent:\n{prompt}"
    );
}

/// A prompt that ends on a turn-CLOSING marker is finished for the model, so
/// the model's own turn is opened before the request goes out.
///
/// This was detected and logged and then sent anyway, which guaranteed a
/// one-token reply. Reported against a Qwen3-8B whose template this renderer
/// declines, leaving every prompt ending on `<|im_end|>`.
#[test]
fn a_prompt_left_on_a_closed_turn_gets_the_models_turn_opened() {
    // A template that renders the conversation and stops — no generation
    // prompt, exactly the shape the report showed.
    let no_gen_prompt =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}";
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "Say OK".into(),
        images: vec![],
    }];

    let prompt = build_prompt_with_model(
        &msgs,
        Some(no_gen_prompt),
        "",
        "<|im_end|>",
        Some("qwen3-8b"),
        None,
    );
    assert!(
        prompt.ends_with("<|im_start|>assistant\n"),
        "the model's turn must be opened; got tail {:?}",
        &prompt[prompt.len().saturating_sub(60)..]
    );
    assert!(
        crate::inference::chat_template::prompt_hands_over_to_the_model(&prompt),
        "and the repaired prompt must satisfy the check that found the problem"
    );
}

/// Each family gets ITS opener, not ChatML's. Reaching ChatML for a model that
/// is not a ChatML model is the failure that produced stray `<|im_end|>` in
/// Llama-3 replies for several releases.
#[test]
fn the_opener_matches_the_family_that_closed_the_turn() {
    assert_eq!(
        super::generation_prompt_after("<|eot_id|>"),
        Some("<|start_header_id|>assistant<|end_header_id|>\n\n")
    );
    assert_eq!(
        super::generation_prompt_after("<end_of_turn>"),
        Some("<start_of_turn>model\n")
    );
    assert_eq!(
        super::generation_prompt_after("<|end|>"),
        Some("<|assistant|>\n")
    );
    assert_eq!(
        super::generation_prompt_after("<|im_end|>"),
        Some("<|im_start|>assistant\n")
    );
}

/// An ambiguous closer names no single family, so nothing is appended — a
/// confidently wrong opener is worse than a diagnosable prompt. `</s>` is
/// Llama-2, Mistral AND vicuna.
#[test]
fn an_ambiguous_closer_is_left_alone() {
    assert_eq!(super::generation_prompt_after("</s>"), None);
    assert_eq!(super::generation_prompt_after("<|endoftext|>"), None);
}

/// The control: a prompt that already opens the model's turn is untouched.
#[test]
fn a_correct_prompt_is_not_rewritten() {
    let good = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "Say OK".into(),
        images: vec![],
    }];
    let prompt =
        build_prompt_with_model(&msgs, Some(good), "", "<|im_end|>", Some("qwen2.5"), None);
    assert_eq!(
        prompt.matches("<|im_start|>assistant").count(),
        1,
        "an already-correct prompt must not gain a second opener"
    );
}

#[test]
fn temp_probe3() {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "SYS".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "MARKER42".into(),
            images: vec![],
        },
    ];
    let t = |src: &str| apply_chat_template(src, &msgs, "", "<|im_end|>", true, None);
    println!("--- ns set only:      {:?}", t("A{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}B"));
    println!(
        "--- reversed loop:    {:?}",
        t("A{%- for message in messages[::-1] %}x{%- endfor %}B")
    );
    println!("--- reversed + inner set: {:?}", t("A{%- for message in messages[::-1] %}{%- set index = (messages|length - 1) - loop.index0 %}x{%- endfor %}B"));
    println!("--- inner if (not-paren): {:?}", t("A{%- for message in messages %}{%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<t>') and message.content.endswith('</t>')) %}y{%- endif %}x{%- endfor %}B"));
}

/// The official Qwen3 template renders byte-for-byte as Jinja2 does, for the
/// ordinary shapes.
///
/// Expected strings captured from `jinja2` 3.1.2 on the same fixture and the
/// same messages, the way the Llama-3 sibling test above is pinned. Verifying
/// against our own past output would have been satisfied by the broken
/// behaviour this replaces: the renderer used to abandon the template at
/// `{% for message in messages[::-1] %}` and emit only the system block.
#[test]
fn the_official_qwen3_template_renders_exactly_as_jinja2_does() {
    let tmpl = include_str!("fixtures/qwen3_official.jinja");

    let single = vec![
        ChatMessage {
            role: Role::System,
            content: "You are a helpful assistant.".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "MARKER42".into(),
            images: vec![],
        },
    ];
    assert_eq!(
        apply_chat_template(tmpl, &single, "", "<|im_end|>", true, None).as_deref(),
        Some(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nMARKER42<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
        "single-turn render diverged from jinja2"
    );

    // Multi-turn, including an assistant turn carrying a <think> block — the
    // template strips the reasoning from history, and that is the branch doing
    // real work rather than concatenating.
    let multi = vec![
        ChatMessage {
            role: Role::System,
            content: "SYS".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "first".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::Assistant,
            content: "<think>pondering</think>answer one".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "second".into(),
            images: vec![],
        },
    ];
    // Including the reasoning block being stripped out of HISTORY, which the
    // hand-rolled engine could not do: jinja2 uses
    // `content.split('</think>')[-1]`, and Python string methods now work.
    assert_eq!(
        apply_chat_template(tmpl, &multi, "", "<|im_end|>", true, None).as_deref(),
        Some(
            "<|im_start|>system\nSYS<|im_end|>\n\
             <|im_start|>user\nfirst<|im_end|>\n\
             <|im_start|>assistant\nanswer one<|im_end|>\n\
             <|im_start|>user\nsecond<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
        "multi-turn render diverged from jinja2"
    );

    // And the same conversation without a reasoning block matches jinja2 exactly.
    let plain = vec![
        ChatMessage {
            role: Role::System,
            content: "SYS".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "first".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::Assistant,
            content: "answer one".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "second".into(),
            images: vec![],
        },
    ];
    assert_eq!(
        apply_chat_template(tmpl, &plain, "", "<|im_end|>", true, None).as_deref(),
        Some(
            "<|im_start|>system\nSYS<|im_end|>\n\
             <|im_start|>user\nfirst<|im_end|>\n\
             <|im_start|>assistant\nanswer one<|im_end|>\n\
             <|im_start|>user\nsecond<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
        "multi-turn render diverged from jinja2"
    );
}

/// A `for` over a slice this renderer does not implement must still walk the
/// list, because an unrecognised `for` leaves a stray `{% endfor %}` and a
/// stray `endfor` ends the enclosing block — silently discarding the rest of
/// the template. `[N:]` keeps its meaning; the others are identity.
#[test]
fn an_unimplemented_slice_does_not_swallow_the_rest_of_the_template() {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: "A".into(),
            images: vec![],
        },
        ChatMessage {
            role: Role::User,
            content: "B".into(),
            images: vec![],
        },
    ];
    let render = |src: &str| apply_chat_template(src, &msgs, "", "<|im_end|>", true, None);

    // The Qwen3 shape. Everything after the loop must survive.
    assert_eq!(
        render("[{%- for message in messages[::-1] %}x{%- endfor %}]TAIL").as_deref(),
        Some("[xx]TAIL")
    );
    // `[N:]` still drops the messages it names.
    assert_eq!(
        render("[{%- for message in messages[1:] %}x{%- endfor %}]TAIL").as_deref(),
        Some("[x]TAIL")
    );
    // A bare index is NOT a list and must not be treated as one.
    assert_eq!(
        render("{{ messages[0]['content'] }}TAIL").as_deref(),
        Some("ATAIL")
    );
}

/// `x is string` must be true for message content, because Qwen3's template
/// BLANKS the message when it is false rather than degrading gracefully.
#[test]
fn message_content_is_recognised_as_a_string() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "KEEP".into(),
        images: vec![],
    }];
    assert_eq!(
        apply_chat_template(
            "{%- for message in messages %}{% if message.content is string %}{{ message.content }}{% else %}BLANKED{% endif %}{%- endfor %}",
            &msgs,
            "",
            "<|im_end|>",
            true,
            None,
        )
        .as_deref(),
        Some("KEEP")
    );
}

/// The template an actual Qwen3 GGUF ships is NOT the one in
/// `qwen3_official.jinja`, and the difference matters.
///
/// Captured from `Qwen/Qwen3-1.7B-GGUF`'s `gguf_header.bin` (4100 bytes; the
/// other fixture is 4169). It is an older revision of the same template and it
/// walks history with `{% for index in range(ns.last_query_index, -1, -1) %}`
/// plus `{% set message = messages[index] %}`, where the newer one uses
/// `messages[::-1]`. `range()`, namespace attributes and indexing the message
/// list by a variable are all unimplemented here, so this one still does not
/// render natively.
///
/// What this test pins is that it FAILS SAFELY: the half-render is caught and
/// the fallback carries the question. That is the property that matters — a
/// prompt without the question in it is answered fluently and wrongly, which
/// is how this was reported in the first place.
#[test]
fn the_template_a_real_qwen3_gguf_ships_still_reaches_the_model_with_the_question() {
    let shipped = include_str!("fixtures/qwen3_gguf_shipped.jinja");
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "BANANE VIOLETTE MYSTERE 42".into(),
        images: vec![],
    }];

    // Whatever the renderer manages, the prompt that would be SENT must carry
    // the question and open the model's turn.
    let prompt = build_prompt_with_model(&msgs, Some(shipped), "", "<|im_end|>", None, None);
    assert!(
        prompt.contains("BANANE VIOLETTE MYSTERE 42"),
        "the shipped Qwen3 template produced a prompt with no question in it:\n{prompt}"
    );
    assert!(
        prompt.trim_end().ends_with("<|im_start|>assistant"),
        "prompt does not open the model's turn: {:?}",
        &prompt[prompt.len().saturating_sub(60)..]
    );
}

/// A chat template is untrusted input — it arrives inside a downloaded GGUF —
/// and it is a program. R101 found a recursion cap does not stop
/// `{% set x = x + x %}`, and capped rendered output. That guard lived in the
/// evaluator replaced by minijinja, so this pins that it still holds.
///
/// Verified by PLANTING each attack rather than by trusting the constants.
#[test]
fn a_hostile_chat_template_cannot_amplify_without_bound() {
    let msgs = test_messages();

    // 1. A runaway loop is stopped by the fuel budget.
    let spin = "{% for a in range(100000) %}{% for b in range(100000) %}x{% endfor %}{% endfor %}";
    assert!(
        apply_chat_template(spin, &msgs, "", "", true, None).is_none(),
        "an unbounded loop must not be allowed to run to completion"
    );

    // 2. A template that EMITS more than the output cap is discarded.
    //    2000 * 4096 characters is over 8 MiB, past the 4 MiB ceiling.
    let flood = "{% for a in range(2000) %}{{ 'x' * 4096 }}{% endfor %}";
    assert!(
        apply_chat_template(flood, &msgs, "", "", true, None).is_none(),
        "output past the ceiling must be discarded, not returned"
    );

    // 3. An implausibly large template source is declined before rendering,
    //    which is what bounds straight-line value doubling.
    let huge = format!("{}{{{{ 'ok' }}}}", "{# pad #}".repeat(40_000));
    assert!(
        huge.len() > 200 * 1024,
        "test needs a template past the cap"
    );
    assert!(
        apply_chat_template(&huge, &msgs, "", "", true, None).is_none(),
        "a template past the source ceiling must be declined"
    );

    // CONTROLS, so `is_none()` above is attributable to the caps rather than to
    // any old render failure: the same shapes just UNDER each ceiling render.
    let under_source_cap = format!("{}{{{{ 'ok' }}}}", "{# pad #}".repeat(1_000));
    assert!(under_source_cap.len() < 200 * 1024);
    assert_eq!(
        apply_chat_template(&under_source_cap, &msgs, "", "", true, None).as_deref(),
        Some("ok"),
        "a template under the source ceiling must still render"
    );
    let under_output_cap = "{% for a in range(100) %}{{ 'x' * 1024 }}{% endfor %}";
    assert_eq!(
        apply_chat_template(under_output_cap, &msgs, "", "", true, None).map(|s| s.len()),
        Some(100 * 1024),
        "output under the ceiling must be returned intact"
    );

    // And an ordinary template is untouched by any of the three.
    assert_eq!(
        apply_chat_template(
            "{% for m in messages %}{{ m.content }};{% endfor %}",
            &msgs,
            "",
            "",
            true,
            None,
        )
        .as_deref(),
        Some("You are helpful.;Hello;")
    );
}

// ---------------------------------------------------------------------------
// Tool framing: the model's own template, or prose for models that have none
// ---------------------------------------------------------------------------

const QWEN3_OFFICIAL: &str = include_str!("fixtures/qwen3_official.jinja");
const LLAMA3_OFFICIAL: &str = include_str!("fixtures/llama3_official.jinja");

fn weather_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    })
}

/// The report this came from: Qwen3's template opens with `{%- if tools %}` and
/// carries its own `# Tools` section, and we never passed `tools`, so that
/// branch was unreachable on every request. The model was told about its tools
/// in a JSON format it had never been trained on while its own framing sat
/// unused.
#[test]
fn qwen3_renders_its_own_tool_framing() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "what time is it?".into(),
        images: vec![],
    }];
    let tools = vec![weather_tool()];
    let prompt = build_prompt(
        &msgs,
        Some(QWEN3_OFFICIAL),
        "",
        "<|im_end|>",
        Some("qwen3-8b"),
        Some(&tools),
    );

    // Qwen's own framing, straight out of its template.
    assert!(prompt.contains("# Tools"), "no Tools section: {prompt}");
    assert!(prompt.contains("<tools>"), "no <tools> block: {prompt}");
    assert!(
        prompt.contains("<tool_call>"),
        "the template's own call format is missing: {prompt}"
    );
    assert!(
        prompt.contains("\"name\": \"get_weather\"") || prompt.contains("get_weather"),
        "the tool itself was not rendered: {prompt}"
    );

    // And NOT the prose fallback, which is what it used to get.
    assert!(
        !prompt.contains("You have access to the following tools"),
        "a template that renders tools must not also be handed the prose form: {prompt}"
    );
    assert!(
        prompt.contains("what time is it?"),
        "question lost: {prompt}"
    );
}

/// A model whose template says nothing about tools cannot render them, so it
/// still gets the prose description — that path is the fallback, not the
/// default.
#[test]
fn a_template_without_tool_support_still_gets_the_prose_description() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "what time is it?".into(),
        images: vec![],
    }];
    let tools = vec![weather_tool()];
    // A minimal template with no mention of tools at all.
    let tmpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
    let prompt = build_prompt(&msgs, Some(tmpl), "", "", Some("some-model"), Some(&tools));

    assert!(
        prompt.contains("You have access to the following tools"),
        "a model whose template cannot render tools must be told in prose: {prompt}"
    );
    assert!(
        prompt.contains("get_weather"),
        "tool name missing: {prompt}"
    );
    assert!(
        prompt.contains("what time is it?"),
        "question lost: {prompt}"
    );
}

/// No tools means neither form appears — a request that sends none must render
/// exactly as it did before any of this.
#[test]
fn no_tools_means_no_tool_framing_of_either_kind() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "what time is it?".into(),
        images: vec![],
    }];
    let with_none = build_prompt(
        &msgs,
        Some(QWEN3_OFFICIAL),
        "",
        "<|im_end|>",
        Some("qwen3-8b"),
        None,
    );
    let with_empty = build_prompt(
        &msgs,
        Some(QWEN3_OFFICIAL),
        "",
        "<|im_end|>",
        Some("qwen3-8b"),
        Some(&[]),
    );
    for prompt in [&with_none, &with_empty] {
        assert!(!prompt.contains("# Tools"), "unasked-for tools: {prompt}");
        assert!(
            !prompt.contains("You have access to the following tools"),
            "unasked-for prose: {prompt}"
        );
    }
    assert_eq!(
        with_none, with_empty,
        "an empty tool list must render identically to no tool list"
    );
}

/// `template_renders_tools` is the predicate that chooses between the two, so
/// pin what it answers rather than only its effect.
#[test]
fn template_renders_tools_reads_the_template_not_the_request() {
    assert!(template_renders_tools(QWEN3_OFFICIAL));
    assert!(template_renders_tools(LLAMA3_OFFICIAL));
    assert!(!template_renders_tools(
        "{% for m in messages %}{{ m.content }}{% endfor %}"
    ));
}

/// `tojson` comes from minijinja's `json` feature, and this crate builds
/// minijinja with `default-features = false`. Without it the filter is unknown,
/// which fails the WHOLE render rather than that one expression — so a model
/// whose template calls it silently gets a fallback template instead of its
/// own. Every real template that renders tools calls it.
///
/// Pinned as behaviour rather than as a line in `Cargo.toml`, because what
/// matters is that the filter resolves, not how it came to.
#[test]
fn the_tojson_filter_is_available_to_templates() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "hi".into(),
        images: vec![],
    }];
    let rendered = apply_chat_template(
        "{% for m in messages %}{{ m | tojson }}{% endfor %}",
        &msgs,
        "",
        "",
        false,
        None,
    );
    let rendered = rendered.expect("a template calling tojson must render");
    assert!(
        rendered.contains("\"content\":\"hi\"") || rendered.contains("\"content\": \"hi\""),
        "tojson did not serialise the message: {rendered}"
    );
}

// ---------------------------------------------------------------------------
// `tojson` — the filter every tool-rendering template calls
// ---------------------------------------------------------------------------

const GLM4_GGUF_SHIPPED: &str = include_str!("fixtures/glm4_gguf_shipped.jinja");

/// GLM-4 asks for `tojson(indent=4, ensure_ascii=False)`. minijinja's builtin
/// accepts `indent` and nothing else, and an error inside a filter fails the
/// WHOLE render — so the model's own `# 可用工具` framing was unreachable and
/// every GLM-4 request carrying tools was answered by a fallback prompt.
#[test]
fn the_glm4_template_renders_its_own_tool_section() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "what is the weather in Paris?".into(),
        images: vec![],
    }];
    let tools = vec![weather_tool()];
    let prompt = build_prompt(
        &msgs,
        Some(GLM4_GGUF_SHIPPED),
        "",
        "<|user|>",
        Some("glm-4-9b-0414"),
        Some(&tools),
    );

    assert!(
        prompt.contains("可用工具"),
        "GLM-4's own tool heading is missing, so this rendered through a fallback: {prompt}"
    );
    assert!(
        prompt.contains("get_weather"),
        "the tool itself was not rendered: {prompt}"
    );
    assert!(
        prompt.contains("<|user|>") && prompt.contains("<|assistant|>"),
        "GLM-4's turn markers are missing: {prompt}"
    );
    assert!(
        !prompt.contains("You have access to the following tools"),
        "a template that renders tools must not also be handed the prose form: {prompt}"
    );
    assert!(
        prompt.contains("what is the weather in Paris?"),
        "question lost: {prompt}"
    );
}

/// minijinja's `tojson` rewrites `<`, `>`, `&` and `'` as `<` and friends,
/// because it is written for embedding JSON in a web page. `transformers`
/// overrides the same filter for exactly that reason. An apostrophe in a tool
/// description is ordinary English, and it was reaching the model escaped.
#[test]
fn a_tool_schema_reaches_the_model_unescaped() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "where am I?".into(),
        images: vec![],
    }];
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "locate",
            "description": "Find the user's location & report it if accuracy < 50m",
            "parameters": {"type": "object", "properties": {}}
        }
    })];
    let prompt = build_prompt(
        &msgs,
        Some(QWEN3_OFFICIAL),
        "",
        "<|im_end|>",
        Some("qwen3-8b"),
        Some(&tools),
    );

    assert!(
        prompt.contains("the user's location & report it if accuracy < 50m"),
        "the description did not survive rendering: {prompt}"
    );
    for escape in ["\\u0027", "\\u0026", "\\u003c", "\\u003e"] {
        assert!(
            !prompt.contains(escape),
            "{escape} reached the model — the HTML-escaping tojson is back: {prompt}"
        );
    }
}

/// Every keyword `transformers` accepts, since that is the signature templates
/// are written against: `tojson(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`.
#[test]
fn tojson_takes_the_arguments_transformers_defines() {
    let msgs = user_only_messages();
    let render = |expr: &str| {
        apply_chat_template(
            &format!("{{% set x = {{\"b\": 1, \"a\": \"café\"}} %}}{expr}"),
            &msgs,
            "",
            "",
            false,
            None,
        )
    };

    // Python's default separators are `", "` and `": "`, NOT serde_json's
    // `","` and `":"` — and a bare `| tojson` is how Qwen and Llama-3.1 render
    // every tool schema, so this is the common case, not a corner.
    //
    // Keys come out sorted whatever is asked for: a map reaching this filter is
    // either a minijinja literal or a `serde_json::Value`, and `serde_json` is
    // built here without `preserve_order`, so its map is a `BTreeMap`.
    // `sort_keys` therefore changes nothing today — it is honoured rather than
    // rejected, which is the whole point.
    assert_eq!(
        render("{{ x | tojson }}").as_deref(),
        Some(r#"{"a": "café", "b": 1}"#),
        "the default must match python's separators"
    );
    assert_eq!(
        render("{{ x | tojson(separators=(',', ':')) }}").as_deref(),
        Some(r#"{"a":"café","b":1}"#)
    );
    assert_eq!(
        render("{{ x | tojson(sort_keys=True) }}").as_deref(),
        Some(r#"{"a": "café", "b": 1}"#)
    );
    assert_eq!(
        render("{{ x | tojson(ensure_ascii=True) }}").as_deref(),
        Some(r#"{"a": "caf\u00e9", "b": 1}"#)
    );
    assert_eq!(
        render("{{ x | tojson(indent=4, ensure_ascii=False) }}").as_deref(),
        Some("{\n    \"a\": \"café\",\n    \"b\": 1\n}"),
        "GLM-4's exact call"
    );
    // The positional slot stays `indent`, as in Jinja2's builtin and minja.
    assert_eq!(
        render("{{ x | tojson(2) }}").as_deref(),
        Some("{\n  \"a\": \"café\",\n  \"b\": 1\n}")
    );
}

/// Byte-for-byte against `jinja2` driven exactly as `transformers` drives it —
/// `ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)`, its own
/// `tojson`, its own `raise_exception`.
///
/// One deliberate difference, and it is the only one: keys arrive
/// ALPHABETICALLY rather than in the order the caller wrote them, because a
/// tool definition is a `serde_json::Value` by the time it reaches here and
/// `serde_json` is built without `preserve_order`, so its map is a `BTreeMap`.
/// The reference below was generated with `sort_keys=True` for that reason.
/// See `docs/FUTURE_WORK.md` — "A tool schema reaches the model with its keys
/// alphabetised".
#[test]
fn the_glm4_template_renders_exactly_as_transformers_does() {
    let msgs = vec![ChatMessage {
        role: Role::User,
        content: "what is the weather in Paris?".into(),
        images: vec![],
    }];
    let tools = vec![weather_tool()];
    let rendered =
        apply_chat_template(GLM4_GGUF_SHIPPED, &msgs, "", "<|user|>", true, Some(&tools))
            .expect("GLM-4's template must render");

    assert_eq!(rendered, "[gMASK]<sop><|system|>\n# 可用工具\n\n## get_weather\n\n{\n    \"description\": \"Get the weather\",\n    \"name\": \"get_weather\",\n    \"parameters\": {\n        \"properties\": {\n            \"city\": {\n                \"type\": \"string\"\n            }\n        },\n        \"type\": \"object\"\n    }\n}\n在调用上述函数时，请使用 Json 格式表示调用的参数。<|user|>\nwhat is the weather in Paris?<|assistant|>");
}

/// An astral character escapes as a surrogate pair, which is what Python does.
#[test]
fn ensure_ascii_escapes_an_astral_character_as_a_surrogate_pair() {
    let msgs = user_only_messages();
    let rendered = apply_chat_template(
        "{% set x = [\"🙂\"] %}{{ x | tojson(ensure_ascii=True) }}",
        &msgs,
        "",
        "",
        false,
        None,
    );
    assert_eq!(rendered.as_deref(), Some(r#"["\ud83d\ude42"]"#));
}
