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
    let template =
        "{% if messages[0]['role'] == 'system' %}{{ raise_exception('no system') }}{% endif %}OK";
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
        apply_chat_template(broken_gemma, &msgs, "", "<eos>", true).is_none(),
        "fixture must actually fail to evaluate"
    );
    let result = build_prompt_with_model(&msgs, Some(broken_gemma), "", "<eos>", Some("llava-7b"));
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
            apply_chat_template(tmpl, &msgs, "", "</s>", true).is_none(),
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
            false
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
    );
    assert!(
        result.contains("<|im_start|>"),
        "expected ChatML for an unrecognised name, got: {result:?}"
    );
}
