use crate::types::{ChatMessage, Role};

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

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

/// Build a Llama-3 formatted prompt.
///
/// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`, with a
/// leading `<|begin_of_text|>`. System messages are their own turn, unlike
/// Mistral.
pub(super) fn llama3_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            role_str(&msg.role),
            msg.content.trim()
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Build a Mistral-Instruct formatted prompt.
///
/// Mistral has no system turn: the system message is folded into the LAST user
/// message, separated by a blank line, inside that message's `[INST]` block —
/// which is what the official template does (`loop.last and system_message is
/// defined`). Emitting it as its own `[INST]` block instead would break the
/// strict user/assistant alternation the model was trained on.
pub(super) fn mistral_fallback(messages: &[ChatMessage]) -> String {
    let system = messages
        .iter()
        .find(|m| matches!(m.role, Role::System))
        .map(|m| m.content.trim());
    let last_user = messages
        .iter()
        .rposition(|m| matches!(m.role, Role::User))
        .unwrap_or(usize::MAX);

    let mut prompt = String::from("<s>");
    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            Role::System => {}
            Role::User | Role::Tool => {
                prompt.push_str("[INST] ");
                if let (Some(sys), true) = (system, i == last_user) {
                    prompt.push_str(sys);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(msg.content.trim());
                prompt.push_str("[/INST]");
            }
            Role::Assistant => {
                prompt.push(' ');
                prompt.push_str(msg.content.trim());
                prompt.push_str("</s>");
            }
        }
    }
    prompt
}

/// Build a Gemma-formatted prompt (for Gemma 1/2 models).
///
/// Gemma uses `<start_of_turn>role\ncontent<end_of_turn>` format.
pub(super) fn gemma_fallback(messages: &[ChatMessage]) -> String {
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
const IMAGE_PLACEHOLDER: &str = "<image>";

/// Build a Vicuna v1.1 formatted prompt (used by LLaVA and other Vicuna-based models).
///
/// When a user message contains images, `<image>\n` is prepended to the user content
/// so that the vision encoder embeddings can replace the `<image>` token embedding
/// at the correct position in the sequence.
pub(super) fn vicuna_fallback(messages: &[ChatMessage]) -> String {
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
