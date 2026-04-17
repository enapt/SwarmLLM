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
