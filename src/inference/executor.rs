use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::SwarmError;
use crate::types::SamplingParams;

/// Manages a loaded model and provides token generation.
///
/// Phase 1: Stub implementation that returns placeholder responses.
/// When llama-cpp-2 is added as a dependency, this wraps the real
/// llama.cpp context for GGUF model loading and inference.
pub struct ModelExecutor {
    model_path: Option<PathBuf>,
    loaded: bool,
    model_name: String,
}

/// Thread-safe handle to the model executor.
pub type SharedExecutor = Arc<Mutex<ModelExecutor>>;

impl Default for ModelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelExecutor {
    pub fn new() -> Self {
        Self {
            model_path: None,
            loaded: false,
            model_name: String::new(),
        }
    }

    /// Load a GGUF model from disk.
    ///
    /// In Phase 1 (stub): validates the path exists and marks as loaded.
    /// With llama-cpp-2: initializes LlamaModel + LlamaContext.
    pub fn load_model(&mut self, path: &Path, gpu_layers: u32) -> Result<(), SwarmError> {
        if !path.exists() {
            return Err(SwarmError::Inference(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        tracing::info!(
            path = %path.display(),
            gpu_layers = gpu_layers,
            "Loading model"
        );

        // TODO: Replace with real llama-cpp-2 loading:
        //   let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        //   let model = LlamaModel::load_from_file(path, model_params)?;
        //   let ctx_params = LlamaContextParams::default().with_n_ctx(4096);
        //   let ctx = LlamaContext::new_with_model(&model, ctx_params)?;

        self.model_path = Some(path.to_path_buf());
        self.model_name = name;
        self.loaded = true;

        tracing::info!(model = %self.model_name, "Model loaded (stub executor)");
        Ok(())
    }

    /// Check if a model is loaded and ready for inference.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get the name of the currently loaded model.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Generate a streaming response for the given prompt.
    ///
    /// Calls the callback with each generated token string. The callback
    /// returns `false` to stop generation early.
    ///
    /// Phase 1 (stub): Generates a canned response token by token.
    /// With llama-cpp-2: Tokenizes prompt, runs forward pass, samples, detokenizes.
    pub async fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut callback: F,
    ) -> Result<GenerationResult, SwarmError>
    where
        F: FnMut(&str) -> bool,
    {
        if !self.loaded {
            return Err(SwarmError::NoModelLoaded);
        }

        tracing::debug!(
            prompt_len = prompt.len(),
            temperature = params.temperature,
            max_tokens = params.max_tokens,
            "Starting generation"
        );

        // TODO: Replace stub with real inference:
        //   1. Tokenize prompt with self.model.tokenize(prompt)
        //   2. Evaluate prompt tokens in batches
        //   3. Sample next token using sampling.rs
        //   4. Detokenize and call callback
        //   5. Repeat until stop condition or max_tokens

        // Stub: emit a placeholder response word by word
        let response = "Hello! I'm SwarmLLM, a decentralized inference node. \
                        The model executor is running in stub mode. \
                        Once a GGUF model is loaded with llama-cpp-2 bindings, \
                        this will produce real model outputs.";

        let words: Vec<&str> = response.split_inclusive(' ').collect();
        let mut completion_tokens = 0u32;
        let max = params.max_tokens.min(words.len() as u32);

        for word in words.iter().take(max as usize) {
            completion_tokens += 1;
            if !callback(word) {
                break;
            }
            // Simulate token generation latency
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Rough prompt token estimate (4 chars per token)
        let prompt_tokens = (prompt.len() / 4).max(1) as u32;

        Ok(GenerationResult {
            prompt_tokens,
            completion_tokens,
            finish_reason: if completion_tokens >= max {
                FinishReason::MaxTokens
            } else {
                FinishReason::Stop
            },
        })
    }

    /// Generate a complete (non-streaming) response.
    pub async fn generate(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<(String, GenerationResult), SwarmError> {
        let mut output = String::new();
        let result = self
            .generate_stream(prompt, params, |token| {
                output.push_str(token);
                true
            })
            .await?;
        Ok((output, result))
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    MaxTokens,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::MaxTokens => "length",
        }
    }
}

/// Build the chat prompt from messages in a simple format.
/// When a real tokenizer is available, this should use the model's chat template.
pub fn build_chat_prompt(messages: &[crate::types::ChatMessage]) -> String {
    use crate::types::Role;
    let mut prompt = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                prompt.push_str(&format!("System: {}\n", msg.content));
            }
            Role::User => {
                prompt.push_str(&format!("User: {}\n", msg.content));
            }
            Role::Assistant => {
                prompt.push_str(&format!("Assistant: {}\n", msg.content));
            }
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_starts_unloaded() {
        let exec = ModelExecutor::new();
        assert!(!exec.is_loaded());
    }

    #[test]
    fn load_nonexistent_model_fails() {
        let mut exec = ModelExecutor::new();
        let result = exec.load_model(&PathBuf::from("/nonexistent/model.gguf"), 0);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn generate_without_model_fails() {
        let exec = ModelExecutor::new();
        let result = exec.generate("test", &SamplingParams::default()).await;
        assert!(result.is_err());
    }

    #[test]
    fn build_chat_prompt_formats_correctly() {
        use crate::types::{ChatMessage, Role};
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "Hi".into(),
            },
        ];
        let prompt = build_chat_prompt(&messages);
        assert!(prompt.contains("System: You are helpful."));
        assert!(prompt.contains("User: Hi"));
        assert!(prompt.ends_with("Assistant: "));
    }
}
