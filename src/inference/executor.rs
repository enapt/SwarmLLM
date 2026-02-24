use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::SwarmError;
use crate::types::SamplingParams;

/// Thread-safe handle to the model executor.
pub type SharedExecutor = Arc<Mutex<ModelExecutor>>;

/// Manages a loaded model and provides token generation.
///
/// With the `llama` feature: wraps llama-cpp-2 for real GGUF model inference with GPU support.
/// Without the feature: stub implementation that returns placeholder responses.
pub struct ModelExecutor {
    model_path: Option<PathBuf>,
    loaded: bool,
    model_name: String,
    #[cfg(feature = "llama")]
    backend: Option<llama_cpp_2::llama_backend::LlamaBackend>,
    #[cfg(feature = "llama")]
    model: Option<llama_cpp_2::model::LlamaModel>,
}

// LlamaModel is Send but not Sync by default — we protect it with Mutex so this is safe.
#[cfg(feature = "llama")]
unsafe impl Send for ModelExecutor {}

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
            #[cfg(feature = "llama")]
            backend: None,
            #[cfg(feature = "llama")]
            model: None,
        }
    }

    /// Load a GGUF model from disk.
    ///
    /// With llama-cpp-2: initializes the backend, loads the model with GPU offloading.
    /// Without: validates the path exists and marks as loaded (stub).
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

        #[cfg(feature = "llama")]
        {
            use llama_cpp_2::llama_backend::LlamaBackend;
            use llama_cpp_2::model::params::LlamaModelParams;
            use llama_cpp_2::model::LlamaModel;

            // Initialize backend (redirect logs to tracing)
            let backend = LlamaBackend::init().map_err(|e| {
                SwarmError::Inference(format!("Failed to init llama backend: {e}"))
            })?;

            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());

            // Log available devices
            let devices = llama_cpp_2::list_llama_ggml_backend_devices();
            for dev in &devices {
                tracing::info!(
                    name = %dev.name,
                    desc = %dev.description,
                    backend = %dev.backend,
                    mem_total_mb = dev.memory_total / (1024 * 1024),
                    mem_free_mb = dev.memory_free / (1024 * 1024),
                    device_type = ?dev.device_type,
                    "Detected compute device"
                );
            }

            tracing::info!(
                gpu_offload = backend.supports_gpu_offload(),
                mmap = backend.supports_mmap(),
                "Backend capabilities"
            );

            // Load model with GPU layer offloading
            let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
            let model = LlamaModel::load_from_file(&backend, path, &model_params).map_err(
                |e| SwarmError::Inference(format!("Failed to load model: {e}")),
            )?;

            tracing::info!(
                n_vocab = model.n_vocab(),
                "Model loaded into llama.cpp"
            );

            self.backend = Some(backend);
            self.model = Some(model);
        }

        self.model_path = Some(path.to_path_buf());
        self.model_name = name;
        self.loaded = true;

        #[cfg(feature = "llama")]
        tracing::info!(model = %self.model_name, "Model loaded (llama-cpp-2 with GPU)");
        #[cfg(not(feature = "llama"))]
        tracing::info!(model = %self.model_name, "Model loaded (stub — enable 'llama' feature for real inference)");

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
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        #[allow(unused_mut)] mut callback: F,
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

        #[cfg(feature = "llama")]
        {
            return self.generate_stream_llama(prompt, params, callback);
        }

        #[cfg(not(feature = "llama"))]
        {
            self.generate_stream_stub(prompt, params, &mut callback)
        }
    }

    /// Real llama.cpp inference path.
    #[cfg(feature = "llama")]
    fn generate_stream_llama<F>(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        mut callback: F,
    ) -> Result<GenerationResult, SwarmError>
    where
        F: FnMut(&str) -> bool,
    {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;
        use std::num::NonZeroU32;

        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Backend not initialized".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Model not loaded".into()))?;

        // Create a fresh context for this generation
        let ctx_size = NonZeroU32::new(4096);
        let ctx_params = LlamaContextParams::default().with_n_ctx(ctx_size);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| SwarmError::Inference(format!("Failed to create context: {e}")))?;

        // Tokenize the prompt
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| SwarmError::Inference(format!("Tokenization failed: {e}")))?;

        let prompt_tokens = tokens.len() as u32;
        let n_ctx = 4096u32;
        let max_gen = params.max_tokens.min(n_ctx.saturating_sub(prompt_tokens));

        if prompt_tokens >= n_ctx {
            return Err(SwarmError::Inference(format!(
                "Prompt too long: {prompt_tokens} tokens exceeds context size {n_ctx}"
            )));
        }

        // Create batch and add prompt tokens
        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|e| SwarmError::Inference(format!("Failed to add tokens to batch: {e}")))?;

        // Set logits for last token
        // The batch's add_sequence with logits_all=false doesn't set logits on any token,
        // so we need to manually set the last one. We'll re-add with logits on last.
        batch.clear();
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| SwarmError::Inference(format!("Batch add failed: {e}")))?;
        }

        // Process the prompt (prefill)
        ctx.decode(&mut batch)
            .map_err(|e| SwarmError::Inference(format!("Decode failed during prefill: {e}")))?;

        // Build sampler chain: top-k → top-p → temperature → dist
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k as i32),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(42),
        ]);
        let mut sampler = sampler;

        // Create a decoder for token-to-string conversion
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Auto-regressive generation loop
        let mut completion_tokens = 0u32;
        let mut cur_pos = tokens.len();
        let eos = model.token_eos();

        for _ in 0..max_gen {
            // Sample next token
            let new_token = sampler.sample(&ctx, -1);
            sampler.accept(new_token);

            // Check for end of sequence
            if new_token == eos {
                break;
            }

            // Convert token to string
            let piece = model
                .token_to_piece(new_token, &mut decoder, true, None)
                .map_err(|e| {
                    SwarmError::Inference(format!("Token to piece failed: {e}"))
                })?;

            completion_tokens += 1;

            // Send to callback
            if !callback(&piece) {
                break;
            }

            // Prepare next batch
            batch.clear();
            batch
                .add(new_token, cur_pos as i32, &[0], true)
                .map_err(|e| SwarmError::Inference(format!("Batch add failed: {e}")))?;

            ctx.decode(&mut batch)
                .map_err(|e| SwarmError::Inference(format!("Decode failed: {e}")))?;

            cur_pos += 1;
        }

        Ok(GenerationResult {
            prompt_tokens,
            completion_tokens,
            finish_reason: if completion_tokens >= max_gen {
                FinishReason::MaxTokens
            } else {
                FinishReason::Stop
            },
        })
    }

    /// Stub path (no llama-cpp-2 feature): emit a placeholder response word by word.
    #[cfg(not(feature = "llama"))]
    fn generate_stream_stub<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        callback: &mut F,
    ) -> Result<GenerationResult, SwarmError>
    where
        F: FnMut(&str) -> bool,
    {
        let response = "Hello! I'm SwarmLLM, a decentralized inference node. \
                        The model executor is running in stub mode. \
                        Once a GGUF model is loaded with the 'llama' feature enabled, \
                        this will produce real model outputs.";

        let words: Vec<&str> = response.split_inclusive(' ').collect();
        let mut completion_tokens = 0u32;
        let max = params.max_tokens.min(words.len() as u32);

        for word in words.iter().take(max as usize) {
            completion_tokens += 1;
            if !callback(word) {
                break;
            }
        }

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
    pub fn generate(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<(String, GenerationResult), SwarmError> {
        let mut output = String::new();
        let result = self.generate_stream(prompt, params, |token| {
            output.push_str(token);
            true
        })?;
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

/// Build the chat prompt from messages using the model's chat template if available,
/// or a simple fallback format.
pub fn build_chat_prompt(messages: &[crate::types::ChatMessage]) -> String {
    use crate::types::Role;
    let mut prompt = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
            }
            Role::User => {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
            }
            Role::Assistant => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n{}<|im_end|>\n",
                    msg.content
                ));
            }
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
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

    #[test]
    fn generate_without_model_fails() {
        let mut exec = ModelExecutor::new();
        let result = exec.generate("test", &SamplingParams::default());
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
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
