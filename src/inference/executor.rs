use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::SwarmError;
use crate::types::SamplingParams;

/// Thread-safe handle to the model executor.
pub type SharedExecutor = Arc<Mutex<ModelExecutor>>;

/// Detected GPU information.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub backend: String,
}

/// Detect GPU devices via llama.cpp backend.
/// Returns None if no GPU found or llama feature not enabled.
pub fn detect_gpu() -> Option<GpuInfo> {
    #[cfg(feature = "llama")]
    {
        let devices = llama_cpp_2::list_llama_ggml_backend_devices();
        for dev in &devices {
            let is_gpu = format!("{:?}", dev.device_type).contains("Gpu");
            if is_gpu {
                return Some(GpuInfo {
                    name: dev.description.clone(),
                    vram_total_mb: (dev.memory_total / (1024 * 1024)) as u64,
                    vram_free_mb: (dev.memory_free / (1024 * 1024)) as u64,
                    backend: dev.backend.clone(),
                });
            }
        }
    }
    None
}

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

        // Prefer friendly name from GGUF metadata, fall back to filename stem
        let name = extract_gguf_name(path).unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        let load_start = std::time::Instant::now();
        tracing::info!(
            path = %path.display(),
            gpu_layers = gpu_layers,
            "DIAG: load_model starting"
        );

        #[cfg(feature = "llama")]
        {
            use llama_cpp_2::llama_backend::LlamaBackend;
            use llama_cpp_2::model::params::LlamaModelParams;
            use llama_cpp_2::model::LlamaModel;

            // Initialize backend (redirect logs to tracing)
            let backend = LlamaBackend::init()
                .map_err(|e| SwarmError::Inference(format!("Failed to init llama backend: {e}")))?;

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
            let model = LlamaModel::load_from_file(&backend, path, &model_params)
                .map_err(|e| SwarmError::Inference(format!("Failed to load model: {e}")))?;

            tracing::info!(n_vocab = model.n_vocab(), "Model loaded into llama.cpp");

            self.backend = Some(backend);
            self.model = Some(model);
        }

        self.model_path = Some(path.to_path_buf());
        self.model_name = name;
        self.loaded = true;

        let backend_type = if cfg!(feature = "llama") {
            "llama-cpp-2"
        } else {
            "stub"
        };
        tracing::info!(
            model = %self.model_name,
            path = %path.display(),
            gpu_layers,
            backend_type,
            elapsed_ms = load_start.elapsed().as_millis() as u64,
            "DIAG: load_model completed"
        );

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

    /// Get the file size of the loaded model in bytes, if available.
    pub fn model_size_bytes(&self) -> Option<u64> {
        self.model_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
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
            "DIAG: generate_stream starting"
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

        // Read context length from GGUF metadata if available, otherwise use default
        let n_ctx = model.n_ctx_train();
        let ctx_size = NonZeroU32::new(n_ctx);
        let ctx_params = LlamaContextParams::default().with_n_ctx(ctx_size);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| SwarmError::Inference(format!("Failed to create context: {e}")))?;

        // Tokenize the prompt
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| SwarmError::Inference(format!("Tokenization failed: {e}")))?;

        let prompt_tokens = tokens.len() as u32;
        let max_gen = params.max_tokens.min(n_ctx.saturating_sub(prompt_tokens));

        tracing::debug!(
            n_ctx,
            prompt_tokens,
            max_gen,
            temperature = params.temperature,
            "DIAG: generate_stream_llama starting decode"
        );

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
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(rand::random::<u32>()),
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
                .map_err(|e| SwarmError::Inference(format!("Token to piece failed: {e}")))?;

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

    /// Generate a response using speculative decoding with a draft model.
    ///
    /// The draft model proposes `gamma` tokens at a time, then the target model
    /// (self) verifies them in a single forward pass. Accepted tokens are emitted
    /// via the callback. The output distribution is mathematically identical to
    /// sampling from the target model alone.
    ///
    /// Falls back to standard generation if the `llama` feature is not enabled.
    pub fn generate_speculative<F>(
        &mut self,
        draft: &mut ModelExecutor,
        prompt: &str,
        params: &SamplingParams,
        gamma: u32,
        mut callback: F,
    ) -> Result<
        (
            GenerationResult,
            crate::inference::speculative::SpeculativeDraftState,
        ),
        SwarmError,
    >
    where
        F: FnMut(&str) -> bool,
    {
        if !self.loaded {
            return Err(SwarmError::NoModelLoaded);
        }
        if !draft.loaded {
            return Err(SwarmError::Inference("Draft model not loaded".to_string()));
        }

        tracing::info!(
            prompt_len = prompt.len(),
            gamma = gamma,
            draft_model = %draft.model_name,
            target_model = %self.model_name,
            "Starting speculative decoding"
        );

        #[cfg(feature = "llama")]
        {
            return self.generate_speculative_llama(draft, prompt, params, gamma, callback);
        }

        #[cfg(not(feature = "llama"))]
        {
            // Stub mode: fall back to standard generation, no speculative benefit
            let mut state = crate::inference::speculative::SpeculativeDraftState::new(
                crate::types::ModelId(draft.model_name.clone()),
                crate::types::ModelId(self.model_name.clone()),
                gamma,
            );
            let result = self.generate_stream(prompt, params, &mut callback)?;
            state.record_batch(0, 0);
            Ok((result, state))
        }
    }

    /// Speculative decoding with llama-cpp backend.
    ///
    /// Algorithm:
    /// 1. Prefill both models with the prompt, saving target's initial logits
    /// 2. Loop:
    ///    a. Draft phase: run draft model gamma times to get candidate tokens + logits
    ///    b. Verify phase: feed candidates through target one-by-one, collecting probs
    ///    c. Accept/reject using rejection sampling
    ///    d. Emit accepted tokens + bonus token
    ///    e. Resynchronize both models' KV-caches to the accepted prefix
    #[cfg(feature = "llama")]
    fn generate_speculative_llama<F>(
        &mut self,
        draft: &mut ModelExecutor,
        prompt: &str,
        params: &SamplingParams,
        gamma: u32,
        mut callback: F,
    ) -> Result<
        (
            GenerationResult,
            crate::inference::speculative::SpeculativeDraftState,
        ),
        SwarmError,
    >
    where
        F: FnMut(&str) -> bool,
    {
        use crate::inference::speculative::{self, SpeculativeDraftState};
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::token::LlamaToken;
        use std::num::NonZeroU32;

        let mut spec_state = SpeculativeDraftState::new(
            crate::types::ModelId(draft.model_name.clone()),
            crate::types::ModelId(self.model_name.clone()),
            gamma,
        );

        // --- Set up target model context ---
        let target_backend = self
            .backend
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Target backend not initialized".into()))?;
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Target model not loaded".into()))?;

        let t_n_ctx = target_model.n_ctx_train();
        let t_ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(t_n_ctx));
        let mut target_ctx = target_model
            .new_context(target_backend, t_ctx_params)
            .map_err(|e| SwarmError::Inference(format!("Target context failed: {e}")))?;

        // --- Set up draft model context ---
        let draft_backend = draft
            .backend
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Draft backend not initialized".into()))?;
        let draft_model = draft
            .model
            .as_ref()
            .ok_or_else(|| SwarmError::Inference("Draft model not loaded".into()))?;

        let d_n_ctx = draft_model.n_ctx_train();
        let d_ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(d_n_ctx));
        let mut draft_ctx = draft_model
            .new_context(draft_backend, d_ctx_params)
            .map_err(|e| SwarmError::Inference(format!("Draft context failed: {e}")))?;

        // Tokenize prompt for both models
        let target_tokens = target_model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| SwarmError::Inference(format!("Target tokenization failed: {e}")))?;
        let draft_tokens_prompt = draft_model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| SwarmError::Inference(format!("Draft tokenization failed: {e}")))?;

        let prompt_tokens = target_tokens.len() as u32;
        let max_gen = params.max_tokens.min(t_n_ctx.saturating_sub(prompt_tokens));

        if prompt_tokens >= t_n_ctx {
            return Err(SwarmError::Inference(format!(
                "Prompt too long: {prompt_tokens} tokens exceeds context size {t_n_ctx}"
            )));
        }

        // --- Prefill target model ---
        let mut target_batch = LlamaBatch::new(t_n_ctx as usize, 1);
        for (i, token) in target_tokens.iter().enumerate() {
            let is_last = i == target_tokens.len() - 1;
            target_batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| SwarmError::Inference(format!("Target batch add failed: {e}")))?;
        }
        target_ctx
            .decode(&mut target_batch)
            .map_err(|e| SwarmError::Inference(format!("Target prefill failed: {e}")))?;

        // Save initial target logits (predict first generated token)
        let target_n_vocab = target_model.n_vocab() as usize;
        let draft_n_vocab = draft_model.n_vocab() as usize;

        // get_logits() returns the logits for the last token in the batch (no index check)
        let initial_target_logits: Vec<f32> = target_ctx.get_logits()[..target_n_vocab].to_vec();
        let mut next_target_probs = speculative::softmax(&initial_target_logits);

        // --- Prefill draft model ---
        let mut draft_batch = LlamaBatch::new(d_n_ctx as usize, 1);
        for (i, token) in draft_tokens_prompt.iter().enumerate() {
            let is_last = i == draft_tokens_prompt.len() - 1;
            draft_batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| SwarmError::Inference(format!("Draft batch add failed: {e}")))?;
        }
        draft_ctx
            .decode(&mut draft_batch)
            .map_err(|e| SwarmError::Inference(format!("Draft prefill failed: {e}")))?;

        let target_eos = target_model.token_eos();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut completion_tokens = 0u32;
        let mut target_pos = target_tokens.len();
        let mut draft_pos = draft_tokens_prompt.len();
        let mut hit_eos = false;
        let mut user_stopped = false;

        while completion_tokens < max_gen && !hit_eos && !user_stopped {
            let remaining = max_gen - completion_tokens;
            let effective_gamma = gamma.min(remaining);

            // === DRAFT PHASE: generate gamma candidate tokens ===
            let mut draft_candidates: Vec<u32> = Vec::with_capacity(effective_gamma as usize);
            let mut draft_probs_list: Vec<Vec<f32>> = Vec::with_capacity(effective_gamma as usize);

            for _ in 0..effective_gamma {
                // get_logits() returns logits for the last decoded token
                let draft_logits: Vec<f32> = draft_ctx.get_logits()[..draft_n_vocab].to_vec();
                let probs = speculative::softmax(&draft_logits);

                // Greedy sample from draft (maximizes acceptance rate)
                let draft_token = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);

                draft_candidates.push(draft_token);
                draft_probs_list.push(probs);

                if LlamaToken(draft_token as i32) == target_eos {
                    break;
                }

                // Feed this token into draft context for next step
                draft_batch.clear();
                draft_batch
                    .add(LlamaToken(draft_token as i32), draft_pos as i32, &[0], true)
                    .map_err(|e| SwarmError::Inference(format!("Draft batch add failed: {e}")))?;
                draft_ctx
                    .decode(&mut draft_batch)
                    .map_err(|e| SwarmError::Inference(format!("Draft decode failed: {e}")))?;
                draft_pos += 1;
            }

            let num_drafted = draft_candidates.len();

            // === VERIFY PHASE ===
            // We already have `next_target_probs` which is the target's prediction
            // for the first candidate position (saved from previous iteration or prefill).
            // Feed each candidate through target one-by-one to get subsequent distributions.
            let mut verify_probs: Vec<Vec<f32>> = Vec::with_capacity(num_drafted + 1);
            verify_probs.push(next_target_probs.clone()); // verifies candidate[0]

            let mut verify_pos = target_pos;
            for &candidate in &draft_candidates {
                target_batch.clear();
                target_batch
                    .add(LlamaToken(candidate as i32), verify_pos as i32, &[0], true)
                    .map_err(|e| {
                        SwarmError::Inference(format!("Target verify batch failed: {e}"))
                    })?;
                target_ctx.decode(&mut target_batch).map_err(|e| {
                    SwarmError::Inference(format!("Target verify decode failed: {e}"))
                })?;

                let logits: Vec<f32> = target_ctx.get_logits()[..target_n_vocab].to_vec();
                verify_probs.push(speculative::softmax(&logits));
                verify_pos += 1;
            }

            // verify_probs[0] = target P(token | prompt + prev_accepted) — verifies candidate[0]
            // verify_probs[i] = target P(token | prompt + prev_accepted + candidates[0..i]) — verifies candidate[i]
            // verify_probs[num_drafted] = target P(token | prompt + prev_accepted + all_candidates) — for bonus

            // Pad draft probs to target vocab size (draft may have smaller vocab)
            let padded_draft_probs: Vec<Vec<f32>> = draft_probs_list
                .iter()
                .map(|dp| {
                    let mut padded = dp.clone();
                    if padded.len() < target_n_vocab {
                        padded.resize(target_n_vocab, 0.0);
                    }
                    padded
                })
                .collect();

            // === ACCEPT/REJECT PHASE ===
            let spec_result =
                speculative::accept_reject(&draft_candidates, &padded_draft_probs, &verify_probs)?;

            let num_accepted = spec_result.accepted_tokens.len();
            spec_state.record_batch(num_drafted as u32, num_accepted as u32);

            // Count how many new tokens we emit this round (accepted + bonus)
            let mut emitted_this_round = 0usize;

            // Emit accepted tokens
            for &token in &spec_result.accepted_tokens {
                if LlamaToken(token as i32) == target_eos {
                    hit_eos = true;
                    break;
                }
                let piece = target_model
                    .token_to_piece(LlamaToken(token as i32), &mut decoder, true, None)
                    .map_err(|e| SwarmError::Inference(format!("Token to piece failed: {e}")))?;
                completion_tokens += 1;
                emitted_this_round += 1;
                if !callback(&piece) {
                    user_stopped = true;
                    break;
                }
            }

            // Emit bonus token
            if !hit_eos && !user_stopped {
                if let Some(bonus) = spec_result.bonus_token {
                    if LlamaToken(bonus as i32) == target_eos {
                        hit_eos = true;
                    } else {
                        let piece = target_model
                            .token_to_piece(LlamaToken(bonus as i32), &mut decoder, true, None)
                            .map_err(|e| {
                                SwarmError::Inference(format!("Token to piece failed: {e}"))
                            })?;
                        completion_tokens += 1;
                        emitted_this_round += 1;
                        if !callback(&piece) {
                            user_stopped = true;
                        }
                    }
                }
            }

            // === RESYNCHRONIZE ===
            // Target KV cache: keep prompt + accepted + bonus, remove rejected candidates.
            let target_keep = target_pos + emitted_this_round;
            if target_keep < verify_pos {
                let _ = target_ctx.clear_kv_cache_seq(Some(0), Some(target_keep as u32), None);
            }
            target_pos = target_keep;

            // Draft KV cache: rewind to match the actual accepted sequence.
            // The draft's KV has prompt + all previously accepted + gamma draft candidates.
            // We need to keep only prompt + all accepted tokens (old + new) so we rewind
            // to that position. But the draft tokenization may differ from target, so we
            // track draft_pos separately.
            let draft_keep = draft_pos.saturating_sub(num_drafted.saturating_sub(num_accepted));
            // If we rejected some tokens, the bonus token replaced them.
            // We need to trim draft KV to the accepted prefix.
            if num_accepted < num_drafted {
                let _ = draft_ctx.clear_kv_cache_seq(Some(0), Some(draft_keep as u32), None);
                draft_pos = draft_keep;
            }

            // Feed bonus/rejected tokens into draft so it stays synchronized with target.
            if !hit_eos && !user_stopped {
                if let Some(bonus) = spec_result.bonus_token {
                    draft_batch.clear();
                    draft_batch
                        .add(LlamaToken(bonus as i32), draft_pos as i32, &[0], true)
                        .map_err(|e| {
                            SwarmError::Inference(format!("Draft sync batch failed: {e}"))
                        })?;
                    draft_ctx.decode(&mut draft_batch).map_err(|e| {
                        SwarmError::Inference(format!("Draft sync decode failed: {e}"))
                    })?;
                    draft_pos += 1;
                }
            }

            // Save target's last logits as `next_target_probs` for the next iteration.
            // After resync, the target's KV cache ends at target_pos.
            // If we have emitted tokens, we need the target's prediction at the
            // new position. The last verify_probs entry that we keep is at index
            // num_accepted (if bonus was emitted) or we need to re-evaluate.
            // If all were accepted + bonus, verify_probs[num_drafted] is the bonus distribution,
            // and we need the distribution AFTER the bonus token. Let's re-evaluate the
            // last emitted token through target to get fresh logits.
            if !hit_eos && !user_stopped && emitted_this_round > 0 {
                // The target KV cache already has the accepted + bonus tokens.
                // We need the logits from the last position. The last decode in the
                // verify phase or resync already has the right logits in target_ctx
                // if we didn't trim. But after trimming, the logits are stale.
                // Re-evaluate the last accepted/bonus token to get fresh logits.
                let last_emitted_idx = if spec_result.bonus_token.is_some() {
                    // verify_probs has num_drafted + 1 entries. The bonus was sampled from
                    // verify_probs[num_accepted]. The distribution AFTER the bonus is at
                    // verify_probs[num_accepted + 1] if it exists. But we fed candidates
                    // beyond num_accepted, so verify_probs[num_accepted+1..] are conditioned
                    // on wrong tokens. We need fresh logits.
                    // Simplest: just use the target's current logits after the KV trim +
                    // the verify steps we kept.
                    None // need re-eval
                } else {
                    // No bonus, all accepted. verify_probs[num_drafted] is the next distribution.
                    Some(num_drafted)
                };

                if let Some(idx) = last_emitted_idx {
                    next_target_probs = verify_probs[idx].clone();
                } else {
                    // Re-evaluate the last emitted token to regenerate target logits.
                    // The KV cache is correct (trimmed to accepted + bonus).
                    // Decoding a token at the current end position regenerates logits.
                    // But we already decoded the bonus in the verify phase... if we
                    // trimmed some tokens after it, we lost those KV entries.
                    // Actually, the bonus token IS the last token in the verify sequence
                    // at position target_pos - 1. Since we cleared KV after target_keep = target_pos,
                    // the bonus is still in KV. We need logits at that position.
                    // The simplest approach: re-decode the bonus token to regenerate logits.
                    let bonus = spec_result
                        .bonus_token
                        .expect("speculative result always has bonus token after accept");
                    target_batch.clear();
                    target_batch
                        .add(
                            LlamaToken(bonus as i32),
                            (target_pos - 1) as i32,
                            &[0],
                            true,
                        )
                        .map_err(|e| {
                            SwarmError::Inference(format!("Target re-eval failed: {e}"))
                        })?;
                    target_ctx.decode(&mut target_batch).map_err(|e| {
                        SwarmError::Inference(format!("Target re-eval decode failed: {e}"))
                    })?;
                    let logits: Vec<f32> = target_ctx.get_logits()[..target_n_vocab].to_vec();
                    next_target_probs = speculative::softmax(&logits);
                }
            }
        }

        tracing::info!(
            completion_tokens,
            acceptance_rate = %spec_state.acceptance_rate(),
            total_proposed = spec_state.total_proposed,
            total_accepted = spec_state.accepted_count,
            "Speculative decoding complete"
        );

        let gen_result = GenerationResult {
            prompt_tokens,
            completion_tokens,
            finish_reason: if completion_tokens >= max_gen {
                FinishReason::MaxTokens
            } else {
                FinishReason::Stop
            },
        };

        Ok((gen_result, spec_state))
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

/// Metadata extracted from a GGUF file for model info caching.
#[derive(Clone, Debug, Default)]
pub struct GgufModelMeta {
    /// Friendly model name from `general.name`.
    pub name: Option<String>,
    /// Chat template from `tokenizer.chat_template` (Jinja2 format).
    pub chat_template: Option<String>,
    /// BOS token string (resolved from token ID + vocabulary).
    pub bos_token: String,
    /// EOS token string (resolved from token ID + vocabulary).
    pub eos_token: String,
}

/// Extract metadata from a GGUF file (name, chat template, special tokens).
/// Returns None if the file can't be read.
pub fn extract_gguf_metadata(path: &Path) -> Option<GgufModelMeta> {
    let mut file = std::fs::File::open(path).ok()?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).ok()?;

    let name = ct
        .metadata
        .get("general.name")
        .and_then(|v| v.to_string().ok().cloned())
        .filter(|s| !s.is_empty());

    let chat_template = ct
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok().cloned())
        .filter(|s| !s.is_empty());

    // Resolve BOS/EOS token strings from their IDs + vocabulary
    let vocab: Vec<String> = ct
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_string().ok().cloned())
                .collect()
        })
        .unwrap_or_default();

    let bos_id = ct
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok());
    let eos_id = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok());

    let bos_token = bos_id
        .and_then(|id| vocab.get(id as usize).cloned())
        .unwrap_or_default();
    let eos_token = eos_id
        .and_then(|id| vocab.get(id as usize).cloned())
        .unwrap_or_default();

    if chat_template.is_some() {
        tracing::info!(
            has_template = true,
            bos = %bos_token,
            eos = %eos_token,
            "Extracted chat template from GGUF"
        );
    }

    Some(GgufModelMeta {
        name,
        chat_template,
        bos_token,
        eos_token,
    })
}

/// Extract the friendly model name from GGUF `general.name` metadata.
/// Returns None if the file can't be read or the field is absent.
fn extract_gguf_name(path: &Path) -> Option<String> {
    extract_gguf_metadata(path).and_then(|m| m.name)
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
}
