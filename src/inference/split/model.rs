// ── Split model: loads only a range of layers from a GGUF ──

use candle_core::{Device, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::SwarmError;

use super::{LayerVariant, ModelArch, QMatMul, SplitTokenizer};

/// A partial transformer model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
/// Supports multiple architectures: Llama, Qwen2, Gemma 2, Phi-3, Mistral, Qwen 3.5.
pub struct SplitModel {
    /// Token embedding table (only loaded by the first segment).
    pub(super) tok_embeddings: Option<Embedding>,
    /// Transformer layers for this segment's range.
    pub(super) layers: Vec<LayerVariant>,
    /// Final RMSNorm (only loaded by the last segment).
    pub(super) norm: Option<RmsNorm>,
    /// LM head / output projection (only loaded by the last segment).
    pub(super) output: Option<QMatMul>,
    /// Additive causal mask for the no-prefix case, cached at its EXACT size.
    /// Tuple is `(query_len, mask)`; `None` means nothing cached yet. See
    /// `SplitModel::mask` for why this is not a narrowed view of a bigger one.
    pub(super) masks: Option<(usize, Tensor)>,
    /// Layer range this model covers: [start, end) out of total_layers.
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
    /// Hidden dimension (embedding_length).
    pub hidden_dim: usize,
    /// Detected model architecture.
    pub arch: ModelArch,
    /// Device (CPU or CUDA).
    pub(super) device: Device,
    /// Vocabulary from GGUF (token ID → string), for decoding generated tokens.
    pub(super) vocabulary: Option<Vec<String>>,
    /// Tokenizer (BPE or sentencepiece/unigram) built from GGUF metadata.
    pub(super) tokenizer: Option<SplitTokenizer>,
    /// EOS token IDs loaded from GGUF metadata.
    pub(super) eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` (Jinja2 format).
    pub(super) chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    pub(super) bos_token: String,
    /// EOS token string from GGUF metadata.
    pub(super) eos_token: String,
    /// Maximum sequence length for KV cache pre-allocation.
    pub(super) max_seq_len: usize,
    /// Bytes of GPU memory this segment's KV cache may occupy in total,
    /// across every concurrent request. `None` on CPU, and on CUDA when free
    /// VRAM could not be read — an unknown budget must not become a zero one.
    ///
    /// Checked before a forward claims another growth quantum. This replaced a
    /// load-time clamp that shrank the model's usable context so ONE
    /// conversation at full length would fit: that made every user's context
    /// shorter to guard against a case most never reach, and did not bound
    /// concurrency at all.
    pub(super) kv_budget_bytes: Option<u64>,
    /// KV bytes one sequence position costs across this segment's layers.
    /// Paired with `kv_budget_bytes`; 0 when unknown.
    pub(super) kv_bytes_per_token: u64,
    /// Pre-computed KV cache store key: "{layer_start}-{layer_end}-{total_layers}".
    /// Avoids a `format!` allocation on every forward pass.
    pub(super) kv_model_key: String,
    /// Gemma 2 final logit soft-capping value (e.g. 30.0).
    pub(super) final_logit_softcap: Option<f32>,
    /// How often `forward_batch` actually batched, versus fell back to running
    /// its items one at a time.
    ///
    /// The fallback is silent and its conditions are easy to meet by accident —
    /// requests only share an `index_pos` while they stay in lockstep, which
    /// concurrent requests stop doing as soon as one starts a token earlier
    /// than another. Without a count there is no way to tell a node that is
    /// batching from one that has been running sequentially all along, and
    /// measured throughput does not distinguish them either: batching four
    /// streams was worth about 20% here, which is well inside the noise of a
    /// single benchmark (`docs/FUTURE_WORK.md`).
    pub(super) batch_calls: u64,
    pub(super) batch_fellback: u64,
    /// When the counters above were last reported. `None` until the first
    /// multi-request call, so an idle model never reports.
    pub(super) batch_stats_reported_at: Option<std::time::Instant>,
}

impl SplitModel {
    /// The model's context window in tokens.
    ///
    /// Exposed so a caller that holds the WHOLE prompt can check it before
    /// starting a chunked prefill. The executor's own guard sees one chunk at a
    /// time and can only report a position just past the limit, which is the
    /// same number for every over-long prompt.
    pub fn context_window(&self) -> usize {
        self.max_seq_len
    }

    /// Return a reference to the loaded vocabulary, if available.
    pub fn vocab(&self) -> Option<&Vec<String>> {
        self.vocabulary.as_ref()
    }

    /// Return a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&SplitTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the device this model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Record one `forward_batch` call and periodically report how much of the
    /// batching is real.
    ///
    /// Reported at INFO, not debug: nodes run at info, and a metric nobody sees
    /// is the same as no metric — the whole reason this fallback went unnoticed
    /// is that it is silent.
    ///
    /// **Paced by time, not only by call count.** It used to report every 256
    /// calls, and a realistic session never gets there: four people asking one
    /// question each is around 96, so the counter reset with the worker and the
    /// line never appeared. Verifying the batching fix on a real node found
    /// exactly that — the answers were right, and the one diagnostic that could
    /// have said whether the batched path ran at all printed nothing.
    ///
    /// A diagnostic that cannot fire during normal use is not a diagnostic. It
    /// now reports on whichever comes first, so a short burst is visible and a
    /// busy node still gets at most one line per `BATCH_STATS_MIN_GAP`.
    pub(super) fn note_batch_attempt(&mut self, fell_back: bool) {
        /// Roughly one line per few seconds of steady decoding.
        const BATCH_STATS_EVERY: u64 = 256;
        /// Floor on spacing, so a busy node cannot be flooded.
        const BATCH_STATS_MIN_GAP: std::time::Duration = std::time::Duration::from_secs(20);

        self.batch_calls += 1;
        if fell_back {
            self.batch_fellback += 1;
        }
        let now = std::time::Instant::now();
        let due_by_time = match self.batch_stats_reported_at {
            // First multi-request call since load: start the clock rather than
            // reporting a single sample, which says nothing.
            None => {
                self.batch_stats_reported_at = Some(now);
                false
            }
            Some(last) => now.duration_since(last) >= BATCH_STATS_MIN_GAP,
        };
        if !due_by_time && !self.batch_calls.is_multiple_of(BATCH_STATS_EVERY) {
            return;
        }
        self.batch_stats_reported_at = Some(now);
        let batched = self.batch_calls - self.batch_fellback;
        tracing::info!(
            model_key = %self.kv_model_key,
            calls = self.batch_calls,
            batched,
            fell_back = self.batch_fellback,
            batched_pct = (batched as f64 * 100.0 / self.batch_calls as f64).round() as u32,
            "DIAG: forward_batch — share of multi-request calls that actually batched \
             (the rest ran their items one at a time because the requests were not at \
             the same position)"
        );
    }

    /// `(calls, fell_back)` for `forward_batch` since this model was loaded.
    pub fn batch_stats(&self) -> (u64, u64) {
        (self.batch_calls, self.batch_fellback)
    }

    /// Return the KV cache model key (used for cache cleanup).
    pub fn kv_model_key(&self) -> &str {
        &self.kv_model_key
    }

    /// Return the EOS token IDs loaded from GGUF metadata.
    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    /// Return the chat template from GGUF metadata, if available.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Return the BOS token string from GGUF metadata.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// Return the EOS token string from GGUF metadata.
    pub fn eos_token_str(&self) -> &str {
        &self.eos_token
    }

    /// Whether this segment has the embedding table (first segment).
    pub fn is_first(&self) -> bool {
        self.tok_embeddings.is_some()
    }

    /// Whether this segment has the output projection (last segment).
    pub fn is_last(&self) -> bool {
        self.output.is_some()
    }

    /// Tokenize a prompt to token IDs as `Vec<u32>`. Used by the prefix
    /// cache for key lookup and by the batched-decode path. Cheap — it's
    /// the same work `tokenize()` does, minus the tensor construction.
    pub fn encode_ids(&self, prompt: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer
                .encode(prompt)
                .into_iter()
                .map(|t| t as u32)
                .collect()
        } else {
            prompt.bytes().map(|b| b as u32).collect()
        }
    }

    /// Build an input tensor from an already-tokenized token id slice.
    /// Mirrors `tokenize()`'s output shape (1, seq_len), i64 dtype. Returns
    /// an error if `ids` is empty.
    pub fn tensor_from_ids(&self, ids: &[u32]) -> Result<Tensor, SwarmError> {
        if ids.is_empty() {
            return Err(SwarmError::Internal(
                "tensor_from_ids: empty token slice".into(),
            ));
        }
        let as_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        Tensor::new(as_i64.as_slice(), &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))
    }

    /// Tokenize a prompt and return (token_ids_tensor, num_tokens).
    ///
    /// Returns the token IDs as an I64 tensor with shape (1, seq_len),
    /// suitable for passing directly to `forward()` which handles embedding.
    pub fn tokenize(&self, prompt: &str) -> Result<(Tensor, usize), SwarmError> {
        let token_ids: Vec<i64> = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt)
        } else {
            prompt.bytes().map(|b| b as i64).collect()
        };
        let num_tokens = token_ids.len();
        // DIAG: dump token IDs for debugging tokenizer issues. Debug-level
        // (not info) — fires on every prompt and would flood default-level
        // logs while leaking prompt content. Other DIAG instrumentation in
        // this module already uses debug! / trace!.
        tracing::debug!(
            num_tokens,
            tokens = ?&token_ids[..token_ids.len().min(30)],
            "DIAG: tokenize result"
        );
        let input = Tensor::new(&token_ids[..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;
        Ok((input, num_tokens))
    }

    /// Create a single-token tensor for autoregressive decoding.
    ///
    /// Returns an I64 tensor with shape (1, 1), suitable for `forward()`.
    pub fn token_tensor(&self, token_id: u32) -> Result<Tensor, SwarmError> {
        Tensor::new(&[token_id as i64][..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))
    }

    /// Estimate GPU memory usage in MB for this model segment.
    ///
    /// Uses a rough heuristic: for quantized models (Q4/Q5/Q6), each parameter
    /// uses ~0.5-0.75 bytes. We estimate based on `num_layers * hidden_dim^2 * 4`
    /// (4 weight matrices per layer: Q, K, V, O + MLP) with a quantization factor.
    pub fn estimate_vram_mb(&self) -> u64 {
        let num_layers = self.layers.len() as u64;
        let hidden = self.hidden_dim as u64;

        // Each layer has roughly:
        //   4 attention matrices (Q, K, V, O): ~4 * hidden^2 params
        //   3 MLP matrices (gate, up, down): gate + up = 2 * hidden * 4*hidden, down = 4*hidden * hidden
        //   So MLP ≈ 12 * hidden^2 params
        //   Total per layer ≈ 16 * hidden^2 params
        //   Quantized at ~0.5 bytes per param (Q4)
        let params_per_layer = 16 * hidden * hidden;
        let bytes_per_param_quantized: f64 = 0.5;
        let layer_bytes = (params_per_layer as f64 * bytes_per_param_quantized) as u64;
        let mut total_bytes = num_layers * layer_bytes;

        // Add embedding table if loaded (~vocab_size * hidden * 2 bytes for f16)
        if self.tok_embeddings.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        // Add output projection if loaded
        if self.output.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        total_bytes / (1024 * 1024) // Convert to MB
    }
}
