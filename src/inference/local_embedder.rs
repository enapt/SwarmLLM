//! Local embedding privacy: performs token→embedding on the requesting node
//! so remote first-segment nodes never see raw token IDs.
//!
//! Loads only the `token_embd.weight` tensor and tokenizer from shard_000.bin,
//! without any transformer layers. Embedding lookup is a simple matmul (~1ms)
//! that adds negligible overhead to the inference path.

use std::path::Path;

use candle_core::{Device, Module, Tensor};
use candle_nn::Embedding;

use crate::error::SwarmError;
use crate::inference::model_arch::ModelArch;
use crate::inference::split::{tensor_to_bytes, SplitTokenizer};

/// Lightweight embedding-only model for local privacy.
/// Holds the dequantized embedding table and tokenizer from GGUF metadata.
pub struct LocalEmbedder {
    embeddings: Embedding,
    tokenizer: Option<SplitTokenizer>,
    hidden_dim: usize,
    arch: ModelArch,
}

// Safety: candle Tensor on CPU is Send + Sync
unsafe impl Send for LocalEmbedder {}
unsafe impl Sync for LocalEmbedder {}

impl LocalEmbedder {
    /// Load the embedding table and tokenizer from a GGUF shard file.
    ///
    /// Opens the GGUF file, reads `token_embd.weight`, dequantizes it,
    /// and builds a tokenizer from GGUF metadata. This is lightweight
    /// compared to loading a full SplitModel (~64MB for a 7B Q4 model).
    pub fn load(shard0_path: &Path) -> Result<Self, SwarmError> {
        let mut file = std::fs::File::open(shard0_path)
            .map_err(|e| SwarmError::Internal(format!("Failed to open shard_000: {e}")))?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF content: {e}")))?;

        let device = Device::Cpu;

        // Get architecture
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        let arch = ModelArch::from_gguf_arch(&arch_str);

        // Get hidden dim
        let hidden_dim = ct
            .metadata
            .get(&format!("{arch_str}.embedding_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2048) as usize;

        // Load and dequantize the embedding table
        let tok_embd = ct
            .tensor(&mut file, "token_embd.weight", &device)
            .map_err(|e| SwarmError::Internal(format!("Failed to load token_embd.weight: {e}")))?;
        let tok_embd = tok_embd
            .dequantize(&device)
            .map_err(|e| SwarmError::Internal(format!("Dequantize embedding: {e}")))?;

        let embeddings = Embedding::new(tok_embd, hidden_dim);

        // Build tokenizer from GGUF metadata (same logic as SplitModel)
        let tokenizer = Self::build_tokenizer(&ct);

        tracing::info!(
            arch = %arch_str,
            hidden_dim,
            has_tokenizer = tokenizer.is_some(),
            path = %shard0_path.display(),
            "Loaded local embedding table for privacy mode"
        );

        Ok(Self {
            embeddings,
            tokenizer,
            hidden_dim,
            arch,
        })
    }

    /// Embed a full prompt string into hidden-state activations.
    ///
    /// Returns serialized tensor bytes (via `tensor_to_bytes`) with shape
    /// `[1, num_tokens, hidden_dim]`. The output is ready to be sent as
    /// `LayerForward.activations` with `pre_embedded = true`.
    pub fn embed_prompt(&self, prompt: &str) -> Result<(Vec<u8>, usize), SwarmError> {
        let token_ids: Vec<i64> = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt)
        } else {
            // Fallback: byte-level encoding
            prompt.bytes().map(|b| b as i64).collect()
        };

        let num_tokens = token_ids.len();
        let input = Tensor::new(&token_ids[..], &Device::Cpu)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;

        let mut emb = self
            .embeddings
            .forward(&input)
            .map_err(|e| SwarmError::Internal(format!("Embedding forward: {e}")))?;

        // Gemma models scale embeddings by sqrt(hidden_dim)
        if self.arch.use_gemma_norm() {
            let scale = (self.hidden_dim as f64).sqrt();
            emb = emb
                .affine(scale, 0.0)
                .map_err(|e| SwarmError::Internal(format!("Embedding scale: {e}")))?;
        }

        let bytes = tensor_to_bytes(&emb)?;
        Ok((bytes, num_tokens))
    }

    /// Embed a single token ID into hidden-state activations.
    ///
    /// Returns serialized tensor bytes with shape `[1, 1, hidden_dim]`.
    pub fn embed_token(&self, token_id: u32) -> Result<Vec<u8>, SwarmError> {
        let input = Tensor::new(&[token_id as i64][..], &Device::Cpu)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;

        let mut emb = self
            .embeddings
            .forward(&input)
            .map_err(|e| SwarmError::Internal(format!("Embedding forward: {e}")))?;

        if self.arch.use_gemma_norm() {
            let scale = (self.hidden_dim as f64).sqrt();
            emb = emb
                .affine(scale, 0.0)
                .map_err(|e| SwarmError::Internal(format!("Embedding scale: {e}")))?;
        }

        tensor_to_bytes(&emb)
    }

    /// Build a tokenizer from GGUF metadata.
    fn build_tokenizer(ct: &candle_core::quantized::gguf_file::Content) -> Option<SplitTokenizer> {
        let vocabulary: Option<Vec<String>> = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            });

        let vocab = vocabulary.as_ref()?;

        let merges_raw: Vec<String> = ct
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let pre_type = ct
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let tokenizer_model = ct
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        if !merges_raw.is_empty() {
            Some(SplitTokenizer::from_bpe(
                vocab,
                &merges_raw,
                &pre_type,
                &tokenizer_model,
            ))
        } else if tokenizer_model == "llama" {
            let scores: Vec<f32> = ct
                .metadata
                .get("tokenizer.ggml.scores")
                .and_then(|v| v.to_vec().ok())
                .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
                .unwrap_or_default();

            let add_space_prefix = ct
                .metadata
                .get("tokenizer.ggml.add_space_prefix")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(true);

            let add_bos_token = ct
                .metadata
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(false);

            if !scores.is_empty() {
                Some(SplitTokenizer::from_sentencepiece(
                    vocab,
                    &scores,
                    add_space_prefix,
                    add_bos_token,
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_embedder_load_from_test_model() {
        // Use the test model's shard_000.bin if available
        let shard0 = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_model/shard_000.bin");
        if !shard0.exists() {
            return; // Skip if test fixture not available
        }

        let embedder = LocalEmbedder::load(&shard0).expect("Failed to load local embedder");
        assert!(embedder.hidden_dim > 0);

        // Test prompt embedding
        let (bytes, num_tokens) = embedder
            .embed_prompt("Hello world")
            .expect("embed_prompt failed");
        assert!(num_tokens > 0);
        assert!(!bytes.is_empty());

        // Test single token embedding
        let token_bytes = embedder.embed_token(1).expect("embed_token failed");
        assert!(!token_bytes.is_empty());
    }
}
