//! Prompt construction, token decoding, and GGUF-header-driven tokenizer
//! helpers used by both the local and distributed execution paths.

use std::collections::HashMap;

use crate::inference::chat_template;
use crate::types::LayerResult;

use super::{PipelineExecutor, LLAMA_FALLBACK_EOS_TOKEN};

/// Extract chat template, BOS, and EOS strings from a GGUF header file on disk.
/// Uses the centralized `GgufTokenizerMeta` extractor.
pub fn template_from_header(
    header_path: &std::path::Path,
) -> Option<(Option<String>, String, String)> {
    let tok = crate::inference::split::GgufTokenizerMeta::from_gguf_file(header_path).ok()?;
    let bos = tok.bos_string();
    let eos = tok.eos_string();
    Some((tok.chat_template, bos, eos))
}

/// Cached vocabulary and tokenizer state for lock-free token decoding during streaming.
/// Extracted once from the model under the mutex, then used for all subsequent decoding
/// without re-acquiring the lock.
pub(super) struct CachedDecoder {
    pub(super) vocab: Vec<String>,
    pub(super) byte_decoder: HashMap<char, u8>,
    pub(super) is_sentencepiece: bool,
    pub(super) has_tokenizer: bool,
}

impl CachedDecoder {
    pub(super) fn decode_tokens(&self, token_ids: &[u32]) -> String {
        if self.has_tokenizer {
            let mut bytes = Vec::new();
            for &id in token_ids {
                if let Some(token_str) = self.vocab.get(id as usize) {
                    bytes.extend(self.decode_token_bytes(token_str));
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        } else if !self.vocab.is_empty() {
            let mut raw = String::new();
            for &id in token_ids {
                if let Some(token_str) = self.vocab.get(id as usize) {
                    raw.push_str(token_str);
                } else {
                    raw.push_str(&format!("[{id}]"));
                }
            }
            decode_bpe_text(&raw)
        } else {
            token_ids
                .iter()
                .map(|id| format!("[{id}]"))
                .collect::<String>()
        }
    }

    fn decode_token_bytes(&self, token_str: &str) -> Vec<u8> {
        // Delegate to the shared decode logic in BpeTokenizer::decode_token_impl.
        crate::inference::tokenizer::decode_token_impl(
            token_str,
            self.is_sentencepiece,
            &self.byte_decoder,
        )
    }
}

impl PipelineExecutor {
    /// Build chat prompt by reading GGUF header from disk (convenience wrapper).
    pub(super) async fn build_prompt(&self) -> String {
        let model_id = &self.request.model_id;
        let header_path = self
            .shared_state
            .model_dir(&model_id.0)
            .join(crate::model::shard::HEADER_FILENAME);
        let header_data = template_from_header(&header_path);
        self.build_prompt_with_header(header_data.as_ref()).await
    }

    /// Build chat prompt using pre-parsed GGUF header data or loaded_model_info fallback.
    pub(super) async fn build_prompt_with_header(
        &self,
        header_data: Option<&(Option<String>, String, String)>,
    ) -> String {
        let model_id = &self.request.model_id;
        if let Some((tmpl, bos, eos)) = header_data {
            let prompt = chat_template::build_prompt_with_model(
                &self.request.messages,
                tmpl.as_deref(),
                bos,
                eos,
                Some(&model_id.0),
            );
            tracing::debug!(
                model = %model_id,
                prompt_len = prompt.len(),
                "DIAG: build_prompt from header"
            );
            return prompt;
        }
        // Fall back to loaded_model_info (singleton, may be wrong model)
        let info = self.shared_state.loaded_model_info.read().await;
        match info.as_ref() {
            Some(i) => chat_template::build_prompt_with_model(
                &self.request.messages,
                i.chat_template.as_deref(),
                &i.bos_token,
                &i.eos_token,
                Some(&model_id.0),
            ),
            // See local_exec: the model id alone is enough for the family
            // fallback, so don't collapse to ChatML.
            None => {
                chat_template::build_prompt(&self.request.messages, None, "", "", Some(&model_id.0))
            }
        }
    }

    pub(super) async fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let entry_key = self.shared_state.find_split_model_key(model_id);
        let entry = entry_key
            .as_ref()
            .and_then(|k| self.shared_state.split_models.get(k));
        if let Some(entry) = entry {
            let vocab = entry.value().vocab.clone().unwrap_or_default();
            if !vocab.is_empty() {
                let mut raw = String::new();
                for &id in token_ids {
                    if let Some(token_str) = vocab.get(id as usize) {
                        raw.push_str(token_str);
                    } else {
                        raw.push_str(&format!("[{id}]"));
                    }
                }
                return decode_bpe_text(&raw);
            }
        }
        // Fallback: return token IDs as text
        token_ids
            .iter()
            .map(|id| format!("[{id}]"))
            .collect::<String>()
    }

    /// Extract prompt token count, EOS tokens, and a cached decoder from metadata.
    /// No model lock needed — uses cached metadata from SplitModelEntry.
    pub(super) async fn extract_model_cache(
        &self,
        prompt: &str,
    ) -> (usize, Vec<u32>, CachedDecoder) {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let entry_key = self.shared_state.find_split_model_key(model_id);
        let entry = entry_key
            .as_ref()
            .and_then(|k| self.shared_state.split_models.get(k));

        if let Some(entry) = entry {
            let eos = entry.value().eos_tokens.clone();
            let vocab = entry.value().vocab.clone().unwrap_or_default();

            // Approximate prompt token count (no tokenizer in-process)
            // Rough estimate: chars / 4 (average BPE token length), minimum 1
            let ptc = (prompt.chars().count() / 4).max(1);

            let decoder = CachedDecoder {
                vocab,
                byte_decoder: HashMap::new(),
                is_sentencepiece: false,
                has_tokenizer: false,
            };

            (ptc, eos, decoder)
        } else {
            // No model loaded — try loading vocab from GGUF header on disk.
            // The header is always available from the probe/manifest exchange.
            let header_path = crate::model::shard::model_dir(
                &self.shared_state.config.node.data_dir,
                &model_id.0,
            )
            .join(crate::model::shard::HEADER_FILENAME);
            if header_path.exists() {
                match Self::decoder_from_header(&header_path) {
                    Some((eos, decoder, tokenizer_opt)) => {
                        let ptc = if let Some(ref tok) = tokenizer_opt {
                            tok.encode(prompt).len()
                        } else {
                            (prompt.chars().count() / 4).max(1)
                        };
                        tracing::debug!(
                            model = %model_id,
                            vocab_size = decoder.vocab.len(),
                            eos_count = eos.len(),
                            "Built decoder from GGUF header (no local model)"
                        );
                        (ptc, eos, decoder)
                    }
                    None => {
                        tracing::warn!(model_id = %model_id, "No GGUF header available — using LLaMA fallback EOS token");
                        let ptc = (prompt.chars().count() / 4).max(1);
                        (
                            ptc,
                            vec![LLAMA_FALLBACK_EOS_TOKEN],
                            CachedDecoder {
                                vocab: Vec::new(),
                                byte_decoder: HashMap::new(),
                                is_sentencepiece: false,
                                has_tokenizer: false,
                            },
                        )
                    }
                }
            } else {
                // No header on disk — try fetching from HuggingFace on-demand
                if let Some(hf_source) = self.shared_state.models.hf_sources.get(model_id) {
                    let model_dir = crate::model::shard::model_dir(
                        &self.shared_state.config.node.data_dir,
                        &model_id.0,
                    );
                    tracing::info!(
                        model = %model_id,
                        repo = %hf_source.repo_id,
                        "Fetching GGUF header from HuggingFace for remote model"
                    );
                    let probe_result = crate::model::huggingface::probe_gguf_file(
                        &hf_source.repo_id,
                        &hf_source.filename,
                        self.shared_state.config.model.shard_size_bytes(),
                    )
                    .await;
                    if let Ok(info) = probe_result {
                        if let Ok(path) = crate::model::huggingface::download_gguf_header(
                            &hf_source.repo_id,
                            &hf_source.filename,
                            &model_dir,
                            info.header_size,
                        )
                        .await
                        {
                            if let Some((eos, decoder, tokenizer_opt)) =
                                Self::decoder_from_header(&path)
                            {
                                let ptc = if let Some(ref tok) = tokenizer_opt {
                                    tok.encode(prompt).len()
                                } else {
                                    prompt.chars().count() / 4
                                };
                                return (ptc, eos, decoder);
                            }
                        }
                    }
                }
                tracing::warn!(model_id = %model_id, "No GGUF header or local model — using LLaMA fallback EOS token");
                let ptc = (prompt.chars().count() / 4).max(1);
                (
                    ptc,
                    vec![LLAMA_FALLBACK_EOS_TOKEN],
                    CachedDecoder {
                        vocab: Vec::new(),
                        byte_decoder: HashMap::new(),
                        is_sentencepiece: false,
                        has_tokenizer: false,
                    },
                )
            }
        }
    }

    /// Build a CachedDecoder + EOS tokens from a GGUF header file on disk.
    /// Used when the node has probe data but no loaded model.
    fn decoder_from_header(
        header_path: &std::path::Path,
    ) -> Option<(
        Vec<u32>,
        CachedDecoder,
        Option<crate::inference::split::SplitTokenizer>,
    )> {
        use crate::inference::split::GgufTokenizerMeta;
        use candle_core::quantized::gguf_file;

        let header_bytes = std::fs::read(header_path).ok()?;
        let mut cursor = std::io::Cursor::new(&header_bytes);
        let ct = gguf_file::Content::read(&mut cursor).ok()?;

        let meta = GgufTokenizerMeta::from_content(&ct);
        if meta.vocab.is_empty() {
            return None;
        }

        let arch = crate::inference::split::gguf_arch_str(&ct);
        let eos_tokens = meta.eos_tokens_with_arch_fallback(&arch);
        let tokenizer = meta.build_tokenizer();

        let decoder = if let Some(ref tok) = tokenizer {
            CachedDecoder {
                vocab: meta.vocab,
                byte_decoder: tok.byte_decoder(),
                is_sentencepiece: tok.is_sentencepiece(),
                has_tokenizer: true,
            }
        } else {
            CachedDecoder {
                vocab: meta.vocab,
                byte_decoder: HashMap::new(),
                is_sentencepiece: false,
                has_tokenizer: false,
            }
        };

        Some((eos_tokens, decoder, tokenizer))
    }

    /// Unseal a LayerResult if it contains sealed token IDs.
    /// Recovers the real token_ids from the sealed envelope using this node's X25519 secret.
    pub(super) fn unseal_result(&self, mut result: LayerResult) -> LayerResult {
        if let Some(ref sealed_bytes) = result.sealed_token_ids {
            match serde_json::from_slice::<crate::types::SealedPrompt>(sealed_bytes) {
                Ok(sealed) => {
                    let local_secret = self.shared_state.identity.x25519_secret();
                    match crate::crypto::pipeline_seal::open_prompt(&sealed, &local_secret) {
                        Ok(plaintext) => {
                            // Deserialize token IDs from the decrypted payload
                            if let Ok(token_ids) = serde_json::from_slice::<Vec<u32>>(&plaintext) {
                                tracing::debug!(
                                    request_id = %result.request_id,
                                    num_tokens = token_ids.len(),
                                    "Pipeline seal: unsealed token IDs from final segment"
                                );
                                result.token_ids = token_ids;
                                result.sealed_token_ids = None;
                            } else {
                                tracing::warn!(
                                    request_id = %result.request_id,
                                    "Pipeline seal: failed to deserialize unsealed token IDs"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                request_id = %result.request_id,
                                error = %e,
                                "Pipeline seal: failed to unseal result — rejecting (no plaintext fallback)"
                            );
                            // Do NOT fall through to plaintext — clear tokens to prevent
                            // accepting unverified data as legitimate decrypted output
                            result.token_ids.clear();
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %result.request_id,
                        error = %e,
                        "Pipeline seal: failed to parse SealedPrompt from result"
                    );
                }
            }
        }
        result
    }
}

fn decode_bpe_text(text: &str) -> String {
    // GPT-2 byte encoder maps bytes 0-255 to Unicode chars.
    // The printable ASCII range (33-126) and some others map to themselves.
    // Others are shifted: byte 0x00 → U+0100 (Ā), 0x01 → U+0101 (ā), etc.
    // Space (0x20) → U+0120 (Ġ), newline (0x0A) → U+010A (Ċ), etc.
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let cp = ch as u32;
        // Printable ASCII and some others map directly
        match cp {
            // Standard printable ASCII
            33..=126 | 161..=172 | 174..=255 => {
                bytes.push(cp as u8);
            }
            // GPT-2 mapped range: U+0100..U+01FF → bytes 0..255
            0x0100..=0x01FF => {
                // The GPT-2 byte encoder maps non-printable/special bytes to U+0100+offset
                // We need to reverse this mapping
                let byte_val = gpt2_unicode_to_byte(cp);
                bytes.push(byte_val);
            }
            _ => {
                // Fallback: try UTF-8 encoding of the character
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Reverse the GPT-2 byte-to-unicode mapping for a Unicode codepoint.
fn gpt2_unicode_to_byte(cp: u32) -> u8 {
    use std::sync::LazyLock;
    static LOOKUP: LazyLock<Vec<u8>> = LazyLock::new(|| {
        // Build the reverse mapping once: the GPT-2 encoder assigns unicode codepoints
        // to bytes that aren't in the "printable" set. The mapping is:
        // printable bytes (33-126, 161-172, 174-255) → themselves
        // remaining bytes 0-32, 127-160, 173 → 256, 257, ... (U+0100, U+0101, ...)
        let mut non_printable = Vec::new();
        for b in 0u16..=255 {
            let is_printable =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            if !is_printable {
                non_printable.push(b as u8);
            }
        }
        non_printable
    });
    let table = &*LOOKUP;

    // non_printable[i] maps to U+0100+i
    let offset = cp.wrapping_sub(0x0100) as usize;
    if offset < table.len() {
        table[offset]
    } else {
        b'?'
    }
}
