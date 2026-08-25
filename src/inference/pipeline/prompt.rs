//! Prompt construction, token decoding, and GGUF-header-driven tokenizer
//! helpers used by both the local and distributed execution paths.

use std::collections::HashMap;

use crate::inference::chat_template;
use crate::types::LayerResult;

use super::PipelineExecutor;

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

    /// The prompt AND the stop strings its chat template implies, resolved
    /// together from one read of the model's header.
    ///
    /// They are returned as a pair because separating them is the recurring
    /// bug. A model ends its turn with a template marker (`<|user|>`,
    /// `<|im_end|>`, `<|eot_id|>`) rather than only the tokenizer's EOS id, so a
    /// path that renders the template but forwards `sampling_params` untouched
    /// lets the model run to `max_tokens` emitting those markers as **visible
    /// text**, then carry on inventing the next turn — the user sees a control
    /// token followed by a conversation they did not have.
    ///
    /// `with_template_stops` was written for exactly this and its own comment
    /// names the three paths that need it: "the streaming split path, the
    /// non-streaming split path, and the router paths". The router paths were
    /// never wired, so `remote_generate` — the fast path a node takes whenever
    /// ONE peer holds the whole model, i.e. the common case for a machine that
    /// stores nothing itself — built a templated prompt and sent the caller's
    /// params through unchanged. Observed on TinyLlama served across the
    /// network: `'Count from six to ten.\n<|user|> Can you give me a summary of
    /// the text material I just read?'`. The serving side cannot rescue it —
    /// it only truncates the list it is handed.
    ///
    /// Returning one value keeps them in step: a caller cannot take the prompt
    /// and forget the stops.
    pub(super) async fn build_prompt_and_stops(
        &self,
        params: swarmllm_types::inference::SamplingParams,
    ) -> (String, swarmllm_types::inference::SamplingParams) {
        let model_id = &self.request.model_id;
        let header_path = self
            .shared_state
            .model_dir(&model_id.0)
            .join(crate::model::shard::HEADER_FILENAME);
        let header_data = template_from_header(&header_path);
        let prompt = self.build_prompt_with_header(header_data.as_ref()).await;

        // Resolve the template the same way `build_prompt_with_header` does, so
        // the stops always describe the template the prompt was actually built
        // from — including the `loaded_model_info` branch, which is only
        // legitimate when that singleton really describes this model (#294).
        let template: Option<String> = match header_data {
            Some((ref tmpl, _, _)) => tmpl.clone(),
            None => {
                let info = self.shared_state.loaded_model_info.read().await;
                info.as_ref()
                    .filter(|i| loaded_info_describes(&i.name, &model_id.0))
                    .and_then(|i| i.chat_template.clone())
            }
        };
        (
            prompt,
            chat_template::with_template_stops(params, template.as_deref()),
        )
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
        // `loaded_model_info` is a SINGLETON describing whichever model this
        // node last loaded — which is frequently not the model being asked for.
        // Its template may only be used when it genuinely describes this model.
        //
        // Using it unconditionally meant a distributed request could be prompted
        // in another family's format entirely: observed 2026-07-29 with a
        // Phi-3.5 request rendered in Llama-3's template, which answered a
        // one-word question with `<|start_header_id|>system<|end_header_id|>`
        // in the visible text and a prompt 4x the expected length. A model
        // answers in whatever format it was asked in (gotcha #169), so a
        // foreign template is a correctness bug, not a cosmetic one.
        //
        // When it does not match, the model id alone drives the family fallback
        // — the same choice `local_exec` makes, and always better than a
        // template belonging to a different model.
        let info = self.shared_state.loaded_model_info.read().await;
        let matching = info
            .as_ref()
            .filter(|i| loaded_info_describes(&i.name, &model_id.0));
        match matching {
            Some(i) => chat_template::build_prompt_with_model(
                &self.request.messages,
                i.chat_template.as_deref(),
                &i.bos_token,
                &i.eos_token,
                Some(&model_id.0),
            ),
            None => {
                if info.is_some() {
                    tracing::debug!(
                        model = %model_id,
                        "DIAG: loaded_model_info describes a different model; \
                         using the name-based template fallback"
                    );
                }
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
                // Decode through the same tokenizer-aware path
                // `extract_model_cache` builds, NOT the raw `decode_bpe_text`
                // fallback. That fallback is a GPT-2 byte decoder: U+2581 is
                // outside every range it maps, so a SentencePiece model leaks
                // the word-boundary marker into the reply.
                //
                // This is the sibling of the decoder fixed in v0.3.46. That fix
                // corrected `CachedDecoder`'s CONSTRUCTION and left this
                // function, which `distributed.rs` calls whenever it has no
                // cached decoder — so the identical corruption survived on the
                // distributed path. Observed 2026-07-29: a Phi-3.5 answer over
                // the network contained `"a▁"` while the same question answered
                // locally was clean.
                let decoder = match self.shared_state.standalone_tokenizer(model_id) {
                    Some(tok) => CachedDecoder {
                        vocab,
                        byte_decoder: tok.byte_decoder(),
                        is_sentencepiece: tok.is_sentencepiece(),
                        has_tokenizer: true,
                    },
                    None => CachedDecoder {
                        vocab,
                        byte_decoder: HashMap::new(),
                        is_sentencepiece: false,
                        has_tokenizer: false,
                    },
                };
                return decoder.decode_tokens(token_ids);
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

            // Use this node's standalone tokenizer when one can be loaded.
            //
            // These flags used to be hardcoded false, on the reasoning that
            // there is "no tokenizer in-process" — true when written, but
            // `standalone_tokenizer` now lazily builds one from
            // `gguf_header.bin` for exactly this purpose. Leaving them false
            // meant `decode_tokens` could ONLY take its fallback, which
            // concatenates raw vocabulary entries and runs them through
            // `decode_bpe_text` — a GPT-2 byte decoder. U+2581 is outside every
            // range that maps, so for a SentencePiece model the word-boundary
            // marker survived into the reply: Phi-3.5 answered
            // `A▁distributed▁system▁is…` locally while the same node returned
            // clean text for the same prompt over the network, because the
            // network path decodes on the serving side with a real tokenizer
            // (observed 2026-07-29). Byte-fallback tokens leaked the same way,
            // which is where the stray `<0x0A>` came from.
            let tokenizer = self.shared_state.standalone_tokenizer(model_id);
            let decoder = match tokenizer {
                Some(tok) => CachedDecoder {
                    vocab,
                    byte_decoder: tok.byte_decoder(),
                    is_sentencepiece: tok.is_sentencepiece(),
                    has_tokenizer: true,
                },
                // Still no tokenizer: the fallback below is the best we can do,
                // but it must not pretend the vocabulary is GPT-2-encoded when
                // it is not — see `decode_tokens`.
                None => CachedDecoder {
                    vocab,
                    byte_decoder: HashMap::new(),
                    is_sentencepiece: false,
                    has_tokenizer: false,
                },
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
                        // No vocabulary, so no way to verify ANY end-of-turn id.
                        // This used to hand back Llama-2's `</s>` (id 2), which
                        // is an ordinary token in every later family — `#` in
                        // Qwen2.5 — so a coding reply beginning with a Rust
                        // attribute or a markdown heading was cut to ONE token
                        // and reported as `finish_reason: "stop"`. Silently
                        // truncating a good answer is worse than letting it run
                        // to `max_tokens`, so say we do not know.
                        tracing::warn!(
                            model_id = %model_id,
                            "No GGUF header available — this node cannot tell where this model \
                             ends its turn, so the reply will run until max_tokens or a stop \
                             sequence. Fetch the model's header to fix this"
                        );
                        let ptc = (prompt.chars().count() / 4).max(1);
                        (
                            ptc,
                            Vec::new(),
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
                // Same reasoning as the branch above: with neither a loaded
                // model nor a header there is no vocabulary to check an
                // end-of-turn id against, and guessing Llama-2's `</s>` (id 2)
                // silently truncates every later family at its first `#`.
                tracing::warn!(
                    model_id = %model_id,
                    "No GGUF header or local model — this node cannot tell where this model ends \
                     its turn, so the reply will run until max_tokens or a stop sequence"
                );
                let ptc = (prompt.chars().count() / 4).max(1);
                (
                    ptc,
                    Vec::new(),
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

/// Whether the singleton `loaded_model_info` genuinely describes `model_id`.
///
/// The stored `name` comes from the manifest and the requested id comes from the
/// API, so they can differ in case and in separator style for the same model.
/// The comparison is deliberately conservative: a false negative costs only the
/// name-based family fallback, while a false positive prompts the model in
/// another family's format.
pub(super) fn loaded_info_describes(loaded_name: &str, model_id: &str) -> bool {
    fn norm(s: &str) -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect()
    }
    !loaded_name.is_empty() && norm(loaded_name) == norm(model_id)
}

#[cfg(test)]
mod loaded_info_tests {
    use super::loaded_info_describes;

    #[test]
    fn same_model_matches_across_separator_and_case_differences() {
        assert!(loaded_info_describes(
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",
            "tinyllama-1.1b-chat-v1.0.q4-k-m"
        ));
        assert!(loaded_info_describes("phi-3.5-mini", "phi-3.5-mini"));
    }

    /// The bug this guards: a Phi request must never be handed the template of
    /// whatever model happened to be loaded last.
    #[test]
    fn different_models_never_match() {
        assert!(!loaded_info_describes(
            "meta-llama-3.1-8b-instruct",
            "phi-3.5-mini-instruct.q4-k-m"
        ));
        assert!(!loaded_info_describes(
            "tinyllama-1.1b-chat-v1.0.q4-k-m",
            "phi-3.5-mini-instruct.q4-k-m"
        ));
        assert!(!loaded_info_describes("", "phi-3.5-mini-instruct.q4-k-m"));
    }

    /// Different quantisations of the same base model are different files with
    /// potentially different metadata — treat them as distinct.
    #[test]
    fn different_quantisations_do_not_match() {
        assert!(!loaded_info_describes(
            "phi-3.5-mini-instruct.q8-0",
            "phi-3.5-mini-instruct.q4-k-m"
        ));
    }
}

#[cfg(test)]
mod decoder_tests {
    use super::CachedDecoder;
    use std::collections::HashMap;

    fn spm_vocab() -> Vec<String> {
        vec![
            "<unk>".to_string(),
            "\u{2581}Hello".to_string(),
            "\u{2581}world".to_string(),
        ]
    }

    /// A SentencePiece vocabulary must decode the word-boundary marker to a
    /// space. Running these through the GPT-2 byte decoder instead leaves
    /// U+2581 in the visible reply — the v0.3.46 corruption, which survived on
    /// the distributed path because a sibling function kept the raw fallback.
    #[test]
    fn sentencepiece_marker_becomes_a_space() {
        let d = CachedDecoder {
            vocab: spm_vocab(),
            byte_decoder: HashMap::new(),
            is_sentencepiece: true,
            has_tokenizer: true,
        };
        let out = d.decode_tokens(&[1, 2]);
        assert!(
            !out.contains('\u{2581}'),
            "U+2581 must never reach the reply, got {out:?}"
        );
        assert_eq!(out, " Hello world", "got {out:?}");
    }
}
