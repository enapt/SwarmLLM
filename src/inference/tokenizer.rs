//! BPE and SentencePiece tokenizers from GGUF metadata.

use std::collections::HashMap;

// ── BPE Tokenizer from GGUF merges ──

/// The GPT-2 pre-tokenizer pattern. Also the fallback for an unrecognised
/// `tokenizer.ggml.pre`: it is a sane general byte-level BPE splitter, unlike a
/// plain whitespace split, which strands every leading space as its own token.
const PRE_GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// The Llama-3 pattern. Shared verbatim by DBRX / SMAUG / CHATGLM4 in
/// llama.cpp, which is why those names map here too.
const PRE_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2. Identical to [`PRE_LLAMA3`] except numbers are split one digit at a
/// time (`\p{N}`) rather than in groups of up to three.
const PRE_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen3.5 — like Qwen2 but combining marks stay attached to their base letter.
const PRE_QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GPT-4o, and Llama-4. Splits runs of case so `SCREAMINGCase` breaks sensibly.
const PRE_GPT4O: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizer regex patterns for a GGUF `tokenizer.ggml.pre` value.
///
/// Mirrors the `regex_exprs` table in llama.cpp's `llm_tokenizer_bpe`
/// constructor (`src/llama-vocab.cpp`) and its `tokenizer_pre` string → enum
/// mapping. Several pre-types apply a LIST of patterns *in sequence*, each pass
/// re-splitting the fragments produced by the previous one — hence the slice.
///
/// Returns `None` for an unrecognised name so the caller can say so out loud;
/// llama.cpp refuses to load the model in that case, which is too harsh for a
/// node that would otherwise serve the rest of the swarm.
///
/// **Adding a model family means adding its name here.** Getting this wrong is
/// not a hard failure — it silently inflates every prompt and feeds the model
/// token sequences it was never trained on, which is why the fallback warns.
fn pre_tokenizer_patterns(pre_type: &str) -> Option<&'static [&'static str]> {
    Some(match pre_type {
        // The Llama-3 pattern and everything llama.cpp routes to it.
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" | "glm4" | "chatglm-bpe" | "jais-2" => {
            &[PRE_LLAMA3]
        }
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2" | "hunyuan"
        | "solar-open" => &[PRE_QWEN2],
        "qwen35" => &[PRE_QWEN35],
        "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" => &[PRE_GPT4O],
        "gpt-2" | "gpt2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
        | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "exaone4" | "jina-v1-en"
        | "jina-v2-code" | "roberta-bpe" | "mpt" | "olmo" | "jais" | "trillion"
        | "granite-docling" => &[PRE_GPT2],
        // Sequential lists: each entry re-splits the previous pass's fragments.
        "default" => &[
            r"[\p{P}\$\+<=>\^~\|]+",
            PRE_GPT2,
            r"\p{N}+",
            r"[0-9][0-9][0-9]",
        ],
        "falcon" => &[r"[\p{P}\$\+<=>\^~\|`]+", PRE_GPT2, r"[0-9][0-9][0-9]"],
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b"
        | "mellum2" => &[r"\p{N}", PRE_GPT2],
        "deepseek-coder" => &[
            r"[\r\n]",
            r"\s?\p{L}+",
            r"\s?\p{P}+",
            r"[一-龥ࠀ-一가-퟿]+",
            r"\p{N}",
        ],
        "whitespace" => &[r"\S+|\s+"],
        _ => return None,
    })
}

/// BPE tokenizer built from GGUF metadata.
/// Supports both GPT-2/Qwen2 byte-level BPE and SentencePiece BPE (LLaMA).
pub struct BpeTokenizer {
    /// token string → token ID
    token_to_id: HashMap<String, u32>,
    /// Merge pair "left\0right" → merge rank (lower = higher priority).
    /// Uses concatenated string key with \0 separator for zero-allocation lookups.
    merge_ranks: HashMap<String, usize>,
    /// Byte → GPT-2 unicode character mapping
    byte_encoder: [char; 256],
    /// GPT-2 unicode char → byte reverse mapping
    byte_decoder: HashMap<char, u8>,
    /// Pre-tokenization regex patterns, applied in sequence: each pass
    /// re-splits the fragments the previous pass produced.
    pre_tok_res: Vec<fancy_regex::Regex>,
    /// Special tokens sorted by length descending (for matching)
    special_tokens: Vec<(String, u32)>,
    /// Whether this is a SentencePiece tokenizer (uses ▁ for spaces, no byte encoding)
    is_sentencepiece: bool,
}

impl BpeTokenizer {
    /// Number of entries in the vocabulary. See
    /// `SplitTokenizer::vocab_size` for why this is a network-cost figure.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Build a BPE tokenizer from GGUF vocabulary tokens, merge rules,
    /// pre-tokenizer type, and tokenizer model type.
    pub(crate) fn from_gguf(
        tokens: &[String],
        merges_raw: &[String],
        pre_type: &str,
        tokenizer_model: &str,
    ) -> Self {
        let is_sentencepiece = tokenizer_model == "llama";
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Build merge rank lookup: "left\0right" → rank (zero-alloc lookups via reusable buffer)
        let mut merge_ranks = HashMap::with_capacity(merges_raw.len());
        for (rank, line) in merges_raw.iter().enumerate() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let mut key = String::with_capacity(parts[0].len() + 1 + parts[1].len());
                key.push_str(parts[0]);
                key.push('\0');
                key.push_str(parts[1]);
                merge_ranks.insert(key, rank);
            }
        }

        // Build GPT-2 byte encoder
        let (byte_encoder, byte_decoder) = build_gpt2_byte_encoder();

        // Pre-tokenization patterns for this model family. An unrecognised name
        // is reported rather than silently mis-tokenising every prompt: the
        // wrong splitter still produces valid-looking tokens, just far more of
        // them, and none of the space-prefixed ones the model was trained on.
        let patterns = pre_tokenizer_patterns(pre_type).unwrap_or_else(|| {
            if !is_sentencepiece {
                tracing::warn!(
                    pre_tokenizer = %pre_type,
                    "Unknown GGUF pre-tokenizer; falling back to the GPT-2 splitter. \
                     Prompts for this model may use more tokens than they should. \
                     Please report this pre-tokenizer name so it can be added."
                );
            }
            &[PRE_GPT2]
        });
        let pre_tok_res: Vec<fancy_regex::Regex> = patterns
            .iter()
            .filter_map(|p| match fancy_regex::Regex::new(p) {
                Ok(re) => Some(re),
                Err(e) => {
                    tracing::error!(
                        pre_tokenizer = %pre_type,
                        error = %e,
                        "Pre-tokenizer pattern failed to compile; skipping it"
                    );
                    None
                }
            })
            .collect();

        // Collect special tokens (e.g., <|im_start|>, <|im_end|>, <s>, </s>, <unk>,
        // <bos>, <eos>, <start_of_turn>, <end_of_turn>)
        let mut special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter(|(t, _)| {
                (t.starts_with("<|") && t.ends_with("|>"))
                    || (t.starts_with('<') && t.ends_with('>') && !t.contains(' ') && t.len() <= 20)
            })
            .map(|(t, &id)| (t.clone(), id))
            .collect();
        // Sort by length descending for longest-match-first
        special_tokens.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));

        Self {
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre_tok_res,
            special_tokens,
            is_sentencepiece,
        }
    }

    /// Encode a string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return vec![];
        }

        // 1. Split on special tokens first
        let segments = self.split_special_tokens(text);
        let mut all_ids = Vec::new();

        for (segment, is_special) in &segments {
            if *is_special {
                if let Some(&id) = self.token_to_id.get(segment.as_str()) {
                    all_ids.push(id as i64);
                }
            } else if self.is_sentencepiece {
                // SentencePiece: replace spaces with ▁, then BPE encode
                // SentencePiece convention: leading space becomes ▁
                let normalized = format!("\u{2581}{}", segment.replace(' ', "\u{2581}"));
                all_ids.extend(self.bpe_encode_word(&normalized));
            } else {
                // GPT-2: pre-tokenize with regex, then BPE encode each piece
                let pre_tokens = self.pre_tokenize(segment);
                for pre_tok in &pre_tokens {
                    all_ids.extend(self.bpe_encode_word(pre_tok));
                }
            }
        }

        all_ids
    }

    /// Split text at special token boundaries.
    fn split_special_tokens(&self, text: &str) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Check if remaining starts with any special token
            let mut found = false;
            for (special, _) in &self.special_tokens {
                if remaining.starts_with(special.as_str()) {
                    result.push((special.clone(), true));
                    remaining = &remaining[special.len()..];
                    found = true;
                    break;
                }
            }
            if !found {
                // Find next special token occurrence
                let next_pos = self
                    .special_tokens
                    .iter()
                    .filter_map(|(s, _)| remaining.find(s.as_str()))
                    .min();
                match next_pos {
                    Some(pos) => {
                        result.push((remaining[..pos].to_string(), false));
                        remaining = &remaining[pos..];
                    }
                    None => {
                        result.push((remaining.to_string(), false));
                        remaining = "";
                    }
                }
            }
        }
        result
    }

    /// Pre-tokenize text using the model's regex patterns.
    ///
    /// Patterns are applied in sequence: each pass re-splits the fragments the
    /// previous pass produced, matching llama.cpp's `unicode_regex_split`.
    ///
    /// Text that a pattern does NOT match is kept as its own fragment rather
    /// than dropped. Several of these patterns cover only part of their input
    /// by design (the GPT-2 one does not match interior whitespace runs), so
    /// discarding the gaps would delete characters from the prompt outright.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut fragments: Vec<&'a str> = vec![text];
        for re in &self.pre_tok_res {
            let mut next: Vec<&'a str> = Vec::with_capacity(fragments.len());
            for frag in fragments.drain(..) {
                let mut cursor = 0;
                while cursor < frag.len() {
                    match re.find_from_pos(frag, cursor) {
                        Ok(Some(m)) if m.end() > m.start() => {
                            if m.start() > cursor {
                                next.push(&frag[cursor..m.start()]);
                            }
                            next.push(&frag[m.start()..m.end()]);
                            cursor = m.end();
                        }
                        // No further match, or a zero-width one that would not
                        // advance the cursor: keep the remainder intact.
                        _ => break,
                    }
                }
                if cursor < frag.len() {
                    next.push(&frag[cursor..]);
                }
            }
            fragments = next;
        }
        fragments
    }

    /// BPE encode a single pre-token word.
    /// For GPT-2: converts bytes → GPT-2 unicode chars, then applies BPE merges.
    /// For SentencePiece: uses raw unicode chars directly with ▁ for leading spaces.
    fn bpe_encode_word(&self, word: &str) -> Vec<i64> {
        let chars: Vec<String> = if self.is_sentencepiece {
            // SentencePiece: each character is used as-is (▁ already inserted by pre_tokenize)
            word.chars().map(|c| c.to_string()).collect()
        } else {
            // GPT-2: convert each byte to its GPT-2 unicode character
            word.bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect()
        };

        if chars.is_empty() {
            return vec![];
        }

        // Single char: direct lookup
        if chars.len() == 1 {
            return vec![self.token_to_id.get(&chars[0]).copied().unwrap_or(0) as i64];
        }

        // Apply BPE merges using the standard algorithm:
        // Repeatedly find the highest-priority (lowest rank) merge pair and apply it.
        // Uses a reusable lookup buffer to avoid String allocations in the scan loop.
        let mut symbols = chars;
        let mut lookup_buf = String::new();
        loop {
            // Find the pair with the lowest merge rank (zero-allocation scan)
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..symbols.len() - 1 {
                lookup_buf.clear();
                lookup_buf.push_str(&symbols[i]);
                lookup_buf.push('\0');
                lookup_buf.push_str(&symbols[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&lookup_buf) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges applicable
            }

            // Apply the merge: combine symbols[best_idx] and symbols[best_idx+1]
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);

            if symbols.len() == 1 {
                break;
            }
        }

        // Convert BPE tokens to IDs
        symbols
            .iter()
            .map(|t| self.token_to_id.get(t).copied().unwrap_or(0) as i64)
            .collect()
    }

    /// Decode a BPE token string back to UTF-8 bytes.
    /// For GPT-2: reverses the GPT-2 unicode byte encoding.
    /// For SentencePiece: converts ▁ back to space, handles <0xNN> byte tokens.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        decode_token_impl(token_str, self.is_sentencepiece, &self.byte_decoder)
    }

    /// Return a reference to the byte decoder mapping (for caching outside the lock).
    pub fn byte_decoder(&self) -> &HashMap<char, u8> {
        &self.byte_decoder
    }

    /// Whether this tokenizer uses SentencePiece encoding (vs GPT-2 byte BPE).
    pub fn is_sentencepiece(&self) -> bool {
        self.is_sentencepiece
    }
}

/// Unified tokenizer that wraps either our BPE tokenizer or a SentencePiece
/// merge-based tokenizer built from GGUF vocab + scores.
pub struct SplitTokenizer {
    kind: TokenizerKind,
    /// BOS token id from the GGUF's `tokenizer.ggml.bos_token_id`.
    bos_id: Option<u32>,
    /// Whether to prepend BOS at position 0.
    add_bos_token: bool,
}

enum TokenizerKind {
    Bpe(Box<BpeTokenizer>),
    SentencePiece(SpmTokenizer),
}

/// Decode a BPE token string back to UTF-8 bytes (shared logic for BpeTokenizer and CachedDecoder).
/// For GPT-2: reverses the GPT-2 unicode byte encoding.
/// For SentencePiece: converts ▁ back to space, handles <0xNN> byte tokens.
pub fn decode_token_impl(
    token_str: &str,
    is_sentencepiece: bool,
    byte_decoder: &HashMap<char, u8>,
) -> Vec<u8> {
    if is_sentencepiece {
        // Handle byte fallback tokens like <0x0A> (newline)
        if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
            if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                return vec![byte];
            }
        }
        // Special tokens like <s>, </s>, <unk> → empty (don't emit)
        if token_str.starts_with('<') && token_str.ends_with('>') {
            return vec![];
        }
        // SentencePiece: ▁ (U+2581) → space, everything else is raw UTF-8
        token_str.replace('\u{2581}', " ").into_bytes()
    } else {
        // GPT-2: reverse byte encoding
        token_str
            .chars()
            .map(|ch| byte_decoder.get(&ch).copied().unwrap_or(b'?'))
            .collect()
    }
}

/// SentencePiece merge-based tokenizer matching llama.cpp's SPM algorithm.
///
/// Uses greedy best-first merge: start with character-level segmentation,
/// iteratively merge adjacent pairs whose concatenation exists in the vocab,
/// ordered by score (highest first). This matches llama.cpp's behavior and
/// produces correct tokenization for GGUF SentencePiece models.
pub struct SpmTokenizer {
    /// Token string → (token_id, score)
    piece_to_id: HashMap<String, (u32, f32)>,
    /// Whether to prepend ▁ to the input
    add_space_prefix: bool,
    /// UNK token ID
    unk_id: Option<u32>,
    /// Special tokens sorted by length (longest first) for greedy matching
    special_tokens: Vec<(String, u32)>,
}

impl SpmTokenizer {
    /// Number of entries in the vocabulary. See
    /// `SplitTokenizer::vocab_size` for why this is a network-cost figure.
    pub fn vocab_size(&self) -> usize {
        self.piece_to_id.len()
    }

    pub fn new(tokens: &[String], scores: &[f32], add_space_prefix: bool) -> Self {
        let mut piece_to_id = HashMap::new();
        for (i, (tok, &score)) in tokens.iter().zip(scores.iter()).enumerate() {
            piece_to_id.insert(tok.clone(), (i as u32, score));
        }
        let unk_id = tokens.iter().position(|t| t == "<unk>").map(|i| i as u32);
        // Trust the GGUF's declared id; only fall back to a vocab search when
        // the file does not declare one. The fallback covers BOTH families'
        // literals — `<s>` (Llama/Mistral/Phi) and `<bos>` (Gemma) — because
        // searching for one family's spelling silently disables BOS for the
        // other, which is exactly the bug this parameter exists to prevent.

        // Collect special tokens (control tokens like <bos>, <start_of_turn>, etc.)
        let mut special_tokens: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.starts_with('<') && t.ends_with('>') && !t.contains(' ') && t.len() <= 30
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        // Sort by length descending for greedy matching
        special_tokens.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));

        tracing::info!(
            vocab_size = tokens.len(),
            special_tokens = special_tokens.len(),
            add_space_prefix,
            unk_id = ?unk_id,
            "Built SPM tokenizer from GGUF vocab"
        );

        Self {
            piece_to_id,
            add_space_prefix,
            unk_id,
            special_tokens,
        }
    }

    /// Encode text to token IDs using SPM merge algorithm.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut result = Vec::new();

        // Split text around special tokens first
        let segments = self.split_special_tokens(text);
        for (segment, is_special) in segments {
            if is_special {
                if let Some(&(id, _)) = self.piece_to_id.get(&segment) {
                    result.push(id as i64);
                }
            } else {
                // Normalize: replace spaces with ▁, optionally prepend ▁
                let normalized = if self.add_space_prefix && result.is_empty() {
                    format!("\u{2581}{}", segment.replace(' ', "\u{2581}"))
                } else {
                    segment.replace(' ', "\u{2581}")
                };
                result.extend(self.spm_encode(&normalized));
            }
        }
        result
    }

    /// Split text around special tokens, returning (segment, is_special) pairs.
    fn split_special_tokens(&self, text: &str) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            // Find the earliest special token match
            let mut best_match: Option<(usize, &str, u32)> = None;
            for (tok, id) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok.as_str()) {
                    if best_match.is_none() || pos < best_match.unwrap().0 {
                        best_match = Some((pos, tok.as_str(), *id));
                    }
                }
            }
            match best_match {
                Some((pos, tok, _)) => {
                    if pos > 0 {
                        result.push((remaining[..pos].to_string(), false));
                    }
                    result.push((tok.to_string(), true));
                    remaining = &remaining[pos + tok.len()..];
                }
                None => {
                    result.push((remaining.to_string(), false));
                    break;
                }
            }
        }
        result
    }

    /// Core SPM merge algorithm (matches llama.cpp's llm_tokenizer_spm).
    ///
    /// 1. Initialize each UTF-8 character as a separate symbol
    /// 2. Build priority queue of all valid bigrams (adjacent pairs in vocab)
    /// 3. Pop highest-score bigram, merge symbols
    /// 4. Add new bigrams with neighbors
    /// 5. Repeat until no more merges
    fn spm_encode(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return vec![];
        }

        // Initialize: each character is a symbol
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        // Symbols: (start_byte, len_bytes, prev, next)
        // Use byte offsets for correct string slicing
        struct Symbol {
            start: usize,
            len: usize,
            prev: i32,
            next: i32,
        }

        let text_bytes = text.as_bytes();
        let mut symbols: Vec<Symbol> = Vec::with_capacity(n);
        let mut byte_pos = 0;
        for (i, ch) in chars.iter().enumerate() {
            let ch_len = ch.len_utf8();
            symbols.push(Symbol {
                start: byte_pos,
                len: ch_len,
                prev: if i > 0 { i as i32 - 1 } else { -1 },
                next: if i + 1 < n { i as i32 + 1 } else { -1 },
            });
            byte_pos += ch_len;
        }

        // Priority queue: (score, left_idx, right_idx, merged_token_id)
        // Use Reverse for max-heap (BinaryHeap is max by default, we want highest score first)
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct Merge {
            score: f32,
            left: usize,
            right: usize,
            token_id: u32,
            /// Combined byte length of `left` + `right` **at the moment this
            /// bigram was scored**. The queue is not invalidated when a symbol
            /// grows, so this is what proves the entry still describes the same
            /// text — see the check in the merge loop.
            size: usize,
        }
        impl Eq for Merge {}
        impl PartialOrd for Merge {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Merge {
            fn cmp(&self, other: &Self) -> Ordering {
                // Higher score first, break ties by position (lower left first)
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| other.left.cmp(&self.left))
            }
        }

        let mut heap: BinaryHeap<Merge> = BinaryHeap::new();

        let try_add_bigram = |heap: &mut BinaryHeap<Merge>,
                              symbols: &[Symbol],
                              left: usize,
                              right: usize,
                              text_bytes: &[u8]| {
            let merged = std::str::from_utf8(
                &text_bytes[symbols[left].start..symbols[right].start + symbols[right].len],
            )
            .unwrap_or("");
            if let Some(&(id, score)) = self.piece_to_id.get(merged) {
                heap.push(Merge {
                    score,
                    left,
                    right,
                    token_id: id,
                    size: symbols[left].len + symbols[right].len,
                });
            }
        };

        // Initialize bigrams
        for i in 0..n.saturating_sub(1) {
            try_add_bigram(&mut heap, &symbols, i, i + 1, text_bytes);
        }

        // Merge loop
        while let Some(merge) = heap.pop() {
            let left = merge.left;
            let right = merge.right;

            // Check if symbols are still valid (not already merged)
            if symbols[left].len == 0 || symbols[right].len == 0 {
                continue;
            }

            // Verify the merge is still valid (symbols are still adjacent)
            if symbols[left].next != right as i32 {
                continue;
            }

            // …and that neither side has GROWN since this bigram was scored.
            //
            // Merging extends `left` to cover `right`, so any queued bigram
            // naming `left` still names a live, adjacent symbol — but one whose
            // text is now longer than the piece we looked up. Applying it built
            // a symbol for text nobody had checked against the vocabulary, and
            // the final lookup then missed and dumped the whole span through
            // byte fallback. Measured on Phi-3.5's real vocabulary: `banana`
            // came out as `▁banan` in raw `<0xNN>` bytes plus a stray `a`, and
            // 5 of 16 ordinary English words failed the same way. The model
            // receives gibberish and says so, which reads as the model being
            // stupid rather than as a tokenizer bug.
            //
            // llama.cpp's `llm_tokenizer_spm` carries the same guard
            // (`left_sym.n + right_sym.n != bigram.size`); it is what makes the
            // lazy queue sound, not an optimisation.
            if symbols[left].len + symbols[right].len != merge.size {
                continue;
            }

            // Merge: extend left symbol to cover right
            symbols[left].len = (symbols[right].start + symbols[right].len) - symbols[left].start;
            symbols[right].len = 0; // mark as deleted

            // Update linked list
            let right_next = symbols[right].next;
            symbols[left].next = right_next;
            if right_next >= 0 {
                symbols[right_next as usize].prev = left as i32;
            }

            // Try new bigrams with neighbors
            if symbols[left].prev >= 0 {
                try_add_bigram(
                    &mut heap,
                    &symbols,
                    symbols[left].prev as usize,
                    left,
                    text_bytes,
                );
            }
            if symbols[left].next >= 0 {
                try_add_bigram(
                    &mut heap,
                    &symbols,
                    left,
                    symbols[left].next as usize,
                    text_bytes,
                );
            }
        }

        // Collect remaining symbols as token IDs
        let mut result = Vec::new();
        let mut idx = 0i32;
        // Find the first symbol
        while idx >= 0 && idx < symbols.len() as i32 {
            if symbols[idx as usize].len == 0 {
                idx += 1;
                continue;
            }
            break;
        }
        // Walk the linked list
        while idx >= 0 && (idx as usize) < symbols.len() {
            let sym = &symbols[idx as usize];
            if sym.len == 0 {
                idx = sym.next;
                continue;
            }
            let piece =
                std::str::from_utf8(&text_bytes[sym.start..sym.start + sym.len]).unwrap_or("");
            if let Some(&(id, _)) = self.piece_to_id.get(piece) {
                result.push(id as i64);
            } else {
                // Byte fallback: encode unknown characters as <0xNN> tokens
                for byte in text_bytes[sym.start..sym.start + sym.len].iter() {
                    let byte_tok = format!("<0x{:02X}>", byte);
                    if let Some(&(id, _)) = self.piece_to_id.get(&byte_tok) {
                        result.push(id as i64);
                    } else if let Some(unk) = self.unk_id {
                        result.push(unk as i64);
                    }
                }
            }
            idx = sym.next;
        }

        result
    }
}

impl SplitTokenizer {
    /// How many distinct tokens this model's vocabulary holds.
    ///
    /// Used to price what a speculative verify round drags back over the
    /// network — `spec_logits` is one f32 per vocabulary entry per position,
    /// so the vocabulary size IS the payload size, and on a 128k-vocab model
    /// that is ~513 KB for a single token
    /// (`ngram_only_spec::required_tokens_per_round_x100`).
    pub fn vocab_size(&self) -> usize {
        match &self.kind {
            TokenizerKind::Bpe(b) => b.vocab_size(),
            TokenizerKind::SentencePiece(s) => s.vocab_size(),
        }
    }

    /// Resolve the BOS id to use, preferring the GGUF's declared value.
    ///
    /// The fallback covers BOTH families' literals — `<s>` (Llama/Mistral/Phi)
    /// and `<bos>` (Gemma). Searching for only one family's spelling is how
    /// every Llama-family model silently lost its BOS token.
    fn resolve_bos(tokens: &[String], declared: Option<u32>) -> Option<u32> {
        declared.or_else(|| {
            tokens
                .iter()
                .position(|t| t == "<s>" || t == "<bos>")
                .map(|i| i as u32)
        })
    }

    /// Build from GGUF BPE merges (existing path).
    pub fn from_bpe(
        tokens: &[String],
        merges: &[String],
        pre_type: &str,
        model: &str,
        add_bos_token: bool,
        bos_id: Option<u32>,
    ) -> Self {
        Self {
            kind: TokenizerKind::Bpe(Box::new(BpeTokenizer::from_gguf(
                tokens, merges, pre_type, model,
            ))),
            bos_id: Self::resolve_bos(tokens, bos_id),
            add_bos_token,
        }
    }

    /// Build a SentencePiece tokenizer from GGUF vocab + scores.
    pub fn from_sentencepiece(
        tokens: &[String],
        scores: &[f32],
        add_space_prefix: bool,
        add_bos_token: bool,
        bos_id: Option<u32>,
    ) -> Self {
        Self {
            kind: TokenizerKind::SentencePiece(SpmTokenizer::new(tokens, scores, add_space_prefix)),
            bos_id: Self::resolve_bos(tokens, bos_id),
            add_bos_token,
        }
    }

    /// Encode text to token IDs.
    ///
    /// BOS is prepended HERE, at the single entry point every variant shares,
    /// rather than inside each variant's own `encode`. It previously lived in
    /// the SentencePiece variant alone, so every model that took the BPE path
    /// — which is any GGUF shipping merges, TinyLlama included — was prefilled
    /// with no BOS at position 0 and produced degenerate replies. Adding a
    /// third variant cannot reintroduce that gap.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut out = Vec::new();
        if self.add_bos_token {
            if let Some(bos) = self.bos_id {
                out.push(bos as i64);
            }
        }
        out.extend(match &self.kind {
            TokenizerKind::Bpe(bpe) => bpe.encode(text),
            TokenizerKind::SentencePiece(spm) => spm.encode(text),
        });
        out
    }

    /// Decode a single token string back to UTF-8 bytes.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        match &self.kind {
            TokenizerKind::Bpe(bpe) => bpe.decode_token(token_str),
            TokenizerKind::SentencePiece(_) => decode_token_impl(token_str, true, &HashMap::new()),
        }
    }

    /// Whether this tokenizer uses SentencePiece encoding.
    pub fn is_sentencepiece(&self) -> bool {
        match &self.kind {
            TokenizerKind::Bpe(bpe) => bpe.is_sentencepiece(),
            TokenizerKind::SentencePiece(_) => true,
        }
    }

    /// Return a reference to the byte decoder mapping (for BPE caching).
    pub fn byte_decoder(&self) -> HashMap<char, u8> {
        match &self.kind {
            TokenizerKind::Bpe(bpe) => bpe.byte_decoder().clone(),
            TokenizerKind::SentencePiece(_) => HashMap::new(),
        }
    }
}

/// Build the GPT-2 byte encoder mapping.
/// Maps each byte (0-255) to a unicode character such that:
/// - Printable bytes map to themselves (as unicode chars)
/// - Non-printable bytes map to U+0100, U+0101, etc.
fn build_gpt2_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    let mut offset = 0u32;

    for b in 0u16..=255 {
        let is_printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if is_printable {
            let ch = char::from_u32(b as u32).unwrap();
            encoder[b as usize] = ch;
            decoder.insert(ch, b as u8);
        } else {
            let ch = char::from_u32(256 + offset).unwrap();
            encoder[b as usize] = ch;
            decoder.insert(ch, b as u8);
            offset += 1;
        }
    }

    (encoder, decoder)
}

#[cfg(test)]
mod pre_tokenizer_tests {
    use super::*;

    /// A tiny GPT-2-style byte-level vocab carrying the space-prefixed word
    /// tokens a real Llama-3 vocab has. `Ġ` is byte 0x20 under the GPT-2 byte
    /// encoding, so `Ġworld` is the token for " world".
    /// Every single-byte token (so nothing ever falls back to the unknown id),
    /// plus the multi-character tokens these tests exercise.
    fn bpe_vocab() -> Vec<String> {
        let (encoder, _) = build_gpt2_byte_encoder();
        let mut v: Vec<String> = encoder.iter().map(|c| c.to_string()).collect();
        for extra in [
            "he", "hel", "hell", "hello", "wo", "wor", "worl", "world", "Ġw", "Ġwo", "Ġwor",
            "Ġworl", "Ġworld", "Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello", "ĠĠ", "12", "123",
        ] {
            v.push(extra.to_string());
        }
        v
    }

    /// Merge rank is list order, and a real merge file ranks the
    /// space-prefixed pairs above the bare ones — so `Ġ w` must outrank `w o`,
    /// or the space never gets absorbed regardless of the pre-tokenizer.
    fn merges() -> Vec<String> {
        [
            "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d", "Ġ h", "Ġh e", "Ġhe l", "Ġhel l",
            "Ġhell o", "Ġ Ġ", "h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d",
            "1 2", "12 3",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn encode_with(pre: &str, text: &str) -> Vec<String> {
        let vocab = bpe_vocab();
        let tok = BpeTokenizer::from_gguf(&vocab, &merges(), pre, "gpt2");
        tok.encode(text)
            .into_iter()
            .map(|id| vocab[id as usize].clone())
            .collect()
    }

    /// The defect this guards: `llama-bpe` — the pre-tokenizer name on every
    /// Llama-3/3.1/3.2 GGUF — was absent from the match and fell through to a
    /// naive whitespace split. That stranded each space as its own `Ġ` token
    /// instead of attaching it to the following word, roughly doubling the
    /// prompt and handing the model sequences it was never trained on.
    /// Measured before the fix: "The quick brown fox jumps over the lazy dog"
    /// took 19 tokens against the reference tokenizer's 9.
    #[test]
    fn llama3_attaches_a_leading_space_to_the_following_word() {
        assert_eq!(
            encode_with("llama-bpe", "hello world"),
            vec!["hello", "Ġworld"],
            "a space must merge into the word after it, not stand alone"
        );
    }

    /// Every alias llama.cpp routes to the Llama-3 pattern must behave the same
    /// — a node that recognises `llama-bpe` but not `llama3` still mis-tokenises.
    #[test]
    fn every_llama3_alias_maps_to_the_same_pattern() {
        for alias in [
            "llama3",
            "llama-v3",
            "llama-bpe",
            "falcon3",
            "falcon-h1",
            "pixtral",
            "dbrx",
            "smaug-bpe",
            "glm4",
        ] {
            assert_eq!(
                encode_with(alias, "hello world"),
                vec!["hello", "Ġworld"],
                "pre-tokenizer alias {alias} did not get the Llama-3 pattern"
            );
        }
    }

    /// An unrecognised name must still attach leading spaces. The old fallback
    /// was a whitespace split, so ANY model family not explicitly listed was
    /// silently mis-tokenised; the GPT-2 splitter is the sane general default.
    #[test]
    fn an_unknown_pre_tokenizer_falls_back_to_gpt2_not_a_whitespace_split() {
        assert!(pre_tokenizer_patterns("some-future-model").is_none());
        assert_eq!(
            encode_with("some-future-model", "hello world"),
            vec!["hello", "Ġworld"],
            "the fallback must not strand the space as its own token"
        );
    }

    /// Qwen2 splits numbers one digit at a time; Llama-3 groups up to three.
    /// The Qwen arm previously held the Llama-3 pattern, so Qwen models were
    /// given three-digit number tokens they were not trained on.
    #[test]
    fn qwen2_splits_digits_singly_and_llama3_groups_them() {
        assert_eq!(encode_with("qwen2", "123"), vec!["1", "2", "3"]);
        assert_eq!(encode_with("llama-bpe", "123"), vec!["123"]);
    }

    /// Pre-tokenizer patterns need not cover their whole input — the GPT-2 one
    /// does not match interior whitespace runs. Fragments between matches must
    /// survive; dropping them deleted characters from the prompt outright.
    #[test]
    fn text_between_matches_is_never_dropped() {
        let vocab = bpe_vocab();
        for pre in ["gpt-2", "llama-bpe", "qwen2", "default", "starcoder"] {
            let tok = BpeTokenizer::from_gguf(&vocab, &merges(), pre, "gpt2");
            for text in [
                "hello   world",
                "hello \t world",
                "  hello  ",
                "hello!!!  world",
            ] {
                let decoded: Vec<u8> = tok
                    .encode(text)
                    .iter()
                    .flat_map(|id| tok.decode_token(&vocab[*id as usize]))
                    .collect();
                assert_eq!(
                    String::from_utf8_lossy(&decoded),
                    text,
                    "pre-tokenizer {pre} lost characters from {text:?}"
                );
            }
        }
    }

    /// Every pattern in the table must compile under `fancy_regex`; a pattern
    /// that does not is dropped at build time and silently weakens splitting.
    #[test]
    fn every_pattern_in_the_table_compiles() {
        for pre in [
            "llama-bpe",
            "qwen2",
            "qwen35",
            "gpt-4o",
            "llama4",
            "gpt-2",
            "default",
            "falcon",
            "starcoder",
            "deepseek-coder",
            "whitespace",
        ] {
            let pats = pre_tokenizer_patterns(pre)
                .unwrap_or_else(|| panic!("{pre} missing from the table"));
            for p in pats {
                assert!(
                    fancy_regex::Regex::new(p).is_ok(),
                    "pattern for {pre} failed to compile: {p}"
                );
            }
        }
    }
}

#[cfg(test)]
mod bos_tests {
    use super::*;

    /// A minimal Llama-style SPM vocab: `<s>` is id 1, as in every
    /// Llama/Mistral/Phi GGUF.
    fn llama_vocab() -> (Vec<String>, Vec<f32>) {
        let toks: Vec<String> = ["<unk>", "<s>", "</s>", "\u{2581}Hi", "\u{2581}there"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let scores = vec![0.0; toks.len()];
        (toks, scores)
    }

    /// The GGUF declares `tokenizer.ggml.bos_token_id`; that is the
    /// authoritative answer and must be used verbatim.
    #[test]
    fn declared_bos_id_is_prepended_for_llama_vocab() {
        let (toks, scores) = llama_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, true, true, Some(1));
        let ids = tok.encode("Hi there");
        assert_eq!(
            ids.first(),
            Some(&1i64),
            "declared BOS id must be prepended, got {ids:?}"
        );
    }

    /// Regression: the id used to be derived by searching the vocab for
    /// Gemma's `<bos>` literal, so every `<s>` model resolved to `None` and
    /// was prefilled with no BOS at all. The fallback must cover both
    /// families' spellings.
    #[test]
    fn undeclared_bos_falls_back_across_both_families() {
        let (toks, scores) = llama_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, true, true, None);
        assert_eq!(
            tok.encode("Hi there").first(),
            Some(&1i64),
            "Llama `<s>` vocab must still resolve a BOS when the GGUF omits the id"
        );

        let gemma: Vec<String> = ["<pad>", "<bos>", "<eos>", "\u{2581}Hi"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gscores = vec![0.0; gemma.len()];
        let gtok = SplitTokenizer::from_sentencepiece(&gemma, &gscores, true, true, None);
        assert_eq!(
            gtok.encode("Hi").first(),
            Some(&1i64),
            "Gemma `<bos>` vocab must keep working"
        );
    }

    /// The BPE path must inherit BOS too. TinyLlama ships merges, so it takes
    /// this branch — BOS handling that lived only in the SentencePiece variant
    /// left every such model prefilled without one.
    #[test]
    fn bpe_path_also_prepends_bos() {
        let toks: Vec<String> = ["<unk>", "<s>", "</s>", "Hi"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let merges: Vec<String> = vec![];
        let tok = SplitTokenizer::from_bpe(&toks, &merges, "default", "llama", true, Some(1));
        assert_eq!(
            tok.encode("Hi").first(),
            Some(&1i64),
            "BPE-path models must get BOS at position 0"
        );
    }

    /// A GGUF that explicitly opts out must still be honoured.
    #[test]
    fn explicit_opt_out_is_respected() {
        let (toks, scores) = llama_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, false, false, Some(1));
        assert_ne!(
            tok.encode("Hi there").first(),
            Some(&1i64),
            "add_bos_token=false must not prepend"
        );
    }
}

#[cfg(test)]
mod spm_merge_tests {
    use super::*;

    /// Vocabulary engineered to force a *stale* bigram to reach the front of
    /// the merge queue.
    ///
    /// Over the input `abcd`: the initial bigrams are `ab` (score 1.0) and
    /// `bc` (score 5.0); `cd` is absent. `bc` wins and merges, which grows
    /// symbol 1 from `b` to `bc`. The queued `ab` entry still names symbols
    /// 0 and 1, and they are still live and still adjacent — but together
    /// they now spell `abc`, which is not in this vocabulary at all.
    fn stale_bigram_vocab() -> (Vec<String>, Vec<f32>) {
        let mut toks: Vec<String> = ["<unk>", "<s>", "</s>", "a", "b", "c", "d", "ab", "bc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut scores = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0];
        // Byte-fallback tokens, so a failed merge is observable as `<0xNN>`
        // rather than collapsing to <unk>.
        for b in 0u16..=255 {
            toks.push(format!("<0x{b:02X}>"));
            scores.push(-1000.0);
        }
        (toks, scores)
    }

    fn pieces(toks: &[String], ids: &[i64]) -> Vec<String> {
        ids.iter()
            .map(|&i| toks[i as usize].clone())
            .collect::<Vec<_>>()
    }

    /// A queued bigram must not be applied once either side has grown.
    ///
    /// Applying it built a symbol spanning text that was never checked against
    /// the vocabulary, so the final lookup missed and dumped the whole span
    /// through byte fallback. Against Phi-3.5's real vocabulary this corrupted
    /// **64.9% of a 4128-line corpus** of ordinary sentences: `banana` came out
    /// as `▁banan` in raw bytes plus a stray `a`. The model receives gibberish
    /// and says so, which reads as the model being stupid.
    ///
    /// llama.cpp's `llm_tokenizer_spm` guards this with
    /// `left_sym.n + right_sym.n != bigram.size`; the guard is what makes the
    /// lazy queue sound, not an optimisation.
    #[test]
    fn a_grown_symbol_invalidates_its_queued_bigram() {
        let (toks, scores) = stale_bigram_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, false, false, None);
        let ids = tok.encode("abcd");
        let got = pieces(&toks, &ids);

        assert_eq!(
            got,
            vec!["a", "bc", "d"],
            "expected the higher-scoring `bc` merge to stand and the stale `ab` \
             bigram to be discarded; got {got:?}"
        );
        assert!(
            !got.iter().any(|p| p.starts_with("<0x")),
            "no byte fallback should occur — every character is in the vocab: {got:?}"
        );
    }

    /// The guard must not block *legitimate* merges: `ab` alone still merges
    /// when nothing has grown underneath it.
    #[test]
    fn a_valid_bigram_still_merges() {
        let (toks, scores) = stale_bigram_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, false, false, None);
        // No `c` to trigger the higher-scoring `bc`, so `ab` is uncontested.
        let ids = tok.encode("abd");
        assert_eq!(pieces(&toks, &ids), vec!["ab", "d"]);
    }

    /// Single characters and empty input must survive the guard untouched.
    #[test]
    fn degenerate_inputs_are_unaffected() {
        let (toks, scores) = stale_bigram_vocab();
        let tok = SplitTokenizer::from_sentencepiece(&toks, &scores, false, false, None);
        assert!(tok.encode("").is_empty());
        assert_eq!(pieces(&toks, &tok.encode("a")), vec!["a"]);
        assert_eq!(pieces(&toks, &tok.encode("bc")), vec!["bc"]);
    }
}
