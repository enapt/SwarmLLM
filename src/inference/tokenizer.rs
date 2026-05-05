//! BPE and SentencePiece tokenizers from GGUF metadata.

use std::collections::HashMap;

// ── BPE Tokenizer from GGUF merges ──

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
    /// Pre-tokenization regex pattern
    pre_tok_re: fancy_regex::Regex,
    /// Special tokens sorted by length descending (for matching)
    special_tokens: Vec<(String, u32)>,
    /// Whether this is a SentencePiece tokenizer (uses ▁ for spaces, no byte encoding)
    is_sentencepiece: bool,
}

impl BpeTokenizer {
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

        // Pre-tokenization regex based on model type
        let pattern = match pre_type {
            "qwen2" => {
                // Qwen2 pre-tokenization pattern (from HuggingFace tokenizers)
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "gpt-2" | "gpt2" => {
                r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            }
            _ => {
                // Default fallback: split on whitespace boundaries
                r"[^\s]+|\s+"
            }
        };
        let pre_tok_re = fancy_regex::Regex::new(pattern)
            .unwrap_or_else(|_| fancy_regex::Regex::new(r"[^\s]+|\s+").unwrap());

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
            pre_tok_re,
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

    /// Pre-tokenize text using the model's regex pattern.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut search_start = 0;
        while search_start < text.len() {
            match self.pre_tok_re.find_from_pos(text, search_start) {
                Ok(Some(m)) => {
                    pieces.push(m.as_str().to_string());
                    search_start = m.end();
                }
                _ => break,
            }
        }
        pieces
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
pub enum SplitTokenizer {
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
    /// BOS token ID
    bos_id: Option<u32>,
    /// Whether to auto-prepend BOS token (from GGUF tokenizer.ggml.add_bos_token)
    add_bos_token: bool,
    /// UNK token ID
    unk_id: Option<u32>,
    /// Special tokens sorted by length (longest first) for greedy matching
    special_tokens: Vec<(String, u32)>,
}

impl SpmTokenizer {
    pub fn new(
        tokens: &[String],
        scores: &[f32],
        add_space_prefix: bool,
        add_bos_token: bool,
    ) -> Self {
        let mut piece_to_id = HashMap::new();
        for (i, (tok, &score)) in tokens.iter().zip(scores.iter()).enumerate() {
            piece_to_id.insert(tok.clone(), (i as u32, score));
        }
        let unk_id = tokens.iter().position(|t| t == "<unk>").map(|i| i as u32);
        let bos_id = tokens.iter().position(|t| t == "<bos>").map(|i| i as u32);

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
            add_bos_token,
            unk_id = ?unk_id,
            bos_id = ?bos_id,
            "Built SPM tokenizer from GGUF vocab"
        );

        Self {
            piece_to_id,
            add_space_prefix,
            bos_id,
            add_bos_token,
            unk_id,
            special_tokens,
        }
    }

    /// Encode text to token IDs using SPM merge algorithm.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut result = Vec::new();

        // Auto-prepend BOS token if configured (Gemma-2 uses this)
        if self.add_bos_token {
            if let Some(bos) = self.bos_id {
                result.push(bos as i64);
            }
        }

        // Split text around special tokens first
        let segments = self.split_special_tokens(text);
        for (segment, is_special) in segments {
            if is_special {
                if let Some(&(id, _)) = self.piece_to_id.get(&segment) {
                    result.push(id as i64);
                }
            } else {
                // Normalize: replace spaces with ▁, optionally prepend ▁
                let normalized = if self.add_space_prefix
                    && result.len() <= (if self.add_bos_token { 1 } else { 0 })
                {
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
    /// Build from GGUF BPE merges (existing path).
    pub fn from_bpe(tokens: &[String], merges: &[String], pre_type: &str, model: &str) -> Self {
        Self::Bpe(Box::new(BpeTokenizer::from_gguf(
            tokens, merges, pre_type, model,
        )))
    }

    /// Build a SentencePiece tokenizer from GGUF vocab + scores.
    pub fn from_sentencepiece(
        tokens: &[String],
        scores: &[f32],
        add_space_prefix: bool,
        add_bos_token: bool,
    ) -> Self {
        Self::SentencePiece(SpmTokenizer::new(
            tokens,
            scores,
            add_space_prefix,
            add_bos_token,
        ))
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        match self {
            Self::Bpe(bpe) => bpe.encode(text),
            Self::SentencePiece(spm) => spm.encode(text),
        }
    }

    /// Decode a single token string back to UTF-8 bytes.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        match self {
            Self::Bpe(bpe) => bpe.decode_token(token_str),
            Self::SentencePiece(_) => decode_token_impl(token_str, true, &HashMap::new()),
        }
    }

    /// Whether this tokenizer uses SentencePiece encoding.
    pub fn is_sentencepiece(&self) -> bool {
        match self {
            Self::Bpe(bpe) => bpe.is_sentencepiece(),
            Self::SentencePiece(_) => true,
        }
    }

    /// Return a reference to the byte decoder mapping (for BPE caching).
    pub fn byte_decoder(&self) -> HashMap<char, u8> {
        match self {
            Self::Bpe(bpe) => bpe.byte_decoder().clone(),
            Self::SentencePiece(_) => HashMap::new(),
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
