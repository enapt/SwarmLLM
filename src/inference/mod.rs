pub mod allreduce;
pub mod attn_kernel;
pub mod chat_template;
pub mod dsd_controller;
pub mod executor;
pub mod hedging;
pub mod kv_cache;
pub(crate) mod layers;
pub mod local_embedder;
pub(crate) mod model_arch;
pub mod model_worker;
pub mod ngram_lookup;
pub mod pipeline;
pub mod prefetch;
pub mod process_pool;
pub mod quant;
pub mod router;
pub mod sampling;
pub mod scheduler;
pub(crate) mod shard_layout;
pub mod slot_table;
pub mod speculative;
pub mod split;
pub mod swift;
pub(crate) mod tensor_util;
pub(crate) mod tokenizer;
pub mod vision;
pub mod worker_ipc;

/// Strip a trailing partial stop-string suffix from `text` in place.
///
/// Token-by-token stop-string checking only catches complete matches, so a
/// partial stop string at the very end of generation can leak into the output
/// (e.g. "<|user" when the stop is "<|user|>"). This trims at most one such
/// prefix — once a trim happens we return immediately so later stops can't
/// cascade across the already-truncated text.
pub(crate) fn trim_trailing_partial_stop(text: &mut String, stops: &[String]) {
    for stop in stops {
        for end_len in (1..stop.len()).rev() {
            let prefix = &stop[..end_len];
            if text.ends_with(prefix) {
                text.truncate(text.len() - end_len);
                return;
            }
        }
    }
}

/// Control-token names that appear in the chat templates of the model families
/// we support. Used by [`strip_control_token_artifacts`].
///
/// Deliberately a NAME list rather than a list of full markers: the whole point
/// is to catch mangled spellings, so `<|im_end|>`, `<|im_end|`, `<|im_end>|`
/// and `<|im_end` all reduce to the same name.
const CONTROL_TOKEN_NAMES: &[&str] = &[
    "im_end",
    "im_start",
    "eot_id",
    "eom_id",
    "start_header_id",
    "end_header_id",
    "endoftext",
    "python_tag",
    "end_of_turn",
    "start_of_turn",
];

/// Put generated text into its final, user-facing form.
///
/// **Every source of reply text must end by calling this, and must not perform
/// these steps itself.** There are three sources — the in-process
/// [`executor`](crate::inference::executor), the worker subprocess
/// ([`process_pool`](crate::inference::process_pool)), and the reply assembled
/// from remote segments ([`pipeline::distributed`]) — and a control token
/// reached users across four releases because each was fixed separately.
///
/// The steps are ordered, and the order is the part that keeps being got
/// wrong, so it lives here rather than in three comment blocks:
///
/// 1. **Scrub control-token artifacts.** Before anything truncates, because
///    some models emit a stray end-of-turn marker *before* their answer
///    (`<|im_end|>hello`) — truncating at that match first would discard the
///    answer, turning a visible leak into an empty reply.
/// 2. **Truncate at the first genuine stop sequence.** Only caller-supplied
///    stops, on what survived step 1.
/// 3. **Trim a trailing partial stop.** A marker decodes across several tokens
///    and the leading pieces match nothing, so they are already emitted.
/// 4. **Drop newlines stranded at the front** by a marker removed in step 1.
///
/// Before this existed the three sources ran different subsets in different
/// orders: the distributed path trimmed before scrubbing and never did step 4,
/// so it could still return an answer-less reply after the scrub was
/// "everywhere". Pass an empty `stops` slice when the caller has none — steps
/// 2 and 3 then do nothing and the rest still applies.
///
/// Returns the stop sequence that truncated the text, if any, so a caller can
/// set `finish_reason` from it. It is safe to call more than once: every step
/// is idempotent, which lets a caller run it again with a different stop set
/// (the router applies template-derived stops that the executor never sees).
pub(crate) fn finalize_reply_text(text: &mut String, stops: &[String]) -> Option<String> {
    strip_control_token_artifacts(text);
    let mut matched = None;
    for stop in stops {
        if let Some(pos) = text.find(stop.as_str()) {
            text.truncate(pos);
            matched = Some(stop.clone());
            break;
        }
    }
    trim_trailing_partial_stop(text, stops);
    let start = text.len() - text.trim_start_matches(['\n', '\r']).len();
    if start > 0 {
        text.drain(..start);
    }
    matched
}

/// Remove control-token artifacts from generated text, in whatever spelling
/// they arrive in.
///
/// Stop-string matching handles the well-formed case, and
/// [`trim_trailing_partial_stop`] handles a marker split across tokens. Neither
/// helps when a model file's template disagrees with the weights it was
/// quantised from: such a model emits another family's markers, and decoding
/// them produces genuinely malformed spellings. Observed live 2026-07-25 on a
/// Llama-3.2 Q8_0 whose declared EOS is `128009` (`<|eot_id|>`) but which
/// emitted `<|im_end|>`, `<|im_end|` and `<|im_end>|` — the last with `>` and
/// `|` transposed, so it is not a prefix of anything and can appear mid-string.
///
/// Only spans naming a KNOWN control token are removed, so a user's own
/// `<|something_custom|>` survives. That is the line between sanitising our own
/// leakage and editing someone's content — a reply that genuinely discusses
/// `<|im_end|>` will lose it, which is the accepted cost of not showing control
/// tokens to every user of a mismatched model file.
pub(crate) fn strip_control_token_artifacts(text: &mut String) {
    if !text.contains("<|") {
        return;
    }
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            // Consume the name: word characters after the opening `<|`.
            let name_start = i + 2;
            let mut j = name_start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            let name = &text[name_start..j];
            if CONTROL_TOKEN_NAMES.contains(&name) {
                // Consume the trailing terminator run in any order and length:
                // `|>`, `>|`, `|`, `>`, `|>>`, or nothing at all (truncated).
                // Not capped at two characters — a mangled marker can carry a
                // third, and capping left a stray `>` in the middle of an
                // otherwise clean sentence ("The ocean> is a vast body...").
                // A legitimate `>` is separated from the marker by other text,
                // so it stops the run naturally.
                let mut k = j;
                while k < bytes.len() && (bytes[k] == b'|' || bytes[k] == b'>') {
                    k += 1;
                }
                i = k;
                continue;
            }
        }
        // Not an artifact — copy this byte through. Safe because we only ever
        // skip whole ASCII spans, so we never split a multi-byte character.
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(&text[i..(i + ch_len).min(text.len())]);
        i += ch_len;
    }
    *text = out;
}

/// Length in bytes of the UTF-8 character starting with `b`.
fn utf8_char_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod stop_trim_tests {
    use super::trim_trailing_partial_stop;

    /// A marker decoded across several tokens leaks its earlier pieces: the
    /// worker withholds only the token that COMPLETES the match, so the
    /// consumer is left holding e.g. `<|im_end|` when the stop is `<|im_end|>`.
    /// Reported three releases running against a Llama-3.2 GGUF that emits
    /// ChatML markers (external report 2026-07-25).
    #[test]
    fn trims_a_partial_marker_left_by_multi_token_decoding() {
        let stops = vec!["<|im_end|>".to_string()];
        for partial in ["<|im_end|", "<|im_end", "<|im_", "<|"] {
            let mut text = format!("Hello{partial}");
            trim_trailing_partial_stop(&mut text, &stops);
            assert_eq!(text, "Hello", "failed to trim {partial:?}");
        }
    }

    /// A bare partial with no preceding content must trim to empty rather than
    /// returning the marker fragment as the whole reply — the exact symptom
    /// reported (`content: "<|im_end|"`).
    #[test]
    fn a_reply_that_is_only_a_partial_marker_becomes_empty() {
        let stops = vec!["<|im_end|>".to_string()];
        let mut text = "<|im_end|".to_string();
        trim_trailing_partial_stop(&mut text, &stops);
        assert_eq!(text, "");
    }

    /// Ordinary text ending in characters that merely START a stop marker must
    /// not be over-trimmed beyond the marker prefix itself.
    #[test]
    fn does_not_touch_text_without_a_marker_prefix() {
        let stops = vec!["<|im_end|>".to_string()];
        for keep in ["Hello there", "2 < 3", "a|b", ""] {
            let mut text = keep.to_string();
            trim_trailing_partial_stop(&mut text, &stops);
            assert_eq!(text, keep, "over-trimmed {keep:?}");
        }
    }

    /// No stop strings configured must be a no-op.
    #[test]
    fn empty_stop_list_is_a_no_op() {
        let mut text = "<|im_end|".to_string();
        trim_trailing_partial_stop(&mut text, &[]);
        assert_eq!(text, "<|im_end|");
    }
}

#[cfg(test)]
mod control_token_strip_tests {
    use super::strip_control_token_artifacts;

    /// The reported spellings, including the transposed one that no
    /// prefix-based trim can catch because `<|im_end>` differs from
    /// `<|im_end|` at position 8.
    #[test]
    fn strips_every_reported_spelling() {
        for (input, want) in [
            ("<|im_end|>", ""),
            ("<|im_end|", ""),
            ("<|im_end>|", ""),
            ("<|im_end", ""),
            ("<|im_end>|Hello", "Hello"),
            ("<|im_end|>hello\n<|im_start|", "hello\n"),
            ("Hello<|eot_id|>", "Hello"),
            ("<|python_tag|>{\"a\":1}", "{\"a\":1}"),
            // Observed live: a mangled marker carrying a third terminator
            // char left a stray `>` mid-sentence.
            ("The ocean<|im_end|>> is vast", "The ocean is vast"),
            ("<|im_end|>>>x", "x"),
        ] {
            let mut got = input.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, want, "input {input:?}");
        }
    }

    /// Ordinary content must survive untouched — including text that merely
    /// contains `<` or `|`, and a user's own angle-pipe construct that does not
    /// name a known control token.
    #[test]
    fn leaves_ordinary_content_alone() {
        for keep in [
            "Hello there",
            "2 < 3 and 4 | 5",
            "if (a<|b) {}",
            "<|my_custom_thing|>",
            "a <|not_a_marker|> b",
            "",
        ] {
            let mut got = keep.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, keep, "over-stripped {keep:?}");
        }
    }

    /// Multi-byte characters must not be split when scanning byte-wise.
    #[test]
    fn preserves_multibyte_text() {
        for keep in ["héllo wörld", "日本語のテキスト", "emoji 🎉 here"] {
            let mut got = keep.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, keep);
        }
        let mut mixed = "日本<|im_end|>語".to_string();
        strip_control_token_artifacts(&mut mixed);
        assert_eq!(mixed, "日本語");
    }
}

#[cfg(test)]
mod finalize_reply_text_tests {
    use super::finalize_reply_text;

    fn stops() -> Vec<String> {
        vec!["<|im_end|>".to_string()]
    }

    /// The ordering rule, asserted once. A marker emitted BEFORE the answer
    /// must be removed without taking the answer with it — scrubbing has to
    /// happen before truncation. Getting this backwards turned a visible leak
    /// into an empty reply, and the distributed path was still doing it in
    /// that order after the scrub was added "everywhere".
    #[test]
    fn marker_before_the_answer_keeps_the_answer() {
        let mut t = "<|im_end|>Red, blue and yellow.".to_string();
        let matched = finalize_reply_text(&mut t, &stops());
        assert_eq!(t, "Red, blue and yellow.");
        assert_eq!(matched, None, "a scrubbed artifact is not a stop match");
    }

    /// A genuine stop still ends the reply, and is reported so the caller can
    /// set finish_reason.
    #[test]
    fn genuine_stop_truncates_and_is_reported() {
        let mut t = "Answer.<|im_end|>trailing".to_string();
        // Not a scrubbable artifact case: prove truncation via a stop the
        // scrubber does not know about.
        let s = vec!["\nUser:".to_string()];
        let mut t2 = "Answer.\nUser: next question".to_string();
        assert_eq!(
            finalize_reply_text(&mut t2, &s),
            Some("\nUser:".to_string())
        );
        assert_eq!(t2, "Answer.");
        // The control-token form is scrubbed rather than truncated.
        finalize_reply_text(&mut t, &stops());
        assert_eq!(t, "Answer.trailing");
    }

    /// Newlines stranded by a removed marker are dropped — a step the
    /// distributed path never had.
    #[test]
    fn strands_no_leading_newlines() {
        let mut t = "<|im_end|>\n\nHello".to_string();
        finalize_reply_text(&mut t, &stops());
        assert_eq!(t, "Hello");
    }

    /// Safe to run twice with different stop sets — the router applies
    /// template-derived stops after the executor has applied the caller's.
    #[test]
    fn is_idempotent_across_repeated_calls() {
        let mut once = "<|im_end|>\nHi there".to_string();
        finalize_reply_text(&mut once, &[]);
        let mut twice = once.clone();
        finalize_reply_text(&mut twice, &stops());
        assert_eq!(once, twice);
        assert_eq!(twice, "Hi there");
    }

    /// No stops configured still scrubs and cleans up.
    #[test]
    fn empty_stops_still_scrubs() {
        let mut t = "<|eot_id|>\nAnswer".to_string();
        assert_eq!(finalize_reply_text(&mut t, &[]), None);
        assert_eq!(t, "Answer");
    }
}
