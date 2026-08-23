pub mod allreduce;
pub mod attn_kernel;
pub(crate) mod attn_softmax;
pub mod chat_template;
pub(crate) mod cpu_pools;
pub mod decode_attn;
pub mod dsd_controller;
pub mod executor;
pub mod fast_math;
pub mod hedging;
pub mod kv_cache;
pub(crate) mod layers;
pub mod local_embedder;
pub mod mem_bandwidth;
pub(crate) mod model_arch;
pub mod model_worker;
pub mod ngram_lookup;
pub mod pipeline;
pub mod prefetch;
pub mod prefill_pacer;
pub mod process_pool;
pub(crate) mod prof;
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
pub mod thermal;
pub(crate) mod tokenizer;
pub mod trace;
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
    // Snapshot enough of the incoming text to identify it if the steps below
    // consume all of it. Costs one small allocation per completion — this runs
    // once per reply, never per token — and nothing at all when the model
    // produced no text to begin with.
    let had_content = !text.trim().is_empty();
    let before: String = if had_content {
        text.chars().take(EMPTIED_REPLY_LOG_CHARS).collect()
    } else {
        String::new()
    };

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

    // The model generated something and finalisation removed all of it. The
    // caller will return `content: ""` with `finish_reason: "stop"` and HTTP
    // 200, which is indistinguishable from a real answer — so without this line
    // the only way to notice is for a human to see a blank reply. Every blank
    // reply diagnosed here so far (a missing system turn, a template from the
    // wrong model family, byte-fallback decoding) began that way.
    //
    // What was removed is the evidence: a leaked marker points at the template,
    // a stop sequence matching at position 0 points at the prompt. Logged at
    // WARN because a user is looking at an empty answer either way.
    if let Some(evidence) = emptied_reply_evidence(had_content, text, &before) {
        tracing::warn!(
            matched_stop = ?matched,
            generated = %evidence,
            "reply is empty after finalisation — the model generated text and all of it was \
             removed as control tokens or stop sequences; check the prompt and chat template \
             for this model before looking at sampling"
        );
    }

    matched
}

/// Longest reply still treated as "a stop sequence ate the answer".
///
/// Two tokens is not an answer to anything. Deliberately tiny: the point is to
/// catch a stop string matching the model's opening words, not to second-guess
/// a genuinely terse reply.
const STOP_ATE_THE_REPLY_TOKENS: u32 = 2;

/// Did the model end its own turn immediately, well inside the token budget?
///
/// No stop sequence matched, so nothing external cut the reply short — the model
/// emitted end-of-turn at once. That points at the prompt (a chat template from
/// the wrong family, a rendered prompt that does not end where the model expects
/// to answer) rather than at sampling.
///
/// `max_tokens` is what separates it from a caller who asked for one token and
/// got exactly one. Warning on that would fire on every deliberate
/// single-token request, which is how a useful warning becomes noise.
fn reply_ended_itself_early(
    completion_tokens: u32,
    max_tokens: u32,
    matched: Option<&str>,
) -> bool {
    matched.is_none()
        && completion_tokens <= STOP_ATE_THE_REPLY_TOKENS
        && completion_tokens < max_tokens
}

/// Did a stop sequence leave the caller with essentially no reply?
///
/// Split from the logging so the threshold can be tested without capturing log
/// output — the same reason `emptied_reply_evidence` is split out.
pub(crate) fn stop_sequence_ate_the_reply(completion_tokens: u32, matched: Option<&str>) -> bool {
    matched.is_some() && completion_tokens <= STOP_ATE_THE_REPLY_TOKENS
}

/// Say why a reply came back with (almost) nothing.
///
/// Two causes, and until this existed the node reported neither: a stop sequence
/// in the request matched immediately, or the model emitted end-of-turn at once.
/// Both surface to an OpenAI client identically — `finish_reason: "stop"`,
/// HTTP 200, a blank or one-word answer — and the distinction is exactly what
/// tells a user whether to look at their client's `stop` array or at the prompt.
///
/// **The OpenAI surface cannot say this in the response.** `finish_reason` is
/// `"stop"` whether the model chose to end its turn or a stop sequence cut it
/// off, and the schema has no field for which one fired. (The Anthropic surface
/// does carry `stop_sequence`, and does report it.) So a reply truncated to
/// nothing is indistinguishable, to the client, from a model with nothing to
/// say — and the node said nothing either.
///
/// That gap has a measured cost. An external report on 2026-08-23 described a
/// coding agent getting `completion_tokens: 1` from a large tool-using request
/// with no error, and a testing session went into narrowing it; plain `curl`
/// could not reproduce it because the attempts left the client's `stop` array
/// out. Reproduced locally afterwards: a stop string matching the model's
/// second token yields exactly `completion_tokens: 1, finish_reason: "stop"`,
/// silently. Whether that was the reported cause is unconfirmed; that the
/// product could not say so is not.
///
/// **Call this wherever a generation finishes with a `matched_stop_sequence`.**
/// It deliberately does NOT live in `finalize_reply_text`, which looks like the
/// choke point and is the wrong place: every generator applies the caller's stop
/// sequences DURING generation and stops before the matching text is kept, so by
/// the time finalisation runs the text no longer contains the stop and there is
/// nothing left to notice. That was tried, and it silently never fired —
/// caught by running a request rather than by reading the diff.
pub(crate) fn report_short_reply(
    request_id: &uuid::Uuid,
    completion_tokens: u32,
    max_tokens: u32,
    matched: Option<&str>,
) {
    if let Some(stop) = matched {
        if !stop_sequence_ate_the_reply(completion_tokens, matched) {
            tracing::debug!(
                %request_id,
                completion_tokens,
                stop_sequence = %stop.escape_debug(),
                "DIAG: reply ended on a caller-supplied stop sequence"
            );
            return;
        }
        tracing::warn!(
            %request_id,
            completion_tokens,
            stop_sequence = %stop.escape_debug(),
            "a stop sequence in the request matched almost immediately, so the reply came back \
             with next to nothing — the caller sees finish_reason \"stop\" and cannot tell that \
             from the model choosing to say nothing; check the `stop` values the client sends"
        );
        return;
    }

    if reply_ended_itself_early(completion_tokens, max_tokens, matched) {
        tracing::warn!(
            %request_id,
            completion_tokens,
            max_tokens,
            "the model ended its turn immediately, well inside the token budget, and no stop \
             sequence matched — it emitted end-of-turn straight away. That is usually the prompt \
             rather than sampling: check the chat template for this model (grep `chat template \
             failed`) and whether the rendered prompt ends where the model expects to answer"
        );
    }
}

/// Whether an emptied reply is worth reporting, and the evidence to report with
/// it — `None` means stay silent.
///
/// **Extracted so the decision can be tested without capturing logs.** The
/// tests for this used to install a global `tracing` subscriber and assert on
/// what it captured, and that arrangement failed intermittently on CI three
/// times across two attempted fixes, each time reporting `got: ` with nothing
/// after it on a commit that had touched none of it. The cause was never
/// established: on the third occurrence the two documented mechanisms were both
/// falsified by direct experiment (a callsite hit before the subscriber exists
/// is NOT left permanently silenced, and neither racing the install from eight
/// threads nor 3,600 concurrent captures against a faithful copy of the helper
/// reproduced a single empty capture).
///
/// So the dependence is removed rather than the race chased further. What is
/// pinned here is what the tests were really for: that it fires only when
/// generated text was wholly removed, and that the evidence names what went —
/// a leaked marker points at the template, a stop matching at position 0 points
/// at the prompt. That the adjacent `warn!` is reached is a single line in plain
/// view, which is the trade being made, and it follows the pattern this codebase
/// already uses for `inbound_warning_decision`: pin the decision, not the log.
pub(crate) fn emptied_reply_evidence(
    had_content: bool,
    finalised: &str,
    before: &str,
) -> Option<String> {
    (had_content && finalised.trim().is_empty()).then(|| before.escape_debug().to_string())
}

/// How much of an emptied reply to put in the log. Enough to recognise a leaked
/// template or a stray marker, short enough not to dump a whole generation.
const EMPTIED_REPLY_LOG_CHARS: usize = 240;

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
    use super::{
        emptied_reply_evidence, finalize_reply_text, reply_ended_itself_early,
        stop_sequence_ate_the_reply, EMPTIED_REPLY_LOG_CHARS, STOP_ATE_THE_REPLY_TOKENS,
    };

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

    /// The condition the "reply is empty after finalisation" warning keys on:
    /// the model DID generate text and every character of it was removed. The
    /// caller then returns HTTP 200 with empty content and
    /// `finish_reason: "stop"`, which no client can tell from a real answer —
    /// so this case has to be recognisable in the log.
    #[test]
    fn a_reply_of_nothing_but_a_marker_finalises_to_empty() {
        let mut t = "<|im_end|>".to_string();
        finalize_reply_text(&mut t, &stops());
        assert!(t.trim().is_empty(), "expected the marker to leave nothing");

        // ...and a stop matching at position 0 does the same thing by a
        // different route, which is the prompt-fault shape rather than the
        // template-fault one.
        let mut t = "\nUser: who asked?".to_string();
        finalize_reply_text(&mut t, &["\nUser:".to_string()]);
        assert!(t.trim().is_empty());
    }
    /// The report has to name what was removed — that text is the entire
    /// diagnostic value, since it distinguishes a template fault (a leaked
    /// marker) from a prompt fault (a stop matching at position 0).
    #[test]
    fn a_stop_that_ate_the_reply_is_reported() {
        // The reported shape: one token, then a stop sequence ended it.
        assert!(stop_sequence_ate_the_reply(1, Some(" tests")));
        assert!(stop_sequence_ate_the_reply(0, Some("Unit")));
        assert!(stop_sequence_ate_the_reply(
            STOP_ATE_THE_REPLY_TOKENS,
            Some("x")
        ));
    }

    #[test]
    fn a_real_answer_cut_by_a_stop_is_not_reported() {
        // A stop trimming the tail of a genuine answer is the feature working.
        // Warning about it would train operators to ignore the warning, which is
        // how the case that matters gets missed.
        assert!(!stop_sequence_ate_the_reply(
            STOP_ATE_THE_REPLY_TOKENS + 1,
            Some("\nUser:")
        ));
        assert!(!stop_sequence_ate_the_reply(80, Some("\nUser:")));
    }

    #[test]
    fn a_caller_who_asked_for_one_token_is_not_a_fault() {
        // `max_tokens: 1` legitimately yields one token. Warning there would
        // fire on every deliberate single-token request — the classic way a
        // useful warning becomes noise and stops being read.
        assert!(!reply_ended_itself_early(1, 1, None));
        assert!(!reply_ended_itself_early(0, 0, None));
        assert!(!reply_ended_itself_early(2, 2, None));
    }

    #[test]
    fn a_model_stopping_well_inside_its_budget_is_reported() {
        // The reported shape: one token out of a hundred asked for, nothing cut
        // it off, so the model emitted end-of-turn straight away.
        assert!(reply_ended_itself_early(1, 100, None));
        assert!(reply_ended_itself_early(0, 100, None));
    }

    #[test]
    fn a_stop_match_is_the_other_reports_business() {
        // When a stop sequence matched, that is the explanation and it is
        // reported separately — this branch must stay quiet so one event never
        // produces two different warnings.
        assert!(!reply_ended_itself_early(1, 100, Some(" tests")));
    }

    #[test]
    fn a_full_length_reply_is_not_reported() {
        assert!(!reply_ended_itself_early(100, 100, None));
        assert!(!reply_ended_itself_early(64, 100, None));
    }

    #[test]
    fn a_short_reply_that_nothing_cut_off_is_not_reported() {
        // Terse but voluntary. `max_tokens: 1` is a legitimate request, and a
        // model answering "Yes." is not a fault to report.
        assert!(!stop_sequence_ate_the_reply(1, None));
        assert!(!stop_sequence_ate_the_reply(0, None));
    }

    #[test]
    fn an_emptied_reply_is_reported_with_what_was_removed() {
        let evidence = emptied_reply_evidence(true, "", "<|im_end|>")
            .expect("generated text that was wholly removed must be reported");
        assert!(
            evidence.contains("im_end"),
            "the report must name what was removed, got: {evidence}"
        );
    }

    /// The counterpart, and the reason the report is conditional: an ordinary
    /// reply must stay silent. A diagnostic that fires on healthy traffic is
    /// noise, and this runs on every completion.
    #[test]
    fn an_ordinary_reply_reports_nothing() {
        assert_eq!(
            emptied_reply_evidence(
                true,
                "Rivers carve landscapes.",
                "Rivers carve landscapes.<|im_end|>"
            ),
            None,
            "a normal reply must not be reported"
        );
        // A model that generated nothing at all is a different condition and is
        // already visible to the caller as zero tokens — it must not report here.
        assert_eq!(
            emptied_reply_evidence(false, "", ""),
            None,
            "no generation must not be reported"
        );
    }

    /// The whole path, not just the decision: a reply that is nothing but a
    /// marker must finalise to empty AND be reportable, which is what ties the
    /// pure helper above to what `finalize_reply_text` actually does.
    #[test]
    fn the_finalised_reply_and_the_report_agree() {
        let mut t = "<|im_end|>".to_string();
        let before: String = t.chars().take(EMPTIED_REPLY_LOG_CHARS).collect();
        let had_content = !t.trim().is_empty();
        finalize_reply_text(&mut t, &stops());
        assert!(t.is_empty(), "the marker-only reply must finalise to empty");
        assert!(
            emptied_reply_evidence(had_content, &t, &before).is_some(),
            "and that must be a reportable condition"
        );
    }

    /// The snapshot taken for that warning is built with `chars().take(..)`, so
    /// it must not split a multi-byte character. A panic here would abort a
    /// perfectly good reply on any non-ASCII generation.
    #[test]
    fn finalising_multibyte_text_does_not_panic() {
        let mut t = "日本語のテキストです。".repeat(40);
        finalize_reply_text(&mut t, &stops());
        assert!(t.starts_with("日本語"), "content must survive untouched");

        let mut emoji = "🙂🙃".repeat(300);
        emoji.push_str("<|im_end|>");
        finalize_reply_text(&mut emoji, &stops());
        assert!(emoji.starts_with("🙂"));
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

/// The vendored multi-row quantized matmul kernels (`GgmlType::vec_dot_rows`,
/// Q4_K four rows per pass, Q6_K four) promise bit-identical results to the
/// per-row upstream ordering — each row keeps its own accumulators and sees
/// the single-row kernel's operations in the same order. `examples/qmatmul_bench`
/// asserts the same thing, but an example is not run by CI; this is.
#[cfg(test)]
mod qmatmul_exactness_tests {
    use candle_core::quantized::k_quants::{matmul, BlockQ4K, BlockQ6K, GgmlType};

    /// The upstream algorithm: batch row outer, one `vec_dot` per element.
    fn reference<T: GgmlType>(
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        rhs_t: &[T],
    ) -> Vec<f32> {
        let kb = k / T::BLCK_SIZE;
        let mut lhs_b = vec![T::VecDotType::zeros(); m * kb];
        for r in 0..m {
            T::VecDotType::from_float(&lhs[r * k..(r + 1) * k], &mut lhs_b[r * kb..(r + 1) * kb]);
        }
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            let row = &lhs_b[r * kb..(r + 1) * kb];
            for c in 0..n {
                out[r * n + c] = T::vec_dot(k, &rhs_t[c * kb..(c + 1) * kb], row);
            }
        }
        out
    }

    fn check<T: GgmlType>(label: &str) {
        let (k, n) = (512usize, 96usize);
        let kb = k / T::BLCK_SIZE;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
        };
        let src: Vec<f32> = (0..n * k).map(|_| noise()).collect();
        let mut rhs = vec![T::zeros(); n * kb];
        T::from_float(&src, &mut rhs);
        // Row counts that exercise the multi-row passes, their tails, the m=1
        // decode path and a whole row block.
        for m in [1usize, 2, 3, 4, 5, 7, 8, 9, 13, 130] {
            let lhs: Vec<f32> = (0..m * k).map(|_| noise()).collect();
            let mut got = vec![0f32; m * n];
            matmul((m, k, n), &lhs, &rhs, &mut got).unwrap();
            let want = reference((m, k, n), &lhs, &rhs);
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad,
                0,
                "{label} m={m}: {bad}/{} elements differ from the per-row ordering",
                m * n
            );
        }
    }

    #[test]
    fn multi_row_kernels_are_bit_identical_to_the_per_row_ordering() {
        check::<BlockQ4K>("Q4_K");
        check::<BlockQ6K>("Q6_K");
    }
}
