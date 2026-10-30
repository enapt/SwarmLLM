//! A reply held to the JSON its caller asked for (#336) — OpenAI's
//! `response_format` (`json_object`, `json_schema`) and the Responses API's
//! `text.format`.
//!
//! Before every draw the tokens that cannot continue the reply as JSON the
//! schema accepts are set to −∞, so the reply is that JSON by construction,
//! not by asking nicely. The engine is llguidance (MIT), the one behind vLLM's
//! `guidance` backend, SGLang's `llguidance` grammar backend and llama.cpp's
//! `%llguidance` grammars: an Earley parser over a lexer of derivative-based
//! regexes, which computes a 128k-token mask in tens of microseconds and needs
//! no per-schema precompilation step.
//!
//! Three parts, one for each place a decision is made:
//! - [`settle_json_object`] / [`settle_json_schema`], at the API door: the
//!   caller's format → the exact schema a worker compiles, or a 400. It is
//!   compiled there, against a one-byte-per-token vocabulary, so a schema the
//!   engine refuses is refused before the reply starts — a compile error is a
//!   property of the grammar, never of the vocabulary.
//! - [`GrammarVocab`], once per loaded model (`SplitModel::grammar_vocab`):
//!   each vocabulary entry as the bytes a reply would show for it, control
//!   tokens marked as never text.
//! - [`HeldReply`], once per request: asked before every draw
//!   ([`HeldReply::mask`]), told every token drawn ([`HeldReply::took`]).
//!
//! Where it is held (this first step): every draw a worker on THIS node makes
//! for a reply it runs whole — the API's fast path and the router's local
//! segment, streamed or not, batched or not (`model_worker`). A split, a whole
//! model handed to a peer and the in-process executor still only ASK for JSON
//! (`ChatCompletionRequest::to_internal_messages`), as everything did before;
//! FUTURE_WORK #336 carries what remains.

use std::sync::{Arc, LazyLock};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::{json, Value};

use crate::error::SwarmError;
pub use crate::types::ReplyGrammar;

/// The layout a held reply is written in: one line, `", "` between items and
/// `": "` after a key — the layout Python's `json.dumps` writes.
///
/// Not "any whitespace": a grammar that lets whitespace through lets a model
/// write newlines inside a reply that is still a valid prefix, for ever — the
/// whitespace runaway that made xgrammar add `any_whitespace=False` and vLLM
/// `disable_any_whitespace`. A JSON reply means the same in either layout.
/// `coerce_one_of`: a `oneOf` is held as an `anyOf` — no token-by-token
/// grammar can rule out a reply matching two branches, and refusing every
/// schema that uses one would refuse what the caller's own validator accepts
/// for nearly every reply.
fn compile_options(lenient: bool) -> Value {
    json!({
        "whitespace_flexible": false,
        "item_separator": ", ",
        "key_separator": ": ",
        "coerce_one_of": true,
        "lenient": lenient,
    })
}

/// A schema with OUR compiler options written in, replacing any the caller
/// sent: `x-guidance` replaces the options wholesale, and a caller's layout or
/// escape rules are not theirs to change on this node.
fn with_options(schema: &Value, lenient: bool) -> Value {
    let mut schema = schema.clone();
    if let Some(members) = schema.as_object_mut() {
        members.insert("x-guidance".into(), compile_options(lenient));
    }
    schema
}

/// `json_object`: the reply is one JSON object.
pub fn settle_json_object() -> Arc<ReplyGrammar> {
    Arc::new(ReplyGrammar {
        schema: with_options(&json!({"type": "object"}), false),
        written: String::new(),
    })
}

/// `json_schema`: the reply matches `schema`.
///
/// `strict` is the caller asking for exactly this schema: one the engine
/// cannot hold is refused (400) with the engine's reason. Without it the
/// caller asked for a best effort, so a keyword the engine does not know is
/// compiled as no constraint (`lenient`), and a schema that still will not
/// compile holds the reply to JSON of any shape — never less than it got
/// before #336, when nothing was held at all.
pub fn settle_json_schema(schema: &Value, strict: bool) -> Result<Arc<ReplyGrammar>, SwarmError> {
    if !schema.is_object() {
        return Err(SwarmError::Validation(
            "the reply format's schema must be a JSON object".into(),
        ));
    }
    let settled = with_options(schema, !strict);
    match compile_error(&settled) {
        None => Ok(Arc::new(ReplyGrammar {
            schema: settled,
            written: String::new(),
        })),
        Some(why) if strict => Err(SwarmError::Validation(format!(
            "the reply format's schema cannot be enforced: {why}"
        ))),
        Some(why) => {
            tracing::info!(
                target: "swarmllm::inference::reply_grammar",
                reason = %why,
                "a reply format's schema cannot be enforced; holding the reply to JSON of any shape",
            );
            Ok(Arc::new(ReplyGrammar {
                schema: with_options(&json!({}), true),
                written: String::new(),
            }))
        }
    }
}

/// The door's compiler: a vocabulary of one token per byte, built once.
static DOOR: LazyLock<Result<ParserFactory, String>> = LazyLock::new(|| {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
        .map_err(|e| e.to_string())?;
    factory.quiet();
    Ok(factory)
});

/// Why `schema` will not compile, or `None` when it does.
fn compile_error(schema: &Value) -> Option<String> {
    let factory = match DOOR.as_ref() {
        Ok(factory) => factory,
        Err(e) => return Some(format!("the grammar engine did not start: {e}")),
    };
    compile(factory, schema).err()
}

/// Compile `schema` into a matcher on `factory`'s vocabulary. The engine
/// catches its own panics inside a matcher's calls, not inside the compile;
/// a panic here is turned into the request's error, never the worker's end.
fn compile(factory: &ParserFactory, schema: &Value) -> Result<Matcher, String> {
    let grammar = TopLevelGrammar::from_json_schema(schema.clone());
    let parser = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        factory.create_parser(grammar)
    }))
    .map_err(|_| "the grammar compiler failed on this schema".to_string())?;
    let matcher = Matcher::new(parser);
    match matcher.get_error() {
        Some(e) => Err(e),
        None => Ok(matcher),
    }
}

/// One model's vocabulary as a held reply reads it: each entry's bytes, every
/// entry that is never text written `0xFF` + its name (the engine's marker for
/// a special token — no UTF-8 text contains the byte), and the model's
/// end-of-reply tokens, which the mask allows exactly where the JSON is whole.
pub struct GrammarVocab {
    factory: ParserFactory,
    eos: Vec<u32>,
}

impl GrammarVocab {
    /// From a loaded model's vocabulary, tokenizer and end-of-reply tokens.
    pub fn new(
        vocab: &[String],
        tokenizer: &crate::inference::split::SplitTokenizer,
        eos: &[u32],
    ) -> Result<Self, SwarmError> {
        let pieces = vocab
            .iter()
            .enumerate()
            .map(|(id, entry)| tokenizer.grammar_piece(id as u32, entry))
            .collect::<Vec<_>>();
        Self::from_pieces(&pieces, vocab, eos)
    }

    /// The vocabulary given as each entry's text bytes (`None`: never text)
    /// beside its spelling, which names a special entry.
    pub fn from_pieces(
        pieces: &[Option<Vec<u8>>],
        names: &[String],
        eos: &[u32],
    ) -> Result<Self, SwarmError> {
        // An end-of-reply id past the vocabulary (a declared id the
        // vocabulary array does not reach) can never be drawn as a word.
        let eos = eos
            .iter()
            .copied()
            .filter(|&id| (id as usize) < pieces.len())
            .collect::<Vec<_>>();
        let eos = eos.as_slice();
        let Some(&first_eos) = eos.first() else {
            return Err(SwarmError::Validation(
                "this model declares no token that ends a reply, so its reply cannot be held to JSON"
                    .into(),
            ));
        };
        let words = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| match piece {
                Some(bytes) if !eos.contains(&(id as u32)) => bytes.clone(),
                _ => {
                    let mut special = vec![0xFF];
                    special.extend_from_slice(names.get(id).map_or("", |n| n.as_str()).as_bytes());
                    special
                }
            })
            .collect::<Vec<_>>();
        let info = TokRxInfo {
            vocab_size: words.len() as u32,
            tok_eos: first_eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let trie = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TokTrie::from(&info, &words).with_eos_tokens(eos)
        }))
        .map_err(|_| {
            SwarmError::Inference("this model's vocabulary could not be read as a grammar's".into())
        })?;
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|e| SwarmError::Inference(format!("the grammar engine did not start: {e}")))?;
        factory.quiet();
        Ok(Self {
            factory,
            eos: eos.to_vec(),
        })
    }
}

/// One reply held to its grammar: asked before every draw, told every token
/// drawn. Speculation never meets one — `model_worker::ngram_spec_eligible`
/// and SWIFT's gate both refuse a request carrying a grammar, because a guess
/// drawn without the mask and taken back would leave the matcher a token
/// ahead of the reply.
pub struct HeldReply {
    /// `None`: let go — a continued reply whose text so far is not the start
    /// of the JSON its format asks for ([`HeldReply::new`]).
    matcher: Option<Matcher>,
    eos: Vec<u32>,
}

impl HeldReply {
    /// Compile `grammar` for `vocab`'s model. The door compiled the same
    /// schema, so a refusal here is a fault of this vocabulary, not of the
    /// caller's request — but it is still the caller's request that cannot be
    /// answered as asked, so it fails as one (400), before the prompt is read.
    ///
    /// A continued reply (`ReplyGrammar::written`, #236) is held from where its
    /// text left off, so it finishes the JSON it began — compiled fresh, the
    /// grammar opened a second object after the first's opening (pre-release
    /// review of v0.3.262). Where that text cannot be the start of the JSON —
    /// a split or a peer, which do not hold the format, wrote something else,
    /// and it has already reached the reader — the rest is let go, as every
    /// reply was before #336; holding it could only add a second document.
    pub fn new(vocab: &GrammarVocab, grammar: &ReplyGrammar) -> Result<Self, SwarmError> {
        let mut matcher = compile(&vocab.factory, &grammar.schema).map_err(|why| {
            SwarmError::Validation(format!(
                "the reply format cannot be held on this model: {why}"
            ))
        })?;
        let eos = vocab.eos.clone();
        if !grammar.written.is_empty() && !went_on_from(&mut matcher, grammar.written.as_bytes()) {
            tracing::warn!(
                target: "swarmllm::inference::reply_grammar",
                written_bytes = grammar.written.len(),
                "a continued reply's text so far is not the start of the JSON its format asks \
                 for, so the rest of it is not held to the format",
            );
            return Ok(Self { matcher: None, eos });
        }
        Ok(Self {
            matcher: Some(matcher),
            eos,
        })
    }

    /// Set every token that cannot come next to −∞: what continues the JSON,
    /// and the end-of-reply tokens once it is whole (only those, once nothing
    /// can follow). Entries past the vocabulary (a padded output head) are
    /// never allowed. A mask that leaves no finite score is an error — a draw
    /// from it would pick an arbitrary token.
    pub fn mask(&mut self, logits: &mut [f32]) -> Result<(), SwarmError> {
        let Some(matcher) = self.matcher.as_mut() else {
            return Ok(());
        };
        let allowed = matcher
            .compute_mask_or_eos()
            .map_err(|e| SwarmError::Inference(format!("the reply's JSON could not go on: {e}")))?;
        let mut any = false;
        for (id, score) in logits.iter_mut().enumerate() {
            if id < allowed.len() && allowed.is_allowed(id as u32) {
                any |= score.is_finite();
            } else {
                *score = f32::NEG_INFINITY;
            }
        }
        if any {
            Ok(())
        } else {
            Err(SwarmError::Inference(
                "no token the model can write continues the reply's JSON".into(),
            ))
        }
    }

    /// Advance past the token just drawn. An end-of-reply token ends the
    /// reply — the mask allowed it only where the JSON is whole — so there is
    /// nothing to advance past.
    pub fn took(&mut self, token: u32) -> Result<(), SwarmError> {
        let Some(matcher) = self.matcher.as_mut() else {
            return Ok(());
        };
        if self.eos.contains(&token) {
            return Ok(());
        }
        matcher
            .consume_token(token)
            .map_err(|e| SwarmError::Inference(format!("the reply's JSON could not go on: {e}")))
    }
}

/// Advance `matcher` past `written`, a continued reply's text so far, spelled
/// in the grammar's own vocabulary: the grammar reads BYTES, so any spelling of
/// the same text leaves it in the same place. Only an exact spelling counts —
/// the approximate tokenizer skips a byte its vocabulary lacks
/// (llguidance #138) — and only text the grammar accepts as a start.
fn went_on_from(matcher: &mut Matcher, written: &[u8]) -> bool {
    let Ok(env) = matcher.tok_env() else {
        return false;
    };
    let tokens = env.tokenize_bytes(written);
    env.tok_trie().decode_raw(&tokens) == written
        && matcher.consume_tokens(&tokens).is_ok()
        && !matcher.is_error()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small vocabulary: every printable ASCII byte, a few merged pieces,
    /// a control token and an end-of-reply token.
    fn vocab() -> (Vec<Option<Vec<u8>>>, Vec<String>, Vec<u32>) {
        let mut names: Vec<String> = (0x20u8..0x7F).map(|b| (b as char).to_string()).collect();
        names.extend(
            [
                "{\"",
                "\": ",
                "\"}",
                "red",
                "blue",
                "color",
                "```json",
                "<|im_start|>",
                "<|im_end|>",
            ]
            .map(String::from),
        );
        let start = names.iter().position(|n| n == "<|im_start|>").unwrap();
        let end = names.iter().position(|n| n == "<|im_end|>").unwrap();
        let pieces = names
            .iter()
            .enumerate()
            .map(|(id, n)| (id != start && id != end).then(|| n.as_bytes().to_vec()))
            .collect();
        (pieces, names, vec![end as u32])
    }

    /// Draw greedily from scores that prefer anything BUT valid JSON, under
    /// the mask, until the reply ends.
    fn held_reply_text(grammar: &ReplyGrammar, prefer: &[&str]) -> String {
        let (pieces, names, eos) = vocab();
        let gv = GrammarVocab::from_pieces(&pieces, &names, &eos).unwrap();
        let mut held = HeldReply::new(&gv, grammar).unwrap();
        let mut out = String::new();
        for _ in 0..64 {
            // The model's own preference: the listed pieces first, then
            // markdown, then the control token, then everything else.
            let mut scores = vec![0.0f32; names.len() + 3];
            for (rank, p) in prefer.iter().enumerate() {
                let id = names.iter().position(|n| n == p).unwrap();
                scores[id] = 100.0 - rank as f32;
            }
            scores[names.iter().position(|n| n == "```json").unwrap()] = 200.0;
            scores[names.iter().position(|n| n == "<|im_start|>").unwrap()] = 300.0;
            scores[names.len() + 1] = 400.0; // past the vocabulary
            held.mask(&mut scores).unwrap();
            let (tok, _) = scores
                .iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |b, (i, &s)| if s > b.1 { (i, s) } else { b },
                );
            held.took(tok as u32).unwrap();
            if eos.contains(&(tok as u32)) {
                return out;
            }
            out.push_str(&names[tok]);
        }
        panic!("the reply never ended: {out}");
    }

    #[test]
    fn a_json_object_reply_is_json_whatever_the_model_prefers() {
        let text = held_reply_text(&settle_json_object(), &["a", "}", "{"]);
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert!(parsed.is_object(), "{text}");
    }

    #[test]
    fn a_schema_reply_takes_one_of_the_values_it_allows() {
        let schema = json!({
            "type": "object",
            "properties": {"color": {"enum": ["red", "blue"]}},
            "required": ["color"],
            "additionalProperties": false,
        });
        let grammar = settle_json_schema(&schema, true).unwrap();
        let text = held_reply_text(&grammar, &["x", "blue", "red", "\""]);
        assert_eq!(text, r#"{"color": "blue"}"#);
    }

    /// A continued reply finishes the JSON it began (pre-release review of
    /// v0.3.262): compiled fresh, the grammar opened a second object after the
    /// text the reader already had.
    #[test]
    fn a_continued_reply_finishes_the_json_it_began() {
        let schema = json!({
            "type": "object",
            "properties": {"color": {"enum": ["red", "blue"]}},
            "required": ["color"],
            "additionalProperties": false,
        });
        let settled = settle_json_schema(&schema, true).unwrap();
        let written = r#"{"color": "bl"#;
        let continued = ReplyGrammar {
            written: written.into(),
            ..(*settled).clone()
        };
        let rest = held_reply_text(&continued, &["{", "{\"", "red", "x", "u", "e", "\"}"]);
        assert_eq!(
            format!("{written}{rest}"),
            r#"{"color": "blue"}"#,
            "rest: {rest}"
        );
    }

    /// Text the format cannot continue — written where nothing held it, and
    /// already with the reader — lets the rest go: held, it could only add a
    /// second document.
    #[test]
    fn a_continued_reply_that_is_not_json_so_far_is_let_go() {
        let (pieces, names, eos) = vocab();
        let gv = GrammarVocab::from_pieces(&pieces, &names, &eos).unwrap();
        let masked = |grammar: &ReplyGrammar| {
            let mut held = HeldReply::new(&gv, grammar).unwrap();
            let mut scores = vec![1.0f32; names.len()];
            held.mask(&mut scores).unwrap();
            scores.iter().filter(|s| s.is_infinite()).count()
        };
        assert!(
            masked(&settle_json_object()) > 0,
            "control: a fresh reply is held"
        );
        let continued = ReplyGrammar {
            written: "Sure! Here is".into(),
            ..(*settle_json_object()).clone()
        };
        assert_eq!(masked(&continued), 0, "let go: nothing is ruled out");
    }

    #[test]
    fn the_end_of_reply_token_is_allowed_only_once_the_json_is_whole() {
        let (pieces, names, eos) = vocab();
        let gv = GrammarVocab::from_pieces(&pieces, &names, &eos).unwrap();
        let mut held = HeldReply::new(&gv, &settle_json_object()).unwrap();
        let end = eos[0] as usize;
        let mut scores = vec![0.0f32; names.len()];
        held.mask(&mut scores).unwrap();
        assert_eq!(scores[end], f32::NEG_INFINITY, "an empty reply is not JSON");
        for piece in ["{", "}"] {
            held.took(names.iter().position(|n| n == piece).unwrap() as u32)
                .unwrap();
        }
        let mut scores = vec![0.0f32; names.len()];
        held.mask(&mut scores).unwrap();
        assert!(scores[end].is_finite(), "`{{}}` is whole");
        let still_open = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .count();
        assert_eq!(
            still_open, 1,
            "nothing may follow a whole object but the end"
        );
    }

    #[test]
    fn a_mask_that_leaves_nothing_drawable_is_an_error() {
        let (pieces, names, eos) = vocab();
        let gv = GrammarVocab::from_pieces(&pieces, &names, &eos).unwrap();
        let mut held = HeldReply::new(&gv, &settle_json_object()).unwrap();
        let mut scores = vec![f32::NEG_INFINITY; names.len()];
        assert!(held.mask(&mut scores).is_err());
    }

    #[test]
    fn a_strict_schema_the_engine_cannot_hold_is_refused_at_the_door() {
        // A reference this node will never fetch.
        let schema =
            json!({"type": "object", "properties": {"a": {"$ref": "https://example.com/a.json"}}});
        assert!(matches!(
            settle_json_schema(&schema, true),
            Err(SwarmError::Validation(_))
        ));
        // Without `strict` the reply is still held to JSON.
        let loose = settle_json_schema(&schema, false).unwrap();
        assert_eq!(loose.schema["x-guidance"]["lenient"], json!(true));
        let text = held_reply_text(&loose, &["a", "{", "}"]);
        assert!(serde_json::from_str::<Value>(&text).is_ok(), "{text}");
    }

    #[test]
    fn a_schema_that_is_not_an_object_is_refused() {
        assert!(matches!(
            settle_json_schema(&json!(true), false),
            Err(SwarmError::Validation(_))
        ));
    }

    #[test]
    fn the_callers_compiler_options_are_replaced_by_ours() {
        let schema = json!({"type": "object", "x-guidance": {"whitespace_flexible": true}});
        let settled = settle_json_schema(&schema, false).unwrap();
        assert_eq!(settled.schema["x-guidance"], compile_options(true));
    }

    /// The draw every worker reply goes through holds the reply — and refuses
    /// a draw whose grammar a call site forgot.
    #[test]
    fn the_worker_draw_holds_the_reply_and_refuses_a_forgotten_grammar() {
        let (pieces, names, eos) = vocab();
        let gv = GrammarVocab::from_pieces(&pieces, &names, &eos).unwrap();
        let grammar = settle_json_object();
        let params = crate::types::SamplingParams {
            temperature: 0.0,
            reply_grammar: Some(grammar.clone()),
            ..Default::default()
        };
        // The model's favourite is a markdown fence.
        let mut scores = vec![0.0f32; names.len()];
        scores[names.iter().position(|n| n == "```json").unwrap()] = 10.0;
        let logits = candle_core::Tensor::new(scores.as_slice(), &candle_core::Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let draw = crate::inference::tensor_util::sample_token_with_logprob_history;
        assert!(draw(&logits, &params, &[], None).is_err());
        let mut held = HeldReply::new(&gv, &grammar).unwrap();
        let (tok, _) = draw(&logits, &params, &[], Some(&mut held)).unwrap();
        assert!(
            names[tok as usize].starts_with('{'),
            "{}",
            names[tok as usize]
        );
    }

    #[test]
    fn a_model_with_no_end_of_reply_token_cannot_hold_a_reply() {
        let (pieces, names, _) = vocab();
        assert!(GrammarVocab::from_pieces(&pieces, &names, &[]).is_err());
    }
}
