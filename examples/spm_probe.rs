//! Probe the SentencePiece encoder against a real GGUF vocabulary.
//!
//! Usage:
//!   cargo run --release --no-default-features --features dev,claude-subscription \
//!     --example spm_probe -- <path/to/gguf_header.bin> [word ...]
//!
//! Prints, per word, the token ids and the decoded pieces, and flags any word
//! that fell through to `<0xNN>` byte fallback — which for an ordinary English
//! word means the encoder failed, not that the vocabulary lacks it.

use swarmllm::inference::split::GgufTokenizerMeta;
use swarmllm::inference::split::SplitTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let header = match args.next() {
        Some(h) => h,
        None => {
            eprintln!("usage: spm_probe <gguf_header.bin> [word ...]");
            std::process::exit(2);
        }
    };
    let words: Vec<String> = {
        let rest: Vec<String> = args.collect();
        if rest.is_empty() {
            // The set from the original report: the first three fail, the rest pass.
            [
                "banana",
                "quantization",
                "pineapple",
                "apple",
                "computer",
                "distributed",
                "hello",
                "system",
                "networking",
                "the",
                "running",
                "strawberry",
                "elephant",
                "tokenizer",
                "unbelievable",
                "consciousness",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        } else {
            rest
        }
    };

    let meta = GgufTokenizerMeta::from_gguf_file(std::path::Path::new(&header))
        .expect("failed to read gguf header");
    // stderr, so `--from-file` stdout is ids-only and diffable.
    eprintln!(
        "vocab={} scores={} model={} pre={} add_space_prefix={}",
        meta.vocab.len(),
        meta.scores.len(),
        meta.tokenizer_model,
        meta.pre_tokenizer,
        meta.add_space_prefix
    );
    // Refuse rather than warn. Warning and carrying on produced a clean bill of
    // health for a run that tested nothing: with a BPE vocabulary there are no
    // scores, so the SPM encoder built below returns ZERO tokens for every
    // word, each of which was then printed "ok" (no byte fallback in an empty
    // list) and summarised as "0/16 words hit byte fallback". A reader checking
    // whether a Llama-3 node tokenises correctly got a pass from a probe that
    // never ran. Exiting is the honest answer.
    if meta.tokenizer_model != "llama" {
        eprintln!(
            "This vocabulary is '{}' (pre-tokenizer '{}'), not SentencePiece — \n\
             spm_probe only exercises the SPM encoder and would report a \n\
             meaningless pass here. Nothing was checked.",
            meta.tokenizer_model, meta.pre_tokenizer
        );
        std::process::exit(2);
    }

    // Build the SPM tokenizer directly, with BOS off so the output is only the
    // word's own pieces.
    let tok = SplitTokenizer::from_sentencepiece(
        &meta.vocab,
        &meta.scores,
        meta.add_space_prefix,
        false,
        meta.bos_token_id,
    );

    // `--from-file <path>`: one input per line, print `id,id,id` per line and
    // nothing else, so the output can be diffed against a reference encoder.
    if words.first().map(|s| s.as_str()) == Some("--from-file") {
        let path = words.get(1).expect("--from-file needs a path");
        let data = std::fs::read_to_string(path).expect("read input file");
        for line in data.lines() {
            let ids = tok.encode(line);
            println!(
                "{}",
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        return;
    }

    let mut failures = 0usize;
    for w in &words {
        let ids = tok.encode(w);
        let pieces: Vec<String> = ids
            .iter()
            .map(|&id| {
                meta.vocab
                    .get(id as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("<oob {id}>"))
            })
            .collect();
        let byte_fallback = pieces
            .iter()
            .any(|p| p.starts_with("<0x") && p.ends_with('>'));
        // An empty encoding is a failure in its own right, not a word that
        // merely avoided byte fallback. Counting only byte fallback let a
        // tokenizer that produced nothing at all report "ok" on every line.
        let empty = ids.is_empty();
        if byte_fallback || empty {
            failures += 1;
        }
        let verdict = if empty {
            "EMPTY"
        } else if byte_fallback {
            "FAIL"
        } else {
            "ok"
        };
        println!("{:<16} {:<6} {:?}  {}", w, verdict, pieces, ids.len());
    }
    println!(
        "\n{failures}/{} words failed to encode cleanly",
        words.len()
    );
    if failures > 0 {
        std::process::exit(1);
    }
}
