//! Times `SplitTokenizer::encode` against prompt length, to tell an O(n)
//! tokenizer from an O(n^2) one. Point SWARM_TOK_HEADER at a model's
//! `gguf_header.bin`.
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let header = std::env::var("SWARM_TOK_HEADER").expect("set SWARM_TOK_HEADER");
    let path = PathBuf::from(header);
    let meta = swarmllm::inference::split::GgufTokenizerMeta::from_gguf_file(&path)
        .expect("read gguf header");
    println!(
        "model={:?} merges={} scores={} vocab={} add_space_prefix={}",
        meta.tokenizer_model,
        meta.merges.len(),
        meta.scores.len(),
        meta.vocab.len(),
        meta.add_space_prefix
    );
    let t0 = Instant::now();
    let tok = meta.build_tokenizer().expect("build tokenizer");
    println!("tokenizer built in {:?}", t0.elapsed());

    if std::env::var("SWARM_TOK_TEMPLATE").is_ok() {
        match &meta.chat_template {
            Some(t) => {
                println!(
                    "template_len={} has_tools={} has_tool_call_tag={}",
                    t.len(),
                    t.contains("tools"),
                    t.contains("tool_call")
                );
                println!("--- template ---\n{t}");
            }
            None => println!("NO CHAT TEMPLATE IN HEADER"),
        }
        return;
    }
    if let Ok(text) = std::env::var("SWARM_TOK_TEXT") {
        let ids = tok.encode(&text);
        println!("TEXT {text:?}");
        println!("IDS  {ids:?}");
        return;
    }
    let unit = "the quick brown fox jumps over the lazy dog. ";
    let mut prev: Option<(usize, f64)> = None;
    for words in [125usize, 250, 500, 1000, 2000] {
        let text = unit.repeat(words);
        let t = Instant::now();
        let ids = tok.encode(&text);
        let secs = t.elapsed().as_secs_f64();
        let ratio = match prev {
            Some((pc, ps)) => format!(
                "  x{:.2} time for x{:.2} chars",
                secs / ps,
                text.len() as f64 / pc as f64
            ),
            None => String::new(),
        };
        println!(
            "chars {:>6}  tokens {:>6}  encode {:>8.3}s{}",
            text.len(),
            ids.len(),
            secs,
            ratio
        );
        prev = Some((text.len(), secs));
    }
}
