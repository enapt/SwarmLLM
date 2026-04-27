//! `swarmllm test-split` — local single-node split-inference diagnostic.

use std::path::PathBuf;

pub async fn test_split_inference(
    model_path: Option<PathBuf>,
    max_tokens: u32,
    prompt: &str,
) -> anyhow::Result<()> {
    use swarmllm::inference::split::{sample_token, SplitModel};

    let model_path = model_path.ok_or_else(|| {
        anyhow::anyhow!(
            "--model is required for test-split.\n  Pass a local GGUF file path, e.g.:\n    swarmllm test-split --model ~/models/tinyllama-q4.gguf"
        )
    })?;
    println!("Loading full model from: {}", model_path.display());

    // Load as a single split covering ALL layers (0..N, is_first=true, is_last=true)
    let mut model = SplitModel::load_from_gguf(&model_path, 0, 999, true, true)?;
    let total_layers = model.total_layers;
    println!(
        "Model loaded: {} layers, hidden_dim={}",
        total_layers, model.hidden_dim
    );

    // Build chat prompt — use model's chat template if available, else fallback to ChatML
    let chat_prompt = if let Some(template) = model.chat_template() {
        let messages = vec![swarmllm::types::ChatMessage {
            role: swarmllm::types::Role::User,
            content: prompt.to_string(),
            images: vec![],
        }];
        let bos = model.bos_token();
        let eos = model.eos_token_str();
        swarmllm::inference::chat_template::apply_chat_template(template, &messages, bos, eos, true)
            .unwrap_or_else(|| {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            })
    } else {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    };
    println!("Chat prompt: {:?}", chat_prompt);

    // Tokenize
    let token_ids: Vec<i64> = if let Some(tokenizer) = model.tokenizer() {
        tokenizer.encode(&chat_prompt)
    } else {
        chat_prompt.bytes().map(|b| b as i64).collect()
    };
    println!("Prompt tokens ({}): {:?}", token_ids.len(), token_ids);

    // Prefill: run all tokens through
    let kv_store =
        swarmllm::inference::split::KvCacheStore::new(std::time::Duration::from_secs(600));
    let test_request_id = "test-generate";

    let input = candle_core::Tensor::from_vec(
        token_ids.clone(),
        &[1, token_ids.len()],
        &candle_core::Device::Cpu,
    )?;

    let logits =
        tokio::task::block_in_place(|| model.forward(&input, 0, &kv_store, test_request_id))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let first_token = sample_token(&logits, 0.0, 1.0).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut generated = vec![first_token];
    print!("Generated: ");

    // Decode and print first token
    if let Some(vocab) = model.vocab() {
        if let Some(t) = vocab.get(first_token as usize) {
            if let Some(tokenizer) = model.tokenizer() {
                let bytes = tokenizer.decode_token(t);
                print!("{}", String::from_utf8_lossy(&bytes));
            } else {
                print!("{t}");
            }
        }
    }

    // Generate remaining tokens
    let mut index_pos = token_ids.len();
    for _ in 1..max_tokens {
        let last_token = *generated
            .last()
            .expect("generated tokens must be non-empty") as i64;
        let input =
            candle_core::Tensor::from_vec(vec![last_token], &[1, 1], &candle_core::Device::Cpu)?;

        let logits = tokio::task::block_in_place(|| {
            model.forward(&input, index_pos, &kv_store, test_request_id)
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let token_id = sample_token(&logits, 0.0, 1.0).map_err(|e| anyhow::anyhow!("{e}"))?;
        index_pos += 1;

        // Check EOS tokens loaded from GGUF metadata
        if model.eos_tokens().contains(&token_id) {
            println!(" [EOS:{}]", token_id);
            break;
        }

        generated.push(token_id);

        if let Some(vocab) = model.vocab() {
            if let Some(t) = vocab.get(token_id as usize) {
                if let Some(tokenizer) = model.tokenizer() {
                    let bytes = tokenizer.decode_token(t);
                    print!("{}", String::from_utf8_lossy(&bytes));
                } else {
                    print!("{t}");
                }
            }
        }
    }
    println!();
    println!("Total generated: {} tokens", generated.len());
    println!("Token IDs: {:?}", generated);

    Ok(())
}
