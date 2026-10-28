//! `swarmllm test-split` — local single-node split-inference diagnostic.

use std::path::PathBuf;

pub async fn test_split_inference(
    model_path: Option<PathBuf>,
    max_tokens: u32,
    prompt: &str,
    prompt_chunk: usize,
    gpu_layers: Option<i32>,
    cpu_moe_layers: usize,
) -> anyhow::Result<()> {
    use swarmllm::inference::split::{sample_token, SplitModel};

    let model_path = model_path.ok_or_else(|| {
        anyhow::anyhow!(
            "--model is required for test-split.\n  Pass a local GGUF file path, e.g.:\n    swarmllm test-split --model ~/models/tinyllama-q4.gguf"
        )
    })?;
    println!("Loading full model from: {}", model_path.display());

    // Load as a single split covering ALL layers (0..N, is_first=true, is_last=true).
    // `--gpu-layers` is read exactly as a model worker reads it: 0 means the
    // processor only, N > 0 the card/processor split through the same
    // `GPU_LAYER_LIMIT` the worker sets. It checks the whole-file (mmap) load
    // path's split placement on a card without a daemon — the path a worker
    // takes for an imported model (`model.gguf` / `source_path`).
    let force_cpu = gpu_layers == Some(0);
    if let Some(n) = gpu_layers.filter(|&n| n > 0) {
        let n = n as usize;
        swarmllm::inference::split::GPU_LAYER_LIMIT.store(n, std::sync::atomic::Ordering::Relaxed);
        // A whole model's processor-held run sits early in the middle
        // (`hybrid::processor_run`); the loader logs which layers it took.
        println!("Placing {n} layers on the graphics card, the rest on the processor");
    }
    // `--cpu-moe-layers`, read through the `CPU_EXPERT_LAYERS` the loader
    // obeys (#138 (3)).
    if cpu_moe_layers > 0 {
        swarmllm::inference::split::CPU_EXPERT_LAYERS
            .store(cpu_moe_layers, std::sync::atomic::Ordering::Relaxed);
        println!(
            "Keeping the routed experts of the first {cpu_moe_layers} card-held layers in RAM"
        );
    }
    let mut model = SplitModel::load_from_gguf(&model_path, 0, 999, true, true, force_cpu)?;
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
            tool: Default::default(),
        }];
        let bos = model.bos_token();
        let eos = model.eos_token_str();
        swarmllm::inference::chat_template::apply_chat_template(
            template, &messages, bos, eos, true, None,
        )
        .unwrap_or_else(|| {
            // Said out loud: a diagnostic that quietly substitutes ChatML for the
            // model's own template hid exactly that failure from the check meant
            // to catch it (Qwen 3.5, gotcha #798).
            println!("Chat template FAILED to apply — using a ChatML fallback, not the model's own format");
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

    // In one forward, or through the node's own chunking
    // (`forward_prompt_in_chunks`): `--prompt-chunk` tokens a forward, raised
    // as a node raises it — to 512 on the card, and on a split copying its
    // weights to the card to what the card has room for (#138 (1e)).
    if token_ids.is_empty() {
        anyhow::bail!("the prompt has no tokens");
    }
    let input = candle_core::Tensor::from_vec(
        token_ids.clone(),
        &[1, token_ids.len()],
        &candle_core::Device::Cpu,
    )?;
    let logits = tokio::task::block_in_place(|| match prompt_chunk {
        0 => model.forward(&input, 0, &kv_store, test_request_id),
        n => model.forward_prompt_in_chunks(&input, 0, &kv_store, test_request_id, None, false, n),
    })
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

    // Generate remaining tokens. Each step is timed from the host's side,
    // sampling included: on a card the forward returns before the card has
    // finished, so only the step that reads the token back waits it out.
    let mut index_pos = token_ids.len();
    let decode_started = std::time::Instant::now();
    let mut decode_steps: u32 = 0;
    // The first steps pay one-time work (a graph's capture, the pool's
    // growth), so the steady rate is read from here on.
    const WARM_AFTER_STEPS: u32 = 8;
    let mut warm_from: Option<std::time::Instant> = None;
    #[allow(clippy::explicit_counter_loop)]
    for _ in 1..max_tokens {
        if decode_steps == WARM_AFTER_STEPS {
            warm_from = Some(std::time::Instant::now());
        }
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
        decode_steps += 1;

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
    let decode_secs = decode_started.elapsed().as_secs_f64();
    if decode_steps > 0 {
        println!(
            "Decode: {decode_steps} steps in {decode_secs:.2} s, {:.2} ms a step ({:.1} tokens a second)",
            decode_secs * 1000.0 / decode_steps as f64,
            decode_steps as f64 / decode_secs
        );
    }
    if let Some(warm) = warm_from {
        let steps = decode_steps - WARM_AFTER_STEPS;
        let secs = warm.elapsed().as_secs_f64();
        if steps > 0 {
            println!(
                "Decode after the first {WARM_AFTER_STEPS} steps: {steps} steps, {:.2} ms a step ({:.1} tokens a second)",
                secs * 1000.0 / steps as f64,
                steps as f64 / secs
            );
        }
    }
    println!("Total generated: {} tokens", generated.len());
    println!("Token IDs: {:?}", generated);

    Ok(())
}
