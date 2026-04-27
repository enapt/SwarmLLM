//! `swarmllm chat` — interactive terminal REPL against a running daemon.

use super::{discover_model, read_api_key};

pub async fn run_chat(
    port: u16,
    data_dir: &std::path::Path,
    model_override: Option<String>,
    max_tokens: u32,
    temperature: f32,
) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let api_key = read_api_key(data_dir).unwrap_or_default();
    if api_key.is_empty() {
        anyhow::bail!(
            "SwarmLLM is not running (no API key at {}).\n  Start the daemon first: swarmllm run",
            data_dir.join("api_key").display()
        );
    }

    let base = format!("http://localhost:{port}");
    let client = reqwest::Client::new();

    let model = discover_model(&client, &base, &api_key, model_override).await?;

    println!("SwarmLLM Chat — model: {model}");
    println!("Type your message and press Enter. Type 'quit' or Ctrl-D to exit.\n");

    let mut messages: Vec<serde_json::Value> = Vec::new();
    let stdin = std::io::stdin();

    loop {
        print!("You: ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            // EOF (Ctrl-D)
            println!();
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        messages.push(serde_json::json!({"role": "user", "content": input}));

        let resp: serde_json::Value = client
            .post(format!("{base}/v1/chat/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": &model,
                "messages": &messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }))
            .send()
            .await?
            .json()
            .await?;

        if let Some(err) = resp.get("error") {
            // Extract just the message field rather than dumping the full
            // JSON envelope. Falls back to the raw object if the shape is
            // unfamiliar (non-OpenAI provider error).
            let msg = err
                .get("message")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| err.to_string());
            eprintln!("Error: {msg}");
            if let Some(hint) = err.get("hint").and_then(|v| v.as_str()) {
                eprintln!("Hint: {hint}");
            }
            messages.pop();
            continue;
        }

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("(no response)");
        println!("\nAssistant: {content}\n");

        messages.push(serde_json::json!({"role": "assistant", "content": content}));
    }

    Ok(())
}
