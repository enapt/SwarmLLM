//! `swarmllm bench` — inference benchmarks against a running daemon.
//!
//! Measures time-to-first-token (TTFT), tokens/sec, total latency, and
//! concurrent throughput when concurrency > 1.

use super::read_api_key;

struct BenchResult {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_ms: f64,
    tokens_per_sec: f64,
}

pub async fn run_bench(
    port: u16,
    data_dir: &std::path::Path,
    max_tokens: u32,
    iterations: u32,
    concurrency: u32,
    prompt: &str,
    json_output: bool,
) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir).unwrap_or_default();
    if api_key.is_empty() {
        anyhow::bail!(
            "No API key at {} — is the daemon running?",
            data_dir.join("api_key").display()
        );
    }

    let base = format!("http://localhost:{port}");
    let client = reqwest::Client::new();

    // Discover model
    let models_resp: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await?
        .json()
        .await?;
    let model = models_resp["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No models available — load a model first"))?
        .to_string();

    if !json_output {
        println!("SwarmLLM Benchmark");
        println!("==================");
        println!("Model:       {model}");
        println!("Max tokens:  {max_tokens}");
        println!("Iterations:  {iterations}");
        println!("Concurrency: {concurrency}");
        println!("Prompt:      {}...", &prompt[..prompt.len().min(60)]);
        println!();
    }

    // --- Sequential latency test ---
    let mut results: Vec<BenchResult> = Vec::new();

    for i in 0..iterations {
        let start = std::time::Instant::now();
        let resp: serde_json::Value = client
            .post(format!("{base}/v1/chat/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": &model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }))
            .send()
            .await?
            .json()
            .await?;

        let elapsed = start.elapsed();
        let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let total_ms = elapsed.as_millis() as f64;
        let tokens_per_sec = if total_ms > 0.0 {
            completion_tokens as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };

        let r = BenchResult {
            prompt_tokens,
            completion_tokens,
            total_ms,
            tokens_per_sec,
        };

        if !json_output {
            println!(
                "  [{}/{}] {}ms | {} tokens | {:.1} tok/s",
                i + 1,
                iterations,
                total_ms as u64,
                completion_tokens,
                tokens_per_sec
            );
        }
        results.push(r);
    }

    // --- Concurrent throughput test ---
    let mut concurrent_results: Vec<BenchResult> = Vec::new();
    if concurrency > 1 {
        if !json_output {
            println!("\nConcurrent throughput ({concurrency} parallel requests):");
        }
        let batch_start = std::time::Instant::now();
        let mut handles = Vec::new();

        for _i in 0..concurrency {
            let client = client.clone();
            let url = format!("{base}/v1/chat/completions");
            let key = api_key.clone();
            let model = model.clone();
            let prompt = prompt.to_string();
            handles.push(tokio::spawn(async move {
                let start = std::time::Instant::now();
                let resp: serde_json::Value = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {key}"))
                    .json(&serde_json::json!({
                        "model": &model,
                        "messages": [{"role": "user", "content": &prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    }))
                    .send()
                    .await?
                    .json()
                    .await?;
                let elapsed = start.elapsed();
                let ct = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
                let pt = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                let ms = elapsed.as_millis() as f64;
                Ok::<_, reqwest::Error>(BenchResult {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    total_ms: ms,
                    tokens_per_sec: if ms > 0.0 {
                        ct as f64 / (ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            }));
        }

        for handle in handles {
            match handle.await? {
                Ok(r) => concurrent_results.push(r),
                Err(e) => eprintln!("  Request failed: {e}"),
            }
        }

        let batch_elapsed = batch_start.elapsed();
        let total_tokens: u32 = concurrent_results.iter().map(|r| r.completion_tokens).sum();
        let batch_tok_per_sec = if batch_elapsed.as_millis() > 0 {
            total_tokens as f64 / (batch_elapsed.as_millis() as f64 / 1000.0)
        } else {
            0.0
        };

        if !json_output {
            println!(
                "  {} requests completed in {}ms | {} total tokens | {:.1} tok/s aggregate",
                concurrent_results.len(),
                batch_elapsed.as_millis(),
                total_tokens,
                batch_tok_per_sec
            );
        }
    }

    // --- Summary ---
    if results.is_empty() {
        println!("No benchmark iterations completed.");
        return Ok(());
    }
    let avg_ms: f64 = results.iter().map(|r| r.total_ms).sum::<f64>() / results.len() as f64;
    let avg_tps: f64 = results.iter().map(|r| r.tokens_per_sec).sum::<f64>() / results.len() as f64;
    let min_tps = results
        .iter()
        .map(|r| r.tokens_per_sec)
        .fold(f64::INFINITY, f64::min);
    let max_tps = results
        .iter()
        .map(|r| r.tokens_per_sec)
        .fold(0.0_f64, f64::max);

    if json_output {
        let summary = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "iterations": iterations,
            "concurrency": concurrency,
            "sequential": {
                "avg_latency_ms": avg_ms,
                "avg_tokens_per_sec": avg_tps,
                "min_tokens_per_sec": min_tps,
                "max_tokens_per_sec": max_tps,
                "results": results.iter().map(|r| serde_json::json!({
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_ms": r.total_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                })).collect::<Vec<_>>(),
            },
            "concurrent": if concurrency > 1 {
                let total_tokens: u32 = concurrent_results.iter().map(|r| r.completion_tokens).sum();
                let total_ms: f64 = concurrent_results.iter().map(|r| r.total_ms).fold(0.0_f64, f64::max);
                Some(serde_json::json!({
                    "requests": concurrent_results.len(),
                    "total_tokens": total_tokens,
                    "wall_time_ms": total_ms,
                    "aggregate_tokens_per_sec": if total_ms > 0.0 { total_tokens as f64 / (total_ms / 1000.0) } else { 0.0 },
                }))
            } else {
                None
            },
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("\nSummary");
        println!("-------");
        println!("Avg latency:  {:.0}ms", avg_ms);
        println!("Avg tok/s:    {:.1}", avg_tps);
        println!("Min tok/s:    {:.1}", min_tps);
        println!("Max tok/s:    {:.1}", max_tps);
    }

    Ok(())
}
