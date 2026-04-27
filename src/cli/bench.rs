//! `swarmllm bench` — inference benchmarks against a running daemon.
//!
//! Measures total latency, tokens/sec, and concurrent throughput. With
//! `--stream` it also captures per-request time-to-first-token (TTFT) — the
//! signal that exposes Item 7 Phase 2 wins (Sarathi chunked prefill reduces
//! decode interruption from a long admission, so concurrent TTFT
//! distributions get tighter).

use super::{discover_model, read_api_key};
use futures::StreamExt;

/// Throughput in tokens/sec, guarded against zero-duration division.
fn tokens_per_sec(completion_tokens: u32, total_ms: f64) -> f64 {
    if total_ms > 0.0 && completion_tokens > 0 {
        completion_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    }
}

struct BenchResult {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_ms: f64,
    tokens_per_sec: f64,
    /// `Some(ms)` when the request was made in streaming mode (`--stream`).
    /// Time from request send to the first non-empty `delta.content` chunk.
    ttft_ms: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_bench(
    port: u16,
    data_dir: &std::path::Path,
    max_tokens: u32,
    iterations: u32,
    concurrency: u32,
    prompt: &str,
    json_output: bool,
    streaming: bool,
    model_override: Option<String>,
) -> anyhow::Result<()> {
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

    if !json_output {
        println!("SwarmLLM Benchmark");
        println!("==================");
        println!("Model:       {model}");
        println!("Max tokens:  {max_tokens}");
        println!("Iterations:  {iterations}");
        println!("Concurrency: {concurrency}");
        println!(
            "Streaming:   {} {}",
            if streaming { "yes" } else { "no" },
            if streaming { "(reports TTFT)" } else { "" }
        );
        println!("Prompt:      {}...", &prompt[..prompt.len().min(60)]);
        println!();
    }

    // --- Sequential latency test ---
    let mut results: Vec<BenchResult> = Vec::new();

    for i in 0..iterations {
        let r = if streaming {
            run_one_stream(&client, &base, &api_key, &model, prompt, max_tokens).await?
        } else {
            run_one_blocking(&client, &base, &api_key, &model, prompt, max_tokens).await?
        };

        if !json_output {
            print!(
                "  [{}/{}] {}ms | {} tokens | {:.1} tok/s",
                i + 1,
                iterations,
                r.total_ms as u64,
                r.completion_tokens,
                r.tokens_per_sec
            );
            if let Some(ttft) = r.ttft_ms {
                print!(" | TTFT {}ms", ttft as u64);
            }
            println!();
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
            let base = base.clone();
            let key = api_key.clone();
            let model = model.clone();
            let prompt = prompt.to_string();
            handles.push(tokio::spawn(async move {
                if streaming {
                    run_one_stream(&client, &base, &key, &model, &prompt, max_tokens).await
                } else {
                    run_one_blocking(&client, &base, &key, &model, &prompt, max_tokens).await
                }
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
            if streaming {
                let ttfts: Vec<f64> = concurrent_results
                    .iter()
                    .filter_map(|r| r.ttft_ms)
                    .collect();
                if !ttfts.is_empty() {
                    let min = ttfts.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = ttfts.iter().cloned().fold(0.0_f64, f64::max);
                    let avg = ttfts.iter().sum::<f64>() / ttfts.len() as f64;
                    println!(
                        "  TTFT: min {}ms / avg {}ms / max {}ms (n={})",
                        min as u64,
                        avg as u64,
                        max as u64,
                        ttfts.len()
                    );
                }
            }
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
    let avg_ttft: Option<f64> = {
        let ttfts: Vec<f64> = results.iter().filter_map(|r| r.ttft_ms).collect();
        if ttfts.is_empty() {
            None
        } else {
            Some(ttfts.iter().sum::<f64>() / ttfts.len() as f64)
        }
    };

    if json_output {
        let summary = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "iterations": iterations,
            "concurrency": concurrency,
            "streaming": streaming,
            "sequential": {
                "avg_latency_ms": avg_ms,
                "avg_tokens_per_sec": avg_tps,
                "min_tokens_per_sec": min_tps,
                "max_tokens_per_sec": max_tps,
                "avg_ttft_ms": avg_ttft,
                "results": results.iter().map(|r| serde_json::json!({
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_ms": r.total_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "ttft_ms": r.ttft_ms,
                })).collect::<Vec<_>>(),
            },
            "concurrent": if concurrency > 1 {
                let total_tokens: u32 = concurrent_results.iter().map(|r| r.completion_tokens).sum();
                let total_ms: f64 = concurrent_results.iter().map(|r| r.total_ms).fold(0.0_f64, f64::max);
                let ttfts: Vec<f64> = concurrent_results.iter().filter_map(|r| r.ttft_ms).collect();
                let (min_ttft, max_ttft, avg_ttft_c) = if ttfts.is_empty() {
                    (None, None, None)
                } else {
                    (
                        Some(ttfts.iter().cloned().fold(f64::INFINITY, f64::min)),
                        Some(ttfts.iter().cloned().fold(0.0_f64, f64::max)),
                        Some(ttfts.iter().sum::<f64>() / ttfts.len() as f64),
                    )
                };
                Some(serde_json::json!({
                    "requests": concurrent_results.len(),
                    "total_tokens": total_tokens,
                    "wall_time_ms": total_ms,
                    "aggregate_tokens_per_sec": if total_ms > 0.0 { total_tokens as f64 / (total_ms / 1000.0) } else { 0.0 },
                    "min_ttft_ms": min_ttft,
                    "avg_ttft_ms": avg_ttft_c,
                    "max_ttft_ms": max_ttft,
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
        if let Some(ttft) = avg_ttft {
            println!("Avg TTFT:     {:.0}ms", ttft);
        }
    }

    Ok(())
}

/// Non-streaming run: POST and wait for the full JSON response. Total time
/// rolls prefill + decode together; `ttft_ms` is `None`.
async fn run_one_blocking(
    client: &reqwest::Client,
    base: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<BenchResult> {
    let start = std::time::Instant::now();
    let resp: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "model": model,
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
    Ok(BenchResult {
        prompt_tokens,
        completion_tokens,
        total_ms,
        tokens_per_sec: tokens_per_sec(completion_tokens, total_ms),
        ttft_ms: None,
    })
}

/// Streaming run: POST with `stream: true`, parse SSE chunks, capture TTFT
/// (time to first non-empty `delta.content`).
async fn run_one_stream(
    client: &reqwest::Client,
    base: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<BenchResult> {
    let start = std::time::Instant::now();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": true,
        }))
        .send()
        .await?
        .error_for_status()?;

    let mut byte_stream = resp.bytes_stream();
    // SSE line buffer — accumulates a single `data: ...\n\n` event.
    let mut buf = String::new();
    let mut ttft_ms: Option<f64> = None;
    let mut completion_tokens: u32 = 0;
    let mut prompt_tokens: u32 = 0;

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // Drain whole SSE events (terminated by blank line `\n\n`).
        while let Some(end) = buf.find("\n\n") {
            let event: String = buf.drain(..end + 2).collect();
            let event = event.trim();
            if event.is_empty() {
                continue;
            }
            // Each event line starts with `data: `. Multiple `data:` lines
            // would concatenate with newlines per spec, but the OpenAI shape
            // emits one per event.
            for line in event.lines() {
                let line = line.trim_start();
                if !line.starts_with("data:") {
                    continue;
                }
                let payload = line.trim_start_matches("data:").trim();
                if payload == "[DONE]" {
                    // Done — fall out of the inner loop; outer loop hits EOS on next read.
                    return finalize_stream(start, ttft_ms, prompt_tokens, completion_tokens);
                }
                if payload.is_empty() {
                    continue;
                }
                // Parse the chunk JSON. Defensive — broken chunks shouldn't kill the bench.
                let v: serde_json::Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                    if !content.is_empty() {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(start.elapsed().as_millis() as f64);
                        }
                        // Approximate token count from chunk count — the
                        // server emits one Token per chunk in practice. This
                        // overcounts when a stop-string trim happens but is
                        // close enough for throughput numbers.
                        completion_tokens += 1;
                    }
                }
                if let Some(usage) = v["usage"].as_object() {
                    if let Some(pt) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
                        prompt_tokens = pt as u32;
                    }
                    if let Some(ct) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
                        completion_tokens = ct as u32;
                    }
                }
            }
        }
    }
    finalize_stream(start, ttft_ms, prompt_tokens, completion_tokens)
}

fn finalize_stream(
    start: std::time::Instant,
    ttft_ms: Option<f64>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> anyhow::Result<BenchResult> {
    let total_ms = start.elapsed().as_millis() as f64;
    Ok(BenchResult {
        prompt_tokens,
        completion_tokens,
        total_ms,
        tokens_per_sec: tokens_per_sec(completion_tokens, total_ms),
        ttft_ms,
    })
}
