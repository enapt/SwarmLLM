use std::path::PathBuf;

use clap::{Parser, Subcommand};

use swarmllm::config::Config;
use swarmllm::daemon::Daemon;
use swarmllm::identity::Identity;
use swarmllm::storage::db::Database;

#[derive(Parser)]
#[command(
    name = "swarmllm",
    version,
    about = "Decentralized peer-to-peer LLM inference network"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Config file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Listen port
    #[arg(short, long, global = true)]
    port: Option<u16>,

    /// Data directory
    #[arg(short, long, global = true)]
    data_dir: Option<PathBuf>,

    /// Increase log verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Path to a GGUF model file to load
    #[arg(short, long, global = true)]
    model: Option<PathBuf>,

    /// Number of layers to offload to GPU (0 = CPU only)
    #[arg(long, global = true)]
    gpu_layers: Option<u32>,

    /// Bootstrap peer multiaddr (e.g. /ip4/127.0.0.1/udp/8800/quic-v1)
    #[arg(long, global = true)]
    bootstrap: Vec<String>,

    /// [Advanced/Dev] Only claim specific shards for split inference (e.g. "0-4" or "5-8").
    /// Not needed for normal use — the node auto-detects all local shards.
    #[arg(long, global = true, hide = true)]
    shards: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the SwarmLLM daemon
    Run {
        /// Disable automatic update checking
        #[arg(long)]
        no_update_check: bool,
    },
    /// Print version information
    Version,
    /// Show node status (queries running daemon)
    Status,
    /// Check for updates and apply if available
    Update {
        /// Only check, do not download or apply
        #[arg(long)]
        check_only: bool,
    },
    /// Test split inference locally (no networking, single-node diagnostic)
    TestSplit {
        /// Number of tokens to generate
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        /// Prompt text
        #[arg(long, default_value = "Hello, how are you?")]
        prompt: String,
    },
    /// Run inference benchmarks against a running daemon
    Bench {
        /// Number of tokens to generate per request
        #[arg(long, default_value = "100")]
        max_tokens: u32,
        /// Number of sequential requests
        #[arg(long, default_value = "5")]
        iterations: u32,
        /// Number of concurrent requests (for throughput test)
        #[arg(long, default_value = "1")]
        concurrency: u32,
        /// Prompt text
        #[arg(
            long,
            default_value = "Explain the theory of relativity in simple terms."
        )]
        prompt: String,
        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    init_tracing(cli.verbose);

    // Determine if update checking is disabled via CLI flag
    let no_update_check = matches!(
        &cli.command,
        Some(Commands::Run {
            no_update_check: true
        })
    );
    // Take command out of cli so we can move cli into run_daemon
    let command = cli.command.take().unwrap_or(Commands::Run {
        no_update_check: false,
    });

    match command {
        Commands::Run { .. } => run_daemon(cli, no_update_check).await,
        Commands::Version => {
            println!("swarmllm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Commands::Status => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = cli
                .data_dir
                .clone()
                .or_else(|| {
                    std::env::var("SWARMLLM_NODE_DATA_DIR")
                        .ok()
                        .map(PathBuf::from)
                })
                .unwrap_or_else(|| {
                    dirs::data_dir()
                        .unwrap_or_else(|| {
                            #[cfg(unix)]
                            {
                                PathBuf::from("/var/lib/swarmllm")
                            }
                            #[cfg(not(unix))]
                            {
                                PathBuf::from(".")
                            }
                        })
                        .join("swarmllm")
                });
            query_status(port, &data_dir).await
        }
        Commands::Update { check_only } => run_update_command(check_only).await,
        Commands::TestSplit { max_tokens, prompt } => {
            test_split_inference(cli.model.clone(), max_tokens, &prompt).await
        }
        Commands::Bench {
            max_tokens,
            iterations,
            concurrency,
            prompt,
            json,
        } => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = cli
                .data_dir
                .clone()
                .or_else(|| {
                    std::env::var("SWARMLLM_NODE_DATA_DIR")
                        .ok()
                        .map(PathBuf::from)
                })
                .unwrap_or_else(|| {
                    dirs::data_dir()
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join("swarmllm")
                });
            run_bench(
                port,
                &data_dir,
                max_tokens,
                iterations,
                concurrency,
                &prompt,
                json,
            )
            .await
        }
    }
}

async fn run_daemon(cli: Cli, no_update_check: bool) -> anyhow::Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "DIAG: daemon starting");

    // Load config
    let mut config = Config::load_or_create(
        cli.config.as_deref(),
        cli.port,
        cli.data_dir.as_deref(),
        cli.model.as_deref(),
        cli.gpu_layers,
        cli.bootstrap,
    )?;

    // Parse --shards range (e.g. "0-4" → (0, 4)) — hidden dev flag
    if let Some(ref shard_str) = cli.shards {
        if let Some((start, end)) = shard_str.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                config.inference.shard_range = Some((s, e));
                tracing::info!(shard_start = s, shard_end = e, "Node claiming shard range");
            } else {
                anyhow::bail!("Invalid --shards format: expected 'START-END' (e.g. '0-4')");
            }
        } else {
            anyhow::bail!("Invalid --shards format: expected 'START-END' (e.g. '0-4')");
        }
    }

    // CLI --no-update-check overrides config
    if no_update_check {
        config.updates.auto_update = swarmllm::config::AutoUpdateMode::Disabled;
    }

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    // Load or generate node identity
    let identity = Identity::load_or_generate(&config.node.data_dir)?;

    // Open database
    let db = Database::open(&config.node.data_dir)?;

    // Persist or restore shard range: CLI flag takes priority, else load from DB
    if let Some((s, e)) = config.inference.shard_range {
        // CLI provided --shards: persist to DB for future runs
        if let Err(err) = db.save_shard_range(s, e) {
            tracing::warn!(error = %err, "Failed to persist shard range to database");
        }
    } else {
        // No --shards flag: try restoring from DB
        match db.load_shard_range() {
            Ok(Some((s, e))) => {
                config.inference.shard_range = Some((s, e));
                tracing::info!(
                    shard_start = s,
                    shard_end = e,
                    "Restored shard range from previous session"
                );
            }
            Ok(None) => {} // No persisted range, normal operation
            Err(err) => {
                tracing::warn!(error = %err, "Failed to load shard range from database");
            }
        }
    }

    // Build and run daemon (spawns network, health, API tasks)
    let daemon = Daemon::new(config, identity, db);
    daemon.run().await
}

async fn run_update_command(check_only: bool) -> anyhow::Result<()> {
    use std::sync::Arc;
    use swarmllm::update::{UpdateChecker, UpdateState};
    use tokio::sync::RwLock;

    println!(
        "SwarmLLM {} — checking for updates...",
        env!("CARGO_PKG_VERSION")
    );

    let config = swarmllm::config::UpdateConfig::default();
    let state = Arc::new(RwLock::new(UpdateState::default()));
    let (update_tx, _) = tokio::sync::broadcast::channel(4);
    let checker = UpdateChecker::new(config, "enapt/SwarmLLM".to_string(), state, update_tx);

    match checker.check_for_update().await {
        Ok(Some(info)) => {
            println!(
                "Update available: v{} -> v{}",
                info.current_version, info.latest_version
            );
            println!("Published: {}", info.published_at);
            if !info.changelog.is_empty() {
                println!("\nChangelog:\n{}", info.changelog);
            }

            if check_only {
                return Ok(());
            }

            println!("\nDownloading...");
            match checker.download_update(&info).await {
                Ok(tmp_path) => {
                    println!("Downloaded to: {}", tmp_path.display());
                    println!("Applying update...");
                    match checker.apply_update(&tmp_path) {
                        Ok(()) => {
                            println!(
                                "Update applied successfully! Restart SwarmLLM to use v{}.",
                                info.latest_version
                            );
                        }
                        Err(e) => {
                            eprintln!("Failed to apply update: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to download update: {e}");
                    std::process::exit(1);
                }
            }
        }
        Ok(None) => {
            println!(
                "You are running the latest version (v{}).",
                env!("CARGO_PKG_VERSION")
            );
        }
        Err(e) => {
            eprintln!("Failed to check for updates: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn query_status(port: u16, data_dir: &std::path::Path) -> anyhow::Result<()> {
    // Read the API key from the plain file written by the daemon
    let key_path = data_dir.join("api_key");
    let api_key = std::fs::read_to_string(&key_path)
        .unwrap_or_default()
        .trim()
        .to_string();

    if api_key.is_empty() {
        eprintln!("Warning: no API key found at {}", key_path.display());
        eprintln!("         (is the daemon running with this data directory?)");
    }

    let url = format!("http://localhost:{port}/v1/status");
    println!("Querying daemon at {url}...");

    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    match req.send().await {
        Ok(resp) => {
            let body = resp.text().await?;
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                println!("{body}");
            }
        }
        Err(_) => {
            eprintln!("Error: SwarmLLM daemon is not running on port {port}");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn test_split_inference(
    model_path: Option<PathBuf>,
    max_tokens: u32,
    prompt: &str,
) -> anyhow::Result<()> {
    use swarmllm::inference::split::{sample_token, SplitModel};

    let model_path =
        model_path.ok_or_else(|| anyhow::anyhow!("--model required for test-split"))?;
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
        let last_token = *generated.last().unwrap() as i64;
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

fn init_tracing(verbose: u8) {
    // CLI verbose flags override any config file setting
    let filter = if verbose > 0 {
        match verbose {
            1 => "swarmllm=debug".to_string(),
            2 => "swarmllm=debug,libp2p=info,libp2p_request_response=debug,libp2p_swarm=debug,yamux=debug,multistream_select=debug,tower_http=debug".to_string(),
            _ => "trace".to_string(),
        }
    } else {
        // Try to read logging.level from default config location
        let config_level = dirs::data_dir()
            .map(|d| d.join("swarmllm").join("config.toml"))
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|contents| toml::from_str::<toml::Value>(&contents).ok())
            .and_then(|v| {
                v.get("logging")
                    .and_then(|l| l.get("level"))
                    .and_then(|l| l.as_str().map(String::from))
            });
        match config_level.as_deref() {
            Some("debug") => "swarmllm=debug".to_string(),
            Some("trace") => "trace".to_string(),
            Some("warn") => "swarmllm=warn".to_string(),
            Some("error") => "swarmllm=error".to_string(),
            _ => "swarmllm=info".to_string(),
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(&filter)
        .with_target(true)
        .with_thread_ids(false)
        .init();
}

/// Run inference benchmarks against a running SwarmLLM daemon.
///
/// Measures: time-to-first-token (TTFT), tokens/sec, total latency, and
/// concurrent throughput when concurrency > 1.
async fn run_bench(
    port: u16,
    data_dir: &std::path::Path,
    max_tokens: u32,
    iterations: u32,
    concurrency: u32,
    prompt: &str,
    json_output: bool,
) -> anyhow::Result<()> {
    // Read API key
    let key_path = data_dir.join("api_key");
    let api_key = std::fs::read_to_string(&key_path)
        .unwrap_or_default()
        .trim()
        .to_string();
    if api_key.is_empty() {
        anyhow::bail!(
            "No API key at {} — is the daemon running?",
            key_path.display()
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
            iteration: i,
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

        for i in 0..concurrency {
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
                    iteration: i,
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

struct BenchResult {
    #[allow(dead_code)]
    iteration: u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_ms: f64,
    tokens_per_sec: f64,
}
