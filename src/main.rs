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
    Run,
    /// Print version information
    Version,
    /// Show node status (queries running daemon)
    Status,
    /// Test split inference locally (no networking, single-node diagnostic)
    TestSplit {
        /// Number of tokens to generate
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        /// Prompt text
        #[arg(long, default_value = "Hello, how are you?")]
        prompt: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    init_tracing(cli.verbose);

    match cli.command.as_ref().unwrap_or(&Commands::Run) {
        Commands::Run => run_daemon(cli).await,
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
        Commands::TestSplit { max_tokens, prompt } => {
            test_split_inference(cli.model, *max_tokens, prompt).await
        }
    }
}

async fn run_daemon(cli: Cli) -> anyhow::Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Starting SwarmLLM");

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

    let logits = model
        .forward(&input, 0, &kv_store, test_request_id)
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

        let logits = model
            .forward(&input, index_pos, &kv_store, test_request_id)
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
            2 => "swarmllm=debug,libp2p=info,tower_http=debug".to_string(),
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
