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

    /// Only claim specific shards for split inference (e.g. "0-4" or "5-8").
    /// When set, the node registers itself as holder of only these shards
    /// instead of all shards, enabling distributed inference across nodes.
    #[arg(long, global = true)]
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
            query_status(port).await
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

    // Parse --shards range (e.g. "0-4" → (0, 4))
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

    // Build and run daemon (spawns network, health, API tasks)
    let daemon = Daemon::new(config, identity, db);
    daemon.run().await
}

async fn query_status(port: u16) -> anyhow::Result<()> {
    let url = format!("http://localhost:{port}/v1/status");
    println!("Querying daemon at {url}...");

    // Simple HTTP GET using TCP directly to avoid adding reqwest dep
    let addr = format!("localhost:{port}");
    let stream = match tokio::net::TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Error: SwarmLLM daemon is not running on port {port}");
            std::process::exit(1);
        }
    };

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let request =
        format!("GET /v1/status HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\n\r\n");
    let (mut reader, mut writer) = stream.into_split();
    writer.write_all(request.as_bytes()).await?;
    writer.shutdown().await?;

    let mut response = String::new();
    reader.read_to_string(&mut response).await?;

    // Extract body from HTTP response
    if let Some(body_start) = response.find("\r\n\r\n") {
        let body = &response[body_start + 4..];
        // Pretty-print JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            println!("{body}");
        }
    } else {
        println!("{response}");
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

    // Build chat prompt
    let chat_prompt = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    println!("Chat prompt: {:?}", chat_prompt);

    // Tokenize
    let token_ids: Vec<i64> = if let Some(tokenizer) = model.tokenizer() {
        tokenizer.encode(&chat_prompt)
    } else {
        chat_prompt.bytes().map(|b| b as i64).collect()
    };
    println!("Prompt tokens ({}): {:?}", token_ids.len(), token_ids);

    // Prefill: run all tokens through
    let input = candle_core::Tensor::from_vec(
        token_ids.clone(),
        &[1, token_ids.len()],
        &candle_core::Device::Cpu,
    )?;

    let logits = model
        .forward(&input, 0)
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
            .forward(&input, index_pos)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let token_id = sample_token(&logits, 0.0, 1.0).map_err(|e| anyhow::anyhow!("{e}"))?;
        index_pos += 1;

        // Qwen2 EOS tokens
        if token_id == 151643 || token_id == 151645 {
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
    let filter = match verbose {
        0 => "swarmllm=info",
        1 => "swarmllm=debug",
        2 => "swarmllm=debug,libp2p=info,tower_http=debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .init();
}
