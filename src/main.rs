use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::sync::Mutex;

use swarmllm::config::Config;
use swarmllm::daemon::Daemon;
use swarmllm::identity::Identity;
use swarmllm::inference::executor::ModelExecutor;
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
}

#[derive(Subcommand)]
enum Commands {
    /// Start the SwarmLLM daemon
    Run,
    /// Print version information
    Version,
    /// Show node status (queries running daemon)
    Status,
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
    }
}

async fn run_daemon(cli: Cli) -> anyhow::Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Starting SwarmLLM");

    // Load config
    let config = Config::load_or_create(
        cli.config.as_deref(),
        cli.port,
        cli.data_dir.as_deref(),
        cli.model.as_deref(),
        cli.gpu_layers,
    )?;

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    // Load or generate node identity
    let identity = Identity::load_or_generate(&config.node.data_dir)?;

    // Open database
    let db = Database::open(&config.node.data_dir)?;

    // Initialize model executor
    let mut executor = ModelExecutor::new();
    if let Some(ref model_path) = config.inference.model_path {
        match executor.load_model(model_path, config.inference.gpu_layers) {
            Ok(()) => tracing::info!("Model ready"),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load model — running without inference")
            }
        }
    } else {
        tracing::info!(
            "No model path configured — running API server without inference capability"
        );
        tracing::info!("Use --model <path.gguf> to load a model");
    }

    let executor = Arc::new(Mutex::new(executor));

    // Build and run daemon (spawns network, health, API tasks)
    let daemon = Daemon::new(config, identity, db, executor);
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
