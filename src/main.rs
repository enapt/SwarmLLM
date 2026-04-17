use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod cli;

use cli::pool::PoolAction;

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
    /// Interactive terminal chat with a running daemon
    Chat {
        /// Model to use (auto-selects first available if omitted)
        #[arg(long)]
        model: Option<String>,
        /// Maximum tokens per response
        #[arg(long, default_value = "1024")]
        max_tokens: u32,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
    /// List connected peers with latency and trust scores
    Peers {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
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
    /// [Internal] Run a model worker subprocess (managed by daemon, not for direct use)
    #[command(hide = true)]
    ModelWorker {
        /// Unix socket path to connect to the daemon
        #[arg(long)]
        socket: PathBuf,
        /// Data directory
        #[arg(long)]
        data_dir: PathBuf,
        /// Comma-separated shard indices to load (e.g. "0,1,7"). If omitted, loads all on-disk shards.
        #[arg(long)]
        shard_window: Option<String>,
        /// KV-cache session TTL in seconds (default 600)
        #[arg(long, default_value = "600")]
        kv_cache_ttl: u64,
    },
    /// Device pool management (combine credits across your devices)
    Pool {
        #[command(subcommand)]
        action: PoolAction,
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

    let no_update_check = matches!(
        &cli.command,
        Some(Commands::Run {
            no_update_check: true
        })
    );
    let command = cli.command.take().unwrap_or(Commands::Run {
        no_update_check: false,
    });

    let resolve_data_dir = |cli_data_dir: &Option<PathBuf>| -> PathBuf {
        swarmllm::config::resolve_data_dir(cli_data_dir.as_deref())
    };

    match command {
        Commands::Run { .. } => {
            let args = cli::run::DaemonArgs {
                config: cli.config,
                port: cli.port,
                data_dir: cli.data_dir,
                model: cli.model,
                gpu_layers: cli.gpu_layers,
                bootstrap: cli.bootstrap,
                shards: cli.shards,
                no_update_check,
            };
            cli::run::run_daemon(args).await
        }
        Commands::Version => {
            println!("swarmllm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Commands::Status => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::status::query_status(port, &data_dir).await
        }
        Commands::Chat {
            model,
            max_tokens,
            temperature,
        } => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::chat::run_chat(port, &data_dir, model, max_tokens, temperature).await
        }
        Commands::Peers { json } => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::peers::query_peers(port, &data_dir, json).await
        }
        Commands::ModelWorker {
            socket,
            data_dir,
            shard_window,
            kv_cache_ttl,
        } => {
            let window: Option<Vec<u32>> = shard_window.map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse::<u32>().ok())
                    .collect()
            });
            swarmllm::inference::model_worker::run_worker(socket, data_dir, window, kv_cache_ttl)
                .await;
            Ok(())
        }
        Commands::Pool { action } => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::pool::run_pool_command(port, &data_dir, action).await
        }
        Commands::Update { check_only } => cli::update::run_update_command(check_only).await,
        Commands::TestSplit { max_tokens, prompt } => {
            cli::split_test::test_split_inference(cli.model.clone(), max_tokens, &prompt).await
        }
        Commands::Bench {
            max_tokens,
            iterations,
            concurrency,
            prompt,
            json,
        } => {
            let port = cli.port.unwrap_or(8800);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::bench::run_bench(
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

    // Respect NO_COLOR (https://no-color.org/) and also disable ANSI
    // when stdout is not a terminal (piped/redirected to file)
    let use_ansi =
        std::env::var("NO_COLOR").is_err() && std::io::IsTerminal::is_terminal(&std::io::stdout());

    tracing_subscriber::fmt()
        .with_env_filter(&filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_ansi(use_ansi)
        .init();
}
