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
    gpu_layers: Option<i32>,

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
        /// Run as a bootstrap/relay anchor: no inference, no models, no
        /// HuggingFace/shard downloads; dashboard bound to loopback only.
        /// The node still relays + helps peers discover each other.
        #[arg(long)]
        anchor: bool,
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
        /// IPC socket name to connect to the daemon.
        /// On Unix this is a filesystem path (AF_UNIX); on Windows it is a
        /// named-pipe namespace name. The daemon chooses the right form.
        #[arg(long)]
        socket: String,
        /// Data directory
        #[arg(long)]
        data_dir: PathBuf,
        /// Comma-separated shard indices to load (e.g. "0,1,7"). If omitted, loads all on-disk shards.
        #[arg(long)]
        shard_window: Option<String>,
        /// Device placement: -1 = auto (GPU when available), 0 = CPU only, >0 = GPU.
        #[arg(long, default_value = "-1", allow_negative_numbers = true)]
        gpu_layers: i32,
        /// KV-cache session TTL in seconds (default 600)
        #[arg(long, default_value = "600")]
        kv_cache_ttl: u64,
        /// Enable cross-request prefix KV-cache in the worker (default true)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        prefix_cache_enabled: bool,
        /// Maximum cached prefix snapshots retained per model (default 16)
        #[arg(long, default_value = "16")]
        prefix_cache_max_entries: u32,
        /// Prompts longer than this (tokens) are not inserted (default 8192)
        #[arg(long, default_value = "8192")]
        prefix_cache_max_prompt_tokens: u32,
        /// Block-alignment granularity for multi-point inserts (default 64)
        #[arg(long, default_value = "64")]
        prefix_cache_block_tokens: u32,
        /// Minimum prefix tokens for cache to engage (default 32)
        #[arg(long, default_value = "32")]
        prefix_cache_min_tokens: u32,
        /// Enable SWIFT self-speculative decoding inside handle_generate
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        swift_self_speculative: bool,
        /// Number of warmup tokens before SWIFT engages
        #[arg(long, default_value = "32")]
        swift_calibration_tokens: u32,
        /// Number of draft tokens proposed per SWIFT verification round
        #[arg(long, default_value = "4")]
        swift_gamma: u32,
        /// Fraction of layers to skip in SWIFT draft pass
        #[arg(long, default_value = "0.45")]
        swift_skip_ratio: f32,
        /// Force every attention call through standard_attention (matmul)
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        force_standard_attn: bool,
        /// Cap GGUF context_length when allocating KV cache. 0 = use GGUF value.
        #[arg(long, default_value = "0")]
        max_seq_len_override: u32,
        /// Quantize intermediate-segment hidden state activations to Q8_0
        /// before returning them to the daemon. Compresses ~3.76× with
        /// negligible quality loss. Off by default; receivers always
        /// auto-dispatch on the dtype tag.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        activation_compression: bool,
        /// Enable Item 7 BatchGenerate: multiple concurrent `Generate`
        /// requests interleave through one `forward_batch` per decode tick.
        /// Off → each `Generate` runs sequentially through the worker.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        batch_generate: bool,
        /// Maximum number of concurrent decode slots when `batch_generate` is
        /// on. Caps the BatchGenerate slot table; new admissions beyond this
        /// fall through to the sequential `handle_generate` path.
        #[arg(long, default_value = "8")]
        batch_generate_max_slots: u32,
        /// Item 7 Phase 2: Sarathi-style chunked prefill chunk size (in
        /// prompt tokens). Per decode tick each Prefilling slot advances by
        /// up to this many tokens before its first decode token is sampled.
        #[arg(long, default_value = "128")]
        prefill_chunk_tokens: u32,
        /// Item 7 Phase 4: fuse concurrent same-shape Prefilling slots into
        /// one `forward_batch` call inside `step_decode_pool`'s Phase A.
        /// Off → Phase A runs singleton forwards per slot (useful for A/B).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        batched_prefill_forward: bool,
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
        /// Use streaming chat completions (`stream: true`) and report
        /// time-to-first-token (TTFT) per request. Streaming TTFT is the
        /// signal that captures Item 7 Phase 2 wins (chunked prefill
        /// admit cost) — non-streaming bench rolls prefill + decode into
        /// one total time and hides the difference.
        #[arg(long, default_value_t = false)]
        stream: bool,
        /// Force a specific model id (matches `/v1/models` data[].id).
        /// When unset, the bench uses the first listed model — which may
        /// be the wrong one if multiple are registered. Long form is
        /// `--model-id` to avoid clashing with the top-level `--model`
        /// flag (which expects a file path).
        #[arg(long = "model-id")]
        model_id: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // SEC: Load .env into process environment BEFORE the Tokio runtime spawns
    // worker threads. `std::env::set_var` is unsound in a multi-threaded
    // process (deprecated in Rust 1.81+) — racing a libc env reader from a
    // worker thread can crash glibc. Doing it here, on the main thread before
    // runtime construction, eliminates the race.
    {
        let data_dir = swarmllm::config::resolve_data_dir(cli.data_dir.as_deref());
        swarmllm::config::load_dotenv(&data_dir);
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async_main(cli))
}

async fn async_main(mut cli: Cli) -> anyhow::Result<()> {
    init_tracing(cli.verbose);

    let no_update_check = matches!(
        &cli.command,
        Some(Commands::Run {
            no_update_check: true,
            ..
        })
    );
    let anchor = matches!(&cli.command, Some(Commands::Run { anchor: true, .. }));
    let command = cli.command.take().unwrap_or(Commands::Run {
        no_update_check: false,
        anchor: false,
    });

    let resolve_data_dir = |cli_data_dir: &Option<PathBuf>| -> PathBuf {
        swarmllm::config::resolve_data_dir(cli_data_dir.as_deref())
    };

    // CLI client port: CLI flag > SWARMLLM_NODE_LISTEN_PORT > 8800.
    // Without env-var support every client subcommand (status/chat/peers/
    // bench/pool) silently connected to 8800 even when the daemon was
    // started on a different port via env var — surprising the user who
    // followed the documented "env > default" precedence.
    let resolve_client_port = |cli_port: Option<u16>| -> u16 {
        if let Some(p) = cli_port {
            return p;
        }
        if let Ok(s) = std::env::var("SWARMLLM_NODE_LISTEN_PORT") {
            if let Ok(p) = s.parse::<u16>() {
                if p > 0 {
                    return p;
                }
            }
        }
        8800
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
                anchor,
            };
            cli::run::run_daemon(args).await
        }
        Commands::Version => {
            println!("swarmllm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Commands::Status => {
            let port = resolve_client_port(cli.port);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::status::query_status(port, &data_dir).await
        }
        Commands::Chat {
            model,
            max_tokens,
            temperature,
        } => {
            let port = resolve_client_port(cli.port);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::chat::run_chat(port, &data_dir, model, max_tokens, temperature).await
        }
        Commands::Peers { json } => {
            let port = resolve_client_port(cli.port);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::peers::query_peers(port, &data_dir, json).await
        }
        Commands::ModelWorker {
            socket,
            data_dir,
            shard_window,
            gpu_layers,
            kv_cache_ttl,
            prefix_cache_enabled,
            prefix_cache_max_entries,
            prefix_cache_max_prompt_tokens,
            prefix_cache_block_tokens,
            prefix_cache_min_tokens,
            swift_self_speculative,
            swift_calibration_tokens,
            swift_gamma,
            swift_skip_ratio,
            force_standard_attn,
            max_seq_len_override,
            activation_compression,
            batch_generate,
            batch_generate_max_slots,
            prefill_chunk_tokens,
            batched_prefill_forward,
        } => {
            let window: Option<Vec<u32>> = shard_window.map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse::<u32>().ok())
                    .collect()
            });
            let prefix_cfg = swarmllm::inference::model_worker::PrefixCacheConfig {
                enabled: prefix_cache_enabled,
                max_entries: prefix_cache_max_entries as usize,
                max_prompt_tokens: prefix_cache_max_prompt_tokens as usize,
                block_tokens: prefix_cache_block_tokens as usize,
                min_tokens: prefix_cache_min_tokens as usize,
            };
            let swift_cfg = swarmllm::inference::swift::SwiftConfig {
                enabled: swift_self_speculative,
                calibration_tokens: swift_calibration_tokens,
                gamma: swift_gamma,
                skip_ratio: swift_skip_ratio,
            };
            let max_seq_override = if max_seq_len_override == 0 {
                None
            } else {
                Some(max_seq_len_override as usize)
            };
            let options = swarmllm::inference::model_worker::WorkerOptions {
                force_standard_attn,
                max_seq_len_override: max_seq_override,
                activation_compression,
                batch_generate,
                batch_generate_max_slots,
                prefill_chunk_tokens,
                batched_prefill_forward,
                gpu_layers,
            };
            swarmllm::inference::model_worker::run_worker(
                socket,
                data_dir,
                window,
                kv_cache_ttl,
                prefix_cfg,
                swift_cfg,
                options,
            )
            .await;
            Ok(())
        }
        Commands::Pool { action } => {
            let port = resolve_client_port(cli.port);
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
            stream,
            model_id,
        } => {
            let port = resolve_client_port(cli.port);
            let data_dir = resolve_data_dir(&cli.data_dir);
            cli::bench::run_bench(
                port,
                &data_dir,
                max_tokens,
                iterations,
                concurrency,
                &prompt,
                json,
                stream,
                model_id,
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
        // Read logging.level from the resolved data dir, NOT a hardcoded
        // default. resolve_data_dir() picks up SWARMLLM_NODE_DATA_DIR so a
        // multi-node setup gets each node's own log level. Without this,
        // node2 silently inherited node1's config (or fell through to "info").
        // Note: this runs before CLI parsing, so a `--data-dir` flag isn't
        // available yet — we accept that limitation for the bootstrap log
        // filter; the env-var case (which is the common multi-node testing
        // pattern) is what matters here.
        let config_level =
            std::fs::read_to_string(swarmllm::config::resolve_data_dir(None).join("config.toml"))
                .ok()
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
