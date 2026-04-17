//! Subcommand implementations for the SwarmLLM CLI.
//!
//! Each submodule owns one top-level subcommand (or family of subcommands).
//! The `main.rs` binary parses args via clap, then dispatches to the public
//! entry point in the matching submodule.

pub mod bench;
pub mod chat;
pub mod peers;
pub mod pool;
pub mod run;
pub mod split_test;
pub mod status;
pub mod update;

/// Read the API key from the data dir. Shared helper used by CLI commands
/// that talk to a running daemon over HTTP.
pub(crate) fn read_api_key(data_dir: &std::path::Path) -> Option<String> {
    let key_path = data_dir.join("api_key");
    std::fs::read_to_string(key_path)
        .ok()
        .map(|s| s.trim().to_string())
}
