//! SwarmLLM — decentralized P2P LLM inference.
//!
//! # Public API surface
//!
//! Only these modules are part of the stable, semver-respected API:
//!
//! - [`api`] — HTTP server router + middleware (used by integration tests).
//! - [`config`] — `Config`, `UpdateConfig`, `OperationalParams`, env-var
//!   resolution. The user-facing knobs.
//! - [`error`] — `SwarmError`, `ApiError`. Error variants are stable.
//! - [`types`] — wire types (`NodeId`, `ModelId`, `ShardId`, etc.).
//! - [`update`] — `UpdateChecker`, `UpdateState`, the binary auto-update
//!   surface. Used by both the daemon and the standalone `swarmllm update`
//!   CLI.
//!
//! # Internal modules
//!
//! Every other `pub mod` below is `#[doc(hidden)]` and exposed only because
//! the integration test crate (and a few CLI subcommands) reach into them.
//! Treat their contents as unstable: they may change in any release.
//! Downstream consumers should not depend on them.

/// mimalloc for every binary built from this crate. The CPU inference path
/// allocates a fresh buffer per tensor op, so the allocator is on the hot path
/// of every layer of every token; measured on llama-3.2-3b decode (see
/// CHANGELOG v0.3.112) against glibc malloc.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod api;
pub mod config;
#[doc(hidden)]
pub mod credit;
#[doc(hidden)]
pub mod crypto;
#[doc(hidden)]
pub mod daemon;
pub mod error;
#[doc(hidden)]
pub mod health;
#[doc(hidden)]
pub mod http;
#[doc(hidden)]
pub mod identity;
#[doc(hidden)]
pub mod inference;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod network;
#[doc(hidden)]
pub mod pool;
#[doc(hidden)]
pub mod storage;
pub mod types;
pub mod update;
pub mod update_restart;

/// Verbosity the daemon was started with (`-v` count), so spawned
/// `model-worker` subprocesses can be given the same.
///
/// Without this a worker fell back to the config file's `logging.level` and
/// emitted INFO only — so running the daemon with `-v` produced no extra output
/// from the process where inference actually happens, and a `debug!` added there
/// while chasing a problem never appeared at all. That is a bad place to be
/// blind: it is the hot path.
pub static DAEMON_VERBOSITY: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The path to the currently running binary, with Linux's `" (deleted)"` marker
/// resolved away.
///
/// `std::env::current_exe` reads `/proc/self/exe`, which the kernel renders as
/// `"/usr/bin/swarmllm (deleted)"` once that file has been unlinked — which is
/// precisely what replacing a binary looks like: an atomic self-update (write
/// alongside, rename over the top) and a package upgrade both do it. Rust
/// returns the decorated string verbatim, and every consumer then works with a
/// path that does not exist.
///
/// The damage is not theoretical. A node whose binary had been swapped
/// underneath it failed **every** inference with `spawn worker: No such file or
/// directory` — while continuing to advertise its shards, so the swarm kept
/// routing work to a node that could not do any (observed 2026-07-27, 4 of 4
/// requests). The self-updater is hit too: it derives the download target from
/// this path, so it would have written a file called `swarmllm (deleted).tmp`
/// and left the real binary untouched.
///
/// Stripping the marker gives the path the binary occupies *now*, which is the
/// new version — the right thing to spawn, and better than failing outright.
/// The daemon still wants restarting, so say so once.
pub fn current_exe_path() -> std::io::Result<std::path::PathBuf> {
    let exe = std::env::current_exe()?;
    Ok(resolve_deleted_exe(exe, |p| p.exists()))
}

/// Marker the kernel appends to `/proc/self/exe` for an unlinked binary.
const DELETED_EXE_SUFFIX: &str = " (deleted)";

/// Pure half of [`current_exe_path`] — `exists` is injected so the rule can be
/// tested without a filesystem.
fn resolve_deleted_exe(
    exe: std::path::PathBuf,
    exists: impl Fn(&std::path::Path) -> bool,
) -> std::path::PathBuf {
    let Some(stripped) = exe
        .to_str()
        .and_then(|s| s.strip_suffix(DELETED_EXE_SUFFIX))
    else {
        return exe;
    };
    let candidate = std::path::PathBuf::from(stripped);
    // Only trust the stripped path if something is actually there. A binary
    // genuinely deleted (not replaced) must keep failing rather than silently
    // resolving to nothing — and a file whose real name ends in " (deleted)"
    // is left alone.
    if !exists(&candidate) {
        return exe;
    }
    tracing::warn!(
        path = %candidate.display(),
        "This binary was replaced while running — restart the daemon to finish \
         picking up the new version"
    );
    candidate
}

#[cfg(test)]
mod exe_path_tests {
    use super::resolve_deleted_exe;
    use std::path::{Path, PathBuf};

    #[test]
    fn replaced_binary_resolves_to_the_new_file() {
        let got = resolve_deleted_exe(PathBuf::from("/usr/bin/swarmllm (deleted)"), |_| true);
        assert_eq!(got, Path::new("/usr/bin/swarmllm"));
    }

    #[test]
    fn genuinely_deleted_binary_is_not_invented() {
        // Nothing at the stripped path: keep the decorated path so the caller
        // still fails, rather than pretending a binary exists.
        let original = PathBuf::from("/usr/bin/swarmllm (deleted)");
        let got = resolve_deleted_exe(original.clone(), |_| false);
        assert_eq!(got, original);
    }

    #[test]
    fn an_ordinary_path_is_untouched() {
        let original = PathBuf::from("/usr/bin/swarmllm");
        let got = resolve_deleted_exe(original.clone(), |_| panic!("must not probe"));
        assert_eq!(got, original);
    }

    /// A file whose real name happens to end in the marker keeps working.
    #[test]
    fn a_file_actually_named_deleted_still_resolves_to_itself() {
        let original = PathBuf::from("/tmp/swarmllm (deleted)");
        // Stripped path is absent, so the real file wins.
        let got = resolve_deleted_exe(original.clone(), |p| p != Path::new("/tmp/swarmllm"));
        assert_eq!(got, original);
    }
}
