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
