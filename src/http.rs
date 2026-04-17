//! Shared reqwest client factory.
//!
//! Centralizes the `Client::builder()...build().unwrap_or_else(Client::new)`
//! pattern so fallback behavior and default configuration stay consistent.

use std::time::Duration;

/// Build a reqwest client with caller-supplied configuration.
///
/// Applies the given closure to a fresh `ClientBuilder`, then falls back to
/// the default client if the builder fails (rare — usually a TLS backend
/// initialization error). Prefer this over direct `reqwest::Client::builder()`
/// so the fallback and defaults stay uniform across the codebase.
pub fn build_client<F>(configure: F) -> reqwest::Client
where
    F: FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
{
    configure(reqwest::Client::builder())
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Convenience: client with only a total-request timeout.
pub fn client_with_timeout(timeout: Duration) -> reqwest::Client {
    build_client(|b| b.timeout(timeout))
}
