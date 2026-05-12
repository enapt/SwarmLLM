//! HfWatcher — continuously polls HuggingFace's trending GGUF models so the
//! wishlist stays warm without waiting for a user to manually browse.
//!
//! Why it exists. Before R112 the swarm only learned about a model when
//! someone (a user, a peer's gossip) introduced it. Auto-manage was
//! reactive — silent until something arrived. With HfWatcher we run a
//! lightweight background task that pulls a snapshot of the HF "trending
//! GGUF" feed every hour, joins it against our local registry + trust
//! gates, and seeds the wishlist with `Discovered` candidates the user
//! can opt into.
//!
//! Design notes
//! - Cadence: 1 hour between polls. HF rate limits anonymous traffic, and
//!   trending data doesn't move faster than this anyway. Adjustable via
//!   `HF_WATCHER_INTERVAL_SECS`.
//! - Errors are non-fatal: a failed fetch leaves the prior snapshot in
//!   place. We never panic the daemon over an HF outage.
//! - The watcher is opt-in via the `auto_manage.hf_watcher_enabled`
//!   config flag (default `true`). Operators with bandwidth concerns or
//!   air-gapped deployments can disable it.
//! - We deliberately do NOT attempt to download anything ourselves — the
//!   watcher's only job is to update `state.models.hf_trending_cache`.
//!   The wishlist computation in R111 reads from that cache and the
//!   wishlist is what decides whether auto-manage acts.
//! - Trust promotion: a model with HF downloads >= `MIN_DOWNLOADS_FOR_TRUST`
//!   AND age >= `MIN_AGE_FOR_TRUST_HOURS` gets bumped to `DemandVerified`
//!   in our local trust map. The age gate stops a flash download-pump
//!   from gaming the auto-manage trust gate.
//!
//! R112.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::daemon::SharedState;
use crate::error::SwarmError;

/// Default poll interval — every hour. HF rate-limits anonymous endpoints
/// at roughly 1k req/h, but we only do ~24 fetches per day, so we're far
/// under the cap. Trending lists don't move faster than this anyway.
const HF_WATCHER_INTERVAL_SECS: u64 = 3600;

/// Initial delay before the first poll. Lets the daemon finish startup
/// (peer discovery, manifest hydration) before we hit HF.
const HF_WATCHER_STARTUP_DELAY_SECS: u64 = 30;

/// On error, exponential back-off doubles the wait up to this cap.
const HF_WATCHER_MAX_BACKOFF_SECS: u64 = 7200;

/// Initial back-off after the first error. Smaller than the success
/// interval so a transient blip retries within minutes rather than
/// jumping straight to MAX_BACKOFF.
const HF_WATCHER_BASE_BACKOFF_SECS: u64 = 300;

/// HF endpoint we query. `library=gguf` filters to GGUF-compatible repos
/// (where llama.cpp / our split path will work); `sort=downloads`
/// dir=-1 ranks by all-time downloads which is the most stable signal
/// for "is this model real and used". Trending is more volatile and we
/// don't want to chase fashions.
const HF_API_URL: &str =
    "https://huggingface.co/api/models?library=gguf&sort=downloads&direction=-1&limit=100&full=true";

/// Cap on how many entries we keep — match wishlist's MAX_WISHLIST_ENTRIES
/// so the watcher can fully cover the wishlist without overflow.
const MAX_TRENDING_ENTRIES: usize = 100;

/// Trust-promotion thresholds. Both must be met to lift a `Discovered`
/// model to `DemandVerified` (which lets auto-manage act on it).
const MIN_DOWNLOADS_FOR_TRUST: u64 = 100_000;
const MIN_AGE_FOR_TRUST_HOURS: i64 = 24;

/// Internal `poll_once` error type — separated so the run loop can
/// honour HF's `Retry-After` on 429 responses instead of applying the
/// generic exponential back-off.
enum PollError {
    RateLimited { retry_after_secs: u64 },
    Other(SwarmError),
}

impl std::fmt::Display for PollError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PollError::RateLimited { retry_after_secs } => {
                write!(f, "rate-limited (retry-after {retry_after_secs}s)")
            }
            PollError::Other(e) => write!(f, "{e}"),
        }
    }
}

/// One model from the HF /api/models response — only the fields we use.
#[derive(Clone, Debug, Deserialize)]
struct HfApiModel {
    #[serde(rename = "id")]
    repo_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    /// ISO-8601 timestamp of repo creation. Used in the trust-promotion
    /// age gate to defeat download-pump attacks against newly-published
    /// models.
    #[serde(rename = "createdAt", default)]
    created_at: Option<String>,
    /// HuggingFace pipeline tag — `text-generation`, `text-to-image`, etc.
    /// We filter to text-gen + a couple of friendly aliases.
    #[serde(rename = "pipeline_tag", default)]
    pipeline_tag: Option<String>,
    /// Free-form tags we use for capability inference (chat / code /
    /// vision / multilingual / reasoning).
    #[serde(default)]
    tags: Vec<String>,
}

/// Cached snapshot of the HF trending feed. Stored on `state.models`
/// (added below) and read by the wishlist scorer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HfTrendingSnapshot {
    pub entries: Vec<HfTrendingEntry>,
    /// Unix seconds of the last successful poll (0 = never).
    pub fetched_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfTrendingEntry {
    pub repo_id: String,
    pub downloads: u64,
    pub likes: u64,
    /// Best-effort capability tags inferred from HF metadata. Each is a
    /// stable token like `chat` / `code` / `vision` / `multilingual` /
    /// `reasoning` so the frontend can localise + filter.
    pub task_tags: Vec<String>,
    /// Created-at parsed as Unix seconds (0 = unknown).
    pub created_at_secs: i64,
}

/// HfWatcher background task. Owns its own HTTP client; keeps a
/// shutdown receiver so the daemon's graceful-stop path terminates it.
pub struct HfWatcher {
    shared_state: Arc<SharedState>,
    shutdown_rx: watch::Receiver<bool>,
    interval: Duration,
    client: reqwest::Client,
}

impl HfWatcher {
    pub fn new(shared_state: Arc<SharedState>, shutdown_rx: watch::Receiver<bool>) -> Self {
        let client = reqwest::Client::builder()
            // Keep the timeout tight — a stuck HF call shouldn't hold
            // up the watcher's shutdown path.
            .timeout(Duration::from_secs(30))
            .user_agent(format!(
                "SwarmLLM/{} (+swarmllm.app)",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            shared_state,
            shutdown_rx,
            interval: Duration::from_secs(HF_WATCHER_INTERVAL_SECS),
            client,
        }
    }

    /// Run the watcher loop. Returns Ok when shutdown_rx fires.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        // Honour the kill-switch.
        if !self.shared_state.config.auto_manage.hf_watcher_enabled {
            tracing::info!("HfWatcher disabled by config — exiting");
            return Ok(());
        }

        // Initial delay so the daemon's noisy startup phase completes
        // before our first network hit.
        let startup_sleep = Duration::from_secs(HF_WATCHER_STARTUP_DELAY_SECS);
        tokio::select! {
            _ = tokio::time::sleep(startup_sleep) => {},
            _ = self.shutdown_rx.changed() => {
                if *self.shutdown_rx.borrow() { return Ok(()); }
            }
        }

        let mut backoff = self.interval;
        loop {
            match self.poll_once().await {
                Ok(count) => {
                    tracing::info!(count, "HfWatcher: snapshot refreshed");
                    backoff = self.interval; // reset on success
                }
                Err(PollError::RateLimited { retry_after_secs }) => {
                    let secs = retry_after_secs.clamp(60, HF_WATCHER_MAX_BACKOFF_SECS);
                    tracing::warn!(
                        retry_after = secs,
                        "HfWatcher rate-limited; honouring Retry-After"
                    );
                    backoff = Duration::from_secs(secs);
                }
                Err(PollError::Other(e)) => {
                    // Start fresh from BASE_BACKOFF instead of from
                    // self.interval, so the first retry happens in minutes
                    // (not 2h — the cap is hit immediately if we double the
                    // success interval).
                    let prev = if backoff == self.interval {
                        Duration::from_secs(HF_WATCHER_BASE_BACKOFF_SECS)
                    } else {
                        backoff
                    };
                    let next = prev
                        .saturating_mul(2)
                        .min(Duration::from_secs(HF_WATCHER_MAX_BACKOFF_SECS));
                    tracing::warn!(error = %e, next_secs = next.as_secs(), "HfWatcher poll failed; backing off");
                    backoff = next;
                }
            }

            tokio::select! {
                _ = tokio::time::sleep(backoff) => {},
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("HfWatcher shutting down");
                        return Ok(());
                    }
                }
            }
        }
    }

    async fn poll_once(&self) -> Result<usize, PollError> {
        let resp =
            self.client.get(HF_API_URL).send().await.map_err(|e| {
                PollError::Other(SwarmError::Internal(format!("HfWatcher fetch: {e}")))
            })?;
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after_secs = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(crate::model::huggingface::parse_retry_after)
                .unwrap_or(HF_WATCHER_BASE_BACKOFF_SECS);
            return Err(PollError::RateLimited { retry_after_secs });
        }
        if !resp.status().is_success() {
            return Err(PollError::Other(SwarmError::Internal(format!(
                "HfWatcher status {}",
                resp.status()
            ))));
        }
        // We bound the body size — HF /api/models with limit=100&full=true
        // returns ~1-2 MB. 4 MB is comfortable headroom.
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| PollError::Other(SwarmError::Internal(format!("HfWatcher body: {e}"))))?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err(PollError::Other(SwarmError::Internal(format!(
                "HfWatcher response too large ({} bytes)",
                bytes.len()
            ))));
        }
        let raw: Vec<HfApiModel> = serde_json::from_slice(&bytes).map_err(|e| {
            PollError::Other(SwarmError::Internal(format!("HfWatcher decode: {e}")))
        })?;

        let mut entries: Vec<HfTrendingEntry> =
            Vec::with_capacity(raw.len().min(MAX_TRENDING_ENTRIES));
        for m in raw.iter().take(MAX_TRENDING_ENTRIES) {
            // Filter to text-generation + chat models — image / audio
            // GGUFs aren't currently runnable through our split path.
            let is_text = match m.pipeline_tag.as_deref() {
                Some("text-generation") | Some("text2text-generation") | Some("conversational") => {
                    true
                }
                _ => m
                    .tags
                    .iter()
                    .any(|t| t == "text-generation" || t == "conversational"),
            };
            if !is_text {
                continue;
            }
            // Defense-in-depth: drop malformed repo_ids (path-traversal, bad
            // chars). The trending snapshot is exposed via REST and feeds
            // wishlist scoring; gotcha #142 covers the same gate elsewhere.
            if crate::model::huggingface::validate_hf_repo_id(&m.repo_id).is_err() {
                tracing::debug!(repo_id = %m.repo_id, "HfWatcher: skipping malformed repo_id");
                continue;
            }
            let task_tags = infer_task_tags(&m.tags, m.pipeline_tag.as_deref());
            let created_at_secs = m
                .created_at
                .as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc).timestamp())
                .unwrap_or(0);
            entries.push(HfTrendingEntry {
                repo_id: m.repo_id.clone(),
                downloads: m.downloads,
                likes: m.likes,
                task_tags,
                created_at_secs,
            });
        }

        // Promote any local model with a matching HF entry above thresholds.
        promote_trust_for_trending(&self.shared_state, &entries);

        // Publish the new snapshot.
        let snapshot = HfTrendingSnapshot {
            entries,
            fetched_at: chrono::Utc::now().timestamp(),
        };
        let count = snapshot.entries.len();
        self.shared_state
            .models
            .hf_trending_cache
            .store(Arc::new(snapshot));
        Ok(count)
    }
}

/// Infer a small set of capability tags from HF metadata. Stable tokens
/// the frontend can localise via `wishlist.task.<tag>` keys (added with
/// the rest of the wishlist i18n; this returns the bare token).
pub fn infer_task_tags(tags: &[String], pipeline_tag: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    let pl = pipeline_tag.unwrap_or("").to_lowercase();

    let has = |needles: &[&str]| -> bool {
        lower.iter().any(|t| needles.iter().any(|n| t.contains(n)))
            || needles.iter().any(|n| pl.contains(n))
    };

    if has(&["chat", "instruct", "instruction"]) {
        out.push("chat".to_string());
    }
    if has(&["code", "coder", "starcoder", "deepseek-coder"]) {
        out.push("code".to_string());
    }
    if has(&["vision", "vlm", "multimodal", "image-text"]) {
        out.push("vision".to_string());
    }
    if has(&["multilingual", "translation"]) {
        out.push("multilingual".to_string());
    }
    if has(&["reasoning", "math", "logic"]) {
        out.push("reasoning".to_string());
    }
    // Default fallback — always at least one tag so the UI has something
    // to render.
    if out.is_empty() {
        out.push("chat".to_string());
    }
    out
}

/// Walk the local model_trust map and promote any `Discovered` entry
/// whose HF repo_id appears in the trending list with downloads above
/// the threshold AND age above the gate. Idempotent — re-running this
/// is safe.
fn promote_trust_for_trending(state: &SharedState, entries: &[HfTrendingEntry]) {
    use std::collections::HashMap;

    if entries.is_empty() {
        return;
    }
    // Index trending by repo_id for O(1) lookup. Repo ids are
    // case-sensitive on HF.
    let by_repo: HashMap<&str, &HfTrendingEntry> =
        entries.iter().map(|e| (e.repo_id.as_str(), e)).collect();

    let now = chrono::Utc::now().timestamp();
    let age_threshold = MIN_AGE_FOR_TRUST_HOURS * 3600;

    for hf in state.models.hf_sources.iter() {
        let model_id = hf.key().clone();
        let repo_id = hf.value().repo_id.as_str();
        let Some(entry) = by_repo.get(repo_id) else {
            continue;
        };
        if entry.downloads < MIN_DOWNLOADS_FOR_TRUST {
            continue;
        }
        if entry.created_at_secs > 0 && (now - entry.created_at_secs) < age_threshold {
            continue;
        }
        // Promote — only if currently Discovered (don't override an
        // explicit user pin, and don't downgrade higher trust).
        let mut upgraded = false;
        state
            .models
            .model_trust
            .entry(model_id.clone())
            .and_modify(|t| {
                if matches!(t.trust_level, crate::types::ModelTrustLevel::Discovered) {
                    t.trust_level = crate::types::ModelTrustLevel::DemandVerified;
                    upgraded = true;
                }
            })
            .or_insert_with(|| {
                upgraded = true;
                crate::types::ModelTrustInfo {
                    trust_level: crate::types::ModelTrustLevel::DemandVerified,
                    first_seen: chrono::Utc::now(),
                    total_requests: 0,
                    pinned_by_user: false,
                    last_request_at: None,
                }
            });
        if upgraded {
            tracing::info!(
                model = %model_id,
                repo = %repo_id,
                downloads = entry.downloads,
                "HfWatcher: promoted to DemandVerified"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_tag_inference_basics() {
        let tags = vec!["text-generation".into(), "instruct".into()];
        let out = infer_task_tags(&tags, Some("text-generation"));
        assert!(out.contains(&"chat".to_string()));
    }

    #[test]
    fn task_tag_code_detection() {
        let tags = vec!["code".into(), "starcoder".into()];
        let out = infer_task_tags(&tags, Some("text-generation"));
        assert!(out.contains(&"code".to_string()));
    }

    #[test]
    fn task_tag_default_fallback() {
        let tags: Vec<String> = vec![];
        let out = infer_task_tags(&tags, Some("text-generation"));
        assert!(out.contains(&"chat".to_string()));
    }

    #[test]
    fn task_tag_vision_detection() {
        let tags = vec!["multimodal".into(), "image-text".into()];
        let out = infer_task_tags(&tags, Some("text-generation"));
        assert!(out.contains(&"vision".to_string()));
    }

    #[test]
    fn snapshot_serialises() {
        let snap = HfTrendingSnapshot::default();
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["fetched_at"], 0);
        assert!(json["entries"].is_array());
    }
}
