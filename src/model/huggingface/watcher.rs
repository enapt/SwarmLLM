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
///
/// Two tiers: trusted curator publishers (`TRUSTED_HF_PUBLISHERS`) get
/// a 10× lower download threshold because their releases are vetted +
/// the publisher reputation acts as anti-gaming. Random publishers
/// keep the original 100k floor. The 24h age gate applies to BOTH
/// tiers — a freshly-published repo can still be a download-pump even
/// from a trusted curator's account if compromised.
const MIN_DOWNLOADS_FOR_TRUST: u64 = 100_000;
const MIN_DOWNLOADS_FOR_TRUST_TRUSTED: u64 = 10_000;
const MIN_AGE_FOR_TRUST_HOURS: i64 = 24;

/// How many origins to ask about per tick for models that are NOT in the
/// trending feed. Bounded because this is one HTTP request each against a
/// third party; the watcher runs hourly, so a handful per tick clears any
/// realistic backlog within a few hours while never looking like a scraper.
const MAX_ORIGIN_TRUST_PROBES_PER_TICK: usize = 8;

/// Curator allowlist: HF publishers whose releases are known-good
/// quantisations / official model weights. Maintainers earn this slot
/// through track record (years of clean releases) — adding a name here
/// loosens the auto-promotion floor 10× for any GGUF they publish.
///
/// Matched against the case-insensitive `<publisher>/<repo>` prefix on
/// the HF repo_id. Update sparingly; each entry is a trust delegation.
const TRUSTED_HF_PUBLISHERS: &[&str] = &[
    // Official model authors
    "meta-llama",
    "mistralai",
    "Qwen",
    "google",
    "microsoft",
    "deepseek-ai",
    "HuggingFaceH4",
    "stabilityai",
    "tiiuae",
    "01-ai",
    "NousResearch",
    "allenai",
    "ibm-granite",
    "CohereForAI",
    // Curator / quantiser community heavyweights
    "bartowski",
    "TheBloke",
    "unsloth",
    "lmstudio-community",
    "MaziyarPanahi",
    "QuantFactory",
    "second-state",
];

/// Return the per-tier download threshold for a given HF repo_id.
/// Repo IDs look like `publisher/model-name`; the prefix before the
/// first `/` selects the tier.
pub(crate) fn min_downloads_for_repo(repo_id: &str) -> u64 {
    if is_trusted_publisher(repo_id) {
        MIN_DOWNLOADS_FOR_TRUST_TRUSTED
    } else {
        MIN_DOWNLOADS_FOR_TRUST
    }
}

/// Whether the given HF repo_id belongs to a trusted curator/publisher.
/// Used by the wishlist (Task #2) to mark Candidate entries that bypass
/// the user's "review before adopt" friction.
pub fn is_trusted_publisher(repo_id: &str) -> bool {
    let publisher = repo_id.split('/').next().unwrap_or("");
    TRUSTED_HF_PUBLISHERS
        .iter()
        .any(|p| p.eq_ignore_ascii_case(publisher))
}

/// R134: anti-gaming cooldown after an auto-promoted model decays back
/// to `Discovered` with zero real swarm requests. The wait grows with
/// each failed promotion attempt: `BASE_COOLDOWN_DAYS * failed_promotions`,
/// capped at `MAX_COOLDOWN_DAYS`. Defeats HF download-pump attacks
/// against newly-published models without permanently locking out
/// legitimate models that simply haven't been discovered yet.
const FAILED_PROMOTION_COOLDOWN_BASE_DAYS: i64 = 7;
const FAILED_PROMOTION_COOLDOWN_MAX_DAYS: i64 = 60;
/// Hard cap on automatic re-promotion attempts after the model has
/// repeatedly failed to attract real demand. Beyond this only a user pin
/// (via the admin API) can lift the trust level.
const MAX_AUTO_PROMOTION_FAILURES: u32 = 4;

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
        if !self.shared_state.cfg().auto_manage.hf_watcher_enabled {
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

        // Decide trust for every model we know an origin for — from this feed
        // where it appears, and by asking the origin directly where it does not.
        promote_trust_for_known_sources(&self.shared_state, &self.client, &entries).await;

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
fn infer_task_tags(tags: &[String], pipeline_tag: Option<&str>) -> Vec<String> {
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

/// R134: check whether HfWatcher is allowed to auto-promote a model
/// given its prior strike count. Returns `true` if either we've never
/// auto-promoted before, or the cooldown after the last attempt has
/// elapsed AND we haven't exceeded `MAX_AUTO_PROMOTION_FAILURES`.
pub(crate) fn should_auto_promote(trust: &crate::types::ModelTrustInfo) -> bool {
    if trust.pinned_by_user {
        return false; // user already has authority; don't touch
    }
    if !matches!(trust.trust_level, crate::types::ModelTrustLevel::Discovered) {
        return false; // only Discovered is eligible for auto-promotion
    }
    if trust.failed_promotions >= MAX_AUTO_PROMOTION_FAILURES {
        return false; // give up on this model — user pin only from here
    }
    if trust.failed_promotions == 0 {
        return true; // first attempt or never been auto-promoted
    }
    // Linear back-off: each strike extends the cooldown by BASE_DAYS
    // up to the MAX cap.
    let cooldown_days = (FAILED_PROMOTION_COOLDOWN_BASE_DAYS * trust.failed_promotions as i64)
        .min(FAILED_PROMOTION_COOLDOWN_MAX_DAYS);
    match trust.last_auto_promoted_at {
        Some(t) => (chrono::Utc::now() - t).num_days() >= cooldown_days,
        None => true,
    }
}

/// Walk the local model_trust map and promote any `Discovered` entry
/// whose HF repo_id appears in the trending list with downloads above
/// the threshold AND age above the gate. Idempotent — re-running this
/// is safe.
///
/// R134: skips models in cooldown after one or more failed promotions
/// (auto-promoted but the model never attracted real swarm requests
/// during the inactivity window — see `should_auto_promote`).
/// Decide trust for every model we know an origin for — from the trending
/// feed when it is there, and by **asking the origin directly** when it is not.
///
/// **Trending membership is a discovery signal, not a verification one, and
/// conflating the two left a hole.** Promotion used to require the repo to
/// appear in the current trending snapshot, so a model that someone
/// deliberately seeded onto the swarm could never be promoted, and every peer
/// on auto-manage therefore declined to help host it — `AutoShardManager: no
/// candidate shards to download` with budget to spare, for ever. Measured
/// 2026-08-31: a 16-shard model seeded with `peer_fair_share` sat at 1/16
/// shards with one holder while six peers with auto-manage enabled ignored it,
/// because a two-year-old 14B is not "trending" and never will be.
///
/// The thresholds are unchanged and are what makes this safe: a peer's
/// manifest cannot make this node download anything. It can only make us ASK
/// HuggingFace about a repo, and we act only on what HuggingFace itself
/// reports — downloads over the bar for that publisher, and old enough to
/// defeat a download pump. That is the same evidence the trending path uses,
/// obtained the same way, about a repo we were told about rather than one that
/// happened to be popular this week. Peer agreement still proves nothing; the
/// origin does (see `origin_verified`, gotcha #382).
async fn promote_trust_for_known_sources(
    state: &SharedState,
    client: &reqwest::Client,
    entries: &[HfTrendingEntry],
) {
    use std::collections::HashMap;

    let by_repo: HashMap<&str, &HfTrendingEntry> =
        entries.iter().map(|e| (e.repo_id.as_str(), e)).collect();

    // Snapshot first: holding a DashMap iterator across an await would keep a
    // shard of the map locked for the length of an HTTP request.
    let sources: Vec<(crate::types::ModelId, String)> = state
        .models
        .hf_sources
        .iter()
        .map(|e| (e.key().clone(), e.value().repo_id.clone()))
        .collect();

    let mut probes = 0usize;
    for (model_id, repo_id) in sources {
        if already_trusted_enough(state, &model_id) {
            continue;
        }
        let stats = match by_repo.get(repo_id.as_str()) {
            Some(e) => Some((e.downloads, e.created_at_secs)),
            None => {
                if probes >= MAX_ORIGIN_TRUST_PROBES_PER_TICK {
                    continue;
                }
                probes += 1;
                fetch_repo_stats(client, &repo_id).await
            }
        };
        if let Some((downloads, created_at_secs)) = stats {
            apply_trust_promotion(state, &model_id, &repo_id, downloads, created_at_secs);
        }
    }
}

/// Ask HuggingFace about one repo: downloads and creation time, the two
/// figures the trust thresholds are expressed in. `None` on any failure —
/// an unreachable origin is not evidence of anything, so the model simply
/// keeps whatever trust it already had.
async fn fetch_repo_stats(client: &reqwest::Client, repo_id: &str) -> Option<(u64, i64)> {
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        tracing::debug!(repo = %repo_id, status = %resp.status(), "HfWatcher: origin lookup failed");
        return None;
    }
    let m: HfApiModel = resp.json().await.ok()?;
    let created = m
        .created_at
        .as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc).timestamp())
        .unwrap_or(0);
    Some((m.downloads, created))
}

/// Apply the trust thresholds to ONE model whose origin statistics we have,
/// wherever they came from. Idempotent.
fn apply_trust_promotion(
    state: &SharedState,
    model_id: &crate::types::ModelId,
    repo_id: &str,
    downloads: u64,
    created_at_secs: i64,
) {
    let now = chrono::Utc::now().timestamp();
    if downloads < min_downloads_for_repo(repo_id) {
        return;
    }
    if created_at_secs > 0 && (now - created_at_secs) < MIN_AGE_FOR_TRUST_HOURS * 3600 {
        return;
    }
    // Promote — only if currently Discovered (don't override an
    // explicit user pin, and don't downgrade higher trust) AND
    // the anti-gaming cooldown allows it.
    let mut upgraded = false;
    let mut cooldown_skip = false;
    state
        .models
        .model_trust
        .entry(model_id.clone())
        .and_modify(|t| {
            if !should_auto_promote(t) {
                if matches!(t.trust_level, crate::types::ModelTrustLevel::Discovered)
                    && t.failed_promotions > 0
                {
                    cooldown_skip = true;
                }
                return;
            }
            t.trust_level = crate::types::ModelTrustLevel::DemandVerified;
            t.last_auto_promoted_at = Some(chrono::Utc::now());
            upgraded = true;
        })
        .or_insert_with(|| {
            upgraded = true;
            let mut info = crate::types::ModelTrustInfo::new_discovered();
            info.trust_level = crate::types::ModelTrustLevel::DemandVerified;
            info.last_auto_promoted_at = Some(chrono::Utc::now());
            info
        });
    if upgraded {
        tracing::info!(
            model = %model_id,
            repo = %repo_id,
            downloads,
            "HfWatcher: promoted to DemandVerified"
        );
    } else if cooldown_skip {
        tracing::debug!(
            model = %model_id,
            repo = %repo_id,
            "HfWatcher: re-promotion blocked by failed-promotion cooldown"
        );
    }
}

/// Does this model already sit at or above `DemandVerified`? Used to avoid
/// asking HuggingFace about a repo whose answer cannot change anything.
fn already_trusted_enough(state: &SharedState, model_id: &crate::types::ModelId) -> bool {
    state
        .models
        .model_trust
        .get(model_id)
        .map(|t| t.trust_level >= crate::types::ModelTrustLevel::DemandVerified)
        .unwrap_or(false)
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

    /// R134: virgin entry is always eligible for auto-promotion — no
    /// previous strikes, no cooldown.
    #[test]
    fn should_auto_promote_virgin_discovered() {
        let info = crate::types::ModelTrustInfo::new_discovered();
        assert!(should_auto_promote(&info));
    }

    /// R134: user-pinned models are never auto-promoted (user already has
    /// the trust level they want).
    #[test]
    fn should_auto_promote_skips_pinned() {
        let mut info = crate::types::ModelTrustInfo::new_pinned();
        info.trust_level = crate::types::ModelTrustLevel::Discovered;
        assert!(!should_auto_promote(&info));
    }

    /// R134: once the failure cap is hit, no further auto-promotion fires.
    #[test]
    fn should_auto_promote_blocks_at_failure_cap() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.failed_promotions = MAX_AUTO_PROMOTION_FAILURES;
        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(365));
        assert!(!should_auto_promote(&info));
    }

    /// R134: one strike → 7-day cooldown. Within the window: blocked.
    #[test]
    fn should_auto_promote_respects_cooldown() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.failed_promotions = 1;
        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(3));
        assert!(!should_auto_promote(&info), "3 days < 7-day cooldown");

        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(10));
        assert!(
            should_auto_promote(&info),
            "past 7-day cooldown is eligible"
        );
    }

    /// R134: cooldown grows with each strike — `BASE_DAYS * failed_promotions`.
    #[test]
    fn should_auto_promote_cooldown_grows_with_strikes() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.failed_promotions = 3; // 21-day cooldown expected
        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(20));
        assert!(!should_auto_promote(&info), "20d < 3*7=21d cooldown");

        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(22));
        assert!(should_auto_promote(&info));
    }

    /// R134: auto-promoted model that decays with 0 requests bumps the
    /// strike count via `maybe_decay`.
    #[test]
    fn maybe_decay_bumps_strikes_on_auto_promoted_zero_requests() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.trust_level = crate::types::ModelTrustLevel::DemandVerified;
        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(8));
        info.first_seen = chrono::Utc::now() - chrono::Duration::days(8);
        info.total_requests = 0;
        info.maybe_decay();
        assert_eq!(info.trust_level, crate::types::ModelTrustLevel::Discovered);
        assert_eq!(info.failed_promotions, 1);
    }

    /// R134: model that earned real requests, then decayed, does NOT
    /// take the strike — `record_request` already cleared the counter.
    #[test]
    fn maybe_decay_no_strike_when_real_usage() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.trust_level = crate::types::ModelTrustLevel::DemandVerified;
        info.last_auto_promoted_at = Some(chrono::Utc::now() - chrono::Duration::days(30));
        info.total_requests = 50;
        info.last_request_at = Some(chrono::Utc::now() - chrono::Duration::days(8));
        // failed_promotions stays at 0 because record_request was called.
        info.maybe_decay();
        assert_eq!(info.trust_level, crate::types::ModelTrustLevel::Discovered);
        assert_eq!(info.failed_promotions, 0);
    }

    /// Trusted publishers get the 10× lower threshold.
    #[test]
    fn min_downloads_threshold_trusted_publisher() {
        assert_eq!(
            min_downloads_for_repo("bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
            MIN_DOWNLOADS_FOR_TRUST_TRUSTED
        );
        assert_eq!(
            min_downloads_for_repo("Qwen/Qwen2.5-7B-Instruct-GGUF"),
            MIN_DOWNLOADS_FOR_TRUST_TRUSTED
        );
        // Case-insensitive match
        assert_eq!(
            min_downloads_for_repo("UNSLOTH/Phi-3-mini-4k-instruct-GGUF"),
            MIN_DOWNLOADS_FOR_TRUST_TRUSTED
        );
    }

    /// Unknown publishers retain the original 100k floor.
    #[test]
    fn min_downloads_threshold_unknown_publisher() {
        assert_eq!(
            min_downloads_for_repo("rando-user/some-model-GGUF"),
            MIN_DOWNLOADS_FOR_TRUST
        );
        assert_eq!(min_downloads_for_repo("no-slash"), MIN_DOWNLOADS_FOR_TRUST);
        assert_eq!(min_downloads_for_repo(""), MIN_DOWNLOADS_FOR_TRUST);
    }

    /// is_trusted_publisher mirrors the threshold helper.
    #[test]
    fn is_trusted_publisher_matches_allowlist() {
        assert!(is_trusted_publisher("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(is_trusted_publisher("bartowski/anything"));
        assert!(is_trusted_publisher("Bartowski/Anything"));
        assert!(!is_trusted_publisher("random/repo"));
        assert!(!is_trusted_publisher("/no-publisher"));
    }

    /// R134: `record_request` clears strikes when real demand finally
    /// arrives — defeats permanent lockout from a one-time download spike.
    #[test]
    fn record_request_clears_strikes() {
        let mut info = crate::types::ModelTrustInfo::new_discovered();
        info.failed_promotions = 2;
        info.record_request();
        assert_eq!(info.failed_promotions, 0);
        assert_eq!(info.total_requests, 1);
    }
}
