use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{NodeId, PipelineAssignment};

/// Maximum number of multi-turn sessions to prevent unbounded memory growth.
const MAX_MULTI_TURN_SESSIONS: usize = 10_000;

/// Unique identifier for a KV-cache session (conversation).
pub type SessionId = uuid::Uuid;

/// Per-session KV-cache entry tracking which nodes hold cached state.
#[derive(Debug, Clone)]
pub struct KvCacheSession {
    pub session_id: SessionId,
    /// The pipeline assignment used for this session — we try to reuse
    /// the same pipeline for subsequent turns to preserve KV-cache.
    pub pipeline: PipelineAssignment,
    /// Token count of the cached prefix (prompt + all prior completions).
    pub cached_tokens: u32,
    /// When the session was last accessed.
    pub last_accessed: Instant,
    /// Nodes holding KV-cache state for this session.
    pub cache_holders: Vec<NodeId>,
    /// The full prompt text that produced the cached state.
    /// Used for prefix matching: if a new request's prompt starts with this
    /// text, we can skip prefill and set start_pos = cached_tokens.
    pub cached_prompt: String,
}

/// Result of checking whether a multi-turn session can reuse its KV-cache.
#[derive(Debug)]
pub enum CacheReuse {
    /// Full prefix match: the new prompt starts with the cached prompt.
    /// `start_pos` is the token position to begin processing new tokens.
    Hit { start_pos: u32 },
    /// No reusable cache (new session, expired, or prompt doesn't share prefix).
    Miss,
}

/// Manages KV-cache sessions across the cluster.
///
/// Tracks which nodes hold cached state for each conversation session.
/// Sessions expire after a configurable TTL (default 10 minutes).
/// When a session's pipeline changes (node dropped), the cache is invalidated
/// and context must be reprocessed.
///
/// ## Multi-turn reuse
///
/// When a request provides a `session_id`, the manager looks up the previous
/// session and checks whether the new prompt starts with the cached prompt
/// prefix. If so, only new tokens need processing (start_pos = cached_tokens).
/// If the pipeline changed (different nodes), the cache is invalidated.
pub struct KvCacheManager {
    sessions: HashMap<SessionId, KvCacheSession>,
    /// Multi-turn sessions keyed by string session_id (from the API).
    /// Maps user-provided session_id → internal SessionId (UUID).
    multi_turn_sessions: HashMap<String, SessionId>,
    ttl: Duration,
}

impl KvCacheManager {
    pub fn new(ttl: Duration) -> Self {
        Self {
            sessions: HashMap::new(),
            multi_turn_sessions: HashMap::new(),
            ttl,
        }
    }

    fn cache_holders(pipeline: &PipelineAssignment) -> Vec<NodeId> {
        pipeline
            .segments
            .iter()
            .map(|s| s.node_id.clone())
            .collect()
    }

    /// Register or update a KV-cache session.
    pub fn register_session(
        &mut self,
        session_id: SessionId,
        pipeline: PipelineAssignment,
        cached_tokens: u32,
    ) {
        let cache_holders = Self::cache_holders(&pipeline);

        let entry = KvCacheSession {
            session_id,
            pipeline,
            cached_tokens,
            last_accessed: Instant::now(),
            cache_holders,
            cached_prompt: String::new(),
        };

        self.sessions.insert(session_id, entry);

        tracing::debug!(
            session_id = %session_id,
            cached_tokens,
            "Registered KV-cache session"
        );
    }

    /// Register a multi-turn session with full prompt text for prefix matching.
    pub fn register_multi_turn(
        &mut self,
        user_session_id: &str,
        internal_id: SessionId,
        pipeline: PipelineAssignment,
        cached_tokens: u32,
        prompt: String,
    ) {
        let cache_holders = Self::cache_holders(&pipeline);

        let entry = KvCacheSession {
            session_id: internal_id,
            pipeline,
            cached_tokens,
            last_accessed: Instant::now(),
            cache_holders,
            cached_prompt: prompt,
        };

        // Evict oldest multi-turn session if at capacity to prevent unbounded growth
        if self.multi_turn_sessions.len() >= MAX_MULTI_TURN_SESSIONS {
            // First, clean up any orphaned entries (multi_turn_sessions pointing to
            // evicted sessions) — these sort as newest with Instant::now() fallback
            let orphaned: Vec<String> = self
                .multi_turn_sessions
                .iter()
                .filter(|(_, id)| !self.sessions.contains_key(id))
                .map(|(k, _)| k.clone())
                .collect();
            for key in orphaned {
                self.multi_turn_sessions.remove(&key);
            }

            // If still at capacity, evict the oldest valid session
            if self.multi_turn_sessions.len() >= MAX_MULTI_TURN_SESSIONS {
                if let Some(oldest_user_key) = self
                    .multi_turn_sessions
                    .iter()
                    .min_by_key(|(_, id)| {
                        self.sessions
                            .get(id)
                            .map(|s| s.last_accessed)
                            // Orphans (session evicted but multi_turn entry remains) should
                            // sort first (oldest) so they get evicted before valid sessions.
                            .unwrap_or(Instant::now() - std::time::Duration::from_secs(86400))
                    })
                    .map(|(k, _)| k.clone())
                {
                    if let Some(internal_id) = self.multi_turn_sessions.remove(&oldest_user_key) {
                        self.sessions.remove(&internal_id);
                    }
                }
            }
        }

        self.sessions.insert(internal_id, entry);
        self.multi_turn_sessions
            .insert(user_session_id.to_string(), internal_id);

        tracing::debug!(
            user_session_id,
            internal_id = %internal_id,
            cached_tokens,
            "Registered multi-turn KV-cache session"
        );
    }

    /// Check if a multi-turn session can reuse its KV-cache.
    ///
    /// If the new prompt starts with the cached prompt prefix and the pipeline
    /// is still valid (all nodes reachable), returns `CacheReuse::Hit` with
    /// the start position for new token processing.
    pub fn check_multi_turn_reuse(
        &mut self,
        user_session_id: &str,
        new_prompt: &str,
        active_peers: &HashSet<NodeId>,
    ) -> CacheReuse {
        let internal_id = match self.multi_turn_sessions.get(user_session_id) {
            Some(id) => *id,
            None => {
                tracing::debug!(
                    user_session_id,
                    total_sessions = self.sessions.len(),
                    total_multi_turn = self.multi_turn_sessions.len(),
                    "DIAG: KV-cache MISS — no multi-turn session found"
                );
                return CacheReuse::Miss;
            }
        };

        // Check if session exists and isn't expired
        let session = match self.sessions.get(&internal_id) {
            Some(s) => s,
            None => {
                tracing::info!(
                    user_session_id,
                    internal_id = %internal_id,
                    "DIAG: KV-cache MISS — internal session evicted"
                );
                self.multi_turn_sessions.remove(user_session_id);
                return CacheReuse::Miss;
            }
        };

        if session.last_accessed.elapsed() > self.ttl {
            tracing::info!(
                user_session_id,
                elapsed_secs = session.last_accessed.elapsed().as_secs(),
                ttl_secs = self.ttl.as_secs(),
                "DIAG: KV-cache MISS — session expired"
            );
            self.sessions.remove(&internal_id);
            self.multi_turn_sessions.remove(user_session_id);
            return CacheReuse::Miss;
        }

        // Check pipeline validity (all nodes still reachable). active_peers
        // is a HashSet so each contains() is O(1) — the holders Vec is small
        // (== pipeline segments) but peer_registry can be large.
        let missing: Vec<&NodeId> = session
            .cache_holders
            .iter()
            .filter(|node| !active_peers.contains(node))
            .collect();

        if !missing.is_empty() {
            tracing::info!(
                user_session_id,
                missing = missing.len(),
                total_holders = session.cache_holders.len(),
                active_peers = active_peers.len(),
                "DIAG: KV-cache MISS — pipeline degraded, nodes unreachable"
            );
            self.sessions.remove(&internal_id);
            self.multi_turn_sessions.remove(user_session_id);
            return CacheReuse::Miss;
        }

        // Check prefix match: new prompt must start with the cached prompt
        if session.cached_prompt.is_empty() {
            tracing::debug!(
                user_session_id,
                "DIAG: KV-cache MISS — empty cached prompt (no invalidation)"
            );
            return CacheReuse::Miss;
        }
        if !new_prompt.starts_with(&session.cached_prompt) {
            tracing::info!(
                user_session_id,
                cached_prompt_len = session.cached_prompt.len(),
                new_prompt_len = new_prompt.len(),
                "DIAG: KV-cache MISS — prompt prefix mismatch"
            );
            // Invalidate stale session since the conversation diverged
            self.sessions.remove(&internal_id);
            self.multi_turn_sessions.remove(user_session_id);
            return CacheReuse::Miss;
        }

        let start_pos = session.cached_tokens;
        tracing::info!(
            user_session_id,
            start_pos,
            cached_prompt_len = session.cached_prompt.len(),
            new_prompt_len = new_prompt.len(),
            cached_tokens = session.cached_tokens,
            cache_holders = session.cache_holders.len(),
            "DIAG: KV-cache HIT — skipping prefill"
        );

        // Touch the session
        if let Some(s) = self.sessions.get_mut(&internal_id) {
            s.last_accessed = Instant::now();
        }

        CacheReuse::Hit { start_pos }
    }

    /// Get the internal session ID for a user session.
    pub fn get_internal_id(&self, user_session_id: &str) -> Option<SessionId> {
        self.multi_turn_sessions.get(user_session_id).copied()
    }

    /// Look up a session and check if it's still valid.
    pub fn get_session(&mut self, session_id: &SessionId) -> Option<&KvCacheSession> {
        // Check if expired
        if let Some(session) = self.sessions.get(session_id) {
            if session.last_accessed.elapsed() > self.ttl {
                tracing::debug!(
                    session_id = %session_id,
                    "KV-cache session expired"
                );
                self.sessions.remove(session_id);
                self.multi_turn_sessions.retain(|_, id| id != session_id);
                return None;
            }
        }

        self.sessions.get(session_id)
    }

    /// Touch a session to refresh its TTL (test-only — production refreshes via update_cached_tokens).
    #[cfg(test)]
    pub fn touch_session(&mut self, session_id: &SessionId) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.last_accessed = Instant::now();
        }
    }

    /// Update the cached token count after a successful completion.
    pub fn update_cached_tokens(&mut self, session_id: &SessionId, new_total: u32) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.cached_tokens = new_total;
            session.last_accessed = Instant::now();
        }
    }

    /// Update the cached prompt text for a session (after completion).
    pub fn update_cached_prompt(&mut self, session_id: &SessionId, prompt: String) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.cached_prompt = prompt;
            session.last_accessed = Instant::now();
        }
    }

    /// Check if a session's pipeline is still valid (all nodes still reachable).
    #[cfg(test)]
    pub fn validate_pipeline(
        &self,
        session_id: &SessionId,
        active_peers: &HashSet<NodeId>,
    ) -> PipelineValidity {
        match self.sessions.get(session_id) {
            None => PipelineValidity::NotFound,
            Some(session) => {
                if session.last_accessed.elapsed() > self.ttl {
                    return PipelineValidity::Expired;
                }

                let missing: Vec<&NodeId> = session
                    .cache_holders
                    .iter()
                    .filter(|node| !active_peers.contains(node))
                    .collect();

                if missing.is_empty() {
                    PipelineValidity::Valid
                } else {
                    tracing::debug!(
                        session_id = %session_id,
                        missing = missing.len(),
                        "KV-cache pipeline has missing nodes"
                    );
                    PipelineValidity::Degraded {
                        missing_nodes: missing.into_iter().cloned().collect(),
                    }
                }
            }
        }
    }

    /// Invalidate a session (e.g., pipeline changed, cache is stale).
    pub fn invalidate_session(&mut self, session_id: &SessionId) {
        if self.sessions.remove(session_id).is_some() {
            tracing::debug!(session_id = %session_id, "Invalidated KV-cache session");
        }
        // Also clean up multi_turn_sessions mapping
        self.multi_turn_sessions.retain(|_, id| id != session_id);
    }

    /// Clean up all expired sessions. Returns count of expired sessions.
    pub fn cleanup_expired(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions
            .retain(|_, session| session.last_accessed.elapsed() <= self.ttl);
        let after = self.sessions.len();
        let expired = before - after;
        if expired > 0 {
            // Also clean up stale multi_turn_sessions entries
            self.multi_turn_sessions
                .retain(|_, id| self.sessions.contains_key(id));
        }
        expired
    }

    /// Get the number of active sessions.
    #[cfg(test)]
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Get a session's previous pipeline assignment for reuse.
    pub fn get_previous_pipeline(&mut self, session_id: &SessionId) -> Option<PipelineAssignment> {
        self.get_session(session_id).map(|s| s.pipeline.clone())
    }

    /// Persist all active multi-turn sessions to the database.
    ///
    /// Called during graceful shutdown so sessions can be restored on next startup.
    /// Only multi-turn sessions (those with a non-empty cached_prompt) are worth
    /// persisting — ephemeral single-request sessions won't be reused.
    ///
    /// When `privacy_mode` is true, the `cached_prompt` field is replaced with an
    /// empty string so user prompts are never written to disk. The session metadata
    /// (pipeline, token count, holders) is still persisted for operational use, but
    /// prefix matching will not work after restart.
    pub fn save_to_db(&self, db: &Database, privacy_mode: bool) -> Result<usize, SwarmError> {
        db.clear_tree("kv_sessions")?;

        let mut saved = 0;
        for (user_session_id, internal_id) in &self.multi_turn_sessions {
            if let Some(session) = self.sessions.get(internal_id) {
                if session.last_accessed.elapsed() > self.ttl {
                    continue;
                }

                let now_system = SystemTime::now();
                let elapsed = session.last_accessed.elapsed();
                let last_accessed_system = now_system.checked_sub(elapsed).unwrap_or(now_system);

                let persisted = PersistedSession {
                    user_session_id: user_session_id.clone(),
                    internal_id: session.session_id,
                    pipeline: session.pipeline.clone(),
                    cached_tokens: session.cached_tokens,
                    cache_holders: session.cache_holders.clone(),
                    cached_prompt: if privacy_mode {
                        String::new()
                    } else {
                        session.cached_prompt.clone()
                    },
                    last_accessed: last_accessed_system,
                    ttl_secs: self.ttl.as_secs(),
                };

                db.put_json("kv_sessions", user_session_id, &persisted)?;
                saved += 1;
            }
        }

        tracing::info!(saved, "Persisted KV-cache sessions to database");
        Ok(saved)
    }

    /// Restore sessions from the database.
    ///
    /// Called on startup to resume multi-turn conversations that are still
    /// within their TTL. Sessions whose TTL has elapsed since the last access
    /// are silently discarded.
    pub fn restore_from_db(&mut self, db: &Database) -> Result<usize, SwarmError> {
        let entries: Vec<PersistedSession> = db.iter_json("kv_sessions")?;
        let now = SystemTime::now();
        let mut restored = 0;

        for persisted in entries {
            let elapsed = now
                .duration_since(persisted.last_accessed)
                .unwrap_or(Duration::from_secs(u64::MAX));

            let ttl = Duration::from_secs(persisted.ttl_secs);
            if elapsed > ttl {
                tracing::debug!(
                    user_session_id = %persisted.user_session_id,
                    elapsed_secs = elapsed.as_secs(),
                    ttl_secs = persisted.ttl_secs,
                    "Skipping expired persisted session"
                );
                continue;
            }

            let last_accessed = Instant::now()
                .checked_sub(elapsed)
                .unwrap_or_else(Instant::now);

            let session = KvCacheSession {
                session_id: persisted.internal_id,
                pipeline: persisted.pipeline,
                cached_tokens: persisted.cached_tokens,
                last_accessed,
                cache_holders: persisted.cache_holders,
                cached_prompt: persisted.cached_prompt,
            };

            self.sessions.insert(persisted.internal_id, session);
            self.multi_turn_sessions
                .insert(persisted.user_session_id.clone(), persisted.internal_id);
            restored += 1;

            tracing::debug!(
                user_session_id = %persisted.user_session_id,
                cached_tokens = persisted.cached_tokens,
                remaining_ttl_secs = ttl.as_secs().saturating_sub(elapsed.as_secs()),
                "Restored persisted KV-cache session"
            );
        }

        if restored > 0 {
            tracing::info!(restored, "Restored KV-cache sessions from database");
        }
        Ok(restored)
    }
}

/// Serializable representation of a KV-cache session for database persistence.
///
/// Stores session metadata (NOT tensor data) so multi-turn conversations
/// can resume after a node restart. The actual KV tensor cache is rebuilt
/// by re-processing the cached prompt prefix on the first request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub user_session_id: String,
    pub internal_id: SessionId,
    pub pipeline: PipelineAssignment,
    pub cached_tokens: u32,
    pub cache_holders: Vec<NodeId>,
    pub cached_prompt: String,
    /// Wall-clock time of last access (serializable, unlike `Instant`).
    pub last_accessed: SystemTime,
    /// The TTL in seconds that was in effect when the session was saved.
    pub ttl_secs: u64,
}

/// Result of pipeline validation for a KV-cache session.
#[cfg(test)]
#[derive(Debug)]
pub enum PipelineValidity {
    /// Pipeline is intact, KV-cache can be reused.
    Valid,
    /// Session not found (new conversation or already expired).
    NotFound,
    /// Session has expired (TTL exceeded).
    Expired,
    /// Some pipeline nodes are unreachable — cache must be invalidated.
    Degraded { missing_nodes: Vec<NodeId> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_pipeline(request_id: uuid::Uuid) -> PipelineAssignment {
        PipelineAssignment {
            request_id,
            segments: vec![
                PipelineSegment {
                    node_id: NodeId([1u8; 32]),
                    shard_id: ShardId {
                        model_id: ModelId("test".into()),
                        index: 0,
                    },
                    layer_range: (0, 16),
                },
                PipelineSegment {
                    node_id: NodeId([2u8; 32]),
                    shard_id: ShardId {
                        model_id: ModelId("test".into()),
                        index: 1,
                    },
                    layer_range: (16, 32),
                },
            ],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        }
    }

    #[test]
    fn register_and_retrieve_session() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        let session = mgr.get_session(&session_id);
        assert!(session.is_some());
        assert_eq!(session.unwrap().cached_tokens, 128);
    }

    #[test]
    fn session_expires() {
        let mut mgr = KvCacheManager::new(Duration::from_millis(1));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        std::thread::sleep(Duration::from_millis(10));

        let session = mgr.get_session(&session_id);
        assert!(session.is_none());
    }

    #[test]
    fn cleanup_removes_expired() {
        let mut mgr = KvCacheManager::new(Duration::from_millis(1));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        std::thread::sleep(Duration::from_millis(10));

        let cleaned = mgr.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(mgr.active_sessions(), 0);
    }

    #[test]
    fn touch_refreshes_ttl() {
        let mut mgr = KvCacheManager::new(Duration::from_millis(50));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        std::thread::sleep(Duration::from_millis(30));
        mgr.touch_session(&session_id);
        std::thread::sleep(Duration::from_millis(30));

        // Should still be valid because we touched it
        let session = mgr.get_session(&session_id);
        assert!(session.is_some());
    }

    #[test]
    fn validate_pipeline_with_all_peers() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);

        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();
        match mgr.validate_pipeline(&session_id, &active) {
            PipelineValidity::Valid => {}
            other => panic!("Expected Valid, got {:?}", other),
        }
    }

    #[test]
    fn validate_pipeline_with_missing_node() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);

        // Only node 1 is active
        let active: HashSet<NodeId> = [NodeId([1u8; 32])].into();
        match mgr.validate_pipeline(&session_id, &active) {
            PipelineValidity::Degraded { missing_nodes } => {
                assert_eq!(missing_nodes.len(), 1);
                assert_eq!(missing_nodes[0], NodeId([2u8; 32]));
            }
            other => panic!("Expected Degraded, got {:?}", other),
        }
    }

    #[test]
    fn invalidate_session_removes_it() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        mgr.invalidate_session(&session_id);
        assert_eq!(mgr.active_sessions(), 0);
    }

    #[test]
    fn update_cached_tokens() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let session_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(session_id);

        mgr.register_session(session_id, pipeline, 128);
        mgr.update_cached_tokens(&session_id, 256);

        let session = mgr.get_session(&session_id).unwrap();
        assert_eq!(session.cached_tokens, 256);
    }

    #[test]
    fn multi_turn_cache_hit() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let internal_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(internal_id);
        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();

        // Register initial turn
        let prompt_turn1 = "Hello, how are you?";
        mgr.register_multi_turn(
            "session-abc",
            internal_id,
            pipeline,
            42,
            prompt_turn1.to_string(),
        );

        // Second turn extends the prompt
        let prompt_turn2 = "Hello, how are you? I'm doing well. What's new?";
        match mgr.check_multi_turn_reuse("session-abc", prompt_turn2, &active) {
            CacheReuse::Hit { start_pos } => {
                assert_eq!(start_pos, 42);
            }
            CacheReuse::Miss => panic!("Expected cache hit"),
        }
    }

    #[test]
    fn multi_turn_cache_miss_no_prefix() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let internal_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(internal_id);
        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();

        mgr.register_multi_turn(
            "session-abc",
            internal_id,
            pipeline,
            42,
            "Hello, how are you?".to_string(),
        );

        // Completely different prompt — no prefix match
        let different_prompt = "What is the weather today?";
        match mgr.check_multi_turn_reuse("session-abc", different_prompt, &active) {
            CacheReuse::Miss => {}
            CacheReuse::Hit { .. } => panic!("Expected cache miss"),
        }
    }

    #[test]
    fn multi_turn_cache_miss_expired() {
        let mut mgr = KvCacheManager::new(Duration::from_millis(1));
        let internal_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(internal_id);

        mgr.register_multi_turn(
            "session-abc",
            internal_id,
            pipeline,
            42,
            "Hello".to_string(),
        );

        std::thread::sleep(Duration::from_millis(10));

        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();
        match mgr.check_multi_turn_reuse("session-abc", "Hello, more text", &active) {
            CacheReuse::Miss => {}
            CacheReuse::Hit { .. } => panic!("Expected miss due to expiry"),
        }
    }

    #[test]
    fn multi_turn_cache_miss_pipeline_degraded() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let internal_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(internal_id);

        mgr.register_multi_turn(
            "session-abc",
            internal_id,
            pipeline,
            42,
            "Hello".to_string(),
        );

        // Only node 1 is active (node 2 dropped)
        let active: HashSet<NodeId> = [NodeId([1u8; 32])].into();
        match mgr.check_multi_turn_reuse("session-abc", "Hello, more text", &active) {
            CacheReuse::Miss => {}
            CacheReuse::Hit { .. } => panic!("Expected miss due to degraded pipeline"),
        }
    }

    #[test]
    fn multi_turn_invalidate_cleans_up() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let internal_id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(internal_id);

        mgr.register_multi_turn(
            "session-abc",
            internal_id,
            pipeline,
            42,
            "Hello".to_string(),
        );

        assert_eq!(mgr.active_sessions(), 1);
        mgr.invalidate_session(&internal_id);
        assert_eq!(mgr.active_sessions(), 0);
        assert!(mgr.get_internal_id("session-abc").is_none());
    }

    #[test]
    fn multi_turn_unknown_session() {
        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let active: HashSet<NodeId> = [NodeId([1u8; 32])].into();

        match mgr.check_multi_turn_reuse("nonexistent", "Hello", &active) {
            CacheReuse::Miss => {}
            CacheReuse::Hit { .. } => panic!("Expected miss for unknown session"),
        }
    }

    #[test]
    fn persist_and_restore_sessions() {
        let db = Database::open_temp().unwrap();

        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let id1 = uuid::Uuid::new_v4();
        let pipeline1 = make_pipeline(id1);
        mgr.register_multi_turn(
            "user-sess-1",
            id1,
            pipeline1,
            100,
            "Hello world".to_string(),
        );

        let id2 = uuid::Uuid::new_v4();
        let pipeline2 = make_pipeline(id2);
        mgr.register_multi_turn(
            "user-sess-2",
            id2,
            pipeline2,
            200,
            "How are you?".to_string(),
        );

        let saved = mgr.save_to_db(&db, false).unwrap();
        assert_eq!(saved, 2);

        let mut mgr2 = KvCacheManager::new(Duration::from_secs(600));
        let restored = mgr2.restore_from_db(&db).unwrap();
        assert_eq!(restored, 2);
        assert_eq!(mgr2.active_sessions(), 2);

        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();
        match mgr2.check_multi_turn_reuse("user-sess-1", "Hello world, more text", &active) {
            CacheReuse::Hit { start_pos } => assert_eq!(start_pos, 100),
            CacheReuse::Miss => panic!("Expected cache hit after restore"),
        }
    }

    #[test]
    fn restore_skips_expired_sessions() {
        let db = Database::open_temp().unwrap();

        let mut mgr = KvCacheManager::new(Duration::from_secs(1));
        let id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(id);
        mgr.register_multi_turn("user-sess-1", id, pipeline, 50, "Hello".to_string());

        let saved = mgr.save_to_db(&db, false).unwrap();
        assert_eq!(saved, 1);

        std::thread::sleep(Duration::from_millis(1500));

        let mut mgr2 = KvCacheManager::new(Duration::from_secs(600));
        let restored = mgr2.restore_from_db(&db).unwrap();
        assert_eq!(restored, 0);
        assert_eq!(mgr2.active_sessions(), 0);
    }

    #[test]
    fn persist_skips_expired_sessions() {
        let db = Database::open_temp().unwrap();

        let mut mgr = KvCacheManager::new(Duration::from_millis(1));
        let id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(id);
        mgr.register_multi_turn("user-sess-1", id, pipeline, 50, "Hello".to_string());

        std::thread::sleep(Duration::from_millis(10));

        let saved = mgr.save_to_db(&db, false).unwrap();
        assert_eq!(saved, 0);
    }

    #[test]
    fn persist_overwrites_previous() {
        let db = Database::open_temp().unwrap();

        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(id);
        mgr.register_multi_turn("sess-1", id, pipeline, 50, "Hello".to_string());

        mgr.save_to_db(&db, false).unwrap();

        let mut mgr2 = KvCacheManager::new(Duration::from_secs(600));
        let id2 = uuid::Uuid::new_v4();
        let pipeline2 = make_pipeline(id2);
        mgr2.register_multi_turn("sess-2", id2, pipeline2, 75, "Goodbye".to_string());

        mgr2.save_to_db(&db, false).unwrap();

        let mut mgr3 = KvCacheManager::new(Duration::from_secs(600));
        let restored = mgr3.restore_from_db(&db).unwrap();
        assert_eq!(restored, 1);
        assert!(mgr3.get_internal_id("sess-2").is_some());
        assert!(mgr3.get_internal_id("sess-1").is_none());
    }

    #[test]
    fn privacy_mode_strips_cached_prompt() {
        let db = Database::open_temp().unwrap();

        let mut mgr = KvCacheManager::new(Duration::from_secs(600));
        let id = uuid::Uuid::new_v4();
        let pipeline = make_pipeline(id);
        mgr.register_multi_turn(
            "private-sess",
            id,
            pipeline,
            100,
            "Secret user prompt".to_string(),
        );

        // Save with privacy_mode = true
        let saved = mgr.save_to_db(&db, true).unwrap();
        assert_eq!(saved, 1);

        // Restore and verify the cached_prompt is empty (no prefix match possible)
        let mut mgr2 = KvCacheManager::new(Duration::from_secs(600));
        let restored = mgr2.restore_from_db(&db).unwrap();
        assert_eq!(restored, 1);

        // Session metadata was restored but prefix matching won't work
        // because cached_prompt was stripped
        let active: HashSet<NodeId> = [NodeId([1u8; 32]), NodeId([2u8; 32])].into();
        match mgr2.check_multi_turn_reuse("private-sess", "Secret user prompt more", &active) {
            CacheReuse::Miss => {} // Expected: empty cached_prompt => miss
            CacheReuse::Hit { .. } => panic!("Expected miss when privacy_mode stripped prompt"),
        }
    }
}
