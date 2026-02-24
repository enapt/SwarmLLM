use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::{NodeId, PipelineAssignment};

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
}

/// Manages KV-cache sessions across the cluster.
///
/// Tracks which nodes hold cached state for each conversation session.
/// Sessions expire after a configurable TTL (default 10 minutes).
/// When a session's pipeline changes (node dropped), the cache is invalidated
/// and context must be reprocessed.
pub struct KvCacheManager {
    sessions: HashMap<SessionId, KvCacheSession>,
    ttl: Duration,
}

impl KvCacheManager {
    pub fn new(ttl: Duration) -> Self {
        Self {
            sessions: HashMap::new(),
            ttl,
        }
    }

    /// Register or update a KV-cache session.
    pub fn register_session(
        &mut self,
        session_id: SessionId,
        pipeline: PipelineAssignment,
        cached_tokens: u32,
    ) {
        let cache_holders: Vec<NodeId> = pipeline
            .segments
            .iter()
            .map(|s| s.node_id.clone())
            .collect();

        let entry = KvCacheSession {
            session_id,
            pipeline,
            cached_tokens,
            last_accessed: Instant::now(),
            cache_holders,
        };

        self.sessions.insert(session_id, entry);

        tracing::debug!(
            session_id = %session_id,
            cached_tokens,
            "Registered KV-cache session"
        );
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
                return None;
            }
        }

        self.sessions.get(session_id)
    }

    /// Touch a session to refresh its TTL.
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

    /// Check if a session's pipeline is still valid (all nodes still reachable).
    pub fn validate_pipeline(
        &self,
        session_id: &SessionId,
        active_peers: &[NodeId],
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
    }

    /// Clean up all expired sessions. Returns count of expired sessions.
    pub fn cleanup_expired(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions
            .retain(|_, session| session.last_accessed.elapsed() <= self.ttl);
        before - self.sessions.len()
    }

    /// Get the number of active sessions.
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Get a session's previous pipeline assignment for reuse.
    pub fn get_previous_pipeline(&mut self, session_id: &SessionId) -> Option<PipelineAssignment> {
        self.get_session(session_id).map(|s| s.pipeline.clone())
    }
}

/// Result of pipeline validation for a KV-cache session.
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

        let active = vec![NodeId([1u8; 32]), NodeId([2u8; 32])];
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
        let active = vec![NodeId([1u8; 32])];
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
}
