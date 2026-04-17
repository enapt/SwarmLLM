//! Integration tests for KV-cache per-request isolation and multi-turn reuse (Phase 11).
//!
//! Tests that concurrent requests get independent KV-cache state and that
//! multi-turn conversations can reuse cached context via prefix matching.

use std::time::Duration;

use swarmllm::inference::kv_cache::{CacheReuse, KvCacheManager};
use swarmllm::types::{ModelId, NodeId, PipelineAssignment, PipelineSegment, ShardId};

fn make_pipeline(request_id: uuid::Uuid) -> PipelineAssignment {
    PipelineAssignment {
        request_id,
        segments: vec![
            PipelineSegment {
                node_id: NodeId([1u8; 32]),
                shard_id: ShardId {
                    model_id: ModelId("test-model".into()),
                    index: 0,
                },
                layer_range: (0, 16),
            },
            PipelineSegment {
                node_id: NodeId([2u8; 32]),
                shard_id: ShardId {
                    model_id: ModelId("test-model".into()),
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

/// Test that two concurrent requests get independent KV-cache sessions.
/// Each request's cache state must not leak into the other.
#[test]
fn test_kv_cache_per_request_isolation() {
    let mut mgr = KvCacheManager::new(Duration::from_secs(600));

    let req1_id = uuid::Uuid::new_v4();
    let req2_id = uuid::Uuid::new_v4();
    let pipeline1 = make_pipeline(req1_id);
    let pipeline2 = make_pipeline(req2_id);

    // Register two concurrent sessions
    mgr.register_session(req1_id, pipeline1, 100);
    mgr.register_session(req2_id, pipeline2, 200);

    // Each session should have its own state
    assert_eq!(mgr.get_session(&req1_id).unwrap().cached_tokens, 100);
    assert_eq!(mgr.get_session(&req2_id).unwrap().cached_tokens, 200);

    // Updating one should not affect the other
    mgr.update_cached_tokens(&req1_id, 150);
    assert_eq!(mgr.get_session(&req1_id).unwrap().cached_tokens, 150);
    assert_eq!(mgr.get_session(&req2_id).unwrap().cached_tokens, 200); // Unchanged

    // Invalidating one should not affect the other
    mgr.invalidate_session(&req1_id);
    assert!(mgr.get_session(&req1_id).is_none());
    assert!(mgr.get_session(&req2_id).is_some());
}

/// Test multi-turn KV-cache reuse: the second message in a conversation
/// reuses the cache from the first turn via prompt prefix matching.
#[test]
fn test_kv_cache_multi_turn_reuse() {
    let mut mgr = KvCacheManager::new(Duration::from_secs(600));
    let internal_id = uuid::Uuid::new_v4();
    let pipeline = make_pipeline(internal_id);
    let active_peers = vec![NodeId([1u8; 32]), NodeId([2u8; 32])];

    // First turn: register with the full prompt
    let turn1_prompt = "User: Hello, how are you?\nAssistant: I'm doing well!";
    mgr.register_multi_turn(
        "conv-123",
        internal_id,
        pipeline,
        42, // 42 tokens cached
        turn1_prompt.to_string(),
    );

    // Second turn: extends the conversation with a new message
    let turn2_prompt = "User: Hello, how are you?\nAssistant: I'm doing well!\nUser: What's new?";
    match mgr.check_multi_turn_reuse("conv-123", turn2_prompt, &active_peers) {
        CacheReuse::Hit { start_pos } => {
            assert_eq!(start_pos, 42); // Should skip the first 42 cached tokens
        }
        CacheReuse::Miss => panic!("Expected cache hit for prefix-matching prompt"),
    }
}

/// Test that a completely different prompt gets a cache miss.
#[test]
fn test_kv_cache_multi_turn_miss_different_prompt() {
    let mut mgr = KvCacheManager::new(Duration::from_secs(600));
    let internal_id = uuid::Uuid::new_v4();
    let pipeline = make_pipeline(internal_id);
    let active_peers = vec![NodeId([1u8; 32]), NodeId([2u8; 32])];

    mgr.register_multi_turn(
        "conv-456",
        internal_id,
        pipeline,
        42,
        "Hello world".to_string(),
    );

    // Completely unrelated prompt — should miss
    match mgr.check_multi_turn_reuse("conv-456", "Goodbye universe", &active_peers) {
        CacheReuse::Miss => {} // Expected
        CacheReuse::Hit { .. } => panic!("Expected cache miss for unrelated prompt"),
    }
}

/// Test that cache is invalidated when pipeline nodes go offline.
#[test]
fn test_kv_cache_invalidated_on_pipeline_degradation() {
    let mut mgr = KvCacheManager::new(Duration::from_secs(600));
    let internal_id = uuid::Uuid::new_v4();
    let pipeline = make_pipeline(internal_id);

    mgr.register_multi_turn("conv-789", internal_id, pipeline, 42, "Hello".to_string());

    // Node 2 went offline — only node 1 is active
    let active_peers = vec![NodeId([1u8; 32])];
    match mgr.check_multi_turn_reuse("conv-789", "Hello, more text", &active_peers) {
        CacheReuse::Miss => {} // Expected — pipeline degraded
        CacheReuse::Hit { .. } => panic!("Expected miss when pipeline node is offline"),
    }
}

/// Test that expired sessions return cache miss.
#[test]
fn test_kv_cache_expired_session_miss() {
    let mut mgr = KvCacheManager::new(Duration::from_millis(1)); // 1ms TTL
    let internal_id = uuid::Uuid::new_v4();
    let pipeline = make_pipeline(internal_id);

    mgr.register_multi_turn(
        "conv-expired",
        internal_id,
        pipeline,
        42,
        "Hello".to_string(),
    );

    // Wait for TTL to expire
    std::thread::sleep(Duration::from_millis(10));

    let active_peers = vec![NodeId([1u8; 32]), NodeId([2u8; 32])];
    match mgr.check_multi_turn_reuse("conv-expired", "Hello, more", &active_peers) {
        CacheReuse::Miss => {} // Expected — session expired
        CacheReuse::Hit { .. } => panic!("Expected miss for expired session"),
    }
}
