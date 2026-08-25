//! Core split-inference tests: tensor roundtrips, sampling, KV cache,
//! LRU eviction, batch forward, flash-attn vs standard, model-arch
//! detection, MLP activation comparison, and decode/prefill bench timings.

use super::super::super::layers::{run_attention, standard_attention};
use super::super::*;
use super::common::*;
use crate::inference::split::kv_cache::LayerKv;
use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};

#[test]
fn tensor_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
    let bytes = tensor_to_bytes(&tensor).unwrap();
    let restored = bytes_to_tensor(&bytes).unwrap();
    let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(data, restored_data);
    assert_eq!(restored.shape().dims(), &[2, 3]);
}

#[test]
fn tensor_q8_0_roundtrip() {
    // Hidden-state-like tensor [1, 32] (one full Q8_0 block) — verify the
    // tensor_to_bytes_q8_0 → bytes_to_tensor dispatch works end-to-end.
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let tensor = Tensor::from_vec(data.clone(), &[1, 32], &Device::Cpu).unwrap();
    let bytes = tensor_to_bytes_q8_0(&tensor).unwrap();

    // Q8_0: 4 (ndim) + 8 (shape: 2 dims × 4) + 4 (dtype tag) + 34 (one block) = 50 bytes
    assert_eq!(bytes.len(), 4 + 8 + 4 + 34);

    let restored = bytes_to_tensor(&bytes).unwrap();
    assert_eq!(restored.shape().dims(), &[1, 32]);

    let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_err = data
        .iter()
        .zip(restored_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.01, "Q8_0 roundtrip max err {max_err} too high");
}

#[test]
fn tensor_q8_0_compresses_vs_f32() {
    // 4096-element hidden state slice: f32 = 4 + 4 + 4 + 16384 = 16396 bytes;
    // Q8_0 = 4 + 4 + 4 + (128 blocks × 34) = 4364 bytes (~3.76× smaller).
    let data: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let tensor = Tensor::from_vec(data, &[1, 4096], &Device::Cpu).unwrap();

    let f32_bytes = tensor_to_bytes(&tensor).unwrap();
    let q8_bytes = tensor_to_bytes_q8_0(&tensor).unwrap();

    let ratio = f32_bytes.len() as f32 / q8_bytes.len() as f32;
    assert!(ratio > 3.5, "expected >3.5× compression, got {ratio}");
}

#[test]
fn sample_greedy() {
    let logits = Tensor::from_vec(vec![0.1f32, 0.2, 5.0, 0.3], &[1, 4], &Device::Cpu).unwrap();
    let token = sample_token(&logits, 0.0, 1.0).unwrap();
    assert_eq!(token, 2); // index of 5.0
}

#[test]
fn available_layer_ranges_from_manifest_basic() {
    use crate::types::{ModelId, ModelManifest, ShardInfo};

    let manifest = ModelManifest {
        id: ModelId("test".into()),
        name: "test".into(),
        architecture: crate::types::ModelArchitecture::Llama,
        num_layers: 12,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: 1000,
        shard_count: 3,
        shards: vec![
            ShardInfo {
                index: 0,
                layer_range: (0, 4),
                size_bytes: 300,
                hash: [0u8; 32],
                tensors: vec![],
            },
            ShardInfo {
                index: 1,
                layer_range: (4, 8),
                size_bytes: 300,
                hash: [0u8; 32],
                tensors: vec![],
            },
            ShardInfo {
                index: 2,
                layer_range: (8, 12),
                size_bytes: 400,
                hash: [0u8; 32],
                tensors: vec![],
            },
        ],
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: crate::types::NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "MIT".into(),
        mmproj: None,
    };

    // Single shard
    let ranges = available_layer_ranges_from_manifest(&manifest, &[0]);
    assert_eq!(ranges, vec![(0, 4)]);

    // Non-contiguous shards
    let ranges = available_layer_ranges_from_manifest(&manifest, &[0, 2]);
    assert_eq!(ranges, vec![(0, 4), (8, 12)]);

    // All shards → single range
    let ranges = available_layer_ranges_from_manifest(&manifest, &[0, 1, 2]);
    assert_eq!(ranges, vec![(0, 12)]);
}

// ── KvCacheStore tests ──

#[test]
fn kv_cache_store_isolates_requests() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Two different request IDs should get independent caches
    let model_key = "test-model";
    let req_a = "request-a";
    let req_b = "request-b";
    let num_layers = 2;

    // Create caches for both requests using KvCache
    {
        let mut entry_a = store.get_or_create(model_key, req_a, num_layers);
        let mut cache = LayerKv::with_dim(2, 128);
        let k = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        let v = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        entry_a.layers[0] = Some(cache);
    }
    {
        let mut entry_b = store.get_or_create(model_key, req_b, num_layers);
        let mut cache = LayerKv::with_dim(2, 128);
        let k = Tensor::from_vec(vec![10.0f32, 20.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        let v = Tensor::from_vec(vec![30.0f32, 40.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        entry_b.layers[0] = Some(cache);
    }

    // Verify request A has its own cache values
    {
        let entry_a = store.get_or_create(model_key, req_a, num_layers);
        let cache = entry_a.layers[0].as_ref().unwrap();
        let k = cache.k().unwrap().unwrap();
        let k_data = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(k_data, vec![1.0, 2.0]);
    }

    // Verify request B has its own separate cache values
    {
        let entry_b = store.get_or_create(model_key, req_b, num_layers);
        let cache = entry_b.layers[0].as_ref().unwrap();
        let k = cache.k().unwrap().unwrap();
        let k_data = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(k_data, vec![10.0, 20.0]);
    }

    assert_eq!(store.active_entries(), 2);
}

#[test]
fn kv_cache_store_clear_request() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let model_key = "test-model";
    let req_a = "request-a";
    let req_b = "request-b";

    // Create caches for two requests
    store.get_or_create(model_key, req_a, 4);
    store.get_or_create(model_key, req_b, 4);
    assert_eq!(store.active_entries(), 2);

    // Clear only request A
    store.clear_request(model_key, req_a);
    assert_eq!(store.active_entries(), 1);

    // Request B should still exist
    let entry_b = store.get_or_create(model_key, req_b, 4);
    assert_eq!(entry_b.layers.len(), 4);
}

#[test]
fn kv_cache_store_cleanup_request_id() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Create caches for the same request across multiple models
    store.get_or_create("model-a", "req-1", 2);
    store.get_or_create("model-b", "req-1", 2);
    store.get_or_create("model-a", "req-2", 2);
    assert_eq!(store.active_entries(), 3);

    // cleanup_request_id removes all entries for req-1
    store.cleanup_request_id("req-1");
    assert_eq!(store.active_entries(), 1);
}

// ── KV truncation (DSD Phase 2) ──

#[test]
fn kv_truncate_to_preserves_prefix_and_drops_suffix() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    {
        let mut entry = store.get_or_create("m", "r", 1);
        let mut cache = LayerKv::with_dim(2, 128);
        append_pos(&mut cache, 1.0, 10.0);
        append_pos(&mut cache, 2.0, 20.0);
        append_pos(&mut cache, 3.0, 30.0);
        append_pos(&mut cache, 4.0, 40.0);
        entry.layers[0] = Some(cache);
    }
    store.truncate_request_to("m", "r", 2).unwrap();

    let entry = store.get_or_create("m", "r", 1);
    let cache = entry.layers[0].as_ref().unwrap();
    assert_eq!(cache.current_seq_len(), 2);
    let k = cache.k().unwrap().unwrap();
    let k_data = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Each appended pos contributed 2 lanes (head_dim=2), so first 4 lanes correspond to positions 0-1.
    assert_eq!(k_data, vec![1.0, 1.0, 2.0, 2.0]);
    let v = cache.v().unwrap().unwrap();
    let v_data = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(v_data, vec![10.0, 10.0, 20.0, 20.0]);
}

#[test]
fn kv_truncate_to_target_geq_current_is_noop() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    {
        let mut entry = store.get_or_create("m", "r", 1);
        let mut cache = LayerKv::with_dim(2, 128);
        append_pos(&mut cache, 1.0, 10.0);
        append_pos(&mut cache, 2.0, 20.0);
        entry.layers[0] = Some(cache);
    }
    // Asking to truncate to 5 when only 2 positions exist must not corrupt the cache.
    store.truncate_request_to("m", "r", 5).unwrap();

    let entry = store.get_or_create("m", "r", 1);
    let cache = entry.layers[0].as_ref().unwrap();
    assert_eq!(cache.current_seq_len(), 2);
}

#[test]
fn kv_truncate_unallocated_layer_is_skipped() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    {
        let mut entry = store.get_or_create("m", "r", 3);
        // Layer 0 has data; layers 1 and 2 are None — must be skipped, not panic.
        let mut cache = LayerKv::with_dim(2, 128);
        append_pos(&mut cache, 1.0, 10.0);
        append_pos(&mut cache, 2.0, 20.0);
        entry.layers[0] = Some(cache);
    }
    store.truncate_request_to("m", "r", 1).unwrap();

    let entry = store.get_or_create("m", "r", 3);
    let cache = entry.layers[0].as_ref().unwrap();
    assert_eq!(cache.current_seq_len(), 1);
    assert!(entry.layers[1].is_none());
    assert!(entry.layers[2].is_none());
}

#[test]
fn kv_truncate_missing_request_is_noop() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    // No entry inserted — must succeed silently.
    store
        .truncate_request_to("m", "absent-request", 4)
        .expect("truncate on missing request should be a silent no-op");
}

#[test]
fn kv_truncate_all_layers_aligned() {
    // DSD partial-accept across multiple layers: every layer must end at the
    // same target_len so subsequent forward passes find a consistent KV state.
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    {
        let mut entry = store.get_or_create("m", "r", 4);
        for layer_idx in 0..4 {
            let mut cache = LayerKv::with_dim(2, 128);
            append_pos(&mut cache, 1.0, 10.0);
            append_pos(&mut cache, 2.0, 20.0);
            append_pos(&mut cache, 3.0, 30.0);
            append_pos(&mut cache, 4.0, 40.0);
            entry.layers[layer_idx] = Some(cache);
        }
    }
    store.truncate_request_to("m", "r", 2).unwrap();

    let entry = store.get_or_create("m", "r", 4);
    for layer_idx in 0..4 {
        let cache = entry.layers[layer_idx].as_ref().unwrap();
        assert_eq!(
            cache.current_seq_len(),
            2,
            "layer {layer_idx} expected len=2 after truncate"
        );
    }
}

#[test]
fn kv_cache_store_cleanup_expired() {
    let store = KvCacheStore::new(std::time::Duration::from_millis(1));

    store.get_or_create("model", "req-1", 2);
    store.get_or_create("model", "req-2", 2);
    assert_eq!(store.active_entries(), 2);

    // Wait for TTL to expire
    std::thread::sleep(std::time::Duration::from_millis(10));

    let cleaned = store.cleanup_expired();
    assert_eq!(cleaned, 2);
    assert_eq!(store.active_entries(), 0);
}

#[test]
fn kv_cache_store_fresh_entry_survives_cleanup() {
    let store = KvCacheStore::new(std::time::Duration::from_millis(50));

    // Create an entry that will expire
    store.get_or_create("model", "req-old", 2);
    std::thread::sleep(std::time::Duration::from_millis(60));

    // Create a fresh entry
    store.get_or_create("model", "req-new", 2);

    // Cleanup should only remove the old one
    let cleaned = store.cleanup_expired();
    assert_eq!(cleaned, 1);
    assert_eq!(store.active_entries(), 1);
}

// ── LRU eviction tests ──

#[test]
fn lru_eviction_respects_budget() {
    use crate::types::*;

    let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> = dashmap::DashMap::new();
    let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
        dashmap::DashMap::new();

    // Add two models: one old, one newer
    let key_a = (ModelId("model-a".into()), 0, 10);
    let mut entry_a = make_dummy_entry(500);
    entry_a.last_used = std::sync::atomic::AtomicU64::new(100); // older
    split_models.insert(key_a.clone(), entry_a);

    let key_b = (ModelId("model-b".into()), 0, 10);
    let mut entry_b = make_dummy_entry(500);
    entry_b.last_used = std::sync::atomic::AtomicU64::new(200); // newer
    split_models.insert(key_b.clone(), entry_b);

    // Budget is 1200MB, we need 400MB more → total 1000 + 400 = 1400 > 1200
    // Must evict 1 model (oldest) to bring it under: 500 + 400 = 900 ≤ 1200
    let evicted = evict_split_models_lru(&split_models, &active_pipelines, 1200, 400);
    assert_eq!(evicted.len(), 1);
    assert_eq!(split_models.len(), 1);
    // The older model (model-a, last_used=100) should have been evicted
    assert!(!split_models.contains_key(&key_a));
    assert!(split_models.contains_key(&key_b));
}

#[test]
fn lru_eviction_no_eviction_under_budget() {
    use crate::types::*;

    let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> = dashmap::DashMap::new();
    let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
        dashmap::DashMap::new();

    let key = (ModelId("model".into()), 0, 10);
    let entry = make_dummy_entry(200);
    split_models.insert(key, entry);

    // Budget is 1000MB, need 100MB → no eviction needed
    let evicted = evict_split_models_lru(&split_models, &active_pipelines, 1000, 100);
    assert_eq!(evicted.len(), 0);
    assert_eq!(split_models.len(), 1);
}

#[test]
fn lru_eviction_protects_active_models() {
    use crate::types::*;

    let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> = dashmap::DashMap::new();
    let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
        dashmap::DashMap::new();

    // Add two models
    let key_a = (ModelId("active-model".into()), 0, 10);
    let mut entry_a = make_dummy_entry(500);
    entry_a.last_used = std::sync::atomic::AtomicU64::new(100); // oldest
    split_models.insert(key_a.clone(), entry_a);

    let key_b = (ModelId("idle-model".into()), 0, 10);
    let mut entry_b = make_dummy_entry(500);
    entry_b.last_used = std::sync::atomic::AtomicU64::new(200);
    split_models.insert(key_b.clone(), entry_b);

    // Mark model-a as having an active pipeline
    let pipeline = PipelineAssignment {
        request_id: uuid::Uuid::new_v4(),
        segments: vec![PipelineSegment {
            node_id: NodeId([1u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("active-model".into()),
                index: 0,
            },
            layer_range: (0, 10),
        }],
        standbys: vec![],
        tp_groups: vec![],
        supports_speculative: false,
    };
    active_pipelines.insert(uuid::Uuid::new_v4(), pipeline);

    // Budget is 800MB, need 400MB → should evict idle-model (not active one)
    let evicted = evict_split_models_lru(&split_models, &active_pipelines, 800, 400);
    assert_eq!(evicted.len(), 1);
    assert!(split_models.contains_key(&key_a)); // Protected by active pipeline
    assert!(!split_models.contains_key(&key_b)); // Evicted
}

#[test]
fn lru_eviction_multiple_models() {
    use crate::types::*;

    let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> = dashmap::DashMap::new();
    let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
        dashmap::DashMap::new();

    // Add 3 models of 400MB each (total 1200MB)
    for i in 0..3 {
        let key = (ModelId(format!("model-{i}")), 0, 10);
        let mut entry = make_dummy_entry(400);
        entry.last_used = std::sync::atomic::AtomicU64::new(i as u64 * 100);
        split_models.insert(key, entry);
    }

    // Budget 800MB, need 200MB → need to free 600MB → evict 2 oldest
    let evicted = evict_split_models_lru(&split_models, &active_pipelines, 800, 200);
    assert_eq!(evicted.len(), 2);
    assert_eq!(split_models.len(), 1);
    // Only model-2 (last_used=200, newest) should remain
    assert!(split_models.contains_key(&(ModelId("model-2".into()), 0, 10)));
}

// ── Batch forward tests ──

#[test]
fn forward_batch_matches_sequential() {
    let hidden_dim = 128;
    let num_layers = 2;
    let mut model = make_test_split_model(num_layers, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Create two different input tensors (simulating decode step, seq_len=1)
    let input_a = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let input_b = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let index_pos = 5;

    // Run sequentially
    let out_a = model
        .forward(&input_a, index_pos, &kv_store, "seq-a")
        .unwrap();

    // Clear KV for a fresh comparison
    kv_store.clear_request(
        &format!(
            "{}-{}-{}",
            model.layer_start, model.layer_end, model.total_layers
        ),
        "seq-a",
    );

    let out_b = model
        .forward(&input_b, index_pos, &kv_store, "seq-b")
        .unwrap();
    kv_store.clear_request(
        &format!(
            "{}-{}-{}",
            model.layer_start, model.layer_end, model.total_layers
        ),
        "seq-b",
    );

    // Run batched
    let items = vec![
        BatchItem {
            input: &input_a,
            index_pos,
            request_id: "batch-a",
        },
        BatchItem {
            input: &input_b,
            index_pos,
            request_id: "batch-b",
        },
    ];
    let batch_out = model.forward_batch(&items, &kv_store).unwrap();

    // Compare shapes
    assert_eq!(out_a.shape(), batch_out[0].shape());
    assert_eq!(out_b.shape(), batch_out[1].shape());

    // Compare values (should be close — same model, same inputs, same index_pos)
    let diff_a = (&out_a - &batch_out[0]).unwrap().abs().unwrap();
    let diff_b = (&out_b - &batch_out[1]).unwrap().abs().unwrap();
    let flat_a = diff_a.flatten_all().unwrap();
    let flat_b = diff_b.flatten_all().unwrap();
    let max_diff_a: f32 = flat_a.max(0).unwrap().to_vec0().unwrap();
    let max_diff_b: f32 = flat_b.max(0).unwrap().to_vec0().unwrap();

    // Allow small numerical differences from batched vs sequential path
    assert!(
        max_diff_a < 1e-4,
        "Batch output A differs from sequential: max_diff={max_diff_a}"
    );
    assert!(
        max_diff_b < 1e-4,
        "Batch output B differs from sequential: max_diff={max_diff_b}"
    );
}

#[test]
fn forward_batch_single_item_matches_forward() {
    let hidden_dim = 128;
    let mut model = make_test_split_model(1, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let index_pos = 0;

    // Single-item batch should use forward() path
    let items = vec![BatchItem {
        input: &input,
        index_pos,
        request_id: "single",
    }];
    let batch_out = model.forward_batch(&items, &kv_store).unwrap();
    assert_eq!(batch_out.len(), 1);

    // Shape should be [1, 1, hidden_dim] for intermediate segment
    assert_eq!(batch_out[0].dims(), &[1, 1, hidden_dim]);
}

#[test]
fn forward_batch_empty_returns_empty() {
    let mut model = make_test_split_model(1, 128);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let items: Vec<BatchItem<'_>> = vec![];
    let out = model.forward_batch(&items, &kv_store).unwrap();
    assert!(out.is_empty());
}

#[test]
fn forward_batch_prefill_chunks_match_sequential() {
    // Item 7 Phase 4: when all items share (seq_len > 1, index_pos), the
    // fused prefill-batch forward should match sequential forwards.
    let hidden_dim = 128;
    let num_layers = 2;
    let mut model = make_test_split_model(num_layers, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let chunk_len = 8;
    let input_a = Tensor::randn(0f32, 1.0, (1, chunk_len, hidden_dim), &Device::Cpu).unwrap();
    let input_b = Tensor::randn(0f32, 1.0, (1, chunk_len, hidden_dim), &Device::Cpu).unwrap();
    let index_pos = 0;

    let out_a = model
        .forward(&input_a, index_pos, &kv_store, "seq-a")
        .unwrap();
    kv_store.clear_request(
        &format!(
            "{}-{}-{}",
            model.layer_start, model.layer_end, model.total_layers
        ),
        "seq-a",
    );
    let out_b = model
        .forward(&input_b, index_pos, &kv_store, "seq-b")
        .unwrap();
    kv_store.clear_request(
        &format!(
            "{}-{}-{}",
            model.layer_start, model.layer_end, model.total_layers
        ),
        "seq-b",
    );

    let items = vec![
        BatchItem {
            input: &input_a,
            index_pos,
            request_id: "batch-a",
        },
        BatchItem {
            input: &input_b,
            index_pos,
            request_id: "batch-b",
        },
    ];
    let batch_out = model.forward_batch(&items, &kv_store).unwrap();

    assert_eq!(batch_out.len(), 2);
    assert_eq!(out_a.shape(), batch_out[0].shape());
    assert_eq!(out_b.shape(), batch_out[1].shape());

    let max_diff = |a: &Tensor, b: &Tensor| -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap()
    };
    let da = max_diff(&out_a, &batch_out[0]);
    let db = max_diff(&out_b, &batch_out[1]);
    assert!(da < 1e-4, "prefill batch A differs: max_diff={da}");
    assert!(db < 1e-4, "prefill batch B differs: max_diff={db}");
}

#[test]
fn forward_batch_mixed_seq_len_falls_back() {
    // Heterogeneous seq_len: one item seq_len=1 (decode), one item seq_len=4
    // (prefill). Batching is unsafe — fall back to sequential forwards.
    let hidden_dim = 128;
    let mut model = make_test_split_model(2, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let decode_input = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let prefill_input = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();

    let items = vec![
        BatchItem {
            input: &decode_input,
            index_pos: 0,
            request_id: "decoder",
        },
        BatchItem {
            input: &prefill_input,
            index_pos: 0,
            request_id: "prefiller",
        },
    ];
    let out = model.forward_batch(&items, &kv_store).unwrap();
    assert_eq!(out.len(), 2);
    // Intermediate segment returns hidden states; shape is
    // [1, seq_len, hidden_dim] per item.
    assert_eq!(out[0].dims(), &[1, 1, hidden_dim]);
    assert_eq!(out[1].dims(), &[1, 4, hidden_dim]);
}

#[test]
fn forward_batch_mixed_index_pos_falls_back() {
    // Heterogeneous index_pos at seq_len > 1: mask would differ per item, so
    // batching is unsafe. Must fall back and still produce per-item output.
    let hidden_dim = 128;
    let mut model = make_test_split_model(2, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input_a = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let input_b = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();

    let items = vec![
        BatchItem {
            input: &input_a,
            index_pos: 0,
            request_id: "pos-0",
        },
        BatchItem {
            input: &input_b,
            index_pos: 3,
            request_id: "pos-3",
        },
    ];
    let out = model.forward_batch(&items, &kv_store).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].dims(), &[1, 4, hidden_dim]);
    assert_eq!(out[1].dims(), &[1, 4, hidden_dim]);
}

/// The fallback inside `forward_batch` is silent, and without a count there is
/// no way to tell a node that is batching from one that has been running
/// sequentially all along.
///
/// It was worth having: the counters are what showed the batched path being
/// taken **0 times out of 156** on a live node, because the gate required every
/// request to sit at the same position with the same history. Decode no longer
/// asks for either (it has no shared mask, and each slot attends to its own
/// cache), so a batch of ONE-token steps now fuses whatever positions the
/// requests are at — measured at 1.99x on a graphics card and 1.27x on a
/// processor, not the ~20% once recorded in `docs/FUTURE_WORK.md`.
///
/// Prompt processing still requires alignment, and that is what this test
/// drives: its inputs are four positions long, so a mismatched `index_pos`
/// must still fall back.
#[test]
fn forward_batch_counts_how_often_it_actually_batched() {
    let hidden_dim = 128;
    let mut model = make_test_split_model(2, hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    assert_eq!(model.batch_stats(), (0, 0), "starts unrecorded");

    let a = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let b = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();

    // Same position: this one really batches.
    model
        .forward_batch(
            &[
                BatchItem {
                    input: &a,
                    index_pos: 0,
                    request_id: "same-a",
                },
                BatchItem {
                    input: &b,
                    index_pos: 0,
                    request_id: "same-b",
                },
            ],
            &kv_store,
        )
        .unwrap();
    assert_eq!(
        model.batch_stats(),
        (1, 0),
        "a homogeneous batch is not a fallback"
    );

    // Different positions: silently runs its items one at a time.
    model
        .forward_batch(
            &[
                BatchItem {
                    input: &a,
                    index_pos: 0,
                    request_id: "drift-a",
                },
                BatchItem {
                    input: &b,
                    index_pos: 3,
                    request_id: "drift-b",
                },
            ],
            &kv_store,
        )
        .unwrap();
    assert_eq!(
        model.batch_stats(),
        (2, 1),
        "a mixed-position batch must be counted as a fallback"
    );

    // A single item takes the fast path above the counter and is not a batch
    // attempt at all — counting it would understate the fallback rate.
    model
        .forward_batch(
            &[BatchItem {
                input: &a,
                index_pos: 0,
                request_id: "solo",
            }],
            &kv_store,
        )
        .unwrap();
    assert_eq!(
        model.batch_stats(),
        (2, 1),
        "a one-item call is not a batch attempt"
    );
}

// ── Flash-attention CPU verification ──

#[test]
fn flash_attn_cpu_vs_standard_attention() {
    // Compare CPU flash attention output vs standard matmul attention
    let device = Device::Cpu;
    let b = 1;
    let n_head = 4;
    let n_kv_head = 2; // GQA: 4 Q heads, 2 KV heads
    let seq_len = 8;
    let head_dim = 32;

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();

    // Build causal mask (u8: 1=masked, 0=visible)
    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    // Standard path
    let out_std =
        standard_attention(&q, &k, &v, Some(&mask), head_dim, n_head, n_kv_head, None).unwrap();

    // Flash path (run_attention dispatches to CPU flash on CPU device)
    let out_flash = run_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        n_head,
        n_kv_head,
        head_dim,
        None,
        None,
    )
    .unwrap();

    assert_eq!(out_std.shape(), out_flash.shape());

    let diff = (&out_std - &out_flash).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(
        max_diff < 1e-4,
        "CPU flash attention differs from standard: max_diff={max_diff}"
    );
}

#[test]
fn flash_attn_cpu_decode_no_mask() {
    // Test decode step (seq_len=1) — no mask needed
    let device = Device::Cpu;
    let b = 1;
    let n_head = 4;
    let n_kv_head = 2;
    let head_dim = 32;
    let kv_len = 16;

    let q = Tensor::randn(0f32, 0.1, (b, n_head, 1, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();

    // Standard path (no mask for decode)
    let out_std = standard_attention(&q, &k, &v, None, head_dim, n_head, n_kv_head, None).unwrap();

    // Flash path
    let out_flash =
        run_attention(&q, &k, &v, None, n_head, n_kv_head, head_dim, None, None).unwrap();

    assert_eq!(out_std.shape(), out_flash.shape());

    let diff = (&out_std - &out_flash).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(
        max_diff < 1e-4,
        "CPU flash decode differs from standard: max_diff={max_diff}"
    );
}

// ── Model architecture detection ──

#[test]
fn model_arch_detection() {
    assert_eq!(ModelArch::from_gguf_arch("llama"), ModelArch::Llama);
    assert_eq!(ModelArch::from_gguf_arch("qwen2"), ModelArch::Qwen2);
    assert_eq!(ModelArch::from_gguf_arch("qwen3"), ModelArch::Qwen2);
    assert_eq!(ModelArch::from_gguf_arch("gemma"), ModelArch::Gemma);
    assert_eq!(ModelArch::from_gguf_arch("gemma2"), ModelArch::Gemma2);
    assert_eq!(ModelArch::from_gguf_arch("phi3"), ModelArch::Phi3);
    assert_eq!(ModelArch::from_gguf_arch("mistral"), ModelArch::Mistral);
    assert_eq!(ModelArch::from_gguf_arch("qwen2moe"), ModelArch::Qwen2);
    assert_eq!(ModelArch::from_gguf_arch("deepseek2"), ModelArch::DeepSeek2);
    assert_eq!(ModelArch::from_gguf_arch("glm4"), ModelArch::Glm4);
    assert_eq!(ModelArch::from_gguf_arch("llama4"), ModelArch::Llama4);
    assert_eq!(
        ModelArch::from_gguf_arch("starcoder2"),
        ModelArch::Starcoder2
    );
    assert_eq!(ModelArch::from_gguf_arch("qwen35"), ModelArch::Qwen35);
    assert_eq!(ModelArch::from_gguf_arch("qwen35moe"), ModelArch::Qwen35Moe);
    assert!(matches!(
        ModelArch::from_gguf_arch("unknown_arch"),
        ModelArch::Unknown(_)
    ));
}

#[test]
fn model_arch_properties() {
    // RoPE contiguous: Qwen2 family and DeepSeek2
    assert!(ModelArch::Qwen2.use_rope_contiguous());
    assert!(ModelArch::DeepSeek2.use_rope_contiguous());
    assert!(!ModelArch::Llama.use_rope_contiguous());
    assert!(ModelArch::Gemma.use_rope_contiguous());
    assert!(ModelArch::Gemma2.use_rope_contiguous());
    assert!(ModelArch::Phi3.use_rope_contiguous());
    assert!(ModelArch::Starcoder2.use_rope_contiguous());
    assert!(!ModelArch::Mistral.use_rope_contiguous());

    // Activation: Gemma uses Gelu, others SiLU
    assert_eq!(ModelArch::Gemma.default_activation(), Activation::Gelu);
    assert_eq!(ModelArch::Gemma2.default_activation(), Activation::Gelu);
    assert_eq!(ModelArch::Llama.default_activation(), Activation::SiLU);
    assert_eq!(ModelArch::Qwen2.default_activation(), Activation::SiLU);
    assert_eq!(ModelArch::Phi3.default_activation(), Activation::SiLU);
    assert_eq!(ModelArch::Mistral.default_activation(), Activation::SiLU);

    // Gemma norm: only Gemma family
    assert!(ModelArch::Gemma.use_gemma_norm());
    assert!(ModelArch::Gemma2.use_gemma_norm());
    assert!(!ModelArch::Llama.use_gemma_norm());
    assert!(!ModelArch::Qwen2.use_gemma_norm());

    // Supported: known architectures are supported, Unknown is not
    assert!(ModelArch::Llama.is_supported());
    assert!(ModelArch::Qwen2.is_supported());
    assert!(ModelArch::Gemma2.is_supported());
    assert!(ModelArch::Phi3.is_supported());
    assert!(ModelArch::DeepSeek2.is_supported());
    assert!(ModelArch::Qwen35.is_supported());
    assert!(ModelArch::Qwen35Moe.is_supported());
    assert!(ModelArch::Qwen35.is_hybrid_ssm());
    assert!(ModelArch::Qwen35Moe.is_hybrid_ssm());
    assert!(!ModelArch::Llama.is_hybrid_ssm());
    assert!(!ModelArch::Unknown("mamba".to_string()).is_supported());
}

// ── MLP activation comparison ──

#[test]
fn mlp_activation_silu_vs_gelu() {
    // Verify SiLU and Gelu produce different outputs
    let device = Device::Cpu;
    let dim = 64;
    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    // Shared weights
    let gate = make_qmatmul(dim, dim * 4);
    let down = make_qmatmul(dim * 4, dim);
    let up = make_qmatmul(dim, dim * 4);

    let mlp_silu = Mlp {
        ffn_gate: Some(gate.clone()),
        ffn_down: down.clone(),
        ffn_up: up.clone(),
        activation: Activation::SiLU,
    };
    let mlp_gelu = Mlp {
        ffn_gate: Some(gate),
        ffn_down: down,
        ffn_up: up,
        activation: Activation::Gelu,
    };

    let input = Tensor::randn(0f32, 1.0, (1, 4, dim), &device).unwrap();
    let out_silu = mlp_silu.forward(&input, None).unwrap();
    let out_gelu = mlp_gelu.forward(&input, None).unwrap();

    assert_eq!(out_silu.shape(), out_gelu.shape());
    let diff = (&out_silu - &out_gelu).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(
        max_diff > 0.0,
        "SiLU and Gelu should produce different outputs"
    );
}

// ── Performance benchmarks (#[ignore] by default) ──

/// Benchmark `SplitModel::forward_batch` vs sequential `forward()` across
/// batch sizes 1/2/4/8 on a medium test model (22 layers, 1024 hidden_dim).
/// Not a correctness test — prints timing to stdout when invoked with
/// `cargo test --release forward_batch_timing -- --nocapture --ignored`.
///
/// Ignored by default because it takes ~10 seconds on CPU and doesn't
/// assert anything; intended for hand-running during perf investigations.
#[test]
#[ignore]
fn forward_batch_timing() {
    let hidden_dim = 1024;
    let num_layers = 22;
    // Try CUDA first (the batching win's home turf); fall back to CPU.
    let (device, device_label) = match candle_core::Device::new_cuda(0) {
        Ok(d) => (d, "CUDA:0"),
        Err(_) => (candle_core::Device::Cpu, "CPU"),
    };
    let mut model = make_test_split_model_on(num_layers, hidden_dim, device);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let iters = 20usize;
    let batch_sizes = [1usize, 2, 4, 8];

    eprintln!(
        "\nforward_batch timing | hidden={hidden_dim} layers={num_layers} iters={iters} device={device_label}\n"
    );
    eprintln!(
        "{:<10} {:<18} {:<18} {:<10}",
        "batch", "batch_ms/iter", "sequential_ms/iter", "speedup"
    );
    eprintln!("{:-<60}", "");

    for &batch_size in &batch_sizes {
        // Fresh inputs on the test device per batch size.
        let inputs: Vec<Tensor> = (0..batch_size)
            .map(|_| Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), model.device()).unwrap())
            .collect();

        // Warm up
        for (slot, input) in inputs.iter().enumerate() {
            let rid = format!("warm-{slot}");
            let _ = model.forward(input, 0, &kv_store, &rid);
            kv_store.clear_request(
                &format!(
                    "{}-{}-{}",
                    model.layer_start, model.layer_end, model.total_layers
                ),
                &rid,
            );
        }

        // Time batched
        let items: Vec<BatchItem> = inputs
            .iter()
            .enumerate()
            .map(|(i, t)| BatchItem {
                input: t,
                index_pos: 0,
                request_id: Box::leak(format!("bench-batch-{i}").into_boxed_str()),
            })
            .collect();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            // Reset KV per iter so each call does equal work
            for item in &items {
                kv_store.clear_request(
                    &format!(
                        "{}-{}-{}",
                        model.layer_start, model.layer_end, model.total_layers
                    ),
                    item.request_id,
                );
            }
            let _ = model.forward_batch(&items, &kv_store).unwrap();
        }
        let batch_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Time sequential
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for (i, t) in inputs.iter().enumerate() {
                let rid = format!("bench-seq-{i}");
                kv_store.clear_request(
                    &format!(
                        "{}-{}-{}",
                        model.layer_start, model.layer_end, model.total_layers
                    ),
                    &rid,
                );
                let _ = model.forward(t, 0, &kv_store, &rid).unwrap();
            }
        }
        let seq_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = seq_ms / batch_ms.max(1e-9);
        eprintln!("{batch_size:<10} {batch_ms:<18.2} {seq_ms:<18.2} {speedup:<10.2}x");
    }
    eprintln!();
}

/// Item 7 Phase 4 timing: fused prefill-chunk `forward_batch` vs N sequential
/// `forward()` calls at seq_len > 1.  Measures the synthetic ceiling of the
/// Phase A batching win under burst-admit of same-shape prompts — the
/// end-to-end bench (`swarmllm bench --concurrency N`) translates this into
/// TTFT improvement once the same admits queue up in the worker's mpsc.
///
/// Run with:
///   cargo test --release forward_prefill_batch_timing -- --nocapture --ignored
#[test]
#[ignore]
fn forward_prefill_batch_timing() {
    let hidden_dim = 1024;
    let num_layers = 22;
    let (device, device_label) = match candle_core::Device::new_cuda(0) {
        Ok(d) => (d, "CUDA:0"),
        Err(_) => (candle_core::Device::Cpu, "CPU"),
    };
    let mut model = make_test_split_model_on(num_layers, hidden_dim, device);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let iters = 10usize;
    // Filter chunk sizes to what the test model's context window permits.
    // `make_test_split_model_on` uses max_seq_len = 128, so 512 won't fit.
    let raw_chunk_sizes = [32usize, 64, 128];
    let batch_sizes = [2usize, 4, 8];
    let chunk_sizes: Vec<usize> = raw_chunk_sizes
        .into_iter()
        .filter(|&c| c <= model.max_seq_len)
        .collect();

    eprintln!(
        "\nforward_prefill_batch timing | hidden={hidden_dim} layers={num_layers} iters={iters} device={device_label}"
    );
    eprintln!("(batch of N `[1, chunk]` inputs vs N sequential `forward()` calls)\n");
    eprintln!(
        "{:<8} {:<8} {:<18} {:<18} {:<10}",
        "chunk", "batch", "batch_ms/iter", "sequential_ms/iter", "speedup"
    );
    eprintln!("{:-<66}", "");

    let cache_prefix = format!(
        "{}-{}-{}",
        model.layer_start, model.layer_end, model.total_layers
    );

    for &chunk_size in chunk_sizes.iter() {
        for &batch_size in &batch_sizes {
            let inputs: Vec<Tensor> = (0..batch_size)
                .map(|_| {
                    Tensor::randn(0f32, 1.0, (1, chunk_size, hidden_dim), model.device()).unwrap()
                })
                .collect();

            // Warm up one batched call.
            {
                let items: Vec<BatchItem> = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, t)| BatchItem {
                        input: t,
                        index_pos: 0,
                        request_id: Box::leak(
                            format!("warm-{chunk_size}-{batch_size}-{i}").into_boxed_str(),
                        ),
                    })
                    .collect();
                let _ = model.forward_batch(&items, &kv_store).unwrap();
                for item in &items {
                    kv_store.clear_request(&cache_prefix, item.request_id);
                }
            }

            // Time batched.
            let items: Vec<BatchItem> = inputs
                .iter()
                .enumerate()
                .map(|(i, t)| BatchItem {
                    input: t,
                    index_pos: 0,
                    request_id: Box::leak(
                        format!("bench-prefill-batch-{chunk_size}-{batch_size}-{i}")
                            .into_boxed_str(),
                    ),
                })
                .collect();
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for item in &items {
                    kv_store.clear_request(&cache_prefix, item.request_id);
                }
                let _ = model.forward_batch(&items, &kv_store).unwrap();
            }
            let batch_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Time sequential.
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for (i, t) in inputs.iter().enumerate() {
                    let rid = format!("bench-prefill-seq-{chunk_size}-{batch_size}-{i}");
                    kv_store.clear_request(&cache_prefix, &rid);
                    let _ = model.forward(t, 0, &kv_store, &rid).unwrap();
                }
            }
            let seq_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = seq_ms / batch_ms.max(1e-9);
            eprintln!(
                "{chunk_size:<8} {batch_size:<8} {batch_ms:<18.2} {seq_ms:<18.2} {speedup:<10.2}x"
            );
        }
    }
    eprintln!();
}

// ── Tail-segment load without shard 0 (real model on disk; #[ignore] by default) ──

/// A node serving the LAST pipeline segment of a weight-tied model must be able
/// to load its output head from `tied_output_weight.bin`, because the tensor the
/// head aliases (`token_embd.weight`) physically lives in shard 0 — which such a
/// node has no reason to hold.
///
/// Point `SWARMLLM_TAIL_MODEL_DIR` at a model directory containing ONLY a late
/// shard plus `gguf_header.bin`, `manifest.json` and `tied_output_weight.bin`.
/// That is the exact on-disk state of a real swarm node serving the tail. Before
/// the sidecar was wired into `ShardReader` this failed with "Failed to load
/// output head: ShardReader: position N is in a missing region".
///
/// Negative control: move the sidecar aside and re-run — the load must fail.
#[test]
#[ignore] // SWARMLLM_TAIL_MODEL_DIR=... cargo test tail_segment_loads -- --ignored --nocapture
fn tail_segment_loads_without_shard_zero() {
    let Ok(dir) = std::env::var("SWARMLLM_TAIL_MODEL_DIR") else {
        eprintln!("Skipping: set SWARMLLM_TAIL_MODEL_DIR to a tail-only model dir");
        return;
    };
    let model_dir = std::path::Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("manifest.json")).unwrap())
            .unwrap();
    let total_size = manifest["total_size_bytes"].as_u64().unwrap();

    // Use the highest-indexed shard actually present — the tail.
    let mut chosen = None;
    for s in manifest["shards"].as_array().unwrap() {
        let idx = s["index"].as_u64().unwrap() as u32;
        let path = model_dir.join(format!("shard_{idx:03}.bin"));
        if !path.exists() {
            continue;
        }
        let entries: Vec<crate::types::ShardTensorEntry> =
            serde_json::from_value(s["tensors"].clone()).unwrap();
        let lr = s["layer_range"].as_array().unwrap();
        let range = (
            lr[0].as_u64().unwrap() as usize,
            lr[1].as_u64().unwrap() as usize,
        );
        chosen = Some((idx, path, entries, range));
    }
    let (idx, path, entries, (layer_start, layer_end)) =
        chosen.expect("no shard_NNN.bin present in SWARMLLM_TAIL_MODEL_DIR");
    assert!(
        layer_start > 0,
        "dir must NOT contain shard 0 or the test proves nothing — got layers {layer_start}..{layer_end}"
    );
    eprintln!("Loading tail: shard {idx}, layers {layer_start}..{layer_end}, no shard 0 present");

    let model = SplitModel::load_from_shards(
        model_dir,
        vec![(idx, path)],
        &[entries],
        total_size,
        layer_start,
        layer_end,
        false, // is_first — embeddings live in shard 0, which we don't have
        true,  // is_last  — so the output head IS needed
    )
    .expect("tail segment must load its output head from tied_output_weight.bin");

    assert_eq!(
        model.total_layers,
        manifest["num_layers"].as_u64().unwrap() as usize
    );
    eprintln!("OK — output head loaded without shard 0");
}

/// The head-room guard must actually refuse a forward, not just compute a
/// verdict nobody reads. Sets a budget of zero so the very first quantum is
/// over it, and checks the error a caller would see.
///
/// This is the wiring test: `kv_budget`'s own tests prove the arithmetic, and
/// would keep passing if the guard were never called (which is how a
/// correctly-computed occupancy counter came to be read in the wrong process
/// earlier the same week).
#[test]
fn a_forward_is_refused_when_the_kv_budget_is_exhausted() {
    let hidden_dim = 128;
    let mut model = make_test_split_model(1, hidden_dim);
    model.kv_budget_bytes = Some(0);
    model.kv_bytes_per_token = 1_000_000;

    let store = KvCacheStore::new(std::time::Duration::from_secs(60));
    let input = Tensor::randn(0f32, 1.0, (1, 3, hidden_dim), &Device::Cpu).unwrap();
    let err = model
        .forward(&input, 0, &store, "over-budget")
        .expect_err("a zero budget must refuse the first quantum");

    assert!(
        matches!(err, crate::error::SwarmError::ServiceUnavailable(_)),
        "must be 503 so a coordinator re-routes to a peer, got {err:?}"
    );
    assert!(
        err.to_string().contains("KV cache"),
        "the message must say what ran out: {err}"
    );
}

/// The same model with no budget recorded — every CPU node, and any GPU node
/// where free VRAM could not be read — must be completely unaffected.
#[test]
fn no_recorded_budget_means_no_refusal() {
    let hidden_dim = 128;
    let mut model = make_test_split_model(1, hidden_dim);
    model.kv_budget_bytes = None;
    model.kv_bytes_per_token = u64::MAX;

    let store = KvCacheStore::new(std::time::Duration::from_secs(60));
    let input = Tensor::randn(0f32, 1.0, (1, 3, hidden_dim), &Device::Cpu).unwrap();
    assert!(
        model.forward(&input, 0, &store, "unbudgeted").is_ok(),
        "an unknown budget must never be read as a zero one"
    );
}

/// A request refused for exceeding the KV budget must not leave its cache
/// behind.
///
/// A long prompt is prefilled in chunks and claims a quantum each time it
/// crosses one, so by the time a chunk is refused the request has usually
/// allocated several. Leaving them made a refusal a RATCHET: the request failed,
/// its cache survived until the session expired ten minutes later, and the next
/// attempt started from a higher `in_use`. Measured on a live node 2026-08-25 —
/// refusals at 1152, then 2304, then 3456 MB against a 1166 MB budget, in exact
/// one-quantum steps, until the card sat at 97% and decode had fallen from
/// 29 tok/s to 1.0 (gotcha #387).
#[test]
fn a_refused_request_gives_back_the_cache_it_had_taken() {
    let mut model = super::common::make_test_split_model(2, 64);
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let dev = candle_core::Device::Cpu;

    model.kv_bytes_per_token = 1024;
    // Room for one 512-position quantum and very little else, so the first
    // forward is admitted and the next claim is not.
    model.kv_budget_bytes = Some(1024 * 600);

    let input = candle_core::Tensor::zeros((1, 8, 64), candle_core::DType::F32, &dev).unwrap();
    let req = "refused-request";
    model
        .forward(&input, 0, &store, req)
        .expect("the first forward fits the budget");
    assert_eq!(
        store.active_entries(),
        1,
        "the first forward must have allocated a cache entry, or the test proves nothing"
    );

    // Claim another quantum with the budget already spent.
    let err = model
        .forward(&input, 0, &store, req)
        .expect_err("a claim past the budget must be refused");
    assert!(
        matches!(err, crate::error::SwarmError::ServiceUnavailable(_)),
        "a budget refusal is a 503 so a coordinator can re-route it: {err:?}"
    );
    assert_eq!(
        store.active_entries(),
        0,
        "the refused request must give its cache back — otherwise every refusal \
         permanently raises the floor and the node ratchets itself to a halt"
    );
}
