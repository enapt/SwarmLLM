use super::super::layers::{run_attention, standard_attention, topk_cpu};
use super::model::SplitModel;
use super::rope::precompute_freqs_cis;
use super::*;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

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
        let mut cache = KvCache::new(2, 128);
        let k = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        let v = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        entry_a.layers[0] = Some(cache);
    }
    {
        let mut entry_b = store.get_or_create(model_key, req_b, num_layers);
        let mut cache = KvCache::new(2, 128);
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

/// Append a single position to a KvCache. Helper for the truncate tests.
fn append_pos(cache: &mut KvCache, key: f32, val: f32) {
    let k = Tensor::from_vec(vec![key, key], &[1, 1, 1, 2], &Device::Cpu).unwrap();
    let v = Tensor::from_vec(vec![val, val], &[1, 1, 1, 2], &Device::Cpu).unwrap();
    cache.append(&k, &v).unwrap();
}

#[test]
fn kv_truncate_to_preserves_prefix_and_drops_suffix() {
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    {
        let mut entry = store.get_or_create("m", "r", 1);
        let mut cache = KvCache::new(2, 128);
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
        let mut cache = KvCache::new(2, 128);
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
        let mut cache = KvCache::new(2, 128);
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
            let mut cache = KvCache::new(2, 128);
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

fn make_dummy_entry(vram_mb: u64) -> SplitModelEntry {
    // Construct a minimal metadata-only SplitModelEntry for eviction tests.
    SplitModelEntry {
        last_used: std::sync::atomic::AtomicU64::new(0),
        estimated_vram_mb: vram_mb,
        is_complete: false,
        eos_tokens: vec![],
        eos_token_str: String::new(),
        bos_token: String::new(),
        cached_chat_template: None,
        vocab: None,
        layer_start: 0,
        layer_end: 0,
    }
}

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
    assert_eq!(evicted, 1);
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
    assert_eq!(evicted, 0);
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
    assert_eq!(evicted, 1);
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
    assert_eq!(evicted, 2);
    assert_eq!(split_models.len(), 1);
    // Only model-2 (last_used=200, newest) should remain
    assert!(split_models.contains_key(&(ModelId("model-2".into()), 0, 10)));
}

// ── Batch forward tests ──

/// Create a minimal SplitModel on the given device. Used by benchmarks that
/// want to test GPU paths.
fn make_test_split_model_on(
    num_layers: usize,
    hidden_dim: usize,
    device: candle_core::Device,
) -> SplitModel {
    make_test_split_model_impl(num_layers, hidden_dim, device)
}

/// Create a minimal SplitModel with real layers for testing forward/forward_batch.
fn make_test_split_model(num_layers: usize, hidden_dim: usize) -> SplitModel {
    make_test_split_model_impl(num_layers, hidden_dim, candle_core::Device::Cpu)
}

fn make_test_split_model_impl(
    num_layers: usize,
    hidden_dim: usize,
    device: candle_core::Device,
) -> SplitModel {
    // Build a minimal model with random weights for testing.
    // Identity-like weight matrices on the caller-chosen device.
    let head_dim = 64;
    let n_head = hidden_dim / head_dim;
    let n_kv_head = n_head; // no GQA in test model

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        // Create a random weight tensor and quantize it
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let rope_dim = head_dim;
    let freq_base = 10000.0f32;
    let theta: Vec<f32> = (0..rope_dim / 2)
        .map(|i| 1.0 / freq_base.powf(i as f32 * 2.0 / rope_dim as f32))
        .collect();
    let idx: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let theta_t = Tensor::from_vec(theta.clone(), (1, rope_dim / 2), &device).unwrap();
    let idx_t = Tensor::from_vec(idx.clone(), (max_seq_len, 1), &device).unwrap();
    let freqs = idx_t.matmul(&theta_t).unwrap();
    let cos = freqs.cos().unwrap();
    let sin = freqs.sin().unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mut layers = Vec::new();
    for _ in 0..num_layers {
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
        };
        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, hidden_dim),
            attention_wv: make_qmatmul(hidden_dim, hidden_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(&norm_w),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: Some(make_qmatmul(hidden_dim, hidden_dim * 4)),
                ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                activation: Activation::SiLU,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            post_attention_norm: None,
            post_ffw_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim,
            skip_rope: false,
        }));
    }

    SplitModel {
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: num_layers,
        total_layers: num_layers + 2, // Not last segment
        hidden_dim,
        arch: ModelArch::Llama,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
        final_logit_softcap: None,
    }
}

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
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    // Build causal mask (u8: 1=masked, 0=visible)
    let mask_data: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    // Standard path
    let out_std = standard_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        head_dim,
        n_head,
        n_kv_head,
        &neg_inf,
        None,
    )
    .unwrap();

    // Flash path (run_attention dispatches to CPU flash on CPU device)
    let out_flash = run_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        n_head,
        n_kv_head,
        head_dim,
        &neg_inf,
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
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    // Standard path (no mask for decode)
    let out_std = standard_attention(
        &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
    )
    .unwrap();

    // Flash path
    let out_flash = run_attention(
        &q, &k, &v, None, n_head, n_kv_head, head_dim, &neg_inf, None,
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
        "CPU flash decode differs from standard: max_diff={max_diff}"
    );
}

// ── Model architecture detection tests ──

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

// ── GQA verification tests ──

/// Helper: create a SplitModel with explicit GQA configuration.
#[allow(clippy::too_many_arguments)]
fn make_gqa_test_model(
    num_layers: usize,
    hidden_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    use_rope_contiguous: bool,
    activation: Activation,
    attn_logit_softcap: Option<f32>,
    arch: ModelArch,
) -> SplitModel {
    let device = candle_core::Device::Cpu;
    let head_dim = hidden_dim / n_head;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let rope_dim = head_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let kv_dim = n_kv_head * head_dim;
    let mut layers = Vec::new();
    for _ in 0..num_layers {
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
        };
        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, kv_dim),
            attention_wv: make_qmatmul(hidden_dim, kv_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(&norm_w),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: Some(make_qmatmul(hidden_dim, hidden_dim * 4)),
                ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                activation,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            post_attention_norm: None,
            post_ffw_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
            use_rope_contiguous,
            attn_logit_softcap,
            rope_dim,
            skip_rope: false,
        }));
    }

    SplitModel {
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: num_layers,
        total_layers: num_layers + 2,
        hidden_dim,
        arch,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
        final_logit_softcap: None,
    }
}

/// Helper: assert two tensors are close within tolerance.
fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, msg: &str) {
    assert_eq!(a.shape(), b.shape(), "{msg}: shape mismatch");
    let diff = (a - b).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff < tol, "{msg}: max_diff={max_diff} >= tol={tol}");
}

#[test]
fn gqa_standard_attention_llama3_ratio() {
    // Llama 3 8B: GQA ratio=4 (scaled: n_head=8, n_kv_head=2)
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 12, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mask_data: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    let out = standard_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        head_dim,
        n_head,
        n_kv_head,
        &neg_inf,
        None,
    )
    .unwrap();
    assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Output contains NaN/Inf"
    );
}

#[test]
fn gqa_standard_attention_mqa_ratio() {
    // Multi-Query Attention: n_kv_head=1 (extreme GQA)
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 1, 6, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let out = standard_attention(
        &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
    )
    .unwrap();
    assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
}

#[test]
fn gqa_flash_vs_standard_llama3_prefill() {
    // CPU flash vs standard with GQA ratio=4, causal mask
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 10, 32);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mask_data: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

    let out_std = standard_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        head_dim,
        n_head,
        n_kv_head,
        &neg_inf,
        None,
    )
    .unwrap();
    let out_flash = run_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        n_head,
        n_kv_head,
        head_dim,
        &neg_inf,
        None,
    )
    .unwrap();
    assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA ratio=4 flash vs standard");
}

#[test]
fn gqa_flash_vs_standard_llama3_decode() {
    // Decode step (seq_len=1 Q, longer KV) with GQA ratio=4
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, head_dim, kv_len) = (1, 8, 2, 32, 20);

    let q = Tensor::randn(0f32, 0.1, (b, n_head, 1, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let out_std = standard_attention(
        &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
    )
    .unwrap();
    let out_flash = run_attention(
        &q, &k, &v, None, n_head, n_kv_head, head_dim, &neg_inf, None,
    )
    .unwrap();
    assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA decode flash vs standard");
}

#[test]
fn gqa_forward_llama3_style() {
    // End-to-end forward with Llama 3-style GQA
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Prefill
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "llama3").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 6, &kv_store, "llama3").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

#[test]
fn gqa_forward_mistral_style() {
    // Mistral 7B: same GQA as Llama 3
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Mistral,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "mistral").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn gqa_forward_phi3_mha_style() {
    // Phi-3-mini: MHA (n_head == n_kv_head)
    let (hidden_dim, n_head, n_kv_head) = (192, 6, 6);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Phi3,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "phi3").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
}

#[test]
fn gemma2_gelu_activation_forward() {
    // Gemma 2 uses Gelu activation in MLP
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
    let mut model = make_gqa_test_model(
        1,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::Gelu,
        None,
        ModelArch::Gemma2,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "gemma2-gelu").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Gemma2 Gelu produced NaN/Inf"
    );
}

#[test]
fn gemma2_attn_logit_softcap() {
    // Test attention logit soft-capping (Gemma 2 feature)
    // Use stddev=1.0 so logits are large enough for softcap to visibly affect output
    let device = Device::Cpu;
    let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 4, 2, 6, 32);

    let q = Tensor::randn(0f32, 1.0, (b, n_head, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    // Without soft-capping
    let out_no_cap = standard_attention(
        &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
    )
    .unwrap();

    // With soft-capping (cap=50.0 like Gemma 2)
    let out_capped = standard_attention(
        &q,
        &k,
        &v,
        None,
        head_dim,
        n_head,
        n_kv_head,
        &neg_inf,
        Some(50.0),
    )
    .unwrap();

    // Both should produce valid output
    assert_eq!(out_no_cap.shape(), out_capped.shape());
    let flat: Vec<f32> = out_capped.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Soft-capped attention NaN/Inf"
    );

    // Outputs should differ (soft-capping changes the attention weights)
    let diff = (&out_no_cap - &out_capped).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff > 0.0, "Soft-capping should change the output");
}

#[test]
fn gemma2_full_forward_with_softcap() {
    // Gemma 2 end-to-end: Gelu + softcap + GQA
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::Gelu,
        Some(50.0),
        ModelArch::Gemma2,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Prefill
    let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "gemma2-full").unwrap();
    assert_eq!(out.dims(), &[1, 8, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 8, &kv_store, "gemma2-full").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

#[test]
fn qwen2_forward_with_biases() {
    // Qwen2: GQA + contiguous RoPE + QKV biases
    let device = Device::Cpu;
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let head_dim = hidden_dim / n_head;
    let kv_dim = n_kv_head * head_dim;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let max_seq_len = 128;
    let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();
    let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
    let make_rms_norm = |w: &Tensor| {
        let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let layer = LayerWeights {
        attention_wq: make_qmatmul(hidden_dim, hidden_dim),
        attention_wk: make_qmatmul(hidden_dim, kv_dim),
        attention_wv: make_qmatmul(hidden_dim, kv_dim),
        attention_wo: make_qmatmul(hidden_dim, hidden_dim),
        attention_bq: Some(Tensor::randn(0f32, 0.01, (hidden_dim,), &device).unwrap()),
        attention_bk: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
        attention_bv: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(hidden_dim, hidden_dim * 4)),
            ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
            ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head,
        head_dim,
        cos,
        sin,
        neg_inf,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: false,
    };

    let mut model = SplitModel {
        tok_embeddings: None,
        layers: vec![LayerVariant::Dense(layer)],
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: 1,
        total_layers: 3,
        hidden_dim,
        arch: ModelArch::Qwen2,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-1-3"),
        final_logit_softcap: None,
    };

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "qwen2").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn gqa_kv_cache_dimensions() {
    // KV-cache stores with n_kv_head (not n_head)
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let head_dim = hidden_dim / n_head;
    let mut model = make_gqa_test_model(
        1,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let seq_len = 5;
    let input = Tensor::randn(0f32, 1.0, (1, seq_len, hidden_dim), &Device::Cpu).unwrap();
    model.forward(&input, 0, &kv_store, "cache-test").unwrap();

    let model_key = format!(
        "{}-{}-{}",
        model.layer_start, model.layer_end, model.total_layers
    );
    let entry = kv_store.get_or_create(&model_key, "cache-test", 1);
    let k = entry.layers[0].as_ref().unwrap().k().unwrap().unwrap();
    assert_eq!(
        k.dims(),
        &[1, n_kv_head, seq_len, head_dim],
        "KV cache should have n_kv_head={n_kv_head}, not n_head={n_head}"
    );
}

#[test]
fn gqa_multiple_decode_steps() {
    // Multiple decode steps with GQA
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    let input = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    model.forward(&input, 0, &kv_store, "multi-decode").unwrap();

    for step in 0..10 {
        let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model
            .forward(&decode, 4 + step, &kv_store, "multi-decode")
            .unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "Step {step} NaN/Inf");
    }
}

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

// ── MoE / DeepSeek architecture tests ──

#[test]
fn test_deepseek_arch_supported() {
    assert!(ModelArch::DeepSeek2.is_supported());
    assert!(ModelArch::DeepSeek2.use_rope_contiguous());
    assert_eq!(ModelArch::DeepSeek2.default_activation(), Activation::SiLU);
    assert!(!ModelArch::DeepSeek2.use_gemma_norm());
}

#[test]
fn test_moe_topk_selection() {
    let device = Device::Cpu;
    // 8 experts, select top-2
    let scores = Tensor::from_vec(
        vec![0.1f32, 0.5, 0.3, 0.8, 0.2, 0.05, 0.7, 0.4],
        (8,),
        &device,
    )
    .unwrap();

    let (indices, weights) = topk_cpu(&scores, 2).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();

    // Top 2 scores: 0.8 at index 3, 0.7 at index 6
    assert_eq!(idx_vec, vec![3, 6]);
    assert_eq!(w_vec.len(), 2);
    // Weights should be softmax-normalized
    let w_sum: f32 = w_vec.iter().sum();
    assert!(
        (w_sum - 1.0).abs() < 1e-5,
        "Weights should sum to 1.0, got {w_sum}"
    );
    // Weight[0] (score 0.8) should be > weight[1] (score 0.7)
    assert!(w_vec[0] > w_vec[1]);
}

#[test]
fn test_moe_topk_single_expert() {
    let device = Device::Cpu;
    let scores = Tensor::from_vec(vec![0.2f32, 0.8, 0.5], (3,), &device).unwrap();
    let (indices, weights) = topk_cpu(&scores, 1).unwrap();
    let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
    let w_vec: Vec<f32> = weights.to_vec1().unwrap();
    assert_eq!(idx_vec, vec![1]);
    assert!(
        (w_vec[0] - 1.0).abs() < 1e-5,
        "Single expert weight should be 1.0"
    );
}

#[test]
fn test_moe_forward_single_expert() {
    // A 1-expert MoE with top-1 should behave like a dense FFN
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 1;

    // Create expert weights: [1, intermediate, hidden] and [1, hidden, intermediate]
    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    // Router: [1, hidden]
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 1,
    };

    let x = Tensor::randn(0f32, 1.0, (1, 4, hidden), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
}

#[test]
fn test_moe_forward_multi_expert() {
    // 4 experts, select top-2, verify output shape and finiteness
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 4;

    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 2,
    };

    let x = Tensor::randn(0f32, 1.0, (1, 3, hidden), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 3, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Multi-expert output NaN/Inf"
    );
}

#[test]
fn test_shared_expert_integration() {
    // MoE with shared experts: output = routed_experts + shared_expert
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 2;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };

    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

    // MoE without shared experts
    let moe_no_shared = MoeFfn {
        gate: gate.clone(),
        gate_exps: gate_exps.clone(),
        down_exps: down_exps.clone(),
        up_exps: up_exps.clone(),
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used: 1,
    };

    // MoE with shared experts
    let moe_with_shared = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: Some(make_qmatmul(hidden, intermediate)),
        shared_down: Some(make_qmatmul(intermediate, hidden)),
        shared_up: Some(make_qmatmul(hidden, intermediate)),
        n_experts_used: 1,
    };

    let x = Tensor::randn(0f32, 1.0, (1, 2, hidden), &device).unwrap();
    let out_no_shared = moe_no_shared.forward(&x).unwrap();
    let out_with_shared = moe_with_shared.forward(&x).unwrap();

    assert_eq!(out_no_shared.dims(), out_with_shared.dims());
    // Outputs should differ due to shared expert contribution
    let diff = (&out_no_shared - &out_with_shared).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0()
        .unwrap();
    assert!(max_diff > 0.0, "Shared expert should change output");
}

#[test]
fn test_mla_q_decompress() {
    // Verify Q path shapes: x → q_a → norm → q_b → reshape
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32; // per-head
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Test that forward_mla runs without error and returns correct shape
    let x = Tensor::randn(0f32, 0.1, (1, 5, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
    assert_eq!(out.dims(), &[1, 5, hidden], "MLA output shape mismatch");

    // KV cache should be populated
    assert!(kv_cache.is_some(), "KV cache should be created");
}

#[test]
fn test_mla_kv_decompress() {
    // Verify KV path shapes and cache dimensions
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32;
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Prefill with seq_len=3
    let x = Tensor::randn(0f32, 0.1, (1, 3, hidden), &device).unwrap();
    let mut kv_cache = None;
    mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();

    // Check KV cache dimensions
    let cache = kv_cache.as_ref().unwrap();
    let k = cache.k().unwrap().unwrap();
    let v = cache.v().unwrap().unwrap();
    // K: [b, n_head, seq_len, key_length]
    assert_eq!(k.dims(), &[1, n_head, 3, key_length]);
    // V: [b, n_head, seq_len, value_length]
    assert_eq!(v.dims(), &[1, n_head, 3, value_length]);
}

#[test]
fn test_mla_rope_split() {
    // Verify that MLA correctly splits q into nope and rope parts
    let device = Device::Cpu;
    let hidden = 64;
    let q_lora_rank = 16;
    let n_head = 4;
    let key_length = 32;
    let value_length = 16;
    let kv_lora_rank = 8;
    let rope_dim = 8;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let mla = MlaWeights {
        q_a: make_qmatmul(hidden, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos,
        sin,
        neg_inf,
    };

    // Prefill + decode: verify output stays finite
    let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();
    let mut kv_cache = None;
    let out_prefill = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
    let flat: Vec<f32> = out_prefill.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MLA prefill NaN/Inf");

    // Decode step
    let x_decode = Tensor::randn(0f32, 0.1, (1, 1, hidden), &device).unwrap();
    let out_decode = mla
        .forward_mla(&x_decode, None, 4, &mut kv_cache, 128)
        .unwrap();
    assert_eq!(out_decode.dims(), &[1, 1, hidden]);
    let flat: Vec<f32> = out_decode.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MLA decode NaN/Inf");
}

#[test]
fn test_layer_variant_dense_unchanged() {
    // Wrapping in LayerVariant::Dense should produce same output as before
    let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
    let mut model = make_gqa_test_model(
        2,
        hidden_dim,
        n_head,
        n_kv_head,
        false,
        Activation::SiLU,
        None,
        ModelArch::Llama,
    );
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Verify layers are Dense variants
    for layer in &model.layers {
        assert!(matches!(layer, LayerVariant::Dense(_)));
    }

    // Forward pass works identically
    let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "dense-test").unwrap();
    assert_eq!(out.dims(), &[1, 6, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));

    // Decode step
    let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 6, &kv_store, "dense-test").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
}

#[test]
fn test_deepseek_meta_parsing() {
    // Verify DeepSeekMeta struct construction with various values
    let meta = DeepSeekMeta {
        n_experts: 64,
        n_experts_used: 6,
        n_shared_experts: 2,
        kv_lora_rank: 512,
        q_lora_rank: 1536,
        key_length: 192,
        value_length: 128,
        rope_dim: 64,
    };
    assert_eq!(meta.n_experts, 64);
    assert_eq!(meta.n_experts_used, 6);
    assert_eq!(meta.n_shared_experts, 2);
    assert_eq!(meta.kv_lora_rank, 512);
    assert_eq!(meta.q_lora_rank, 1536);
    assert_eq!(meta.key_length, 192);
    assert_eq!(meta.value_length, 128);
    assert_eq!(meta.rope_dim, 64);
}

/// Build a test model with DeepSeek-style mixed layers (1 dense + 1 MLA/MoE)
fn make_deepseek_test_model(hidden_dim: usize) -> SplitModel {
    let device = Device::Cpu;
    let n_head = 4;
    let key_length = hidden_dim / n_head; // per-head key dim
    let value_length = hidden_dim / n_head;
    let kv_lora_rank = 16;
    let q_lora_rank = 16;
    let rope_dim = 8;
    let intermediate = hidden_dim * 2;
    let n_experts = 4;
    let n_experts_used = 2;

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let nope_dim = key_length - rope_dim;
    let max_seq_len = 128;
    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    // Layer 0: Dense (like first few DeepSeek layers)
    let head_dim = hidden_dim / n_head;
    let (dense_cos, dense_sin) =
        precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
    let dense_layer = LayerVariant::Dense(LayerWeights {
        attention_wq: make_qmatmul(hidden_dim, hidden_dim),
        attention_wk: make_qmatmul(hidden_dim, hidden_dim),
        attention_wv: make_qmatmul(hidden_dim, hidden_dim),
        attention_wo: make_qmatmul(hidden_dim, hidden_dim),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(hidden_dim),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(hidden_dim, intermediate)),
            ffn_down: make_qmatmul(intermediate, hidden_dim),
            ffn_up: make_qmatmul(hidden_dim, intermediate),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(hidden_dim),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos: dense_cos,
        sin: dense_sin,
        neg_inf: neg_inf.clone(),
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: false,
    });

    // Layer 1: DeepSeek MLA + MoE
    let mla = MlaWeights {
        q_a: make_qmatmul(hidden_dim, q_lora_rank),
        q_a_norm: make_rms_norm(q_lora_rank),
        q_b: make_qmatmul(q_lora_rank, n_head * key_length),
        kv_a: make_qmatmul(hidden_dim, kv_lora_rank + rope_dim),
        kv_a_norm: make_rms_norm(kv_lora_rank),
        kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
        output: make_qmatmul(n_head * value_length, hidden_dim),
        n_head,
        key_length,
        value_length,
        kv_lora_rank,
        rope_dim,
        cos: cos.clone(),
        sin: sin.clone(),
        neg_inf: neg_inf.clone(),
    };

    let moe = MoeFfn {
        gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
        gate_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
            .unwrap(),
        down_exps: Tensor::randn(0f32, 0.02, (n_experts, hidden_dim, intermediate), &device)
            .unwrap(),
        up_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device).unwrap(),
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used,
    };

    let deepseek_layer = LayerVariant::DeepSeek {
        attention: mla,
        ffn: FfnVariant::MoE(moe),
        attention_norm: make_rms_norm(hidden_dim),
        ffn_norm: make_rms_norm(hidden_dim),
    };

    SplitModel {
        tok_embeddings: None,
        layers: vec![dense_layer, deepseek_layer],
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: 2,
        total_layers: 4,
        hidden_dim,
        arch: ModelArch::DeepSeek2,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-2-4"),
        final_logit_softcap: None,
    }
}

#[test]
fn test_deepseek_mixed_layers_forward() {
    // Full forward pass through mixed dense + MLA/MoE layers
    let hidden_dim = 64;
    let mut model = make_deepseek_test_model(hidden_dim);
    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

    // Prefill
    let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&input, 0, &kv_store, "ds-test").unwrap();
    assert_eq!(out.dims(), &[1, 4, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "DeepSeek prefill NaN/Inf"
    );

    // Decode
    let decode = Tensor::randn(0f32, 0.1, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let out = model.forward(&decode, 4, &kv_store, "ds-test").unwrap();
    assert_eq!(out.dims(), &[1, 1, hidden_dim]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "DeepSeek decode NaN/Inf"
    );
}

#[test]
fn test_glm4_arch_supported() {
    assert!(ModelArch::Glm4.is_supported());
    assert!(ModelArch::Glm4.use_rope_contiguous());
    assert_eq!(ModelArch::Glm4.default_activation(), Activation::SiLU);
    assert!(!ModelArch::Glm4.use_gemma_norm());
    assert_eq!(ModelArch::from_gguf_arch("glm4"), ModelArch::Glm4);
}

#[test]
fn test_llama4_arch_supported() {
    assert!(ModelArch::Llama4.is_supported());
    assert!(ModelArch::Llama4.use_rope_contiguous());
    assert_eq!(ModelArch::Llama4.default_activation(), Activation::SiLU);
    assert!(!ModelArch::Llama4.use_gemma_norm());
    assert_eq!(ModelArch::from_gguf_arch("llama4"), ModelArch::Llama4);
}

#[test]
fn test_partial_rope_glm4_style() {
    // GLM-4 uses partial RoPE: only first half of head_dim gets rotated
    let device = Device::Cpu;
    let head_dim = 16;
    let rope_dim = 8; // half of head_dim
    let n_head = 2;
    let seq_len = 4;
    let max_seq_len = 32;

    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
    let make_rms_norm = |w: &Tensor| {
        let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let lw = LayerWeights {
        attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(n_head * head_dim, n_head * head_dim * 4)),
            ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim),
            ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos,
        sin,
        neg_inf,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim,
        skip_rope: false,
    };

    // Test that apply_rotary_emb handles partial RoPE
    let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
    let result = lw.apply_rotary_emb(&x, 0).unwrap();
    assert_eq!(result.dims(), &[1, n_head, seq_len, head_dim]);

    // Verify first rope_dim dims are different (rotated) and last dims unchanged
    let x_pass = x.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
    let r_pass = result.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
    let diff: f32 = (&x_pass - &r_pass)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        diff < 1e-5,
        "Non-rotated dims should be unchanged, diff={diff}"
    );
}

#[test]
fn test_nope_skip_rope() {
    // Llama 4 iRoPE: every 4th layer skips RoPE entirely
    let device = Device::Cpu;
    let head_dim = 16;
    let n_head = 2;
    let seq_len = 4;

    let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, 32, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
    let make_rms_norm = |w: &Tensor| {
        let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let lw = LayerWeights {
        attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim),
        attention_bq: None,
        attention_bk: None,
        attention_bv: None,
        attention_norm: make_rms_norm(&norm_w),
        attn_q_norm: None,
        attn_k_norm: None,
        ffn: FfnVariant::Dense(Mlp {
            ffn_gate: Some(make_qmatmul(n_head * head_dim, n_head * head_dim * 4)),
            ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim),
            ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
            activation: Activation::SiLU,
        }),
        ffn_norm: make_rms_norm(&norm_w),
        post_attention_norm: None,
        post_ffw_norm: None,
        n_head,
        n_kv_head: n_head,
        head_dim,
        cos,
        sin,
        neg_inf,
        use_rope_contiguous: true,
        attn_logit_softcap: None,
        rope_dim: head_dim,
        skip_rope: true, // NoPE layer
    };

    let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
    let result = lw.apply_rotary_emb(&x, 0).unwrap();

    // skip_rope=true means output == input (no rotation applied)
    let diff: f32 = (&x - &result)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        diff < 1e-6,
        "NoPE layer should not modify input, diff={diff}"
    );
}

#[test]
fn test_ffn_variant_moe_forward() {
    // Test that FfnVariant::MoE dispatches through MoeFfn correctly
    let device = Device::Cpu;
    let hidden = 32;
    let intermediate = 64;
    let n_experts = 4;
    let n_experts_used = 2;

    // Build small MoE FFN
    let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();
    let gate_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
    let down_exps = Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
    let up_exps = Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();

    let moe = MoeFfn {
        gate,
        gate_exps,
        down_exps,
        up_exps,
        shared_gate: None,
        shared_down: None,
        shared_up: None,
        n_experts_used,
    };

    let ffn = FfnVariant::MoE(moe);
    let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();

    let out = match &ffn {
        FfnVariant::Dense(mlp) => mlp.forward(&x, None).unwrap(),
        FfnVariant::MoE(moe) => moe.forward(&x).unwrap(),
    };
    assert_eq!(out.dims(), &[1, 4, hidden]);
    let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
}

#[test]
fn test_irope_nope_pattern() {
    // Verify the iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE
    let nope_layers: Vec<bool> = (0..12).map(|i| i % 4 == 3).collect();
    assert_eq!(
        nope_layers,
        vec![false, false, false, true, false, false, false, true, false, false, false, true]
    );
}

#[test]
fn test_llama4_moe_layer_forward() {
    // End-to-end test: model with MoE FFN layers via FfnVariant
    let device = Device::Cpu;
    let hidden_dim = 32;
    let n_head = 4;
    let head_dim = hidden_dim / n_head;
    let n_kv_head = 2;
    let n_experts = 4;
    let n_experts_used = 2;
    let intermediate = 64;
    let max_seq_len = 32;
    let rope_dim = head_dim;

    let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
    let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

    let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
        let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        QMatMul::from_qtensor(qt).expect("QMatMul load failed")
    };
    let make_rms_norm = |dim: usize| -> RmsNorm {
        let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
        RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
    };

    let kv_dim = n_kv_head * head_dim;

    // Build a mix of dense + MoE layers (like Llama 4)
    let mut layers = Vec::new();
    for layer_idx in 0..4 {
        let is_nope = layer_idx % 4 == 3;

        let ffn = if layer_idx % 2 == 1 {
            // MoE layers on odd indices
            FfnVariant::MoE(MoeFfn {
                gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
                gate_exps: Tensor::randn(
                    0f32,
                    0.02,
                    (n_experts, intermediate, hidden_dim),
                    &device,
                )
                .unwrap(),
                down_exps: Tensor::randn(
                    0f32,
                    0.02,
                    (n_experts, hidden_dim, intermediate),
                    &device,
                )
                .unwrap(),
                up_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
                    .unwrap(),
                shared_gate: None,
                shared_down: None,
                shared_up: None,
                n_experts_used,
            })
        } else {
            // Dense FFN on even indices
            FfnVariant::Dense(Mlp {
                ffn_gate: Some(make_qmatmul(hidden_dim, intermediate)),
                ffn_down: make_qmatmul(intermediate, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, intermediate),
                activation: Activation::SiLU,
            })
        };

        layers.push(LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, kv_dim),
            attention_wv: make_qmatmul(hidden_dim, kv_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(hidden_dim),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn,
            ffn_norm: make_rms_norm(hidden_dim),
            post_attention_norm: None,
            post_ffw_norm: None,
            n_head,
            n_kv_head,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim,
            skip_rope: is_nope,
        }));
    }

    let mut model = SplitModel {
        tok_embeddings: None,
        layers,
        norm: None,
        output: None,
        masks: None,
        layer_start: 0,
        layer_end: 4,
        total_layers: 8,
        hidden_dim,
        arch: ModelArch::Llama4,
        device,
        vocabulary: None,
        tokenizer: None,
        eos_tokens: vec![2],
        chat_template: None,
        bos_token: String::new(),
        eos_token: String::new(),
        max_seq_len,
        kv_model_key: String::from("0-4-8"),
        final_logit_softcap: None,
    };

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
    let output = model.forward(&input, 0, &kv_store, "llama4-test").unwrap();
    assert_eq!(output.dims(), &[1, 4, hidden_dim]);
    let flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "Llama 4 MoE output NaN/Inf"
    );
}

/// Test Gemma-2 with real GGUF file — compare load_from_gguf vs load_from_shards.
/// Requires the Gemma-2-2B-IT Q4_K_M model to be present.
#[test]
#[ignore] // Run with: cargo test gemma2_real_gguf -- --ignored --nocapture
fn gemma2_real_gguf_vs_shards() {
    use candle_core::Tensor;
    let gguf_path = std::path::Path::new(
        "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
    );
    if !gguf_path.exists() {
        eprintln!("Skipping: GGUF not found at {}", gguf_path.display());
        return;
    }

    // Load from full GGUF
    let mut model =
        SplitModel::load_from_gguf(gguf_path, 0, 26, true, true).expect("Failed to load from GGUF");

    // Use the tokenizer to get the same tokens our API uses
    let prompt_tokens: Vec<u32> = vec![
        2, 2, 106, 1645, 108, 1841, 603, 573, 6037, 576, 6081, 235336, 107, 108, 106, 2516, 108,
    ];
    let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0) // [1, 17]
        .unwrap()
        .to_dtype(candle_core::DType::I64)
        .unwrap();

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let logits = model
        .forward(&input, 0, &kv_store, "gemma2-gguf-test")
        .expect("Forward pass failed");

    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let (argmax_idx, argmax_val) = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    eprintln!("GGUF logits: argmax={} score={:.4}", argmax_idx, argmax_val);
    eprintln!(
        "  min={:.4} max={:.4} dim={}",
        flat.iter().cloned().fold(f32::INFINITY, f32::min),
        flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        flat.len()
    );

    // Expected: token 651 ("The") should be near the top
    let the_score = flat[651];
    eprintln!("  token 651 ('The') score={:.4}", the_score);
    eprintln!("  token 235274 ('1') score={:.4}", flat[235274]);

    // Save logits for external comparison
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("/tmp/gemma2_our_logits.bin", &bytes).ok();
    eprintln!(
        "  Saved {} logits to /tmp/gemma2_our_logits.bin",
        flat.len()
    );
}

/// Test Gemma-2 with single token (no mask) to eliminate mask issues.
#[test]
#[ignore]
fn gemma2_single_token() {
    use candle_core::{Device, Tensor};
    let gguf_path = std::path::Path::new(
        "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
    );
    if !gguf_path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    // Single BOS token — no mask needed, no flash attention
    let prompt_tokens: Vec<u32> = vec![2];
    let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .to_dtype(candle_core::DType::I64)
        .unwrap();

    let mut model =
        SplitModel::load_from_gguf(gguf_path, 0, 26, true, true).expect("Failed to load");

    let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let logits = model
        .forward(&input, 0, &kv_store, "gemma2-single-tok")
        .expect("Forward failed");

    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let (argmax_idx, argmax_val) = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("Single token (BOS): argmax={argmax_idx} score={argmax_val:.4}");
    eprintln!("  token 2 score: {:.4}", flat[2]);
    eprintln!("  token 108 score: {:.4}", flat[108]);

    // Save for comparison
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("/tmp/gemma2_single_token_logits.bin", &bytes).ok();
    eprintln!("  Saved logits to /tmp/gemma2_single_token_logits.bin");
}

/// Test embedding dequantization matches Python reference.
#[test]
#[ignore]
fn gemma2_embedding_verification() {
    use candle_core::{quantized::gguf_file, Device, Tensor};

    let gguf_path = "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
    let path = std::path::Path::new(gguf_path);
    if !path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut cursor).unwrap();
    let device = Device::Cpu;

    // Load and dequantize embedding
    let embd_qt = ct
        .tensor(&mut cursor, "token_embd.weight", &device)
        .unwrap();
    let embd = embd_qt.dequantize(&device).unwrap();
    eprintln!("Embedding shape: {:?}", embd.shape());

    // Get row 2 (BOS token)
    let row2 = embd.i(2).unwrap();
    let row2_vals: Vec<f32> = row2.to_vec1().unwrap();
    eprintln!("Row 2 (BOS) first 8: {:?}", &row2_vals[..8]);
    eprintln!(
        "Row 2 (BOS) last 8: {:?}",
        &row2_vals[row2_vals.len() - 8..]
    );

    // Compare with Python reference
    let py_ref = std::fs::read("/tmp/gemma2_embed_row2.npy").ok();
    if let Some(ref npy_bytes) = py_ref {
        // Parse npy format: skip header, read f32 values
        // Simple npy parser: header starts with \x93NUMPY, has length info
        let header_len = 10 + npy_bytes[8] as usize + ((npy_bytes[9] as usize) << 8);
        let data_bytes = &npy_bytes[header_len..];
        let py_vals: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("Python ref first 8: {:?}", &py_vals[..8]);

        let mut max_diff = 0f32;
        let mut mismatches = 0;
        for (i, (a, b)) in row2_vals.iter().zip(py_vals.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-4 && i < 20 {
                eprintln!("  MISMATCH [{i}] rust={a:.6} python={b:.6} diff={diff:.6}");
                mismatches += 1;
            }
        }
        eprintln!("Max embedding diff: {max_diff:.6}, mismatches (>1e-4): {mismatches}");
    } else {
        eprintln!("No Python reference found at /tmp/gemma2_embed_row2.npy");
    }

    // Now test: embedding lookup → scale by sqrt(2304) → final norm → output projection
    // This is the 0-layer forward pass
    let emb = Embedding::new(embd.clone(), 2304);

    // Token 2 (BOS) lookup
    let ids = Tensor::new(&[2u32], &device).unwrap();
    let looked_up = emb.forward(&ids).unwrap(); // (1, 2304)
    let scaled = looked_up.affine((2304f64).sqrt(), 0.0).unwrap(); // scale by sqrt(hidden_dim)

    // Apply final norm (with +1 offset)
    let norm_qt = ct
        .tensor(&mut cursor, "output_norm.weight", &device)
        .unwrap();
    let norm_w = norm_qt.dequantize(&device).unwrap();
    let norm_w_plus1 = (norm_w + 1.0).unwrap(); // Gemma +1
    let normed = candle_nn::ops::rms_norm(&scaled, &norm_w_plus1, 1e-6).unwrap();

    // Output projection using dequantized embedding
    let logits = normed.matmul(&embd.t().unwrap()).unwrap();
    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    let argmax = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    eprintln!("0-layer logits: argmax={} score={:.4}", argmax.0, argmax.1);
    eprintln!("  token 2 score: {:.4}", flat[2]);
    eprintln!("  token 108 score: {:.4}", flat[108]);
}

/// Test QMatMul vs dequantized matmul for output projection.
/// Diagnoses whether the sorted-correlation issue is in the output projection.
#[test]
#[ignore]
fn gemma2_output_projection_qmatmul_vs_deq() {
    use candle_core::{quantized::gguf_file, Device, Tensor};

    let gguf_path = "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
    let path = std::path::Path::new(gguf_path);
    if !path.exists() {
        eprintln!("Skipping: GGUF not found");
        return;
    }

    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut cursor).unwrap();
    let device = Device::Cpu;

    // Load token_embd.weight as both QTensor and dequantized
    let embd_qt = ct
        .tensor(&mut cursor, "token_embd.weight", &device)
        .unwrap();
    eprintln!("token_embd.weight QTensor shape: {:?}", embd_qt.shape());

    let embd_deq = embd_qt.dequantize(&device).unwrap();
    eprintln!("Dequantized embedding shape: {:?}", embd_deq.shape());

    // Create QMatMul from the QTensor
    let qmm = QMatMul::from_qtensor(
        ct.tensor(&mut cursor, "token_embd.weight", &device)
            .unwrap(),
    )
    .unwrap();

    // Create a random hidden state (simulating post-norm output)
    let hidden = Tensor::randn(0f32, 1.0, (1, 2304), &device).unwrap();

    // Method 1: QMatMul (our current approach)
    let logits_qmm = qmm.forward(&hidden).unwrap();
    let flat_qmm: Vec<f32> = logits_qmm.flatten_all().unwrap().to_vec1().unwrap();

    // Method 2: Dequantized matmul (reference approach)
    // embd_deq shape is (256000, 2304), we need hidden @ embd_deq.T
    let logits_deq = hidden.matmul(&embd_deq.t().unwrap()).unwrap();
    let flat_deq: Vec<f32> = logits_deq.flatten_all().unwrap().to_vec1().unwrap();

    eprintln!(
        "QMatMul logits: argmax={}",
        flat_qmm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );
    eprintln!(
        "Deq logits: argmax={}",
        flat_deq
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    );

    // Check if they agree
    let mut max_diff = 0f32;
    let mut sum_diff_sq = 0f64;
    for (i, (a, b)) in flat_qmm.iter().zip(flat_deq.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff_sq += (diff as f64) * (diff as f64);
        if i < 5 {
            eprintln!("  [{i}] qmm={a:.6} deq={b:.6} diff={diff:.6}");
        }
    }
    let rmse = (sum_diff_sq / flat_qmm.len() as f64).sqrt();
    eprintln!("Max diff: {max_diff:.6}, RMSE: {rmse:.6}");

    // Check correlation
    let mean_q: f64 = flat_qmm.iter().map(|v| *v as f64).sum::<f64>() / flat_qmm.len() as f64;
    let mean_d: f64 = flat_deq.iter().map(|v| *v as f64).sum::<f64>() / flat_deq.len() as f64;
    let mut cov = 0f64;
    let mut var_q = 0f64;
    let mut var_d = 0f64;
    for (q, d) in flat_qmm.iter().zip(flat_deq.iter()) {
        let dq = *q as f64 - mean_q;
        let dd = *d as f64 - mean_d;
        cov += dq * dd;
        var_q += dq * dq;
        var_d += dd * dd;
    }
    let corr = cov / (var_q.sqrt() * var_d.sqrt());
    eprintln!("Pearson correlation (QMatMul vs Deq): {corr:.6}");

    // They should be highly correlated (>0.99) — just quantization error
    assert!(
        corr > 0.99,
        "QMatMul and dequantized matmul should agree: corr={corr}"
    );
}

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
