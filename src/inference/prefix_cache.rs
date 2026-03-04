//! Cross-request prefix cache for sharing KV state across requests with
//! identical system prompts.
//!
//! When multiple requests share the same system prompt (e.g., "You are a helpful
//! assistant..."), the KV-cache state for processing those tokens is computed once
//! and reused across requests. This saves 50-80% of prefill computation for
//! applications with long, repeated system prompts.
//!
//! Entries are keyed by BLAKE3 hash of the prefix token IDs combined with the
//! model segment key (layer range). LRU eviction keeps memory bounded.

use std::collections::HashMap;
use std::time::Instant;

use candle_core::Tensor;

/// Cross-request prefix cache for sharing KV state.
///
/// Stores per-layer (K, V) tensor pairs after processing a common token prefix.
/// Subsequent requests with the same prefix restore the cached KV state instead
/// of recomputing it, then only process the new (suffix) tokens.
pub struct PrefixCache {
    entries: HashMap<PrefixCacheKey, PrefixCacheEntry>,
    max_entries: usize,
}

/// Cache key: prefix content hash + model segment identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefixCacheKey {
    /// BLAKE3 hash of the prefix token IDs (8 bytes per token, LE).
    prefix_hash: [u8; 32],
    /// Model segment key: "layer_start-layer_end-total_layers".
    model_key: String,
}

/// Cached KV state for a prefix token sequence.
struct PrefixCacheEntry {
    /// Per-layer (K, V) tensor pairs after processing the prefix.
    /// K shape: (batch=1, n_kv_heads, prefix_len, head_dim)
    layer_kv: Vec<(Tensor, Tensor)>,
    /// Number of prefix tokens.
    prefix_len: usize,
    /// Last access time for LRU eviction.
    last_accessed: Instant,
}

impl PrefixCache {
    /// Create a new prefix cache with the given max entry count.
    /// Set `max_entries` to 0 to disable caching.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    /// Look up cached KV state for a prefix hash and model segment.
    ///
    /// Returns the per-layer (K, V) tensors and the prefix token count on hit.
    /// Updates the last-accessed time for LRU tracking.
    pub fn get(
        &mut self,
        prefix_hash: &[u8; 32],
        model_key: &str,
    ) -> Option<(&[(Tensor, Tensor)], usize)> {
        let key = PrefixCacheKey {
            prefix_hash: *prefix_hash,
            model_key: model_key.to_string(),
        };
        let cache_entries = self.entries.len();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed = Instant::now();
            tracing::debug!(cache_entries, hit = true, "DIAG: prefix_cache lookup");
            Some((&entry.layer_kv, entry.prefix_len))
        } else {
            tracing::debug!(cache_entries, hit = false, "DIAG: prefix_cache lookup");
            None
        }
    }

    /// Insert a new prefix cache entry, evicting the LRU entry if at capacity.
    pub fn insert(
        &mut self,
        prefix_hash: [u8; 32],
        model_key: String,
        layer_kv: Vec<(Tensor, Tensor)>,
        prefix_len: usize,
    ) {
        if self.max_entries == 0 {
            return;
        }

        // Evict LRU if at capacity
        while self.entries.len() >= self.max_entries {
            let oldest_key = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                self.entries.remove(&key);
                tracing::debug!("Prefix cache: evicted LRU entry");
            } else {
                break;
            }
        }

        self.entries.insert(
            PrefixCacheKey {
                prefix_hash,
                model_key,
            },
            PrefixCacheEntry {
                layer_kv,
                prefix_len,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

/// Compute the BLAKE3 hash of a token ID sequence for use as a prefix cache key.
pub fn hash_token_ids(tokens: &[i64]) -> [u8; 32] {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    *blake3::hash(&bytes).as_bytes()
}

/// Build a system-prompt-only prefix string using ChatML format (without the
/// trailing assistant tag) for prefix matching against the full prompt.
///
/// Returns `None` if there are no system messages.
pub fn build_system_prefix(messages: &[crate::types::ChatMessage]) -> Option<String> {
    let has_system = messages
        .iter()
        .any(|m| matches!(m.role, crate::types::Role::System));
    if !has_system {
        return None;
    }

    let mut prefix = String::new();
    for msg in messages {
        if matches!(msg.role, crate::types::Role::System) {
            prefix.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
        }
    }

    if prefix.is_empty() {
        None
    } else {
        Some(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_kv_pair(seq_len: usize) -> (Tensor, Tensor) {
        let k = Tensor::zeros((1, 4, seq_len, 64), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 4, seq_len, 64), DType::F32, &Device::Cpu).unwrap();
        (k, v)
    }

    #[test]
    fn cache_miss_returns_none() {
        let mut cache = PrefixCache::new(10);
        let hash = [0u8; 32];
        assert!(cache.get(&hash, "0-16-32").is_none());
    }

    #[test]
    fn cache_hit_after_insert() {
        let mut cache = PrefixCache::new(10);
        let hash = hash_token_ids(&[1, 2, 3, 4]);
        let kv = vec![make_kv_pair(4), make_kv_pair(4)];

        cache.insert(hash, "0-16-32".to_string(), kv, 4);

        let result = cache.get(&hash, "0-16-32");
        assert!(result.is_some());
        let (layer_kv, prefix_len) = result.unwrap();
        assert_eq!(layer_kv.len(), 2);
        assert_eq!(prefix_len, 4);
    }

    #[test]
    fn different_prompts_produce_different_keys() {
        let hash_a = hash_token_ids(&[1, 2, 3]);
        let hash_b = hash_token_ids(&[4, 5, 6]);
        assert_ne!(hash_a, hash_b);

        let mut cache = PrefixCache::new(10);
        cache.insert(hash_a, "0-16-32".to_string(), vec![make_kv_pair(3)], 3);
        cache.insert(hash_b, "0-16-32".to_string(), vec![make_kv_pair(3)], 3);

        assert!(cache.get(&hash_a, "0-16-32").is_some());
        assert!(cache.get(&hash_b, "0-16-32").is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn different_model_keys_are_separate() {
        let mut cache = PrefixCache::new(10);
        let hash = hash_token_ids(&[1, 2, 3]);

        cache.insert(hash, "0-16-32".to_string(), vec![make_kv_pair(3)], 3);

        assert!(cache.get(&hash, "0-16-32").is_some());
        assert!(cache.get(&hash, "16-32-32").is_none());
    }

    #[test]
    fn lru_eviction_removes_oldest() {
        let mut cache = PrefixCache::new(2);

        let hash_a = hash_token_ids(&[1]);
        let hash_b = hash_token_ids(&[2]);
        let hash_c = hash_token_ids(&[3]);

        cache.insert(hash_a, "m".to_string(), vec![make_kv_pair(1)], 1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.insert(hash_b, "m".to_string(), vec![make_kv_pair(1)], 1);
        std::thread::sleep(std::time::Duration::from_millis(10));

        // This should evict hash_a (oldest)
        cache.insert(hash_c, "m".to_string(), vec![make_kv_pair(1)], 1);

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&hash_a, "m").is_none()); // evicted
        assert!(cache.get(&hash_b, "m").is_some());
        assert!(cache.get(&hash_c, "m").is_some());
    }

    #[test]
    fn lru_eviction_refreshes_on_access() {
        let mut cache = PrefixCache::new(2);

        let hash_a = hash_token_ids(&[1]);
        let hash_b = hash_token_ids(&[2]);
        let hash_c = hash_token_ids(&[3]);

        cache.insert(hash_a, "m".to_string(), vec![make_kv_pair(1)], 1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.insert(hash_b, "m".to_string(), vec![make_kv_pair(1)], 1);
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Access hash_a to refresh its timestamp
        let _ = cache.get(&hash_a, "m");
        std::thread::sleep(std::time::Duration::from_millis(10));

        // This should evict hash_b (now oldest)
        cache.insert(hash_c, "m".to_string(), vec![make_kv_pair(1)], 1);

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&hash_a, "m").is_some()); // refreshed, not evicted
        assert!(cache.get(&hash_b, "m").is_none()); // evicted
        assert!(cache.get(&hash_c, "m").is_some());
    }

    #[test]
    fn zero_capacity_disables_caching() {
        let mut cache = PrefixCache::new(0);
        let hash = hash_token_ids(&[1, 2, 3]);
        cache.insert(hash, "m".to_string(), vec![make_kv_pair(3)], 3);
        assert_eq!(cache.len(), 0);
        assert!(cache.get(&hash, "m").is_none());
    }

    #[test]
    fn build_system_prefix_extracts_system_messages() {
        use crate::types::{ChatMessage, Role};

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are helpful".to_string(),
                images: vec![],
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".to_string(),
                images: vec![],
            },
        ];

        let prefix = build_system_prefix(&messages).unwrap();
        assert_eq!(prefix, "<|im_start|>system\nYou are helpful<|im_end|>\n");
    }

    #[test]
    fn build_system_prefix_none_without_system() {
        use crate::types::{ChatMessage, Role};

        let messages = vec![ChatMessage {
            role: Role::User,
            content: "Hello".to_string(),
            images: vec![],
        }];

        assert!(build_system_prefix(&messages).is_none());
    }

    #[test]
    fn hash_deterministic() {
        let tokens = vec![100i64, 200, 300, 400];
        let h1 = hash_token_ids(&tokens);
        let h2 = hash_token_ids(&tokens);
        assert_eq!(h1, h2);
    }
}
